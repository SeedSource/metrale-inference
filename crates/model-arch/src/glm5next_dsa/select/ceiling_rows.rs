// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: The scratch check of a [`DsaSelectLaunch::CeilingRows`] pass: one ceiling
//! selection over the `q_rows` verify rows of ONE sequence, which share one `geom_dev`
//! written from the last row's `seq_len`.
//!
//! Why one geometry is enough (kernels/gb10/common/dsa_indexer.cu): every kernel of the chain
//! reads its row stride from that geometry (`P` for `out`/`valid_cand` in the scorers and the
//! radix passes, `select_k` for `selected` in the sort and the expand), so the strides agree
//! inside one launch. A pool is a candidate for row `r` only when it ends at or before
//! `q_pos[r]` (`dsa_index_scores*`), and `dsa_expand_selection` clamps each row's select_k to
//! `(q_pos[r] + 1) / KP`, so each row should get the bytes of a single-row pass at its own
//! length. That is measured, not assumed: `examples/dsa_ceiling_rows_microtest.rs`.
//!
//! The scores region holds `q_rows` rows of the live pool count, which under a ceiling launch
//! is known only on the device, so the scratch must fit `q_rows` rows at the ceiling.
//!
//! 2026-10-12: Also the lever [`dsa_ceiling_rows`] (`METRALE_GLM_DSA_CEILING_ROWS`), which
//! lets the replay-safe verify (`layer/ceiling_verify.rs`) run one such pass per sequence
//! instead of one single-row ceiling pass per verify row.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - 2026-10-12: The only production construction of `CeilingRows` is in
//!   `layer/ceiling_verify.rs`, behind [`dsa_ceiling_rows`] (source guard in the tests).

use std::sync::OnceLock;

use anyhow::{Result, bail};

use super::{DsaSelectGeometry, DsaSelectInputs, DsaSelectScratch};
use crate::glm5next_dsa::Glm5NextDsaConfig;

/// 2026-10-12: Fails unless `scratch` holds `q_rows` rows of a pass at the ceiling of
/// `max_pools` pools (`q_rows` at least 1).
pub(super) fn check_fit(
    cfg: &Glm5NextDsaConfig,
    scratch: &DsaSelectScratch,
    max_pools: usize,
    q_rows: usize,
    inputs: &DsaSelectInputs,
) -> Result<()> {
    if q_rows == 0 {
        bail!("DSA select: a CeilingRows launch needs at least one row");
    }
    let at_ceiling = DsaSelectGeometry::plan(cfg, max_pools * cfg.index_kpool, q_rows)?;
    scratch.fits_with(cfg, &at_ceiling, inputs.pool_cache.is_some())
}

/// 2026-10-12: `METRALE_GLM_DSA_CEILING_ROWS=1` (default off), read once: a graph-captured
/// verify of `k >= 2` rows of one sequence with the pool cache on selects all `k` rows in one
/// `CeilingRows` pass (one `dsa_write_geom_pk` from the last row's `seq_len`) instead of `k`
/// single-row ceiling passes. Byte-identical by construction; GPU gate
/// `examples/dsa_ceiling_rows_microtest.rs`. A value other than unset, empty, `0` or `1` is
/// off, with one warning.
pub(crate) fn dsa_ceiling_rows() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_CEILING_ROWS").ok();
        let on = raw.as_deref().map(str::trim) == Some("1");
        let odd = |r: &&str| !r.is_empty() && r.trim() != "0" && r.trim() != "1";
        if let Some(r) = raw.as_deref().filter(odd) {
            tracing::warn!("METRALE_GLM_DSA_CEILING_ROWS={r} is not 0 or 1 - treated as off");
        }
        if on {
            tracing::warn!(
                "METRALE_GLM_DSA_CEILING_ROWS=1 - a graph-captured DSA verify of one sequence's \
                 rows selects them in one CeilingRows pass (byte-identical by construction)"
            );
        }
        on
    })
}

#[cfg(test)]
#[path = "ceiling_rows_tests.rs"]
mod tests;
