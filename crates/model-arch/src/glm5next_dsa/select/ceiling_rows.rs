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
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - 2026-10-12: No production caller constructs `CeilingRows` (source guard below).

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

#[cfg(test)]
#[path = "ceiling_rows_tests.rs"]
mod tests;
