// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: `METRALE_GLM_DSA_SCORES_TC2=1` (default off): the tensor-core DSA scores of
//! `dsa_index_scores_tc` in mode 1 (`METRALE_GLM_DSA_SCORES_TC=bf16`) from
//! `dsa_index_scores_tc2` (`kernels/gb10/common/dsa_indexer.cu`, section 2d), which writes the
//! same bytes (`out` bit patterns and `valid_cand`) on a faster dataflow: 32 rows x 128 pools
//! per 256-thread block, keys as BF16 B fragments in registers for every head, q staged once
//! per head per block as BF16 in shared memory (double-buffered) and read with ldmatrix.
//!
//! Why: 2c measured about 13 TF/s at 8K+ pools per 128-row call (q re-read from global in FP32
//! per head and K step, 142 registers, 25% occupancy). Target (not yet measured): 60 TF/s.
//!
//! Dispatch: [`scores_tc2_for`]. It runs only where `dsa_index_scores_tc` would run
//! (`scores_tc_for`), in mode 1 only, with its entry point resolved and at most
//! [`SCORES_TC2_MAX_H`] heads; otherwise the existing path runs unchanged. GPU gate:
//! `examples/dsa_indexer_tc2_bitparity_microtest.rs` (bitwise against 2c mode 1, and timing).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - The `SCORES_TC2_*` tile constants equal the `DSA_TC2_*` defines in `dsa_indexer.cu`, and
//!   the kernel takes `dsa_index_scores_tc`'s argument list (checked by a test that reads the
//!   file), so the launcher passes the same arguments, including the mode.
//! - Lever off: `select_tokens` launches exactly what it did before.

use std::sync::OnceLock;

use super::DsaSelectGeometry;
use crate::glm5next_dsa::Glm5NextDsaKernels;

/// 2026-10-07: Query rows per `dsa_index_scores_tc2` block (`DSA_TC2_ROWS`: two 16-row m-tiles).
pub const SCORES_TC2_ROWS: usize = 32;
/// 2026-10-07: Pools per block (`DSA_TC2_POOLS`: 8 warps x two 8-pool n-tiles).
pub const SCORES_TC2_POOLS: usize = 128;
/// 2026-10-07: Threads per block (`DSA_TC2_THREADS`).
pub const SCORES_TC2_BLOCK: u32 = 256;
/// 2026-10-07: Most index heads: the block keeps `[32][H + 1]` f32 weights in shared memory,
/// so this bounds that region (8,320 B at 64).
pub const SCORES_TC2_MAX_H: usize = 64;
/// 2026-10-07: The only `dsa_index_scores_tc` precision mode the kernel reproduces (`bf16`);
/// it traps on any other.
pub const SCORES_TC2_MODE: u32 = 1;

/// 2026-10-07: Logged once, on the first launch that takes `dsa_index_scores_tc2`.
pub const SCORES_TC2_ENGAGED_LINE: &str = "METRALE_GLM_DSA_SCORES_TC2=1: ENGAGED";

/// 2026-10-07: Grid for `q_rows` rows and `n_pools` pools: 128-pool tiles on x, 32-row tiles
/// on y.
pub fn scores_tc2_grid(q_rows: usize, n_pools: usize) -> [u32; 3] {
    [
        n_pools.div_ceil(SCORES_TC2_POOLS) as u32,
        q_rows.div_ceil(SCORES_TC2_ROWS) as u32,
        1,
    ]
}

/// 2026-10-07: Dynamic shared memory of one block at head dim `d` and `heads` heads: two
/// `[32][d + 8]` BF16 q buffers, then `[32][heads + 1]` f32 weights (21,632 B at 128 / 32).
pub fn scores_tc2_smem(d: usize, heads: usize) -> usize {
    2 * SCORES_TC2_ROWS * (d + 8) * 2 + SCORES_TC2_ROWS * (heads + 1) * 4
}

/// 2026-10-07: `1` (surrounding blanks ignored) is on; unset, `0` and anything else are off.
pub fn parse_scores_tc2(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-07: `METRALE_GLM_DSA_SCORES_TC2`, read once; a value other than unset, empty, `0`
/// or `1` is off, with one warning.
pub fn dsa_scores_tc2() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_SCORES_TC2").ok();
        let odd = |r: &&str| !r.is_empty() && r.trim() != "0" && r.trim() != "1";
        if let Some(r) = raw.as_deref().filter(odd) {
            tracing::warn!("METRALE_GLM_DSA_SCORES_TC2={r} is not 0 or 1 - treated as off");
        }
        let on = parse_scores_tc2(raw.as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_DSA_SCORES_TC2=1 - where METRALE_GLM_DSA_SCORES_TC=bf16 scores on \
                 tensor cores, dsa_index_scores_tc2 runs (the same bytes as dsa_index_scores_tc \
                 mode 1)"
            );
        }
        on
    })
}

/// 2026-10-07: Whether a scores launch takes `dsa_index_scores_tc2`: the lever on, the entry
/// point resolved, `dsa_index_scores_tc` selected (`tc`, from `scores_tc_for`: exact host
/// geometry, head dim a multiple of 16 up to 128, at least one head, enough pools), in mode
/// [`SCORES_TC2_MODE`], with at most [`SCORES_TC2_MAX_H`] heads.
pub(crate) fn scores_tc2_for(
    requested: bool,
    resolved: bool,
    tc: bool,
    mode: u32,
    heads: usize,
) -> bool {
    requested && resolved && tc && mode == SCORES_TC2_MODE && heads <= SCORES_TC2_MAX_H
}

/// 2026-10-07: Log once whether the lever engaged, and once why a launch that took
/// `dsa_index_scores_tc` with the lever on did not. A launch that did not take the tensor-core
/// path at all is `log_scores_tc`'s to report.
pub(crate) fn log_scores_tc2(
    requested: bool,
    engaged: bool,
    tc: bool,
    mode: u32,
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
) {
    let (d, heads) = (geom.index_head_dim, geom.index_heads);
    if engaged {
        static ENGAGED: OnceLock<()> = OnceLock::new();
        ENGAGED.get_or_init(|| {
            tracing::warn!(
                "{SCORES_TC2_ENGAGED_LINE} - dsa_index_scores_tc2 scores exact DSA selections \
                 (index_head_dim {d}, {heads} heads; bytes of dsa_index_scores_tc mode 1)"
            );
        });
    } else if requested && tc {
        static FELL_BACK: OnceLock<()> = OnceLock::new();
        FELL_BACK.get_or_init(|| {
            tracing::warn!(
                "METRALE_GLM_DSA_SCORES_TC2=1: NOT engaged, dsa_index_scores_tc runs (mode \
                 {mode}, needs {SCORES_TC2_MODE} = bf16; entry point resolved {}; {heads} heads, \
                 at most {SCORES_TC2_MAX_H})",
                kernels.index_scores_tc2.0 != 0
            );
        });
    }
}

#[cfg(test)]
#[path = "tc2_tests.rs"]
mod tests;
