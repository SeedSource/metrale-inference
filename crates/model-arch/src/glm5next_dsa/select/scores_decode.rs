// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `METRALE_GLM_DSA_SCORES_DECODE=1` (default off): a ceiling (graph-replay decode)
//! DSA selection scores its pools with `dsa_index_scores_decode`
//! (`kernels/gb10/common/dsa_indexer.cu`, section 2e) instead of the plain `dsa_index_scores`,
//! writing the same bytes (`out` bit patterns and `valid_cand`): q and the weights staged once
//! per block in shared memory, one pool per lane over 32-pool warp tiles with a grid stride,
//! keys read coalesced through a per-warp transpose buffer into registers, each score in the
//! plain kernel's exact arithmetic order.
//!
//! Why: the ceiling launch can take neither the tiled nor the tensor-core scorer (both need
//! exact host geometry), so decode ran the plain kernel, which re-reads the row's 16 KB q from
//! global per pool: 20.2 / 41.9 ms a decode step (11 layers x 3 rows) at 131K / 262K against a
//! ~4.4 ms key-byte floor at 262K (measured 2026-10-08, one GB10; spark-bench
//! `runs/race/mem/SCORES-ESTIMATE.md` plan A). Target not yet measured.
//!
//! Dispatch: [`scores_decode_for`]. GPU gate:
//! `examples/dsa_scores_decode_bitparity_microtest.rs` (bitwise against `dsa_index_scores` under
//! the ceiling launch, and timing).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - The `SCORES_DECODE_*` constants equal the `DSA_DEC_*` defines in `dsa_indexer.cu`, and the
//!   kernel takes `dsa_index_scores`' argument list (checked by a test that reads the file), so
//!   the launcher passes the same arguments.
//! - The grid is a function of the ceiling pool count and the SM count only, never of the live
//!   context, so a captured graph replays unchanged as the context grows.
//! - Lever off: `select_tokens` launches exactly what it did before.

use std::sync::OnceLock;

use super::DsaSelectGeometry;
use crate::glm5next_dsa::Glm5NextDsaKernels;

/// 2026-10-08: Threads per block (`DSA_DEC_THREADS`): four warps.
pub const SCORES_DECODE_BLOCK: u32 = 128;
/// 2026-10-08: Warps per block (`DSA_DEC_WARPS`); each owns 32-pool tiles, one pool per lane.
pub const SCORES_DECODE_WARPS: usize = 4;
/// 2026-10-08: The only `index_head_dim` the kernel takes (`DSA_DEC_D`; the lane holds a whole
/// pool key in registers); it traps on any other.
pub const SCORES_DECODE_D: usize = 128;
/// 2026-10-08: Row stride of the per-warp key transpose buffer, in floats (`DSA_DEC_LD`).
pub const SCORES_DECODE_LD: usize = 33;
/// 2026-10-08: Most index heads: the block stages q `[H][128]` and the weights `[H]` in shared
/// memory, and at this many heads the total ([`scores_decode_smem`]) is 41,664 B, inside the
/// default 48 KiB (the launcher does not raise the dynamic shared-memory limit).
pub const SCORES_DECODE_MAX_H: usize = 48;
/// 2026-10-08: Blocks per SM in the fixed grid. PROVISIONAL (estimate, not measured): the
/// kernel's `__launch_bounds__(128, 2)` lets two blocks share an SM; tune from the timing in
/// `examples/dsa_scores_decode_bitparity_microtest.rs`.
pub const SCORES_DECODE_BLOCKS_PER_SM: usize = 2;

/// 2026-10-08: Logged once, on the first launch that takes `dsa_index_scores_decode`.
pub const SCORES_DECODE_ENGAGED_LINE: &str = "METRALE_GLM_DSA_SCORES_DECODE=1: ENGAGED";

/// 2026-10-08: Pools one block scores per grid-stride step.
pub const fn scores_decode_block_pools() -> usize {
    SCORES_DECODE_WARPS * 32
}

/// 2026-10-08: Dynamic shared memory of one block at `heads` heads: q `[heads][128]`, the
/// weights `[heads]`, then one `[32][33]` key transpose buffer per warp (33,408 B at 32 heads).
pub fn scores_decode_smem(heads: usize) -> usize {
    (heads * SCORES_DECODE_D + heads + SCORES_DECODE_WARPS * 32 * SCORES_DECODE_LD) * 4
}

/// 2026-10-08: Grid x of a ceiling launch over `max_pools` pools on a device with `sm_count`
/// SMs: [`SCORES_DECODE_BLOCKS_PER_SM`] per SM, never more blocks than the ceiling has 128-pool
/// block steps, at least one. Fixed for the ceiling, so a graph replays at any live context;
/// the kernel walks the live pools with a grid stride.
pub fn scores_decode_grid_x(max_pools: usize, sm_count: usize) -> usize {
    let cap = sm_count.max(1) * SCORES_DECODE_BLOCKS_PER_SM;
    max_pools
        .div_ceil(scores_decode_block_pools())
        .min(cap)
        .max(1)
}

/// 2026-10-08: `1` (surrounding blanks ignored) is on; unset, `0` and anything else are off.
pub fn parse_scores_decode(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-08: `METRALE_GLM_DSA_SCORES_DECODE`, read once; a value other than unset, empty,
/// `0` or `1` is off, with one warning.
pub fn dsa_scores_decode() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_SCORES_DECODE").ok();
        let odd = |r: &&str| !r.is_empty() && r.trim() != "0" && r.trim() != "1";
        if let Some(r) = raw.as_deref().filter(odd) {
            tracing::warn!("METRALE_GLM_DSA_SCORES_DECODE={r} is not 0 or 1 - treated as off");
        }
        let on = parse_scores_decode(raw.as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_DSA_SCORES_DECODE=1 - ceiling (decode) DSA selections score pools \
                 with dsa_index_scores_decode (the same bytes as dsa_index_scores)"
            );
        }
        on
    })
}

/// 2026-10-08: Whether a scores launch takes `dsa_index_scores_decode`: the lever on, the entry
/// point resolved, a ceiling launch (device geometry; exact launches keep their own dispatch),
/// `index_head_dim` [`SCORES_DECODE_D`], 1 to [`SCORES_DECODE_MAX_H`] heads, and pool keys at a
/// 16-byte-aligned address (the kernel reads them as `float4`).
pub fn scores_decode_for(
    requested: bool,
    resolved: bool,
    ceiling: bool,
    d: usize,
    heads: usize,
    keys_aligned: bool,
) -> bool {
    let shape = d == SCORES_DECODE_D && (1..=SCORES_DECODE_MAX_H).contains(&heads);
    requested && resolved && ceiling && shape && keys_aligned
}

/// 2026-10-08: Log once whether the lever engaged, and once why a ceiling launch with the lever
/// on did not.
pub(crate) fn log_scores_decode(
    requested: bool,
    engaged: bool,
    ceiling: bool,
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
) {
    let (d, heads) = (geom.index_head_dim, geom.index_heads);
    if engaged {
        static ENGAGED: OnceLock<()> = OnceLock::new();
        ENGAGED.get_or_init(|| {
            tracing::warn!(
                "{SCORES_DECODE_ENGAGED_LINE} - dsa_index_scores_decode scores ceiling DSA \
                 selections (index_head_dim {d}, {heads} heads; bytes of dsa_index_scores)"
            );
        });
    } else if requested && ceiling {
        static FELL_BACK: OnceLock<()> = OnceLock::new();
        FELL_BACK.get_or_init(|| {
            tracing::warn!(
                "METRALE_GLM_DSA_SCORES_DECODE=1: NOT engaged, dsa_index_scores runs (entry \
                 point resolved {}; index_head_dim {d}, needs {SCORES_DECODE_D}; {heads} heads, \
                 at most {SCORES_DECODE_MAX_H}; or pool keys not 16-byte aligned)",
                kernels.index_scores_decode.0 != 0
            );
        });
    }
}

#[cfg(test)]
#[path = "scores_decode_tests.rs"]
mod tests;
