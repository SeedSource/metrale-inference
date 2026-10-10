// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_GLM_DSA_SCORES_TC3=1` (default off): `dsa_index_scores_tc2`'s bytes
//! (`out` bit patterns and `valid_cand`) from `dsa_q_to_bf16` + `dsa_index_scores_tc3`
//! (`kernels/gb10/common/dsa_indexer.cu`, section 2d'). The prepass writes the pass's q once as
//! BF16 into select scratch region 7; the scores kernel covers 32 rows x 256 pools per block
//! (tc2: 128), so q is staged P / 256 times (tc2: P / 128) as BF16, with no conversion.
//!
//! Why: tc2 re-reads its block's FP32 q slice per head through L2 (P / 128 times a call) and
//! measured 53 TF/s; q traffic is the suspected limiter. GPU gate:
//! `examples/dsa_indexer_tc3_microtest.rs` (bitwise against tc2, and timing).
//!
//! Dispatch: [`scores_tc3_for`]. It runs only where `dsa_index_scores_tc2` would run
//! (`tc2::scores_tc2_for`), with both entry points resolved, at least [`SCORES_TC3_MIN_ROWS`]
//! query rows and a BF16 q region in the scratch; otherwise tc2 runs exactly as before.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - The `SCORES_TC3_*` constants equal the `DSA_TC3_*` / `DSA_TC2_*` defines in `dsa_indexer.cu`
//!   (checked by a test that reads the file); the scores kernel takes tc2's argument list with
//!   q the BF16 buffer, and shares tc2's shared-memory size ([`super::tc2::scores_tc2_smem`]).
//! - Lever off: region 7 is 0 bytes and `select_tokens` launches exactly what it did before.

use std::sync::OnceLock;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::DsaSelectGeometry;
use crate::glm5next_dsa::Glm5NextDsaKernels;

/// 2026-10-10: Query rows per block (`DSA_TC2_ROWS`, shared with tc2).
pub const SCORES_TC3_ROWS: usize = 32;
/// 2026-10-10: Pools per block (`DSA_TC3_POOLS`: 8 warps x four 8-pool n-tiles).
pub const SCORES_TC3_POOLS: usize = 256;
/// 2026-10-10: Threads per block (`DSA_TC3_THREADS`).
pub const SCORES_TC3_BLOCK: u32 = 256;
/// 2026-10-10: Fewest query rows a call needs for the lever to engage (one full row tile).
pub const SCORES_TC3_MIN_ROWS: usize = 32;
/// 2026-10-10: Threads per block of `dsa_q_to_bf16` and the most blocks it launches (it walks
/// the q floats with a grid stride).
pub const Q_TO_BF16_BLOCK: u32 = 256;
pub const Q_TO_BF16_MAX_BLOCKS: u32 = 4096;

/// 2026-10-10: Logged once, on the first launch that takes `dsa_index_scores_tc3`.
pub const SCORES_TC3_ENGAGED_LINE: &str = "METRALE_GLM_DSA_SCORES_TC3 ENGAGED";

/// 2026-10-10: Grid for `q_rows` rows and `n_pools` pools: 256-pool tiles on x, 32-row tiles on
/// y.
pub fn scores_tc3_grid(q_rows: usize, n_pools: usize) -> [u32; 3] {
    [
        n_pools.div_ceil(SCORES_TC3_POOLS) as u32,
        q_rows.div_ceil(SCORES_TC3_ROWS) as u32,
        1,
    ]
}

/// 2026-10-10: Grid x of `dsa_q_to_bf16` for `n4` float4 groups of q.
pub fn q_to_bf16_blocks(n4: usize) -> u32 {
    (n4.div_ceil(Q_TO_BF16_BLOCK as usize) as u32).clamp(1, Q_TO_BF16_MAX_BLOCKS)
}

/// 2026-10-10: Bytes of the BF16 q copy (select scratch region 7) a pass of `geom` needs:
/// `q_rows * heads * head_dim * 2` when `lever` is on and the pass has at least
/// [`SCORES_TC3_MIN_ROWS`] rows, else 0 (nothing allocated).
pub fn qbf_region_bytes(geom: &DsaSelectGeometry, lever: bool) -> usize {
    if lever && geom.q_rows >= SCORES_TC3_MIN_ROWS {
        geom.q_rows * geom.index_heads * geom.index_head_dim * 2
    } else {
        0
    }
}

/// 2026-10-10: `1` (surrounding blanks ignored) is on; unset, `0` and anything else are off.
pub fn parse_scores_tc3(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-10: `METRALE_GLM_DSA_SCORES_TC3`, read once; a value other than unset, empty, `0` or
/// `1` is off, with one warning.
pub fn dsa_scores_tc3() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_SCORES_TC3").ok();
        let odd = |r: &&str| !r.is_empty() && r.trim() != "0" && r.trim() != "1";
        if let Some(r) = raw.as_deref().filter(odd) {
            tracing::warn!("METRALE_GLM_DSA_SCORES_TC3={r} is not 0 or 1 - treated as off");
        }
        parse_scores_tc3(raw.as_deref())
    })
}

/// 2026-10-10: Whether a scores launch takes the BF16-q prepass and `dsa_index_scores_tc3`: the
/// lever on, `dsa_index_scores_tc2` selected (`tc2_on`), both entry points resolved, at least
/// [`SCORES_TC3_MIN_ROWS`] rows, and a scratch whose region 7 holds `need` bytes.
pub(crate) fn scores_tc3_for(
    requested: bool,
    resolved: bool,
    tc2_on: bool,
    q_rows: usize,
    region_ok: bool,
) -> bool {
    requested && resolved && tc2_on && q_rows >= SCORES_TC3_MIN_ROWS && region_ok
}

/// 2026-10-10: Launch `dsa_q_to_bf16`: the pass's q (`[q_rows, heads, head_dim]` f32) to the
/// BF16 region.
pub(crate) fn launch_q_to_bf16(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    q: DevicePtr,
    qbf: DevicePtr,
    geom: &DsaSelectGeometry,
    stream: u64,
) -> Result<()> {
    let n4 = geom.q_rows * geom.index_heads * geom.index_head_dim / 4;
    KernelLaunch::new(gpu, kernels.q_to_bf16)
        .grid([q_to_bf16_blocks(n4), 1, 1])
        .block([Q_TO_BF16_BLOCK, 1, 1])
        .arg_ptr(q)
        .arg_ptr(qbf)
        .arg_u32(n4 as u32)
        .launch(stream)
}

/// 2026-10-10: Both entry points resolved.
pub(crate) fn tc3_resolved(k: &Glm5NextDsaKernels) -> bool {
    k.index_scores_tc3.0 != 0 && k.q_to_bf16.0 != 0
}

/// 2026-10-10: Log once that the lever engaged, and once why a launch that took tc2 with the
/// lever on did not take tc3.
pub(crate) fn log_scores_tc3(
    requested: bool,
    engaged: bool,
    tc2_on: bool,
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    region_ok: bool,
) {
    let (d, heads) = (geom.index_head_dim, geom.index_heads);
    if engaged {
        static ENGAGED: OnceLock<()> = OnceLock::new();
        ENGAGED.get_or_init(|| {
            eprintln!(
                "{SCORES_TC3_ENGAGED_LINE}: dsa_q_to_bf16 + dsa_index_scores_tc3 ({SCORES_TC3_ROWS} \
                 rows x {SCORES_TC3_POOLS} pools per block; index_head_dim {d}, {heads} heads, \
                 {} query rows, {} pools; bytes of dsa_index_scores_tc2)",
                geom.q_rows, geom.n_pools
            );
        });
    } else if requested && tc2_on {
        static FELL_BACK: OnceLock<()> = OnceLock::new();
        FELL_BACK.get_or_init(|| {
            eprintln!(
                "METRALE_GLM_DSA_SCORES_TC3=1: NOT engaged, dsa_index_scores_tc2 runs (entry \
                 points resolved {}; {} query rows, needs {SCORES_TC3_MIN_ROWS}; scratch BF16 q \
                 region ok {region_ok})",
                tc3_resolved(kernels),
                geom.q_rows
            );
        });
    }
}

#[cfg(test)]
#[path = "tc3_tests.rs"]
mod tests;
