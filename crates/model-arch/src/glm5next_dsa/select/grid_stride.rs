// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: `METRALE_GLM_DSA_GRID_STRIDE` (default on): the grid of a ceiling DSA selection
//! launch, the graph-replay decode path of `select_tokens`.
//!
//! A ceiling launch fixes its grid at the context ceiling so one captured graph serves every
//! context length. `dsa_kpool_compress` and `dsa_index_scores` took one block per pool, so
//! their grids were `max_pools + 1` and `max_pools` blocks: 16,385 and 16,384 at
//! `--max-seq-len 65536` (32,769 together), 196,609 and 196,608 at 786,432 (393,217
//! together). A block past the live pool count returned at once, but it still took a launch
//! slot, so every replay paid for the whole grid whatever the context, 33 times a step at
//! K = 3 (the +5 ms a step the race measured at short prompts, 2026-10-05; not re-measured
//! here).
//!
//! Both kernels now walk the live pools with a grid stride (see `dsa_indexer.cu`), so the
//! host launches [`stride_blocks`] blocks and a replay costs that, not the ceiling. The live
//! pool count still comes from `geom_dev`, and each pool is computed by one block exactly as
//! before (same thread mapping, same reduction order), so the selection is bit-identical to
//! the one-block-per-pool grid at every live count. The GPU gate is
//! `examples/dsa_grid_stride_microtest.rs`. Exact launches are not touched.
//!
//! `METRALE_GLM_DSA_GRID_STRIDE=0` launches the ceiling grid again, one block per ceiling pool.
//!
//! The stride grid is only sound for kernels that carry the loop: a kernel without it would
//! cover pools `0..G` and no others. `dsa_indexer.cu` therefore defines the no-op kernel
//! [`GRID_STRIDE_MARKER`] iff both kernels carry the loop, and a copy of the file without the
//! loop (the b300 fork) does not define it. [`stride_mode`] engages the stride grid only when
//! the lever is on and the module defines the marker; the lookup is made once, when the
//! kernels are resolved (`Glm5NextDsaKernels::grid_stride_marker`). Any other state launches
//! the earlier grids. The first ceiling launch logs one line, [`grid_stride_log_line`]:
//! `GRID_STRIDE: ENGAGED (G=<n>)`, `GRID_STRIDE: OFF (lever=0)` or
//! `GRID_STRIDE: OFF (kernel module lacks dsa_indexer_grid_stride_v1)`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`ceiling_grid_x`] returns at most the grid it is given, and exactly that grid with the
//!   lever off, so a lever-off launch is the earlier one.
//! - Once a stride grid has as many blocks as there are live pools, each block takes one pool
//!   and block `b` takes pool `b`, as in the earlier kernels.
//! - A ceiling launch gets the stride grid only when [`stride_mode`] is
//!   [`StrideMode::Engaged`] (lever on and marker resolved); every other mode gets the
//!   one-block-per-ceiling-pool grids, so a kernel module without the loop is never launched
//!   with fewer blocks than pools.

use std::sync::OnceLock;

use metrale_gpu_runtime::gpu::GpuBackend;

/// 2026-10-05: Blocks per SM in a stride grid. PROVISIONAL, from residency and not measured:
/// the kernels run 128 threads a block and an SM holds 1,536 threads, so 8 blocks are one
/// resident wave with room to spare. `examples/dsa_grid_stride_microtest.rs` times a sweep
/// of this factor.
pub const STRIDE_BLOCKS_PER_SM: usize = 8;

/// 2026-10-05: SM count assumed when the backend cannot say: the GB10's
/// (`kernels/gb10/HARDWARE.toml`).
pub const FALLBACK_SM_COUNT: usize = 48;

/// 2026-10-05: Entry point of the no-op kernel `dsa_indexer.cu` defines iff
/// `dsa_kpool_compress` and `dsa_index_scores` both carry the stride loop.
/// `Glm5NextDsaKernels::resolve` looks it up with the name spelled out, where
/// `crates/kernels/tests/kernel_lookups.rs` can read it; a test ties that spelling to this one.
pub const GRID_STRIDE_MARKER: &str = "dsa_indexer_grid_stride_v1";

/// 2026-10-05: Why a ceiling launch does or does not use the stride grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrideMode {
    /// 2026-10-05: The lever is on and the kernel module defines [`GRID_STRIDE_MARKER`].
    Engaged,
    /// 2026-10-05: `METRALE_GLM_DSA_GRID_STRIDE=0`.
    LeverOff,
    /// 2026-10-05: The lever is on but the kernel module does not define the marker: its
    /// kernels are one block per pool.
    MarkerMissing,
}

impl StrideMode {
    /// 2026-10-05: Whether the stride grid is launched.
    pub fn engaged(self) -> bool {
        self == Self::Engaged
    }
}

/// 2026-10-05: The mode for the lever's state and whether the marker resolved. The lever is
/// judged first, so a lever that is off reads [`StrideMode::LeverOff`] whatever the module has.
pub fn stride_mode(lever_on: bool, marker_resolved: bool) -> StrideMode {
    match (lever_on, marker_resolved) {
        (false, _) => StrideMode::LeverOff,
        (true, false) => StrideMode::MarkerMissing,
        (true, true) => StrideMode::Engaged,
    }
}

/// 2026-10-05: `METRALE_GLM_DSA_GRID_STRIDE=0` (surrounding blanks ignored) turns the stride
/// grid off; unset, blank and anything else leave it on.
pub(crate) fn parse_grid_stride(v: Option<&str>) -> bool {
    v.map(str::trim) != Some("0")
}

/// 2026-10-05: Whether the lever allows a ceiling selection the stride grid. On unless
/// `METRALE_GLM_DSA_GRID_STRIDE=0`; read once.
pub fn dsa_grid_stride() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_GRID_STRIDE").ok();
        parse_grid_stride(raw.as_deref())
    })
}

/// 2026-10-05: Blocks in a stride grid on a device with `sm_count` SMs (0 counts as 1).
pub fn stride_blocks(sm_count: usize) -> usize {
    sm_count.max(1) * STRIDE_BLOCKS_PER_SM
}

/// 2026-10-05: Grid x of a ceiling launch of a pool-indexed kernel whose one-block-per-pool
/// grid is `ceiling_x` blocks: [`stride_blocks`] of them when `stride`, never more than
/// `ceiling_x`; `ceiling_x` itself otherwise. 0 stays 0, as it was.
pub fn ceiling_grid_x(ceiling_x: usize, sm_count: usize, stride: bool) -> usize {
    if stride {
        ceiling_x.min(stride_blocks(sm_count))
    } else {
        ceiling_x
    }
}

/// 2026-10-05: Grid x of `dsa_kpool_compress` and of `dsa_index_scores` under a ceiling launch
/// over `max_pools` pools. One block per pool is `max_pools + 1` blocks for compress (the extra
/// one is the trailing partial pool's slot; the context is a whole number of pools, so it is
/// never live) and `max_pools` for the scores.
pub fn ceiling_grids(max_pools: usize, sm_count: usize, stride: bool) -> (usize, usize) {
    (
        ceiling_grid_x(max_pools + 1, sm_count, stride),
        ceiling_grid_x(max_pools, sm_count, stride),
    )
}

/// 2026-10-05: The SM count stride grids are sized with: asked of the first backend that
/// launches a ceiling selection, then kept (one device model per process; `sm_count` asks the
/// driver, which the trait says to do once, not per launch). [`FALLBACK_SM_COUNT`] when it
/// cannot say.
pub(super) fn device_sms(gpu: &dyn GpuBackend) -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        gpu.sm_count()
            .map_or(FALLBACK_SM_COUNT, |n| n as usize)
            .max(1)
    })
}

/// 2026-10-05: The one line logged on the first ceiling launch. `g` is [`stride_blocks`]; only
/// the engaged line names it.
pub(crate) fn grid_stride_log_line(mode: StrideMode, g: usize) -> String {
    match mode {
        StrideMode::Engaged => format!("GRID_STRIDE: ENGAGED (G={g})"),
        StrideMode::LeverOff => "GRID_STRIDE: OFF (lever=0)".to_string(),
        StrideMode::MarkerMissing => {
            format!("GRID_STRIDE: OFF (kernel module lacks {GRID_STRIDE_MARKER})")
        }
    }
}

/// 2026-10-05: Log once, on the first ceiling launch, which mode the lever and the kernel
/// module chose.
fn log_grid_stride(mode: StrideMode, g: usize) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::warn!("{}", grid_stride_log_line(mode, g)));
}

/// 2026-10-05: Grids `(compress, scores)` of a ceiling launch over `max_pools` pools, for the
/// lever's state as the environment gives it ([`dsa_grid_stride`]) and whether the kernel
/// module defines the marker (`Glm5NextDsaKernels::grid_stride_marker` resolved).
pub(super) fn ceiling_launch_grids(
    gpu: &dyn GpuBackend,
    marker_resolved: bool,
    max_pools: usize,
) -> (usize, usize) {
    ceiling_launch_grids_for(dsa_grid_stride(), gpu, marker_resolved, max_pools)
}

/// 2026-10-05: [`ceiling_launch_grids`] for an explicit lever state: the stride grids when
/// [`stride_mode`] engages them, else the one-block-per-ceiling-pool grids; logs the mode once.
pub(super) fn ceiling_launch_grids_for(
    lever_on: bool,
    gpu: &dyn GpuBackend,
    marker_resolved: bool,
    max_pools: usize,
) -> (usize, usize) {
    let mode = stride_mode(lever_on, marker_resolved);
    let sms = if mode.engaged() { device_sms(gpu) } else { 0 };
    log_grid_stride(mode, stride_blocks(sms));
    ceiling_grids(max_pools, sms, mode.engaged())
}

#[cfg(test)]
#[path = "grid_stride_tests.rs"]
mod tests;
