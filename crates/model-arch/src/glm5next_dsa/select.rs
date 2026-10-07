// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3 DSA token selection: launch geometry, scratch, and the launcher for
//! the selection kernels.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`DsaSelectGeometry::plan`] returns `Ok` only when `select_k <= topk_np2`.
//! - `select_tokens` checks [`DsaSelectScratch::fits`] before its first launch and fails
//!   when the pass needs more scratch than was allocated.
//! - 2026-10-06: With `DsaSelectInputs::pool_cache` set, `select_tokens` fails before any
//!   launch when `dsa_kpool_compress_incr` is unresolved or `first_key` is not 0.
//! - 2026-10-06: With `METRALE_GLM_DSA_POOL_CACHE=1` the scratch plans and allocates no pool
//!   key/index/validity regions (null pointers), and `fits_with` refuses a pass without
//!   `pool_cache` on such a scratch. Lever off: the same regions and bytes as before.
//!
//! ```text
//! k_normed, gate, valid, ape  -> dsa_kpool_compress   -> pool keys / indices / valid
//! q, weights, q_pos           -> dsa_index_scores     -> [Q, P] scores + candidacy
//!                             -> dsa_topk_pools       -> [Q, select_k] pool ids
//!                             -> dsa_expand_selection -> [Q, out_width] token ids
//! ```
//!
//! # Shared memory does not grow with the context
//!
//! `dsa_topk_pools` walks the pool axis in tiles of [`topk_tile`] pools and keeps a running
//! best list of one tile, so its shared memory ([`topk_smem_for_tile`]) is fixed. The one
//! limit left is `select_k` of at most one tile, checked in [`DsaSelectGeometry::plan`]: at
//! `index_topk` 2048 and `index_kpool` 4, `select_k` is 512 against a 2,048-pool tile. The
//! context is bounded by the indexer cache (`state::max_dsa_context`).
//!
//! # No compaction pass
//!
//! [`crate::glm5next_dsa_ref::kept_pools`] keeps pool `p` only when every one of its slots
//! is in range and valid, counting from the first valid token. Over a contiguous cache with
//! no padding that set is the prefix `0 .. seq / kpool` ([`contiguous_pool_count`], checked
//! against `kept_pools` in `tests`), so the launcher uses the full arrays in place and
//! `dsa_compact_pools` is not launched. A left-padded batch would need the compaction;
//! [`DsaSelectGeometry::plan`] handles contiguous caches only.
//!
//! 2026-10-06: Scratch region 6 is the radix top-k work buffer ([`radix`],
//! `METRALE_GLM_DSA_TOPK_RADIX=1`), planned only with the lever on: lever-off sizes are unchanged.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{Glm5NextDsaConfig, Glm5NextDsaKernels};

/// 2026-09-25: Shared-memory ceiling the top-k select is budgeted against; the same value
/// as `SMEM_CEILING` in `examples/dsa_indexer_microtest.rs`.
pub const TOPK_SMEM_CEILING: usize = 49_152;

/// 2026-09-25: Threads per block for `dsa_index_scores`, and the least shared memory it is
/// given, in bytes (`select_tokens` requests `max(SCORES_BLOCK, 4 * index_heads)`).
const SCORES_BLOCK: u32 = 128;
/// 2026-09-25: Threads per block for `dsa_topk_pools` and `dsa_expand_selection`.
const ROW_BLOCK: u32 = 256;

/// 2026-10-01: Query rows per `dsa_index_scores_tiled` block (`DSA_TILE_ROWS` in
/// `kernels/gb10/common/dsa_indexer.cu`; 32 since v2, 16 in v1).
pub const SCORES_TILE_ROWS: usize = 32;
/// 2026-10-01: Pools per `dsa_index_scores_tiled` block (`DSA_TILE_POOLS`).
pub const SCORES_TILE_POOLS: usize = 64;
/// 2026-10-01: Threads per `dsa_index_scores_tiled` block (`DSA_TILE_THREADS`); the kernel's
/// thread-to-output map (4 rows x 4 pools per thread) assumes exactly this many.
pub const SCORES_TILED_BLOCK: u32 = 128;
/// 2026-10-01: Widest `index_head_dim` the tiled scores kernel is selected for: at 128 its
/// shared memory ([`scores_tiled_smem`]) is 49,152 B, exactly the 48 KiB default (the
/// launcher does not raise the dynamic shared-memory limit).
pub const SCORES_TILED_MAX_D: usize = 128;
/// 2026-10-01: Most index heads the tiled scores kernel is selected for, the envelope
/// `examples/dsa_indexer_tiled_bitparity_microtest.rs` is written for (the kernel itself has
/// no per-head storage).
pub const SCORES_TILED_MAX_H: usize = 64;

/// 2026-10-01: Fewest pools (`n_pools`) at which an exact launch takes the tiled kernel.
/// PROVISIONAL (estimate, not measured): at Q = 256 a 32 x 64 tile grid has
/// `8 * ceil(P / 64)` blocks, so below a few hundred pools most of GB10's 48 SMs idle and one
/// block's fixed per-head work (about 75 us estimated) is not repaid, while `dsa_index_scores`
/// costs about 0.08 ms per 64 pools there (measured 2026-10-01, n1). Tune from the P sweep in
/// `examples/dsa_indexer_tiled_bitparity_microtest.rs`.
pub const SCORES_TILED_MIN_POOLS: usize = 256;

/// 2026-10-01: Dynamic shared memory one `dsa_index_scores_tiled` block takes for head dim
/// `d`, in bytes: the tile's pool keys plus one head's q rows, both `[rows][d]` (swizzled,
/// unpadded).
pub fn scores_tiled_smem(d: usize) -> usize {
    (SCORES_TILE_POOLS + SCORES_TILE_ROWS) * d * 4
}

/// 2026-10-01: `dsa_index_scores_tiled` grid for `q_rows` rows and `n_pools` pools: pool
/// tiles on x, row tiles on y.
pub fn scores_tiled_grid(q_rows: usize, n_pools: usize) -> [u32; 3] {
    [
        n_pools.div_ceil(SCORES_TILE_POOLS) as u32,
        q_rows.div_ceil(SCORES_TILE_ROWS) as u32,
        1,
    ]
}

/// 2026-10-01: Whether a scores launch takes `dsa_index_scores_tiled`: only when requested
/// (`METRALE_GLM_DSA_SCORES_TILED=1`), with the entry point resolved, on an exact launch
/// with host geometry (the tiled kernel has no `geom_dev` path), and for `index_head_dim` a
/// nonzero multiple of 32 up to [`SCORES_TILED_MAX_D`] with 1 to [`SCORES_TILED_MAX_H`]
/// heads. Otherwise `dsa_index_scores` runs.
/// 2026-10-01 (v2): and only at `n_pools >= SCORES_TILED_MIN_POOLS` (PROVISIONAL).
pub(crate) fn scores_tiled_for(
    requested: bool,
    resolved: bool,
    exact_host_geom: bool,
    d: usize,
    heads: usize,
    n_pools: usize,
) -> bool {
    let shape = d > 0 && d.is_multiple_of(32) && d <= SCORES_TILED_MAX_D;
    let heads_ok = (1..=SCORES_TILED_MAX_H).contains(&heads);
    let wide = n_pools >= SCORES_TILED_MIN_POOLS;
    requested && resolved && exact_host_geom && shape && heads_ok && wide
}

/// 2026-10-01: Query rows per `dsa_index_scores_tc` block (`DSA_TC_ROWS`, the MMA's M).
pub const SCORES_TC_ROWS: usize = 16;
/// 2026-10-01: Pools per `dsa_index_scores_tc` block (`DSA_TC_POOLS`: 4 warps x two 8-pool
/// n-tiles).
pub const SCORES_TC_POOLS: usize = 64;
/// 2026-10-01: Threads per `dsa_index_scores_tc` block (`DSA_TC_THREADS`).
pub const SCORES_TC_BLOCK: u32 = 128;
/// 2026-10-01: Widest `index_head_dim` the tensor-core scores kernel takes (its B fragments
/// are register arrays of `DSA_TC_MAX_KSTEP` = 8 K steps of 16); it also needs a multiple of
/// 16.
pub const SCORES_TC_MAX_D: usize = 128;
/// 2026-10-01: Fewest pools at which an exact launch takes the tensor-core kernel.
/// PROVISIONAL (estimate, not measured): the same floor as [`SCORES_TILED_MIN_POOLS`]; below
/// it the scores pass is a small share of prefill. Tune from the timing sweep in
/// `examples/dsa_indexer_tc_microtest.rs`.
pub const SCORES_TC_MIN_POOLS: usize = 256;

/// 2026-10-01: `dsa_index_scores_tc` grid for `q_rows` rows and `n_pools` pools: 64-pool tiles
/// on x, 16-row tiles on y.
pub fn scores_tc_grid(q_rows: usize, n_pools: usize) -> [u32; 3] {
    [
        n_pools.div_ceil(SCORES_TC_POOLS) as u32,
        q_rows.div_ceil(SCORES_TC_ROWS) as u32,
        1,
    ]
}

/// 2026-10-01: Whether a scores launch takes `dsa_index_scores_tc`: only when a precision mode
/// is requested (`METRALE_GLM_DSA_SCORES_TC`, `mode` 1..=3; 0 is off), with the entry point
/// resolved, on an exact launch with host geometry (the kernel has no `geom_dev` path), for
/// `index_head_dim` a nonzero multiple of 16 up to [`SCORES_TC_MAX_D`], at least one head, and
/// at `n_pools >= SCORES_TC_MIN_POOLS` (PROVISIONAL). It takes precedence over the tiled
/// kernel when both are requested (`select_tokens`).
pub(crate) fn scores_tc_for(
    mode: u32,
    resolved: bool,
    exact_host_geom: bool,
    d: usize,
    heads: usize,
    n_pools: usize,
) -> bool {
    let shape = d > 0 && d.is_multiple_of(16) && d <= SCORES_TC_MAX_D;
    let mode_ok = (1..=3).contains(&mode);
    let wide = n_pools >= SCORES_TC_MIN_POOLS;
    mode_ok && resolved && exact_host_geom && shape && heads > 0 && wide
}

/// 2026-09-25: Tile width `dsa_topk_pools` walks the pool axis in.
///
/// The block holds two tiles, the running best list and the candidate tile, of `[f32, i32]`
/// pairs: `16 · T` bytes. This is the largest power of two `T` that fits
/// [`TOPK_SMEM_CEILING`]: 2,048.
///
/// `Glm5NextDsaLayer::decode_k` passes this value to `dsa_write_geom` as `tile`; the kernel
/// has no copy of it.
pub fn topk_tile() -> usize {
    let mut t = 2usize;
    while t * 2 * 16 <= TOPK_SMEM_CEILING {
        t *= 2;
    }
    t
}

/// 2026-09-25: Shared memory one `dsa_topk_pools` block needs for a tile of `t` pools.
pub fn topk_smem_for_tile(t: usize) -> usize {
    t * 2 * 8
}

/// 2026-09-25: Pools kept over a contiguous, unpadded cache of `seq` tokens.
///
/// A pool needs all `kpool` slots, so the trailing partial pool is not a pool. Checked
/// against `glm5next_dsa_ref::kept_pools` in `tests`.
pub fn contiguous_pool_count(kpool: usize, seq: usize) -> usize {
    seq / kpool
}

/// 2026-09-25: Launch geometry for one selection pass, computed by [`Self::plan`] before any
/// launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsaSelectGeometry {
    /// 2026-09-25: Tokens resident in the indexer cache.
    pub seq: usize,
    /// 2026-09-25: Query rows in this pass.
    pub q_rows: usize,
    /// 2026-09-25: Pools `dsa_kpool_compress` writes, including the trailing partial one.
    pub n_pools_full: usize,
    /// 2026-09-25: Complete pools, the prefix the later kernels read.
    pub n_pools: usize,
    /// 2026-09-25: Pools selected per query.
    pub select_k: usize,
    /// 2026-09-25: Emitted index-row width.
    pub out_width: usize,
    /// 2026-09-25: Tile width the top-k select walks the pool axis in: [`topk_tile`], or the
    /// next power of two (at least 2) at or above `n_pools` when that is smaller.
    pub topk_np2: usize,
    /// 2026-09-25: Shared memory `dsa_topk_pools` needs, in bytes.
    pub topk_smem: usize,
    /// 2026-09-25: Channels per compress block.
    pub index_head_dim: usize,
    pub index_heads: usize,
    pub index_kpool: usize,
}

impl DsaSelectGeometry {
    /// 2026-09-25: Plan a selection over a contiguous, unpadded cache.
    ///
    /// Fails when `cfg` fails validation, when `q_rows` or `seq` is 0, or when `select_k`
    /// exceeds `topk_np2`.
    pub fn plan(cfg: &Glm5NextDsaConfig, seq: usize, q_rows: usize) -> Result<Self> {
        cfg.validate()?;
        if q_rows == 0 {
            bail!("DSA select: q_rows must be > 0");
        }
        if seq == 0 {
            bail!("DSA select: seq must be > 0");
        }
        let kp = cfg.index_kpool;
        let n_pools = contiguous_pool_count(kp, seq);
        // 2026-09-25: `n_pools == 0` (fewer than `index_kpool` tokens) is planned, not
        // refused: `select_k` is 0, `select_tokens` skips scoring and top-k, and
        // `dsa_expand_selection` emits the visible tail, which is every token. The reference
        // `glm5next_dsa_ref::expand_selection` gives the same row
        // (`sub_pool_selection_is_dense_over_the_visible_tokens`).
        // `topk_np2` is the tile, smaller when the context is shorter than one tile.
        let tile = topk_tile();
        let topk_np2 = n_pools.next_power_of_two().max(2).min(tile);
        let topk_smem = topk_smem_for_tile(topk_np2);
        let select_k = cfg.select_k(n_pools);
        if select_k > topk_np2 {
            // 2026-09-25: The running best list is one tile, so it cannot hold more winners.
            // At GLM-5.3's index_topk 2048 and kpool 4, select_k is at most 512.
            bail!(
                "DSA select: select_k {select_k} exceeds the {topk_np2}-pool top-k tile \
                 ({TOPK_SMEM_CEILING} B shared-memory ceiling, index_topk={} \
                 index_kpool={kp}). Raise the ceiling or lower index_topk.",
                cfg.index_topk,
            );
        }
        Ok(Self {
            seq,
            q_rows,
            n_pools_full: seq.div_ceil(kp),
            n_pools,
            select_k,
            out_width: cfg.out_width(),
            topk_np2,
            topk_smem,
            index_head_dim: cfg.index_head_dim,
            index_heads: cfg.index_heads,
            index_kpool: kp,
        })
    }

    /// 2026-09-25: Bytes of each scratch region this pass writes, in [`DsaSelectScratch`]
    /// field order: pool keys (f32), pool indices (i32), pool validity (u8), scores (f32),
    /// candidacy (u8), selected pools (i32).
    /// 2026-10-06: Test-only since production plans through [`Self::scratch_bytes_with`].
    #[cfg(test)]
    fn scratch_bytes(&self) -> [usize; 6] {
        self.scratch_bytes_with(false)
    }

    /// 2026-10-06: [`Self::scratch_bytes`], with the three pool regions 0 when `pool_cache`
    /// (`METRALE_GLM_DSA_POOL_CACHE=1`): the pass then compresses into the state's persistent
    /// pool arrays and never touches the scratch's.
    fn scratch_bytes_with(&self, pool_cache: bool) -> [usize; 6] {
        let pools = if pool_cache { 0 } else { self.n_pools_full };
        [
            pools * self.index_head_dim * 4,
            pools * self.index_kpool * 4,
            pools,
            self.q_rows * self.n_pools * 4,
            self.q_rows * self.n_pools,
            self.q_rows * self.select_k * 4,
        ]
    }
}

/// 2026-09-25: Device-side inputs to a selection pass, all owned by the caller.
#[derive(Debug, Clone, Copy)]
pub struct DsaSelectInputs {
    /// 2026-09-25: `[seq, index_head_dim]` BF16 indexer keys, after the `k_norm` LayerNorm.
    pub k_normed: DevicePtr,
    /// 2026-09-25: `[seq, index_head_dim]` BF16 `index_kpool_compress_gate` projection.
    pub gate: DevicePtr,
    /// 2026-09-25: `[seq]` u8 per-key validity.
    pub valid: DevicePtr,
    /// 2026-09-25: `[index_kpool, index_head_dim]` f32 APE table. The checkpoint stores BF16;
    /// `build_dsa_weights` uploads it as f32.
    pub ape: DevicePtr,
    /// 2026-09-25: `[q_rows, index_heads, index_head_dim]` f32.
    pub q: DevicePtr,
    /// 2026-09-25: `[q_rows, index_heads]` f32, already carrying the `index_heads^-0.5`
    /// factor, which `dsa_index_scores` does not apply.
    pub weights: DevicePtr,
    /// 2026-09-25: `[q_rows]` i32 absolute position of each query.
    pub q_pos: DevicePtr,
    /// 2026-09-25: `[q_rows]` u8; a row whose entry is 0 selects nothing and stays all `-1`.
    pub q_mask: DevicePtr,
    /// 2026-09-25: Index of the first valid key; pooling starts here, so left padding is
    /// skipped.
    pub first_key: i32,
    /// 2026-09-25: `[5]` i32 device geometry written by `dsa_write_geom`, or NULL for the
    /// scalar path.
    ///
    /// A captured graph fixes every scalar argument, while S, the pool counts, the tile and
    /// `select_k` grow with the context; the kernels read them from here when it is non-null.
    /// Only with `q_rows == 1`: `select_tokens` refuses a ceiling launch otherwise.
    pub geom_dev: DevicePtr,
    /// 2026-10-06: `METRALE_GLM_DSA_POOL_CACHE=1`: the state's persistent pool arrays. Set,
    /// `select_tokens` compresses only pools from the watermark (`dsa_kpool_compress_incr`,
    /// `k_normed`/`gate` then being rings) into them, and the scores, top-k and expand kernels
    /// read them instead of the scratch's pool regions. `None`: today's full compress into the
    /// scratch.
    pub pool_cache: Option<super::pool_cache::DsaPoolCacheArgs>,
}

/// 2026-09-25: How a pass is launched: exactly, or at the context ceiling so one graph serves
/// any length.
///
/// `Ceiling` requires `DsaSelectInputs::geom_dev` and `q_rows == 1` (`select_tokens` refuses
/// otherwise). The kernels read their row strides (`out`/`valid_cand` stride `P`, `selected`
/// stride `select_k`) from the same device geometry, which is sound only because every
/// `r * stride` is then `0 * stride`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsaSelectLaunch {
    /// 2026-09-25: This step's exact grid, from the host geometry.
    Exact,
    /// 2026-09-25: Grid and shared memory fixed at `max_pools`; live extents come from
    /// `geom_dev`.
    /// 2026-10-05: The two pool-indexed grids are capped at a few waves of blocks that walk
    /// the live pools with a grid stride ([`grid_stride`], `METRALE_GLM_DSA_GRID_STRIDE`,
    /// default on), when the kernel module defines `dsa_indexer_grid_stride_v1`
    /// (`Glm5NextDsaKernels::grid_stride_marker`); `=0`, or a module without it, launches one
    /// block per ceiling pool.
    Ceiling { max_pools: usize },
}

/// 2026-09-25: Scratch the selection kernels write through, allocated once and reused across
/// steps.
///
/// Sized from the largest geometry the caller will run; [`Self::fits`] refuses a pass that
/// would outgrow it.
#[derive(Debug, Clone, Copy)]
pub struct DsaSelectScratch {
    pool_keys: DevicePtr,
    pool_indices: DevicePtr,
    pool_valid: DevicePtr,
    scores: DevicePtr,
    valid_cand: DevicePtr,
    selected: DevicePtr,
    /// 2026-09-25: `[q_rows, out_width]` i32 token ids, `-1` where nothing was selected. The
    /// result of the pass; `dsa_expand_selection` writes every slot of each row.
    tokens: DevicePtr,
    /// 2026-10-06: Radix top-k work buffer (region 6), NULL when none was planned.
    radix: DevicePtr,
    capacity: [usize; 7],
    tokens_bytes: usize,
}

mod launch;
mod scratch;
pub use launch::{PoolRows, compress_pools_only, select_tokens};
pub mod pool_once;
pub mod grid_stride;
pub mod radix;
pub mod shared;
pub mod split;
pub mod tc2;

#[cfg(test)]
mod tests;
