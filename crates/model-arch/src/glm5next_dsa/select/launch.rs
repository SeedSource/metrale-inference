// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `select_tokens`, the launcher for one DSA selection pass.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - 2026-10-06: With `inputs.pool_cache` unset every launch is today's, argument for
//!   argument; set, only the compress kernel and the three pool pointers the later kernels
//!   read differ.

use super::grid_stride::ceiling_launch_grids;
use super::*;

/// 2026-09-25: Run the selection kernels, leaving `[q_rows, out_width]` token ids in
/// [`DsaSelectScratch::tokens`].
///
/// Fails before any launch when `scratch` is too small for `geom`, or when a ceiling launch
/// has no `geom_dev` or has `q_rows > 1`. The kernels are enqueued on `stream`; nothing here
/// synchronises.
#[allow(clippy::too_many_arguments)]
pub fn select_tokens(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    launch: DsaSelectLaunch,
    stream: u64,
) -> Result<()> {
    select_tokens_with(
        gpu, kernels, cfg, geom, inputs, scratch, launch, true, stream,
    )
}

/// 2026-10-07: [`select_tokens`], with `compress` choosing whether `dsa_kpool_compress` runs.
/// `false` (`METRALE_GLM_DSA_KPOOL_ONCE`, `pool_once`) leaves the pool keys, indices and
/// validity the scratch already holds: the caller compressed a window covering this pass's
/// pools earlier, and the first `geom.n_pools` pools there are what this pass's own compress
/// would write. `true` is `select_tokens` exactly.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_tokens_with(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    launch: DsaSelectLaunch,
    compress: bool,
    stream: u64,
) -> Result<()> {
    scratch.fits_with(cfg, geom, inputs.pool_cache.is_some())?;

    let d = geom.index_head_dim;
    let kp = geom.index_kpool;

    let ceiling = match launch {
        DsaSelectLaunch::Exact => None,
        DsaSelectLaunch::Ceiling { max_pools } => {
            if inputs.geom_dev.0 == 0 {
                bail!(
                    "DSA select: a ceiling launch has no host geometry to fall back on and \
                     needs `geom_dev`; passing NULL would run the frozen scalars."
                );
            }
            if geom.q_rows != 1 {
                bail!(
                    "DSA select: ceiling launch is decode-only (q_rows must be 1, got {}). \
                     At q_rows > 1 the row strides vary with the context and a replayed \
                     graph would index the wrong rows.",
                    geom.q_rows
                );
            }
            Some(max_pools)
        }
    };
    let gd = inputs.geom_dev;

    // 2026-09-25: Under a ceiling launch the kernels read S, the pool counts, the tile and
    // select_k from `geom_dev`, so the scalar arguments are unused. They are set from the
    // ceiling, so every capture at the same ceiling passes the same values.
    let (seq_a, npools_a, np2_a, selk_a) = match ceiling {
        Some(m) => (
            m * kp,
            m,
            m.next_power_of_two().max(2).min(topk_tile()),
            cfg.select_k(m),
        ),
        None => (geom.seq, geom.n_pools, geom.topk_np2, geom.select_k),
    };

    // 2026-09-25: Below `index_kpool` tokens there are no pools to score or sort, and a
    // zero-extent grid is not a valid launch, so `dsa_index_scores` and `dsa_topk_pools` are
    // skipped. `dsa_kpool_compress` still runs (`n_pools_full >= 1`), and
    // `dsa_expand_selection` writes the row to -1 and appends the visible tail, which is the
    // whole selection here. Under a ceiling launch both always run; with no live pools
    // `dsa_index_scores` blocks return at once and `dsa_topk_pools` selects nothing.
    let has_pools = geom.n_pools > 0 || ceiling.is_some();

    // 2026-10-05: Grid x of the two pool-indexed kernels. An exact launch is one block per pool.
    // A ceiling launch can make the kernels walk the live pools with a grid stride, so under
    // `METRALE_GLM_DSA_GRID_STRIDE` (default on) its grid is a few waves of blocks
    // (`grid_stride::stride_blocks`), not one per ceiling pool: a graph replay pays for the
    // grid, and every block past the live pool count was a block that did nothing. Only a
    // kernel module that defines `dsa_indexer_grid_stride_v1` has the loop
    // (`kernels.grid_stride_marker`); without it, or with the lever off, the grid is the
    // ceiling grid, `m + 1` blocks for compress (the trailing partial pool's slot) and `m`
    // for the scores.
    let (compress_x, scores_x) = match ceiling {
        None => (geom.n_pools_full, geom.n_pools),
        Some(m) => ceiling_launch_grids(gpu, kernels.grid_stride_marker.0 != 0, m),
    };

    // 2026-10-06: `METRALE_GLM_DSA_POOL_CACHE=1`: the incremental compress into the state's
    // persistent arrays, which the kernels below then read; otherwise the full compress into
    // the scratch, unchanged.
    let (pool_keys, pool_indices, pool_valid) = match inputs.pool_cache {
        Some(pc) => {
            launch_compress_incr(gpu, kernels, geom, inputs, &pc, ceiling, compress_x, stream)?;
            (pc.pk, pc.pidx, pc.pvalid)
        }
        None => {
            // 2026-09-25: Pool compression over the full pool count; the trailing partial
            // pool is written and marked invalid, and no later kernel reads it.
            // 2026-10-07: Skipped when the caller compressed the window's pools once
            // (`compress`, `METRALE_GLM_DSA_KPOOL_ONCE`); applies to the scratch path only,
            // the pool-cache arm above always compresses incrementally.
            if compress {
                launch_kpool_compress(
                    gpu,
                    kernels,
                    inputs,
                    scratch,
                    (compress_x, seq_a),
                    (d, kp),
                    stream,
                )?;
            }
            (scratch.pool_keys, scratch.pool_indices, scratch.pool_valid)
        }
    };

    if has_pools {
        // 2026-10-01: `METRALE_GLM_DSA_SCORES_TILED=1` swaps in `dsa_index_scores_tiled`
        // (same arguments, same bytes) on an exact launch; see `scores_tiled_for`.
        // 2026-10-01: `METRALE_GLM_DSA_SCORES_TC=<mode>` swaps in `dsa_index_scores_tc`
        // (tensor cores, NOT byte-identical; the same arguments plus `mode`) on an exact
        // launch, ahead of the tiled kernel; see `scores_tc_for`.
        let tc_mode = crate::glm5next_layer::levers::dsa_scores_tc();
        let tc = scores_tc_for(
            tc_mode,
            kernels.index_scores_tc.0 != 0,
            ceiling.is_none() && gd.0 == 0,
            d,
            geom.index_heads,
            geom.n_pools,
        );
        // 2026-10-07: `METRALE_GLM_DSA_SCORES_TC2=1` swaps `dsa_index_scores_tc2` (same
        // arguments, same bytes as mode 1) in for `dsa_index_scores_tc`; see `scores_tc2_for`.
        let tc2_requested = tc2::dsa_scores_tc2();
        let tc2_on = tc2::scores_tc2_for(
            tc2_requested,
            kernels.index_scores_tc2.0 != 0,
            tc,
            tc_mode,
            geom.index_heads,
        );
        tc2::log_scores_tc2(tc2_requested, tc2_on, tc, tc_mode, kernels, geom);
        if !tc2_on {
            log_scores_tc(tc_mode, tc, kernels, ceiling.is_some(), geom);
        }
        let requested = crate::glm5next_layer::levers::dsa_scores_tiled();
        let tiled = !tc
            && scores_tiled_for(
                requested,
                kernels.index_scores_tiled.0 != 0,
                ceiling.is_none() && gd.0 == 0,
                d,
                geom.index_heads,
                geom.n_pools,
            );
        if !tc {
            log_scores_tiled(requested, tiled, kernels, ceiling.is_some(), geom);
        }
        let (handle, grid, block, smem) = if tc2_on {
            (
                kernels.index_scores_tc2,
                tc2::scores_tc2_grid(geom.q_rows, geom.n_pools),
                tc2::SCORES_TC2_BLOCK,
                tc2::scores_tc2_smem(d, geom.index_heads) as u32,
            )
        } else if tc {
            (
                kernels.index_scores_tc,
                scores_tc_grid(geom.q_rows, geom.n_pools),
                SCORES_TC_BLOCK,
                0u32,
            )
        } else if tiled {
            (
                kernels.index_scores_tiled,
                scores_tiled_grid(geom.q_rows, geom.n_pools),
                SCORES_TILED_BLOCK,
                scores_tiled_smem(d) as u32,
            )
        } else {
            (
                kernels.index_scores,
                [scores_x as u32, geom.q_rows as u32, 1],
                SCORES_BLOCK,
                // 2026-09-25: `dsa_index_scores` keeps one f32 per index head in shared
                // memory and sums them in head order.
                SCORES_BLOCK.max((geom.index_heads * 4) as u32),
            )
        };
        let scores = KernelLaunch::new(gpu, handle)
            .grid(grid)
            .block([block, 1, 1])
            .shared_mem(smem)
            .arg_ptr(inputs.q)
            .arg_ptr(pool_keys)
            .arg_ptr(inputs.weights)
            .arg_ptr(pool_indices)
            .arg_ptr(pool_valid)
            .arg_ptr(inputs.valid)
            .arg_ptr(inputs.q_pos)
            .arg_ptr(scratch.scores)
            .arg_ptr(scratch.valid_cand)
            .arg_u32(geom.q_rows as u32)
            .arg_u32(npools_a as u32)
            .arg_u32(geom.index_heads as u32)
            .arg_u32(d as u32)
            .arg_u32(kp as u32)
            .arg_u32(seq_a as u32)
            .arg_f32((d as f32).powf(-0.5))
            .arg_ptr(gd);
        // 2026-10-01: `dsa_index_scores_tc` takes one argument more, the precision mode.
        // 2026-10-07: So does `dsa_index_scores_tc2` (`tc2_on` implies `tc`).
        let scores = if tc { scores.arg_u32(tc_mode) } else { scores };
        scores.launch(stream)?;

        // 2026-10-06: `METRALE_GLM_DSA_TOPK_RADIX=1` swaps in the exact radix select (same
        // `selected` bytes) when the pool count this launch dispatches on, `npools_a` (the
        // ceiling under a ceiling launch, so a graph never changes path), exceeds one top-k
        // tile; see `radix::radix_mode`. The scratch must hold its rows' work buffers.
        let mode = radix::radix_mode(
            radix::dsa_topk_radix(),
            radix::radix_resolved(kernels),
            scratch.capacity[6] >= radix::radix_work_bytes(cfg, geom.q_rows),
            npools_a,
        );
        radix::log_radix_mode(mode);
        if mode == radix::RadixMode::Engaged {
            let t = radix::RadixTopk {
                scores: scratch.scores,
                selected: scratch.selected,
                work: scratch.radix,
                q_rows: geom.q_rows,
                n_pools: npools_a,
                select_k: selk_a,
                kcap: radix::radix_kcap(cfg),
                geom_dev: gd,
            };
            radix::launch_topk_radix(gpu, kernels, &t, stream)?;
        } else {
            // 2026-09-25: `np2_a` is at most `topk_tile()`, so the request is at most
            // `topk_smem_for_tile(topk_tile())`, within `TOPK_SMEM_CEILING`.
            KernelLaunch::new(gpu, kernels.topk_pools)
                .grid([geom.q_rows as u32, 1, 1])
                .block([ROW_BLOCK, 1, 1])
                .shared_mem(topk_smem_for_tile(np2_a) as u32)
                .arg_ptr(scratch.scores)
                .arg_ptr(scratch.selected)
                .arg_u32(geom.q_rows as u32)
                .arg_u32(npools_a as u32)
                .arg_u32(np2_a as u32)
                .arg_u32(selk_a as u32)
                .arg_ptr(gd)
                .launch(stream)?;
        }
    }

    KernelLaunch::new(gpu, kernels.expand_selection)
        .grid([geom.q_rows as u32, 1, 1])
        .block([ROW_BLOCK, 1, 1])
        .arg_ptr(scratch.selected)
        .arg_ptr(pool_indices)
        .arg_ptr(scratch.valid_cand)
        .arg_ptr(inputs.valid)
        .arg_ptr(inputs.q_pos)
        .arg_ptr(inputs.q_mask)
        .arg_ptr(scratch.tokens)
        .arg_u32(geom.q_rows as u32)
        .arg_u32(npools_a as u32)
        .arg_u32(kp as u32)
        .arg_u32(seq_a as u32)
        .arg_u32(selk_a as u32)
        .arg_u32(geom.out_width as u32)
        .arg_i32(inputs.first_key)
        .arg_i32(cfg.always_select_tail as i32)
        .arg_ptr(gd)
        .launch(stream)?;

    Ok(())
}

/// 2026-10-06: `dsa_kpool_compress_incr` for a pool-cache pass: pools `[pk_start, live)` of
/// the state's arrays from the raw-row ring, then the device `pk_len` set to `S / kpool`.
/// Exact: `pc.pk_start` (the host watermark, already range-checked by
/// `Glm5NextDsaState::pool_select_args`) and `n_pools_full - pk_start` blocks (at least one,
/// which writes the device `pk_len` when no pool is due). Ceiling: the start comes from the geom
/// slot `dsa_write_geom_pk` fills, on the same grid as the full compress. Fails before the
/// launch when the kernel is unresolved or `first_key` is not 0 (the watermark counts pools
/// from row 0).
#[allow(clippy::too_many_arguments)]
fn launch_compress_incr(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    pc: &super::super::pool_cache::DsaPoolCacheArgs,
    ceiling: Option<usize>,
    compress_x: usize,
    stream: u64,
) -> Result<()> {
    let kp = geom.index_kpool;
    let (seq_a, grid_x, start) = match ceiling {
        Some(m) => (m * kp, compress_x, 0),
        None => {
            let start = pc.pk_start.min(geom.n_pools_full);
            (geom.seq, (geom.n_pools_full - start).max(1), start)
        }
    };
    let rows = PoolRows {
        k_normed: inputs.k_normed,
        gate: inputs.gate,
        valid: inputs.valid,
        ape: inputs.ape,
        first_key: inputs.first_key,
    };
    let dims = (geom.index_head_dim, kp);
    let grid = (seq_a, grid_x, start);
    launch_incr_raw(gpu, kernels, &rows, pc, dims, grid, inputs.geom_dev, stream)
}

/// 2026-10-06: The per-key inputs of a pool compress.
#[derive(Debug, Clone, Copy)]
pub struct PoolRows {
    /// 2026-10-06: The `k_normed` ring.
    pub k_normed: DevicePtr,
    /// 2026-10-06: The `gate` ring.
    pub gate: DevicePtr,
    /// 2026-10-06: `[seq]` u8 per-key validity (absolute rows).
    pub valid: DevicePtr,
    /// 2026-10-06: `[index_kpool, index_head_dim]` f32 APE table.
    pub ape: DevicePtr,
    /// 2026-10-06: Must be 0 (the cache counts pools from row 0).
    pub first_key: i32,
}

/// 2026-10-06: A compress-only pass with the pool cache (`METRALE_GLM_DSA_POOL_CACHE=1`):
/// pools `[pc.pk_start, ceil(seq / kpool))` into the persistent arrays with no scores, top-k
/// or expand, as an exact launch of `dsa_kpool_compress_incr` (the very launch
/// `select_tokens` makes for the same `seq` and `pk_start`). It advances the pool watermark
/// for rows written without a selection (MTP drafter context rows), so later writes stay
/// within the ring. The caller records it with `Glm5NextDsaState::note_selected`.
#[allow(clippy::too_many_arguments)]
pub fn compress_pools_only(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    rows: &PoolRows,
    pc: &super::super::pool_cache::DsaPoolCacheArgs,
    seq: usize,
    stream: u64,
) -> Result<()> {
    let kp = cfg.index_kpool.max(1);
    let n_full = seq.div_ceil(kp);
    let start = pc.pk_start.min(n_full);
    let grid_x = (n_full - start).max(1);
    let dims = (cfg.index_head_dim, kp);
    let grid = (seq, grid_x, start);
    launch_incr_raw(gpu, kernels, rows, pc, dims, grid, DevicePtr(0), stream)
}

/// 2026-10-06: One `dsa_kpool_compress_incr` launch (15 arguments, kernel order).
#[allow(clippy::too_many_arguments)]
fn launch_incr_raw(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    rows: &PoolRows,
    pc: &super::super::pool_cache::DsaPoolCacheArgs,
    (d, kp): (usize, usize),
    (seq_a, grid_x, start): (usize, usize, usize),
    geom_dev: DevicePtr,
    stream: u64,
) -> Result<()> {
    if kernels.kpool_compress_incr.0 == 0 {
        bail!(
            "DSA select: METRALE_GLM_DSA_POOL_CACHE=1 needs dsa_kpool_compress_incr, which this \
             kernel module does not define"
        );
    }
    if rows.first_key != 0 {
        bail!(
            "DSA select: the pool cache counts pools from row 0; first_key {} is not supported",
            rows.first_key
        );
    }
    KernelLaunch::new(gpu, kernels.kpool_compress_incr)
        .grid([grid_x as u32, 1, 1])
        .block([d.min(1024) as u32, 1, 1])
        .arg_ptr(rows.k_normed)
        .arg_ptr(rows.gate)
        .arg_ptr(rows.valid)
        .arg_ptr(rows.ape)
        .arg_ptr(pc.pk)
        .arg_ptr(pc.pidx)
        .arg_ptr(pc.pvalid)
        .arg_u32(seq_a as u32)
        .arg_u32(d as u32)
        .arg_u32(kp as u32)
        .arg_i32(rows.first_key)
        .arg_u32(pc.ring_rows as u32)
        .arg_u32(start as u32)
        .arg_ptr(pc.pk_len_dev)
        .arg_ptr(geom_dev)
        .launch(stream)?;
    Ok(())
}


/// 2026-10-07: The `dsa_kpool_compress` launch of [`select_tokens_with`] (and of
/// `pool_once::compress_window`): `grid_x` blocks over `seq` tokens of `[seq, d]` keys, `kp`
/// slots per pool, into `scratch`'s pool regions. The scalar arguments are those of the launch
/// `select_tokens` always issued, so the compress is the same call from both.
pub(super) fn launch_kpool_compress(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    (grid_x, seq): (usize, usize),
    (d, kp): (usize, usize),
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.kpool_compress)
        .grid([grid_x as u32, 1, 1])
        .block([d.min(1024) as u32, 1, 1])
        .arg_ptr(inputs.k_normed)
        .arg_ptr(inputs.gate)
        .arg_ptr(inputs.valid)
        .arg_ptr(inputs.ape)
        .arg_ptr(scratch.pool_keys)
        .arg_ptr(scratch.pool_indices)
        .arg_ptr(scratch.pool_valid)
        .arg_u32(seq as u32)
        .arg_u32(d as u32)
        .arg_u32(kp as u32)
        .arg_i32(inputs.first_key)
        .arg_ptr(inputs.geom_dev)
        .launch(stream)
}

/// 2026-10-01: Log once whether `METRALE_GLM_DSA_SCORES_TILED=1` engaged, and once why a
/// requested launch kept `dsa_index_scores`. A ceiling (graph-replay decode) launch and one
/// below `SCORES_TILED_MIN_POOLS` pools are expected fallbacks and log nothing, so the
/// warning names a prefill that did not engage.
fn log_scores_tiled(
    requested: bool,
    tiled: bool,
    kernels: &Glm5NextDsaKernels,
    ceiling: bool,
    geom: &DsaSelectGeometry,
) {
    let (d, heads) = (geom.index_head_dim, geom.index_heads);
    if tiled {
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_DSA_SCORES_TILED=1: ENGAGED - dsa_index_scores_tiled scores \
                 exact DSA selections of at least {SCORES_TILED_MIN_POOLS} pools \
                 (index_head_dim {d}, {heads} heads)"
            );
        });
    } else if requested && !ceiling && geom.n_pools >= SCORES_TILED_MIN_POOLS {
        static FELL_BACK: std::sync::Once = std::sync::Once::new();
        FELL_BACK.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_DSA_SCORES_TILED=1: NOT engaged, dsa_index_scores runs \
                 (entry point resolved {}, index_head_dim {d}, {heads} heads; the tiled \
                 kernel needs a multiple of 32 up to {SCORES_TILED_MAX_D} and up to \
                 {SCORES_TILED_MAX_H} heads, host geometry)",
                kernels.index_scores_tiled.0 != 0
            );
        });
    }
}

/// 2026-10-01: Log once whether `METRALE_GLM_DSA_SCORES_TC` engaged, and once why a requested
/// exact launch of at least `SCORES_TC_MIN_POOLS` pools kept the FP32 scorer. A ceiling
/// (graph-replay decode) launch and a short selection are expected fallbacks and log nothing.
fn log_scores_tc(
    mode: u32,
    tc: bool,
    kernels: &Glm5NextDsaKernels,
    ceiling: bool,
    geom: &DsaSelectGeometry,
) {
    let (d, heads) = (geom.index_head_dim, geom.index_heads);
    if tc {
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_DSA_SCORES_TC: ENGAGED (mode {mode}) - dsa_index_scores_tc scores \
                 exact DSA selections of at least {SCORES_TC_MIN_POOLS} pools on tensor cores \
                 (index_head_dim {d}, {heads} heads; NOT byte-identical)"
            );
        });
    } else if mode != 0 && !ceiling && geom.n_pools >= SCORES_TC_MIN_POOLS {
        static FELL_BACK: std::sync::Once = std::sync::Once::new();
        FELL_BACK.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_DSA_SCORES_TC: NOT engaged, the FP32 scorer runs (entry point \
                 resolved {}, index_head_dim {d}, {heads} heads; the tensor-core kernel needs a \
                 multiple of 16 up to {SCORES_TC_MAX_D} and host geometry)",
                kernels.index_scores_tc.0 != 0
            );
        });
    }
}
