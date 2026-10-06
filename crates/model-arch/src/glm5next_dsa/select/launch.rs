// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `select_tokens`, the launcher for one DSA selection pass.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

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
    scratch.fits(cfg, geom)?;

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

    // 2026-09-25: Pool compression over the full pool count; the trailing partial pool is
    // written and marked invalid, and no later kernel reads it.
    KernelLaunch::new(gpu, kernels.kpool_compress)
        .grid([compress_x as u32, 1, 1])
        .block([d.min(1024) as u32, 1, 1])
        .arg_ptr(inputs.k_normed)
        .arg_ptr(inputs.gate)
        .arg_ptr(inputs.valid)
        .arg_ptr(inputs.ape)
        .arg_ptr(scratch.pool_keys)
        .arg_ptr(scratch.pool_indices)
        .arg_ptr(scratch.pool_valid)
        .arg_u32(seq_a as u32)
        .arg_u32(d as u32)
        .arg_u32(kp as u32)
        .arg_i32(inputs.first_key)
        .arg_ptr(gd)
        .launch(stream)?;

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
        log_scores_tc(tc_mode, tc, kernels, ceiling.is_some(), geom);
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
        let (handle, grid, block, smem) = if tc {
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
            .arg_ptr(scratch.pool_keys)
            .arg_ptr(inputs.weights)
            .arg_ptr(scratch.pool_indices)
            .arg_ptr(scratch.pool_valid)
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
        let scores = if tc { scores.arg_u32(tc_mode) } else { scores };
        scores.launch(stream)?;

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

    KernelLaunch::new(gpu, kernels.expand_selection)
        .grid([geom.q_rows as u32, 1, 1])
        .block([ROW_BLOCK, 1, 1])
        .arg_ptr(scratch.selected)
        .arg_ptr(scratch.pool_indices)
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
