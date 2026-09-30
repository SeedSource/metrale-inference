// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The grouped-GEMM launch, and `forward_moe_grouped_prefill`, which sorts the routed
//! slots by expert and runs gate, up, SwiGLU and down over them.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::super::forward::Glm5NextMlpWorkspace;
use super::super::weights::Glm5NextMoeWeights;
use super::super::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use super::tile::{GemmTile, max_m_tiles_from_offsets, prefill_gemm_exact_tiles};

/// 2026-09-25: `C = gather(A) @ dequant(W_expert)^T` for every expert in one launch: grid
/// `tile.grid_dims(n_out, max_m_tiles, num_experts)`, `tile.threads` threads per block.
#[allow(clippy::too_many_arguments)]
fn grouped_gemm(
    gpu: &dyn GpuBackend,
    k: metrale_gpu_runtime::gpu::KernelHandle,
    a: DevicePtr,
    t: &super::super::weights::Glm5NextExpertPtrTable,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: usize,
    n_out: usize,
    kk: usize,
    max_m_tiles: u32,
    tile: GemmTile,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid(tile.grid_dims(n_out, max_m_tiles, num_experts))
        .block([tile.threads, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed_ptrs)
        .arg_ptr(t.scale_ptrs)
        .arg_ptr(t.scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts as u32)
        .arg_u32(n_out as u32)
        .arg_u32(kk as u32)
        .launch(stream)
}

/// 2026-09-25: Sort, grouped gate GEMM, grouped up GEMM, clamped SwiGLU, grouped down GEMM, over
/// all `rows` rows in one launch each. Reads the router's `ws.ids` and leaves the routed outputs
/// in `ws.expert_out` in expert-sorted row order, addressed by `ws.token_to_perm`; the caller
/// finishes with `glm5next_moe_combine_indexed`. The caller must zero `ws.expert_out` first: the
/// GEMM writes nothing for an expert another EP rank owns. Errors when `rows * top_k` exceeds
/// the workspace.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_moe_grouped_prefill(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    rows: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    let te = rows * cfg.top_k;
    let mi = cfg.moe_intermediate;
    if te > ws.max_total_expanded() {
        bail!(
            "GLM MoE grouped prefill: {te} routed slots exceed the {} a workspace built for \
             {} rows holds",
            ws.max_total_expanded(),
            ws.max_rows()
        );
    }

    // 2026-09-25: `moe_sort_by_expert` indexes its counts with each id unchecked, so every id
    // must be a real expert. `glm5next_router_topk` writes -1 only when `top_k > num_experts`,
    // which `Glm5NextMlpConfig::validate` refuses.
    KernelLaunch::new(gpu, k.moe_sort_by_expert)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ws.ids())
        .arg_ptr(ws.sorted_token_ids())
        .arg_ptr(ws.sorted_expert_ids())
        .arg_ptr(ws.expert_offsets())
        .arg_ptr(ws.token_to_perm())
        .arg_u32(te as u32)
        .arg_u32(cfg.num_experts as u32)
        .arg_u32(cfg.top_k as u32)
        .launch(stream)?;

    // 2026-09-25: With `prefill_gemm_exact_tiles()`, the grid height comes from the real expert
    // histogram; `copy_d2h_on_stream` synchronises the stream first, so the read sees the sort's
    // output. Otherwise it is the worst case, `ceil(rows * top_k / m_tile)`.
    // 2026-09-29: The tile `resolve` actually bound (the base tile after a fallback), so the grid
    // matches the kernel that runs.
    let tile = k.moe_grouped_tile;
    let worst_case = te.div_ceil(tile.m_tile).max(1) as u32;
    let max_m_tiles = if prefill_gemm_exact_tiles() {
        let mut off_raw = vec![0u8; (cfg.num_experts + 1) * 4];
        gpu.copy_d2h_on_stream(ws.expert_offsets(), &mut off_raw, stream)?;
        let offsets: Vec<i32> = off_raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        max_m_tiles_from_offsets(&offsets, worst_case, tile.m_tile)
    } else {
        worst_case
    };

    // 2026-09-25: Gate and up: the kernel gathers the rows of `x` through `sorted_token_ids`.
    grouped_gemm(
        gpu,
        k.moe_grouped_gemm,
        x,
        &w.ptrs.gate,
        ws.a_gate(),
        ws.expert_offsets(),
        ws.sorted_token_ids(),
        cfg.num_experts,
        mi,
        cfg.hidden,
        max_m_tiles,
        tile,
        stream,
    )?;
    grouped_gemm(
        gpu,
        k.moe_grouped_gemm,
        x,
        &w.ptrs.up,
        ws.a_up(),
        ws.expert_offsets(),
        ws.sorted_token_ids(),
        cfg.num_experts,
        mi,
        cfg.hidden,
        max_m_tiles,
        tile,
        stream,
    )?;

    // 2026-09-25: GLM's clamped SwiGLU, elementwise over every sorted row. Rows of an expert
    // another rank owns hold stale values; the down GEMM skips that expert, so they are not
    // used.
    super::super::forward::swiglu_rows(
        gpu,
        k.swiglu,
        ws.a_gate(),
        ws.a_up(),
        ws.a_act(),
        te * mi,
        cfg.swiglu_limit,
        stream,
    )?;

    // 2026-09-25: Down: the activations are already in expert-sorted order, so the gather map is
    // null (`DevicePtr(0)`) and the kernel reads row `cta_m + row` directly.
    grouped_gemm(
        gpu,
        k.moe_grouped_gemm,
        ws.a_act(),
        &w.ptrs.down,
        ws.expert_out(),
        ws.expert_offsets(),
        DevicePtr(0),
        cfg.num_experts,
        cfg.hidden,
        mi,
        max_m_tiles,
        tile,
        stream,
    )?;
    Ok(())
}
