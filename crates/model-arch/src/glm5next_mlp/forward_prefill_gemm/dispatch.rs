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
use super::tile::{
    GemmTile, max_m_tiles_from_offsets, prefill_gemm_exact_tiles, prefill_gemm_permute,
};

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

/// 2026-09-25: The grid height of the production grouped GEMM: the real expert histogram with
/// `prefill_gemm_exact_tiles()` (`copy_d2h_on_stream` synchronises the stream first, so the read
/// sees the sort's output), else the worst case, `ceil(rows * top_k / m_tile)`.
/// 2026-10-04: Lifted out of `forward_moe_grouped_prefill` unchanged so the W4A4 down arm shares
/// it.
fn grouped_max_m_tiles(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    ws: &Glm5NextMlpWorkspace,
    te: usize,
    tile: GemmTile,
    stream: u64,
) -> Result<u32> {
    let worst_case = te.div_ceil(tile.m_tile).max(1) as u32;
    if prefill_gemm_exact_tiles() {
        let mut off_raw = vec![0u8; (cfg.num_experts + 1) * 4];
        gpu.copy_d2h_on_stream(ws.expert_offsets(), &mut off_raw, stream)?;
        let offsets: Vec<i32> = off_raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(max_m_tiles_from_offsets(&offsets, worst_case, tile.m_tile))
    } else {
        Ok(worst_case)
    }
}

/// 2026-10-04: The down projection alone, over `ws.a_act` (BF16, expert-sorted, `te` rows of
/// `moe_intermediate`), into `ws.expert_out`, on the W4A16 kernel the non-W4A4 path selects for
/// this call: the fused `moe_w4a16_prefill_mma_down_*` when its `usable` contract holds (the
/// `METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1` arm of [`forward_moe_grouped_prefill`]), else the
/// production `moe_grouped_gemm` tile with the same grid and null gather map as its down launch.
#[allow(clippy::too_many_arguments)]
fn down_w4a16(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    te: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    if super::w4a16_mma::usable(&k.moe_prefill_mma, cfg, w, x, te, ws) {
        return super::w4a16_mma::forward_down(gpu, &k.moe_prefill_mma, cfg, w, te, ws, stream);
    }
    let tile = k.moe_grouped_tile;
    let max_m_tiles = grouped_max_m_tiles(gpu, cfg, ws, te, tile, stream)?;
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
        cfg.moe_intermediate,
        max_m_tiles,
        tile,
        stream,
    )
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

    // 2026-10-03: `METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1`: gate/up, SwiGLU and down on the
    // CUTLASS NVFP4 grouped GEMM (`cutlass_w4a4.rs`), same buffers and row order. Off (default):
    // one cached env read, nothing else. A refusal (logged) returns false before any output is
    // trusted and the W4A16 paths below run unchanged.
    // 2026-10-04: With `METRALE_GLM_MOE_W4A4_DOWN_W4A16=1` the CUTLASS arm stops after the
    // SwiGLU (`DownPending`) and the down GEMM runs on the W4A16 path this call would have taken.
    if super::cutlass_w4a4::lever_on() {
        match super::cutlass_w4a4::forward(gpu, k, cfg, w, x, te, ws, stream)? {
            super::cutlass_w4a4::Outcome::Done => return Ok(()),
            super::cutlass_w4a4::Outcome::DownPending => {
                return down_w4a16(gpu, k, cfg, w, x, te, ws, stream);
            }
            super::cutlass_w4a4::Outcome::Refused => {}
        }
    }

    // 2026-10-01: `METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1`: tile list + fused gate/up/SwiGLU +
    // down (`w4a16_mma.rs`), writing `ws.a_act` and `ws.expert_out` as below. Off (default), or
    // a contract miss: `usable` is false and everything below runs unchanged.
    if super::w4a16_mma::usable(&k.moe_prefill_mma, cfg, w, x, te, ws) {
        return super::w4a16_mma::forward(
            gpu,
            &k.moe_prefill_mma,
            cfg,
            w,
            x,
            ws.sorted_token_ids(),
            te,
            ws,
            stream,
        );
    }

    // 2026-09-25: With `prefill_gemm_exact_tiles()`, the grid height comes from the real expert
    // histogram; `copy_d2h_on_stream` synchronises the stream first, so the read sees the sort's
    // output. Otherwise it is the worst case, `ceil(rows * top_k / m_tile)`.
    // 2026-09-29: The tile `resolve` actually bound (the base tile after a fallback), so the grid
    // matches the kernel that runs.
    let tile = k.moe_grouped_tile;
    let max_m_tiles = grouped_max_m_tiles(gpu, cfg, ws, te, tile, stream)?;

    // 2026-09-30: `METRALE_GLM_MOE_PREFILL_PERMUTE=1`: gather `x` into `ws.moe_perm()` once, in
    // expert-sorted order (`moe_permute_tokens`, `moe_permute.cu`, the same module
    // `moe_sort_by_expert` ships in; launched through the existing, previously-uncalled wrapper
    // `metrale_model_layers::layers::ops::moe_permute_tokens` — its own doc: "permuted[i] =
    // hidden[sorted_token_ids[i]]. One block per output row."), then run gate AND up against
    // that buffer with `sorted_token_ids` NULL. Byte-identical to gathering through
    // `sorted_token_ids` inside each GEMM: `perm[row]` holds the exact BF16 bits
    // `A[sorted_token_ids[row]]` would have, so `smem_A[row][col]` is the same either way
    // (`examples/glm5next_moe_prefill_permute_microtest.rs`). Off (the `k.0 != 0` fallback
    // covers a PTX missing the kernel while the env var is set): unchanged from before this
    // lever existed — `a_in` is `x`, `stid_arg` is `ws.sorted_token_ids()`.
    let permute = prefill_gemm_permute() && k.moe_permute_tokens.0 != 0;
    let (a_in, stid_arg) = if permute {
        metrale_model_layers::layers::ops::moe_permute_tokens(
            gpu,
            k.moe_permute_tokens,
            x,
            ws.moe_perm(),
            ws.sorted_token_ids(),
            cfg.hidden as u32,
            te as u32,
            stream,
        )?;
        (ws.moe_perm(), DevicePtr(0))
    } else {
        (x, ws.sorted_token_ids())
    };

    // 2026-09-25: Gate and up: the kernel gathers the rows of `x` through `sorted_token_ids`,
    // unless the permute lever above already did it and handed a null `stid_arg`.
    grouped_gemm(
        gpu,
        k.moe_grouped_gemm,
        a_in,
        &w.ptrs.gate,
        ws.a_gate(),
        ws.expert_offsets(),
        stid_arg,
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
        a_in,
        &w.ptrs.up,
        ws.a_up(),
        ws.expert_offsets(),
        stid_arg,
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

/// 2026-09-29: Whether `forward_moe` over `rows` rows sends its routed experts through the
/// grouped GEMM (`forward_prefill_gemm`). The staged prefill merges only sub-chunks for which
/// this holds, so a merged window takes the same routed path each sub-chunk took on its own.
pub fn grouped_prefill_selected(
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    ws: &Glm5NextMlpWorkspace,
    rows: usize,
) -> bool {
    use crate::glm5next_layer::profile;
    rows > super::super::forward::MOE_ROW_BATCH_MAX_ROWS
        && rows >= super::tile::prefill_gemm_min_rows()
        && super::tile::prefill_gemm_enabled()
        && !super::super::forward::host_dispatch_forced()
        && !profile::trace_on()
        && k.moe_sort_by_expert.0 != 0
        && k.moe_grouped_gemm.0 != 0
        && k.combine_indexed.0 != 0
        && rows * cfg.top_k <= ws.max_total_expanded()
}
