// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the grouped (per-expert) MoE prefill GEMMs over NVFP4 pointer tables, the token gather and `moe_silu_mul`; the batched routers are in `moe_grouped_a/topk.rs`.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! The pointer-table GEMMs launch `(n tiles, max_m_tiles, num_experts)`: `blockIdx.z` is the
//! expert and `blockIdx.y` a row tile within its rows (`expert_offsets[e]..expert_offsets[e+1]`
//! of the expert-sorted rows, `sorted_token_ids`). `max_m_tiles` must cover the busiest expert
//! in the kernel's own M tile.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

// 2026-09-25: `ops.rs` loads this file through `#[path]`, so its child needs a `#[path]` too.
#[path = "moe_grouped_a/topk.rs"]
mod topk;
// 2026-09-25: Re-exported so the routers are reached as `ops::moe_topk_*`.
pub use topk::{moe_topk_sigmoid_batched, moe_topk_softmax_batched, moe_topk_sqrtsoftplus_batched};

/// 2026-09-25: Launcher for `moe_w4a16_grouped_gemm`, whose weights are contiguous per expert
/// rather than in pointer tables. No production code calls it (@human-review: its grid
/// `(num_experts, 1, 1)` does not match the kernel, which takes the expert from `blockIdx.z`).
pub fn moe_w4a16_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed: DevicePtr,
    b_scale: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    num_experts: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_experts, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b_packed)
        .arg_ptr(b_scale)
        .arg_f32(scale2)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

const PTRTABLE_LEGACY_N_TILE: u32 = 64;

fn ptrtable_legacy_grid_x(n_out: u32) -> u32 {
    div_ceil(n_out, PTRTABLE_LEGACY_N_TILE)
}

/// 2026-09-25: `moe_w4a16_grouped_gemm_ptrtable` with a 256-row M tile and a 512-thread block.
/// `max_m_tiles` counts 256-row tiles: `MoeLayer::launch_grouped_gemm` divides the 64-row count
/// by 4. Loaded only under `METRALE_MOE_GROUPED_M256=1` (`moe/init.rs`); the measurement behind
/// that is on `launch_grouped_gemm`.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_ptrtable_m256(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([ptrtable_legacy_grid_x(n_out), max_m_tiles, num_experts])
        .block([512, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Pointer-table grouped W4A16 GEMM with 64-column tiles; one launch covers all
/// experts.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_ptrtable(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([ptrtable_legacy_grid_x(n_out), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Pointer-table grouped W4A16 GEMM for kernels with 128-column tiles.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_ptrtable_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Pointer-table grouped GEMM with FP8 E4M3 activations and transposed NVFP4
/// weights; `a_fp8` must already be FP8. Same launch shape as
/// [`moe_w4a16_grouped_gemm_ptrtable_n128`].
#[allow(clippy::too_many_arguments)]
pub fn moe_fp8_grouped_gemm_ptrtable_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Same launch shape as [`moe_w4a16_grouped_gemm_ptrtable_n128`], for the kernels
/// with a 64-wide K step.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_ptrtable_k64_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: [`moe_w4a16_fused_gate_up_n128`] for the kernels with a 64-wide K step.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_fused_gate_up_k64_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(2 * n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Gather token rows into expert-sorted order:
/// `permuted[i] = hidden[sorted_token_ids[i]]`, `permuted` being `[total_expanded, hidden]`. One
/// block per output row.
/// 2026-09-30: Called by `glm5next_mlp::forward_prefill_gemm`'s `METRALE_GLM_MOE_PREFILL_PERMUTE=1`
/// gate/up gather-once lever.
#[allow(clippy::too_many_arguments)]
pub fn moe_permute_tokens(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden_states: DevicePtr,
    permuted: DevicePtr,
    sorted_token_ids: DevicePtr,
    hidden: u32,
    total_expanded: u32,
    stream: u64,
) -> Result<()> {
    let threads = hidden.clamp(1, 256);
    KernelLaunch::new(gpu, kernel)
        .grid([total_expanded, 1, 1])
        .block([threads, 1, 1])
        .arg_ptr(hidden_states)
        .arg_ptr(permuted)
        .arg_ptr(sorted_token_ids)
        .arg_u32(hidden)
        .arg_u32(total_expanded)
        .launch(stream)
}

/// 2026-09-25: [`moe_w4a16_fused_gate_up_k64_n128`] with a 128-row M tile and a 256-thread
/// block. `max_m_tiles_m128` counts 128-row tiles: the call site halves the 64-row count
/// (`forward_prefill_routed.rs`).
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_fused_gate_up_k64_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles_m128: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(2 * n_out, 128), max_m_tiles_m128, num_experts])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Gate and up grouped GEMMs in one launch: the grid spans `2 * n_out` columns, the
/// first `n_out` for gate and the rest for up.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_fused_gate_up_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(2 * n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: `output[i] = silu(gate[i]) * up[i]` over `total_elements` BF16 values.
pub fn moe_silu_mul(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    output: DevicePtr,
    total_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(output)
        .arg_u32(total_elements)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::ptrtable_legacy_grid_x;

    #[test]
    fn legacy_ptrtable_grid_covers_every_64_column_tile() {
        assert_eq!(ptrtable_legacy_grid_x(1), 1);
        assert_eq!(ptrtable_legacy_grid_x(64), 1);
        assert_eq!(ptrtable_legacy_grid_x(65), 2);
        assert_eq!(ptrtable_legacy_grid_x(1024), 16);
        assert_eq!(ptrtable_legacy_grid_x(3072), 48);
    }
}
