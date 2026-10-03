// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the FP8 x FP8 and FP8-weight GEMMs, the per-token
//! FP8 activation quantizer, the block-scaled W8A8 GEMM, the FP8 weight and
//! scale transposes, block-scale widening and the M-dispatched BF16 GEMM
//! [`dense_mm_bf16`]. Re-exports the grouped MoE, W8A16 and GEMV launchers.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;
#[path = "gemm_quant_moe.rs"]
mod moe_grouped;
pub use moe_grouped::{
    MOE_E4M3_DN, MOE_E4M3_GU, MoeE4m3Tile, moe_bf16_grouped_gemm, moe_build_tile_worklist,
    moe_fp8_grouped_gemm, moe_gate_topk_fused, moe_w8a8_gateup_silu_e4m3, moe_w8a8_grouped_gemm,
    moe_w8a8_grouped_gemm_e4m3, moe_w8a8_grouped_gemm_pm4,
};
#[path = "gemm_quant_w8a16.rs"]
mod w8a16;
pub use w8a16::{
    w8a16_gemm, w8a16_gemm_n128_m128, w8a16_gemm_pipelined, w8a16_gemm_t, w8a16_gemm_t_pipelined,
    w8a16_gemv, w8a16_gemv_row_tiered,
};
#[path = "gemm_quant_gemv.rs"]
mod gemv;
pub use gemv::{
    DENSE_GEMV_BATCHM_DECODE_MAX_M, DENSE_GEMV_BATCHM_MAX_M, DENSE_GEMV_FP8W_BATCHM_MAX_M,
    dense_gemv, dense_gemv_batch2, dense_gemv_batchm, dense_gemv_batchm_fp32out,
    dense_gemv_batchm_split, dense_gemv_fp8w, dense_gemv_fp8w_batchm, dequant_fp8_rowscale_bf16,
};

/// 2026-09-25: FP8 x FP8 GEMM: A `[M, K]` FP8 E4M3 and B `[N, K]` FP8 E4M3
/// give C `[M, N]` BF16.
pub fn fp8_fp8_gemm_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: FP8-weight GEMM with a 128-row CTA (two 64-row chunks); each
/// CTA loads a B tile once for both chunks.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_n128_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: [`fp8_fp8_gemm_n128`] with a 128-row CTA (two 64-row chunks);
/// each CTA loads a B tile once for both chunks.
#[allow(clippy::too_many_arguments)]
pub fn fp8_fp8_gemm_n128_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Per-token FP8 activation quantization over 128-element
/// K-groups: A_fp8 `[M, K]` FP8 E4M3 and a_scale `[M, K/128]` FP32.
///
/// [`Fp8ActQuant::pick`] chooses the shared kernel or the Hopper twin for this
/// width and returns the kernel and its grid together; both run 128-thread
/// blocks. This is the only caller of `fp8_quant_log`, which logs the choice
/// once per branch.
#[allow(clippy::too_many_arguments)]
pub fn per_token_group_quant_fp8(
    gpu: &dyn GpuBackend,
    quant: Fp8ActQuant,
    input_bf16: DevicePtr,
    output_fp8: DevicePtr,
    a_scale: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let pick = quant.pick(m, k);
    super::fp8_quant_log(&pick, m, k);
    KernelLaunch::new(gpu, pick.kernel)
        .grid(pick.grid)
        .block([128, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(output_fp8)
        .arg_ptr(a_scale)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: W8A8 GEMM with per-token activation scales and per-block weight
/// scales, applied in an FP32 epilogue per 128-element K-group:
///
///   C[M, N] = bf16( Σ_g (FP8 MMA over K-group g) × a_scale[M, g] × b_scale[N/128, g] )
///
/// Inputs:
///   - `a_fp8`     [M, K] FP8 E4M3
///   - `a_scale`   [M, K/128] FP32 (from [`per_token_group_quant_fp8`])
///   - `b_fp8`     [N, K] FP8 E4M3
///   - `b_scale`   [N/128, K/128] FP32
///   - `output`    [M, N] BF16
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_t_blockscaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    a_scale: DevicePtr,
    b_fp8: DevicePtr,
    b_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    super::log_gemm_shape(gpu, "fp8_gemm_t_blockscaled", m, n, k);
    if let Some(pipe) = fp8_gemm_pipe_kernel(gpu, m, n, k)? {
        // 2026-09-27: `fp8_gemm_blockscaled_pipe_128x64`: 128 x 64 tiles, 256 threads, SmemBytes<128, 64, 3>.
        const PIPE_SMEM: u32 = 3 * (128 + 64) * 64 + 3 * 128 * 4 + 128 * 4;
        return KernelLaunch::new(gpu, pipe)
            .grid([n / 64, div_ceil(m, 128), 1])
            .block([256, 1, 1])
            .shared_mem(PIPE_SMEM)
            .arg_ptr(a_fp8)
            .arg_ptr(a_scale)
            .arg_ptr(b_fp8)
            .arg_ptr(b_scale)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream);
    }
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(a_scale)
        .arg_ptr(b_fp8)
        .arg_ptr(b_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-27: The pipelined twin of `fp8_gemm_t_blockscaled`
/// (`kernels/gb10/common/fp8_gemm_blockscaled_pipe.cu`), when the backend carries it, the
/// shape fits (N a multiple of its 64-wide tile, K of 128) and `METRALE_NO_FP8_GEMM_PIPE`
/// is absent. It performs the same MMAs and scale folds in the same order per output
/// element, so its output is bit-identical (microbench: 0 differing values at every
/// 35B projection shape, over every non-NaN E4M3 code), 1.7-3.2x faster on GB10. The
/// handle is memoized in the backend's `OpCache`.
fn fp8_gemm_pipe_kernel(
    gpu: &dyn GpuBackend,
    m: u32,
    n: u32,
    k: u32,
) -> Result<Option<KernelHandle>> {
    const MODULE: &str = "fp8_gemm_blockscaled_pipe";
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let off = *OFF.get_or_init(|| std::env::var_os("METRALE_NO_FP8_GEMM_PIPE").is_some());
    if off
        || m == 0
        || n == 0
        || !n.is_multiple_of(64)
        || !k.is_multiple_of(128)
        || !gpu.has_module(MODULE)
    {
        return Ok(None);
    }
    Ok(Some(gpu.op_cache().kernel(
        gpu,
        MODULE,
        "fp8_gemm_blockscaled_pipe_128x64",
    )?))
}

/// 2026-09-25: Transpose an FP8 weight on the GPU: `B[N, K]` to `B_t[K, N]`.
pub fn transpose_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let total = n as u64 * k as u64;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Widen an FP8 block-scale tensor to FP32 on the GPU, so the
/// block-scale kernels read `const float*`. `src` is `[total]` BF16 (dtype 0),
/// FP32 (1) or E8M0 (2); `dst` is `[total]` FP32. An E8M0 byte `e` becomes
/// `2^(e - 127)` (the bit pattern `e << 23`), `e = 0` becomes `2^-127` and
/// `e = 255` becomes 0.0.
pub fn widen_block_scale_f32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    total: u32,
    input_dtype: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(total)
        .arg_u32(input_dtype)
        .launch(stream)
}

/// 2026-09-25: Transpose FP32 block scales: `[N/128, K/128]` to
/// `[K/128, N/128]`.
pub fn transpose_block_scale(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    n_blocks: u32,
    k_blocks: u32,
    stream: u64,
) -> Result<()> {
    let total = n_blocks * k_blocks;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(n_blocks)
        .arg_u32(k_blocks)
        .launch(stream)
}

/// 2026-09-25: The three BF16 dense kernels [`dense_mm_bf16`] chooses from,
/// resolved once per projection site (the GLM-5.3 mixer and MLP sites bind
/// one each).
#[derive(Clone, Copy)]
pub struct DenseMmKernels {
    /// 2026-09-25: `dense_gemm_bf16`, the 16x16 tile GEMM: the arm for M above
    /// [`DENSE_GEMV_BATCHM_MAX_M`], and for any M whose GEMV kernel is missing.
    pub gemm: KernelHandle,
    /// 2026-09-25: `dense_gemv_bf16`, for M == 1.
    pub gemv: KernelHandle,
    /// 2026-09-25: `dense_gemv_bf16_batchm`, for 2..=[`DENSE_GEMV_BATCHM_MAX_M`]
    /// rows in one pass over the weight; `KernelHandle(0)` when unavailable.
    pub batchm: KernelHandle,
}

/// 2026-09-25: `C[M, N] = A[M, K] @ B[N, K]^T`, BF16 in and out, output row
/// stride `N`, dispatched on M: `dense_gemv_bf16` at M == 1,
/// `dense_gemv_bf16_batchm` at 2..=[`DENSE_GEMV_BATCHM_MAX_M`], and the tile
/// GEMM otherwise or when the chosen GEMV kernel is missing.
///
/// `batchm` reads the weight once for all M rows. Each of its rows follows
/// `dense_gemv_bf16`'s K order and reduction (gb10 kernels), so the two GEMV
/// arms give the same bits per row; the tile GEMM accumulates in a different
/// order.
#[allow(clippy::too_many_arguments)]
pub fn dense_mm_bf16(
    gpu: &dyn GpuBackend,
    k: &DenseMmKernels,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: Each grid follows its kernel's tile: `N_PER_BLOCK` = 4 outputs
    // per 256-thread block in both GEMV kernels, `GEMM_TILE` in the tile GEMM.
    if m == 1 && k.gemv.0 != 0 {
        return KernelLaunch::new(gpu, k.gemv)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            .launch(stream);
    }
    // 2026-09-25: Without `batchm`, M > 1 goes to the tile GEMM; warn once per
    // process.
    if m > 1 && k.batchm.0 == 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                "dense_mm_bf16: no dense_gemv_bf16_batchm on this target -- M>1 sites fall \
                 back to the tile GEMM (measured 3.6x slower at M<=8)"
            );
        });
    }
    if (2..=DENSE_GEMV_BATCHM_MAX_M as usize).contains(&m) && k.batchm.0 != 0 {
        return KernelLaunch::new(gpu, k.batchm)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            // 2026-09-25: `out_stride = N`: contiguous `[M, N]`, as the tile GEMM
            // writes.
            .arg_u32(n as u32)
            .launch(stream);
    }
    const GEMM_TILE: u32 = 16;
    KernelLaunch::new(gpu, k.gemm)
        .grid([
            (n as u32).div_ceil(GEMM_TILE),
            (m as u32).div_ceil(GEMM_TILE),
            1,
        ])
        .block([GEMM_TILE, GEMM_TILE, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)
}
