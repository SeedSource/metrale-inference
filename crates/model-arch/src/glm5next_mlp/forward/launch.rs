// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The kernel launches of the GLM-5.3 MLP forward: the BF16 GEMM, the NVFP4 GEMVs
//! (one projection, every routed slot, and the row-batched union) and the clamped SwiGLU.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::ACT_BLOCK;
use crate::glm5next_mlp::weights::{Glm5NextExpertPtrTable, Nvfp4Proj};

const W4_TILE: u32 = 64;

/// 2026-09-25: `C[M, N] = A[M, K] @ B[N, K]^T`, BF16 in and out. With `M` above
/// `DENSE_GEMV_BATCHM_MAX_M` and `cublas_wide_proj()` it runs on cuBLASLt; otherwise
/// `ops::dense_mm_bf16` picks the GEMV (M=1), the batched GEMV (M=2..=16) or the tile GEMM.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    gemv: KernelHandle,
    batchm: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    // 2026-10-03: `METRALE_GLM_DENSE_FP8=1`: a registered weight at <= 16 rows runs on its FP8
    // copy, and a wider call reads its BF16 dequant (`glm5next_layer::dense_fp8`); off, this
    // returns the caller's `b` without launching.
    let b = match crate::glm5next_layer::dense_fp8::route(gpu, gemv, a, b, c, m, n, kk, stream)? {
        crate::glm5next_layer::dense_fp8::Route::Done => return Ok(()),
        crate::glm5next_layer::dense_fp8::Route::Weight(w) => w,
    };
    if m > metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize
        && crate::glm5next_layer::cublas_wide_proj()
    {
        return metrale_model_layers::layers::ops::cublas_bf16_proj_dense(
            a, b, c, m as u32, n as u32, kk as u32, stream,
        );
    }
    metrale_model_layers::layers::ops::dense_mm_bf16(
        gpu,
        &metrale_model_layers::layers::ops::DenseMmKernels {
            gemm: k,
            gemv,
            batchm,
        },
        a,
        b,
        c,
        m,
        n,
        kk,
        stream,
    )
}

/// 2026-09-25: `C[1, N] = A[1, K] @ dequant(B)[N, K]^T` for one NVFP4 projection, on the
/// single-warp `k_sw` kernel when the target has it and on `k` otherwise. The host-dispatch
/// expert loop uses it.
pub(super) fn w4a16_gemv(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    k_sw: KernelHandle,
    a: DevicePtr,
    w: &Nvfp4Proj,
    c: DevicePtr,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    if k_sw.0 != 0 {
        return metrale_model_layers::layers::ops::w4a16_gemv_sw_raw(
            gpu, k_sw, a, w.packed, w.scale, w.scale_2, c, n as u32, kk as u32, stream,
        );
    }
    KernelLaunch::new(gpu, k)
        .grid([
            metrale_model_layers::layers::ops::w4a16_gemv_grid_x(n as u32),
            1,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.packed)
        .arg_ptr(w.scale)
        .arg_f32(w.scale_2)
        .arg_ptr(c)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)?;
    Ok(())
}

/// 2026-09-25: `C[M, N] = A[M, K] @ dequant(B)[N, K]^T` on the `w4a16_gemm` tile kernel. No
/// code calls it.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn w4a16(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    w: &Nvfp4Proj,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([
            (n as u32).div_ceil(W4_TILE),
            (m as u32).div_ceil(W4_TILE),
            1,
        ])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.packed)
        .arg_ptr(w.scale)
        .arg_f32(w.scale_2)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)?;
    Ok(())
}

pub(super) fn swiglu(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    n: usize,
    limit: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([(n as u32).div_ceil(ACT_BLOCK), 1, 1])
        .block([ACT_BLOCK, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_f32(limit)
        .launch(stream)?;
    Ok(())
}

/// 2026-09-25: `C[slot] = A[slot] @ dequant(expert[ids[slot]])^T` for every routed slot in one
/// launch; grid.y is the slot. A slot whose expert another rank owns is skipped, so `c` must
/// already hold that slot's contribution (zero).
#[allow(clippy::too_many_arguments)]
pub(super) fn w4a16_gemv_moe(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &Glm5NextExpertPtrTable,
    c: DevicePtr,
    ids: DevicePtr,
    n: usize,
    kk: usize,
    top_k: usize,
    num_experts: usize,
    input_stride: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([
            metrale_model_layers::layers::ops::w4a16_gemv_sw_grid_x(n as u32),
            top_k as u32,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed_ptrs)
        .arg_ptr(t.scale_ptrs)
        .arg_ptr(t.scale2_vals)
        .arg_ptr(c)
        .arg_ptr(ids)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(num_experts as u32)
        .arg_u32(input_stride as u32)
        .launch(stream)
}

/// 2026-09-25: The routed slots of `rows` rows, reading each selected expert's weights once:
/// grid.y walks the `rows * top_k` union entries of `glm5next_moe_row_union` (`u_eid`,
/// `u_slot`), and an unused entry (`u_eid < 0`) returns at once. The union tables stay on the
/// device.
#[allow(clippy::too_many_arguments)]
pub(super) fn w4a16_gemv_moe_batchm(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &Glm5NextExpertPtrTable,
    c: DevicePtr,
    u_eid: DevicePtr,
    u_slot: DevicePtr,
    n: usize,
    kk: usize,
    rows: usize,
    top_k: usize,
    num_experts: usize,
    a_row_stride: usize,
    a_slot_stride: usize,
    c_row_stride: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([
            metrale_model_layers::layers::ops::w4a16_gemv_sw_grid_x(n as u32),
            (rows * top_k) as u32,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed_ptrs)
        .arg_ptr(t.scale_ptrs)
        .arg_ptr(t.scale2_vals)
        .arg_ptr(c)
        .arg_ptr(u_eid)
        .arg_ptr(u_slot)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(num_experts as u32)
        .arg_u32(a_row_stride as u32)
        .arg_u32(a_slot_stride as u32)
        .arg_u32(c_row_stride as u32)
        .launch(stream)
}
