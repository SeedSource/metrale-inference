// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`gemm`], the BF16 projection dispatch every `Glm5NextDsaLayer` method
//! launches its dense projections through.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - M above `DENSE_GEMV_BATCHM_MAX_M` with `glm5next_layer::cublas_wide_proj` on writes
//!   BF16, whatever `k` is.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// 2026-09-25: `C[M, N] = A[M, K] @ B[N, K]^T` with BF16 inputs.
///
/// M above `DENSE_GEMV_BATCHM_MAX_M` goes to cuBLASLt when `glm5next_layer::cublas_wide_proj`
/// is on (the default), and that arm writes BF16, so FP32-out callers pass M = 1 (all in this
/// file do). Otherwise `ops::dense_mm_bf16` picks the GEMV at M = 1, `batchm` at
/// 2..=`DENSE_GEMV_BATCHM_MAX_M` when it is not `KernelHandle(0)`, and the tile GEMM `k`
/// otherwise.
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
