// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `METRALE_GLM_NV4_TC`: GLM NVFP4 GEMVs on the tensor-core kernels.
//!
//! Owner: model-arch (GLM-5.3).
//! `METRALE_GLM_NV4_TC=1` routes every `dense_fp8::nv4_gemv_uncounted` launch (1..=16 rows,
//! `K % 128 == 0`) through `w4a16_gemv_tc8` (M <= 8) or `w4a16_gemv_tc16`
//! (`kernels/gb10/common/w4a16_gemv_tc.cu`, routed by `ops::gemv_tc::tc_kernel`) instead of the
//! CUDA-core `w4a16_gemv` / `w4a16_gemv_batch*` tiers. The tensor-core work per weight byte does
//! not depend on M. Each token is its own MMA row, and the k-block order and warp reduction are
//! the same at every M, so a row's bits do not depend on M or on the other rows (gate
//! `examples/glm5next_nv4_tc_microtest.rs`). NOT byte-identical to the CUDA-core tiers: the
//! FP32 sums run in another order. A shape the route declines (`K % 128 != 0`,
//! `METRALE_NO_W4A16_TC`, kernels missing) keeps its CUDA-core tier at every M. Read once.

use std::sync::{Once, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops::gemv_tc;
use metrale_model_layers::weight_map::QuantizedWeight;

/// Whether `METRALE_GLM_NV4_TC=1` is set (read once).
pub fn nv4_tc() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_NV4_TC").as_deref() == Ok("1"))
}

/// Launches `C[m, n] = A[m, k] @ dequant(q)^T` on the tensor-core kernel when the lever is on
/// and the route takes the shape: `Ok(true)` when it launched, `Ok(false)` to keep the
/// CUDA-core tier. No allocation or sync, so it is capture-safe.
#[allow(clippy::too_many_arguments)]
pub fn try_launch(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    if !nv4_tc() {
        return Ok(false);
    }
    let Some((h, grid_x)) = gemv_tc::tc_kernel(gpu, m as u32, n as u32, k as u32) else {
        return Ok(false);
    };
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_NV4_TC=1: ENGAGED - GLM NVFP4 GEMVs (1..16 rows, K % 128 == 0) run \
             w4a16_gemv_tc8/tc16 (first: {m} rows, {n}x{k}); NOT byte-identical to the \
             CUDA-core tiers"
        );
    });
    KernelLaunch::new(gpu, h)
        .grid([grid_x, 1, 1])
        .block([gemv_tc::TC_BLOCK, 1, 1])
        .arg_ptr(a)
        .arg_ptr(q.weight)
        .arg_ptr(q.weight_scale)
        .arg_f32(q.weight_scale_2)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    Ok(true)
}
