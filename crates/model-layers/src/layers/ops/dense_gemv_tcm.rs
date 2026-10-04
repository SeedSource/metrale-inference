// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `METRALE_GLM_GEMV_TC=1`: the small-M (1..=16 rows) tensor-core dense GEMVs of
//! `kernels/gb10/common/dense_gemv_tcm.cu`, for BF16 weights and for FP8 E4M3 weight-only
//! copies (`Fp8DenseWeight`, one FP32 scale per output row), BF16 activations and output.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`try_bf16`] / [`try_fp8`] launch only when the lever is on, `1 <= m <= TCM_MAX_M`,
//!   `n > 0`, `k` is a positive multiple of the kind's k-block ([`K_STEP_BF16`] /
//!   [`K_STEP_FP8`]) and the chosen entry resolved; otherwise they return `Ok(false)` having
//!   launched nothing, and the caller runs its own kernel.
//! - The entry is chosen from the weight kind, `n` and `k` only ([`variant_for`]), never from
//!   `m`: every entry is row-invariant (row t's bits depend only on its own activation row,
//!   the weight and the shape), so a 1-row decode and a 16-row verify produce the same bits
//!   for a row. That holds across M only while the caller sends every M of a weight here; a
//!   caller that keeps M = 1 on another kernel gives up that agreement.
//! - Not bit-identical to the CUDA-core GEMVs (`dense_gemv_bf16*`, `dense_gemv_fp8w*`): the
//!   MMA accumulates in another order.
//! - The lever is read once per process, so CUDA-graph capture and replay see one route.

use std::sync::{Mutex, OnceLock};

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{DenseWeight, Fp8DenseWeight};

/// 2026-10-04: Rows per launch: two 8-token MMA tiles (`TCM_NB` in the kernel).
pub const TCM_MAX_M: u32 = 16;
/// 2026-10-04: Threads per CTA (`TCM_WARPS` = 8 warps that split K).
pub const BLOCK: u32 = 256;
/// 2026-10-04: K per k-block: 128 weight bytes per row in both kinds.
pub const K_STEP_BF16: u32 = 64;
pub const K_STEP_FP8: u32 = 128;
/// 2026-10-04: Kernel module (the file stem).
pub const MODULE: &str = "dense_gemv_tcm";

/// 2026-10-04: The kernel entries. Variants of one kind are bit-identical to each other
/// (kernel header), so [`variant_for`] may pick any of them per shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcmVariant {
    /// BF16, 16 rows per CTA, 8 k-blocks per trip, 1 CTA/SM.
    Bf16,
    /// BF16, 16 rows per CTA, 4 k-blocks per trip, 2 CTAs/SM.
    Bf16Ku4,
    /// BF16, 32 rows per CTA, 4 k-blocks per trip.
    Bf16Nt2,
    /// FP8, 16 rows per CTA, 4 k-blocks per trip, 2 CTAs/SM.
    Fp8,
    /// FP8, 16 rows per CTA, 8 k-blocks per trip, 1 CTA/SM.
    Fp8Ku8,
    /// FP8, 32 rows per CTA, 4 k-blocks per trip.
    Fp8Nt2,
}

impl TcmVariant {
    pub const ALL: [TcmVariant; 6] = [
        TcmVariant::Bf16,
        TcmVariant::Bf16Ku4,
        TcmVariant::Bf16Nt2,
        TcmVariant::Fp8,
        TcmVariant::Fp8Ku8,
        TcmVariant::Fp8Nt2,
    ];

    pub fn entry(self) -> &'static str {
        match self {
            TcmVariant::Bf16 => "dense_gemv_tcm_bf16",
            TcmVariant::Bf16Ku4 => "dense_gemv_tcm_bf16_ku4",
            TcmVariant::Bf16Nt2 => "dense_gemv_tcm_bf16_nt2",
            TcmVariant::Fp8 => "dense_gemv_tcm_fp8w",
            TcmVariant::Fp8Ku8 => "dense_gemv_tcm_fp8w_ku8",
            TcmVariant::Fp8Nt2 => "dense_gemv_tcm_fp8w_nt2",
        }
    }

    pub fn is_fp8(self) -> bool {
        matches!(
            self,
            TcmVariant::Fp8 | TcmVariant::Fp8Ku8 | TcmVariant::Fp8Nt2
        )
    }

    /// 2026-10-04: Weight rows per CTA (`16 * NT`).
    pub fn rows_per_cta(self) -> u32 {
        match self {
            TcmVariant::Bf16Nt2 | TcmVariant::Fp8Nt2 => 32,
            _ => 16,
        }
    }

    fn index(self) -> usize {
        TcmVariant::ALL.iter().position(|&v| v == self).unwrap()
    }
}

/// 2026-10-04: The `METRALE_GLM_GEMV_TC` rule over the looked-up value: on only for `1`.
pub fn gemv_tc_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// 2026-10-04: Whether `METRALE_GLM_GEMV_TC=1`; read once per process.
pub fn gemv_tc_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = gemv_tc_from(std::env::var("METRALE_GLM_GEMV_TC").ok().as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_GEMV_TC=1 - GLM-5.3 dense GEMVs of 1..={TCM_MAX_M} rows (and the \
                 BF16 LM head at 1..={TCM_MAX_M} rows) run on the tensor-core dense_gemv_tcm \
                 kernels, row-invariant; NOT byte-identical to the CUDA-core GEMVs"
            );
        }
        on
    })
}

/// 2026-10-04: Pure shape gate: whether `(m, n, k)` can run on a `fp8` / BF16 entry.
pub fn routes(m: u32, n: u32, k: u32, fp8: bool) -> bool {
    let step = if fp8 { K_STEP_FP8 } else { K_STEP_BF16 };
    (1..=TCM_MAX_M).contains(&m) && n > 0 && k > 0 && k.is_multiple_of(step)
}

/// 2026-10-04: The entry for a weight kind and shape. A function of `(n, k)` only, so a
/// weight runs the same entry at every M (variants of a kind are bit-identical anyway).
/// Chosen from the 2026-10-04 microtest (glm5next_gemv_tc_microtest, TIMING-VAR lines).
pub fn variant_for(_n: u32, _k: u32, fp8: bool) -> TcmVariant {
    if fp8 {
        TcmVariant::Fp8
    } else {
        TcmVariant::Bf16
    }
}

/// 2026-10-04: All six handles, cached per backend (a `KernelHandle` belongs to one
/// backend's loaded module). A missing entry is the zero handle.
pub fn handles(gpu: &dyn GpuBackend) -> [KernelHandle; 6] {
    type Cache = Mutex<Vec<(usize, [KernelHandle; 6])>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let key = gpu as *const dyn GpuBackend as *const () as usize;
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, h)) = guard.iter().find(|(k, _)| *k == key) {
        return *h;
    }
    let h = TcmVariant::ALL.map(|v| crate::layers::try_kernel(gpu, MODULE, v.entry()));
    guard.push((key, h));
    h
}

/// 2026-10-04: Launch `variant` (handle `kernel`) without consulting the lever:
/// `C[t] = A[t] @ W^T` (BF16) or `row_scale * (A[t] @ fp8(W)^T)` (FP8) for `t < m`, row `t`
/// of `C` at `output + t * out_stride` elements. `row_scale` is ignored (pass null) by the
/// BF16 entries. Refuses a shape [`routes`] declines.
#[allow(clippy::too_many_arguments)]
pub fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    variant: TcmVariant,
    input: DevicePtr,
    weight: DevicePtr,
    row_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        routes(m, n, k, variant.is_fp8()),
        "dense_gemv_tcm: m={m} n={n} k={k} outside the {} entry's contract (1..={TCM_MAX_M} \
         rows, k a multiple of {})",
        variant.entry(),
        if variant.is_fp8() {
            K_STEP_FP8
        } else {
            K_STEP_BF16
        }
    );
    ensure!(
        kernel.0 != 0,
        "dense_gemv_tcm: {} is not loaded",
        variant.entry()
    );
    ensure!(
        out_stride >= n,
        "dense_gemv_tcm: out_stride {out_stride} < n {n}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, variant.rows_per_cta()), 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// 2026-10-04: The routed entry for a kind and shape when the lever is on, the shape routes
/// and the entry resolved; `None` keeps the caller's kernel.
fn routed(
    gpu: &dyn GpuBackend,
    m: u32,
    n: u32,
    k: u32,
    fp8: bool,
) -> Option<(TcmVariant, KernelHandle)> {
    if !gemv_tc_enabled() || !routes(m, n, k, fp8) {
        return None;
    }
    let v = variant_for(n, k, fp8);
    let h = handles(gpu)[v.index()];
    (h.0 != 0).then_some((v, h))
}

/// 2026-10-04: Whether [`try_bf16`] (`fp8` false) / [`try_fp8`] would launch for this shape:
/// the lever is on, the shape routes and the entry resolved.
pub fn ready(gpu: &dyn GpuBackend, m: u32, n: u32, k: u32, fp8: bool) -> bool {
    routed(gpu, m, n, k, fp8).is_some()
}

/// 2026-10-04: `C[t] = A[t] @ W^T` on the BF16 tensor-core entry when the lever is on and the
/// shape routes. `Ok(false)`: nothing launched, the caller runs its own kernel.
#[allow(clippy::too_many_arguments)]
pub fn try_bf16(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<bool> {
    let Some((v, h)) = routed(gpu, m, n, k, false) else {
        return Ok(false);
    };
    launch(
        gpu,
        h,
        v,
        input,
        weight.weight,
        DevicePtr(0),
        output,
        m,
        n,
        k,
        out_stride,
        stream,
    )?;
    Ok(true)
}

/// 2026-10-04: `C[t] = row_scale * (A[t] @ fp8(W)^T)` on the FP8 tensor-core entry when the
/// lever is on and the shape routes. `Ok(false)`: nothing launched.
#[allow(clippy::too_many_arguments)]
pub fn try_fp8(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<bool> {
    let Some((v, h)) = routed(gpu, m, n, k, true) else {
        return Ok(false);
    };
    launch(
        gpu,
        h,
        v,
        input,
        weight.weight,
        weight.row_scale,
        output,
        m,
        n,
        k,
        out_stride,
        stream,
    )?;
    Ok(true)
}

#[cfg(test)]
#[path = "dense_gemv_tcm_tests.rs"]
mod dense_gemv_tcm_tests;
