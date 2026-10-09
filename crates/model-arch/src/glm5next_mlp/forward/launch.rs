// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The kernel launches of the GLM-5.3 MLP forward: the BF16 GEMM, the NVFP4 GEMVs
//! (one projection, every routed slot, and the row-batched union) and the clamped SwiGLU.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use std::sync::{Once, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::ACT_BLOCK;
use crate::glm5next_mlp::Glm5NextMlpKernels;
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
/// device. `down` marks the down projection, which [`batchm_down_fast`] may take.
#[allow(clippy::too_many_arguments)]
pub(super) fn w4a16_gemv_moe_batchm(
    gpu: &dyn GpuBackend,
    down: bool,
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
    let fast = if down { batchm_down_fast(gpu, rows, kk) } else { None };
    let (k, grid_x) = match (fast, batchm_cols(gpu, rows)) {
        (Some(h), _) => (h, div_ceil(n as u32, MOE_DOWN_FAST_COLS)),
        (None, Some((h, j))) => (h, div_ceil(n as u32, 8 * j)),
        (None, None) => (
            k,
            metrale_model_layers::layers::ops::w4a16_gemv_sw_grid_x(n as u32),
        ),
    };
    KernelLaunch::new(gpu, k)
        .grid([grid_x, (rows * top_k) as u32, 1])
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

/// 2026-10-08: `METRALE_GLM_MOE_BATCHM_COLS=2|4|8`: columns per warp of the row-batched union
/// sweep (`w4a16_gemv_sw_moe_batchm_m<R>_c<J>`, kernels/gb10/common/w4a16_gemv.cu). Each output
/// keeps the J = 1 arithmetic, so the results are bit-identical; a block covers 8 J columns,
/// sharing its setup. Read once; any other value, or an image without the entries, keeps J = 1.
pub fn moe_batchm_cols() -> u32 {
    static J: OnceLock<u32> = OnceLock::new();
    *J.get_or_init(
        || match std::env::var("METRALE_GLM_MOE_BATCHM_COLS").as_deref() {
            Ok("2") => 2,
            Ok("4") => 4,
            Ok("8") => 8,
            _ => 1,
        },
    )
}

/// 2026-10-08: Output columns per block of `w4a16_gemv_sw_moe_batchm_down_m<R>` (8 warps x 16).
const MOE_DOWN_FAST_COLS: u32 = 128;

/// 2026-10-08: `METRALE_GLM_MOE_DOWN_FAST=1`: the down projection's union sweep runs
/// `w4a16_gemv_sw_moe_batchm_down_m<R>` (kernels/gb10/common/w4a16_gemv.cu), bit-identical to
/// `_m<R>` / `_m<R>_c<J>`, when `rows` is 2..=8, K <= 1024 and the image has all seven entries
/// (resolved once, ENGAGED logged once). Otherwise `None` keeps the car entry.
fn batchm_down_fast(gpu: &dyn GpuBackend, rows: usize, kk: usize) -> Option<KernelHandle> {
    static ON: OnceLock<bool> = OnceLock::new();
    static H: OnceLock<Option<Vec<KernelHandle>>> = OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("METRALE_GLM_MOE_DOWN_FAST").as_deref() == Ok("1"));
    if !on || !(2..=8).contains(&rows) || kk > 1024 {
        return None;
    }
    let hs = H.get_or_init(|| {
        (2..=8)
            .map(|r| {
                gpu.kernel("w4a16_gemv", &format!("w4a16_gemv_sw_moe_batchm_down_m{r}"))
                    .ok()
            })
            .collect::<Option<Vec<_>>>()
    });
    let h = hs.as_ref()?[rows - 2];
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_MOE_DOWN_FAST=1: ENGAGED - row-batched MoE down sweeps run \
             w4a16_gemv_sw_moe_batchm_down_m<R> (bit-identical)"
        );
    });
    Some(h)
}

/// 2026-10-09: Output columns per block of `w4a16_gemv_sw_moe_batchm_gateup_m<R>` (8 warps x 8),
/// per matrix.
const MOE_GATEUP_FAST_COLS: u32 = 64;

/// 2026-10-09: `METRALE_GLM_MOE_GATEUP_FAST=1`: the gate and up union sweeps of a `rows`-row
/// group run as ONE launch of `w4a16_gemv_sw_moe_batchm_gateup_m<R>`
/// (kernels/gb10/common/w4a16_gemv.cu), each output bit-identical to `_m<R>` / `_m<R>_c<J>`,
/// when `rows` is 2..=8 and the image has all seven entries (resolved once, ENGAGED logged
/// once). Otherwise `None` keeps the two car launches.
fn batchm_gateup_fast(gpu: &dyn GpuBackend, rows: usize) -> Option<KernelHandle> {
    static ON: OnceLock<bool> = OnceLock::new();
    static H: OnceLock<Option<Vec<KernelHandle>>> = OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("METRALE_GLM_MOE_GATEUP_FAST").as_deref() == Ok("1"));
    if !on || !(2..=8).contains(&rows) {
        return None;
    }
    let hs = H.get_or_init(|| {
        (2..=8)
            .map(|r| {
                gpu.kernel(
                    "w4a16_gemv",
                    &format!("w4a16_gemv_sw_moe_batchm_gateup_m{r}"),
                )
                .ok()
            })
            .collect::<Option<Vec<_>>>()
    });
    let h = hs.as_ref()?[rows - 2];
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_MOE_GATEUP_FAST=1: ENGAGED - row-batched MoE gate+up sweeps run \
             w4a16_gemv_sw_moe_batchm_gateup_m<R> (one launch, bit-identical)"
        );
    });
    Some(h)
}

/// 2026-10-09: The gate and up union sweeps of [`w4a16_gemv_moe_batchm`] (same `n`, `kk`,
/// strides and union tables for both) as one `w4a16_gemv_sw_moe_batchm_gateup_m<R>` launch
/// (grid x ceil(n / 64)). Returns `false` without launching when [`batchm_gateup_fast`] does
/// not engage; the caller then runs the two car launches.
#[allow(clippy::too_many_arguments)]
pub(super) fn w4a16_gemv_moe_batchm_gateup(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    gate: &Glm5NextExpertPtrTable,
    up: &Glm5NextExpertPtrTable,
    c_gate: DevicePtr,
    c_up: DevicePtr,
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
) -> Result<bool> {
    let Some(k) = batchm_gateup_fast(gpu, rows) else {
        return Ok(false);
    };
    KernelLaunch::new(gpu, k)
        .grid([
            div_ceil(n as u32, MOE_GATEUP_FAST_COLS),
            (rows * top_k) as u32,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate.packed_ptrs)
        .arg_ptr(gate.scale_ptrs)
        .arg_ptr(gate.scale2_vals)
        .arg_ptr(up.packed_ptrs)
        .arg_ptr(up.scale_ptrs)
        .arg_ptr(up.scale2_vals)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(u_eid)
        .arg_ptr(u_slot)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(num_experts as u32)
        .arg_u32(a_row_stride as u32)
        .arg_u32(a_slot_stride as u32)
        .arg_u32(c_row_stride as u32)
        .launch(stream)?;
    Ok(true)
}

/// 2026-10-08: The `_c<J>` entry for a `rows`-row sweep and its J, when [`moe_batchm_cols`] is
/// above 1 and the image has all seven entries for that J (resolved once, ENGAGED logged once).
fn batchm_cols(gpu: &dyn GpuBackend, rows: usize) -> Option<(KernelHandle, u32)> {
    static H: OnceLock<Option<Vec<KernelHandle>>> = OnceLock::new();
    let j = moe_batchm_cols();
    if j == 1 || !(2..=8).contains(&rows) {
        return None;
    }
    let hs = H.get_or_init(|| {
        (2..=8)
            .map(|r| {
                gpu.kernel("w4a16_gemv", &format!("w4a16_gemv_sw_moe_batchm_m{r}_c{j}"))
                    .ok()
            })
            .collect::<Option<Vec<_>>>()
    });
    let h = hs.as_ref()?[rows - 2];
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_MOE_BATCHM_COLS={j}: ENGAGED - row-batched MoE union sweeps run \
             w4a16_gemv_sw_moe_batchm_m<R>_c{j} ({j} columns per warp, bit-identical)"
        );
    });
    Some((h, j))
}

/// 2026-10-08: The router logits `[n, experts]` (FP32) of `n` rows of `x`: one
/// `gemv_f32` per row, or one batched launch per [`router_batchm`].
#[allow(clippy::too_many_arguments)]
pub(super) fn router_logits(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    x: DevicePtr,
    router: DevicePtr,
    logits: DevicePtr,
    n: usize,
    experts: usize,
    hidden: usize,
    stream: u64,
) -> Result<()> {
    let (bm, m) = router_batchm(gpu, n);
    for r in (0..n).step_by(m) {
        gemm(
            gpu,
            k.gemm_f32,
            k.gemv_f32,
            bm,
            x.offset(r * hidden * 2),
            router,
            logits.offset(r * experts * 4),
            m,
            experts,
            hidden,
            stream,
        )?;
    }
    Ok(())
}

/// 2026-10-08: `METRALE_GLM_MOE_ROUTER_BATCHM=1`: the router logits of 2..=16 rows run as one
/// `dense_gemv_bf16_fp32out_batchm` launch (the weight read once) instead of one
/// `dense_gemv_bf16_fp32out` per row. Each row keeps the single-row kernel's reduction (kernel
/// header, `examples/glm5next_moe_rowb_microtest.rs`), so the logits are bit-identical.
/// Returns the handle and the rows per launch; `(KernelHandle(0), 1)` keeps the per-row loop
/// (lever off, another row count, or no kernel).
fn router_batchm(gpu: &dyn GpuBackend, rows: usize) -> (KernelHandle, usize) {
    static ON: OnceLock<bool> = OnceLock::new();
    static H: OnceLock<KernelHandle> = OnceLock::new();
    let max = metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize;
    if !*ON.get_or_init(|| std::env::var("METRALE_GLM_MOE_ROUTER_BATCHM").as_deref() == Ok("1"))
        || !(2..=max).contains(&rows)
    {
        return (KernelHandle(0), 1);
    }
    let h = *H.get_or_init(|| {
        metrale_model_layers::layers::try_kernel(
            gpu,
            "dense_gemv_bf16_batchm",
            "dense_gemv_bf16_fp32out_batchm",
        )
    });
    if h.0 == 0 {
        return (h, 1);
    }
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_MOE_ROUTER_BATCHM=1: ENGAGED - router logits of 2..={max} rows run one \
             dense_gemv_bf16_fp32out_batchm launch (first: {rows} rows, bit-identical)"
        );
    });
    (h, rows)
}

/// 2026-10-08: `METRALE_GLM_MOE_UNION_SCAN=1`: the row union runs `glm5next_moe_row_union_scan`
/// (ids staged in shared memory, each first occurrence's index counted from shared flags)
/// instead of `glm5next_moe_row_union`'s serial global-memory rescans. Same launch shape and
/// the same tables, entry for entry. `default` when the lever is off or the kernel is missing.
pub(super) fn union_kernel(gpu: &dyn GpuBackend, default: KernelHandle) -> KernelHandle {
    static H: OnceLock<KernelHandle> = OnceLock::new();
    let h = *H.get_or_init(|| {
        if std::env::var("METRALE_GLM_MOE_UNION_SCAN").as_deref() != Ok("1") {
            return KernelHandle(0);
        }
        let h = metrale_model_layers::layers::try_kernel(
            gpu,
            crate::glm5next_mlp::W4A16_GEMV_MODULE,
            "glm5next_moe_row_union_scan",
        );
        if h.0 != 0 {
            tracing::warn!(
                "METRALE_GLM_MOE_UNION_SCAN=1: ENGAGED - the MoE row union runs \
                 glm5next_moe_row_union_scan (same tables)"
            );
        }
        h
    });
    if h.0 != 0 { h } else { default }
}
