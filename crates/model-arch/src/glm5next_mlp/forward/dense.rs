// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The GLM-5.3 dense SwiGLU MLP forward, used by the dense layers and the shared
//! expert.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `forward_dense` leaves a partial sum in `out` when `inter` is a TP shard; the caller
//!   all-reduces.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextMlpWorkspace;
use super::launch::{gemm, swiglu};
use crate::glm5next_mlp::weights::Glm5NextDenseMlpWeights;
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-09-29: Consecutive `(start, len)` slices of `rows` rows, each `slice` long except a
/// shorter last one; `slice >= rows` (or 0) is the single slice `(0, rows)`.
pub(crate) fn row_slices(rows: usize, slice: usize) -> impl Iterator<Item = (usize, usize)> {
    let s = if slice == 0 { rows.max(1) } else { slice };
    (0..rows).step_by(s).map(move |a| (a, s.min(rows - a)))
}

/// 2026-09-25: A BF16 SwiGLU MLP of width `inter`, `down(clamped_swiglu(gate(x), up(x)))`, for a
/// dense layer or the shared expert. Errors when `inter` or `m` does not fit the workspace.
#[allow(clippy::too_many_arguments)]
pub fn forward_dense(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextDenseMlpWeights,
    inter: usize,
    x: DevicePtr,
    out: DevicePtr,
    m: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    forward_dense_sliced(gpu, k, cfg, w, inter, x, out, m, m, ws, stream)
}

/// 2026-09-29: `forward_dense` with each GEMM issued once per `row_slices(m, slice)` slice, so
/// every GEMM sees the M it would see if the slices were separate calls; the elementwise SwiGLU
/// runs once over all `m * inter` values. `slice >= m` is exactly `forward_dense`.
#[allow(clippy::too_many_arguments)]
pub fn forward_dense_sliced(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextDenseMlpWeights,
    inter: usize,
    x: DevicePtr,
    out: DevicePtr,
    m: usize,
    slice: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    if inter == 0 || inter > ws.max_inter {
        bail!(
            "GLM dense MLP: width {inter} does not fit a workspace built for {}",
            ws.max_inter
        );
    }
    if m == 0 || m > ws.max_rows {
        bail!(
            "GLM dense MLP: {m} rows do not fit a workspace built for {}",
            ws.max_rows
        );
    }
    for (a, n) in row_slices(m, slice) {
        let xs = x.offset(a * cfg.hidden * 2);
        let act = a * inter * 2;
        gemm(
            gpu,
            k.gemm,
            k.gemv,
            k.gemv_batchm,
            xs,
            w.gate_proj,
            ws.a_gate.offset(act),
            n,
            inter,
            cfg.hidden,
            stream,
        )?;
        gemm(
            gpu,
            k.gemm,
            k.gemv,
            k.gemv_batchm,
            xs,
            w.up_proj,
            ws.a_up.offset(act),
            n,
            inter,
            cfg.hidden,
            stream,
        )?;
    }
    swiglu(
        gpu,
        k.swiglu,
        ws.a_gate,
        ws.a_up,
        ws.a_act,
        // 2026-09-25: Elementwise over all `m * inter` values.
        m * inter,
        cfg.swiglu_limit,
        stream,
    )?;
    for (a, n) in row_slices(m, slice) {
        gemm(
            gpu,
            k.gemm,
            k.gemv,
            k.gemv_batchm,
            ws.a_act.offset(a * inter * 2),
            w.down_proj,
            out.offset(a * cfg.hidden * 2),
            n,
            cfg.hidden,
            inter,
            stream,
        )?;
    }
    Ok(())
}
