// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the dense GEMVs: BF16 for one row, two rows and M
//! rows, and FP8-weight for one row.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`dense_gemv_batchm`] returns an error, without launching, when `m` is
//!   outside `1..=DENSE_GEMV_BATCHM_MAX_M`.

use super::*;

/// 2026-09-25: Dense BF16 GEMV for one row, `C = A @ B^T`: A `[1, K]`, B
/// `[N, K]`, C `[1, N]`. Each 256-thread block computes 4 outputs, 64 threads
/// per output (kernel `dense_gemv_bf16(A, B, C, N, K)`).
pub fn dense_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Dense BF16 GEMV for two rows in one pass over the weight. Each
/// row follows `dense_gemv_bf16`'s K order and reduction (gb10 kernels).
///
/// `input`: `[2, K]` BF16, contiguous; `output`: two rows at
/// `output + t * out_stride` (BF16 elements). The batched SSM decode uses it
/// for a BF16 `in_proj_qkvz` (`qwen3_ssm/trait_decode_batched.rs`).
///
/// Kernel: `dense_gemv_bf16_batch2(A, B, C, N, K, out_stride)`
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
/// 2026-09-25: Rows `dense_gemv_bf16_batchm` computes: `MAX_M` in
/// `kernels/gb10/common/dense_gemv_bf16_batchm.cu`. The kernel clamps a
/// larger M, so [`dense_gemv_batchm`] refuses above this value. Each row's
/// accumulation does not depend on M, so a row's result is the same at every
/// batch width.
///
/// Other readers: the row thresholds of the GLM-5.3 DSA, KDA and MLP dispatch
/// (`glm5next_dsa/layer.rs`, `glm5next_kda/mod.rs`, `glm5next_mlp/forward.rs`,
/// `glm5next_layer/levers.rs`), the `verify_k` workspace size
/// (`glm5_next_load/loader.rs`) and the clamp in
/// `target_defaults::resolve_batchm_max`.
pub const DENSE_GEMV_BATCHM_MAX_M: u32 = 16;

/// 2026-09-25: The widest batch that the MTP row dispatch
/// (`mtp_head/row_dispatch.rs`) and the BF16 arm of the model's
/// `lm_head_batched` (`impl_a3_lm_head.rs`) hand to `dense_gemv_bf16_batchm`;
/// `target_defaults::BASELINE_BATCHM_MAX` is set from it. Above it those paths
/// choose other kernels, some of which accumulate in a different order, so
/// this value decides which bits a decode of a given width produces.
pub const DENSE_GEMV_BATCHM_DECODE_MAX_M: u32 = 8;

/// 2026-09-25: Dense BF16 GEMV for M rows, `C[t] = A[t] @ B^T` for `t` in
/// `[0, M)`, reading the weight once for all rows. Each row follows
/// `dense_gemv_bf16`'s K order and reduction (gb10 kernels).
///
/// `input`: `[M, K]` BF16, contiguous; `output`: M rows at
/// `output + t * out_stride` (BF16 elements). Refuses `m` outside
/// `1..=DENSE_GEMV_BATCHM_MAX_M`.
///
/// Kernel: `dense_gemv_bf16_batchm(A, B, C, M, N, K, out_stride)`
pub fn dense_gemv_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: The kernel clamps M to its MAX_M instead of failing, which
    // would leave the rows past it unwritten.
    ensure!(
        (1..=DENSE_GEMV_BATCHM_MAX_M).contains(&m),
        "dense_gemv_batchm: m={m} outside 1..={DENSE_GEMV_BATCHM_MAX_M} \
         (kernel MAX_M clamps silently; use dense_gemm_tc for wider batches)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// 2026-10-01: [`dense_gemv_batchm`] with FP32 output: `C[t] = A[t] @ B^T` for
/// `t` in `[0, M)`, the FP32 accumulator stored unrounded. Each row is
/// bit-identical to `dense_gemv_bf16_fp32out` on that row (the kernel is the
/// BF16 batchm body with only the store changed).
///
/// `output`: M FP32 rows at `output + t * out_stride` (FP32 elements). Refuses
/// `m` outside `1..=DENSE_GEMV_BATCHM_MAX_M`, like [`dense_gemv_batchm`].
///
/// Kernel: `dense_gemv_bf16_fp32out_batchm(A, B, C, M, N, K, out_stride)`
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batchm_fp32out(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    // 2026-10-01: The launch contract is `dense_gemv_bf16_batchm`'s; only the
    // element type behind `output` (and so the unit of `out_stride`) differs.
    dense_gemv_batchm(gpu, kernel, input, weight, output, m, n, k, out_stride, stream)
}

/// 2026-09-27: [`dense_gemv_batchm`] over `m` rows split into `y_blocks` block
/// rows of `ceil(m / y_blocks)` rows each (at most `DENSE_GEMV_BATCHM_MAX_M`):
/// more blocks for a narrow weight such as the MoE router, and one launch past
/// 16 rows. Each row's result is the same as in any other split.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batchm_split(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    y_blocks: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        m >= 1 && (1..=m).contains(&y_blocks) && m.div_ceil(y_blocks) <= DENSE_GEMV_BATCHM_MAX_M,
        "dense_gemv_batchm_split: m={m} over y_blocks={y_blocks} must give 1..={DENSE_GEMV_BATCHM_MAX_M} rows per block row"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), y_blocks, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// 2026-09-25: FP8-weight GEMV for one row,
/// `C = A @ (dequant(B_fp8) * row_scale)^T`: A `[1, K]` BF16, B `[N, K]` FP8
/// E4M3, row_scale `[N]` f32, C `[1, N]` BF16. One byte per weight, against
/// two for [`dense_gemv`]. Each 256-thread block computes 4 outputs, 64
/// threads per output (kernel `dense_gemv_fp8w(A, B, row_scale, C, N, K)`).
pub fn dense_gemv_fp8w(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-10-03: Rows `dense_gemv_fp8w_batchm` (and its FP32-output twin)
/// computes per block row: `FP8BM_MAX_M` in
/// `kernels/gb10/common/dense_gemv_fp8w_batchm.cu`.
pub const DENSE_GEMV_FP8W_BATCHM_MAX_M: u32 = 16;

/// 2026-10-03: FP8-weight GEMV for M rows in one pass over the weight,
/// `C[t] = A[t] @ (fp8(B) * row_scale)^T` for `t` in `[0, M)`, split over
/// `y_blocks` block rows of `ceil(m / y_blocks)` rows each. Each row's BF16
/// result is bit-identical to [`dense_gemv_fp8w`] on that row, at every M and
/// split (kernel header). `fp32_out` selects the twin that stores the FP32 sum
/// unrounded; `out_stride` counts output elements of that type. Refuses a
/// split giving more than `DENSE_GEMV_FP8W_BATCHM_MAX_M` rows per block row
/// and a `k` that is not a multiple of 16 (the kernel has no K tail).
///
/// Kernels: `dense_gemv_fp8w_batchm` / `dense_gemv_fp8w_fp32out_batchm`
/// `(A, B, row_scale, C, M, N, K, out_stride)`, module
/// `dense_gemv_fp8w_batchm`; dynamic shared memory `rows_per_block * 2048` B.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_fp8w_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    y_blocks: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        m >= 1
            && (1..=m).contains(&y_blocks)
            && m.div_ceil(y_blocks) <= DENSE_GEMV_FP8W_BATCHM_MAX_M,
        "dense_gemv_fp8w_batchm: m={m} over y_blocks={y_blocks} must give \
         1..={DENSE_GEMV_FP8W_BATCHM_MAX_M} rows per block row"
    );
    ensure!(
        k.is_multiple_of(16),
        "dense_gemv_fp8w_batchm: K={k} not a multiple of 16"
    );
    let rows = m.div_ceil(y_blocks);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), y_blocks, 1])
        .block([256, 1, 1])
        .shared_mem(rows * 2048)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}
