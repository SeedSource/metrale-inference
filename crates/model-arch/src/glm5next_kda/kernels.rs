// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The kernel handles a KDA block launches (`Glm5NextKdaKernels`).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - `resolve` fails if any entry point other than `dense_gemv_bf16_batchm`,
//!   `kda_recurrent_decode_bf16_smem`, `causal_conv1d_update_l2norm_rows`,
//!   `kda_recurrent_prefill_bf16_smem` and `kda_recurrent_prefill_bf16_pf` is missing; those five
//!   resolve to handle 0 when absent.

use super::*;

#[derive(Clone, Copy)]
pub struct Glm5NextKdaKernels {
    pub gemm: KernelHandle,
    /// 2026-09-25: The M = 1 kernel: `dense_mm_bf16` sends every decode projection here. Measured
    /// 2026-08-28: `dense_gemm_bf16` at M = 1 moved 58 GB/s and took 32% of the GLM decode step.
    pub gemv: KernelHandle,
    /// 2026-09-25: The batched GEMV: `2..=DENSE_GEMV_BATCHM_MAX_M` rows in one pass over the
    /// weight, so a K-row `decode_k` reads each projection weight once. `0` on a target without
    /// it; [`ops::dense_mm_bf16`] then sends those rows to the tile GEMM.
    pub gemv_batchm: KernelHandle,
    pub conv_decode: KernelHandle,
    pub conv_prefill: KernelHandle,
    pub l2: KernelHandle,
    pub gate: KernelHandle,
    pub chunk_prepare: KernelHandle,
    pub chunk_scan: KernelHandle,
    pub recurrent: KernelHandle,
    /// 2026-09-25: 1R+1W sibling of `recurrent`: the decayed state column stays in shared memory
    /// between the two passes. Resolved with `try_kernel`; `0` selects the 2R+2W kernel.
    pub recurrent_smem: KernelHandle,
    /// 2026-10-01: `causal_conv1d_update_l2norm` over K rows of one sequence in one launch, for
    /// the opt-in token loop (`METRALE_GLM_KDA_TOKEN_LOOP=1`). Resolved with `try_kernel`; `0`
    /// keeps the per-row walk.
    pub conv_rows: KernelHandle,
    /// 2026-10-01: `recurrent_smem` over K rows in one launch, the state column kept on chip
    /// across rows. Resolved with `try_kernel`; `0` keeps the per-row walk.
    pub recurrent_rows: KernelHandle,
    /// 2026-10-01: The prefetching twin of `recurrent_rows` (same arguments), launched instead of
    /// it under `METRALE_GLM_KDA_PREFETCH=1`. Resolved with `try_kernel`; `0` keeps
    /// `recurrent_rows`.
    pub recurrent_pf: KernelHandle,
    pub o_norm: KernelHandle,
    pub split_widen: KernelHandle,
    pub sigmoid: KernelHandle,
    pub fill: KernelHandle,
    pub pack: KernelHandle,
}

impl Glm5NextKdaKernels {
    pub const ENTRY_POINTS: usize = 14;

    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            gemv_batchm: metrale_model_layers::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            conv_decode: gpu.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?,
            conv_prefill: gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
            l2: gpu.kernel("norm", "l2_norm_bf16")?,
            gate: gpu.kernel("kda_gate", "kda_gate_bf16")?,
            chunk_prepare: gpu.kernel("kda_chunk", "kda_chunk_prepare")?,
            chunk_scan: gpu.kernel("kda_chunk", "kda_chunk_scan")?,
            recurrent: gpu.kernel("kda_recurrent", "kda_recurrent_decode_bf16")?,
            recurrent_smem: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_recurrent",
                "kda_recurrent_decode_bf16_smem",
            ),
            conv_rows: metrale_model_layers::layers::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_l2norm_rows",
            ),
            recurrent_rows: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_recurrent",
                "kda_recurrent_prefill_bf16_smem",
            ),
            recurrent_pf: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_recurrent",
                "kda_recurrent_prefill_bf16_pf",
            ),
            o_norm: gpu.kernel("kda_layer_ops", "kda_o_norm_gated_bf16")?,
            split_widen: gpu.kernel("kda_layer_ops", "kda_split_widen")?,
            sigmoid: gpu.kernel("kda_layer_ops", "kda_sigmoid_bf16_f32")?,
            fill: gpu.kernel("kda_layer_ops", "kda_fill_f32")?,
            pack: gpu.kernel("kda_layer_ops", "kda_pack_qkv_bf16")?,
        })
    }
}
