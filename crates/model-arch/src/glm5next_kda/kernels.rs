// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The kernel handles a KDA block launches (`Glm5NextKdaKernels`).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - `resolve` fails if any entry point other than `dense_gemv_bf16_batchm`,
//!   `kda_recurrent_decode_bf16_smem`, `causal_conv1d_update_l2norm_rows`,
//!   `kda_recurrent_prefill_bf16_smem` and `kda_recurrent_prefill_bf16_pf` is missing; those five
//!   resolve to handle 0 when absent.
//! - 2026-10-01: The four `kda_chunk_tc` entry points also resolve to handle 0 when absent;
//!   [`Glm5NextKdaKernels::has_chunked_tc`] is false unless all four resolved.
//! - 2026-10-03: The three `kda_flashkda_glue` entry points resolve to handle 0 when absent;
//!   [`Glm5NextKdaKernels::has_flashkda_glue`] is false unless they and `kda_tc_conv_rows` /
//!   `kda_tc_conv_state_tail` resolved.
//! - 2026-10-06: The two `kda_snap_fuse` entry points resolve to handle 0 when absent;
//!   [`Glm5NextKdaKernels::has_snap_fuse`] is false unless both resolved.
//! - 2026-10-07: The two `kda_front_fuse` entry points resolve to handle 0 when absent;
//!   [`Glm5NextKdaKernels::has_front_fuse`] is false unless both resolved.

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
    /// 2026-10-01: The tensor-core chunked prefill (`METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1`,
    /// prefill_tc.rs), kernels/gb10/common/kda_chunk_tc.cu: the row-parallel conv, the conv-state
    /// tail, the per-chunk prepare and the cross-chunk scan. Resolved with `try_kernel`; any `0`
    /// sends the prefill to `decode_k`.
    pub tc_conv: KernelHandle,
    pub tc_conv_tail: KernelHandle,
    pub tc_prepare: KernelHandle,
    pub tc_scan: KernelHandle,
    /// 2026-10-03: The FlashKDA prefill's glue (`METRALE_GLM_KDA_PREFILL_FLASHKDA=1`,
    /// prefill_flashkda.rs), kernels/gb10/common/kda_flashkda_glue.cu: the beta transpose, the
    /// recurrent-state transpose and the BF16-input output norm. Resolved with `try_kernel`; any
    /// `0` keeps the prefill off FlashKDA.
    pub flk_beta_t: KernelHandle,
    pub flk_state_t: KernelHandle,
    pub flk_o_norm: KernelHandle,
    /// 2026-10-06: The out-of-place twins of `conv_decode` and `recurrent_smem` the fused verify
    /// snapshot launches (`METRALE_GLM_KDA_SNAP_FUSE=1`, snap_fuse.rs),
    /// kernels/gb10/common/kda_snap_fuse.cu. Resolved with `try_kernel`; either `0` keeps the
    /// walk with its snapshot copies.
    pub conv_io: KernelHandle,
    pub recurrent_io: KernelHandle,
    /// 2026-10-07: The fused front of the chunked-TC prefill (`METRALE_GLM_KDA_FRONT_FUSE=1`,
    /// prefill_tc_fuse.rs), kernels/gb10/common/kda_front_fuse.cu: `tc_prepare` computing its
    /// inputs from the projections (pack, conv, gate and beta sigmoid in registers) and the
    /// conv-state tail read from the projections. Resolved with `try_kernel`; either `0` keeps
    /// the unfused launches.
    pub ff_prepare: KernelHandle,
    pub ff_tail: KernelHandle,
    pub o_norm: KernelHandle,
    /// 2026-10-07: `kda_o_norm_gated_bf16_fp8q` and `kda_o_norm_gated_bf16in_fp8q`
    /// (kernels/gb10/common/kda_layer_ops.cu): `o_norm` and `flk_o_norm` that also write the FP8
    /// bytes and scales `per_token_group_quant_fp8` would make from their BF16 output
    /// (`METRALE_GLM_NORM_FP8_QUANT_FUSE=1`). Resolved with `try_kernel`; `0` keeps the unfused
    /// norm + quant.
    pub o_norm_fp8q: KernelHandle,
    pub flk_o_norm_fp8q: KernelHandle,
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
            tc_conv: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_chunk_tc",
                "kda_tc_conv_rows",
            ),
            tc_conv_tail: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_chunk_tc",
                "kda_tc_conv_state_tail",
            ),
            tc_prepare: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_chunk_tc",
                "kda_tc_prepare",
            ),
            tc_scan: metrale_model_layers::layers::try_kernel(gpu, "kda_chunk_tc", "kda_tc_scan"),
            flk_beta_t: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_flashkda_glue",
                "kda_flk_beta_t",
            ),
            flk_state_t: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_flashkda_glue",
                "kda_flk_state_t",
            ),
            flk_o_norm: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_flashkda_glue",
                "kda_o_norm_gated_bf16in",
            ),
            conv_io: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_snap_fuse",
                "causal_conv1d_update_l2norm_io",
            ),
            recurrent_io: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_snap_fuse",
                "kda_recurrent_decode_bf16_smem_io",
            ),
            ff_prepare: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_front_fuse",
                "kda_ff_prepare",
            ),
            ff_tail: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_front_fuse",
                "kda_ff_conv_state_tail",
            ),
            o_norm: gpu.kernel("kda_layer_ops", "kda_o_norm_gated_bf16")?,
            o_norm_fp8q: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_layer_ops",
                "kda_o_norm_gated_bf16_fp8q",
            ),
            flk_o_norm_fp8q: metrale_model_layers::layers::try_kernel(
                gpu,
                "kda_layer_ops",
                "kda_o_norm_gated_bf16in_fp8q",
            ),
            split_widen: gpu.kernel("kda_layer_ops", "kda_split_widen")?,
            sigmoid: gpu.kernel("kda_layer_ops", "kda_sigmoid_bf16_f32")?,
            fill: gpu.kernel("kda_layer_ops", "kda_fill_f32")?,
            pack: gpu.kernel("kda_layer_ops", "kda_pack_qkv_bf16")?,
        })
    }

    /// 2026-10-01: Whether all four tensor-core chunked-prefill kernels resolved.
    pub fn has_chunked_tc(&self) -> bool {
        [
            self.tc_conv,
            self.tc_conv_tail,
            self.tc_prepare,
            self.tc_scan,
        ]
        .iter()
        .all(|h| h.0 != 0)
    }

    /// 2026-10-03: Whether every kernel the FlashKDA prefill launches around the library
    /// resolved: the row-parallel conv and its state tail (`kda_chunk_tc`) and the three
    /// `kda_flashkda_glue` kernels.
    pub fn has_flashkda_glue(&self) -> bool {
        [
            self.tc_conv,
            self.tc_conv_tail,
            self.flk_beta_t,
            self.flk_state_t,
            self.flk_o_norm,
        ]
        .iter()
        .all(|h| h.0 != 0)
    }

    /// 2026-10-06: Whether both `kda_snap_fuse` kernels resolved.
    pub fn has_snap_fuse(&self) -> bool {
        self.conv_io.0 != 0 && self.recurrent_io.0 != 0
    }

    /// 2026-10-07: Whether both `kda_front_fuse` kernels resolved.
    pub fn has_front_fuse(&self) -> bool {
        self.ff_prepare.0 != 0 && self.ff_tail.0 != 0
    }
}
