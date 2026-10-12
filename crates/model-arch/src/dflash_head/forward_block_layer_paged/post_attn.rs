// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The post-attention call of the paged drafter layer: o_proj, the attention
//! residual, post_attention_layernorm and the SwiGLU MLP, with the DFlash2 conv applications.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;

use super::PagedLayerArgs;
use crate::dflash_head::{BlockDiffusionDraftHead, DflashLayer};
use metrale_model_layers::layer::ForwardContext;

/// 2026-10-12: The three pieces of the post-attention call, split at its two row-parallel
/// all-reduces (`tp_reduce_rows`), so `METRALE_DFLASH_TP_GRAPHS` can capture each piece while
/// every collective stays eager. `post_attn_layer` runs them in this order with the
/// all-reduces between them, launch for launch what the single body issued before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::dflash_head) enum PostAttnPart {
    /// 2026-10-12: o_proj into `stream_acc` (the first all-reduce follows).
    OProj,
    /// 2026-10-12: attention conv finish, residual add, post_attention_layernorm, MLP conv
    /// prepare, gate/up, SiLU-mul and down_proj into `stream_acc` (the second all-reduce
    /// follows).
    Mlp,
    /// 2026-10-12: MLP conv finish, residual add, and the per-layer block dump when armed.
    Out,
}

impl BlockDiffusionDraftHead {
    /// 2026-09-26: The body of `forward_block_layer_post_attn`.
    pub(super) fn post_attn_layer(
        &self,
        layer: &DflashLayer,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let g = self.block_g() as u32 * args.n_seq.max(1);
        self.post_attn_part(layer, args, ctx, PostAttnPart::OProj)?;
        // 2026-10-09: Two-rank drafter: o_proj is row-parallel, so sum the partials.
        self.tp_reduce_rows(ctx, self.scratch.stream_acc, g, args.stream)?;
        self.post_attn_part(layer, args, ctx, PostAttnPart::Mlp)?;
        // 2026-10-09: Two-rank drafter: down_proj is row-parallel.
        self.tp_reduce_rows(ctx, self.scratch.stream_acc, g, args.stream)?;
        self.post_attn_part(layer, args, ctx, PostAttnPart::Out)
    }

    /// 2026-10-12: One piece of the post-attention call (see [`PostAttnPart`]). Device
    /// launches only, except the block dump of `Out`, whose lever keeps capture off.
    pub(in crate::dflash_head) fn post_attn_part(
        &self,
        layer: &DflashLayer,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
        part: PostAttnPart,
    ) -> Result<()> {
        use metrale_model_layers::layers::ops;

        let PagedLayerArgs {
            h,
            q_dim,
            inter,
            stream,
            ..
        } = *args;
        let gpu = ctx.gpu;
        // 2026-09-25: All rows: o_proj and the MLP are weight-bearing, so they span every
        // sequence in the batch.
        let g = self.block_g() as u32 * args.n_seq.max(1);

        // 2026-09-25: The same FP8/BF16 GEMM choice as `forward_block_layer_pre_attn`.
        let use_fp8 = matches!(
            self.quant,
            crate::dflash_head::DflashQuantization::Fp8Weights
        );
        let gemm_swap = |w_bf16: &metrale_model_layers::weight_map::DenseWeight,
                         w_fp8: &Option<metrale_model_layers::weight_map::Fp8DenseWeight>,
                         w_nv4: &Option<metrale_model_layers::weight_map::QuantizedWeight>,
                         src: metrale_gpu_runtime::gpu::DevicePtr,
                         dst: metrale_gpu_runtime::gpu::DevicePtr,
                         n_out: u32,
                         k_in: u32|
         -> Result<()> {
            // 2026-10-12: METRALE_DFLASH_NV4_LAYERS (`nv4.rs`): the NVFP4 copy when built (its
            // FP8 mirror was not), on the CUDA-core tiers.
            if let Some(q) = w_nv4 {
                return crate::dflash_head::nv4::gemv(gpu, src, q, dst, g, n_out, k_in, stream);
            }
            if use_fp8 && let Some(fp8) = w_fp8 {
                if self.kernels.fp8_gemv_rt2.0 != 0
                    && g <= 8
                    && k_in.is_multiple_of(16)
                    && crate::dflash_head::fp8_rt_enabled()
                {
                    return ops::fp8_gemv_rowscale_batch8_rt2(
                        gpu,
                        self.kernels.fp8_gemv_rt2,
                        src,
                        fp8,
                        dst,
                        g,
                        n_out,
                        k_in,
                        stream,
                    );
                }
                if self.kernels.fp8_gemv_rt2_16.0 != 0
                    && g <= 16
                    && k_in.is_multiple_of(16)
                    && crate::dflash_head::fp8_rt_enabled()
                {
                    return ops::fp8_gemv_rowscale_batch16_rt2(
                        gpu,
                        self.kernels.fp8_gemv_rt2_16,
                        src,
                        fp8,
                        dst,
                        g,
                        n_out,
                        k_in,
                        stream,
                    );
                }
                return ops::fp8_gemm_n128_row_scaled(
                    gpu,
                    self.kernels.fp8_gemm_n128_row_scaled,
                    src,
                    fp8,
                    dst,
                    g,
                    n_out,
                    k_in,
                    stream,
                );
            }
            // 2026-10-08: BF16: `draft_gemm_bf16` (METRALE_DFLASH_DRAFT_GEMV).
            self.draft_gemm_bf16(gpu, src, w_bf16, dst, g, n_out, k_in, stream)
        };

        match part {
            PostAttnPart::OProj => {
                // 2026-09-25: stream_acc = o_proj(attn_out). stream_buf still holds the
                // layer input: pre_attn read it into norm_buf and did not write it.
                gemm_swap(
                    &layer.o_proj,
                    &layer.o_proj_fp8,
                    &layer.nv4.o,
                    self.scratch.attn_out,
                    self.scratch.stream_acc,
                    h,
                    q_dim,
                )
            }
            PostAttnPart::Mlp => {
                // 2026-09-25: DFlash2 attention conv finish on the o_proj output, with the
                // dynamic slice computed in pre_attn.
                let attn_res_src = self.conv_finish(
                    layer,
                    crate::dflash_head::dflash2::ConvSite::Attention,
                    self.scratch.stream_acc,
                    ctx,
                    args.n_seq.max(1),
                    stream,
                )?;

                // 2026-09-25: stream_buf += attn_res_src (the o_proj output, convolved
                // when DFlash2 is active).
                ops::residual_add(
                    gpu,
                    self.kernels.residual_add,
                    self.scratch.stream_buf,
                    attn_res_src,
                    g * h,
                    stream,
                )?;

                // 2026-09-25: stream_buf is both the MLP's residual and the input to
                // post_attention_layernorm, whose output goes to norm_buf.
                ops::rms_norm(
                    gpu,
                    self.kernels.rms_norm,
                    self.scratch.stream_buf,
                    &layer.post_attention_layernorm,
                    self.scratch.norm_buf,
                    g,
                    h,
                    self.rms_norm_eps,
                    stream,
                )?;

                // 2026-09-25: DFlash2 MLP conv prepare. It overwrites scratch.conv_dyn,
                // which the attention conv finish above has already read.
                let mlp_src = self.conv_prepare(
                    layer,
                    crate::dflash_head::dflash2::ConvSite::Mlp,
                    self.scratch.norm_buf,
                    ctx,
                    args.n_seq.max(1),
                    stream,
                )?;

                // 2026-09-25: MLP: down(silu(gate(x)) * up(x)) with x = mlp_src.
                // silu_mul writes into mlp_intermediate, down_proj into stream_acc.
                gemm_swap(
                    &layer.gate_proj,
                    &layer.gate_proj_fp8,
                    &layer.nv4.gate,
                    mlp_src,
                    self.scratch.mlp_intermediate,
                    inter,
                    h,
                )?;
                gemm_swap(
                    &layer.up_proj,
                    &layer.up_proj_fp8,
                    &layer.nv4.up,
                    mlp_src,
                    self.scratch.mlp_up,
                    inter,
                    h,
                )?;
                ops::silu_mul(
                    gpu,
                    self.kernels.silu_mul,
                    self.scratch.mlp_intermediate,
                    self.scratch.mlp_up,
                    self.scratch.mlp_intermediate,
                    g * inter,
                    stream,
                )?;
                gemm_swap(
                    &layer.down_proj,
                    &layer.down_proj_fp8,
                    &layer.nv4.down,
                    self.scratch.mlp_intermediate,
                    self.scratch.stream_acc,
                    h,
                    inter,
                )
            }
            PostAttnPart::Out => {
                // 2026-09-25: DFlash2 MLP conv finish on the down_proj output.
                let mlp_res_src = self.conv_finish(
                    layer,
                    crate::dflash_head::dflash2::ConvSite::Mlp,
                    self.scratch.stream_acc,
                    ctx,
                    args.n_seq.max(1),
                    stream,
                )?;

                // 2026-09-25: stream_buf += mlp_res_src. stream_buf is then the layer
                // output, read by the next layer or the tail.
                ops::residual_add(
                    gpu,
                    self.kernels.residual_add,
                    self.scratch.stream_buf,
                    mlp_res_src,
                    g * h,
                    stream,
                )?;

                if args.block_dump {
                    self.block_dump_buf(
                        ctx,
                        self.scratch.stream_buf,
                        args.layer_idx,
                        "layer_out",
                        g,
                        h,
                        stream,
                    )?;
                }
                Ok(())
            }
        }
    }
}
