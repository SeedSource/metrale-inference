// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The pre-attention call of the paged drafter layer: input_layernorm, DFlash2
//! attention conv prepare, the Q/K/V projections and norms, RoPE, and the block K/V write into
//! the paged cache.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::PagedLayerArgs;
use crate::dflash_head::{BlockDiffusionDraftHead, DflashLayer};
use metrale_model_layers::layer::ForwardContext;

impl BlockDiffusionDraftHead {
    /// 2026-09-26: The body of `forward_block_layer_pre_attn`.
    pub(super) fn pre_attn_layer(
        &self,
        layer: &DflashLayer,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
    ) -> Result<(DevicePtr, DevicePtr)> {
        use metrale_model_layers::layers::ops;

        let PagedLayerArgs {
            layer_idx,
            ctx_count,
            h,
            q_dim,
            kv_dim,
            slot_mapping_gamma,
            stream,
            ..
        } = *args;
        let gpu = ctx.gpu;
        // 2026-09-25: `block_g` is the rows per sequence, `g` the rows in this forward.
        // Weight-bearing ops take `g`; one sequence's attention window takes `block_g`.
        let block_g = self.block_g() as u32;
        let g = block_g * args.n_seq.max(1);
        let kv_len = ctx_count + block_g;

        // 2026-09-25: `stream_buf` holds the layer input, which is also the residual;
        // `norm_buf` gets the normed rows.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            g,
            h,
            self.rms_norm_eps,
            stream,
        )?;

        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.norm_buf,
                layer_idx,
                "input_norm",
                g,
                h,
                stream,
            )?;
        }

        // 2026-09-25: DFlash2 attention conv prepare on the normed rows. Q, K and V are
        // all projected from its output; the finish application's dynamic slice stays
        // in scratch.conv_dyn until post_attn uses it.
        let qkv_src = self.conv_prepare(
            layer,
            crate::dflash_head::dflash2::ConvSite::Attention,
            self.scratch.norm_buf,
            ctx,
            args.n_seq.max(1),
            stream,
        )?;

        // 2026-09-25: With `Fp8Weights` and the weight's FP8 mirror present, a projection
        // runs on the FP8 weight: `fp8_gemv_rt2` up to 8 rows, `fp8_gemv_rt2_16` up to
        // 16 (either only when present, K is a multiple of 16 and
        // `METRALE_NO_DFLASH_FP8_RT` is not `1`), else the row-scaled GEMM. Otherwise
        // the BF16 GEMM runs.
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

        // 2026-09-25: Q is `[g, q_dim]`, one row per token; q_norm treats it as
        // `[g * num_q_heads, head_dim]`.
        gemm_swap(
            &layer.q_proj,
            &layer.q_proj_fp8,
            &layer.nv4.q,
            qkv_src,
            self.scratch.q_buf,
            q_dim,
            h,
        )?;
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.q_buf,
                layer_idx,
                "q_postproj",
                g,
                q_dim,
                stream,
            )?;
        }
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.q_buf,
            &layer.q_norm,
            self.scratch.q_buf,
            g * self.num_q_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.q_buf,
                layer_idx,
                "q_postnorm",
                g,
                q_dim,
                stream,
            )?;
        }

        // 2026-09-25: k_norm normalises each token's heads independently, so norming the
        // block K here and the ctx K in `precompute_ctx_kv` (same `k_norm` weight) equals
        // norming them together.
        gemm_swap(
            &layer.k_proj,
            &layer.k_proj_fp8,
            &layer.nv4.k,
            qkv_src,
            self.scratch.k_buf,
            kv_dim,
            h,
        )?;
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.k_buf,
                layer_idx,
                "k_postproj",
                g,
                kv_dim,
                stream,
            )?;
        }
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.k_buf,
            &layer.k_norm,
            self.scratch.k_buf,
            g * self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.k_buf,
                layer_idx,
                "k_postnorm",
                g,
                kv_dim,
                stream,
            )?;
        }

        // 2026-09-25: V has no norm.
        gemm_swap(
            &layer.v_proj,
            &layer.v_proj_fp8,
            &layer.nv4.v,
            qkv_src,
            self.scratch.v_buf,
            kv_dim,
            h,
        )?;

        // 2026-09-25: RoPE on Q and the block K at `position_ids`, each sequence's
        // `[position, position + block_g)` as `forward_block` wrote them. The ctx K was
        // rotated at its own slot positions in `precompute_ctx_kv`.
        ops::rope_yarn(
            gpu,
            self.kernels.rope_qwen3,
            self.scratch.q_buf,
            self.scratch.k_buf,
            self.scratch.position_ids,
            g,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            stream,
        )?;

        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.q_buf,
                layer_idx,
                "q_postrope",
                g,
                q_dim,
                stream,
            )?;
            self.block_dump_buf(
                ctx,
                self.scratch.k_buf,
                layer_idx,
                "k_postrope",
                g,
                kv_dim,
                stream,
            )?;
            self.block_dump_buf(ctx, self.scratch.v_buf, layer_idx, "v", g, kv_dim, stream)?;
        }

        // 2026-09-25: Write the block K/V into this layer's paged cache at
        // `slot_mapping_gamma`: slots `[ctx_count, ctx_count + block_g)` of each
        // sequence, after its ctx K/V at `[0, ctx_count)`.
        let (k_pool, v_pool) = {
            let cache = self.kv_cache.lock();
            (cache.k_pool_ptr(layer_idx), cache.v_pool_ptr(layer_idx))
        };
        ops::reshape_and_cache(
            gpu,
            self.kernels.reshape_cache_bf16,
            self.scratch.k_buf,
            self.scratch.v_buf,
            k_pool,
            v_pool,
            slot_mapping_gamma,
            g,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            16,
            kv_dim,
            kv_dim,
            0,
            stream,
        )?;

        self.option_b_diag(args, ctx, k_pool, kv_len)?;
        let _ = kv_len;

        Ok((k_pool, v_pool))
    }
}
