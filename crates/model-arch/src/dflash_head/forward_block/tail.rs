// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The tail of `forward_block`: the final RMSNorm, lm_head and token selection, and
//! the host-side parts of the one-shot block dumps.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::layers::ops;

use super::dims::BlockDims;
use super::rt2_16_covers_the_batch;
use crate::dflash_head::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    /// 2026-09-26: Final RMSNorm of the block rows, lm_head and token selection.
    pub(super) fn block_tail_select(
        &self,
        d: &BlockDims<'_>,
        ctx: &ForwardContext,
        stream: u64,
        stream_noise_local: DevicePtr,
        norm_noise_local: DevicePtr,
    ) -> Result<()> {
        let BlockDims {
            gpu,
            n_seq,
            width,
            g,
            h: h_local,
            bf16: bf16_local,
            ..
        } = *d;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            stream_noise_local,
            &self.norm,
            norm_noise_local,
            g,
            h_local,
            self.rms_norm_eps,
            stream,
        )?;
        let lm_head_fp8 = matches!(
            self.quant,
            crate::dflash_head::DflashQuantization::Fp8Weights
        );
        // 2026-09-25: lm_head: NVFP4 when the target's lm_head is NVFP4
        // (`lm_head_shared` is valid only for a BF16 lm_head), else FP8 with
        // `Fp8Weights` and an FP8 copy present, else BF16.
        if let Some(q) = self.lm_head_nvfp4.as_ref() {
            ops::w4a16_gemm(
                gpu,
                self.kernels.w4a16_gemm,
                norm_noise_local,
                q,
                self.scratch.logits,
                self.gamma as u32,
                self.vocab_size as u32,
                h_local,
                stream,
            )?;
        } else if lm_head_fp8 {
            if let Some(fp8) = self.lm_head_shared_fp8.as_ref() {
                // 2026-09-25: FP8 kernel order: `fp8_gemv_rt2` up to 8 rows,
                // `fp8_gemv_rt2_16` up to 16, then a row-scaled GEMM.
                if self.kernels.fp8_gemv_rt2.0 != 0
                    && g <= 8
                    && h_local.is_multiple_of(16)
                    && crate::dflash_head::fp8_rt_enabled()
                {
                    ops::fp8_gemv_rowscale_batch8_rt2(
                        gpu,
                        self.kernels.fp8_gemv_rt2,
                        norm_noise_local,
                        fp8,
                        self.scratch.logits,
                        g,
                        self.vocab_size as u32,
                        h_local,
                        stream,
                    )?;
                } else if self.kernels.fp8_gemv_rt2_16.0 != 0
                    && rt2_16_covers_the_batch(g)
                    && h_local.is_multiple_of(16)
                    && crate::dflash_head::fp8_rt_enabled()
                {
                    ops::fp8_gemv_rowscale_batch16_rt2(
                        gpu,
                        self.kernels.fp8_gemv_rt2_16,
                        norm_noise_local,
                        fp8,
                        self.scratch.logits,
                        g,
                        self.vocab_size as u32,
                        h_local,
                        stream,
                    )?;
                } else {
                    // 2026-09-25: `fp8_gemm_t_row_scaled_m16` computes at most 16 rows
                    // (`ops::fp8_gemm_n128_row_scaled_m16`), so above 16 rows take
                    // the general row-scaled GEMM.
                    let (k_lm, h_lm) = if g <= 16 {
                        (
                            self.kernels.fp8_gemm_n128_row_scaled_m16,
                            ops::fp8_gemm_n128_row_scaled_m16
                                as fn(_, _, _, _, _, _, _, _, _) -> Result<()>,
                        )
                    } else {
                        (
                            self.kernels.fp8_gemm_n128_row_scaled,
                            ops::fp8_gemm_n128_row_scaled
                                as fn(_, _, _, _, _, _, _, _, _) -> Result<()>,
                        )
                    };
                    h_lm(
                        gpu,
                        k_lm,
                        norm_noise_local,
                        fp8,
                        self.scratch.logits,
                        g,
                        self.vocab_size as u32,
                        h_local,
                        stream,
                    )?;
                }
            } else {
                self.draft_gemm_bf16(
                    gpu,
                    norm_noise_local,
                    &metrale_model_layers::weight_map::DenseWeight {
                        weight: self.lm_head_shared,
                    },
                    self.scratch.logits,
                    g,
                    self.vocab_size as u32,
                    h_local,
                    stream,
                )?;
            }
        } else {
            self.draft_gemm_bf16(
                gpu,
                norm_noise_local,
                &metrale_model_layers::weight_map::DenseWeight {
                    weight: self.lm_head_shared,
                },
                self.scratch.logits,
                g,
                self.vocab_size as u32,
                h_local,
                stream,
            )?;
        }
        // 2026-09-25: DFlash2: per-row top-16 and the chain walk (dflash2.rs), device
        // work only, so it captures with the tail. Row 0 of each band is not
        // written; propose drops it.
        if self.dflash2_active() {
            self.dflash2_select_block(ctx, norm_noise_local, n_seq as u32, stream)?;
        } else
        // 2026-09-25: DSpark: with a Markov head (`markov_active`),
        // `markov_argmax_block` (markov.rs) picks the tokens left to right with the
        // Markov bias; otherwise each row takes its argmax.
        if self.markov_active() {
            self.markov_argmax_block(ctx, norm_noise_local, stream)?;
        } else {
            for i in 0..width {
                let logits_row = self.scratch.logits.offset(i * self.vocab_size * bf16_local);
                let token_slot = self.scratch.draft_tokens_dev.offset(i * 4);
                ops::argmax_bf16(
                    gpu,
                    self.kernels.argmax,
                    logits_row,
                    token_slot,
                    self.vocab_size as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// 2026-09-26: The block dump's drafted tokens of the first `width` rows and its JSON.
    pub(super) fn block_dump_drafts_meta(&self, d: &BlockDims<'_>) -> Result<(Vec<u32>, String)> {
        let BlockDims {
            gpu,
            width,
            last_token,
            position,
            ..
        } = *d;
        let mut dbuf = vec![0u8; width * 4];
        gpu.copy_d2h(self.scratch.draft_tokens_dev, &mut dbuf)?;
        let drafts: Vec<u32> = (0..width)
            .map(|i| {
                u32::from_le_bytes([
                    dbuf[i * 4],
                    dbuf[i * 4 + 1],
                    dbuf[i * 4 + 2],
                    dbuf[i * 4 + 3],
                ])
            })
            .collect();
        let meta = format!(
            "{{\"drafts\":{:?},\"last_token\":{},\"position\":{},\"gamma\":{},\"vocab_size\":{},\"hidden_size\":{},\"mask_token_id\":{},\"num_drafter_layers\":{},\"target_hidden_size\":{},\"n_target_layers\":{},\"rope_theta\":{}}}",
            drafts,
            last_token,
            position,
            width,
            self.vocab_size,
            self.hidden_size,
            self.mask_token_id,
            self.num_layers,
            self.target_hidden_size,
            self.target_layer_ids.len(),
            self.rope_theta,
        );
        Ok((drafts, meta))
    }

    /// 2026-09-26: The block-input dump's `(kv_len, q_offset, JSON)`.
    pub(super) fn block_input_meta(&self, d: &BlockDims<'_>) -> (u32, u32, String) {
        let BlockDims {
            width,
            eff_ctx,
            option_b,
            position,
            ..
        } = *d;
        let (kv_len_dump, q_offset_dump) = match option_b {
            Some((_, cc)) => (cc + width as u32, cc),
            None => (eff_ctx as u32 + width as u32, eff_ctx as u32),
        };
        let q_rope_pos_dump = position as u32;
        let input_meta = format!(
            "{{\"eff_ctx\":{},\"gamma\":{},\"hidden_size\":{},\"option_b_kv_len\":{},\"option_b_q_offset\":{},\"q_rope_pos\":{},\"q_block_positions\":{:?}}}",
            eff_ctx,
            width,
            self.hidden_size,
            kv_len_dump,
            q_offset_dump,
            q_rope_pos_dump,
            (0..width)
                .map(|r| q_rope_pos_dump as usize + r)
                .collect::<Vec<_>>(),
        );
        (kv_len_dump, q_offset_dump, input_meta)
    }
}
