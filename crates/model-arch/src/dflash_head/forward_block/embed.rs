// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The block inputs of `forward_block`: position ids, token ids, the embedding of
//! each sequence's last token and mask tokens, and the paged path's cache slots and indirect
//! attention arguments.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layers::ops;

use super::dims::BlockDims;
use crate::dflash_head::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    /// 2026-09-26: Writes the position ids to `scratch.position_ids` and returns them.
    pub(super) fn block_positions(&self, d: &BlockDims<'_>) -> Result<Vec<i32>> {
        let BlockDims {
            gpu,
            n_seq,
            width,
            eff_ctx,
            position,
            batch,
            ..
        } = *d;
        // 2026-09-25: Position ids: the `eff_ctx` ctx rows at `position - eff_ctx + i`,
        // then each sequence's `width` rows from its own position.
        let ctx_start = position.saturating_sub(eff_ctx);
        let pos_host: Vec<i32> = (0..eff_ctx)
            .map(|i| (ctx_start + i) as i32)
            .chain((0..n_seq).flat_map(|b| {
                let base = batch.map_or(position, |x| x.positions[b]);
                (0..width).map(move |i| (base + i) as i32)
            }))
            .collect();
        let pos_bytes: Vec<u8> = pos_host.iter().flat_map(|p| p.to_le_bytes()).collect();
        gpu.copy_h2d(&pos_bytes, self.scratch.position_ids)?;
        Ok(pos_host)
    }

    /// 2026-09-26: Zeroes the ctx rows of `stream_buf` and returns the token ids to embed.
    pub(super) fn block_token_ids(&self, d: &BlockDims<'_>) -> Result<Vec<i32>> {
        let BlockDims {
            gpu,
            n_seq,
            width,
            bf16,
            eff_ctx,
            last_token,
            batch,
            ..
        } = *d;
        // 2026-09-25: Embed each sequence's `[last token, mask × (width - 1)]` into
        // `stream_buf` after the `eff_ctx` ctx rows, which are left zero. The token ids
        // stay in `draft_tokens_dev`, where the DFlash2 walk and the Markov chain read
        // them.
        if eff_ctx > 0 {
            gpu.memset(
                self.scratch.stream_buf,
                0,
                eff_ctx * self.hidden_size * bf16,
            )?;
        }
        let token_ids_host: Vec<i32> = std::iter::repeat_n(0i32, eff_ctx)
            .chain((0..n_seq).flat_map(|b| {
                let anchor = batch.map_or(last_token, |x| x.last_tokens[b]);
                std::iter::once(anchor as i32)
                    .chain(std::iter::repeat_n(self.mask_token_id as i32, width - 1))
            }))
            .collect();
        Ok(token_ids_host)
    }

    /// 2026-09-26: Embeds `token_ids_host` into `stream_buf` and re-zeroes the ctx rows.
    pub(super) fn block_embed(
        &self,
        d: &BlockDims<'_>,
        stream: u64,
        token_ids_host: &[i32],
    ) -> Result<()> {
        let BlockDims {
            gpu,
            n_seq,
            width,
            h,
            bf16,
            levers,
            eff_ctx,
            n_attn,
            ..
        } = *d;
        let tid_bytes: Vec<u8> = token_ids_host
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        gpu.copy_h2d(&tid_bytes, self.scratch.draft_tokens_dev)?;
        ops::batched_embed(
            gpu,
            self.kernels.batched_embed,
            self.scratch.draft_tokens_dev,
            self.embed_tokens_shared,
            self.scratch.stream_buf,
            n_attn,
            h,
            stream,
        )?;
        // 2026-09-25: Re-zero the ctx rows, which batched_embed filled with token 0's
        // embedding.
        if eff_ctx > 0 {
            gpu.memset(
                self.scratch.stream_buf,
                0,
                eff_ctx * self.hidden_size * bf16,
            )?;
        }
        // 2026-10-01: METRALE_DFLASH_CKPT_ARCH=1 with a learned mask embedding
        // (`mask_embedding.pt`): overwrite each sequence's `width - 1` mask rows,
        // which batched_embed filled with the target's `embed_tokens[mask_token_id]`,
        // with the drafter's trained mask vector. `mask_rows` holds
        // `mask_rows_count` (= `gamma - 1`) replicated copies, so one contiguous
        // copy per sequence covers its rows (`width <= gamma`). The anchor row and
        // the ctx rows are untouched.
        if let Some(mask_rows) = self.mask_rows
            && width > 1
        {
            let row_bytes = self.hidden_size * bf16;
            debug_assert!(width - 1 <= self.mask_rows_count);
            let n_rows = (width - 1).min(self.mask_rows_count);
            for b in 0..n_seq {
                gpu.copy_d2d_async(
                    mask_rows,
                    self.scratch
                        .stream_buf
                        .offset((eff_ctx + b * width + 1) * row_bytes),
                    n_rows * row_bytes,
                    stream,
                )?;
            }
        }
        // 2026-09-25: METRALE_DFLASH_DEBUG_FORCE_NOISE_PATTERN=1 overwrites sequence 0's
        // `width` block rows with `0.001 * (t + 1) * (j + 1) / hidden_size`.
        if levers.force_noise_pattern {
            let mut bytes = Vec::with_capacity(width * self.hidden_size * 2);
            for t in 0..width {
                for j in 0..self.hidden_size {
                    let v =
                        0.001_f32 * ((t + 1) as f32) * ((j + 1) as f32) / (self.hidden_size as f32);
                    let bf16_bits = (v.to_bits() >> 16) as u16;
                    bytes.extend_from_slice(&bf16_bits.to_le_bytes());
                }
            }
            gpu.copy_h2d(
                &bytes,
                self.scratch
                    .stream_buf
                    .offset(eff_ctx * self.hidden_size * bf16),
            )?;
        }
        Ok(())
    }

    /// 2026-09-26: On the paged path, the block rows' cache slots and the indirect attention
    /// arguments; returns `scratch.slot_mapping_dev`, or `None` off the paged path.
    pub(super) fn block_paged_slots(
        &self,
        d: &BlockDims<'_>,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let BlockDims {
            gpu,
            n_seq,
            width,
            block_g,
            option_b_block_table,
            option_b_ctx_count,
            option_b_on,
            position,
            batch,
            ..
        } = *d;
        let slot_mapping_gamma_opt = if option_b_on {
            let bt = option_b_block_table.unwrap();
            // 2026-09-25: Each sequence's `width` slots start at its own ctx_count and go
            // through its own block table, packed seq-major, so the one
            // reshape_and_cache over all rows writes every sequence's K/V to its pages.
            for b in 0..n_seq {
                let (bt_b, cc_b) = match batch {
                    Some(x) => (x.block_tables[b], x.ctx_counts[b]),
                    None => (bt, option_b_ctx_count),
                };
                ops::fill_slots_from_block_table(
                    gpu,
                    self.kernels.fill_slots,
                    self.scratch.slot_mapping_dev.offset(b * width * 8),
                    bt_b,
                    cc_b,
                    width as u32,
                    16,
                    stream,
                )?;
            }
            // 2026-09-25: Each sequence's `[kv_len, q_offset, q_rope_pos]` (3 × u32, at
            // byte `b * 12`) for the indirect paged attention, which reads it at kernel
            // entry: `kv_len = ctx_count + width`, `q_offset = ctx_count`, and
            // `q_rope_pos` = the sequence's position.
            let mut indirect_bytes: Vec<u8> = Vec::with_capacity(n_seq * 12);
            for b in 0..n_seq {
                let cc_b = match batch {
                    Some(x) => x.ctx_counts[b],
                    None => option_b_ctx_count,
                };
                let pos_b = batch.map_or(position, |x| x.positions[b]) as u32;
                indirect_bytes.extend_from_slice(&(cc_b + block_g).to_ne_bytes());
                indirect_bytes.extend_from_slice(&cc_b.to_ne_bytes());
                indirect_bytes.extend_from_slice(&pos_b.to_ne_bytes());
            }
            gpu.copy_h2d(&indirect_bytes, self.scratch.option_b_indirect_args_dev)?;
            Some(self.scratch.slot_mapping_dev)
        } else {
            None
        };
        Ok(slot_mapping_gamma_opt)
    }
}
