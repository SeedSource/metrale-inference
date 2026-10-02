// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Mixed forward (`mixed_forward_dispatch`): the decode rows and one
//! prefill chunk in a single fused layer loop, or, when fusing is not allowed,
//! `decode_batch` followed by `prefill_chunk`.
//!
//! Owner: model-engine (decode).
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::ModelForward;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

mod build_states;
mod embed_chunk;
mod stage_decode;

impl TransformerModel {
    pub(super) fn mixed_forward_dispatch(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_tokens: &[u32],
        prefill_seq: &mut SequenceState,
        prefill_chunk_start: usize,
        prefill_chunk_len: usize,
        prefill_is_last: bool,
        stream: u64,
    ) -> Result<crate::traits::MixedForwardResult> {
        let n_decode = decode_tokens.len();
        let n_prefill = prefill_chunk_len;
        // 2026-09-25: `--ssm-h-dtype f16`: narrow each decode sequence's SSM h-state
        // to FP16 here, before any graph region. No-op without the flag. The
        // prefill sequence is left in FP32.
        for s in decode_seqs.iter_mut() {
            self.ssm_h_to_f16_dispatch(s)?;
        }

        let padded_n_guard = crate::traits::padded_batch_n(n_decode);

        // 2026-09-25: Run `decode_batch` then `prefill_chunk` instead of fusing for
        // EP, MLA (its batched decode is the `decode_batch` route), `hc_qsa_perseq`,
        // a step over `max_batch_tokens` (counting the padded decode rows), or no
        // decode rows. `hc_qsa_perseq` is `decode_a2`'s layer veto plus mHC rows
        // with QSA active, since the fused batched decode has no per-sequence QSA
        // arm. The veto is needed here as well: this is the single-GPU fused
        // caller, and it keeps a declining layer away from `prefill_ctx`, the one
        // context built with a non-zero `hc_row_offset`.
        // 2026-10-01: `fused_decode_prefill_unsupported` joins it (GLM-5.3's prefill numbers
        // highway slots from 0, over the decode rows).
        let ms_layer_veto = self
            .layers
            .iter()
            .any(|l| l.decode_multi_seq_unsupported() || l.fused_decode_prefill_unsupported());
        let hc_qsa_perseq = ms_layer_veto
            || (self.config.hc_mult > 0 && self.config.index_topk > 0 && {
                let bound = self.config.index_topk + self.config.index_compress_ratio - 1;
                decode_seqs.iter().any(|s| s.seq_len >= bound)
            });
        if self.comm.is_some()
            || self.is_mla_dispatch()
            || hc_qsa_perseq
            || (padded_n_guard + n_prefill) > self.buffers.max_batch_tokens()
            || n_decode == 0
        {
            let decode_logits =
                self.mixed_stage_decode_logits(decode_tokens, decode_seqs, n_decode, stream)?;
            // 2026-09-25: The prefill must run after the staging copy, since it
            // writes the same logits arena; one stream orders decode, copy and
            // prefill.
            let prefill_logits = self.prefill_chunk(
                prefill_tokens,
                prefill_seq,
                prefill_chunk_start,
                prefill_chunk_len,
                prefill_is_last,
                self.gpu.default_stream(),
            )?;
            return Ok(crate::traits::MixedForwardResult {
                decode_logits,
                prefill_logits,
            });
        }

        // 2026-09-25: Fused mixed forward, one layer loop. Rows [0, padded_n) of
        // hidden/residual are the decode rows (one per sequence, padding
        // included); rows [padded_n, padded_n + M) are the prefill chunk. Per
        // layer: `decode_multi_seq` on the decode rows, then `prefill` on the
        // chunk, in stream order, so shared scratch can be reused.

        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // 2026-09-25: Pad the decode count to the ladder in `traits::padded_batch_n`.
        let padded_n = crate::traits::padded_batch_n(n_decode);

        for (i, &tok) in decode_tokens.iter().enumerate() {
            // 2026-09-25: Each decode row is a different sequence, so the n-gram
            // context comes from that sequence's own history.
            self.embed_ctx(
                &decode_seqs[i].tokens,
                tok,
                hidden.offset(i * h * fp32),
                stream,
            )?;
        }
        for i in n_decode..padded_n {
            self.gpu.memset(hidden.offset(i * h * fp32), 0, h * fp32)?;
        }
        // 2026-09-25: Prefill chunk tokens into rows [padded_n, padded_n + M),
        // with one batched embed.
        let prefill_hidden = hidden.offset(padded_n * h * fp32);
        let prefill_residual = residual.offset(padded_n * h * fp32);
        self.mixed_embed_prefill_chunk(
            prefill_tokens,
            prefill_chunk_start,
            n_prefill,
            prefill_hidden,
            h,
            stream,
        )?;

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();

        for seq in decode_seqs.iter_mut() {
            let blocks_needed = (seq.seq_len / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )?;
        }

        let prefill_end_pos = prefill_chunk_start + n_prefill;
        let prefill_blocks_needed = (prefill_end_pos - 1) / bs + 1;
        ensure_blocks_through_prefill(
            prefill_seq,
            prefill_blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
            self.levers.kv_poison,
        )?;

        // 2026-09-25: Decode metadata goes in the logits buffer, unused until the
        // LM head, at offset 64 KiB: MoE uses the start of `logits` as shared-gate
        // scratch during the layer loop.
        let decode_meta_base = self.buffers.logits().offset(65536);

        // 2026-09-25: Fail fast if the decode metadata does not fit in the logits
        // arena above that 64 KiB band.
        let meta_lay = self.buffers.decode_meta();
        anyhow::ensure!(
            65536 + meta_lay.meta_bytes(self.max_blocks_per_seq as usize)
                <= self.buffers.sizes().logits,
            "mixed_forward decode metadata ({} B at {} rows) overflows the logits arena ({} B)",
            meta_lay.meta_bytes(self.max_blocks_per_seq as usize),
            meta_lay.rows(),
            self.buffers.sizes().logits
        );

        let decode_metadata = self.upload_batch_metadata_at(
            decode_seqs,
            padded_n,
            &mut kv_cache,
            decode_meta_base,
            stream,
        )?;

        // 2026-09-25: Prefill metadata in scratch, after the MoE routing scratch.
        let proc_start = prefill_chunk_start;
        let proc_count = n_prefill;
        let effective_seq_len_start = prefill_chunk_start;
        let moe_scratch_bytes = proc_count * self.config.num_experts_per_tok * 4 * 2;
        let meta_offset = (moe_scratch_bytes + 7) & !7;
        let prefill_meta_base = self.buffers.scratch().offset(meta_offset);
        let slot_offset = (proc_count * 4 + 7) & !7;
        let needs_paged = effective_seq_len_start > 0;

        {
            // 2026-09-25: SAFETY: single-threaded scheduler access.
            let stg = unsafe { &mut *self.pinned_staging.get() };
            stg.positions.clear();
            stg.positions
                .extend(proc_start as u32..(proc_start + proc_count) as u32);

            if !needs_paged {
                stg.slots.clear();
                stg.slots
                    .extend((proc_start..proc_start + proc_count).map(|i| {
                        let block_idx = prefill_seq
                            .physical_block_for(i / bs)
                            .unwrap_or(self.dummy_kv_block);
                        (block_idx as i64) * (bs as i64) + ((i % bs) as i64)
                    }));
            }

            // 2026-09-25: Rounding `slot_offset` up to 8 leaves up to 4 pad bytes after
            // the positions array that no copy writes; they are still initialised
            // (see the `pinned_pack` module docs).
            let mut pack = stg.packer_for(self.buffers.scratch_bytes().saturating_sub(meta_offset));
            pack.put_prefix_at("positions", 0, &stg.positions, proc_count)?;
            if !needs_paged {
                pack.put_prefix_at("slots", slot_offset, &stg.slots, proc_count)?;
            }
            self.gpu
                .copy_h2d_async_retained(pack.packed(), prefill_meta_base, stream)?;
        }

        if needs_paged {
            let current_blocks = prefill_seq.block_table.len();
            let upload_start = self
                .ensure_chunked_prefill_meta(prefill_seq, prefill_tokens.len(), bs)?
                .uploaded_blocks;
            // 2026-09-25: No block-table upload in high-speed-swap mode
            // (`hss_window_start() != 0`).
            if upload_start < current_blocks && prefill_seq.hss_window_start() == 0 {
                let new_blocks = &prefill_seq.block_table[upload_start..];
                // 2026-09-25: SAFETY: the length is `size_of_val(new_blocks)`, derived
                // from the slice itself, over a live `&[u32]` sub-slice of
                // `prefill_seq.block_table`.
                let bt_bytes = unsafe {
                    std::slice::from_raw_parts(
                        new_blocks.as_ptr() as *const u8,
                        std::mem::size_of_val(new_blocks),
                    )
                };
                let block_table_base = prefill_seq
                    .chunked_prefill_meta
                    .as_ref()
                    .unwrap()
                    .block_table;
                self.gpu.copy_h2d_async(
                    bt_bytes,
                    block_table_base.offset(upload_start * std::mem::size_of::<u32>()),
                    stream,
                )?;
                prefill_seq
                    .chunked_prefill_meta
                    .as_mut()
                    .unwrap()
                    .uploaded_blocks = current_blocks;
            }

            let seq_len_val = (proc_start + proc_count) as u32;
            // 2026-09-25: SAFETY: exactly `size_of::<u32>()` bytes over the live,
            // initialised `seq_len_val` local above.
            let seq_len_bytes = unsafe {
                std::slice::from_raw_parts(
                    &seq_len_val as *const u32 as *const u8,
                    std::mem::size_of::<u32>(),
                )
            };
            let seq_len_base = prefill_seq.chunked_prefill_meta.as_ref().unwrap().seq_len;
            self.gpu
                .copy_h2d_async(seq_len_bytes, seq_len_base, stream)?;

            let block_table_base = prefill_seq
                .chunked_prefill_meta
                .as_ref()
                .unwrap()
                .block_table;
            ops::fill_slots_from_block_table(
                self.gpu.as_ref(),
                self.fill_slots_kernel,
                prefill_meta_base.offset(slot_offset),
                block_table_base,
                proc_start as u32,
                proc_count as u32,
                bs as u32,
                stream,
            )?;
        }

        self.gpu.synchronize(stream)?;

        let (prefill_bt_dev, prefill_sl_dev) = if needs_paged {
            let page_meta = prefill_seq.chunked_prefill_meta.as_ref().unwrap();
            (page_meta.block_table, page_meta.seq_len)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        // 2026-09-25: LoRA routing for the prefill rows: `proc_count` copies of the
        // prefilling sequence's slot in `lora_seq_slot`, apart from the decode rows'
        // slots in their own metadata.
        let prefill_seq_slot = self.upload_seq_slot_uniform(
            prefill_seq.adapter_slot,
            proc_count,
            self.buffers.lora_seq_slot(),
            stream,
        )?;
        let prefill_metadata = AttnMetadataDev {
            positions: prefill_meta_base,
            positions_h: prefill_meta_base,
            positions_w: prefill_meta_base,
            slot: prefill_meta_base.offset(slot_offset),
            seq_len: prefill_sl_dev,
            block_table: prefill_bt_dev,
            max_blocks_per_seq: prefill_seq.block_table.len() as u32,
            num_seqs: 1,
            seq_slot: prefill_seq_slot,
            // 2026-09-25: The prefill rows' MoE LoRA route is `prefill_ctx.moe_lora_route`.
            moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        };

        let (seq_lens, block_tables, mut all_layer_states) =
            self.mixed_build_decode_layer_states(decode_seqs, padded_n, n_decode)?;

        // 2026-09-25: Fused layer loop: per layer, the decode rows, then the
        // prefill chunk.
        let decode_ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(decode_metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            // 2026-09-25: The decode rows' host ids, for PLE's host-side hash.
            host_token_ids: Some(decode_tokens),
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };

        let prefill_ctx = ForwardContext {
            buffers: &self.buffers,
            // 2026-09-25: The chunk's highway rows sit above the padded decode rows,
            // as in hidden/residual.
            hc_row_offset: padded_n,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(prefill_metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            // 2026-09-25: The chunk's ids, for the PLE prefill hash.
            host_token_ids: Some(
                &prefill_tokens[prefill_chunk_start..prefill_chunk_start + prefill_chunk_len],
            ),
            // 2026-09-25: `Some` only when the prefilling sequence routes to a
            // non-active LoRA slot (`routed_slot_layers`).
            routed_lora_layers: self.routed_slot_layers(prefill_seq.adapter_slot),
            midchunk_capture: None,
            moe_lora_route: self.moe_lora_route(prefill_seq.adapter_slot),
        };

        // 2026-09-25: Refuse a decode row routed to a non-active adapter, as
        // `decode_batch_compute_main` does.
        metrale_model_layers::lora::ensure_decode_route_servable(
            decode_ctx.moe_lora_route,
            "mixed_forward decode",
        )?;

        // 2026-09-25: The decode sequences' layer states are out (in
        // `all_layer_states`) while this runs. It is a closure so that they are
        // restored below whatever it returns.
        let fused_body = (|| -> Result<()> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let mut layer_state_refs = extract_layer_refs(&mut all_layer_states, layer_idx);
                layer.decode_multi_seq(
                    hidden,
                    residual,
                    padded_n,
                    &mut layer_state_refs,
                    &mut kv_cache,
                    &seq_lens,
                    &block_tables,
                    &decode_ctx,
                    stream,
                )?;

                layer.prefill(
                    prefill_hidden,
                    prefill_residual,
                    proc_count,
                    prefill_seq.layer_states[layer_idx].as_mut(),
                    &mut kv_cache,
                    effective_seq_len_start,
                    &mut prefill_seq.block_table,
                    &mut prefill_seq.disk_block_ids,
                    &mut prefill_seq.disk_last_offloaded_per_layer,
                    0,
                    &prefill_ctx,
                    stream,
                )?;
            }

            // 2026-09-25: Normalise the prefill sequence's SSM state on `stream`, where
            // the layer loop just wrote it, for every chunk including the last.
            self.normalize_ssm_states_dispatch(prefill_seq, stream)?;

            // 2026-09-25: Drafter prefill capture of this chunk's final-layer hidden
            // rows. The source is `prefill_hidden`: the buffer head holds the
            // decode rows.
            self.try_mtp_prefill_capture_from(
                prefill_seq,
                effective_seq_len_start,
                proc_count,
                prefill_hidden,
                stream,
            )?;
            Ok(())
        })();

        // 2026-09-25: Give the decode sequences their layer states back before the
        // fused-body result is inspected.
        for (seq, ls) in decode_seqs
            .iter_mut()
            .zip(all_layer_states.drain(..n_decode))
        {
            seq.layer_states = ls;
        }
        fused_body?;

        let head_out = self.mixed_final_norm_lm_head(
            hidden,
            prefill_hidden,
            padded_n,
            proc_count,
            prefill_is_last,
            h,
            bf16,
            fp32,
            stream,
        )?;
        let decode_logits = head_out.decode_logits;
        let prefill_logits = head_out.prefill_logits;

        for (i, seq) in decode_seqs.iter_mut().enumerate() {
            seq.tokens.push(decode_tokens[i]);
            seq.seq_len += 1;
        }
        prefill_seq.tokens.extend_from_slice(
            &prefill_tokens[prefill_chunk_start..prefill_chunk_start + n_prefill],
        );
        prefill_seq.seq_len = prefill_chunk_start + n_prefill;

        Ok(crate::traits::MixedForwardResult {
            decode_logits,
            prefill_logits,
        })
    }
}
