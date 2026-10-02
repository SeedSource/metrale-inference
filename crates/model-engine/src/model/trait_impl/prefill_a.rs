// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-pass (non-chunked) prefill, `prefill_dispatch`.
//!
//! Owner: model-engine.
//! Invariants:
//! - A prompt of at most one token runs through `decode`, not the single-pass forward.
//! - A prompt longer than `buffers.max_batch_tokens()` is refused before any buffer or sequence state is touched.

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

mod vision;
mod vision_sync;

impl TransformerModel {
    pub(super) fn prefill_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = tokens.len();
        if n <= 1 {
            for &token in tokens {
                self.decode(token, seq, stream)?;
            }
            return Ok(self.decode_logits_ptr());
        }

        let arena_cap = self.buffers.max_batch_tokens();
        if n > arena_cap {
            anyhow::bail!(
                "Prompt ({n} tokens) exceeds buffer arena capacity ({arena_cap} tokens). \
                 Use chunked prefill (--max-prefill-tokens) or reduce prompt length."
            );
        }

        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let _bf16 = 2usize;
        let fp32 = 2usize;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        self.buffers.zero_all(self.gpu.as_ref(), stream)?;

        let mut kv_cache = self.kv_cache.lock();

        // 2026-09-25: 1. Prefix-cache lookup, before embedding, because a Marconi
        // hit skips tokens. A prompt with vision-pad tokens gets an empty match.
        let bs = kv_cache.block_size();
        let prefix_match = if self.tokens_have_vision_pad(tokens) {
            metrale_telemetry::prefix_cache::PrefixMatch::empty()
        } else {
            self.prefix_cache
                .lookup(tokens, bs, seq.session_hash, seq.adapter_id)
        };
        let mut kv_write_start = prefix_match.matched_tokens;
        seq.cached_prefix_tokens = prefix_match.matched_tokens;
        seq.cached_prefix_blocks = prefix_match.matched_blocks.len();
        // 2026-09-25: `cache_sequence` passes `prompt_len` as the matched length so
        // the prompt blocks are not ref-bumped a second time.
        seq.prompt_len = n;

        for &block_idx in &prefix_match.matched_blocks {
            kv_cache.inc_ref(block_idx);
            seq.block_table.push(block_idx);
        }
        reuse_prefix_match_disk_ids(
            &prefix_match.matched_disk_block_ids,
            &mut seq.disk_block_ids,
        );

        let blocks_needed = (n - 1) / bs + 1;
        if let Some(cap) = kv_cache.config().cache_blocks_per_seq
            && blocks_needed > cap as usize
        {
            anyhow::bail!(
                "high-speed-swap: prompt of {} blocks exceeds \
                     --high-speed-swap-cache-blocks-per-seq={}; this single-shot \
                     prefill path requires the whole prompt fit in HBM. Use \
                     chunked prefill (set --max-prefill-tokens ≤ {} × block_size) \
                     to stream long prompts to disk.",
                blocks_needed,
                cap,
                cap
            );
        }
        ensure_blocks_through_prefill(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
            self.levers.kv_poison,
        )?;

        // 2026-09-25: Marconi: restore an SSM snapshot and skip the cached prefix.
        // The snapshot can end before the KV match does, so its depth
        // (`eff_snapshot_tokens`) is the skip point. `eff_ssm_snapshot` falls
        // back to a spilled snapshot faulted back in when no resident one
        // matches. Only a snapshot of this session is restored.
        let (eff_snapshot, eff_snapshot_tokens) =
            self.eff_ssm_snapshot(&prefix_match, seq.session_hash, stream);
        let marconi_skip = if let Some(snap_id) = eff_snapshot {
            let snap_tok = eff_snapshot_tokens;
            // 2026-09-25: A snapshot shallower than `marconi_min_tokens()` is not
            // restored; that function's doc holds the measurement behind the floor.
            if snap_tok >= metrale_model_layers::mtp_carry::marconi_min_tokens()
                && snap_tok > 0
                && kv_write_start <= n
                && self
                    .ssm_snapshots
                    .session_matches(snap_id, seq.session_hash)
                // 2026-09-25: A model whose layers carry aux state declines a
                // snapshot that has no aux blobs.
                && (!self.requires_aux_state() || self.ssm_snapshots.has_aux(snap_id))
            {
                self.ssm_snapshots.restore(
                    snap_id,
                    seq.slot_idx,
                    &self.ssm_pool,
                    self.gpu.as_ref(),
                    stream,
                )?;
                if let Some(aux) = self.ssm_snapshots.aux(snap_id) {
                    self.apply_aux_states(seq, &aux, stream)?;
                }
                if snap_tok < kv_write_start {
                    tracing::info!(
                        "Marconi intermediate hit: restored from checkpoint at token {} \
                         (skipping {} tokens, recomputing {} SSM tokens to match point {})",
                        snap_tok,
                        snap_tok,
                        kv_write_start - snap_tok,
                        kv_write_start,
                    );
                } else {
                    tracing::info!(
                        "Marconi SSM cache hit: {} tokens skipped ({} blocks), snapshot {}",
                        kv_write_start,
                        prefix_match.matched_blocks.len(),
                        snap_id,
                    );
                }
                // 2026-09-25: Skip to `n` only when the whole prompt matched and
                // the snapshot covers the match. Otherwise the restored SSM state
                // is at `snap_tok`, so skip only to `snap_tok` and recompute from
                // there; skipping further would put the SSM state behind the KV
                // and positions.
                kv_write_start = if kv_write_start >= n && snap_tok >= kv_write_start {
                    n
                } else {
                    snap_tok
                };
                true
            } else {
                if kv_write_start > 0 {
                    tracing::info!(
                        "Prefix cache hit: {} tokens ({} blocks) reused (KV only)",
                        kv_write_start,
                        prefix_match.matched_blocks.len(),
                    );
                }
                false
            }
        } else {
            let has_ssm_layers = self.config.num_ssm_layers() > 0;
            if kv_write_start > 0 && has_ssm_layers {
                // 2026-09-25: Without a snapshot the SSM state is recomputed from
                // token 0, so the whole prompt is processed and its KV rewritten.
                tracing::info!(
                    "Prefix cache hit: {} tokens ({} blocks) but no SSM snapshot — recomputing all KV",
                    kv_write_start,
                    prefix_match.matched_blocks.len(),
                );
                kv_write_start = 0;
                false
            } else if kv_write_start > 0 && kv_write_start < n {
                // 2026-09-25: A model without SSM layers needs only the cached KV,
                // so only the uncached suffix is embedded and forwarded.
                tracing::info!(
                    "Prefix cache hit: {} tokens ({} blocks) reused, processing {} new tokens (no SSM in this model)",
                    kv_write_start,
                    prefix_match.matched_blocks.len(),
                    n - kv_write_start,
                );
                true
            } else {
                false
            }
        };

        // 2026-09-25: Reports the KV actually reused, not the lookup's match. The
        // SSM-without-snapshot arm above has already set `kv_write_start` to 0.
        seq.reused_prefix_tokens = crate::model::trait_impl::prefix_reuse::reused_prefix_tokens(
            prefix_match.matched_tokens,
            kv_write_start,
            marconi_skip,
        );

        let (proc_tokens, proc_count, seq_len_start) = if marconi_skip && kv_write_start >= n {
            // 2026-09-25: The whole prompt is covered: process only the last
            // token, to produce its logits.
            (&tokens[n - 1..], 1, n - 1)
        } else if marconi_skip {
            (
                &tokens[kv_write_start..],
                n - kv_write_start,
                kv_write_start,
            )
        } else {
            (tokens, n, 0usize)
        };

        // 2026-09-25: 2. Embed the processed tokens into `[proc_count, H]`.
        {
            // 2026-09-25: SAFETY: `proc_count` is not an independent count. Each of the
            // three arms of the `(proc_tokens, proc_count, seq_len_start)`
            // binding above pairs a subslice of `tokens` with that subslice's
            // OWN length — `&tokens[n-1..]`/1, `&tokens[kv_write_start..]`/
            // `n - kv_write_start`, `tokens`/`n` — so
            // `proc_count == proc_tokens.len()` on every path and the byte
            // length is exactly `proc_tokens.len() * size_of::<u32>()`. The
            // bytes are initialised: `tokens` is a live `&[u32]`.
            let token_ids_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(proc_tokens.as_ptr() as *const u8, proc_count * 4)
            };
            let token_ids_dev = self.buffers.scratch();
            self.gpu
                .copy_h2d_async(token_ids_bytes, token_ids_dev, stream)?;
            // 2026-09-25: Also stage the token IDs in `buffers.token_ids()`, which
            // stays stable across the layer loop while scratch holds MoE routing
            // data. Hash-MoE layers read `tid2eid[token_id]` from it, in this order.
            self.gpu
                .copy_h2d_async(token_ids_bytes, self.buffers.token_ids(), stream)?;
            // 2026-09-25: `proc_tokens` is always a suffix of `tokens` (all three arms of
            // the binding above slice from the tail), so the tokens preceding
            // it are exactly `tokens[..start]` — which is what the n-gram hash
            // needs to read backwards into. `ngram_lookbehind()` is 0 for
            // models without one, making `ctx` just the processed tokens.
            let start = tokens.len() - proc_count;
            let ctx_start = start.saturating_sub(self.ngram_lookbehind());
            self.embed_tokens_fused(
                &tokens[ctx_start..start + proc_count],
                proc_count,
                hidden,
                stream,
            )?;
            self.scale_embeddings(hidden, proc_count, stream)?;
        }

        // 2026-09-25: 3. Upload the attention metadata through pinned staging, in
        // one H2D copy.
        let moe_scratch_bytes = proc_count * self.config.num_experts_per_tok * 4 * 2;
        let meta_offset = (moe_scratch_bytes + 7) & !7;
        let meta_base = self.buffers.scratch().offset(meta_offset);

        let slot_offset = (proc_count * 4 + 7) & !7;

        let (block_table_dev, seq_len_dev) = {
            // 2026-09-25: SAFETY: Single-threaded scheduler access (see the
            // `unsafe impl Send/Sync for TransformerModel` notes in types.rs).
            let stg = unsafe { &mut *self.pinned_staging.get() };
            stg.positions.clear();
            stg.positions
                .extend(seq_len_start as u32..(seq_len_start + proc_count) as u32);
            stg.slots.clear();
            stg.slots
                .extend((seq_len_start..seq_len_start + proc_count).map(|i| {
                    let block_idx = seq
                        .physical_block_for(i / bs)
                        .unwrap_or(self.dummy_kv_block);
                    (block_idx as i64) * (bs as i64) + ((i % bs) as i64)
                }));

            // 2026-09-25: Each field is bounds-checked before it is written; the
            // packer returns an error instead of overrunning.
            //
            // Rounding `slot_offset` up to 8 leaves up to 4 pad bytes after the
            // positions array that no copy writes; they are still initialised
            // (see the `pinned_pack` module docs). `slot_offset + proc_count * 8`
            // is a multiple of 8 and `block_table` holds u32s, so `bt_start` and
            // `sl_start` round to their inputs and open no further gap.
            let mut pack = stg.packer_for(self.buffers.scratch_bytes().saturating_sub(meta_offset));
            pack.put_prefix_at("positions", 0, &stg.positions, proc_count)?;
            pack.put_prefix_at("slots", slot_offset, &stg.slots, proc_count)?;

            let devs = if marconi_skip {
                let bt_start = (pack.high_water() + 3) & !3;
                pack.put_at("block_table", bt_start, &seq.block_table)?;
                let sl_start = (pack.high_water() + 3) & !3;
                pack.put_at("seq_len", sl_start, &[n as u32])?;
                (meta_base.offset(bt_start), meta_base.offset(sl_start))
            } else {
                (DevicePtr::NULL, DevicePtr::NULL)
            };

            self.gpu
                .copy_h2d_async_retained(pack.packed(), meta_base, stream)?;
            devs
        };

        // 2026-09-25: Request-scoped LoRA routing: every processed token carries
        // this request's adapter slot, uploaded to the `lora_seq_slot` arena
        // buffer. `DevicePtr(0)` (no adapter pool) makes the apply sites take the
        // installed-pair path; `adapter_slot == -1` resolves to the active slot.
        let seq_slot = self.upload_seq_slot_uniform(
            seq.adapter_slot,
            proc_count,
            self.buffers.lora_seq_slot(),
            stream,
        )?;

        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: seq.block_table.len() as u32,
            num_seqs: 1,
            seq_slot,
            moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        };

        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(attn_metadata),
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: false,
            decode_step: false,
            // 2026-09-25: A Marconi warm hit makes GDN layers use the exact WY4
            // recurrence; the field's doc in layer.rs gives the reason.
            gdn_exact_replay: marconi_skip,
            gdn_write_on_accept: false,
            // 2026-09-25: Hash-MoE token IDs for the processed tokens, uploaded
            // above.
            token_ids: Some(self.buffers.token_ids()),
            host_token_ids: None,
            // 2026-09-25: `Some` only when the request routes to a non-active
            // adapter slot.
            routed_lora_layers: self.routed_slot_layers(seq.adapter_slot),
            midchunk_capture: None,
            moe_lora_route: self.moe_lora_route(seq.adapter_slot),
        };

        // 2026-09-25: 4. Forward through all layers. On a Marconi hit the KV
        // write floor is `cached_prefix_tokens - seq_len_start` (capped at
        // `proc_count`): processed rows whose positions are already in shared
        // prefix-cache blocks do not rewrite that KV.
        let layer_kv_write_start = if marconi_skip {
            seq.cached_prefix_tokens
                .saturating_sub(seq_len_start)
                .min(proc_count)
        } else {
            kv_write_start
        };
        let diag_prefill = self.profile && proc_count > 1;
        for (i, layer) in self.layers.iter().enumerate() {
            layer
                .prefill(
                    hidden,
                    residual,
                    proc_count,
                    seq.layer_states[i].as_mut(),
                    &mut kv_cache,
                    seq_len_start,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    layer_kv_write_start,
                    &ctx,
                    stream,
                )
                .map_err(|e| anyhow::anyhow!("Prefill layer {i} failed: {e}"))?;
            // 2026-09-25: DFlash prefill capture of layer i's output for the
            // processed rows, into accumulator slots from `seq_len_start` (the
            // position of buffer row 0), not from the KV write floor. A no-op
            // when the model has no DFlash capture layers.
            self.try_dflash_prefill_capture_layer(seq, i, seq_len_start, proc_count, stream)?;

            // 2026-09-25: Mistral diagnostic under `self.profile`: log each layer's
            // hidden-state norm, once per model (`stats.dumped` is per model).
            if self.profile
                && self.config.model_type == "mistral"
                && self.stats.dumped.keyed("mla_prefill_norms")
            {
                self.gpu.synchronize(stream)?;
                let last_offset = (proc_count - 1) * self.config.hidden_size * 4;
                let h_sz = self.config.hidden_size;
                let mut buf = vec![0u16; h_sz];
                // 2026-09-25: SAFETY: `buf` is `vec![0u16; h_sz]` on the line above, so it
                // owns exactly `h_sz * size_of::<u16>()` initialised bytes and
                // the length matches its capacity. `bytes` is the only live
                // reference to that allocation for its whole lifetime — it is
                // last used on the `copy_d2h` line below, and `buf` is not read
                // again until after that.
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, h_sz * 2)
                };
                if self.gpu.copy_d2h(hidden.offset(last_offset), bytes).is_ok() {
                    let vals: Vec<f32> = buf
                        .iter()
                        .map(|&b| f32::from_bits((b as u32) << 16))
                        .collect();
                    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::info!("LAYER_NORM L{i}: hidden_norm={norm:.4}");
                    if i == self.layers.len() - 1 {}
                }
            }

            // 2026-09-25: Profile diagnostic: read back the last processed token's
            // hidden state after each layer.
            if diag_prefill {
                self.gpu.synchronize(stream)?;
                let last_start = (proc_count - 1) * h;
                let (last_vals, last_norm) =
                    self.readback_bf16(hidden.offset(last_start * fp32), h.min(64))?;
                let last_nan = last_vals.iter().filter(|v| v.is_nan()).count();
                let last_inf = last_vals.iter().filter(|v| v.is_infinite()).count();
                let lt = self.config.layer_type(i);
                if i % 4 == 0 || i == self.layers.len() - 1 || last_nan > 0 || last_inf > 0 {
                    tracing::warn!(
                        "DIAG L{i} ({lt:?}) last_tok: norm={last_norm:.4} nan={last_nan} inf={last_inf} first4={:.4?}",
                        &last_vals[..4.min(last_vals.len())]
                    );
                }
            }
        }

        // 2026-09-25: Capture the processed rows' final-layer hiddens for the
        // whole-prompt drafter prefill. A no-op without a capture buffer.
        self.try_mtp_prefill_capture(seq, seq_len_start, proc_count, stream)?;

        // 2026-09-25: 5. Final norm on the last token only.
        let last_hidden = hidden.offset((proc_count - 1) * h * fp32);
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        self.final_norm_apply(last_hidden, normed, 1, h as u32, eps, stream)?;

        // 2026-09-25: 6. LM head on the last token.
        self.lm_head(normed, stream)?;

        // 2026-09-25: 7. Update the sequence state.
        seq.tokens.extend_from_slice(tokens);
        seq.seq_len = n;
        // 2026-09-25: Prime the decode-checkpoint gate with the prompt's
        // full-block count, so decode does not checkpoint at a block boundary
        // the prompt already reached.
        seq.last_decode_ckpt_block = seq.tokens.len() / bs;

        // 2026-09-25: 8. Save the Marconi SSM snapshot and insert into the prefix
        // cache.
        self.prefill_save_snapshot_with_vision_gate(tokens, seq, &mut kv_cache, bs, stream);

        // 2026-09-25: Set the DFlash `ctx_len` to `seq_len_start + proc_count`,
        // the same base as the capture above, so the next propose reads every
        // prefilled position.
        self.update_dflash_ctx_len_after_prefill(seq, seq_len_start, proc_count)?;

        Ok(self.decode_logits_ptr())
    }
}
