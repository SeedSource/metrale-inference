// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Last-chunk prefill finish: final norm and LM head for the last token,
//! the prefix-cache insert with the prompt-end SSM snapshot, and the DFlash context
//! length.
//!
//! Owner: model-engine prefill.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;
use metrale_model_layers::layers::ops;

impl TransformerModel {
    pub(in crate::model) fn prefill_b_finalize_last(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_b_finalize_last_at(
            tokens,
            seq,
            kv_cache,
            chunk_start,
            chunk_len,
            proc_count,
            0,
            0,
            stream,
        )
    }

    /// 2026-09-25: `prefill_b_finalize_last` with the last token read from row
    /// `hidden_stream_offset_tokens + proc_count - 1` of `hidden_states()` and the logits
    /// written to row `logits_row`. `batch_kernel.rs` passes the stream's packed offset
    /// (the sum of earlier streams' `proc_count`); `batch.rs` passes 0.
    pub(in crate::model) fn prefill_b_finalize_last_at(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        hidden_stream_offset_tokens: usize,
        logits_row: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size;
        let fp32 = 2usize;
        let hidden = self.buffers.hidden_states();
        let bs = kv_cache.block_size();

        if std::env::var("METRALE_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("final_state@{}", tokens.len()),
            );
            // 2026-09-25: Per-layer K/V checksums over the block table, split at block
            // `marconi_skip_to / bs`: reused prefix below, recomputed suffix above.
            let boundary_idx = seq.marconi_skip_to / bs;
            kv_cache.debug_kv_checksum_per_layer(
                &seq.block_table,
                boundary_idx,
                self.gpu.as_ref(),
                stream,
                &format!("final@{}/skip{}", tokens.len(), seq.marconi_skip_to),
            );
            // 2026-09-25: Per-block checksums of the fixed logical blocks 250..266 for layers
            // 0 and 9, so two runs dump the same positions and a wrong block-to-position
            // mapping shows up even when the region sums agree.
            let _ = boundary_idx;
            let lo = 250usize.min(seq.block_table.len());
            let hi = 266usize.min(seq.block_table.len());
            if hi > lo {
                for layer_idx in [0usize, 9usize] {
                    kv_cache.debug_kv_per_block(
                        layer_idx,
                        &seq.block_table[lo..hi],
                        self.gpu.as_ref(),
                        stream,
                        &format!(
                            "abswin@{}/skip{}/lo{}",
                            tokens.len(),
                            seq.marconi_skip_to,
                            lo
                        ),
                    );
                }
            }
            let tail_lo = seq.block_table.len().saturating_sub(3);
            kv_cache.debug_kv_per_block(
                0,
                &seq.block_table[tail_lo..],
                self.gpu.as_ref(),
                stream,
                &format!("tail@{}/lo{}", tokens.len(), tail_lo),
            );
            tracing::warn!(
                "METRALE_BTBL[final@{}/skip{}] nblk={} bt={:?}",
                tokens.len(),
                seq.marconi_skip_to,
                seq.block_table.len(),
                seq.block_table,
            );
        }

        // 2026-09-25: Final norm of the last processed token only.
        let last_token_offset = hidden_stream_offset_tokens + proc_count - 1;
        let last_hidden = hidden.offset(last_token_offset * h * fp32);
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        self.final_norm_apply(last_hidden, normed, 1, h as u32, eps, stream)?;

        if std::env::var("METRALE_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true") {
            self.gpu.synchronize(stream)?;
            let (vals, norm) = self.readback_bf16(normed, h.min(16))?;
            tracing::warn!(
                "DIAG post-norm: norm={norm:.4} first2={:.4?}",
                &vals[..2.min(vals.len())]
            );
        }

        if let Ok(dir) = std::env::var("METRALE_NEMO_DUMP")
            && !dir.is_empty()
        {
            self.gpu.synchronize(stream)?;
            let (vals, _) = self.readback_bf16(normed, h)?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::create_dir_all(&dir).ok();
            std::fs::write(
                std::path::Path::new(&dir).join("metrale_final_norm.bin"),
                &bytes,
            )
            .ok();
        }

        // 2026-09-25: Exact full-prompt restore: `proc_range` re-ran the last prompt token
        // (`Compute { N-1, 1 }`) on top of the restored state after N tokens, so SSM
        // layers applied that token twice. Restore the snapshot again, and replace
        // `normed` with the snapshot's stashed last-token hidden, so `lm_head` and
        // decode both start from the state after N tokens.
        if let Some(snap_id) = seq.marconi_exact_snap {
            self.ssm_snapshots.restore(
                snap_id,
                seq.slot_idx,
                &self.ssm_pool,
                self.gpu.as_ref(),
                stream,
            )?;
            if self.ssm_snapshots.has_hidden(snap_id) {
                self.ssm_snapshots
                    .restore_hidden(snap_id, normed, self.gpu.as_ref(), stream)?;
            } else {
                // 2026-09-25: `snap_agree::local_proposal` refuses an exact restore of a
                // snapshot without a stashed hidden, so this arm is not expected. It keeps
                // the re-run's `normed` and logs a warning.
                tracing::warn!(
                    "Marconi exact hit on snapshot {snap_id} without stashed hidden — \
                     first-token logits may be degraded (SSM state restored)"
                );
            }
        }

        // 2026-09-25: LM head for the last token. Each batched stream writes its own
        // logits row: row 0 through `lm_head` (returning `decode_logits_ptr()`), other
        // rows through `lm_head_batched` at `logits_row * vocab_size` BF16 elements.
        let logits_ptr = if logits_row == 0 {
            self.lm_head(normed, stream)?;
            self.decode_logits_ptr()
        } else {
            let v = self.config.vocab_size;
            let dst = self.buffers.logits().offset(logits_row * v * 2);
            self.lm_head_batched(normed, 1, dst, stream)?;
            dst
        };

        if let Ok(dir) = std::env::var("METRALE_NEMO_DUMP")
            && !dir.is_empty()
        {
            self.gpu.synchronize(stream)?;
            let n_logits = self.config.vocab_size;
            let mut buf = vec![0u8; n_logits * 2];
            self.gpu.copy_d2h(self.buffers.logits(), &mut buf)?;
            let logit_vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let lbytes: Vec<u8> = logit_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::create_dir_all(&dir).ok();
            std::fs::write(
                std::path::Path::new(&dir).join("metrale_logits.bin"),
                &lbytes,
            )
            .ok();
            let mut idx: Vec<usize> = (0..logit_vals.len()).collect();
            idx.sort_by(|&a, &b| logit_vals[b].partial_cmp(&logit_vals[a]).unwrap());
            let top: Vec<(usize, f32)> = idx.iter().take(10).map(|&i| (i, logit_vals[i])).collect();
            tracing::info!("METRALE_NEMO_DUMP: top-10 logits = {top:?}");
        }

        if std::env::var("METRALE_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true") {
            self.gpu.synchronize(stream)?;
            let logits_ptr = self.buffers.logits();
            let n_logits = self.config.vocab_size;
            let mut buf = vec![0u8; n_logits * 2];
            self.gpu.copy_d2h(logits_ptr, &mut buf)?;
            let logit_vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let max = logit_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = logit_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let nan_count = logit_vals.iter().filter(|v| v.is_nan()).count();
            let mut idx: Vec<usize> = (0..logit_vals.len()).collect();
            idx.sort_by(|&a, &b| {
                logit_vals[b]
                    .partial_cmp(&logit_vals[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, logit_vals[i])).collect();
            tracing::warn!(
                "DIAG logits[0..{}]: max={max:.4} min={min:.4} nan={nan_count} top5={top5:?}",
                n_logits,
            );
        }

        // 2026-09-25: Prefix-cache insert. Only complete blocks below `seq.kv_valid_tokens`
        // (the prefix whose K/V this or an earlier prefill wrote, kept by `proc_range`)
        // may be cached. When every complete block qualifies, the whole prompt is
        // inserted. Otherwise only the valid block-aligned prefix is, with no SSM
        // snapshot: a snapshot keyed at the full prompt length would be unreachable
        // through the shorter tree.
        let full_blocks = tokens.len() / bs;
        let valid_blocks = seq.kv_valid_tokens / bs;
        let cache_blocks = full_blocks.min(valid_blocks);
        let cap_applied = cache_blocks < full_blocks;
        let cache_tokens_len = if cap_applied {
            cache_blocks * bs
        } else {
            tokens.len()
        };
        let cache_tokens = &tokens[..cache_tokens_len];
        let cache_block_table = &seq.block_table[..cache_blocks.min(seq.block_table.len())];
        let cache_disk_block_ids = if seq.disk_block_ids.is_empty() {
            &seq.disk_block_ids[..]
        } else {
            &seq.disk_block_ids[..cache_blocks.min(seq.disk_block_ids.len())]
        };
        if cap_applied {
            tracing::warn!(
                "Prefix-cache stale-V cap: caching {} of {} complete blocks \
                 (kv_valid_tokens={} < prompt_len={}); trailing blocks had \
                 unwritten K/V and are excluded",
                cache_blocks,
                full_blocks,
                seq.kv_valid_tokens,
                tokens.len(),
            );
        }

        // 2026-10-03: Under the absolute grid (`grid_restore.rs`) only the blocks below the last
        // chunk's start are inserted (`prefill_plan::grid_insert_len`), all written by full
        // `G`-row grid chunks, and no prompt-end leaf snapshot is saved (the restore never uses a
        // point at the prompt end). `prefix_ref_tokens` then names exactly the radix path this
        // sequence holds refs on: the lookup's `[0, cached_prefix_tokens)` plus the nodes this
        // insert gives it (`matched_tokens = cached_prefix_tokens`). Off: `None`.
        let grid = if seq.prefix_grid_refs {
            self.prefix_grid_for_bs(tokens, bs)
        } else {
            None
        };
        if seq.marconi_exact_snap.is_some() {
            // 2026-09-25: Exact restore: the prompt's snapshot came from the cache; nothing
            // new is saved.
        } else if cache_blocks == 0 {
            // 2026-09-25: No complete block with written K/V, or a prompt under one block.
        } else if let Some(g) = grid {
            let ins = crate::prefill_plan::grid_insert_len(tokens.len(), g).min(cache_blocks * bs);
            let ins_blocks = (ins / bs).min(seq.block_table.len());
            let ins = ins_blocks * bs;
            let insertable = ins > 0
                && !self.tokens_have_vision_pad(&tokens[..ins])
                && !self.hss_window_slid(seq);
            if insertable {
                let ins_disk = if seq.disk_block_ids.is_empty() {
                    &seq.disk_block_ids[..]
                } else {
                    &seq.disk_block_ids[..ins_blocks.min(seq.disk_block_ids.len())]
                };
                let acquired = self.prefix_cache.insert(
                    &tokens[..ins],
                    &seq.block_table[..ins_blocks],
                    ins_disk,
                    bs,
                    seq.cached_prefix_tokens.min(ins),
                    seq.adapter_id,
                );
                super::super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
                seq.prefix_ref_tokens = tokens[..ins.max(seq.cached_prefix_tokens)].to_vec();
            }
        } else if cap_applied {
            if !self.tokens_have_vision_pad(cache_tokens) && !self.hss_window_slid(seq) {
                let acquired = self.prefix_cache.insert(
                    cache_tokens,
                    cache_block_table,
                    cache_disk_block_ids,
                    bs,
                    seq.cached_prefix_tokens.min(cache_tokens_len),
                    seq.adapter_id,
                );
                super::super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
            }
        } else if self.ssm_snapshots.is_enabled()
            && let super::exact_leaf::ExactLeaf::Redundant { tail, replay } =
                super::exact_leaf::exact_leaf(
                    seq.tail_checkpoint_tokens,
                    tokens.len(),
                    bs,
                    super::exact_leaf::marconi_exact_enabled(),
                )
        {
            // 2026-09-25: A checkpoint within two blocks of the prompt end already exists
            // (`exact_leaf`), so only the K/V is inserted and no pool slot goes to the leaf.
            tracing::debug!(
                "exact leaf not saved for {} tokens: tail checkpoint at {tail} covers it \
                 (replay {replay} tokens)",
                tokens.len()
            );
            if !self.tokens_have_vision_pad(tokens) && !self.hss_window_slid(seq) {
                let acquired = self.prefix_cache.insert(
                    tokens,
                    &seq.block_table,
                    &seq.disk_block_ids,
                    bs,
                    seq.cached_prefix_tokens,
                    seq.adapter_id,
                );
                super::super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
            }
        } else if self.ssm_snapshots.is_enabled() {
            if std::env::var("METRALE_SSM_SAVE_DUMP").is_ok() {
                self.ssm_pool.debug_state_checksum(
                    seq.slot_idx,
                    self.gpu.as_ref(),
                    stream,
                    &format!("leaf_save@{}", tokens.len()),
                );
            }
            let snap_result = match self.ssm_snapshots.save(
                seq.slot_idx,
                seq.session_hash,
                self.seq_ssm_h_is_f16(seq),
                &self.ssm_pool,
                self.gpu.as_ref(),
                stream,
            ) {
                Ok(Some(id)) => Some(id),
                Ok(None) => {
                    tracing::debug!("Snapshot pool full, reclaiming...");
                    if self.ssm_snapshots.reclaim_from_cache(
                        self.prefix_cache.as_ref(),
                        kv_cache,
                        self.ssm_tier_store.as_deref(),
                        self.gpu.as_ref(),
                    ) {
                        self.ssm_snapshots
                            .save(
                                seq.slot_idx,
                                seq.session_hash,
                                self.seq_ssm_h_is_f16(seq),
                                &self.ssm_pool,
                                self.gpu.as_ref(),
                                stream,
                            )
                            .ok()
                            .flatten()
                    } else {
                        tracing::debug!("Reclaim failed — no evictable snapshots");
                        None
                    }
                }
                Err(e) => {
                    tracing::warn!("SSM snapshot save error: {e}");
                    None
                }
            };
            if let Some(snap_id) = snap_result {
                if self.tokens_have_vision_pad(tokens) || self.hss_window_slid(seq) {
                    // 2026-09-25: No radix insert for a vision prompt (the pad tokens are the
                    // same for different images) or a slid HSS window (`block_table` no
                    // longer parallels the tokens). The snapshot is reachable only through
                    // that insert, so it is freed.
                    self.ssm_snapshots.free(snap_id);
                } else {
                    tracing::info!(
                        "Saved SSM snapshot {} for {} tokens ({} blocks) [chunk]",
                        snap_id,
                        tokens.len(),
                        seq.block_table.len(),
                    );
                    // 2026-09-25: Per-sequence aux layer state (`collect_aux_states`) rides the
                    // snapshot; on restore, a model that carries aux state refuses a snapshot
                    // without it (`snap_agree::local_proposal`).
                    // 2026-10-03: An aux capture or hidden stash that fails frees the slot
                    // before returning the error (it was leaked before).
                    let aux = match self.collect_aux_states(seq, stream) {
                        Ok(aux) => aux,
                        Err(e) => {
                            self.ssm_snapshots.free(snap_id);
                            return Err(e);
                        }
                    };
                    if !aux.is_empty() {
                        self.ssm_snapshots.set_aux(snap_id, aux);
                    }
                    // 2026-09-25: Stash the last token's final-norm output (`normed`, an
                    // `lm_head` input) for the exact-restore fixup above.
                    let stashed = self
                        .ssm_snapshots
                        .save_hidden(snap_id, normed, self.gpu.as_ref(), stream);
                    if let Err(e) = stashed {
                        self.ssm_snapshots.free(snap_id);
                        return Err(e);
                    }
                    let (displaced, acquired) = self.prefix_cache.insert_with_snapshot(
                        tokens,
                        &seq.block_table,
                        &seq.disk_block_ids,
                        bs,
                        snap_id,
                        seq.session_hash,
                        seq.cached_prefix_tokens,
                        seq.adapter_id,
                    );
                    super::super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
                    if let Some(old) = displaced {
                        self.ssm_snapshots.free(old);
                    }
                }
            } else if !self.tokens_have_vision_pad(tokens) && !self.hss_window_slid(seq) {
                let acquired = self.prefix_cache.insert(
                    tokens,
                    &seq.block_table,
                    &seq.disk_block_ids,
                    bs,
                    seq.cached_prefix_tokens,
                    seq.adapter_id,
                );
                super::super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
            }
        } else if !self.tokens_have_vision_pad(tokens) && !self.hss_window_slid(seq) {
            let acquired = self.prefix_cache.insert(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
                seq.adapter_id,
            );
            super::super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
        }

        self.update_dflash_ctx_len_after_prefill(seq, chunk_start, chunk_len)?;

        Ok(logits_ptr)
    }
}
