// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Chunked prefill, `prefill_chunk_dispatch`: one chunk of a prompt through the step modules under `prefill_b/`.
//!
//! Steps, in order: `embed_chunk` (embedding and vision-pad overlay),
//! `prefix_lookup` (prefix cache, EP agreement, Marconi restore), `proc_range`
//! (processing range; may return early), `upload_meta` and `upload_paged`
//! (metadata upload), `forward_layers`, then `finalize_last` on the last chunk
//! or `save_checkpoint` on any other. Once taken, the `kv_cache` lock is held
//! for the rest of the chunk and passed to each step as `&mut`. The multi-stream
//! path is in `batch.rs` and `batch_kernel.rs`.
//!
//! Owner: model-engine.
//! Invariants:
//! - A chunk that returns `Ok`, including a fully cached one, appends its tokens
//!   to `seq.tokens` and sets `seq.seq_len` to `chunk_start + chunk_len`.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

mod batch;
mod batch_kernel;
#[cfg(test)]
mod batch_kernel_tests;
mod batched_layer;
mod embed_chunk;
mod exact_leaf;
mod finalize_last;
mod forward_layers;
mod grid_restore;
mod h_state_ptrs;
mod inpass_capture;
// 2026-10-05: Also the verdict of the decode-time vote (`decode_lazy_agree.rs`).
pub(in crate::model) mod lazy_agree;
#[cfg(test)]
mod lazy_agree_tests;
mod midchunk_capture;
mod prefix_lookup;
mod prefix_reserve;
mod proc_range;
mod prompt_logprobs;
mod save_checkpoint;
mod snap_agree;
// 2026-10-05: Its two-rank gather harness also serves `decode_lazy_agree_tests.rs`.
#[cfg(test)]
pub(in crate::model) mod snap_agree_tests;
mod stage_batched;
mod upload_meta;
mod upload_paged;

impl TransformerModel {
    /// 2026-09-27: The token at which this prompt's prefill is split so that an SSM
    /// snapshot lands at `prefill_plan::tail_split_point`: `Some` for an SSM model with
    /// snapshots and prefix caching on and no vision pads in the prompt.
    /// `METRALE_NO_TAIL_SPLIT=1` turns the split off.
    /// 2026-10-03: `None` as well when the in-pass capture replaces the split
    /// (`METRALE_GLM_SSM_INPASS_CAPTURE=1`, `inpass_capture.rs`), so the scheduler and this
    /// dispatcher run the cache-off pass sequence.
    /// 2026-10-03: `None` as well under the absolute grid (`METRALE_PREFIX_GRID_RESTORE=1`,
    /// `grid_restore.rs`), whose chunks end on multiples of `G` and carry the snapshots.
    pub(in crate::model) fn prefill_tail_split_dispatch(&self, tokens: &[u32]) -> Option<usize> {
        if !self.tail_split_eligible(tokens) {
            return None;
        }
        let bs = self.kv_cache.lock().block_size();
        // 2026-10-03: The absolute grid (`grid_restore.rs`) replaces the tail split.
        if self.prefix_grid_for_bs(tokens, bs).is_some() {
            return None;
        }
        let cut = crate::prefill_plan::tail_split_point(tokens.len(), bs);
        crate::prefill_plan::effective_split(cut, self.inpass_ssm_capture_active())
    }

    /// 2026-10-03: Whether this prompt gets a prefix-cache tail snapshot at all: an SSM model
    /// with snapshots and prefix caching on, `METRALE_NO_TAIL_SPLIT` not `1`, and no vision
    /// pads in the prompt (the conditions `prefill_tail_split_dispatch` has always checked, in
    /// the same order). Takes no lock.
    pub(in crate::model) fn tail_split_eligible(&self, tokens: &[u32]) -> bool {
        !(self.config.num_ssm_layers() == 0
            || !self.ssm_snapshots.is_enabled()
            || !self.prefix_cache.is_active()
            || std::env::var("METRALE_NO_TAIL_SPLIT").as_deref() == Ok("1")
            || self.tokens_have_vision_pad(tokens))
    }

    pub(super) fn prefill_chunk_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let total = tokens.len();
        assert!(
            chunk_start + chunk_len <= total,
            "chunk_start({chunk_start}) + chunk_len({chunk_len}) > total({total})"
        );

        // 2026-09-25: Tail-checkpoint split. A later turn's prefix match is
        // block-aligned, and a snapshot deeper than the match cannot be
        // restored. A last chunk that spans the split point
        // (`prefill_tail_split_dispatch`) is split there once, so
        // `prefill_b_save_checkpoint` saves a snapshot at it. 2026-09-27: the
        // scheduler ends a non-last chunk at the same point
        // (`prefill_plan::plan_chunk_len`), so only a last chunk can span it
        // here. The split does not depend on radix contents, so a prompt is
        // processed in the same passes cold and warm, and on every rank.
        if is_last_chunk
            && let Some(cut) = self.prefill_tail_split_dispatch(tokens)
            && cut > chunk_start
        {
            self.prefill_chunk_dispatch(
                tokens,
                seq,
                chunk_start,
                cut - chunk_start,
                false,
                stream,
            )?;
            return self.prefill_chunk_dispatch(tokens, seq, cut, total - cut, true, stream);
        }

        let arena_cap = self.buffers.max_batch_tokens();
        if chunk_len > arena_cap {
            anyhow::bail!(
                "Prefill chunk ({chunk_len} tokens) exceeds buffer arena capacity ({arena_cap} tokens). \
                 Reduce --max-prefill-tokens or prompt length."
            );
        }

        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };

        // 2026-09-25: With `comm` set, every buffer is zeroed on every chunk.
        // Otherwise only the first chunk zeroes, and only the prefill
        // essentials; later chunks rely on the embedding and the layer forward
        // writing each buffer before it is read.
        if self.comm.is_some() {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        } else if chunk_start == 0 {
            self.buffers
                .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
        }

        let mut kv_cache = self.kv_cache.lock();

        // 2026-09-25: Embed the chunk and overlay vision-pad positions.
        self.prefill_b_embed_chunk(tokens, chunk_start, chunk_len, stream)?;

        // 2026-09-25: Prefix-cache lookup, EP agreement and Marconi snapshot restore.
        let (kv_write_start, marconi_skip) = self.prefill_b_prefix_lookup(
            tokens,
            seq,
            chunk_start,
            total,
            &mut kv_cache,
            stream,
            None,
        )?;
        // 2026-10-03: Under the grid the scheduler plans every chunk with `grid_chunk_len`
        // (`Model::prefill_grid`); a chunk off that plan means a caller bypassed it, and a warm
        // pass then no longer matches a cold one (the restore itself stays correct).
        // 2026-10-03: `METRALE_PREFILL_CHUNK_WHILE_DECODING` (server `prefill_chunk_cap`) may
        // end a chunk short of the next grid point, never past it, so only a chunk that crosses
        // a grid point is off the plan.
        if seq.prefix_grid_refs
            && let Some(g) = self.prefix_grid_active_bs(kv_cache.block_size())
            && chunk_len > crate::prefill_plan::grid_chunk_len(chunk_start, total, g)
        {
            static W: std::sync::Once = std::sync::Once::new();
            W.call_once(|| {
                tracing::warn!(
                    "grid restore: prefill chunk [{chunk_start}, +{chunk_len}) of {total} is off \
                     the {g}-token grid; warm and cold passes may differ"
                )
            });
        }

        if std::env::var("METRALE_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("chunk_entry start={chunk_start} len={chunk_len} kvws={kv_write_start}"),
            );
        }

        let bs = kv_cache.block_size();
        let end_pos = chunk_start + chunk_len;
        let blocks_needed = (end_pos - 1) / bs + 1;
        let admitted = super::super::block_mgmt::ensure_blocks_through_prefill(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
            self.levers.kv_poison,
        );
        // 2026-10-05: Rank-agreed admission (race-memory #79, A168): the KV blocks and the
        // lazily mapped DSA indexer rows are admitted per rank (the lazy-map floor reads this
        // rank's free memory), so every rank votes before the chunk's first forward
        // collective, and a refusal on any rank fails the chunk on all of them. One
        // `ep_gather_u32` per chunk; single-rank returns `admitted` unchanged.
        self.agree_admission(
            admitted,
            &format!("prefill chunk [{chunk_start}, +{chunk_len}) of {total}"),
        )?;

        // 2026-09-25: Processing range for this chunk; a fully cached chunk
        // returns early.
        let (proc_start, proc_count, effective_seq_len_start) = match self.prefill_b_proc_range(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            is_last_chunk,
            kv_write_start,
            marconi_skip,
            // 2026-09-25: A single stream's hidden rows start at the buffer base.
            self.buffers.hidden_states(),
            stream,
        )? {
            proc_range::ProcRange::Compute {
                proc_start,
                proc_count,
                effective_seq_len_start,
            } => (proc_start, proc_count, effective_seq_len_start),
            proc_range::ProcRange::EarlyReturn(ptr) => {
                // 2026-09-25: A fully cached chunk still records its tokens:
                // decode-checkpoint registration and the radix insert in
                // `cache_sequence` read `seq.tokens`, paired with the full block
                // table.
                seq.tokens
                    .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
                seq.seq_len = chunk_start + chunk_len;
                seq.last_decode_ckpt_block = seq.tokens.len() / bs;
                return Ok(ptr);
            }
        };

        // 2026-09-25: Upload positions (MRoPE when enabled) and slot metadata.
        let upload_meta::MetaLayout {
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
        } = self.prefill_b_upload_meta(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_start,
            proc_count,
            effective_seq_len_start,
            &kv_cache,
            stream,
        )?;

        // 2026-09-25: Paged metadata (block table and `seq_len`).
        if needs_paged {
            self.prefill_b_upload_paged(
                seq,
                total,
                proc_start,
                proc_count,
                meta_base,
                slot_offset,
                &kv_cache,
                stream,
            )?;
        }

        self.gpu.synchronize(stream)?;

        // 2026-09-25: Mid-chunk tail SSM capture is planned before the forward
        // pass, which uses the plan. `None` (flag off, or the pass does not span
        // `tb`, among other cases) means no capture.
        // 2026-10-03: Under `METRALE_GLM_SSM_INPASS_CAPTURE=1` the in-pass tail-split capture
        // (`inpass_capture.rs`) takes the plan's place; with the lever off it is `None` and
        // the mid-chunk plan is made as before.
        let midcap_plan = match self.prepare_inpass_capture(
            tokens,
            seq,
            &mut kv_cache,
            proc_start,
            proc_count,
        ) {
            Some(plan) => Some(plan),
            None => self.prepare_midchunk_capture(
                tokens,
                seq,
                &mut kv_cache,
                proc_start,
                proc_count,
                stream,
            ),
        };

        // 2026-09-25: Forward through all layers.
        let forward = self.prefill_b_forward_layers(
            seq,
            &mut kv_cache,
            chunk_start,
            chunk_len,
            is_last_chunk,
            proc_count,
            effective_seq_len_start,
            kv_write_start,
            marconi_skip,
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
            midcap_plan.as_ref(),
            stream,
        );
        // 2026-10-03: A failed pass returns the in-pass slot it reserved.
        if let Some(plan) = midcap_plan.as_ref().filter(|p| p.inpass && forward.is_err()) {
            self.ssm_snapshots.free(plan.snap_slot);
        }
        forward?;

        // 2026-09-25: Register the captured slots once the pass has written the
        // `tb` state into them.
        // 2026-10-03: The in-pass capture registers as an intermediate checkpoint, before
        // `finalize_last` inserts the whole prompt, as the split's first pass did.
        if let Some(plan) = midcap_plan.as_ref() {
            if plan.inpass {
                self.finalize_inpass_capture(tokens, seq, &mut kv_cache, plan, stream);
            } else {
                self.finalize_midchunk_capture(tokens, seq, plan);
            }
        }

        // 2026-09-25: Append this chunk's tokens; the early-return arm above
        // appends them itself.
        seq.tokens
            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
        seq.seq_len = chunk_start + chunk_len;
        // 2026-09-25: Prime the decode-checkpoint gate; after the last chunk it
        // holds the prompt's full-block count (see prefill_a.rs).
        seq.last_decode_ckpt_block = seq.tokens.len() / bs;

        // 2026-09-25: Prompt logprobs are projected while this chunk's hidden
        // rows are live, before `prefill_b_finalize_last` overwrites the norm
        // output and logits. A no-op unless `seq.collect_prompt_logprobs` is
        // set.
        self.collect_prompt_logprobs_chunk(
            tokens,
            seq,
            chunk_start,
            proc_start,
            proc_count,
            stream,
        )?;

        if is_last_chunk {
            // 2026-09-25: Final norm, LM head, prefix-cache insert and snapshot save.
            self.prefill_b_finalize_last(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                proc_count,
                stream,
            )
        } else {
            // 2026-09-25: Intermediate Marconi checkpoint.
            self.prefill_b_save_checkpoint(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                stream,
            )?;
            Ok(DevicePtr::NULL)
        }
    }
}
