// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Speculative-decoding plumbing on `TransformerModel`: one-shot speculative
//! generation, the MTP drafter's prompt context on a sequence's first propose (cold
//! prefill or the cross-turn carry), and the target-hidden copies the MTP head reads.
//!
//! Owner: model-engine.
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
use crate::traits::ModelLifecycle;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn generate_speculative_dispatch(
        &self,
        prompt_tokens: &[u32],
        params: &metrale_sampling::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        // 2026-09-25: Self-speculative mode drafts by skipping SSM layers, without MTP weights.
        if self.self_speculative {
            let mut seq = self.alloc_sequence()?;
            let stream = self.gpu.default_stream();
            let result = self.generate_self_speculative_inner(
                prompt_tokens,
                params,
                num_drafts,
                &mut seq,
                stream,
            );
            self.free_sequence(&mut seq)?;
            return result;
        }

        let proposer = match &self.proposer {
            Some(p) => p.clone(),
            None => {
                return crate::engine::generate(self, prompt_tokens, params);
            }
        };

        let mut seq = self.alloc_sequence()?;
        let stream = self.gpu.default_stream();

        let result = self.generate_speculative_inner(
            prompt_tokens,
            params,
            num_drafts,
            &proposer,
            &mut seq,
            stream,
        );

        self.free_sequence(&mut seq)?;

        result
    }

    pub(super) fn has_proposer_dispatch(&self) -> bool {
        self.proposer.is_some() || self.self_speculative
    }

    pub(super) fn has_self_speculative_dispatch(&self) -> bool {
        self.self_speculative
    }

    pub(super) fn decode_draft_dispatch(
        &self,
        token: u32,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        TransformerModel::decode_draft(self, token, seq, stream)
    }

    /// 2026-09-25: Give the drafter its prompt context on the first propose of a sequence.
    ///
    /// Cold turn: the whole-prompt hidden capture covers the prompt, so run
    /// `prefill_drafter`. Warm turn: a reused prefix is not recomputed, so the capture
    /// does not cover it; with the carry armed, adopt the previous turn's drafter KV and
    /// append only the new span (`try_carry_drafter`).
    pub(in crate::model) fn ensure_drafter_context(
        &self,
        proposer: &dyn DraftProposer,
        seq: &mut SequenceState,
        ctx: &ForwardContext,
        stream: u64,
    ) {
        // 2026-09-25: The proposer state is mutated while the token slice is read, so
        // `seq` is destructured into disjoint field borrows instead of cloning the tokens.
        // The scalars below are read first: `session_hash` gates the carry slot
        // (`CarriedDrafter`), `store_gen` the shared hidden rows (`StoreRange`).
        let capture_gen = seq.mtp_capture_gen;
        let session_hash = seq.session_hash;
        let store_gen = seq.mtp_store_gen;
        let SequenceState {
            tokens: seq_tokens,
            prompt_len,
            proposer_state,
            ..
        } = seq;
        let Some(prop_state) = proposer_state.as_mut() else {
            return;
        };
        let prompt_len = *prompt_len;
        // 2026-09-25: `prefill_drafter` batch-prefills the drafter KV over the prompt. It
        // needs the hidden capture to cover the whole prompt, and returns 0 without work
        // when the drafter already has rows or its weights are not supported.
        if !self.mtp_prefill_hidden.is_null() {
            let p = prompt_len;
            let captured = self
                .mtp_prefill_capture_len
                .load(std::sync::atomic::Ordering::Relaxed);
            // 2026-09-25: The capture is shared by the model, so it must still be this
            // sequence's (generation stamp): several sequences can prefill before any of
            // them proposes, and pairing this sequence's tokens with another's hiddens
            // would build wrong drafter rows. Without the stamp the cold prefill is skipped.
            let owns_capture = capture_gen != 0
                && capture_gen
                    == self
                        .mtp_prefill_capture_gen
                        .load(std::sync::atomic::Ordering::Relaxed);
            let cold_prefill_ok = p >= 2 && captured >= p && seq_tokens.len() >= p && owns_capture;
            let carry_on = metrale_model_layers::mtp_carry::mtp_carry_drafter_enabled(&self.levers);
            // 2026-09-25: Both branches run only on the first propose: `prefill_drafter`
            // returns early once the drafter has rows, and the carry checks `first_propose`.
            let first_propose = proposer.drafter_rows(prop_state.as_mut()) == 0;
            if cold_prefill_ok {
                // 2026-09-25: A cold turn builds its own rows, so a carried entry is freed
                // first: with the carry armed MTP runs single-sequence (`carry_armed_with`),
                // the drafter KV pool holds one sequence's blocks, and a live carried entry
                // would starve this prefill.
                if carry_on && let Some(old) = self.mtp_carry.lock().take() {
                    proposer.free_drafter_kv(&old.block_table);
                }
                // 2026-10-01: Under the chunked capture (`mtp_stage`) the earlier chunks are
                // already drafter rows; only the staged rest is drained.
                let prompt = &seq_tokens[..p];
                let hid = self.mtp_prefill_hidden;
                let done = if self.mtp_chunked() {
                    self.drain_stage_at_propose(
                        proposer,
                        prompt,
                        captured,
                        prop_state.as_mut(),
                        ctx,
                        stream,
                    )
                } else {
                    proposer.prefill_drafter(prompt, hid, prop_state.as_mut(), ctx, stream)
                };
                if let Err(e) = done {
                    tracing::warn!("MTP drafter prefill failed (continuing without): {e:#}");
                }
            } else if carry_on && first_propose && p >= 2 {
                let outcome = self.try_carry_drafter(
                    proposer,
                    seq_tokens,
                    p,
                    session_hash,
                    store_gen,
                    prop_state.as_mut(),
                    ctx,
                    stream,
                );
                if metrale_model_layers::mtp_carry::mtp_carry_debug() {
                    tracing::info!(
                        "MTP_CARRY adopt: prompt_len={p} store={:?} -> {outcome}",
                        *self.mtp_store_range.lock(),
                    );
                }
            }
        }
    }

    /// 2026-09-25: Drafter carry (`mtp_carry_drafter_enabled`): on the first propose of a
    /// sequence, adopt the previous turn's drafter KV and append only the span this turn
    /// computed, so the work is proportional to the new tokens.
    ///
    /// Conventions (see the `mtp_carry` module docs): pair key `k` is
    /// `(embed(t_{k+1}), hidden_k)` with RoPE `k + 1`; `mtp_prefill_hidden`
    /// row `i` is `hidden_i`. Rows are compacted, so a skipped key leaves a
    /// missing row, not a hole.
    ///
    /// Returns the outcome for logging and has no error path: every failure leaves the
    /// drafter with fewer rows.
    pub(in crate::model) fn try_carry_drafter(
        &self,
        proposer: &dyn DraftProposer,
        seq_tokens: &[u32],
        prompt_len: usize,
        session_hash: u64,
        store_gen: u64,
        prop_state: &mut dyn metrale_model_layers::speculative::ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> metrale_model_layers::mtp_carry::CarryOutcome {
        use metrale_model_layers::mtp_carry::{CarryOutcome, hidden_row_offset, plan_append};
        let prompt = &seq_tokens[..prompt_len.min(seq_tokens.len())];
        let Some(entry) = self.mtp_carry.lock().take() else {
            return CarryOutcome::NoCarry;
        };
        let Some((rows, last_key)) = entry.usable_by(prompt, session_hash) else {
            // 2026-09-25: `usable_by` decides admission; this only labels the refusal,
            // using the same session predicate to tell the two rules apart.
            let outcome = if entry.session_matches(session_hash) {
                CarryOutcome::PrefixMismatch {
                    common: entry.common_prefix_len(prompt),
                    entry_rows: entry.rows,
                }
            } else {
                CarryOutcome::ForeignSession {
                    entry_session: entry.session_hash,
                    prompt_session: session_hash,
                }
            };
            proposer.free_drafter_kv(&entry.block_table);
            return outcome;
        };
        // 2026-09-25: `install_drafter_kv` takes ownership on success only; keep a copy of
        // the ids so a refused install frees them instead of leaking.
        let block_ids = entry.block_table.clone();
        if !proposer.install_drafter_kv(prop_state, entry.block_table, rows, Some(last_key)) {
            proposer.free_drafter_kv(&block_ids);
            return CarryOutcome::NoCarry;
        }
        // 2026-09-25: Only hidden rows this sequence wrote are visible; another owner's
        // interval reads as empty, which `plan_append` refuses. This runs after the
        // install, so a refused append keeps the carried rows, and the blocks now belong
        // to the proposer state.
        let stored = *self.mtp_store_range.lock();
        let (lo, hi) = stored.visible_to(store_gen);
        let Some(plan) = plan_append(last_key, prompt.len(), lo, hi) else {
            return if stored.owner != store_gen && stored.owner != 0 {
                CarryOutcome::ForeignHiddens {
                    owner: stored.owner,
                    expected: store_gen,
                }
            } else {
                CarryOutcome::NoHiddens
            };
        };
        // 2026-09-25: `drafter_rows_impl` reads `tokens[r + 1]` and `hiddens` row `r` for
        // row r, and RoPE `pos_base + r`. Row r must be pair key
        // `first_key + r`, i.e. `(embed(t_{first_key+r+1}), hidden_{first_key+r})`
        // at RoPE `first_key + r + 1`.
        let tokens = &prompt[plan.first_key..];
        let hiddens = hidden_row_offset(
            self.mtp_prefill_hidden,
            plan.first_key,
            self.config.hidden_size,
        );
        match proposer.catchup_drafter(
            tokens,
            hiddens,
            rows,
            plan.first_key + 1,
            prop_state,
            ctx,
            stream,
        ) {
            Ok(appended) => CarryOutcome::Adopted {
                rows,
                appended,
                first_key: plan.first_key,
            },
            Err(e) => {
                tracing::warn!("MTP carry append failed (drafter keeps carried rows): {e:#}");
                CarryOutcome::Adopted {
                    rows,
                    appended: 0,
                    first_key: plan.first_key,
                }
            }
        }
    }

    pub(super) fn save_hidden_for_mtp_dispatch(
        &self,
        token_idx: usize,
        _stream: u64,
    ) -> Result<()> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        // 2026-09-25: `hidden_states` rows are BF16 (2 bytes), despite the name below.
        let fp32 = 2usize;
        // 2026-09-25: Save the hidden state before the final norm: the MTP head applies
        // its own `pre_fc_norm_hidden`.
        let src = self.buffers.hidden_states().offset(token_idx * h * fp32);
        self.gpu
            .copy_d2d_async(src, self.mtp_hidden_save, h * fp32, stream)?;
        self.last_mtp_hidden_idx
            .store(token_idx, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// 2026-09-25: Batched verify: copy the pre-final-norm hidden row `rows[i]` (sequence i's
    /// accepted position in the batched verify forward) into stash slot i, before a propose
    /// overwrites the shared `hidden_states` buffer (the MTP head's `forward_one` writes it).
    /// Errors when the stash is not allocated or `rows` exceeds its slots.
    pub(super) fn stash_verify_hidden_rows_dispatch(
        &self,
        rows: &[usize],
        _stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            !self.verify_hidden_stash.is_null(),
            "stash_verify_hidden_rows: verify_hidden_stash not allocated (no MTP proposer)"
        );
        anyhow::ensure!(
            rows.len() <= metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS,
            "stash_verify_hidden_rows: {} rows exceeds the {}-slot stash",
            rows.len(),
            metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS
        );
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        for (i, &row) in rows.iter().enumerate() {
            let src = self.buffers.hidden_states().offset(row * h * bf16);
            let dst = self.verify_hidden_stash.offset(i * h * bf16);
            self.gpu.copy_d2d_async(src, dst, h * bf16, stream)?;
        }
        Ok(())
    }

    /// 2026-09-25: Copy stash slot `idx` to `mtp_hidden_save`, the MTP head's input: the
    /// stashed-row form of `save_hidden_for_mtp_dispatch`, for verdicts applied after a
    /// propose has overwritten the live rows.
    pub(super) fn save_hidden_for_mtp_from_stash_dispatch(
        &self,
        idx: usize,
        _stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            !self.verify_hidden_stash.is_null(),
            "save_hidden_for_mtp_from_stash: verify_hidden_stash not allocated"
        );
        anyhow::ensure!(
            idx < metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS,
            "save_hidden_for_mtp_from_stash: idx {idx} >= {}",
            metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS
        );
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let src = self.verify_hidden_stash.offset(idx * h * bf16);
        self.gpu
            .copy_d2d_async(src, self.mtp_hidden_save, h * bf16, stream)?;
        Ok(())
    }

    /// 2026-09-25: MTP catch-up ring: store the hidden of a serially decoded token under
    /// label `pos`. The ring's label range stays contiguous: a gap resets it to this row,
    /// and a full ring drops its oldest row. A no-op when the ring is not allocated.
    pub(super) fn save_hidden_for_catchup_dispatch(
        &self,
        token_idx: usize,
        pos: usize,
    ) -> Result<()> {
        if self.mtp_catchup_ring.is_null() {
            return Ok(());
        }
        let ring_rows = super::super::types::MTP_CATCHUP_RING_ROWS;
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let src = self.buffers.hidden_states().offset(token_idx * h * bf16);
        let dst = self.mtp_catchup_ring.offset((pos % ring_rows) * h * bf16);
        self.gpu.copy_d2d_async(src, dst, h * bf16, stream)?;
        if metrale_model_layers::speculative::mtp_refeed_debug() {
            self.gpu.synchronize(stream)?;
            let fp_src =
                metrale_model_layers::speculative::hidden_fingerprint(self.gpu.as_ref(), src, h);
            let fp_dst =
                metrale_model_layers::speculative::hidden_fingerprint(self.gpu.as_ref(), dst, h);
            tracing::info!(
                "REFEED_DBG ring_write label={pos} row={token_idx} slot={} \
                 fp_src={fp_src:016x} fp_dst={fp_dst:016x} match={}",
                pos % ring_rows,
                fp_src == fp_dst,
            );
        }
        let mut meta = self.mtp_catchup_meta.lock();
        let (start, count) = *meta;
        *meta = if count > 0 && pos == start + count {
            if count == ring_rows {
                (start + 1, ring_rows)
            } else {
                (start, count + 1)
            }
        } else {
            (pos, 1)
        };
        Ok(())
    }

    pub(super) fn read_deferred_draft_token_dispatch(&self) -> Result<u32> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(0),
        };
        proposer.read_deferred_draft_token(self.gpu.as_ref())
    }

    pub(super) fn trim_proposer_state_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(()),
        };
        let stream = self.gpu.default_stream();
        if let Some(ref mut state) = seq.proposer_state {
            proposer.after_verify(num_accepted, state.as_mut(), stream)?;
        }
        Ok(())
    }
}
