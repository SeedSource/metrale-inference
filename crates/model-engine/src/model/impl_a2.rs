// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Vision-placeholder and high-speed-swap predicates, chunked
//! prefill metadata, and the EP worker's command dispatch.
//!
//! Owner: model-engine.
//! Invariants:
//! - `ep_worker_step_impl` returns a receive error unchanged and wraps every
//!   error from executing the command in `EpCommandFailed`.

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::traits::{ModelDraft, ModelForward, ModelLifecycle, ModelSsmState, ModelVerify};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn comm_ref(&self) -> Option<&dyn metrale_comm::CommBackend> {
        self.comm.as_deref()
    }

    /// 2026-09-25: The two vision placeholder token ids,
    /// `(<|image_pad|>, <|video_pad|>)`. Each falls back to the vision
    /// encoder's default when the checkpoint has no vision config or sets 0.
    pub(super) fn vision_pad_ids(&self) -> (u32, u32) {
        let v = self.config.vision.as_ref();
        let image = v
            .map(|v| v.image_pad_token_id)
            .filter(|id| *id != 0)
            .unwrap_or(metrale_model_layers::layers::vision_encoder::IMAGE_PAD_TOKEN_ID);
        let video = v
            .map(|v| v.video_pad_token_id)
            .filter(|id| *id != 0)
            .unwrap_or(metrale_model_layers::layers::vision_encoder::VIDEO_PAD_TOKEN_ID);
        (image, video)
    }

    /// 2026-09-25: Whether `tok` is a vision placeholder of either modality.
    /// Comparing against the image id alone would treat video pads as text:
    /// the prompt still tokenises, and the model never receives the pixels.
    pub(super) fn is_vision_pad(&self, tok: u32) -> bool {
        let (image, video) = self.vision_pad_ids();
        tok == image || tok == video
    }

    pub(super) fn tokens_have_vision_pad(&self, tokens: &[u32]) -> bool {
        let (image, video) = self.vision_pad_ids();
        tokens.iter().any(|&t| t == image || t == video)
    }

    /// 2026-09-25: Whether `--high-speed-swap` has slid this sequence's window
    /// (`hss_window_start() > 0`), so `block_table[i]` no longer holds logical
    /// block `i`.
    ///
    /// A prefix-cache insert files `block_table[i]` on the node for token chunk
    /// `i`, so after a slide it would file blocks under chunks whose KV they do
    /// not hold. The window then holds a mid-sequence suffix, and the tree
    /// indexes prefixes from the root, so a slid sequence has nothing to insert.
    pub(super) fn hss_window_slid(&self, seq: &SequenceState) -> bool {
        seq.hss_window_start() > 0
    }

    /// 2026-09-25: Free the pinned metadata staging buffer; called from `Drop`.
    pub(super) fn drop_pinned_staging(&self) {
        // 2026-09-25: SAFETY: the only caller is `Drop`, which has the model to
        // itself, so no other reference to the staging cell is live.
        let staging = unsafe { &*self.pinned_staging.get() };
        if !staging.ptr.is_null()
            && let Err(e) = self.gpu.free_host_pinned(staging.ptr, staging.bytes)
        {
            tracing::warn!("Failed to free pinned staging: {e}");
        }
    }

    pub(super) fn ensure_chunked_prefill_meta<'a>(
        &self,
        seq: &'a mut SequenceState,
        total_tokens: usize,
        block_size: usize,
    ) -> Result<&'a mut ChunkedPrefillPageMetadata> {
        let required_blocks = total_tokens.saturating_sub(1) / block_size + 1;
        if seq.chunked_prefill_meta.is_none() {
            seq.chunked_prefill_meta = Some(ChunkedPrefillPageMetadata {
                block_table: self.gpu.alloc(required_blocks.max(1) * 4)?,
                seq_len: self.gpu.alloc(std::mem::size_of::<u32>())?,
                block_capacity: required_blocks,
                uploaded_blocks: 0,
            });
        }

        let meta = seq.chunked_prefill_meta.as_mut().unwrap();
        if meta.block_capacity < required_blocks {
            bail!(
                "chunked prefill metadata capacity {} < required {} blocks",
                meta.block_capacity,
                required_blocks,
            );
        }
        Ok(meta)
    }

    pub(super) fn free_chunked_prefill_meta(&self, seq: &mut SequenceState) -> Result<()> {
        if let Some(meta) = seq.chunked_prefill_meta.take() {
            if !meta.block_table.is_null() {
                self.gpu.free(meta.block_table)?;
            }
            if !meta.seq_len.is_null() {
                self.gpu.free(meta.seq_len)?;
            }
        }
        Ok(())
    }

    /// 2026-09-25: EP worker step: receive a `(seq_id, cmd)` preamble from
    /// rank 0 and execute the command in the addressed slot. Returns false
    /// when the worker should shut down.
    ///
    /// With `METRALE_EP_PROTOCOL=v2` rank 0 broadcasts the slot id, then the
    /// command code, then the command's data. Otherwise there is no slot id and
    /// every command targets slot 0.
    ///
    /// Command codes:
    /// - 0xFFFFFFE0: batched decode → `N`, `seq_ids[N]`, `tokens[N]`
    /// - 0xFFFFFFF0: prefill chunk → chunk_len, chunk_start, full_len, then
    ///   full_len tokens, then, only when the model has a vision tower, the
    ///   merged vision-row count and that many rows when it is non-zero
    /// - 0xFFFFFFF1: alloc slot (frees any prior occupant, then allocates)
    /// - 0xFFFFFFF2/3/4: verify K=2/3/4 → K tokens, then the accept count
    /// - 0xFFFFFFF5 (`EP_CMD_MTP_PROPOSE`): last_token, position, num_drafts,
    ///   hidden_idx
    /// - 0xFFFFFFF6 (`EP_CMD_VERIFY_KGAMMA`): k, then k tokens in one bulk
    ///   broadcast, then the accept count (DFlash K=γ verify)
    /// - 0xFFFFFFF8 (`EP_CMD_DECODE_CKPT`): decode-time Marconi checkpoint →
    ///   `EP_CKPT_WORDS` words in one bulk broadcast
    /// - 0xFFFFFFF9 (`EP_CMD_VERIFY_BATCH`): batched MTP verify → `verify_ep.rs`
    /// - 0xFFFFFFFA (`EP_CMD_MTP_PROPOSE_BATCH`): batched MTP propose → `propose_batch_ep.rs`
    /// - 0xFFFFFFFF: shutdown (seq_id is ignored)
    /// - any other value: a token id, decoded in the addressed slot
    pub(super) fn ep_worker_step_impl(&self, slots: &mut [Option<SequenceState>]) -> Result<bool> {
        // 2026-09-25: Only the receive is fatal: if it fails, the link to the
        // head is gone and the worker must exit. An error while executing the
        // command is request-scoped, so it is tagged `EpCommandFailed` and the
        // worker loop keeps running.
        let (seq_id, cmd) = self.ep_recv_seq_and_cmd(self.ep_protocol_v2)?;
        self.ep_worker_execute(seq_id, cmd, slots)
            .map_err(|e| anyhow::Error::new(crate::traits::EpCommandFailed(e)))
    }

    /// 2026-09-25: Execute one received worker command. The caller tags every
    /// error from here as `EpCommandFailed`.
    fn ep_worker_execute(
        &self,
        seq_id: u32,
        cmd: u32,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        // 2026-09-25: Shutdown applies to the whole worker; seq_id is ignored.
        if cmd == 0xFFFFFFFF {
            return Ok(false);
        }

        // 2026-09-25: Batched decode: the preamble seq_id is 0, and the
        // per-row slots arrive in the `seq_ids[N]` payload the batched handler
        // reads.
        if cmd == 0xFFFFFFE0 {
            return self.ep_worker_decode_batch(slots);
        }

        // 2026-10-02: Batched MTP verify: the preamble seq_id is 0 and the per-row slots arrive
        // in its payload (`trait_impl/verify_ep.rs`).
        if cmd == crate::model::trait_impl::verify_ep::EP_CMD_VERIFY_BATCH {
            return self.ep_worker_verify_batch(slots);
        }

        // 2026-10-04: Batched MTP propose: the preamble seq_id is 0 and the per-row slots
        // arrive in its payload (`trait_impl/propose_batch_ep.rs`).
        if cmd == crate::model::trait_impl::propose_batch_ep::EP_CMD_MTP_PROPOSE_BATCH {
            return self.ep_worker_propose_batch(slots);
        }

        let slot_idx = seq_id as usize;
        if slot_idx >= slots.len() {
            anyhow::bail!(
                "ep_worker_step: seq_id {} exceeds slot capacity {} \
                 (head and worker likely disagree on max_batch_size)",
                seq_id,
                slots.len(),
            );
        }

        // 2026-09-25: Alloc slot: free the prior occupant, then allocate a
        // fresh sequence. Under v2 its SSM-pool slot must equal `slot_idx`:
        // both ranks pop the same free list in the same order, and a mismatch
        // is an error rather than a silent KV mix-up.
        if cmd == 0xFFFFFFF1 {
            if let Some(mut old) = slots[slot_idx].take() {
                self.free_sequence(&mut old)?;
            }
            let new_seq = self.alloc_sequence()?;
            if self.ep_protocol_v2 && new_seq.slot_idx != slot_idx {
                anyhow::bail!(
                    "ep_worker_step: SSM-pool slot {} doesn't match head's seq_id {} \
                     after alloc — claim_slot ordering invariant violated",
                    new_seq.slot_idx,
                    slot_idx,
                );
            }
            slots[slot_idx] = Some(new_seq);
            return Ok(true);
        }

        let seq = slots[slot_idx].as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "ep_worker_step: cmd {:#x} arrived for unallocated slot {} \
                 — head dispatched without a prior alloc",
                cmd,
                slot_idx,
            )
        })?;

        self.ep_worker_dispatch_cmd(cmd, seq)
    }

    /// 2026-09-25: Per-command dispatch for [`Self::ep_worker_step_impl`]. The
    /// caller has already handled the preamble, the slot lookup, shutdown,
    /// batched decode and alloc; `seq` is the addressed slot's sequence.
    fn ep_worker_dispatch_cmd(&self, cmd: u32, seq: &mut SequenceState) -> Result<bool> {
        let stream = self.gpu.default_stream();

        match cmd {
            0xFFFFFFF0 => {
                // 2026-09-25: Prefill chunk: chunk_len, chunk_start, the full prompt
                // length, then every prompt token in one bulk broadcast.
                let chunk_len = self.ep_broadcast_u32(0)? as usize;
                let chunk_start = self.ep_broadcast_u32(0)? as usize;
                let full_len = self.ep_broadcast_u32(0)? as usize;
                let full_tokens = self.ep_broadcast_tokens(&vec![0u32; full_len])?;
                // 2026-09-25: The merged vision rows, at the same point of the
                // head's send sequence, so the worker splices the same image rows
                // as rank 0. A no-op for a model without a vision tower.
                self.ep_sync_vision_embeds(&full_tokens)?;
                // 2026-09-25: `is_last` from the chunk bounds; it must equal rank
                // 0's value, since `prefill_chunk` branches on it.
                let is_last = chunk_start + chunk_len >= full_len;
                let _ =
                    self.prefill_chunk(&full_tokens, seq, chunk_start, chunk_len, is_last, stream)?;
                // 2026-09-25: Normalize the SSM states after the chunk, as the
                // head's scheduler does after a prefill chunk, so the ranks'
                // states stay identical. A failure is logged, not returned.
                if let Err(e) = self.normalize_ssm_states(seq, stream) {
                    tracing::warn!("Worker SSM state normalization failed: {e:#}");
                }
            }
            0xFFFFFFF2 => {
                // 2026-09-25: Verify K=2: 2 tokens, verify, then accept (1) or
                // reject.
                let t0 = self.ep_broadcast_u32(0)?;
                let t1 = self.ep_broadcast_u32(0)?;
                self.sync_secondary()?;
                self.decode_verify_graphed(&[t0, t1], seq, stream)?;
                let accepted = self.ep_broadcast_u32(0)?;
                if accepted == 1 {
                    self.start_checkpoint_async(seq)?;
                    self.trim_proposer_state(seq, 1, 0)?;
                } else {
                    seq.seq_len -= 1;
                    seq.tokens.pop();
                    self.trim_proposer_state(seq, 0, 0)?;
                    self.start_rollback_and_checkpoint_async(seq, 1)?;
                }
            }
            crate::model::trait_impl::decode_checkpoint::EP_CMD_DECODE_CKPT => {
                // 2026-09-25: Rank 0 saved a decode-time Marconi checkpoint;
                // save the same one here so a rank-agreed restore finds it on
                // every rank. The slot is the preamble's, the rest the payload.
                let words = self.ep_broadcast_tokens(
                    &[0u32; crate::model::trait_impl::decode_checkpoint::EP_CKPT_WORDS],
                )?;
                self.decode_marconi_checkpoint_worker(seq, &words)?;
            }
            metrale_model_layers::speculative::EP_CMD_MTP_PROPOSE => {
                // 2026-09-25: Run the same drafter forward as rank 0, so its
                // collectives have a partner. The drafts are discarded (rank 0
                // broadcasts the tokens it verifies), but the drafter KV this
                // writes stays in step because both ranks use the same
                // `(last_token, position)` and the same saved hidden row.
                let last_token = self.ep_broadcast_u32(0)?;
                let position = self.ep_broadcast_u32(0)? as usize;
                let num_drafts = self.ep_broadcast_u32(0)? as usize;
                let hidden_idx = self.ep_broadcast_u32(0)? as usize;
                // 2026-09-25: Save the hidden row the head saved
                // (`save_hidden_for_mtp`), so both ranks feed the drafter the same
                // input and its all-reduce sums partials of one vector.
                // 2026-10-02: With `MTP_HIDDEN_FROM_STASH` set the head loaded batched-verify
                // stash slot `hidden_idx & !flag` (`verify_ep.rs`), which this rank stashed too.
                let flag = crate::model::trait_impl::verify_ep::MTP_HIDDEN_FROM_STASH as usize;
                let saved = if hidden_idx & flag != 0 {
                    self.save_hidden_for_mtp_from_stash(hidden_idx & !flag, stream)
                } else {
                    self.save_hidden_for_mtp(hidden_idx, stream)
                };
                if let Err(e) = saved {
                    tracing::warn!("EP worker save_hidden_for_mtp({hidden_idx}) failed: {e:#}");
                }
                if let Err(e) =
                    self.run_mtp_propose_inner(last_token, position, num_drafts, seq, None)
                {
                    // 2026-09-25: A drafter error is logged, not returned: the
                    // worker's drafts are discarded, and rank 0 decides what is
                    // verified.
                    tracing::warn!("EP worker MTP propose failed (continuing): {e:#}");
                }
            }
            0xFFFFFFF3 => {
                // 2026-09-25: Verify K=3: 3 tokens, verify, then num_accepted (0..=2).
                let t0 = self.ep_broadcast_u32(0)?;
                let t1 = self.ep_broadcast_u32(0)?;
                let t2 = self.ep_broadcast_u32(0)?;
                self.sync_secondary()?;
                self.decode_verify_graphed_k3(&[t0, t1, t2], seq, stream)?;
                let num_accepted = self.ep_broadcast_u32(0)?;
                self.trim_proposer_state(seq, num_accepted as usize, 0)?;
                match num_accepted {
                    2 => {
                        self.start_checkpoint_async(seq)?;
                    }
                    1 => {
                        seq.seq_len -= 1;
                        seq.tokens.pop();
                        self.start_rollback_and_checkpoint_async(seq, 2)?;
                    }
                    _ => {
                        seq.seq_len -= 2;
                        seq.tokens.pop();
                        seq.tokens.pop();
                        self.start_rollback_and_checkpoint_async(seq, 1)?;
                    }
                }
            }
            0xFFFFFFF4 => {
                // 2026-09-25: Verify K=4: 4 tokens, verify, then num_accepted (0..=3).
                let t0 = self.ep_broadcast_u32(0)?;
                let t1 = self.ep_broadcast_u32(0)?;
                let t2 = self.ep_broadcast_u32(0)?;
                let t3 = self.ep_broadcast_u32(0)?;
                self.sync_secondary()?;
                self.decode_verify_graphed_k4(&[t0, t1, t2, t3], seq, stream)?;
                let num_accepted = self.ep_broadcast_u32(0)?;
                self.trim_proposer_state(seq, num_accepted as usize, 0)?;
                match num_accepted {
                    3 => {
                        self.start_checkpoint_async(seq)?;
                    }
                    2 => {
                        seq.seq_len -= 1;
                        seq.tokens.pop();
                        self.start_rollback_and_checkpoint_async(seq, 3)?;
                    }
                    1 => {
                        seq.seq_len -= 2;
                        seq.tokens.pop();
                        seq.tokens.pop();
                        self.start_rollback_and_checkpoint_async(seq, 2)?;
                    }
                    _ => {
                        seq.seq_len -= 3;
                        seq.tokens.pop();
                        seq.tokens.pop();
                        seq.tokens.pop();
                        self.start_rollback_and_checkpoint_async(seq, 1)?;
                    }
                }
            }
            metrale_model_layers::speculative::glm_dflash::EP_CMD_VERIFY_KGAMMA => {
                // 2026-10-01: DFlash K=γ verify (`step_verify_dflash` on rank 0): `k`, the `k`
                // tokens in one bulk broadcast, the verify, then `num_accepted`. The worker
                // mirrors rank 0's bookkeeping: keep the pre-verify prefix + anchor + accepted
                // drafts, and commit the SSM state to that row. Its drafter state is unused
                // (the DFlash drafter proposes on rank 0 alone), so it is not trimmed.
                use metrale_model_layers::speculative::glm_dflash::{KGAMMA_MAX_K, kgamma_keep_len};
                let k = self.ep_broadcast_u32(0)? as usize;
                if k == 0 || k > KGAMMA_MAX_K {
                    bail!("EP_CMD_VERIFY_KGAMMA: k={k} outside 1..={KGAMMA_MAX_K}");
                }
                let tokens = self.ep_broadcast_tokens(&vec![0u32; k])?;
                self.sync_secondary()?;
                self.decode_verify_graphed_kgamma(&tokens, seq, stream)?;
                let num_accepted = self.ep_broadcast_u32(0)? as usize;
                let keep = kgamma_keep_len(seq.seq_len, k, num_accepted)?;
                let drop = seq.seq_len - keep;
                seq.seq_len = keep;
                for _ in 0..drop.min(seq.tokens.len()) {
                    seq.tokens.pop();
                }
                self.commit_accepted_prefix(seq, num_accepted + 1, k)?;
                // 2026-10-01: The commit runs on the secondary stream; order it before whatever
                // this rank runs next (the next verify syncs too, a plain decode does not).
                self.sync_secondary()?;
            }
            token => {
                self.decode(token, seq, stream)?;
            }
        }

        Ok(true)
    }

    /// 2026-09-25: Worker side of the batched decode (`0xFFFFFFE0`). Reads `N`,
    /// `seq_ids[N]` and `tokens[N]` as `ep_broadcast_decode_batch_dispatch`
    /// sends them, orders the addressed slots as `seq_ids`, and runs
    /// `decode_batch_compute_main`, the compute the head runs after sending,
    /// so the per-layer collectives pair up.
    ///
    /// Out-of-range and duplicate seq_ids are rejected before any slot is
    /// touched.
    fn ep_worker_decode_batch(&self, slots: &mut [Option<SequenceState>]) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)? as usize;
        let seq_ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let tokens = self.ep_broadcast_tokens(&vec![0u32; n])?;

        let mut seen = std::collections::HashSet::new();
        for &id in &seq_ids {
            let idx = id as usize;
            if idx >= slots.len() {
                anyhow::bail!(
                    "ep_worker_decode_batch: seq_id {} exceeds slot capacity {}",
                    id,
                    slots.len(),
                );
            }
            if !seen.insert(id) {
                anyhow::bail!("ep_worker_decode_batch: duplicate seq_id {} in batch", id);
            }
        }

        // 2026-09-25: Collect `(idx, &mut)` for the populated slots and take
        // them out with `swap_remove`: indexing `slots[seq_ids[i]]` mutably in
        // a loop does not borrow-check, since the compiler cannot prove the
        // indices distinct.
        let mut slot_refs: Vec<(usize, &mut SequenceState)> = slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, opt)| opt.as_mut().map(|s| (i, s)))
            .collect();

        // 2026-09-25: Order the refs as the head's seq_ids, so batch row `i` is
        // the same sequence on both ranks.
        let mut refs: Vec<&mut SequenceState> = Vec::with_capacity(n);
        for &id in &seq_ids {
            let idx = id as usize;
            let pos = slot_refs
                .iter()
                .position(|(i, _)| *i == idx)
                .ok_or_else(|| {
                    anyhow::anyhow!("ep_worker_decode_batch: slot {} not allocated", idx)
                })?;
            let (_, seq) = slot_refs.swap_remove(pos);
            refs.push(seq);
        }

        let stream = self.gpu.default_stream();
        self.decode_batch_compute_main(&tokens, &mut refs, stream)?;
        Ok(true)
    }
}
