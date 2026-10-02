// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! 2026-09-25: Marconi SSM snapshots taken during decode and at sequence retire, and their EP worker command.
//!
//! Owner: model-engine prefix cache.
//! Invariants:
//! - Rank 0 sends [`EP_CMD_DECODE_CKPT`] only after its own save was registered.
//! - The worker handler takes the checkpoint position from rank 0's payload.

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
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

// 2026-09-25: The wire protocol and the fire/skip decision, which need no GPU. Re-exported
// because `impl_a2.rs` and `snap_agree_tests.rs` name them as `decode_checkpoint::X`.
mod plan;
pub(in crate::model) use plan::*;
// 2026-10-01: Which snapshot families are saved (`METRALE_MARCONI_PREFILL_ONLY`).
mod policy;
use policy::{decode_ckpt_enabled, snapshot_policy};

impl TransformerModel {
    /// 2026-09-25: Save a Marconi SSM snapshot during decode, each time `seq.tokens` completes
    /// a multiple of `interval` KV blocks ([`decode_ckpt_plan`]), so the next turn's warm hit
    /// can restore near the end of this turn instead of replaying its decode tokens.
    /// The caller must call it after the step's SSM state is committed.
    pub(super) fn decode_marconi_checkpoint_dispatch(&self, seq: &mut SequenceState) {
        // 2026-10-01: `METRALE_MARCONI_PREFILL_ONLY=1` turns decode checkpoints off; rank 0
        // then sends no `EP_CMD_DECODE_CKPT` either (`policy`).
        let enabled = decode_ckpt_enabled(
            self.ssm_snapshots.is_enabled(),
            self.prefix_cache.is_active(),
            snapshot_policy(),
        );
        // 2026-09-25: The cheap preconditions run before the env read and the KV lock.
        if !ckpt_preconditions(
            enabled,
            self.config.num_ssm_layers(),
            seq.hss_window_start(),
            seq.slot_idx,
        ) {
            return;
        }
        // 2026-09-25: KV blocks between decode checkpoints. A positive `METRALE_DECODE_CKPT_BLOCKS`
        // overrides the default of 4.
        let interval = std::env::var("METRALE_DECODE_CKPT_BLOCKS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(4);
        let block_size = self.kv_cache.lock().block_size();
        let Some(plan) = decode_ckpt_plan(&CkptInputs {
            enabled,
            num_ssm_layers: self.config.num_ssm_layers(),
            hss_window_start: seq.hss_window_start(),
            slot_idx: seq.slot_idx,
            tokens_len: seq.tokens.len(),
            block_size,
            block_table_len: seq.block_table.len(),
            last_ckpt_block: seq.last_decode_ckpt_block,
            interval,
        }) else {
            return;
        };
        // 2026-09-25: The registered prefix is the full token slice, so vision-pad is checked
        // over the full slice.
        if self.tokens_have_vision_pad(&seq.tokens) {
            return;
        }
        let session_hash = seq.session_hash;
        let adapter_id = seq.adapter_id;
        if !self.decode_ckpt_save_and_register(seq, plan, session_hash, adapter_id, "decode-ckpt") {
            // 2026-09-25: Nothing was saved here, so no worker is asked to save one.
            return;
        }
        // 2026-09-25: Ask every worker rank to save the same (slot, token, session) checkpoint.
        // A broadcast failure is logged and the ranks may then hold different checkpoints.
        if let Err(e) = self.ep_broadcast_decode_ckpt(plan, session_hash, adapter_id, seq.slot_idx)
        {
            tracing::warn!("A109 decode-ckpt broadcast failed (ranks may diverge): {e:#}");
        }
    }

    /// 2026-09-25: Head side of [`EP_CMD_DECODE_CKPT`]. A no-op when the multi-rank protocol is
    /// inactive and on every rank but 0.
    fn ep_broadcast_decode_ckpt(
        &self,
        plan: CkptPlan,
        session_hash: u64,
        adapter_id: u64,
        slot_idx: usize,
    ) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        // 2026-09-25: Only rank 0 writes the command stream.
        if self.comm.as_ref().map(|c| c.rank()) != Some(0) {
            return Ok(());
        }
        // 2026-09-25: Under v2 the preamble routes the command to the worker's matching slot:
        // the worker's alloc-slot handler bails if its SSM-pool slot differs from the head's
        // seq_id, so `slot_idx` is the seq_id. Under v1 there is no preamble and the worker
        // targets slot 0.
        self.ep_broadcast_seq_and_cmd(slot_idx as u32, EP_CMD_DECODE_CKPT, self.ep_protocol_v2)?;
        self.ep_broadcast_tokens(&encode_ckpt_payload(plan, session_hash, adapter_id))?;
        Ok(())
    }

    /// 2026-09-25: Worker side of [`EP_CMD_DECODE_CKPT`]: save the checkpoint rank 0 saved, at
    /// the position rank 0 chose. The worker does not re-derive the decision, because the
    /// cadence and the session are head-side state.
    ///
    /// Errors when the payload is malformed, this rank has no SSM slot, or its sequence is
    /// shorter than the payload. A failed local save is logged, not returned.
    ///
    /// `session_hash` is taken from the head. This rank's own lookups pass the session hash
    /// its sequence was allocated with, 0, and `session_matches` accepts any tag for 0.
    /// `adapter_id` stays rank-local: it keys `hash_token_prefix`, so the head's value would
    /// register the entry where this rank's own lookup never searches.
    pub(in crate::model) fn decode_marconi_checkpoint_worker(
        &self,
        seq: &mut SequenceState,
        words: &[u32],
    ) -> Result<()> {
        let (plan, head_session, head_adapter) = decode_ckpt_payload(words)?;
        if seq.slot_idx == usize::MAX {
            bail!("A109 decode-ckpt: rank 0 checkpointed a slot this rank has no SSM state for");
        }
        if plan.snap_tokens > seq.tokens.len() || plan.end_block > seq.block_table.len() {
            bail!(
                "A109 decode-ckpt: head asked for snap_tokens={} end_block={} but this rank has \
                 tokens={} blocks={} — head and worker have diverged",
                plan.snap_tokens,
                plan.end_block,
                seq.tokens.len(),
                seq.block_table.len(),
            );
        }
        if head_adapter != seq.adapter_id {
            // 2026-09-25: Only the scheduler sets `adapter_id`, so a worker keeps the value its
            // sequence was allocated with.
            tracing::debug!(
                "A109 decode-ckpt: head adapter_id={head_adapter} != local {} — registering \
                 under the local id so this rank's own lookup can find it",
                seq.adapter_id,
            );
        }
        let adapter_id = seq.adapter_id;
        if !self.decode_ckpt_save_and_register(seq, plan, head_session, adapter_id, "decode-ckpt/w")
        {
            tracing::warn!(
                "A109 decode-ckpt: rank 0 saved (slot={} tok={}) but this rank could not — the \
                 A100 vote will refuse the restore (correct, slower)",
                seq.slot_idx,
                plan.snap_tokens,
            );
        }
        Ok(())
    }

    /// 2026-09-25: The save, shared by the head and the worker. Returns whether a checkpoint was
    /// registered.
    ///
    /// `session_hash` and `adapter_id` are parameters because the worker registers the head's
    /// session tag.
    fn decode_ckpt_save_and_register(
        &self,
        seq: &mut SequenceState,
        plan: CkptPlan,
        session_hash: u64,
        adapter_id: u64,
        who: &str,
    ) -> bool {
        // 2026-09-25: The default stream waits on the secondary-stream event, so the snapshot
        // reads the state a verify rollback on the secondary stream wrote.
        let _ = self.sync_secondary_dispatch();
        let stream = self.gpu.default_stream();
        let mut kv = self.kv_cache.lock();
        let bs = kv.block_size();
        let snap_id = match self.ssm_snapshots.save(
            seq.slot_idx,
            session_hash,
            self.seq_ssm_h_is_f16(seq),
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        ) {
            Ok(Some(id)) => id,
            Ok(None) => {
                if self.ssm_snapshots.reclaim_from_cache(
                    self.prefix_cache.as_ref(),
                    &mut kv,
                    self.ssm_tier_store.as_deref(),
                    self.gpu.as_ref(),
                ) {
                    match self.ssm_snapshots.save(
                        seq.slot_idx,
                        session_hash,
                        self.seq_ssm_h_is_f16(seq),
                        &self.ssm_pool,
                        self.gpu.as_ref(),
                        stream,
                    ) {
                        Ok(Some(id)) => id,
                        _ => return false,
                    }
                } else {
                    return false;
                }
            }
            Err(e) => {
                tracing::warn!("{who} Marconi checkpoint save error: {e}");
                return false;
            }
        };
        // 2026-09-25: Order any later warm restore (prefill stream) after this save's D2D.
        if let Err(e) = self.record_snapshot_save_dispatch(stream) {
            tracing::warn!("{who} Marconi checkpoint: record snapshot event: {e}");
        }
        drop(kv);
        // 2026-09-25: A failed aux collection is logged and leaves the snapshot without aux.
        // Models that need aux decline such a snapshot at restore.
        match self.collect_aux_states(seq, stream) {
            Ok(aux) => {
                if !aux.is_empty() {
                    self.ssm_snapshots.set_aux(snap_id, aux);
                }
            }
            Err(e) => tracing::warn!("{who} Marconi checkpoint: aux collect failed: {e:#}"),
        }
        // 2026-09-25: Registered at `snap_tokens`, the tokens the saved state covers, which can
        // be up to `bs - 1` past `end_token`. Registering at `end_token` would make a restore
        // replay tokens the state already holds through the GDN recurrence.
        let CkptPlan {
            snap_tokens,
            end_block,
        } = plan;
        let end_token = end_block * bs;
        let boundary_tokens = &seq.tokens[..snap_tokens];
        let boundary_blocks = &seq.block_table[..end_block];
        let boundary_disk: &[u32] = if seq.disk_block_ids.len() >= end_block {
            &seq.disk_block_ids[..end_block]
        } else {
            &[]
        };
        let displaced = self.prefix_cache.insert_intermediate_snapshot(
            boundary_tokens,
            boundary_blocks,
            boundary_disk,
            bs,
            snap_id,
            session_hash,
            snap_tokens,
            adapter_id,
        );
        if let Some(old) = displaced {
            self.ssm_snapshots.free(old);
        }
        tracing::info!(
            "{who} SAVE: snap_tokens={snap_tokens} end_block={end_block} snap_id={snap_id} \
             block_table_len={} straddle={} slot={} session={session_hash:#x}",
            seq.block_table.len(),
            snap_tokens.saturating_sub(end_token),
            seq.slot_idx,
        );
        if std::env::var("METRALE_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("decode_ckpt_save snap={snap_id} tok={snap_tokens}"),
            );
        }
        seq.last_decode_ckpt_block = end_block;
        true
    }

    /// 2026-09-25: Save the finish-leaf SSM snapshot at sequence retire, covering prompt and
    /// generated tokens, so the next warm turn does not replay this turn's decode tokens.
    /// Called by `cache_sequence_dispatch` before the radix insert. Returns `None` when the
    /// model has no SSM layers, the sequence has no slot, or the save fails. No hidden row is
    /// stashed, so an exact hit on this snapshot is declined at restore.
    pub(super) fn finish_leaf_snapshot(&self, seq: &SequenceState) -> Option<usize> {
        // 2026-10-01: `METRALE_MARCONI_PREFILL_ONLY=1` turns the finish leaf off; the caller
        // then inserts the tokens without a snapshot (`policy`).
        if !snapshot_policy().saves_finish_leaf() {
            return None;
        }
        if self.config.num_ssm_layers() == 0 || seq.slot_idx == usize::MAX {
            return None;
        }
        // 2026-09-25: A verify rollback can still be restoring the live h/conv state on the
        // secondary stream (`start_rollback_and_checkpoint_async_dispatch` records an event
        // rather than waiting). The default stream waits on that event, so the snapshot does
        // not capture state that still holds a rejected draft token.
        let _ = self.sync_secondary_dispatch();
        let stream = self.gpu.default_stream();
        let saved = match self.ssm_snapshots.save(
            seq.slot_idx,
            seq.session_hash,
            self.seq_ssm_h_is_f16(seq),
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        ) {
            Ok(Some(id)) => Some(id),
            Ok(None) => {
                if self.ssm_snapshots.reclaim_from_cache(
                    self.prefix_cache.as_ref(),
                    &mut self.kv_cache.lock(),
                    self.ssm_tier_store.as_deref(),
                    self.gpu.as_ref(),
                ) {
                    let retry = self.ssm_snapshots.save(
                        seq.slot_idx,
                        seq.session_hash,
                        self.seq_ssm_h_is_f16(seq),
                        &self.ssm_pool,
                        self.gpu.as_ref(),
                        stream,
                    );
                    retry.ok().flatten()
                } else {
                    None
                }
            }
            Err(e) => {
                tracing::warn!("finish-leaf SSM snapshot save error: {e}");
                None
            }
        };
        if let Some(id) = saved {
            // 2026-09-25: Order any later warm restore (prefill stream) after this save.
            if let Err(e) = self.record_snapshot_save_dispatch(stream) {
                tracing::warn!("finish-leaf snapshot: record snapshot event: {e}");
            }
            // 2026-09-25: A failed aux collection is logged and leaves the snapshot without aux.
            // Models that need aux decline such a snapshot at restore.
            match self.collect_aux_states(seq, stream) {
                Ok(aux) => {
                    if !aux.is_empty() {
                        self.ssm_snapshots.set_aux(id, aux);
                    }
                }
                Err(e) => tracing::warn!("finish-leaf snapshot: aux collect failed: {e:#}"),
            }
            tracing::info!(
                "Saved finish-leaf SSM snapshot {} for {} tokens",
                id,
                seq.tokens.len(),
            );
            if std::env::var("METRALE_SSM_SAVE_DUMP").is_ok() {
                self.ssm_pool.debug_state_checksum(
                    seq.slot_idx,
                    self.gpu.as_ref(),
                    stream,
                    &format!("finish_leaf_save snap={id} tok={}", seq.tokens.len()),
                );
            }
        }
        saved
    }
}
