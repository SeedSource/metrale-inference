// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MTP propose entry points (single-sequence, with the EP worker handshake, and
//! batched across sequences) and the batched drafter catch-up of `mtp_kv_exact`.
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
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::traits::{ModelDraft, ModelEp};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn run_mtp_propose_dispatch(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        let drafts = self.run_mtp_propose_multi(token, position, 1, seq, 0, None)?;
        Ok(drafts.into_iter().next())
    }

    pub(super) fn run_mtp_propose_multi_dispatch(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        _stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        // 2026-09-25: Whether the worker rank joins the propose is a property of the proposer
        // (`DraftProposer::needs_comm`). A proposer that loads every expert on every rank
        // proposes on rank 0 alone. The GLM-5.3 MTP block is EP-sharded, so while
        // `mtp_ep_propose_enabled` (on unless `METRALE_NO_MTP_EP_PROPOSE=1`) the head sends
        // the worker `EP_CMD_MTP_PROPOSE` and both ranks run the same propose.
        //
        // The command and its words go on the wire before this rank enters the drafter
        // forward, or the worker is still waiting for a command when rank 0 reaches its
        // first all-reduce.
        if self.multi_rank_protocol_active()
            && self.proposer.as_ref().is_some_and(|p| p.needs_comm())
        {
            self.ep_broadcast_cmd_for_seq(
                seq.slot_idx as u32,
                metrale_model_layers::speculative::EP_CMD_MTP_PROPOSE,
            )?;
            self.ep_broadcast_u32(token)?;
            self.ep_broadcast_u32(position as u32)?;
            self.ep_broadcast_u32(num_drafts as u32)?;
            // 2026-09-25: The fourth word is the `hidden_states` row the head last saved into
            // `mtp_hidden_save` (latched by `save_hidden_for_mtp_dispatch`). The propose reads
            // its target hidden from `mtp_hidden_save`, and the worker saves the same row
            // before proposing, so both ranks reduce partials of the same input.
            // `save_hidden_for_mtp_from_stash_dispatch` does not latch this index.
            self.ep_broadcast_u32(
                self.last_mtp_hidden_idx
                    .load(std::sync::atomic::Ordering::Relaxed) as u32,
            )?;
        }
        self.run_mtp_propose_inner(token, position, num_drafts, seq, grammar_bitmask)
    }

    /// 2026-09-25: Batched cross-sequence propose. Target hiddens are read from the verify
    /// stash rows (`stash_idx[i]`), not from `mtp_hidden_save`. `Ok(None)` tells the caller
    /// to fall back to the per-sequence propose: no proposer, the draft-confidence clamp
    /// armed, no stash, a sequence without proposer state, or `propose_batch` unsupported.
    pub(super) fn run_mtp_propose_batched_dispatch(
        &self,
        tokens: &[u32],
        positions: &[usize],
        stash_idx: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(None),
        };
        // 2026-10-04: A proposer whose batch mirrors its per-sequence propose
        // (`METRALE_GLM_MTP_BATCH_DRAFT=1`) takes the mirrored path, with the per-sequence
        // context and, multi-rank, its own worker command (`propose_batch_ep.rs`).
        let mirrors = proposer.propose_batch_mirrors_serial();
        // 2026-10-02: Its forward runs without the communicator, so an EP-sharded drafter
        // (`needs_comm`) on a multi-rank serve proposes per sequence, through the worker
        // handshake of `run_mtp_propose_multi_dispatch`.
        if !mirrors && self.multi_rank_protocol_active() && proposer.needs_comm() {
            return Ok(None);
        }
        if self.levers.draft_conf_tau > 0.0 {
            return Ok(None);
        }
        if self.verify_hidden_stash.is_null() {
            return Ok(None);
        }
        if mirrors {
            return self.run_mtp_propose_batched_mirrored(
                proposer, tokens, positions, stash_idx, num_drafts, seqs, out_conf,
            );
        }
        let stream = self.gpu.default_stream();
        let ctx = self.mtp_propose_ctx();
        // 2026-09-25: Drafter context on each sequence's first propose; returns without
        // work on later calls.
        for seq in seqs.iter_mut() {
            self.ensure_drafter_context(proposer, seq, &ctx, stream);
        }
        let h = self.config.hidden_size;
        let hiddens: Vec<metrale_gpu_runtime::gpu::DevicePtr> = stash_idx
            .iter()
            .map(|&i| self.verify_hidden_stash.offset(i * h * 2))
            .collect();
        let mut states: Vec<&mut dyn metrale_model_layers::speculative::ProposerState> = Vec::new();
        for seq in seqs.iter_mut() {
            match seq.proposer_state.as_mut() {
                Some(s) => states.push(s.as_mut()),
                None => return Ok(None),
            }
        }
        proposer.propose_batch(
            tokens,
            &hiddens,
            positions,
            num_drafts,
            &mut states,
            &ctx,
            stream,
            out_conf,
        )
    }

    /// 2026-09-25: The `ForwardContext` of the batched MTP propose and the batched catch-up.
    pub(super) fn mtp_propose_ctx(&self) -> ForwardContext<'_> {
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            moe_lora_route: self.decode_moe_route(),
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
        }
    }

    /// 2026-09-25: `ModelLevers::mtp_kv_exact`: copy verify-forward `hidden_states` rows into
    /// catch-up stash slots (`slot_rows[j] = (slot, row)`) before anything overwrites them.
    /// A no-op when the lever is off or the stash is not allocated.
    pub(super) fn stash_verify_catchup_rows_dispatch(
        &self,
        slot_rows: &[(usize, usize)],
    ) -> Result<()> {
        if !self.levers.mtp_kv_exact || self.verify_catchup_stash.is_null() {
            return Ok(());
        }
        let cap = metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS
            * metrale_model_layers::layer::MTP_CATCHUP_MAX;
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        for &(slot, row) in slot_rows {
            anyhow::ensure!(
                slot < cap,
                "stash_verify_catchup_rows: slot {slot} >= {cap}"
            );
            let src = self.buffers.hidden_states().offset(row * h * 2);
            let dst = self.verify_catchup_stash.offset(slot * h * 2);
            self.gpu.copy_d2d_async(src, dst, h * 2, stream)?;
        }
        Ok(())
    }

    /// 2026-09-25: `ModelLevers::mtp_kv_exact`: append the drafter rows for every accepted
    /// draft. `tokens[i]` are sequence i's accepted drafts, their hiddens in catch-up stash
    /// slots `first_slot[i]..`, `first_pos[i]` the RoPE position of `tokens[i][0]`.
    /// Returns the rows written, 0 when the lever is off or nothing can run.
    pub(super) fn run_mtp_catchup_batched_dispatch(
        &self,
        tokens: &[Vec<u32>],
        first_slot: &[usize],
        first_pos: &[usize],
        seqs: &mut [&mut SequenceState],
    ) -> Result<usize> {
        if !self.levers.mtp_kv_exact || self.verify_catchup_stash.is_null() {
            return Ok(0);
        }
        let Some(proposer) = self.proposer.as_ref().map(|p| p.as_ref()) else {
            return Ok(0);
        };
        // 2026-10-02: No communicator in this forward either, and no worker mirror: an
        // EP-sharded drafter on a multi-rank serve skips the batched catch-up.
        if self.multi_rank_protocol_active() && proposer.needs_comm() {
            return Ok(0);
        }
        let h = self.config.hidden_size;
        let hiddens: Vec<Vec<metrale_gpu_runtime::gpu::DevicePtr>> = tokens
            .iter()
            .zip(first_slot)
            .map(|(t, &s)| {
                (0..t.len())
                    .map(|k| self.verify_catchup_stash.offset((s + k) * h * 2))
                    .collect()
            })
            .collect();
        let mut states: Vec<&mut dyn metrale_model_layers::speculative::ProposerState> = Vec::new();
        for seq in seqs.iter_mut() {
            match seq.proposer_state.as_mut() {
                Some(s) => states.push(s.as_mut()),
                None => return Ok(0),
            }
        }
        let ctx = self.mtp_propose_ctx();
        let stream = self.gpu.default_stream();
        proposer.catchup_batch(tokens, &hiddens, first_pos, &mut states, &ctx, stream)
    }
}
