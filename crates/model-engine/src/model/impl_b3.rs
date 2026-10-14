// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Draft proposal and DFlash hidden-state capture on `TransformerModel`.
//!
//! Owner: model-engine.
//! Invariants:
//! - The DFlash capture and context-length methods do nothing on a comm rank other than 0.
//! - DFlash prefill capture keeps the last `max_ctx_len` prompt positions
//!   (`PrefillCtxWindow`, 2026-10-10; the first ones before), never more, and
//!   `try_dflash_capture_all_at` never writes past `dflash_hidden_save_rows`.

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
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
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn run_mtp_propose_inner(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        // 2026-10-10: The step's proposer (`hybrid_spec.rs`): on a hybrid serve's DFLASH step the
        // DFlash drafter and its state, else `proposer` and `proposer_state` as before.
        let hybrid_dflash = self.hybrid_on() && self.dflash_step();
        let proposer = match self.step_proposer() {
            Some(p) => p.as_ref(),
            None => return Ok(Vec::new()),
        };
        // 2026-09-25: With the `dflash_debug_dump_full` lever
        // (METRALE_DFLASH_DEBUG_DUMP_FULL=1), write this sequence's tokens to
        // /tmp/metrale_tokens.json once per model. The lever is tested before
        // the per-model latch so a disabled dump does not consume it.
        if self.levers.dflash_debug_dump_full && self.stats.dumped.keyed("dump:dflash_tokens") {
            let tokens_json = serde_json::json!({
                "prompt_len": position - seq.tokens.len() + seq.tokens.len(),
                "position": position,
                "last_token": token,
                "all_tokens": seq.tokens.clone(),
                "generated_tokens": seq.tokens.iter().skip(seq.prompt_len).copied().collect::<Vec<u32>>(),
            });
            if let Err(e) = std::fs::write(
                "/tmp/metrale_tokens.json",
                serde_json::to_string_pretty(&tokens_json).unwrap_or_default(),
            ) {
                tracing::warn!("DFLASH DUMP_FULL: tokens write failed: {e}");
            } else {
                tracing::info!(
                    "DFLASH DUMP_FULL: wrote /tmp/metrale_tokens.json (position={}, all_tokens.len()={}, prompt_len={})",
                    position,
                    seq.tokens.len(),
                    seq.prompt_len,
                );
            }
        }
        let stream = self.gpu.default_stream();
        let draft_embed_target = None;
        // 2026-09-25: The drafter gets the communicator only when its
        // `needs_comm()` reports a block sharded across ranks. See
        // `DraftProposer::needs_comm` for which drafters that is, and why the
        // others must not get one.
        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: if proposer.needs_comm() {
                self.comm_ref()
            } else {
                None
            },
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };
        // 2026-09-25: Give the drafter its prompt context on the first
        // propose of this sequence: a whole-prompt drafter prefill on a cold
        // turn or, with the carry lever on, the previous turn's drafter rows
        // plus an append on a warm one. See `ensure_drafter_context`.
        // 2026-10-10: Skipped on a hybrid DFLASH step: it prefills the MTP head from the MTP
        // capture (the eager consume at end of prefill already did on a hybrid serve).
        if !hybrid_dflash {
            self.ensure_drafter_context(proposer, seq, &ctx, stream);
            // 2026-10-14: The slot's first MTP propose after a hybrid DFlash stretch: catch the
            // drafter up (`METRALE_HYBRID_MTP_CATCHUP`) and log `HYBRID MTP resume`. The worker
            // runs this propose too (`EP_CMD_MTP_PROPOSE`), with the same `position`.
            self.hybrid_mtp_resume(proposer, seq, position, stream);
        }
        let prop_state = if hybrid_dflash {
            seq.dflash_state.as_mut()
        } else {
            seq.proposer_state.as_mut()
        }
        .ok_or_else(|| anyhow::anyhow!("No proposer state for sequence"))?;
        // 2026-09-25: METRALE_MTP_CATCHUP: before proposing, feed the drafter
        // the pairs it missed during serial decode. Pair key k is
        // (embed(tokens[k+1]), hidden_k) at RoPE k+1. The serial-decode hook
        // writes each step's hidden under the post-step `seq_len`, so ring
        // label n holds hidden_{n-1} and pair key k reads label k+1.
        // 2026-10-10: MTP head only, so never on a hybrid DFLASH step.
        if !hybrid_dflash
            && metrale_model_layers::speculative::mtp_catchup_enabled()
            && !self.mtp_catchup_ring.is_null()
        {
            let rows = proposer.drafter_rows(prop_state.as_mut());
            let last_key = proposer.last_pair_key(prop_state.as_mut());
            let (start, count) = *self.mtp_catchup_meta.lock();
            // 2026-09-25: METRALE_MTP_REFEED_DEBUG: log whether the ring row
            // at label `position` matches `mtp_hidden_save`. This checks the
            // ring's slot arithmetic, not the label convention (see
            // `mtp_refeed_shift`).
            if metrale_model_layers::speculative::mtp_refeed_debug() {
                let ring_rows = super::types::MTP_CATCHUP_RING_ROWS;
                let h = self.config.hidden_size;
                let fp_save = metrale_model_layers::speculative::hidden_fingerprint(
                    self.gpu.as_ref(),
                    self.mtp_hidden_save,
                    h,
                );
                let fp_ring = metrale_model_layers::speculative::hidden_fingerprint(
                    self.gpu.as_ref(),
                    self.mtp_catchup_ring.offset((position % ring_rows) * h * 2),
                    h,
                );
                let covered = count > 0 && position >= start && position < start + count;
                tracing::info!(
                    "REFEED_DBG propose position={position} rows={rows} \
                     last_key={last_key:?} ring=[{start},+{count}) \
                     fp_save={fp_save:016x} fp_ring[{position}]={fp_ring:016x} \
                     covered={covered} roundtrip_ok={}",
                    covered && fp_save == fp_ring,
                );
            }
            if let Some(last) = last_key
                && rows > 0
                && count > 0
            {
                // 2026-09-25: Feed the missing pair keys last+1 ..= position-2,
                // clipped to ring coverage [start, start+count) in label space
                // (label = key + 1).
                let mut k0 = (last + 1).max(start.saturating_sub(1));
                let k1 = (position.saturating_sub(2)).min((start + count).saturating_sub(2));
                let want = (position.saturating_sub(1)).saturating_sub(last + 1);
                if k0 <= k1 && want > 0 {
                    let ring_rows = super::types::MTP_CATCHUP_RING_ROWS;
                    let h = self.config.hidden_size;
                    let bf16 = 2usize;
                    let fed_from = k0;
                    while k0 <= k1 {
                        // 2026-09-25: Ring-contiguous segment: labels k0+1 .. until wrap.
                        let slot = (k0 + 1) % ring_rows;
                        let seg_last = k1.min(k0 + (ring_rows - slot) - 1);
                        let n_rows = seg_last - k0 + 1;
                        // 2026-09-25: Row r feeds pair key k0+r = embed(tokens[k0+r+1]):
                        // the drafter reads tokens[r+1] for row r, so pass the
                        // window starting at index k0 (n_rows + 1 tokens).
                        let toks = &seq.tokens[k0..=seg_last + 1];
                        let hid = self.mtp_catchup_ring.offset(slot * h * bf16);
                        if metrale_model_layers::speculative::mtp_refeed_debug() {
                            for r in 0..n_rows {
                                let fp = metrale_model_layers::speculative::hidden_fingerprint(
                                    self.gpu.as_ref(),
                                    hid.offset(r * h * bf16),
                                    h,
                                );
                                tracing::info!(
                                    "REFEED_DBG feed key={} label={} tok={} rope={} fp={fp:016x}",
                                    k0 + r,
                                    k0 + r + 1,
                                    toks[r + 1],
                                    k0 + r + 1,
                                );
                            }
                        }
                        let row_base = proposer.drafter_rows(prop_state.as_mut());
                        match proposer.catchup_drafter(
                            toks,
                            hid,
                            row_base,
                            k0 + 1,
                            prop_state.as_mut(),
                            &ctx,
                            stream,
                        ) {
                            Ok(w) if w == n_rows => k0 = seg_last + 1,
                            Ok(w) => {
                                tracing::debug!(
                                    "MTP catch-up: short feed ({w}/{n_rows} rows) — degrading"
                                );
                                break;
                            }
                            Err(e) => {
                                tracing::debug!("MTP catch-up: feed failed ({e:#}) — degrading");
                                break;
                            }
                        }
                    }
                    if k0 > k1 {
                        tracing::debug!(
                            "MTP catch-up: fed pair keys {fed_from}..={k1} \
                             (missed {want}, position {position})"
                        );
                    }
                } else if want > 0 {
                    tracing::debug!(
                        "MTP catch-up: gap of {want} pairs outside ring coverage \
                         (last_key={last} position={position} ring=[{start},+{count}))"
                    );
                }
            }
        }
        let drafts = proposer.propose(
            token,
            self.mtp_hidden_save,
            position,
            num_drafts,
            prop_state.as_mut(),
            &ctx,
            stream,
            draft_embed_target,
            grammar_bitmask,
            self.dflash_hidden_save,
        )?;
        // 2026-09-25: Confidence clamp (METRALE_MTP_DRAFT_CONF, 0 = off when
        // unset): when the drafter's chain confidence is below tau, discard
        // the drafts so the next step decodes serially. The drafter rows this
        // propose wrote are trimmed with `after_verify(0)`, as a full
        // rejection would trim them.
        let tau = self.levers.draft_conf_tau;
        if tau > 0.0
            && !drafts.is_empty()
            && let Some(conf) = proposer.last_confidence()
            && conf < tau
        {
            tracing::debug!(
                "MTP draft skipped: chain confidence {conf:.3} < tau {tau:.3}                  (pos {position}, {} drafts trimmed)",
                drafts.len(),
            );
            proposer.after_verify(0, prop_state.as_mut(), stream)?;
            return Ok(Vec::new());
        }
        Ok(drafts)
    }

    /// 2026-09-25: DFlash prefill capture: copy `proc_count` BF16 rows of
    /// `hidden_states()` (the layer just run) into the sequence's DFlash
    /// context accumulator. The prefill layer loops call it after each layer.
    ///
    /// Row `t` lands at element `((chunk_start + t - start) * n_capture + slot_idx) * h`
    /// of the accumulator, one `copy_d2d_async` per row, for the positions of the window
    /// `[start, prompt_len)` (`PrefillCtxWindow`, 2026-10-10: the last `max_ctx_len` prompt
    /// positions; the rest are dropped). For a prompt within `max_ctx_len`, `start` is 0 and
    /// the copies are the ones the first-W capture made.
    ///
    /// No-op when there are no DFlash capture layers, `layer_idx` is not one
    /// of them, the sequence has no `DflashProposerState`, or this is a comm
    /// rank other than 0 (2026-10-09: unless `dflash_capture_on_every_rank`).
    pub(super) fn try_dflash_prefill_capture_layer(
        &self,
        seq: &mut crate::traits::SequenceState,
        layer_idx: usize,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        let slot_idx = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        // 2026-10-10: Hybrid serve under SP prefill: every rank, before the rank-0 check, has the
        // layer swap the tap rows this capture keeps (`hybrid_spec.rs`).
        self.hybrid_sp_tap_gather(seq, layer_idx, chunk_start, proc_count, stream)?;
        if let Some(ref c) = self.comm
            && c.rank() != 0
            && !self.dflash_capture_on_every_rank()
        {
            return Ok(());
        }
        let (prompt_len, run_start) = (seq.prompt_len, seq.prefill_run_start);
        // 2026-10-10: `dflash_ctx_state`: the hybrid serve's DFlash state, else `proposer_state`.
        let dstate = match self.dflash_ctx_state(seq) {
            Some(ps) => match ps
                .as_any_mut()
                .downcast_mut::<metrale_model_arch::dflash_head::DflashProposerState>()
            {
                Some(s) => s,
                None => return Ok(()),
            },
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();
        let acc_base = dstate.ctx_hidden_acc;
        let win = metrale_model_arch::dflash_head::PrefillCtxWindow::new(
            prompt_len,
            chunk_start + proc_count,
            dstate.max_ctx_len,
            run_start,
        );
        let src_base = self.buffers.hidden_states();
        let (t0, rows) = win.chunk_rows(chunk_start, proc_count);
        if rows == 0 {
            return Ok(());
        }
        // 2026-10-11: One pitched copy for the chunk's window rows (source rows contiguous,
        // accumulator rows `n_capture * h` apart) instead of one `copy_d2d_async` per row: up to
        // 2057 launches per tap layer were a flat TTFT cost on the hybrid serve (#82 row G,
        // 2K/8K +49 ms). Same bytes to the same places.
        let slot_row0 = chunk_start + t0 - win.start;
        self.gpu.copy_d2d_2d_async(
            src_base.offset(t0 * h * bf16),
            h * bf16,
            acc_base.offset(slot_row0 * n_capture * h * bf16 + slot_idx * h * bf16),
            n_capture * h * bf16,
            h * bf16,
            rows,
            stream,
        )?;
        Ok(())
    }

    /// 2026-09-25: After a prefill chunk, set the sequence's DFlash `ctx_len`
    /// to the rows captured so far (2026-10-10: of the window `PrefillCtxWindow`, the last
    /// `max_ctx_len` prompt positions) so the first
    /// propose sees every captured prompt position. No-op without capture
    /// layers or a `DflashProposerState`, and on comm ranks other than 0 (2026-10-09: unless
    /// `dflash_capture_on_every_rank`).
    pub(super) fn update_dflash_ctx_len_after_prefill(
        &self,
        seq: &mut crate::traits::SequenceState,
        chunk_start: usize,
        proc_count: usize,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
            && !self.dflash_capture_on_every_rank()
        {
            return Ok(());
        }
        let (prompt_len, run_start) = (seq.prompt_len, seq.prefill_run_start);
        if let Some(ps) = self.dflash_ctx_state(seq)
            && let Some(dstate) =
                ps.as_any_mut()
                    .downcast_mut::<metrale_model_arch::dflash_head::DflashProposerState>()
        {
            let chunk_end = chunk_start + proc_count;
            let win = metrale_model_arch::dflash_head::PrefillCtxWindow::new(
                prompt_len,
                chunk_end,
                dstate.max_ctx_len,
                run_start,
            );
            dstate.ctx_len = win.len_after(chunk_end);
            // 2026-09-25: Prefill slot i holds prompt position i, so each
            // slot's fixed RoPE position is its index. `ctx_positions` is
            // rebuilt to the same length as `ctx_len` on every chunk.
            // 2026-10-10: Slot i holds absolute position `start + i` (`start` is 0 for a prompt
            // within the window, as before).
            dstate.ctx_positions = win.positions_after(chunk_end);
            if win.start > 0 {
                // 2026-10-10: No drafter K/V of an earlier window survives a tail window.
                dstate.ctx_committed = 0;
            }
            // 2026-10-09: Two-rank drafter: rank 1 captured the same prompt rows.
            // 2026-10-10: Same window on both ranks (same prompt length and `max_ctx_len`).
            dstate.tp_synced_pos = dstate.ctx_positions.last().copied();
        }
        Ok(())
    }

    /// 2026-09-25: DFlash hidden capture for one layer: copy row `token_idx`
    /// of `hidden_states()` into this layer's slot of `dflash_hidden_save`.
    /// Decode layer loops pass 0; the K-row verify loops pass `k - 1`.
    ///
    /// No-op when `dflash_hidden_save` is `None`, `layer_idx` is not a
    /// capture layer, or this is a comm rank other than 0.
    pub(super) fn try_dflash_capture(
        &self,
        layer_idx: usize,
        token_idx: usize,
        stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let src = self.buffers.hidden_states().offset(token_idx * h * bf16);
        let dst_slot = dst.offset(slot * h * bf16);
        self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
        Ok(())
    }

    /// 2026-10-10: The DFlash capture of a single-sequence K-row verify (K2/K3/K4): row `k - 1`
    /// as before, or, on a hybrid serve, rows `0..k`, so the scheduler commits every accepted
    /// row to the drafter context (`commit_ctx`) whichever proposer drafted. Fixed per process,
    /// so a captured verify graph never bakes the other choice.
    pub(super) fn try_dflash_capture_verify(
        &self,
        layer_idx: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        if self.hybrid_on() {
            self.try_dflash_capture_all(layer_idx, k, stream)
        } else {
            self.try_dflash_capture(layer_idx, k - 1, stream)
        }
    }

    /// 2026-09-25: `Model::save_dflash_hidden_for_propose`: run
    /// `try_dflash_capture` on row `token_idx` for every DFlash capture layer.
    pub(super) fn save_dflash_hidden_dispatch(&self, token_idx: usize, stream: u64) -> Result<()> {
        for &layer_idx in &self.dflash_capture_layers {
            self.try_dflash_capture(layer_idx, token_idx, stream)?;
        }
        Ok(())
    }

    /// 2026-09-25: Copy rows `0..k` of `hidden_states()` into the row-major
    /// `dflash_hidden_save` (row stride `n_capture * hidden_size` BF16), at
    /// this layer's slot in each row. [`Self::try_dflash_capture_all_at`]
    /// with both row bases 0.
    pub(super) fn try_dflash_capture_all(
        &self,
        layer_idx: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        self.try_dflash_capture_all_at(layer_idx, 0, k, 0, stream)
    }

    /// 2026-09-25: [`Self::try_dflash_capture_all`] with explicit source and
    /// destination row bases.
    ///
    /// A batched K=γ verify packs `n` sequences seq-major into
    /// `hidden_states`, so sequence `i`'s rows start at `src_row0 = off[i]`
    /// and its capture band at `dst_row0 = i * dflash_kgamma`, the row the
    /// scheduler passes to `commit_ctx` as `scratch_row`. Writes stop at
    /// `dflash_hidden_save_rows`. No-op when `dflash_hidden_save` is `None`,
    /// `layer_idx` is not a capture layer, or this is a comm rank other than 0.
    pub(super) fn try_dflash_capture_all_at(
        &self,
        layer_idx: usize,
        src_row0: usize,
        k: usize,
        dst_row0: usize,
        stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let ctx_slot_bytes = self.dflash_capture_layers.len() * h * bf16;
        let kmax = self.dflash_hidden_save_rows;
        debug_assert!(
            dst_row0 + k <= kmax,
            "try_dflash_capture_all: rows {dst_row0}..{} exceed capacity {kmax}",
            dst_row0 + k
        );
        let k_capped = k.min(kmax.saturating_sub(dst_row0));
        for t in 0..k_capped {
            let src = self
                .buffers
                .hidden_states()
                .offset((src_row0 + t) * h * bf16);
            let dst_slot = dst.offset((dst_row0 + t) * ctx_slot_bytes + slot * h * bf16);
            self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
        }
        Ok(())
    }
}
