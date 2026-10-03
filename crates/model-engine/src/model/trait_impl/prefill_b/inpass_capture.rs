// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: In-pass tail-split snapshot (`METRALE_GLM_SSM_INPASS_CAPTURE=1`, race #69):
//! take the prefix-cache SSM snapshot at `prefill_plan::tail_split_point` inside the prefill
//! pass that crosses it, instead of splitting the prompt there.
//!
//! The split (`prefill_chunk_dispatch`, `prefill_plan::plan_chunk_len`) changed the pass
//! sequence of every prompt, cache miss included, and with it the output (TEB 146 vs 156 on
//! GLM-5.3, measured 2026-10-02/03; the same image with the cache off matched the control).
//! With this lever on, `prefill_tail_split_dispatch` returns `None`, so the scheduler and the
//! dispatcher run the cache-off pass sequence, and the snapshot comes from the pass itself:
//!
//!   1. [`TransformerModel::prepare_inpass_capture`], before the forward pass: when the pass
//!      strictly crosses the cut, reserve one snapshot slot and record, per SSM ordinal, its
//!      h/conv destinations and the sequence's live h_state address.
//!   2. The forward pass carries the plan as `ForwardContext::midchunk_capture`. Each SSM layer
//!      whose call covers the cut copies its recurrent and conv state as of the cut into the
//!      slot (GLM-5.3 KDA: `Glm5NextKdaLayer::decode_k_capture`, which splits the token loop's
//!      two launches at that row; bit-identical to the uncaptured pass) and counts itself.
//!   3. [`TransformerModel::finalize_inpass_capture`], after the pass: when every SSM layer
//!      captured, attach the aux blobs as of the cut (`LayerAuxState::snapshot_aux_prefix`; the
//!      DSA indexer rows `[0, cut)`) and register the slot exactly as `save_checkpoint` registers
//!      the split's snapshot (`prefill_b_register_checkpoint`). Otherwise free it.
//!
//! What the snapshot holds, per part:
//! - SSM h_state and conv_state of every SSM layer: copied in the pass, at the cut.
//! - Session tag: set when the slot is reserved (`reserve_tail_slot`), as `save` sets it.
//! - Aux blobs: per DSA layer the indexer rows `[0, cut)`, read after the pass; row `p` is
//!   written once, by the pass over position `p`.
//! - No last-token hidden: an intermediate checkpoint never carries one (`save_hidden` is only
//!   for the exact leaf).
//! - The K/V of `tokens[..cut]` is in the paged blocks the pass wrote; the registration inserts
//!   them, under the same `kv_valid_tokens` cap as `save_checkpoint`.
//!
//! EP: both ranks see the same prompt, block size and pass ranges, so they plan the same cut in
//! the same pass. Slot reservation and the registration checks are the split path's, with its
//! per-rank outcomes; the restore-side agreement (`snap_agree`) is unchanged.
//!
//! Owner: model-engine prefill (SSM prefix cache).
//! Invariants:
//! - With the lever off, `prepare_inpass_capture` returns `None` before reading anything, and
//!   `prefill_tail_split_dispatch` returns what it returned before the lever existed.
//! - A slot is registered only when `MidCapturePlan::captured` equals the SSM layer count;
//!   every other exit frees it.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::LayerType;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::midchunk_capture::MidCapturePlan;
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-10-03: Whether the in-pass capture replaces the tail split for this model: the
    /// lever is on, the model has SSM layers, and every one of them honours the capture
    /// (`LayerCapabilities::inpass_ssm_capture_supported`). Otherwise the split runs as before.
    pub(in crate::model) fn inpass_ssm_capture_active(&self) -> bool {
        if !crate::prefill_plan::inpass_capture_lever() {
            return false;
        }
        let mut ssm_layers = 0usize;
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                if !layer.inpass_ssm_capture_supported() {
                    Self::warn_inpass_unsupported(i);
                    return false;
                }
                ssm_layers += 1;
            }
        }
        ssm_layers > 0 && ssm_layers == self.ssm_snapshots.num_ssm_layers()
    }

    /// 2026-10-03: The one warning when the lever is set on a model (or lever combination)
    /// whose SSM layers cannot capture in-pass; the tail split stays on.
    fn warn_inpass_unsupported(layer: usize) {
        static W: std::sync::Once = std::sync::Once::new();
        W.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_SSM_INPASS_CAPTURE=1 ignored: SSM layer {layer} cannot capture its \
                 state inside a prefill pass (only the GLM-5.3 KDA decode_k prefill can); the \
                 prefix-cache tail split stays on"
            )
        });
    }

    /// 2026-10-03: Plan the in-pass snapshot for the pass over `[proc_start, proc_start +
    /// proc_count)`, or `None` (no capture) when the lever is off or inert, the prompt does not
    /// qualify for a tail snapshot (`tail_split_eligible`, `tail_split_point`), the pass does
    /// not strictly cross the cut, the h-state is not FP32, or no slot can be reserved.
    pub(in crate::model) fn prepare_inpass_capture(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        proc_start: usize,
        proc_count: usize,
    ) -> Option<MidCapturePlan> {
        if !self.inpass_ssm_capture_active() || !self.tail_split_eligible(tokens) {
            return None;
        }
        let bs = kv_cache.block_size();
        let cut = crate::prefill_plan::tail_split_point(tokens.len(), bs)?;
        if !crate::prefill_plan::inpass_capture_spans(cut, proc_start, proc_count) {
            return None;
        }
        // 2026-10-03: The layers copy `h_bytes` straight out of the pool slot, as `save` does
        // for an FP32 slot. An f16 h-state (or an f16-sized pool) gets no in-pass snapshot;
        // the GLM-5.3 KDA prefill refuses an f16 h-state anyway (`Glm5NextLayer::kda_state`).
        if self.seq_ssm_h_is_f16(seq)
            || self.ssm_pool.h_stored_bytes != self.ssm_snapshots.h_bytes()
        {
            tracing::warn!(
                "in-pass SSM capture at token {cut} skipped: f16 h-state or f16-sized pool"
            );
            return None;
        }
        let Some(snap_slot) = self.reserve_snapshot_slot(seq.session_hash, kv_cache) else {
            tracing::warn!(
                "SSM snapshot pool exhausted and no evictable cached entries — no in-pass tail \
                 snapshot at token {cut}. Consider raising --ssm-cache-slots."
            );
            return None;
        };
        let n = self.ssm_snapshots.num_ssm_layers();
        let mut h_dsts = Vec::with_capacity(n);
        let mut conv_dsts = Vec::with_capacity(n);
        let mut live_h = Vec::with_capacity(n);
        for l in 0..n {
            h_dsts.push(self.ssm_snapshots.tail_h_dst(l, snap_slot));
            conv_dsts.push(self.ssm_snapshots.tail_conv_dst(l, snap_slot));
            live_h.push(self.ssm_pool.h_state(l, seq.slot_idx));
        }
        Some(MidCapturePlan {
            cap_local: cut - proc_start,
            snap_slot,
            tb: cut,
            h_dsts,
            conv_dsts,
            h_bytes: self.ssm_snapshots.h_bytes(),
            conv_bytes: self.ssm_snapshots.conv_bytes(),
            bs,
            cap_local_early: None,
            snap_slot_early: None,
            tb_early: None,
            h_dsts_early: Vec::new(),
            conv_dsts_early: Vec::new(),
            inpass: true,
            live_h,
            captured: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// 2026-10-03: After the pass: register the in-pass snapshot as the intermediate checkpoint
    /// at `plan.tb`, or free its slot when some SSM layer did not capture, the K/V below the cut
    /// is not known to be written, or the aux blobs cannot be read. Never fails the prefill: the
    /// snapshot is a cache entry, and a missing one costs only a later recompute.
    pub(in crate::model) fn finalize_inpass_capture(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        plan: &MidCapturePlan,
        stream: u64,
    ) {
        let cut = plan.tb;
        let snap_id = plan.snap_slot;
        let n = self.ssm_snapshots.num_ssm_layers();
        let got = plan.captured.load(std::sync::atomic::Ordering::Relaxed);
        if got != n {
            tracing::warn!(
                "in-pass SSM capture at token {cut}: {got} of {n} SSM layers captured; \
                 snapshot {snap_id} dropped"
            );
            self.ssm_snapshots.free(snap_id);
            return;
        }
        let bs = kv_cache.block_size();
        let end_block = cut / bs;
        // 2026-10-03: The cap `save_checkpoint` applies before it saves.
        if seq.kv_valid_tokens / bs < end_block {
            tracing::debug!(
                "Skip in-pass checkpoint at block {end_block}: kv_valid_tokens={} only covers \
                 {} complete blocks",
                seq.kv_valid_tokens,
                seq.kv_valid_tokens / bs,
            );
            self.ssm_snapshots.free(snap_id);
            return;
        }
        let aux = match self.collect_aux_states_prefix(seq, cut, stream) {
            Ok(aux) => aux,
            Err(e) => {
                tracing::warn!(
                    "in-pass SSM capture at token {cut}: aux state as of the cut failed \
                     ({e:#}); snapshot {snap_id} dropped"
                );
                self.ssm_snapshots.free(snap_id);
                return;
            }
        };
        if !aux.is_empty() {
            self.ssm_snapshots.set_aux(snap_id, aux);
        }
        let is_prompt_tail = crate::prefill_plan::is_prompt_tail_end(cut, tokens.len(), bs);
        self.prefill_b_register_checkpoint(tokens, seq, kv_cache, cut, snap_id, is_prompt_tail);
    }

    /// 2026-10-03: `collect_aux_states` as of the first `rows` positions
    /// (`LayerAuxState::snapshot_aux_prefix`).
    fn collect_aux_states_prefix(
        &self,
        seq: &SequenceState,
        rows: usize,
        stream: u64,
    ) -> Result<Vec<(u32, Vec<u8>)>> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            if let Some(blob) = l.snapshot_aux_prefix(
                seq.layer_states[i].as_ref(),
                rows,
                self.gpu.as_ref(),
                stream,
            )? {
                out.push((i as u32, blob));
            }
        }
        Ok(out)
    }
}
