// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Mid-chunk SSM tail capture: inside the prefill pass, save each SSM
//! layer's state at the tail boundary `tb` (`ssm_tail_boundary`) and, when the pass
//! covers it, at `tb - block_size`, by splitting the per-token recurrence and conv1d
//! kernels there. No extra forward pass is run.
//!
//! The capture at `tb - block_size` is registered as the tail's sibling: the next
//! turn's block-floored match can land one block below `tb`, where the tail itself
//! is too deep to restore.
//!
//! Flow:
//!   1. [`TransformerModel::prepare_midchunk_capture`], before
//!      `prefill_b_forward_layers`, decides whether this pass spans `tb`, reserves
//!      the snapshot slots and precomputes the per-SSM-layer destination pointers.
//!   2. `prefill_b_forward_layers` passes the plan as
//!      `ForwardContext::midchunk_capture`; each SSM layer splits its kernels at
//!      `cap_local` (and `cap_local_early`) and copies the state into the reserved
//!      slots.
//!   3. [`TransformerModel::finalize_midchunk_capture`], after
//!      `prefill_b_forward_layers`, registers the slots in the snapshot index.
//!
//! The switch is on by default; `--no-ssm-tail-midchunk` or
//! `METRALE_SSM_TAIL_MIDCHUNK=0` turns it off (`ssm_tail_midchunk_enabled`). With
//! the switch on, only `metrale_scale` builds (`METRALE_TARGET_HW=strix*`) plan a
//! capture: on every other build `prepare_midchunk_capture` returns `None`.
//!
//! Owner: model-engine prefill (SSM prefix cache).
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// 2026-09-25: Per-pass plan for a mid-chunk tail capture. Owns the per-SSM-layer
/// destination pointers that `ForwardContext::midchunk_capture` borrows during
/// `forward_layers`.
pub(in crate::model) struct MidCapturePlan {
    /// 2026-09-25: Split point in pass-local token coordinates (`tb - proc_start`).
    pub cap_local: usize,
    /// 2026-09-25: Reserved tail snapshot slot, also the snapshot id it is registered under.
    pub snap_slot: usize,
    /// 2026-09-25: Token boundary of the tail snapshot (`ssm_tail_boundary`).
    pub tb: usize,
    /// 2026-09-25: Per-SSM-layer h_state destination in `snap_slot`.
    pub h_dsts: Vec<DevicePtr>,
    /// 2026-09-25: Per-SSM-layer conv_state destination in `snap_slot`.
    pub conv_dsts: Vec<DevicePtr>,
    /// 2026-09-25: Bytes per layer of h_state.
    pub h_bytes: usize,
    /// 2026-09-25: Bytes per layer of conv_state.
    pub conv_bytes: usize,
    /// 2026-09-25: KV block size, the grid of `ssm_tail_boundary`.
    pub bs: usize,
    /// 2026-09-25: Split point for the capture at `tb - bs` (`cap_local - bs`). `Some` only
    /// when that point lies inside the pass and a second slot was reserved.
    pub cap_local_early: Option<usize>,
    /// 2026-09-25: Slot of the `tb - bs` snapshot, either reserved for an in-pass split or
    /// saved before the pass (see `prepare_midchunk_capture`).
    pub snap_slot_early: Option<usize>,
    /// 2026-09-25: Token boundary of the earlier snapshot (`tb - bs`).
    pub tb_early: Option<usize>,
    /// 2026-09-25: Per-SSM-layer h_state destination for the in-pass `tb - bs` capture.
    pub h_dsts_early: Vec<DevicePtr>,
    /// 2026-09-25: Per-SSM-layer conv_state destination for the in-pass `tb - bs` capture.
    pub conv_dsts_early: Vec<DevicePtr>,
    /// 2026-10-03: True for an in-pass tail-split capture (`inpass_capture.rs`,
    /// `METRALE_GLM_SSM_INPASS_CAPTURE=1`): `tb` is then `prefill_plan::tail_split_point`, the
    /// slot is registered as an intermediate checkpoint (`finalize_inpass_capture`), and the
    /// layers find their ordinal by `live_h`. False for the tail mid-chunk plan above.
    pub inpass: bool,
    /// 2026-10-03: Per SSM ordinal, this sequence's live h_state pool address; empty unless
    /// `inpass`.
    pub live_h: Vec<DevicePtr>,
    /// 2026-10-03: SSM layers that issued their in-pass capture copies during the pass
    /// (`MidchunkCapture::captured`).
    pub captured: std::sync::atomic::AtomicUsize,
}

impl TransformerModel {
    /// 2026-09-25: Reserve a tail snapshot slot, reclaiming one from the prefix cache when
    /// the pool is full. `None` when the pool is full and the reclaim fails.
    pub(super) fn reserve_snapshot_slot(
        &self,
        session_hash: u64,
        kv_cache: &mut PagedKvCache,
    ) -> Option<usize> {
        match self.ssm_snapshots.reserve_tail_slot(session_hash) {
            Some(s) => Some(s),
            None => {
                if self.ssm_snapshots.reclaim_from_cache(
                    self.prefix_cache.as_ref(),
                    kv_cache,
                    self.ssm_tier_store.as_deref(),
                    self.gpu.as_ref(),
                ) {
                    self.ssm_snapshots.reserve_tail_slot(session_hash)
                } else {
                    None
                }
            }
        }
    }

    /// 2026-09-25: Plan an in-pass tail capture for the prefill pass over tokens
    /// `[proc_start, proc_start + proc_count)`.
    ///
    /// Returns `None` (no capture) when the switch is off, the snapshot pool is
    /// disabled, the build is not `metrale_scale`, the reuse gate below says no, the
    /// prompt has no `tb`, the pass does not strictly span `tb`, or no slot can be
    /// reserved. The `tb - bs` snapshot is best effort: without a second slot only
    /// the tail is captured.
    pub(in crate::model) fn prepare_midchunk_capture(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        proc_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Option<MidCapturePlan> {
        if !metrale_gpu_runtime::ssm_tail_midchunk_enabled() || !self.ssm_snapshots.is_enabled() {
            return None;
        }
        // 2026-09-25: Only `metrale_scale` builds write `h_dsts` (the split4 arm of
        // `qwen3_ssm/trait_prefill_recur.rs`). On other targets the snapshot would be
        // registered with an h_state that was never written, so no plan is made.
        if !cfg!(metrale_scale) {
            return None;
        }
        // 2026-09-25: Reuse gate. A tail snapshot is restored only when `session_matches`
        // passes (`snap_agree::local_proposal`: `!is_tail || session_ok`), so capture
        // only when a later request can use it: this request hit the prefix cache, or a
        // snapshot is already tagged with its session (`session_has_history`).
        if seq.cached_prefix_tokens == 0
            && !self.ssm_snapshots.session_has_history(seq.session_hash)
        {
            return None;
        }
        let bs = kv_cache.block_size();
        let tb = metrale_gpu_runtime::ssm_tail_boundary(tokens.len(), bs)?;
        // 2026-09-25: Only a pass that strictly crosses `tb` can split there.
        if !(proc_start < tb && tb < proc_start + proc_count) {
            return None;
        }
        let cap_local = tb - proc_start;
        let n = self.ssm_snapshots.num_ssm_layers();

        // 2026-09-25: The tail slot at `tb` is required.
        let snap_slot = self.reserve_snapshot_slot(seq.session_hash, kv_cache)?;
        let mut h_dsts = Vec::with_capacity(n);
        let mut conv_dsts = Vec::with_capacity(n);
        for l in 0..n {
            h_dsts.push(self.ssm_snapshots.tail_h_dst(l, snap_slot));
            conv_dsts.push(self.ssm_snapshots.tail_conv_dst(l, snap_slot));
        }

        // 2026-09-25: The slot at `tb - bs` is optional. When the pass covers that point
        // (`cap_local > bs`, i.e. `proc_start < tb - bs`) it is split in-pass.
        let mut cap_local_early = None;
        let mut snap_slot_early = None;
        let mut tb_early = None;
        let mut h_dsts_early = Vec::new();
        let mut conv_dsts_early = Vec::new();
        if bs > 0 && cap_local > bs && tb > bs {
            match self.reserve_snapshot_slot(seq.session_hash, kv_cache) {
                Some(slot2) => {
                    cap_local_early = Some(cap_local - bs);
                    snap_slot_early = Some(slot2);
                    tb_early = Some(tb - bs);
                    h_dsts_early = Vec::with_capacity(n);
                    conv_dsts_early = Vec::with_capacity(n);
                    for l in 0..n {
                        h_dsts_early.push(self.ssm_snapshots.tail_h_dst(l, slot2));
                        conv_dsts_early.push(self.ssm_snapshots.tail_conv_dst(l, slot2));
                    }
                }
                None => tracing::info!(
                    "midchunk EARLY skipped: no snapshot slot (pool exhausted, \
                     reclaim failed) tb_early={}",
                    tb - bs,
                ),
            }
        } else if bs > 0 && cap_local == bs && tb > bs {
            // 2026-09-25: The pass starts exactly at `tb - bs`, so the live pool state
            // already is the state after `tb - bs` tokens and no split is needed: save
            // it now. The save is issued on `stream` before the forward pass, so it is
            // ordered before the pass advances the state.
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
                        None
                    }
                }
                Err(e) => {
                    tracing::warn!("midchunk EARLY pre-pass save failed: {e:#}");
                    None
                }
            };
            match saved {
                Some(slot2) => {
                    // 2026-09-25: Registration only: `cap_local_early` stays `None`, so the
                    // forward pass does not split at this point.
                    snap_slot_early = Some(slot2);
                    tb_early = Some(tb - bs);
                    tracing::info!(
                        "midchunk EARLY pre-pass copy at token {} (snap {slot2})",
                        tb - bs,
                    );
                }
                None => tracing::info!(
                    "midchunk EARLY skipped: no snapshot slot for pre-pass copy \
                     tb_early={}",
                    tb - bs,
                ),
            }
        } else if bs > 0 && tb > bs {
            tracing::info!(
                "midchunk EARLY skipped: pass starts after tb-bs \
                 (proc_start={proc_start} tb={tb} cap_local={cap_local} bs={bs})",
            );
        }

        Some(MidCapturePlan {
            cap_local,
            snap_slot,
            tb,
            h_dsts,
            conv_dsts,
            h_bytes: self.ssm_snapshots.h_bytes(),
            conv_bytes: self.ssm_snapshots.conv_bytes(),
            bs,
            cap_local_early,
            snap_slot_early,
            tb_early,
            h_dsts_early,
            conv_dsts_early,
            inpass: false,
            live_h: Vec::new(),
            captured: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// 2026-09-25: Register the captured slots after the forward pass:
    ///
    /// * the tail slot through `insert_tail_snapshot`, which supersedes the session's
    ///   previous tail and sibling;
    /// * the `tb - bs` slot, when present, through `insert_tail_sibling_snapshot`.
    ///
    /// Frees every snapshot id either insert displaces.
    pub(in crate::model) fn finalize_midchunk_capture(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        plan: &MidCapturePlan,
    ) {
        for old in self.prefix_cache.insert_tail_snapshot(
            &tokens[..plan.tb],
            plan.snap_slot,
            seq.session_hash,
            seq.adapter_id,
        ) {
            self.ssm_snapshots.free(old);
        }
        tracing::info!(
            "midchunk tail SSM capture at token {} (snap {})",
            plan.tb,
            plan.snap_slot
        );

        if let (Some(tb_early), Some(slot2)) = (plan.tb_early, plan.snap_slot_early) {
            // 2026-09-25: Must follow `insert_tail_snapshot`, whose sweep clears the
            // session's previous tail and sibling (cache radix_tree/snapshot_insert.rs).
            if let Some(old) = self.prefix_cache.insert_tail_sibling_snapshot(
                &tokens[..tb_early],
                slot2,
                seq.session_hash,
                seq.adapter_id,
            ) {
                self.ssm_snapshots.free(old);
            }
            tracing::info!(
                "midchunk EARLY tail SSM capture at token {} (snap {})",
                tb_early,
                slot2
            );
        }
    }
}
