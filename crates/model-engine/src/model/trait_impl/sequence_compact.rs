// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: SSM pool slot migration: move a sequence's SSM state to another slot
//! (`compact_sequence_dispatch`), and disown a slot another sequence took over
//! (`detach_slot_for_reuse_dispatch`).
//! 2026-10-10: `METRALE_MTP_POOL_CAP` slot migration under EP v2 (`migrate_to_covered_slot_dispatch`,
//! the worker's `ep_worker_migrate_slot`), and the slot-keyed graph drops it shares with
//! `free_sequence` (`drop_slot_graphs`, `drop_slot_kgamma_graphs`).
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
use crate::traits::{ChunkedPrefillPageMetadata, Model, ModelLifecycle, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    /// 2026-09-25: Disown a retired sequence's SSM slot after `compact_sequence` moved
    /// another sequence into it: take the index out of this sequence's guard without
    /// releasing it, and set the `slot_idx = usize::MAX` sentinel that
    /// `free_sequence` checks. Call it right after that `compact_sequence` and before
    /// any fallible step that could drop this sequence, or the guard's `Drop` releases
    /// a slot the other sequence owns.
    pub(super) fn detach_slot_for_reuse_dispatch(&self, seq: &mut SequenceState) {
        if let Some(g) = seq.ssm_slot.as_mut() {
            let _ = g.take();
        }
        seq.slot_idx = usize::MAX;
    }

    pub(super) fn compact_sequence_dispatch(
        &self,
        seq: &mut SequenceState,
        new_slot: usize,
    ) -> Result<()> {
        let old_slot = seq.slot_idx;
        if old_slot == new_slot {
            return Ok(());
        }

        let stream = self.gpu.default_stream();
        self.ssm_pool
            .copy_slot(old_slot, new_slot, self.gpu.as_ref(), stream)?;

        // 2026-09-25: Repoint every slot-addressed pointer in each `SsmLayerState` (state,
        // prefill staging, MTP checkpoints and intermediates): the old slot is released
        // below and can be handed to another sequence.
        let has_mtp = self.ssm_pool.has_mtp;
        // 2026-09-25: The H-intermediate count is per slot (`h_inter_count`); the conv
        // count is the same for every slot.
        let num_intermediates = self.ssm_pool.num_intermediates;
        let h_intermediates = self.ssm_pool.h_inter_count(new_slot);
        let mut ssm_layer_idx = 0usize;
        for (i, state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                    ssm.h_state = self.ssm_pool.h_state(ssm_layer_idx, new_slot);
                    ssm.conv_state = self.ssm_pool.conv_state(ssm_layer_idx, new_slot);
                    // 2026-09-25: `None` under an FP32-sized pool, which has no staging.
                    ssm.h_prefill_stage = self.ssm_pool.h_prefill_stage(new_slot);
                    if has_mtp {
                        if ssm.h_state_checkpoint.is_some() {
                            ssm.h_state_checkpoint =
                                self.ssm_pool.h_checkpoint(ssm_layer_idx, new_slot);
                        }
                        if ssm.conv_state_checkpoint.is_some() {
                            ssm.conv_state_checkpoint =
                                self.ssm_pool.conv_checkpoint(ssm_layer_idx, new_slot);
                        }
                        if !ssm.h_state_intermediates.is_empty() {
                            ssm.h_state_intermediates.clear();
                            for t in 0..h_intermediates {
                                ssm.h_state_intermediates.push(self.ssm_pool.h_intermediate(
                                    ssm_layer_idx,
                                    new_slot,
                                    t,
                                ));
                            }
                        }
                        if !ssm.conv_state_intermediates.is_empty() {
                            ssm.conv_state_intermediates.clear();
                            for t in 0..num_intermediates {
                                ssm.conv_state_intermediates
                                    .push(self.ssm_pool.conv_intermediate(
                                        ssm_layer_idx,
                                        new_slot,
                                        t,
                                    ));
                            }
                        }
                    }
                }
                ssm_layer_idx += 1;
            }
        }

        seq.slot_idx = new_slot;
        // 2026-10-14: The hybrid MTP catch-up ring keeps neither slot's rows across a move.
        self.hybrid_mtp_ring_reset_slot(old_slot);
        self.hybrid_mtp_ring_reset_slot(new_slot);
        // 2026-09-25: `copy_slot` is enqueued, so wait for it before the old slot is
        // released and can be claimed while the copy still reads it.
        self.gpu.synchronize(stream)?;
        // 2026-09-25: Take the new slot off the free list if it is there, so it is never
        // both owned by this guard and free. Then release the old slot once and point the
        // guard at the new one; the guard's later release frees the new slot once.
        self.ssm_pool.claim_specific(new_slot);
        if let Some(g) = seq.ssm_slot.as_mut() {
            let owned = g.take();
            debug_assert_eq!(
                owned,
                Some(old_slot),
                "compact_sequence: guard owned {owned:?}, expected old_slot {old_slot}"
            );
            self.ssm_pool.release_slot(old_slot);
            g.migrate(new_slot);
        } else {
            self.ssm_pool.release_slot(old_slot);
        }
        Ok(())
    }

    /// 2026-10-10: Drop every graph keyed by `slot_idx` that bakes per-sequence state: the
    /// n=1 decode graph, the K-row verify graphs, each batched decode key containing the slot
    /// and each batched-verify key with a `(slot, k)` pair on `ssm_slot` (or on `slot_idx`).
    /// `free_sequence` calls it when a layer owns per-sequence device state
    /// (`graph_stale_on_new_sequence`); slot migration calls it for both slots.
    pub(super) fn drop_slot_graphs(&self, slot_idx: usize, ssm_slot: Option<u32>) {
        let slot = slot_idx as u32;
        let mut stale: Vec<metrale_gpu_runtime::gpu::GraphHandle> = self
            .decode_graph
            .lock()
            .remove(&slot_idx)
            .into_iter()
            .collect();
        // 2026-09-25: The K-row verify graphs are slot-keyed too and bake the same
        // per-sequence state (`Glm5NextDsaState::{k_normed, gate, valid}`).
        for m in [
            &self.verify2_graph,
            &self.verify3_graph,
            &self.verify4_graph,
        ] {
            stale.extend(m.lock().remove(&slot_idx));
        }
        {
            // 2026-09-25: A batched graph bakes every row's state pointers, so every key
            // that contains this slot is dropped.
            let mut batch = self.batch_decode_graphs.lock();
            let keys: Vec<Vec<u32>> = batch
                .0
                .keys()
                .filter(|k| k.contains(&slot))
                .cloned()
                .collect();
            for k in keys {
                if let Some((g, _)) = batch.0.remove(&k) {
                    stale.push(g);
                }
            }
        }
        {
            // 2026-10-07: Batched-verify keys are `(slot, k)` pairs then a sentinel
            // (`verify_graph_key`); a graphed own-state verify bakes every pair's state, so
            // every key with a pair on this sequence's SSM slot (or slot index) is dropped.
            let ssm = ssm_slot.unwrap_or(slot);
            let mut vb = self.verify_batched_graphs.lock();
            let keys: Vec<Vec<u32>> =
                vb.0.keys()
                    .filter(|k| {
                        k[..k.len().saturating_sub(1)]
                            .chunks(2)
                            .any(|p| p[0] == ssm || p[0] == slot)
                    })
                    .cloned()
                    .collect();
            for k in keys {
                if let Some((g, _)) = vb.0.remove(&k) {
                    stale.push(g);
                }
            }
        }
        for g in stale {
            if g.0 != 0
                && let Err(e) = self.gpu.destroy_graph(g)
            {
                tracing::warn!("free_sequence: destroy graph for slot {slot}: {e:#}");
            }
        }
    }

    /// 2026-10-10: Drop every K of `slot_idx` from `verify_kgamma_graph` and `fused_graph`,
    /// which are keyed by (slot, K) and bake the request's LoRA routing.
    pub(super) fn drop_slot_kgamma_graphs(&self, slot_idx: usize) {
        for graph_map in [&self.verify_kgamma_graph, &self.fused_graph] {
            let mut cache = graph_map.lock();
            let keys: Vec<(usize, usize)> =
                cache.keys().filter(|k| k.0 == slot_idx).copied().collect();
            for k in keys {
                if let Some(graph) = cache.remove(&k)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "free_sequence: destroy_graph(kgamma/fused[{},{}]): {e:#}",
                        k.0,
                        k.1
                    );
                }
            }
        }
    }

    /// 2026-10-10: `METRALE_MTP_POOL_CAP` under EP v2 (rank 0, between steps, no step in
    /// flight): move a sequence on a slot the MTP verify pools do not cover onto the lowest
    /// free covered slot, on both ranks. Returns the new slot, or `None` when the cap is off,
    /// the slot is covered, or no covered slot is free. Without it a survivor on an uncovered
    /// slot keeps the whole batch on plain decode until it finishes (mtpcapval E: a lone
    /// sequence on slot 8 lost 512+ MTP steps).
    ///
    /// Order on both ranks: fold the pending GDN carry rows, make the default stream wait
    /// for the secondary stream's commit copies, `compact_sequence` (claim the target, release
    /// the source), then drop both slots' graphs: they bake per-sequence DSA pointers, and a
    /// later sequence on the source slot must not replay the migrated sequence's graph. The
    /// decode-rollback ring needs nothing: it is skipped while speculative decode is active,
    /// and the cap requires MTP.
    pub(super) fn migrate_to_covered_slot_dispatch(
        &self,
        seq: &mut SequenceState,
    ) -> Result<Option<usize>> {
        let (cover, max) = (self.ssm_pool.mtp_slots, self.ssm_pool.max_slots);
        let from = seq.slot_idx;
        if !metrale_model_layers::ssm_reserve::mtp_pool_cap()
            || !self.ssm_pool.has_mtp
            || cover >= max
            || from < cover
            || from >= max
            || (self.multi_rank_protocol_active() && !self.ep_protocol_v2)
        {
            // 2026-10-10: Without v2 the worker has no slot-addressed commands to mirror a
            // migration with (and the scheduler compacts instead).
            return Ok(None);
        }
        let Some(to) = self.ssm_pool.lowest_free_below(cover) else {
            return Ok(None);
        };
        if self.ep_protocol_v2 {
            self.ep_broadcast_seq_and_cmd(from as u32, EP_CMD_MIGRATE_SLOT, true)?;
            if self.multi_rank_protocol_active() {
                self.ep_broadcast_u32(to as u32)?;
            }
        }
        // 2026-10-10: Past the broadcast the worker migrates too; a failure below is a GPU
        // fault (flush, copy or synchronize), after which the ranks may disagree on this
        // sequence's slot. It is returned and logged by the scheduler like any device error.
        self.migrate_slot_local(seq, to)?;
        Ok(Some(to))
    }

    /// 2026-10-10: The worker side of [`Self::migrate_to_covered_slot_dispatch`] for
    /// `EP_CMD_MIGRATE_SLOT` on slot `from`: receive the target, free the phantom the worker
    /// holds there (`EP_FREE_REALLOC` re-allocates one on every head release), migrate, then
    /// re-allocate a phantom on `from`. That keeps the v2 mirror invariant the alloc check in
    /// `ep_worker_execute` relies on: every slot free on the head below the claim high-water
    /// mark is phantom-held here, so `from` is the lowest slot free on this rank and the
    /// phantom's claim returns it (`ep_v2_mirrored_claims_agree_with_migration`).
    pub(crate) fn ep_worker_migrate_slot(
        &self,
        from: usize,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        let to = self.ep_broadcast_u32(0)? as usize;
        if to >= slots.len() || from >= slots.len() {
            bail!(
                "ep_worker_migrate_slot: slot {from} -> {to} out of range ({})",
                slots.len()
            );
        }
        let mut seq = slots[from]
            .take()
            .ok_or_else(|| anyhow::anyhow!("ep_worker_migrate_slot: no sequence on slot {from}"))?;
        if let Some(mut phantom) = slots[to].take() {
            self.free_sequence(&mut phantom)?;
        }
        let moved = self.migrate_slot_local(&mut seq, to);
        slots[to] = Some(seq);
        moved?;
        let phantom = self.alloc_sequence()?;
        if phantom.slot_idx != from {
            bail!(
                "ep_worker_migrate_slot: phantom took SSM-pool slot {} instead of {from} \
                 — claim_slot ordering invariant violated",
                phantom.slot_idx
            );
        }
        slots[from] = Some(phantom);
        Ok(true)
    }

    /// 2026-10-10: The per-rank half of a cap migration (see
    /// [`Self::migrate_to_covered_slot_dispatch`]).
    fn migrate_slot_local(&self, seq: &mut SequenceState, to: usize) -> Result<()> {
        let from = seq.slot_idx;
        self.gdn_carry_flush_pending()?;
        self.sync_secondary_dispatch()?;
        self.compact_sequence_dispatch(seq, to)?;
        for slot in [from, to] {
            self.drop_slot_graphs(slot, Some(slot as u32));
            self.drop_slot_kgamma_graphs(slot);
        }
        tracing::info!(
            "METRALE_MTP_POOL_CAP: migrated sequence from SSM slot {from} to {to} (cover {})",
            self.ssm_pool.mtp_slots
        );
        Ok(())
    }
}

/// 2026-10-10: EP command for a `METRALE_MTP_POOL_CAP` slot migration: preamble `seq_id` =
/// the source slot, then the target slot (`ep_worker_migrate_slot`).
pub(in crate::model) const EP_CMD_MIGRATE_SLOT: u32 = 0xFFFF_FFFB;
const _: () = assert!(
    EP_CMD_MIGRATE_SLOT != crate::model::trait_impl::verify_ep::EP_CMD_VERIFY_BATCH
        && EP_CMD_MIGRATE_SLOT
            != crate::model::trait_impl::propose_batch_ep::EP_CMD_MTP_PROPOSE_BATCH
        && EP_CMD_MIGRATE_SLOT != crate::model::trait_impl::decode_checkpoint::EP_CMD_DECODE_CKPT
        && EP_CMD_MIGRATE_SLOT
            != metrale_model_layers::speculative::glm_dflash::EP_CMD_VERIFY_KGAMMA
        && EP_CMD_MIGRATE_SLOT != metrale_model_layers::speculative::EP_CMD_MTP_PROPOSE
        && EP_CMD_MIGRATE_SLOT != 0xFFFF_FFF7
        && !matches!(
            EP_CMD_MIGRATE_SLOT,
            0xFFFF_FFE0 | 0xFFFF_FFF0..=0xFFFF_FFF4 | 0xFFFF_FFFF
        ),
    "EP_CMD_MIGRATE_SLOT collides with another EP command (0xFFFF_FFF7 is reserved: DFlash ctx-commit)"
);
