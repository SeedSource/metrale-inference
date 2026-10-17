// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: SSM state pool slot lifecycle (claim, release, [`SlotGuard`]) and the
//! per-slot device pointer accessors.
//!
//! Owner: model-engine SSM state pool.
//! Invariants:
//! - The free list starts as `[0, max_slots)` (`ssm_pool.rs`); the dummy slot at
//!   index `max_slots` is outside it.
//! - MTP accessors resolve a slot at or above `mtp_slots` to the MTP dummy at index
//!   `mtp_slots` ([`SsmStatePool::mtp_slot`]), so they never address past the MTP pools.

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
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl SsmStatePool {
    /// 2026-09-25: Map a pool slot onto the MTP verify pools, which cover
    /// `mtp_slots` slots (`ssm_reserve::mtp_state_slots`, equal to `max_slots` when
    /// `max_slots <= 32`). Covered slots map to themselves; any other slot maps to
    /// the MTP dummy at index `mtp_slots`. The scheduler's speculative branches in
    /// `lane_decode.rs` require every active slot to be covered (`spec_slots_covered`).
    #[inline]
    fn mtp_slot(&self, slot: usize) -> usize {
        if slot < self.mtp_slots {
            slot
        } else {
            self.mtp_slots
        }
    }

    /// 2026-10-10: On a hybrid serve (`METRALE_HYBRID_DFLASH=1`) the lowest free index, so
    /// a request that starts alone gets `ssm_reserve::HYBRID_DFLASH_SLOT`, the only slot sized
    /// for the DFlash verify. Both ranks claim and release in the same order, so they agree.
    pub(super) fn claim_slot(&self) -> Result<usize> {
        let mut free = self.free_slots.lock();
        // 2026-10-10: Under `METRALE_MTP_POOL_CAP` with an uncovered tail
        // (`ssm_reserve::mtp_pool_cap`), claim the lowest free slot so a batch of at most
        // `mtp_max_seqs` sequences sits on covered slots without compaction (EP v2 has
        // none). Both EP ranks run this rule, so the worker's mirrored claim still
        // returns the head's slot. Otherwise the last released slot (LIFO), as before.
        // 2026-10-11: The hybrid MTP + DFlash serve also claims lowest-first, so a lone
        // sequence takes slot 0 (the DFlash slot, `HYBRID_DFLASH_SLOT`).
        let cap_lowest =
            self.mtp_slots < self.max_slots && metrale_model_layers::ssm_reserve::mtp_pool_cap();
        let lowest_first =
            cap_lowest || metrale_model_layers::speculative::hybrid::hybrid_dflash_enabled();
        let slot = take_free_slot(&mut free, lowest_first);
        if cap_lowest {
            // 2026-10-10: Gate evidence (boundary probe): which slot each sequence gets.
            if let Some(s) = slot {
                tracing::info!(
                    "METRALE_MTP_POOL_CAP: claimed SSM slot {s} (cover {}, {} free)",
                    self.mtp_slots,
                    free.len()
                );
            }
        }
        slot.ok_or_else(|| {
            anyhow::anyhow!("SSM state pool exhausted (max {} slots)", self.max_slots)
        })
    }

    /// 2026-09-25: Claim a slot and wrap it in a [`SlotGuard`] that returns it to the
    /// free list when dropped while still owning it. The guard is stored on the
    /// owning [`SequenceState`]; see [`SlotGuard`] for the explicit release paths.
    pub(super) fn claim_guarded(self: &Arc<Self>) -> Result<SlotGuard> {
        let idx = self.claim_slot()?;
        Ok(SlotGuard {
            pool: Arc::clone(self),
            idx: Some(idx),
        })
    }

    pub(super) fn release_slot(&self, idx: usize) {
        let mut free = self.free_slots.lock();
        debug_assert!(
            !free.contains(&idx),
            "release_slot: slot {idx} already free (double-release hands it to two seqs)"
        );
        free.push(idx);
        drop(free);
        // 2026-10-10: `METRALE_SSM_NO_CHECKPOINT` gate evidence on every released slot (rank 0
        // and the EP worker both release through here): one atomic load plus one line, only
        // with the lever on. The harness snapshots rank logs before teardown.
        if metrale_model_layers::ssm_reserve::ssm_no_checkpoint() {
            tracing::info!(
                "METRALE_SSM_NO_CHECKPOINT: checkpoint reads refused = {}",
                metrale_model_layers::ssm_reserve::ssm_checkpoint_reads_refused()
            );
        }
    }

    /// 2026-09-25: Remove `slot` from the free list if present; returns whether it
    /// was. `compact_sequence` claims its migration target with this, so the target
    /// is not left on the free list while a sequence owns it.
    pub(super) fn claim_specific(&self, slot: usize) -> bool {
        let mut free = self.free_slots.lock();
        if let Some(pos) = free.iter().position(|&s| s == slot) {
            free.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// 2026-09-25: Whether `idx` is on the free list. Decode and verify graph
    /// borrowing (`decode_a2.rs`, `verify_e.rs`) use it to accept a cached graph
    /// whose extra rows address only free slots.
    pub(super) fn slot_is_free(&self, idx: usize) -> bool {
        self.free_slots.lock().contains(&idx)
    }

    /// 2026-10-10: The lowest free slot below `n`, left on the free list (the caller's
    /// `compact_sequence` claims it). `METRALE_MTP_POOL_CAP` slot migration picks its target
    /// with it (`migrate_to_covered_slot_dispatch`).
    pub(super) fn lowest_free_below(&self, n: usize) -> Option<usize> {
        self.free_slots.lock().iter().copied().filter(|&s| s < n).min()
    }

    /// 2026-09-25: The reserved slot at index `max_slots`, outside the free list's
    /// initial range; batched decode points padding rows at it.
    #[inline]
    pub(super) fn dummy_slot(&self) -> usize {
        self.max_slots
    }

    /// 2026-09-25: Queue zeroing of `idx`'s h and conv state in every SSM layer on
    /// `stream`. Sequence allocation (`trait_impl/meta.rs`) calls it before prefill
    /// so the new sequence does not start from a previous occupant's state.
    pub(super) fn zero_slot(&self, idx: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.memset_async(self.h_state(i, idx), 0, self.h_stored_bytes, stream)?;
            gpu.memset_async(self.conv_state(i, idx), 0, self.conv_bytes, stream)?;
        }
        Ok(())
    }

    pub(super) fn h_state(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.h_state_pools[ssm_layer_idx].offset(slot * self.h_stored_bytes)
    }

    /// 2026-09-25: This slot's FP32 prefill staging blob. `Some` only for an
    /// f16-sized h pool; one blob per slot, shared by all layers (see
    /// `h_prefill_stage_pool`).
    pub(super) fn h_prefill_stage(&self, slot: usize) -> Option<DevicePtr> {
        self.h_prefill_stage_pool
            .map(|p| p.offset(slot * self.h_bytes))
    }

    pub(super) fn conv_state(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.conv_state_pools[ssm_layer_idx].offset(slot * self.conv_bytes)
    }

    /// 2026-09-25: Diagnostic: synchronise `stream`, then log at WARN the signed sum,
    /// sum of squares and sum of absolute values of each SSM layer's h and conv
    /// state for `slot`, read as f32, plus the totals over all layers. The two
    /// cancellation-free sums catch divergence that a signed sum can hide. Returns
    /// early, silently, if a device read fails.
    pub(super) fn debug_state_checksum(
        &self,
        slot: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
        tag: &str,
    ) {
        gpu.synchronize(stream).ok();
        let mut g_h_sum = 0f64;
        let mut g_h_ssq = 0f64;
        let mut g_h_sabs = 0f64;
        let mut g_c_sum = 0f64;
        let mut g_c_ssq = 0f64;
        let mut g_c_sabs = 0f64;
        for i in 0..self.num_ssm_layers {
            // 2026-09-25: Read the storage width (`h_stored_bytes`); `h_bytes` would
            // run past an f16-sized slot. On such a slot the f32 sums are not values,
            // but equal bytes still give equal sums.
            let mut hb = vec![0u8; self.h_stored_bytes];
            let mut cb = vec![0u8; self.conv_bytes];
            if gpu.copy_d2h(self.h_state(i, slot), &mut hb).is_err() {
                return;
            }
            if gpu.copy_d2h(self.conv_state(i, slot), &mut cb).is_err() {
                return;
            }
            let (mut h_sum, mut h_ssq, mut h_sabs) = (0f64, 0f64, 0f64);
            for c in hb.chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64;
                h_sum += v;
                h_ssq += v * v;
                h_sabs += v.abs();
            }
            let (mut c_sum, mut c_ssq, mut c_sabs) = (0f64, 0f64, 0f64);
            for c in cb.chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64;
                c_sum += v;
                c_ssq += v * v;
                c_sabs += v.abs();
            }
            g_h_sum += h_sum;
            g_h_ssq += h_ssq;
            g_h_sabs += h_sabs;
            g_c_sum += c_sum;
            g_c_ssq += c_ssq;
            g_c_sabs += c_sabs;
            tracing::warn!(
                "METRALE_SSM_CKSUM[{tag}] slot={slot} L{i} \
                 h_sum={h_sum:.6} h_ssq={h_ssq:.6} h_sabs={h_sabs:.6} \
                 c_sum={c_sum:.6} c_ssq={c_ssq:.6} c_sabs={c_sabs:.6}"
            );
        }
        tracing::warn!(
            "METRALE_SSM_CKSUM[{tag}] slot={slot} GLOBAL \
             h_sum={g_h_sum:.6} h_ssq={g_h_ssq:.6} h_sabs={g_h_sabs:.6} \
             c_sum={g_c_sum:.6} c_ssq={g_c_ssq:.6} c_sabs={g_c_sabs:.6}"
        );
    }

    /// 2026-09-25: Fixed address of H intermediate `token_idx` of `slot` (after the
    /// MTP slot mapping). `token_idx` must be below that slot's `h_inter_counts`
    /// entry; debug builds assert it.
    pub(super) fn h_intermediate(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        token_idx: usize,
    ) -> DevicePtr {
        let slot = self.mtp_slot(slot);
        debug_assert!(
            token_idx < self.h_inter_counts[slot],
            "h_intermediate: token_idx {token_idx} >= slot {slot}'s tiered capacity {}",
            self.h_inter_counts[slot],
        );
        self.h_intermediate_pools[ssm_layer_idx]
            .offset((self.h_inter_offsets[slot] + token_idx) * self.h_stored_bytes)
    }

    /// 2026-09-25: Number of H intermediates allocated for `slot` after the MTP
    /// slot mapping; 0 without MTP.
    #[inline]
    pub(super) fn h_inter_count(&self, slot: usize) -> usize {
        if !self.has_mtp {
            return 0;
        }
        self.h_inter_counts[self.mtp_slot(slot)]
    }

    /// 2026-09-25: The deepest `num_drafts` a speculative step may dispatch to a
    /// sequence in `slot`, read from the allocated H intermediates.
    /// `Model::mtp_slot_draft_capacity` returns this (`trait_impl/mod.rs`).
    /// `usize::MAX` without MTP or under the replay rollback mode; 0 for a slot
    /// at or above `mtp_slots`.
    #[inline]
    pub(crate) fn verify_draft_capacity(&self, slot: usize) -> usize {
        if !self.has_mtp {
            return usize::MAX;
        }
        // 2026-09-25: Replay mode allocates no per-token H intermediates. Reporting
        // no limit lets a verify reach `require_verify_rollback_supported`, which
        // refuses it, instead of speculation being clamped to zero drafts.
        if self.rollback_mode == metrale_model_layers::ssm_reserve::SsmRollbackMode::Replay {
            return usize::MAX;
        }
        if slot >= self.mtp_slots {
            return 0;
        }
        // 2026-09-25: A verify of K rows (1 + drafts) writes K-1 H intermediates, so
        // the count equals the draft capacity.
        self.h_inter_counts[slot]
    }

    /// 2026-09-25: Error under the replay rollback mode on a model with SSM layers:
    /// that mode allocates no per-token H intermediates and has no replay device
    /// path. Every `decode_verify*` entry in `trait_impl/mod.rs` calls it.
    pub(crate) fn require_verify_rollback_supported(&self) -> Result<()> {
        if self.rollback_mode == metrale_model_layers::ssm_reserve::SsmRollbackMode::Replay
            && self.num_ssm_layers > 0
        {
            bail!(
                "--ssm-rollback-mode replay is an EXPERIMENTAL scaffold: the verify-window \
                 input capture and checkpoint-replay reconstruction are not wired yet, so \
                 speculative verify cannot run. The serve boots (reserve sizing shows the \
                 replay capacity win) but --speculative traffic must use \
                 --ssm-rollback-mode snapshot."
            );
        }
        Ok(())
    }

    pub(super) fn conv_intermediate(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        token_idx: usize,
    ) -> DevicePtr {
        let ni = self.num_intermediates;
        let slot = self.mtp_slot(slot);
        self.conv_intermediate_pools[ssm_layer_idx]
            .offset((slot * ni + token_idx) * self.conv_bytes)
    }

    /// 2026-10-10: Whether the pre-verify checkpoint pools exist: false under
    /// `METRALE_SSM_NO_CHECKPOINT=1` (and without MTP).
    pub(super) fn has_checkpoints(&self) -> bool {
        !self.h_checkpoint_pools.is_empty()
    }

    /// 2026-10-10: `None` when the checkpoint pools are not allocated
    /// ([`Self::has_checkpoints`]).
    pub(super) fn h_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> Option<DevicePtr> {
        self.h_checkpoint_pools
            .get(ssm_layer_idx)
            .map(|p| p.offset(self.mtp_slot(slot) * self.h_stored_bytes))
    }

    /// 2026-10-10: `None` when the checkpoint pools are not allocated.
    pub(super) fn conv_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> Option<DevicePtr> {
        self.conv_checkpoint_pools
            .get(ssm_layer_idx)
            .map(|p| p.offset(self.mtp_slot(slot) * self.conv_bytes))
    }

    pub(super) fn reset_slot(&self, slot: usize, gpu: &dyn GpuBackend) -> Result<()> {
        // 2026-09-25: A slot at or above `mtp_slots` has no MTP state of its own;
        // its MTP accessors resolve to the shared MTP dummy.
        let reset_mtp = self.has_mtp && slot < self.mtp_slots;
        for i in 0..self.num_ssm_layers {
            gpu.memset(self.h_state(i, slot), 0, self.h_stored_bytes)?;
            gpu.memset(self.conv_state(i, slot), 0, self.conv_bytes)?;
            if reset_mtp {
                for t in 0..self.h_inter_count(slot) {
                    gpu.memset(self.h_intermediate(i, slot, t), 0, self.h_stored_bytes)?;
                }
                for t in 0..self.num_intermediates {
                    gpu.memset(self.conv_intermediate(i, slot, t), 0, self.conv_bytes)?;
                }
                if let (Some(h), Some(c)) =
                    (self.h_checkpoint(i, slot), self.conv_checkpoint(i, slot))
                {
                    gpu.memset(h, 0, self.h_stored_bytes)?;
                    gpu.memset(c, 0, self.conv_bytes)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn copy_slot(
        &self,
        from: usize,
        to: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: MTP state is copied only between two covered slots. An
        // uncovered slot's MTP accessors resolve to the shared MTP dummy, whose
        // bytes are scratch.
        let copy_mtp = self.has_mtp && from < self.mtp_slots && to < self.mtp_slots;
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.h_state(i, from),
                self.h_state(i, to),
                self.h_stored_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_state(i, from),
                self.conv_state(i, to),
                self.conv_bytes,
                stream,
            )?;
            if copy_mtp {
                for t in 0..self.h_inter_count(from).min(self.h_inter_count(to)) {
                    gpu.copy_d2d_async(
                        self.h_intermediate(i, from, t),
                        self.h_intermediate(i, to, t),
                        self.h_stored_bytes,
                        stream,
                    )?;
                }
                for t in 0..self.num_intermediates {
                    gpu.copy_d2d_async(
                        self.conv_intermediate(i, from, t),
                        self.conv_intermediate(i, to, t),
                        self.conv_bytes,
                        stream,
                    )?;
                }
                if let (Some(hf), Some(ht), Some(cf), Some(ct)) = (
                    self.h_checkpoint(i, from),
                    self.h_checkpoint(i, to),
                    self.conv_checkpoint(i, from),
                    self.conv_checkpoint(i, to),
                ) {
                    gpu.copy_d2d_async(hf, ht, self.h_stored_bytes, stream)?;
                    gpu.copy_d2d_async(cf, ct, self.conv_bytes, stream)?;
                }
            }
        }
        Ok(())
    }
}

/// 2026-09-25: RAII owner of a claimed SSM pool slot, stored in
/// [`crate::traits::SequenceState`]'s `ssm_slot`. While `idx` is `Some`, dropping
/// the guard releases that slot. `free_sequence` (`trait_impl/sequence.rs`) calls
/// [`take`](Self::take) and releases the slot itself, or discards it when the
/// slot was handed on by compaction; `compact_sequence` takes the old slot,
/// releases it and calls [`migrate`](Self::migrate) with the new one. Each path
/// takes the index before releasing it, so the guard's `Drop` does not release
/// it a second time.
pub(crate) struct SlotGuard {
    pool: Arc<SsmStatePool>,
    idx: Option<usize>,
}

impl SlotGuard {
    /// 2026-09-25: A guard that owns no slot; its `Drop` releases nothing.
    pub(crate) fn empty(pool: Arc<SsmStatePool>) -> Self {
        Self { pool, idx: None }
    }

    #[inline]
    pub(crate) fn idx(&self) -> Option<usize> {
        self.idx
    }

    /// 2026-09-25: Return the owned slot index, if any, without releasing it. The
    /// caller becomes responsible for the release; the guard's `Drop` then
    /// releases nothing.
    #[inline]
    pub(crate) fn take(&mut self) -> Option<usize> {
        self.idx.take()
    }

    /// 2026-09-25: Point the guard at `new_idx`. The caller must have taken and
    /// released the old slot first; debug builds assert the guard is empty.
    #[inline]
    pub(crate) fn migrate(&mut self, new_idx: usize) {
        debug_assert!(
            self.idx.is_none(),
            "SlotGuard::migrate called before the old slot was released/taken"
        );
        self.idx = Some(new_idx);
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if let Some(idx) = self.idx.take() {
            // 2026-09-25: Reached only when no explicit path took the index.
            tracing::debug!("SlotGuard::drop releasing un-freed SSM slot {idx}");
            self.pool.release_slot(idx);
        }
    }
}

/// 2026-10-10: Take a slot from the free list: the lowest index when `lowest_first`
/// (`METRALE_MTP_POOL_CAP`), else the last pushed (LIFO). `None` when empty.
pub(super) fn take_free_slot(free: &mut Vec<usize>, lowest_first: bool) -> Option<usize> {
    if lowest_first {
        let i = free
            .iter()
            .enumerate()
            .min_by_key(|&(_, &s)| s)
            .map(|(i, _)| i)?;
        Some(free.swap_remove(i))
    } else {
        free.pop()
    }
}
