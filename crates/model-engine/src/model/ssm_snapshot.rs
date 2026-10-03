// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU pool of SSM state snapshots: the Marconi prefix-cache region and
//! the per-sequence decode-rollback ring.
//!
//! Owner: model-engine SSM snapshot pool.
//! Invariants:
//! - `save` writes FP32 h state: it widens an f16 source slot, and returns an error
//!   when it cannot.
//! - A slot popped off the free list, and a slot pushed back by `free`, has no entry
//!   in `slot_has_hidden`, `session_tags` or `aux_blobs` (`clear_slot_bookkeeping`).

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
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// 2026-09-25: Pre-allocated GPU snapshots of every SSM layer's h and conv state.
/// Two regions with separate allocations:
///
/// 1. Marconi prefix caching: slots `[0, num_slots)`, taken from `free_slots` by
///    [`save`](Self::save) and returned by [`free`](Self::free). A prefix-cache hit
///    restores a snapshot instead of recomputing the cached tokens.
/// 2. Decode-time rollback: `decode_ring_slots` slots per SSM pool slot, with no free
///    list. Ring slot `r` of pool slot `s` is flat index `s * decode_ring_slots + r`
///    (`ssm_snapshot_decode.rs`).
pub(crate) struct SsmSnapshotPool {
    pub(super) h_snapshots: Vec<DevicePtr>,
    pub(super) conv_snapshots: Vec<DevicePtr>,
    pub(super) free_slots: Mutex<Vec<usize>>,
    pub(super) num_slots: usize,
    pub(super) h_bytes: usize,
    pub(super) conv_bytes: usize,
    pub(super) num_ssm_layers: usize,
    /// 2026-09-25: Snapshot slot -> session hash. [`Self::session_matches`] rejects a
    /// slot tagged with another session.
    pub(super) session_tags: Mutex<std::collections::HashMap<usize, u64>>,
    /// 2026-09-25: Per-slot aux layer state as host blobs `(layer_idx, bytes)`, set by
    /// [`Self::set_aux`]. A model that `requires_aux_state` restores a slot only
    /// when it has aux (`trait_impl/prefill_a.rs`, `prefill_c.rs`).
    pub(super) aux_blobs: Mutex<std::collections::HashMap<usize, Vec<(u32, Vec<u8>)>>>,
    /// 2026-10-03: Host-byte accounting for `aux_blobs` (`ssm_snapshot_auxbudget.rs`): the live
    /// total and, per slot, its bytes and (for a grid checkpoint) its owning sequence.
    pub(super) aux_meta: Mutex<super::ssm_snapshot_auxbudget::AuxMeta>,
    /// 2026-09-25: Decode-rollback h region, one allocation per layer of
    /// `decode_max_seqs * decode_ring_slots * h_bytes`. Empty when the ring is disabled.
    pub(super) decode_h_snapshots: Vec<DevicePtr>,
    pub(super) decode_conv_snapshots: Vec<DevicePtr>,
    /// 2026-09-25: Ring slots per SSM pool slot; 0 when the ring is disabled.
    pub(super) decode_ring_slots: usize,
    /// 2026-09-25: SSM pool slots the ring is sized for; `TransformerModel::new`
    /// passes `max_batch_size`. A slot index at or above it is an error.
    pub(super) decode_max_seqs: usize,
    /// 2026-09-25: Last-token post-final-norm hidden state per Marconi slot, one
    /// buffer of `num_slots * hidden_bytes`; NULL when Marconi is disabled. A leaf
    /// snapshot holds the SSM state after the last prompt token, so re-running that
    /// token for the first logits would apply its SSM update twice; an exact
    /// full-prompt hit feeds this hidden state to `lm_head` instead.
    pub(super) hidden_snapshot: DevicePtr,
    /// 2026-09-25: Bytes of one slot's last-token hidden state; `TransformerModel::new`
    /// passes `hidden_size * 2` (BF16).
    pub(super) hidden_bytes: usize,
    /// 2026-09-25: Marconi slots whose `hidden_snapshot` entry was written by
    /// [`Self::save_hidden`].
    pub(super) slot_has_hidden: Mutex<std::collections::HashSet<usize>>,
    /// 2026-09-25: f16 -> FP32 h-state kernel used by `save` for an f16 source slot
    /// (`--ssm-h-dtype f16`). `KernelHandle(0)` when the kernel is not loaded.
    pub(super) h_f16_to_f32_k: KernelHandle,
    /// 2026-09-25: FP32 -> f16 h-state kernel used by `restore` into an f16-sized pool
    /// slot, which a plain `h_bytes` copy would overrun. `KernelHandle(0)` when the
    /// kernel is not loaded.
    pub(super) h_f32_to_f16_k: KernelHandle,
    /// 2026-09-25: Reusable page-locked staging blob for the tier spill and fault-in
    /// paths ([`super::ssm_spill_staging::SpillStaging`]); `TransformerModel::drop`
    /// frees it through `free_staging`.
    pub(super) spill_staging: super::ssm_spill_staging::SpillStaging,
}

impl SsmSnapshotPool {
    /// 2026-09-25: Marconi occupancy `(slots not on the free list, num_slots)`. The
    /// decode-rollback ring is not counted.
    pub(super) fn occupancy(&self) -> (usize, usize) {
        (
            self.num_slots - self.free_slots.lock().len().min(self.num_slots),
            self.num_slots,
        )
    }

    pub(super) fn is_enabled(&self) -> bool {
        self.num_slots > 0
    }

    /// 2026-09-25: Queue a copy of pool slot `ssm_slot`'s state into a free Marconi
    /// slot, tag it with `session_hash` when non-zero, and return the slot; `None`
    /// when the region is disabled or full. `h_is_f16` is the source slot's h dtype;
    /// an f16 source is widened, so the snapshot is always FP32. Errors when the
    /// pool is f16-sized and the source is not flagged f16, or when the source is
    /// f16 and the widening kernel is not loaded.
    pub(super) fn save(
        &self,
        ssm_slot: usize,
        session_hash: u64,
        h_is_f16: bool,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<usize>> {
        if !self.is_enabled() {
            return Ok(None);
        }
        if main_pool.h_stored_bytes < self.h_bytes && !h_is_f16 {
            bail!(
                "f16-sized SSM h pool: cannot snapshot an FP32-flagged state out of a \
                 2-byte-sized pool slot (the copy would overrun the slot). Prefill has \
                 not narrowed this sequence's h-state — stage 3 is not serveable yet."
            );
        }
        if h_is_f16 && self.h_f16_to_f32_k.0 == 0 {
            bail!(
                "METRALE_SSM_H_FP16: cannot widen a decode-produced snapshot —                  ssm_h_dtype::ssm_h_state_f16_to_f32 did not resolve"
            );
        }
        let snap_slot = match self.claim_free_slot() {
            Some(s) => s,
            None => return Ok(None),
        };
        for i in 0..self.num_ssm_layers {
            if h_is_f16 {
                metrale_model_layers::layers::ops::ssm_h_state_f16_to_f32(
                    gpu,
                    self.h_f16_to_f32_k,
                    main_pool.h_state(i, ssm_slot),
                    self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                    (self.h_bytes / 4) as u64,
                    stream,
                )?;
            } else {
                gpu.copy_d2d_async(
                    main_pool.h_state(i, ssm_slot),
                    self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                    self.h_bytes,
                    stream,
                )?;
            }
            gpu.copy_d2d_async(
                main_pool.conv_state(i, ssm_slot),
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                self.conv_bytes,
                stream,
            )?;
        }
        if session_hash != 0 {
            self.session_tags.lock().insert(snap_slot, session_hash);
        }
        Ok(Some(snap_slot))
    }

    /// 2026-09-25: True when `session_hash` is 0, the slot is untagged, or its tag
    /// equals `session_hash`.
    pub(super) fn session_matches(&self, snap_slot: usize, session_hash: u64) -> bool {
        if session_hash == 0 {
            return true;
        }
        let tags = self.session_tags.lock();
        match tags.get(&snap_slot) {
            None => true,
            Some(&tag) => tag == session_hash,
        }
    }

    /// 2026-09-25: Queue a copy of Marconi slot `snap_slot` into pool slot
    /// `ssm_slot`. For an f16-sized pool the FP32 h snapshot is narrowed; that
    /// errors when the narrowing kernel is not loaded.
    pub(super) fn restore(
        &self,
        snap_slot: usize,
        ssm_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let narrow = main_pool.h_stored_bytes < self.h_bytes;
        if narrow && self.h_f32_to_f16_k.0 == 0 {
            bail!(
                "f16-sized SSM h pool: cannot restore an FP32 snapshot into a \
                 2-byte-sized pool slot — ssm_h_dtype::ssm_h_state_f32_to_f16 did \
                 not resolve on this target"
            );
        }
        for i in 0..self.num_ssm_layers {
            if narrow {
                metrale_model_layers::layers::ops::ssm_h_state_f32_to_f16(
                    gpu,
                    self.h_f32_to_f16_k,
                    self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                    main_pool.h_state(i, ssm_slot),
                    (self.h_bytes / 4) as u64,
                    stream,
                )?;
            } else {
                gpu.copy_d2d_async(
                    self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                    main_pool.h_state(i, ssm_slot),
                    self.h_bytes,
                    stream,
                )?;
            }
            gpu.copy_d2d_async(
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                main_pool.conv_state(i, ssm_slot),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: Pop a free slot and clear its side tables, so it carries nothing
    /// from its previous holder. `save` and `reserve_tail_slot` acquire through here;
    /// `try_pop_free_slot` (`ssm_snapshot_spill.rs`) pops and clears the same way. A
    /// stale `aux_blobs` entry would not be caught later: the restore gate checks
    /// only `has_aux(snap_id)` (`trait_impl/prefill_a.rs`).
    fn claim_free_slot(&self) -> Option<usize> {
        let snap_slot = self.free_slots.lock().pop()?;
        self.clear_slot_bookkeeping(snap_slot);
        Some(snap_slot)
    }

    /// 2026-09-25: Remove `snap_slot` from every side table. A new side table belongs
    /// here, which keeps `free` and the acquire paths in agreement.
    pub(super) fn clear_slot_bookkeeping(&self, snap_slot: usize) {
        self.slot_has_hidden.lock().remove(&snap_slot);
        self.session_tags.lock().remove(&snap_slot);
        self.aux_blobs.lock().remove(&snap_slot);
        self.aux_meta.lock().remove(snap_slot);
    }

    /// 2026-09-25: Clear the slot's side tables and push it on the free list. A tag
    /// left behind would make [`Self::session_has_history`] count a slot that holds
    /// no restorable state.
    pub(super) fn free(&self, snap_slot: usize) {
        self.clear_slot_bookkeeping(snap_slot);
        self.free_slots.lock().push(snap_slot);
    }

    pub(super) fn set_aux(&self, snap_slot: usize, blobs: Vec<(u32, Vec<u8>)>) {
        let bytes: usize = blobs.iter().map(|(_, b)| b.len()).sum();
        self.aux_blobs.lock().insert(snap_slot, blobs);
        self.aux_meta.lock().set(snap_slot, bytes, None);
    }

    /// 2026-10-03: [`Self::set_aux`] for a grid checkpoint saved by sequence `owner`
    /// (its SSM pool slot) at `tokens` tokens: the budget evicts the owner's earlier
    /// checkpoints first (`make_room_for_aux`).
    pub(super) fn set_aux_owned(
        &self,
        snap_slot: usize,
        blobs: Vec<(u32, Vec<u8>)>,
        owner: u64,
        tokens: usize,
    ) {
        let bytes: usize = blobs.iter().map(|(_, b)| b.len()).sum();
        self.aux_blobs.lock().insert(snap_slot, blobs);
        self.aux_meta
            .lock()
            .set(snap_slot, bytes, Some((owner, tokens)));
    }

    pub(super) fn aux(&self, snap_slot: usize) -> Option<Vec<(u32, Vec<u8>)>> {
        self.aux_blobs.lock().get(&snap_slot).cloned()
    }

    /// 2026-10-01: Whether `snap_slot` carries aux blobs, equal to `aux(snap_slot).is_some()`
    /// without copying them. The restore gates ask only this; [`Self::aux`] copies the blobs
    /// (5,643 B/token for GLM-5.3, ~170 MB at 30K) and is left to the restore itself.
    pub(super) fn has_aux(&self, snap_slot: usize) -> bool {
        self.aux_blobs.lock().contains_key(&snap_slot)
    }

    /// 2026-09-25: Whether any tagged slot carries the non-zero `session_hash`. The
    /// mid-chunk tail capture (`prefill_b/midchunk_capture.rs`) skips a sequence with
    /// no cached prefix and no history.
    pub(crate) fn session_has_history(&self, session_hash: u64) -> bool {
        session_hash != 0
            && self
                .session_tags
                .lock()
                .values()
                .any(|&s| s == session_hash)
    }

    /// 2026-09-25: Reserve a Marconi slot for a mid-chunk tail capture: pop a clean
    /// slot and tag it with `session_hash` when non-zero, as `save` does. `None` when
    /// the region is disabled or full.
    pub(crate) fn reserve_tail_slot(&self, session_hash: u64) -> Option<usize> {
        if !self.is_enabled() {
            return None;
        }
        let snap_slot = self.claim_free_slot()?;
        if session_hash != 0 {
            self.session_tags.lock().insert(snap_slot, session_hash);
        }
        Some(snap_slot)
    }

    /// 2026-09-25: Device address of `snap_slot`'s h snapshot in SSM layer `ssm_layer`.
    pub(crate) fn tail_h_dst(&self, ssm_layer: usize, snap_slot: usize) -> DevicePtr {
        self.h_snapshots[ssm_layer].offset(snap_slot * self.h_bytes)
    }

    pub(crate) fn tail_conv_dst(&self, ssm_layer: usize, snap_slot: usize) -> DevicePtr {
        self.conv_snapshots[ssm_layer].offset(snap_slot * self.conv_bytes)
    }

    pub(crate) fn h_bytes(&self) -> usize {
        self.h_bytes
    }

    pub(crate) fn conv_bytes(&self) -> usize {
        self.conv_bytes
    }

    pub(crate) fn num_ssm_layers(&self) -> usize {
        self.num_ssm_layers
    }

    /// 2026-09-25: Queue a copy of the last-token hidden state (`hidden_bytes`) into
    /// `snap_slot` and mark the slot in `slot_has_hidden` (see `hidden_snapshot`).
    /// Does nothing when the Marconi region is disabled.
    pub(super) fn save_hidden(
        &self,
        snap_slot: usize,
        last_hidden: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if !self.is_enabled() || self.hidden_snapshot.is_null() {
            return Ok(());
        }
        gpu.copy_d2d_async(
            last_hidden,
            self.hidden_snapshot.offset(snap_slot * self.hidden_bytes),
            self.hidden_bytes,
            stream,
        )?;
        self.slot_has_hidden.lock().insert(snap_slot);
        Ok(())
    }

    pub(super) fn has_hidden(&self, snap_slot: usize) -> bool {
        self.slot_has_hidden.lock().contains(&snap_slot)
    }

    /// 2026-09-25: Queue a copy of `snap_slot`'s last-token hidden state into `dst`;
    /// errors when the region was not allocated.
    pub(super) fn restore_hidden(
        &self,
        snap_slot: usize,
        dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if self.hidden_snapshot.is_null() {
            bail!("SSM hidden snapshot region not allocated");
        }
        gpu.copy_d2d_async(
            self.hidden_snapshot.offset(snap_slot * self.hidden_bytes),
            dst,
            self.hidden_bytes,
            stream,
        )?;
        Ok(())
    }
}
