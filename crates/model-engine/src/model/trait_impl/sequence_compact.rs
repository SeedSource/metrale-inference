// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: SSM pool slot migration: move a sequence's SSM state to another slot
//! (`compact_sequence_dispatch`), and disown a slot another sequence took over
//! (`detach_slot_for_reuse_dispatch`).
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
}
