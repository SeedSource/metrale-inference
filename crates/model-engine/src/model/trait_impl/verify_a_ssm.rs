// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: SSM state around speculative verify (checkpoint before, roll back to the
//! last accepted token after) and the decode-time rollback ring's save and restore.
//!
//! Owner: model-engine.
//! Invariants: `rollback_ssm_states_dispatch` enqueues its copies only after every layer's
//! copy is planned, so an error while validating or planning enqueues none.

// SPDX-License-Identifier: MIT OR Apache-2.0

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
use super::super::ssm_batched_copy::{StateCopy, run_ssm_state_copies};
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
    pub(super) fn checkpoint_ssm_states_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        use metrale_model_layers::layer::SsmLayerState;

        // 2026-10-10: A stray ngram / self-spec call must fail loudly, never lazily
        // allocate per-sequence checkpoint memory (counted for the gate).
        if metrale_model_layers::ssm_reserve::ssm_no_checkpoint() {
            let n = metrale_model_layers::ssm_reserve::note_ssm_checkpoint_read_refused();
            anyhow::bail!(
                "checkpoint_ssm_states: METRALE_SSM_NO_CHECKPOINT=1 allocates no SSM \
                 checkpoint (ngram / self-speculative are not supported with it). \
                 METRALE_SSM_NO_CHECKPOINT: checkpoint reads refused = {n}"
            );
        }

        let stream = self.gpu.default_stream();
        let mut h_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        let mut conv_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == metrale_config::LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                // 2026-09-25: The pool's h storage width (`ssm_h_stored_bytes`), which is half
                // the FP32 size under the f16-sized pool.
                let h_bytes = self.ssm_pool.h_stored_bytes;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if ssm.h_state_checkpoint.is_none() {
                    ssm.h_state_checkpoint = Some(self.gpu.alloc(h_bytes)?);
                }
                if ssm.conv_state_checkpoint.is_none() {
                    ssm.conv_state_checkpoint = Some(self.gpu.alloc(conv_bytes)?);
                }

                h_plan.push(StateCopy {
                    src: ssm.h_state,
                    dst: ssm.h_state_checkpoint.unwrap(),
                    bytes: h_bytes,
                });
                conv_plan.push(StateCopy {
                    src: ssm.conv_state,
                    dst: ssm.conv_state_checkpoint.unwrap(),
                    bytes: conv_bytes,
                });
            }
        }
        run_ssm_state_copies(self.gpu.as_ref(), &h_plan, &conv_plan, stream)?;
        self.gpu.synchronize(stream)?;
        Ok(())
    }

    pub(super) fn rollback_ssm_states_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        use metrale_model_layers::layer::SsmLayerState;

        // 2026-10-10: A rollback to 0 restores the checkpoint, which `METRALE_SSM_NO_CHECKPOINT=1`
        // does not allocate (counted for the gate).
        if num_accepted == 0 && metrale_model_layers::ssm_reserve::ssm_no_checkpoint() {
            let n = metrale_model_layers::ssm_reserve::note_ssm_checkpoint_read_refused();
            anyhow::bail!(
                "rollback_ssm_states: num_accepted == 0 needs the pre-verify SSM checkpoint, \
                 which METRALE_SSM_NO_CHECKPOINT=1 does not allocate (ngram / \
                 self-speculative are not supported with it). \
                 METRALE_SSM_NO_CHECKPOINT: checkpoint reads refused = {n}"
            );
        }

        // 2026-09-25: Validate every SSM layer before planning any copy: an error part-way
        // through the copies would leave some layers rewound and the rest advanced.
        if num_accepted > 0 {
            for (i, layer_state) in seq.layer_states.iter().enumerate() {
                if self.config.layer_type(i) != metrale_config::LayerType::LinearAttention {
                    continue;
                }
                let ssm = layer_state
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
                if num_accepted > ssm.h_state_intermediates.len() {
                    anyhow::bail!(
                        "rollback_ssm_states: cannot restore SSM to N={num_accepted} \
                         (layer {i}): only {} per-token intermediate(s) available. \
                         With no intermediates this is the self-speculative / ngram \
                         path — use --speculative (MTP) or --num-drafts 1 for SSM \
                         models. With too few, the MTP h-intermediate pool \
                         (num_drafts per slot, tiered — K-1 snapshots for a \
                         K-row verify) is smaller than this rollback target. \
                         No rollback copies were enqueued.",
                        ssm.h_state_intermediates.len(),
                    );
                }
            }
        }

        let stream = self.gpu.default_stream();
        let mut h_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        let mut conv_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == metrale_config::LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let kd = self.config.linear_key_head_dim;
                let nk = self.config.linear_num_key_heads;
                let h_bytes = self.ssm_pool.h_stored_bytes;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if num_accepted == 0 {
                    if let Some(ckpt) = ssm.h_state_checkpoint {
                        h_plan.push(StateCopy {
                            src: ckpt,
                            dst: ssm.h_state,
                            bytes: h_bytes,
                        });
                    }
                    if let Some(ckpt) = ssm.conv_state_checkpoint {
                        conv_plan.push(StateCopy {
                            src: ckpt,
                            dst: ssm.conv_state,
                            bytes: conv_bytes,
                        });
                    }
                } else if num_accepted <= ssm.h_state_intermediates.len() {
                    // 2026-09-25: Intermediate `n - 1` holds the state after accepted token n.
                    let idx = num_accepted - 1;
                    h_plan.push(StateCopy {
                        src: ssm.h_state_intermediates[idx],
                        dst: ssm.h_state,
                        bytes: h_bytes,
                    });
                    conv_plan.push(StateCopy {
                        src: ssm.conv_state_intermediates[idx],
                        dst: ssm.conv_state,
                        bytes: conv_bytes,
                    });
                } else {
                    // 2026-09-25: Unreachable: the validation pass bailed for every
                    // `num_accepted > intermediates.len()`, and 0 took the first branch.
                    unreachable!(
                        "rollback_ssm_states: layer {i} passed pre-validation but \
                         num_accepted={num_accepted} exceeds {} intermediates",
                        ssm.h_state_intermediates.len(),
                    );
                }
            }
        }
        // 2026-09-25: All copies are enqueued together, after every layer was planned.
        run_ssm_state_copies(self.gpu.as_ref(), &h_plan, &conv_plan, stream)?;
        Ok(())
    }

    /// 2026-09-25: Decode-time rollback: copy the sequence's live SSM state (pool slot
    /// `seq.slot_idx`) into ring slot `(seq.slot_idx, ring_slot)` of [`SsmSnapshotPool`], on
    /// the default stream. Errors when the decode-rollback region is not allocated.
    pub(super) fn save_decode_ssm_snapshot_dispatch(
        &self,
        seq: &SequenceState,
        ring_slot: usize,
    ) -> Result<()> {
        if !self.ssm_snapshots.decode_rollback_enabled() {
            anyhow::bail!("save_decode_ssm_snapshot: decode-rollback region not allocated");
        }
        let stream = self.gpu.default_stream();
        self.ssm_snapshots.save_decode(
            seq.slot_idx,
            ring_slot,
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        )
    }

    /// 2026-09-25: Inverse of [`Self::save_decode_ssm_snapshot_dispatch`]: copy ring snapshot
    /// `(seq.slot_idx, ring_slot)` back into the live pool slot. Errors when the
    /// decode-rollback region is not allocated.
    pub(super) fn restore_decode_ssm_snapshot_dispatch(
        &self,
        seq: &SequenceState,
        ring_slot: usize,
    ) -> Result<()> {
        if !self.ssm_snapshots.decode_rollback_enabled() {
            anyhow::bail!("restore_decode_ssm_snapshot: decode-rollback region not allocated");
        }
        let stream = self.gpu.default_stream();
        self.ssm_snapshots.restore_decode(
            seq.slot_idx,
            ring_slot,
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        )
    }
}
