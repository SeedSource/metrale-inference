// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: SSM state copies for speculative verify on `secondary_stream`:
//! checkpoint, rollback, and commit of the accepted prefix; plus the event that
//! orders snapshot saves before a warm snapshot restore.
//!
//! Owner: model-engine (SSM state).
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
    pub(super) fn start_checkpoint_async_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        use metrale_model_layers::layer::SsmLayerState;

        let stream = self.secondary_stream;
        // 2026-10-10: `METRALE_SSM_NO_CHECKPOINT=1`: no copies and no lazy alloc; the event
        // is still recorded so `sync_secondary_dispatch` has something to wait on.
        if metrale_model_layers::ssm_reserve::ssm_no_checkpoint() {
            self.gpu.record_event(self.secondary_event, stream)?;
            return Ok(());
        }
        let mut h_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        let mut conv_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                // 2026-09-25: The pool's h storage width (`ssm_reserve::ssm_h_stored_bytes`).
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
        // 2026-09-25: The event `sync_secondary_dispatch` makes the default stream wait on.
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }

    pub(super) fn start_rollback_and_checkpoint_async_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        use metrale_model_layers::layer::SsmLayerState;

        let stream = self.secondary_stream;
        // 2026-10-10: `METRALE_SSM_NO_CHECKPOINT=1`: a rollback to 0 would need the
        // checkpoint; refuse loudly (counted for the gate). num_accepted >= 1 restores an
        // intermediate as before; only the checkpoint copies are skipped.
        let no_ckpt = metrale_model_layers::ssm_reserve::ssm_no_checkpoint();
        if no_ckpt && num_accepted == 0 {
            let n = metrale_model_layers::ssm_reserve::note_ssm_checkpoint_read_refused();
            bail!(
                "start_rollback_and_checkpoint_async: num_accepted == 0 needs the pre-verify \
                 SSM checkpoint, which METRALE_SSM_NO_CHECKPOINT=1 does not allocate \
                 (ngram / self-speculative are not supported with it). \
                 METRALE_SSM_NO_CHECKPOINT: checkpoint reads refused = {n}"
            );
        }
        let mut ssm_layer_idx = 0usize;
        // 2026-09-25: All rollback copies are issued before all checkpoint copies,
        // on one stream, so each checkpoint reads the state its rollback wrote.
        // Layer order within a plan does not matter: layers' buffers are disjoint.
        let n_ssm = self.ssm_pool.num_ssm_layers;
        let mut h_back = Vec::with_capacity(n_ssm);
        let mut conv_back = Vec::with_capacity(n_ssm);
        let mut h_ckpt = Vec::with_capacity(n_ssm);
        let mut conv_ckpt = Vec::with_capacity(n_ssm);

        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                // 2026-09-25: The pool's h storage width (`ssm_reserve::ssm_h_stored_bytes`).
                let h_bytes = self.ssm_pool.h_stored_bytes;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if num_accepted == 0 {
                    if let Some(ckpt) = ssm.h_state_checkpoint {
                        h_back.push(StateCopy {
                            src: ckpt,
                            dst: ssm.h_state,
                            bytes: h_bytes,
                        });
                    }
                    if let Some(ckpt) = ssm.conv_state_checkpoint {
                        conv_back.push(StateCopy {
                            src: ckpt,
                            dst: ssm.conv_state,
                            bytes: conv_bytes,
                        });
                    }
                } else {
                    let slot = seq.slot_idx;
                    let inter_idx = num_accepted - 1;
                    h_back.push(StateCopy {
                        src: self.ssm_pool.h_intermediate(ssm_layer_idx, slot, inter_idx),
                        dst: ssm.h_state,
                        bytes: h_bytes,
                    });
                    conv_back.push(StateCopy {
                        src: self
                            .ssm_pool
                            .conv_intermediate(ssm_layer_idx, slot, inter_idx),
                        dst: ssm.conv_state,
                        bytes: conv_bytes,
                    });
                }

                // 2026-09-25: Checkpoint the rolled-back state for the next verify.
                // 2026-10-10: Skipped under `METRALE_SSM_NO_CHECKPOINT=1`.
                if let Some(ckpt) = ssm.h_state_checkpoint.filter(|_| !no_ckpt) {
                    h_ckpt.push(StateCopy {
                        src: ssm.h_state,
                        dst: ckpt,
                        bytes: h_bytes,
                    });
                }
                if let Some(ckpt) = ssm.conv_state_checkpoint.filter(|_| !no_ckpt) {
                    conv_ckpt.push(StateCopy {
                        src: ssm.conv_state,
                        dst: ckpt,
                        bytes: conv_bytes,
                    });
                }

                ssm_layer_idx += 1;
            }
        }
        run_ssm_state_copies(self.gpu.as_ref(), &h_back, &conv_back, stream)?;
        run_ssm_state_copies(self.gpu.as_ref(), &h_ckpt, &conv_ckpt, stream)?;
        // 2026-09-25: The event `sync_secondary_dispatch` makes the default stream wait on.
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }

    pub(super) fn sync_secondary_dispatch(&self) -> Result<()> {
        // 2026-09-25: The default stream waits for `secondary_event` on the GPU;
        // the host does not block.
        self.gpu
            .stream_wait_event(self.gpu.default_stream(), self.secondary_event)
    }

    /// 2026-09-25: Record the snapshot-ordering event on `save_stream` after an
    /// SSM-snapshot save's D2D copies are enqueued. The warm restore in
    /// `prefill_b/prefix_lookup.rs` waits on it ([`Self::wait_snapshot_saves_dispatch`])
    /// before reading a snapshot slot. The `snapshot_event` doc in types.rs
    /// describes the race.
    pub(super) fn record_snapshot_save_dispatch(&self, save_stream: u64) -> Result<()> {
        self.gpu.record_event(self.snapshot_event, save_stream)
    }

    /// 2026-09-25: Make `restore_stream` wait on the snapshot-ordering event, so it
    /// runs after every snapshot save recorded so far. The host does not block.
    pub(super) fn wait_snapshot_saves_dispatch(&self, restore_stream: u64) -> Result<()> {
        self.gpu
            .stream_wait_event(restore_stream, self.snapshot_event)
    }

    /// 2026-09-25: Commit a verify step's accepted prefix. The verify kernels
    /// update `h_state`/`conv_state` in place, so:
    ///
    /// - `num_accepted == k`: the state after the last token is already live;
    ///   nothing is copied.
    /// - `0 < num_accepted < k`: copy the intermediates after the last accepted
    ///   token (`[num_accepted - 1]`) into `h_state` and `conv_state`. The `h`
    ///   copy is skipped when `gdn_fold_accepted_dispatch` already folded this
    ///   slot (`gdn_woa_folded_slots`).
    /// - `num_accepted > k` or `num_accepted == 0`: `Err`.
    ///
    /// Runs on `secondary_stream`; pair with `sync_secondary`.
    pub(super) fn commit_accepted_prefix_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        use metrale_model_layers::layer::SsmLayerState;

        // 2026-09-26: A carried verify's verdict is already committed
        // (`gdn_carry_commit`): pending rows and any restore, h and conv alike.
        if self.gdn_carry_take_committed(seq.slot_idx) {
            return Ok(());
        }

        // 2026-09-25: Consume this slot's entry from the write-on-accept fold: when
        // present, the fold already wrote the accepted `h` state.
        let h_folded = {
            let mut f = self.gdn_woa_folded_slots.lock();
            match f.iter().position(|&s| s == seq.slot_idx) {
                Some(p) => {
                    f.swap_remove(p);
                    true
                }
                None => false,
            }
        };

        // 2026-09-25: With the full-accept return and the `num_accepted == 0` check
        // below, this keeps the intermediate index in `[0, k - 2]`.
        // `num_accepted > k` commits more tokens than were verified: the shape of
        // a bonus-token off-by-one, such as a caller passing the draft count as `k`.
        if num_accepted > k {
            anyhow::bail!(
                "commit_accepted_prefix: num_accepted ({num_accepted}) > k ({k}) — more \
                 tokens committed than were verified. Check that the caller's `k` is the \
                 VERIFY WIDTH (drafts + 1), not the draft count."
            );
        }

        // 2026-10-09: `METRALE_GLM_KDA_DEFER_COMMIT=1` (`kda_defer_hooks.rs`): after a consumer
        // verify a partial accept records its intermediate instead of copying it; the next verify
        // reads it there, or the next other entry point copies it. A no-op when not capable.
        if self.kda_defer_commit(seq, num_accepted, k, h_folded)? {
            self.gpu.record_event(self.secondary_event, self.secondary_stream)?;
            return Ok(());
        }

        if num_accepted == k {
            return Ok(());
        }

        // 2026-09-25: `num_accepted == 0` has no intermediate to rewind to:
        // `num_accepted - 1` would wrap in a release build and hand
        // `h_intermediate()` an out-of-range index. A full-reject rewind is
        // `rollback_ssm_states`, which restores the pre-verify checkpoint.
        if num_accepted == 0 {
            anyhow::bail!(
                "commit_accepted_prefix: num_accepted == 0 (k={k}) has no intermediate to \
                 rewind to — position 0 of a verify batch is accepted by construction. \
                 Use rollback_ssm_states() for a full-reject rewind to the pre-verify \
                 checkpoint."
            );
        }

        let stream = self.secondary_stream;
        let mut ssm_layer_idx = 0usize;
        // 2026-09-25: Two plans, because h and conv blobs have different widths and
        // one pitched 2-D copy carries one width. Both are in ascending layer order.
        let mut h_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        let mut conv_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

            let nv = self.config.linear_num_value_heads;
            let vd = self.config.linear_value_head_dim;
            let nk = self.config.linear_num_key_heads;
            let kd = self.config.linear_key_head_dim;
            // 2026-09-25: The pool's h storage width (`ssm_reserve::ssm_h_stored_bytes`).
            let h_bytes = self.ssm_pool.h_stored_bytes;
            let conv_bytes = (nk * kd * 2 + nv * vd) * self.config.linear_conv_kernel_dim * 4;

            let slot = seq.slot_idx;
            let inter_idx = num_accepted - 1;
            if !h_folded {
                h_plan.push(StateCopy {
                    src: self.ssm_pool.h_intermediate(ssm_layer_idx, slot, inter_idx),
                    dst: ssm.h_state,
                    bytes: h_bytes,
                });
            }
            conv_plan.push(StateCopy {
                src: self
                    .ssm_pool
                    .conv_intermediate(ssm_layer_idx, slot, inter_idx),
                dst: ssm.conv_state,
                bytes: conv_bytes,
            });

            ssm_layer_idx += 1;
        }
        run_ssm_state_copies(self.gpu.as_ref(), &h_plan, &conv_plan, stream)?;
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }
}
