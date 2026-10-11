// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Drop` for `TransformerModel`: frees the host staging buffers
//! owned by the model and by its SSM snapshot pool.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

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

impl Drop for TransformerModel {
    fn drop(&mut self) {
        self.drop_pinned_staging();
        // 2026-10-10: Shutdown stat for the `METRALE_SSM_NO_CHECKPOINT` gate: the number of
        // checkpoint reads/writes refused in this process (must be 0 on an MTP serve).
        if metrale_model_layers::ssm_reserve::ssm_no_checkpoint() {
            tracing::info!(
                "METRALE_SSM_NO_CHECKPOINT: checkpoint reads refused = {}",
                metrale_model_layers::ssm_reserve::ssm_checkpoint_reads_refused()
            );
        }
        // 2026-10-10: Shutdown stat for the `METRALE_MTP_POOL_CAP` gate (rank 0 counts;
        // the worker reports 0).
        if metrale_model_layers::ssm_reserve::mtp_pool_cap() {
            tracing::info!(
                "METRALE_MTP_POOL_CAP: MTP steps lost to uncovered slots = {}",
                metrale_model_layers::ssm_reserve::mtp_uncovered_steps()
            );
        }
        // 2026-09-25: The snapshot pool's spill staging buffer is freed here
        // because the pool holds no `gpu` handle. It is allocated on first use,
        // so this is a no-op when the spill tier never ran.
        self.ssm_snapshots.free_staging(self.gpu.as_ref());
    }
}
