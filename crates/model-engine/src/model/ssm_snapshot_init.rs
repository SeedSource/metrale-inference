// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Construction of the SSM snapshot pool (`SsmSnapshotPool::new`).
//!
//! Owner: model-engine SSM snapshot pool.
//! Invariants:
//! - With no SSM layers both regions are empty and no device memory is allocated.
//! - `hidden_snapshot` is allocated exactly when the Marconi region is.

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

impl SsmSnapshotPool {
    /// 2026-09-25: Build the snapshot pool. `num_slots` sizes the Marconi region and
    /// `decode_ring_slots * decode_max_seqs` the decode-rollback ring; either may be
    /// zero independently of the other. The f16/FP32 conversion kernels are looked up
    /// with `try_kernel` and are `KernelHandle(0)` when not loaded, and also when
    /// both regions are empty.
    pub(super) fn new(
        num_slots: usize,
        h_bytes: usize,
        conv_bytes: usize,
        num_ssm_layers: usize,
        decode_ring_slots: usize,
        decode_max_seqs: usize,
        hidden_bytes: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let decode_enabled = num_ssm_layers > 0 && decode_ring_slots > 0 && decode_max_seqs > 0;
        let marconi_enabled = num_ssm_layers > 0 && num_slots > 0;

        if !marconi_enabled && !decode_enabled {
            return Ok(Self {
                h_snapshots: Vec::new(),
                conv_snapshots: Vec::new(),
                free_slots: Mutex::new(Vec::new()),
                num_slots: 0,
                h_bytes,
                conv_bytes,
                num_ssm_layers,
                session_tags: Mutex::new(std::collections::HashMap::new()),
                aux_blobs: Mutex::new(std::collections::HashMap::new()),
                aux_meta: Mutex::new(Default::default()),
                decode_h_snapshots: Vec::new(),
                decode_conv_snapshots: Vec::new(),
                decode_ring_slots: 0,
                decode_max_seqs: 0,
                hidden_snapshot: DevicePtr::NULL,
                hidden_bytes,
                slot_has_hidden: Mutex::new(std::collections::HashSet::new()),
                h_f16_to_f32_k: KernelHandle(0),
                h_f32_to_f16_k: KernelHandle(0),
                spill_staging: Default::default(),
            });
        }

        let mut h_snapshots = Vec::new();
        let mut conv_snapshots = Vec::new();
        let mut hidden_snapshot = DevicePtr::NULL;
        if marconi_enabled {
            for _ in 0..num_ssm_layers {
                h_snapshots.push(gpu.alloc(num_slots * h_bytes)?);
                conv_snapshots.push(gpu.alloc(num_slots * conv_bytes)?);
            }
            hidden_snapshot = gpu.alloc(num_slots * hidden_bytes)?;
        }

        let mut decode_h_snapshots = Vec::new();
        let mut decode_conv_snapshots = Vec::new();
        let decode_region = if decode_enabled {
            decode_max_seqs * decode_ring_slots
        } else {
            0
        };
        if decode_enabled {
            for _ in 0..num_ssm_layers {
                decode_h_snapshots.push(gpu.alloc(decode_region * h_bytes)?);
                decode_conv_snapshots.push(gpu.alloc(decode_region * conv_bytes)?);
            }
        }

        let free_slots: Vec<usize> = if marconi_enabled {
            (0..num_slots).rev().collect()
        } else {
            Vec::new()
        };
        let marconi_mb = num_ssm_layers * num_slots * (h_bytes + conv_bytes) / (1024 * 1024);
        let decode_mb = num_ssm_layers * decode_region * (h_bytes + conv_bytes) / (1024 * 1024);
        tracing::info!(
            "SSM snapshot pool: Marconi {num_slots} slots ({marconi_mb} MB), \
             decode-rollback {decode_ring_slots} slots × {decode_max_seqs} seqs \
             ({decode_mb} MB), {num_ssm_layers} layers",
        );

        Ok(Self {
            h_snapshots,
            conv_snapshots,
            free_slots: Mutex::new(free_slots),
            num_slots: if marconi_enabled { num_slots } else { 0 },
            h_bytes,
            conv_bytes,
            num_ssm_layers,
            session_tags: Mutex::new(std::collections::HashMap::new()),
            aux_blobs: Mutex::new(std::collections::HashMap::new()),
            aux_meta: Mutex::new(Default::default()),
            decode_h_snapshots,
            decode_conv_snapshots,
            decode_ring_slots: if decode_enabled { decode_ring_slots } else { 0 },
            decode_max_seqs: if decode_enabled { decode_max_seqs } else { 0 },
            hidden_snapshot,
            hidden_bytes,
            slot_has_hidden: Mutex::new(std::collections::HashSet::new()),
            h_f16_to_f32_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "ssm_h_dtype",
                "ssm_h_state_f16_to_f32",
            ),
            h_f32_to_f16_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "ssm_h_dtype",
                "ssm_h_state_f32_to_f16",
            ),
            spill_staging: Default::default(),
        })
    }
}
