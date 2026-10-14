// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Swap-file save and restore of a sequence's state. The record is, for each
//! block of `block_table` in order and each KV layer, the K bytes then the V bytes; then, for
//! each linear-attention layer, the SSM h state at the pool's storage width and the conv
//! state.
//!
//! Owner: model-engine.
//! Invariants: save and restore write and read the record in the same order.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use crate::model::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use crate::model::ssm_pool::SsmStatePool;
use crate::model::ssm_snapshot::SsmSnapshotPool;
use crate::model::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(crate) fn save_sequence_state_dispatch(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();

        // 2026-09-25: The KV lock is held only for the device reads, not for the writes.
        let kv_buffers = {
            let kv = self.kv_cache.lock();
            let mut bufs = Vec::with_capacity(seq.block_table.len() * kv.num_layers());
            for &block_idx in &seq.block_table {
                for layer_idx in 0..kv.num_layers() {
                    bufs.push(kv.read_block(layer_idx, block_idx, gpu)?);
                }
            }
            bufs
        };

        for (k_data, v_data) in &kv_buffers {
            writer.write_all(k_data)?;
            writer.write_all(v_data)?;
        }

        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                // 2026-09-25: `h_stored_bytes`: the record holds the h blob at the pool's
                // storage width, which is f16 under the f16-sized pool and FP32 otherwise.
                let mut h_buf = vec![0u8; self.ssm_pool.h_stored_bytes];
                let mut c_buf = vec![0u8; self.ssm_pool.conv_bytes];
                gpu.copy_d2h(ssm.h_state, &mut h_buf)?;
                gpu.copy_d2h(ssm.conv_state, &mut c_buf)?;
                writer.write_all(&h_buf)?;
                writer.write_all(&c_buf)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    pub(crate) fn restore_sequence_state_dispatch(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();

        // 2026-09-25: Read the whole KV part before taking the KV lock.
        let (num_layers, layer_strides) = {
            let kv = self.kv_cache.lock();
            let n = kv.num_layers();
            let strides: Vec<usize> = (0..n).map(|i| kv.block_stride_bytes_for_layer(i)).collect();
            (n, strides)
        };

        let mut kv_buffers = Vec::with_capacity(num_blocks * num_layers);
        for _ in 0..num_blocks {
            for layer_idx in 0..num_layers {
                let stride = layer_strides[layer_idx];
                let mut k_data = vec![0u8; stride];
                let mut v_data = vec![0u8; stride];
                reader.read_exact(&mut k_data)?;
                reader.read_exact(&mut v_data)?;
                kv_buffers.push((k_data, v_data));
            }
        }

        {
            let mut kv = self.kv_cache.lock();
            let mut new_block_table = Vec::with_capacity(num_blocks);
            let mut buf_idx = 0;
            for _ in 0..num_blocks {
                let block_idx = kv.alloc_block()?;
                for layer_idx in 0..num_layers {
                    let (ref k_data, ref v_data) = kv_buffers[buf_idx];
                    kv.write_block(layer_idx, block_idx, k_data, v_data, gpu)?;
                    buf_idx += 1;
                }
                new_block_table.push(block_idx);
            }
            seq.block_table = new_block_table;
            // 2026-10-14 (race-memory, A183): a rebuilt table votes afresh.
            seq.kv_blocks_agreed = 0;
        }

        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                // 2026-09-25: `h_stored_bytes`: the record holds the h blob at the pool's
                // storage width, which is f16 under the f16-sized pool and FP32 otherwise.
                let mut h_buf = vec![0u8; self.ssm_pool.h_stored_bytes];
                let mut c_buf = vec![0u8; self.ssm_pool.conv_bytes];
                reader.read_exact(&mut h_buf)?;
                reader.read_exact(&mut c_buf)?;
                gpu.copy_h2d(&h_buf, ssm.h_state)?;
                gpu.copy_h2d(&c_buf, ssm.conv_state)?;
            }
        }

        Ok(())
    }
}
