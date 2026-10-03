// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! 2026-09-25: Sequence allocation, SSM h-state dtype conversion and normalisation, and model metadata accessors.
//!
//! Owner: model-engine.
//! Invariants:
//! - Every sequence `alloc_sequence_dispatch` returns has an SSM slot whose state was zeroed
//!   and synchronised before it returned.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

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
    pub(super) fn vocab_size_dispatch(&self) -> usize {
        self.config.vocab_size
    }

    pub(super) fn high_speed_swap_dims_dispatch(&self) -> Option<metrale_storage::ModelDims> {
        // 2026-09-25: Dims are returned unconditionally; the swap is installed only when
        // `--high-speed-swap` supplied a config.
        //
        // The KV paging identity (`METRALE_KV_PAGING`) is the fingerprint the SSM tier uses,
        // derived with the KV convention (`derive_kv`, blob_bytes = 0). When it cannot be
        // derived the field is `None` with a warning, and a `METRALE_KV_PAGING=1` connect then
        // fails unless `METRALE_KV_PAGING_NS` is set.
        let model_fp = match crate::model::ssm_tier::ModelFingerprint::derive_kv(&self.config) {
            Ok(fp) => Some(fp.nonzero()),
            Err(e) => {
                tracing::warn!(
                    "KV paging fingerprint underivable ({e:#}); METRALE_KV_PAGING=1 \
                     will fail fast unless METRALE_KV_PAGING_NS is set"
                );
                None
            }
        };
        Some(metrale_storage::ModelDims {
            num_layers: self.config.num_hidden_layers as u32,
            max_blocks_per_layer: self.max_blocks_per_seq,
            num_q_heads: self.config.num_attention_heads as u16,
            num_kv_heads: self.config.num_key_value_heads as u16,
            head_dim: self.config.head_dim as u16,
            block_size: self.kv_cache.lock().block_size() as u16,
            model_fp,
        })
    }

    /// 2026-09-25: Storage dtype of this sequence's SSM h-state: true when FP16.
    ///
    /// Read from the sequence's first SSM layer state. Allocation and
    /// `ssm_h_to_f16_dispatch` set the flag on all of a sequence's SSM layers together,
    /// so the first is representative.
    pub(super) fn seq_ssm_h_is_f16(&self, seq: &SequenceState) -> bool {
        seq.layer_states
            .iter()
            .find_map(|ls| ls.as_any().downcast_ref::<SsmLayerState>())
            .is_some_and(|s| s.h_is_f16)
    }

    /// 2026-09-25: Narrow this sequence's SSM h-state to FP16 (`METRALE_SSM_H_FP16`), once.
    ///
    /// Must be called from the model's decode entry points, outside the CUDA graph region:
    /// inside a captured graph the conversion would re-run on every replay over state that
    /// is already FP16, whatever the host-side `h_is_f16` flag says.
    ///
    /// A no-op when FP16 h-state is off or the sequence is already converted.
    /// It runs on `default_stream()`, not a caller's stream: the scratch buffer is one
    /// shared allocation, so the conversions of different sequences must be serialised
    /// on one stream.
    pub(crate) fn ssm_h_to_f16_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        let stream = self.gpu.default_stream();
        if !metrale_model_layers::layers::qwen3_ssm::ssm_h_fp16_enabled()
            || self.ssm_pool.num_ssm_layers == 0
        {
            return Ok(());
        }
        let h_bytes = self.ssm_pool.h_bytes;
        let f16_bytes = h_bytes / 2;
        let mut pending = false;
        for ls in seq.layer_states.iter() {
            if let Some(s) = ls.as_any().downcast_ref::<SsmLayerState>()
                && !s.h_is_f16
            {
                pending = true;
                break;
            }
        }
        if !pending {
            return Ok(());
        }
        if self.ssm_h_f32_to_f16_kernel.0 == 0 {
            bail!(
                "METRALE_SSM_H_FP16: ssm_h_dtype::ssm_h_state_f32_to_f16 did not resolve on this                  target — refusing to run the FP16 decode kernels over an FP32 pool"
            );
        }
        let scratch = match self.ssm_h_f16_scratch.get() {
            Some(p) => *p,
            None => {
                let p = self.gpu.alloc(f16_bytes)?;
                let _ = self.ssm_h_f16_scratch.set(p);
                p
            }
        };
        for ls in seq.layer_states.iter_mut() {
            let Some(s) = ls.as_any_mut().downcast_mut::<SsmLayerState>() else {
                continue;
            };
            if s.h_is_f16 {
                continue;
            }
            metrale_model_layers::layers::ops::ssm_h_state_f32_to_f16(
                self.gpu.as_ref(),
                self.ssm_h_f32_to_f16_kernel,
                s.h_state,
                scratch,
                (h_bytes / 4) as u64,
                stream,
            )?;
            self.gpu
                .copy_d2d_async(scratch, s.h_state, f16_bytes, stream)?;
            s.h_is_f16 = true;
        }
        if std::env::var("METRALE_SSM_H_FP16_DEBUG").is_ok() {
            tracing::info!(
                "SSM_H_FP16_CONVERT slot={} seq_len={} prompt_len={} hptr={:#x}",
                seq.slot_idx,
                seq.seq_len,
                seq.prompt_len,
                seq.layer_states
                    .iter()
                    .find_map(|l| l.as_any().downcast_ref::<SsmLayerState>())
                    .map(|x| x.h_state.0)
                    .unwrap_or(0)
            );
        }
        Ok(())
    }

    pub(super) fn normalize_ssm_states_dispatch(
        &self,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        use metrale_gpu_runtime::kernel_args::KernelLaunch;

        let num_ssm = self.ssm_pool.num_ssm_layers;
        if num_ssm == 0 || self.ssm_state_norm_kernel.0 == 0 {
            return Ok(());
        }
        // 2026-09-25: The kernel follows the slot's storage dtype (`seq_ssm_h_is_f16`): FP16
        // under the f16-sized pool or after the decode conversion, FP32 otherwise.
        let norm_k = if self.seq_ssm_h_is_f16(seq) {
            if self.ssm_state_norm_f16_kernel.0 == 0 {
                anyhow::bail!(
                    "METRALE_SSM_H_FP16: ssm_state_norm::ssm_state_clamp_norm_fused_f16 did not                      resolve, refusing to clamp an FP16 state through the FP32 kernel"
                );
            }
            self.ssm_state_norm_f16_kernel
        } else {
            self.ssm_state_norm_kernel
        };
        let slot = seq.slot_idx;

        let ptrs: Vec<u64> = (0..num_ssm)
            .map(|i| self.ssm_pool.h_state(i, slot).0)
            .collect();
        // 2026-09-25: SAFETY: the length is derived from `ptrs` itself,
        // `ptrs.len() * size_of::<u64>()`, over the `Vec<u64>` the `collect`
        // above just materialised (`len == num_ssm`, every element written by
        // the map, no `with_capacity` gap).
        let ptr_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(ptrs.as_ptr() as *const u8, ptrs.len() * 8) };
        self.gpu
            .copy_h2d_async(ptr_bytes, self.ssm_norm_ptrs_buf, stream)?;

        let (num_heads, k_dim, v_dim) = self.config.ssm_state_norm_dims();

        KernelLaunch::new(self.gpu.as_ref(), norm_k)
            .grid([num_heads as u32, num_ssm as u32, 1])
            .block([v_dim as u32, 1, 1])
            .arg_ptr(self.ssm_norm_ptrs_buf)
            .arg_u32(num_heads as u32)
            .arg_u32(k_dim as u32)
            .arg_u32(v_dim as u32)
            .launch(stream)?;

        Ok(())
    }

    pub(super) fn bind_gpu_to_thread_dispatch(&self) -> Result<()> {
        self.gpu.bind_to_thread()
    }

    pub(super) fn alloc_sequence_dispatch(&self, budget_tokens: usize) -> Result<SequenceState> {
        // 2026-09-25: `METRALE_SEQ_MEMTRACE`: the opening half of this sequence's memory bracket.
        crate::model::seq_memtrace::trace(self.gpu.as_ref(), "alloc");
        // 2026-09-25: Claimed through the RAII guard, which returns the slot to the pool when
        // it drops; the explicit `free_sequence` / `compact_sequence` paths take the index
        // out of the guard first, so the slot is released once. `slot_idx` comes from the guard.
        let slot_guard = self.ssm_pool.claim_guarded()?;
        let slot = slot_guard
            .idx()
            .expect("claim_guarded returns a guard owning a slot");
        // 2026-09-25: Zero the slot's h and conv state on the default stream and wait for it,
        // so no prefill kernel reads a previous sequence's state.
        let stream = self.gpu.default_stream();
        self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream)?;
        self.gpu.synchronize(stream)?;
        let has_mtp = self.proposer.is_some() || self.self_speculative;

        // 2026-09-25: A fresh sequence resets the whole-prompt capture length, so the capture
        // cannot cover its prompt until its own chunk 0 is captured.
        self.mtp_prefill_capture_len
            .store(0, std::sync::atomic::Ordering::Relaxed);
        // 2026-09-25: This sequence's ownership ticket for the shared hidden-row interval,
        // which is reset to empty. See `mtp_carry::StoreRange`.
        let store_gen = self.mtp_store_gen_seq.fetch_add(1, Relaxed) + 1;
        *self.mtp_store_range.lock() = metrale_model_layers::mtp_carry::StoreRange::EMPTY;

        // 2026-09-25: Linear-attention layers that use the SSM pool get the slot's pool
        // addresses, which are fixed; every other layer uses its own `alloc_state`. With MTP,
        // the checkpoint and intermediate pointers are pool addresses too.
        let mut ssm_layer_idx = 0usize;
        let mut layer_states: Vec<Box<dyn LayerState>> = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention && layer.uses_ssm_pool() {
                // 2026-09-25: The FP32 staging blob is per slot, shared by all layers.
                let stage = self.ssm_pool.h_prefill_stage(slot);
                let mut ssm_state = SsmLayerState {
                    h_state: self.ssm_pool.h_state(ssm_layer_idx, slot),
                    conv_state: self.ssm_pool.conv_state(ssm_layer_idx, slot),
                    h_state_checkpoint: None,
                    conv_state_checkpoint: None,
                    h_state_intermediates: Vec::new(),
                    conv_state_intermediates: Vec::new(),
                    // 2026-09-25: The slot was just zeroed, and zero is zero in both formats.
                    // The pool width decides the format: an f16-sized pool (a staging blob
                    // exists) holds FP16 from here on, and prefill stages its FP32 work in the
                    // blob; an FP32-sized pool holds FP32 until the decode conversion.
                    h_is_f16: stage.is_some(),
                    h_prefill_stage: stage,
                    ple: None,
                };

                if has_mtp {
                    // 2026-09-25: Pool addresses are fixed per slot, so a CUDA graph replays
                    // without stale pointers.
                    ssm_state.h_state_checkpoint =
                        Some(self.ssm_pool.h_checkpoint(ssm_layer_idx, slot));
                    ssm_state.conv_state_checkpoint =
                        Some(self.ssm_pool.conv_checkpoint(ssm_layer_idx, slot));

                    // 2026-09-25: Tiered pools: the h intermediate count is per slot
                    // (`h_inter_count`); the conv count is the same for every slot.
                    for t in 0..self.ssm_pool.h_inter_count(slot) {
                        ssm_state
                            .h_state_intermediates
                            .push(self.ssm_pool.h_intermediate(ssm_layer_idx, slot, t));
                    }
                    for t in 0..self.ssm_pool.num_intermediates {
                        ssm_state
                            .conv_state_intermediates
                            .push(self.ssm_pool.conv_intermediate(ssm_layer_idx, slot, t));
                    }
                }

                layer_states.push(Box::new(ssm_state));
                ssm_layer_idx += 1;
            } else {
                layer_states.push(layer.alloc_state(self.gpu.as_ref())?);
            }
        }

        // 2026-09-25: Zero the slot again, with its MTP checkpoints and intermediates when MTP
        // covers the slot, then synchronise.
        self.ssm_pool.reset_slot(slot, self.gpu.as_ref())?;
        self.gpu.synchronize(self.gpu.default_stream())?;

        // 2026-09-25: MTP proposer state, sized to this request's reach, not --max-seq-len.
        let proposer_state = match &self.proposer {
            Some(p) => Some(p.alloc_state_for(self.gpu.as_ref(), budget_tokens)?),
            None => None,
        };

        // 2026-09-25: `disk_last_offloaded_per_layer` is sized once to the attention-layer
        // count, so the offload helper never grows it.
        let num_attn_layers = self.config.num_attention_layers();
        Ok(SequenceState {
            adapter_id: 0,
            adapter_slot: -1,
            acquired_adapter_slot: -1,
            src_lang_id: 0,
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
            tokens: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            layer_states,
            proposer_state,
            slot_idx: slot,
            ssm_slot: Some(slot_guard),
            marconi_skip_to: 0,
            marconi_exact_snap: None,
            session_hash: 0,
            mtp_capture_gen: 0,
            mtp_store_gen: store_gen,
            chunked_prefill_meta: None,
            cached_prefix_tokens: 0,
            reused_prefix_tokens: 0,
            cached_prefix_blocks: 0,
            prefix_ref_tokens: Vec::new(),
            prefix_lookup_applied: false,
            tail_checkpoint_tokens: None,
            prefix_grid_refs: false,
            prefix_lookup_skip: false,
            kv_valid_tokens: 0,
            last_decode_ckpt_block: 0,
            prompt_len: 0,
            collect_prompt_logprobs: None,
            prompt_logprobs: Vec::new(),
            disk_block_ids: Vec::new(),
            disk_last_offloaded_per_layer: vec![0; num_attn_layers],
        })
    }

    pub(super) fn copy_logits_to_host_dispatch(
        &self,
        logits_ptr: DevicePtr,
        dst: &mut [u8],
    ) -> Result<()> {
        self.gpu.copy_d2h(logits_ptr, dst)
    }

    pub(super) fn logits_ptr_is_fp32_dispatch(&self, logits_ptr: DevicePtr) -> bool {
        self.use_fp32_logits && logits_ptr.0 == self.logits_fp32_buf.0
    }

    pub(super) fn logits_buffer_ptr_dispatch(&self) -> DevicePtr {
        self.buffers.logits()
    }

    pub(super) fn hidden_after_norm_dispatch(&self) -> DevicePtr {
        // 2026-09-25: The final norm writes its output to `norm_output()`.
        self.buffers.norm_output()
    }
}
