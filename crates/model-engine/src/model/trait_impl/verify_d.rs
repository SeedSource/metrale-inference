// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: K=γ graphed speculative verify of one sequence for DFlash
//! (`decode_verify_graphed_kgamma`): the K rows go through every layer in one pass,
//! captured as a CUDA graph per (SSM slot, K) when graphs are allowed. The `unsafe` blocks
//! view local integer `Vec`s as bytes for `copy_h2d_async`, which lets its source drop once
//! it returns.
//!
//! Owner: model-engine.
//! Invariants:
//! - 2026-10-01: A cached graph is launched only after every layer's `check_replay_room`
//!   passed, and a successful launch is followed by every layer's `sync_replayed_step`.

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
    pub(super) fn decode_verify_graphed_kgamma_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;

        // 2026-09-25: The verify updates the SSM `h_state` in place, and after a partial
        // accept the commit (`commit_accepted_prefix`) restores the state after the last
        // accepted row. No state is copied before the verify.

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // 2026-09-25: Everything up to the graph varies per step and is not captured.
        for t in 0..k {
            self.embed(tokens[t], hidden.offset(t * h * fp32), stream)?;
        }

        let bs = kv_cache.block_size();
        for t in 0..k {
            let pos = seq.seq_len + t;
            let blocks_needed = (pos / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )?;
        }

        // 2026-09-25: Metadata at scratch + 32768: positions @0, seq_slot @128, slots @256,
        // seq_lens @512, block table @768 (K rows of `max_blocks_per_seq`).
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        let positions: Vec<u32> = (0..k).map(|t| (seq.seq_len + t) as u32).collect();
        // 2026-09-25: SAFETY: `positions` is built one line above by `(0..k).map(..)
        // .collect()`, so `positions.len() == k` exactly (collect on a
        // `Range` yields one element per step) and `k * 4 == size_of_val(&
        // positions[..])`. Every element is written by the collect, so no
        // uninitialised spare capacity is read. `u32` is POD.
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, k * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        let mut slots = vec![0i64; k];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        // 2026-09-25: SAFETY: `slots` is `vec![0i64; k]`, so its len (not merely its
        // capacity) is `k` and every element is zero-initialised before the
        // `for t in 0..k` loop overwrites it; `k * 8 == size_of_val(&
        // slots[..])`, with no read past `len` into spare capacity.
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, k * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        let seq_lens: Vec<i32> = (0..k).map(|t| (seq.seq_len + t + 1) as i32).collect();
        // 2026-09-25: SAFETY: `seq_lens` is `(0..k).map(..).collect()` on the line above,
        // so `seq_lens.len() == k` and `k * 4 == size_of_val(&seq_lens[..])`;
        // all `k` elements are initialised by the collect. `i32` is POD.
        let sl_bytes = unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, k * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        let mb = max_blocks as usize;
        let needed = k * mb;
        let mut bt_buf = vec![0i32; needed];
        for row in 0..k {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[row * mb + j] = block as i32;
            }
        }
        // 2026-09-25: SAFETY: `bt_buf` is `vec![0i32; needed]`, so its len is `needed` and
        // `needed * 4 == size_of_val(&bt_buf[..])`; the read stops at `len`, never in the
        // `Vec`'s spare capacity. The zero-init at construction covers the tail the
        // `for row in 0..k` fill leaves untouched when `block_table.len() < mb`.
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

        // 2026-09-25: Request-scoped LoRA routing: one adapter for all K rows, as a [K]
        // buffer at +128 uploaded before capture. K must stay <= 32, or the buffer would
        // overrun the slots at +256. `DevicePtr(0)` selects the installed-pair path.
        debug_assert!(k <= 32, "γ verify seq_slot +128 gap holds K ≤ 32");
        let seq_slot =
            self.upload_seq_slot_uniform(seq.adapter_slot, k, meta_base.offset(128), stream)?;

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: k as u32,
            seq_slot,
            moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        };

        // 2026-09-25: The HSS path does host I/O, which is illegal under capture.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // 2026-09-25: `METRALE_DFLASH_DEBUG_NO_GRAPH=1` runs the verify without a graph, so
        // `CUDA_LAUNCH_BLOCKING=1` can name the failing kernel.
        let force_eager = std::env::var("METRALE_DFLASH_DEBUG_NO_GRAPH")
            .ok()
            .as_deref()
            == Some("1");
        // 2026-09-25: The `lora_eager` lever runs LoRA verifies without graphs.
        let lora_eager = self.lora.is_some() && self.levers.lora_eager;
        // 2026-10-01: Under a communicator the K=γ verify is captured only with
        // `METRALE_GLM_DFLASH=1` and unless `METRALE_GLM_VERIFY_GRAPHS=0`, the switch the
        // GLM K=3/K=4 verify graphs use. Every rank runs this verify (`EP_CMD_VERIFY_KGAMMA`),
        // so both capture and replay the same per-(slot, K) graph and its collectives pair up.
        // Read once per process, so a captured graph cannot outlive a changed value.
        static EP_GRAPHS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let ep_graphs = *EP_GRAPHS.get_or_init(|| {
            metrale_model_layers::speculative::glm_dflash::glm_dflash_enabled()
                && std::env::var("METRALE_GLM_VERIFY_GRAPHS").ok().as_deref() != Some("0")
        });
        let use_graphs = (self.comm.is_none() || ep_graphs)
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !force_eager
            && !lora_eager
            // 2026-09-25: Sliding-window layers use `verify_attention_per_token`, whose
            // per-token H2D uploads are illegal under capture.
            && !(0..self.layers.len())
                .any(|i| self.config.layer_type(i) == LayerType::SlidingAttention);

        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };

        let mut graph_cache = if use_graphs {
            Some(self.verify_kgamma_graph.lock())
        } else {
            None
        };

        let cache_key = (seq.slot_idx, k);
        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&cache_key).copied());
        if let Some(graph) = cached_for_slot
            && graph.0 != 0
        {
            // 2026-10-01: As in the K=3/K=4 verify: refuse a replay whose GLM-5.3 DSA indexer
            // rows would run past the buffer before it launches, and after it reconcile the
            // host-side indexer length to `seq_len + k` (a replay runs only kernels, and the
            // previous verify kept only its accepted prefix).
            for (i, layer) in self.layers.iter().enumerate() {
                layer.check_replay_room(&*seq.layer_states[i], seq.seq_len, k)?;
            }
            self.gpu.launch_graph(graph, stream)?;
            for (i, layer) in self.layers.iter().enumerate() {
                layer.sync_replayed_step(seq.layer_states[i].as_mut(), seq.seq_len, k)?;
            }
        }
        let need_run = cached_for_slot.is_none();
        if need_run {
            let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);

                if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        // 2026-09-25: Under HSS, `decode_multi_seq` reads only the KV in HBM;
                        // `decode_batched` runs single-token decodes, which also read the disk
                        // tier.
                        layer.decode_batched(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )?;
                    } else {
                        let mut dummy_states: Vec<Box<dyn LayerState>> = (0..k)
                            .map(|_| layer.alloc_state(self.gpu.as_ref()))
                            .collect::<Result<_>>()?;
                        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                            dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                        layer.decode_multi_seq(
                            hidden,
                            residual,
                            k,
                            &mut refs,
                            &mut kv_cache,
                            &seq_lens_vec,
                            &block_tables_vec,
                            &ctx,
                            stream,
                        )?;
                    }
                } else if layer_type == LayerType::SlidingAttention {
                    // 2026-09-25: The default `decode_batched` would decode every row at the
                    // same position; graphs are off for these models (above).
                    self.verify_attention_per_token(
                        layer.as_ref(),
                        layer_idx,
                        hidden,
                        residual,
                        k,
                        seq,
                        &mut kv_cache,
                        stream,
                    )?;
                } else {
                    layer.decode_batched(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[layer_idx].as_mut(),
                        &mut kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                }
                // 2026-09-25: DFlash: capture this layer's output while `hidden_states` still
                // holds it, inside the captured region. By default every verify row is
                // captured (`try_dflash_capture_all`), because the scheduler's `commit_ctx`
                // copies rows 0..=num_accepted; `METRALE_DFLASH_EAGLE_FIX=0` or
                // `METRALE_DFLASH_UNIFIED_CTX=0` captures only the last row. The env is read
                // once, so a captured graph cannot bake a changed value.
                static CAPTURE_ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let capture_all = *CAPTURE_ALL.get_or_init(|| {
                    std::env::var("METRALE_DFLASH_EAGLE_FIX").ok().as_deref() != Some("0")
                        && std::env::var("METRALE_DFLASH_UNIFIED_CTX").ok().as_deref() != Some("0")
                });
                if capture_all {
                    self.try_dflash_capture_all(layer_idx, k, stream)?;
                } else {
                    self.try_dflash_capture(layer_idx, k - 1, stream)?;
                }
            }

            let normed = self.buffers.norm_output();
            self.final_norm_apply(
                hidden,
                normed,
                k as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            self.lm_head_batched(normed, k as u32, self.buffers.logits(), stream)?;

            // 2026-09-25: The argmax is part of the graph; it writes fixed scratch addresses.
            let vocab = self.config.vocab_size;
            let argmax_out = self.buffers.scratch();
            for t in 0..k {
                let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                let out_t = argmax_out.offset(t * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_t,
                    out_t,
                    vocab as u32,
                    stream,
                )?;
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "Captured CUDA graph for K=γ verify (slot={} K={})",
                        seq.slot_idx,
                        k
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(cache_key, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        let out_ptr = self.buffers.scratch();
        let mut buf = vec![0u8; k * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut out = Vec::with_capacity(k);
        for t in 0..k {
            let off = t * 4;
            out.push(u32::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
            ]));
        }

        // 2026-09-25: All K tokens are appended and `seq_len` advances by K, as in the K=2
        // verify.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok(out)
    }
}
