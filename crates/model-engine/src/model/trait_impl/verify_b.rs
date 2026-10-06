// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: K=2 graphed speculative verify of one sequence (`decode_verify_graphed`): both
//! rows go through every layer in one pass, captured as a CUDA graph per SSM slot when
//! graphs are allowed.
//!
//! Owner: model-engine.
//! Invariants:
//! - A cached graph is launched only after every layer's `check_replay_room` passed, and a
//!   successful launch is followed by every layer's `sync_replayed_step`.

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
    pub(super) fn decode_verify_graphed_dispatch(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 2]> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;
        let k = 2usize;

        // 2026-09-25: The verify updates the SSM `h_state` in place, and after a partial
        // accept the commit (`commit_accepted_prefix`) restores the state after the last
        // accepted row. No state is copied before the verify.

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // 2026-09-25: Everything up to the graph varies per step and is not captured.
        self.embed(tokens[0], hidden, stream)?;
        self.embed(tokens[1], hidden.offset(h * fp32), stream)?;

        let bs = kv_cache.block_size();
        // 2026-10-05: Rank-agreed lazy-map admission (A168), before the step's first collective
        // and any capture or replay (`decode_lazy_agree.rs`).
        self.agree_decode_lazy_maps(seq, (seq.seq_len + k).saturating_sub(1) / bs, bs)?;
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
        // seq_lens @512, block table @768.
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        let positions = [seq.seq_len as u32, (seq.seq_len + 1) as u32];
        // 2026-09-25: SAFETY: `positions` is the `[u32; 2]` literal on the line above, a
        // fully-initialised local whose size is 2 * 4 = 8, exactly the byte
        // length requested. `u32` is POD (no padding, every bit pattern
        // valid), so the `&[u8]` view is sound, and `positions` is live for
        // the whole function so the borrow outlives the H2D enqueue below.
        let pos_bytes = unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, 8) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        let mut slots = [0i64; 2];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            // 2026-09-25: Fall back to the dummy block, not block 0: physical block 0 can be a
            // live block, possibly shared through the prefix cache.
            let physical_block = seq.physical_block_for(block_idx).unwrap_or_else(|| {
                tracing::error!(
                    "verify_k2: no physical block for pos {pos} (block_table len {}); \
                     writing KV to dummy block",
                    seq.block_table.len(),
                );
                self.dummy_kv_block
            });
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        // 2026-09-25: SAFETY: `slots` is the `[0i64; 2]` declared above, zero at
        // declaration and then overwritten by the `for t in 0..k` loop (`k == 2`). Its size
        // is 2 * 8 = 16, exactly the byte length requested; `i64` is POD.
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, 16) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        let seq_lens = [(seq.seq_len + 1) as i32, (seq.seq_len + 2) as i32];
        // 2026-09-25: SAFETY: `seq_lens` is the `[i32; 2]` literal on the line above, so
        // its size is 2 * 4 = 8, exactly the byte length requested. `i32` is
        // POD and the local outlives this borrow.
        let sl_bytes = unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, 8) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        // 2026-09-25: The block table repeats the sequence's table once per row.
        let mb = max_blocks as usize;
        let needed = k * mb;
        let mut bt_buf_vec;
        let mut bt_buf_stack = [0i32; 1024];
        let bt_buf: &mut [i32] = if needed <= 1024 {
            &mut bt_buf_stack[..needed]
        } else {
            bt_buf_vec = vec![0i32; needed];
            &mut bt_buf_vec
        };
        for row in 0..k {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[row * mb + j] = block as i32;
            }
        }
        // 2026-09-25: SAFETY: `bt_buf.len() == needed` in both arms of the `if needed <=
        // 1024` above: the stack arm slices `bt_buf_stack[..needed]` and the
        // heap arm is `vec![0i32; needed]`, whose len (not just capacity) is
        // `needed`. So `needed * 4 == size_of_val(bt_buf)`, never past the
        // end and never over a `Vec`'s uninitialised spare capacity. Both
        // arms are zero-initialised before the `for row in 0..k` fill, so
        // every byte read is initialised even when `block_table.len() < mb`.
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

        // 2026-09-25: Request-scoped LoRA routing: one adapter for all K rows, as a [K]
        // buffer at +128, which fits below the slots at +256 for K <= 32. It is uploaded
        // before capture, so a graph reads a fixed address. `DevicePtr(0)` selects the
        // installed-pair path.
        debug_assert!(k <= 32, "verify seq_slot +128 gap holds K ≤ 32");
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

        // 2026-09-25: `suppress_graphs` keeps FP8 KV calibration eager, because its host
        // syncs are illegal under capture. Once calibration is frozen this lifts the
        // suppression, as `decode_dispatch` does, and only when calibration is configured
        // (`fp8_kv_calibration_tokens > 0`), so a suppression for another reason stays.
        if self.config.fp8_kv_calibration_tokens > 0
            && self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && self.fp8_calibration_frozen()
        {
            self.suppress_graphs
                .store(false, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("FP8 calibration frozen — re-enabling CUDA graphs (MTP verify)");
        }
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // 2026-09-25: `METRALE_K2_DIAG=1` arms synchronize checkpoints in `forward_k2`, which
        // are illegal under capture, so the verify then runs eagerly.
        let k2_diag_eager = std::env::var("METRALE_K2_DIAG").ok().as_deref() == Some("1");
        // 2026-09-25: The `lora_eager` lever runs LoRA verifies without graphs.
        let lora_eager = self.lora.is_some() && self.levers.lora_eager;
        // 2026-09-25: With a communicator (EP), the verify is captured unless
        // `METRALE_GLM_VERIFY_GRAPHS=0`. It does not read `METRALE_EP_GRAPHS` (the
        // `ep_graphs` lever of `decode_a`'s decode graph), so the two are gated apart.
        // `free_sequence_dispatch` drops this slot-keyed cache for layers that own
        // per-sequence state.
        let ep_graphs = std::env::var("METRALE_GLM_VERIFY_GRAPHS").ok().as_deref() != Some("0");
        // 2026-09-25: `METRALE_GLM_VERIFY_GRAPH_TRACE=1` captures every step, never replays,
        // and logs the enqueued ops that differ from the previous step's (host values a
        // replay would freeze). It turns graphs on and caches none.
        let graph_trace = std::env::var("METRALE_GLM_VERIFY_GRAPH_TRACE").is_ok_and(|v| v == "1");
        let ep_graphs = ep_graphs || graph_trace;
        let use_graphs = (self.comm.is_none() || ep_graphs)
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            // 2026-09-25: The HSS path does host I/O, which is illegal under capture.
            && !hss_engaged
            && !k2_diag_eager
            && !lora_eager
            // 2026-09-25: Sliding-window layers use `verify_attention_per_token`, whose
            // per-token H2D uploads are illegal under capture.
            && !(0..self.layers.len())
                .any(|i| self.config.layer_type(i) == LayerType::SlidingAttention);

        // 2026-09-25: Hash-routed MoE layers pick experts by token id (`tid2eid`), so the
        // verify tokens go to the fixed `token_ids` buffer before capture, as in `decode`.
        let tid_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        self.gpu
            .copy_h2d_async(&tid_bytes, self.buffers.token_ids(), stream)?;

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
            token_ids: Some(self.buffers.token_ids()),
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };

        let mut graph_cache = if use_graphs {
            Some(self.verify2_graph.lock())
        } else {
            None
        };

        // 2026-09-25: Replay only when this sequence's SSM slot has a captured graph.
        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&seq.slot_idx).copied());
        if let Some(graph) = cached_for_slot
            && graph.0 != 0
        {
            // 2026-09-25: Before the replay: the graph writes the GLM-5.3 DSA indexer row at a
            // device position, so a step past the buffer must be refused before it runs; the
            // reconcile after the launch is too late (see `check_replay_room`).
            for (i, layer) in self.layers.iter().enumerate() {
                layer.check_replay_room(&*seq.layer_states[i], seq.seq_len, k)?;
            }
            self.gpu.launch_graph(graph, stream)?;
            // 2026-09-25: A replay runs only kernels, so host-side per-sequence bookkeeping
            // (the GLM-5.3 DSA indexer length) is reconciled here to `seq_len + k`, not
            // advanced by k, because the previous verify kept only its accepted prefix
            // (see `sync_replayed_step`).
            for (i, layer) in self.layers.iter().enumerate() {
                layer.sync_replayed_step(seq.layer_states[i].as_mut(), seq.seq_len, k)?;
            }
        }
        let need_run = cached_for_slot.is_none();
        if need_run {
            let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

            if graph_trace {
                metrale_telemetry::launch_trace::begin();
            }
            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);

                if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        // 2026-09-25: Under HSS, HBM holds only `cache_blocks_per_seq` blocks and
                        // older KV is on disk. `decode_multi_seq` reads HBM only; the
                        // single-token `decode` also reads the disk tier, so `decode_batched`
                        // runs the rows one at a time.
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
                        // 2026-09-25: The K rows run as K sequences of one token through
                        // `decode_multi_seq`, each with a throwaway attention layer state.
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
                // 2026-09-25: DFlash: capture this layer's hidden at the last row (K-1) into
                // `dflash_hidden_save` for the next propose. A no-op without DFlash.
                self.try_dflash_capture(layer_idx, k - 1, stream)?;
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
                    tracing::info!("Captured CUDA graph for K=2 verify (slot={})", seq.slot_idx);
                    // 2026-09-25: `METRALE_GLM_VERIFY_GRAPH_NOCACHE=1` captures and runs every
                    // step without caching, which separates a wrong capture from a stale replay.
                    if let Some(ref mut cache) = graph_cache
                        && !graph_trace
                        && !std::env::var("METRALE_GLM_VERIFY_GRAPH_NOCACHE")
                            .is_ok_and(|v| v == "1")
                    {
                        cache.insert(seq.slot_idx, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
            if graph_trace && let Some(report) = metrale_telemetry::launch_trace::end_and_diff(40) {
                tracing::info!("A56 trace K=2 (this step vs previous): {}", report);
            }
        }

        let out_ptr = self.buffers.scratch();
        let mut buf = [0u8; 8];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let tok0 = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let tok1 = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);

        // 2026-09-25: All K tokens, including `tokens[0]`, are appended and `seq_len` advances
        // by K: the scheduler passes `[last_token, draft]`, and `last_token` is not yet in
        // `seq.tokens`.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok([tok0, tok1])
    }
}
