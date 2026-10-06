// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The single-sequence decode step (`decode_dispatch_with`): host-side
//! staging, then a slot-keyed CUDA-graph replay, a capture, or an eager run.
//!
//! Owner: model-engine (decode).
//! Invariants:
//! - A replay calls `check_replay_room` on every layer before `launch_graph`.
//! - `seq.tokens` and `seq.seq_len` advance only for a host token, and only when
//!   the step returns `Ok`.
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

/// 2026-09-25: `METRALE_REDZONE_RANGE_FILE=<path>`: a file holding "LO HI", re-read
/// every decode step and passed to `poison_redzones`. Unset: no bisection.
fn redzone_range_file() -> Option<&'static str> {
    static P: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    P.get_or_init(|| std::env::var("METRALE_REDZONE_RANGE_FILE").ok())
        .as_deref()
}

/// 2026-09-25: Scan the guard bands every `n`-th decode step: 0 (never) when
/// `METRALE_REDZONE` is unset, else `METRALE_REDZONE_EVERY` (1 when unset or
/// unparsable; 0 disables the scan).
fn redzone_every() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        if std::env::var("METRALE_REDZONE").is_err() {
            return 0;
        }
        std::env::var("METRALE_REDZONE_EVERY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
    })
}

impl TransformerModel {
    /// 2026-09-25: Whether every layer that runs online FP8-KV calibration reports
    /// frozen (`graphs_ready_after_fp8_kv_cal`). True when no layer calibrates.
    /// All layers must agree: a BF16 layer that never observes reports `None`
    /// and does not hold the result back.
    pub(in crate::model) fn fp8_calibration_frozen(&self) -> bool {
        metrale_model_layers::layers::fp8_calibration::graphs_ready_after_fp8_kv_cal(
            self.layers.iter().map(|l| l.fp8_calibration_frozen()),
        )
    }

    pub(super) fn decode_dispatch(
        &self,
        token: u32,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.decode_dispatch_with(super::feed::DecodeInput::Host(token), seq, stream)
    }

    /// 2026-09-25: The single-sequence decode step. With a host token
    /// (`DecodeInput::Host`) it embeds that id and pushes it onto `seq.tokens`;
    /// with the device feed it gathers the id from `feed_ids` and leaves the
    /// sequence bookkeeping to the caller.
    pub(super) fn decode_dispatch_with(
        &self,
        input: super::feed::DecodeInput,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<DevicePtr> {
        // 2026-09-25: The backend's own stream (`default_stream`), which CUDA-graph
        // capture records.
        let stream = self.gpu.default_stream();
        // 2026-09-25: `--ssm-h-dtype f16`: narrow this sequence's SSM h-state to FP16
        // here, before any graph region. No-op without the flag.
        self.ssm_h_to_f16_dispatch(seq)?;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // 2026-09-25: With `levers.ssm_save_dump`, at the first decode step, checksum
        // the scratch buffers, this slot's SSM state and every block of KV-cache
        // layer 0 before any compute, to compare a prefix-cache hit against a
        // cold run.
        if seq.seq_len == seq.prompt_len && self.levers.ssm_save_dump {
            self.buffers
                .debug_buffer_checksum(self.gpu.as_ref(), stream, "decode_step0_pre");
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                "decode_step0_pre",
            );
            kv_cache.debug_kv_per_block(
                0,
                &seq.block_table,
                self.gpu.as_ref(),
                stream,
                "decode_step0_pre",
            );
        }

        // 2026-09-25: Red-zone bisection. The range is re-read and re-applied every
        // step, so it can be swept without a restart; `poison_redzones` allocates
        // and moves nothing, only the guard-band contents change.
        if let Some(path) = redzone_range_file() {
            let txt = std::fs::read_to_string(path).unwrap_or_default();
            let mut it = txt.split_whitespace();
            let lo: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            let hi: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            self.gpu.synchronize(stream)?;
            self.gpu.poison_redzones(lo, hi)?;
        }

        // 2026-09-25: MLA with absorbed attention (`kv_lora_rank > 0`,
        // `o_lora_rank == 0`): zero the scratch rows this step can read, so stale
        // prefill data does not reach the absorbed attention. Models with
        // `o_lora_rank > 0` (DeepSeek-V4-Flash) take the direct path, which writes
        // these buffers before reading them. A decode step reads only row 0 of each
        // token-major arena, so `zero_all_rows(.., 1)` is enough. Measured
        // 2026-08-28 on GLM-5.3 (nsys, `max_batch_tokens = 4096`): zeroing the whole
        // arena cost 8.01 ms of an 85 ms step.
        if self.config.kv_lora_rank > 0 && self.config.o_lora_rank == 0 {
            self.buffers.zero_all_rows(self.gpu.as_ref(), stream, 1)?;
        }

        // 2026-09-25: Embedding lookup. `seq.tokens` does not yet hold `token` (it
        // is pushed after the forward), which is the n-gram contract: preceding
        // context, then the token being embedded.
        match input.host() {
            Some(token) => self.embed_ctx(&seq.tokens, token, hidden, stream)?,
            None => self.feed_embed_rows(hidden, 1, stream)?,
        }

        let bs = kv_cache.block_size();
        let blocks_needed = (seq.seq_len / bs) + 1;
        // 2026-10-05: Rank-agreed lazy-map admission (A168), before the step's first collective
        // and any capture or replay (`decode_lazy_agree.rs`).
        self.agree_decode_lazy_maps(seq, blocks_needed - 1, bs)?;
        ensure_blocks_through_decode(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
            self.levers.kv_poison,
        )?;

        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = seq.block_table.len() as u32;

        let pos_val = seq.seq_len as u32;
        self.gpu
            .copy_h2d_async(&pos_val.to_le_bytes(), meta_base, stream)?;

        let block_idx = seq
            .physical_block_for(seq.seq_len / bs)
            .unwrap_or(self.dummy_kv_block);
        let global_slot = (block_idx as i64) * (bs as i64) + ((seq.seq_len % bs) as i64);
        self.gpu
            .copy_h2d_async(&global_slot.to_le_bytes(), meta_base.offset(8), stream)?;

        let actual_seq_len = (seq.seq_len + 1) as i32;
        self.gpu
            .copy_h2d_async(&actual_seq_len.to_le_bytes(), meta_base.offset(16), stream)?;

        let bt_i32: Vec<i32> = seq.block_table.iter().map(|&b| b as i32).collect();
        // 2026-09-25: SAFETY: the length is `bt_i32.len() * 4` bytes of the `Vec<i32>` above.
        let bt_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(bt_i32.as_ptr() as *const u8, bt_i32.len() * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(256), stream)?;

        // 2026-09-25: Upload this step's token id into the stable `token_ids` buffer
        // before any replay, for layers that read token ids on the device
        // (hash-MoE).
        match input.host() {
            Some(token) => {
                self.gpu
                    .copy_h2d_async(&token.to_le_bytes(), self.buffers.token_ids(), stream)?;

                // 2026-09-25: Per-step host work of each layer (`decode_prestage`),
                // before any replay or capture, so a replayed graph reads only
                // buffers refreshed here.
                for (li, l) in self.layers.iter().enumerate() {
                    l.decode_prestage(
                        token,
                        seq.layer_states[li].as_mut(),
                        self.gpu.as_ref(),
                        stream,
                    )?;
                }
            }
            // 2026-09-25: A fed row: the id is already on the device; copy it into
            // `token_ids`.
            None => {
                self.gpu
                    .copy_d2d_async(self.feed_ids, self.buffers.token_ids(), 4, stream)?;
            }
        }

        // 2026-09-25: Request-scoped LoRA routing. `upload_seq_slot_uniform` writes
        // this request's slot into the free gap at `meta_base + 128` and returns it,
        // or returns `DevicePtr(0)` (the installed-pair path) when no LoRA pool is
        // loaded or the request resolves to the active adapter (`adapter_slot ==
        // -1` resolves to active).
        let seq_slot =
            self.upload_seq_slot_uniform(seq.adapter_slot, 1, meta_base.offset(128), stream)?;

        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: max_blocks,
            num_seqs: 1,
            seq_slot,
            moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        };

        // 2026-09-25: Lift the FP8-KV calibration graph suppression once every
        // calibrating layer has frozen.
        if self.config.fp8_kv_calibration_tokens > 0
            && self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && self.fp8_calibration_frozen()
        {
            self.suppress_graphs
                .store(false, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("FP8 calibration frozen — re-enabling CUDA graphs");
        }
        // 2026-09-25: High-speed swap (`cache_blocks_per_seq` set) runs eagerly: its
        // per-step host D2H and disk I/O cannot run under capture.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // 2026-09-25: Run the first decode step eagerly when dumping, so the
        // per-layer probes can synchronise.
        let dump_step0 = seq.seq_len == seq.prompt_len && self.levers.ssm_save_dump;
        // 2026-09-25: `METRALE_EP_GRAPHS` (`levers.ep_graphs`) allows capture with an
        // EP `comm` backend.
        let ep_graphs = self.levers.ep_graphs;
        // 2026-09-25: `METRALE_GDN_DECODE_GRAPH` (`levers.gdn_decode_graph`, off by
        // default) allows capture with a `comm` backend (GDN HeadParallel TP). The
        // collectives go through `all_reduce_async`, and `begin_capture` uses
        // relaxed capture mode. Every per-token input is uploaded to a stable
        // buffer before any replay.
        let gdn_graphs = self.levers.gdn_decode_graph;
        // 2026-09-25: `METRALE_LORA_EAGER` (`levers.lora_eager`) forces eager decode
        // while a LoRA pool is loaded, to compare graph and eager output.
        let lora_eager = self.lora.is_some() && self.levers.lora_eager;
        // 2026-09-25: A layer that can never be captured (QSA's host top-k) vetoes
        // graphs for the whole model: a graph captured on the dense path would
        // replay the wrong attention once selection activates.
        let layer_veto = self.decode_graph_veto;
        let use_graphs = (self.comm.is_none() || ep_graphs || gdn_graphs)
            && !self.profile
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !dump_step0
            && !lora_eager
            && !layer_veto;

        let host_tok = input.host();
        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(attn_metadata),
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            decode_step: true,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            // 2026-09-25: The token id uploaded above, read by hash-MoE at offset 0.
            token_ids: Some(self.buffers.token_ids()),
            // 2026-09-25: The same id on the host, for PLE's host-side hash.
            host_token_ids: host_tok.as_ref().map(std::slice::from_ref),
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };

        // 2026-09-25: Profile mode: per-layer synchronised decode for a timing breakdown.
        if let (true, Some(token)) = (self.profile, input.host()) {
            return self.decode_profiled(token, hidden, residual, seq, &mut kv_cache, &ctx, stream);
        }
        // 2026-09-25: Graph path: the periodic SSM normalization runs here, outside capture.
        self.normalize_ssm_outside_graph(seq, stream, use_graphs);
        let mut graph_cache = if use_graphs {
            Some(self.decode_graph.lock())
        } else {
            None
        };

        // 2026-09-25: Graphs are keyed by the sequence's SSM slot. Positions, slot,
        // seq_len and block table are read from the buffers uploaded above.
        if let Some(ref cache) = graph_cache
            && let Some(graph) = cache.get(&seq.slot_idx)
            && graph.0 != 0
        {
            // 2026-09-25: Check room before the replay. The graph writes GLM-5.3's
            // DSA indexer row from a device position, so past the DSA ceiling it
            // would write past the buffer; `sync_replayed_step` below runs only
            // after that write.
            for (i, layer) in self.layers.iter().enumerate() {
                layer.check_replay_room(&*seq.layer_states[i], seq.seq_len, 1)?;
            }
            self.gpu.launch_graph(*graph, stream)?;
            // 2026-09-25: A replay runs kernels only, so a layer with per-sequence
            // host bookkeeping (GLM-5.3's DSA indexer cache length) advances it
            // here. The default `sync_replayed_step` does nothing.
            for (i, layer) in self.layers.iter().enumerate() {
                layer.sync_replayed_step(seq.layer_states[i].as_mut(), seq.seq_len, 1)?;
            }
            if let Some(token) = input.host() {
                seq.tokens.push(token);
                seq.seq_len += 1;
            }
            return Ok(self.decode_logits_ptr());
        }

        // 2026-09-25: Whether a capture is recording. A `begin_capture` failure
        // runs the step eagerly and disables graphs for the rest of the run.
        let mut capture_active = false;
        if use_graphs {
            tracing::info!(
                "CUDA graph capture: starting for {} layers",
                self.layers.len()
            );
            match self.gpu.begin_capture(stream) {
                Ok(()) => capture_active = true,
                Err(e) => {
                    tracing::warn!(
                        "CUDA graph begin_capture failed ({e:#}) — \
                         running eagerly and disabling graph capture"
                    );
                    self.suppress_graphs
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        let probe_layers =
            !use_graphs && seq.seq_len == seq.prompt_len && self.levers.ssm_save_dump;
        if let Err(e) = self.decode_forward_body(
            hidden,
            residual,
            seq,
            &mut kv_cache,
            &ctx,
            probe_layers,
            use_graphs,
            stream,
        ) {
            // 2026-09-25: A body error during capture (for example a MoE LoRA refusal)
            // leaves the stream capturing, and every later op on it would fail.
            // Release the stream, discarding the partial graph, before handling
            // the error; no-op when not capturing.
            self.gpu.abort_capture_if_active(stream);
            // 2026-09-25: A capture error (status 900/901, `STREAM_CAPTURE`) belongs to
            // the graph attempt: capture records without executing, so nothing ran
            // for this token. Re-run eagerly and disable graphs, as the
            // `end_capture` failure arm does. Any other error is returned, and
            // graphs stay enabled.
            let msg = format!("{e:#}");
            let capture_poison = capture_active
                && (msg.contains("status 901")
                    || msg.contains("status 900")
                    || msg.contains("STREAM_CAPTURE"));
            if !capture_poison {
                return Err(e);
            }
            tracing::warn!(
                "decode body failed under CUDA graph capture ({msg}) — \
                 re-running eagerly and disabling graph capture"
            );
            self.suppress_graphs
                .store(true, std::sync::atomic::Ordering::Relaxed);
            capture_active = false;
            // 2026-09-25: Re-arm the per-step prestaged layer state the recorded
            // attempt consumed.
            for (li, l) in self.layers.iter().enumerate() {
                l.decode_prestage_rearm(seq.layer_states[li].as_mut());
            }
            self.decode_forward_body(
                hidden,
                residual,
                seq,
                &mut kv_cache,
                &ctx,
                false,
                false,
                stream,
            )?;
        }

        // 2026-09-25: Gemma-4 logits diagnostic; no-op unless `METRALE_DIAG_GEMMA4`.
        if let Some(token) = input.host() {
            self.diag_gemma4_decode_logits(token, stream)?;
        }

        if capture_active {
            match self.gpu.end_capture(stream) {
                Ok(graph) if graph.0 != 0 => {
                    tracing::info!(
                        "CUDA graph captured successfully for slot={} (handle={:?})",
                        seq.slot_idx,
                        graph.0
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(seq.slot_idx, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
                Ok(_) => {
                    tracing::warn!("CUDA graph capture returned null handle — running eagerly");
                }
                Err(e) => {
                    // 2026-09-25: Capture records without executing, so nothing
                    // ran for this token: re-run the body eagerly (a failed
                    // `end_capture` ends the capture) and disable graphs for the
                    // rest of the run.
                    tracing::warn!(
                        "CUDA graph end_capture failed ({e:#}) — \
                         re-running decode step eagerly and disabling graph capture"
                    );
                    self.suppress_graphs
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    // 2026-09-25: Re-arm the prestaged state, as in the body-error
                    // arm above.
                    for (li, l) in self.layers.iter().enumerate() {
                        l.decode_prestage_rearm(seq.layer_states[li].as_mut());
                    }
                    self.decode_forward_body(
                        hidden,
                        residual,
                        seq,
                        &mut kv_cache,
                        &ctx,
                        false,
                        false,
                        stream,
                    )?;
                }
            }
        }

        // 2026-09-25: Red-zone scan (`METRALE_REDZONE`), after the whole step with
        // the device drained, so a violation names the step that caused it.
        if redzone_every() > 0 && seq.seq_len.is_multiple_of(redzone_every()) {
            self.gpu.synchronize(stream)?;
            match self.gpu.scan_redzones() {
                Ok(0) => {}
                Ok(n) => tracing::error!(
                    "🔴 {n} red-zone violation(s) after decode step at seq_len={}",
                    seq.seq_len
                ),
                Err(e) => tracing::warn!("red-zone scan failed: {e:#}"),
            }
        }

        if let Some(token) = input.host() {
            seq.tokens.push(token);
            seq.seq_len += 1;
        }

        Ok(self.decode_logits_ptr())
    }
}
