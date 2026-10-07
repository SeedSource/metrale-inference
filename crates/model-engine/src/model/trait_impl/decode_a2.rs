// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched decode (`decode_batch_dispatch`): one sequence goes to
//! `decode()`; an EP batch is announced to the workers; a batch that needs the
//! per-sequence path runs `decode()` per row; every other batch runs
//! `decode_batch_compute_main`: pad to the graph ladder, replay, borrow or capture
//! a CUDA graph, each layer's `decode_multi_seq`, final norm and a batched LM head.
//!
//! Owner: model-engine (decode).
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_config::LayerType;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::block_mgmt::{ensure_blocks_through_decode, extract_layer_refs};
use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};
use crate::traits::{ModelEp, ModelForward};
use metrale_model_layers::layer::{ForwardContext, LayerState, SsmLayerState};
use metrale_model_layers::layers::ops;

mod pad_states;
mod perseq;
mod replay_sync;
mod route;

/// 2026-09-25: Multi-sequence decode CUDA graphs: on unless
/// `METRALE_NO_DECODE_GRAPHS_MULTISEQ` is exactly `1`. Read once per process.
fn multiseq_graphs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_DECODE_GRAPHS_MULTISEQ").as_deref() != Ok("1"))
}

impl TransformerModel {
    pub(super) fn decode_batch_dispatch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = tokens.len();
        assert_eq!(n, seqs.len(), "tokens.len() must equal seqs.len()");
        // 2026-09-25: `--ssm-h-dtype f16`: narrow each sequence's SSM h-state to FP16
        // here, before any graph region. No-op without the flag.
        for s in seqs.iter_mut() {
            self.ssm_h_to_f16_dispatch(s)?;
        }

        // 2026-09-25: One sequence: announce it to EP workers, then `decode()` with
        // its slot-keyed graph cache. The announcement is made here, not in the
        // scheduler, so the per-sequence loop below can interleave announcements
        // with `decode()` calls.
        if n == 1 {
            self.ep_broadcast_cmd_for_seq(seqs[0].slot_idx as u32, tokens[0])?;
            self.decode(tokens[0], seqs[0], stream)?;
            return Ok(self.decode_logits_ptr());
        }

        // 2026-09-25: EP with n > 1: the head announces the batch
        // (`ep_broadcast_decode_batch_dispatch`: command `0xFFFFFFE0`, N, the slot
        // ids, the tokens), then both ranks run `decode_batch_compute_main` (the
        // worker from `ep_worker_decode_batch`), so each layer's collectives match
        // in shape and order across ranks.
        let mla_perseq_fallback = self.is_mla_dispatch() && self.levers.mla_perseq_fallback;
        let qsa_active = self.config.index_topk > 0 && {
            // 2026-09-25: `QsaIndexer::inert_bound`: below `index_topk +
            // index_compress_ratio - 1` visible tokens every block is selected,
            // so selection is inert.
            let bound = self.config.index_topk + self.config.index_compress_ratio - 1;
            seqs.iter().any(|s| s.seq_len >= bound)
        };
        // 2026-09-25: A layer may decline the batched multi-seq step. The veto sits
        // outside the `hc_mult > 0` conjunction on purpose: keyed on `hc_mult`,
        // `index_topk` or `model_type` instead, a declining model would be routed
        // per-sequence only once `qsa_active` holds, i.e. correct on long contexts
        // and wrong on short ones.
        let ms_layer_veto = self.layers.iter().any(|l| l.decode_multi_seq_unsupported());
        // 2026-10-01: Active selection forces the loop only when some layer shares one.
        let qsa_shared = qsa_active && !self.ms_selection_per_seq();
        let hc_perseq = ms_layer_veto
            || (self.config.hc_mult > 0 && (qsa_shared || self.levers.hc_perseq_decode));
        self.log_multi_seq_route(n, hc_perseq, ms_layer_veto, qsa_active);
        // 2026-09-25: The per-sequence decision is made before the EP branch, so an
        // EP batch that needs the per-sequence path takes it too.
        if self.comm.is_some() && !(mla_perseq_fallback || hc_perseq) {
            let seq_ids: Vec<u32> = seqs.iter().map(|s| s.slot_idx as u32).collect();
            self.ep_broadcast_decode_batch_dispatch(&seq_ids, tokens)?;
            return self.decode_batch_compute_main(tokens, seqs, stream);
        }

        // 2026-09-25: Per-sequence loop: `METRALE_MLA_PERSEQ_FALLBACK` on an MLA
        // model, or `hc_perseq` (a layer declines the batched step, or an mHC model
        // with QSA active for some sequence or `METRALE_HC_PERSEQ_DECODE` set). Each
        // sequence runs `decode()`, and its logits row is staged through the host.
        if mla_perseq_fallback || hc_perseq {
            return self.decode_batch_perseq(tokens, seqs, n, stream);
        }

        self.decode_batch_compute_main(tokens, seqs, stream)
    }

    /// 2026-09-25: The batched compute path shared by the head's EP branch and the
    /// worker's `ep_worker_decode_batch`: embed, KV-block allocation, metadata
    /// upload, each layer's `decode_multi_seq`, final norm and the batched LM head.
    /// It broadcasts nothing; the head announces the batch before calling it.
    pub(crate) fn decode_batch_compute_main(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        self.decode_batch_compute_main_with(
            super::feed::BatchInput::Host(tokens),
            seqs,
            stream,
            &mut None,
        )
    }

    /// 2026-09-25: `decode_batch_compute_main` over host ids or the device feed
    /// (`super::feed`). `graph_key_out` receives this step's exact graph key
    /// (`None` without graphs), which `feed.rs` pins while the launch is in
    /// flight. On a borrowed replay it is not the key of the graph replayed.
    pub(super) fn decode_batch_compute_main_with(
        &self,
        input: super::feed::BatchInput<'_>,
        seqs: &mut [&mut SequenceState],
        _stream: u64,
        graph_key_out: &mut Option<Vec<u32>>,
    ) -> Result<DevicePtr> {
        let n = input.len();
        // 2026-09-25: A batch with a row routed to a non-active adapter (`Refuse`)
        // cannot be served by the single-active MoE LoRA fold, whose per-row map
        // writes such rows as base. Refuse before any per-step work or graph
        // lookup. `stamp_decode_moe_batch` records the route at the `Model` entry.
        metrale_model_layers::lora::ensure_decode_route_servable(
            self.decode_moe_route(),
            "decode_batch_compute_main",
        )?;
        // 2026-09-25: `--ssm-h-dtype f16`: narrow each sequence's SSM h-state to FP16
        // here, before any graph region. No-op without the flag.
        for s in seqs.iter_mut() {
            self.ssm_h_to_f16_dispatch(s)?;
        }
        if self.levers.decode_batch_log {
            let slots: Vec<i64> = seqs
                .iter()
                .map(|s| {
                    s.ssm_slot
                        .as_ref()
                        .and_then(|g| g.idx())
                        .map(|x| x as i64)
                        .unwrap_or(-1)
                })
                .collect();
            let contiguous = slots.iter().enumerate().all(|(i, &s)| s == i as i64);
            tracing::info!(
                "METRALE_DECODE_BATCH: n={n} slots={slots:?} contiguous_0..n={contiguous}"
            );
        }
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // 2026-09-25: Pad to the graph ladder in `traits::padded_batch_n`.
        // 2026-10-01: An eager-only layer stack (`ms_eager_only`) runs exactly `n` rows.
        let ms_eager = self.ms_eager_only();
        // 2026-10-07: So does a layer stack that is graphed but unpadded (`ms_unpadded`).
        let ms_unpadded = self.ms_unpadded();
        let padded_n = if ms_eager || ms_unpadded {
            n
        } else {
            crate::traits::padded_batch_n(n)
        };

        // 2026-09-25: SSM state pointers are baked into the captured kernel
        // arguments, so batched graphs are keyed by the per-row SSM slot vector
        // (`batch_decode_graph_key`); every other captured input is at a fixed
        // address refreshed before replay.
        let ms_profile = self.levers.ms_profile;
        // 2026-09-25: `METRALE_MS_PROFILE` runs eagerly so its per-phase syncs are
        // legal; `METRALE_LORA_EAGER` is the same LoRA hatch as in `decode_a`.
        let lora_eager = self.lora.is_some() && self.levers.lora_eager;
        // 2026-09-25: The per-layer graph veto, as in `decode_a` (QSA's host top-k,
        // PLE's host hash on the hc multi-seq path).
        let layer_veto = self.decode_graph_veto;
        let graph_key = if !ms_profile
            && !lora_eager
            && !layer_veto
            && !ms_eager
            && multiseq_graphs_enabled()
        {
            self.batch_decode_graph_key(&*seqs, padded_n)
        } else {
            None
        };
        let use_graphs = graph_key.is_some();

        // 2026-09-25: Lock order: kv_cache before the graph cache, as in verify_e.
        let mut kv_cache = self.kv_cache.lock();

        let mut graphs = if use_graphs {
            Some(self.batch_decode_graphs.lock())
        } else {
            None
        };

        // 2026-09-25: Exact hit (LRU-touched), else borrow a wider captured graph
        // whose first `n` rows are this batch's slots and whose tail rows are the
        // dummy slot or free slots (`find_borrowable_decode_key`). `dispatch_n` is
        // the width the per-step work below prepares: the borrowed width, else
        // `padded_n`.
        let mut replay: Option<metrale_gpu_runtime::gpu::GraphHandle> = None;
        let mut dispatch_n = padded_n;
        if let (Some(g), Some(key)) = (&mut graphs, &graph_key) {
            g.1 += 1;
            let tick = g.1;
            if let Some(e) = g.0.get_mut(key) {
                e.1 = tick;
                replay = Some(e.0);
            } else if super::graph_borrow::graph_borrow_enabled()
                && !ms_unpadded
                && self.comm.is_none()
                && self.config.num_ssm_layers() > 0
            {
                let dummy = self.ssm_pool.dummy_slot() as u32;
                let borrowed =
                    super::graph_borrow::find_borrowable_decode_key(&key[..n], g.0.keys(), |s| {
                        s == dummy || self.ssm_pool.slot_is_free(s as usize)
                    });
                // 2026-09-25: A key that keeps borrowing has settled at its width:
                // decline and capture it (`borrow_streak.rs`).
                if let Some(bk) = borrowed
                    && super::borrow_streak::DECODE_BORROW_STREAK.allow(key)
                {
                    dispatch_n = bk.len();
                    let e =
                        g.0.get_mut(&bk)
                            .expect("borrowed key comes from this cache");
                    e.1 = tick;
                    replay = Some(e.0);
                    // 2026-09-25: Logged once per (key, borrowed key) transition;
                    // repeats stay silent.
                    if super::graph_borrow::DECODE_BORROW_LOG.should_log(key, &bk) {
                        tracing::info!(
                            "decode graph borrow: n={n} padded_n={padded_n} -> replaying \
                             captured width {dispatch_n}"
                        );
                    }
                }
            }
        }

        match input.host() {
            Some(tokens) => {
                for (i, &tok) in tokens.iter().enumerate() {
                    // 2026-09-25: Each row is a different sequence: the n-gram
                    // context comes from that sequence's own history.
                    self.embed_ctx(&seqs[i].tokens, tok, hidden.offset(i * h * fp32), stream)?;
                }
            }
            None => self.feed_embed_rows(hidden, n, stream)?,
        }

        for i in n..dispatch_n {
            self.gpu.memset(hidden.offset(i * h * fp32), 0, h * fp32)?;
        }

        let bs = kv_cache.block_size();
        // 2026-10-05: Rank-agreed lazy-map admission (A168), before the step's first collective
        // and any capture or replay (`decode_lazy_agree.rs`).
        for seq in seqs.iter_mut() {
            self.agree_decode_lazy_maps(seq, seq.seq_len / bs, bs)?;
        }
        for seq in seqs.iter_mut() {
            let blocks_needed = (seq.seq_len / bs) + 1;
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

        // 2026-09-25: Upload metadata for `dispatch_n` rows (active and padding).
        let metadata = self.upload_batch_metadata_fixed(seqs, dispatch_n, &mut kv_cache, stream)?;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            // 2026-09-25: `Refuse` was rejected above. `upload_batch_metadata_fixed`
            // supplies the per-row MoE adapter map for `Fold`.
            moe_lora_route: self.decode_moe_route(),
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            decode_step: true,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            // 2026-09-25: The batch's host token ids, for layers that hash them on
            // the host (PLE).
            host_token_ids: input.host(),
            routed_lora_layers: None,
            midchunk_capture: None,
        };

        if let Some(graph) = replay {
            if graph.0 != 0 {
                // 2026-10-07: Per-row room check and host bookkeeping (`replay_sync.rs`).
                self.ms_replay_check_room(seqs)?;
                self.gpu.launch_graph(graph, stream)?;
                self.ms_replay_sync(seqs)?;
            }

            if let Some(tokens) = input.host() {
                for (i, seq) in seqs.iter_mut().enumerate() {
                    seq.tokens.push(tokens[i]);
                    seq.seq_len += 1;
                }
            }
            *graph_key_out = graph_key;
            return Ok(self.decode_logits_ptr());
        }
        {
            // 2026-09-25: No graph to replay: capture one when `use_graphs`, else run
            // eagerly. Per-row inputs cover all `padded_n` rows.
            let seq_lens: Vec<usize> = (0..padded_n)
                .map(|i| if i < n { seqs[i].seq_len } else { 0 })
                .collect();
            let block_tables: Vec<Vec<u32>> = (0..padded_n)
                .map(|i| {
                    if i < n {
                        seqs[i].block_table.clone()
                    } else {
                        vec![self.dummy_kv_block]
                    }
                })
                .collect();

            let mut all_layer_states: Vec<Vec<Box<dyn LayerState>>> = seqs
                .iter_mut()
                .map(|s| std::mem::take(&mut s.layer_states))
                .collect();

            self.decode_batch_push_pad_states(&mut all_layer_states, n, padded_n)?;

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            // 2026-09-25: `METRALE_CONC_HSD`: after the embed and after each layer,
            // log the first 16 bytes of each row's hidden state read as 4 f32
            // values.
            let conc_hsd = self.levers.conc_hsd && padded_n >= 2 && self.comm.is_none();
            let dump_hidden = |label: &str, stream: u64| -> Result<()> {
                if !conc_hsd {
                    return Ok(());
                }
                self.gpu.synchronize(stream)?;
                let mut bufs: Vec<Vec<f32>> = Vec::with_capacity(padded_n);
                for i in 0..padded_n {
                    let mut buf = vec![0u8; 4 * 4];
                    let _ = self.gpu.copy_d2h(hidden.offset(i * h * fp32), &mut buf);
                    let vals: Vec<f32> = buf
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    bufs.push(vals);
                }
                let pretty: Vec<String> = bufs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("s{i}=[{:.4},{:.4},{:.4},{:.4}]", v[0], v[1], v[2], v[3]))
                    .collect();
                tracing::info!("CONC_HSD {label}: {}", pretty.join(" "));
                Ok(())
            };

            dump_hidden("post_embed", stream)?;

            let mut ssm_us: u128 = 0;
            let mut attn_us: u128 = 0;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let mut layer_state_refs = extract_layer_refs(&mut all_layer_states, layer_idx);
                let t0 = if ms_profile {
                    self.gpu.synchronize(stream).ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                layer.decode_multi_seq(
                    hidden,
                    residual,
                    padded_n,
                    &mut layer_state_refs,
                    &mut kv_cache,
                    &seq_lens,
                    &block_tables,
                    &ctx,
                    stream,
                )?;
                if let Some(t0) = t0 {
                    self.gpu.synchronize(stream).ok();
                    let dt = t0.elapsed().as_micros();
                    if self.config.layer_type(layer_idx) == LayerType::LinearAttention {
                        ssm_us += dt;
                    } else {
                        attn_us += dt;
                    }
                }
                // 2026-09-25: DFlash capture of every row's hidden (row i =
                // sequence i). It runs inside the graph region with fixed source
                // and destination addresses, so replays capture too; a borrowed
                // wider graph also writes the rows past `n`. No-op without DFlash
                // or off a capture layer.
                self.try_dflash_capture_all(layer_idx, padded_n, stream)?;
                if conc_hsd {
                    let _ = dump_hidden(&format!("after_L{:02}", layer_idx), stream);
                }
            }
            if ms_profile {
                self.gpu.synchronize(stream).ok();
            }
            let lmhead_t0 = if ms_profile {
                Some(std::time::Instant::now())
            } else {
                None
            };

            let normed = self.buffers.norm_output();
            self.final_norm_apply(
                hidden,
                normed,
                padded_n as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            // 2026-09-25: One batched LM head for all `padded_n` rows, so the vocab
            // weight is read once per step. The kernel ladder is in
            // `lm_head_batched.rs` (an FP8 head still runs one GEMV per row); the
            // mixed co-dispatch head `decode_b2::mixed_final_norm_lm_head` calls the
            // same function. The returned pointer is dropped: the function ends
            // with `decode_logits_ptr()`, the same buffer.
            self.lm_head_project_batched(normed, padded_n, h, bf16, stream)?;
            if let Some(t0) = lmhead_t0 {
                self.gpu.synchronize(stream).ok();
                let head_us = t0.elapsed().as_micros();
                let total = ssm_us + attn_us + head_us;
                tracing::info!(
                    "METRALE_MS_PROFILE n={n} padded_n={padded_n}: total={}us  ssm={}us({}L)  attn={}us({}L)  head={}us  [per-tok {:.2}ms]",
                    total,
                    ssm_us,
                    self.config.num_ssm_layers(),
                    attn_us,
                    self.layers.len() - self.config.num_ssm_layers(),
                    head_us,
                    total as f64 / 1000.0 / padded_n as f64,
                );
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "Captured CUDA graph for batch size {padded_n} (n={n}, slots={graph_key:?})"
                    );
                    if let (Some(g), Some(key)) = (graphs.as_mut(), graph_key.clone()) {
                        self.insert_batch_decode_graph(g, key, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }

            // 2026-09-25: Give the real layer states back; the padding states drop.
            for (seq, ls) in seqs.iter_mut().zip(all_layer_states.drain(..n)) {
                seq.layer_states = ls;
            }
        }

        if let Some(tokens) = input.host() {
            for (i, seq) in seqs.iter_mut().enumerate() {
                seq.tokens.push(tokens[i]);
                seq.seq_len += 1;
            }
        }
        *graph_key_out = graph_key;

        Ok(self.decode_logits_ptr())
    }
}
