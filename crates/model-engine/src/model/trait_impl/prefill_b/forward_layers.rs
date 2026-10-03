// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-stream prefill layer loop: every layer's prefill (or, for a
//! single-token pass after position 0, its decode), the per-layer DFlash capture,
//! the MTP drafter capture, and env-gated timing and dumps.
//!
//! Owner: model-engine prefill.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext};

impl TransformerModel {
    pub(super) fn prefill_b_forward_layers(
        &self,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        proc_count: usize,
        effective_seq_len_start: usize,
        kv_write_start: usize,
        marconi_skip: bool,
        meta_base: DevicePtr,
        slot_offset: usize,
        pos_stream_bytes: usize,
        use_mrope: bool,
        needs_paged: bool,
        midcap: Option<&super::midchunk_capture::MidCapturePlan>,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let elem_bytes = 2usize;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let (block_table_dev, seq_len_dev) = if needs_paged {
            let page_meta = seq.chunked_prefill_meta.as_ref().unwrap();
            (page_meta.block_table, page_meta.seq_len)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        let (positions_h_dev, positions_w_dev) = if use_mrope {
            (
                meta_base.offset(pos_stream_bytes),
                meta_base.offset(pos_stream_bytes * 2),
            )
        } else {
            (meta_base, meta_base)
        };

        // 2026-09-25: Adapter routing: `proc_count` rows all holding this request's
        // adapter slot, or `DevicePtr(0)` when there is no adapter pool or the request
        // uses the active adapter (`upload_seq_slot_uniform`).
        let seq_slot = self.upload_seq_slot_uniform(
            seq.adapter_slot,
            proc_count,
            self.buffers.lora_seq_slot(),
            stream,
        )?;
        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: positions_h_dev,
            positions_w: positions_w_dev,
            slot: meta_base.offset(slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: seq.block_table.len() as u32,
            num_seqs: 1,
            seq_slot,
            moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        };

        // 2026-09-25: `METRALE_PROFILE_FIRST` profiles one pass: the flag is swapped off here.
        let profile_now = self.profile
            || self
                .profile_first_pending
                .swap(false, std::sync::atomic::Ordering::Relaxed);

        // 2026-09-25: Mid-chunk tail capture: a fresh counter per pass. Each SSM layer's
        // prefill takes one ordinal from it, in model order, to index the plan's
        // per-layer destinations (qwen3_ssm/trait_prefill_block.rs).
        let midcap_counter = std::sync::atomic::AtomicUsize::new(0);
        let midchunk_capture = midcap.map(|p| metrale_model_layers::layer::MidchunkCapture {
            cap_local: p.cap_local,
            h_dsts: &p.h_dsts,
            conv_dsts: &p.conv_dsts,
            h_bytes: p.h_bytes,
            conv_bytes: p.conv_bytes,
            ssm_layer_counter: &midcap_counter,
            cap_local_early: p.cap_local_early,
            h_dsts_early: &p.h_dsts_early,
            conv_dsts_early: &p.conv_dsts_early,
            // 2026-10-03: In-pass capture fields (`inpass_capture.rs`); `live_h` is empty for
            // the tail mid-chunk plan, whose layers use the counter above.
            seq_pos_start: effective_seq_len_start,
            live_h: &p.live_h,
            captured: &p.captured,
        });

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
            profile: profile_now,
            comm: self.comm_ref(),
            graph_capture: false,
            decode_step: false,
            // 2026-09-25: After a snapshot restore, GDN layers do not take the FLA
            // chunked kernel (`gdn_exact_replay` in qwen3_ssm/trait_prefill_recur.rs).
            gdn_exact_replay: marconi_skip,
            gdn_write_on_accept: false,
            // 2026-09-25: Hash-MoE reads this chunk's token ids, staged by
            // `prefill_b_embed_chunk_at`.
            token_ids: Some(self.buffers.token_ids()),
            host_token_ids: None,
            // 2026-09-25: `None` unless the request routes to a non-active adapter slot.
            routed_lora_layers: self.routed_slot_layers(seq.adapter_slot),
            midchunk_capture,
            moe_lora_route: self.moe_lora_route(seq.adapter_slot),
        };

        // 2026-09-25: A single-token pass after position 0 (for example the last-token
        // re-run in `proc_range`) runs each layer's decode path.
        let use_decode_path = proc_count == 1 && effective_seq_len_start > 0;
        // 2026-09-25: After a snapshot restore this pass recomputes the positions between
        // the snapshot and `seq.cached_prefix_tokens` (the radix match), whose K/V already
        // sits in shared prefix-cache blocks. Their count within this pass is the layers'
        // K/V write floor, so a recomputed value that is not bit-identical never
        // overwrites a shared block. Positions from the match on are written normally.
        let layer_kv_write_start = if marconi_skip {
            seq.cached_prefix_tokens
                .saturating_sub(effective_seq_len_start)
                .min(proc_count)
        } else {
            kv_write_start
        };
        let prefill_t0 = if profile_now {
            self.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut layer_times: Vec<u128> = Vec::new();
        // 2026-09-25: `METRALE_PREFILL_HOST_TIMING=1`: host wall-clock spans with no
        // synchronize, unlike `profile_now`, which synchronizes per layer.
        let host_timing = std::env::var("METRALE_PREFILL_HOST_TIMING").as_deref() == Ok("1");
        let t_loop = host_timing.then(std::time::Instant::now);
        let mut t_in_prefill = std::time::Duration::ZERO;
        let mut t_dflash = std::time::Duration::ZERO;
        for (i, layer) in self.layers.iter().enumerate() {
            let t_pf = host_timing.then(std::time::Instant::now);
            let lt0 = if profile_now {
                self.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            if use_decode_path {
                layer
                    .decode(
                        hidden,
                        residual,
                        seq.layer_states[i].as_mut(),
                        kv_cache,
                        effective_seq_len_start,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )
                    .map_err(|e| anyhow::anyhow!("Prefill-as-decode layer {i} failed: {e}"))?;
            } else {
                layer
                    .prefill(
                        hidden,
                        residual,
                        proc_count,
                        seq.layer_states[i].as_mut(),
                        kv_cache,
                        effective_seq_len_start,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        layer_kv_write_start,
                        &ctx,
                        stream,
                    )
                    .map_err(|e| anyhow::anyhow!("Prefill chunk layer {i} failed: {e}"))?;
            }
            if let Some(t) = t_pf {
                t_in_prefill += t.elapsed();
            }
            let t_df = host_timing.then(std::time::Instant::now);
            // 2026-09-25: The DFlash capture's base is `effective_seq_len_start`, the absolute
            // position of hidden row 0 (row t lands at position base + t), not the K/V
            // write floor.
            self.try_dflash_prefill_capture_layer(
                seq,
                i,
                effective_seq_len_start,
                proc_count,
                stream,
            )?;
            if let Some(t) = t_df {
                t_dflash += t.elapsed();
            }
            if let Some(lt0) = lt0 {
                self.gpu.synchronize(stream)?;
                layer_times.push(lt0.elapsed().as_micros());
            }
            // 2026-09-25: `METRALE_DUMP_HYPER_RMS=1` (hyper-connection models): after each
            // layer, log the RMS of the FP32 `hc_streams` buffer, `[proc_count, hc_mult, H]`.
            // Debug only: one D2H copy per layer.
            if std::env::var("METRALE_DUMP_HYPER_RMS").as_deref() == Ok("1")
                && self.config.hc_mult > 0
            {
                let n = proc_count * self.config.hc_mult * self.config.hidden_size;
                let mut buf = vec![0u8; n * 4];
                self.gpu
                    .copy_d2h_on_stream(ctx.buffers.hc_streams(), &mut buf, stream)?;
                let vals: &[f32] =
                    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, n) };
                let ssq: f64 = vals.iter().map(|&v| (v as f64) * (v as f64)).sum();
                tracing::info!(
                    "HYPER_RMS layer {i} tokens {proc_count} rms {:.6}",
                    (ssq / n as f64).sqrt()
                );
            }
            // 2026-09-25: Mistral profiling diagnostic, latched once per model by
            // `ModelStats::dumped`.
            if profile_now
                && self.config.model_type == "mistral"
                && self.stats.dumped.keyed("mla_chunk_norms")
            {
                self.gpu.synchronize(stream)?;
                let last_offset = (proc_count - 1) * self.config.hidden_size * 4;
                let h_sz = self.config.hidden_size;
                let mut buf = vec![0u16; h_sz];
                // 2026-09-25: SAFETY: `buf` owns `h_sz * 2` initialised bytes, and `bytes` is
                // the only reference to them until the `copy_d2h` below returns.
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, h_sz * 2)
                };
                if self.gpu.copy_d2h(hidden.offset(last_offset), bytes).is_ok() {
                    let vals: Vec<f32> = buf
                        .iter()
                        .map(|&b| f32::from_bits((b as u32) << 16))
                        .collect();
                    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::info!(
                        "LAYER_NORM L{i}/{}: hidden_norm={norm:.4}",
                        self.layers.len()
                    );
                    if i == self.layers.len() - 1 {}
                }
            }
            if profile_now && (i < 4 || i >= self.layers.len() - 4) {
                self.gpu.synchronize(stream)?;
                let (_, norm) = self.readback_bf16(hidden, self.config.hidden_size.min(64))?;
                tracing::info!("L{i} hidden[0] norm={norm:.4}");
            }
            // 2026-09-25: `METRALE_NEMO_DUMP=<dir>`, last chunk only: after each layer, write
            // the last token's hidden row as headerless little-endian f32 to
            // `<dir>/metrale_L{i}.bin`.
            if is_last_chunk
                && let Ok(dir) = std::env::var("METRALE_NEMO_DUMP")
                && !dir.is_empty()
            {
                self.gpu.synchronize(stream)?;
                let last_start = (proc_count - 1) * h;
                let (vals, _) = self.readback_bf16(hidden.offset(last_start * elem_bytes), h)?;
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::create_dir_all(&dir).ok();
                let path = std::path::Path::new(&dir).join(format!("metrale_L{i}.bin"));
                std::fs::write(&path, &bytes).ok();
                if i == self.layers.len() - 1 {
                    tracing::info!(
                        "METRALE_NEMO_DUMP: wrote {} per-layer hidden \
                         vectors ({h} f32 each) to {dir}",
                        self.layers.len()
                    );
                }
            }
            if profile_now && is_last_chunk && proc_count > 1 && (chunk_start + chunk_len) > 16384 {
                self.gpu.synchronize(stream)?;
                let last_start = (proc_count - 1) * h;
                let (vals, norm) =
                    self.readback_bf16(hidden.offset(last_start * elem_bytes), h.min(16))?;
                let lt = self.config.layer_type(i);
                tracing::warn!(
                    "DIAG L{i} ({lt:?}) last_tok_norm={norm:.4} first2={:.4?}",
                    &vals[..2.min(vals.len())]
                );
            }
        }
        if let Some(t) = t_loop {
            let wall = t.elapsed();
            let ffn_us = metrale_model_layers::layers::qwen3_attention::take_ffn_host_us();
            let ph = metrale_model_layers::layers::qwen3_attention::take_attn_phase_us();
            // 2026-09-25: `loop_wall` is host time for the whole layer loop, `in_prefill`
            // the time inside each layer's `prefill` or `decode` call, and `dflash` the
            // per-layer DFlash capture.
            tracing::info!(
                "PREFILL HOST TIMING layers={} tokens={}: loop_wall={:.1}ms in_prefill={:.1}ms ffn={:.1}ms attn_rest={:.1}ms dflash={:.1}ms | qkv={:.1}ms mid={:.1}ms attn_kernel={:.1}ms",
                self.layers.len(),
                proc_count,
                wall.as_secs_f64() * 1e3,
                t_in_prefill.as_secs_f64() * 1e3,
                ffn_us as f64 / 1e3,
                (t_in_prefill.as_micros() as f64 - ffn_us as f64) / 1e3,
                t_dflash.as_secs_f64() * 1e3,
                ph[0] as f64 / 1e3,
                ph[1] as f64 / 1e3,
                ph[2] as f64 / 1e3,
            );
        }

        self.try_mtp_prefill_capture(seq, effective_seq_len_start, proc_count, stream)?;
        if let Some(t0) = prefill_t0 {
            self.gpu.synchronize(stream)?;
            let total_us = t0.elapsed().as_micros();
            let mut indexed: Vec<(usize, u128)> = layer_times.iter().copied().enumerate().collect();
            indexed.sort_by_key(|x| std::cmp::Reverse(x.1));
            let top5: Vec<String> = indexed
                .iter()
                .take(5)
                .map(|(i, us)| format!("L{}={:.2}ms", i, *us as f64 / 1000.0))
                .collect();
            let path_label = if use_decode_path { "decode" } else { "prefill" };
            // 2026-09-25: The same per-layer samples, summed by layer type.
            let mut by_type: std::collections::BTreeMap<String, (u128, usize)> =
                std::collections::BTreeMap::new();
            for (i, us) in layer_times.iter().copied().enumerate() {
                let e = by_type
                    .entry(format!("{:?}", self.config.layer_type(i)))
                    .or_insert((0, 0));
                e.0 += us;
                e.1 += 1;
            }
            let per_type: Vec<String> = by_type
                .iter()
                .map(|(k, (us, n))| {
                    format!(
                        "{}x{}={:.0}ms(avg {:.1})",
                        n,
                        k,
                        *us as f64 / 1000.0,
                        *us as f64 / 1000.0 / *n as f64
                    )
                })
                .collect();
            tracing::info!(
                "Prefill chunk {} tok (proc {}, {}): {:.1}ms total, by_type: {}, top5: {}",
                chunk_len,
                proc_count,
                path_label,
                total_us as f64 / 1000.0,
                per_type.join(", "),
                top5.join(", "),
            );
        }
        Ok(())
    }
}
