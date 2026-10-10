// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The DFlash drafter's block forward: embed each sequence's last token and
//! mask tokens, run the drafter layers, then the final norm, lm_head and token
//! selection, and copy the drafted tokens to the host.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;
use metrale_model_layers::layer::ForwardContext;

mod ctx_prologue;
mod dims;
mod embed;
mod graphs;
mod host_drafts;
mod tail;

use dims::BlockDims;

/// 2026-09-25: Whether the 16-row rt2 vocab GEMV covers every row of this forward.
///
/// `g` is the total row count (`width * n_seq`), not the per-sequence width:
/// `ops::fp8_gemv_rowscale_batch16_rt2` computes only the `m` rows it is given and
/// refuses `m` outside `1..=16`, so a batch above 16 rows must take another kernel.
pub(super) fn rt2_16_covers_the_batch(g: u32) -> bool {
    (1..=16).contains(&g)
}

impl BlockDiffusionDraftHead {
    /// 2026-09-25: Run one drafter block forward and return the drafted token of every
    /// row (`block_g()` rows per sequence, seq-major), after the DSpark shift and
    /// confidence truncation when those apply.
    ///
    /// `option_b`: `Some((block_table_dev, ctx_count))` runs the paged layer body. The
    /// caller has already written the ctx K/V into the drafter's paged cache at slots
    /// `[0, ctx_count)`; each layer writes the block K/V at
    /// `[ctx_count, ctx_count + block_g())` and attends over both. `None` runs the
    /// non-paged layer body over the ctx rows in `ctx_buffer`.
    ///
    /// `batch`: `Some` packs several sequences seq-major. `last_token`, `position` and
    /// `option_b` then describe sequence 0, and the batch supplies every sequence's
    /// values.
    pub(super) fn forward_block(
        &self,
        last_token: u32,
        position: usize,
        ctx: &ForwardContext,
        stream: u64,
        ctx_buffer: Option<(DevicePtr, usize)>,
        option_b: Option<(DevicePtr, u32)>,
        batch: Option<&super::DflashBatch<'_>>,
    ) -> Result<Vec<u32>> {
        let d = self.block_dims(last_token, position, ctx, ctx_buffer, option_b, batch);
        let BlockDims {
            gpu,
            n_seq,
            width,
            g,
            h,
            q_dim,
            kv_dim,
            inter,
            bf16,
            inv_sqrt_d,
            levers,
            ctx_base_ptr,
            ctx_total,
            eff_ctx,
            option_b_on,
            n_attn,
            ctx_slot_bytes,
            ..
        } = d;

        // 2026-09-25: METRALE_DFLASH_DEBUG_DUMP=1: log the first `n` BF16 values of an
        // intermediate after syncing the stream.
        let debug_dump = levers.debug_dump;
        let dump_bf16 =
            |label: &str, ptr: metrale_gpu_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
                if !debug_dump {
                    return Ok(());
                }
                let mut buf = vec![0u8; n * 2];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(ptr, &mut buf)?;
                let vals: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                tracing::info!("DFLASH DUMP {label} [{n}]: {:?}", &vals);
                Ok(())
            };

        self.block_precompute_diag(&d, ctx, stream)?;

        // 2026-09-25: Project each of the `eff_ctx` most recent ctx rows through `fc`
        // (`target_layer_ids.len() * target_hidden_size` → `hidden_size`), one GEMV per
        // row, then RMSNorm them with `hidden_norm`, into `scratch.fc_proj` as
        // `[eff_ctx, hidden]`.
        if let Some(base) = ctx_base_ptr {
            let start_slot = self.block_ctx_seed(&d, base, &dump_bf16)?;
            // 2026-09-25: METRALE_DFLASH_DEBUG_DUMP_FULL=1, once per model: write the
            // `eff_ctx` ctx slots (`ctx_slot_bytes` each) to
            // /tmp/metrale_target_hidden.bin and the shapes to
            // /tmp/metrale_dflash_meta.json.
            if eff_ctx > 0
                && ctx.stats.dumped.keyed("dflash_target_hidden")
                && levers.debug_dump_full
            {
                let n_bytes = eff_ctx * ctx_slot_bytes;
                let mut buf = vec![0u8; n_bytes];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(base.offset(start_slot * ctx_slot_bytes), &mut buf)?;
                if let Err(e) = std::fs::write("/tmp/metrale_target_hidden.bin", &buf) {
                    tracing::warn!("DFLASH DUMP_FULL: target_hidden write failed: {e}");
                } else {
                    tracing::info!(
                        "DFLASH DUMP_FULL: wrote {} bytes ({} ctx slots × {} BF16 elements) to /tmp/metrale_target_hidden.bin (last_token={}, position={}, eff_ctx={})",
                        n_bytes,
                        eff_ctx,
                        ctx_slot_bytes / 2,
                        last_token,
                        position,
                        eff_ctx,
                    );
                }

                let meta = self.dump_full_meta(&d);
                if let Err(e) = std::fs::write("/tmp/metrale_dflash_meta.json", &meta) {
                    tracing::warn!("DFLASH DUMP_FULL: meta JSON write failed: {e}");
                } else {
                    tracing::info!(
                        "DFLASH DUMP_FULL: wrote /tmp/metrale_dflash_meta.json companion to target_hidden"
                    );
                }
            }
            self.block_ctx_project(&d, stream, base, start_slot, &dump_bf16)?;
        }

        let pos_host = self.block_positions(&d)?;
        if debug_dump {
            tracing::info!(
                "DFLASH DUMP positions: eff_ctx={} ctx_total={} position={} pos_ids[0..min(8,n_attn)]={:?}",
                eff_ctx,
                ctx_total,
                position,
                &pos_host[..pos_host.len().min(8)]
            );
        }

        let token_ids_host = self.block_token_ids(&d)?;
        if debug_dump {
            tracing::info!(
                "DFLASH DUMP token_ids_host: last_token={} mask={} eff_ctx={} ids[0..8]={:?}",
                last_token,
                self.mask_token_id,
                eff_ctx,
                &token_ids_host[..token_ids_host.len().min(8)],
            );
        }
        self.block_embed(&d, stream, &token_ids_host)?;

        // 2026-09-25: Drafter layers: the paged body (forward_block_layer_paged.rs) over
        // the block rows, or the non-paged body (forward_block_layer.rs) over `n_attn`
        // rows. For the paged body the block rows' cache slots are built once here and
        // shared by every layer.
        let slot_mapping_gamma_opt = self.block_paged_slots(&d, stream)?;

        // 2026-09-25: CUDA graphs. Each layer's pre-attention and post-attention calls
        // and the tail are captured as separate subgraphs, with attention run eagerly
        // between them. The host writes above go to fixed device buffers the subgraphs
        // read, so one capture per block width serves every later propose at that
        // width. Capture needs the paged path, one sequence, `suppress_graphs` unset,
        // and no debug dump or graph-suppressing lever; `levers.propose_warmup_n` eager
        // passes per width come first.
        let graph_eligible = option_b_on
            // 2026-09-25: A capture fixes the row count, so a batch runs eagerly.
            && n_seq == 1
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !debug_dump
            // 2026-09-25: Presence, not value: `METRALE_DFLASH_BLOCK_DUMP=0` also keeps
            // capture off (`GRAPH_SUPPRESSING_DIAGNOSTICS` in levers.rs).
            && !levers.any_diagnostic_armed
            // 2026-10-09: The two-rank drafter's collectives run eagerly.
            && self.tp.is_none();

        let warmup_target = levers.propose_warmup_n;

        let bf16_local = bf16;
        let inv_sqrt_d_local = inv_sqrt_d;
        let h_local = h;
        let n_attn_local = n_attn;
        let q_dim_local = q_dim;
        let kv_dim_local = kv_dim;
        let inter_local = inter;
        let eff_ctx_local = eff_ctx;
        let noise_byte_offset_local = eff_ctx * self.hidden_size * bf16;
        let stream_noise_local = self.scratch.stream_buf.offset(noise_byte_offset_local);
        let norm_noise_local = self.scratch.norm_buf.offset(noise_byte_offset_local);

        // 2026-09-25: The per-layer block dump arms under the same position gate as the
        // tail's logits and input dumps (`block_dump_armed_at`), once per model, so all
        // three come from the same propose.
        let block_dump_armed = {
            let want = levers.block_dump_armed_at(position);
            want && ctx.stats.dumped.keyed("dflash_per_layer")
        };
        let make_paged_args =
            |layer_idx: usize| -> Option<super::forward_block_layer_paged::PagedLayerArgs> {
                graphs::paged_layer_args(
                    &d,
                    layer_idx,
                    slot_mapping_gamma_opt,
                    stream,
                    block_dump_armed,
                )
            };

        // 2026-09-25: The non-paged layer body, always run eagerly.
        let run_legacy_layer = |layer_idx: usize, layer: &super::DflashLayer| -> Result<()> {
            let args = super::forward_block_layer::LayerArgs {
                layer_idx,
                n_attn: n_attn_local,
                eff_ctx: eff_ctx_local,
                h: h_local,
                q_dim: q_dim_local,
                kv_dim: kv_dim_local,
                inter: inter_local,
                bf16: bf16_local,
                inv_sqrt_d: inv_sqrt_d_local,
                stream,
            };
            self.forward_block_layer(layer, &args, ctx, debug_dump)
        };

        // 2026-09-25: Tail: final RMSNorm of the block rows, lm_head, token selection
        // (DFlash2 selector, DSpark Markov chain, or per-row argmax), then the one-shot
        // block dumps. Captured as the last subgraph (slot `num_layers * 2`).
        let run_tail = || -> Result<()> {
            self.block_tail_select(&d, ctx, stream, stream_noise_local, norm_noise_local)?;

            // 2026-09-25: METRALE_DFLASH_BLOCK_DUMP=1, once per model, at the first propose
            // with position >= METRALE_DFLASH_BLOCK_DUMP_AT_POS: write the first `width`
            // rows of `scratch.logits` (BF16, as token selection left them: the DFlash2
            // top-16 and the DSpark Markov bias both modify them) to
            // /tmp/metrale_block_logits.bin, and those rows' tokens with the shapes to
            // /tmp/metrale_block_drafts.json.
            {
                if levers.block_dump_armed_at(position)
                    // 2026-09-25: The latch is spent at this check, so a dump that
                    // fails partway is not retried.
                    && ctx.stats.dumped.keyed("dflash_block_logits")
                {
                    gpu.synchronize(stream)?;
                    let n_logits_bytes = width * self.vocab_size * bf16_local;
                    let mut lbuf = vec![0u8; n_logits_bytes];
                    if let Err(e) = gpu.copy_d2h(self.scratch.logits, &mut lbuf) {
                        tracing::warn!("DFLASH BLOCK_DUMP: logits copy failed: {e}");
                    } else if let Err(e) = std::fs::write("/tmp/metrale_block_logits.bin", &lbuf) {
                        tracing::warn!("DFLASH BLOCK_DUMP: logits write failed: {e}");
                    } else {
                        let (drafts, meta) = self.block_dump_drafts_meta(&d)?;
                        let _ = std::fs::write("/tmp/metrale_block_drafts.json", meta);
                        tracing::info!(
                            "DFLASH BLOCK_DUMP: wrote {} γ×vocab logit bytes + drafts={:?} (last_token={}, position={}) to /tmp/metrale_block_*.{{bin,json}}",
                            n_logits_bytes,
                            drafts,
                            last_token,
                            position,
                        );
                    }
                }
            }

            // 2026-09-25: Same gate, its own latch: write sequence 0's `width` block rows of
            // `stream_buf` to /tmp/metrale_block_noise_embed.bin, and the attention's
            // kv_len, q_offset and q_rope_pos with the block's query positions to
            // /tmp/metrale_block_input_meta.json. At this point `stream_buf` holds the last
            // layer's output, not the embedding.
            {
                if levers.block_dump_armed_at(position)
                    && ctx.stats.dumped.keyed("dflash_block_inputs")
                {
                    gpu.synchronize(stream)?;
                    let noise_off = eff_ctx * self.hidden_size * bf16_local;
                    let n_noise_bytes = width * self.hidden_size * bf16_local;
                    let mut nbuf = vec![0u8; n_noise_bytes];
                    if let Err(e) =
                        gpu.copy_d2h(self.scratch.stream_buf.offset(noise_off), &mut nbuf)
                    {
                        tracing::warn!("DFLASH BLOCK_INPUT: noise embed copy failed: {e}");
                    } else {
                        let _ = std::fs::write("/tmp/metrale_block_noise_embed.bin", &nbuf);
                    }
                    let (kv_len_dump, q_offset_dump, input_meta) = self.block_input_meta(&d);
                    let _ = std::fs::write("/tmp/metrale_block_input_meta.json", input_meta);
                    tracing::info!(
                        "DFLASH BLOCK_INPUT: wrote noise_embed ({}×{} BF16) + input_meta (q_offset={}, kv_len={}, position={})",
                        width,
                        self.hidden_size,
                        q_offset_dump,
                        kv_len_dump,
                        position,
                    );
                }
            }
            Ok(())
        };

        // 2026-09-25: Everything eagerly, no capture: warm-up passes and forwards that
        // cannot be captured.
        let run_all_eager = || -> Result<()> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if option_b_on {
                    let args = make_paged_args(layer_idx).expect("option_b args available");
                    let (k_pool, v_pool) = self.forward_block_layer_pre_attn(layer, &args, ctx)?;
                    self.forward_block_layer_attention(&args, ctx, k_pool, v_pool)?;
                    self.forward_block_layer_post_attn(layer, &args, ctx)?;
                } else {
                    run_legacy_layer(layer_idx, layer)?;
                }
            }
            run_tail()
        };

        if graph_eligible && option_b_on {
            // 2026-09-25: Subgraph slots: [pre_0, post_0, ..., pre_{N-1}, post_{N-1}, tail],
            // 2 × num_layers + 1 in all.
            let num_layers = self.layers.len();
            let total_slots = num_layers * 2 + 1;
            let tail_slot = num_layers * 2;

            // 2026-09-25: Graphs are keyed by the block width in flight, so each width
            // warms up and captures once and a later switch back to it only replays.
            let mut g = self.propose_graphs.lock();
            let cached_ready = matches!(g.by_width.get(&width), Some(v) if v.len() == total_slots);

            if cached_ready {
                // 2026-09-25: Replay each cached subgraph in order, with attention run
                // eagerly between each layer's pre and post.
                let graphs = g.by_width.get(&width).unwrap();
                self.replay_block_graphs(
                    ctx,
                    stream,
                    graphs,
                    tail_slot,
                    &make_paged_args,
                    &run_tail,
                )?;
            } else {
                let warmed = g.warmup.get(&width).copied().unwrap_or(0);
                if warmed < warmup_target {
                    // 2026-09-25: Warm-up pass for this width: eager, no capture.
                    g.warmup.insert(width, warmed + 1);
                    run_all_eager()?;
                } else {
                    // 2026-09-25: Capture pass: capture every subgraph in this propose,
                    // launching each right after its capture. A capture that comes back
                    // as `GraphHandle(0)` is stored as zero and that part runs eagerly.
                    tracing::info!(
                        "DFlash piecewise capture: starting (warmup_count={}, target={}, slots={}, block width {width})",
                        warmed,
                        warmup_target,
                        total_slots
                    );
                    let mut new_graphs: Vec<metrale_gpu_runtime::gpu::GraphHandle> =
                        Vec::with_capacity(total_slots);

                    for (layer_idx, layer) in self.layers.iter().enumerate() {
                        let args = make_paged_args(layer_idx).expect("option_b args available");

                        gpu.begin_capture(stream)?;
                        let _captured = self.forward_block_layer_pre_attn(layer, &args, ctx)?;
                        let pre_graph = gpu.end_capture(stream)?;
                        new_graphs.push(pre_graph);
                        if pre_graph.0 != 0 {
                            gpu.launch_graph(pre_graph, stream)?;
                        } else {
                            tracing::warn!(
                                "DFlash piecewise: pre_attn layer {} empty capture — eager fallback",
                                layer_idx
                            );
                            self.forward_block_layer_pre_attn(layer, &args, ctx)?;
                        }

                        let (k_pool, v_pool) = {
                            let cache = self.kv_cache.lock();
                            (cache.k_pool_ptr(layer_idx), cache.v_pool_ptr(layer_idx))
                        };
                        self.forward_block_layer_attention(&args, ctx, k_pool, v_pool)?;

                        gpu.begin_capture(stream)?;
                        self.forward_block_layer_post_attn(layer, &args, ctx)?;
                        let post_graph = gpu.end_capture(stream)?;
                        new_graphs.push(post_graph);
                        if post_graph.0 != 0 {
                            gpu.launch_graph(post_graph, stream)?;
                        } else {
                            tracing::warn!(
                                "DFlash piecewise: post_attn layer {} empty capture — eager fallback",
                                layer_idx
                            );
                            self.forward_block_layer_post_attn(layer, &args, ctx)?;
                        }
                    }

                    gpu.begin_capture(stream)?;
                    run_tail()?;
                    let tail_graph = gpu.end_capture(stream)?;
                    new_graphs.push(tail_graph);
                    if tail_graph.0 != 0 {
                        gpu.launch_graph(tail_graph, stream)?;
                    } else {
                        tracing::warn!("DFlash piecewise: tail empty capture — eager fallback");
                        run_tail()?;
                    }

                    let success_count = new_graphs.iter().filter(|g| g.0 != 0).count();
                    tracing::info!(
                        "DFlash piecewise capture: complete ({}/{} subgraphs captured, block width {})",
                        success_count,
                        total_slots,
                        width
                    );
                    g.by_width.insert(width, new_graphs);
                }
            }
        } else {
            run_all_eager()?;
        }

        let mut drafts = self.copy_drafts_out(&d, stream)?;
        // 2026-09-25: DSpark confidence truncation: keep the drafts before the first row
        // j whose confidence sigmoid is below METRALE_DSPARK_CONF_TAU, or all of them.
        // With the shift above, row j's confidence gates the draft propose returns at
        // index j. Needs the anchor bias: `markov_argmax_block` writes a row's
        // confidence only for biased rows, and row 0 is biased only with it.
        if self.markov_active() && self.confidence_active() && levers.dspark_anchor_bias {
            let tau = levers.conf_tau;
            let mut cbuf = vec![0u8; width * 2];
            gpu.copy_d2h(self.scratch.conf_out, &mut cbuf)?;
            if levers.dspark_conf_trace {
                let logits: Vec<f32> = (0..width)
                    .map(|j| {
                        let bits = u16::from_le_bytes([cbuf[j * 2], cbuf[j * 2 + 1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                tracing::info!(
                    "DSPARK CONF logits (pos={position}, tau={tau}): {:?} sigmoids: {:?}",
                    logits,
                    logits
                        .iter()
                        .map(|x| 1.0 / (1.0 + (-x).exp()))
                        .collect::<Vec<f32>>(),
                );
            }
            host_drafts::conf_truncate(&mut drafts, &cbuf, width, tau);
        }
        // 2026-09-25: METRALE_DFLASH_DEBUG_DUMP_FULL=1 or METRALE_DFLASH_LOG_DRAFTS=1,
        // once per model: log the returned drafts.
        if ctx.stats.dumped.keyed("dflash_drafts") && (levers.debug_dump_full || levers.log_drafts)
        {
            tracing::info!(
                "DFLASH DUMP_FULL drafts (γ={}, last_token={}, position={}, eff_ctx={}): {:?}",
                width,
                last_token,
                position,
                eff_ctx,
                drafts,
            );
        }
        let _ = g;
        Ok(drafts)
    }
}

#[cfg(test)]
mod tests;
