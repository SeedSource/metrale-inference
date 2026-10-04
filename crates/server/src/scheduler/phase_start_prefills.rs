// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Start the tick's new requests: chunked prefill
//! (`start_chunked_prefill`) or non-chunked prefill (`prefill_request`),
//! after the optional batched image encode and fused beam search. A request
//! may land in `active`, land in `prefilling`, finish at once, or fail.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use metrale_model_engine::traits::Model;

use super::*;
use crate::api::InferenceRequest;
use crate::grammar::GrammarEngine;

#[allow(clippy::too_many_arguments)]
pub(super) fn start_new_requests(
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    new_reqs: Vec<InferenceRequest>,
    chunked: bool,
    always_mixed: bool,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    eos_tokens: &[u32],
    prefill_stream: u64,
    prefill_event: u64,
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut Vec<PrefillInProgress>,
) {
    // 2026-09-25: `--prefill-codispatch`: on a non-EP model, with two or
    // more new requests, no image among them, and nothing active or
    // prefilling, defer every chunk 0 so `continue_in_progress_prefills`
    // can run them as one batched prefill.
    let want_codispatch = sched.levers.prefill_codispatch
        && chunked
        && new_reqs.len() >= 2
        && active.is_empty()
        && prefilling.is_empty()
        && !model.is_ep()
        && !new_reqs.iter().any(|r| r.has_image_pixels());
    // 2026-09-25: `--prefill-varlen-batch`: on a non-EP model with nothing
    // active, defer chunk 0 of each request without images when there are
    // two or more new requests or a prefill is already in flight. A lone
    // request with nothing in flight keeps its inline chunk 0 and the
    // `max_batch_tokens` budget.
    let want_varlen_defer = chunked
        && !model.is_ep()
        && active.is_empty()
        && (new_reqs.len() >= 2 || !prefilling.is_empty())
        && sched.levers.prefill_varlen;
    // 2026-09-25: `METRALE_HOLO_ALWAYS_MIXED` on a non-EP model with decodes
    // active: defer chunk 0 of each request without images, so the prefill
    // runs in `continue_in_progress_prefills` alongside the decodes instead
    // of an inline prefill that stalls them. `want_codispatch` needs
    // `active` empty, so the two never both apply.
    let mixed_defer = always_mixed && chunked && !active.is_empty() && !model.is_ep();

    // 2026-09-25: `METRALE_VISION_CODISPATCH` (default off): encode the
    // images of this tick's image requests within `max_prefill_tokens` in
    // one `prepare_vision_embed_batched` call, when at least two qualify.
    // Each request then reads its slice of the shared encoder output (its
    // `VisionSlice`).
    let vision_codispatch = sched.levers.vision_codispatch;
    const VISION_P_MAX: usize = 6400;
    let mut vision_slices: Vec<VisionSlice> = vec![VisionSlice::default(); new_reqs.len()];
    if vision_codispatch && chunked {
        let mut batched_idx: Vec<usize> = Vec::new();
        let mut per_request_imgs: Vec<Vec<metrale_model_layers::VisionItem>> = Vec::new();
        let mut running_patches = 0usize;
        let mut overflow = false;
        for (k, req) in new_reqs.iter().enumerate() {
            if !req.has_image_pixels() {
                continue;
            }
            if req.prompt_len() > max_prefill_tokens {
                continue;
            }
            let imgs = req.image_pixels_ref();
            // 2026-09-25: Count the grid once per temporal group (`t_len`),
            // or a video would be under-counted.
            let req_patches: usize = imgs
                .iter()
                .map(|it| it.t_len() * it.grid_h * it.grid_w)
                .sum();
            if running_patches + req_patches > VISION_P_MAX {
                overflow = true;
                break;
            }
            running_patches += req_patches;
            batched_idx.push(k);
            per_request_imgs.push(imgs.to_vec());
        }
        if overflow {
            // 2026-09-25: On overflow nothing is batch-encoded this tick, so
            // batched and per-request encodes never share the encoder
            // output within one tick.
            batched_idx.clear();
            per_request_imgs.clear();
        }
        if batched_idx.len() >= 2 {
            match model.prepare_vision_embed_batched(&per_request_imgs) {
                Ok(descs) if descs.len() == batched_idx.len() => {
                    for (slot, (row_off, grid_off, n_img, row_cnt)) in
                        batched_idx.iter().zip(descs.into_iter())
                    {
                        vision_slices[*slot] = VisionSlice {
                            patch_row_offset: row_off,
                            grid_index_offset: grid_off,
                            num_images: n_img,
                            patch_row_count: row_cnt,
                        };
                    }
                    // 2026-09-25: One fence for the batch: `prefill_stream`
                    // waits for the default stream before any chunk-0 prefill.
                    if let Err(e) = model
                        .record_event(prefill_event, model.default_stream())
                        .and_then(|_| model.stream_wait_event(prefill_stream, prefill_event))
                    {
                        tracing::error!("vision co-dispatch fence failed: {e:#}");
                    }
                    tracing::info!(
                        "Vision co-dispatch: batched {} image requests this tick",
                        batched_idx.len()
                    );
                }
                Ok(_) => {
                    tracing::warn!(
                        "vision co-dispatch desc count mismatch; per-request fallback this tick"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "vision co-dispatch batched encode failed: {e:#}; per-request fallback"
                    );
                }
            }
        }
    }

    // 2026-09-25: `METRALE_BEAM_CODISPATCH` (default on; `0`/`false` turns
    // it off): the tick's blocking beam requests (`num_beams > 1`), grouped
    // by adapter and packed up to `BEAM_C_CAP` beams per chunk, run as one
    // `generate_beam_batch` call per chunk of two or more. Each hypothesis
    // goes to `beam_hyps[req_idx]` for the per-request beam branch; a chunk
    // of one, or a failed batch, leaves the search to that branch.
    let mut beam_hyps: Vec<Option<Vec<u32>>> = (0..new_reqs.len()).map(|_| None).collect();
    let beam_codispatch = model.supports_beam() && sched.levers.beam_codispatch;
    if beam_codispatch {
        // 2026-09-25: Most beams in one fused batch.
        const BEAM_C_CAP: usize = 320;
        let items: Vec<(usize, i32, usize)> = new_reqs
            .iter()
            .enumerate()
            .filter(|(_, r)| r.num_beams() > 1 && matches!(r, InferenceRequest::Blocking { .. }))
            .map(|(idx, r)| (idx, r.adapter_slot(), r.num_beams() as usize))
            .collect();
        for chunk in pack_beam_chunks(&items, BEAM_C_CAP) {
            if chunk.len() < 2 {
                continue;
            }
            let reqs: Vec<metrale_model_engine::traits::BeamReq> = chunk
                .iter()
                .map(|&i| {
                    let r = &new_reqs[i];
                    metrale_model_engine::traits::BeamReq {
                        prompt_tokens: r.prompt_tokens_arc().as_ref().clone(),
                        src_lang_id: r.src_lang_id(),
                        tgt_lang_id: r.tgt_lang_id(),
                        adapter_slot: r.adapter_slot(),
                        num_beams: r.num_beams() as usize,
                        max_new: r.max_tokens(),
                        length_penalty: r.length_penalty(),
                        early_stopping: r.early_stopping(),
                    }
                })
                .collect();
            let total_beams: usize = reqs.iter().map(|r| r.num_beams).sum();
            match model.generate_beam_batch(&reqs) {
                Ok(hyps) if hyps.len() == chunk.len() => {
                    for (&i, h) in chunk.iter().zip(hyps.into_iter()) {
                        beam_hyps[i] = Some(h);
                    }
                    tracing::info!(
                        "Beam co-dispatch: fused {} requests ({total_beams} beams) this tick",
                        chunk.len(),
                    );
                }
                Ok(_) => {
                    tracing::warn!("beam co-dispatch count mismatch; per-request fallback")
                }
                Err(e) => {
                    tracing::warn!("beam co-dispatch failed: {e:#}; per-request fallback")
                }
            }
        }
    }

    for (req_idx, req) in new_reqs.into_iter().enumerate() {
        let precomputed_beam_hyp = beam_hyps[req_idx].take();
        if chunked {
            let defer =
                want_codispatch || ((mixed_defer || want_varlen_defer) && !req.has_image_pixels());
            // 2026-09-25: `num_images > 0`: batch-encoded above.
            let slice = vision_slices[req_idx];
            let vision_slice = if slice.num_images > 0 {
                Some(slice)
            } else {
                None
            };
            // 2026-09-25: With nothing active or prefilling, chunk 0 may use
            // `max_batch_tokens`.
            let budget = if active.is_empty() && prefilling.is_empty() {
                max_batch_tokens
            } else {
                max_prefill_tokens
            };
            // 2026-10-03: `METRALE_PREFILL_CHUNK_WHILE_DECODING`: while a sequence decodes,
            // chunk 0 is planned from the capped budget (`prefill_chunk_cap`). Not for MLA
            // (one chunk) or image prompts: the embed splice (`prefill_b/embed_chunk.rs`) restarts
            // its row index every chunk, so image pads must stay inside chunk 0 as uncapped.
            let decoders = super::prefill_chunk_cap::decoders(active);
            let cap = if model.is_mla() || req.has_image_pixels() {
                None
            } else {
                super::prefill_chunk_cap::active_cap(
                    sched.levers.prefill_chunk_while_decoding,
                    decoders,
                    model.kv_block_size(),
                )
            };
            let uncapped_budget = budget;
            let budget = cap.map_or(budget, |c| budget.min(c));
            match start_chunked_prefill(
                sched,
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                model,
                req,
                eos_tokens,
                budget,
                prefill_stream,
                prefill_event,
                grammar_engine,
                spontaneous_think_budget,
                defer,
                vision_slice,
                precomputed_beam_hyp,
            ) {
                Ok(StartPrefillResult::Active(a)) => {
                    tracing::info!(
                        "Prefilled (single chunk): seq_len={}, remaining={}",
                        a.seq.seq_len,
                        a.remaining,
                    );
                    active.push(a);
                }
                Ok(StartPrefillResult::InProgress(mut p)) => {
                    // 2026-10-03: Chunk 0 ran shorter than its uncapped plan: the cap fired.
                    if budget < uncapped_budget && p.chunk_offset > 0 {
                        let total = p.prompt_tokens.len();
                        let (_, uncapped) = super::prefill_chunk_cap::plan_capped(
                            0,
                            total,
                            total.min(uncapped_budget),
                            model.kv_block_size(),
                            model.prefill_tail_split(&p.prompt_tokens),
                            model.prefill_grid(&p.prompt_tokens),
                            None,
                        );
                        if p.chunk_offset < uncapped {
                            super::prefill_chunk_cap::log_capped(
                                &mut p.chunk_cap_logged,
                                p.chunk_offset,
                                decoders,
                            );
                        }
                    }
                    tracing::info!(
                        "Prefill chunk 0/{}: {}/{} tokens",
                        p.prompt_tokens.len(),
                        p.chunk_offset,
                        p.prompt_tokens.len(),
                    );
                    prefilling.push(p);
                }
                Ok(StartPrefillResult::Finished) => {}
                Err(e) => {
                    handle_prefill_start_error(&sched.io, &e, active);
                }
            }
        } else {
            match prefill_request(
                sched,
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                model,
                req,
                eos_tokens,
                grammar_engine,
                spontaneous_think_budget,
                precomputed_beam_hyp,
            ) {
                Ok(Some(a)) => {
                    tracing::info!(
                        "Prefilled: seq_len={}, remaining={}",
                        a.seq.seq_len,
                        a.remaining,
                    );
                    active.push(a);
                }
                Ok(None) => {}
                Err(e) => {
                    handle_prefill_start_error(&sched.io, &e, active);
                }
            }
        }
    }
}

/// 2026-09-25: Group beam requests by adapter, then pack each group, in
/// order, into chunks whose summed beam count does not exceed `cap` (a
/// single request larger than `cap` gets a chunk of its own). `items` is
/// `(req_idx, adapter_slot, num_beams)`; returns chunks of `req_idx`,
/// adapters in ascending order.
fn pack_beam_chunks(items: &[(usize, i32, usize)], cap: usize) -> Vec<Vec<usize>> {
    let mut groups: std::collections::BTreeMap<i32, Vec<(usize, usize)>> =
        std::collections::BTreeMap::new();
    for &(idx, adapter, nb) in items {
        groups.entry(adapter).or_default().push((idx, nb));
    }
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    for reqs in groups.into_values() {
        let (mut chunk, mut beams) = (Vec::new(), 0usize);
        for (idx, nb) in reqs {
            if !chunk.is_empty() && beams + nb > cap {
                chunks.push(std::mem::take(&mut chunk));
                beams = 0;
            }
            chunk.push(idx);
            beams += nb;
        }
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
    }
    chunks
}

/// 2026-09-25: After a failed prefill start: if the error says "pool
/// exhausted" and a sequence is active, end the last active sequence with
/// an error to free its resources. Otherwise log the error. The failed
/// request was already sent its own error.
fn handle_prefill_start_error(
    io: &crate::scheduler::io::SchedIo,
    e: &anyhow::Error,
    active: &mut Vec<ActiveSeq>,
) {
    let err_msg = format!("{e:#}");
    if err_msg.contains("pool exhausted") && !active.is_empty() {
        let victim_idx = active.len() - 1;
        let mut victim = active.swap_remove(victim_idx);
        tracing::warn!(
            "SSM pool full: preempting seq (slot={}, tokens={}) for new request",
            victim.seq.slot_idx,
            victim.output_tokens.len(),
        );
        send_error(io, &mut victim, "Preempted: server resource pressure");
    } else {
        tracing::error!("Prefill start error: {err_msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::pack_beam_chunks;

    #[test]
    fn groups_by_adapter_then_packs_by_cap() {
        // 2026-09-25: two adapters (-1, 7), five beams each request; cap=15
        // allows at most 3 requests per chunk.
        let items = [
            (0, -1, 5),
            (1, 7, 5),
            (2, -1, 5),
            (3, 7, 5),
            (4, -1, 5),
            (5, 7, 5),
            (6, -1, 5),
            (7, 7, 5),
        ];
        let chunks = pack_beam_chunks(&items, 15);
        assert_eq!(chunks, vec![vec![0, 2, 4], vec![6], vec![1, 3, 5], vec![7]]);
        for c in &chunks {
            let beams: usize = c.iter().map(|&i| items[i].2).sum();
            assert!(beams <= 15, "chunk {c:?} exceeds cap");
        }
    }

    #[test]
    fn single_adapter_all_fit_one_chunk() {
        let items = [(0, -1, 5), (1, -1, 5), (2, -1, 5)];
        assert_eq!(pack_beam_chunks(&items, 320), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn heterogeneous_beam_counts_respect_cap() {
        // 2026-09-25: 8+8+4 with cap 16 packs [8,8] then [4]; adding the 4
        // to the first chunk would reach 20 > 16.
        let items = [(0, -1, 8), (1, -1, 8), (2, -1, 4)];
        assert_eq!(pack_beam_chunks(&items, 16), vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn empty_items_no_chunks() {
        assert!(pack_beam_chunks(&[], 320).is_empty());
    }
}
