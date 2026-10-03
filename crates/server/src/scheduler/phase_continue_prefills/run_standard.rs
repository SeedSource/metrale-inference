// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-stream prefill step: one chunk of one prefilling
//! stream, fused with the active decode through `mixed_forward` when
//! `can_mix` allows it, otherwise a plain `prefill_chunk`.
//!
//! Owner: scheduler.
//! Invariants:
//! - A failed EP broadcast, forward or first-token sample pushes
//!   `(idx, None)` to `completed_indices`.

use anyhow::Result;
use metrale_model_engine::traits::{Model, SequenceState};

use super::super::decode_logits_step::process_decode_logits;
use super::super::lifecycle::send_error;
use super::super::types::{ActiveSeq, PrefillInProgress};
use super::super::{FirstTokenPolicy, sample_first_token};

#[allow(clippy::too_many_arguments)]
pub(super) fn run_standard_chunk_loop(
    model: &dyn Model,
    p: &mut PrefillInProgress,
    idx: usize,
    active: &mut Vec<ActiveSeq>,
    max_prefill_tokens: usize,
    slice_budget: usize,
    prefill_stream: u64,
    prefill_event: u64,
    use_mtp: bool,
    use_self_speculative: bool,
    use_ngram_speculative: bool,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    did_mixed_step: &mut bool,
) {
    // 2026-09-25: InnerQ calibration poll (`poll_innerq`).
    super::poll_innerq(model);
    // 2026-09-25: One chunk per call; a later tick advances the next one.
    let remaining = p.prompt_tokens.len() - p.chunk_offset;
    // 2026-09-25: MLA prefill attends only over the K/V of the tokens in the
    // current call, not the paged cache (`paged_mla`, see
    // `mla_prefill_needs_full_recompute`), so an MLA prompt must not be
    // split: its chunk is the whole remaining prompt.
    let effective_max = if model.is_mla() {
        remaining
    } else {
        max_prefill_tokens
    };
    // 2026-09-25: Cap the chunk at the slice budget. Without
    // `METRALE_HOLO_ALWAYS_MIXED` the caller passes `max_prefill_tokens`, so
    // the cap changes nothing. MLA ignores the slice budget.
    let cap = if model.is_mla() {
        effective_max
    } else {
        effective_max.min(slice_budget)
    };
    let mut chunk_len = remaining.min(cap);
    // 2026-09-25: With `METRALE_SSM_TAIL_CKPT=1` and mid-chunk capture off,
    // end a chunk at `ssm_tail_boundary`, so an SSM snapshot lands where the
    // next turn's prefix match stops (see `ssm_tail_boundary`, gpu-runtime).
    if sched.levers.ssm_tail_ckpt
        && !sched.levers.ssm_tail_midchunk
        && let Some(bs) = model.kv_block_size()
        && let Some(tb) = metrale_gpu_runtime::ssm_tail_boundary(p.prompt_tokens.len(), bs)
        && p.chunk_offset < tb
        && p.chunk_offset + chunk_len > tb
    {
        chunk_len = tb - p.chunk_offset;
    }
    // 2026-09-27: A non-last chunk ends on a KV block boundary, or at the model's
    // tail split point when it would span it, so its SSM snapshot is restorable
    // (`prefill_plan::plan_chunk_len`).
    let chunk_len = metrale_model_engine::prefill_plan::plan_chunk_len_grid(
        p.chunk_offset,
        p.prompt_tokens.len(),
        chunk_len,
        model.kv_block_size(),
        model.prefill_tail_split(&p.prompt_tokens),
        model.prefill_grid(&p.prompt_tokens),
    );
    let is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();

    // 2026-09-25: `METRALE_BISECT_NO_MIX=1` turns the fused `mixed_forward`
    // path off.
    let no_mix_bisect = sched.levers.bisect_no_mix;
    // 2026-09-25: The speculation check depends on this tick's width
    // (`spec_mixing::mixing_blocked_by_spec`), so a speculative serve still
    // fuses when more than one sequence is active. `spec_mixing` documents
    // the widths where this disagrees with the MTP dispatch gate.
    let any_spec = use_mtp || use_self_speculative || use_ngram_speculative;
    let spec_step_this_tick = super::spec_mixing::mixing_blocked_by_spec(active.len(), any_spec);
    let can_mix = !no_mix_bisect && !active.is_empty() && !model.is_ep() && !spec_step_this_tick;

    if can_mix {
        // 2026-09-25: Under a speculative serve, drop every sequence's
        // pending drafts and confidences before the plain mixed decode, and
        // when any were pending, wait for the secondary stream (the last
        // verify's async state copies) with `sync_secondary`.
        if any_spec {
            let had_drafts = active.iter().any(|a| !a.pending_drafts.is_empty());
            for a in active.iter_mut() {
                a.pending_drafts.clear();
                a.pending_draft_conf.clear();
            }
            if had_drafts && let Err(e) = model.sync_secondary() {
                tracing::error!("mtp->mixed sync_secondary: {e:#}");
            }
        }
        // 2026-09-25: Sort by ssm pool slot, as `decode_step.rs` does: the
        // batched GDN recurrence needs the sequences on consecutive pool
        // slots in slice order. Moving whole `ActiveSeq`s keeps each decode
        // row matched to its sequence.
        if active.len() > 1 {
            active.sort_by_key(|a| a.seq.ssm_slot_idx().unwrap_or(a.seq.slot_idx));
        }
        let decode_tokens: Vec<u32> = active.iter().map(|a| a.last_token).collect();
        let mut decode_refs: Vec<&mut SequenceState> =
            active.iter_mut().map(|a| &mut a.seq).collect();
        let t0_mixed = sched.io.clock.now();

        match model.mixed_forward(
            &decode_tokens,
            &mut decode_refs,
            &p.prompt_tokens,
            &mut p.seq,
            p.chunk_offset,
            chunk_len,
            is_last,
            prefill_stream,
        ) {
            Ok(result) => {
                p.chunk_offset += chunk_len;
                tracing::info!(
                    "Mixed forward: prefill {}/{} tokens + {} decode",
                    p.chunk_offset,
                    p.prompt_tokens.len(),
                    decode_tokens.len(),
                );

                // 2026-09-25: Last chunk: sample the prefill's first token.
                if is_last {
                    // 2026-09-25: No SSM normalize here: `mixed_forward`
                    // normalizes the prefill's SSM state on the default
                    // stream after every chunk, this last one included
                    // (trait_impl/decode_b.rs), and a normalize on
                    // `prefill_stream` would not be ordered after it.
                    let _ = model.record_event(prefill_event, prefill_stream);
                    let _ = model.stream_wait_event(model.default_stream(), prefill_event);
                    // 2026-09-25: `sample_first_token` applies the
                    // sequence's `min_p` (0.0 under `METRALE_NO_MTP_MINP=1`)
                    // and, when `FirstTokenPolicy` lets it act on token 0,
                    // the grammar.
                    match sample_first_token(
                        model,
                        result.prefill_logits,
                        p.temperature,
                        p.top_k,
                        p.top_p,
                        p.min_p,
                        &p.eos_tokens,
                        p.grammar_state.as_mut(),
                        FirstTokenPolicy::for_birth(
                            p.enable_thinking,
                            think_end_token,
                            tool_call_start_token,
                        ),
                        &sched.levers.sampling(),
                        sched.io.tel.dumps(),
                    ) {
                        Ok(first) => {
                            tracing::info!("Mixed prefill first token: {first}");
                            completed_indices.push((idx, Some(first)));
                        }
                        Err(e) => {
                            tracing::error!("Mixed prefill sampling: {e:#}");
                            completed_indices.push((idx, None));
                        }
                    }
                }

                // 2026-09-25: Decode logits for the active sequences.
                let _ = model.record_event(prefill_event, prefill_stream);
                let _ = model.stream_wait_event(model.default_stream(), prefill_event);
                let mut staging = sched.scratch.host_bytes.borrow_mut().split_off(0);
                if let Some(step) = super::super::decode_logits_step::decode_readback_or_fail(
                    active,
                    result.decode_logits,
                    sched,
                    &mut staging,
                ) {
                    process_decode_logits(
                        active,
                        step,
                        &mut staging,
                        t0_mixed,
                        think_end_token,
                        think_start_token,
                        code_fence_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        adaptive_sampling,
                        sched,
                    );
                }
                *did_mixed_step = true;
            }
            Err(e) => {
                tracing::error!("Mixed forward error: {e:#}");
                completed_indices.push((idx, None));
            }
        }
        return;
    }

    // 2026-09-25: Plain prefill chunk; decode runs separately. Under EP the
    // chunk is first broadcast to the worker.
    let ep_ok = (|| -> Result<()> {
        model.ep_broadcast_cmd_for_seq(p.seq.slot_idx as u32, 0xFFFFFFF0)?;
        model.ep_broadcast_cmd(chunk_len as u32)?;
        model.ep_broadcast_cmd(p.chunk_offset as u32)?;
        model.ep_broadcast_cmd(p.prompt_tokens.len() as u32)?;
        model.ep_broadcast_tokens(&p.prompt_tokens)?;
        // 2026-09-25: Every `0xFFFFFFF0` broadcast must be followed by this
        // call: the worker runs the matching collective in its `0xFFFFFFF0`
        // handler (`Model::ep_sync_vision_embeds`).
        model.ep_sync_vision_embeds(&p.prompt_tokens)?;
        Ok(())
    })();
    if let Err(e) = ep_ok {
        tracing::error!("EP broadcast chunk: {e:#}");
        completed_indices.push((idx, None));
        return;
    }

    // 2026-09-25: While `prefill_chunk` fails with "KV cache exhausted",
    // end the active sequence without a grammar that holds the most KV
    // blocks with a "preempted" error (`send_error`), then retry. With no
    // such sequence left, the error stands and this prefill fails.
    let mut chunk_res = model.prefill_chunk(
        &p.prompt_tokens,
        &mut p.seq,
        p.chunk_offset,
        chunk_len,
        is_last,
        prefill_stream,
    );
    while chunk_res
        .as_ref()
        .err()
        .is_some_and(|e| format!("{e:#}").contains("KV cache exhausted"))
    {
        let Some(vi) = active
            .iter()
            .enumerate()
            .filter(|(_, a)| a.grammar_state.is_none())
            .max_by_key(|(_, a)| a.seq.block_table.len())
            .map(|(i, _)| i)
        else {
            break;
        };
        let mut victim = active.remove(vi);
        tracing::warn!(
            "KV cache exhausted during prefill chunk: preempting slot={} ({} blocks) \
             so this prefill can proceed ({} sequence(s) remain)",
            victim.seq.slot_idx,
            victim.seq.block_table.len(),
            active.len(),
        );
        send_error(
            &sched.io,
            &mut victim,
            "preempted: KV cache exhausted (a prefill needed its blocks)",
        );
        chunk_res = model.prefill_chunk(
            &p.prompt_tokens,
            &mut p.seq,
            p.chunk_offset,
            chunk_len,
            is_last,
            prefill_stream,
        );
    }
    match chunk_res {
        Ok(logits) => {
            p.chunk_offset += chunk_len;
            tracing::info!(
                "Prefill chunk {}/{} tokens",
                p.chunk_offset,
                p.prompt_tokens.len(),
            );
            // 2026-09-25: Normalize the SSM state after every plain chunk
            // (`Model::normalize_ssm_states` bounds h_state norms over a long
            // chunked prefill); a failure is logged.
            if let Err(e) = model.normalize_ssm_states(&p.seq, prefill_stream) {
                tracing::warn!("SSM state normalization failed: {e:#}");
            }
            if is_last {
                let _ = model.record_event(prefill_event, prefill_stream);
                let _ = model.stream_wait_event(model.default_stream(), prefill_event);
                // 2026-09-25: `sample_first_token` applies the sequence's
                // `min_p` (0.0 under `METRALE_NO_MTP_MINP=1`) and, when
                // `FirstTokenPolicy` lets it act on token 0, the grammar.
                match sample_first_token(
                    model,
                    logits,
                    p.temperature,
                    p.top_k,
                    p.top_p,
                    p.min_p,
                    &p.eos_tokens,
                    p.grammar_state.as_mut(),
                    FirstTokenPolicy::for_birth(
                        p.enable_thinking,
                        think_end_token,
                        tool_call_start_token,
                    ),
                    &sched.levers.sampling(),
                    sched.io.tel.dumps(),
                ) {
                    Ok(first) => {
                        tracing::info!("Prefill first token: {first}");
                        completed_indices.push((idx, Some(first)));
                    }
                    Err(e) => {
                        tracing::error!("Chunked prefill argmax: {e:#}");
                        completed_indices.push((idx, None));
                    }
                }
            }
        }
        Err(e) => {
            tracing::error!("Prefill chunk error: {e:#}");
            completed_indices.push((idx, None));
        }
    }
}
