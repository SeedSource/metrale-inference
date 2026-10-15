// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Decode-time KV preemption. When a batched decode launch
//! fails with "KV cache exhausted" and more than one sequence is active,
//! `decode_launch::launch_with_preemption` removes one victim and retries.
//! The victim is spilled with [`spill_out_sequence`] when a spill pool is
//! configured and the spill succeeds; otherwise it is requeued with
//! [`preempt_requeue`] and later re-prefilled by [`resume_preempted_seqs`].
//! No error frame is sent at preemption; a resume that fails sends one.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

#[cfg(test)]
pub(super) use super::decode_launch::choose_decode_victim;
use super::decode_launch::fail_readback;
pub(super) use super::decode_launch::{
    FeedReadback, PREEMPT_IMMUNITY_TOKENS, decode_ctx_commit, launch_with_preemption,
};
use super::*;
use crate::scheduler::io::{Effect, SchedIo, SpillIo, StepOutcome};

/// 2026-09-25: Launch and await one decode step for `active`, preempting
/// victims on KV exhaustion (see the module doc). Returns `None` when the
/// step failed and every remaining sequence in `active` has been sent an
/// error and removed.
pub(super) fn decode_batch_with_preemption(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    active: &mut Vec<ActiveSeq>,
    spill: Option<&dyn SpillIo>,
    swapped: &mut Vec<SwappedSeq>,
    preempted: &mut Vec<PreemptedSeq>,
    staging: &mut Vec<u8>,
) -> Option<StepOutcome> {
    let ticket = launch_with_preemption(
        sched,
        active,
        spill,
        swapped,
        preempted,
        staging,
        FeedReadback::Plain,
    )?;
    match sched.io.dev.await_result(ticket) {
        Ok(r) => Some(r.outcome),
        Err(e) => {
            fail_readback(&sched.io, active, &e.into_inner());
            None
        }
    }
}

/// 2026-09-25: Save a sequence already removed from `active` to the spill
/// pool, release its device sequence and build the [`SwappedSeq`]. Both
/// `swap_out_sequence` and decode-time preemption use it. If creating the
/// spill entry or saving fails, the `ActiveSeq` is handed back with the
/// error.
#[allow(clippy::result_large_err)]
pub(super) fn spill_out_sequence(
    io: &SchedIo,
    mut a: ActiveSeq,
    spill: &dyn SpillIo,
) -> Result<SwappedSeq, (ActiveSeq, anyhow::Error)> {
    let (swap_id, mut writer) = match spill.create() {
        Ok(v) => v,
        Err(e) => return Err((a, e)),
    };
    if let Err(e) = io.dev.apply(Effect::SaveSequenceState {
        seq: &a.seq,
        writer: &mut writer,
    }) {
        drop(writer);
        let _ = spill.remove(swap_id);
        return Err((a, e.into_inner()));
    }
    drop(writer);
    spill.record_usage(swap_id);

    let num_blocks = a.seq.block_table.len();
    let seq_len = a.seq.seq_len;
    let tokens = a.seq.tokens.clone();

    // 2026-09-25: The state is saved, so a release failure is not fatal;
    // `Effect::ReleaseSeq` logs its own errors.
    let _ = io.dev.apply(Effect::ReleaseSeq {
        seq: &mut a.seq,
        cache: false,
        what: "spill_out_sequence",
    });

    Ok(SwappedSeq {
        tokens,
        session_hash: a.session_hash,
        adapter_slot: a.seq.adapter_slot,
        adapter_id: a.seq.adapter_id,
        seq_len,
        num_blocks,
        last_token: a.last_token,
        output_tokens: a.output_tokens,
        remaining: a.remaining,
        min_tokens: a.min_tokens,
        eos_tokens: a.eos_tokens,
        sink: a.sink,
        temperature: a.temperature,
        top_k: a.top_k,
        top_p: a.top_p,
        top_n_sigma: a.top_n_sigma,
        min_p: a.min_p,
        repetition_penalty: a.repetition_penalty,
        presence_penalty: a.presence_penalty,
        frequency_penalty: a.frequency_penalty,
        repetition_penalty_window: 256,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: a.dry_multiplier,
        dry_base: a.dry_base,
        dry_allowed_length: a.dry_allowed_length,
        dry_sequence_breakers: a.dry_sequence_breakers,
        logit_bias: a.logit_bias,
        inside_thinking: a.inside_thinking,
        enable_thinking: a.enable_thinking,
        thinking_budget: a.thinking_budget,
        repetition_detection: a.repetition_detection,
        spontaneous_think_budget: a.spontaneous_think_budget,
        thinking_tokens: a.thinking_tokens,
        force_end_thinking: a.force_end_thinking,
        think_force_closed: a.think_force_closed,
        sentence_defer_count: a.sentence_defer_count,
        consecutive_confident: a.consecutive_confident,
        in_code_fence: a.in_code_fence,
        think_end_token: a.think_end_token,
        think_start_token: a.think_start_token,
        think_ended: a.think_ended,
        think_just_ended: a.think_just_ended,
        post_think_emitted: a.post_think_emitted,
        think_skip_count: a.think_skip_count,
        require_tool_call: a.require_tool_call,
        tool_request: a.tool_request,
        tools_present: a.tools_present,
        suppress_tool_call: a.suppress_tool_call,
        disable_mtp: a.disable_mtp,
        mtp_acct: a.mtp_acct,
        content_started: a.content_started,
        content_tokens: a.content_tokens,
        prose_tokens_since_last_tool: a.prose_tokens_since_last_tool,
        think_watchdog_fires: a.think_watchdog_fires,
        rollback_count: a.rollback_count,
        tool_call_start_token: a.tool_call_start_token,
        tool_call_opened: a.tool_call_opened,
        tool_call_end_token: a.tool_call_end_token,
        last_token_time: a.last_token_time,
        request_start: a.request_start,
        decode_start: a.decode_start,
        seed: a.seed,
        top_logprobs: a.top_logprobs,
        logprobs_data: a.logprobs_data,
        timeout_at: a.timeout_at,
        swap_id,
        cached_prompt_tokens: a.cached_prompt_tokens,
    })
}

/// 2026-09-25: Requeue a decode-preempted victim: offer its KV to the
/// prefix cache, release its device sequence and keep the whole
/// `ActiveSeq` on the host for [`resume_preempted_seq`].
pub(super) fn preempt_requeue(io: &SchedIo, mut a: ActiveSeq) -> PreemptedSeq {
    let tokens = a.seq.tokens.clone();
    let _ = io.dev.apply(Effect::ReleaseSeq {
        seq: &mut a.seq,
        cache: true,
        what: "preempt_requeue",
    });
    a.pending_drafts.clear();
    a.pending_draft_conf.clear();
    a.spec_adapt = Default::default();
    PreemptedSeq { a, tokens }
}

/// 2026-10-14 (race-memory, A184): Whether a batched speculative step's error (`msg`, the
/// full `{e:#}` chain) is a KV refusal that is safe to retry after a preemption. It must
/// name "KV cache exhausted" (block admission, before any forward), and on a multi-rank
/// serve it must come from the rank vote (`lazy_agree.rs`, "rank-agreed"), so no rank ran
/// the step. A refusal without the vote may leave the ranks apart (A183): it fails the
/// step's sequences instead.
pub(super) fn spec_kv_refusal_retryable(model: &dyn Model, msg: &str) -> bool {
    msg.contains("KV cache exhausted") && (!model.is_ep() || msg.contains("rank-agreed"))
}

/// 2026-10-14 (race-memory, A184): Free room after a batched MTP or DFlash step was refused for KV
/// (`step_mtp` returned `msg`), as a plain decode step does in `launch_with_preemption`:
/// preempt one victim (`choose_decode_victim`; spilled, or requeued for re-prefill) and
/// let the others retry on the next step. Every live sequence drops its drafts and the
/// secondary stream is ordered first, as on an MTP-gate switch to plain decode, so the
/// next step bootstraps them. With no eligible victim (one live sequence left, or only
/// grammar rows) every live sequence is failed with `msg`: the client gets an error
/// frame, never a cut reply ending "stop" (kvagcheck2, 2026-10-14).
pub(super) fn preempt_after_spec_refusal(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    active: &mut Vec<ActiveSeq>,
    msg: &str,
    swapped: &mut Vec<SwappedSeq>,
    preempted: &mut Vec<PreemptedSeq>,
) {
    let io = &sched.io;
    let model = io.dev.model();
    for a in active.iter_mut() {
        a.pending_drafts.clear();
        a.pending_draft_conf.clear();
    }
    if let Err(e) = io.dev.apply(Effect::SyncSecondary) {
        tracing::error!("spec KV refusal sync_secondary: {e}");
    }
    let spill = io.spill.as_deref();
    let live = active.iter().filter(|a| !a.finished).count();
    let victim = if live > 1 {
        super::decode_launch::choose_decode_victim(model, active, spill.is_some())
    } else {
        None
    };
    let Some(vi) = victim else {
        tracing::error!("{msg}: no victim to preempt; failing {live} live sequence(s)");
        for a in active.iter_mut().filter(|a| !a.finished) {
            super::lifecycle::fail_sequence(a, msg.to_string());
        }
        return;
    };
    // 2026-10-14: `remove`, not `swap_remove`, as in `launch_with_preemption`.
    let v = active.remove(vi);
    tracing::warn!(
        "KV cache exhausted in a batched MTP step: preempting slot={} ({} blocks, {} tokens \
         generated) for later RESUME so the other {} sequence(s) can continue",
        v.seq.slot_idx,
        v.seq.block_table.len(),
        v.output_tokens.len(),
        live - 1,
    );
    match spill {
        Some(sp) => match spill_out_sequence(io, v, sp) {
            Ok(s) => swapped.push(s),
            Err((v, spill_err)) => {
                tracing::warn!(
                    "spec-preempt spill failed ({spill_err:#}); requeuing victim for re-prefill instead"
                );
                preempted.push(preempt_requeue(io, v));
            }
        },
        None => preempted.push(preempt_requeue(io, v)),
    }
}

/// 2026-10-15 (race-memory, gap (a); cap2h lost a 324K reply to "preempted: KV cache exhausted
/// (a prefill needed its blocks)"): free room for a prefill chunk refused for KV by requeuing one
/// active row for a later resume, as decode-time preemption does, instead of ending it with an
/// error. The victim rule is decode's (`choose_decode_victim`: fewest output tokens, the resume
/// immunity, never a finished or grammar row; no spill pool here, so never a vision row). Returns
/// `false` when no row is eligible: the prefill's own error then stands.
pub(super) fn preempt_for_prefill(
    io: &SchedIo,
    active: &mut Vec<ActiveSeq>,
    preempted: &mut Vec<PreemptedSeq>,
) -> bool {
    let Some(vi) = super::decode_launch::choose_decode_victim(io.dev.model(), active, false) else {
        return false;
    };
    // `remove`, not `swap_remove`, as in `launch_with_preemption`.
    let v = active.remove(vi);
    tracing::warn!(
        "KV cache exhausted during prefill chunk: preempting slot={} ({} blocks, {} tokens \
         generated) for later RESUME so this prefill can proceed ({} sequence(s) remain)",
        v.seq.slot_idx,
        v.seq.block_table.len(),
        v.output_tokens.len(),
        active.len(),
    );
    preempted.push(preempt_requeue(io, v));
    true
}

/// 2026-09-25: Resume a requeued victim: allocate a new sequence and
/// prefill the saved token history into it. The prefill's logits are
/// discarded and `a.last_token` stays the next decode input, so no token
/// is sampled or sent twice. On error the client has already been sent an
/// error frame.
/// 2026-10-14 (race-memory, kvagcheck8): the re-prefill runs as planned `prefill_chunk` calls,
/// each announced with its own `0xFFFFFFF0` broadcast, as `prefill_a_step` and
/// `run_standard` do. The worker's `0xFFFFFFF0` handler calls `prefill_chunk`, whose prefix
/// lookup joins an `ep_min_u32` agreement; a rank-0 `prefill` never joins it, and both ranks
/// of the 2-node car hung on the first resume.
pub(super) fn resume_preempted_seq(
    model: &dyn Model,
    io: &SchedIo,
    p: PreemptedSeq,
    max_prefill_tokens: usize,
) -> Result<ActiveSeq> {
    let PreemptedSeq { mut a, tokens } = p;
    let mut seq = match model.alloc_sequence() {
        Ok(s) => s,
        Err(e) => {
            send_error_to_sink(
                io,
                &mut a.sink,
                &format!("preempt-resume alloc failed: {e:#}"),
            );
            return Err(e);
        }
    };
    seq.session_hash = a.session_hash;
    seq.adapter_slot = a.seq.adapter_slot;
    // 2026-09-25: Re-acquire the adapter slot the preempt-time release
    // gave up. LoRA rotations wait while `preempted` is non-empty
    // (`core/tick.rs`), so the slot still holds the same adapter.
    seq.adapter_id = a.seq.adapter_id;
    seq.acquired_adapter_slot = model.acquire_adapter_slot(a.seq.adapter_slot);

    // 2026-09-25: The same EP prefill preamble as `prefill_b_step`, so an
    // EP worker runs the re-prefill too.
    let prefill_result = (|| -> Result<()> {
        let total = tokens.len();
        let mut offset = 0;
        while offset < total {
            let chunk_len = resume_chunk_len(model, &tokens, offset, max_prefill_tokens);
            let is_last = offset + chunk_len >= total;
            model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF0)?;
            model.ep_broadcast_cmd(chunk_len as u32)?;
            model.ep_broadcast_cmd(offset as u32)?;
            model.ep_broadcast_cmd(total as u32)?;
            model.ep_broadcast_tokens(&tokens)?;
            // 2026-09-25: The worker makes the matching call in its
            // `0xFFFFFFF0` handler, so it must run here too.
            model.ep_sync_vision_embeds(&tokens)?;
            model.prefill_chunk(&tokens, &mut seq, offset, chunk_len, is_last, 0)?;
            // 2026-10-14: The worker normalizes after every chunk; so does `run_standard`.
            if let Err(e) = model.normalize_ssm_states(&seq, 0) {
                tracing::warn!("preempt-resume SSM state normalization failed: {e:#}");
            }
            offset += chunk_len;
        }
        Ok(())
    })();
    if let Err(e) = prefill_result {
        a.seq = seq;
        send_error(
            io,
            &mut a,
            &format!("preempt-resume re-prefill failed: {e:#}"),
        );
        return Err(e);
    }
    // 2026-09-25: `seq.prompt_len` stays as prefill set it: the whole
    // history. `cache_sequence` treats the first `prompt_len` tokens as
    // already inserted by prefill and does not count them again.
    a.seq = seq;
    // 2026-09-25: The old ring's snapshots belonged to the released
    // sequence; start an empty ring, as `resume_swapped_seq` does.
    a.ssm_rollback_ring = SsmDecodeRing::new(model.decode_rollback_ring_slots());
    a.preempt_immune_until_tokens = a.output_tokens.len() + PREEMPT_IMMUNITY_TOKENS;
    Ok(a)
}

/// 2026-10-14 (race-memory, kvagcheck8): The next re-prefill chunk at `offset`, planned as
/// `prefill_a_step` plans chunk 0 and `run_standard` the later ones: at most
/// `max_prefill_tokens` (0 = unchunked), and a non-last chunk ends on the block grid or the
/// tail split point. MLA prompts are not split further.
fn resume_chunk_len(
    model: &dyn Model,
    tokens: &[u32],
    offset: usize,
    max_prefill_tokens: usize,
) -> usize {
    let total = tokens.len();
    let remaining = total - offset;
    let cap = if max_prefill_tokens == 0 {
        remaining
    } else {
        remaining.min(max_prefill_tokens)
    };
    if model.is_mla() {
        return cap;
    }
    metrale_model_engine::prefill_plan::plan_chunk_len_grid(
        offset,
        total,
        cap,
        model.kv_block_size(),
        model.prefill_tail_split(tokens),
        model.prefill_grid(tokens),
    )
    .max(1)
}

/// 2026-09-25: Resume requeued victims, fewest blocks first, while batch
/// slots and KV blocks allow. A victim needs its history's blocks plus one
/// block of growth, and prefix-cache blocks are reclaimed to make room. A
/// victim that could never fit the pool is sent an error and dropped.
pub(super) fn resume_preempted_seqs(
    model: &dyn Model,
    io: &SchedIo,
    active: &mut Vec<ActiveSeq>,
    preempted: &mut Vec<PreemptedSeq>,
    max_batch_size: usize,
    block_size: usize,
    max_prefill_tokens: usize,
) {
    while !preempted.is_empty() && active.len() < max_batch_size {
        let Some((idx, needed)) = preempted
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.tokens.len() / block_size.max(1) + 1))
            .min_by_key(|&(_, n)| n)
        else {
            return;
        };
        // 2026-09-25: One block of growth: resuming into an exactly full
        // pool would preempt again on the next decode step.
        let want = needed + 1;
        let total = model.num_total_blocks();
        if total > 0 && want > total {
            // 2026-09-25: It can never fit; waiting would never end.
            let mut p = preempted.remove(idx);
            send_error_to_sink(
                io,
                &mut p.a.sink,
                &format!(
                    "preempted sequence needs {want} KV blocks but the pool has {total}; \
                     cannot resume"
                ),
            );
            continue;
        }
        let mut free = model.num_free_blocks();
        while free < want {
            let got = model.reclaim_prefix_blocks(want - free);
            if got == 0 {
                break;
            }
            free = model.num_free_blocks();
        }
        if free < want {
            return;
        }
        let p = preempted.remove(idx);
        let n_tokens = p.tokens.len();
        match resume_preempted_seq(model, io, p, max_prefill_tokens) {
            Ok(a) => {
                tracing::info!(
                    "Preempt-resume: re-prefilled {n_tokens} tokens \
                     ({} generated so far), decode continues",
                    a.output_tokens.len(),
                );
                active.push(a);
            }
            Err(e) => tracing::error!("Preempt-resume failed: {e:#}"),
        }
    }
}
