// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Sequence lifecycle: finish, errors, swap-out, resume.
//!
//! Owner: scheduler.
//! Invariants:
//! - `guard_stop_wire_reason` maps the request-timeout guard to `"timeout"`
//!   and every other guard to `"length"`, never to `"stop"`.

use super::*;
use crate::scheduler::io::{Effect, FinishFrame, SchedIo, SpillIo};

/// 2026-09-25: The mapping from a server-side guard (`ActiveSeq::guard_stop`)
/// to the wire `finish_reason`: the request timeout reports
/// `ir::FINISH_REASON_TIMEOUT`, every other guard `"length"`. A guard cut
/// the response short, so `"stop"` ("the model finished") would be false;
/// `"length"` tells the client the output is incomplete. The guard's name
/// travels separately, in `StreamEvent::Done.guard_stop`.
fn guard_stop_wire_reason(guard: &'static str) -> &'static str {
    match guard {
        // 2026-09-25: `derive_finish_reason`, the only caller, returns for
        // this guard before calling here; the arm keeps the mapping total.
        GUARD_STOP_REQUEST_TIMEOUT => crate::ir::FINISH_REASON_TIMEOUT,
        _ => "length",
    }
}

/// 2026-09-25: Derive the wire finish reason for a completed sequence.
///
/// `"length"` comes only from a guard cut or an exhausted budget
/// (`hard_ceiling_hit`: the `max_tokens` countdown or the served context
/// ceiling). A sequence that ended for any other reason without a guard
/// reports `"stop"`, even when its last token was not EOS.
///
/// Precedence:
///   1. the server-side deadline → `"timeout"` — a truncation must never
///      be mistaken for a natural stop, even when the last token is EOS;
///   2. token-derived natural stops (EOS → `"stop"`, tool-call close →
///      `"tool_calls"`) — what the model actually sampled outranks any
///      other guard that tripped on the same step;
///   3. any other guard → `guard_stop_wire_reason` (single mapping above);
///   4. token budget exhausted → `"length"`;
///   5. otherwise `"stop"`: the sequence ended with budget left and no
///      guard.
///
/// Pure, so `lifecycle_tests.rs` tests the precedence without a model.
pub(super) fn derive_finish_reason(
    guard_stop: Option<&'static str>,
    last_tok: Option<u32>,
    eos_tokens: &[u32],
    tool_call_end_token: Option<u32>,
    remaining: usize,
    seq_len: usize,
    max_seq_len: usize,
) -> &'static str {
    if guard_stop == Some(GUARD_STOP_REQUEST_TIMEOUT) {
        return crate::ir::FINISH_REASON_TIMEOUT;
    }
    if let Some(t) = last_tok {
        if eos_tokens.contains(&t) {
            return "stop";
        }
        // 2026-09-25: Guarded on `Some(t)` so an empty output (max_tokens==0
        // scoring path) on a model with no tool-call end token configured
        // cannot satisfy `None == None` and misreport "tool_calls".
        if Some(t) == tool_call_end_token {
            return "tool_calls";
        }
    }
    if let Some(guard) = guard_stop {
        return guard_stop_wire_reason(guard);
    }
    if hard_ceiling_hit(remaining, seq_len, max_seq_len) {
        return "length";
    }
    "stop"
}

/// 2026-09-25: Send the final response and release the sequence; a sequence
/// with `error` set gets an error frame instead (`send_error`).
///
/// `max_seq_len` is the served context ceiling (`sched.limits.max_seq_len`,
/// 0 = unlimited), for `hard_ceiling_hit` in the `"length"` decision.
pub fn finish_sequence(io: &SchedIo, a: &mut ActiveSeq, max_seq_len: usize) {
    // 2026-09-25: A failed sequence is not a finished one: it gets an error
    // frame and is not cached. An ordinary finish would report a normal
    // finish_reason over a truncated answer and seed the prefix cache from
    // a failed generation.
    if let Some(msg) = a.error.take() {
        send_error(io, a, &msg);
        return;
    }
    let reason = derive_finish_reason(
        a.guard_stop,
        a.output_tokens.last().copied(),
        &a.eos_tokens,
        a.tool_call_end_token,
        a.remaining,
        a.seq.seq_len,
        max_seq_len,
    );
    {
        let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
        let decode_ms = io
            .clock
            .now()
            .saturating_duration_since(a.decode_start)
            .as_secs_f64()
            * 1000.0;
        io.req.finish(
            &mut a.sink,
            FinishFrame {
                finish_reason: reason,
                output_tokens: &a.output_tokens,
                time_to_first_token_ms: ttft_ms,
                decode_time_ms: decode_ms,
                reasoning_tokens: a.thinking_tokens,
                cached_prompt_tokens: a.cached_prompt_tokens,
                accepted_prediction_tokens: a.mtp_acct.accepted_total() as usize,
                guard_stop: a.guard_stop,
                logprobs: &mut a.logprobs_data,
                prompt_logprobs: &mut a.seq.prompt_logprobs,
            },
        );
        io.tel
            .request_finished(metrale_telemetry::request::RequestTiming {
                ttft_ms,
                decode_ms,
                output_tokens: a.output_tokens.len() as u64,
            });
    }
    let decode_s = io
        .clock
        .now()
        .saturating_duration_since(a.decode_start)
        .as_secs_f64();
    let n = a.output_tokens.len();
    let tps = if decode_s > 0.0 {
        n as f64 / decode_s
    } else {
        0.0
    };
    let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
    super::mtp_accept_debug::RequestAccept::log_done(n, reason, tps, ttft_ms, &a.mtp_acct);
    // 2026-09-25: Cache the full sequence (prompt + generated) in the prefix
    // cache before the free, while its block indices are valid; then free
    // it and tell the EP worker to free and re-allocate its mirror.
    let _ = io.dev.apply(Effect::ReleaseSeq {
        seq: &mut a.seq,
        cache: true,
        what: "finish_sequence",
    });
}

/// 2026-09-25: Send an error frame to the client and release the sequence
/// without caching it.
pub fn send_error(io: &SchedIo, a: &mut ActiveSeq, msg: &str) {
    io.req.error(&mut a.sink, msg, "error frame");
    let _ = io.dev.apply(Effect::ReleaseSeq {
        seq: &mut a.seq,
        cache: false,
        what: "send_error",
    });
}

/// 2026-09-25: Send an error frame to a `ResponseSink` that has no
/// `ActiveSeq` (a failed prefill or swap-in, or the shutdown drain). A
/// dropped sink reaches a blocking client as "Inference cancelled".
pub fn send_error_to_sink(io: &SchedIo, sink: &mut ResponseSink, msg: &str) {
    io.req.error(sink, msg, "pre-seq error frame");
}

/// 2026-09-25: Swap out an active sequence to disk, freeing its GPU blocks.
///
/// Removes the sequence at `victim_idx` from `active` (`swap_remove`),
/// saves its state to a swap file, frees GPU resources, and returns a
/// `SwappedSeq`. On error the victim gets an error frame and is released.
pub fn swap_out_sequence(
    io: &SchedIo,
    active: &mut Vec<ActiveSeq>,
    victim_idx: usize,
    spill: &dyn SpillIo,
) -> Result<SwappedSeq> {
    let mut a = active.swap_remove(victim_idx);

    // 2026-09-25: `swap_remove` moved the last row into `victim_idx`;
    // compact its slot to match.
    if victim_idx < active.len() && active[victim_idx].seq.slot_idx != victim_idx {
        // 2026-09-25: Not `?`: `a` is already out of `active`, so an early
        // return would drop its sink (a blocking client would read
        // "Inference cancelled") and never free its KV blocks. `send_error`
        // answers the client and releases the sequence.
        if let Err(e) = io.dev.apply(Effect::CompactSlot {
            seq: &mut active[victim_idx].seq,
            target: victim_idx,
        }) {
            let e = e.into_inner();
            send_error(io, &mut a, &format!("swap-out failed: {e:#}"));
            return Err(e);
        }
        // 2026-09-25: Disown the slot the moved row now owns before the
        // fallible save below, so freeing or dropping `a` cannot release it
        // (`detach_slot_for_reuse`).
        let _ = io.dev.apply(Effect::DetachSlot { seq: &mut a.seq });
    }

    // 2026-09-25: Save, free and build are `preempt::spill_out_sequence`,
    // shared with decode-time preemption. On error the victim gets an error
    // frame and is released.
    match super::preempt::spill_out_sequence(io, a, spill) {
        Ok(s) => Ok(s),
        Err((mut a, e)) => {
            send_error(io, &mut a, &format!("swap-out failed: {e:#}"));
            Err(e)
        }
    }
}

/// 2026-09-25: Rebuild the GPU sequence for a parked request from its
/// spill image.
///
/// Every fallible step of a swap-in is here, so [`resume_swapped_seq`], the
/// sole owner of the `SwappedSeq`, handles one `Err` and answers the client
/// there.
///
/// A sequence allocated before a later step fails is freed here. Once the
/// allocation succeeds, removal of the spill file is attempted on every
/// exit.
fn restore_swapped_image(
    model: &dyn Model,
    s: &SwappedSeq,
    spill: &dyn SpillIo,
) -> Result<SequenceState> {
    let mut seq = model.alloc_sequence()?;
    // 2026-09-25: `reader` is dropped at the end of the closure, before
    // `remove`.
    let restored = spill
        .open(s.swap_id)
        .and_then(|mut reader| model.restore_sequence_state(&mut seq, s.num_blocks, &mut reader));
    if let Err(e) = restored {
        if let Err(fe) = model.free_sequence(&mut seq) {
            tracing::error!("restore_swapped_image: free_sequence after failed restore: {fe:#}");
        }
        let _ = spill.remove(s.swap_id);
        return Err(e);
    }
    if let Err(e) = spill.remove(s.swap_id) {
        if let Err(fe) = model.free_sequence(&mut seq) {
            tracing::error!("restore_swapped_image: free_sequence after failed unlink: {fe:#}");
        }
        return Err(e);
    }
    Ok(seq)
}

/// 2026-09-25: Resume a swapped-out sequence by restoring its state from disk.
pub fn resume_swapped_seq(
    _think_end_token: Option<u32>,
    _think_start_token: Option<u32>,
    model: &dyn Model,
    io: &SchedIo,
    mut s: SwappedSeq,
    spill: &dyn SpillIo,
) -> Result<ActiveSeq> {
    // 2026-09-25: Starvation guard: for its next `PREEMPT_IMMUNITY_TOKENS`
    // tokens a just-resumed sequence is picked as a KV preemption victim
    // only when no other row is eligible (`choose_decode_victim`).
    let immune_until = s.output_tokens.len() + super::preempt::PREEMPT_IMMUNITY_TOKENS;
    let mut seq = match restore_swapped_image(model, &s, spill) {
        Ok(seq) => seq,
        Err(e) => {
            // 2026-09-25: The caller (`core/end_tick.rs`) has already
            // removed this request from `swapped` and only logs the `Err`,
            // so the client must get its error frame here; a dropped sink
            // reads as "Inference cancelled" on the blocking side.
            send_error_to_sink(io, &mut s.sink, &format!("swap-in failed: {e:#}"));
            return Err(e);
        }
    };

    seq.tokens = s.tokens;
    seq.seq_len = s.seq_len;
    seq.adapter_slot = s.adapter_slot;
    seq.adapter_id = s.adapter_id;
    // 2026-09-25: Swap-out released this sequence's adapter-slot ref (in
    // `free_sequence`) and a resume does not re-run prefill, so re-acquire
    // it here. The resolved index is stored because the release at the
    // next free keys off it.
    seq.acquired_adapter_slot = model.acquire_adapter_slot(s.adapter_slot);

    Ok(ActiveSeq {
        seq,
        session_hash: s.session_hash,
        last_token: s.last_token,
        output_tokens: s.output_tokens,
        remaining: s.remaining,
        min_tokens: s.min_tokens,
        eos_tokens: s.eos_tokens,
        finished: false,
        error: None,
        guard_stop: None,
        param_close_pending: 0,
        sink: s.sink,
        // 2026-09-25: `SwappedSeq` carries no cancel flag, so a resumed
        // sequence cannot be cancelled through one.
        cancel_flag: None,
        temperature: s.temperature,
        top_k: s.top_k,
        top_p: s.top_p,
        top_n_sigma: s.top_n_sigma,
        min_p: s.min_p,
        repetition_penalty: s.repetition_penalty,
        presence_penalty: s.presence_penalty,
        frequency_penalty: s.frequency_penalty,
        repetition_penalty_window: 256,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: s.dry_multiplier,
        dry_base: s.dry_base,
        dry_allowed_length: s.dry_allowed_length,
        dry_sequence_breakers: s.dry_sequence_breakers,
        logit_bias: s.logit_bias,
        inside_thinking: s.inside_thinking,
        enable_thinking: s.enable_thinking,
        thinking_budget: s.thinking_budget,
        repetition_detection: s.repetition_detection,
        spontaneous_think_budget: s.spontaneous_think_budget,
        thinking_tokens: s.thinking_tokens,
        force_end_thinking: s.force_end_thinking,
        sentence_defer_count: s.sentence_defer_count,
        consecutive_confident: s.consecutive_confident,
        in_code_fence: s.in_code_fence,
        think_end_token: s.think_end_token,
        think_start_token: s.think_start_token,
        think_ended: s.think_ended,
        think_just_ended: s.think_just_ended,
        post_think_emitted: s.post_think_emitted,
        spec_adapt: Default::default(),
        spec_think_trail: Default::default(),
        think_skip_count: s.think_skip_count,
        require_tool_call: s.require_tool_call,
        tool_request: s.tool_request,
        tools_present: s.tools_present,
        suppress_tool_call: s.suppress_tool_call,
        disable_mtp: s.disable_mtp,
        mtp_acct: s.mtp_acct,
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: s.think_watchdog_fires,
        think_force_closed: s.think_force_closed,
        rollback_count: s.rollback_count,
        // 2026-09-25: Decode-rollback SSM snapshots are not part of the
        // swap image, so a resumed sequence starts with an empty ring; until
        // a new snapshot exists, a hybrid-model rollback declines.
        ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
        tool_call_start_token: s.tool_call_start_token,
        tool_call_opened: s.tool_call_opened,
        // 2026-09-25: A resumed sequence starts outside a tool body, even
        // if it was swapped out inside one.
        inside_tool_body: false,
        tool_call_completed: false,
        post_completion_tool_opens: 0,
        tool_body_streak_tokens: 0,
        inside_parameter_body: false,
        param_body_chars_emitted: 0,
        tool_call_end_token: s.tool_call_end_token,
        // 2026-09-25: `SwappedSeq` keeps no grammar state, so a resumed
        // sequence decodes without a grammar.
        grammar_state: None,
        pending_drafts: Vec::new(),
        pending_draft_conf: Vec::new(),
        last_token_time: io.clock.now(),
        request_start: s.request_start,
        decode_start: s.decode_start,
        seed: s.seed,
        top_logprobs: s.top_logprobs,
        logprobs_data: s.logprobs_data,
        timeout_at: s.timeout_at,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(s.temperature),
        cached_prompt_tokens: s.cached_prompt_tokens,
        preempt_immune_until_tokens: immune_until,
    })
}

/// 2026-09-25: Mark a sequence for retirement as a failure. The caller
/// keeps it in `active`; `finish_sequence` turns `error` into an error frame
/// for the client instead of a normal finish.
pub fn fail_sequence(a: &mut ActiveSeq, msg: String) {
    a.error = Some(msg);
    a.finished = true;
}
