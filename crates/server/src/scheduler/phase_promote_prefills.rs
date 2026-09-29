// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Promote completed prefills into `active`, or finish them at
//! once. A completed entry without a first token has its sequence freed.
//!
//! Owner: scheduler.
//! Invariants:
//! - Entries leave `prefilling` through `Vec::remove`, so the rest of the
//!   queue keeps its arrival order.

use metrale_model_engine::traits::Model;
use std::time::Instant;

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn promote_completed_prefills(
    model: &dyn Model,
    io: &crate::scheduler::io::SchedIo,
    prefilling: &mut Vec<PrefillInProgress>,
    mut completed_indices: Vec<(usize, Option<u32>)>,
    active: &mut Vec<ActiveSeq>,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    // 2026-09-25: The served context ceiling, passed to `finish_sequence`.
    max_seq_len: usize,
) {
    // 2026-09-25: Reverse index order, so a removal never shifts an index
    // still to be removed.
    completed_indices.sort_unstable_by_key(|x| std::cmp::Reverse(x.0));
    for (idx, maybe_token) in completed_indices {
        // 2026-09-25: `remove`, not `swap_remove`: the single-stream path
        // advances the head (`prefilling.first_mut()`), and `swap_remove`
        // would move the newest arrival there. `enforce_request_deadlines`
        // walks only `active`, so a request passed over this way would not
        // time out either. See `prefill_fifo_tests`.
        let mut p = prefilling.remove(idx);
        let Some(first) = maybe_token else {
            let mut seq = p.seq;
            if let Err(e) = model.free_sequence(&mut seq) {
                tracing::error!("phase_promote_prefills: free_sequence (error path): {e:#}");
            }
            if let Err(e) = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1) {
                tracing::error!(
                    "phase_promote_prefills: ep_broadcast free+realloc (error path): {e:#}"
                );
            }
            continue;
        };
        let spontaneous_think = !p.enable_thinking && think_start_token == Some(first);
        // 2026-09-25: Prompt logprobs go to a streaming client before any
        // token event.
        if p.seq.collect_prompt_logprobs.is_some() && p.sink.is_streaming() {
            let lps: Vec<crate::api::TokenLogprobs> = p
                .seq
                .prompt_logprobs
                .drain(..)
                .map(|lp| crate::api::TokenLogprobs {
                    token_id: lp.token_id,
                    logprob: lp.logprob,
                    top: lp.top,
                })
                .collect();
            if !io.req.emit(
                &p.sink,
                StreamEvent::PromptLogprobs(lps),
                "promote prompt-logprobs",
            ) {
                tracing::warn!("phase_promote_prefills: prompt-logprobs send failed");
            }
        }
        // 2026-09-25: An EOS first token is not sent.
        if !spontaneous_think
            && p.max_tokens > 0
            && !p.eos_tokens.contains(&first)
            && !io
                .req
                .emit(&p.sink, StreamEvent::Token(first), "promote first token")
        {
            tracing::warn!("phase_promote_prefills: first-token send failed (receiver dropped)");
        }
        let use_legacy_tool_call =
            p.require_tool_call && p.grammar_state.is_none() && tool_call_start_token.is_some();
        let now = io.clock.now();
        let cached_prompt_tok = p.seq.reused_prefix_tokens as u32;
        let immediate_finish =
            !spontaneous_think && (p.eos_tokens.contains(&first) || p.max_tokens <= 1);

        let mut a = build_active_seq_from_prefill(
            p,
            first,
            spontaneous_think,
            use_legacy_tool_call,
            cached_prompt_tok,
            immediate_finish,
            now,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            model.decode_rollback_ring_slots(),
        );
        if immediate_finish {
            finish_sequence(io, &mut a, max_seq_len);
        } else {
            active.push(a);
        }
    }
}

/// 2026-09-25: Build an `ActiveSeq` from a finished prefill and its first
/// sampled token. The caller pushes it onto `active` or passes it to
/// `finish_sequence`.
#[allow(clippy::too_many_arguments)]
fn build_active_seq_from_prefill(
    p: PrefillInProgress,
    first: u32,
    spontaneous_think: bool,
    use_legacy_tool_call: bool,
    cached_prompt_tok: u32,
    immediate_finish: bool,
    now: Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    // 2026-09-25: Capacity of the sequence's SSM rollback ring.
    ssm_ring_capacity: usize,
) -> ActiveSeq {
    let temperature = p.temperature;
    // 2026-09-25: Computed before `p.grammar_state` moves into the struct.
    let tool_request = p.grammar_state.is_some() || use_legacy_tool_call;
    ActiveSeq {
        seq: p.seq,
        session_hash: p.session_hash,
        last_token: first,
        output_tokens: if (!immediate_finish && spontaneous_think) || p.max_tokens == 0 {
            vec![]
        } else {
            vec![first]
        },
        remaining: if immediate_finish {
            0
        } else {
            p.max_tokens - 1
        },
        min_tokens: p.min_tokens,
        eos_tokens: p.eos_tokens,
        finished: immediate_finish,
        error: None,
        guard_stop: None,
        param_close_pending: 0,
        sink: p.sink,
        cancel_flag: p.cancel_flag,
        temperature: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        top_n_sigma: p.top_n_sigma,
        min_p: p.min_p,
        repetition_penalty: p.repetition_penalty,
        presence_penalty: p.presence_penalty,
        frequency_penalty: p.frequency_penalty,
        repetition_penalty_window: 256,
        lz_penalty: p.lz_penalty,
        dry_multiplier: p.dry_multiplier,
        dry_base: p.dry_base,
        dry_allowed_length: p.dry_allowed_length,
        dry_sequence_breakers: Vec::new(),
        logit_bias: p.logit_bias,
        pending_drafts: Vec::new(),
        pending_draft_conf: Vec::new(),
        inside_thinking: if immediate_finish {
            born_inside_thinking(p.enable_thinking, think_end_token)
        } else {
            spontaneous_think || born_inside_thinking(p.enable_thinking, think_end_token)
        },
        enable_thinking: p.enable_thinking,
        thinking_budget: if !immediate_finish && spontaneous_think {
            Some(p.spontaneous_think_budget)
        } else {
            p.thinking_budget
        },
        repetition_detection: p.repetition_detection,
        spontaneous_think_budget: p.spontaneous_think_budget,
        thinking_tokens: 0,
        cached_prompt_tokens: cached_prompt_tok,
        preempt_immune_until_tokens: 0,
        force_end_thinking: false,
        think_force_closed: false,
        sentence_defer_count: 0,
        consecutive_confident: 0,
        in_code_fence: false,
        think_end_token,
        think_start_token,
        // 2026-09-25: With thinking off and a `</think>` token configured,
        // the sequence starts with thinking ended; a spontaneous `<think>`
        // that continues to decode starts it open.
        think_ended: if !immediate_finish && spontaneous_think {
            false
        } else {
            !p.enable_thinking && think_end_token.is_some()
        },
        think_just_ended: false,
        post_think_emitted: 0,
        spec_adapt: Default::default(),
        spec_think_trail: Default::default(),
        think_skip_count: 0,
        require_tool_call: use_legacy_tool_call,
        tool_request,
        tools_present: p.tools_present,
        tool_call_start_token,
        tool_call_opened: false,
        inside_tool_body: false,
        tool_call_completed: false,
        post_completion_tool_opens: 0,
        tool_body_streak_tokens: 0,
        inside_parameter_body: false,
        param_body_chars_emitted: 0,
        suppress_tool_call: p.suppress_tool_call,
        disable_mtp: p.disable_mtp,
        mtp_acct: Default::default(),
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(ssm_ring_capacity),
        tool_call_end_token,
        grammar_state: p.grammar_state,
        last_token_time: now,
        request_start: p.request_start,
        decode_start: now,
        seed: p.seed,
        top_logprobs: p.top_logprobs,
        logprobs_data: Vec::new(),
        timeout_at: p.timeout_at,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
    }
}

#[cfg(test)]
#[path = "cached_tokens_tests.rs"]
mod cached_tokens_tests;
