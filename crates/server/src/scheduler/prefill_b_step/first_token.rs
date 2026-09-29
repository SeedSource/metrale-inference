// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The end of `prefill_request` once the first token is sampled:
//! stream it, then finish the request or return its `ActiveSeq`.
//!
//! Owner: scheduler.
//! Invariants:
//! - Parameters that own a value are declared in the order the caller
//!   declared them, so they drop in the caller's order on an early return.

use super::*;

/// 2026-09-26: Finish `prefill_request` after its first token; the caller's
/// return value.
pub(super) fn finish_first_token(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    model: &dyn Model,
    eos_tokens: &[u32],
    spontaneous_think_budget: u32,
    top_k: u32,
    top_p: f32,
    top_n_sigma: f32,
    min_p: f32,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    req_lz_penalty: f32,
    logit_bias: Vec<(u32, f32)>,
    req_min_tokens: usize,
    req_session_hash: u64,
    req_enable_thinking: bool,
    req_thinking_budget: Option<u32>,
    req_repetition_detection: Option<crate::api::inference_types::RepetitionDetectionParams>,
    req_require_tool_call: bool,
    req_tools_present: bool,
    req_suppress_tool_call: bool,
    req_disable_mtp: bool,
    req_seed: Option<u64>,
    req_top_logprobs: Option<u8>,
    req_timeout_at: Option<std::time::Instant>,
    request_start: std::time::Instant,
    grammar_state: Option<GrammarState>,
    max_tokens: usize,
    sink: ResponseSink,
    temperature: f32,
    cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    mut seq: SequenceState,
    first: u32,
) -> Result<Option<ActiveSeq>> {
    // 2026-09-25: A first token of `<think>` without thinking requested is
    // not sent or recorded; the sequence starts inside thinking instead.
    let spontaneous_think = !req_enable_thinking && think_start_token == Some(first);
    // 2026-09-25: Prompt logprobs go to a streaming client before any
    // token event.
    if seq.collect_prompt_logprobs.is_some() && sink.is_streaming() {
        let lps: Vec<crate::api::TokenLogprobs> = seq
            .prompt_logprobs
            .drain(..)
            .map(|p| crate::api::TokenLogprobs {
                token_id: p.token_id,
                logprob: p.logprob,
                top: p.top,
            })
            .collect();
        if !sched.io.req.emit(
            &sink,
            StreamEvent::PromptLogprobs(lps),
            "prefill_b prompt-logprobs",
        ) {
            tracing::warn!(target: "met::scheduler::prefill_b_step", "prefill_b_step: prompt-logprobs send failed");
        }
    }
    if !spontaneous_think
        && max_tokens > 0
        && !sched
            .io
            .req
            .emit(&sink, StreamEvent::Token(first), "prefill_b first token")
    {
        tracing::warn!(target: "met::scheduler::prefill_b_step", "prefill_b_step: first-token send failed (receiver dropped)");
    }

    let output_tokens = if spontaneous_think || max_tokens == 0 {
        vec![]
    } else {
        vec![first]
    };

    let use_legacy_tool_call =
        req_require_tool_call && grammar_state.is_none() && tool_call_start_token.is_some();
    // 2026-09-25: Computed before `grammar_state` moves into the
    // `ActiveSeq`.
    let tool_request = grammar_state.is_some() || use_legacy_tool_call;

    let now = sched.io.clock.now();
    let cached_prompt_tok = seq.reused_prefix_tokens as u32;

    if !spontaneous_think && (eos_tokens.contains(&first) || max_tokens <= 1) {
        let mut a = ActiveSeq {
            seq,
            session_hash: req_session_hash,
            last_token: first,
            output_tokens,
            remaining: 0,
            min_tokens: req_min_tokens,
            eos_tokens: eos_tokens.to_vec(),
            finished: true,
            error: None,
            guard_stop: None,
            param_close_pending: 0,
            sink,
            cancel_flag: cancel_flag.clone(),
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            repetition_penalty_window: 256,
            presence_penalty,
            frequency_penalty,
            lz_penalty: req_lz_penalty,
            dry_multiplier: DEFAULT_DRY_MULTIPLIER,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            logit_bias: logit_bias.clone(),
            pending_drafts: Vec::new(),
            pending_draft_conf: Vec::new(),
            inside_thinking: born_inside_thinking(req_enable_thinking, think_end_token),
            enable_thinking: req_enable_thinking,
            thinking_budget: req_thinking_budget,
            repetition_detection: req_repetition_detection,
            spontaneous_think_budget,
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
            think_ended: !req_enable_thinking && think_end_token.is_some(),
            think_just_ended: false,
            post_think_emitted: 0,
            spec_adapt: Default::default(),
            spec_think_trail: Default::default(),
            think_skip_count: 0,
            require_tool_call: use_legacy_tool_call,
            tool_request,
            tools_present: req_tools_present,
            suppress_tool_call: req_suppress_tool_call,
            disable_mtp: req_disable_mtp,
            mtp_acct: Default::default(),
            content_started: false,
            content_tokens: 0,
            prose_tokens_since_last_tool: 0,
            think_watchdog_fires: 0,
            rollback_count: 0,
            ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
            tool_call_start_token,
            tool_call_opened: false,
            inside_tool_body: false,
            tool_call_completed: false,
            post_completion_tool_opens: 0,
            tool_body_streak_tokens: 0,
            inside_parameter_body: false,
            param_body_chars_emitted: 0,
            tool_call_end_token,
            grammar_state,
            last_token_time: now,
            request_start,
            decode_start: now,
            seed: req_seed,
            top_logprobs: req_top_logprobs,
            logprobs_data: Vec::new(),
            timeout_at: req_timeout_at,
            adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
        };
        finish_sequence(&sched.io, &mut a, sched.limits.max_seq_len);
        return Ok(None);
    }

    Ok(Some(ActiveSeq {
        seq,
        session_hash: req_session_hash,
        last_token: first,
        output_tokens,
        remaining: max_tokens - 1,
        min_tokens: req_min_tokens,
        eos_tokens: eos_tokens.to_vec(),
        finished: false,
        error: None,
        guard_stop: None,
        param_close_pending: 0,
        sink,
        cancel_flag,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        repetition_penalty_window: 256,
        presence_penalty,
        frequency_penalty,
        lz_penalty: req_lz_penalty,
        dry_multiplier: DEFAULT_DRY_MULTIPLIER,
        dry_base: DEFAULT_DRY_BASE,
        dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
        dry_sequence_breakers: Vec::new(),
        logit_bias,
        pending_drafts: Vec::new(),
        pending_draft_conf: Vec::new(),
        inside_thinking: spontaneous_think
            || born_inside_thinking(req_enable_thinking, think_end_token),
        enable_thinking: req_enable_thinking,
        thinking_budget: if spontaneous_think {
            Some(spontaneous_think_budget)
        } else {
            req_thinking_budget
        },
        repetition_detection: req_repetition_detection,
        spontaneous_think_budget,
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
        think_ended: if spontaneous_think {
            false
        } else {
            !req_enable_thinking && think_end_token.is_some()
        },
        think_just_ended: false,
        post_think_emitted: 0,
        spec_adapt: Default::default(),
        spec_think_trail: Default::default(),
        think_skip_count: 0,
        require_tool_call: use_legacy_tool_call,
        tool_request,
        tools_present: req_tools_present,
        suppress_tool_call: req_suppress_tool_call,
        disable_mtp: req_disable_mtp,
        mtp_acct: Default::default(),
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
        tool_call_start_token,
        tool_call_opened: false,
        inside_tool_body: false,
        tool_call_completed: false,
        post_completion_tool_opens: 0,
        tool_body_streak_tokens: 0,
        inside_parameter_body: false,
        param_body_chars_emitted: 0,
        tool_call_end_token,
        grammar_state,
        last_token_time: now,
        request_start,
        decode_start: now,
        seed: req_seed,
        top_logprobs: req_top_logprobs,
        logprobs_data: Vec::new(),
        timeout_at: req_timeout_at,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
    }))
}
