// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The end of `start_chunked_prefill` once chunk 0 has run:
//! sample the first token when the prompt fit one chunk, else hand the
//! request on as `InProgress`.
//!
//! Owner: scheduler.
//! Invariants:
//! - Every `Err` is preceded by `send_error_to_sink`, as in
//!   `start_chunked_prefill`.
//! - Parameters that own a value are declared in the order the caller
//!   declared them, so they drop in the caller's order on an early return.

use super::*;

/// 2026-09-26: Finish `start_chunked_prefill` after its chunk-0 prefill; the
/// caller's return value.
pub(super) fn finish_first_chunk(
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
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: u32,
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
    req_prompt_logprobs: Option<u8>,
    req_timeout_at: Option<std::time::Instant>,
    request_start: std::time::Instant,
    mut grammar_state: Option<GrammarState>,
    prompt_tokens: std::sync::Arc<Vec<u32>>,
    max_tokens: usize,
    mut sink: ResponseSink,
    temperature: f32,
    cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    chunk_len: usize,
    is_last: bool,
    mut seq: SequenceState,
    logits: DevicePtr,
) -> Result<StartPrefillResult> {
    if is_last {
        // 2026-09-25: The prompt fit one chunk: sample the first token.
        let first = match sample_first_token(
            model,
            logits,
            temperature,
            top_k,
            top_p,
            min_p,
            eos_tokens,
            grammar_state.as_mut(),
            FirstTokenPolicy::for_birth(
                req_enable_thinking,
                think_end_token,
                tool_call_start_token,
            ),
            &sched.levers.sampling(),
            sched.io.tel.dumps(),
        ) {
            Ok(t) => {
                tracing::info!(target: "met::scheduler::prefill_a_step", "Prefill first token: {t}");
                t
            }
            Err(e) => {
                let msg = format!("sample_token failed: {e:#}");
                send_error_to_sink(&sched.io, &mut sink, &msg);
                if let Err(free_err) = model.free_sequence(&mut seq) {
                    tracing::error!(target: "met::scheduler::prefill_a_step", "prefill_a_step: free_sequence (after sample error): {free_err:#}"
                    );
                }
                if let Err(bcast_err) =
                    model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1)
                {
                    tracing::error!(target: "met::scheduler::prefill_a_step", "prefill_a_step: ep_broadcast (after sample error): {bcast_err:#}"
                    );
                }
                return Err(e);
            }
        };

        let spontaneous_think = !req_enable_thinking && think_start_token == Some(first);
        // 2026-09-25: Prompt logprobs go to a streaming client before any
        // token event; a blocking request gets them from `finish_sequence`.
        if req_prompt_logprobs.is_some() && sink.is_streaming() {
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
                "prefill_a prompt-logprobs",
            ) {
                tracing::warn!(target: "met::scheduler::prefill_a_step", "prefill_a_step: prompt-logprobs send failed");
            }
        }
        // 2026-09-25: With `max_tokens == 0` (scoring only) no generated
        // token is sent.
        if !spontaneous_think
            && max_tokens > 0
            && !sched
                .io
                .req
                .emit(&sink, StreamEvent::Token(first), "prefill_a first token")
        {
            tracing::warn!(target: "met::scheduler::prefill_a_step", "prefill_a_step: first-token send failed (receiver dropped)");
        }

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
                // 2026-09-25: Scoring only: the sampled token is dropped.
                output_tokens: if max_tokens == 0 {
                    Vec::new()
                } else {
                    vec![first]
                },
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
                dry_multiplier,
                dry_base,
                dry_allowed_length,
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
                tool_call_start_token,
                tool_call_opened: false,
                inside_tool_body: false,
                tool_call_completed: false,
                post_completion_tool_opens: 0,
                tool_body_streak_tokens: 0,
                inside_parameter_body: false,
                param_body_chars_emitted: 0,
                suppress_tool_call: req_suppress_tool_call,
                disable_mtp: req_disable_mtp,
                mtp_acct: Default::default(),
                content_started: false,
                content_tokens: 0,
                prose_tokens_since_last_tool: 0,
                think_watchdog_fires: 0,
                rollback_count: 0,
                ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
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
            Ok(StartPrefillResult::Finished)
        } else {
            Ok(StartPrefillResult::Active(ActiveSeq {
                seq,
                session_hash: req_session_hash,
                last_token: first,
                output_tokens: if spontaneous_think {
                    vec![]
                } else {
                    vec![first]
                },
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
                dry_multiplier,
                dry_base,
                dry_allowed_length,
                dry_sequence_breakers: Vec::new(),
                logit_bias: logit_bias.clone(),
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
                tool_call_start_token,
                tool_call_opened: false,
                inside_tool_body: false,
                tool_call_completed: false,
                post_completion_tool_opens: 0,
                tool_body_streak_tokens: 0,
                inside_parameter_body: false,
                param_body_chars_emitted: 0,
                suppress_tool_call: req_suppress_tool_call,
                disable_mtp: req_disable_mtp,
                mtp_acct: Default::default(),
                content_started: false,
                content_tokens: 0,
                prose_tokens_since_last_tool: 0,
                think_watchdog_fires: 0,
                rollback_count: 0,
                ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
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
    } else {
        Ok(StartPrefillResult::InProgress(
            super::prefill_a_step_params::build_prefill_in_progress(
                prompt_tokens,
                req_session_hash,
                seq,
                chunk_len,
                max_tokens,
                req_min_tokens,
                eos_tokens.to_vec(),
                sink,
                cancel_flag,
                request_start,
                temperature,
                top_k,
                top_p,
                top_n_sigma,
                min_p,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
                req_lz_penalty,
                dry_multiplier,
                dry_base,
                dry_allowed_length,
                logit_bias,
                req_enable_thinking,
                req_thinking_budget,
                req_repetition_detection,
                spontaneous_think_budget,
                req_require_tool_call,
                req_tools_present,
                req_suppress_tool_call,
                req_disable_mtp,
                grammar_state,
                req_seed,
                req_top_logprobs,
                req_timeout_at,
            ),
        ))
    }
}
