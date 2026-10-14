// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `StreamEvent::Done` arm of the streaming chat handler: flush what
//! the stream still holds, resolve the wire finish reason, and emit the terminal
//! `Finish` delta with usage.
//!
//! Owner: server streaming API.
//! Invariants:
//! - `handle_done` returns exactly one `Finish` delta, and it is the last one.

use crate::ir::StreamDelta;
use crate::tool_parser;

use super::super::sanitizer::sanitize_content_chunk;
use super::super::stream_guards::flush_content_sanitizer;
use super::ctx::StreamCtx;
use super::state::StreamState;
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_args_fragment, handle_tool_call_delta,
    handle_tool_call_end, handle_tool_call_start,
};

type DeltaVec = Vec<StreamDelta>;

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_done(
    state: &mut StreamState,
    ctx: &StreamCtx,
    finish_reason: String,
    completion_tokens: usize,
    time_to_first_token_ms: f64,
    decode_time_ms: f64,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
    accepted_prediction_tokens: usize,
) -> DeltaVec {
    let mut deltas: DeltaVec = Vec::new();

    // 2026-09-26: With no stop match and no other stop flag, the stop-string hold-back
    // tail is real output: send it through the detector or the sanitizer, as live
    // content is.
    if !ctx.stop_strings.is_empty()
        && !state.stop_string_triggered
        && state.stop_string_emitted_len < state.accumulated_content.len()
    {
        let tail = state.accumulated_content[state.stop_string_emitted_len..].to_string();
        state.stop_string_emitted_len = state.accumulated_content.len();
        if !tail.is_empty() {
            if let Some(det) = state.detector.as_mut() {
                let outputs = det.process(&tail);
                for output in outputs {
                    match output {
                        tool_parser::DetectorOutput::Content(text) => {
                            let sanitized = sanitize_content_chunk(
                                &text,
                                &mut state.tag_scan_buf,
                                &mut state.suppressing_param_leak,
                                &mut state.inside_envelope,
                                &ctx.leak_markers,
                            );
                            if !sanitized.is_empty() {
                                deltas.push(StreamDelta::Content {
                                    text: sanitized,
                                    token_ids: Vec::new(),
                                });
                            }
                        }
                        tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                            handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut deltas);
                        }
                        tool_parser::DetectorOutput::ToolCallStart {
                            id: tc_id,
                            name,
                            idx,
                        } => {
                            handle_tool_call_start(state, ctx, tc_id, name, idx, &mut deltas);
                        }
                        tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                            handle_tool_call_delta(state, ctx, args, idx, &mut deltas);
                        }
                        tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                            handle_tool_call_args_fragment(state, ctx, fragment, idx, &mut deltas);
                        }
                        tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                            handle_tool_call_end(state, ctx, idx);
                        }
                    }
                }
            } else {
                let sanitized = sanitize_content_chunk(
                    &tail,
                    &mut state.tag_scan_buf,
                    &mut state.suppressing_param_leak,
                    &mut state.inside_envelope,
                    &ctx.leak_markers,
                );
                if !sanitized.is_empty() {
                    if state.refusal_scan_buf.len() < 16_384 {
                        state.refusal_scan_buf.push_str(&sanitized);
                    }
                    deltas.push(StreamDelta::Content {
                        text: sanitized,
                        token_ids: Vec::new(),
                    });
                }
            }
        }
    }

    if state.detector.is_some() {
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            det.flush()
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    let sanitized = sanitize_content_chunk(
                        &text,
                        &mut state.tag_scan_buf,
                        &mut state.suppressing_param_leak,
                        &mut state.inside_envelope,
                        &ctx.leak_markers,
                    );
                    if !sanitized.is_empty() {
                        deltas.push(StreamDelta::Content {
                            text: sanitized,
                            token_ids: state.take_ids_if(ctx.req_return_token_ids),
                        });
                    }
                }
                tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                    handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallStart {
                    id: tc_id,
                    name,
                    idx,
                } => {
                    handle_tool_call_start(state, ctx, tc_id, name, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                    handle_tool_call_delta(state, ctx, args, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                    handle_tool_call_args_fragment(state, ctx, fragment, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                    handle_tool_call_end(state, ctx, idx);
                }
            }
        }
    }

    let tail = flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if !tail.is_empty() {
        if state.refusal_scan_buf.len() < 16_384 {
            state.refusal_scan_buf.push_str(&tail);
        }
        deltas.push(StreamDelta::Content {
            text: tail,
            token_ids: state.take_ids_if(ctx.req_return_token_ids),
        });
    }

    // 2026-09-26: Neutral usage; the wire encoder derives its own fields from it.
    let usage = crate::ir::Usage {
        prompt_tokens: ctx.prompt_len,
        completion_tokens,
        cached_prompt_tokens: cached_prompt_tokens as usize,
        reasoning_tokens: reasoning_tokens as usize,
        accepted_prediction_tokens,
        time_to_first_token_ms,
        // 2026-09-26: The scheduler's decode window, as on the blocking path; the rate
        // below is derived from it.
        decode_time_ms,
        response_tokens_per_second: crate::ir::Usage::decode_rate_tok_s(
            completion_tokens,
            decode_time_ms,
        ),
    };

    let fr = resolve_wire_finish_reason(
        &finish_reason,
        state.tool_loop_capped,
        state.detector.as_ref().is_some_and(|d| d.has_tool_calls()) || state.salvaged_tool_call,
        state.stop_string_matched,
        state.guard_stop,
    );

    // 2026-09-26: The refusal classifier runs only when the detector saw no tool call.
    let refusal_signal = if state.detector.as_ref().is_none_or(|d| !d.has_tool_calls()) {
        crate::refusal::detect(&state.refusal_scan_buf)
    } else {
        None
    };
    if let Some(ref r) = refusal_signal {
        deltas.push(StreamDelta::Refusal { text: r.clone() });
    }

    // 2026-09-26: Terminal delta: finish reason and usage. How usage is framed on the
    // wire (`include_usage`) is the encoder's choice (`openai::encode_stream`). Token
    // ids not yet sent on a content delta ride this one.
    deltas.push(StreamDelta::Finish {
        reason: crate::ir::FinishReason::from(fr),
        usage,
        token_ids: state.take_ids_if(ctx.req_return_token_ids),
    });

    // 2026-09-26: REQUESTS_ACTIVE is released by the `ActiveRequestGuard` in `StreamCtx`
    // when the stream is dropped, so a stream that ends without `Done` releases it too.
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(ctx.prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(completion_tokens as u64);
    crate::metrics::TTFT_SECONDS
        .with_label_values(&[ctx.model.as_str()])
        .observe(time_to_first_token_ms / 1000.0);

    // 2026-09-26: Refund the part of the rate-limit reservation that was not used.
    if let Some(ref rctx) = ctx.req_ctx {
        let actual = (ctx.prompt_len + completion_tokens) as u64;
        let refund = rctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            ctx.state.rate_limiter.refund_tokens(&rctx.identity, refund);
        }
    }

    // 2026-09-26: `--dump` record synthesized from the post-sanitizer accumulators, with
    // usage in the OpenAI wire shape (`openai::Usage::from`, as the encoder uses).
    if let (Some(seq), Some(dump)) = (ctx.dump_seq, ctx.state.dump_writer.as_ref()) {
        let has_tool_calls = state.detector.as_ref().is_some_and(|d| d.has_tool_calls());
        let usage_for_dump = crate::openai::Usage::from(&usage);
        let body = serde_json::json!({
            "id": ctx.id,
            "model": ctx.model,
            "object": "chat.completion.synthesized",
            "finish_reason": fr,
            "content": state.refusal_scan_buf,
            "has_tool_calls": has_tool_calls,
            "usage": usage_for_dump,
            "stop_string_triggered": state.stop_string_triggered,
            "loop_watchdog_triggered": state.loop_watchdog_triggered,
            "tool_loop_capped": state.tool_loop_capped,
            "guard_stop": state.guard_stop,
            "_note": "Synthesized from post-sanitizer accumulators; \
                      per-chunk capture is a follow-up.",
        });
        dump.dump_response("/v1/chat/completions", seq, &body, true);
    }

    deltas
}

/// 2026-09-26: Stream-side overrides of the scheduler's finish reason, in order:
///
/// 1. `"timeout"`: the request deadline cut the response. It outranks everything
///    below, so a partial tool call from a truncated turn is not reported as
///    `"tool_calls"`.
/// 2. `tool_loop_capped` → `"length"`: a tool-call loop guard ended the response (the
///    tool-arg dedup, the within-response dedup, the same-name run cap in
///    `tool_handlers.rs`, or the in-think leak cut in `handle_token.rs`). Tool calls
///    were emitted, but the response was truncated.
/// 3. parsed or salvaged tool calls → `"tool_calls"`.
/// 4. `stop_string_matched` → `"stop"`: a client stop sequence ended the response. The
///    scheduler can still say `"length"` when the budget ran out before it saw the
///    cancel.
/// 5. a stream-side guard (`guard_stop`, such as the token loop watchdog or the
///    SimHash trip) → `"repetition"` (`ir::FinishReason::Repetition`: wire `"stop"` with
///    `stop_reason` `"repetition_detected"`). These guards cancel without naming a guard
///    to the scheduler, which then reports `"stop"`.
/// 6. otherwise the scheduler's reason.
///
/// 2026-10-14: Rung 5 was `"length"`, which made a guard cut read as the token budget
/// running out (#82 guard fix; race-memory OPEN 14(a): 7 of 24 normal answers cut).
fn resolve_wire_finish_reason<'a>(
    scheduler_reason: &'a str,
    tool_loop_capped: bool,
    has_tool_calls: bool,
    stop_string_matched: bool,
    stream_guard_stop: Option<&'static str>,
) -> &'a str {
    if scheduler_reason == crate::ir::FINISH_REASON_TIMEOUT {
        scheduler_reason
    } else if tool_loop_capped {
        "length"
    } else if has_tool_calls {
        "tool_calls"
    } else if stop_string_matched {
        "stop"
    } else if stream_guard_stop.is_some() {
        // 2026-09-26: It ranks below tool calls and a stop match, so a guard that trips
        // on the same step reports what actually happened.
        crate::ir::FINISH_REASON_REPETITION
    } else {
        scheduler_reason
    }
}

#[cfg(test)]
mod wire_finish_reason_tests {
    use super::resolve_wire_finish_reason;
    use crate::ir::FINISH_REASON_TIMEOUT;

    #[test]
    fn stop_string_match_is_stop_not_length() {
        // 2026-09-26: A matched client stop sequence reports "stop" even when the
        // scheduler said "length"; without a match the scheduler's reason passes.
        assert_eq!(
            resolve_wire_finish_reason("length", false, false, true, None),
            "stop"
        );
        assert_eq!(
            resolve_wire_finish_reason("length", false, false, false, None),
            "length"
        );
        assert_eq!(
            resolve_wire_finish_reason("stop", false, false, false, None),
            "stop"
        );
    }

    #[test]
    fn stream_side_guard_cut_reports_repetition() {
        // 2026-10-14: The scheduler reports "stop" for a stream-side guard cut; the reason
        // is "repetition" (wire "stop" + stop_reason "repetition_detected"), not "length".
        for guard in ["simhash_semantic_loop", "token_loop_watchdog"] {
            let r = resolve_wire_finish_reason("stop", false, false, false, Some(guard));
            assert_eq!(r, "repetition", "guard={guard}");
            let f = crate::ir::FinishReason::from(r);
            assert_eq!(
                (f.as_wire(), f.stop_reason()),
                ("stop", Some("repetition_detected"))
            );
        }
    }

    #[test]
    fn stream_guard_does_not_outrank_what_actually_happened() {
        // 2026-09-26: The guard rung ranks below tool calls and the stop match.
        assert_eq!(
            resolve_wire_finish_reason("stop", false, true, false, Some("token_loop_watchdog")),
            "tool_calls"
        );
        assert_eq!(
            resolve_wire_finish_reason("stop", false, false, true, Some("token_loop_watchdog")),
            "stop"
        );
        assert_eq!(
            resolve_wire_finish_reason("stop", false, false, false, None),
            "stop"
        );
    }

    #[test]
    fn timeout_and_tool_overrides_keep_their_rank() {
        assert_eq!(
            resolve_wire_finish_reason(FINISH_REASON_TIMEOUT, true, true, true, None),
            FINISH_REASON_TIMEOUT
        );
        assert_eq!(
            resolve_wire_finish_reason("stop", true, true, true, None),
            "length",
            "tool-loop cap outranks tool_calls and the stop-string match"
        );
        assert_eq!(
            resolve_wire_finish_reason("stop", false, true, true, None),
            "tool_calls",
            "parsed tool calls outrank the stop-string override"
        );
    }
}
