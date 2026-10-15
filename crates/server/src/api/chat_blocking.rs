// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Blocking (non-streaming) `/v1/chat/completions` path. For `n >= 1`
//! choices it runs the scheduler send, decode and tool parse once per choice index.
//!
//! Owner: server chat API.
//! Invariants:
//! - Choices run one after another, one scheduler request each. The first failed
//!   send or inference returns an error response, and no choice is returned.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::AppState;
use crate::ir;
use crate::tool_parser;

use super::chat_blocking_choice::{build_choice_message, build_logprobs};
use super::compact::openai_error_response;
use super::inference_impl::{extract_thinking, strip_stop_sequences};
use super::inference_types::{GrammarSpec, InferenceRequest};

#[cfg(test)]
mod tests;

pub(super) struct BlockingPathArgs {
    pub state: Arc<AppState>,
    pub req: crate::ir::ChatRequest,
    pub req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    pub prompt_tokens: Vec<u32>,
    pub session_hash: u64,
    /// 2026-09-26: Per-request LoRA adapter slot; a negative slot means the active adapter.
    pub adapter_slot: i32,
    /// 2026-09-26: Source-language token id (0 = the deployment default).
    pub src_lang_id: u32,
    /// 2026-09-26: Target-language token id (0 = the deployment default).
    pub tgt_lang_id: u32,
    /// 2026-09-26: NLLB beam search: beams per request (beam search runs only above 1).
    pub num_beams: u32,
    pub length_penalty: f32,
    pub early_stopping: bool,
    pub image_pixels: Vec<metrale_model_layers::VisionItem>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub lz_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub stop_tokens: Vec<u32>,
    pub enable_thinking: bool,
    pub thinking_budget: Option<u32>,
    pub tools_active: bool,
    pub tool_choice_required: bool,
    pub suppress_tool_call: bool,
    pub grammar_spec: Option<GrammarSpec>,
    pub top_logprobs: Option<u8>,
    pub timeout_at: Option<std::time::Instant>,
    pub cwd_hint: Option<String>,
    pub prompt_len: usize,
}

pub(super) async fn run_blocking_path(args: BlockingPathArgs) -> super::chat::ChatOutcome {
    let BlockingPathArgs {
        state,
        req,
        req_ctx,
        prompt_tokens,
        session_hash,
        adapter_slot,
        src_lang_id,
        tgt_lang_id,
        num_beams,
        length_penalty,
        early_stopping,
        image_pixels,
        max_tokens,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        stop_tokens,
        enable_thinking,
        thinking_budget,
        tools_active,
        tool_choice_required,
        suppress_tool_call,
        grammar_spec,
        top_logprobs,
        timeout_at,
        cwd_hint,
        prompt_len,
    } = args;

    let n = req.n.max(1);
    let mut all_choices: Vec<ir::Choice> = Vec::with_capacity(n);
    let mut total_completion_tokens = 0usize;
    let mut first_ttft = 0.0f64;
    let mut last_decode_time_ms = 0.0f64;
    let mut total_reasoning_tokens = 0u32;
    let mut total_cached_prompt_tokens = 0u32;
    let mut total_accepted_prediction_tokens = 0usize;

    let prompt_tokens = std::sync::Arc::new(prompt_tokens);

    for choice_idx in 0..n {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let request = InferenceRequest::Blocking {
            prompt_tokens: prompt_tokens.clone(),
            session_hash,
            cache_salt_hash: metrale_cache::radix_tree::cache_salt_hash(req.cache_salt.as_deref()),
            adapter_slot,
            src_lang_id,
            tgt_lang_id,
            num_beams,
            length_penalty,
            early_stopping,
            image_pixels: if choice_idx == 0 {
                image_pixels.clone()
            } else {
                Vec::new()
            },
            max_tokens,
            min_tokens: req.min_tokens,
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            lz_penalty,
            logit_bias: logit_bias.clone(),
            stop_tokens: stop_tokens.clone(),
            enable_thinking,
            thinking_budget,
            repetition_detection: req.repetition_detection,
            require_tool_call: tool_choice_required,
            tools_present: tools_active,
            suppress_tool_call,
            disable_mtp: false,
            grammar_spec: grammar_spec.clone(),
            seed: req.seed.map(|s| s.wrapping_add(choice_idx as u64)),
            top_logprobs,
            prompt_logprobs: None,
            echo: false,
            timeout_at,
            response_tx: tx,
        };

        if state.request_tx.send(request).await.is_err() {
            return super::chat::ChatOutcome::Http(openai_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Scheduler queue full".to_string(),
            ));
        }

        let response = match rx.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                return super::chat::ChatOutcome::Http(openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Inference error: {e}"),
                ));
            }
            Err(_) => {
                return super::chat::ChatOutcome::Http(openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Inference cancelled".to_string(),
                ));
            }
        };

        if choice_idx == 0 {
            first_ttft = response.time_to_first_token_ms;
        }
        last_decode_time_ms = response.decode_time_ms;

        let num_completion = response.output_tokens.len();
        total_completion_tokens += num_completion;
        total_reasoning_tokens += response.reasoning_tokens;
        total_accepted_prediction_tokens += response.accepted_prediction_tokens;
        // 2026-09-26: Every choice sends the same prompt, so usage reports the largest
        // per-choice prefix-cache hit, not the sum.
        total_cached_prompt_tokens = total_cached_prompt_tokens.max(response.cached_prompt_tokens);

        let (reasoning_content_i, output_text_i) =
            decode_response_text(&state, &response, enable_thinking);
        let (output_text_i, matched_stop) =
            super::inference_impl::strip_stop_sequences_matched(output_text_i, &req.stop);

        let mut choice = build_choice_message(
            &state,
            &req,
            &response,
            reasoning_content_i,
            output_text_i,
            tools_active,
            cwd_hint.as_deref(),
            choice_idx,
        );
        choice.index = choice_idx;
        choice.matched_stop = matched_stop;
        choice.finish_reason =
            stop_match_corrected(choice.finish_reason, choice.matched_stop.is_some());
        choice.logprobs = build_logprobs(&state, &response);
        all_choices.push(choice);
    }

    finalize_response(
        state,
        req_ctx,
        all_choices,
        total_completion_tokens,
        first_ttft,
        last_decode_time_ms,
        total_reasoning_tokens,
        total_cached_prompt_tokens,
        total_accepted_prediction_tokens,
        prompt_len,
    )
}

/// 2026-09-26: Decode `(reasoning_content, output_text)` from the scheduler's response,
/// after `output_tokens_without_stop`. With `enable_thinking` and a think-end token id
/// in the output, split at its first occurrence. Otherwise decode the text and hand it
/// to the model's reasoning parser (`extract_thinking`, called with `enable_thinking`).
/// When the model has a think-end token id, leftover think markers are scrubbed from
/// the content.
fn decode_response_text(
    state: &AppState,
    response: &super::inference_types::InferenceResponse,
    enable_thinking: bool,
) -> (Option<String>, String) {
    let output_tokens =
        output_tokens_without_stop(&response.output_tokens, response.finish_reason.as_str());
    if let Some(think_tok) = state.think_end_token_id {
        if let Some((thinking_tokens, content_tokens)) =
            split_at_first_think_end(output_tokens, think_tok, enable_thinking)
        {
            let reasoning = if !thinking_tokens.is_empty() {
                state
                    .tokenizer
                    .decode(thinking_tokens)
                    .ok()
                    .filter(|s| !s.trim().is_empty())
            } else {
                None
            };
            // 2026-09-26: The split consumes only the first think-end; scrub any later
            // marker from the content, as the streaming path does (`handle_token.rs`).
            let content = super::strip::scrub_think_markers(
                state
                    .tokenizer
                    .decode(content_tokens)
                    .unwrap_or_default()
                    .trim_start(),
            );
            return (reasoning, content);
        }
        // 2026-09-26: No think-end token id in the output (or thinking off). The model
        // can still have closed the block with the marker's text spelling: the
        // scheduler's `PostCloseThinkMask` and `MidWordThinkEndMask` mask the id at some
        // positions. The reasoning parser works on the decoded string.
        let text = state.tokenizer.decode(output_tokens).unwrap_or_default();
        let (reasoning, content) =
            extract_thinking(&text, enable_thinking, state.reasoning_parser.as_deref());
        (reasoning, super::strip::scrub_think_markers(&content))
    } else {
        let text = state.tokenizer.decode(output_tokens).unwrap_or_default();
        extract_thinking(&text, enable_thinking, state.reasoning_parser.as_deref())
    }
}

/// 2026-09-26: Split reasoning from content at the first think-end token; `None` when
/// thinking is off or the token is absent. Later think-end tokens stay in the content.
fn split_at_first_think_end(
    tokens: &[u32],
    think_end_token: u32,
    enable_thinking: bool,
) -> Option<(&[u32], &[u32])> {
    if !enable_thinking {
        return None;
    }

    let pos = tokens.iter().position(|&token| token == think_end_token)?;
    Some((&tokens[..pos], &tokens[pos + 1..]))
}

/// 2026-09-26: With finish reason `"stop"`, drop the last output token before decoding,
/// so a terminal EOS the tokenizer does not mark as special is not decoded into text.
fn output_tokens_without_stop<'a>(tokens: &'a [u32], finish_reason: &str) -> &'a [u32] {
    if finish_reason == "stop" {
        tokens.split_last().map_or(tokens, |(_, visible)| visible)
    } else {
        tokens
    }
}

/// 2026-09-26: Parse tool calls out of the reasoning text with
/// `tool_parser::parse_tool_calls`, returning what is left of the reasoning and the
/// calls. For `poolside_v1` the reasoning is returned unchanged with no calls.
pub(super) fn extract_hoisted_tool_calls(
    reasoning_content: Option<&str>,
    parser_name: Option<&str>,
) -> (Option<String>, Vec<tool_parser::ToolCall>) {
    let Some(reasoning) = reasoning_content else {
        return (None, Vec::new());
    };
    if parser_name == Some("poolside_v1") {
        return (Some(reasoning.to_string()), Vec::new());
    }

    tool_parser::parse_tool_calls(reasoning)
}

/// 2026-09-26: Merge calls recovered from reasoning with calls parsed from content. A
/// reasoning call with the same name and arguments as a content call is dropped in
/// favour of the content copy; repeats within one channel are kept. Reasoning calls
/// come first.
pub(super) fn merge_hoisted_tool_calls(
    mut hoisted: Vec<tool_parser::ToolCall>,
    parsed: Vec<tool_parser::ToolCall>,
) -> Vec<tool_parser::ToolCall> {
    hoisted.retain(|candidate| {
        !parsed.iter().any(|call| {
            call.function.name == candidate.function.name
                && call.function.arguments == candidate.function.arguments
        })
    });
    hoisted.extend(parsed);
    hoisted
}

/// 2026-09-26: Assemble usage, update the metrics, refund the unused rate-limit
/// reservation, and return the response IR; the caller encodes it for the wire.
#[allow(clippy::too_many_arguments)]
fn finalize_response(
    state: Arc<AppState>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    all_choices: Vec<ir::Choice>,
    total_completion_tokens: usize,
    first_ttft: f64,
    last_decode_time_ms: f64,
    total_reasoning_tokens: u32,
    total_cached_prompt_tokens: u32,
    total_accepted_prediction_tokens: usize,
    prompt_len: usize,
) -> super::chat::ChatOutcome {
    let usage = ir::Usage {
        prompt_tokens: prompt_len,
        completion_tokens: total_completion_tokens,
        cached_prompt_tokens: total_cached_prompt_tokens as usize,
        reasoning_tokens: total_reasoning_tokens as usize,
        accepted_prediction_tokens: total_accepted_prediction_tokens,
        time_to_first_token_ms: first_ttft,
        // 2026-09-26: The last choice's decode window, the same one the rate below uses.
        decode_time_ms: last_decode_time_ms,
        response_tokens_per_second: ir::Usage::decode_rate_tok_s(
            total_completion_tokens,
            last_decode_time_ms,
        ),
    };

    // 2026-09-26: REQUESTS_ACTIVE is released by the caller's `ActiveRequestGuard`.
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(total_completion_tokens as u64);
    crate::metrics::TTFT_SECONDS
        .with_label_values(&[state.model_name.as_str()])
        .observe(first_ttft / 1000.0);

    // 2026-09-26: The rate-limit middleware reserved `max_seq_len` tokens; refund what
    // prompt and completion did not use.
    if let Some(axum::extract::Extension(ref ctx)) = req_ctx {
        let actual = (prompt_len + total_completion_tokens) as u64;
        let refund = ctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            state.rate_limiter.refund_tokens(&ctx.identity, refund);
        }
    }

    super::chat::ChatOutcome::Blocking(Box::new(ir::ChatResponse {
        id: crate::ids::uuid_v4(),
        model: state.model_name.clone(),
        created: crate::ids::unix_timestamp(),
        choices: all_choices,
        usage,
    }))
}

/// 2026-09-26: A response ended by a client stop sequence reports `"stop"`, not
/// `"length"`. The blocking path finds stop strings afterwards (the caller's suffix
/// strip), so only `Length` is rewritten; `"timeout"` and `"tool_calls"` outrank a stop
/// match, as in `chat_stream::handle_done::resolve_wire_finish_reason`.
fn stop_match_corrected(fr: ir::FinishReason, stop_matched: bool) -> ir::FinishReason {
    if stop_matched && fr == ir::FinishReason::Length {
        ir::FinishReason::Stop
    } else {
        fr
    }
}

#[cfg(test)]
mod stop_match_corrected_tests {
    use super::stop_match_corrected;
    use crate::ir::{FINISH_REASON_TIMEOUT, FinishReason};

    #[test]
    fn matched_stop_corrects_length_to_stop() {
        assert_eq!(
            stop_match_corrected(FinishReason::Length, true),
            FinishReason::Stop
        );
    }

    #[test]
    fn everything_else_passes_through() {
        assert_eq!(
            stop_match_corrected(FinishReason::Length, false),
            FinishReason::Length
        );
        assert_eq!(
            stop_match_corrected(FinishReason::ToolCalls, true),
            FinishReason::ToolCalls
        );
        assert_eq!(
            stop_match_corrected(FinishReason::Other(FINISH_REASON_TIMEOUT.into()), true),
            FinishReason::Other(FINISH_REASON_TIMEOUT.into())
        );
    }
}
