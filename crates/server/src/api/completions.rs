// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `/v1/completions`: request validation and parameter resolution,
//! the streaming path, and the shared 501 `not_supported` response.
//!
//! Owner: server completions API.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code)]

use crate::main_modules::model_host::CurrentModel;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, PromptInput, Usage,
};
use crate::tool_parser;

use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};
use super::strip::strip_thinking_tags;

use super::inference_types::*;
use super::sanitizer::*;

#[cfg(test)]
mod min_tokens_wiring_tests;
#[cfg(test)]
mod cache_salt_wiring_tests;
mod prompts;

use prompts::resolve_prompts;

#[cfg(test)]
mod prompt_logprobs_refusal_tests;

/// 2026-10-17 (A191): Why a `/v1/completions` request for prompt logprobs (`echo` with
/// `logprobs`) is refused on a serve of more than one rank, else `None`. The prompt-logprob
/// windows (`prefill_b/prompt_logprobs.rs`) run on rank 0 only: rank 1's prefill command does
/// not carry the request. A window of at most 16 rows takes the vocab-split tensor-core LM head,
/// whose all-reduce rank 1 never joins, so the serve hangs (A191, 2026-10-15: a 333-token
/// prompt, 12-row last window). Interim until rank 1 mirrors the windows.
pub(crate) fn multi_rank_prompt_logprobs_refusal(
    world_size: usize,
    echo: bool,
    logprobs: bool,
) -> Option<String> {
    (world_size > 1 && echo && logprobs).then(|| {
        format!(
            "echo with logprobs (prompt logprobs) is not supported on a multi-rank serve \
             (world_size={world_size}): the prompt-logprob pass runs on rank 0 only and can \
             hang the serve. Send echo without logprobs, or logprobs without echo."
        )
    })
}

pub async fn completions(
    CurrentModel(state): CurrentModel,
    req: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    if let Some(msg) =
        multi_rank_prompt_logprobs_refusal(state.world_size, req.echo, req.logprobs.is_some())
    {
        return openai_error_response(StatusCode::BAD_REQUEST, msg);
    }
    let prompts = match resolve_prompts(&state, &req.prompt) {
        Ok(t) => t,
        Err((status, msg)) => return openai_error_response(status, msg),
    };
    if prompts.is_empty() {
        return openai_error_response(StatusCode::BAD_REQUEST, "Empty prompt".to_string());
    }
    for prompt_tokens in &prompts {
        let prompt_len = prompt_tokens.len();
        if prompt_len >= state.max_seq_len {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "Prompt too long: {prompt_len} tokens exceeds max_seq_len {}",
                    state.max_seq_len
                ),
            );
        }
    }

    // 2026-09-26: The same ranges as the chat path (`chat_phases.rs`
    // `validate_input`, `chat/sampling_setup.rs`); out of range is a 400.
    if let Some(t) = req.temperature
        && !(0.0..=2.0).contains(&t)
    {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("temperature must be between 0 and 2, got {t}"),
        );
    }
    if let Some(pp) = req.presence_penalty
        && !(-2.0..=2.0).contains(&pp)
    {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("presence_penalty must be between -2 and 2, got {pp}"),
        );
    }
    if let Some(fp) = req.frequency_penalty
        && !(-2.0..=2.0).contains(&fp)
    {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("frequency_penalty must be between -2 and 2, got {fp}"),
        );
    }

    let temperature = req.temperature.unwrap_or(state.default_temperature);
    let top_k = req.top_k.unwrap_or(state.default_top_k);
    let top_p = req.top_p.unwrap_or(state.default_top_p);
    let top_n_sigma = req.top_n_sigma.unwrap_or(state.default_top_n_sigma);
    let min_p = req.min_p.unwrap_or(state.default_min_p);
    let repetition_penalty = req
        .repetition_penalty
        .unwrap_or(state.sampling_presets.non_thinking.repetition_penalty);
    let presence_penalty = req.presence_penalty.unwrap_or(0.0);
    let frequency_penalty = req.frequency_penalty.unwrap_or(0.0);
    // 2026-09-26: `logit_bias` keys are token-ID strings; a key that does not
    // parse as `u32` is dropped.
    let logit_bias: Vec<(u32, f32)> = req.logit_bias.as_ref().map_or(Vec::new(), |map| {
        map.iter()
            .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
            .collect()
    });
    // 2026-09-26: `n` bounds the sequential inference loop in
    // `completions_exec::run_blocking`, so it is limited to 1..=128.
    if req.n == 0 || req.n > 128 {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("n must be between 1 and 128, got {}", req.n),
        );
    }
    let stop_tokens = tokenize_stop_sequences(&state.tokenizer, &req.stop);
    let logprobs_k = req.logprobs.map(|k| k.min(20));

    let adapter_slot = match super::lora_control::resolve_request_adapter_slot(
        &state,
        req.adapter.as_deref(),
        &req.model,
    )
    .await
    {
        Ok(slot) => slot,
        Err(resp) => return resp,
    };

    // 2026-09-26: Source/target language token names resolve to token IDs;
    // absent is 0 (the deployment default) and an unknown token is a 400.
    let resolve_lang = |name: &Option<String>| -> Result<u32, Response> {
        match name {
            None => Ok(0),
            Some(s) => state.tokenizer.inner().token_to_id(s).ok_or_else(|| {
                openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("unknown language token '{s}'"),
                )
            }),
        }
    };
    let src_lang_id = match resolve_lang(&req.src_lang) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let tgt_lang_id = match resolve_lang(&req.tgt_lang) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let num_beams = req.num_beams.unwrap_or(1);
    let length_penalty = req.length_penalty.unwrap_or(1.0);
    let early_stopping = req.early_stopping.unwrap_or(false);

    let params = super::completions_exec::CompletionParams {
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        logit_bias,
        stop_tokens,
        repetition_detection: req.repetition_detection,
        logprobs_k,
        adapter_slot,
        src_lang_id,
        tgt_lang_id,
        num_beams,
        length_penalty,
        early_stopping,
    };

    if req.stream {
        if prompts.len() > 1 || req.n > 1 {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                "stream=true supports a single prompt with n=1".to_string(),
            );
        }
        if num_beams > 1 {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                "num_beams > 1 is not supported in streaming mode".to_string(),
            );
        }
        let prompt_tokens = prompts.into_iter().next().expect("checked non-empty");
        return match completions_stream(state, prompt_tokens, req, params).await {
            Ok(r) => r,
            Err((status, msg)) => openai_error_response(status, msg),
        };
    }

    super::completions_exec::run_blocking(state, &req, prompts, params).await
}

/// 2026-09-26: Build the streaming `InferenceRequest` for `/v1/completions`.
/// Separate from [`completions_stream`] so tests can build it without a model.
fn build_streaming_request(
    req: &CompletionRequest,
    p: super::completions_exec::CompletionParams,
    prompt_tokens: std::sync::Arc<Vec<u32>>,
    session_hash: u64,
    timeout_at: Option<std::time::Instant>,
    token_tx: tokio::sync::mpsc::Sender<StreamEvent>,
) -> InferenceRequest {
    let echo = req.echo;
    let logprobs_k = p.logprobs_k;
    InferenceRequest::Streaming {
        prompt_tokens,
        session_hash,
        cache_salt_hash: metrale_cache::radix_tree::cache_salt_hash(req.cache_salt.as_deref()),
        adapter_slot: p.adapter_slot,
        src_lang_id: p.src_lang_id,
        tgt_lang_id: p.tgt_lang_id,
        num_beams: p.num_beams,
        length_penalty: p.length_penalty,
        early_stopping: p.early_stopping,
        image_pixels: Vec::new(),
        max_tokens: req.max_tokens,
        min_tokens: req.min_tokens,
        temperature: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        top_n_sigma: p.top_n_sigma,
        min_p: p.min_p,
        repetition_penalty: p.repetition_penalty,
        presence_penalty: p.presence_penalty,
        frequency_penalty: p.frequency_penalty,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias: p.logit_bias,
        stop_tokens: p.stop_tokens,
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: p.repetition_detection,
        require_tool_call: false,
        tools_present: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_spec: None,
        seed: req.seed,
        top_logprobs: logprobs_k,
        prompt_logprobs: if echo { logprobs_k } else { None },
        echo,
        timeout_at,
        token_tx,
        // 2026-09-26: The handler keeps no handle to this flag, so it never
        // cancels the request.
        cancel_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    }
}

/// 2026-09-26: SSE path for `/v1/completions` (one prompt, `n = 1`, checked by
/// [`completions`]). With `echo`, the prompt text (with its logprobs when
/// requested) is the first chunk; with `stream_options.include_usage`, a
/// `choices: []` usage chunk precedes `[DONE]`.
pub(super) async fn completions_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    req: CompletionRequest,
    p: super::completions_exec::CompletionParams,
) -> Result<Response, (StatusCode, String)> {
    // 2026-09-26: Same capacity as the chat stream's channel (`chat_stream/mod.rs`).
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
    let prompt_len = prompt_tokens.len();
    let echo = req.echo;
    let logprobs_k = p.logprobs_k;
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    // 2026-09-26: Echo needs the prompt tokens after the request takes them.
    let echo_prompt = if echo {
        Some(prompt_tokens.clone())
    } else {
        None
    };
    // 2026-09-26: Echo without logprobs gets no `PromptLogprobs` event, so the
    // prompt text is decoded here and sent as the first chunk (`echo_prefix`).
    let echo_only_text = if echo && logprobs_k.is_none() {
        Some(state.tokenizer.decode(&prompt_tokens).unwrap_or_default())
    } else {
        None
    };

    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let request = build_streaming_request(
        &req,
        p,
        std::sync::Arc::new(prompt_tokens),
        session_hash,
        state.request_deadline(None),
        token_tx,
    );

    state.request_tx.send(request).await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler queue full".to_string(),
        )
    })?;

    let chunk_id = crate::openai::new_completion_id();
    let model_name = state.model_name.clone();

    let model = model_name.clone();
    let id = chunk_id.clone();
    let mut all_toks: Vec<u32> = Vec::new();
    let mut emitted: usize = 0;
    // 2026-09-26: Incremental-detokenizer state for
    // `ChatTokenizer::incremental_decode`, which decodes a bounded suffix of
    // `all_toks` per token.
    let mut content_decoded = String::new();
    let mut detok_prefix_offset: usize = 0;
    let mut detok_read_offset: usize = 0;
    // 2026-09-26: `terminated` adds an `Error` event if the channel closes
    // without `Done` or `Error` (`stream_terminal.rs`).
    let events = super::stream_terminal::terminated(ReceiverStream::new(token_rx));
    let token_stream = events.flat_map(move |event| {
        let events: Vec<Result<Event, std::convert::Infallible>> = match event {
            // 2026-09-26: Echo with logprobs: the scheduler sends this before
            // the first generated token (`scheduler/prefill_a_step.rs`).
            StreamEvent::PromptLogprobs(lps) => {
                let prompt_toks = echo_prompt.clone().unwrap_or_default();
                let text = state.tokenizer.decode(&prompt_toks).unwrap_or_default();
                let decode = |tid: u32| state.tokenizer.decode(&[tid]).unwrap_or_default();
                let lp = super::completions_logprobs::build_completion_logprobs(
                    &decode,
                    true,
                    &prompt_toks,
                    &lps,
                    &[],
                    &[],
                );
                let chunk = CompletionChunk::echo_chunk(&model, &id, text, Some(lp));
                vec![Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                )]
            }
            StreamEvent::Token(tok) | StreamEvent::TokenWithLogprobs(tok, _) => {
                all_toks.push(tok);
                content_decoded.push_str(&state.tokenizer.incremental_decode(
                    &all_toks,
                    &mut detok_prefix_offset,
                    &mut detok_read_offset,
                ));
                let stable_end = content_decoded.len();
                let delta = if stable_end <= emitted {
                    String::new()
                } else {
                    let d = content_decoded[emitted..stable_end].to_string();
                    emitted = stable_end;
                    d
                };
                let chunk = CompletionChunk::text_chunk(&model, &id, delta);
                vec![Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                )]
            }
            StreamEvent::Done {
                finish_reason,
                prompt_tokens: _,
                completion_tokens,
                time_to_first_token_ms,
                decode_time_ms,
                reasoning_tokens,
                cached_prompt_tokens,
                accepted_prediction_tokens,
                guard_stop: _,
            } => {
                let tps = crate::ir::Usage::decode_rate_tok_s(completion_tokens, decode_time_ms);
                let usage = Usage {
                    prompt_tokens: prompt_len,
                    completion_tokens,
                    total_tokens: prompt_len + completion_tokens,
                    prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
                        cached_tokens: cached_prompt_tokens as usize,
                        audio_tokens: 0,
                    }),
                    completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
                        reasoning_tokens: reasoning_tokens as usize,
                        audio_tokens: 0,
                        accepted_prediction_tokens,
                        rejected_prediction_tokens: 0,
                    }),
                    time_to_first_token_ms,
                    response_tokens_per_second: tps,
                    decode_time_ms,
                    total_time_ms: time_to_first_token_ms + decode_time_ms,
                };
                if include_usage {
                    // 2026-09-26: A finish chunk without usage, then a
                    // usage-only chunk with `choices: []`.
                    let fin = CompletionChunk::finish_chunk_no_usage(&model, &id, &finish_reason);
                    let usage_chunk = CompletionChunk::usage_only_chunk(&model, &id, usage);
                    vec![
                        Ok(Event::default().data(serde_json::to_string(&fin).unwrap_or_default())),
                        Ok(Event::default()
                            .data(serde_json::to_string(&usage_chunk).unwrap_or_default())),
                    ]
                } else {
                    let chunk = CompletionChunk::done_chunk(&model, &id, &finish_reason, usage);
                    vec![Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                    )]
                }
            }
            StreamEvent::Error(msg) => vec![Ok(
                Event::default().data(super::compact::completion_error_frame(&msg))
            )],
        };
        futures::stream::iter(events)
    });

    let echo_prefix: Option<Event> = echo_only_text.map(|text| {
        let chunk = CompletionChunk::echo_chunk(&model_name, &chunk_id, text, None);
        Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
    });

    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    let prefix = futures::stream::iter(
        echo_prefix
            .into_iter()
            .map(Ok::<_, std::convert::Infallible>),
    );
    let full_stream = prefix.chain(token_stream).chain(done_event);

    Ok(Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// 2026-09-26: 501 "not supported" response (`error.type = server_error`) for
/// endpoints this server does not implement (`stubs.rs` and others).
pub(super) fn not_supported(message: &'static str) -> Response {
    openai_error_response(StatusCode::NOT_IMPLEMENTED, message.into())
}
