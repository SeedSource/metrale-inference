// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused_imports, dead_code)]

//! 2026-09-26: Streaming `/v1/chat/completions`: send one streaming request to the
//! scheduler and map its events to provider-neutral deltas (`ir::StreamDelta`); the
//! caller's surface encoder writes the wire format.
//!
//! Sub-files:
//! - `state`         — `StreamState`, the per-stream mutable state
//! - `ctx`           — `StreamCtx`, the per-stream read-only context
//! - `handle_token`  — the `Token` / `TokenWithLogprobs` arm
//! - `handle_done`   — the `Done` arm (flush, salvage, usage, dump, metrics)
//! - `handle_error`  — the `Error` arm
//! - `tool_handlers` — the streaming tool-call detector's outputs, for both arms
//! - `strip`         — boundary-preserving strip helpers for `handle_token`
//! - `token_ids`     — the `return_token_ids` drain, for both arms
//!
//! Owner: server streaming API.
//! Invariants:
//! - The scheduler channel is read through `stream_terminal::terminated`, so a sender
//!   dropped without `Done` or `Error` still produces an `Error` event.

#[cfg(test)]
mod cancel_guard_tests;
mod ctx;
mod handle_done;
mod handle_error;
mod handle_token;
mod state;
mod strip;
mod token_ids;
mod tool_handlers;

use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::tool_parser;

use super::inference_types::{GrammarSpec, InferenceRequest, StreamEvent};

use ctx::StreamCtx;
use state::StreamState;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_chat_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    session_hash: u64,
    // 2026-10-14: `cache_salt_hash` of the request's `cache_salt`; 0 = none.
    cache_salt_hash: u64,
    // 2026-09-26: A negative slot means the active LoRA adapter.
    adapter_slot: i32,
    // 2026-09-26: 0 means the deployment default, for both language ids.
    src_lang_id: u32,
    tgt_lang_id: u32,
    // 2026-09-26: `chat::mod.rs` rejects `num_beams > 1` on a streaming request, so
    // beam search never runs here.
    num_beams: u32,
    length_penalty: f32,
    early_stopping: bool,
    image_pixels: Vec<metrale_model_layers::VisionItem>,
    max_tokens: usize,
    min_tokens: usize,
    ignore_eos: bool,
    temperature: f32,
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
    lz_penalty: f32,
    logit_bias: Vec<(u32, f32)>,
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    repetition_detection: Option<crate::api::inference_types::RepetitionDetectionParams>,
    tools_active: bool,
    tool_choice_required: bool,
    suppress_tool_call: bool,
    tool_defs: Vec<tool_parser::ToolDefinition>,
    cwd_hint: Option<String>,
    stop_tokens: Vec<u32>,
    grammar_spec: Option<GrammarSpec>,
    seed: Option<u64>,
    top_logprobs: Option<u8>,
    timeout_at: Option<std::time::Instant>,
    stop_strings: Vec<String>,
    req_return_token_ids: bool,
    req_ctx: Option<crate::rate_limiter::RequestContext>,
    dump_seq: Option<u64>,
    active_guard: crate::metrics::ActiveRequestGuard,
) -> Result<crate::ir::DeltaStream, (StatusCode, String)> {
    // 2026-09-26: The scheduler thread sends with `bounded_stream_send`, which gives up
    // on a full channel after `METRALE_STREAM_SEND_DEADLINE_MS` (5000 ms when unset);
    // 1024 events of buffer ride out a client that reads late.
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
    let prompt_len = prompt_tokens.len();
    // 2026-09-26: Cooperative cancellation shared with the scheduler. The stream-side
    // guards set it (`cancel_guard_tests` lists the sites), and the scheduler finishes
    // the sequence at its next token instead of generating on with output suppressed.
    let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let scheduler_thinking = enable_thinking;
    let prompt_tokens = std::sync::Arc::new(prompt_tokens);
    let prompt_tokens_for_retry = prompt_tokens.clone();
    let grammar_spec_for_retry = grammar_spec.clone();
    let request = InferenceRequest::Streaming {
        prompt_tokens,
        session_hash,
        cache_salt_hash,
        adapter_slot,
        src_lang_id,
        tgt_lang_id,
        num_beams,
        length_penalty,
        early_stopping,
        image_pixels,
        max_tokens,
        min_tokens,
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
        enable_thinking: scheduler_thinking,
        thinking_budget,
        repetition_detection,
        require_tool_call: tool_choice_required,
        tools_present: tools_active,
        suppress_tool_call,
        disable_mtp: false,
        grammar_spec,
        seed,
        top_logprobs,
        prompt_logprobs: None,
        echo: false,
        timeout_at,
        token_tx,
        cancel_flag: cancel_flag.clone(),
    };

    state.request_tx.send(request).await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler queue full".to_string(),
        )
    })?;

    // 2026-09-26: `ctx.id` and `ctx.model` feed the `--dump` record and log lines; the
    // wire ids come from the surface encoders.
    let chunk_id = format!("chatcmpl-{}", crate::ids::uuid_v4());
    let model_name = state.model_name.clone();

    // 2026-09-26: With no tool parser the markers are empty, and
    // `sanitize_content_chunk` passes text through unchanged.
    let leak_markers: tool_parser::LeakMarkers = state
        .tool_call_parser
        .as_ref()
        .map(|p| p.leak_markers())
        .unwrap_or(tool_parser::LeakMarkers::EMPTY);

    let max_tool_calls_per_response: usize = std::env::var("METRALE_MAX_TOOL_CALLS_PER_RESPONSE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    // 2026-09-26: Hold back the longest stop string's byte length minus one, so a stop
    // string split across two decoded chunks is not sent in part; 0 without stop
    // strings.
    let stop_string_buffer_len: usize = stop_strings
        .iter()
        .map(|s| s.len())
        .max()
        .map(|m| m.saturating_sub(1))
        .unwrap_or(0);

    let prompt_vocab: Arc<std::collections::HashSet<String>> =
        Arc::new(std::collections::HashSet::new());

    let ctx = StreamCtx {
        state: state.clone(),
        model: model_name.clone(),
        id: chunk_id.clone(),
        prompt_len,
        enable_thinking,
        tool_defs_for_backfill: tool_defs,
        cwd_for_normalize: cwd_hint,
        stop_strings,
        stop_string_buffer_len,
        leak_markers,
        wants_typed_arguments: state
            .tool_call_parser
            .as_ref()
            .is_some_and(|p| p.wants_typed_arguments()),
        max_tool_calls_per_response,
        req_return_token_ids,
        req_ctx,
        dump_seq,
        ignore_eos,
        min_tokens,
        tool_retry_enabled: false,
        prompt_tokens: prompt_tokens_for_retry,
        prompt_vocab,
        grammar_spec: grammar_spec_for_retry,
        max_tokens,
        timeout_at,
        _active_guard: active_guard,
    };

    let mut stream_state = StreamState::new(
        tools_active,
        enable_thinking,
        cancel_flag.clone(),
        ctx.tool_defs_for_backfill.clone(),
    );
    if let Some(detector) = stream_state.detector.as_mut() {
        // 2026-09-26: Only `poolside_v1` promotes a bare name inside `<tool_call>` to a
        // zero-argument call; for every other parser that shape is not a call.
        detector.set_promote_bare_names(
            state
                .tool_call_parser
                .as_ref()
                .is_some_and(|p| p.promotes_bare_call_names()),
        );
    }

    // 2026-09-26: `terminated`: a stream the scheduler drops without Done/Error still
    // ends in an error frame (`stream_terminal.rs`).
    let events = super::stream_terminal::terminated(ReceiverStream::new(token_rx));
    let token_stream = events.flat_map(move |event| {
        use futures::StreamExt;
        let deltas = match event {
            StreamEvent::Token(tok) | StreamEvent::TokenWithLogprobs(tok, _) => {
                handle_token::handle_token(&mut stream_state, &ctx, tok)
            }
            // 2026-09-26: The request sets `prompt_logprobs: None`; nothing to emit.
            StreamEvent::PromptLogprobs(_) => Vec::new(),
            StreamEvent::Done {
                finish_reason,
                prompt_tokens: _,
                completion_tokens,
                time_to_first_token_ms,
                decode_time_ms,
                reasoning_tokens,
                cached_prompt_tokens,
                accepted_prediction_tokens,
                guard_stop,
            } => {
                // 2026-09-26: `.or()`: the stream-side guards in `handle_token.rs` may have
                // set `guard_stop` before this frame arrives; the scheduler's value fills
                // it only when unset.
                stream_state.guard_stop = stream_state.guard_stop.or(guard_stop);
                handle_done::handle_done(
                    &mut stream_state,
                    &ctx,
                    finish_reason,
                    completion_tokens,
                    time_to_first_token_ms,
                    decode_time_ms,
                    reasoning_tokens,
                    cached_prompt_tokens,
                    accepted_prediction_tokens,
                )
            }
            StreamEvent::Error(msg) => handle_error::handle_error(&ctx, msg),
        };

        futures::stream::iter(deltas).boxed()
    });

    Ok(Box::pin(token_stream))
}
