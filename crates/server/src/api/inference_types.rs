// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The types passed between the API handlers and the scheduler:
//! `InferenceRequest`, `InferenceResponse`, `StreamEvent`, `TokenLogprobs`,
//! `GrammarSpec` and `RepetitionDetectionParams`.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code)]

use axum::extract::State;
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
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

/// 2026-09-26: Per-request loop-detector parameters: the smallest and largest
/// repeating pattern, in tokens (`min_pattern_size`, `max_pattern_size`), and
/// the repeat count that makes a loop (`min_count`). When set, they replace the
/// built-in period range and repeat threshold of the thinking-loop and
/// content-loop detectors for that sequence (`scheduler/helpers/detectors.rs`).
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RepetitionDetectionParams {
    pub min_pattern_size: u32,
    pub max_pattern_size: u32,
    pub min_count: u32,
}

/// 2026-09-26: Grammar for constrained decoding: a tool-call grammar, or a
/// `response_format` grammar (any JSON, or JSON matching a schema).
#[derive(Clone)]
pub enum GrammarSpec {
    /// 2026-09-26: Tool-call grammar. The scheduler compiles it with
    /// `parser.compile_tool_grammar(engine, tools, use_triggers)`
    /// (`scheduler/emit_step/grammar_close.rs`), so the parser that scans the
    /// response also defines the grammar.
    ToolCall {
        tools: Vec<tool_parser::ToolDefinition>,
        parser: std::sync::Arc<dyn tool_parser::ToolCallParser>,
        use_triggers: bool,
    },
    /// 2026-09-26: `response_format` `json_object`: any valid JSON.
    JsonObject,
    /// 2026-09-26: `response_format` `json_schema`: JSON matching `schema`.
    JsonSchema { schema: String },
}

/// 2026-09-26: A request submitted to the scheduler.
pub enum InferenceRequest {
    /// 2026-09-26: The whole response is sent once, on `response_tx`.
    Blocking {
        prompt_tokens: std::sync::Arc<Vec<u32>>,
        /// 2026-09-26: `session_manager::compute_session_hash` of the prompt
        /// (its first 1024 tokens).
        session_hash: u64,
        /// 2026-10-14: `metrale_cache::radix_tree::cache_salt_hash` of the request's
        /// `cache_salt`; 0 = no salt. The scheduler folds it into `seq.adapter_id`.
        cache_salt_hash: u64,
        /// 2026-09-26: LoRA adapter slot; `-1` follows the active adapter.
        adapter_slot: i32,
        /// 2026-09-26: Source-language token id; 0 means the deployment default.
        src_lang_id: u32,
        /// 2026-09-26: Target-language token id; 0 means the deployment default.
        tgt_lang_id: u32,
        /// 2026-09-26: NLLB beam count; above 1 selects beam search.
        num_beams: u32,
        length_penalty: f32,
        early_stopping: bool,
        /// 2026-09-26: Preprocessed images and videos, one `VisionItem` each.
        image_pixels: Vec<metrale_model_layers::VisionItem>,
        max_tokens: usize,
        /// 2026-09-26: EOS ids are masked while the output is shorter than this.
        min_tokens: usize,
        temperature: f32,
        /// 2026-09-26: Top-k; 0 disables it.
        top_k: u32,
        /// 2026-09-26: Top-p (nucleus); 1.0 disables it.
        top_p: f32,
        /// 2026-09-26: Top-n-sigma; 0.0 disables it.
        top_n_sigma: f32,
        /// 2026-09-26: Min-p: keep tokens with prob >= min_p * max_prob; 0.0
        /// disables it.
        min_p: f32,
        /// 2026-09-26: Repetition penalty; 1.0 disables it.
        repetition_penalty: f32,
        /// 2026-09-26: Additive presence penalty; 0.0 disables it.
        presence_penalty: f32,
        /// 2026-09-26: Additive frequency penalty; 0.0 disables it.
        frequency_penalty: f32,
        /// 2026-09-26: DRY multiplier (0.0 turns DRY off). Chat takes the DRY
        /// and LZ values from the resolved MODEL.toml sampling preset
        /// (`chat/sampling_setup.rs`); `/v1/completions` sends 0.0.
        dry_multiplier: f32,
        dry_base: f32,
        dry_allowed_length: u32,
        /// 2026-09-26: Strength of `metrale_sampling::apply_lz_penalty`, an
        /// n-gram repeat penalty over the last 256 tokens; 0.0 disables it.
        lz_penalty: f32,
        /// 2026-09-26: `(token_id, bias)` pairs.
        logit_bias: Vec<(u32, f32)>,
        /// 2026-09-26: Stop strings that are one token each
        /// (`tokenize_stop_sequences`).
        stop_tokens: Vec<u32>,
        enable_thinking: bool,
        /// 2026-09-26: Thinking-token count at which the scheduler arms a forced
        /// `</think>` (`scheduler/decode_logits_step/per_token.rs`).
        thinking_budget: Option<u32>,
        /// 2026-09-26: Per-request loop-detector parameters; `None` uses the
        /// server's.
        repetition_detection: Option<RepetitionDetectionParams>,
        /// 2026-09-26: Set by the chat handler from
        /// `tool_choice_required_for_parser`.
        require_tool_call: bool,
        /// 2026-09-26: The chat handler's `tools_active`: a tool-call parser
        /// is configured, `tools` is non-empty and `tool_choice` is not
        /// `none` (`chat/prepare.rs`). It does not depend on `grammar_spec`.
        /// When true, a `</tool_call>` outside thinking does not end the turn,
        /// so the model can emit further calls
        /// (`scheduler/decode_logits_step/per_token.rs`).
        tools_present: bool,
        /// 2026-09-26: Set by the chat tool-loop detector (`chat/loop_detect.rs`)
        /// to push the `<tool_call>` logit down.
        suppress_tool_call: bool,
        /// 2026-09-26: When true, the sequence is not eligible for speculative
        /// decode (`mtp_gate::spec_dispatch_eligible`). Every API handler
        /// sets `false`.
        disable_mtp: bool,
        grammar_spec: Option<GrammarSpec>,
        /// 2026-09-26: Sampling seed, advanced by one per generated token
        /// (`scheduler/decode_logits_seq.rs`); `None` leaves sampling unseeded.
        seed: Option<u64>,
        /// 2026-09-26: Alternatives per generated token; `None` returns no logprobs.
        top_logprobs: Option<u8>,
        /// 2026-09-26: `/v1/completions` with `echo`: collect prompt-token
        /// logprobs with `k` alternatives during prefill. Such a request skips
        /// the prefix cache, so every position is computed
        /// (`model-engine .../prefill_b/prefix_lookup.rs`).
        prompt_logprobs: Option<u8>,
        /// 2026-09-26: `/v1/completions` `echo`. The handler prepends the
        /// prompt; the scheduler does not read this field.
        echo: bool,
        /// 2026-09-26: Absolute deadline; `None` means none.
        timeout_at: Option<std::time::Instant>,
        response_tx: tokio::sync::oneshot::Sender<anyhow::Result<InferenceResponse>>,
    },
    /// 2026-09-26: Events are sent on `token_tx` as they are generated. The
    /// fields shared with `Blocking` mean the same.
    Streaming {
        prompt_tokens: std::sync::Arc<Vec<u32>>,
        session_hash: u64,
        cache_salt_hash: u64,
        adapter_slot: i32,
        src_lang_id: u32,
        tgt_lang_id: u32,
        num_beams: u32,
        length_penalty: f32,
        early_stopping: bool,
        image_pixels: Vec<metrale_model_layers::VisionItem>,
        max_tokens: usize,
        min_tokens: usize,
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
        stop_tokens: Vec<u32>,
        enable_thinking: bool,
        thinking_budget: Option<u32>,
        repetition_detection: Option<RepetitionDetectionParams>,
        require_tool_call: bool,
        tools_present: bool,
        suppress_tool_call: bool,
        disable_mtp: bool,
        grammar_spec: Option<GrammarSpec>,
        seed: Option<u64>,
        top_logprobs: Option<u8>,
        prompt_logprobs: Option<u8>,
        echo: bool,
        timeout_at: Option<std::time::Instant>,
        token_tx: tokio::sync::mpsc::Sender<StreamEvent>,
        /// 2026-09-26: Cancel flag shared with the stream. The chat stream's
        /// guards set it; the scheduler checks it for each emitted token and
        /// finishes the sequence (`scheduler/emit_step/token.rs`), so a
        /// guarded stream stops generating instead of only hiding output.
        cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    },
}

/// 2026-09-26: Logprobs for one token position.
#[derive(Clone)]
pub struct TokenLogprobs {
    pub token_id: u32,
    pub logprob: f32,
    /// 2026-09-26: The top-k alternatives, descending by logprob
    /// (`scheduler/logprobs.rs`).
    pub top: Vec<(u32, f32)>,
}

/// 2026-09-26: The scheduler's response to a `Blocking` request.
pub struct InferenceResponse {
    pub output_tokens: Vec<u32>,
    pub finish_reason: String,
    pub time_to_first_token_ms: f64,
    pub decode_time_ms: f64,
    /// 2026-09-26: Per generated token; empty unless `top_logprobs` was set.
    pub logprobs: Vec<TokenLogprobs>,
    /// 2026-09-26: Thinking tokens (`ActiveSeq::thinking_tokens`), reported as
    /// `usage.completion_tokens_details.reasoning_tokens`.
    pub reasoning_tokens: u32,
    /// 2026-09-26: Prompt tokens served by the prefix cache, reported as
    /// `usage.prompt_tokens_details.cached_tokens`.
    pub cached_prompt_tokens: u32,
    /// 2026-09-26: Speculative draft tokens accepted for this request
    /// (`RequestAccept::accepted_total`), reported as
    /// `usage.completion_tokens_details.accepted_prediction_tokens`.
    pub accepted_prediction_tokens: usize,
    /// 2026-09-26: Prompt-token logprobs, empty unless `prompt_logprobs` was
    /// set: entry `i` scores prompt token `i + 1`, and the last prompt position
    /// is excluded. `completions_logprobs` gives the first prompt token `null`.
    pub prompt_logprobs: Vec<TokenLogprobs>,
}

/// 2026-09-26: Events sent to a `Streaming` request.
pub enum StreamEvent {
    Token(u32),
    TokenWithLogprobs(u32, TokenLogprobs),
    /// 2026-09-26: Prompt-token logprobs, sent before the first generated
    /// token when `prompt_logprobs` is set (`scheduler/prefill_a_step.rs`).
    PromptLogprobs(Vec<TokenLogprobs>),
    Done {
        finish_reason: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        time_to_first_token_ms: f64,
        decode_time_ms: f64,
        reasoning_tokens: u32,
        cached_prompt_tokens: u32,
        accepted_prediction_tokens: usize,
        /// 2026-09-25: The scheduler guard that force-finished the sequence,
        /// if any. The chat stream reports it as `finish_reason` `length` when
        /// no higher rung of `resolve_wire_finish_reason` applies (a timeout,
        /// a tool-loop cap, tool calls, a stop-string match;
        /// `chat_stream/handle_done.rs`), and the guard's name goes only into
        /// the `--dump` body. `/v1/completions` ignores it.
        guard_stop: Option<&'static str>,
    },
    Error(String),
}
