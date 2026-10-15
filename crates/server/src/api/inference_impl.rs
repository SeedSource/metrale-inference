// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Accessors on `InferenceRequest` read by the scheduler, and the
//! stop-sequence and reasoning-split helpers shared by the API handlers.
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

use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};

use super::chat::chat_completions_inner;
use super::strip::strip_thinking_tags;

use super::inference_types::*;
use super::sanitizer::*;

impl InferenceRequest {
    pub fn prompt_len(&self) -> usize {
        match self {
            InferenceRequest::Blocking { prompt_tokens, .. } => prompt_tokens.len(),
            InferenceRequest::Streaming { prompt_tokens, .. } => prompt_tokens.len(),
        }
    }

    pub fn has_image_pixels(&self) -> bool {
        match self {
            InferenceRequest::Blocking { image_pixels, .. } => !image_pixels.is_empty(),
            InferenceRequest::Streaming { image_pixels, .. } => !image_pixels.is_empty(),
        }
    }

    pub fn take_image_pixels(&mut self) -> Vec<metrale_model_layers::VisionItem> {
        match self {
            InferenceRequest::Blocking { image_pixels, .. } => std::mem::take(image_pixels),
            InferenceRequest::Streaming { image_pixels, .. } => std::mem::take(image_pixels),
        }
    }

    /// 2026-09-26: Borrow the preprocessed images, read by the vision batch
    /// pre-pass in `scheduler/phase_start_prefills.rs`.
    pub fn image_pixels_ref(&self) -> &[metrale_model_layers::VisionItem] {
        match self {
            InferenceRequest::Blocking { image_pixels, .. } => image_pixels.as_slice(),
            InferenceRequest::Streaming { image_pixels, .. } => image_pixels.as_slice(),
        }
    }

    pub fn take_stop_tokens(&mut self) -> Vec<u32> {
        match self {
            InferenceRequest::Blocking { stop_tokens, .. } => std::mem::take(stop_tokens),
            InferenceRequest::Streaming { stop_tokens, .. } => std::mem::take(stop_tokens),
        }
    }

    pub fn top_k(&self) -> u32 {
        match self {
            InferenceRequest::Blocking { top_k, .. } => *top_k,
            InferenceRequest::Streaming { top_k, .. } => *top_k,
        }
    }

    pub fn top_p(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { top_p, .. } => *top_p,
            InferenceRequest::Streaming { top_p, .. } => *top_p,
        }
    }

    pub fn top_n_sigma(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { top_n_sigma, .. } => *top_n_sigma,
            InferenceRequest::Streaming { top_n_sigma, .. } => *top_n_sigma,
        }
    }

    pub fn min_p(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { min_p, .. } => *min_p,
            InferenceRequest::Streaming { min_p, .. } => *min_p,
        }
    }

    pub fn repetition_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking {
                repetition_penalty, ..
            } => *repetition_penalty,
            InferenceRequest::Streaming {
                repetition_penalty, ..
            } => *repetition_penalty,
        }
    }

    pub fn presence_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking {
                presence_penalty, ..
            } => *presence_penalty,
            InferenceRequest::Streaming {
                presence_penalty, ..
            } => *presence_penalty,
        }
    }

    pub fn frequency_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking {
                frequency_penalty, ..
            } => *frequency_penalty,
            InferenceRequest::Streaming {
                frequency_penalty, ..
            } => *frequency_penalty,
        }
    }

    /// 2026-09-26: DRY penalty multiplier; 0.0 turns DRY off
    /// (`metrale_sampling::apply_dry_penalty`).
    pub fn dry_multiplier(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { dry_multiplier, .. } => *dry_multiplier,
            InferenceRequest::Streaming { dry_multiplier, .. } => *dry_multiplier,
        }
    }

    /// 2026-09-26: LZ penalty strength for `metrale_sampling::apply_lz_penalty`;
    /// 0.0 subtracts nothing.
    pub fn lz_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { lz_penalty, .. } => *lz_penalty,
            InferenceRequest::Streaming { lz_penalty, .. } => *lz_penalty,
        }
    }

    pub fn dry_base(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { dry_base, .. } => *dry_base,
            InferenceRequest::Streaming { dry_base, .. } => *dry_base,
        }
    }

    /// 2026-09-26: DRY allowed length: only a match longer than this is
    /// penalised (`metrale_sampling::apply_dry_penalty`).
    pub fn dry_allowed_length(&self) -> u32 {
        match self {
            InferenceRequest::Blocking {
                dry_allowed_length, ..
            } => *dry_allowed_length,
            InferenceRequest::Streaming {
                dry_allowed_length, ..
            } => *dry_allowed_length,
        }
    }

    pub fn logit_bias(&self) -> &[(u32, f32)] {
        match self {
            InferenceRequest::Blocking { logit_bias, .. } => logit_bias,
            InferenceRequest::Streaming { logit_bias, .. } => logit_bias,
        }
    }

    /// 2026-10-14: Hash of the request's `cache_salt` (0 = none), folded into
    /// `SequenceState.adapter_id` at prefill.
    pub fn cache_salt_hash(&self) -> u64 {
        match self {
            InferenceRequest::Blocking {
                cache_salt_hash, ..
            } => *cache_salt_hash,
            InferenceRequest::Streaming {
                cache_salt_hash, ..
            } => *cache_salt_hash,
        }
    }

    /// 2026-09-26: Session hash, copied onto `SequenceState.session_hash` at
    /// prefill.
    pub fn session_hash(&self) -> u64 {
        match self {
            InferenceRequest::Blocking { session_hash, .. } => *session_hash,
            InferenceRequest::Streaming { session_hash, .. } => *session_hash,
        }
    }

    /// 2026-09-26: LoRA adapter slot; `-1` follows the active adapter. Copied
    /// onto `SequenceState.adapter_slot` at prefill.
    pub fn adapter_slot(&self) -> i32 {
        match self {
            InferenceRequest::Blocking { adapter_slot, .. } => *adapter_slot,
            InferenceRequest::Streaming { adapter_slot, .. } => *adapter_slot,
        }
    }

    /// 2026-09-26: Source-language token id (0: deployment default), copied
    /// onto `SequenceState.src_lang_id` at prefill.
    pub fn src_lang_id(&self) -> u32 {
        match self {
            InferenceRequest::Blocking { src_lang_id, .. } => *src_lang_id,
            InferenceRequest::Streaming { src_lang_id, .. } => *src_lang_id,
        }
    }

    /// 2026-09-26: Target-language token id (0: deployment default), copied
    /// onto `SequenceState.tgt_lang_id` at prefill.
    pub fn tgt_lang_id(&self) -> u32 {
        match self {
            InferenceRequest::Blocking { tgt_lang_id, .. } => *tgt_lang_id,
            InferenceRequest::Streaming { tgt_lang_id, .. } => *tgt_lang_id,
        }
    }

    /// 2026-09-26: NLLB beam count (above 1 selects beam search), copied onto
    /// `SequenceState.num_beams` at prefill.
    pub fn num_beams(&self) -> u32 {
        match self {
            InferenceRequest::Blocking { num_beams, .. } => *num_beams,
            InferenceRequest::Streaming { num_beams, .. } => *num_beams,
        }
    }

    /// 2026-09-26: NLLB beam-search length penalty, copied onto
    /// `SequenceState.length_penalty` at prefill.
    pub fn length_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { length_penalty, .. } => *length_penalty,
            InferenceRequest::Streaming { length_penalty, .. } => *length_penalty,
        }
    }

    /// 2026-09-26: NLLB beam-search early stopping, copied onto
    /// `SequenceState.early_stopping` at prefill.
    pub fn early_stopping(&self) -> bool {
        match self {
            InferenceRequest::Blocking { early_stopping, .. } => *early_stopping,
            InferenceRequest::Streaming { early_stopping, .. } => *early_stopping,
        }
    }

    pub fn max_tokens(&self) -> usize {
        match self {
            InferenceRequest::Blocking { max_tokens, .. } => *max_tokens,
            InferenceRequest::Streaming { max_tokens, .. } => *max_tokens,
        }
    }

    /// 2026-09-26: Clone of the prompt-token `Arc`, read by the beam
    /// co-dispatch pre-pass in `scheduler/phase_start_prefills.rs`.
    pub fn prompt_tokens_arc(&self) -> std::sync::Arc<Vec<u32>> {
        match self {
            InferenceRequest::Blocking { prompt_tokens, .. } => prompt_tokens.clone(),
            InferenceRequest::Streaming { prompt_tokens, .. } => prompt_tokens.clone(),
        }
    }

    pub fn enable_thinking(&self) -> bool {
        match self {
            InferenceRequest::Blocking {
                enable_thinking, ..
            } => *enable_thinking,
            InferenceRequest::Streaming {
                enable_thinking, ..
            } => *enable_thinking,
        }
    }

    pub fn thinking_budget(&self) -> Option<u32> {
        match self {
            InferenceRequest::Blocking {
                thinking_budget, ..
            } => *thinking_budget,
            InferenceRequest::Streaming {
                thinking_budget, ..
            } => *thinking_budget,
        }
    }

    /// 2026-09-26: Per-request loop-detector parameters; `None` falls back to
    /// the server's (`WatchdogParams::content_loop_params`).
    pub fn repetition_detection(
        &self,
    ) -> Option<crate::api::inference_types::RepetitionDetectionParams> {
        match self {
            InferenceRequest::Blocking {
                repetition_detection,
                ..
            } => *repetition_detection,
            InferenceRequest::Streaming {
                repetition_detection,
                ..
            } => *repetition_detection,
        }
    }

    pub fn require_tool_call(&self) -> bool {
        match self {
            InferenceRequest::Blocking {
                require_tool_call, ..
            } => *require_tool_call,
            InferenceRequest::Streaming {
                require_tool_call, ..
            } => *require_tool_call,
        }
    }

    /// 2026-09-26: The request's `tools_present`: the chat handler's
    /// `tools_active`, false for `/v1/completions`. When true, a
    /// `</tool_call>` outside thinking does not end the turn
    /// (`scheduler/decode_logits_step/per_token.rs`).
    pub fn tools_present(&self) -> bool {
        match self {
            InferenceRequest::Blocking { tools_present, .. } => *tools_present,
            InferenceRequest::Streaming { tools_present, .. } => *tools_present,
        }
    }

    /// 2026-09-26: Set by the chat tool-loop detector (`chat/loop_detect.rs`):
    /// lowers the `<tool_call>` logit outside thinking
    /// (`logit_processors/tool_during_think.rs`) and blocks speculative decode.
    pub fn suppress_tool_call(&self) -> bool {
        match self {
            InferenceRequest::Blocking {
                suppress_tool_call, ..
            } => *suppress_tool_call,
            InferenceRequest::Streaming {
                suppress_tool_call, ..
            } => *suppress_tool_call,
        }
    }

    /// 2026-09-26: When true, the sequence is not eligible for speculative
    /// decode (`metrale_speculative::mtp_gate::spec_dispatch_eligible`). Every
    /// API handler sets it to `false`.
    pub fn disable_mtp(&self) -> bool {
        match self {
            InferenceRequest::Blocking { disable_mtp, .. } => *disable_mtp,
            InferenceRequest::Streaming { disable_mtp, .. } => *disable_mtp,
        }
    }

    pub fn seed(&self) -> Option<u64> {
        match self {
            InferenceRequest::Blocking { seed, .. } => *seed,
            InferenceRequest::Streaming { seed, .. } => *seed,
        }
    }

    pub fn take_grammar_spec(&mut self) -> Option<GrammarSpec> {
        match self {
            InferenceRequest::Blocking { grammar_spec, .. } => grammar_spec.take(),
            InferenceRequest::Streaming { grammar_spec, .. } => grammar_spec.take(),
        }
    }

    /// 2026-09-26: EOS ids are masked while the output is shorter than this
    /// (`logit_processors/min_tokens_eos.rs`).
    pub fn min_tokens(&self) -> usize {
        match self {
            InferenceRequest::Blocking { min_tokens, .. } => *min_tokens,
            InferenceRequest::Streaming { min_tokens, .. } => *min_tokens,
        }
    }

    pub fn top_logprobs(&self) -> Option<u8> {
        match self {
            InferenceRequest::Blocking { top_logprobs, .. } => *top_logprobs,
            InferenceRequest::Streaming { top_logprobs, .. } => *top_logprobs,
        }
    }

    /// 2026-09-26: Prompt-token logprobs, set by `/v1/completions` with `echo`.
    pub fn prompt_logprobs(&self) -> Option<u8> {
        match self {
            InferenceRequest::Blocking {
                prompt_logprobs, ..
            } => *prompt_logprobs,
            InferenceRequest::Streaming {
                prompt_logprobs, ..
            } => *prompt_logprobs,
        }
    }

    pub fn timeout_at(&self) -> Option<std::time::Instant> {
        match self {
            InferenceRequest::Blocking { timeout_at, .. } => *timeout_at,
            InferenceRequest::Streaming { timeout_at, .. } => *timeout_at,
        }
    }
}

/// 2026-09-26: Stop strings that encode to exactly one token, as sorted,
/// deduplicated token IDs. Longer stop strings are logged and left out; they
/// are matched as text.
pub(crate) fn tokenize_stop_sequences(
    tokenizer: &crate::tokenizer::ChatTokenizer,
    stops: &[String],
) -> Vec<u32> {
    let mut tokens = Vec::new();
    for s in stops {
        match tokenizer.encode(s) {
            Ok(ids) if ids.len() == 1 => tokens.push(ids[0]),
            Ok(ids) if ids.len() > 1 => {
                tracing::info!(
                    "Multi-token stop '{}' ({} tokens) — use string matching",
                    s,
                    ids.len()
                );
            }
            Ok(_) => {}
            Err(e) => tracing::warn!("Failed to tokenize stop '{}': {e}", s),
        }
    }
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// 2026-09-26: Strip one stop string from the end of `text`, if it ends with one.
pub(crate) fn strip_stop_sequences(text: String, stops: &[String]) -> String {
    strip_stop_sequences_matched(text, stops).0
}

/// 2026-09-26: [`strip_stop_sequences`], also returning the stop string that
/// matched; `chat_blocking.rs` stores it as `matched_stop`, which the Anthropic
/// translator reports as `stop_sequence`.
pub(crate) fn strip_stop_sequences_matched(
    mut text: String,
    stops: &[String],
) -> (String, Option<String>) {
    // 2026-09-26: Longest first, so when one stop string ends another
    // (`"b>"` and `"</b>"`), the longer one is stripped.
    let mut sorted: Vec<&String> = stops.iter().collect();
    sorted.sort_by_key(|s| std::cmp::Reverse(s.len()));
    for s in sorted {
        if let Some(stripped) = text.strip_suffix(s.as_str()) {
            text.truncate(stripped.len());
            return (text, Some(s.clone()));
        }
    }
    (text, None)
}

/// 2026-09-26: Split model output into `(reasoning, response)` with the model's
/// reasoning parser (`ReasoningParser::extract_thinking`, which drops the
/// reasoning when `enable_thinking` is false). With no parser: `(None, text)`.
pub(crate) fn extract_thinking(
    text: &str,
    enable_thinking: bool,
    parser: Option<&dyn crate::reasoning_parser::ReasoningParser>,
) -> (Option<String>, String) {
    if let Some(p) = parser {
        p.extract_thinking(text, enable_thinking)
    } else {
        (None, text.to_string())
    }
}
