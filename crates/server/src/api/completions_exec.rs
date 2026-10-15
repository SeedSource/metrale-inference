// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Blocking `/v1/completions`: one sequential scheduler request per
//! (prompt, `n`) choice, choice index `prompt_i * n + n_i`. Token counts are
//! summed over choices; TTFT, decode time and rate come from the last choice.
//!
//! Owner: server completions API.
//! Invariants: none beyond the types.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;

use crate::AppState;
use crate::openai::{CompletionChoice, CompletionRequest, CompletionResponse, Usage};

use super::compact::openai_error_response;
use super::completions_logprobs::build_completion_logprobs;
use super::inference_impl::strip_stop_sequences;
use super::inference_types::{InferenceRequest, RepetitionDetectionParams};
use super::strip::strip_thinking_tags;

/// 2026-09-26: Sampling/request parameters resolved once by the handler and
/// shared by every choice in the request.
pub(super) struct CompletionParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub stop_tokens: Vec<u32>,
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// 2026-09-26: `logprobs` clamped to 20. Requests generated-token
    /// logprobs, and prompt logprobs when `echo` is set.
    pub logprobs_k: Option<u8>,
    /// 2026-09-26: Resolved LoRA adapter slot; `-1` follows the active adapter.
    pub adapter_slot: i32,
    /// 2026-09-26: Source-language token id; 0 means the deployment default.
    pub src_lang_id: u32,
    /// 2026-09-26: Target-language token id; 0 means the deployment default.
    pub tgt_lang_id: u32,
    /// 2026-09-26: NLLB beam count; above 1 selects beam search.
    pub num_beams: u32,
    pub length_penalty: f32,
    pub early_stopping: bool,
}

/// 2026-09-26: Run every (prompt, `n`) choice in turn and assemble the
/// response; the first failed choice returns its error response alone.
pub(super) async fn run_blocking(
    state: Arc<AppState>,
    req: &CompletionRequest,
    prompts: Vec<Vec<u32>>,
    p: CompletionParams,
) -> Response {
    // 2026-09-26: `completions` limits `n` to 1..=128. The capacity is capped
    // at 1024 whatever the request says, and the vec grows past it if needed.
    let n = req.n.clamp(1, 128);
    let mut choices: Vec<CompletionChoice> =
        Vec::with_capacity(prompts.len().saturating_mul(n).min(1024));
    let mut sum_prompt = 0usize;
    let mut sum_completion = 0usize;
    let mut sum_cached = 0usize;
    let mut sum_reasoning = 0usize;
    let mut sum_accepted = 0usize;
    let mut last_ttft = 0.0f64;
    let mut last_decode_time_ms = 0.0f64;
    let mut last_tps = 0.0f64;

    for (prompt_i, prompt_tokens) in prompts.iter().enumerate() {
        for n_i in 0..n {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let session_hash = crate::session_manager::compute_session_hash(prompt_tokens);
            let request = InferenceRequest::Blocking {
                prompt_tokens: Arc::new(prompt_tokens.clone()),
                session_hash,
                cache_salt_hash: metrale_cache::radix_tree::cache_salt_hash(
                    req.cache_salt.as_deref(),
                ),
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
                logit_bias: p.logit_bias.clone(),
                stop_tokens: p.stop_tokens.clone(),
                enable_thinking: false,
                thinking_budget: None,
                repetition_detection: p.repetition_detection,
                require_tool_call: false,
                tools_present: false,
                suppress_tool_call: false,
                disable_mtp: false,
                grammar_spec: None,
                // 2026-09-26: Each choice gets its own seed, the request seed
                // plus the choice index, as in `chat_blocking.rs`.
                seed: req
                    .seed
                    .map(|s| s.wrapping_add((prompt_i * n + n_i) as u64)),
                top_logprobs: p.logprobs_k,
                prompt_logprobs: if req.echo { p.logprobs_k } else { None },
                echo: req.echo,
                timeout_at: state.request_deadline(None),
                response_tx: tx,
            };

            if state.request_tx.send(request).await.is_err() {
                return openai_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Scheduler queue full".to_string(),
                );
            }
            let response = match rx.await {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    return openai_error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Inference error: {e}"),
                    );
                }
                Err(_) => {
                    return openai_error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Inference cancelled".to_string(),
                    );
                }
            };

            let completion_text = match state.tokenizer.decode(&response.output_tokens) {
                Ok(t) => t,
                Err(e) => {
                    return openai_error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Decode error: {e}"),
                    );
                }
            };
            // 2026-09-26: The strips apply to the completion only; echoed
            // prompt text is returned verbatim.
            let completion_text = finish_completion_text(
                completion_text,
                &req.stop,
                state.tokenizer.uses_kimi_k3_xtml(),
            );
            let text = if req.echo {
                let prompt_text = state.tokenizer.decode(prompt_tokens).unwrap_or_default();
                format!("{prompt_text}{completion_text}")
            } else {
                completion_text
            };

            let logprobs = p.logprobs_k.map(|_| {
                let decode = |id: u32| state.tokenizer.decode(&[id]).unwrap_or_default();
                build_completion_logprobs(
                    &decode,
                    req.echo,
                    prompt_tokens,
                    &response.prompt_logprobs,
                    &response.output_tokens,
                    &response.logprobs,
                )
            });

            sum_prompt += prompt_tokens.len();
            sum_completion += response.output_tokens.len();
            sum_cached += response.cached_prompt_tokens as usize;
            sum_reasoning += response.reasoning_tokens as usize;
            sum_accepted += response.accepted_prediction_tokens;
            last_ttft = response.time_to_first_token_ms;
            last_decode_time_ms = response.decode_time_ms;
            last_tps = crate::ir::Usage::decode_rate_tok_s(
                response.output_tokens.len(),
                response.decode_time_ms,
            );

            choices.push(CompletionChoice {
                index: prompt_i * n + n_i,
                text,
                finish_reason: response.finish_reason,
                logprobs,
            });
        }
    }

    let usage = Usage {
        prompt_tokens: sum_prompt,
        completion_tokens: sum_completion,
        total_tokens: sum_prompt + sum_completion,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: sum_cached,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: sum_reasoning,
            audio_tokens: 0,
            accepted_prediction_tokens: sum_accepted,
            rejected_prediction_tokens: 0,
        }),
        time_to_first_token_ms: last_ttft,
        response_tokens_per_second: last_tps,
        decode_time_ms: last_decode_time_ms,
        total_time_ms: last_ttft + last_decode_time_ms,
    };

    Json(CompletionResponse::from_choices(
        &state.model_name,
        choices,
        usage,
    ))
    .into_response()
}

fn finish_completion_text(text: String, stops: &[String], raw_xtml: bool) -> String {
    let text = strip_stop_sequences(text, stops);
    // 2026-09-26: In Kimi K3 XTML output a `<think>` string can be literal
    // response or tool data, so `strip_thinking_tags` is skipped.
    if raw_xtml {
        text
    } else {
        strip_thinking_tags(&text)
    }
}

#[cfg(test)]
impl CompletionParams {
    pub(super) fn test() -> Self {
        Self {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            top_n_sigma: 0.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: Vec::new(),
            stop_tokens: Vec::new(),
            repetition_detection: None,
            logprobs_k: None,
            adapter_slot: -1,
            src_lang_id: 0,
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
        }
    }
}

#[cfg(test)]
#[path = "completions_exec_tests.rs"]
mod tests;
