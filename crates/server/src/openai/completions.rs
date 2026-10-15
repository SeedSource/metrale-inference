// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `/v1/completions` request, response and stream-chunk
//! types, and the `/tokenize` request and response.
//!
//! Owner: server (OpenAI adapter).
//! Invariants: none beyond the types.

use serde::{Deserialize, Serialize};

use super::*;
use crate::api::inference_types::RepetitionDetectionParams;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct CompletionRequest {
    pub model: String,
    /// 2026-09-26: LoRA adapter name for this request, resolved by
    /// `api::lora_control::resolve_request_adapter_slot` as on the chat path.
    #[serde(default)]
    pub adapter: Option<String>,
    /// 2026-10-14: vLLM / OpenAI `cache_salt`. Requests with different salts never
    /// share prefix-cache entries (`metrale_cache::radix_tree::salted_adapter_id`);
    /// unset or empty = no salt.
    #[serde(default)]
    pub cache_salt: Option<String>,
    /// 2026-09-26: Source-language token name, resolved to a token id by the
    /// server tokenizer. Unset = token id 0; an unknown name is a 400.
    #[serde(default)]
    pub src_lang: Option<String>,
    /// 2026-09-26: Target-language token name, resolved like `src_lang`.
    #[serde(default)]
    pub tgt_lang: Option<String>,
    /// 2026-09-26: Beam-search width; `None` = 1.
    #[serde(default)]
    pub num_beams: Option<u32>,
    /// 2026-09-26: Beam-search length penalty; `None` = 1.0.
    #[serde(default)]
    pub length_penalty: Option<f32>,
    /// 2026-09-26: Beam-search early stopping; `None` = false.
    #[serde(default)]
    pub early_stopping: Option<bool>,
    /// 2026-09-26: A string, an array of strings, an array of token ids, or
    /// an array of token-id arrays; see [`PromptInput`].
    pub prompt: PromptInput,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// 2026-09-26: Tokens to generate before EOS is allowed; 0 (the
    /// default) = no minimum.
    #[serde(default)]
    pub min_tokens: usize,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub top_n_sigma: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    /// 2026-09-26: Per-request token-loop detector parameters; `None` = the
    /// server's.
    #[serde(default)]
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// 2026-09-26: Put the prompt text before the completion text. With
    /// `logprobs`, the prompt tokens' logprobs come first too.
    #[serde(default)]
    pub echo: bool,
    /// 2026-09-26: Return each token's logprob plus this many alternatives,
    /// capped at 20. Covers the prompt tokens too when `echo` is set.
    pub logprobs: Option<u8>,
    /// 2026-09-26: Completions per prompt, 1 to 128 (default 1). Choice
    /// indices are prompt-major: `prompt_i * n + n_i`.
    #[serde(default = "default_n")]
    pub n: usize,
    /// 2026-09-26: `include_usage` sends usage in its own `choices: []`
    /// chunk before `[DONE]`.
    pub stream_options: Option<StreamOptions>,
    /// 2026-09-26: Accepted and not read, like `suffix` and `best_of`.
    #[allow(dead_code)]
    pub user: Option<String>,
    #[allow(dead_code)]
    pub suffix: Option<String>,
    #[allow(dead_code)]
    pub best_of: Option<usize>,
}

/// 2026-09-26: The completions logprobs block: four parallel arrays. With
/// `echo` they cover the prompt tokens, then the generated ones, and the
/// first prompt token's `token_logprobs` and `top_logprobs` entries are
/// `null`: nothing precedes it.
#[derive(Debug, Serialize)]
pub struct CompletionLogprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<Option<std::collections::HashMap<String, f32>>>,
    pub text_offset: Vec<usize>,
}

/// 2026-09-26: The `prompt` field:
///   - `"hello"`              → `Text`
///   - `[128000, 9906, ...]`  → `TokenIds`
///   - `[[128000], [9906]]`   → `TokenIdBatch`
///   - `["hello", "world"]`   → `TextArray`
///
/// `#[serde(untagged)]` takes the first variant that deserializes. Only the
/// empty array `[]` fits more than one; it becomes `TokenIds([])`, the first
/// array variant. A negative or out-of-`u32` number fits none, and the
/// handler answers the parse error with a 400.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    Text(String),
    /// 2026-09-26: Token ids, used as given after a vocabulary range check.
    TokenIds(Vec<u32>),
    /// 2026-09-26: One independent prompt per sub-array, each with its own
    /// choices.
    TokenIdBatch(Vec<Vec<u32>>),
    /// 2026-09-26: One independent prompt per element, each tokenized
    /// separately and with its own choices.
    TextArray(Vec<String>),
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
    pub system_fingerprint: String,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogprobs>,
}

impl CompletionResponse {
    pub fn new(model: &str, text: String, usage: Usage, finish_reason: &str) -> Self {
        Self::from_choices(
            model,
            vec![CompletionChoice {
                index: 0,
                text,
                finish_reason: finish_reason.to_string(),
                logprobs: None,
            }],
            usage,
        )
    }

    /// 2026-09-26: Multi-choice constructor. The caller sets the
    /// prompt-major indices (`prompt_i * n + n_i`).
    pub fn from_choices(model: &str, choices: Vec<CompletionChoice>, usage: Usage) -> Self {
        Self {
            id: format!("cmpl-{}", uuid_v4()),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices,
            usage,
            system_fingerprint: "fp_metrale".to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub index: usize,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// 2026-09-26: Set only by [`CompletionChunk::echo_chunk`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogprobs>,
}

impl CompletionChunk {
    pub fn text_chunk(model: &str, id: &str, text: String) -> Self {
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text,
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// 2026-09-26: The prompt text, with its logprobs when requested; sent
    /// before any generated text.
    pub fn echo_chunk(
        model: &str,
        id: &str,
        text: String,
        logprobs: Option<CompletionLogprobs>,
    ) -> Self {
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text,
                finish_reason: None,
                logprobs,
            }],
            usage: None,
        }
    }

    /// 2026-09-26: The finish chunk without usage, for
    /// `stream_options.include_usage`, where usage has its own chunk.
    pub fn finish_chunk_no_usage(model: &str, id: &str, finish_reason: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some(finish_reason.to_string()),
                logprobs: None,
            }],
            usage: None,
        }
    }

    pub fn usage_only_chunk(model: &str, id: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices: Vec::new(),
            usage: Some(usage),
        }
    }

    pub fn done_chunk(model: &str, id: &str, finish_reason: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some(finish_reason.to_string()),
                logprobs: None,
            }],
            usage: Some(usage),
        }
    }
}

/// 2026-09-26: Request body of `POST /tokenize`.
#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub prompt: Option<String>,
    pub messages: Option<Vec<IncomingMessage>>,
}

#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
    pub count: usize,
}

#[cfg(test)]
mod min_tokens_tests {
    use super::*;

    fn req_with(json: serde_json::Value) -> CompletionRequest {
        serde_json::from_value(json).expect("valid CompletionRequest JSON")
    }

    #[test]
    fn min_tokens_defaults_to_zero() {
        let req = req_with(serde_json::json!({ "model": "m", "prompt": "hi" }));
        assert_eq!(req.min_tokens, 0);
    }

    #[test]
    fn min_tokens_explicit_deserialization() {
        let req = req_with(serde_json::json!({
            "model": "m",
            "prompt": "hi",
            "min_tokens": 2048
        }));
        assert_eq!(req.min_tokens, 2048);
    }

    #[test]
    fn min_tokens_zero_is_valid_explicit_value() {
        let req = req_with(serde_json::json!({
            "model": "m",
            "prompt": "hi",
            "min_tokens": 0
        }));
        assert_eq!(req.min_tokens, 0);
    }
}
