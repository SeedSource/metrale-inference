// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The OpenAI chat completions request body, and the resolution
//! of its thinking and reasoning-effort fields into IR values.
//!
//! Owner: server (OpenAI adapter).
//! Invariants: none beyond the types.

use serde::Deserialize;

use super::*;
use crate::api::inference_types::RepetitionDetectionParams;
use crate::ir::ThinkingDirective;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: String,
    /// 2026-09-26: LoRA adapter name for this request; see
    /// `ir::ChatRequest::adapter`.
    #[serde(default)]
    pub adapter: Option<String>,
    /// 2026-09-26: Source-language token name; see
    /// `ir::ChatRequest::src_lang`.
    #[serde(default)]
    pub src_lang: Option<String>,
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
    pub messages: Vec<IncomingMessage>,
    #[serde(default = "default_max_tokens", alias = "max_completion_tokens")]
    pub max_tokens: usize,
    /// 2026-09-26: `None` = server default (`api/chat/sampling_setup.rs`),
    /// here and for `top_k` through `frequency_penalty`.
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
    /// 2026-09-26: `{"token_id": bias}`; a key that is not a token id is
    /// dropped when the request is lowered to the IR.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(default)]
    pub stream: bool,
    /// 2026-09-26: Put the streamed tokens' ids in each chunk's
    /// `choices[0].token_ids`. Off by default.
    #[serde(default)]
    pub return_token_ids: bool,
    /// 2026-09-26: Top-level thinking switch, the lowest-priority channel of
    /// [`Self::client_thinking_directive`]. `None` leaves the decision to
    /// the other channels and then the defaults.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// 2026-09-26: Anthropic-style `{"thinking": {"budget_tokens": N}}`, the
    /// highest-priority thinking channel.
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    /// 2026-09-26: Top-level thinking budget, also accepted as
    /// `max_thinking_tokens` or `thinking_budget`. 0 turns thinking off.
    #[serde(default, alias = "max_thinking_tokens", alias = "thinking_budget")]
    pub thinking_token_budget: Option<u32>,
    /// 2026-09-26: Per-request token-loop detector parameters
    /// (`{min_pattern_size, max_pattern_size, min_count}`); `None` = the
    /// server's.
    #[serde(default)]
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// 2026-09-26: `{"reasoning": {"effort": "low"}}`.
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    #[serde(default)]
    pub tools: Option<Vec<crate::tool_parser::ToolDefinition>>,
    /// 2026-09-26: `"auto"`, `"none"`, `"required"`, or a function object;
    /// any other string is a 400.
    #[serde(default)]
    pub tool_choice: Option<crate::tool_parser::ToolChoice>,
    /// 2026-09-26: Stop sequences: null, one string, or an array.
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Vec<String>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// 2026-09-26: Tokens to generate before EOS is allowed; 0 (the
    /// default) = no minimum.
    #[serde(default)]
    pub min_tokens: usize,
    pub seed: Option<u64>,
    /// 2026-09-26: `true` alone returns the sampled token's logprob with no
    /// alternatives; `top_logprobs` sets the count when present.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// 2026-09-26: Alternatives per token, capped at 20.
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    /// 2026-09-26: Request timeout in seconds; `None` = the server's.
    #[serde(default)]
    pub timeout: Option<f32>,
    /// 2026-09-26: Number of choices, 1 to 128 (default 1). A streaming
    /// request with more than 1 is a 400.
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// 2026-09-26: Accepted and not read, like `verbosity`,
    /// `safety_identifier`, `prompt_cache_key`, `user`, `modalities`,
    /// `audio`, `prediction` and `web_search_options` below.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub verbosity: Option<String>,
    /// 2026-09-26: Echoed back in the blocking response; see
    /// `api::ResponseEcho`.
    #[serde(default)]
    pub service_tier: Option<String>,
    /// 2026-09-26: `true` stores the blocking response for
    /// `GET /v1/chat/completions/{id}` (`openai::encode_chat_response`).
    #[serde(default)]
    pub store: Option<bool>,
    /// 2026-09-26: Echoed back in the blocking response.
    #[serde(default)]
    pub metadata: Option<std::collections::HashMap<String, String>>,
    #[serde(default)]
    pub safety_identifier: Option<String>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub modalities: Option<Vec<String>>,
    #[serde(default)]
    pub audio: Option<serde_json::Value>,
    #[serde(default)]
    pub prediction: Option<serde_json::Value>,
    #[serde(default)]
    pub web_search_options: Option<serde_json::Value>,
    /// 2026-09-26: Top-level effort string. `reasoning.effort` wins when both
    /// are set.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(default)]
pub struct StreamOptions {
    /// 2026-09-26: Send usage in its own `choices: []` chunk: before the finish
    /// chunk on `/v1/chat/completions` (`openai/encode_stream.rs`), after it
    /// on `/v1/completions` (`api/completions.rs`).
    pub include_usage: bool,
    /// 2026-09-26: Accepted and not read; no padding is sent.
    pub include_obfuscation: bool,
}

/// 2026-09-26: `response_format`, tagged by `"type"`. `text` means no
/// constraint and lowers to `None` in the IR.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaSpec },
}

#[derive(Debug, Deserialize)]
pub struct JsonSchemaSpec {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub schema: serde_json::Value,
    #[serde(default = "default_true")]
    pub strict: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct ThinkingConfig {
    /// 2026-09-26: Thinking budget. Any value, 0 included, turns thinking
    /// on unless `type` is `"disabled"`.
    pub budget_tokens: Option<u32>,
    /// 2026-09-26: Only `"disabled"` is read; it turns thinking off.
    #[serde(rename = "type")]
    pub thinking_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ReasoningConfig {
    pub effort: Option<String>,
}

/// 2026-09-26: The request's `chat_template_kwargs`. The
/// `--default-chat-template-kwargs` flag is parsed separately, in
/// `main_modules/serve.rs`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatTemplateKwargs {
    pub enable_thinking: Option<bool>,
    pub thinking_budget: Option<u32>,
    /// 2026-09-26: Keep earlier turns' `<think>` blocks when re-rendering
    /// them; see `ir::ChatRequest::preserve_thinking`.
    pub preserve_thinking: Option<bool>,
    /// 2026-09-26: The lowest-priority effort channel, after
    /// `reasoning.effort` and the top-level `reasoning_effort`. Within this
    /// object, `thinking_budget` and `enable_thinking: false` win over it.
    pub reasoning_effort: Option<String>,
    /// 2026-10-12: Alternate spellings of `enable_thinking` sent by other
    /// engines' clients (SGLang-style `thinking`, MiniMax-style
    /// `thinking_mode: "enabled" | "disabled"`); read only when
    /// `enable_thinking` is absent. See [`Self::thinking_switch`].
    pub thinking: Option<bool>,
    pub thinking_mode: Option<String>,
}

impl ChatTemplateKwargs {
    /// 2026-10-12: The on/off switch: `enable_thinking`, then `thinking`,
    /// then `thinking_mode` (`"enabled"` / `"disabled"`; any other string is
    /// ignored).
    fn thinking_switch(&self) -> Option<bool> {
        let mode = match self.thinking_mode.as_deref() {
            Some("enabled") => Some(true),
            Some("disabled") => Some(false),
            _ => None,
        };
        self.enable_thinking.or(self.thinking).or(mode)
    }
}

impl ChatCompletionRequest {
    /// 2026-09-26: The dedicated effort channels: `reasoning.effort`, then
    /// the top-level `reasoning_effort`.
    fn body_reasoning_effort(&self) -> Option<&str> {
        self.reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref())
            .or(self.reasoning_effort.as_deref())
    }

    /// 2026-09-26: The template-facing effort string: the dedicated
    /// channels, then `chat_template_kwargs.reasoning_effort`.
    fn requested_reasoning_effort(&self) -> Option<&str> {
        self.body_reasoning_effort().or_else(|| {
            self.chat_template_kwargs
                .as_ref()
                .and_then(|kw| kw.reasoning_effort.as_deref())
        })
    }

    /// 2026-09-26: Reject an effort string outside the
    /// [`crate::ir::parse_wire_effort`] vocabulary on any of the three
    /// channels, including one shadowed by a higher-priority valid value.
    /// The raw string does not survive lowering to the IR, so the
    /// `/v1/chat/completions` handler calls this first and answers `Err`
    /// with a 400.
    pub fn validate_reasoning_effort(&self) -> Result<(), String> {
        let channels = [
            self.reasoning.as_ref().and_then(|r| r.effort.as_deref()),
            self.reasoning_effort.as_deref(),
            self.chat_template_kwargs
                .as_ref()
                .and_then(|kw| kw.reasoning_effort.as_deref()),
        ];
        match channels
            .into_iter()
            .flatten()
            .find(|s| crate::ir::parse_wire_effort(s).is_none())
        {
            None => Ok(()),
            Some(bad) => Err(format!(
                "invalid reasoning_effort {bad:?}: expected one of \
                 none, minimal, low, medium, high, xhigh, max"
            )),
        }
    }

    /// 2026-09-26: The template-facing effort. An unknown spelling gives
    /// `None`, the unset default.
    pub fn client_reasoning_effort(&self) -> Option<crate::ir::ReasoningEffort> {
        crate::ir::parse_wire_effort(self.requested_reasoning_effort()?)
            .and_then(|(template_effort, _)| template_effort)
    }

    /// 2026-09-26: Resolve the client's thinking intent from the request
    /// body into a [`ThinkingDirective`]. The server default directive is
    /// applied in `api/chat/prepare.rs`; the model default and
    /// `--disable-thinking` in `api/chat/thinking.rs`.
    ///
    /// Priority, highest first:
    /// 1. the `thinking` object;
    /// 2. `thinking_token_budget` (and its aliases);
    /// 3. `reasoning.effort`, then the top-level `reasoning_effort`;
    /// 4. `chat_template_kwargs`;
    /// 5. the top-level `enable_thinking`.
    ///
    /// No channel set gives [`ThinkingDirective::Unspecified`].
    pub fn client_thinking_directive(&self) -> ThinkingDirective {
        if let Some(ref tc) = self.thinking {
            if let Some(ref t) = tc.thinking_type
                && t == "disabled"
            {
                return ThinkingDirective::Off;
            }
            if let Some(budget) = tc.budget_tokens {
                return ThinkingDirective::On {
                    budget: Some(budget),
                };
            }
            // 2026-09-26: No budget: the model's `max_thinking_budget`.
            return ThinkingDirective::On { budget: None };
        }

        if let Some(budget) = self.thinking_token_budget {
            return if budget > 0 {
                ThinkingDirective::On {
                    budget: Some(budget),
                }
            } else {
                ThinkingDirective::Off
            };
        }

        if let Some(effort) = self.body_reasoning_effort() {
            // 2026-09-26: The kwargs effort string waits for step 4, after
            // `kwargs.enable_thinking`, so `enable_thinking: false` beats an
            // effort string in the same object. An unknown spelling falls
            // through as if absent.
            if let Some((_, directive)) = crate::ir::parse_wire_effort(effort) {
                return directive;
            }
        }

        // 2026-09-26: Inside `chat_template_kwargs`, `thinking_budget` wins,
        // then `enable_thinking: false`, then a valid `reasoning_effort`.
        // `enable_thinking: true` counts only when `reasoning_effort` is
        // absent.
        if let Some(ref kwargs) = self.chat_template_kwargs {
            if let Some(budget) = kwargs.thinking_budget {
                return if budget > 0 {
                    ThinkingDirective::On {
                        budget: Some(budget),
                    }
                } else {
                    ThinkingDirective::Off
                };
            }
            if let Some(enabled) = kwargs.thinking_switch() {
                if !enabled {
                    return ThinkingDirective::Off;
                }
                // 2026-09-26: With an effort string beside it, `true` defers
                // to the effort rung, whose budget matches the tier the
                // template renders.
                if kwargs.reasoning_effort.is_none() {
                    return ThinkingDirective::On { budget: None };
                }
            }
            if let Some(effort) = kwargs.reasoning_effort.as_deref()
                && let Some((_, directive)) = crate::ir::parse_wire_effort(effort)
            {
                return directive;
            }
            if let Some(effort) = kwargs.reasoning_effort.as_deref()
                && let Some((_, directive)) = crate::ir::parse_wire_effort(effort)
            {
                return directive;
            }
        }

        if let Some(enabled) = self.enable_thinking {
            return if enabled {
                ThinkingDirective::On { budget: None }
            } else {
                ThinkingDirective::Off
            };
        }

        ThinkingDirective::Unspecified
    }
}

pub(super) fn default_max_tokens() -> usize {
    4096
}
pub(super) fn default_n() -> usize {
    1
}

/// 2026-09-26: Deserialize `stop` from null, one string, or an array of
/// strings.
pub(super) fn deserialize_stop<'de, D>(d: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RawStop {
        Str(String),
        Arr(Vec<String>),
        Null(()),
    }
    match RawStop::deserialize(d)? {
        RawStop::Str(s) => Ok(vec![s]),
        RawStop::Arr(v) => Ok(v),
        RawStop::Null(()) => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod alias_tests {
    use super::ChatCompletionRequest;

    fn base(extra: &str) -> String {
        format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"max_tokens":16{extra}}}"#
        )
    }

    #[test]
    fn max_thinking_tokens_aliases_thinking_token_budget() {
        let req: ChatCompletionRequest =
            serde_json::from_str(&base(r#","max_thinking_tokens":128"#)).unwrap();
        assert_eq!(req.thinking_token_budget, Some(128));
    }

    #[test]
    fn canonical_thinking_token_budget_still_works() {
        let req: ChatCompletionRequest =
            serde_json::from_str(&base(r#","thinking_token_budget":256"#)).unwrap();
        assert_eq!(req.thinking_token_budget, Some(256));
    }
}
