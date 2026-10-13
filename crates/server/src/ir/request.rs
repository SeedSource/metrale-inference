// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The provider-neutral chat request that `chat_completions_inner`
//! runs. The OpenAI chat, Responses and Anthropic adapters lower their wire
//! requests into it.
//!
//! The echo-only fields (`service_tier`, `metadata`, `store`,
//! `stream_options.include_usage`) are not here: they travel beside it as
//! `api::ResponseEcho`.
//!
//! Owner: server (chat IR).
//! Invariants: none beyond the types.

#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// 2026-09-26: The client's `model` field. It can also select a resident
    /// LoRA adapter (see [`Self::adapter`]).
    pub model: String,
    pub messages: Vec<super::Message>,
    /// 2026-09-26: Tool definitions; empty = none.
    pub tools: Vec<crate::tool_parser::ToolDefinition>,
    pub tool_choice: Option<crate::tool_parser::ToolChoice>,
    pub sampling: SamplingParams,
    pub max_tokens: usize,
    pub min_tokens: usize,
    /// 2026-10-12: The client asked to generate to `max_tokens` (vLLM `ignore_eos`).
    pub ignore_eos: bool,
    pub stop: Vec<String>,
    pub stream: bool,
    /// 2026-09-26: Number of choices. Only OpenAI chat can ask for more than
    /// one; the Anthropic and Responses adapters set 1.
    pub n: usize,
    /// 2026-09-26: Output shape constraint. `None` = unconstrained text; the
    /// OpenAI adapter maps `{"type":"text"}` to `None`.
    pub response_format: Option<ResponseFormat>,
    /// 2026-09-26: The client's thinking intent. The server and model
    /// defaults are applied later (`api/chat/prepare.rs`,
    /// `api/chat/thinking.rs`).
    pub thinking: ThinkingDirective,
    /// 2026-09-26: The template-facing effort tier, kept apart from the
    /// token budget. The DeepSeek-V4 encoder picks its prompt prefix from it.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// 2026-09-26: `chat_template_kwargs.preserve_thinking`: keep earlier
    /// turns' `<think>` blocks when re-rendering them. `None` = the
    /// MODEL.toml `[behavior].preserve_thinking` value; when that is unset
    /// too, the template variable is left undefined and the template's own
    /// default applies (`tokenizer/chat_render.rs`).
    pub preserve_thinking: Option<bool>,
    pub repetition_detection: Option<crate::api::inference_types::RepetitionDetectionParams>,
    /// 2026-09-26: LoRA adapter name for this request. The handler resolves
    /// it to a pool slot with `api::lora_control::resolve_request_adapter_slot`;
    /// when it is `None`, a `model` naming a resident or stageable adapter
    /// selects that adapter, and otherwise the installed active adapter runs.
    pub adapter: Option<String>,
    /// 2026-09-26: Source-language token name (e.g. `eng_Latn`), resolved to
    /// a token id by the server tokenizer; an unknown name is a 400. `None`
    /// = token id 0.
    pub src_lang: Option<String>,
    /// 2026-09-26: Target-language token name; see [`Self::src_lang`].
    pub tgt_lang: Option<String>,
    /// 2026-09-26: Beam-search width. `None` = 1. Above 1 is refused for a
    /// streaming request.
    pub num_beams: Option<u32>,
    /// 2026-09-26: Beam-search length penalty. `None` = 1.0.
    pub length_penalty: Option<f32>,
    /// 2026-09-26: Beam-search early stopping. `None` = false.
    pub early_stopping: Option<bool>,
    /// 2026-09-26: Per-token logit bias. The OpenAI adapter parses the wire's
    /// string keys and drops any key that is not a token id.
    pub logit_bias: Vec<(u32, f32)>,
    /// 2026-09-26: Logprob alternatives per token. `None` = no logprobs;
    /// `Some(0)` = the sampled token's logprob only.
    pub top_logprobs: Option<u8>,
    pub seed: Option<u64>,
    /// 2026-09-26: Request timeout in seconds; `None` = the server's
    /// `request_timeout`.
    pub timeout_secs: Option<f32>,
    /// 2026-09-26: Put the streamed tokens' ids on the stream chunks.
    pub return_token_ids: bool,
}

/// 2026-09-26: Client sampling parameters. `None` = the client did not set
/// it; `api/chat/sampling_setup.rs` then uses the server default or the
/// MODEL.toml sampling preset.
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub top_n_sigma: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum ResponseFormat {
    JsonObject,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        strict: bool,
    },
}

/// 2026-09-26: The client's thinking intent, resolved by each adapter from
/// its wire fields. The MODEL.toml `[behavior].thinking_default` is applied
/// in `api/chat/thinking.rs` only for
/// [`Unspecified`](ThinkingDirective::Unspecified).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingDirective {
    /// 2026-09-26: No client signal. The `--default-chat-template-kwargs`
    /// directive applies; when that is `Unspecified` too, the model default
    /// does.
    #[default]
    Unspecified,
    /// 2026-09-26: Thinking off, whatever the defaults say.
    Off,
    /// 2026-09-26: Thinking on. `budget: None` = the model's
    /// `max_thinking_budget`.
    On { budget: Option<u32> },
    /// 2026-09-26: Thinking on at an effort level; `api/chat/thinking.rs`
    /// turns it into a budget scaled from the model's `max_thinking_budget`.
    OnEffort(EffortLevel),
}

/// 2026-09-26: The budget-facing effort ladder. [`ReasoningEffort`] is the
/// separate template-facing vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffortLevel {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    Max,
}

impl ReasoningEffort {
    /// 2026-09-26: The string handed to chat templates (Jinja
    /// `reasoning_effort`) and to the DeepSeek-V4 encoder.
    /// - The Qwen3.8 template maps `high` to `xhigh` and then accepts only
    ///   `xhigh`, `medium` and `low`, so `Max` renders as `xhigh`: `max`
    ///   would raise.
    /// - The DeepSeek-V4 encoder accepts all four.
    /// - `jinja-templates/mistral.jinja` maps `medium` to `high`, accepts
    ///   `none` and `high`, and raises on `low` and `xhigh`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "xhigh",
        }
    }
}

/// 2026-09-26: The wire `reasoning_effort` vocabulary
/// (`none|minimal|low|medium|high|xhigh|max`). One match gives both the
/// template-facing [`ReasoningEffort`] and the budget-facing
/// [`ThinkingDirective`], so the two cannot name different tiers.
///
/// Returns `None` for an unknown spelling. `/v1/chat/completions` answers
/// that with a 400 (`validate_reasoning_effort`) and
/// `--default-chat-template-kwargs` with a startup error
/// (`main_modules/serve.rs`); the other callers treat it as absent.
///
/// `high` keeps its own budget rung (2x the model's `max_thinking_budget`,
/// against 4x for `xhigh`, unless the model caps effort budgets at that
/// value) although the Qwen3.8 template renders it as `xhigh`.
pub fn parse_wire_effort(s: &str) -> Option<(Option<ReasoningEffort>, ThinkingDirective)> {
    Some(match s {
        "none" => (None, ThinkingDirective::Off),
        "minimal" => (
            Some(ReasoningEffort::Low),
            ThinkingDirective::OnEffort(EffortLevel::Minimal),
        ),
        "low" => (
            Some(ReasoningEffort::Low),
            ThinkingDirective::OnEffort(EffortLevel::Low),
        ),
        "medium" => (
            Some(ReasoningEffort::Medium),
            ThinkingDirective::OnEffort(EffortLevel::Medium),
        ),
        "high" => (
            Some(ReasoningEffort::High),
            ThinkingDirective::OnEffort(EffortLevel::High),
        ),
        "xhigh" | "max" => (
            Some(ReasoningEffort::Max),
            ThinkingDirective::OnEffort(EffortLevel::XHigh),
        ),
        _ => return None,
    })
}

impl ThinkingDirective {
    /// 2026-09-26: True for any directive but `Unspecified`. MODEL.toml
    /// `thinking_in_tools = false` turns thinking off for a tools request
    /// only when this is false.
    pub fn is_explicit(&self) -> bool {
        !matches!(self, ThinkingDirective::Unspecified)
    }
}
