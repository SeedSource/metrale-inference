// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The provider-neutral blocking chat response. The blocking
//! path builds one per request, and the OpenAI chat, Anthropic and Responses
//! surfaces each encode it into their own wire format.
//!
//! Owner: server (chat IR).
//! Invariants: none beyond the types.

use super::message::ToolCall;

#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    /// 2026-09-26: A bare uuid; each surface adds its own prefix
    /// (`chatcmpl-`, `msg_`, `resp_`).
    pub id: String,
    /// 2026-09-26: The served model name.
    pub model: String,
    /// 2026-09-26: Unix seconds when the response was built.
    pub created: u64,
    /// 2026-09-26: One entry per requested choice. Only OpenAI chat asks for
    /// more than one; the Anthropic and Responses encoders read the first.
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub index: usize,
    /// 2026-09-26: Assistant text; `None` when the refusal check moved it
    /// into [`Self::refusal`].
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    /// 2026-09-26: The refusal sentence `refusal::detect` found at the start
    /// of the text, when no tool call fired.
    pub refusal: Option<String>,
    pub finish_reason: FinishReason,
    /// 2026-09-26: The client stop sequence that ended generation, if any.
    /// The Anthropic encoder reports it as `stop_sequence`.
    pub matched_stop: Option<String>,
    /// 2026-09-26: Per-token logprobs, when requested. Only the OpenAI chat
    /// encoder writes them.
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprob>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub top: Vec<(String, f32)>,
}

/// 2026-09-26: Token counts and server-side timings of one response.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// 2026-09-26: Prompt tokens served from the prefix cache (OpenAI
    /// `prompt_tokens_details.cached_tokens`, Anthropic
    /// `cache_read_input_tokens`).
    pub cached_prompt_tokens: usize,
    /// 2026-09-26: Completion tokens spent inside the thinking block (OpenAI
    /// `completion_tokens_details.reasoning_tokens`).
    pub reasoning_tokens: usize,
    /// 2026-09-26: Speculative draft tokens the verify step accepted for
    /// this request (OpenAI
    /// `completion_tokens_details.accepted_prediction_tokens`).
    pub accepted_prediction_tokens: usize,
    /// 2026-09-26: `request_start` → `decode_start` on the scheduler's
    /// clock. `request_start` is taken when the prefill step picks the
    /// request up, so HTTP parsing, template rendering and tokenisation
    /// come before it, and grammar compilation falls inside it.
    pub time_to_first_token_ms: f64,
    /// 2026-09-26: `decode_start` → the scheduler's finish frame, on the
    /// same clock. `decode_time_ms / (completion_tokens − 1)` is the
    /// server-side inter-token latency.
    pub decode_time_ms: f64,
    /// 2026-09-26: [`Usage::decode_rate_tok_s`] of the two fields above; the
    /// OpenAI wire key is `response_token/s`.
    pub response_tokens_per_second: f64,
}

impl Usage {
    /// 2026-09-26: `time_to_first_token_ms + decode_time_ms`: scheduler
    /// receipt → finish frame. The encoder's own serialisation comes after
    /// it.
    pub fn total_time_ms(&self) -> f64 {
        self.time_to_first_token_ms + self.decode_time_ms
    }

    /// 2026-09-26: The `response_token/s` rate, `(completion_tokens − 1) /
    /// decode seconds`: prefill produces the first token, so only the rest
    /// are decode work. The blocking and streaming chat paths and both
    /// completions paths call this.
    ///
    /// `0.0` when there is no decode window or fewer than two tokens.
    pub fn decode_rate_tok_s(completion_tokens: usize, decode_time_ms: f64) -> f64 {
        if decode_time_ms > 0.0 && completion_tokens > 0 {
            completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
        } else {
            0.0
        }
    }
}

/// 2026-09-26: Finish reason for a response cut short by the server-side
/// request deadline (`--request-timeout`, or the per-request `timeout`).
/// It is not one of the four OpenAI reasons, so a deadline cut stays
/// distinguishable from `"length"` and `"stop"`. It travels as
/// `FinishReason::Other` and `as_wire` returns it unchanged.
/// `scheduler::lifecycle::guard_stop_wire_reason` maps every other guard to
/// `"length"`.
pub const FINISH_REASON_TIMEOUT: &str = "timeout";

/// 2026-10-14: The internal reason a stream-side repetition guard cut carries
/// (`resolve_wire_finish_reason`); it becomes [`FinishReason::Repetition`], whose wire
/// reason is `"stop"`.
pub const FINISH_REASON_REPETITION: &str = "repetition";

/// 2026-09-26: Why generation stopped. `Other` keeps any other engine
/// reason string unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    /// 2026-10-14: A stream-side repetition guard (the SimHash loop detector or the line
    /// watchdog) ended the response. On the OpenAI wire it is `"stop"` with `stop_reason`
    /// `"repetition_detected"` (vLLM's stop_reason string for its repetition stop), so a
    /// guard cut stays inside the OpenAI enum and is not confused with the token budget
    /// (`"length"`), which it was reported as until this change (#82 guard fix).
    Repetition,
    Other(String),
}

impl From<&str> for FinishReason {
    fn from(s: &str) -> Self {
        match s {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            FINISH_REASON_REPETITION => FinishReason::Repetition,
            other => FinishReason::Other(other.to_string()),
        }
    }
}

impl FinishReason {
    pub fn as_wire(&self) -> &str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Repetition => "stop",
            FinishReason::Other(s) => s,
        }
    }

    /// 2026-10-14: The OpenAI choice's `stop_reason` beside `finish_reason`: set only for a
    /// repetition guard cut (`"repetition_detected"`, as vLLM sends it).
    pub fn stop_reason(&self) -> Option<&'static str> {
        match self {
            FinishReason::Repetition => Some("repetition_detected"),
            _ => None,
        }
    }
}
