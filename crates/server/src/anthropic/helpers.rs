// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Anthropic error body, and the tool, tool-choice and
//! stop-reason conversions between the Anthropic and OpenAI vocabularies.
//!
//! Owner: server (Anthropic adapter).
//! Invariants: none beyond the types.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::tool_parser;

use super::types::*;

pub(super) fn anthropic_error(status: StatusCode, error_type: &str, message: String) -> Response {
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        }
    });
    (status, Json(body)).into_response()
}

impl From<&AnthropicTool> for tool_parser::ToolDefinition {
    /// 2026-09-26: An Anthropic tool as an OpenAI `function` tool, with
    /// `input_schema` passed through unchanged as `parameters`.
    fn from(t: &AnthropicTool) -> Self {
        tool_parser::ToolDefinition {
            tool_type: "function".to_string(),
            function: tool_parser::FunctionDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: Some(t.input_schema.clone()),
            },
        }
    }
}

impl From<&AnthropicToolChoice> for tool_parser::ToolChoice {
    /// 2026-09-26: `any` becomes `required`; `auto` and `none` keep their
    /// names; `tool` with a name selects that function. `tool` without a name,
    /// and any other type, becomes `auto`.
    fn from(tc: &AnthropicToolChoice) -> Self {
        match tc.choice_type.as_str() {
            "any" => tool_parser::ToolChoice::Mode("required".to_string()),
            "auto" => tool_parser::ToolChoice::Mode("auto".to_string()),
            "none" => tool_parser::ToolChoice::Mode("none".to_string()),
            "tool" => {
                if let Some(ref name) = tc.name {
                    tool_parser::ToolChoice::Specific {
                        function: tool_parser::ToolChoiceFunction { name: name.clone() },
                    }
                } else {
                    tool_parser::ToolChoice::Mode("auto".to_string())
                }
            }
            _ => tool_parser::ToolChoice::Mode("auto".to_string()),
        }
    }
}

/// 2026-09-26: An OpenAI finish reason as Anthropic's `stop_reason`. Unknown
/// reasons become `end_turn`.
pub(super) fn convert_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        "content_filter" => "refusal",
        // 2026-09-26: Anthropic has no deadline reason. `max_tokens` reports
        // the turn as cut short; `end_turn` would report it complete.
        crate::ir::FINISH_REASON_TIMEOUT => "max_tokens",
        _ => "end_turn",
    }
}

/// 2026-10-14: [`convert_stop_reason`] for an IR finish reason. A repetition guard cut is
/// `"stop"` on the OpenAI wire (with its own `stop_reason`), but Anthropic has no such
/// field, and `end_turn` would report the cut turn complete: it stays `max_tokens`, as the
/// guard cut was reported before (#82 guard fix).
pub(super) fn anthropic_stop_reason(reason: &crate::ir::FinishReason) -> &'static str {
    match reason {
        crate::ir::FinishReason::Repetition => "max_tokens",
        other => convert_stop_reason(other.as_wire()),
    }
}
