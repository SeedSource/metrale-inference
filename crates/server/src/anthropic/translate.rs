// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The non-streaming response direction: chat IR response →
//! Anthropic `MessagesResponse`.
//!
//! Owner: server (Anthropic adapter).
//! Invariants: none beyond the types.

use super::types::*;

impl From<crate::ir::ChatResponse> for MessagesResponse {
    /// 2026-09-26: Encode a [`crate::ir::ChatResponse`] as Anthropic's
    /// `MessagesResponse`. Only the first choice is read: the Anthropic
    /// request adapter always sets `n = 1`.
    fn from(ir: crate::ir::ChatResponse) -> Self {
        let id = format!("msg_{}", ir.id);
        let choice = ir.choices.into_iter().next();

        let mut content: Vec<ResponseBlock> = Vec::new();
        let mut stop_reason = "end_turn";
        let mut stop_sequence: Option<String> = None;
        if let Some(c) = choice {
            if let Some(r) = c.reasoning {
                content.push(ResponseBlock::Thinking { thinking: r });
            }
            if let Some(text) = c.content
                && !text.is_empty()
            {
                content.push(ResponseBlock::Text { text });
            }
            for tc in c.tool_calls {
                content.push(ResponseBlock::ToolUse {
                    id: tc.id,
                    name: tc.name,
                    input: tc.arguments,
                });
            }
            stop_reason = match c.finish_reason {
                crate::ir::FinishReason::Stop if c.matched_stop.is_some() => "stop_sequence",
                crate::ir::FinishReason::Stop => "end_turn",
                crate::ir::FinishReason::ToolCalls => "tool_use",
                crate::ir::FinishReason::Length => "max_tokens",
                crate::ir::FinishReason::ContentFilter => "refusal",
                crate::ir::FinishReason::Repetition => "max_tokens",
                crate::ir::FinishReason::Other(_) => "end_turn",
            };
            stop_sequence = c.matched_stop;
        }

        MessagesResponse {
            id,
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content,
            model: ir.model,
            stop_reason: Some(stop_reason.to_string()),
            stop_sequence,
            usage: AnthropicUsage {
                input_tokens: ir.usage.prompt_tokens,
                output_tokens: ir.usage.completion_tokens,
                cache_read_input_tokens: ir.usage.cached_prompt_tokens,
            },
        }
    }
}
