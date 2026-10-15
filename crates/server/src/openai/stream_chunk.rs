// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: OpenAI `chat.completion.chunk` SSE payload types and their constructors.
//!
//! Owner: server (OpenAI API layer).
//! Invariants:
//! - Every constructor sets `object` to `chat.completion.chunk` and `system_fingerprint`
//!   to `fp_metrale`.
//! - Every constructor except `usage_only_chunk` builds exactly one choice, at index 0;
//!   `usage_only_chunk` builds none.

use serde::Serialize;

use super::*;

/// 2026-09-26: One `chat.completion.chunk` SSE payload.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
    pub logprobs: Option<ChoiceLogprobs>,
    /// 2026-10-14: vLLM's extra choice field beside `finish_reason`: `"repetition_detected"`
    /// on the final chunk of a repetition guard cut (`ir::FinishReason::stop_reason`);
    /// omitted from the JSON otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<&'static str>,
    /// 2026-09-26: Sampled token ids carried by this chunk; omitted from the JSON when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// 2026-09-26: Reasoning text of this chunk. There is no `reasoning` alias field, so
    /// the text appears once per delta.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tool_parser::ChunkToolCall>>,
    /// 2026-09-26: Refusal sentence, set only by `refusal_chunk`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl ChatCompletionChunk {
    /// 2026-09-26: Opening chunk of a chat stream: `role: "assistant"` and nothing else.
    pub fn role_chunk(model: &str, id: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_metrale".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".to_string()),
                    reasoning_content: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                stop_reason: None,
                token_ids: Vec::new(),
            }],
            usage: None,
        }
    }

    /// 2026-09-26: Delta chunk carrying only `reasoning_content`.
    pub fn reasoning_chunk(model: &str, id: &str, text: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_metrale".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: Some(text),
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                stop_reason: None,
                token_ids: Vec::new(),
            }],
            usage: None,
        }
    }

    /// 2026-09-26: Delta chunk carrying only `content`.
    pub fn content_chunk(model: &str, id: &str, text: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_metrale".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    content: Some(text),
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                stop_reason: None,
                token_ids: Vec::new(),
            }],
            usage: None,
        }
    }

    /// 2026-09-26: First delta of a tool call: `role: "assistant"` plus the call's id,
    /// type and name, with empty arguments. `content` is omitted, not null.
    pub fn tool_call_start_chunk(
        model: &str,
        id: &str,
        tc: &crate::tool_parser::ToolCall,
        tc_index: usize,
    ) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_metrale".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".to_string()),
                    reasoning_content: None,
                    content: None,
                    tool_calls: Some(vec![crate::tool_parser::ChunkToolCall {
                        index: tc_index,
                        id: Some(tc.id.clone()),
                        call_type: Some(tc.call_type.clone()),
                        function: crate::tool_parser::ChunkFunction {
                            name: Some(tc.function.name.clone()),
                            arguments: String::new(),
                        },
                    }]),
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                stop_reason: None,
                token_ids: Vec::new(),
            }],
            usage: None,
        }
    }

    /// 2026-09-26: Delta carrying one fragment of the arguments string of tool call
    /// `tc_index`; id, type and name are omitted.
    pub fn tool_call_args_fragment(model: &str, id: &str, tc_index: usize, fragment: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_metrale".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    content: None,
                    tool_calls: Some(vec![crate::tool_parser::ChunkToolCall {
                        index: tc_index,
                        id: None,
                        call_type: None,
                        function: crate::tool_parser::ChunkFunction {
                            name: None,
                            arguments: fragment.to_string(),
                        },
                    }]),
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                stop_reason: None,
                token_ids: Vec::new(),
            }],
            usage: None,
        }
    }

    /// 2026-09-26: Final chunk with `finish_reason`, an empty delta, and usage.
    pub fn done_chunk(model: &str, id: &str, finish_reason: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_metrale".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: Some(finish_reason.to_string()),
                logprobs: None,
                stop_reason: None,
                token_ids: Vec::new(),
            }],
            usage: Some(usage),
        }
    }

    /// 2026-09-26: Chunk with `choices: []` and usage. When `stream_options.include_usage`
    /// is set, the stream encoder sends it just before `final_chunk_no_usage`.
    pub fn usage_only_chunk(model: &str, id: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_metrale".to_string()),
            choices: Vec::new(),
            usage: Some(usage),
        }
    }

    /// 2026-09-26: Delta chunk carrying only `refusal`. The chat stream sends it at most
    /// once, just before the finish chunk(s), when `crate::refusal::detect` flags the
    /// streamed text and the response made no tool call.
    pub fn refusal_chunk(model: &str, id: &str, refusal: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_metrale".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    content: None,
                    tool_calls: None,
                    refusal: Some(refusal),
                },
                finish_reason: None,
                logprobs: None,
                stop_reason: None,
                token_ids: Vec::new(),
            }],
            usage: None,
        }
    }

    /// 2026-09-26: Final chunk with `finish_reason` and no `usage` key, for streams whose
    /// usage went out in `usage_only_chunk`.
    pub fn final_chunk_no_usage(model: &str, id: &str, finish_reason: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_metrale".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: Some(finish_reason.to_string()),
                logprobs: None,
                stop_reason: None,
                token_ids: Vec::new(),
            }],
            usage: None,
        }
    }

    /// 2026-10-14: Set the first choice's `stop_reason` (`ir::FinishReason::stop_reason`).
    pub(crate) fn with_stop_reason(mut self, stop_reason: Option<&'static str>) -> Self {
        if let Some(choice) = self.choices.first_mut() {
            choice.stop_reason = stop_reason;
        }
        self
    }

    /// 2026-09-26: Set `ids` as the first choice's `token_ids`. No-op when `ids` is empty or
    /// the chunk has no choice.
    pub(crate) fn with_token_ids(mut self, ids: Vec<u32>) -> Self {
        if !ids.is_empty()
            && let Some(choice) = self.choices.first_mut()
        {
            choice.token_ids = ids;
        }
        self
    }
}
