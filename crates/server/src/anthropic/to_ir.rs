// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Anthropic wire request → chat IR
//! (`From<MessagesRequest> for ir::ChatRequest`).
//!
//! Owner: server (Anthropic adapter).
//! Invariants:
//! - The lowered request has `n = 1`.
//! - A system message, when one is produced, is the first message.

use crate::ir;
use crate::ir::message::{ImageSource, Reasoning, ToolCall};
use crate::ir::{ContentPart, ImageData, Message, Role, ThinkingDirective};

use super::types::{
    AnthropicContent, ContentBlock, MessagesRequest, SystemContent, ToolResultContent,
};

impl From<MessagesRequest> for ir::ChatRequest {
    /// 2026-09-26: Lower the Anthropic request into the [`ir::ChatRequest`]
    /// envelope.
    ///
    /// * `system` (a string, or `text` blocks joined with `\n`, dropping
    ///   blocks whose text starts with `x-anthropic-`) → one leading System
    ///   message when non-empty.
    /// * Text blocks within one message are joined with no separator.
    /// * An assistant message → one assistant message: images then text,
    ///   `tool_use` blocks as tool calls with JSON arguments, `thinking`
    ///   blocks as reasoning joined with `\n`.
    /// * A user message → a user message (images then text) only when it has
    ///   text or images, then one Tool message per `tool_result` in block
    ///   order, with `is_error` as `Message::tool_error`.
    /// * Any role other than `assistant` is lowered as `user`.
    fn from(req: MessagesRequest) -> Self {
        // 2026-09-26: `req.messages.len()` comes from the request body, so the
        // reservation is capped. The Vec still grows to the length the request
        // needs.
        const PREALLOC_MESSAGES: usize = 4096;
        let mut messages: Vec<Message> =
            Vec::with_capacity((req.messages.len() + 1).min(PREALLOC_MESSAGES));

        if let Some(sys) = &req.system {
            let sys_text = match sys {
                SystemContent::Text(s) => s.clone(),
                SystemContent::Blocks(blocks) => blocks
                    .iter()
                    .filter(|b| {
                        b.block_type == "text"
                            && !b.text.as_deref().unwrap_or("").starts_with("x-anthropic-")
                    })
                    .filter_map(|b| b.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            if !sys_text.is_empty() {
                messages.push(Message::synthetic_system(sys_text));
            }
        }

        for m in req.messages {
            let role = match m.role.as_str() {
                "assistant" => Role::Assistant,
                _ => Role::User,
            };
            match m.content {
                AnthropicContent::Text(s) => {
                    let content = if s.is_empty() {
                        Vec::new()
                    } else {
                        vec![ContentPart::Text(s)]
                    };
                    messages.push(Message {
                        role,
                        content,
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        name: None,
                        reasoning: None,
                        tool_error: false,
                    });
                }
                AnthropicContent::Blocks(blocks) => {
                    let mut text_parts: Vec<String> = Vec::new();
                    let mut images: Vec<String> = Vec::new();
                    let mut reasoning_parts: Vec<String> = Vec::new();
                    let mut tool_calls: Vec<ToolCall> = Vec::new();
                    // 2026-09-26: (tool_use_id, text, images, is_error).
                    let mut tool_results: Vec<(String, String, Vec<String>, bool)> = Vec::new();
                    for b in blocks {
                        match b {
                            ContentBlock::Text { text } => text_parts.push(text),
                            ContentBlock::Image { source } => {
                                if let Some(uri) = source.maybe_get_image_uri() {
                                    images.push(uri);
                                }
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                tool_calls.push(ToolCall {
                                    id,
                                    name,
                                    arguments: input,
                                });
                            }
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                is_error,
                            } => {
                                let text =
                                    content.as_ref().map(|c| c.to_text()).unwrap_or_default();
                                // 2026-09-26: Images inside a tool result
                                // ride on its Tool message.
                                let tr_images: Vec<String> = match content {
                                    Some(ToolResultContent::Blocks(inner)) => inner
                                        .iter()
                                        .filter_map(|ib| match ib {
                                            ContentBlock::Image { source } => {
                                                source.maybe_get_image_uri()
                                            }
                                            _ => None,
                                        })
                                        .collect(),
                                    _ => Vec::new(),
                                };
                                tool_results.push((
                                    tool_use_id,
                                    text,
                                    tr_images,
                                    is_error.unwrap_or(false),
                                ));
                            }
                            ContentBlock::Thinking { thinking } => {
                                if let Some(t) = thinking
                                    && !t.is_empty()
                                {
                                    reasoning_parts.push(t);
                                }
                            }
                            ContentBlock::Unknown => {}
                        }
                    }
                    let text_content = text_parts.join("");
                    if role == Role::Assistant {
                        messages.push(Message {
                            role: Role::Assistant,
                            content: parts_from(&images, &text_content),
                            tool_calls,
                            tool_call_id: None,
                            name: None,
                            reasoning: if reasoning_parts.is_empty() {
                                None
                            } else {
                                Some(Reasoning {
                                    text: reasoning_parts.join("\n"),
                                })
                            },
                            tool_error: false,
                        });
                    } else {
                        if !text_content.is_empty() || !images.is_empty() {
                            messages.push(Message {
                                role: Role::User,
                                content: parts_from(&images, &text_content),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                name: None,
                                reasoning: None,
                                tool_error: false,
                            });
                        }
                        for (tool_use_id, text, tr_images, is_error) in tool_results {
                            messages.push(Message {
                                role: Role::Tool,
                                content: parts_from(&tr_images, &text),
                                tool_calls: Vec::new(),
                                tool_call_id: Some(tool_use_id),
                                name: None,
                                reasoning: None,
                                tool_error: is_error,
                            });
                        }
                    }
                }
            }
        }

        // 2026-09-26: Same order as the `thinking` object in
        // `ChatCompletionRequest::client_thinking_directive`: `disabled`, then
        // an explicit budget, then any other type with no budget (`budget:
        // None`).
        let thinking = match &req.thinking {
            None => ThinkingDirective::Unspecified,
            Some(tc) if tc.thinking_type == "disabled" => ThinkingDirective::Off,
            Some(tc) => match tc.budget_tokens {
                Some(b) => ThinkingDirective::On {
                    budget: Some(u32::try_from(b).unwrap_or(u32::MAX)),
                },
                None => ThinkingDirective::On { budget: None },
            },
        };

        ir::ChatRequest {
            model: req.model,
            messages,
            tools: req
                .tools
                .as_deref()
                .map(|ts| ts.iter().map(Into::into).collect())
                .unwrap_or_default(),
            tool_choice: req.tool_choice.as_ref().map(Into::into),
            sampling: ir::SamplingParams {
                temperature: req.temperature,
                top_k: req.top_k,
                top_p: req.top_p,
                ..Default::default()
            },
            max_tokens: req.max_tokens,
            min_tokens: 0,
            ignore_eos: false,
            stop: req.stop_sequences,
            stream: req.stream,
            n: 1,
            response_format: None,
            thinking,
            reasoning_effort: None,
            // 2026-09-26: This surface has no `chat_template_kwargs`;
            // `prepare_chat_prompt` falls back to `behavior.preserve_thinking`.
            preserve_thinking: None,
            repetition_detection: None,
            adapter: None,
            cache_salt: None,
            src_lang: None,
            tgt_lang: None,
            num_beams: None,
            length_penalty: None,
            early_stopping: None,
            logit_bias: Vec::new(),
            top_logprobs: None,
            seed: None,
            timeout_secs: None,
            return_token_ids: false,
        }
    }
}

/// 2026-09-26: Content parts: each image in order, then `text` as one part
/// when it is non-empty.
fn parts_from(images: &[String], text: &str) -> Vec<ContentPart> {
    let mut content: Vec<ContentPart> = Vec::with_capacity(images.len() + 1);
    for uri in images {
        content.push(ContentPart::Image(ImageSource {
            data: ImageData::from_uri(uri.clone()),
        }));
    }
    if !text.is_empty() {
        content.push(ContentPart::Text(text.to_string()));
    }
    content
}
