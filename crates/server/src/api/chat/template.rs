// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Chat-template rendering: build the JSON message array, run the optional
//! auto-compact, apply the Jinja template, expand vision pad tokens, and detect
//! template-forced thinking.
//!
//! Owner: server chat API.
//! Invariants: none beyond the types.

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;

use super::super::compact::{compact_messages, openai_error_response};
use super::msg_entry::MsgEntry;

pub(super) struct TemplateOut {
    pub(super) prompt_tokens: Vec<u32>,
    /// 2026-09-26: The caller's value, or `true` when the rendered prompt ends in an
    /// unclosed think block.
    pub(super) enable_thinking: bool,
    pub(super) thinking_budget: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(super) fn render_template(
    state: &Arc<AppState>,
    tools: &[crate::tool_parser::ToolDefinition],
    messages: &[MsgEntry],
    image_pad_counts: &[usize],
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    reasoning_effort: Option<crate::ir::ReasoningEffort>,
    preserve_thinking: Option<bool>,
    // 2026-10-15: `prepare::template_renders_tools`: tools active, or GLM-5.3's
    // `tool_choice` `none` (MODEL.toml `render_tools_on_tool_choice_none`).
    render_tools: bool,
    close_open_think: bool,
) -> Result<TemplateOut, Response> {
    let template_thinking = enable_thinking;

    let json_messages =
        build_json_messages_for(messages, state.tokenizer.uses_deepseek_v4_encoding());
    let jinja_tools = template_tools(
        tools,
        render_tools,
        state.tokenizer.uses_deepseek_v4_encoding(),
        state.chat.prompt.tscg,
    );

    let auto_compact_active = state
        .auto_compact_threshold
        .map(|t| t > 0.0)
        .unwrap_or(false);
    let json_messages = if auto_compact_active && json_messages.len() > 4 {
        let trial_tokens = state
            .tokenizer
            .apply_chat_template_openai_with_effort(
                &json_messages,
                jinja_tools.as_deref(),
                template_thinking,
                state.behavior.disable_tool_steering,
                reasoning_effort.map(crate::ir::ReasoningEffort::as_str),
                preserve_thinking,
            )
            .map(|t| t.len())
            .unwrap_or(0);
        if trial_tokens > (state.max_seq_len as f32 * 0.70) as usize {
            compact_messages(&json_messages, trial_tokens, state.max_seq_len)
        } else {
            json_messages
        }
    } else {
        json_messages
    };

    let prompt_tokens = match state.tokenizer.apply_chat_template_openai_with_effort(
        &json_messages,
        jinja_tools.as_deref(),
        template_thinking,
        state.behavior.disable_tool_steering,
        reasoning_effort.map(crate::ir::ReasoningEffort::as_str),
        preserve_thinking,
    ) {
        Ok(t) => t,
        Err(e) => {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Tokenization error: {e}"),
            ));
        }
    };

    let mut prompt_tokens = if image_pad_counts.iter().any(|&c| c > 1) {
        // 2026-09-26: Pass the checkpoint's declared pad ids. `expand_vision_pads` falls
        // back to probing the `<|image_pad|>` / `<|video_pad|>` literals only for an id
        // of 0, and a tokenizer without those literals expands nothing.
        let declared = state
            .vision_config
            .as_ref()
            .map(|v| (v.image_pad_token_id, v.video_pad_token_id))
            .unwrap_or((0, 0));
        state
            .tokenizer
            .expand_vision_pads(prompt_tokens, image_pad_counts, declared)
    } else {
        prompt_tokens
    };

    let (enable_thinking, thinking_budget) = resolve_think_tail(
        &mut prompt_tokens,
        ThinkTokens {
            start: state.think_start_token_id,
            end: state.think_end_token_id,
        },
        enable_thinking,
        thinking_budget,
        close_open_think,
        // 2026-10-14: `effort_prompt_only` (GLM-5.3): template-forced thinking gets no
        // budget, as the official template's default (effort Max) has none.
        (!state.behavior.effort_prompt_only).then_some(state.behavior.max_thinking_budget),
    );

    Ok(TemplateOut {
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    })
}

/// 2026-09-26: With TSCG on, the parser's `system_prompt()` already put compact tool
/// signatures into the first system message (`prepare::inject_tool_system_prompt`),
/// so the template gets no `tools` and does not render the JSON schemas again. The
/// DeepSeek-V4 encoding always gets `tools`.
///
/// 2026-10-15: Moved out of `render_template` so the tool-render tests run it.
pub(super) fn template_tools(
    tools: &[crate::tool_parser::ToolDefinition],
    render_tools: bool,
    deepseek_v4_encoding: bool,
    tscg: bool,
) -> Option<Vec<serde_json::Value>> {
    (render_tools && (deepseek_v4_encoding || !tscg)).then(|| {
        tools
            .iter()
            .map(|t| serde_json::to_value(t).unwrap_or_default())
            .collect()
    })
}

/// 2026-10-12: The model's think delimiter token ids.
#[derive(Clone, Copy)]
struct ThinkTokens {
    start: Option<u32>,
    end: Option<u32>,
}

/// 2026-09-26: A think-start in the last 8 prompt tokens with no think-end after it
/// means the template forced thinking on; thinking is then enabled with the model's
/// `max_thinking_budget`.
///
/// 2026-10-14: `forced_budget` is that budget: `None` under `effort_prompt_only` (no
/// budget, as on the official model, whose template always opens `<think>`).
///
/// 2026-10-12: `close_open_think` (the request itself asked for thinking off) appends
/// the think-end token to such a tail instead, so the prompt ends `<think></think>` and
/// thinking stays off. This is the convention of the official zai-org GLM-4.5
/// template (`enable_thinking` false renders `<think></think>` after `<|assistant|>`);
/// the GLM-5.3 template has no `enable_thinking` branch and always ends in an open
/// `<think>`. Without `close_open_think` the prompt and result are unchanged.
fn resolve_think_tail(
    prompt_tokens: &mut Vec<u32>,
    think: ThinkTokens,
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    close_open_think: bool,
    forced_budget: Option<u32>,
) -> (bool, Option<u32>) {
    let Some(think_start) = think.start else {
        return (enable_thinking, thinking_budget);
    };
    let tail = &prompt_tokens[prompt_tokens.len().saturating_sub(8)..];
    let last_start = tail.iter().rposition(|t| *t == think_start);
    let has_unclosed_think = match (last_start, think.end) {
        (Some(si), Some(end_tok)) => !tail[si + 1..].contains(&end_tok),
        (Some(_), None) => true,
        (None, _) => false,
    };
    if !has_unclosed_think || enable_thinking {
        return (enable_thinking, thinking_budget);
    }
    if close_open_think && let Some(end_tok) = think.end {
        tracing::debug!("Thinking off by request: closed the template's open \\<think\\>");
        prompt_tokens.push(end_tok);
        return (false, None);
    }
    tracing::info!(
        "Template-forced thinking detected (unclosed \\<think\\> in prompt tail) — \
         overriding enable_thinking=true with budget={forced_budget:?}",
    );
    (true, forced_budget)
}

/// 2026-09-26: Build the Jinja-facing JSON message array from [`MsgEntry`] values,
/// without `tool_call_id`. Pure (no tokenizer or state):
///   * no media → `content` is a plain string,
///   * media    → `content` is `[<one marker per media item, in
///     `MsgEntry::media` order>, {type:text}]` (text part omitted when empty),
///   * `tool_calls` / `reasoning_content` attached when present.
///
/// Roles are copied as given; `build_msg_entries` maps `developer` to `system`
/// unless `preserve_developer_role` is set.
pub(super) fn build_json_messages(messages: &[MsgEntry]) -> Vec<serde_json::Value> {
    build_json_messages_for(messages, false)
}

fn build_json_messages_for(
    messages: &[MsgEntry],
    include_tool_call_ids: bool,
) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|m| {
            let content_val = if !m.media.is_empty() {
                let mut items: Vec<serde_json::Value> = Vec::with_capacity(m.media.len() + 1);
                // 2026-09-26: One marker per media item, in content order, the same order
                // in which `collect_message_media` appends the pad counts and encoder
                // inputs. The Qwen templates (`jinja-templates/qwen3_5_moe.jinja`) emit one
                // pad per marker left to right, and `expand_vision_pads` gives the i-th
                // pad the i-th count.
                for kind in &m.media {
                    items.push(match kind {
                        crate::ir::MediaKind::Image => serde_json::json!({"type": "image"}),
                        crate::ir::MediaKind::Video => serde_json::json!({"type": "video"}),
                    });
                }
                if !m.content.is_empty() {
                    items.push(serde_json::json!({"type": "text", "text": m.content}));
                }
                serde_json::Value::Array(items)
            } else {
                serde_json::Value::String(m.content.clone())
            };
            let mut msg = serde_json::json!({"role": m.role, "content": content_val});
            if let Some(ref tcs) = m.tool_calls {
                msg["tool_calls"] = serde_json::Value::Array(tcs.clone());
            }
            if include_tool_call_ids && let Some(ref id) = m.tool_call_id {
                msg["tool_call_id"] = serde_json::Value::String(id.clone());
            }
            // 2026-09-26: Templates such as `jinja-templates/qwen3_5_moe.jinja` read
            // `message.reasoning_content` to render a past assistant turn's think block.
            if let Some(ref rc) = m.reasoning_content {
                msg["reasoning_content"] = serde_json::Value::String(rc.clone());
            }
            msg
        })
        .collect()
}

#[cfg(test)]
mod json_message_tests {
    use super::MsgEntry;
    use super::build_json_messages;

    fn entry(role: &str, content: &str, image_count: usize) -> MsgEntry {
        MsgEntry {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            media: vec![crate::ir::MediaKind::Image; image_count],
            reasoning_content: None,
        }
    }

    /// 2026-09-26: Golden for the `Vec<ir::Message>` → `build_msg_entries` →
    /// `build_json_messages` path, the JSON the Jinja chat template consumes. A change
    /// to the expected value is a change to rendered prompts.
    #[test]
    fn prompt_json_stability_gate() {
        use crate::ir::message::{Reasoning, ToolCall};
        use crate::ir::{ContentPart, Message, Role};

        fn text_msg(role: Role, t: &str) -> Message {
            Message {
                role,
                content: vec![ContentPart::Text(t.into())],
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
                reasoning: None,
                tool_error: false,
            }
        }

        let mut assistant = text_msg(Role::Assistant, "Sure.");
        assistant.reasoning = Some(Reasoning {
            text: "plan the verification".into(),
        });
        assistant.tool_calls = vec![
            ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "cargo test"}),
            },
            ToolCall {
                id: "c2".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "a.rs"}),
            },
        ];
        let mut tool_result = text_msg(Role::Tool, "total 0");
        tool_result.tool_call_id = Some("c1".into());
        tool_result.name = Some("bash".into());

        let msgs = vec![
            text_msg(
                Role::System,
                "You are helpful.\nworking directory: /tmp/proj",
            ),
            text_msg(Role::Other("developer".into()), "be terse"),
            text_msg(Role::User, "run the tests"),
            assistant,
            tool_result,
            text_msg(Role::User, "thanks"),
        ];

        let out = super::super::msg_entry::build_msg_entries(
            None,
            None,
            &crate::api::chat::remote_image::RemoteImagePolicy::default(),
            &crate::api::chat::msg_entry::VideoDecode {
                ffmpeg: &metrale_model_layers::video_decode_ffmpeg::FfmpegPolicy {
                    enabled: false,
                    ..Default::default()
                },
                fps: 2.0,
            },
            &msgs,
            true,
            &crate::api::chat::levers::ChatLevers::OFF,
            false,
        )
        .expect("fixture builds");
        assert_eq!(out.cwd_hint.as_deref(), Some("/tmp/proj"));
        let json = build_json_messages(&out.messages);

        let expected = serde_json::json!([
            {
                "role": "system",
                "content": "You are helpful.\nworking directory: /tmp/proj\n<environment>\nworking_directory: /tmp/proj\n</environment>"
            },
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "run the tests"},
            {
                "role": "assistant",
                "content": "Sure.",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": {"command": "cargo test"}}},
                    {"id": "c2", "type": "function", "function": {"name": "read", "arguments": {"path": "a.rs"}}}
                ],
                "reasoning_content": "plan the verification"
            },
            {"role": "tool", "content": "total 0"},
            {"role": "user", "content": "thanks"}
        ]);
        assert_eq!(
            serde_json::Value::Array(json.clone()),
            expected,
            "prompt JSON drifted — this breaks kv-cache prefix stability:\n{}",
            serde_json::to_string_pretty(&json).unwrap()
        );
    }

    #[test]
    fn plain_text_message_serializes_to_string_content() {
        let out = build_json_messages(&[entry("user", "hi", 0)]);
        assert_eq!(
            out,
            vec![serde_json::json!({"role": "user", "content": "hi"})]
        );
    }

    #[test]
    fn images_expand_to_structured_content_array_with_text_last() {
        let out = build_json_messages(&[entry("user", "look", 2)]);
        assert_eq!(
            out,
            vec![serde_json::json!({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "look"}
                ]
            })]
        );
    }

    #[test]
    fn empty_text_with_images_omits_text_part() {
        let out = build_json_messages(&[entry("user", "", 1)]);
        assert_eq!(
            out,
            vec![serde_json::json!({"role": "user", "content": [{"type": "image"}]})]
        );
    }

    /// 2026-09-26: Markers keep the client's mixed video/image order. The template
    /// renders this array left to right, so its order is the order the model reads.
    #[test]
    fn media_markers_render_in_the_clients_order() {
        use crate::ir::MediaKind;

        let mut e = entry("user", "which came first?", 0);
        e.media = vec![MediaKind::Video, MediaKind::Image, MediaKind::Video];
        let out = build_json_messages(&[e]);
        assert_eq!(
            out,
            vec![serde_json::json!({
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "image"},
                    {"type": "video"},
                    {"type": "text", "text": "which came first?"}
                ]
            })]
        );
    }

    #[test]
    fn tool_calls_and_reasoning_are_attached() {
        let mut e = entry("assistant", "", 0);
        e.tool_calls = Some(vec![serde_json::json!({"id": "c1"})]);
        e.reasoning_content = Some("because".to_string());
        let out = build_json_messages(&[e]);
        assert_eq!(
            out,
            vec![serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1"}],
                "reasoning_content": "because"
            })]
        );
    }
}

#[cfg(test)]
#[path = "template_think_tail_tests.rs"]
mod think_tail_tests;

#[cfg(test)]
#[path = "template_tool_render_tests.rs"]
mod tool_render_tests;
