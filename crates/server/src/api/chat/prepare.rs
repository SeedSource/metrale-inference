// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Prompt preparation shared by `chat_completions_inner` and
//! `/v1/messages/count_tokens`: tool gating and tool-prompt injection, the
//! `MsgEntry` build, thinking resolution, and the template render.
//!
//! Owner: server (chat API).
//! Invariants: none beyond the types.

use axum::response::Response;
use std::sync::Arc;

use crate::AppState;
use crate::ir::ChatRequest;

use super::{msg_entry, template, thinking};

/// 2026-09-26: Outputs of [`prepare_chat_prompt`].
pub(crate) struct PreparedChat {
    pub(crate) tools_active: bool,
    pub(crate) cwd_hint: Option<String>,
    pub(crate) image_pixels: Vec<metrale_model_layers::VisionItem>,
    pub(crate) prompt_tokens: Vec<u32>,
    pub(crate) enable_thinking: bool,
    pub(crate) thinking_budget: Option<u32>,
}

/// 2026-09-26: Prepare `req`'s prompt tokens, media and thinking settings.
/// With tools active, the parser's tool prompt is added to `req.messages`.
#[allow(clippy::result_large_err)]
pub(crate) fn prepare_chat_prompt(
    state: &Arc<AppState>,
    req: &mut ChatRequest,
) -> Result<PreparedChat, Response> {
    // 2026-09-26: Tools are active when a parser is configured, the request
    // declares tools, and `tool_choice` is not `none`.
    let tools_active = state.tool_call_parser.is_some()
        && !req.tools.is_empty()
        && !req.tool_choice.as_ref().is_some_and(|tc| tc.is_none());

    if tools_active && let Some(ref parser) = state.tool_call_parser {
        let default_choice = crate::tool_parser::ToolChoice::Mode("auto".to_string());
        let tool_choice = req.tool_choice.as_ref().unwrap_or(&default_choice);
        let tool_prompt = parser_tool_prompt(
            parser.as_ref(),
            &req.tools,
            tool_choice,
            &state.chat.prompt,
            state.tokenizer.uses_native_qwen_tool_template(),
        );
        inject_tool_system_prompt(&mut req.messages, tool_prompt);
    }

    tracing::info!(
        "Request: model={}, messages={}, tools={}, tools_active={}, tool_choice={:?}, stream={}, temp={:?}, max_tokens={}, freq_pen={:?}, rep_pen={:?}",
        req.model,
        req.messages.len(),
        req.tools.len(),
        tools_active,
        req.tool_choice,
        req.stream,
        req.sampling.temperature,
        req.max_tokens,
        req.sampling.frequency_penalty,
        req.sampling.repetition_penalty,
    );

    let _t_phase = std::time::Instant::now();

    let msg_entry::BuildOut {
        messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
    } = msg_entry::build_msg_entries(
        state.vision_config.as_ref(),
        state.vision_max_pixels,
        &state.remote_image_policy,
        &msg_entry::VideoDecode {
            ffmpeg: &state.video_ffmpeg,
            fps: state.video_fps,
        },
        &req.messages,
        tools_active,
        &state.chat,
        state.tokenizer.uses_deepseek_v4_encoding(),
    )?;
    let us_msg_entry = _t_phase.elapsed().as_micros();

    // 2026-09-26: The request's thinking directive wins; when it is not
    // explicit, the `--default-chat-template-kwargs` directive applies; when
    // that is unspecified too, `resolve_thinking` uses MODEL.toml
    // `[behavior].thinking_default`.
    let mut thinking_directive = req.thinking;
    if !thinking_directive.is_explicit() {
        thinking_directive = state.default_thinking;
    }
    let gen_max =
        thinking::generation_max_tokens(req.max_tokens, tools_active, state.tool_max_tokens);
    let (enable_thinking, thinking_budget) =
        thinking::resolve_thinking(state, thinking_directive, gen_max as u32, tools_active);
    let reasoning_effort = effective_reasoning_effort(
        req.reasoning_effort,
        state.default_reasoning_effort,
        enable_thinking,
    );
    let us_thinking = _t_phase.elapsed().as_micros() - us_msg_entry;

    let template::TemplateOut {
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    } = template::render_template(
        state,
        &req.tools,
        &messages,
        &image_pad_counts,
        enable_thinking,
        thinking_budget,
        reasoning_effort,
        // 2026-09-26: The request's `preserve_thinking` wins, then
        // `behavior.preserve_thinking` (MODEL.toml, overridden by
        // `--default-chat-template-kwargs`). `None` leaves the template
        // variable undefined.
        req.preserve_thinking.or(state.behavior.preserve_thinking),
        tools_active,
        // 2026-10-12: Only the request's own `Off` closes a template-opened think
        // block; `--disable-thinking`, the server default directive and the model
        // default keep the prompt unchanged.
        matches!(req.thinking, crate::ir::ThinkingDirective::Off),
    )?;
    if state.chat.phase_timing {
        let us_template = _t_phase.elapsed().as_micros() - us_msg_entry - us_thinking;
        tracing::info!(
            "CHAT_PHASE prepare: msg_entry={us_msg_entry}us thinking={us_thinking}us \
             template_render_and_tokenize={us_template}us prompt_tokens={}",
            prompt_tokens.len()
        );
    }

    Ok(PreparedChat {
        tools_active,
        cwd_hint,
        image_pixels,
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    })
}

/// 2026-09-26: The reasoning effort handed to the template: with thinking on,
/// the request's value, else the `--default-chat-template-kwargs` default;
/// `None` otherwise. `tokenizer/chat_render.rs` renders `None` as `"medium"`
/// when thinking is on and as `"none"` when it is off.
fn effective_reasoning_effort(
    request: Option<crate::ir::ReasoningEffort>,
    server_default: Option<crate::ir::ReasoningEffort>,
    enable_thinking: bool,
) -> Option<crate::ir::ReasoningEffort> {
    if enable_thinking {
        request.or(server_default)
    } else {
        None
    }
}

/// 2026-09-26: The tool prompt the parser contributes. With a native Qwen tool
/// template, TSCG off and the `qwen3_coder` or `qwen3_xml` parser, only the
/// tool-choice instruction; otherwise the parser's full system prompt.
pub(crate) fn parser_tool_prompt(
    parser: &dyn crate::tool_parser::ToolCallParser,
    tools: &[crate::tool_parser::ToolDefinition],
    choice: &crate::tool_parser::ToolChoice,
    levers: &crate::tool_parser::PromptLevers,
    native_qwen_template: bool,
) -> String {
    if native_qwen_template && !levers.tscg && matches!(parser.name(), "qwen3_coder" | "qwen3_xml")
    {
        // 2026-09-26: The checkpoint template renders the tool schemas, so
        // only the tool-choice instruction is added.
        let mut prompt = String::new();
        crate::tool_parser::append_tool_choice_instruction(&mut prompt, choice);
        return prompt;
    }
    parser.system_prompt(tools, choice, levers)
}

/// 2026-09-26: Prepend `tool_prompt` and a blank line to the leading system
/// message, or insert it as a new leading system message. An empty prompt
/// changes nothing.
pub(crate) fn inject_tool_system_prompt(
    messages: &mut Vec<crate::ir::Message>,
    tool_prompt: String,
) {
    if tool_prompt.is_empty() {
        return;
    }

    if let Some(first) = messages
        .first_mut()
        .filter(|m| m.role == crate::ir::Role::System)
    {
        first.prepend_text(&format!("{tool_prompt}\n\n"));
    } else {
        messages.insert(0, crate::ir::Message::synthetic_system(tool_prompt));
    }
}

#[cfg(test)]
mod tests {
    use super::{effective_reasoning_effort, inject_tool_system_prompt};
    use crate::ir::{Message, ReasoningEffort, Role};

    /// 2026-09-26: The request beats the server default, the server default
    /// fills a silent request, and thinking off gives `None` from both.
    #[test]
    fn effort_resolution_contract() {
        let low = Some(ReasoningEffort::Low);
        let max = Some(ReasoningEffort::Max);
        assert_eq!(effective_reasoning_effort(low, max, true), low);
        assert_eq!(effective_reasoning_effort(None, max, true), max);
        // 2026-09-26: `chat_render.rs` renders this `None` as "medium".
        assert_eq!(effective_reasoning_effort(None, None, true), None);
        assert_eq!(effective_reasoning_effort(low, max, false), None);
        assert_eq!(effective_reasoning_effort(None, max, false), None);
    }

    #[test]
    fn empty_tool_prompt_does_not_change_existing_system_message() {
        let mut messages = vec![Message::synthetic_system("original".into())];
        let before = messages.clone();

        inject_tool_system_prompt(&mut messages, String::new());

        assert_eq!(messages, before);
    }

    #[test]
    fn empty_tool_prompt_does_not_insert_system_message() {
        let mut messages = vec![Message {
            role: Role::User,
            content: vec![crate::ir::ContentPart::Text("hello".into())],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
            tool_error: false,
        }];
        let before = messages.clone();

        inject_tool_system_prompt(&mut messages, String::new());

        assert_eq!(messages, before);
    }

    #[test]
    fn nonempty_tool_prompt_preserves_existing_injection_behavior() {
        let mut messages = vec![Message::synthetic_system("original".into())];

        inject_tool_system_prompt(&mut messages, "tool instructions".into());

        assert_eq!(messages[0].text(), "tool instructions\n\noriginal");
    }
}

#[cfg(test)]
#[path = "../../tokenizer/tests/native_tool_prompt.rs"]
mod native_tool_prompt;
