// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for wire-type deserialization and the stop-reason,
//! tool-choice and tool-definition conversions.
//!
//! Owner: server (Anthropic adapter) tests.
//! Invariants: none beyond the types.

use crate::tool_parser;

use super::super::helpers::convert_stop_reason;
use super::super::types::{
    AnthropicTool, AnthropicToolChoice, ContentBlock, MessagesRequest, SystemContent,
};

#[test]
fn stop_reason_maps_the_whole_wire_vocabulary() {
    assert_eq!(convert_stop_reason("stop"), "end_turn");
    assert_eq!(convert_stop_reason("tool_calls"), "tool_use");
    assert_eq!(convert_stop_reason("length"), "max_tokens");
    assert_eq!(convert_stop_reason("content_filter"), "refusal");
    // 2026-09-26: A deadline cut is a truncation, so it maps to `max_tokens`
    // rather than the `end_turn` fallback.
    assert_eq!(
        convert_stop_reason(crate::ir::FINISH_REASON_TIMEOUT),
        "max_tokens"
    );
    assert_eq!(convert_stop_reason("something_new"), "end_turn");
}

#[test]
fn stop_reason_covers_every_finish_reason_wire_string() {
    // 2026-09-26: The streaming translator passes `FinishReason::as_wire()`
    // to `convert_stop_reason`, and a wire string that reaches the `_` arm
    // becomes "end_turn". This pins each variant's pairing.
    use crate::ir::FinishReason;
    for (reason, expected) in [
        (FinishReason::Stop, "end_turn"),
        (FinishReason::Length, "max_tokens"),
        (FinishReason::ToolCalls, "tool_use"),
        (FinishReason::ContentFilter, "refusal"),
        (
            FinishReason::Other(crate::ir::FINISH_REASON_TIMEOUT.to_string()),
            "max_tokens",
        ),
    ] {
        assert_eq!(
            convert_stop_reason(reason.as_wire()),
            expected,
            "finish reason {reason:?}"
        );
    }
}

#[test]
fn repetition_guard_cut_stays_max_tokens_for_anthropic() {
    // 2026-10-14: "stop" on the OpenAI wire, but Anthropic has no stop_reason field
    // beside it, so the cut is reported as cut short, as before (#82 guard fix).
    use super::super::helpers::anthropic_stop_reason;
    use crate::ir::FinishReason;
    assert_eq!(
        anthropic_stop_reason(&FinishReason::Repetition),
        "max_tokens"
    );
    assert_eq!(anthropic_stop_reason(&FinishReason::Stop), "end_turn");
    assert_eq!(anthropic_stop_reason(&FinishReason::Length), "max_tokens");
}

fn choice(kind: &str, name: Option<&str>) -> AnthropicToolChoice {
    AnthropicToolChoice {
        choice_type: kind.to_string(),
        name: name.map(str::to_string),
    }
}

#[test]
fn tool_choice_modes_map_to_openai_vocabulary() {
    let cases = [("any", "required"), ("auto", "auto"), ("none", "none")];
    for (anthropic, openai) in cases {
        match tool_parser::ToolChoice::from(&choice(anthropic, None)) {
            tool_parser::ToolChoice::Mode(m) => assert_eq!(m, openai, "{anthropic}"),
            other => panic!("{anthropic}: expected Mode, got {other:?}"),
        }
    }
}

#[test]
fn tool_choice_specific_tool_carries_the_name() {
    match tool_parser::ToolChoice::from(&choice("tool", Some("get_weather"))) {
        tool_parser::ToolChoice::Specific { function } => {
            assert_eq!(function.name, "get_weather");
        }
        other => panic!("expected Specific, got {other:?}"),
    }
}

#[test]
fn tool_choice_tool_without_a_name_degrades_to_auto() {
    // 2026-09-26: A `tool` choice without a `name` is malformed; it becomes
    // "auto" so the request is still served.
    match tool_parser::ToolChoice::from(&choice("tool", None)) {
        tool_parser::ToolChoice::Mode(m) => assert_eq!(m, "auto"),
        other => panic!("expected Mode(auto), got {other:?}"),
    }
    match tool_parser::ToolChoice::from(&choice("wat", None)) {
        tool_parser::ToolChoice::Mode(m) => assert_eq!(m, "auto"),
        other => panic!("expected Mode(auto), got {other:?}"),
    }
}

#[test]
fn tool_definition_carries_description_and_schema() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    });
    let tool = AnthropicTool {
        name: "get_weather".to_string(),
        description: Some("Get weather".to_string()),
        input_schema: schema.clone(),
    };
    let def = tool_parser::ToolDefinition::from(&tool);
    assert_eq!(def.tool_type, "function");
    assert_eq!(def.function.name, "get_weather");
    assert_eq!(def.function.description.as_deref(), Some("Get weather"));
    assert_eq!(def.function.parameters.as_ref(), Some(&schema));
}

#[test]
fn tool_definition_without_a_description_keeps_none() {
    let tool = AnthropicTool {
        name: "x".to_string(),
        description: None,
        input_schema: serde_json::json!({"type": "object"}),
    };
    let def = tool_parser::ToolDefinition::from(&tool);
    assert!(def.function.description.is_none());
}

#[test]
fn request_deserializes_string_system_and_defaults() {
    let req: MessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "qwen3-80b",
        "max_tokens": 1024,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hello!"}],
    }))
    .expect("MessagesRequest");
    assert_eq!(req.model, "qwen3-80b");
    assert_eq!(req.max_tokens, 1024);
    assert!(matches!(req.system, Some(SystemContent::Text(ref s)) if s == "You are helpful."));
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, "user");
    // 2026-09-26: Unset sampling fields stay `None`, so `build_sampling`
    // takes them from the preset or the server default.
    assert!(req.temperature.is_none());
    assert!(req.top_k.is_none());
    assert!(req.top_p.is_none());
    assert!(req.tools.is_none());
    assert!(req.tool_choice.is_none());
    assert!(req.thinking.is_none());
    assert!(req.stop_sequences.is_empty());
    assert!(!req.stream);
}

#[test]
fn thinking_config_deserializes_type_and_budget() {
    let req: MessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "qwen3-80b",
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "messages": [{"role": "user", "content": "Think hard."}],
    }))
    .expect("MessagesRequest");
    let t = req.thinking.expect("thinking config");
    assert_eq!(t.thinking_type, "enabled");
    assert_eq!(t.budget_tokens, Some(4096));
}

#[test]
fn tool_result_is_error_defaults_to_none_when_absent() {
    let with_err: ContentBlock = serde_json::from_str(
        r#"{"type":"tool_result","tool_use_id":"x","content":"oops","is_error":true}"#,
    )
    .expect("tool_result with is_error");
    match with_err {
        ContentBlock::ToolResult { is_error, .. } => assert_eq!(is_error, Some(true)),
        other => panic!("wrong variant: {other:?}"),
    }

    let no_field: ContentBlock =
        serde_json::from_str(r#"{"type":"tool_result","tool_use_id":"x","content":"ok"}"#)
            .expect("tool_result without is_error");
    match no_field {
        ContentBlock::ToolResult { is_error, .. } => assert_eq!(is_error, None),
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn unrecognised_block_types_deserialize_to_unknown_instead_of_failing() {
    // 2026-09-26: An unknown block type deserializes to `Unknown`, which
    // lowering skips, so the rest of the request is still served.
    let block: ContentBlock = serde_json::from_str(r#"{"type":"redacted_thinking","data":"AAAA"}"#)
        .expect("unknown block type must parse");
    assert!(matches!(block, ContentBlock::Unknown));
}
