// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Tool-parser tests for the MiniMax XML envelope and the
//! Qwen3-Coder `<function=…>` block form.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

use super::*;

#[test]
fn parse_minimax_xml_single_param() {
    let input = "<minimax:tool_call>\n\
            <invoke name=\"get_weather\">\n\
            <parameter name=\"location\">Paris</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let (content, calls) = parse_tool_calls(input);
    assert!(
        content.is_none(),
        "expected no leading content, got {content:?}"
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Paris");
}

#[test]
fn parse_minimax_xml_multiple_params() {
    let input = "<minimax:tool_call>\n\
            <invoke name=\"search\">\n\
            <parameter name=\"query\">rust async</parameter>\n\
            <parameter name=\"limit\">10</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["query"], "rust async");
    assert_eq!(args["limit"], "10");
}

#[test]
fn parse_minimax_xml_with_content_prefix() {
    let input = "Let me check. <minimax:tool_call>\n\
            <invoke name=\"ls\">\n\
            <parameter name=\"path\">/tmp</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let (content, calls) = parse_tool_calls(input);
    assert_eq!(content.as_deref(), Some("Let me check."));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "ls");
}

#[test]
fn minimax_xml_format_tool_calls_roundtrip() {
    let parser = MinimaxXmlParser;
    let call = IncomingToolCall {
        id: None,
        function: IncomingFunction {
            name: "get_weather".into(),
            arguments: "{\"location\":\"Tokyo\"}".into(),
        },
    };
    let formatted = parser.format_tool_calls(&[call]);
    let (_, parsed) = parse_tool_calls(&formatted);
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&parsed[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
}

#[test]
fn parse_qwen3_coder_empty_body_then_backfill() {
    // 2026-09-26: A call with no `<parameter=…>` parses to `{}`.
    // `backfill_required_params`, which the blocking and streaming paths both
    // run, adds the required `command` as `""`; validation then refuses it,
    // since `exec` is in `SHELL_FAMILY` and its command is empty.
    let input = "<tool_call>\n\
            <function=exec>\n\
            </function>\n\
            </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(
        calls.len(),
        1,
        "parser must yield the named call even with no params"
    );
    assert_eq!(calls[0].function.name, "exec");
    assert_eq!(
        calls[0].function.arguments, "{}",
        "no params → empty JSON object"
    );

    let tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "exec".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            })),
        },
    };
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    // 2026-10-03: backfill no longer fabricates `""`; the key stays absent and
    // `assess_tool_call` still refuses the shell call (hard).
    assert!(
        args.get("command").is_none(),
        "backfill must not add an omitted required string key"
    );
    let err = validate_single_tool_call(&calls[0], std::slice::from_ref(&tool))
        .expect_err("SHELL_FAMILY rejects `exec` with no command");
    assert!(
        err.contains("non-empty 'command'"),
        "rejection must name the offending key so the model can recover; got {err:?}"
    );
}

#[test]
fn parse_qwen3_coder_single_param() {
    let input = "<tool_call>\n\
            <function=get_weather>\n\
            <parameter=location>\nParis\n</parameter>\n\
            </function>\n\
            </tool_call>";
    let (c, calls) = parse_tool_calls(input);
    assert!(c.is_none());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Paris");
}
