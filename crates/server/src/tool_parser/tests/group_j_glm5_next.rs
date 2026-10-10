// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tool calling for GLM-5.3-Flash (`model_type = "glm5_next"`).
//! `tool_defaults.toml` maps it to the `poolside_v1` parser; these tests pin
//! that mapping and the
//! `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
//! format through that parser.
//!
//! Without the mapping, and with no parser set on the command line or in
//! MODEL.toml, `resolve_tool_call_parser` returns `None`, `tools_active` is
//! false (`api/chat/prepare.rs`), and the chat template is rendered without
//! the request's tools.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

/// 2026-09-26: A two-argument call in the GLM format.
const GLM_TWO_ARG: &str = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value><arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>";

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"},
                    "days": {"type": "integer"},
                    "verbose": {"type": "boolean"}
                },
                "required": ["location"]
            })),
        },
    }
}

fn args_of(call: &ToolCall) -> serde_json::Value {
    serde_json::from_str(&call.function.arguments).expect("arguments are valid JSON")
}

/// 2026-09-26: `tool_defaults.toml` maps `glm5_next` to `poolside_v1`.
#[test]
fn glm5_next_is_registered_to_the_poolside_v1_wire_format() {
    let defaults: toml::Value =
        toml::from_str(include_str!("../../../tool_defaults.toml")).expect("tool_defaults parses");
    let fmt_str = defaults
        .get("model_type")
        .and_then(|t| t.get("glm5_next"))
        .and_then(|s| s.as_str())
        .expect("tool_defaults [model_type] must register glm5_next");
    assert_eq!(
        fmt_str, "poolside_v1",
        "GLM-5.3 emits the poolside_v1 envelope; see chat_template.jinja"
    );
    let fmt: ToolCallFormat = fmt_str.parse().expect("glm5_next tool format parses");
    assert_eq!(fmt.into_parser().name(), "poolside_v1");
}

/// 2026-09-26: `GLM_TWO_ARG` gives one call with both arguments, and no
/// content is left over.
#[test]
fn glm_template_format_line_parses_to_a_structured_call() {
    let (content, calls) = parse_tool_calls_promoting_bare_names(GLM_TWO_ARG);
    assert_eq!(calls.len(), 1, "one call from the template's own example");
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(
        args_of(&calls[0]),
        serde_json::json!({"location": "Paris", "unit": "celsius"}),
        "both arg_key/arg_value pairs, in order, as strings"
    );
    assert!(
        content.as_deref().unwrap_or("").trim().is_empty(),
        "the envelope is entirely consumed, leaving no stray content"
    );
}

/// 2026-09-26: Text before the call is kept as content.
#[test]
fn leading_prose_is_preserved_as_content() {
    let text = format!("Let me check that for you.{GLM_TWO_ARG}");
    let (content, calls) = parse_tool_calls_promoting_bare_names(&text);
    assert_eq!(calls.len(), 1);
    assert_eq!(
        content.as_deref().unwrap().trim(),
        "Let me check that for you."
    );
}

/// 2026-09-26: Everything before `</think>` is dropped before parsing, so only
/// the call after it counts.
#[test]
fn calls_inside_thinking_are_not_invocations() {
    let text = format!(
        "<think>I could call <tool_call>get_weather<arg_key>location</arg_key><arg_value>Berlin</arg_value></tool_call> but Paris was asked.</think>{GLM_TWO_ARG}"
    );
    let (_, calls) = parse_tool_calls_promoting_bare_names(&text);
    assert_eq!(calls.len(), 1, "only the call after </think> is real");
    assert_eq!(args_of(&calls[0])["location"], "Paris");
}

/// 2026-09-26: With bare names promoted, a bare name inside the envelope is a
/// zero-argument call.
#[test]
fn zero_argument_call_is_a_bare_name_in_the_envelope() {
    let (_, calls) = parse_tool_calls_promoting_bare_names("<tool_call>get_status</tool_call>");
    assert_eq!(calls.len(), 1, "zero-arg call must not be dropped");
    assert_eq!(calls[0].function.name, "get_status");
    assert_eq!(args_of(&calls[0]), serde_json::json!({}));
}

/// 2026-09-26: Two envelopes back to back, with no separator, give two calls.
#[test]
fn two_calls_in_one_turn_are_both_extracted() {
    let text = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>Lyon</arg_value></tool_call>";
    let (_, calls) = parse_tool_calls_promoting_bare_names(text);
    assert_eq!(calls.len(), 2);
    assert_eq!(args_of(&calls[0])["location"], "Paris");
    assert_eq!(args_of(&calls[1])["location"], "Lyon");
}

/// 2026-10-10: A bare number on the wire for a `string` property (BFCL's
/// `123123` ids) reaches the client as the string the schema declares.
#[test]
fn bare_number_for_a_string_property_stays_a_string() {
    let text = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>12345</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
    let (_, mut calls) = parse_tool_calls_promoting_bare_names(text);
    assert_eq!(calls.len(), 1);
    coerce_all(&mut calls, &[weather_tool()]);
    let args = args_of(&calls[0]);
    assert_eq!(
        args["location"],
        serde_json::json!("12345"),
        "string, not 12345"
    );
    assert_eq!(args["days"], serde_json::json!(3), "integer, not \"3\"");
}

/// 2026-09-26: `parse_poolside_v1_call` reads an `<arg_value>` as JSON when it
/// parses (`3`, `true`) and as a string otherwise; after `coerce_all`, `days`
/// is an integer and `verbose` a boolean, as the schema declares.
#[test]
fn untyped_wire_values_are_coerced_to_the_schema() {
    let text = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value><arg_key>verbose</arg_key><arg_value>true</arg_value></tool_call>";
    let (_, mut calls) = parse_tool_calls_promoting_bare_names(text);
    assert_eq!(calls.len(), 1);
    assert!(
        PoolsideV1Parser.wants_typed_arguments(),
        "GLM's wire format carries no types; coercion is mandatory"
    );
    coerce_all(&mut calls, &[weather_tool()]);
    let args = args_of(&calls[0]);
    assert_eq!(
        args["location"],
        serde_json::json!("Paris"),
        "stays a string"
    );
    assert_eq!(args["days"], serde_json::json!(3), "integer, not \"3\"");
    assert_eq!(
        args["verbose"],
        serde_json::json!(true),
        "bool, not \"true\""
    );
}

/// 2026-09-26: `format_tool_calls` writes a string value raw and any other
/// value as JSON, and the text parses back to the same call.
#[test]
fn formatted_history_matches_the_templates_own_rendering_rule() {
    let calls = vec![IncomingToolCall {
        id: Some("call_1".to_string()),
        function: IncomingFunction {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"Paris","days":3}"#.to_string(),
        },
    }];
    let rendered = PoolsideV1Parser.format_tool_calls(&calls);
    assert_eq!(
        rendered,
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
        "string raw, non-string via tojson — matches chat_template.jinja"
    );
    let (_, reparsed) = parse_tool_calls_promoting_bare_names(&rendered);
    assert_eq!(reparsed.len(), 1);
    assert_eq!(reparsed[0].function.name, "get_weather");
    assert_eq!(args_of(&reparsed[0])["location"], "Paris");
}
