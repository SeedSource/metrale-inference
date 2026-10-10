// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Tool-parser tests for `coerce_all` (type coercion and
//! empty-key repair) and the `Qwen3XmlParser` trait methods.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

fn make_tool(name: &str, props: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: None,
            parameters: Some(serde_json::json!({ "type": "object", "properties": props })),
        },
    }
}

fn make_call(name: &str, args: &str) -> ToolCall {
    ToolCall {
        id: "call_test".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: args.to_string(),
        },
    }
}

#[test]
fn coerce_integer_string() {
    let tools = vec![make_tool(
        "search",
        serde_json::json!({ "limit": { "type": "integer" } }),
    )];
    let mut calls = vec![make_call("search", r#"{"limit":"10"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], 10);
}

#[test]
fn coerce_number_float() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "score": { "type": "number" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"score":"3.5"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    let got = args["score"].as_f64().unwrap();
    assert!((got - 3.5).abs() < 1e-9, "expected 3.5, got {got}");
}

#[test]
fn coerce_boolean_lower() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "flag": { "type": "boolean" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"flag":"true"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["flag"], true);
}

#[test]
fn coerce_boolean_capitalized() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "flag": { "type": "boolean" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"flag":"False"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["flag"], false);
}

#[test]
fn coerce_array_string() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "items": { "type": "array" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"items":"[1,2,3]"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["items"], serde_json::json!([1, 2, 3]));
}

#[test]
fn coerce_object_string() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "opts": { "type": "object" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"opts":"{\"a\":1}"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["opts"]["a"], 1);
}

#[test]
fn coerce_null_string() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "cursor": { "type": "null" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"cursor":"null"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert!(args["cursor"].is_null());
}

#[test]
fn no_coerce_already_number() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "limit": { "type": "integer" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"limit":42}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], 42);
}

#[test]
fn no_coerce_unparseable() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "limit": { "type": "integer" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"limit":"notanumber"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], "notanumber");
}

#[test]
fn empty_arg_preserved() {
    // 2026-09-26: `""` does not parse as an integer, so it stays `""`.
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "limit": { "type": "integer" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"limit":""}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], "");
}

#[test]
fn coerce_all_multi_call() {
    let tools = vec![
        make_tool(
            "search",
            serde_json::json!({ "limit": { "type": "integer" } }),
        ),
        make_tool("toggle", serde_json::json!({ "on": { "type": "boolean" } })),
    ];
    let mut calls = vec![
        make_call("search", r#"{"limit":"5"}"#),
        make_call("toggle", r#"{"on":"true"}"#),
    ];
    coerce_all(&mut calls, &tools);
    let a0: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    let a1: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
    assert_eq!(a0["limit"], 5);
    assert_eq!(a1["on"], true);
}

#[test]
fn coerce_ignores_missing_tool() {
    let tools: Vec<ToolDefinition> = vec![];
    let mut calls = vec![make_call("unknown", r#"{"x":"1"}"#)];
    coerce_all(&mut calls, &tools);
    assert_eq!(calls[0].function.arguments, r#"{"x":"1"}"#);
}

#[test]
fn qwen3_xml_name() {
    assert_eq!(Qwen3XmlParser.name(), "qwen3_xml");
}

#[test]
fn qwen3_xml_wants_typed() {
    assert!(Qwen3XmlParser.wants_typed_arguments());
}

#[test]
fn qwen3_coder_wants_typed() {
    assert!(Qwen3CoderParser.wants_typed_arguments());
}

#[test]
fn qwen3_xml_has_grammar() {
    assert!(Qwen3XmlParser.has_tool_grammar());
}

#[test]
fn qwen3_xml_system_prompt_contains_markers() {
    let tools: Vec<ToolDefinition> = vec![];
    let tc = ToolChoice::Mode("auto".to_string());
    let prompt = Qwen3XmlParser.system_prompt(&tools, &tc, &crate::tool_parser::PromptLevers::OFF);
    assert!(prompt.contains("<tool_call>"), "missing <tool_call>");
    assert!(prompt.contains("<function="), "missing <function=");
    assert!(prompt.contains("<parameter="), "missing <parameter=");
}

#[test]
fn qwen3_xml_coerced_via_parse_and_coerce_all() {
    let raw = "<tool_call>\n\
        <function=search>\n\
        <parameter=query>\nrust async\n</parameter>\n\
        <parameter=limit>\n5\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_, mut calls) = parse_tool_calls(raw);
    assert_eq!(calls.len(), 1);

    let tools = vec![make_tool(
        "search",
        serde_json::json!({
            "query": { "type": "string" },
            "limit": { "type": "integer" }
        }),
    )];
    coerce_all(&mut calls, &tools);

    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], 5, "limit must be integer 5 after coercion");
    assert_eq!(args["query"], "rust async");
}

#[test]
fn coerce_boolean_false_lower() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "flag": { "type": "boolean" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"flag":"false"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["flag"], false);
}

#[test]
fn coerce_boolean_true_capitalized() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "flag": { "type": "boolean" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"flag":"True"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["flag"], true);
}

#[test]
fn coerce_invalid_arguments_json_is_noop() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "limit": { "type": "integer" } }),
    )];
    let raw = r#"{"limit":"10""#;
    let mut calls = vec![make_call("fn", raw)];
    coerce_all(&mut calls, &tools);
    assert_eq!(calls[0].function.arguments, raw);
}

#[test]
fn tool_call_format_from_str_qwen3_xml() {
    let fmt = "qwen3_xml".parse::<ToolCallFormat>().unwrap();
    assert_eq!(fmt.name(), "qwen3_xml");
}

fn exitplanmode_tool() -> ToolDefinition {
    // 2026-09-26: `allowedPrompts` is an array of objects that require `tool`
    // (enum `["Bash"]`) and `prompt`.
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "ExitPlanMode".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "plan": { "type": "string" },
                    "allowedPrompts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": { "type": "string", "enum": ["Bash"] },
                                "prompt": { "type": "string" }
                            },
                            "required": ["tool", "prompt"]
                        }
                    }
                },
                "required": ["plan"]
            })),
        },
    }
}

#[test]
fn repair_empty_key_in_nested_array_item() {
    let tools = vec![exitplanmode_tool()];
    let mut calls = vec![make_call(
        "ExitPlanMode",
        r#"{"plan":"build","allowedPrompts":[{"":"Bash","prompt":"run cargo"}]}"#,
    )];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    let item = &args["allowedPrompts"][0];
    assert_eq!(
        item["tool"], "Bash",
        "empty key should be repaired to `tool`"
    );
    assert_eq!(item["prompt"], "run cargo");
    assert!(item.get("").is_none(), "empty key must be removed");
}

#[test]
fn repair_empty_key_skips_when_value_not_in_enum() {
    // 2026-09-26: `"Python"` is not in `tool`'s enum, so the key is not
    // renamed.
    let tools = vec![exitplanmode_tool()];
    let mut calls = vec![make_call(
        "ExitPlanMode",
        r#"{"plan":"build","allowedPrompts":[{"":"Python","prompt":"x"}]}"#,
    )];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert!(
        args["allowedPrompts"][0].get("tool").is_none(),
        "must NOT repair when orphan value fails the enum"
    );
}

#[test]
fn repair_empty_key_skips_when_multiple_required_missing() {
    // 2026-09-26: Two required properties are missing, so the key is not
    // renamed.
    let tools = vec![exitplanmode_tool()];
    let mut calls = vec![make_call(
        "ExitPlanMode",
        r#"{"plan":"build","allowedPrompts":[{"":"Bash"}]}"#,
    )];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert!(
        args["allowedPrompts"][0].get("tool").is_none(),
        "must NOT repair when >1 required prop is missing (ambiguous)"
    );
}

#[test]
fn repair_empty_key_in_stringified_nested_array() {
    // 2026-09-26: `allowedPrompts` arrives as JSON text. The first repair pass
    // cannot descend into a string; the pass after coercion repairs the
    // parsed array.
    let tools = vec![exitplanmode_tool()];
    let stringified = serde_json::json!({
        "plan": "build",
        "allowedPrompts": r#"[{"":"Bash","prompt":"run cargo"}]"#
    })
    .to_string();
    let mut calls = vec![make_call("ExitPlanMode", &stringified)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert!(
        args["allowedPrompts"].is_array(),
        "stringified array must be coerced to a real array"
    );
    let item = &args["allowedPrompts"][0];
    assert_eq!(
        item["tool"], "Bash",
        "empty key repaired to `tool` post-coercion"
    );
    assert_eq!(item["prompt"], "run cargo");
    assert!(item.get("").is_none(), "empty key must be removed");
}

/// 2026-10-10: A non-string value of a `string` property becomes its JSON
/// text: numbers and bools verbatim, objects and arrays as compact JSON in
/// key order. A string, a `null`, and an undeclared key are left as they are.
#[test]
fn coerce_non_string_to_declared_string() {
    let tools = vec![make_tool(
        "lookup",
        serde_json::json!({
            "id": { "type": "string" },
            "ratio": { "type": "string" },
            "flag": { "type": "string" },
            "obj": { "type": "string" },
            "list": { "type": "string" },
            "name": { "type": "string" },
            "none": { "type": "string" },
        }),
    )];
    let args = r#"{"id":123123,"ratio":1.5,"flag":true,"obj":{"b":1,"a":[2]},"list":[1,"x"],"name":"kept","none":null,"extra":7}"#;
    let mut calls = vec![make_call("lookup", args)];
    coerce_all(&mut calls, &tools);
    let got: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(got["id"], serde_json::json!("123123"));
    assert_eq!(got["ratio"], serde_json::json!("1.5"));
    assert_eq!(got["flag"], serde_json::json!("true"));
    assert_eq!(got["obj"], serde_json::json!(r#"{"b":1,"a":[2]}"#));
    assert_eq!(got["list"], serde_json::json!(r#"[1,"x"]"#));
    assert_eq!(got["name"], serde_json::json!("kept"));
    assert!(got["none"].is_null(), "null stays null");
    assert_eq!(
        got["extra"],
        serde_json::json!(7),
        "undeclared key untouched"
    );
}

/// 2026-10-10: All-string arguments are not rewritten: `arguments` keeps its
/// original text byte for byte.
#[test]
fn coerce_string_property_already_string_is_untouched() {
    let tools = vec![make_tool(
        "lookup",
        serde_json::json!({ "id": { "type": "string" } }),
    )];
    let original = r#"{ "id" : "12345" }"#;
    let mut calls = vec![make_call("lookup", original)];
    coerce_all(&mut calls, &tools);
    assert_eq!(calls[0].function.arguments, original);
}
