// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: E1/L01 characterization fixtures (race-toolfix). Pin what the GLM-5.3
//! route (`poolside_v1` -> `backfill_required_params` -> `assess_tool_call`) does with
//! omitted vs supplied-empty required arguments. Assertions state CURRENT behaviour
//! ("before"); (updated 2026-10-03 after the fix: now pins the fixed behaviour). Run with `--nocapture` for bytes.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

fn tool(name: &str, desc: Option<&str>, params: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: desc.map(String::from),
            parameters: Some(params),
        },
    }
}

fn city_tool() -> ToolDefinition {
    tool(
        "get_weather",
        None,
        serde_json::json!({"type":"object","properties":{
            "city":{"type":"string","minLength":2},
            "days":{"type":"integer"},
            "unit":{"type":"string","enum":["c","f"]}},
          "required":["city"]}),
    )
}

fn call(name: &str, args: &str) -> ToolCall {
    ToolCall {
        id: "c1".into(),
        call_type: "function".into(),
        function: FunctionCall { name: name.into(), arguments: args.into() },
    }
}

/// Backfill then assess, as blocking and streaming both do. Returns (bytes, class).
fn run(tools: &[ToolDefinition], name: &str, args: &str) -> (String, String) {
    let mut cs = vec![call(name, args)];
    backfill_required_params(&mut cs, tools);
    let class = match assess_tool_call(&cs[0], tools) {
        Ok(()) => "Ok".to_string(),
        Err(i) => format!("{i:?}"),
    };
    let out = cs[0].function.arguments.clone();
    println!("E1 {name} in={args} out={out} assess={class} empties={:?}",
        find_empty_required_params(&cs[0], tools));
    (out, class)
}

#[test]
fn required_string_omitted_stays_omitted_with_diagnostic() {
    let (out, class) = run(&[city_tool()], "get_weather", "{}");
    // Old failure (fixed 2026-10-03): out was `{"city":""}` and class Ok.
    assert_eq!(out, "{}");
    assert!(class.starts_with("MissingParam"), "{class}");
}

#[test]
fn omitted_and_explicit_empty_are_distinguishable_after_backfill() {
    let (a, ca) = run(&[city_tool()], "get_weather", "{}");
    let (b, cb) = run(&[city_tool()], "get_weather", r#"{"city":""}"#);
    assert_eq!(a, "{}");
    assert_eq!(b, r#"{"city":""}"#); // explicit empty untouched
    assert!(ca.starts_with("MissingParam"), "{ca}");
    assert_eq!(cb, "Ok");
}

#[test]
fn valid_string_untouched() {
    let (out, class) = run(&[city_tool()], "get_weather", r#"{"city":"Paris"}"#);
    assert_eq!(out, r#"{"city":"Paris"}"#);
    assert_eq!(class, "Ok");
}

#[test]
fn missing_required_integer_is_not_fabricated() {
    let t = tool("t", None, serde_json::json!({"type":"object",
        "properties":{"n":{"type":"integer"}},"required":["n"]}));
    let (out, class) = run(&[t], "t", "{}");
    assert_eq!(out, "{}");
    assert!(class.starts_with("MissingParam"), "{class}");
}

#[test]
fn enum_and_minlength_are_not_enforced_server_side() {
    // Omitted required enum string: filled with "" which violates the enum; passes.
    let t = tool("t", None, serde_json::json!({"type":"object",
        "properties":{"unit":{"type":"string","enum":["c","f"]}},"required":["unit"]}));
    let (out, class) = run(&[t], "t", "{}");
    assert_eq!(out, "{}");
    assert!(class.starts_with("MissingParam"), "{class}");
    // Supplied value violating enum / minLength passes untouched.
    let (o2, c2) = run(&[city_tool()], "get_weather", r#"{"city":"P","unit":"kelvin"}"#);
    assert_eq!(o2, r#"{"city":"P","unit":"kelvin"}"#);
    assert_eq!(c2, "Ok");
}

#[test]
fn required_without_declared_type_stays_omitted() {
    let t = tool("t", None, serde_json::json!({"type":"object",
        "properties":{"x":{}},"required":["x"]}));
    let (out, _) = run(&[t], "t", "{}");
    assert_eq!(out, "{}");
}

#[test]
fn subagent_type_inference_kept_for_absent_and_blank() {
    let params = serde_json::json!({"type":"object",
        "properties":{"subagent_type":{"type":"string"},"prompt":{"type":"string"}},
        "required":["subagent_type","prompt"]});
    let with_bullets = tool("task", Some("Agents:\n- explore: reads code\n- general-purpose: anything"), params.clone());
    // Absent subagent_type is still inferred (kept on purpose, tests/group_f); so is blank.
    let (o0, _) = run(&[with_bullets.clone()], "task", r#"{"prompt":"x"}"#);
    assert_eq!(o0, r#"{"prompt":"x","subagent_type":"general-purpose"}"#);
    let (o1, _) = run(&[with_bullets], "task", r#"{"prompt":"x","subagent_type":""}"#);
    assert_eq!(o1, r#"{"prompt":"x","subagent_type":"general-purpose"}"#);
    let no_bullets = tool("task", Some("Delegates work."), params.clone());
    let (o2, _) = run(&[no_bullets], "task", r#"{"prompt":"x","subagent_type":" "}"#);
    assert_eq!(o2, r#"{"prompt":"x","subagent_type":"general"}"#);
    let irrelevant = tool("task", Some("- Note: be nice"), params);
    let (o3, _) = run(&[irrelevant], "task", r#"{"prompt":"x","subagent_type":""}"#);
    assert_eq!(o3, r#"{"prompt":"x","subagent_type":"Note"}"#); // prose bullet taken as agent
}

/// Full GLM wire route: poolside_v1 text with the required arg omitted.
#[test]
fn glm_wire_omitted_required_reaches_client_omitted_with_diagnostic() {
    let (_, mut calls) = parse_tool_calls_promoting_bare_names(
        "<tool_call>get_weather<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>");
    assert_eq!(calls.len(), 1);
    let tools = [city_tool()];
    backfill_required_params(&mut calls, &tools);
    coerce_all(&mut calls, &tools);
    let v = validate_tool_calls(calls, &tools);
    println!("E1 glm-wire out={} errors={:?}", v.valid[0].function.arguments, v.errors);
    assert_eq!(v.valid[0].function.arguments, r#"{"days":3}"#);
    assert_eq!(v.errors.len(), 1, "diagnostic carried: {:?}", v.errors);
}
