// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Schema-driven repair of tool-call arguments: string values are
//! coerced to their declared types and empty keys are relabelled.
//!
//! Owner: server (tool parser).
//! Invariants:
//! - A call's arguments are left unchanged when they are not valid JSON, or
//!   when no tool matches its name, or the matched tool has no `properties`
//!   object.

use super::{ToolCall, ToolDefinition};

/// 2026-09-26: Coerce each call's arguments to its tool's schema.
///
/// The [`ToolDefinition`] is found by name. String values of top-level
/// properties typed `integer`, `number`, `boolean`, `array`, `object` or `null`
/// are parsed into that type, a non-string, non-null value of a property typed
/// `string` becomes its JSON text, and an empty `""` key is relabelled by
/// `repair_empty_keys`. A value that does not parse is left as it is.
pub fn coerce_all(calls: &mut [ToolCall], tools: &[ToolDefinition]) {
    for call in calls.iter_mut() {
        let def = tools.iter().find(|t| t.function.name == call.function.name);
        coerce_call_args(call, def);
    }
}

fn coerce_call_args(call: &mut ToolCall, tool_def: Option<&ToolDefinition>) {
    let Some(schema) = tool_def.and_then(|t| t.function.parameters.as_ref()) else {
        return;
    };
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
        return;
    };

    let Ok(mut args) = serde_json::from_str::<serde_json::Value>(&call.function.arguments) else {
        return;
    };

    let mut changed = repair_empty_keys(&mut args, schema);

    let Some(obj) = args.as_object_mut() else {
        if changed && let Ok(s) = serde_json::to_string(&args) {
            call.function.arguments = s;
        }
        return;
    };

    for (key, prop) in props {
        let Some(ty) = prop.get("type").and_then(|t| t.as_str()) else {
            continue;
        };
        let Some(val) = obj.get_mut(key) else {
            continue;
        };
        match ty {
            "integer" => {
                // 2026-09-26: `i64` first, so `"10"` becomes `10`, not `10.0`.
                // A string that parses only as `f64` becomes that float.
                if let serde_json::Value::String(s) = val {
                    if let Ok(n) = s.parse::<i64>() {
                        *val = serde_json::Value::Number(n.into());
                        changed = true;
                    } else if let Ok(f) = s.parse::<f64>()
                        && let Some(num) = serde_json::Number::from_f64(f)
                    {
                        *val = serde_json::Value::Number(num);
                        changed = true;
                    }
                }
            }
            "number" => {
                if let serde_json::Value::String(s) = val
                    && let Ok(n) = s.parse::<f64>()
                    && let Some(num) = serde_json::Number::from_f64(n)
                {
                    *val = serde_json::Value::Number(num);
                    changed = true;
                }
            }
            "boolean" => {
                if let serde_json::Value::String(s) = val {
                    match s.as_str() {
                        "true" | "True" => {
                            *val = serde_json::Value::Bool(true);
                            changed = true;
                        }
                        "false" | "False" => {
                            *val = serde_json::Value::Bool(false);
                            changed = true;
                        }
                        _ => {}
                    }
                }
            }
            "array" | "object" => {
                if let serde_json::Value::String(s) = val
                    && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s)
                {
                    *val = parsed;
                    changed = true;
                }
            }
            "string" => {
                // 2026-10-10: A wire format without types (poolside_v1 reads each
                // `<arg_value>` as JSON when it parses) turns a bare `123123` or
                // `true` into a number or bool. The schema says string, so the
                // client gets the value's JSON text: numbers and bools verbatim,
                // objects and arrays as compact JSON. `null` is left as it is.
                if !val.is_string()
                    && !val.is_null()
                    && let Ok(text) = serde_json::to_string(val)
                {
                    *val = serde_json::Value::String(text);
                    changed = true;
                }
            }
            "null" => {
                if matches!(val, serde_json::Value::String(s) if s == "null") {
                    *val = serde_json::Value::Null;
                    changed = true;
                }
            }
            _ => {}
        }
    }

    // 2026-09-26: Repair again: an array or object that arrived as a JSON
    // string is parsed only by the loop above, so the first pass could not
    // descend into it.
    changed |= repair_empty_keys(&mut args, schema);

    if changed && let Ok(s) = serde_json::to_string(&args) {
        call.function.arguments = s;
    }
}

/// 2026-09-26: Rename an empty-string key (`""`) to the one `required`
/// property the object lacks, descending into array `items` and nested
/// `properties`. Returns `true` if any key was renamed.
///
/// A key is renamed only when exactly one required property is missing and
/// the value passes `value_matches_schema` for it; otherwise the object is
/// left as it is.
fn repair_empty_keys(val: &mut serde_json::Value, schema: &serde_json::Value) -> bool {
    let mut changed = false;
    match val {
        serde_json::Value::Object(map) => {
            if map.contains_key("") {
                let props = schema.get("properties").and_then(|p| p.as_object());
                let missing: Vec<String> = schema
                    .get("required")
                    .and_then(|r| r.as_array())
                    .map(|req| {
                        req.iter()
                            .filter_map(|r| r.as_str())
                            .filter(|r| !map.contains_key(*r))
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                if missing.len() == 1 {
                    let cand = &missing[0];
                    let prop_schema = props.and_then(|p| p.get(cand));
                    let orphan = map.get("").cloned().unwrap_or(serde_json::Value::Null);
                    if value_matches_schema(&orphan, prop_schema) {
                        map.remove("");
                        map.insert(cand.clone(), orphan);
                        changed = true;
                    }
                }
            }
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (k, v) in map.iter_mut() {
                    if let Some(child) = props.get(k) {
                        changed |= repair_empty_keys(v, child);
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            if let Some(items) = schema.get("items") {
                for elem in arr.iter_mut() {
                    changed |= repair_empty_keys(elem, items);
                }
            }
        }
        _ => {}
    }
    changed
}

/// 2026-09-26: True when `val` is in `prop_schema`'s `enum` (if it has one)
/// and of its `type`, where `integer`, `number` and `boolean` also accept a
/// string. With no schema, or an unknown type, any value passes.
fn value_matches_schema(val: &serde_json::Value, prop_schema: Option<&serde_json::Value>) -> bool {
    let Some(ps) = prop_schema else {
        return true;
    };
    if let Some(en) = ps.get("enum").and_then(|e| e.as_array())
        && !en.iter().any(|e| e == val)
    {
        return false;
    }
    if let Some(ty) = ps.get("type").and_then(|t| t.as_str()) {
        let ok = match ty {
            "string" => val.is_string(),
            "integer" | "number" => val.is_number() || val.is_string(),
            "boolean" => val.is_boolean() || val.is_string(),
            "array" => val.is_array(),
            "object" => val.is_object(),
            "null" => val.is_null(),
            _ => true,
        };
        if !ok {
            return false;
        }
    }
    true
}
