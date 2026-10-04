// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `backfill_required_params`, the schema-driven argument repair, and
//! the delegation-tool helpers it uses to fill an empty `subagent_type`.
//!
//! Owner: server (tool parser).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Agent-type names from a delegation tool's prose description:
/// lines shaped `- <name>: …`, where `<name>`, the text before the first `:`,
/// is 1 to 64 ASCII alphanumerics, `-` or `_`. That excludes prose bullets
/// such as `- If you want to read a file, use Read instead`.
fn parse_agent_types(description: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in description.lines() {
        let Some(rest) = line.trim_start().strip_prefix("- ") else {
            continue;
        };
        let Some((name, _)) = rest.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if !name.is_empty()
            && name.len() <= 64
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            out.push(name.to_string());
        }
    }
    out
}

/// 2026-09-26: A `subagent_type` for a delegation tool whose agent names are
/// listed only in its description (`parse_agent_types`): the first listed
/// agent whose name contains `general` (any case), else the first listed,
/// else `"general"`. `backfill_required_params` uses it for an empty value.
fn infer_default_subagent_type(description: Option<&str>) -> String {
    let candidates = description.map(parse_agent_types).unwrap_or_default();
    candidates
        .iter()
        .find(|c| c.to_ascii_lowercase().contains("general"))
        .or_else(|| candidates.first())
        .cloned()
        .unwrap_or_else(|| "general".to_string())
}

/// 2026-09-26: Repair each call's arguments against its tool's schema, in
/// order: coerce string values to the declared type, rename keys that match a
/// property up to case and underscores, re-split keys leaked into values,
/// and fill a present-but-blank required `description` or `subagent_type`
/// (an absent parameter is never inserted). A call with no matching tool,
/// no `parameters`, or arguments that are not a JSON object is left alone.
pub fn backfill_required_params(calls: &mut [ToolCall], tools: &[ToolDefinition]) {
    for call in calls.iter_mut() {
        let Some(tool_def) = tools.iter().find(|t| t.function.name == call.function.name) else {
            continue;
        };
        let Some(ref params_schema) = tool_def.function.parameters else {
            continue;
        };
        let required = params_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();
        let properties = params_schema.get("properties").and_then(|p| p.as_object());
        let Ok(mut args) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            &call.function.arguments,
        ) else {
            continue;
        };
        let mut changed = false;

        // 2026-09-26: A boolean also accepts `1`/`0`/`yes`/`no`, in any case.
        if let Some(props) = properties {
            for (key, value) in args.iter_mut() {
                let expected_type = props.get(key).and_then(|p| resolve_schema_type(p));
                if let (Some(expected), serde_json::Value::String(s)) = (expected_type, &value) {
                    let coerced = match expected {
                        "number" => s.parse::<f64>().ok().map(|n| {
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(n)
                                    .unwrap_or(serde_json::Number::from(0)),
                            )
                        }),
                        "integer" => s
                            .parse::<i64>()
                            .ok()
                            .map(|n| serde_json::Value::Number(n.into())),
                        "boolean" => match s.to_lowercase().as_str() {
                            "true" | "1" | "yes" => Some(serde_json::Value::Bool(true)),
                            "false" | "0" | "no" => Some(serde_json::Value::Bool(false)),
                            _ => None,
                        },
                        "object" | "array" => serde_json::from_str(s).ok(),
                        _ => None,
                    };
                    if let Some(new_val) = coerced {
                        *value = new_val;
                        changed = true;
                    }
                }
            }
        }

        // 2026-09-26: When the property's own name is also present, the
        // misnamed key's value is dropped.
        if properties.is_some() {
            let keys_to_fix: Vec<(String, String)> = args
                .keys()
                .map(|k| {
                    (
                        k.clone(),
                        normalize_param_name(tools, &call.function.name, k),
                    )
                })
                .filter(|(orig, mapped)| orig != mapped)
                .collect();

            for (wrong_key, right_key) in keys_to_fix {
                if let Some(val) = args.remove(&wrong_key) {
                    args.entry(right_key).or_insert(val);
                    changed = true;
                }
            }
        }

        // 2026-09-26: Runs before the backfill below, so a recovered key is
        // not also inserted as `""`. The real key takes the value only when it
        // is absent or a blank string.
        if properties.is_some() {
            let salvageable: Vec<(String, String, String)> = args
                .iter()
                .filter_map(|(k, v)| {
                    let s = v.as_str()?;
                    salvage_echoed_param(tools, &call.function.name, k, s)
                        .map(|(real_key, real_val)| (k.clone(), real_key, real_val))
                })
                .collect();
            for (echoed_key, real_key, real_val) in salvageable {
                let target_empty = matches!(
                    args.get(&real_key),
                    None | Some(serde_json::Value::String(_))
                ) && args
                    .get(&real_key)
                    .and_then(|v| v.as_str())
                    .is_none_or(|s| s.trim().is_empty());
                if target_empty {
                    args.remove(&echoed_key);
                    args.insert(real_key, serde_json::Value::String(real_val));
                    changed = true;
                }
            }
        }

        // 2026-10-03: An absent required parameter stays absent. It used to be
        // inserted as `""`, which passed the presence check and made "omitted"
        // and "supplied empty" indistinguishable. Now `assess_tool_call` reports
        // `MissingParam`, a soft issue: the call is delivered unmodified with the
        // diagnostic. An explicit `""` is untouched.

        // 2026-10-03: The one absent-required fill that stays: a delegation tool's
        // `subagent_type`. It is a named client-compat default (the value is inferred
        // from the tool's own agent list below), not a generic empty-string fabrication,
        // and tests/group_f pins it. Every other absent required key stays absent.
        for key in ["subagent_type", "subagentType"] {
            if required.contains(&key) && !args.contains_key(key) {
                args.insert(key.to_string(), serde_json::Value::String(String::new()));
                changed = true;
            }
        }

        let func_name = call.function.name.clone();
        for key in &required {
            if let Some(serde_json::Value::String(val)) = args.get(*key) {
                if !val.trim().is_empty() {
                    continue;
                }
                let auto_val = match *key {
                    "description" => {
                        if let Some(serde_json::Value::String(cmd)) = args.get("command") {
                            if cmd.len() > 50 {
                                // 2026-09-26: 47 chars, not bytes: a byte slice
                                // could split a multibyte char and panic.
                                let head: String = cmd.chars().take(47).collect();
                                format!("Run: {head}...")
                            } else {
                                format!("Run: {cmd}")
                            }
                        } else {
                            format!("{func_name} operation")
                        }
                    }
                    "filePath" | "file_path" => {
                        continue;
                    }
                    "oldString" | "old_string" => {
                        continue;
                    }
                    "subagent_type" | "subagentType" => {
                        infer_default_subagent_type(tool_def.function.description.as_deref())
                    }
                    _ => continue,
                };
                args.insert(key.to_string(), serde_json::Value::String(auto_val));
                changed = true;
            }
        }

        if changed && let Ok(new_args) = serde_json::to_string(&serde_json::Value::Object(args)) {
            call.function.arguments = new_args;
        }
    }
}
