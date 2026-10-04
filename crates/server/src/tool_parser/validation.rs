// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Post-parse repair and validation of tool calls against the
//! request's tool schemas: argument backfill, path clean-up, and
//! `assess_tool_call`, which classifies what is wrong with a call.
//!
//! Owner: server (tool parser).
//! Invariants:
//! - `validate_tool_calls` puts every call in `valid`, or its message in
//!   `errors`; a call missing a required parameter is in both.

use super::fuzzy_match::fuzzy_match_tool_name;
use super::*;

mod backfill;
mod paths;

pub use backfill::backfill_required_params;
pub use paths::normalize_paths;

/// 2026-09-26: A property's type: its `type`, else the first non-null `type`
/// among its `anyOf` or `oneOf` variants, as in
/// `{"anyOf": [{"type":"integer"},{"type":"null"}]}`.
fn resolve_schema_type(schema: &serde_json::Value) -> Option<&str> {
    if let Some(t) = schema.get("type").and_then(|t| t.as_str()) {
        return Some(t);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = schema.get(key).and_then(|v| v.as_array()) {
            for variant in variants {
                if let Some(t) = variant.get("type").and_then(|t| t.as_str())
                    && t != "null"
                {
                    return Some(t);
                }
            }
        }
    }
    None
}

/// 2026-09-26: Map a model-emitted parameter `key` to the schema property of
/// tool `call_name` that it matches up to case and underscores (`filepath` →
/// `file_path`). Used by `backfill_required_params` and by live argument
/// streaming (`coerce_kv` in `streaming_emit.rs`). Returns `key` unchanged when
/// the tool is unknown, has no `properties`, `key` is already a property, or
/// nothing matches.
pub(crate) fn normalize_param_name(tools: &[ToolDefinition], call_name: &str, key: &str) -> String {
    let Some(tool_def) = tools.iter().find(|t| t.function.name == call_name) else {
        return key.to_string();
    };
    let Some(props) = tool_def
        .function
        .parameters
        .as_ref()
        .and_then(|p| p.get("properties"))
        .and_then(|p| p.as_object())
    else {
        return key.to_string();
    };
    if props.contains_key(key) {
        return key.to_string();
    }
    let schema_normalized: std::collections::HashMap<String, &str> = props
        .keys()
        .map(|k| (k.to_lowercase().replace('_', ""), k.as_str()))
        .collect();
    let norm = key.to_lowercase().replace('_', "");
    schema_normalized
        .get(&norm)
        .map(|schema_key| schema_key.to_string())
        .unwrap_or_else(|| key.to_string())
}

/// 2026-09-26: Recover a parameter whose real key leaked into its value, as in
/// `<parameter=parameter>filePath>\n/tmp/x</parameter>`: when `key` is not a
/// schema property of `call_name` and `value` starts with `PROP>` for a
/// property `PROP` (longest name tried first), return `(PROP, rest)` with
/// `rest` trimmed. Never fires when `key` is itself a property, since a real
/// value may start with `IDENT>`.
///
/// Used by `backfill_required_params` and by live argument streaming
/// (`coerce_kv` in `streaming_emit.rs`).
pub(crate) fn salvage_echoed_param(
    tools: &[ToolDefinition],
    call_name: &str,
    key: &str,
    value: &str,
) -> Option<(String, String)> {
    let props = tools
        .iter()
        .find(|t| t.function.name == call_name)?
        .function
        .parameters
        .as_ref()?
        .get("properties")?
        .as_object()?;
    if props.is_empty() || props.contains_key(key) {
        return None;
    }
    let mut names: Vec<&String> = props.keys().collect();
    names.sort_by_key(|n| std::cmp::Reverse(n.len()));
    for prop in names {
        if let Some(rest) = value
            .strip_prefix(prop.as_str())
            .and_then(|r| r.strip_prefix('>'))
        {
            // 2026-09-26: Trimmed, as both XML parameter parsers trim values.
            return Some((prop.clone(), rest.trim().to_string()));
        }
    }
    None
}

/// 2026-09-26: Names of the call's required parameters that are absent, `null`
/// or a blank string; empty when there are none or the tool or its schema is
/// unknown. Arguments that do not parse count as an empty object.
pub fn find_empty_required_params(call: &ToolCall, tools: &[ToolDefinition]) -> Vec<String> {
    let Some(tool_def) = tools.iter().find(|t| t.function.name == call.function.name) else {
        return Vec::new();
    };
    let Some(ref params_schema) = tool_def.function.parameters else {
        return Vec::new();
    };
    let required = params_schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let args: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&call.function.arguments).unwrap_or_default();
    let mut empty = Vec::new();
    for key in &required {
        match args.get(key.as_str()) {
            None => empty.push(key.clone()),
            Some(serde_json::Value::String(s)) if s.trim().is_empty() => empty.push(key.clone()),
            Some(serde_json::Value::Null) => empty.push(key.clone()),
            _ => {}
        }
    }
    empty
}

/// 2026-09-26: Result of validating a batch of tool calls against their schemas.
pub struct ValidatedToolCalls {
    /// 2026-09-26: Calls to deliver: those that passed, and those missing only
    /// a required parameter.
    pub valid: Vec<ToolCall>,
    /// 2026-09-26: One message per call that failed; the blocking chat path
    /// logs them.
    pub errors: Vec<String>,
}

/// 2026-09-26: Validate tool calls with [`assess_tool_call`], after replacing
/// an unknown tool name with its closest match (`fuzzy_match_tool_name`), when
/// there is one.
pub fn validate_tool_calls(
    mut calls: Vec<ToolCall>,
    tools: &[ToolDefinition],
) -> ValidatedToolCalls {
    let mut valid = Vec::new();
    let mut errors = Vec::new();

    for call in &mut calls {
        if tools.iter().all(|t| t.function.name != call.function.name)
            && let Some(best) = fuzzy_match_tool_name(&call.function.name, tools)
        {
            tracing::info!(
                "Fuzzy tool name repair: '{}' -> '{}'",
                call.function.name,
                best
            );
            call.function.name = best;
        }
        match assess_tool_call(call, tools) {
            Ok(()) => valid.push(call.clone()),
            Err(ToolCallIssue::MissingParam(msg)) => {
                valid.push(call.clone());
                errors.push(msg);
            }
            Err(issue) => errors.push(issue.into_message()),
        }
    }

    ValidatedToolCalls { valid, errors }
}

/// 2026-09-26: What is wrong with a tool call. The variant decides whether
/// the call is still delivered.
#[derive(Debug)]
pub enum ToolCallIssue {
    /// 2026-09-26: A required parameter is absent. Blocking and streaming both
    /// deliver the call.
    MissingParam(String),
    /// 2026-09-26: A write-family tool's path is blank, or a shell tool's
    /// command is under 2 chars after trimming. Streaming delivers the call;
    /// blocking drops it.
    EmptyRequired(String),
    /// 2026-09-26: Unknown tool, arguments that are not a JSON object, or a
    /// path that is too short, holds whitespace or a shell metacharacter, or
    /// looks like a directory. Neither path delivers the arguments.
    Hard(String),
}

impl ToolCallIssue {
    pub fn message(&self) -> &str {
        match self {
            Self::MissingParam(m) | Self::EmptyRequired(m) | Self::Hard(m) => m,
        }
    }
    pub fn into_message(self) -> String {
        match self {
            Self::MissingParam(m) | Self::EmptyRequired(m) | Self::Hard(m) => m,
        }
    }
}

/// 2026-09-26: [`assess_tool_call`] with the issue reduced to its message.
pub fn validate_single_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<(), String> {
    assess_tool_call(call, tools).map_err(ToolCallIssue::into_message)
}

/// 2026-09-26: Validate one tool call. Returns `Ok(())`, or the first
/// [`ToolCallIssue`] found; the caller decides delivery from its variant.
pub fn assess_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<(), ToolCallIssue> {
    let name = &call.function.name;

    let tool_def = tools.iter().find(|t| t.function.name == *name);
    if tool_def.is_none() {
        let available: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        return Err(ToolCallIssue::Hard(format!(
            "Error: Unknown tool '{}'. Available tools: {}",
            name,
            available.join(", ")
        )));
    }
    let tool_def = tool_def.unwrap();

    let args: serde_json::Map<String, serde_json::Value> =
        match serde_json::from_str(&call.function.arguments) {
            Ok(a) => a,
            Err(_) => {
                // 2026-09-26: 100 chars, not bytes: a byte slice could split a
                // multibyte char and panic.
                let preview: String = call.function.arguments.chars().take(100).collect();
                return Err(ToolCallIssue::Hard(format!(
                    "Error: {name} arguments must be valid JSON. Got: {preview}"
                )));
            }
        };

    // 2026-09-26: Only presence is checked; an empty string passes here. The
    // write-family path and shell command checks below are the exceptions.
    if let Some(ref params_schema) = tool_def.function.parameters {
        let required: Vec<&str> = params_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        for key in &required {
            if args.get(*key).is_none() {
                // 2026-10-03: Backfill no longer turns an absent required string into
                // `""`, which `EmptyRequired` below used to catch. Keep that safety: an
                // absent write-family path or shell command is still refused (hard).
                let write_path = matches!(
                    name.as_str(),
                    "Write" | "write" | "Edit" | "edit" | "MultiEdit" | "multiEdit"
                        | "multi_edit" | "write_file" | "writeFile"
                ) && matches!(*key, "file_path" | "filePath" | "path");
                let shell_cmd = matches!(
                    name.as_str(),
                    "bash" | "Bash" | "shell" | "Shell" | "exec" | "Exec" | "run" | "Run"
                        | "execute" | "Execute" | "terminal" | "Terminal"
                ) && matches!(*key, "command" | "cmd" | "script" | "code");
                if write_path || shell_cmd {
                    return Err(ToolCallIssue::EmptyRequired(format!(
                        "Error: {name} requires a non-empty '{key}' but it was not provided."
                    )));
                }
                return Err(ToolCallIssue::MissingParam(format!(
                    "Error: {} requires parameter '{}' but it was not provided.",
                    name, key
                )));
            }
        }
    }

    const FILE_TOOLS: &[&str] = &["Write", "write", "Edit", "edit", "Read", "read"];
    const PATH_KEYS: &[&str] = &["file_path", "filePath", "path"];
    // 2026-09-26: An empty path is refused only for `WRITE_FAMILY`; the
    // `FILE_TOOLS` check further down lets it through.
    const WRITE_FAMILY: &[&str] = &[
        "Write",
        "write",
        "Edit",
        "edit",
        "MultiEdit",
        "multiEdit",
        "multi_edit",
        "write_file",
        "writeFile",
    ];
    if WRITE_FAMILY.contains(&name.as_str()) {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                let trimmed = path.trim();
                if trimmed.is_empty() {
                    // 2026-09-26: With `METRALE_TOOLCALL_DEBUG=1`, log each
                    // argument's key with its string length or value.
                    if std::env::var("METRALE_TOOLCALL_DEBUG").as_deref() == Ok("1") {
                        let shape: Vec<String> = args
                            .iter()
                            .map(|(k, v)| match v {
                                serde_json::Value::String(s) => {
                                    format!("{k}=str(len={})", s.len())
                                }
                                other => format!("{k}={}", other),
                            })
                            .collect();
                        tracing::warn!(
                            tool = %name, empty_key = %key,
                            "METRALE_TOOLCALL_DEBUG empty-path arg shape: [{}]",
                            shape.join(", ")
                        );
                    }
                    return Err(ToolCallIssue::EmptyRequired(format!(
                        "Error: {name} requires a non-empty '{key}'. \
                             Got empty string — provide an absolute path \
                             like '/tmp/calc-test75/Cargo.toml'."
                    )));
                }
                // 2026-09-26: Relative paths such as `Cargo.toml` pass. A path
                // under 3 chars, or one holding whitespace or a `SHELL_META`
                // char (a leaked command such as `created && ls -R`), is refused.
                const SHELL_META: &[char] = &[
                    ' ', '\t', '\n', '\r', '&', '|', ';', '`', '$', '<', '>', '(', ')', '*', '?',
                ];
                let looks_like_command = trimmed.contains(SHELL_META);
                if looks_like_command || trimmed.len() < 3 {
                    return Err(ToolCallIssue::Hard(format!(
                        "Error: {name} '{key}' must be a filesystem path (absolute or relative \
                         to the working directory), at least 3 chars, with no shell \
                         metacharacters or whitespace. Got {path:?}."
                    )));
                }
            }
        }
    }
    // 2026-09-26: A shell tool's command must be at least 2 chars after
    // trimming.
    const SHELL_FAMILY: &[&str] = &[
        "bash", "Bash", "shell", "Shell", "exec", "Exec", "run", "Run", "execute", "Execute",
        "terminal", "Terminal",
    ];
    const CMD_KEYS: &[&str] = &["command", "cmd", "script", "code"];
    if SHELL_FAMILY.contains(&name.as_str()) {
        for key in CMD_KEYS {
            if let Some(serde_json::Value::String(cmd)) = args.get(*key)
                && (cmd.trim().is_empty() || cmd.trim().len() < 2)
            {
                return Err(ToolCallIssue::EmptyRequired(format!(
                    "Error: {name} requires a non-empty '{key}'. \
                         Got empty string — provide the shell command \
                         to execute, e.g. 'ls /tmp'."
                )));
            }
        }
    }
    if FILE_TOOLS.contains(&name.as_str()) {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                if path.ends_with('/') {
                    return Err(ToolCallIssue::Hard(format!(
                        "Error: {} file_path must be a FILE, not a directory. Got '{}'. Use e.g. '{}/index.ts'",
                        name,
                        path,
                        path.trim_end_matches('/')
                    )));
                }
                // 2026-09-26: A bare name of only lowercase letters, `-` and `_`
                // (`src`, `my-dir`) is taken for a directory; `Makefile` or
                // `LICENSE` pass because of their capitals.
                if !path.is_empty()
                    && !path.contains('.')
                    && !path.contains('/')
                    && path
                        .chars()
                        .all(|c| c.is_lowercase() || c == '-' || c == '_')
                {
                    return Err(ToolCallIssue::Hard(format!(
                        "Error: {} file_path '{}' looks like a directory. Add a filename, e.g. '{}/index.ts'",
                        name, path, path
                    )));
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod path_sanitizer_tests {
    use crate::tool_parser::{FunctionCall, ToolCall};

    fn call(name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "x".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.to_string(),
            },
        }
    }

    #[test]
    fn malformed_quoted_comma_filepath_sanitized() {
        // 2026-09-26: The quotes and trailing comma are stripped, then the
        // path is made relative to `cwd`.
        let mut calls = vec![call(
            "write",
            serde_json::json!({
                "filePath": "\"/tmp/proj/Cargo.toml\",",
                "content": "[package]\nname = \"x\"\n"
            }),
        )];
        super::normalize_paths(&mut calls, "/tmp/proj");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["filePath"], "Cargo.toml");
    }

    #[test]
    fn punctuation_drifted_workdir_is_repaired_to_cwd() {
        let cwd = "/tmp/harness-laguna_s21-carddefaults-r1";
        let mut calls = vec![call(
            "bash",
            serde_json::json!({
                "command": "cargo test",
                "workdir": "/tmp/harness-laguna-s21-carddefaults-r1"
            }),
        )];

        super::normalize_paths(&mut calls, cwd);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["workdir"], cwd);
    }

    #[test]
    fn punctuation_drifted_cwd_inside_bash_command_is_repaired() {
        let cwd = "/tmp/harness-laguna_s21-agentfix-r1";
        let mut calls = vec![call(
            "bash",
            serde_json::json!({
                "command": "ls /tmp/harness-laguna-s21-agentfix-r1/opencode && pwd",
                "workdir": cwd
            }),
        )];

        super::normalize_paths(&mut calls, cwd);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args["command"],
            "ls /tmp/harness-laguna_s21-agentfix-r1/opencode && pwd"
        );
    }
}
