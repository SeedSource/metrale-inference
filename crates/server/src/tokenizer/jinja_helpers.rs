// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Chat-template loading for `ChatTokenizer` (override, OpenAI variant, the
//! model's own template, ChatML default) and the minijinja environment they render in.
//!
//! Owner: server (tokenizer).
//! Invariants: none beyond the types.

use anyhow::{Context, Result};
use std::path::Path;

pub(super) const TEMPLATE_OVERRIDE_DIR: &str = "jinja-templates";

/// 2026-09-26: How the template's `tojson` filter serializes values. `build_jinja_env`
/// picks it from `METRALE_USE_HF_REF_JSON_DUMPS`; tests pass it to `build_jinja_env_with`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolJsonStyle {
    /// 2026-09-26: minijinja's builtin `tojson`, compact: `{"a":1}`. Used unless the env
    /// var is `1`.
    Compact,
    /// 2026-09-26: `python_tojson`, spaced: `{"a": 1}`.
    HfSpaced,
}

/// 2026-09-26: Production entry point: `HfSpaced` when `METRALE_USE_HF_REF_JSON_DUMPS` is
/// `1`, else `Compact`, then [`build_jinja_env_with`]. Reads the env var on every call.
pub(super) fn build_jinja_env(chat_template: &str) -> Result<minijinja::Environment<'static>> {
    build_jinja_env_for(chat_template, false)
}

/// 2026-10-15: [`build_jinja_env`] for a model whose MODEL.toml sets
/// `[behavior].hf_tool_json` (`hf_default`, GLM-5.3): `HfSpaced` whatever the env var says.
pub(super) fn build_jinja_env_for(
    chat_template: &str,
    hf_default: bool,
) -> Result<minijinja::Environment<'static>> {
    build_jinja_env_with(chat_template, tool_json_style(hf_default))
}

/// 2026-10-15: `HfSpaced` when the model asks for it or `METRALE_USE_HF_REF_JSON_DUMPS` is
/// `1`, else `Compact`.
pub(super) fn tool_json_style(hf_default: bool) -> ToolJsonStyle {
    if hf_default || std::env::var("METRALE_USE_HF_REF_JSON_DUMPS").as_deref() == Ok("1") {
        ToolJsonStyle::HfSpaced
    } else {
        ToolJsonStyle::Compact
    }
}

/// 2026-09-26: Build a chat-template environment with an explicit tool-JSON style.
///
/// Tests call this instead of setting `METRALE_USE_HF_REF_JSON_DUMPS`: the test harness
/// runs tests as threads of one process, so a `set_var` would change what concurrently
/// running tests render.
///
/// Each call leaks one copy of `chat_template` to get the `'static` lifetime the
/// environment needs.
pub(crate) fn build_jinja_env_with(
    chat_template: &str,
    tool_json_style: ToolJsonStyle,
) -> Result<minijinja::Environment<'static>> {
    let template_static: &'static str = Box::leak(chat_template.to_string().into_boxed_str());
    let mut env = minijinja::Environment::new();
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);

    env.add_function(
        "raise_exception",
        |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_filter("rtrim", |s: String| -> String {
        s.trim_end_matches('\n').to_string()
    });
    env.add_filter("ltrim", |s: String| -> String {
        s.trim_start_matches('\n').to_string()
    });
    env.add_filter("split_first", |s: String, sep: String| -> String {
        s.split(&sep).next().unwrap_or("").to_string()
    });
    env.add_filter("split_last", |s: String, sep: String| -> String {
        s.rsplit(&sep).next().unwrap_or("").to_string()
    });

    // 2026-09-26: Python-style methods minijinja does not have. The callback runs only for
    // a method minijinja does not know: on strings `split`, `find`, `rfind`, `replace`,
    // `strip`, `rstrip`; on maps `items`, `keys`, `values`, `get`; anything else stays an
    // `UnknownMethod` error. Override templates use some of them: `_args.items()` in
    // jinja-templates/openai/minimax_m2.jinja, `.split(...)` in the `strip_thinking` macro
    // of jinja-templates/gemma4.jinja.
    env.set_unknown_method_callback(
        |state, value, method, args| -> Result<minijinja::Value, minijinja::Error> {
            use minijinja::value::{ValueKind, from_args};
            if value.kind() == ValueKind::String {
                match method {
                    "split" => {
                        // 2026-09-26: With no separator the `split` filter splits on
                        // whitespace (`split_whitespace` in minijinja 2.21).
                        let (sep,): (Option<minijinja::Value>,) = from_args(args)?;
                        let mut filter_args = vec![value.clone()];
                        if let Some(sep) = sep {
                            filter_args.push(sep);
                        }
                        return state.apply_filter("split", &filter_args);
                    }
                    "find" | "rfind" => {
                        let (needle,): (String,) = from_args(args)?;
                        let haystack = value.as_str().unwrap_or_default();
                        let idx = if method == "find" {
                            haystack.find(&needle)
                        } else {
                            haystack.rfind(&needle)
                        }
                        .map(|v| v as i64)
                        .unwrap_or(-1);
                        return Ok(minijinja::Value::from(idx));
                    }
                    "replace" => {
                        let (from, to): (String, String) = from_args(args)?;
                        let s = value.as_str().unwrap_or_default().replace(&from, &to);
                        return Ok(minijinja::Value::from(s));
                    }
                    "strip" => {
                        let _: () = from_args(args)?;
                        let s = value.as_str().unwrap_or_default().trim().to_string();
                        return Ok(minijinja::Value::from(s));
                    }
                    "rstrip" => {
                        let _: () = from_args(args)?;
                        let s = value.as_str().unwrap_or_default().trim_end().to_string();
                        return Ok(minijinja::Value::from(s));
                    }
                    _ => {}
                }
            }
            if value.kind() == ValueKind::Map {
                match method {
                    "items" => {
                        let _: () = from_args(args)?;
                        return state.apply_filter("items", std::slice::from_ref(value));
                    }
                    "keys" => {
                        let _: () = from_args(args)?;
                        return state
                            .apply_filter("dictsort", std::slice::from_ref(value))
                            .and_then(|sorted| {
                                state.apply_filter(
                                    "map",
                                    &[
                                        sorted,
                                        minijinja::Value::from("attribute"),
                                        minijinja::Value::from(0u32),
                                    ],
                                )
                            })
                            .or_else(|_| state.apply_filter("list", std::slice::from_ref(value)));
                    }
                    "values" => {
                        let _: () = from_args(args)?;
                        return state
                            .apply_filter("dictsort", std::slice::from_ref(value))
                            .and_then(|sorted| {
                                state.apply_filter(
                                    "map",
                                    &[
                                        sorted,
                                        minijinja::Value::from("attribute"),
                                        minijinja::Value::from(1u32),
                                    ],
                                )
                            });
                    }
                    "get" => {
                        let (key, default): (minijinja::Value, Option<minijinja::Value>) =
                            from_args(args)?;
                        return Ok(value
                            .get_item(&key)
                            .ok()
                            .filter(|v| !v.is_undefined())
                            .unwrap_or_else(|| default.unwrap_or(minijinja::Value::UNDEFINED)));
                    }
                    _ => {}
                }
            }
            Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod))
        },
    );

    // 2026-09-26: `tojson_hf` is always the spaced `python_tojson`; `tojson` is replaced by
    // it only for `HfSpaced`, and otherwise stays minijinja's compact builtin. Both keep map
    // key order (`preserve_order` on minijinja and serde_json). Compact is the default
    // because of a measurement: on 2026-06-24, Qwen3.6-27B scored about 30 on BFCL
    // irrelevance prompts with the spaced form and about 96 with the compact form, with
    // rendered prompts that differed only in this serialization.
    env.add_filter("tojson_hf", python_tojson);
    if tool_json_style == ToolJsonStyle::HfSpaced {
        env.add_filter("tojson", python_tojson);
    }

    env.add_template("chat", template_static)
        .context("Failed to compile Jinja chat template")?;
    Ok(env)
}

/// 2026-09-26: `serde_json` formatter that writes `", "` between array items and between
/// object entries, and `": "` after an object key.
#[derive(Clone, Debug)]
struct PythonJsonFormatter;

fn python_tojson(value: minijinja::Value) -> Result<minijinja::Value, minijinja::Error> {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, PythonJsonFormatter);
    serde::Serialize::serialize(&value, &mut serializer).map_err(|error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson serialization failed: {error}"),
        )
    })?;
    let json = String::from_utf8(buf).map_err(|error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson produced invalid UTF-8: {error}"),
        )
    })?;
    Ok(minijinja::Value::from_safe_string(json))
}

impl serde_json::ser::Formatter for PythonJsonFormatter {
    #[inline]
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }
}

/// 2026-09-26: Load `jinja-templates/{model_type}.jinja`, looking under `repo_root` first and
/// then relative to the working directory. A file that exists but cannot be read is logged
/// and skipped.
pub(super) fn load_override_template(model_type: &str, repo_root: Option<&Path>) -> Option<String> {
    let candidates = [
        repo_root.map(|r| {
            r.join(TEMPLATE_OVERRIDE_DIR)
                .join(format!("{model_type}.jinja"))
        }),
        Some(std::path::PathBuf::from(TEMPLATE_OVERRIDE_DIR).join(format!("{model_type}.jinja"))),
    ];
    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            match std::fs::read_to_string(&candidate) {
                Ok(raw) => {
                    let converted = convert_python_jinja_to_minijinja(&raw);
                    tracing::info!(
                        "Using override Jinja template from {} ({} chars)",
                        candidate.display(),
                        converted.len(),
                    );
                    return Some(converted);
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to read override template {}: {e}",
                        candidate.display()
                    );
                }
            }
        }
    }
    None
}

/// 2026-09-26: Load `jinja-templates/openai/{model_type}.jinja`, with the same search order
/// and read-error handling as `load_override_template`.
pub(super) fn load_openai_template(model_type: &str, repo_root: Option<&Path>) -> Option<String> {
    let candidates = [
        repo_root.map(|r| {
            r.join(TEMPLATE_OVERRIDE_DIR)
                .join("openai")
                .join(format!("{model_type}.jinja"))
        }),
        Some(
            std::path::PathBuf::from(TEMPLATE_OVERRIDE_DIR)
                .join("openai")
                .join(format!("{model_type}.jinja")),
        ),
    ];
    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            match std::fs::read_to_string(&candidate) {
                Ok(raw) => {
                    return Some(convert_python_jinja_to_minijinja(&raw));
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to read OpenAI template {}: {e}",
                        candidate.display()
                    );
                }
            }
        }
    }
    None
}

/// 2026-09-26: Load the model's own template: the `chat_template` string of
/// `tokenizer_config.json`, else a standalone `chat_template.jinja`. `Ok(None)` when
/// `tokenizer_config.json` is missing.
pub(super) fn load_config_template(model_dir: &Path) -> Result<Option<String>> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let config_json =
        std::fs::read_to_string(&config_path).context("Failed to read tokenizer_config.json")?;
    let config: serde_json::Value =
        serde_json::from_str(&config_json).context("Failed to parse tokenizer_config.json")?;
    match config.get("chat_template").and_then(|v| v.as_str()) {
        Some(t) => {
            let converted = convert_python_jinja_to_minijinja(t);
            tracing::info!(
                "Loaded Jinja chat template from tokenizer_config.json ({} chars)",
                converted.len()
            );
            Ok(Some(converted))
        }
        None => {
            let jinja_path = model_dir.join("chat_template.jinja");
            if jinja_path.exists() {
                let raw = std::fs::read_to_string(&jinja_path)
                    .context("Failed to read chat_template.jinja")?;
                let converted = convert_python_jinja_to_minijinja(&raw);
                tracing::info!(
                    "Loaded standalone chat_template.jinja ({} chars)",
                    converted.len()
                );
                return Ok(Some(converted));
            }
            Ok(None)
        }
    }
}

/// 2026-09-26: ChatML template for a model with no override and no template of its own. It
/// writes an empty system block when the first message is not a system message, renders a
/// system message only in first position, skips roles other than system, user and
/// assistant, and opens `<think>` in the generation prompt when `supports_thinking`.
pub(super) fn default_chatml_template(supports_thinking: bool) -> String {
    let gen_prompt = if supports_thinking {
        "{{ '<|im_start|>assistant\\n<think>\\n' }}"
    } else {
        "{{ '<|im_start|>assistant\\n' }}"
    };
    format!(
        r#"{{% if messages[0].role != 'system' %}}{{{{ '<|im_start|>system\n<|im_end|>\n' }}}}{{% endif %}}{{% for message in messages %}}{{% if message.role == 'system' %}}{{% if loop.first %}}{{{{ '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}}}{{% endif %}}{{% elif message.role == 'user' %}}{{{{ '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}}}{{% elif message.role == 'assistant' %}}{{{{ '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}}}{{% endif %}}{{% endfor %}}{{% if add_generation_prompt %}}{gen_prompt}{{% endif %}}"#
    )
}

/// 2026-09-26: Literal text rewrites of Python Jinja2 constructs that minijinja cannot parse
/// or run: `messages[::-1]`, fixed `startswith`/`endswith` arguments, `rstrip`/`lstrip`/
/// `strip` of `'\n'`, `split` of the think tags indexed `[0]`/`[-1]`, and the `ensure_ascii`
/// argument of `tojson`. Any other spelling passes through unchanged.
pub(crate) fn convert_python_jinja_to_minijinja(template: &str) -> String {
    let mut t = template.to_string();

    t = t.replace("messages[::-1]", "messages | reverse");

    t = t.replace(
        ".startswith('<tool_response>')",
        " is startingwith '<tool_response>'",
    );
    t = t.replace(".startswith('\\n')", " is startingwith '\\n'");
    t = t.replace(
        ".endswith('</tool_response>')",
        " is endingwith '</tool_response>'",
    );
    t = t.replace(".endswith('\\n')", " is endingwith '\\n'");

    // 2026-09-26: jinja-templates/openai/minimax_m2.jinja chains
    // `.split('</think>')[0].strip('\n').split('<think>')[-1].strip('\n')`; after these
    // rewrites the chain is a left-to-right filter pipeline.
    t = t.replace(".rstrip('\\n')", " | rtrim");
    t = t.replace(".lstrip('\\n')", " | ltrim");
    t = t.replace(".strip('\\n')", " | rtrim | ltrim");

    t = t.replace(".split('</think>')[0]", " | split_first('</think>')");
    t = t.replace(".split('</think>')[-1]", " | split_last('</think>')");
    t = t.replace(
        ".split(think_end_token)[0]",
        " | split_first(think_end_token)",
    );
    t = t.replace(
        ".split(think_end_token)[-1]",
        " | split_last(think_end_token)",
    );

    t = t.replace(".split('<think>')[-1]", " | split_last('<think>')");
    t = t.replace(".split('<think>')[0]", " | split_first('<think>')");

    // 2026-09-26: minijinja's `tojson` rejects an `ensure_ascii` argument, so it is dropped
    // (jinja-templates/openai/minimax_m2.jinja passes `ensure_ascii=False`). Non-ASCII is
    // then written unescaped whichever value was given.
    t = t.replace("tojson(ensure_ascii=False)", "tojson");
    t = t.replace("tojson(ensure_ascii=True)", "tojson");

    t
}
