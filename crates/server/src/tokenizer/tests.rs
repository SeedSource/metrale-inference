// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the tokenizer module: tool-argument normalization, renders of the
//! override templates in jinja-templates/, the DeepSeek-V4 reasoning parser entry, and the
//! two `tojson` styles. The child modules hold per-model tests.
//!
//! Owner: server (tokenizer) tests.
//! Invariants: none beyond the types.

use super::*;

mod deepseek_v4;
mod kimi_k3;
use serde_json::json;

mod laguna;
mod mistral_effort;
mod qwen_dense;
mod qwen_dense_parity;

fn render_minimax_openai_template(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
) -> String {
    let template_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/openai/minimax_m2.jinja"
    );
    let raw = std::fs::read_to_string(template_path)
        .expect("bundled MiniMax OpenAI template must be present in the repo");
    let converted = super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let messages_for_render = normalize_tool_call_arguments(messages);
    let messages_val = minijinja::Value::from_serialize(&messages_for_render);
    let tools_val = tools.map(minijinja::Value::from_serialize);
    let reasoning_effort: minijinja::Value = if enable_thinking {
        "high".into()
    } else {
        "none".into()
    };
    let ctx = minijinja::context! {
        messages => messages_val,
        tools => tools_val.unwrap_or(minijinja::Value::UNDEFINED),
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
        reasoning_effort => reasoning_effort,
        disable_tool_steering => false,
        add_vision_id => false,
    };
    tmpl.render(ctx).expect("template renders")
}

fn render_holo_template(messages: &[serde_json::Value], enable_thinking: bool) -> String {
    let template_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/holo3_1_moe.jinja"
    );
    let raw = std::fs::read_to_string(template_path)
        .expect("bundled Holo3.1 template must be present in the repo");
    let converted = super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let messages_for_render = normalize_tool_call_arguments(messages);
    let messages_val = minijinja::Value::from_serialize(&messages_for_render);
    let ctx = minijinja::context! {
        messages => messages_val,
        tools => minijinja::Value::UNDEFINED,
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
        reasoning_effort => "none",
        disable_tool_steering => false,
        add_vision_id => false,
    };
    tmpl.render(ctx).expect("template renders")
}

#[test]
fn normalize_tool_call_arguments_parses_string_to_dict() {
    let messages = vec![json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_0",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": "{\"command\":\"mkdir -p /tmp/x\",\"description\":\"make dir\"}"
            }
        }]
    })];
    let normalized = normalize_tool_call_arguments(&messages);
    let args = &normalized[0]["tool_calls"][0]["function"]["arguments"];
    assert!(args.is_object(), "expected dict, got {args:?}");
    assert_eq!(args["command"], "mkdir -p /tmp/x");
    assert_eq!(args["description"], "make dir");
}

#[test]
fn normalize_tool_call_arguments_leaves_non_tool_messages_alone() {
    let messages = vec![
        json!({"role": "user", "content": "hi"}),
        json!({"role": "assistant", "content": "hello"}),
    ];
    let normalized = normalize_tool_call_arguments(&messages);
    assert_eq!(normalized, messages);
}

#[test]
fn render_holo_template_accepts_vllm_thinking_controls() {
    let messages = vec![
        json!({"role": "developer", "content": "<|think_off|>Follow the instruction."}),
        json!({"role": "user", "content": "Reply with OK."}),
    ];
    let rendered = render_holo_template(&messages, true);
    assert!(rendered.contains("Follow the instruction."));
    assert!(!rendered.contains("<|think_off|>"));
    assert!(
        rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "expected closed thinking prompt from think_off: {rendered}"
    );
}

#[test]
fn render_holo_template_autocloses_think_before_tool_call() {
    let messages = vec![
        json!({"role": "user", "content": "Use bash."}),
        json!({
            "role": "assistant",
            "content": "<think>\nNeed a directory listing.\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
        }),
    ];
    let rendered = render_holo_template(&messages, true);
    assert!(
        rendered.contains("Need a directory listing.\n</think>\n\n<tool_call>"),
        "expected unclosed think to be closed before tool call: {rendered}"
    );
}

#[test]
fn normalize_tool_call_arguments_passes_through_already_dict() {
    let messages = vec![json!({
        "role": "assistant",
        "tool_calls": [{
            "function": {"name": "bash", "arguments": {"command": "ls"}}
        }]
    })];
    let normalized = normalize_tool_call_arguments(&messages);
    assert_eq!(
        normalized[0]["tool_calls"][0]["function"]["arguments"]["command"],
        "ls"
    );
}

/// 2026-09-26: Renders the MiniMax M2.7 checkpoint template from a fixed local Hugging Face
/// cache path, after `normalize_tool_call_arguments`, and returns early without asserting
/// when that file is absent.
#[test]
fn render_minimax_template_with_string_tool_call_args() {
    let template_path = "/workspace/.cache/huggingface/hub/models--lukealonso--MiniMax-M2.7-NVFP4/snapshots/ba6a625013cdacdc560f6203d177c0f27d41775e/chat_template.jinja";
    let Ok(template) = std::fs::read_to_string(template_path) else {
        eprintln!("MiniMax template not on disk; skipping");
        return;
    };
    let env = super::jinja_helpers::build_jinja_env(&template).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let messages = vec![
        json!({"role": "user", "content": "List /tmp"}),
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": "{\"command\":\"ls -la /tmp\"}"
                }
            }]
        }),
        json!({"role": "tool", "tool_call_id": "call_0", "content": "total 0"}),
        json!({"role": "user", "content": "Now uname -r"}),
    ];
    let normalized = normalize_tool_call_arguments(&messages);
    let messages_val = minijinja::Value::from_serialize(&normalized);
    let ctx = minijinja::context! {
        messages => messages_val,
        tools => minijinja::Value::UNDEFINED,
        add_generation_prompt => true,
        enable_thinking => true,
        reasoning_effort => "high",
        disable_tool_steering => false,
        add_vision_id => false,
    };
    let rendered = tmpl
        .render(ctx)
        .expect("F76 must keep MiniMax template from raising on second-turn");
    assert!(
        rendered.contains("<invoke name=\"bash\">"),
        "expected `<invoke name=\"bash\">` in render: {rendered}"
    );
    assert!(
        rendered.contains("<parameter name=\"command\">"),
        "expected `<parameter name=\"command\">` from .items() iteration: {rendered}"
    );
    assert!(
        rendered.contains("ls -la /tmp"),
        "expected the parsed command value in render: {rendered}"
    );
}

#[test]
fn render_minimax_openai_template_closes_think_prompt_when_disabled() {
    let messages = vec![json!({"role": "user", "content": "Reply with exactly: OK"})];
    let rendered = render_minimax_openai_template(&messages, None, false);
    assert!(
        rendered.ends_with("]~b]ai\n<think>\n\n</think>\n\n"),
        "expected closed-thinking assistant generation prompt: {rendered}"
    );
    let generation_tail = rendered
        .rsplit_once("]~b]ai\n")
        .map(|(_, tail)| tail)
        .expect("assistant generation prompt is present");
    assert_eq!(
        generation_tail, "<think>\n\n</think>\n\n",
        "disabled thinking must not leave the model inside <think>: {rendered}"
    );
}

#[test]
fn render_minimax_openai_template_opens_think_prompt_when_enabled() {
    let messages = vec![json!({"role": "user", "content": "Think before answering"})];
    let rendered = render_minimax_openai_template(&messages, None, true);
    assert!(
        rendered.ends_with("]~b]ai\n<think>\n"),
        "expected thinking assistant generation prompt: {rendered}"
    );
}

#[test]
fn render_minimax_openai_template_omits_think_prompt_with_tools_when_disabled() {
    let messages = vec![json!({"role": "user", "content": "List the current directory"})];
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    })];
    let rendered = render_minimax_openai_template(&messages, Some(&tools), false);
    assert!(
        rendered.contains("<tools>"),
        "expected tool schema block in render: {rendered}"
    );
    assert!(
        rendered.contains("<minimax:tool_call>"),
        "expected MiniMax tool-call instructions in render: {rendered}"
    );
    assert!(
        rendered.ends_with("]~b]ai\n<think>\n\n</think>\n\n"),
        "tool-active disabled-thinking requests must use a closed-thinking assistant prompt: {rendered}"
    );
}

#[test]
fn normalize_tool_call_arguments_invalid_json_string_left_alone() {
    let messages = vec![json!({
        "role": "assistant",
        "tool_calls": [{
            "function": {"name": "bash", "arguments": "not valid json {"}
        }]
    })];
    let normalized = normalize_tool_call_arguments(&messages);
    assert_eq!(
        normalized[0]["tool_calls"][0]["function"]["arguments"],
        "not valid json {"
    );
}

/// 2026-09-26: jinja-templates/gemma4.jinja calls `.split(...)` on strings in its
/// `strip_thinking` macro, which minijinja runs only through the unknown-method callback
/// installed by `build_jinja_env_with`. The conversation also holds a tool message with
/// null content.
#[test]
fn render_gemma4_template_with_assistant_and_null_tool_content() {
    let template_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/gemma4.jinja"
    );
    let raw = std::fs::read_to_string(template_path)
        .expect("bundled gemma4.jinja must be present in the repo");
    let converted = super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let messages = vec![
        json!({"role": "user", "content": "What time is it?"}),
        json!({"role": "assistant", "content": "I'll check."}),
        json!({"role": "tool", "content": null}),
        json!({"role": "user", "content": "Thanks."}),
    ];
    let messages_val = minijinja::Value::from_serialize(&messages);
    let ctx = minijinja::context! {
        messages => messages_val,
        tools => minijinja::Value::UNDEFINED,
        add_generation_prompt => true,
        enable_thinking => false,
        bos_token => "<bos>",
    };
    let rendered = tmpl
        .render(ctx)
        .expect("Gemma-4 template must render assistant + null-content tool message");
    assert!(
        rendered.contains("I'll check."),
        "expected assistant content in render: {rendered}"
    );
}

/// 2026-09-26: jinja-templates/deepseek_v4.jinja ends the generation prompt with `<think>`
/// only when `enable_thinking` is true, and with `</think>` when it is false or undefined.
#[test]
fn deepseek_v4_reasoning_primer_is_opt_in() {
    let template_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/deepseek_v4.jinja"
    );
    let raw = std::fs::read_to_string(template_path)
        .expect("bundled deepseek_v4.jinja must be present in the repo");
    let converted = super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();

    let messages = vec![json!({"role": "user", "content": "2+2?"})];
    let mv = minijinja::Value::from_serialize(&messages);
    let render = |et: minijinja::Value| {
        tmpl.render(minijinja::context! {
            messages => mv.clone(),
            tools => minijinja::Value::UNDEFINED,
            add_generation_prompt => true,
            enable_thinking => et,
        })
        .expect("deepseek_v4 template must render")
    };

    let thinking = render(minijinja::Value::from(true));
    assert!(
        thinking.ends_with("<｜Assistant｜><think>"),
        "reasoning mode must prime <think>: {thinking:?}"
    );

    let direct = render(minijinja::Value::from(false));
    assert!(
        direct.ends_with("<｜Assistant｜></think>"),
        "direct suffix must be the official <｜Assistant｜></think>: {direct:?}"
    );
    assert!(
        !direct.contains("<think>"),
        "direct mode must NOT open a <think> block: {direct:?}"
    );

    let default = render(minijinja::Value::UNDEFINED);
    assert_eq!(
        default, direct,
        "default (unspecified) must render the official direct suffix"
    );
}

/// 2026-09-26: `supports_thinking` is false for a model without linear-attention or Mamba-2
/// layers, and the DeepSeek-V4 config parser makes every layer full attention. Its reasoning
/// parser, and with it `think_end_token`, therefore comes only from the `[reasoning]` table
/// of tool_defaults.toml; the test checks that entry and its `<think>`/`</think>` tags.
#[test]
fn deepseek_v4_reasoning_parser_is_registered() {
    use crate::reasoning_parser::ReasoningFormat;
    let defaults_toml = include_str!("../../tool_defaults.toml");
    let defaults: toml::Value = toml::from_str(defaults_toml).expect("tool_defaults parses");
    let fmt_str = defaults
        .get("reasoning")
        .and_then(|t| t.get("deepseek_v4"))
        .and_then(|s| s.as_str())
        .expect("tool_defaults [reasoning] must register deepseek_v4");
    let fmt: ReasoningFormat = fmt_str
        .parse()
        .expect("deepseek_v4 reasoning format parses");
    let p = fmt.into_parser();
    assert_eq!(p.start_tag(), "<think>", "DS4F reasoning start tag");
    assert_eq!(p.end_tag(), "</think>", "DS4F reasoning end tag");
}

/// 2026-09-26: `tojson` is compact from `build_jinja_env` without the env var and spaced with
/// `ToolJsonStyle::HfSpaced`; both keep the input's key order.
#[test]
fn tojson_filter_default_compact_hf_ref_opt_in() {
    // 2026-09-26: The key order `ToolDefinition` serializes in (tool_parser.rs).
    let tool_value = serde_json::json!({
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run"
                    }
                },
                "required": ["command"]
            }
        }
    });

    let env_default = super::jinja_helpers::build_jinja_env("{{ tool | tojson }}")
        .expect("inline template compiles");
    let compact = "{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"description\":\"Execute a bash command\",\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\",\"description\":\"The command to run\"}},\"required\":[\"command\"]}}}";
    let got_default = env_default
        .get_template("chat")
        .unwrap()
        .render(minijinja::context! { tool => minijinja::Value::from_serialize(&tool_value) })
        .expect("tojson render");
    assert_eq!(
        got_default, compact,
        "default tojson must be COMPACT (ST-995 GDN irrelevance fix)\ngot:\n{got_default}\n"
    );

    let env_hf = super::jinja_helpers::build_jinja_env_with(
        "{{ tool | tojson }}",
        super::jinja_helpers::ToolJsonStyle::HfSpaced,
    )
    .expect("inline template compiles");
    let spaced = "{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"description\": \"Execute a bash command\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\", \"description\": \"The command to run\"}}, \"required\": [\"command\"]}}}";
    let got_hf = env_hf
        .get_template("chat")
        .unwrap()
        .render(minijinja::context! { tool => minijinja::Value::from_serialize(&tool_value) })
        .expect("tojson render");
    assert_eq!(
        got_hf, spaced,
        "METRALE_USE_HF_REF_JSON_DUMPS=1 must restore Python json.dumps byte parity\ngot:\n{got_hf}\n"
    );
}

/// 2026-10-15: MODEL.toml `[behavior].hf_tool_json` (GLM-5.3): `build_jinja_env_for` with
/// `hf_default` renders the spaced form without the env var; without it, the production
/// default stays compact.
#[test]
fn hf_tool_json_flag_selects_the_spaced_tojson() {
    assert_eq!(
        super::jinja_helpers::tool_json_style(true),
        super::jinja_helpers::ToolJsonStyle::HfSpaced
    );
    let value = serde_json::json!({"type": "object", "required": ["a", "b"]});
    let render = |hf: bool| {
        super::jinja_helpers::build_jinja_env_for("{{ v | tojson }}", hf)
            .expect("inline template compiles")
            .get_template("chat")
            .unwrap()
            .render(minijinja::context! { v => minijinja::Value::from_serialize(&value) })
            .expect("tojson render")
    };
    assert_eq!(render(true), "{\"type\": \"object\", \"required\": [\"a\", \"b\"]}");
    if std::env::var("METRALE_USE_HF_REF_JSON_DUMPS").as_deref() != Ok("1") {
        assert_eq!(render(false), "{\"type\":\"object\",\"required\":[\"a\",\"b\"]}");
    }
}
