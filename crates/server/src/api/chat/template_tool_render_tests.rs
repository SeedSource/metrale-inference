// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Tool rendering parity for GLM-5.3: MODEL.toml `[behavior].hf_tool_json`
//! (spaced HF `tojson`) and `render_tools_on_tool_choice_none` (tools in the prompt for
//! `tool_choice` `none`). The gated test renders TEB turn-1 requests through Metrale's own
//! path and compares the token ids with the official render
//! (`crates/server/tests/fixtures/glm53-teb-tool-renders.json`).
//!
//! Owner: server chat API.
//! Invariants: none beyond the types.

use metrale_kernels::ModelBehavior;

use super::super::prepare::{template_renders_tools, tools_active};
use super::{build_json_messages_for, template_tools};

/// 2026-10-15: GLM-5.3's tool-render flags as `kernels/gb10/glm-5.3-flash/MODEL.toml` sets
/// them.
fn glm53_behavior() -> ModelBehavior {
    ModelBehavior {
        hf_tool_json: true,
        render_tools_on_tool_choice_none: true,
        ..ModelBehavior::default()
    }
}

fn request(tool_choice: &str, with_tools: bool) -> crate::ir::ChatRequest {
    let mut body = serde_json::json!({
        "model": "glm",
        "messages": [{"role": "user", "content": "What is the value of pi?"}],
        "tool_choice": tool_choice,
    });
    if with_tools {
        body["tools"] = serde_json::json!([{"type": "function", "function": {
            "name": "calculator", "description": "Evaluate an expression",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                "required": ["expression"]}}}]);
    }
    let req: crate::openai::ChatCompletionRequest =
        serde_json::from_value(body).expect("chat request");
    req.into()
}

#[test]
fn tool_choice_none_renders_tools_only_under_the_flag() {
    let glm = glm53_behavior();
    let default = ModelBehavior::default();
    let none = request("none", true);
    assert!(!tools_active(&none, true), "none keeps tools inactive");
    assert!(template_renders_tools(false, &none, true, &glm));
    assert!(!template_renders_tools(false, &none, true, &default));
    // 2026-10-15: No parser configured, or no tools declared: nothing to render.
    assert!(!template_renders_tools(false, &none, false, &glm));
    assert!(!template_renders_tools(
        false,
        &request("none", false),
        true,
        &glm
    ));
    // 2026-10-15: Active tools render for every model, as before.
    let auto = request("auto", true);
    assert!(tools_active(&auto, true));
    for b in [&glm, &default] {
        assert!(template_renders_tools(true, &auto, true, b));
    }
    assert!(template_tools(&none.tools, true, false, false).is_some());
    assert!(template_tools(&none.tools, false, false, false).is_none());
}

/// 2026-10-15: The first byte where `got` and `want` differ, with context, for a failure
/// message that says exactly where a render diverges.
fn first_difference(got: &str, want: &str) -> String {
    let at = got
        .bytes()
        .zip(want.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(got.len().min(want.len()));
    let ctx = |s: &str| {
        let mut lo = at.saturating_sub(80).min(s.len());
        while !s.is_char_boundary(lo) {
            lo -= 1;
        }
        let mut hi = (at + 80).min(s.len());
        while !s.is_char_boundary(hi) {
            hi += 1;
        }
        s[lo..hi].to_string()
    };
    format!(
        "first difference at byte {at} (got {} bytes, want {}):\n got: {:?}\nwant: {:?}",
        got.len(),
        want.len(),
        ctx(got),
        ctx(want)
    )
}

/// 2026-10-15: TEB TC-10, TC-38, TC-65 (tools, `tool_choice` auto) and TC-44 (`tool_choice`
/// `none`) through Metrale's path under GLM-5.3's behavior: tokenizer built with
/// `hf_tool_json`, `tools_active` / `template_renders_tools`, `build_msg_entries`,
/// `build_json_messages_for`, `template_tools`, the template render (thinking as the
/// no-key request renders it: off, no effort). The ids must equal the official text's ids
/// (the checkpoint tokenizer). Needs `METRALE_TEST_GLM53_DIR` (the nvidia tokenizer and
/// template, sha256 34d5ee66); skipped when unset.
#[test]
fn glm53_teb_tool_prompts_match_the_official_render() {
    let Ok(dir) = std::env::var("METRALE_TEST_GLM53_DIR") else {
        eprintln!("METRALE_TEST_GLM53_DIR unset; skipping");
        return;
    };
    let glm = glm53_behavior();
    let tok = crate::tokenizer::ChatTokenizer::from_model_dir_with(
        std::path::Path::new(&dir),
        0,
        true,
        "glm5_next",
        None,
        false,
        glm.hf_tool_json,
    )
    .expect("GLM-5.3 tokenizer");
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/glm53-teb-tool-renders.json");
    let fx: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("fixture")).expect("json");
    let rows = fx["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 4);
    let mut failures = Vec::new();
    for row in rows {
        let id = row["id"].as_str().expect("id");
        let mut body = row["request"].clone();
        body["model"] = "glm".into();
        let wire: crate::openai::ChatCompletionRequest =
            serde_json::from_value(body).expect("chat request");
        let req: crate::ir::ChatRequest = wire.into();
        let active = tools_active(&req, true);
        let render_tools = template_renders_tools(active, &req, true, &glm);
        assert!(render_tools, "{id}: every fixture row renders its tools");
        let built = super::super::msg_entry::build_msg_entries(
            None,
            None,
            &super::super::remote_image::RemoteImagePolicy::default(),
            &super::super::msg_entry::VideoDecode {
                ffmpeg: &metrale_model_layers::video_decode_ffmpeg::FfmpegPolicy {
                    enabled: false,
                    ..Default::default()
                },
                fps: 2.0,
            },
            &req.messages,
            active,
            &super::super::levers::ChatLevers::OFF,
            false,
        )
        .unwrap_or_else(|_| panic!("{id}: message build"));
        let json_messages = build_json_messages_for(&built.messages, false);
        let jinja_tools = template_tools(&req.tools, render_tools, false, false);
        let ids = tok
            .apply_chat_template_openai_with_effort(
                &json_messages,
                jinja_tools.as_deref(),
                false,
                glm.disable_tool_steering,
                None,
                None,
            )
            .expect("render");
        let official = row["official_text"].as_str().expect("official_text");
        let want = tok.encode(official).expect("encode official");
        if ids != want {
            let got = tok.decode_with_special(&ids).expect("decode");
            failures.push(format!(
                "{id}: {} ids vs official {}; {}",
                ids.len(),
                want.len(),
                first_difference(&got, official)
            ));
        } else {
            eprintln!("{id}: {} ids identical to the official render", ids.len());
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
