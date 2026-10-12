// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `ChatCompletionRequest::client_thinking_directive`: each request
//! channel for thinking intent and the priority between them.
//!
//! Owner: server (OpenAI API layer) tests.
//! Invariants: none beyond the types.

use crate::ir::{EffortLevel, ThinkingDirective};
use crate::openai::*;

fn chat_req(body: serde_json::Value) -> ChatCompletionRequest {
    serde_json::from_value(body).expect("valid chat request")
}

fn base_body() -> serde_json::Value {
    serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    })
}

#[test]
fn silent_request_is_unspecified() {
    let req = chat_req(base_body());
    assert_eq!(
        req.client_thinking_directive(),
        ThinkingDirective::Unspecified
    );
    assert!(!req.client_thinking_directive().is_explicit());
}

#[test]
fn anthropic_thinking_channel() {
    let mut b = base_body();
    b["thinking"] = serde_json::json!({"type": "disabled", "budget_tokens": 100});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    let mut b = base_body();
    b["thinking"] = serde_json::json!({"type": "enabled", "budget_tokens": 512});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(512) }
    );

    // 2026-09-26: A thinking object without a budget gives `budget: None`, which the chat
    // path resolves to the model's `max_thinking_budget`.
    let mut b = base_body();
    b["thinking"] = serde_json::json!({"type": "adaptive"});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: None }
    );
}

#[test]
fn thinking_token_budget_channel() {
    let mut b = base_body();
    b["thinking_token_budget"] = serde_json::json!(512);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(512) }
    );

    let mut b = base_body();
    b["thinking_token_budget"] = serde_json::json!(0);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );
}

#[test]
fn reasoning_effort_channel() {
    for (effort, expect) in [
        ("none", ThinkingDirective::Off),
        ("minimal", ThinkingDirective::OnEffort(EffortLevel::Minimal)),
        ("low", ThinkingDirective::OnEffort(EffortLevel::Low)),
        ("medium", ThinkingDirective::OnEffort(EffortLevel::Medium)),
        ("high", ThinkingDirective::OnEffort(EffortLevel::High)),
        ("xhigh", ThinkingDirective::OnEffort(EffortLevel::XHigh)),
        ("max", ThinkingDirective::OnEffort(EffortLevel::XHigh)),
        // 2026-09-26: An unknown effort resolves as if the field were absent; the chat
        // handler rejects it with a 400 before this runs (`unknown_effort_fails_validation`).
        ("bogus", ThinkingDirective::Unspecified),
    ] {
        let mut b = base_body();
        b["reasoning"] = serde_json::json!({"effort": effort});
        assert_eq!(
            chat_req(b).client_thinking_directive(),
            expect,
            "effort={effort}"
        );
    }
}

#[test]
fn top_level_reasoning_effort_channel() {
    for (effort, expect) in [
        ("none", ThinkingDirective::Off),
        ("minimal", ThinkingDirective::OnEffort(EffortLevel::Minimal)),
        ("low", ThinkingDirective::OnEffort(EffortLevel::Low)),
        ("medium", ThinkingDirective::OnEffort(EffortLevel::Medium)),
        ("high", ThinkingDirective::OnEffort(EffortLevel::High)),
    ] {
        let mut b = base_body();
        b["reasoning_effort"] = serde_json::json!(effort);
        assert_eq!(
            chat_req(b).client_thinking_directive(),
            expect,
            "effort={effort}"
        );
    }

    let mut b = base_body();
    b["reasoning"] = serde_json::json!({"effort": "high"});
    b["reasoning_effort"] = serde_json::json!("low");
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::OnEffort(EffortLevel::High)
    );
}

#[test]
fn thinking_budget_aliases_thinking_token_budget() {
    let mut b = base_body();
    b["thinking_budget"] = serde_json::json!(2048);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(2048) }
    );

    let mut b = base_body();
    b["thinking_budget"] = serde_json::json!(0);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    let mut b = base_body();
    b["thinking_budget"] = serde_json::json!(2048);
    b["reasoning_effort"] = serde_json::json!("low");
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(2048) }
    );
}

#[test]
fn top_level_reasoning_effort_channel_and_nested_priority() {
    let mut top_level = base_body();
    top_level["reasoning_effort"] = serde_json::json!("max");
    let req = chat_req(top_level);
    assert_eq!(
        req.client_thinking_directive(),
        ThinkingDirective::OnEffort(EffortLevel::XHigh)
    );
    assert_eq!(
        req.client_reasoning_effort(),
        Some(crate::ir::ReasoningEffort::Max)
    );

    let mut both = base_body();
    both["reasoning_effort"] = serde_json::json!("max");
    both["reasoning"] = serde_json::json!({"effort": "high"});
    let req = chat_req(both);
    assert_eq!(
        req.client_thinking_directive(),
        ThinkingDirective::OnEffort(EffortLevel::High)
    );
    assert_eq!(
        req.client_reasoning_effort(),
        Some(crate::ir::ReasoningEffort::High)
    );
}

#[test]
fn chat_template_kwargs_channel() {
    let kw: ChatTemplateKwargs =
        serde_json::from_str(r#"{"enable_thinking":true,"thinking_budget":1024}"#)
            .expect("should parse");
    assert_eq!(kw.enable_thinking, Some(true));
    assert_eq!(kw.thinking_budget, Some(1024));
    assert_eq!(kw.preserve_thinking, None);
    let kw: ChatTemplateKwargs =
        serde_json::from_str(r#"{"preserve_thinking":false}"#).expect("should parse");
    assert_eq!(kw.preserve_thinking, Some(false));

    let mut b = base_body();
    b["chat_template_kwargs"] =
        serde_json::json!({"enable_thinking": false, "thinking_budget": 1024});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(1024) }
    );

    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"thinking_budget": 0});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    // 2026-09-26: `enable_thinking: true` alone gives `budget: None`, which the chat path
    // resolves to the model's `max_thinking_budget`.
    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"enable_thinking": true});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: None }
    );

    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Unspecified
    );
}

/// 2026-09-26: `chat_template_kwargs.reasoning_effort` reaches both the thinking directive
/// and the template-facing effort (`client_reasoning_effort`).
#[test]
fn chat_template_kwargs_reasoning_effort_channel() {
    use crate::ir::ReasoningEffort;

    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"reasoning_effort": "low"});
    let req = chat_req(b);
    assert_eq!(
        req.client_thinking_directive(),
        ThinkingDirective::OnEffort(EffortLevel::Low)
    );
    assert_eq!(req.client_reasoning_effort(), Some(ReasoningEffort::Low));

    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"reasoning_effort": "none"});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    // 2026-09-26: `enable_thinking: true` beside an effort string leaves the effort's tier
    // in charge.
    let mut b = base_body();
    b["chat_template_kwargs"] =
        serde_json::json!({"enable_thinking": true, "reasoning_effort": "xhigh"});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::OnEffort(EffortLevel::XHigh)
    );

    // 2026-09-26: Within the kwargs object `enable_thinking: false` outranks the effort
    // string. The template side is covered separately: `effective_reasoning_effort` in
    // api/chat/prepare.rs passes no effort while thinking is off.
    let mut b = base_body();
    b["chat_template_kwargs"] =
        serde_json::json!({"enable_thinking": false, "reasoning_effort": "low"});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    let mut b = base_body();
    b["reasoning_effort"] = serde_json::json!("xhigh");
    b["chat_template_kwargs"] = serde_json::json!({"reasoning_effort": "low"});
    let req = chat_req(b);
    assert_eq!(
        req.client_thinking_directive(),
        ThinkingDirective::OnEffort(EffortLevel::XHigh)
    );
    assert_eq!(req.client_reasoning_effort(), Some(ReasoningEffort::Max));
}

/// 2026-09-26: An unknown effort string fails validation on every channel, including one
/// shadowed by a valid higher-priority value. The chat handler turns the error into a 400
/// before lowering to the IR.
#[test]
fn unknown_effort_fails_validation() {
    for body in [
        serde_json::json!({"reasoning": {"effort": "hgih"}}),
        serde_json::json!({"reasoning_effort": "hgih"}),
        serde_json::json!({"chat_template_kwargs": {"reasoning_effort": "hgih"}}),
    ] {
        let mut b = base_body();
        for (k, v) in body.as_object().unwrap() {
            b[k] = v.clone();
        }
        let err = chat_req(b).validate_reasoning_effort().unwrap_err();
        assert!(err.contains("hgih"), "message names the bad value: {err}");
    }

    let mut b = base_body();
    b["reasoning_effort"] = serde_json::json!("low");
    b["chat_template_kwargs"] = serde_json::json!({"reasoning_effort": "hgih"});
    assert!(chat_req(b).validate_reasoning_effort().is_err());

    for effort in ["none", "minimal", "low", "medium", "high", "xhigh", "max"] {
        let mut b = base_body();
        b["reasoning_effort"] = serde_json::json!(effort);
        assert!(chat_req(b).validate_reasoning_effort().is_ok());
    }
    assert!(chat_req(base_body()).validate_reasoning_effort().is_ok());
}

#[test]
fn legacy_enable_thinking_channel() {
    let mut b = base_body();
    b["enable_thinking"] = serde_json::json!(true);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: None }
    );

    let mut b = base_body();
    b["enable_thinking"] = serde_json::json!(false);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    let b = base_body();
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Unspecified
    );
}

#[test]
fn preserve_thinking_lowers_to_ir_tri_state() {
    // 2026-09-26: `chat_template_kwargs.preserve_thinking` reaches the IR request. `None`
    // lets api/chat/prepare.rs fall back to MODEL.toml `[behavior].preserve_thinking`,
    // and then to the template's own default.
    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"preserve_thinking": true});
    let ir: crate::ir::ChatRequest = chat_req(b).into();
    assert_eq!(ir.preserve_thinking, Some(true));

    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"preserve_thinking": false});
    let ir: crate::ir::ChatRequest = chat_req(b).into();
    assert_eq!(ir.preserve_thinking, Some(false));

    let ir: crate::ir::ChatRequest = chat_req(base_body()).into();
    assert_eq!(ir.preserve_thinking, None);
}

#[test]
fn reasoning_effort_strings_render_template_safe() {
    // 2026-09-26: Templates read these spellings. The Qwen3.8 template in
    // test_data/chat_templates/qwen3.8-27b-unsloth.jinja maps "high" to "xhigh", accepts
    // only xhigh, medium and low, and raises on anything else, so `Max` must render as
    // "xhigh". "medium" and "low" render different instructions there.
    use crate::ir::ReasoningEffort;
    assert_eq!(ReasoningEffort::Max.as_str(), "xhigh");
    assert_eq!(ReasoningEffort::Medium.as_str(), "medium");
    assert_eq!(ReasoningEffort::High.as_str(), "high");
    assert_eq!(ReasoningEffort::Low.as_str(), "low");

    let mut b = base_body();
    b["reasoning_effort"] = serde_json::json!("medium");
    assert_eq!(
        chat_req(b).client_reasoning_effort(),
        Some(ReasoningEffort::Medium)
    );
}

/// 2026-10-12: Other engines' clients spell the switch `thinking` (SGLang) or
/// `thinking_mode` (MiniMax) inside `chat_template_kwargs`; sparkDash sends all
/// three, SparkGLM's `staggered_openai.py` sends `thinking` + `enable_thinking`.
#[test]
fn chat_template_kwargs_thinking_spellings() {
    let directive = |kwargs: serde_json::Value| {
        let mut b = base_body();
        b["chat_template_kwargs"] = kwargs;
        chat_req(b).client_thinking_directive()
    };
    use serde_json::json;
    assert_eq!(
        directive(json!({"thinking": false})),
        ThinkingDirective::Off
    );
    assert_eq!(
        directive(json!({"thinking_mode": "disabled"})),
        ThinkingDirective::Off
    );
    assert_eq!(
        directive(
            json!({"enable_thinking": false, "thinking": false, "thinking_mode": "disabled"})
        ),
        ThinkingDirective::Off
    );
    assert_eq!(
        directive(json!({"thinking": true})),
        ThinkingDirective::On { budget: None }
    );
    assert_eq!(
        directive(json!({"thinking_mode": "enabled"})),
        ThinkingDirective::On { budget: None }
    );
    // 2026-10-12: `enable_thinking` wins over the alternate spellings.
    assert_eq!(
        directive(json!({"enable_thinking": true, "thinking": false})),
        ThinkingDirective::On { budget: None }
    );
    // 2026-10-12: An unknown `thinking_mode` string is ignored.
    assert_eq!(
        directive(json!({"thinking_mode": "auto"})),
        ThinkingDirective::Unspecified
    );
    // 2026-10-12: `thinking_budget` still outranks the switch.
    assert_eq!(
        directive(json!({"thinking": false, "thinking_budget": 512})),
        ThinkingDirective::On { budget: Some(512) }
    );
}
