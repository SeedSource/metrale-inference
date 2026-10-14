// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `resolve_think_tail` tests and the gated GLM-5.3 template tests (real tokenizer and
//! official template from `METRALE_TEST_GLM53_DIR`), moved out of `template.rs` so the
//! `METRALE_TEST_*` reads sit in a test file (`metrale-config` lever placement).
//!
//! Owner: server chat API.
//! Invariants: none beyond the types.

use super::{ThinkTokens, resolve_think_tail};

const START: u32 = 10;
const END: u32 = 11;
const MAX_BUDGET: u32 = 256;
const THINK: ThinkTokens = ThinkTokens {
    start: Some(START),
    end: Some(END),
};

/// 2026-10-12: The GLM-5.3 generation prompt: `<|assistant|>` (7) then an open
/// `<think>`.
fn glm_open_tail() -> Vec<u32> {
    vec![1, 2, 3, 7, START]
}

#[test]
fn request_off_closes_the_open_think() {
    let mut toks = glm_open_tail();
    let out = resolve_think_tail(&mut toks, THINK, false, None, true, Some(MAX_BUDGET));
    assert_eq!(toks, vec![1, 2, 3, 7, START, END]);
    assert_eq!(out, (false, None), "thinking stays off, no budget");
}

#[test]
fn without_request_off_the_template_still_forces_thinking() {
    // 2026-10-12: The car's behaviour (absent directive, `--disable-thinking`,
    // a server or model default) is unchanged: prompt untouched, thinking forced on.
    // 2026-10-14: with a budget, i.e. a model without `effort_prompt_only`.
    let mut toks = glm_open_tail();
    let out = resolve_think_tail(&mut toks, THINK, false, None, false, Some(MAX_BUDGET));
    assert_eq!(toks, glm_open_tail());
    assert_eq!(out, (true, Some(MAX_BUDGET)));
}

#[test]
fn prompt_only_template_forced_thinking_has_no_budget() {
    // 2026-10-14: `effort_prompt_only` (GLM-5.3): the no-key request thinks with no
    // budget, as the official template's default (Max) does; the prompt is untouched.
    let mut toks = glm_open_tail();
    let out = resolve_think_tail(&mut toks, THINK, false, None, false, None);
    assert_eq!(toks, glm_open_tail());
    assert_eq!(out, (true, None));
    // 2026-10-14: the request's own off still closes the block.
    let mut toks = glm_open_tail();
    assert_eq!(
        resolve_think_tail(&mut toks, THINK, false, None, true, None),
        (false, None)
    );
}

#[test]
fn thinking_on_is_unchanged() {
    for close in [false, true] {
        let mut toks = glm_open_tail();
        let out = resolve_think_tail(&mut toks, THINK, true, Some(128), close, Some(MAX_BUDGET));
        assert_eq!(toks, glm_open_tail());
        assert_eq!(out, (true, Some(128)));
    }
}

#[test]
fn a_template_that_already_closed_think_gets_nothing_appended() {
    // 2026-10-12: Qwen-style templates render `<think></think>` themselves when
    // thinking is off.
    let closed = vec![1, 7, START, 5, END, 5];
    let mut toks = closed.clone();
    let out = resolve_think_tail(&mut toks, THINK, false, None, true, Some(MAX_BUDGET));
    assert_eq!(toks, closed);
    assert_eq!(out, (false, None));
}

#[test]
fn no_think_tokens_or_no_end_token_leave_the_prompt_alone() {
    let mut toks = glm_open_tail();
    let none = ThinkTokens {
        start: None,
        end: None,
    };
    assert_eq!(
        resolve_think_tail(&mut toks, none, false, None, true, Some(MAX_BUDGET)),
        (false, None)
    );
    assert_eq!(toks, glm_open_tail());
    // 2026-10-12: No `</think>` id: nothing to close with, the old override applies.
    let start_only = ThinkTokens {
        start: Some(START),
        end: None,
    };
    assert_eq!(
        resolve_think_tail(&mut toks, start_only, false, None, true, Some(MAX_BUDGET)),
        (true, Some(MAX_BUDGET))
    );
    assert_eq!(toks, glm_open_tail());
}

#[test]
fn request_off_answer_is_content_from_its_first_token() {
    // 2026-10-12: The resolved `enable_thinking` drives the reasoning split. After
    // the close it is false, so the GLM (`qwen` format) parser gives no reasoning and
    // keeps the answer's first words; the old forced-on result put a tagless answer
    // in reasoning and left content empty.
    let parser = crate::reasoning_parser::ReasoningFormat::Qwen.into_parser();
    let answer = "Hello there. The answer is 4.";
    let mut toks = glm_open_tail();
    let (et, _) = resolve_think_tail(&mut toks, THINK, false, None, true, Some(MAX_BUDGET));
    assert_eq!(
        parser.extract_thinking(answer, et),
        (None, answer.to_string())
    );
    let mut toks = glm_open_tail();
    let (et_old, _) = resolve_think_tail(&mut toks, THINK, false, None, false, Some(MAX_BUDGET));
    assert_eq!(
        parser.extract_thinking(answer, et_old),
        (Some(answer.to_string()), String::new())
    );
}

/// 2026-10-12: The real GLM-5.3 tokenizer and official zai-org chat template, from
/// `METRALE_TEST_GLM53_DIR` (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`;
/// without the config the loader falls back to ChatML); skipped when
/// unset. The template renders the same tokens for `enable_thinking` absent, true and
/// false (it has no such branch); the request-off prompt is that render plus exactly
/// the `</think>` token, decoding to `...<|assistant|><think></think>`.
#[test]
fn glm53_official_template_off_is_render_plus_think_end() {
    let Ok(dir) = std::env::var("METRALE_TEST_GLM53_DIR") else {
        eprintln!("METRALE_TEST_GLM53_DIR unset; skipping");
        return;
    };
    let tok = crate::tokenizer::ChatTokenizer::from_model_dir(
        std::path::Path::new(&dir),
        0,
        true,
        "glm5_next",
        None,
        false,
    )
    .expect("GLM-5.3 tokenizer");
    let start = tok.encode("<think>").expect("encode")[0];
    let end = tok.think_end_token_id().expect("</think> is one token");
    let think = ThinkTokens {
        start: Some(start),
        end: Some(end),
    };
    let messages = [
        serde_json::json!({"role": "system", "content": "You are a helpful assistant."}),
        serde_json::json!({"role": "user", "content": "Say hello."}),
    ];
    let render = |thinking: bool, effort: Option<&str>| {
        tok.apply_chat_template_openai_with_effort(&messages, None, thinking, false, effort, None)
            .expect("render")
    };
    let base = render(true, None);
    assert_eq!(
        render(false, None),
        base,
        "enable_thinking=false renders the same"
    );
    assert_eq!(
        render(false, Some("none")),
        base,
        "effort none renders the same"
    );
    assert_eq!(
        *base.last().unwrap(),
        start,
        "official render ends in an open <think>"
    );

    let mut off = render(false, None);
    let out = resolve_think_tail(&mut off, think, false, None, true, Some(MAX_BUDGET));
    assert_eq!(out, (false, None));
    assert_eq!(off[..base.len()], base[..]);
    assert_eq!(off[base.len()..], [end]);
    let tail = tok
        .decode_with_special(&off[off.len() - 3..])
        .expect("decode");
    assert!(
        tail.ends_with("<|assistant|><think></think>"),
        "tail {tail:?}"
    );
    assert_eq!(
        tok.decode_with_special(&off).unwrap(),
        format!("{}</think>", tok.decode_with_special(&base).unwrap())
    );
}

/// 2026-10-14: Token-id diff against the official serve. `METRALE_TEST_GLM53_OFFICIAL_RENDERS`
/// names a fixture from spark-bench `scripts/race-pf/thinkcap/official_render.py` (HF
/// `apply_chat_template` on the official nvidia/GLM-5.3-Flash-NVFP4 template, with the
/// kwargs vLLM v0.30.0 builds per request). Each row's request body goes through
/// Metrale's own directive and effort parsing, the template render and the think-tail
/// rule (`effort_prompt_only`: no forced budget). Every effort level and the no-key
/// request must give the official ids exactly, thinking on with no budget; a
/// thinking-off request (`none`, `enable_thinking: false`) gives them plus `</think>`
/// (the official template has no off branch; this matches the THINK=closed reference).
/// Skipped when either variable is unset.
#[test]
fn glm53_effort_prompt_ids_match_the_official_serve() {
    let (Ok(dir), Ok(fixture)) = (
        std::env::var("METRALE_TEST_GLM53_DIR"),
        std::env::var("METRALE_TEST_GLM53_OFFICIAL_RENDERS"),
    ) else {
        eprintln!("METRALE_TEST_GLM53_DIR / METRALE_TEST_GLM53_OFFICIAL_RENDERS unset; skipping");
        return;
    };
    let tok = crate::tokenizer::ChatTokenizer::from_model_dir(
        std::path::Path::new(&dir),
        0,
        true,
        "glm5_next",
        None,
        false,
    )
    .expect("GLM-5.3 tokenizer");
    let think = ThinkTokens {
        start: Some(tok.encode("<think>").expect("encode")[0]),
        end: tok.think_end_token_id(),
    };
    let end = think.end.expect("</think> is one token");
    let fx: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("fixture")).expect("json");
    let rows = fx["rows"].as_array().expect("rows");
    assert!(rows.len() >= 24, "{} rows", rows.len());
    let mut lines = Vec::new();
    for row in rows {
        let mut body = serde_json::json!({"model": "glm", "messages": row["messages"]});
        if let Some(e) = row["effort"].as_str() {
            body["reasoning_effort"] = e.into();
        }
        if row["chat_template_kwargs"]
            .as_object()
            .is_some_and(|o| !o.is_empty())
        {
            body["chat_template_kwargs"] = row["chat_template_kwargs"].clone();
        }
        let req: crate::openai::ChatCompletionRequest =
            serde_json::from_value(body).expect("chat request");
        let directive = req.client_thinking_directive();
        let off = matches!(directive, crate::ir::ThinkingDirective::Off);
        // 2026-10-14: GLM-5.3 `thinking_default` is false: only an explicit
        // directive turns thinking on before the think-tail rule.
        let et = directive.is_explicit() && !off;
        let effort = if et {
            req.client_reasoning_effort()
        } else {
            None
        };
        let mut ids = tok
            .apply_chat_template_openai_with_effort(
                row["messages"].as_array().expect("messages"),
                None,
                et,
                false,
                effort.map(crate::ir::ReasoningEffort::as_str),
                None,
            )
            .expect("render");
        let (et, budget) = resolve_think_tail(&mut ids, think, et, None, off, None);
        let official: Vec<u32> = row["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        let mut want = official.clone();
        if off {
            want.push(end);
        }
        let case = format!("{} {}", row["prompt"], row["case"]);
        assert_eq!(
            ids, want,
            "{case}: token ids differ from the official render"
        );
        assert_eq!((et, budget), (!off, None), "{case}");
        lines.push(format!(
            "{case}: {} ids identical{}",
            ids.len(),
            if off { " + </think>" } else { "" }
        ));
    }
    eprintln!("{}", lines.join("\n"));
}
