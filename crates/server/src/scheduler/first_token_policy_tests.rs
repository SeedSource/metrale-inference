// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the token-0 grammar policy (`first_token_policy`)
//! and for `</think>` on the `emit_token` path.
//!
//! Grammars are real matchers over the toy vocabulary of `grammar::tests`.
//! The model's sampler is replaced by the closure in `run`, which returns
//! `HELLO` without a grammar and `masked_pick` with one.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::emit_step::emit_token;
use super::first_token_policy::{FirstTokenPolicy, born_inside_thinking, first_token_with};
use super::sched_ctx::SchedCtx;
use super::test_support::test_seq;
use crate::grammar::tests::{test_tool_defs, test_vocab};
use crate::grammar::{GrammarEngine, GrammarState};

// 2026-09-25: ids 128 and 130 in the toy vocab: `<tool_call>` and `<eos>`.
const TOOL_CALL_OPEN: u32 = 128;
const EOS: u32 = 130;
/// 2026-09-25: A content byte neither test grammar allows as its first
/// token.
const HELLO: u32 = b'h' as u32;
/// 2026-09-25: `</think>`, outside the toy vocab (ids below `VOCAB_LEN`):
/// the matcher must never be fed it.
const THINK_END: u32 = 151_668;
const VOCAB_LEN: u32 = 131;

/// 2026-09-25: Hermes tool grammar without triggers (the
/// `tool_choice="required"` form): the output must open a tool call.
fn required_tool_grammar() -> GrammarState {
    let vocab = test_vocab();
    let mut engine = GrammarEngine::new(&vocab, &[EOS as i32]).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), false)
        .unwrap();
    GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS])
}

fn json_schema_grammar() -> GrammarState {
    let vocab = test_vocab();
    let mut engine = GrammarEngine::new(&vocab, &[EOS as i32]).unwrap();
    let compiled = engine
        .compile_json_schema(
            r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#,
        )
        .unwrap();
    GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS])
}

/// 2026-09-25: Stand-in for a masked argmax: `<tool_call>` when the
/// grammar allows it, else the lowest allowed id.
fn masked_pick(gs: &mut GrammarState) -> u32 {
    gs.fill_bitmask();
    if gs.is_token_allowed(TOOL_CALL_OPEN) {
        return TOOL_CALL_OPEN;
    }
    (0..VOCAB_LEN)
        .find(|&t| gs.is_token_allowed(t))
        .expect("grammar allows at least one token")
}

fn allows(gs: &mut GrammarState, tok: u32) -> bool {
    gs.fill_bitmask();
    gs.is_token_allowed(tok)
}

/// 2026-09-25: Run `first_token_with` once with suppress set `[EOS]`;
/// returns (token, suppress ids the sampler saw, whether it got the
/// grammar).
fn run(policy: FirstTokenPolicy, gs: Option<&mut GrammarState>) -> (u32, Vec<u32>, bool) {
    let mut seen: Option<(Vec<u32>, bool)> = None;
    let tok = first_token_with(policy, &[EOS], gs, |ids, g| {
        seen = Some((ids.to_vec(), g.is_some()));
        Ok(match g {
            Some(g) => masked_pick(g),
            None => HELLO,
        })
    })
    .unwrap();
    let (ids, masked) = seen.expect("sampler runs exactly once");
    (tok, ids, masked)
}

fn thinking(tool_call_start: Option<u32>) -> FirstTokenPolicy {
    FirstTokenPolicy::for_birth(true, Some(THINK_END), tool_call_start)
}

fn direct(tool_call_start: Option<u32>) -> FirstTokenPolicy {
    FirstTokenPolicy::for_birth(false, Some(THINK_END), tool_call_start)
}

#[test]
fn policy_is_the_birth_predicate() {
    assert!(born_inside_thinking(true, Some(THINK_END)));
    assert!(
        !born_inside_thinking(true, None),
        "no </think> id = no thinking span"
    );
    assert!(!born_inside_thinking(false, Some(THINK_END)));
    assert!(thinking(None).grammar_suspended);
    assert!(!direct(None).grammar_suspended);
}

#[test]
fn required_tool_grammar_leaves_token0_alone_when_born_thinking() {
    let mut gs = required_tool_grammar();
    let (tok, ids, masked) = run(thinking(Some(TOOL_CALL_OPEN)), Some(&mut gs));
    assert!(!masked, "grammar must not mask token 0 inside <think>");
    assert!(
        ids.contains(&TOOL_CALL_OPEN),
        "<tool_call> suppressed in thinking"
    );
    assert!(ids.contains(&EOS), "caller's suppress set preserved");
    assert_eq!(tok, HELLO, "the model's free pick stands");
    assert_eq!(gs.num_history_steps(), 0, "grammar history must stay 0");
    assert!(allows(&mut gs, TOOL_CALL_OPEN), "grammar still pristine");
    assert!(!allows(&mut gs, HELLO));
}

#[test]
fn required_tool_grammar_still_forces_token0_when_born_direct() {
    let mut gs = required_tool_grammar();
    let (tok, ids, masked) = run(direct(Some(TOOL_CALL_OPEN)), Some(&mut gs));
    assert!(masked, "grammar masks token 0 outside <think>");
    assert_eq!(ids, vec![EOS], "no extra suppression outside <think>");
    assert_eq!(tok, TOOL_CALL_OPEN);
    assert!(gs.num_history_steps() >= 1, "matcher advanced past token 0");
    assert!(
        !allows(&mut gs, TOOL_CALL_OPEN),
        "cannot re-open after the opener"
    );
}

#[test]
fn json_schema_leaves_token0_alone_when_born_thinking() {
    let mut gs = json_schema_grammar();
    let opener = masked_pick(&mut gs);
    assert!(
        !allows(&mut gs, HELLO),
        "fixture: 'h' is schema-illegal first"
    );
    let (tok, _ids, masked) = run(thinking(None), Some(&mut gs));
    assert!(!masked);
    assert_eq!(tok, HELLO);
    assert_eq!(gs.num_history_steps(), 0);
    assert_eq!(masked_pick(&mut gs), opener, "schema still pristine");
    assert!(!allows(&mut gs, HELLO));
}

#[test]
fn json_schema_still_forces_token0_when_born_direct() {
    let mut gs = json_schema_grammar();
    let expected = masked_pick(&mut gs);
    let (tok, _ids, masked) = run(direct(None), Some(&mut gs));
    assert!(masked);
    assert_eq!(tok, expected);
    assert!(gs.num_history_steps() >= 1);
}

#[test]
fn no_grammar_path_is_a_pass_through_even_when_thinking() {
    let (tok, ids, masked) = run(thinking(Some(TOOL_CALL_OPEN)), None);
    assert!(!masked);
    assert_eq!(ids, vec![EOS]);
    assert_eq!(tok, HELLO);
}

#[test]
fn think_end_leaves_grammar_pristine_and_first_content_token_is_step_one() {
    // 2026-09-25: on the `emit_token` path, `</think>` does not advance the
    // matcher, and the next token is the grammar's first step.
    let (mut a, _rx) = test_seq(Vec::new(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.grammar_state = Some(required_tool_grammar());
    let sched = SchedCtx::for_test();

    emit_token(&mut a, THINK_END, None, &sched);
    assert!(
        !a.inside_thinking && a.think_ended,
        "</think> closes the span"
    );
    let gs = a.grammar_state.as_mut().expect("grammar survives </think>");
    assert_eq!(
        gs.num_history_steps(),
        0,
        "</think> must not advance the matcher"
    );
    assert!(
        allows(gs, TOOL_CALL_OPEN),
        "grammar pristine after </think>"
    );

    emit_token(&mut a, TOOL_CALL_OPEN, None, &sched);
    let gs = a
        .grammar_state
        .as_mut()
        .expect("first post-think token must not disengage the grammar");
    assert!(
        gs.num_history_steps() >= 1,
        "first content token is grammar step 1"
    );
    assert!(!a.finished);
}

/// 2026-10-15: `</tool_call>` in the toy vocab.
const TOOL_CALL_CLOSE: u32 = 129;

#[test]
fn stray_tool_call_end_joins_token0_suppression_only_where_the_mask_applies() {
    // 2026-10-15: flag on, no tools, no grammar: token 0 suppresses `</tool_call>`
    // beside the caller's ids.
    let on = direct(Some(TOOL_CALL_OPEN)).with_stray_tool_call_end(
        true,
        false,
        false,
        Some(TOOL_CALL_CLOSE),
    );
    assert_eq!(on.stray_tool_call_end, Some(TOOL_CALL_CLOSE));
    let (_, ids, masked) = run(on, None);
    assert_eq!(ids, vec![EOS, TOOL_CALL_CLOSE]);
    assert!(!masked);
    // 2026-10-15: flag off, or declared tools, or a grammar: unchanged.
    for (enabled, tools, grammar) in [
        (false, false, false),
        (true, true, false),
        (true, false, true),
    ] {
        let p = direct(Some(TOOL_CALL_OPEN)).with_stray_tool_call_end(
            enabled,
            tools,
            grammar,
            Some(TOOL_CALL_CLOSE),
        );
        assert_eq!(p, direct(Some(TOOL_CALL_OPEN)));
        let (_, ids, _) = run(p, None);
        assert_eq!(ids, vec![EOS]);
    }
}
