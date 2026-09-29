// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Tests for the plain-chat `</tool_call>` hard stop on the
//! speculative emit path (`emit_step::emit_token`), A143 part B.
//!
//! The serial decode path (`decode_logits_step/per_token.rs`) ends the turn
//! when a request with no tools declared and no active grammar emits
//! `</tool_call>`. `emit_token`, driven by MTP and the K2/K3/K4/DFlash
//! verify-accept steps, had no twin, so a spec-on turn kept decoding past
//! the call that spec-off stops at. These tests drive the real
//! `emit_token` and check both sides of the #192 distinction:
//!  * no tools declared, no grammar: `</tool_call>` finishes the turn;
//!  * tools declared (`tools_present`): the turn survives, so the model can
//!    emit parallel calls.
//!
//! A third test checks the call-site shape every verify-accept caller uses:
//! `emit_token(...)` then `if a.finished { break }` over one verify window's
//! accepted tokens. Driving the step functions end to end would need
//! draft-proposal, graph-verify and DFlash plumbing on a stub `Model`, which
//! `test_support` does not provide; its fixtures stop at `emit_token`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::emit_step::emit_token;
use super::sched_ctx::SchedCtx;
use super::test_support::test_seq;
use super::types::ActiveSeq;

/// 2026-09-29: a live sequence with no grammar and no tools declared, the
/// plain-chat shape the guard targets. Seven content tokens are already
/// emitted, so the fixture's `min_tokens` (7) is not in play.
fn plain_chat_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq((1000..1007).collect(), 50, None, 10);
    a.finished = false;
    a.inside_thinking = false;
    debug_assert!(a.grammar_state.is_none());
    debug_assert!(!a.tools_present);
    a
}

#[test]
fn plain_chat_tool_call_end_finishes_the_sequence() {
    // 2026-09-29: fails if the emit path lacks the decode twin: a no-tools
    // turn would keep decoding past `</tool_call>` with spec on.
    let sched = SchedCtx::for_test();
    let mut a = plain_chat_seq();
    let end_tok = a
        .tool_call_end_token
        .expect("fixture sets tool_call_end_token");
    emit_token(&mut a, end_tok, None, &sched);
    assert!(
        a.finished,
        "a no-tools, no-grammar </tool_call> must hard-stop the turn on the \
         speculative emit path, as decode_logits_step does"
    );
    assert!(
        a.tool_call_completed,
        "the completion flag must still be set"
    );
    // 2026-09-29: `</tool_call>` itself is emitted and nothing after it:
    // `emit_token` falls through to its push rather than returning early.
    assert_eq!(
        a.output_tokens.last(),
        Some(&end_tok),
        "</tool_call> must still be pushed to output_tokens before the turn ends"
    );
}

#[test]
fn tools_present_tool_call_end_does_not_finish_the_sequence() {
    // 2026-09-29: #192: a request that declared tools survives a closed call
    // so the model can emit parallel calls; the turn ends at EOS or a
    // watchdog, never at this guard.
    let sched = SchedCtx::for_test();
    let mut a = plain_chat_seq();
    a.tools_present = true;
    let end_tok = a
        .tool_call_end_token
        .expect("fixture sets tool_call_end_token");
    emit_token(&mut a, end_tok, None, &sched);
    assert!(
        !a.finished,
        "tools_present must not hard-stop at </tool_call> (#192 parallel calls)"
    );
    assert!(a.tool_call_completed);
    assert_eq!(a.output_tokens.last(), Some(&end_tok));
}

#[test]
fn finished_stops_the_accepted_token_loop_mid_window() {
    // 2026-09-29: the verify-accept call-site pattern:
    //   for tok in accepted_window { emit_token(a, tok, ..); if a.finished { break; } }
    // A window whose first accepted token is `</tool_call>` on a no-tools
    // turn must emit that one token and never reach the other two.
    let sched = SchedCtx::for_test();
    let mut a = plain_chat_seq();
    let end_tok = a
        .tool_call_end_token
        .expect("fixture sets tool_call_end_token");
    let window = [end_tok, 2000, 2001];
    let mut calls = 0usize;
    for tok in window {
        calls += 1;
        emit_token(&mut a, tok, None, &sched);
        if a.finished {
            break;
        }
    }
    assert_eq!(
        calls, 1,
        "the loop must stop after </tool_call>; the window's remaining \
         accepted drafts must never reach emit_token"
    );
    assert_eq!(
        a.output_tokens.last(),
        Some(&end_tok),
        "output_tokens must end at </tool_call>, not run on into 2000/2001"
    );
    assert!(a.finished);
}
