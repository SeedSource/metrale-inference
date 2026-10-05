// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: A95 on the spec-off decode commit
//! (`per_token::process_decoded_token`): after a token-0 `</think>` with
//! `METRALE_THINK_END_AT_TOKEN0` on, no answer token counts as thinking,
//! no `</think>` is forced, and EOS / `<|user|>` end the turn; with it off,
//! `<|user|>` is held back and becomes the next input token (the A103 feed,
//! seen in TEB TC-45). Fixtures: `scheduler/think_end_token0_tests.rs`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::Arc;

use super::per_token::process_decoded_token;
use crate::scheduler::confidence::THINK_DEFER_BUDGET_FACTOR;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::PreemptStubModel;
use crate::scheduler::think_end_token0_tests::{
    ANSWER, BUDGET, EOS_IDS, SITES, THINK_END, USER, VOCAB, WORD, born, pick, sched_with,
};
use crate::scheduler::types::ActiveSeq;

/// 2026-10-04: The stub model and a context over it (the lever is read
/// only at birth).
fn decode_ctx() -> (Arc<PreemptStubModel>, SchedCtx) {
    let model = Arc::new(PreemptStubModel::default());
    let sched = sched_with(model.clone(), true);
    (model, sched)
}

/// 2026-10-04: One spec-off decode step: the pipeline's pick for `want`,
/// committed through `process_decoded_token`.
fn decode(a: &mut ActiveSeq, sched: &SchedCtx, model: &PreemptStubModel, want: u32) -> u32 {
    let tok = pick(a, sched, want, VOCAB);
    let (te, ts) = (a.think_end_token, a.think_start_token);
    let (tc, tce) = (a.tool_call_start_token, a.tool_call_end_token);
    let now = std::time::Instant::now();
    process_decoded_token(a, tok, None, now, te, ts, None, tc, tce, model, sched);
    tok
}

#[test]
fn lever_on_decode_counts_no_thinking_forces_no_close_and_honours_eos() {
    let (model, sched) = decode_ctx();
    let window = 4 * BUDGET * THINK_DEFER_BUDGET_FACTOR;
    for site in SITES {
        for eos in EOS_IDS {
            let mut a = born(site, THINK_END, true, false);
            for w in WORD..WORD + window {
                let tok = decode(&mut a, &sched, &model, w);
                assert_ne!(tok, THINK_END, "{site:?}: forced </think> mid-answer");
            }
            assert!(!a.inside_thinking && !a.force_end_thinking, "{site:?}");
            assert_eq!(a.thinking_tokens, 0, "{site:?}: answer counted as thinking");
            assert!(!a.finished, "{site:?}");
            assert_eq!(decode(&mut a, &sched, &model, eos), eos);
            assert!(a.finished, "{site:?}: EOS {eos} after a token-0 close");
        }
    }
}

#[test]
fn lever_off_decode_holds_user_back_and_feeds_it_to_the_model() {
    let (model, sched) = decode_ctx();
    for site in SITES {
        let mut a = born(site, THINK_END, false, false);
        for w in ANSWER {
            decode(&mut a, &sched, &model, w);
        }
        assert_eq!(
            a.thinking_tokens,
            ANSWER.len() as u32,
            "{site:?}: old count"
        );
        assert_eq!(decode(&mut a, &sched, &model, USER), USER);
        assert!(
            !a.finished,
            "{site:?}: <|user|> ended the turn with the lever off"
        );
        assert!(!a.output_tokens.contains(&USER), "{site:?}: held back");
        assert_eq!(
            a.last_token, USER,
            "{site:?}: the held-back EOS is the next input"
        );
    }
}

/// 2026-10-04: The decode-only post-think EOS guard (16 output tokens past
/// the thinking ones, `per_token.rs`) holds a short answer's EOS on a tool
/// turn after ANY close, token 0 or later, so this fix does not lift it.
/// `tool_request` alone arms it here; a grammar or `require_tool_call`
/// would add their own holds.
#[test]
fn tool_turn_post_think_guard_treats_a_token0_close_like_a_later_one() {
    let (model, sched) = decode_ctx();
    for site in SITES {
        let mut token0 = born(site, THINK_END, true, false);
        let mut later = born(site, WORD, true, false);
        assert_eq!(decode(&mut later, &sched, &model, THINK_END), THINK_END);
        for a in [&mut token0, &mut later] {
            a.tool_request = true;
            for w in ANSWER {
                decode(a, &sched, &model, w);
            }
            decode(a, &sched, &model, USER);
            assert!(!a.finished, "{site:?}: guard expected to hold this EOS");
            for w in WORD + 100..WORD + 108 {
                decode(a, &sched, &model, w);
            }
            decode(a, &sched, &model, USER);
            assert!(a.finished, "{site:?}: guard expected to release this EOS");
        }
    }
}
