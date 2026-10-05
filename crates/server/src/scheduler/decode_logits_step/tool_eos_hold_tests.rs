// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: A103 on spec-off decode: the post-think tool-turn EOS hold
//! drops a short answer's stop token and feeds it back as the next input;
//! `METRALE_TOOL_EOS_HOLD_MASK=1` masks the stop tokens before sampling
//! instead (`logit_processors::tool_eos_hold`). The steps run the production
//! decode pick (`process_seq_logits`) and commit (`process_decoded_token`).
//! Fixtures: `scheduler/think_end_token0_tests.rs`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::Arc;

use super::per_token::process_decoded_token;
use crate::scheduler::decode_logits_seq::process_seq_logits;
use crate::scheduler::levers::SchedLevers;
use crate::scheduler::logit_processors::process_position_logits;
use crate::scheduler::logit_processors::tool_eos_hold::{POST_THINK_MIN_CONTENT, mask_stops};
use crate::scheduler::sample_step::{PositionKind, penalty_params_for};
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::PreemptStubModel;
use crate::scheduler::think_end_token0_tests::{
    ANSWER, EOS_IDS, SITES, Site, THINK_END, USER, VOCAB, WORD, born, ctx, sched_with,
};
use crate::scheduler::types::ActiveSeq;

/// 2026-10-04: A context with the mask lever as given, the rest at defaults.
fn sched(mask: bool) -> (Arc<PreemptStubModel>, SchedCtx) {
    let model = Arc::new(PreemptStubModel::default());
    let mut sched = sched_with(model.clone(), true);
    let mut levers = SchedLevers::defaults();
    levers.tool_eos_hold_mask = mask;
    sched.levers = Arc::new(levers);
    (model, sched)
}

/// 2026-10-04: A turn with tools declared (`tool_request`), after a token-0
/// `</think>`: the TEB TC-45 shape once the tool result is in.
fn tool_turn(site: Site) -> ActiveSeq {
    let mut a = born(site, THINK_END, true, false);
    a.tool_request = true;
    a
}

/// 2026-10-04: Commit `tok` through the spec-off decode commit.
fn commit(a: &mut ActiveSeq, sched: &SchedCtx, model: &PreemptStubModel, tok: u32) {
    let (te, ts) = (a.think_end_token, a.think_start_token);
    let (tc, tce) = (a.tool_call_start_token, a.tool_call_end_token);
    let now = std::time::Instant::now();
    process_decoded_token(a, tok, None, now, te, ts, None, tc, tce, model, sched);
}

/// 2026-10-04: One decode step over BF16 host logits that prefer `want`
/// (10.0), then `next` (5.0). Returns the token decode picked.
fn decode(
    a: &mut ActiveSeq,
    sched: &SchedCtx,
    model: &PreemptStubModel,
    want: u32,
    next: u32,
) -> u32 {
    let mut row = vec![0.0f32; VOCAB];
    row[want as usize] = 10.0;
    row[next as usize] = 5.0;
    let buf: Vec<u8> = row
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let c = ctx(sched, a);
    let (tok, _) = process_seq_logits(model, a, &buf, 0, VOCAB, 2, false, &c, false);
    commit(a, sched, model, tok);
    tok
}

/// 2026-10-04: Output tokens past the thinking ones, as the hold counts them.
fn content(a: &ActiveSeq) -> u32 {
    (a.output_tokens.len() as u32).saturating_sub(a.thinking_tokens)
}

#[test]
fn lever_off_the_hold_drops_a_short_answers_stop_and_feeds_it_back() {
    let (model, sched) = sched(false);
    for site in SITES {
        let mut a = tool_turn(site);
        for w in ANSWER {
            decode(&mut a, &sched, &model, w, WORD);
        }
        assert_eq!(decode(&mut a, &sched, &model, USER, WORD), USER, "{site:?}");
        assert!(
            !a.finished,
            "{site:?}: the hold expected to keep the turn open"
        );
        assert!(!a.output_tokens.contains(&USER), "{site:?}: dropped");
        assert_eq!(
            a.last_token, USER,
            "{site:?}: and fed back as the next input"
        );
    }
}

#[test]
fn lever_on_the_mask_picks_the_runner_up_until_the_window_closes() {
    let (model, sched) = sched(true);
    for site in SITES {
        let mut a = tool_turn(site);
        for w in ANSWER {
            decode(&mut a, &sched, &model, w, WORD);
        }
        for (k, eos) in (0u32..).zip(EOS_IDS) {
            let tok = decode(&mut a, &sched, &model, eos, WORD + k);
            assert_eq!(tok, WORD + k, "{site:?}: stop {eos} masked under the hold");
            assert_eq!(
                a.last_token, tok,
                "{site:?}: only the committed token is fed"
            );
            assert_eq!(a.output_tokens.last(), Some(&tok), "{site:?}");
        }
        let mut filler = WORD + 100;
        while content(&a) < POST_THINK_MIN_CONTENT {
            assert_eq!(decode(&mut a, &sched, &model, USER, filler), filler);
            filler += 1;
        }
        assert!(!a.finished, "{site:?}");
        assert_eq!(decode(&mut a, &sched, &model, USER, WORD), USER, "{site:?}");
        assert!(
            a.finished,
            "{site:?}: past the window the stop ends the turn"
        );
    }
}

#[test]
fn lever_on_rows_under_the_hold_read_back_host_logits() {
    for mask in [false, true] {
        let (_model, sched) = sched(mask);
        let mut a = tool_turn(Site::PrefillRequest);
        let device = |a: &ActiveSeq| super::argmax_readback_eligible(std::iter::once(a), &sched);
        assert_eq!(device(&a), !mask, "mask={mask}: under the hold");
        a.output_tokens.extend(WORD..WORD + POST_THINK_MIN_CONTENT);
        assert!(device(&a), "mask={mask}: past the window");
    }
}

#[test]
fn the_mask_is_decode_only_and_a_stop_reaching_the_commit_ends_the_turn() {
    let (model, sched) = sched(true);
    let mut a = tool_turn(Site::PrefillRequest);
    let c = ctx(&sched, &a);
    let floor = sched.watchdog.min_reasoning_floor;
    for (kind, masked) in [
        (PositionKind::FinalDecode, true),
        (PositionKind::Verify, false),
    ] {
        let mut logits = vec![0.0f32; VOCAB];
        logits[USER as usize] = 10.0;
        let p = penalty_params_for(&a, kind, 0.0, None, Vec::new(), floor);
        assert_eq!(
            process_position_logits(&mut logits, &mut a, &c, &p, kind),
            None
        );
        let gone = logits[USER as usize] == f32::NEG_INFINITY;
        assert_eq!(gone, masked, "{kind:?}: the speculative paths have no hold");
    }
    // 2026-10-04: a stop that is the only live token (a grammar that allows
    // only a stop) is left alone.
    let mut lone = vec![f32::NEG_INFINITY; VOCAB];
    lone[USER as usize] = 1.0;
    assert!(!mask_stops(&mut lone, &a));
    assert_eq!(lone[USER as usize], 1.0);
    // 2026-10-04: with the mask on, the post-sample hold is off, so such a
    // stop ends the turn instead of being fed back.
    commit(&mut a, &sched, &model, USER);
    assert!(a.finished, "the stop ends the turn");
}

#[test]
fn the_mask_ships_off_and_one_turns_it_on_in_the_live_resolver() {
    const VAR: &str = "METRALE_TOOL_EOS_HOLD_MASK";
    assert!(!SchedLevers::defaults().tool_eos_hold_mask);
    // 2026-10-04: SAFETY: nothing else in this test binary writes this
    // variable. `cargo test` runs tests on parallel threads, so a
    // concurrent environment access from another test is not excluded.
    let resolve = |v: Option<&str>| {
        unsafe {
            match v {
                Some(v) => std::env::set_var(VAR, v),
                None => std::env::remove_var(VAR),
            }
        }
        SchedLevers::from_env(None).tool_eos_hold_mask
    };
    let (unset, zero, one) = (resolve(None), resolve(Some("0")), resolve(Some("1")));
    resolve(None);
    assert!(!unset && !zero, "ships off");
    assert!(one, "=1 turns it on");
}
