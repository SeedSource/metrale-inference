// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `step_mtp` on a DFlash serve under the batched GLM pin
//! (`METRALE_GLM_DFLASH_BATCHED_VERIFY`): at C=2 and C=4 consecutive steps stay batched when
//! every row is grammarless and has drafts, thinking rows included (their verdict goes through
//! the host pipeline at their own rows), and a row just verified per sequence at a different
//! width rejoins the batch the next step. Without the pin (and under the per-sequence control
//! arm) a thinking row keeps the per-sequence verify, as before.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

mod fake;

use std::sync::Arc;

use super::levers::SchedLevers;
use super::mtp_step::step_mtp;
use super::sched_ctx::SchedCtx;
use super::test_support::{RespRx, active_seq};
use super::think_end_token0_tests::ctx;
use super::types::ActiveSeq;
use fake::StepFake;

/// 2026-10-09: The serve-wide draft count of a `--dflash-gamma 8` serve.
const SERVE_DRAFTS: usize = 7;
const THINK_END: u32 = 7001;
const THINK_START: u32 = 7002;

/// 2026-10-09: The rung state of the three arms.
#[derive(Clone, Copy, Debug)]
enum Arm {
    /// 2026-10-09: No GLM lever.
    Off,
    /// 2026-10-09: `METRALE_GLM_DFLASH_PERSEQ_K4`: the pin without the batched mark.
    PerSeqControl,
    /// 2026-10-09: `METRALE_GLM_DFLASH_BATCHED_VERIFY`.
    Batched,
}

fn sched(model: Arc<StepFake>, arm: Arm) -> SchedCtx {
    let mut s = SchedCtx::for_test_with(model);
    let mut levers = SchedLevers::defaults();
    levers.dflash_unified_ctx = true;
    s.levers = Arc::new(levers);
    match arm {
        Arm::Off => {}
        Arm::PerSeqControl => s.dflash_rung.pin_multi(4),
        Arm::Batched => {
            s.dflash_rung.pin_multi(4);
            s.dflash_rung.mark_multi_batched();
        }
    }
    s
}

/// 2026-10-09: An unfinished grammarless row with `nd` fresh drafts, inside `<think>` when
/// `thinking`. The receiver keeps the sink open.
fn row(m: &StepFake, slot: usize, nd: usize, thinking: bool) -> (ActiveSeq, RespRx) {
    let (mut a, rx) = active_seq(slot, 3);
    a.remaining = 1000;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.inside_thinking = thinking;
    a.enable_thinking = thinking;
    a.pending_drafts = m.fresh(nd);
    (a, rx)
}

fn step(m: &StepFake, s: &SchedCtx, active: &mut [ActiveSeq], num_drafts: usize) -> Vec<String> {
    let c = ctx(s, &active[0]);
    step_mtp(m, active, s, num_drafts, &c, true);
    m.take_trace()
}

fn count(t: &[String], prefix: &str) -> usize {
    t.iter().filter(|l| l.starts_with(prefix)).count()
}

#[test]
fn batched_pin_keeps_c2_and_c4_batched_with_thinking_rows() {
    for n in [2usize, 4] {
        let m = Arc::new(StepFake::new());
        let s = sched(m.clone(), Arm::Batched);
        // 2026-10-09: Even rows are inside `<think>` (the A146 rows the old split sent serial).
        let (mut active, _rx): (Vec<ActiveSeq>, Vec<RespRx>) =
            (0..n).map(|i| row(&m, i, 3, i % 2 == 0)).unzip();
        for t in 0..4 {
            let thinking = active.iter().filter(|a| a.inside_thinking).count();
            let tr = step(&m, &s, &mut active, SERVE_DRAFTS);
            let c = format!("C={n} step {t}: {tr:#?}");
            assert_eq!(
                tr.iter().filter(|l| l.starts_with("batched")).count(),
                1,
                "{c}"
            );
            assert!(tr.contains(&format!("batched ks={:?}", vec![4; n])), "{c}");
            assert_eq!(count(&tr, "perseq"), 0, "no per-sequence fallback: {c}");
            // 2026-10-09: Each thinking row is judged through the host pipeline at its own rows.
            assert_eq!(count(&tr, "d2h"), thinking, "{c}");
            assert!(active.iter().all(|a| !a.finished), "{c}");
            assert!(
                active.iter().all(|a| a.pending_drafts.len() == 3),
                "every row proposes at the rung width: {c}"
            );
        }
        // 2026-10-09: Accept-all fake: 4 tokens per row per step on top of the 3 seeded.
        assert!(
            active.iter().all(|a| a.output_tokens.len() == 3 + 16),
            "C={n}"
        );
    }
}

#[test]
fn row_verified_per_sequence_rejoins_the_batch_next_step() {
    let m = Arc::new(StepFake::new());
    let s = sched(m.clone(), Arm::Batched);
    // 2026-10-09: C=1: the lone row verifies per sequence at the serve's K=8 and proposes 7.
    let (b, _rb) = row(&m, 1, SERVE_DRAFTS, false);
    let mut active = vec![b];
    let tr = step(&m, &s, &mut active, SERVE_DRAFTS);
    assert_eq!(tr, ["perseq k=8"], "{tr:#?}");
    assert_eq!(active[0].pending_drafts.len(), SERVE_DRAFTS);
    // 2026-10-09: C=2: a row with the rung's 3 drafts arrives first. The 7-draft block is cut
    // to the rung width and batched, not verified alone at K=8.
    let (a, _ra) = row(&m, 0, 3, true);
    active.insert(0, a);
    for t in 0..3 {
        let tr = step(&m, &s, &mut active, SERVE_DRAFTS);
        assert_eq!(count(&tr, "batched ks=[4, 4]"), 1, "step {t}: {tr:#?}");
        assert_eq!(count(&tr, "perseq"), 0, "step {t}: {tr:#?}");
    }
}

#[test]
fn without_the_batched_mark_a_thinking_row_stays_per_sequence() {
    for arm in [Arm::Off, Arm::PerSeqControl] {
        let m = Arc::new(StepFake::new());
        let s = sched(m.clone(), arm);
        let (mut active, _rx): (Vec<ActiveSeq>, Vec<RespRx>) =
            (0..3).map(|i| row(&m, i, 3, i == 0)).unzip();
        // 2026-10-09: Unpinned the rung returns the serve's count unchanged, so pass the
        // 3-draft width the rows carry.
        let nd = if matches!(arm, Arm::Off) {
            3
        } else {
            SERVE_DRAFTS
        };
        let tr = step(&m, &s, &mut active, nd);
        let c = format!("{arm:?}: {tr:#?}");
        assert!(tr.contains(&"batched ks=[4, 4]".to_string()), "{c}");
        assert_eq!(count(&tr, "perseq k=4"), 1, "the thinking row: {c}");
        // 2026-10-09: The two batched rows keep the raw verdict; only the per-sequence
        // thinking row reads its logits (row 0 of its own forward).
        assert_eq!(count(&tr, "d2h"), 1, "{c}");
        assert!(tr.contains(&"d2h row=0 n=4".to_string()), "{c}");
    }
}
