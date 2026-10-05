// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: A95 follow-up: the MTP bootstrap picks the first token after a
//! `</think>` through decode's full logits pipeline
//! (`verify_pipeline_helper::bootstrap_takes_pipeline`), so the `<tool_call>`
//! pin, the post-close think mask and the `min_tokens` EOS mask act on a
//! token-0 close's first answer token as on a later close's. A token-0 close
//! always reaches that token here: a newborn row has no drafts.
//!
//! These call the production bootstraps (`serial_bootstrap::bootstrap_seq`,
//! `step_mtp_bootstrap_batched`) over a `PreemptStubModel` that decodes to
//! host-readable logits (`logit_rows`). Fixtures:
//! `scheduler/think_end_token0_tests.rs`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::Arc;

use super::serial_bootstrap::bootstrap_seq;
use crate::scheduler::emit_step::emit_token;
use crate::scheduler::mtp_bootstrap_step::step_mtp_bootstrap_batched;
use crate::scheduler::test_support::PreemptStubModel;
use crate::scheduler::think_end_token0_tests::{
    EOS_IDS, SITES, Site, THINK_END, THINK_START, TOOL_CALL, VOCAB, WORD, born, ctx, sched_with,
};
use crate::scheduler::types::ActiveSeq;

/// 2026-10-04: The production bootstrap under test.
#[derive(Clone, Copy, Debug)]
enum Boot {
    Serial,
    Batched,
}

/// 2026-10-04: The token the batched run's plain row prefers.
const CONTROL: u32 = WORD + 60;

/// 2026-10-04: Logits that prefer `want` (10.0), then `next` (5.0).
fn prefer(want: u32, next: u32) -> Vec<f32> {
    let mut row = vec![0.0f32; VOCAB];
    row[want as usize] = 10.0;
    row[next as usize] = 5.0;
    row
}

/// 2026-10-04: A row whose thinking just closed, at token 0 (A95) or after
/// one thinking token (a later close), optionally with a forced tool call
/// and no grammar (`require_tool_call`).
fn closed(site: Site, token0: bool, legacy_tool: bool) -> ActiveSeq {
    if token0 {
        return born(site, THINK_END, true, legacy_tool);
    }
    let mut a = born(site, WORD, true, legacy_tool);
    let sched = sched_with(Arc::new(PreemptStubModel::default()), true);
    emit_token(&mut a, THINK_END, None, &sched);
    assert!(
        a.think_just_ended && !a.inside_thinking,
        "{site:?}: fixture"
    );
    a
}

/// 2026-10-04: A row past its first answer token, which stays on the
/// penalties-only sampler.
fn content_row() -> ActiveSeq {
    let mut a = closed(Site::PrefillRequest, true, false);
    let sched = sched_with(Arc::new(PreemptStubModel::default()), true);
    emit_token(&mut a, WORD + 50, None, &sched);
    assert!(!a.think_just_ended && !a.inside_thinking, "fixture");
    a
}

/// 2026-10-04: One bootstrap step for `a` over logits that prefer `want`,
/// then `next`, with the lever as given. Returns the row and the token it
/// committed (`last_token`). `Batched` runs a plain row beside it, which
/// must keep its own argmax.
fn boot(kind: Boot, a: ActiveSeq, want: u32, next: u32, lever: bool) -> (ActiveSeq, u32) {
    let mut rows = vec![prefer(want, next)];
    let mut active = vec![a];
    if let Boot::Batched = kind {
        rows.push(prefer(CONTROL, CONTROL + 1));
        active.push(content_row());
    }
    let model = Arc::new(PreemptStubModel {
        logit_rows: rows,
        ..Default::default()
    });
    let sched = sched_with(model.clone(), lever);
    let c = ctx(&sched, &active[0]);
    match kind {
        Boot::Serial => bootstrap_seq(&*model, &mut active[0], &sched, 3, 3, &c, false),
        Boot::Batched => step_mtp_bootstrap_batched(&*model, &mut active, &sched, &[0, 1], 3, &c),
    }
    if let Boot::Batched = kind {
        let plain = active.pop().expect("plain row");
        assert_eq!(plain.last_token, CONTROL, "the plain row keeps the sampler");
    }
    let a = active.pop().expect("row under test");
    let tok = a.last_token;
    (a, tok)
}

#[test]
fn both_bootstraps_give_the_first_token_after_a_close_the_decode_masks() {
    // 2026-10-04: (stage, forced tool call, min_tokens, preferred, runner-up,
    // expected). Before the fix the bootstrap took the preferred token.
    let cases = [
        ("pin", true, 0, WORD, WORD + 1, TOOL_CALL),
        ("<think> mask", false, 0, THINK_START, WORD, WORD),
        ("</think> mask", false, 0, THINK_END, WORD, WORD),
        ("min_tokens EOS mask", false, 4, EOS_IDS[1], WORD, WORD),
    ];
    for (stage, tool, min, want, next, expect) in cases {
        for kind in [Boot::Serial, Boot::Batched] {
            for site in SITES {
                for token0 in [true, false] {
                    let mut a = closed(site, token0, tool);
                    a.min_tokens = min;
                    let (a, tok) = boot(kind, a, want, next, true);
                    let at = format!("{stage}/{kind:?}/{site:?}/token0={token0}");
                    assert_eq!(tok, expect, "{at}");
                    assert_eq!(a.output_tokens.last(), Some(&expect), "{at}");
                    assert!(!a.inside_thinking && !a.finished, "{at}");
                    assert!(!a.think_just_ended, "{at}: the one-shot is spent");
                }
            }
        }
    }
}

#[test]
fn lever_off_keeps_the_old_bootstrap_pick_after_a_close() {
    for kind in [Boot::Serial, Boot::Batched] {
        for site in SITES {
            // 2026-10-04: a later close: the penalties-only sampler, no pin.
            let (a, tok) = boot(kind, closed(site, false, true), WORD, WORD + 1, false);
            assert_eq!(tok, WORD, "{kind:?}/{site:?}: later close, no pin");
            assert!(a.require_tool_call, "{kind:?}/{site:?}");
            // 2026-10-04: a token-0 close stays "thinking", where the pin
            // does not act and `<tool_call>` is masked.
            let off = born(site, THINK_END, false, true);
            let (a, tok) = boot(kind, off, WORD, WORD + 1, false);
            assert_eq!(tok, WORD, "{kind:?}/{site:?}: token-0 close, no pin");
            assert!(
                a.inside_thinking && a.require_tool_call,
                "{kind:?}/{site:?}"
            );
        }
    }
}
