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
    EOS_IDS, SITES, Site, THINK_END, THINK_START, TOOL_CALL, TOOL_CALL_END, VOCAB, WORD, born, ctx,
    sched_with,
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
        Boot::Serial => {
            let refusal = bootstrap_seq(&*model, &mut active[0], &sched, 3, 3, &c, false);
            assert!(refusal.is_none(), "no KV refusal in this test: {refusal:?}");
        }
        Boot::Batched => {
            let refusal = step_mtp_bootstrap_batched(&*model, &mut active, &sched, &[0, 1], 3, &c);
            assert!(refusal.is_none(), "no KV refusal in this test: {refusal:?}");
        }
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

/// 2026-10-14: kvagcheck9: a rank-agreed KV refusal in the serial bootstrap's
/// decode ended the reply as HTTP 200 "stop" at 178 tokens (a bare
/// `finished`). The row must carry an error, which `finish_sequence` turns
/// into an error frame. The stub with no `logit_rows` fails every decode.
#[test]
fn a_failed_serial_bootstrap_decode_ends_the_row_with_an_error_not_a_stop() {
    let mut a = content_row();
    let model = Arc::new(PreemptStubModel::default());
    let sched = sched_with(model.clone(), true);
    let c = ctx(&sched, &a);
    let refusal = bootstrap_seq(&*model, &mut a, &sched, 3, 3, &c, false);
    assert!(refusal.is_none(), "not a KV refusal: {refusal:?}");
    assert!(a.finished, "the row ends");
    let err = a.error.as_deref().expect("an error, not a normal finish");
    assert!(err.starts_with("bootstrap decode:"), "{err}");
}

/// 2026-10-16: the same for the stray `</tool_call>` re-pick and the pipeline pick: the
/// decode succeeds, the device argmax is the masked `</tool_call>` (no tools, flag
/// on), and the host read the re-pick needs fails. A bare `finished` here was an
/// HTTP 200 "stop" with no error (A184 class).
fn failing_copy_boot(kind: Boot, a: ActiveSeq, want: u32) -> ActiveSeq {
    let mut rows = vec![prefer(want, WORD)];
    let mut active = vec![a];
    if let Boot::Batched = kind {
        rows.push(prefer(CONTROL, CONTROL + 1));
        active.push(content_row());
    }
    let model = Arc::new(PreemptStubModel {
        logit_rows: rows,
        fail_logit_copy: true,
        ..Default::default()
    });
    let mut sched = sched_with(model.clone(), true);
    sched.watchdog.mask_stray_tool_call_end = true;
    let c = ctx(&sched, &active[0]);
    match kind {
        Boot::Serial => {
            let refusal = bootstrap_seq(&*model, &mut active[0], &sched, 3, 3, &c, false);
            assert!(refusal.is_none(), "no KV refusal in this test: {refusal:?}");
        }
        Boot::Batched => {
            let refusal = step_mtp_bootstrap_batched(&*model, &mut active, &sched, &[0, 1], 3, &c);
            assert!(refusal.is_none(), "no KV refusal in this test: {refusal:?}");
        }
    }
    active.swap_remove(0)
}

#[test]
fn a_failed_stray_tool_call_end_repick_d2h_ends_the_row_with_an_error_not_a_stop() {
    for (kind, prefix) in [
        (
            Boot::Serial,
            "bootstrap stray </tool_call> re-pick: D2H failed",
        ),
        (
            Boot::Batched,
            "batched bootstrap stray </tool_call> re-pick: D2H failed",
        ),
    ] {
        let a = failing_copy_boot(kind, content_row(), TOOL_CALL_END);
        let at = format!("{kind:?}");
        assert!(a.finished, "{at}: the row ends");
        let err = a.error.as_deref().expect("an error, not a normal finish");
        assert!(err.starts_with(prefix), "{at}: {err}");
    }
}

#[test]
fn a_failed_batched_bootstrap_pipeline_pick_d2h_ends_the_row_with_an_error_not_a_stop() {
    let a = failing_copy_boot(
        Boot::Batched,
        closed(Site::PrefillRequest, false, false),
        WORD,
    );
    assert!(a.finished, "the row ends");
    let err = a.error.as_deref().expect("an error, not a normal finish");
    assert!(
        err.starts_with("batched bootstrap pipeline pick: D2H failed"),
        "{err}"
    );
}

/// 2026-10-15 (race-memory, CAPACITY.md gap b): a retryable KV refusal of the serial
/// bootstrap's decode is handed back so `step_mtp`'s caller pauses a victim, as for a batched
/// bootstrap; the row stays live and unchanged and retries next step. On a 2-rank serve only
/// a rank-agreed refusal is retryable; any other ends the row with an error.
#[test]
fn a_kv_refused_serial_bootstrap_decode_is_handed_back_not_failed() {
    for (ep, msg, retry) in [
        (false, "KV cache exhausted: no free blocks", true),
        (
            true,
            "KV cache exhausted: decode/verify step of slot 0 through KV block 256: rank(s) [1] \
             refused this step's KV/lazy-map admission, so every rank refuses it (rank-agreed)",
            true,
        ),
        (true, "KV cache exhausted: no free blocks", false),
    ] {
        let mut a = content_row();
        let last = a.last_token;
        let n_out = a.output_tokens.len();
        let model = Arc::new(PreemptStubModel {
            ep,
            hard_error: Some(msg),
            ..Default::default()
        });
        let sched = sched_with(model.clone(), true);
        let c = ctx(&sched, &a);
        let refusal = bootstrap_seq(&*model, &mut a, &sched, 3, 3, &c, false);
        if retry {
            assert!(
                refusal.as_deref().is_some_and(|m| m.contains(msg)),
                "{ep}: {refusal:?}"
            );
            assert!(!a.finished && a.error.is_none(), "{ep}: the row stays live");
            assert_eq!(
                (a.last_token, a.output_tokens.len()),
                (last, n_out),
                "{ep}: unchanged"
            );
        } else {
            assert!(
                refusal.is_none(),
                "an unvoted 2-rank refusal is not retryable"
            );
            assert!(
                a.finished && a.error.is_some(),
                "it ends the row with an error"
            );
        }
    }
}
