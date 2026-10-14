// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for decode-time KV preemption and resume
//! (`preempt.rs`, `choose_decode_victim`), run against `PreemptStubModel`.
//! They check each request's response channel as the API layer sees it.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::io::SchedIo;
use super::preempt::{
    PREEMPT_IMMUNITY_TOKENS, choose_decode_victim, decode_batch_with_preemption, preempt_requeue,
    resume_preempted_seq, resume_preempted_seqs,
};
use super::sched_ctx::SchedCtx;
use super::test_support::{PreemptStubModel, active_seq, streaming_seq};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn kv_exhaustion_requeues_least_progress_victim_and_sends_nothing() {
    let model = std::sync::Arc::new(PreemptStubModel::failing(1));
    let (a0, _rx0) = active_seq(0, 5);
    let (victim, mut victim_rx) = streaming_seq(1, 2);
    let (a2, _rx2) = active_seq(2, 9);
    let mut active = vec![a0, victim, a2];
    let mut swapped = Vec::new();
    let mut preempted = Vec::new();

    let logits = decode_batch_with_preemption(
        &SchedCtx::for_test_with(model.clone()),
        &mut active,
        None,
        &mut swapped,
        &mut preempted,
        &mut Vec::new(),
    );

    assert!(logits.is_some());
    assert_eq!(model.decode_calls.load(Ordering::SeqCst), 2);
    assert_eq!(active.len(), 2);
    assert_eq!(preempted.len(), 1);
    assert!(swapped.is_empty());
    assert_eq!(preempted[0].a.output_tokens.len(), 2);
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![1]);
    assert_eq!(model.cached_seqs.load(Ordering::SeqCst), 1);
    assert!(matches!(
        victim_rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    assert_eq!(preempted[0].tokens, vec![1, 2, 3, 4, 100]);
    assert_eq!(preempted[0].a.last_token, 101);
}

#[test]
fn non_kv_error_still_fails_the_whole_batch() {
    let model = std::sync::Arc::new(PreemptStubModel {
        hard_error: Some("CUDA error 700: illegal memory access"),
        ..Default::default()
    });
    let (a0, mut rx0) = active_seq(0, 3);
    let (a1, mut rx1) = active_seq(1, 4);
    let mut active = vec![a0, a1];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    let logits = decode_batch_with_preemption(
        &SchedCtx::for_test_with(model.clone()),
        &mut active,
        None,
        &mut swapped,
        &mut preempted,
        &mut Vec::new(),
    );
    assert!(logits.is_none());
    assert!(active.is_empty() && preempted.is_empty() && swapped.is_empty());
    assert!(rx0.try_recv().expect("response sent").is_err());
    assert!(rx1.try_recv().expect("response sent").is_err());
}

#[test]
fn single_sequence_exhaustion_is_not_preemptible() {
    // 2026-09-25: The retry loop preempts only when more than one sequence
    // is active, so a lone sequence's exhaustion is terminal.
    let model = std::sync::Arc::new(PreemptStubModel {
        hard_error: Some("KV cache exhausted: no free blocks"),
        ..Default::default()
    });
    let (a0, mut rx0) = active_seq(0, 3);
    let mut active = vec![a0];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    let logits = decode_batch_with_preemption(
        &SchedCtx::for_test_with(model.clone()),
        &mut active,
        None,
        &mut swapped,
        &mut preempted,
        &mut Vec::new(),
    );
    assert!(logits.is_none());
    assert!(preempted.is_empty());
    assert!(rx0.try_recv().expect("response sent").is_err());
}

#[test]
fn victim_policy_least_progress_wins() {
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (a0, _r0) = active_seq(0, 7);
    let (a1, _r1) = active_seq(1, 3);
    let (a2, _r2) = active_seq(2, 12);
    let active = vec![a0, a1, a2];
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(1));
}

#[test]
fn victim_policy_starvation_guard_skips_resumed_until_progress() {
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (mut a0, _r0) = active_seq(0, 3);
    a0.preempt_immune_until_tokens = a0.output_tokens.len() + PREEMPT_IMMUNITY_TOKENS;
    let (a1, _r1) = active_seq(1, 9);
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(1));

    let (mut a0, _r0) = active_seq(0, 3 + PREEMPT_IMMUNITY_TOKENS);
    a0.preempt_immune_until_tokens = 3 + PREEMPT_IMMUNITY_TOKENS;
    let (a1, _r1) = active_seq(1, 200);
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(0));
}

#[test]
fn victim_policy_all_immune_still_yields_a_victim() {
    // 2026-09-25: With every candidate immune, one is still chosen, so
    // immunity never turns a recoverable exhaustion into a batch-wide error.
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (mut a0, _r0) = active_seq(0, 4);
    a0.preempt_immune_until_tokens = usize::MAX;
    let (mut a1, _r1) = active_seq(1, 2);
    a1.preempt_immune_until_tokens = usize::MAX;
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(1));
}

#[test]
fn victim_policy_vision_requeue_excluded_but_spill_allowed() {
    const PAD: u32 = 999;
    let model = std::sync::Arc::new(PreemptStubModel {
        vision_pad: Some(PAD),
        ..Default::default()
    });
    let (mut a0, _r0) = active_seq(0, 2);
    a0.seq.tokens.insert(2, PAD);
    let (a1, _r1) = active_seq(1, 8);
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(1));
    assert_eq!(choose_decode_victim(&*model, &active, true), Some(0));
}

#[test]
fn resume_reprefills_exact_history_and_preserves_stream_state() {
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (a, _rx) = active_seq(3, 6);
    let last_token = a.last_token;
    let out_before = a.output_tokens.clone();
    let remaining_before = a.remaining;
    let history = a.seq.tokens.clone();

    let p = preempt_requeue(&SchedIo::for_test_with(model.clone()), a);
    assert_eq!(p.tokens, history);
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![3]);

    let resumed = resume_preempted_seq(&*model, &SchedIo::for_test_with(model.clone()), p)
        .expect("resume succeeds");
    assert_eq!(*model.prefilled.lock().unwrap(), vec![history.clone()]);
    assert_eq!(resumed.seq.tokens, history);
    assert_eq!(resumed.seq.seq_len, history.len());
    assert_eq!(resumed.last_token, last_token);
    assert_eq!(resumed.output_tokens, out_before);
    assert_eq!(resumed.remaining, remaining_before);
    assert!(!resumed.finished);
    assert_eq!(
        resumed.preempt_immune_until_tokens,
        out_before.len() + PREEMPT_IMMUNITY_TOKENS
    );
}

#[test]
fn resume_loop_gates_on_blocks_and_reclaims_from_prefix_cache() {
    let model = std::sync::Arc::new(PreemptStubModel {
        total_blocks: 100,
        free_blocks: AtomicUsize::new(0),
        reclaimable: AtomicUsize::new(50),
        ..Default::default()
    });
    let (a, _rx) = active_seq(0, 4);
    let p = {
        let mut history_seq = a;
        history_seq.seq.tokens = (0..32).collect();
        preempt_requeue(&SchedIo::for_test_with(model.clone()), history_seq)
    };
    let mut preempted = vec![p];
    let mut active = Vec::new();
    // 2026-09-25: 32 tokens at block size 16 need 32 / 16 + 1 = 3 blocks
    // plus 1 of growth. None are free, so the loop must reclaim.
    resume_preempted_seqs(
        &*model,
        &SchedIo::for_test_with(model.clone()),
        &mut active,
        &mut preempted,
        8,
        16,
    );
    assert_eq!(active.len(), 1);
    assert!(preempted.is_empty());

    let model2 = std::sync::Arc::new(PreemptStubModel {
        total_blocks: 100,
        ..Default::default()
    });
    let (a2, mut rx2) = active_seq(0, 4);
    let p2 = preempt_requeue(&SchedIo::for_test_with(model2.clone()), a2);
    let mut preempted2 = vec![p2];
    let mut active2 = Vec::new();
    resume_preempted_seqs(
        &*model2,
        &SchedIo::for_test_with(model2.clone()),
        &mut active2,
        &mut preempted2,
        8,
        16,
    );
    assert!(active2.is_empty());
    assert_eq!(preempted2.len(), 1);
    assert!(matches!(
        rx2.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
}

#[test]
fn resume_loop_errors_out_a_sequence_that_can_never_fit() {
    let model = std::sync::Arc::new(PreemptStubModel {
        total_blocks: 2,
        ..Default::default()
    });
    let (a, mut rx) = active_seq(0, 4);
    let p = {
        let mut s = a;
        s.seq.tokens = (0..64).collect();
        preempt_requeue(&SchedIo::for_test_with(model.clone()), s)
    };
    let mut preempted = vec![p];
    let mut active = Vec::new();
    resume_preempted_seqs(
        &*model,
        &SchedIo::for_test_with(model.clone()),
        &mut active,
        &mut preempted,
        8,
        16,
    );
    assert!(preempted.is_empty() && active.is_empty());
    assert!(rx.try_recv().expect("error delivered").is_err());
}

// 2026-10-14 (race-memory, A184): batched MTP steps refused for KV. kvagcheck2 (n3n4) saw
// one refusal in a C=4 batched verify end all four replies with finish "stop" and no error.

/// 2026-10-14: The message `lazy_agree::agreed_outcome` gives a rank that admitted when
/// another rank refused.
const VOTED: &str = "KV cache exhausted: decode/verify step of slot 2 through KV block 256: \
                     rank(s) [1] refused this step's KV/lazy-map admission, so every rank \
                     refuses it (rank-agreed)";

#[test]
fn spec_refusal_is_retryable_only_with_the_rank_vote_on_multi_rank() {
    use super::preempt::spec_kv_refusal_retryable;
    let one = PreemptStubModel::default();
    let ep = PreemptStubModel {
        ep: true,
        ..Default::default()
    };
    assert!(spec_kv_refusal_retryable(
        &one,
        "KV cache exhausted: no free blocks"
    ));
    assert!(!spec_kv_refusal_retryable(
        &one,
        "CUDA error 700: illegal memory access"
    ));
    // 2026-10-14: without the vote a worker may have run the step (A183).
    assert!(!spec_kv_refusal_retryable(
        &ep,
        "KV cache exhausted: no free blocks"
    ));
    assert!(spec_kv_refusal_retryable(&ep, VOTED));
}

/// 2026-10-14: Two unfinished rows at slots 0 and 1, 3 drafts each (verify width 4).
fn two_drafted() -> (
    Vec<super::types::ActiveSeq>,
    super::test_support::RespRx,
    super::test_support::RespRx,
) {
    let (mut a0, rx0) = active_seq(0, 3);
    let (mut a1, rx1) = active_seq(1, 4);
    a0.pending_drafts = vec![7, 8, 9];
    a1.pending_drafts = vec![7, 8, 9];
    (vec![a0, a1], rx0, rx1)
}

/// 2026-10-14: Run `step_verify_k4_batched` over both rows of `two_drafted` on `model`.
fn verify_batched(
    model: std::sync::Arc<PreemptStubModel>,
    active: &mut [super::types::ActiveSeq],
) -> Option<String> {
    let sched = SchedCtx::for_test_with(model.clone());
    let c = super::think_end_token0_tests::ctx(&sched, &active[0]);
    let mut batch: Vec<&mut super::types::ActiveSeq> = active.iter_mut().collect();
    super::verify_k4_batch_step::step_verify_k4_batched(&*model, &mut batch, &sched, &[4, 4], 3, &c)
}

#[test]
fn batched_verify_kv_refusal_is_returned_and_finishes_nothing() {
    for model in [
        PreemptStubModel::failing(1),
        PreemptStubModel {
            ep: true,
            kv_error: Some(VOTED),
            ..PreemptStubModel::failing(1)
        },
    ] {
        let (mut active, mut rx0, mut rx1) = two_drafted();
        let refusal = verify_batched(std::sync::Arc::new(model), &mut active);
        assert!(
            refusal
                .as_deref()
                .is_some_and(|m| m.contains("KV cache exhausted")),
            "{refusal:?}"
        );
        for a in &active {
            assert!(!a.finished && a.error.is_none(), "a refused row stays live");
        }
        for rx in [&mut rx0, &mut rx1] {
            assert!(matches!(
                rx.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ));
        }
    }
}

#[test]
fn batched_verify_failure_that_cannot_retry_fails_rows_with_an_error_not_a_stop() {
    for model in [
        // 2026-10-14: a device error, and a KV refusal on a multi-rank serve without the vote.
        PreemptStubModel {
            hard_error: Some("CUDA error 700: illegal memory access"),
            ..Default::default()
        },
        PreemptStubModel {
            ep: true,
            ..PreemptStubModel::failing(1)
        },
    ] {
        let (mut active, _rx0, _rx1) = two_drafted();
        let refusal = verify_batched(std::sync::Arc::new(model), &mut active);
        assert!(refusal.is_none());
        for a in &active {
            assert!(a.finished, "the row ends");
            assert!(
                a.error.is_some(),
                "with an error, never a bare finish read as stop"
            );
        }
    }
}

#[test]
fn spec_refusal_preempts_the_least_progress_live_row_never_a_finished_one() {
    let model = std::sync::Arc::new(PreemptStubModel::default());
    // 2026-10-14: slot 0 finished this step with the fewest tokens; it must not be chosen.
    let (mut done, _rxd) = active_seq(0, 1);
    done.finished = true;
    let (mut a1, mut rx1) = active_seq(1, 3);
    let (mut a2, _rx2) = active_seq(2, 6);
    a1.pending_drafts = vec![7, 8, 9];
    a2.pending_drafts = vec![7, 8, 9];
    let mut active = vec![done, a1, a2];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    super::preempt::preempt_after_spec_refusal(
        &SchedCtx::for_test_with(model.clone()),
        &mut active,
        VOTED,
        &mut swapped,
        &mut preempted,
    );
    assert_eq!(preempted.len(), 1);
    assert_eq!(
        preempted[0].a.seq.slot_idx, 1,
        "least progress among the live rows"
    );
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![1]);
    assert!(swapped.is_empty());
    let slots: Vec<usize> = active.iter().map(|a| a.seq.slot_idx).collect();
    assert_eq!(slots, vec![0, 2]);
    assert!(
        active[0].finished && active[0].error.is_none(),
        "the finished row is untouched"
    );
    assert!(!active[1].finished && active[1].error.is_none());
    assert!(
        active[1].pending_drafts.is_empty(),
        "survivors re-bootstrap their drafts"
    );
    assert!(matches!(
        rx1.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
}

#[test]
fn spec_refusal_with_one_live_row_fails_it_with_an_error() {
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (mut done, _rxd) = active_seq(0, 1);
    done.finished = true;
    let (a1, _rx1) = active_seq(1, 3);
    let mut active = vec![done, a1];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    super::preempt::preempt_after_spec_refusal(
        &SchedCtx::for_test_with(model),
        &mut active,
        VOTED,
        &mut swapped,
        &mut preempted,
    );
    assert!(preempted.is_empty() && swapped.is_empty());
    assert_eq!(active.len(), 2);
    assert!(
        active[0].error.is_none(),
        "the finished row keeps its own ending"
    );
    assert!(active[1].finished);
    assert!(
        active[1]
            .error
            .as_deref()
            .is_some_and(|e| e.contains("KV cache exhausted"))
    );
}
