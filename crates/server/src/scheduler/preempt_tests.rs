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
use super::prefill_b_step::prefill_request;
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
    let model = std::sync::Arc::new(PreemptStubModel {
        chunks_ok: true,
        ..Default::default()
    });
    let (a, _rx) = active_seq(3, 6);
    let last_token = a.last_token;
    let out_before = a.output_tokens.clone();
    let remaining_before = a.remaining;
    let history = a.seq.tokens.clone();

    let p = preempt_requeue(&SchedIo::for_test_with(model.clone()), a);
    assert_eq!(p.tokens, history);
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![3]);

    let resumed = resume_preempted_seq(&*model, &SchedIo::for_test_with(model.clone()), p, 0)
        .expect("resume succeeds");
    // 2026-10-14: one chunk with chunking off (`max_prefill_tokens` 0).
    assert_eq!(
        *model.chunks.lock().unwrap(),
        vec![(0, history.len(), true)]
    );
    assert!(model.prefilled.lock().unwrap().is_empty());
    assert_eq!(resumed.seq.prompt_len, history.len());
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
fn a_resume_start_with_the_ssm_pool_full_waits_and_never_fails_the_row() {
    // 2026-10-15 (race-prefill, A195; race-memory's code read of 57c4fc20): the SSM pool has one
    // slot per batch slot, and rows still prefilling (gap (e)'s waiting chunk-0 heads) hold
    // theirs, so a resume can start with every slot taken. Before the fix the victim got
    // "preempt-resume alloc failed" and lost its reply; now it stays preempted, nothing is sent,
    // and it resumes once a slot frees. Both the synchronous and the one-chunk-per-tick paths.
    for one_chunk in [false, true] {
        let model = std::sync::Arc::new(PreemptStubModel {
            total_blocks: 100,
            free_blocks: AtomicUsize::new(100),
            chunks_ok: true,
            fail_allocs: AtomicUsize::new(1),
            ..Default::default()
        });
        let io = SchedIo::for_test_with(model.clone());
        let (a, mut rx) = active_seq(0, 4);
        let mut preempted = vec![preempt_requeue(&io, a)];
        let mut active = Vec::new();
        let run = |active: &mut Vec<_>, preempted: &mut Vec<_>| {
            resume_preempted_seqs(&*model, &io, active, preempted, 8, 16, 0, one_chunk, None)
        };
        assert!(!run(&mut active, &mut preempted), "one_chunk={one_chunk}: no chunk ran");
        assert!(active.is_empty(), "one_chunk={one_chunk}");
        assert_eq!(preempted.len(), 1, "one_chunk={one_chunk}: still preempted");
        assert!(preempted[0].resume.is_none(), "one_chunk={one_chunk}");
        assert!(
            matches!(rx.try_recv(), Err(tokio::sync::oneshot::error::TryRecvError::Empty)),
            "one_chunk={one_chunk}: no error frame"
        );
        // 2026-10-15: a slot is free on the next tick: the resume runs to the end (nobody decodes).
        run(&mut active, &mut preempted);
        assert_eq!(active.len(), 1, "one_chunk={one_chunk}: resumed");
        assert!(preempted.is_empty(), "one_chunk={one_chunk}");
    }
}

#[test]
fn resume_loop_gates_on_blocks_and_reclaims_from_prefix_cache() {
    let model = std::sync::Arc::new(PreemptStubModel {
        total_blocks: 100,
        free_blocks: AtomicUsize::new(0),
        reclaimable: AtomicUsize::new(50),
        chunks_ok: true,
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
        0,
        false,
        None,
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
        0,
        false,
        None,
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
        0,
        false,
        None,
    );
    assert!(preempted.is_empty() && active.is_empty());
    assert!(rx.try_recv().expect("error delivered").is_err());
}

// 2026-10-14 (race-memory, kvagcheck8): on the 2-node car the first resume after a batched-MTP
// preemption hung both ranks: rank 0 announced one `0xFFFFFFF0` chunk and then ran a whole
// `prefill`, while the worker ran `prefill_chunk`, whose prefix lookup joins a collective.

#[test]
fn resume_on_two_ranks_announces_each_planned_chunk_and_runs_prefill_chunk() {
    let model = std::sync::Arc::new(PreemptStubModel {
        ep: true,
        chunks_ok: true,
        block_size: Some(16),
        ..Default::default()
    });
    let (a, _rx) = active_seq(2, 4);
    let p = {
        let mut s = a;
        s.seq.tokens = (0..40).collect();
        preempt_requeue(&SchedIo::for_test_with(model.clone()), s)
    };
    // 2026-10-14: the requeue's own release word (`0xFFFFFFF1`) is not under test here.
    model.ep_words.lock().unwrap().clear();
    let resumed = resume_preempted_seq(&*model, &SchedIo::for_test_with(model.clone()), p, 16)
        .expect("resume succeeds on two ranks");

    // 2026-10-14: 40 tokens at a 16-token chunk cap: 16 + 16 + 8, the same plan as a prompt.
    assert_eq!(
        *model.chunks.lock().unwrap(),
        vec![(0, 16, false), (16, 16, false), (32, 8, true)]
    );
    // 2026-10-14: each chunk is announced to the worker with the bounds rank 0 then runs.
    assert_eq!(
        *model.ep_words.lock().unwrap(),
        vec![
            0xFFFFFFF0, 16, 0, 40, //
            0xFFFFFFF0, 16, 16, 40, //
            0xFFFFFFF0, 8, 32, 40,
        ]
    );
    assert!(model.prefilled.lock().unwrap().is_empty());
    assert_eq!(resumed.seq.tokens, (0..40).collect::<Vec<u32>>());
    assert_eq!(resumed.seq.prompt_len, 40);
}

// 2026-10-15 (race-prefill, A192): `METRALE_RESUME_ONE_CHUNK_PER_TICK`. A synchronous resume
// re-prefills the whole history inside one tick, so every decoding row froze for it (a 324K
// resume ~160 s). With the lever, each call runs one planned chunk while a row decodes.

/// 2026-10-15: A two-rank stub with room for the resume: 100 free blocks of 16 tokens.
fn chunked_model() -> std::sync::Arc<PreemptStubModel> {
    std::sync::Arc::new(PreemptStubModel {
        ep: true,
        chunks_ok: true,
        block_size: Some(16),
        total_blocks: 100,
        free_blocks: AtomicUsize::new(100),
        ..Default::default()
    })
}

/// 2026-10-15: A requeued victim with a 40-token history, its requeue words cleared.
fn victim_40(model: &std::sync::Arc<PreemptStubModel>) -> super::types::PreemptedSeq {
    let (a, _rx) = active_seq(2, 4);
    let mut s = a;
    s.seq.tokens = (0..40).collect();
    let p = preempt_requeue(&SchedIo::for_test_with(model.clone()), s);
    model.ep_words.lock().unwrap().clear();
    model.chunks.lock().unwrap().clear();
    p
}

fn resume_calls(
    model: &std::sync::Arc<PreemptStubModel>,
    active: &mut Vec<super::types::ActiveSeq>,
    preempted: &mut Vec<super::types::PreemptedSeq>,
    one_chunk_per_tick: bool,
) {
    resume_preempted_seqs(
        &**model,
        &SchedIo::for_test_with(model.clone()),
        active,
        preempted,
        8,
        16,
        16,
        one_chunk_per_tick,
        None,
    );
}

#[test]
fn chunked_resume_runs_one_chunk_per_tick_while_a_row_decodes_and_matches_the_sync_resume() {
    // 2026-10-15: the synchronous resume of the same history, as the reference.
    let sync_model = chunked_model();
    let p = victim_40(&sync_model);
    let reference = resume_preempted_seq(
        &*sync_model,
        &SchedIo::for_test_with(sync_model.clone()),
        p,
        16,
    )
    .expect("sync resume");
    let sync_chunks = sync_model.chunks.lock().unwrap().clone();
    let sync_words = sync_model.ep_words.lock().unwrap().clone();
    assert_eq!(sync_chunks, vec![(0, 16, false), (16, 16, false), (32, 8, true)]);

    let model = chunked_model();
    let mut preempted = vec![victim_40(&model)];
    let (decoder, _drx) = active_seq(5, 3);
    let mut active = vec![decoder];
    // 2026-10-15: one chunk per call while slot 5 decodes; the row stays preempted in between.
    for (n, offset) in [(1, 16), (2, 32)] {
        resume_calls(&model, &mut active, &mut preempted, true);
        assert_eq!(model.chunks.lock().unwrap().len(), n, "one chunk per tick");
        assert_eq!(active.len(), 1, "not decoding until the last chunk");
        assert_eq!(preempted.len(), 1);
        assert_eq!(preempted[0].resume.as_ref().map(|r| r.offset), Some(offset));
    }
    resume_calls(&model, &mut active, &mut preempted, true);
    assert!(preempted.is_empty());
    assert_eq!(active.len(), 2);
    // 2026-10-15: the same chunks and the same EP words, in the same order, as the sync resume.
    assert_eq!(*model.chunks.lock().unwrap(), sync_chunks);
    assert_eq!(*model.ep_words.lock().unwrap(), sync_words);
    let resumed = &active[1];
    assert_eq!(resumed.seq.tokens, reference.seq.tokens);
    assert_eq!(resumed.seq.prompt_len, reference.seq.prompt_len);
    assert_eq!(resumed.seq.slot_idx, reference.seq.slot_idx);
    assert_eq!(resumed.last_token, reference.last_token);
    assert_eq!(resumed.output_tokens, reference.output_tokens);
    assert_eq!(resumed.remaining, reference.remaining);
    assert_eq!(
        resumed.preempt_immune_until_tokens,
        reference.preempt_immune_until_tokens
    );
}

#[test]
fn chunked_resume_with_nobody_decoding_runs_to_the_end_in_one_call() {
    let model = chunked_model();
    let mut preempted = vec![victim_40(&model)];
    let mut active = Vec::new();
    resume_calls(&model, &mut active, &mut preempted, true);
    assert!(preempted.is_empty());
    assert_eq!(active.len(), 1);
    assert_eq!(
        *model.chunks.lock().unwrap(),
        vec![(0, 16, false), (16, 16, false), (32, 8, true)]
    );
}

#[test]
fn chunked_resume_waits_while_the_rest_does_not_fit_and_never_runs_a_short_chunk() {
    let model = chunked_model();
    let mut preempted = vec![victim_40(&model)];
    let (decoder, _drx) = active_seq(5, 3);
    let mut active = vec![decoder];
    resume_calls(&model, &mut active, &mut preempted, true);
    assert_eq!(model.chunks.lock().unwrap().len(), 1);
    // 2026-10-15: the decoder took the free blocks: 40 / 16 + 2 = 4 needed, 3 free, none to
    // reclaim. The resume waits; it does not start a chunk it might not finish.
    model.free_blocks.store(3, Ordering::Relaxed);
    for _ in 0..3 {
        resume_calls(&model, &mut active, &mut preempted, true);
    }
    assert_eq!(model.chunks.lock().unwrap().len(), 1, "held, no chunk");
    assert_eq!(preempted.len(), 1);
    assert_eq!(active.len(), 1);
    // 2026-10-15: blocks come back (a row finished): it carries on where it stopped.
    model.free_blocks.store(100, Ordering::Relaxed);
    resume_calls(&model, &mut active, &mut preempted, true);
    assert_eq!(
        *model.chunks.lock().unwrap(),
        vec![(0, 16, false), (16, 16, false)]
    );
}

#[test]
fn chunked_resume_of_a_cancelled_row_is_released_between_chunks() {
    let model = chunked_model();
    let mut preempted = vec![victim_40(&model)];
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    preempted[0].a.cancel_flag = Some(flag.clone());
    let (decoder, _drx) = active_seq(5, 3);
    let mut active = vec![decoder];
    resume_calls(&model, &mut active, &mut preempted, true);
    let resume_slot = preempted[0].resume.as_ref().unwrap().seq.slot_idx;
    let freed_before = model.freed_slots.lock().unwrap().len();
    flag.store(true, Ordering::Relaxed);
    resume_calls(&model, &mut active, &mut preempted, true);
    assert!(preempted.is_empty(), "dropped");
    assert_eq!(active.len(), 1, "never resumed");
    assert_eq!(model.chunks.lock().unwrap().len(), 1, "no chunk after the cancel");
    let freed = model.freed_slots.lock().unwrap().clone();
    assert_eq!(freed.len(), freed_before + 1);
    assert_eq!(*freed.last().unwrap(), resume_slot);
}

#[test]
fn lever_off_resume_is_unchanged_one_call_and_never_in_flight() {
    let model = chunked_model();
    let mut preempted = vec![victim_40(&model)];
    let (decoder, _drx) = active_seq(5, 3);
    let mut active = vec![decoder];
    resume_calls(&model, &mut active, &mut preempted, false);
    assert!(preempted.is_empty());
    assert_eq!(active.len(), 2);
    assert_eq!(model.chunks.lock().unwrap().len(), 3);
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

/// 2026-10-14: Run `step_verify_dflash_batched` (γ=3) over both rows of `two_drafted`.
fn verify_batched_dflash(
    model: std::sync::Arc<PreemptStubModel>,
    active: &mut [super::types::ActiveSeq],
) -> Option<String> {
    let sched = SchedCtx::for_test_with(model.clone());
    let c = super::think_end_token0_tests::ctx(&sched, &active[0]);
    let mut batch: Vec<&mut super::types::ActiveSeq> = active.iter_mut().collect();
    super::verify_dflash_batch_step::step_verify_dflash_batched(
        &*model, &mut batch, &sched, 3, 3, &c,
    )
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

#[test]
fn dflash_batched_verify_kv_refusal_is_returned_and_finishes_nothing() {
    // 2026-10-14 (race-memory, A184): before, the refusal put the drafts back and returned
    // nothing, so it repeated every step and the batch stalled with no preemption.
    for model in [
        PreemptStubModel::failing(1),
        PreemptStubModel {
            ep: true,
            kv_error: Some(VOTED),
            ..PreemptStubModel::failing(1)
        },
    ] {
        let (mut active, _rx0, _rx1) = two_drafted();
        let refusal = verify_batched_dflash(std::sync::Arc::new(model), &mut active);
        assert!(
            refusal
                .as_deref()
                .is_some_and(|m| m.contains("KV cache exhausted")),
            "{refusal:?}"
        );
        for a in &active {
            assert!(!a.finished && a.error.is_none(), "a refused row stays live");
        }
    }
}

#[test]
fn dflash_batched_verify_failure_that_cannot_retry_fails_rows_with_an_error() {
    // 2026-10-14: before, any verify error also put the drafts back, so a device error
    // repeated forever.
    for model in [
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
        let refusal = verify_batched_dflash(std::sync::Arc::new(model), &mut active);
        assert!(refusal.is_none());
        for a in &active {
            assert!(
                a.finished && a.error.is_some(),
                "an error, never a bare finish"
            );
        }
    }
}

#[test]
fn a_serial_verify_kv_refusal_is_handed_back_and_other_failures_end_the_row() {
    // 2026-10-14 (race-memory, A184): a failed serial verify used to reach the client as a bare
    // `finished` (HTTP 200 "stop"); it must end with an error frame.
    // 2026-10-15 (race-memory, gap d, gpause1 VOID): a retryable KV refusal (single rank, or the
    // 2-rank vote's "rank-agreed") now goes back to `step_mtp` with the row live and unchanged,
    // so `preempt_after_spec_refusal` can pause a victim (a grammar row included) instead of
    // the lone plain row being failed. An unvoted 2-rank refusal and a non-KV error still end
    // the row with an error frame.
    type Step = fn(
        &dyn metrale_model_engine::traits::Model,
        &mut super::types::ActiveSeq,
        &SchedCtx,
        &[u32],
        usize,
        &crate::scheduler::logit_processors::LogitsContext,
        bool,
    ) -> Option<String>;
    let steps: [(&str, Step, bool); 4] = [
        ("k2", super::verify_k2_step::step_verify_k2, false),
        ("k3", super::verify_k3_step::step_verify_k3, false),
        ("k4", super::verify_k4_step::step_verify_k4, false),
        (
            "dflash",
            super::verify_dflash_step::step_verify_dflash,
            true,
        ),
    ];
    let cases: [(bool, &'static str, bool); 4] = [
        (false, "KV cache exhausted: no free blocks", true),
        (true, VOTED, true),
        (true, "KV cache exhausted: no free blocks", false),
        (false, "CUDA error 700: illegal memory access", false),
    ];
    for (name, step, raw) in steps {
        for (ep, msg, retry) in cases {
            let model = std::sync::Arc::new(PreemptStubModel {
                ep,
                hard_error: Some(msg),
                ..Default::default()
            });
            let (mut active, _rx0, _rx1) = two_drafted();
            let sched = SchedCtx::for_test_with(model.clone());
            let c = super::think_end_token0_tests::ctx(&sched, &active[0]);
            let a = &mut active[0];
            let (last, n_out) = (a.last_token, a.output_tokens.len());
            let refusal = step(&*model, a, &sched, &[7, 8, 9], 3, &c, raw);
            assert_eq!(
                model.decode_calls.load(Ordering::SeqCst),
                1,
                "{name} {ep} {msg}: one verify forward"
            );
            if retry {
                assert!(
                    refusal.as_deref().is_some_and(|m| m.contains(msg)),
                    "{name} {ep}: {refusal:?}"
                );
                assert!(
                    !a.finished && a.error.is_none(),
                    "{name} {ep}: the row stays live"
                );
                assert_eq!(
                    (a.last_token, a.output_tokens.len()),
                    (last, n_out),
                    "{name} {ep}: unchanged"
                );
            } else {
                assert!(refusal.is_none(), "{name} {ep} {msg}: not retryable");
                assert!(a.finished, "{name} {ep} {msg}: the row ends");
                assert!(
                    a.error.as_deref().is_some_and(|m| m.contains(msg)),
                    "{name} {ep}: {:?}",
                    a.error
                );
            }
        }
    }
}

#[test]
fn sliced_resume_chunk_is_capped_like_a_prefill_chunk_and_unchanged_without_a_cap() {
    // 2026-10-15 (race-prefill, A192): `METRALE_RESUME_SLICED` passes the while-decoding cap in
    // force; without one the chunk is the synchronous resume's.
    let model = PreemptStubModel {
        block_size: Some(16),
        ..Default::default()
    };
    let tokens: Vec<u32> = (0..10_000).collect();
    assert_eq!(super::preempt::resume_chunk_len(&model, &tokens, 0, 8192, None), 8192);
    assert_eq!(super::preempt::resume_chunk_len(&model, &tokens, 0, 8192, Some(4096)), 4096);
    assert_eq!(super::preempt::resume_chunk_len(&model, &tokens, 8192, 8192, Some(4096)), 1808);
    // 2026-10-15: a cap above the chunk changes nothing.
    assert_eq!(super::preempt::resume_chunk_len(&model, &tokens, 0, 2048, Some(4096)), 2048);
}

/// 2026-10-15 (race-memory, gap (a)): a prefill chunk refused for KV frees room by requeuing one
/// decoding row for a later resume, never by ending it. cap2h (L34, 2026-10-15) lost a 324K reply to
/// the old path ("preempted: KV cache exhausted (a prefill needed its blocks)"). The victim follows
/// decode's rule (fewest output tokens), the prefill retries the same chunk, and no row gets a frame.
#[test]
fn a_prefill_chunk_refused_for_kv_requeues_a_decoder_and_ends_none() {
    use super::phase_continue_prefills::continue_in_progress_prefills;
    use super::test_support::test_prefill_ident;
    // EP (2 ranks, as cap2h): no fused `mixed_forward`, the plain chunk path.
    let model = std::sync::Arc::new(PreemptStubModel {
        first_token: Some(7),
        ep: true,
        ..Default::default()
    });
    model.fail_prefill_chunks.store(1, Ordering::SeqCst);
    let sched = SchedCtx::for_test_with(model.clone());
    let (a0, mut rx0) = active_seq(0, 9);
    let (a1, mut rx1) = active_seq(1, 3);
    let mut active = vec![a0, a1];
    let (mut p, _rxp) = test_prefill_ident(7, 16);
    // 2026-10-15 (gap (e)): mid-prefill, as cap2h's f#2 (311,296 of 324,383); a prefill still at
    // chunk 0 waits instead (`a_new_prefill_refused_at_chunk_0_waits_...`).
    p.chunk_offset = 8;
    let mut prefilling = vec![p];
    let mut preempted = Vec::new();
    continue_in_progress_prefills(
        &*model,
        &crate::scheduling_policy::FifoPolicy,
        &mut active,
        &mut prefilling,
        &mut preempted,
        16,
        16,
        false,
        0,
        0,
        false,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        false,
        &sched,
    );
    assert_eq!(preempted.len(), 1, "one row requeued for resume");
    assert_eq!(
        preempted[0].a.seq.slot_idx, 1,
        "decode's victim rule: fewest output tokens"
    );
    assert_eq!(
        preempted[0].a.output_tokens.len(),
        3,
        "its reply so far is kept for the resume"
    );
    assert!(
        active.iter().any(|a| a.seq.slot_idx == 0),
        "the other decoder keeps running"
    );
    assert!(
        rx0.try_recv().is_err() && rx1.try_recv().is_err(),
        "no row got a response frame"
    );
    let chunks = model.chunks.lock().unwrap().clone();
    assert_eq!(
        chunks.len(),
        2,
        "the refused chunk is retried once: {chunks:?}"
    );
    assert_eq!(chunks[0], chunks[1]);
}

/// 2026-10-15 (race-memory, gap (a)): with no decoding row to requeue, the prefill's own refusal
/// stands (its request fails); nothing is requeued.
#[test]
fn a_prefill_chunk_refused_for_kv_with_no_decoder_fails_the_prefill_only() {
    use super::phase_continue_prefills::continue_in_progress_prefills;
    use super::test_support::test_prefill_ident;
    let model = std::sync::Arc::new(PreemptStubModel {
        first_token: Some(7),
        ..Default::default()
    });
    model.fail_prefill_chunks.store(1, Ordering::SeqCst);
    let sched = SchedCtx::for_test_with(model.clone());
    let mut active = Vec::new();
    let (p, mut rxp) = test_prefill_ident(7, 16);
    let mut prefilling = vec![p];
    let mut preempted = Vec::new();
    continue_in_progress_prefills(
        &*model,
        &crate::scheduling_policy::FifoPolicy,
        &mut active,
        &mut prefilling,
        &mut preempted,
        16,
        16,
        false,
        0,
        0,
        false,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        false,
        &sched,
    );
    assert!(preempted.is_empty() && active.is_empty() && prefilling.is_empty());
    let r = rxp
        .try_recv()
        .expect("the prefill's request got its answer");
    assert!(r.is_err(), "the prefill fails with its own refusal");
    assert_eq!(
        model.chunks.lock().unwrap().len(),
        1,
        "no retry without a victim"
    );
}

/// 2026-10-15 (race-memory, gap (e)): fchr arm A (L34, 2026-10-15 23:59:41 UTC) lost two new
/// requests to "prefill_chunk failed: ... rank-agreed refusal ...: KV cache exhausted: the DSA
/// indexer pool cannot map 2.0 MiB more". A prefill still at chunk 0 holds no KV blocks: while rows
/// decode it waits for room and retries next tick; it never requeues a decoder (gap (a)'s rule is
/// for a prefill that already holds blocks) and nobody gets a frame.
#[test]
fn a_new_prefill_refused_at_chunk_0_waits_while_rows_decode_and_preempts_none() {
    use super::phase_continue_prefills::continue_in_progress_prefills;
    use super::test_support::test_prefill_ident;
    let model = std::sync::Arc::new(PreemptStubModel {
        first_token: Some(7),
        ep: true,
        kv_error: Some(VOTED),
        ..Default::default()
    });
    model.fail_prefill_chunks.store(1, Ordering::SeqCst);
    let sched = SchedCtx::for_test_with(model.clone());
    let (a0, mut rx0) = active_seq(0, 9);
    let (a1, mut rx1) = active_seq(1, 3);
    let mut active = vec![a0, a1];
    let (p, mut rxp) = test_prefill_ident(7, 16);
    let mut prefilling = vec![p];
    let mut preempted = Vec::new();
    let tick = |active: &mut Vec<_>, prefilling: &mut Vec<_>, preempted: &mut Vec<_>| {
        continue_in_progress_prefills(
            &*model,
            &crate::scheduling_policy::FifoPolicy,
            active,
            prefilling,
            preempted,
            16,
            16,
            false,
            0,
            0,
            false,
            false,
            false,
            None,
            None,
            None,
            None,
            None,
            false,
            &sched,
        )
    };
    tick(&mut active, &mut prefilling, &mut preempted);
    assert!(
        preempted.is_empty(),
        "no decoder is requeued for a new request"
    );
    assert_eq!(active.len(), 2, "both decoders keep running");
    assert_eq!(prefilling.len(), 1, "the request waits as a prefill");
    assert!(prefilling[0].kv_wait_logged && prefilling[0].chunk_offset == 0);
    assert!(
        rx0.try_recv().is_err() && rx1.try_recv().is_err() && rxp.try_recv().is_err(),
        "no frame to anyone"
    );
    assert_eq!(
        model.chunks.lock().unwrap().len(),
        1,
        "no retry in the same tick"
    );
    // Next tick there is room: the same chunk 0 is re-sent to the worker and runs.
    tick(&mut active, &mut prefilling, &mut preempted);
    let chunks = model.chunks.lock().unwrap().clone();
    assert_eq!(chunks.len(), 2, "{chunks:?}");
    assert_eq!(chunks[0], chunks[1]);
    assert!(prefilling.is_empty() && preempted.is_empty());
    assert_eq!(active.len(), 3, "the request now decodes");
    let starts = model
        .ep_words
        .lock()
        .unwrap()
        .iter()
        .filter(|&&w| w == 0xFFFFFFF0)
        .count();
    assert_eq!(
        starts, 2,
        "the worker got the chunk command for each attempt"
    );
}

/// 2026-10-15 (race-memory, gap (e)): the start of a request: a rank-agreed (or single-rank) KV
/// refusal of chunk 0 keeps the request as a waiting prefill at offset 0 with its sink unanswered;
/// on an EP serve a refusal without the rank vote still fails the request (the ranks may be apart).
#[test]
fn a_new_request_refused_for_kv_at_chunk_0_is_kept_waiting_not_failed() {
    use super::emit_step::StartPrefillResult;
    use crate::api::InferenceRequest;
    let run = |ep: bool, kv_error: &'static str| {
        let model = PreemptStubModel {
            first_token: Some(7),
            ep,
            kv_error: Some(kv_error),
            ..Default::default()
        };
        model.fail_prefill_chunks.store(1, Ordering::SeqCst);
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        let mut engine = None;
        let req = InferenceRequest::Blocking {
            prompt_tokens: std::sync::Arc::new(vec![1, 2, 3]),
            session_hash: 0,
            cache_salt_hash: 0,
            adapter_slot: -1,
            src_lang_id: 0,
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
            image_pixels: vec![],
            max_tokens: 4,
            min_tokens: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 0.0,
            dry_allowed_length: 0,
            lz_penalty: 0.0,
            logit_bias: vec![],
            stop_tokens: vec![],
            enable_thinking: false,
            thinking_budget: None,
            repetition_detection: None,
            require_tool_call: false,
            tools_present: false,
            suppress_tool_call: false,
            disable_mtp: true,
            grammar_spec: None,
            seed: Some(42),
            top_logprobs: None,
            prompt_logprobs: None,
            echo: false,
            timeout_at: None,
            response_tx,
        };
        let r = super::prefill_a_step::start_chunked_prefill(
            &SchedCtx::for_test(),
            None,
            None,
            None,
            None,
            &model,
            req,
            &[],
            64,
            0,
            0,
            &mut engine,
            0,
            false,
            None,
            None,
        );
        let waiting = match r {
            Ok(StartPrefillResult::InProgress(p)) => {
                assert_eq!(p.chunk_offset, 0);
                assert_eq!(p.seq.slot_idx, 0);
                true
            }
            Ok(_) => panic!("neither waiting nor failed"),
            Err(_) => false,
        };
        (waiting, response_rx.try_recv().is_ok())
    };
    assert_eq!(
        run(true, VOTED),
        (true, false),
        "EP, rank-agreed: waits, no answer yet"
    );
    assert_eq!(
        run(false, "KV cache exhausted: no free blocks"),
        (true, false),
        "single rank: waits"
    );
    assert_eq!(
        run(true, "KV cache exhausted: no free blocks"),
        (false, true),
        "EP without the rank vote: fails, and the caller is answered"
    );
}

// 2026-10-15 (race-memory, A187): the same mismatch on a new request. With chunking off
// (`max_prefill_tokens == 0`) `prefill_request` announced one `0xFFFFFFF0` chunk and then ran
// a whole `prefill` on rank 0, while the worker ran `prefill_chunk`: an unchunked 2-rank serve
// would hang both ranks on its first prompt.

#[test]
fn unchunked_prefill_on_two_ranks_runs_the_announced_chunk() {
    let model = std::sync::Arc::new(PreemptStubModel {
        ep: true,
        first_token: Some(5),
        ..Default::default()
    });
    let mut req = super::test_support::blocking_request(None);
    if let crate::api::InferenceRequest::Blocking {
        prompt_tokens,
        max_tokens,
        ..
    } = &mut req
    {
        *prompt_tokens = std::sync::Arc::new((0..40).collect());
        *max_tokens = 8;
    }
    let a = prefill_request(
        &SchedCtx::for_test_with(model.clone()),
        None,
        None,
        None,
        None,
        &*model,
        req,
        &[],
        &mut None,
        64,
        None,
    )
    .expect("prefill succeeds on two ranks")
    .expect("joins decode");

    assert_eq!(*model.chunks.lock().unwrap(), vec![(0, 40, true)]);
    assert_eq!(model.ep_words.lock().unwrap()[..4], [0xFFFFFFF0, 40, 0, 40]);
    assert!(model.prefilled.lock().unwrap().is_empty());
    assert_eq!(a.seq.prompt_len, 40);
}

// 2026-10-15 (race-memory, CAPACITY.md step-2 condition c): a tool-call or forced-format row
// was never a victim, so under lazy admission a KV refusal on a batch of grammar rows ended
// them with an error. It is now requeued (never spilled: `SwappedSeq` drops the matcher).

/// 2026-10-15: A JSON-schema matcher over the toy vocabulary of `grammar::tests`.
fn json_grammar() -> crate::grammar::GrammarState {
    let vocab = crate::grammar::tests::test_vocab();
    let mut engine = crate::grammar::GrammarEngine::new(&vocab, &[130]).unwrap();
    let compiled = engine
        .compile_json_schema(r#"{"type":"object","properties":{"a":{"type":"integer"}}}"#)
        .unwrap();
    crate::grammar::GrammarState::new(&compiled, engine.vocab_size()).unwrap()
}

#[test]
fn victim_policy_grammar_row_is_a_requeue_victim() {
    const PAD: u32 = 999;
    let model = std::sync::Arc::new(PreemptStubModel {
        vision_pad: Some(PAD),
        ..Default::default()
    });
    let (mut g, _rg) = active_seq(0, 2);
    g.grammar_state = Some(json_grammar());
    let (a1, _r1) = active_seq(1, 8);
    let active = vec![g, a1];
    // 2026-10-15: fewest generated tokens wins, with or without spill.
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(0));
    assert_eq!(choose_decode_victim(&*model, &active, true), Some(0));

    // 2026-10-15: a grammar row with vision pads can be neither requeued nor spilled.
    let (mut g, _rg) = active_seq(0, 2);
    g.grammar_state = Some(json_grammar());
    g.seq.tokens.insert(2, PAD);
    let (a1, _r1) = active_seq(1, 8);
    let active = vec![g, a1];
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(1));
    assert_eq!(choose_decode_victim(&*model, &active, true), Some(1));
}

#[test]
fn kv_exhaustion_requeues_a_grammar_victim_with_its_matcher_even_with_spill() {
    let model = std::sync::Arc::new(PreemptStubModel::failing(1));
    let (a0, _rx0) = active_seq(0, 5);
    let (mut g, mut g_rx) = streaming_seq(1, 2);
    let mut gs = json_grammar();
    assert!(
        gs.accept_token(b'{' as u32),
        "toy vocab id of '{{' opens the object"
    );
    let steps = gs.num_history_steps();
    g.grammar_state = Some(gs);
    let mut active = vec![a0, g];
    let mut swapped = Vec::new();
    let mut preempted = Vec::new();
    let dir = std::env::temp_dir().join(format!(
        "metrale-preempt-tests-{}-grammar",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let spill = super::io::FileSpill::new(
        metrale_cache::kv_spill::KvSpillManager::new(dir, 1 << 20).expect("spill dir"),
    );

    let logits = decode_batch_with_preemption(
        &SchedCtx::for_test_with(model.clone()),
        &mut active,
        Some(&spill),
        &mut swapped,
        &mut preempted,
        &mut Vec::new(),
    );

    assert!(logits.is_some());
    assert_eq!(active.len(), 1);
    assert!(swapped.is_empty(), "a grammar row is never spilled");
    assert_eq!(preempted.len(), 1);
    let gs = preempted[0].a.grammar_state.as_ref().expect("matcher kept");
    assert_eq!(gs.num_history_steps(), steps);
    assert!(matches!(
        g_rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}
