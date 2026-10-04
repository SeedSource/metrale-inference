// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests that `prefilling` is served in arrival order. The
//! single-stream path advances `prefilling.first_mut()`, and
//! `promote_completed_prefills` must remove entries without reordering the
//! rest. The tick-loop tests run `continue_in_progress_prefills` with one
//! active decode and `use_mtp`: `mixing_blocked_by_spec` then rules out
//! the batched mixed step, and the non-empty `active` rules out the batched
//! prefill-only step, so every tick takes the single-stream path.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{
    Model, ModelAdapters, ModelDeviceFeed, ModelDraft, ModelEp, ModelForward, ModelLifecycle,
    ModelLogits, ModelSsmState, ModelStreams, ModelVerify, ModelVision, SequenceState,
};

use super::io::SchedIo;
use super::phase_continue_prefills::continue_in_progress_prefills;
use super::phase_promote_prefills::promote_completed_prefills;
use super::sched_ctx::SchedCtx;
use super::test_support::{test_prefill_ident, test_seq};
use super::types::{ActiveSeq, PrefillInProgress};
use crate::scheduling_policy::FifoPolicy;

/// 2026-09-25: The sampled first token. The fixture's `eos_tokens` is
/// empty, so it is never an EOS.
const FIRST: u32 = 7;

/// 2026-09-25: The prompt length and the prefill budget, so each prompt is
/// one chunk.
const CHUNK: usize = 4;

/// 2026-09-25: Minimal `Model`: every prefill chunk succeeds, and the
/// greedy first-token sample (temperature 0.0, no suppressed ids) is
/// `argmax_on_device`. The other methods are unreachable in these tests.
#[derive(Default)]
struct PrefillStubModel;

impl Model for PrefillStubModel {}

impl ModelLifecycle for PrefillStubModel {
    fn free_sequence(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn cache_sequence(&self, _seq: &SequenceState) {}
    fn detach_slot_for_reuse(&self, _seq: &mut SequenceState) {}
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _new_slot: usize) -> Result<()> {
        unreachable!("no compaction in this harness")
    }
}

impl ModelForward for PrefillStubModel {
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        _is_last: bool,
        _stream: u64,
    ) -> Result<DevicePtr> {
        seq.tokens
            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
        seq.seq_len = seq.tokens.len();
        Ok(DevicePtr::NULL)
    }
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("chunked prefill only")
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("decode is driven by mod.rs, not by this harness")
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        unreachable!("decode is driven by mod.rs, not by this harness")
    }
}

impl ModelLogits for PrefillStubModel {
    fn argmax_on_device(&self, _logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        Ok(FIRST)
    }
    fn vocab_size(&self) -> usize {
        32
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn copy_logits_to_host(&self, _l: DevicePtr, _dst: &mut [u8]) -> Result<()> {
        unreachable!("greedy fast path never reads logits back")
    }
    fn argmax_batch(&self, _l: DevicePtr, _n: usize, _st: u64) -> Result<Vec<u32>> {
        unreachable!("batched decode is not driven here")
    }
}

impl ModelAdapters for PrefillStubModel {}

impl ModelSsmState for PrefillStubModel {
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
}

impl ModelVerify for PrefillStubModel {
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        unreachable!("no speculation in this harness")
    }
}

impl ModelDraft for PrefillStubModel {
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no speculation in this harness")
    }
    fn save_hidden_for_mtp(&self, _token_idx: usize, _st: u64) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _mask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &metrale_sampling::SamplingParams,
        _n: usize,
    ) -> Result<metrale_model_engine::engine::GenerateResult> {
        unreachable!("no speculation in this harness")
    }
}

impl ModelVision for PrefillStubModel {}

impl ModelEp for PrefillStubModel {}

impl ModelStreams for PrefillStubModel {}

impl ModelDeviceFeed for PrefillStubModel {}

/// 2026-09-25: What one run of the tick loop observed.
struct Run {
    /// 2026-09-25: `(session_hash, tick)` for every prefill promoted into
    /// `active`.
    promoted: Vec<(u64, usize)>,
    /// 2026-09-25: Initially queued ids that were never promoted.
    stuck: Vec<u64>,
}

/// 2026-09-25: Run the arrive → continue → promote loop for `ticks` ticks.
///
/// `seed_depth` requests (ids `1..=seed_depth`) are queued at the start, and
/// one new request arrives every tick, before the continue phase, as in
/// `core/lanes.rs` (`start_new_requests`, then
/// `continue_in_progress_prefills`).
fn drive(seed_depth: usize, ticks: usize) -> Run {
    let model = PrefillStubModel;
    let policy = FifoPolicy;
    let sched = SchedCtx::for_test();

    // 2026-09-25: Response receivers, kept alive so no sink sees a closed
    // channel.
    let mut keep = Vec::new();
    let mut active: Vec<ActiveSeq> = Vec::new();
    let mut prefilling: Vec<PrefillInProgress> = Vec::new();

    // 2026-09-25: The one decode occupant (see the module doc). Its
    // `session_hash` is 0; prefill ids start at 1.
    let (occupant, rx) = test_seq(vec![1], usize::MAX, None, 8);
    keep.push(rx);
    active.push(occupant);

    let seeded: Vec<u64> = (1..=seed_depth as u64).collect();
    for &id in &seeded {
        let (p, rx) = test_prefill_ident(id, CHUNK);
        keep.push(rx);
        prefilling.push(p);
    }

    let mut next_id = seed_depth as u64 + 1;
    let mut promoted: Vec<(u64, usize)> = Vec::new();

    for tick in 0..ticks {
        let (p, rx) = test_prefill_ident(next_id, CHUNK);
        keep.push(rx);
        prefilling.push(p);
        next_id += 1;

        continue_in_progress_prefills(
            &model,
            &policy,
            &mut active,
            &mut prefilling,
            CHUNK,
            CHUNK,
            false,
            0,
            0,
            true,
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

        // 2026-09-25: Retire every promoted prefill at once; promotion
        // appends, so the occupant stays at index 0.
        for a in active.drain(1..) {
            promoted.push((a.session_hash, tick));
        }
    }

    let stuck = seeded
        .into_iter()
        .filter(|id| !promoted.iter().any(|(p, _)| p == id))
        .collect();
    Run { promoted, stuck }
}

/// 2026-09-25: Every request queued at the start is promoted although a
/// new one arrives every tick. A removal by `swap_remove` would move the
/// newest arrival to the head, so ids 2..=depth would never be served;
/// depth 1 passes either way.
#[test]
fn every_queued_prefill_advances_under_sustained_arrivals() {
    for depth in 1..=8 {
        let run = drive(depth, 500);
        assert!(
            run.stuck.is_empty(),
            "queue depth {depth}: request(s) {:?} never got a prefill chunk in 500 ticks",
            run.stuck,
        );
    }
}

/// 2026-09-25: With one chunk per request, the request `k` places from the
/// head is promoted on tick `k`.
#[test]
fn queued_prefills_are_served_in_arrival_order() {
    for depth in 1..=8 {
        let run = drive(depth, 500);
        for (rank, id) in (1..=depth as u64).enumerate() {
            let tick = run
                .promoted
                .iter()
                .find(|(p, _)| *p == id)
                .map(|(_, t)| *t)
                .unwrap_or_else(|| panic!("depth {depth}: request {id} never promoted"));
            assert_eq!(
                tick, rank,
                "depth {depth}: request {id} was {rank} places from the head but ran on tick {tick}",
            );
        }
    }
}

/// 2026-09-25: Control: each tick promotes one prefill, so a failure above
/// is about which requests ran, not a loop that makes no progress.
#[test]
fn the_tick_loop_makes_progress_every_tick() {
    let run = drive(8, 500);
    assert_eq!(
        run.promoted.len(),
        500,
        "expected one promotion per tick, got {}",
        run.promoted.len(),
    );
}

/// 2026-09-25: Promoting the head leaves the rest of the queue in arrival
/// order.
#[test]
fn promoting_the_head_preserves_the_order_of_the_remainder() {
    let model = PrefillStubModel;
    let mut keep = Vec::new();
    let mut prefilling: Vec<PrefillInProgress> = (1..=4)
        .map(|id| {
            let (p, rx) = test_prefill_ident(id, CHUNK);
            keep.push(rx);
            p
        })
        .collect();
    let mut active: Vec<ActiveSeq> = Vec::new();

    promote_completed_prefills(
        &model,
        &SchedIo::for_test_with(std::sync::Arc::new(PrefillStubModel)),
        &mut prefilling,
        vec![(0, Ok(FIRST))],
        &mut active,
        None,
        None,
        None,
        None,
        4096,
    );

    let order: Vec<u64> = prefilling.iter().map(|p| p.session_hash).collect();
    assert_eq!(
        order,
        vec![2, 3, 4],
        "head removal must not reorder the queue"
    );
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].session_hash, 1);
}

/// 2026-09-25: The batched paths pass several completed indices at once.
/// They are removed in reverse index order, and the rest of the queue stays
/// in arrival order.
#[test]
fn promoting_several_at_once_preserves_the_order_of_the_remainder() {
    let model = PrefillStubModel;
    let mut keep = Vec::new();
    let mut prefilling: Vec<PrefillInProgress> = (1..=5)
        .map(|id| {
            let (p, rx) = test_prefill_ident(id, CHUNK);
            keep.push(rx);
            p
        })
        .collect();
    let mut active: Vec<ActiveSeq> = Vec::new();

    promote_completed_prefills(
        &model,
        &SchedIo::for_test_with(std::sync::Arc::new(PrefillStubModel)),
        &mut prefilling,
        vec![(0, Ok(FIRST)), (2, Ok(FIRST))],
        &mut active,
        None,
        None,
        None,
        None,
        4096,
    );

    let order: Vec<u64> = prefilling.iter().map(|p| p.session_hash).collect();
    assert_eq!(
        order,
        vec![2, 4, 5],
        "multi-removal must not reorder the queue"
    );
    let promoted: Vec<u64> = active.iter().map(|a| a.session_hash).collect();
    assert_eq!(
        promoted,
        vec![3, 1],
        "promotion still walks the indices in reverse"
    );
}

/// 2026-10-03: A prefill that fails mid-way (e.g. "KV cache exhausted" on a later chunk)
/// answers its client with the reason. The sink used to be dropped, which a streaming
/// client read as an empty HTTP 200 stream with no finish reason (race #79).
#[test]
fn a_failed_prefill_answers_its_client_with_the_reason() {
    let model = PrefillStubModel;
    let (p1, mut rx1) = test_prefill_ident(1, CHUNK);
    let (p2, _rx2) = test_prefill_ident(2, CHUNK);
    let mut prefilling = vec![p1, p2];
    let mut active: Vec<ActiveSeq> = Vec::new();

    promote_completed_prefills(
        &model,
        &SchedIo::for_test_with(std::sync::Arc::new(PrefillStubModel)),
        &mut prefilling,
        vec![(
            0,
            Err("prefill failed: KV cache exhausted: no free blocks".into()),
        )],
        &mut active,
        None,
        None,
        None,
        None,
        4096,
    );

    let sent = rx1
        .try_recv()
        .expect("the sink must be answered, not dropped");
    let err = format!("{:#}", sent.err().expect("an error, not a response"));
    assert!(err.contains("KV cache exhausted"), "{err}");
    assert!(active.is_empty());
    assert_eq!(prefilling.len(), 1);
    assert_eq!(prefilling[0].session_hash, 2);
}
