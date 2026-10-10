// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The multi-rank handshake of `step_verify_dflash_batched`
//! (`METRALE_GLM_DFLASH_BATCHED_VERIFY`): on an EP model rank 0 sends the verdict words once
//! after the forward and makes exactly one failure-mask gather on every exit after them, before
//! any propose; an `Err` forward sends neither; without EP neither is ever sent. Uniform k = 2, 3
//! and 4. `EpFake` records every call that matters to the worker's wire order.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{
    Model, ModelAdapters, ModelDeviceFeed, ModelDraft, ModelEp, ModelForward, ModelLifecycle,
    ModelLogits, ModelSsmState, ModelStreams, ModelVerify, ModelVision, SequenceState,
    VerifyBatchedOpts,
};

use super::levers::SchedLevers;
use super::sched_ctx::SchedCtx;
use super::test_support::active_seq;
use super::think_end_token0_tests::ctx;
use super::types::ActiveSeq;
use super::verify_dflash_batch_step::step_verify_dflash_batched;

/// 2026-10-09: What the fake's batched forward returns.
#[derive(Clone, Copy, PartialEq)]
enum Fwd {
    /// 2026-10-09: Every row; sequence 0 accepts every draft, sequence 1 none.
    Ok,
    /// 2026-10-09: One row short.
    Short,
    /// 2026-10-09: An error (`decode_verify_batched` sent the abort words).
    Err,
}

struct EpFake {
    ep: bool,
    fwd: Fwd,
    fold_fails: bool,
    /// 2026-10-09: The OR the workers report at the gather.
    worker_mask: u32,
    trace: Mutex<Vec<String>>,
}

const BONUS: u32 = 900;
const MISS: u32 = 800;

fn drafts(i: usize, k: usize) -> Vec<u32> {
    (0..k - 1).map(|j| (10 * (i + 1) + j) as u32).collect()
}

impl EpFake {
    fn new(ep: bool, fwd: Fwd) -> Self {
        Self {
            ep,
            fwd,
            fold_fails: false,
            worker_mask: 0,
            trace: Mutex::new(Vec::new()),
        }
    }
    fn rec(&self, s: String) {
        self.trace.lock().unwrap().push(s);
    }
    fn trace(&self) -> Vec<String> {
        self.trace.lock().unwrap().clone()
    }
}

impl Model for EpFake {}

impl ModelLifecycle for EpFake {
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn cache_sequence(&self, _s: &SequenceState) {}
    fn free_sequence(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _slot: usize) -> Result<()> {
        Ok(())
    }
    fn detach_slot_for_reuse(&self, _s: &mut SequenceState) {}
    fn num_free_blocks(&self) -> usize {
        1 << 20
    }
    fn num_total_blocks(&self) -> usize {
        1 << 20
    }
    fn reclaim_prefix_blocks(&self, _n: usize) -> usize {
        0
    }
}

impl ModelForward for EpFake {
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused")
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused")
    }
    fn prefill_chunk(
        &self,
        _t: &[u32],
        _s: &mut SequenceState,
        _cs: usize,
        _cl: usize,
        _last: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        anyhow::bail!("unused")
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        anyhow::bail!("unused")
    }
}

impl ModelLogits for EpFake {
    fn vocab_size(&self) -> usize {
        0
    }
    fn copy_logits_to_host(&self, _p: DevicePtr, _d: &mut [u8]) -> Result<()> {
        Ok(())
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _p: DevicePtr, _st: u64) -> Result<u32> {
        anyhow::bail!("unused")
    }
    fn argmax_batch(&self, _p: DevicePtr, n: usize, _st: u64) -> Result<Vec<u32>> {
        Ok(vec![0; n])
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
}

impl ModelAdapters for EpFake {}

impl ModelSsmState for EpFake {
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }
    fn commit_accepted_prefix(&self, s: &mut SequenceState, n: usize, k: usize) -> Result<()> {
        self.rec(format!("commit slot={} rows={n}/{k}", s.slot_idx));
        Ok(())
    }
}

impl ModelVerify for EpFake {
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unused")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        anyhow::bail!("unused")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        anyhow::bail!("unused")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        anyhow::bail!("unused")
    }
    fn decode_verify_batched(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &mut [&mut SequenceState],
        _st: u64,
        _o: VerifyBatchedOpts,
    ) -> Result<Vec<u32>> {
        self.rec(format!("verify ks={ks:?}"));
        if self.fwd == Fwd::Err {
            anyhow::bail!("fake forward error");
        }
        // 2026-10-09: The `decode_verify_batched` contract on Ok: each sequence advanced by k.
        let mut out = Vec::new();
        let mut off = 0;
        for (i, (s, &k)) in seqs.iter_mut().zip(ks).enumerate() {
            s.tokens.extend_from_slice(&tokens[off..off + k]);
            s.seq_len += k;
            off += k;
            if i == 0 {
                out.extend(drafts(0, k));
                out.push(BONUS);
            } else {
                out.extend(std::iter::repeat_n(MISS, k));
            }
        }
        if self.fwd == Fwd::Short {
            out.pop();
        }
        Ok(out)
    }
    fn gdn_fold_accepted(&self, _s: &[usize], rows: &[u32], k: usize) -> Result<bool> {
        self.rec(format!("fold rows={rows:?} k={k}"));
        if self.fold_fails {
            anyhow::bail!("fake fold error");
        }
        Ok(false)
    }
}

impl ModelDraft for EpFake {
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &metrale_sampling::SamplingParams,
        _n: usize,
    ) -> Result<metrale_model_engine::engine::GenerateResult> {
        anyhow::bail!("unused")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        anyhow::bail!("unused")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        n: usize,
        s: &mut SequenceState,
        _st: u64,
        _bm: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.rec(format!("propose slot={}", s.slot_idx));
        Ok(vec![1; n])
    }
    fn run_mtp_propose_batched(
        &self,
        tokens: &[u32],
        _p: &[usize],
        _si: &[usize],
        n: usize,
        _s: &mut [&mut SequenceState],
        _st: u64,
        _c: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        self.rec(format!("propose_batched n={}", tokens.len()));
        Ok(Some(vec![vec![1; n]; tokens.len()]))
    }
    fn mtp_propose_batch_max(&self) -> usize {
        8
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn has_proposer(&self) -> bool {
        true
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused")
    }
    fn save_hidden_for_mtp(&self, _i: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn save_hidden_for_mtp_from_stash(&self, _i: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn stash_verify_hidden_rows(&self, rows: &[usize], _st: u64) -> Result<()> {
        self.rec(format!("stash rows={rows:?}"));
        Ok(())
    }
    fn commit_ctx(&self, s: &mut SequenceState, n: usize, _b: usize, row: usize) -> Result<()> {
        self.rec(format!("commit_ctx slot={} n={n} row={row}", s.slot_idx));
        Ok(())
    }
    fn dflash_capture_band(&self) -> usize {
        8
    }
}

impl ModelVision for EpFake {
    fn tokens_contain_vision_pad(&self, _t: &[u32]) -> bool {
        false
    }
}

impl ModelEp for EpFake {
    fn is_ep(&self) -> bool {
        self.ep
    }
    fn ep_broadcast_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        self.rec(format!("verdict {tokens:?}"));
        Ok(tokens.to_vec())
    }
    fn ep_agree_verify_failures(&self, local: u32) -> Result<u32> {
        self.rec(format!("gather {local:#x}"));
        Ok(local | self.worker_mask)
    }
}

impl ModelStreams for EpFake {}

impl ModelDeviceFeed for EpFake {}

/// 2026-10-09: One batched DFlash step over two sequences at uniform `k`. Returns the trace
/// and the sequences.
fn run(model: EpFake, k: usize) -> (Vec<String>, Vec<ActiveSeq>) {
    let model = Arc::new(model);
    let mut sched = SchedCtx::for_test_with(model.clone());
    let mut levers = SchedLevers::defaults();
    levers.dflash_unified_ctx = true;
    sched.levers = Arc::new(levers);
    let mut seqs: Vec<ActiveSeq> = (0..2)
        .map(|i| {
            let (mut a, _rx) = active_seq(i, 3);
            a.pending_drafts = drafts(i, k);
            a
        })
        .collect();
    let c = ctx(&sched, &seqs[0]);
    {
        let mut refs: Vec<&mut ActiveSeq> = seqs.iter_mut().collect();
        step_verify_dflash_batched(&*model, &mut refs, &sched, k - 1, k - 1, &c);
    }
    (model.trace(), seqs)
}

fn pos(t: &[String], prefix: &str) -> Vec<usize> {
    t.iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with(prefix))
        .map(|(i, _)| i)
        .collect()
}

/// 2026-10-09: Exactly one verdict and one gather, the verdict first, and no propose before
/// the gather.
fn assert_one_handshake(t: &[String], words: &str, mask: &str, ctx: &str) {
    let v = pos(t, "verdict");
    let g = pos(t, "gather");
    assert_eq!(v.len(), 1, "{ctx}: one verdict broadcast: {t:#?}");
    assert_eq!(g.len(), 1, "{ctx}: one failure gather: {t:#?}");
    assert_eq!(t[v[0]], format!("verdict {words}"), "{ctx}");
    assert_eq!(t[g[0]], format!("gather {mask}"), "{ctx}");
    assert!(v[0] < g[0], "{ctx}: verdict before gather: {t:#?}");
    assert!(
        pos(t, "verify")[0] < v[0],
        "{ctx}: verdict after the forward"
    );
    assert!(
        pos(t, "propose").iter().all(|&p| p > g[0]),
        "{ctx}: no propose before the gather: {t:#?}"
    );
}

#[test]
fn ep_normal_tail_sends_verdict_then_gather_before_propose_at_k_2_3_4() {
    for k in 2..=4 {
        let (t, seqs) = run(EpFake::new(true, Fwd::Ok), k);
        let c = format!("k={k}");
        assert_eq!(t[0], format!("verify ks=[{k}, {k}]"), "{c}");
        assert_one_handshake(&t, &format!("[{}, 0]", k - 1), "0x0", &c);
        // 2026-10-09: The verdict goes out before rank 0's ctx commit, fold and SSM commit.
        let v = pos(&t, "verdict")[0];
        for p in ["commit_ctx", "fold", "commit slot", "stash"] {
            assert!(
                pos(&t, p).iter().all(|&i| i > v),
                "{c}: {p} after the verdict"
            );
        }
        assert!(
            t.contains(&format!("commit slot=0 rows={k}/{k}")),
            "{c}: {t:#?}"
        );
        assert!(
            t.contains(&format!("commit slot=1 rows=1/{k}")),
            "{c}: {t:#?}"
        );
        assert!(
            t.contains(&"propose_batched n=2".to_string()),
            "{c}: {t:#?}"
        );
        // 2026-10-09: From `seq_len` 7 (`active_seq(_, 3)`): sequence 0 kept all k rows,
        // sequence 1 only its anchor.
        assert_eq!(seqs[0].seq.seq_len, 7 + k, "{c}");
        assert_eq!(seqs[1].seq.seq_len, 7 + 1, "{c}");
        assert!(seqs.iter().all(|a| !a.finished), "{c}");
        assert_eq!(seqs[0].last_token, BONUS, "{c}");
        assert_eq!(seqs[1].last_token, MISS, "{c}");
    }
}

#[test]
fn ep_worker_failure_retires_that_sequence_before_propose() {
    for k in 2..=4 {
        let mut m = EpFake::new(true, Fwd::Ok);
        m.worker_mask = 0b10;
        let (t, seqs) = run(m, k);
        assert_one_handshake(&t, &format!("[{}, 0]", k - 1), "0x0", &format!("k={k}"));
        assert!(!seqs[0].finished && seqs[1].finished, "k={k}");
        assert!(!t.iter().any(|l| l == "propose slot=1"), "k={k}: {t:#?}");
    }
}

#[test]
fn ep_short_result_sends_zero_verdict_and_an_all_failed_gather() {
    for k in 2..=4 {
        let (t, seqs) = run(EpFake::new(true, Fwd::Short), k);
        assert_one_handshake(&t, "[0, 0]", "0xffffffff", &format!("k={k}"));
        assert!(seqs.iter().all(|a| a.finished), "k={k}");
        assert!(
            pos(&t, "propose").is_empty() && pos(&t, "commit").is_empty(),
            "k={k}"
        );
    }
}

#[test]
fn ep_fold_error_still_makes_the_gather() {
    for k in 2..=4 {
        let mut m = EpFake::new(true, Fwd::Ok);
        m.fold_fails = true;
        let (t, seqs) = run(m, k);
        assert_one_handshake(
            &t,
            &format!("[{}, 0]", k - 1),
            "0xffffffff",
            &format!("k={k}"),
        );
        assert!(seqs.iter().all(|a| a.finished), "k={k}");
        assert!(pos(&t, "propose").is_empty(), "k={k}");
    }
}

#[test]
fn ep_forward_error_sends_nothing_more_and_ends_the_batch() {
    for k in 2..=4 {
        let (t, seqs) = run(EpFake::new(true, Fwd::Err), k);
        assert_eq!(t, [format!("verify ks=[{k}, {k}]")], "k={k}");
        assert!(seqs.iter().all(|a| a.finished), "k={k}");
    }
}

#[test]
fn without_ep_nothing_goes_on_the_wire() {
    for k in 2..=4 {
        for fwd in [Fwd::Ok, Fwd::Short] {
            let (t, _) = run(EpFake::new(false, fwd), k);
            assert!(
                pos(&t, "verdict").is_empty() && pos(&t, "gather").is_empty(),
                "{t:#?}"
            );
        }
        // 2026-10-09: One GPU keeps the retry: the drafts go back and nothing finishes.
        let (t, seqs) = run(EpFake::new(false, Fwd::Err), k);
        assert_eq!(t.len(), 1, "k={k}");
        assert!(seqs.iter().all(|a| !a.finished), "k={k}");
        assert_eq!(seqs[1].pending_drafts, drafts(1, k), "k={k}");
    }
}
