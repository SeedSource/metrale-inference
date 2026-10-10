// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: A single-rank fake `Model` that drives `step_mtp` on a DFlash serve: the batched
//! and the per-sequence verify both accept every draft, the proposers return fresh drafts, and
//! the logits rows hold a one-hot at each row's verified token so the host pipeline picks what
//! the raw argmax picked. Every call that tells the tests which route ran is traced.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{
    Model, ModelAdapters, ModelDeviceFeed, ModelDraft, ModelEp, ModelForward, ModelLifecycle,
    ModelLogits, ModelSsmState, ModelStreams, ModelVerify, ModelVision, SequenceState,
    VerifyBatchedOpts,
};

/// 2026-10-09: Vocabulary of the fake's logits rows (BF16).
pub(super) const VOCAB: usize = 8192;
/// 2026-10-09: First token id the fake hands out (drafts and bonuses); below the think ids.
const FIRST_TOKEN: u32 = 2000;

pub(super) struct StepFake {
    /// 2026-10-09: Verified token per row of the last forward (the logits one-hots).
    rows: Mutex<Vec<u32>>,
    next: AtomicU32,
    trace: Mutex<Vec<String>>,
}

impl StepFake {
    pub(super) fn new() -> Self {
        Self {
            rows: Mutex::new(Vec::new()),
            next: AtomicU32::new(FIRST_TOKEN),
            trace: Mutex::new(Vec::new()),
        }
    }
    fn rec(&self, s: String) {
        self.trace.lock().unwrap().push(s);
    }
    /// 2026-10-09: The trace since the last call, cleared.
    pub(super) fn take_trace(&self) -> Vec<String> {
        std::mem::take(&mut *self.trace.lock().unwrap())
    }
    pub(super) fn fresh(&self, n: usize) -> Vec<u32> {
        (0..n)
            .map(|_| self.next.fetch_add(1, Ordering::Relaxed))
            .collect()
    }
    /// 2026-10-09: Accept-all verdict for one sequence's `[last, d0..]`: the drafts, then a
    /// fresh bonus.
    fn verdict(&self, toks: &[u32]) -> Vec<u32> {
        let mut v = toks[1..].to_vec();
        v.extend(self.fresh(1));
        v
    }
}

impl Model for StepFake {}

impl ModelLifecycle for StepFake {
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

impl ModelForward for StepFake {
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

impl ModelLogits for StepFake {
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    /// 2026-10-09: `p` is `logits_buffer_ptr() + row * VOCAB * 2`; each requested row gets a
    /// BF16 1.0 at its verified token.
    fn copy_logits_to_host(&self, p: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let row0 = p.0 as usize / (VOCAB * 2);
        let n = dst.len() / (VOCAB * 2);
        self.rec(format!("d2h row={row0} n={n}"));
        dst.fill(0);
        let rows = self.rows.lock().unwrap();
        for r in 0..n {
            let tok = rows[row0 + r] as usize;
            let at = (r * VOCAB + tok) * 2;
            dst[at..at + 2].copy_from_slice(&0x3F80u16.to_le_bytes());
        }
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

impl ModelAdapters for StepFake {}

impl ModelSsmState for StepFake {
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }
    fn commit_accepted_prefix(&self, _s: &mut SequenceState, _n: usize, _k: usize) -> Result<()> {
        Ok(())
    }
}

impl ModelVerify for StepFake {
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
    fn decode_verify_dflash(
        &self,
        tokens: &[u32],
        s: &mut SequenceState,
        _st: u64,
    ) -> Result<Vec<u32>> {
        self.rec(format!("perseq k={}", tokens.len()));
        s.tokens.extend_from_slice(tokens);
        s.seq_len += tokens.len();
        let v = self.verdict(tokens);
        *self.rows.lock().unwrap() = v.clone();
        Ok(v)
    }
    fn can_batch_verify(&self, _ks: &[usize]) -> bool {
        true
    }
    fn decode_verify_batched(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &mut [&mut SequenceState],
        _st: u64,
        _o: VerifyBatchedOpts,
    ) -> Result<Vec<u32>> {
        self.rec(format!("batched ks={ks:?}"));
        let mut out = Vec::new();
        let mut off = 0;
        for (s, &k) in seqs.iter_mut().zip(ks) {
            s.tokens.extend_from_slice(&tokens[off..off + k]);
            s.seq_len += k;
            out.extend(self.verdict(&tokens[off..off + k]));
            off += k;
        }
        *self.rows.lock().unwrap() = out.clone();
        Ok(out)
    }
    fn gdn_fold_accepted(&self, _s: &[usize], _rows: &[u32], _k: usize) -> Result<bool> {
        Ok(false)
    }
}

impl ModelDraft for StepFake {
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
        _s: &mut SequenceState,
        _st: u64,
        _bm: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        Ok(self.fresh(n))
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
        Ok(Some(tokens.iter().map(|_| self.fresh(n)).collect()))
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
    fn stash_verify_hidden_rows(&self, _rows: &[usize], _st: u64) -> Result<()> {
        Ok(())
    }
    fn commit_ctx(&self, _s: &mut SequenceState, _n: usize, _b: usize, _row: usize) -> Result<()> {
        Ok(())
    }
    fn dflash_capture_band(&self) -> usize {
        8
    }
}

impl ModelVision for StepFake {
    fn tokens_contain_vision_pad(&self, _t: &[u32]) -> bool {
        false
    }
}

impl ModelEp for StepFake {}

impl ModelStreams for StepFake {}

impl ModelDeviceFeed for StepFake {}
