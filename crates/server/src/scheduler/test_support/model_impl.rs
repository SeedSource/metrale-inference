// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `Model` and its supertraits for `PreemptStubModel`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{
    Model, ModelAdapters, ModelDeviceFeed, ModelDraft, ModelEp, ModelForward, ModelLifecycle,
    ModelLogits, ModelSsmState, ModelStreams, ModelVerify, ModelVision, SequenceState,
};
use std::sync::atomic::Ordering;

use super::PreemptStubModel;

impl Model for PreemptStubModel {}

impl ModelLifecycle for PreemptStubModel {
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn cache_sequence(&self, _s: &SequenceState) {
        self.cached_seqs.fetch_add(1, Ordering::SeqCst);
    }
    fn free_sequence(&self, s: &mut SequenceState) -> Result<()> {
        self.freed_slots.lock().unwrap().push(s.slot_idx);
        Ok(())
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _slot: usize) -> Result<()> {
        self.compact_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(msg) = self.fail_compact {
            anyhow::bail!("{msg}");
        }
        Ok(())
    }
    /// 2026-09-25: The trait default fails; the tests that swap out
    /// (`swap_out_tests`, `shutdown_drain_tests`) need the save to succeed,
    /// so this writes one byte and returns Ok.
    fn save_sequence_state(
        &self,
        _seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        writer.write_all(&[0u8])?;
        Ok(())
    }
    fn detach_slot_for_reuse(&self, _s: &mut SequenceState) {}
    fn num_free_blocks(&self) -> usize {
        self.free_blocks.load(Ordering::SeqCst)
    }
    fn num_total_blocks(&self) -> usize {
        self.total_blocks
    }
    fn reclaim_prefix_blocks(&self, num_blocks: usize) -> usize {
        let take = num_blocks.min(self.reclaimable.load(Ordering::SeqCst));
        self.reclaimable.fetch_sub(take, Ordering::SeqCst);
        self.free_blocks.fetch_add(take, Ordering::SeqCst);
        take
    }
}

impl ModelForward for PreemptStubModel {
    fn kv_block_size(&self) -> Option<usize> {
        self.block_size
    }
    fn prefill(&self, t: &[u32], s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        self.prefilled.lock().unwrap().push(t.to_vec());
        // 2026-09-25: like the engine's prefill, record the prompt in tokens,
        // seq_len and prompt_len.
        s.tokens.extend_from_slice(t);
        s.seq_len = s.tokens.len();
        s.prompt_len = t.len();
        Ok(DevicePtr::NULL)
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        // 2026-10-04: only a logits stub (`logit_rows`) decodes.
        if self.logit_rows.is_empty() {
            anyhow::bail!("unused in preempt tests")
        }
        Ok(DevicePtr::NULL)
    }
    fn prefill_chunk(
        &self,
        t: &[u32],
        s: &mut SequenceState,
        cs: usize,
        cl: usize,
        _last: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        // 2026-10-04: only a first-token stub (`first_token`) prefills.
        if self.first_token.is_none() {
            anyhow::bail!("unused in preempt tests")
        }
        s.tokens.extend_from_slice(&t[cs..cs + cl]);
        s.seq_len = s.tokens.len();
        Ok(DevicePtr::NULL)
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        self.decode_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(msg) = self.hard_error {
            anyhow::bail!("{msg}");
        }
        if self
            .fail_decodes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
        {
            anyhow::bail!("KV cache exhausted: no free blocks");
        }
        Ok(DevicePtr::NULL)
    }
}

/// 2026-10-04: The first index of the row's maximum.
fn row_argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

impl ModelLogits for PreemptStubModel {
    fn vocab_size(&self) -> usize {
        self.logit_rows.first().map_or(0, Vec::len)
    }
    /// 2026-10-04: BF16 (each f32's high half) from the rows laid end to
    /// end, starting at the pointer's element offset; a no-op without rows.
    fn copy_logits_to_host(&self, p: DevicePtr, d: &mut [u8]) -> Result<()> {
        let src = self.logit_rows.iter().flatten().skip(p.0 as usize / 2);
        for (dst, v) in d.chunks_exact_mut(2).zip(src) {
            dst.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        Ok(())
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, p: DevicePtr, _st: u64) -> Result<u32> {
        let row = p.0 as usize / 2 / self.vocab_size().max(1);
        match self.logit_rows.get(row) {
            Some(r) => Ok(row_argmax(r)),
            None => self
                .first_token
                .ok_or_else(|| anyhow::anyhow!("unused in preempt tests")),
        }
    }
    fn argmax_batch(&self, _p: DevicePtr, n: usize, _st: u64) -> Result<Vec<u32>> {
        if self.logit_rows.is_empty() {
            return Ok(vec![0; n]);
        }
        Ok(self
            .logit_rows
            .iter()
            .take(n)
            .map(|r| row_argmax(r))
            .collect())
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
}

impl ModelAdapters for PreemptStubModel {}

impl ModelSsmState for PreemptStubModel {
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }
}

impl ModelVerify for PreemptStubModel {
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unused in preempt tests")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        anyhow::bail!("unused in preempt tests")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        anyhow::bail!("unused in preempt tests")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        anyhow::bail!("unused in preempt tests")
    }
}

impl ModelDraft for PreemptStubModel {
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &metrale_sampling::SamplingParams,
        _n: usize,
    ) -> Result<metrale_model_engine::engine::GenerateResult> {
        anyhow::bail!("unused in preempt tests")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        anyhow::bail!("unused in preempt tests")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _bm: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("unused in preempt tests")
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused in preempt tests")
    }
    fn save_hidden_for_mtp(&self, _i: usize, _st: u64) -> Result<()> {
        Ok(())
    }
}

impl ModelVision for PreemptStubModel {
    fn tokens_contain_vision_pad(&self, tokens: &[u32]) -> bool {
        self.vision_pad
            .map(|pad| tokens.contains(&pad))
            .unwrap_or(false)
    }
}

impl ModelEp for PreemptStubModel {}

impl ModelStreams for PreemptStubModel {}

impl ModelDeviceFeed for PreemptStubModel {}
