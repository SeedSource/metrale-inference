// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl ModelForward for TransformerModel`, mostly delegating to `<method>_dispatch`
//! helpers in the sibling modules.
//!
//! Owner: model-engine.
//! Invariants: the ones in `trait_impl/mod.rs`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use crate::model::types::TransformerModel;
use crate::traits::{ModelForward, PrefillSlice, SequenceState};

impl ModelForward for TransformerModel {
    // 2026-09-25: `prefill`, `prefill_chunk`, `prefill_twophase` and
    // `mixed_forward` each call `try_eager_drafter_prefill` after the forward.
    // The whole-prompt drafter capture (`mtp_prefill_hidden`) is one slot shared
    // by all sequences, and a first chunk of any sequence restarts it, so it is
    // consumed while this sequence still owns it. `METRALE_NO_MTP_EAGER_DRAFTER`
    // turns the eager consume off.
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.gdn_carry_flush_pending()?;
        self.stamp_overlay_route(seq.adapter_slot);
        let logits = self.prefill_dispatch(tokens, seq, stream)?;
        self.try_eager_drafter_prefill(seq, true, stream);
        Ok(logits)
    }

    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.gdn_carry_flush_pending()?;
        self.stamp_overlay_route(seq.adapter_slot);
        let logits = self.prefill_chunk_dispatch(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            is_last_chunk,
            stream,
        )?;
        self.try_chunked_drafter_drain(tokens, seq, is_last_chunk, stream);
        self.try_eager_drafter_prefill(seq, is_last_chunk, stream);
        Ok(logits)
    }

    fn prefill_twophase(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.gdn_carry_flush_pending()?;
        self.stamp_overlay_route(seq.adapter_slot);
        let logits = self.prefill_twophase_dispatch(tokens, seq, chunk_size, stream)?;
        self.try_eager_drafter_prefill(seq, true, stream);
        Ok(logits)
    }

    fn decode(&self, token: u32, seq: &mut SequenceState, _stream: u64) -> Result<DevicePtr> {
        self.gdn_carry_flush_pending()?;
        self.stamp_overlay_route(seq.adapter_slot);
        self.stamp_decode_moe_single(seq.adapter_slot);
        self.decode_dispatch(token, seq, _stream)
    }

    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        self.gdn_carry_flush_pending()?;
        self.stamp_overlay_route_batch(seqs);
        self.stamp_decode_moe_batch(seqs);
        let r = self.decode_batch_dispatch(tokens, seqs, stream);
        if r.is_err() {
            // 2026-09-25: An error raised during graph capture (for example
            // `reject_decode_lora` refusing an adapter-routed MoE row) leaves the
            // capture open. End and discard it so the caller's cleanup does not
            // run on a capturing stream. The batched decode captures on the
            // default stream (decode_a2.rs).
            self.gpu.abort_capture_if_active(self.gpu.default_stream());
        }
        r
    }

    fn mixed_forward(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_tokens: &[u32],
        prefill_seq: &mut SequenceState,
        prefill_chunk_start: usize,
        prefill_chunk_len: usize,
        prefill_is_last: bool,
        stream: u64,
    ) -> Result<crate::traits::MixedForwardResult> {
        self.gdn_carry_flush_pending()?;
        // 2026-09-25: A mixed step always stamps `i32::MIN` (mixed adapters), so
        // the token-overlay hooks skip for the whole step.
        self.overlay_route_slot
            .store(i32::MIN, std::sync::atomic::Ordering::Relaxed);
        self.stamp_decode_moe_batch(decode_seqs);
        let r = self.mixed_forward_dispatch(
            decode_tokens,
            decode_seqs,
            prefill_tokens,
            prefill_seq,
            prefill_chunk_start,
            prefill_chunk_len,
            prefill_is_last,
            stream,
        );
        if r.is_err() {
            // 2026-09-25: The same capture release as in `decode_batch`.
            self.gpu.abort_capture_if_active(self.gpu.default_stream());
        }
        let out = r?;
        self.try_eager_drafter_prefill(prefill_seq, prefill_is_last, stream);
        Ok(out)
    }

    /// 2026-09-25: `prefill_batch_chunk_dispatch` sends a batch that is
    /// ineligible, or whose cache plan is not admitted, to its per-stream loop
    /// before any sequence is mutated. An error from an admitted kernel batch is
    /// returned: a per-stream retry could apply prefix-cache and KV state twice.
    fn prefill_batch_chunk(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        self.gdn_carry_flush_pending()?;
        self.prefill_batch_chunk_rows(streams, stream, 0)
    }

    /// 2026-09-25: Mixed-step variant: finishing streams write their logits from
    /// row `row_base`, clear of the decode lanes. The trait docs describe the
    /// aliasing this prevents.
    fn prefill_batch_chunk_rows(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
        row_base: usize,
    ) -> Result<Vec<DevicePtr>> {
        self.gdn_carry_flush_pending()?;
        self.prefill_batch_chunk_dispatch(streams, stream, row_base)
    }

    fn normalize_ssm_states(&self, seq: &SequenceState, stream: u64) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.normalize_ssm_states_dispatch(seq, stream)
    }

    fn hc_mult(&self) -> usize {
        self.config.hc_mult
    }

    fn is_mla(&self) -> bool {
        self.is_mla_dispatch()
    }

    fn kv_block_size(&self) -> Option<usize> {
        Some(self.kv_cache.lock().block_size())
    }

    fn prefill_tail_split(&self, tokens: &[u32]) -> Option<usize> {
        self.prefill_tail_split_dispatch(tokens)
    }
}
