// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelForward`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use super::{BeamReq, ModelStreams};
use crate::traits::{MixedBatchResult, MixedForwardResult, PrefillSlice, SequenceState};
use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::DevicePtr;

/// 2026-09-26: Prefill, decode and mixed forward passes, beam search, and the model-shape queries
/// `is_mla`, `hc_mult` and `kv_block_size`.
pub trait ModelForward: ModelStreams {
    /// 2026-09-25: Whether this model implements [`Self::generate_beam_batch`]. Default `false`;
    /// `NllbGpuModel` returns `true`.
    fn supports_beam(&self) -> bool {
        false
    }

    /// 2026-09-25: Run beam search to completion for each request and return each winning
    /// hypothesis. The scheduler calls it at prefill for `num_beams > 1`, instead of the
    /// token-by-token decode loop. Default: an error.
    fn generate_beam_batch(&self, _reqs: &[BeamReq]) -> Result<Vec<Vec<u32>>> {
        bail!("this model does not support beam search")
    }

    /// 2026-09-25: Prefill the whole prompt and return the logits of its last position.
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;

    /// 2026-09-25: Prefill `chunk_len` prompt tokens from `chunk_start`. Only a chunk with
    /// `is_last_chunk` returns logits (of the last prompt position); any other chunk returns
    /// `DevicePtr::NULL`.
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr>;

    /// 2026-09-25: One decode step for one sequence; returns the logits for the next token.
    fn decode(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;

    /// 2026-09-25: One decode step for each sequence; returns logits `[seqs.len(), vocab]`.
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr>;

    /// 2026-09-25: One decode step for each of `decode_seqs` plus one prefill chunk. Default:
    /// `decode_batch` (skipped when there are no decode tokens), then `prefill_chunk`.
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
    ) -> Result<MixedForwardResult> {
        let decode_logits = if !decode_tokens.is_empty() {
            self.decode_batch(decode_tokens, decode_seqs, stream)?
        } else {
            metrale_gpu_runtime::gpu::DevicePtr::NULL
        };
        let prefill_logits = self.prefill_chunk(
            prefill_tokens,
            prefill_seq,
            prefill_chunk_start,
            prefill_chunk_len,
            prefill_is_last,
            stream,
        )?;
        Ok(MixedForwardResult {
            decode_logits,
            prefill_logits,
        })
    }

    /// 2026-09-25: Prefill one chunk for each of `streams`. Returns one entry per stream, in
    /// order: the last-token logits when that stream's chunk is its last, else
    /// `DevicePtr::NULL`. Default: `prefill_chunk` per stream.
    fn prefill_batch_chunk(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        let mut out = Vec::with_capacity(streams.len());
        for slice in streams.iter_mut() {
            let logits = self.prefill_chunk(
                slice.prompt_tokens,
                slice.seq,
                slice.chunk_start,
                slice.chunk_len,
                slice.is_last_chunk,
                stream,
            )?;
            out.push(logits);
        }
        Ok(out)
    }

    /// 2026-09-25: [`Self::prefill_batch_chunk`], with finishing stream `i`'s logits written to
    /// row `row_base + i` of the shared logits arena instead of row `i`.
    ///
    /// `decode_batch` writes lane `i`'s logits to row `i` of the same arena, and in a mixed
    /// step the caller samples the decode rows after the prefill has run. Without the shift a
    /// finishing prefill stream overwrites a decode lane's logits, and that lane samples
    /// another request's first-token distribution. `TransformerModel` falls back to
    /// `row_base = 0` when `row_base + streams.len()` exceeds the arena's rows. Default: ignore
    /// `row_base`.
    fn prefill_batch_chunk_rows(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
        _row_base: usize,
    ) -> Result<Vec<DevicePtr>> {
        self.prefill_batch_chunk(streams, stream)
    }

    /// 2026-09-25: One decode step for each of `decode_seqs` plus one prefill chunk for each of
    /// `prefill_streams`. Default: `decode_batch`, a synchronise of the default stream, then
    /// `prefill_batch_chunk_rows` with the prefill rows placed after the decode rows.
    fn mixed_forward_batch(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<MixedBatchResult> {
        let decode_logits = if !decode_tokens.is_empty() {
            let lg = self.decode_batch(decode_tokens, decode_seqs, stream)?;
            // 2026-09-25: The batched prefill below reuses the decode pass's arena buffers on
            // `stream`, so the decode work on the default stream must finish first.
            self.synchronize(self.default_stream())?;
            lg
        } else {
            metrale_gpu_runtime::gpu::DevicePtr::NULL
        };
        // 2026-09-25: Decode owns rows `0..decode_tokens.len()`, so the prefill rows start
        // there (see `prefill_batch_chunk_rows`).
        let prefill_logits =
            self.prefill_batch_chunk_rows(prefill_streams, stream, decode_tokens.len())?;
        Ok(MixedBatchResult {
            decode_logits,
            prefill_logits,
        })
    }

    /// 2026-09-25: Normalise the sequence's SSM `h_state`. The scheduler calls it between
    /// prefill chunks. Default: no-op.
    fn normalize_ssm_states(&self, _seq: &SequenceState, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Prefill the whole prompt in one call, with `chunk_size` as the chunk length
    /// when an implementation splits it. Returns the last position's logits. Default: one
    /// `prefill_chunk` over all of `tokens`.
    fn prefill_twophase(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_chunk(tokens, seq, 0, tokens.len(), true, stream)
    }

    /// 2026-09-25: Whether chunked prefill must run as one chunk. `TransformerModel` answers
    /// `requires_single_chunk_prefill`: MLA models (`kv_lora_rank > 0`) except `glm5_next` and
    /// `kimi_k3`. Default `false`.
    fn is_mla(&self) -> bool {
        false
    }

    /// 2026-09-25: mHC hyper-connection stream count (`config.hc_mult` in `TransformerModel`);
    /// `0` means no highway. Default `0`.
    fn hc_mult(&self) -> usize {
        0
    }

    /// 2026-09-25: Tokens per paged-KV block, or `None` without a paged KV cache. The scheduler
    /// uses it to end a prefill chunk at `metrale_gpu_runtime::ssm_tail_boundary`. Default
    /// `None`.
    fn kv_block_size(&self) -> Option<usize> {
        None
    }

    /// 2026-09-27: The token at which the prefill of `tokens` is split so that an SSM
    /// snapshot lands at `prefill_plan::tail_split_point`, or `None` when this model does
    /// not split it. The scheduler ends a non-last chunk there
    /// (`prefill_plan::plan_chunk_len`). Default `None`.
    fn prefill_tail_split(&self, _tokens: &[u32]) -> Option<usize> {
        None
    }

    /// 2026-10-03: The absolute prefill grid size `G` for `tokens`
    /// (`METRALE_PREFIX_GRID_RESTORE=1`), or `None`. With `Some(G)` the scheduler plans every
    /// chunk with `prefill_plan::plan_chunk_len_grid`, so non-last chunks end on multiples of
    /// `G`. Default `None`.
    fn prefill_grid(&self, _tokens: &[u32]) -> Option<usize> {
        None
    }
}
