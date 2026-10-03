// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched prefill for N concurrent streams, `prefill_batch_chunk_dispatch`.
//!
//! A batch that `kernel_batched_eligible` accepts, with `METRALE_Q12_BATCHED`
//! not set to 0 or false, goes to the kernel-batched path in `batch_kernel.rs`.
//! Otherwise, or when that path returns `NotAdmitted`, the streams run one
//! after another through the same step functions as `prefill_chunk_dispatch`,
//! with the KV-cache lock taken once for the whole loop.
//!
//! Owner: model-engine.
//! Invariants:
//! - In the per-stream loop a stream whose prefill fails gets
//!   `DevicePtr::NULL` and the remaining streams still run.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::batch_kernel::KernelBatchResult;
use super::proc_range::ProcRange;
use super::upload_meta::MetaLayout;
use crate::traits::{Model, PrefillSlice, SequenceState};

impl TransformerModel {
    /// 2026-09-25: Batched-prefill dispatch for N concurrent streams.
    ///
    /// On the per-stream path the result is parallel to `streams`: the stream's
    /// last-token logits pointer when its chunk is the last, `DevicePtr::NULL`
    /// otherwise or when the stream failed.
    ///
    /// `row_base` shifts each finishing stream's logits row to `row_base +
    /// stream_idx`. It is 0 for a prefill-only step and the number of decode
    /// rows inside a mixed step, whose rows `0..n_decode` belong to decode
    /// lanes (see `Model::prefill_batch_chunk_rows`).
    pub(in crate::model) fn prefill_batch_chunk_dispatch(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
        row_base: usize,
    ) -> Result<Vec<DevicePtr>> {
        let n = streams.len();
        // 2026-09-25: `METRALE_NO_PREFILL_ROW_SHIFT` set to 1 or true forces
        // `row_base = 0`, which puts prefill logits on the decode lanes' rows.
        // When the logits buffer cannot hold rows `row_base..row_base + n`,
        // `row_base` falls back to 0 with a warning rather than writing past it.
        let shift_disabled = std::env::var("METRALE_NO_PREFILL_ROW_SHIFT")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        let row_base = if shift_disabled { 0 } else { row_base };
        let logits_rows = self.buffers.sizes().logits / (self.config.vocab_size * 2);
        let row_base = if row_base + n > logits_rows {
            tracing::warn!(
                "Batched prefill: logits arena holds {logits_rows} rows, need \
                 row_base={row_base}+n={n}; falling back to row_base=0 \
                 (decode lanes may alias prefill rows this step)"
            );
            0
        } else {
            row_base
        };
        tracing::debug!(
            target: "metrale::q12",
            n = n,
            "prefill_batch_chunk_dispatch entry"
        );
        if n == 0 {
            return Ok(Vec::new());
        }
        // 2026-09-25: `prefill_chunk_dispatch` takes no logits row, so a single
        // stream uses it only when `row_base == 0`; otherwise it goes through
        // the per-stream loop, which passes the row.
        if n == 1 && row_base == 0 {
            let s = &mut streams[0];
            let logits = self.prefill_chunk_dispatch(
                s.prompt_tokens,
                s.seq,
                s.chunk_start,
                s.chunk_len,
                s.is_last_chunk,
                stream,
            )?;
            return Ok(vec![logits]);
        }

        let arena_cap = self.buffers.max_batch_tokens();
        for (i, s) in streams.iter().enumerate() {
            if s.chunk_len > arena_cap {
                anyhow::bail!(
                    "Batched prefill stream {i} chunk_len={} exceeds arena \
                     capacity {arena_cap}. Reduce --max-prefill-tokens.",
                    s.chunk_len
                );
            }
        }

        // 2026-09-25: Kernel-batched path, when `kernel_batched_eligible`
        // accepts the batch. `NotAdmitted` falls through to the per-stream loop
        // below; an error is returned to the caller. `METRALE_Q12_BATCHED` set
        // to 0 or false disables the path; unset or any other value leaves it on.
        let q12_batched_enabled = std::env::var("METRALE_Q12_BATCHED")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);
        if q12_batched_enabled && self.kernel_batched_eligible(streams) {
            tracing::debug!(
                target: "metrale::q12",
                n = n,
                chunk_len = streams[0].chunk_len,
                is_last_chunk = streams[0].is_last_chunk,
                "Q12 kernel-batched dispatch attempt"
            );
            match self.prefill_batch_chunk_kernel_batched(streams, stream, row_base) {
                Ok(KernelBatchResult::Completed(v)) => {
                    // 2026-09-25: Logged at info, once per completed kernel-batched
                    // prefill. `total_tokens` is the sum of `chunk_len`.
                    let total: usize = streams.iter().map(|s| s.chunk_len).sum();
                    tracing::info!(
                        target: "metrale::q12",
                        n = n,
                        total_tokens = total,
                        "Q12 kernel-batched prefill dispatched (fused large-M)"
                    );
                    return Ok(v);
                }
                Ok(KernelBatchResult::NotAdmitted) => {
                    tracing::info!(
                        target: "metrale::q12",
                        "Q12 kernel-batched cache plan not admitted → falling back to per-stream"
                    );
                }
                // 2026-09-25: An admitted batch can already own KV and sequence
                // state, and a per-stream retry would allocate or restore it a
                // second time, so the error is returned.
                Err(e) => return Err(e),
            }
        } else if !q12_batched_enabled {
            tracing::trace!(
                target: "metrale::q12",
                "Q12 kernel-batched disabled via METRALE_Q12_BATCHED=0"
            );
        } else {
            // 2026-09-25: Ineligible: log the batch shape, at info when varlen
            // batching is on (`varlen_prefill_enabled`) and at debug otherwise.
            let chunk_lens: Vec<usize> = streams.iter().map(|s| s.chunk_len).collect();
            let chunk_starts: Vec<usize> = streams.iter().map(|s| s.chunk_start).collect();
            let total: usize = chunk_lens.iter().sum();
            if super::batch_kernel::varlen_prefill_enabled() {
                tracing::info!(
                    target: "metrale::q12",
                    n = n,
                    chunk_lens = ?chunk_lens,
                    chunk_starts = ?chunk_starts,
                    total = total,
                    arena_cap = self.buffers.max_batch_tokens(),
                    head_dim = self.config.head_dim,
                    model_type = self.config.model_type.as_str(),
                    "Q12 kernel-batched ineligible — falling back to per-stream"
                );
            } else {
                tracing::debug!(
                    target: "metrale::q12",
                    n = n,
                    chunk_lens = ?chunk_lens,
                    chunk_starts = ?chunk_starts,
                    total = total,
                    arena_cap = self.buffers.max_batch_tokens(),
                    head_dim = self.config.head_dim,
                    model_type = self.config.model_type.as_str(),
                    "Q12 kernel-batched ineligible — falling back to per-stream"
                );
            }
        }

        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };

        let mut kv_cache = self.kv_cache.lock();

        let mut logits_out: Vec<DevicePtr> = Vec::with_capacity(n);

        for (stream_idx, slice) in streams.iter_mut().enumerate() {
            // 2026-09-25: A stream whose prefill fails gets NULL logits and the
            // loop goes on with the others. Each stream has its own sequence
            // state, and the next stream re-embeds the shared hidden buffer
            // from row 0.
            let stream_res: Result<DevicePtr> = (|| {
                let tokens = slice.prompt_tokens;
                let chunk_start = slice.chunk_start;
                let chunk_len = slice.chunk_len;
                let is_last_chunk = slice.is_last_chunk;
                let total = tokens.len();
                let seq = &mut *slice.seq;

                // 2026-09-25: The same zeroing rule as `prefill_chunk_dispatch`.
                if self.comm.is_some() {
                    self.buffers.zero_all(self.gpu.as_ref(), stream)?;
                } else if chunk_start == 0 {
                    self.buffers
                        .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
                }

                // 2026-09-25: Embed at row 0 of the shared hidden buffer. This
                // stream's layer loop consumes it before the next stream
                // overwrites it.
                self.prefill_b_embed_chunk(tokens, chunk_start, chunk_len, stream)?;

                // 2026-09-25: Prefix cache, EP agreement and Marconi restore.
                let (kv_write_start, marconi_skip) = self.prefill_b_prefix_lookup(
                    tokens,
                    seq,
                    chunk_start,
                    total,
                    &mut kv_cache,
                    stream,
                    None,
                )?;

                let bs = kv_cache.block_size();
                let end_pos = chunk_start + chunk_len;
                let blocks_needed = (end_pos - 1) / bs + 1;
                super::super::super::block_mgmt::ensure_blocks_through_prefill(
                    seq,
                    blocks_needed - 1,
                    &mut kv_cache,
                    self.prefix_cache.as_ref(),
                    self.gpu.as_ref(),
                    stream,
                    self.levers.kv_poison,
                )?;

                // 2026-09-25: Processing range; a fully cached non-last chunk
                // returns early.
                let (proc_start, proc_count, effective_seq_len_start) = match self
                    .prefill_b_proc_range(
                        tokens,
                        seq,
                        chunk_start,
                        chunk_len,
                        is_last_chunk,
                        kv_write_start,
                        marconi_skip,
                        // 2026-09-25: The hidden rows start at the buffer base.
                        self.buffers.hidden_states(),
                        stream,
                    )? {
                    ProcRange::Compute {
                        proc_start,
                        proc_count,
                        effective_seq_len_start,
                    } => (proc_start, proc_count, effective_seq_len_start),
                    ProcRange::EarlyReturn(ptr) => {
                        // 2026-09-25: A fully cached chunk still records its
                        // tokens, as in `prefill_chunk_dispatch`.
                        seq.tokens
                            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
                        seq.seq_len = chunk_start + chunk_len;
                        seq.last_decode_ckpt_block = seq.tokens.len() / bs;
                        return Ok(ptr);
                    }
                };

                // 2026-09-25: Positions, MRoPE and paged metadata.
                let MetaLayout {
                    meta_base,
                    slot_offset,
                    pos_stream_bytes,
                    use_mrope,
                    needs_paged,
                } = self.prefill_b_upload_meta(
                    tokens,
                    seq,
                    chunk_start,
                    chunk_len,
                    proc_start,
                    proc_count,
                    effective_seq_len_start,
                    &kv_cache,
                    stream,
                )?;

                if needs_paged {
                    self.prefill_b_upload_paged(
                        seq,
                        total,
                        proc_start,
                        proc_count,
                        meta_base,
                        slot_offset,
                        &kv_cache,
                        stream,
                    )?;
                }

                self.gpu.synchronize(stream)?;

                // 2026-10-03: The in-pass tail-split capture (`inpass_capture.rs`), `None`
                // unless `METRALE_GLM_SSM_INPASS_CAPTURE=1`.
                let inpass_plan = self.prepare_inpass_capture(
                    tokens,
                    seq,
                    &mut kv_cache,
                    proc_start,
                    proc_count,
                );

                // 2026-09-25: Forward through all layers.
                let forward = self.prefill_b_forward_layers(
                    seq,
                    &mut kv_cache,
                    chunk_start,
                    chunk_len,
                    is_last_chunk,
                    proc_count,
                    effective_seq_len_start,
                    kv_write_start,
                    marconi_skip,
                    meta_base,
                    slot_offset,
                    pos_stream_bytes,
                    use_mrope,
                    needs_paged,
                    // 2026-09-25: No mid-chunk tail capture on this path.
                    // 2026-10-03: Only the in-pass capture, when planned.
                    inpass_plan.as_ref(),
                    stream,
                );
                if let Some(plan) = inpass_plan.as_ref() {
                    if forward.is_err() {
                        self.ssm_snapshots.free(plan.snap_slot);
                    } else {
                        self.finalize_inpass_capture(tokens, seq, &mut kv_cache, plan, stream);
                    }
                }
                forward?;

                // 2026-09-25: Update the sequence state.
                seq.tokens
                    .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
                seq.seq_len = chunk_start + chunk_len;
                // 2026-09-25: Prime the decode-checkpoint gate (see prefill_a.rs).
                seq.last_decode_ckpt_block = seq.tokens.len() / bs;

                let logits = if is_last_chunk {
                    // 2026-09-25: The caller samples the logits after the whole
                    // loop, so each stream writes its own logits row,
                    // `row_base + stream_idx`, while its hidden row stays 0.
                    self.prefill_b_finalize_last_at(
                        tokens,
                        seq,
                        &mut kv_cache,
                        chunk_start,
                        chunk_len,
                        proc_count,
                        0,
                        row_base + stream_idx,
                        stream,
                    )?
                } else {
                    self.prefill_b_save_checkpoint(
                        tokens,
                        seq,
                        &mut kv_cache,
                        chunk_start,
                        chunk_len,
                        stream,
                    )?;
                    DevicePtr::NULL
                };
                Ok(logits)
            })();
            match stream_res {
                Ok(l) => logits_out.push(l),
                Err(e) => {
                    tracing::error!(
                        "Batched prefill fallback: stream {stream_idx} failed: {e:#} \
                         — isolating (NULL logits; only this stream fails, batch continues)"
                    );
                    logits_out.push(DevicePtr::NULL);
                }
            }
        }

        Ok(logits_out)
    }
}
