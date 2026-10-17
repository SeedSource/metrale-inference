// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Prompt-token logprobs for `/v1/completions` with `echo` and `logprobs`.
//!
//! During prefill the hidden buffer holds one row per processed position, but only
//! the last row is normally projected to logits. When `seq.collect_prompt_logprobs`
//! is set, this projects every scored position through the final norm and LM head in
//! batches of up to 32 rows, copies each batch to the host, and records
//! `log P(tokens[i+1] | tokens[..=i])` with the top-k alternatives.
//!
//! It runs in `prefill_chunk_dispatch` before `prefill_b_finalize_last`, because it
//! overwrites `buffers.logits()` and `buffers.norm_output()`, which `finalize_last`
//! then recomputes for the first sampled token. Collecting requests skip the prefix
//! cache (`prefix_lookup.rs`), so every position has a computed hidden row.
//!
//! Owner: model-engine prefill.
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};

use super::super::super::types::TransformerModel;
use crate::traits::{SequenceState, extract_bf16};
use metrale_model_layers::layers::ops;

/// 2026-09-25: Rows projected per LM-head batch. A batch never exceeds `proc_count`,
/// which fits the arena's `m` rows, and `buffers.logits()` holds at least
/// `min(m, 160)` rows (gpu-runtime `buffers/sizes.rs`).
const BATCH_ROWS: usize = 32;

impl TransformerModel {
    /// 2026-09-25: Collect prompt-token logprobs for one prefill chunk. No-op unless
    /// `seq.collect_prompt_logprobs` is set. Appends to `seq.prompt_logprobs` one entry
    /// per position, scoring the next prompt token; the final prompt position, whose
    /// target is the first generated token, is excluded. Errors when the chunk was not
    /// fully recomputed (`proc_start != chunk_start`).
    pub(in crate::model) fn collect_prompt_logprobs_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        proc_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(k) = seq.collect_prompt_logprobs else {
            return Ok(());
        };
        // 2026-09-25: Hidden row r must be prompt position chunk_start + r; a partial
        // processing range would misalign the rows, so it is refused.
        ensure!(
            proc_start == chunk_start,
            "prompt-logprob collection requires a full-recompute chunk \
             (proc_start {proc_start} != chunk_start {chunk_start}); \
             prefix-cache bypass failed"
        );

        let h = self.config.hidden_size;
        let v = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;
        let elem = 2usize;
        let hidden = self.buffers.hidden_states();

        // 2026-09-25: Score positions chunk_start..min(chunk_end, prompt_len - 1):
        // position p's logits predict tokens[p + 1].
        let last_scored_excl = tokens.len().saturating_sub(1);
        let rows_to_score = proc_count.min(last_scored_excl.saturating_sub(chunk_start));
        if rows_to_score == 0 {
            return Ok(());
        }

        // 2026-10-17 (A191): Only this rank runs these windows, so the head must not
        // split the vocab: a split head's all-reduce has no peer and hangs the serve.
        let _local_head = crate::model::impl_a3_lm_head::RankLocalHead::enter();
        let mut host = vec![0u8; BATCH_ROWS * v * 2];
        let mut start = 0usize;
        while start < rows_to_score {
            let count = (rows_to_score - start).min(BATCH_ROWS);
            let batch_hidden = hidden.offset((start) * h * elem);
            let normed = self.buffers.norm_output();
            self.final_norm_apply(batch_hidden, normed, count as u32, h as u32, eps, stream)?;
            self.lm_head_batched(normed, count as u32, self.buffers.logits(), stream)?;
            self.gpu.synchronize(stream)?;
            let bytes = &mut host[..count * v * 2];
            self.gpu.copy_d2h(self.buffers.logits(), bytes)?;
            for j in 0..count {
                let target = tokens[chunk_start + start + j + 1];
                let row = &bytes[j * v * 2..(j + 1) * v * 2];
                seq.prompt_logprobs
                    .push(extract_bf16(row, target, k as usize, v));
            }
            start += count;
        }
        Ok(())
    }
}
