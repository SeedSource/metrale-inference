// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched mixed step: the active decode plus one chunk of every
//! prefilling stream in one `model.mixed_forward_batch` call (the trait
//! default runs `decode_batch`, then `prefill_batch_chunk_rows`).
//!
//! Owner: scheduler.
//! Invariants:
//! - Sets `did_mixed_step` exactly when `mixed_forward_batch` succeeded; a
//!   failure pushes `(i, None)` for every prefilling stream.

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{Model, PrefillSlice, SequenceState};
use std::time::Instant;

use super::super::decode_logits_step::process_decode_logits;
use super::super::types::{ActiveSeq, PrefillInProgress};
use super::super::{FirstTokenPolicy, sample_first_token};

#[allow(clippy::too_many_arguments)]
pub(super) fn run_batched_mixed_step(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut [PrefillInProgress],
    completed_indices: &mut Vec<(usize, Result<u32, String>)>,
    max_prefill_tokens: usize,
    prefill_stream: u64,
    prefill_event: u64,
    t0_step: Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    did_mixed_step: &mut bool,
) {
    // 2026-09-25: InnerQ calibration poll (`poll_innerq`).
    super::poll_innerq(model);
    let n_prefill = prefilling.len();
    let n_decode = active.len();

    // 2026-09-25: Per-stream `chunk_len` and `is_last`: the MLA whole-prompt
    // chunk and the rounding to 4 tokens of `run_batched_prefill_step`, plus
    // the SSM tail-boundary clamp of `run_standard_chunk_loop`.
    let mut chunk_lens: Vec<usize> = Vec::with_capacity(n_prefill);
    let mut is_last_flags: Vec<bool> = Vec::with_capacity(n_prefill);
    for p in prefilling.iter() {
        let remaining = p.prompt_tokens.len() - p.chunk_offset;
        let effective_max = if model.is_mla() {
            remaining
        } else {
            max_prefill_tokens
        };
        let mut chunk_len = remaining.min(effective_max);
        // 2026-09-25: With `METRALE_SSM_TAIL_CKPT=1` and mid-chunk capture
        // off, end a chunk at `ssm_tail_boundary`, so an SSM snapshot lands
        // where the next turn's prefix match stops.
        if sched.levers.ssm_tail_ckpt
            && !sched.levers.ssm_tail_midchunk
            && let Some(bs) = model.kv_block_size()
            && let Some(tb) = metrale_gpu_runtime::ssm_tail_boundary(p.prompt_tokens.len(), bs)
            && p.chunk_offset < tb
            && p.chunk_offset + chunk_len > tb
        {
            chunk_len = tb - p.chunk_offset;
        }
        // 2026-09-27: Non-last chunk ends as in `run_standard_chunk_loop`
        // (`prefill_plan::plan_chunk_len`).
        let chunk_len = metrale_model_engine::prefill_plan::plan_chunk_len_grid(
            p.chunk_offset,
            p.prompt_tokens.len(),
            chunk_len,
            model.kv_block_size(),
            model.prefill_tail_split(&p.prompt_tokens),
            model.prefill_grid(&p.prompt_tokens),
        );
        let is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();
        chunk_lens.push(chunk_len);
        is_last_flags.push(is_last);
    }

    // 2026-09-25: Sort by ssm pool slot, as `decode_step.rs` does: the
    // batched GDN recurrence needs the sequences on consecutive pool slots
    // in slice order. Moving whole `ActiveSeq`s keeps each decode row
    // matched to its sequence.
    if active.len() > 1 {
        active.sort_by_key(|a| a.seq.ssm_slot_idx().unwrap_or(a.seq.slot_idx));
    }
    let decode_tokens: Vec<u32> = active.iter().map(|a| a.last_token).collect();

    // 2026-09-25: Build the slices in a temporary scope so the `&mut`
    // borrows of `prefilling` and `active` end before `active` is borrowed
    // again for `process_decode_logits`.
    let result = {
        let mut decode_refs: Vec<&mut SequenceState> =
            active.iter_mut().map(|a| &mut a.seq).collect();
        let mut prefill_slices: Vec<PrefillSlice<'_>> = prefilling
            .iter_mut()
            .enumerate()
            .map(|(i, p)| PrefillSlice {
                prompt_tokens: &p.prompt_tokens,
                seq: &mut p.seq,
                chunk_start: p.chunk_offset,
                chunk_len: chunk_lens[i],
                is_last_chunk: is_last_flags[i],
            })
            .collect();
        model.mixed_forward_batch(
            &decode_tokens,
            &mut decode_refs,
            &mut prefill_slices,
            prefill_stream,
        )
    };

    let result = match result {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                "Mixed-batch forward error (n_decode={n_decode}, n_prefill={n_prefill}): {e:#}",
            );
            for i in 0..n_prefill {
                completed_indices.push((i, Err(format!("prefill failed: {e:#}"))));
            }
            return;
        }
    };

    let _ = model.record_event(prefill_event, prefill_stream);
    let _ = model.stream_wait_event(model.default_stream(), prefill_event);

    // 2026-09-25: Advance prefill offsets and sample first tokens for
    // streams that just finished their last chunk.
    debug_assert_eq!(result.prefill_logits.len(), n_prefill);
    for (i, p) in prefilling.iter_mut().enumerate() {
        p.chunk_offset += chunk_lens[i];
        if !is_last_flags[i] {
            continue;
        }
        let logits = result.prefill_logits[i];
        if logits == DevicePtr::NULL {
            tracing::error!(
                "Mixed-batch: stream {i} marked is_last but model returned NULL logits"
            );
            completed_indices.push((i, Err("prefill returned no logits".to_string())));
            continue;
        }
        // 2026-09-25: `sample_first_token` applies the sequence's `min_p` (0.0
        // under `METRALE_NO_MTP_MINP=1`) and, when `FirstTokenPolicy` lets it
        // act on token 0, the grammar.
        match sample_first_token(
            model,
            logits,
            p.temperature,
            p.top_k,
            p.top_p,
            p.min_p,
            &p.eos_tokens,
            p.grammar_state.as_mut(),
            FirstTokenPolicy::for_birth(p.enable_thinking, think_end_token, tool_call_start_token),
            &sched.levers.sampling(),
            sched.io.tel.dumps(),
        ) {
            Ok(first) => {
                tracing::info!(
                    "Mixed-batch prefill[{i}/{n_prefill}] first token: {first} (chunk_len={}, total_tokens={})",
                    chunk_lens[i],
                    p.prompt_tokens.len(),
                );
                completed_indices.push((i, Ok(first)));
            }
            Err(e) => {
                tracing::error!("Mixed-batch prefill[{i}] sampling: {e:#}");
                completed_indices.push((i, Err(format!("prefill failed: {e:#}"))));
            }
        }
    }

    // 2026-09-25: Decode logits for the active sequences, handled as in
    // `run_standard_chunk_loop`'s mixed branch.
    if n_decode > 0 && result.decode_logits != DevicePtr::NULL {
        let mut staging = sched.scratch.host_bytes.borrow_mut().split_off(0);
        if let Some(step) = super::super::decode_logits_step::decode_readback_or_fail(
            active,
            result.decode_logits,
            sched,
            &mut staging,
        ) {
            process_decode_logits(
                active,
                step,
                &mut staging,
                t0_step,
                think_end_token,
                think_start_token,
                code_fence_token,
                tool_call_start_token,
                tool_call_end_token,
                adaptive_sampling,
                sched,
            );
        }
    }
    *did_mixed_step = true;
}
