// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched prefill step: advance every prefilling stream by one
//! chunk through `model.prefill_batch_chunk`, one call per wave
//! (`plan_prefill_waves`).
//!
//! Owner: scheduler.
//! Invariants:
//! - A stream that completes its last chunk, or whose wave fails, gets an
//!   entry in `completed_indices` (`None` on failure).

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{Model, PrefillSlice};

use super::super::types::PrefillInProgress;
use super::super::{FirstTokenPolicy, sample_first_token};
use super::prefill_waves::{WaveGeom, plan_prefill_waves};

pub(super) fn run_batched_prefill_step(
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    prefilling: &mut [PrefillInProgress],
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    prefill_stream: u64,
    prefill_event: u64,
    think_end_token: Option<u32>,
    tool_call_start_token: Option<u32>,
) {
    // 2026-09-25: InnerQ calibration poll (`poll_innerq`).
    super::poll_innerq(model);
    // 2026-09-25: Per-stream `chunk_len` and `is_last`, computed before the
    // slices borrow `p.seq`, so `chunk_offset` can advance after each
    // wave's call.
    let n = prefilling.len();
    let mut chunk_lens: Vec<usize> = Vec::with_capacity(n);
    let mut is_last_flags: Vec<bool> = Vec::with_capacity(n);
    // 2026-09-25: Varlen batched prefill (`--prefill-varlen-batch`) drives
    // the wave planner below and turns off the co-dispatch shared geometry:
    // when both it and `--prefill-codispatch` are set, varlen wins.
    let varlen = sched.levers.prefill_varlen;
    // 2026-09-25: Co-dispatch (`--prefill-codispatch`, not MLA): when two or
    // more streams are all at chunk 0 with equal prompt lengths, give them
    // one shared chunk geometry. Outside varlen, `check_kernel_batched_eligible`
    // (model-engine) admits a batch only when `chunk_len`, `chunk_start` and
    // `is_last` match across its streams. Other batches keep per-stream
    // geometry.
    let shared_geom: Option<(usize, bool)> = if !varlen
        && sched.levers.prefill_codispatch
        && !model.is_mla()
        && n >= 2
        && prefilling.iter().all(|p| p.chunk_offset == 0)
        && prefilling
            .iter()
            .all(|p| p.prompt_tokens.len() == prefilling[0].prompt_tokens.len())
    {
        let total = prefilling[0].prompt_tokens.len();
        // 2026-09-27: Non-last chunk ends as in `run_standard_chunk_loop`
        // (`prefill_plan::plan_chunk_len`); equal lengths give equal plans.
        let cl = metrale_model_engine::prefill_plan::plan_chunk_len_grid(
            0,
            total,
            total.min(max_prefill_tokens),
            model.kv_block_size(),
            model.prefill_tail_split(&prefilling[0].prompt_tokens),
            model.prefill_grid(&prefilling[0].prompt_tokens),
        );
        let is_last = cl >= total;
        Some((cl, is_last))
    } else {
        None
    };
    for p in prefilling.iter() {
        let (chunk_len, is_last) = if let Some((cl, il)) = shared_geom {
            (cl, il)
        } else {
            let remaining = p.prompt_tokens.len() - p.chunk_offset;
            // 2026-09-25: MLA: one chunk for the whole remaining prompt, as
            // in `run_standard_chunk_loop`.
            let effective_max = if model.is_mla() {
                remaining
            } else {
                max_prefill_tokens
            };
            let chunk_len = metrale_model_engine::prefill_plan::plan_chunk_len_grid(
                p.chunk_offset,
                p.prompt_tokens.len(),
                remaining.min(effective_max),
                model.kv_block_size(),
                model.prefill_tail_split(&p.prompt_tokens),
                model.prefill_grid(&p.prompt_tokens),
            );
            let is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();
            (chunk_len, is_last)
        };
        chunk_lens.push(chunk_len);
        is_last_flags.push(is_last);
    }

    // 2026-09-25: Waves (`plan_prefill_waves`). With varlen, streams sharing
    // `chunk_start` and `is_last` are grouped, at most `wave_cap` tokens per
    // multi-stream wave; without it, one wave holds every stream. The waves
    // run back to back in this call; a failed wave returns early, leaving
    // later waves for the next tick.
    let wave_cap = max_prefill_tokens.min(max_batch_tokens).max(1);
    let geoms: Vec<WaveGeom> = prefilling
        .iter()
        .enumerate()
        .map(|(i, p)| WaveGeom {
            chunk_start: p.chunk_offset,
            chunk_len: chunk_lens[i],
            is_last: is_last_flags[i],
        })
        .collect();
    let waves = plan_prefill_waves(&geoms, varlen, wave_cap);
    let n_waves = waves.len();
    if varlen {
        // 2026-09-25: One info line per call with the planned waves; M per
        // wave is the Σ `chunk_len` of its members.
        let wave_m: Vec<usize> = waves
            .iter()
            .map(|w| w.iter().map(|&i| chunk_lens[i]).sum())
            .collect();
        tracing::info!(
            "Varlen prefill waves: {n} streams -> {n_waves} wave(s), M per wave {wave_m:?} \
             (cap {wave_cap})"
        );
    }

    let t0_batch = sched.io.clock.now();
    for wave in waves {
        // 2026-09-25: Slices for this wave's members only: each borrows
        // `p.prompt_tokens` and `p.seq` of a distinct `PrefillInProgress`.
        let mut in_wave = vec![false; n];
        for &i in &wave {
            in_wave[i] = true;
        }
        let mut slices: Vec<PrefillSlice<'_>> = prefilling
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| in_wave[*i])
            .map(|(i, p)| PrefillSlice {
                prompt_tokens: &p.prompt_tokens,
                seq: &mut p.seq,
                chunk_start: p.chunk_offset,
                chunk_len: chunk_lens[i],
                is_last_chunk: is_last_flags[i],
            })
            .collect();

        let logits_per_stream = match model.prefill_batch_chunk(&mut slices, prefill_stream) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(
                    "Batched prefill error (wave of {} streams, {n} prefilling): {e:#}",
                    wave.len()
                );
                // 2026-09-25: Fail only this wave's streams (freed in
                // `promote_completed_prefills`). Later waves have not
                // advanced and retry next tick.
                for &i in &wave {
                    completed_indices.push((i, None));
                }
                return;
            }
        };
        // 2026-09-25: Release the `p.seq` borrows so `chunk_offset` can
        // advance below.
        drop(slices);

        // 2026-09-25: Make the default stream wait for the prefill stream
        // (`prefill_event`), as the single-stream path does.
        let _ = model.record_event(prefill_event, prefill_stream);
        let _ = model.stream_wait_event(model.default_stream(), prefill_event);

        debug_assert_eq!(
            logits_per_stream.len(),
            wave.len(),
            "prefill_batch_chunk returned wrong logit count"
        );

        // 2026-09-25: Advance offsets and sample first tokens before the next
        // wave dispatches: every `prefill_batch_chunk` call writes its logits
        // from row 0 (`prefill_batch_chunk_rows(.., 0)`, trait_impl/mod.rs).
        for (k, &i) in wave.iter().enumerate() {
            let p = &mut prefilling[i];
            p.chunk_offset += chunk_lens[i];
            if !is_last_flags[i] {
                continue;
            }
            let logits = logits_per_stream[k];
            if logits == DevicePtr::NULL {
                tracing::error!(
                    "Batched prefill: stream {i} marked is_last but model returned NULL logits",
                );
                completed_indices.push((i, None));
                continue;
            }
            // 2026-09-25: `sample_first_token` applies the sequence's `min_p`
            // (0.0 under `METRALE_NO_MTP_MINP=1`) and, when `FirstTokenPolicy`
            // lets it act on token 0, the grammar.
            match sample_first_token(
                model,
                logits,
                p.temperature,
                p.top_k,
                p.top_p,
                p.min_p,
                &p.eos_tokens,
                p.grammar_state.as_mut(),
                FirstTokenPolicy::for_birth(
                    p.enable_thinking,
                    think_end_token,
                    tool_call_start_token,
                ),
                &sched.levers.sampling(),
                sched.io.tel.dumps(),
            ) {
                Ok(first) => {
                    tracing::info!(
                        "Batched prefill[{i}/{n}] first token: {first} (chunk_len={}, total_tokens={})",
                        chunk_lens[i],
                        p.prompt_tokens.len(),
                    );
                    completed_indices.push((i, Some(first)));
                }
                Err(e) => {
                    tracing::error!("Batched prefill[{i}] sampling: {e:#}");
                    completed_indices.push((i, None));
                }
            }
        }
    }

    let elapsed = sched
        .io
        .clock
        .now()
        .saturating_duration_since(t0_batch)
        .as_micros();
    if elapsed > 1000 {
        tracing::debug!("Batched prefill step: {n} streams, {n_waves} waves, {elapsed}µs total");
    }
}
