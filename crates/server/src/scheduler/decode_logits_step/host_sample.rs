// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The host-sampling arm of `process_decode_logits_skipping`: read
//! the logits block back when the step returned device tokens, then sample
//! every live row on the host, serially or on the rayon pool.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;
use crate::scheduler::io::{Effect, EffectOutcome};

/// 2026-09-26: Sample every row of `active` from host logits. `None` when the
/// logits readback failed; every row has then been told and drained.
pub(super) fn sample_on_host(
    active: &mut Vec<ActiveSeq>,
    staging: &mut Vec<u8>,
    logits: DevicePtr,
    host_elem: Option<usize>,
    n: usize,
    model: &dyn Model,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    skipped: &(impl Fn(usize) -> bool + Sync),
) -> Option<Vec<(u32, Option<crate::api::TokenLogprobs>)>> {
    let vocab_size = model.vocab_size();
    let t_copy = sched.io.clock.now();
    let elem_bytes = if let Some(e) = host_elem {
        e
    } else {
        match sched.io.dev.apply(Effect::ReadLogits {
            logits,
            rows: n,
            into: staging,
        }) {
            Ok(EffectOutcome::HostLogits { elem_bytes }) => elem_bytes,
            Ok(_) => unreachable!("ReadLogits answers HostLogits"),
            Err(e) => {
                tracing::error!(target: "met::scheduler::decode_logits_step", "copy_logits_to_host error: {e}");
                for mut a in active.drain(..) {
                    send_error(&sched.io, &mut a, &format!("{e}"));
                }
                return None;
            }
        }
    };
    let logits_fp32 = elem_bytes == 4;
    let buf = std::mem::take(staging);
    let copy_us = sched
        .io
        .clock
        .now()
        .saturating_duration_since(t_copy)
        .as_micros() as u64;
    let t_sample = sched.io.clock.now();
    // 2026-09-25: Rows are independent (a disjoint `buf` slice, their own
    // `ActiveSeq` and scratch, a per-sequence seed) and the collect keeps
    // order, so the parallel arm emits the same tokens as the serial one. It
    // runs only for `n > 1`, with `METRALE_PARALLEL_SAMPLE` on and no
    // `METRALE_LOGIT_DUMP` sink, whose per-step records would interleave.
    // `process_seq_logits` ignores its model argument, so no device call
    // crosses threads.
    let parallel_sample =
        n > 1 && sched.io.tel.dumps().logits.is_none() && sched.levers.parallel_sample;
    let sampled: Vec<(u32, Option<crate::api::TokenLogprobs>)> = if parallel_sample {
        use rayon::prelude::*;
        // 2026-09-25: `SchedCtx` is not `Sync` (its scratch is a `RefCell`), so the
        // closure captures only the pieces the pipeline reads.
        let tel = &*sched.io.tel;
        let clock = &*sched.io.clock;
        let watchdog = sched.watchdog;
        let boundary_mask = sched.masks.boundary.clone();
        let mid_word_mask = sched.masks.mid_word.clone();
        let sampling = sched.levers.sampling();
        let code_fence_token = sched.limits.code_fence_token;
        active
            .par_iter_mut()
            .enumerate()
            .map(|(i, a)| {
                if skipped(i) {
                    return (0, None);
                }
                PAR_SAMPLE_SCRATCH.with(|scratch| {
                    let ctx = crate::scheduler::logit_processors::LogitsContext {
                        think_end_token,
                        think_start_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        code_fence_token,
                        verify_pos: 0,
                        watchdog,
                        scratch,
                        tel,
                        clock,
                        boundary_mask: boundary_mask.clone(),
                        mid_word_mask: mid_word_mask.clone(),
                        sampling,
                    };
                    process_seq_logits(
                        model,
                        a,
                        &buf,
                        i,
                        vocab_size,
                        elem_bytes,
                        logits_fp32,
                        &ctx,
                        adaptive_sampling,
                    )
                })
            })
            .collect()
    } else {
        let ctx = logits_ctx(
            sched,
            &sched.scratch,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
        );
        active
            .iter_mut()
            .enumerate()
            .map(|(i, a)| {
                if skipped(i) {
                    return (0, None);
                }
                process_seq_logits(
                    model,
                    a,
                    &buf,
                    i,
                    vocab_size,
                    elem_bytes,
                    logits_fp32,
                    &ctx,
                    adaptive_sampling,
                )
            })
            .collect()
    };
    decode_timing_record(
        sched,
        copy_us,
        sched
            .io
            .clock
            .now()
            .saturating_duration_since(t_sample)
            .as_micros() as u64,
    );
    // 2026-09-25: Hand the staging buffer back for reuse. The error return above
    // skips this and only loses the buffer's capacity.
    *staging = buf;
    Some(sampled)
}
