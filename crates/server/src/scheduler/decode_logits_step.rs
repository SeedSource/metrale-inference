// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`process_decode_logits`]: turn a decode step's logits into one token per
//! active sequence (device argmax or host sampling), then run each token's
//! bookkeeping: hard stops, thinking state, grammar, EOS handling, streaming,
//! length and context stops, and the fuzzy-repetition watchdog.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;
use crate::scheduler::io::{DecodeRows, Readback, StepOutcome};

mod content_emit;
mod host_sample;
mod per_token;
#[cfg(test)]
mod token0_tests;
#[cfg(test)]
mod tool_eos_hold_tests;

thread_local! {
    /// 2026-09-25: Dequant scratch for one rayon worker on the parallel host-sampling arm.
    ///
    /// The run's `SchedCtx::scratch` is a `RefCell`, which the workers cannot
    /// share, so each worker uses its own. `process_seq_logits` rebuilds
    /// `seq_f32` on every call, so nothing carries over between calls.
    static PAR_SAMPLE_SCRATCH: crate::scheduler::sched_ctx::DecodeScratch =
        crate::scheduler::sched_ctx::DecodeScratch::default();
}

/// 2026-09-25: Build the pre-sample pipeline's context around a chosen scratch buffer.
///
/// The serial host-sampling arm uses it with the run's scratch.
fn logits_ctx<'a>(
    sched: &'a crate::scheduler::sched_ctx::SchedCtx,
    scratch: &'a crate::scheduler::sched_ctx::DecodeScratch,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
) -> crate::scheduler::logit_processors::LogitsContext<'a> {
    crate::scheduler::logit_processors::LogitsContext {
        think_end_token,
        think_start_token,
        tool_call_start_token,
        tool_call_end_token,
        // 2026-09-29: A146: read only by the verify pick window.
        code_fence_token: sched.limits.code_fence_token,
        verify_pos: 0,
        watchdog: sched.watchdog,
        scratch,
        tel: &*sched.io.tel,
        clock: &*sched.io.clock,
        boundary_mask: sched.masks.boundary.clone(),
        mid_word_mask: sched.masks.mid_word.clone(),
        sampling: sched.levers.sampling(),
    }
}

/// 2026-09-25: `METRALE_DECODE_TIMING`: split the host path's per-token wall into `copy`
/// (the logits D2H, which also absorbs the wait for the forward) and `sample`
/// (dequant, pipeline and sampling on the host), logging a summary every 100
/// tokens. Returns at once when the lever is off.
fn decode_timing_record(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    copy_us: u64,
    sample_us: u64,
) {
    use std::sync::atomic::Ordering;
    if !sched.levers.decode_timing {
        return;
    }
    let stats = sched.io.tel.stats();
    stats.decode_copy_us.fetch_add(copy_us, Ordering::Relaxed);
    stats
        .decode_sample_us
        .fetch_add(sample_us, Ordering::Relaxed);
    let n = stats.decode_count.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_multiple_of(100) {
        let c = stats.decode_copy_us.swap(0, Ordering::Relaxed);
        let s = stats.decode_sample_us.swap(0, Ordering::Relaxed);
        stats.decode_count.store(0, Ordering::Relaxed);
        tracing::info!(
            "DECODE_TIMING (last 100 host-path tokens): copy+fwd-wait={:.2}ms/tok sample(248k host)={:.2}ms/tok",
            c as f64 / 100_000.0,
            s as f64 / 100_000.0,
        );
    }
}

/// 2026-09-25: Which readback the decode step needs, decided over the batch before it
/// runs: the device argmax when [`argmax_readback_eligible`], else the whole
/// block on the host.
pub(super) fn decode_readback_plan<'a>(
    active: &[ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    staging: &'a mut Vec<u8>,
) -> Readback<'a> {
    if argmax_readback_eligible(active.iter(), sched) {
        Readback::Argmax
    } else {
        Readback::HostLogits { into: staging }
    }
}

/// 2026-09-29: A144: would the single-row decode step emit this row straight
/// from the device argmax, with no host pipeline and so no penalties and no
/// `logit_bias`?
///
/// The per-row half of [`argmax_readback_eligible`] (the model-level
/// `decode_logits_fp32` term stays there): greedy temperature, no grammar, no
/// logprobs, the `min_tokens` floor met at `emitted_len`, and either outside
/// thinking entirely or a `think_ended` row with neutral penalties admitted by
/// `admit_think_ended` (`METRALE_NO_THINKENDED_GPU_ARGMAX` turns that off).
/// `emitted_len` is the output length at the position being decided:
/// `a.output_tokens.len()` on the decode path, plus `verify_pos` on the
/// speculative paths.
///
/// The speculative paths (verify, MTP bootstrap, DFlash) consult it to apply
/// `logit_bias` exactly when decode would, including where decode's device
/// argmax skips it. A `true` row still falls back to the host pipeline, bias
/// included, when the argmax lands on a post-think `</think>`/`<think>` id;
/// callers mirror that separately.
pub(super) fn decode_row_uses_gpu_argmax(
    a: &ActiveSeq,
    emitted_len: usize,
    admit_think_ended: bool,
) -> bool {
    // 2026-09-25: A `think_ended` row may still take the device argmax when it is
    // outside thinking, has no grammar and neutral penalties
    // (`METRALE_NO_THINKENDED_GPU_ARGMAX` turns this off). `PostCloseThinkMask`
    // masks two ids for such a row; `process_decode_logits_skipping` falls back
    // to the host when the device argmax lands on one of them.
    let think_ended_gpu_ok = a.think_ended
        && !a.inside_thinking
        && a.grammar_state.is_none()
        && a.repetition_penalty == 1.0
        && a.presence_penalty == 0.0
        && a.frequency_penalty == 0.0
        && a.lz_penalty == 0.0
        && a.dry_multiplier == 0.0;
    let excused = admit_think_ended && think_ended_gpu_ok;
    let row_needs_host =
        (a.inside_thinking || a.think_ended || a.grammar_state.is_some()) && !excused;
    a.temperature == 0.0
        && a.grammar_state.is_none()
        && a.top_logprobs.is_none()
        // 2026-09-25: GPU argmax bypasses the pre-sampling EOS mask. Keep requests with
        // an active minimum-token floor on the host pipeline.
        && a.min_tokens <= emitted_len
        && !row_needs_host
}

/// 2026-09-25: The predicate behind [`decode_readback_plan`]: may these rows take the
/// device argmax, or must the block come to the host? Also what the
/// pipelined lane asks of a batch before it runs a step ahead.
///
/// 2026-09-29: per-row eligibility is [`decode_row_uses_gpu_argmax`] (A144),
/// term for term the earlier `all(temperature == 0) && !any_grammar &&
/// !needs_host_logits` conjunction.
///
/// 2026-10-04: A103: with `METRALE_TOOL_EOS_HOLD_MASK` on, a row under the
/// tool-turn EOS hold needs the host, where its stop tokens are masked.
pub(super) fn argmax_readback_eligible<'a>(
    rows: impl Iterator<Item = &'a ActiveSeq> + Clone,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) -> bool {
    let mut active = rows;
    let model_logits_fp32 = sched.io.dev.model().decode_logits_fp32();
    let admit_think_ended = sched.levers.think_ended_gpu_argmax;
    let mask = sched.levers.tool_eos_hold_mask;
    !model_logits_fp32
        && active.all(|a| {
            decode_row_uses_gpu_argmax(a, a.output_tokens.len(), admit_think_ended)
                && !(mask && crate::scheduler::logit_processors::tool_eos_hold::applies(a))
        })
}

/// 2026-09-25: The readback for a decode block a lane produced beside its own forward
/// (the mixed lanes): the same plan and the same execution the
/// decode step's launch performs. `None` when the readback failed (every row
/// has been told).
pub(super) fn decode_readback_or_fail(
    active: &mut Vec<ActiveSeq>,
    logits: DevicePtr,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    staging: &mut Vec<u8>,
) -> Option<StepOutcome> {
    let model = sched.io.dev.model();
    let n = active.len();
    let readback = decode_readback_plan(active, sched, staging);
    match crate::scheduler::io::sync_device::execute_readback(model, logits, n, readback) {
        Ok(rows) => Some(StepOutcome::Decode { logits, rows }),
        Err(e) => {
            tracing::error!("argmax_batch error: {e:#}");
            for mut a in active.drain(..) {
                send_error(&sched.io, &mut a, &format!("{e:#}"));
            }
            None
        }
    }
}

/// 2026-09-25: Sample and process decode logits for all active sequences; called by
/// `step_decode_only` and by the mixed prefill lanes.
///
/// `step` is the decode step's outcome: `[n, vocab_size]` logits on device
/// (n = active.len()) plus their readback, which — for the host path — sits in
/// `staging` (handed back to the run's scratch on every exit but the error
/// paths).
pub fn process_decode_logits(
    active: &mut Vec<ActiveSeq>,
    step: StepOutcome,
    staging: &mut Vec<u8>,
    t0: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    process_decode_logits_skipping(
        active,
        step,
        staging,
        t0,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        adaptive_sampling,
        sched,
        &[],
    )
}

/// 2026-09-25: [`process_decode_logits`] with rows to leave untouched: `skip[i]` marks
/// a row whose result is an over-run of a pipelined step (the row finished
/// or was re-steered before its input existed) — it is neither sampled nor
/// emitted. Empty = every row is live, the synchronous case.
pub fn process_decode_logits_skipping(
    active: &mut Vec<ActiveSeq>,
    step: StepOutcome,
    staging: &mut Vec<u8>,
    t0: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    skip: &[bool],
) {
    let skipped = |i: usize| skip.get(i).copied().unwrap_or(false);
    let n = active.len();
    let model = sched.io.dev.model();
    let StepOutcome::Decode { logits, rows } = step;
    let host_elem = match &rows {
        DecodeRows::HostLogits { elem_bytes } => Some(*elem_bytes),
        DecodeRows::Tokens(_) => None,
    };
    // 2026-09-25: Take the device argmax unless, for a `think_ended` row, it landed on
    // one of the two masked think ids; then every row is sampled on the host.
    let fast_tokens: Option<Vec<(u32, Option<crate::api::TokenLogprobs>)>> = match rows {
        DecodeRows::Tokens(t) => {
            let hit_mask = t.iter().zip(active.iter()).any(|(&tok, a)| {
                a.think_ended && (Some(tok) == think_end_token || Some(tok) == a.think_start_token)
            });
            if hit_mask {
                // 2026-09-25: Counts steps that fell back to the host path because a device
                // argmax landed on a masked think token.
                sched
                    .think_mask_fallbacks
                    .set(sched.think_mask_fallbacks.get() + 1);
                None
            } else {
                Some(t.into_iter().map(|tok| (tok, None)).collect())
            }
        }
        DecodeRows::HostLogits { .. } => None,
    };

    let new_tokens: Vec<(u32, Option<crate::api::TokenLogprobs>)> = if let Some(t) = fast_tokens {
        t
    } else {
        // 2026-09-25: Host path: sample each row from host logits, read back here when
        // the step returned device tokens (the masked-argmax fallback).
        match host_sample::sample_on_host(
            active,
            staging,
            logits,
            host_elem,
            n,
            model,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            adaptive_sampling,
            sched,
            &skipped,
        ) {
            Some(sampled) => sampled,
            None => return,
        }
    };
    *sched.scratch.host_bytes.borrow_mut() = std::mem::take(staging);
    let step_ms = sched
        .io
        .clock
        .now()
        .saturating_duration_since(t0)
        .as_secs_f64()
        * 1000.0;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let token_ids: Vec<u32> = new_tokens.iter().map(|(t, _)| *t).collect();
        tracing::debug!(
            "DECODE: n={n} step={step_ms:.1}ms ({:.1} tok/s) tokens={:?}",
            1000.0 * n as f64 / step_ms,
            token_ids,
        );
    }

    let now = sched.io.clock.now();
    for (i, (tok, logprobs)) in new_tokens.into_iter().enumerate() {
        if skipped(i) {
            continue;
        }
        let a = &mut active[i];
        per_token::process_decoded_token(
            a,
            tok,
            logprobs,
            now,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            model,
            sched,
        );
    }
}
