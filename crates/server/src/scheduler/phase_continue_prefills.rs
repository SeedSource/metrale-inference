// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Continue in-progress prefills: one scheduler tick of prefill
//! work.
//!
//! Owner: scheduler.
//! Invariants:
//! - Every entry pushed to `completed_indices` reaches
//!   `promote_completed_prefills` before this returns.
//! - Returns true only after a mixed step (`mixed_forward` in
//!   `run_standard_chunk_loop`, or `run_batched_mixed_step`) succeeded; the
//!   caller then skips its decode lane for the tick.
//!
//! The paths live in sub-modules, which keeps each file under the 500-line
//! cap (`.github/workflows/file-size-cap.yml`):
//!
//!  - `run_batched_prefill` — two or more streams prefilling, no active
//!                            decode.
//!  - `run_batched_mixed`   — two or more streams prefilling, with active
//!                            decode.
//!  - `run_standard`        — one chunk of the head of `prefilling`, fused
//!                            with the active decode when it can be.
//!  - `prefill_waves`       — the wave planner for `run_batched_prefill`.
//!  - `spec_mixing`         — whether a speculative step blocks fusing.

#[path = "phase_continue_prefills/prefill_waves.rs"]
mod prefill_waves;
#[path = "phase_continue_prefills/run_batched_mixed.rs"]
mod run_batched_mixed;
#[path = "phase_continue_prefills/run_batched_prefill.rs"]
mod run_batched_prefill;
#[path = "phase_continue_prefills/run_standard.rs"]
mod run_standard;
mod spec_mixing;

use metrale_model_engine::traits::Model;

use super::phase_promote_prefills::promote_completed_prefills;
use super::types::{ActiveSeq, PrefillInProgress};
use super::{FirstTokenPolicy, sample_first_token};
use crate::scheduling_policy::{ActiveSeqTiming, SchedulingPolicy};

use run_batched_mixed::run_batched_mixed_step;
use run_batched_prefill::run_batched_prefill_step;
use run_standard::run_standard_chunk_loop;
use spec_mixing::mixing_blocked_by_spec;

/// 2026-09-25: Poll the model's InnerQ calibration (`Model::poll_innerq`)
/// once per prefill step; the standard, batched-prefill and batched-mixed
/// paths all call it. A model without an InnerQ driver does nothing.
pub(super) fn poll_innerq(model: &dyn Model) {
    model.poll_innerq();
}

#[allow(clippy::too_many_arguments)]
pub(super) fn continue_in_progress_prefills(
    model: &dyn Model,
    policy: &dyn SchedulingPolicy,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut Vec<PrefillInProgress>,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    always_mixed: bool,
    prefill_stream: u64,
    prefill_event: u64,
    use_mtp: bool,
    use_self_speculative: bool,
    use_ngram_speculative: bool,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) -> bool {
    let mut did_mixed_step = false;

    if prefilling.is_empty() {
        return did_mixed_step;
    }

    // 2026-09-25: Last-token times of the active sequences, for the
    // policy's TBT checks below.
    let timings: Vec<ActiveSeqTiming> = active
        .iter()
        .map(|a| ActiveSeqTiming {
            last_token_time: a.last_token_time,
        })
        .collect();

    // 2026-09-25: Whether a speculative step blocks fusing prefill into
    // decode this tick (`spec_mixing`); used by the always-mixed gate and
    // the batched-mixed gate below.
    let single_active_with_spec = mixing_blocked_by_spec(
        active.len(),
        use_mtp || use_self_speculative || use_ngram_speculative,
    );

    // 2026-09-25: `slice_budget` is this tick's prefill token budget.
    //
    // With `always_mixed` (`METRALE_HOLO_ALWAYS_MIXED`), a prefill that can
    // fuse into the active decode (`fusable_mixed`) runs even when
    // `should_prefill` says wait, sized by `prefill_slice_budget`. It returns
    // early only when the policy says wait and the chunk cannot fuse (EP, or
    // a speculative step this tick), or when the slice budget is 0.
    // Without it, `should_prefill` alone decides and the budget stays
    // `max_prefill_tokens`.
    let mut slice_budget = max_prefill_tokens;
    if always_mixed {
        // 2026-09-25: The conditions of `run_standard`'s `can_mix`, minus
        // `METRALE_BISECT_NO_MIX`: active decode, not EP, no speculative
        // step this tick.
        let fusable_mixed = !active.is_empty() && !model.is_ep() && !single_active_with_spec;
        let slas_ok = active.is_empty() || policy.should_prefill(sched.io.clock.now(), &timings);
        // 2026-09-25: The policy says wait and the chunk cannot fuse.
        if !slas_ok && !fusable_mixed {
            return did_mixed_step;
        }
        // 2026-09-25: Only a fusable step gets a slice budget; otherwise it
        // stays `max_prefill_tokens`.
        if fusable_mixed {
            // 2026-09-25: The policy's slice; 0 means no prefill this tick.
            slice_budget =
                policy.prefill_slice_budget(sched.io.clock.now(), &timings, max_prefill_tokens);
            // 2026-09-25: `METRALE_MIXED_SLICE_TOKENS` (default 0: no cap)
            // caps a nonzero slice; it never turns a 0 into a nonzero slice.
            // Measured 2026-08-27 on qwen4_exp: slice=256 on a 1598-token
            // prompt cut the co-tenant's worst gap from 2.8 to 1.3 s but
            // raised the prefill's TTFT from 3 to 10 s, which is why the
            // default is no cap.
            let cap = sched.levers.mixed_slice_tokens;
            if cap > 0 && slice_budget > 0 {
                slice_budget = slice_budget.min(cap);
            }
            // 2026-09-25: Slice 0: no prefill this tick.
            if slice_budget == 0 {
                return did_mixed_step;
            }
            // 2026-09-25: Keep padded decode rows plus the slice within
            // `max_batch_tokens`; above it `mixed_forward` runs decode and
            // prefill as separate forwards (trait_impl/decode_b.rs).
            let padded_n = metrale_model_engine::traits::padded_batch_n(active.len());
            let fuse_cap = max_batch_tokens.saturating_sub(padded_n).max(4);
            debug_assert!(
                fuse_cap >= 4,
                "fuse cap underflow: max_batch_tokens={max_batch_tokens} padded_n={padded_n}"
            );
            slice_budget = slice_budget.min(fuse_cap);
        }
    } else {
        // 2026-09-25: Without always-mixed: prefill only when no decode is
        // active or `should_prefill` allows it.
        let do_chunks = active.is_empty() || policy.should_prefill(sched.io.clock.now(), &timings);
        if !do_chunks {
            return did_mixed_step;
        }
    }

    let mut completed_indices = Vec::new();

    // 2026-09-25: Batched paths, for two or more prefilling streams: with no
    // active decode, `run_batched_prefill_step` (`prefill_batch_chunk`);
    // with active decode, `run_batched_mixed_step` (`mixed_forward_batch`).
    // Neither runs under EP, while a stream collects prompt logprobs, or
    // with `METRALE_BISECT_Q12_DISABLE=1`, which leaves the single-stream
    // path below. The mixed one also needs `always_mixed` off and no
    // speculative step this tick (`spec_mixing`).
    let q12_dispatch_disabled = sched.levers.bisect_q12_disable;
    // 2026-09-25: Prompt-logprob collection runs only on the single-stream
    // path, so a collecting stream keeps both batched paths off.
    let any_collecting = prefilling
        .iter()
        .any(|p| p.seq.collect_prompt_logprobs.is_some());
    let can_batch_prefill_only = !q12_dispatch_disabled
        && !any_collecting
        && prefilling.len() >= 2
        && active.is_empty()
        && !model.is_ep();
    // 2026-09-25: With `always_mixed`, several prefills plus active decode
    // take the single-stream path below instead: the head of `prefilling`
    // is fused with the active decode through `mixed_forward`, sized by the
    // slice budget, and the other streams wait for later ticks.
    let can_batch_mixed = !always_mixed
        && !q12_dispatch_disabled
        && !any_collecting
        && prefilling.len() >= 2
        && !active.is_empty()
        && !single_active_with_spec
        && !model.is_ep();

    if can_batch_prefill_only {
        run_batched_prefill_step(
            model,
            sched,
            prefilling,
            &mut completed_indices,
            max_prefill_tokens,
            max_batch_tokens,
            prefill_stream,
            prefill_event,
            think_end_token,
            tool_call_start_token,
        );
        promote_completed_prefills(
            model,
            &sched.io,
            prefilling,
            completed_indices,
            active,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            sched.limits.max_seq_len,
        );
        return did_mixed_step;
    }

    if can_batch_mixed {
        let t0_mixed = sched.io.clock.now();
        run_batched_mixed_step(
            model,
            active,
            prefilling,
            &mut completed_indices,
            max_prefill_tokens,
            prefill_stream,
            prefill_event,
            t0_mixed,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            adaptive_sampling,
            sched,
            &mut did_mixed_step,
        );
        promote_completed_prefills(
            model,
            &sched.io,
            prefilling,
            completed_indices,
            active,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            sched.limits.max_seq_len,
        );
        return did_mixed_step;
    }

    // 2026-09-25: Single-stream path: the head of `prefilling` advances by
    // one chunk (`run_standard_chunk_loop`), or by the whole prompt through
    // the two-phase prefill below.
    if let Some(p) = prefilling.first_mut() {
        let idx = 0usize;

        // 2026-09-25: Two-phase prefill (`Model::prefill_twophase`, where the
        // SSM recurrence sees the whole prompt in one launch) for a prompt
        // longer than one chunk that has not started chunking. It runs the
        // whole prompt in one call with no decode fused, so under
        // `always_mixed` it runs only when no decode is active.
        // 2026-10-01: Never on a multi-rank world. Two-phase sends the workers no
        // prefill command, and its prefix lookup (`prefill_c/marconi.rs`) skips the
        // F83 match agreement and the A100 restore vote. EP heads reach here only
        // with `chunk_offset > 0` today (`phase_start_prefills` never defers under
        // EP), so this changes nothing now; it keeps a new deferral path from
        // desyncing the ranks.
        let use_twophase = !model.is_ep()
            && (!always_mixed || active.is_empty())
            && p.chunk_offset == 0
            && p.prompt_tokens.len() > max_prefill_tokens;
        if use_twophase {
            tracing::info!(
                "Two-phase prefill: {} tokens, chunk_size={}",
                p.prompt_tokens.len(),
                max_prefill_tokens,
            );
            match model.prefill_twophase(
                &p.prompt_tokens,
                &mut p.seq,
                max_prefill_tokens,
                prefill_stream,
            ) {
                Ok(logits) => {
                    p.chunk_offset = p.prompt_tokens.len();
                    let _ = model.record_event(prefill_event, prefill_stream);
                    let _ = model.stream_wait_event(model.default_stream(), prefill_event);
                    // 2026-09-25: First token: `sample_first_token` applies
                    // the sequence's `min_p` (0.0 under
                    // `METRALE_NO_MTP_MINP=1`) and, when `FirstTokenPolicy`
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
                            tracing::info!("Two-phase prefill first token: {first}");
                            completed_indices.push((idx, Ok(first)));
                        }
                        Err(e) => {
                            tracing::error!("Two-phase prefill sampling: {e:#}");
                            completed_indices.push((idx, Err(format!("prefill failed: {e:#}"))));
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Two-phase prefill failed, falling back to chunked: {e:#}");
                }
            }
        }

        // 2026-09-25: Standard chunked prefill (also used as fallback if
        // two-phase fails).
        if p.chunk_offset < p.prompt_tokens.len() {
            run_standard_chunk_loop(
                model,
                p,
                idx,
                active,
                max_prefill_tokens,
                slice_budget,
                prefill_stream,
                prefill_event,
                use_mtp,
                use_self_speculative,
                use_ngram_speculative,
                think_end_token,
                think_start_token,
                code_fence_token,
                tool_call_start_token,
                tool_call_end_token,
                adaptive_sampling,
                sched,
                &mut completed_indices,
                &mut did_mixed_step,
            );
        }
    }

    // 2026-09-25: Move completed prefills to active (or free on error).
    promote_completed_prefills(
        model,
        &sched.io,
        prefilling,
        completed_indices,
        active,
        think_end_token,
        think_start_token,
        tool_call_start_token,
        tool_call_end_token,
        sched.limits.max_seq_len,
    );

    did_mixed_step
}
