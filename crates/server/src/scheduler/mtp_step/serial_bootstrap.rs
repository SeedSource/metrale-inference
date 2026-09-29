// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `step_mtp`'s per-sequence bootstrap for a sequence without
//! pending drafts: the DFlash direct propose, else a standalone decode,
//! sample and emit, then the next propose.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Bootstrap one sequence. Each early `return` ends this
/// sequence's bootstrap; `step_mtp` moves on to the next one.
pub(super) fn bootstrap_seq(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    ladder_nd: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    // 2026-09-25: DFlash with `dflash_seam_serial` off
    // (`METRALE_DFLASH_SEAM_SERIAL=0`; it is on by default) and
    // speculation allowed: skip the standalone decode. Propose drafts
    // directly (one under a grammar) and hand them to
    // `step_verify_k4`/`k3`/`k2` by draft count.
    if dflash_verify_raw_argmax
        && !sched.levers.dflash_seam_serial
        && crate::scheduler::adaptive_spec::spec_allowed(a, sched)
    {
        let eff = if a.grammar_state.is_some() {
            1
        } else {
            num_drafts
        };
        let _gmask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            eff,
            &mut a.seq,
            0,
            _gmask.as_deref(),
        ) {
            Ok(init) if !init.is_empty() => {
                if eff >= 3 && init.len() >= 3 {
                    step_verify_k4(
                        model,
                        a,
                        sched,
                        &init,
                        num_drafts,
                        verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                } else if eff >= 2 && init.len() >= 2 {
                    step_verify_k3(
                        model,
                        a,
                        sched,
                        &init,
                        num_drafts,
                        verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                } else {
                    step_verify_k2(
                        model,
                        a,
                        sched,
                        &init,
                        num_drafts,
                        verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                }
                return;
            }
            Ok(_) => {
                tracing::warn!(target: "met::scheduler::mtp_step", "DFlash bootstrap propose returned empty; falling back to standalone decode"
                );
            }
            Err(e) => {
                tracing::error!(target: "met::scheduler::mtp_step", "DFlash bootstrap propose: {e:#}");
            }
        }
        // 2026-09-25: Propose failed or returned no drafts: fall through
        // to the standalone decode below.
    }

    // 2026-09-25: Standalone decode. Under EP the token is broadcast to
    // the worker first (`ep_broadcast_cmd_for_seq`).
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
        tracing::error!(target: "met::scheduler::mtp_step", "EP broadcast bootstrap token: {e:#}");
        a.finished = true;
        return;
    }
    let logits = match model.decode(a.last_token, &mut a.seq, 0) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(target: "met::scheduler::mtp_step", "bootstrap decode error: {e:#}");
            a.finished = true;
            return;
        }
    };
    // 2026-09-29: A146, spec-in-think parity: a thinking row takes decode's
    // full host pipeline (forced `</think>`, mid-word mask, F2, pin), not the
    // penalties-only sampler below.
    let tok = if a.inside_thinking {
        match crate::scheduler::verify_pipeline_helper::pick_decode_row_with_pipeline(
            model, logits, a, verify_ctx,
        ) {
            Some(t) => t,
            None => {
                tracing::error!(target: "met::scheduler::mtp_step", "bootstrap in-think pipeline pick: D2H failed");
                a.finished = true;
                return;
            }
        }
    } else {
        // 2026-09-25: The sequence's penalties from `penalty_params_for`, the
        // builder the non-MTP decode path also uses. Built before
        // `grammar_state` is borrowed mutably below.
        //
        // 2026-09-29: A144: the bootstrap token carries the base `logit_bias`
        // decode would apply at this position (the tools-active `<tool_call>`
        // nudge included); a non-empty bias blocks `sample_token_with_grammar`'s
        // device-argmax path via `classify_penalties`, so it is applied on host.
        let base_bias = crate::scheduler::sample_step::speculative_base_logit_bias(
            a,
            0,
            verify_ctx.think_end_token,
            sched.levers.think_ended_gpu_argmax,
            || model.argmax_on_device(logits, 0).unwrap_or(u32::MAX),
        );
        let penalties = crate::scheduler::sample_step::penalty_params_for(
            a,
            crate::scheduler::sample_step::PositionKind::Verify,
            0.0,
            None,
            base_bias,
            sched.watchdog.min_reasoning_floor,
        );
        // 2026-09-25: Penalty history scoped to the current tool-call
        // segment (`penalty_history_scope`), as in the logits pipeline.
        let history = crate::scheduler::sample_step::penalty_history_scope(
            &a.output_tokens,
            a.tool_call_end_token,
        )
        .to_vec();
        // 2026-09-25: The sampler's min_p is `penalties.min_p`, copied from
        // `a.min_p` (the request's min_p raised to MODEL.toml `min_p_floor`
        // in `sampling_setup`). `METRALE_NO_MTP_MINP=1` passes 0.0 instead
        // (`effective_min_p`).
        match sample_token_with_grammar(
            model,
            logits,
            a.temperature,
            a.top_k,
            a.top_p,
            &[],
            a.grammar_state.as_mut(),
            &penalties,
            &history,
            &sched.levers.sampling(),
        ) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(target: "met::scheduler::mtp_step", "bootstrap sample error: {e:#}");
                a.finished = true;
                return;
            }
        }
    };

    let lp = if let Some(k) = a.top_logprobs {
        extract_single_logprobs(model, logits, tok, k)
    } else {
        None
    };

    emit_token(a, tok, lp, sched);
    if a.finished {
        return;
    }
    a.last_token = tok;
    // 2026-09-25: Adaptive speculation: count serial tokens toward the
    // re-probe window.
    crate::scheduler::adaptive_spec::tick_serial(a, sched);

    // 2026-09-25: Drafter context for this token (the unified ctx
    // commit when `dflash_unified_ctx`, else the serial append under
    // `METRALE_DFLASH_SERIAL_APPEND`) is added here only when the propose
    // below will not run, or on the step that resumes speculation after
    // an adaptive suspension (`reprobe_resume`). `spec_allowed` mutates
    // re-probe state, so it is evaluated once here and its verdict is
    // reused for the propose gate.
    let was_suspended = crate::scheduler::adaptive_spec::is_suspended(a, sched);
    let will_propose = crate::scheduler::adaptive_spec::spec_allowed(a, sched);
    let reprobe_resume = was_suspended && will_propose;
    if sched.levers.dflash_unified_ctx {
        if !will_propose || reprobe_resume {
            let base_pos = a.seq.seq_len.saturating_sub(1);
            if let Err(e) = model.commit_ctx(&mut a.seq, 1, base_pos, 0) {
                tracing::error!(target: "met::scheduler::mtp_step", "commit_ctx (mtp serial): {e:#}");
            }
        }
    } else if sched.levers.dflash_serial_append
        && (!will_propose || reprobe_resume)
        && let Err(e) = model.dflash_serial_ctx_append(&mut a.seq)
    {
        tracing::error!(target: "met::scheduler::mtp_step", "dflash_serial_ctx_append: {e:#}");
    }

    if let Err(e) = model.save_hidden_for_mtp(0, 0) {
        tracing::error!(target: "met::scheduler::mtp_step", "save_hidden_for_mtp: {e:#}");
        return;
    }
    let _mtp_grammar_mask = mtp_grammar_mask_for(a);
    // 2026-09-25: Under a grammar, propose one draft
    // (`effective_drafts_under_grammar`): `run_mtp_propose_multi`
    // (`MtpHead::propose`, mtp_head/draft_proposer.rs) applies the same
    // position-0 grammar bitmask to every draft position, so a later
    // draft can violate its own position's mask. One draft verifies on
    // the K=2 arm. Without a grammar the propose is sized to this step's
    // `ladder_nd`.
    let effective_num_drafts =
        crate::scheduler::spec_step::effective_drafts_under_grammar(a, ladder_nd);
    // 2026-09-25: A sequence suspended by adaptive speculation does not
    // propose, so it stays on this bootstrap path until `spec_allowed`
    // re-probes.
    if will_propose {
        match model.run_mtp_propose_multi(
            tok,
            a.seq.seq_len,
            effective_num_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(drafts) if !drafts.is_empty() => {
                tracing::debug!(target: "met::scheduler::mtp_step", "MTP bootstrap: tok={tok} → drafts={drafts:?}");
                a.pending_drafts = drafts;
            }
            Ok(_) => {
                tracing::warn!(target: "met::scheduler::mtp_step", "MTP propose returned empty")
            }
            Err(e) => {
                tracing::error!(target: "met::scheduler::mtp_step", "run_mtp_propose_multi: {e:#}");
            }
        }
    }

    if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
        tracing::error!(target: "met::scheduler::mtp_step", "bootstrap start_checkpoint_async: {e:#}");
    }
}
