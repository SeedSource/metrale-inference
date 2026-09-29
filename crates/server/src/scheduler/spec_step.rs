// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The self-speculative and N-gram speculative decoding steps, plus the grammar helpers the MTP propose sites use.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Self-speculative step for `active[0]`: drafts with `decode_draft`, which
/// skips the SSM layers, then verifies with the full model, all in one call.
///
/// `verify_ctx` feeds `verify_pick_all_with_pipeline`, which replaces each
/// verify position's raw argmax with the logits-processor pipeline's pick.
pub fn step_self_spec(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let a = &mut active[0];

    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
        tracing::error!("EP broadcast self-spec token: {e:#}");
        a.finished = true;
        return;
    }
    let logits = match model.decode(a.last_token, &mut a.seq, 0) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("self-spec decode error: {e:#}");
            a.finished = true;
            return;
        }
    };
    // 2026-09-29: A146, spec-in-think parity: decode never takes a device
    // argmax for a thinking row.
    let token_0 = match if a.inside_thinking {
        crate::scheduler::verify_pipeline_helper::pick_decode_row_with_pipeline(
            model, logits, a, verify_ctx,
        )
        .ok_or_else(|| anyhow::anyhow!("in-think pipeline pick: D2H failed"))
    } else {
        model.argmax_on_device(logits, 0)
    } {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("self-spec argmax error: {e:#}");
            a.finished = true;
            return;
        }
    };

    let seq_len_before_draft = a.seq.seq_len;
    let tokens_before_draft = a.seq.tokens.len();

    let mut draft_tokens = Vec::with_capacity(num_drafts);
    let mut draft_token = token_0;
    for _ in 0..num_drafts {
        let logits = match model.decode_draft(draft_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("self-spec draft error: {e:#}");
                break;
            }
        };
        draft_token = match model.argmax_on_device(logits, 0) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("self-spec draft argmax error: {e:#}");
                break;
            }
        };
        draft_tokens.push(draft_token);
    }

    // 2026-09-25: rewind to the pre-draft state; the SSM state needs no
    // rewind because `decode_draft` skips the SSM layers.
    a.seq.seq_len = seq_len_before_draft;
    a.seq.tokens.truncate(tokens_before_draft);

    if draft_tokens.is_empty() {
        emit_token(a, token_0, None, sched);
        if !a.finished {
            a.last_token = token_0;
        }
        return;
    }

    if let Err(e) = model.checkpoint_ssm_states(&mut a.seq) {
        tracing::error!("self-spec checkpoint: {e:#}");
        a.finished = true;
        return;
    }
    let seq_len_before_verify = a.seq.seq_len;

    let mut verify_tokens = vec![token_0];
    verify_tokens.extend_from_slice(&draft_tokens);

    let verified_argmax = match model.decode_verify(&verify_tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("self-spec verify error: {e:#}");
            a.finished = true;
            return;
        }
    };

    // 2026-09-25: replace each position's raw argmax with the
    // logits-processor pipeline's pick; the helper falls back to the raw
    // argmax if the logits copy fails.
    //
    // 2026-09-29: A146: commit token_0, THEN pick the window and accept.
    // Verify position 0 is the token AFTER token_0, so the window's base
    // state (history, thinking_tokens, pipeline accumulators) must already
    // include it; see `self_spec_commit`.
    let Some(num_accepted) = self_spec_commit(a, token_0, &draft_tokens, sched, |a| {
        crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &verified_argmax,
            a,
            verify_ctx,
            0,
        )
    }) else {
        return;
    };

    // 2026-09-25: drop the verify tokens past token_0 and the accepted
    // drafts.
    let tokens_added = 1 + num_accepted;
    let expected_seq_len = seq_len_before_verify + tokens_added;

    if a.seq.seq_len > expected_seq_len {
        let extra = a.seq.seq_len - expected_seq_len;
        for _ in 0..extra {
            a.seq.seq_len -= 1;
            a.seq.tokens.pop();
        }
        // 2026-09-25: +1 because token_0, the first verify token, is
        // always emitted.
        if let Err(e) = model.rollback_ssm_states(&mut a.seq, num_accepted + 1) {
            tracing::error!("self-spec rollback: {e:#}");
        }
    }
}

/// 2026-09-29: A146: the self-speculative commit: emit `token_0`, run the
/// verify window (`pick`) on the post-`token_0` state, then emit the
/// accepted drafts and the correction or bonus token. Returns the number of
/// accepted drafts, or `None` when the sequence finished mid-commit.
///
/// Spec-in-think parity: the window used to run BEFORE `token_0` was
/// committed, so every position's history and `thinking_tokens` were one
/// short (the mid-word and boundary previous token, the penalties, the F2
/// and budget gates) and the trail never matched at commit, resetting the
/// F2 and defer accumulators every step.
pub(crate) fn self_spec_commit(
    a: &mut ActiveSeq,
    token_0: u32,
    draft_tokens: &[u32],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    pick: impl FnOnce(&mut ActiveSeq) -> Vec<u32>,
) -> Option<usize> {
    emit_token(a, token_0, None, sched);
    if a.finished {
        return None;
    }
    let verified = pick(a);
    let n_drafts = draft_tokens.len();
    let mut num_accepted = 0;
    for i in 0..n_drafts {
        if draft_tokens[i] == verified[i] {
            emit_token(a, draft_tokens[i], None, sched);
            if a.finished {
                return None;
            }
            num_accepted += 1;
        } else {
            emit_token(a, verified[i], None, sched);
            if a.finished {
                return None;
            }
            a.last_token = verified[i];
            break;
        }
    }
    if num_accepted == n_drafts && n_drafts > 0 {
        emit_token(a, verified[n_drafts], None, sched);
        if !a.finished {
            a.last_token = verified[n_drafts];
        }
    } else if num_accepted < n_drafts {
        // 2026-09-25: a.last_token was already set above, in the break.
    } else {
        a.last_token = token_0;
    }
    Some(num_accepted)
}

/// 2026-09-25: N-gram speculative step for `active[0]`: CPU proposer, CUDA-graphed K=2 verify.
///
/// Each call runs one of two phases:
/// 1. Bootstrap (no pending draft): regular decode, argmax, then an N-gram
///    proposal into `pending_drafts`.
/// 2. Verify (a pending draft): `decode_verify_graphed` (K=2), accept or
///    reject, SSM rollback on reject.
pub fn step_ngram(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    proposer: &mut NgramProposer,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let a = &mut active[0];

    if !a.pending_drafts.is_empty() {
        let drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        a.pending_draft_conf.clear();
        step_ngram_verify(model, a, sched, &drafts, proposer, verify_ctx);
    } else {
        if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
            tracing::error!("EP broadcast ngram bootstrap: {e:#}");
            a.finished = true;
            return;
        }
        let logits = match model.decode(a.last_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("ngram bootstrap decode error: {e:#}");
                a.finished = true;
                return;
            }
        };
        // 2026-09-29: A146, spec-in-think parity: decode never takes a
        // device argmax for a thinking row.
        let tok = match if a.inside_thinking {
            crate::scheduler::verify_pipeline_helper::pick_decode_row_with_pipeline(
                model, logits, a, verify_ctx,
            )
            .ok_or_else(|| anyhow::anyhow!("in-think pipeline pick: D2H failed"))
        } else {
            model.argmax_on_device(logits, 0)
        } {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("ngram bootstrap argmax error: {e:#}");
                a.finished = true;
                return;
            }
        };

        proposer.observe(&a.seq.tokens, tok);

        emit_token(a, tok, None, sched);
        if a.finished {
            return;
        }
        a.last_token = tok;

        if let Some(draft) = proposer.propose(&a.seq.tokens) {
            a.pending_drafts = vec![draft];

            // 2026-09-25: checkpoint the SSM state so a rejected draft can be
            // rolled back in `step_ngram_verify`.
            if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
                tracing::error!("ngram start_checkpoint_async: {e:#}");
            }
        }
        // 2026-09-25: with no proposal, the next iteration is another
        // bootstrap (a regular decode).
    }
}

/// 2026-09-25: Verify a single N-gram draft via the CUDA-graphed K=2 path.
pub fn step_ngram_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    proposer: &mut NgramProposer,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let t_sync = sched.io.clock.now();
    if let Err(e) = model.sync_secondary() {
        tracing::error!("ngram sync_secondary: {e:#}");
        a.finished = true;
        return;
    }
    let sync_us = sched
        .io
        .clock
        .now()
        .saturating_duration_since(t_sync)
        .as_micros();

    // 2026-09-25: EP workers answer 0xFFFFFFF2 by reading the two tokens
    // below, running the same K=2 verify and then reading the verdict.
    let tokens_k2 = [a.last_token, drafts[0]];
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF2) {
        tracing::error!("EP broadcast ngram verify cmd: {e:#}");
        a.finished = true;
        return;
    }
    for &t in &tokens_k2 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast ngram verify token: {e:#}");
            a.finished = true;
            return;
        }
    }

    let t_verify = sched.io.clock.now();
    let result = match model.decode_verify_graphed(&tokens_k2, &mut a.seq, 0) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("ngram decode_verify_graphed: {e:#}");
            a.finished = true;
            return;
        }
    };
    let verify_us = sched
        .io
        .clock
        .now()
        .saturating_duration_since(t_verify)
        .as_micros();
    a.last_token_time = sched.io.clock.now();
    let [v0_argmax, v1_argmax] = result;

    // 2026-09-25: accept or reject against the logits-processor
    // pipeline's pick at each position, not the raw argmax.
    let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
        model,
        &[v0_argmax, v1_argmax],
        a,
        verify_ctx,
        0,
    );
    let v0 = processed.first().copied().unwrap_or(v0_argmax);
    let v1 = processed.get(1).copied().unwrap_or(v1_argmax);
    let accepted = drafts[0] == v0;

    if let Err(e) = model.ep_broadcast_cmd(accepted as u32) {
        tracing::error!("EP broadcast ngram verify result: {e:#}");
        a.finished = true;
        return;
    }

    if accepted {
        // 2026-09-25: accepted: emit the draft and v1.
        proposer.observe(&a.seq.tokens[..a.seq.tokens.len() - 1], drafts[0]);
        proposer.observe(&a.seq.tokens, v1);

        emit_token(a, drafts[0], None, sched);
        if !a.finished {
            emit_token(a, v1, None, sched);
        }
        if a.finished {
            return;
        }
        a.last_token = v1;

        if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
            tracing::error!("ngram accept checkpoint: {e:#}");
        }

        if let Some(draft) = proposer.propose(&a.seq.tokens) {
            a.pending_drafts = vec![draft];
        }

        if a.seq.seq_len.is_multiple_of(50) {
            tracing::info!(
                "NGRAM K2 ACCEPT: sync={sync_us}μs verify={verify_us}μs cache={} seq_len={}",
                proposer.len(),
                a.seq.seq_len,
            );
        }
    } else {
        // 2026-09-25: rejected: `decode_verify_graphed` appended both
        // tokens, so drop the draft, roll the SSM back to its state after
        // `last_token`, and emit v0 only.
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();

        if let Err(e) = model.start_rollback_and_checkpoint_async(&mut a.seq, 1) {
            tracing::error!("ngram rollback: {e:#}");
            a.finished = true;
            return;
        }

        proposer.observe(&a.seq.tokens, v0);

        emit_token(a, v0, None, sched);
        if a.finished {
            return;
        }
        a.last_token = v0;

        if let Some(draft) = proposer.propose(&a.seq.tokens) {
            a.pending_drafts = vec![draft];
        }

        tracing::info!(
            "NGRAM K2 REJECT: sync={sync_us}μs verify={verify_us}μs cache={} seq_len={}",
            proposer.len(),
            a.seq.seq_len,
        );
    }
}

/// 2026-09-25: Fills the XGrammar bitmask for the current matcher position and clones it
/// into an owned `Vec<i32>` for `run_mtp_propose_multi`'s `grammar_bitmask`.
///
/// Returns `None` when the grammar is inactive, the sequence is inside a
/// `<think>` span, the grammar has terminated, or `fill_bitmask` reports
/// that the mask allows every token. The proposer then drafts unconstrained.
pub fn mtp_grammar_mask_for(a: &mut ActiveSeq) -> Option<Vec<i32>> {
    if a.inside_thinking {
        return None;
    }
    let gs = a.grammar_state.as_mut()?;
    if gs.is_terminated() {
        return None;
    }
    if !gs.fill_bitmask() {
        return None;
    }
    Some(gs.bitmask_data().to_vec())
}

/// 2026-09-25: The number of tokens to draft: 1 while a grammar is active, else
/// `num_drafts`. `run_mtp_propose_multi` takes one bitmask, for the matcher's
/// current position, so drafts after the first would be sampled against a
/// stale mask.
pub fn effective_drafts_under_grammar(a: &ActiveSeq, num_drafts: usize) -> usize {
    if a.grammar_state.is_some() {
        1
    } else {
        num_drafts
    }
}

/// 2026-09-25: Returns how many leading drafts the grammar accepts in sequence.
///
/// `run_mtp_propose_multi` masks every draft with the bitmask of the
/// matcher's current position, which is right for `drafts[0]` only; a later
/// draft can be illegal once the earlier ones are applied. This feeds each
/// draft to `accept_token` in order, stops at the first rejection, and rolls
/// the matcher back to where it started.
///
/// Fewer than two drafts, or a terminated grammar, return `drafts.len()`
/// without checking.
pub fn truncate_drafts_at_grammar_boundary(gs: &mut GrammarState, drafts: &[u32]) -> usize {
    if drafts.len() < 2 || gs.is_terminated() {
        return drafts.len();
    }
    // 2026-09-25: roll back the actual matcher advances (a history
    // delta), not the `accepted` tally. `accept_token` returns true
    // for stop tokens and in the terminated state without advancing
    // the matcher, so counting the rollback from `accepted` would
    // over-rewind when such a token is in the draft span.
    // `accepted` still drives truncation.
    let steps_before = gs.num_history_steps();
    let mut accepted = 0usize;
    for &tok in drafts {
        if !gs.accept_token(tok) {
            break;
        }
        accepted += 1;
    }
    let advanced = gs.num_history_steps().saturating_sub(steps_before);
    if advanced > 0 {
        gs.rollback(advanced);
    }
    if accepted < drafts.len() {
        tracing::warn!(
            kept = accepted,
            dropped = drafts.len() - accepted,
            "spec-decode boundary: truncated drafts crossing grammar transition"
        );
    }
    accepted
}
