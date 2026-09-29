// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: single-sequence DFlash verify of a variable-length draft
//! block.
//!
//! Owner: scheduler.
//! Invariants:
//! - No EP broadcast: this step issues no worker command, unlike the
//!   single-sequence K=2/3/4 steps.
//! - No logprobs: tokens are emitted with `None`.

use super::*;

/// 2026-09-25: verify `[last_token, drafts..]` with
/// `model.decode_verify_dflash`. Drafts are accepted up to the first one
/// that differs from its row's pick; the pick at that row is emitted as the
/// bonus token and the later drafts are dropped. Picks go through
/// `verify_pick_all_with_pipeline` unless `dflash_masked_verify` is off on a
/// DFlash serve.
pub fn step_verify_dflash(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }

    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    tokens.push(a.last_token);
    tokens.extend_from_slice(drafts);

    // 2026-09-25: `METRALE_DFLASH_STEP_TIMING=1` logs the verify and propose
    // walls separately at the end of the step.
    let step_timing = sched.levers.dflash_step_timing;
    let t_verify = sched.io.clock.now();
    let verified_argmax = match model.decode_verify_dflash(&tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("decode_verify_dflash: {e:#}");
            a.finished = true;
            return;
        }
    };
    let verify_ms = if step_timing {
        sched
            .io
            .clock
            .now()
            .saturating_duration_since(t_verify)
            .as_secs_f64()
            * 1000.0
    } else {
        0.0
    };
    a.last_token_time = sched.io.clock.now();

    // 2026-09-25: on a DFlash serve with `dflash_masked_verify` off, judge on
    // the raw argmax, with no masks or penalties.
    // 2026-09-29: A144: raw argmax never sees `logit_bias`; when decode
    // would apply a non-empty bias to this row, take the masked pipeline
    // instead.
    let verified = if dflash_verify_raw_argmax
        && !sched.levers.dflash_masked_verify
        && !crate::scheduler::sample_step::speculative_bias_forces_host(
            a,
            sched.levers.think_ended_gpu_argmax,
        ) {
        verified_argmax
    } else {
        crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &verified_argmax,
            a,
            verify_ctx,
            0,
        )
    };

    // 2026-09-25: `drafts[i]` is accepted while it equals `verified[i]`, the
    // pick for the token after `tokens[i]`, and only while a bonus row
    // `verified[i + 1]` exists.
    let mut num_accepted = 0usize;
    for i in 0..drafts.len() {
        if i + 1 >= verified.len() {
            break;
        }
        if drafts[i] == verified[i] {
            num_accepted += 1;
        } else {
            break;
        }
    }

    // 2026-09-25: with `METRALE_DFLASH_ADAPTIVE=1`, a low mean over the
    // accept window suspends this sequence's speculation (`adaptive_spec`).
    crate::scheduler::adaptive_spec::record_verify(a, num_accepted, sched);
    // 2026-09-25: the gamma resolver's accept signal: was the first draft
    // accepted.
    sched.dflash_rung.observe_step(num_accepted >= 1);

    // 2026-09-25: the verify forward advanced `seq_len` and `seq.tokens` by
    // `tokens.len()`. Keep the pre-verify prefix, the accepted drafts and the
    // bonus position (`pre_verify_len + num_accepted + 1`) and drop the rest.
    // `emit_token` writes `output_tokens` only, so the bonus stays where the
    // verify put it in `seq.tokens`.
    let pre_verify_len = a.seq.seq_len.saturating_sub(tokens.len());
    let target_seq_len = pre_verify_len + num_accepted + 1;
    let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
    if to_drop > 0 {
        a.seq.seq_len = target_seq_len;
        let pop_n = to_drop.min(a.seq.tokens.len());
        for _ in 0..pop_n {
            a.seq.tokens.pop();
        }
    }

    // 2026-09-25: the drafter context gets rows `0..=num_accepted` at RoPE
    // base `pre_verify_len`, one of two ways. With `dflash_unified_ctx` (on
    // unless `METRALE_DFLASH_UNIFIED_CTX=0`), `commit_ctx` commits them.
    // Otherwise, when `dflash_eagle_fix` is on, `dflash_eagle_kgamma_append`
    // appends them with row `num_accepted`, the hidden that produced the
    // bonus, last. A failure of either is logged and the step continues.
    tracing::debug!(
        "CTX_VERIFY slot={} pre_verify_len={} na={} k={}",
        a.seq.slot_idx,
        pre_verify_len,
        num_accepted,
        drafts.len() + 1,
    );
    if sched.levers.dflash_unified_ctx {
        if let Err(e) = model.commit_ctx(&mut a.seq, num_accepted + 1, pre_verify_len, 0) {
            tracing::error!("commit_ctx (kgamma): {e:#}");
        }
    } else {
        let eagle_fix = sched.levers.dflash_eagle_fix;
        if eagle_fix
            && let Err(e) =
                model.dflash_eagle_kgamma_append(&mut a.seq, num_accepted, pre_verify_len)
        {
            tracing::error!("dflash_eagle_kgamma_append: {e:#}");
        }
    }

    for i in 0..num_accepted {
        emit_token(a, drafts[i], None, sched);
        if a.finished {
            return;
        }
    }

    // 2026-09-25: the bonus is `verified[num_accepted]`: the correction at
    // the first mismatch, or the token after a full accept.
    let bonus_idx = num_accepted;
    if bonus_idx < verified.len() {
        let bonus = verified[bonus_idx];
        emit_token(a, bonus, None, sched);
        if a.finished {
            return;
        }
        a.last_token = bonus;
    }

    sched.io.tel.count_spec_verify(
        "dflash",
        if num_accepted == drafts.len() {
            "accept_all"
        } else {
            "accept_partial"
        },
    );
    sched.io.tel.spec_verified(drafts.len(), num_accepted);

    tracing::info!(
        "DFLASH K=γ verify: γ={} accepted={}/{} ({:.0}%) seq_len={}",
        drafts.len(),
        num_accepted,
        drafts.len(),
        100.0 * (num_accepted as f64) / (drafts.len() as f64),
        a.seq.seq_len,
    );

    // 2026-09-25: commit the SSM state to rows `0..=num_accepted` of the
    // `drafts.len() + 1` verified rows; `commit_accepted_prefix` treats a
    // full accept as a no-op. A failed commit finishes the sequence.
    let k_verify = drafts.len() + 1;
    let total_accepted = num_accepted + 1;
    if let Err(e) = model.commit_accepted_prefix(&mut a.seq, total_accepted, k_verify) {
        tracing::error!("commit_accepted_prefix (dflash): {e:#}");
        a.finished = true;
        return;
    }

    let bonus_token_idx = total_accepted.saturating_sub(1);
    if let Err(e) = model.save_hidden_for_mtp(bonus_token_idx, 0) {
        tracing::error!("save_hidden_for_mtp (dflash): {e:#}");
    }

    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state: {e:#}");
    }

    // 2026-09-25: no propose while adaptive speculation has suspended this
    // sequence; with no drafts, `step_mtp` bootstraps it next step.
    let _mtp_grammar_mask = mtp_grammar_mask_for(a);
    let t_propose = sched.io.clock.now();
    if crate::scheduler::adaptive_spec::spec_allowed(a, sched) {
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            num_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => tracing::error!("run_mtp_propose_multi (dflash): {e:#}"),
        }
    }
    if step_timing {
        let propose_ms = sched
            .io
            .clock
            .now()
            .saturating_duration_since(t_propose)
            .as_secs_f64()
            * 1000.0;
        tracing::info!(
            "DFLASH STEP_TIMING: verify={:.1}ms propose={:.1}ms (K={}, accepted={})",
            verify_ms,
            propose_ms,
            tokens.len(),
            num_accepted,
        );
    }
}
