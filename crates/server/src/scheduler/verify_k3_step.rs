// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: single-sequence K=3 verify: one 3-row forward judges two
//! drafts.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

pub(super) mod stats;
use stats::{k3_record_outcome, k3_record_positional};

/// 2026-09-25: verify `[last_token, drafts[0], drafts[1]]` and accept 0 to
/// 2 drafts; `drafts` must hold at least 2. Picks go through
/// `verify_pick_all_with_pipeline` unless this is a DFlash serve with
/// `dflash_masked_verify` off, which judges on the raw argmax.
pub fn step_verify_k3(
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
        super::lifecycle::fail_sequence(a, format!("sync_secondary: {e:#}"));
        return;
    }

    // 2026-09-25: the pre-verify `seq_len`, the join key of the SHADOW_TGT
    // log line below.
    let shadow_base = a.seq.seq_len;

    // 2026-09-25: EP: command 0xFFFFFFF3 makes the worker ranks run
    // `decode_verify_graphed_k3` (the model's EP worker loop), then the 3
    // tokens follow.
    let tokens_k3 = [a.last_token, drafts[0], drafts[1]];
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF3) {
        tracing::error!("EP broadcast verify_k3 cmd: {e:#}");
        super::lifecycle::fail_sequence(a, format!("EP broadcast verify_k3 cmd: {e:#}"));
        return;
    }
    for &t in &tokens_k3 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast verify_k3 token: {e:#}");
            super::lifecycle::fail_sequence(a, format!("EP broadcast verify_k3 token: {e:#}"));
            return;
        }
    }

    let t_verify = sched.io.clock.now();
    // 2026-09-25: the fused forward is for DFlash on a single rank. Under EP
    // the workers run `decode_verify_graphed_k3` for the command above, so
    // the master runs it too and the collectives stay matched.
    let result_vec: Vec<u32> = if dflash_verify_raw_argmax && !model.is_ep() {
        // 2026-09-25: one M=3 forward; it captures the DFlash hidden at
        // row 0 (`Model::decode_and_verify_fused`).
        match model.decode_and_verify_fused(&tokens_k3, &mut a.seq, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_and_verify_fused (k3): {e:#}");
                super::lifecycle::fail_sequence(a, format!("decode_and_verify_fused (k3): {e:#}"));
                return;
            }
        }
    } else {
        match model.decode_verify_graphed_k3(&tokens_k3, &mut a.seq, 0) {
            Ok(r) => r.to_vec(),
            Err(e) => {
                tracing::error!("decode_verify_graphed_k3: {e:#}");
                super::lifecycle::fail_sequence(a, format!("decode_verify_graphed_k3: {e:#}"));
                return;
            }
        }
    };
    let verify_us = sched
        .io
        .clock
        .now()
        .saturating_duration_since(t_verify)
        .as_micros();
    a.last_token_time = sched.io.clock.now();
    let (v0_argmax, v1_argmax, v2_argmax) = (result_vec[0], result_vec[1], result_vec[2]);

    // 2026-09-29: A144: raw argmax never sees `logit_bias`; when decode
    // would apply a non-empty bias to this row, take the masked pipeline
    // instead.
    let (v0, v1, v2) = if dflash_verify_raw_argmax
        && !sched.levers.dflash_masked_verify
        && !crate::scheduler::sample_step::speculative_bias_forces_host(
            a,
            sched.levers.think_ended_gpu_argmax,
        ) {
        // 2026-09-25: DFlash without masked verify: judge on the raw argmax,
        // with no masks or penalties.
        (v0_argmax, v1_argmax, v2_argmax)
    } else {
        // 2026-09-25: a position with no pick falls back to its GPU argmax.
        let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &[v0_argmax, v1_argmax, v2_argmax],
            a,
            verify_ctx,
            0,
        );
        (
            processed.first().copied().unwrap_or(v0_argmax),
            processed.get(1).copied().unwrap_or(v1_argmax),
            processed.get(2).copied().unwrap_or(v2_argmax),
        )
    };

    let num_accepted = if drafts[0] != v0 {
        0
    } else if drafts[1] != v1 {
        1
    } else {
        2
    };

    // 2026-09-25: `METRALE_MTP_SHADOW_TOPK`: the target side of the
    // drafter's top-k probe, joined offline on `base`.
    if sched.levers.shadow_topk > 0 {
        tracing::info!(
            "SHADOW_TGT base={shadow_base} v=[{v0},{v1},{v2}] drafts=[{},{}]",
            drafts[0],
            drafts[1],
        );
    }

    // 2026-09-25: per-position draft match, scored at both positions even
    // when draft 1 was rejected (see `stats.rs`).
    k3_record_positional(sched, drafts[0] == v0, drafts[1] == v1, a.seq.seq_len);

    // 2026-09-25: `METRALE_MTP_REFEED_ACCEPTED`: ring the target's hidden for
    // each verify row `t` in `0..=num_accepted` under label `L+t+1`, where
    // `L = seq_len - 3` is the pre-verify length (the verify forward has
    // already added 3), plus `mtp_refeed_shift()` (default 0). Label n holds
    // the hidden at position n-1, the hidden that produced token n.
    //
    // Both ends of the inclusive range are needed:
    // - rows `0..num_accepted` cover the accepted positions; the drafter
    //   rows `mtp_rows_to_trim` drops for them are rebuilt from the ring;
    // - row `num_accepted` writes label `L + num_accepted + 1`, the next
    //   step's base. `save_hidden_for_catchup_dispatch` restarts its
    //   contiguous window on any gap, so skipping it would reset the window
    //   every step.
    //
    // Row `num_accepted` is always a committed row (a reject still commits
    // `v0`), so no `num_accepted > 0` guard is needed.
    if metrale_model_layers::speculative::mtp_refeed_accepted_enabled() {
        let base = a.seq.seq_len.saturating_sub(3);
        let shift = metrale_model_layers::speculative::mtp_refeed_shift();
        for t in 0..=num_accepted {
            let label = ((base + t + 1) as isize + shift).max(0) as usize;
            if let Err(e) = model.save_hidden_for_catchup(t, label) {
                tracing::debug!("save_hidden_for_catchup(K=3, t={t}): {e:#} — degrading");
                break;
            }
        }
    }

    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &[v0, v1, v2], top_logprobs, 0)
    } else {
        Vec::new()
    };

    // 2026-09-25: EP: the workers block on this count after their verify,
    // so it is sent before any emit below can finish the sequence.
    if let Err(e) = model.ep_broadcast_cmd(num_accepted as u32) {
        tracing::error!("EP broadcast verify_k3 result: {e:#}");
        super::lifecycle::fail_sequence(a, format!("EP broadcast verify_k3 result: {e:#}"));
        return;
    }

    // 2026-09-25: debug level: this fires on every verify step.
    tracing::debug!(
        "K3 verify: tokens=[{},{},{}] → v=[{v0},{v1},{v2}] drafts=[{},{}] accepted={num_accepted} seq_len={}",
        tokens_k3[0],
        tokens_k3[1],
        tokens_k3[2],
        drafts[0],
        drafts[1],
        a.seq.seq_len
    );

    if num_accepted == 2 {
        emit_token(a, drafts[0], verify_lps.first().cloned(), sched);
        if !a.finished {
            emit_token(a, drafts[1], verify_lps.get(1).cloned(), sched);
        }
        if !a.finished {
            emit_token(a, v2, verify_lps.get(2).cloned(), sched);
        }
        if a.finished {
            return;
        }
        a.last_token = v2;

        // 2026-09-25: commit all 3 rows; `commit_accepted_prefix` treats a
        // full accept as a no-op.
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 3, 3) {
            // 2026-09-25: the SSM state cannot be trusted after a failed
            // commit, so the sequence fails.
            tracing::error!("commit_accepted_prefix (K=3 accept-3): {e:#}");
            super::lifecycle::fail_sequence(
                a,
                format!("commit_accepted_prefix (K=3 accept-3): {e:#}"),
            );
            return;
        }
        if let Err(e) = model.save_hidden_for_mtp(2, 0) {
            tracing::error!("save_hidden_for_mtp(2): {e:#}");
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 2, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        let t_propose = sched.io.clock.now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v2,
            a.seq.seq_len,
            crate::scheduler::spec_step::effective_drafts_under_grammar(a, num_drafts),
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
            }
        }
        let propose_us = sched
            .io
            .clock
            .now()
            .saturating_duration_since(t_propose)
            .as_micros();
        tracing::debug!(
            "K3 ACCEPT-2: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k3_record_outcome(sched, 2, a.seq.seq_len);
    } else if num_accepted == 1 {
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // 2026-09-25: one draft accepted: commit rows 0..=1.
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 2, 3) {
            tracing::error!("commit_accepted_prefix (K=3 accept-2): {e:#}");
            super::lifecycle::fail_sequence(
                a,
                format!("commit_accepted_prefix (K=3 accept-2): {e:#}"),
            );
            return;
        }
        emit_token(a, drafts[0], verify_lps.first().cloned(), sched);
        if !a.finished {
            emit_token(a, v1, verify_lps.get(1).cloned(), sched);
        }
        if a.finished {
            return;
        }
        a.last_token = v1;
        if let Err(e) = model.save_hidden_for_mtp(1, 0) {
            tracing::error!("save_hidden_for_mtp(1): {e:#}");
            return;
        }
        let t_propose = sched.io.clock.now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v1,
            a.seq.seq_len,
            crate::scheduler::spec_step::effective_drafts_under_grammar(a, num_drafts),
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
            }
        }
        let propose_us = sched
            .io
            .clock
            .now()
            .saturating_duration_since(t_propose)
            .as_micros();
        tracing::debug!(
            "K3 ACCEPT-1: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k3_record_outcome(sched, 1, a.seq.seq_len);
    } else {
        a.seq.seq_len -= 2;
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // 2026-09-25: reject: commit row 0 only.
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 1, 3) {
            tracing::error!("commit_accepted_prefix (K=3 accept-1): {e:#}");
            super::lifecycle::fail_sequence(
                a,
                format!("commit_accepted_prefix (K=3 accept-1): {e:#}"),
            );
            return;
        }
        emit_token(a, v0, verify_lps.first().cloned(), sched);
        if a.finished {
            return;
        }
        a.last_token = v0;
        if let Err(e) = model.save_hidden_for_mtp(0, 0) {
            tracing::error!("save_hidden_for_mtp(0): {e:#}");
            return;
        }
        let t_propose = sched.io.clock.now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v0,
            a.seq.seq_len,
            crate::scheduler::spec_step::effective_drafts_under_grammar(a, num_drafts),
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
            }
        }
        let propose_us = sched
            .io
            .clock
            .now()
            .saturating_duration_since(t_propose)
            .as_micros();
        tracing::debug!(
            "K3 REJECT: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k3_record_outcome(sched, 0, a.seq.seq_len);
    }
}
