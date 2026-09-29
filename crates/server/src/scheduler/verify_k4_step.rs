// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: single-sequence K=4 verify: one 4-row forward judges three
//! drafts.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

#[path = "verify_k4_step/stats.rs"]
pub(super) mod stats;
use stats::k4_record_positional;

/// 2026-09-25: verify `[last_token, drafts[0], drafts[1], drafts[2]]` and
/// accept 0 to 3 drafts; `drafts` must hold at least 3. Picks go through
/// `verify_pick_all_with_pipeline` unless this is a DFlash serve with
/// `dflash_masked_verify` off, which judges on the raw argmax. The verdict
/// and the re-propose of `num_drafts` drafts are `k4_apply_verdict`'s.
pub fn step_verify_k4(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    // 2026-09-25: the step timer records on drop, so every return below is
    // covered.
    let _step_timer = sched.io.tel.step_timer(a.seq.seq_len);

    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        super::lifecycle::fail_sequence(a, format!("sync_secondary: {e:#}"));
        return;
    }

    // 2026-09-25: the pre-verify `seq_len`, the join key of the SHADOW_TGT
    // log line below.
    let shadow_base = a.seq.seq_len;

    let tokens_k4 = [a.last_token, drafts[0], drafts[1], drafts[2]];

    // 2026-09-25: EP: command 0xFFFFFFF4 makes the worker ranks run the same
    // K=4 verify (the model's EP worker loop), then the 4 tokens follow.
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF4) {
        tracing::error!("EP broadcast verify_k4 cmd: {e:#}");
        super::lifecycle::fail_sequence(a, format!("EP broadcast verify_k4 cmd: {e:#}"));
        return;
    }
    for &t in &tokens_k4 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast verify_k4 token: {e:#}");
            super::lifecycle::fail_sequence(a, format!("EP broadcast verify_k4 token: {e:#}"));
            return;
        }
    }

    let t_verify = sched.io.clock.now();
    // 2026-09-25: the fused forward is for DFlash on a single rank. Under EP
    // the workers run `decode_verify_graphed_k4` for the command above, so
    // the master runs it too and the collectives stay matched.
    let result_vec: Vec<u32> = if dflash_verify_raw_argmax && !model.is_ep() {
        // 2026-09-25: one M=4 forward; it captures the DFlash hidden at
        // row 0 (`Model::decode_and_verify_fused`).
        match model.decode_and_verify_fused(&tokens_k4, &mut a.seq, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_and_verify_fused (k4): {e:#}");
                super::lifecycle::fail_sequence(a, format!("decode_and_verify_fused (k4): {e:#}"));
                return;
            }
        }
    } else {
        match model.decode_verify_graphed_k4(&tokens_k4, &mut a.seq, 0) {
            Ok(r) => r.to_vec(),
            Err(e) => {
                tracing::error!("decode_verify_graphed_k4: {e:#}");
                super::lifecycle::fail_sequence(a, format!("decode_verify_graphed_k4: {e:#}"));
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
    let (v0_argmax, v1_argmax, v2_argmax, v3_argmax) =
        (result_vec[0], result_vec[1], result_vec[2], result_vec[3]);

    // 2026-09-29: A144: raw argmax never sees `logit_bias`; when decode
    // would apply a non-empty bias to this row, take the masked pipeline
    // instead.
    let (v0, v1, v2, v3) = if dflash_verify_raw_argmax
        && !sched.levers.dflash_masked_verify
        && !crate::scheduler::sample_step::speculative_bias_forces_host(
            a,
            sched.levers.think_ended_gpu_argmax,
        ) {
        // 2026-09-25: DFlash without masked verify: judge on the raw argmax,
        // with no masks or penalties.
        (v0_argmax, v1_argmax, v2_argmax, v3_argmax)
    } else {
        // 2026-09-25: a position with no pick falls back to its GPU argmax.
        let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &[v0_argmax, v1_argmax, v2_argmax, v3_argmax],
            a,
            verify_ctx,
            0,
        );
        (
            processed.first().copied().unwrap_or(v0_argmax),
            processed.get(1).copied().unwrap_or(v1_argmax),
            processed.get(2).copied().unwrap_or(v2_argmax),
            processed.get(3).copied().unwrap_or(v3_argmax),
        )
    };

    let num_accepted = if drafts[0] != v0 {
        0
    } else if drafts[1] != v1 {
        1
    } else if drafts[2] != v2 {
        2
    } else {
        3
    };

    // 2026-09-25: `METRALE_MTP_SHADOW_TOPK`: the target side of the
    // drafter's top-k probe, joined offline on `base`.
    if sched.levers.shadow_topk > 0 {
        tracing::info!(
            "SHADOW_TGT base={shadow_base} v=[{v0},{v1},{v2},{v3}] drafts=[{},{},{}]",
            drafts[0],
            drafts[1],
            drafts[2],
        );
    }

    // 2026-09-25: per-position draft match, scored at every position even
    // when the accept chain stopped earlier.
    k4_record_positional(
        sched,
        drafts[0] == v0,
        drafts[1] == v1,
        drafts[2] == v2,
        a.seq.seq_len,
    );
    // 2026-09-25: accept telemetry in the width-1 bucket; the batched step
    // records its own width.
    sched.accept.record(
        &sched.rung,
        sched.levers.mtp_accept_debug,
        1,
        3,
        drafts[0] == v0,
        num_accepted,
    );

    // 2026-09-25: `METRALE_MTP_REFEED_ACCEPTED`: ring the target's hidden for
    // verify rows `0..=num_accepted` under labels `L+1..=L+num_accepted+1`
    // plus `mtp_refeed_shift()`, where L is the pre-verify `seq_len`. With
    // the lever on, the drafter's trim (`mtp_rows_to_trim`) drops the
    // accepted rows past the first at every width, so every verify width
    // must re-feed them.
    if metrale_model_layers::speculative::mtp_refeed_accepted_enabled() {
        let base = a.seq.seq_len.saturating_sub(4);
        let shift = metrale_model_layers::speculative::mtp_refeed_shift();
        for t in 0..=num_accepted {
            let label = ((base + t + 1) as isize + shift).max(0) as usize;
            if let Err(e) = model.save_hidden_for_catchup(t, label) {
                tracing::debug!("save_hidden_for_catchup(K=4, t={t}): {e:#} — degrading");
                break;
            }
        }
    }

    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &[v0, v1, v2, v3], top_logprobs, 0)
    } else {
        Vec::new()
    };

    if let Err(e) = model.ep_broadcast_cmd(num_accepted as u32) {
        tracing::error!("EP broadcast verify_k4 result: {e:#}");
        super::lifecycle::fail_sequence(a, format!("EP broadcast verify_k4 result: {e:#}"));
        return;
    }

    // 2026-09-25: debug level: this fires on every verify step.
    tracing::debug!(
        "K4 verify: tokens=[{},{},{},{}] → v=[{v0},{v1},{v2},{v3}] drafts=[{},{},{}] accepted={num_accepted} seq_len={}",
        tokens_k4[0],
        tokens_k4[1],
        tokens_k4[2],
        tokens_k4[3],
        drafts[0],
        drafts[1],
        drafts[2],
        a.seq.seq_len
    );

    k4_apply_verdict(
        model,
        a,
        sched,
        drafts,
        &[v0, v1, v2, v3],
        verify_lps,
        num_drafts,
        num_accepted,
        K4Hidden::VerifyRow,
        verify_us,
    );
}
