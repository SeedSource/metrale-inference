// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: batched MTP verify: one forward pass verifies the drafts of
//! several sequences.
//!
//! Owner: scheduler.
//! Invariants:
//! - `step_mtp` calls this only when `METRALE_MTP_MAX_SEQS` is above 1, the
//!   serve has no DFlash drafter, `mtp_batch_verify` is on, and the chunk
//!   holds at least 2 grammarless sequences that `model.can_batch_verify`
//!   accepts.
//! - Every sequence's logits rows are read, and a stash of every accepted
//!   row's hidden state is attempted, before any sequence's verdict or
//!   drafter propose runs. The propose overwrites the shared hidden-state
//!   rows (`Model::stash_verify_hidden_rows`).

use super::*;

/// 2026-09-25: verify `batch.len() >= 2` sequences in one forward.
///
/// Sequence `i` must hold exactly `ks[i] - 1` pending drafts. `ks` is ragged
/// when D-Cut prunes (`mtp_dcut::plan`) and uniform otherwise. The caller
/// also guarantees: grammarless sequences, no DFlash drafter,
/// `Σ ks <= VERIFY_ROW_BUDGET` (`mtp_dcut::chunk_ranges`), the batch in the
/// order `verify_key::verify_batch_permutation` returns, and
/// `model.can_batch_verify(ks)`.
///
/// `propose_nd` is the ladder's draft count, not the pruned one: the drafter
/// refills every sequence to full depth, and the next step's D-Cut prunes
/// again.
pub(super) fn step_verify_k4_batched(
    model: &dyn Model,
    batch: &mut [&mut ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    ks: &[usize],
    propose_nd: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let n = batch.len();
    debug_assert_eq!(ks.len(), n);
    // 2026-09-25: `off[i]` is sequence i's first row in the flat seq-major
    // layout (prefix sum of `ks`); `off[n]` is the total.
    let mut off: Vec<usize> = Vec::with_capacity(n + 1);
    let mut acc = 0usize;
    for &k in ks {
        off.push(acc);
        acc += k;
    }
    let r_total = acc;
    off.push(r_total);
    debug_assert!(
        (2..=32).contains(&n)
            && ks.iter().all(|k| (2..=4).contains(k))
            && r_total <= crate::scheduler::mtp_dcut::VERIFY_ROW_BUDGET
    );

    // 2026-09-25: one step timer for the whole batch; it records on drop.
    let _step_timer = sched.io.tel.step_timer(batch[0].seq.seq_len);

    // 2026-09-25: one secondary-stream sync for the whole batch before the
    // forward. On failure every sequence in the batch is finished.
    if let Err(e) = model.sync_secondary() {
        tracing::error!("batched-verify sync_secondary: {e:#}");
        for a in batch.iter_mut() {
            a.finished = true;
        }
        return;
    }

    // 2026-09-25: each sequence contributes the rows
    // `[last_token, d0, .., d_{ks[i]-2}]`, flat and seq-major.
    let mut drafts_per_seq: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut tokens: Vec<u32> = Vec::with_capacity(r_total);
    for (i, a) in batch.iter_mut().enumerate() {
        let d = std::mem::take(&mut a.pending_drafts);
        // 2026-09-25: the confidences describe the drafts just taken, so they
        // are cleared with them and D-Cut never ranks a stale vector.
        a.pending_draft_conf.clear();
        debug_assert!(
            d.len() + 1 == ks[i],
            "batchable classification requires exactly {} drafts",
            ks[i] - 1
        );
        tokens.push(a.last_token);
        tokens.extend_from_slice(&d);
        drafts_per_seq.push(d);
    }

    // 2026-09-25: `Model::decode_verify_batched` contract: on Ok each
    // sequence's `tokens` and `seq_len` advanced by its own `ks[i]` (the
    // verdict rewinds them); on Err no sequence state advanced, and this
    // step finishes every sequence in the batch.
    let t_verify = sched.io.clock.now();
    let results: Vec<u32> = {
        let mut seq_refs: Vec<&mut SequenceState> = batch.iter_mut().map(|a| &mut a.seq).collect();
        // 2026-09-26: write-on-accept: the GDN layers carry their state
        // (`model-engine gdn_carry.rs`), and every verdict reaches
        // `gdn_fold_accepted` below before any `commit_accepted_prefix`.
        let opts = metrale_model_engine::traits::VerifyBatchedOpts {
            write_on_accept: true,
        };
        match model.decode_verify_batched(&tokens, ks, &mut seq_refs, 0, opts) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_verify_batched (n={n} ks={ks:?}): {e:#}");
                for a in batch.iter_mut() {
                    a.finished = true;
                }
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
    sched
        .io
        .tel
        .mark(crate::scheduler::mtp_timing::Phase::VerifyForward, t_verify);
    let t_phase1 = sched.io.clock.now();

    // 2026-09-25: read every sequence's logits rows (picks and logprobs)
    // before any propose. Sequence i's rows start at `off[i]`.
    let mut verdicts: Vec<(Vec<u32>, usize, Vec<crate::api::TokenLogprobs>)> =
        Vec::with_capacity(n);
    for (i, a) in batch.iter_mut().enumerate() {
        let rows = ks[i];
        let k_drafts = rows - 1;
        let r = &results[off[i]..off[i + 1]];
        // 2026-09-25: the same pick function as the single-sequence MTP
        // steps, with `row_base = off[i]`. A position it returns no pick for
        // falls back to the GPU argmax.
        let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model, r, a, verify_ctx, off[i],
        );
        let v: Vec<u32> = (0..rows)
            .map(|j| processed.get(j).copied().unwrap_or(r[j]))
            .collect();
        let drafts = &drafts_per_seq[i];
        let mut num_accepted = 0usize;
        while num_accepted < k_drafts && drafts[num_accepted] == v[num_accepted] {
            num_accepted += 1;
        }
        // 2026-09-25: per-position draft match, scored for every position
        // whether or not the accept chain stopped earlier. The counters have
        // three positions, so only 3-draft sequences record them.
        if k_drafts == 3 {
            crate::scheduler::verify_k4_step::stats::k4_record_positional(
                sched,
                drafts[0] == v[0],
                drafts[1] == v[1],
                drafts[2] == v[2],
                a.seq.seq_len,
            );
        }
        // 2026-09-25: accept telemetry bucketed by batch width, recorded for
        // every `k_drafts`; `METRALE_MTP_ACCEPT_DEBUG` gates its log lines.
        sched.accept.record(
            &sched.rung,
            sched.levers.mtp_accept_debug,
            n,
            k_drafts,
            drafts[0] == v[0],
            num_accepted,
        );
        let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
            extract_verify_logprobs(model, &v, top_logprobs, off[i])
        } else {
            Vec::new()
        };
        a.last_token_time = sched.io.clock.now();
        verdicts.push((v, num_accepted, verify_lps));
    }

    sched
        .io
        .tel
        .mark(crate::scheduler::mtp_timing::Phase::PipelineProc, t_phase1);
    // 2026-10-02: Multi-rank (`METRALE_GLM_BATCHED_VERIFY`, model-engine `verify_ep.rs`): the
    // worker ranks wait for every sequence's accepted count, in batch order, then stash and roll
    // back as this rank does below. It goes out before anything else this step sends.
    if model.is_ep() {
        let words: Vec<u32> = verdicts.iter().map(|&(_, na, _)| na as u32).collect();
        if let Err(e) = model.ep_broadcast_tokens(&words) {
            tracing::error!("EP broadcast batched-verify verdict (n={n}): {e:#}");
            for a in batch.iter_mut() {
                a.finished = true;
            }
            return;
        }
    }
    let t_stash = sched.io.clock.now();
    // 2026-09-25: stash each sequence's accepted-position hidden row
    // (`off[i] + num_accepted`) into stash slot i before any propose
    // overwrites the live rows; the proposes below read the stash.
    let stash_rows: Vec<usize> = verdicts
        .iter()
        .enumerate()
        .map(|(i, &(_, num_accepted, _))| off[i] + num_accepted)
        .collect();
    if let Err(e) = model.stash_verify_hidden_rows(&stash_rows, 0) {
        // 2026-09-25: logged and not fatal: the verdicts below still apply.
        tracing::error!("stash_verify_hidden_rows: {e:#}");
    }
    // 2026-09-25: exact drafter KV (`METRALE_MTP_KV_EXACT`; the model no-ops
    // when it is off): stash the verify rows of every accepted draft, at
    // most `MTP_CATCHUP_MAX` per sequence, for the catch-up below.
    const CATCHUP_MAX: usize = metrale_model_layers::layer::MTP_CATCHUP_MAX;
    let accepted: Vec<usize> = verdicts
        .iter()
        .map(|&(_, na, _)| na.min(CATCHUP_MAX))
        .collect();
    let catchup_rows: Vec<(usize, usize)> = accepted
        .iter()
        .enumerate()
        .flat_map(|(i, &na)| (0..na).map(move |k| (i, k)))
        .map(|(i, k)| (i * CATCHUP_MAX + k, off[i] + k))
        .collect();
    if !catchup_rows.is_empty()
        && let Err(e) = model.stash_verify_catchup_rows(&catchup_rows)
    {
        tracing::error!("stash_verify_catchup_rows: {e:#}");
    }

    sched
        .io
        .tel
        .mark(crate::scheduler::mtp_timing::Phase::SaveHidden, t_stash);
    let t_verdict = sched.io.clock.now();
    // 2026-09-26: the GDN h-state commit of the whole batch, rows `0..=na`
    // per sequence (`k4_apply_verdict` clamps `na` to `ks[i] - 1` the same
    // way). A failure leaves the batch's recurrent state untrusted, so every
    // sequence finishes.
    {
        let slots: Vec<usize> = batch.iter().map(|a| a.seq.slot_idx).collect();
        let rows: Vec<u32> = verdicts
            .iter()
            .enumerate()
            .map(|(i, &(_, na, _))| (na.min(ks[i] - 1) + 1) as u32)
            .collect();
        let k_max = ks.iter().copied().max().unwrap_or(2);
        if let Err(e) = model.gdn_fold_accepted(&slots, &rows, k_max) {
            tracing::error!("gdn_fold_accepted (mtp batched): {e:#}");
            for a in batch.iter_mut() {
                a.finished = true;
            }
            // 2026-10-02: the workers wait for this gather after real verdict words, on
            // every path (`verify_ep.rs`); every sequence is failed here.
            retire_failed_on_workers(model, batch, u32::MAX);
            return;
        }
    }
    // 2026-09-25: per-sequence verdict through `k4_apply_verdict`, with the
    // propose deferred (`K4Hidden::DeferPropose`) so it can be batched below.
    for (i, (a, (v, num_accepted, verify_lps))) in
        batch.iter_mut().zip(verdicts.into_iter()).enumerate()
    {
        k4_apply_verdict(
            model,
            a,
            sched,
            &drafts_per_seq[i],
            &v,
            verify_lps,
            ks[i] - 1,
            num_accepted,
            K4Hidden::DeferPropose,
            verify_us,
        );
    }

    // 2026-10-02: multi-rank: the workers' commit failures (`verify_ep.rs`, one gather after
    // their per-sequence commits); those sequences finish before any propose.
    retire_failed_on_workers(model, batch, 0);

    // 2026-09-25: propose fresh drafts for the sequences still alive with no
    // drafts, in groups of up to `model.mtp_propose_batch_max()`. Groups of
    // one, groups the model declines (`Ok(None)`), and every sequence when
    // `mtp_batch_propose` is off, take the per-sequence propose, which first
    // copies the stash slot into the MTP input buffer.
    sched
        .io
        .tel
        .mark(crate::scheduler::mtp_timing::Phase::Commit, t_verdict);
    let t_propose = sched.io.clock.now();
    let mut pending: Vec<usize> = (0..n)
        .filter(|&i| !batch[i].finished && batch[i].pending_drafts.is_empty())
        .collect();
    if pending.is_empty() {
        // 2026-09-25: `_step_timer` records the step on drop.
        return;
    }
    // 2026-09-25: exact drafter KV: append the accepted drafts' drafter rows
    // for all pending sequences in one batched call before any propose.
    let catchup_tokens: Vec<Vec<u32>> = pending
        .iter()
        .map(|&i| drafts_per_seq[i][..accepted[i]].to_vec())
        .collect();
    if catchup_tokens.iter().any(|t| !t.is_empty()) {
        let first_slot: Vec<usize> = pending.iter().map(|&i| i * CATCHUP_MAX).collect();
        let first_pos: Vec<usize> = pending
            .iter()
            .zip(&catchup_tokens)
            .map(|(&i, t)| batch[i].seq.seq_len - t.len())
            .collect();
        let mut seq_refs: Vec<&mut SequenceState> = batch
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| pending.contains(i))
            .map(|(_, a)| &mut a.seq)
            .collect();
        if let Err(e) =
            model.run_mtp_catchup_batched(&catchup_tokens, &first_slot, &first_pos, &mut seq_refs)
        {
            tracing::error!("run_mtp_catchup_batched: {e:#}");
        }
    }
    let mut need_fallback: Vec<usize> = Vec::new();
    // 2026-10-04: A batch that mirrors the per-sequence propose
    // (`METRALE_GLM_MTP_BATCH_DRAFT=1`) takes no grammar, so sequences with one propose per
    // sequence, after the batched groups, as the fallbacks do.
    if model.mtp_propose_batch_mirrors_serial() {
        let (grammar, free): (Vec<usize>, Vec<usize>) = pending
            .iter()
            .partition(|&&i| batch[i].grammar_state.is_some());
        need_fallback = grammar;
        pending = free;
    }
    let mut groups_batched = 0usize;
    // 2026-09-25: draft confidences (the D-Cut ranking key) are requested
    // only when D-Cut is on.
    let want_conf = sched.levers.dcut_enabled;
    let mut conf: Vec<Vec<f32>> = Vec::new();
    let group_cap = model.mtp_propose_batch_max().max(1);
    if pending.len() >= 2 && group_cap >= 2 && sched.levers.mtp_batch_propose {
        for group in pending.chunks(group_cap) {
            if group.len() < 2 {
                need_fallback.extend_from_slice(group);
                continue;
            }
            let tokens: Vec<u32> = group.iter().map(|&i| batch[i].last_token).collect();
            let positions: Vec<usize> = group.iter().map(|&i| batch[i].seq.seq_len).collect();
            let stash_idx: Vec<usize> = group.to_vec();
            let result = {
                let mut seq_refs: Vec<&mut SequenceState> = Vec::with_capacity(group.len());
                let mut it = batch.iter_mut();
                let mut prev = 0usize;
                for (j, &i) in group.iter().enumerate() {
                    let step = if j == 0 { i } else { i - prev - 1 };
                    let a = it.nth(step).expect("group index in batch");
                    seq_refs.push(&mut a.seq);
                    prev = i;
                }
                model.run_mtp_propose_batched(
                    &tokens,
                    &positions,
                    &stash_idx,
                    propose_nd,
                    &mut seq_refs,
                    0,
                    want_conf.then_some(&mut conf),
                )
            };
            match result {
                Ok(Some(all)) => {
                    for (j, &i) in group.iter().enumerate() {
                        if !all[j].is_empty() {
                            batch[i].pending_drafts = all[j].clone();
                            batch[i].pending_draft_conf = conf.get(j).cloned().unwrap_or_default();
                        }
                    }
                    groups_batched += 1;
                }
                Ok(None) => need_fallback.extend_from_slice(group),
                Err(e) => {
                    // 2026-09-25: no per-sequence retry for a failed group:
                    // its sequences keep no drafts, so `step_mtp` bootstraps
                    // them next step.
                    log_propose_batched_err("run_mtp_propose_batched", &e);
                }
            }
        }
    } else {
        need_fallback.extend_from_slice(&pending);
    }
    for &i in &need_fallback {
        let a = &mut batch[i];
        if let Err(e) = model.save_hidden_for_mtp_from_stash(i, 0) {
            tracing::error!("save_hidden_for_mtp_from_stash({i}): {e:#}");
            continue;
        }
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            propose_nd,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            // 2026-09-25: the per-sequence propose returns no confidences,
            // so `pending_draft_conf` stays empty.
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
            }
        }
    }
    tracing::debug!(
        "K{propose_nd} batched propose: n={} cap={group_cap} groups_batched={groups_batched} \
         fallback={} propose={}μs",
        pending.len(),
        need_fallback.len(),
        sched
            .io
            .clock
            .now()
            .saturating_duration_since(t_propose)
            .as_micros()
    );
    sched
        .io
        .tel
        .mark(crate::scheduler::mtp_timing::Phase::Propose, t_propose);
    // 2026-09-25: `_step_timer` records the step on drop.
}

/// 2026-10-02: Multi-rank batched verify: join the workers' commit-failure gather
/// (`Model::ep_agree_verify_failures`, a no-op without EP) with `local_mask`, and finish every
/// sequence whose bit (batch order) any rank set. Must run on every path that follows the
/// verdict broadcast, since the workers block on the gather.
fn retire_failed_on_workers(model: &dyn Model, batch: &mut [&mut ActiveSeq], local_mask: u32) {
    if !model.is_ep() {
        return;
    }
    match model.ep_agree_verify_failures(local_mask) {
        Ok(mask) => {
            for (i, a) in batch.iter_mut().enumerate().take(32) {
                if (mask >> i) & 1 == 1 {
                    tracing::error!("batched verify: worker commit failed, retiring seq {i}");
                    a.finished = true;
                }
            }
        }
        Err(e) => {
            tracing::error!("EP batched-verify failure gather: {e:#}");
            for a in batch.iter_mut() {
                a.finished = true;
            }
        }
    }
}
