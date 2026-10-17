// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Cross-sequence batched DFlash K=γ verify.
//!
//! Owner: scheduler.
//! Invariants:
//! - Every read of the shared verify outputs (the accept walk, `commit_ctx`, the hidden stash) happens for all sequences before the first propose; the early returns before that point propose nothing.
//! - 2026-10-09: Multi-rank (`METRALE_GLM_DFLASH_BATCHED_VERIFY`): after an `Ok` forward rank 0
//!   sends the verdict words (`broadcast_batch_verdict`) once every accept is walked, and then
//!   makes exactly one failure-mask gather (`retire_failed_on_workers`) on every exit (short
//!   result, fold error, normal tail) before any propose. An `Err` forward sends nothing more
//!   (`decode_verify_batched` sent the abort words) and finishes the batch, since a worker may
//!   have advanced. The verify width `k` is whatever the caller's drafts carry (2..=4 across
//!   ranks); nothing here assumes 4.
//! - 2026-10-09: `METRALE_GLM_DFLASH_KSEL` (`dflash_ksel`) may cut every row to one narrower
//!   width before the rows are built; the worker learns it from the `ks` meta only.
//! - 2026-10-09: Under the batched GLM pin (`DflashRung::multi_batched`) each sequence's
//!   verdict is picked on `step_verify_dflash`'s basis (`dflash_batch_rows::verdict_uses_pipeline`,
//!   the logits pipeline at rows `i * k..`), before the verdict words; otherwise on the raw argmax.
//!
//! The single-sequence [`super::verify_dflash_step::step_verify_dflash`] runs
//! one target forward per sequence. This step packs the `n` sequences'
//! `[last_token, d0..d_{γ-1}]` rows into one `decode_verify_batched` call of
//! `n*(γ+1)` rows, which reads the weights once for the whole batch.

use super::*;

/// 2026-09-25: Batched DFlash verify for `batch.len()` sequences at uniform K = γ+1.
///
/// `drafts_per_seq` is γ: every sequence must carry exactly that many pending
/// drafts (`mtp_step.rs` groups only sequences with the same draft count).
///
/// 2026-10-14 (race-memory, A184): Returns the error when the forward was refused for
/// KV and the refusal is safe to retry (`preempt::spec_kv_refusal_retryable`), as
/// `step_verify_k4_batched` does. No sequence is finished then; the caller must pass it
/// to `preempt::preempt_after_spec_refusal`. Before this, the refusal put the drafts back
/// and returned, so the same refusal repeated every step and the batch stalled. Every
/// other failure fails its sequences with the error (an error frame), never a bare
/// `finished`, which the client would read as a normal "stop".
#[must_use = "a KV refusal must reach preempt::preempt_after_spec_refusal"]
pub fn step_verify_dflash_batched(
    model: &dyn Model,
    batch: &mut [&mut ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts_per_seq: usize,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) -> Option<String> {
    // 2026-10-09: `METRALE_GLM_DFLASH_KSEL`: under the batched GLM pin, one width for this batch
    // from the drafter p and the measured cost table (`dflash_ksel`); the rows' drafts are cut
    // to it here, so `ks` below (and the `EP_CMD_VERIFY_BATCH` meta the worker reads) carry it.
    let drafts_per_seq =
        super::dflash_ksel::select_and_truncate(model, batch, sched, drafts_per_seq);
    let n = batch.len();
    let k = drafts_per_seq + 1;
    debug_assert!(n >= 2 && drafts_per_seq >= 1);

    // 2026-09-25: one secondary-stream sync for the whole batch, so the
    // async checkpoint copies are complete before this verify.
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary (dflash batched): {e:#}");
        fail_dflash_batch(batch, &format!("sync_secondary (dflash batched): {e:#}"));
        return None;
    }

    // 2026-09-25: flat seq-major token rows, plus the per-sequence drafts they encode.
    let mut tokens: Vec<u32> = Vec::with_capacity(n * k);
    let mut drafts_all: Vec<Vec<u32>> = Vec::with_capacity(n);
    for a in batch.iter_mut() {
        let drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        a.pending_draft_conf.clear();
        debug_assert_eq!(drafts.len(), drafts_per_seq);
        tokens.push(a.last_token);
        tokens.extend_from_slice(&drafts);
        drafts_all.push(drafts);
    }
    let ks = vec![k; n];

    let t_verify = sched.io.clock.now();
    let results = {
        let mut seq_refs: Vec<&mut SequenceState> = batch.iter_mut().map(|a| &mut a.seq).collect();
        // 2026-09-25: write-on-accept obliges this step to run
        // `gdn_fold_accepted` with every verdict before any
        // `commit_accepted_prefix`, which it does below.
        let opts = metrale_model_engine::traits::VerifyBatchedOpts {
            write_on_accept: true,
        };
        match model.decode_verify_batched(&tokens, &ks, &mut seq_refs, 0, opts) {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("decode_verify_batched (dflash, n={n} k={k}): {e:#}");
                // 2026-10-14 (race-memory, A184): per the `decode_verify_batched`
                // contract no sequence state advanced on Err. A retryable KV refusal
                // finishes nothing: the caller preempts a victim and drops every
                // draft, so the rest bootstrap again on the next step. Any other
                // error would repeat on every step, so it fails the batch.
                if crate::scheduler::preempt::spec_kv_refusal_retryable(model, &msg) {
                    tracing::warn!("{msg}");
                    return Some(msg);
                }
                tracing::error!("{msg}");
                fail_dflash_batch(batch, &msg);
                return None;
            }
        }
    };
    let verify_ms = sched
        .io
        .clock
        .now()
        .saturating_duration_since(t_verify)
        .as_secs_f64()
        * 1000.0;
    if results.len() < n * k {
        tracing::error!(
            "decode_verify_batched (dflash): short result {} < {}",
            results.len(),
            n * k
        );
        fail_dflash_batch(
            batch,
            &format!(
                "decode_verify_batched (dflash): short result {} < {}",
                results.len(),
                n * k
            ),
        );
        // 2026-10-09: Multi-rank: the workers wait for verdict words and then the gather;
        // zero accepts (a valid verdict) and every sequence failed.
        if broadcast_batch_verdict(model, batch, &vec![0; n], &ks) {
            retire_failed_on_workers(model, batch, u32::MAX);
        }
        return None;
    }

    // 2026-10-09: The accept walk for every sequence, then the verdict words (multi-rank)
    // before anything else this step does.
    // 2026-10-09: Under the batched GLM pin (`DflashRung::multi_batched`) a row is judged on
    // the basis `step_verify_dflash` would use for it (`verdict_uses_pipeline`): thinking and
    // biased rows, and every row while `dflash_masked_verify` is on, go through the logits
    // pipeline at their own rows (`row_base = i * k`), as the MTP batch does. Without the pin
    // every row is judged on the raw argmax, the basis the drafter proposes on.
    let pinned = sched.dflash_rung.multi_batched();
    let now = sched.io.clock.now();
    let mut host_rows = 0usize;
    let verdicts: Vec<Vec<u32>> = batch
        .iter_mut()
        .enumerate()
        .map(|(i, a)| {
            let raw = &results[i * k..i * k + k];
            // 2026-10-15: the partition in `mtp_step.rs` runs before this forward, so a raw
            // argmax on a stray `</tool_call>` that `StrayToolCallEndMask` masks on this row
            // is caught here, pinned or not: this row's verdict comes from the masked pipeline
            // over its own rows of the shared logits buffer (`row_base = i * k`), and the
            // accept walk below compares the drafts with those picks. Read before any propose,
            // like every other read of the shared outputs.
            let stray =
                crate::scheduler::verify_pipeline_helper::raw_verdict_hits_stray_tool_call_end(
                    a, raw, verify_ctx,
                );
            if !stray && !(pinned && super::dflash_batch_rows::verdict_uses_pipeline(sched, a)) {
                return raw.to_vec();
            }
            host_rows += 1;
            let picked = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
                model,
                raw,
                a,
                verify_ctx,
                i * k,
            );
            // 2026-10-09: A position with no pick falls back to the GPU argmax (as the MTP
            // batch does).
            (0..k)
                .map(|j| picked.get(j).copied().unwrap_or(raw[j]))
                .collect()
        })
        .collect();
    let accepted_per_seq: Vec<usize> = batch
        .iter_mut()
        .enumerate()
        .map(|(i, a)| {
            let verified = &verdicts[i];
            let drafts = &drafts_all[i];
            a.last_token_time = now;
            let mut num_accepted = 0usize;
            for j in 0..drafts.len() {
                if j + 1 >= verified.len() || drafts[j] != verified[j] {
                    break;
                }
                num_accepted += 1;
            }
            crate::scheduler::adaptive_spec::record_verify(a, num_accepted, sched);
            num_accepted
        })
        .collect();
    if !broadcast_batch_verdict(model, batch, &accepted_per_seq, &ks) {
        return None;
    }

    // 2026-09-25: First pass, per sequence: rewind, ctx commit and emit. The
    // shared-buffer reads happen here, before any propose.
    let mut stash_rows: Vec<usize> = Vec::with_capacity(n);
    for (i, a) in batch.iter_mut().enumerate() {
        let off = i * k;
        let verified = &verdicts[i];
        let drafts = &drafts_all[i];
        let num_accepted = accepted_per_seq[i];

        // 2026-09-25: rewind the forward's +k to the accepted prefix plus
        // the bonus slot (the same arithmetic as `verify_dflash_step`).
        let pre_verify_len = a.seq.seq_len.saturating_sub(k);
        let target_seq_len = pre_verify_len + num_accepted + 1;
        let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
        if to_drop > 0 {
            a.seq.seq_len = target_seq_len;
            let pop_n = to_drop.min(a.seq.tokens.len());
            for _ in 0..pop_n {
                a.seq.tokens.pop();
            }
        }

        // 2026-09-25: commit this sequence's ctx rows from its own
        // capture band, `i * dflash_capture_band()`, the stride the
        // model captured with.
        tracing::debug!(
            "CTX_VERIFY slot={} pre_verify_len={} na={} k={} band={}",
            a.seq.slot_idx,
            pre_verify_len,
            num_accepted,
            k,
            i * model.dflash_capture_band(),
        );
        if sched.levers.dflash_unified_ctx
            && let Err(e) = model.commit_ctx(
                &mut a.seq,
                num_accepted + 1,
                pre_verify_len,
                i * model.dflash_capture_band(),
            )
        {
            tracing::error!("commit_ctx (dflash batched): {e:#}");
        }

        for j in 0..num_accepted {
            emit_token(a, drafts[j], None, sched);
            if a.finished {
                break;
            }
        }
        if !a.finished && num_accepted < verified.len() {
            let bonus = verified[num_accepted];
            emit_token(a, bonus, None, sched);
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

        // 2026-09-25: the SSM commit waits until after this loop: the
        // write-on-accept fold takes every sequence's verdict at once.
        // This is the row of this sequence's bonus token in the shared
        // hidden buffer.
        stash_rows.push(off + num_accepted);
    }

    // 2026-09-25: the write-on-accept fold over the whole batch, then the
    // per-sequence commit.
    {
        let slots: Vec<usize> = batch.iter().map(|a| a.seq.slot_idx).collect();
        let rows: Vec<u32> = accepted_per_seq.iter().map(|&na| (na + 1) as u32).collect();
        if let Err(e) = model.gdn_fold_accepted(&slots, &rows, k) {
            // 2026-09-25: a failed fold leaves the whole batch's
            // recurrent state untrusted, so every sequence finishes, as
            // a commit error does for one sequence.
            tracing::error!("gdn_fold_accepted (dflash batched): {e:#}");
            fail_dflash_batch(batch, &format!("gdn_fold_accepted (dflash batched): {e:#}"));
            // 2026-10-09: the workers wait for this gather after real verdict words.
            retire_failed_on_workers(model, batch, u32::MAX);
            return None;
        }
        for (i, a) in batch.iter_mut().enumerate() {
            let num_accepted = accepted_per_seq[i];
            if let Err(e) = model.commit_accepted_prefix(&mut a.seq, num_accepted + 1, k) {
                tracing::error!("commit_accepted_prefix (dflash batched): {e:#}");
                super::lifecycle::fail_sequence(
                    a,
                    format!("commit_accepted_prefix (dflash batched): {e:#}"),
                );
            }
        }
    }

    // 2026-09-25: park every sequence's bonus hidden in the verify stash
    // while the rows are still live; proposes overwrite the shared
    // `hidden_states` buffer.
    if let Err(e) = model.stash_verify_hidden_rows(&stash_rows, 0) {
        tracing::warn!("stash_verify_hidden_rows (dflash batched): {e:#}");
    }
    // 2026-10-09: multi-rank: the workers' commit failures, gathered once, before any propose
    // (the DFlash propose runs on rank 0 alone and sends the workers nothing).
    retire_failed_on_workers(model, batch, 0);

    // 2026-09-25: Second pass: per-sequence trim, then the batched re-propose
    // (with a per-sequence fallback). The shared buffers may be
    // overwritten from here on.
    let t_propose = sched.io.clock.now();
    // 2026-09-25: trim first for everyone: the batched propose reads
    // each sequence's proposer state, so every state must already
    // reflect what its verify accepted. `spec_allowed` takes &mut, so
    // eligibility is decided in this pass while the mutable borrow is
    // already held.
    let mut eligible = vec![false; batch.len()];
    for (i, a) in batch.iter_mut().enumerate() {
        if a.finished {
            continue;
        }
        let num_accepted = accepted_per_seq[i];
        // 2026-09-25: no `save_hidden_for_mtp_from_stash` here. It
        // copies into the one MTP input buffer, so calling it for every
        // sequence before any propose would leave only the last
        // sequence's hidden staged. The batched arm reads each
        // sequence's own stash slot (`stash_idx`); the per-sequence arm
        // below stages each sequence right before its own propose.
        if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
            tracing::error!("trim_proposer_state (dflash batched): {e:#}");
        }
        eligible[i] =
            crate::scheduler::adaptive_spec::spec_allowed(a, sched) && a.grammar_state.is_none();
    }
    // 2026-09-25: eligible means not finished, speculation allowed and no
    // grammar (`run_mtp_propose_batched` takes grammarless sequences
    // only).
    let prop_idx: Vec<usize> = (0..batch.len()).filter(|&i| eligible[i]).collect();
    let group_cap = model.mtp_propose_batch_max().max(1);
    // 2026-09-25: chunk by the width the proposer declared
    // (`mtp_propose_batch_max`), not by how many sequences are eligible.
    let mut batched_done = !prop_idx.is_empty();
    for group in prop_idx.chunks(group_cap.max(1)) {
        if group_cap < 2 || group.len() < 2 {
            batched_done = false;
            break;
        }
        let prop_idx: Vec<usize> = group.to_vec();
        let tokens: Vec<u32> = prop_idx.iter().map(|&i| batch[i].last_token).collect();
        let positions: Vec<usize> = prop_idx.iter().map(|&i| batch[i].seq.seq_len).collect();
        let stash_idx: Vec<usize> = prop_idx.clone();
        let result = {
            let mut seq_refs: Vec<&mut SequenceState> = Vec::with_capacity(prop_idx.len());
            for (i, a) in batch.iter_mut().enumerate() {
                if prop_idx.contains(&i) {
                    seq_refs.push(&mut a.seq);
                }
            }
            model.run_mtp_propose_batched(
                &tokens,
                &positions,
                &stash_idx,
                num_drafts,
                &mut seq_refs,
                0,
                None,
            )
        };
        match result {
            Ok(Some(all)) if all.len() == prop_idx.len() => {
                for (g, &row) in prop_idx.iter().enumerate() {
                    if !all[g].is_empty() {
                        batch[row].pending_drafts = all[g].clone();
                    }
                }
            }
            Ok(_) => {
                batched_done = false;
                break;
            }
            Err(e) => {
                tracing::warn!("DFlash batched propose: {e:#} — per-sequence path");
                batched_done = false;
                break;
            }
        }
    }
    if batched_done {
        // 2026-09-25: batched path covered every eligible sequence; nothing left to do.
    } else {
        for (i, a) in batch.iter_mut().enumerate() {
            if a.finished || !crate::scheduler::adaptive_spec::spec_allowed(a, sched) {
                continue;
            }
            if let Err(e) = model.save_hidden_for_mtp_from_stash(i, 0) {
                tracing::warn!("save_hidden_for_mtp_from_stash (dflash batched): {e:#}");
            }
            let gmask = mtp_grammar_mask_for(a);
            match model.run_mtp_propose_multi(
                a.last_token,
                a.seq.seq_len,
                num_drafts,
                &mut a.seq,
                0,
                gmask.as_deref(),
            ) {
                Ok(d) if !d.is_empty() => a.pending_drafts = d,
                Ok(_) => {}
                Err(e) => tracing::error!("run_mtp_propose_multi (dflash batched): {e:#}"),
            }
        }
    }

    let total_accepted: usize = accepted_per_seq.iter().sum();
    tracing::info!(
        "DFLASH BATCHED verify: n={n} γ={drafts_per_seq} accepted={total_accepted}/{} ({:.0}%) \
         verify={verify_ms:.1}ms propose={:.1}ms host_rows={host_rows}",
        n * drafts_per_seq,
        100.0 * (total_accepted as f64) / ((n * drafts_per_seq) as f64),
        sched
            .io
            .clock
            .now()
            .saturating_duration_since(t_propose)
            .as_secs_f64()
            * 1000.0,
    );
    None
}

/// 2026-10-14 (race-memory, A184): Fail every sequence of the batch with `msg`, so each
/// client gets an error frame (`lifecycle::fail_sequence`), not a cut reply ending "stop".
fn fail_dflash_batch(batch: &mut [&mut ActiveSeq], msg: &str) {
    for a in batch.iter_mut() {
        super::lifecycle::fail_sequence(a, msg.to_string());
    }
}
