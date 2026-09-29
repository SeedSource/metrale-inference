// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MTP speculative draft proposal and verify step.
//!
//! Owner: scheduler.
//! Invariants:
//! - On entry, `active` is split into bootstrap (no pending drafts) and
//!   verify (pending drafts) sequences; each sequence is in exactly one.
//! - The step's draft width `ladder_nd` never exceeds the smallest
//!   verify-slot draft capacity in `active` (`clamp_drafts_to_slot_capacity`).

use super::*;

mod serial_bootstrap;

/// 2026-09-25: One speculative step: bootstrap the sequences without drafts,
/// then verify the ones with drafts.
///
/// `verify_ctx` is the `logit_processors` context handed to every bootstrap
/// and verify call, so verified tokens go through the logits processors
/// (`verify_pipeline_helper`).
pub fn step_mtp(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    // 2026-09-25: Start of the `Phase::StepOuter` span, recorded at both
    // exits of this function.
    let t_step_outer = sched.io.clock.now();
    // 2026-09-25: DFlash: this step's draft count from the gamma resolver
    // (`dflash_rung`), keyed on `active.len()`, which counts bootstrap
    // sequences as well as verifying ones. It shadows the serve-wide value
    // for every propose below. The resolver returns `num_drafts` unchanged
    // when it is not armed (pinned by `--dflash-gamma` or
    // `METRALE_DFLASH_STATIC_GAMMA`, or a cap below K=3).
    let num_drafts = if dflash_verify_raw_argmax {
        sched.dflash_rung.drafts_for(active.len(), num_drafts)
    } else {
        num_drafts
    };
    let mut bootstrap_idxs: Vec<usize> = Vec::new();
    let mut verify_idxs: Vec<usize> = Vec::new();
    for (i, a) in active.iter().enumerate() {
        if !a.pending_drafts.is_empty() {
            verify_idxs.push(i);
        } else {
            bootstrap_idxs.push(i);
        }
    }

    // 2026-09-25: MTP: the draft count depends on the number of active
    // sequences. The static ladder (`metrale_model_layers::speculative::ladder`,
    // default `4:3,8:3,16:1,32:1`, overridden by `METRALE_MTP_K_LADDER` or
    // `METRALE_NO_MTP_K_LADDER`) gives 3 drafts through n=8 and 1 above,
    // clamped to `[1, num_drafts]`. `adaptive_rung::drafts_for` may raise
    // n in 9..=16 to 2 drafts from the observed accept rates. DFlash keeps
    // the resolver's count.
    let ladder_nd = if dflash_verify_raw_argmax {
        num_drafts
    } else {
        sched.rung.drafts_for(active.len(), num_drafts)
    };
    // 2026-09-25: Clamp to the smallest draft capacity of any active
    // sequence's verify slot (`mtp_slot_draft_capacity`), so no sequence gets
    // more drafts than its slot holds. `usize::MAX` means no limit.
    let ladder_nd = metrale_speculative::spec_capacity::clamp_drafts_to_slot_capacity(
        ladder_nd,
        active
            .iter()
            .map(|a| model.mtp_slot_draft_capacity(a.seq.slot_idx)),
    );

    // 2026-09-25: Bootstrap: the sequences without drafts.
    if !bootstrap_idxs.is_empty() {
        // 2026-09-25: The previous verify's async rollback and checkpoint
        // copies run on the secondary stream; make the default stream wait
        // for them before the bootstrap decode reads the SSM state.
        // `sync_secondary` is a GPU-side event wait; a failure is logged and
        // the step continues.
        if let Err(e) = model.sync_secondary() {
            tracing::error!("bootstrap sync_secondary: {e:#}");
        }
    }
    // 2026-09-25: Batched bootstrap (`step_mtp_bootstrap_batched`: one
    // `decode_batch` and a batched propose) when `can_batch_bootstrap`
    // allows it, which `METRALE_NO_MTP_BATCH_BOOTSTRAP` prevents. Otherwise
    // the per-sequence loop below runs.
    if can_batch_bootstrap(model, sched, bootstrap_idxs.len(), dflash_verify_raw_argmax) {
        step_mtp_bootstrap_batched(model, active, sched, &bootstrap_idxs, ladder_nd, verify_ctx);
        bootstrap_idxs.clear();
    }
    for &idx in &bootstrap_idxs {
        let a = &mut active[idx];
        serial_bootstrap::bootstrap_seq(
            model,
            a,
            sched,
            num_drafts,
            ladder_nd,
            verify_ctx,
            dflash_verify_raw_argmax,
        );
    }

    // 2026-09-25: Verify. MTP sequences are verified as a batch when at
    // least two have drafts, `mtp_max_seqs > 1` (`METRALE_MTP_MAX_SEQS`) and
    // `METRALE_NO_MTP_BATCH_VERIFY` is not set. A sequence is batchable when
    // it has no grammar and at least `ladder_nd` drafts; drafts beyond
    // `ladder_nd` are truncated. The model decides per chunk through
    // `can_batch_verify(&ks)`; everything else takes the per-sequence loop.
    let mut serial_idxs: Vec<usize> = Vec::new();
    let mut batchable_idxs: Vec<usize> = Vec::new();
    // 2026-09-25: DFlash batched verify. It needs at least two sequences
    // with drafts, `dflash_batch_verify` (off with
    // `METRALE_DFLASH_BATCH_VERIFY=0`) and `mtp_batch_verify` (off with
    // `METRALE_NO_MTP_BATCH_VERIFY`). The batch is every sequence without a
    // grammar whose draft count equals the first such sequence's; the rest
    // take the per-sequence step. The debug line logs the gate inputs once.
    if sched.io.tel.stats().once("log:dflash_batch_verify_gate") {
        tracing::debug!(
            raw_argmax = dflash_verify_raw_argmax,
            n_verify = verify_idxs.len(),
            lever = sched.levers.dflash_batch_verify,
            killed = !sched.levers.mtp_batch_verify,
            "DFlash batched verify gate (first tick with drafts)"
        );
    }
    if dflash_verify_raw_argmax
        && verify_idxs.len() >= 2
        && sched.levers.dflash_batch_verify
        && sched.levers.mtp_batch_verify
    {
        let mut gamma = 0usize;
        for &idx in &verify_idxs {
            let a = &active[idx];
            let g = a.pending_drafts.len();
            // 2026-09-29: A144: the batched DFlash verdict is raw argmax
            // only (no pipeline, no `logit_bias`); a row whose
            // decode-effective bias is non-empty (the tools-active
            // `<tool_call>` nudge) takes the per-sequence step, which routes
            // it through the masked pipeline.
            if a.grammar_state.is_some()
                || g < 1
                // 2026-09-29: A146: a thinking row needs the host pipeline too.
                || crate::scheduler::sample_step::speculative_raw_argmax_forbidden(
                    a,
                    sched.levers.think_ended_gpu_argmax,
                )
            {
                serial_idxs.push(idx);
            } else if gamma == 0 || g == gamma {
                gamma = g;
                batchable_idxs.push(idx);
            } else {
                serial_idxs.push(idx);
            }
        }
        // 2026-09-25: `m_max` is the widest group `can_batch_verify` accepts
        // at `gamma + 1` rows, found by shrinking from the whole group. The
        // group is verified in chunks of `m_max` instead of being refused
        // whole.
        let k_rows = gamma + 1;
        let mut m_max = batchable_idxs.len();
        while m_max >= 2 && !model.can_batch_verify(&vec![k_rows; m_max]) {
            m_max -= 1;
        }
        if batchable_idxs.len() >= 2 && m_max >= 2 {
            // 2026-09-25: Ssm-slot order, not arrival order: the batched
            // conv+WY GDN path needs the batch on consecutive ssm-pool slots
            // in batch order, and declines otherwise (it checks the pointers,
            // trait_decode_batched_conv_gdn_multi.rs). Drafts travel inside
            // each `ActiveSeq`, so the order is free to choose.
            let mut by_slot = batchable_idxs.clone();
            sort_batch_by_slot(&mut by_slot, |i| active[i].seq.ssm_slot_idx());
            for chunk in by_slot.chunks(m_max) {
                if chunk.len() >= 2 {
                    // 2026-09-25: Borrow in ascending active-index order
                    // (one forward walk), then restore the chunk's slot
                    // order.
                    let mut asc: Vec<usize> = chunk.to_vec();
                    asc.sort_unstable();
                    let mut tagged: Vec<(usize, &mut ActiveSeq)> = Vec::with_capacity(asc.len());
                    let mut it = active.iter_mut();
                    let mut consumed = 0usize;
                    for &i in &asc {
                        let a = it.nth(i - consumed).expect("verify index within active");
                        consumed = i + 1;
                        let pos = chunk.iter().position(|&c| c == i).expect("chunk member");
                        tagged.push((pos, a));
                    }
                    tagged.sort_by_key(|t| t.0);
                    let mut refs: Vec<&mut ActiveSeq> = tagged.into_iter().map(|t| t.1).collect();
                    step_verify_dflash_batched(
                        model, &mut refs, sched, gamma, num_drafts, verify_ctx,
                    );
                } else {
                    // 2026-09-25: A trailing single cannot batch; it takes
                    // the serial loop below.
                    serial_idxs.extend_from_slice(chunk);
                }
            }
        } else {
            // 2026-09-25: The first decline is logged once, at info level.
            if !batchable_idxs.is_empty() && sched.io.tel.stats().once("log:dflash_batch_decline") {
                tracing::info!(
                    "DFlash batched verify DECLINED: n={} k={} (model.can_batch_verify said no) \
                     — running the per-sequence loop",
                    batchable_idxs.len(),
                    gamma + 1,
                );
            }
            serial_idxs.extend_from_slice(&batchable_idxs);
        }
        batchable_idxs.clear();
        for &idx in &serial_idxs {
            let a = &mut active[idx];
            let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
            a.pending_draft_conf.clear();
            if drafts.is_empty() {
                continue;
            }
            if let Some(ref mut gs) = a.grammar_state {
                let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
                drafts.truncate(kept);
                if drafts.is_empty() {
                    continue;
                }
            }
            step_verify_dflash(
                model,
                a,
                sched,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        }
        // 2026-09-25: This arm returns early, so it records `StepOuter`
        // itself, as the end of the function does.
        sched
            .io
            .tel
            .mark(crate::scheduler::mtp_timing::Phase::StepOuter, t_step_outer);
        return;
    }
    if verify_idxs.len() >= 2
        && sched.levers.mtp_max_seqs > 1
        && !dflash_verify_raw_argmax
        && sched.levers.mtp_batch_verify
        && ladder_nd >= 1
    {
        for &idx in &verify_idxs {
            let a = &mut active[idx];
            if a.grammar_state.is_none() && a.pending_drafts.len() >= ladder_nd {
                if a.pending_drafts.len() > ladder_nd {
                    a.pending_drafts.truncate(ladder_nd);
                }
                batchable_idxs.push(idx);
            } else {
                serial_idxs.push(idx);
            }
        }
    } else {
        serial_idxs.extend_from_slice(&verify_idxs);
    }
    let rows = ladder_nd + 1;

    // 2026-09-25: D-Cut (`mtp_dcut::plan`): each batched sequence's verify
    // depth from the drafter's confidences; drafts are truncated and
    // `batchable_idxs` is put in dispatch order. When D-Cut is off
    // (`METRALE_NO_MTP_DCUT`), `ladder_nd < 2`, or the batch is wider than
    // `dcut_width_cap`, every entry of `ks` is `rows`.
    let ks = mtp_dcut::plan(sched, active, &mut batchable_idxs, ladder_nd, rows);
    // 2026-09-25: The width the assignment was gated on (`plan` reorders
    // `batchable_idxs`, never resizes it): the per-chunk re-ordering below
    // must ask the gate with this width, never the chunk's, or a chunked
    // batch could take the opposite arm from the one its depths were
    // assigned under.
    let batch_n = batchable_idxs.len();

    // 2026-09-25: Chunks that fit the verify row budget and stash width
    // (`mtp_dcut::chunk_ranges`). A lone sequence, or a chunk the model
    // refuses, takes the per-sequence loop.
    for (lo, hi) in mtp_dcut::chunk_ranges(&ks) {
        let chunk = &batchable_idxs[lo..hi];
        let chunk_ks = &ks[lo..hi];
        if chunk.len() >= 2 && model.can_batch_verify(chunk_ks) {
            // 2026-09-25: Disjoint `&mut` refs need an ascending walk, so
            // walk a sorted copy of the chunk (each index keeps its k); the
            // dispatch order is recomputed below.
            let mut asc: Vec<(usize, usize)> = chunk
                .iter()
                .copied()
                .zip(chunk_ks.iter().copied())
                .collect();
            asc.sort_unstable();
            let mut refs: Vec<(&mut ActiveSeq, usize)> = Vec::with_capacity(chunk.len());
            let mut it = active.iter_mut();
            let mut consumed = 0usize;
            for &(i, k) in &asc {
                let a = it.nth(i - consumed).expect("chunk index within active");
                consumed = i + 1;
                refs.push((a, k));
            }
            // 2026-09-25: Dispatch order from
            // `verify_key::verify_batch_permutation`, asked with the width
            // `plan` used (`batch_n`). Canonical: ascending ssm slot, which
            // after `plan`'s canonical assignment is also deepest first.
            // Otherwise: deepest first, then slot. This only reorders; each
            // sequence keeps the depth `plan` truncated it to
            // (`verify_k4_batch_step` debug-asserts `drafts + 1 == ks[i]`).
            // It also orders the batch when `plan` returned it unchanged.
            let chunk_slots: Vec<usize> = refs
                .iter()
                .map(|(a, _)| a.seq.ssm_slot_idx().unwrap_or(usize::MAX))
                .collect();
            let chunk_depths: Vec<usize> = refs.iter().map(|&(_, k)| k).collect();
            let order = metrale_model_layers::speculative::verify_key::verify_batch_permutation(
                &chunk_slots,
                &chunk_depths,
                metrale_model_layers::speculative::verify_key::canonical_assignment(batch_n),
            );
            let sorted_ks: Vec<usize> = order.iter().map(|&p| chunk_depths[p]).collect();
            let mut slotted: Vec<Option<&mut ActiveSeq>> =
                refs.into_iter().map(|(a, _)| Some(a)).collect();
            let mut batch: Vec<&mut ActiveSeq> = order
                .iter()
                .map(|&p| {
                    slotted[p]
                        .take()
                        .expect("verify_batch_permutation is a permutation")
                })
                .collect();
            step_verify_k4_batched(model, &mut batch, sched, &sorted_ks, ladder_nd, verify_ctx);
        } else {
            // 2026-09-25: A lone sequence, or a chunk `can_batch_verify`
            // refuses: per-sequence loop.
            serial_idxs.extend_from_slice(chunk);
        }
    }
    for &idx in &serial_idxs {
        let a = &mut active[idx];
        let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        // 2026-09-25: The confidences describe the drafts just taken, so
        // they are cleared with them.
        a.pending_draft_conf.clear();
        if drafts.is_empty() {
            continue;
        }

        // 2026-09-25: Under a grammar, keep only the drafts before the first
        // one the grammar would reject (`truncate_drafts_at_grammar_boundary`);
        // fewer drafts select a smaller verify arm below.
        if let Some(ref mut gs) = a.grammar_state {
            let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
            if kept < drafts.len() {
                drafts.truncate(kept);
            }
            if drafts.is_empty() {
                continue;
            }
        }

        // 2026-09-25: Verify arm from `spec_capacity::serial_verify_plan`:
        // 4 or more drafts take `step_verify_dflash`, fewer take K4/K3/K2.
        // MTP drafts beyond this step's `ladder_nd` are truncated first;
        // DFlash keeps all of its drafts.
        let step_nd = if dflash_verify_raw_argmax {
            num_drafts
        } else {
            ladder_nd
        };
        let (keep, arm) = metrale_speculative::spec_capacity::serial_verify_plan(
            drafts.len(),
            step_nd,
            dflash_verify_raw_argmax,
        );
        drafts.truncate(keep);
        if drafts.is_empty() {
            continue;
        }
        let verify = match arm {
            metrale_speculative::spec_capacity::SerialArm::DFlash => step_verify_dflash,
            metrale_speculative::spec_capacity::SerialArm::K4 => step_verify_k4,
            metrale_speculative::spec_capacity::SerialArm::K3 => step_verify_k3,
            metrale_speculative::spec_capacity::SerialArm::K2 => step_verify_k2,
        };
        verify(
            model,
            a,
            sched,
            &drafts,
            step_nd,
            verify_ctx,
            dflash_verify_raw_argmax,
        );
    }
    sched
        .io
        .tel
        .mark(crate::scheduler::mtp_timing::Phase::StepOuter, t_step_outer);
}

/// 2026-09-25: Sort the DFlash batched-verify group by ssm slot (see the
/// call site in `step_mtp`). Sequences without a slot sort last; the active
/// index breaks ties, so the order is total.
fn sort_batch_by_slot(by_slot: &mut [usize], slot_of: impl Fn(usize) -> Option<usize>) {
    by_slot.sort_by_key(|&i| (slot_of(i).unwrap_or(usize::MAX), i));
}

#[cfg(test)]
#[path = "mtp_step_tests.rs"]
mod tests;
