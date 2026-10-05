// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched MTP bootstrap: one `decode_batch` for the draftless sequences, then batched drafter proposes.
//!
//! `mtp_step.rs` bootstraps each sequence without pending drafts with one
//! `decode` and one `run_mtp_propose_multi`. When [`can_batch_bootstrap`]
//! holds it calls [`step_mtp_bootstrap_batched`] instead, which runs one
//! `decode_batch` over all of them, samples and emits per row, stashes the
//! hidden rows, and proposes from the stash in groups. Otherwise the
//! per-sequence loop runs.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Log a `run_mtp_propose_batched` failure. Callers: this file and
/// `verify_k4_batch_step.rs`.
///
/// A message containing "exceeds meta stride" logs at debug, every time; all
/// other failures log at error. That text comes from the `ensure!` in
/// `metrale_model_layers::layers::mtp_meta::pack_mtp_attn_meta`, raised when a
/// sequence's drafter block table outgrows its per-sequence `propose_meta`
/// stride. The stride is sized from `max_seq_len`
/// (`METRALE_PROPOSE_META_STRIDE` overrides it). The match is on the alternate
/// `{e:#}` form, which includes any added context.
pub(super) fn log_propose_batched_err(prefix: &str, e: &anyhow::Error) {
    let msg = format!("{e:#}");
    if msg.contains("exceeds meta stride") {
        tracing::debug!("{prefix}: {msg}");
    } else {
        tracing::error!("{prefix}: {msg}");
    }
}

/// 2026-09-25: Whether [`step_mtp_bootstrap_batched`] can run for `n` sequences:
/// at least two, not a DFlash serve, `mtp_batch_bootstrap` on (unless
/// `METRALE_NO_MTP_BATCH_BOOTSTRAP` is set), an MTP dispatch cap above 1, the
/// DFlash unified-ctx and serial-append modes off, BF16 decode logits, and the
/// model's batched-verify envelope.
pub(super) fn can_batch_bootstrap(
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    n: usize,
    dflash_verify_raw_argmax: bool,
) -> bool {
    n >= 2
        && !dflash_verify_raw_argmax
        && sched.levers.mtp_batch_bootstrap
        && sched.levers.mtp_max_seqs > 1
        && !sched.levers.dflash_unified_ctx
        && !sched.levers.dflash_serial_append
        // 2026-09-25: FP32 decode logits are excluded: `logits_ptr_is_fp32`
        // compares the pointer with the FP32 scratch buffer's start, so a
        // per-row offset pointer would be read as BF16.
        && !model.decode_logits_fp32()
        // 2026-10-02: Multi-rank: the batched bootstrap's stash and per-sequence proposes have no
        // worker mirror (only the batched verify does, `verify_ep.rs`), so it stays per sequence.
        && !model.is_ep()
        // 2026-09-25: k=2 is the narrowest verify width. This asks whether the
        // model's batched-MTP envelope holds (its checks include an allocated
        // hidden stash, no EP and no HSS), not for a verify.
        && model.can_batch_verify(&vec![2usize; n])
}

/// 2026-09-25: Batched bootstrap for the `idxs` (ascending) draftless sequences.
///
/// Per sequence it uses what the per-sequence loop in `mtp_step.rs` uses:
/// `penalty_params_for` penalties, `penalty_history_scope` history,
/// `sample_token_with_grammar`, `emit_token`, the `adaptive_spec` calls and the
/// `effective_drafts_under_grammar` clamp. One difference: when saving the
/// hidden state fails, the per-sequence loop's `continue` also skips
/// `start_checkpoint_async`; here the checkpoint still runs.
pub(super) fn step_mtp_bootstrap_batched(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    idxs: &[usize],
    ladder_nd: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    // 2026-09-25: Disjoint &mut for the (ascending) indices, sorted by SSM slot,
    // the order `decode_step.rs` also sorts `active` into before `decode_batch`.
    // `active` itself is not reordered. Row j <-> refs[j].
    let mut refs: Vec<&mut ActiveSeq> = Vec::with_capacity(idxs.len());
    let mut it = active.iter_mut();
    let mut consumed = 0usize;
    for &i in idxs {
        let a = it.nth(i - consumed).expect("bootstrap index within active");
        consumed = i + 1;
        refs.push(a);
    }
    refs.sort_by_key(|a| a.seq.ssm_slot_idx().unwrap_or(a.seq.slot_idx));

    let n = refs.len();
    let tokens: Vec<u32> = refs.iter().map(|a| a.last_token).collect();

    // 2026-09-25: One `decode_batch` for all n rows. A failure finishes every
    // sequence in the batch: retrying one at a time could decode a sequence
    // twice if the failed call had already advanced it.
    let logits = {
        let mut seq_refs: Vec<&mut SequenceState> = refs.iter_mut().map(|a| &mut a.seq).collect();
        match model.decode_batch(&tokens, &mut seq_refs, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("batched bootstrap decode_batch (n={n}): {e:#}");
                for a in refs.iter_mut() {
                    a.finished = true;
                }
                return;
            }
        }
    };

    let vocab = model.vocab_size();
    let elem = if model.decode_logits_fp32() { 4 } else { 2 };

    // 2026-09-25: One `argmax_batch` call for the greedy rows, in place of one
    // `argmax_on_device` per row inside `sample_token_with_grammar`. A row is
    // eligible when `mtp_boot_argmax` is on (unless `METRALE_NO_MTP_BOOT_ARGMAX`
    // is set) and the row would take that function's fast-greedy branch with
    // no grammar and `PenaltyGate::Neutral` penalties. The batch call runs
    // only with at least two eligible rows; every other row samples per row.
    //
    // 2026-09-29: A144: each row's base bias is what decode would apply at
    // this position (`sample_step::speculative_base_logit_bias`); a
    // non-empty bias classifies `Blocked`, so such a row leaves the
    // batched-argmax set and its per-row sample applies the bias on host.
    let pen: Vec<_> = refs
        .iter()
        .enumerate()
        .map(|(j, a)| {
            let base_bias = crate::scheduler::sample_step::speculative_base_logit_bias(
                a,
                0,
                verify_ctx.think_end_token,
                sched.levers.think_ended_gpu_argmax,
                || {
                    model
                        .argmax_on_device(logits.offset(j * vocab * elem), 0)
                        .unwrap_or(u32::MAX)
                },
            );
            crate::scheduler::sample_step::penalty_params_for(
                a,
                crate::scheduler::sample_step::PositionKind::Verify,
                0.0,
                None,
                base_bias,
                sched.watchdog.min_reasoning_floor,
            )
        })
        .collect();
    let greedy: Vec<bool> = refs
        .iter()
        .zip(pen.iter())
        .map(|(a, p)| {
            sched.levers.mtp_boot_argmax
                && verify_ctx.sampling.fast_greedy_grammar
                && (a.temperature == 0.0 || verify_ctx.sampling.force_temp_zero)
                && a.grammar_state.is_none()
                && crate::scheduler::fast_greedy::classify_penalties(p)
                    == crate::scheduler::fast_greedy::PenaltyGate::Neutral
        })
        .collect();
    let n_greedy = greedy.iter().filter(|&&g| g).count();
    let batch_toks: Option<Vec<u32>> = if n_greedy >= 2 {
        match model.argmax_batch(logits, n, 0) {
            Ok(t) => {
                if sched
                    .io
                    .tel
                    .stats()
                    .once("log:mtp_bootstrap_batched_argmax")
                {
                    tracing::info!(
                        "MTP bootstrap batched argmax ENGAGED (n={n}, greedy_rows={n_greedy}): \
                         one launch + one D2H replaces {n_greedy} single-CTA scans + syncs"
                    );
                }
                Some(t)
            }
            Err(e) => {
                // 2026-09-25: Not fatal: every row falls back to the per-row call.
                tracing::error!("batched bootstrap argmax_batch (n={n}): {e:#}");
                None
            }
        }
    } else {
        None
    };

    // 2026-09-25: Per-row sample and emit. `propose_rows[j] = Some(j)` marks a
    // sequence that should propose drafts.
    let mut propose_rows: Vec<Option<usize>> = vec![None; n];
    for (j, a) in refs.iter_mut().enumerate() {
        let row_logits = logits.offset(j * vocab * elem);
        let batched = batch_toks.as_ref().filter(|_| greedy[j]).map(|t| t[j]);
        let tok = match batched {
            // 2026-09-29: A146, spec-in-think parity: a thinking row takes
            // decode's full host pipeline (twin of the per-sequence bootstrap
            // in `mtp_step/serial_bootstrap.rs`).
            // 2026-10-04: A95: so does the first token after a `</think>`
            // (`bootstrap_takes_pipeline`).
            _ if crate::scheduler::verify_pipeline_helper::bootstrap_takes_pipeline(
                a,
                sched.levers.think_end_at_token0,
            ) =>
            {
                match crate::scheduler::verify_pipeline_helper::pick_decode_row_with_pipeline(
                    model, row_logits, a, verify_ctx,
                ) {
                    Some(t) => t,
                    None => {
                        tracing::error!("batched bootstrap pipeline pick: D2H failed");
                        a.finished = true;
                        continue;
                    }
                }
            }
            Some(t) => t,
            None => {
                let history = crate::scheduler::sample_step::penalty_history_scope(
                    &a.output_tokens,
                    a.tool_call_end_token,
                )
                .to_vec();
                match sample_token_with_grammar(
                    model,
                    row_logits,
                    a.temperature,
                    a.top_k,
                    a.top_p,
                    &[],
                    a.grammar_state.as_mut(),
                    &pen[j],
                    &history,
                    &verify_ctx.sampling,
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("batched bootstrap sample error: {e:#}");
                        a.finished = true;
                        continue;
                    }
                }
            }
        };
        let lp = if let Some(k) = a.top_logprobs {
            extract_single_logprobs(model, row_logits, tok, k)
        } else {
            None
        };
        emit_token(a, tok, lp, sched);
        if a.finished {
            continue;
        }
        a.last_token = tok;
        crate::scheduler::adaptive_spec::tick_serial(a, sched);
        // 2026-09-25: `spec_allowed` can change re-probe state: call it once.
        if crate::scheduler::adaptive_spec::spec_allowed(a, sched) {
            propose_rows[j] = Some(j);
        }
    }

    // 2026-09-25: Stash every proposing row before any propose: a propose
    // overwrites the shared `hidden_states` buffer
    // (`Model::stash_verify_hidden_rows`). Stash slot s <-> proposing[s].
    let proposing: Vec<usize> = (0..n).filter(|&j| propose_rows[j].is_some()).collect();
    let mut stash_ok = false;
    if !proposing.is_empty() {
        match model.stash_verify_hidden_rows(&proposing, 0) {
            Ok(()) => stash_ok = true,
            Err(e) => tracing::error!("batched bootstrap stash_verify_hidden_rows: {e:#}"),
        }
    }

    // 2026-09-25: Batched propose in groups of at most
    // `mtp_propose_batch_max()`. Sequences with a grammar, or whose draft count
    // is clamped below `ladder_nd`, and groups of one, use the per-sequence
    // propose below.
    if stash_ok {
        let group_cap = model.mtp_propose_batch_max().max(1);
        // 2026-09-25: Logged once: the values that decide whether proposes batch.
        if sched.io.tel.stats().once("log:dflash_batch_propose_gate") {
            tracing::debug!(
                group_cap,
                proposing = proposing.len(),
                ladder_nd,
                "DFlash batched propose gate (first tick)"
            );
        }
        let batchable: Vec<usize> = (0..proposing.len())
            .filter(|&s| {
                let a = &refs[proposing[s]];
                a.grammar_state.is_none()
                    && crate::scheduler::spec_step::effective_drafts_under_grammar(a, ladder_nd)
                        == ladder_nd
            })
            .collect();
        let mut done = vec![false; proposing.len()];
        // 2026-09-25: Draft confidences (D-Cut's ranking key) are requested only
        // when `dcut_enabled`.
        let want_conf = sched.levers.dcut_enabled;
        let mut conf: Vec<Vec<f32>> = Vec::new();
        if group_cap >= 2 && ladder_nd >= 1 {
            for group in batchable.chunks(group_cap) {
                if group.len() < 2 {
                    continue;
                }
                let tokens: Vec<u32> = group
                    .iter()
                    .map(|&s| refs[proposing[s]].last_token)
                    .collect();
                let positions: Vec<usize> = group
                    .iter()
                    .map(|&s| refs[proposing[s]].seq.seq_len)
                    .collect();
                let stash_idx: Vec<usize> = group.to_vec();
                let result = {
                    let mut seq_refs: Vec<&mut SequenceState> = Vec::with_capacity(group.len());
                    let mut it = refs.iter_mut();
                    let mut prev = 0usize;
                    for (g, &s) in group.iter().enumerate() {
                        let row = proposing[s];
                        let step = if g == 0 { row } else { row - prev - 1 };
                        let a = it.nth(step).expect("group row within batch");
                        seq_refs.push(&mut a.seq);
                        prev = row;
                    }
                    model.run_mtp_propose_batched(
                        &tokens,
                        &positions,
                        &stash_idx,
                        ladder_nd,
                        &mut seq_refs,
                        0,
                        want_conf.then_some(&mut conf),
                    )
                };
                match result {
                    Ok(Some(all)) => {
                        for (g, &s) in group.iter().enumerate() {
                            if !all[g].is_empty() {
                                refs[proposing[s]].pending_drafts = all[g].clone();
                                refs[proposing[s]].pending_draft_conf =
                                    conf.get(g).cloned().unwrap_or_default();
                            }
                            done[s] = true;
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        // 2026-09-25: These sequences get no drafts this
                        // step: `done` keeps them out of the per-sequence
                        // propose below.
                        log_propose_batched_err("batched bootstrap run_mtp_propose_batched", &e);
                        for &s in group {
                            done[s] = true;
                        }
                    }
                }
            }
        }
        for (s, &row) in proposing.iter().enumerate() {
            if done[s] {
                continue;
            }
            if let Err(e) = model.save_hidden_for_mtp_from_stash(s, 0) {
                tracing::error!("batched bootstrap save_hidden_for_mtp_from_stash({s}): {e:#}");
                continue;
            }
            let a = &mut refs[row];
            let mask = mtp_grammar_mask_for(a);
            let eff = crate::scheduler::spec_step::effective_drafts_under_grammar(a, ladder_nd);
            match model.run_mtp_propose_multi(
                a.last_token,
                a.seq.seq_len,
                eff,
                &mut a.seq,
                0,
                mask.as_deref(),
            ) {
                Ok(d) if !d.is_empty() => a.pending_drafts = d,
                Ok(_) => tracing::warn!("MTP propose returned empty"),
                Err(e) => tracing::error!("run_mtp_propose_multi: {e:#}"),
            }
        }
    }

    for a in refs.iter_mut() {
        if a.finished {
            continue;
        }
        if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
            tracing::error!("batched bootstrap start_checkpoint_async: {e:#}");
        }
    }

    if sched.io.tel.stats().once("log:mtp_batched_bootstrap") {
        tracing::info!(
            "MTP batched bootstrap ENGAGED (n={n}): one decode_batch + batched propose \
             replaces {n} M=1 decodes + {n} drafter forwards per draft position"
        );
    }
}
