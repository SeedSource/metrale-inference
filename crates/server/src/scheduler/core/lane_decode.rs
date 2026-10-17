// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The decode lane: one step of n-gram, self-speculative, MTP (through
//! its runtime gate) or plain decode, the plain step pipelined when the router
//! allows it. Skipped on a tick whose prefill lane already ran a mixed step.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

impl SchedulerCore {
    pub(super) fn decode_lane(&mut self) -> LaneVerdict {
        let pipelined = self.pipelining_configured();
        let commit_tokens = self.commit_tokens();
        let Self {
            ctx: sched,
            active,
            prefilling,
            swapped,
            preempted,
            mtp_gate,
            ngram_proposer,
            pipeline,
            hybrid,
            ..
        } = self;
        let prefill_stream = self.prefill_stream;
        let prefill_event = self.prefill_event;
        let think_end_token = self.think_end_token;
        let think_start_token = self.think_start_token;
        let code_fence_token = self.code_fence_token;
        let tool_call_start_token = self.tool_call_start_token;
        let tool_call_end_token = self.tool_call_end_token;
        let spec_slot_cap = self.spec_slot_cap;
        let use_mtp = self.use_mtp;
        let use_ngram_speculative = self.use_ngram_speculative;
        let use_self_speculative = self.use_self_speculative;
        let num_drafts = self.num_drafts;
        let dflash_verify_raw_argmax = self.dflash_verify_raw_argmax;
        let mtp_spec_think_default = self.mtp_spec_think_default;
        let adaptive_sampling = self.adaptive_sampling;
        if !self.did_mixed_step {
            // 2026-09-25: Order this decode on the default stream after the work queued
            // on the prefill stream (a device-side event wait).
            if !prefilling.is_empty() {
                let _ = sched
                    .io
                    .dev
                    .model()
                    .record_event(prefill_event, prefill_stream);
                let _ = sched
                    .io
                    .dev
                    .model()
                    .stream_wait_event(sched.io.dev.model().default_stream(), prefill_event);
            }

            // 2026-09-29: A146: a spec-in-think trail lives for ONE verify
            // commit run. Entries left by a partial accept must never reach a
            // later step's emit (the fast-path, raw-argmax and bootstrap emits
            // run no window), so drop them at every step boundary.
            // `verify_pick_all_with_pipeline` also clears on entry.
            for a in active.iter_mut() {
                a.spec_think_trail.clear();
            }

            // 2026-09-25: The LogitsContext the speculative steps below pass to the
            // logits processors: special-token ids, masks and sampling levers.
            let verify_ctx = crate::scheduler::logit_processors::LogitsContext {
                watchdog: sched.watchdog,
                scratch: &sched.scratch,
                tel: &*sched.io.tel,
                clock: &*sched.io.clock,
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                code_fence_token,
                verify_pos: 0,
                boundary_mask: sched.masks.boundary.clone(),
                mid_word_mask: sched.masks.mid_word.clone(),
                sampling: sched.levers.sampling(),
            };
            // 2026-09-25: `METRALE_DFLASH_RESUME_GUARD` (tokens, default 0): a sequence
            // is not speculated until it has emitted this many tokens after
            // `</think>` (see `spec_dispatch_eligible`).
            let dflash_resume_guard = sched.levers.dflash_resume_guard;
            // 2026-10-10: Hybrid serve (`hybrid_mode`): this step's proposer, decided here on
            // rank 0 and handed to the model (which sends it to the workers) before any propose.
            // `None` on every other serve, which keeps the serve-wide flag and draft count.
            let step_mode = hybrid.as_mut().map(|h| {
                super::hybrid_lane::decide_and_apply(
                    h,
                    active,
                    sched,
                    spec_slot_cap,
                    dflash_resume_guard,
                )
            });
            let (dflash_verify_raw_argmax, num_drafts) = crate::scheduler::hybrid_mode::step_flags(
                hybrid.as_ref(),
                step_mode,
                dflash_verify_raw_argmax,
                num_drafts,
            );
            // 2026-09-29: A146: spec-in-think, per lane and per model. The MTP
            // lane speculates inside `<think>` by default only on models whose
            // `mtp_spec_think_default` is on (GLM-5.3);
            // `METRALE_MTP_SPEC_THINK=0` or `METRALE_DFLASH_SPEC_THINK=0`
            // disables it, `=1` opts other models in. The DFlash lane
            // (`dflash_verify_raw_argmax` = `args.dflash`, every DFlash verify
            // mode) stays serial inside `<think>` unless
            // `METRALE_DFLASH_SPEC_THINK=1` (`spec_dispatch_eligible`).
            let spec_think = metrale_speculative::mtp_gate::spec_think_for_lane(
                dflash_verify_raw_argmax,
                sched.levers.mtp_spec_think(mtp_spec_think_default),
                sched.levers.dflash_spec_think,
            );
            // 2026-09-25: Every speculative branch also requires each active slot to be
            // below `spec_slot_cap` (see `SchedulerCore::new`); otherwise the batch
            // takes the plain-decode branch.
            let spec_slots_covered = active.iter().all(|a| a.seq.slot_idx < spec_slot_cap);
            // 2026-09-25: MTP also needs `active.len() <= mtp_max_seqs`.
            // `note_width_regime` only records this decision for reporting.
            // 2026-10-10: A hybrid PLAIN step never speculates.
            let spec_width_ok = active.len() <= sched.levers.mtp_max_seqs
                && step_mode
                    != Some(metrale_model_layers::speculative::hybrid::SpecStepMode::Plain);
            // 2026-10-10: `METRALE_MTP_POOL_CAP` evidence: MTP steps lost only because an
            // active sequence sits on a slot the verify pools do not cover.
            if use_mtp && spec_width_ok && !spec_slots_covered {
                let n = metrale_model_layers::ssm_reserve::note_mtp_uncovered_step();
                if n.is_power_of_two() {
                    let slots: Vec<usize> = active.iter().map(|a| a.seq.slot_idx).collect();
                    tracing::info!(
                        "MTP steps lost to uncovered slots: {n} (cover {spec_slot_cap}, active slots {slots:?})"
                    );
                }
            }
            if use_mtp {
                sched.rung.note_width_regime(
                    active.len(),
                    spec_width_ok,
                    sched.levers.mtp_max_seqs,
                );
            }
            if use_ngram_speculative
                && active.len() == 1
                && spec_slots_covered
                && active[0].grammar_state.is_none()
            {
                if let Some(proposer) = ngram_proposer {
                    step_ngram(sched.io.dev.model(), active, sched, proposer, &verify_ctx);
                }
            } else if use_self_speculative
                && active.len() == 1
                && spec_slots_covered
                && active[0].grammar_state.is_none()
            {
                // 2026-09-25: The draft count is clamped to the slot's verify capacity,
                // as `step_mtp` does.
                let nd = metrale_speculative::spec_capacity::clamp_drafts_to_slot_capacity(
                    num_drafts,
                    active
                        .iter()
                        .map(|a| sched.io.dev.model().mtp_slot_draft_capacity(a.seq.slot_idx)),
                );
                step_self_spec(sched.io.dev.model(), active, sched, nd, &verify_ctx);
            } else if use_mtp
                && spec_width_ok
                && spec_slots_covered
                && (
                    // 2026-09-25: Every active sequence must be eligible, not only
                    // `active[0]`.
                    active.iter().all(|a| {
                        metrale_speculative::mtp_gate::spec_dispatch_eligible(
                            a.inside_thinking,
                            a.post_think_emitted,
                            a.output_tokens.len() as u32,
                            a.suppress_tool_call,
                            a.disable_mtp,
                            spec_think,
                            dflash_resume_guard,
                            dflash_verify_raw_argmax,
                        )
                    })
                )
            {
                // 2026-10-14 (race-memory, A184): a batched MTP step's retryable KV refusal,
                // handled after the step's accounting below.
                let mut spec_refusal: Option<String> = None;
                if let Some(h) = hybrid.as_mut() {
                    h.count_step(
                        step_mode.unwrap_or(
                            metrale_model_layers::speculative::hybrid::SpecStepMode::Mtp,
                        ),
                    );
                }
                // 2026-09-25: The MTP gate chooses each step (plain decode or MTP verify)
                // by measured delivered throughput (`metrale_speculative::mtp_gate`).
                // 2026-10-10: A hybrid DFLASH step bypasses the gate, as the C<=2 DFlash pin
                // would, and is not recorded in it: the gate measures the MTP lane only.
                let hybrid_dflash_step = step_mode
                    == Some(metrale_model_layers::speculative::hybrid::SpecStepMode::Dflash);
                if hybrid_dflash_step {
                    let lens_before: Vec<usize> = active.iter().map(|a| a.seq.seq_len).collect();
                    // 2026-10-16 (race-memory, union line c31df): a hybrid DFLASH step's KV
                    // refusal takes the same preemption as an MTP step's (A184); dropped, the
                    // batched DFlash verify would put its drafts back and stall every step.
                    spec_refusal = step_mtp(
                        sched.io.dev.model(),
                        active,
                        sched,
                        num_drafts,
                        &verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                    for (a, &b) in active.iter_mut().zip(lens_before.iter()) {
                        a.mtp_acct
                            .record_verify_emitted(a.seq.seq_len.saturating_sub(b));
                    }
                } else if let Some(gate) = mtp_gate.as_mut() {
                    if gate.maybe_remeasure(active[0].seq.seq_len) {
                        for a in active.iter_mut() {
                            a.mtp_acct.note_regime_reprobe();
                        }
                    }
                    gate.note_depth(active[0].seq.seq_len);
                    // 2026-09-25: Spec-entry pin (`METRALE_SPEC_ENTRY_PIN`): while any
                    // active sequence has emitted fewer than that many tokens after
                    // `</think>`, run the verify step even where the gate would run
                    // plain decode. Pinned steps are not recorded in the gate.
                    let min_post_think_emitted = active
                        .iter()
                        .map(|a| a.post_think_emitted)
                        .min()
                        .unwrap_or(u32::MAX);
                    // 2026-09-25: DFlash pin (`SchedLevers::dflash_gate_pin_c2`, whose doc
                    // gives the reason): with raw-argmax DFlash verify and at most 2
                    // active sequences, always run the verify step. Not recorded in
                    // the gate either.
                    let dflash_pin = dflash_verify_raw_argmax
                        && active.len() <= 2
                        && sched.levers.dflash_gate_pin_c2;
                    if (dflash_pin
                        || metrale_speculative::mtp_gate::entry_pin_forces_verify(
                            min_post_think_emitted,
                            sched.levers.spec_entry_pin_tokens,
                        ))
                        && gate.next_step()
                            == metrale_speculative::mtp_gate::GateStep::MeasureDecode
                    {
                        let lens_before: Vec<usize> =
                            active.iter().map(|a| a.seq.seq_len).collect();
                        // 2026-10-14: Hybrid MTP-mode acceptance, post-switch vs other (`hybrid_lane`).
                        let hy_snap = super::hybrid_lane::mtp_snapshot(hybrid.as_ref(), active);
                        spec_refusal = step_mtp(
                            sched.io.dev.model(),
                            active,
                            sched,
                            num_drafts,
                            &verify_ctx,
                            dflash_verify_raw_argmax,
                        );
                        super::hybrid_lane::note_mtp_step(hybrid.as_mut(), active, &hy_snap);
                        for (a, &b) in active.iter_mut().zip(lens_before.iter()) {
                            a.mtp_acct
                                .record_verify_emitted(a.seq.seq_len.saturating_sub(b));
                        }
                    } else {
                        match gate.next_step() {
                            metrale_speculative::mtp_gate::GateStep::MeasureDecode => {
                                let t0 = sched.io.clock.now();
                                step_decode_only(
                                    active,
                                    think_end_token,
                                    think_start_token,
                                    code_fence_token,
                                    tool_call_start_token,
                                    tool_call_end_token,
                                    adaptive_sampling,
                                    sched,
                                    sched.io.spill.as_deref(),
                                    swapped,
                                    preempted,
                                );
                                // 2026-09-25: A plain decode step emits one token per
                                // sequence, so it is charged `active.len()` tokens.
                                gate.record_decode(
                                    sched.io.clock.now().saturating_duration_since(t0),
                                    active.len(),
                                );
                                for a in active.iter_mut() {
                                    a.mtp_acct.record_serial();
                                }
                                // 2026-09-25: Single-sequence batches only (the ring has one
                                // label space): copy row 0's hidden into the MTP catch-up
                                // ring at label `seq_len`, which this step has already
                                // advanced past its input token. A no-op when the model
                                // allocated no ring.
                                if active.len() == 1
                                    && let Err(e) =
                                        sched.io.dev.apply(io::Effect::SaveHiddenCatchup {
                                            row: 0,
                                            pos: active[0].seq.seq_len,
                                        })
                                {
                                    tracing::warn!("save_hidden_for_catchup: {e}");
                                }
                            }
                            metrale_speculative::mtp_gate::GateStep::MeasureVerify => {
                                // 2026-09-25: Charged to the MTP arm: the tokens every
                                // active sequence emitted, bootstrap-only steps included.
                                let lens_before: Vec<usize> =
                                    active.iter().map(|a| a.seq.seq_len).collect();
                                let t0 = sched.io.clock.now();
                                // 2026-10-14: Hybrid MTP-mode acceptance, post-switch vs other (`hybrid_lane`).
                                let hy_snap =
                                    super::hybrid_lane::mtp_snapshot(hybrid.as_ref(), active);
                                spec_refusal = step_mtp(
                                    sched.io.dev.model(),
                                    active,
                                    sched,
                                    num_drafts,
                                    &verify_ctx,
                                    dflash_verify_raw_argmax,
                                );
                                super::hybrid_lane::note_mtp_step(
                                    hybrid.as_mut(),
                                    active,
                                    &hy_snap,
                                );
                                let emitted: usize = active
                                    .iter()
                                    .zip(lens_before.iter())
                                    .map(|(a, &b)| a.seq.seq_len.saturating_sub(b))
                                    .sum();
                                gate.record_verify_step(
                                    sched.io.clock.now().saturating_duration_since(t0),
                                    emitted,
                                    active.len(),
                                );
                                for (a, &b) in active.iter_mut().zip(lens_before.iter()) {
                                    a.mtp_acct
                                        .record_verify_emitted(a.seq.seq_len.saturating_sub(b));
                                }
                            }
                        }
                    }
                    // 2026-09-25: When the gate switches to plain decode, drop pending
                    // drafts and order the secondary stream before the next plain
                    // decode. Switching back needs nothing: the next MTP step
                    // bootstraps from empty drafts.
                    if gate.take_fresh_decision()
                        == Some(metrale_speculative::mtp_gate::GateDecision::DisableMtp)
                    {
                        for a in active.iter_mut() {
                            a.pending_drafts.clear();
                            a.pending_draft_conf.clear();
                        }
                        if let Err(e) = sched.io.dev.apply(io::Effect::SyncSecondary) {
                            tracing::error!("mtp-gate→decode sync_secondary: {e}");
                        }
                    }
                } else {
                    // 2026-09-25: Gate disarmed (`--mtp-gate force` or
                    // `METRALE_MTP_GATE_FORCE`): always MTP.
                    let lens_before: Vec<usize> = active.iter().map(|a| a.seq.seq_len).collect();
                    // 2026-10-14: Hybrid MTP-mode acceptance, post-switch vs other (`hybrid_lane`).
                    let hy_snap = super::hybrid_lane::mtp_snapshot(hybrid.as_ref(), active);
                    spec_refusal = step_mtp(
                        sched.io.dev.model(),
                        active,
                        sched,
                        num_drafts,
                        &verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                    super::hybrid_lane::note_mtp_step(hybrid.as_mut(), active, &hy_snap);
                    for (a, &b) in active.iter_mut().zip(lens_before.iter()) {
                        a.mtp_acct
                            .record_verify_emitted(a.seq.seq_len.saturating_sub(b));
                    }
                }
                // 2026-10-14 (race-memory, A184): preempt a victim so the refused rows retry
                // (or fail the lone one with an error frame), as a plain decode step does.
                if let Some(msg) = spec_refusal {
                    crate::scheduler::preempt::preempt_after_spec_refusal(
                        sched, active, &msg, swapped, preempted,
                    );
                }
            } else {
                if let Some(h) = hybrid.as_mut() {
                    h.count_step(metrale_model_layers::speculative::hybrid::SpecStepMode::Plain);
                }
                // 2026-09-25: Plain decode. With MTP on, drop any pending drafts first.
                if use_mtp {
                    for a in active.iter_mut() {
                        a.pending_drafts.clear();
                        a.pending_draft_conf.clear();
                    }
                    // 2026-09-25: Verify commits may still run on the secondary stream:
                    // make the default stream wait for it (a device-side event wait).
                    if let Err(e) = sched.io.dev.apply(io::Effect::SyncSecondary) {
                        tracing::error!("mtp→decode sync_secondary: {e}");
                    }
                }
                if pipelined {
                    pipeline::step_decode_pipelined(
                        active,
                        pipeline,
                        commit_tokens,
                        sched,
                        sched.io.spill.as_deref(),
                        swapped,
                        preempted,
                    );
                } else {
                    step_decode_only(
                        active,
                        think_end_token,
                        think_start_token,
                        code_fence_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        adaptive_sampling,
                        sched,
                        sched.io.spill.as_deref(),
                        swapped,
                        preempted,
                    );
                }
                for a in active.iter_mut() {
                    a.mtp_acct.record_serial();
                }
            }
        }

        LaneVerdict::Proceed
    }
}
