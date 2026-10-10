// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The decode lane's hybrid step (`METRALE_HYBRID_DFLASH=1`): gather the inputs of
//! [`crate::scheduler::hybrid_mode::HybridState::decide`] from the active batch, and on a mode
//! change drop the pending drafts and hand the new mode to the model.
//!
//! Owner: scheduler.
//! Invariants:
//! - The model takes the new mode (`Model::set_spec_step_mode`, which sends it to the worker
//!   ranks) before this step's first propose or verify.
//! - On a mode change every active sequence's pending drafts are dropped, as the MTP gate's
//!   switch to plain decode drops them: a step never verifies drafts of the other proposer.

use super::*;

use crate::scheduler::hybrid_mode::{HybridState, SoleSeq};
use metrale_model_layers::speculative::hybrid::SpecStepMode;

/// 2026-10-10: Decide this step's mode and apply it. Returns `Plain` when the model refused the
/// change (it keeps its previous mode, so this step must not propose); the next step retries.
pub(super) fn decide_and_apply(
    h: &mut HybridState,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    spec_slot_cap: usize,
    resume_guard: u32,
) -> SpecStepMode {
    let model = sched.io.dev.model();
    let sole = (active.len() == 1).then(|| SoleSeq {
        slot: active[0].seq.slot_idx,
        prompt_len: active[0].seq.prompt_len,
        generated: active[0].output_tokens.len(),
    });
    // 2026-10-10: The sole sequence may take a DFlash step: no grammar, a slot the speculative
    // branches cover, the model's DFlash readiness (state + wide SSM slot), and the DFlash lane's
    // own dispatch rule (inside `<think>` only with `METRALE_DFLASH_SPEC_THINK=1`). Otherwise it
    // stays on MTP, which keeps its own spec-in-think default.
    let sole_dflash_ok = active.len() == 1 && {
        let a = &active[0];
        a.grammar_state.is_none()
            && a.seq.slot_idx < spec_slot_cap
            && model.hybrid_dflash_ready(&a.seq)
            && metrale_speculative::mtp_gate::spec_dispatch_eligible(
                a.inside_thinking,
                a.post_think_emitted,
                a.output_tokens.len() as u32,
                a.suppress_tool_call,
                a.disable_mtp,
                metrale_speculative::mtp_gate::spec_think_for_lane(
                    true,
                    false,
                    sched.levers.dflash_spec_think,
                ),
                resume_guard,
                true,
            )
    };
    let mode = h.decide(
        active.len(),
        sched.levers.mtp_max_seqs,
        sole,
        sole_dflash_ok,
        sched.io.clock.now(),
    );
    let prev = h.last();
    if prev == Some(mode) {
        return mode;
    }
    if prev.is_some() {
        for a in active.iter_mut() {
            a.pending_drafts.clear();
            a.pending_draft_conf.clear();
        }
        // 2026-10-10: Verify commits may still run on the secondary stream (as at the gate's
        // switch to plain decode).
        if let Err(e) = sched.io.dev.apply(io::Effect::SyncSecondary) {
            tracing::error!("hybrid mode switch sync_secondary: {e}");
        }
    }
    if let Err(e) = model.set_spec_step_mode(mode) {
        tracing::error!(
            "HYBRID DFLASH: set_spec_step_mode({}) failed: {e:#} — this step decodes plain",
            mode.name()
        );
        return SpecStepMode::Plain;
    }
    h.commit(mode);
    mode
}
