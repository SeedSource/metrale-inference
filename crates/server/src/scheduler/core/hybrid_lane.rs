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

/// 2026-10-14: Before an MTP-mode `step_mtp` on a hybrid serve: each active sequence's slot,
/// pending draft count and length (empty off a hybrid serve), for [`note_mtp_step`].
pub(super) fn mtp_snapshot(h: Option<&HybridState>, active: &[ActiveSeq]) -> Vec<MtpSnap> {
    if h.is_none() {
        return Vec::new();
    }
    active
        .iter()
        .map(|a| MtpSnap {
            slot: a.seq.slot_idx,
            pending: a.pending_drafts.len(),
            seq_len: a.seq.seq_len,
        })
        .collect()
}

/// 2026-10-14: One active sequence before an MTP step ([`mtp_snapshot`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MtpSnap {
    pub slot: usize,
    pub pending: usize,
    pub seq_len: usize,
}

/// 2026-10-14: After that step: count each sequence's MTP acceptance into the hybrid counters
/// (`HybridState::note_mtp_verify`, post-switch vs other). Proposed = the drafts pending at the
/// step's start; accepted = the step's growth minus the bonus, capped there. Host-side only.
pub(super) fn note_mtp_step(h: Option<&mut HybridState>, active: &[ActiveSeq], snap: &[MtpSnap]) {
    let Some(h) = h else {
        return;
    };
    for (slot, proposed, accepted) in mtp_step_accepts(active, snap) {
        h.note_mtp_verify(slot, proposed, accepted);
    }
}

/// 2026-10-14: `(slot, proposed, accepted)` per snapshotted sequence still active (pure).
fn mtp_step_accepts(active: &[ActiveSeq], snap: &[MtpSnap]) -> Vec<(usize, usize, usize)> {
    snap.iter()
        .filter_map(|s| {
            let a = active.iter().find(|a| a.seq.slot_idx == s.slot)?;
            let grown = a.seq.seq_len.saturating_sub(s.seq_len);
            Some((s.slot, s.pending, grown.saturating_sub(1).min(s.pending)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::hybrid_mode::HybridConfig;
    use crate::scheduler::test_support::active_seq;

    /// 2026-10-14: The lane's MTP acceptance count: drafts pending at the step's start, accepted
    /// = growth minus the bonus (capped), a bootstrap (no drafts) not counted; nothing off a
    /// hybrid serve.
    #[test]
    fn mtp_step_counts_the_drafts_pending_at_its_start() {
        let (mut a0, _r0) = active_seq(0, 3);
        let (mut a1, _r1) = active_seq(1, 3);
        a0.pending_drafts = vec![7, 8, 9];
        a1.pending_drafts.clear();
        let mut active = vec![a0, a1];
        let mut h = HybridState::new(HybridConfig {
            mtp_num_drafts: 3,
            dflash_num_drafts: 7,
            dwell: std::time::Duration::from_millis(2000),
            rate_gate: None,
        });
        assert!(mtp_snapshot(None, &active).is_empty());
        let snap = mtp_snapshot(Some(&h), &active);
        active[0].seq.seq_len += 3;
        active[1].seq.seq_len += 2;
        assert_eq!(mtp_step_accepts(&active, &snap), vec![(0, 3, 2), (1, 0, 0)]);
        note_mtp_step(Some(&mut h), &active, &snap);
        let o = h.counters.other_mtp;
        assert_eq!((o.accepted, o.proposed, o.steps), (2, 3, 1));
        assert_eq!(h.counters.post_switch_mtp.steps, 0);
        // 2026-10-14: A sequence gone from the batch is skipped.
        active.truncate(1);
        assert_eq!(mtp_step_accepts(&active, &snap), vec![(0, 3, 2)]);
    }
}
