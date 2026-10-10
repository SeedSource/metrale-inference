// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `METRALE_DFLASH_CONF_TAU=<t>`: DFlash2 drafter confidence truncation at C=1.
//!
//! After the DFlash2 head proposes its block on rank 0, `step_mtp`'s per-sequence loop keeps
//! the drafts while the drafter's top-1 probability `p_i >= t`, stopping at the first position
//! below `t` and keeping at least one (`conf_log::keep_count`). `p_i` is the
//! `METRALE_DFLASH_CONF_LOG` p: a softmax over the selector's top-16 logits
//! (`metrale_model_arch::dflash_head::conf_log`). The verify then runs at width `kept + 1`:
//! `step_verify_dflash` builds `[last_token, drafts..]` and sends that `k` to the EP worker in
//! the existing `EP_CMD_VERIFY_KGAMMA` words.
//!
//! Owner: scheduler.
//! Invariants:
//! - Exact under the raw-argmax verify: only how many drafts are checked changes, never a
//!   drafted, verified or emitted token's value.
//! - v1 is C=1 only: it applies when exactly one sequence is active, the serve verifies DFlash on
//!   the raw argmax, and the row has no grammar. At C>=2 it is inert (logged once): the batched
//!   DFlash verify needs one k per batch.
//! - A block it shortens keeps the K=gamma DFlash arm ([`keeps_dflash_arm`]); the MTP K2..K4
//!   arms never commit the drafter context.
//! - The lever is read once per process ([`tau`]); serve setup calls [`init`] on rank 0 so the
//!   ENGAGED line is logged at startup.

use std::sync::OnceLock;

static TAU: OnceLock<f32> = OnceLock::new();

/// 2026-10-09: Resolve the lever (first call only) and log `ENGAGED` once when it is on.
pub(crate) fn init() -> f32 {
    *TAU.get_or_init(|| {
        let t = metrale_model_arch::dflash_head::levers::DFlashLevers::from_env().dflash_conf_tau;
        if t > 0.0 {
            tracing::info!(
                "METRALE_DFLASH_CONF_TAU: ENGAGED (tau={t}) - C=1 DFlash2 blocks verify only the \
                 drafts before the first top-1 prob below tau (min 1); inert at C>=2"
            );
        }
        t
    })
}

/// 2026-10-09: The threshold, `0.0` when off.
pub(crate) fn tau() -> f32 {
    init()
}

/// 2026-10-09: Whether this step may truncate: lever on, raw-argmax DFlash, and `n_active == 1`.
/// At C>=2 with the lever on, logs the inert notice once.
pub(crate) fn step_applies(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    n_active: usize,
    dflash_verify_raw_argmax: bool,
) -> bool {
    if !dflash_verify_raw_argmax || tau() <= 0.0 {
        return false;
    }
    if n_active != 1 {
        if sched.io.tel.stats().once("log:dflash_conf_tau_inert") {
            tracing::info!(
                "METRALE_DFLASH_CONF_TAU: inert at C={n_active} (v1 truncates only at C=1; the \
                 batched DFlash verify needs one k per batch)"
            );
        }
        return false;
    }
    true
}

/// 2026-10-09: Shorten `drafts` to the kept count. No matching p record (a batched propose,
/// another sequence's record, a non-DFlash2 drafter) leaves the block unchanged.
pub(crate) fn truncate(drafts: &mut Vec<u32>) {
    use metrale_model_arch::dflash_head::conf_log;
    if let Some(p) = conf_log::p_for(drafts) {
        let keep = conf_log::keep_count(&p, drafts.len(), tau());
        drafts.truncate(keep);
    }
}

/// 2026-10-09: Whether a grammarless raw-argmax DFlash block takes the K=gamma DFlash arm
/// whatever its length: under the GLM DFlash C>=2 pin (792236f3), or when CONF_TAU truncated
/// this step. Otherwise `serial_verify_plan`'s arm stands (fewer than 4 drafts take K2..K4).
pub(crate) fn keeps_dflash_arm(
    dflash_verify_raw_argmax: bool,
    multi_pinned: bool,
    conf_tau_applies: bool,
    grammarless: bool,
) -> bool {
    dflash_verify_raw_argmax && grammarless && (multi_pinned || conf_tau_applies)
}

#[cfg(test)]
mod tests {
    use super::keeps_dflash_arm;
    use metrale_speculative::spec_capacity::{SerialArm, serial_verify_plan};

    /// 2026-10-09: The arm `step_mtp`'s per-sequence loop runs for `n` drafts.
    fn arm(n: usize, pinned: bool, tau_applies: bool, grammarless: bool) -> SerialArm {
        let (_, planned) = serial_verify_plan(n, 7, true);
        if keeps_dflash_arm(true, pinned, tau_applies, grammarless) {
            SerialArm::DFlash
        } else {
            planned
        }
    }

    #[test]
    fn conf_tau_keeps_the_dflash_arm_for_short_blocks() {
        for n in 1..=3 {
            assert_eq!(arm(n, false, true, true), SerialArm::DFlash, "{n} drafts");
            // 2026-10-09: Lever off and no pin: the old routing (K2..K4) stands.
            assert_ne!(
                arm(n, false, false, true),
                SerialArm::DFlash,
                "{n} drafts, off"
            );
            // 2026-10-09: The 792236f3 pin alone still keeps it.
            assert_eq!(
                arm(n, true, false, true),
                SerialArm::DFlash,
                "{n} drafts, pinned"
            );
            // 2026-10-09: A grammar row keeps the arm it took before.
            assert_ne!(
                arm(n, false, true, false),
                SerialArm::DFlash,
                "{n} drafts, grammar"
            );
        }
        for n in 4..=7 {
            assert_eq!(arm(n, false, false, true), SerialArm::DFlash, "{n} drafts");
        }
        // 2026-10-09: Never on an MTP (non raw-argmax DFlash) serve.
        assert!(!keeps_dflash_arm(false, true, true, true));
    }
}
