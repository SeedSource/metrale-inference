// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Hybrid speculative decode (`METRALE_HYBRID_DFLASH=1`, issue #82): rank 0's choice
//! of proposer for each decode step, and the per-mode counters.
//!
//! A hybrid serve holds the MTP head and a DFlash drafter. Before every decode step the decode
//! lane asks [`HybridState::decide`] for the step's [`SpecStepMode`]:
//! - `Dflash` when exactly one sequence is active, it may take a DFlash step (grammarless, a
//!   DFlash state and the wide SSM slot, and the DFlash lane would speculate it now) and it has
//!   been alone for the dwell (policy D2s, `METRALE_HYBRID_DFLASH_DWELL_MS`, default 2000 ms);
//! - else `Mtp` while `active <= METRALE_MTP_MAX_SEQS`;
//! - else `Plain`.
//!
//! The lane sends the mode to the model (`Model::set_spec_step_mode`), which carries it to the
//! worker ranks in `EP_CMD_SPEC_MODE`; no rank derives it on its own.
//!
//! Owner: scheduler.
//! Invariants:
//! - [`decide_mode`] and [`Dwell::observe`] are pure; the lane gathers their inputs.
//! - Leaving `Dflash` is immediate (any step with `active != 1`); returning needs one sequence
//!   alone, continuously, for the dwell. A sequence that starts alone waits the dwell too.
//! - Without a hybrid config the lane never builds a [`HybridState`] and runs as before.

use std::time::{Duration, Instant};

use metrale_model_layers::speculative::hybrid::SpecStepMode;

/// 2026-10-10: What serve setup resolves for a hybrid serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridConfig {
    /// 2026-10-10: The MTP head's draft depth (`--num-drafts` / MODEL.toml).
    pub mtp_num_drafts: usize,
    /// 2026-10-10: The DFlash drafter's draft count, gamma - 1.
    pub dflash_num_drafts: usize,
    /// 2026-10-10: Policy D2s dwell.
    pub dwell: Duration,
}

/// 2026-10-10: Identity of the sole active sequence, as far as the dwell needs it. A sequence on
/// the same slot with the same prompt length whose output did not shrink is the same sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoleSeq {
    pub slot: usize,
    pub prompt_len: usize,
    pub generated: usize,
}

impl SoleSeq {
    fn same_as(&self, earlier: &SoleSeq) -> bool {
        self.slot == earlier.slot
            && self.prompt_len == earlier.prompt_len
            && self.generated >= earlier.generated
    }
}

/// 2026-10-10: The inputs of one step's mode decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeInputs {
    /// 2026-10-10: Active (decoding) sequences this step.
    pub active: usize,
    /// 2026-10-10: `METRALE_MTP_MAX_SEQS`.
    pub mtp_max_seqs: usize,
    /// 2026-10-10: With `active == 1`: the sequence may take a DFlash step.
    pub sole_dflash_ok: bool,
    /// 2026-10-10: The dwell has been met ([`Dwell::observe`]).
    pub dwell_met: bool,
}

/// 2026-10-10: The step mode (module docs).
pub fn decide_mode(i: &ModeInputs) -> SpecStepMode {
    if i.active == 1 && i.sole_dflash_ok && i.dwell_met {
        SpecStepMode::Dflash
    } else if i.active <= i.mtp_max_seqs {
        SpecStepMode::Mtp
    } else {
        SpecStepMode::Plain
    }
}

/// 2026-10-10: Policy D2s: how long the current sole sequence has been alone.
#[derive(Debug, Clone)]
pub struct Dwell {
    dwell: Duration,
    alone: Option<(SoleSeq, Instant)>,
}

impl Dwell {
    pub fn new(dwell: Duration) -> Self {
        Self { dwell, alone: None }
    }

    /// 2026-10-10: Note this step's sole sequence (`None` unless exactly one is active) and
    /// return whether it has been alone for the dwell. Any step that is not alone, or a different
    /// sole sequence, restarts the clock.
    pub fn observe(&mut self, sole: Option<SoleSeq>, now: Instant) -> bool {
        let Some(sole) = sole else {
            self.alone = None;
            return false;
        };
        let since = match self.alone {
            Some((prev, since)) if sole.same_as(&prev) => since,
            _ => now,
        };
        self.alone = Some((sole, since));
        now.saturating_duration_since(since) >= self.dwell
    }
}

/// 2026-10-10: Steps per executed mode and switches between decided modes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModeCounters {
    /// 2026-10-10: Indexed by `SpecStepMode as usize`.
    pub steps: [u64; 3],
    /// 2026-10-10: `switches[from][to]`.
    pub switches: [[u64; 3]; 3],
}

impl ModeCounters {
    pub fn count_step(&mut self, executed: SpecStepMode) {
        self.steps[executed as usize] += 1;
    }

    pub fn count_switch(&mut self, from: SpecStepMode, to: SpecStepMode) {
        self.switches[from as usize][to as usize] += 1;
    }

    /// 2026-10-10: One log line: `steps PLAIN=a MTP=b DFLASH=c; switches MTP->DFLASH=d ...` (only
    /// the switches that happened).
    pub fn summary(&self) -> String {
        const MODES: [SpecStepMode; 3] =
            [SpecStepMode::Plain, SpecStepMode::Mtp, SpecStepMode::Dflash];
        let mut s = format!(
            "steps PLAIN={} MTP={} DFLASH={}; switches",
            self.steps[0], self.steps[1], self.steps[2]
        );
        let mut any = false;
        for from in MODES {
            for to in MODES {
                let n = self.switches[from as usize][to as usize];
                if n > 0 {
                    s.push_str(&format!(" {}->{}={n}", from.name(), to.name()));
                    any = true;
                }
            }
        }
        if !any {
            s.push_str(" none");
        }
        s
    }
}

/// 2026-10-10: Steps between periodic counter lines.
const SUMMARY_EVERY_STEPS: u64 = 4096;

/// 2026-10-10: The decode lane's hybrid state.
#[derive(Debug, Clone)]
pub struct HybridState {
    pub cfg: HybridConfig,
    dwell: Dwell,
    last: Option<SpecStepMode>,
    pub counters: ModeCounters,
}

impl HybridState {
    pub fn new(cfg: HybridConfig) -> Self {
        Self {
            dwell: Dwell::new(cfg.dwell),
            cfg,
            last: None,
            counters: ModeCounters::default(),
        }
    }

    /// 2026-10-10: This step's mode, from the active count, the sole sequence (`sole`, with
    /// `sole_dflash_ok` its DFlash eligibility) and the clock. Does not record it: the lane calls
    /// [`Self::commit`] once the model has taken the mode.
    pub fn decide(
        &mut self,
        active: usize,
        mtp_max_seqs: usize,
        sole: Option<SoleSeq>,
        sole_dflash_ok: bool,
        now: Instant,
    ) -> SpecStepMode {
        let sole = if active == 1 { sole } else { None };
        let dwell_met = self.dwell.observe(sole, now);
        decide_mode(&ModeInputs {
            active,
            mtp_max_seqs,
            sole_dflash_ok,
            dwell_met,
        })
    }

    /// 2026-10-10: The mode the previous step ran under, if any.
    pub fn last(&self) -> Option<SpecStepMode> {
        self.last
    }

    /// 2026-10-10: Record that the model now runs `mode`; returns whether it changed. A switch
    /// into or out of `Dflash` is logged with the counters.
    pub fn commit(&mut self, mode: SpecStepMode) -> bool {
        let prev = self.last.replace(mode);
        match prev {
            Some(p) if p != mode => {
                self.counters.count_switch(p, mode);
                if p == SpecStepMode::Dflash || mode == SpecStepMode::Dflash {
                    tracing::info!(
                        "HYBRID DFLASH: switch {} -> {} ({})",
                        p.name(),
                        mode.name(),
                        self.counters.summary()
                    );
                }
                true
            }
            _ => false,
        }
    }

    /// 2026-10-10: Count one executed step; every `SUMMARY_EVERY_STEPS` steps log the counters.
    pub fn count_step(&mut self, executed: SpecStepMode) {
        self.counters.count_step(executed);
        let total: u64 = self.counters.steps.iter().sum();
        if total.is_multiple_of(SUMMARY_EVERY_STEPS) {
            tracing::info!("HYBRID DFLASH counters: {}", self.counters.summary());
        }
    }

    /// 2026-10-10: The draft count of a step in `mode`.
    pub fn num_drafts(&self, mode: SpecStepMode) -> usize {
        if mode == SpecStepMode::Dflash {
            self.cfg.dflash_num_drafts
        } else {
            self.cfg.mtp_num_drafts
        }
    }
}

/// 2026-10-10: The decode lane's `(dflash_verify_raw_argmax, num_drafts)` for this step: the
/// step mode's on a hybrid serve, else the serve-wide pair unchanged.
pub fn step_flags(
    hybrid: Option<&HybridState>,
    mode: Option<SpecStepMode>,
    serve_raw_argmax: bool,
    serve_num_drafts: usize,
) -> (bool, usize) {
    match (hybrid, mode) {
        (Some(h), Some(m)) => (m == SpecStepMode::Dflash, h.num_drafts(m)),
        _ => (serve_raw_argmax, serve_num_drafts),
    }
}

/// 2026-10-10: Hybrid serve: commit a K2/K3/K4 single-sequence verify's accepted rows (rows
/// `0..=num_accepted`, positions from the pre-verify length) to the DFlash drafter context, so
/// it has no gap when the sequence next takes a DFlash step. Call after the verify forward and
/// before any rewind of `seq_len`. A no-op on any other serve, where these arms never committed.
pub(crate) fn commit_verify_ctx(
    model: &dyn metrale_model_engine::traits::Model,
    seq: &mut metrale_model_engine::traits::SequenceState,
    num_accepted: usize,
    k_rows: usize,
) {
    if !model.hybrid_dflash() {
        return;
    }
    let pre_verify_len = seq.seq_len.saturating_sub(k_rows);
    if let Err(e) = model.commit_ctx(seq, num_accepted + 1, pre_verify_len, 0) {
        tracing::error!("commit_ctx (hybrid K{k_rows}): {e:#}");
    }
}

#[cfg(test)]
#[path = "hybrid_mode_tests.rs"]
mod tests;
