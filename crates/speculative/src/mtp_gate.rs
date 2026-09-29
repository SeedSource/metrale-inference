// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The MTP runtime gate: it chooses, step by step, between the MTP
//! verify step and plain decode by the delivered throughput (emitted tokens
//! per second of step wall time) each mode measured over windows of
//! `WINDOW_STEPS` steps.
//!
//! Owner: speculative.
//! Invariants:
//! - The mode changes only in `arbitrate`, and only when both modes have an
//!   estimate, the other mode's is not stale, and it beat the current mode by
//!   the margin in `SWITCH_DWELL_WINDOWS` consecutive arbitrations.
//! - A window never mixes power-of-two batch-width buckets: `note_width`
//!   discards the partial window before a step from a new bucket is added.
//!
//! Policy:
//! - Run the current mode, and fold each closed window into that mode's
//!   tok/s EWMA and deviation EWMA (`ModeStats::update`).
//! - In Serial mode, probe MTP for one window after `reprobe_tokens`
//!   recorded tokens; in Mtp mode, refresh the serial estimate after
//!   `serial_refresh_tokens`.
//! - A depth-regime change (factor `REMEASURE_DEPTH_FACTOR`, floor
//!   `REMEASURE_DEPTH_FLOOR`, `maybe_remeasure`) or a change of
//!   power-of-two batch-width bucket (`note_width`) marks both estimates
//!   stale and brings the next probe forward. A stale other-mode estimate
//!   cannot win a switch.
//! - A plain decode step over n sequences is charged n tokens, and a verify
//!   step the tokens all its sequences emitted, so both modes are measured as
//!   whole-batch throughput.
//!
//! With `--mtp-gate force` (or `METRALE_MTP_GATE_FORCE`) the scheduler builds
//! no gate and runs the MTP step whenever speculation is eligible.

use std::time::Duration;

mod driver;

/// 2026-09-25: A context depth that has grown or shrunk by this factor since
/// the last measurement marks both estimates stale.
const REMEASURE_DEPTH_FACTOR: usize = 2;
/// 2026-09-25: Both depths are raised to this floor before the comparison,
/// so contexts below it never change regime among themselves.
const REMEASURE_DEPTH_FLOOR: usize = 512;
/// 2026-09-25: Steps per throughput window.
const WINDOW_STEPS: usize = 16;
/// 2026-09-25: Consecutive arbitrations the other mode must win by the
/// margin before the gate switches to it.
const SWITCH_DWELL_WINDOWS: usize = 2;
/// 2026-09-25: EWMA weight of a new window in the per-mode tok/s estimate
/// (effective window about 3 windows).
const TPS_ALPHA: f64 = 0.3;
/// 2026-09-25: The switch margin is the larger of this fraction of the
/// current mode's estimate and half the sum of the two deviation EWMAs.
const MARGIN_REL_FLOOR: f64 = 0.05;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// 2026-09-25: Tokens between MTP probes in Serial mode:
/// `METRALE_MTP_GATE_REPROBE`, default 256 (the same default as
/// `METRALE_DFLASH_ADAPTIVE_REPROBE`).
fn reprobe_tokens() -> usize {
    env_usize("METRALE_MTP_GATE_REPROBE", 256)
}

/// 2026-09-25: Tokens between serial-estimate refreshes in Mtp mode:
/// `METRALE_MTP_GATE_REFRESH`, default 1024. 2026-09-26: per sequence of the
/// batch-width bucket, so the refresh comes after the same number of steps at
/// every width (`MtpGate::event_interval`).
fn serial_refresh_tokens() -> usize {
    env_usize("METRALE_MTP_GATE_REFRESH", 1024)
}

/// 2026-09-25: The spec-entry verify pin, in tokens after `</think>`, from
/// the value of `METRALE_SPEC_ENTRY_PIN`: 8 when unset or unparseable, and
/// `0` disables it. While any active sequence is within it, the scheduler
/// runs the MTP verify step even when the gate chose plain decode.
///
/// The serial (one-row) and verify (batch-K) forwards can pick different
/// low-margin tokens at T=0, and the gate picks between them by wall-clock
/// throughput, so without the pin the first tokens of an answer would depend
/// on the gate's measurements. Measured 2026-07-07/08: every observed flip
/// between the two forwards fell within 7 tokens of speculation entry; 8
/// adds one token of margin.
///
/// `METRALE_DFLASH_RESUME_GUARD` is checked earlier, in
/// `spec_dispatch_eligible`: outside `<think>`, a sequence with fewer tokens
/// after `</think>` than the guard does not reach the gate at all.
pub(crate) fn parse_entry_pin_tokens(env: Option<&str>) -> u32 {
    env.and_then(|v| v.parse().ok()).unwrap_or(8)
}

/// 2026-09-25: The pin width from `METRALE_SPEC_ENTRY_PIN`. The scheduler
/// stores it in `SchedLevers::spec_entry_pin_tokens`.
pub fn entry_pin_tokens_from_env() -> u32 {
    parse_entry_pin_tokens(std::env::var("METRALE_SPEC_ENTRY_PIN").ok().as_deref())
}

/// 2026-09-25: Whether the spec-entry pin overrides a Serial gate decision
/// for this step. `min_post_think_emitted` is the minimum over the active
/// batch, so one entering sequence pins the whole (already spec-eligible)
/// batch. `pin_tokens` is the run's [`entry_pin_tokens_from_env`] value.
pub fn entry_pin_forces_verify(min_post_think_emitted: u32, pin_tokens: u32) -> bool {
    min_post_think_emitted < pin_tokens
}

/// 2026-09-29: A146: the per-model default for MTP-lane speculation inside
/// `<think>`, keyed on `ModelConfig::model_type` like the other
/// glm5_next-specific serve behaviour (`seq_state_reserve.rs`). On only for
/// architectures that passed the spec-in-think quality and safety gates:
/// GLM-5.3 (`glm5_next` and its text-only `glm5_next_text`); measured on
/// Atlas 2026-09-26: K=3 byte-identical 6/6 vs spec-off, TEB 156/176
/// identical to spec-off per scenario. Every other model keeps the pre-split
/// behaviour (serial inside thinking unless opted in) until it passes the
/// same gates.
pub fn mtp_spec_think_default(model_type: &str) -> bool {
    matches!(model_type, "glm5_next" | "glm5_next_text")
}

/// 2026-09-29: A146: the per-lane spec-in-think lever
/// [`spec_dispatch_eligible`] must receive. `dflash_lane` is
/// `dflash_verify_raw_argmax` (`args.dflash`, serve_load.rs), true for EVERY
/// DFlash verify mode. The MTP lane uses `SchedLevers::mtp_spec_think(model
/// default)`; the DFlash lane stays opt-in (`SchedLevers::dflash_spec_think`)
/// on every model.
pub fn spec_think_for_lane(
    dflash_lane: bool,
    mtp_spec_think: bool,
    dflash_spec_think: bool,
) -> bool {
    if dflash_lane {
        dflash_spec_think
    } else {
        mtp_spec_think
    }
}

/// 2026-09-25: Whether one sequence may take the speculative path this step.
/// Never with `suppress_tool_call` or `disable_mtp`, and never inside
/// `<think>` unless `spec_think`. 2026-09-29: A146: `spec_think` is the
/// ACTIVE lane's lever, resolved by [`spec_think_for_lane`]: MTP speculates
/// inside `<think>` by default on models whose [`mtp_spec_think_default`] is
/// on (`METRALE_MTP_SPEC_THINK=0` or `METRALE_DFLASH_SPEC_THINK=0`
/// disables) and on others only when opted in (`=1`); DFlash stays serial
/// inside `<think>` unless `METRALE_DFLASH_SPEC_THINK=1`. Otherwise it needs `resume_guard` emitted tokens: counted
/// after `</think>`, except inside `<think>` (reachable only with
/// `spec_think`), where the whole output counts.
pub fn spec_dispatch_eligible(
    inside_thinking: bool,
    post_think_emitted: u32,
    output_len: u32,
    suppress_tool_call: bool,
    disable_mtp: bool,
    spec_think: bool,
    resume_guard: u32,
    dflash_raw_argmax: bool,
) -> bool {
    if suppress_tool_call || disable_mtp {
        return false;
    }
    // 2026-09-25: Batch-K verify can pick a different low-margin token than
    // serial decode at T=0 (see `parse_entry_pin_tokens`).
    //
    // 2026-09-29: A146: this was once opt-in for BOTH lanes because batch-K
    // verify committed low-margin tokens spec-off decode would not. The
    // spec-in-think parity chain (verify window and `emit_token` commit
    // thinking state exactly like spec-off decode, plus A143/A144/A144b)
    // closed that for MTP on GLM-5.3, so MTP is default-on there only
    // (`mtp_spec_think_default`); other models and DFlash stay opt-in until
    // they pass their own GPU and TEB qualification.
    if inside_thinking && !spec_think {
        return false;
    }
    if dflash_raw_argmax && !spec_think {
        return post_think_emitted >= resume_guard;
    }
    if inside_thinking {
        output_len >= resume_guard
    } else {
        post_think_emitted >= resume_guard
    }
}

/// 2026-09-25: The step the gate wants the scheduler to run next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateStep {
    /// 2026-09-25: Plain decode (Serial mode, or a serial refresh from Mtp).
    MeasureDecode,
    /// 2026-09-25: MTP step (Mtp mode, or a probe from Serial).
    MeasureVerify,
}

/// 2026-09-25: A mode switch, handed to the scheduler once
/// (`take_fresh_decision`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    /// 2026-09-25: Switched to Mtp. The scheduler does nothing: the next MTP
    /// step bootstraps from empty drafts.
    KeepMtp,
    /// 2026-09-25: Switched to Serial. The scheduler clears pending drafts
    /// and synchronises the secondary stream.
    DisableMtp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Mtp,
    Serial,
}

#[derive(Default)]
struct ModeStats {
    /// 2026-09-25: Delivered-throughput EWMA (tokens/sec), `None` until the
    /// first window.
    tps: Option<f64>,
    /// 2026-09-25: EWMA of |window tps − updated tps|, for the switch margin.
    dev: f64,
    /// 2026-09-25: Set by a depth or width regime change; cleared by the next
    /// window folded in.
    stale: bool,
    /// 2026-09-26: Windows folded in so far, so an arbitration can tell a new
    /// measurement from a repeat of the last one.
    windows: u64,
}

impl ModeStats {
    /// 2026-09-25: Folds one closed window into the estimate. With `replace`
    /// (a probe window, or the first window after the estimate went stale)
    /// the window replaces the estimate and `dev` is halved: blending a
    /// possibly drifted baseline would add the drift to `dev`. Consecutive
    /// windows of the running mode blend as an EWMA.
    fn update(&mut self, window_tps: f64, replace: bool) {
        match (self.tps, replace) {
            (None, _) | (_, true) => {
                self.tps = Some(window_tps);
                self.dev *= 0.5;
            }
            (Some(prev), false) => {
                let next = (1.0 - TPS_ALPHA) * prev + TPS_ALPHA * window_tps;
                self.dev = (1.0 - TPS_ALPHA) * self.dev + TPS_ALPHA * (window_tps - next).abs();
                self.tps = Some(next);
            }
        }
        self.stale = false;
        self.windows += 1;
    }
}

/// 2026-09-25: The gate. The scheduler builds one when MTP is on and the gate
/// is not forced off, and records each gated plain-decode or MTP step in it;
/// steps run under a pin are not recorded.
pub struct MtpGate {
    /// 2026-09-25: `reprobe_tokens`, read once.
    reprobe: usize,
    /// 2026-09-25: `serial_refresh_tokens`, read once.
    refresh: usize,
    mode: Mode,
    /// 2026-09-25: True while the gate runs the other mode for a probe.
    probing: bool,
    /// 2026-09-25: Windows remaining in the current probe.
    probe_windows_left: usize,
    mtp: ModeStats,
    serial: ModeStats,
    // 2026-09-25: Accumulators of the open window, for whichever mode ran.
    win_tokens: f64,
    win_wall: f64,
    win_steps: usize,
    /// 2026-09-25: Consecutive arbitrations in which the other mode beat the
    /// current one by more than the margin. 2026-09-26: Only arbitrations on a
    /// new measurement of the other mode count.
    losing_windows: usize,
    /// 2026-09-26: The other mode's `windows` at the last counted loss.
    loss_other_windows: u64,
    /// 2026-09-25: Tokens recorded outside probes since the last probe or
    /// switch; a regime change raises it to the event interval.
    tokens_since_event: usize,
    observed_depth: usize,
    measured_at_depth: usize,
    /// 2026-09-25: Power-of-two batch-width bucket of the last recorded step
    /// (0 before any). See [`Self::note_width`].
    width_regime: usize,
    fresh: Option<GateDecision>,
    /// 2026-09-26: Steps still to be left out of every window after a
    /// batch-width bucket change ([`Self::note_width`]).
    settle_steps: usize,
    /// 2026-09-25: Depth-regime changes seen by this gate.
    regime_reprobes: usize,
}

impl MtpGate {
    /// 2026-09-25: Tracks the batch-width bucket (`next_power_of_two`).
    /// Throughput at different widths is not comparable, so a bucket change
    /// discards the open window, marks both estimates stale and brings the
    /// next probe forward, as [`Self::maybe_remeasure`] does for depth.
    /// Widths within one bucket are treated as the same regime.
    fn note_width(&mut self, width: usize) {
        let regime = width.next_power_of_two();
        if regime == self.width_regime {
            return;
        }
        if self.width_regime != 0 {
            tracing::info!(
                "MTP gate: batch-width regime changed ({} -> {regime}); partial window \
                 discarded, baselines stale, will re-probe on cadence",
                self.width_regime,
            );
            self.mtp.stale = true;
            self.serial.stale = true;
            self.win_tokens = 0.0;
            self.win_wall = 0.0;
            self.win_steps = 0;
            self.losing_windows = 0;
            // 2026-09-26: The interval of the new bucket, which the Mtp-mode
            // refresh scales with.
            self.width_regime = regime;
            self.tokens_since_event = self.tokens_since_event.max(self.event_interval());
            // 2026-09-26: A sequence that joined (or left) is bootstrapping its
            // drafts and the batch is still reshaping, so the next window's
            // steps would not measure the new regime's steady state.
            self.settle_steps = WINDOW_STEPS;
        }
        self.width_regime = regime;
    }

    fn other(m: Mode) -> Mode {
        match m {
            Mode::Mtp => Mode::Serial,
            Mode::Serial => Mode::Mtp,
        }
    }

    /// 2026-09-25: Tokens recorded outside probes before the next probe.
    /// 2026-09-26: In Mtp mode the refresh interval is per sequence: it scales
    /// with the batch-width bucket, because a step at width W records W times
    /// the tokens. Unscaled, the one-window serial refresh came every
    /// 1024 / (W * tokens per step) steps, a third of all steps at W = 16 (the
    /// `serial` share of the MTP Done lines). Serial mode's MTP re-probe keeps
    /// its token interval, so a gate that chose serial re-tests MTP as often.
    fn event_interval(&self) -> usize {
        match self.mode {
            Mode::Mtp => self.refresh * self.width_regime.max(1),
            Mode::Serial => self.reprobe,
        }
    }

    fn stats_mut(&mut self, m: Mode) -> &mut ModeStats {
        match m {
            Mode::Mtp => &mut self.mtp,
            Mode::Serial => &mut self.serial,
        }
    }

    fn record_step(&mut self, wall: Duration, tokens: usize) {
        // 2026-09-26: The settle steps after a width change count toward
        // nothing, not even the probe cadence.
        if self.settle_steps > 0 {
            self.settle_steps -= 1;
            return;
        }
        self.win_tokens += tokens as f64;
        self.win_wall += wall.as_secs_f64();
        self.win_steps += 1;
        if !self.probing {
            self.tokens_since_event += tokens;
        }
        if self.win_steps >= WINDOW_STEPS {
            self.close_window();
        } else if !self.probing
            && self.tokens_since_event >= self.event_interval()
            && !self.stats_mut(self.mode).stale
        {
            // 2026-09-25: A probe is due: close the window early so the
            // probe starts on empty accumulators. 2026-09-26: Not while the
            // running mode's estimate is stale: the window that replaces it
            // must be a full one, or one step decides the next arbitration.
            // A partial window shorter than half a window is dropped rather
            // than blended in.
            if self.win_steps < WINDOW_STEPS / 2 {
                self.win_tokens = 0.0;
                self.win_wall = 0.0;
                self.win_steps = 0;
            }
            self.close_window();
        }
    }

    fn close_window(&mut self) {
        let ran = if self.probing {
            Self::other(self.mode)
        } else {
            self.mode
        };
        if self.win_wall > 0.0 && self.win_steps > 0 {
            let window_tps = self.win_tokens / self.win_wall;
            let replace = self.probing || self.stats_mut(ran).stale;
            self.stats_mut(ran).update(window_tps, replace);
        }
        self.win_tokens = 0.0;
        self.win_wall = 0.0;
        self.win_steps = 0;

        if self.probing {
            self.probe_windows_left = self.probe_windows_left.saturating_sub(1);
            if self.probe_windows_left == 0 {
                self.probing = false;
                // 2026-09-26: Reset before arbitrating, which may make a
                // confirming probe due at once.
                self.tokens_since_event = 0;
                self.arbitrate();
            }
            return;
        }

        if self.tokens_since_event >= self.event_interval() {
            self.probing = true;
            self.probe_windows_left = 1;
            return;
        }
        self.arbitrate();
    }

    /// 2026-09-25: Compares the two estimates with the margin and switches
    /// after [`SWITCH_DWELL_WINDOWS`] consecutive losses.
    fn arbitrate(&mut self) {
        let (Some(mtp), Some(serial)) = (self.mtp.tps, self.serial.tps) else {
            return;
        };
        // 2026-09-25: A stale other-mode estimate predates a depth or width
        // regime change. Wait for the early probe to refresh it rather than
        // switch on it.
        let other_stale = match self.mode {
            Mode::Mtp => self.serial.stale,
            Mode::Serial => self.mtp.stale,
        };
        if other_stale {
            return;
        }
        let (cur, other, other_dev) = match self.mode {
            Mode::Mtp => (mtp, serial, self.serial.dev),
            Mode::Serial => (serial, mtp, self.mtp.dev),
        };
        let margin = (MARGIN_REL_FLOOR * cur).max(0.5 * (self.dev_of(self.mode) + other_dev));
        if other > cur + margin {
            // 2026-09-26: A loss counts toward the dwell only on a measurement
            // of the other mode newer than the last counted loss's: two
            // arbitrations against one probe window are one sample, and one
            // noisy serial window (140-180 tok/s at C=4 on GB10) switched the
            // gate. After a loss short of the dwell, the confirming probe is
            // due at once.
            let other_windows = match self.mode {
                Mode::Mtp => self.serial.windows,
                Mode::Serial => self.mtp.windows,
            };
            if self.losing_windows > 0 && other_windows == self.loss_other_windows {
                return;
            }
            self.losing_windows += 1;
            self.loss_other_windows = other_windows;
            if self.losing_windows < SWITCH_DWELL_WINDOWS {
                self.tokens_since_event = self.tokens_since_event.max(self.event_interval());
            }
            if self.losing_windows >= SWITCH_DWELL_WINDOWS {
                let to = Self::other(self.mode);
                tracing::info!(
                    "MTP gate: switching {:?} -> {:?} (current {cur:.1} tok/s vs other \
                     {other:.1} tok/s, margin {margin:.1}, depth={})",
                    self.mode,
                    to,
                    self.observed_depth,
                );
                self.mode = to;
                self.losing_windows = 0;
                self.tokens_since_event = 0;
                self.measured_at_depth = self.observed_depth;
                self.fresh = Some(match to {
                    Mode::Mtp => GateDecision::KeepMtp,
                    Mode::Serial => GateDecision::DisableMtp,
                });
            }
        } else {
            self.losing_windows = 0;
        }
    }

    fn dev_of(&self, m: Mode) -> f64 {
        match m {
            Mode::Mtp => self.mtp.dev,
            Mode::Serial => self.serial.dev,
        }
    }

    pub fn mtp_tps_debug(&self) -> Option<f64> {
        self.mtp.tps
    }
    pub fn serial_tps_debug(&self) -> Option<f64> {
        self.serial.tps
    }
    pub fn in_serial_mode(&self) -> bool {
        self.mode == Mode::Serial
    }
    pub fn regime_reprobe_count(&self) -> usize {
        self.regime_reprobes
    }
    pub fn is_probing(&self) -> bool {
        self.probing
    }
}

#[cfg(test)]
mod tests;
