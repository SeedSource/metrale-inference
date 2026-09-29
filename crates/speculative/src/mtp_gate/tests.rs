// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the MTP runtime gate. They drive it through the calls
//! the scheduler makes (`next_step`, `record_*`, `maybe_remeasure`) with
//! synthetic walls; tokens per MTP step stand in for acceptance.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use super::*;

fn ms(x: u64) -> Duration {
    Duration::from_millis(x)
}

/// 2026-09-25: Drives `n` MTP steps at `emitted` tokens and `wall` each.
fn drive_mtp(g: &mut MtpGate, n: usize, emitted: usize, wall: Duration) {
    for _ in 0..n {
        assert_eq!(
            g.next_step(),
            GateStep::MeasureVerify,
            "expected Mtp-path step"
        );
        g.record_verify_step(wall, emitted, 1);
    }
}

/// 2026-09-25: Drives `n` plain decode steps at `wall` each.
fn drive_serial(g: &mut MtpGate, n: usize, wall: Duration) {
    for _ in 0..n {
        assert_eq!(
            g.next_step(),
            GateStep::MeasureDecode,
            "expected serial step"
        );
        g.record_decode(wall, 1);
    }
}

/// 2026-09-25: Runs MTP steps until the gate opens a serial-refresh probe.
fn run_mtp_until_probe(g: &mut MtpGate, emitted: usize, wall: Duration) {
    for _ in 0..10_000 {
        if g.next_step() == GateStep::MeasureDecode {
            return;
        }
        g.record_verify_step(wall, emitted, 1);
    }
    panic!("gate never opened a serial probe");
}

/// 2026-09-25: Runs plain decode steps until the gate opens an MTP probe.
fn run_serial_until_probe(g: &mut MtpGate, wall: Duration) {
    for _ in 0..10_000 {
        if g.next_step() == GateStep::MeasureVerify {
            return;
        }
        g.record_decode(wall, 1);
    }
    panic!("gate never opened an MTP re-probe");
}

/// 2026-09-26: Switches a fresh gate to Serial: MTP at 20 tok/s against serial
/// probes at 100 tok/s, one probe window per loss of the dwell (each loss
/// needs a new serial measurement, and the confirming probe opens at once).
fn switch_to_serial(g: &mut MtpGate) {
    for _ in 0..SWITCH_DWELL_WINDOWS {
        run_mtp_until_probe(g, 2, ms(100));
        drive_serial(g, WINDOW_STEPS, ms(10));
    }
}

#[test]
fn starts_in_mtp_mode() {
    let g = MtpGate::new(1);
    assert_eq!(g.next_step(), GateStep::MeasureVerify);
    assert!(!g.in_serial_mode());
}

#[test]
fn no_switch_without_both_baselines() {
    let mut g = MtpGate::new(1);
    // 2026-09-25: Serial was never measured, so there is no switch.
    drive_mtp(&mut g, 64, 1, ms(100));
    assert!(!g.in_serial_mode());
    assert_eq!(g.take_fresh_decision(), None);
}

#[test]
fn refresh_probe_opens_after_interval_and_returns() {
    let mut g = MtpGate::new(1);
    // 2026-09-25: At 2 tokens per step, about 512 steps reach the default
    // 1024-token refresh interval.
    run_mtp_until_probe(&mut g, 2, ms(50));
    // 2026-09-25: The probe is one window of plain decode steps.
    drive_serial(&mut g, WINDOW_STEPS, ms(40));
    // 2026-09-25: Serial 25 tok/s against MTP 40 tok/s: stays Mtp.
    assert_eq!(g.next_step(), GateStep::MeasureVerify);
    assert!(!g.in_serial_mode());
    assert!(
        g.serial_tps_debug().is_some(),
        "probe must set the serial baseline"
    );
}

#[test]
fn switches_to_serial_when_clearly_faster_with_dwell() {
    let mut g = MtpGate::new(1);
    // 2026-09-25: MTP at 20 tok/s (2 tokens per 100 ms).
    run_mtp_until_probe(&mut g, 2, ms(100));
    // 2026-09-25: Serial probe at 100 tok/s, far past the margin.
    drive_serial(&mut g, WINDOW_STEPS, ms(10));
    // 2026-09-25: The probe's arbitration is the first loss; a second is
    // needed before the switch.
    assert!(
        !g.in_serial_mode(),
        "dwell must prevent single-window switches"
    );
    // 2026-09-26: More MTP windows do not count as a second loss: the serial
    // estimate has not been measured again. The confirming probe opens after
    // the next MTP step.
    g.record_verify_step(ms(100), 2, 1);
    assert!(!g.in_serial_mode(), "one serial window is one sample");
    assert_eq!(
        g.next_step(),
        GateStep::MeasureDecode,
        "confirming probe due at once"
    );
    drive_serial(&mut g, WINDOW_STEPS, ms(10));
    assert!(
        g.in_serial_mode(),
        "sustained 5x serial advantage must switch"
    );
    assert_eq!(g.take_fresh_decision(), Some(GateDecision::DisableMtp));
    assert_eq!(g.take_fresh_decision(), None, "fresh decision is one-shot");
    assert_eq!(g.next_step(), GateStep::MeasureDecode);
}

/// 2026-09-26: One fast serial window that the confirming probe does not
/// repeat leaves the gate in Mtp and clears the loss.
#[test]
fn unconfirmed_serial_window_does_not_switch() {
    let mut g = MtpGate::new(1);
    // 2026-09-26: MTP at 40 tok/s; the first serial probe reads 100 tok/s.
    run_mtp_until_probe(&mut g, 2, ms(50));
    drive_serial(&mut g, WINDOW_STEPS, ms(10));
    assert!(!g.in_serial_mode());
    // 2026-09-26: The confirming probe reads 25 tok/s.
    run_mtp_until_probe(&mut g, 2, ms(50));
    drive_serial(&mut g, WINDOW_STEPS, ms(40));
    assert!(
        !g.in_serial_mode(),
        "an unconfirmed serial window must not switch"
    );
    assert_eq!(g.losing_windows, 0);
    assert_eq!(g.take_fresh_decision(), None);
}

#[test]
fn hysteresis_blocks_within_margin_switches() {
    let mut g = MtpGate::new(1);
    // 2026-09-25: MTP at 40 tok/s.
    run_mtp_until_probe(&mut g, 2, ms(50));
    // 2026-09-25: Serial probe at 41 tok/s, inside the 5% margin floor (a
    // switch needs more than 42).
    drive_serial(&mut g, WINDOW_STEPS, Duration::from_micros(24_390));
    for _ in 0..(WINDOW_STEPS * 4) {
        if g.next_step() != GateStep::MeasureVerify {
            break;
        }
        g.record_verify_step(ms(50), 2, 1);
    }
    assert!(
        !g.in_serial_mode(),
        "a within-margin advantage must not switch modes"
    );
    assert_eq!(g.take_fresh_decision(), None);
}

#[test]
fn serial_mode_reprobes_mtp_and_recovers() {
    let mut g = MtpGate::new(1);
    // 2026-09-25: MTP at 20 tok/s, serial at 100 tok/s: switch to Serial.
    switch_to_serial(&mut g);
    assert!(g.in_serial_mode());
    g.take_fresh_decision();

    // 2026-09-25: MTP now at 300 tok/s (3 tokens per 10 ms): two probe
    // windows (the dwell) bring Mtp back.
    for _ in 0..SWITCH_DWELL_WINDOWS {
        run_serial_until_probe(&mut g, ms(10));
        drive_mtp(&mut g, WINDOW_STEPS, 3, ms(10));
    }
    assert!(
        !g.in_serial_mode(),
        "re-probe must recover MTP when it wins again"
    );
    assert_eq!(g.take_fresh_decision(), Some(GateDecision::KeepMtp));
    assert_eq!(g.next_step(), GateStep::MeasureVerify);
}

#[test]
fn depth_change_schedules_early_probe_without_state_wipe() {
    let mut g = MtpGate::new(1);
    g.note_depth(600);
    drive_mtp(&mut g, WINDOW_STEPS * 2, 2, ms(50));
    let tps_before = g.mtp_tps_debug();
    assert!(tps_before.is_some());
    // 2026-09-25: 1300 is more than twice the floored measured depth (512):
    // both estimates go stale, a probe is due, and the estimates are kept.
    assert!(g.maybe_remeasure(1300));
    assert_eq!(g.regime_reprobe_count(), 1);
    assert_eq!(
        g.mtp_tps_debug(),
        tps_before,
        "no state wipe on regime change"
    );
    // 2026-09-26: The due probe opens once the stale Mtp estimate has been
    // replaced by a full window in the new regime.
    drive_mtp(&mut g, WINDOW_STEPS, 2, ms(50));
    assert_eq!(
        g.next_step(),
        GateStep::MeasureDecode,
        "stale regime must probe soon"
    );
    // 2026-09-25: The early probe is one window. Serial at 20 tok/s against
    // MTP at 40 keeps Mtp, and serial stays a small share of a 1000-token
    // turn.
    drive_serial(&mut g, WINDOW_STEPS, ms(50));
    assert_eq!(g.next_step(), GateStep::MeasureVerify);
    assert!(!g.in_serial_mode());
    let mut serial_steps = WINDOW_STEPS;
    let mut mtp_steps = WINDOW_STEPS * 2 + 1;
    let mut tokens = (WINDOW_STEPS * 2 + 1) * 2 + WINDOW_STEPS;
    while tokens < 1000 {
        match g.next_step() {
            GateStep::MeasureVerify => {
                g.record_verify_step(ms(50), 2, 1);
                mtp_steps += 1;
                tokens += 2;
            }
            GateStep::MeasureDecode => {
                g.record_decode(ms(50), 1);
                serial_steps += 1;
                tokens += 1;
            }
        }
    }
    let serial_frac = serial_steps as f64 / (serial_steps + mtp_steps) as f64;
    assert!(
        serial_steps <= WINDOW_STEPS,
        "regime change must not add serial beyond the one-window probe ({serial_steps})"
    );
    assert!(
        serial_frac < 0.05,
        "serial must not dominate a 1000-token turn (frac={serial_frac:.3})"
    );
    assert!(!g.in_serial_mode());
}

#[test]
fn bootstrap_steps_count_at_least_one_token() {
    let mut g = MtpGate::new(1);
    // 2026-09-25: `emitted = 0` is charged as one token.
    for _ in 0..WINDOW_STEPS {
        g.record_verify_step(ms(10), 0, 1);
    }
    let tps = g.mtp_tps_debug().expect("window closed");
    assert!(tps > 0.0);
}

/// 2026-09-25: While any active sequence is inside the entry window, the pin
/// overrides a Serial verdict with the verify step.
#[test]
fn entry_pin_overrides_serial_mode_for_answer_openings() {
    let mut g = MtpGate::new(1);
    // 2026-09-25: Switch the gate to Serial (MTP 20 tok/s, serial 100 tok/s).
    switch_to_serial(&mut g);
    assert!(g.in_serial_mode());
    assert_eq!(g.next_step(), GateStep::MeasureDecode);

    // 2026-09-25: A sequence 0 to 7 tokens past `</think>` pins the step to
    // verify despite the Serial verdict.
    assert!(entry_pin_forces_verify(0, 8));
    assert!(entry_pin_forces_verify(7, 8));
    // 2026-09-25: From 8 tokens on, the gate's verdict stands.
    assert!(!entry_pin_forces_verify(8, 8));
    assert!(!entry_pin_forces_verify(u32::MAX, 8));
}

/// 2026-09-25: A pinned step makes no gate call, so the pin leaves the
/// estimates and the mode as they were.
#[test]
fn entry_pin_steps_do_not_touch_arbitration_state() {
    let mut g = MtpGate::new(1);
    switch_to_serial(&mut g);
    assert!(g.in_serial_mode());
    g.take_fresh_decision();
    let serial_before = g.serial_tps_debug();
    let mtp_before = g.mtp_tps_debug();
    // 2026-09-25: The scheduler runs a pinned step without any `record_*`
    // call (`scheduler/core/lane_decode.rs`); this test only states that
    // contract.
    assert_eq!(g.serial_tps_debug(), serial_before);
    assert_eq!(g.mtp_tps_debug(), mtp_before);
    assert!(g.in_serial_mode(), "pin must not flip the gate's mode");
}

/// 2026-09-25: `METRALE_SPEC_ENTRY_PIN` parses as a `u32`, default 8; `0`
/// disables.
#[test]
fn entry_pin_env_parse() {
    assert_eq!(parse_entry_pin_tokens(None), 8);
    assert_eq!(parse_entry_pin_tokens(Some("0")), 0);
    assert_eq!(parse_entry_pin_tokens(Some("12")), 12);
    assert_eq!(parse_entry_pin_tokens(Some("garbage")), 8);
    assert_eq!(parse_entry_pin_tokens(Some("-3")), 8);
}

#[test]
fn stale_other_baseline_cannot_steal_mode() {
    let mut g = MtpGate::new(1);
    run_mtp_until_probe(&mut g, 2, ms(50));
    drive_serial(&mut g, WINDOW_STEPS, ms(40));
    assert!(!g.in_serial_mode());
    assert!(g.serial_tps_debug().is_some());
    // 2026-09-25: After the regime change the serial estimate (25 tok/s) is
    // stale. MTP drops to 10 tok/s, and the gate still does not switch onto
    // the stale estimate.
    assert!(g.maybe_remeasure(2000));
    for _ in 0..(WINDOW_STEPS * SWITCH_DWELL_WINDOWS * 2) {
        if g.next_step() != GateStep::MeasureVerify {
            break;
        }
        g.record_verify_step(ms(200), 2, 1);
    }
    assert!(
        !g.in_serial_mode(),
        "a stale serial baseline must not win a switch after a depth-regime change"
    );
    assert_eq!(g.take_fresh_decision(), None);
}

#[test]
fn mtp_spec_think_lever_off_stays_serial_in_think() {
    assert!(!spec_dispatch_eligible(
        true, 0, 0, false, false, false, 0, false
    ));
    assert!(!spec_dispatch_eligible(
        true, 0, 50, false, false, false, 0, false
    ));
    assert!(spec_dispatch_eligible(
        false, 0, 50, false, false, false, 0, false
    ));
}

#[test]
fn mtp_spec_think_lever_on_speculates_in_think() {
    assert!(spec_dispatch_eligible(
        true, 0, 50, false, false, true, 0, false
    ));
}

#[test]
fn dflash_raw_argmax_stays_serial_in_think() {
    assert!(!spec_dispatch_eligible(
        true, 0, 50, false, false, false, 0, true
    ));
    assert!(spec_dispatch_eligible(
        false, 0, 50, false, false, false, 0, true
    ));
}

#[test]
fn dflash_spec_think_opts_in() {
    assert!(spec_dispatch_eligible(
        true, 0, 0, false, false, true, 0, true
    ));
}

/// 2026-09-29: A146: the MTP-lane default is on for GLM-5.3 only.
#[test]
fn mtp_spec_think_default_is_glm53_only() {
    assert!(mtp_spec_think_default("glm5_next"));
    assert!(mtp_spec_think_default("glm5_next_text"));
    for other in [
        "qwen3_next",
        "qwen3_5_moe",
        "deepseek_v4",
        "step3p7",
        "glm5",
        "",
    ] {
        assert!(!mtp_spec_think_default(other), "{other} must stay opt-in");
    }
}

/// 2026-09-29: A146: each lane reads only its own lever.
#[test]
fn spec_think_for_lane_reads_only_the_active_lanes_lever() {
    assert!(spec_think_for_lane(false, true, false));
    assert!(!spec_think_for_lane(false, false, true));
    assert!(spec_think_for_lane(true, false, true));
    assert!(!spec_think_for_lane(true, true, false));
}

/// 2026-09-25: A 2048-token think from depth 64 crosses two depth regimes
/// (at 1024 and 2048). Each opens a one-window probe; serial stays a small
/// share of the steps and never becomes the mode.
#[test]
fn think_budget_2048_two_crossings_do_not_dump_serial() {
    let mut g = MtpGate::new(1);
    let mut depth = 64usize;
    g.note_depth(depth);
    let mut serial_steps = 0usize;
    let mut mtp_steps = 0usize;
    let mut tokens = 0usize;
    while tokens < 2048 {
        g.note_depth(depth);
        let _ = g.maybe_remeasure(depth);
        match g.next_step() {
            GateStep::MeasureVerify => {
                g.record_verify_step(ms(50), 2, 1);
                mtp_steps += 1;
                tokens += 2;
                depth += 2;
            }
            GateStep::MeasureDecode => {
                g.record_decode(ms(50), 1);
                serial_steps += 1;
                tokens += 1;
                depth += 1;
            }
        }
    }
    let serial_frac = serial_steps as f64 / (serial_steps + mtp_steps) as f64;
    assert_eq!(g.regime_reprobe_count(), 2);
    assert!(
        serial_steps <= WINDOW_STEPS * 3,
        "at most one-window probe per crossing plus shipped refresh, got {serial_steps}"
    );
    assert!(
        serial_frac < 0.08,
        "serial must not dominate a 2048-token think (frac={serial_frac:.3})"
    );
    assert!(!g.in_serial_mode());
}

mod width;
