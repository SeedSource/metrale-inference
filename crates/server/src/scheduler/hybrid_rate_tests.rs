// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Tests for `hybrid_rate.rs` (policy D2sR) and its levers. CPU only, fake `Instant`s.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::time::{Duration, Instant};

use metrale_model_layers::speculative::hybrid::{self as hy, SpecStepMode};

use super::hybrid_mode::{Dwell, HybridConfig, HybridState, ModeInputs, SoleSeq, decide_mode};
use super::hybrid_rate::RateGateCfg;

use SpecStepMode::{Dflash, Mtp, Plain};

const TICK_MS: u64 = 50;

fn gate(margin_pct: u32) -> Option<RateGateCfg> {
    Some(RateGateCfg {
        margin_pct,
        cooldown: Duration::from_millis(8000),
    })
}

fn cfg(dwell_ms: u64, rate_gate: Option<RateGateCfg>) -> HybridConfig {
    HybridConfig {
        mtp_num_drafts: 3,
        dflash_num_drafts: 7,
        dwell: Duration::from_millis(dwell_ms),
        rate_gate,
    }
}

/// 2026-10-11: A scripted decode: one 50 ms tick per `step`; the sole sequence gains tokens at
/// `r_m` tok/s on an MTP/plain step and `r_d` on a DFlash step.
struct Sim {
    h: HybridState,
    t0: Instant,
    ms: u64,
    toks: f64,
    prompt_len: usize,
    r_m: f64,
    r_d: f64,
    log: Vec<(u64, SpecStepMode)>,
}

impl Sim {
    fn new(c: HybridConfig, r_m: f64, r_d: f64) -> Self {
        Self {
            h: HybridState::new(c),
            t0: Instant::now(),
            ms: 0,
            toks: 1.0,
            prompt_len: 100,
            r_m,
            r_d,
            log: Vec::new(),
        }
    }

    fn step(&mut self, active: usize) -> SpecStepMode {
        let sole = Some(SoleSeq {
            slot: 0,
            prompt_len: self.prompt_len,
            generated: self.toks as usize,
        });
        let now = self.t0 + Duration::from_millis(self.ms);
        let mode = self.h.decide(active, 8, sole, true, now);
        self.h.commit(mode);
        self.h.count_step(mode);
        self.log.push((self.ms, mode));
        let r = if mode == Dflash { self.r_d } else { self.r_m };
        self.toks += r * TICK_MS as f64 / 1000.0;
        self.ms += TICK_MS;
        mode
    }

    /// 2026-10-11: Steps alone until `ms >= end`.
    fn run_to(&mut self, end: u64) {
        while self.ms < end {
            self.step(1);
        }
    }

    /// 2026-10-11: Steps alone until the mode is `want`; returns the time of that step.
    fn run_until_mode(&mut self, want: SpecStepMode, limit: u64) -> u64 {
        while self.ms < limit {
            let t = self.ms;
            if self.step(1) == want {
                return t;
            }
        }
        panic!(
            "mode {want:?} not reached by {limit} ms; log tail {:?}",
            &self.log[self.log.len().saturating_sub(5)..]
        );
    }

    fn modes_between(&self, from: u64, to: u64) -> Vec<SpecStepMode> {
        self.log
            .iter()
            .filter(|(t, _)| *t >= from && *t < to)
            .map(|(_, m)| *m)
            .collect()
    }
}

/// 2026-10-11: (b) Prose: r_D = 0.9 r_M. Demotes once bootstrap + W_D have passed on DFlash,
/// stays on MTP through the cool-down, probes again after cool-down + dwell, and the second
/// demotion doubles the cool-down.
#[test]
fn prose_trace_demotes_with_doubling_cooldown() {
    let mut s = Sim::new(cfg(2000, gate(0)), 30.0, 27.0);
    // Dwell met at 2000 ms: DFlash from there.
    let enter1 = s.run_until_mode(Dflash, 3000);
    assert_eq!(enter1, 2000);
    // First step on MTP after entering: the demotion. Not before bootstrap (1 s) + W_D (1 s).
    let demote1 = s.run_until_mode(Mtp, 6000);
    assert!(
        (4000..=4000 + 2 * TICK_MS).contains(&demote1),
        "demoted at {demote1}"
    );
    assert_eq!(s.h.counters.rate_demotions, 1);
    // Barred for C_0 = 8000 ms from the demotion step, then a fresh dwell (2000 ms).
    s.run_to(demote1 + 8000 + 2000 - TICK_MS);
    assert!(
        s.modes_between(demote1, demote1 + 8000 + 2000)
            .iter()
            .all(|m| *m == Mtp),
        "DFlash during cool-down + dwell"
    );
    let enter2 = s.run_until_mode(Dflash, demote1 + 12_000);
    assert!(
        (demote1 + 10_000..=demote1 + 10_000 + 2 * TICK_MS).contains(&enter2),
        "re-probe at {enter2}, demotion at {demote1}"
    );
    let demote2 = s.run_until_mode(Mtp, enter2 + 4000);
    assert!(
        (enter2 + 2000..=enter2 + 2000 + 2 * TICK_MS).contains(&demote2),
        "second demotion at {demote2}"
    );
    assert_eq!(s.h.counters.rate_demotions, 2);
    // k = 1: the cool-down is 16 000 ms.
    s.run_to(demote2 + 16_000 + 2000 - TICK_MS);
    assert!(
        s.modes_between(demote2, demote2 + 16_000 + 2000)
            .iter()
            .all(|m| *m == Mtp)
    );
    let enter3 = s.run_until_mode(Dflash, demote2 + 20_000);
    assert!(
        (demote2 + 18_000..=demote2 + 18_000 + 2 * TICK_MS).contains(&enter3),
        "third probe at {enter3}"
    );
    assert!(
        s.h.counters
            .summary()
            .contains("rate demotions=2; MTP accept"),
        "{}",
        s.h.counters.summary()
    );
}

/// 2026-10-11: (c) Code: r_D = 1.5 r_M never demotes, in either gate setting.
#[test]
fn code_trace_never_demotes() {
    let mut s = Sim::new(cfg(2000, gate(0)), 40.0, 60.0);
    s.run_to(60_000);
    assert_eq!(s.h.counters.rate_demotions, 0);
    assert!(s.modes_between(2000, 60_000).iter().all(|m| *m == Dflash));
}

/// 2026-10-11: The margin: r_D = 0.95 r_M is demoted at 0 %, kept at 10 %.
#[test]
fn margin_keeps_a_small_loss() {
    for (margin, want) in [(0u32, 1u64), (10, 0), (4, 1)] {
        let mut s = Sim::new(cfg(2000, gate(margin)), 40.0, 38.0);
        s.run_to(8000);
        assert_eq!(s.h.counters.rate_demotions.min(1), want, "margin {margin}");
    }
}

/// 2026-10-11: The first 1 s of a stretch is excluded: a slow bootstrap alone never demotes.
#[test]
fn bootstrap_is_excluded_from_r_d() {
    struct Slow(Sim);
    let mut s = Slow(Sim::new(cfg(2000, gate(0)), 30.0, 60.0));
    // Fast DFlash, but its first second is dead (a switch cost).
    s.0.run_to(2000);
    let t_enter = s.0.ms;
    while s.0.ms < t_enter + 1000 {
        s.0.r_d = 0.0;
        s.0.step(1);
    }
    s.0.r_d = 60.0;
    s.0.run_to(t_enter + 6000);
    assert_eq!(s.0.h.counters.rate_demotions, 0);
}

/// 2026-10-11: (d) Leaving DFlash for occupancy is not a demotion: k, and a running cool-down,
/// survive it.
#[test]
fn occupancy_exit_does_not_count_as_a_demotion() {
    let mut s = Sim::new(cfg(2000, gate(0)), 30.0, 27.0);
    let demote1 = {
        s.run_until_mode(Dflash, 3000);
        s.run_until_mode(Mtp, 6000)
    };
    // Occupancy during the cool-down does not end it.
    for _ in 0..4 {
        s.step(2);
    }
    let resume = s.ms;
    s.run_to(demote1 + 8000 - TICK_MS);
    assert!(
        s.modes_between(resume, demote1 + 8000)
            .iter()
            .all(|m| *m == Mtp),
        "cool-down ended early"
    );
    // Cool-down ends; a fresh dwell later DFlash again (k = 1 after one demotion).
    let enter2 = s.run_until_mode(Dflash, demote1 + 12_000);
    // Occupancy while on DFlash: leaves at once, no demotion counted.
    s.run_to(enter2 + 500);
    assert_eq!(s.step(2), Mtp);
    assert_eq!(s.h.counters.rate_demotions, 1);
    // Alone again: the dwell, then DFlash; the demotion that follows has k = 1 (16 s), not 2.
    let enter3 = s.run_until_mode(Dflash, s.ms + 4000);
    let demote2 = s.run_until_mode(Mtp, enter3 + 4000);
    assert_eq!(s.h.counters.rate_demotions, 2);
    s.run_to(demote2 + 16_000 - TICK_MS);
    assert!(
        s.modes_between(demote2, demote2 + 16_000)
            .iter()
            .all(|m| *m == Mtp),
        "cool-down shorter than C_1"
    );
    let enter4 = s.run_until_mode(Dflash, demote2 + 20_000);
    assert!(
        enter4 >= demote2 + 16_000 + 2000,
        "k advanced past 1: re-probe at {enter4}, demotion at {demote2}"
    );
    // And the occupancy exit itself changed nothing about k: the third cool-down would be 32 s.
    let demote3 = s.run_until_mode(Mtp, enter4 + 4000);
    s.run_to(demote3 + 32_000 - TICK_MS);
    assert!(
        s.modes_between(demote3, demote3 + 32_000)
            .iter()
            .all(|m| *m == Mtp)
    );
}

/// 2026-10-11: (e) A new sole sequence starts with k = 0 and no cool-down.
#[test]
fn new_sole_sequence_resets_k() {
    let mut s = Sim::new(cfg(2000, gate(0)), 30.0, 27.0);
    s.run_until_mode(Dflash, 3000);
    let demote1 = s.run_until_mode(Mtp, 6000);
    // Second demotion on the same sequence would bar 16 s; first re-probe at +10 s.
    let enter2 = s.run_until_mode(Dflash, demote1 + 12_000);
    let demote2 = s.run_until_mode(Mtp, enter2 + 4000);
    assert_eq!(s.h.counters.rate_demotions, 2);
    // Request B replaces A while A is barred (k = 2 now): new prompt length, fresh output.
    s.prompt_len = 70;
    s.toks = 1.0;
    let b_start = s.ms;
    let enter_b = s.run_until_mode(Dflash, b_start + 3000);
    assert!(
        (b_start + 2000..=b_start + 2000 + TICK_MS).contains(&enter_b),
        "B waits only the dwell: DFlash at {enter_b}, B started {b_start}"
    );
    // B's own first demotion has the base cool-down, 8000 ms (k = 0).
    let demote_b = s.run_until_mode(Mtp, enter_b + 4000);
    assert!(demote_b > demote2);
    s.run_to(demote_b + 8000 + 2000 - TICK_MS);
    let enter_b2 = s.run_until_mode(Dflash, demote_b + 12_000);
    assert!(
        (demote_b + 10_000..=demote_b + 10_000 + 2 * TICK_MS).contains(&enter_b2),
        "B re-probe at {enter_b2}, demotion at {demote_b}"
    );
}

/// 2026-10-11: (f) A dwell window shorter than 0.5 s, or with fewer than 8 tokens, gives no gate
/// decision even when DFlash is slower.
#[test]
fn short_or_sparse_dwell_window_never_demotes() {
    // Window 300 ms (dwell 300): r_M invalid.
    let mut s = Sim::new(cfg(300, gate(0)), 30.0, 10.0);
    s.run_to(20_000);
    assert_eq!(s.h.counters.rate_demotions, 0);
    assert!(s.modes_between(300, 20_000).iter().all(|m| *m == Dflash));
    // Window 2 s but 3 tok/s -> 6 tokens: r_M invalid.
    let mut s = Sim::new(cfg(2000, gate(0)), 3.0, 1.0);
    s.run_to(20_000);
    assert_eq!(s.h.counters.rate_demotions, 0);
    // Boundary: 4.5 tok/s over 2 s = 9 tokens is valid, and DFlash at 3 demotes.
    let mut s = Sim::new(cfg(2000, gate(0)), 4.5, 3.0);
    s.run_to(8000);
    assert_eq!(s.h.counters.rate_demotions, 1);
}

/// 2026-10-11: (a) With the gate off the decisions are exactly D2s (`Dwell` + `decide_mode`),
/// whatever the rates; and a gate that never fires (code-like) decides the same.
#[test]
fn gate_off_matches_plain_d2s_on_a_scripted_trace() {
    let t0 = Instant::now();
    let mut off = HybridState::new(cfg(2000, None));
    let mut on = HybridState::new(cfg(2000, gate(0)));
    let mut dwell = Dwell::new(Duration::from_millis(2000));
    let mut toks = 1usize;
    let (mut ms, mut seed) = (0u64, 7u64);
    let (mut hold, mut level) = (0u64, 1usize);
    let mut modes = [0u64; 3];
    for i in 0..3000u64 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Occupancy churn in bursts: a level holds for 1.5-6.5 s (30-130 ticks).
        if hold == 0 {
            hold = 30 + (seed >> 33) % 100;
            level = match (seed >> 40) % 8 {
                0 => 2,
                1 => 9,
                2 => 0,
                _ => 1,
            };
        }
        hold -= 1;
        let active = level;
        let ok = (seed >> 20) % 17 != 0;
        // A new request on the same slot now and then.
        let prompt_len = 100 + (i / 700) as usize;
        let sole = (active == 1).then_some(SoleSeq {
            slot: 0,
            prompt_len,
            generated: toks,
        });
        let now = t0 + Duration::from_millis(ms);
        let want = decide_mode(&ModeInputs {
            active,
            mtp_max_seqs: 8,
            sole_dflash_ok: ok,
            dwell_met: dwell.observe(sole, now),
        });
        let got_off = off.decide(active, 8, sole, ok, now);
        let got_on = on.decide(active, 8, sole, ok, now);
        assert_eq!(got_off, want, "gate off, step {i} t={ms}");
        // The on-gate trace differs only if a demotion fires; r_D is as fast as r_M here
        // (the same tokens per tick whatever the mode), so it must not.
        assert_eq!(got_on, want, "gate on, equal rates, step {i} t={ms}");
        modes[want as usize] += 1;
        toks += 2;
        ms += TICK_MS;
    }
    assert_eq!(on.counters.rate_demotions, 0);
    assert!(
        modes[Dflash as usize] > 100,
        "trace never reached DFlash: {modes:?}"
    );
    assert!(
        modes[Plain as usize] > 0,
        "trace never went plain: {modes:?}"
    );
    assert!(modes[Mtp as usize] > 0);
}

/// 2026-10-11: Gate off: a slow-DFlash trace stays on DFlash (D2s), where gate on demotes.
#[test]
fn gate_off_never_bars() {
    let mut s = Sim::new(cfg(2000, None), 30.0, 15.0);
    s.run_to(30_000);
    assert_eq!(s.h.counters.rate_demotions, 0);
    assert!(s.modes_between(2000, 30_000).iter().all(|m| *m == Dflash));
}

#[test]
fn cooldown_doubles_and_caps_at_k6() {
    let g = RateGateCfg {
        margin_pct: 0,
        cooldown: Duration::from_millis(8000),
    };
    assert_eq!(g.cooldown_for(0), Duration::from_millis(8000));
    assert_eq!(g.cooldown_for(1), Duration::from_millis(16_000));
    assert_eq!(g.cooldown_for(3), Duration::from_millis(64_000));
    assert_eq!(g.cooldown_for(6), Duration::from_millis(512_000));
    assert_eq!(g.cooldown_for(7), g.cooldown_for(6));
    assert_eq!(g.cooldown_for(u32::MAX), g.cooldown_for(6));
}

/// 2026-10-11: (g) The lever parsers.
#[test]
fn rate_lever_parsers() {
    assert_eq!(hy::parse_rate_margin_pct(None), 0);
    assert_eq!(hy::HYBRID_RATE_MARGIN_PCT_DEFAULT, 0);
    assert_eq!(hy::parse_rate_margin_pct(Some("0")), 0);
    assert_eq!(hy::parse_rate_margin_pct(Some(" 12 ")), 12);
    assert_eq!(hy::parse_rate_margin_pct(Some("50")), 50);
    assert_eq!(hy::parse_rate_margin_pct(Some("51")), 0);
    assert_eq!(hy::parse_rate_margin_pct(Some("-5")), 0);
    assert_eq!(hy::parse_rate_margin_pct(Some("10%")), 0);
    assert_eq!(hy::parse_rate_margin_pct(Some("")), 0);

    assert_eq!(hy::parse_rate_cooldown_ms(None), 8000);
    assert_eq!(hy::HYBRID_RATE_COOLDOWN_MS_DEFAULT, 8000);
    assert_eq!(hy::parse_rate_cooldown_ms(Some("0")), 0);
    assert_eq!(hy::parse_rate_cooldown_ms(Some(" 500 ")), 500);
    assert_eq!(hy::parse_rate_cooldown_ms(Some("-1")), 8000);
    assert_eq!(hy::parse_rate_cooldown_ms(Some("8s")), 8000);

    // The gate switch: exactly `1`, as the other hybrid switches.
    assert!(metrale_model_layers::speculative::glm_dflash::parse_switch(
        Some("1")
    ));
    assert!(!metrale_model_layers::speculative::glm_dflash::parse_switch(Some("true")));
    assert!(!metrale_model_layers::speculative::glm_dflash::parse_switch(None));
}
