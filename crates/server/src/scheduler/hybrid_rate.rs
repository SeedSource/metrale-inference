// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Policy D2sR of the hybrid design note (f): per sole sequence, the MTP wall rate
//! r_M over the dwell, the DFlash wall rate r_D over the last second of a DFlash stretch, and the
//! rate gate that demotes a sequence whose DFlash rate falls below its MTP rate
//! (`METRALE_HYBRID_RATE_GATE`). The tracker observes whether or not the gate is on (the stretch
//! log line is observation only); only the demotion needs the gate.
//!
//! Owner: scheduler.
//! Invariants:
//! - Pure: the caller feeds `now` and the sole sequence's `generated` count; no clock or I/O.
//! - With `cfg == None` the tracker never bars a sequence, so the step modes are exactly D2s.
//! - Everything per sequence (demotion count k, cool-down, MTP window) resets when the sole
//!   sequence is a different one (`SoleSeq::same_as`). Leaving DFlash because a second sequence
//!   arrived is not a demotion and leaves k alone.
//! - r_M needs a dwell window of at least [`MIN_MTP_WINDOW`] and [`MIN_MTP_TOKENS`]; else the
//!   stretch has no gate decision (D2s behaviour).
//! - r_D covers the last [`WINDOW_D`] of DFlash steps and is evaluated only once that whole
//!   window lies after the first [`BOOTSTRAP`] of the stretch.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::hybrid_mode::SoleSeq;
use metrale_model_layers::speculative::hybrid::SpecStepMode;

/// 2026-10-11: The first part of each DFlash stretch is a switch cost, not the steady rate.
pub const BOOTSTRAP: Duration = Duration::from_millis(1000);
/// 2026-10-11: W_D: the rolling window of r_D.
pub const WINDOW_D: Duration = Duration::from_millis(1000);
/// 2026-10-11: The shortest dwell window whose MTP rate counts.
pub const MIN_MTP_WINDOW: Duration = Duration::from_millis(500);
/// 2026-10-11: The fewest tokens a dwell window must hold for its MTP rate to count.
pub const MIN_MTP_TOKENS: usize = 8;
/// 2026-10-11: Cap on the cool-down exponent k.
pub const MAX_COOLDOWN_EXP: u32 = 6;

/// 2026-10-11: The rate gate's settings (`None` in `HybridConfig` = gate off).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateGateCfg {
    /// 2026-10-11: `METRALE_HYBRID_RATE_MARGIN_PCT`, 0..=50.
    pub margin_pct: u32,
    /// 2026-10-11: `METRALE_HYBRID_RATE_COOLDOWN_MS`: the cool-down of a first demotion.
    pub cooldown: Duration,
}

impl RateGateCfg {
    /// 2026-10-11: `C_k = cooldown * 2^k`, k capped at [`MAX_COOLDOWN_EXP`].
    pub fn cooldown_for(&self, k: u32) -> Duration {
        self.cooldown
            .saturating_mul(1u32 << k.min(MAX_COOLDOWN_EXP))
    }
}

/// 2026-10-11: Tokens per second over `[t0, t1]`; `None` for an empty interval.
fn rate(t0: Instant, g0: usize, t1: Instant, g1: usize) -> Option<f64> {
    let dt = t1.saturating_duration_since(t0).as_secs_f64();
    (dt > 0.0).then(|| g1.saturating_sub(g0) as f64 / dt)
}

fn fmt_rate(r: Option<f64>) -> String {
    r.map_or_else(|| "n/a".to_string(), |r| format!("{r:.1}tok/s"))
}

/// 2026-10-11: One DFlash stretch of the sole sequence, entry to the last DFlash step.
#[derive(Debug, Clone)]
struct Stretch {
    t_enter: Instant,
    g_enter: usize,
    t_last: Instant,
    g_last: usize,
    /// 2026-10-11: The sequence's r_M at entry (`None` = no valid window).
    r_m: Option<f64>,
    /// 2026-10-11: Steps after the bootstrap; the front is the newest sample at least
    /// [`WINDOW_D`] old (older ones are dropped).
    samples: VecDeque<(Instant, usize)>,
    /// 2026-10-11: The first post-bootstrap step, for the whole-stretch r_D of the log line.
    post: Option<(Instant, usize)>,
    /// 2026-10-11: The rolling r_D at the last step, once a full window exists.
    r_d: Option<f64>,
}

impl Stretch {
    fn new(now: Instant, generated: usize, r_m: Option<f64>) -> Self {
        Self {
            t_enter: now,
            g_enter: generated,
            t_last: now,
            g_last: generated,
            r_m,
            samples: VecDeque::new(),
            post: None,
            r_d: None,
        }
    }

    fn observe(&mut self, now: Instant, generated: usize) {
        self.t_last = now;
        self.g_last = generated;
        if now.saturating_duration_since(self.t_enter) < BOOTSTRAP {
            self.r_d = None;
            return;
        }
        self.post.get_or_insert((now, generated));
        self.samples.push_back((now, generated));
        while self.samples.len() >= 2
            && now.saturating_duration_since(self.samples[1].0) >= WINDOW_D
        {
            self.samples.pop_front();
        }
        let (t0, g0) = self.samples[0];
        self.r_d = if now.saturating_duration_since(t0) >= WINDOW_D {
            rate(t0, g0, now, generated)
        } else {
            None
        };
    }

    /// 2026-10-11: The mean DFlash rate after the bootstrap, for the stretch line.
    fn r_d_stretch(&self) -> Option<f64> {
        let (t, g) = self.post?;
        rate(t, g, self.t_last, self.g_last)
    }
}

/// 2026-10-11: Per-sequence rate tracking and the demotion gate (module docs).
#[derive(Debug, Clone)]
pub struct RateTracker {
    cfg: Option<RateGateCfg>,
    /// 2026-10-11: The last sole sequence seen; kept across steps with several active.
    ident: Option<SoleSeq>,
    /// 2026-10-11: k: demotions of the sequence in `ident`.
    demotions: u32,
    barred_until: Option<Instant>,
    /// 2026-10-11: Start of the current MTP window (the dwell): time and `generated`.
    mtp_start: Option<(Instant, usize)>,
    stretch: Option<Stretch>,
}

impl RateTracker {
    pub fn new(cfg: Option<RateGateCfg>) -> Self {
        Self {
            cfg,
            ident: None,
            demotions: 0,
            barred_until: None,
            mtp_start: None,
            stretch: None,
        }
    }

    /// 2026-10-11: Before the dwell is observed: note this step's sole sequence and return
    /// whether DFlash is barred for it (cool-down running). A barred step also drops the MTP
    /// window, so the dwell, and with it r_M, restarts when the cool-down ends.
    pub fn begin_step(&mut self, sole: Option<SoleSeq>, now: Instant) -> bool {
        if let Some(s) = sole {
            if !self.ident.is_some_and(|p| s.same_as(&p)) {
                self.demotions = 0;
                self.barred_until = None;
                self.mtp_start = None;
            }
            self.ident = Some(s);
        }
        let barred = self.barred_until.is_some_and(|u| now < u);
        if !barred {
            self.barred_until = None;
        }
        match sole {
            Some(s) if !barred => {
                if self.mtp_start.is_none() && self.stretch.is_none() {
                    self.mtp_start = Some((now, s.generated));
                }
            }
            _ => self.mtp_start = None,
        }
        barred
    }

    /// 2026-10-11: After the mode is decided: track the stretch. Returns `true` when the gate
    /// demotes the sequence this step (the caller then runs `Mtp`/`Plain`, not DFlash).
    pub fn end_step(&mut self, mode: SpecStepMode, sole: Option<SoleSeq>, now: Instant) -> bool {
        let Some(s) = sole.filter(|_| mode == SpecStepMode::Dflash) else {
            self.end_stretch();
            return false;
        };
        let Some(st) = self.stretch.as_mut() else {
            let r_m = self.mtp_start.take().and_then(|(t0, g0)| {
                let long_enough = now.saturating_duration_since(t0) >= MIN_MTP_WINDOW
                    && s.generated.saturating_sub(g0) >= MIN_MTP_TOKENS;
                long_enough
                    .then(|| rate(t0, g0, now, s.generated))
                    .flatten()
            });
            self.stretch = Some(Stretch::new(now, s.generated, r_m));
            return false;
        };
        st.observe(now, s.generated);
        let (Some(cfg), Some(r_m), Some(r_d)) = (self.cfg, st.r_m, st.r_d) else {
            return false;
        };
        if r_d >= r_m * (1.0 - f64::from(cfg.margin_pct) / 100.0) {
            return false;
        }
        let k = self.demotions;
        let cooldown = cfg.cooldown_for(k);
        self.demotions += 1;
        self.barred_until = Some(now + cooldown);
        self.mtp_start = None;
        tracing::info!(
            "HYBRID DFLASH: rate demote slot={} r_D={:.1}tok/s r_M={:.1}tok/s k={k} cooldown_ms={}",
            s.slot,
            r_d,
            r_m,
            cooldown.as_millis()
        );
        self.end_stretch();
        true
    }

    /// 2026-10-11: Close the open stretch, if any, with its log line (observation only).
    fn end_stretch(&mut self) {
        if let Some(st) = self.stretch.take() {
            tracing::info!(
                "HYBRID DFLASH: stretch r_D={} r_M={} tokens={} ms={}",
                fmt_rate(st.r_d_stretch()),
                fmt_rate(st.r_m),
                st.g_last.saturating_sub(st.g_enter),
                st.t_last.saturating_duration_since(st.t_enter).as_millis()
            );
        }
    }
}
