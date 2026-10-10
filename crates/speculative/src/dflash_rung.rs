// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: The DFlash gamma resolver: the per-step verify width K of a
//! block-diffusion drafter, chosen by concurrency and, for a single stream,
//! by the first-draft accept rate.
//!
//! Owner: speculative.
//! Invariants:
//! - Unarmed (pinned, never configured, or a cap below 3),
//!   `DflashRung::drafts_for` returns `num_drafts` unchanged.
//! - Armed, it returns `K - 1` capped at `num_drafts`, and at least 1.
//! - The single-stream state changes at most once per `dwell` single-stream
//!   verify steps.
//! - A C >= 2 pin (`DflashRung::pin_multi`, 2026-10-09) sets K at C >= 2 armed or
//!   not, capped at `num_drafts + 1`; C = 1 is unchanged by it.
//!
//! K is `Rungs::multi` at C >= 2, and at C = 1 `Rungs::wide` or
//! `Rungs::narrow` by the single-stream state (`k_for`). That state is decided
//! on the popcount of a shift register holding one bit per C = 1 verify step
//! (first draft accepted) over the last [`WINDOW`] steps: it goes wide at
//! `enter` hits or more and narrow at `leave` hits or fewer (`next_wide`).
//!
//! Measured on Qwen3.8-27B NVFP4 + DFlash-2 (block 8), one DGX Spark,
//! 2026-09-03, tok/s:
//!
//! | C     | prose (Volvo)        | code (MinHeap)        |
//! |-------|----------------------|-----------------------|
//! | 1     | γ5 27.0 / γ10 24.6   | γ10 66.8 / γ4 ~23     |
//! | 16    | γ4 180.7 / γ5 119.5  | γ4 279.5 / γ10 214.2  |
//!
//! At C = 16, K = 4 won both workloads. K = 4 is the width of the
//! write-on-accept verify kernel, which is opt-in (`METRALE_GDN_WOA=1`,
//! `gdn_flags::gdn_woa_enabled`); without it `Rungs::from_env` puts the
//! C >= 2 rung at the cap. At C = 1 prose preferred a narrow block and code
//! the full block.
//!
//! Measured at C = 1 on 2026-09-04 (MinHeap / Volvo): code accepted the first
//! draft 0.97 of the time on the narrow rung and 0.875 on the wide rung, prose
//! 0.56..0.76 on either. Code on the narrow rung lost about 18% (51 vs
//! 63 tok/s), prose on the wide rung about 8% (24.8 vs 27.0). So `LEAVE` sits
//! low: at 46, a 64-step window of code at 0.875 has 46 hits or fewer with
//! probability about 6.5e-4 (binomial), and one of prose at 0.76 about 0.26.
//!
//! `DflashRung::configure` leaves the resolver unarmed when the operator
//! passes `--dflash-gamma` or sets `METRALE_DFLASH_STATIC_GAMMA` (presence);
//! `METRALE_DFLASH_GAMMA_RESOLVER` (presence) arms it even under an explicit
//! flag. It also reads the overrides, once, through `Rungs::from_env`:
//! `METRALE_DFLASH_RUNG_MULTI` (K at C >= 2, default 4),
//! `METRALE_DFLASH_RUNG_NARROW` (default 5) and `METRALE_DFLASH_RUNG_WIDE`
//! (default the cap), each clamped to `2..=cap`; `METRALE_DFLASH_RUNG_ENTER`
//! (default 60), `METRALE_DFLASH_RUNG_LEAVE` (46) and
//! `METRALE_DFLASH_RUNG_DWELL` (64).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// 2026-09-25: Verify width K at C >= 2: the write-on-accept kernel's width.
const MULTI_K: usize = 4;
/// 2026-09-25: Single-stream narrow width (prose). Measured 2026-09-03: 27.0
/// tok/s at K = 5 vs 24.6 at K = 10 (module doc).
const NARROW_K: usize = 5;
/// 2026-09-25: Window length in verify steps: every bit of the `u64` shift
/// register.
pub const WINDOW: u32 = 64;
/// 2026-09-25: Go wide when at least this many of the last `WINDOW` first
/// drafts were accepted (60/64 = 0.9375; on the narrow rung code measured
/// 0.97 and prose at most 0.76, module doc).
const ENTER: u32 = 60;
/// 2026-09-25: Go narrow when at most this many were (46/64 = 0.72). On the
/// wide rung code measured 0.875 (56/64) and prose 0.56..0.76 (36..49 of 64),
/// so 46 is 3.8 binomial sigma under code and inside prose. Between `LEAVE`
/// and `ENTER` the state holds.
const LEAVE: u32 = 46;
/// 2026-09-25: Minimum single-stream verify steps between two switches: one
/// full window, so every decision sees a register refilled since the last
/// switch.
const DWELL: u64 = WINDOW as u64;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// 2026-09-25: The resolver's widths and thresholds. `defaults` and
/// `from_env` clamp each width to `2..=max(cap, 2)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rungs {
    /// 2026-09-25: K at C >= 2.
    pub multi: usize,
    /// 2026-09-25: K at C = 1 in the narrow state.
    pub narrow: usize,
    /// 2026-09-25: K at C = 1 in the wide state.
    pub wide: usize,
    /// 2026-09-25: Window hits at or above which C = 1 goes wide.
    pub enter: u32,
    /// 2026-09-25: Window hits at or below which C = 1 goes narrow.
    pub leave: u32,
    /// 2026-09-25: Minimum single-stream verify steps between two switches.
    pub dwell: u64,
}

impl Rungs {
    /// 2026-09-25: The compiled defaults for a head whose widest verify width
    /// is `cap`.
    pub fn defaults(cap: usize) -> Self {
        Self {
            multi: MULTI_K.clamp(2, cap.max(2)),
            narrow: NARROW_K.clamp(2, cap.max(2)),
            wide: cap.max(2),
            enter: ENTER,
            leave: LEAVE,
            dwell: DWELL,
        }
    }

    /// 2026-09-25: The defaults with the environment overrides applied, and
    /// the C >= 2 rung at the cap when the write-on-accept kernel is off.
    pub fn from_env(cap: usize, woa_available: bool) -> Self {
        let d = Self::defaults(cap);
        Self {
            multi: if woa_available {
                env_usize("METRALE_DFLASH_RUNG_MULTI", d.multi).clamp(2, cap.max(2))
            } else {
                cap.max(2)
            },
            narrow: env_usize("METRALE_DFLASH_RUNG_NARROW", d.narrow).clamp(2, cap.max(2)),
            wide: env_usize("METRALE_DFLASH_RUNG_WIDE", d.wide).clamp(2, cap.max(2)),
            enter: env_u32("METRALE_DFLASH_RUNG_ENTER", d.enter),
            leave: env_u32("METRALE_DFLASH_RUNG_LEAVE", d.leave),
            dwell: env_usize("METRALE_DFLASH_RUNG_DWELL", d.dwell as usize) as u64,
        }
    }
}

struct Ctl {
    /// 2026-09-25: The `cap_k` of the last `configure_with`; 0 before any.
    cap: AtomicUsize,
    /// 2026-09-25: `false`: `drafts_for` returns `num_drafts` and
    /// `observe_step` does nothing.
    armed: AtomicBool,
    /// 2026-09-25: The configured [`Rungs`], one atomic per field.
    multi: AtomicUsize,
    narrow: AtomicUsize,
    wide_k: AtomicUsize,
    enter: AtomicU32,
    leave: AtomicU32,
    dwell: AtomicU64,
    /// 2026-09-25: Single-stream state: `true` = wide.
    wide: AtomicBool,
    /// 2026-09-25: Shift register: bit i is set when the first draft was
    /// accepted i single-stream steps ago.
    hits: AtomicU64,
    /// 2026-09-25: Single-stream verify steps observed since the last
    /// `configure_with` (the dwell clock).
    tick: AtomicU64,
    last_switch: AtomicU64,
    /// 2026-09-25: `n_active` of the last armed `drafts_for` call;
    /// `observe_step` scores a step only when it is 1.
    last_n: AtomicUsize,
    flips: AtomicU64,
    /// 2026-10-09: K at C >= 2 whether armed or not (`pin_multi`); 0 = no pin.
    multi_pin: AtomicUsize,
    /// 2026-10-09: The pin belongs to the batched-verify lever, not the per-sequence control
    /// (`mark_multi_batched`).
    multi_batched: AtomicBool,
}

/// 2026-09-25: The DFlash gamma resolver. The serve builds and configures one
/// and the scheduler context holds it (`SchedCtx::dflash_rung`).
pub struct DflashRung {
    ctl: Ctl,
}

impl Default for DflashRung {
    fn default() -> Self {
        Self::new()
    }
}

impl DflashRung {
    /// 2026-09-25: Unconfigured and unarmed: `drafts_for` returns its input.
    pub const fn new() -> Self {
        Self {
            ctl: Ctl {
                cap: AtomicUsize::new(0),
                armed: AtomicBool::new(false),
                multi: AtomicUsize::new(MULTI_K),
                narrow: AtomicUsize::new(NARROW_K),
                wide_k: AtomicUsize::new(0),
                enter: AtomicU32::new(ENTER),
                leave: AtomicU32::new(LEAVE),
                dwell: AtomicU64::new(DWELL),
                wide: AtomicBool::new(true),
                hits: AtomicU64::new(0),
                tick: AtomicU64::new(0),
                last_switch: AtomicU64::new(0),
                last_n: AtomicUsize::new(0),
                flips: AtomicU64::new(0),
                multi_pin: AtomicUsize::new(0),
                multi_batched: AtomicBool::new(false),
            },
        }
    }

    fn rungs(&self) -> Rungs {
        let c = &self.ctl;
        Rungs {
            multi: c.multi.load(Ordering::Relaxed),
            narrow: c.narrow.load(Ordering::Relaxed),
            wide: c.wide_k.load(Ordering::Relaxed),
            enter: c.enter.load(Ordering::Relaxed),
            leave: c.leave.load(Ordering::Relaxed),
            dwell: c.dwell.load(Ordering::Relaxed),
        }
    }

    /// 2026-09-25: Serve-time configuration. `cap_k` is the head's widest
    /// verify width (its gamma); `explicit_flag` says the operator passed
    /// `--dflash-gamma`; `woa_available` says the K = 4 write-on-accept
    /// kernel is on (otherwise the C >= 2 rung is the cap). Reads the
    /// environment here.
    pub fn configure(&self, cap_k: usize, explicit_flag: bool, woa_available: bool) {
        let pinned = std::env::var_os("METRALE_DFLASH_STATIC_GAMMA").is_some()
            || (explicit_flag && std::env::var_os("METRALE_DFLASH_GAMMA_RESOLVER").is_none());
        self.configure_with(cap_k, pinned, Rungs::from_env(cap_k, woa_available));
        if !self.armed() {
            tracing::info!(
                "DFlash gamma PINNED at K={cap_k} ({})",
                if explicit_flag {
                    "--dflash-gamma explicit"
                } else {
                    "METRALE_DFLASH_STATIC_GAMMA"
                }
            );
        } else if !woa_available {
            tracing::info!(
                "DFlash gamma resolver: write-on-accept off (METRALE_GDN_WOA=1 not set), C>=2 rung falls back to the cap K={cap_k}"
            );
        }
    }

    /// 2026-09-25: Resets the controller to the wide state with explicit
    /// rungs and no environment reads. It arms only when `!pinned` and
    /// `cap_k >= 3`.
    pub fn configure_with(&self, cap_k: usize, pinned: bool, r: Rungs) {
        let c = &self.ctl;
        let armed = !pinned && cap_k >= 3;
        c.cap.store(cap_k, Ordering::Relaxed);
        c.multi.store(r.multi, Ordering::Relaxed);
        c.narrow.store(r.narrow, Ordering::Relaxed);
        c.wide_k.store(r.wide, Ordering::Relaxed);
        c.enter.store(r.enter, Ordering::Relaxed);
        c.leave.store(r.leave, Ordering::Relaxed);
        c.dwell.store(r.dwell, Ordering::Relaxed);
        // 2026-09-25: Single-stream starts wide: of the two wrong states, prose
        // on the wide rung measured the smaller loss (module doc).
        c.wide.store(true, Ordering::Relaxed);
        c.hits.store(0, Ordering::Relaxed);
        c.tick.store(0, Ordering::Relaxed);
        c.last_switch.store(0, Ordering::Relaxed);
        c.last_n.store(0, Ordering::Relaxed);
        c.armed.store(armed, Ordering::Relaxed);
        if armed {
            tracing::info!(
                "DFlash GAMMA RESOLVER armed: cap K={cap_k}; C>=2 -> K={}; C=1 -> K={} wide / K={} \
                 narrow on first-draft hits in the last {WINDOW}: enter>={} leave<={} dwell={}",
                r.multi,
                r.wide,
                r.narrow,
                r.enter,
                r.leave,
                r.dwell,
            );
        }
    }

    /// 2026-10-09: Pin K at C >= 2 to `k` (>= 2) even when the resolver is
    /// unarmed (an explicit `--dflash-gamma`): the GLM DFlash levers
    /// (`glm_dflash_batch::glm_dflash_multi_k`). `configure_with` keeps the pin.
    pub fn pin_multi(&self, k: usize) {
        let k = k.max(2);
        self.ctl.multi_pin.store(k, Ordering::Relaxed);
        tracing::info!("DFlash gamma: C>=2 PINNED at K={k} (C=1 unchanged)");
    }

    /// 2026-10-09: Whether a C >= 2 pin is set (`pin_multi`).
    pub fn multi_pinned(&self) -> bool {
        self.ctl.multi_pin.load(Ordering::Relaxed) >= 2
    }

    /// 2026-10-09: Record that the C >= 2 pin is the batched-verify lever's
    /// (`METRALE_GLM_DFLASH_BATCHED_VERIFY`), so the scheduler batches every eligible row at the
    /// pinned width; the per-sequence control arm (`METRALE_GLM_DFLASH_PERSEQ_K4`) never calls it.
    pub fn mark_multi_batched(&self) {
        self.ctl.multi_batched.store(true, Ordering::Relaxed);
    }

    /// 2026-10-09: A C >= 2 pin is set and it is the batched-verify lever's
    /// (`mark_multi_batched`).
    pub fn multi_batched(&self) -> bool {
        self.multi_pinned() && self.ctl.multi_batched.load(Ordering::Relaxed)
    }

    pub fn armed(&self) -> bool {
        self.ctl.armed.load(Ordering::Relaxed)
    }

    /// 2026-09-25: Single-stream switches (wide <-> narrow) so far.
    pub fn flips(&self) -> u64 {
        self.ctl.flips.load(Ordering::Relaxed)
    }

    /// 2026-09-25: The per-step draft count for `n_active` sequences.
    /// `num_drafts` is the serve's configured count (cap - 1) and is returned
    /// unchanged when the resolver is unarmed.
    pub fn drafts_for(&self, n_active: usize, num_drafts: usize) -> usize {
        let pin = self.ctl.multi_pin.load(Ordering::Relaxed);
        if !self.armed() {
            if pin >= 2 && n_active >= 2 {
                return (pin - 1).min(num_drafts).max(1);
            }
            return num_drafts;
        }
        self.ctl.last_n.store(n_active, Ordering::Relaxed);
        if pin >= 2 && n_active >= 2 {
            return (pin - 1).min(num_drafts).max(1);
        }
        let k = k_for(
            n_active,
            self.ctl.wide.load(Ordering::Relaxed),
            &self.rungs(),
        );
        (k - 1).min(num_drafts).max(1)
    }

    /// 2026-09-25: Scores one verify step: `d1_match` says the first draft
    /// was accepted. The DFlash verify step calls it every step; it does
    /// nothing unless the resolver is armed and the last dispatch was C = 1.
    /// A switch is logged when it happens.
    pub fn observe_step(&self, d1_match: bool) {
        let c = &self.ctl;
        if !self.armed() || c.last_n.load(Ordering::Relaxed) != 1 {
            return;
        }
        let register = shift_in(c.hits.load(Ordering::Relaxed), d1_match);
        c.hits.store(register, Ordering::Relaxed);
        let tick = c.tick.fetch_add(1, Ordering::Relaxed) + 1;
        // 2026-09-25: No decision within `dwell` steps of the last switch or
        // of arming.
        let r = self.rungs();
        if tick.saturating_sub(c.last_switch.load(Ordering::Relaxed)) < r.dwell {
            return;
        }
        let hits = register.count_ones();
        let was = c.wide.load(Ordering::Relaxed);
        let now = next_wide(was, hits, &r);
        if now != was {
            c.wide.store(now, Ordering::Relaxed);
            c.last_switch.store(tick, Ordering::Relaxed);
            let flips = c.flips.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(
                "DFlash gamma resolver C=1 -> K={} ({}; hits={hits}/{WINDOW} tick={tick} flips={flips})",
                k_for(1, now, &r),
                if now { "wide/code" } else { "narrow/prose" },
            );
        }
    }
}

/// 2026-09-25: Verify K for `n_active` sequences, given the rungs and the
/// single-stream state.
pub fn k_for(n_active: usize, wide: bool, r: &Rungs) -> usize {
    if n_active >= 2 {
        r.multi
    } else if wide {
        r.wide
    } else {
        r.narrow
    }
}

/// 2026-09-25: The next single-stream state from the number of first-draft
/// hits in the last `WINDOW` steps.
pub fn next_wide(wide: bool, hits: u32, r: &Rungs) -> bool {
    if wide {
        hits > r.leave
    } else {
        hits >= r.enter
    }
}

/// 2026-09-25: Shifts one step into the register and returns the new
/// register.
#[inline]
pub fn shift_in(register: u64, hit: bool) -> u64 {
    ((register << 1) | u64::from(hit)) & (u64::MAX >> (64 - WINDOW))
}

#[cfg(test)]
#[path = "dflash_rung_tests.rs"]
mod tests;
