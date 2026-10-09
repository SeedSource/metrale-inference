// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: Unit tests for the DFlash gamma resolver. The pure rules are
//! tested with explicit [`Rungs`]; the controller tests drive
//! `configure_with`, `drafts_for` and `observe_step` on their own
//! `DflashRung`.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use std::sync::Mutex;

use super::*;

/// 2026-09-25: The controller tests take this lock and so run one at a time.
/// Each builds its own `DflashRung`; they share no controller state.
static SERIAL: Mutex<()> = Mutex::new(());

fn r10() -> Rungs {
    Rungs::defaults(10)
}

#[test]
fn concurrency_ladder_lands_the_records() {
    let r = r10();
    // 2026-09-25: K = 4 at C >= 2, the measured winner at C = 16 for both
    // workloads (module doc table).
    assert_eq!(k_for(16, true, &r), 4);
    assert_eq!(k_for(16, false, &r), 4);
    assert_eq!(k_for(2, true, &r), 4);
    // 2026-09-25: C = 1 code measured best at the full block (66.8), prose at
    // K = 5 (27.0).
    assert_eq!(k_for(1, true, &r), 10);
    assert_eq!(k_for(1, false, &r), 5);
}

#[test]
fn widths_never_exceed_the_cap() {
    // 2026-09-25: A head sized for K = 4 is never asked for more.
    let r4 = Rungs::defaults(4);
    assert_eq!(k_for(1, true, &r4), 4);
    assert_eq!(k_for(1, false, &r4), 4);
    assert_eq!(k_for(16, true, &Rungs::defaults(3)), 3);
}

/// 2026-09-25: Without the write-on-accept kernel (`METRALE_GDN_WOA=1` not
/// set) the C >= 2 rung is the cap; the C = 1 rungs are unaffected.
#[test]
fn no_woa_pins_the_multi_rung_at_the_cap() {
    let r = Rungs::from_env(10, false);
    assert_eq!(r.multi, 10);
    assert_eq!(r.narrow, Rungs::defaults(10).narrow);
    assert_eq!(r.wide, 10);
    assert_eq!(k_for(16, true, &r), 10);
    assert_eq!(k_for(1, false, &r), 5);
}

#[test]
fn defaults_below_cap_two_pin_instead_of_panicking() {
    // 2026-09-25: `Ord::clamp` panics when min > max, so `defaults` clamps to
    // `2..=max(cap, 2)`: a cap below 2 gives 2 for every width.
    for cap in [0, 1] {
        let r = Rungs::defaults(cap);
        assert_eq!((r.multi, r.narrow, r.wide), (2, 2, 2), "cap {cap}");
    }
}

#[test]
fn hysteresis_separates_prose_from_code() {
    let r = r10();
    // 2026-09-25: Hit counts from the 2026-09-04 rates (module doc): narrow
    // rung prose 0.76 (49/64) and code 0.97 (62/64); wide rung code 0.875
    // (56/64) and prose 0.56..0.76.
    assert!(!next_wide(true, 40, &r), "prose must leave wide");
    assert!(next_wide(false, 62, &r), "code must enter wide");
    assert!(next_wide(true, 56, &r), "wide-rung code must hold wide");
    // 2026-09-25: From 47 to 59 hits either state holds.
    assert!(next_wide(true, 49, &r));
    assert!(!next_wide(false, 49, &r));
    assert!(next_wide(true, 58, &r));
    assert!(!next_wide(false, 58, &r));
    assert!(next_wide(false, 60, &r));
    assert!(!next_wide(true, 46, &r));
    assert!(next_wide(true, 47, &r));
}

#[test]
fn shift_register_is_exactly_one_window() {
    let mut r = 0u64;
    for _ in 0..WINDOW {
        r = shift_in(r, true);
    }
    assert_eq!(r.count_ones(), WINDOW);
    // 2026-09-25: The next step evicts the oldest hit.
    r = shift_in(r, false);
    assert_eq!(r.count_ones(), WINDOW - 1);
}

/// 2026-09-25: Deterministic Bernoulli stream (LCG) at hit rate `p`, `steps`
/// long.
fn stream(p: f64, steps: usize, seed: u64) -> Vec<bool> {
    let mut x = seed;
    (0..steps)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) < p
        })
        .collect()
}

/// 2026-09-25: Drives a controller (cap 10, default rungs) from `start_wide`
/// over a C = 1 stream, each step one `drafts_for(1, 9)` then one
/// `observe_step`. Returns (flips, final state is wide). The caller holds
/// `SERIAL`.
fn simulate(start_wide: bool, hits: &[bool]) -> (u64, bool) {
    let rung = DflashRung::new();
    rung.configure_with(10, false, r10());
    assert!(rung.armed());
    if !start_wide {
        // 2026-09-25: Two windows of misses; the first decision, after one
        // window, goes narrow.
        for _ in 0..(2 * WINDOW) {
            rung.drafts_for(1, 9);
            rung.observe_step(false);
        }
        assert_eq!(rung.drafts_for(1, 9), 4, "setup did not reach narrow");
    }
    let before = rung.flips();
    for &h in hits {
        rung.drafts_for(1, 9);
        rung.observe_step(h);
    }
    let wide = rung.drafts_for(1, 9) == 9;
    (rung.flips() - before, wide)
}

/// 2026-09-25: Most flips allowed over 20 × 2000 steps of steady code at the
/// wide-rung first-draft rate (0.875).
const CODE_WIDE_FLIP_BOUND: u64 = 16;

#[test]
fn steady_workloads_never_flip_and_transitions_do() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // 2026-09-25: 20 seeds × 2000 steps of steady prose (0.76) from the
    // narrow state and of steady code (0.97) from the wide state.
    let prose: u64 = (1..=20u64)
        .map(|s| simulate(false, &stream(0.76, 2000, s)).0)
        .sum();
    let code: u64 = (1..=20u64)
        .map(|s| simulate(true, &stream(0.97, 2000, s)).0)
        .sum();
    // 2026-09-25: Code at the rate measured on the wide rung (0.875,
    // 2026-09-04).
    let code_wide: u64 = (1..=20u64)
        .map(|s| simulate(true, &stream(0.875, 2000, s)).0)
        .sum();
    assert!(
        prose <= 6,
        "steady prose flipped {prose} times in 40k steps"
    );
    assert!(code <= 2, "steady code flipped {code} times in 40k steps");
    assert!(
        code_wide <= CODE_WIDE_FLIP_BOUND,
        "steady wide-rung code (0.875) flipped {code_wide} times in 40k steps (bound {CODE_WIDE_FLIP_BOUND})"
    );
    // 2026-09-25: A change of workload (prose then code, and back) must end
    // in the matching state. Over 20 seeds × 2 directions that is 40 real
    // flips; the total allows 6 more.
    let mut total = 0u64;
    for seed in 1..=20u64 {
        let mut s = stream(0.76, 400, seed);
        s.extend(stream(0.97, 400, seed + 100));
        let (flips, wide) = simulate(false, &s);
        assert!(
            wide && flips >= 1,
            "prose->code seed {seed}: flips={flips} wide={wide}"
        );
        total += flips;
        let mut s = stream(0.97, 400, seed);
        s.extend(stream(0.76, 400, seed + 100));
        let (flips, wide) = simulate(true, &s);
        assert!(
            !wide && flips >= 1,
            "code->prose seed {seed}: flips={flips} wide={wide}"
        );
        total += flips;
    }
    assert!(total <= 46, "transitions: {total} flips for 40 real ones");
}

/// 2026-09-25: After a switch, a full window must pass before the next
/// decision.
#[test]
fn dwell_holds_one_full_window_after_a_switch() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let rung = DflashRung::new();
    rung.configure_with(10, false, r10());
    // 2026-09-25: Starting wide, misses reach the first decision after one
    // dwell, and it goes narrow.
    for i in 0..WINDOW {
        rung.drafts_for(1, 9);
        rung.observe_step(false);
        assert_eq!(
            rung.drafts_for(1, 9),
            if i + 1 < WINDOW { 9 } else { 4 },
            "step {i}"
        );
    }
    // 2026-09-25: Then hits: no flip inside the dwell.
    let before = rung.flips();
    for i in 0..(WINDOW - 1) {
        rung.drafts_for(1, 9);
        rung.observe_step(true);
        assert_eq!(
            rung.drafts_for(1, 9),
            4,
            "flipped inside the dwell at step {i}"
        );
    }
    rung.drafts_for(1, 9);
    rung.observe_step(true);
    assert_eq!(
        rung.drafts_for(1, 9),
        9,
        "a full window of hits must go wide"
    );
    assert_eq!(rung.flips() - before, 1);
}

/// 2026-09-25: `observe_step` after a C >= 2 dispatch is not scored.
#[test]
fn observer_ignores_multi_sequence_steps() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let rung = DflashRung::new();
    rung.configure_with(10, false, r10());
    for _ in 0..(4 * WINDOW) {
        rung.drafts_for(16, 9);
        rung.observe_step(false);
    }
    assert_eq!(
        rung.drafts_for(1, 9),
        9,
        "C=16 misses must not move the C=1 state"
    );
}

#[test]
fn pinned_returns_num_drafts_unchanged() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let rung = DflashRung::new();
    // 2026-09-25: Pinned, as under an explicit `--dflash-gamma`: the draft
    // count passes through.
    rung.configure_with(10, true, r10());
    assert!(!rung.armed());
    assert_eq!(rung.drafts_for(16, 9), 9);
    assert_eq!(rung.drafts_for(1, 9), 9);
    // 2026-09-25: Armed: C = 16 gives 3 drafts (K = 4), C = 1 wide 9 (K = 10).
    rung.configure_with(10, false, r10());
    assert!(rung.armed());
    assert_eq!(rung.drafts_for(16, 9), 3);
    assert_eq!(rung.drafts_for(1, 9), 9);
}

/// 2026-10-09: The GLM DFlash levers (`glm_dflash_multi_k`) under `--dflash-gamma 8`
/// (pinned, write-on-accept off so the C>=2 rung is the cap): lever off, every C keeps K=8;
/// lever on, C>=2 gets K=4 (3 drafts) and C=1 keeps K=8. Armed by the resolver env, the pin
/// still sets C>=2 and C=1 keeps the wide rung.
#[test]
fn glm_dflash_levers_pin_c2_to_k4_under_explicit_gamma_8() {
    use metrale_model_layers::speculative::glm_dflash_batch::glm_dflash_multi_k;
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let cap_rungs = Rungs {
        multi: 8,
        ..Rungs::defaults(8)
    };
    for (batched, perseq) in [(false, false), (true, false), (false, true), (true, true)] {
        let rung = DflashRung::new();
        rung.configure_with(8, true, cap_rungs);
        assert!(!rung.armed());
        if let Some(k) = glm_dflash_multi_k("glm5_next", batched, perseq) {
            rung.pin_multi(k);
        }
        let on = batched || perseq;
        assert_eq!(rung.drafts_for(1, 7), 7, "C=1 unchanged (lever {on})");
        for c in [2, 4] {
            let want = if on { 3 } else { 7 };
            assert_eq!(rung.drafts_for(c, 7), want, "C={c} lever {on}");
        }
    }
    // 2026-10-09: Not GLM: no pin.
    assert_eq!(glm_dflash_multi_k("qwen3_5", true, true), None);
    // 2026-10-09: Armed (METRALE_DFLASH_GAMMA_RESOLVER): the pin wins at C>=2 only.
    let rung = DflashRung::new();
    rung.configure_with(8, false, cap_rungs);
    rung.pin_multi(4);
    assert!(rung.armed());
    assert_eq!(rung.drafts_for(2, 7), 3);
    assert_eq!(rung.drafts_for(4, 7), 3);
    assert_eq!(rung.drafts_for(1, 7), 7);
    // 2026-10-09: Other pinned widths flow through (a later per-step K in 2..=4).
    for (k, nd) in [(2, 1), (3, 2), (4, 3)] {
        let rung = DflashRung::new();
        rung.configure_with(8, true, cap_rungs);
        rung.pin_multi(k);
        assert_eq!(rung.drafts_for(2, 7), nd, "K={k}");
        assert_eq!(rung.drafts_for(1, 7), 7, "K={k} C=1");
    }
}
