// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Tests for `hybrid_mode.rs` (mode decision, D2s dwell, counters) and the shared
//! pieces in `metrale_model_layers::speculative::hybrid` (the `EP_CMD_SPEC_MODE` word, the dwell
//! parser, the drafter window) plus the hybrid SSM reserve. CPU only.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::time::{Duration, Instant};

use metrale_model_layers::speculative::hybrid::{
    self as hy, EP_CMD_SPEC_MODE, HYBRID_DWELL_MS_DEFAULT, SpecStepMode,
};

use super::*;

fn inputs(active: usize, mtp_max: usize, ok: bool, dwell: bool) -> ModeInputs {
    ModeInputs {
        active,
        mtp_max_seqs: mtp_max,
        sole_dflash_ok: ok,
        dwell_met: dwell,
    }
}

#[test]
fn decide_mode_table() {
    use SpecStepMode::*;
    // (active, mtp_max_seqs, sole_dflash_ok, dwell_met) -> mode
    let table: &[((usize, usize, bool, bool), SpecStepMode)] = &[
        ((1, 8, true, true), Dflash),
        ((1, 8, true, false), Mtp),
        ((1, 8, false, true), Mtp),
        ((1, 8, false, false), Mtp),
        // n >= 2 never runs DFlash, whatever the other inputs say.
        ((2, 8, true, true), Mtp),
        ((8, 8, true, true), Mtp),
        ((9, 8, true, true), Plain),
        ((16, 8, false, false), Plain),
        // MTP_MAX_SEQS = 1: n = 1 is MTP when DFlash is not allowed, n = 2 is plain.
        ((1, 1, false, true), Mtp),
        ((2, 1, true, true), Plain),
        // MTP_MAX_SEQS = 0 still lets the lone sequence run DFlash.
        ((1, 0, true, true), Dflash),
        ((1, 0, true, false), Plain),
    ];
    for &((a, m, ok, d), want) in table {
        assert_eq!(
            decide_mode(&inputs(a, m, ok, d)),
            want,
            "active={a} mtp_max={m} ok={ok} dwell={d}"
        );
    }
}

fn sole(slot: usize, prompt_len: usize, generated: usize) -> Option<SoleSeq> {
    Some(SoleSeq {
        slot,
        prompt_len,
        generated,
    })
}

#[test]
fn dwell_d2s_state_machine() {
    let t0 = Instant::now();
    let ms = |m: u64| t0 + Duration::from_millis(m);
    let mut d = Dwell::new(Duration::from_millis(2000));
    // (step time ms, sole sequence, dwell met)
    let steps: &[(u64, Option<SoleSeq>, bool)] = &[
        // A starts alone on an idle server: it still waits out the dwell.
        (0, sole(0, 100, 1), false),
        (1999, sole(0, 100, 40), false),
        (2000, sole(0, 100, 41), true),
        (2500, sole(0, 100, 60), true),
        // B arrives: leave at once.
        (2600, None, false),
        // B finishes; A alone again: the clock restarts.
        (3000, sole(0, 100, 90), false),
        (4999, sole(0, 100, 150), false),
        (5000, sole(0, 100, 151), true),
        // A finishes and C starts alone on the same slot: a new sequence (prompt differs).
        (5100, sole(0, 70, 1), false),
        (7100, sole(0, 70, 50), true),
        // Same slot and prompt length but the output shrank: a new sequence.
        (7200, sole(0, 70, 2), false),
        (9200, sole(0, 70, 30), true),
        // A different slot is a different sequence.
        (9300, sole(1, 70, 31), false),
    ];
    for &(t, s, want) in steps {
        assert_eq!(d.observe(s, ms(t)), want, "t={t} sole={s:?}");
    }
}

#[test]
fn dwell_zero_is_immediate() {
    let t0 = Instant::now();
    let mut d = Dwell::new(Duration::ZERO);
    assert!(d.observe(sole(0, 10, 1), t0));
    assert!(!d.observe(None, t0));
    assert!(d.observe(sole(0, 10, 2), t0));
}

fn cfg() -> HybridConfig {
    HybridConfig {
        mtp_num_drafts: 3,
        dflash_num_drafts: 7,
        dwell: Duration::from_millis(2000),
        rate_gate: None,
    }
}

#[test]
fn hybrid_state_churn_1_2_1() {
    use SpecStepMode::*;
    let t0 = Instant::now();
    let ms = |m: u64| t0 + Duration::from_millis(m);
    let mut h = HybridState::new(cfg());
    let run = |h: &mut HybridState, t: u64, active: usize, s: Option<SoleSeq>| {
        let m = h.decide(active, 8, s, true, ms(t));
        h.commit(m);
        h.count_step(m);
        m
    };
    assert_eq!(run(&mut h, 0, 1, sole(0, 10, 1)), Mtp);
    assert_eq!(run(&mut h, 2000, 1, sole(0, 10, 30)), Dflash);
    // `sole` is ignored unless exactly one sequence is active.
    assert_eq!(run(&mut h, 2100, 2, sole(0, 10, 31)), Mtp);
    assert_eq!(run(&mut h, 3000, 1, sole(0, 10, 50)), Mtp);
    assert_eq!(run(&mut h, 5000, 1, sole(0, 10, 80)), Dflash);
    assert_eq!(run(&mut h, 5100, 9, None), Plain);
    assert_eq!(h.counters.steps, [1, 3, 2]);
    let s = |f: SpecStepMode, t: SpecStepMode| h.counters.switches[f as usize][t as usize];
    assert_eq!(s(Mtp, Dflash), 2);
    assert_eq!(s(Dflash, Mtp), 1);
    assert_eq!(s(Dflash, Plain), 1);
    assert_eq!(s(Mtp, Plain), 0);
    assert_eq!(h.last(), Some(Plain));
    let line = h.counters.summary();
    assert!(line.contains("steps PLAIN=1 MTP=3 DFLASH=2"), "{line}");
    assert!(line.contains("MTP->DFLASH=2"), "{line}");
    assert!(line.contains("DFLASH->MTP=1"), "{line}");
    assert!(!line.contains("MTP->PLAIN"), "{line}");
}

#[test]
fn hybrid_state_counts_sole_sequence_off_slot0() {
    use SpecStepMode::*;
    let t0 = Instant::now();
    let mut h = HybridState::new(cfg());
    let mut run = |active: usize, s: Option<SoleSeq>, ok: bool, executed: SpecStepMode| {
        let m = h.decide(active, 8, s, ok, t0);
        h.commit(m);
        h.count_step(executed);
    };
    // Alone on slot 3: no DFlash state, so it runs MTP; counted.
    run(1, sole(3, 10, 1), false, Mtp);
    run(1, sole(3, 10, 2), false, Mtp);
    // Alone on slot 3 but the gate took plain decode: not an MTP step.
    run(1, sole(3, 10, 3), false, Plain);
    // Alone on slot 0 (dwell not met yet): MTP, but not deviation 2.
    run(1, sole(0, 10, 1), true, Mtp);
    // Two active: `sole` is ignored.
    run(2, sole(3, 10, 4), false, Mtp);
    assert_eq!(h.counters.sole_off_slot0_mtp, 2);
    assert!(
        h.counters
            .summary()
            .ends_with("; sole off slot 0 MTP=2; rate demotions=0"),
        "{}",
        h.counters.summary()
    );
}

#[test]
fn hybrid_state_commit_reports_changes_only() {
    let mut h = HybridState::new(cfg());
    assert!(
        !h.commit(SpecStepMode::Mtp),
        "the first mode is not a switch"
    );
    assert!(!h.commit(SpecStepMode::Mtp));
    assert!(h.commit(SpecStepMode::Dflash));
    assert!(!h.commit(SpecStepMode::Dflash));
    assert_eq!(
        h.counters.summary(),
        "steps PLAIN=0 MTP=0 DFLASH=0; switches MTP->DFLASH=1; sole off slot 0 MTP=0; rate demotions=0"
    );
    assert_eq!(
        ModeCounters::default().summary(),
        "steps PLAIN=0 MTP=0 DFLASH=0; switches none; sole off slot 0 MTP=0; rate demotions=0"
    );
}

#[test]
fn hybrid_state_draft_counts() {
    let h = HybridState::new(cfg());
    assert_eq!(h.num_drafts(SpecStepMode::Dflash), 7);
    assert_eq!(h.num_drafts(SpecStepMode::Mtp), 3);
    assert_eq!(h.num_drafts(SpecStepMode::Plain), 3);
}

#[test]
fn spec_mode_word_round_trips() {
    for m in [SpecStepMode::Plain, SpecStepMode::Mtp, SpecStepMode::Dflash] {
        let w = hy::encode_spec_mode(m);
        assert_eq!(hy::decode_spec_mode(w).unwrap(), m);
        assert_eq!(SpecStepMode::from_u8(m as u8), Some(m));
    }
    assert_eq!(SpecStepMode::from_u8(3), None);
}

#[test]
fn spec_mode_word_rejects_foreign_words() {
    // Token ids, other EP command codes and corrupted mode words.
    for w in [
        0u32,
        1,
        2,
        151_329,
        0xFFFF_FFF5,
        EP_CMD_SPEC_MODE,
        0xFFFF_FFFF,
        hy::encode_spec_mode(SpecStepMode::Mtp) ^ 0x0001_0000,
        hy::encode_spec_mode(SpecStepMode::Mtp) | 0x0100,
        hy::encode_spec_mode(SpecStepMode::Plain) | 0x03,
    ] {
        assert!(hy::decode_spec_mode(w).is_err(), "{w:#010x} decoded");
    }
}

#[test]
fn spec_mode_command_is_distinct() {
    // The worker dispatches on the command word before the slot lookup; it must not collide
    // with the other EP commands.
    let others = [
        0xFFFF_FFE0u32,
        0xFFFF_FFF0,
        0xFFFF_FFF1,
        0xFFFF_FFF2,
        0xFFFF_FFF3,
        0xFFFF_FFF4,
        metrale_model_layers::speculative::EP_CMD_MTP_PROPOSE,
        metrale_model_layers::speculative::glm_dflash::EP_CMD_VERIFY_KGAMMA,
        0xFFFF_FFF7,
        0xFFFF_FFF8,
        0xFFFF_FFF9,
        0xFFFF_FFFA,
        0xFFFF_FFFF,
    ];
    assert!(!others.contains(&EP_CMD_SPEC_MODE));
}

#[test]
fn dwell_lever_parse() {
    assert_eq!(hy::parse_dwell_ms(None), HYBRID_DWELL_MS_DEFAULT);
    assert_eq!(HYBRID_DWELL_MS_DEFAULT, 2000);
    assert_eq!(hy::parse_dwell_ms(Some("0")), 0);
    assert_eq!(hy::parse_dwell_ms(Some(" 750 ")), 750);
    assert_eq!(hy::parse_dwell_ms(Some("-1")), HYBRID_DWELL_MS_DEFAULT);
    assert_eq!(hy::parse_dwell_ms(Some("2s")), HYBRID_DWELL_MS_DEFAULT);
}

#[test]
fn drafter_window_caps_msl() {
    assert_eq!(hy::hybrid_drafter_window(540_672, 16_384), 16_384);
    assert_eq!(hy::hybrid_drafter_window(8_192, 16_384), 8_192);
    assert_eq!(hy::hybrid_drafter_window(540_672, 0), 540_672);
}

#[test]
fn hybrid_slot_widths() {
    use metrale_model_layers::ssm_reserve as sr;
    // Only the DFlash slot widens; it never narrows below its MTP capacity.
    assert_eq!(
        sr::hybrid_slot_h_intermediates_with(sr::HYBRID_DFLASH_SLOT, 3, 8),
        8
    );
    assert_eq!(
        sr::hybrid_slot_h_intermediates_with(sr::HYBRID_DFLASH_SLOT, 3, 2),
        3
    );
    assert_eq!(sr::hybrid_slot_h_intermediates_with(1, 3, 8), 3);
    assert_eq!(sr::hybrid_slot_h_intermediates_with(7, 1, 8), 1);
}

#[test]
fn hybrid_reserve_adds_one_wide_slot() {
    use metrale_model_layers::ssm_reserve as sr;
    let (h, c, mb, slots) = (1000usize, 10usize, 4usize, 4usize);
    let cap = |_slot: usize| 3usize;
    // Pure MTP at 3 drafts on every slot, with checkpoints.
    let mtp = sr::ssm_pool_reserve_bytes_with(
        mb,
        h,
        c,
        true,
        3,
        slots,
        true,
        false,
        sr::SsmRollbackMode::Snapshot,
        false,
    );
    let hybrid = sr::ssm_pool_reserve_bytes_hybrid_with(mb, h, c, 3, 8, slots, false, false, cap);
    // Slot 0 holds 8 h intermediates instead of 3; every slot's conv side is 9 rows wide.
    let want = mtp + 5 * h + slots * (9 - 4) * c;
    assert_eq!(hybrid, want);
    // Without checkpoints each verify slot drops its checkpoint blob.
    let no_ckpt = sr::ssm_pool_reserve_bytes_hybrid_with(mb, h, c, 3, 8, slots, false, true, cap);
    assert_eq!(hybrid - no_ckpt, slots * (h + c));
}

#[test]
fn step_flags_hybrid_off_is_the_serve_wide_pair() {
    // Every serve without a hybrid state keeps today's flag and draft count, whatever mode.
    for raw in [false, true] {
        for nd in [1usize, 3, 15] {
            for mode in [None, Some(SpecStepMode::Dflash), Some(SpecStepMode::Mtp)] {
                assert_eq!(step_flags(None, mode, raw, nd), (raw, nd));
            }
        }
    }
    let h = HybridState::new(cfg());
    assert_eq!(
        step_flags(Some(&h), Some(SpecStepMode::Dflash), false, 3),
        (true, 7)
    );
    assert_eq!(
        step_flags(Some(&h), Some(SpecStepMode::Mtp), true, 7),
        (false, 3)
    );
    assert_eq!(
        step_flags(Some(&h), Some(SpecStepMode::Plain), true, 7),
        (false, 3)
    );
}

#[test]
fn hybrid_ctx_state_reserve_counts_every_tap() {
    // Review F1: the accumulator holds `taps` hidden rows per window row, not one.
    assert_eq!(
        hy::hybrid_ctx_state_bytes(5, 2057, 4096),
        5 * 2057 * 4096 * 2
    );
    assert_eq!(
        hy::hybrid_ctx_state_bytes(1, 16_384, 4096),
        16_384 * 4096 * 2
    );
}
