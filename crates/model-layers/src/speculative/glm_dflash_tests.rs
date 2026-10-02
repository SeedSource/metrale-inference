// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Tests for `glm_dflash`: the lever predicate, the build gate, the tap flags, the
//! worker keep-length against the literal K=3/K=4 worker tables, and the acceptance stats.
//!
//! Owner: model-layers (speculative).
//! Invariants: none beyond the types.
use super::*;

#[test]
fn switch_accepts_only_one() {
    assert!(parse_switch(Some("1")));
    for v in [None, Some("0"), Some("true"), Some(""), Some("01"), Some(" 1")] {
        assert!(!parse_switch(v), "{v:?}");
    }
}

#[test]
fn gate_refuses_glm_drafter_without_lever_only() {
    assert!(glm_dflash_gate("glm5_next", true, false).is_err());
    assert!(glm_dflash_gate("glm5_next_text", true, false).is_err());
    assert!(glm_dflash_gate("glm5_next", true, true).is_ok());
    assert!(glm_dflash_gate("glm5_next", false, false).is_ok());
    assert!(glm_dflash_gate("qwen3_6_moe", true, false).is_ok());
}

#[test]
fn command_is_the_reserved_word_and_distinct() {
    assert_eq!(EP_CMD_VERIFY_KGAMMA, 0xFFFF_FFF6);
    for other in [
        0xFFFF_FFF0u32,
        0xFFFF_FFF1,
        0xFFFF_FFF2,
        0xFFFF_FFF3,
        0xFFFF_FFF4,
        super::super::EP_CMD_MTP_PROPOSE,
        0xFFFF_FFF8,
        0xFFFF_FFE0,
        0xFFFF_FFFF,
    ] {
        assert_ne!(EP_CMD_VERIFY_KGAMMA, other);
    }
}

#[test]
fn tap_flags_for_the_g_drafter() {
    // canada-quant/GLM-5.3-Flash-DFlash2-G config.json, 45-layer text stack.
    let taps = [5usize, 9, 14, 19, 24, 28, 33, 38, 42];
    let f = glm_dflash_tap_flags(&taps, 45).unwrap();
    assert_eq!(f.len(), 45);
    assert_eq!(f.iter().filter(|&&b| b).count(), 9);
    for (i, &b) in f.iter().enumerate() {
        assert_eq!(b, taps.contains(&i), "layer {i}");
    }
    // The last text layer (44) is not a tap; it collapses anyway (`is_last`).
    assert!(!f[44]);
}

#[test]
fn tap_flags_refuse_out_of_range_and_duplicates() {
    assert!(glm_dflash_tap_flags(&[45], 45).is_err());
    assert!(glm_dflash_tap_flags(&[5, 5], 45).is_err());
    assert_eq!(glm_dflash_tap_flags(&[], 45).unwrap(), vec![false; 45]);
}

#[test]
fn keep_len_matches_the_k3_and_k4_worker_tables() {
    // impl_a2.rs worker arms: after a K-row verify the worker pops `k - 1 - num_accepted`
    // rows. K=3: na=2 pops 0, na=1 pops 1, na=0 pops 2. K=4: na=3..0 pops 0..3.
    let base = 100usize;
    for k in [3usize, 4] {
        for na in 0..k {
            let after = base + k;
            let keep = kgamma_keep_len(after, k, na).unwrap();
            assert_eq!(after - keep, k - 1 - na, "k={k} na={na}");
            assert_eq!(keep, base + na + 1);
        }
    }
}

#[test]
fn keep_len_k8_full_and_zero_accept() {
    assert_eq!(kgamma_keep_len(108, 8, 7).unwrap(), 108);
    assert_eq!(kgamma_keep_len(108, 8, 0).unwrap(), 101);
}

#[test]
fn keep_len_refuses_impossible_payloads() {
    assert!(kgamma_keep_len(108, 8, 8).is_err(), "bonus counted as a draft");
    assert!(kgamma_keep_len(108, 0, 0).is_err());
    assert!(kgamma_keep_len(7, 8, 0).is_err());
    assert!(kgamma_keep_len(1000, KGAMMA_MAX_K + 1, 0).is_err());
}

#[test]
fn accept_stats_accumulate() {
    let mut s = AcceptStats::default();
    assert_eq!(s.mean_len(), 0.0);
    s.record(7, 3);
    s.record(7, 7);
    s.record(7, 0);
    s.record(7, 9); // clamped to 7
    assert_eq!(s.steps, 4);
    assert_eq!(s.drafted, 28);
    assert_eq!(s.accepted, 17);
    assert_eq!(s.hist, vec![1, 0, 0, 1, 0, 0, 0, 2]);
    assert!((s.mean_accepted() - 4.25).abs() < 1e-12);
    assert!((s.mean_len() - 5.25).abs() < 1e-12);
    let line = s.summary();
    assert!(line.starts_with("steps=4 mean_accepted=4.250 mean_len=5.250 rate=0.607"), "{line}");
}
