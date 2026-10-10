// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Tests for `binstat`: lever parsing, sampling, percentile helper, line format and
//! the kernel/host slot agreement.

use super::*;

#[test]
fn lever_parses_like_the_other_dsa_switches() {
    assert!(!parse_binstat(None));
    assert!(!parse_binstat(Some("")));
    assert!(!parse_binstat(Some("0")));
    assert!(!parse_binstat(Some("true")));
    assert!(parse_binstat(Some("1")));
    assert!(parse_binstat(Some(" 1 ")));
}

#[test]
fn every_parses_positive_integers_else_default() {
    assert_eq!(parse_binstat_every(None), 64);
    assert_eq!(parse_binstat_every(Some("")), 64);
    assert_eq!(parse_binstat_every(Some("0")), 64);
    assert_eq!(parse_binstat_every(Some("-3")), 64);
    assert_eq!(parse_binstat_every(Some("abc")), 64);
    assert_eq!(parse_binstat_every(Some("1")), 1);
    assert_eq!(parse_binstat_every(Some(" 8 ")), 8);
}

#[test]
fn sampling_hits_the_first_call_then_every_nth() {
    let hits: Vec<u64> = (0..200).filter(|&n| is_sampled(n, 64)).collect();
    assert_eq!(hits, vec![0, 64, 128, 192]);
    assert!((0..5).all(|n| is_sampled(n, 1)));
    assert!(is_sampled(7, 0));
}

#[test]
fn percentile_is_nearest_rank() {
    assert_eq!(percentile_sorted(&[], 50.0), 0);
    assert_eq!(percentile_sorted(&[7], 50.0), 7);
    assert_eq!(percentile_sorted(&[7], 95.0), 7);
    let v: Vec<u32> = (1..=100).collect();
    assert_eq!(percentile_sorted(&v, 50.0), 50);
    assert_eq!(percentile_sorted(&v, 95.0), 95);
    assert_eq!(percentile_sorted(&v, 100.0), 100);
    assert_eq!(percentile_sorted(&v, 0.0), 1);
    let w = [10, 20, 30, 40];
    assert_eq!(percentile_sorted(&w, 50.0), 20);
    assert_eq!(percentile_sorted(&w, 95.0), 40);
}

#[test]
fn summary_and_line_format() {
    let mut rows: Vec<(u32, u32)> = (0..128).map(|r| (100 + r, 32768)).collect();
    rows[0].0 = 5000;
    rows.push((9, 0)); // select_k 0 row: skipped
    let s = summarize(&rows).unwrap();
    assert_eq!((s.p, s.rows, s.max, s.over_cap), (32768, 128, 5000, 1));
    assert_eq!(s.median, 164);
    assert_eq!(
        format_line(&s),
        format!(
            "DSA BINSTAT P=32768 rows=128 b0_median=164 ({:.3}% of P) b0_p95={} b0_max=5000 over2048=1",
            100.0 * 164.0 / 32768.0,
            s.p95
        )
    );
    assert!(summarize(&[(1, 0)]).is_none());
    assert!(summarize(&[]).is_none());
}

#[test]
fn slots_match_the_kernel_source() {
    let src = include_str!("../../../../../kernels/gb10/common/dsa_indexer.cu");
    assert!(src.contains(&format!("#define DSA_RS_B0CNT  {}u", RS_B0CNT)));
    assert!(src.contains(&format!("#define DSA_RS_BINP   {}u", RS_BINP)));
    assert!(src.contains(&format!("#define DSA_COMPACT_CAP {}u", BINSTAT_CAP)));
    assert!(RS_BINP < RADIX_STATE_WORDS);
}
