// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The pure `METRALE_GLM_DFLASH_KSEL` selector: cost-table parsing (malformed
//! input refused), interpolation in K, the expected-token sum, the argmax over K, and the
//! fallbacks (no entry for n, nothing costed or allowed).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

fn table(s: &str) -> CostTable {
    parse_cost_table(s).expect("well-formed")
}

#[test]
fn parse_accepts_the_documented_form_and_sorts() {
    let t = table(" 3:4:148.3, 2:4:101.2 ,2:3:90");
    assert_eq!(t.cost(2, 4), Some(101.2));
    assert_eq!(t.cost(2, 3), Some(90.0));
    assert_eq!(t.cost(3, 4), Some(148.3));
    assert!(t.has_n(2) && t.has_n(3) && !t.has_n(4));
}

#[test]
fn parse_refuses_malformed_input() {
    for bad in [
        "",
        "2:4",
        "2:4:101:5",
        "2:4:abc",
        "x:4:10",
        "2:y:10",
        "2:4:101.2,",
        "2:4:0",
        "2:4:-3",
        "2:4:NaN",
        "2:4:inf",
        "1:4:10",
        "2:1:10",
        "2:4:10,2:4:11",
    ] {
        assert!(parse_cost_table(bad).is_err(), "{bad:?} parsed");
        // 2026-10-09: The lever-facing wrapper warns and yields no table.
        assert_eq!(parse_kcost(bad), None, "{bad:?}");
    }
}

#[test]
fn cost_interpolates_linearly_in_k_and_never_extrapolates() {
    let t = table("2:2:80,2:4:100,3:4:150");
    assert_eq!(t.cost(2, 2), Some(80.0));
    assert!((t.cost(2, 3).unwrap() - 90.0).abs() < 1e-9, "midpoint");
    assert_eq!(t.cost(2, 4), Some(100.0));
    // 2026-10-09: Outside the measured K range, or another n: no cost.
    assert_eq!(t.cost(2, 5), None);
    assert_eq!(t.cost(3, 2), None);
    assert_eq!(t.cost(3, 3), None);
    assert_eq!(t.cost(4, 4), None);
    // 2026-10-09: Uneven spacing interpolates by K distance.
    let u = table("2:2:60,2:5:90");
    assert!((u.cost(2, 3).unwrap() - 70.0).abs() < 1e-9);
    assert!((u.cost(2, 4).unwrap() - 80.0).abs() < 1e-9);
}

#[test]
fn expected_tokens_is_one_plus_the_prefix_products() {
    // 2026-10-09: K=2 verifies one draft: 1 + p1.
    assert!((expected_tokens(&[0.5, 0.5, 0.5], 0.6, 2) - 1.5).abs() < 1e-9);
    // 2026-10-09: K=4: 1 + .5 + .25 + .125.
    assert!((expected_tokens(&[0.5, 0.5, 0.5], 0.6, 4) - 1.875).abs() < 1e-9);
    // 2026-10-09: Missing positions take pdef: 1 + .6 + .36 + .216.
    assert!((expected_tokens(&[], 0.6, 4) - 2.176).abs() < 1e-6);
    // 2026-10-09: An out-of-range p is clamped.
    assert!((expected_tokens(&[1.5, -1.0], 0.6, 3) - 2.0).abs() < 1e-9);
}

#[test]
fn select_takes_the_argmax_of_tokens_per_ms() {
    let t = table("2:2:60,2:3:80,2:4:100");
    // 2026-10-09: Confident streams: K=4 wins (2*(1+.9+.81+.729)/100 = .06878 vs K=3
    // 2*2.71/80 = .06775 vs K=2 2*1.9/60 = .06333).
    let hi = vec![Some(vec![0.9, 0.9, 0.9]), Some(vec![0.9, 0.9, 0.9])];
    let s = select_k(&hi, 4, 0.6, &t, |_| true).expect("costed");
    assert_eq!(s.k, 4);
    assert_eq!(
        s.alts.iter().map(|a| a.0).collect::<Vec<_>>(),
        vec![2, 3, 4]
    );
    assert!((s.score - 2.0 * 3.439 / 100.0).abs() < 1e-6);
    // 2026-10-09: Unconfident streams: K=2 wins (2*1.2/60 = .04 vs 2*1.24/80 = .031).
    let lo = vec![Some(vec![0.2, 0.2, 0.2]), Some(vec![0.2, 0.2, 0.2])];
    assert_eq!(select_k(&lo, 4, 0.6, &t, |_| true).unwrap().k, 2);
    // 2026-10-09: Streams without p score on pdef; a mixed batch sums per stream.
    // 2026-10-09: pdef 0.6: K=2 2*1.6/60 = .0533, K=3 2*1.96/80 = .049, K=4 2*2.176/100 = .0435.
    let none = vec![None, None];
    let s = select_k(&none, 4, 0.6, &t, |_| true).unwrap();
    assert_eq!(s.k, 2);
    assert!((s.score - 3.2 / 60.0).abs() < 1e-9);
    // 2026-10-09: One p=1 stream (E = 2, 3, 4) beside a pdef stream: K=2 3.6/60 =
    // .06, K=3 4.96/80 = .062, K=4 6.176/100 = .06176: K=3.
    let mixed = vec![Some(vec![1.0, 1.0, 1.0]), None];
    assert_eq!(select_k(&mixed, 4, 0.6, &t, |_| true).unwrap().k, 3);
}

#[test]
fn select_respects_k_max_allowed_and_breaks_ties_wide() {
    let t = table("2:2:60,2:3:80,2:4:100");
    let hi = vec![Some(vec![0.9, 0.9, 0.9]), Some(vec![0.9, 0.9, 0.9])];
    // 2026-10-09: The rung caps K at 3.
    assert_eq!(select_k(&hi, 3, 0.6, &t, |_| true).unwrap().k, 3);
    // 2026-10-09: A width the model refuses is never chosen.
    let s = select_k(&hi, 4, 0.6, &t, |k| k != 4).unwrap();
    assert_eq!(s.k, 3);
    assert!(s.alts.iter().all(|a| a.0 != 4));
    // 2026-10-09: Equal scores keep the wider K: p=0 gives E=1 at every K, so equal costs tie.
    let flat = table("2:2:50,2:4:50");
    let zero = vec![Some(vec![0.0; 3]), Some(vec![0.0; 3])];
    assert_eq!(select_k(&zero, 4, 0.6, &flat, |_| true).unwrap().k, 4);
}

#[test]
fn select_falls_back_without_an_entry_for_n_or_a_costed_candidate() {
    let t = table("2:4:101.2,2:3:90,3:4:148.3");
    let three = vec![None, None, None];
    // 2026-10-09: n=3 has only K=4: that is the only candidate.
    assert_eq!(select_k(&three, 4, 0.6, &t, |_| true).unwrap().k, 4);
    // 2026-10-09: ...and capped below it nothing is costed: fixed width.
    assert_eq!(select_k(&three, 3, 0.6, &t, |_| true), None);
    // 2026-10-09: No entry for n=4: fixed width.
    assert_eq!(
        select_k(&[None, None, None, None], 4, 0.6, &t, |_| true),
        None
    );
    // 2026-10-09: Nothing allowed: fixed width.
    assert_eq!(select_k(&[None, None], 4, 0.6, &t, |_| false), None);
    // 2026-10-09: A single stream never selects (the C=1 path is not batched).
    assert_eq!(select_k(&[None], 4, 0.6, &t, |_| true), None);
}

#[test]
fn pdef_parses_a_probability_or_keeps_the_default() {
    assert_eq!(parse_pdef(None), PDEF_DEFAULT);
    assert_eq!(parse_pdef(Some(" 0.75 ")), 0.75);
    assert_eq!(parse_pdef(Some("1")), 1.0);
    for bad in ["", "x", "1.5", "-0.1", "NaN"] {
        assert_eq!(parse_pdef(Some(bad)), PDEF_DEFAULT, "{bad:?}");
    }
}
