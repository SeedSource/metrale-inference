// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests for `hybrid_ring.rs` (#82 hybrid B2(i)): the ring's label, contiguity and
//! slot bookkeeping, the catch-up key ranges, and the lever parsers in `hybrid.rs`. CPU only.

use super::*;
use crate::speculative::hybrid::{hybrid_mtp_catchup_bytes, parse_catchup_rows};

/// 2026-10-14: One hybrid DFlash step on rank 0 or the worker: the verify of `k` rows at
/// pre-verify length `pre` stages labels `pre + 1 ..`, then the commit keeps `na + 1` rows.
fn dflash_step(m: &mut RingMeta, slot: usize, pre: usize, k: usize, na: usize, rows: usize) {
    m.stage(slot, pre + 1, k, rows);
    assert!(m.commit(slot, pre + na + 1, na + 1, k));
}

#[test]
fn split_wrap_splits_at_the_ring_end_only() {
    assert_eq!(
        split_wrap(10, 4, 64),
        vec![Piece {
            offset: 0,
            ring_row: 10,
            n: 4
        }]
    );
    assert_eq!(
        split_wrap(126, 5, 64),
        vec![
            Piece {
                offset: 0,
                ring_row: 62,
                n: 2
            },
            Piece {
                offset: 2,
                ring_row: 0,
                n: 3
            },
        ]
    );
    assert!(split_wrap(5, 0, 64).is_empty());
    assert!(split_wrap(5, 3, 0).is_empty());
}

#[test]
fn committed_verifies_extend_the_coverage_under_the_k2_label_rule() {
    // 2026-10-14: K2 verify rule (`verify_k2_step.rs`): row t of a verify at pre-verify length
    // L is labelled L + t + 1. Two DFlash steps at pre 100 (k 8, 3 accepted) and 104 (k 8, 0).
    let mut m = RingMeta::default();
    let pieces = m.stage(0, 101, 8, 64);
    assert_eq!(
        pieces,
        vec![Piece {
            offset: 0,
            ring_row: 101 % 64,
            n: 8
        }]
    );
    assert_eq!(m.resume, Some(0));
    assert_eq!(m.count, 0, "staged rows are not covered before the commit");
    assert!(m.commit(0, 104, 4, 8));
    assert_eq!((m.owner, m.start, m.count), (Some(0), 101, 4));
    dflash_step(&mut m, 0, 104, 8, 0, 64);
    assert_eq!((m.start, m.count), (101, 5));
    // 2026-10-14: Labels 101..106 feed pair keys 100..105.
    assert_eq!(m.covered_keys(0), Some((100, 105)));
    assert_eq!(m.covered_keys(1), None);
}

#[test]
fn a_label_gap_or_another_slot_restarts_the_coverage() {
    let mut m = RingMeta::default();
    dflash_step(&mut m, 0, 100, 4, 3, 64);
    assert_eq!((m.start, m.count), (101, 4));
    // 2026-10-14: Plain or MTP steps committed 105 and 106 off the ring: next verify at 107.
    dflash_step(&mut m, 0, 107, 4, 1, 64);
    assert_eq!((m.start, m.count), (108, 2));
    dflash_step(&mut m, 2, 300, 4, 0, 64);
    assert_eq!((m.owner, m.start, m.count), (Some(2), 301, 1));
}

#[test]
fn a_full_ring_drops_the_labels_the_write_overwrites() {
    let rows = 64;
    let mut m = RingMeta::default();
    let mut pre = 10;
    for _ in 0..30 {
        dflash_step(&mut m, 0, pre, 8, 3, rows);
        pre += 4;
    }
    assert!(m.count <= rows);
    assert_eq!(
        m.start + m.count,
        pre + 1,
        "coverage ends at the last committed label"
    );
    // 2026-10-14: The staged write of the next verify overwrites the rows of the 8 oldest-but-
    // one labels; they leave the coverage before the copy, committed or not.
    let before_end = m.start + m.count;
    m.stage(0, pre + 1, 8, rows);
    assert_eq!(m.start + m.count, before_end);
    assert!(m.start >= pre + 1 + 8 - rows);
    assert!(m.count <= rows - 8);
    // 2026-10-14: Never committed (rank 0 finished the sequence in its emit loop): the next
    // commit of any other shape drops it, and the coverage keeps no overwritten label.
    assert!(!m.commit(0, pre + 2, 1, 4));
    assert!(m.start >= pre + 1 + 8 - rows);
}

#[test]
fn a_commit_that_does_not_match_the_staged_write_changes_nothing_but_the_stage() {
    let mut m = RingMeta::default();
    dflash_step(&mut m, 0, 100, 4, 3, 64);
    let covered = (m.start, m.count);
    m.stage(0, 105, 4, 64);
    // 2026-10-14: Wrong slot, wrong width, wrong length: each drops the stage.
    assert!(!m.commit(1, 106, 1, 4));
    assert_eq!(m.staged, None);
    m.stage(0, 105, 4, 64);
    assert!(!m.commit(0, 106, 1, 8));
    m.stage(0, 105, 4, 64);
    assert!(!m.commit(0, 107, 1, 4));
    assert_eq!((m.start, m.count), covered);
    assert!(!m.commit(0, 106, 1, 4), "no stage: nothing to commit");
}

#[test]
fn reset_slot_forgets_only_that_slots_rows_and_resume() {
    let mut m = RingMeta::default();
    dflash_step(&mut m, 0, 100, 4, 3, 64);
    m.reset_slot(1);
    assert_eq!(m.covered_keys(0), Some((100, 104)));
    assert_eq!(m.resume, Some(0));
    m.reset_slot(0);
    assert_eq!(m, RingMeta::default());
}

#[test]
fn take_resume_fires_once_per_stretch() {
    let mut m = RingMeta::default();
    assert!(!m.take_resume(0));
    m.note_dflash(0);
    assert!(!m.take_resume(1));
    assert!(m.take_resume(0));
    assert!(!m.take_resume(0));
    dflash_step(&mut m, 0, 100, 4, 3, 64);
    assert!(m.take_resume(0));
}

#[test]
fn the_switch_plan_rewinds_the_untrimmed_mtp_drafts_and_feeds_to_position_minus_two() {
    // 2026-10-14: Last MTP propose at position 100 wrote key 99 and 2 drafts (rows 100, 101,
    // never trimmed); a DFlash bootstrap decoded position 100; DFlash verifies committed
    // labels 102..=150 (keys 101..=149), the last one leaving its bonus at position 150. The
    // first MTP step after the switch bootstraps: it decodes the bonus (hidden_150) and proposes
    // at 151, which writes key 150; the drafter must hold keys 0..150 first.
    let mut m = RingMeta::default();
    dflash_step(&mut m, 0, 101, 16, 15, 2048);
    dflash_step(&mut m, 0, 117, 16, 15, 2048);
    dflash_step(&mut m, 0, 133, 16, 15, 2048);
    dflash_step(&mut m, 0, 149, 16, 0, 2048);
    assert_eq!(m.covered_keys(0), Some((101, 150)));
    let p = plan_catchup(102, 151, m.covered_keys(0), true);
    assert_eq!(p.missing, 48);
    assert_eq!(p.rewind_to, Some(101));
    assert_eq!(p.head_fill, None);
    assert_eq!(p.feed, Some((101, 150)));
    assert_eq!(p.tail_fill, None);
    assert_eq!(p.skip, None);
    // 2026-10-14: One more plain step before the first propose (propose at 152, writing key
    // 151): key 150 is not in the ring, so it is a one-row zero-filled tail.
    let p = plan_catchup(102, 152, m.covered_keys(0), true);
    assert_eq!(p.feed, Some((101, 150)));
    assert_eq!(p.tail_fill, Some((150, 151)));
    // 2026-10-14: Without rewind support the feed starts at the drafter's rows.
    let p = plan_catchup(102, 151, m.covered_keys(0), false);
    assert_eq!((p.rewind_to, p.feed), (None, Some((102, 150))));
}

#[test]
fn a_stretch_longer_than_the_ring_is_zero_filled_up_to_the_ring_start() {
    let p = plan_catchup(1000, 6001, Some((3952, 6000)), true);
    assert_eq!(p.missing, 5000);
    assert_eq!(p.rewind_to, None);
    assert_eq!(p.head_fill, Some((1000, 3952)));
    assert_eq!(p.feed, Some((3952, 6000)));
    assert_eq!(p.fill_rows(), 2952);
    let p = plan_catchup(1000, 30_001, Some((27_952, 30_000)), true);
    assert_eq!(p.skip, Some(Skip::FillCap));
    assert_eq!((p.feed, p.head_fill), (None, None));
}

#[test]
fn no_drafter_prefill_no_coverage_and_current_drafters_write_nothing() {
    let p = plan_catchup(0, 500, Some((100, 499)), true);
    assert_eq!((p.skip, p.missing), (Some(Skip::NoDrafterPrefill), 499));
    assert_eq!(
        plan_catchup(80, 500, None, true).skip,
        Some(Skip::NoCoverage)
    );
    // 2026-10-14: Coverage only at or past the propose's own key.
    assert_eq!(
        plan_catchup(80, 500, Some((499, 520)), true).skip,
        Some(Skip::NoCoverage)
    );
    let p = plan_catchup(499, 500, Some((400, 499)), false);
    assert_eq!((p.skip, p.missing), (Some(Skip::Current), 0));
}

#[test]
fn the_plan_ends_with_position_minus_one_rows_in_order() {
    // 2026-10-14: Property over a grid: applying the plan's steps leaves `position - 1` rows
    // unless it skipped, and the steps touch increasing keys with no hole.
    for rows in [1usize, 50, 99, 100, 120, 160, 199] {
        for (lo, hi) in [(100usize, 150usize), (100, 200), (60, 90), (150, 199)] {
            for can_rewind in [false, true] {
                let pos = 200;
                let p = plan_catchup(rows, pos, Some((lo, hi)), can_rewind);
                if p.skip.is_some() {
                    continue;
                }
                let mut cur = p.rewind_to.unwrap_or(rows);
                for r in [p.head_fill, p.feed, p.tail_fill].into_iter().flatten() {
                    assert_eq!(r.0, cur, "rows={rows} cov=({lo},{hi}) {p:?}");
                    assert!(r.1 > r.0);
                    cur = r.1;
                }
                assert_eq!(cur, pos - 1, "rows={rows} {p:?}");
            }
        }
    }
}

#[test]
fn catchup_rows_parser_and_reserve() {
    assert_eq!(parse_catchup_rows(None), DEFAULT_RING_ROWS);
    assert_eq!(parse_catchup_rows(Some("4096")), 4096);
    assert_eq!(parse_catchup_rows(Some(" 64 ")), 64);
    assert_eq!(parse_catchup_rows(Some("63")), DEFAULT_RING_ROWS);
    assert_eq!(parse_catchup_rows(Some("x")), DEFAULT_RING_ROWS);
    assert_eq!(
        parse_catchup_rows(Some(&(MAX_RING_ROWS + 1).to_string())),
        DEFAULT_RING_ROWS
    );
    assert_eq!(hybrid_mtp_catchup_bytes(false, 2048, 6144), 0);
    assert_eq!(
        hybrid_mtp_catchup_bytes(true, 2048, 6144),
        (2048 + ZERO_ROWS) * 6144 * 2
    );
}
