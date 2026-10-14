// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests for `decode_slice` (`METRALE_DECODE_STEPS_PER_PREFILL_CHUNK`).

use super::*;

#[test]
fn resolve_off_values() {
    for raw in [None, Some(""), Some("  "), Some("0"), Some("1"), Some("abc"), Some("-3")] {
        assert_eq!(resolve(raw), None, "{raw:?}");
    }
}

#[test]
fn resolve_on_values() {
    assert_eq!(resolve(Some("2")), Some(2));
    assert_eq!(resolve(Some(" 26 ")), Some(26));
}

#[test]
fn off_never_holds() {
    for left in 0..4 {
        assert!(!hold(None, 3, left));
        assert_eq!(next_left(None, 3, left, false, true), 0);
    }
}

#[test]
fn no_decoder_never_holds() {
    assert!(!hold(Some(8), 0, 5));
    assert_eq!(next_left(Some(8), 0, 5, false, true), 0);
}

/// Simulate ticks: a prefill wants a chunk on every tick it is allowed one; the decoders take one
/// step per tick. With k, each chunk is followed by k - 1 held ticks, so k decode steps per chunk.
fn simulate(lever: Option<usize>, decoders: usize, ticks: usize) -> (usize, usize) {
    let (mut left, mut chunks, mut steps) = (0usize, 0usize, 0usize);
    for _ in 0..ticks {
        let held = hold(lever, decoders, left);
        let prefill_ran = !held;
        if prefill_ran {
            chunks += 1;
        }
        if decoders > 0 {
            steps += 1;
        }
        left = next_left(lever, decoders, left, held, prefill_ran);
    }
    (chunks, steps)
}

#[test]
fn k_steps_per_chunk() {
    assert_eq!(simulate(None, 2, 30), (30, 30));
    assert_eq!(simulate(Some(2), 2, 30), (15, 30));
    assert_eq!(simulate(Some(6), 2, 30), (5, 30));
    // 2026-10-14: no decoder: every tick runs a chunk, lever or not.
    assert_eq!(simulate(Some(6), 0, 30), (30, 0));
}

#[test]
fn hold_is_bounded_and_resets() {
    // A chunk ran: k - 1 held ticks follow, then the hold is used up.
    let mut left = next_left(Some(4), 1, 0, false, true);
    assert_eq!(left, 3);
    for want in [2, 1, 0] {
        assert!(hold(Some(4), 1, left));
        left = next_left(Some(4), 1, left, true, false);
        assert_eq!(left, want);
    }
    assert!(!hold(Some(4), 1, left));
    // The last decoder finishes mid-hold: the hold is dropped at once.
    let left = next_left(Some(4), 1, 0, false, true);
    assert_eq!(next_left(Some(4), 0, left, true, false), 0);
    // A tick with no prefill pending and no hold leaves the counter alone.
    assert_eq!(next_left(Some(4), 2, 0, false, false), 0);
}
