// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Host tests of the deferred-commit bookkeeping (`kda_defer.rs`).
//!
//! Owner: model-engine (SSM state).
//! Invariants: none beyond the types.

use super::DeferBook;

#[test]
fn a_new_book_has_nothing_pending_and_the_device_words_at_minus_one() {
    let mut b = DeferBook::new(4);
    for s in 0..4 {
        assert_eq!(b.pending(s), None);
        assert_eq!(b.sync_dev(s), None, "slot {s}: device word starts at -1");
    }
    assert!(!b.any_pending());
    assert!(
        !b.covers(4),
        "the MTP dummy and uncovered slots never defer"
    );
    assert_eq!(b.pending(9), None);
}

#[test]
fn full_accept_sets_minus_one() {
    let mut b = DeferBook::new(2);
    b.defer(1, 1);
    assert_eq!(b.pending(1), Some(1));
    b.clear(1);
    assert_eq!(b.pending(1), None);
    assert!(!b.any_pending());
}

#[test]
fn materialise_returns_the_index_once_and_clears_it() {
    let mut b = DeferBook::new(3);
    b.defer(0, 0);
    b.defer(2, 1);
    assert_eq!(b.take_pending(0), Some(0));
    assert_eq!(
        b.take_pending(0),
        None,
        "a second materialise copies nothing"
    );
    assert_eq!(b.take_all_except(None), vec![(2, 1)]);
    assert!(!b.any_pending());
}

#[test]
fn a_consumer_verify_keeps_its_own_slot_and_flushes_the_rest() {
    let mut b = DeferBook::new(3);
    b.defer(0, 1);
    b.defer(1, 0);
    assert_eq!(b.take_all_except(Some(1)), vec![(0, 1)]);
    assert_eq!(b.pending(1), Some(0));
}

#[test]
fn the_device_word_follows_pending_before_each_verify() {
    let mut b = DeferBook::new(2);
    b.defer(0, 1);
    assert_eq!(b.sync_dev(0), Some(1), "write the deferred index");
    assert_eq!(b.sync_dev(0), None, "already there");
    // 2026-10-09: The verify consumed it; the word still holds 1 until the next sync.
    b.clear(0);
    b.defer(0, 1);
    assert_eq!(b.sync_dev(0), None, "same index again: no write");
    b.clear(0);
    assert_eq!(b.sync_dev(0), Some(-1), "back to the live state");
}

#[test]
fn slot_reuse_resets_pending_arm_and_device_word() {
    let mut b = DeferBook::new(2);
    b.arm(1);
    b.defer(1, 2);
    assert_eq!(b.sync_dev(1), Some(2));
    // 2026-10-09: The sequence is freed with a deferral pending and the slot is reallocated.
    assert_eq!(
        b.reset_slot(1),
        Some(-1),
        "the device word must go back to -1"
    );
    assert_eq!(b.pending(1), None);
    assert!(!b.take_armed(1));
    assert_eq!(b.reset_slot(1), None, "idempotent");
    assert_eq!(b.reset_slot(7), None, "an uncovered slot has no word");
}

#[test]
fn only_a_commit_after_a_consumer_verify_may_defer() {
    let mut b = DeferBook::new(2);
    assert!(!b.take_armed(0), "no consumer verify yet");
    b.arm(0);
    assert!(b.take_armed(0));
    assert!(!b.take_armed(0), "one commit per arm");
    b.arm(5);
    assert!(!b.take_armed(5), "uncovered slots never arm");
}

#[test]
fn a_non_consumer_verify_after_a_consumer_disarms() {
    let mut b = DeferBook::new(2);
    b.arm(0);
    b.disarm(0);
    assert!(
        !b.take_armed(0),
        "an eager or batched verify ran last: copy"
    );
    b.disarm(9);
    b.arm(1);
    b.disarm_all();
    assert!(!b.take_armed(1));
}
