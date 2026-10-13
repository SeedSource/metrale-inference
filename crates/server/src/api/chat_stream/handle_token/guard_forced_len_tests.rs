// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GUARD_HONOR_FORCED_LEN`: a stream-side content guard fire does
//! not end a request that forces its length (`ignore_eos`, or below `min_tokens`).
//!
//! Owner: server streaming API.
//! Invariants: none beyond the tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::super::super::state::StreamState;
use super::apply_guard_trip;

fn fresh() -> (StreamState, Arc<AtomicBool>) {
    let flag = Arc::new(AtomicBool::new(false));
    (
        StreamState::new(false, false, flag.clone(), Vec::new()),
        flag,
    )
}

#[test]
fn lever_on_ignore_eos_suppresses_and_counts() {
    let (mut s, flag) = fresh();
    s.tokens_seen = 100;
    assert!(!apply_guard_trip(&mut s, true, true, 0, true));
    assert_eq!(s.guard_stop, None);
    assert_eq!(s.guard_suppressed, 1);
    assert!(!flag.load(Ordering::Acquire));
    assert!(!s.loop_watchdog_triggered && !s.stop_string_triggered);
    // the token watchdog is covered the same way
    assert!(!apply_guard_trip(&mut s, true, true, 0, false));
    assert_eq!(s.guard_suppressed, 2);
}

#[test]
fn lever_on_min_tokens_floor_then_stop() {
    let (mut s, flag) = fresh();
    s.tokens_seen = 5;
    assert!(!apply_guard_trip(&mut s, true, false, 10, true));
    assert_eq!(s.guard_suppressed, 1);
    assert_eq!(s.guard_stop, None);
    s.tokens_seen = 12;
    assert!(apply_guard_trip(&mut s, true, false, 10, true));
    assert_eq!(s.guard_stop, Some("simhash_semantic_loop"));
    assert!(flag.load(Ordering::Acquire));
    assert_eq!(s.guard_suppressed, 1);
}

#[test]
fn lever_on_without_forced_len_stops_as_today() {
    let (mut s, flag) = fresh();
    s.tokens_seen = 3;
    assert!(apply_guard_trip(&mut s, true, false, 0, false));
    assert_eq!(s.guard_stop, Some("token_loop_watchdog"));
    assert!(flag.load(Ordering::Acquire));
    assert_eq!(s.guard_suppressed, 0);
    assert!(s.loop_watchdog_triggered && s.stop_string_triggered);
}

#[test]
fn lever_off_ignore_eos_and_min_tokens_stop_as_today() {
    let (mut s, flag) = fresh();
    s.tokens_seen = 3;
    assert!(apply_guard_trip(&mut s, false, true, 1000, true));
    assert_eq!(s.guard_stop, Some("simhash_semantic_loop"));
    assert!(flag.load(Ordering::Acquire));
    assert_eq!(s.guard_suppressed, 0);
}
