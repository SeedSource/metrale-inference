// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The issue order of the staged prefill's overlapped all-reduces
//! (`METRALE_GLM_PREFILL_COMM_OVERLAP=1`, `prefill_comm_overlap`).
//!
//! Each item (an attention sub-chunk, or an FFN window) has a front, everything up to and
//! including its partial sum, ending with a deferred all-reduce of that partial
//! (`CommBackend::all_reduce_deferred`), and a back: the join (`all_reduce_join`) and the
//! `hc_post` that folds the reduced partial into the highway. `overlap_schedule` issues item
//! `i + 1`'s front before item `i`'s back, so item `i`'s all-reduce runs on the comm stream while
//! item `i + 1` computes.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants (checked by the tests below):
//! - Every item gets exactly one `Issue` and, after it, exactly one `Finish`, under the same
//!   slot.
//! - Fronts run in item order and backs run in item order, so every all-reduce is issued in the
//!   order the unoverlapped loop issues it (the collective sequence is unchanged).
//! - A slot is never reused while the all-reduce issued under it is outstanding, and at most
//!   `ALL_REDUCE_DEFERRED_SLOTS` (2) all-reduces are outstanding at once.
//! - The schedule ends with every item finished, so nothing is outstanding when the pass
//!   returns.

use metrale_comm::ALL_REDUCE_DEFERRED_SLOTS;

/// 2026-10-01: One step of the overlapped pass over `n` items.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlapStep {
    /// 2026-10-01: Run item `item`'s front and issue its deferred all-reduce under `slot`.
    Issue { item: usize, slot: usize },
    /// 2026-10-01: Join `slot` and run item `item`'s back.
    Finish { item: usize, slot: usize },
}

/// 2026-10-01: The slot item `item` uses: items alternate between the two slots.
#[must_use]
pub fn slot_of(item: usize) -> usize {
    item % ALL_REDUCE_DEFERRED_SLOTS
}

/// 2026-10-01: `Issue(0), Issue(1), Finish(0), Issue(2), Finish(1), ..., Issue(n-1),
/// Finish(n-2), Finish(n-1)`: one item of lookahead. Empty for `n == 0`; `Issue(0), Finish(0)`
/// for `n == 1`.
#[must_use]
pub fn overlap_schedule(n: usize) -> Vec<OverlapStep> {
    let mut out = Vec::with_capacity(2 * n);
    for item in 0..n {
        out.push(OverlapStep::Issue {
            item,
            slot: slot_of(item),
        });
        if item >= 1 {
            out.push(OverlapStep::Finish {
                item: item - 1,
                slot: slot_of(item - 1),
            });
        }
    }
    if n >= 1 {
        out.push(OverlapStep::Finish {
            item: n - 1,
            slot: slot_of(n - 1),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-01: Replays the schedule against a slot table and checks every module invariant.
    fn check(n: usize) {
        let steps = overlap_schedule(n);
        assert_eq!(steps.len(), 2 * n, "n={n}: one Issue and one Finish per item");
        let mut busy: [Option<usize>; ALL_REDUCE_DEFERRED_SLOTS] =
            [None; ALL_REDUCE_DEFERRED_SLOTS];
        let mut next_issue = 0usize;
        let mut next_finish = 0usize;
        for (i, s) in steps.iter().enumerate() {
            match *s {
                OverlapStep::Issue { item, slot } => {
                    assert_eq!(item, next_issue, "n={n} step {i}: fronts in item order");
                    assert!(slot < ALL_REDUCE_DEFERRED_SLOTS);
                    assert_eq!(busy[slot], None, "n={n} step {i}: slot {slot} reused while busy");
                    busy[slot] = Some(item);
                    next_issue += 1;
                }
                OverlapStep::Finish { item, slot } => {
                    assert_eq!(item, next_finish, "n={n} step {i}: backs in item order");
                    assert_eq!(busy[slot], Some(item), "n={n} step {i}: join of the wrong slot");
                    busy[slot] = None;
                    next_finish += 1;
                }
            }
            let outstanding = busy.iter().filter(|b| b.is_some()).count();
            assert!(outstanding <= ALL_REDUCE_DEFERRED_SLOTS);
        }
        assert_eq!(next_issue, n);
        assert_eq!(next_finish, n);
        assert!(busy.iter().all(Option::is_none), "n={n}: nothing outstanding at the end");
    }

    #[test]
    fn schedule_invariants_hold() {
        // 2026-10-01: 32 = 8192 tokens at 256 rows; 2 = 8192 tokens in 4096-row FFN windows.
        for n in [0, 1, 2, 3, 4, 5, 16, 22, 32, 33, 64] {
            check(n);
        }
    }

    #[test]
    fn small_schedules_spelled_out() {
        use OverlapStep::{Finish, Issue};
        assert!(overlap_schedule(0).is_empty());
        assert_eq!(
            overlap_schedule(1),
            vec![Issue { item: 0, slot: 0 }, Finish { item: 0, slot: 0 }]
        );
        assert_eq!(
            overlap_schedule(3),
            vec![
                Issue { item: 0, slot: 0 },
                Issue { item: 1, slot: 1 },
                Finish { item: 0, slot: 0 },
                Issue { item: 2, slot: 0 },
                Finish { item: 1, slot: 1 },
                Finish { item: 2, slot: 0 },
            ]
        );
    }

    /// 2026-10-01: Each item's front comes before the back of the item before it (the overlap
    /// itself), except item 0, which has no predecessor.
    #[test]
    fn next_front_precedes_previous_back() {
        let steps = overlap_schedule(10);
        let pos = |want: OverlapStep| steps.iter().position(|s| *s == want).unwrap();
        for item in 1..10 {
            let issue = pos(OverlapStep::Issue {
                item,
                slot: slot_of(item),
            });
            let prev_finish = pos(OverlapStep::Finish {
                item: item - 1,
                slot: slot_of(item - 1),
            });
            assert!(issue < prev_finish, "item {item}");
        }
    }
}
