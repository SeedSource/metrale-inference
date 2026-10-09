// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `METRALE_DFLASH_CONF_LOG` support: the drafter's per-position top-1 probability and
//! top-1 minus top-2 margin, handed from the propose to the verify step's log line.
//! 2026-10-09: `METRALE_DFLASH_CONF_TAU` reads the same p ([`p_for`]) to shorten a C=1 block
//! ([`keep_count`]); the scheduler applies it (`scheduler::dflash_conf_tau`).
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Nothing here feeds a drafted token or a kernel launch. Under `METRALE_DFLASH_CONF_TAU` the
//!   p decides how many drafts are verified, never which tokens are drafted or emitted.
//! - [`p_for`] answers only for the drafts the stored rows were recorded with (token-equal
//!   prefix), so a stale record (another sequence, a batched propose) truncates nothing.
//! - The probabilities are a softmax over the selector's top-16 logits only (the candidate
//!   set `dflash2_topk16` leaves in `scratch.sel_vals`), renormalised over those 16, not
//!   over the full vocabulary. A full-vocabulary top-1 probability is lower or equal.
//! - Recorded only for a single-sequence propose (`n_seq == 1`); a batched propose records
//!   nothing, so the verify line is absent rather than attributed to the wrong sequence.

use parking_lot::Mutex;

/// 2026-10-08: The latest single-sequence propose's per-draft `(p, m)`, in draft order, and
/// (2026-10-09) the drafted tokens they describe.
struct Last {
    rows: Vec<(f32, f32)>,
    tokens: Vec<u32>,
}

static LAST: Mutex<Option<Last>> = Mutex::new(None);

/// 2026-10-08: `(top-1 prob, top-1 minus top-2 prob)` of one row of descending top-16 logits.
pub(super) fn row_conf(vals: &[f32]) -> (f32, f32) {
    let top = vals.first().copied().unwrap_or(0.0);
    let sum: f32 = vals.iter().map(|v| (v - top).exp()).sum();
    let p1 = 1.0 / sum;
    let p2 = vals.get(1).map_or(0.0, |v| (v - top).exp() / sum);
    (p1, p1 - p2)
}

/// 2026-10-08: Store the latest propose's rows and drafted tokens (both in draft order).
pub(super) fn record(rows: Vec<(f32, f32)>, tokens: Vec<u32>) {
    *LAST.lock() = Some(Last { rows, tokens });
}

/// 2026-10-08: Take the latest propose's `(p, m)` per draft; `None` when the lever is off,
/// the propose was batched, or it was already taken.
pub fn take_last() -> Option<Vec<(f32, f32)>> {
    LAST.lock().take().map(|l| l.rows)
}

/// 2026-10-09: The top-1 probability of each of `drafts`, without taking the record (the
/// verify's `DFLASH CONF` line still takes it). `None` unless the latest record's tokens start
/// with `drafts` (a `METRALE_DFLASH_DRAFT_CAP` cap keeps a prefix) and cover them with rows.
pub fn p_for(drafts: &[u32]) -> Option<Vec<f32>> {
    let last = LAST.lock();
    let l = last.as_ref()?;
    (l.tokens.len() >= drafts.len()
        && l.rows.len() >= drafts.len()
        && l.tokens[..drafts.len()] == *drafts)
        .then(|| l.rows[..drafts.len()].iter().map(|r| r.0).collect())
}

/// 2026-10-09: `METRALE_DFLASH_CONF_TAU`: how many of `n` drafts to verify given their top-1
/// probabilities `p`: the drafts before the first `p[i] < tau`, at least 1 (when `n >= 1`);
/// `n` when `tau <= 0` (off) or no `p` falls below it. Drafts past `p.len()` have no p and
/// are kept unless an earlier one stops the block.
pub fn keep_count(p: &[f32], n: usize, tau: f32) -> usize {
    if tau <= 0.0 {
        return n;
    }
    match p.iter().take(n).position(|&pi| pi < tau) {
        Some(i) => i.max(1).min(n),
        None => n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_conf_is_a_softmax_over_the_given_logits() {
        let (p, m) = row_conf(&[2.0, 2.0, -1e30, -1e30]);
        assert!((p - 0.5).abs() < 1e-6 && m.abs() < 1e-6);
        let (p, m) = row_conf(&[0.0, -1e30]);
        assert!((p - 1.0).abs() < 1e-6 && (m - 1.0).abs() < 1e-6);
    }

    #[test]
    fn keep_count_stops_at_the_first_p_below_tau_and_keeps_one() {
        let p = [0.9, 0.8, 0.3, 0.95, 0.1];
        // 2026-10-09: First p < 0.5 is index 2: keep drafts 0 and 1.
        assert_eq!(keep_count(&p, 5, 0.5), 2);
        // 2026-10-09: A later high p does not resume the block.
        assert_eq!(keep_count(&p, 5, 0.25), 4);
        // 2026-10-09: Even the first draft below tau keeps one draft.
        assert_eq!(keep_count(&[0.1, 0.9], 2, 0.5), 1);
        // 2026-10-09: No p below tau keeps all; equality keeps (p >= tau).
        assert_eq!(keep_count(&[0.5, 0.5], 2, 0.5), 2);
        // 2026-10-09: Off (tau 0) never changes the block.
        assert_eq!(keep_count(&[0.0, 0.0, 0.0], 3, 0.0), 3);
        // 2026-10-09: An empty block stays empty; a cap below p.len() bounds the count.
        assert_eq!(keep_count(&[], 0, 0.5), 0);
        assert_eq!(keep_count(&p, 2, 0.1), 2);
    }

    #[test]
    fn p_for_answers_only_for_the_recorded_drafts() {
        // 2026-10-09: One test owns the process-global record, so the cases run in order.
        record(vec![(0.9, 0.5), (0.4, 0.1), (0.2, 0.0)], vec![7, 8, 9]);
        assert_eq!(p_for(&[7, 8, 9]), Some(vec![0.9, 0.4, 0.2]));
        assert_eq!(
            p_for(&[7, 8]),
            Some(vec![0.9, 0.4]),
            "a capped prefix still matches"
        );
        assert_eq!(p_for(&[7, 5]), None, "other tokens: stale record");
        assert_eq!(p_for(&[7, 8, 9, 10]), None, "more drafts than recorded");
        // 2026-10-09: Peeking leaves the record for the verify's log line.
        assert_eq!(take_last().map(|r| r.len()), Some(3));
        assert_eq!(p_for(&[7]), None);
    }
}
