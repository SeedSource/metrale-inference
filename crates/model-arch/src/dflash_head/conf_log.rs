// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `METRALE_DFLASH_CONF_LOG` support: the drafter's per-position top-1 probability and
//! top-1 minus top-2 margin, handed from the propose to the verify step's log line.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Log-only. Nothing here feeds a drafted token, a kernel launch or a control-flow decision.
//! - The probabilities are a softmax over the selector's top-16 logits only (the candidate
//!   set `dflash2_topk16` leaves in `scratch.sel_vals`), renormalised over those 16, not
//!   over the full vocabulary. A full-vocabulary top-1 probability is lower or equal.
//! - Recorded only for a single-sequence propose (`n_seq == 1`); a batched propose records
//!   nothing, so the verify line is absent rather than attributed to the wrong sequence.

use parking_lot::Mutex;

/// 2026-10-08: Per-draft `(p, m)` of the latest single-sequence propose, in draft order.
static LAST: Mutex<Option<Vec<(f32, f32)>>> = Mutex::new(None);

/// 2026-10-08: `(top-1 prob, top-1 minus top-2 prob)` of one row of descending top-16 logits.
pub(super) fn row_conf(vals: &[f32]) -> (f32, f32) {
    let top = vals.first().copied().unwrap_or(0.0);
    let sum: f32 = vals.iter().map(|v| (v - top).exp()).sum();
    let p1 = 1.0 / sum;
    let p2 = vals.get(1).map_or(0.0, |v| (v - top).exp() / sum);
    (p1, p1 - p2)
}

/// 2026-10-08: Store the latest propose's rows (already sliced to draft order).
pub(super) fn record(rows: Vec<(f32, f32)>) {
    *LAST.lock() = Some(rows);
}

/// 2026-10-08: Take the latest propose's `(p, m)` per draft; `None` when the lever is off,
/// the propose was batched, or it was already taken.
pub fn take_last() -> Option<Vec<(f32, f32)>> {
    LAST.lock().take()
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
}
