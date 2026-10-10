// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Per-slot bookkeeping of the deferred KDA commit (`METRALE_GLM_KDA_DEFER_COMMIT=1`).
//!
//! A partial-accept commit (`commit_accepted_prefix_dispatch`) used to copy verify intermediate
//! `a - 1` of every KDA layer into the live state (68 MiB + 6.4 MiB on GLM-5.3 TP2, about 0.6 ms
//! per step, serial on both ranks). Under the lever, after a verify that read through the
//! deferred source, it records `a - 1` here instead and copies nothing. Then either:
//! - the next single-sequence verify of that slot consumes it: the model writes the index into
//!   the slot's device `i32` and row 0 of every KDA layer reads its input state from that
//!   intermediate (model-arch `glm5next_kda/snap_fuse/defer.rs`), or
//! - any other entry point first materialises it: today's copy, then the index is cleared
//!   (`trait_impl/kda_defer_hooks.rs`).
//!
//! Owner: model-engine (SSM state).
//! Invariants:
//! - `pending[s] = i >= 0` means the live state of slot `s` is stale and intermediate `i` holds
//!   the accepted state; `-1` means the live state is current.
//! - `dev[s]` is the value the device word of slot `s` holds once the work enqueued so far has
//!   run. Before a verify reads the word, [`DeferBook::sync_dev`] makes it equal `pending[s]`.
//! - Slot reuse (alloc, free, compaction) resets `pending` to `-1` and the device word to `-1`
//!   ([`DeferBook::reset_slot`]): a new occupant must never inherit an index into the previous
//!   occupant's intermediates (skill snapshot-slot-reuse-invalidation).
//! - Only slots below `slots` (the MTP-covered slots, which own intermediates) are ever pending.

use std::sync::atomic::{AtomicU64, Ordering};

use metrale_gpu_runtime::gpu::DevicePtr;
use parking_lot::Mutex;

/// 2026-10-09: The host side of the deferred commit, one entry per MTP-covered slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeferBook {
    pending: Vec<i32>,
    dev: Vec<i32>,
    /// 2026-10-09: The slot's last verify read through the deferred source, so its commit may
    /// defer. A commit after any other verify (batched, eager) copies as before.
    armed: Vec<bool>,
}

impl DeferBook {
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            pending: vec![-1; slots],
            dev: vec![-1; slots],
            armed: vec![false; slots],
        }
    }

    pub(crate) fn slots(&self) -> usize {
        self.pending.len()
    }

    pub(crate) fn covers(&self, slot: usize) -> bool {
        slot < self.pending.len()
    }

    pub(crate) fn pending(&self, slot: usize) -> Option<usize> {
        let v = *self.pending.get(slot)?;
        (v >= 0).then_some(v as usize)
    }

    pub(crate) fn any_pending(&self) -> bool {
        self.pending.iter().any(|&v| v >= 0)
    }

    /// 2026-10-09: A consumer verify of `slot` ran: its commit may defer.
    pub(crate) fn arm(&mut self, slot: usize) {
        if let Some(a) = self.armed.get_mut(slot) {
            *a = true;
        }
    }

    /// 2026-10-09: A verify that did not read through the deferred source (eager, batched, or
    /// `k < 2`) ran on `slot`: its commit must copy.
    pub(crate) fn disarm(&mut self, slot: usize) {
        if let Some(a) = self.armed.get_mut(slot) {
            *a = false;
        }
    }

    /// 2026-10-09: Disarm every slot.
    pub(crate) fn disarm_all(&mut self) {
        self.armed.iter_mut().for_each(|a| *a = false);
    }

    /// 2026-10-09: Take the arm (every commit does): whether this commit may defer.
    pub(crate) fn take_armed(&mut self, slot: usize) -> bool {
        self.armed.get_mut(slot).is_some_and(std::mem::take)
    }

    /// 2026-10-09: A partial accept deferred: intermediate `idx` holds the accepted state.
    pub(crate) fn defer(&mut self, slot: usize, idx: usize) {
        if let Some(p) = self.pending.get_mut(slot) {
            *p = idx as i32;
        }
    }

    /// 2026-10-09: A full accept, or a verify that consumed the source: the live state is current.
    pub(crate) fn clear(&mut self, slot: usize) {
        if let Some(p) = self.pending.get_mut(slot) {
            *p = -1;
        }
    }

    /// 2026-10-09: Materialise: return the pending index of `slot` and clear it.
    pub(crate) fn take_pending(&mut self, slot: usize) -> Option<usize> {
        let p = self.pending(slot)?;
        self.pending[slot] = -1;
        Some(p)
    }

    /// 2026-10-09: Every pending `(slot, idx)` except `keep`, cleared, in slot order.
    pub(crate) fn take_all_except(&mut self, keep: Option<usize>) -> Vec<(usize, usize)> {
        let slots: Vec<usize> = (0..self.pending.len())
            .filter(|&s| Some(s) != keep && self.pending[s] >= 0)
            .collect();
        slots
            .into_iter()
            .filter_map(|s| self.take_pending(s).map(|i| (s, i)))
            .collect()
    }

    /// 2026-10-09: The value to write into the device word of `slot` so that it equals
    /// `pending`, or `None` when it already does; records the write.
    pub(crate) fn sync_dev(&mut self, slot: usize) -> Option<i32> {
        let want = *self.pending.get(slot)?;
        if self.dev[slot] == want {
            return None;
        }
        self.dev[slot] = want;
        Some(want)
    }

    /// 2026-10-09: Slot reuse: nothing pending, not armed, and the device word back to `-1`
    /// (the returned value to write, when it is not already).
    pub(crate) fn reset_slot(&mut self, slot: usize) -> Option<i32> {
        if !self.covers(slot) {
            return None;
        }
        self.pending[slot] = -1;
        self.armed[slot] = false;
        self.sync_dev(slot)
    }
}

/// 2026-10-09: The pool's deferred-commit state: the book, the device words (one `i32` per
/// covered slot, allocated on the first deferred-capable verify, `-1` filled, never moved) and
/// the once-computed capability.
pub(crate) struct KdaDeferSlots {
    pub(crate) book: Mutex<DeferBook>,
    pub(crate) dev: Mutex<DevicePtr>,
    pub(crate) capable: std::sync::OnceLock<bool>,
}

impl KdaDeferSlots {
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            book: Mutex::new(DeferBook::new(slots)),
            dev: Mutex::new(DevicePtr::NULL),
            capable: std::sync::OnceLock::new(),
        }
    }
}

/// 2026-10-09: Per-process (per-rank) counters behind the ENGAGED line and the per-1000-commit
/// summary.
pub(crate) static COMMITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEFERRED: AtomicU64 = AtomicU64::new(0);
pub(crate) static CONSUMED: AtomicU64 = AtomicU64::new(0);
pub(crate) static MATERIALISED: AtomicU64 = AtomicU64::new(0);

/// 2026-10-09: Count one commit; every 1000th logs the counters.
pub(crate) fn count_commit() {
    let n = COMMITS.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_multiple_of(1000) {
        tracing::info!(
            "METRALE_GLM_KDA_DEFER_COMMIT: {n} commits, {} deferred, {} consumed by a verify, {} \
             materialised (copied by another entry point)",
            DEFERRED.load(Ordering::Relaxed),
            CONSUMED.load(Ordering::Relaxed),
            MATERIALISED.load(Ordering::Relaxed)
        );
    }
}

#[cfg(test)]
#[path = "kda_defer_tests.rs"]
mod tests;
