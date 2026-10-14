// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Hybrid MTP catch-up (`METRALE_HYBRID_MTP_CATCHUP=1`, #82 hybrid B2(i)): the
//! host-side bookkeeping of the ring of target hidden rows the hybrid DFlash verifies commit, and
//! the plan that feeds them to the MTP drafter at the sequence's first MTP propose after a DFlash
//! stretch. Pure (no GPU, no environment), so it is CPU-tested (`hybrid_ring_tests.rs`); the
//! model engine (`model/trait_impl/hybrid_mtp_catchup.rs`) does the copies and drafter calls.
//!
//! Why: during DFlash steps the MTP drafter state is untouched, so at a DFlash -> MTP switch it
//! lacks every token the stretch committed, and its next rows land at the stale row count (the
//! GLM MTP drafter's KV slot is its RoPE position, so those rows sit at the wrong positions).
//!
//! Label convention (the one of `METRALE_MTP_CATCHUP`, `impl_b3.rs`): ring label `n` holds the
//! target hidden of position `n - 1` (`hidden_{n-1}`); drafter pair key `k` is
//! `(embed(tokens[k + 1]), hidden_k)` and reads label `k + 1`. The GLM MTP drafter's row space is
//! dense (slot == key), so a drafter with `rows` rows holds keys `0..rows`, and the propose at
//! `position` writes key `position - 1`; the catch-up brings it to `position - 1` rows.
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - Coverage `[start, start + count)` is contiguous in label space, `count <= ring rows`, and it
//!   never holds a label whose ring row a later write overwrote.
//! - Every method is a function of its arguments and the meta alone, so two ranks that make the
//!   same calls in the same order hold the same meta and get the same plan.

/// 2026-10-14: Default ring rows (`METRALE_HYBRID_MTP_CATCHUP_ROWS`).
pub const DEFAULT_RING_ROWS: usize = 2048;
/// 2026-10-14: Smallest accepted ring: at least the widest K=gamma verify
/// (`glm_dflash::KGAMMA_MAX_K` = 32), so one verify never overwrites its own rows.
pub const MIN_RING_ROWS: usize = 64;
/// 2026-10-14: Largest accepted ring (2^20 rows, 12 GiB at hidden 6144: a typo guard).
pub const MAX_RING_ROWS: usize = 1 << 20;
/// 2026-10-14: Rows of the zeroed hidden buffer the gap fill reads; a gap is fed in chunks of
/// at most this many rows.
pub const ZERO_ROWS: usize = 256;
/// 2026-10-14: Most gap rows one catch-up fills (head and tail together). A larger gap skips the
/// whole catch-up (the propose then appends at the stale row count, as without the lever).
pub const FILL_CAP: usize = 16384;

const _: () = assert!(MIN_RING_ROWS >= super::glm_dflash::KGAMMA_MAX_K);

/// 2026-10-14: The rows a hybrid DFlash verify copied into the ring, not yet committed:
/// `k` rows of `slot` at labels `base..base + k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Staged {
    pub slot: usize,
    pub base: usize,
    pub k: usize,
}

/// 2026-10-14: One contiguous piece of a ring access: `n` rows, the first `offset` rows into the
/// access and at ring row `ring_row`. A piece never wraps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub offset: usize,
    pub ring_row: usize,
    pub n: usize,
}

/// 2026-10-14: The labels `first_label..first_label + n` of a ring of `rows` rows, split where
/// the ring wraps (label `l` lives at row `l % rows`). Empty for `n == 0` or `rows == 0`.
pub fn split_wrap(first_label: usize, n: usize, rows: usize) -> Vec<Piece> {
    let mut out = Vec::new();
    if rows == 0 {
        return out;
    }
    let mut offset = 0;
    while offset < n {
        let ring_row = (first_label + offset) % rows;
        let len = (n - offset).min(rows - ring_row);
        out.push(Piece {
            offset,
            ring_row,
            n: len,
        });
        offset += len;
    }
    out
}

/// 2026-10-14: The ring's meta (module docs). `owner` is the slot whose rows the coverage holds;
/// `resume` the slot that took DFlash steps since its last MTP propose (tracked with the lever
/// off too, for the `HYBRID MTP resume` diagnostic).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RingMeta {
    pub owner: Option<usize>,
    pub start: usize,
    pub count: usize,
    pub staged: Option<Staged>,
    pub resume: Option<usize>,
}

impl RingMeta {
    /// 2026-10-14: A hybrid DFlash verify of `slot` ran (lever off: nothing is copied).
    pub fn note_dflash(&mut self, slot: usize) {
        self.resume = Some(slot);
    }

    /// 2026-10-14: A hybrid DFlash verify of `slot` is about to copy its `k` verify rows to
    /// labels `base..base + k` of a ring of `rows` rows. Returns the pieces to copy (offset =
    /// verify row). A write that does not continue the coverage (another slot, or a label gap)
    /// restarts it at `base`; labels whose ring rows the write overwrites leave the coverage now,
    /// so a write that is never committed cannot leave a stale label covered. `k > rows` (refused
    /// by the parser's minimum) copies nothing and empties the coverage.
    pub fn stage(&mut self, slot: usize, base: usize, k: usize, rows: usize) -> Vec<Piece> {
        self.resume = Some(slot);
        self.staged = None;
        if self.owner != Some(slot) || self.count == 0 || self.start + self.count != base {
            self.owner = Some(slot);
            self.start = base;
            self.count = 0;
        }
        if k == 0 || k > rows {
            self.start = base;
            self.count = 0;
            return Vec::new();
        }
        // 2026-10-14: The write takes ring rows of labels `base..base + k`, which are also the
        // rows of labels `base - rows .. base + k - rows`: every covered label below
        // `base + k - rows` is overwritten. Coverage ends at `base` here.
        let first_kept = (base + k).saturating_sub(rows);
        if self.start < first_kept {
            self.count = self.count.saturating_sub(first_kept - self.start);
            self.start = first_kept.min(base);
            if self.count == 0 {
                self.start = base;
            }
        }
        self.staged = Some(Staged { slot, base, k });
        split_wrap(base, k, rows)
    }

    /// 2026-10-14: `commit_accepted_prefix(seq, n, k)` of a hybrid DFlash verify: `seq_len` is the
    /// sequence's length after the verify's rewind (`pre + n`). Extends the coverage by the `n`
    /// committed labels when they are the staged write's first `n` (same slot and width, `base ==
    /// pre + 1`); any other commit drops the staged rows. Returns whether it extended.
    pub fn commit(&mut self, slot: usize, seq_len: usize, n: usize, k: usize) -> bool {
        let Some(st) = self.staged.take() else {
            return false;
        };
        let ok = st.slot == slot
            && st.k == k
            && n >= 1
            && n <= k
            && seq_len >= n
            && st.base == seq_len - n + 1
            && self.owner == Some(slot)
            && self.start + self.count == st.base;
        if ok {
            self.count += n;
        }
        ok
    }

    /// 2026-10-14: A new sequence on `slot` (allocation), or a sequence moved off or onto it
    /// (compaction): the rows of whatever held `slot` before are not this sequence's.
    pub fn reset_slot(&mut self, slot: usize) {
        if self.owner == Some(slot) {
            self.owner = None;
            self.start = 0;
            self.count = 0;
            self.staged = None;
        }
        if self.resume == Some(slot) {
            self.resume = None;
        }
    }

    /// 2026-10-14: Whether `slot` took DFlash steps since its last MTP propose; clears the flag.
    pub fn take_resume(&mut self, slot: usize) -> bool {
        if self.resume == Some(slot) {
            self.resume = None;
            true
        } else {
            false
        }
    }

    /// 2026-10-14: The pair keys the ring holds for `slot`, `[lo, hi)`: label `l` feeds key
    /// `l - 1`. `None` when the coverage is another slot's or empty.
    pub fn covered_keys(&self, slot: usize) -> Option<(usize, usize)> {
        if self.owner != Some(slot) || self.count == 0 || self.start == 0 {
            return None;
        }
        Some((self.start - 1, self.start - 1 + self.count))
    }
}

/// 2026-10-14: Why a catch-up writes nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    /// The drafter has no rows: it never had its prompt prefill. Not fed (no prompt re-prefill).
    NoDrafterPrefill,
    /// The ring holds no key below `position - 1` for this slot.
    NoCoverage,
    /// The gap rows exceed [`FILL_CAP`].
    FillCap,
    /// The drafter already holds every key the ring could give it.
    Current,
}

/// 2026-10-14: What one catch-up does, in order: rewind the drafter to `rewind_to` rows, fill the
/// keys `head_fill` with zero hiddens, feed the keys `feed` from the ring, fill `tail_fill`. Key
/// ranges are half-open. `missing` = keys below `position - 1` the drafter lacked before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatchupPlan {
    pub missing: usize,
    pub rewind_to: Option<usize>,
    pub head_fill: Option<(usize, usize)>,
    pub feed: Option<(usize, usize)>,
    pub tail_fill: Option<(usize, usize)>,
    pub skip: Option<Skip>,
}

impl CatchupPlan {
    fn skipped(missing: usize, why: Skip) -> Self {
        Self {
            missing,
            rewind_to: None,
            head_fill: None,
            feed: None,
            tail_fill: None,
            skip: Some(why),
        }
    }

    /// 2026-10-14: Zero-hidden rows the plan writes.
    pub fn fill_rows(&self) -> usize {
        [self.head_fill, self.tail_fill]
            .iter()
            .flatten()
            .map(|&(a, b)| b - a)
            .sum()
    }
}

/// 2026-10-14: The catch-up of a drafter holding `drafter_rows` rows (keys `0..drafter_rows`)
/// before the MTP propose at `position`, which writes key `position - 1`; so the drafter should
/// hold keys `0..position - 1` first. `covered` is [`RingMeta::covered_keys`]; `can_rewind`
/// whether the drafter can drop rows (`DraftProposer::rewind_drafter_to`).
///
/// - A drafter with rows at or past the ring's first key is rewound to it (when it can be):
///   those rows are the MTP drafts the last MTP propose before the stretch wrote and no verify
///   trimmed (the switch drops the pending drafts), and the ring has the true rows.
/// - A drafter short of the ring's first key (a stretch longer than the ring) is filled with
///   zero-hidden rows up to it (`(embed(token), 0)`: finite, token-only context), so every later
///   row keeps its RoPE position. So is a tail the ring does not reach (plain steps after the
///   stretch). Both fills together are capped at [`FILL_CAP`].
/// - A drafter with no rows is left alone (`NoDrafterPrefill`): filling it would be a prompt
///   re-prefill.
pub fn plan_catchup(
    drafter_rows: usize,
    position: usize,
    covered: Option<(usize, usize)>,
    can_rewind: bool,
) -> CatchupPlan {
    let end = position.saturating_sub(1);
    let missing = end.saturating_sub(drafter_rows);
    if drafter_rows == 0 {
        return CatchupPlan::skipped(missing, Skip::NoDrafterPrefill);
    }
    let Some((lo, hi)) = covered else {
        return CatchupPlan::skipped(missing, Skip::NoCoverage);
    };
    let hi = hi.min(end);
    if lo >= hi {
        return CatchupPlan::skipped(missing, Skip::NoCoverage);
    }
    let rewind_to = (can_rewind && drafter_rows > lo).then_some(lo);
    let cur = rewind_to.unwrap_or(drafter_rows);
    let head_fill = (cur < lo).then_some((cur, lo));
    let feed_lo = cur.max(lo);
    let feed = (feed_lo < hi).then_some((feed_lo, hi));
    let after = hi.max(cur);
    let tail_fill = (after < end).then_some((after, end));
    let plan = CatchupPlan {
        missing,
        rewind_to,
        head_fill,
        feed,
        tail_fill,
        skip: None,
    };
    if plan.fill_rows() > FILL_CAP {
        return CatchupPlan::skipped(missing, Skip::FillCap);
    }
    if feed.is_none() && head_fill.is_none() && tail_fill.is_none() && rewind_to.is_none() {
        return CatchupPlan::skipped(missing, Skip::Current);
    }
    plan
}

#[cfg(test)]
#[path = "hybrid_ring_tests.rs"]
mod tests;
