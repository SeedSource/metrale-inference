// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Carry the MTP drafter's KV from one turn of a session to the
//! next, so a warm turn adopts the previous turn's drafter rows and appends
//! only the new span, plus the SSM snapshot restore threshold.
//!
//! On a warm turn the reused prefix is not recomputed, so the whole-prompt
//! hidden capture does not cover the prompt and the drafter's prompt prefill
//! does not run (the model's `ensure_drafter_context`). Rebuilding it instead
//! was measured 2026-07-21 on GB10 at 1136 ms for 11,947 rows, about as long
//! as the warm TTFT of 1134 ms.
//!
//! Conventions:
//! - Drafter row `r` holds a pair key `k`: `(embed(t_{k+1}), hidden_k)` at
//!   RoPE `k + 1`. Rows are compacted while RoPE stays in sequence space, so
//!   a skipped key leaves no hole.
//! - `mtp_prefill_hidden` row `i` holds `hidden_i`, by absolute position.
//!
//! Owner: model-layers (MTP drafter).
//! Invariants:
//! - [`carry_armed`] is false whenever `mtp_max_seqs() > 1`.
//! - [`CarriedDrafter::usable_by`] returns `None` unless the request's session
//!   hash is non-zero and equals the entry's.
//! - [`StoreRange::visible_to`] returns `(0, 0)` unless the reader's ticket is
//!   non-zero and owns the range, and [`stamped_merge`] never extends a range
//!   written by another owner.

use metrale_gpu_runtime::gpu::DevicePtr;

/// 2026-09-25: Minimum snapshot depth, in matched tokens, for restoring an SSM
/// snapshot; a shallower snapshot is not restored. 0 removes the floor.
///
/// A restore skips recomputing the matched prefix, so the drafter's prompt
/// prefill does not run, and a request with no earlier turn in its session has
/// nothing to carry. Measured 2026-07-28 at C=1, warm against caching off:
/// 99 matched tokens -9.7%, 219 +9.8%, 349 +7.5%, 629 +10.6%.
///
/// A serve sets the value from `--marconi-min-tokens` through
/// [`set_marconi_min_tokens`] before building the model, so
/// `METRALE_MARCONI_MIN_TOKENS` is used only when nothing set it first.
pub fn marconi_min_tokens() -> usize {
    *MARCONI_MIN.get_or_init(|| {
        std::env::var("METRALE_MARCONI_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MARCONI_MIN_TOKENS)
    })
}

/// 2026-09-25: The default of `--marconi-min-tokens` and of the env fallback.
pub const DEFAULT_MARCONI_MIN_TOKENS: usize = 256;

static MARCONI_MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// 2026-09-25: Pin the restore threshold. The first write or read fixes it
/// for the process; a later call changes nothing and returns false, so the
/// caller can warn that its value was not applied.
pub fn set_marconi_min_tokens(v: usize) -> bool {
    MARCONI_MIN.set(v).is_ok()
}

/// 2026-09-25: Whether the carry is armed: configured on (`cfg.carry`) and MTP
/// not in multi-sequence mode. It takes `multi_seq` as an argument so tests can
/// set it; [`crate::speculative::mtp_max_seqs`] caches its env read.
///
/// The default cap is 32 (4 with `METRALE_NO_MTP_K_LADDER`), so `multi_seq` is
/// true and the carry is off unless `METRALE_MTP_MAX_SEQS` is at most 1,
/// whatever `cfg.carry` says. Report this, not `cfg.carry`, as the carry's state.
pub fn carry_armed_with(cfg: crate::drafter_context::DrafterContext, multi_seq: bool) -> bool {
    cfg.carry && !multi_seq
}

/// 2026-09-25: [`carry_armed_with`] against the live dispatch cap.
pub fn carry_armed(cfg: crate::drafter_context::DrafterContext) -> bool {
    carry_armed_with(cfg, crate::speculative::mtp_multi_seq_mode())
}

pub fn mtp_carry_drafter_enabled(levers: &crate::layers::ops::ModelLevers) -> bool {
    carry_armed(levers.drafter)
}

/// 2026-09-25: `METRALE_MTP_CARRY_DEBUG=1` logs one line per carry store and
/// per adopt decision.
pub fn mtp_carry_debug() -> bool {
    std::env::var("METRALE_MTP_CARRY_DEBUG").ok().as_deref() == Some("1")
}

/// 2026-09-25: The drafter KV of a finished turn, held in the model's single
/// carry slot for the next turn of the same session.
pub struct CarriedDrafter {
    /// 2026-09-25: Drafter KV blocks moved out of the finished sequence's
    /// proposer state (`DraftProposer::take_drafter_kv`), so freeing that
    /// state does not release them.
    pub block_table: Vec<u32>,
    /// 2026-09-25: Drafter rows resident in those blocks.
    pub rows: usize,
    /// 2026-09-25: Sequence-space pair key of the newest resident row.
    pub last_pair_key: Option<usize>,
    /// 2026-09-25: The finished sequence's tokens. [`Self::usable_by`] keeps
    /// only the rows within their common prefix with a new prompt; a common
    /// prefix does not show that the rows belong to the new request.
    pub tokens: Vec<u32>,
    /// 2026-09-25: `SequenceState::session_hash` of the sequence that left
    /// these rows. The slot holds whichever sequence finished last, so
    /// adoption also requires this to match.
    pub session_hash: u64,
}

impl CarriedDrafter {
    /// 2026-09-25: Length of the common prefix of `self.tokens` and `prompt`.
    pub fn common_prefix_len(&self, prompt: &[u32]) -> usize {
        self.tokens
            .iter()
            .zip(prompt.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// 2026-09-25: Whether these rows belong to `session_hash`. A zero hash
    /// never matches: an unstamped request has no session to check. The SSM
    /// snapshot pool's `session_matches` instead accepts 0.
    pub fn session_matches(&self, session_hash: u64) -> bool {
        session_hash != 0 && self.session_hash == session_hash
    }

    /// 2026-09-25: The `(rows, last_pair_key)` of this entry that `prompt` may
    /// adopt, or `None`. `None` unless [`Self::session_matches`].
    ///
    /// Pair key `k` consumed `tokens[0..=k + 1]`, so keys up to the common
    /// prefix length minus 2 survive. The rows above that key are dropped from
    /// the tail, one row per key. When the tail had key gaps the returned key
    /// can be above the newest surviving row's real key, which only makes the
    /// append start later.
    pub fn usable_by(&self, prompt: &[u32], session_hash: u64) -> Option<(usize, usize)> {
        if !self.session_matches(session_hash) {
            return None;
        }
        let k = self.last_pair_key?;
        if self.rows == 0 {
            return None;
        }
        let common = self.common_prefix_len(prompt);
        // 2026-09-25: Pair key 0 survives only if tokens[0..=1] are common.
        let max_key = common.checked_sub(2)?;
        let key = k.min(max_key);
        let dropped = k - key;
        let rows = self.rows.checked_sub(dropped)?;
        if rows == 0 { None } else { Some((rows, key)) }
    }
}

/// 2026-10-03: Keep only carried pair keys up to `max_key` of an adopted `(rows, key)`, with the
/// same tail arithmetic as [`CarriedDrafter::usable_by`] (one row per dropped key). The absolute
/// prefill grid (`METRALE_PREFIX_GRID_RESTORE`) caps at the restore point minus one, so every key
/// from the restore point on is rebuilt from this turn's hidden rows, as a cold turn builds it.
/// `None` when no row survives.
pub fn cap_adopted(rows: usize, key: usize, max_key: usize) -> Option<(usize, usize)> {
    if key <= max_key {
        return Some((rows, key));
    }
    let rows = rows.checked_sub(key - max_key)?;
    if rows == 0 { None } else { Some((rows, max_key)) }
}

/// 2026-09-25: The warm-turn append after `last_pair_key` for a prompt of
/// `prompt_len` tokens, reading hidden rows `[hidden_lo, hidden_hi)`.
///
/// - `first_key` is `last_pair_key + 1`, raised to `hidden_lo` when the
///   hidden store starts later; the skipped keys leave no hole (module docs).
/// - `rows` covers pair keys `first_key ..= prompt_len - 2`.
///
/// `None` when no key is missing, or when the store does not cover
/// `[first_key, prompt_len - 2]`.
pub fn plan_append(
    last_pair_key: usize,
    prompt_len: usize,
    hidden_lo: usize,
    hidden_hi: usize,
) -> Option<AppendPlan> {
    // 2026-09-25: Pair keys run 0 ..= prompt_len - 2.
    let last_key_needed = prompt_len.checked_sub(2)?;
    let first_key = (last_pair_key + 1).max(hidden_lo);
    if first_key > last_key_needed {
        return None;
    }
    // 2026-09-25: Pair key k reads hidden row k; `hidden_hi` is exclusive.
    if hidden_hi <= last_key_needed || hidden_lo > first_key {
        return None;
    }
    Some(AppendPlan {
        first_key,
        rows: last_key_needed - first_key + 1,
    })
}

#[derive(Debug, PartialEq, Eq)]
pub struct AppendPlan {
    pub first_key: usize,
    pub rows: usize,
}

/// 2026-09-25: Address of hidden row `pos` in a `[capacity, hidden_size]`
/// BF16 store starting at `base`.
pub fn hidden_row_offset(base: DevicePtr, pos: usize, hidden_size: usize) -> DevicePtr {
    base.offset(pos * hidden_size * 2)
}

/// 2026-09-25: The interval of `mtp_prefill_hidden` rows written, and the
/// sequence ticket that wrote them. The buffer is model-level and indexed by
/// absolute position, so the interval alone cannot say whose rows it holds.
///
/// Owner 0 never matches: it marks an unclaimed range, and it is the ticket of
/// a `SequenceState` not built by the model's `alloc_sequence`, which draws
/// tickets from 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreRange {
    /// 2026-09-25: The writer's ticket (`SequenceState::mtp_store_gen`); 0
    /// means unclaimed.
    pub owner: u64,
    /// 2026-09-25: First absolute position written.
    pub lo: usize,
    /// 2026-09-25: One past the last absolute position written.
    pub hi: usize,
}

impl StoreRange {
    /// 2026-09-25: Nothing claimed.
    pub const EMPTY: Self = Self {
        owner: 0,
        lo: 0,
        hi: 0,
    };

    /// 2026-09-25: The interval `reader_gen` may read, or `(0, 0)` when the
    /// rows belong to another ticket or `reader_gen` is 0. [`plan_append`]
    /// refuses `(0, 0)` like any interval that does not cover its span.
    pub fn visible_to(self, reader_gen: u64) -> (usize, usize) {
        if reader_gen != 0 && self.owner == reader_gen {
            (self.lo, self.hi)
        } else {
            (0, 0)
        }
    }
}

/// 2026-09-25: Record a write of `[start, start + count)` by `writer_gen`. The
/// same non-zero owner merges it with [`merge_interval`]; any other owner, or
/// an unclaimed range, is replaced by the write. Merging across owners would
/// leave another sequence's rows inside the interval.
///
/// The same-owner case relies on [`merge_interval`] replacing on a gap, which
/// `merge_interval_replaces_on_a_gap` pins.
pub fn stamped_merge(cur: StoreRange, writer_gen: u64, start: usize, count: usize) -> StoreRange {
    if cur.owner != 0 && cur.owner == writer_gen {
        let (lo, hi) = merge_interval((cur.lo, cur.hi), start, count);
        StoreRange {
            owner: writer_gen,
            lo,
            hi,
        }
    } else {
        StoreRange {
            owner: writer_gen,
            lo: start,
            hi: start + count,
        }
    }
}

/// 2026-09-25: Merge a write of `[start, start + count)` into the interval
/// `[lo, hi)`. An overlapping or abutting write extends it; a disjoint write
/// replaces it, because one interval cannot describe two runs and claiming
/// the gap would mark unwritten rows valid.
pub fn merge_interval(cur: (usize, usize), start: usize, count: usize) -> (usize, usize) {
    let (lo, hi) = cur;
    let (ns, ne) = (start, start + count);
    if hi > lo && ns <= hi && ne >= lo {
        (lo.min(ns), hi.max(ne))
    } else {
        (ns, ne)
    }
}

/// 2026-09-25: Result of a carry attempt, for logging and tests.
#[derive(Debug, PartialEq, Eq)]
pub enum CarryOutcome {
    Adopted {
        rows: usize,
        appended: usize,
        first_key: usize,
    },
    NoCarry,
    PrefixMismatch {
        common: usize,
        entry_rows: usize,
    },
    /// 2026-09-25: No append plan because another non-zero ticket owns the
    /// stored hidden rows; otherwise the same refusal as `NoHiddens`.
    ForeignHiddens {
        /// 2026-09-25: The ticket that owns the rows.
        owner: u64,
        /// 2026-09-25: The ticket that wanted to read them.
        expected: u64,
    },
    /// 2026-09-25: The slot held another session's rows, or this request has
    /// no session. `PrefixMismatch` is instead an entry of this session whose
    /// prefix did not match.
    ForeignSession {
        entry_session: u64,
        prompt_session: u64,
    },
    NoHiddens,
}

impl std::fmt::Display for CarryOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CarryOutcome::Adopted {
                rows,
                appended,
                first_key,
            } => write!(
                f,
                "adopted rows={rows} appended={appended} first_key={first_key}"
            ),
            CarryOutcome::NoCarry => write!(f, "no carried state"),
            CarryOutcome::PrefixMismatch { common, entry_rows } => {
                write!(
                    f,
                    "prefix mismatch (common={common} entry_rows={entry_rows})"
                )
            }
            CarryOutcome::ForeignSession {
                entry_session,
                prompt_session,
            } => write!(
                f,
                "foreign session (entry={entry_session:#x} prompt={prompt_session:#x})"
            ),
            CarryOutcome::ForeignHiddens { owner, expected } => write!(
                f,
                "hidden rows belong to sequence gen {owner}, not {expected}"
            ),
            CarryOutcome::NoHiddens => write!(f, "hidden store does not cover the append span"),
        }
    }
}

#[cfg(test)]
#[path = "mtp_carry_tests.rs"]
mod tests;
