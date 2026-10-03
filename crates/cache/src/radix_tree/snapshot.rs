// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The SSM snapshot index, separate from the radix tree: which
//! snapshot slot (or spilled tier key) holds the SSM state for a token prefix,
//! and which entry to evict.
//!
//! An entry is found by `prefix_hash` (`hash_token_prefix` of its first
//! `token_count` tokens and the adapter id); `session_hash` gates only tail
//! entries (`snapshot_session`).
//!
//! Owner: cache.
//! Invariants:
//! - At most one entry has a given `prefix_hash`: every insert path
//!   overwrites a matching entry instead of adding one.
//! - A `tiered` entry's `snapshot_id` is never returned for freeing, and
//!   `lookup` and the victim scans skip tiered entries.

use super::hash_token_prefix;
use super::snapshot_stats::SnapshotStats;

pub(super) struct SnapshotEntry {
    pub(super) snapshot_id: usize,
    pub(super) session_hash: u64,
    pub(super) token_count: usize,
    pub(super) prefix_hash: u64,
    pub(super) last_access: u64,
    /// 2026-09-25: `false`: resident in the snapshot pool at `snapshot_id`.
    /// `true`: spilled to the tier under the key `prefix_hash`, and
    /// `snapshot_id` is stale. Only `evict_to_tier` sets it.
    pub(super) tiered: bool,
    /// 2026-09-25: The session's tail snapshot, written by `insert_tail`,
    /// which first removes the session's previous tail and sibling; so a
    /// nonzero session has at most one.
    ///
    /// Outside `--hermetic`, the lookup session gate applies only to tail
    /// entries (`snapshot_session::session_gate_blocks`), so any other entry
    /// can be restored by another session. A caller whose captured state
    /// reflects tokens beyond `token_count` must therefore register it with
    /// `insert_tail`.
    pub(super) is_tail: bool,
    /// 2026-09-25: The tail's sibling, one block below it, written by
    /// `insert_tail_sibling`. Not session-gated in lookups, but leased with
    /// the tail; `insert_tail` removes it together with the tail.
    pub(super) is_tail_sibling: bool,
}

/// 2026-09-25: Where a matched snapshot's state is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SnapLoc {
    /// 2026-09-25: Resident in the snapshot pool at this slot.
    Hbm(usize),
    /// 2026-09-25: Spilled under this key (the prefix hash). The caller faults
    /// it into a fresh slot and then calls `promote`.
    Tier(u64),
}

/// 2026-09-25: The deepest snapshot for a prefix, and where its state is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SnapMatch {
    pub token_count: usize,
    pub loc: SnapLoc,
    /// 2026-09-25: Whether the matched entry is a tail entry.
    pub is_tail: bool,
}

pub(super) struct SsmSnapshotIndex {
    pub(super) entries: Vec<SnapshotEntry>,
    pub(super) access_counter: u64,
    /// 2026-09-25: Session of the most recent `lookup`/`lookup_tiered` with
    /// a nonzero session hash. Eviction leases this session's tail and
    /// sibling entries (`session_aware_victim_with_alpha`).
    pub(super) last_lookup_session: u64,
    /// 2026-09-25: Evictions since that lookup. The lease lapses at
    /// [`tail_lease_ttl`] evictions, so a session that never looks up again
    /// does not hold its slots for good.
    pub(super) evictions_since_lookup: u32,
    /// 2026-09-25: Counters; outside tests only `log_stats_if_due` reads them.
    pub(super) stats: SnapshotStats,
}

/// 2026-09-25: Whether the tail lease is on: yes unless
/// `METRALE_DISABLE_SSM_TAIL_PROTECT` is `1`, `on` or `true`.
///
/// The lease protects only tail and sibling entries. The model engine writes
/// those only from `finalize_midchunk_capture` (`midchunk_capture.rs`), so
/// with mid-chunk capture off (`METRALE_SSM_TAIL_MIDCHUNK=0`) the lease has
/// nothing to protect.
fn tail_lease_enabled() -> bool {
    // 2026-09-25: Read once per process: `tail_lease_active` runs on every
    // eviction.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("METRALE_DISABLE_SSM_TAIL_PROTECT").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
    })
}

/// 2026-09-25: Evictions a leased entry survives without its session looking
/// up again: `METRALE_SSM_TAIL_LEASE_TTL`, else 64 (also when the value does
/// not parse). Measured 2026-07-20 on an eviction test rig with 8 slots and 6
/// other requests per turn: about 18 evictions between a session's turns.
fn tail_lease_ttl() -> u32 {
    // 2026-09-25: Read once per process, like `tail_lease_enabled`.
    static TTL: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *TTL.get_or_init(|| {
        std::env::var("METRALE_SSM_TAIL_LEASE_TTL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    })
}

/// 2026-09-25: Depth weight α of the within-session eviction score
/// `S(e) = norm(recency) + α·norm(token_count)`, after Marconi
/// (arXiv:2411.19379), with token depth standing in for FLOPs saved.
/// `METRALE_SNAP_EVICT_ALPHA`, clamped to `[0, 8]`; unset or unparsable is 0,
/// which ranks by recency alone.
fn snap_evict_alpha() -> f64 {
    // 2026-09-25: Read once per process: `session_aware_victim` runs on
    // every eviction.
    static ALPHA: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *ALPHA.get_or_init(|| {
        std::env::var("METRALE_SNAP_EVICT_ALPHA")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|a| a.clamp(0.0, 8.0))
            .unwrap_or(0.0)
    })
}

impl SsmSnapshotIndex {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            access_counter: 0,
            last_lookup_session: 0,
            evictions_since_lookup: 0,
            stats: SnapshotStats::default(),
        }
    }

    /// 2026-09-25: Whether the lease is in force: a session has looked up,
    /// the lease is enabled, and fewer than `tail_lease_ttl` evictions have
    /// passed since.
    pub(super) fn tail_lease_active(&self) -> bool {
        self.last_lookup_session != 0
            && tail_lease_enabled()
            && self.evictions_since_lookup < tail_lease_ttl()
    }

    /// 2026-09-25: The deepest resident entry with `token_count <=
    /// matched_tokens` whose prefix hash (with `adapter_id` folded in)
    /// matches `tokens` and that passes the session gate, as
    /// `(snapshot_id, token_count)`. Only tests call it; `RadixTree::lookup`
    /// uses `lookup_tiered`.
    #[allow(dead_code)]
    pub(super) fn lookup(
        &mut self,
        tokens: &[u32],
        matched_tokens: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Option<(usize, usize)> {
        // 2026-09-25: A lookup with a session renews that session's lease.
        if session_hash != 0 {
            self.last_lookup_session = session_hash;
            self.evictions_since_lookup = 0;
        }
        // 2026-09-25: Only the winner's recency is bumped, after the scan, so
        // the shallow entries a deep lookup passes over do not stay fresh.
        let hermetic = metrale_gpu_runtime::hermetic_enabled();
        let mut best: Option<(usize, usize)> = None;
        let mut best_idx: Option<usize> = None;
        for (i, entry) in self.entries.iter().enumerate() {
            // 2026-09-25: A tiered entry's slot is stale.
            if entry.tiered {
                continue;
            }
            if entry.token_count > matched_tokens {
                continue;
            }
            if super::snapshot_session::session_gate_blocks(entry, session_hash, hermetic) {
                continue;
            }
            let h = hash_token_prefix(tokens, entry.token_count, adapter_id);
            if h != entry.prefix_hash {
                continue;
            }
            tracing::debug!(
                "snapshot candidate: id={} tokens={} tail={} sibling={} (matched {matched_tokens})",
                entry.snapshot_id,
                entry.token_count,
                entry.is_tail,
                entry.is_tail_sibling
            );
            if best.is_none() || entry.token_count > best.unwrap().1 {
                best = Some((entry.snapshot_id, entry.token_count));
                best_idx = Some(i);
            }
        }
        if let Some(i) = best_idx {
            self.access_counter += 1;
            self.entries[i].last_access = self.access_counter;
        }
        // 2026-09-25: Recompute counts the matched tokens the anchor does not
        // cover.
        self.stats.lookups += 1;
        match best {
            Some((_, anchor)) => {
                self.stats.hits += 1;
                self.stats.anchor_depth_sum += anchor as u64;
                self.stats.recompute_tokens_on_hit += matched_tokens.saturating_sub(anchor) as u64;
            }
            None => {
                self.stats.recompute_tokens_on_miss += matched_tokens as u64;
            }
        }
        // 2026-09-25: Read once per process.
        static DBG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *DBG.get_or_init(|| std::env::var_os("METRALE_SNAP_LOOKUP_DBG").is_some()) {
            let mut cands: Vec<usize> = self.entries.iter().map(|e| e.token_count).collect();
            cands.sort_unstable();
            tracing::info!(
                "snap-lookup: matched={matched_tokens} selected={:?} n_entries={} token_counts={:?}",
                best.map(|b| b.1),
                self.entries.len(),
                cands,
            );
        }
        self.log_stats_if_due();
        best
    }

    /// 2026-09-25: Log the counters every 64th lookup when
    /// `METRALE_SSM_SNAP_STATS` is set (to any value). Changes nothing.
    pub(super) fn log_stats_if_due(&self) {
        // 2026-09-25: Read once per process.
        static STATS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !self.stats.lookups.is_multiple_of(64)
            || !*STATS.get_or_init(|| std::env::var_os("METRALE_SSM_SNAP_STATS").is_some())
        {
            return;
        }
        let s = &self.stats;
        let hit_rate = s.hits as f64 / s.lookups.max(1) as f64;
        let mean_anchor = s.anchor_depth_sum as f64 / s.hits.max(1) as f64;
        let mean_recompute_hit = s.recompute_tokens_on_hit as f64 / s.hits.max(1) as f64;
        let tiered = self.entries.iter().filter(|e| e.tiered).count();
        tracing::info!(
            "ssm-snap-stats: lookups={} hits={} hit_rate={:.2} saves={} evictions(drops)={} \
             mean_anchor={:.0}tok mean_recompute_on_hit={:.0}tok recompute_on_miss={}tok \
             resident={} tiered={} tier_spills={} tier_hits={} tier_fault_ins={} tier_reaps={}",
            s.lookups,
            s.hits,
            hit_rate,
            s.saves,
            s.evictions,
            mean_anchor,
            mean_recompute_hit,
            s.recompute_tokens_on_miss,
            self.entries.len() - tiered,
            tiered,
            s.tier_spills,
            s.tier_hits,
            s.tier_fault_ins,
            s.tier_reaps,
        );
    }

    pub(super) fn evict_lru(&mut self) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        // 2026-09-25: Pure recency (LRU) for the per-entry path below.
        let escore = |e: &SnapshotEntry| e.last_access;

        // 2026-09-25: Session-aware eviction, unless `METRALE_SNAP_EVICT_LEGACY`
        // is set to any value (even `0`), which selects per-entry LRU. Read
        // once per process.
        static LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*LEGACY.get_or_init(|| std::env::var_os("METRALE_SNAP_EVICT_LEGACY").is_some()) {
            let tail_protect = self.tail_lease_active();
            // 2026-09-25: Tiered entries have no slot to free, so skip them.
            let victim_idx = self.session_aware_victim(tail_protect, true)?;
            let entry = self.entries.swap_remove(victim_idx);
            self.stats.evictions += 1;
            self.evictions_since_lookup = self.evictions_since_lookup.saturating_add(1);
            return Some(entry.snapshot_id);
        }

        let mut victim_idx = None;
        let mut victim_score = u64::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.tiered {
                continue;
            }
            let score = escore(entry);
            if score < victim_score {
                victim_score = score;
                victim_idx = Some(i);
            }
        }
        let entry = self.entries.swap_remove(victim_idx?);
        self.stats.evictions += 1;
        Some(entry.snapshot_id)
    }

    /// 2026-10-03: Remove the resident entry holding `slot`, counted as an eviction. `false`
    /// when none does (a tiered entry's `snapshot_id` is stale and never matches).
    pub(super) fn evict_slot(&mut self, slot: usize) -> bool {
        let Some(i) = self
            .entries
            .iter()
            .position(|e| !e.tiered && e.snapshot_id == slot)
        else {
            return false;
        };
        self.entries.swap_remove(i);
        self.stats.evictions += 1;
        self.evictions_since_lookup = self.evictions_since_lookup.saturating_add(1);
        true
    }

    /// 2026-09-25: The index of the entry to evict, without changing
    /// anything: the stalest session first, then the lowest score within it
    /// (`snap_evict_alpha`). `tail_protect` leases the last looked-up
    /// session's tail and sibling entries, while an unleased eligible entry
    /// exists. `skip_tiered` makes spilled entries ineligible. `None` if no
    /// entry is eligible.
    pub(super) fn session_aware_victim(
        &self,
        tail_protect: bool,
        skip_tiered: bool,
    ) -> Option<usize> {
        self.session_aware_victim_with_alpha(tail_protect, skip_tiered, snap_evict_alpha())
    }

    /// 2026-09-25: [`Self::session_aware_victim`] with α passed in, so tests
    /// need not set the environment.
    pub(super) fn session_aware_victim_with_alpha(
        &self,
        tail_protect: bool,
        skip_tiered: bool,
        alpha: f64,
    ) -> Option<usize> {
        let eligible = |e: &SnapshotEntry| !(skip_tiered && e.tiered);

        // 2026-09-25: A session's freshness is the newest `last_access` among
        // its eligible entries.
        let mut session_fresh: std::collections::HashMap<u64, u64> =
            std::collections::HashMap::with_capacity(self.entries.len());
        for e in self.entries.iter().filter(|e| eligible(e)) {
            let f = session_fresh.entry(e.session_hash).or_insert(0);
            if e.last_access > *f {
                *f = e.last_access;
            }
        }
        let leased = |e: &SnapshotEntry| {
            tail_protect
                && (e.is_tail || e.is_tail_sibling)
                && e.session_hash != 0
                && e.session_hash == self.last_lookup_session
        };
        // 2026-09-25: Leased entries are skipped only while an unleased
        // eligible entry exists, so any non-empty eligible pool yields a
        // victim.
        let n_unleased = self
            .entries
            .iter()
            .filter(|e| eligible(e) && !leased(e))
            .count();
        // 2026-09-25: S(e) = norm(recency) + α·norm(depth), min-max
        // normalised over the whole eligible pool on each call; an empty
        // range (max == min) normalises to 0.
        let (mut min_a, mut max_a, mut min_t, mut max_t) = (u64::MAX, 0u64, usize::MAX, 0usize);
        for e in self.entries.iter().filter(|e| eligible(e)) {
            min_a = min_a.min(e.last_access);
            max_a = max_a.max(e.last_access);
            min_t = min_t.min(e.token_count);
            max_t = max_t.max(e.token_count);
        }
        let norm = |x: u64, min: u64, max: u64| {
            if max > min {
                (x - min) as f64 / (max - min) as f64
            } else {
                0.0
            }
        };
        let score = |e: &SnapshotEntry| {
            norm(e.last_access, min_a, max_a)
                + alpha * norm(e.token_count as u64, min_t as u64, max_t as u64)
        };
        let mut victim: Option<(usize, u64, f64)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if !eligible(e) {
                continue;
            }
            if leased(e) && n_unleased >= 1 {
                continue;
            }
            let sf = *session_fresh.get(&e.session_hash).unwrap_or(&0);
            let s = score(e);
            let better = match victim {
                None => true,
                Some((_, vsf, vs)) => sf < vsf || (sf == vsf && s.total_cmp(&vs).is_lt()),
            };
            if better {
                victim = Some((i, sf, s));
            }
        }
        victim.map(|(i, _, _)| i)
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
#[path = "tests/snapshot_index.rs"]
mod tests;
