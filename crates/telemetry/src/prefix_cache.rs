// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The [`PrefixCache`] trait (KV block reuse across requests that
//! share a prompt prefix, plus the SSM snapshot index) and the prefix-cache
//! hit and miss counters.
//!
//! Implementations: [`NoPrefixCaching`] (caching disabled) and the
//! `metrale-cache` crate's `RadixTree`.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use std::sync::atomic::Ordering;

mod no_caching;
mod tier_evict;
pub use no_caching::NoPrefixCaching;
pub use tier_evict::TierEvict;

pub fn record_cache_hit(matched_tokens: usize) {
    let m = crate::run_metrics::metrics();
    m.cache_hits.fetch_add(1, Ordering::Relaxed);
    m.cache_hit_tokens
        .fetch_add(matched_tokens as u64, Ordering::Relaxed);
}

pub fn record_cache_miss() {
    crate::run_metrics::metrics()
        .cache_misses
        .fetch_add(1, Ordering::Relaxed);
}

pub fn cache_hit_count() -> u64 {
    crate::run_metrics::metrics()
        .cache_hits
        .load(Ordering::Relaxed)
}
pub fn cache_miss_count() -> u64 {
    crate::run_metrics::metrics()
        .cache_misses
        .load(Ordering::Relaxed)
}
pub fn cache_hit_tokens_total() -> u64 {
    crate::run_metrics::metrics()
        .cache_hit_tokens
        .load(Ordering::Relaxed)
}

/// 2026-09-26: What [`PrefixCache::evict`] removed.
#[derive(Debug, Clone, Default)]
pub struct EvictedBlocks {
    /// 2026-09-26: KV blocks the cache no longer stores; the caller returns the
    /// cache's reference on each (`PagedKvCache::return_evicted_block`).
    pub physical: Vec<u32>,
    /// 2026-09-26: The disk-block ids the evicted blocks carried; the caller
    /// `dec_disk_ref`s each. Empty without `--high-speed-swap`.
    pub disk_block_ids: Vec<u32>,
}

/// 2026-09-26: What an insert started or stopped storing, so the caller can
/// take and release the matching references.
///
/// The cache holds one KV reference on each block it stores, taken when it
/// starts storing it and handed back on eviction. Where a node already exists
/// for a chunk, the insert keeps the node's block, which can differ from the
/// inserting sequence's block, so the caller references the blocks listed
/// here rather than its own block table.
#[derive(Debug, Clone, Default)]
pub struct InsertAcquired {
    /// 2026-09-26: Disk-block ids the cache newly references; the caller
    /// `inc_disk_ref`s each.
    pub disk_block_ids: Vec<u32>,
    /// 2026-09-26: KV blocks the cache started storing (new nodes, and a new
    /// partial-suffix slot); the caller `inc_ref`s each once.
    pub blocks: Vec<u32>,
    /// 2026-09-26: KV blocks the cache stopped storing: a partial-suffix slot
    /// that was replaced. The caller `dec_ref`s each once.
    pub released_blocks: Vec<u32>,
}

impl EvictedBlocks {
    pub fn is_empty(&self) -> bool {
        self.physical.is_empty()
    }

    pub fn len(&self) -> usize {
        self.physical.len()
    }
}

/// 2026-09-26: Result of looking up a token sequence in the prefix cache.
#[derive(Debug, Clone)]
pub struct PrefixMatch {
    /// 2026-09-26: KV blocks to reuse, in token order.
    pub matched_blocks: Vec<u32>,
    /// 2026-09-26: `--high-speed-swap` disk-block ids parallel to
    /// `matched_blocks`, `u32::MAX` where a block has none; empty when no
    /// matched block has one.
    pub matched_disk_block_ids: Vec<u32>,
    /// 2026-09-26: Tokens matched. Block-aligned unless partial-tail sharing
    /// (`METRALE_PREFIX_SUBBLOCK`) matched into a block.
    pub matched_tokens: usize,
    /// 2026-09-26: The resident SSM snapshot slot of the deepest snapshot
    /// within the match, if that snapshot is resident.
    pub ssm_snapshot: Option<usize>,
    /// 2026-09-26: Tokens covered by `ssm_snapshot`, at most `matched_tokens`;
    /// SSM state for the tokens between is not covered.
    pub ssm_snapshot_tokens: usize,
    /// 2026-09-26: The tier key (prefix hash) when the deepest snapshot within
    /// the match is spilled; `ssm_snapshot` is then `None`. The caller faults
    /// it in and calls [`PrefixCache::promote_snapshot`].
    pub ssm_snapshot_tier_key: Option<u64>,
    /// 2026-09-26: Tokens covered by `ssm_snapshot_tier_key`.
    pub ssm_snapshot_tier_tokens: usize,
    /// 2026-09-26: Whether the matched snapshot is a session tail
    /// ([`PrefixCache::insert_tail_snapshot`]).
    pub ssm_snapshot_is_tail: bool,
}

impl PrefixMatch {
    /// 2026-09-26: Empty match (no cached prefix found).
    pub fn empty() -> Self {
        Self {
            matched_blocks: Vec::new(),
            matched_disk_block_ids: Vec::new(),
            matched_tokens: 0,
            ssm_snapshot: None,
            ssm_snapshot_tokens: 0,
            ssm_snapshot_tier_key: None,
            ssm_snapshot_tier_tokens: 0,
            ssm_snapshot_is_tail: false,
        }
    }

    /// 2026-09-26: Whether any prefix was matched.
    pub fn is_empty(&self) -> bool {
        self.matched_tokens == 0
    }
}

/// 2026-09-26: A prefix cache. Every method takes `&self`, so an
/// implementation keeps its state behind interior mutability.
pub trait PrefixCache: Send + Sync {
    /// 2026-09-26: Whether this cache stores blocks and holds references.
    /// `NoPrefixCaching` answers `false`, and callers then skip the cache's
    /// bookkeeping.
    fn is_active(&self) -> bool {
        true
    }

    /// 2026-09-26: Look up `tokens` and return the cached KV blocks of the
    /// longest matching prefix, taking a reference on each full matched block
    /// so it survives eviction until [`PrefixCache::release`].
    ///
    /// `session_hash` scopes session-tail snapshots; `adapter_id` keys the
    /// cache, so only blocks and snapshots computed under the same adapter
    /// match (`0` is the base model).
    fn lookup(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> PrefixMatch;

    /// 2026-09-26: The number of tokens `lookup` would match, without taking
    /// references, touching LRU state or counting a hit or miss. Keyed by
    /// `adapter_id` like `lookup`. The default answers 0.
    fn peek_matched_tokens(&self, _tokens: &[u32], _block_size: usize, _adapter_id: u64) -> usize {
        0
    }

    /// 2026-09-26: Insert a completed prefill's blocks.
    ///
    /// `block_table[i]` is the block for tokens
    /// `[i*block_size .. (i+1)*block_size]`. `disk_block_ids` is empty or
    /// parallel to `block_table` (`--high-speed-swap`). `matched_tokens` is
    /// the prefix the sequence already holds through `lookup`; the sequence
    /// gets a reference on every node past it, so that its `release` leaves
    /// the cache's own reference in place. The caller applies the returned
    /// [`InsertAcquired`].
    fn insert(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    ) -> InsertAcquired;

    /// 2026-09-26: `insert`, then register `snapshot_id` (a snapshot-pool
    /// slot) for the whole of `tokens`, tagged with `session_hash`. Returns
    /// the displaced resident slot for the caller to free, if any, and the
    /// insert's [`InsertAcquired`].
    #[allow(clippy::too_many_arguments)]
    fn insert_with_snapshot(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        snapshot_id: usize,
        session_hash: u64,
        matched_tokens: usize,
        adapter_id: u64,
    ) -> (Option<usize>, InsertAcquired);

    /// 2026-09-26: Register `snapshot_id` for `tokens`, the sequence up to the
    /// snapshot point, tagged with `session_hash`. Returns the displaced
    /// resident slot for the caller to free, if any.
    #[allow(clippy::too_many_arguments)]
    fn insert_intermediate_snapshot(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        snapshot_id: usize,
        session_hash: u64,
        matched_tokens: usize,
        adapter_id: u64,
    ) -> Option<usize>;

    /// 2026-09-26: Register the session's tail snapshot in the index only. For
    /// a nonzero `session_hash` it replaces the session's previous tail and
    /// tail sibling. Returns the displaced snapshot ids to free.
    fn insert_tail_snapshot(
        &self,
        tokens: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Vec<usize>;

    /// 2026-09-26: Register the tail's sibling snapshot in the index. Call it
    /// after `insert_tail_snapshot`, which removes the session's previous
    /// sibling. Returns a displaced snapshot id to free, if any.
    fn insert_tail_sibling_snapshot(
        &self,
        tokens: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Option<usize>;

    /// 2026-09-26: Drop the sequence's reference on each full block of
    /// `tokens` the cache holds, when the sequence finishes. `adapter_id` must
    /// be the one used at `lookup` and `insert`.
    fn release(&self, tokens: &[u32], block_size: usize, adapter_id: u64);

    /// 2026-09-26: Drop the references one `lookup` took: the full blocks of
    /// the first `matched_tokens` tokens and no more, for a caller that rolls
    /// back a lookup before any sequence owns its blocks.
    fn release_matched(
        &self,
        tokens: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    );

    /// 2026-09-26: Evict least-recently-used leaves that no sequence holds
    /// until at least `num_blocks` blocks are freed or none is left. A leaf's
    /// partial-suffix block goes with it, so one more block than asked can
    /// come back. The caller applies the [`EvictedBlocks`].
    fn evict(&self, num_blocks: usize) -> EvictedBlocks;

    /// 2026-09-26: Evict one SSM snapshot from the index; returns its slot for
    /// the caller to free.
    fn evict_snapshot_lru(&self) -> Option<usize>;

    /// 2026-10-03: Remove the resident snapshot entry that holds `slot` from the index
    /// (the caller then frees the slot). `false` when no resident entry holds it; the
    /// default answers `false`. Used by the host aux byte budget to drop a chosen victim.
    fn evict_snapshot_slot(&self, slot: usize) -> bool {
        let _ = slot;
        false
    }

    /// 2026-09-26: Pick a resident snapshot to evict: one at least
    /// `min_tokens` deep is marked spilled and stays findable, a shallower one
    /// is removed ([`TierEvict`]). `min_tokens == 0` always spills. `None`
    /// when nothing is resident; the default answers `None`.
    fn evict_snapshot_to_tier(&self, min_tokens: usize) -> Option<TierEvict> {
        let _ = min_tokens;
        None
    }

    /// 2026-09-26: After the caller faulted a spilled snapshot into
    /// `new_slot`, make its entry resident there. `false` if the key is
    /// unknown; the default answers `false`.
    fn promote_snapshot(&self, key: u64, new_slot: usize) -> bool {
        let _ = (key, new_slot);
        false
    }

    /// 2026-09-26: Remove the spilled entry for `key`, for a caller whose
    /// stored bytes for it are gone, so later lookups stop returning it. A
    /// resident entry is left alone, since its slot is live. Returns whether
    /// an entry was removed; the default answers `false`.
    fn forget_snapshot_tier_key(&self, key: u64) -> bool {
        let _ = key;
        false
    }

    /// 2026-09-26: Number of SSM snapshots in the snapshot index.
    fn snapshot_count(&self) -> usize;

    /// 2026-09-26: `(entries, cached_blocks)`. `RadixTree` stores one block
    /// per entry and answers its node count for both.
    fn stats(&self) -> (usize, usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_match_empty() {
        let m = PrefixMatch::empty();
        assert!(m.is_empty());
        assert_eq!(m.matched_tokens, 0);
        assert!(m.matched_blocks.is_empty());
    }

    #[test]
    fn test_no_prefix_caching_is_noop() {
        let cache = NoPrefixCaching;
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let block_table = vec![0, 1];
        let disk_block_ids: Vec<u32> = vec![];

        let m = cache.lookup(&tokens, 4, 0, 0);
        assert!(m.is_empty());

        let new_acq = cache.insert(&tokens, &block_table, &disk_block_ids, 4, 0, 0);
        assert!(new_acq.disk_block_ids.is_empty());
        assert!(new_acq.blocks.is_empty());
        cache.release(&tokens, 4, 0);

        let evicted = cache.evict(10);
        assert!(evicted.is_empty());

        assert_eq!(cache.evict_snapshot_lru(), None);
        assert_eq!(cache.snapshot_count(), 0);

        assert_eq!(cache.stats(), (0, 0));
    }
}
