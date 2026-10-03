// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The radix-tree prefix cache (`PrefixCache` implementation):
//! KV block reuse plus the SSM snapshot index.
//!
//! Token sequences are chunked at `block_size`; each tree node stores one KV
//! block. A lookup walks block-aligned chunks and returns the cached block
//! indices, then asks the snapshot index for the deepest SSM snapshot within
//! the matched prefix.
//!
//! Owner: cache.
//! Invariants:
//! - The tree and the snapshot index are behind separate mutexes, and no
//!   method holds both: `inner` is released before `snapshot_index` is taken.
//! - One radix root per `adapter_id`, so no lookup returns another adapter's
//!   blocks.

use parking_lot::Mutex;

use metrale_telemetry::prefix_cache::{EvictedBlocks, PrefixCache, PrefixMatch};

mod inner;
mod partial_tail;
mod snapshot;
mod snapshot_insert;
mod snapshot_session;
mod snapshot_stats;
mod snapshot_tier;

#[cfg(test)]
mod tests;

use inner::RadixTreeInner;
use snapshot::SsmSnapshotIndex;

/// 2026-09-25: FNV-1a hash of the first `count` tokens, the key of an SSM
/// snapshot. A nonzero `adapter_id` is folded in first, so adapters sharing a
/// token prefix get different keys; `adapter_id == 0` (the base model) is not
/// folded in.
pub(crate) fn hash_token_prefix(tokens: &[u32], count: usize, adapter_id: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    if adapter_id != 0 {
        h ^= adapter_id;
        h = h.wrapping_mul(0x100000001b3);
    }
    for &t in &tokens[..count] {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 2026-09-25: The radix-tree prefix cache. SSM snapshots live in a separate
/// `SsmSnapshotIndex`, so evicting tree nodes does not remove them.
pub struct RadixTree {
    inner: Mutex<RadixTreeInner>,
    snapshot_index: Mutex<SsmSnapshotIndex>,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTree {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(RadixTreeInner::new()),
            snapshot_index: Mutex::new(SsmSnapshotIndex::new()),
        }
    }
}

impl PrefixCache for RadixTree {
    fn lookup(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> PrefixMatch {
        // 2026-09-25: Walk the tree under `inner`, then release it.
        let (matched_blocks, matched_disk_block_ids, matched_tokens) = {
            let mut inner = self.inner.lock();
            let (blocks, disk, matched) = inner.walk(tokens, block_size, adapter_id);
            if matched > 0 {
                inner.inc_refs(tokens, block_size, matched, adapter_id);
                metrale_telemetry::prefix_cache::record_cache_hit(matched);
            } else {
                metrale_telemetry::prefix_cache::record_cache_miss();
            }
            (blocks, disk, matched)
        };
        // 2026-09-25: Then the snapshot index. `lookup_tiered` returns the
        // deepest anchor among resident and spilled entries: a resident hit
        // fills `ssm_snapshot`, a spilled one `ssm_snapshot_tier_key`, which
        // the caller faults in.
        let mut ssm_snapshot = None;
        let mut ssm_snapshot_tokens = 0;
        let mut ssm_snapshot_tier_key = None;
        let mut ssm_snapshot_tier_tokens = 0;
        let mut ssm_snapshot_is_tail = false;
        if matched_tokens > 0 {
            let mut idx = self.snapshot_index.lock();
            if let Some(m) = idx.lookup_tiered(tokens, matched_tokens, session_hash, adapter_id) {
                ssm_snapshot_is_tail = m.is_tail;
                match m.loc {
                    snapshot::SnapLoc::Hbm(slot) => {
                        ssm_snapshot = Some(slot);
                        ssm_snapshot_tokens = m.token_count;
                    }
                    snapshot::SnapLoc::Tier(key) => {
                        ssm_snapshot_tier_key = Some(key);
                        ssm_snapshot_tier_tokens = m.token_count;
                    }
                }
            }
        }
        // 2026-09-25: A list of only `u32::MAX` (no node carries a disk id)
        // becomes empty, so a caller can test `is_empty()`. A list with any
        // real id is passed through whole, `u32::MAX` entries included.
        let matched_disk_block_ids = if matched_disk_block_ids.iter().all(|&id| id == u32::MAX) {
            Vec::new()
        } else {
            matched_disk_block_ids
        };
        PrefixMatch {
            matched_blocks,
            matched_disk_block_ids,
            matched_tokens,
            ssm_snapshot,
            ssm_snapshot_tokens,
            ssm_snapshot_tier_key,
            ssm_snapshot_tier_tokens,
            ssm_snapshot_is_tail,
        }
    }

    fn peek_matched_tokens(&self, tokens: &[u32], block_size: usize, adapter_id: u64) -> usize {
        self.inner.lock().walk(tokens, block_size, adapter_id).2
    }

    fn insert(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    ) -> metrale_telemetry::prefix_cache::InsertAcquired {
        self.inner.lock().insert(
            tokens,
            block_table,
            disk_block_ids,
            block_size,
            matched_tokens,
            adapter_id,
        )
    }

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
    ) -> (
        Option<usize>,
        metrale_telemetry::prefix_cache::InsertAcquired,
    ) {
        // 2026-09-25: Tree nodes under `inner`, released before the snapshot
        // index is taken.
        let newly_acquired = self.inner.lock().insert(
            tokens,
            block_table,
            disk_block_ids,
            block_size,
            matched_tokens,
            adapter_id,
        );
        let prefix_hash = hash_token_prefix(tokens, tokens.len(), adapter_id);
        let mut idx = self.snapshot_index.lock();
        let displaced = idx.insert(prefix_hash, snapshot_id, session_hash, tokens.len());
        (displaced, newly_acquired)
    }

    fn insert_tail_snapshot(
        &self,
        tokens: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Vec<usize> {
        // 2026-09-25: Index only; the tree is not touched.
        let prefix_hash = hash_token_prefix(tokens, tokens.len(), adapter_id);
        self.snapshot_index
            .lock()
            .insert_tail(prefix_hash, snapshot_id, session_hash, tokens.len())
    }

    fn insert_tail_sibling_snapshot(
        &self,
        tokens: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Option<usize> {
        // 2026-09-25: Index only; the tree is not touched.
        let prefix_hash = hash_token_prefix(tokens, tokens.len(), adapter_id);
        self.snapshot_index.lock().insert_tail_sibling(
            prefix_hash,
            snapshot_id,
            session_hash,
            tokens.len(),
        )
    }

    fn insert_intermediate_snapshot(
        &self,
        tokens: &[u32],
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _block_size: usize,
        snapshot_id: usize,
        session_hash: u64,
        _matched_tokens: usize,
        adapter_id: u64,
    ) -> Option<usize> {
        // 2026-09-25: Index only, at `tokens.len()`; the tree and its ref
        // counts are not touched, and the block arguments are unused.
        let prefix_hash = hash_token_prefix(tokens, tokens.len(), adapter_id);
        let mut idx = self.snapshot_index.lock();
        idx.insert(prefix_hash, snapshot_id, session_hash, tokens.len())
    }

    fn release(&self, tokens: &[u32], block_size: usize, adapter_id: u64) {
        self.inner
            .lock()
            .dec_refs(tokens, block_size, tokens.len(), adapter_id);
    }

    fn release_matched(
        &self,
        tokens: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    ) {
        self.inner
            .lock()
            .dec_refs(tokens, block_size, matched_tokens, adapter_id);
    }

    fn evict(&self, num_blocks: usize) -> EvictedBlocks {
        let (physical, disk) = self.inner.lock().evict(num_blocks);
        // 2026-09-25: `u32::MAX` means no disk id, so only real ids are
        // returned for the caller to `dec_disk_ref`.
        let disk_block_ids: Vec<u32> = disk.into_iter().filter(|&id| id != u32::MAX).collect();
        EvictedBlocks {
            physical,
            disk_block_ids,
        }
    }

    fn evict_snapshot_lru(&self) -> Option<usize> {
        self.snapshot_index.lock().evict_lru()
    }

    fn evict_snapshot_slot(&self, slot: usize) -> bool {
        self.snapshot_index.lock().evict_slot(slot)
    }

    fn evict_snapshot_to_tier(
        &self,
        min_tokens: usize,
    ) -> Option<metrale_telemetry::prefix_cache::TierEvict> {
        self.snapshot_index.lock().evict_to_tier(min_tokens)
    }

    fn promote_snapshot(&self, key: u64, new_slot: usize) -> bool {
        self.snapshot_index.lock().promote(key, new_slot)
    }

    fn forget_snapshot_tier_key(&self, key: u64) -> bool {
        self.snapshot_index.lock().forget_tiered(key)
    }

    fn snapshot_count(&self) -> usize {
        self.snapshot_index.lock().len()
    }

    fn stats(&self) -> (usize, usize) {
        let inner = self.inner.lock();
        let entries = inner.num_entries();
        (entries, entries)
    }
}
