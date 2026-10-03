// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `RadixTree` tests through the `PrefixCache` API: insert, lookup, release, eviction, branching, sub-block prompts, owned-block accounting and SSM snapshots.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use crate::radix_tree::RadixTree;
use metrale_telemetry::prefix_cache::PrefixCache;

#[path = "basic_hss.rs"]
mod hss;

#[test]
fn test_insert_and_lookup_exact() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    let block_table = vec![42];

    tree.insert(&tokens, &block_table, &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);

    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![42]);
}

#[test]
fn test_insert_and_lookup_multi_block() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..48).collect();
    let block_table = vec![10, 20, 30];

    tree.insert(&tokens, &block_table, &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);

    assert_eq!(m.matched_tokens, 48);
    assert_eq!(m.matched_blocks, vec![10, 20, 30]);
}

#[test]
fn test_partial_match() {
    let tree = RadixTree::new();
    let tokens_a: Vec<u32> = (0..32).collect();
    tree.insert(&tokens_a, &[10, 20], &[], 16, 0, 0);

    let tokens_b: Vec<u32> = (0..48).collect();
    let m = tree.lookup(&tokens_b, 16, 0, 0);

    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_b, 16, 0);

    // 2026-09-25: The checks above also pass for a walk that stops one block
    // short of the deepest cached node, since block 3 was never cached. Cache
    // a third block under one continuation, then look up a different one: the
    // match must stop exactly at the divergent block.
    let mut tokens_c: Vec<u32> = (0..32).collect();
    tokens_c.extend(200..216);
    tree.insert(&tokens_c, &[10, 20, 30], &[], 16, 0, 0);
    tree.release(&tokens_c, 16, 0);

    let deep = tree.lookup(&tokens_c, 16, 0, 0);
    assert_eq!(
        deep.matched_tokens, 48,
        "the deepest cached block must match"
    );
    assert_eq!(deep.matched_blocks, vec![10, 20, 30]);
    tree.release(&tokens_c, 16, 0);

    let mut tokens_d: Vec<u32> = (0..32).collect();
    tokens_d.extend(700..716);
    let diverged = tree.lookup(&tokens_d, 16, 0, 0);
    assert_eq!(
        diverged.matched_tokens, 32,
        "the match stops at the divergent block, not before it"
    );
    assert_eq!(diverged.matched_blocks, vec![10, 20]);
    tree.release(&tokens_d, 16, 0);
}

#[test]
fn test_no_match() {
    let tree = RadixTree::new();
    let tokens_a: Vec<u32> = (0..16).collect();
    tree.insert(&tokens_a, &[10], &[], 16, 0, 0);

    let tokens_b: Vec<u32> = (100..116).collect();
    let m = tree.lookup(&tokens_b, 16, 0, 0);

    assert!(m.is_empty());
    assert!(m.matched_blocks.is_empty(), "a miss hands back no blocks");

    // 2026-09-25: Positive control: the same cache still hits its own key, so
    // the miss above comes from the token mismatch and not from a cache that
    // misses everything.
    let hit = tree.lookup(&tokens_a, 16, 0, 0);
    assert_eq!(
        hit.matched_tokens, 16,
        "the same cache still hits its own key"
    );
    assert_eq!(hit.matched_blocks, vec![10]);
    tree.release(&tokens_a, 16, 0);
}

#[test]
fn test_release_decrements_refcount() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    // 2026-09-25: A sequence that matched nothing (`matched_tokens = 0`)
    // inserts and then releases: the node is created at ref_count 2
    // (`RadixTreeInner::insert`) and drops back to the cache's own ref of 1.
    tree.insert(&tokens, &[42], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    let _ = tree.lookup(&tokens, 16, 0, 0);
    tree.release(&tokens, 16, 0);

    // 2026-09-25: Back at the cache's own ref; `evict` takes nodes with
    // ref_count <= 1.
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![42]);
}

#[test]
fn test_insert_release_lookup_survives() {
    // 2026-09-25: `walk` only matches nodes with ref_count > 0. `insert`
    // gives blocks past `matched_tokens` an extra ref for the inserting
    // sequence, so that sequence's `release` leaves the cache's own ref and the
    // entry stays findable.
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32, "stale cache entry after seq exit");
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_evict_lru_order() {
    let tree = RadixTree::new();

    let tokens_a: Vec<u32> = (0..16).collect();
    tree.insert(&tokens_a, &[10], &[], 16, 0, 0);
    tree.release(&tokens_a, 16, 0);

    let tokens_b: Vec<u32> = (100..116).collect();
    tree.insert(&tokens_b, &[20], &[], 16, 0, 0);
    tree.release(&tokens_b, 16, 0);

    // 2026-09-25: A hit on A makes it the most recently used, while by
    // insertion order it is still the oldest. The next eviction tells LRU from
    // FIFO: it observes `inc_refs` refreshing `last_access` on a hit.
    let hit = tree.lookup(&tokens_a, 16, 0, 0);
    assert_eq!(hit.matched_blocks, vec![10]);
    tree.release(&tokens_a, 16, 0);

    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![20], "the least recently USED block");

    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);
    assert_eq!(tree.stats(), (0, 0), "and nothing else survived");
}

#[test]
fn test_evict_skips_referenced_nodes() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    // 2026-09-25: After insert and release the node holds only the cache's
    // own ref (ref_count 1).
    tree.insert(&tokens, &[42], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    // 2026-09-25: A live match raises ref_count to 2; `evict` skips nodes
    // above 1.
    let _ = tree.lookup(&tokens, 16, 0, 0);

    let evicted = tree.evict(1);
    assert!(evicted.is_empty());

    tree.release(&tokens, 16, 0);
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![42]);
}

#[test]
fn test_evict_chain_from_leaf() {
    let tree = RadixTree::new();
    // 2026-09-25: `evict` takes only childless nodes, so a chain is freed
    // leaf first.
    let tokens: Vec<u32> = (0..48).collect();
    tree.insert(&tokens, &[10, 20, 30], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![30]);

    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![20]);

    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);

    assert_eq!(tree.stats(), (0, 0));
}

#[test]
fn test_insert_idempotent() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();

    tree.insert(&tokens, &[42], &[], 16, 0, 0);
    tree.insert(&tokens, &[42], &[], 16, 0, 0);

    assert_eq!(tree.stats(), (1, 1));

    // 2026-09-25: A node count cannot say which block the node holds. A
    // re-insert of the same chunk with a different block (43) inside the
    // matched prefix must keep the block the cache holds a ref on (42):
    // `insert` leaves the node's `block_idx` unchanged and reports nothing.
    let second = tree.insert(&tokens, &[43], &[], 16, 16, 0);
    assert_eq!(tree.stats(), (1, 1), "still one node");
    assert!(
        second.blocks.is_empty() && second.released_blocks.is_empty(),
        "no block changed hands, so no KV ref is owed or released; got {second:?}"
    );
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(
        m.matched_blocks,
        vec![42],
        "the node keeps the block the cache references"
    );
    tree.release(&tokens, 16, 0);

    // 2026-10-03: Past the matched prefix the sequence owns its block (43), so
    // the node adopts it and reports the swap: a ref owed on 43 and one
    // released on 42 (race #79).
    let third = tree.insert(&tokens, &[43], &[], 16, 0, 0);
    assert_eq!(tree.stats(), (1, 1), "still one node");
    assert_eq!(third.blocks, vec![43]);
    assert_eq!(third.released_blocks, vec![42]);
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_blocks, vec![43], "the node holds the sequence's block");
    tree.release(&tokens, 16, 0);
}

/// 2026-10-03: Race #79 (196K needle after a 131K needle, prefix cache on,
/// grid restore at point 0): a sequence that recomputes a cached prefix into
/// its own blocks and then inserts must not leave the old blocks pinned. Before
/// the fix every existing node kept the old block and took the sequence's ref,
/// so nothing could be evicted while the sequence lived; each old block was a
/// second copy the sequence never read.
#[test]
fn test_recomputed_prefix_insert_frees_old_blocks() {
    let tree = RadixTree::new();
    let shared: Vec<u32> = (0..64).collect();

    // 2026-10-03: Request 1 cached 4 blocks, then finished (refs released).
    let first = tree.insert(&shared, &[10, 11, 12, 13], &[], 16, 0, 0);
    assert_eq!(first.blocks, vec![10, 11, 12, 13]);
    tree.release(&shared, 16, 0);

    // 2026-10-03: Request 2 shares the first 3 chunks, restored nothing
    // (matched_tokens 0) and computed every chunk into its own blocks.
    let mut second_toks: Vec<u32> = (0..48).collect();
    second_toks.extend(500..532);
    let second = tree.insert(&second_toks, &[20, 21, 22, 23, 24], &[], 16, 0, 0);
    assert_eq!(
        second.released_blocks,
        vec![10, 11, 12],
        "the shared nodes hand back request 1's blocks"
    );
    assert_eq!(second.blocks, vec![20, 21, 22, 23, 24]);

    // 2026-10-03: While request 2 lives, only request 1's private tail (13) is
    // evictable; the shared path now holds request 2's live blocks.
    let evicted = tree.evict(16);
    assert_eq!(evicted.physical, vec![13]);
    let m = tree.lookup(&second_toks, 16, 0, 0);
    assert_eq!(m.matched_blocks, vec![20, 21, 22, 23, 24]);
    tree.release(&second_toks, 16, 0);
}

#[test]
fn test_branching_tree() {
    let tree = RadixTree::new();

    let tokens_a: Vec<u32> = (0..32).collect();
    tree.insert(&tokens_a, &[10, 20], &[], 16, 0, 0);

    let mut tokens_b: Vec<u32> = (0..16).collect();
    tokens_b.extend(100..116);
    tree.insert(&tokens_b, &[10, 30], &[], 16, 0, 0);

    let m = tree.lookup(&tokens_a, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_a, 16, 0);

    let m = tree.lookup(&tokens_b, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 30]);
    tree.release(&tokens_b, 16, 0);

    // 2026-09-25: The shared first chunk is one node: 10, 20 and 30.
    assert_eq!(tree.stats(), (3, 3));
}

#[test]
fn test_sub_block_tokens_ignored() {
    let tree = RadixTree::new();
    // 2026-09-25: Fewer tokens than one block: `insert` creates no node, and
    // a partial suffix is never stored on the root.
    let tokens: Vec<u32> = (0..10).collect();
    let acquired = tree.insert(&tokens, &[42], &[], 16, 0, 0);

    assert_eq!(tree.stats(), (0, 0));

    // 2026-09-25: No ref is owed on block 42 either: a ref the cache reports
    // but never stores is never handed back by `evict`.
    assert!(
        acquired.blocks.is_empty(),
        "no ref is owed on a block the cache did not store; got {acquired:?}"
    );
    assert!(acquired.disk_block_ids.is_empty());
    assert!(acquired.released_blocks.is_empty());
    assert!(
        tree.lookup(&tokens, 16, 0, 0).is_empty(),
        "and nothing is findable"
    );
}

#[test]
fn test_ssm_snapshot_insert_and_lookup() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();

    tree.insert_with_snapshot(&tokens, &[10, 20], &[], 16, 42, 0, 0, 0);

    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.ssm_snapshot, Some(42));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_ssm_snapshot_partial_match_returns_deepest() {
    let tree = RadixTree::new();

    let tokens: Vec<u32> = (0..48).collect();
    tree.insert_with_snapshot(&tokens, &[10, 20, 30], &[], 16, 99, 0, 0, 0);

    // 2026-09-25: The only snapshot is at 48 tokens, deeper than the 32
    // matched, so there is none to restore.
    let tokens_short: Vec<u32> = (0..32).collect();
    let m = tree.lookup(&tokens_short, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.ssm_snapshot, None);
    assert_eq!(m.ssm_snapshot_tokens, 0);
    tree.release(&tokens_short, 16, 0);
}

#[test]
fn test_ssm_snapshot_survives_tree_eviction() {
    // 2026-09-25: Snapshots live in a separate index, so evicting the tree
    // node leaves the snapshot in place.
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();

    tree.insert_with_snapshot(&tokens, &[10], &[], 16, 7, 0, 0, 0);
    tree.release(&tokens, 16, 0);

    let evicted_blocks = tree.evict(1);
    assert_eq!(evicted_blocks.physical, vec![10]);

    assert_eq!(tree.snapshot_count(), 1);

    let snap = tree.evict_snapshot_lru();
    assert_eq!(snap, Some(7));
    assert_eq!(tree.snapshot_count(), 0);

    assert_eq!(tree.evict_snapshot_lru(), None);
}

#[test]
fn test_ssm_snapshot_overwrite_returns_displaced() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();

    let (displaced, _acquired) = tree.insert_with_snapshot(&tokens, &[10], &[], 16, 5, 0, 0, 0);
    assert_eq!(displaced, None);

    let (displaced, _acquired) = tree.insert_with_snapshot(&tokens, &[10], &[], 16, 8, 0, 0, 0);
    assert_eq!(displaced, Some(5));

    assert_eq!(tree.snapshot_count(), 1);

    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.ssm_snapshot, Some(8));
    assert_eq!(m.ssm_snapshot_tokens, 16);
    tree.release(&tokens, 16, 0);
}

/// 2026-09-25: The prefix-cache key is built from token IDs only, so two
/// images behind the same prompt and the same run of image-pad tokens collide
/// on one entry, KV blocks and SSM snapshot alike. This is why model-engine
/// keeps prompts with vision pads out of the prefix cache
/// (`tokens_have_vision_pad`).
#[test]
fn test_vision_pad_tokens_are_image_blind_collision() {
    let tree = RadixTree::new();
    const IMAGE_PAD: u32 = 151655;

    let page_a: Vec<u32> = [1u32, 2, 3]
        .into_iter()
        .chain(std::iter::repeat_n(IMAGE_PAD, 29))
        .collect();
    // 2026-09-25: A different image with the same prompt and pad count has the
    // same token stream.
    let page_b = page_a.clone();

    tree.insert_with_snapshot(&page_a, &[10, 20], &[], 16, 77, 0, 0, 0);
    let m = tree.lookup(&page_b, 16, 0, 0);

    assert_eq!(
        m.matched_tokens, 32,
        "distinct images with identical prompt+pad token streams collide on \
         the same prefix-cache key (issue #58); the model layer must not admit \
         vision prefills into the radix cache"
    );
    assert_eq!(m.matched_blocks, vec![10, 20]);
    assert_eq!(
        m.ssm_snapshot,
        Some(77),
        "page B also restores page A's SSM snapshot"
    );
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&page_b, 16, 0);
}

/// 2026-09-25: The cache owns one KV ref on a `partial_suffix` block, as on a
/// full block: `insert` reports it in `blocks`, and `evict` hands it back
/// with its node. Replacing the slot's block reports the old one in
/// `released_blocks`.
#[test]
fn test_partial_suffix_block_is_owned_and_released() {
    let tree = RadixTree::new();
    // 2026-09-25: 20 tokens at block size 16: full block 10 and a 4-token
    // partial block 11.
    let tokens: Vec<u32> = (0..20).collect();
    let acquired = tree.insert(&tokens, &[10, 11], &[], 16, 0, 0);
    // 2026-09-25: Exact equality, not `contains`: a block reported twice
    // would take a second ref that the cache never gives back.
    assert_eq!(
        acquired.blocks,
        vec![10, 11],
        "one ref on the full block and one on the partial-suffix block"
    );
    assert!(acquired.released_blocks.is_empty());

    let again = tree.insert(&tokens, &[10, 11], &[], 16, 0, 0);
    assert!(
        again.blocks.is_empty(),
        "re-insert creates no node and re-refs no block; got {:?}",
        again.blocks
    );

    let other: Vec<u32> = (0..16).chain(90..94).collect();
    let swapped = tree.insert(&other, &[10, 12], &[], 16, 0, 0);
    assert_eq!(
        swapped.blocks,
        vec![12],
        "only the new partial block is acquired"
    );
    assert_eq!(
        swapped.released_blocks,
        vec![11],
        "displaced partial block released"
    );
}
