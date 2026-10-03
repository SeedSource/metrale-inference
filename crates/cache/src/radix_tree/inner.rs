// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The radix tree's node arena, behind the `Mutex` in
//! [`super::RadixTree`]: walk, ref counting and insert. Token sequences are
//! chunked at `block_size`; each non-root node holds one physical KV block.
//!
//! Owner: cache.
//! Invariants:
//! - Root nodes and freed nodes carry `block_idx == u32::MAX` and no parent.
//! - `walk` returns a node's block only if the node's `ref_count > 0` and its
//!   context hash matches the path from the root.

use std::collections::HashMap;

mod evict;

type NodeId = usize;

pub(super) struct RadixNode {
    /// 2026-09-25: Children keyed by their `block_size`-token chunk.
    children: HashMap<Vec<u32>, NodeId>,
    /// 2026-09-25: Physical KV block stored at this node.
    block_idx: u32,
    /// 2026-09-25: `--high-speed-swap` disk-block id, `u32::MAX` when none.
    /// The cache owns one disk ref on it: `insert` reports it in
    /// `InsertAcquired::disk_block_ids`, and `evict` returns it.
    disk_block_id: u32,
    /// 2026-09-25: FNV-1a over this node's tokens, seeded with the parent's
    /// context hash; `walk` reuses a node only if it matches, so the whole
    /// prefix must match, not only this chunk.
    context_hash: u64,
    /// 2026-09-25: Counts the cache's own ref (set to 1 by insert) plus one
    /// per sequence holding the node through `inc_refs` or an insert past its
    /// matched prefix. `walk` skips a node at 0; eviction takes leaves at 1 or
    /// less.
    ref_count: u32,
    /// 2026-09-25: Value of the access counter at the last insert or
    /// `inc_refs`, for LRU eviction.
    last_access: u64,
    /// 2026-09-25: `None` for a root and for a freed node.
    parent: Option<NodeId>,
    /// 2026-09-25: The chunk this node is keyed by in its parent, so eviction
    /// can unlink it.
    parent_key: Option<Vec<u32>>,
    /// 2026-09-25: A cached sequence's final, incomplete block:
    /// `(partial_tokens, block_idx, disk_block_id)` with
    /// `partial_tokens.len() < block_size`. `walk` considers it only with
    /// `partial_tail_sharing`.
    partial_suffix: Option<(Vec<u32>, u32, u32)>,
}

/// 2026-09-25: The child's context hash: FNV-1a over `tokens`, seeded with
/// the parent's hash.
fn context_hash_combine(parent_hash: u64, tokens: &[u32]) -> u64 {
    let mut h = parent_hash;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 2026-09-25: The node arena and the per-adapter roots.
pub(super) struct RadixTreeInner {
    nodes: Vec<RadixNode>,
    /// 2026-09-25: Freed node indices, reused by `alloc_node`.
    free_nodes: Vec<NodeId>,
    /// 2026-09-25: One root per `adapter_id`; 0 is the base model's, created
    /// in `new()`. Disjoint subtrees keep one adapter's walk from reaching
    /// another's blocks, and its insert from reusing another's node: children
    /// are keyed by token chunk alone, so a hash seed would not prevent that.
    roots: HashMap<u64, NodeId>,
    access_counter: u64,
    /// 2026-09-25: Whether `walk` may end a match inside a block, handing the
    /// requester a block another owner still writes (see
    /// [`super::partial_tail`]). Set in `new()` from
    /// `METRALE_PREFIX_SUBBLOCK`, and a field rather than a read at the match
    /// site so a test can set it on one tree.
    pub(super) partial_tail_sharing: bool,
}

impl RadixTreeInner {
    pub(super) fn new() -> Self {
        let root = RadixNode {
            children: HashMap::new(),
            block_idx: u32::MAX,
            disk_block_id: u32::MAX,
            context_hash: 0,
            ref_count: 0,
            last_access: 0,
            parent: None,
            parent_key: None,
            partial_suffix: None,
        };
        let mut roots = HashMap::new();
        roots.insert(0u64, 0usize);
        Self {
            nodes: vec![root],
            free_nodes: Vec::new(),
            roots,
            access_counter: 0,
            partial_tail_sharing: super::partial_tail::partial_tail_sharing_enabled(),
        }
    }

    pub(super) fn next_access(&mut self) -> u64 {
        self.access_counter += 1;
        self.access_counter
    }

    /// 2026-09-25: The root of `adapter_id`, or `None` if it has never
    /// inserted (then `walk` returns no match and `inc_refs`/`dec_refs` do
    /// nothing).
    fn root_for_read(&self, adapter_id: u64) -> Option<NodeId> {
        self.roots.get(&adapter_id).copied()
    }

    /// 2026-09-25: The root of `adapter_id`, created on its first insert. A
    /// root has `block_idx == u32::MAX` and no parent, so eviction and
    /// `num_entries` skip it.
    fn root_for_insert(&mut self, adapter_id: u64) -> NodeId {
        if let Some(&id) = self.roots.get(&adapter_id) {
            return id;
        }
        let id = self.alloc_node(RadixNode {
            children: HashMap::new(),
            block_idx: u32::MAX,
            disk_block_id: u32::MAX,
            context_hash: 0,
            ref_count: 0,
            last_access: 0,
            parent: None,
            parent_key: None,
            partial_suffix: None,
        });
        self.roots.insert(adapter_id, id);
        id
    }

    pub(super) fn alloc_node(&mut self, node: RadixNode) -> NodeId {
        if let Some(id) = self.free_nodes.pop() {
            self.nodes[id] = node;
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(node);
            id
        }
    }

    /// 2026-09-25: Match `tokens` chunk by chunk from the adapter's root.
    /// Returns `(matched_blocks, matched_disk_block_ids, matched_tokens)`; the
    /// disk ids parallel the blocks, `u32::MAX` where a node has none. Takes
    /// no refs; `inc_refs` does.
    pub(super) fn walk(
        &self,
        tokens: &[u32],
        block_size: usize,
        adapter_id: u64,
    ) -> (Vec<u32>, Vec<u32>, usize) {
        let mut current = match self.root_for_read(adapter_id) {
            Some(r) => r,
            None => return (Vec::new(), Vec::new(), 0),
        };
        let mut matched_blocks = Vec::new();
        let mut matched_disk = Vec::new();
        let mut matched_tokens = 0;
        let mut parent_ctx_hash: u64 = 0;

        let num_full_blocks = tokens.len() / block_size;
        for i in 0..num_full_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            let expected_hash = context_hash_combine(parent_ctx_hash, chunk);
            match self.nodes[current].children.get(chunk) {
                Some(&child)
                    if self.nodes[child].context_hash == expected_hash
                        && self.nodes[child].ref_count > 0 =>
                {
                    matched_blocks.push(self.nodes[child].block_idx);
                    matched_disk.push(self.nodes[child].disk_block_id);
                    matched_tokens += block_size;
                    parent_ctx_hash = expected_hash;
                    current = child;
                }
                _ => break,
            }
        }

        // 2026-09-25: Sub-block tail, only with `partial_tail_sharing` and
        // only after every full block matched. First a child whose key starts
        // with the remaining tokens, then this node's `partial_suffix`. Either
        // gives a `matched_tokens` that is not block-aligned and hands out a
        // block another owner still has; `super::partial_tail` says why that
        // is off by default.
        let remainder = tokens.len() - matched_tokens;
        if self.partial_tail_sharing
            && remainder > 0
            && remainder < block_size
            && matched_tokens == num_full_blocks * block_size
        {
            let suffix = &tokens[matched_tokens..];
            let mut found = false;
            for (key, &child_id) in &self.nodes[current].children {
                if key.len() >= suffix.len() && &key[..suffix.len()] == suffix {
                    let expected = context_hash_combine(parent_ctx_hash, key);
                    if self.nodes[child_id].context_hash == expected
                        && self.nodes[child_id].ref_count > 0
                    {
                        matched_blocks.push(self.nodes[child_id].block_idx);
                        matched_disk.push(self.nodes[child_id].disk_block_id);
                        matched_tokens += remainder;
                        found = true;
                        break;
                    }
                }
            }
            if !found
                && let Some((ref partial_toks, partial_block, partial_disk)) =
                    self.nodes[current].partial_suffix
                && partial_toks.len() >= suffix.len()
                && &partial_toks[..suffix.len()] == suffix
            {
                // 2026-09-25: A partial suffix has no context hash or ref count
                // of its own; it is reached only through a fully matched chain.
                matched_blocks.push(partial_block);
                matched_disk.push(partial_disk);
                matched_tokens += remainder;
            }
        }

        (matched_blocks, matched_disk, matched_tokens)
    }

    /// 2026-09-25: Add one ref to each node on the path of the first
    /// `num_matched / block_size` full chunks, and mark them accessed.
    pub(super) fn inc_refs(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        num_matched: usize,
        adapter_id: u64,
    ) {
        let access = self.next_access();
        let mut current = match self.root_for_read(adapter_id) {
            Some(r) => r,
            None => return,
        };
        let num_blocks = num_matched / block_size;

        for i in 0..num_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            if let Some(&child) = self.nodes[current].children.get(chunk) {
                self.nodes[child].ref_count += 1;
                self.nodes[child].last_access = access;
                current = child;
            } else {
                break;
            }
        }
    }

    /// 2026-09-25: Drop one ref from each node on the path of the first
    /// `matched_tokens / block_size` full chunks; a count at 0 stays 0.
    pub(super) fn dec_refs(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    ) {
        let mut current = match self.root_for_read(adapter_id) {
            Some(r) => r,
            None => return,
        };
        let num_full_blocks = (matched_tokens / block_size).min(tokens.len() / block_size);

        for i in 0..num_full_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            if let Some(&child) = self.nodes[current].children.get(chunk) {
                if self.nodes[child].ref_count > 0 {
                    self.nodes[child].ref_count -= 1;
                }
                current = child;
            } else {
                break;
            }
        }
    }

    /// 2026-09-25: Insert the full chunks of `tokens` with their blocks from
    /// `block_table`; a chunk that already has a node keeps that node's
    /// block, except past `matched_tokens`, where the node adopts the
    /// sequence's block (2026-10-03, race #79). A final incomplete chunk becomes the last node's
    /// `partial_suffix` when `block_table` has its block and at least one full
    /// chunk precedes it. Returns what the caller must now reference and
    /// release (`InsertAcquired`).
    ///
    /// `matched_tokens` is the prefix the sequence already holds through
    /// `lookup`'s `inc_refs`. Nodes past it get one extra ref for the
    /// sequence, because `release` drops a ref on every node up to
    /// `tokens.len()`; without it the cache's own ref would go and `walk`
    /// would skip the node.
    pub(super) fn insert(
        &mut self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    ) -> metrale_telemetry::prefix_cache::InsertAcquired {
        let access = self.next_access();
        let root_id = self.root_for_insert(adapter_id);
        let mut current = root_id;
        let mut parent_ctx_hash: u64 = 0;
        let num_full_blocks = tokens.len() / block_size;
        let num_blocks = num_full_blocks.min(block_table.len());
        // 2026-09-25: `disk_block_ids` is empty or parallel to `block_table`;
        // empty leaves every new node's disk id at `u32::MAX`.
        let hss_active = !disk_block_ids.is_empty();
        debug_assert!(
            !hss_active || disk_block_ids.len() == block_table.len(),
            "disk_block_ids length mismatch: {} vs {}",
            disk_block_ids.len(),
            block_table.len(),
        );

        // 2026-09-25: Disk ids the cache newly takes a ref on, for the caller
        // to `inc_disk_ref`: a new node with a real id, an existing node going
        // from `u32::MAX` to a real id, and a new partial-suffix id. An id the
        // node already holds is not reported again.
        let mut newly_acquired: Vec<u32> = Vec::new();
        // 2026-09-25: Blocks the cache starts and stops storing; the caller
        // takes and releases one KV ref on each (`InsertAcquired`).
        let mut newly_owned_blocks: Vec<u32> = Vec::new();
        let mut released_blocks: Vec<u32> = Vec::new();

        for i in 0..num_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            let ctx_hash = context_hash_combine(parent_ctx_hash, chunk);
            let token_start = i * block_size;
            let is_seq_owned = token_start >= matched_tokens;
            let disk_id = if hss_active {
                disk_block_ids[i]
            } else {
                u32::MAX
            };

            if let Some(&child) = self.nodes[current].children.get(chunk) {
                // 2026-09-25: Existing node: mark it accessed, and restore the
                // cache's ref if the count fell to 0.
                self.nodes[child].last_access = access;
                self.nodes[child].context_hash = ctx_hash;
                if self.nodes[child].ref_count == 0 {
                    self.nodes[child].ref_count = 1;
                }
                if is_seq_owned {
                    self.nodes[child].ref_count += 1;
                    // 2026-10-03: The sequence computed this chunk into its own block, and its
                    // ref (just taken) pins the node until it is released, so a node keeping
                    // its older block would hold a second, unusable copy of the same K/V for
                    // the sequence's whole life. Race #79: a 196K prompt recomputed from 0
                    // over a 131K cached prompt left 1,898 such blocks pinned and decode ran
                    // out of KV. The node adopts the sequence's block, and the old one is
                    // reported for release (a sequence that matched it holds its own ref).
                    // Not with high-speed-swap, whose disk id pairs with the node's block.
                    let seq_block = block_table[i];
                    if !hss_active
                        && seq_block != u32::MAX
                        && self.nodes[child].block_idx != seq_block
                    {
                        released_blocks.push(self.nodes[child].block_idx);
                        newly_owned_blocks.push(seq_block);
                        self.nodes[child].block_idx = seq_block;
                    }
                }
                // 2026-09-25: A node without a disk id takes this insert's.
                if hss_active && self.nodes[child].disk_block_id == u32::MAX && disk_id != u32::MAX
                {
                    self.nodes[child].disk_block_id = disk_id;
                    newly_acquired.push(disk_id);
                }
                parent_ctx_hash = ctx_hash;
                current = child;
            } else {
                // 2026-09-25: A new child replaces this node's partial slot.
                // Its block is reported for release; its disk id is not.
                released_blocks.extend(self.nodes[current].partial_suffix.take().map(|p| p.1));
                let node = RadixNode {
                    children: HashMap::new(),
                    block_idx: block_table[i],
                    disk_block_id: disk_id,
                    context_hash: ctx_hash,
                    ref_count: if is_seq_owned { 2 } else { 1 },
                    last_access: access,
                    parent: Some(current),
                    parent_key: Some(chunk.to_vec()),
                    partial_suffix: None,
                };
                let child_id = self.alloc_node(node);
                newly_owned_blocks.push(block_table[i]);
                self.nodes[current]
                    .children
                    .insert(chunk.to_vec(), child_id);
                if hss_active && disk_id != u32::MAX {
                    newly_acquired.push(disk_id);
                }
                parent_ctx_hash = ctx_hash;
                current = child_id;
            }
        }

        let remainder = tokens.len() % block_size;
        if remainder > 0 && block_table.len() > num_full_blocks && current != root_id {
            let partial_toks = tokens[num_full_blocks * block_size..].to_vec();
            let partial_block = block_table[num_full_blocks];
            let partial_disk = if hss_active && disk_block_ids.len() > num_full_blocks {
                disk_block_ids[num_full_blocks]
            } else {
                u32::MAX
            };
            // 2026-09-25: The new partial slot replaces any prior one. A
            // replaced block is reported for release; a replaced disk id is
            // not reported anywhere, so the cache's ref on it is dropped
            // without a `dec_disk_ref`.
            let prior = self.nodes[current].partial_suffix.as_ref().map(|p| p.2);
            // 2026-09-25: The cache holds a KV ref on the partial block as on a
            // node's block, since `evict` returns it too.
            let prior_block = self.nodes[current].partial_suffix.as_ref().map(|p| p.1);
            self.nodes[current].partial_suffix = Some((partial_toks, partial_block, partial_disk));
            if prior_block != Some(partial_block) {
                newly_owned_blocks.push(partial_block);
                released_blocks.extend(prior_block);
            }
            if hss_active && partial_disk != u32::MAX && prior != Some(partial_disk) {
                newly_acquired.push(partial_disk);
            }
        }

        metrale_telemetry::prefix_cache::InsertAcquired {
            disk_block_ids: newly_acquired,
            blocks: newly_owned_blocks,
            released_blocks,
        }
    }
}
