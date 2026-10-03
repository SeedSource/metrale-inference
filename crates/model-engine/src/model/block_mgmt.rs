// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: KV block allocation for one sequence (decode and prefill, with the
//! high-speed-swap window), the prefix cache's eviction and ref handoff, and the
//! per-layer state borrow used by batched forwards.
//!
//! Owner: model-engine.
//! Invariants:
//! - A block is filled (zeroed, or 0xFF under `kv_poison`) before it is pushed
//!   onto `block_table`.
//! - Only the decode path slides the window, and only after
//!   `check_safe_to_evict` has passed for the block it frees.

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// 2026-09-25: Fill a freshly allocated block with 0xFF bytes (a NaN pattern)
/// when `poison` is set, otherwise with zeros, so attention reads past
/// `seq_len` never see another sequence's KV.
#[inline]
fn fill_fresh_block(
    kv_cache: &PagedKvCache,
    blk: u32,
    gpu: &dyn GpuBackend,
    stream: u64,
    // 2026-09-25: `ModelLevers::kv_poison`, passed per call: it changes what
    // the attention kernels read, so it must not carry over to another model.
    poison: bool,
) -> Result<()> {
    if poison {
        kv_cache.poison_block(blk, gpu, stream)
    } else {
        kv_cache.zero_block(blk, gpu, stream)
    }
}

/// 2026-09-25: Apply an `EvictedBlocks` result. Each physical block loses the
/// prefix cache's one reference (`return_evicted_block`) and reaches the free
/// list only if no sequence still holds it. Each disk-block ID is
/// `dec_disk_ref`'d when a high-speed-swap instance is installed on this thread.
pub(crate) fn apply_evicted_blocks(
    evicted: metrale_telemetry::prefix_cache::EvictedBlocks,
    kv_cache: &mut PagedKvCache,
) {
    let free_before = kv_cache.num_free_blocks();
    let n_evicted = evicted.physical.len();
    for block in &evicted.physical {
        kv_cache.return_evicted_block(*block);
    }
    // 2026-09-25: An evicted node hands back exactly one ref, so a block that a
    // live sequence also holds survives the eviction and is freed later by that
    // sequence. `gained < n_evicted` is therefore normal under load, not a leak.
    let gained = kv_cache.num_free_blocks().saturating_sub(free_before);
    if gained < n_evicted {
        tracing::debug!(
            "prefix-cache evict reclaimed {gained}/{n_evicted} blocks (free={}): \
             the rest are still held by live sequences and free with them",
            kv_cache.num_free_blocks(),
        );
    }
    if !evicted.disk_block_ids.is_empty()
        && let Some(res) = metrale_storage::with_local(|hss| {
            for id in &evicted.disk_block_ids {
                let _new_refcount = hss.dec_disk_ref(*id);
            }
            Ok(())
        })
        && let Err(e) = res
    {
        tracing::debug!("apply_evicted_blocks: metrale_storage::with_local closure: {e:#}");
    }
}

/// 2026-09-25: Allocate one block, evicting from the prefix cache until one
/// comes free.
///
/// Evicting a radix node releases only the cache's reference to its block, so
/// when a live sequence still holds that block one `evict(1)` frees nothing.
/// Stopping after one eviction would fail the allocation while evictable
/// entries remain, with this log pair:
///
/// ```text
/// DEBUG prefix-cache evict reclaimed 0/1 blocks (free=0)
/// ERROR alloc failed in ensure_blocks_through_decode: ... free_blocks=0 ...
/// ```
///
/// Each non-empty eviction removes a leaf node from the tree, so the loop
/// terminates. `None` means no block is free and nothing is left to evict.
pub(crate) fn alloc_block_evicting(
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn metrale_telemetry::prefix_cache::PrefixCache,
) -> Option<u32> {
    if let Some(b) = kv_cache.try_alloc_block() {
        return Some(b);
    }
    let mut evicted_nodes = 0usize;
    loop {
        let evicted = prefix_cache.evict(1);
        if evicted.is_empty() {
            if evicted_nodes > 0 {
                tracing::debug!(
                    "alloc: evicted {evicted_nodes} prefix-cache node(s) without freeing a \
                     block (every one is still held by a live sequence) — out of capacity"
                );
            }
            return None;
        }
        evicted_nodes += evicted.len();
        apply_evicted_blocks(evicted, kv_cache);
        if let Some(b) = kv_cache.try_alloc_block() {
            if evicted_nodes > 1 {
                tracing::debug!(
                    "alloc: freed a block after evicting {evicted_nodes} prefix-cache node(s)"
                );
            }
            return Some(b);
        }
    }
}

/// 2026-09-25: Apply the disk-ref obligation reported by a `prefix_cache.insert*`
/// call. The cache returns the disk-block IDs on which it newly took an
/// ownership ref (a node was created, or an existing node had its disk-block ID
/// set for the first time). This takes one `inc_disk_ref` on each, skipping
/// `u32::MAX`, so the swap allocator's refcount matches the cache's reachability.
///
/// Without it, the only live ref on such an ID is the sequence's. When
/// `free_sequence` drops that ref the allocator frees the ID while the cache
/// still stores it, and the next prefix hit's `inc_disk_ref` panics on the
/// freed ID.
pub(crate) fn cache_acquires_disk_refs(newly_acquired: &[u32]) {
    if newly_acquired.is_empty() {
        return;
    }
    if let Some(res) = metrale_storage::with_local(|hss| {
        for &id in newly_acquired {
            if id != u32::MAX {
                hss.inc_disk_ref(id);
            }
        }
        Ok(())
    }) && let Err(e) = res
    {
        tracing::debug!("cache_acquires_disk_refs: metrale_storage::with_local: {e:#}");
    }
}

/// 2026-09-25: Apply both ref obligations reported by a `prefix_cache.insert*`
/// call: the disk-side refs, one KV `inc_ref` on each block the insert started
/// storing, and one KV `dec_ref` on each block it stopped storing. See
/// `InsertAcquired`.
///
/// The KV refs follow the blocks the cache stored, not the sequence's
/// `block_table`: a node that already existed keeps its original block, so the
/// sequence's block at that position can be a different one. A ref taken on
/// the sequence's block would leave the node's block without the cache's ref,
/// and evicting that node would then release a live sequence's ref instead.
pub(crate) fn cache_acquires_refs(
    acquired: &metrale_telemetry::prefix_cache::InsertAcquired,
    kv_cache: &mut PagedKvCache,
) {
    cache_acquires_disk_refs(&acquired.disk_block_ids);
    // 2026-09-25: Acquire before release, so a block that is in both lists
    // never drops to 0 refs and lands on the free list in between.
    for &block in &acquired.blocks {
        kv_cache.inc_ref(block);
    }
    for &block in &acquired.released_blocks {
        kv_cache.dec_ref(block);
    }
}

/// 2026-09-25: Take the sequence's disk ref on each disk-block ID reused from a
/// prefix-cache hit and append the ID to `seq_disk_block_ids`, so
/// `free_sequence` can release it. `u32::MAX` entries have no disk copy and are
/// skipped. Does nothing when no high-speed-swap instance is installed.
pub(crate) fn reuse_prefix_match_disk_ids(
    matched_disk_block_ids: &[u32],
    seq_disk_block_ids: &mut Vec<u32>,
) {
    if matched_disk_block_ids.is_empty() {
        return;
    }
    if let Some(res) = metrale_storage::with_local(|hss| {
        for &id in matched_disk_block_ids {
            if id == u32::MAX {
                continue;
            }
            hss.inc_disk_ref(id);
            seq_disk_block_ids.push(id);
        }
        Ok(())
    }) && let Err(e) = res
    {
        tracing::debug!("reuse_prefix_match_disk_ids: metrale_storage::with_local: {e:#}");
    }
}

/// 2026-09-25: Check that the block at logical position `evict_pos` has been
/// offloaded by every attention layer: `disk_last_offloaded[L] > evict_pos` for
/// all L, strictly greater because each entry counts offloaded blocks.
///
/// Returns `Err` naming the first lagging layer, else `Ok(())`. No side effects.
pub(crate) fn check_safe_to_evict(
    disk_last_offloaded_per_layer: &[u32],
    evict_pos: usize,
) -> Result<()> {
    for (layer_idx, &cursor) in disk_last_offloaded_per_layer.iter().enumerate() {
        if (cursor as usize) <= evict_pos {
            bail!(
                "high-speed-swap: attempting to evict block at logical position {} \
                 from HBM, but attention layer {} only offloaded up to position {}. \
                 Eviction would lose K/V data. Per-layer cursors: {:?}",
                evict_pos,
                layer_idx,
                cursor,
                disk_last_offloaded_per_layer,
            );
        }
    }
    Ok(())
}

/// 2026-09-25: Raise every attention layer's offload cursor to at least
/// `new_window_start`, the window start after a slide. Cursors already at or
/// above it are left alone.
pub(crate) fn advance_layer_cursors_after_slide(
    disk_last_offloaded_per_layer: &mut [u32],
    new_window_start: usize,
) {
    let new_ws = new_window_start as u32;
    for cursor in disk_last_offloaded_per_layer.iter_mut() {
        if *cursor < new_ws {
            *cursor = new_ws;
        }
    }
}

/// 2026-09-25: Grow `seq.block_table` (decode path) until logical block
/// `abs_block_idx` is inside the window, i.e.
/// `abs_block_idx < hss_window_start() + block_table.len()` on `Ok`.
///
/// Each new block comes from `alloc_block_evicting` and is filled by
/// `fill_fresh_block`. With `cache_blocks_per_seq` set (high-speed swap), a
/// full window first slides: `block_table[0]` is freed once
/// `check_safe_to_evict` confirms every attention layer offloaded it (a failed
/// check returns `Err`), and every new block gets a disk-block ID pushed onto
/// `seq.disk_block_ids`.
pub(crate) fn ensure_blocks_through_decode(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn metrale_telemetry::prefix_cache::PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
    kv_poison: bool,
) -> Result<()> {
    let bs = kv_cache.config().block_size;
    grow_blocks_decode(
        seq,
        abs_block_idx,
        kv_cache,
        prefix_cache,
        gpu,
        stream,
        kv_poison,
    )?;
    map_lazy_rows_through(seq, abs_block_idx, bs)
}

/// 2026-10-03: With `METRALE_DSA_INDEXER_LAZY=1`, back every lazily mapped per-position state
/// of `seq` (the DSA indexer caches and the proposer's) for positions `[0, (abs_block_idx + 1)
/// * block_size)`: every position the step can write, because the KV write for a position
/// needs its block first. It runs here, at step entry, before any capture or replay, and its
/// extents depend only on token positions, so EP ranks map identically. A full indexer pool
/// fails with "KV cache exhausted", which the scheduler's decode path answers by preempting a
/// sequence. Lever off: returns at once.
fn map_lazy_rows_through(
    seq: &SequenceState,
    abs_block_idx: usize,
    block_size: usize,
) -> Result<()> {
    if !metrale_model_arch::glm5next_dsa::lazy::dsa_indexer_lazy() {
        return Ok(());
    }
    let end = abs_block_idx.saturating_add(1).saturating_mul(block_size);
    for st in &seq.layer_states {
        st.map_rows_through(end)?;
    }
    if let Some(p) = &seq.proposer_state {
        p.map_rows_through(end)?;
    }
    Ok(())
}

fn grow_blocks_decode(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn metrale_telemetry::prefix_cache::PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
    kv_poison: bool,
) -> Result<()> {
    let cap = kv_cache.config().cache_blocks_per_seq.map(|c| c as usize);
    // 2026-09-25: Each iteration returns, slides (the window end stays put) or
    // allocates one block (the window end rises by one), so the loop ends.
    let mut slide_count = 0usize;
    let mut alloc_count = 0usize;
    loop {
        let ws = seq.hss_window_start();
        let bt_len = seq.block_table.len();
        let in_window = bt_len > 0 && abs_block_idx < ws + bt_len;
        if in_window {
            if slide_count > 0 || alloc_count > 0 {
                tracing::trace!(
                    "ensure_blocks_through_decode: abs={} ws={} bt_len={} slid={} alloc'd={}",
                    abs_block_idx,
                    ws,
                    bt_len,
                    slide_count,
                    alloc_count
                );
            }
            return Ok(());
        }
        if let Some(c) = cap
            && bt_len >= c
        {
            // 2026-09-25: `block_table[0]` is logical position `ws`; it may be
            // freed only once every attention layer has offloaded it.
            check_safe_to_evict(&seq.disk_last_offloaded_per_layer, ws).map_err(|e| {
                anyhow::anyhow!(
                    "{e} (decode path; disk_block_ids.len()={}, block_table.len()={}, \
                     slid={}, alloc'd={})",
                    seq.disk_block_ids.len(),
                    seq.block_table.len(),
                    slide_count,
                    alloc_count,
                )
            })?;
            let evicted = seq.block_table.remove(0);
            kv_cache.free_block(evicted);
            advance_layer_cursors_after_slide(&mut seq.disk_last_offloaded_per_layer, ws + 1);
            slide_count += 1;
            continue;
        }
        let blk = match alloc_block_evicting(kv_cache, prefix_cache) {
            Some(b) => b,
            None => {
                return Err(anyhow::anyhow!(
                    "alloc failed in ensure_blocks_through_decode: abs={} ws={} bt_len={} \
                     cap={:?} free_blocks={} slid={} alloc'd={}: {}",
                    abs_block_idx,
                    ws,
                    bt_len,
                    cap,
                    kv_cache.num_free_blocks(),
                    slide_count,
                    alloc_count,
                    "KV cache exhausted: no free blocks"
                ));
            }
        };
        fill_fresh_block(kv_cache, blk, gpu, stream, kv_poison)?;
        seq.block_table.push(blk);
        alloc_count += 1;
        if cap.is_some() {
            let id = metrale_storage::with_local(|hss| {
                hss.alloc_disk_block_id().ok_or_else(|| {
                    anyhow::anyhow!(
                        "high-speed-swap: disk-block-id pool exhausted; \
                         increase --high-speed-swap-bytes or shorten --max-seq-len"
                    )
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "high-speed-swap: orchestrator not installed but cache_blocks_per_seq is set"
                )
            })??;
            seq.disk_block_ids.push(id);
        }
    }
}

/// 2026-09-25: Grow `seq.block_table` (prefill path) until logical block
/// `abs_block_idx` is inside the window, allocating as
/// `ensure_blocks_through_decode` does but never sliding.
///
/// Only decode attention reads slid-out blocks back from disk
/// (`attend_layer_on_stream`); prefill attention reads `block_table` alone, so a
/// slide here would drop K/V the prefill still attends to. The
/// `cache_blocks_per_seq` cap is therefore applied by the first
/// `ensure_blocks_through_decode` call that finds `bt_len >= cap`.
pub(crate) fn ensure_blocks_through_prefill(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn metrale_telemetry::prefix_cache::PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
    kv_poison: bool,
) -> Result<()> {
    let cap = kv_cache.config().cache_blocks_per_seq.map(|c| c as usize);
    let bs = kv_cache.config().block_size;
    loop {
        let ws = seq.hss_window_start();
        let bt_len = seq.block_table.len();
        let in_window = bt_len > 0 && abs_block_idx < ws + bt_len;
        if in_window {
            return map_lazy_rows_through(seq, abs_block_idx, bs);
        }
        let blk = alloc_block_evicting(kv_cache, prefix_cache)
            .ok_or_else(|| anyhow::anyhow!("KV cache exhausted: no free blocks"))?;
        fill_fresh_block(kv_cache, blk, gpu, stream, kv_poison)?;
        seq.block_table.push(blk);
        if cap.is_some() {
            let id = metrale_storage::with_local(|hss| {
                hss.alloc_disk_block_id().ok_or_else(|| {
                    anyhow::anyhow!(
                        "high-speed-swap: disk-block-id pool exhausted; \
                         increase --high-speed-swap-bytes or shorten --max-seq-len"
                    )
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "high-speed-swap: orchestrator not installed but cache_blocks_per_seq is set"
                )
            })??;
            seq.disk_block_ids.push(id);
        }
    }
}

/// 2026-09-25: Borrow layer `layer_idx`'s state mutably from each sequence, in
/// the order of `all`. Panics if a sequence has fewer than `layer_idx + 1`
/// layer states.
pub(crate) fn extract_layer_refs<'a>(
    all: &'a mut [Vec<Box<dyn LayerState>>],
    layer_idx: usize,
) -> Vec<&'a mut (dyn LayerState + 'static)> {
    let mut refs = Vec::with_capacity(all.len());
    for seq_states in all.iter_mut() {
        refs.push(seq_states[layer_idx].as_mut());
    }
    refs
}

#[cfg(test)]
#[path = "block_mgmt_tests.rs"]
mod tests;
