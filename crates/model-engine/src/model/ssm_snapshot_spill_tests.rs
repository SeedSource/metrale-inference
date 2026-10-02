// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `spill_slot`, `fault_in_slot`, `spill_blob_bytes` and
//! `acquire_or_spill_slot` on a `MockGpuBackend` with a host-RAM `MemBlobStore`.
//!
//! Owner: model-engine SSM snapshot pool.
//! Invariants: none beyond the types.

use super::*;
use crate::model::ssm_tier::{MemBlobStore, SnapshotBlobStore};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-09-25: A small Marconi-only pool, with no decode-rollback ring.
fn pool(gpu: &dyn GpuBackend, slots: usize, layers: usize) -> SsmSnapshotPool {
    SsmSnapshotPool::new(
        slots, 32, 16, layers, 0,
        // 2026-09-25: `spill_blob_bytes` counts only h and conv, so the hidden size is
        // not part of a blob.
        0, 8, gpu,
    )
    .unwrap()
}

/// 2026-09-25: Fill slot `s`'s per-layer h and conv chunks with a byte unique per
/// (layer, field), so a misplaced scatter shows up.
fn write_pattern(p: &SsmSnapshotPool, gpu: &dyn GpuBackend, s: usize) {
    for i in 0..p.num_ssm_layers {
        let h = vec![(0x10 + i) as u8; p.h_bytes];
        let c = vec![(0x80 + i) as u8; p.conv_bytes];
        gpu.copy_h2d(&h, p.h_snapshots[i].offset(s * p.h_bytes))
            .unwrap();
        gpu.copy_h2d(&c, p.conv_snapshots[i].offset(s * p.conv_bytes))
            .unwrap();
    }
}

fn read_slot(p: &SsmSnapshotPool, gpu: &dyn GpuBackend, s: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut hs = Vec::new();
    let mut cs = Vec::new();
    for i in 0..p.num_ssm_layers {
        let mut h = vec![0u8; p.h_bytes];
        let mut c = vec![0u8; p.conv_bytes];
        gpu.copy_d2h(p.h_snapshots[i].offset(s * p.h_bytes), &mut h)
            .unwrap();
        gpu.copy_d2h(p.conv_snapshots[i].offset(s * p.conv_bytes), &mut c)
            .unwrap();
        hs.push(h);
        cs.push(c);
    }
    (hs, cs)
}

/// 2026-09-25: The gather is `2 * layers` async enqueues between one leading and one
/// trailing synchronise, with no blocking `copy_d2h`. A blocking copy per chunk gives
/// the same bytes, so only these counts catch it.
#[test]
fn spill_issues_exactly_one_trailing_sync() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 2, 5);
    let store = MemBlobStore::new(0);
    write_pattern(&p, &gpu, 0);

    const STREAM: u64 = 0x5a17;
    let syncs_before = gpu.sync_count();
    let d2h_before = gpu.d2h_blocking_count();
    assert!(p.spill_slot(0, 0x11, &store, &gpu, STREAM).unwrap());

    assert_eq!(
        gpu.d2h_async_count(),
        2 * p.num_ssm_layers,
        "every (h,conv) chunk must be an async enqueue"
    );
    assert_eq!(
        gpu.d2h_blocking_count() - d2h_before,
        0,
        "a blocking copy_d2h in the gather is the defect itself"
    );
    assert_eq!(
        gpu.sync_count() - syncs_before,
        2,
        "exactly two drains: the leading save-drain and ONE trailing commit"
    );
    assert_eq!(gpu.d2h_async_streams(), vec![STREAM; 2 * p.num_ssm_layers]);
    assert_eq!(
        gpu.sync_d2h_async_counts(),
        [(STREAM, 0), (STREAM, 2 * p.num_ssm_layers)],
        "the save drain must precede every enqueue and the commit must follow all of them"
    );
}

/// 2026-09-25: Spill and fault-in share one staging buffer, allocated once.
#[test]
fn staging_buffer_allocated_once() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 3, 4);
    let store = MemBlobStore::new(0);
    write_pattern(&p, &gpu, 0);
    write_pattern(&p, &gpu, 1);

    assert!(p.spill_slot(0, 0xA, &store, &gpu, 0).unwrap());
    assert!(p.spill_slot(1, 0xB, &store, &gpu, 0).unwrap());
    assert!(p.fault_in_slot(2, 0xA, &store, &gpu, 0).unwrap());
    assert_eq!(
        gpu.host_pinned_alloc_count(),
        1,
        "one buffer, shared by spill and fault-in, for the model's lifetime"
    );
    p.free_staging(&gpu);
}

/// 2026-09-25: Faulting in an absent key returns `Ok(false)`, not an error.
#[test]
fn fault_in_absent_key_is_miss() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 4, 2);
    let store = MemBlobStore::new(0);
    assert!(!p.fault_in_slot(0, 999, &store, &gpu, 0).unwrap());
}

#[test]
fn spill_blob_bytes_matches_layout() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 2, 5);
    assert_eq!(p.spill_blob_bytes(), 5 * (32 + 16));
}

/// 2026-09-25: With no free slot, `acquire_or_spill_slot` spills a resident victim to
/// the store and returns the victim's slot.
#[test]
fn acquire_or_spill_frees_a_slot_under_full_pool() {
    use metrale_cache::radix_tree::RadixTree;
    use metrale_telemetry::prefix_cache::PrefixCache;

    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 2, 2);
    let store = MemBlobStore::new(0);
    let tree = RadixTree::new();

    // 2026-09-25: Two resident snapshots in slots 0 and 1, then an empty free list.
    // The prefixes are 2048 tokens, above the default spill gate of 1024
    // (`DEFAULT_SPILL_MIN_TOKENS`), so the victim is spilled; the dropped case is
    // `shallow_victim_is_dropped_but_still_yields_a_slot`.
    let toks_a: Vec<u32> = (0..2048).collect();
    let toks_b: Vec<u32> = (100_000..102_048).collect();
    tree.insert_with_snapshot(
        &toks_a,
        &[10],
        &[],
        16,
        // 2026-09-25: Either entry may be the victim: the assert below accepts slot 0
        // or 1.
        0,
        7,
        0,
        0,
    );
    tree.insert_with_snapshot(
        &toks_b,
        &[20],
        &[],
        16,
        // 2026-09-25: A prefix disjoint from `toks_a`, so the two snapshots are separate
        // index entries.
        1,
        9,
        0,
        0,
    );
    assert!(p.try_pop_free_slot().is_some());
    assert!(p.try_pop_free_slot().is_some());
    assert_eq!(p.try_pop_free_slot(), None, "pool is now full");

    let slot = p
        .acquire_or_spill_slot(&tree, &store, &gpu)
        .expect("a resident victim exists to spill");
    assert!(slot == 0 || slot == 1);
    assert_eq!(
        store.len(),
        1,
        "the evicted victim was spilled, not dropped"
    );
    assert!(tree.evict_snapshot_lru().is_some());
}

/// 2026-09-25: A victim below the spill gate is dropped, so nothing reaches the
/// store, and the caller still gets its slot.
#[test]
fn shallow_victim_is_dropped_but_still_yields_a_slot() {
    use metrale_cache::radix_tree::RadixTree;
    use metrale_telemetry::prefix_cache::PrefixCache;

    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 1, 2);
    let store = MemBlobStore::new(0);
    let tree = RadixTree::new();

    // 2026-09-25: 16 tokens, below the default spill gate of 1024.
    let toks: Vec<u32> = (0..16).collect();
    tree.insert_with_snapshot(
        &toks,
        &[10],
        &[],
        16,
        // 2026-09-25: The only resident snapshot: the pool has one slot.
        0,
        7,
        0,
        0,
    );
    assert!(p.try_pop_free_slot().is_some());
    assert_eq!(p.try_pop_free_slot(), None, "pool is now full");

    let slot = p
        .acquire_or_spill_slot(&tree, &store, &gpu)
        .expect("the gate must still free a slot");
    assert_eq!(slot, 0);
    assert_eq!(store.len(), 0, "a ~45ms spill cannot repay 16 tokens");
}

/// 2026-09-25: Blobs are keyed independently of the slot they came from: spill A from
/// slot 0, reuse slot 0 for B and spill it under another key, then fault both into
/// other slots; each recovers its own bytes.
#[test]
fn tier_survives_slot_recycling() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 3, 2);
    let store = MemBlobStore::new(0);
    let (key_a, key_b) = (0xAAAA, 0xBBBB);

    write_pattern(&p, &gpu, 0);
    let want_a = read_slot(&p, &gpu, 0);
    assert!(p.spill_slot(0, key_a, &store, &gpu, 0).unwrap());

    for i in 0..p.num_ssm_layers {
        let h = vec![0xEE; p.h_bytes];
        let c = vec![0xDD; p.conv_bytes];
        gpu.copy_h2d(&h, p.h_snapshots[i].offset(0)).unwrap();
        gpu.copy_h2d(&c, p.conv_snapshots[i].offset(0)).unwrap();
    }
    let want_b = read_slot(&p, &gpu, 0);
    assert_ne!(want_a, want_b, "B must differ from A for the test to bite");
    assert!(p.spill_slot(0, key_b, &store, &gpu, 0).unwrap());
    assert_eq!(store.len(), 2);

    assert!(p.fault_in_slot(1, key_a, &store, &gpu, 0).unwrap());
    assert!(p.fault_in_slot(2, key_b, &store, &gpu, 0).unwrap());
    assert_eq!(
        read_slot(&p, &gpu, 1),
        want_a,
        "key A recovered after slot recycle"
    );
    assert_eq!(read_slot(&p, &gpu, 2), want_b, "key B recovered");
}

/// 2026-10-01: `has_aux` answers what `aux(..).is_some()` answered, without copying the
/// blobs, through set, free and re-acquire, so the restore gates that switched to it decide
/// the same way.
#[test]
fn has_aux_tracks_the_aux_table_through_set_free_and_reuse() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 2, 2);
    let s = p.try_pop_free_slot().expect("a fresh pool has a free slot");
    assert!(!p.has_aux(s) && p.aux(s).is_none());
    p.set_aux(s, vec![(3, vec![7u8; 64])]);
    assert!(p.has_aux(s) && p.aux(s).is_some());
    p.free(s);
    assert!(!p.has_aux(s) && p.aux(s).is_none());
    let again = p.try_pop_free_slot().expect("the freed slot is free again");
    assert!(!p.has_aux(again), "a re-acquired slot carries no aux from its last holder");
}
