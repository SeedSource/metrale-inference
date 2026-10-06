// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: `WeightArena` on the mock backend: packing, alignment, no overlap, byte content,
//! release, and the disabled default.

use super::*;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

const MIB: usize = 1024 * 1024;

#[test]
fn packing_sizes_each_chunk_to_its_content() {
    // Three 1 GiB-ish items under a 2 GiB cap: the third starts a new chunk.
    let gib = 1usize << 30;
    assert_eq!(
        pack_chunks([gib - 100, gib - 100, 5], 2 * gib),
        vec![2 * arena_footprint(gib - 100), arena_footprint(5)]
    );
    // An item above the cap gets a chunk of its own.
    assert_eq!(pack_chunks([3 * gib], 2 * gib), vec![3 * gib]);
    assert!(pack_chunks(std::iter::empty(), 2 * gib).is_empty());
    assert_eq!(arena_footprint(0), WEIGHT_ARENA_ALIGN);
    assert_eq!(arena_footprint(4), WEIGHT_ARENA_ALIGN);
    assert_eq!(arena_footprint(2 * MIB), 2 * MIB);
}

#[test]
fn disabled_arena_allocates_nothing() {
    let gpu = MockGpuBackend::new();
    let a = WeightArena::default();
    assert!(!a.is_active());
    assert_eq!(a.alloc(&gpu, 64).unwrap(), None);
    assert_eq!(a.upload(&gpu, &[1, 2, 3]).unwrap(), None);
    assert_eq!(a.stats(), ArenaStats::default());
}

#[test]
fn planned_allocations_are_aligned_disjoint_and_leave_no_tail() {
    let gpu = MockGpuBackend::new();
    let a = WeightArena::default();
    // The expert-TP layer pattern in miniature: packed, scale, a 4-byte scalar.
    let items: Vec<usize> = (0..12).flat_map(|_| [2048, 256, 4]).collect();
    let planned = a.plan("test", items.iter().copied());
    let mut spans = Vec::new();
    for (i, &b) in items.iter().enumerate() {
        let src: Vec<u8> = (0..b).map(|k| (i * 31 + k) as u8).collect();
        let p = a.upload(&gpu, &src).unwrap().expect("arena is planned");
        assert!(p.0.is_multiple_of(WEIGHT_ARENA_ALIGN as u64));
        assert!(a.contains(p));
        let mut back = vec![0u8; b];
        gpu.copy_d2h(p, &mut back).unwrap();
        assert_eq!(back, src, "item {i}");
        spans.push((p.0, p.0 + b as u64));
    }
    spans.sort_unstable();
    assert!(
        spans.windows(2).all(|w| w[0].1 <= w[1].0),
        "sub-allocations overlap"
    );
    let s = a.stats();
    assert_eq!(s.chunks, 1);
    assert_eq!(s.chunk_bytes, planned);
    assert_eq!(s.tail_waste, 0);
    assert_eq!(s.unplanned_chunks, 0);
    assert_eq!(s.sub_allocations, items.len());
    assert_eq!(s.allocations_saved(), items.len() - 1);
    assert_eq!(s.requested, items.iter().sum::<usize>());
    a.release(&gpu).unwrap();
    assert_eq!(a.stats(), ArenaStats::default());
    assert!(!a.is_active());
    // Idempotent: nothing left to free twice.
    a.release(&gpu).unwrap();
}

#[test]
fn an_unforeseen_allocation_gets_an_exact_chunk() {
    let gpu = MockGpuBackend::new();
    let a = WeightArena::default();
    a.plan("test", [512]);
    a.alloc(&gpu, 512).unwrap().unwrap();
    a.alloc(&gpu, 1000).unwrap().unwrap();
    let s = a.stats();
    assert_eq!(s.chunks, 2);
    assert_eq!(s.unplanned_chunks, 1);
    assert_eq!(s.chunk_bytes, 512 + arena_footprint(1000));
    assert_eq!(s.tail_waste, 0);
    a.release(&gpu).unwrap();
}

#[test]
fn a_new_plan_closes_the_open_chunk_and_counts_its_tail() {
    let gpu = MockGpuBackend::new();
    let a = WeightArena::default();
    a.plan("test", [256, 256]);
    a.alloc(&gpu, 256).unwrap().unwrap();
    a.plan("test", [256]);
    let p = a.alloc(&gpu, 256).unwrap().unwrap();
    let s = a.stats();
    assert_eq!(s.chunks, 2);
    assert_eq!(s.tail_waste, 256);
    assert!(a.contains(p));
    a.release(&gpu).unwrap();
}
