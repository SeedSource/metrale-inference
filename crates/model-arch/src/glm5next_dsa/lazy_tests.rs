// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Tests for the lazily mapped DSA indexer cache: the sizing arithmetic at GLM-5.3
//! dimensions, the mapping schedule of a lazily allocated `Glm5NextDsaState` (granules, high
//! water, look-ahead, refusals, release) and aux restore into it.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.
//!
//! CPU-only: the mock backend's lazy buffers are eager allocations that run the same budget
//! and high-water logic as the CUDA VMM ones (`metrale_gpu_runtime::lazy_buffer`).

use std::sync::Arc;

use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::lazy_buffer::{DEFAULT_GRANULE, MapBudget};
use metrale_model_layers::layer::LayerState;

use super::*;
use crate::glm5next_dsa::Glm5NextDsaConfig;
use crate::glm5next_dsa::state::Glm5NextDsaState;

const G: usize = DEFAULT_GRANULE;
const MIB: usize = 1 << 20;
/// 2026-10-03: Rows of 256 B (`index_head_dim` 128, BF16) in one 2 MiB granule.
const ROWS_PER_GRANULE: usize = 8192;

fn glm53(capacity: usize, proposer: bool) -> LazyShape {
    LazyShape {
        dsa_layers: 11,
        proposer,
        index_head_dim: 128,
        capacity,
    }
}

fn cfg(max_context: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context,
    }
}

#[test]
fn one_granule_holds_8192_rows() {
    assert_eq!(G / row_bytes(128), ROWS_PER_GRANULE);
    assert_eq!(mapped_bytes_for_rows(0, 128, G), 0);
    assert_eq!(mapped_bytes_for_rows(1, 128, G), G);
    assert_eq!(mapped_bytes_for_rows(8192, 128, G), G);
    assert_eq!(mapped_bytes_for_rows(8193, 128, G), 2 * G);
    assert_eq!(mapped_bytes_for_rows(131_072, 128, G), 32 * MIB);
}

/// 2026-10-03: GLM-5.3 at msl 131,072, max batch 4, MTP on: 24 buffers per sequence; a 32K
/// sequence maps 4 granules per target buffer and 5 per proposer buffer (look-ahead), 196 MiB;
/// the default pool is 4 of those, 784 MiB, which is exactly 16 granule sets = 131,072 tokens.
#[test]
fn default_pool_at_glm53_131k_mb4_is_784_mib() {
    let s = glm53(131_072, true);
    assert_eq!(s.bufs_per_seq(), 24);
    assert_eq!(s.seq_mapped_bytes(32_768, G), (22 * 4 + 2 * 5) * G);
    assert_eq!(s.seq_mapped_bytes(32_768, G), 196 * MIB);
    assert_eq!(s.default_pool_bytes(4, G), 784 * MIB);
    assert_eq!(
        s.default_pool_bytes(0, G),
        196 * MIB,
        "max batch 0 counts as 1"
    );
    assert_eq!(s.free_tokens(784 * MIB, G), 16 * ROWS_PER_GRANULE);
    // 2026-10-03: The eager reserve this replaces: 4 x (11 + 1) x 67,239,936 B.
    let eager: usize = 4 * 12 * 67_239_936;
    assert!(
        eager > 3 * (784 * MIB),
        "the eager charge is over 3x the default pool"
    );
    // 2026-10-03: A context shorter than 32K caps the default at its capacity.
    let short = glm53(16_384, false);
    assert_eq!(short.default_pool_bytes(4, G), 4 * 22 * 2 * G);
}

#[test]
fn free_tokens_counts_whole_granule_sets_only() {
    let s = glm53(131_072, true);
    let set = 24 * G;
    assert_eq!(s.free_tokens(set - 1, G), 0);
    assert_eq!(s.free_tokens(set, G), ROWS_PER_GRANULE);
    assert_eq!(s.free_tokens(3 * set + set / 2, G), 3 * ROWS_PER_GRANULE);
}

/// 2026-10-03: A lazily allocated state maps nothing at creation, then whole granules as rows
/// are made room for; its pointers never move and every byte is charged to the pool.
#[test]
fn lazy_state_maps_granules_on_demand_with_stable_pointers() {
    let gpu = MockGpuBackend::new();
    let pool = Arc::new(MapBudget::new("test", usize::MAX));
    let mut st = Glm5NextDsaState::alloc_lazy(&gpu, &cfg(65_536), 0, pool.clone()).unwrap();
    let (k0, g0) = (st.k_normed, st.gate);
    assert_eq!(st.mapped_rows(), Some(0));
    assert_eq!(pool.used(), 0);

    st.ensure_room(1).unwrap();
    assert_eq!(st.mapped_rows(), Some(ROWS_PER_GRANULE));
    assert_eq!(pool.used(), 2 * G, "k_normed and gate each map one granule");
    st.advance(ROWS_PER_GRANULE).unwrap();
    st.ensure_room(1).unwrap();
    assert_eq!(
        st.mapped_rows(),
        Some(2 * ROWS_PER_GRANULE),
        "row 8192 needs granule 2"
    );

    // 2026-10-03: The graph-replay pre-check maps through seq_len + k.
    st.ensure_room_through(5 * ROWS_PER_GRANULE - 3).unwrap();
    assert_eq!(st.mapped_rows(), Some(5 * ROWS_PER_GRANULE));
    // 2026-10-03: A rewind keeps the mapping (high-water mark never shrinks).
    st.rewind_to(10).unwrap();
    st.ensure_room(1).unwrap();
    assert_eq!(st.mapped_rows(), Some(5 * ROWS_PER_GRANULE));
    assert_eq!(
        (st.k_normed, st.gate),
        (k0, g0),
        "pointers stable across mapping"
    );

    // 2026-10-03: Past capacity is refused by the capacity check, before any mapping.
    let used = pool.used();
    assert!(st.ensure_room_through(65_537).is_err());
    assert_eq!(pool.used(), used);

    st.free(&gpu).unwrap();
    assert_eq!(pool.used(), 0, "free refunds the pool");
    assert_eq!(gpu.live_bytes(), Some(0), "everything freed");
}

/// 2026-10-03: `map_rows_through` (the step-entry hook) maps through `end + lookahead`, capped
/// at capacity, and a target layer (look-ahead 0) maps exactly the block grid's end.
#[test]
fn map_rows_through_adds_the_lookahead_and_caps_at_capacity() {
    let gpu = MockGpuBackend::new();
    let pool = Arc::new(MapBudget::new("test", usize::MAX));
    let mut target = Glm5NextDsaState::alloc_lazy(&gpu, &cfg(32_768), 0, pool.clone()).unwrap();
    target.map_rows_through(ROWS_PER_GRANULE).unwrap();
    assert_eq!(target.mapped_rows(), Some(ROWS_PER_GRANULE));
    let mut prop =
        Glm5NextDsaState::alloc_lazy(&gpu, &cfg(32_768), PROPOSER_LOOKAHEAD_ROWS, pool.clone())
            .unwrap();
    prop.map_rows_through(ROWS_PER_GRANULE).unwrap();
    assert_eq!(
        prop.mapped_rows(),
        Some(2 * ROWS_PER_GRANULE),
        "look-ahead crosses into granule 2"
    );
    prop.map_rows_through(usize::MAX).unwrap();
    assert_eq!(prop.mapped_rows(), Some(32_768), "capped at capacity");
    prop.free(&gpu).unwrap();
    target.free(&gpu).unwrap();
    assert_eq!(pool.used(), 0);
}

/// 2026-10-03: A full pool refuses with the "KV cache exhausted" phrase the scheduler's decode
/// path preempts on, leaves the cursor and mapping unchanged, and frees up when another
/// sequence is released.
#[test]
fn a_full_pool_refuses_cleanly_and_recovers_on_free() {
    let gpu = MockGpuBackend::new();
    let pool = Arc::new(MapBudget::new("DSA indexer", 6 * G));
    let mut a = Glm5NextDsaState::alloc_lazy(&gpu, &cfg(65_536), 0, pool.clone()).unwrap();
    let mut b = Glm5NextDsaState::alloc_lazy(&gpu, &cfg(65_536), 0, pool.clone()).unwrap();
    a.ensure_room_through(2 * ROWS_PER_GRANULE).unwrap();
    b.ensure_room_through(ROWS_PER_GRANULE).unwrap();
    assert_eq!(pool.used(), 6 * G);
    b.advance(ROWS_PER_GRANULE).unwrap();
    let e = b.ensure_room(1).unwrap_err().to_string();
    assert!(
        e.contains("KV cache exhausted") && e.contains("DSA indexer"),
        "{e}"
    );
    assert_eq!(b.len(), ROWS_PER_GRANULE);
    assert_eq!(b.mapped_rows(), Some(ROWS_PER_GRANULE));
    a.free(&gpu).unwrap();
    b.ensure_room(1).unwrap();
    assert_eq!(b.mapped_rows(), Some(2 * ROWS_PER_GRANULE));
    b.free(&gpu).unwrap();
    assert_eq!(pool.used(), 0);
}

/// 2026-10-03: A prefix-cache aux restore into a fresh lazy state maps the rows it writes
/// (`restore_blob` -> `ensure_room_through(len)`) and is byte exact.
#[test]
fn aux_restore_maps_rows_before_writing_them() {
    let gpu = MockGpuBackend::new();
    let pool = Arc::new(MapBudget::new("test", usize::MAX));
    let c = cfg(32_768);
    let mut src = Glm5NextDsaState::alloc(&gpu, &c).unwrap();
    let len = ROWS_PER_GRANULE + 100;
    let keys: Vec<u8> = (0..len * 256).map(|i| (i as u8).wrapping_mul(31)).collect();
    let valid: Vec<u8> = (0..len).map(|i| (i % 7) as u8).collect();
    gpu.copy_h2d(&keys, src.k_normed).unwrap();
    gpu.copy_h2d(&keys, src.gate).unwrap();
    gpu.copy_h2d(&valid, src.valid).unwrap();
    src.advance(len).unwrap();
    let blob = src.snapshot_blob(&gpu, 0).unwrap();

    let mut dst = Glm5NextDsaState::alloc_lazy(&gpu, &c, 0, pool.clone()).unwrap();
    assert_eq!(dst.mapped_rows(), Some(0));
    dst.restore_blob(&blob, &gpu, 0).unwrap();
    assert_eq!(dst.len(), len);
    assert_eq!(dst.mapped_rows(), Some(2 * ROWS_PER_GRANULE));
    assert_eq!(dst.snapshot_blob(&gpu, 0).unwrap(), blob);
    src.free(&gpu).unwrap();
    dst.free(&gpu).unwrap();
    assert_eq!(pool.used(), 0);
}

/// 2026-10-03: Lever off (`alloc`; the lever is unset in tests), the state is eager: no
/// mapping, the same three allocations as before, and the step-entry hook does nothing.
#[test]
fn eager_state_has_no_mapping_and_the_hook_is_a_no_op() {
    assert!(
        !dsa_indexer_lazy(),
        "tests run with METRALE_DSA_INDEXER_LAZY unset"
    );
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(16_384)).unwrap();
    assert_eq!(st.mapped_rows(), None);
    st.map_rows_through(usize::MAX).unwrap();
    st.ensure_room_through(16_384).unwrap();
    assert_eq!(
        gpu.live_bytes(),
        Some(16_384 * 256 * 2 + 16_384),
        "exactly the three eager allocations"
    );
    st.free(&gpu).unwrap();
}
