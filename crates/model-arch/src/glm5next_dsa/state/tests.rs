// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of the DSA indexer cache: sizing, refusal past the cap, and release.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;

fn cfg() -> Glm5NextDsaConfig {
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
        max_context: 16_384,
    }
}

/// 2026-09-25: The context cap comes from `max_context`, and `DsaSelectGeometry::plan` accepts
/// contexts past 16,384 tokens: the top-k walks the pools in fixed-size tiles.
#[test]
fn the_context_cap_follows_max_seq_len_not_the_kernel() {
    let c = cfg();
    assert_eq!(max_dsa_context(&c), 16_384, "the fixture reserves 16,384");
    // 2026-09-25: 131,072 tokens is 32,768 pools, sixteen `topk_tile()` tiles of 2,048, and the
    // shared-memory request stays the same.
    let mut big = cfg();
    big.max_context = 131_072;
    assert_eq!(max_dsa_context(&big), 131_072);
    for seq in [16_384usize, 16_388, 65_536, 131_072] {
        let g = DsaSelectGeometry::plan(&big, seq, 1).unwrap_or_else(|e| {
            panic!("plan refused {seq} tokens: {e}");
        });
        assert_eq!(g.topk_np2, super::super::select::topk_tile());
        assert_eq!(
            g.topk_smem,
            super::super::select::topk_smem_for_tile(g.topk_np2)
        );
        assert!(g.topk_smem <= super::super::select::TOPK_SMEM_CEILING);
    }
}

/// 2026-09-25: The reservation is rounded down to a whole number of pools.
#[test]
fn the_cap_is_rounded_down_to_whole_pools() {
    let mut c = cfg();
    c.max_context = 16_386;
    assert_eq!(max_dsa_context(&c), 16_384);
    c.index_kpool = 8;
    c.max_context = 1_001;
    assert_eq!(max_dsa_context(&c), 1_000);
}

/// 2026-09-25: Advancing past the cap is refused and leaves the length unchanged.
#[test]
fn advancing_past_the_cap_is_refused_not_clamped() {
    let c = cfg();
    let cap = max_dsa_context(&c);
    // 2026-09-25: Built by hand with null pointers: the length bookkeeping touches no memory.
    let mut s = Glm5NextDsaState {
        k_normed: metrale_gpu_runtime::gpu::DevicePtr(0),
        gate: metrale_gpu_runtime::gpu::DevicePtr(0),
        valid: metrale_gpu_runtime::gpu::DevicePtr(0),
        len: 0,
        capacity: cap,
        index_head_dim: c.index_head_dim,
        released: false,
        lazy: None,
        lookahead: 0,
        pool: None,
    };
    assert!(s.is_empty());
    s.advance(cap - 1).unwrap();
    assert_eq!(s.len(), cap - 1);
    s.advance(1).unwrap();
    assert_eq!(s.len(), cap, "exactly full is legal");
    let e = s.advance(1).unwrap_err().to_string();
    assert!(
        e.contains("--max-seq-len"),
        "name the knob that moves it: {e}"
    );
    assert_eq!(s.len(), cap, "a refused advance must not move the cursor");
}

/// 2026-09-25: Row offsets are in bytes over a flat `[capacity, index_head_dim]` BF16 buffer.
#[test]
fn row_offsets_are_flat_bf16_rows() {
    let c = cfg();
    let s = Glm5NextDsaState {
        k_normed: metrale_gpu_runtime::gpu::DevicePtr(0),
        gate: metrale_gpu_runtime::gpu::DevicePtr(0),
        valid: metrale_gpu_runtime::gpu::DevicePtr(0),
        len: 0,
        capacity: max_dsa_context(&c),
        index_head_dim: c.index_head_dim,
        released: false,
        lazy: None,
        lookahead: 0,
        pool: None,
    };
    assert_eq!(s.row_offset(0), 0);
    assert_eq!(s.row_offset(1), 128 * 2);
    assert_eq!(s.row_offset(1000), 1000 * 128 * 2);
}

/// 2026-09-25: At a 16,384-token cap the reservation is 8,404,992 B (about 8 MiB) per layer per
/// sequence, under 100 MiB over 11 layers.
#[test]
fn the_reservation_is_small_enough_to_preallocate() {
    let c = cfg();
    let cap = max_dsa_context(&c);
    let per_layer = cap * c.index_head_dim * 2 * 2 + cap;
    assert_eq!(per_layer, 8_404_992);
    assert!(
        per_layer * 11 < 100 << 20,
        "under 100 MiB for the DSA stack"
    );
}

/// 2026-09-25: `ensure_room` answers before any write and moves nothing: `indexer_forward` calls it
/// before it writes row `len()`.
#[test]
fn ensure_room_refuses_before_the_write_and_moves_nothing() {
    let c = cfg();
    let cap = max_dsa_context(&c);
    let mut s = Glm5NextDsaState {
        k_normed: metrale_gpu_runtime::gpu::DevicePtr(0),
        gate: metrale_gpu_runtime::gpu::DevicePtr(0),
        valid: metrale_gpu_runtime::gpu::DevicePtr(0),
        len: 0,
        capacity: cap,
        index_head_dim: c.index_head_dim,
        released: false,
        lazy: None,
        lookahead: 0,
        pool: None,
    };
    s.advance(cap).unwrap();
    assert!(
        s.ensure_room(0).is_ok(),
        "exactly full still has room for zero rows"
    );
    let e = s.ensure_room(1).unwrap_err().to_string();
    assert!(
        e.contains("--max-seq-len"),
        "name the knob that moves it: {e}"
    );
    assert_eq!(
        s.len(),
        cap,
        "a refused ensure_room must not move the cursor"
    );
}

/// 2026-09-25: Neither `Glm5NextDsaState` nor `DevicePtr` implements `Drop`, so the three indexer
/// buffers stay allocated until `free` releases them. `free` must return the backend to its
/// baseline, and a second call must be a no-op.
#[test]
fn free_returns_every_indexer_buffer_and_is_idempotent() {
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    let gpu = MockGpuBackend::new();
    let c = cfg();
    let base = gpu.alloc_count();
    let mut s = Glm5NextDsaState::alloc(&gpu, &c).unwrap();
    assert_eq!(
        gpu.alloc_count(),
        base + 3,
        "k_normed + gate + valid are three live device allocations"
    );
    s.free(&gpu).unwrap();
    assert_eq!(
        gpu.alloc_count(),
        base,
        "free returns to the baseline exactly"
    );
    assert_eq!(s.k_normed.0, 0, "a released state must not look live");

    // 2026-09-25: Two owners can release a DSA state (the MTP drafter's `free_state` and the
    // target layer's), so a second call must not free again.
    let other = gpu.alloc(4096).unwrap();
    s.free(&gpu).unwrap();
    assert_eq!(
        gpu.alloc_count(),
        base + 1,
        "the second free must not touch an unrelated allocation"
    );
    gpu.free(other).unwrap();
}

/// 2026-09-25: 100 alloc/free cycles on one backend leave the allocation count at its baseline.
#[test]
fn a_hundred_alloc_free_cycles_leak_nothing() {
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    let gpu = MockGpuBackend::new();
    let c = cfg();
    let base = gpu.alloc_count();
    for _ in 0..100 {
        let mut s = Glm5NextDsaState::alloc(&gpu, &c).unwrap();
        s.advance(16).unwrap();
        s.free(&gpu).unwrap();
    }
    assert_eq!(gpu.alloc_count(), base, "no per-sequence growth");
}

/// 2026-09-25: Alloc then release returns the backend ledger to its baseline in count and in
/// bytes. The byte check catches a leak that keeps the count balanced.
#[test]
fn alloc_then_release_returns_the_ledger_to_baseline_in_count_and_bytes() {
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    let gpu = MockGpuBackend::new();
    let cfg = Glm5NextDsaConfig {
        max_context: 131_072,
        ..cfg()
    };

    let base_count = gpu.live_alloc_count();
    let base_bytes = gpu.live_bytes().expect("the mock keeps a ledger");

    for _ in 0..11 {
        let mut st = Glm5NextDsaState::alloc(&gpu, &cfg).expect("alloc");

        // 2026-09-25: In flight: exactly the three buffers, and exactly `indexer_state_bytes`.
        assert_eq!(gpu.live_alloc_count(), base_count + 3);
        let in_flight = gpu.live_bytes().expect("ledger") - base_bytes;
        assert_eq!(
            in_flight,
            indexer_state_bytes(max_dsa_context(&cfg), cfg.index_head_dim),
            "what alloc actually takes must equal what the reserve charges"
        );

        st.free(&gpu).expect("free");
        assert_eq!(gpu.live_alloc_count(), base_count, "count back to baseline");
        assert_eq!(
            gpu.live_bytes().expect("ledger"),
            base_bytes,
            "BYTES back to baseline"
        );

        st.free(&gpu).expect("second free is a no-op");
        assert_eq!(gpu.live_bytes().expect("ledger"), base_bytes);
    }
}
