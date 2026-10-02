// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of `KvCacheConfig` block sizes and `PagedKvCache`
//! allocation, reference counting and host block copies, on the mock GPU.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use super::*;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

fn test_config() -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 256,
        num_layers: 12,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
        v_aliases_k: false,
    }
}

#[test]
fn test_block_bytes_fp8() {
    let cfg = test_config();
    // 2026-09-25: 16 tokens * 2 heads * 256 dim * 1 byte = 8192.
    assert_eq!(cfg.block_bytes(), 8192);
    assert_eq!(cfg.block_bytes_kv(), 16384);
}

#[test]
fn test_block_bytes_bf16() {
    let cfg = KvCacheConfig {
        dtype: KvCacheDtype::Bf16,
        ..test_config()
    };
    // 2026-09-25: 16 * 2 * 256 * 2 = 16384.
    assert_eq!(cfg.block_bytes(), 16384);
}

#[test]
fn test_block_bytes_nvfp4() {
    let cfg = KvCacheConfig {
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };
    // 2026-09-25: data 16 * 2 * 256 / 2 = 4096, scales 16 * 2 * 256 / 16 = 512,
    // total 4608.
    assert_eq!(cfg.block_bytes(), 4608);
    assert_eq!(cfg.nvfp4_data_bytes(), 4096);
    assert_eq!(cfg.nvfp4_scale_bytes(), 512);
}

#[test]
fn test_alloc_free() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 10, &gpu).unwrap();
    assert_eq!(cache.num_free_blocks(), 10);

    let b0 = cache.alloc_block().unwrap();
    let b1 = cache.alloc_block().unwrap();
    assert_ne!(b0, b1);
    assert_eq!(cache.num_free_blocks(), 8);

    cache.free_block(b0);
    assert_eq!(cache.num_free_blocks(), 9);
}

#[test]
fn test_exhaust() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 2, &gpu).unwrap();
    cache.alloc_block().unwrap();
    cache.alloc_block().unwrap();
    assert!(cache.alloc_block().is_err());
}

#[test]
fn test_compute_num_blocks() {
    let cfg = test_config();
    // 2026-09-25: K + V 16384 bytes * 12 layers = 196608 bytes per block.
    let n = PagedKvCache::compute_num_blocks(&cfg, 1_000_000).unwrap();
    assert_eq!(n, 1_000_000 / 196608);
}

#[test]
fn test_compute_num_blocks_nvfp4() {
    let cfg = KvCacheConfig {
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };
    // 2026-09-25: 4608 * 2 (K + V) * 12 layers = 110592 bytes per block.
    let n = PagedKvCache::compute_num_blocks(&cfg, 1_000_000).unwrap();
    assert_eq!(n, 1_000_000 / 110592);
}

#[test]
fn test_ref_counting() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 5, &gpu).unwrap();

    let b0 = cache.alloc_block().unwrap();
    assert_eq!(cache.ref_count(b0), 1);
    assert_eq!(cache.num_free_blocks(), 4);

    cache.inc_ref(b0);
    assert_eq!(cache.ref_count(b0), 2);

    // 2026-09-25: 2 to 1 keeps the block; 1 to 0 returns it to the free list.
    let freed = cache.dec_ref(b0);
    assert!(!freed);
    assert_eq!(cache.num_free_blocks(), 4);

    let freed = cache.dec_ref(b0);
    assert!(freed);
    assert_eq!(cache.num_free_blocks(), 5);
}

#[test]
fn test_try_alloc_block() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 2, &gpu).unwrap();

    assert!(cache.try_alloc_block().is_some());
    assert!(cache.try_alloc_block().is_some());
    assert!(cache.try_alloc_block().is_none());
}

#[test]
fn test_return_evicted_block() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 2, &gpu).unwrap();

    let b0 = cache.alloc_block().unwrap();
    let _b1 = cache.alloc_block().unwrap();
    assert_eq!(cache.num_free_blocks(), 0);

    cache.return_evicted_block(b0);
    assert_eq!(cache.num_free_blocks(), 1);
    assert_eq!(cache.ref_count(b0), 0);

    let b2 = cache.alloc_block().unwrap();
    assert_eq!(b2, b0);
    assert_eq!(cache.ref_count(b2), 1);
}

#[test]
fn return_evicted_block_keeps_shared_block_alive() {
    // 2026-09-25: A block held by a sequence (ref 1) and the prefix cache
    // (ref 2) stays allocated when the cache evicts it: eviction releases one
    // ref. A second release then frees it.
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 2, &gpu).unwrap();

    let b0 = cache.alloc_block().unwrap();
    cache.inc_ref(b0);
    assert_eq!(cache.ref_count(b0), 2);
    let free_before = cache.num_free_blocks();

    cache.return_evicted_block(b0);
    assert_eq!(cache.ref_count(b0), 1, "still referenced by the owner seq");
    assert_eq!(
        cache.num_free_blocks(),
        free_before,
        "a still-live block must not return to the free list"
    );

    cache.return_evicted_block(b0);
    assert_eq!(cache.ref_count(b0), 0);
    assert_eq!(cache.num_free_blocks(), free_before + 1);
}

#[test]
fn test_read_write_block_roundtrip() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 4, &gpu).unwrap();
    let b0 = cache.alloc_block().unwrap();

    // 2026-09-25: The stride is pinned to a hand-derived 8192 (as in
    // `test_block_bytes_fp8`): the write and read buffers are both sized from
    // `block_stride_bytes()`, so a wrong stride would otherwise still round
    // trip.
    let stride = cache.block_stride_bytes();
    assert_eq!(
        stride, 8192,
        "FP8 block stride: 16 tokens * 2 heads * 256 dim * 1 byte"
    );
    let k_data: Vec<u8> = (0..stride).map(|i| (i % 256) as u8).collect();
    let v_data: Vec<u8> = (0..stride).map(|i| ((i + 128) % 256) as u8).collect();
    cache.write_block(0, b0, &k_data, &v_data, &gpu).unwrap();

    let (k_out, v_out) = cache.read_block(0, b0, &gpu).unwrap();
    assert_eq!(k_out, k_data);
    assert_eq!(v_out, v_data);
}

#[test]
fn test_read_write_block_multiple_layers() {
    let gpu = MockGpuBackend::new();
    let cfg = test_config();
    let mut cache = PagedKvCache::new(cfg, 4, &gpu).unwrap();
    let b0 = cache.alloc_block().unwrap();

    // 2026-09-25: Pinned for the reason in `test_read_write_block_roundtrip`.
    let stride = cache.block_stride_bytes();
    assert_eq!(
        stride, 8192,
        "FP8 block stride: 16 tokens * 2 heads * 256 dim * 1 byte"
    );
    for layer in 0..12 {
        let k: Vec<u8> = vec![layer as u8; stride];
        let v: Vec<u8> = vec![(layer + 100) as u8; stride];
        cache.write_block(layer, b0, &k, &v, &gpu).unwrap();
    }
    for layer in 0..12 {
        let (k, v) = cache.read_block(layer, b0, &gpu).unwrap();
        assert!(k.iter().all(|&b| b == layer as u8));
        assert!(v.iter().all(|&b| b == (layer + 100) as u8));
    }
}

#[test]
fn test_num_layers() {
    let cfg = test_config();
    let gpu = MockGpuBackend::new();
    let cache = PagedKvCache::new(cfg, 2, &gpu).unwrap();
    assert_eq!(cache.num_layers(), 12);
}

#[test]
fn test_kv_cache_dtype_parse() {
    assert_eq!("fp8".parse::<KvCacheDtype>().unwrap(), KvCacheDtype::Fp8);
    assert_eq!("bf16".parse::<KvCacheDtype>().unwrap(), KvCacheDtype::Bf16);
    assert_eq!(
        "nvfp4".parse::<KvCacheDtype>().unwrap(),
        KvCacheDtype::Nvfp4
    );
    assert!("int8".parse::<KvCacheDtype>().is_err());
}

#[test]
fn test_dtype_for_layer_empty_fallback() {
    let cfg = test_config();
    assert_eq!(cfg.dtype_for_layer(0), KvCacheDtype::Fp8);
    assert_eq!(cfg.dtype_for_layer(11), KvCacheDtype::Fp8);
}

#[test]
fn test_dtype_for_layer_mixed() {
    let mut layer_dtypes = vec![KvCacheDtype::Nvfp4; 12];
    layer_dtypes[0] = KvCacheDtype::Bf16;
    layer_dtypes[1] = KvCacheDtype::Bf16;
    layer_dtypes[10] = KvCacheDtype::Bf16;
    layer_dtypes[11] = KvCacheDtype::Bf16;

    let cfg = KvCacheConfig {
        layer_dtypes,
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };

    assert_eq!(cfg.dtype_for_layer(0), KvCacheDtype::Bf16);
    assert_eq!(cfg.dtype_for_layer(1), KvCacheDtype::Bf16);
    assert_eq!(cfg.dtype_for_layer(2), KvCacheDtype::Nvfp4);
    assert_eq!(cfg.dtype_for_layer(5), KvCacheDtype::Nvfp4);
    assert_eq!(cfg.dtype_for_layer(9), KvCacheDtype::Nvfp4);
    assert_eq!(cfg.dtype_for_layer(10), KvCacheDtype::Bf16);
    assert_eq!(cfg.dtype_for_layer(11), KvCacheDtype::Bf16);
}

#[test]
fn test_block_bytes_for_layer_mixed() {
    let mut layer_dtypes = vec![KvCacheDtype::Nvfp4; 12];
    layer_dtypes[0] = KvCacheDtype::Bf16;
    layer_dtypes[11] = KvCacheDtype::Bf16;

    let cfg = KvCacheConfig {
        layer_dtypes,
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };

    // 2026-09-25: BF16 16 * 2 * 256 * 2 = 16384; NVFP4 4096 data + 512
    // scales = 4608.
    assert_eq!(cfg.block_bytes_for_layer(0), 16384);
    assert_eq!(cfg.block_bytes_for_layer(11), 16384);
    assert_eq!(cfg.block_bytes_for_layer(1), 4608);
    assert_eq!(cfg.block_bytes_for_layer(5), 4608);
}

#[test]
fn test_block_bytes_kv_all_layers_uniform() {
    let cfg = test_config();
    // 2026-09-25: FP8 8192 bytes per side, K + V 16384, 12 layers: 196608.
    assert_eq!(cfg.block_bytes_kv_all_layers(), 196608);
}

#[test]
fn test_block_bytes_kv_all_layers_mixed() {
    let mut layer_dtypes = vec![KvCacheDtype::Nvfp4; 12];
    layer_dtypes[0] = KvCacheDtype::Bf16;
    layer_dtypes[11] = KvCacheDtype::Bf16;

    let cfg = KvCacheConfig {
        layer_dtypes,
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };

    // 2026-09-25: 2 BF16 layers and 10 NVFP4 layers, K + V each.
    let expected = 2 * 16384 * 2 + 10 * 4608 * 2;
    assert_eq!(cfg.block_bytes_kv_all_layers(), expected);
}

#[test]
fn test_compute_num_blocks_mixed_dtype() {
    let mut layer_dtypes = vec![KvCacheDtype::Nvfp4; 12];
    layer_dtypes[0] = KvCacheDtype::Bf16;
    layer_dtypes[11] = KvCacheDtype::Bf16;

    let cfg = KvCacheConfig {
        layer_dtypes,
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };

    // 2026-09-25: The expected size is hand-derived, not read from
    // `block_bytes_kv_all_layers()`, which `compute_num_blocks` itself calls;
    // this test checks that `compute_num_blocks` divides by it.
    let expected_bytes_per_block = 2 * 16384 * 2 + 10 * 4608 * 2;
    assert_eq!(
        cfg.block_bytes_kv_all_layers(),
        expected_bytes_per_block,
        "sanity: cfg matches the independently-derived per-block size"
    );
    let n = PagedKvCache::compute_num_blocks(&cfg, 1_000_000).unwrap();
    assert_eq!(n, 1_000_000 / expected_bytes_per_block);
}

#[test]
fn test_mixed_dtype_pool_allocation() {
    let gpu = MockGpuBackend::new();
    let mut layer_dtypes = vec![KvCacheDtype::Fp8; 4];
    layer_dtypes[0] = KvCacheDtype::Bf16;
    layer_dtypes[3] = KvCacheDtype::Bf16;

    let cfg = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 256,
        num_layers: 4,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes,
        layer_dims: vec![],
        cache_blocks_per_seq: None,
        v_aliases_k: false,
    };
    let cache = PagedKvCache::new(cfg, 4, &gpu).unwrap();

    assert_eq!(cache.block_stride_bytes_for_layer(0), 16384);
    assert_eq!(cache.block_stride_bytes_for_layer(1), 8192);
    assert_eq!(cache.block_stride_bytes_for_layer(2), 8192);
    assert_eq!(cache.block_stride_bytes_for_layer(3), 16384);

    assert_eq!(cache.dtype_for_layer(0), KvCacheDtype::Bf16);
    assert_eq!(cache.dtype_for_layer(1), KvCacheDtype::Fp8);
    assert_eq!(cache.dtype_for_layer(3), KvCacheDtype::Bf16);
}

/// 2026-09-25: A `--high-speed-swap`-style sliding window of 4 blocks per
/// sequence, 100 allocations each, in an 8-block pool: the pool never runs
/// dry, and each allocation after a slide gets back the block just freed.
#[test]
fn sliding_window_recycles_blocks() {
    let gpu = MockGpuBackend::new();
    let cfg = KvCacheConfig {
        cache_blocks_per_seq: Some(4),
        v_aliases_k: false,
        ..test_config()
    };
    let mut cache = PagedKvCache::new(cfg, 8, &gpu).unwrap();

    for seq_id in 0..2 {
        let mut block_table: Vec<u32> = Vec::new();
        // 2026-09-25: The free list is LIFO, so the next alloc must return the
        // block the last slide freed; a count-only check would miss a FIFO
        // list or a wrong evicted block.
        let mut last_evicted: Option<u32> = None;
        for step in 0..100 {
            let blk = cache.alloc_block().unwrap_or_else(|_| {
                panic!("alloc failed at seq {seq_id} step {step}: no free blocks")
            });
            if let Some(evicted) = last_evicted {
                assert_eq!(
                    blk, evicted,
                    "seq {seq_id} step {step}: expected the slide to recycle the \
                     just-evicted physical block {evicted}, got {blk} instead"
                );
            }
            block_table.push(blk);
            last_evicted = None;
            while block_table.len() > 4 {
                let evicted = block_table.remove(0);
                cache.free_block(evicted);
                last_evicted = Some(evicted);
            }
        }
        for &blk in &block_table {
            cache.free_block(blk);
        }
        assert!(
            cache.num_free_blocks() >= 4,
            "after seq {seq_id} all blocks freed: {} free",
            cache.num_free_blocks()
        );
    }
    assert_eq!(cache.num_free_blocks(), 8);
}

/// 2026-09-25: After `free_block` returns a block to the pool, the next
/// `alloc_block` returns that block (the free list is LIFO).
#[test]
fn alloc_after_free_round_trip_returns_same_block_lifo() {
    let gpu = MockGpuBackend::new();
    let cfg = test_config();
    let mut cache = PagedKvCache::new(cfg, 4, &gpu).unwrap();

    let b0 = cache.alloc_block().unwrap();
    let b1 = cache.alloc_block().unwrap();
    let b2 = cache.alloc_block().unwrap();
    let b3 = cache.alloc_block().unwrap();
    assert_eq!(cache.num_free_blocks(), 0);
    assert!(cache.alloc_block().is_err(), "pool exhausted as expected");

    cache.free_block(b0);
    let recycled = cache.alloc_block().unwrap();
    assert_eq!(
        recycled, b0,
        "alloc-after-free must return the just-freed block (LIFO precondition for HSS slide)"
    );

    cache.free_block(b1);
    cache.free_block(b2);
    assert_eq!(cache.alloc_block().unwrap(), b2);
    assert_eq!(cache.alloc_block().unwrap(), b1);

    assert_eq!(cache.num_free_blocks(), 0);
    cache.free_block(b3);
    cache.free_block(recycled);
}
