// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Tests of `KvCacheConfig::v_aliases_k` (`METRALE_GLM_KV_V_ALIAS`):
//! the V pool is the K pool, costs nothing in the block budget, is never
//! written through, and is freed once. On the mock GPU.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use super::*;
use metrale_core::scope::ModelResource;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-10-01: GLM-5.3's geometry: 11 DSA layers, one FP8 head of the
/// 512-dim MLA latent, 16-token blocks.
fn glm_config(v_aliases_k: bool) -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: 11,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
        v_aliases_k,
    }
}

#[test]
fn v_alias_halves_block_bytes_and_doubles_blocks() {
    let sep = glm_config(false);
    let ali = glm_config(true);
    // 2026-10-01: 16 tok * 512 B * 11 layers = 90,112 B of K per block,
    // 5,632 B per token; V doubles it when it has its own pool.
    assert_eq!(ali.block_bytes_kv_all_layers(), 90_112);
    assert_eq!(sep.block_bytes_kv_all_layers(), 2 * 90_112);
    assert_eq!(ali.block_bytes_kv_all_layers() / ali.block_size, 5_632);
    let budget = 30usize << 30;
    let n_sep = PagedKvCache::compute_num_blocks(&sep, budget).unwrap();
    let n_ali = PagedKvCache::compute_num_blocks(&ali, budget).unwrap();
    assert_eq!(n_ali, budget / 90_112);
    assert!((2 * n_sep..=2 * n_sep + 1).contains(&n_ali));
}

#[test]
fn v_alias_allocates_one_pool_per_layer_and_shares_pointers() {
    let gpu = MockGpuBackend::new();
    let cache = PagedKvCache::new(glm_config(true), 8, &gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 11, "one K pool per layer, no V pool");
    for l in 0..11 {
        assert_eq!(cache.v_pool_ptr(l), cache.k_pool_ptr(l));
        assert_eq!(cache.v_cache_ptr(l, 5), cache.k_cache_ptr(l, 5));
    }
    let gpu2 = MockGpuBackend::new();
    let _sep = PagedKvCache::new(glm_config(false), 8, &gpu2).unwrap();
    assert_eq!(gpu2.alloc_count(), 22);
}

#[test]
fn v_alias_write_block_never_clobbers_k() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(glm_config(true), 4, &gpu).unwrap();
    let b = cache.alloc_block().unwrap();
    let stride = cache.block_stride_bytes();
    let k_data: Vec<u8> = (0..stride).map(|i| (i % 251) as u8).collect();
    let v_data = vec![0xA5u8; stride];
    cache.write_block(3, b, &k_data, &v_data, &gpu).unwrap();
    let (k_out, v_out) = cache.read_block(3, b, &gpu).unwrap();
    assert_eq!(k_out, k_data, "the V write must not land on K");
    assert_eq!(v_out, k_data, "an aliased V reads back as K");
}

#[test]
fn v_alias_release_frees_each_pool_once() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(glm_config(true), 4, &gpu).unwrap();
    // 2026-10-01: The mock `free` errors on a pointer already freed, so a
    // double free of the shared pool fails here.
    cache.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn v_alias_lever_is_glm_only() {
    // 2026-10-01: Every other model keeps its V pool whatever the env says.
    assert!(!glm_kv_v_alias("deepseek_v3"));
    assert!(!glm_kv_v_alias("qwen3_next"));
}
