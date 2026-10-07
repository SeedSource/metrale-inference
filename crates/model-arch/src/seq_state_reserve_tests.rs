// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Exact-integer pins for the per-sequence reserve, from the GLM-5.3
//! dimensions. No GPU and no checkpoint.
//!
//! Owner: model-arch (serve memory reserve).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::state::{dsa_capacity, indexer_state_bytes};

/// 2026-09-25: GLM-5.3, from the checked-in config fixture
/// (`crates/model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json`):
/// `index_head_dim = 128`, `index_kpool = 4`, `hidden_size = 4096`,
/// `vocab_size = 154880`, 11 `deepseek_sparse_attention` layers.
const HEAD_DIM: usize = 128;
const KPOOL: usize = 4;
const HIDDEN: usize = 4096;
const VOCAB: usize = 154_880;
const DSA_LAYERS: usize = 11;
const MSL: usize = 131_072;

/// 2026-09-25: One KV block over the 11 DSA layers for a 1-byte (FP8) cache of
/// `kv_lora_rank` 512 wide rows: `2 (k + v) * block_size 16 * num_kv_heads 1 * 512`
/// = 16,384 B per layer per block.
const KV_BLOCK_BYTES: usize = (2 * 16) * 512 * DSA_LAYERS;

#[test]
fn the_kv_block_is_180_224_bytes_exactly() {
    assert_eq!(KV_BLOCK_BYTES, 180_224);
}

#[test]
fn indexer_bytes_are_513_per_token_per_layer() {
    assert_eq!(indexer_state_bytes(1, HEAD_DIM), 4 * HEAD_DIM + 1);
    assert_eq!(indexer_state_bytes(1, HEAD_DIM), 513);
    assert_eq!(indexer_state_bytes(MSL, HEAD_DIM), 67_239_936);
}

#[test]
fn capacity_rounds_down_to_whole_pools_and_is_the_only_spelling() {
    // 2026-09-25: Exact multiples are unchanged.
    assert_eq!(dsa_capacity(MSL, KPOOL), MSL);
    assert_eq!(dsa_capacity(65_536, KPOOL), 65_536);
    // 2026-09-25: A context that does not divide evenly rounds down to whole pools.
    assert_eq!(dsa_capacity(4_099, KPOOL), 4_096);
    assert_eq!(dsa_capacity(3, KPOOL), 0);
    // 2026-09-25: A kpool of 0 is treated as 1.
    assert_eq!(dsa_capacity(MSL, 0), MSL);
}

/// 2026-09-25: The reserve must equal what `Glm5NextDsaState::alloc` takes: three
/// allocations, `capacity*d*2`, `capacity*d*2`, `capacity`.
#[test]
fn the_reserve_equals_what_dsa_alloc_actually_allocates() {
    let cap = dsa_capacity(MSL, KPOOL);
    let what_alloc_takes = cap * HEAD_DIM * 2 + cap * HEAD_DIM * 2 + cap;
    assert_eq!(indexer_state_bytes(cap, HEAD_DIM), what_alloc_takes);
}

#[test]
fn target_layer_charge_at_131072_is_exact() {
    let per_layer = indexer_state_bytes(dsa_capacity(MSL, KPOOL), HEAD_DIM);
    assert_eq!(DSA_LAYERS * per_layer, 739_639_296);
}

#[test]
fn proposer_charge_at_131072_is_exact_and_is_a_separate_owner() {
    let per_layer = indexer_state_bytes(dsa_capacity(MSL, KPOOL), HEAD_DIM);
    let proposer = per_layer + 2 * HIDDEN * 2 + HIDDEN * 2 + VOCAB * 2 + 4 + 16;
    assert_eq!(proposer, 67_574_292);
    // 2026-09-25: The proposer owns one indexer block plus five small buffers (334,356 B).
    assert_eq!(proposer - per_layer, 334_356);
    assert_eq!(12 * per_layer, 806_879_232);
    assert_ne!(
        proposer,
        12 * per_layer - DSA_LAYERS * per_layer + 334_356 + 1
    );
}

#[test]
fn total_per_sequence_and_the_batch_3_charge_are_exact() {
    let s = PerSequenceState {
        target_layers: 739_639_296,
        proposer: 67_574_292,
    };
    assert_eq!(s.total(), 807_213_588);
    assert_eq!(s.for_batch(3), 2_421_640_764);
    assert_eq!(s.for_batch(1), s.total());
    assert_eq!(
        s.for_batch(0),
        s.total(),
        "batch 0 is clamped to 1, never zero-charged"
    );
}

/// 2026-09-25: The batch-3 charge in `KV_BLOCK_BYTES` blocks.
#[test]
fn the_batch_3_charge_converts_to_exactly_13_436_kv_blocks() {
    let charge = 2_421_640_764usize;
    assert_eq!(charge / KV_BLOCK_BYTES, 13_436);
    let reachable = 3 * MSL.div_ceil(16) + 3 + 1;
    assert_eq!(reachable, 24_580);
    assert_eq!(reachable + charge / KV_BLOCK_BYTES, 38_016);
}

#[test]
fn the_source_derived_block_size_reproduces_both_measured_clamp_lines() {
    let gib = |b: usize| (b as f64) / (1024.0 * 1024.0 * 1024.0);
    let freed_131k = (46_887 - 8_194) * KV_BLOCK_BYTES;
    let freed_65k = (47_537 - 4_098) * KV_BLOCK_BYTES;
    assert_eq!(format!("{:.2}", gib(freed_131k)), "6.49");
    assert_eq!(format!("{:.2}", gib(freed_65k)), "7.29");
}

#[test]
fn a_non_glm_config_is_charged_nothing() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    assert_ne!(cfg.model_type, "glm5_next");
    let s = per_sequence_state_bytes(&cfg, MSL, true).expect("non-GLM is inert, not an error");
    assert_eq!(s, PerSequenceState::default());
    assert_eq!(s.total(), 0);
}

/// 2026-10-06: The real GLM-5.3 config (`model-engine/tests/fixtures`), for the pool-cache
/// reserve pins below.
fn glm53_config() -> ModelConfig {
    metrale_config::parse_config(include_str!(
        "../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json"
    ))
    .expect("the real checkpoint config parses")
}

/// 2026-10-06: `METRALE_GLM_DSA_POOL_CACHE`: each indexer cache (11 target layers and the
/// proposer's) is charged `pool_cache_state_bytes`, the rest of the proposer term unchanged;
/// lever off, the `_with` form is the function the server calls.
#[test]
fn the_pool_cache_reserve_charges_each_indexer_cache_its_pool_arrays_and_rings() {
    use crate::glm5next_dsa::pool_cache::pool_cache_state_bytes;
    let cfg = glm53_config();
    let msl = 540_672;
    let off = per_sequence_state_bytes_with(&cfg, msl, true, false).unwrap();
    assert_eq!(off, per_sequence_state_bytes(&cfg, msl, true).unwrap());
    let on = per_sequence_state_bytes_with(&cfg, msl, true, true).unwrap();
    let per = pool_cache_state_bytes(dsa_capacity(msl, KPOOL), HEAD_DIM, KPOOL);
    assert_eq!(per, 76_369_924);
    assert_eq!(on.target_layers, DSA_LAYERS * per);
    let old = indexer_state_bytes(dsa_capacity(msl, KPOOL), HEAD_DIM);
    assert_eq!(
        off.proposer - old,
        on.proposer - per,
        "proposer scratch unchanged"
    );
    // 2026-10-06: 2.41 GB less per sequence at 540,672 tokens with MTP on.
    assert_eq!(off.total() - on.total(), 2_411_937_744);
}

/// 2026-10-06: The lazily mapped reserve with the pool cache: the pool backs pool keys and
/// ids (`LazyShape::pool_cache`), the eager part adds `pvalid`, `pk_len_dev` and the rings per
/// cache.
#[test]
fn the_lazy_pool_cache_reserve_maps_pool_arrays_and_keeps_rings_eager() {
    let cfg = glm53_config();
    let msl = 131_072;
    let off = lazy_indexer_reserve_with(&cfg, msl, true, 4, None, false)
        .unwrap()
        .unwrap();
    let lever_read = lazy_indexer_reserve(&cfg, msl, true, 4, None);
    assert_eq!(off, lever_read.unwrap().unwrap());
    let on = lazy_indexer_reserve_with(&cfg, msl, true, 4, None, true)
        .unwrap()
        .unwrap();
    assert!(on.pool.shape.pool_cache && !off.pool.shape.pool_cache);
    let ring = 2 * 8_448 * HEAD_DIM * 2;
    assert_eq!(
        on.eager_per_seq - off.eager_per_seq,
        12 * (msl / KPOOL + 4 + ring)
    );
    // 2026-10-06: Default pool, 4 x 32,768 tokens: 2 key granules + 1 id granule per target
    // cache, 3 + 1 for the proposer's (its 256-row look-ahead crosses a key granule); was 8 and
    // 10.
    assert_eq!(on.pool.limit_bytes, 4 * (11 * 3 + 4) * (2 << 20));
    assert_eq!(off.pool.limit_bytes, 784 << 20);
}
