// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the post-load yardstick and the ring fit against it.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.
//!
//! The fixture is a Qwen3.8-27B-shaped FP8 serve on a 79.2 GiB device: weights
//! 28.75 GiB, a 3,658 MiB arena, and an inference reserve per batch size
//! that includes an 8-slot ring (4 slots at batch 64).

use super::super::decode_ring::{fit_ring, slot_bytes};
use super::*;

use clap::Parser as _;
use metrale_config::{LayerType, QuantizationConfig};

const GIB_F: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB: usize = 1024 * 1024;

/// 2026-09-26: 48 layers x (h + conv state) = 151.5 MiB, the expression
/// `decode_ring_tests.rs` uses.
const PER_SEQ_BLOB: usize = 48 * ((48 * 128 * 128 * 4) + ((16 * 128 * 2 + 48 * 128) * 4 * 4));

fn gib(x: f64) -> usize {
    (x * GIB_F) as usize
}

fn args(batch: usize) -> cli::ServeArgs {
    cli::ServeArgs::parse_from([
        "met",
        "Qwen/Qwen3.8-27B-FP8",
        "--max-batch-size",
        &batch.to_string(),
        "--max-seq-len",
        "24576",
        "--kv-cache-dtype",
        "fp8",
    ])
}

/// 2026-09-26: The config fixture `predicted_residency_tests.rs` builds.
fn qwen38_27b() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3_5".to_string();
    c.num_experts = 0;
    c.num_experts_per_tok = 0;
    c.moe_intermediate_size = 0;
    c.hidden_size = 5120;
    c.intermediate_size = 17408;
    c.num_hidden_layers = 64;
    c.num_attention_heads = 24;
    c.num_key_value_heads = 4;
    c.head_dim = 256;
    c.attn_gated = true;
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 48;
    c.linear_value_head_dim = 128;
    c.full_attention_interval = 4;
    c.layer_types = (0..64)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    c.quantization_config = Some(QuantizationConfig {
        quant_method: "fp8".to_string(),
        quant_algo: String::new(),
        format: String::new(),
        ignore_modules: Vec::new(),
    });
    c
}

/// 2026-09-26: The fixture's pre-KV terms. `fixed` is the reserve less an
/// 8-slot ring, since preflight composes the reserve as
/// `fixed_reserve + slots * slot_bytes`.
fn round6_headroom(batch: usize, reserve_mib: usize, util: f64) -> Headroom {
    let ring8 = 8 * batch * PER_SEQ_BLOB;
    let fixed = reserve_mib * MIB - ring8;
    let budget = (gib(79.2) as f64 * util) as usize;
    let weights = gib(28.75);
    // 2026-09-26: The prediction for this config
    // (`predicted_residency_tests::the_27b_prediction_matches_the_round6_residency_summary`).
    let derived = 4_242_882_560usize;
    let arena = 3658 * MIB;
    let a = args(batch);
    Headroom {
        budget,
        weights,
        derived,
        arena,
        fixed,
        kv_floor: kv_floor_bytes(&a, &qwen38_27b(), KvCacheDtype::Fp8),
        headroom: budget.saturating_sub(weights + derived + arena + fixed),
    }
}

fn fitted(batch: usize, reserve_mib: usize, util: f64) -> usize {
    let a = args(batch);
    let y = Yardstick::PostLoad(round6_headroom(batch, reserve_mib, util));
    fit_ring(
        &a,
        8,
        slot_bytes(&a, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        // 2026-09-26: `ladder_basis` ignores `reserve_without_ring` and
        // `free_mem` on the post-load yardstick.
        usize::MAX,
        0,
        &y,
        false,
    )
    .slots
}

/// 2026-09-26: 16 attention layers x (K + V) x 4 KV heads x 256 dims x 1
/// FP8 byte = 32 KiB per token, so the floor at 4096 tokens is 128 MiB per
/// sequence.
#[test]
fn the_kv_floor_is_the_cache_s_own_bytes_per_token() {
    let c = qwen38_27b();
    assert_eq!(c.num_attention_layers(), 16);
    assert_eq!(
        kv_bytes_per_token(&args(16), &c, KvCacheDtype::Fp8),
        16 * 2 * 4 * 256,
    );
    assert_eq!(kv_floor_tokens(), DEFAULT_KV_FLOOR_TOKENS);
    assert_eq!(kv_floor_bytes(&args(16), &c, KvCacheDtype::Fp8), 2 << 30);
    assert_eq!(kv_floor_bytes(&args(32), &c, KvCacheDtype::Fp8), 4 << 30);
    assert_eq!(kv_floor_bytes(&args(64), &c, KvCacheDtype::Fp8), 8 << 30);
    assert_eq!(
        kv_floor_bytes(&args(16), &c, KvCacheDtype::Bf16),
        2 * kv_floor_bytes(&args(16), &c, KvCacheDtype::Fp8),
    );
}

/// 2026-10-01: `glm5_next` aliases V onto K (`METRALE_GLM_KV_V_ALIAS`, on by
/// default), so the preflight prices K alone, as the engine's pool does.
#[test]
fn glm_kv_prices_k_only() {
    let mut c = qwen38_27b();
    c.model_type = "glm5_next".to_string();
    if std::env::var("METRALE_GLM_KV_V_ALIAS").as_deref() == Ok("0") {
        return;
    }
    assert_eq!(
        kv_bytes_per_token(&args(16), &c, KvCacheDtype::Fp8),
        16 * 4 * 256,
    );
}

/// 2026-09-26: `--max-seq-len` below the floor's token count caps the floor.
#[test]
fn the_floor_never_exceeds_max_seq_len() {
    let a = cli::ServeArgs::parse_from([
        "met",
        "Qwen/Qwen3.8-27B-FP8",
        "--max-batch-size",
        "16",
        "--max-seq-len",
        "1024",
    ]);
    assert_eq!(
        kv_floor_bytes(&a, &qwen38_27b(), KvCacheDtype::Fp8),
        16 * 1024 * 16 * 2 * 4 * 256,
    );
}

/// 2026-09-26: At batch 16 the 8-slot ring (18.94 GiB) and the 2 GiB floor
/// fit in about 29 GiB of headroom, so the depth stays 8.
#[test]
fn round6_batch16_keeps_the_full_ring() {
    let h = round6_headroom(16, 25_107, 0.90);
    assert!(
        h.headroom > 8 * 16 * PER_SEQ_BLOB + h.kv_floor,
        "headroom {:.2} GB must cover ring 18.94 + floor 2.00",
        h.headroom as f64 / GIB_F,
    );
    assert_eq!(fitted(16, 25_107, 0.90), 8);
}

/// 2026-09-26: At batch 32 the 8-slot ring (37.88 GiB) does not fit in about
/// 27 GiB of headroom, so the ladder drops to 4 (18.94 GiB).
#[test]
fn round6_batch32_fits_the_ring_the_operator_had_to_pin_by_hand() {
    assert_eq!(fitted(32, 46_923, 0.90), 4);
    let a = args(32);
    let h = round6_headroom(32, 46_923, 0.90);
    assert!(
        4 * slot_bytes(&a, PER_SEQ_BLOB) + h.kv_floor <= h.headroom,
        "the chosen depth must actually fit",
    );
    assert!(
        8 * slot_bytes(&a, PER_SEQ_BLOB) + h.kv_floor > h.headroom,
        "and depth 8 must not — otherwise this test proves nothing",
    );
}

/// 2026-09-26: At batch 64 the ring and the floor both double again, and the
/// ladder drops to 1.
#[test]
fn batch64_falls_to_a_single_anchor() {
    assert_eq!(fitted(64, 51_771 + (4 * 64 * PER_SEQ_BLOB) / MIB, 0.90), 1);
}

/// 2026-09-26: At `--gpu-memory-utilization 0.84` the batch-32 serve drops
/// one more rung, to 2; batch 16 keeps 8.
#[test]
fn a_tighter_utilization_costs_rollback_depth_not_concurrency() {
    assert_eq!(fitted(16, 25_107, 0.84), 8);
    assert_eq!(fitted(32, 46_923, 0.84), 2);
    assert!(
        fitted(32, 46_923, 0.84) < fitted(32, 46_923, 0.90),
        "a smaller budget must never buy MORE ring",
    );
}

/// 2026-09-26: The shrink warning names the yardstick, quotes the formula at
/// the fitted depth and the requested bytes, and names the flag that pins a
/// depth; the decision line names every term.
#[test]
fn the_shrink_warning_carries_the_formula_and_the_yardstick() {
    let a = args(32);
    let y = Yardstick::PostLoad(round6_headroom(32, 46_923, 0.90));
    let fit = fit_ring(
        &a,
        8,
        slot_bytes(&a, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        usize::MAX,
        0,
        &y,
        false,
    );
    let w = fit.warning.expect("a shrink must be logged, never silent");
    assert!(
        w.contains("ring: 4 slots x 32 seqs x 151.5 MB/seq = 18.94 GB"),
        "{w}"
    );
    assert!(w.contains("(was 37.88 GB)"), "{w}");
    assert!(w.contains("ring + KV floor"), "{w}");
    assert!(
        w.contains("Sized from the predicted post-load KV headroom"),
        "{w}"
    );
    assert!(w.contains("--ssm-decode-ring-slots"), "{w}");
    for term in [
        "budget",
        "weights",
        "derived",
        "arena",
        "fixed reserve",
        "headroom",
        "ring(4)",
        "KV floor",
    ] {
        assert!(
            fit.decision.contains(term),
            "{term} missing: {}",
            fit.decision
        );
    }
}

/// 2026-09-26: An explicit depth is kept on the post-load yardstick too, with
/// no warning.
#[test]
fn an_explicit_depth_is_not_fitted_against_the_headroom_either() {
    let a = args(32);
    let y = Yardstick::PostLoad(round6_headroom(32, 46_923, 0.90));
    let fit = fit_ring(
        &a,
        8,
        slot_bytes(&a, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        usize::MAX,
        0,
        &y,
        true,
    );
    assert_eq!(fit.slots, 8);
    assert!(fit.warning.is_none());
}

/// 2026-09-26: When the KV floor alone exceeds the headroom, the post-load fit
/// goes to depth 0 with a warning that defers to the KV budget stage.
#[test]
fn an_unmeetable_floor_shrinks_to_zero_and_defers_to_the_kv_stage() {
    let a = args(32);
    let mut h = round6_headroom(32, 46_923, 0.90);
    h.headroom = h.kv_floor / 2;
    let fit = fit_ring(
        &a,
        8,
        slot_bytes(&a, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        usize::MAX,
        0,
        &Yardstick::PostLoad(h),
        false,
    );
    assert_eq!(fit.slots, 0);
    let w = fit.warning.expect("this must not be silent");
    assert!(w.contains("the KV budget stage will decide"), "{w}");
}

/// 2026-09-26: A route with no residency prediction gets the pre-load
/// yardstick, with a nonempty reason.
#[test]
fn an_unpredictable_route_falls_back_to_pre_load_free_memory_and_says_so() {
    let a = args(32);
    let mut plain = qwen38_27b();
    plain.quantization_config = None;
    let y = post_load_yardstick(
        &a,
        &plain,
        &PostLoadInputs {
            total_mem: gib(79.2),
            model_dir: std::path::Path::new("/nonexistent-checkpoint"),
            kv_dtype: KvCacheDtype::Fp8,
            w8a8_prefill_kernels: true,
        },
        gib(8.0),
        gib(3.5),
    );
    // 2026-09-26: Which reason comes back depends on `METRALE_DENSE_FP8` in
    // the process environment, so only its presence is asserted.
    match y {
        Yardstick::PreLoadFree(why) => assert!(!why.is_empty(), "the fallback must say why"),
        Yardstick::PostLoad(_) => panic!("must not predict a route it cannot see"),
    }
}

/// 2026-09-26: A missing directory gives `None`, and without an index only
/// the `.safetensors` files are summed.
#[test]
fn a_missing_checkpoint_is_a_fallback_not_a_zero_weight_estimate() {
    assert_eq!(
        checkpoint_bytes(std::path::Path::new("/nonexistent-ckpt")),
        None
    );
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("a-00001-of-00002.safetensors"),
        vec![7u8; 2048],
    )
    .unwrap();
    std::fs::write(
        dir.path().join("a-00002-of-00002.safetensors"),
        vec![7u8; 1024],
    )
    .unwrap();
    std::fs::write(dir.path().join("README.md"), b"not a weight").unwrap();
    assert_eq!(checkpoint_bytes(dir.path()), Some(3072));
}

/// 2026-09-26: Each shard the index names is counted once, and the index's
/// `metadata.total_size` is not used.
#[test]
fn an_index_counts_each_shard_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("s1.safetensors"), vec![0u8; 1000]).unwrap();
    std::fs::write(dir.path().join("s2.safetensors"), vec![0u8; 500]).unwrap();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        br#"{"metadata":{"total_size":9999},"weight_map":{
            "a":"s1.safetensors","b":"s1.safetensors","c":"s2.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(checkpoint_bytes(dir.path()), Some(1500));
}

/// 2026-09-26: `autofit` publishes the fitted depth. This is the only unit
/// test in the crate that writes the process-global depth cell.
#[test]
fn the_fitted_depth_is_published_for_the_allocation_side() {
    let a = args(32);
    let y = Yardstick::PostLoad(round6_headroom(32, 46_923, 0.90));
    assert_eq!(
        metrale_model_layers::ssm_reserve::published_decode_ring_slots(),
        None
    );
    let fit = super::super::decode_ring::autofit(
        &a,
        8,
        slot_bytes(&a, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        usize::MAX,
        0,
        &y,
    );
    assert_eq!(fit.slots, 4);
    assert_eq!(
        metrale_model_layers::ssm_reserve::published_decode_ring_slots(),
        Some(4),
        "the reserve and the allocation must read one cell",
    );
}
