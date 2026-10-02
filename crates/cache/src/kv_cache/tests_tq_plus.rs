// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `KvCacheDtype` coverage tests for the turbo formats.
//!
//! 1. Every variant listed in `ALL_VARIANTS` round-trips through `Display`
//!    and `FromStr`, and `kv_pair` and `is_asymmetric` agree with its
//!    symmetric or asymmetric listing. The lists are written by hand, so a
//!    new variant is covered only once it is added to them.
//! 2. Block bytes match the bits per element of turbo3 (3.5), turbo2 (2.5)
//!    and turbo8 (9, with a BF16 scale), and `Bf16KTurbo3V` sizes K and V
//!    separately.
//!
//! These tests cannot see a `match` on the dtype elsewhere (the dispatch in
//! `metrale-model-layers`) that sends a variant to a `_` arm;
//! `tests/test_kv_dtype_smoke.py` exercises that end to end.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use super::{KvCacheConfig, KvCacheDtype};

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

const ALL_VARIANTS: &[KvCacheDtype] = &[
    KvCacheDtype::Bf16,
    KvCacheDtype::Fp8,
    KvCacheDtype::Nvfp4,
    KvCacheDtype::Turbo4,
    KvCacheDtype::Turbo3,
    KvCacheDtype::Turbo2,
    KvCacheDtype::Turbo8,
    KvCacheDtype::Turbo4KTurbo3V,
    KvCacheDtype::Turbo4KTurbo8V,
    KvCacheDtype::Turbo3KTurbo8V,
    KvCacheDtype::Bf16KTurbo4V,
    KvCacheDtype::Bf16KTurbo3V,
    KvCacheDtype::Bf16KTurbo2V,
    KvCacheDtype::Fp8KTurbo4V,
    KvCacheDtype::Fp8KTurbo3V,
    KvCacheDtype::Fp8KTurbo2V,
];

const ASYM_VARIANTS: &[KvCacheDtype] = &[
    KvCacheDtype::Turbo4KTurbo3V,
    KvCacheDtype::Turbo4KTurbo8V,
    KvCacheDtype::Turbo3KTurbo8V,
    KvCacheDtype::Bf16KTurbo4V,
    KvCacheDtype::Bf16KTurbo3V,
    KvCacheDtype::Bf16KTurbo2V,
    KvCacheDtype::Fp8KTurbo4V,
    KvCacheDtype::Fp8KTurbo3V,
    KvCacheDtype::Fp8KTurbo2V,
];

const SYM_VARIANTS: &[KvCacheDtype] = &[
    KvCacheDtype::Bf16,
    KvCacheDtype::Fp8,
    KvCacheDtype::Nvfp4,
    KvCacheDtype::Turbo4,
    KvCacheDtype::Turbo3,
    KvCacheDtype::Turbo2,
    KvCacheDtype::Turbo8,
];

#[test]
fn display_fromstr_roundtrip_all_variants() {
    use std::str::FromStr;
    for &d in ALL_VARIANTS {
        let s = d.to_string();
        let parsed = KvCacheDtype::from_str(&s)
            .unwrap_or_else(|e| panic!("{d:?} → {s:?} did not parse: {e}"));
        assert_eq!(parsed, d, "Display/FromStr roundtrip broken for {d:?}");
    }
}

#[test]
fn fromstr_short_alias_parses_to_canonical_variant() {
    use std::str::FromStr;
    let aliases = [
        ("turbo4k3v", KvCacheDtype::Turbo4KTurbo3V),
        ("turbo4k8v", KvCacheDtype::Turbo4KTurbo8V),
        ("turbo3k8v", KvCacheDtype::Turbo3KTurbo8V),
        ("bf16k4v", KvCacheDtype::Bf16KTurbo4V),
        ("bf16k3v", KvCacheDtype::Bf16KTurbo3V),
        ("bf16k2v", KvCacheDtype::Bf16KTurbo2V),
        ("fp8k4v", KvCacheDtype::Fp8KTurbo4V),
        ("fp8k3v", KvCacheDtype::Fp8KTurbo3V),
        ("fp8k2v", KvCacheDtype::Fp8KTurbo2V),
    ];
    for (s, expected) in aliases {
        assert_eq!(KvCacheDtype::from_str(s).unwrap(), expected, "alias {s}");
    }
}

#[test]
fn asym_is_asymmetric_and_pair_differs() {
    for &d in ASYM_VARIANTS {
        assert!(d.is_asymmetric(), "{d:?} should be asymmetric");
        let (k, v) = d.kv_pair();
        assert_ne!(k, v, "{d:?} kv_pair should differ (K={k:?}, V={v:?})");
    }
}

#[test]
fn sym_is_symmetric_and_pair_self() {
    for &d in SYM_VARIANTS {
        assert!(!d.is_asymmetric(), "{d:?} should be symmetric");
        let (k, v) = d.kv_pair();
        assert_eq!(k, d, "{d:?} kv_pair K should be self");
        assert_eq!(v, d, "{d:?} kv_pair V should be self");
    }
}

#[test]
fn asym_kv_pair_components_are_symmetric() {
    // 2026-09-25: Both sides of an asymmetric pair must be symmetric formats.
    for &d in ASYM_VARIANTS {
        let (k, v) = d.kv_pair();
        assert!(
            !k.is_asymmetric(),
            "{d:?} K side {k:?} is itself asymmetric"
        );
        assert!(
            !v.is_asymmetric(),
            "{d:?} V side {v:?} is itself asymmetric"
        );
    }
}

// 2026-09-25: Expected block bytes, from data bits plus per-group scale bits
// per element:
//
//   nvfp4/turbo4: 4 b data + 0.5 b scale = 4.5 b/elem
//   turbo3:       3 b data + 0.5 b scale = 3.5 b/elem
//   turbo2:       2 b data + 0.5 b scale = 2.5 b/elem
//   turbo8:       8 b data + 1.0 b scale (BF16) = 9.0 b/elem
//
// At test_config (block_size=16, num_kv_heads=2, head_dim=256, GROUP_SIZE=16),
// n_elems = 512 per token, num_groups = 32. Per-token bytes:
//   nvfp4: 256 data + 32 scale = 288
//   turbo3: 192 data + 32 scale = 224
//   turbo2: 128 data + 32 scale = 160
//   turbo8: 512 data + 64 scale = 576
// Block (16 tokens) bytes:
//   nvfp4: 16 * 288 = 4608
//   turbo3: 16 * 224 = 3584
//   turbo2: 16 * 160 = 2560
//   turbo8: 16 * 576 = 9216

#[test]
fn block_bytes_turbo3() {
    let cfg = KvCacheConfig {
        dtype: KvCacheDtype::Turbo3,
        ..test_config()
    };
    assert_eq!(cfg.turbo3_data_bytes(), 16 * 2 * 256 * 3 / 8, "turbo3 data");
    assert_eq!(cfg.block_bytes(), 3584, "turbo3 block (data + scales)");
}

#[test]
fn block_bytes_turbo2() {
    let cfg = KvCacheConfig {
        dtype: KvCacheDtype::Turbo2,
        ..test_config()
    };
    assert_eq!(cfg.turbo2_data_bytes(), 16 * 2 * 256 / 4, "turbo2 data");
    assert_eq!(cfg.block_bytes(), 2560, "turbo2 block (data + scales)");
}

#[test]
fn block_bytes_turbo8() {
    let cfg = KvCacheConfig {
        dtype: KvCacheDtype::Turbo8,
        ..test_config()
    };
    assert_eq!(cfg.turbo8_data_bytes(), 16 * 2 * 256, "turbo8 data");
    // 2026-09-25: Turbo8 scales are BF16, 2 bytes per group.
    let n_groups = 16 * 2 * 256 / 16;
    assert_eq!(
        cfg.block_bytes(),
        16 * 2 * 256 + n_groups * 2,
        "turbo8 block"
    );
}

#[test]
fn asym_bf16k_turbo3v_uses_separate_strides() {
    // 2026-09-25: K is BF16 (16 bits per element), V is turbo3 (3.5 bits per
    // element with its scales).
    let cfg = KvCacheConfig {
        dtype: KvCacheDtype::Bf16KTurbo3V,
        ..test_config()
    };
    let k_bytes = cfg.k_block_bytes_for_layer(0);
    let v_bytes = cfg.v_block_bytes_for_layer(0);
    assert_eq!(k_bytes, 16 * 2 * 256 * 2, "K block (bf16)");
    assert_eq!(v_bytes, 3584, "V block (turbo3 with scales)");
    assert_ne!(
        k_bytes, v_bytes,
        "Bf16KTurbo3V must use separate K/V strides"
    );
}
