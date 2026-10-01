// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host tests of the selection geometry: the pool counts against
//! `glm5next_dsa_ref::kept_pools`, the top-k tile and its limit, and a host model of the
//! tiled top-k.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa_ref::{DsaDims, kept_pools};

/// 2026-09-25: GLM-5.3 DSA geometry (`kernels/gb10/glm-5.3-flash/MODEL.toml`) at TP 1.
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

fn dims(c: &Glm5NextDsaConfig) -> DsaDims {
    DsaDims {
        hidden: c.hidden,
        index_heads: c.index_heads,
        index_head_dim: c.index_head_dim,
        index_kpool: c.index_kpool,
        index_topk: c.index_topk,
        always_select_tail: c.always_select_tail,
        q_lora_rank: c.q_lora_rank,
        heads: c.local_heads,
        kv_lora_rank: c.kv_lora_rank,
        qk_nope_head_dim: c.qk_nope_head_dim,
        qk_rope_head_dim: c.qk_rope_head_dim,
        v_head_dim: c.v_head_dim,
    }
}

/// 2026-09-25: Over a contiguous, all-valid cache, `kept_pools` returns exactly
/// `0 .. contiguous_pool_count(kpool, seq)`, for kpool 1, 2, 3, 4 and 8 and every seq in
/// `0..=257`. This is what lets `select_tokens` skip `dsa_compact_pools`.
#[test]
fn contiguous_pool_count_equals_the_reference_kept_set() {
    for kpool in [1usize, 2, 3, 4, 8] {
        let mut c = cfg();
        c.index_kpool = kpool;
        c.index_topk = kpool * 512;
        let d = dims(&c);
        for seq in 0usize..=257 {
            let valid = vec![1u8; seq];
            let reference = kept_pools(&valid, d, seq);
            let ours = contiguous_pool_count(kpool, seq);
            assert_eq!(
                reference.len(),
                ours,
                "kpool={kpool} seq={seq}: count disagrees with kept_pools"
            );
            let expected: Vec<i32> = (0..ours as i32).collect();
            assert_eq!(
                reference, expected,
                "kpool={kpool} seq={seq}: kept pools are not the leading prefix, so \
                 skipping dsa_compact_pools would misalign every downstream index"
            );
        }
    }
}

/// 2026-09-25: The full pool count includes the trailing partial pool; the kept count does
/// not.
#[test]
fn full_and_kept_pool_counts_differ_exactly_on_a_partial_tail() {
    let c = cfg();
    for seq in [4usize, 5, 7, 8, 4096, 4097] {
        let g = DsaSelectGeometry::plan(&c, seq, 1).unwrap();
        assert_eq!(g.n_pools, seq / 4, "seq={seq} kept");
        assert_eq!(g.n_pools_full, seq.div_ceil(4), "seq={seq} full");
        assert!(g.n_pools_full >= g.n_pools);
        assert_eq!(
            g.n_pools_full - g.n_pools,
            usize::from(!seq.is_multiple_of(4))
        );
    }
}

/// 2026-09-25: `plan` keeps the top-k at one 2,048-pool tile (32,768 B of shared memory)
/// from 16,384 to 262,144 tokens, and uses a smaller tile below one tile of pools.
#[test]
fn plan_holds_shared_memory_constant_at_any_context() {
    let c = cfg();
    let tile = topk_tile();
    assert_eq!(
        tile, 2_048,
        "two tiles of [f32,i32] against a 49,152 B ceiling"
    );
    assert_eq!(topk_smem_for_tile(tile), 32_768);

    for (seq, pools) in [
        (16_384usize, 4_096usize),
        (16_388, 4_097),
        (65_536, 16_384),
        (262_144, 65_536),
    ] {
        let g = DsaSelectGeometry::plan(&c, seq, 1)
            .unwrap_or_else(|e| panic!("plan refused {seq} tokens: {e}"));
        assert_eq!(g.n_pools, pools);
        assert_eq!(
            g.topk_np2, tile,
            "the sort axis is the tile, not the context"
        );
        assert_eq!(g.topk_smem, 32_768);
        assert!(g.topk_smem <= TOPK_SMEM_CEILING);
    }

    let small = DsaSelectGeometry::plan(&c, 1_200, 1).unwrap();
    assert_eq!(small.n_pools, 300);
    assert_eq!(small.topk_np2, 512);
    assert_eq!(small.topk_smem, 8_192);
}

/// 2026-09-25: `plan` refuses a `select_k` wider than one tile, since the running best list
/// is one tile. At GLM-5.3's geometry `select_k` is at most 512.
#[test]
fn plan_refuses_a_select_k_wider_than_one_tile() {
    let mut c = cfg();
    c.index_topk = topk_tile() * c.index_kpool * 2;
    let err = DsaSelectGeometry::plan(&c, 262_144, 1)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("exceeds the") && err.contains("top-k tile"),
        "the refusal must name the tile: {err}"
    );
}

/// 2026-09-25: A host model of the `dsa_topk_pools` walk (bitonic sort of each tile, a
/// half-cleaner against the running best list, a bitonic merge), checked against a sort of
/// the whole axis under the same order. It tests the algorithm, not the CUDA code.
mod tiled_select_model {
    /// 2026-09-25: The kernel's comparator (`DSA_TOPK_GT`): score descending, then pool index
    /// ascending. Indices are unique, so this is a total order and the top-k prefix is
    /// unique.
    fn gt(a: (f32, i32), b: (f32, i32)) -> bool {
        a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
    }

    fn bitonic_sort_desc(v: &mut [(f32, i32)]) {
        let n = v.len();
        let mut k = 2;
        while k <= n {
            let mut j = k >> 1;
            while j > 0 {
                for i in 0..n {
                    let l = i ^ j;
                    if l > i {
                        let want_desc = (i & k) == 0;
                        if want_desc != gt(v[i], v[l]) {
                            v.swap(i, l);
                        }
                    }
                }
                j >>= 1;
            }
            k <<= 1;
        }
    }

    fn bitonic_merge_desc(v: &mut [(f32, i32)]) {
        let n = v.len();
        let mut j = n >> 1;
        while j > 0 {
            for i in 0..n {
                let l = i ^ j;
                if l > i && !gt(v[i], v[l]) {
                    v.swap(i, l);
                }
            }
            j >>= 1;
        }
    }

    /// 2026-09-25: `dsa_topk_pools` for one query row.
    pub fn tiled_topk(scores: &[f32], tile: usize, select_k: usize) -> Vec<i32> {
        let pad = (f32::NEG_INFINITY, i32::MAX);
        let mut best = vec![pad; tile];
        let mut base = 0;
        while base < scores.len() {
            let mut cand: Vec<(f32, i32)> = (0..tile)
                .map(|i| {
                    let idx = base + i;
                    if idx < scores.len() {
                        (scores[idx], idx as i32)
                    } else {
                        pad
                    }
                })
                .collect();
            bitonic_sort_desc(&mut cand);
            // 2026-09-25: Half-cleaner: best[i] against cand[tile-1-i].
            for i in 0..tile {
                let b = tile - 1 - i;
                if !gt(best[i], cand[b]) {
                    std::mem::swap(&mut best[i], &mut cand[b]);
                }
            }
            bitonic_merge_desc(&mut best);
            base += tile;
        }
        best.into_iter().take(select_k).map(|e| e.1).collect()
    }

    /// 2026-09-25: The reference: sort the whole axis under the same order, take the prefix.
    pub fn whole_axis_topk(scores: &[f32], select_k: usize) -> Vec<i32> {
        let mut all: Vec<(f32, i32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i as i32))
            .collect();
        all.sort_by(|a, b| {
            if gt(*a, *b) {
                std::cmp::Ordering::Less
            } else if a == b {
                std::cmp::Ordering::Equal
            } else {
                std::cmp::Ordering::Greater
            }
        });
        all.into_iter().take(select_k).map(|e| e.1).collect()
    }
}

#[test]
fn the_tiled_walk_returns_exactly_what_a_whole_axis_sort_would() {
    // 2026-09-25: A fixed-seed LCG. The 3-value alphabet forces tied scores, which exercise
    // the pool-index tie-break.
    let mut state: u64 = 0x5EED_5EED;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    for &tile in &[4usize, 8, 64] {
        for &n_pools in &[1usize, 3, 7, 8, 9, 33, 64, 65, 200, 511, 512] {
            for &alphabet in &[3u32, 1_000_000] {
                let scores: Vec<f32> = (0..n_pools).map(|_| (next() % alphabet) as f32).collect();
                for &select_k in &[1usize, 2, 5, tile.min(n_pools)] {
                    let select_k = select_k.min(n_pools).min(tile).max(1);
                    let got = tiled_select_model::tiled_topk(&scores, tile, select_k);
                    let want = tiled_select_model::whole_axis_topk(&scores, select_k);
                    assert_eq!(
                        got, want,
                        "tile={tile} n_pools={n_pools} alphabet={alphabet} select_k={select_k}"
                    );
                }
            }
        }
    }
}

/// 2026-09-25: `select_k` is `index_topk / index_kpool`, clamped to the pools that exist.
#[test]
fn select_k_clamps_to_available_pools() {
    let c = cfg();
    let long = DsaSelectGeometry::plan(&c, 8_192, 1).unwrap();
    assert_eq!(long.n_pools, 2_048);
    assert_eq!(
        long.select_k, 512,
        "budget applies when pools are plentiful"
    );

    let short = DsaSelectGeometry::plan(&c, 400, 1).unwrap();
    assert_eq!(short.n_pools, 100);
    assert_eq!(
        short.select_k, 100,
        "clamped: cannot select 512 of 100 pools"
    );
}

/// 2026-09-25: The emitted row is `index_topk` wide plus `index_kpool - 1` tail slots when
/// `always_select_tail`. `dsa_expand_selection` writes no slot at or past `width`, so a row
/// sized without the tail would drop the in-progress pool with no error.
#[test]
fn out_width_carries_the_always_select_tail_slots() {
    let mut c = cfg();
    assert!(c.always_select_tail);
    assert_eq!(c.out_width(), 2_048 + 3);
    c.always_select_tail = false;
    assert_eq!(c.out_width(), 2_048);
}

/// 2026-09-25: An 8,192-token pass needs more scratch than a 4,096-token plan, which is the
/// comparison `DsaSelectScratch::fits` makes.
#[test]
fn scratch_refuses_a_pass_larger_than_its_reservation() {
    let c = cfg();
    let reserved = DsaSelectGeometry::plan(&c, 4_096, 1).unwrap();
    let grown = DsaSelectGeometry::plan(&c, 8_192, 1).unwrap();

    // 2026-09-25: `fits` compares these byte counts, so the reservation is modelled without
    // allocating.
    let cap = reserved.scratch_bytes();
    let want = grown.scratch_bytes();
    assert!(
        want.iter().zip(cap.iter()).any(|(w, c)| w > c),
        "an 8,192-token pass must exceed a 4,096-token reservation somewhere"
    );
}

/// 2026-09-25: `plan` refuses zero query rows and zero tokens, and plans one pool and fewer
/// tokens than one pool (a tail-only pass).
#[test]
fn plan_refuses_degenerate_geometry() {
    let c = cfg();
    assert!(DsaSelectGeometry::plan(&c, 4_096, 0).is_err(), "q_rows = 0");
    assert!(DsaSelectGeometry::plan(&c, 0, 1).is_err(), "seq = 0");
    assert!(
        DsaSelectGeometry::plan(&c, 4, 1).is_ok(),
        "exactly one pool is fine"
    );
    assert!(
        DsaSelectGeometry::plan(&c, 3, 1).is_ok(),
        "fewer tokens than one pool is the tail-only regime, not a refusal"
    );
}

/// 2026-09-25: Below `index_kpool` tokens, `plan` gives no pools and `select_k` 0,
/// `kept_pools` keeps none, and the reference `expand_selection` returns `[0 .. seq)` then
/// -1 padding: every visible token is selected.
#[test]
fn sub_pool_selection_is_dense_over_the_visible_tokens() {
    use crate::glm5next_dsa_ref::{INVALID, Pools, expand_selection};

    let c = cfg();
    let d = dims(&c);
    let width = d.out_width();

    for seq in 1usize..c.index_kpool {
        let g = DsaSelectGeometry::plan(&c, seq, 1).unwrap();
        assert_eq!(g.n_pools, 0, "seq={seq}: no complete pool");
        assert_eq!(g.select_k, 0, "seq={seq}: nothing to select");
        assert_eq!(g.out_width, width, "seq={seq}: row width is unchanged");

        let valid = vec![1u8; seq];
        assert!(
            kept_pools(&valid, d, seq).is_empty(),
            "seq={seq}: the reference keeps no pool either"
        );

        let pools = Pools {
            keys: Vec::new(),
            indices: Vec::new(),
            valid: Vec::new(),
            n_pools: 0,
        };
        let row = expand_selection(&[], &pools, &[], &valid, &[seq - 1], &[1u8], d, seq, 0);

        let mut expected = vec![INVALID; width];
        for (t, e) in expected.iter_mut().enumerate().take(seq) {
            *e = t as i32;
        }
        assert_eq!(
            row, expected,
            "seq={seq}: the row must be every visible token, then -1 padding"
        );
    }
}

/// 2026-09-25: At `index_kpool` tokens `plan` has one pool and `select_k` 1; one token fewer
/// has neither.
#[test]
fn the_first_complete_pool_switches_the_sparse_arm_on() {
    let c = cfg();
    let g3 = DsaSelectGeometry::plan(&c, 3, 1).unwrap();
    let g4 = DsaSelectGeometry::plan(&c, 4, 1).unwrap();
    assert_eq!((g3.n_pools, g3.select_k), (0, 0));
    assert_eq!((g4.n_pools, g4.select_k), (1, 1));
}

/// 2026-10-01: `METRALE_GLM_DSA_SCORES_TILED` takes the tiled kernel only when requested,
/// resolved, on an exact host-geometry launch and inside the D/H envelope.
#[test]
fn dsa_tiled_scores_need_request_handle_exact_launch_and_envelope() {
    assert!(scores_tiled_for(true, true, true, 128, 32), "GLM-5.3: D 128, 32 heads");
    assert!(scores_tiled_for(true, true, true, 96, 64));
    assert!(scores_tiled_for(true, true, true, 32, 1));
    assert!(!scores_tiled_for(false, true, true, 128, 32), "not requested");
    assert!(!scores_tiled_for(true, false, true, 128, 32), "handle 0");
    assert!(!scores_tiled_for(true, true, false, 128, 32), "ceiling / geom_dev launch");
    for d in [0, 16, 100, 160, 256] {
        assert!(!scores_tiled_for(true, true, true, d, 32), "D {d}");
    }
    for h in [0, 65, 128] {
        assert!(!scores_tiled_for(true, true, true, 128, h), "{h} heads");
    }
}

/// 2026-10-01: The tiled launch's shared memory stays under the 48 KiB default across the
/// envelope, and its grid covers every (row, pool) once.
#[test]
fn dsa_tiled_smem_and_grid_cover_the_envelope() {
    assert_eq!(scores_tiled_smem(128), 41_216);
    assert!(scores_tiled_smem(SCORES_TILED_MAX_D) <= TOPK_SMEM_CEILING);
    assert_eq!(scores_tiled_grid(1, 1), [1, 1, 1]);
    assert_eq!(scores_tiled_grid(16, 64), [1, 1, 1]);
    assert_eq!(scores_tiled_grid(17, 65), [2, 2, 1]);
    assert_eq!(scores_tiled_grid(8192, 2048), [32, 512, 1]);
    assert_eq!(
        SCORES_TILED_BLOCK as usize * 4,
        SCORES_TILE_ROWS * SCORES_TILE_POOLS,
        "four outputs per thread"
    );
}

/// 2026-10-01: `dsa_indexer.cu` defines `dsa_index_scores_tiled` with `dsa_index_scores`'
/// parameter list, and its tile defines mirror the Rust constants; a rename or a resize in
/// the `.cu` alone fails here instead of silently falling back (or mis-launching) at serve.
#[test]
fn dsa_tiled_entry_point_and_tile_match_the_kernel_file() {
    let cu = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common")
        .join(format!("{}.cu", super::super::DSA_MODULE));
    let src = std::fs::read_to_string(&cu).expect("dsa_indexer.cu readable");
    let params = |name: &str| -> String {
        let head = format!("extern \"C\" __global__ void {name}(");
        let Some(at) = src.find(&head) else {
            panic!("{cu:?} lacks {name}");
        };
        let start = at + head.len();
        let len = src[start..].find(')').expect("parameter list closes");
        src[start..start + len]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    assert_eq!(
        params("dsa_index_scores_tiled"),
        params("dsa_index_scores"),
        "the launcher passes one argument list to both kernels"
    );
    for (define, v) in [
        ("DSA_TILE_ROWS", SCORES_TILE_ROWS),
        ("DSA_TILE_POOLS", SCORES_TILE_POOLS),
        ("DSA_TILE_THREADS", SCORES_TILED_BLOCK as usize),
    ] {
        let line = format!("#define {define} {v}u");
        assert!(
            src.lines().any(|l| l.trim() == line),
            "{cu:?}: `{line}` does not match the Rust constant"
        );
    }
}
