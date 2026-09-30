// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host-only tests of the grouped-GEMM prefill: a host model of `moe_sort_by_expert`'s
//! outputs, the grid height, and the tile table.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Host model of `moe_sort_by_expert`'s outputs: `(sorted_token_ids,
/// sorted_expert_ids, expert_offsets, token_to_perm)`.
fn sort_ref(
    ids: &[u32],
    num_experts: usize,
    topk: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let te = ids.len();
    let mut counts = vec![0usize; num_experts];
    for &e in ids {
        counts[e as usize] += 1;
    }
    let mut offsets = vec![0i32; num_experts + 1];
    for e in 0..num_experts {
        offsets[e + 1] = offsets[e] + counts[e] as i32;
    }
    // 2026-09-25: The kernel places rows within an expert with `atomicAdd`, so their order is
    // unspecified; this model places them in order, and the tests assert only properties that do
    // not depend on that order.
    let mut cursor: Vec<i32> = offsets[..num_experts].to_vec();
    let mut stid = vec![-1i32; te];
    let mut seid = vec![-1i32; te];
    let mut t2p = vec![-1i32; te];
    for (i, &e) in ids.iter().enumerate() {
        let pos = cursor[e as usize];
        cursor[e as usize] += 1;
        stid[pos as usize] = (i / topk) as i32;
        seid[pos as usize] = e as i32;
        t2p[i] = pos;
    }
    (stid, seid, offsets, t2p)
}

fn routing(rows: usize, topk: usize, num_experts: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(rows * topk);
    for _ in 0..rows {
        // 2026-09-25: A row's top-k ids are distinct, as `glm5next_router_topk` produces them.
        let mut picked: Vec<u32> = Vec::with_capacity(topk);
        while picked.len() < topk {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let e = ((s >> 33) as usize % num_experts) as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        out.extend(picked);
    }
    out
}

/// 2026-09-25: The sort model's properties: `expert_offsets` is a prefix sum over `rows * top_k`
/// slots, each expert's range holds only that expert, `token_to_perm` is a bijection, and each
/// slot's sorted row carries its token and its expert.
#[test]
fn sort_contract_holds_at_glm_shapes() {
    for &(rows, topk, ne) in &[(16usize, 8usize, 288usize), (256, 8, 288), (5, 4, 7)] {
        let ids = routing(rows, topk, ne, 0xC0FFEE ^ rows as u64);
        let (stid, seid, offsets, t2p) = sort_ref(&ids, ne, topk);
        let te = rows * topk;

        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[ne], te as i32);
        for e in 0..ne {
            assert!(offsets[e + 1] >= offsets[e], "offsets not monotone at {e}");
        }
        for e in 0..ne {
            for p in offsets[e]..offsets[e + 1] {
                assert_eq!(seid[p as usize], e as i32, "expert block {e} is ragged");
            }
        }
        let mut seen = vec![false; te];
        for &p in &t2p {
            assert!(p >= 0 && (p as usize) < te, "perm {p} out of range");
            assert!(!seen[p as usize], "perm {p} claimed twice");
            seen[p as usize] = true;
        }
        for (i, &e) in ids.iter().enumerate() {
            let p = t2p[i] as usize;
            assert_eq!(stid[p], (i / topk) as i32, "slot {i} lost its token");
            assert_eq!(seid[p], e as i32, "slot {i} lost its expert");
        }
    }
}

/// 2026-09-25: `max_m_tiles_from_offsets` on balanced, fully skewed, 65-row and empty routings.
#[test]
fn max_m_tiles_covers_the_busiest_expert_and_never_exceeds_the_worst_case() {
    let balanced: Vec<i32> = (0..=288i32).map(|e| e * 2048 / 288).collect();
    assert_eq!(max_m_tiles_from_offsets(&balanced, 32, GROUPED_M_TILE), 1);

    let mut skewed = vec![0i32; 289];
    for o in skewed.iter_mut().skip(1) {
        *o = 2048;
    }
    assert_eq!(max_m_tiles_from_offsets(&skewed, 32, GROUPED_M_TILE), 32);

    let mut sixty_five = vec![0i32; 289];
    for (e, o) in sixty_five.iter_mut().enumerate() {
        *o = if e == 0 { 0 } else { 65 };
    }
    assert_eq!(max_m_tiles_from_offsets(&sixty_five, 32, GROUPED_M_TILE), 2);

    assert_eq!(max_m_tiles_from_offsets(&[0i32; 289], 1, GROUPED_M_TILE), 1);
}

/// 2026-09-25: For eight pseudo-random routings at 256 rows, top_k 8 and 288 experts, the bound
/// covers the busiest expert and stays within the worst case.
#[test]
fn max_m_tiles_is_never_short_for_a_real_routing() {
    for seed in 0..8u64 {
        let (rows, topk, ne) = (256usize, 8usize, 288usize);
        let ids = routing(rows, topk, ne, seed);
        let (_, _, offsets, _) = sort_ref(&ids, ne, topk);
        let worst = (rows * topk).div_ceil(GROUPED_M_TILE) as u32;
        let tiles = max_m_tiles_from_offsets(&offsets, worst, GROUPED_M_TILE);
        let busiest = (0..ne)
            .map(|e| offsets[e + 1] - offsets[e])
            .max()
            .unwrap_or(0) as u32;
        assert!(
            tiles * GROUPED_M_TILE as u32 >= busiest,
            "seed {seed}: {tiles} tiles cover {} rows, busiest expert has {busiest}",
            tiles * GROUPED_M_TILE as u32
        );
        assert!(
            tiles <= worst,
            "seed {seed}: {tiles} exceeds worst case {worst}"
        );
    }
}

/// 2026-09-25: Each tile's geometry follows its name: `m16` tiles have `m_tile` 16, `m128`
/// tiles 128 and the rest 64; `n128` and `m128` tiles have 256 threads, the rest 128; `n128`
/// tiles have `n_tile` 128, the rest 64; `mfast` tiles, and only they, have `m_fast`.
#[test]
fn every_gemm_tile_matches_its_kernel_geometry() {
    for t in GEMM_TILES {
        assert!(
            t.name.starts_with("moe_w4a16_grouped_gemm_ptrtable"),
            "{} is not an entry point of moe_w4a16_grouped_gemm.cu",
            t.name
        );
        let expect_m16 = t.name.contains("m16");
        let expect_m128 = t.name.contains("m128");
        assert_eq!(
            t.m_tile,
            if expect_m16 {
                16
            } else if expect_m128 {
                128
            } else {
                64
            },
            "{}: m_tile must equal the kernel's M_TILE or rows are dropped",
            t.name
        );
        let expect_n128 = t.name.contains("n128");
        assert_eq!(
            t.n_tile,
            if expect_n128 { 128 } else { 64 },
            "{}: n_tile must equal the kernel's NTILE or the output is half-written",
            t.name
        );
        assert_eq!(
            t.threads,
            if expect_n128 || expect_m128 { 256 } else { 128 },
            "{}: block width must equal WARPS*32 or the cooperative load is short",
            t.name
        );
        assert_eq!(
            t.m_fast,
            t.name.ends_with("_mfast"),
            "{}: m_fast must match the kernel's grid order or the wrong tiles are computed",
            t.name
        );
    }
}

/// 2026-09-25: `select_gemm_tile` resolves `base`, the empty string, a suffix and a full name, and
/// rejects near-misses.
#[test]
fn tile_selection_is_exact_and_rejects_unknown_names() {
    assert_eq!(select_gemm_tile("base").unwrap().name, GEMM_TILES[0].name);
    assert_eq!(select_gemm_tile("").unwrap().name, GEMM_TILES[0].name);
    assert_eq!(
        select_gemm_tile("bt_m16_k128").unwrap().name,
        DEFAULT_GEMM_TILE.name
    );
    assert_eq!(
        select_gemm_tile(DEFAULT_GEMM_TILE.name).unwrap().name,
        DEFAULT_GEMM_TILE.name
    );
    assert!(select_gemm_tile("bt_m16_k129").is_none());
    assert!(select_gemm_tile("m16").is_none());
}

/// 2026-09-25: With 40 rows on one expert the bound is 1 tile of 64 rows but 3 tiles of 16, and
/// the default tile's `m_tile` is 16.
#[test]
fn m16_tile_needs_more_rows_of_grid_than_the_base_tile() {
    let mut off = vec![0i32; 289];
    for (e, o) in off.iter_mut().enumerate() {
        *o = if e == 0 { 0 } else { 40 };
    }
    assert_eq!(max_m_tiles_from_offsets(&off, 128, 64), 1);
    assert_eq!(max_m_tiles_from_offsets(&off, 128, 16), 3);
    assert_eq!(DEFAULT_GEMM_TILE.m_tile, 16);
}

/// 2026-09-29: An `m_fast` tile puts the M tiles in grid x and the N tiles in grid y; any other
/// tile the reverse.
#[test]
fn grid_dims_follow_m_fast() {
    let t = select_gemm_tile("bt_m16_k128").unwrap();
    assert_eq!(t.grid_dims(2048, 7, 288), [32, 7, 288]);
    let f = select_gemm_tile("bt_m16_k128_mfast").unwrap();
    assert!(f.m_fast);
    assert_eq!(f.grid_dims(2048, 7, 288), [7, 32, 288]);
    let big = select_gemm_tile("bt_m128_k64_mfast").unwrap();
    assert_eq!((big.m_tile, big.threads), (128, 256));
    assert_eq!(big.grid_dims(4096, 2, 288), [2, 64, 288]);
}
