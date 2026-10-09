// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Host tests of the two-rank DFlash drafter maths: shard ranges, weight
//! partition coverage, the top-16 merge, the header codec and the follower's conform plan.

use super::header::{self, TpHeader};
use super::shard::*;

/// 2026-10-09: Deterministic pseudo-random stream (xorshift64*), so the tests need no crate.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

#[test]
fn shard_range_covers_total_exactly_once() {
    for &(total, align) in &[
        (154_880, 128),
        (1, 1),
        (7, 1),
        (12_288, 128),
        (128 * 1211, 128),
    ] {
        for world in 1..=5 {
            let mut next = 0;
            for rank in 0..world {
                let (s, l) = shard_range(total, world, rank, align).unwrap();
                assert_eq!(s, next, "total {total} world {world} rank {rank}");
                assert!(s.is_multiple_of(align) && l.is_multiple_of(align));
                next = s + l;
            }
            assert_eq!(next, total);
        }
    }
    // 2026-10-09: Uneven: 1211 units over 2 ranks give 606 + 605; 7 over 2 give 4 + 3.
    assert_eq!(shard_range(128 * 1211, 2, 0, 128).unwrap(), (0, 606 * 128));
    assert_eq!(
        shard_range(128 * 1211, 2, 1, 128).unwrap(),
        (606 * 128, 605 * 128)
    );
    assert_eq!(shard_range(7, 2, 1, 1).unwrap(), (4, 3));
    assert!(
        shard_range(100, 2, 0, 128).is_err(),
        "not a multiple of align"
    );
    assert!(shard_range(128, 2, 2, 1).is_err(), "rank out of range");
}

fn geom(world: usize) -> TpGeometry {
    TpGeometry {
        world,
        q_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        hidden: 4096,
        inter: 128 * 97,
        vocab: 154_880,
        fc_in: 5 * 6144,
    }
}

#[test]
fn geometry_check_rejects_unsplittable_shapes() {
    assert!(geom(2).check().is_ok());
    assert!(
        TpGeometry {
            kv_heads: 1,
            q_heads: 32,
            ..geom(2)
        }
        .check()
        .is_err()
    );
    assert!(
        TpGeometry {
            kv_heads: 3,
            q_heads: 30,
            ..geom(2)
        }
        .check()
        .is_err()
    );
    assert!(
        TpGeometry {
            q_heads: 30,
            ..geom(2)
        }
        .check()
        .is_err()
    );
    assert!(
        TpGeometry {
            vocab: 154_881,
            ..geom(2)
        }
        .check()
        .is_err()
    );
    assert!(
        TpGeometry {
            hidden: 4096 + 128,
            ..geom(2)
        }
        .check()
        .is_err()
    );
    assert!(
        TpGeometry {
            world: 1,
            ..geom(2)
        }
        .check()
        .is_err()
    );
}

/// 2026-10-09: Every cell of every weight is owned by exactly one rank: row-split parts tile
/// the rows with full columns, column-split parts tile the columns with full rows.
#[test]
fn weight_partitions_cover_each_matrix_exactly_once() {
    for g in [
        geom(2),
        TpGeometry {
            inter: 128 * 3,
            vocab: 128 * 5,
            ..geom(2)
        },
    ] {
        let per_rank: Vec<Vec<WeightPart>> =
            (0..g.world).map(|r| weight_parts(&g, r).unwrap()).collect();
        for w in 0..per_rank[0].len() {
            let parts: Vec<WeightPart> = per_rank.iter().map(|p| p[w]).collect();
            let (n, k, split) = (parts[0].n, parts[0].k, parts[0].split);
            assert!(
                parts
                    .iter()
                    .all(|p| p.n == n && p.k == k && p.split == split)
            );
            let extent = match split {
                Split::Rows => n,
                Split::Cols => k,
            };
            // 2026-10-09: Count owners per index along the split axis; the other axis is
            // whole in every part, so this is the per-cell owner count.
            let mut owners = vec![0u8; extent];
            for p in &parts {
                assert!(p.start + p.len <= extent, "{} past the matrix", p.name);
                for o in &mut owners[p.start..p.start + p.len] {
                    *o += 1;
                }
            }
            assert!(
                owners.iter().all(|&o| o == 1),
                "{} not covered once",
                parts[0].name
            );
        }
        // 2026-10-09: q heads follow their kv group: rank r's q heads are its kv heads
        // times the GQA ratio.
        for r in 0..g.world {
            let l = g.layout(r).unwrap();
            let ratio = g.q_heads / g.kv_heads;
            assert_eq!(l.q_heads, (l.kv_heads.0 * ratio, l.kv_heads.1 * ratio));
        }
    }
}

fn local_top16(row: &[f32], start: usize, len: usize) -> (Vec<f32>, Vec<u32>) {
    let t = top16_reference(&row[start..start + len], 0);
    (
        t.iter().map(|c| c.0).collect(),
        t.iter().map(|c| c.1).collect(),
    )
}

/// 2026-10-09: The merged top-16 of per-shard top-16s equals the full row's top-16, with
/// ties (values on a coarse grid) broken by the lower id, for even and uneven vocab splits.
#[test]
fn merge_top16_equals_single_rank_top16() {
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    for &(vocab, world, align) in &[(128 * 40, 2, 128), (128 * 41, 2, 128), (1003, 3, 1)] {
        let rows = 5;
        let logits: Vec<Vec<f32>> = (0..rows)
            .map(|_| {
                (0..vocab)
                    .map(|_| (rng.next() % 64) as f32 * 0.25 - 4.0)
                    .collect()
            })
            .collect();
        let shards: Vec<(usize, usize)> = (0..world)
            .map(|r| shard_range(vocab, world, r, align).unwrap())
            .collect();
        let per: Vec<(Vec<f32>, Vec<u32>)> = shards
            .iter()
            .map(|&(s, l)| {
                let (mut v, mut i) = (Vec::new(), Vec::new());
                for row in &logits {
                    let (rv, ri) = local_top16(row, s, l);
                    v.extend(rv);
                    i.extend(ri);
                }
                (v, i)
            })
            .collect();
        let parts: Vec<RankTop<'_>> = per
            .iter()
            .zip(&shards)
            .map(|((v, i), &(s, _))| RankTop {
                vals: v,
                ids: i,
                vocab_start: s as u32,
            })
            .collect();
        let (mv, mi) = merge_top16_rows(rows, &parts).unwrap();
        for (r, row) in logits.iter().enumerate() {
            let want = top16_reference(row, 0);
            let got: Vec<(f32, u32)> = (0..TOP_K)
                .map(|k| (mv[r * TOP_K + k], mi[r * TOP_K + k]))
                .collect();
            assert_eq!(got, want, "vocab {vocab} world {world} row {r}");
        }
    }
}

#[test]
fn merge_rejects_short_parts() {
    let v = vec![0.0f32; 15];
    let i = vec![0u32; 15];
    let p = [RankTop {
        vals: &v,
        ids: &i,
        vocab_start: 0,
    }];
    assert!(merge_top16_rows(1, &p).is_err());
}

#[test]
fn header_round_trips_and_rejects_corruption() {
    let cap = 17;
    let h = TpHeader {
        flags: header::FLAG_RESYNC,
        last_token: 151_329,
        position: 4097,
        width: 16,
        ctx_len: 300,
        committed: 0,
        chunk: 4096,
        keep: 290,
        first_pos: 12,
        keep_last_pos: 4085,
        n_new: 10,
        positions: (4086..4096).collect(),
    };
    let w = h.encode_main(cap);
    assert_eq!(w.len(), header::HDR_FIXED_WORDS + cap);
    assert!(header::hdr_bytes(cap) >= w.len() * 4);
    let back =
        TpHeader::decode_main(&header::bytes_to_words(&header::words_to_bytes(&w)), cap).unwrap();
    assert_eq!(back, h);
    let mut bad = w.clone();
    bad[4] ^= 1;
    assert!(TpHeader::decode_main(&bad, cap).is_err(), "checksum");
    let mut bad = w.clone();
    bad[0] = 0;
    assert!(TpHeader::decode_main(&bad, cap).is_err(), "magic");
    let abort = TpHeader::decode_main(&TpHeader::abort().encode_main(cap), cap).unwrap();
    assert!(abort.is_abort());

    // 2026-10-09: More new rows than one transfer holds: the main header carries the first
    // `cap` positions, chunks carry the rest.
    let big = TpHeader {
        keep: 0,
        n_new: 40,
        ctx_len: 40,
        positions: (0..40).collect(),
        ..h.clone()
    };
    let rows = header::transfer_rows(40, cap);
    assert_eq!(rows, vec![17, 17, 6]);
    let main = TpHeader::decode_main(&big.encode_main(cap), cap).unwrap();
    assert_eq!(main.positions, (0..17).collect::<Vec<i32>>());
    let c = header::encode_chunk(&big.positions[34..40], cap);
    assert_eq!(
        header::decode_chunk(&c, 6, cap).unwrap(),
        (34..40).collect::<Vec<i32>>()
    );
    assert!(header::decode_chunk(&c, 5, cap).is_err());
    assert!(header::transfer_rows(0, cap).is_empty());
}

#[test]
fn leader_keep_counts_synced_rows() {
    let p: Vec<i32> = (0..10).map(|i| i * 2).collect();
    assert_eq!(leader_keep(&p, None), 0);
    assert_eq!(leader_keep(&p, Some(-1)), 0);
    assert_eq!(leader_keep(&p, Some(6)), 4);
    assert_eq!(leader_keep(&p, Some(7)), 4);
    assert_eq!(leader_keep(&p, Some(100)), 10);
    assert_eq!(
        leader_keep(&[0, 1, 1, 2], Some(5)),
        0,
        "not strictly increasing"
    );
}

#[test]
fn conform_plan_matches_rank0_rows() {
    let f: Vec<i32> = (0..100).collect();
    // 2026-10-09: No slide: keep the first 90 rows.
    assert_eq!(
        plan_conform(&f, 200, 90, 0, 89, 95).unwrap(),
        Conform { drop: 0, keep: 90 }
    );
    // 2026-10-09: Rank 0 slid: its row 0 is position 50.
    assert_eq!(
        plan_conform(&f, 200, 40, 50, 89, 45).unwrap(),
        Conform { drop: 50, keep: 40 }
    );
    assert_eq!(
        plan_conform(&f, 200, 0, 7, -1, 30).unwrap(),
        Conform { drop: 0, keep: 0 }
    );
    assert!(
        plan_conform(&f, 200, 40, 500, 89, 45).is_err(),
        "no such position"
    );
    assert!(
        plan_conform(&f, 200, 40, 50, 88, 45).is_err(),
        "last kept differs"
    );
    assert!(
        plan_conform(&f, 200, 60, 50, 109, 70).is_err(),
        "too few rows"
    );
    assert!(
        plan_conform(&f, 60, 40, 50, 89, 70).is_err(),
        "over capacity"
    );
}

#[test]
fn precompute_chunks_split_by_window() {
    assert_eq!(precompute_chunks(0, 10, 4), vec![(0, 4), (4, 4), (8, 2)]);
    assert_eq!(precompute_chunks(10, 10, 4), vec![]);
    assert_eq!(precompute_chunks(12, 10, 4), vec![]);
    assert_eq!(precompute_chunks(3, 5, 4096), vec![(3, 2)]);
    assert_eq!(precompute_chunks(0, 5, 0), vec![]);
}
