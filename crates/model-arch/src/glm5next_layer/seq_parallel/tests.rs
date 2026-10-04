// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Unit tests of the sequence-parallel row ownership (`SpPlan`) and the pass's call
//! check (`calls_tile`); moved out of `seq_parallel.rs` 2026-10-04 (500-line cap).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_layer::steps::staged::full_width_attn;
use crate::glm5next_layer::{ffn_windows, sub_chunks};

/// 2026-10-01: Rows of `spans`, expanded.
fn rows(spans: &[Span]) -> Vec<usize> {
    spans.iter().flat_map(|&(t, n)| t..t + n).collect()
}

/// 2026-10-04: The pre-2026-10-04 `spans` (halves of `sub_chunks(k, width)` relative to
/// the item's start), the reference for items of whole sub-chunks.
fn spans_v1(width: usize, rank: usize, (t, k): Span) -> Vec<Span> {
    sub_chunks(k, width)
        .into_iter()
        .map(|(s, n)| SpPlan::half(rank, (t + s, n)))
        .filter(|s| s.1 > 0)
        .collect()
}

/// 2026-10-04: `rank`'s rows of `[t, t + k)` by brute force: the row's sub-chunk in the
/// chunk, and its side of that sub-chunk's split.
fn owned_rows(plan: SpPlan, rank: usize, (t, k): Span) -> Vec<usize> {
    let subs = sub_chunks(plan.total, plan.width);
    (t..(t + k).min(plan.total))
        .filter(|&r| {
            let &(s, n) = subs.iter().find(|&&(s, n)| s <= r && r < s + n).unwrap();
            (r < s + n.div_ceil(2)) == (rank == 0)
        })
        .collect()
}

/// 2026-10-04: Every item `prefill_staged_run` / `prefill_staged_sp` issues for
/// `n` tokens at `rows` / `rows_ffn`: the sub-chunks, the full-width attention windows,
/// the FFN windows with and without the tail merge, and the whole chunk.
fn items(n: usize, rows: usize, ffn: usize) -> Vec<Span> {
    let subs = sub_chunks(n, rows);
    let mut v = subs.clone();
    v.extend(sub_chunks(n, ffn));
    v.extend(ffn_windows(&subs, ffn, false, |_| true));
    v.extend(ffn_windows(&subs, ffn, true, |_| true));
    v.push((0, n));
    v
}

/// 2026-10-04: The shapes the gate and the serve exercise: odd chunks, a one-row tail,
/// a chunk shorter than one sub-chunk, ROWS 256..2048 against ROWS_FFN 2048 / 8192, and
/// a window that is not a multiple of the sub-chunk (a `max_batch_tokens` cap).
const SHAPES: [(usize, usize, usize); 21] = [
    (5400, 256, 2048),
    (5401, 256, 2048),
    (8192, 256, 2048),
    (8193, 256, 2048),
    (8191, 256, 2048),
    (2049, 256, 2048),
    (255, 256, 2048),
    (1, 256, 2048),
    (8192, 512, 2048),
    (8193, 512, 2048),
    (5401, 1024, 2048),
    (8193, 2048, 2048),
    (4097, 2048, 4096),
    (8192, 256, 8192),
    (8193, 256, 8192),
    (16383, 256, 8192),
    (8100, 256, 8000),
    (8100, 256, 300),
    (1000, 7, 50),
    (3, 2, 2),
    (2, 1, 1),
];

#[test]
fn halves_tile_every_item_and_match_across_ranks() {
    for (n, w, ffn) in SHAPES {
        let subs = sub_chunks(n, w);
        let plan = SpPlan::new(&subs).unwrap();
        assert_eq!(plan.total, n);
        for (t, k) in items(n, w, ffn) {
            let (a, b) = (plan.spans(0, (t, k)), plan.spans(1, (t, k)));
            let mut all = [rows(&a), rows(&b)].concat();
            all.sort_unstable();
            assert_eq!(all, (t..t + k).collect::<Vec<_>>(), "n={n} item ({t}, {k})");
            assert!(a.iter().chain(&b).all(|s| s.1 > 0));
            assert!(a.windows(2).all(|p| p[0].0 + p[0].1 <= p[1].0), "row order");
            assert!(b.windows(2).all(|p| p[0].0 + p[0].1 <= p[1].0), "row order");
            for r in 0..2 {
                let got = rows(&plan.spans(r, (t, k)));
                assert_eq!(got, owned_rows(plan, r, (t, k)), "n={n} w={w} ({t}, {k}) r{r}");
            }
        }
        // 2026-10-01: Each rank's spans of the chunk are its spans of the sub-chunks.
        for r in 0..2 {
            let per_sub: Vec<Span> = subs.iter().flat_map(|&s| plan.spans(r, s)).collect();
            assert_eq!(plan.spans(r, (0, n)), per_sub);
        }
    }
}

/// 2026-10-04: One owner per row whatever the item: the owned rows of any tiling of the
/// chunk (sub-chunks, full-width windows, FFN windows) are the owned rows of the chunk.
#[test]
fn a_row_has_one_owner_in_every_tiling() {
    for (n, w, ffn) in SHAPES {
        let subs = sub_chunks(n, w);
        let plan = SpPlan::new(&subs).unwrap();
        let tilings = [
            subs.clone(),
            sub_chunks(n, ffn),
            ffn_windows(&subs, ffn, false, |_| true),
            ffn_windows(&subs, ffn, true, |_| true),
            ffn_windows(&subs, ffn, true, |k| k == w),
        ];
        for r in 0..2 {
            let whole = rows(&plan.spans(r, (0, n)));
            for tiling in &tilings {
                assert!(calls_tile(tiling, n), "n={n} w={w} ffn={ffn}");
                let got: Vec<usize> =
                    tiling.iter().flat_map(|&c| rows(&plan.spans(r, c))).collect();
                assert_eq!(got, whole, "n={n} w={w} ffn={ffn} rank {r}");
            }
        }
    }
}

/// 2026-10-04: For an item of whole sub-chunks (every item before the full-width lever,
/// and every full-width window when `rows_ffn` is a multiple of `rows`, which
/// `resolve_rows_ffn` guarantees) `spans` is the pre-2026-10-04 formula.
#[test]
fn whole_sub_chunk_items_keep_the_old_spans() {
    for (n, w, ffn) in SHAPES {
        let subs = sub_chunks(n, w);
        let plan = SpPlan::new(&subs).unwrap();
        let bounds: Vec<usize> = subs.iter().map(|s| s.0).chain([n]).collect();
        for (t, k) in items(n, w, ffn) {
            if !(bounds.contains(&t) && bounds.contains(&(t + k))) {
                continue;
            }
            for r in 0..2 {
                assert_eq!(plan.spans(r, (t, k)), spans_v1(w, r, (t, k)), "n={n} ({t},{k})");
            }
        }
    }
}

/// 2026-10-04: The production recipe (ROWS 256..2048, ROWS_FFN 2048, and the 8192-row
/// window): every full-width attention window starts on a sub-chunk boundary, the
/// full-width arm is taken only when the window is wider, and a one-row tail is owned by
/// rank 0 alone (rank 1 sends and receives nothing for it).
#[test]
fn production_widths_and_tails() {
    for rows in [256usize, 512, 1024, 2048] {
        for ffn in [2048usize, 8192] {
            let wide = full_width_attn(true, rows, ffn);
            assert_eq!(wide, ffn > rows, "rows {rows} ffn {ffn}");
            for n in [1usize, 2, 255, 256, 257, 2047, 2049, 5400, 8191, 8192, 8193] {
                let subs = sub_chunks(n, rows);
                let plan = SpPlan::new(&subs).unwrap();
                for (t, k) in sub_chunks(n, ffn) {
                    assert_eq!(t % rows, 0, "window start n={n} rows={rows} ffn={ffn}");
                    assert!(k > 0);
                }
                let tail = *subs.last().unwrap();
                if tail.1 == 1 {
                    assert_eq!(plan.spans(0, tail), vec![tail]);
                    assert!(plan.spans(1, tail).is_empty());
                }
            }
        }
    }
}

#[test]
fn halves_of_a_sub_chunk() {
    assert_eq!(SpPlan::half(0, (512, 256)), (512, 128));
    assert_eq!(SpPlan::half(1, (512, 256)), (640, 128));
    assert_eq!(SpPlan::half(0, (8192, 1)), (8192, 1));
    assert_eq!(SpPlan::half(1, (8192, 1)), (8193, 0));
    let plan = SpPlan::new(&sub_chunks(257, 256)).unwrap();
    assert_eq!(plan.spans(1, (0, 257)), vec![(128, 128)]);
    assert_eq!(SpPlan::new(&[]), None);
    // 2026-10-04: A window that splits the 164-row tail sub-chunk [7936, 8100) at 8000:
    // rank 0 owns [7936, 8018), rank 1 [8018, 8100).
    let plan = SpPlan::new(&sub_chunks(8100, 256)).unwrap();
    assert_eq!(plan.spans(0, (0, 8000)).last(), Some(&(7936, 64)));
    assert!(plan.spans(1, (0, 8000)).last().unwrap().0 < 7936);
    assert_eq!(plan.spans(0, (8000, 100)), vec![(8000, 18)]);
    assert_eq!(plan.spans(1, (8000, 100)), vec![(8018, 82)]);
}

#[test]
fn calls_must_tile_the_chunk() {
    assert!(calls_tile(&[(0, 4), (4, 3)], 7));
    assert!(!calls_tile(&[(0, 4), (4, 3)], 8), "short");
    assert!(!calls_tile(&[(0, 4), (5, 2)], 7), "gap");
    assert!(!calls_tile(&[(0, 4), (4, 0), (4, 3)], 7), "empty call");
    assert!(!calls_tile(&[(4, 3), (0, 4)], 7), "order");
}
