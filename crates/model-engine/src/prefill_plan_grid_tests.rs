// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Tests for the absolute prefill grid (`METRALE_PREFIX_GRID_RESTORE`, race #69): the
//! grid chunk planner, the restore-point search, and cold/warm replays through the radix prefix
//! cache with a 4-slot snapshot pool.
//!
//! Owner: model-engine prefill (SSM prefix cache).
//! Invariants: none beyond the types.

use super::*;
use metrale_cache::radix_tree::RadixTree;
use metrale_telemetry::prefix_cache::PrefixCache;

const BS: usize = 16;
const G: usize = 8192;

/// 2026-10-03: The chunks the scheduler runs for a `total`-token prompt from `start` under the
/// grid, with chunk 0 proposing `first` tokens (the idle `max_batch_tokens`) and later chunks
/// `cont`, through `plan_chunk_len_grid` as the scheduler calls it.
fn grid_chunks(total: usize, start: usize, first: usize, cont: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut off = start;
    while off < total {
        let cap = if off == 0 { first } else { cont };
        let len = plan_chunk_len_grid(off, total, (total - off).min(cap), Some(BS), None, Some(G));
        assert!(len >= 1, "total={total} off={off}: empty chunk");
        out.push((off, off + len));
        off += len;
    }
    out
}

#[test]
fn grid_chunks_end_on_absolute_multiples_whatever_the_budget() {
    for total in [1, 255, 8191, 8192, 8193, 8208, 16384, 16385, 32251, 32768, 32772, 40000] {
        // 2026-10-03: Idle 8193 / busy 8192 / max_batch_size 32 (8224) chunk-0 budgets and a
        // small continuation budget all give the same chunks.
        let a = grid_chunks(total, 0, 8193, 8192);
        for (first, cont) in [(8192, 8192), (8224, 8192), (8193, 4096), (100, 100)] {
            assert_eq!(grid_chunks(total, 0, first, cont), a, "total={total} first={first}");
        }
        for (i, &(s, e)) in a.iter().enumerate() {
            assert_eq!(s % G, 0, "total={total}: chunk {i} starts off the grid at {s}");
            if e < total {
                assert_eq!(e - s, G, "total={total}: non-last chunk {i} is {} rows", e - s);
            } else {
                assert!(e - s <= G && e == total);
            }
        }
    }
    // 2026-10-03: The case the budget plan gets wrong: an idle 8193-token prompt is one 8193-row
    // chunk there, two chunks on the grid.
    assert_eq!(
        plan_chunk_len(0, 8193, 8193, Some(BS), None),
        8193,
        "budget plan: one chunk"
    );
    assert_eq!(grid_chunks(8193, 0, 8193, 8192), vec![(0, 8192), (8192, 8193)]);
}

#[test]
fn a_warm_prefill_from_a_grid_point_runs_the_cold_chunks_from_there() {
    for total in [8193, 16385, 32251, 32768, 32772, 40000, 65537] {
        let cold = grid_chunks(total, 0, 8193, 8192);
        let mut k = G;
        while k < total {
            // 2026-10-03: The scheduler always starts at 0; the warm request's chunks before `k`
            // are fully cached (early return) and the rest compute. Same plan either way.
            let warm_compute: Vec<_> = grid_chunks(total, 0, 8192, 8192)
                .into_iter()
                .filter(|&(s, _)| s >= k)
                .collect();
            let cold_from_k: Vec<_> = cold.iter().copied().filter(|&(s, _)| s >= k).collect();
            assert_eq!(warm_compute, cold_from_k, "total={total} k={k}");
            assert_eq!(warm_compute.first().map(|c| c.0), Some(k));
            k += G;
        }
    }
}

#[test]
fn grid_off_is_the_budget_plan() {
    for (off, total, proposed) in [(0, 32772, 8193), (8192, 32772, 8192), (24576, 32772, 8196)] {
        for split in [None, tail_split_point(total, BS)] {
            assert_eq!(
                plan_chunk_len_grid(off, total, proposed, Some(BS), split, None),
                plan_chunk_len(off, total, proposed, Some(BS), split)
            );
        }
    }
    assert_eq!(plan_chunk_len_grid(0, 100, 0, Some(BS), None, Some(G)), 0);
}

#[test]
fn grid_points_caps_and_validity() {
    assert_eq!(grid_chunk_len(0, 5, G), 5);
    assert_eq!(grid_chunk_len(8192, 8192, G), 0);
    assert_eq!(grid_chunk_len(100, 20000, G), 8092);
    assert_eq!(grid_chunk_len(0, 5, 0), 0);
    // 2026-10-03: Restore cap: at most the match, strictly below the prompt end.
    assert_eq!(grid_restore_cap(32240, 32251, G), 24576);
    assert_eq!(grid_restore_cap(32768, 32768, G), 24576);
    assert_eq!(grid_restore_cap(32768, 32769, G), 32768);
    assert_eq!(grid_restore_cap(8191, 40000, G), 0);
    assert_eq!(grid_restore_cap(0, 0, G), 0);
    assert_eq!(grid_insert_len(32251, G), 24576);
    assert_eq!(grid_insert_len(32768, G), 24576);
    assert_eq!(grid_insert_len(8192, G), 0);
    assert_eq!(grid_insert_len(8193, G), 8192);
    // 2026-10-03: Snapshots only at non-last grid chunk ends.
    assert!(is_grid_checkpoint_end(8192, 8193, G, BS));
    assert!(!is_grid_checkpoint_end(8192, 8192, G, BS));
    assert!(!is_grid_checkpoint_end(8176, 20000, G, BS), "a block end off the grid");
    assert!(!is_grid_checkpoint_end(8193, 20000, G, BS), "no flooring to 8192");
    assert!(!is_grid_checkpoint_end(0, 20000, G, BS));
    assert!(!is_grid_checkpoint_end(8192, 20000, G, 0));
    assert!(grid_valid(8192, 16, 8193));
    assert!(!grid_valid(8192, 16, 8191), "larger than the arena");
    assert!(!grid_valid(8200, 16, 9000), "not a block multiple");
    assert!(!grid_valid(0, 16, 9000));
    assert!(!grid_valid(8192, 0, 9000));
    assert!(grid_restore_requested(Some("1")));
    assert!(!grid_restore_requested(Some("0")) && !grid_restore_requested(None));
    assert_eq!(grid_tokens_from(None), DEFAULT_GRID_TOKENS);
    assert_eq!(grid_tokens_from(Some("4096")), 4096);
    assert_eq!(grid_tokens_from(Some("0")), DEFAULT_GRID_TOKENS);
    assert_eq!(grid_tokens_from(Some("x")), DEFAULT_GRID_TOKENS);
}

#[test]
fn grid_step_cases() {
    // 2026-10-03: Exact grid match with its snapshot: accept.
    assert_eq!(grid_step(24576, 24576, 32251, G), GridStep::Accept);
    // 2026-10-03: Snapshot at the cap but the match runs past it: truncate to the cap.
    assert_eq!(grid_step(32240, 24576, 32251, G), GridStep::Probe(24576));
    // 2026-10-03: Off-grid snapshot past the cap (a tail at 31984): probe the cap.
    assert_eq!(grid_step(32240, 31984, 32251, G), GridStep::Probe(24576));
    // 2026-10-03: Deepest snapshot below the cap: probe its grid floor.
    assert_eq!(grid_step(32240, 16384, 32251, G), GridStep::Probe(16384));
    assert_eq!(grid_step(24576, 20000, 32251, G), GridStep::Probe(16384));
    // 2026-10-03: Full-prompt match on a grid-multiple prompt: the cap stays below the end.
    assert_eq!(grid_step(32768, 32768, 32768, G), GridStep::Probe(24576));
    // 2026-10-03: Nothing restorable.
    assert_eq!(grid_step(8000, 8000, 32251, G), GridStep::Miss);
    assert_eq!(grid_step(24576, 0, 32251, G), GridStep::Miss);
    assert_eq!(grid_step(24576, 4000, 32251, G), GridStep::Miss);
    assert_eq!(grid_step(0, 0, 32251, G), GridStep::Miss);
}

/// 2026-10-03: A radix tree plus a snapshot pool of `slots` ids, driven as the engine drives it
/// under the grid: `cold`/`warm` prefill a prompt on the grid, saving a snapshot at every non-last
/// chunk end (evicting through `evict_snapshot_lru` when the pool is full) and inserting the
/// prompt up to `grid_insert_len`.
struct Rig {
    tree: RadixTree,
    free: Vec<usize>,
    next_block: u32,
}

impl Rig {
    fn new(slots: usize) -> Self {
        Self {
            tree: RadixTree::new(),
            free: (0..slots).rev().collect(),
            next_block: 0,
        }
    }

    fn snap_depth(m: &metrale_telemetry::prefix_cache::PrefixMatch) -> usize {
        if m.ssm_snapshot.is_some() {
            m.ssm_snapshot_tokens
        } else if m.ssm_snapshot_tier_key.is_some() {
            m.ssm_snapshot_tier_tokens
        } else {
            0
        }
    }

    /// 2026-10-03: The engine's restore-point search (`grid_restore.rs`) on this tree. Returns
    /// the restore point and the match the sequence holds.
    fn restore_point(&self, tokens: &[u32], session: u64) -> (usize, usize) {
        let total = tokens.len();
        let first = self.tree.lookup(tokens, BS, session, 0);
        let mut held = first.matched_tokens;
        let r = grid_search(
            first.matched_tokens,
            Self::snap_depth(&first),
            total,
            G,
            |n| {
                if held > 0 {
                    self.tree.release_matched(tokens, BS, held, 0);
                }
                if n == 0 {
                    held = 0;
                    return (0, 0);
                }
                let m = self.tree.lookup(&tokens[..n], BS, session, 0);
                held = m.matched_tokens;
                (m.matched_tokens, Self::snap_depth(&m))
            },
        );
        (r, held)
    }

    fn save(&mut self, tokens: &[u32], end: usize, session: u64) {
        let slot = match self.free.pop() {
            Some(s) => s,
            None => self.tree.evict_snapshot_lru().expect("a resident victim"),
        };
        let blocks: Vec<u32> = (0..end / BS).map(|i| i as u32 + 1_000_000).collect();
        self.tree.insert_intermediate_snapshot(
            &tokens[..end],
            &blocks,
            &[],
            BS,
            slot,
            session,
            end,
            0,
        );
    }

    /// 2026-10-03: Prefill `tokens` (warm when the search accepts), as the engine does under the
    /// grid; returns the restore point.
    fn prefill(&mut self, tokens: &[u32], session: u64) -> usize {
        let total = tokens.len();
        let (r, held) = self.restore_point(tokens, session);
        assert_eq!(held, r, "the sequence holds exactly the restored prefix");
        for (s, e) in grid_chunks(total, 0, 8193, 8192) {
            if e <= r {
                continue;
            }
            assert!(s >= r, "a computed chunk starts at or after the restore point");
            if is_grid_checkpoint_end(e, total, G, BS) {
                self.save(tokens, e, session);
            }
        }
        let ins = grid_insert_len(total, G);
        if ins > 0 {
            let blocks: Vec<u32> = (0..ins / BS)
                .map(|_| {
                    self.next_block += 1;
                    self.next_block
                })
                .collect();
            self.tree.insert(&tokens[..ins], &blocks, &[], BS, r, 0);
        }
        // 2026-10-03: Retire: release exactly the inserted prefix (`free_sequence`).
        if ins > 0 {
            self.tree.release_matched(tokens, BS, ins, 0);
        }
        r
    }
}

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| i.wrapping_mul(2654435761).wrapping_add(seed))
        .collect()
}

#[test]
fn multi_turn_restores_at_the_deepest_grid_point() {
    let mut rig = Rig::new(4);
    let turn1 = prompt(7, 32251);
    assert_eq!(rig.prefill(&turn1, 1), 0, "cold");
    // 2026-10-03: Exact repeat: restores at 24576 (the last grid point below the end) and runs the
    // cold last chunk [24576, 32251).
    assert_eq!(rig.prefill(&turn1, 1), 24576);
    // 2026-10-03: Turn 2 extends turn 1: restore 24576, compute [24576, 32768) and
    // [32768, 33251), saving 32768.
    let mut turn2 = turn1.clone();
    turn2.extend(prompt(9, 1000));
    assert_eq!(rig.prefill(&turn2, 1), 24576);
    // 2026-10-03: Turn 3 restores at the snapshot turn 2 saved.
    let mut turn3 = turn2.clone();
    turn3.extend(prompt(11, 500));
    assert_eq!(rig.prefill(&turn3, 1), 32768);
}

#[test]
fn off_grid_snapshots_are_skipped_and_released() {
    let mut rig = Rig::new(4);
    let p = prompt(3, 32251);
    rig.prefill(&p, 0);
    // 2026-10-03: A stray off-grid snapshot (an old tail at 31984) deeper than every grid point,
    // with the tree extended past it.
    let blocks: Vec<u32> = (0..32240 / BS).map(|i| 5_000_000 + i as u32).collect();
    rig.tree.insert(&p[..32240], &blocks, &[], BS, 0, 0);
    rig.tree
        .insert_intermediate_snapshot(&p[..31984], &[], &[], BS, 3, 0, 31984, 0);
    let (r, held) = rig.restore_point(&p, 0);
    assert_eq!((r, held), (24576, 24576));
    rig.tree.release_matched(&p, BS, held, 0);
}

#[test]
fn a_missing_deep_snapshot_falls_back_to_a_shallower_grid_point() {
    let rig = Rig::new(4);
    let p = prompt(5, 40000);
    // 2026-10-03: The tree holds 32768 tokens but the pool kept only the 8192 and 16384
    // boundaries (the deeper ones were evicted).
    let blocks: Vec<u32> = (0..32768 / BS).map(|i| 7_000_000 + i as u32).collect();
    rig.tree.insert(&p[..32768], &blocks, &[], BS, 32768, 0);
    rig.tree
        .insert_intermediate_snapshot(&p[..8192], &[], &[], BS, 0, 0, 8192, 0);
    rig.tree
        .insert_intermediate_snapshot(&p[..16384], &[], &[], BS, 1, 0, 16384, 0);
    let (r, held) = rig.restore_point(&p, 0);
    assert_eq!((r, held), (16384, 16384));
    rig.tree.release_matched(&p, BS, held, 0);
}

#[test]
fn four_slots_keep_the_deepest_boundary_of_the_latest_prompt() {
    let mut rig = Rig::new(4);
    // 2026-10-03: Session 1 fills all four slots (boundaries 8192..32768).
    let a = prompt(21, 40000);
    rig.prefill(&a, 1);
    // 2026-10-03: Session 2, a different prompt with three boundaries, then its next turn.
    let b = prompt(22, 33000);
    rig.prefill(&b, 2);
    let mut b2 = b.clone();
    b2.extend(prompt(23, 700));
    assert_eq!(rig.prefill(&b2, 2), 32768, "session 2's deepest boundary survived");
    // 2026-10-03: The stalest session lost slots first; session 1 keeps at most one.
    let ma = rig.tree.lookup(&a, BS, 1, 0);
    assert!(ma.ssm_snapshot_tokens <= 32768);
    rig.tree.release_matched(&a, BS, ma.matched_tokens, 0);
}

#[test]
fn prompts_up_to_one_grid_get_no_snapshot_and_no_restore() {
    let mut rig = Rig::new(4);
    for len in [100, 8191, 8192] {
        let p = prompt(len as u32, len);
        assert_eq!(rig.prefill(&p, 0), 0);
        assert_eq!(rig.prefill(&p, 0), 0, "len {len}: nothing restorable");
    }
}
