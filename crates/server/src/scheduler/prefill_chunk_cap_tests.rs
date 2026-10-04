// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Tests for `prefill_chunk_cap` (`METRALE_PREFILL_CHUNK_WHILE_DECODING`).

use super::*;

const BS: Option<usize> = Some(16);

#[test]
fn resolve_off_values() {
    for raw in [
        None,
        Some(""),
        Some("  "),
        Some("0"),
        Some("abc"),
        Some("-5"),
    ] {
        assert_eq!(resolve(raw), None, "{raw:?}");
    }
}

#[test]
fn resolve_rounds_to_granule() {
    assert_eq!(resolve(Some("2048")), Some(2048));
    assert_eq!(resolve(Some(" 4096 ")), Some(4096));
    assert_eq!(resolve(Some("2100")), Some(2048));
    assert_eq!(resolve(Some("1")), Some(GRANULE));
    assert_eq!(resolve(Some("255")), Some(GRANULE));
}

#[test]
fn aligned_cap_follows_the_block_size() {
    assert_eq!(aligned_cap(2048, Some(16)), 2048);
    assert_eq!(aligned_cap(2048, None), 2048);
    assert_eq!(aligned_cap(2048, Some(0)), 2048);
    // 2026-10-03: lcm(256, 48) = 768: 2048 -> 1536; lcm(256, 512) = 512.
    assert_eq!(aligned_cap(2048, Some(48)), 1536);
    assert_eq!(aligned_cap(256, Some(512)), 512);
}

#[test]
fn active_cap_needs_a_decoder() {
    assert_eq!(active_cap(Some(2048), 0, BS), None);
    assert_eq!(active_cap(Some(2048), 1, BS), Some(2048));
    assert_eq!(active_cap(Some(2048), 3, BS), Some(2048));
    assert_eq!(active_cap(None, 3, BS), None);
}

/// 2026-10-03: The scheduler's call: `proposed = min(remaining, max_prefill_tokens)`.
fn plan(offset: usize, total: usize, split: Option<usize>, cap: Option<usize>) -> (usize, usize) {
    plan_capped(
        offset,
        total,
        (total - offset).min(8192),
        BS,
        split,
        None,
        cap,
    )
}

#[test]
fn lever_off_or_no_decoder_is_the_old_plan() {
    for total in [1usize, 15, 16, 17, 255, 4096, 8193, 8200, 32772, 40000] {
        let split = metrale_model_engine::prefill_plan::tail_split_point_min(total, 16, 256);
        for split in [None, split] {
            let mut off = 0;
            while off < total {
                let old = metrale_model_engine::prefill_plan::plan_chunk_len_grid(
                    off,
                    total,
                    (total - off).min(8192),
                    BS,
                    split,
                    None,
                );
                for cap in [None, active_cap(Some(2048), 0, BS)] {
                    assert_eq!(plan(off, total, split, cap), (old, old), "{total} {off}");
                }
                off += old;
            }
        }
    }
}

/// 2026-10-03: Walk a prompt with the cap on: every chunk is at most the cap, every non-last
/// chunk ends on the block grid (and on the granule, absent a split point), the chunks tile the
/// prompt, and the split point is still a chunk end.
fn walk(total: usize, split: Option<usize>, cap: usize) -> Vec<usize> {
    let mut ends = Vec::new();
    let mut off = 0;
    while off < total {
        let (len, uncapped) = plan(off, total, split, Some(cap));
        assert!(
            len > 0 && len <= uncapped && len <= cap,
            "{total} {off} {len}"
        );
        off += len;
        if off < total {
            assert_eq!(off % 16, 0, "non-last chunk end {off} off the block grid");
        }
        ends.push(off);
    }
    assert_eq!(off, total);
    ends
}

#[test]
fn capped_chunks_tile_on_the_grid() {
    let ends = walk(32772, None, 2048);
    assert_eq!(ends.len(), 17);
    assert!(ends[..16].iter().all(|e| e % 2048 == 0));
    let split = metrale_model_engine::prefill_plan::tail_split_point_min(32772, 16, 256);
    assert_eq!(split, Some(32512));
    let ends = walk(32772, split, 4096);
    assert!(ends.contains(&32512), "{ends:?}");
    // 2026-10-03: 4096 is a multiple of the uncapped 8192 chunk ends: the capped ends are a
    // refinement of the uncapped ones (each uncapped end is also a capped end).
    let mut off = 0;
    while off < 32772 {
        let (_, u) = plan(off, 32772, split, None);
        off += u;
        assert!(
            ends.contains(&off),
            "uncapped end {off} missing from {ends:?}"
        );
    }
}

#[test]
fn remaining_below_cap_is_unchanged() {
    // 2026-10-03: a 1500-token last chunk under a 2048 cap, and a whole 1000-token prompt.
    assert_eq!(plan(30720, 32220, None, Some(2048)), (1500, 1500));
    assert_eq!(plan(0, 1000, None, Some(2048)), (1000, 1000));
}

#[test]
fn chunk0_from_the_capped_budget_matches_plan_capped() {
    // 2026-10-03: `phase_start_prefills` plans chunk 0 from `min(budget, cap)` through
    // `start_chunked_prefill`; that must equal `plan_capped` of the uncapped budget.
    for total in [100usize, 2048, 2049, 9000, 32772] {
        let split = metrale_model_engine::prefill_plan::tail_split_point_min(total, 16, 256);
        for cap in [256usize, 1024, 2048, 4096] {
            let via_budget = metrale_model_engine::prefill_plan::plan_chunk_len_grid(
                0,
                total,
                total.min(8192usize.min(cap)),
                BS,
                split,
                None,
            );
            let (capped, _) = plan_capped(0, total, total.min(8192), BS, split, None, Some(cap));
            assert_eq!(via_budget, capped, "{total} {cap}");
        }
    }
}

#[test]
fn grid_ignores_the_cap() {
    let g = Some(8192);
    assert_eq!(
        plan_capped(0, 32772, 8192, BS, None, g, Some(2048)),
        (8192, 8192)
    );
}

#[test]
fn log_capped_once() {
    let mut logged = false;
    log_capped(&mut logged, 2048, 1);
    assert!(logged);
    log_capped(&mut logged, 2048, 1);
    assert!(logged);
}
