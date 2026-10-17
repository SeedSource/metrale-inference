// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17 (A200): `CtxSlide` and `DflashProposerState::make_room_for`, the slide every
//! append to the DFlash ctx accumulator runs first, on `MockGpuBackend` (whose copies error
//! when they run past an allocation, as the K=2 accept append did at a full window).
//!
//! Owner: model-arch (DFlash drafter).

use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::tp::shard::{leader_keep, plan_conform};
use super::{CtxSlide, DflashProposerState};

/// 2026-10-17 (A200): Bytes per row in these tests: one u32 row marker.
const SLOT: usize = 4;

/// 2026-10-17 (A200): A state with `len` of `max` rows filled; row `i` holds marker `i` and
/// position `100 + i`, and every row is committed.
fn state(gpu: &MockGpuBackend, max: usize, len: usize) -> DflashProposerState {
    let acc = gpu.alloc(max * SLOT).unwrap();
    let mut host = vec![0u8; max * SLOT];
    for i in 0..len {
        host[i * SLOT..(i + 1) * SLOT].copy_from_slice(&(i as u32).to_le_bytes());
    }
    gpu.copy_h2d(&host, acc).unwrap();
    DflashProposerState {
        block_table: Vec::new(),
        seq_len: 0,
        last_num_drafted: 0,
        prefill_done: true,
        ctx_hidden_acc: acc,
        ctx_len: len,
        last_num_accepted: 0,
        skip_next_decode_append: false,
        max_ctx_len: max,
        ctx_slot_bytes: SLOT,
        block_table_dev: None,
        ctx_count_drafter: 0,
        max_ctx_count_drafter: 0,
        ctx_committed: len,
        ctx_positions: (0..len as i32).map(|i| 100 + i).collect(),
        tp_synced_pos: None,
    }
}

fn markers(gpu: &MockGpuBackend, d: &DflashProposerState) -> Vec<u32> {
    let mut b = vec![0u8; d.ctx_len * SLOT];
    gpu.copy_d2h(d.ctx_hidden_acc, &mut b).unwrap();
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 2026-10-17 (A200): The append the K=2 accept did before the fix: two rows at `ctx_len`,
/// with no bound. `make_room` = the fix's call first.
fn append_two(
    gpu: &MockGpuBackend,
    d: &mut DflashProposerState,
    make_room: bool,
) -> anyhow::Result<()> {
    if make_room {
        d.make_room_for(gpu, 2, 0)?;
    }
    let src = gpu.alloc(2 * SLOT)?;
    gpu.copy_h2d(&[0xAA; 2 * SLOT], src)?;
    for r in 0..2 {
        gpu.copy_d2d_async(
            src.offset(r * SLOT),
            d.ctx_hidden_acc.offset(d.ctx_len * SLOT),
            SLOT,
            0,
        )?;
        let next = d.ctx_positions.last().map_or(0, |p| p + 1);
        d.ctx_positions.push(next);
        d.ctx_len += 1;
    }
    Ok(())
}

#[test]
fn plan_slides_only_when_rows_do_not_fit() {
    assert_eq!(CtxSlide::plan(2056, 2057, 1).unwrap(), None);
    assert_eq!(CtxSlide::plan(2055, 2057, 2).unwrap(), None);
    // 2026-10-17 (A200): fp4lbdf: a 2,155-token prompt filled the 2057-row window, then a K=2
    // accept appended 2.
    assert_eq!(
        CtxSlide::plan(2057, 2057, 2).unwrap(),
        Some(CtxSlide {
            drop: 1029,
            keep: 1028
        })
    );
    assert_eq!(
        CtxSlide::plan(2056, 2057, 2).unwrap(),
        Some(CtxSlide {
            drop: 1028,
            keep: 1028
        })
    );
    // 2026-10-17 (A200): A K=gamma append of 9 rows next to a nearly full window keeps
    // max/2, so drop < keep is possible: `pieces` must split the move.
    assert_eq!(
        CtxSlide::plan(2050, 2057, 9).unwrap(),
        Some(CtxSlide {
            drop: 1022,
            keep: 1028
        })
    );
    // 2026-10-17 (A200): n == max keeps nothing.
    assert_eq!(
        CtxSlide::plan(5, 10, 10).unwrap(),
        Some(CtxSlide { drop: 5, keep: 0 })
    );
    assert!(
        CtxSlide::plan(2058, 2057, 1).is_err(),
        "already past the window"
    );
    assert!(
        CtxSlide::plan(0, 10, 11).is_err(),
        "more rows than the window"
    );
}

#[test]
fn pieces_never_overlap_and_move_the_kept_rows() {
    for max in 1..=40usize {
        for len in 0..=max {
            for n in 0..=max {
                let Some(s) = CtxSlide::plan(len, max, n).unwrap() else {
                    assert!(len + n <= max);
                    continue;
                };
                assert!(
                    s.keep + n <= max && s.drop + s.keep == len,
                    "{max} {len} {n}: {s:?}"
                );
                let mut rows: Vec<usize> = (0..len).collect();
                for (src, dst, k) in s.pieces() {
                    assert!(k > 0 && src + k <= len);
                    assert!(
                        dst + k <= src || src + k <= dst,
                        "{max} {len} {n}: piece overlaps"
                    );
                    let moved: Vec<usize> = rows[src..src + k].to_vec();
                    rows[dst..dst + k].copy_from_slice(&moved);
                }
                assert_eq!(&rows[..s.keep], &(s.drop..len).collect::<Vec<_>>()[..]);
            }
        }
    }
}

#[test]
fn old_append_at_a_full_window_overruns_and_the_fix_slides() {
    let gpu = MockGpuBackend::new();
    // 2026-10-17 (A200): ctx_len = max_ctx_len: the old two-row append writes past the
    // accumulator (the mock refuses the copy; the GPU wrote into a neighbour).
    let mut old = state(&gpu, 10, 10);
    let err = append_two(&gpu, &mut old, false).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("overruns") || msg.contains("not allocated"),
        "the first row lands at the accumulator's end: {msg}"
    );

    let mut d = state(&gpu, 10, 10);
    append_two(&gpu, &mut d, true).unwrap();
    assert_eq!(d.ctx_len, 7);
    assert_eq!(d.ctx_positions.len(), d.ctx_len);
    assert_eq!(
        &markers(&gpu, &d)[..5],
        &[5, 6, 7, 8, 9],
        "the newest max/2 rows moved to the front"
    );
    assert_eq!(
        &d.ctx_positions[..5],
        &[105, 106, 107, 108, 109],
        "absolute positions kept"
    );
    assert_eq!(d.ctx_committed, 0, "moved rows' K/V is recomputed");
}

#[test]
fn make_room_at_each_boundary() {
    let gpu = MockGpuBackend::new();
    // 2026-10-17 (A200): ctx_len = max - 1 with one row: fits, nothing moves.
    let mut d = state(&gpu, 10, 9);
    assert_eq!(d.make_room_for(&gpu, 1, 0).unwrap(), None);
    assert_eq!((d.ctx_len, d.ctx_committed), (9, 9));
    // 2026-10-17 (A200): ctx_len = max - 1 with two rows: slides.
    let mut d = state(&gpu, 10, 9);
    assert_eq!(
        d.make_room_for(&gpu, 2, 0).unwrap(),
        Some(CtxSlide { drop: 4, keep: 5 })
    );
    assert_eq!(markers(&gpu, &d), vec![4, 5, 6, 7, 8]);
    // 2026-10-17 (A200): ctx_len = max with a K=gamma append (n = 4): keep min(5, 6) = 5.
    let mut d = state(&gpu, 10, 10);
    assert_eq!(
        d.make_room_for(&gpu, 4, 0).unwrap(),
        Some(CtxSlide { drop: 5, keep: 5 })
    );
    assert_eq!(markers(&gpu, &d), vec![5, 6, 7, 8, 9]);
    // 2026-10-17 (A200): drop < keep (window 20, 15 rows, 9 new): split pieces, same rows.
    let mut d = state(&gpu, 20, 15);
    assert_eq!(
        d.make_room_for(&gpu, 9, 0).unwrap(),
        Some(CtxSlide { drop: 5, keep: 10 })
    );
    assert_eq!(markers(&gpu, &d), (5..15).collect::<Vec<u32>>());
    assert_eq!(d.ctx_positions, (105..115).collect::<Vec<i32>>());
    // 2026-10-17 (A200): ctx_len past the window (the state the bug left) is refused.
    let mut d = state(&gpu, 10, 10);
    d.ctx_len = 12;
    assert!(d.make_room_for(&gpu, 1, 0).is_err());
}

#[test]
fn two_rank_follower_conforms_to_a_slid_leader() {
    // 2026-10-17 (A200): Rank 0 and rank 1 hold the same full window (positions 100..110,
    // rank 1 synced to the newest). Rank 0 slides for a K=2 accept and appends 2 rows; its
    // next header must apply on rank 1 (`plan_conform`, then the carried rows), with
    // ctx_len within the follower's window: before the fix rank 0 sent ctx_len 12.
    let gpu = MockGpuBackend::new();
    let mut lead = state(&gpu, 10, 10);
    let follower_positions = lead.ctx_positions.clone();
    let synced = lead.ctx_positions.last().copied();
    append_two(&gpu, &mut lead, true).unwrap();
    assert!(lead.ctx_len <= lead.max_ctx_len);

    let keep = leader_keep(&lead.ctx_positions, synced);
    assert_eq!(keep, 5, "rank 1 already holds the 5 kept rows");
    let committed = lead.ctx_committed.min(keep);
    let c = plan_conform(
        &follower_positions,
        10,
        keep,
        lead.ctx_positions[0],
        lead.ctx_positions[keep - 1],
        lead.ctx_len,
    )
    .unwrap();
    assert_eq!((c.drop, c.keep), (5, 5));
    assert!(
        c.drop == 0 || committed == 0,
        "follower requires committed 0 after a drop"
    );
    // 2026-10-17 (A200): the follower's rows after conform + install equal rank 0's.
    let mut f: Vec<i32> = follower_positions[c.drop..c.drop + c.keep].to_vec();
    f.extend_from_slice(&lead.ctx_positions[keep..]);
    assert_eq!(f, lead.ctx_positions);
    assert!(f.len() <= 10, "tp_install's room check holds");
}

/// 2026-10-17 (A201): a one-row source holding `marker`.
fn row(gpu: &MockGpuBackend, marker: u32) -> metrale_gpu_runtime::gpu::DevicePtr {
    let r = gpu.alloc(SLOT).unwrap();
    gpu.copy_h2d(&marker.to_le_bytes(), r).unwrap();
    r
}

#[test]
fn decode_append_at_full_window_slides_then_appends() {
    let gpu = MockGpuBackend::new();
    let mut d = state(&gpu, 10, 10);
    let s = d.decode_append(&gpu, row(&gpu, 0xAB), 999, 0).unwrap();
    assert_eq!(s, Some(CtxSlide { drop: 5, keep: 5 }));
    assert_eq!(d.ctx_len, 6);
    let m = markers(&gpu, &d);
    assert_eq!(&m[..5], &[5, 6, 7, 8, 9]);
    assert_eq!(m[5], 0xAB);
    assert_eq!(*d.ctx_positions.last().unwrap(), 999);
    assert_eq!(d.ctx_positions.len(), 6);
    assert_eq!(d.ctx_committed, 0);
}

#[test]
fn decode_append_below_full_window_does_not_slide() {
    let gpu = MockGpuBackend::new();
    let mut d = state(&gpu, 10, 4);
    let s = d.decode_append(&gpu, row(&gpu, 0xAB), 999, 0).unwrap();
    assert_eq!(s, None);
    assert_eq!(d.ctx_len, 5);
    assert_eq!(d.ctx_positions, vec![100, 101, 102, 103, 999]);
    assert_eq!(d.ctx_committed, 4);
    assert_eq!(markers(&gpu, &d)[4], 0xAB);
}

#[test]
fn decode_append_repeated_from_full_never_errors() {
    let gpu = MockGpuBackend::new();
    let mut d = state(&gpu, 10, 10);
    for i in 0..20 {
        d.decode_append(&gpu, row(&gpu, 1000 + i), 500 + i as i32, 0)
            .unwrap();
        assert!(d.ctx_len <= d.max_ctx_len);
        assert_eq!(d.ctx_positions.len(), d.ctx_len);
        assert_eq!(*markers(&gpu, &d).last().unwrap(), 1000 + i);
    }
}

#[test]
fn both_decode_append_sites_slide() {
    let propose = include_str!("propose.rs");
    let leader = include_str!("tp/leader.rs");
    for (name, src) in [("propose.rs", propose), ("tp/leader.rs", leader)] {
        assert!(src.contains("decode_append("), "{name} must call decode_append(");
        assert!(
            !src.contains("ctx_len < dstate.max_ctx_len"),
            "{name} still skips the append at a full window"
        );
    }
}
