// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `SlotGuard`: release on drop, `take`, and migration.
//!
//! Owner: model-engine SSM state pool.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: A pool with no device memory: every pointer vector is empty and no
/// SSM layer exists. `claim_slot`, `release_slot` and the guard touch only
/// `free_slots`, so these tests need no GPU.
fn bare_pool(max_slots: usize) -> Arc<SsmStatePool> {
    Arc::new(SsmStatePool {
        owned_allocations: Vec::new(),
        h_state_pools: Vec::new(),
        conv_state_pools: Vec::new(),
        h_intermediate_pools: Vec::new(),
        conv_intermediate_pools: Vec::new(),
        h_checkpoint_pools: Vec::new(),
        conv_checkpoint_pools: Vec::new(),
        h_bytes: 0,
        h_stored_bytes: 0,
        h_prefill_stage_pool: None,
        conv_bytes: 0,
        max_slots,
        mtp_slots: 0,
        num_ssm_layers: 0,
        has_mtp: false,
        num_intermediates: 0,
        h_inter_counts: Vec::new(),
        h_inter_offsets: Vec::new(),
        rollback_mode: metrale_model_layers::ssm_reserve::SsmRollbackMode::Snapshot,
        replay_input_rings: Vec::new(),
        free_slots: Mutex::new((0..max_slots).rev().collect()),
    })
}

fn free_count(pool: &SsmStatePool) -> usize {
    pool.free_slots.lock().len()
}

#[test]
fn guard_releases_on_drop() {
    let pool = bare_pool(2);
    let claimed;
    {
        let g = pool.claim_guarded().unwrap();
        // 2026-09-25: The free list is `(0..max).rev()`, so `pop()` returns 0 first.
        claimed = g.idx().expect("guard owns a slot");
        assert_eq!(claimed, 0);
        assert_eq!(free_count(&pool), 1);
    }
    assert_eq!(
        free_count(&pool),
        2,
        "drop must return the slot exactly once"
    );
    assert!(pool.free_slots.lock().contains(&claimed));
}

#[test]
fn take_neutralizes_drop_no_double_release() {
    let pool = bare_pool(2);
    let mut g = pool.claim_guarded().unwrap();
    let idx = g.take().expect("guard owns a slot");
    pool.release_slot(idx);
    assert_eq!(free_count(&pool), 2);
    drop(g);
    assert_eq!(
        free_count(&pool),
        2,
        "take() must make Drop a no-op (no double-release)"
    );
}

#[test]
fn migration_releases_old_once_then_owns_new() {
    // 2026-09-25: Two claimed slots, so the migration target is owned by another
    // guard rather than sitting on the free list.
    let pool = bare_pool(2);
    let mut survivor = pool.claim_guarded().unwrap();
    let donor = pool.claim_guarded().unwrap();
    assert_eq!(free_count(&pool), 0);
    let donor_slot = donor.idx().unwrap();

    // 2026-09-25: The survivor releases its old slot and migrates onto the donor's.
    let old = survivor.take().unwrap();
    pool.release_slot(old);
    // 2026-09-25: The donor gives up its slot without releasing it.
    let mut donor = donor;
    let _ = donor.take();
    drop(donor);
    survivor.migrate(donor_slot);
    assert_eq!(survivor.idx(), Some(donor_slot));

    let final_idx = survivor.take().unwrap();
    pool.release_slot(final_idx);
    drop(survivor);

    let free = pool.free_slots.lock();
    let mut sorted = free.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1], "both slots free exactly once, no dupes");
}

/// 2026-10-10: A pool with no checkpoint pools (`METRALE_SSM_NO_CHECKPOINT`, or no MTP)
/// reports none and returns `None` from the accessors instead of indexing.
#[test]
fn bare_pool_has_no_checkpoints() {
    let pool = bare_pool(2);
    assert!(!pool.has_checkpoints());
    assert!(pool.h_checkpoint(0, 0).is_none());
    assert!(pool.conv_checkpoint(0, 0).is_none());
}

/// 2026-10-10: `METRALE_MTP_POOL_CAP` claim rule (`take_free_slot`): the lowest free
/// index first; LIFO without the cap.
#[test]
fn take_free_slot_lowest_first_or_lifo() {
    use crate::model::ssm_pool_slots::take_free_slot;
    let mut free = vec![5, 1, 9, 3];
    assert_eq!(take_free_slot(&mut free, true), Some(1));
    assert_eq!(take_free_slot(&mut free, true), Some(3));
    assert_eq!(take_free_slot(&mut free, false), Some(9));
    assert_eq!(take_free_slot(&mut free, true), Some(5));
    assert_eq!(take_free_slot(&mut free, true), None);
    assert_eq!(take_free_slot(&mut free, false), None);
}

/// 2026-10-10: EP v2 mirror invariant under either claim rule. The worker frees the
/// slot's prior occupant and allocates again on every head alloc, and re-allocates a
/// phantom on every head release (`ep_worker_step`, `EP_FREE_REALLOC`); its claim must
/// return the head's slot each time, or the worker bails. With the lowest-first rule a
/// batch never holds a slot at or above `n` while fewer than `n + 1` sequences are
/// claimed at that sequence's claim time.
#[test]
fn ep_v2_mirrored_claims_agree_under_both_rules() {
    use crate::model::ssm_pool_slots::take_free_slot;
    for lowest_first in [false, true] {
        let max = 16usize;
        let mut head: Vec<usize> = (0..max).rev().collect();
        let mut worker: Vec<usize> = (0..max).rev().collect();
        let mut worker_held = vec![false; max];
        let mut live: Vec<usize> = Vec::new();
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..20_000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let alloc = live.is_empty() || (live.len() < max && (x >> 33) % 2 == 0);
            if alloc {
                let p = take_free_slot(&mut head, lowest_first).unwrap();
                if lowest_first {
                    assert!(
                        p <= live.len(),
                        "lowest-first gave slot {p} with {} live",
                        live.len()
                    );
                }
                if worker_held[p] {
                    worker.push(p);
                }
                assert_eq!(take_free_slot(&mut worker, lowest_first), Some(p));
                worker_held[p] = true;
                live.push(p);
            } else {
                let i = ((x >> 17) as usize) % live.len();
                let p = live.swap_remove(i);
                head.push(p);
                worker.push(p);
                assert_eq!(take_free_slot(&mut worker, lowest_first), Some(p));
            }
        }
    }
}

/// 2026-10-10: The EP v2 mirror invariant with `METRALE_MTP_POOL_CAP` slot migration
/// (`migrate_to_covered_slot_dispatch` on the head, `ep_worker_migrate_slot` on the worker).
/// After every release the head moves each survivor on a slot at or above `cover` onto the
/// lowest free covered slot; the worker frees its phantom on the target, migrates, and
/// re-allocates a phantom on the source, which must be the slot its lowest-first claim
/// returns. Every later mirrored claim must still agree, and once at most `cover`
/// sequences are live after a release, all of them sit on covered slots.
#[test]
fn ep_v2_mirrored_claims_agree_with_migration() {
    use crate::model::ssm_pool_slots::take_free_slot;
    let (max, cover) = (16usize, 8usize);
    let mut head: Vec<usize> = (0..max).rev().collect();
    let mut worker: Vec<usize> = (0..max).rev().collect();
    let mut worker_held = vec![false; max];
    let mut live: Vec<usize> = Vec::new();
    let mut migrations = 0usize;
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    let remove = |v: &mut Vec<usize>, s: usize| {
        let i = v
            .iter()
            .position(|&f| f == s)
            .expect("target on the free list");
        v.swap_remove(i);
    };
    for _ in 0..50_000 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let alloc = live.is_empty() || (live.len() < max && (x >> 33) % 2 == 0);
        if alloc {
            let p = take_free_slot(&mut head, true).unwrap();
            if worker_held[p] {
                worker.push(p);
            }
            assert_eq!(take_free_slot(&mut worker, true), Some(p));
            worker_held[p] = true;
            live.push(p);
        } else {
            let i = ((x >> 17) as usize) % live.len();
            let p = live.swap_remove(i);
            head.push(p);
            worker.push(p);
            assert_eq!(take_free_slot(&mut worker, true), Some(p));
            for slot in live.iter_mut() {
                let from = *slot;
                if from < cover {
                    continue;
                }
                let Some(to) = head.iter().copied().filter(|&s| s < cover).min() else {
                    continue;
                };
                remove(&mut head, to);
                head.push(from);
                if worker_held[to] {
                    worker.push(to);
                }
                remove(&mut worker, to);
                worker.push(from);
                assert_eq!(
                    take_free_slot(&mut worker, true),
                    Some(from),
                    "worker phantom must re-take the migration source"
                );
                worker_held[to] = true;
                *slot = to;
                migrations += 1;
            }
            if live.len() <= cover {
                assert!(
                    live.iter().all(|&s| s < cover),
                    "uncovered survivor: {live:?}"
                );
            }
        }
    }
    assert!(
        migrations > 1000,
        "simulation exercised {migrations} migrations"
    );
}
