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
