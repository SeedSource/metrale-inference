// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for [`super`] (`ssm_reserve`): the verify-pool slot count
//! and tiers, the pool reserve bytes, the f16-sized pool, the replay mode and
//! the Marconi snapshot-slot gate.
//!
//! Owner: model-layers (SSM reserve).
//! Invariants: none beyond the types.
use super::*;

/// 2026-09-25: Qwen3.6-27B geometry, from the config.json of
/// centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf, with `--num-drafts 3`.
///
/// Per-seq SSM blob: 48 GDN layers × (h 48·128·128·4 B + conv
/// (16·128·2 + 48·128)·4·4 B) = 48 × 3,309,568 = 158,859,264 B (151.5 MiB).
/// H is 95% of the blob, conv the other 5%.
const H_BLOB: usize = 48 * (48 * 128 * 128 * 4);
const CONV_BLOB: usize = 48 * ((16 * 128 * 2 + 48 * 128) * 4 * 4);
const BLOB: usize = H_BLOB + CONV_BLOB;
const ND: usize = 3;

/// 2026-09-25: Pool bytes with every slot covered at full K and no K-1 shrink:
/// `max_batch × blob × (1 + (nd+1) + 1)` with spec on, `max_batch × blob` off.
fn legacy_pool_bytes(bs: usize, spec_on: bool) -> usize {
    let mult = if spec_on { 1 + (ND + 1) + 1 } else { 1 };
    bs * BLOB * mult
}

/// 2026-09-25: The default ladder shape (`4:3,8:3,16:1,32:1`, from
/// `speculative/ladder.rs`), spelled out so these tests do not read the
/// process env.
fn default_ladder(n: usize) -> usize {
    if n <= 8 { 3 } else { 1 }
}

fn tiered_pool_bytes(bs: usize, spec_on: bool) -> usize {
    ssm_pool_reserve_bytes(
        bs,
        H_BLOB,
        CONV_BLOB,
        spec_on,
        ND,
        mtp_state_slots_with(bs, 32, false),
        false,
        false,
        SsmRollbackMode::Snapshot,
    )
}

/// 2026-09-25: `tiered_pool_bytes` with the f16-sized pool (`h_f16_pool = true`).
fn tiered_pool_bytes_f16(bs: usize, spec_on: bool) -> usize {
    ssm_pool_reserve_bytes(
        bs,
        H_BLOB,
        CONV_BLOB,
        spec_on,
        ND,
        mtp_state_slots_with(bs, 32, false),
        false,
        true,
        SsmRollbackMode::Snapshot,
    )
}

#[test]
fn cap_identity_at_or_below_32_every_config() {
    // 2026-09-25: At bs<=32 the slot count is bs for every dispatch cap
    // (including the 4 of METRALE_NO_MTP_K_LADDER), because the floor is
    // VERIFY_WY_TABLE_SEQS = 32.
    for bs in 1..=32 {
        for cap in [1, 4, 16, 32, 64] {
            assert_eq!(
                mtp_state_slots_with(bs, cap, false),
                bs,
                "bs={bs} cap={cap}"
            );
        }
    }
}

#[test]
fn tier_capacity_default_ladder_shape() {
    // 2026-09-25: Slots 0..8 keep 3 drafts (K=4); slots 8.. get the deepest
    // draft count the ladder gives at the widths that reach them, 1 (K=2) under
    // the default ladder.
    for slot in 0..8 {
        assert_eq!(verify_slot_drafts_with(slot, 32, 3, default_ladder), 3);
    }
    for slot in 8..32 {
        assert_eq!(verify_slot_drafts_with(slot, 32, 3, default_ladder), 1);
    }
    // 2026-09-25: Beyond the dispatch cap: the last rung's depth, never zero.
    assert_eq!(verify_slot_drafts_with(40, 32, 3, default_ladder), 1);
    // 2026-09-25: `--num-drafts` is the ceiling.
    for slot in 0..32 {
        assert_eq!(verify_slot_drafts_with(slot, 32, 1, default_ladder), 1);
        assert_eq!(verify_slot_drafts_with(slot, 32, 0, default_ladder), 0);
    }
    // 2026-09-25: A deeper ladder (e.g. "4:3,8:3,16:2,24:2,32:2") deepens the
    // high slots with it: capacity follows the ladder policy.
    let deep = |n: usize| if n <= 8 { 3 } else { 2 };
    assert_eq!(verify_slot_drafts_with(8, 32, 3, deep), 2);
    assert_eq!(verify_slot_drafts_with(31, 32, 3, deep), 2);
}

#[test]
fn k_minus_1_shrink_and_kill_switch_shape() {
    // 2026-09-25: Every sizing is `legacy_pool_bytes` minus one h blob per
    // verify slot (the K-1 shrink), minus the tier savings where tiers apply.
    // At bs<=8 all slots are full K, so the difference is the shrink alone.
    for bs in 1..=8 {
        assert_eq!(
            legacy_pool_bytes(bs, true) - tiered_pool_bytes(bs, true),
            bs * H_BLOB,
            "bs={bs}: exactly one dead h blob per slot"
        );
        // 2026-09-25: Spec off: base only, equal to `legacy_pool_bytes`.
        assert_eq!(tiered_pool_bytes(bs, false), legacy_pool_bytes(bs, false));
    }
    // 2026-09-25: `uniform_verify` (every slot at the full `num_drafts`): the
    // same one-h-blob shrink and no tiers, at every bs.
    for bs in 1..=32 {
        for spec_on in [false, true] {
            let expect = legacy_pool_bytes(bs, spec_on) - if spec_on { bs * H_BLOB } else { 0 };
            assert_eq!(
                ssm_pool_reserve_bytes(
                    bs,
                    H_BLOB,
                    CONV_BLOB,
                    spec_on,
                    ND,
                    bs,
                    true,
                    false,
                    SsmRollbackMode::Snapshot,
                ),
                expect,
                "bs={bs} spec={spec_on}: uniform = legacy minus the dead h blob/slot"
            );
        }
    }
}

#[test]
fn cap_bites_above_32_and_kill_switch_restores() {
    // 2026-09-25: Dispatch cap 32: a 64-slot pool covers 32 verify slots.
    assert_eq!(mtp_state_slots_with(64, 32, false), 32);
    // 2026-09-25: A dispatch cap of 48 (METRALE_MTP_MAX_SEQS=48) widens the
    // pools with it.
    assert_eq!(mtp_state_slots_with(64, 48, false), 48);
    // 2026-09-25: The cap of 4 under METRALE_NO_MTP_K_LADDER still floors at 32.
    assert_eq!(mtp_state_slots_with(64, 4, false), 32);
    // 2026-09-25: `full_width` (METRALE_MTP_POOL_FULL_WIDTH or EP v2).
    assert_eq!(mtp_state_slots_with(64, 32, true), 64);
}

#[test]
fn tiered_totals_pinned() {
    // 2026-09-25: Verify-pool bytes per covered slot: the slot's h
    // intermediates (its draft capacity) + `num_drafts + 1` conv intermediates
    // + one checkpoint blob.
    //
    // bs=16: slots 8..16 hold 2 fewer h blobs each (tier) and every slot one
    // fewer (K-1).
    assert_eq!(legacy_pool_bytes(16, true), 15_250_489_344);
    assert_eq!(tiered_pool_bytes(16, true), 10_418_651_136);
    assert_eq!(
        legacy_pool_bytes(16, true) - tiered_pool_bytes(16, true),
        (16 + 16) * H_BLOB
    );
    // 2026-09-25: bs=32: the tier saves 48 h blobs (slots 8..32 × 2, 6.75 GiB)
    // and the K-1 shrink 32 (4.5 GiB): 80 h blobs, 11.25 GiB. Conv stays
    // uniform (see `verify_slot_h_intermediates`).
    assert_eq!(
        legacy_pool_bytes(32, true) - tiered_pool_bytes(32, true),
        80 * H_BLOB
    );
    assert_eq!(tiered_pool_bytes(32, true) - 32 * BLOB, 13_337_886_720);
    // 2026-09-25: Spec off: base only, at any bs.
    assert_eq!(tiered_pool_bytes(64, false), 64 * BLOB);
}

#[test]
fn bs64_ledger_before_after_and_fit() {
    let full_width = legacy_pool_bytes(64, true);
    assert_eq!(full_width, 61_001_957_376);
    // 2026-09-25: The slot-count cap alone: 32 covered slots at full K, no K-1
    // shrink.
    let slot_capped = 64 * BLOB + 32 * (ND + 2) * BLOB;
    assert_eq!(slot_capped, 35_584_475_136);
    assert_eq!(full_width - slot_capped, 25_417_482_240);
    // 2026-09-25: `uniform_verify` at 32 covered slots: the slot-count cap minus
    // one h blob per slot.
    assert_eq!(
        ssm_pool_reserve_bytes(
            64,
            H_BLOB,
            CONV_BLOB,
            true,
            ND,
            32,
            true,
            false,
            SsmRollbackMode::Snapshot,
        ),
        slot_capped - 32 * H_BLOB
    );
    let tiered = tiered_pool_bytes(64, true);
    assert_eq!(tiered, 23_504_879_616);
    assert_eq!(slot_capped - tiered, 80 * H_BLOB);

    // 2026-09-25: The other reserve terms as preflight computes them: a 32-slot
    // snapshot region (the decode ring is 0 under spec), the GDN two-phase
    // prefill scratch, and the spec CUDA headroom.
    let snapshot = 32 * BLOB;
    // 2026-09-25: 4096 tokens × (conv_dim 10240×2 + nv 48×2×4 + value_dim
    // 6144×2 + 6144×2) B/tok.
    let gdn = 4096 * (10240 * 2 + 48 * 2 * 4 + 6144 * 2 + 6144 * 2);
    assert_eq!(gdn, 186_122_240);
    let headroom = 4usize * 1024 * 1024 * 1024;

    let full_reserve = full_width + snapshot + gdn + headroom;
    assert_eq!(full_reserve, 70_566_543_360);
    assert_eq!(full_reserve / (1024 * 1024), 67_297);

    let capped_reserve = slot_capped + snapshot + gdn + headroom;
    assert_eq!(capped_reserve, 45_149_061_120);
    let tiered_reserve = tiered + snapshot + gdn + headroom;
    assert_eq!(tiered_reserve, 33_069_465_600);

    // 2026-09-25: Fit inputs: an 85.2 GiB budget (util 0.70) and 38.5 GiB
    // consumed before KV (weights, arena, twins).
    let budget = (85.2f64 * 1024.0 * 1024.0 * 1024.0) as usize;
    let pre_kv = (38.5f64 * 1024.0 * 1024.0 * 1024.0) as usize;
    // 2026-09-25: KV floor: 64 sequences × (128 + 1024) tokens × 64 KiB/token
    // (16 attention layers × 2 × 4 kv_heads × 256 head_dim × 2 B bf16).
    let kv_floor = 64 * (128 + 1024) * (16 * 2 * 4 * 256 * 2);
    assert_eq!(kv_floor, 4_831_838_208);

    // 2026-09-25: The full-width reserve is over budget before any KV.
    assert!(pre_kv + full_reserve > budget);
    // 2026-09-25: The slot-capped reserve leaves the KV floor plus at least
    // 150 MiB.
    let kv_left = budget - pre_kv - capped_reserve;
    assert!(
        kv_left >= kv_floor,
        "bs=64 KV budget {kv_left} must cover the decode_short peak {kv_floor}"
    );
    assert!(kv_left - kv_floor >= 150 * 1024 * 1024);
    // 2026-09-25: The tiered reserve leaves 11.25 GiB more.
    assert!(budget - pre_kv - tiered_reserve - kv_floor >= 150 * 1024 * 1024);
}

#[test]
fn h_stored_bytes_is_identity_off_and_half_on() {
    // 2026-09-25: Flag off: identity at any width.
    for b in [4usize, 128, H_BLOB, CONV_BLOB] {
        assert_eq!(ssm_h_stored_bytes(b, false), b);
    }
    // 2026-09-25: f16-sized pool: half. h blobs are FP32-element sized, so /2 is
    // exact.
    assert_eq!(ssm_h_stored_bytes(H_BLOB, true), H_BLOB / 2);
    assert_eq!(ssm_h_stored_bytes(4, true), 2);
}

#[test]
fn h_stored_bytes_rejects_non_fp32_width() {
    for f16_pool in [false, true] {
        let result = std::panic::catch_unwind(|| ssm_h_stored_bytes(3, f16_pool));
        assert!(result.is_err(), "f16_pool={f16_pool}");
    }
}

#[test]
fn f16_pool_sizing_pinned_and_flag_off_untouched() {
    // 2026-09-25: With `h_f16_pool = false` the totals equal the ones pinned
    // above; a transposed argument would show here.
    assert_eq!(tiered_pool_bytes(32, true) - 32 * BLOB, 13_337_886_720);
    assert_eq!(tiered_pool_bytes(128, true), 33_671_872_512);

    // 2026-09-25: With the f16-sized pool every h term (base, tiered
    // intermediates, checkpoints) halves and conv is unchanged. bs=128, K=4:
    //   base 128 × (H/2 + CONV)               = 10_670_309_376
    //   slots 0..8:  8 × (3·H/2 + 4·CONV + (H/2 + CONV)) = 2_730_491_904
    //   slots 8..32: 24 × (1·H/2 + 4·CONV + (H/2 + CONV)) = 4_567_597_056
    let f16 = tiered_pool_bytes_f16(128, true);
    assert_eq!(f16, 17_968_398_336);
    // 2026-09-25: 14.62 GiB less than the FP32-sized pool.
    assert_eq!(tiered_pool_bytes(128, true) - f16, 15_703_474_176);
    // 2026-09-25: Spec off: only the base h blobs narrow.
    assert_eq!(
        tiered_pool_bytes_f16(64, false),
        64 * (H_BLOB / 2 + CONV_BLOB)
    );
}

/// 2026-09-25: One layer's FP32 h blob, what the prefill staging is sized by.
/// `H_BLOB` is the per-seq total across the 48 GDN layers.
const H_LAYER: usize = H_BLOB / 48;

#[test]
fn prefill_staging_costs_one_fp32_layer_blob_per_slot() {
    // 2026-09-25: Flag off: zero at every batch size.
    for bs in [1usize, 32, 64, 128] {
        assert_eq!(ssm_h_prefill_stage_bytes(bs, H_LAYER, false), 0);
    }
    // 2026-09-25: f16-sized pool: one FP32 layer blob per slot, not per slot per
    // layer.
    assert_eq!(H_LAYER, 3_145_728);
    assert_eq!(ssm_h_prefill_stage_bytes(128, H_LAYER, true), 402_653_184);
    assert_eq!(ssm_h_prefill_stage_bytes(32, H_LAYER, true), 100_663_296);
}

/// 2026-09-25: `tiered_pool_bytes` under `SsmRollbackMode::Replay`.
fn replay_pool_bytes(bs: usize, spec_on: bool) -> usize {
    ssm_pool_reserve_bytes(
        bs,
        H_BLOB,
        CONV_BLOB,
        spec_on,
        ND,
        mtp_state_slots_with(bs, 32, false),
        false,
        false,
        SsmRollbackMode::Replay,
    )
}

#[test]
fn replay_mode_keeps_one_checkpoint_blob_per_slot() {
    // 2026-09-25: Replay drops every per-token intermediate (h and conv): the
    // verify term is one checkpoint blob per covered slot.
    assert_eq!(replay_pool_bytes(128, true), (128 + 32) * BLOB);
    assert_eq!(replay_pool_bytes(8, true), (8 + 8) * BLOB);
    // 2026-09-25: Spec off: equal to snapshot mode.
    for bs in [1, 32, 128] {
        assert_eq!(replay_pool_bytes(bs, false), tiered_pool_bytes(bs, false));
    }
}

#[test]
fn replay_ring_bytes_pinned_27b() {
    // 2026-09-25: One cached row per token per layer: qkvz (16384 BF16 = 32768 B)
    // + gate/beta (48 x 2 FP32 = 384 B) = 33152 B.
    let row = ssm_replay_row_bytes(16384, 48);
    assert_eq!(row, 33_152);
    // 2026-09-25: A K=4 window caches K-1 = 3 rows per covered slot across 48
    // layers.
    let ring = ssm_replay_ring_bytes(48, row, 4, 32);
    assert_eq!(ring, 152_764_416);
    // 2026-09-25: Zero slots, or K<=1: zero bytes.
    assert_eq!(ssm_replay_ring_bytes(48, row, 4, 0), 0);
    assert_eq!(ssm_replay_ring_bytes(48, row, 1, 32), 0);
}

#[test]
fn rollback_mode_parses_and_rejects() {
    use std::str::FromStr;
    assert_eq!(
        SsmRollbackMode::from_str("snapshot").unwrap(),
        SsmRollbackMode::Snapshot
    );
    assert_eq!(
        SsmRollbackMode::from_str("replay").unwrap(),
        SsmRollbackMode::Replay
    );
    // 2026-09-25: Anything else is an error; CLI validation uses this parse.
    assert!(SsmRollbackMode::from_str("Replay").is_err());
    assert!(SsmRollbackMode::from_str("").is_err());
}

// 2026-09-25: The decode-rollback ring's depth decision, its publication cell
// and the auto-fit are tested in `ssm_reserve/decode_ring_tests.rs`.

// 2026-09-25: The Marconi snapshot-slot gate. The region's only reader is a
// prefix-cache lookup, so with the cache inactive no slot is kept. Preflight
// and `TransformerModel::new` both decide through `marconi_snapshot_slots`.
mod marconi_gate {
    use crate::ssm_reserve::{marconi_snapshot_slots_with, prefix_caching_active};

    #[test]
    fn active_cache_keeps_every_requested_slot() {
        let d = marconi_snapshot_slots_with(16, true, false);
        assert_eq!(d.slots, 16);
        assert!(d.skip_reason.is_none());
    }

    #[test]
    fn inactive_cache_drops_the_region_and_says_why() {
        let d = marconi_snapshot_slots_with(16, false, false);
        assert_eq!(d.slots, 0);
        assert!(d.skip_reason.is_some(), "implicit skip must be logged once");
    }

    #[test]
    fn explicit_zero_is_not_an_implicit_skip() {
        // 2026-09-25: `--ssm-cache-slots 0` is honoured without a skip_reason, so
        // it is not logged as the gate's skip.
        for caching in [true, false] {
            let d = marconi_snapshot_slots_with(0, caching, false);
            assert_eq!(d.slots, 0);
            assert!(d.skip_reason.is_none());
        }
    }

    #[test]
    fn full_reserve_kill_switch_restores_the_old_behaviour() {
        let d = marconi_snapshot_slots_with(16, false, true);
        assert_eq!(d.slots, 16, "METRALE_SSM_MARCONI_FULL must over-reserve");
        assert!(
            d.skip_reason.is_none(),
            "an explicit override is not a skip"
        );
    }

    #[test]
    fn preflight_and_allocator_agree_on_every_combination() {
        // 2026-09-25: Preflight decides from `prefix_caching_active(flag,
        // kv_safe)`; the allocator from the constructed cache's `is_active()`.
        // `build_prefix_cache` installs a real cache exactly when that predicate
        // holds, so both sides call the core with the same input.
        for &requested in &[0usize, 1, 16, 256] {
            for &flag in &[true, false] {
                for &kv_safe in &[true, false] {
                    for &full in &[true, false] {
                        let effective = prefix_caching_active(flag, kv_safe);
                        let pre = marconi_snapshot_slots_with(requested, effective, full);
                        let alloc = marconi_snapshot_slots_with(requested, effective, full);
                        assert_eq!(pre.slots, alloc.slots);
                    }
                }
            }
        }
    }

    #[test]
    fn v4_compressed_downgrade_is_treated_as_inactive() {
        // 2026-09-25: `--enable-prefix-caching` with an unsafe KV-only config
        // installs `NoPrefixCaching`, so the flag alone does not keep the region.
        assert!(!prefix_caching_active(true, false));
        let d = marconi_snapshot_slots_with(16, prefix_caching_active(true, false), false);
        assert_eq!(d.slots, 0);
    }
}

/// 2026-10-10: `METRALE_SSM_NO_CHECKPOINT`: the reserve drops exactly one blob per
/// covered verify slot, in both rollback modes, for both lever values (the `_with`
/// core takes the bool, so no process env is set).
#[test]
fn no_checkpoint_drops_one_blob_per_verify_slot() {
    for rollback in [SsmRollbackMode::Snapshot, SsmRollbackMode::Replay] {
        for (bs, nd) in [(1usize, 1usize), (16, 2), (16, 3), (64, 3), (128, 3)] {
            let slots = mtp_state_slots_with(bs, 32, false);
            let run = |no_ckpt| {
                ssm_pool_reserve_bytes_with(
                    bs, H_BLOB, CONV_BLOB, true, nd, slots, false, false, rollback, no_ckpt,
                )
            };
            assert_eq!(
                run(false) - run(true),
                slots * BLOB,
                "{rollback:?} bs={bs} nd={nd}"
            );
            // 2026-10-10: Lever off is the legacy entry point's value.
            assert_eq!(
                run(false),
                ssm_pool_reserve_bytes(
                    bs, H_BLOB, CONV_BLOB, true, nd, slots, false, false, rollback
                )
            );
        }
    }
    // Spec off: nothing to drop.
    assert_eq!(
        ssm_pool_reserve_bytes_with(
            16,
            H_BLOB,
            CONV_BLOB,
            false,
            2,
            16,
            false,
            false,
            SsmRollbackMode::Snapshot,
            true
        ),
        16 * BLOB
    );
}

/// 2026-10-10: Pinned numbers at max_batch 16, num_drafts 2 (27B geometry above): the
/// 16-slot checkpoint share is the saving, and replay-mode verify cost goes to 0.
#[test]
fn no_checkpoint_pinned_bs16_nd2() {
    let on =
        |m| ssm_pool_reserve_bytes_with(16, H_BLOB, CONV_BLOB, true, 2, 16, false, false, m, true);
    let off =
        |m| ssm_pool_reserve_bytes_with(16, H_BLOB, CONV_BLOB, true, 2, 16, false, false, m, false);
    assert_eq!(
        off(SsmRollbackMode::Snapshot) - on(SsmRollbackMode::Snapshot),
        16 * BLOB
    );
    assert_eq!(on(SsmRollbackMode::Replay), 16 * BLOB);
    assert_eq!(off(SsmRollbackMode::Replay), 32 * BLOB);
    assert_eq!(ssm_checkpoint_bytes(16, BLOB, false), 16 * BLOB);
    assert_eq!(ssm_checkpoint_bytes(16, BLOB, true), 0);
}
