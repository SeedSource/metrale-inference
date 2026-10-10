// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Equivalence tests for the batched SSM verify-state copies.
//!
//! The property: the batched form lands the same bytes in the same slots as the
//! per-layer `copy_d2d_async` loop, for per-sequence commit and rollback plans. The
//! plans mirror the dispatchers' plans but do not instantiate `TransformerModel`. The
//! equivalence tests run both executors over the same plan on two identically built
//! mock devices and compare the live state byte for byte, because a launch count alone
//! would pass a batched form that copied the wrong rows.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use super::super::ssm_pool::SsmStatePool;
use super::*;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::ssm_reserve::SsmRollbackMode;

/// 2026-09-25: The Qwen3-Next-80B layer pattern (every fourth of 48 layers is
/// full attention, leaving 36 SSM layers) with tiny head dims, so a test
/// allocates kilobytes. The geometry under test, the pool's layer, slot and
/// intermediate striding, does not depend on the blob width.
fn tiny_config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.linear_num_key_heads = 2;
    c.linear_key_head_dim = 4;
    c.linear_num_value_heads = 2;
    c.linear_value_head_dim = 4;
    c.linear_conv_kernel_dim = 4;
    c
}

const MAX_SLOTS: usize = 4;
const NUM_INTERMEDIATES: usize = 4;
const NUM_DRAFTS: usize = 3;

fn pool(gpu: &MockGpuBackend) -> SsmStatePool {
    SsmStatePool::new(
        &tiny_config(),
        MAX_SLOTS,
        true,
        NUM_INTERMEDIATES,
        NUM_DRAFTS,
        false,
        SsmRollbackMode::Snapshot,
        gpu,
    )
    .unwrap()
}

/// 2026-09-25: A distinct fill per layer, slot and region, so restoring from
/// the wrong layer, slot or token index shows in the bytes.
fn seed(gpu: &MockGpuBackend, p: &SsmStatePool) {
    for l in 0..p.num_ssm_layers {
        for s in 0..=p.mtp_slots {
            let tag = |k: usize| ((l * 131 + s * 17 + k * 7) % 251 + 1) as u8;
            gpu.copy_h2d(&vec![tag(0); p.h_stored_bytes], p.h_state(l, s))
                .unwrap();
            gpu.copy_h2d(&vec![tag(1); p.conv_bytes], p.conv_state(l, s))
                .unwrap();
            gpu.copy_h2d(
                &vec![tag(2); p.h_stored_bytes],
                p.h_checkpoint(l, s).unwrap(),
            )
            .unwrap();
            gpu.copy_h2d(
                &vec![tag(3); p.conv_bytes],
                p.conv_checkpoint(l, s).unwrap(),
            )
            .unwrap();
            for t in 0..p.h_inter_count(s) {
                gpu.copy_h2d(
                    &vec![tag(4 + t); p.h_stored_bytes],
                    p.h_intermediate(l, s, t),
                )
                .unwrap();
            }
            for t in 0..p.num_intermediates {
                gpu.copy_h2d(
                    &vec![tag(9 + t); p.conv_bytes],
                    p.conv_intermediate(l, s, t),
                )
                .unwrap();
            }
        }
    }
}

/// 2026-09-25: The live state of every slot of every layer: what a rollback
/// changes, and what the two executors must agree on.
fn live_state(gpu: &MockGpuBackend, p: &SsmStatePool) -> Vec<u8> {
    let mut out = Vec::new();
    for l in 0..p.num_ssm_layers {
        for s in 0..=p.mtp_slots {
            let mut h = vec![0u8; p.h_stored_bytes];
            gpu.copy_d2h(p.h_state(l, s), &mut h).unwrap();
            out.extend_from_slice(&h);
            let mut c = vec![0u8; p.conv_bytes];
            gpu.copy_d2h(p.conv_state(l, s), &mut c).unwrap();
            out.extend_from_slice(&c);
        }
    }
    out
}

/// 2026-09-25: The h/conv plans `commit_accepted_prefix_dispatch` builds for one
/// sequence: `h_intermediate[na-1] → h_state`, same for conv, one row per SSM
/// layer in ascending layer order.
fn commit_plans(
    p: &SsmStatePool,
    slot: usize,
    num_accepted: usize,
) -> (Vec<StateCopy>, Vec<StateCopy>) {
    let cfg = tiny_config();
    let conv_bytes = (cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
        + cfg.linear_num_value_heads * cfg.linear_value_head_dim)
        * cfg.linear_conv_kernel_dim
        * 4;
    let mut h = Vec::new();
    let mut c = Vec::new();
    let mut ssm_layer_idx = 0usize;
    for i in 0..cfg.num_hidden_layers {
        if cfg.layer_type(i) != LayerType::LinearAttention {
            continue;
        }
        let idx = num_accepted - 1;
        h.push(StateCopy {
            src: p.h_intermediate(ssm_layer_idx, slot, idx),
            dst: p.h_state(ssm_layer_idx, slot),
            bytes: p.h_stored_bytes,
        });
        c.push(StateCopy {
            src: p.conv_intermediate(ssm_layer_idx, slot, idx),
            dst: p.conv_state(ssm_layer_idx, slot),
            bytes: conv_bytes,
        });
        ssm_layer_idx += 1;
    }
    (h, c)
}

/// 2026-09-25: The full-reject arm of `rollback_ssm_states_dispatch`: restore
/// the pre-verify checkpoint (`num_accepted == 0`).
fn reject_plans(p: &SsmStatePool, slot: usize) -> (Vec<StateCopy>, Vec<StateCopy>) {
    let (mut h, mut c) = commit_plans(p, slot, 1);
    for (l, row) in h.iter_mut().enumerate() {
        row.src = p.h_checkpoint(l, slot).unwrap();
    }
    for (l, row) in c.iter_mut().enumerate() {
        row.src = p.conv_checkpoint(l, slot).unwrap();
    }
    (h, c)
}

/// 2026-09-25: Run one plan pair both ways on two identically built devices and
/// assert the live state is byte-identical.
fn assert_equivalent(make_plans: impl Fn(&SsmStatePool) -> (Vec<StateCopy>, Vec<StateCopy>)) {
    let looped = MockGpuBackend::new();
    let lp = pool(&looped);
    seed(&looped, &lp);
    let (lh, lc) = make_plans(&lp);
    run_state_copies_with(&looped, &lh, false, 0).unwrap();
    run_state_copies_with(&looped, &lc, false, 0).unwrap();

    let batched = MockGpuBackend::new();
    let bp = pool(&batched);
    seed(&batched, &bp);
    let (bh, bc) = make_plans(&bp);
    run_state_copies_with(&batched, &bh, true, 0).unwrap();
    run_state_copies_with(&batched, &bc, true, 0).unwrap();
    // 2026-09-25: Both plans must have collapsed, or the comparison is only
    // "the loop equals the loop".
    assert_eq!(
        (batched.d2d_count(), batched.d2d_2d_count()),
        (0, 2),
        "batched run fell back to the per-layer loop"
    );

    // 2026-09-25: The two devices are built the same way, so the plans must be
    // address-identical; otherwise the comparison below compares different
    // geometries.
    assert_eq!(lh, bh, "h plan differs between runs");
    assert_eq!(lc, bc, "conv plan differs between runs");

    let want = live_state(&looped, &lp);
    let got = live_state(&batched, &bp);
    assert_eq!(want.len(), got.len());
    assert!(
        want == got,
        "batched rollback wrote different bytes than the per-layer loop"
    );
    // 2026-09-25: The plan must also have changed the live state: two no-op
    // executors would agree byte for byte.
    let untouched = MockGpuBackend::new();
    let up = pool(&untouched);
    seed(&untouched, &up);
    assert!(
        want != live_state(&untouched, &up),
        "live state unchanged — the plan moved nothing"
    );
}

#[test]
fn commit_is_byte_identical_at_every_accepted_depth() {
    let probe = MockGpuBackend::new();
    let p = pool(&probe);
    let k = p.h_inter_count(0);
    assert!(k >= 2, "fixture must offer more than one rollback depth");
    // 2026-09-25: num_accepted == 1 is the shallowest commit (index 0); k is
    // the deepest (index k-1). Both ends and the interior are covered.
    for na in 1..=k {
        assert_equivalent(|pool| commit_plans(pool, 0, na));
    }
}

#[test]
fn full_reject_restore_is_byte_identical() {
    // 2026-09-25: `num_accepted == 0`: the source is the checkpoint pool, not
    // the intermediates.
    assert_equivalent(|p| reject_plans(p, 0));
}

#[test]
fn every_slot_rolls_back_to_its_own_state() {
    // 2026-09-25: Slot addressing is where a batched form would write a
    // neighbour's state: slot s's H rows start at `h_inter_offsets[s]`.
    let probe = MockGpuBackend::new();
    let p = pool(&probe);
    for s in 0..MAX_SLOTS {
        let k = p.h_inter_count(s);
        assert!(k >= 1, "slot {s} has no intermediates");
        assert_equivalent(|pool| commit_plans(pool, s, k));
    }
}

#[test]
fn single_ssm_layer_still_copies_the_right_bytes() {
    // 2026-09-25: A one-row plan cannot collapse (no pitch to derive) and
    // falls through to a single `copy_d2d_async`.
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu);
    seed(&gpu, &p);
    let (h, _) = commit_plans(&p, 0, 1);
    let one = [h[0]];
    assert!(copy_plan_as_strided_run(&one).is_none());
    run_state_copies_with(&gpu, &one, true, 0).unwrap();
    assert_eq!(gpu.d2d_count(), 1, "single row = one plain d2d");
    assert_eq!(gpu.d2d_2d_count(), 0);
    let mut got = vec![0u8; p.h_stored_bytes];
    gpu.copy_d2h(p.h_state(0, 0), &mut got).unwrap();
    let mut want = vec![0u8; p.h_stored_bytes];
    gpu.copy_d2h(p.h_intermediate(0, 0, 0), &mut want).unwrap();
    assert_eq!(got, want);
}

#[test]
fn batching_collapses_the_layer_loop_to_one_launch() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu);
    seed(&gpu, &p);
    let (h, c) = commit_plans(&p, 0, 1);
    let layers = p.num_ssm_layers;
    assert_eq!(h.len(), layers);

    const STREAM: u64 = 0x5a17;
    run_ssm_state_copies(&gpu, &h, &c, STREAM).unwrap();
    assert_eq!(
        (gpu.d2d_count(), gpu.d2d_2d_count()),
        (0, 2),
        "{layers} h + {layers} conv copies must become 2 pitched launches"
    );
    assert_eq!(gpu.d2d_2d_async_streams(), [STREAM, STREAM]);

    // 2026-09-25: Unbatched, the same plan runs as 2 × layers plain copies.
    let off = MockGpuBackend::new();
    let op = pool(&off);
    let (oh, oc) = commit_plans(&op, 0, 1);
    run_state_copies_with(&off, &oh, false, STREAM).unwrap();
    run_state_copies_with(&off, &oc, false, STREAM).unwrap();
    assert_eq!((off.d2d_count(), off.d2d_2d_count()), (2 * layers, 0));
    assert_eq!(off.d2d_async_streams(), vec![STREAM; 2 * layers]);
}

#[test]
fn kill_switch_default_is_batched() {
    // 2026-09-25: The variable is a presence check latched in a `OnceLock`;
    // under `cargo test`, with it unset, the production reader must report
    // batched.
    assert!(
        batched_ssm_copy_enabled(),
        "METRALE_NO_BATCHED_SSM_ROLLBACK is set in this environment"
    );
}

// 2026-09-25: copy_plan_as_strided_run, the collapse predicate itself.

fn row(src: u64, dst: u64, bytes: usize) -> StateCopy {
    StateCopy {
        src: DevicePtr(src),
        dst: DevicePtr(dst),
        bytes,
    }
}

#[test]
fn a_plan_that_a_2d_copy_cannot_reproduce_is_refused() {
    let base: Vec<StateCopy> = (0..4)
        .map(|l| row(1000 + l * 64, 9000 + l * 64, 16))
        .collect();
    assert!(copy_plan_as_strided_run(&base).is_some(), "control");

    // 2026-09-25: Ragged width: no single `width_bytes`.
    let mut ragged = base.clone();
    ragged[2].bytes = 8;
    assert!(copy_plan_as_strided_run(&ragged).is_none());

    // 2026-09-25: A relocated layer (as with per-layer allocations) breaks the src stride.
    let mut hole = base.clone();
    hole[2].src = DevicePtr(500_000);
    assert!(copy_plan_as_strided_run(&hole).is_none());

    // 2026-09-25: The same for the dst stride, independently.
    let mut dhole = base.clone();
    dhole[3].dst = DevicePtr(500_000);
    assert!(copy_plan_as_strided_run(&dhole).is_none());

    // 2026-09-25: Descending family: a negative pitch has no 2-D form.
    let down: Vec<StateCopy> = (0..4)
        .map(|l| row(1000 - l * 64, 9000 - l * 64, 16))
        .collect();
    assert!(copy_plan_as_strided_run(&down).is_none());

    // 2026-09-25: Pitch below width: rows would overlap, so the 2-D form is not the plan.
    let tight: Vec<StateCopy> = (0..4)
        .map(|l| row(1000 + l * 8, 9000 + l * 8, 16))
        .collect();
    assert!(copy_plan_as_strided_run(&tight).is_none());

    // 2026-09-25: Zero-width and empty plans are not 2-D shapes either.
    assert!(copy_plan_as_strided_run(&[]).is_none());
    assert!(copy_plan_as_strided_run(&[row(1, 2, 0), row(3, 4, 0)]).is_none());
}

#[test]
fn the_collapsed_run_reproduces_the_plan_row_for_row() {
    // 2026-09-25: Expanding the 2-D run by its definition (`row r = width bytes
    // at base + r*pitch`) must give back exactly the plan that produced it.
    let plan: Vec<StateCopy> = (0..7)
        .map(|l| row(4096 + l * 256, 1_000_000 + l * 128, 96))
        .collect();
    let r = copy_plan_as_strided_run(&plan).unwrap();
    let expanded: Vec<StateCopy> = (0..r.height)
        .map(|i| StateCopy {
            src: r.src.offset(i * r.src_pitch),
            dst: r.dst.offset(i * r.dst_pitch),
            bytes: r.width_bytes,
        })
        .collect();
    assert_eq!(expanded, plan);
}

/// 2026-10-10: `METRALE_SSM_NO_CHECKPOINT` pool: no checkpoint pools, the accessors say
/// so, and `reset_slot` / `copy_slot` still work on the remaining families.
#[test]
fn no_checkpoint_pool_skips_checkpoint_in_reset_and_copy() {
    let gpu = MockGpuBackend::new();
    let on = SsmStatePool::new_with_checkpoint_mode(
        &tiny_config(),
        MAX_SLOTS,
        true,
        NUM_INTERMEDIATES,
        NUM_DRAFTS,
        false,
        SsmRollbackMode::Snapshot,
        true,
        &gpu,
    )
    .unwrap();
    assert!(!on.has_checkpoints());
    assert!(on.h_checkpoint(0, 0).is_none() && on.conv_checkpoint(0, 0).is_none());
    on.reset_slot(0, &gpu).unwrap();
    on.copy_slot(0, 1, &gpu, 0).unwrap();
    let off = pool(&gpu);
    assert!(off.has_checkpoints());
    assert!(off.h_checkpoint(0, 0).is_some());
    off.reset_slot(0, &gpu).unwrap();
    off.copy_slot(0, 1, &gpu, 0).unwrap();
}
