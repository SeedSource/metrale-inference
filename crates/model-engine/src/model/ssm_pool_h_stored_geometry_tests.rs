// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Geometry tests for `SsmStatePool` on the mock GPU backend: per-family
//! contiguity, h storage width, replay-mode pools and the FP32 prefill staging arena.
//!
//! Owner: model-engine SSM state pool.
//! Invariants: none beyond the types.

use super::*;
use metrale_config::ModelConfig;
use metrale_core::scope::ModelResource;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-09-25: Claimable slots in the test pool; the pool allocates `SLOTS + 1`,
/// the extra one being the dummy slot.
const SLOTS: usize = 4;

fn pool(h_f16_pool: bool) -> SsmStatePool {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    SsmStatePool::new(
        &config,
        SLOTS,
        true,
        4,
        3,
        h_f16_pool,
        metrale_model_layers::ssm_reserve::SsmRollbackMode::Snapshot,
        &gpu,
    )
    .unwrap()
}

/// 2026-09-25: Every pool family is one allocation with a uniform per-layer stride.
/// `model::ssm_batched_copy` needs that stride to turn the per-layer copies into one
/// pitched 2-D copy per family. Without it the addresses stay correct and only the
/// per-layer copy loop comes back, so other tests would not notice.
#[test]
fn layer_pools_are_one_contiguous_block_per_family() {
    let p = pool(false);
    assert_eq!(
        p.owned_allocations.len(),
        6,
        "each family must own one bulk allocation"
    );
    let families: [(&str, &[DevicePtr], usize); 6] = [
        (
            "h_state",
            &p.h_state_pools,
            (p.max_slots + 1) * p.h_stored_bytes,
        ),
        (
            "conv_state",
            &p.conv_state_pools,
            (p.max_slots + 1) * p.conv_bytes,
        ),
        (
            "h_intermediate",
            &p.h_intermediate_pools,
            *p.h_inter_offsets.last().unwrap() * p.h_stored_bytes,
        ),
        (
            "conv_intermediate",
            &p.conv_intermediate_pools,
            (p.mtp_slots + 1) * p.num_intermediates * p.conv_bytes,
        ),
        (
            "h_checkpoint",
            &p.h_checkpoint_pools,
            (p.mtp_slots + 1) * p.h_stored_bytes,
        ),
        (
            "conv_checkpoint",
            &p.conv_checkpoint_pools,
            (p.mtp_slots + 1) * p.conv_bytes,
        ),
    ];
    for (name, pools, stride) in families {
        assert_eq!(
            pools.len(),
            p.num_ssm_layers,
            "{name}: one region per layer"
        );
        assert!(stride > 0, "{name}: degenerate stride");
        for (l, ptr) in pools.iter().enumerate() {
            assert_eq!(
                ptr.0,
                pools[0].0 + (l * stride) as u64,
                "{name}: layer {l} is not at base + {l}*{stride}"
            );
        }
    }
}

#[test]
fn rejected_bulk_allocation_falls_back_to_owned_layer_allocations() {
    let gpu = MockGpuBackend::new();
    gpu.set_max_allocation_bytes(1024);
    let (layers, owners) = alloc_layer_pools(&gpu, 4, 512).unwrap();
    assert_eq!(layers, owners, "fallback views must be allocation bases");
    assert_eq!(gpu.alloc_count(), 4);
    for ptr in owners {
        assert_eq!(gpu.read_alloc(ptr).unwrap(), vec![0; 512]);
        gpu.free(ptr).unwrap();
    }
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn release_frees_backing_allocations_not_layer_views() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    for (mode, narrowed) in [
        (
            metrale_model_layers::ssm_reserve::SsmRollbackMode::Snapshot,
            true,
        ),
        (
            metrale_model_layers::ssm_reserve::SsmRollbackMode::Replay,
            false,
        ),
    ] {
        let gpu = MockGpuBackend::new();
        let mut p = SsmStatePool::new(&config, SLOTS, true, 4, 3, narrowed, mode, &gpu).unwrap();
        assert!(gpu.alloc_count() > 0, "fixture must own device allocations");
        p.release(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0, "mode={mode:?}, narrowed={narrowed}");
        assert!(p.owned_allocations.is_empty());
        assert!(p.h_prefill_stage_pool.is_none());
        assert!(p.replay_input_rings.is_empty());
    }
}

/// 2026-09-25: Replay-mode pool: no per-token intermediates, checkpoints and the
/// input ring allocated, verify refused with an error, draft capacity unlimited.
#[test]
fn replay_pool_has_checkpoints_and_ring_but_no_intermediates() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let p = SsmStatePool::new(
        &config,
        4,
        true,
        4,
        3,
        false,
        metrale_model_layers::ssm_reserve::SsmRollbackMode::Replay,
        &gpu,
    )
    .unwrap();
    assert!(p.h_intermediate_pools.is_empty());
    assert!(p.conv_intermediate_pools.is_empty());
    assert_eq!(p.h_inter_counts, vec![0; p.mtp_slots + 1]);
    assert_eq!(p.h_checkpoint_pools.len(), p.num_ssm_layers);
    assert_eq!(p.replay_input_rings.len(), p.num_ssm_layers);
    assert_eq!(p.verify_draft_capacity(0), usize::MAX);
    let err = p.require_verify_rollback_supported().unwrap_err();
    assert!(err.to_string().contains("EXPERIMENTAL"), "{err}");
    let snap = pool(false);
    assert!(snap.require_verify_rollback_supported().is_ok());
    assert!(snap.replay_input_rings.is_empty());
    assert!(!snap.h_intermediate_pools.is_empty());
}

/// 2026-09-25: Every h family (base slots, intermediates, checkpoints) strides by
/// `h_stored_bytes`, which is `h_bytes / 2` for the f16-sized pool and `h_bytes`
/// otherwise.
#[test]
fn h_families_stride_by_the_stored_width() {
    for f16 in [false, true] {
        let p = pool(f16);
        let expect = if f16 { p.h_bytes / 2 } else { p.h_bytes };
        assert_eq!(p.h_stored_bytes, expect, "f16={f16}");
        assert_eq!(
            p.h_state(0, 1).0 - p.h_state(0, 0).0,
            expect as u64,
            "base slot stride (f16={f16})"
        );
        assert_eq!(
            p.h_intermediate(0, 0, 1).0 - p.h_intermediate(0, 0, 0).0,
            expect as u64,
            "intermediate stride (f16={f16})"
        );
        assert_eq!(
            p.h_checkpoint(0, 1).unwrap().0 - p.h_checkpoint(0, 0).unwrap().0,
            expect as u64,
            "checkpoint stride (f16={f16})"
        );
        // 2026-09-25: Conv slots are `ssm_conv_state_bytes` wide in both modes.
        assert_eq!(
            p.conv_state(0, 1).0 - p.conv_state(0, 0).0,
            p.conv_bytes as u64,
            "conv stride (f16={f16})"
        );
    }
}

/// 2026-09-25: `h_bytes` is the FP32 width in both modes; only `h_stored_bytes`
/// narrows.
#[test]
fn h_bytes_stays_the_fp32_width() {
    let p32 = pool(false);
    let p16 = pool(true);
    assert_eq!(p16.h_bytes, p32.h_bytes);
    assert_eq!(p16.h_stored_bytes * 2, p16.h_bytes);
}

/// 2026-09-25: The FP32 prefill staging arena exists only for the f16-sized pool,
/// strides by `h_bytes`, and holds one blob per slot rather than one per slot and
/// layer, which would be `num_ssm_layers` times larger.
#[test]
fn prefill_staging_is_one_fp32_blob_per_slot_and_only_when_narrowed() {
    let p32 = pool(false);
    assert!(p32.h_prefill_stage_pool.is_none());
    assert!(p32.h_prefill_stage(0).is_none());
    assert!(p32.h_prefill_stage(SLOTS - 1).is_none());

    let p16 = pool(true);
    assert!(p16.h_prefill_stage_pool.is_some());
    let s0 = p16.h_prefill_stage(0).expect("narrowed pool stages");
    let s1 = p16.h_prefill_stage(1).expect("narrowed pool stages");
    assert_eq!(s1.0 - s0.0, p16.h_bytes as u64);
    assert_eq!(s1.0 - s0.0, 2 * p16.h_stored_bytes as u64);
    // 2026-09-25: `SsmStatePool::new` sizes the arena for `max_slots + 1` slots, so
    // the dummy slot has a staging blob too.
    let dummy = p16.h_prefill_stage(p16.dummy_slot()).unwrap();
    assert_eq!(dummy.0 - s0.0, (p16.dummy_slot() * p16.h_bytes) as u64);

    // 2026-09-25: The pool and the preflight reserve (`serve_phases/preflight.rs`)
    // both size the arena with `ssm_h_prefill_stage_bytes`.
    assert_eq!(
        metrale_model_layers::ssm_reserve::ssm_h_prefill_stage_bytes(SLOTS + 1, p16.h_bytes, true),
        (SLOTS + 1) * p16.h_bytes
    );
    assert_eq!(
        metrale_model_layers::ssm_reserve::ssm_h_prefill_stage_bytes(SLOTS + 1, p16.h_bytes, false),
        0
    );
}

/// 2026-10-10: `METRALE_SSM_NO_CHECKPOINT`: the pool allocates exactly the checkpoint
/// share less (2 bulk allocations; `ssm_checkpoint_bytes` bytes), keeps both vecs empty,
/// and the other four families are unchanged. Reserve and allocation use the same
/// `ssm_checkpoint_bytes`, so the saving matches `ssm_pool_reserve_bytes_with`.
#[test]
fn no_checkpoint_pool_omits_exactly_the_checkpoint_share() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    for f16 in [false, true] {
        let mode = metrale_model_layers::ssm_reserve::SsmRollbackMode::Snapshot;
        let build = |no_ckpt| {
            let gpu = MockGpuBackend::new();
            let p = SsmStatePool::new_with_checkpoint_mode(
                &config, SLOTS, true, 4, 3, f16, mode, no_ckpt, &gpu,
            )
            .unwrap();
            (p, gpu.alloc_count())
        };
        let (off, off_allocs) = build(false);
        let (on, on_allocs) = build(true);
        assert!(off.has_checkpoints() && !on.has_checkpoints(), "f16={f16}");
        assert!(on.h_checkpoint_pools.is_empty() && on.conv_checkpoint_pools.is_empty());
        assert_eq!(off_allocs - on_allocs, 2, "f16={f16}");
        assert_eq!(on.owned_allocations.len(), 4 + usize::from(f16));
        assert_eq!(on.mtp_slots, off.mtp_slots);
        let blob = off.num_ssm_layers * (off.h_stored_bytes + off.conv_bytes);
        assert_eq!(
            metrale_model_layers::ssm_reserve::ssm_checkpoint_bytes(off.mtp_slots + 1, blob, false),
            (off.mtp_slots + 1) * blob
        );
    }
}
