// SPDX-License-Identifier: MIT OR Apache-2.0

//! `WeightStore` teardown + FP8 KV-scale-count tests — hoisted from
//! `weights.rs` to keep it under the 500 LoC cap.

use super::*;
use metrale_core::scope::{ModelResource, Teardown};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use std::collections::HashMap;

fn store_with(gpu: &dyn GpuBackend, n: usize) -> WeightStore {
    let mut map = HashMap::new();
    for i in 0..n {
        map.insert(
            format!("w{i}"),
            WeightTensor {
                ptr: gpu.alloc(1024).expect("alloc"),
                shape: vec![16, 16],
                dtype: WeightDtype::BF16,
            },
        );
    }
    WeightStore::from_map(map)
}

#[test]
fn releasing_frees_every_tensor() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 8);
    assert_eq!(gpu.alloc_count(), 8);
    store.release(&gpu).expect("released");
    assert_eq!(gpu.alloc_count(), 0, "every weight was freed");
    assert_eq!(store.len(), 0, "and the map does not hold dead pointers");
}

/// The contract says idempotent: the host calls it, and a `Drop` backstop
/// may call it again. A second call must not double-free.
#[test]
fn releasing_twice_is_harmless() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 4);
    store.release(&gpu).expect("first");
    store.release(&gpu).expect("second");
    assert_eq!(gpu.alloc_count(), 0);
}

/// `fp8_kv_scale_count` counts exactly the `*.k_scale` tensors — one per
/// attention layer in checkpoints that ship calibrated FP8 KV scales —
/// and ignores `v_scale` (paired 1:1 with `k_scale`, counting both would
/// double-report) and lookalike suffixes.
#[test]
fn fp8_kv_scale_count_counts_only_k_scale_tensors() {
    let gpu = MockGpuBackend::new();
    let tensor = || WeightTensor {
        ptr: gpu.alloc(1024).expect("alloc"),
        shape: vec![1],
        dtype: WeightDtype::BF16,
    };
    let mut map = HashMap::new();
    for name in [
        "model.layers.0.self_attn.k_scale",
        "model.layers.0.self_attn.v_scale",
        "model.layers.7.self_attn.k_scale",
        "model.layers.7.self_attn.v_scale",
        "model.layers.0.self_attn.q_proj.weight",
        // Lookalikes that must NOT count: no dot before the suffix, and a
        // different scale kind entirely.
        "model.layers.0.self_attn.attnk_scale",
        "model.layers.0.mlp.weight_scale",
    ] {
        map.insert(name.to_string(), tensor());
    }
    let store = WeightStore::from_map(map);
    assert_eq!(store.fp8_kv_scale_count(), 2);
}

/// A checkpoint without shipped KV scales reports zero — the case where
/// serve logs the "needs calibration or a non-FP8 KV dtype" warning.
#[test]
fn fp8_kv_scale_count_zero_without_scales() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, 4);
    assert_eq!(store.fp8_kv_scale_count(), 0);
}

/// Reverse order, and one failure does not abandon the rest — the whole
/// reason `Teardown` exists rather than `Drop`.
#[test]
fn teardown_releases_in_reverse_registration_order() {
    let gpu = MockGpuBackend::new();
    let mut teardown: Teardown<dyn GpuBackend> = Teardown::new();
    teardown.push(Box::new(store_with(&gpu, 3)));
    teardown.push(Box::new(store_with(&gpu, 5)));
    assert_eq!(gpu.alloc_count(), 8);
    teardown.release_all(&gpu).expect("released");
    assert_eq!(gpu.alloc_count(), 0);
    assert!(teardown.is_empty());
}

/// #736/#915: a buffer a loader DERIVED from these tensors must be released
/// here, not left for `MetraleCudaBackend::sweep_unreleased` to reclaim unowned.
///
/// The mock backend's live-allocation count is the same instrument the CUDA
/// ledger is: "every allocation this backend made and nobody released".
#[test]
fn releasing_frees_adopted_derived_buffers_too() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 4);
    // Two derived copies per tensor, the shape the dense loader produces:
    // a fused concat and its widened block-scale grid.
    for _ in 0..4 {
        store
            .derived()
            .adopt("fused concat", gpu.alloc(2048).expect("alloc"), 2048);
        store
            .derived()
            .adopt("block scale", gpu.alloc(64).expect("alloc"), 64);
    }
    assert_eq!(gpu.alloc_count(), 12);
    assert_eq!(store.derived().len(), 8);
    assert_eq!(store.derived().bytes(), 4 * (2048 + 64));

    store.release(&gpu).expect("released");
    assert_eq!(
        gpu.alloc_count(),
        0,
        "an owned derived buffer must leave nothing for the teardown sweep"
    );
    assert!(store.derived().is_empty());
}

/// The negative control, and the pre-#915 state: a derived buffer nobody
/// adopted survives `release` and is exactly what the H100 sweep reported as
/// "28.01 GB ... had no owner".
#[test]
fn an_unadopted_derived_buffer_is_what_the_sweep_would_report() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 2);
    let orphan = gpu.alloc(4096).expect("alloc");
    store.release(&gpu).expect("released");
    assert_eq!(
        gpu.alloc_count(),
        1,
        "the orphan outlives teardown — adopt it via `store.derived()`"
    );
    gpu.free(orphan).expect("freed");
}

/// Releasing twice must not double-free an adopted buffer either.
#[test]
fn releasing_twice_is_harmless_for_derived_buffers() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 1);
    store
        .derived()
        .adopt("twin", gpu.alloc(128).expect("alloc"), 128);
    store.release(&gpu).expect("released");
    store.release(&gpu).expect("released again");
    assert_eq!(gpu.alloc_count(), 0);
}

/// 2026-10-14: `freeable_bytes_matching` predicts exactly the bytes `free_matching` then frees
/// for the same predicate (the DFlash KV credit, #82 hybrid M1), and skips what it skips.
#[test]
fn freeable_bytes_matching_equals_what_free_matching_frees() {
    let gpu = MockGpuBackend::new();
    let mut map = HashMap::new();
    for (name, shape, dtype) in [
        (
            "layers.0.self_attn.q_proj.weight",
            vec![64, 32],
            WeightDtype::BF16,
        ),
        (
            "layers.0.mlp.down_proj.weight",
            vec![32, 128],
            WeightDtype::BF16,
        ),
        (
            "layers.1.self_attn.q_proj.weight",
            vec![64, 32],
            WeightDtype::FP8E4M3,
        ),
        ("fc.weight", vec![32, 32], WeightDtype::BF16),
    ] {
        let t = WeightTensor {
            ptr: gpu.alloc(1024).expect("alloc"),
            shape,
            dtype,
        };
        map.insert(name.to_string(), t);
    }
    let mut store = WeightStore::from_map(map);
    let pred =
        |n: &str, t: &WeightTensor| n.ends_with("_proj.weight") && t.dtype == WeightDtype::BF16;
    let predicted = store.freeable_bytes_matching(pred);
    assert_eq!(
        predicted,
        (64 * 32 + 32 * 128) * 2,
        "the two BF16 projections only"
    );
    let names: std::collections::HashSet<String> = store
        .names()
        .filter(|n| pred(n, store.get(n).expect("present")))
        .map(str::to_owned)
        .collect();
    let (count, freed) = store
        .free_matching(&gpu, |n| names.contains(n))
        .expect("freed");
    assert_eq!((count, freed), (2, predicted));
    assert_eq!(
        store.freeable_bytes_matching(pred),
        0,
        "nothing left to free"
    );
}
