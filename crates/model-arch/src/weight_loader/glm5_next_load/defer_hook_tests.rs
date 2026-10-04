// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of the defer hook: which tensors `Glm5NextWeightLoader`
//! keeps off the device, what it never defers, and that a deferred expert
//! quantises to the same bytes as a resident one.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

use super::{bind_expert, defer_rule, input_scale_value, is_full_width_mtp_expert, read_input_scale};
use crate::weight_loader::ModelWeightLoader;

const LAYERS: usize = 45;

fn qualified(layer: usize, leaf: &str) -> String {
    format!("model.language_model.layers.{layer}.{leaf}")
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect()
}

/// 2026-09-25: `n` sign-mixed, evenly spaced values, so a swapped nibble or a
/// dropped scale changes the quantised bytes.
fn ramp(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 - n as f32 / 2.0) * 0.125)
        .collect()
}

#[test]
fn only_the_mtp_layers_full_width_routed_experts_are_deferred() {
    let yes = |n: &str| is_full_width_mtp_expert(n, WeightDtype::BF16, LAYERS);
    for p in ["gate_proj", "up_proj", "down_proj"] {
        assert!(yes(&qualified(
            LAYERS,
            &format!("mlp.experts.0.{p}.weight")
        )));
        assert!(yes(&qualified(
            LAYERS,
            &format!("mlp.experts.287.{p}.weight")
        )));
    }

    // 2026-09-25: The MTP layer's shared expert, router and other non-expert
    // tensors are not deferred.
    for leaf in [
        "mlp.shared_experts.gate_proj.weight",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "self_attn.q_proj.weight",
        "eh_proj.weight",
    ] {
        assert!(!yes(&qualified(LAYERS, leaf)), "{leaf}");
    }

    // 2026-09-25: Text-layer experts are not deferred.
    assert!(!yes(&qualified(44, "mlp.experts.0.gate_proj.weight")));
    assert!(!yes(&qualified(0, "mlp.experts.0.gate_proj.weight")));

    assert!(!yes("lm_head.weight"));
    assert!(!yes(
        "model.language_model.layers.x.mlp.experts.0.up_proj.weight"
    ));
}

/// 2026-09-25: A routed expert of the MTP layer stored as U8, FP8 or FP32 is
/// not deferred, and neither is a `weight_scale` sibling.
#[test]
fn a_packed_expert_is_never_deferred_whatever_the_layer() {
    for dtype in [WeightDtype::UInt8, WeightDtype::FP8E4M3, WeightDtype::FP32] {
        assert!(!is_full_width_mtp_expert(
            &qualified(LAYERS, "mlp.experts.0.gate_proj.weight"),
            dtype,
            LAYERS
        ));
    }
    assert!(!is_full_width_mtp_expert(
        &qualified(LAYERS, "mlp.experts.0.gate_proj.weight_scale"),
        WeightDtype::BF16,
        LAYERS
    ));
}

/// 2026-09-25: The deferred layer is `num_hidden_layers` of the config given to
/// `defer_predicate`.
#[test]
fn the_deferred_layer_comes_from_the_config_not_a_literal() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.num_hidden_layers = 7;
    let hook = super::Glm5NextWeightLoader
        .defer_predicate(&config)
        .expect("glm5_next declares a defer predicate");

    assert!(hook(
        &qualified(7, "mlp.experts.3.down_proj.weight"),
        WeightDtype::BF16
    ));
    assert!(!hook(
        &qualified(45, "mlp.experts.3.down_proj.weight"),
        WeightDtype::BF16
    ));
}

/// 2026-09-25: A loader that does not override `defer_predicate` gets `None`,
/// which defers nothing.
#[test]
fn a_loader_that_does_not_override_the_hook_defers_nothing() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(
        crate::weight_loader::qwen35::Qwen35WeightLoader
            .defer_predicate(&config)
            .is_none()
    );
}

/// 2026-09-25: Write `bytes` into a scratch shard at `offset`, after `offset`
/// filler bytes.
fn stage_shard(tag: &str, offset: u64, bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("metrale-glm5next-defer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{tag}.safetensors"));
    let mut blob = vec![0xAAu8; offset as usize];
    blob.extend_from_slice(bytes);
    std::fs::write(&path, &blob).unwrap();
    path
}

/// 2026-09-25: An expert read from its shard and the same expert read back
/// from the device quantise to identical NVFP4 bytes and global scales.
#[test]
fn a_deferred_expert_quantises_to_the_same_bytes_as_a_resident_one() {
    let values = ramp(32);
    let (rows, cols) = (2usize, 16usize);
    let bytes = bf16_bytes(&values);

    let gpu_a = MockGpuBackend::new();
    let mut map = std::collections::HashMap::new();
    for p in ["gate_proj", "up_proj", "down_proj"] {
        let ptr = gpu_a.alloc(bytes.len()).unwrap();
        gpu_a.copy_h2d(&bytes, ptr).unwrap();
        map.insert(
            qualified(LAYERS, &format!("mlp.experts.0.{p}.weight")),
            WeightTensor {
                ptr,
                shape: vec![rows, cols],
                dtype: WeightDtype::BF16,
            },
        );
    }
    let resident = bind_expert(&gpu_a, &WeightStore::from_map(map), LAYERS, 0).unwrap();

    let path = stage_shard("expert", 137, &bytes);
    let gpu_b = MockGpuBackend::new();
    let mut store = WeightStore::from_map(std::collections::HashMap::new());
    for p in ["gate_proj", "up_proj", "down_proj"] {
        store.defer(
            qualified(LAYERS, &format!("mlp.experts.0.{p}.weight")),
            DeferredTensor {
                path: path.clone(),
                offset: 137,
                shape: vec![rows, cols],
                dtype: WeightDtype::BF16,
            },
        );
    }
    let deferred = bind_expert(&gpu_b, &store, LAYERS, 0).unwrap();

    for (a, b, what) in [
        (&resident.gate_proj, &deferred.gate_proj, "gate"),
        (&resident.up_proj, &deferred.up_proj, "up"),
        (&resident.down_proj, &deferred.down_proj, "down"),
    ] {
        assert_eq!(
            gpu_a.read_alloc(a.packed).unwrap(),
            gpu_b.read_alloc(b.packed).unwrap(),
            "{what}: packed codes differ between the shard and device paths"
        );
        assert_eq!(
            gpu_a.read_alloc(a.scale).unwrap(),
            gpu_b.read_alloc(b.scale).unwrap(),
            "{what}: block scales differ"
        );
        assert_eq!(a.scale_2, b.scale_2, "{what}: global scale differs");
    }

    // 2026-09-25: The deferred arm adopted only the NVFP4 buffers: per
    // projection, packed `[2, 8]` and scales `[2, 1]`.
    assert_eq!(store.derived().len(), 6);
    assert_eq!(store.derived().bytes(), 3 * (rows * cols / 2 + rows));
    assert!(
        store.derived().bytes() < 3 * bytes.len(),
        "the point of deferring is that less reaches the device than is on disk"
    );

    let _ = std::fs::remove_file(&path);
}

/// 2026-09-25: A deferred expert that is not BF16 is refused, with an error that
/// says it was deferred.
#[test]
fn a_deferred_expert_at_an_unsupported_width_is_refused() {
    let path = stage_shard("odd-width", 0, &[0x21u8; 16]);
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::from_map(std::collections::HashMap::new());
    store.defer(
        qualified(LAYERS, "mlp.experts.0.gate_proj.weight"),
        DeferredTensor {
            path: path.clone(),
            offset: 0,
            shape: vec![2, 8],
            dtype: WeightDtype::UInt8,
        },
    );
    let err = bind_expert(&gpu, &store, LAYERS, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("deferred"), "{err}");

    let _ = std::fs::remove_file(&path);
}

/// 2026-10-03: The CUTLASS W4A4 lever adds every `*.input_scale` to the defer rule and nothing
/// else; off, the rule is exactly `is_full_width_mtp_expert`.
#[test]
fn the_cutlass_w4a4_lever_defers_activation_scales_only() {
    let names = [
        qualified(3, "mlp.experts.7.gate_proj.input_scale"),
        qualified(3, "mlp.experts.7.down_proj.input_scale"),
        qualified(0, "mlp.gate_proj.input_scale"),
        qualified(3, "mlp.shared_experts.up_proj.input_scale"),
    ];
    for n in &names {
        assert!(defer_rule(n, WeightDtype::FP32, LAYERS, true), "{n}");
        assert!(!defer_rule(n, WeightDtype::FP32, LAYERS, false), "{n}");
    }
    for leaf in [
        "mlp.experts.7.gate_proj.weight_scale",
        "mlp.experts.7.gate_proj.weight_scale_2",
        "mlp.experts.7.gate_proj.weight",
        "mlp.gate.e_score_correction_bias",
    ] {
        for dtype in [WeightDtype::FP32, WeightDtype::UInt8, WeightDtype::FP8E4M3] {
            assert!(!defer_rule(&qualified(3, leaf), dtype, LAYERS, true), "{leaf}");
        }
    }
    // 2026-10-03: The MTP experts stay deferred either way.
    let mtp = qualified(LAYERS, "mlp.experts.0.gate_proj.weight");
    assert!(defer_rule(&mtp, WeightDtype::BF16, LAYERS, false));
    assert!(defer_rule(&mtp, WeightDtype::BF16, LAYERS, true));
}

/// 2026-10-03: Only one finite positive F32 is a usable activation scale.
#[test]
fn an_input_scale_must_be_one_finite_positive_f32() {
    let b = |v: f32| v.to_le_bytes().to_vec();
    assert_eq!(input_scale_value(&b(0.0123), WeightDtype::FP32, 1), Some(0.0123));
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(input_scale_value(&b(bad), WeightDtype::FP32, 1), None, "{bad}");
    }
    assert_eq!(input_scale_value(&b(1.0), WeightDtype::BF16, 1), None);
    assert_eq!(input_scale_value(&[0u8; 8], WeightDtype::FP32, 2), None);
}

/// 2026-10-03: A deferred `input_scale` is read from its shard; an absent one is 0.0.
#[test]
fn a_deferred_input_scale_is_read_from_its_shard() {
    let v = 2.5e-4f32;
    let path = stage_shard("input-scale", 41, &v.to_le_bytes());
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::from_map(std::collections::HashMap::new());
    store.defer(
        qualified(5, "mlp.experts.9.up_proj.input_scale"),
        DeferredTensor {
            path: path.clone(),
            offset: 41,
            shape: vec![],
            dtype: WeightDtype::FP32,
        },
    );
    assert_eq!(read_input_scale(&gpu, &store, 5, "mlp.experts.9.up_proj"), v);
    assert_eq!(read_input_scale(&gpu, &store, 5, "mlp.experts.9.gate_proj"), 0.0);
    assert_eq!(read_input_scale(&gpu, &store, 6, "mlp.experts.9.up_proj"), 0.0);
    let _ = std::fs::remove_file(&path);
}
