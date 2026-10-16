// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of which arm each stored layout takes, through a real
//! [`WeightStore`]: BF16 and packed-NVFP4 dense MLP weights, the mHC tensors at
//! either width, U8, BF16 and unsupported routed experts, and the prune
//! predicate.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use std::collections::HashMap;

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{LayerSource, bind_expert, is_quantized_expert_weight, nvfp4_dequant};

const LAYER: usize = 45;

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

struct StoreBuilder {
    gpu: MockGpuBackend,
    map: HashMap<String, WeightTensor>,
}

impl StoreBuilder {
    fn new() -> Self {
        Self {
            gpu: MockGpuBackend::new(),
            map: HashMap::new(),
        }
    }

    fn put(&mut self, name: &str, bytes: &[u8], shape: &[usize], dtype: WeightDtype) -> DevicePtr {
        let p = self.gpu.alloc(bytes.len().max(1)).unwrap();
        self.gpu.copy_h2d(bytes, p).unwrap();
        self.map.insert(
            name.to_string(),
            WeightTensor {
                ptr: p,
                shape: shape.to_vec(),
                dtype,
            },
        );
        p
    }

    fn finish(self) -> (MockGpuBackend, WeightStore) {
        (self.gpu, WeightStore::from_map(self.map))
    }
}

fn qualified(leaf: &str) -> String {
    format!("model.language_model.layers.{LAYER}.{leaf}")
}

/// 2026-09-25: A BF16 dense-MLP weight with no scale siblings reads back through
/// `LayerSource::f32` as its BF16 values.
#[test]
fn a_bf16_dense_mlp_still_takes_the_plain_float_path() {
    let values = ramp(64);
    let mut b = StoreBuilder::new();
    b.put(
        &qualified("mlp.gate_proj.weight"),
        &bf16_bytes(&values),
        &[4, 16],
        WeightDtype::BF16,
    );
    let (gpu, store) = b.finish();
    let src = LayerSource::collect(&gpu, &store, LAYER).unwrap();

    let got = src.f32("mlp.gate_proj.weight").unwrap();
    let want: Vec<f32> = values
        .iter()
        .map(|x| half::bf16::from_f32(*x).to_f32())
        .collect();
    assert_eq!(got, want, "BF16 must round-trip through f32 unchanged");
}

/// 2026-09-25: A packed-NVFP4 dense-MLP weight with its two scale siblings
/// reads back through the same `LayerSource::f32` as its dequantized values; an
/// `input_scale` beside it changes nothing.
#[test]
fn a_packed_dense_mlp_is_dequantised_at_the_same_entry_point() {
    let packed: Vec<u8> = (0..32u8).map(|i| i.wrapping_mul(37)).collect();
    let scales = vec![0x38u8, 0x3C, 0x30, 0x40];
    let scale_2 = 0.25f32;

    let mut b = StoreBuilder::new();
    b.put(
        &qualified("mlp.gate_proj.weight"),
        &packed,
        &[4, 8],
        WeightDtype::UInt8,
    );
    b.put(
        &qualified("mlp.gate_proj.weight_scale"),
        &scales,
        &[4, 1],
        WeightDtype::FP8E4M3,
    );
    b.put(
        &qualified("mlp.gate_proj.weight_scale_2"),
        &scale_2.to_le_bytes(),
        &[],
        WeightDtype::FP32,
    );
    b.put(
        &qualified("mlp.gate_proj.input_scale"),
        &1.5f32.to_le_bytes(),
        &[],
        WeightDtype::FP32,
    );
    let (gpu, store) = b.finish();
    let src = LayerSource::collect(&gpu, &store, LAYER).unwrap();

    let got = src.f32("mlp.gate_proj.weight").unwrap();
    let want =
        nvfp4_dequant::dequant_nvfp4_to_f32("ref", &packed, &[4, 8], &scales, scale_2).unwrap();
    assert_eq!(got.len(), 64, "[4, 8] U8 is a [4, 16] weight");
    assert_eq!(got, want);
}

/// 2026-09-25: A packed weight without its `weight_scale` is an error.
#[test]
fn a_packed_weight_without_its_scales_is_an_error() {
    let mut b = StoreBuilder::new();
    b.put(
        &qualified("mlp.gate_proj.weight"),
        &[0u8; 8],
        &[1, 8],
        WeightDtype::UInt8,
    );
    let (gpu, store) = b.finish();
    let src = LayerSource::collect(&gpu, &store, LAYER).unwrap();
    let err = src.f32("mlp.gate_proj.weight").unwrap_err().to_string();
    assert!(err.contains("weight_scale is absent"), "{err}");
}

/// 2026-09-25: An mHC tensor holding BF16-representable values reads back
/// exactly the same whether it is stored as FP32 or BF16. `bind_mhc_site` reads
/// the mHC tensors through `LayerSource::f32`, and `plan_dtype` makes no copy
/// of them (they are neither in `KDA_TENSORS` nor `self_attn.*`).
#[test]
fn the_mhc_tensors_read_the_same_at_either_stored_width() {
    let vals = ramp(24);
    let rounded: Vec<f32> = vals
        .iter()
        .map(|x| half::bf16::from_f32(*x).to_f32())
        .collect();

    let mut wide = StoreBuilder::new();
    wide.put(
        &qualified("hc_attn_base"),
        &rounded
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
        &[24],
        WeightDtype::FP32,
    );
    let (gpu, store) = wide.finish();
    let from_f32 = LayerSource::collect(&gpu, &store, LAYER)
        .unwrap()
        .f32("hc_attn_base")
        .unwrap();

    let mut narrow = StoreBuilder::new();
    narrow.put(
        &qualified("hc_attn_base"),
        &bf16_bytes(&rounded),
        &[24],
        WeightDtype::BF16,
    );
    let (gpu, store) = narrow.finish();
    let from_bf16 = LayerSource::collect(&gpu, &store, LAYER)
        .unwrap()
        .f32("hc_attn_base")
        .unwrap();

    assert_eq!(from_f32, rounded);
    assert_eq!(from_bf16, rounded);
    assert_eq!(
        from_f32, from_bf16,
        "storage width must not change the mHC values"
    );
}

fn put_expert(b: &mut StoreBuilder, id: usize, quantized: bool) -> Vec<DevicePtr> {
    let mut ptrs = Vec::new();
    for p in ["gate_proj", "up_proj", "down_proj"] {
        let base = qualified(&format!("mlp.experts.{id}.{p}"));
        if quantized {
            ptrs.push(b.put(
                &format!("{base}.weight"),
                &[0x21u8; 16],
                &[2, 8],
                WeightDtype::UInt8,
            ));
            b.put(
                &format!("{base}.weight_scale"),
                &[0x38u8, 0x38],
                &[2, 1],
                WeightDtype::FP8E4M3,
            );
            b.put(
                &format!("{base}.weight_scale_2"),
                &0.5f32.to_le_bytes(),
                &[],
                WeightDtype::FP32,
            );
        } else {
            ptrs.push(b.put(
                &format!("{base}.weight"),
                &bf16_bytes(&ramp(32)),
                &[2, 16],
                WeightDtype::BF16,
            ));
        }
    }
    ptrs
}

/// 2026-09-25: A U8 expert is bound zero-copy: the store's own pointers, and
/// nothing added to the derived-weight ledger.
#[test]
fn a_packed_expert_is_still_bound_zero_copy() {
    let mut b = StoreBuilder::new();
    let ptrs = put_expert(&mut b, 0, true);
    let (gpu, store) = b.finish();

    let e = bind_expert(&gpu, &store, LAYER, 0).unwrap();
    assert_eq!(e.gate_proj.packed, ptrs[0]);
    assert_eq!(e.up_proj.packed, ptrs[1]);
    assert_eq!(e.down_proj.packed, ptrs[2]);
    assert_eq!(e.gate_proj.scale_2, 0.5);
    assert!(
        store.derived().is_empty(),
        "the packed path must derive nothing"
    );
}

/// 2026-09-25: A BF16 expert is quantised into new buffers, adopted by the
/// store, byte-identical to quantising the same values directly.
#[test]
fn a_bf16_expert_is_quantised_into_a_fresh_smaller_buffer() {
    let mut b = StoreBuilder::new();
    let ptrs = put_expert(&mut b, 0, false);
    let (gpu, store) = b.finish();

    let e = bind_expert(&gpu, &store, LAYER, 0).unwrap();
    assert_ne!(
        e.gate_proj.packed, ptrs[0],
        "a BF16 expert cannot be bound in place"
    );

    let values: Vec<f32> = ramp(32)
        .iter()
        .map(|x| half::bf16::from_f32(*x).to_f32())
        .collect();
    let want = super::nvfp4_quant::quantize_to_nvfp4("ref", &values, 2, 16).unwrap();
    assert_eq!(gpu.read_alloc(e.gate_proj.packed).unwrap(), want.packed);
    assert_eq!(gpu.read_alloc(e.gate_proj.scale).unwrap(), want.scales);
    assert_eq!(e.gate_proj.scale_2, want.scale_2);
    assert_eq!(want.packed.len(), 16, "[2, 16] BF16 -> [2, 8] packed");

    // 2026-09-25: Packed codes and scales for each of the three projections are
    // in the derived-weight ledger.
    assert_eq!(store.derived().len(), 6);
    assert_eq!(store.derived().bytes(), 3 * (16 + 2));
}

/// 2026-09-25: A resident expert that is neither U8 nor BF16 is refused.
#[test]
fn an_expert_at_an_unsupported_width_is_refused() {
    let mut b = StoreBuilder::new();
    b.put(
        &qualified("mlp.experts.0.gate_proj.weight"),
        &[0u8; 16],
        &[1, 16],
        WeightDtype::FP8E4M3,
    );
    let (gpu, store) = b.finish();
    let err = bind_expert(&gpu, &store, LAYER, 0).unwrap_err().to_string();
    assert!(err.contains("expected packed U8 NVFP4 or BF16"), "{err}");
}

/// 2026-09-25: `is_quantized_expert_weight` matches a BF16 routed-expert
/// weight only: not a U8 expert, a shared expert, the router, an attention
/// weight, a scale sibling, or a name outside `model.language_model.layers.`.
#[test]
fn only_a_bf16_expert_weight_is_prunable() {
    let w = qualified("mlp.experts.7.down_proj.weight");
    assert!(is_quantized_expert_weight(&w, WeightDtype::BF16));
    assert!(!is_quantized_expert_weight(&w, WeightDtype::UInt8));
    assert!(!is_quantized_expert_weight(
        &qualified("mlp.shared_experts.gate_proj.weight"),
        WeightDtype::BF16
    ));
    assert!(!is_quantized_expert_weight(
        &qualified("mlp.gate.weight"),
        WeightDtype::BF16
    ));
    assert!(!is_quantized_expert_weight(
        &qualified("self_attn.q_proj.weight"),
        WeightDtype::BF16
    ));
    assert!(!is_quantized_expert_weight(
        &qualified("mlp.experts.7.down_proj.weight_scale"),
        WeightDtype::BF16
    ));
    assert!(!is_quantized_expert_weight(
        "model.layers.7.mlp.experts.3.down_proj.weight",
        WeightDtype::BF16
    ));
}

/// 2026-10-16: `METRALE_GLM_DENSE_MLP_CKPT`: a packed dense MLP read through
/// `LayerSource::packed_nvfp4` and `dense_ckpt::for_rank` gives each TP rank an NVFP4 copy whose
/// GEMV decode (`dense_ckpt::gemv_weights_f32`) matches the same rank's slice of the BF16-path
/// source (`LayerSource::f32`, the loader's dequant, sliced as `build_dense_mlp` slices it) to
/// within one f32 ulp (the GEMV forms `e4m3 * scale_2` first), with no second rounding; a BF16
/// tensor is not packed (`None`).
#[test]
fn a_packed_dense_mlp_gives_each_rank_its_checkpoint_nvfp4_copy() {
    use crate::glm5next_mlp::dense_ckpt::{for_rank, gemv_weights_f32};
    let (hidden, inter, tp) = (32usize, 64usize, 2usize);
    let mut b = StoreBuilder::new();
    for (i, (p, rows, cols)) in [
        ("gate_proj", inter, hidden),
        ("up_proj", inter, hidden),
        ("down_proj", hidden, inter),
    ]
    .into_iter()
    .enumerate()
    {
        let packed: Vec<u8> = (0..rows * cols / 2)
            .map(|j| (j as u8).wrapping_mul(37).wrapping_add(i as u8 * 11))
            .collect();
        let scales: Vec<u8> = (0..rows * cols / 16)
            .map(|j| 0x30 + ((j * 7 + i) % 0x18) as u8)
            .collect();
        let s2 = [0.0123f32, 0.25, 3.1e-4][i];
        b.put(&qualified(&format!("mlp.{p}.weight")), &packed, &[rows, cols / 2], WeightDtype::UInt8);
        b.put(
            &qualified(&format!("mlp.{p}.weight_scale")),
            &scales,
            &[rows, cols / 16],
            WeightDtype::FP8E4M3,
        );
        b.put(
            &qualified(&format!("mlp.{p}.weight_scale_2")),
            &s2.to_le_bytes(),
            &[],
            WeightDtype::FP32,
        );
    }
    b.put(&qualified("mlp.extra.weight"), &bf16_bytes(&ramp(64)), &[4, 16], WeightDtype::BF16);
    let (gpu, store) = b.finish();
    let src = LayerSource::collect(&gpu, &store, LAYER).unwrap();
    assert!(src.packed_nvfp4("mlp.extra.weight").unwrap().is_none(), "BF16 is not packed");

    let get = |n: &str| src.packed_nvfp4(n);
    let gate = src.f32("mlp.gate_proj.weight").unwrap();
    let down = src.f32("mlp.down_proj.weight").unwrap();
    let near = |got: &[f32], want: &[f32], what: &str| {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let ulp = f32::from_bits(w.abs().to_bits() + 1) - w.abs();
            assert!((g - w).abs() <= ulp, "{what}[{i}]: {g} vs {w}");
        }
    };
    let local = inter / tp;
    for rank in 0..tp {
        let lo = rank * local;
        let c = for_rank(hidden, inter, rank, tp, "mlp", &get).unwrap().expect("packed");
        let g = gemv_weights_f32(&c.gate_proj, local, hidden);
        near(&g, &gate[lo * hidden..(lo + local) * hidden], "gate");
        let d = gemv_weights_f32(&c.down_proj, hidden, local);
        let want: Vec<f32> = down
            .chunks(inter)
            .flat_map(|r| r[lo..lo + local].iter().copied())
            .collect();
        near(&d, &want, "down");
    }
}
