// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: [`super::fp8_dequant`] on the FP4-Local layouts (MXFP8 `[N, K/32]`
//! U8 E8M0; block-128 F32 `[N/128, 1, K/128, 1]`). Expected values are worked by
//! hand from the encodings: E4M3 0x38 = 1.0, 0xB8 = -1.0, 0x40 = 2.0, 0x01 =
//! 2^-9 (smallest subnormal), 0x7E = 448 (largest finite); E8M0 e = 2^(e-127).
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants: none beyond the types.

use super::fp8_dequant::{HostTensors, dequant_one, dequant_scaled_fp8};
use super::*;
use crate::glm5next_kda::binding::KdaDtype;

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()
}

#[test]
fn mxfp8_is_exact_per_32_group() {
    // [2, 64]: row 0 groups scale 2^3 and 2^-2, row 1 groups 2^0 and 2^10.
    let mut w = vec![0x38u8; 2 * 64];
    w[1] = 0xB8; // -1.0
    w[32] = 0x40; // 2.0, group 1
    w[64] = 0x01; // 2^-9, row 1 group 0
    w[64 + 63] = 0x7E; // 448, row 1 group 1
    let s = vec![130u8, 125, 127, 137];
    let out = bf16s(&dequant_one("x.weight", &[2, 64], &w, WeightDtype::UInt8, &[2, 2], &s).unwrap());
    assert_eq!(out[0], 8.0);
    assert_eq!(out[1], -8.0);
    assert_eq!(out[31], 8.0);
    assert_eq!(out[32], 0.5); // 2.0 * 2^-2
    assert_eq!(out[33], 0.25);
    assert_eq!(out[64], 2f32.powi(-9));
    assert_eq!(out[65], 1.0);
    assert_eq!(out[64 + 63], 448.0 * 1024.0);
}

#[test]
fn block128_scales_by_tile_and_rounds_to_bf16() {
    // [256, 256]: tiles (0,0) 0.5, (0,1) 3.0, (1,0) -2.0, (1,1) 0.1 (rounds in BF16).
    let w = vec![0x38u8; 256 * 256];
    let s: Vec<u8> = [0.5f32, 3.0, -2.0, 0.1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = bf16s(&dequant_one("x.weight", &[256, 256], &w, WeightDtype::FP32, &[2, 1, 2, 1], &s).unwrap());
    assert_eq!(out[0], 0.5);
    assert_eq!(out[127], 0.5);
    assert_eq!(out[128], 3.0);
    assert_eq!(out[128 * 256], -2.0);
    let v = out[255 * 256 + 255];
    assert_eq!(v, half::bf16::from_f32(0.1).to_f32());
    assert!(((v - 0.1) / 0.1).abs() <= 2f32.powi(-9));
}

#[test]
fn ragged_block_edge_uses_ceil_tiles() {
    // DSA kv_a is [512, 4096] -> [4, 1, 32, 1]; a K not divisible by 128 must still map its tail.
    let w = vec![0x38u8; 130 * 130];
    let s: Vec<u8> = [1.0f32, 2.0, 4.0, 8.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = bf16s(&dequant_one("x.weight", &[130, 130], &w, WeightDtype::FP32, &[2, 1, 2, 1], &s).unwrap());
    assert_eq!(out[129], 2.0);
    assert_eq!(out[129 * 130], 4.0);
    assert_eq!(out[129 * 130 + 129], 8.0);
}

#[test]
fn mismatched_scales_and_nan_are_refused() {
    let w = vec![0x38u8; 2 * 64];
    let e = dequant_one("x.weight", &[2, 64], &w, WeightDtype::UInt8, &[2, 4], &[127; 8]).unwrap_err().to_string();
    assert!(e.contains("MXFP8 scale shape"), "{e}");
    let e = dequant_one("x.weight", &[2, 64], &w, WeightDtype::UInt8, &[2, 2], &[127, 255, 127, 127]).unwrap_err().to_string();
    assert!(e.contains("NaN"), "{e}");
    let e = dequant_one("x.weight", &[2, 64], &w, WeightDtype::FP32, &[1, 1, 2, 1], &[0; 8]).unwrap_err().to_string();
    assert!(e.contains("block-128 scale shape"), "{e}");
    let e = dequant_one("x.weight", &[2, 64], &w, WeightDtype::BF16, &[2, 2], &[0; 8]).unwrap_err().to_string();
    assert!(e.contains("neither MXFP8"), "{e}");
}

fn layer(entries: Vec<(&str, WeightDtype, Vec<usize>, Vec<u8>)>) -> (HostTensors, Vec<String>) {
    let names = entries.iter().map(|e| e.0.to_string()).collect();
    let t = entries.into_iter().map(|(n, d, s, b)| (n.to_string(), (d, s, b))).collect();
    (t, names)
}

#[test]
fn layer_pass_converts_and_drops_scale_siblings() {
    let (mut t, mut names) = layer(vec![
        ("self_attn.q_proj.weight", WeightDtype::FP8E4M3, vec![2, 64], vec![0x38; 128]),
        ("self_attn.q_proj.weight_scale", WeightDtype::UInt8, vec![2, 2], vec![127; 4]),
        ("self_attn.b_proj.weight", WeightDtype::BF16, vec![2, 2], vec![0; 8]),
        ("input_layernorm.weight", WeightDtype::BF16, vec![2], vec![0; 4]),
    ]);
    assert_eq!(dequant_scaled_fp8(&mut t, &mut names).unwrap(), 1);
    assert_eq!(t["self_attn.q_proj.weight"].0, WeightDtype::BF16);
    assert_eq!(t["self_attn.q_proj.weight"].2.len(), 2 * 64 * 2);
    assert!(!t.contains_key("self_attn.q_proj.weight_scale"));
    assert!(!names.iter().any(|n| n.ends_with("weight_scale")), "{names:?}");
    assert_eq!(names.len(), 3);
    assert_eq!(t["self_attn.b_proj.weight"].2, vec![0; 8], "BF16 tensors are untouched");
}

#[test]
fn official_layer_is_unchanged() {
    // Official NVFP4: packed U8 weight + F8_E4M3 `.weight_scale` (a scale, not a weight).
    let entries = vec![
        ("mlp.gate_proj.weight", WeightDtype::UInt8, vec![4, 8], vec![0u8; 32]),
        ("mlp.gate_proj.weight_scale", WeightDtype::FP8E4M3, vec![4, 1], vec![0x38; 4]),
        ("mlp.gate_proj.weight_scale_2", WeightDtype::FP32, vec![], 1f32.to_le_bytes().to_vec()),
    ];
    let (mut t, mut names) = layer(entries);
    let before = (t.clone(), names.clone());
    assert_eq!(dequant_scaled_fp8(&mut t, &mut names).unwrap(), 0);
    assert_eq!((t, names), before);
}

#[test]
fn unscaled_fp8_weight_is_an_error() {
    let (mut t, mut names) = layer(vec![("self_attn.o_proj.weight", WeightDtype::FP8E4M3, vec![2, 64], vec![0x38; 128])]);
    let e = dequant_scaled_fp8(&mut t, &mut names).unwrap_err().to_string();
    assert!(e.contains("is absent"), "{e}");
}

#[test]
fn kda_binder_sees_bf16_and_no_scale_names() {
    // The KDA source after the pass: q_proj offered as BF16, the scale gone from names().
    let (mut tensors, mut names) = layer(vec![
        ("self_attn.q_proj.weight", WeightDtype::FP8E4M3, vec![2, 64], vec![0x38; 128]),
        ("self_attn.q_proj.weight_scale", WeightDtype::UInt8, vec![2, 2], vec![127; 4]),
    ]);
    dequant_scaled_fp8(&mut tensors, &mut names).unwrap();
    let src = LayerSource {
        names,
        tensors,
        plan_cast: std::collections::BTreeMap::new(),
    };
    let t = src.get("self_attn.q_proj.weight").expect("BF16 q_proj is offered");
    assert!(matches!(t.dtype, KdaDtype::Bf16));
    assert_eq!(src.names(), vec!["self_attn.q_proj.weight".to_string()]);
    assert_eq!(src.f32("self_attn.q_proj.weight").unwrap()[0], 1.0);
}

#[test]
fn lm_head_plan_bf16_and_prepacked_are_bound_as_is() {
    assert!(super::loader::lm_head_bf16_plan(WeightDtype::BF16, &[4, 64], &[], None).unwrap().is_none());
    assert!(super::loader::lm_head_bf16_plan(WeightDtype::UInt8, &[4, 32], &[], None).unwrap().is_none());
}

#[test]
fn lm_head_plan_dequantizes_mxfp8_head() {
    // FP4-Local: lm_head F8 [154880, 4096] + U8 [154880, 128]; here [2, 64] + [2, 2].
    let scale = (WeightDtype::UInt8, vec![2, 2], vec![127u8, 128, 126, 127]);
    let b = super::loader::lm_head_bf16_plan(WeightDtype::FP8E4M3, &[2, 64], &[0x38; 128], Some(&scale))
        .unwrap()
        .expect("F8 head is dequantized");
    let v = bf16s(&b);
    assert_eq!(v.len(), 128, "BF16 element count = vocab x hidden");
    assert_eq!((v[0], v[32], v[64], v[96]), (1.0, 2.0, 0.5, 1.0));
}

#[test]
fn lm_head_plan_refuses_unscaled_fp8_and_other_dtypes() {
    let e = super::loader::lm_head_bf16_plan(WeightDtype::FP8E4M3, &[2, 64], &[0x38; 128], None).unwrap_err().to_string();
    assert!(e.contains("weight_scale is absent"), "{e}");
    for dt in [WeightDtype::FP32, WeightDtype::FP8E8M0, WeightDtype::Int64, WeightDtype::UInt16] {
        let e = super::loader::lm_head_bf16_plan(dt, &[2, 64], &[], None).unwrap_err().to_string();
        assert!(e.contains("only BF16, prepacked NVFP4 (U8) or scaled F8_E4M3"), "{dt:?}: {e}");
    }
}

mod head_free {
    use super::super::loader::free_superseded_fp8_head;
    use metrale_gpu_runtime::gpu::GpuBackend;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};
    use std::collections::HashMap;

    fn t(gpu: &MockGpuBackend, dtype: WeightDtype, shape: Vec<usize>) -> WeightTensor {
        let n: usize = shape.iter().product::<usize>() * dtype.byte_size();
        WeightTensor { ptr: gpu.alloc(n).unwrap(), shape, dtype }
    }

    /// 2026-10-16 (race-lead: free before any serve, test the resident bytes): the
    /// FP4-Local F8 head (here [8, 64] + U8 [8, 2] = 528 B) leaves the store; the norm stays.
    #[test]
    fn fp8_head_and_scale_leave_the_store() {
        let gpu = MockGpuBackend::new();
        let mut m = HashMap::new();
        m.insert("lm_head.weight".to_string(), t(&gpu, WeightDtype::FP8E4M3, vec![8, 64]));
        m.insert("lm_head.weight_scale".to_string(), t(&gpu, WeightDtype::UInt8, vec![8, 2]));
        m.insert("model.language_model.norm.weight".to_string(), t(&gpu, WeightDtype::BF16, vec![64]));
        let mut store = WeightStore::from_map(m);
        let before = store.resident_bytes();
        assert_eq!(free_superseded_fp8_head(&mut store, &gpu).unwrap(), (2, 528));
        assert_eq!(before - store.resident_bytes(), 528);
        assert!(!store.contains("lm_head.weight") && !store.contains("lm_head.weight_scale"));
        assert!(store.contains("model.language_model.norm.weight"));
    }

    /// The official BF16 head is the tensor `load_lm_head` binds: it must stay.
    #[test]
    fn bf16_and_prepacked_heads_are_kept() {
        for dt in [WeightDtype::BF16, WeightDtype::UInt8] {
            let gpu = MockGpuBackend::new();
            let mut m = HashMap::new();
            m.insert("lm_head.weight".to_string(), t(&gpu, dt, vec![8, 64]));
            let mut store = WeightStore::from_map(m);
            let before = store.resident_bytes();
            assert_eq!(free_superseded_fp8_head(&mut store, &gpu).unwrap(), (0, 0), "{dt:?}");
            assert_eq!(store.resident_bytes(), before);
            assert!(store.contains("lm_head.weight"));
        }
    }
}
