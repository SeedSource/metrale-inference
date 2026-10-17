// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Store tensors bound as-is must be BF16; the final norm may arrive as F32.
//!
//! 🔴 nvidia/GLM-5.3-Flash-FP4-Local ships `model.language_model.norm.weight` as F32 [4096] (the
//! official NVFP4 checkpoint: BF16). `load_final_norm` bound the store buffer as BF16, so the
//! final norm read half the F32 bytes as BF16 pairs and every reply was EOS at token 1
//! (race-pf-fp4lm3-L34, GSM8K500 0/500).
//!
//! Owner: model-arch weight loader (GLM-5.3).

use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{Glm5NextWeightLoader, dense};
use crate::weight_loader::ModelWeightLoader;

const NORM: &str = "model.language_model.norm.weight";

fn store_with(
    gpu: &MockGpuBackend,
    name: &str,
    dtype: WeightDtype,
    bytes: &[u8],
    n: usize,
) -> WeightStore {
    let ptr = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(bytes, ptr).unwrap();
    let mut map = std::collections::HashMap::new();
    map.insert(
        name.to_string(),
        WeightTensor {
            ptr,
            shape: vec![n],
            dtype,
        },
    );
    WeightStore::from_map(map)
}

fn read_bf16(gpu: &MockGpuBackend, ptr: metrale_gpu_runtime::gpu::DevicePtr, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(ptr, &mut b).unwrap();
    b.chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn norm_values(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + i as f32 * 0.03125).collect()
}

#[test]
fn an_f32_final_norm_is_cast_to_bf16() {
    let gpu = MockGpuBackend::new();
    let v = norm_values(64);
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let store = store_with(&gpu, NORM, WeightDtype::FP32, &bytes, v.len());
    let w = Glm5NextWeightLoader
        .load_final_norm(&store, &ModelConfig::qwen3_next_80b_nvfp4(), &gpu)
        .unwrap();
    let got = read_bf16(&gpu, w.weight, v.len());
    let want: Vec<f32> = v
        .iter()
        .map(|x| half::bf16::from_f32(*x).to_f32())
        .collect();
    assert_eq!(
        got, want,
        "the bound final norm must be the F32 values cast to BF16"
    );
    assert_ne!(
        w.weight,
        store.get(NORM).unwrap().ptr,
        "the F32 buffer itself must not be bound"
    );
}

#[test]
fn a_bf16_final_norm_binds_the_store_buffer() {
    let gpu = MockGpuBackend::new();
    let v = norm_values(64);
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let store = store_with(&gpu, NORM, WeightDtype::BF16, &bytes, v.len());
    let w = Glm5NextWeightLoader
        .load_final_norm(&store, &ModelConfig::qwen3_next_80b_nvfp4(), &gpu)
        .unwrap();
    assert_eq!(
        w.weight,
        store.get(NORM).unwrap().ptr,
        "official layout: zero-copy, as before"
    );
}

#[test]
fn a_non_bf16_tensor_is_never_bound_as_is() {
    let gpu = MockGpuBackend::new();
    let store = store_with(
        &gpu,
        "model.language_model.embed_tokens.weight",
        WeightDtype::FP32,
        &[0u8; 16],
        4,
    );
    let err = dense(&store, "model.language_model.embed_tokens.weight")
        .err()
        .expect("F32 refused");
    assert!(
        format!("{err:#}").contains("would be bound as BF16"),
        "{err:#}"
    );
    let store = store_with(&gpu, NORM, WeightDtype::FP8E4M3, &[0u8; 4], 4);
    let err = Glm5NextWeightLoader
        .load_final_norm(&store, &ModelConfig::qwen3_next_80b_nvfp4(), &gpu)
        .err()
        .expect("FP8 refused");
    assert!(
        format!("{err:#}").contains("would be bound as BF16"),
        "{err:#}"
    );
}
