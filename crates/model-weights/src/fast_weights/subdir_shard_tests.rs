// SPDX-License-Identifier: MIT OR Apache-2.0

//! An indexed shard in a subdirectory loads its tensors.
//!
//! 🔴 nvidia/GLM-5.3-Flash-FP4-Local routes 214 BF16 tensors (layer 0's
//! `self_attn.f_a_proj.weight`, which sizes the KDA gate, among them) to
//! `tensors/native-QB.safetensors`. The shard filter compared that index value
//! with the bare file name, so the shard loaded nothing and the model build
//! failed on the first missing name (race-pf-fp4lm-L34, 2026-10-17).

use super::FastSafetensorsLoader;
use crate::weights::WeightLoader;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// Write one safetensors file of BF16 `(name, values)` entries at `path`.
fn write_file(path: &std::path::Path, entries: &[(&str, &[f32])]) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, v) in entries {
        let start = data.len();
        data.extend(
            v.iter()
                .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()),
        );
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": [v.len()],
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut blob = (header.len() as u64).to_le_bytes().to_vec();
    blob.extend_from_slice(&header);
    blob.extend_from_slice(&data);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, blob).unwrap();
}

/// A model dir whose index routes one tensor to a top-level shard and two to
/// `tensors/native-QB.safetensors`, the FP4-Local layout.
fn write_model(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("metrale-subdir-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_file(
        &dir.join("model-00001-of-00001.safetensors"),
        &[("model.layers.0.mlp.gate_proj.weight", &[1.0, 2.0])],
    );
    write_file(
        &dir.join("tensors/native-QB.safetensors"),
        &[
            ("model.layers.0.self_attn.f_a_proj.weight", &[3.0, 4.0, 5.0]),
            ("model.layers.0.self_attn.b_proj.weight", &[6.0]),
        ],
    );
    let map = serde_json::json!({
        "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00001.safetensors",
        "model.layers.0.self_attn.f_a_proj.weight": "tensors/native-QB.safetensors",
        "model.layers.0.self_attn.b_proj.weight": "tensors/native-QB.safetensors",
    });
    let index = serde_json::json!({ "metadata": {}, "weight_map": map });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    dir
}

#[test]
fn every_indexed_tensor_loads_including_the_subdirectory_shard() {
    let dir = write_model("ok");
    let gpu = MockGpuBackend::new();
    let store = FastSafetensorsLoader::new().load(&dir, &gpu, 0).unwrap();
    for name in [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.self_attn.f_a_proj.weight",
        "model.layers.0.self_attn.b_proj.weight",
    ] {
        assert!(store.contains(name), "{name} missing from the store");
    }
    assert_eq!(
        store
            .get("model.layers.0.self_attn.f_a_proj.weight")
            .unwrap()
            .shape,
        vec![3]
    );
    let _ = std::fs::remove_dir_all(&dir);
}
