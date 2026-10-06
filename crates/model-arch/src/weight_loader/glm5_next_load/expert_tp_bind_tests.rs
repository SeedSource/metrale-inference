// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Tests of the expert-TP bind (`expert_tp_bind::bind_expert_tp`) on a mock device:
//! each rank's uploaded bytes are exactly its slice of the whole expert, read from deferred
//! shards; a full-width BF16 expert is quantised whole and then sliced; a resident expert is
//! refused; and with the lever off `bind_expert_cfg` binds exactly what `bind_expert` binds.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use std::collections::HashMap;

use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

use super::{bind_expert, bind_expert_cfg, is_routed_expert_tensor};
use crate::glm5next_mlp::Glm5NextMlpConfig;
use crate::glm5next_mlp::expert_tp::{Cut, Nvfp4Host, slice_nvfp4};
use crate::glm5next_mlp::weights::Nvfp4Proj;

const LAYER: usize = 3;
/// 2026-10-05: hidden 32, full I 64, so each rank's slice is 32 wide and down_proj's 64 columns
/// halve to two whole 16-element scale groups.
const HIDDEN: usize = 32;
const FULL_I: usize = 64;

fn qualified(leaf: &str) -> String {
    format!("model.language_model.layers.{LAYER}.{leaf}")
}

fn cfg(rank: usize, expert_tp: bool) -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: HIDDEN,
        local_dense_intermediate: 64,
        moe_intermediate: if expert_tp { FULL_I / 2 } else { FULL_I },
        local_shared_intermediate: 32,
        num_experts: 2,
        local_experts: if expert_tp { 2 } else { 1 },
        ep_rank: rank,
        top_k: 1,
        routed_scale: 1.0,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 2,
        ep_world_size: 2,
    }
}

fn bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

/// 2026-10-05: `[rows, cols]` of each projection of one expert.
fn shape_of(p: &str) -> (usize, usize) {
    match Cut::of(p) {
        Cut::Rows => (FULL_I, HIDDEN),
        Cut::Cols => (HIDDEN, FULL_I),
    }
}

/// 2026-10-05: A scratch shard holding `blobs` back to back after 64 filler bytes; returns the
/// path and each blob's absolute offset.
fn stage(tag: &str, blobs: &[&[u8]]) -> (std::path::PathBuf, Vec<u64>) {
    let dir = std::env::temp_dir().join(format!("metrale-glm5next-etp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{tag}.safetensors"));
    let mut blob = vec![0xAAu8; 64];
    let mut offs = Vec::new();
    for b in blobs {
        offs.push(blob.len() as u64);
        blob.extend_from_slice(b);
    }
    std::fs::write(&path, &blob).unwrap();
    (path, offs)
}

fn defer(
    store: &mut WeightStore,
    name: String,
    path: &std::path::Path,
    offset: u64,
    shape: Vec<usize>,
    dtype: WeightDtype,
) {
    store.defer(
        name,
        DeferredTensor {
            path: path.to_path_buf(),
            offset,
            shape,
            dtype,
        },
    );
}

fn read(gpu: &MockGpuBackend, p: &Nvfp4Proj) -> (Vec<u8>, Vec<u8>) {
    (
        gpu.read_alloc(p.packed).unwrap(),
        gpu.read_alloc(p.scale).unwrap(),
    )
}

/// 2026-10-05: A packed U8 expert, every tensor deferred: each rank's device bytes are its
/// `slice_nvfp4` of the whole projection (gate/up by rows, down by columns), `scale_2` is the
/// whole tensor's on both ranks, and only the slices were adopted.
#[test]
fn expert_tp_bind_uploads_exactly_this_ranks_slice_of_a_packed_expert() {
    let mut full = HashMap::new();
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    for (k, p) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
        let (rows, cols) = shape_of(p);
        let s2 = 0.01 * (k + 1) as f32;
        let h = Nvfp4Host {
            packed: bytes(3 + k as u64, rows * cols / 2),
            scale: bytes(30 + k as u64, rows * cols / 16),
            scale_2: s2,
            input_scale: 0.0,
        };
        blobs.push(h.packed.clone());
        blobs.push(h.scale.clone());
        blobs.push(s2.to_le_bytes().to_vec());
        full.insert(*p, h);
    }
    let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
    let (path, offs) = stage("packed", &refs);

    for rank in 0..2 {
        let mut store = WeightStore::from_map(HashMap::new());
        for (k, p) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let (rows, cols) = shape_of(p);
            let base = format!("mlp.experts.1.{p}");
            let o = &offs[3 * k..3 * k + 3];
            for (leaf, off, shape, dtype) in [
                ("weight", o[0], vec![rows, cols / 2], WeightDtype::UInt8),
                ("weight_scale", o[1], vec![rows, cols / 16], WeightDtype::FP8E4M3),
                ("weight_scale_2", o[2], vec![], WeightDtype::FP32),
            ] {
                let name = qualified(&format!("{base}.{leaf}"));
                defer(&mut store, name, &path, off, shape, dtype);
            }
        }
        let gpu = MockGpuBackend::new();
        let e = bind_expert_cfg(&gpu, &store, LAYER, 1, &cfg(rank, true)).unwrap();
        for (p, got) in [
            ("gate_proj", &e.gate_proj),
            ("up_proj", &e.up_proj),
            ("down_proj", &e.down_proj),
        ] {
            let (rows, cols) = shape_of(p);
            let want = slice_nvfp4(&full[p], rows, cols, Cut::of(p), rank, 2).unwrap();
            let (packed, scale) = read(&gpu, got);
            assert_eq!(packed, want.packed, "rank {rank} {p}: packed");
            assert_eq!(scale, want.scale, "rank {rank} {p}: block scales");
            assert_eq!(
                got.scale_2, full[p].scale_2,
                "rank {rank} {p}: scale_2 must stay whole"
            );
        }
        // 2026-10-05: Half of each projection's packed and scale bytes, nothing else.
        let want_bytes: usize = ["gate_proj", "up_proj", "down_proj"]
            .iter()
            .map(|p| {
                let (r, c) = shape_of(p);
                (r * c / 2 + r * c / 16) / 2
            })
            .sum();
        assert_eq!(store.derived().len(), 6);
        assert_eq!(store.derived().bytes(), want_bytes);
    }
    let _ = std::fs::remove_file(&path);
}

/// 2026-10-05: A full-width BF16 expert (the official export's MTP experts) is quantised whole,
/// exactly as `bind_expert` quantises it, and the NVFP4 result is sliced, so both ranks share
/// that `scale_2`.
#[test]
fn expert_tp_bind_quantises_a_bf16_expert_whole_then_slices_it() {
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    for k in 0..3usize {
        let n = FULL_I * HIDDEN;
        let v: Vec<u8> = (0..n)
            .flat_map(|i| {
                let x = ((i * (7 + k)) % 97) as f32 * 0.03 - 1.4;
                half::bf16::from_f32(x).to_le_bytes()
            })
            .collect();
        blobs.push(v);
    }
    let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
    let (path, offs) = stage("bf16", &refs);
    let mut store = WeightStore::from_map(HashMap::new());
    for (k, p) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
        let (rows, cols) = shape_of(p);
        let name = qualified(&format!("mlp.experts.0.{p}.weight"));
        defer(&mut store, name, &path, offs[k], vec![rows, cols], WeightDtype::BF16);
    }

    let gpu_full = MockGpuBackend::new();
    let whole = bind_expert(&gpu_full, &store, LAYER, 0).unwrap();
    for rank in 0..2 {
        let gpu = MockGpuBackend::new();
        let e = bind_expert_cfg(&gpu, &store, LAYER, 0, &cfg(rank, true)).unwrap();
        for (p, w, got) in [
            ("gate_proj", &whole.gate_proj, &e.gate_proj),
            ("up_proj", &whole.up_proj, &e.up_proj),
            ("down_proj", &whole.down_proj, &e.down_proj),
        ] {
            let (rows, cols) = shape_of(p);
            let (wp, ws) = read(&gpu_full, w);
            let full = Nvfp4Host {
                packed: wp,
                scale: ws,
                scale_2: w.scale_2,
                input_scale: 0.0,
            };
            let want = slice_nvfp4(&full, rows, cols, Cut::of(p), rank, 2).unwrap();
            let (packed, scale) = read(&gpu, got);
            assert_eq!(packed, want.packed, "rank {rank} {p}: packed");
            assert_eq!(scale, want.scale, "rank {rank} {p}: block scales");
            assert_eq!(
                got.scale_2, w.scale_2,
                "rank {rank} {p}: one scale_2 for both ranks"
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// 2026-10-05: Under expert-TP a resident (not deferred) expert is refused rather than sliced,
/// and a deferred one of the wrong width is refused by name.
#[test]
fn expert_tp_bind_refuses_resident_and_misshapen_experts() {
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(FULL_I * HIDDEN / 2).unwrap();
    let mut map = HashMap::new();
    map.insert(
        qualified("mlp.experts.0.gate_proj.weight"),
        WeightTensor {
            ptr,
            shape: vec![FULL_I, HIDDEN / 2],
            dtype: WeightDtype::UInt8,
        },
    );
    let err = bind_expert_cfg(&gpu, &WeightStore::from_map(map), LAYER, 0, &cfg(0, true))
        .unwrap_err()
        .to_string();
    assert!(err.contains("not deferred"), "{err}");

    let zeros = vec![0u8; 16 * HIDDEN];
    let (path, offs) = stage("narrow", &[zeros.as_slice()]);
    let mut store = WeightStore::from_map(HashMap::new());
    let name = qualified("mlp.experts.0.gate_proj.weight");
    defer(&mut store, name, &path, offs[0], vec![32, HIDDEN / 2], WeightDtype::UInt8);
    let err = bind_expert_cfg(&gpu, &store, LAYER, 0, &cfg(0, true))
        .unwrap_err()
        .to_string();
    assert!(err.contains("gate_proj.weight"), "{err}");
    let _ = std::fs::remove_file(&path);
}

/// 2026-10-05: Lever off (an EP2 config), `bind_expert_cfg` is `bind_expert`: the same
/// zero-copy pointers and scalars for a resident packed expert.
#[test]
fn expert_tp_off_binds_exactly_what_bind_expert_binds() {
    let gpu = MockGpuBackend::new();
    let mut map = HashMap::new();
    for p in ["gate_proj", "up_proj", "down_proj"] {
        let (rows, cols) = shape_of(p);
        let base = format!("mlp.experts.0.{p}");
        let w = gpu.alloc(rows * cols / 2).unwrap();
        let s = gpu.alloc(rows * cols / 16).unwrap();
        let s2 = gpu.alloc(4).unwrap();
        gpu.copy_h2d(&0.5f32.to_le_bytes(), s2).unwrap();
        for (leaf, ptr, shape, dtype) in [
            ("weight", w, vec![rows, cols / 2], WeightDtype::UInt8),
            ("weight_scale", s, vec![rows, cols / 16], WeightDtype::FP8E4M3),
            ("weight_scale_2", s2, vec![], WeightDtype::FP32),
        ] {
            let t = WeightTensor { ptr, shape, dtype };
            map.insert(qualified(&format!("{base}.{leaf}")), t);
        }
    }
    let store = WeightStore::from_map(map);
    let c = cfg(0, false);
    assert!(!c.is_expert_tp());
    let a = bind_expert(&gpu, &store, LAYER, 0).unwrap();
    let b = bind_expert_cfg(&gpu, &store, LAYER, 0, &c).unwrap();
    for (x, y) in [
        (&a.gate_proj, &b.gate_proj),
        (&a.up_proj, &b.up_proj),
        (&a.down_proj, &b.down_proj),
    ] {
        assert_eq!(x.packed, y.packed);
        assert_eq!(x.scale, y.scale);
        assert_eq!(x.scale_2, y.scale_2);
        assert_eq!(x.input_scale, y.input_scale);
    }
    assert_eq!(store.derived().len(), 0, "the EP arm uploads nothing");
}

/// 2026-10-05: The expert-TP defer family is every per-expert tensor of every layer, MTP
/// included, and nothing else.
#[test]
fn expert_tp_defers_every_routed_expert_tensor_only() {
    for leaf in [
        "mlp.experts.0.gate_proj.weight",
        "mlp.experts.287.down_proj.weight_scale",
        "mlp.experts.5.up_proj.weight_scale_2",
        "mlp.experts.5.up_proj.input_scale",
    ] {
        for layer in [0usize, 44, 45] {
            let n = format!("model.language_model.layers.{layer}.{leaf}");
            assert!(is_routed_expert_tensor(&n), "{n}");
        }
    }
    for n in [
        "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.language_model.layers.3.mlp.gate.weight",
        "model.language_model.layers.3.mlp.gate.e_score_correction_bias",
        "model.language_model.layers.3.mlp.experts.gate_up_proj",
        "model.language_model.layers.3.mlp.gate_proj.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    ] {
        assert!(!is_routed_expert_tensor(n), "{n}");
    }
}
