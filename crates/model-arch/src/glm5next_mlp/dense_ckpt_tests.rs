// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `METRALE_GLM_DENSE_MLP_CKPT` on the host: the dense-MLP NVFP4 decode copy made
//! from checkpoint bytes (`dense_ckpt::rank_copy` / `for_rank`, `dense_fp8::upload_ckpt_copy`)
//! decodes, through the GEMV's own weight arithmetic (`dense_ckpt::gemv_weights_f32`, from
//! `w4a16_gemv.cu`), to exactly the checkpoint's weights, whole and per TP rank; and with the
//! lever unset the copy-source selection is today's.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::dense_ckpt::{PackedTensor, check_block_scales, for_rank, gemv_weights_f32, rank_copy};
use super::expert_tp::{Cut, Nvfp4Host};
use crate::glm5next_layer::dense_fp8::{Nv4Source, nv4_source, parse_mlp_ckpt, upload_ckpt_copy};

fn lcg(seed: u64, n: usize) -> Vec<u8> {
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

/// 2026-10-16: A `[rows, cols]` checkpoint projection: random codes in both nibbles, with the
/// first bytes forced to negative zero (0x88), the max code (0x77, 0xFF) and mixed pairs;
/// scales random finite non-negative E4M3 with 0x00 (zero), 0x01 (subnormal) and 0x7E (448)
/// forced in.
fn ckpt(rows: usize, cols: usize, seed: u64, scale_2: f32) -> Nvfp4Host {
    let mut packed = lcg(seed, rows * cols / 2);
    for (i, b) in [0x88u8, 0x77, 0xFF, 0x80, 0x08, 0x7F, 0xF7, 0x00]
        .iter()
        .enumerate()
    {
        packed[i] = *b;
    }
    let mut scale: Vec<u8> = lcg(seed ^ 0x5ca1e, rows * cols / 16)
        .into_iter()
        .map(|b| b % 0x7F)
        .collect();
    scale[0] = 0x38;
    scale[1] = 0x00;
    scale[2] = 0x01;
    scale[3] = 0x7E;
    Nvfp4Host {
        packed,
        scale,
        scale_2,
        input_scale: 0.75,
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn row_slice(v: &[f32], cols: usize, lo: usize, hi: usize) -> Vec<f32> {
    v[lo * cols..hi * cols].to_vec()
}

fn col_slice(v: &[f32], cols: usize, lo: usize, hi: usize) -> Vec<f32> {
    v.chunks(cols)
        .flat_map(|r| r[lo..hi].iter().copied())
        .collect()
}

/// 2026-10-16: (i) The whole-tensor copy is the checkpoint's bytes and `scale_2`, and decodes
/// bit-exactly to the checkpoint's weights, -0.0 included, for several `scale_2`. Against the
/// modelopt product `E2M1 * e4m3 * scale_2` (exact in f64) the GEMV decode is within one f32
/// ulp, and exact on codes whose E2M1 value is 0 or a power of two.
#[test]
fn ckpt_copy_decodes_bit_exactly_to_the_checkpoint() {
    let (rows, cols) = (12, 64);
    for (seed, s2) in [
        (1u64, 0.0123f32),
        (2, 1.0 / (6.0 * 448.0)),
        (3, 3.7e-5),
        (4, 1.0),
    ] {
        let full = ckpt(rows, cols, seed, s2);
        let copy = rank_copy("t", &full, rows, cols, Cut::Rows, 0, 1).unwrap();
        assert_eq!(copy.packed, full.packed);
        assert_eq!(copy.scale, full.scale);
        assert_eq!(copy.scale_2.to_bits(), full.scale_2.to_bits());
        assert_eq!(copy.input_scale, 0.0, "input_scale is dropped");
        let got = gemv_weights_f32(&copy, rows, cols);
        assert_eq!(bits(&got), bits(&gemv_weights_f32(&full, rows, cols)));
        assert_eq!(
            got[0].to_bits(),
            (-0.0f32).to_bits(),
            "0x88 low nibble is -0.0"
        );
        assert_eq!(
            got[1].to_bits(),
            (-0.0f32).to_bits(),
            "0x88 high nibble is -0.0"
        );

        let e2m1 = metrale_cache::kv_dequant::NVFP4_E2M1_LUT;
        let e4m3 = metrale_cache::kv_dequant::e4m3_lut();
        let (mut exact, mut ulp1) = (0, 0);
        for r in 0..rows {
            for c in 0..cols {
                let b = full.packed[r * cols / 2 + c / 2];
                let nib = if c % 2 == 0 { b & 0xF } else { b >> 4 } as usize;
                let s = e4m3[full.scale[r * cols / 16 + c / 16] as usize] as f64;
                let want = (e2m1[nib] as f64 * s * s2 as f64) as f32;
                let g = got[r * cols + c];
                if g.to_bits() == want.to_bits() {
                    exact += 1;
                    continue;
                }
                let pow2 = matches!(nib & 7, 0 | 1 | 2 | 4 | 6);
                assert!(
                    !pow2,
                    "[{r},{c}] code {nib}: a power-of-two code must be exact"
                );
                let ulp = f32::from_bits(want.abs().to_bits() + 1) - want.abs();
                assert!(
                    (g - want).abs() <= ulp,
                    "[{r},{c}] code {nib}: {g} vs {want}"
                );
                ulp1 += 1;
            }
        }
        assert_eq!(exact + ulp1, rows * cols);
    }
}

/// 2026-10-16: (ii) Per TP rank, gate/up (`[I, H]`, rows) and down (`[H, I]`, columns on whole
/// scale groups) slices decode to exactly the same slice of the decoded whole matrix, as
/// `build::build_dense_mlp` slices the BF16 dequant (`lo = rank * I / tp`).
#[test]
fn ckpt_rank_copies_equal_slices_of_the_decoded_matrix() {
    let (hidden, inter, tp) = (64, 96, 2);
    let gate = ckpt(inter, hidden, 11, 0.02);
    let up = ckpt(inter, hidden, 12, 0.03);
    let down = ckpt(hidden, inter, 13, 0.004);
    let tensors = [
        ("mlp.gate_proj.weight", &gate, inter, hidden),
        ("mlp.up_proj.weight", &up, inter, hidden),
        ("mlp.down_proj.weight", &down, hidden, inter),
    ];
    let get = |n: &str| -> anyhow::Result<Option<PackedTensor>> {
        let (_, h, rows, cols) = tensors.iter().find(|t| t.0 == n).expect("known name");
        Ok(Some(PackedTensor {
            host: (*h).clone(),
            rows: *rows,
            cols: *cols,
        }))
    };
    let local = inter / tp;
    for rank in 0..tp {
        let lo = rank * local;
        let c = for_rank(hidden, inter, rank, tp, "mlp", &get)
            .unwrap()
            .expect("packed");
        let fg = gemv_weights_f32(&gate, inter, hidden);
        let fu = gemv_weights_f32(&up, inter, hidden);
        let fd = gemv_weights_f32(&down, hidden, inter);
        assert_eq!(
            bits(&gemv_weights_f32(&c.gate_proj, local, hidden)),
            bits(&row_slice(&fg, hidden, lo, lo + local)),
            "rank {rank} gate"
        );
        assert_eq!(
            bits(&gemv_weights_f32(&c.up_proj, local, hidden)),
            bits(&row_slice(&fu, hidden, lo, lo + local)),
            "rank {rank} up"
        );
        assert_eq!(
            bits(&gemv_weights_f32(&c.down_proj, hidden, local)),
            bits(&col_slice(&fd, inter, lo, lo + local)),
            "rank {rank} down"
        );
        assert_eq!(c.down_proj.scale_2.to_bits(), down.scale_2.to_bits());
    }
}

/// 2026-10-16: The checks: a down_proj row split off 16-element scale groups (I / tp = 40), a
/// mix of packed and BF16 projections, a wrong shape and a NaN / negative scale byte are
/// errors; an all-BF16 MLP is `None` (the lever keeps today's path).
#[test]
fn ckpt_copies_refuse_what_they_cannot_slice_exactly() {
    let (hidden, inter) = (64, 80);
    let t = |rows: usize, cols: usize, s: u64| PackedTensor {
        host: ckpt(rows, cols, s, 0.01),
        rows,
        cols,
    };
    let all = |n: &str| -> anyhow::Result<Option<PackedTensor>> {
        Ok(Some(if n.contains("down") {
            t(hidden, inter, 3)
        } else {
            t(inter, hidden, 4)
        }))
    };
    let err = for_rank(hidden, inter, 0, 2, "mlp", &all).unwrap_err();
    assert!(
        format!("{err:#}").contains("16-element scale groups"),
        "{err:#}"
    );
    assert!(
        for_rank(hidden, inter, 0, 1, "mlp", &all)
            .unwrap()
            .is_some(),
        "tp 1 needs no split"
    );

    let mixed = |n: &str| -> anyhow::Result<Option<PackedTensor>> {
        Ok(if n.contains("gate") { None } else { all(n)? })
    };
    assert!(for_rank(hidden, inter, 0, 1, "mlp", &mixed).is_err());
    let wrong = |_: &str| -> anyhow::Result<Option<PackedTensor>> { Ok(Some(t(inter, hidden, 5))) };
    assert!(
        for_rank(hidden, inter, 0, 1, "mlp", &wrong).is_err(),
        "down_proj is [H, I]"
    );
    let none = |_: &str| -> anyhow::Result<Option<PackedTensor>> { Ok(None) };
    assert_eq!(for_rank(hidden, inter, 0, 2, "mlp", &none).unwrap(), None);

    assert!(check_block_scales("s", &[0x00, 0x7E]).is_ok());
    assert!(
        check_block_scales("s", &[0x38, 0x7F]).is_err(),
        "0x7F is NaN"
    );
    assert!(check_block_scales("s", &[0x80]).is_err(), "negative");
    let mut bad = ckpt(16, 32, 6, 0.01);
    bad.scale[5] = 0xB8;
    assert!(rank_copy("t", &bad, 16, 32, Cut::Rows, 0, 1).is_err());
}

/// 2026-10-16: `upload_ckpt_copy` puts the slice's bytes on the device unchanged, with the
/// checkpoint `scale_2` and no input scale, and refuses a byte count off `[n, k]`.
#[test]
fn ckpt_upload_copies_the_bytes_as_they_are() {
    let gpu = MockGpuBackend::new();
    let (n, k) = (8, 48);
    let h = ckpt(n, k, 21, 0.0071);
    let q = upload_ckpt_copy(&gpu, &h, n, k).unwrap();
    let mut packed = vec![0u8; n * k / 2];
    gpu.copy_d2h(q.weight, &mut packed).unwrap();
    let mut scale = vec![0u8; n * k / 16];
    gpu.copy_d2h(q.weight_scale, &mut scale).unwrap();
    assert_eq!(packed, h.packed);
    assert_eq!(scale, h.scale);
    assert_eq!(q.weight_scale_2.to_bits(), h.scale_2.to_bits());
    assert!(q.input_scale.is_null() && q.weight_scale_2_vec.is_null());
    assert!(upload_ckpt_copy(&gpu, &h, n, k + 16).is_err());
}

/// 2026-10-16: (iii) Lever unset (or anything but `1`) is off, and with no checkpoint slice the
/// copy source is today's: `METRALE_GLM_DENSE_NVFP4` listing the class re-quantizes, else no
/// copy. A checkpoint slice (lever on, packed MLP) wins either way.
#[test]
fn ckpt_lever_off_keeps_todays_copy_selection() {
    for raw in [
        None,
        Some(""),
        Some("0"),
        Some("off"),
        Some("true"),
        Some("2"),
    ] {
        assert!(!parse_mlp_ckpt(raw), "{raw:?}");
    }
    assert!(parse_mlp_ckpt(Some("1")));
    assert!(parse_mlp_ckpt(Some(" 1 ")));
    assert_eq!(nv4_source(true, false), Nv4Source::Requant);
    assert_eq!(nv4_source(false, false), Nv4Source::None);
    assert_eq!(nv4_source(true, true), Nv4Source::Ckpt);
    assert_eq!(nv4_source(false, true), Nv4Source::Ckpt);
}
