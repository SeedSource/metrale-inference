// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Scaled-FP8 weights (nvidia/GLM-5.3-Flash-FP4-Local) dequantized
//! to BF16 on the host, so every binder reads them like the official
//! checkpoint's BF16.
//!
//! FP4-Local stores its attention, dense-MLP and shared-expert weights as
//! `F8_E4M3` with a `.weight_scale` sibling in one of two layouts (shapes read
//! from the rev 76348ddb safetensors headers):
//! - MXFP8: `U8` E8M0 exponents, one per 32 elements along K, shape `[N, K/32]`
//!   (KDA q/k/v/o, dense MLP layers 0-2, the MTP layer's attention, indexer and
//!   shared expert). value = e4m3 * 2^(e - 127). Exact in BF16: the result keeps
//!   E4M3's 3 mantissa bits and BF16 has f32's exponent range.
//! - FP8 block 128x128, weight only: `F32` scale, shape `[N/128, 1, K/128, 1]`
//!   (DSA q_a/q_b/kv_a/o and shared experts of layers 3-44). value = e4m3 * s,
//!   rounded to BF16 (nearest even): at most 2^-9 relative.
//!
//! The scale sibling is dropped from the layer, so a binder that lists the
//! layer's names (the KDA binder refuses unknown ones) never sees it. Any other
//! FP8 weight, or a scale whose shape does not match its weight, is an error:
//! nothing is guessed.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants: runs before the plan-dtype cast, so a dequantized tensor is
//! cast like any BF16 one.

use anyhow::{Result, bail};
use metrale_cache::kv_dequant::e4m3_lut;
use metrale_model_weights::weights::WeightDtype;
use std::collections::BTreeMap;

/// A layer's host tensors under layer-relative names: (dtype, shape, bytes).
pub(super) type HostTensors = BTreeMap<String, (WeightDtype, Vec<usize>, Vec<u8>)>;

const MX_GROUP: usize = 32;
const BLOCK: usize = 128;

/// 2026-10-16: Replace every `F8_E4M3` `.weight` that has a `.weight_scale`
/// sibling by its BF16 dequantization, and remove that sibling from `tensors`
/// and `names`. Returns the number of weights converted.
pub(super) fn dequant_scaled_fp8(tensors: &mut HostTensors, names: &mut Vec<String>) -> Result<usize> {
    let fp8: Vec<String> = tensors
        .iter()
        .filter(|(n, (dt, _, _))| *dt == WeightDtype::FP8E4M3 && n.ends_with(".weight"))
        .map(|(n, _)| n.clone())
        .collect();
    let mut dropped = Vec::new();
    for name in &fp8 {
        let base = &name[..name.len() - ".weight".len()];
        let scale_name = format!("{base}.weight_scale");
        let Some((sdt, sshape, sbytes)) = tensors.get(&scale_name) else {
            bail!("{name} is F8_E4M3 but {scale_name} is absent: unscaled FP8 weights are not a format this loader reads");
        };
        let (_, shape, bytes) = &tensors[name];
        let bf16 = dequant_one(name, shape, bytes, *sdt, sshape, sbytes)?;
        let shape = shape.clone();
        tensors.insert(name.clone(), (WeightDtype::BF16, shape, bf16));
        tensors.remove(&scale_name);
        dropped.push(scale_name);
    }
    names.retain(|n| !dropped.contains(n));
    Ok(fp8.len())
}

/// 2026-10-16: E8M0 exponent byte `e` (0..=254) as the exact f32 2^(e-127): the f32 bit
/// pattern for e >= 1, the one subnormal 2^-127 for e = 0. One per 32 elements, not per element.
fn e8m0(e: u8) -> f32 {
    if e == 0 { f32::from_bits(1 << 22) } else { f32::from_bits((e as u32) << 23) }
}

/// 2026-10-16: One scaled-FP8 weight `[N, K]` to little-endian BF16 bytes.
pub(super) fn dequant_one(
    name: &str,
    shape: &[usize],
    bytes: &[u8],
    scale_dtype: WeightDtype,
    scale_shape: &[usize],
    scale: &[u8],
) -> Result<Vec<u8>> {
    let [n, k] = shape[..] else {
        bail!("{name}: scaled FP8 weight must be 2-D, got {shape:?}");
    };
    if bytes.len() != n * k {
        bail!("{name}: {} B for a [{n}, {k}] F8_E4M3 weight", bytes.len());
    }
    let mut out = Vec::with_capacity(n * k * 2);
    let e4m3 = e4m3_lut();
    let bf16 = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    match scale_dtype {
        WeightDtype::UInt8 | WeightDtype::FP8E8M0 => {
            if !k.is_multiple_of(MX_GROUP) || scale_shape != [n, k / MX_GROUP] || scale.len() != n * (k / MX_GROUP) {
                bail!("{name}: MXFP8 scale shape {scale_shape:?} ({} B) does not match weight [{n}, {k}] / 32", scale.len());
            }
            let g = k / MX_GROUP;
            for r in 0..n {
                for gi in 0..g {
                    let e = scale[r * g + gi];
                    // E8M0: 255 is NaN by the OCP MX spec; refuse rather than carry it.
                    if e == 255 {
                        bail!("{name}: E8M0 scale NaN (0xFF) at row {r}, group {gi}");
                    }
                    let f = e8m0(e);
                    for &b in &bytes[r * k + gi * MX_GROUP..r * k + (gi + 1) * MX_GROUP] {
                        out.extend_from_slice(&bf16(e4m3[b as usize] * f));
                    }
                }
            }
        }
        WeightDtype::FP32 => {
            let (nb, kb) = (n.div_ceil(BLOCK), k.div_ceil(BLOCK));
            let ok_shape = scale_shape == [nb, 1, kb, 1] || scale_shape == [nb, kb];
            if !ok_shape || scale.len() != nb * kb * 4 {
                bail!("{name}: block-128 scale shape {scale_shape:?} ({} B) does not match weight [{n}, {k}]", scale.len());
            }
            let s: Vec<f32> = scale.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            for r in 0..n {
                for c in 0..k {
                    let v = e4m3[bytes[r * k + c] as usize] * s[(r / BLOCK) * kb + c / BLOCK];
                    out.extend_from_slice(&bf16(v));
                }
            }
        }
        other => bail!("{name}: scale dtype {other:?} is neither MXFP8 (U8 E8M0) nor block-128 (F32)"),
    }
    Ok(out)
}
