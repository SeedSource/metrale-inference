// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `METRALE_GLM_DENSE_MLP_CKPT`: this rank's NVFP4 decode copy of a dense MLP
//! (layers 0-2) taken from the checkpoint's own packed NVFP4 bytes, on the host.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The layout is the checkpoint's (ModelOpt NVFP4) and the one `w4a16_gemv.cu` reads (its
//!   header, lines 11-14) and `quantize_to_nvfp4` writes: packed `[rows, cols / 2]` (low
//!   nibble = even column), E4M3 block scales `[rows, cols / 16]`, both row-major and
//!   unswizzled, one FP32 `scale_2`. The copy is therefore the checkpoint's bytes, sliced; no
//!   nibble or scale transform.
//! - TP as `build::build_dense_mlp` slices the BF16 dequant: gate_proj/up_proj (`[I, H]`) by
//!   rows, down_proj (`[H, I]`) by columns on whole 16-element scale groups
//!   (`expert_tp::slice_nvfp4`, which refuses any other split). `scale_2` is kept whole.
//! - A block-scale byte must be finite and non-negative E4M3 (`0x00..=0x7E`), the loader's
//!   dequant rule (`nvfp4_dequant.rs`); `input_scale` is dropped (0.0).

use anyhow::{Context, Result, bail};

use super::expert_tp::{Cut, Nvfp4Host, slice_nvfp4};

/// 2026-10-16: Host model of the weight `w4a16_gemv` applies to element `[r, c]` of a
/// `[rows, cols]` NVFP4 copy, in its arithmetic (`kernels/gb10/common/w4a16_gemv.cu`
/// `w4a16_gemv_partial`, lines 142-158): `scale = (float)e4m3(byte) * scale2` first, then
/// `E2M1_LUT[nibble] * scale`; nibble of byte `c / 2`, low for even `c`. Row-major `f32`.
/// (The loader's dequant multiplies `(E2M1 * e4m3) * scale_2`; for codes +-1.5/3/6 the two
/// orders can differ by 1 f32 ulp.)
pub fn gemv_weights_f32(h: &Nvfp4Host, rows: usize, cols: usize) -> Vec<f32> {
    use metrale_cache::kv_dequant::{NVFP4_E2M1_LUT, e4m3_lut};
    let e4m3 = e4m3_lut();
    let (half, groups) = (cols / 2, cols / 16);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let byte = h.packed[r * half + c / 2];
            let nib = if c.is_multiple_of(2) {
                byte & 0xF
            } else {
                byte >> 4
            };
            let scale = e4m3[h.scale[r * groups + c / 16] as usize] * h.scale_2;
            out[r * cols + c] = NVFP4_E2M1_LUT[nib as usize] * scale;
        }
    }
    out
}

/// 2026-10-16: One rank's checkpoint NVFP4 slices of a dense MLP's three projections.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMlpCkpt {
    /// `[I / tp, H]`.
    pub gate_proj: Nvfp4Host,
    /// `[I / tp, H]`.
    pub up_proj: Nvfp4Host,
    /// `[H, I / tp]`.
    pub down_proj: Nvfp4Host,
}

/// 2026-10-16: A packed NVFP4 checkpoint tensor on the host: its bytes and logical
/// `[rows, cols]` (`cols` = 2 x the stored U8 width).
#[derive(Debug, Clone, PartialEq)]
pub struct PackedTensor {
    pub host: Nvfp4Host,
    pub rows: usize,
    pub cols: usize,
}

/// 2026-10-16: An error on a block-scale byte that is not a finite non-negative E4M3 value.
pub fn check_block_scales(what: &str, scale: &[u8]) -> Result<()> {
    if let Some((i, b)) = scale.iter().enumerate().find(|&(_, &b)| b >= 0x7F) {
        bail!(
            "{what}: block scale byte 0x{b:02X} at {i} is not a finite non-negative E4M3 value \
             (0x7F is NaN, 0x80.. are negative)"
        );
    }
    Ok(())
}

/// 2026-10-16: Rank `rank` of `world`'s NVFP4 decode copy of the `[rows, cols]` checkpoint
/// projection `full`, cut by `cut`: the checkpoint's bytes for that slice, `scale_2` whole,
/// `input_scale` 0.0.
pub fn rank_copy(
    what: &str,
    full: &Nvfp4Host,
    rows: usize,
    cols: usize,
    cut: Cut,
    rank: usize,
    world: usize,
) -> Result<Nvfp4Host> {
    check_block_scales(what, &full.scale)?;
    if !full.scale_2.is_finite() || full.scale_2 <= 0.0 {
        bail!(
            "{what}: weight_scale_2 {} is not a finite positive scale",
            full.scale_2
        );
    }
    let mut h =
        slice_nvfp4(full, rows, cols, cut, rank, world).with_context(|| what.to_string())?;
    h.input_scale = 0.0;
    Ok(h)
}

/// 2026-10-16: This rank's [`DenseMlpCkpt`] from `get` (layer-relative `.weight` name to the
/// packed tensor, `None` when it is not packed NVFP4): `Ok(None)` when no projection is
/// packed (a BF16 checkpoint), an error when only some are or a shape is not
/// `[full_inter, hidden]` / `[hidden, full_inter]`.
pub fn for_rank(
    hidden: usize,
    full_inter: usize,
    tp_rank: usize,
    tp: usize,
    prefix: &str,
    get: &dyn Fn(&str) -> Result<Option<PackedTensor>>,
) -> Result<Option<DenseMlpCkpt>> {
    let names = ["gate_proj", "up_proj", "down_proj"];
    let mut got = Vec::with_capacity(3);
    for p in names {
        got.push(get(&format!("{prefix}.{p}.weight"))?);
    }
    let packed = got.iter().filter(|t| t.is_some()).count();
    if packed == 0 {
        return Ok(None);
    }
    if packed != 3 {
        bail!("{prefix}: {packed} of gate/up/down_proj are packed NVFP4; expected all or none");
    }
    let mut out = Vec::with_capacity(3);
    for (p, t) in names.iter().zip(got) {
        let t = t.expect("counted above");
        let (want, cut) = if *p == "down_proj" {
            ((hidden, full_inter), Cut::Cols)
        } else {
            ((full_inter, hidden), Cut::Rows)
        };
        if (t.rows, t.cols) != want {
            bail!(
                "{prefix}.{p}.weight: packed NVFP4 is [{}, {}], expected {want:?}",
                t.rows,
                t.cols
            );
        }
        let what = format!("{prefix}.{p}");
        out.push(rank_copy(&what, &t.host, t.rows, t.cols, cut, tp_rank, tp)?);
    }
    let down_proj = out.pop().expect("three");
    let up_proj = out.pop().expect("three");
    let gate_proj = out.pop().expect("three");
    Ok(Some(DenseMlpCkpt {
        gate_proj,
        up_proj,
        down_proj,
    }))
}
