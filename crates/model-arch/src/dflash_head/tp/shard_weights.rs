// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Device-side weight slicing of the two-rank DFlash drafter: the row-parallel
//! FP8 o/down slices and this rank's rows of a shared FP8 lm_head.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - A row-parallel slice keeps the full row's FP8 scale: the full `[n, k]` BF16 weight is
//!   quantized first (one f32 scale per row, as on one rank), then columns
//!   `[c0, c0 + len)` are copied out. The ranks' partial sums therefore use exactly the
//!   single-rank FP8 values and scales.
//! - A column-parallel slice quantizes its own rows; per-row scales make that identical to
//!   the same rows of the full quantization.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight};

use super::TpPlan;
use super::shard::Split;

/// 2026-10-09: FP8 copy of columns `[c0, c0 + len)` of the BF16 `w[n, k_full]`, with the
/// full rows' scales. The full FP8 temporary is freed before returning.
pub fn quantize_cols(
    gpu: &dyn GpuBackend,
    quant_k: KernelHandle,
    w: &DenseWeight,
    n: usize,
    k_full: usize,
    (c0, len): (usize, usize),
) -> Result<Fp8DenseWeight> {
    anyhow::ensure!(
        c0 + len <= k_full && len > 0,
        "quantize_cols: [{c0}, +{len}) of {k_full}"
    );
    let full = w.quantize_to_fp8(gpu, quant_k, n, k_full, 0)?;
    let stream = gpu.default_stream();
    let weight = gpu.alloc(n * len)?;
    let row_scale = gpu.alloc(n * 4)?;
    // 2026-10-09: FP8 E4M3 is one byte per element, so pitches are element counts.
    gpu.copy_d2d_2d_async(full.weight.offset(c0), k_full, weight, len, len, n, stream)?;
    gpu.copy_d2d_async(full.row_scale, row_scale, n * 4, stream)?;
    gpu.synchronize(stream)?;
    gpu.free(full.weight)?;
    gpu.free(full.row_scale)?;
    Ok(Fp8DenseWeight { weight, row_scale })
}

impl TpPlan {
    /// 2026-10-09: This rank's rows of a per-row-scaled FP8 lm_head `[vocab, hidden]`: a view
    /// into the shared buffers (no copy).
    pub fn lm_head_fp8_rows(&self, full: &Fp8DenseWeight) -> Fp8DenseWeight {
        let p = self.part("lm_head");
        Fp8DenseWeight {
            weight: full.weight.offset(p.start * p.k),
            row_scale: full.row_scale.offset(p.start * 4),
        }
    }

    /// 2026-10-09: `(k_full, (col0, cols))` of a row-parallel weight (`o_proj`, `down_proj`).
    fn cols(&self, name: &str) -> (usize, (usize, usize)) {
        let p = self.part(name);
        debug_assert_eq!(p.split, Split::Cols);
        (p.k, (p.start, p.len))
    }

    /// 2026-10-09: The o_proj slice: columns of this rank's q heads.
    pub fn o_cols(&self) -> (usize, (usize, usize)) {
        self.cols("o_proj")
    }

    /// 2026-10-09: The down_proj slice: columns of this rank's intermediate units.
    pub fn down_cols(&self) -> (usize, (usize, usize)) {
        self.cols("down_proj")
    }
}
