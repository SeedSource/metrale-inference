// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The DFlash drafter's FP8 weight copies: each layer's seven dense GEMM
//! weights and the lm_head.
//!
//! 2026-10-08: `arm_drafter_fp8` decides whether the head goes FP8 (moved here from
//! `from_weights.rs`) and, on targets whose `w4a16` module lacks the row-scaled FP8 GEMMs
//! (GLM-5.3), takes them from the target's `dflash_fp8_gemm` module under an explicit
//! METRALE_DFLASH_DRAFTER_FP8=1 (SeedHQ/seed-skills#71 S1b).
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Unset or any value but `1` on a target without the `w4a16` kernels keeps the BF16
//!   drafter, as before 2026-10-08; `0` keeps BF16 everywhere.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::kernel_handles;
use crate::dflash_head::{BlockDiffusionDraftHead, DflashQuantization};

/// 2026-09-25: FP8 drafter weights, on unless METRALE_DFLASH_DRAFTER_FP8=0: quantize each
/// layer's seven dense GEMM weights (q/k/v/o/gate/up/down) to FP8 E4M3 with one f32 scale
/// per row, give the lm_head an FP8 copy, and set `quant = Fp8Weights`. Needs the
/// row-scaled FP8 GEMM and its m16 variant; without them the drafter stays on BF16 with a
/// warning.
///
/// 2026-10-08: When the `w4a16` module lacks them and the variable is exactly `1`, the
/// pair comes from `dflash_fp8_gemm` (GLM-5.3's own port). The serve's footprint
/// pre-flight and KV reserve already count the mirrors for any value but `0`.
pub(super) fn arm_drafter_fp8(
    head: &mut BlockDiffusionDraftHead,
    gpu: &dyn GpuBackend,
    q_dim: usize,
    kv_dim: usize,
    lm_head_native_fp8: Option<(metrale_model_layers::weight_map::Fp8DenseWeight, usize)>,
) -> Result<()> {
    let fp8_env = std::env::var("METRALE_DFLASH_DRAFTER_FP8").ok();
    let fp8_requested = fp8_env.as_deref() != Some("0");
    let present = |h: &BlockDiffusionDraftHead| {
        h.kernels.fp8_gemm_n128_row_scaled.0 != 0 && h.kernels.fp8_gemm_n128_row_scaled_m16.0 != 0
    };
    let mut source = "w4a16";
    if fp8_requested && !present(head) && fp8_env.as_deref() == Some("1") {
        let (k, k16) = kernel_handles::load_fp8_row_scaled_fallback(gpu);
        if k.0 != 0 && k16.0 != 0 {
            head.kernels.fp8_gemm_n128_row_scaled = k;
            head.kernels.fp8_gemm_n128_row_scaled_m16 = k16;
            source = kernel_handles::GLM_FP8_GEMM_MODULE;
        }
    }
    if !fp8_requested {
        return Ok(());
    }
    if !present(head) {
        tracing::warn!(
            "METRALE_DFLASH_DRAFTER_FP8 is not 0 but fp8_gemm_t_row_scaled(_m16) are in \
             neither this target's w4a16 module nor (with =1) its dflash_fp8_gemm module \
             — staying on the BF16 drafter path."
        );
        return Ok(());
    }
    quantize_drafter_fp8(head, gpu, q_dim, kv_dim, lm_head_native_fp8)?;
    // 2026-10-08: The kernels the FP8 projections take, in the layers' and tail's order.
    let rt2 = crate::dflash_head::fp8_rt_enabled();
    tracing::info!(
        target: "metrale_model_arch::dflash_head::from_weights",
        "METRALE_DFLASH_DRAFTER_FP8: ENGAGED - drafter weights FP8 E4M3 row-scaled: \
         <=8 rows fp8_gemv_rowscale_batch8_rt2 (present {}), <=16 rows \
         fp8_gemv_rowscale_batch16_rt2 (present {}), rt2 enabled {}, else \
         {source}::fp8_gemm_t_row_scaled(_m16)",
        head.kernels.fp8_gemv_rt2.0 != 0,
        head.kernels.fp8_gemv_rt2_16.0 != 0,
        rt2,
    );
    Ok(())
}

/// 2026-09-26: Gives `head` FP8 copies of its layer weights and lm_head, and sets
/// `quant = Fp8Weights`.
fn quantize_drafter_fp8(
    head: &mut BlockDiffusionDraftHead,
    gpu: &dyn GpuBackend,
    q_dim: usize,
    kv_dim: usize,
    lm_head_native_fp8: Option<(metrale_model_layers::weight_map::Fp8DenseWeight, usize)>,
) -> Result<()> {
    tracing::info!(
        target: "metrale_model_arch::dflash_head::from_weights",
        "DFlash Phase G: quantizing drafter weights to FP8 E4M3 ({} layers × 7 GEMMs)",
        head.num_layers
    );
    let stream = 0u64;
    let q_dim_local = q_dim;
    let kv_dim_local = kv_dim;
    let h = head.hidden_size;
    let inter = head.intermediate_size;
    let quant_k = head.kernels.quantize_bf16_to_fp8;
    for (layer_idx, layer) in head.layers.iter_mut().enumerate() {
        layer.q_proj_fp8 =
            Some(
                layer
                    .q_proj
                    .quantize_to_fp8(gpu, quant_k, q_dim_local, h, stream)?,
            );
        layer.k_proj_fp8 =
            Some(
                layer
                    .k_proj
                    .quantize_to_fp8(gpu, quant_k, kv_dim_local, h, stream)?,
            );
        layer.v_proj_fp8 =
            Some(
                layer
                    .v_proj
                    .quantize_to_fp8(gpu, quant_k, kv_dim_local, h, stream)?,
            );
        layer.o_proj_fp8 =
            Some(
                layer
                    .o_proj
                    .quantize_to_fp8(gpu, quant_k, h, q_dim_local, stream)?,
            );
        layer.gate_proj_fp8 = Some(
            layer
                .gate_proj
                .quantize_to_fp8(gpu, quant_k, inter, h, stream)?,
        );
        layer.up_proj_fp8 = Some(
            layer
                .up_proj
                .quantize_to_fp8(gpu, quant_k, inter, h, stream)?,
        );
        layer.down_proj_fp8 = Some(
            layer
                .down_proj
                .quantize_to_fp8(gpu, quant_k, h, inter, stream)?,
        );
        tracing::debug!(
            target: "metrale_model_arch::dflash_head::from_weights",
            "DFlash Phase G: layer {} quantized", layer_idx
        );
    }
    // 2026-09-25: The lm_head's FP8 copy: the checkpoint's own FP8 lm_head when the
    // model factory passes one (`lm_head_native_fp8`, an `Fp8DenseWeight` like the
    // layer copies, so the same tail kernels read it), else a new FP8
    // quantization of the shared BF16 lm_head in its own buffer, which leaves the
    // target's weight untouched.
    if let Some((shared, rows)) = lm_head_native_fp8 {
        anyhow::ensure!(
            rows == head.vocab_size,
            "native FP8 lm_head share rows ({rows}) != drafter vocab ({}) — \
                     the drafter's tail GEMM iterates head.vocab_size rows",
            head.vocab_size
        );
        tracing::info!(
            target: "metrale_model_arch::dflash_head::from_weights",
            "DFlash Phase G: sharing the checkpoint's NATIVE FP8 lm_head \
             [{} × {}] (1.27 GB runtime mirror skipped)",
            head.vocab_size,
            head.hidden_size
        );
        head.lm_head_shared_fp8 = Some(shared);
    } else {
        tracing::info!(
            target: "metrale_model_arch::dflash_head::from_weights",
            "DFlash Phase G: quantizing shared lm_head [{} × {}]",
            head.vocab_size,
            head.hidden_size
        );
        let lm_head_bf16 = metrale_model_layers::weight_map::DenseWeight {
            weight: head.lm_head_shared,
        };
        head.lm_head_shared_fp8 = Some(lm_head_bf16.quantize_to_fp8(
            gpu,
            quant_k,
            head.vocab_size,
            head.hidden_size,
            stream,
        )?);
    }
    head.quant = DflashQuantization::Fp8Weights;
    tracing::info!(
        target: "metrale_model_arch::dflash_head::from_weights",
        "DFlash Phase G: drafter weights ready as FP8 (quant = Fp8Weights). \
         Set METRALE_DFLASH_DRAFTER_FP8=0 to revert to BF16."
    );
    Ok(())
}
