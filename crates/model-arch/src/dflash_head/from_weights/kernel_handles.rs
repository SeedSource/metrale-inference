// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The DFlash drafter's kernel handles, resolved once when the head is built.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: the γ-block paged-indirect attention is the build for the drafter's `head_dim`
//! (`attn_width`, A93); a width without a build fails construction.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::dflash_head::DflashKernels;
use crate::dflash_head::attn_width;

/// 2026-09-26: Every handle in `DflashKernels`, resolved in field order.
pub(super) fn load_kernels(gpu: &dyn GpuBackend, head_dim: usize) -> Result<DflashKernels> {
    // 2026-10-10: The paged-indirect kernel's tile width is compile-time; pick the build for this
    // drafter's head_dim and refuse a mismatch before binding it (A93).
    let indirect = attn_width::paged_indirect_spec_for(head_dim)?;
    attn_width::assert_width(indirect, head_dim)?;
    tracing::info!(
        "DFlash γ-block attention: {}::{} (HDIM {}, drafter head_dim {head_dim})",
        indirect.module,
        indirect.func,
        indirect.hdim
    );
    Ok(DflashKernels {
        // 2026-09-25: The drafter's norms use `rms_norm_vanilla`
        // (`x * w / RMS(x)`), not `rms_norm`, which computes `x * (1 + w) / RMS(x)`.
        rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
        residual_rms_norm: gpu
            .kernel("norm", "rms_norm_residual")
            .or_else(|_| gpu.kernel("residual_add", "bf16_residual_add"))?,
        dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
        dense_gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
        w4a16_gemm: metrale_model_layers::layers::try_kernel(gpu, "w4a16", "w4a16_gemm"),
        dense_gemm_pipelined: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
        // 2026-10-08: Filled by `arm_draft_gemv` when METRALE_DFLASH_DRAFT_GEMV=1.
        dense_gemv_batchm: metrale_gpu_runtime::gpu::KernelHandle(0),
        // 2026-09-25: `rope_forward_yarn` takes the inverse-frequency table
        // (`yarn_inv_freq`, built below) as an argument, so it also runs plain RoPE.
        rope_qwen3: gpu.kernel("rope", "rope_forward_yarn")?,
        reshape_cache_fp8: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash_fp8")?,
        reshape_cache_bf16: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash")?,
        prefill_attn_dflash_fp8: gpu.kernel("prefill_paged_fp8", "attn_prefill_paged_fp8")?,
        prefill_attn_dflash_bf16: gpu.kernel("prefill_paged", "attn_prefill_paged")?,
        prefill_attn_dflash_bf16_indirect: gpu.kernel(indirect.module, indirect.func)?,
        silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
        residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
        argmax: gpu.kernel("argmax", "argmax_bf16")?,
        batched_embed: gpu.kernel("embed_from_argmax", "batched_embed")?,
        fill_slots: gpu.kernel("metadata_fill", "fill_slots_from_block_table")?,
        // 2026-09-25: The HDIM=128 build of the non-paged prefill kernel, from
        // `kernels/<hw>/common/attn_prefill_h128.cu`; it requires head_dim == 128.
        prefill_attn: gpu
            .kernel("attn_prefill_h128", "attn_prefill_h128")
            .map_err(|e| {
                anyhow::anyhow!(
                    "{e}\n\nDFlash needs the HDIM=128 prefill kernel \
                         (`attn_prefill_h128`) compiled for this target. \
                         The kernel source lives at \
                         `kernels/<hw>/common/attn_prefill_h128.cu`. \
                         If you've added a new hardware target, copy the \
                         .cu file there."
                )
            })?,
        // 2026-09-25: Used only here, at load, for the FP8 weight copies below.
        quantize_bf16_to_fp8: gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?,
        // 2026-09-25: The row-scaled FP8 GEMM from the `w4a16` module, first found of
        // `fp8_gemm_t_row_scaled_k64`, `fp8_gemm_t_row_scaled_p4`,
        // `fp8_gemm_t_row_scaled`. `METRALE_DFLASH_FP8_GEMM_P4=1` skips `_k64`, and
        // `METRALE_DFLASH_FP8_GEMM_P2=1` takes only the last. `KernelHandle(0)` when
        // none is present, and the FP8 drafter path then stays off (the
        // `fp8_kernels_present` check below).
        fp8_gemm_n128_row_scaled: {
            let pin_p2 = std::env::var("METRALE_DFLASH_FP8_GEMM_P2").ok().as_deref() == Some("1");
            let pin_p4 = std::env::var("METRALE_DFLASH_FP8_GEMM_P4").ok().as_deref() == Some("1");
            let mut h = metrale_gpu_runtime::gpu::KernelHandle(0);
            if !pin_p2 && !pin_p4 {
                h = metrale_model_layers::layers::try_kernel(
                    gpu,
                    "w4a16",
                    "fp8_gemm_t_row_scaled_k64",
                );
                if h.0 != 0 {
                    tracing::info!(
                        target: "metrale_model_arch::dflash_head::from_weights",
                        "DFlash drafter FP8 GEMM: fp8_gemm_t_row_scaled_k64 (deep-K)"
                    );
                }
            }
            if h.0 == 0 && !pin_p2 {
                h = metrale_model_layers::layers::try_kernel(
                    gpu,
                    "w4a16",
                    "fp8_gemm_t_row_scaled_p4",
                );
                if h.0 != 0 {
                    tracing::info!(
                        target: "metrale_model_arch::dflash_head::from_weights",
                        "DFlash drafter FP8 GEMM: fp8_gemm_t_row_scaled_p4 (4-stage ring)"
                    );
                }
            }
            if h.0 == 0 {
                h = metrale_model_layers::layers::try_kernel(gpu, "w4a16", "fp8_gemm_t_row_scaled");
            }
            h
        },
        dense_gemv_fp8w: gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?,
        fp8_gemm_n128_row_scaled_m16: metrale_model_layers::layers::try_kernel(
            gpu,
            "w4a16",
            "fp8_gemm_t_row_scaled_m16",
        ),
        fp8_gemv_rt2: metrale_model_layers::layers::try_kernel(
            gpu,
            "fp8_gemv_rt",
            "fp8_gemv_rowscale_batch8_rt2",
        ),
        fp8_gemv_rt2_16: metrale_model_layers::layers::try_kernel(
            gpu,
            "fp8_gemv_rt",
            "fp8_gemv_rowscale_batch16_rt2",
        ),
        // 2026-09-25: `try_kernel`, so a target without the `dflash2` module still
        // loads other drafters; `dflash2_active` is then false.
        dflash2_conv2: metrale_model_layers::layers::try_kernel(gpu, "dflash2", "dflash2_conv2"),
        dflash2_topk16: metrale_model_layers::layers::try_kernel(gpu, "dflash2", "dflash2_topk16"),
        dflash2_topk16_reg: metrale_model_layers::layers::try_kernel(
            gpu,
            "dflash2",
            "dflash2_topk16_reg",
        ),
        dflash2_selector_walk: metrale_model_layers::layers::try_kernel(
            gpu,
            "dflash2",
            "dflash2_selector_walk",
        ),
    })
}

/// 2026-10-08: Module of the GLM-5.3 target's own row-scaled FP8 drafter GEMMs
/// (`kernels/gb10/glm-5.3-flash/nvfp4/dflash_fp8_gemm.cu`, SeedHQ/seed-skills#71 S1b).
pub(super) const GLM_FP8_GEMM_MODULE: &str = "dflash_fp8_gemm";

/// 2026-10-08: `fp8_gemm_t_row_scaled` and `fp8_gemm_t_row_scaled_m16` from
/// [`GLM_FP8_GEMM_MODULE`], each `KernelHandle(0)` when absent. Probed only when the
/// `w4a16` module lacks them and METRALE_DFLASH_DRAFTER_FP8=1 is explicit
/// (`fp8_weights::arm_drafter_fp8`), so other targets and lever-off boots issue no new
/// lookup.
pub(super) fn load_fp8_row_scaled_fallback(
    gpu: &dyn GpuBackend,
) -> (
    metrale_gpu_runtime::gpu::KernelHandle,
    metrale_gpu_runtime::gpu::KernelHandle,
) {
    (
        metrale_model_layers::layers::try_kernel(gpu, "dflash_fp8_gemm", "fp8_gemm_t_row_scaled"),
        metrale_model_layers::layers::try_kernel(
            gpu,
            "dflash_fp8_gemm",
            "fp8_gemm_t_row_scaled_m16",
        ),
    )
}
