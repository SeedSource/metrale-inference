// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! 2026-09-25: Final norm and LM head for the mixed decode + prefill forward.
//!
//! Owner: model-engine decode.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use metrale_model_layers::layers::ops;

/// 2026-09-25: Logits of `mixed_final_norm_lm_head`. `prefill_logits` is
/// `DevicePtr::NULL` when `prefill_is_last` is false.
pub(super) struct MixedHeadOut {
    pub decode_logits: DevicePtr,
    pub prefill_logits: DevicePtr,
}

impl TransformerModel {
    pub(super) fn mixed_final_norm_lm_head(
        &self,
        hidden: DevicePtr,
        prefill_hidden: DevicePtr,
        padded_n: usize,
        proc_count: usize,
        prefill_is_last: bool,
        h: usize,
        bf16: usize,
        fp32: usize,
        stream: u64,
    ) -> Result<MixedHeadOut> {
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;

        self.final_norm_apply(hidden, normed, padded_n as u32, h as u32, eps, stream)?;

        // 2026-09-25: The pure-decode head (`decode_batch_compute_main_with`) calls the same
        // function, so both heads pick the same kernel, and the same numerics, at a given
        // `padded_n`.
        let logits = self.lm_head_project_batched(normed, padded_n, h, bf16, stream)?;
        let v = self.config.vocab_size;
        let decode_logits = logits;

        let prefill_logits = if prefill_is_last {
            let last_hidden = prefill_hidden.offset((proc_count - 1) * h * fp32);
            // 2026-09-25: The prefill row goes after the `padded_n` decode rows, in both
            // the norm output and the logits, so it does not overwrite them.
            let prefill_normed = normed.offset(padded_n * h * bf16);
            self.final_norm_apply(last_hidden, prefill_normed, 1, h as u32, eps, stream)?;

            let prefill_logits_ptr = logits.offset(padded_n * v * bf16);
            if let Some(ref fp8) = self.lm_head_fp8 {
                // 2026-10-05: `METRALE_GLM_GEMV_TC=1`: the tensor-core FP8 GEMV for this one
                // row too, so it matches the batched decode rows.
                if !ops::dense_gemv_tcm::try_fp8(
                    self.gpu.as_ref(),
                    prefill_normed,
                    fp8,
                    prefill_logits_ptr,
                    1,
                    v as u32,
                    h as u32,
                    v as u32,
                    stream,
                )? {
                    ops::dense_gemv_fp8w(
                        self.gpu.as_ref(),
                        self.dense_gemv_fp8w_kernel,
                        prefill_normed,
                        fp8,
                        prefill_logits_ptr,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                }
            } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
                ops::w4a16_gemv(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_kernel,
                    prefill_normed,
                    nvfp4,
                    prefill_logits_ptr,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    self.gpu.as_ref(),
                    self.dense_gemv_kernel,
                    prefill_normed,
                    &self.lm_head_weight,
                    prefill_logits_ptr,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            }
            prefill_logits_ptr
        } else {
            DevicePtr::NULL
        };

        Ok(MixedHeadOut {
            decode_logits,
            prefill_logits,
        })
    }
}
