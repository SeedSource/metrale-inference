// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The drafter's BF16 projection launcher, and `METRALE_DFLASH_DRAFT_GEMV`
//! (SeedHQ/seed-skills#71 S1a).
//!
//! The paged propose path's BF16 weight GEMMs (the layer projections when the head is not
//! FP8, the DFlash2 conv and selector projections, the BF16 lm_head and the ctx
//! precompute's `fc` and fused K/V) go through [`BlockDiffusionDraftHead::draft_gemm_bf16`];
//! the non-paged body (`forward_block_layer.rs`, METRALE_DFLASH_OPTION_B=0) is unchanged.
//! Lever off, it is `dense_gemm_bf16_pipelined`, launch for launch what the call sites
//! issued before. That kernel tiles 128 rows of M by 128 of N, so at the propose's 8..16
//! rows its grid is `ceil(N / 128)` CTAs: 8 for a 1024-wide projection, 32 for 4096, on
//! 48 SMs (rank-0 nsys: 75.5 launches/step at a 52-CTA mean, 37.2 of 39 ms/step).
//! Lever on, rows `1..=DENSE_GEMV_BATCHM_MAX_M` with `K % 8 == 0` run
//! `dense_gemv_bf16_batchm` (common gb10 kernel, `ceil(N / 4)` CTAs, one weight pass for
//! all rows); wider calls keep the pipelined GEMM.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Lever off, or the batchm kernel absent, the launch is the pipelined GEMM with the
//!   arguments the caller passed.
//! - Drafter numerics only: the target model never reads these outputs, so its tokens
//!   do not change; the drafts (and so acceptance) may, by rounding.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

use super::BlockDiffusionDraftHead;

/// 2026-10-08: Set by the first batchm launch, so the engaged line prints once per process.
static ENGAGED_LOGGED: AtomicBool = AtomicBool::new(false);

/// 2026-10-08: Whether a BF16 projection of `m` rows and depth `k` runs on the batchm GEMV:
/// the lever is on, the kernel resolved, `m` is within the kernel's row cap and `k` keeps
/// its 16-byte row loads aligned.
pub(super) fn draft_gemv_routes(lever: bool, kernel_present: bool, m: u32, k: u32) -> bool {
    lever
        && kernel_present
        && (1..=ops::DENSE_GEMV_BATCHM_MAX_M).contains(&m)
        && k.is_multiple_of(8)
}

impl BlockDiffusionDraftHead {
    /// 2026-10-08: Resolve the batchm GEMV when `METRALE_DFLASH_DRAFT_GEMV=1`. The lookup
    /// happens only then, so a lever-off boot issues no new kernel lookup.
    pub(super) fn arm_draft_gemv(&mut self, gpu: &dyn GpuBackend) {
        if !self.levers.draft_gemv {
            return;
        }
        self.kernels.dense_gemv_batchm = metrale_model_layers::layers::try_kernel(
            gpu,
            "dense_gemv_bf16_batchm",
            "dense_gemv_bf16_batchm",
        );
        if self.kernels.dense_gemv_batchm.0 == 0 {
            tracing::warn!(
                "METRALE_DFLASH_DRAFT_GEMV=1 but dense_gemv_bf16_batchm is not in this \
                 target's PTX: the drafter keeps dense_gemm_bf16_pipelined"
            );
        }
    }

    /// 2026-10-08: `C[m, n] = A[m, k] @ W[n, k]^T`, BF16 in and out. See the module docs
    /// for the kernel choice.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draft_gemm_bf16(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let kernel = self.kernels.dense_gemv_batchm;
        if draft_gemv_routes(self.levers.draft_gemv, kernel.0 != 0, m, k) {
            if !ENGAGED_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "METRALE_DFLASH_DRAFT_GEMV: ENGAGED - drafter projections on \
                     dense_gemv_bf16_batchm (M<={}, first launch M={m} N={n} K={k})",
                    ops::DENSE_GEMV_BATCHM_MAX_M
                );
            }
            return ops::dense_gemv_batchm(gpu, kernel, input, weight, output, m, n, k, n, stream);
        }
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            input,
            weight,
            output,
            m,
            n,
            k,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::draft_gemv_routes;

    #[test]
    fn routes_only_with_lever_kernel_and_small_aligned_shapes() {
        assert!(draft_gemv_routes(true, true, 8, 4096));
        assert!(draft_gemv_routes(true, true, 1, 36864));
        assert!(draft_gemv_routes(true, true, 16, 12288));
        assert!(!draft_gemv_routes(false, true, 8, 4096), "lever off");
        assert!(!draft_gemv_routes(true, false, 8, 4096), "kernel absent");
        assert!(!draft_gemv_routes(true, true, 0, 4096), "no rows");
        assert!(!draft_gemv_routes(true, true, 17, 4096), "above MAX_M");
        assert!(!draft_gemv_routes(true, true, 8, 4100), "K % 8 != 0");
    }
}
