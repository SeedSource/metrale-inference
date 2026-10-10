// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `DflashKernels`, the kernel handles the DFlash drafter launches.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Kernel handles for the DFlash drafter, all resolved once in
/// `BlockDiffusionDraftHead::from_weights`. A handle resolved there with
/// `try_kernel` is `KernelHandle(0)` when the target's build lacks the kernel.
pub struct DflashKernels {
    pub rms_norm: KernelHandle,
    pub residual_rms_norm: KernelHandle,
    pub dense_gemv: KernelHandle,
    pub dense_gemm: KernelHandle,
    /// 2026-09-25: NVFP4 GEMM for the logits when the target's lm_head is NVFP4
    /// (`lm_head_nvfp4` is `Some`), whose packed bytes the BF16 GEMM cannot
    /// read. `.0 == 0` when the target has no `w4a16_gemm` kernel.
    pub w4a16_gemm: KernelHandle,
    pub dense_gemm_pipelined: KernelHandle,
    /// 2026-10-08: `dense_gemv_bf16_batchm`, resolved only under
    /// `METRALE_DFLASH_DRAFT_GEMV=1` (`draft_gemm.rs`); `KernelHandle(0)` otherwise.
    pub dense_gemv_batchm: KernelHandle,
    pub rope_qwen3: KernelHandle,
    pub reshape_cache_fp8: KernelHandle,
    /// 2026-09-25: BF16 paged KV write: the ctx rows in `precompute_ctx_kv` and
    /// the block rows in `forward_block_layer_pre_attn`.
    pub reshape_cache_bf16: KernelHandle,
    pub prefill_attn_dflash_fp8: KernelHandle,
    pub prefill_attn_dflash_bf16: KernelHandle,
    /// 2026-09-25: `attn_prefill_paged_indirect`: BF16 paged attention that reads
    /// `kv_len`, `q_offset` and `q_rope_pos` from device memory at kernel entry.
    /// The drafter's paged attention (`forward_block_layer_attention`) launches it
    /// non-causally.
    pub prefill_attn_dflash_bf16_indirect: KernelHandle,
    pub silu_mul: KernelHandle,
    pub residual_add: KernelHandle,
    pub argmax: KernelHandle,
    pub batched_embed: KernelHandle,
    /// 2026-09-25: Builds `count` i64 cache slot indices on the device from a
    /// block table: the ctx rows' slots in propose and the batched precompute,
    /// and the block rows' slots in `forward_block`.
    pub fill_slots: KernelHandle,
    /// 2026-09-25: Non-paged prefill attention, built for head_dim 128 only
    /// (`attn_prefill_h128`): the non-paged layer body and the
    /// `dflash_contig_attn` path use it.
    pub prefill_attn: KernelHandle,
    /// 2026-09-25: BF16 → FP8 E4M3 weight quantization with one f32 scale per
    /// row, run only in `from_weights` to build the FP8 mirrors of the seven
    /// dense GEMM weights (q/k/v/o/gate/up/down) and, when the checkpoint has
    /// no FP8 lm_head, the lm_head.
    pub quantize_bf16_to_fp8: KernelHandle,
    /// 2026-09-25: Row-scaled BF16 × FP8 → BF16 GEMM over an `Fp8DenseWeight`,
    /// from the `w4a16` module (`from_weights` picks the variant). With
    /// `DflashQuantization::Fp8Weights` the layer GEMMs use it when neither rt2
    /// GEMV applies, and the lm_head uses it above 16 rows.
    pub fp8_gemm_n128_row_scaled: KernelHandle,
    pub dense_gemv_fp8w: KernelHandle,
    /// 2026-09-25: `fp8_gemm_t_row_scaled_m16`, a single-warp row-scaled FP8 GEMM
    /// that computes at most 16 rows. The FP8 lm_head uses it at 16 rows or fewer
    /// when neither rt2 GEMV applies.
    pub fp8_gemm_n128_row_scaled_m16: KernelHandle,
    /// 2026-09-25: `fp8_gemv_rowscale_batch8_rt2`, a register-tiled row-scaled FP8
    /// GEMV for at most 8 rows. The FP8 layer GEMMs and the FP8 lm_head try it
    /// first, then `fp8_gemv_rt2_16`, then a tile GEMM. `.0 == 0` on targets
    /// without the `fp8_gemv_rt` module.
    /// Kill-switch: METRALE_NO_DFLASH_FP8_RT=1. provenance-id:
    /// 526f6e616c6420522e205374657369616b
    pub fp8_gemv_rt2: KernelHandle,
    /// 2026-09-25: `fp8_gemv_rowscale_batch16_rt2`, the 16-row sibling of
    /// `fp8_gemv_rt2`, for at most 16 rows. `.0 == 0` on targets without it.
    /// Both rt2 GEMVs also need K to be a multiple of 16, and
    /// `METRALE_NO_DFLASH_FP8_RT=1` disables both.
    /// provenance-id: 526f6e616c6420522e205374657369616b
    pub fp8_gemv_rt2_16: KernelHandle,
    /// 2026-09-25: DFlash2 two-tap grouped dynamic conv (`kernels/gb10/common/dflash2.cu`).
    /// `.0 == 0` on targets without the module, and `dflash2_active` is then false.
    pub dflash2_conv2: KernelHandle,
    /// 2026-09-25: DFlash2 per-row top-16 over the drafter logits; overwrites each
    /// selected logit.
    pub dflash2_topk16: KernelHandle,
    /// 2026-09-25: DFlash2 candidate-selector chain walk, one launch, one block per
    /// sequence.
    pub dflash2_selector_walk: KernelHandle,
}
