// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Comm-backend setup for `TransformerModel::new`: the startup check that the
//! ranks agree on the collective-shaping levers, and the world-size-2 reduce-buffer
//! registration.
//!
//! Owner: model-engine.
//! Invariants:
//! - `register_reduce_buffers` never fails construction: each step logs its failure and continues.

use std::sync::Arc;

use anyhow::Result;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::GpuBackend;

/// 2026-09-26: Fail unless every rank resolved the same collective-shaping levers.
pub(super) fn assert_rank_levers_agree(
    gpu: &dyn GpuBackend,
    comm: &Arc<dyn metrale_comm::CommBackend>,
) -> Result<()> {
    crate::rank_agree::assert_ranks_agree(
        gpu,
        comm.as_ref(),
        &[
            // 2026-09-25: Splits a prefill chunk into sub-chunks, and each
            // sub-chunk issues its own collectives.
            (
                "METRALE_GLM_PREFILL_ROWS",
                metrale_model_arch::glm5next_layer::prefill_rows() as u64,
            ),
            // 2026-09-29: The staged prefill (`METRALE_GLM_PREFILL_STAGED`) runs the FFN in
            // windows of `METRALE_GLM_PREFILL_ROWS_FFN` rows, one MLP all-reduce per window;
            // which sub-chunks share a window also follows the MoE grouped-GEMM levers, packed
            // in `staged_merge_signature`. All three change the collective count.
            (
                "METRALE_GLM_PREFILL_STAGED",
                u64::from(metrale_model_arch::glm5next_layer::prefill_staged()),
            ),
            (
                "METRALE_GLM_PREFILL_ROWS_FFN",
                metrale_model_arch::glm5next_layer::prefill_rows_ffn() as u64,
            ),
            // 2026-10-01: A merged tail drops one FFN window, and its all-reduce.
            (
                "METRALE_GLM_PREFILL_TAIL_MERGE",
                u64::from(metrale_model_arch::glm5next_layer::prefill_tail_merge()),
            ),
            // 2026-10-01: The full-width staged prefill runs the attention pass per FFN window,
            // so it issues one attention all-reduce per window instead of one per sub-chunk.
            (
                "METRALE_GLM_PREFILL_FULLWIDTH_GEMM",
                u64::from(metrale_model_arch::glm5next_layer::prefill_fullwidth_gemm()),
            ),
            (
                "GLM staged-prefill merge levers (MOE_PREFILL_GEMM[_MIN_ROWS], HOST_DISPATCH, \
                 ROUTE_TRACE)",
                metrale_model_arch::glm5next_layer::staged_merge_signature(),
            ),
            // 2026-10-01: The sequence-parallel staged prefill replaces each all-reduce with
            // normed-row exchanges and reduce-scatters, a different collective sequence.
            // 2026-10-04: Also under the full-width arm (FULLWIDTH_GEMM, checked above).
            (
                "METRALE_GLM_PREFILL_SEQ_PARALLEL",
                u64::from(metrale_model_arch::glm5next_layer::prefill_seq_parallel()),
            ),
            // 2026-10-05: Window ownership changes how many send/recv pairs each exchange posts.
            (
                "METRALE_GLM_PREFILL_SP_WINDOW_OWNER",
                u64::from(metrale_model_arch::glm5next_layer::prefill_sp_window_owner()),
            ),
            // 2026-10-01: Each DSA prefill sub-chunk swaps its index-selection halves with
            // one grouped send/recv; a rank without it would leave the peer waiting.
            (
                "METRALE_GLM_DSA_INDEX_SPLIT (with DSA_ROW_BATCH, DSA_BATCH_QIDX)",
                u64::from(metrale_model_arch::glm5next_layer::dsa_index_split()),
            ),
            // 2026-10-04: The same swap per full-width DSA sub-chunk (`decode_k_wide`).
            (
                "METRALE_GLM_DSA_INDEX_SPLIT_WIDE",
                u64::from(metrale_model_arch::glm5next_layer::dsa_index_split_wide()),
            ),
            // 2026-09-25: Performance only (the MLP reduces once per site
            // whichever arm runs); checked because the check is free.
            (
                "METRALE_GLM_MOE_ROW_BATCH_MAX",
                metrale_model_arch::glm5next_mlp::forward::row_batch_max() as u64,
            ),
            // 2026-09-29 (A153): Not collective-shaping, but every TP rank's NoPE MLA decode
            // launch must use the same softmax scale, or the attention each rank computes over
            // its local heads is numerically inconsistent with the others; checked because the
            // check is free.
            (
                "METRALE_GLM_MLA_SCALE_AUTHOR",
                u64::from(metrale_model_arch::glm5next_dsa::attend::mla_scale_author()),
            ),
            // 2026-10-02: The GLM batched MTP verify changes which collectives a decode step
            // issues (one batch-wide forward instead of one per sequence).
            (
                "METRALE_GLM_BATCHED_VERIFY",
                u64::from(metrale_model_arch::glm5next_layer::batched_verify()),
            ),
            // 2026-09-25: The EP command protocol (`ep_protocol_v2`), which
            // both ranks must share.
            (
                "METRALE_EP_PROTOCOL(v2)",
                u64::from(matches!(
                    std::env::var("METRALE_EP_PROTOCOL").as_deref(),
                    Ok("v2")
                )),
            ),
            // 2026-10-03: Lazily mapped DSA indexer caches: the pool decides when a step fails
            // with "KV cache exhausted" (and a sequence is preempted), so a rank with another
            // lever value or pool size would fail a step the other one runs.
            (
                "METRALE_DSA_INDEXER_LAZY",
                u64::from(metrale_model_arch::glm5next_dsa::lazy::dsa_indexer_lazy()),
            ),
            (
                "DSA indexer pool MiB (METRALE_DSA_INDEXER_POOL_GB)",
                if metrale_model_arch::glm5next_dsa::lazy::dsa_indexer_lazy() {
                    let l = metrale_model_arch::glm5next_dsa::lazy::indexer_pool().limit();
                    if l == usize::MAX {
                        u64::MAX
                    } else {
                        (l >> 20) as u64
                    }
                } else {
                    0
                },
            ),
            // 2026-10-06: The DSA pool cache changes per-sequence state, the reserve and when a
            // write or rewind is refused (ring bounds), so ranks must agree on it.
            (
                "METRALE_GLM_DSA_POOL_CACHE",
                u64::from(metrale_model_arch::glm5next_dsa::pool_cache::dsa_pool_cache()),
            ),
        ],
    )?;
    // 2026-09-29 (A153): Log the resolved NoPE MLA softmax-scale choice once, on rank 0 only
    // (every rank reads the same env var independently; the rank-agree check above is what
    // guarantees they agree).
    if comm.rank() == 0 {
        let author = metrale_model_arch::glm5next_dsa::attend::mla_scale_author();
        tracing::info!(
            "METRALE_GLM_MLA_SCALE_AUTHOR={}: NoPE MLA softmax scale = {}",
            u64::from(author),
            if author {
                "qk_head_dim^-0.5 (author convention)"
            } else {
                "kv_lora_rank^-0.5 (A153 default)"
            }
        );
    }
    Ok(())
}

/// 2026-09-26: Register the reduce targets with the comm backend and hand it the
/// `bf16_add_inplace` kernel.
pub(super) fn register_reduce_buffers(
    comm: &Arc<dyn metrale_comm::CommBackend>,
    buffers: &BufferArena,
    gpu: &dyn GpuBackend,
) {
    let moe_ptr = buffers.moe_output().0;
    let moe_bytes = buffers.sizes().moe_output;
    match comm.register_buffer(moe_ptr, moe_bytes) {
        Ok(_) => {
            tracing::info!(target: "metrale_model_engine::model::impl_a1", "Registered moe_output ({moe_bytes} B) with NCCL")
        }
        Err(e) => {
            tracing::warn!(target: "metrale_model_engine::model::impl_a1", "ncclCommRegister moe_output failed (non-fatal): {e}")
        }
    }
    let norm_ptr = buffers.norm_output().0;
    let norm_bytes = buffers.sizes().norm_output;
    match comm.register_buffer(norm_ptr, norm_bytes) {
        Ok(_) => {
            tracing::info!(target: "metrale_model_engine::model::impl_a1", "Registered norm_output ({norm_bytes} B) with NCCL")
        }
        Err(e) => {
            tracing::warn!(target: "metrale_model_engine::model::impl_a1", "ncclCommRegister norm_output failed (non-fatal): {e}")
        }
    }
    let logits_ptr = buffers.logits().0;
    let logits_bytes = buffers.sizes().logits;
    match comm.register_buffer(logits_ptr, logits_bytes) {
        Ok(_) => {
            tracing::info!(target: "metrale_model_engine::model::impl_a1", "Registered logits ({logits_bytes} B) with NCCL")
        }
        Err(e) => {
            tracing::warn!(target: "metrale_model_engine::model::impl_a1", "ncclCommRegister logits failed (non-fatal): {e}")
        }
    }
    match gpu.kernel("bf16_add", "bf16_add_inplace") {
        Ok(k) => comm.set_add_kernel(k.0),
        Err(e) => {
            tracing::warn!(target: "metrale_model_engine::model::impl_a1", "bf16_add_inplace kernel not found (send/recv disabled): {e}")
        }
    }
}
