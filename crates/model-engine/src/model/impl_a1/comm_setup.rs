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
            (
                "GLM staged-prefill merge levers (MOE_PREFILL_GEMM[_MIN_ROWS], HOST_DISPATCH, \
                 ROUTE_TRACE)",
                metrale_model_arch::glm5next_layer::staged_merge_signature(),
            ),
            // 2026-09-25: Performance only (the MLP reduces once per site
            // whichever arm runs); checked because the check is free.
            (
                "METRALE_GLM_MOE_ROW_BATCH_MAX",
                metrale_model_arch::glm5next_mlp::forward::row_batch_max() as u64,
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
        ],
    )?;
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
