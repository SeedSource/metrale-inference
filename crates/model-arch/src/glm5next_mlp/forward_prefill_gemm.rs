// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Routed-MoE prefill for GLM-5.3 through the shared grouped W4A16 GEMM
//! (`moe_w4a16_grouped_gemm_ptrtable` and its tile variants): one launch per projection for all
//! experts. `tile.rs` holds the tile table and the env levers, `dispatch.rs` the launch.
//!
//! The GEMM stages each dequantised weight as BF16 for `mma.sync`; the GEMV path keeps it in
//! FP32, so the two paths do not produce the same bits. `forward_moe` takes this path only for
//! row groups wider than `MOE_ROW_BATCH_MAX_ROWS` and at least `prefill_gemm_min_rows()` wide.
//! Measured 2026-09-22 with `examples/glm5next_moe_grouped_prefill_microtest.rs` (M=64, top_k=8,
//! 16 experts, N=256, K=512): against an FP32 reference that rounds each dequantised weight to
//! BF16 first, the grouped GEMM's max_rel was 0.003888; the GEMV's against the exact FP32
//! reference was 0.003889.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The GEMM reads the same `Glm5NextMoePtrTables` as the GEMV paths, indexed by GLOBAL expert
//!   id, and writes nothing for an expert whose `packed` pointer is null.
//! - Every buffer the path uses is in `Glm5NextMlpWorkspace`; the path allocates nothing.

pub mod cutlass_w4a4;
mod dispatch;
mod tile;
mod w4a16_mma;

pub(crate) use tile::*;
pub(crate) use w4a16_mma::PrefillMmaKernels;

pub(super) use dispatch::forward_moe_grouped_prefill;
pub use dispatch::grouped_prefill_selected;

#[cfg(test)]
mod tests;
