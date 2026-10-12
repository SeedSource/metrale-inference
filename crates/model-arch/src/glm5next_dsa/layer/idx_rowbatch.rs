// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_DSA_IDX_ROWBATCH=1` (default off): in a single-sequence MTP verify
//! (`decode_k_rows`, `k >= 2` rows) the DSA indexer's four projections run once over the `k`
//! rows instead of once per row at M = 1 (12 launches per layer at k = 3 become 4:
//! `wk`, `k_norm`, `compress_gate`, `weights_proj`, `wq_b` over `k` rows, 5 with the norm).
//!
//! The projection block is [`Glm5NextDsaLayer::indexer_project_rows`], the one body shared with
//! the cross-sequence batch (`xseq/group.rs`), so C = 1 and C > 1 run the same launches. The
//! result is a [`PreIndexer`] over fixed workspace buffers (`k_normed_rows`, `gate_rows`,
//! `q_idx_rows`, `head_weights_rows`), which `decode_k_rows` and `select_rows_ceiling` already
//! consume for the cross-sequence path.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - 2026-10-12: `wk` and `compress_gate` go through `gemm` -> `dense_fp8::route` (the
//!   row-invariant tensor-core GEMV, `METRALE_GLM_GEMV_TC=1`), never `batchm_rows`: that
//!   diverged from the per-row path under the ship env (race-dec-lmhbf-L34, C = 2/4).
//!   `weights_proj` and `wq_b` go through `batchm_rows` with FP32 out.
//! - 2026-10-12: Engages only where [`super::ceiling_verify::ceiling_rows_group`] holds
//!   (`group`: replay-safe, `meta.num_seqs == k`, not the batched selector) and the
//!   tensor-core GEMV routes `wk` / `compress_gate` at `k` rows; the buffers are allocated at
//!   load ([`super::workspace`]), nothing is allocated or varies per call, so a captured graph
//!   replays valid addresses. Lever off: exactly the old per-row path.
//! - 2026-10-12: `wk` and `compress_gate` stay two `M = k` launches (not stacked): a stacked
//!   `[wk; compress_gate]` output has row stride 256, but `k_norm` normalises contiguous
//!   `[rows, 128]` rows, so it would need `k` per-row `k_norm` launches (more than it saves).

use std::sync::OnceLock;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops::dense_gemv_tcm;

use super::row_batch::batchm_rows;
use super::row_src::{PreIndexer, RowSrc};
use super::{Glm5NextDsaLayer, gemm};
use crate::glm5next_dsa::select::dsa_ceiling_rows;

/// 2026-10-12: `METRALE_GLM_DSA_IDX_ROWBATCH=1` (default off), read once. A value other than
/// unset, empty, `0` or `1` is off, with one warning. With the lever on and
/// `METRALE_GLM_DSA_CEILING_ROWS` off it never engages (the `group` gate), which is warned
/// about once here.
pub(crate) fn dsa_idx_rowbatch() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_IDX_ROWBATCH").ok();
        let on = raw.as_deref().map(str::trim) == Some("1");
        let odd = |r: &&str| !r.is_empty() && r.trim() != "0" && r.trim() != "1";
        if let Some(r) = raw.as_deref().filter(odd) {
            tracing::warn!("METRALE_GLM_DSA_IDX_ROWBATCH={r} is not 0 or 1 - treated as off");
        }
        if on && !dsa_ceiling_rows() {
            tracing::warn!(
                "METRALE_GLM_DSA_IDX_ROWBATCH=1: NOT ENGAGED (requires METRALE_GLM_DSA_CEILING_ROWS=1)"
            );
        }
        on
    })
}

/// 2026-10-12: The FP32-out batched GEMV the indexer's `weights_proj` / `wq_b` run on
/// (`KernelHandle(0)` when the target lacks it).
pub fn idx_batchm_f32_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    metrale_model_layers::layers::try_kernel(
        gpu,
        "dense_gemv_bf16_batchm",
        "dense_gemv_bf16_fp32out_batchm",
    )
}

/// 2026-10-12: Where [`Glm5NextDsaLayer::indexer_project_rows`] writes `rows` rows.
#[derive(Clone, Copy)]
pub struct IndexerRowsOut {
    /// `[rows, index_head_dim]` BF16: `wk(hidden)` after `k_norm`.
    pub k_normed: DevicePtr,
    /// `[rows, index_head_dim]` BF16: `compress_gate(hidden)`.
    pub gate: DevicePtr,
    /// `[rows, index_heads]` FP32: `weights_proj(hidden)`.
    pub head_weights: DevicePtr,
    /// `[rows, index_heads * index_head_dim]` FP32: `wq_b(q_resid)`.
    pub q_idx: DevicePtr,
}

impl Glm5NextDsaLayer {
    /// 2026-10-12: The indexer projections of `rows` rows (the block `xseq_group` ran inline,
    /// moved here unchanged): `wk` (`gemm`) then `k_norm` in place, `compress_gate` (`gemm`),
    /// `weights_proj` and `wq_b` (`batchm_rows`, FP32 out, kernel `f32k`).
    ///
    /// `wk` / `compress_gate` go through `gemm`, the dispatch the per-sequence decode uses
    /// (`dense_fp8::route`: the row-invariant TC GEMV under `METRALE_GLM_GEMV_TC=1`), so a
    /// sequence's indexer keys do not depend on how many rows decode with it. `batchm_rows` on
    /// `dense_gemv_bf16_batchm` matched the per-sequence path only with TC off
    /// (race-dec-lmhbf-L34: C=2/4 texts diverged from C=1 under the ship env).
    /// `nllb_layernorm_bf16` normalises row `blockIdx.x` in place; every block does the same
    /// arithmetic on its own row at any `rows` (`indexer_rows_batched`).
    #[allow(clippy::too_many_arguments)]
    pub fn indexer_project_rows(
        &self,
        gpu: &dyn GpuBackend,
        f32k: KernelHandle,
        hidden: DevicePtr,
        q_resid: DevicePtr,
        out: &IndexerRowsOut,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (hid, ql, d, heads) = (c.hidden, c.q_lora_rank, c.index_head_dim, c.index_heads);
        let kn = &self.kernels;
        let wt = &self.weights;
        gemm(
            gpu,
            kn.gemm,
            kn.gemv,
            kn.gemv_batchm,
            hidden,
            wt.wk,
            out.k_normed,
            rows,
            d,
            hid,
            stream,
        )?;
        KernelLaunch::new(gpu, self.select_kernels.k_norm)
            .grid([rows as u32, 1, 1])
            .block([d.min(1024) as u32, 1, 1])
            .shared_mem((d.min(1024) * 4) as u32)
            .arg_ptr(out.k_normed)
            .arg_ptr(wt.k_norm_weight)
            .arg_ptr(wt.k_norm_bias)
            .arg_u32(rows as u32)
            .arg_u32(d as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        gemm(
            gpu,
            kn.gemm,
            kn.gemv,
            kn.gemv_batchm,
            hidden,
            wt.compress_gate,
            out.gate,
            rows,
            d,
            hid,
            stream,
        )?;
        batchm_rows(
            gpu,
            f32k,
            4,
            hidden,
            wt.weights_proj,
            out.head_weights,
            rows,
            heads,
            hid,
            stream,
        )?;
        let nq = heads * d;
        batchm_rows(
            gpu, f32k, 4, q_resid, wt.wq_b, out.q_idx, rows, nq, ql, stream,
        )
    }

    /// 2026-10-12: `wq_b` of ONE `q_resid` row (M = 1, FP32 out) into `out` (`index_heads *
    /// index_head_dim` FP32): the launch `select_rows_ceiling` makes per row without a
    /// precomputed indexer, moved here unchanged.
    pub fn indexer_qidx_one(
        &self,
        gpu: &dyn GpuBackend,
        q_resid_row: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let cfg = &self.cfg;
        gemm(
            gpu,
            self.kernels.gemm_f32,
            self.kernels.gemv_f32,
            // 2026-09-25: No FP32-out batched GEMV kernel exists.
            KernelHandle(0),
            q_resid_row,
            self.weights.wq_b,
            out,
            1,
            cfg.index_heads * cfg.index_head_dim,
            cfg.q_lora_rank,
            stream,
        )
    }

    /// 2026-10-12: Under the lever and when `decode_k_rows` runs the `CeilingRows` pass
    /// (`group`), run the indexer projections of the call's `k` rows once, into the workspace's
    /// fixed row buffers, and return them as the call's [`PreIndexer`]. `None` (nothing
    /// launched) when the lever is off, `src` already carries a precomputed indexer, or any
    /// buffer / kernel / route is missing; the row loop then runs as before.
    pub(super) fn idx_rowbatch_pre(
        &self,
        gpu: &dyn GpuBackend,
        group: bool,
        hidden: DevicePtr,
        src: &RowSrc,
        k: usize,
        stream: u64,
    ) -> Result<Option<PreIndexer>> {
        if !group || !dsa_idx_rowbatch() || src.pre.is_some() {
            return Ok(None);
        }
        let w = &self.workspace;
        let c = &self.cfg;
        let (hid, d) = (c.hidden as u32, c.index_head_dim as u32);
        let have_buffers = w.idx_k_normed_rows.0 != 0
            && w.idx_gate_rows.0 != 0
            && w.q_idx_rows.0 != 0
            && w.head_weights_rows.0 != 0
            && w.idx_batchm_f32.0 != 0
            && k <= w.max_rows;
        // 2026-10-12: Both `gemm` calls must take the TC GEMV at `k` rows, as they do at one row;
        // without it `gemm` falls to `dense_gemv_bf16_batchm` at M = k against the plain
        // GEMV at M = 1 (not proven identical).
        let tc = dense_gemv_tcm::ready(gpu, k as u32, d, hid, false);
        if !have_buffers || !tc {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::warn!(
                    "METRALE_GLM_DSA_IDX_ROWBATCH=1: NOT ENGAGED (buffers {have_buffers}, \
                     tensor-core GEMV routes k={k}: {tc}; needs METRALE_GLM_GEMV_TC=1)"
                )
            });
            return Ok(None);
        }
        let out = IndexerRowsOut {
            k_normed: w.idx_k_normed_rows,
            gate: w.idx_gate_rows,
            head_weights: w.head_weights_rows,
            q_idx: w.q_idx_rows,
        };
        self.indexer_project_rows(gpu, w.idx_batchm_f32, hidden, src.q_resid, &out, k, stream)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "METRALE_GLM_DSA_IDX_ROWBATCH=1: ENGAGED (single-seq verify: indexer projections \
                 batched over k={k} rows; wk+cgate two launches; weights_proj/wq_b fp32out batchm)"
            )
        });
        Ok(Some(PreIndexer {
            k_normed: out.k_normed,
            gate: out.gate,
            head_weights: out.head_weights,
            q_idx: out.q_idx,
        }))
    }
}
