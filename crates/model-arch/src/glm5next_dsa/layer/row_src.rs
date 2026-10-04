// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: [`RowSrc`], where `Glm5NextDsaLayer::decode_k_rows` reads one sequence's
//! projected rows and writes its attend output, with the optional precomputed indexer
//! projections ([`PreIndexer`]) of the cross-sequence batch (`xseq.rs`), and
//! `store_pre_indexer_row`, which places one precomputed indexer row.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`RowSrc::workspace`] names exactly the workspace buffers `decode_k` projects into,
//!   with no precomputed indexer, so `decode_k` reads what it read before `RowSrc` existed.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::state::Glm5NextDsaState;
use super::Glm5NextDsaLayer;

/// 2026-10-03: Where `decode_k_rows` reads one sequence's projected rows and writes its
/// attend output: row 0 of each buffer is the sequence's first row.
#[derive(Clone, Copy)]
pub(super) struct RowSrc {
    /// `[k, kv_lora_rank]` BF16, `kv_a_proj(hidden)`.
    pub(super) kv_a: DevicePtr,
    /// `[k, q_lora_rank]` BF16, after `q_a_layernorm`.
    pub(super) q_resid: DevicePtr,
    /// `[k, local_heads * kv_lora_rank]` BF16, the absorbed queries.
    pub(super) q_abs: DevicePtr,
    /// `[k, local_heads * kv_lora_rank]` BF16, the gather-attend output.
    pub(super) attn_out: DevicePtr,
    /// The indexer projections, already run for these rows; `None`: `indexer_forward` and
    /// `select_row` run them per row.
    pub(super) pre: Option<PreIndexer>,
}

impl RowSrc {
    /// 2026-10-03: The workspace buffers `decode_k` projects into, with no precomputed
    /// indexer: exactly what the row loop read before `RowSrc` existed.
    pub(super) fn workspace(w: &super::Glm5NextDsaWorkspace) -> Self {
        Self {
            kv_a: w.kv_a,
            q_resid: w.q_resid,
            q_abs: w.q_abs,
            attn_out: w.attn_out,
            pre: None,
        }
    }
}

/// 2026-10-03: One sequence's precomputed indexer projections (`xseq.rs`), row 0 first.
#[derive(Clone, Copy)]
pub(super) struct PreIndexer {
    /// `[k, index_head_dim]` BF16, `wk(hidden)` after `k_norm`.
    pub(super) k_normed: DevicePtr,
    /// `[k, index_head_dim]` BF16, `compress_gate(hidden)`.
    pub(super) gate: DevicePtr,
    /// `[k, index_heads]` FP32, `weights_proj(hidden)`.
    pub(super) head_weights: DevicePtr,
    /// `[k, index_heads * index_head_dim]` FP32, `wq_b(q_resid)`.
    pub(super) q_idx: DevicePtr,
}

impl PreIndexer {
    /// 2026-10-03: Row `row`'s selector query and head weights.
    pub(super) fn row(&self, row: usize, cfg: &super::super::Glm5NextDsaConfig) -> PreRow {
        PreRow {
            q_idx: self
                .q_idx
                .offset(row * cfg.index_heads * cfg.index_head_dim * 4),
            head_weights: self.head_weights.offset(row * cfg.index_heads * 4),
        }
    }
}

/// 2026-10-03: One row's precomputed selector inputs, for `select_row_at`.
#[derive(Clone, Copy)]
pub(super) struct PreRow {
    pub(super) q_idx: DevicePtr,
    pub(super) head_weights: DevicePtr,
}

impl Glm5NextDsaLayer {
    /// 2026-10-03: `indexer_forward` for row `row` of `pre`, whose projections and `k_norm`
    /// already ran: copy the row's key and gate into indexer cache row `state.len()`, then
    /// `store_indexer_row` (validity mark, advance). Host-offset placement only; `pos_dev`
    /// (graph capture) is refused, and `xseq_ready` never engages while capturing.
    pub(super) fn store_pre_indexer_row(
        &self,
        gpu: &dyn GpuBackend,
        pre: &PreIndexer,
        row: usize,
        state: &mut Glm5NextDsaState,
        pos_dev: Option<DevicePtr>,
        stream: u64,
    ) -> Result<()> {
        state.ensure_room(1)?;
        if pos_dev.is_some() {
            bail!(
                "DSA layer {}: a precomputed indexer row has no replay-safe placement",
                self.layer_idx
            );
        }
        let d = self.cfg.index_head_dim;
        let pos = state.len();
        let off = state.row_offset(pos);
        let (src_k, src_g) = (
            pre.k_normed.offset(row * d * 2),
            pre.gate.offset(row * d * 2),
        );
        gpu.copy_d2d_async(src_k, state.k_normed.offset(off), d * 2, stream)?;
        gpu.copy_d2d_async(src_g, state.gate.offset(off), d * 2, stream)?;
        self.store_indexer_row(gpu, state, None, pos, d, stream)
    }
}
