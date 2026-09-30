// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The byte sizes of the GLM-5.3 MLP scratch buffers and
//! `Glm5NextMlpWorkspace::new`, which allocates them.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `mlp_ws_bytes` lists the sizes `Glm5NextMlpWorkspace::new` allocates, in its order.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextMlpWorkspace;
use crate::glm5next_mlp::Glm5NextMlpConfig;

/// 2026-09-25: Byte size of each `Glm5NextMlpWorkspace` buffer, in the order `new` allocates
/// them: a_gate, a_up, a_act, logits, ids, wts, expert_out, shared_out, u_eid, u_slot,
/// sorted_token_ids, sorted_expert_ids, expert_offsets, token_to_perm. The loader logs their
/// sum. `new` computes the same sizes on its own, so the two must change together.
pub fn mlp_ws_bytes(cfg: &Glm5NextMlpConfig, max_rows: usize) -> [usize; 14] {
    mlp_ws_bytes_sized(cfg, max_rows, max_rows)
}

/// 2026-09-29: `mlp_ws_bytes` with `u_slot` sized for `union_rows` rows (clamped to
/// `1..=max_rows`) instead of `max_rows`. `u_slot` is `[w * top_k, w]` for a row-batched group
/// of `w <= MOE_ROW_BATCH_MAX_ROWS` rows, reused from offset 0 by every group
/// (`row_batched_experts`), so it never needs `max_rows` rows; its `max_rows^2` term is what a
/// wide staged-prefill workspace would otherwise pay.
pub fn mlp_ws_bytes_sized(
    cfg: &Glm5NextMlpConfig,
    max_rows: usize,
    union_rows: usize,
) -> [usize; 14] {
    let rows = max_rows.max(1);
    let ur = union_rows.clamp(1, rows);
    let max_inter = cfg
        .local_dense_intermediate
        .max(cfg.moe_intermediate)
        .max(cfg.local_shared_intermediate)
        .max(1);
    let act_elems = (rows * max_inter)
        .max(rows * cfg.top_k * cfg.moe_intermediate)
        .max(1);
    [
        act_elems * 2,
        act_elems * 2,
        act_elems * 2,
        rows * cfg.num_experts * 4,
        rows * cfg.top_k * 4,
        rows * cfg.top_k * 4,
        rows * cfg.top_k * cfg.hidden * 2,
        rows * cfg.hidden * 2,
        rows * cfg.top_k * 4,
        ur * cfg.top_k * ur * 4,
        rows * cfg.top_k * 4,
        rows * cfg.top_k * 4,
        (cfg.num_experts + 1) * 4,
        rows * cfg.top_k * 4,
    ]
}

/// 2026-09-25: Total device bytes of one MLP workspace of `max_rows` rows.
pub fn mlp_ws_total_bytes(cfg: &Glm5NextMlpConfig, max_rows: usize) -> usize {
    mlp_ws_bytes(cfg, max_rows).iter().sum()
}

/// 2026-09-29: Total device bytes of `Glm5NextMlpWorkspace::new_sized(_, cfg, max_rows,
/// union_rows)`.
pub fn mlp_ws_total_bytes_sized(
    cfg: &Glm5NextMlpConfig,
    max_rows: usize,
    union_rows: usize,
) -> usize {
    mlp_ws_bytes_sized(cfg, max_rows, union_rows).iter().sum()
}

/// 2026-09-30: Bytes of the `METRALE_GLM_MOE_PREFILL_PERMUTE=1` gather-once buffer
/// (`Glm5NextMlpWorkspace::moe_perm`): `[max_rows * top_k, hidden]` BF16 when `on`, 0 (no
/// allocation) when `on` is false — the caller passes `forward_prefill_gemm::prefill_gemm_permute()`
/// explicitly so this stays a pure function of its arguments, not of process env state. Kept out
/// of `mlp_ws_bytes`/`mlp_ws_total_bytes*`: those describe the workspace the lever leaves
/// unchanged when off, and their hand-computed-footprint tests assume every entry is always
/// positive.
pub fn mlp_ws_permute_bytes(cfg: &Glm5NextMlpConfig, max_rows: usize, on: bool) -> usize {
    if !on {
        return 0;
    }
    max_rows.max(1) * cfg.top_k * cfg.hidden * 2
}

impl Glm5NextMlpWorkspace {
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextMlpConfig, max_rows: usize) -> Result<Self> {
        Self::new_sized(gpu, cfg, max_rows, max_rows)
    }

    /// 2026-09-29: `new` with `u_slot` sized for `union_rows` rows (`mlp_ws_bytes_sized`).
    pub fn new_sized(
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextMlpConfig,
        max_rows: usize,
        union_rows: usize,
    ) -> Result<Self> {
        let rows = max_rows.max(1);
        let ur = union_rows.clamp(1, rows);
        let max_inter = cfg
            .local_dense_intermediate
            .max(cfg.moe_intermediate)
            .max(cfg.local_shared_intermediate)
            .max(1);
        // 2026-09-25: The activation buffers hold the widest of a dense or shared-expert pass
        // (`rows * max_inter`) and a routed pass over every (row, slot)
        // (`rows * top_k * moe_intermediate`).
        let act_elems = (rows * max_inter)
            .max(rows * cfg.top_k * cfg.moe_intermediate)
            .max(1);
        // 2026-09-30: `METRALE_GLM_MOE_PREFILL_PERMUTE=1` gather-once buffer for gate/up
        // (`forward_prefill_gemm::prefill_gemm_permute`). Off (default): `DevicePtr::NULL`, no
        // `gpu.alloc` call at all — the lever changes nothing about this workspace when unset.
        let permute_on = crate::glm5next_mlp::forward_prefill_gemm::prefill_gemm_permute();
        let moe_perm_bytes = mlp_ws_permute_bytes(cfg, rows, permute_on);
        let moe_perm = if moe_perm_bytes > 0 {
            gpu.alloc(moe_perm_bytes)?
        } else {
            DevicePtr::NULL
        };
        Ok(Self {
            a_gate: gpu.alloc(act_elems * 2)?,
            a_up: gpu.alloc(act_elems * 2)?,
            a_act: gpu.alloc(act_elems * 2)?,
            logits: gpu.alloc(rows * cfg.num_experts * 4)?,
            ids: gpu.alloc(rows * cfg.top_k * 4)?,
            wts: gpu.alloc(rows * cfg.top_k * 4)?,
            expert_out: gpu.alloc(rows * cfg.top_k * cfg.hidden * 2)?,
            shared_out: gpu.alloc(rows * cfg.hidden * 2)?,
            u_eid: gpu.alloc(rows * cfg.top_k * 4)?,
            u_slot: gpu.alloc(ur * cfg.top_k * ur * 4)?,
            // 2026-09-25: The grouped-GEMM routing tables are allocated whatever the grouped
            // levers say.
            sorted_token_ids: gpu.alloc(rows * cfg.top_k * 4)?,
            sorted_expert_ids: gpu.alloc(rows * cfg.top_k * 4)?,
            expert_offsets: gpu.alloc((cfg.num_experts + 1) * 4)?,
            token_to_perm: gpu.alloc(rows * cfg.top_k * 4)?,
            moe_perm,
            max_inter,
            max_rows: rows,
            max_total_expanded: rows * cfg.top_k,
        })
    }
}
