// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`Glm5NextDsaWorkspace`], the per-layer scratch `decode_k` reuses, with the
//! two switches that decide whether its batched-selector buffers exist and are used
//! ([`dsa_select_rows_enabled`], [`batch_select_enabled`]).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - The batched-selector buffers are NULL exactly when [`dsa_select_rows_enabled`] is off,
//!   and `bt`/`sl` exactly when `METRALE_GLM_DSA_ALLOC_PER_STEP=1`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::Glm5NextDsaConfig;
use super::super::select::DsaSelectScratch;

/// 2026-09-25: Whether one selection pass covers all `k` rows (`select_rows_batched`) or each
/// row selects on its own (`select_row`). All four terms must hold:
///
/// * `workspace_ready`: the `[max_rows]` selector buffers exist. [`dsa_select_rows_enabled`]
///   off (`METRALE_DSA_SELECT_ROWS=0`) leaves them unallocated.
/// * `is_prefill`: stated by the caller, because `ForwardContext` does not separate a
///   prefill from a speculative verify: both have `decode_step` false, and `verify_a` sets
///   `graph_capture` false.
/// * `!graph_capture`: `select_rows_batched` copies `q_pos` host-to-device, which a
///   capturing stream does not allow.
/// * `k > 1`.
///
/// One pass over the group selects, for each row, the pools a single-row pass at that row's
/// length would. `dsa_index_scores` makes pool `p` a candidate for row `r` only when the
/// pool is complete and its last token is valid and at or before `q_pos[r]`, so the extra
/// pools of the group's larger `P` score `-FLT_MAX` for row `r`; `dsa_topk_pools` orders by
/// score, then pool index, so the top `select_k` is unique; and `dsa_expand_selection`
/// clamps `select_k` to the row's own pool count, so the row's tail lands in the slots a
/// single-row pass uses (the kernel's comment carries the measurement behind that clamp).
pub(crate) fn batch_select_enabled(
    workspace_ready: bool,
    is_prefill: bool,
    graph_capture: bool,
    k: usize,
) -> bool {
    workspace_ready && is_prefill && !graph_capture && k > 1
}

/// 2026-09-25: `METRALE_DSA_SELECT_ROWS`: on unless set to `0`. When off,
/// `Glm5NextDsaWorkspace::new` does not allocate the batched-selector buffers.
pub(crate) fn dsa_select_rows_enabled() -> bool {
    std::env::var("METRALE_DSA_SELECT_ROWS").as_deref() != Ok("0")
}

/// 2026-09-25: Scratch reused across `decode_k` calls; each layer owns one.
pub struct Glm5NextDsaWorkspace {
    pub(super) q_a: DevicePtr,
    pub(super) q_resid: DevicePtr,
    pub(super) q_abs: DevicePtr,
    pub(super) kv_a: DevicePtr,
    pub(super) q_idx: DevicePtr,
    pub(super) head_weights: DevicePtr,
    pub(super) q_pos: DevicePtr,
    pub(super) q_mask: DevicePtr,
    /// 2026-09-25: `[max_rows, ...]` copies of `q_idx` / `head_weights` / `q_pos` / `q_mask`
    /// for the batched prefill selector. NULL when [`dsa_select_rows_enabled`] is off.
    pub(super) q_idx_rows: DevicePtr,
    pub(super) head_weights_rows: DevicePtr,
    pub(super) q_pos_rows: DevicePtr,
    pub(super) q_mask_rows: DevicePtr,
    pub(super) slot: DevicePtr,
    pub(super) attn_out: DevicePtr,
    /// 2026-09-25: Block table for the paged gather, `bt_cap` i32 entries. NULL when
    /// `METRALE_GLM_DSA_ALLOC_PER_STEP=1`; `decode_k` then allocates one per call.
    pub(super) bt: DevicePtr,
    /// 2026-09-25: `seq_lens` for the paged gather, `[max_rows]` i32; NULL under the same
    /// setting as `bt`.
    pub(super) sl: DevicePtr,
    /// 2026-09-25: Capacity of `bt` in entries; `decode_k` refuses a larger upload.
    pub(super) bt_cap: usize,
    /// 2026-09-25: The largest `k` `decode_k` accepts.
    pub(super) max_rows: usize,
    /// 2026-09-25: `[index_head_dim]` BF16 staging rows at fixed addresses (`stage_k`,
    /// `stage_gate`). On the replay-safe path the indexer projections write here and
    /// `dsa_indexer_store` copies them to the row at a device-side position, which a
    /// replayed graph reads live.
    pub(super) stage_k: DevicePtr,
    pub(super) stage_gate: DevicePtr,
    /// 2026-09-25: `[5]` i32 selector geometry, written on the device by `dsa_write_geom`
    /// for each row on the replay-safe path.
    pub(super) geom_dev: DevicePtr,
    pub(super) select: DsaSelectScratch,
    /// 2026-10-01: `METRALE_GLM_DSA_ROW_BATCH` staging (`row_batch.rs`); `None` unless the
    /// lever is on, `bt`/`sl` persist, the batched selector is on and `max_rows > 1`.
    pub(super) row_batch: Option<super::row_batch::DsaRowBatch>,
    /// 2026-10-01: `METRALE_GLM_PREFILL_FULLWIDTH_GEMM` scratch for `decode_k_wide`
    /// (`wide.rs`), one arena shared by every DSA layer; `None` unless the loader attached it
    /// with [`Glm5NextDsaWorkspace::with_wide`].
    pub(super) wide: Option<std::sync::Arc<super::wide::DsaWideArena>>,
}

impl Glm5NextDsaWorkspace {
    /// 2026-09-25: `max_rows` (at least 1) is the largest `k` this workspace serves. Sized by
    /// it: the projection outputs, `attn_out`, `sl`, the batched-selector buffers, and the
    /// selection scratch, which is planned at [`super::super::state::max_dsa_context`] tokens and
    /// `max_rows` query rows. The staging rows and `bt` do not depend on it.
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextDsaConfig, max_rows: usize) -> Result<Self> {
        let rows = max_rows.max(1);
        let geom = super::super::select::DsaSelectGeometry::plan(
            cfg,
            super::super::state::max_dsa_context(cfg),
            rows,
        )?;
        let bt_cap = super::super::state::max_dsa_context(cfg).max(1);
        let persist = std::env::var("METRALE_GLM_DSA_ALLOC_PER_STEP").as_deref() != Ok("1");
        let batch_select = dsa_select_rows_enabled();
        // 2026-10-01: The row batch needs the persistent `bt`/`sl` and the batched selector.
        let row_batch = persist && batch_select && rows > 1;
        let row_batch = row_batch && crate::glm5next_layer::dsa_row_batch();
        Ok(Self {
            q_a: gpu.alloc(rows * (cfg.q_lora_rank * 2))?,
            q_resid: gpu.alloc(rows * (cfg.q_lora_rank * 2))?,
            q_abs: gpu.alloc(rows * (cfg.local_heads * cfg.kv_lora_rank * 2))?,
            kv_a: gpu.alloc(rows * (cfg.kv_lora_rank * 2))?,
            q_idx: gpu.alloc(cfg.index_heads * cfg.index_head_dim * 4)?,
            head_weights: gpu.alloc(cfg.index_heads * 4)?,
            q_pos: gpu.alloc(4)?,
            q_idx_rows: if batch_select {
                gpu.alloc(rows * cfg.index_heads * cfg.index_head_dim * 4)?
            } else {
                DevicePtr(0)
            },
            head_weights_rows: if batch_select {
                gpu.alloc(rows * cfg.index_heads * 4)?
            } else {
                DevicePtr(0)
            },
            q_pos_rows: if batch_select {
                gpu.alloc(rows * 4)?
            } else {
                DevicePtr(0)
            },
            q_mask_rows: if batch_select {
                // 2026-09-25: All 1s (every row is a real query), set once here.
                let p = gpu.alloc(rows)?;
                gpu.memset_async(p, 1, rows, 0)?;
                gpu.synchronize(0)?;
                p
            } else {
                DevicePtr(0)
            },
            q_mask: {
                // 2026-09-25: Always 1 (one real query), set once here so no step writes it.
                let p = gpu.alloc(1)?;
                gpu.memset_async(p, 1, 1, 0)?;
                gpu.synchronize(0)?;
                p
            },
            slot: gpu.alloc(8)?,
            attn_out: gpu.alloc(rows * (cfg.local_heads * cfg.kv_lora_rank * 2))?,
            bt: if persist {
                gpu.alloc(bt_cap * 4)?
            } else {
                DevicePtr(0)
            },
            // 2026-09-25: `[rows]`: `attend_rows` launches all rows at once and
            // `glm5next_dsa_mla_decode_fp8` reads `seq_lens[blockIdx.y]`.
            sl: if persist {
                gpu.alloc(rows * 4)?
            } else {
                DevicePtr(0)
            },
            bt_cap,
            max_rows: rows,
            stage_k: gpu.alloc(cfg.index_head_dim * 2)?,
            stage_gate: gpu.alloc(cfg.index_head_dim * 2)?,
            geom_dev: gpu.alloc(5 * 4)?,
            select: DsaSelectScratch::alloc(gpu, cfg, &geom)?,
            row_batch: if row_batch {
                Some(super::row_batch::DsaRowBatch::new(gpu, rows, bt_cap)?)
            } else {
                None
            },
            wide: None,
        })
    }

    /// 2026-10-01: This workspace with the shared full-width arena attached, so `decode_k_wide`
    /// can run its projections over a whole prefill window (`METRALE_GLM_PREFILL_FULLWIDTH_GEMM`).
    /// The arena is shared: the layers run one after another on one stream.
    pub fn with_wide(mut self, arena: std::sync::Arc<super::wide::DsaWideArena>) -> Self {
        self.wide = Some(arena);
        self
    }
}
