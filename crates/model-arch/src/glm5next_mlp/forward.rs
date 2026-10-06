// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GLM-5.3 MLP forward over `rows` rows: the dense SwiGLU FFN (also used for the
//! shared expert) and the routed MoE site.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `forward_dense` and `forward_moe` leave a partial sum in `out` when the site is split over
//!   TP or EP ranks; the caller all-reduces.
//! - `forward_moe` zeroes `expert_out` before any expert writes it, so an expert another EP rank
//!   owns adds zero.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::forward_prefill_gemm;
use super::weights::Glm5NextMoeWeights;
use super::{Glm5NextMlpConfig, Glm5NextMlpKernels};

mod dense;
mod launch;
mod moe_experts;
mod workspace;

use dense::row_slices;
pub use dense::{forward_dense, forward_dense_sliced};
use launch::{gemm, swiglu};
pub use workspace::{
    mlp_ws_bytes, mlp_ws_bytes_sized, mlp_ws_permute_bytes, mlp_ws_total_bytes,
    mlp_ws_total_bytes_sized,
};

const ACT_BLOCK: u32 = 256;

/// 2026-09-25: `out = silu(min(gate, limit)) * clamp(up, -limit, limit)` over `n` elements
/// (`glm5next_swiglu_clamp`), for the grouped-prefill module.
pub(super) fn swiglu_rows(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    n: usize,
    limit: f32,
    stream: u64,
) -> Result<()> {
    swiglu(gpu, k, gate, up, out, n, limit, stream)
}

/// 2026-09-25: MLP scratch for up to `max_rows` rows, allocated once at load. The buffer sizes
/// are listed in `mlp_ws_bytes`.
pub struct Glm5NextMlpWorkspace {
    /// 2026-09-25: Gate, up and activated values, BF16, each
    /// `max(rows * max_inter, rows * top_k * moe_intermediate)` elements.
    a_gate: DevicePtr,
    a_up: DevicePtr,
    a_act: DevicePtr,
    /// 2026-09-25: `[rows, num_experts]` F32 router logits.
    logits: DevicePtr,
    /// 2026-09-25: `[rows, top_k]` selected ids (I32); `wts` holds their F32 weights.
    ids: DevicePtr,
    wts: DevicePtr,
    /// 2026-09-25: `[rows, top_k, hidden]` BF16 routed outputs; `forward_moe` zeroes it before
    /// the experts run.
    expert_out: DevicePtr,
    /// 2026-09-25: `[rows, hidden]` BF16 shared-expert output.
    shared_out: DevicePtr,
    /// 2026-09-25: `[rows * top_k]` I32 union expert ids, `-1` for an unused entry. Row-batched
    /// path only.
    u_eid: DevicePtr,
    /// 2026-09-25: `[rows * top_k, rows]` I32: the slot row `r` gave union entry `u`, `-1` when
    /// row `r` did not select it.
    u_slot: DevicePtr,
    /// 2026-09-25: `[rows * top_k]` I32: sorted row to original token. Grouped GEMM only.
    sorted_token_ids: DevicePtr,
    /// 2026-09-25: `[rows * top_k]` I32: sorted row to expert id. Grouped GEMM only.
    sorted_expert_ids: DevicePtr,
    /// 2026-09-25: `[num_experts + 1]` I32 prefix sum over the expert-sorted rows.
    expert_offsets: DevicePtr,
    /// 2026-09-25: `[rows, top_k]` I32: a slot's row in the expert-sorted output, read by
    /// `glm5next_moe_combine_indexed`.
    token_to_perm: DevicePtr,
    /// 2026-09-30: `[rows * top_k, hidden]` BF16 gate/up gather-once buffer, `moe_permute_tokens`
    /// (`METRALE_GLM_MOE_PREFILL_PERMUTE=1`, `forward_prefill_gemm::prefill_gemm_permute`).
    /// `DevicePtr::NULL`, unallocated, when the lever is off — off leaves this workspace exactly
    /// the size it was before the lever existed.
    moe_perm: DevicePtr,
    max_inter: usize,
    /// 2026-09-25: Widest row group this scratch serves, at least 1.
    max_rows: usize,
    /// 2026-09-25: `max_rows * top_k`, the routed-slot count the grouped-path buffers hold.
    max_total_expanded: usize,
}

/// 2026-09-25: Whether the text layers share one MLP workspace: yes unless
/// `METRALE_GLM_MLP_WS_SHARED=0`. Read once.
pub fn mlp_ws_shared() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let shared = std::env::var("METRALE_GLM_MLP_WS_SHARED").as_deref() != Ok("0");
        if !shared {
            tracing::warn!(
                "GLM MLP workspace: PER-LAYER (METRALE_GLM_MLP_WS_SHARED=0) — the pre-P2 \
                 allocation, one scratch per layer"
            );
        }
        shared
    })
}

impl Glm5NextMlpWorkspace {
    /// 2026-09-25: Widest row group this scratch serves.
    /// 2026-10-02: `pub(crate)` for the batched verify's row cap (`steps/verify_multi.rs`).
    pub(crate) fn max_rows(&self) -> usize {
        self.max_rows
    }
    /// 2026-09-25: `max_rows * top_k`, the routed-slot count the grouped buffers hold.
    pub(super) fn max_total_expanded(&self) -> usize {
        self.max_total_expanded
    }
    pub(super) fn ids(&self) -> DevicePtr {
        self.ids
    }
    pub(super) fn a_gate(&self) -> DevicePtr {
        self.a_gate
    }
    pub(super) fn a_up(&self) -> DevicePtr {
        self.a_up
    }
    pub(super) fn a_act(&self) -> DevicePtr {
        self.a_act
    }
    pub(super) fn expert_out(&self) -> DevicePtr {
        self.expert_out
    }
    pub(super) fn sorted_token_ids(&self) -> DevicePtr {
        self.sorted_token_ids
    }
    pub(super) fn sorted_expert_ids(&self) -> DevicePtr {
        self.sorted_expert_ids
    }
    pub(super) fn expert_offsets(&self) -> DevicePtr {
        self.expert_offsets
    }
    pub(super) fn token_to_perm(&self) -> DevicePtr {
        self.token_to_perm
    }
    /// 2026-09-30: The `METRALE_GLM_MOE_PREFILL_PERMUTE=1` gather-once buffer; `DevicePtr::NULL`
    /// when the lever is off.
    pub(super) fn moe_perm(&self) -> DevicePtr {
        self.moe_perm
    }
    /// 2026-10-01: `u_eid` (`max_total_expanded` I32) lent to the grouped-W4A16 prefill's tile
    /// list; only the row-batched path, never taken in the same call, uses it otherwise.
    pub(super) fn prefill_tile_scratch(&self) -> DevicePtr {
        self.u_eid
    }
}

/// 2026-09-25: `METRALE_NO_GLM_MOE_ROW_BATCH=1` turns the row-batched routed path off; each row
/// then runs its own expert launches. Read once.
fn row_batch_disabled() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("METRALE_NO_GLM_MOE_ROW_BATCH").as_deref() == Ok("1"))
}

/// 2026-09-25: Widest sub-group of the row-batched routed path: `METRALE_GLM_MOE_ROW_BATCH_MAX`,
/// default `MOE_ROW_BATCH_MAX_ROWS`, clamped to `1..=MOE_ROW_BATCH_MAX_ROWS`. Read once.
pub fn row_batch_max() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let m = std::env::var("METRALE_GLM_MOE_ROW_BATCH_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(MOE_ROW_BATCH_MAX_ROWS)
            .clamp(1, MOE_ROW_BATCH_MAX_ROWS);
        if m != MOE_ROW_BATCH_MAX_ROWS {
            tracing::warn!(
                "GLM MoE row-batch width capped at {m} (default {MOE_ROW_BATCH_MAX_ROWS})"
            );
        }
        m
    })
}

/// 2026-09-25: Widest compiled `w4a16_gemv_sw_moe_batchm_mR` tier: the
/// `METRALE_MOE_BATCHM_ENTRY(2..=8)` instances in `kernels/gb10/common/w4a16_gemv.cu`, held at
/// index `R - 2` of `Glm5NextMlpKernels::w4a16_gemv_sw_moe_batchm`. `forward_moe` splits a wider
/// row group into sub-groups no wider than this.
pub const MOE_ROW_BATCH_MAX_ROWS: usize = 8;

/// 2026-09-25: Split `rows` into consecutive `(start, width)` sub-groups of at most `cap`, as
/// even as the count allows. There is no width-1 tier; at `cap = MOE_ROW_BATCH_MAX_ROWS` no
/// group of a split of 2 or more rows is one row wide. At a smaller `cap` one can be (3 rows at
/// cap 2), and `forward_moe` then does not take the row-batched path.
///
/// The split does not change a row's result: its routed output is the sum over its own top-k
/// slots, and each slot's GEMV accumulates in per-row registers, whatever other rows share the
/// sweep.
fn moe_row_groups(rows: usize, cap: usize) -> Vec<(usize, usize)> {
    let n = rows.div_ceil(cap.max(1)).max(1);
    let mut out = Vec::with_capacity(n);
    let mut start = 0usize;
    for i in 0..n {
        let w = (rows - start).div_ceil(n - i);
        out.push((start, w));
        start += w;
    }
    out
}

/// 2026-09-25: Most ids `glm5next_moe_row_union` resolves: it runs as one block of
/// `rows * top_k` threads, and `forward_moe` does not take the row-batched path above this.
pub const MOE_ROW_UNION_MAX_IDS: usize = 64;

/// 2026-09-25: Log once per row count (counts from 15 up share one bit) whether the routed
/// experts took the row-batched path.
fn announce_row_batch(batched: bool, rows: usize) {
    use std::sync::atomic::{AtomicU16, Ordering};
    static SEEN: AtomicU16 = AtomicU16::new(0);
    let bit = 1u16 << rows.min(15);
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
        return;
    }
    if batched {
        tracing::info!("GLM MoE: row-batched expert union ({rows} rows, one sweep each)");
    } else {
        tracing::info!("GLM MoE: per-row expert sweeps ({rows} rows)");
    }
}

/// 2026-09-25: Log once per row count (counts from 31 up share one bit) whether a row group
/// wider than `MOE_ROW_BATCH_MAX_ROWS` took the grouped GEMM; narrower groups log nothing.
fn announce_grouped_prefill(on: bool, rows: usize) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    if rows <= MOE_ROW_BATCH_MAX_ROWS {
        return;
    }
    let bit = 1u32 << (rows.min(31));
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
        return;
    }
    if on {
        let t = super::forward_prefill_gemm::gemm_tile();
        tracing::info!(
            "GLM MoE prefill: grouped tensor-core W4A16 GEMM, {rows} rows in ONE launch \
             per projection, tile `{}` (M_TILE {}, N_TILE {}, {} threads) \
             (METRALE_GLM_MOE_PREFILL_GEMM=0 to restore the GEMV path; \
             METRALE_GLM_MOE_GEMM_TILE=base for the prior tile)",
            t.name,
            t.m_tile,
            t.n_tile,
            t.threads
        );
    } else {
        tracing::info!(
            "GLM MoE prefill: row-batched GEMV, {rows} rows split into \
             {MOE_ROW_BATCH_MAX_ROWS}-row sweeps"
        );
    }
}

/// 2026-09-25: `METRALE_GLM_MOE_HOST_DISPATCH=1` forces the host-dispatch expert loop (ids read
/// back to the host, one GEMV per local expert) and turns off every device-dispatched routed
/// path. Read once.
pub(crate) fn host_dispatch_forced() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("METRALE_GLM_MOE_HOST_DISPATCH").as_deref() == Ok("1"))
}

/// 2026-09-25: Log, once per process, whether the per-row expert path is device-dispatched
/// (`w4a16_gemv_sw_moe`) or host-dispatched.
fn announce_dispatch(grouped: bool) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if grouped {
            tracing::info!("GLM MoE: grouped device dispatch (no per-layer D2H)");
        } else {
            tracing::info!("GLM MoE: host dispatch (per-layer stream sync + D2H)");
        }
    });
}

/// 2026-09-25: One routed MoE site over `rows` rows of `x` into `out`: router, top-k, routed
/// experts, shared expert, and the weighted combine.
///
/// The routed experts take one of three paths:
/// - the grouped GEMM (`forward_prefill_gemm`), for `rows > MOE_ROW_BATCH_MAX_ROWS` and
///   `rows >= prefill_gemm_min_rows()`;
/// - the row-batched union GEMV over the sub-groups of `moe_row_groups`, for `rows >= 2`;
/// - per row otherwise, device-dispatched (`w4a16_gemv_sw_moe`) or host-dispatched.
///
/// Route tracing (`METRALE_GLM_ROUTE_TRACE=1`) forces the per-row path, the one that reads `ids`
/// back. Errors when `rows` does not fit the workspace or the bound expert count is not this
/// rank's share.
#[allow(clippy::too_many_arguments)]
pub fn forward_moe(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    out: DevicePtr,
    rows: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    forward_moe_sliced(gpu, k, cfg, w, x, out, rows, rows, ws, stream)
}

/// 2026-09-29: `forward_moe` with the router GEMM and the shared expert's GEMMs issued once per
/// `row_slices(rows, dense_slice)` slice (each sees the M a separate call of that slice would);
/// top-k, the routed experts, the combine and everything else run over all `rows` at once.
/// `dense_slice >= rows` is exactly `forward_moe`.
#[allow(clippy::too_many_arguments)]
pub fn forward_moe_sliced(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    out: DevicePtr,
    rows: usize,
    dense_slice: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    if rows == 0 || rows > ws.max_rows {
        bail!(
            "GLM MoE: {rows} rows do not fit a workspace built for {}",
            ws.max_rows
        );
    }
    if w.experts.len() != cfg.local_experts {
        bail!(
            "GLM MoE: {} bound experts but this rank owns {} of {}",
            w.experts.len(),
            cfg.local_experts,
            cfg.num_experts
        );
    }

    use crate::glm5next_layer::profile;

    let groups = moe_row_groups(rows, row_batch_max());
    let grouped_prefill = forward_prefill_gemm::grouped_prefill_selected(k, cfg, ws, rows);
    let batched = !grouped_prefill
        && rows >= 2
        && !host_dispatch_forced()
        && !row_batch_disabled()
        && !profile::trace_on()
        && k.moe_row_union.0 != 0
        && groups.iter().all(|&(_, w)| {
            w >= 2
                && w * cfg.top_k <= MOE_ROW_UNION_MAX_IDS
                && k.w4a16_gemv_sw_moe_batchm[w - 2].0 != 0
        });
    announce_row_batch(batched, rows);
    announce_grouped_prefill(grouped_prefill, rows);

    let t = profile::start();
    for (a, n) in row_slices(rows, dense_slice) {
        let (xs, ls) = (
            x.offset(a * cfg.hidden * 2),
            ws.logits.offset(a * cfg.num_experts * 4),
        );
        if n > metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize
            && crate::glm5next_layer::cublas_wide_proj()
        {
            metrale_model_layers::layers::ops::cublas_bf16_proj_dense_f32_out(
                xs,
                w.router,
                ls,
                n as u32,
                cfg.num_experts as u32,
                cfg.hidden as u32,
                stream,
            )?;
        } else {
            for r in 0..n {
                gemm(
                    gpu,
                    k.gemm_f32,
                    k.gemv_f32,
                    KernelHandle(0),
                    xs.offset(r * cfg.hidden * 2),
                    w.router,
                    ls.offset(r * cfg.num_experts * 4),
                    1,
                    cfg.num_experts,
                    cfg.hidden,
                    stream,
                )?;
            }
        }
    }
    // 2026-09-25: One top-k launch for all rows: `glm5next_router_topk` handles row `blockIdx.x`.
    KernelLaunch::new(gpu, k.router)
        .grid([rows as u32, 1, 1])
        .block([ACT_BLOCK, 1, 1])
        .arg_ptr(ws.logits)
        .arg_ptr(w.router_bias)
        .arg_ptr(ws.ids)
        .arg_ptr(ws.wts)
        .arg_u32(cfg.num_experts as u32)
        .arg_u32(cfg.top_k as u32)
        .arg_u32(1)
        .arg_f32(cfg.routed_scale)
        .arg_u32(u32::from(cfg.renormalize))
        .arg_u32(u32::from(cfg.router_bf16_ladder))
        .launch(stream)?;
    profile::end(profile::MOE_ROUTER, t, gpu, stream);

    // 2026-09-25: Zero every routed output first: an expert this rank does not own writes
    // nothing, and its slots must add zero to the sum.
    // 2026-10-06: `METRALE_GLM_MOE_COMBINE_LOCAL=1` on the grouped prefill: no zeroing; the
    // combine below skips those slots instead (`glm5next_moe_combine_indexed_local`).
    let combine_local = grouped_prefill
        && k.combine_indexed_local.0 != 0
        && metrale_config::glm_moe_combine_local();
    if !combine_local {
        gpu.memset_async(ws.expert_out, 0, rows * cfg.top_k * cfg.hidden * 2, stream)?;
    }

    let site = moe_experts::MoeSite {
        gpu,
        k,
        cfg,
        w,
        x,
        rows,
        ws,
        stream,
    };
    moe_experts::per_row_experts(&site, batched, grouped_prefill)?;

    if grouped_prefill {
        // 2026-09-25: Leaves the routed outputs in expert-sorted order; the combine below reads
        // them through `token_to_perm`.
        let t = profile::start();
        forward_prefill_gemm::forward_moe_grouped_prefill(gpu, k, cfg, w, x, rows, ws, stream)?;
        profile::end(profile::MOE_EXPERTS, t, gpu, stream);
    }

    if batched {
        moe_experts::row_batched_experts(&site, &groups)?;
    }

    let t = profile::start();
    forward_dense_sliced(
        gpu,
        k,
        cfg,
        &w.shared,
        cfg.local_shared_intermediate,
        x,
        ws.shared_out,
        rows,
        dense_slice,
        ws,
        stream,
    )?;

    // 2026-09-25: The shared expert joins the routed sum in the combine, before the caller's
    // all-reduce, so one collective reduces both partial sums.
    profile::end(profile::MOE_SHARED, t, gpu, stream);
    let t = profile::start();
    // 2026-09-25: One combine launch for all rows (row = `blockIdx.x`). After the grouped GEMM
    // the routed rows are in expert-sorted order, so that path uses
    // `glm5next_moe_combine_indexed`, which finds each slot's row through `token_to_perm`.
    if combine_local {
        let local = cfg.local_expert_range();
        super::log_combine_local(&local, rows);
        KernelLaunch::new(gpu, k.combine_indexed_local)
            .grid([rows as u32, 1, 1])
            .block([ACT_BLOCK, 1, 1])
            .arg_ptr(ws.expert_out)
            .arg_ptr(ws.token_to_perm)
            .arg_ptr(ws.wts)
            .arg_ptr(ws.shared_out)
            .arg_ptr(out)
            .arg_u32(cfg.hidden as u32)
            .arg_u32(cfg.top_k as u32)
            .arg_ptr(ws.expert_offsets())
            .arg_u32(local.start as u32)
            .arg_u32(local.end as u32)
            .launch(stream)?;
    } else if grouped_prefill {
        KernelLaunch::new(gpu, k.combine_indexed)
            .grid([rows as u32, 1, 1])
            .block([ACT_BLOCK, 1, 1])
            .arg_ptr(ws.expert_out)
            .arg_ptr(ws.token_to_perm)
            .arg_ptr(ws.wts)
            .arg_ptr(ws.shared_out)
            .arg_ptr(out)
            .arg_u32(cfg.hidden as u32)
            .arg_u32(cfg.top_k as u32)
            .launch(stream)?;
    } else {
        KernelLaunch::new(gpu, k.combine)
            .grid([rows as u32, 1, 1])
            .block([ACT_BLOCK, 1, 1])
            .arg_ptr(ws.expert_out)
            .arg_ptr(ws.wts)
            .arg_ptr(ws.shared_out)
            .arg_ptr(out)
            .arg_u32(cfg.hidden as u32)
            .arg_u32(cfg.top_k as u32)
            .launch(stream)?;
    }
    profile::end(profile::MOE_COMBINE, t, gpu, stream);
    Ok(())
}

#[cfg(test)]
mod tests;
