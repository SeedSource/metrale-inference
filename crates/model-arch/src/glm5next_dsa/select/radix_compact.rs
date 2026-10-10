// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_GLM_DSA_TOPK_RADIX_COMPACT=1` (default off; needs
//! `METRALE_GLM_DSA_TOPK_RADIX=1`): an exact shortcut inside the radix top-k
//! ([`super::radix`]). After find pass 0, a row whose pass-0 threshold bin b0 holds at most
//! [`COMPACT_CAP`] keys (COMPACT) is finished by one more read of the row
//! (`dsa_topk_radix_compact`: keys above b0 to the candidate slots, the pairs of b0 to a per-row
//! buffer) and a sort of that bin in shared memory (`dsa_topk_radix_select0`), instead of hist
//! and find passes 1 and 2 and the gather (three more reads of the row). Rows above the cap
//! (FALLBACK) run the lever-off passes unchanged: those launches go through `_rc` entry points,
//! the same code plus an early return for a COMPACT row.
//!
//! Why: measured 2026-10-10 on GB10 at 131K (P = 32,768 pools, Q = 128 rows, one chunk a row):
//! hist 1 + hist 2 + gather = 105 + 94 + 124 us a call, and the compaction pass 0.88-0.93x the
//! gather. In a 524K serve the b0 bin held a median of at most 1% of P (p95 <= 1,487); 3 of 128
//! rows in 1 of 37 samples exceeded 2,048 keys.
//!
//! Output contract (exact): `selected` is bit-identical to the lever-off radix path (itself
//! byte-identical to `dsa_topk_pools`); the argument is in the `.cu` comment above
//! `dsa_topk_radix_binstat`, and `radix_compact_tests` runs a host model of the two kernels
//! against the full order.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`COMPACT_CAP`], [`RS_BINCNT`] and [`RS_MODE`] equal `DSA_COMPACT_CAP`, `DSA_RS_BINCNT` and
//!   `DSA_RS_MODE` in `dsa_indexer.cu` (checked by a test).
//! - The pair buffer is the tail of select scratch region 6, after the radix work buffer of the
//!   pass's rows ([`compact_bytes`] more, planned only with this lever on): no allocation per
//!   call. A scratch without it, or a module without one of the five entry points, keeps the
//!   lever-off radix sequence, with one warning.
//! - Ceiling (graph-replay) launches engage too: every kernel reads P and select_k from `geom`
//!   exactly as the lever-off ones do, and the grids and buffer depend on the row count only.
//! - Lever off: no bytes planned, the buffer pointer is NULL and `launch_topk_radix` launches
//!   exactly what it did before.

use std::sync::OnceLock;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::radix::RADIX_THREADS;
use crate::glm5next_dsa::Glm5NextDsaKernels;

/// 2026-10-10: Most keys a COMPACT row's bin b0 may hold (`DSA_COMPACT_CAP`): the pairs
/// `dsa_topk_radix_select0` sorts in one block's shared memory (16 KiB).
pub const COMPACT_CAP: usize = 2048;
/// 2026-10-10: State word offsets (`DSA_RS_BINCNT`: pairs appended to the row's buffer;
/// `DSA_RS_MODE`: 1 once select0 finished the row).
pub const RS_BINCNT: usize = 6;
pub const RS_MODE: usize = 7;
/// 2026-10-10: Logged once, on the first radix call that takes the compaction.
pub const RADIX_COMPACT_ENGAGED_LINE: &str = "METRALE_GLM_DSA_TOPK_RADIX_COMPACT ENGAGED";

/// 2026-10-10: `1` (blanks ignored) is on; unset, `0` and anything else are off.
pub fn parse_radix_compact(v: Option<&str>) -> bool {
    crate::glm5next_layer::levers::parse_dsa_switch(v)
}

/// 2026-10-10: `METRALE_GLM_DSA_TOPK_RADIX_COMPACT`, read once; a value other than unset,
/// empty, `0` or `1` is off, with one warning.
pub fn dsa_topk_radix_compact() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_TOPK_RADIX_COMPACT").ok();
        let odd = |r: &&str| !r.is_empty() && r.trim() != "0" && r.trim() != "1";
        if let Some(r) = raw.as_deref().filter(odd) {
            tracing::warn!("METRALE_GLM_DSA_TOPK_RADIX_COMPACT={r} is not 0 or 1 - treated as off");
        }
        parse_radix_compact(raw.as_deref())
    })
}

/// 2026-10-10: Bytes of the pair buffer for `q_rows` rows: [`COMPACT_CAP`] u32 keys then as
/// many u32 pool indices a row (16 KiB a row; 2 MiB at a 128-row prefill pass).
pub fn compact_bytes(q_rows: usize) -> usize {
    q_rows * COMPACT_CAP * 8
}

/// 2026-10-10: Whether the five entry points the compaction launches resolved.
pub fn radix_compact_resolved(k: &Glm5NextDsaKernels) -> bool {
    [
        k.topk_radix_compact,
        k.topk_radix_select0,
        k.topk_radix_hist_rc,
        k.topk_radix_find_rc,
        k.topk_radix_gather_rc,
    ]
    .iter()
    .all(|h| h.0 != 0)
}

/// 2026-10-10: Whether a radix call (already engaged) takes the compaction: the lever on, the
/// entry points resolved, and region 6 holding the work buffers (`work_bytes`) plus the pair
/// buffer of the pass's rows.
pub fn radix_compact_for(
    requested: bool,
    resolved: bool,
    region_bytes: usize,
    work_bytes: usize,
    q_rows: usize,
) -> bool {
    requested && resolved && region_bytes >= work_bytes + compact_bytes(q_rows)
}

/// 2026-10-10: Log once that the lever engaged, and once why a radix call with it on did not.
pub(super) fn log_radix_compact(requested: bool, engaged: bool) {
    if engaged {
        static ENGAGED: OnceLock<()> = OnceLock::new();
        ENGAGED.get_or_init(|| {
            tracing::warn!(
                "{RADIX_COMPACT_ENGAGED_LINE} - radix rows whose pass-0 threshold bin holds at \
                 most {COMPACT_CAP} keys are finished by dsa_topk_radix_compact + \
                 dsa_topk_radix_select0; the others run passes 1-2 and gather (exact either way)"
            );
        });
    } else if requested {
        static FELL_BACK: OnceLock<()> = OnceLock::new();
        FELL_BACK.get_or_init(|| {
            tracing::warn!(
                "METRALE_GLM_DSA_TOPK_RADIX_COMPACT=1: NOT engaged on this radix call, the \
                 lever-off radix sequence runs (an unresolved compact/select0/_rc entry point, \
                 or a select scratch planned without the pair buffer)"
            );
        });
    }
}

/// 2026-10-10: Launch `dsa_topk_radix_compact` (grid `chunks` x `rows`, as the gather).
pub(super) fn launch_compact(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    scores: DevicePtr,
    work: DevicePtr,
    bins: DevicePtr,
    dims: [u32; 5],
    geom_dev: DevicePtr,
    stream: u64,
) -> Result<()> {
    let [chunks, rows, p, selk, kcap] = dims;
    KernelLaunch::new(gpu, kernels.topk_radix_compact)
        .grid([chunks, rows, 1])
        .block([RADIX_THREADS, 1, 1])
        .arg_ptr(scores)
        .arg_ptr(work)
        .arg_ptr(bins)
        .arg_u32(p)
        .arg_u32(selk)
        .arg_u32(kcap)
        .arg_ptr(geom_dev)
        .launch(stream)
}

/// 2026-10-10: Launch `dsa_topk_radix_select0` (grid `rows`).
pub(super) fn launch_select0(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    work: DevicePtr,
    bins: DevicePtr,
    rows: u32,
    select_k: u32,
    kcap: u32,
    geom_dev: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.topk_radix_select0)
        .grid([rows, 1, 1])
        .block([RADIX_THREADS, 1, 1])
        .arg_ptr(work)
        .arg_ptr(bins)
        .arg_u32(select_k)
        .arg_u32(kcap)
        .arg_ptr(geom_dev)
        .launch(stream)
}

#[cfg(test)]
#[path = "radix_compact_tests.rs"]
mod tests;
