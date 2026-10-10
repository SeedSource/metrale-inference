// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_GLM_DSA_RADIX_BINSTAT=1` (default off, DEBUG): gate-0 statistics for the
//! planned radix compaction redesign. After find pass 0 of a radix top-k call, row r's
//! threshold bin b0 = `prefix >> 20` holds `hist0[b0]` keys; this lever reports how many.
//!
//! How: `dsa_topk_radix_binstat` (grid Q, one thread a row, launched between find 0 and hist 1)
//! copies `hist0[b0]` and the live P into two state slots of the row's work buffer
//! (`DSA_RS_B0CNT`, `DSA_RS_BINP`; slots 4 and 5 of the 16 state words, which no other kernel
//! reads and which init zeroes). Every Nth sampled call (`METRALE_GLM_DSA_RADIX_BINSTAT_EVERY`,
//! default 64; the first radix call is sampled) the host synchronises the stream after the last
//! launch, reads each row's state words (one small D2H a row) and logs one line:
//! `DSA BINSTAT P=<P> rows=<Q> b0_median=<n> (<pct>% of P) b0_p95=<n> b0_max=<n> over2048=<rows>`.
//!
//! Cannot change an output byte: the kernel only writes two state slots nothing reads, only on
//! sampled calls, and the host reads happen after the whole radix sequence. The sync and D2H are
//! skipped (with the kernel) while the stream is capturing, so a graph captured with the lever
//! on is the lever-off graph; use an EAGER profiling serve (no graph replay) to get samples.
//!
//! Owner: model-arch (GLM-5.3 DSA). Invariants: the slot numbers equal `DSA_RS_B0CNT` /
//! `DSA_RS_BINP` in `dsa_indexer.cu` (checked by a test).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::radix::{RADIX_STATE, RADIX_STATE_WORDS, radix_row_bytes};
use crate::glm5next_dsa::Glm5NextDsaKernels;

/// 2026-10-10: Offsets within the state words (`DSA_RS_B0CNT`, `DSA_RS_BINP`).
pub const RS_B0CNT: usize = 4;
pub const RS_BINP: usize = 5;
/// 2026-10-10: Default sampling period, in radix calls.
pub const BINSTAT_EVERY_DEFAULT: u64 = 64;
/// 2026-10-10: A bin above this many keys cannot be sorted in one block's shared arrays.
pub const BINSTAT_CAP: u32 = 2048;

/// 2026-10-10: `1` (blanks ignored) is on; unset, `0` and anything else are off.
pub fn parse_binstat(v: Option<&str>) -> bool {
    crate::glm5next_layer::levers::parse_dsa_switch(v)
}

/// 2026-10-10: The sampling period from `METRALE_GLM_DSA_RADIX_BINSTAT_EVERY`: a positive
/// integer; unset, blank, `0` or unparsable gives [`BINSTAT_EVERY_DEFAULT`].
pub fn parse_binstat_every(v: Option<&str>) -> u64 {
    v.and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(BINSTAT_EVERY_DEFAULT)
}

/// 2026-10-10: `METRALE_GLM_DSA_RADIX_BINSTAT`, read once.
pub fn dsa_radix_binstat() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_RADIX_BINSTAT").ok();
        let odd = |r: &&str| !r.is_empty() && r.trim() != "0" && r.trim() != "1";
        if let Some(r) = raw.as_deref().filter(odd) {
            tracing::warn!("METRALE_GLM_DSA_RADIX_BINSTAT={r} is not 0 or 1 - treated as off");
        }
        parse_binstat(raw.as_deref())
    })
}

/// 2026-10-10: `METRALE_GLM_DSA_RADIX_BINSTAT_EVERY`, read once.
pub fn dsa_radix_binstat_every() -> u64 {
    static E: OnceLock<u64> = OnceLock::new();
    *E.get_or_init(|| {
        parse_binstat_every(
            std::env::var("METRALE_GLM_DSA_RADIX_BINSTAT_EVERY")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-10-10: Whether the call numbered `n` (0-based, counted over eager radix calls) is
/// sampled: calls 0, every, 2 * every, ...
pub fn is_sampled(n: u64, every: u64) -> bool {
    n % every.max(1) == 0
}

/// 2026-10-10: Nearest-rank percentile of an ascending slice: the value at 1-based rank
/// `ceil(pct / 100 * n)`, clamped to the slice; 0 for an empty slice.
pub fn percentile_sorted(sorted: &[u32], pct: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((pct / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// 2026-10-10: Summary of one sampled call.
#[derive(Debug, Clone, PartialEq)]
pub struct BinStat {
    pub p: u32,
    pub rows: usize,
    pub median: u32,
    pub p95: u32,
    pub max: u32,
    pub over_cap: usize,
}

/// 2026-10-10: Summarise per-row `(b0 count, P)` pairs; rows whose P is 0 (select_k 0) are
/// skipped. `None` when no row counts. P is the largest row P (all rows of a call share it).
pub fn summarize(rows: &[(u32, u32)]) -> Option<BinStat> {
    let mut v: Vec<u32> = rows.iter().filter(|r| r.1 != 0).map(|r| r.0).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    let p = rows.iter().map(|r| r.1).max().unwrap_or(0);
    Some(BinStat {
        p,
        rows: v.len(),
        median: percentile_sorted(&v, 50.0),
        p95: percentile_sorted(&v, 95.0),
        max: *v.last().unwrap_or(&0),
        over_cap: v.iter().filter(|&&c| c > BINSTAT_CAP).count(),
    })
}

/// 2026-10-10: The log line.
pub fn format_line(s: &BinStat) -> String {
    let pct = 100.0 * f64::from(s.median) / f64::from(s.p.max(1));
    format!(
        "DSA BINSTAT P={} rows={} b0_median={} ({pct:.3}% of P) b0_p95={} b0_max={} over{BINSTAT_CAP}={}",
        s.p, s.rows, s.median, s.p95, s.max, s.over_cap
    )
}

/// 2026-10-10: Per-call decision made at the start of a radix call.
pub fn begin_call(gpu: &dyn GpuBackend, kernels: &Glm5NextDsaKernels, stream: u64) -> bool {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    if !dsa_radix_binstat() || kernels.topk_radix_binstat.0 == 0 || gpu.stream_is_capturing(stream)
    {
        return false;
    }
    is_sampled(
        CALLS.fetch_add(1, Ordering::Relaxed),
        dsa_radix_binstat_every(),
    )
}

/// 2026-10-10: Launch `dsa_topk_radix_binstat` for `rows` rows (between find 0 and hist 1).
pub fn launch_binstat(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    work: DevicePtr,
    rows: u32,
    p: u32,
    select_k: u32,
    kcap: u32,
    geom_dev: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.topk_radix_binstat)
        .grid([rows, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(work)
        .arg_u32(p)
        .arg_u32(select_k)
        .arg_u32(kcap)
        .arg_ptr(geom_dev)
        .launch(stream)
}

/// 2026-10-10: After the whole radix sequence of a sampled call: synchronise `stream`, read each
/// row's two slots and log the line. Debug only (a sync and `rows` small D2H copies).
pub fn report(
    gpu: &dyn GpuBackend,
    work: DevicePtr,
    rows: usize,
    kcap: usize,
    stream: u64,
) -> Result<()> {
    gpu.synchronize(stream)?;
    let stride = radix_row_bytes(kcap);
    let mut state = [0u8; RADIX_STATE_WORDS * 4];
    let mut got = Vec::with_capacity(rows);
    for r in 0..rows {
        gpu.copy_d2h(work.offset(r * stride + RADIX_STATE * 4), &mut state)?;
        let word = |i: usize| {
            u32::from_le_bytes([
                state[4 * i],
                state[4 * i + 1],
                state[4 * i + 2],
                state[4 * i + 3],
            ])
        };
        got.push((word(RS_B0CNT), word(RS_BINP)));
    }
    if let Some(s) = summarize(&got) {
        tracing::warn!("{}", format_line(&s));
    }
    Ok(())
}

#[cfg(test)]
#[path = "binstat_tests.rs"]
mod tests;
