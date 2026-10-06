// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: `METRALE_GLM_DSA_TOPK_RADIX=1` (default off): the DSA top-k over pools as an
//! exact, multi-block radix select (`dsa_topk_radix_{init,hist,find,gather,sort}` in
//! `kernels/gb10/common/dsa_indexer.cu`, section 3b) instead of `dsa_topk_pools`.
//!
//! Why: `dsa_topk_pools` runs one block per query row and walks the pool axis in tiles of
//! [`topk_tile`] pools, bitonic-sorting every tile, so its time is linear in the pool count on
//! one SM per row. Measured 2026-10-06 (race-mem-depth-L34, `dsa_depth_decode_microtest`): 39.6 /
//! 79.7 / 161.9 ms per K = 3 decode step (33 calls) at 131K / 262K / 532K context. The radix
//! select reads the row four times on `C` chunks per row and sorts only the `select_k` winners.
//! Expected (not yet measured): well under 0.1 ms a call at 532K.
//!
//! Output contract (exact): `selected[r][0..select_k)` is byte-equal to what `dsa_topk_pools`
//! writes, the first `select_k` pools under score descending then pool index ascending, with
//! -0.0 tying +0.0 and -FLT_MAX an ordinary (lowest finite) value. [`radix_key`] is the
//! kernel's key transform; the tests below run a host model of the five kernels against a sort.
//!
//! Dispatch: [`radix_mode`]. The radix path runs only with the lever on, the five entry points
//! resolved, more than [`topk_tile`] pools to dispatch on (the host `n_pools` of an exact launch,
//! the ceiling pool count of a graph launch, so a captured graph never changes path with the
//! live context), and a radix region in the select scratch. At or below one tile
//! `dsa_topk_pools` sorts a single tile and is kept.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - The `RADIX_*` layout constants equal the `DSA_RADIX_*` defines in `dsa_indexer.cu`
//!   (checked by a test that reads the file).
//! - Every launch re-initialises the scratch it reads (the init kernel is the first of the nine
//!   launches), so a CUDA graph that captured them replays correctly.
//! - Lever off: no radix region is planned or allocated, and `select_tokens` launches exactly
//!   what it did before.

use std::sync::OnceLock;

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{DsaSelectGeometry, topk_tile};
use crate::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};

/// 2026-10-06: Threads per block of every radix kernel (`DSA_RADIX_THREADS`); the find kernel's
/// bin split (16 bins a thread, then 1) assumes exactly this many and traps otherwise.
pub const RADIX_THREADS: u32 = 256;
/// 2026-10-06: Most chunks per row (`DSA_RADIX_MAX_CHUNKS`): the tie bases and the pass-2
/// per-chunk histograms are sized for it.
pub const RADIX_MAX_CHUNKS: usize = 32;
/// 2026-10-06: Blocks a hist or gather launch aims for across all rows. PROVISIONAL (from the
/// GB10's 48 SMs, two blocks each; not measured): a 1-row decode call gets 32 chunks, a 256-row
/// prefill pass 1 chunk a row.
pub const RADIX_TARGET_BLOCKS: usize = 96;
/// 2026-10-06: Bins of passes 0 and 1 (`DSA_RADIX_BINS`, 12 key bits each) and of pass 2
/// (`DSA_RADIX_BINS_LAST`, the last 8 bits).
pub const RADIX_BINS: usize = 4096;
pub const RADIX_BINS_LAST: usize = 256;
/// 2026-10-06: Word offsets in a row's work buffer: the state words (`DSA_RADIX_STATE`, after
/// the two 4096-bin histograms), the per-chunk tie bases (`DSA_RADIX_TIE`) and the candidates
/// (`DSA_RADIX_CAND`: `kcap` keys, then `kcap` pool indices).
pub const RADIX_STATE: usize = 2 * RADIX_BINS;
pub const RADIX_STATE_WORDS: usize = 16;
pub const RADIX_TIE: usize = RADIX_STATE + RADIX_STATE_WORDS;
pub const RADIX_CAND: usize = RADIX_TIE + RADIX_MAX_CHUNKS;
/// 2026-10-06: Most candidates the sort kernel holds in shared memory (`DSA_RADIX_SORT_MAX`):
/// one top-k tile, the most `select_k` that `DsaSelectGeometry::plan` accepts.
pub const RADIX_SORT_MAX: usize = 2048;
/// 2026-10-06: Histogram/find pass pairs per call: key bits 31..20, 19..8, 7..0.
pub const RADIX_PASSES: u32 = 3;

/// 2026-10-06: Logged once, on the first launch that takes the radix path, and on no other path.
pub const RADIX_ENGAGED_LINE: &str = "METRALE_GLM_DSA_TOPK_RADIX=1: ENGAGED - radix top-k";

/// 2026-10-06: The kernels' monotone key (`dsa_radix_key`): for finite `a`, `b`,
/// `a > b` iff `radix_key(a) > radix_key(b)`, and `a == b` (so -0.0 with +0.0) iff the keys are
/// equal. -0.0 is canonicalised to +0.0, then a non-negative float gets its sign bit set and a
/// negative one is bit-inverted.
pub fn radix_key(v: f32) -> u32 {
    let mut u = v.to_bits();
    if u == 0x8000_0000 {
        u = 0;
    }
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

/// 2026-10-06: `METRALE_GLM_DSA_TOPK_RADIX`: `1` (blanks ignored) is on; unset, `0` and anything
/// else are off.
pub(crate) fn parse_topk_radix(v: Option<&str>) -> bool {
    crate::glm5next_layer::levers::parse_dsa_switch(v)
}

/// 2026-10-06: Whether `METRALE_GLM_DSA_TOPK_RADIX=1` is set; read once. A value other than
/// unset, empty, `0` or `1` is warned about and treated as off. Nothing else is logged here: the
/// engage line is [`RADIX_ENGAGED_LINE`], printed by the first radix launch.
pub fn dsa_topk_radix() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_TOPK_RADIX").ok();
        let odd = |r: &&str| !r.is_empty() && r.trim() != "0" && r.trim() != "1";
        if let Some(r) = raw.as_deref().filter(odd) {
            tracing::warn!("METRALE_GLM_DSA_TOPK_RADIX={r} is not 0 or 1 - treated as off");
        }
        parse_topk_radix(raw.as_deref())
    })
}

/// 2026-10-06: Candidate capacity of a row's work buffer: the largest `select_k` the config can
/// plan, `index_topk / index_kpool` (512 on GLM-5.3), capped at one top-k tile.
pub fn radix_kcap(cfg: &Glm5NextDsaConfig) -> usize {
    (cfg.index_topk / cfg.index_kpool).min(topk_tile())
}

/// 2026-10-06: u32 words of one row's work buffer for candidate capacity `kcap`.
pub fn radix_row_words(kcap: usize) -> usize {
    RADIX_CAND + 2 * kcap
}

/// 2026-10-06: Bytes of one row's work buffer (37,056 at `kcap` 512).
pub fn radix_row_bytes(kcap: usize) -> usize {
    radix_row_words(kcap) * 4
}

/// 2026-10-06: Bytes of the work buffer for `q_rows` rows of `cfg`.
pub fn radix_work_bytes(cfg: &Glm5NextDsaConfig, q_rows: usize) -> usize {
    q_rows * radix_row_bytes(radix_kcap(cfg))
}

/// 2026-10-06: Bytes of the radix work buffer (select scratch region 6) a pass of `geom` needs:
/// `geom.q_rows` row buffers when `lever` is on and the pass has more than one top-k tile of
/// pools, else 0. Planned at the context ceiling it covers every ceiling launch of as many rows,
/// whatever the live pool count.
pub fn radix_region_bytes(geom: &DsaSelectGeometry, cfg: &Glm5NextDsaConfig, lever: bool) -> usize {
    if lever && geom.n_pools > topk_tile() {
        radix_work_bytes(cfg, geom.q_rows)
    } else {
        0
    }
}

/// 2026-10-06: All seven select scratch regions of `geom`, in `DsaSelectScratch` field order:
/// `scratch_bytes` (regions 0 to 5), then [`radix_region_bytes`].
/// 2026-10-06: `pool_cache` (`METRALE_GLM_DSA_POOL_CACHE=1`) plans regions 0 to 2 (pool keys,
/// indices, validity) at 0 bytes (`scratch_bytes_with`); region 6 does not depend on it.
pub(super) fn regions(
    geom: &DsaSelectGeometry,
    cfg: &Glm5NextDsaConfig,
    lever: bool,
    pool_cache: bool,
) -> [usize; 7] {
    let b = geom.scratch_bytes_with(pool_cache);
    [
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        b[5],
        radix_region_bytes(geom, cfg, lever),
    ]
}

/// 2026-10-06: Allocate region 6: NULL (no allocation) when `bytes` is 0, so a lever-off scratch
/// makes the same allocations as before the region existed.
#[track_caller]
pub(super) fn alloc_region(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    if bytes == 0 {
        Ok(DevicePtr::NULL)
    } else {
        gpu.alloc(bytes)
    }
}

/// 2026-10-06: Chunks per row (grid x of the hist and gather launches) for `q_rows` rows:
/// [`RADIX_TARGET_BLOCKS`] spread over the rows, between 1 and [`RADIX_MAX_CHUNKS`]. Depends on
/// `q_rows` only, so every capture of a 1-row ceiling launch gets the same grid.
pub fn radix_chunks(q_rows: usize) -> usize {
    RADIX_TARGET_BLOCKS
        .div_ceil(q_rows.max(1))
        .clamp(1, RADIX_MAX_CHUNKS)
}

/// 2026-10-06: Whether all five radix entry points resolved.
pub fn radix_resolved(k: &Glm5NextDsaKernels) -> bool {
    [
        k.topk_radix_init,
        k.topk_radix_hist,
        k.topk_radix_find,
        k.topk_radix_gather,
        k.topk_radix_sort,
    ]
    .iter()
    .all(|h| h.0 != 0)
}

/// 2026-10-06: Which top-k a selection pass launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadixMode {
    /// 2026-10-06: Lever off: `dsa_topk_pools`.
    Off,
    /// 2026-10-06: Lever on, but the kernel module lacks a radix entry point: `dsa_topk_pools`.
    MissingKernels,
    /// 2026-10-06: Lever on, at most one top-k tile of pools: `dsa_topk_pools` (one tile).
    Small,
    /// 2026-10-06: Lever on, but the select scratch has no (or too small a) radix region:
    /// `dsa_topk_pools`.
    NoScratch,
    /// 2026-10-06: The radix select.
    Engaged,
}

/// 2026-10-06: The mode for the lever, whether the entry points resolved, whether the scratch
/// holds the pass's work buffer, and the pool count the launch dispatches on (`n_pools`, or the
/// ceiling pool count of a ceiling launch).
pub fn radix_mode(lever: bool, resolved: bool, scratch_ok: bool, pools: usize) -> RadixMode {
    if !lever {
        RadixMode::Off
    } else if !resolved {
        RadixMode::MissingKernels
    } else if pools <= topk_tile() {
        RadixMode::Small
    } else if !scratch_ok {
        RadixMode::NoScratch
    } else {
        RadixMode::Engaged
    }
}

/// 2026-10-06: Log `mode` once per kind: the engage line on the first radix launch, an
/// `ignored` warning when the lever is on but cannot apply. `Off` and `Small` log nothing.
pub(crate) fn log_radix_mode(mode: RadixMode) {
    match mode {
        RadixMode::Engaged => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("{RADIX_ENGAGED_LINE}"));
        }
        RadixMode::MissingKernels => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    "METRALE_GLM_DSA_TOPK_RADIX=1 ignored: the kernel module lacks the \
                     dsa_topk_radix_* entry points; dsa_topk_pools runs"
                )
            });
        }
        RadixMode::NoScratch => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    "METRALE_GLM_DSA_TOPK_RADIX=1 ignored for a pass: the DSA select scratch \
                     has no radix region for its rows (planned without the lever?); \
                     dsa_topk_pools runs"
                )
            });
        }
        RadixMode::Off | RadixMode::Small => {}
    }
}

/// 2026-10-06: Buffers and extents of one radix top-k call.
#[derive(Debug, Clone, Copy)]
pub struct RadixTopk {
    /// 2026-10-06: `[q_rows, P]` f32 scores, row stride the live `P`.
    pub scores: DevicePtr,
    /// 2026-10-06: `[q_rows, select_k]` i32 pool indices, row stride the live `select_k`.
    pub selected: DevicePtr,
    /// 2026-10-06: At least `q_rows * radix_row_bytes(kcap)` bytes; contents need no init.
    pub work: DevicePtr,
    pub q_rows: usize,
    /// 2026-10-06: Pools `P` and `select_k` as scalars; when `geom_dev` is non-null the kernels
    /// read both from it instead (`dsa_topk_pools` does the same).
    pub n_pools: usize,
    pub select_k: usize,
    /// 2026-10-06: Candidate capacity per row ([`radix_kcap`]); at least every `select_k` the
    /// call can see, at most [`RADIX_SORT_MAX`].
    pub kcap: usize,
    /// 2026-10-06: `[5]` i32 device geometry from `dsa_write_geom`, or NULL.
    pub geom_dev: DevicePtr,
}

/// 2026-10-06: Enqueue the nine radix launches on `stream` (init; hist + find for each of the
/// three passes; gather; sort), writing `t.selected` exactly as `dsa_topk_pools` would.
///
/// Fails before any launch when an entry point is missing, `q_rows` is 0 or above 65,535 (grid
/// y), `kcap` exceeds [`RADIX_SORT_MAX`], or a scalar `select_k` exceeds `kcap` or `n_pools`.
pub fn launch_topk_radix(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    t: &RadixTopk,
    stream: u64,
) -> Result<()> {
    if !radix_resolved(kernels) {
        bail!("DSA radix top-k: a dsa_topk_radix_* entry point is missing from the kernel module");
    }
    if t.q_rows == 0 || t.q_rows > u16::MAX as usize {
        bail!("DSA radix top-k: q_rows {} is outside 1..=65535", t.q_rows);
    }
    if t.kcap > RADIX_SORT_MAX {
        bail!("DSA radix top-k: kcap {} exceeds {RADIX_SORT_MAX}", t.kcap);
    }
    if t.geom_dev.is_null() && (t.select_k > t.kcap || t.select_k > t.n_pools) {
        bail!(
            "DSA radix top-k: select_k {} exceeds kcap {} or the {} pools",
            t.select_k,
            t.kcap,
            t.n_pools
        );
    }
    let rows = t.q_rows as u32;
    let chunks = radix_chunks(t.q_rows) as u32;
    let block = [RADIX_THREADS, 1, 1];
    let (p, selk, kcap, gd) = (
        t.n_pools as u32,
        t.select_k as u32,
        t.kcap as u32,
        t.geom_dev,
    );

    KernelLaunch::new(gpu, kernels.topk_radix_init)
        .grid([rows, 1, 1])
        .block(block)
        .arg_ptr(t.work)
        .arg_u32(selk)
        .arg_u32(kcap)
        .arg_ptr(gd)
        .launch(stream)?;
    for pass in 0..RADIX_PASSES {
        KernelLaunch::new(gpu, kernels.topk_radix_hist)
            .grid([chunks, rows, 1])
            .block(block)
            .arg_ptr(t.scores)
            .arg_ptr(t.work)
            .arg_u32(p)
            .arg_u32(selk)
            .arg_u32(kcap)
            .arg_u32(pass)
            .arg_ptr(gd)
            .launch(stream)?;
        KernelLaunch::new(gpu, kernels.topk_radix_find)
            .grid([rows, 1, 1])
            .block(block)
            .arg_ptr(t.work)
            .arg_u32(selk)
            .arg_u32(kcap)
            .arg_u32(pass)
            .arg_u32(chunks)
            .arg_ptr(gd)
            .launch(stream)?;
    }
    KernelLaunch::new(gpu, kernels.topk_radix_gather)
        .grid([chunks, rows, 1])
        .block(block)
        .arg_ptr(t.scores)
        .arg_ptr(t.work)
        .arg_u32(p)
        .arg_u32(selk)
        .arg_u32(kcap)
        .arg_ptr(gd)
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.topk_radix_sort)
        .grid([rows, 1, 1])
        .block(block)
        .arg_ptr(t.work)
        .arg_ptr(t.selected)
        .arg_u32(selk)
        .arg_u32(kcap)
        .arg_ptr(gd)
        .launch(stream)
}

#[cfg(test)]
#[path = "radix_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "radix_model_tests.rs"]
mod model_tests;
