// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Routing of the narrow NVFP4 batched GEMV to its tensor-core kernels (`w4a16_gemv_tc8` / `w4a16_gemv_tc16` in kernels/gb10/common/w4a16_gemv_tc.cu).
//!
//! Owner: model-layers ops.
//! Invariants:
//! - The tensor-core entries take the same arguments as the CUDA-core `w4a16_gemv_batch*`
//!   kernels, with their own grid (`ceil(N / cols_per_cta)`) and a 256-thread block.
//! - A launch routes only when `K % 128 == 0` and `1 <= m <= 16`; otherwise the caller keeps its
//!   CUDA-core kernel ([`tc_route`]).
//! - [`super::w4a16_gemv_batchm`] and [`tc_fixed_m`] both route through [`tc_kernel`].
//! - `METRALE_NO_W4A16_TC` set to any non-empty value, read once per process, turns the
//!   tensor-core path off ([`tc_enabled`]). The bench crate's `PERF_CONTROLS` discloses it,
//!   and `record_env_tests` reads [`tc_enabled`] and [`wide_rows_enabled`] from this file.

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::weight_map::QuantizedWeight;

/// 2026-09-25: Rows the `w4a16_gemv_tc8` entry covers (its `MT` is 8).
pub const TC8_MAX_M: u32 = 8;
/// 2026-09-25: Rows the `w4a16_gemv_tc16` entry covers (its `MT` is 16).
pub const TC16_MAX_M: u32 = 16;
/// 2026-09-25: Columns per CTA, `8 * NT`: tc8 has `NT = 1`, tc16 has `NT = 2`.
pub const TC8_COLS_PER_CTA: u32 = 8;
pub const TC16_COLS_PER_CTA: u32 = 16;
/// 2026-09-25: Threads per CTA, `TC_WARPS * 32` with `TC_WARPS = 8` in the .cu; the warps split K.
pub const TC_BLOCK: u32 = 256;

/// 2026-09-25: Which tensor-core entry serves a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcKind {
    M8,
    M16,
}

impl TcKind {
    pub fn cols_per_cta(self) -> u32 {
        match self {
            TcKind::M8 => TC8_COLS_PER_CTA,
            TcKind::M16 => TC16_COLS_PER_CTA,
        }
    }
}

/// 2026-09-25: The routing decision, without I/O. `None` keeps the caller's CUDA-core kernel.
///
/// The kernel walks K in 128-element blocks, so a `K` that is not a multiple of 128 declines.
/// Any `N` routes: in a partial last tile the kernel loads zeros for columns past `N` and never
/// stores them.
/// `tc8` serves `m <= 8`; `tc16` serves `m <= 16`, including `m <= 8` when `tc8` is missing.
pub fn tc_route(
    m: u32,
    n: u32,
    k: u32,
    enabled: bool,
    have8: bool,
    have16: bool,
) -> Option<TcKind> {
    if !enabled || m == 0 || n == 0 || k == 0 || !k.is_multiple_of(128) {
        return None;
    }
    if m <= TC8_MAX_M && have8 {
        Some(TcKind::M8)
    } else if m <= TC16_MAX_M && have16 {
        Some(TcKind::M16)
    } else {
        None
    }
}

/// 2026-09-25: True when `METRALE_NO_W4A16_TC` is unset or empty. Any non-empty value, `0`
/// included, turns the tensor-core path off. Read once per process, so every launch, including
/// one captured in a CUDA graph, sees the same choice.
pub fn tc_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_W4A16_TC").is_none_or(|v| v.is_empty()))
}

/// 2026-09-25: Row edges of the narrow-GEMV dispatch arms: `NARROW_MAX_ROWS` by default, and
/// `WIDE_MAX_ROWS` (the `tc16` reach) when [`wide_rows_enabled`]. Read through
/// [`narrow_gemv_max_rows`].
pub const NARROW_MAX_ROWS: u32 = 8;
pub const WIDE_MAX_ROWS: u32 = TC16_MAX_M;

/// 2026-09-25: The opt-in widening: true when `METRALE_W4A16_TC_WIDE` is set to a non-empty value
/// and [`tc_enabled`]. Read once per process. It raises [`narrow_gemv_max_rows`] to 16 and
/// lets [`tc_fixed_m`] route the fixed-M launchers (`w4a16_gemv_batch2/3`,
/// `w4a16_gemv_dual_batch2/3`) to the tensor-core kernels.
pub fn wide_rows_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        tc_enabled() && std::env::var_os("METRALE_W4A16_TC_WIDE").is_some_and(|v| !v.is_empty())
    })
}

/// 2026-09-25: Row edge of the narrow-GEMV dispatch arms (see [`wide_rows_enabled`]).
pub fn narrow_gemv_max_rows() -> u32 {
    if wide_rows_enabled() {
        WIDE_MAX_ROWS
    } else {
        NARROW_MAX_ROWS
    }
}

/// 2026-09-25: Resolved handles, cached per backend. A `KernelHandle` names a function in one
/// backend's loaded module, so the cache is keyed by the backend object's address. A missing
/// entry is `KernelHandle(0)`.
#[derive(Clone, Copy)]
struct TcHandles {
    tc8: KernelHandle,
    tc16: KernelHandle,
}

fn tc_handles(gpu: &dyn GpuBackend) -> TcHandles {
    static CACHE: OnceLock<Mutex<Vec<(usize, TcHandles)>>> = OnceLock::new();
    let key = gpu as *const dyn GpuBackend as *const () as usize;
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, h)) = guard.iter().find(|(k, _)| *k == key) {
        return *h;
    }
    let h = TcHandles {
        tc8: crate::layers::try_kernel(gpu, "w4a16_gemv_tc", "w4a16_gemv_tc8"),
        tc16: crate::layers::try_kernel(gpu, "w4a16_gemv_tc", "w4a16_gemv_tc16"),
    };
    guard.push((key, h));
    h
}

/// 2026-09-25: The tensor-core kernel and grid x for this launch, or `None` to keep the
/// CUDA-core kernel. It does not consult [`wide_rows_enabled`]; the callers' row edges do.
pub fn tc_kernel(gpu: &dyn GpuBackend, m: u32, n: u32, k: u32) -> Option<(KernelHandle, u32)> {
    if !tc_enabled() {
        return None;
    }
    let h = tc_handles(gpu);
    let kind = tc_route(m, n, k, true, h.tc8.0 != 0, h.tc16.0 != 0)?;
    let handle = match kind {
        TcKind::M8 => h.tc8,
        TcKind::M16 => h.tc16,
    };
    Some((handle, n.div_ceil(kind.cols_per_cta())))
}

/// 2026-09-25: The tensor-core route of the fixed-M launchers, taken only when
/// [`wide_rows_enabled`]: `Ok(true)` when it launched, `Ok(false)` to keep the CUDA-core kernel.
#[allow(clippy::too_many_arguments)]
pub fn tc_fixed_m(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<bool> {
    if !wide_rows_enabled() {
        return Ok(false);
    }
    let Some((tc, grid_x)) = tc_kernel(gpu, m, n, k) else {
        return Ok(false);
    };
    KernelLaunch::new(gpu, tc)
        .grid([grid_x, 1, 1])
        .block([TC_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    Ok(true)
}

/// 2026-10-09: Most projections one grouped launch takes (`w4a16_gemv_tc{8,16}_group` have
/// three weight slots).
pub const TC_GROUP_MAX: usize = 3;

/// 2026-10-09: One projection of a grouped launch: `out[m, n] = A[m, k] @ dequant(weight)^T`.
#[derive(Clone, Copy)]
pub struct TcGroupMember<'a> {
    pub weight: &'a QuantizedWeight,
    pub out: DevicePtr,
    pub n: u32,
}

/// 2026-10-09: Tiles of a grouped launch: the sum of each member's `ceil(n / cols_per_cta)`,
/// the grid the per-projection launches would use together.
pub fn tc_group_tiles(kind: TcKind, ns: &[u32]) -> u32 {
    ns.iter().map(|&n| n.div_ceil(kind.cols_per_cta())).sum()
}

/// 2026-10-09: Persistent grid of a grouped launch: `min(tiles, sm_count * resident)`, where
/// `resident` is the occupancy calculator's CTAs per SM for the kernel at `TC_BLOCK` threads.
/// `cap == 0` (backend cannot say) launches one CTA per tile.
pub fn tc_group_grid(tiles: u32, cap: u32) -> u32 {
    if cap == 0 { tiles } else { tiles.min(cap) }
}

/// 2026-10-09: Resolved grouped handles and their persistent CTA caps (`sm_count *` resident
/// CTAs per SM, 0 when unknown), cached per backend like [`TcHandles`].
#[derive(Clone, Copy)]
struct TcGroupHandles {
    tc8: (KernelHandle, u32),
    tc16: (KernelHandle, u32),
}

fn tc_group_handles(gpu: &dyn GpuBackend) -> TcGroupHandles {
    static CACHE: OnceLock<Mutex<Vec<(usize, TcGroupHandles)>>> = OnceLock::new();
    let key = gpu as *const dyn GpuBackend as *const () as usize;
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, h)) = guard.iter().find(|(k, _)| *k == key) {
        return *h;
    }
    let sms = gpu.sm_count().unwrap_or(0);
    let resolve = |name: &str| {
        let h = crate::layers::try_kernel(gpu, "w4a16_gemv_tc", name);
        let per_sm = if h.0 == 0 {
            0
        } else {
            gpu.max_active_blocks_per_sm(h, TC_BLOCK, 0).unwrap_or(0)
        };
        (h, sms.saturating_mul(per_sm))
    };
    let h = TcGroupHandles {
        tc8: resolve("w4a16_gemv_tc8_group"),
        tc16: resolve("w4a16_gemv_tc16_group"),
    };
    tracing::info!(
        "w4a16_gemv_tc group: {sms} SMs, persistent CTA cap tc8 {} tc16 {}",
        h.tc8.1,
        h.tc16.1
    );
    guard.push((key, h));
    h
}

/// 2026-10-09: The grouped persistent launch of up to [`TC_GROUP_MAX`] projections that read
/// the same `input` `[m, k]`, each member's bytes equal to its own [`tc_kernel`] launch (the
/// kernel runs the same per-tile body; see `w4a16_gemv_tc_group_impl`). `Ok(false)` launches
/// nothing: tensor cores off, a member the route would not send to the same entry as the
/// others, an empty or too-large group, or the grouped entry missing from the image.
pub fn tc_group_launch(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    members: &[TcGroupMember],
    m: u32,
    k: u32,
    stream: u64,
) -> Result<bool> {
    if !tc_enabled() || members.is_empty() || members.len() > TC_GROUP_MAX {
        return Ok(false);
    }
    let h = tc_handles(gpu);
    let (have8, have16) = (h.tc8.0 != 0, h.tc16.0 != 0);
    let Some(kind) = tc_route(m, members[0].n, k, true, have8, have16) else {
        return Ok(false);
    };
    if members
        .iter()
        .any(|g| tc_route(m, g.n, k, true, have8, have16) != Some(kind))
    {
        return Ok(false);
    }
    let gh = tc_group_handles(gpu);
    let (handle, cap) = match kind {
        TcKind::M8 => gh.tc8,
        TcKind::M16 => gh.tc16,
    };
    if handle.0 == 0 {
        return Ok(false);
    }
    let ns: Vec<u32> = members.iter().map(|g| g.n).collect();
    let grid_x = tc_group_grid(tc_group_tiles(kind, &ns), cap);
    let mut l = KernelLaunch::new(gpu, handle)
        .grid([grid_x, 1, 1])
        .block([TC_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_u32(m)
        .arg_u32(k);
    for i in 0..TC_GROUP_MAX {
        l = match members.get(i) {
            Some(g) => l
                .arg_ptr(g.weight.weight)
                .arg_ptr(g.weight.weight_scale)
                .arg_f32(g.weight.weight_scale_2)
                .arg_ptr(g.out)
                .arg_u32(g.n),
            None => l
                .arg_ptr(DevicePtr(0))
                .arg_ptr(DevicePtr(0))
                .arg_f32(0.0)
                .arg_ptr(DevicePtr(0))
                .arg_u32(0),
        };
    }
    l.launch(stream)?;
    Ok(true)
}

#[cfg(test)]
#[path = "gemv_tc_tests.rs"]
mod gemv_tc_tests;
