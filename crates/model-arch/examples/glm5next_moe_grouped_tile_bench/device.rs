// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Upload and download helpers, and the grouped-GEMM launch the tile bench makes
//! for every variant.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::NUM_EXPERTS;

pub(crate) fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

pub(crate) fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

pub(crate) fn up_u64(g: &dyn GpuBackend, v: &[u64]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}

pub(crate) fn up_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}

pub(crate) fn up_i32(g: &dyn GpuBackend, v: &[i32]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}

pub(crate) fn dn_i32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<i32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub(crate) fn dn_raw(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2: DevicePtr,
    c: DevicePtr,
    off: DevicePtr,
    stid: DevicePtr,
    n: usize,
    kk: usize,
    max_m_tiles: u32,
    n_tile: u32,
    threads: u32,
    m_fast: bool,
) -> Result<()> {
    let n_tiles = (n as u32).div_ceil(n_tile);
    // 2026-09-29: `_mfast` kernels take the M tile from grid x and the N tile from grid y.
    let grid = if m_fast {
        [max_m_tiles, n_tiles, NUM_EXPERTS as u32]
    } else {
        [n_tiles, max_m_tiles, NUM_EXPERTS as u32]
    };
    KernelLaunch::new(gpu, k)
        .grid(grid)
        .block([threads, 1, 1])
        .arg_ptr(a)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2)
        .arg_ptr(c)
        .arg_ptr(off)
        .arg_ptr(stid)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(0)
}
