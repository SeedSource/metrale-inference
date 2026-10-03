// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: CUDA virtual memory management for [`crate::lazy_buffer::LazyBuffer`]: reserve
//! an address range, map granules into it on demand, release it.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - `map_granule` either maps and enables read/write access on the whole granule and returns
//!   its handle, or leaves nothing behind (no handle, no mapping).
//!
//! The driver calls and struct layouts follow the GB10 go/no-go probe (spark-bench
//! `scripts/race/vmm_probe.py`, run 2026-10-03, PASSED: VMM supported, granule 2 MiB,
//! ~119 µs per create + map + set-access, memory charged at `cuMemCreate`, a graph captured
//! before a later map replays correctly). Layouts match `vendor/cudarc/src/driver/sys`
//! (CUDA 13): `CUmemAllocationProp` 32 B, `CUmemAccessDesc` 12 B.

use std::ffi::c_void;

use anyhow::{Result, bail};

use crate::gpu::DevicePtr;
use crate::lazy_buffer::{LazyBuffer, LazyKind, MapBudget, round_up_to_granule};

/// 2026-10-03: `CU_MEM_ALLOCATION_TYPE_PINNED`.
const ALLOC_TYPE_PINNED: i32 = 1;
/// 2026-10-03: `CU_MEM_LOCATION_TYPE_DEVICE`.
const LOC_TYPE_DEVICE: i32 = 1;
/// 2026-10-03: `CU_MEM_ACCESS_FLAGS_PROT_READWRITE`.
const ACCESS_RW: i32 = 3;
/// 2026-10-03: `CU_MEM_ALLOC_GRANULARITY_MINIMUM`.
const GRAN_MINIMUM: i32 = 0;
/// 2026-10-03: `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`.
const ATTR_VMM_SUPPORTED: u32 = 102;

#[repr(C)]
#[derive(Clone, Copy)]
struct CuMemLocation {
    kind: i32,
    id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CuMemAllocFlags {
    compression_type: u8,
    gpu_direct_rdma_capable: u8,
    usage: u16,
    reserved: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CuMemAllocationProp {
    kind: i32,
    requested_handle_types: i32,
    location: CuMemLocation,
    win32_handle_meta_data: *mut c_void,
    alloc_flags: CuMemAllocFlags,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CuMemAccessDesc {
    location: CuMemLocation,
    flags: i32,
}

const _: () = assert!(std::mem::size_of::<CuMemAllocationProp>() == 32);
const _: () = assert!(std::mem::size_of::<CuMemAccessDesc>() == 12);

unsafe extern "C" {
    fn cuMemGetAllocationGranularity(
        granularity: *mut usize,
        prop: *const CuMemAllocationProp,
        option: i32,
    ) -> i32;
    fn cuMemAddressReserve(
        ptr: *mut u64,
        size: usize,
        alignment: usize,
        addr: u64,
        flags: u64,
    ) -> i32;
    fn cuMemAddressFree(ptr: u64, size: usize) -> i32;
    fn cuMemCreate(
        handle: *mut u64,
        size: usize,
        prop: *const CuMemAllocationProp,
        flags: u64,
    ) -> i32;
    fn cuMemRelease(handle: u64) -> i32;
    fn cuMemMap(ptr: u64, size: usize, offset: usize, handle: u64, flags: u64) -> i32;
    fn cuMemUnmap(ptr: u64, size: usize) -> i32;
    fn cuMemSetAccess(ptr: u64, size: usize, desc: *const CuMemAccessDesc, count: usize) -> i32;
    fn cuCtxSynchronize() -> i32;
}

fn prop(device: i32) -> CuMemAllocationProp {
    CuMemAllocationProp {
        kind: ALLOC_TYPE_PINNED,
        requested_handle_types: 0,
        location: CuMemLocation {
            kind: LOC_TYPE_DEVICE,
            id: device,
        },
        win32_handle_meta_data: std::ptr::null_mut(),
        alloc_flags: CuMemAllocFlags {
            compression_type: 0,
            gpu_direct_rdma_capable: 0,
            usage: 0,
            reserved: [0; 4],
        },
    }
}

/// 2026-10-03: The current context's device, whether it supports VMM, and its minimum
/// granularity.
fn device_and_granule() -> Result<(i32, usize)> {
    let mut dev: i32 = 0;
    let st = unsafe { super::cuCtxGetDevice(&mut dev) };
    if st != 0 {
        bail!("cuCtxGetDevice failed: status {st}");
    }
    let mut supported: i32 = 0;
    let st = unsafe { super::cuDeviceGetAttribute(&mut supported, ATTR_VMM_SUPPORTED, dev) };
    if st != 0 || supported == 0 {
        bail!("device {dev} does not support CUDA virtual memory management (status {st})");
    }
    let p = prop(dev);
    let mut g: usize = 0;
    let st = unsafe { cuMemGetAllocationGranularity(&mut g, &p, GRAN_MINIMUM) };
    if st != 0 || g == 0 {
        bail!("cuMemGetAllocationGranularity failed: status {st}");
    }
    Ok((dev, g))
}

/// 2026-10-03: `METRALE_LAZY_MAP_FREE_FLOOR_MB` (default 4096): the device free memory a
/// lazy map must leave. Above the OOM watchdog's 2 GiB threshold, so a map is refused well
/// before the watchdog would end the process.
pub(crate) fn free_floor_bytes() -> usize {
    static FLOOR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *FLOOR.get_or_init(|| {
        std::env::var("METRALE_LAZY_MAP_FREE_FLOOR_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096)
            * 1024
            * 1024
    })
}

/// 2026-10-03: Reserve address space for `bytes` (rounded up to the granularity) and return
/// the unmapped buffer.
pub(crate) fn reserve(
    bytes: usize,
    budget: Option<std::sync::Arc<MapBudget>>,
) -> Result<LazyBuffer> {
    let (dev, granule) = device_and_granule()?;
    let reserved = round_up_to_granule(bytes.max(1), granule);
    let mut base: u64 = 0;
    let st = unsafe { cuMemAddressReserve(&mut base, reserved, granule, 0, 0) };
    if st != 0 || base == 0 {
        bail!("cuMemAddressReserve({reserved} bytes) failed: status {st}");
    }
    Ok(LazyBuffer::from_parts(
        DevicePtr(base),
        reserved,
        granule,
        LazyKind::Vmm { device: dev },
        budget,
        free_floor_bytes(),
    ))
}

/// 2026-10-03: Back `[va, va + granule)` with a new physical allocation on `device` and enable
/// read/write access to it. On failure nothing created here survives.
pub(crate) fn map_granule(va: u64, granule: usize, device: i32) -> Result<u64> {
    let p = prop(device);
    let mut h: u64 = 0;
    let st = unsafe { cuMemCreate(&mut h, granule, &p, 0) };
    if st != 0 {
        bail!("cuMemCreate({granule} bytes) failed: status {st}");
    }
    let st = unsafe { cuMemMap(va, granule, 0, h, 0) };
    if st != 0 {
        unsafe { cuMemRelease(h) };
        bail!("cuMemMap({va:#x}, {granule}) failed: status {st}");
    }
    let desc = CuMemAccessDesc {
        location: CuMemLocation {
            kind: LOC_TYPE_DEVICE,
            id: device,
        },
        flags: ACCESS_RW,
    };
    let st = unsafe { cuMemSetAccess(va, granule, &desc, 1) };
    if st != 0 {
        unsafe {
            cuMemUnmap(va, granule);
            cuMemRelease(h);
        }
        bail!("cuMemSetAccess({va:#x}, {granule}) failed: status {st}");
    }
    Ok(h)
}

/// 2026-10-03: Unmap and release the granules `handles` (mapped from `base` up, one granule
/// each), then free the `reserved`-byte address range. Every step runs even after a failure;
/// the first failure is returned. A context-gone status is not a failure.
pub(crate) fn release_range(
    base: u64,
    reserved: usize,
    granule: usize,
    handles: &[u64],
) -> Result<()> {
    let mut first_err: Option<String> = None;
    let mut note = |what: &str, st: i32| {
        if st != 0 && !crate::registry::is_teardown_noop(st) && first_err.is_none() {
            first_err = Some(format!("{what} failed: status {st}"));
        }
    };
    // 2026-10-03: Wait for queued work first. `cuMemFree`, which the eager layout uses, waits
    // for the device implicitly; unmapping under a kernel or graph still reading the range
    // would fault. A sequence is freed once per request, so the wait is off the step path.
    if !handles.is_empty() {
        note("cuCtxSynchronize", unsafe { cuCtxSynchronize() });
    }
    // 2026-10-03: One unmap per granule: each `cuMemMap` is its own mapping, and the driver
    // documents unmapping whole mappings (PyTorch's expandable segments also unmap per handle).
    for (i, &h) in handles.iter().enumerate() {
        note("cuMemUnmap", unsafe {
            cuMemUnmap(base + (i * granule) as u64, granule)
        });
        note("cuMemRelease", unsafe { cuMemRelease(h) });
    }
    note("cuMemAddressFree", unsafe {
        cuMemAddressFree(base, reserved)
    });
    match first_err {
        Some(e) => bail!("lazy buffer {base:#x} release: {e}"),
        None => Ok(()),
    }
}
