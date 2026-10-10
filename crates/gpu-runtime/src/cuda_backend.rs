// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The CUDA `GpuBackend`: [`MetraleCudaBackend`], the driver API
//! declarations its submodules call, and its device-allocation ledger.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - A pointer that `alloc` or `alloc_managed` returns is on `live_allocs`
//!   until `free` or `sweep_unreleased` takes it off; `free` takes it off
//!   before calling `cuMemFree_v2`, whatever that call returns.
//! - `alloc_arena` allocations are never on `live_allocs`.

use std::ffi::c_void;

use anyhow::{Result, bail};
use std::sync::Arc;

use crate::registry::MetraleRegistry;

pub mod arch_preflight;
mod fault_probe;
mod gpu_copy;
mod gpu_impl;
mod gpu_impl_graph;
mod redzone;
pub mod tensormap;
pub(crate) use redzone::{
    ALLOC_SEQ, RedZone, redzone_bytes, redzone_fill, redzone_min_idx, redzone_trace_idx,
};
mod memory;
// 2026-10-03: Not under `metrale_scale`: the HIP libcuda shim has no VMM entry points, so
// AMD builds keep the default eager `alloc_lazy`.
#[cfg(not(metrale_scale))]
pub(crate) mod vmm;
pub use memory::{cuda_free_memory_bytes, spawn_oom_watchdog};
pub(crate) use memory::{device_is_integrated, effective_free_bytes};

unsafe extern "C" {
    pub(super) fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    pub(super) fn cuMemFree_v2(dptr: u64) -> i32;
    pub(super) fn cuMemcpyHtoDAsync_v2(
        dst: u64,
        src: *const c_void,
        bytes: usize,
        stream: u64,
    ) -> i32;
    pub(super) fn cuMemcpyDtoHAsync_v2(
        dst: *mut c_void,
        src: u64,
        bytes: usize,
        stream: u64,
    ) -> i32;
    pub(super) fn cuMemcpyDtoDAsync_v2(dst: u64, src: u64, bytes: usize, stream: u64) -> i32;
    pub(super) fn cuStreamSynchronize(stream: u64) -> i32;
    pub(super) fn cuStreamQuery(stream: u64) -> i32;
    pub(super) fn cuMemHostGetDevicePointer_v2(
        dptr: *mut u64,
        host: *mut std::ffi::c_void,
        flags: u32,
    ) -> i32;
    pub(super) fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
    /// 2026-09-25: The device of the calling thread's current context.
    pub(super) fn cuCtxGetDevice(device: *mut i32) -> i32;
    /// 2026-09-25: The `CUdevice` for a device ordinal, with no current
    /// context needed on the calling thread; `arch_preflight` uses it for
    /// that reason.
    pub(super) fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    pub(super) fn cuDeviceGetAttribute(pi: *mut i32, attrib: u32, dev: i32) -> i32;
    /// 2026-10-09: Not declared under `metrale_scale` (the HIP libcuda shim); there
    /// `max_active_blocks_per_sm` reports 0.
    #[cfg(not(metrale_scale))]
    pub(super) fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        num_blocks: *mut i32,
        func: *mut c_void,
        block_size: i32,
        dynamic_smem: usize,
    ) -> i32;
    pub(super) fn cuMemsetD8Async(dst: u64, value: u8, n: usize, stream: u64) -> i32;
    /// 2026-09-25: Synchronous variants, used only by the red-zone diagnostic
    /// (`redzone.rs`).
    pub(super) fn cuMemsetD8_v2(dst: u64, value: u8, n: usize) -> i32;
    pub(super) fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> i32;
    pub(super) fn cuStreamBeginCapture(hStream: u64, mode: u32) -> i32;
    // 2026-09-25: Not declared under `metrale_scale`; there
    // `stream_is_capturing` reports false without asking the driver.
    #[cfg(not(metrale_scale))]
    pub(super) fn cuStreamIsCapturing(hStream: u64, captureStatus: *mut u32) -> i32;
    pub(super) fn cuStreamEndCapture(hStream: u64, phGraph: *mut u64) -> i32;
    // 2026-09-25: Graph instantiate: `cuGraphInstantiate` under `metrale_scale`
    // (build.rs sets it when METRALE_TARGET_HW starts with `strix`),
    // `cuGraphInstantiateWithFlags` otherwise. Both take
    // `(CUgraphExec*, CUgraph, u64)`.
    #[cfg(not(metrale_scale))]
    pub(super) fn cuGraphInstantiateWithFlags(
        phGraphExec: *mut u64,
        hGraph: u64,
        flags: u64,
    ) -> i32;
    #[cfg(metrale_scale)]
    pub(super) fn cuGraphInstantiate(phGraphExec: *mut u64, hGraph: u64, flags: u64) -> i32;
    pub(super) fn cuGraphLaunch(hGraphExec: u64, hStream: u64) -> i32;
    pub(super) fn cuGraphExecDestroy(hGraphExec: u64) -> i32;
    pub(super) fn cuGraphDestroy(hGraph: u64) -> i32;
    fn cuCtxGetCurrent(pctx: *mut u64) -> i32;
    pub(super) fn cuCtxSetCurrent(ctx: u64) -> i32;
    pub(super) fn cuStreamCreate(phStream: *mut u64, flags: u32) -> i32;
    pub(super) fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> i32;
    pub(super) fn cuMemFreeHost(p: *mut c_void) -> i32;
    pub(super) fn cuMemAllocManaged(dptr: *mut u64, bytesize: usize, flags: u32) -> i32;
    pub(super) fn cuEventCreate(phEvent: *mut u64, flags: u32) -> i32;
    pub(super) fn cuEventRecord(hEvent: u64, hStream: u64) -> i32;
    pub(super) fn cuStreamWaitEvent(hStream: u64, hEvent: u64, flags: u32) -> i32;
    pub(super) fn cuEventSynchronize(hEvent: u64) -> i32;
    pub(super) fn cuEventQuery(hEvent: u64) -> i32;
    pub(super) fn cuEventDestroy_v2(hEvent: u64) -> i32;
}

/// 2026-09-25: The CUDA `GpuBackend`, holding one model's kernel modules.
///
/// The modules unload when the last `Arc<MetraleRegistry>` handle drops
/// (`MetraleRegistry`'s `Drop`); this struct holds one of those handles.
pub struct MetraleCudaBackend {
    /// 2026-09-25: This model's kernel modules. `registry()` and
    /// `kernel_registry()` hand out clones of the `Arc`.
    registry: Arc<MetraleRegistry>,
    /// 2026-09-25: `METRALE_DEBUG_SYNC_KERNELS=1`: sync after every launch.
    /// Read once, in `new`.
    debug_sync_kernels: bool,
    /// 2026-09-25: This model's kernel handles and op scratch, dropped with
    /// the backend.
    op_cache: crate::op_cache::OpCache,
    /// 2026-09-25: Every device allocation `alloc` and `alloc_managed` made
    /// and nobody freed, keyed by pointer, with its size and allocating call
    /// site (`AllocRecord`).
    ///
    /// Model teardown ends with `sweep_unreleased`, which frees what is left:
    /// allocations no `ModelResource` released, such as weights the loaders
    /// fused into layer-owned buffers (`TransformerModel::release_pools`).
    ///
    /// Process-lifetime workspaces are not on it: the cuBLASLt, CUTLASS and
    /// FlashInfer workspaces (`ctx` in `cublaslt.rs` and `cutlass.rs`,
    /// `workspaces` in `flashinfer.rs`) call `cuMemAlloc_v2` directly, so a sweep
    /// cannot free memory a static still points at.
    live_allocs: parking_lot::Mutex<std::collections::HashMap<u64, AllocRecord>>,
    /// 2026-09-25: The trailing guard band of every live allocation padded by
    /// `alloc` (`METRALE_REDZONE=<bytes>` set, creation index at least
    /// `METRALE_REDZONE_MIN_IDX`). A zone covers
    /// `[user_ptr + user_bytes, user_ptr + user_bytes + pad_bytes)` and is
    /// filled with `METRALE_REDZONE_FILL` when allocated.
    /// [`MetraleCudaBackend::scan_redzones`] reads the zones back and reports
    /// the ones that changed: a write past the end of the buffer.
    redzones: parking_lot::Mutex<Vec<RedZone>>,
    /// 2026-09-25: The process CUDA host's stream (`MetraleRegistry::raw_stream`).
    default_stream: u64,
    /// 2026-09-25: The context current on the thread that ran `new`, after the
    /// registry load; `bind_to_thread` makes it current on another thread.
    cuda_ctx: u64,
}

impl MetraleCudaBackend {
    /// 2026-09-25: Load `ptx_modules` on GPU `ordinal` and build the backend.
    ///
    /// The module set (a `TargetPtxSet`'s `modules`, e.g. from
    /// `metrale_kernels::ptx_for_model`) is loaded afresh for each call; the
    /// CUDA context and stream are the process host's, shared by every
    /// backend (`cuda_host::host`). Fails if no context is current after the
    /// load.
    pub fn new(ordinal: usize, ptx_modules: &[(&'static str, &'static [u8])]) -> Result<Self> {
        // 2026-09-25: Reset the per-run telemetry before any kernel lookup,
        // so the kernel audit lists only this backend's lookups.
        metrale_telemetry::run_metrics::reset_for_new_run();
        let registry = MetraleRegistry::load(ordinal, ptx_modules)
            .map_err(|e| anyhow::anyhow!("MetraleRegistry load failed: {e}"))?;
        let default_stream = registry.raw_stream();
        crate::timing::install_for_stream(default_stream);

        let mut cuda_ctx: u64 = 0;
        let status = unsafe { cuCtxGetCurrent(&mut cuda_ctx) };
        if status != 0 || cuda_ctx == 0 {
            bail!("cuCtxGetCurrent failed: status {status}, ctx {cuda_ctx:#x}");
        }

        tracing::info!(
            "MetraleCudaBackend initialized on GPU {ordinal} with {} PTX modules",
            ptx_modules.len()
        );

        Ok(Self {
            live_allocs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            redzones: parking_lot::Mutex::new(Vec::new()),
            registry,
            debug_sync_kernels: std::env::var("METRALE_DEBUG_SYNC_KERNELS").as_deref() == Ok("1"),
            op_cache: crate::op_cache::OpCache::new(),
            default_stream,
            cuda_ctx,
        })
    }

    /// 2026-09-25: Number of allocations on the ledger; one lock, no copy.
    pub(crate) fn live_alloc_len(&self) -> usize {
        self.live_allocs.lock().len()
    }

    /// 2026-09-25: Drain the ledger and free every allocation on it. Returns
    /// how many there were; when there were any, one warning gives their
    /// total size and the five largest allocating call sites.
    ///
    /// `TransformerModel::release_pools` calls it last, after every
    /// `ModelResource::release`. A freed pointer is already off the ledger
    /// (`free` calls `forget_alloc` first), so nothing is freed twice.
    pub fn sweep_unreleased(&self) -> usize {
        let outstanding: Vec<(u64, AllocRecord)> = self.live_allocs.lock().drain().collect();
        let count = outstanding.len();
        if count > 0 {
            let bytes: usize = outstanding.iter().map(|(_, r)| r.bytes).sum();
            // 2026-09-25: Aggregated by call site, so a pool allocated per
            // layer from one line is one entry.
            let mut by_site: std::collections::HashMap<String, (usize, usize)> =
                std::collections::HashMap::new();
            for (_, r) in &outstanding {
                let e = by_site
                    .entry(format!("{}:{}", r.site.file(), r.site.line()))
                    .or_insert((0, 0));
                e.0 += r.bytes;
                e.1 += 1;
            }
            let mut rows: Vec<_> = by_site.into_iter().collect();
            rows.sort_by(|a, b| b.1.0.cmp(&a.1.0));
            let top: Vec<String> = rows
                .iter()
                .take(5)
                .map(|(site, (b, n))| {
                    format!("{site} ({:.1} MB x{n})", *b as f64 / (1024.0 * 1024.0))
                })
                .collect();
            tracing::warn!(
                "sweep: {count} allocation(s) totalling {:.2} GB had no owner; \
                 largest sites: {}",
                bytes as f64 / 1e9,
                top.join(", ")
            );
        }
        for (raw, _) in outstanding {
            // 2026-09-25: Not `free`: the ledger is already drained, and a
            // failed free is logged so the loop still frees the rest.
            let status = unsafe { cuMemFree_v2(raw) };
            if status != 0 && !crate::registry::is_teardown_noop(status) {
                tracing::warn!("sweep: cuMemFree failed for {raw:#x}: status {status}");
            }
        }
        count
    }

    pub fn registry(&self) -> &Arc<MetraleRegistry> {
        &self.registry
    }

    pub(crate) fn debug_sync_kernels(&self) -> bool {
        self.debug_sync_kernels
    }
}

/// 2026-09-25: Dropping the backend runs `sweep_unreleased`.
///
/// A load that fails part-way builds no model, so no teardown frees what it
/// had allocated; this does. After a model's teardown the ledger is already
/// drained (`TransformerModel::release_pools`), and the sweep frees nothing.
impl Drop for MetraleCudaBackend {
    fn drop(&mut self) {
        let swept = self.sweep_unreleased();
        if swept > 0 {
            // 2026-09-25: The warning names both possible causes, an
            // abandoned load or an allocation with no registered owner; the
            // count alone cannot tell them apart.
            tracing::warn!(
                "backend drop reclaimed {swept} allocation(s) that no owner released — \
                 expected if a load was abandoned part-way, otherwise an unregistered owner"
            );
        }
    }
}

/// 2026-09-25: [`effective_free_bytes`] for a poll, where the integrated
/// query may have failed. `None` counts as discrete, so an unanswered query
/// can only lower the reading, never add host memory to it.
pub(crate) fn polled_free_bytes(
    cu_free: usize,
    mem_available: Option<usize>,
    integrated: Option<bool>,
) -> usize {
    effective_free_bytes(cu_free, mem_available, integrated.unwrap_or(false))
}

/// 2026-09-25: `MemAvailable` from `/proc/meminfo`, in bytes. `None` when the
/// file cannot be read or has no parsable `MemAvailable` line.
fn system_available_memory_bytes() -> Option<usize> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if line.starts_with("MemAvailable:") {
            let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[path = "cuda_backend/alloc_ledger.rs"]
mod alloc_ledger;
use alloc_ledger::AllocRecord;

#[cfg(test)]
mod tests;
