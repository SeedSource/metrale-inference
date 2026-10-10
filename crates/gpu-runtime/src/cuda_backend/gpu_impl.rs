// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `impl GpuBackend for MetraleCudaBackend`: allocation, the
//! async copies, kernel launch and lookup, and one-line delegations to the
//! inherent methods in `gpu_copy.rs`, `gpu_impl_graph.rs` and `redzone.rs`.
//!
//! Each `unsafe` block is one call: a driver function declared in
//! `cuda_backend.rs`, `cudaMemcpy2DAsync`, or the `unsafe fn`
//! `MetraleRegistry::launch_on_stream`.
//! The caller provides what they need and nothing here checks it: the
//! backend's context current on the calling thread (`bind_to_thread`),
//! device pointers into live allocations, and byte counts that fit them.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - `alloc` and `alloc_managed` record every pointer they return `Ok` in the
//!   allocation ledger; `free` removes it before calling `cuMemFree_v2`.
//! - `alloc_arena` and `free_arena` never touch the ledger.

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::registry::{RawCudaFunc, cuda_error_text};
use anyhow::{Result, bail};
use cudarc::driver::LaunchConfig;

use super::{
    MetraleCudaBackend, cuMemAlloc_v2, cuMemAllocManaged, cuMemFree_v2, cuMemGetInfo_v2,
    cuMemcpyDtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuStreamSynchronize,
};
use crate::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

mod h2d;
mod raw;

use h2d::{h2d_enqueue, warn_pinned_transient_source};

/// 2026-09-25: D2H copy counter for `METRALE_D2H_TRACE=<N>`: a backtrace at
/// call N and the running count at every 10,000th call. Advanced by
/// `copy_d2h`, `copy_d2h_on_stream` and `copy_d2h_async` only while the
/// variable is set.
static D2H_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn d2h_trace_tick() {
    use std::sync::atomic::Ordering;
    // 2026-09-25: The variable is read once. Unset, this returns before the
    // counter. A value that does not parse as `u64` reads as 0: no backtrace,
    // only the every-10,000th count.
    static TARGET: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let Some(target) = *TARGET.get_or_init(|| {
        std::env::var("METRALE_D2H_TRACE")
            .ok()
            .map(|v| v.parse().unwrap_or(0))
    }) else {
        return;
    };
    let n = D2H_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if target != 0 && n == target {
        tracing::warn!(
            "METRALE_D2H_TRACE: call #{n} backtrace:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
    }
    if n.is_multiple_of(10_000) {
        tracing::warn!("METRALE_D2H_TRACE: {n} D2H copies so far (each forces a stream sync)");
    }
}

impl GpuBackend for MetraleCudaBackend {
    #[track_caller]
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let site = std::panic::Location::caller();
        let mut dptr: u64 = 0;
        // 2026-09-25: Red zone (`METRALE_REDZONE=<bytes>`, unset or 0 = off,
        // and only for creation indices >= `METRALE_REDZONE_MIN_IDX`):
        // allocate `pad` extra bytes after the buffer and return the base, so
        // the caller's pointer is what an unpadded allocation would give. The
        // pad is filled with `redzone_fill()` here and checked by
        // `scan_redzones`.
        let seq = super::ALLOC_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let pad = if seq >= super::redzone_min_idx() {
            super::redzone_bytes()
        } else {
            0
        };
        let status = unsafe { cuMemAlloc_v2(&mut dptr, bytes + pad) };
        if status != 0 {
            let mut free: usize = 0;
            let mut total: usize = 0;
            unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
            bail!(
                "cuMemAlloc_v2 failed: status {status}, requested {bytes} bytes \
                 (device reports {:.1} MB free / {:.1} GB total)",
                free as f64 / (1024.0 * 1024.0),
                total as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        if pad > 0 {
            self.alloc_poison_redzone(dptr, bytes, pad, seq)?;
        }
        self.record_alloc(DevicePtr(dptr), bytes, site);
        if bytes >= 32 * 1024 * 1024 {
            tracing::debug!(
                "alloc {:.1} MB (device ptr {dptr:#x})",
                bytes as f64 / (1024.0 * 1024.0)
            );
        }
        Ok(DevicePtr(dptr))
    }

    fn scan_redzones(&self) -> Result<usize> {
        if super::redzone_bytes() == 0 {
            return Ok(0);
        }
        MetraleCudaBackend::scan_redzones(self)
    }

    fn poison_redzones(&self, lo: usize, hi: usize) -> Result<()> {
        if super::redzone_bytes() == 0 {
            return Ok(());
        }
        MetraleCudaBackend::poison_redzones(self, lo, hi)
    }

    #[track_caller]
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        let site = std::panic::Location::caller();
        let mut dptr: u64 = 0;
        const CU_MEM_ATTACH_GLOBAL: u32 = 0x1;
        let status = unsafe { cuMemAllocManaged(&mut dptr, bytes, CU_MEM_ATTACH_GLOBAL) };
        if status != 0 {
            bail!(
                "cuMemAllocManaged failed: status {status}, requested {bytes} bytes. \
                 Check system swap space: swapon --show"
            );
        }
        self.record_alloc(DevicePtr(dptr), bytes, site);
        Ok(DevicePtr(dptr))
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // 2026-09-25: Off the ledger before the free: an entry left behind
        // would be freed again by `sweep_unreleased`.
        self.forget_alloc(ptr);
        if super::redzone_bytes() > 0 {
            self.forget_redzone(ptr.0);
        }
        let status = unsafe { cuMemFree_v2(ptr.0) };
        // 2026-09-25: A status meaning the context is already gone
        // (`registry::is_teardown_noop`: 4, 201, 709) is not an error.
        if status != 0 && !crate::registry::is_teardown_noop(status) {
            bail!("cuMemFree_v2 failed: status {status}, ptr {ptr}");
        }
        Ok(())
    }

    fn live_bytes(&self) -> Option<usize> {
        Some(MetraleCudaBackend::live_bytes(self))
    }

    /// 2026-10-03: VMM: reserve address space now, map granules on `ensure_mapped`. Mapped
    /// bytes are counted by `live_bytes` (`lazy_buffer::vmm_mapped_bytes`), not per site.
    #[cfg(not(metrale_scale))]
    #[track_caller]
    fn alloc_lazy(
        &self,
        bytes: usize,
        budget: Option<std::sync::Arc<crate::lazy_buffer::MapBudget>>,
    ) -> Result<crate::lazy_buffer::LazyBuffer> {
        super::vmm::reserve(bytes, budget)
    }

    fn alloc_report(&self, top_n: usize, min_mb: usize) -> Option<String> {
        Some(MetraleCudaBackend::alloc_report(self, top_n, min_mb))
    }

    fn sweep_unreleased(&self) -> usize {
        MetraleCudaBackend::sweep_unreleased(self)
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        MetraleCudaBackend::copy_h2d_impl(self, src, dst)
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        d2h_trace_tick();
        MetraleCudaBackend::copy_d2h_impl(self, src, dst)
    }

    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        d2h_trace_tick();
        MetraleCudaBackend::copy_d2h_on_stream_impl(self, src, dst, stream)
    }

    fn copy_d2h_async(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // 2026-09-25: No wait here, unlike `copy_d2h` and
        // `copy_d2h_on_stream`. The caller must synchronise `stream` before
        // reading `dst`.
        d2h_trace_tick();
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src.0, dst.len(), stream)
        };
        if status != 0 {
            bail!("cuMemcpyDtoHAsync_v2 (async) failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if metrale_telemetry::launch_trace::on() {
            metrale_telemetry::launch_trace::record(metrale_telemetry::launch_trace::Entry {
                kind: "d2d",
                func: 0,
                grid: [0, 0, 0],
                block: [0, 0, 0],
                smem: 0,
                args: vec![src.0, dst.0, bytes as u64],
            });
        }
        MetraleCudaBackend::copy_d2d_impl(self, src, dst, bytes)
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let raw_func = RawCudaFunc(func.0 as *mut c_void);
        let cfg = LaunchConfig {
            grid_dim: (grid[0], grid[1], grid[2]),
            block_dim: (block[0], block[1], block[2]),
            shared_mem_bytes: shared_mem,
        };
        let registry = self.registry();
        let span = crate::timing::kernel_begin(self, func.0, stream);
        let launched =
            unsafe { registry.launch_on_stream(raw_func, cfg, stream, params) }.map_err(|e| {
                // 2026-09-25: Probe the context and latch the fault if it is
                // gone (`fault_probe`); the error is returned either way.
                super::fault_probe::note_failure("kernel launch", &e.to_string());
                anyhow::anyhow!("Kernel launch failed: {e}")
            });
        crate::timing::kernel_end(span, stream);
        launched
    }

    fn stream_is_capturing(&self, stream: u64) -> bool {
        // 2026-09-25: Under `metrale_scale` `cuStreamIsCapturing` is not
        // declared (`cuda_backend.rs`), so report not capturing.
        #[cfg(metrale_scale)]
        {
            let _ = stream;
            false
        }
        #[cfg(not(metrale_scale))]
        {
            let mut status: u32 = 0;
            // 2026-09-25: CU_STREAM_CAPTURE_STATUS_NONE = 0. A failed query
            // counts as capturing.
            let rc = unsafe { super::cuStreamIsCapturing(stream, &mut status) };
            rc != 0 || status != 0
        }
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        metrale_telemetry::global().stream_sync();
        let status = unsafe { cuStreamSynchronize(stream) };
        if status != 0 {
            bail!("cuStreamSynchronize failed: {}", cuda_error_text(status));
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        self.default_stream
    }

    fn op_cache(&self) -> &crate::op_cache::OpCache {
        &self.op_cache
    }

    fn debug_sync_kernels(&self) -> bool {
        MetraleCudaBackend::debug_sync_kernels(self)
    }

    fn kernel_registry(&self) -> Option<std::sync::Arc<crate::registry::MetraleRegistry>> {
        Some(self.registry().clone())
    }

    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        // 2026-09-25: The caller's `file:line`, carried through by
        // `#[track_caller]` here and on the trait declaration.
        let site = std::panic::Location::caller();
        // 2026-09-25: A fresh `OnceLock` per call: nothing is cached across
        // calls.
        let cache: OnceLock<RawCudaFunc> = OnceLock::new();
        let registry = self.registry();
        match registry.raw_function_cached(&cache, module, func_name) {
            Ok(raw) => {
                metrale_telemetry::kernel_audit::record(module, func_name, true, site);
                metrale_telemetry::launch_trace::name_kernel(raw.0 as u64, module, func_name);
                Ok(KernelHandle(raw.0 as u64))
            }
            Err(e) => {
                // 2026-09-25: A failed lookup is recorded in the kernel audit
                // before the error is returned.
                metrale_telemetry::kernel_audit::record(module, func_name, false, site);
                Err(anyhow::anyhow!("Kernel lookup {module}::{func_name}: {e}"))
            }
        }
    }

    fn has_module(&self, module: &str) -> bool {
        self.registry().has_module(module)
    }

    fn copy_h2d_async(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
        h2d_enqueue(src, dst, stream)?;
        // 2026-09-25: The trait lets the caller drop `src` on return. For a
        // page-locked source (`pinned_hosts::is_pinned`) the copy reads `src`
        // after the enqueue, so wait for `stream` before returning.
        if crate::pinned_hosts::is_pinned(src) {
            warn_pinned_transient_source();
            let sync = unsafe { cuStreamSynchronize(stream) };
            if sync != 0 {
                bail!(
                    "cuStreamSynchronize after pinned-source H2D failed: {}",
                    cuda_error_text(sync)
                );
            }
        }
        Ok(())
    }

    fn copy_h2d_async_retained(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
        // 2026-09-25: The caller keeps `src` alive until its next sync on
        // `stream`, so no wait is added.
        h2d_enqueue(src, dst, stream)
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let status = unsafe { cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, stream) };
        if status != 0 {
            // 2026-09-25: Backtrace for the same reason as in `copy_d2d_impl`.
            tracing::error!(
                "copy_d2d_async failed (status {status}) at:\n{}",
                std::backtrace::Backtrace::force_capture()
            );
            bail!("cuMemcpyDtoDAsync_v2 (copy_d2d_async) failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2d_2d_async(
        &self,
        src: DevicePtr,
        src_pitch: usize,
        dst: DevicePtr,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
        stream: u64,
    ) -> Result<()> {
        raw::memcpy_2d_async(src, src_pitch, dst, dst_pitch, width_bytes, height, stream)
    }

    fn begin_capture(&self, stream: u64) -> Result<()> {
        self.begin_capture_cu(stream)
    }
    fn end_capture(&self, stream: u64) -> Result<GraphHandle> {
        self.end_capture_cu(stream)
    }

    fn abort_capture_if_active(&self, stream: u64) {
        self.abort_capture_if_active_cu(stream)
    }

    fn launch_graph(&self, graph: GraphHandle, stream: u64) -> Result<()> {
        self.launch_graph_cu(graph, stream)
    }
    fn destroy_graph(&self, graph: GraphHandle) -> Result<()> {
        self.destroy_graph_cu(graph)
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.memset_cu(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        if metrale_telemetry::launch_trace::on() {
            metrale_telemetry::launch_trace::record(metrale_telemetry::launch_trace::Entry {
                kind: "memset",
                func: 0,
                grid: [0, 0, 0],
                block: [0, 0, 0],
                smem: 0,
                args: vec![ptr.0, value as u64, bytes as u64],
            });
        }
        self.memset_async_cu(ptr, value, bytes, stream)
    }
    fn total_memory(&self) -> Result<usize> {
        self.total_memory_cu()
    }
    fn free_memory(&self) -> Result<usize> {
        self.free_memory_cu()
    }
    fn device_free_memory(&self) -> Result<usize> {
        self.device_free_memory_cu()
    }
    fn live_alloc_count(&self) -> usize {
        self.live_alloc_len()
    }
    fn sm_count(&self) -> Result<u32> {
        self.sm_count_cu()
    }
    fn max_active_blocks_per_sm(
        &self,
        kernel: KernelHandle,
        block: u32,
        dyn_smem: u32,
    ) -> Result<u32> {
        self.max_active_blocks_cu(kernel.0, block, dyn_smem)
    }
    fn create_stream(&self) -> Result<u64> {
        self.create_stream_cu()
    }
    fn bind_to_thread(&self) -> Result<()> {
        self.bind_to_thread_cu()
    }
    fn create_event(&self) -> Result<u64> {
        self.create_event_cu()
    }
    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.record_event_cu(event, stream)
    }
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event_cu(stream, event)
    }
    fn event_synchronize(&self, event: u64) -> Result<()> {
        self.event_synchronize_cu(event)
    }
    fn event_query(&self, event: u64) -> Result<bool> {
        self.event_query_cu(event)
    }
    fn destroy_event(&self, event: u64) -> Result<()> {
        self.destroy_event_cu(event)
    }
    fn host_ptr_to_device(&self, host: *mut u8) -> Result<DevicePtr> {
        let mut dptr: u64 = 0;
        let status =
            unsafe { super::cuMemHostGetDevicePointer_v2(&mut dptr, host as *mut c_void, 0) };
        if status != 0 {
            bail!("cuMemHostGetDevicePointer_v2 failed: status {status}");
        }
        Ok(DevicePtr(dptr))
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        if bytes >= 32 * 1024 * 1024 {
            tracing::debug!(
                "alloc_host_pinned {:.1} MB",
                bytes as f64 / (1024.0 * 1024.0)
            );
        }
        self.alloc_host_pinned_cu(bytes)
    }
    fn free_host_pinned(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        self.free_host_pinned_cu(ptr, _bytes)
    }

    fn alloc_arena(&self, bytes: usize) -> Result<DevicePtr> {
        let mut dptr: u64 = 0;
        let status = unsafe { cuMemAlloc_v2(&mut dptr, bytes) };
        if status != 0 {
            let mut free: usize = 0;
            let mut total: usize = 0;
            unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
            bail!(
                "cuMemAlloc_v2 (arena) failed: status {status}, requested {bytes} bytes \
                 (device reports {:.1} GB free / {:.1} GB total)",
                free as f64 / (1024.0 * 1024.0 * 1024.0),
                total as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        tracing::info!(
            "arena: {:.2} GiB of device memory at {dptr:#x}, off the allocation ledger",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        Ok(DevicePtr(dptr))
    }
    fn free_arena(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        let status = unsafe { cuMemFree_v2(ptr.0) };
        if status != 0 {
            // 2026-09-25: Logged, not returned, for any nonzero status.
            tracing::warn!("cuMemFree_v2 (arena) returned status {status}");
        }
        Ok(())
    }
}
