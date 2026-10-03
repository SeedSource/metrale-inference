// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `GpuBackend` trait: allocation, copies, kernel launches,
//! graphs, events and memory queries for one device.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use anyhow::Result;
use std::fmt;
use std::sync::atomic::Ordering;
mod baseline;
pub use baseline::{baseline_free_bytes, set_baseline_free_bytes};
mod handles;
pub use handles::{DevicePtr, GraphHandle, KernelArg, KernelHandle};

pub use crate::gpu_args::pack_kernel_args;

/// 2026-09-25: The device interface the model code calls. Implementations:
/// `MetraleCudaBackend` (`cuda` feature), `MetalGpuBackend` (`metal` feature) and
/// `MockGpuBackend` (tests and the `test-utils` feature).
pub trait GpuBackend: Send + Sync {
    /// 2026-09-25: Allocate `bytes` of device memory.
    ///
    /// `#[track_caller]` is on the declaration as well as the CUDA impl, so the
    /// allocation ledger records the caller's location through `&dyn GpuBackend`.
    #[track_caller]
    fn alloc(&self, bytes: usize) -> Result<DevicePtr>;

    /// 2026-09-25: Allocate managed memory: `cuMemAllocManaged` on CUDA, plain
    /// `alloc` on Metal and the mock.
    #[track_caller]
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr>;

    /// 2026-09-25: Free device memory. A null pointer is a no-op.
    fn free(&self, ptr: DevicePtr) -> Result<()>;

    /// 2026-09-25: Free every allocation still on this backend's ledger and return
    /// how many there were. The default returns 0 and frees nothing; only the
    /// CUDA backend overrides it.
    fn sweep_unreleased(&self) -> usize {
        0
    }

    /// 2026-09-25: Live device bytes this backend has allocated and not freed:
    /// `Some` on CUDA and the mock, `None` (the default) on Metal.
    fn live_bytes(&self) -> Option<usize> {
        None
    }

    /// 2026-09-25: Live device memory by allocating call site; `None` (the
    /// default) on every backend but CUDA.
    fn alloc_report(&self, _top_n: usize, _min_mb: usize) -> Option<String> {
        None
    }

    /// 2026-09-25: Copy from host to device; returns when the copy is done.
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()>;

    /// 2026-09-25: Copy from device to host; returns when the copy is done.
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()>;

    /// 2026-09-25: Blocking device-to-host copy ordered after the work already
    /// on `stream`. `copy_d2h` orders only against the default stream; use this
    /// to read what kernels on another stream just wrote. The CUDA backend
    /// enqueues the copy on `stream` and waits; the default synchronises
    /// `stream` and then calls `copy_d2h`.
    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        self.synchronize(stream)?;
        self.copy_d2h(src, dst)
    }

    /// 2026-09-25: Copy device to device; returns when the copy is done.
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()>;

    /// 2026-09-25: Launch a kernel with untyped parameter pointers. The Metal
    /// backend refuses this; use `launch_typed`.
    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()>;

    /// 2026-09-25: Launch a kernel with typed arguments.
    ///
    /// The default packs them with `pack_kernel_args` and calls `launch`. The
    /// Metal backend binds each `KernelArg::Buffer` with `setBuffer:offset:atIndex:`
    /// and each `KernelArg::Bytes` with `setBytes:length:atIndex:`; the mock
    /// records them.
    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        // 2026-09-25: With `launch_trace` on, record each launch so two steps can
        // be diffed; a `Bytes` arg is recorded as its first 8 bytes.
        if metrale_telemetry::launch_trace::on() {
            let words = args
                .iter()
                .map(|a| match a {
                    KernelArg::Buffer(p) => p.0,
                    KernelArg::Bytes(b) => {
                        let mut w = [0u8; 8];
                        let n = b.len().min(8);
                        w[..n].copy_from_slice(&b[..n]);
                        u64::from_le_bytes(w)
                    }
                })
                .collect();
            metrale_telemetry::launch_trace::record(metrale_telemetry::launch_trace::Entry {
                kind: "kernel",
                func: func.0,
                grid,
                block,
                smem: shared_mem,
                args: words,
            });
        }
        // 2026-09-25: `params` points into `storage`, which lives until
        // `launch` returns.
        let (storage, starts) = pack_kernel_args(args);
        let mut params: Vec<*mut std::ffi::c_void> = starts
            .iter()
            .map(|&i| &storage[i] as *const u64 as *mut std::ffi::c_void)
            .collect();
        self.launch(func, grid, block, shared_mem, stream, &mut params)
    }

    /// 2026-09-25: Whether `stream` is inside an active CUDA-graph capture. Check
    /// it before a sync or D2H on a stream that may be capturing: those invalidate
    /// the capture (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`, 901). The CUDA
    /// backend reports a failed query as capturing, and `metrale_scale` builds
    /// always report false; the default is false.
    fn stream_is_capturing(&self, _stream: u64) -> bool {
        false
    }

    /// 2026-09-25: Block until all work queued on `stream` has completed.
    fn synchronize(&self, stream: u64) -> Result<()>;

    /// 2026-09-25: Read every allocation's trailing guard band back and return how
    /// many no longer hold the fill byte. `Ok(0)` when `METRALE_REDZONE` is unset,
    /// and on every backend but CUDA.
    fn scan_redzones(&self) -> Result<usize> {
        Ok(0)
    }

    /// 2026-09-25: Refill the guard bands of allocations with index in `[lo, hi)`
    /// with `0xEE` and the rest with `0x00`; nothing is allocated, moved or
    /// resized. A no-op when `METRALE_REDZONE` is unset, and on every backend but
    /// CUDA.
    fn poison_redzones(&self, _lo: usize, _hi: usize) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: The default stream handle.
    fn default_stream(&self) -> u64;

    /// 2026-09-25: Look up a kernel function by module and function name.
    ///
    /// `#[track_caller]` is on the declaration so the CUDA backend's kernel audit
    /// records the caller's location through `&dyn GpuBackend`.
    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle>;

    /// 2026-09-25: Whether `module` is loaded in this backend. Ask before looking
    /// up a kernel that only some targets carry: the CUDA backend's kernel audit
    /// records every failed lookup.
    fn has_module(&self, module: &str) -> bool;

    /// 2026-09-25: This backend's memoized kernel handles and scratch
    /// allocations. It has no default, so every backend owns its own cache.
    fn op_cache(&self) -> &crate::op_cache::OpCache;

    /// 2026-09-25: Whether `KernelLaunch::launch` synchronises after every
    /// launch, so an asynchronous fault is reported at the kernel that caused it.
    /// The CUDA backend reads `METRALE_DEBUG_SYNC_KERNELS=1` once, when it is
    /// built; the default is false.
    fn debug_sync_kernels(&self) -> bool {
        false
    }

    /// 2026-09-25: This backend's kernel module registry, for callers that need
    /// more than a kernel handle. `None` (the default) on every backend but CUDA.
    #[cfg(feature = "cuda")]
    fn kernel_registry(&self) -> Option<std::sync::Arc<crate::registry::MetraleRegistry>> {
        None
    }

    /// 2026-09-25: Host-to-device copy on `stream`; `src` may be dropped or
    /// overwritten as soon as this returns.
    ///
    /// The CUDA backend synchronises `stream` after the enqueue when `src` is
    /// page-locked memory registered in [`crate::pinned_hosts`]. The default is
    /// the blocking `copy_h2d`. Use [`GpuBackend::copy_h2d_async_retained`] when
    /// the source outlives the next synchronisation.
    fn copy_h2d_async(&self, src: &[u8], dst: DevicePtr, _stream: u64) -> Result<()> {
        self.copy_h2d(src, dst)
    }

    /// 2026-09-25: Host-to-device copy on `stream` for a source the caller keeps
    /// alive: `src` must stay valid and unchanged until the next synchronisation
    /// point on `stream`. The CUDA backend adds no synchronisation; the default
    /// is `copy_h2d_async`.
    fn copy_h2d_async_retained(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
        self.copy_h2d_async(src, dst, stream)
    }

    /// 2026-09-25: Device-to-host copy enqueued on `stream` without waiting.
    /// `dst` must stay valid, and must not be read or reused, until the next
    /// synchronisation point on `stream`. On CUDA, `copy_d2h` and
    /// `copy_d2h_on_stream` each wait inside the call, so this is the D2H to use
    /// for many chunks followed by one `synchronize`. The default, used by Metal,
    /// is the blocking `copy_d2h`; the mock copies at once and counts the call.
    fn copy_d2h_async(&self, src: DevicePtr, dst: &mut [u8], _stream: u64) -> Result<()> {
        self.copy_d2h(src, dst)
    }

    /// 2026-09-25: Device-to-device copy enqueued on `stream`. The default is
    /// the blocking `copy_d2d`.
    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        _stream: u64,
    ) -> Result<()> {
        self.copy_d2d(src, dst, bytes)
    }

    /// 2026-09-25: Pitched device-to-device copy: `height` rows of `width_bytes`,
    /// source rows `src_pitch` apart and destination rows `dst_pitch` apart. The
    /// default loops `copy_d2d_async` per row; the CUDA backend issues one
    /// `cudaMemcpy2DAsync`.
    #[allow(clippy::too_many_arguments)]
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
        for r in 0..height {
            self.copy_d2d_async(
                src.offset(r * src_pitch),
                dst.offset(r * dst_pitch),
                width_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: Begin capturing the work enqueued on `stream` into a graph
    /// (`cuStreamBeginCapture` on CUDA); captured work is recorded, not run. The
    /// default does nothing.
    fn begin_capture(&self, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: End the capture and return the instantiated graph. The
    /// default returns `GraphHandle(0)`.
    fn end_capture(&self, _stream: u64) -> Result<GraphHandle> {
        Ok(GraphHandle(0))
    }

    /// 2026-09-25: Replay a captured graph on `stream`.
    fn launch_graph(&self, _graph: GraphHandle, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Destroy an instantiated graph.
    fn destroy_graph(&self, _graph: GraphHandle) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Best effort: end any capture active on `stream` and discard
    /// the partial graph, so the stream runs work again. Call it on an error path
    /// that left a `begin_capture`/`end_capture` region early. The default does
    /// nothing.
    fn abort_capture_if_active(&self, _stream: u64) {}

    /// 2026-09-25: Set device memory to a byte value; returns when done.
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()>;

    /// 2026-09-25: Set device memory to a byte value, enqueued on `stream` on
    /// CUDA. Metal and the mock write at once.
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()>;

    /// 2026-09-25: Total device memory in bytes.
    fn total_memory(&self) -> Result<usize>;

    /// 2026-09-25: Free device memory in bytes. On an integrated GPU the CUDA
    /// backend reports the larger of the driver's figure and host `MemAvailable`.
    fn free_memory(&self) -> Result<usize>;

    /// 2026-09-25: Free device memory as the driver reports it
    /// (`cuMemGetInfo` on CUDA), never substituted by host memory. The default
    /// is `free_memory`.
    fn device_free_memory(&self) -> Result<usize> {
        self.free_memory()
    }

    /// 2026-09-25: Number of live device allocations on this backend: a count,
    /// not bytes. The default, 0, means not tracked (Metal).
    fn live_alloc_count(&self) -> usize {
        0
    }

    /// 2026-09-25: Number of streaming multiprocessors on the device. The CUDA
    /// backend asks the driver, the mock returns 48, and Metal returns an error.
    /// Resolve it once at construction, not per launch.
    fn sm_count(&self) -> Result<u32>;

    /// 2026-09-25: Create a stream. The default returns 0, the default stream.
    fn create_stream(&self) -> Result<u64> {
        Ok(0)
    }

    /// 2026-09-25: Make the backend's CUDA context current on this thread. Call it
    /// on any thread other than the one that built the backend before using the
    /// GPU. The default does nothing.
    fn bind_to_thread(&self) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Create an event for ordering work across streams. The
    /// default returns 0.
    fn create_event(&self) -> Result<u64> {
        Ok(0)
    }

    /// 2026-09-25: Record `event` at the current point of `stream`'s work.
    fn record_event(&self, _event: u64, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Make `stream` wait for `event` on the device; the host does
    /// not block.
    fn stream_wait_event(&self, _stream: u64, _event: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Block the host until the work recorded before `event` has
    /// completed. Unlike `synchronize(stream)`, it does not wait for work
    /// enqueued after the event.
    fn event_synchronize(&self, _event: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Whether the work recorded before `event` has completed; never
    /// blocks. The default returns `true`.
    fn event_query(&self, _event: u64) -> Result<bool> {
        Ok(true)
    }

    /// 2026-09-25: Destroy an event.
    fn destroy_event(&self, _event: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: The device address of a page-locked host pointer from
    /// [`Self::alloc_host_pinned`] (`cuMemHostGetDevicePointer_v2` on CUDA), so a
    /// kernel can write into host-visible memory. The default returns an error.
    fn host_ptr_to_device(&self, _host: *mut u8) -> Result<DevicePtr> {
        anyhow::bail!("host_ptr_to_device: not supported by this backend")
    }

    /// 2026-09-25: Allocate `bytes` of host memory for H2D staging: page-locked on
    /// CUDA (`cuMemAllocHost_v2`), a shared `MTLBuffer` on Metal, and a 64-byte
    /// aligned heap block by default. Release it with `free_host_pinned`.
    ///
    /// The region is zeroed on every backend, so a caller may form a `&[u8]` over
    /// all of it, padding included.
    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        crate::host_heap::alloc_zeroed(bytes)
    }

    /// 2026-09-25: Free memory from `alloc_host_pinned`; `bytes` must be the
    /// allocated size.
    fn free_host_pinned(&self, ptr: *mut u8, bytes: usize) -> Result<()> {
        crate::host_heap::free(ptr, bytes)
    }

    /// 2026-10-03: A [`crate::lazy_buffer::LazyBuffer`] of `bytes`, physical memory mapped on
    /// demand by `ensure_mapped` and charged to `budget`. The CUDA backend reserves virtual
    /// address space and maps granules (VMM); the default allocates the whole buffer eagerly
    /// through `alloc` and only does the budget accounting, so CPU tests exercise the same
    /// charge and refusal logic. Release it with `LazyBuffer::release`.
    #[track_caller]
    fn alloc_lazy(
        &self,
        bytes: usize,
        budget: Option<std::sync::Arc<crate::lazy_buffer::MapBudget>>,
    ) -> Result<crate::lazy_buffer::LazyBuffer> {
        let g = crate::lazy_buffer::DEFAULT_GRANULE;
        let reserved = crate::lazy_buffer::round_up_to_granule(bytes.max(1), g);
        let base = self.alloc(reserved)?;
        Ok(crate::lazy_buffer::LazyBuffer::eager(
            base, reserved, g, budget,
        ))
    }

    /// 2026-09-25: Device memory for a weight arena, freed with `free_arena`. The
    /// CUDA backend allocates it off the allocation ledger, whose `live_bytes`
    /// `factory::build` counts as this process's own memory when it sizes the KV
    /// cache. The default is the ledgered `alloc`.
    fn alloc_arena(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes)
    }
    fn free_arena(&self, ptr: DevicePtr) -> Result<()> {
        self.free(ptr)
    }
}

#[cfg(any(test, feature = "test-utils"))]
pub mod mock;

#[cfg(test)]
#[path = "gpu_tests.rs"]
mod tests;
