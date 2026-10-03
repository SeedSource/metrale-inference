// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Graph capture and replay, streams, events, memset, memory
//! queries and pinned host allocation for [`MetraleCudaBackend`], as inherent
//! `_cu` methods that the `GpuBackend` impl in `gpu_impl.rs` delegates to.
//!
//! Each `unsafe` block is one driver call, except the `write_bytes` in
//! `alloc_host_pinned_cu`; the caller requirements in `gpu_impl.rs`'s module
//! doc apply here too.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - Every region `alloc_host_pinned_cu` returns is zeroed, and when it is
//!   non-empty it is registered in `pinned_hosts`; `free_host_pinned_cu`
//!   unregisters a pointer before freeing it.

use std::ffi::c_void;

use anyhow::{Result, bail};

use super::{
    MetraleCudaBackend, cuCtxGetDevice, cuCtxSetCurrent, cuDeviceGetAttribute, cuEventCreate,
    cuEventDestroy_v2, cuEventQuery, cuEventRecord, cuEventSynchronize, cuGraphDestroy,
    cuGraphExecDestroy, cuGraphLaunch, cuMemAllocHost_v2, cuMemFreeHost, cuMemGetInfo_v2,
    cuMemsetD8Async, cuStreamBeginCapture, cuStreamCreate, cuStreamEndCapture, cuStreamSynchronize,
    cuStreamWaitEvent,
};
use crate::gpu::{DevicePtr, GraphHandle};

impl MetraleCudaBackend {
    pub(super) fn begin_capture_cu(&self, stream: u64) -> Result<()> {
        // 2026-09-25: Mode 2 is CU_STREAM_CAPTURE_MODE_RELAXED.
        let status = unsafe { cuStreamBeginCapture(stream, 2) };
        if status != 0 {
            bail!("cuStreamBeginCapture failed: status {status}");
        }
        metrale_telemetry::global().graph_capture();
        crate::lazy_buffer::note_capture_begin(stream);
        Ok(())
    }

    /// 2026-09-25: End any capture on `stream`, for use after an error
    /// mid-capture. The status is ignored; a graph is destroyed only when the
    /// call succeeded and returned one.
    pub(super) fn abort_capture_if_active_cu(&self, stream: u64) {
        let mut graph: u64 = 0;
        let status = unsafe { cuStreamEndCapture(stream, &mut graph) };
        crate::lazy_buffer::note_capture_end(stream);
        if status == 0 && graph != 0 {
            unsafe { cuGraphDestroy(graph) };
        }
    }

    pub(super) fn end_capture_cu(&self, stream: u64) -> Result<GraphHandle> {
        let mut graph: u64 = 0;
        let status = unsafe { cuStreamEndCapture(stream, &mut graph) };
        crate::lazy_buffer::note_capture_end(stream);
        if status != 0 {
            bail!("cuStreamEndCapture failed: status {status}");
        }
        // 2026-09-25: Which instantiate symbol is declared depends on
        // `metrale_scale`; see `cuda_backend.rs`.
        let mut graph_exec: u64 = 0;
        #[cfg(not(metrale_scale))]
        let status = unsafe { super::cuGraphInstantiateWithFlags(&mut graph_exec, graph, 0) };
        #[cfg(metrale_scale)]
        let status = unsafe { super::cuGraphInstantiate(&mut graph_exec, graph, 0) };
        if status != 0 {
            unsafe { cuGraphDestroy(graph) };
            bail!("cuGraphInstantiate failed: status {status}");
        }
        // 2026-09-25: Only the executable graph is kept; the template is
        // destroyed on both paths.
        unsafe { cuGraphDestroy(graph) };
        Ok(GraphHandle(graph_exec))
    }

    pub(super) fn launch_graph_cu(&self, graph: GraphHandle, stream: u64) -> Result<()> {
        let status = unsafe { cuGraphLaunch(graph.0, stream) };
        if status != 0 {
            bail!("cuGraphLaunch failed: status {status}");
        }
        metrale_telemetry::global().graph_replay();
        Ok(())
    }

    pub(super) fn destroy_graph_cu(&self, graph: GraphHandle) -> Result<()> {
        if graph.0 != 0 {
            let status = unsafe { cuGraphExecDestroy(graph.0) };
            if status != 0 {
                bail!("cuGraphExecDestroy failed: status {status}");
            }
        }
        Ok(())
    }

    pub(super) fn memset_cu(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        let status = unsafe { cuMemsetD8Async(ptr.0, value, bytes, self.default_stream) };
        if status != 0 {
            // 2026-09-25: Probe the context and latch the fault if it is gone
            // (`fault_probe`).
            super::fault_probe::note_failure("cuMemsetD8Async", &format!("status {status}"));
            bail!("cuMemsetD8Async failed: status {status}");
        }
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            super::fault_probe::note_failure(
                "cuStreamSynchronize after memset",
                &format!("status {sync}"),
            );
            bail!("cuStreamSynchronize after memset failed: status {sync}");
        }
        Ok(())
    }

    pub(super) fn memset_async_cu(
        &self,
        ptr: DevicePtr,
        value: u8,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let status = unsafe { cuMemsetD8Async(ptr.0, value, bytes, stream) };
        if status != 0 {
            super::fault_probe::note_failure("cuMemsetD8Async", &format!("status {status}"));
            bail!("cuMemsetD8Async failed: status {status}");
        }
        Ok(())
    }

    pub(super) fn total_memory_cu(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {status}");
        }
        Ok(total)
    }

    /// 2026-09-25: `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` of the current
    /// context's device. Fails, rather than returning a default, when the
    /// driver does not answer or reports a count <= 0.
    pub(super) fn sm_count_cu(&self) -> Result<u32> {
        const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: u32 = 16;
        let mut dev: i32 = 0;
        let status = unsafe { cuCtxGetDevice(&mut dev) };
        if status != 0 {
            bail!("cuCtxGetDevice failed: status {status}");
        }
        let mut count: i32 = 0;
        let status = unsafe {
            cuDeviceGetAttribute(&mut count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)
        };
        if status != 0 {
            bail!("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed: status {status}");
        }
        if count <= 0 {
            bail!("driver reported {count} multiprocessors on device {dev}");
        }
        Ok(count as u32)
    }

    /// 2026-09-25: The driver's free figure (`cuMemGetInfo_v2`) alone, without
    /// the host `MemAvailable` leg that `free_memory_cu` may take.
    pub(super) fn device_free_memory_cu(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {status}");
        }
        Ok(free)
    }

    pub(super) fn free_memory_cu(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {status}");
        }
        // 2026-09-25: On an integrated GPU (`device_is_integrated`) the
        // result is the larger of the driver figure and host `MemAvailable`;
        // on a discrete GPU, host RAM is a separate pool and only the driver
        // figure counts (`effective_free_bytes`). So this is not a pure
        // driver query; `device_free_memory_cu` is. When `MemAvailable` is
        // readable, a debug line logs both figures and which one was
        // returned. A failed integrated query is an error here.
        let mem_available = super::system_available_memory_bytes();
        let integrated = super::device_is_integrated()?;
        if let Some(avail) = mem_available {
            let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::debug!(
                "free_memory legs: cuMemGetInfo={:.3} GiB, MemAvailable={:.3} GiB, \
                 integrated={}, winner={}, spread={:.3} GiB",
                gib(free),
                gib(avail),
                integrated,
                if integrated && avail > free {
                    "MemAvailable"
                } else {
                    "cuMemGetInfo"
                },
                gib(avail.abs_diff(free)),
            );
        }
        Ok(super::effective_free_bytes(free, mem_available, integrated))
    }

    pub(super) fn create_stream_cu(&self) -> Result<u64> {
        let mut stream: u64 = 0;
        // 2026-09-25: Flag 1 is CU_STREAM_NON_BLOCKING: no implicit
        // synchronisation with stream 0.
        let status = unsafe { cuStreamCreate(&mut stream, 1) };
        if status != 0 {
            bail!("cuStreamCreate failed: status {status}");
        }
        Ok(stream)
    }

    pub(super) fn bind_to_thread_cu(&self) -> Result<()> {
        let status = unsafe { cuCtxSetCurrent(self.cuda_ctx) };
        if status != 0 {
            bail!("cuCtxSetCurrent failed: status {status}");
        }
        Ok(())
    }

    pub(super) fn create_event_cu(&self) -> Result<u64> {
        let mut event: u64 = 0;
        // 2026-09-25: Flag 0x02 is CU_EVENT_DISABLE_TIMING.
        let status = unsafe { cuEventCreate(&mut event, 0x02) };
        if status != 0 {
            bail!("cuEventCreate failed: status {status}");
        }
        Ok(event)
    }

    pub(super) fn record_event_cu(&self, event: u64, stream: u64) -> Result<()> {
        let status = unsafe { cuEventRecord(event, stream) };
        if status != 0 {
            bail!("cuEventRecord failed: status {status}");
        }
        Ok(())
    }

    pub(super) fn stream_wait_event_cu(&self, stream: u64, event: u64) -> Result<()> {
        let status = unsafe { cuStreamWaitEvent(stream, event, 0) };
        if status != 0 {
            bail!("cuStreamWaitEvent failed: status {status}");
        }
        Ok(())
    }

    pub(super) fn event_synchronize_cu(&self, event: u64) -> Result<()> {
        // 2026-09-25: Blocks the calling thread until the work recorded
        // against `event` has completed.
        let status = unsafe { cuEventSynchronize(event) };
        if status != 0 {
            bail!("cuEventSynchronize failed: status {status}");
        }
        Ok(())
    }

    /// 2026-09-25: `cuEventQuery`: `Ok(true)` once the work recorded against
    /// `event` has completed, `Ok(false)` while it is pending, `Err` for any
    /// other status. Never blocks. `AsyncDeviceIo::wait` polls it
    /// `POLL_SPINS` times before blocking in `event_synchronize`.
    pub(super) fn event_query_cu(&self, event: u64) -> Result<bool> {
        // 2026-09-25: 0 is CUDA_SUCCESS, 600 is CUDA_ERROR_NOT_READY.
        match unsafe { cuEventQuery(event) } {
            0 => Ok(true),
            600 => Ok(false),
            status => bail!("cuEventQuery failed: status {status}"),
        }
    }

    pub(super) fn destroy_event_cu(&self, event: u64) -> Result<()> {
        if event != 0 {
            let status = unsafe { cuEventDestroy_v2(event) };
            if status != 0 {
                bail!("cuEventDestroy_v2 failed: status {status}");
            }
        }
        Ok(())
    }

    pub(super) fn alloc_host_pinned_cu(&self, bytes: usize) -> Result<*mut u8> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cuMemAllocHost_v2(&mut ptr, bytes) };
        if status != 0 {
            bail!("cuMemAllocHost_v2 failed: status {status}, requested {bytes} bytes");
        }
        // 2026-09-25: A null pointer with a success status is refused:
        // `write_bytes` below requires a non-null pointer even for zero bytes.
        if ptr.is_null() {
            bail!("cuMemAllocHost_v2 reported success but returned null for {bytes} bytes");
        }
        // 2026-09-25: `GpuBackend::alloc_host_pinned` returns a zeroed
        // region (its doc, and the default's `host_heap::alloc_zeroed`), so
        // zero it here.
        // SAFETY: `cuMemAllocHost_v2` returned success, so `ptr` is a valid,
        // uniquely-owned, writable page-locked region of exactly `bytes`.
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes) };
        // 2026-09-25: Registered so `copy_h2d_async` can recognise it as a
        // page-locked source (`crate::pinned_hosts`).
        crate::pinned_hosts::register(ptr as *const u8, bytes);
        Ok(ptr as *mut u8)
    }

    pub(super) fn free_host_pinned_cu(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        if !ptr.is_null() {
            // 2026-09-25: Before the free, so a reused address is not reported
            // as page-locked.
            crate::pinned_hosts::unregister(ptr as *const u8);
            let status = unsafe { cuMemFreeHost(ptr as *mut c_void) };
            // 2026-09-25: A status meaning the context is already gone
            // (`registry::is_teardown_noop`) is not an error.
            if status != 0 && !crate::registry::is_teardown_noop(status) {
                bail!(
                    "cuMemFreeHost failed: {}",
                    crate::registry::cuda_error_text(status)
                );
            }
        }
        Ok(())
    }
}
