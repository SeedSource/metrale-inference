// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl CommBackend for NcclBackend`, plus the broadcast wait and
//! the per-submission check. The `unsafe` contract is the one in the
//! `nccl_backend` module header.
//!
//! Owner: metrale-comm.
//! Invariants: none beyond the types.

use anyhow::{Context, Result, ensure};
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::{ALL_REDUCE_DTYPE_BYTES, COLLECTIVE_TIMEOUT_SECS, NcclBackend};
use crate::CommBackend;
use crate::collective_diagnostics::Dtype;
use metrale_gpu_sys::nccl::{self, NcclDataType, NcclRedOp};

impl CommBackend for NcclBackend {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()> {
        self.begin_submission("all_reduce", Dtype::Bf16, bytes, self.legacy_stream, None)?;
        if self.world_size == 2 && self.add_kernel.load(Ordering::Relaxed) != 0 {
            return self.all_reduce_2rank(ptr, bytes, self.legacy_stream);
        }
        // 2026-09-26: In place on `ptr`; `recv_buffer` is not used, so its
        // capacity does not bound this path.
        let count = bytes / ALL_REDUCE_DTYPE_BYTES;
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclAllReduce(
                ptr as *const _,
                ptr as *mut _,
                count,
                NcclDataType::Bfloat16,
                NcclRedOp::Sum,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllReduce")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn all_reduce_async(&self, ptr: u64, bytes: usize, compute_stream: u64) -> Result<()> {
        self.begin_submission(
            "all_reduce_async",
            Dtype::Bf16,
            bytes,
            self.comm_stream,
            None,
        )?;
        if self.world_size == 2 && self.add_kernel.load(Ordering::Relaxed) != 0 {
            nccl::record_event(self.compute_done_event, compute_stream)?;
            nccl::stream_wait_event(self.comm_stream, self.compute_done_event)?;

            self.all_reduce_2rank(ptr, bytes, self.comm_stream)?;

            nccl::record_event(self.comm_done_event, self.comm_stream)?;
            nccl::stream_wait_event(compute_stream, self.comm_done_event)?;
            return Ok(());
        }

        // 2026-09-26: In place on `ptr`; `recv_buffer` is not used.
        let count = bytes / ALL_REDUCE_DTYPE_BYTES;
        let comm = *self.comm.lock();

        nccl::record_event(self.compute_done_event, compute_stream)?;

        nccl::stream_wait_event(self.comm_stream, self.compute_done_event)?;

        let result = unsafe {
            nccl::ncclAllReduce(
                ptr as *const _,
                ptr as *mut _,
                count,
                NcclDataType::Bfloat16,
                NcclRedOp::Sum,
                comm,
                self.comm_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllReduce (async)")?;

        nccl::record_event(self.comm_done_event, self.comm_stream)?;

        nccl::stream_wait_event(compute_stream, self.comm_done_event)?;

        self.check_async_error(comm);

        Ok(())
    }

    /// 2026-10-01: The 2-rank path of `all_reduce_async` without its final
    /// `stream_wait_event(compute_stream, ..)`: the same `all_reduce_2rank` on
    /// `comm_stream`, after the work queued on `compute_stream`, then
    /// `deferred_done_events[slot]` recorded on `comm_stream` for
    /// `all_reduce_join`. Every deferred and async all-reduce runs on
    /// `comm_stream` in submission order, so the single `recv_buffer` is never
    /// shared by two in flight. Off the 2-rank path it is `all_reduce_async`,
    /// joined at once.
    fn all_reduce_deferred(
        &self,
        ptr: u64,
        bytes: usize,
        compute_stream: u64,
        slot: usize,
    ) -> Result<()> {
        ensure!(
            slot < crate::ALL_REDUCE_DEFERRED_SLOTS,
            "deferred all-reduce slot {slot} out of range (< {})",
            crate::ALL_REDUCE_DEFERRED_SLOTS
        );
        if !self.send_recv_path() {
            return self.all_reduce_async(ptr, bytes, compute_stream);
        }
        self.begin_submission(
            "all_reduce_deferred",
            Dtype::Bf16,
            bytes,
            self.comm_stream,
            None,
        )?;
        nccl::record_event(self.compute_done_event, compute_stream)?;
        nccl::stream_wait_event(self.comm_stream, self.compute_done_event)?;

        self.all_reduce_2rank(ptr, bytes, self.comm_stream)?;

        nccl::record_event(self.deferred_done_events[slot], self.comm_stream)
    }

    /// 2026-10-01: `compute_stream` waits for `deferred_done_events[slot]`. Off
    /// the 2-rank path `all_reduce_deferred` already joined, so nothing to do.
    fn all_reduce_join(&self, compute_stream: u64, slot: usize) -> Result<()> {
        ensure!(
            slot < crate::ALL_REDUCE_DEFERRED_SLOTS,
            "deferred all-reduce slot {slot} out of range (< {})",
            crate::ALL_REDUCE_DEFERRED_SLOTS
        );
        if !self.send_recv_path() {
            return Ok(());
        }
        nccl::stream_wait_event(compute_stream, self.deferred_done_events[slot])
    }

    fn register_buffer(&self, ptr: u64, bytes: usize) -> Result<u64> {
        let mut handle: *mut c_void = ptr::null_mut();
        let comm = *self.comm.lock();
        let result =
            unsafe { nccl::ncclCommRegister(comm, ptr as *mut c_void, bytes, &mut handle) };
        nccl::check_nccl(result, "ncclCommRegister")?;
        self.registered_handles.lock().push(handle);
        Ok(handle as u64)
    }

    fn deregister_buffer(&self, handle: u64) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe { nccl::ncclCommDeregister(comm, handle as *mut c_void) };
        nccl::check_nccl(result, "ncclCommDeregister")
    }

    fn symmetric_alloc(&self, bytes: usize) -> Result<u64> {
        // 2026-09-26: SAFETY: `nccl_mem_alloc` only requires that the pointer
        // is later freed with `nccl_mem_free`, which `symmetric_free` does.
        let ptr = unsafe { nccl::nccl_mem_alloc(bytes) }?;
        Ok(ptr as u64)
    }

    fn symmetric_free(&self, ptr: u64) -> Result<()> {
        // 2026-09-26: SAFETY: The trait contract: `ptr` came from
        // `symmetric_alloc` and has not been freed.
        unsafe { nccl::nccl_mem_free(ptr as *mut c_void) }
    }

    fn set_add_kernel(&self, handle: u64) {
        self.add_kernel.store(handle, Ordering::Relaxed);
        tracing::info!(
            "NCCL backend: bf16_add_inplace kernel set \
             (2-rank send/recv enabled)"
        );
    }

    fn all_gather(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()> {
        self.begin_submission("all_gather", Dtype::U8, bytes, self.legacy_stream, None)?;
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclAllGather(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllGather")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn reduce_scatter(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()> {
        self.begin_submission("reduce_scatter", Dtype::U8, bytes, self.legacy_stream, None)?;
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclReduceScatter(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                NcclRedOp::Sum,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclReduceScatter")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
        self.broadcast_with_wait(ptr, bytes, root, false)
    }

    fn recv_command_u32(&self, ptr: u64, root: usize) -> Result<()> {
        ensure!(
            self.rank != root,
            "idle command receive requires non-root rank"
        );
        self.broadcast_with_wait(ptr, 4, root, true)
    }

    fn barrier(&self) -> Result<()> {
        self.begin_submission("barrier", Dtype::F32, 0, self.legacy_stream, None)?;
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclAllReduce(
                ptr::null(),
                ptr::null_mut(),
                0,
                NcclDataType::Float32,
                NcclRedOp::Sum,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "barrier (ncclAllReduce count=0)")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn send_to(&self, ptr: u64, bytes: usize, dest_rank: usize, stream: u64) -> Result<()> {
        self.begin_submission("send", Dtype::U8, bytes, stream, Some(dest_rank))?;
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclSend(
                ptr as *const c_void,
                bytes,
                NcclDataType::Uint8,
                dest_rank as i32,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclSend (send_to)")
    }

    fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, stream: u64) -> Result<()> {
        self.begin_submission("recv", Dtype::U8, bytes, stream, Some(src_rank))?;
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclRecv(
                ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                src_rank as i32,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclRecv (recv_from)")
    }

    fn group_start(&self) -> Result<()> {
        let result = unsafe { nccl::ncclGroupStart() };
        nccl::check_nccl(result, "ncclGroupStart")
    }

    fn group_end(&self) -> Result<()> {
        let result = unsafe { nccl::ncclGroupEnd() };
        nccl::check_nccl(result, "ncclGroupEnd")
    }

    fn is_healthy(&self) -> bool {
        if self.unhealthy.load(Ordering::Acquire) {
            return false;
        }
        let comm = *self.comm.lock();
        self.check_async_error(comm)
    }

    fn attempt_reconnect(&self) -> Result<()> {
        self.reconnect_inner()
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

impl NcclBackend {
    /// 2026-10-01: Whether all-reduce takes the 2-rank send/recv path
    /// (`world_size == 2` and the add kernel set), the test `all_reduce` and
    /// `all_reduce_async` make inline.
    fn send_recv_path(&self) -> bool {
        self.world_size == 2 && self.add_kernel.load(Ordering::Relaxed) != 0
    }

    fn broadcast_with_wait(
        &self,
        ptr: u64,
        bytes: usize,
        root: usize,
        idle_command: bool,
    ) -> Result<()> {
        self.begin_submission(
            "broadcast",
            Dtype::U8,
            bytes,
            self.legacy_stream,
            Some(root),
        )?;
        let start = Instant::now();
        let comm = *self.comm.lock();

        let result = unsafe {
            nccl::ncclBroadcast(
                ptr as *const _,
                ptr as *mut _,
                bytes,
                NcclDataType::Uint8,
                root as i32,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclBroadcast")?;

        // 2026-09-26: Poll `cuStreamQuery` between pauses, so the deadline
        // applies while waiting, not after an unbounded synchronise.
        let ready = || {
            ensure!(self.check_async_error(comm), "NCCL asynchronous failure");
            nccl::stream_ready(self.legacy_stream)
        };
        // 2026-09-29 (A141): a fixed 1ms sleep on every non-ready poll cost +10.6 ms/step
        // on the decode hot path (10 broadcasts/step, each essentially never
        // ready on the first query but usually done within microseconds).
        // adaptive_pause() spins briefly, then yields, then falls back to an
        // increasing-but-capped sleep — see collective_wait::AdaptiveBackoff.
        let pause = crate::collective_wait::adaptive_pause();
        let completion = if idle_command {
            crate::collective_wait::poll_idle_command(ready, pause)
        } else {
            crate::collective_wait::poll_completion(
                Duration::from_secs(COLLECTIVE_TIMEOUT_SECS),
                || start.elapsed(),
                ready,
                pause,
            )
        };
        crate::collective_wait::poison_on_error(completion, &self.unhealthy).with_context(
            || {
                format!(
                    "NCCL broadcast rank={} world_size={} root={root} bytes={bytes}; \
                     communicator poisoned; stop all ranks before retrying",
                    self.rank, self.world_size,
                )
            },
        )?;

        Ok(())
    }

    fn begin_submission(
        &self,
        op: &str,
        dtype: Dtype,
        bytes: usize,
        stream: u64,
        peer: Option<usize>,
    ) -> Result<()> {
        crate::collective_wait::ensure_healthy(&self.unhealthy, self.rank, self.world_size, op)?;
        self.diagnostics.submit(op, dtype, bytes, stream, peer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_collective_timeout_constant() {
        assert!(COLLECTIVE_TIMEOUT_SECS >= 10);
        assert!(COLLECTIVE_TIMEOUT_SECS <= 300);
    }
}
