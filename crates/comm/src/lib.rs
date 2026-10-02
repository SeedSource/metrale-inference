// SPDX-License-Identifier: MIT OR Apache-2.0

#![deny(warnings)]
#![deny(clippy::all)]

//! 2026-09-26: Collective-communication backends behind one trait,
//! [`CommBackend`]: [`SingleGpuBackend`], where every operation is a no-op,
//! and, with the `nccl` feature, `NcclBackend`. No other crate calls NCCL.
//!
//! Owner: metrale-comm.
//! Invariants: none beyond the types.

use anyhow::Result;

/// 2026-10-01: Completion slots of [`CommBackend::all_reduce_deferred`]: at most this many
/// deferred all-reduces may be outstanding (issued, not yet joined) at once, each under its own
/// slot index `0..ALL_REDUCE_DEFERRED_SLOTS`.
pub const ALL_REDUCE_DEFERRED_SLOTS: usize = 2;

// 2026-09-26: Gated on `nccl` because the bindings link libnccl. The release
// targets built with `--features cuda` alone, such as the AMD SCALE one, have
// no NCCL library.
#[cfg(feature = "nccl")]
mod collective_diagnostics;
#[cfg(feature = "nccl")]
mod collective_wait;
#[cfg(feature = "nccl")]
pub mod nccl_backend;
#[cfg(feature = "nccl")]
pub use nccl_backend::NcclBackend;

/// 2026-09-26: Collective and point-to-point operations on raw device
/// pointers and byte counts. A pointer is a `u64` (a CUDA `CUdeviceptr`), so
/// this crate does not depend on metrale-gpu-runtime.
pub trait CommBackend: Send + Sync {
    /// 2026-09-26: Sum `bytes` at `ptr` across all ranks, in place; every rank
    /// gets the result. `NcclBackend` reduces BF16 elements.
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()>;

    /// 2026-09-26: Each rank contributes `bytes` from `send_ptr`; every rank
    /// receives all contributions, in rank order, at `recv_ptr`.
    fn all_gather(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()>;

    /// 2026-09-26: Sum across ranks and give each rank its `bytes`-sized share
    /// at `recv_ptr`. `NcclBackend` sums `u8` elements.
    fn reduce_scatter(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()>;

    /// 2026-09-26: Copy `bytes` at `ptr` from rank `root` to every rank, in
    /// place.
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()>;

    /// 2026-09-26: On a non-root rank, receive the first `u32` of the next
    /// worker command from `root`, the receive side of a 4-byte `broadcast`.
    /// Unlike `broadcast`, it may wait without a deadline, so an implementation
    /// must keep checking for transport errors while it waits. The command's
    /// later words use `broadcast`. The default refuses `root` and otherwise
    /// calls `broadcast(ptr, 4, root)`.
    fn recv_command_u32(&self, ptr: u64, root: usize) -> Result<()> {
        anyhow::ensure!(
            self.rank() != root,
            "idle command receive requires non-root rank"
        );
        self.broadcast(ptr, 4, root)
    }

    /// 2026-09-26: A barrier across ranks. `NcclBackend` enqueues a zero-count
    /// all-reduce on its stream and returns without waiting for it.
    fn barrier(&self) -> Result<()>;

    /// 2026-09-26: All-reduce ordered against `compute_stream` on the device:
    /// the reduce starts after the work already queued there and later work
    /// there waits for it. The host does not wait. The default ignores
    /// `compute_stream` and calls `all_reduce`.
    fn all_reduce_async(&self, ptr: u64, bytes: usize, compute_stream: u64) -> Result<()> {
        let _ = compute_stream;
        self.all_reduce(ptr, bytes)
    }

    /// 2026-10-01: All-reduce ordered after the work already queued on `compute_stream`, like
    /// `all_reduce_async`, but `compute_stream` does NOT wait for it: later work there runs
    /// concurrently until `all_reduce_join(compute_stream, slot)` adds that wait. `slot` must be
    /// below [`ALL_REDUCE_DEFERRED_SLOTS`] and must not belong to another deferred all-reduce
    /// that has not been joined yet. The caller must not touch `ptr..ptr + bytes` on any
    /// stream, nor issue `all_reduce` (the blocking form) on another stream, until the join.
    /// The collective sequence and the arithmetic are those of `all_reduce_async`; only the
    /// point where the compute stream waits moves. The default calls `all_reduce_async`
    /// (already joined), so `all_reduce_join` has nothing left to wait for.
    fn all_reduce_deferred(
        &self,
        ptr: u64,
        bytes: usize,
        compute_stream: u64,
        slot: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            slot < ALL_REDUCE_DEFERRED_SLOTS,
            "deferred all-reduce slot {slot} out of range (< {ALL_REDUCE_DEFERRED_SLOTS})"
        );
        self.all_reduce_async(ptr, bytes, compute_stream)
    }

    /// 2026-10-01: Whether `all_reduce_async` on this backend is the two-rank exchange plus
    /// `bf16_add_inplace` (each rank's buffer becomes `__hadd(own, peer)`), so a caller that
    /// moves the partials with `send_to` / `recv_from` and adds them with the same kernel gets
    /// the same bits. The default (no such path) is `false`.
    fn all_reduce_is_send_recv_add(&self) -> bool {
        false
    }

    /// 2026-10-01: Make `compute_stream` wait for the deferred all-reduce issued under `slot`.
    /// Joining a slot with nothing outstanding is a no-op. The default does nothing beyond the
    /// range check, matching the default `all_reduce_deferred`.
    fn all_reduce_join(&self, compute_stream: u64, slot: usize) -> Result<()> {
        let _ = compute_stream;
        anyhow::ensure!(
            slot < ALL_REDUCE_DEFERRED_SLOTS,
            "deferred all-reduce slot {slot} out of range (< {ALL_REDUCE_DEFERRED_SLOTS})"
        );
        Ok(())
    }

    /// 2026-09-26: Register a device buffer with the backend and return an
    /// opaque handle (`ncclCommRegister` in `NcclBackend`). The default
    /// registers nothing and returns 0.
    fn register_buffer(&self, _ptr: u64, _bytes: usize) -> Result<u64> {
        Ok(0)
    }

    /// 2026-09-26: Deregister a handle from `register_buffer`.
    fn deregister_buffer(&self, _handle: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-26: Allocate `bytes` of device memory from the backend's
    /// allocator (`ncclMemAlloc` in `NcclBackend`) and return the pointer. The
    /// default returns `Err`.
    fn symmetric_alloc(&self, _bytes: usize) -> Result<u64> {
        anyhow::bail!("symmetric_alloc not supported by this CommBackend");
    }

    /// 2026-09-26: Free a pointer from `symmetric_alloc`. The default returns
    /// `Err`.
    fn symmetric_free(&self, _ptr: u64) -> Result<()> {
        anyhow::bail!("symmetric_free not supported by this CommBackend");
    }

    /// 2026-09-26: Hand the backend the `bf16_add_inplace` kernel handle. The
    /// model engine looks it up and passes it after registering its buffers.
    /// `NcclBackend` takes its 2-rank send/recv all-reduce path only once it
    /// is set.
    fn set_add_kernel(&self, _handle: u64) {
        // 2026-09-26: The default keeps no kernel.
    }

    /// 2026-09-26: Send `bytes` at `ptr` to `dest_rank`, enqueued on `stream`.
    /// Pairs with a `recv_from` of the same size on `dest_rank`.
    fn send_to(&self, ptr: u64, bytes: usize, dest_rank: usize, stream: u64) -> Result<()>;

    /// 2026-09-26: Receive `bytes` into `ptr` from `src_rank`, enqueued on
    /// `stream`. Pairs with a `send_to` of the same size on `src_rank`.
    fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, stream: u64) -> Result<()>;

    /// 2026-09-26: Open a group: `NcclBackend` calls `ncclGroupStart`, so the
    /// operations up to `group_end` are submitted together. The default does
    /// nothing.
    fn group_start(&self) -> Result<()> {
        Ok(())
    }

    /// 2026-09-26: Close the group opened by `group_start`.
    fn group_end(&self) -> Result<()> {
        Ok(())
    }

    /// 2026-09-26: `false` after the backend has seen a transport error or a
    /// completion timeout, until a successful `attempt_reconnect`.
    /// `NcclBackend` also queries `ncclCommGetAsyncError`. The default returns
    /// `true`.
    fn is_healthy(&self) -> bool {
        true
    }

    /// 2026-09-26: Replace a failed communicator. `NcclBackend` aborts it and
    /// repeats the TCP bootstrap, so every rank must call this at the same
    /// time. The default does nothing.
    fn attempt_reconnect(&self) -> Result<()> {
        Ok(())
    }

    /// 2026-09-26: This rank's index, from 0.
    fn rank(&self) -> usize;

    /// 2026-09-26: Number of ranks.
    fn world_size(&self) -> usize;
}

/// 2026-09-26: One rank: every collective and point-to-point operation
/// returns `Ok` without doing anything, `rank` is 0 and `world_size` is 1.
pub struct SingleGpuBackend;

impl CommBackend for SingleGpuBackend {
    fn all_reduce(&self, _ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn all_gather(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn reduce_scatter(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn broadcast(&self, _ptr: u64, _bytes: usize, _root: usize) -> Result<()> {
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        Ok(())
    }

    fn send_to(&self, _ptr: u64, _bytes: usize, _dest_rank: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn recv_from(&self, _ptr: u64, _bytes: usize, _src_rank: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_gpu_noop() {
        let comm = SingleGpuBackend;
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.world_size(), 1);
        comm.all_reduce(0x1000, 1024).unwrap();
        comm.all_reduce_async(0x1000, 1024, 0x3000).unwrap();
        for slot in 0..ALL_REDUCE_DEFERRED_SLOTS {
            comm.all_reduce_deferred(0x1000, 1024, 0x3000, slot).unwrap();
            comm.all_reduce_join(0x3000, slot).unwrap();
        }
        comm.all_gather(0x1000, 0x2000, 512).unwrap();
        comm.reduce_scatter(0x1000, 0x2000, 512).unwrap();
        comm.broadcast(0x1000, 256, 0).unwrap();
        comm.barrier().unwrap();
        comm.send_to(0x1000, 256, 0, 0).unwrap();
        comm.recv_from(0x2000, 256, 0, 0).unwrap();
        comm.group_start().unwrap();
        comm.group_end().unwrap();
        let registration = comm.register_buffer(0x1000, 1024).unwrap();
        assert_eq!(registration, 0, "single-GPU registration is a no-op handle");
        comm.deregister_buffer(registration).unwrap();
        comm.set_add_kernel(0x4000);
        assert!(comm.is_healthy());
        comm.attempt_reconnect().unwrap();
    }

    /// 2026-10-01: The default deferred all-reduce and join refuse a slot outside
    /// `0..ALL_REDUCE_DEFERRED_SLOTS`.
    #[test]
    fn test_deferred_slot_range() {
        let comm = SingleGpuBackend;
        assert!(comm.all_reduce_deferred(0x1000, 2, 0, ALL_REDUCE_DEFERRED_SLOTS).is_err());
        assert!(comm.all_reduce_join(0, ALL_REDUCE_DEFERRED_SLOTS).is_err());
        assert!(!comm.all_reduce_is_send_recv_add());
    }

    #[test]
    fn test_single_gpu_symmetric_alloc_unsupported() {
        // 2026-09-26: `SingleGpuBackend` keeps the trait defaults, which
        // return `Err`.
        let comm = SingleGpuBackend;
        assert!(comm.symmetric_alloc(1024).is_err());
        assert!(comm.symmetric_free(0x1000).is_err());
    }
}
