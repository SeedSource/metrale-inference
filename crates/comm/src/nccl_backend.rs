// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `NcclBackend`, the multi-rank [`CommBackend`](crate::CommBackend)
//! over NCCL. Bootstrap: rank 0 creates an NCCL unique id and sends it over
//! TCP to every other rank, then all ranks call `ncclCommInitRank`.
//!
//! At `world_size == 2`, once the add kernel is set, all-reduce is a grouped
//! `ncclSend`/`ncclRecv` into a registered receive buffer plus a local BF16
//! add; otherwise it is `ncclAllReduce`. After each collective the backend
//! queries `ncclCommGetAsyncError`; a broadcast waits for completion for at
//! most `COLLECTIVE_TIMEOUT_SECS`, an idle command receive without a deadline,
//! and a failure in either marks the communicator unhealthy. `attempt_reconnect`
//! aborts the communicator and bootstraps again. `METRALE_COMM_DIAGNOSTICS=1`
//! logs every host submission.
//!
//! Owner: metrale-comm.
//! Invariants:
//! - Every collective and point-to-point operation first refuses an unhealthy
//!   communicator and validates its byte count and peer (`begin_submission`).
//! - A 2-rank send/recv all-reduce never receives more than `recv_capacity`
//!   bytes: `all_reduce_2rank` checks before any NCCL call.
//!
//! Each `unsafe` block wraps one NCCL or CUDA driver call. The declarations
//! follow nccl.h and cuda.h. Callers must pass device pointers valid for
//! `bytes` on the communicator's device and streams that outlive the enqueued
//! work.

use anyhow::{Context, Result};
use parking_lot::Mutex;
use std::ffi::c_void;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use metrale_gpu_sys::nccl::{self, NcclComm, NcclDataType, NcclResult, NcclUniqueId};

// 2026-09-26: CUDA driver calls for the receive buffer and the add kernel.
unsafe extern "C" {
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuLaunchKernel(
        f: u64,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: u64,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
}

mod recv_buffer;
use recv_buffer::ensure_payload_fits;
pub use recv_buffer::{ALL_REDUCE_DTYPE_BYTES, required_model_recv_bytes, required_recv_bytes};

/// 2026-09-26: Deadline, in seconds, for a broadcast to complete. It bounds
/// only the completion polling, not an NCCL or driver call that hangs.
pub(super) const COLLECTIVE_TIMEOUT_SECS: u64 = 30;

/// 2026-09-26: NCCL communicator plus the streams, events and receive buffer
/// its operations use.
pub struct NcclBackend {
    /// 2026-09-26: The communicator handle. Operations copy it out under the
    /// lock and release the lock before calling NCCL; `reconnect_inner`
    /// holds the lock while it replaces the handle.
    comm: Mutex<NcclComm>,
    diagnostics: crate::collective_diagnostics::Diagnostics,
    rank: usize,
    world_size: usize,
    /// 2026-09-26: Stream this backend creates for `all_reduce_async`.
    comm_stream: u64,
    /// 2026-09-26: Recorded on the caller's compute stream before an async
    /// all-reduce.
    compute_done_event: u64,
    /// 2026-09-26: Recorded on `comm_stream` after an async all-reduce.
    comm_done_event: u64,
    /// 2026-10-01: One per deferred-all-reduce slot: recorded on `comm_stream` after the
    /// deferred all-reduce issued under that slot, waited on by `all_reduce_join`.
    deferred_done_events: [u64; crate::ALL_REDUCE_DEFERRED_SLOTS],
    /// 2026-09-26: The caller's stream from `new`, used by every operation
    /// that takes no stream argument.
    legacy_stream: u64,
    /// 2026-09-26: Receive buffer of the 2-rank send/recv all-reduce; 0 when
    /// `world_size != 2`.
    recv_buffer: u64,
    /// 2026-09-26: Size of `recv_buffer` in bytes; 0 when `world_size != 2`.
    recv_capacity: usize,
    /// 2026-09-26: Handles from `register_buffer`, deregistered in `Drop` and
    /// dropped without deregistering on reconnect.
    registered_handles: Mutex<Vec<*mut c_void>>,
    /// 2026-09-26: `bf16_add_inplace` handle from `set_add_kernel`; 0 until
    /// set.
    add_kernel: AtomicU64,
    /// 2026-09-26: Set by an async error or a failed completion, cleared by a
    /// successful reconnect.
    unhealthy: AtomicBool,
    /// 2026-09-26: Successful reconnects, for the log.
    reconnect_count: AtomicU64,
    /// 2026-09-26: Bootstrap address, reused by reconnect.
    master_addr: String,
    master_port: u16,
}

// 2026-09-26: SAFETY: The raw pointers are an NCCL communicator handle and
// registration handles, which the backend only copies and passes to NCCL
// (`register_buffer` also returns a copy as `u64`). The mutex guards only
// reading and replacing the communicator handle; NCCL calls run after the
// guard is released and are not serialised by it.
unsafe impl Send for NcclBackend {}
unsafe impl Sync for NcclBackend {}

impl NcclBackend {
    /// 2026-09-26: Bootstrap the communicator. Rank 0 binds `0.0.0.0:master_port`
    /// and sends the unique id to `world_size - 1` connections; other ranks
    /// connect to `master_addr:master_port`, retrying once a second for up to
    /// 600 attempts. `recv_capacity` is the largest 2-rank all-reduce payload
    /// in bytes (serve passes [`required_model_recv_bytes`]); it is used only
    /// when `world_size == 2`.
    ///
    /// # Errors
    /// An invalid `METRALE_COMM_DIAGNOSTICS` value or rank/world pair, a failed
    /// bootstrap or NCCL init, a failed stream or event creation, a zero
    /// `recv_capacity` at `world_size == 2`, or a failed receive-buffer
    /// allocation. A failed registration of the receive buffer is only logged.
    pub fn new(
        rank: usize,
        world_size: usize,
        master_addr: &str,
        master_port: u16,
        stream: u64,
        recv_capacity: usize,
    ) -> Result<Self> {
        let diagnostic_env = std::env::var(crate::collective_diagnostics::ENV).ok();
        let diagnostics = crate::collective_diagnostics::Diagnostics::new(
            diagnostic_env.as_deref(),
            rank,
            world_size,
        )?;
        Self::log_nccl_env_vars();

        let unique_id = if rank == 0 {
            let id = Self::generate_unique_id()?;
            Self::distribute_id(&id, master_addr, master_port, world_size)?;
            id
        } else {
            Self::receive_id(master_addr, master_port)?
        };

        let mut comm: NcclComm = ptr::null_mut();
        let result =
            unsafe { nccl::ncclCommInitRank(&mut comm, world_size as i32, unique_id, rank as i32) };
        nccl::check_nccl(result, "ncclCommInitRank")?;

        let comm_stream = nccl::create_stream()?;
        let compute_done_event = nccl::create_event()?;
        let comm_done_event = nccl::create_event()?;
        let mut deferred_done_events = [0u64; crate::ALL_REDUCE_DEFERRED_SLOTS];
        for ev in &mut deferred_done_events {
            *ev = nccl::create_event()?;
        }

        let mut recv_buffer: u64 = 0;
        if world_size == 2 {
            if recv_capacity == 0 {
                anyhow::bail!(
                    "world_size == 2 requires a non-zero receive-buffer capacity; \
                     compute it with required_recv_bytes(max_batch_tokens, hidden_size, \
                     ALL_REDUCE_DTYPE_BYTES)"
                );
            }
            let status = unsafe { cuMemAlloc_v2(&mut recv_buffer, recv_capacity) };
            if status != 0 {
                anyhow::bail!(
                    "cuMemAlloc_v2 for recv_buffer ({recv_capacity} bytes) failed: status {status}"
                );
            }
            let mut handle: *mut c_void = ptr::null_mut();
            let result = unsafe {
                nccl::ncclCommRegister(comm, recv_buffer as *mut c_void, recv_capacity, &mut handle)
            };
            if result != NcclResult::Success {
                tracing::warn!("ncclCommRegister for recv_buffer failed (non-fatal): {result:?}");
            } else {
                tracing::info!(
                    "Registered recv_buffer ({} KB) with NCCL",
                    recv_capacity / 1024
                );
            }
        }

        Ok(Self {
            comm: Mutex::new(comm),
            diagnostics,
            rank,
            world_size,
            comm_stream,
            compute_done_event,
            comm_done_event,
            deferred_done_events,
            legacy_stream: stream,
            recv_buffer,
            recv_capacity: if world_size == 2 { recv_capacity } else { 0 },
            registered_handles: Mutex::new(Vec::new()),
            add_kernel: AtomicU64::new(0),
            unhealthy: AtomicBool::new(false),
            reconnect_count: AtomicU64::new(0),
            master_addr: master_addr.to_owned(),
            master_port,
        })
    }

    /// 2026-09-26: Log the listed NCCL environment variables at `info` when
    /// set, `debug` when not.
    fn log_nccl_env_vars() {
        let vars = [
            "NCCL_TIMEOUT",
            "NCCL_WATCHDOG_TIMEOUT",
            "NCCL_IB_TIMEOUT",
            "NCCL_IB_RETRY_CNT",
            "NCCL_SOCKET_IFNAME",
            "NCCL_DEBUG",
        ];
        for var in &vars {
            match std::env::var(var) {
                Ok(val) => tracing::info!("NCCL env: {var}={val}"),
                Err(_) => tracing::debug!("NCCL env: {var} not set"),
            }
        }
    }

    /// 2026-09-26: Query `ncclCommGetAsyncError`. Returns `true` when there is
    /// no error; otherwise, or when the query itself fails, sets the unhealthy
    /// flag and returns `false`.
    fn check_async_error(&self, comm: NcclComm) -> bool {
        let mut async_err = NcclResult::Success;
        let result = unsafe { nccl::ncclCommGetAsyncError(comm, &mut async_err) };
        if result != NcclResult::Success {
            tracing::error!(
                "ncclCommGetAsyncError call itself failed: {result:?} \
                 — marking unhealthy"
            );
            self.unhealthy.store(true, Ordering::Release);
            return false;
        }
        if async_err != NcclResult::Success {
            tracing::error!("NCCL async error detected: {async_err:?} — marking unhealthy");
            self.unhealthy.store(true, Ordering::Release);
            return false;
        }
        true
    }

    /// 2026-09-26: Abort the communicator and bootstrap a new one on
    /// `master_port + 1`, the same exchange as `new`, so every rank must call
    /// it. Returns at once when the backend is not marked unhealthy. The
    /// receive buffer is registered again; other registrations are dropped.
    fn reconnect_inner(&self) -> Result<()> {
        let mut comm_guard = self.comm.lock();

        // 2026-09-26: Another thread may have reconnected while this one
        // waited for the lock.
        if !self.unhealthy.load(Ordering::Acquire) {
            tracing::info!("NCCL communicator already recovered by another thread");
            return Ok(());
        }

        let old_comm = *comm_guard;
        let attempt = self.reconnect_count.load(Ordering::Relaxed) + 1;
        tracing::warn!(
            "NCCL reconnect: aborting old communicator \
             (rank={}, reconnect #{})",
            self.rank,
            attempt,
        );

        if !old_comm.is_null() {
            let result = unsafe { nccl::ncclCommAbort(old_comm) };
            if result != NcclResult::Success {
                tracing::warn!("ncclCommAbort returned {result:?} (proceeding anyway)");
            }
        }

        // 2026-09-26: `master_port + 1`, so the bind does not collide with the
        // first bootstrap's listener if it has not closed.
        let reconnect_port = self.master_port.wrapping_add(1);
        let unique_id = if self.rank == 0 {
            let id = Self::generate_unique_id()?;
            Self::distribute_id(&id, &self.master_addr, reconnect_port, self.world_size)?;
            id
        } else {
            Self::receive_id(&self.master_addr, reconnect_port)?
        };

        let mut new_comm: NcclComm = ptr::null_mut();
        let result = unsafe {
            nccl::ncclCommInitRank(
                &mut new_comm,
                self.world_size as i32,
                unique_id,
                self.rank as i32,
            )
        };
        nccl::check_nccl(result, "ncclCommInitRank (reconnect)")?;

        if self.world_size == 2 && self.recv_buffer != 0 {
            let mut handle: *mut c_void = ptr::null_mut();
            let result = unsafe {
                nccl::ncclCommRegister(
                    new_comm,
                    self.recv_buffer as *mut c_void,
                    self.recv_capacity,
                    &mut handle,
                )
            };
            if result != NcclResult::Success {
                tracing::warn!(
                    "ncclCommRegister for recv_buffer after reconnect \
                     failed: {result:?}"
                );
            } else {
                tracing::info!("Re-registered recv_buffer after reconnect");
            }
        }

        // 2026-09-26: The old handles belong to the aborted communicator, and
        // only handles were stored, not (ptr, size), so they cannot be
        // registered again.
        let mut handles = self.registered_handles.lock();
        if !handles.is_empty() {
            tracing::warn!(
                "Clearing {} stale registered buffer handles after reconnect",
                handles.len()
            );
            handles.clear();
        }
        drop(handles);

        *comm_guard = new_comm;
        self.unhealthy.store(false, Ordering::Release);
        self.reconnect_count.fetch_add(1, Ordering::Relaxed);

        tracing::info!(
            "NCCL reconnect successful (rank={}, total reconnects={})",
            self.rank,
            self.reconnect_count.load(Ordering::Relaxed),
        );

        Ok(())
    }

    /// 2026-09-26: 2-rank all-reduce: refuse a payload larger than
    /// `recv_capacity`, then a grouped `ncclSend`/`ncclRecv` with the partner
    /// rank, then `ptr[i] += recv_buffer[i]` with the BF16 add kernel on
    /// `stream`. Errors when the add kernel is not set.
    fn all_reduce_2rank(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        ensure_payload_fits(bytes, self.recv_capacity, self.rank, self.world_size)?;

        // 2026-09-26: Nothing to reduce, and a zero-block launch is invalid.
        // Both ranks skip the send/recv, provided both pass the same `bytes`.
        if bytes == 0 {
            return Ok(());
        }

        let count = bytes / ALL_REDUCE_DTYPE_BYTES;
        let partner = (1 - self.rank) as i32;
        let comm = *self.comm.lock();

        let result = unsafe { nccl::ncclGroupStart() };
        nccl::check_nccl(result, "ncclGroupStart")?;

        let result = unsafe {
            nccl::ncclSend(
                ptr as *const c_void,
                count,
                NcclDataType::Bfloat16,
                partner,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclSend")?;

        let result = unsafe {
            nccl::ncclRecv(
                self.recv_buffer as *mut c_void,
                count,
                NcclDataType::Bfloat16,
                partner,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclRecv")?;

        let result = unsafe { nccl::ncclGroupEnd() };
        nccl::check_nccl(result, "ncclGroupEnd")?;

        self.check_async_error(comm);

        let kernel = self.add_kernel.load(Ordering::Relaxed);
        if kernel != 0 {
            let threads: u32 = 256;
            let blocks: u32 = (count as u32).div_ceil(threads);
            let mut p_dst = ptr;
            let mut p_src = self.recv_buffer;
            let mut p_n = count as i32;
            let mut params: [*mut c_void; 3] = [
                &mut p_dst as *mut u64 as *mut c_void,
                &mut p_src as *mut u64 as *mut c_void,
                &mut p_n as *mut i32 as *mut c_void,
            ];
            let status = unsafe {
                cuLaunchKernel(
                    kernel,
                    blocks,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            if status != 0 {
                anyhow::bail!("cuLaunchKernel (bf16_add_inplace) failed: status {status}");
            }
        } else {
            anyhow::bail!("bf16_add_inplace kernel not set — call set_add_kernel() first");
        }

        Ok(())
    }

    fn generate_unique_id() -> Result<NcclUniqueId> {
        let mut id = NcclUniqueId {
            internal: [0u8; 128],
        };
        let result = unsafe { nccl::ncclGetUniqueId(&mut id) };
        nccl::check_nccl(result, "ncclGetUniqueId")?;
        Ok(id)
    }

    /// 2026-09-26: Rank 0: bind `0.0.0.0:port`, accept `world_size - 1`
    /// connections and send each the unique id. `addr` is unused.
    fn distribute_id(id: &NcclUniqueId, addr: &str, port: u16, world_size: usize) -> Result<()> {
        let bind_addr = format!("0.0.0.0:{port}");
        let listener = TcpListener::bind(&bind_addr)
            .with_context(|| format!("Rank 0: failed to bind {bind_addr}"))?;
        tracing::info!(
            "Rank 0: waiting for {} worker(s) on {}",
            world_size - 1,
            bind_addr
        );

        for i in 0..(world_size - 1) {
            let (mut stream, peer_addr) = listener.accept().context("Rank 0: accept failed")?;
            stream
                .write_all(&id.internal)
                .context("Rank 0: failed to send unique ID")?;
            tracing::info!("Rank 0: sent unique ID to worker {} ({})", i + 1, peer_addr);
        }
        let _ = addr;
        Ok(())
    }

    /// 2026-09-26: Other ranks: connect to rank 0, retrying, and read the
    /// unique id.
    fn receive_id(addr: &str, port: u16) -> Result<NcclUniqueId> {
        let target = format!("{addr}:{port}");
        tracing::info!("Rank N: connecting to master at {target}");

        // 2026-09-26: Serve bootstraps after loading weights, so rank 0 may
        // open the port minutes after a worker starts connecting; the worker
        // retries once a second for about ten minutes.
        const MAX_ATTEMPTS: u32 = 600;
        let mut stream = None;
        for attempt in 0..MAX_ATTEMPTS {
            match TcpStream::connect(&target) {
                Ok(s) => {
                    stream = Some(s);
                    break;
                }
                Err(e) => {
                    if attempt + 1 < MAX_ATTEMPTS {
                        if attempt < 30 || attempt.is_multiple_of(30) {
                            tracing::info!(
                                "Connect attempt {}/{MAX_ATTEMPTS}: {e}, retrying in 1s (rank 0 may be loading weights)",
                                attempt + 1
                            );
                        }
                        std::thread::sleep(std::time::Duration::from_secs(1));
                    } else {
                        return Err(e).with_context(|| {
                            format!(
                                "Failed to connect to master at {target} \
                                 after {MAX_ATTEMPTS} attempts (~{} minutes)",
                                MAX_ATTEMPTS / 60
                            )
                        });
                    }
                }
            }
        }

        let mut id = NcclUniqueId {
            internal: [0u8; 128],
        };
        stream
            .unwrap()
            .read_exact(&mut id.internal)
            .context("Failed to receive unique ID from rank 0")?;
        tracing::info!("Received NCCL unique ID from master");
        Ok(id)
    }
}

impl Drop for NcclBackend {
    fn drop(&mut self) {
        let comm = *self.comm.lock();
        let mut handles = self.registered_handles.lock();
        for handle in handles.drain(..) {
            unsafe { nccl::ncclCommDeregister(comm, handle) };
        }
        drop(handles);
        if self.recv_buffer != 0 {
            unsafe { cuMemFree_v2(self.recv_buffer) };
        }
        nccl::destroy_event(self.compute_done_event);
        nccl::destroy_event(self.comm_done_event);
        for ev in self.deferred_done_events {
            nccl::destroy_event(ev);
        }
        nccl::destroy_stream(self.comm_stream);
        if !comm.is_null() {
            unsafe { nccl::ncclCommDestroy(comm) };
        }
    }
}

mod comm_impl;
