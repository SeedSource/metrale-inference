// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Device buffers whose physical memory is mapped on demand, and the byte budget
//! a set of them draws from.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - A [`LazyBuffer`]'s base pointer never changes between creation and `release`, so a CUDA
//!   graph that captured it stays valid while more of it is mapped (measured on GB10
//!   2026-10-03, spark-bench `runs/race/vmm-probe/20261003-114732-n1.log`).
//! - `[0, mapped_bytes())` is backed; the mapped size only grows, in whole granules, until
//!   `release`.
//! - A budget is charged for every mapped byte and refunded on `release`, so
//!   `MapBudget::used` is the sum of the mapped sizes of the live buffers drawing from it.
//! - No mapping happens while a stream capture is active ([`capture_active`]): mapping is a
//!   host-side driver call that a capture would not record, so it must precede the capture.
//!
//! Two kinds: `Eager` allocates the whole buffer up front through `GpuBackend::alloc` (the
//! default for backends without virtual memory management, the mock included) and only does
//! the budget accounting on `ensure_mapped`; `Vmm` (CUDA) reserves virtual address space for
//! the whole buffer and maps 2 MiB granules (`cuMemCreate` + `cuMemMap` + `cuMemSetAccess`)
//! as `ensure_mapped` asks.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, bail};
use parking_lot::Mutex;

use crate::gpu::{DevicePtr, GpuBackend};

/// 2026-10-03: The VMM allocation granularity measured on GB10
/// (`cuMemGetAllocationGranularity`, minimum and recommended both 2 MiB, 2026-10-03 probe).
/// Callers that size a budget before a backend exists use it.
pub const DEFAULT_GRANULE: usize = 2 * 1024 * 1024;

/// 2026-10-03: `bytes` rounded up to a whole number of `granule`s (`granule` 0 counts as 1).
pub fn round_up_to_granule(bytes: usize, granule: usize) -> usize {
    let g = granule.max(1);
    bytes.div_ceil(g) * g
}

/// 2026-10-03: The mapped size that backs `[0, want)` of a buffer of `reserved` bytes: `want`
/// capped at `reserved`, rounded up to whole granules. `reserved` is itself a granule multiple
/// for every buffer this module builds, so the result never exceeds it.
pub fn mapped_target(want: usize, reserved: usize, granule: usize) -> usize {
    round_up_to_granule(want.min(reserved), granule).min(reserved)
}

/// 2026-10-03: Streams with an active capture in this process: `begin_capture` adds the
/// stream, `end_capture` and `abort_capture_if_active` remove it (CUDA backend). A set rather
/// than a count, so an abort on a stream that was not capturing changes nothing.
static CAPTURING_STREAMS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

pub fn note_capture_begin(stream: u64) {
    CAPTURING_STREAMS.lock().push(stream);
}

pub fn note_capture_end(stream: u64) {
    let mut v = CAPTURING_STREAMS.lock();
    if let Some(i) = v.iter().position(|&s| s == stream) {
        v.swap_remove(i);
    }
}

/// 2026-10-03: Whether any stream capture is active in this process.
pub fn capture_active() -> bool {
    !CAPTURING_STREAMS.lock().is_empty()
}

/// 2026-10-03: Bytes currently mapped by VMM lazy buffers in this process. The CUDA backend's
/// `live_bytes` adds it to the allocation ledger.
static VMM_MAPPED_BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn vmm_mapped_bytes() -> usize {
    VMM_MAPPED_BYTES.load(Ordering::Relaxed)
}

/// 2026-10-03: A byte budget shared by lazy buffers. Charged when a buffer maps, refunded when
/// it releases. The charge depends only on how far each buffer is mapped, so two processes
/// that map the same buffers to the same extents make the same charge and refusal decisions.
pub struct MapBudget {
    name: &'static str,
    limit: usize,
    used: AtomicUsize,
}

impl MapBudget {
    pub fn new(name: &'static str, limit: usize) -> Self {
        Self {
            name,
            limit,
            used: AtomicUsize::new(0),
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
    pub fn limit(&self) -> usize {
        self.limit
    }
    pub fn used(&self) -> usize {
        self.used.load(Ordering::SeqCst)
    }
    pub fn available(&self) -> usize {
        self.limit.saturating_sub(self.used())
    }

    /// 2026-10-03: Charge `bytes`, or refuse without charging when that would pass the limit.
    ///
    /// The refusal text starts with "KV cache exhausted" on purpose: the scheduler's decode
    /// path matches that phrase to preempt a sequence and requeue it (`decode_launch.rs`),
    /// which is the right response to a full budget too.
    pub fn try_charge(&self, bytes: usize) -> Result<()> {
        let r = self
            .used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |u| {
                u.checked_add(bytes).filter(|&n| n <= self.limit)
            });
        if let Err(used) = r {
            bail!(
                "KV cache exhausted: the {} pool cannot map {:.1} MiB more ({:.1} of {:.1} MiB \
                 in use). Fewer or shorter concurrent sequences fit; raise the pool size to \
                 serve more.",
                self.name,
                bytes as f64 / 1048576.0,
                used as f64 / 1048576.0,
                self.limit as f64 / 1048576.0,
            );
        }
        Ok(())
    }

    pub fn refund(&self, bytes: usize) {
        let _ = self
            .used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |u| {
                Some(u.saturating_sub(bytes))
            });
    }
}

/// 2026-10-03: How a [`LazyBuffer`] is backed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyKind {
    /// 2026-10-03: The whole buffer was allocated up front with `GpuBackend::alloc`.
    Eager,
    /// 2026-10-03: Reserved virtual address space; granules mapped on demand on `device`.
    Vmm { device: i32 },
}

struct LazyMapState {
    mapped: usize,
    /// 2026-10-03: One `CUmemGenericAllocationHandle` per mapped granule, in address order
    /// (`Vmm` only).
    handles: Vec<u64>,
    released: bool,
}

/// 2026-10-03: A device buffer of `reserved` bytes whose `[0, mapped)` prefix is backed.
pub struct LazyBuffer {
    base: DevicePtr,
    reserved: usize,
    granule: usize,
    kind: LazyKind,
    budget: Option<Arc<MapBudget>>,
    free_floor: usize,
    state: Mutex<LazyMapState>,
}

impl LazyBuffer {
    /// 2026-10-03: Wrap an eager allocation of `reserved` bytes at `base` (already rounded to
    /// `granule` by the caller). `ensure_mapped` then only charges `budget`.
    pub fn eager(
        base: DevicePtr,
        reserved: usize,
        granule: usize,
        budget: Option<Arc<MapBudget>>,
    ) -> Self {
        Self::from_parts(base, reserved, granule, LazyKind::Eager, budget, 0)
    }

    /// 2026-10-03: Build from parts; the CUDA backend uses it for a `Vmm` reservation.
    /// `free_floor` is the device free memory (bytes) a map must leave.
    pub fn from_parts(
        base: DevicePtr,
        reserved: usize,
        granule: usize,
        kind: LazyKind,
        budget: Option<Arc<MapBudget>>,
        free_floor: usize,
    ) -> Self {
        Self {
            base,
            reserved,
            granule: granule.max(1),
            kind,
            budget,
            free_floor,
            state: Mutex::new(LazyMapState {
                mapped: 0,
                handles: Vec::new(),
                released: false,
            }),
        }
    }

    pub fn ptr(&self) -> DevicePtr {
        self.base
    }
    pub fn reserved_bytes(&self) -> usize {
        self.reserved
    }
    pub fn granule_bytes(&self) -> usize {
        self.granule
    }
    pub fn kind(&self) -> LazyKind {
        self.kind
    }
    /// 2026-10-03: The device free memory a map must leave (`Vmm`; 0 for `Eager`).
    pub fn free_floor_bytes(&self) -> usize {
        self.free_floor
    }
    pub fn mapped_bytes(&self) -> usize {
        self.state.lock().mapped
    }

    /// 2026-10-03: Make `[0, want)` backed (capped at the reservation). A no-op when it already
    /// is. Otherwise: refuse during a stream capture, charge the budget for the new granules
    /// (refusing without side effects when it is full), then map them. A failed map refunds
    /// what it did not map and leaves the already-mapped prefix in place.
    pub fn ensure_mapped(&self, want: usize) -> Result<()> {
        let mut st = self.state.lock();
        if st.released {
            bail!("lazy buffer {}: ensure_mapped after release", self.base);
        }
        let target = mapped_target(want, self.reserved, self.granule);
        if target <= st.mapped {
            return Ok(());
        }
        if capture_active() {
            bail!(
                "lazy buffer {}: mapping {} more bytes requested inside a stream capture; \
                 the caller must map before capturing (or before replaying)",
                self.base,
                target - st.mapped
            );
        }
        let need = target - st.mapped;
        if let Some(b) = &self.budget {
            b.try_charge(need)?;
        }
        let done = self.map_physical(&mut st, target);
        let mapped_now = st.mapped;
        if let Err(e) = done {
            if let Some(b) = &self.budget {
                b.refund(target - mapped_now);
            }
            return Err(e);
        }
        Ok(())
    }

    fn map_physical(&self, st: &mut LazyMapState, target: usize) -> Result<()> {
        match self.kind {
            LazyKind::Eager => {
                st.mapped = target;
                Ok(())
            }
            #[cfg(all(feature = "cuda", not(metrale_scale)))]
            LazyKind::Vmm { device } => {
                if let Some(free) = crate::cuda_backend::cuda_free_memory_bytes() {
                    let need = target - st.mapped;
                    if free < need.saturating_add(self.free_floor) {
                        bail!(
                            "KV cache exhausted: device free memory {:.0} MiB would fall below \
                             the {:.0} MiB lazy-map floor (METRALE_LAZY_MAP_FREE_FLOOR_MB) \
                             if {:.1} MiB more were mapped",
                            free as f64 / 1048576.0,
                            self.free_floor as f64 / 1048576.0,
                            need as f64 / 1048576.0,
                        );
                    }
                }
                while st.mapped < target {
                    let h = crate::cuda_backend::vmm::map_granule(
                        self.base.0 + st.mapped as u64,
                        self.granule,
                        device,
                    )?;
                    st.handles.push(h);
                    st.mapped += self.granule;
                    VMM_MAPPED_BYTES.fetch_add(self.granule, Ordering::Relaxed);
                }
                Ok(())
            }
            #[cfg(not(all(feature = "cuda", not(metrale_scale))))]
            LazyKind::Vmm { .. } => bail!("VMM lazy buffers need the cuda feature"),
        }
    }

    /// 2026-10-03: Release the buffer: unmap and free every granule and the address range
    /// (`Vmm`), or free the eager allocation; refund the budget. A second call does nothing.
    pub fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut st = self.state.lock();
        if st.released {
            return Ok(());
        }
        st.released = true;
        if let Some(b) = &self.budget {
            b.refund(st.mapped);
        }
        let mapped = std::mem::take(&mut st.mapped);
        let handles = std::mem::take(&mut st.handles);
        match self.kind {
            LazyKind::Eager => {
                let _ = handles;
                let _ = mapped;
                gpu.free(self.base)
            }
            #[cfg(all(feature = "cuda", not(metrale_scale)))]
            LazyKind::Vmm { .. } => {
                let _ = gpu;
                VMM_MAPPED_BYTES.fetch_sub(handles.len() * self.granule, Ordering::Relaxed);
                crate::cuda_backend::vmm::release_range(
                    self.base.0,
                    self.reserved,
                    self.granule,
                    &handles,
                )
            }
            #[cfg(not(all(feature = "cuda", not(metrale_scale))))]
            LazyKind::Vmm { .. } => {
                let _ = (gpu, handles, mapped);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
#[path = "lazy_buffer_tests.rs"]
mod tests;
