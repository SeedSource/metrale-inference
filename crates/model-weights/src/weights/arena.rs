// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: A device weight arena: a few large allocations, sub-allocated by a bump pointer,
//! for weight tensors that live as long as the model.
//!
//! **WHY.** Every `cuMemAlloc` costs about 14-18 KiB of memory outside the request on GB10
//! unified memory, mostly kernel unreclaimable slab for the driver's per-allocation objects,
//! and the allocation ledger cannot see it (spark-bench `runs/race/alloccost/RESULT.md`,
//! measured 2026-10-06: 74,304 separate allocations cost 1.0-1.3 GB more than the same bytes in
//! 258 large ones). GLM-5.3 makes 55-75 thousand weight allocations per rank, so that overhead
//! is 0.8-1.1 GB a rank. Placing them in an arena removes it; only addresses change.
//!
//! Owner: model-weights.
//! Invariants:
//! - Every chunk comes from `GpuBackend::alloc` (this file's call site), so the allocation
//!   ledger, and through it the KV-cache sizing, counts every arena byte.
//! - Every sub-allocation starts at a multiple of [`WEIGHT_ARENA_ALIGN`] from its chunk's base,
//!   and occupies `round_up(bytes.max(1), WEIGHT_ARENA_ALIGN)` bytes; no two overlap.
//! - A sub-allocation is never freed alone. [`WeightArena::release`] frees whole chunks, so an
//!   owner must never pass an arena pointer to `GpuBackend::free` ([`WeightArena::contains`]).
//! - Disabled (the default, and after a chunk allocation fails) `alloc` returns `Ok(None)` and
//!   allocates nothing; the caller then runs its own per-tensor path unchanged.
//!
//! Tail waste: [`WeightArena::plan`] takes the exact sizes of the sub-allocations about to be
//! made, in order, and packs them greedily into chunks of at most [`WEIGHT_ARENA_CHUNK_CAP`],
//! each sized to exactly what it will hold. When allocations follow the plan no chunk has an
//! unused tail; only the alignment padding is overhead. An allocation the plan did not
//! foresee gets an exactly-sized chunk of its own, and is counted in the summary.

use std::collections::VecDeque;

use parking_lot::Mutex;

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::WeightDtype;

/// 2026-10-06: Alignment of every sub-allocation. `cuMemAlloc` guarantees 256 bytes, the most a
/// separate allocation was ever promised; the strictest consumer check in the tree is 64 bytes
/// (`ops/w4a4_proj.rs`), and a TMA global address needs 16 (`cuda_backend/tensormap.rs`).
pub const WEIGHT_ARENA_ALIGN: usize = 256;

/// 2026-10-06: The largest chunk a plan makes. 2 GiB holds one GLM-5.3 routed-expert layer
/// whole (1,944 MiB at EP2 and at expert-TP), so a layer's plan is one chunk; a single
/// allocation this size is routine (the KV pool is far larger).
pub const WEIGHT_ARENA_CHUNK_CAP: usize = 2 << 30;

/// 2026-10-06: A model loader's answer to "which checkpoint tensors live until teardown and are
/// never freed alone?" `(tensor name, store dtype) -> in arena`. Same shape as `DeferHook`.
pub type ArenaHook = std::sync::Arc<dyn Fn(&str, WeightDtype) -> bool + Send + Sync>;

#[derive(Clone, Copy, Debug)]
struct Chunk {
    base: DevicePtr,
    len: usize,
    used: usize,
}

#[derive(Default)]
struct Inner {
    label: &'static str,
    enabled: bool,
    failed: bool,
    chunks: Vec<Chunk>,
    /// Exact sizes of the planned chunks not yet allocated, in order.
    planned: VecDeque<usize>,
    /// Unused bytes at the end of chunks closed before they were full.
    closed_tail: usize,
    requested: usize,
    used: usize,
    subs: usize,
    unplanned_chunks: usize,
}

/// 2026-10-06: Counters for the boot log and the microtest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaStats {
    pub chunks: usize,
    /// Sum of chunk sizes: what the ledger holds for this arena.
    pub chunk_bytes: usize,
    /// Bytes callers asked for.
    pub requested: usize,
    /// Bytes handed out, alignment padding included.
    pub used: usize,
    /// Chunk bytes never handed out (closed tails plus the open chunk's remainder).
    pub tail_waste: usize,
    pub sub_allocations: usize,
    /// Chunks no plan foresaw.
    pub unplanned_chunks: usize,
}

impl ArenaStats {
    /// 2026-10-06: Driver allocations the arena avoided: one per sub-allocation, less the
    /// chunks it made instead.
    pub fn allocations_saved(&self) -> usize {
        self.sub_allocations.saturating_sub(self.chunks)
    }
}

/// 2026-10-06: Round `bytes` (at least 1) up to [`WEIGHT_ARENA_ALIGN`].
pub fn arena_footprint(bytes: usize) -> usize {
    bytes.max(1).next_multiple_of(WEIGHT_ARENA_ALIGN)
}

/// 2026-10-06: Greedy packing of `items` (in order) into chunks of at most `cap` bytes; each
/// returned size is exactly the footprint of the items it holds. An item larger than `cap`
/// gets a chunk of its own.
pub fn pack_chunks(items: impl IntoIterator<Item = usize>, cap: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut cur = 0usize;
    for b in items {
        let a = arena_footprint(b);
        if cur > 0 && cur + a > cap {
            out.push(cur);
            cur = 0;
        }
        cur += a;
    }
    if cur > 0 {
        out.push(cur);
    }
    out
}

/// 2026-10-06: Close the open chunk: its unhanded remainder becomes tail waste.
fn close_open_chunk(g: &mut Inner) {
    if let Some(c) = g.chunks.last_mut() {
        let rest = c.len - c.used;
        c.used = c.len;
        g.closed_tail += rest;
    }
}

/// 2026-10-06: See the module doc.
#[derive(Default)]
pub struct WeightArena {
    // parking_lot: no poisoning, so a panic elsewhere cannot block teardown.
    inner: Mutex<Inner>,
}

impl WeightArena {
    /// 2026-10-06: Enable the arena under `label` and plan the next sub-allocations: `items` are
    /// their requested sizes, in the order they will be made. The open chunk is closed first
    /// (its unused remainder is counted as tail waste) and any planned chunk not yet allocated
    /// is dropped, so one plan never feeds another. Returns the planned chunk bytes. Does not
    /// re-enable an arena whose chunk allocation failed.
    pub fn plan(&self, label: &'static str, items: impl IntoIterator<Item = usize>) -> usize {
        let sizes = pack_chunks(items, WEIGHT_ARENA_CHUNK_CAP);
        let mut guard = self.inner.lock();
        let g = &mut *guard;
        g.label = label;
        g.enabled = true;
        close_open_chunk(g);
        g.planned = sizes.into();
        g.planned.iter().sum()
    }

    /// 2026-10-06: Whether `alloc` would sub-allocate (enabled and no chunk failure).
    pub fn is_active(&self) -> bool {
        let g = self.inner.lock();
        g.enabled && !g.failed
    }

    /// 2026-10-06: `bytes` of device memory from the arena, or `Ok(None)` when it is disabled
    /// or a chunk allocation failed (logged once); the caller then allocates on its own.
    pub fn alloc(&self, gpu: &dyn GpuBackend, bytes: usize) -> anyhow::Result<Option<DevicePtr>> {
        let mut guard = self.inner.lock();
        let g = &mut *guard;
        if !g.enabled || g.failed {
            return Ok(None);
        }
        let a = arena_footprint(bytes);
        let fits = g.chunks.last().is_some_and(|c| c.len - c.used >= a);
        if !fits {
            let size = match g.planned.pop_front() {
                Some(s) if s >= a => s,
                Some(_) => {
                    // The plan no longer matches what is being allocated: drop it rather
                    // than make chunks of the wrong sizes.
                    g.planned.clear();
                    g.unplanned_chunks += 1;
                    a
                }
                None => {
                    g.unplanned_chunks += 1;
                    a
                }
            };
            let base = match gpu.alloc(size) {
                Ok(p) => p,
                Err(e) => {
                    g.failed = true;
                    g.planned.clear();
                    tracing::warn!(
                        "weight arena ({}): a {:.1} MiB chunk failed ({e:#}); later tensors are \
                         allocated one by one, as without the arena",
                        g.label,
                        size as f64 / (1024.0 * 1024.0)
                    );
                    return Ok(None);
                }
            };
            if !base.0.is_multiple_of(WEIGHT_ARENA_ALIGN as u64) {
                // `cuMemAlloc` promises 256; refuse to break the alignment invariant.
                let _ = gpu.free(base);
                g.failed = true;
                anyhow::bail!(
                    "weight arena ({}): chunk base {base} is not {WEIGHT_ARENA_ALIGN}-byte aligned",
                    g.label
                );
            }
            close_open_chunk(g);
            g.chunks.push(Chunk {
                base,
                len: size,
                used: 0,
            });
        }
        let Some(c) = g.chunks.last_mut() else {
            anyhow::bail!("weight arena: no chunk after allocating one");
        };
        let p = c.base.offset(c.used);
        c.used += a;
        g.requested += bytes;
        g.used += a;
        g.subs += 1;
        Ok(Some(p))
    }

    /// 2026-10-06: [`Self::alloc`] for `src.len()` bytes, then copy `src` there.
    pub fn upload(&self, gpu: &dyn GpuBackend, src: &[u8]) -> anyhow::Result<Option<DevicePtr>> {
        let Some(p) = self.alloc(gpu, src.len())? else {
            return Ok(None);
        };
        gpu.copy_h2d(src, p)?;
        Ok(Some(p))
    }

    /// 2026-10-06: Whether `ptr` lies inside one of this arena's chunks.
    pub fn contains(&self, ptr: DevicePtr) -> bool {
        if ptr.is_null() {
            return false;
        }
        self.inner
            .lock()
            .chunks
            .iter()
            .any(|c| ptr.0 >= c.base.0 && ptr.0 < c.base.0 + c.len as u64)
    }

    pub fn stats(&self) -> ArenaStats {
        let g = self.inner.lock();
        let open: usize = g.chunks.last().map_or(0, |c| c.len - c.used);
        ArenaStats {
            chunks: g.chunks.len(),
            chunk_bytes: g.chunks.iter().map(|c| c.len).sum(),
            requested: g.requested,
            used: g.used,
            tail_waste: g.closed_tail + open,
            sub_allocations: g.subs,
            unplanned_chunks: g.unplanned_chunks,
        }
    }

    /// 2026-10-06: The boot-log line: chunks, bytes requested and used, tail waste, and
    /// allocations saved. Silent for an arena that never allocated.
    pub fn log_summary(&self) {
        let s = self.stats();
        if s.chunks == 0 {
            return;
        }
        let label = self.inner.lock().label;
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "weight arena ({label}): {} chunk(s), {:.1} MiB; {} tensors, {:.1} MiB requested, \
             {:.1} MiB used (alignment padding {:.2} MiB), tail waste {:.2} MiB, {} \
             allocation(s) saved, {} unplanned chunk(s)",
            s.chunks,
            mib(s.chunk_bytes),
            s.sub_allocations,
            mib(s.requested),
            mib(s.used),
            mib(s.used - s.requested),
            mib(s.tail_waste),
            s.allocations_saved(),
            s.unplanned_chunks,
        );
    }

    /// 2026-10-06: Free every chunk and reset to disabled. Drains first, so a failure part-way
    /// cannot leave a freed chunk listed; idempotent. Every pointer the arena handed out is
    /// dangling afterwards.
    pub fn release(&self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        let chunks: Vec<Chunk> = {
            let mut g = self.inner.lock();
            let c = std::mem::take(&mut g.chunks);
            *g = Inner::default();
            c
        };
        let mut first_error = None;
        for c in chunks {
            if let Err(e) = gpu.free(c.base)
                && first_error.is_none()
            {
                first_error = Some(e.context("freeing a weight-arena chunk"));
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
#[path = "arena_tests.rs"]
mod tests;
