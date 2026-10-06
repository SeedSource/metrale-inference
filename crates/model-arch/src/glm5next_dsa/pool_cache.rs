// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: `METRALE_GLM_DSA_POOL_CACHE=1` (race #79, default off): a persistent per-state
//! DSA pool-key cache, so a selection pass compresses only the pools completed since the last
//! pass instead of every pool of `[0, len)`, and the raw indexer keys and gates shrink to a
//! ring of [`POOL_CACHE_RING_ROWS`] rows. Design: `runs/race/mem/POOLCACHE-DESIGN.md`
//! (spark-bench), stages 1-3 (state, kernel and launch, writers).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Lever off: nothing here is allocated or consulted; `Glm5NextDsaState` keeps full-length
//!   `k_normed`/`gate`, `select_tokens` launches `dsa_kpool_compress` into the scratch, and the
//!   reserve is today's.
//! - Lever on, per state: pools `[0, pk_len)` of `pk`/`pidx`/`pvalid` are final (each the bytes
//!   `dsa_kpool_compress` writes for that pool over `[0, len)`: the incremental kernel runs the
//!   same per-pool body); `ring_lo <= kpool * pk_len <= len`; and every row in
//!   `[ring_lo, len)` is intact at ring slot `row % ring_rows` ([`RingBook`]).
//! - The three release-mode checks (they return `Err`; they are not debug assertions): a write
//!   ending at `end` needs `end - kpool * pk_len <= ring_rows` ([`RingBook::check_write`]); a
//!   rewind must leave every row the next compress needs in the ring
//!   ([`RingBook::check_rewind`]); a compress reads only rows in `[kpool * pk_start, len)` with
//!   `kpool * pk_start >= ring_lo` ([`RingBook::check_read`]).
//! - The device copy of `pk_len` (`pk_len_dev`, one i32 per state) is never above the host's
//!   `pk_len` when a replay or an exact compress reads it: every compress sets it to
//!   `S / kpool`, `dsa_indexer_store_ring` clamps it to the row it writes, and a host-path
//!   write clamps it on the stream first when the host's upper bound (`dev_hi`) says it may be
//!   above the written row's pool (`RingBook::dev_clamp_for`).
//!
//! `pvalid` is allocated eagerly, like `valid` (`lazy.rs`): at a quarter byte per token one
//! 2 MiB granule would hold 8 M tokens, so mapping it lazily would round it up to a granule.
//!
//! TODO(pool-cache stage 4): prefix-cache aux blob v2 (`aux_state.rs` refuses with the lever
//! on). TODO(pool-cache stage 5): GPU parity microtest (cached pools and selections against the
//! full recompute) before the lever is advertised.

use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::lazy_buffer::{LazyBuffer, MapBudget};

/// 2026-10-06: Rows in the raw `k_normed`/`gate` ring with the lever on: the largest single
/// write before a selection (the full-width prefill window, 8,192 rows written before its first
/// 256-row sub-chunk selects) plus 256 rows of rewind slack. 4.3 MB per layer per sequence at
/// `index_head_dim` 128.
pub const POOL_CACHE_RING_ROWS: usize = 8_448;

/// 2026-10-06: `METRALE_GLM_DSA_POOL_CACHE=1`: persistent pool keys and a raw-row ring per DSA
/// state. Read once.
pub fn dsa_pool_cache() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = matches!(
            std::env::var("METRALE_GLM_DSA_POOL_CACHE").as_deref(),
            Ok("1")
        );
        if on {
            tracing::warn!(
                "METRALE_GLM_DSA_POOL_CACHE=1 - DSA states keep persistent pool keys and a \
                 {POOL_CACHE_RING_ROWS}-row raw key ring; selections compress only new pools"
            );
        }
        on
    })
}

/// 2026-10-06: Ring rows for a state of `capacity` rows: [`POOL_CACHE_RING_ROWS`], or the
/// capacity when that is smaller (the ring is then the flat cache: `row % ring == row`).
pub fn ring_rows_for(capacity: usize) -> usize {
    POOL_CACHE_RING_ROWS.min(capacity).max(1)
}

/// 2026-10-06: Bytes one state takes with the lever on at `capacity` rows: what
/// [`PoolCache::alloc`] allocates (`pk` `[pools, d]` f32, `pidx` `[pools, kpool]` i32,
/// `pvalid` `[pools]` u8, `pk_len_dev` 4 B, the `k_normed` and `gate` rings `[ring, d]` BF16
/// each) plus the state's `valid` `[capacity]` u8. `pools = capacity / kpool` (the capacity
/// is a whole number of pools, `state::dsa_capacity`). Per token that is
/// `(4 d + 4 kpool + 1) / kpool + 1`: 133.25 B at d 128, kpool 4 (was 513).
pub fn pool_cache_state_bytes(capacity: usize, index_head_dim: usize, index_kpool: usize) -> usize {
    let kp = index_kpool.max(1);
    let pools = capacity / kp;
    let ring = ring_rows_for(capacity);
    // 2026-10-06: pk, pidx, pvalid, then valid, the two rings, pk_len_dev.
    pools * index_head_dim * 4
        + pools * kp * 4
        + pools
        + capacity
        + 2 * ring * index_head_dim * 2
        + 4
}

/// 2026-10-06: The per-state device side of a selection with the lever on, passed through
/// `DsaSelectInputs::pool_cache`: `select_tokens` runs `dsa_kpool_compress_incr` into these
/// arrays and points the scores, top-k and expand kernels at them instead of the scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsaPoolCacheArgs {
    /// 2026-10-06: `[capacity / kpool, index_head_dim]` f32 pool keys.
    pub pk: DevicePtr,
    /// 2026-10-06: `[capacity / kpool, kpool]` i32 pool token ids.
    pub pidx: DevicePtr,
    /// 2026-10-06: `[capacity / kpool]` u8 pool validity.
    pub pvalid: DevicePtr,
    /// 2026-10-06: One i32, the device copy of `pk_len`; the kernel sets it to `S / kpool`.
    pub pk_len_dev: DevicePtr,
    /// 2026-10-06: First pool an exact launch computes (the host `pk_len`). A ceiling launch
    /// reads its start from the geom slot `dsa_write_geom_pk` fills instead.
    pub pk_start: usize,
    /// 2026-10-06: Rows in the `k_normed`/`gate` ring; row `r` is at slot `r % ring_rows`.
    pub ring_rows: usize,
}

/// 2026-10-06: Host bookkeeping of one state's ring and pool watermark. Pure: no device
/// memory, so the three release-mode checks are unit-tested on both sides of each bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingBook {
    ring_rows: usize,
    kpool: usize,
    /// 2026-10-06: Pools `[0, pk_len)` are final.
    pk_len: usize,
    /// 2026-10-06: Every row in `[ring_lo, len)` is intact at its ring slot.
    ring_lo: usize,
    /// 2026-10-06: An upper bound of the device `pk_len_dev`.
    dev_hi: usize,
}

impl RingBook {
    pub fn new(ring_rows: usize, kpool: usize) -> Self {
        Self {
            ring_rows: ring_rows.max(1),
            kpool: kpool.max(1),
            pk_len: 0,
            ring_lo: 0,
            dev_hi: 0,
        }
    }

    pub fn ring_rows(&self) -> usize {
        self.ring_rows
    }
    pub fn pk_len(&self) -> usize {
        self.pk_len
    }
    pub fn ring_lo(&self) -> usize {
        self.ring_lo
    }
    pub fn kpool(&self) -> usize {
        self.kpool
    }

    /// 2026-10-06: Release-mode check before writing rows `[len, end)`: the rows the next
    /// compress reads, `[kpool * pk_len, end)`, must all stay in the ring, so
    /// `end - kpool * pk_len <= ring_rows` (and the invariant `ring_lo <= kpool * pk_len`).
    pub fn check_write(&self, len: usize, end: usize) -> Result<()> {
        let first = self.kpool * self.pk_len;
        let lo = self.ring_lo.max(end.saturating_sub(self.ring_rows));
        if end < len || first > len || lo > first {
            bail!(
                "DSA pool cache: writing rows [{len}, {end}) would overwrite ring rows the next \
                 compress still needs (pools final through row {first}, ring of {} rows, ring \
                 floor {}): a write ending at {end} needs end - {} * pk_len <= {}. Refused \
                 rather than selecting over overwritten keys.",
                self.ring_rows,
                self.ring_lo,
                self.kpool,
                self.ring_rows
            );
        }
        Ok(())
    }

    /// 2026-10-06: Record rows written through `end` (after `check_write` passed): rows below
    /// `end - ring_rows` may have been overwritten.
    pub fn note_write(&mut self, end: usize) {
        self.ring_lo = self.ring_lo.max(end.saturating_sub(self.ring_rows));
    }

    /// 2026-10-06: Release-mode check before a rewind from `len` to `n`: the next compress
    /// starts at pool `min(pk_len, n / kpool)` and needs rows `[kpool * that, n)`, which must
    /// all be in the ring (`>= ring_lo`). In steady decode `ring_lo` is about
    /// `len - ring_rows`, so this is the "len - n within the ring" bound.
    pub fn check_rewind(&self, len: usize, n: usize) -> Result<()> {
        let pk = self.pk_len.min(n / self.kpool);
        let first = self.kpool * pk;
        if n > len || (first < n && first < self.ring_lo) {
            bail!(
                "DSA pool cache: rewind from {len} to {n} needs rows [{first}, {n}) to recompute \
                 pool {pk}, but the ring of {} rows holds only rows from {}. Refused rather than \
                 selecting over overwritten keys.",
                self.ring_rows,
                self.ring_lo
            );
        }
        Ok(())
    }

    /// 2026-10-06: Record a rewind to `n` (after `check_rewind` passed). The device copy is not
    /// touched here: the next write clamps it (`dev_clamp_for`, `dsa_indexer_store_ring`).
    pub fn note_rewind(&mut self, n: usize) {
        self.pk_len = self.pk_len.min(n / self.kpool);
        // 2026-10-06: `[ring_lo, n)` is empty when `ring_lo > n`; claim nothing below `n`.
        self.ring_lo = self.ring_lo.min(n);
    }

    /// 2026-10-06: Release-mode check before a compress over `[kpool * pk_start, len)`: no row
    /// outside `[ring_lo, len)` (and so outside `[kpool * pk_len, len)`) is read from the ring.
    pub fn check_read(&self, len: usize, pk_start: usize) -> Result<()> {
        let first = self.kpool * pk_start;
        if pk_start > self.pk_len
            || first > len
            || first < self.ring_lo
            || len - first > self.ring_rows
        {
            bail!(
                "DSA pool cache: a compress from pool {pk_start} over {len} rows would read ring \
                 rows outside [{}, {len}) (pk_len {}, ring of {} rows). Refused.",
                self.ring_lo.max(self.kpool * self.pk_len),
                self.pk_len,
                self.ring_rows
            );
        }
        Ok(())
    }

    /// 2026-10-06: Record a compress over `len` rows: pools `[0, len / kpool)` are final, and
    /// the kernel set the device copy to the same value.
    pub fn note_compressed(&mut self, len: usize) {
        self.pk_len = len / self.kpool;
        self.dev_hi = self.pk_len;
    }

    /// 2026-10-06: The value a host-path write starting at row `pos0` must clamp the device
    /// `pk_len_dev` to before it writes, or `None` when the device copy is already at or below
    /// it (no launch: the common append).
    pub fn dev_clamp_for(&self, pos0: usize) -> Option<usize> {
        let pool = pos0 / self.kpool;
        (self.dev_hi > pool).then_some(pool)
    }

    /// 2026-10-06: Record a clamp of the device copy to `pools` (host clamp kernel, or the
    /// `dsa_indexer_store_ring` clamp a captured write performs).
    pub fn note_dev_clamp(&mut self, pools: usize) {
        self.dev_hi = self.dev_hi.min(pools);
    }
}

/// 2026-10-06: One state's persistent pool arrays, the raw-row rings and the device `pk_len`.
pub struct PoolCache {
    pub pk: DevicePtr,
    pub pidx: DevicePtr,
    pub pvalid: DevicePtr,
    pub pk_len_dev: DevicePtr,
    /// 2026-10-06: `[ring_rows, d]` BF16 rings (become the state's `k_normed`/`gate`).
    pub k_ring: DevicePtr,
    pub g_ring: DevicePtr,
    pub book: RingBook,
    /// 2026-10-06: `[pk, pidx]` lazily mapped from the shared indexer pool; `None` = eager.
    lazy: Option<[LazyBuffer; 2]>,
    index_head_dim: usize,
    capacity: usize,
}

impl PoolCache {
    /// 2026-10-06: Allocate for `capacity` rows. `pool` set: `pk` and `pidx` lazily mapped from
    /// it (`METRALE_DSA_INDEXER_LAZY`); `None`: eager. The rings, `pvalid` and `pk_len_dev` are
    /// eager; `pk_len_dev` is zeroed. On error everything allocated so far is released.
    pub fn alloc(
        gpu: &dyn GpuBackend,
        capacity: usize,
        index_head_dim: usize,
        index_kpool: usize,
        pool: Option<Arc<MapBudget>>,
    ) -> Result<Self> {
        let kp = index_kpool.max(1);
        let pools = capacity / kp;
        let ring = ring_rows_for(capacity);
        let d = index_head_dim;
        let mut eager: Vec<DevicePtr> = Vec::with_capacity(6);
        let mut lazy: Vec<LazyBuffer> = Vec::with_capacity(2);
        let r = (|| -> Result<()> {
            match &pool {
                Some(p) => {
                    lazy.push(gpu.alloc_lazy(pools * d * 4, Some(p.clone()))?);
                    lazy.push(gpu.alloc_lazy(pools * kp * 4, Some(p.clone()))?);
                }
                None => {
                    eager.push(gpu.alloc(pools * d * 4)?);
                    eager.push(gpu.alloc(pools * kp * 4)?);
                }
            }
            eager.push(gpu.alloc(pools)?);
            eager.push(gpu.alloc(4)?);
            eager.push(gpu.alloc(ring * d * 2)?);
            eager.push(gpu.alloc(ring * d * 2)?);
            Ok(())
        })();
        if let Err(e) = r {
            for b in &lazy {
                let _ = b.release(gpu);
            }
            for p in &eager {
                let _ = gpu.free(*p);
            }
            return Err(e);
        }
        let (pk, pidx, rest) = match lazy.len() {
            2 => (lazy[0].ptr(), lazy[1].ptr(), &eager[..]),
            _ => (eager[0], eager[1], &eager[2..]),
        };
        let (pvalid, pk_len_dev, k_ring, g_ring) = (rest[0], rest[1], rest[2], rest[3]);
        let pc = Self {
            pk,
            pidx,
            pvalid,
            pk_len_dev,
            k_ring,
            g_ring,
            book: RingBook::new(ring, kp),
            lazy: <[LazyBuffer; 2]>::try_from(lazy).ok(),
            index_head_dim: d,
            capacity,
        };
        if let Err(e) = gpu.copy_h2d(&0i32.to_le_bytes(), pk_len_dev) {
            let _ = pc.free(gpu);
            return Err(e);
        }
        Ok(pc)
    }

    /// 2026-10-06: Whether `pk`/`pidx` are lazily mapped.
    pub fn is_lazy(&self) -> bool {
        self.lazy.is_some()
    }

    /// 2026-10-06: Token rows backed in `pk` and `pidx` (the smaller of the two, in whole
    /// pools); `None` when eager.
    pub fn mapped_rows(&self) -> Option<usize> {
        let kp = self.book.kpool;
        let (pk_row, pidx_row) = (self.index_head_dim * 4, kp * 4);
        self.lazy.as_ref().map(|[a, b]| {
            let pools = (a.mapped_bytes() / pk_row.max(1)).min(b.mapped_bytes() / pidx_row);
            (pools * kp).min(self.capacity)
        })
    }

    /// 2026-10-06: Back the pools that cover rows `[0, min(rows, capacity))`. A no-op when
    /// eager or already backed.
    pub fn map_rows(&self, rows: usize) -> Result<()> {
        if let Some([a, b]) = &self.lazy {
            let kp = self.book.kpool;
            let pools = rows.min(self.capacity).div_ceil(kp);
            a.ensure_mapped(pools * self.index_head_dim * 4)?;
            b.ensure_mapped(pools * kp * 4)?;
        }
        Ok(())
    }

    /// 2026-10-06: The arguments `select_tokens` takes for an exact compress from `pk_len`.
    pub fn select_args(&self) -> DsaPoolCacheArgs {
        DsaPoolCacheArgs {
            pk: self.pk,
            pidx: self.pidx,
            pvalid: self.pvalid,
            pk_len_dev: self.pk_len_dev,
            pk_start: self.book.pk_len,
            ring_rows: self.book.ring_rows,
        }
    }

    /// 2026-10-06: Release everything; the first error is returned after every release was
    /// attempted.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut first: Result<()> = Ok(());
        let mut keep = |r: Result<()>| {
            if first.is_ok() {
                first = r;
            }
        };
        match &self.lazy {
            Some([a, b]) => {
                keep(a.release(gpu));
                keep(b.release(gpu));
            }
            None => {
                keep(gpu.free(self.pk));
                keep(gpu.free(self.pidx));
            }
        }
        for p in [self.pvalid, self.pk_len_dev, self.k_ring, self.g_ring] {
            keep(gpu.free(p));
        }
        first
    }
}

#[cfg(test)]
#[path = "pool_cache_tests.rs"]
mod tests;
