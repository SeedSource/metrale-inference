// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The `Glm5NextDsaState` methods of the DSA pool cache
//! (`METRALE_GLM_DSA_POOL_CACHE=1`, `pool_cache.rs`): ring layout, the watermark records, the
//! device-clamp bookkeeping and the replay pre-check. Split out of `state.rs` (500-line cap).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: those of `state.rs` and `pool_cache.rs`; every method here is a no-op (or the
//! flat-cache answer) with the lever off.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::pool_cache::{DsaPoolCacheArgs, PoolCache, RingBook};
use super::Glm5NextDsaState;

impl Glm5NextDsaState {
    /// 2026-10-06: The runs `(first row of the call, rows)` that are contiguous in `k_normed` /
    /// `gate` for a write of `k` rows from row `pos0`: one run, `(0, k)`, unless the pool-cache
    /// ring wraps inside the write, then two.
    pub fn ring_runs(&self, pos0: usize, k: usize) -> Vec<(usize, usize)> {
        match &self.pool {
            Some(pc) => ring_runs_for(pc.book.ring_rows(), pos0, k),
            None => vec![(0, k)],
        }
    }

    /// 2026-10-06: Whether this state was allocated with the pool cache.
    pub fn is_pool_cache(&self) -> bool {
        self.pool.is_some()
    }

    /// 2026-10-06: The pool cache, if allocated.
    pub fn pool_cache(&self) -> Option<&PoolCache> {
        self.pool.as_ref()
    }

    /// 2026-10-06: The pool-cache arguments for a compress over `[0, len)` from the host
    /// watermark, after the release-mode read check (`RingBook::check_read`); `None` with the
    /// lever off.
    pub fn pool_select_args(&self) -> Result<Option<DsaPoolCacheArgs>> {
        match &self.pool {
            Some(pc) => {
                pc.book.check_read(self.len, pc.book.pk_len())?;
                Ok(Some(pc.select_args()))
            }
            None => Ok(None),
        }
    }

    /// 2026-10-06: After an aux blob v2 restore wrote every byte (`aux_state.rs`): set the
    /// cursor to `len` and the pool cache's book to `book`. No-op on the book with the lever off.
    pub(in crate::glm5next_dsa) fn set_restored(&mut self, len: usize, book: RingBook) {
        if let Some(pc) = &mut self.pool {
            pc.book = book;
        }
        self.len = len;
    }

    /// 2026-10-06: Whether rows `[0, len)` completed a pool past the watermark that no
    /// selection compressed yet (rows written without one: MTP drafter context rows). Always
    /// false with the lever off.
    pub fn pool_compress_due(&self) -> bool {
        self.pool
            .as_ref()
            .is_some_and(|pc| self.len / pc.book.kpool() > pc.book.pk_len())
    }

    /// 2026-10-06: Record a selection pass over `[0, len)`: its compress finalized pools
    /// `[0, len / kpool)` (host and device copies). No-op with the lever off.
    pub fn note_selected(&mut self) {
        let len = self.len;
        if let Some(pc) = &mut self.pool {
            pc.book.note_compressed(len);
        }
    }

    /// 2026-10-06: Record rows physically written through `end` ahead of the `advance` calls
    /// that publish them (the full-width prefill writes a whole window first). Call only after
    /// `ensure_room` covered `end`.
    pub fn note_ring_write(&mut self, end: usize) {
        if let Some(pc) = &mut self.pool {
            pc.book.note_write(end);
        }
    }

    /// 2026-10-06: For a host-path write from row `pos0`: `Some((pk_len_dev, pools))` when the
    /// device `pk_len` may be above `pos0 / kpool` (after a rewind) and the caller must clamp it
    /// on the stream (`dsa_pk_len_clamp`) before writing; recorded as done. `None` otherwise.
    pub fn take_dev_clamp(&mut self, pos0: usize) -> Option<(DevicePtr, usize)> {
        let pc = self.pool.as_mut()?;
        let pools = pc.book.dev_clamp_for(pos0)?;
        pc.book.note_dev_clamp(pools);
        Some((pc.pk_len_dev, pools))
    }

    /// 2026-10-06: Record the device-side clamp `dsa_indexer_store_ring` performs when it
    /// writes row `pos` (a captured write).
    pub fn note_device_store(&mut self, pos: usize) {
        if let Some(pc) = &mut self.pool {
            let kp = pc.book.kpool();
            pc.book.note_dev_clamp(pos / kp);
        }
    }

    /// 2026-10-06: The graph-replay pre-check (`check_replay_room`): the ring checks for the
    /// rewind and write the replay performs ([`Self::check_replay_ring`], a no-op with the
    /// lever off), then [`Self::ensure_room_through`]`(seq_len + k)` as before.
    pub fn replay_room(&self, seq_len: usize, k: usize) -> Result<()> {
        self.check_replay_ring(seq_len, k)?;
        self.ensure_room_through(seq_len + k)
    }

    /// 2026-10-06: The graph-replay pre-check with the pool cache: a replay from `seq_len`
    /// writing and selecting `k` rows (after the host rewind `sync_to` performs when the
    /// counter is ahead) passes the rewind and write checks. No-op with the lever off.
    pub fn check_replay_ring(&self, seq_len: usize, k: usize) -> Result<()> {
        let Some(pc) = &self.pool else {
            return Ok(());
        };
        let mut book = pc.book;
        if self.len > seq_len {
            book.check_rewind(self.len, seq_len)?;
            book.note_rewind(seq_len);
        }
        book.check_write(seq_len.min(self.len), seq_len + k)
    }
}

/// 2026-10-06: [`Glm5NextDsaState::ring_runs`] for a ring of `ring` rows.
pub fn ring_runs_for(ring: usize, pos0: usize, k: usize) -> Vec<(usize, usize)> {
    let ring = ring.max(1);
    let mut runs = Vec::with_capacity(2);
    let mut r0 = 0;
    while r0 < k {
        let slot = (pos0 + r0) % ring;
        let n = (ring - slot).min(k - r0);
        runs.push((r0, n));
        r0 += n;
    }
    if runs.is_empty() {
        runs.push((0, 0));
    }
    runs
}
