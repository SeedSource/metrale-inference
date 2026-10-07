// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Snapshot and restore of the DSA indexer cache, the part of a GLM-5.3
//! sequence that a KV-only prefix-cache hit cannot rebuild.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `restore_blob` checks the header, `index_head_dim`, total size and
//!   capacity before any copy, and returns an error for any mismatch.
//! - The cursor moves only after every copy is enqueued, so a failed restore
//!   leaves `len` unchanged.
//! - A blob is `HEADER_BYTES + len * (4 * index_head_dim + 1)` bytes: it
//!   carries `len` rows, not `capacity`.
//! - 2026-10-06: A state allocated with `METRALE_GLM_DSA_POOL_CACHE=1` uses
//!   blob v2 (stage 4 of the pool-cache design):
//!   `[tag][len][d][pk][pool keys][pool ids][pool validity][tail K][tail G][V]`,
//!   all u64 little-endian in the 32-byte header, with `pk` the final pools it
//!   carries (`min(pk_len, len / kpool)`) and the tail rows `[kpool * pk, len)`
//!   read from the ring. A v1 blob is refused by a v2 state and a v2 blob (its
//!   first word is [`V2_TAG`], above any v1 `len`) by a v1 state, before any
//!   copy. Restore is bit-exact and sets the host and device `pk_len`.
//! - `blob_bytes_for_rows(rows)` is the exact size `snapshot_blob_prefix(rows)`
//!   returns for the same state, in both formats.
//!
//! # Why a blob and not a rewind
//!
//! `rewind_to` restores within a live sequence: the rows stay and only the
//! cursor moves. A prefix-cache hit hands the prefix to a sequence whose
//! indexer buffer was never written, and the rows cannot be rebuilt from the
//! MLA latent: `k_normed` and `gate` are projections of the hidden state
//! (`indexer.wk`, `index_kpool_compress_gate`). So the rows travel with the
//! snapshot.
//!
//! `layers/qsa_snapshot.rs` in model-layers does the same for the QSA indexer,
//! and re-pools its block keys on restore. The DSA state stores no pooled keys
//! (pooling runs inside the selector kernel each step), so the blob is the
//! whole reachable cache.
//!
//! Applying a mismatched blob would make the selector read another sequence's
//! keys, which is why every mismatch is an error rather than a repair.
//!
//! # Cost
//!
//! `len * (4 * index_head_dim + 1)` bytes per DSA layer, 513 B per token and
//! layer at `index_head_dim = 128`.
//!
//! 2026-10-06: v2 (pool cache): `pk * (4 * index_head_dim + 4 * kpool + 1)
//! + tail * 4 * index_head_dim + len` bytes, about 134 B per token and layer
//! at `index_head_dim = 128`, `kpool = 4`.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;

use super::pool_cache::RingBook;
use super::state::{Glm5NextDsaState, ring_runs_for};

/// 2026-09-25: `[len u64][index_head_dim u64]`, little-endian.
const HEADER_BYTES: usize = 16;

/// 2026-09-25: Bytes a blob occupies for `len` rows at `index_head_dim`: the
/// per-row size of [`super::state::indexer_state_bytes`], over `len` rows, plus
/// the header.
fn blob_bytes(len: usize, index_head_dim: usize) -> usize {
    HEADER_BYTES + len * (index_head_dim * 4 + 1)
}

/// 2026-10-06: First word of a v2 blob (`METRALE_GLM_DSA_POOL_CACHE=1`). Above any v1 `len`
/// (a context of 2^63 rows), so a v1 reader never mistakes it for a length.
pub const V2_TAG: u64 = 0xD5A0_AC5E_0000_0002;

/// 2026-10-06: `[tag][len][d][pk]`, little-endian u64.
const V2_HEADER_BYTES: usize = 32;

/// 2026-10-06: The layout of a v2 blob of `len` rows carrying `pk` final pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct V2Layout {
    len: usize,
    pk: usize,
    tail: usize,
    d: usize,
    kp: usize,
}

impl V2Layout {
    fn new(len: usize, pk: usize, d: usize, kp: usize) -> Self {
        Self {
            len,
            pk,
            tail: len - kp * pk,
            d,
            kp,
        }
    }
    /// 2026-10-06: Byte offsets of pool keys, pool ids, pool validity, tail K, tail G, V.
    fn offsets(&self) -> [usize; 6] {
        let (pk, t, d, kp) = (self.pk, self.tail, self.d, self.kp);
        let o0 = V2_HEADER_BYTES;
        let o1 = o0 + pk * d * 4;
        let o2 = o1 + pk * kp * 4;
        let o3 = o2 + pk;
        let o4 = o3 + t * d * 2;
        let o5 = o4 + t * d * 2;
        [o0, o1, o2, o3, o4, o5]
    }
    fn bytes(&self) -> usize {
        self.offsets()[5] + self.len
    }
}

/// 2026-10-06: Final pools a v2 blob of `rows` rows carries: those already compressed, and
/// none past `rows`.
fn v2_pools(book: &RingBook, rows: usize) -> usize {
    book.pk_len().min(rows / book.kpool())
}

impl Glm5NextDsaState {
    /// 2026-10-03: Bytes `snapshot_blob_prefix(rows)` returns, without allocating.
    /// 2026-10-06: With the pool cache, the v2 size: it depends on the state's `pk_len` as
    /// well as `rows` (identical on every rank, which run the same passes).
    pub fn blob_bytes_for_rows(&self, rows: usize) -> usize {
        match self.pool_cache() {
            Some(pc) => {
                let pk = v2_pools(&pc.book, rows);
                V2Layout::new(rows, pk, self.index_head_dim(), pc.book.kpool()).bytes()
            }
            None => blob_bytes(rows, self.index_head_dim()),
        }
    }

    /// 2026-09-25: Serialize the reachable indexer rows `[0, len)` for a
    /// prefix-cache snapshot. Rows past `len` were never written or are
    /// unreachable (`rewind_to` leaves them in place).
    pub fn snapshot_blob(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<Vec<u8>> {
        self.snapshot_blob_prefix(self.len(), gpu, stream)
    }

    /// 2026-10-03: Serialize rows `[0, rows)` only, as a blob of `rows` rows: the blob
    /// `snapshot_blob` returns for a state whose cursor is at `rows`. For an SSM snapshot
    /// captured inside a prefill pass that continued past `rows` (model-engine
    /// `prefill_b/inpass_capture.rs`): rows are written once each, by the pass over their
    /// position, so the first `rows` rows are unchanged by the rest of the pass. Refuses
    /// `rows > len`.
    pub fn snapshot_blob_prefix(
        &self,
        rows: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<u8>> {
        ensure!(
            rows <= self.len(),
            "DSA aux prefix of {rows} rows past the {} rows this state holds",
            self.len()
        );
        // 2026-10-06: `METRALE_GLM_DSA_POOL_CACHE=1`: blob v2 (`k_normed`/`gate` are a ring).
        if self.is_pool_cache() {
            return self.snapshot_v2(rows, gpu, stream);
        }
        let len = rows;
        let d = self.index_head_dim();
        let key_bytes = len * d * 2;

        let mut blob = vec![0u8; blob_bytes(len, d)];
        blob[..8].copy_from_slice(&(len as u64).to_le_bytes());
        blob[8..16].copy_from_slice(&(d as u64).to_le_bytes());

        if len > 0 {
            let (k_off, g_off, v_off) = self.blob_offsets(len, d);
            gpu.copy_d2h_on_stream(self.k_normed, &mut blob[k_off..k_off + key_bytes], stream)?;
            gpu.copy_d2h_on_stream(self.gate, &mut blob[g_off..g_off + key_bytes], stream)?;
            gpu.copy_d2h_on_stream(self.valid, &mut blob[v_off..v_off + len], stream)?;
        }
        Ok(blob)
    }

    /// 2026-09-25: Restore a snapshot's indexer rows into this state.
    ///
    /// A blob that is truncated, of another `index_head_dim`, inconsistent with
    /// its header, or longer than this sequence's reservation is an `Err`;
    /// model-engine's `apply_aux_states` propagates it with `?`, so the
    /// prefix-cache hit fails.
    pub fn restore_blob(&mut self, blob: &[u8], gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        ensure!(
            blob.len() >= HEADER_BYTES,
            "DSA aux blob truncated: {} bytes, need at least {HEADER_BYTES} for the header",
            blob.len()
        );
        let word0 = u64::from_le_bytes(blob[..8].try_into().unwrap());
        // 2026-10-06: The format must match the state's: v2 with the pool cache, v1 without.
        if self.is_pool_cache() {
            ensure!(
                word0 == V2_TAG,
                "DSA aux blob is v1 but this state uses the pool cache \
                 (METRALE_GLM_DSA_POOL_CACHE=1 restores only v2 blobs)"
            );
            return self.restore_v2(blob, gpu, stream);
        }
        ensure!(
            word0 != V2_TAG,
            "DSA aux blob is v2 (pool cache) but this state has no pool cache \
             (METRALE_GLM_DSA_POOL_CACHE is off)"
        );
        let len = word0 as usize;
        let d = u64::from_le_bytes(blob[8..16].try_into().unwrap()) as usize;

        let want_d = self.index_head_dim();
        ensure!(
            d == want_d,
            "DSA aux blob index_head_dim {d} != this layer's {want_d} — the snapshot was \
             taken under a different model geometry"
        );
        ensure!(
            blob.len() == blob_bytes(len, d),
            "DSA aux blob size mismatch: {} bytes for {len} rows at head_dim {d}, expected {}",
            blob.len(),
            blob_bytes(len, d)
        );
        // 2026-09-25: Capacity comes from the configured context, so it can
        // differ from the state that took the snapshot. A longer blob is refused,
        // not truncated: a clamped `len` would select over a prefix while the MLA
        // cache holds the full context.
        self.ensure_room_through(len)?;

        let key_bytes = len * d * 2;
        if len > 0 {
            let (k_off, g_off, v_off) = self.blob_offsets(len, d);
            gpu.copy_h2d_async(&blob[k_off..k_off + key_bytes], self.k_normed, stream)?;
            gpu.copy_h2d_async(&blob[g_off..g_off + key_bytes], self.gate, stream)?;
            gpu.copy_h2d_async(&blob[v_off..v_off + len], self.valid, stream)?;
        }
        // 2026-09-25: The cursor moves last, through `rewind_to` and `advance`,
        // so an early `?` above leaves `len` unchanged and `decode_k`'s lockstep
        // check never sees a partial restore.
        self.rewind_to(0)?;
        self.advance(len)?;
        Ok(())
    }

    /// 2026-10-06: Blob v2 of rows `[0, rows)`: the final pools `[0, pk)` from the persistent
    /// arrays, the tail rows `[kpool * pk, rows)` from the ring, and `valid`. The tail must be
    /// intact in the ring (`RingBook::check_read`, release mode), or this is an `Err`.
    fn snapshot_v2(&self, rows: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<Vec<u8>> {
        let pc = self.pool_cache().expect("pool cache checked by the caller");
        let (d, kp) = (self.index_head_dim(), pc.book.kpool());
        let pk = v2_pools(&pc.book, rows);
        let lay = V2Layout::new(rows, pk, d, kp);
        // 2026-10-06: Release-mode: every tail row must still be intact in the ring (no tail,
        // no ring read: a cut on a pool boundary below the ring travels as pools only).
        if lay.tail > 0 {
            pc.book.check_read(rows, pk)?;
        }
        let mut blob = vec![0u8; lay.bytes()];
        let header = [V2_TAG, rows as u64, d as u64, pk as u64];
        for (i, w) in header.iter().enumerate() {
            blob[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        let [o_pk, o_pi, o_pv, o_k, o_g, o_v] = lay.offsets();
        if pk > 0 {
            gpu.copy_d2h_on_stream(pc.pk, &mut blob[o_pk..o_pi], stream)?;
            gpu.copy_d2h_on_stream(pc.pidx, &mut blob[o_pi..o_pv], stream)?;
            gpu.copy_d2h_on_stream(pc.pvalid, &mut blob[o_pv..o_k], stream)?;
        }
        let row = d * 2;
        for (r0, n) in ring_runs_for(pc.book.ring_rows(), kp * pk, lay.tail) {
            let slot = self.row_offset(kp * pk + r0);
            let (k0, g0) = (o_k + r0 * row, o_g + r0 * row);
            gpu.copy_d2h_on_stream(
                self.k_normed.offset(slot),
                &mut blob[k0..k0 + n * row],
                stream,
            )?;
            gpu.copy_d2h_on_stream(self.gate.offset(slot), &mut blob[g0..g0 + n * row], stream)?;
        }
        if rows > 0 {
            gpu.copy_d2h_on_stream(self.valid, &mut blob[o_v..o_v + rows], stream)?;
        }
        Ok(blob)
    }

    /// 2026-10-06: Restore a v2 blob: pools, tail rows into their ring slots, `valid`, then the
    /// device `pk_len` and, last, the host cursor and watermark. Every check runs before the
    /// first copy.
    fn restore_v2(&mut self, blob: &[u8], gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        ensure!(
            blob.len() >= V2_HEADER_BYTES,
            "DSA aux blob v2 truncated: {} bytes, need at least {V2_HEADER_BYTES} for the header",
            blob.len()
        );
        let word = |i: usize| u64::from_le_bytes(blob[i * 8..i * 8 + 8].try_into().unwrap());
        let (len, d, pk) = (word(1) as usize, word(2) as usize, word(3) as usize);
        let want_d = self.index_head_dim();
        ensure!(
            d == want_d,
            "DSA aux blob index_head_dim {d} != this layer's {want_d} — the snapshot was \
             taken under a different model geometry"
        );
        let (ring, kp) = {
            let pc = self.pool_cache().expect("pool cache checked by the caller");
            let b = &pc.book;
            (b.ring_rows(), b.kpool())
        };
        ensure!(
            pk.checked_mul(kp).is_some_and(|r| r <= len) && len - kp * pk <= ring,
            "DSA aux blob v2: {pk} pools and {len} rows leave a tail the {ring}-row ring \
             cannot hold"
        );
        let lay = V2Layout::new(len, pk, d, kp);
        ensure!(
            blob.len() == lay.bytes(),
            "DSA aux blob v2 size mismatch: {} bytes for {len} rows, {pk} pools at head_dim \
             {d}, expected {}",
            blob.len(),
            lay.bytes()
        );
        // 2026-10-06: Maps the pools (and checks capacity) before any copy, as in v1.
        self.ensure_room_through(len)?;
        let pc = self.pool_cache().expect("pool cache checked by the caller");
        let (p_pk, p_pi, p_pv, p_dev) = (pc.pk, pc.pidx, pc.pvalid, pc.pk_len_dev);
        let [o_pk, o_pi, o_pv, o_k, o_g, o_v] = lay.offsets();
        if pk > 0 {
            gpu.copy_h2d_async(&blob[o_pk..o_pi], p_pk, stream)?;
            gpu.copy_h2d_async(&blob[o_pi..o_pv], p_pi, stream)?;
            gpu.copy_h2d_async(&blob[o_pv..o_k], p_pv, stream)?;
        }
        let row = d * 2;
        for (r0, n) in ring_runs_for(ring, kp * pk, lay.tail) {
            let slot = (kp * pk + r0) % ring * row;
            let (k0, g0) = (o_k + r0 * row, o_g + r0 * row);
            gpu.copy_h2d_async(&blob[k0..k0 + n * row], self.k_normed.offset(slot), stream)?;
            gpu.copy_h2d_async(&blob[g0..g0 + n * row], self.gate.offset(slot), stream)?;
        }
        if len > 0 {
            gpu.copy_h2d_async(&blob[o_v..o_v + len], self.valid, stream)?;
        }
        gpu.copy_h2d_async(&(pk as i32).to_le_bytes(), p_dev, stream)?;
        // 2026-10-06: The cursor and watermark move last, so an early `?` above leaves the
        // state as it was.
        self.set_restored(len, RingBook::restored(ring, kp, pk));
        Ok(())
    }

    /// 2026-09-25: `(k_normed, gate, valid)` byte offsets within a blob of `len`
    /// rows.
    fn blob_offsets(&self, len: usize, d: usize) -> (usize, usize, usize) {
        let key_bytes = len * d * 2;
        (
            HEADER_BYTES,
            HEADER_BYTES + key_bytes,
            HEADER_BYTES + 2 * key_bytes,
        )
    }
}

#[cfg(test)]
mod tests;
