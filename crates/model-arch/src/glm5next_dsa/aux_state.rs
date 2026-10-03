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

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;

use super::state::Glm5NextDsaState;

/// 2026-09-25: `[len u64][index_head_dim u64]`, little-endian.
const HEADER_BYTES: usize = 16;

/// 2026-09-25: Bytes a blob occupies for `len` rows at `index_head_dim`: the
/// per-row size of [`super::state::indexer_state_bytes`], over `len` rows, plus
/// the header.
fn blob_bytes(len: usize, index_head_dim: usize) -> usize {
    HEADER_BYTES + len * (index_head_dim * 4 + 1)
}

impl Glm5NextDsaState {
    /// 2026-10-03: Bytes `snapshot_blob_prefix(rows)` returns, without allocating.
    pub fn blob_bytes_for_rows(&self, rows: usize) -> usize {
        blob_bytes(rows, self.index_head_dim())
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
        let len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
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
