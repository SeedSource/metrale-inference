// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-sequence DSA indexer cache: the `k_normed`, `gate` and `valid` rows the
//! selector reads.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `len <= capacity`: `advance` and `sync_to` refuse to pass `capacity`, and `rewind_to`
//!   only shrinks.
//!
//! The indexer keys and gates are projections of the hidden state (`indexer.wk`,
//! `index_kpool_compress_gate`), which the MLA latent cache does not hold, so they get their
//! own cache. It is a `LayerState`, allocated per sequence by `TransformerLayer::alloc_state`.
//!
//! # Flat, not paged
//!
//! `dsa_kpool_compress` reads `k[raw * D + d]` and `gate[raw * D + d]` at absolute
//! positions, so this is one contiguous buffer per sequence. The MLA latent cache stays
//! paged.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::lazy_buffer::LazyBuffer;

use super::Glm5NextDsaConfig;
use super::select::DsaSelectGeometry;
use metrale_model_layers::layer::LayerState;

/// 2026-09-25: Rows a sequence's indexer cache holds: `cfg.max_context` rounded down to
/// whole pools ([`dsa_capacity`]).
///
/// Each row costs `2 · index_head_dim` BF16 plus 1 B per DSA layer (513 B at
/// `index_head_dim` 128, 5,643 B over GLM-5.3's 11 DSA layers), and every rank holds a full
/// copy. `Glm5NextTextSkeleton::state_budget` charges the same bytes per token.
pub fn max_dsa_context(cfg: &Glm5NextDsaConfig) -> usize {
    dsa_capacity(cfg.max_context, cfg.index_kpool)
}

/// 2026-09-25: Indexer cache rows for `max_context` tokens, rounded down to whole pools (a
/// trailing partial pool is not a pool).
///
/// The allocation ([`max_dsa_context`]) and the serve's per-sequence reserve
/// (`seq_state_reserve::per_sequence_state_bytes`) both call it. `index_kpool` 0 is treated
/// as 1, so it is total for callers that have not validated the config.
pub fn dsa_capacity(max_context: usize, index_kpool: usize) -> usize {
    let kpool = index_kpool.max(1);
    (max_context / kpool) * kpool
}

/// 2026-09-25: Bytes one sequence's indexer cache takes for one DSA layer at `capacity` rows:
/// `capacity * (4 * index_head_dim + 1)`, the sizes [`Glm5NextDsaState::alloc`] allocates
/// for `k_normed`, `gate` and `valid`. The serve's per-sequence reserve uses it.
pub fn indexer_state_bytes(capacity: usize, index_head_dim: usize) -> usize {
    capacity * index_head_dim * 2   // 2026-09-25: k_normed and gate (BF16), then valid (u8)
        + capacity * index_head_dim * 2
        + capacity
}

/// 2026-09-25: One sequence's indexer cache for one DSA layer, allocated once at
/// [`max_dsa_context`] rows and never grown.
///
/// 2026-10-03: With `METRALE_DSA_INDEXER_LAZY=1` (`lazy::dsa_indexer_lazy`), `k_normed` and
/// `gate` are lazily mapped buffers of the same full-length address range: the pointers stay
/// fixed for the sequence's life, and the rows `[0, n)` are backed before anything writes them
/// (`ensure_room_through`, `map_rows_through`). `valid` stays eager.
pub struct Glm5NextDsaState {
    /// 2026-09-25: `[capacity, index_head_dim]` BF16 indexer keys, after the `k_norm`
    /// LayerNorm.
    pub k_normed: DevicePtr,
    /// 2026-09-25: `[capacity, index_head_dim]` BF16 compress-gate projection.
    pub gate: DevicePtr,
    /// 2026-09-25: `[capacity]` u8 per-position validity.
    pub valid: DevicePtr,
    /// 2026-09-25: Tokens written so far. The selector reads `[0, len)`.
    len: usize,
    capacity: usize,
    index_head_dim: usize,
    /// 2026-09-25: Set by [`Self::free`], which then does nothing on a second call.
    released: bool,
    /// 2026-10-03: `[k_normed, gate]` as lazily mapped buffers (lever on); `None` = eager.
    lazy: Option<[LazyBuffer; 2]>,
    /// 2026-10-03: Rows `map_rows_through` maps past the end it is given (the MTP proposer's
    /// draft look-ahead, `lazy::PROPOSER_LOOKAHEAD_ROWS`; 0 for a target layer).
    lookahead: usize,
}

impl Glm5NextDsaState {
    /// 2026-09-25: Reserve for the whole addressable context. `alloc_state` has no length
    /// argument, so the cap, not the prompt, sizes this.
    pub fn alloc(gpu: &dyn GpuBackend, cfg: &Glm5NextDsaConfig) -> Result<Self> {
        Self::alloc_with_lookahead(gpu, cfg, 0)
    }

    /// 2026-10-03: [`Self::alloc`] for a state that maps `lookahead` rows past the end
    /// `map_rows_through` is given (the MTP proposer). Lever off, identical to `alloc`.
    pub fn alloc_with_lookahead(
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextDsaConfig,
        lookahead: usize,
    ) -> Result<Self> {
        if super::lazy::dsa_indexer_lazy() {
            return Self::alloc_lazy(gpu, cfg, lookahead, super::lazy::indexer_pool());
        }
        cfg.validate()?;
        let capacity = max_dsa_context(cfg);
        let d = cfg.index_head_dim;
        Ok(Self {
            k_normed: gpu.alloc(capacity * d * 2)?,
            gate: gpu.alloc(capacity * d * 2)?,
            valid: gpu.alloc(capacity)?,
            len: 0,
            capacity,
            index_head_dim: d,
            released: false,
            lazy: None,
            lookahead,
        })
    }

    /// 2026-10-03: The lazily mapped layout: `k_normed` and `gate` reserved at full length and
    /// charged to `pool` as rows are mapped, `valid` allocated eagerly. Nothing is mapped yet.
    pub fn alloc_lazy(
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextDsaConfig,
        lookahead: usize,
        pool: std::sync::Arc<metrale_gpu_runtime::lazy_buffer::MapBudget>,
    ) -> Result<Self> {
        cfg.validate()?;
        let capacity = max_dsa_context(cfg);
        let d = cfg.index_head_dim;
        let k = gpu.alloc_lazy(capacity * d * 2, Some(pool.clone()))?;
        let g = match gpu.alloc_lazy(capacity * d * 2, Some(pool)) {
            Ok(g) => g,
            Err(e) => {
                let _ = k.release(gpu);
                return Err(e);
            }
        };
        let valid = match gpu.alloc(capacity) {
            Ok(v) => v,
            Err(e) => {
                let _ = k.release(gpu);
                let _ = g.release(gpu);
                return Err(e);
            }
        };
        Ok(Self {
            k_normed: k.ptr(),
            gate: g.ptr(),
            valid,
            len: 0,
            capacity,
            index_head_dim: d,
            released: false,
            lazy: Some([k, g]),
            lookahead,
        })
    }

    /// 2026-10-03: Rows backed in `k_normed` and `gate` (the smaller of the two); `None` when
    /// the buffers are eager.
    pub fn mapped_rows(&self) -> Option<usize> {
        let row = self.index_head_dim * 2;
        self.lazy
            .as_ref()
            .map(|[k, g]| k.mapped_bytes().min(g.mapped_bytes()) / row.max(1))
    }

    /// 2026-10-03: Back rows `[0, min(end, capacity))` of `k_normed` and `gate`. A no-op for
    /// eager buffers and for rows already backed. Fails (with nothing new charged) when the
    /// indexer pool is full or a stream capture is active.
    fn map_rows(&self, end: usize) -> Result<()> {
        if let Some(bufs) = &self.lazy {
            let bytes = end.min(self.capacity) * self.index_head_dim * 2;
            for b in bufs {
                b.ensure_mapped(bytes)?;
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// 2026-09-25: Indexer key width. The aux-state codec (`aux_state.rs`) refuses a blob
    /// taken at a different width.
    pub fn index_head_dim(&self) -> usize {
        self.index_head_dim
    }

    /// 2026-09-25: Byte offset of row `pos` in `k_normed` / `gate`.
    pub fn row_offset(&self, pos: usize) -> usize {
        pos * self.index_head_dim * 2
    }

    /// 2026-09-25: Fails when `n` more rows would pass `capacity`.
    ///
    /// `Glm5NextDsaLayer::indexer_forward` writes row `len()` before it advances, so it calls
    /// this first: at `len == capacity` that row is past the end of every buffer.
    pub fn ensure_room(&self, n: usize) -> Result<()> {
        self.ensure_room_through(self.len + n)
    }

    /// 2026-09-25: The same refusal for an absolute end row.
    ///
    /// Before a graph replay the counter may still be ahead of the sequence (a rejected
    /// draft; `sync_to` fixes it after the replay), so the replay check
    /// (`check_replay_room`) asks about the end row `seq_len + k` instead.
    pub fn ensure_room_through(&self, end: usize) -> Result<()> {
        if end > self.capacity {
            bail!(
                "DSA indexer cache: {end} tokens exceeds the {} rows reserved for this \
                 sequence. The indexer cache is sized from --max-seq-len at sequence \
                 creation and never grows; raise --max-seq-len (and re-check the memory \
                 budget) to serve a longer context.",
                self.capacity
            );
        }
        // 2026-10-03: Lever on, back the rows before the caller writes them. Every write
        // path (`indexer_forward`, the wide and row-batch prefills, aux restore) and the graph
        // replay pre-check (`check_replay_room`) come through here first.
        self.map_rows(end)
    }

    /// 2026-09-25: Advance after writing `n` rows at `[len, len + n)`. Fails, without
    /// changing `len`, when that would pass `capacity`.
    pub fn advance(&mut self, n: usize) -> Result<()> {
        self.ensure_room(n)?;
        self.len += n;
        Ok(())
    }

    /// 2026-09-25: Rewind to `n` rows after a rejected speculative draft.
    ///
    /// The rows in `[n, len)` are left in the cache but become unreachable: the selector reads
    /// `[0, len)` and the next write starts at `n`, so they are overwritten before anything
    /// can select over them. Only shrinks — growing is `advance`'s job, and a request to
    /// "rewind" forward would mean the caller lost track of where the sequence is.
    pub fn rewind_to(&mut self, n: usize) -> Result<()> {
        if n > self.len {
            bail!(
                "DSA indexer rewind to {n} from {}: rewind only shrinks; a forward 'rewind' \
                 means the caller lost the sequence position",
                self.len
            );
        }
        self.len = n;
        Ok(())
    }

    /// 2026-09-25: Put the counter where `decode_k` would have left it, for a step served by
    /// a replayed CUDA graph. `seq_len` is the sequence length before this step's `k` rows.
    ///
    /// The same reconcile `decode_k` does on entry: the counter is rewound to `seq_len` when
    /// it is ahead (a rejected draft), a counter behind `seq_len` is an error, and then it
    /// advances by `k`.
    pub fn sync_to(&mut self, seq_len: usize, k: usize) -> Result<()> {
        match self.len.cmp(&seq_len) {
            std::cmp::Ordering::Greater => self.rewind_to(seq_len)?,
            std::cmp::Ordering::Less => bail!(
                "DSA indexer cache holds {} tokens but the replayed step starts at {seq_len} \
                 — rows are MISSING, not merely stale.",
                self.len
            ),
            std::cmp::Ordering::Equal => {}
        }
        self.advance(k)
    }

    /// 2026-09-25: Plan a selection over everything cached so far.
    pub fn geometry(&self, cfg: &Glm5NextDsaConfig, q_rows: usize) -> Result<DsaSelectGeometry> {
        DsaSelectGeometry::plan(cfg, self.len, q_rows)
    }

    /// 2026-09-25: Release the per-sequence device buffers.
    ///
    /// Takes `&mut self` because callers hold the state behind a `dyn` state object. A second
    /// call does nothing (`released` is set first), since `release_state` on the layer and the
    /// MTP proposer's `free_state` can both reach one state. After the three frees succeed the
    /// pointers are nulled and `len` is 0.
    pub fn free(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.released {
            return Ok(());
        }
        self.released = true;
        match self.lazy.take() {
            Some([k, g]) => {
                k.release(gpu)?;
                g.release(gpu)?;
                gpu.free(self.valid)?;
            }
            None => {
                for p in [self.k_normed, self.gate, self.valid] {
                    gpu.free(p)?;
                }
            }
        }
        self.k_normed = DevicePtr(0);
        self.gate = DevicePtr(0);
        self.valid = DevicePtr(0);
        self.len = 0;
        Ok(())
    }
}

impl LayerState for Glm5NextDsaState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// 2026-10-03: Back rows through `end + lookahead` (capped at capacity) ahead of the step,
    /// so the write paths and any capture or replay find them mapped. Eager: nothing to do.
    fn map_rows_through(&self, end: usize) -> Result<()> {
        if self.lazy.is_none() || self.released {
            return Ok(());
        }
        self.map_rows(end.saturating_add(self.lookahead))
    }

    /// 2026-10-05: `map_rows_through(end)` maps `min(end + lookahead, capacity)` rows, so it is a
    /// no-op for every `end` up to `mapped_rows - lookahead`, and for any `end` once the whole
    /// capacity is backed. Eager or released: `None` (nothing is ever mapped).
    fn rows_backed_through(&self) -> Option<usize> {
        if self.released {
            return None;
        }
        let mapped = self.mapped_rows()?;
        if mapped >= self.capacity {
            return Some(usize::MAX);
        }
        Some(mapped.saturating_sub(self.lookahead))
    }
}

#[cfg(test)]
mod tests;
