// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Lazily mapped DSA indexer caches (`METRALE_DSA_INDEXER_LAZY=1`, default off):
//! the lever, the shared indexer pool, and the sizing arithmetic preflight and the runtime
//! share.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Lever off: nothing here is consulted by the allocation, write or reserve paths, which run
//!   exactly as before.
//! - Lever on: each sequence's `k_normed` and `gate` buffers keep their full-length address
//!   range (so pointers, and graphs that baked them, are stable), but only the granules that
//!   cover written positions are backed. Every mapped byte is charged to one process-wide
//!   [`MapBudget`] whose limit preflight reserved, so mapping never takes memory the KV pool
//!   was sized from.
//! - Every mapping decision depends only on token positions (the KV block grid and the row a
//!   write path is about to write), and the pool limit is checked equal on every rank
//!   (`rank_agree`), so EP ranks map, charge and refuse identically.
//!   2026-10-05: Except the VMM lazy-map floor, which reads each rank's own free device memory
//!   (A168): a prefill chunk's admission is therefore voted on by every rank before its first
//!   collective (model-engine `prefill_b/lazy_agree.rs`), and so is each decode or verify step
//!   that maps past the rank-agreed extent (model-engine `trait_impl/decode_lazy_agree.rs`).
//!
//! The `valid` byte array stays eagerly allocated: one 2 MiB granule holds 2 M of its rows,
//! so mapping it lazily would round 131 KB up to 2 MiB per layer.

use std::sync::{Arc, OnceLock};

use metrale_gpu_runtime::lazy_buffer::{DEFAULT_GRANULE, MapBudget, round_up_to_granule};

/// 2026-10-03: Rows past the requested end that a proposer's indexer cache maps ahead, for the
/// draft rows it writes beyond the target's KV block grid (`num_drafts` per propose, a few at
/// most). Small next to the 8,192 rows one granule holds, so it costs at most one granule early.
pub const PROPOSER_LOOKAHEAD_ROWS: usize = 256;

/// 2026-10-03: Tokens per sequence the default pool guarantees: `max_batch` sequences of
/// `min(capacity, DEFAULT_POOL_TOKENS_PER_SEQ)` tokens always fit.
pub const DEFAULT_POOL_TOKENS_PER_SEQ: usize = 32_768;

/// 2026-10-03: `METRALE_DSA_INDEXER_LAZY=1`: map indexer cache rows on demand. Read once.
pub fn dsa_indexer_lazy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("METRALE_DSA_INDEXER_LAZY").as_deref(),
            Ok("1")
        )
    })
}

/// 2026-10-05: `METRALE_LAZY_MAP_FAIL_ABOVE_ROWS` (test-only fault injection, default 0 = off):
/// a lazy map past this many token rows refuses like the free-memory floor does. Read once,
/// per process (per rank); deliberately NOT part of any startup rank-agreement check, so a
/// test can set it on one rank only.
pub fn fail_above_rows() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("METRALE_LAZY_MAP_FAIL_ABOVE_ROWS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// 2026-10-03: `METRALE_DSA_INDEXER_POOL_GB` (GiB, fractional allowed), or `None` when unset or
/// not a positive number.
pub fn pool_gb_from_env() -> Option<f64> {
    std::env::var("METRALE_DSA_INDEXER_POOL_GB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g > 0.0)
}

/// 2026-10-03: Bytes one indexer row takes in `k_normed` or in `gate` (BF16).
pub fn row_bytes(index_head_dim: usize) -> usize {
    index_head_dim * 2
}

/// 2026-10-03: Mapped bytes of one lazily mapped buffer holding `rows` rows, in whole granules.
pub fn mapped_bytes_for_rows(rows: usize, index_head_dim: usize, granule: usize) -> usize {
    round_up_to_granule(rows * row_bytes(index_head_dim), granule)
}

/// 2026-10-03: The shape of one sequence's lazily mapped indexer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LazyShape {
    /// 2026-10-03: Text DSA layers, each holding a `k_normed` and a `gate` buffer.
    pub dsa_layers: usize,
    /// 2026-10-03: Whether the MTP proposer holds one more DSA indexer cache.
    pub proposer: bool,
    pub index_head_dim: usize,
    /// 2026-10-03: Rows each buffer can hold (`dsa_capacity`).
    pub capacity: usize,
    /// 2026-10-06: `METRALE_GLM_DSA_POOL_CACHE=1`: the two lazily mapped buffers per indexer
    /// cache are the pool keys `[pools, d]` f32 and pool ids `[pools, kpool]` i32
    /// (`pool_cache.rs`), not `k_normed`/`gate`.
    pub pool_cache: bool,
    /// 2026-10-06: Tokens per pool; read only with `pool_cache`.
    pub index_kpool: usize,
}

impl LazyShape {
    /// 2026-10-03: Lazily mapped buffers per sequence (two per DSA indexer cache).
    pub fn bufs_per_seq(&self) -> usize {
        2 * (self.dsa_layers + usize::from(self.proposer))
    }

    /// 2026-10-03: Mapped bytes one sequence of `tokens` tokens needs, granule rounding and the
    /// proposer's look-ahead included.
    pub fn seq_mapped_bytes(&self, tokens: usize, granule: usize) -> usize {
        if self.pool_cache {
            return self.pool_cache_mapped_bytes(tokens, granule);
        }
        let t = tokens.min(self.capacity);
        let target = 2 * self.dsa_layers * mapped_bytes_for_rows(t, self.index_head_dim, granule);
        let proposer = if self.proposer {
            let p = (t + PROPOSER_LOOKAHEAD_ROWS).min(self.capacity);
            2 * mapped_bytes_for_rows(p, self.index_head_dim, granule)
        } else {
            0
        };
        target + proposer
    }

    /// 2026-10-03: The default pool: `max_batch` (0 counts as 1) sequences of
    /// `min(capacity, 32,768)` tokens each.
    pub fn default_pool_bytes(&self, max_batch: usize, granule: usize) -> usize {
        max_batch.max(1)
            * self.seq_mapped_bytes(self.capacity.min(DEFAULT_POOL_TOKENS_PER_SEQ), granule)
    }

    /// 2026-10-03: Tokens a new sequence is certain to get from `available` pool bytes: whole
    /// granule sets (one granule in every buffer), each worth `granule / row_bytes` tokens.
    ///
    /// 2026-10-06: With `pool_cache`, tokens are counted in whole pool-key granules
    /// (`granule / (4 d)` pools each), each set of them needing a pool-id granule per cache
    /// every `granule / 4` pools ([`Self::pool_cache_free_tokens`]).
    pub fn free_tokens(&self, available: usize, granule: usize) -> usize {
        if self.pool_cache {
            return self.pool_cache_free_tokens(available, granule);
        }
        let set = self.bufs_per_seq() * granule;
        let rows_per_granule = granule / row_bytes(self.index_head_dim).max(1);
        if set == 0 {
            return usize::MAX;
        }
        (available / set) * rows_per_granule
    }
}

impl LazyShape {
    /// 2026-10-06: Indexer caches per sequence (text DSA layers and the proposer's).
    fn caches(&self) -> usize {
        self.dsa_layers + usize::from(self.proposer)
    }

    /// 2026-10-06: Mapped bytes of one cache's `pk` and `pidx` at `rows` rows, as
    /// `PoolCache::map_rows` maps them (whole pools, then whole granules).
    fn pool_cache_cache_bytes(&self, rows: usize, granule: usize) -> usize {
        let kp = self.index_kpool.max(1);
        let pools = rows.min(self.capacity).div_ceil(kp);
        round_up_to_granule(pools * self.index_head_dim * 4, granule)
            + round_up_to_granule(pools * kp * 4, granule)
    }

    /// 2026-10-06: `seq_mapped_bytes` with the pool cache.
    fn pool_cache_mapped_bytes(&self, tokens: usize, granule: usize) -> usize {
        let t = tokens.min(self.capacity);
        let target = self.dsa_layers * self.pool_cache_cache_bytes(t, granule);
        let proposer = if self.proposer {
            let p = (t + PROPOSER_LOOKAHEAD_ROWS).min(self.capacity);
            self.pool_cache_cache_bytes(p, granule)
        } else {
            0
        };
        target + proposer
    }

    /// 2026-10-06: `free_tokens` with the pool cache: the most tokens `n` whole pool-key
    /// granules cover (`n * rows_pk`, `rows_pk = granule / (4 d) * kpool`) such that every cache
    /// can map them and the `ceil(n * rows_pk / rows_id)` pool-id granules they need
    /// (`rows_id = granule / 4`, at kpool 4 and d 128 one id granule per 32 key granules) out
    /// of `available`.
    fn pool_cache_free_tokens(&self, available: usize, granule: usize) -> usize {
        let caches = self.caches();
        let kp = self.index_kpool.max(1);
        let pk_pools = granule / (self.index_head_dim * 4).max(1);
        let id_pools = granule / (kp * 4);
        if caches == 0 || granule == 0 {
            return usize::MAX;
        }
        if pk_pools == 0 || id_pools == 0 {
            return 0;
        }
        // 2026-10-06: Granules each cache may map, then the most key granules `n` with
        // `n + ceil(n * pk_pools / id_pools) <= budget`.
        let budget = (available / granule / caches) as u128;
        let (a, b) = (pk_pools as u128, id_pools as u128);
        let cost = |n: u128| n + (n * a).div_ceil(b);
        let mut n = budget * b / (a + b);
        while n > 0 && cost(n) > budget {
            n -= 1;
        }
        while cost(n + 1) <= budget {
            n += 1;
        }
        let tokens = n * a * kp as u128;
        usize::try_from(tokens).unwrap_or(usize::MAX)
    }
}

/// 2026-10-03: The pool preflight sized and the runtime enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolConfig {
    pub limit_bytes: usize,
    pub shape: LazyShape,
}

static PUBLISHED: OnceLock<PoolConfig> = OnceLock::new();
static POOL: OnceLock<Arc<MapBudget>> = OnceLock::new();

/// 2026-10-03: Publish the pool preflight reserved, before the model is built. A second call
/// with a different value is ignored with a warning (the first one sized the reserve).
pub fn publish_pool(cfg: PoolConfig) {
    if PUBLISHED.set(cfg).is_err()
        && let Some(first) = PUBLISHED.get()
        && *first != cfg
    {
        tracing::warn!("DSA indexer pool already published ({first:?}); ignoring {cfg:?}");
    }
}

/// 2026-10-03: The published pool, if any.
pub fn published_pool() -> Option<PoolConfig> {
    PUBLISHED.get().copied()
}

/// 2026-10-03: The process-wide budget every lazily mapped indexer cache charges. Its limit is
/// the published pool; without one (a caller that skipped preflight) it is
/// `METRALE_DSA_INDEXER_POOL_GB`, else unlimited.
pub fn indexer_pool() -> Arc<MapBudget> {
    POOL.get_or_init(|| {
        let limit = match (published_pool(), pool_gb_from_env()) {
            (Some(p), _) => p.limit_bytes,
            (None, Some(gb)) => (gb * (1u64 << 30) as f64) as usize,
            (None, None) => usize::MAX,
        };
        Arc::new(MapBudget::new("DSA indexer", limit))
    })
    .clone()
}

/// 2026-10-03: Tokens a new sequence is certain to get from the pool now, or `None` when the
/// lever is off or no pool shape was published. The scheduler's free-block count is capped
/// by it (`num_free_blocks_dispatch`), so admission and preemption-resume wait for room.
pub fn pool_free_tokens() -> Option<usize> {
    if !dsa_indexer_lazy() {
        return None;
    }
    let cfg = published_pool()?;
    Some(
        cfg.shape
            .free_tokens(indexer_pool().available(), DEFAULT_GRANULE),
    )
}

/// 2026-10-03: Tokens the whole pool holds for new sequences when empty (whole granule sets),
/// or `None` when the lever is off or no pool shape was published.
pub fn pool_total_tokens() -> Option<usize> {
    if !dsa_indexer_lazy() {
        return None;
    }
    let cfg = published_pool()?;
    Some(
        cfg.shape
            .free_tokens(indexer_pool().limit(), DEFAULT_GRANULE),
    )
}

#[cfg(test)]
#[path = "lazy_tests.rs"]
mod tests;
