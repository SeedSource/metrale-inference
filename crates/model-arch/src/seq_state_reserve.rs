// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-sequence device state the serve must reserve, computed from
//! the config before the model exists.
//!
//! The server's `preflight_reserve` adds this charge (`per_sequence_reserve`)
//! before the model is built, when there is no layer list, proposer or
//! `GpuBackend`, so it is a function of the config alone, like the
//! `ssm_reserve` terms.
//!
//! Charged, per sequence:
//!   * the target stack's DSA indexer caches (`Glm5NextDsaState`), one per text DSA layer;
//!   * the draft proposer's own state (what `Glm5NextMtpHead`'s `alloc_state` allocates),
//!     in its own field: the proposer is not a `TransformerLayer` and is not in the layer list.
//!
//! Not charged here:
//!   * KDA recurrent and conv state: pool-owned (`ssm_reserve::ssm_pool_reserve_bytes`);
//!     `meta.rs` gives pool-backed linear-attention layers pool addresses instead of calling
//!     `alloc_state`;
//!   * the paged KV pool: this reserve is subtracted from the KV budget.
//!
//! The result is per-rank bytes; nothing here divides by a world size.
//!
//! Owner: model-arch (serve memory reserve).
//! Invariants:
//! - Any `model_type` other than `glm5_next` is charged zero.

use anyhow::Result;
use metrale_config::{LayerType, ModelConfig};

use crate::glm5next_dsa::state::{dsa_capacity, indexer_state_bytes};
use crate::glm5next_skeleton::{Glm5NextTextSkeleton, Mixer};

/// 2026-09-25: Per-sequence, per-rank device state, split by owner.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PerSequenceState {
    /// 2026-09-25: Indexer caches owned by the target stack's DSA layers.
    pub target_layers: usize,
    /// 2026-09-25: State owned by the draft proposer (a `DraftProposer`, not a `TransformerLayer`).
    pub proposer: usize,
}

impl PerSequenceState {
    pub fn total(&self) -> usize {
        self.target_layers + self.proposer
    }

    /// 2026-09-25: What the reserve must hold for `max_batch_size` concurrent
    /// sequences (0 counts as 1).
    pub fn for_batch(&self, max_batch_size: usize) -> usize {
        self.total() * max_batch_size.max(1)
    }
}

/// 2026-09-25: Per-sequence owned device state for `config`, at a context of `max_seq_len`
/// tokens.
///
/// Returns zeros for any `model_type` other than `glm5_next`. The proposer term needs
/// `spec_on` and a sparse-attention MTP layer. Errors when the GLM skeleton cannot be built
/// from `config`.
pub fn per_sequence_state_bytes(
    config: &ModelConfig,
    max_seq_len: usize,
    spec_on: bool,
) -> Result<PerSequenceState> {
    if config.model_type != "glm5_next" {
        return Ok(PerSequenceState::default());
    }
    let capacity = dsa_capacity(max_seq_len, config.index_kpool);
    let per_layer = indexer_state_bytes(capacity, config.index_head_dim);

    // 2026-09-25: The skeleton, built from the config alone, says which layers carry a DSA
    // mixer. The per-layer bytes use the pool-rounded capacity (`dsa_capacity`), the same
    // capacity `Glm5NextDsaState::alloc` allocates.
    let skeleton = Glm5NextTextSkeleton::from_config(config)?;
    let dsa_layers = skeleton
        .layers
        .iter()
        .filter(|l| l.mixer == Mixer::Dsa)
        .count();

    let target_layers = dsa_layers * per_layer;

    // 2026-09-25: The GLM MTP head's `alloc_state` (`glm5next_mtp_head/proposer.rs`)
    // allocates one DSA indexer block for its drafter layer and five buffers: `concat`,
    // `x`, `logits`, `arg`, `head_xchg`, in that order below. Its `block_table` is a host
    // `Vec<u32>`, not device memory.
    let proposer = if spec_on && config.mtp_layer_types.contains(&LayerType::SparseAttention) {
        per_layer
            + 2 * config.hidden_size * 2
            + config.hidden_size * 2
            + config.vocab_size * 2
            + 4
            + 16
    } else {
        0
    };

    Ok(PerSequenceState {
        target_layers,
        proposer,
    })
}

/// 2026-10-03: The reserve with lazily mapped indexer caches (`METRALE_DSA_INDEXER_LAZY=1`):
/// one shared pool for every sequence's `k_normed`/`gate` rows instead of `max_batch` full-length
/// copies, plus the per-sequence parts that stay eager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LazyIndexerReserve {
    /// 2026-10-03: The pool preflight charges and publishes (`lazy::publish_pool`).
    pub pool: crate::glm5next_dsa::lazy::PoolConfig,
    /// 2026-10-03: Eager bytes per sequence: the `valid` byte array of every indexer cache at
    /// full capacity, and the proposer's five scratch buffers.
    pub eager_per_seq: usize,
}

impl LazyIndexerReserve {
    /// 2026-10-03: What preflight charges for `max_batch_size` sequences (0 counts as 1).
    pub fn for_batch(&self, max_batch_size: usize) -> usize {
        self.pool.limit_bytes + self.eager_per_seq * max_batch_size.max(1)
    }
}

/// 2026-10-03: [`LazyIndexerReserve`] for `config` at `max_seq_len`, or `None` for any
/// `model_type` other than `glm5_next`. The pool is `pool_gb` GiB when given, else the default
/// (`LazyShape::default_pool_bytes`: `max_batch_size` sequences of `min(capacity, 32,768)`
/// tokens), and never more than `max_batch_size` full-length sequences need. Pure: the caller
/// reads the lever and the environment.
pub fn lazy_indexer_reserve(
    config: &ModelConfig,
    max_seq_len: usize,
    spec_on: bool,
    max_batch_size: usize,
    pool_gb: Option<f64>,
) -> Result<Option<LazyIndexerReserve>> {
    use crate::glm5next_dsa::lazy::{LazyShape, PoolConfig};
    use metrale_gpu_runtime::lazy_buffer::DEFAULT_GRANULE;
    if config.model_type != "glm5_next" {
        return Ok(None);
    }
    let skeleton = Glm5NextTextSkeleton::from_config(config)?;
    let dsa_layers = skeleton
        .layers
        .iter()
        .filter(|l| l.mixer == Mixer::Dsa)
        .count();
    let proposer = spec_on && config.mtp_layer_types.contains(&LayerType::SparseAttention);
    let capacity = dsa_capacity(max_seq_len, config.index_kpool);
    let shape = LazyShape {
        dsa_layers,
        proposer,
        index_head_dim: config.index_head_dim,
        capacity,
    };
    let mb = max_batch_size.max(1);
    let full = mb * shape.seq_mapped_bytes(capacity, DEFAULT_GRANULE);
    let limit = match pool_gb {
        Some(gb) => ((gb * (1u64 << 30) as f64) as usize).min(full),
        None => shape.default_pool_bytes(mb, DEFAULT_GRANULE),
    };
    let proposer_scratch = if proposer {
        2 * config.hidden_size * 2 + config.hidden_size * 2 + config.vocab_size * 2 + 4 + 16
    } else {
        0
    };
    Ok(Some(LazyIndexerReserve {
        pool: PoolConfig {
            limit_bytes: limit,
            shape,
        },
        eager_per_seq: (dsa_layers + usize::from(proposer)) * capacity + proposer_scratch,
    }))
}

#[cfg(test)]
#[path = "seq_state_reserve_tests.rs"]
mod tests;
