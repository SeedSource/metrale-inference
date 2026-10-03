// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The per-sequence owned-state term of the preflight reserve.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.
//!
//! The term has two owners, reported apart so the boot log names the owner
//! of each byte: the DSA indexer caches of the target's layers, and the state
//! of the draft proposer, which is not in the layer list.

use metrale_config::ModelConfig;
use metrale_model_arch::seq_state_reserve::{
    LazyIndexerReserve, lazy_indexer_reserve, per_sequence_state_bytes,
};

use crate::cli;

/// 2026-09-26: Per-sequence owned device state times `--max-batch-size` (0
/// counts as 1), the charge preflight adds to `inference_reserve`. Logs the
/// split by owner when the charge is nonzero. An error from
/// `per_sequence_state_bytes` counts as no state.
///
/// The batch multiplier is applied here, in the only non-test call to
/// `for_batch`.
pub(super) fn per_sequence_reserve(args: &cli::ServeArgs, config: &ModelConfig) -> usize {
    let spec_on = args.speculative || args.self_speculative || args.dflash;
    // 2026-10-03: `METRALE_DSA_INDEXER_LAZY=1`: one shared, lazily mapped indexer pool instead
    // of `max_batch` full-length caches; the pool is published for the runtime to enforce.
    if metrale_model_arch::glm5next_dsa::lazy::dsa_indexer_lazy()
        && let Ok(Some(lazy)) = lazy_indexer_reserve(
            config,
            args.max_seq_len,
            spec_on,
            args.max_batch_size,
            metrale_model_arch::glm5next_dsa::lazy::pool_gb_from_env(),
        )
    {
        return lazy_reserve(lazy, args.max_batch_size);
    }
    let per_seq = per_sequence_state_bytes(config, args.max_seq_len, spec_on).unwrap_or_default();
    let charge = per_seq.for_batch(args.max_batch_size);
    if charge > 0 {
        tracing::info!(
            "Per-sequence state reserve: {} MB = {} seq x ({} MB target DSA layers + {} MB \
             proposer). Owned per sequence, replicated per rank (EP does not shard the \
             indexer); previously covered only by cuda_headroom.",
            charge / (1024 * 1024),
            args.max_batch_size.max(1),
            per_seq.target_layers / (1024 * 1024),
            per_seq.proposer / (1024 * 1024),
        );
    }
    charge
}

/// 2026-10-03: Publish the lazy pool and return its charge (`LazyIndexerReserve::for_batch`).
fn lazy_reserve(lazy: LazyIndexerReserve, max_batch_size: usize) -> usize {
    use metrale_model_arch::glm5next_dsa::lazy;
    lazy::publish_pool(lazy.pool);
    let charge = lazy.for_batch(max_batch_size);
    let g = metrale_gpu_runtime::lazy_buffer::DEFAULT_GRANULE;
    tracing::info!(
        "Per-sequence state reserve (METRALE_DSA_INDEXER_LAZY=1): {} MB = {} MB shared DSA \
         indexer pool ({} buffers/seq, {} pool tokens in whole granules; \
         METRALE_DSA_INDEXER_POOL_GB={}) + {} seq x {} MB eager (valid flags, proposer \
         scratch). Replicated per rank.",
        charge >> 20,
        lazy.pool.limit_bytes >> 20,
        lazy.pool.shape.bufs_per_seq(),
        lazy.pool.shape.free_tokens(lazy.pool.limit_bytes, g),
        lazy::pool_gb_from_env().map_or_else(|| "default".to_string(), |v| v.to_string()),
        max_batch_size.max(1),
        lazy.eager_per_seq >> 20,
    );
    charge
}

#[cfg(test)]
#[path = "per_sequence_state_tests.rs"]
mod tests;
