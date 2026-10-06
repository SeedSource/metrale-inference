// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The fast loader's weight-arena hook (`ModelWeightLoader::arena_predicate`).
//!
//! Owner: server (serve phases, weight load).
//! Invariants: asked once per load, before it; `None` leaves every tensor allocated alone.

use metrale_config::ModelConfig;

/// 2026-10-06: The checkpoint tensors the model's weight loader keeps until teardown and never
/// frees alone (`arena_predicate`), which the fast loader then places in one weight arena
/// instead of one allocation each. `None` (every tensor allocated alone) when the model type
/// does not resolve or the loader claims nothing, the trait default; GLM-5.3 claims its routed
/// experts under `METRALE_GLM_WEIGHT_ARENA=1`.
pub(super) fn arena_hook(
    config: &ModelConfig,
) -> Option<metrale_model_weights::weights::ArenaHook> {
    let hook = metrale_model_engine::factory::loader_for_config(config)
        .ok()?
        .arena_predicate(config)?;
    tracing::info!(
        "Weight loader for model_type '{}' places part of the checkpoint in a weight arena: \
         a few large allocations, sub-allocated, instead of one per tensor.",
        config.model_type,
    );
    Some(hook)
}
