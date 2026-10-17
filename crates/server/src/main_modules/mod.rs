// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The server binary's `met serve` modules: startup, model load
//! and swap, `AppState`, HTTP middleware and telemetry boot.
//!
//! Owner: server.
//! Invariants: none beyond the types.

pub(crate) mod app_state;
pub(crate) mod auto_swap;
pub(crate) mod byte_count;
pub(crate) mod kernel_flag_plan;
pub(crate) mod kv_dtypes;
pub(crate) mod middleware;
pub(crate) mod model_host;
pub(crate) mod model_swap;
pub(crate) mod promotion;
pub(crate) mod serve;
pub(crate) mod serve_flags;
pub(crate) mod serve_load;
pub(crate) mod serve_phases;
mod serve_warmup;
mod serve_router;
pub(crate) mod telemetry_boot;
#[cfg(feature = "otlp")]
mod telemetry_otlp;

#[cfg(test)]
mod tests;

pub(crate) use app_state::AppState;
pub(crate) use kv_dtypes::{auto_high_precision_layers, build_layer_kv_dtypes};
pub(crate) use serve::serve;

/// 2026-10-10: Whether this is a hybrid serve: `METRALE_HYBRID_DFLASH=1` with `--dflash
/// --speculative` (`validate_serve_args` refuses the lever without both).
pub(crate) fn hybrid_serve(args: &crate::cli::ServeArgs) -> bool {
    args.dflash
        && args.speculative
        && metrale_model_layers::speculative::hybrid::hybrid_dflash_enabled()
}
