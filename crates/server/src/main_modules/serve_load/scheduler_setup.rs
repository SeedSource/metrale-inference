// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The scheduler steps of `load_model` (phase 9) that run before the
//! scheduler thread starts: the batch size, the speculative-decoding mode, the
//! scheduling policy and the device router.
//!
//! Owner: server startup (`met serve`).
//! Invariants: the `tracing` events keep `load_model`'s target (`met::main_modules::serve_load`).
//! `max_batch_size` is not clamped by a model's decode capability here; that
//! is decided at dispatch (`decode_multi_seq_gate`).

use anyhow::Result;

use crate::{cli, scheduler, scheduling_policy};

pub(super) fn resolve_max_batch_size(
    args: &cli::ServeArgs,
    world_size: usize,
    scheduler_model: &dyn metrale_model_engine::traits::Model,
) -> Result<usize> {
    // 2026-09-26: Multi-rank serving honours `--max-batch-size` only when the
    // model runs EP protocol v2 (`METRALE_EP_PROTOCOL=v2`); v1 forces a batch
    // of 1.
    let max_batch_size = if world_size > 1 {
        if scheduler_model.ep_protocol_v2() {
            tracing::info!(target: "met::main_modules::serve_load", "EP v2 active: honoring max_batch_size={}",
                args.max_batch_size,
            );
            args.max_batch_size
        } else {
            tracing::info!(target: "met::main_modules::serve_load", "EP v1 active: forcing max_batch_size=1");
            1
        }
    } else {
        args.max_batch_size
    };
    if scheduler_model.hc_mult() > 0 && max_batch_size > 1 {
        tracing::info!(target: "met::main_modules::serve_load", "mHC highway model: concurrency {max_batch_size} via the per-seq \
             highway decode loop (batched highway kernels are the perf \
             follow-up)"
        );
    }
    // 2026-09-26: The decode-metadata layout is sized from `max_batch_size`
    // (`DecodeMetaLayout`, at least 32 rows); above `DECODE_META_MAX_ROWS`,
    // fail at boot instead of mid-decode.
    anyhow::ensure!(
        max_batch_size <= metrale_gpu_runtime::buffers::DECODE_META_MAX_ROWS,
        "--max-batch-size {max_batch_size} exceeds the derived decode-metadata \
         ceiling of {} rows (DECODE_META_MAX_ROWS)",
        metrale_gpu_runtime::buffers::DECODE_META_MAX_ROWS
    );
    Ok(max_batch_size)
}

/// 2026-09-26: `(use_speculative, use_self_spec, use_ngram_spec, num_drafts,
/// dflash_rung)`.
pub(super) fn resolve_speculation(
    args: &cli::ServeArgs,
    scheduler_model: &dyn metrale_model_engine::traits::Model,
    model_type: &str,
) -> (
    bool,
    bool,
    bool,
    usize,
    metrale_speculative::dflash_rung::DflashRung,
) {
    let use_speculative = (args.speculative || args.dflash) && scheduler_model.has_proposer();
    let use_self_spec = args.self_speculative && scheduler_model.has_self_speculative();
    let use_ngram_spec = args.ngram_speculative;
    let num_drafts = if args.dflash {
        // 2026-09-26: The drafter head's resolved gamma wins over the flag.
        let g = scheduler_model
            .dflash_gamma()
            .unwrap_or_else(|| args.resolved_dflash_gamma(None));
        g.saturating_sub(1).max(1)
    } else {
        args.resolved_num_drafts()
    };
    let dflash_rung = metrale_speculative::dflash_rung::DflashRung::new();
    if args.dflash {
        // 2026-09-26: The head's gamma is the cap; `--dflash-gamma` pins it
        // unless `METRALE_DFLASH_GAMMA_RESOLVER` is set.
        dflash_rung.configure(
            num_drafts + 1,
            args.dflash_gamma.is_some(),
            metrale_model_layers::layers::qwen3_ssm::gdn_flags::gdn_woa_enabled(),
        );
        // 2026-10-09: GLM DFlash C>=2 width (`METRALE_GLM_DFLASH_BATCHED_VERIFY`, or the
        // per-sequence control `METRALE_GLM_DFLASH_PERSEQ_K4`): pinned even under an explicit
        // `--dflash-gamma`, which still sets C=1 and the pools.
        use metrale_model_layers::speculative::glm_dflash_batch as gdb;
        if let Some(k) = gdb::glm_dflash_multi_k(
            model_type,
            gdb::glm_dflash_batched_verify(),
            gdb::glm_dflash_perseq_k4(),
        ) {
            dflash_rung.pin_multi(k);
        }
        // 2026-10-09: `METRALE_DFLASH_CONF_TAU`: read once here (rank 0) and log ENGAGED.
        scheduler::dflash_conf_tau::init();
    }

    if args.dflash {
        tracing::info!(target: "met::main_modules::serve_load", "DFlash speculative decoding: ENABLED (γ={}, window={}, drafter installed)",
            num_drafts + 1,
            if args.dflash_window_size == 0 {
                "full".to_string()
            } else {
                args.dflash_window_size.to_string()
            }
        );
    } else if use_ngram_spec {
        tracing::info!(target: "met::main_modules::serve_load", "N-gram speculative decoding: ENABLED (K=2 verify, CPU proposer)");
    } else if use_self_spec {
        tracing::info!(target: "met::main_modules::serve_load", "Self-speculative decoding: ENABLED ({num_drafts} drafts/step, layer-skipping)"
        );
    } else if use_speculative {
        tracing::info!(target: "met::main_modules::serve_load", "Speculative decoding: ENABLED ({num_drafts} drafts/step)");
    } else if scheduler_model.has_proposer() {
        tracing::info!(target: "met::main_modules::serve_load", "MTP proposer available but speculative decoding disabled (use --speculative to enable)"
        );
    }
    (
        use_speculative,
        use_self_spec,
        use_ngram_spec,
        num_drafts,
        dflash_rung,
    )
}

pub(super) fn scheduling_policy(
    args: &cli::ServeArgs,
) -> Result<Box<dyn scheduling_policy::SchedulingPolicy>> {
    let policy: Box<dyn scheduling_policy::SchedulingPolicy> = match args.scheduler.as_str() {
        "fifo" => {
            tracing::info!(target: "met::main_modules::serve_load", "Scheduling policy: FIFO");
            Box::new(scheduling_policy::FifoPolicy)
        }
        "slai" => {
            tracing::info!(target: "met::main_modules::serve_load", "Scheduling policy: SLAI (TBT deadline={}ms)",
                args.tbt_deadline_ms,
            );
            Box::new(scheduling_policy::SlaiPolicy::new(args.tbt_deadline_ms))
        }
        other => anyhow::bail!("Unknown scheduler '{}'. Supported: fifo, slai", other,),
    };
    Ok(policy)
}

pub(super) fn scheduler_device(
    args: &cli::ServeArgs,
    scheduler_model: std::sync::Arc<dyn metrale_model_engine::traits::Model>,
    max_batch_size: usize,
) -> Box<scheduler::io::DynDevice> {
    if args.scheduler_config == "async" {
        match scheduler::io::AsyncDeviceIo::new(
            std::sync::Arc::clone(&scheduler_model),
            max_batch_size,
        ) {
            Ok(dev) => Box::new(dev),
            Err(e) => {
                tracing::warn!(target: "met::main_modules::serve_load", "--scheduler-config async: {e:#}; serving with the synchronous router"
                );
                Box::new(scheduler::io::SyncDeviceIo::new(scheduler_model))
            }
        }
    } else {
        Box::new(scheduler::io::SyncDeviceIo::new(scheduler_model))
    }
}
