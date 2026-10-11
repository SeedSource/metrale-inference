// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Model factory call, prefix-cache and high-speed-swap setup, and
//! the EP worker loop that ranks above 0 run instead of serving.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use anyhow::{Context, Result};

use metrale_config::ModelConfig;

use crate::cli;

pub(crate) fn build_prefix_cache(
    args: &cli::ServeArgs,
    config: &ModelConfig,
) -> Box<dyn metrale_telemetry::prefix_cache::PrefixCache> {
    if args.prefix_caching_enabled() && !config.kv_only_prefix_cache_is_safe() {
        tracing::warn!(
            model_type = %config.model_type,
            "Prefix caching: DISABLED because this model builds per-sequence state outside KV; \
             the KV-only cache cannot resume it exactly"
        );
        return Box::new(metrale_telemetry::prefix_cache::NoPrefixCaching);
    }
    if args.prefix_caching_enabled() {
        if args.high_speed_swap {
            tracing::info!(
                "Prefix caching: ENABLED (radix tree, with --high-speed-swap disk-side refcounts)"
            );
        } else {
            tracing::info!("Prefix caching: ENABLED (radix tree)");
        }
        Box::new(metrale_cache::radix_tree::RadixTree::new())
    } else {
        tracing::info!("Prefix caching: disabled");
        Box::new(metrale_telemetry::prefix_cache::NoPrefixCaching)
    }
}

/// 2026-09-26: The effective `--swap-space-gb`: 0, with a warning, when the
/// model reports a KV-only swap-out as unsafe (`kv_only_swap_out_is_safe`).
/// Disabled rather than refused, because the flag's default is 3 and such a
/// model would otherwise fail to boot on a value nobody typed.
pub(crate) fn resolve_swap_space_gb(args: &cli::ServeArgs, config: &ModelConfig) -> usize {
    if args.swap_space_gb > 0 && !config.kv_only_swap_out_is_safe() {
        tracing::warn!(
            model_type = %config.model_type,
            requested_gb = args.swap_space_gb,
            "Swap space: DISABLED because this model builds per-sequence state outside KV; \
             the KV-only spill image cannot restore it. Decode preemption falls back to \
             requeue-resume, which re-prefills and is always correct."
        );
        return 0;
    }
    args.swap_space_gb
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_model(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    store: metrale_model_weights::weights::WeightStore,
    gpu: Box<dyn metrale_gpu_runtime::gpu::GpuBackend>,
    max_batch_tokens: usize,
    kv_dtype: metrale_cache::kv_cache::KvCacheDtype,
    inference_reserve: usize,
    layer_dtypes: Vec<metrale_cache::kv_cache::KvCacheDtype>,
    hss_cache_blocks_per_seq: Option<u32>,
    prefix_cache: Box<dyn metrale_telemetry::prefix_cache::PrefixCache>,
    comm: Option<std::sync::Arc<dyn metrale_comm::CommBackend>>,
    dflash_args: Option<metrale_model_engine::factory::DflashBuildArgs<'_>>,
    lora_args: Option<metrale_model_engine::factory::LoraBuildArgs<'_>>,
    nllb_lang: Option<(u32, u32)>,
    nllb_lora_dir: Option<std::path::PathBuf>,
) -> Result<Box<dyn metrale_model_engine::traits::Model>> {
    // 2026-09-26: `marconi_min_tokens` is a process-wide `OnceLock` fixed by its
    // first read or write, so it is set here, before the model exists. A
    // `false` return means it was already fixed and the flag had no effect.
    if !metrale_model_layers::set_marconi_min_tokens(args.marconi_min_tokens) {
        tracing::warn!(
            "--marconi-min-tokens={} was NOT applied: the threshold had already \
             been read and is fixed for this process. The serve is running with \
             the earlier value, and any record it writes would misstate its \
             configuration.",
            args.marconi_min_tokens,
        );
    }

    let mtp_quant: metrale_model_layers::layers::MtpQuantization = args
        .mtp_quantization
        .parse()
        .context("Invalid --mtp-quantization value")?;
    metrale_model_engine::factory::build_model(
        config.clone(),
        store,
        gpu,
        max_batch_tokens,
        args.block_size,
        args.max_seq_len,
        args.max_batch_size,
        mtp_quant,
        args.speculative || args.dflash,
        prefix_cache,
        args.mtp_vocab,
        comm,
        args.self_speculative || args.ngram_speculative,
        if args.dflash {
            // 2026-09-26: The drafter head is not built yet: `--dflash-gamma`
            // (else 16) minus one. `serve_load` takes the scheduler's
            // `num_drafts` from the built head's gamma.
            args.resolved_dflash_gamma(None).saturating_sub(1).max(1)
        } else {
            args.resolved_num_drafts()
        },
        kv_dtype,
        inference_reserve,
        args.gpu_memory_utilization,
        args.ssm_cache_slots,
        layer_dtypes,
        args.ssm_checkpoint_interval,
        hss_cache_blocks_per_seq,
        dflash_args,
        lora_args,
        nllb_lang,
        nllb_lora_dir,
    )
    .context("Failed to build model")
}

pub(crate) fn build_high_speed_swap_config(
    args: &cli::ServeArgs,
) -> Result<Option<metrale_storage::HighSpeedSwapConfig>> {
    if !args.high_speed_swap {
        return Ok(None);
    }
    let dir = args
        .high_speed_swap_dir
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from("/var/tmp/metrale-hsw"));
    let bytes_gb = args.high_speed_swap_gb.unwrap_or(64);
    let resident_blocks = args.high_speed_swap_resident_blocks.unwrap_or(8192);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        anyhow::bail!(
            "--high-speed-swap: failed to create dir {}: {e}",
            dir.display()
        );
    }
    let cfg = metrale_storage::HighSpeedSwapConfig {
        dir,
        bytes: bytes_gb * (1 << 30),
        resident_blocks,
        rank: args.high_speed_swap_rank,
        qd: args.high_speed_swap_qd,
        graph: !args.no_high_speed_swap_graph,
        projection_seed: 0xCAFE_F00D,
    };
    cfg.validate()?;
    Ok(Some(cfg))
}

pub(crate) fn validate_head_high_speed_swap(
    args: &cli::ServeArgs,
    early_high_speed_swap_cfg: &Option<metrale_storage::HighSpeedSwapConfig>,
    swap_space_gb: usize,
) -> Result<Option<metrale_storage::HighSpeedSwapConfig>> {
    let Some(cfg) = early_high_speed_swap_cfg.as_ref() else {
        return Ok(None);
    };
    if swap_space_gb > 0
        && cfg.dir.canonicalize().ok().as_deref()
            == std::path::Path::new("/tmp/metrale-swap")
                .canonicalize()
                .ok()
                .as_deref()
    {
        let _ = args;
        anyhow::bail!(
            "--high-speed-swap-dir must not be /tmp/metrale-swap (already used \
             by --swap-space-gb sequence-level fallback)"
        );
    }
    tracing::info!(
        "--high-speed-swap enabled: dir={}, budget={} GiB, scratch={} blocks, \
         rank={}, qd={}, graph={}",
        cfg.dir.display(),
        cfg.bytes / (1 << 30),
        cfg.resident_blocks,
        cfg.rank,
        cfg.qd,
        cfg.graph,
    );
    Ok(Some(cfg.clone()))
}

pub(crate) fn maybe_run_ep_worker(
    args: &cli::ServeArgs,
    model: &mut Option<Box<dyn metrale_model_engine::traits::Model>>,
    early_high_speed_swap_cfg: &Option<metrale_storage::HighSpeedSwapConfig>,
) -> Result<bool> {
    if args.rank == 0 {
        return Ok(false);
    }
    let rank = args.rank;
    let mut model_owned = model.take().expect("EP worker requires owned model");
    let model_has_proposer = model_owned.has_proposer();
    // 2026-09-26: `--dflash` counts as a speculative flag for this check.
    let worker_spec =
        args.speculative || args.self_speculative || args.ngram_speculative || args.dflash;
    if !worker_spec && model_has_proposer {
        let override_set = matches!(
            std::env::var("METRALE_ALLOW_SPEC_MISMATCH").as_deref(),
            Ok("1") | Ok("true")
        );
        if !override_set {
            anyhow::bail!(
                "EP worker (rank {rank}) started WITHOUT any --speculative flag, \
                 but this checkpoint has MTP weights and the head will likely use them. \
                 Mirror the head's --speculative / --mtp-quantization / --num-drafts \
                 flags here, or set METRALE_ALLOW_SPEC_MISMATCH=1 if the head is also \
                 non-speculative."
            );
        }
        tracing::warn!(
            "EP worker (rank {rank}) running WITHOUT speculative flags but \
             METRALE_ALLOW_SPEC_MISMATCH=1 — head must NOT issue MTP commands."
        );
    } else if !model_has_proposer && !worker_spec {
        tracing::info!(
            "EP worker (rank {rank}): checkpoint has no MTP weights; \
             spec-mismatch guard auto-skipped (head can't use MTP either)."
        );
    }
    let worker_hss_cfg = early_high_speed_swap_cfg.clone();
    // 2026-09-26: Copied out of `args`: the `'static` worker thread cannot
    // borrow it.
    let max_batch_size = args.max_batch_size;
    let handle = std::thread::spawn(move || {
        model_owned
            .bind_gpu_to_thread()
            .expect("Failed to bind GPU to EP worker thread");
        if let Some(cfg) = worker_hss_cfg {
            match model_owned.high_speed_swap_dims() {
                Some(dims) => {
                    if let Err(e) = metrale_storage::install_local(rank as u64, cfg, dims) {
                        tracing::error!(
                            "EP worker (rank {rank}): --high-speed-swap install failed: {e:#}"
                        );
                    } else {
                        tracing::info!(
                            "EP worker (rank {rank}): --high-speed-swap orchestrator installed"
                        );
                    }
                }
                None => {
                    tracing::warn!(
                        "EP worker (rank {rank}): --high-speed-swap requested but model \
                         does not expose high_speed_swap_dims; skipping install"
                    );
                }
            }
        }
        // 2026-09-26: Every slot is allocated up front, in index order. The
        // SSM pool pops from `(0..max_slots).rev()`, so on both ranks the i-th
        // allocation gets slot i and `slots[i]` matches the head's i-th
        // `alloc_sequence`.
        let mut slots: Vec<Option<metrale_model_engine::traits::SequenceState>> =
            (0..max_batch_size).map(|_| None).collect();
        // 2026-10-11: `METRALE_EP_WORKER_LAZY_SLOTS=1`: reserve each slot's index in the same
        // order and leave it empty; its per-layer states are allocated when a command first
        // addresses it (`ep_worker_materialize`). The free list ends up empty either way.
        let lazy = model_owned.ep_worker_lazy_slots();
        for (i, slot) in slots.iter_mut().enumerate() {
            if lazy {
                let got = model_owned
                    .ep_worker_reserve_slot()
                    .expect("Failed to reserve EP worker slot");
                assert_eq!(got, i, "EP worker slot reservation out of order");
            } else {
                *slot = Some(
                    model_owned
                        .alloc_sequence()
                        .expect("Failed to allocate EP worker sequence"),
                );
            }
        }
        tracing::info!(
            "EP worker ready (rank {rank}, {} slots{}), waiting for commands",
            slots.len(),
            if lazy {
                ", METRALE_EP_WORKER_LAZY_SLOTS: ENGAGED, slot states allocated on first use"
            } else {
                ""
            }
        );
        loop {
            match model_owned.ep_worker_step(&mut slots) {
                Ok(true) => {}
                Ok(false) => break,
                // 2026-09-26: A command that was received and then failed
                // (`EpCommandFailed`) concerns one request: the head raises the
                // same error for it, so the worker logs it and keeps running.
                Err(e)
                    if e.downcast_ref::<metrale_model_engine::traits::EpCommandFailed>()
                        .is_some() =>
                {
                    tracing::error!(
                        "EP worker command failed (rank {rank}); worker STAYS UP: {e:#}"
                    );
                }
                // 2026-09-26: Any other error came from the receive: the link to
                // the head is gone.
                Err(e) => {
                    tracing::error!("EP worker error: {e:#}");
                    break;
                }
            }
        }
        for slot in slots.iter_mut() {
            if let Some(mut seq) = slot.take() {
                let _ = model_owned.free_sequence(&mut seq);
            }
        }
        // 2026-09-26: Synchronise the default stream before teardown releases
        // the pools; a failure is logged and teardown still runs.
        if let Err(error) = model_owned.synchronize(model_owned.default_stream()) {
            tracing::error!("EP worker stream quiescence failed (rank {rank}): {error:#}");
        }
        if let Err(error) = model_owned.teardown() {
            tracing::error!("EP worker teardown failed (rank {rank}): {error:#}");
        }
        tracing::info!("EP worker stopped (rank {rank})");
    });
    handle.join().expect("EP worker thread panicked");
    Ok(true)
}

#[cfg(test)]
mod prefix_cache_tests {
    use clap::Parser;
    use metrale_config::ModelConfig;

    use super::build_prefix_cache;
    use crate::cli::ServeArgs;

    fn enabled_args() -> ServeArgs {
        ServeArgs::parse_from(["met", "--enable-prefix-caching"])
    }

    #[test]
    fn safe_model_keeps_requested_prefix_cache() {
        let cache = build_prefix_cache(&enabled_args(), &ModelConfig::qwen3_next_80b_nvfp4());
        assert!(cache.is_active());
    }

    /// 2026-09-26: For a model whose capability predicate is true, the flag
    /// alone decides: without `--enable-prefix-caching` the cache is
    /// `NoPrefixCaching`.
    #[test]
    fn an_open_predicate_without_the_flag_still_installs_no_prefix_caching() {
        let args = ServeArgs::parse_from(["met"]);
        assert!(
            !args.prefix_caching_enabled(),
            "clap default must stay false"
        );

        let config = ModelConfig::qwen3_next_80b_nvfp4();
        assert!(
            config.kv_only_prefix_cache_is_safe(),
            "this model's predicate is the open case the flag has to gate"
        );
        assert!(!build_prefix_cache(&args, &config).is_active());
    }

    #[test]
    fn compressed_deepseek_v4_disables_incomplete_prefix_cache() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "deepseek_v4".to_string();
        config.compress_ratios = vec![0, 4, 128];

        let cache = build_prefix_cache(&enabled_args(), &config);
        assert!(!cache.is_active());
    }

    #[test]
    fn glm5_next_disables_incomplete_prefix_cache() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".to_string();

        let cache = build_prefix_cache(&enabled_args(), &config);
        assert!(!cache.is_active());
    }
}

#[cfg(test)]
mod swap_space_tests {
    use clap::Parser;
    use metrale_config::ModelConfig;

    use super::resolve_swap_space_gb;
    use crate::cli::ServeArgs;

    fn args_with(swap_gb: &str) -> ServeArgs {
        ServeArgs::parse_from(["met", "--swap-space-gb", swap_gb])
    }

    #[test]
    fn a_kv_complete_model_keeps_the_requested_swap_space() {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        assert_eq!(resolve_swap_space_gb(&args_with("3"), &config), 3);
    }

    /// 2026-09-26: `--swap-space-gb` defaults to a non-zero value, so a model
    /// with state outside KV hits the gate without any swap flag.
    #[test]
    fn the_default_swap_space_is_nonzero_so_the_gate_has_work_to_do() {
        assert!(ServeArgs::parse_from(["met"]).swap_space_gb > 0);
    }

    #[test]
    fn a_model_with_state_outside_kv_gets_zero() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();

        for model_type in ["glm5_next", "glm5_next_text"] {
            config.model_type = model_type.to_string();
            assert_eq!(
                resolve_swap_space_gb(&ServeArgs::parse_from(["met"]), &config),
                0
            );
            assert_eq!(resolve_swap_space_gb(&args_with("64"), &config), 0);
        }

        config.model_type = "deepseek_v4".to_string();
        config.compress_ratios = vec![0, 4, 128];
        assert_eq!(resolve_swap_space_gb(&args_with("64"), &config), 0);
    }

    #[test]
    fn an_explicit_zero_stays_zero_for_every_model() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        assert_eq!(resolve_swap_space_gb(&args_with("0"), &config), 0);
        config.model_type = "glm5_next".to_string();
        assert_eq!(resolve_swap_space_gb(&args_with("0"), &config), 0);
    }
}

#[cfg(test)]
mod ep_worker_loop_tests {
    /// 2026-09-26: The worker loop keeps running after an `EpCommandFailed` and
    /// exits on any other error. Asserted on the source: the `EpCommandFailed`
    /// arm comes before the catch-all `break`.
    #[test]
    fn a_command_failure_keeps_the_worker_up_and_a_link_failure_does_not() {
        let src = include_str!("build.rs");
        let loop_body = src
            .split_once("match model_owned.ep_worker_step(&mut slots)")
            .expect("the EP worker loop must exist")
            .1;
        let recoverable = loop_body
            .find("EpCommandFailed")
            .expect("the loop must classify command failures");
        let stays_up = loop_body
            .find("worker STAYS UP")
            .expect("the recoverable arm must say so in the log");
        let fatal = loop_body
            .find("break;\n                }\n            }\n        }")
            .expect("the fatal arm must still break");
        assert!(
            recoverable < stays_up && stays_up < fatal,
            "the EpCommandFailed arm must come BEFORE the catch-all break, or every command \
             failure is fatal again"
        );
    }
}
