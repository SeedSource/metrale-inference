// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The model-dependent half of startup: resolve, configure, load
//! and build one checkpoint, then spawn its scheduler. A model swap runs it
//! again (`model_swap::swap`).
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - The `progress::phase` indices here are 1 to 9 and positional: the TUI
//!   ignores the name and enters `PHASE_NAMES[index]`.
//! - The response store, conversation store and rate limiter come in through
//!   `Carried` and are moved into `AppState` unchanged; nothing here builds
//!   them.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;

use super::serve::Prepared;
use crate::api::InferenceRequest;
use crate::main_modules::AppState;
use crate::main_modules::serve_phases;

use crate::{cli, scheduler, session_manager};

mod adapters;
mod carried;
mod load_phases;
mod model_setup;
mod scheduler_setup;

pub(crate) use carried::Carried;

/// 2026-09-26: Load a model and build everything derived from it. `Ok(None)`
/// means this rank is an EP worker: it ran its command loop and has nothing
/// for the async tail to serve.
pub(crate) fn load_model(
    mut args: cli::ServeArgs,
    tui_handles_tx: Option<std::sync::mpsc::Sender<crate::tui::RunHandles>>,
    carried: Carried,
) -> Result<Option<Prepared>> {
    metrale_telemetry::progress::phase(1, "model resolve");
    let model_dir = serve_phases::resolve_model_dir(&args)?;

    tracing::info!("Port: {}", args.port);

    tracing::info!(
        "SSM decode h-state dtype: {} (--ssm-h-dtype)",
        if metrale_model_layers::layers::qwen3_ssm::ssm_h_f16_pool_enabled() {
            "f16 + f16-sized pools (stage 3)"
        } else if metrale_model_layers::layers::qwen3_ssm::ssm_h_fp16_enabled() {
            "f16"
        } else {
            "f32 (full precision)"
        }
    );

    let (mut config, config_json) = model_setup::configure_model(&args, &model_dir)?;

    let (vision_max_pixels, remote_image_policy, video_ffmpeg) =
        model_setup::resolve_media_policies(&args, &model_dir, &mut config)?;

    model_setup::log_model_config(&config);

    let ptx_set = model_setup::select_kernel_target(&args, &config, &model_dir)?;
    let sampling_presets = ptx_set.sampling;
    model_setup::check_kernel_target(&ptx_set, &mut config)?;
    model_setup::publish_row_tiers(&args, &config);

    // 2026-09-26: After this call `args.num_drafts` is `Some`, so
    // `args.resolved_num_drafts()` is valid.
    serve_phases::apply_model_default_num_drafts(&mut args, &ptx_set);

    let (gpu, free_mem) = serve_phases::init_gpu_backend(&args, &ptx_set)?;

    // 2026-09-26: Topology runs before `preflight_reserve`: `resolve_topology`
    // divides the attention and linear-attention head counts by `tp_size`
    // (with a divisibility check), and the reserve is sized from those fields.
    // At `--tp-size 1` it divides nothing.
    metrale_telemetry::progress::phase(4, "topology");
    let serve_phases::Topology {
        world_size,
        tp_size: _tp_size,
        ep_size,
        tp_rank: _tp_rank,
        ep_rank,
    } = serve_phases::resolve_topology(&args, &mut config)?;

    let serve_phases::ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
    } = load_phases::reserve_preflight(
        &args,
        &config,
        gpu.as_ref(),
        free_mem,
        &model_dir,
        &ptx_set,
    )?;
    let total_reserve = inference_reserve + buffer_arena_bytes;

    // 2026-09-26: OOM watchdog (CUDA builds): every 2 s it reads free device
    // memory and, after three consecutive readings below 2048 MiB, exits the
    // process with status 1. Spawned once per process.
    #[cfg(feature = "cuda")]
    let _oom_watchdog = metrale_gpu_runtime::cuda_backend::spawn_oom_watchdog(
        2048,
        std::time::Duration::from_secs(2),
    );
    #[cfg(feature = "cuda")]
    tracing::info!("OOM watchdog started (threshold: 2 GB, interval: 2s)");

    // 2026-09-26: An explicit `--fp8-kv-calibration-tokens` wins, including 0,
    // which turns calibration off for a model whose MODEL.toml enables it.
    // Omitted: MODEL.toml `[behavior].fp8_kv_calibration_tokens`, 0 when absent.
    config.fp8_kv_calibration_tokens = args
        .fp8_kv_calibration_tokens
        .unwrap_or(ptx_set.behavior.fp8_kv_calibration_tokens);
    // 2026-09-26: Set on every load: config parsing leaves it at 0.0
    // (`#[serde(skip)]`), and `validate_serve_args` has checked the flag is at
    // least 1.0.
    config.fp8_kv_headroom = args.fp8_kv_headroom;

    let store = load_phases::load_weights(
        &args,
        &mut config,
        &model_dir,
        gpu.as_ref(),
        ep_rank,
        ep_size,
        &ptx_set,
        free_mem,
        inference_reserve,
        total_reserve,
        gdn_two_phase_bytes,
        max_batch_tokens_pre,
    )?;

    metrale_telemetry::progress::phase(6, "kv cache");
    let serve_phases::PrefillBudget {
        prefill_budget,
        max_batch_tokens,
        spec_tokens: _spec_tokens,
    } = serve_phases::resolve_prefill_budget(&args, ssm_prefill_chunk);
    let prefix_cache = serve_phases::build_prefix_cache(&args, &config);
    let comm = serve_phases::init_nccl_comm(
        &args,
        gpu.as_ref(),
        world_size,
        max_batch_tokens,
        config.hidden_size,
        config.vocab_size,
    )?;
    config.profile = args.profile;
    serve_phases::cap_vocab_size_to_tokenizer(&model_dir, &mut config);
    let serve_phases::KvCacheConfig {
        effective_kv_dtype_str: _,
        kv_dtype,
        layer_dtypes,
        hss_cache_blocks_per_seq,
    } = serve_phases::resolve_kv_cache_config(
        &args,
        &config,
        ptx_set.behavior.default_kv_dtype,
        store.fp8_kv_scale_count(),
    )?;

    load_phases::validate_kv_kernels(gpu.as_ref(), kv_dtype, &layer_dtypes, &config)?;
    let dflash_drafter_state =
        serve_phases::load_dflash_drafter(&args, &config, &ptx_set, gpu.as_ref())?;
    // 2026-09-26: LoRA adapters load before `gpu` moves into `build_model`.
    // `lora_states` outlives it: `lora_args` borrows each `store`, and
    // `AppState` takes the adapter names. NLLB loads its adapter through its
    // own path (`nllb_lora_dir` below).
    let is_nllb = matches!(config.model_type.as_str(), "m2m_100" | "nllb");
    let lora_states = if is_nllb {
        Vec::new()
    } else {
        serve_phases::load_lora_adapters(&args, gpu.as_ref())?
    };
    if !lora_states.is_empty() && world_size > 1 {
        anyhow::bail!(
            "--lora-adapter requires world_size=1 in v0 (got {world_size}); \
             TP adapter sharding is M3"
        );
    }
    let lora_args = adapters::lora_build_args(&args, &lora_states);
    let dflash_args = adapters::dflash_build_args(&args, &dflash_drafter_state);
    let nllb_lang = adapters::resolve_nllb_lang(&args, &config, &model_dir)?;
    let (nllb_lora_dir, nllb_adapter_name) = adapters::resolve_nllb_adapter(&args, is_nllb)?;
    let model = serve_phases::build_model(
        &args,
        &config,
        // 2026-09-26: Moved: the model owns the weight store from here.
        store,
        gpu,
        max_batch_tokens,
        kv_dtype,
        inference_reserve,
        layer_dtypes,
        hss_cache_blocks_per_seq,
        prefix_cache,
        comm,
        dflash_args,
        lora_args,
        nllb_lang,
        nllb_lora_dir,
    )?;

    // 2026-09-26: Kernel load audit and the boot gate (`kernel_gate`). Under
    // `--check-kernels` this call does not return: it prints the report and
    // exits with the unresolved count as the status.
    metrale_telemetry::progress::phase(7, "kernel audit");
    serve_phases::audit_and_gate(&args, &ptx_set)?;

    // 2026-09-26: Built before `maybe_run_ep_worker`, which installs it on
    // worker ranks.
    let early_high_speed_swap_cfg = serve_phases::build_high_speed_swap_config(&args)?;

    let mut model_opt = Some(model);
    if serve_phases::maybe_run_ep_worker(&args, &mut model_opt, &early_high_speed_swap_cfg)? {
        // 2026-09-26: An EP worker rank ran its command loop and the head has
        // exited: nothing to serve.
        return Ok(None);
    }
    let model = model_opt.expect("head retains model on rank 0");

    // 2026-09-26: EOS ids from generation_config.json, else from the config.
    let mut eos_tokens = serve_phases::load_eos_tokens(&model_dir, &config);

    let serve_phases::SamplingDefaults {
        temperature: default_temperature,
        top_k: default_top_k,
        top_p: default_top_p,
        top_n_sigma: default_top_n_sigma,
        min_p: default_min_p,
    } = serve_phases::load_sampling_defaults(&model_dir, &args, &sampling_presets.non_thinking);

    let (tokenizer, supports_thinking) =
        model_setup::load_tokenizer(&args, &config, &model_dir, &eos_tokens)?;

    let serve_phases::TokenizerRuntime {
        vocab_masks,
        limits: tokenizer_limits,
        reasoning_parser_box,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        grammar_engine,
    } = serve_phases::resolve_tokenizer_runtime(
        &args,
        &mut config,
        &tokenizer,
        &mut eos_tokens,
        supports_thinking,
        &model_dir,
    );

    metrale_telemetry::progress::phase(9, "scheduler");
    let (request_tx, request_rx) = mpsc::channel::<InferenceRequest>(args.max_num_seqs);
    // 2026-09-26: Control channel for `POST /v1/lora/active`.
    let (rotation_tx, rotation_rx) = mpsc::channel::<scheduler::LoraRotation>(8);

    let model_name = serve_phases::resolve_model_name(&args, &config_json, &model_dir);

    let scheduler_model = model;
    let scheduler_eos = eos_tokens;
    let max_batch_size =
        scheduler_setup::resolve_max_batch_size(&args, world_size, scheduler_model.as_ref())?;
    let (use_speculative, use_self_spec, use_ngram_spec, num_drafts, dflash_rung) =
        scheduler_setup::resolve_speculation(&args, scheduler_model.as_ref());

    let policy = scheduler_setup::scheduling_policy(&args)?;

    let max_prefill_tokens = prefill_budget;
    let swap_space_gb = serve_phases::resolve_swap_space_gb(&args, &config);
    let block_size = args.block_size;

    let high_speed_swap_cfg = serve_phases::validate_head_high_speed_swap(
        &args,
        &early_high_speed_swap_cfg,
        swap_space_gb,
    )?;

    let adaptive_sampling = args.adaptive_sampling;
    let session_manager = session_manager::SessionSsmManager::new(600);
    // 2026-09-26: The thinking budget a sequence gets when the model opens
    // `<think>` on its own (`emit_step/token.rs`): `--max-thinking-budget`,
    // else MODEL.toml `[behavior].max_thinking_budget`.
    let scheduler_spontaneous_think_budget = args
        .max_thinking_budget
        .unwrap_or(ptx_set.behavior.max_thinking_budget);
    // 2026-09-26: With `--dflash` the verify steps pick on raw argmax, without
    // the pre-sample pipeline. The `dflash_masked_verify` lever
    // (`METRALE_DFLASH_MASKED_VERIFY`, on unless `0`) sends the picks back
    // through the masking at the pick sites; it does not change this bool.
    let dflash_verify_raw_argmax = args.dflash;
    // 2026-09-29: A146: MTP-lane speculation inside `<think>`: default on only
    // for architectures that passed the spec-in-think quality and safety
    // gates (GLM-5.3); resolved from the model config once, here, so the
    // scheduler never string-matches a model. The env overrides it (see
    // `SchedLevers::mtp_spec_think`).
    let mtp_spec_think_default =
        metrale_speculative::mtp_gate::mtp_spec_think_default(&config.model_type);
    let watchdog_params = crate::scheduler::WatchdogParams::from_behavior(
        &ptx_set.behavior,
        args.max_inter_tool_prose,
        args.content_loop_min_repeats,
    );
    // 2026-09-26: The run's levers, shared with the dashboard so `/watchdog
    // on|off` toggles this run's flag. Its starting value is
    // `--content-loop-watchdog`, else `METRALE_CONTENT_LOOP_WATCHDOG`, else
    // MODEL.toml `[behavior].enable_loop_watchdog`.
    let sched_levers = std::sync::Arc::new(crate::scheduler::levers::SchedLevers::from_env(
        args.mtp_gate_force(),
    ));
    sched_levers.set_loop_watchdog(crate::scheduler::resolve_content_loop_watchdog(
        ptx_set.behavior.enable_loop_watchdog,
        std::env::var("METRALE_CONTENT_LOOP_WATCHDOG")
            .ok()
            .as_deref(),
        crate::cli::flag_values::Tristate::validated(&args.content_loop_watchdog).pinned(),
    ));
    // 2026-09-26: The run's snapshot cell, shared with the dashboard the same
    // way.
    let sched_snapshot =
        std::sync::Arc::new(metrale_speculative::snapshot::SnapshotCell::default());
    if let Some(tx) = &tui_handles_tx {
        let _ = tx.send(crate::tui::RunHandles {
            levers: sched_levers.clone(),
            snapshot: sched_snapshot.clone(),
        });
    }
    let run_levers = sched_levers.clone();
    let run_snapshot = sched_snapshot.clone();
    let sched_limits = crate::scheduler::limits::SchedLimits {
        max_seq_len: args.max_seq_len,
        ..tokenizer_limits
    };
    // 2026-09-26: Capture the tokio runtime handle so the scheduler thread can
    // send terminal stream events (Done / Error) as tokio tasks.
    scheduler::capture_runtime_handle();
    // 2026-09-26: The scheduler thread's handle is kept (`Prepared::scheduler`):
    // a swap joins it once the drain has closed `request_tx`, and only then
    // tears the model down. `--scheduler-config async` needs a model with a
    // device token feed; without one it falls back to `sync` with a warning.
    let scheduler_model: std::sync::Arc<dyn metrale_model_engine::traits::Model> =
        std::sync::Arc::from(scheduler_model);
    let scheduler_dev = scheduler_setup::scheduler_device(&args, scheduler_model, max_batch_size);
    let scheduler_handle = std::thread::spawn(move || {
        scheduler::run(
            scheduler_dev,
            request_rx,
            rotation_rx,
            scheduler::config::SchedulerConfig {
                eos_tokens: scheduler_eos,
                max_batch_size,
                use_speculative,
                dflash_verify_raw_argmax,
                mtp_spec_think_default,
                num_drafts,
                policy,
                max_prefill_tokens,
                max_batch_tokens,
                use_self_speculative: use_self_spec,
                use_ngram_speculative: use_ngram_spec,
                swap_space_gb,
                high_speed_swap_cfg,
                block_size,
                think_end_token,
                think_start_token,
                code_fence_token,
                tool_call_start_token,
                tool_call_end_token,
                grammar_engine,
                adaptive_sampling,
                session_manager,
                spontaneous_think_budget: scheduler_spontaneous_think_budget,
                vocab_masks,
                limits: sched_limits,
                watchdog: watchdog_params,
                levers: run_levers,
                snapshot: run_snapshot,
                telemetry: metrale_telemetry::global(),
                pipeline_faults: scheduler::PipelineFaults::NONE,
                dflash_rung,
            },
        );
    });

    // 2026-09-26: Tool call parser: `--tool-call-parser`, then MODEL.toml, then
    // `tool_defaults.toml`.
    let tool_call_parser = serve_phases::resolve_tool_call_parser(&args, &ptx_set, &config)?;

    // 2026-09-26: Carried, never rebuilt; see `Carried`.
    let Carried {
        response_store,
        rate_limiter,
        conversation_store,
    } = carried;
    serve_phases::log_response_store_audit(&response_store, &rate_limiter);
    let dump_writer = serve_phases::open_dump_writer(&args);
    let adapters::LoraPromotion {
        lora_peer_addr,
        lora_stageable,
        lora_disk_stageable,
        promotion,
    } = adapters::build_lora_promotion(&args, &lora_states)?;

    let default_kwargs = model_setup::parse_default_kwargs(&args)?;

    let state = Arc::new(AppState {
        tokenizer,
        model_name,
        adapter_name: nllb_adapter_name
            .clone()
            .or_else(|| lora_states.first().map(|l| l.name.clone())),
        adapter_names: if let Some(ref n) = nllb_adapter_name {
            vec![n.clone()]
        } else {
            lora_states.iter().map(|l| l.name.clone()).collect()
        },
        active_adapter: std::sync::Arc::new(std::sync::Mutex::new(
            lora_states.first().map(|l| l.name.clone()),
        )),
        max_seq_len: args.max_seq_len,
        request_tx,
        rotation_tx: if lora_states.is_empty() {
            None
        } else {
            Some(rotation_tx)
        },
        chat: crate::api::chat::levers::ChatLevers::resolve(
            ptx_set.behavior.tscg,
            ptx_set.behavior.disable_cwd_hint_injection,
        ),
        vision_config: config.vision.clone(),
        vision_max_pixels,
        remote_image_policy,
        video_ffmpeg,
        video_fps: args.video_fps,
        default_temperature,
        default_top_k,
        default_top_p,
        default_top_n_sigma,
        default_min_p,
        tool_call_parser,
        reasoning_parser: reasoning_parser_box,
        think_end_token_id: think_end_token,
        think_start_token_id: think_start_token,
        tool_max_tokens: args.tool_max_tokens,
        sampling_presets,
        tool_call_start_token_id: tool_call_start_token,
        auto_compact_threshold: args.auto_compact,
        request_timeout: args.request_timeout,
        effective_context: 0,
        // 2026-09-26: `behavior` is MODEL.toml's, embedded at build time, with
        // the CLI overrides below.
        behavior: model_setup::resolve_behavior(&ptx_set, &args, &default_kwargs),
        disable_thinking: args.disable_thinking,
        default_thinking: default_kwargs.thinking,
        default_reasoning_effort: default_kwargs.reasoning_effort,
        response_store,
        rate_limiter,
        conversation_store,
        dump_writer,
        lora_stageable,
        lora_peer_addr,
        promotion,
        promoted_slots: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
        lora_disk_stageable,
    });

    serve_phases::log_behavior_audit(&args, &ptx_set);

    // 2026-09-26: Phases 10 and 11 (router, listening) run on the async side
    // (`serve_router`).
    Ok(Some(Prepared {
        state,
        bind: args.scheduling.service.bind,
        port: args.port,
        scheduler: scheduler_handle,
    }))
}

#[cfg(test)]
#[path = "serve_load_tests.rs"]
mod tests;
