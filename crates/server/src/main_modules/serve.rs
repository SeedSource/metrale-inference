// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `met serve`: install the process-scoped state on a `ModelHost`,
//! run startup on the blocking pool, then serve the loaded model or, with no
//! model named, bind the listener and wait for one.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - The auth policy and the process-scoped stores are installed on the host
//!   before startup runs and before the listener binds.

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::cli;
use crate::main_modules::AppState;

/// 2026-09-26: What the blocking startup hands to the async tail.
pub(crate) struct Prepared {
    pub state: Arc<AppState>,
    pub bind: String,
    pub port: u16,
    /// 2026-09-26: The scheduler thread. A swap joins it after the drain, and
    /// only then loads the next model.
    pub scheduler: std::thread::JoinHandle<()>,
}

/// 2026-09-26: How startup ended.
enum Startup {
    /// 2026-09-26: A model is loaded; serve it.
    Serve(Prepared),
    /// 2026-09-26: EP worker rank: its command loop has returned; no router.
    Worker,
    /// 2026-09-26: No model was named and a dashboard is running: the
    /// listener still binds, and a model can be chosen from the Library.
    AwaitingModel,
}

/// 2026-09-26: Bring the engine up, then serve.
///
/// Startup has no `await` in it, so it runs on the blocking pool
/// (`spawn_blocking`) and is awaited here. That await yields, which keeps the
/// async workers free during a load and lets `main` race this future against
/// a shutdown.
pub(crate) async fn serve(
    args: cli::ServeArgs,
    tui_progress: Option<std::sync::mpsc::Receiver<crate::tui::capture_layer::ProgressEvent>>,
) -> Result<()> {
    // 2026-09-26: One host for the process lifetime, created before startup so
    // the dashboard (`tui::start`) can hold it. The load publishes into it.
    let host = Arc::new(crate::main_modules::model_host::ModelHost::empty());
    // 2026-09-26: Recorded before `args` moves into startup: a swap whose load
    // fails restores this argv.
    host.set_args(args.clone());
    // 2026-09-26: Installed in every process that serves, so the
    // scheduler-equivalence gate can re-serve this checkpoint under each
    // device router.
    metrale_bench::benchmarks::scheduler_equivalence::host::install(Arc::new(
        crate::tui::bench_host::TuiServeHost::new(host.clone(), args.cache_dir.clone()),
    ));
    // 2026-09-26: The listener binds once for the process lifetime, so a model
    // chosen later serves on this address whatever its recipe says (see the
    // port warning in `model_swap::swap`).
    let (bind_addr, bind_port) = (args.bind.clone(), args.port);
    // 2026-09-26: Before any load, so the policy is in force from the moment
    // the listener is up, also while no model is loaded.
    host.set_auth(build_auth_config(&args)?);
    // 2026-09-26: Process-scoped too, and installed before the first model.
    host.set_process(super::serve_load::Carried::from_env().map_err(|e| anyhow::anyhow!("{e}"))?);
    let startup_host = host.clone();
    match tokio::task::spawn_blocking(move || startup(args, tui_progress, startup_host)).await?? {
        Startup::Serve(prepared) => {
            // 2026-10-08: Before the publish and the bind, so the readiness
            // line follows it and no clone of the state outlives it.
            super::serve_warmup::run_if_enabled(&prepared.state).await;
            host.publish(prepared.state);
            // 2026-09-26: The host owns the first load's scheduler thread too,
            // so the first swap can join it before tearing the model down.
            host.set_scheduler(prepared.scheduler);
            crate::main_modules::serve_router::build_and_serve(host, &prepared.bind, prepared.port)
                .await
        }
        Startup::Worker => Ok(()),
        // 2026-09-26: Nothing to serve yet, but the listener still binds, so a
        // model chosen from the Library later has a port. Until one is
        // published, routes that need a model answer 503 `model_not_loaded`.
        Startup::AwaitingModel => {
            crate::main_modules::serve_router::build_and_serve(host, &bind_addr, bind_port).await
        }
    }
}

fn startup(
    args: cli::ServeArgs,
    tui_progress: Option<std::sync::mpsc::Receiver<crate::tui::capture_layer::ProgressEvent>>,
    host: Arc<crate::main_modules::model_host::ModelHost>,
) -> Result<Startup> {
    tracing::info!("Metrale Engine starting...");
    tracing::info!("Licensed under MIT OR Apache-2.0 — see /LICENSE-MIT and /LICENSE-APACHE");
    crate::disk_guard::warn_if_nearly_full(args.cache_dir.as_deref());
    metrale_telemetry::progress::phase(0, "banner");

    // 2026-09-26: SIGINT and SIGTERM request a clean shutdown. In the TUI,
    // Ctrl+C arrives as a key event and calls the same `shutdown::request`.
    crate::tui::shutdown::install_signal_listeners();

    // 2026-09-26: Publishes each run's levers to the dashboard (see
    // `tui::start`). `None` in plain mode and on worker ranks.
    let mut tui_handles_tx: Option<std::sync::mpsc::Sender<crate::tui::RunHandles>> = None;

    // 2026-09-26: Started before the model load so the dashboard shows it.
    // Head rank only.
    if let Some(progress_rx) = tui_progress
        && args.rank == 0
    {
        let tx = crate::tui::start(args.clone(), progress_rx, host.clone());
        // 2026-09-26: Every later load sends its handles through this, so the
        // dashboard follows the model that is serving.
        host.set_tui_handles(tx.clone());
        tui_handles_tx = Some(tx);
    }

    // 2026-09-26: Reject contradictory flag combinations before the model
    // load, as a hard error.
    if let Err(msg) = cli::validate_serve_args(&args) {
        anyhow::bail!("{msg}");
    }

    // 2026-09-26: Publish the kernel-path flags before anything reads their
    // cells. An absent flag publishes nothing, so its `METRALE_*` variable
    // still decides (`serve_flags`).
    super::serve_flags::publish_kernel_flags(&args);
    // 2026-09-26: Process-scoped, like the flags: started before the first
    // load and kept across swaps.
    super::telemetry_boot::start(&args)?;

    // 2026-09-26: Everything above is process-scoped; everything below is
    // model-dependent, and a swap re-runs only that part (`load_model`).
    if args.model.is_none() && args.model_from_path.is_none() {
        // 2026-09-26: Without a dashboard there is nothing to choose a model
        // with, so a modelless boot is an error.
        if tui_handles_tx.is_none() {
            anyhow::bail!(
                "no model given, and no dashboard to choose one from.\n\
                 Pass a MODEL (or --model-from-path), or run on a TTY without \
                 --no-tui to browse the Library."
            );
        }
        tracing::info!("No model specified — open the Library to choose one");
        return Ok(Startup::AwaitingModel);
    }

    let carried = host
        .process()
        .expect("process-scoped state is installed before startup");
    match super::serve_load::load_model(args, tui_handles_tx, carried)? {
        Some(prepared) => Ok(Startup::Serve(prepared)),
        None => Ok(Startup::Worker),
    }
}

/// 2026-09-26: Parsed `--default-chat-template-kwargs`: the server-level
/// defaults a request falls back to.
#[derive(Debug, Default, PartialEq)]
pub(super) struct DefaultChatTemplateKwargs {
    /// 2026-09-26: Thinking directive; `Unspecified` when the flag sets none of
    /// the thinking keys and no `reasoning_effort`.
    pub thinking: crate::ir::ThinkingDirective,
    /// 2026-09-26: Template-side effort. `api/chat/prepare.rs` uses it when
    /// thinking is on and the request names none; with neither,
    /// `tokenizer/chat_render.rs` renders `"medium"`.
    pub reasoning_effort: Option<crate::ir::ReasoningEffort>,
    /// 2026-09-26: Server-level `preserve_thinking`: overrides the MODEL.toml
    /// `[behavior]` value; a request's own value overrides it.
    pub preserve_thinking: Option<bool>,
}

/// 2026-09-26: Parse `--default-chat-template-kwargs`
/// (`{"enable_thinking":bool,"thinking_budget":u32,
/// "reasoning_effort":str,"preserve_thinking":bool}`). The directive: a
/// positive `thinking_budget` turns thinking on with that budget and 0 turns
/// it off; else `enable_thinking`; else the effort's directive.
///
/// Invalid JSON, an unknown key or an unknown `reasoning_effort` value is an
/// error, which fails the boot.
pub(super) fn parse_default_chat_template_kwargs(
    s: &str,
) -> anyhow::Result<DefaultChatTemplateKwargs> {
    use crate::ir::ThinkingDirective;

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Kwargs {
        enable_thinking: Option<bool>,
        thinking_budget: Option<u32>,
        reasoning_effort: Option<String>,
        preserve_thinking: Option<bool>,
    }

    if s.trim().is_empty() {
        return Ok(DefaultChatTemplateKwargs::default());
    }
    let kw: Kwargs = serde_json::from_str(s)
        .map_err(|e| anyhow::anyhow!("--default-chat-template-kwargs is not valid JSON: {e}"))?;
    // 2026-09-26: `ir::parse_wire_effort` is the vocabulary a request's
    // `reasoning_effort` is parsed with too.
    let (reasoning_effort, effort_directive) = match kw.reasoning_effort.as_deref() {
        None => (None, None),
        Some(v) => match crate::ir::parse_wire_effort(v) {
            Some((template_effort, directive)) => (template_effort, Some(directive)),
            None => anyhow::bail!(
                "--default-chat-template-kwargs reasoning_effort {v:?}: expected one of \
                 none, minimal, low, medium, high, xhigh, max"
            ),
        },
    };
    let thinking = match (kw.thinking_budget, kw.enable_thinking) {
        (Some(b), _) if b > 0 => ThinkingDirective::On { budget: Some(b) },
        (Some(_), _) => ThinkingDirective::Off,
        (None, Some(true)) => ThinkingDirective::On { budget: None },
        (None, Some(false)) => ThinkingDirective::Off,
        // 2026-09-26: No thinking keys: an effort default carries the
        // directive `parse_wire_effort` gives it.
        (None, None) => effort_directive.unwrap_or(ThinkingDirective::Unspecified),
    };
    Ok(DefaultChatTemplateKwargs {
        thinking,
        reasoning_effort,
        preserve_thinking: kw.preserve_thinking,
    })
}

/// 2026-09-26: Resolve `--require-auth` / `--auth-tokens-file` /
/// `--auth-token` into an optional `AuthConfig`, failing at startup when
/// `--require-auth` has no token source.
pub(super) fn build_auth_config(
    args: &cli::ServeArgs,
) -> Result<Option<Arc<crate::auth::AuthConfig>>> {
    if !args.require_auth {
        if args.auth_tokens_file.is_some() || args.auth_token.is_some() {
            tracing::warn!(
                "--auth-tokens-file / --auth-token supplied without --require-auth; \
                 tokens are loaded but the auth gate is OFF. Pass --require-auth to enforce."
            );
        }
        return Ok(None);
    }
    let cfg = match (&args.auth_tokens_file, &args.auth_token) {
        (Some(path), None) => crate::auth::AuthConfig::from_file(path)?,
        (None, Some(tok)) => {
            tracing::warn!(
                "--auth-token sets the bearer token via the command line; the value \
                 is visible to other local users via `ps`/`/proc/<pid>/cmdline`. \
                 Use --auth-tokens-file with permissions 0600 in production."
            );
            crate::auth::AuthConfig::from_inline(tok)?
        }
        (None, None) => {
            return Err(anyhow::anyhow!(
                "--require-auth was set but neither --auth-tokens-file nor \
                 --auth-token was supplied. Pick one (a tokens file is preferred)."
            ));
        }
        (Some(_), Some(_)) => unreachable!("clap conflicts_with should have rejected this"),
    };
    tracing::info!(
        "auth: require_auth=ON ({} bearer token{} loaded)",
        cfg.token_count(),
        if cfg.token_count() == 1 { "" } else { "s" },
    );
    Ok(Some(Arc::new(cfg)))
}

/// 2026-09-26: The vision area bound, first match wins: `--vision-max-pixels`
/// above 0, then a positive `METRALE_VISION_MAX_PIXELS`, then the checkpoint's
/// processor config (`read_preprocessor_max_pixels`), else `None`.
///
/// # Errors
/// When `METRALE_VISION_MAX_PIXELS` is set to something other than blank, `0`
/// or an unsigned integer.
pub(super) fn resolve_vision_max_pixels(
    args: &cli::ServeArgs,
    model_dir: &std::path::Path,
) -> Result<Option<usize>> {
    if args.vision_max_pixels > 0 {
        return Ok(Some(args.vision_max_pixels));
    }
    if let Ok(raw) = std::env::var("METRALE_VISION_MAX_PIXELS") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() && trimmed != "0" {
            let parsed = trimmed.parse::<usize>().with_context(|| {
                format!("METRALE_VISION_MAX_PIXELS must be a positive integer, got {raw:?}")
            })?;
            if parsed > 0 {
                return Ok(Some(parsed));
            }
        }
    }
    Ok(read_preprocessor_max_pixels(model_dir))
}

/// 2026-09-26: The image area bound from the checkpoint's processor config,
/// or `None` when no source below has a positive one; an unreadable or
/// malformed file is skipped, not an error.
///
/// Read in order, first positive bound wins: the top level of
/// `preprocessor_config.json`, its `image_processor` object, then the
/// `image_processor` object of `processor_config.json`. In each,
/// `size.longest_edge` is read before `max_pixels`; both are pixel counts
/// (an area), whatever the name says. The image bound is addressed by key,
/// never by searching for the first `longest_edge`, because
/// `processor_config.json` can also hold a `video_processor` bound.
pub(super) fn read_preprocessor_max_pixels(model_dir: &std::path::Path) -> Option<usize> {
    // 2026-09-26: Ordered by precedence.
    const SOURCES: [(&str, Option<&str>); 3] = [
        ("preprocessor_config.json", None),
        ("preprocessor_config.json", Some("image_processor")),
        ("processor_config.json", Some("image_processor")),
    ];
    for (file, nest) in SOURCES {
        let path = model_dir.join(file);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let scope = match nest {
            Some(key) => match json.get(key) {
                Some(v) => v,
                None => continue,
            },
            None => &json,
        };
        let from_size = scope
            .get("size")
            .and_then(|s| s.get("longest_edge"))
            .and_then(serde_json::Value::as_u64);
        let direct = scope.get("max_pixels").and_then(serde_json::Value::as_u64);
        let Some(px) = from_size.or(direct).filter(|&p| p > 0) else {
            continue;
        };
        tracing::info!(
            "Vision area bound {} px from {}{} (was: hard-coded 1280px long side)",
            px,
            path.display(),
            nest.map(|k| format!(" [{k}]")).unwrap_or_default(),
        );
        return Some(px as usize);
    }
    None
}

#[path = "serve_quant.rs"]
mod quant;
pub(super) use quant::{canonicalize_model_quant, describe_quant_source, quant_pair_compatible};

#[cfg(test)]
#[path = "serve_tests.rs"]
mod tests;
