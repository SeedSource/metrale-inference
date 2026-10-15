// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The config, kernel-target and tokenizer steps of `load_model`
//! (phases 2, 3 and 8): the lm_head dtype, the media policies, the kernel
//! target and its quant check, the chat tokenizer, the server's default chat
//! template kwargs, and the MODEL.toml behavior with its CLI overrides.
//!
//! Owner: server startup (`met serve`).
//! Invariants: each function runs its steps in `load_model`'s order; the
//! `tracing` events keep `load_model`'s target (`met::main_modules::serve_load`).

use std::path::Path;

use anyhow::{Context, Result};
use metrale_config::ModelConfig;

use crate::cli;
use crate::main_modules::serve::{
    DefaultChatTemplateKwargs, canonicalize_model_quant, describe_quant_source,
    parse_default_chat_template_kwargs, quant_pair_compatible, resolve_vision_max_pixels,
};
use crate::main_modules::serve_phases;
use crate::tokenizer::ChatTokenizer;

pub(super) fn configure_model(
    args: &cli::ServeArgs,
    model_dir: &Path,
) -> Result<(ModelConfig, String)> {
    metrale_telemetry::progress::phase(2, "config");
    let (mut config, config_json) = serve_phases::load_model_config(model_dir)?;

    // 2026-09-26: `--lm-head-dtype` sets `lm_head_bf16_override` (read first by
    // `skip_lm_head_quantization`) and `lm_head_fp8`; `setup_lm_heads` reads
    // both, and uses a checkpoint's pre-packed NVFP4 head whatever they say.
    // An unknown value fails the boot.
    let (lm_head_bf16_override, lm_head_fp8) = match args.lm_head_dtype.as_str() {
        "default" => (None, false),
        "bf16" => (Some(true), false),
        // 2026-09-26: Quantize the lm_head to NVFP4 whatever the model's own
        // default is.
        "nvfp4" => (Some(false), false),
        // 2026-09-26: Quantize the lm_head to FP8 E4M3 with per-row scales,
        // or use the checkpoint's own FP8 head when it ships one.
        "fp8" => (Some(false), true),
        other => {
            anyhow::bail!(
                "--lm-head-dtype must be 'default', 'bf16', 'nvfp4', or 'fp8', got '{other}'"
            )
        }
    };
    config.lm_head_bf16_override = lm_head_bf16_override;
    config.lm_head_fp8 = lm_head_fp8;

    // 2026-09-26: A sibling `hf_quant_config.json` fills `quantization_config`
    // when config.json has none; its top level is read as the quantization
    // block.
    serve_phases::merge_sidecar_quant_config(model_dir, &mut config);
    Ok((config, config_json))
}

pub(super) fn resolve_media_policies(
    args: &cli::ServeArgs,
    model_dir: &Path,
    config: &mut ModelConfig,
) -> Result<(
    Option<usize>,
    crate::api::chat::remote_image::RemoteImagePolicy,
    metrale_model_layers::video_decode_ffmpeg::FfmpegPolicy,
)> {
    // 2026-09-26: The vision area bound is resolved once and installed on the
    // config before `build_model`, which builds the vision encoder; the
    // encoder sizes its patch buffers from it (`derive_max_patches`).
    let vision_max_pixels = resolve_vision_max_pixels(args, model_dir)?;
    if let Some(v) = config.vision.as_mut() {
        v.max_pixels = vision_max_pixels;
    }
    match vision_max_pixels {
        Some(px) => {
            tracing::info!(target: "met::main_modules::serve_load", "Vision area bound: {} px ({})",
                px,
                if args.vision_max_pixels > 0 {
                    "--vision-max-pixels"
                } else {
                    // 2026-09-26: Names no file: `read_preprocessor_max_pixels`
                    // logs the file and key it read.
                    "checkpoint processor config / METRALE_VISION_MAX_PIXELS"
                }
            )
        }
        None => {
            tracing::info!(target: "met::main_modules::serve_load", "Vision area bound: none declared — falling back to the 1280px long-side clamp"
            )
        }
    }

    // 2026-09-26: Logged at WARN when on: it lets the server make outbound HTTP
    // requests to URLs that clients supply.
    let remote_image_policy = crate::api::chat::remote_image::RemoteImagePolicy {
        enabled: args.vision_allow_remote_images,
        max_bytes: args.vision_remote_image_max_mb.saturating_mul(1024 * 1024),
        timeout_secs: args.vision_remote_image_timeout_s,
        allow_private: args.vision_remote_image_allow_private,
    };
    if remote_image_policy.enabled {
        tracing::warn!(target: "met::main_modules::serve_load", "Remote image fetching ENABLED (--vision-allow-remote-images): this server will \
             issue outbound HTTP to URLs supplied in chat requests. Cap {} MiB, timeout {} s, \
             private/loopback/link-local destinations {}.",
            args.vision_remote_image_max_mb,
            remote_image_policy.timeout_secs,
            if remote_image_policy.allow_private {
                "ALLOWED (--vision-remote-image-allow-private)"
            } else {
                "refused"
            }
        );
    } else {
        tracing::info!(target: "met::main_modules::serve_load", "Remote image fetching disabled (default); image_url parts carrying an http(s) \
             URL are refused with a 400. Send base64 data: URIs, or pass \
             --vision-allow-remote-images."
        );
    }

    // 2026-09-26: Probed at boot, so a missing ffmpeg shows in the startup log
    // instead of on the first video request.
    let video_ffmpeg = metrale_model_layers::video_decode_ffmpeg::FfmpegPolicy {
        enabled: args.video_allow_ffmpeg,
        binary: args.video_ffmpeg_path.clone(),
        max_frames: args.video_max_frames,
        timeout_secs: args.video_decode_timeout_s,
        ..Default::default()
    };
    match metrale_model_layers::video_decode_ffmpeg::probe(&video_ffmpeg) {
        metrale_model_layers::video_decode_ffmpeg::Availability::Ready(v) => {
            tracing::info!(target: "met::main_modules::serve_load", "Video decoding ENABLED via {} ({}); sampling at {} fps, max {} frames",
                args.video_ffmpeg_path,
                v,
                args.video_fps,
                args.video_max_frames,
            )
        }
        // 2026-09-26: A warning, not a boot failure: text and image serving do
        // not need ffmpeg.
        metrale_model_layers::video_decode_ffmpeg::Availability::Missing(why) => {
            tracing::warn!(target: "met::main_modules::serve_load", "Video decoding was ENABLED (--video-allow-ffmpeg) but the decoder is NOT \
                 USABLE: {why}. Every video request will fail. Install ffmpeg (apt install \
                 ffmpeg) or point --video-ffmpeg-path at it. Animated GIF still decodes \
                 in-process; text and image serving are unaffected.",
            )
        }
        metrale_model_layers::video_decode_ffmpeg::Availability::Disabled => {
            tracing::info!(target: "met::main_modules::serve_load", "Video decoding disabled (default). Animated GIF decodes in-process; every \
                 other container needs ffmpeg — pass --video-allow-ffmpeg to enable it."
            )
        }
    }
    Ok((vision_max_pixels, remote_image_policy, video_ffmpeg))
}

pub(super) fn log_model_config(config: &ModelConfig) {
    if let Some(ref qc) = config.quantization_config {
        tracing::info!(target: "met::main_modules::serve_load", "Quantization config: method={:?}, algo={:?}, format={:?}, {} module(s) in ignore list",
            qc.quant_method,
            qc.quant_algo,
            qc.format,
            qc.ignore_modules.len(),
        );
    }

    tracing::info!(target: "met::main_modules::serve_load", "Model config: {} layers, {} attention, {} SSM, {} experts, rope_theta={}, head_dim={}, rotary_dim={}",
        config.num_hidden_layers,
        config.num_attention_layers(),
        config.num_ssm_layers(),
        config.num_experts,
        config.rope_theta,
        config.head_dim,
        config.rotary_dim(),
    );
}

pub(super) fn select_kernel_target(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model_dir: &Path,
) -> Result<metrale_kernels::TargetPtxSet> {
    metrale_telemetry::progress::phase(3, "gpu init");
    // 2026-09-26: Target resolution rules are in `metrale_kernels::resolve`:
    // exact `hidden_size` before wildcards, a collision broken by each
    // target's `match_names` against `model_refs`, an unbroken tie an error,
    // and `--kernel-target` pins the target.
    let model_dir_str = model_dir.display().to_string();
    let model_refs: Vec<&str> = [
        args.model.as_deref(),
        args.model_name.as_deref(),
        Some(model_dir_str.as_str()),
    ]
    .into_iter()
    .flatten()
    .collect();
    let ptx_set = metrale_kernels::ptx_for_config(
        &config.model_type,
        config.hidden_size,
        &model_refs,
        args.kernel_target.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?
    .with_context(|| {
        format!(
            "No compiled kernel target matches model_type '{}' / hidden_size={}. \
             Available targets: {:?}",
            config.model_type,
            config.hidden_size,
            metrale_kernels::available_targets()
                .iter()
                .map(|t| &t.target.model)
                .collect::<Vec<_>>(),
        )
    })?;
    // 2026-09-26: kimi_k3 MXFP4 weights need the exact mxfp4 target: for several
    // quant variants of one target, `ptx_for_config` returns the first.
    let ptx_set = if config.model_type == "kimi_k3" && canonicalize_model_quant(config) == "mxfp4" {
        metrale_kernels::ptx_for_exact_target(ptx_set.target.model, "mxfp4")
            .context("K3 MXFP4 requires its compiled mxfp4 target; rebuild with METRALE_TARGET_QUANT=mxfp4 or *")?
    } else {
        ptx_set
    };
    Ok(ptx_set)
}

pub(super) fn check_kernel_target(
    ptx_set: &metrale_kernels::TargetPtxSet,
    config: &mut ModelConfig,
) -> Result<()> {
    // 2026-09-26: Record the resolved target for the dashboard's kernel table.
    crate::tui::data::kernels::publish_loaded_target(ptx_set.target.model, ptx_set.target.quant);

    // 2026-09-26: `ptx_for_config` does not select on quant, so the kernel and
    // model quant pair is checked against `quant_pair_compatible` here, before
    // any weight loads.
    let model_quant = canonicalize_model_quant(config);
    let kernel_quant = ptx_set.target.quant;
    if !quant_pair_compatible(kernel_quant, &model_quant) {
        anyhow::bail!(
            "Kernel/model QUANT MISMATCH. Kernel target: {} (quant={kernel_quant}). \
             Model declares quant={model_quant} ({}). \
             The compiled kernel set has no known dispatch path for \
             quant '{model_quant}' — loading would produce silent garbage. \
             Rebuild with METRALE_TARGET_QUANT={model_quant} (or =* to bundle multiple \
             variants) and restart.",
            ptx_set.target,
            describe_quant_source(config),
        );
    }
    tracing::info!(target: "met::main_modules::serve_load", "Selected kernel target: {} ({} modules) — quant compat: kernel={kernel_quant} \
         model={model_quant} OK",
        ptx_set.target,
        ptx_set.modules.len(),
    );

    // 2026-09-26: A checkpoint that declares a vision tower, on a target that
    // ships neither vision module (`vision_encoder`, `glm_vit`), is served
    // text-only: `config.vision` is cleared.
    const VISION_MODULES: [&str; 2] = ["vision_encoder", "glm_vit"];
    if config.vision.is_some()
        && !ptx_set
            .modules
            .iter()
            .any(|(name, _)| VISION_MODULES.contains(name))
    {
        tracing::warn!(target: "met::main_modules::serve_load", "Checkpoint declares a vision tower but kernel target {} ships neither a \
             vision_encoder nor a glm_vit module — serving TEXT-ONLY (image inputs \
             ignored). Rebuild the target with vision to enable images.",
            ptx_set.target,
        );
        config.vision = None;
    }
    Ok(())
}

/// 2026-09-27: Publish the model's row-tier policy before it is built:
/// canonical for an FP8 MoE checkpoint unless `--no-canonical-tiers`, with the two
/// `METRALE_*` levers over the checkpoint default (`row_tiers`).
pub(super) fn publish_row_tiers(args: &cli::ServeArgs, config: &ModelConfig) {
    use metrale_model_layers::layers::{publish_row_tiers, resolve_row_tiers};
    publish_row_tiers(resolve_row_tiers(
        args.no_canonical_tiers,
        std::env::var_os("METRALE_ROW_EXACT_TIERS").is_some(),
        std::env::var_os("METRALE_CANONICAL_TIERS").is_some(),
        config.num_experts > 0 && canonicalize_model_quant(config) == "fp8",
    ));
}

pub(super) fn resolve_behavior(
    ptx_set: &metrale_kernels::TargetPtxSet,
    args: &cli::ServeArgs,
    default_kwargs: &DefaultChatTemplateKwargs,
) -> metrale_kernels::ModelBehavior {
    let mut b = ptx_set.behavior.clone();
    if let Some(cli_budget) = args.max_thinking_budget {
        b.max_thinking_budget = cli_budget;
    }
    // 2026-10-14: `effort_prompt_only` (MODEL.toml) gives thinking no budget at any
    // effort. `--max-thinking-budget` or `METRALE_EFFORT_THINKING_BUDGET=1` brings the
    // budget ladder back (`api/chat/thinking.rs`).
    b.effort_prompt_only = effort_prompt_only(
        b.effort_prompt_only,
        args.max_thinking_budget.is_some(),
        std::env::var("METRALE_EFFORT_THINKING_BUDGET")
            .ok()
            .as_deref(),
    );
    if let Some(grammar_on) =
        crate::cli::flag_values::Tristate::validated(&args.tool_grammar).pinned()
    {
        b.disable_tool_grammar = !grammar_on;
    }
    // 2026-09-26: The server-level `preserve_thinking` overrides the
    // MODEL.toml value; a request's own value still wins
    // (`api/chat/prepare.rs`).
    if let Some(p) = default_kwargs.preserve_thinking {
        b.preserve_thinking = Some(p);
    }
    b
}

/// 2026-10-14: Whether effort reaches the model only through the template (no thinking
/// budget): MODEL.toml's `effort_prompt_only`, off when `--max-thinking-budget` is given or
/// `METRALE_EFFORT_THINKING_BUDGET` is `1` (default off).
fn effort_prompt_only(model: bool, cli_budget: bool, lever: Option<&str>) -> bool {
    model && !cli_budget && lever.map(str::trim) != Some("1")
}

#[cfg(test)]
mod effort_prompt_only_tests {
    use super::effort_prompt_only;

    #[test]
    fn lever_and_cli_budget_bring_the_ladder_back() {
        assert!(effort_prompt_only(true, false, None));
        assert!(effort_prompt_only(true, false, Some("0")));
        assert!(effort_prompt_only(true, false, Some("")));
        assert!(!effort_prompt_only(true, false, Some("1")));
        assert!(!effort_prompt_only(true, false, Some(" 1 ")));
        assert!(!effort_prompt_only(true, true, None));
        // 2026-10-14: a model without the key keeps its ladder whatever the lever says.
        assert!(!effort_prompt_only(false, false, None));
        assert!(!effort_prompt_only(false, false, Some("0")));
    }
}

pub(super) fn load_tokenizer(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model_dir: &Path,
    eos_tokens: &[u32],
    hf_tool_json: bool,
) -> Result<(ChatTokenizer, bool)> {
    metrale_telemetry::progress::phase(8, "tokenizer");
    // 2026-09-26: `supports_thinking` comes from `ModelCapabilities`.
    let caps = config.capabilities();
    let supports_thinking = caps.supports_thinking;
    // 2026-10-15: `hf_tool_json` is MODEL.toml `[behavior].hf_tool_json` (GLM-5.3).
    let tokenizer = ChatTokenizer::from_model_dir_with(
        model_dir,
        eos_tokens[0],
        supports_thinking,
        &config.model_type,
        Some(std::path::Path::new(".")),
        args.disable_template_overrides,
        hf_tool_json,
    )?;
    Ok((tokenizer, supports_thinking))
}

pub(super) fn parse_default_kwargs(args: &cli::ServeArgs) -> Result<DefaultChatTemplateKwargs> {
    // 2026-09-26: An unknown key or `reasoning_effort` value in
    // `--default-chat-template-kwargs` fails the boot.
    let default_kwargs = args
        .default_chat_template_kwargs
        .as_deref()
        .map(parse_default_chat_template_kwargs)
        .transpose()?
        .unwrap_or_default();
    Ok(default_kwargs)
}
