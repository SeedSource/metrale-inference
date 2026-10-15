// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Target resolution for build.rs: which (hw, model, quant)
//! targets the `METRALE_TARGET_*` env vars select, and the per-target
//! configuration read from HARDWARE.toml, KERNEL.toml and MODEL.toml.
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - The returned targets are sorted by (model, quant) and have passed
//!   `validate_collision_match_names`.
//!
//! Included via `#[path = "build_resolve.rs"] mod build_resolve;` so types
//! from build.rs (`Target`) are reachable via `super::`.

use std::collections::HashMap;
use std::env;

use super::build_parse::{
    parse_behavior, parse_dflash, parse_expected_absent, parse_kernel_toml, parse_match_names,
    parse_model_types, parse_sampling_presets, parse_shadow_exempt,
};
use super::{Target, build_diagnose, build_flags, validate_collision_match_names};

/// 2026-09-25: Resolve all compilation targets from env vars, expanding wildcards.
///
/// What each target compiles is the resolver's answer (`metrale_closure::
/// layout::discover`); this function only chooses which targets, reads the
/// per-target configuration the build bakes in, and merges the KERNEL.toml
/// layers the resolver reports in its order.
pub(super) fn resolve_targets(workspace_root: &std::path::Path) -> Vec<Target> {
    use metrale_closure::layout;

    let kernels_root = workspace_root.join("kernels");

    let hw = env::var("METRALE_TARGET_HW").unwrap_or_else(|_| build_diagnose::DEFAULT_HW.into());
    let model_spec = env::var("METRALE_TARGET_MODEL").unwrap_or_else(|_| "*".into());
    let quant_spec = env::var("METRALE_TARGET_QUANT").unwrap_or_else(|_| "nvfp4".into());

    let hw_dir = kernels_root.join(&hw);
    assert!(
        hw_dir.is_dir(),
        "{}",
        build_diagnose::unknown_hw(&kernels_root, &hw)
    );

    // 2026-09-25: Announce an implicit target selection before anything can
    // fail somewhere that never mentions the env var, but after the hardware
    // directory is known to exist, so a rejected target is not also announced
    // as the one being built.
    build_diagnose::warn_if_defaulted(&kernels_root, &hw, &model_spec, &quant_spec);

    let hw_toml_path = hw_dir.join("HARDWARE.toml");
    let hw_toml: toml::Value = toml::from_str(
        &std::fs::read_to_string(&hw_toml_path)
            .unwrap_or_else(|e| panic!("{}: {e}", hw_toml_path.display())),
    )
    .unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", hw_toml_path.display()));
    let arch = hw_toml["hardware"]["arch"]
        .as_str()
        .expect("hardware.arch must be a string in HARDWARE.toml")
        .to_string();
    // 2026-09-25: Vendor selects the flag key (`build_flags::flag_key`:
    // extra_metal_flags vs extra_nvcc_flags). Absent means nvidia.
    let target_vendor = hw_toml["hardware"]["vendor"]
        .as_str()
        .unwrap_or("nvidia")
        .to_string();
    println!("cargo:rerun-if-changed={}", hw_toml_path.display());
    // 2026-09-25: `[hardware] inherits` is validated by the resolver; surfaced here so a
    // bad declaration names the file rather than a target that failed to
    // resolve under it.
    let hardware = layout::hardware(&kernels_root, &hw).unwrap_or_else(|e| panic!("{e}"));

    // 2026-09-25: The least specific flag layer: `[build] extra_*_flags` in
    // HARDWARE.toml, for facts about the architecture rather than the model.
    // `kernels/hopper`, `kernels/b200` and `kernels/b300` define
    // `-DMETRALE_NO_WARP_BLOCKSCALE_MMA` here; gb10 declares none. See
    // `build_flags::hardware_extra_flags`. Read from this hardware's own
    // HARDWARE.toml only, so an overlay does not inherit gb10's.
    let hw_extra_flags = build_flags::hardware_extra_flags(&hw_toml, &target_vendor);

    // 2026-09-25: `*` expands to every model `layout::walk` finds for this
    // hardware (directories carrying a MODEL.toml).
    let models: Vec<String> = if model_spec == "*" {
        layout::walk(workspace_root)
            .unwrap_or_else(|e| panic!("{e}"))
            .into_iter()
            .filter(|t| t.hardware == hw)
            .map(|t| t.model)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    } else {
        vec![model_spec]
    };

    let mut targets = Vec::new();
    for model in &models {
        let model_dir = hw_dir.join(model);
        if !model_dir.is_dir() {
            panic!("{}", build_diagnose::unknown_model(&hw_dir, &hw, model));
        }

        // 2026-09-25: `*` expands to every quant directory the source model owns
        // here or in the inherited tree, as `layout::walk` enumerates them.
        let quants: Vec<String> = if quant_spec == "*" {
            layout::walk(workspace_root)
                .unwrap_or_else(|e| panic!("{e}"))
                .into_iter()
                .filter(|t| t.hardware == hw && &t.model == model)
                .map(|t| t.quant)
                .collect()
        } else {
            vec![quant_spec.clone()]
        };

        for quant in &quants {
            let target_id = layout::Target {
                hardware: hw.clone(),
                model: model.clone(),
                quant: quant.clone(),
            };
            let resolved = layout::discover(workspace_root, &target_id)
                .unwrap_or_else(|e| panic!("{target_id}: {e}"));
            assert_eq!(
                resolved.hardware, hardware,
                "kernels/{hw}/HARDWARE.toml changed between reads"
            );

            // 2026-09-25: KERNEL.toml: merge every layer the resolver consulted,
            // least specific first (`Layout::configs`: parent common, own common,
            // parent leaf, own leaf). A more specific layer appends flags
            // (deduped) and wins per key on `[modules]`; `[shadow_exempt]` is
            // the union.
            //
            // Flags have HARDWARE.toml's `hw_extra_flags` under all of these,
            // merged least-specific-first by `build_flags::merge_extra_flags`,
            // the SSOT for the order and the deduping: the common-role layers
            // concatenate into its `common` argument and the leaf-role layers
            // into `model`, which is the same dedup-in-order as one list.
            // `[modules]` has no hardware layer: a module rename is a property
            // of the source file, not the GPU.
            let mut common_flags: Vec<String> = Vec::new();
            let mut model_flags: Vec<String> = Vec::new();
            let mut module_overrides: HashMap<String, String> = HashMap::new();
            let mut shadow_exempt: Vec<(String, String)> = Vec::new();
            for config in resolved.configs() {
                let dir = config.parent().expect("KERNEL.toml has a directory");
                let (f, m) = parse_kernel_toml(dir, &target_vendor);
                if dir.file_name().is_some_and(|n| n == "common") {
                    common_flags.extend(f);
                } else {
                    model_flags.extend(f);
                }
                module_overrides.extend(m);
                shadow_exempt.extend(parse_shadow_exempt(dir));
            }
            let extra_flags =
                build_flags::merge_extra_flags(&hw_extra_flags, &common_flags, &model_flags);
            shadow_exempt.sort();
            shadow_exempt.dedup();

            // 2026-09-25: MODEL.toml is a build input: its sampling presets,
            // behavior, model_types, match_names, dflash and expected_absent are
            // generated into the binary, so an edit to it must rebuild.
            println!(
                "cargo:rerun-if-changed={}",
                model_dir.join("MODEL.toml").display()
            );
            let (s_tt, s_tc, s_nt, s_tools) = parse_sampling_presets(&model_dir);
            let pb = parse_behavior(&model_dir);
            let model_type_matches = parse_model_types(&model_dir);
            let match_names = parse_match_names(&model_dir);
            let dflash = parse_dflash(&model_dir);
            let expected_absent = parse_expected_absent(&model_dir);

            targets.push(Target {
                hw: hw.clone(),
                model: model.clone(),
                quant: quant.clone(),
                arch: arch.clone(),
                layout: resolved,
                extra_flags,
                module_overrides,
                shadow_exempt,
                expected_absent,
                sampling_thinking_text: s_tt,
                sampling_thinking_coding: s_tc,
                sampling_non_thinking: s_nt,
                sampling_tools: s_tools,
                behavior_thinking_in_tools: pb.thinking_in_tools,
                behavior_max_thinking_budget: pb.max_thinking_budget,
                behavior_effort_capped_at_ceiling: pb.effort_capped_at_ceiling,
                behavior_effort_prompt_only: pb.effort_prompt_only,
                behavior_render_tools_on_tool_choice_none: pb.render_tools_on_tool_choice_none,
                behavior_hf_tool_json: pb.hf_tool_json,
                behavior_tool_call_start_bias: pb.tool_call_start_bias,
                behavior_response_format_with_tools: pb.response_format_with_tools,
                behavior_thinking_default: pb.thinking_default,
                behavior_fp8_kv_calibration_tokens: pb.fp8_kv_calibration_tokens,
                behavior_default_kv_dtype: pb.default_kv_dtype,
                behavior_default_num_drafts: pb.default_num_drafts,
                behavior_disable_tool_steering: pb.disable_tool_steering,
                behavior_disable_cwd_hint_injection: pb.disable_cwd_hint_injection,
                behavior_use_sampling_presets_for_core: pb.use_sampling_presets_for_core,
                behavior_tool_call_parser: pb.tool_call_parser,
                behavior_enable_loop_watchdog: pb.enable_loop_watchdog,
                behavior_enable_think_loop_watchdog: pb.enable_think_loop_watchdog,
                behavior_honor_eos_inside_thinking: pb.honor_eos_inside_thinking,
                behavior_min_reasoning_floor_tokens: pb.min_reasoning_floor_tokens,
                behavior_cap_thinking_at_max_tokens: pb.cap_thinking_at_max_tokens,
                behavior_min_p_floor: pb.min_p_floor,
                behavior_temperature_max: pb.temperature_max,
                behavior_think_loop_min_repeats: pb.think_loop_min_repeats,
                behavior_think_loop_scan_window: pb.think_loop_scan_window,
                behavior_confidence_early_stop: pb.confidence_early_stop,
                behavior_confidence_run_length: pb.confidence_run_length,
                behavior_fuzzy_repeat_tolerance_div: pb.fuzzy_repeat_tolerance_div,
                behavior_max_inter_tool_prose: pb.max_inter_tool_prose,
                behavior_max_post_think_content_tokens: pb.max_post_think_content_tokens,
                behavior_tscg: pb.tscg,
                behavior_disable_tool_grammar: pb.disable_tool_grammar,
                behavior_rollback_resteer: pb.rollback_resteer,
                behavior_rom_head: pb.rom_head,
                behavior_tool_retry: pb.tool_retry,
                behavior_preserve_thinking: pb.preserve_thinking,
                model_type_matches,
                match_names,
                dflash,
            });
        }
    }

    // 2026-09-25: Sort by (model, quant) for deterministic ordering.
    targets.sort_by(|a, b| (&a.model, &a.quant).cmp(&(&b.model, &b.quant)));
    validate_collision_match_names(&targets);
    targets
}
