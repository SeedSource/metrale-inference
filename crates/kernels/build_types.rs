// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The types build.rs resolves a kernel target into: `Target`,
//! its sampling presets (`SamplingCat`), `[[model_types]]` entries
//! (`ModelTypeMatch`) and `[dflash]` section (`DflashRaw`).
//!
//! Owner: metrale-kernels build.
//! Invariants: none beyond the types.
//!
//! Included via `#[path = "build_types.rs"] mod build_types;` and re-imported
//! at the root of build.rs, so the other build modules reach them via
//! `super::`. Items and fields are `pub(super)`: visible to build.rs and its
//! modules only.

use std::collections::HashMap;

use metrale_closure::layout::Layout;

/// 2026-09-25: Per-category sampling defaults parsed from MODEL.toml `[sampling.*]`.
#[derive(Debug, Clone)]
pub(super) struct SamplingCat {
    pub(super) temperature: f32,
    pub(super) top_p: f32,
    pub(super) top_k: u32,
    pub(super) presence_penalty: f32,
    pub(super) frequency_penalty: f32,
    pub(super) repetition_penalty: f32,
    // 2026-09-25: DRY sampler parameters (`SamplingCategory` in src/lib.rs
    // documents them). The default `dry_multiplier` of 0.0 disables DRY; a
    // MODEL.toml `[sampling.*]` table opts in.
    pub(super) dry_multiplier: f32,
    pub(super) dry_base: f32,
    pub(super) dry_allowed_length: u32,
    // 2026-09-25: LZ penalty strength (`SamplingCategory` in src/lib.rs
    // documents it). 0.0 disables it.
    pub(super) lz_penalty: f32,
    // 2026-09-25: Model-declared min-p. None = MODEL.toml is silent, so the
    // server's --default-min-p stands (see SamplingCategory in src/lib.rs).
    pub(super) min_p: Option<f32>,
    pub(super) top_n_sigma: Option<f32>,
}

impl Default for SamplingCat {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
            min_p: None,
            top_n_sigma: None,
        }
    }
}

/// 2026-09-25: A `(model_type, optional hidden_size)` pair declaring which models a kernel target supports.
#[derive(Debug, Clone)]
pub(super) struct ModelTypeMatch {
    pub(super) model_type: String,
    pub(super) hidden_size: Option<usize>,
}

/// 2026-09-25: A resolved (hw, model, quant) compilation target.
pub(super) struct Target {
    pub(super) hw: String,
    pub(super) model: String,
    pub(super) quant: String,
    pub(super) arch: String,
    /// 2026-09-25: Everything this target compiles, resolved by
    /// `metrale_closure::layout`: its layers, their files and the KERNEL.tomls
    /// that apply. Compiled from a staged copy of the two role directories
    /// (see `build_stage.rs`).
    pub(super) layout: Layout,
    pub(super) extra_flags: Vec<String>,
    pub(super) module_overrides: HashMap<String, String>,
    /// 2026-09-25: `(module, kernel)` pairs declared in `[shadow_exempt]`:
    /// kernels this target may drop from `common/` without it being drift,
    /// each with a stated reason in its KERNEL.toml. The union over every
    /// KERNEL.toml layer. Filters the build warning only: the pairs stay in
    /// `shadowed_dropped`, and the boot audit judges an unresolved lookup by
    /// `expected_absent`, not by this list.
    pub(super) shadow_exempt: Vec<(String, String)>,
    /// 2026-09-25: `(module, kernel)` pairs declared in MODEL.toml
    /// `[expected_absent]`: lookups this model's dispatch may issue and fail
    /// to resolve without that being an error, each with a mandatory stated
    /// reason. Carried onto `TargetPtxSet::expected_absent`; the boot audit
    /// fails on every unresolved lookup not declared here, unless
    /// `--dangerously-allow-unresolved-kernel-lookups` is set.
    pub(super) expected_absent: Vec<(String, String)>,
    pub(super) sampling_thinking_text: SamplingCat,
    pub(super) sampling_thinking_coding: SamplingCat,
    pub(super) sampling_non_thinking: SamplingCat,
    pub(super) sampling_tools: SamplingCat,
    pub(super) behavior_thinking_in_tools: bool,
    pub(super) behavior_max_thinking_budget: u32,
    pub(super) behavior_effort_capped_at_ceiling: bool,
    pub(super) behavior_effort_prompt_only: bool,
    pub(super) behavior_tool_call_start_bias: bool,
    pub(super) behavior_response_format_with_tools: bool,
    pub(super) behavior_thinking_default: bool,
    pub(super) behavior_fp8_kv_calibration_tokens: usize,
    pub(super) behavior_default_kv_dtype: String,
    pub(super) behavior_default_num_drafts: u32,
    pub(super) behavior_disable_tool_steering: bool,
    pub(super) behavior_disable_cwd_hint_injection: bool,
    pub(super) behavior_use_sampling_presets_for_core: bool,
    pub(super) behavior_tool_call_parser: String,
    pub(super) behavior_enable_loop_watchdog: bool,
    pub(super) behavior_enable_think_loop_watchdog: bool,
    pub(super) behavior_honor_eos_inside_thinking: bool,
    pub(super) behavior_min_reasoning_floor_tokens: u32,
    pub(super) behavior_cap_thinking_at_max_tokens: bool,
    pub(super) behavior_min_p_floor: f32,
    pub(super) behavior_temperature_max: f32,
    pub(super) behavior_think_loop_min_repeats: u32,
    pub(super) behavior_think_loop_scan_window: u32,
    pub(super) behavior_confidence_early_stop: bool,
    pub(super) behavior_confidence_run_length: u32,
    pub(super) behavior_fuzzy_repeat_tolerance_div: u32,
    pub(super) behavior_max_inter_tool_prose: u32,
    pub(super) behavior_max_post_think_content_tokens: u32,
    pub(super) behavior_tscg: bool,
    pub(super) behavior_disable_tool_grammar: bool,
    pub(super) behavior_rollback_resteer: bool,
    pub(super) behavior_rom_head: String,
    pub(super) behavior_tool_retry: bool,
    pub(super) behavior_preserve_thinking: Option<bool>,
    /// 2026-09-25: Which `(model_type, hidden_size)` pairs this kernel target
    /// supports, parsed from `[[model_types]]` in MODEL.toml.
    pub(super) model_type_matches: Vec<ModelTypeMatch>,
    /// 2026-09-25: `[model] match_names`: checkpoint-reference needles that
    /// break the tie when several targets declare the same
    /// `(model_type, hidden_size)`, as `qwen3.6-27b` and `qwen3.8-27b` both
    /// declare `(qwen3_5, 5120)`. `validate_collision_match_names` panics if a
    /// colliding target omits them.
    pub(super) match_names: Vec<String>,
    /// 2026-09-25: MODEL.toml `[dflash]`, the drafter pairing for
    /// block-diffusion speculative decoding. `None` when the section is absent.
    pub(super) dflash: Option<DflashRaw>,
}

#[derive(Default, Clone)]
pub(super) struct DflashRaw {
    pub(super) draft_model: String,
    pub(super) gamma: usize,
    pub(super) window_size: usize,
    pub(super) mask_token_id: u32,
    pub(super) target_layer_ids: Vec<usize>,
}
