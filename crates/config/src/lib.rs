// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The typed model config (`ModelConfig`) parsed from a checkpoint's `config.json`
//! or GGUF metadata, its capabilities, and the `METRALE_*` lever table.
//!
//! Owner: config.
//! Invariants: none beyond the types.

use anyhow::Result;
use serde::Deserialize;

/// 2026-09-26: Deserialize a u32 that may be JSON null; null gives 0.
fn nullable_u32<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<u32, D::Error> {
    Option::<u32>::deserialize(d).map(|v| v.unwrap_or(0))
}

/// 2026-09-26: `eos_token_id` as null (0), a scalar, or an array (its first element).
///
/// `parse_config` recovers the whole array into [`ModelConfig::eos_token_ids`]. A parser that
/// wants another primary (`step3p7` takes the last element) rewrites the field before
/// deserializing.
fn eos_token_id_field<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<u32, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(u32),
        Many(Vec<u32>),
    }
    Ok(match Option::<OneOrMany>::deserialize(d)? {
        None => 0,
        Some(OneOrMany::One(v)) => v,
        Some(OneOrMany::Many(v)) => v.first().copied().unwrap_or(0),
    })
}

/// 2026-09-26: Which dtype ladder GLM-5.3's MoE router runs in. The glm5_next parser reads
/// `moe_router_dtype` and uses `HfFp32` when it is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Glm5NextRouterMode {
    /// 2026-09-26: The router math in fp32 (`is_fp32`). The default.
    #[default]
    HfFp32,
    /// 2026-09-26: The router's bf16 ladder (`router_bf16_ladder` in glm5next_mlp).
    VllmBf16,
}

impl Glm5NextRouterMode {
    /// 2026-09-26: Parse a `moe_router_dtype` value; an unrecognised one gives `None`, which
    /// the glm5_next parser turns into an error.
    pub fn from_config_str(s: &str) -> Option<Self> {
        match s {
            "float32" | "fp32" => Some(Self::HfFp32),
            "bfloat16" | "bf16" => Some(Self::VllmBf16),
            _ => None,
        }
    }

    pub fn is_fp32(self) -> bool {
        matches!(self, Self::HfFp32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    SlidingAttention,
    LinearAttention,
    /// 2026-09-26: A layer with no mixer, only the routed-expert FFN (Nemotron-H pattern `E`,
    /// Puzzle `moe`).
    Moe,
    /// 2026-09-26: Sparse attention (`deepseek_sparse_attention`): a full-rank mixer whose
    /// visible key set an indexer selects per query at run time.
    SparseAttention,
}

impl LayerType {
    pub fn is_attention(self) -> bool {
        matches!(
            self,
            Self::FullAttention | Self::SlidingAttention | Self::SparseAttention
        )
    }

    pub fn hf_name(self) -> &'static str {
        match self {
            Self::FullAttention => "full_attention",
            Self::SlidingAttention => "sliding_attention",
            Self::LinearAttention => "linear_attention",
            Self::Moe => "moe",
            Self::SparseAttention => "deepseek_sparse_attention",
        }
    }
}

/// 2026-09-26: The weight-quantization block of a checkpoint config (`quantization_config`,
/// into which the server merges a sibling `hf_quant_config.json`), as read by
/// `parse_quantization_config`. That parser returns `None` when the block declares no method,
/// algorithm or ignore list.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// 2026-09-26: `quant_method` as written, `modelopt` for a ModelOpt sidecar that omits it,
    /// or empty.
    pub quant_method: String,
    /// 2026-09-26: `quant_algo`, else `NVFP4`/`FP8` derived from `config_groups.group_0.weights`,
    /// else empty.
    pub quant_algo: String,
    /// 2026-09-26: `format`, or empty.
    pub format: String,
    /// 2026-09-26: Module-path patterns from `ignore`, then those from `exclude_modules` not
    /// already listed.
    pub ignore_modules: Vec<String>,
}

/// 2026-09-26: Whether GLM-5.3's vision tower is enabled for this process
/// (`METRALE_GLM_VISION`, see [`glm_vision_enabled_from`]); off by default.
///
/// The glm5_next parser (which leaves `config.vision` `None` when off) and
/// `Glm5NextWeightLoader::binds_vision_encoder` both call it, so they cannot disagree.
pub fn glm_vision_enabled() -> bool {
    glm_vision_enabled_from(std::env::var("METRALE_GLM_VISION").ok().as_deref())
}

/// 2026-09-26: The policy half of [`glm_vision_enabled`], taking the value instead of reading
/// the environment, so tests need not set a variable. Only `1`, `true`, `yes` or `on`
/// (trimmed, any case) enable; anything else, unset included, leaves the tower off.
pub fn glm_vision_enabled_from(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

#[cfg(test)]
mod glm_vision_gate_tests {
    use super::glm_vision_enabled_from;

    #[test]
    fn only_an_explicit_affirmative_enables_the_tower() {
        for off in [
            None,
            Some(""),
            Some("0"),
            Some("off"),
            Some("false"),
            Some("no"),
            Some("yes please"),
        ] {
            assert!(!glm_vision_enabled_from(off), "{off:?} must not enable");
        }
        for on in ["1", "true", "TRUE", "Yes", " on "] {
            assert!(glm_vision_enabled_from(Some(on)), "{on:?} must enable");
        }
    }
}

/// 2026-10-03: Whether GLM-5.3's routed-MoE prefill runs W4A4 on the CUTLASS grouped NVFP4 GEMM
/// (`METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4`, see [`glm_moe_prefill_cutlass_w4a4_from`]); off by
/// default. Read once per process.
///
/// Three places read it and must agree: the server's weight-load skip list (the lever keeps the
/// `*.input_scale` names so the GLM loader can defer them), `Glm5NextWeightLoader`'s defer
/// predicate and expert binder (which read the deferred scales from disk), and the GLM prefill
/// dispatch.
pub fn glm_moe_prefill_cutlass_w4a4() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        glm_moe_prefill_cutlass_w4a4_from(
            std::env::var("METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-10-03: The policy half of [`glm_moe_prefill_cutlass_w4a4`]: only `1` (trimmed) turns
/// it on, like its neighbour levers (`METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1`).
pub fn glm_moe_prefill_cutlass_w4a4_from(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

#[cfg(test)]
mod glm_moe_cutlass_w4a4_gate_tests {
    use super::glm_moe_prefill_cutlass_w4a4_from as on;

    #[test]
    fn only_one_turns_it_on() {
        for off in [None, Some(""), Some("0"), Some("true"), Some("on"), Some("11"), Some("1x")] {
            assert!(!on(off), "{off:?} must leave the lever off");
        }
        for v in ["1", " 1", "1\n"] {
            assert!(on(Some(v)), "{v:?} must turn the lever on");
        }
    }
}

/// 2026-10-04: Whether the CUTLASS W4A4 routed-MoE prefill ignores the checkpoint's static
/// `input_scale`s and quantizes every activation with the dynamic per-call global scale
/// (`METRALE_GLM_MOE_W4A4_DYNAMIC_SCALE`, see [`glm_moe_w4a4_dynamic_scale_from`]); off by
/// default, and only meaningful with `METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1`. Read once per
/// process.
pub fn glm_moe_w4a4_dynamic_scale() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        glm_moe_w4a4_dynamic_scale_from(
            std::env::var("METRALE_GLM_MOE_W4A4_DYNAMIC_SCALE")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-10-04: The policy half of [`glm_moe_w4a4_dynamic_scale`]: only `1` (trimmed) turns it on.
pub fn glm_moe_w4a4_dynamic_scale_from(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// 2026-10-04: Whether the CUTLASS W4A4 routed-MoE prefill runs its DOWN projection as the
/// W4A16 grouped GEMM (BF16 activations) instead of W4A4 (`METRALE_GLM_MOE_W4A4_DOWN_W4A16`, see
/// [`glm_moe_w4a4_down_w4a16_from`]); gate/up stay W4A4. Off by default, and only meaningful with
/// `METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1`. Read once per process.
pub fn glm_moe_w4a4_down_w4a16() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        glm_moe_w4a4_down_w4a16_from(
            std::env::var("METRALE_GLM_MOE_W4A4_DOWN_W4A16")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-10-04: The policy half of [`glm_moe_w4a4_down_w4a16`]: only `1` (trimmed) turns it on.
pub fn glm_moe_w4a4_down_w4a16_from(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// 2026-10-05: Whether GLM-5.3 runs its routed experts tensor-parallel over the intermediate
/// dimension instead of expert-parallel (`METRALE_GLM_EXPERT_TP`, see [`glm_expert_tp_from`]);
/// off by default. Read once per process.
///
/// Four places read it and must agree: the server's topology check and weight-store EP view
/// (`serve_phases`), `Glm5NextWeightLoader`'s defer predicate, and
/// `Glm5NextMlpConfig::from_config` (which the text and MTP loaders both call, so the two never
/// mix layouts).
pub fn glm_expert_tp() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| glm_expert_tp_from(std::env::var("METRALE_GLM_EXPERT_TP").ok().as_deref()))
}

/// 2026-10-05: The policy half of [`glm_expert_tp`]: only `1` (trimmed) turns it on.
pub fn glm_expert_tp_from(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

#[cfg(test)]
mod glm_expert_tp_gate_tests {
    use super::glm_expert_tp_from as on;

    #[test]
    fn only_one_turns_expert_tp_on() {
        for off in [None, Some(""), Some("0"), Some("true"), Some("on"), Some("11")] {
            assert!(!on(off), "{off:?} must leave the lever off");
        }
        for v in ["1", " 1", "1\n"] {
            assert!(on(Some(v)), "{v:?} must turn the lever on");
        }
    }
}

#[cfg(test)]
mod glm_moe_w4a4_fix_lever_tests {
    use super::{glm_moe_w4a4_down_w4a16_from, glm_moe_w4a4_dynamic_scale_from};

    #[test]
    fn only_one_turns_either_fix_lever_on() {
        for f in [
            glm_moe_w4a4_dynamic_scale_from,
            glm_moe_w4a4_down_w4a16_from,
        ] {
            for off in [
                None,
                Some(""),
                Some("0"),
                Some("true"),
                Some("on"),
                Some("11"),
                Some("1x"),
            ] {
                assert!(!f(off), "{off:?} must leave the lever off");
            }
            for v in ["1", " 1", "1\n"] {
                assert!(f(Some(v)), "{v:?} must turn the lever on");
            }
        }
    }
}

/// 2026-09-26: Vision tower configuration: `parse_vision_config` for the Qwen family, and the
/// GLM-5.3 parser, which also fills the fields the Qwen parser leaves at their defaults.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    pub deepstack_visual_indexes: Vec<usize>,
    /// 2026-09-26: The token that marks where vision embeddings are spliced into the text.
    /// 0 means undeclared: the engine then uses `vision_encoder::IMAGE_PAD_TOKEN_ID`, and the
    /// chat tokenizer the id of `<|image_pad|>`.
    pub image_pad_token_id: u32,
    /// 2026-09-26: The video-frame counterpart of [`Self::image_pad_token_id`], with the same
    /// 0 rule (`VIDEO_PAD_TOKEN_ID`, `<|video_pad|>`).
    pub video_pad_token_id: u32,
    /// 2026-09-26: The vision area bound in pixels. The parsers leave it `None`; `serve_load`
    /// sets it from `resolve_vision_max_pixels` before the model is built, and the vision
    /// loaders pass it to the encoder.
    pub max_pixels: Option<usize>,

    pub hidden_act: String,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub projection_intermediate_size: usize,
    pub swiglu_limit: f32,
    pub rope_theta: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    /// 2026-09-26: Patch tokens in 2x2-block-major order rather than raster; the preprocessor
    /// reads it (`vision_preprocess`). The GLM-5.3 parser sets it.
    pub block_major_patches: bool,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            depth: 0,
            hidden_size: 0,
            num_heads: 0,
            patch_size: 0,
            temporal_patch_size: 0,
            spatial_merge_size: 0,
            intermediate_size: 0,
            out_hidden_size: 0,
            deepstack_visual_indexes: Vec::new(),
            image_pad_token_id: 0,
            video_pad_token_id: 0,
            max_pixels: None,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            rms_norm_eps: 1e-6,
            attention_bias: true,
            projection_intermediate_size: 0,
            swiglu_limit: 0.0,
            rope_theta: 10_000.0,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            min_image_tokens: 0,
            max_image_tokens: 0,
            block_major_patches: false,
        }
    }
}

impl VisionConfig {
    pub fn merger_input_size(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size * self.hidden_size
    }
}

pub(crate) fn default_one() -> usize {
    1
}
pub(crate) fn default_one_f64() -> f64 {
    1.0
}
pub(crate) fn default_one_f32() -> f32 {
    1.0
}
pub(crate) fn default_rope_theta() -> f64 {
    10000.0
}
pub(crate) fn default_rms_eps() -> f64 {
    1e-6
}
pub(crate) fn default_partial_rotary() -> f64 {
    1.0
}
pub(crate) fn default_conv_kernel() -> usize {
    4
}

mod dispatch;
mod factory;
mod gguf;
mod kv_completeness;
#[cfg(test)]
mod kv_completeness_tests;
pub mod levers;
mod methods;
mod model_config;
mod parsers;
#[cfg(test)]
mod tests;

pub use dispatch::parse_config;
pub use gguf::{GgufConfigInputs, GgufMeta, config_from_gguf};
pub use model_config::ModelConfig;
pub use parsers::{
    PEFT_SUPPORTED_TARGET_MODULES, PeftAdapterConfig, allow_partial_targets,
    glm5_next_mtp_layer_index, parse_mistral_params, parse_peft_adapter_config,
    parse_quantization_config,
};
pub(crate) use parsers::{
    parse_deepseek_v4, parse_gemma4_params, parse_glm5_next, parse_kimi_k3, parse_laguna,
    parse_longcat_ngram, parse_minimax_m2, parse_qwen4_exp, parse_step3p7, parse_vision_config,
    sanitize_kimi_k3_eos,
};

pub(crate) fn finalize_config(config: &mut ModelConfig, raw: &serde_json::Value) -> Result<()> {
    if config.quantization_config.is_none() {
        config.quantization_config = parse_quantization_config(raw);
    }
    validate_config(config)
}

/// 2026-09-26: Post-parse validation: `layer_types` and `num_attention_heads_per_layer`
/// lengths against `num_hidden_layers`, and the SSM and Mamba-2 head fields.
pub(crate) fn validate_config(config: &ModelConfig) -> Result<()> {
    if !config.layer_types.is_empty() && config.layer_types.len() != config.num_hidden_layers {
        anyhow::bail!(
            "layer_types length ({}) doesn't match num_hidden_layers ({}) in config.json",
            config.layer_types.len(),
            config.num_hidden_layers,
        );
    }

    if !config.num_attention_heads_per_layer.is_empty()
        && config.num_attention_heads_per_layer.len() != config.num_hidden_layers
    {
        anyhow::bail!(
            "num_attention_heads_per_layer length ({}) doesn't match num_hidden_layers ({}) in config.json",
            config.num_attention_heads_per_layer.len(),
            config.num_hidden_layers,
        );
    }

    let has_ssm =
        config.layer_types.contains(&LayerType::LinearAttention) || config.linear_num_key_heads > 0;
    if has_ssm && config.linear_num_key_heads == 0 && config.mamba_num_heads == 0 {
        anyhow::bail!(
            "SSM model detected but linear_num_key_heads is 0 in config.json. \
             This field is required for SSM/GDN layer initialization."
        );
    }

    if config.mamba_num_heads > 0 {
        if config.mamba_head_dim == 0 {
            anyhow::bail!("mamba_head_dim must be greater than zero");
        }
        if config.ssm_state_size == 0 {
            anyhow::bail!("ssm_state_size must be greater than zero");
        }
        if config.n_groups == 0 {
            anyhow::bail!("n_groups must be greater than zero");
        }
        if !config.mamba2_d_inner().is_multiple_of(config.n_groups) {
            anyhow::bail!("mamba_num_heads * mamba_head_dim must be divisible by n_groups");
        }
    }

    Ok(())
}

pub mod capabilities;
