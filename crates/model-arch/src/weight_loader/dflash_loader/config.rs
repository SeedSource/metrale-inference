// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The drafter `config.json` schema: [`DflashConfig`] with the
//! nested [`DflashRopeScaling`] and [`DflashSubConfig`] blocks, and their
//! serde defaults.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use serde::Deserialize;

/// 2026-09-25: The fields of a drafter's `config.json` that the engine reads,
/// under their `config.json` names. Unknown keys are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub draft_vocab_size: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// 2026-09-25: Top-level block size γ; 16 when absent. Read it through
    /// [`DflashConfig::effective_block_size`].
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    #[serde(default)]
    pub dflash_config: Option<DflashSubConfig>,
    /// 2026-09-25: Drafter RoPE θ; 10,000,000 when absent.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// 2026-09-25: The `rope_scaling` block, also read under the key
    /// `rope_parameters`. `None` means plain RoPE. With `rope_type == "yarn"`
    /// the head builds a YaRN inv_freq table; any other block falls back to
    /// plain RoPE with a warning.
    #[serde(default, alias = "rope_parameters")]
    pub rope_scaling: Option<DflashRopeScaling>,
    /// 2026-09-25: DSpark Markov head rank, a top-level key; 0 when absent,
    /// which loads no Markov head.
    #[serde(default)]
    pub markov_rank: usize,
    /// 2026-09-25: DSpark Markov head type. When the Markov head loads, any
    /// value other than `"vanilla"` fails the load.
    #[serde(default)]
    pub markov_head_type: Option<String>,
    /// 2026-09-25: Load the DSpark confidence head (`confidence_head.proj.*`)
    /// when the checkpoint has it; false when absent.
    #[serde(default)]
    pub enable_confidence_head: bool,
    /// 2026-09-25: When true the confidence head reads the hidden state and
    /// then the Markov embedding (`hidden_size + markov_rank` inputs); when
    /// false, the hidden state only. True when absent.
    #[serde(default = "default_true")]
    pub confidence_head_with_markov: bool,

    /// 2026-10-01: The drafter's RMSNorm epsilon, when its `config.json`
    /// states it. Read only under `METRALE_DFLASH_CKPT_ARCH=1`
    /// ([`DflashConfig::resolved_rms_norm_eps`]); otherwise the head keeps
    /// its built-in 1e-6. Both GLM-5.3 DFlash2 drafters (incoai, canada-quant
    /// G) state 1e-5.
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,
    /// 2026-10-01: Sliding-window size, when the config states one. Logged
    /// only: the paged drafter attention does not window (see
    /// `ckpt_arch::log_arch_summary`).
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// 2026-10-01: `use_sliding_window`; false when absent.
    #[serde(default)]
    pub use_sliding_window: bool,
    /// 2026-10-01: Per-layer attention kinds (`"full_attention"`,
    /// `"sliding_attention"`); empty when absent. Logged only.
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// 2026-10-01: The exporter's block (canada-quant `export`), when present.
    /// `ships_mask_embedding` marks a drafter trained with a learned mask
    /// embedding that lives outside the safetensors, in `mask_embedding.pt`.
    #[serde(default)]
    pub export: Option<DflashExport>,
    /// 2026-10-01: The learned mask embedding as BF16 little-endian bytes
    /// (`hidden_size * 2`), loaded from the drafter directory by
    /// `ckpt_arch::attach_mask_embedding` under `METRALE_DFLASH_CKPT_ARCH=1`.
    /// Never read from `config.json`. `None` means the mask rows embed the
    /// target's `embed_tokens[mask_token_id]`, as before.
    #[serde(skip)]
    pub mask_embedding_bf16: Option<Vec<u8>>,
}

/// 2026-10-01: The `export` block a canada-quant DFlash2 checkpoint carries.
/// Unknown keys are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DflashExport {
    /// 2026-10-01: The drafter was trained with a learned mask embedding that
    /// ships as `mask_embedding.pt` next to the weights.
    #[serde(default)]
    pub ships_mask_embedding: bool,
    /// 2026-10-01: The checkpoint carries its own `embed_tokens.weight`.
    #[serde(default)]
    pub ships_embed_tokens: bool,
    /// 2026-10-01: The checkpoint carries its own `lm_head.weight`.
    #[serde(default)]
    pub ships_lm_head: bool,
}

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

/// 2026-09-25: The `rope_scaling` fields the drafter head reads.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRopeScaling {
    /// 2026-09-25: Only `"yarn"` is recognised; any other value, or none, falls
    /// back to plain RoPE with a warning when the head is built.
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<f32>,
    /// 2026-10-01: RoPE θ nested in the block, the transformers 5.x layout
    /// (`"rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}`),
    /// which carries no top-level `rope_theta`. Read only under
    /// `METRALE_DFLASH_CKPT_ARCH=1` ([`DflashConfig::resolved_rope_theta`]).
    #[serde(default)]
    pub rope_theta: Option<f32>,
}

fn default_block_size() -> usize {
    16
}

/// 2026-09-25: The nested `dflash_config` block of the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashSubConfig {
    /// 2026-09-25: Token id that fills the draft block after its anchor token.
    pub mask_token_id: u32,
    /// 2026-09-25: Target-model layers whose hidden states the drafter reads.
    /// The head refuses to build when the list is empty.
    pub target_layer_ids: Vec<usize>,
    /// 2026-09-25: `"dspark"` marks a drafter whose row j predicts position
    /// j + 1; the head then rotates the draft vector right by one, unless
    /// `METRALE_DSPARK_SHIFT` overrides it.
    #[serde(default)]
    pub projector_type: Option<String>,

    /// 2026-09-25: DFlash2 conv kernel size; 0 when absent. With
    /// `conv_group_size`, a non-zero value lets the conv tensors load.
    #[serde(default)]
    pub conv_kernel_size: usize,
    /// 2026-09-25: DFlash2 channels per conv group; 0 when absent.
    #[serde(default)]
    pub conv_group_size: usize,
    /// 2026-09-25: DFlash2 selector codebook rank; 0 when absent. With
    /// `selector_top_k`, a non-zero value lets the selector tensors load.
    #[serde(default)]
    pub selector_rank: usize,
    /// 2026-09-25: DFlash2 candidates kept per position; 0 when absent.
    #[serde(default)]
    pub selector_top_k: usize,
    /// 2026-09-25: The drafter's trained block size, when the checkpoint states
    /// it here. `None` falls back to the top-level field. Read it through
    /// [`DflashConfig::effective_block_size`].
    #[serde(default)]
    pub block_size: Option<usize>,
}

impl DflashConfig {
    /// 2026-09-25: Resolved block size γ: `dflash_config.block_size` when set,
    /// else the top-level field. The sub-config comes first because serde
    /// fills the top-level default of 16 for a checkpoint that never states
    /// it.
    pub fn effective_block_size(&self) -> usize {
        self.dflash_config
            .as_ref()
            .and_then(|c| c.block_size)
            .unwrap_or(self.block_size)
    }

    /// 2026-10-01: The RoPE θ the head uses. With `ckpt_arch` (the
    /// `METRALE_DFLASH_CKPT_ARCH=1` lever) a θ nested in
    /// `rope_parameters`/`rope_scaling` wins over the top-level field, whose
    /// serde default (10,000,000) is what a transformers 5.x config without a
    /// top-level `rope_theta` otherwise gets. Without `ckpt_arch` it is the
    /// top-level field, unchanged.
    pub fn resolved_rope_theta(&self, ckpt_arch: bool) -> f32 {
        if ckpt_arch && let Some(t) = self.nested_rope_theta() {
            return t;
        }
        self.rope_theta
    }

    /// 2026-10-01: The θ nested in `rope_parameters`/`rope_scaling`, if any.
    pub fn nested_rope_theta(&self) -> Option<f32> {
        self.rope_scaling.as_ref().and_then(|r| r.rope_theta)
    }

    /// 2026-10-01: The RMSNorm epsilon the head uses: the config's
    /// `rms_norm_eps` with `ckpt_arch`, else (or when absent) the head's
    /// built-in 1e-6.
    pub fn resolved_rms_norm_eps(&self, ckpt_arch: bool) -> f32 {
        if ckpt_arch && let Some(e) = self.rms_norm_eps {
            return e;
        }
        1e-6
    }

    /// 2026-10-01: True when the exporter states the drafter needs a learned
    /// mask embedding (`export.ships_mask_embedding`).
    pub fn needs_mask_embedding(&self) -> bool {
        self.export.as_ref().is_some_and(|e| e.ships_mask_embedding)
    }
}
