// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: DFlash drafter loader: parses the drafter's `config.json` and
//! loads its weights into [`DflashWeights`] for
//! [`crate::dflash_head::BlockDiffusionDraftHead`].
//!
//! The loader reads no embedding or LM head: `BlockDiffusionDraftHead::from_weights`
//! takes the target model's. Tensor names may be bare or `model.`-prefixed.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::weight_map::{DenseWeight, dense};

pub mod ckpt_arch;
mod config;
pub use config::*;

/// 2026-09-25: Load a drafter GEMM weight as BF16. A U8 tensor is packed NVFP4
/// `[n, k/2]` under the same name a BF16 tensor would have; it must be 2-D
/// and is dequantized to BF16 on the GPU. Any other dtype is returned as the
/// store's pointer, unconverted.
fn dense_bf16_or_nvfp4(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let t = store.get(name)?;
    if t.dtype == metrale_model_weights::weights::WeightDtype::UInt8 {
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        anyhow::ensure!(
            t.shape.len() == 2,
            "NVFP4-packed drafter tensor {name} has shape {:?} (want 2-D)",
            t.shape
        );
        let (n, half_k) = (t.shape[0], t.shape[1]);
        tracing::info!(
            "DFlash drafter: dequantizing NVFP4 {name} [{n}, {}] -> BF16",
            half_k * 2
        );
        return metrale_model_layers::weight_map::fp8_lut::dequant_nvfp4_to_bf16(
            store,
            prefix,
            n,
            half_k * 2,
            gpu,
        );
    }
    Ok(DenseWeight { weight: t.ptr })
}

/// 2026-09-25: The drafter weights `load_dflash_weights` returns. It holds no
/// embedding or LM head.
#[allow(dead_code)]
pub struct DflashWeights {
    pub config: DflashConfig,

    pub fc: DenseWeight,
    pub hidden_norm: DenseWeight,
    pub norm: DenseWeight,

    pub layers: Vec<DflashLayerWeights>,

    /// 2026-09-25: `Some` (and empty) when the store has a `d2t` or
    /// `draft_id_to_target_id` tensor; the table itself is not loaded.
    pub draft_id_to_target_id: Option<Vec<i64>>,

    /// 2026-09-25: `markov_head.markov_w1.weight`. `markov_w1` and `markov_w2` are
    /// `Some` when `markov_rank > 0` and the store has `markov_w1`.
    pub markov_w1: Option<DenseWeight>,
    /// 2026-09-25: `markov_head.markov_w2.weight`.
    pub markov_w2: Option<DenseWeight>,
    /// 2026-09-25: `confidence_head.proj.weight`. The weight and the bias are
    /// `Some` when `enable_confidence_head` is set and the store has the weight.
    pub confidence_proj: Option<DenseWeight>,
    /// 2026-09-25: `confidence_head.proj.bias`.
    pub confidence_bias: Option<DenseWeight>,

    /// 2026-09-25: `candidate_selector.predecessor_codebook`. The three selector
    /// tensors are `Some` when `dflash_config` sets `selector_rank` and
    /// `selector_top_k` and the store has this codebook.
    pub selector_pred: Option<DenseWeight>,
    /// 2026-09-25: `candidate_selector.successor_codebook`.
    pub selector_succ: Option<DenseWeight>,
    /// 2026-09-25: `candidate_selector.hidden_projection.weight`.
    pub selector_hidden_proj: Option<DenseWeight>,
}

/// 2026-09-25: One drafter layer's weights. The projections go through
/// `dense_bf16_or_nvfp4`; the norms and conv tensors are the store's pointers.
#[allow(dead_code)]
pub struct DflashLayerWeights {
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,

    /// 2026-09-25: `attention_conv.base_kernel`. The four conv tensors are `Some`
    /// on every layer or on none (see `load_dflash_weights`).
    pub attention_conv_base: Option<DenseWeight>,
    /// 2026-09-25: `attention_conv.kernel_projection.weight`.
    pub attention_conv_proj: Option<DenseWeight>,
    /// 2026-09-25: `mlp_conv.base_kernel`.
    pub mlp_conv_base: Option<DenseWeight>,
    /// 2026-09-25: `mlp_conv.kernel_projection.weight`.
    pub mlp_conv_proj: Option<DenseWeight>,
}

/// 2026-09-25: True when the store has `fc.weight` or `model.fc.weight`. Loads
/// nothing.
pub fn store_has_dflash_weights(store: &WeightStore) -> bool {
    store.contains("fc.weight") || store.contains("model.fc.weight")
}

/// 2026-09-25: Parse a drafter's `config.json` into a [`DflashConfig`]. The
/// error is "Parsing DFlash drafter config.json" with the serde error as its
/// cause.
pub fn parse_dflash_config(json: &str) -> Result<DflashConfig> {
    serde_json::from_str(json).context("Parsing DFlash drafter config.json")
}

/// 2026-09-25: Load the drafter's weights from the drafter's own
/// [`WeightStore`].
///
/// Returns `Ok(None)` when `store_has_dflash_weights` is false. Otherwise
/// `fc`, the two model norms and `num_hidden_layers` layers of norms and
/// projections are required; the conv, selector, Markov and confidence
/// tensors are optional as their fields describe. `_tp_size` is unused, so
/// every rank loads the whole drafter.
pub fn load_dflash_weights(
    drafter_store: &WeightStore,
    drafter_config: &DflashConfig,
    _gpu: &dyn GpuBackend,
    _tp_size: usize,
) -> Result<Option<DflashWeights>> {
    if !store_has_dflash_weights(drafter_store) {
        tracing::debug!("DFlash drafter store has no `fc.weight` — skipping");
        return Ok(None);
    }

    let prefix = if drafter_store.contains("model.fc.weight") {
        "model."
    } else {
        ""
    };

    let fc = dense_bf16_or_nvfp4(drafter_store, &format!("{prefix}fc.weight"), _gpu)
        .context("DFlash drafter: load fc.weight")?;
    let hidden_norm = dense(drafter_store, &format!("{prefix}hidden_norm.weight"))
        .context("DFlash drafter: load hidden_norm.weight")?;
    let norm = dense(drafter_store, &format!("{prefix}norm.weight"))
        .context("DFlash drafter: load norm.weight")?;

    // 2026-09-25: The conv tensors load on every layer when `dflash_config`
    // sets `conv_kernel_size` and `conv_group_size` and layer 0 has
    // `attention_conv.base_kernel`; a later layer missing one is an error.
    let dflash2_conv = drafter_config
        .dflash_config
        .as_ref()
        .map(|c| c.conv_kernel_size > 0 && c.conv_group_size > 0)
        .unwrap_or(false)
        && drafter_store.contains(&format!("{prefix}layers.0.attention_conv.base_kernel"));

    let layer_count = drafter_config.num_hidden_layers;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let lp = format!("{prefix}layers.{i}");
        let (attention_conv_base, attention_conv_proj, mlp_conv_base, mlp_conv_proj) =
            if dflash2_conv {
                (
                    Some(dense(
                        drafter_store,
                        &format!("{lp}.attention_conv.base_kernel"),
                    )?),
                    Some(dense(
                        drafter_store,
                        &format!("{lp}.attention_conv.kernel_projection.weight"),
                    )?),
                    Some(dense(drafter_store, &format!("{lp}.mlp_conv.base_kernel"))?),
                    Some(dense(
                        drafter_store,
                        &format!("{lp}.mlp_conv.kernel_projection.weight"),
                    )?),
                )
            } else {
                (None, None, None, None)
            };
        let layer = DflashLayerWeights {
            input_layernorm: dense(drafter_store, &format!("{lp}.input_layernorm.weight"))?,
            post_attention_layernorm: dense(
                drafter_store,
                &format!("{lp}.post_attention_layernorm.weight"),
            )?,
            q_proj: dense_bf16_or_nvfp4(
                drafter_store,
                &format!("{lp}.self_attn.q_proj.weight"),
                _gpu,
            )?,
            k_proj: dense_bf16_or_nvfp4(
                drafter_store,
                &format!("{lp}.self_attn.k_proj.weight"),
                _gpu,
            )?,
            v_proj: dense_bf16_or_nvfp4(
                drafter_store,
                &format!("{lp}.self_attn.v_proj.weight"),
                _gpu,
            )?,
            o_proj: dense_bf16_or_nvfp4(
                drafter_store,
                &format!("{lp}.self_attn.o_proj.weight"),
                _gpu,
            )?,
            q_norm: dense(drafter_store, &format!("{lp}.self_attn.q_norm.weight"))?,
            k_norm: dense(drafter_store, &format!("{lp}.self_attn.k_norm.weight"))?,
            gate_proj: dense_bf16_or_nvfp4(
                drafter_store,
                &format!("{lp}.mlp.gate_proj.weight"),
                _gpu,
            )?,
            up_proj: dense_bf16_or_nvfp4(drafter_store, &format!("{lp}.mlp.up_proj.weight"), _gpu)?,
            down_proj: dense_bf16_or_nvfp4(
                drafter_store,
                &format!("{lp}.mlp.down_proj.weight"),
                _gpu,
            )?,
            attention_conv_base,
            attention_conv_proj,
            mlp_conv_base,
            mlp_conv_proj,
        };
        layers.push(layer);
    }

    // 2026-09-25: The two codebook tensor names have no `.weight` suffix.
    let selector_key = format!("{prefix}candidate_selector.predecessor_codebook");
    let (selector_pred, selector_succ, selector_hidden_proj) = if drafter_config
        .dflash_config
        .as_ref()
        .map(|c| c.selector_rank > 0 && c.selector_top_k > 0)
        .unwrap_or(false)
        && drafter_store.contains(&selector_key)
    {
        (
            Some(
                dense(drafter_store, &selector_key)
                    .context("DFlash2: load candidate_selector.predecessor_codebook")?,
            ),
            Some(
                dense(
                    drafter_store,
                    &format!("{prefix}candidate_selector.successor_codebook"),
                )
                .context("DFlash2: load candidate_selector.successor_codebook")?,
            ),
            Some(
                dense(
                    drafter_store,
                    &format!("{prefix}candidate_selector.hidden_projection.weight"),
                )
                .context("DFlash2: load candidate_selector.hidden_projection.weight")?,
            ),
        )
    } else {
        (None, None, None)
    };
    if dflash2_conv || selector_pred.is_some() {
        tracing::info!(
            "DFlash2 heads loaded: convs={} (k={}, group={}), selector={} (rank={}, top_k={})",
            dflash2_conv,
            drafter_config
                .dflash_config
                .as_ref()
                .map(|c| c.conv_kernel_size)
                .unwrap_or(0),
            drafter_config
                .dflash_config
                .as_ref()
                .map(|c| c.conv_group_size)
                .unwrap_or(0),
            selector_pred.is_some(),
            drafter_config
                .dflash_config
                .as_ref()
                .map(|c| c.selector_rank)
                .unwrap_or(0),
            drafter_config
                .dflash_config
                .as_ref()
                .map(|c| c.selector_top_k)
                .unwrap_or(0),
        );
    }

    let draft_id_to_target_id = if drafter_store.contains(&format!("{prefix}d2t"))
        || drafter_store.contains(&format!("{prefix}draft_id_to_target_id"))
    {
        tracing::warn!(
            "DFlash drafter has draft-id→target-id mapping; remapping path is not yet wired (Phase 2.5 follow-up)"
        );
        Some(Vec::new())
    } else {
        None
    };

    let markov_key = format!("{prefix}markov_head.markov_w1.weight");
    let (markov_w1, markov_w2) =
        if drafter_config.markov_rank > 0 && drafter_store.contains(&markov_key) {
            if let Some(kind) = drafter_config.markov_head_type.as_deref()
                && kind != "vanilla"
            {
                anyhow::bail!(
                    "DSpark drafter declares markov_head_type={kind:?}; only \"vanilla\" \
                 (low-rank bigram bias) is defined by the reference implementation"
                );
            }
            let w1 = dense(drafter_store, &markov_key)
                .context("DSpark drafter: load markov_head.markov_w1.weight")?;
            let w2 = dense(
                drafter_store,
                &format!("{prefix}markov_head.markov_w2.weight"),
            )
            .context("DSpark drafter: load markov_head.markov_w2.weight")?;
            (Some(w1), Some(w2))
        } else {
            if drafter_config.markov_rank > 0 {
                tracing::warn!(
                    "DSpark drafter config declares markov_rank={} but the checkpoint has \
                 no {markov_key} — running as plain DFlash (Markov bias disabled)",
                    drafter_config.markov_rank,
                );
            }
            (None, None)
        };
    let conf_key = format!("{prefix}confidence_head.proj.weight");
    let (confidence_proj, confidence_bias) =
        if drafter_config.enable_confidence_head && drafter_store.contains(&conf_key) {
            let w = dense(drafter_store, &conf_key)
                .context("DSpark drafter: load confidence_head.proj.weight")?;
            let b = dense(drafter_store, &format!("{prefix}confidence_head.proj.bias"))
                .context("DSpark drafter: load confidence_head.proj.bias")?;
            (Some(w), Some(b))
        } else {
            (None, None)
        };
    if markov_w1.is_some() || confidence_proj.is_some() {
        tracing::info!(
            "DSpark heads loaded: markov={} (rank={}), confidence={} (with_markov={})",
            markov_w1.is_some(),
            drafter_config.markov_rank,
            confidence_proj.is_some(),
            drafter_config.confidence_head_with_markov,
        );
    }

    tracing::info!(
        "DFlash drafter loaded: {} layers, hidden={}, vocab={}, γ={}, target_layers={:?}",
        layers.len(),
        drafter_config.hidden_size,
        drafter_config.vocab_size,
        drafter_config.block_size,
        drafter_config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.as_slice())
            .unwrap_or(&[]),
    );

    Ok(Some(DflashWeights {
        config: drafter_config.clone(),
        fc,
        hidden_norm,
        norm,
        layers,
        draft_id_to_target_id,
        markov_w1,
        markov_w2,
        confidence_proj,
        confidence_bias,
        selector_pred,
        selector_succ,
        selector_hidden_proj,
    }))
}

#[cfg(test)]
#[path = "dflash_loader/loader_tests.rs"]
mod loader_tests;
