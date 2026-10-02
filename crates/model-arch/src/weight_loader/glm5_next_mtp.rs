// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Load GLM-5.3's MTP block, `model.language_model.layers.{num_hidden_layers}`,
//! as the body of the draft proposer.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants:
//! - The block is one DSA mixer and one routed MoE with `mhc: None`, the plain-residual path
//!   of [`crate::glm5next_layer::Glm5NextLayer`].
//! - Its DSA layer uses `attn_layer_idx = 0` of the drafter's own one-layer KV pool
//!   (`glm5next_mtp_head`), never the target's.
//! - `Glm5NextWeightLoader::prune_after_load` keeps this layer's tensors: `is_reuploaded`
//!   matches only layer indices below `num_hidden_layers`.

use anyhow::{Context, Result};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use crate::glm5next_dsa::build::build_dsa_weights;
use crate::glm5next_dsa::layer::{Glm5NextDsaLayer, Glm5NextDsaLayerKernels, Glm5NextDsaWorkspace};
use crate::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use crate::glm5next_layer::{Glm5NextLayer, Glm5NextMixer, Glm5NextMlpSite};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels, build as mlp_build};
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-09-25: The MTP block plus the four tensors around it.
pub struct Glm5NextMtpModule {
    /// 2026-09-25: The block as a `mhc: None` layer: DSA mixer, routed MoE, plain residual.
    pub layer: Glm5NextLayer,
    /// 2026-09-25: `[hidden, 2 * hidden]`: projects `concat(enorm(embed), hnorm(hidden))`
    /// back to `hidden` (`glm5next_mtp_head.rs`).
    pub eh_proj: DenseWeight,
    /// 2026-09-25: RMSNorm weights for the embedding half and the target-hidden half of the
    /// concat.
    pub enorm: DevicePtr,
    pub hnorm: DevicePtr,
    /// 2026-09-25: `shared_head.norm`, the final norm before the target's `lm_head`.
    pub final_norm: DevicePtr,
}

/// 2026-09-25: Build the MTP block, or `None` when the checkpoint has no
/// `layers.{num_hidden_layers}`.
///
/// Runs on every rank: `build_moe` binds only this rank's `local_expert_range()`, so each
/// rank holds its own share of the routed experts, as in the text stack.
pub fn load_glm5next_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<Glm5NextMtpModule>> {
    let idx = config.num_hidden_layers;
    let prefix = format!("model.language_model.layers.{idx}.");
    if !store.names().any(|n| n.starts_with(&prefix)) {
        return Ok(None);
    }
    let src = super::glm5_next_load::layer_source(gpu, store, idx)
        .with_context(|| format!("glm5_next MTP: collecting layer {idx}"))?;
    let load = |n: &str| src.f32(n);

    let dsa_cfg = Glm5NextDsaConfig::from_config(config)?;
    let mlp_cfg = Glm5NextMlpConfig::from_config(config)?;
    let dsa_plan = crate::glm5next_dsa::tp::DsaTpPlan::new(
        config.tp_rank,
        config.tp_world_size.max(1),
        &dsa_cfg,
    )?;
    let dsa_layer_kernels = Glm5NextDsaLayerKernels::resolve(gpu)?;
    let dsa_kernels = Glm5NextDsaKernels::resolve(gpu)?;
    let mlp_kernels = Glm5NextMlpKernels::resolve(gpu)?;

    let mixer = Glm5NextMixer::Dsa(Box::new(Glm5NextDsaLayer {
        persist_bt: std::env::var("METRALE_GLM_DSA_ALLOC_PER_STEP").as_deref() != Ok("1"),
        cfg: dsa_cfg,
        weights: build_dsa_weights(gpu, &dsa_cfg, &dsa_plan, &load)?,
        kernels: dsa_layer_kernels,
        select_kernels: dsa_kernels,
        decode_kernel: crate::glm5next_dsa::attend::Glm5NextDsaDecodeKernel::resolve(gpu)?,
        // 2026-09-25: Single-row workspace: the drafter runs this layer one row at a time.
        workspace: Glm5NextDsaWorkspace::new(gpu, &dsa_cfg, 1)?,
        layer_idx: idx,
        // 2026-09-25: Sole consumer of its own one-layer KV pool.
        attn_layer_idx: 0,
        rms_eps: config.rms_norm_eps as f32,
        kv_scale: 1.0,
    }));

    let expert = |id: usize| super::glm5_next_load::bind_expert_at(gpu, store, idx, id);
    let mlp = Glm5NextMlpSite::Moe(Box::new(mlp_build::build_moe(
        gpu,
        &mlp_cfg,
        config.tp_rank,
        config.shared_expert_intermediate_size,
        &load,
        &expert,
    )?));

    let up =
        |n: &str| -> Result<DevicePtr> { super::glm5_next_load::upload_bf16(gpu, &src.f32(n)?) };
    Ok(Some(Glm5NextMtpModule {
        layer: Glm5NextLayer {
            layer_idx: idx,
            mixer,
            mlp,
            mlp_cfg,
            mlp_kernels,
            // 2026-09-25: Its own one-row workspace, not the text stack's shared one, which
            // is sized for a prefill sub-chunk.
            mlp_ws: std::sync::Arc::new(crate::glm5next_mlp::forward::Glm5NextMlpWorkspace::new(
                gpu, &mlp_cfg, 1,
            )?),
            // 2026-09-25: No hyper-connection: nothing here binds `hc_*` tensors, and
            // `mhc: None` selects the plain residual path.
            mhc: None,
            input_norm: up("input_layernorm.weight")?,
            post_attn_norm: up("post_attention_layernorm.weight")?,
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            add_k: metrale_model_layers::layers::try_kernel(gpu, "bf16_add", "bf16_add_inplace"),
            rms_eps: config.rms_norm_eps as f32,
            hidden: config.hidden_size,
            // 2026-09-25: Row-parallel `o_proj`, as in the DSA text layers.
            mixer_all_reduce: dsa_plan.needs_output_all_reduce(),
            // 2026-09-25: No mHC highway to expand or collapse.
            is_first: false,
            is_last: false,
            // 2026-10-01: No decode L2 prefetch plan: the MTP block runs the plain path.
            prefetch: Default::default(),
            // 2026-10-01: The MTP block is never a DFlash tap.
            dflash_tap: false,
        },
        eh_proj: DenseWeight {
            weight: store.get(&format!("{prefix}eh_proj.weight"))?.ptr,
        },
        enorm: up("enorm.weight")?,
        hnorm: up("hnorm.weight")?,
        final_norm: up("shared_head.norm.weight")?,
    }))
}
