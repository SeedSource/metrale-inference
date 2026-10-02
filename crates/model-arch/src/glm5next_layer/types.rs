// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The data of a bound GLM-5.3 layer: `Glm5NextLayer` and its mixer, MLP site and
//! hyper-connection types.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: The layer's mixer. KDA's `decode` is an inherent method that takes a
/// `KdaSeqState` and a workspace and leaves its output in `ws.final_out`; DSA's `decode` is the
/// `TransformerLayer` method and writes its output over the buffer it was given.
/// `Glm5NextLayer::mixer_forward` handles both.
pub enum Glm5NextMixer {
    Kda {
        layer: Box<Glm5NextKdaLayer>,
        /// 2026-09-25: One workspace shared by every KDA layer of the stack: the loader builds
        /// one and clones the `Arc`.
        ws: Arc<Glm5NextKdaWorkspace>,
        cfg: Glm5NextKdaConfig,
    },
    Dsa(Box<Glm5NextDsaLayer>),
}

/// 2026-09-25: The layer's MLP: dense for the layers the config lists in `mlp_only_layers`,
/// routed MoE for the rest.
pub enum Glm5NextMlpSite {
    Dense(Glm5NextDenseMlpWeights),
    Moe(Box<Glm5NextMoeWeights>),
}

/// 2026-09-25: The layer's hyper-connection: the kernels, each site's weights, and the config's
/// `hc_mult`, `hc_sinkhorn_iters` and `hc_eps`.
pub struct Glm5NextMhc {
    pub kernels: Glm5NextMhcKernels,
    pub attn: Glm5NextMhcSiteWeights,
    pub ffn: Glm5NextMhcSiteWeights,
    pub hc_mult: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
}

/// 2026-09-25: One bound GLM-5.3 decoder layer.
pub struct Glm5NextLayer {
    pub layer_idx: usize,
    pub mixer: Glm5NextMixer,
    pub mlp: Glm5NextMlpSite,
    pub mlp_cfg: Glm5NextMlpConfig,
    pub mlp_kernels: Glm5NextMlpKernels,
    /// 2026-09-25: MLP scratch. The text layers share one workspace unless
    /// `METRALE_GLM_MLP_WS_SHARED=0` (`mlp_ws_shared`); the MTP block builds its own.
    pub mlp_ws: Arc<Glm5NextMlpWorkspace>,
    /// 2026-09-25: `None` for the MTP block, which runs `forward_one_plain`. Every text layer has
    /// one: the skeleton marks every text layer `hyper_connection`.
    pub mhc: Option<Glm5NextMhc>,
    /// 2026-09-25: `input_layernorm.weight`; `post_attn_norm` is `post_attention_layernorm.weight`.
    pub input_norm: DevicePtr,
    pub post_attn_norm: DevicePtr,
    /// 2026-09-25: `rms_norm_vanilla`; see the module invariants.
    pub rms_norm_k: KernelHandle,
    /// 2026-09-25: `bf16_add_inplace`, the residual add of the MTP block, whose residual is not
    /// in a highway. `0` on a target without the kernel; `add_inplace` then returns an error.
    pub add_k: KernelHandle,
    pub rms_eps: f32,
    pub hidden: usize,
    /// 2026-09-25: The mixer output is a partial sum across TP ranks, all-reduced before
    /// `hc_post` (or, in the MTP block, before the residual add). Set from the mixer's TP plan
    /// (`needs_output_all_reduce`).
    pub mixer_all_reduce: bool,
    /// 2026-09-25: Expand the highway in this layer; true for layer 0 only.
    pub is_first: bool,
    /// 2026-09-25: Collapse the highway in this layer; true for the last text layer only.
    pub is_last: bool,
    /// 2026-10-01: This layer is a DFlash drafter tap (`config.dflash_capture_layers`, under
    /// `METRALE_GLM_DFLASH=1`): after its FFN-site `hc_post` it also writes `hc_head_mean` of the
    /// highway into `hidden`, the tensor the drafter was trained on (mean of the `hc_mult`
    /// streams of the layer's completed output). False for the MTP block and without DFlash.
    pub dflash_tap: bool,
}
