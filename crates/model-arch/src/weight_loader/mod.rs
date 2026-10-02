// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The [`ModelWeightLoader`] trait, and one loader per architecture that turns a
//! flat [`WeightStore`] into typed [`TransformerLayer`]s.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

pub mod deepseek_v4;
pub mod deepseek_v41;
pub mod dflash_loader;
mod gemma4;
/// 2026-09-25: GLM-5.3 checkpoint tensor classification (`classify`).
pub mod glm5_next;
mod laguna;
mod longcat;
mod minimax;
mod nemotron;
mod nllb;
mod qwen3;
mod qwen35;
mod qwen35_dense;
mod qwen3_vl;
#[cfg_attr(test, allow(unreachable_pub))]
pub(crate) mod qwen4_exp;
mod step3p7;

pub use deepseek_v4::DeepSeekV4WeightLoader;
pub use dflash_loader::{
    DflashConfig, DflashExport, DflashLayerWeights, DflashSubConfig, DflashWeights,
    load_dflash_weights, store_has_dflash_weights,
};
pub mod glm5_next_load;
pub mod glm5_next_mtp;
pub mod glm5_next_vision;
pub use gemma4::Gemma4WeightLoader;
pub use glm5_next_load::Glm5NextWeightLoader;
pub(crate) use glm5_next_mtp::Glm5NextMtpModule;

pub use laguna::LagunaWeightLoader;
pub use longcat::LongcatWeightLoader;
pub use minimax::MinimaxM2WeightLoader;
pub use nemotron::NemotronHWeightLoader;
pub use nllb::NllbWeightLoader;
pub use qwen3::Qwen3WeightLoader;
pub use qwen3_vl::Qwen3VLWeightLoader;
pub use qwen4_exp::Qwen4ExpWeightLoader;
pub use qwen35::Qwen35WeightLoader;
pub use qwen35_dense::Qwen35DenseWeightLoader;
/// 2026-09-25: The native-FP8 dense loader's derived-copy decision table and the
/// shape arithmetic that prices it.
pub use qwen35_dense::fp8_residency;
/// 2026-09-25: That table evaluated from the config before the checkpoint
/// loads; preflight's headroom check reads it.
pub use qwen35_dense::predicted_residency;
pub use step3p7::Step3p7WeightLoader;

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::{DeferHook, WeightStore};

use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::VisionTower;
use metrale_model_layers::weight_map::{
    DenseWeight, MtpWeights, Nvfp4Variant, detect_nvfp4_variant,
};

/// 2026-09-25: Can this box hold the transposed MoE prefill copies for every
/// layer, and does the operator want them? Without them MoE prefill uses the
/// fallback grouped GEMM. The qwen3, qwen3_vl and gemma4 loaders ask here.
///
/// `METRALE_MOE_PREFILL_COPIES=0` forces the fallback. Any other value, or
/// unset, builds them when their NVFP4 bytes for all `num_hidden_layers` fit in
/// the measured free memory less 2 GiB; if free memory cannot be read, they are
/// not built.
pub(crate) fn moe_prefill_copies_fit(config: &ModelConfig, gpu: &dyn GpuBackend) -> bool {
    if std::env::var("METRALE_MOE_PREFILL_COPIES").ok().as_deref() == Some("0") {
        tracing::info!("METRALE_MOE_PREFILL_COPIES=0: MoE prefill uses the fallback grouped GEMM");
        return false;
    }
    let inter = config.moe_intermediate_size;
    let h = config.hidden_size;
    // 2026-09-25: NVFP4: one scale byte per 16 elements beside the packed
    // pairs (half a byte per element).
    let group_size = 16usize;
    let gu_bytes = inter * h / 2 + inter * h / group_size;
    let d_bytes = h * inter / 2 + h * inter / group_size;
    let per_layer = config.num_experts * (2 * gu_bytes + d_bytes);
    let total = per_layer * config.num_hidden_layers;
    let available = gpu.free_memory().unwrap_or(0);
    let headroom = 2 * 1024 * 1024 * 1024;
    let fits = total <= available.saturating_sub(headroom);
    if !fits {
        tracing::warn!(
            "Skipping MoE weight transposition ({:.1} GB needed, {:.1} GB available). \
             Prefill will use fallback grouped GEMM.",
            total as f64 / (1024.0 * 1024.0 * 1024.0),
            available as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    fits
}

/// 2026-09-25: Runtime quantization format of a loader's weights, which the
/// qwen3 and qwen35 loaders resolve from the checkpoint's NVFP4 variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// 2026-09-25: NVFP4 E2M1; every variant other than `Fp8Dequanted`.
    Nvfp4,
    /// 2026-09-25: FP8 E4M3 block-scaled, served natively (`Fp8Dequanted`).
    Fp8,
}

impl QuantFormat {
    /// 2026-09-25: Peak GPU memory as a multiple of on-disk weight bytes, for an
    /// OOM pre-flight estimate.
    pub fn peak_memory_multiplier(&self) -> f64 {
        match self {
            Self::Nvfp4 => 1.15,
            Self::Fp8 => 1.5,
        }
    }
}

/// 2026-09-25: On-disk weight format, derived from `detect_nvfp4_variant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// 2026-09-25: ModelOpt (`Standard`) or compressed-tensors NVFP4 on disk,
    /// and `Bf16Raw` checkpoints, which the loader quantises to NVFP4.
    Nvfp4,
    /// 2026-09-25: Block-scaled FP8 E4M3 on disk (`Fp8Dequanted`).
    Fp8BlockScaled,
    /// 2026-09-25: BF16 on disk. `WeightFormat::detect` never returns it.
    Bf16Dense,
}

impl WeightFormat {
    /// 2026-09-25: Detect the weight format from a [`WeightStore`] by probing
    /// tensor names.
    pub fn detect(store: &WeightStore, config: &ModelConfig) -> Self {
        match detect_nvfp4_variant(store, config) {
            Nvfp4Variant::Fp8Dequanted => Self::Fp8BlockScaled,
            Nvfp4Variant::CompressedTensors | Nvfp4Variant::Standard => Self::Nvfp4,
            // 2026-09-25: A `Bf16Raw` weight is quantized to NVFP4 at load.
            Nvfp4Variant::Bf16Raw => Self::Nvfp4,
        }
    }

    pub fn is_fp8(&self) -> bool {
        matches!(self, Self::Fp8BlockScaled)
    }
}

/// 2026-09-25: Loads weights from a [`WeightStore`] into typed layer objects.
pub trait ModelWeightLoader {
    /// 2026-09-25: Whether this loader's weight slicing honours
    /// `config.tp_world_size` / `config.tp_rank`. There is no default, so every
    /// loader declares it.
    ///
    /// With `--tp-size > 1`, startup (`serve_phases/topology.rs`) refuses a
    /// loader that returns `false`. `weight_loader/minimax.rs` is a loader
    /// that returns `true`.
    fn supports_tp(&self) -> bool;

    /// 2026-09-25: Load all transformer layers from the weight store.
    ///
    /// `layer_kv_dtypes` holds one KV cache dtype per attention layer, indexed
    /// by the layer's 0-based ordinal among the attention layers.
    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>>;

    /// 2026-09-25: Drop store tensors this loader has finished with. `factory::build`
    /// calls it after every `load_*` reader and before the KV budget is
    /// computed, so the freed memory counts.
    ///
    /// The default keeps everything, which is right for a loader that binds
    /// the store's device pointers zero-copy. A loader that uploads its own
    /// copies (a TP shard, a host round-trip, a dtype conversion) overrides it
    /// to free the originals.
    fn prune_after_load(
        &self,
        _store: &mut WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Per-(layer, role) weight precision schedule. The default is
    /// `PrecisionSchedule::default()`, in which every role is `Dtype::Inherit`.
    fn precision_schedule(
        &self,
        _config: &ModelConfig,
    ) -> crate::precision_schedule::PrecisionSchedule {
        crate::precision_schedule::PrecisionSchedule::default()
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight>;

    /// 2026-09-25: Build the n-gram embedding, for an architecture that fuses
    /// hashed n-gram lookups into the input embedding (the LongCat loader).
    /// The default `None` leaves the plain `embed_tokens` gather in place.
    fn load_ngram_embedding(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
        _max_tokens: usize,
    ) -> Result<Option<metrale_model_layers::layers::ngram_embed::NgramEmbedding>> {
        Ok(None)
    }
    /// 2026-09-25: Load the final RMSNorm weight used before the LM head.
    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight>;
    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight>;

    /// 2026-09-25: Load single-module MTP head weights, or `None`.
    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>>;

    /// 2026-09-25: Load every MTP module; `factory::build` calls this one.
    /// The default wraps `load_mtp_weights` (zero or one module). The MiniMax
    /// and Step3p7 loaders override it.
    fn load_mtp_weights_multi(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Vec<MtpWeights>> {
        Ok(self
            .load_mtp_weights(store, config, gpu)?
            .into_iter()
            .collect())
    }

    /// 2026-09-25: Per-attention-layer `(num_kv_heads, head_dim)`, indexed like
    /// `layer_kv_dtypes`, for models whose attention geometry differs per layer
    /// (the Gemma-4 loader). The default is empty: every layer uses the global
    /// dims.
    fn kv_layer_dims(&self, _config: &ModelConfig) -> Vec<(usize, usize)> {
        Vec::new()
    }

    /// 2026-09-25: Load DFlash drafter weights from the drafter's own
    /// `WeightStore`. The default returns `None`, and no loader overrides it:
    /// `factory::build` loads the drafter with the free function
    /// `dflash_loader::load_dflash_weights`.
    fn load_dflash_weights(
        &self,
        _drafter_store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
        _tp_size: usize,
    ) -> Result<Option<DflashWeights>> {
        Ok(None)
    }

    /// 2026-09-25: Load startup-static PEFT LoRA adapters into the LoRA pool,
    /// through `metrale_model_layers::lora::load_lora_adapters_multi`.
    /// `factory::build` calls it before the buffer arena and the KV budget, so
    /// the pool counts as used memory.
    fn load_lora_adapters(
        &self,
        adapters: &[metrale_model_layers::lora::LoraAdapterInput<'_>],
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        max_loras: usize,
        max_lora_rank: usize,
    ) -> Result<Option<metrale_model_layers::lora::LoraWeights>> {
        metrale_model_layers::lora::load_lora_adapters_multi(
            adapters,
            config,
            gpu,
            max_loras,
            max_lora_rank,
        )
        .map(Some)
    }

    /// 2026-09-25: Will this loader bind a vision encoder for a multimodal
    /// checkpoint?
    ///
    /// Default `true`, so a loader that does not override it never loses a
    /// weight. On `false` the checkpoint loader skips the tower's tensors
    /// (`serve_phases/weights.rs`). `factory::build` frees an unbound tower
    /// afterwards either way, keyed off the bind result, so this only lowers
    /// peak memory.
    fn binds_vision_encoder(&self) -> bool {
        true
    }

    /// 2026-09-25: Which checkpoint tensors will this loader read from the host
    /// at bind time, so the checkpoint loader records their location instead
    /// of uploading them?
    ///
    /// Default `None`: upload everything. `serve_phases/weights.rs` asks once,
    /// before the load. A deferred tensor is not skipped: it is recorded in
    /// `WeightStore::deferred`, and the loader that asked reads it. See
    /// [`metrale_model_weights::weights::DeferHook`] for the contract.
    fn defer_predicate(&self, _config: &ModelConfig) -> Option<DeferHook> {
        None
    }

    /// 2026-09-25: Load vision encoder weights; the default returns `None`.
    fn load_vision_encoder(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<VisionTower>> {
        Ok(None)
    }
}
