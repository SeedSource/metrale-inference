// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Model factory: [`loader_for_config`] picks the
//! [`ModelWeightLoader`] for a config's `model_type`, and [`build_model`]
//! builds the model from the config and the weights.
//!
//! Owner: metrale-model-engine.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use metrale_config::ModelConfig;
use metrale_model_weights::weights::WeightStore;

use crate::kimi_k3_loader::KimiK3WeightLoader;
use metrale_model_arch::mistral_loader::MistralWeightLoader;
use metrale_model_arch::weight_loader::LongcatWeightLoader;
use metrale_model_arch::weight_loader::Qwen4ExpWeightLoader;
use metrale_model_arch::weight_loader::{
    DeepSeekV4WeightLoader, DflashConfig, Gemma4WeightLoader, Glm5NextWeightLoader,
    LagunaWeightLoader, MinimaxM2WeightLoader, ModelWeightLoader, NemotronHWeightLoader,
    NllbWeightLoader, Qwen3VLWeightLoader, Qwen3WeightLoader, Qwen35DenseWeightLoader,
    Qwen35WeightLoader, Step3p7WeightLoader,
};

/// 2026-09-25: DFlash drafter inputs for [`build_model`]: the drafter's own
/// [`WeightStore`], its parsed `config.json`, and the CLI overrides for γ and
/// the sliding-window size.
///
/// The server (`serve_phases/weights.rs`) loads the drafter checkpoint and
/// parses its config with
/// `metrale_model_arch::weight_loader::dflash_loader::parse_dflash_config`.
/// [`build_model`] passes them to
/// `metrale_model_arch::dflash_head::BlockDiffusionDraftHead::from_weights`.
pub struct DflashBuildArgs<'a> {
    /// 2026-10-10: `mut`: the install frees the BF16 layer weights once the head runs on
    /// their FP8 mirrors (`BlockDiffusionDraftHead::release_bf16_layer_weights`).
    pub drafter_store: &'a mut WeightStore,
    pub drafter_config: DflashConfig,
    pub gamma: Option<usize>,
    pub window_size: Option<usize>,
}

/// 2026-09-25: LoRA inputs for [`build_model`] (`--lora-adapter NAME=PATH`):
/// the adapters and the pool-shape CLI values.
///
/// [`build_model`] allocates the LoRA pool before the buffer arena and the
/// free-memory sample the KV budget is computed from, so the pool is counted
/// against the budget.
pub struct LoraBuildArgs<'a> {
    /// 2026-09-25: The adapters to pack, one per `--lora-adapter NAME=PATH`;
    /// slot k is `adapters[k]`.
    pub adapters: Vec<metrale_model_layers::lora::LoraAdapterInput<'a>>,
    pub max_lora_rank: usize,
    pub max_loras: usize,
}

/// 2026-09-25: The weight loader for `config.model_type`, matched after
/// lower-casing it and turning `-` and `.` into `_`. An unknown type is an
/// error.
pub fn loader_for_config(config: &ModelConfig) -> Result<Box<dyn ModelWeightLoader>> {
    let normalized = config.model_type.to_lowercase().replace(['-', '.'], "_");
    match normalized.as_str() {
        "kimi_k3" | "kimi_linear" => Ok(Box::new(KimiK3WeightLoader)),
        "qwen3_next" => Ok(Box::new(Qwen3WeightLoader)),
        "qwen3_vl_moe" => Ok(Box::new(Qwen3VLWeightLoader)),
        "qwen3_5_moe" | "qwen3_5" | "qwen35_moe" | "qwen35" => {
            // 2026-09-25: `is_qwen35_dense()` (model_type `qwen3_5` with no
            // experts) is tested first: a dense config can carry a
            // `vision_config`, which `is_qwen3_vl()` also accepts.
            if config.is_qwen35_dense() {
                Ok(Box::new(Qwen35DenseWeightLoader))
            } else if config.is_qwen3_vl() {
                Ok(Box::new(Qwen3VLWeightLoader))
            } else {
                Ok(Box::new(Qwen35WeightLoader))
            }
        }
        "qwen3_6_moe" | "holo3_1_moe" => Ok(Box::new(Qwen35WeightLoader)),
        "nemotron_h" | "nemotron_h_puzzle" => Ok(Box::new(NemotronHWeightLoader)),
        "m2m_100" | "nllb" => Ok(Box::new(NllbWeightLoader)),
        "gemma4" | "gemma_4" => Ok(Box::new(Gemma4WeightLoader)),
        "mistral" => Ok(Box::new(MistralWeightLoader)),
        "longcat_flash_ngram" | "longcat_flash" => Ok(Box::new(LongcatWeightLoader)),
        // 2026-09-25: The config parser also maps `qwen3_8_flash_next` to
        // `qwen4_exp` (metrale-config `dispatch.rs`, `parsers/qwen4_exp.rs`).
        "qwen4_exp" => Ok(Box::new(Qwen4ExpWeightLoader)),
        "minimax_m2" => Ok(Box::new(MinimaxM2WeightLoader)),
        "step3p7" => Ok(Box::new(Step3p7WeightLoader)),
        "laguna" => Ok(Box::new(LagunaWeightLoader)),
        "deepseek_v4" => Ok(Box::new(DeepSeekV4WeightLoader)),
        // 2026-09-25: The config parser sets `glm5_next` for both
        // (metrale-config `parsers/glm5_next/parse.rs`).
        "glm5_next" | "glm5_next_text" => Ok(Box::new(Glm5NextWeightLoader)),
        "deepseek_v41" => Ok(Box::new(
            metrale_model_arch::weight_loader::deepseek_v41::DeepSeekV41WeightLoader,
        )),
        _ => bail!(
            "Unsupported model type: '{}' (normalized: '{}'). \
             Supported: qwen3_next, glm5_next, qwen3_5_moe, qwen3_5, qwen3_6_moe, holo3_1_moe, qwen3_vl_moe, nemotron_h, nemotron_h_puzzle, gemma4, mistral, minimax_m2, step3p7, laguna, deepseek_v4, qwen4_exp, m2m_100, deepseek_v41",
            config.model_type,
            normalized,
        ),
    }
}

mod build;
mod lm_head_setup;
mod m2_setup;

pub use build::build_model;

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_cache::kv_cache::KvCacheDtype;
    use metrale_model_layers::layers::mtp_head::MtpQuantization;
    use metrale_telemetry::prefix_cache::PrefixCache;

    #[test]
    fn test_unsupported_model_type() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "llama".to_string();

        let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
        let store = WeightStore::empty();

        let prefix_cache: Box<dyn PrefixCache> =
            Box::new(metrale_telemetry::prefix_cache::NoPrefixCaching);
        let result = build_model(
            config,
            store,
            Box::new(gpu),
            1,
            16,
            4096,
            8,
            MtpQuantization::Nvfp4,
            false,
            prefix_cache,
            0,
            None,
            false,
            1,
            KvCacheDtype::Fp8,
            1024 * 1024 * 1024,
            0.90,
            0,
            vec![],
            0,
            None,
            None,
            None,
            None,
            None,
        );
        match result {
            Err(e) => assert!(e.to_string().contains("Unsupported model type: 'llama'")),
            Ok(_) => panic!("Expected error for unsupported model type"),
        }
    }

    #[test]
    fn all_declared_model_type_spellings_are_accepted() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        for model_type in [
            "qwen3_next",
            "qwen3_vl_moe",
            "qwen3_5_moe",
            "qwen3_5",
            "qwen35_moe",
            "qwen35",
            "qwen3_6_moe",
            "holo3_1_moe",
            "nemotron_h",
            "nemotron_h_puzzle",
            "m2m_100",
            "nllb",
            "gemma4",
            "gemma_4",
            "mistral",
            "minimax_m2",
            "step3p7",
            "laguna",
            "deepseek_v4",
            // 2026-09-25: Case, hyphens and dots are normalized before the match.
            "QWEN3-NEXT",
            "nemotron.h.puzzle",
            "M2M-100",
        ] {
            config.model_type = model_type.to_string();
            assert!(
                loader_for_config(&config).is_ok(),
                "declared model type {model_type:?} was rejected"
            );
        }

        config.model_type = "unsupported_model".to_string();
        assert!(loader_for_config(&config).is_err());
    }

    // 2026-09-25: `build_model` serves NLLB with `NllbGpuModel` before it asks
    // for a loader, so NLLB's generic loader fails every mandatory load; its
    // optional hooks return their defaults.
    #[test]
    fn nllb_mandatory_generic_loads_fail_fast() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "nllb".to_string();
        let loader = loader_for_config(&config).unwrap();
        let store = WeightStore::empty();
        let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();

        let errors = [
            loader
                .load_layers(&store, &config, &gpu, &[])
                .err()
                .expect("generic layer load unexpectedly succeeded"),
            loader
                .load_embedding(&store, &config, &gpu)
                .expect_err("generic embedding load unexpectedly succeeded"),
            loader
                .load_final_norm(&store, &config, &gpu)
                .expect_err("generic final-norm load unexpectedly succeeded"),
            loader
                .load_lm_head(&store, &config, &gpu)
                .expect_err("generic LM-head load unexpectedly succeeded"),
        ];
        for err in errors {
            assert!(
                err.to_string()
                    .contains("dedicated GPU encoder-decoder runtime"),
                "unexpected fail-fast diagnostic: {err}"
            );
        }
    }
}
