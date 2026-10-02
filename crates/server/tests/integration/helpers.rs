// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Helpers for the GPU integration tests in `integration.rs`: the
//! model directory, model setup, greedy generation, the chat tokenizer and a
//! free-device-memory reading. No test lives here, so the test names stay at
//! the crate root.
//!
//! Owner: server tests.
//! Invariants: `tracing` events log under the `integration` target, as they
//! did when this code was in `integration.rs`.

use anyhow::Result;
use std::path::Path;

/// 2026-09-26: The snapshot used when `METRALE_INTEGRATION_MODEL_DIR` is
/// unset.
const DEFAULT_MODEL_DIR: &str = "~/.cache/huggingface/hub/models--nvidia--Qwen3-Next-80B-A3B-Instruct-NVFP4/snapshots/8fb2682f136cf94d932a498f18cb1e428832a912";

/// 2026-09-26: The model dir from `METRALE_INTEGRATION_MODEL_DIR`, else
/// `DEFAULT_MODEL_DIR`, with a leading `~/` expanded from `HOME`.
pub(super) fn model_dir_path() -> std::path::PathBuf {
    let raw = std::env::var("METRALE_INTEGRATION_MODEL_DIR")
        .unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string());
    if let Some(rest) = raw.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return std::path::PathBuf::from(home).join(rest);
    }
    std::path::PathBuf::from(raw)
}

/// 2026-09-26: Build the GPU backend and the model from a model directory.
pub(super) fn setup_model(
    model_dir: &Path,
) -> Result<(
    Box<dyn metrale_model_engine::traits::Model>,
    metrale_config::ModelConfig,
)> {
    let config_path = model_dir.join("config.json");
    let config_json = std::fs::read_to_string(&config_path)?;
    // 2026-09-26: `parse_config`, the parser the server uses
    // (`serve_phases/config.rs`), handles a config nested under
    // `text_config`, which raw serde into `ModelConfig` does not.
    let mut config = metrale_config::parse_config(&config_json)?;
    tracing::info!(target: "integration", "Config: {} layers, vocab={}, hidden={}",
        config.num_hidden_layers,
        config.vocab_size,
        config.hidden_size
    );

    let ptx_modules = metrale_kernels::ptx_modules();
    let gpu: Box<dyn metrale_gpu_runtime::gpu::GpuBackend> = Box::new(
        metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &ptx_modules)?,
    );
    let total = gpu.total_memory()?;
    let free = gpu.free_memory()?;
    tracing::info!(target: "integration", "GPU: {:.1} GB total, {:.1} GB free",
        total as f64 / (1 << 30) as f64,
        free as f64 / (1 << 30) as f64,
    );

    let loader = metrale_model_weights::weights::SafetensorsLoader {
        ep_rank: 0,
        ep_world_size: 1,
        num_experts: 0,
        peak_memory_multiplier: None,
        skip_activation_scales: false,
        skip_mtp: false,
        skip_layers_from: None,
        skip_suffixes: &[],
        defer: None,
    };
    use metrale_model_weights::weights::WeightLoader;
    let store = loader.load(model_dir, gpu.as_ref(), 1024 * 1024 * 1024)?;
    tracing::info!(target: "integration", "Loaded {} tensors ({:.2} GB)",
        store.len(),
        store.total_bytes() as f64 / (1 << 30) as f64,
    );

    let post_weight_free = gpu.free_memory()?;
    let kv_budget = (post_weight_free as f64 * 0.85) as usize;
    let block_size = 16;
    let kv_config = metrale_cache::kv_cache::KvCacheConfig {
        block_size,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        num_layers: config.num_attention_layers(),
        dtype: metrale_cache::kv_cache::KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
        v_aliases_k: false,
    };
    let num_blocks =
        metrale_cache::kv_cache::PagedKvCache::compute_num_blocks(&kv_config, kv_budget)?;
    tracing::info!(target: "integration", "KV cache: {} blocks", num_blocks);

    let prefix_cache: Box<dyn metrale_telemetry::prefix_cache::PrefixCache> =
        Box::new(metrale_telemetry::prefix_cache::NoPrefixCaching);
    // 2026-09-26: A nested checkpoint stores its weights under
    // `model.language_model.`; the server calls the same function
    // (`serve_load.rs`).
    metrale_model_weights::weights::auto_detect_weight_prefix(&store, &mut config);
    let model = metrale_model_engine::factory::build_model(
        config.clone(),
        store,
        gpu,
        4,
        block_size,
        4096,
        8,
        metrale_model_layers::layers::MtpQuantization::Nvfp4,
        false,
        prefix_cache,
        0,
        None,
        false,
        1,
        metrale_cache::kv_cache::KvCacheDtype::Fp8,
        1024 * 1024 * 1024,
        0.90,
        0,
        Vec::new(),
        0,
        None,
        None,
        None,
        None,
        None,
    )?;

    Ok((model, config))
}

/// 2026-09-26: Greedy generation from a prompt; returns the generated ids and
/// the decode rate in tokens per second.
pub(super) fn generate(
    model: &dyn metrale_model_engine::traits::Model,
    config: &metrale_config::ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
) -> Result<(Vec<u32>, f64)> {
    let mut seq = model.alloc_sequence()?;
    let eos = config.eos_token_id;

    let logits = model.prefill(prompt_tokens, &mut seq, 0)?;
    let first_token = model.argmax_on_device(logits, 0)?;

    let mut generated = vec![first_token];
    if first_token == eos {
        return Ok((generated, 0.0));
    }

    let start = std::time::Instant::now();
    for _ in 1..max_new_tokens {
        let last = *generated.last().unwrap();
        let logits = model.decode(last, &mut seq, 0)?;
        let token = model.argmax_on_device(logits, 0)?;
        generated.push(token);
        if token == eos {
            break;
        }
    }
    let elapsed = start.elapsed();
    let decode_tokens = generated.len().saturating_sub(1).max(1);
    let tok_per_sec = decode_tokens as f64 / elapsed.as_secs_f64();

    Ok((generated, tok_per_sec))
}

/// 2026-09-26: Free device memory as the system reports it, not as the
/// engine's allocator does: a leak check that asked the allocator under test
/// would pass on a bug in that same accounting. The caller has already
/// dropped the model and its backend.
///
/// `nvidia-smi --query-gpu=memory.free` when it answers with a number;
/// otherwise `MemAvailable` from `/proc/meminfo`, which is the GPU's pool on a
/// unified-memory part.
pub(super) fn free_device_memory_bytes() -> Result<usize> {
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        && out.status.success()
        && let Some(mib) = String::from_utf8_lossy(&out.stdout)
            .lines()
            .next()
            .and_then(|l| l.trim().parse::<usize>().ok())
    {
        return Ok(mib * 1024 * 1024);
    }
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let kb: usize = meminfo
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| anyhow::anyhow!("no MemAvailable in /proc/meminfo"))?;
    Ok(kb * 1024)
}

/// 2026-09-26: The chat tokenizer for `model_dir`, built from the model's
/// EOS id and model type.
pub(super) fn chat_tokenizer(
    model_dir: &Path,
    config: &metrale_config::ModelConfig,
) -> Result<metrale_server::tokenizer::ChatTokenizer> {
    use metrale_server::tokenizer::ChatTokenizer;
    let tokenizer = ChatTokenizer::from_model_dir(
        model_dir,
        config.eos_token_id,
        false,
        &config.model_type,
        None,
        false,
    )?;
    Ok(tokenizer)
}
