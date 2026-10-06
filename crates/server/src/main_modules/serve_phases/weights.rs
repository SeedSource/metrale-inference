// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Weight-store loading: the checkpoint, the DFlash drafter and
//! the startup LoRA adapters.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use std::path::Path;

use anyhow::{Context, Result};

use metrale_config::ModelConfig;

use crate::cli;

#[cfg(test)]
#[path = "weights_allowlist_tests.rs"]
mod weights_allowlist_tests;
// 2026-10-06: `METRALE_GLM_WEIGHT_ARENA`'s fast-loader hook, in its own file (size cap).
#[path = "weights_arena.rs"]
mod weights_arena;
use weights_arena::arena_hook;

/// 2026-09-26: The load pre-flight's peak-memory multiplier for this model;
/// `None` leaves the loader's own (1.3x, or 1.5x with FP8 tensors).
pub(crate) fn quant_multiplier(config: &ModelConfig) -> Option<f64> {
    if config.model_type == "glm5_next" {
        // 2026-09-26: 1.10 is a headroom allowance, not a measured value.
        Some(1.10)
    } else if config.model_type == "minimax_m2" || config.model_type == "step3p7" {
        Some(1.02)
    } else if config
        .quantization_config
        .as_ref()
        .is_some_and(|qc| qc.quant_method == "fp8")
    {
        Some(1.05)
    } else {
        None
    }
}

pub(crate) fn load_weight_store(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model_dir: &Path,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    ep_rank: usize,
    ep_size: usize,
    oom_reserve_bytes: usize,
) -> Result<metrale_model_weights::weights::WeightStore> {
    use metrale_model_weights::weights::WeightLoader;
    if matches!(
        config.model_type.as_str(),
        "kimi_k3" | "kimi_linear" | "Kimi-K3"
    ) {
        anyhow::ensure!(
            ep_size == 1,
            "K3 rank-aware loading supports TP only; expert parallelism is not implemented"
        );
        anyhow::ensure!(
            metrale_model_weights::weights::find_gguf(model_dir).is_none(),
            "K3 rank-aware loading requires safetensors"
        );
        tracing::info!(
            rank = config.tp_rank,
            world = config.tp_world_size,
            "K3 uses rank-aware safetensors loading before GPU allocation"
        );
        return metrale_model_weights::weights::K3SafetensorsLoader::new(config.clone())?
            .load(model_dir, gpu, oom_reserve_bytes)
            .context("Failed to load rank-local K3 weights");
    }
    let mult = quant_multiplier(config);
    // 2026-10-05: GLM-5.3 expert-TP (`METRALE_GLM_EXPERT_TP=1`) owns every routed expert on every
    // rank, each at half width, so the store must see all of them: it loads at EP 1 (no remote
    // expert skip) and the model's defer rule keeps every routed-expert tensor off the device
    // until `Glm5NextWeightLoader` reads back this rank's slice. `resolve_topology` has already
    // refused the lever off GLM-5.3 TP2/EP2.
    let (ep_rank, ep_size) = if expert_tp_store(config) {
        tracing::info!(
            "METRALE_GLM_EXPERT_TP=1: loading the weight store at EP 1 (rank {ep_rank} of \
             {ep_size} keeps every routed expert, deferred, and binds its half of each)"
        );
        (0, 1)
    } else {
        (ep_rank, ep_size)
    };

    // 2026-09-26: Any `.gguf` file in the directory selects `GgufLoader`.
    if metrale_model_weights::weights::find_gguf(model_dir).is_some() {
        tracing::info!("Detected GGUF weights; using GgufLoader (GPU dequant → BF16)");
        let mut loader = if ep_size > 1 {
            metrale_model_weights::weights::GgufLoader::with_ep(
                ep_rank,
                ep_size,
                config.num_experts,
            )
        } else {
            metrale_model_weights::weights::GgufLoader::new()
        };
        loader.peak_memory_multiplier = mult;
        let store = loader
            .load(model_dir, gpu, oom_reserve_bytes)
            .context("Failed to load model weights (GGUF loader)")?;
        tracing::info!("Loaded {} weight tensors (GGUF)", store.len());
        return Ok(store);
    }

    let use_fast_load =
        !args.no_fast_load && std::env::var("METRALE_FAST_LOAD").ok().as_deref() != Some("0");
    let store = if use_fast_load {
        #[cfg(unix)]
        {
            tracing::info!("Using fast weight loader (O_DIRECT + pipelined read/copy)");
            let mut loader = if ep_size > 1 {
                metrale_model_weights::fast_weights::FastSafetensorsLoader::with_ep(
                    ep_rank,
                    ep_size,
                    config.num_experts,
                )
            } else {
                metrale_model_weights::fast_weights::FastSafetensorsLoader::new()
            };
            loader.peak_memory_multiplier = mult;
            loader.skip_activation_scales = skip_activation_scales(config);
            loader.skip_mtp = skip_mtp(config);
            loader.skip_layers_from = skip_mtp_layers_from(args, config);
            // 2026-09-26: A loader that binds no vision encoder skips the
            // tower here. `factory::build` frees an unbound tower too, but
            // only after the reserve preflight has measured free memory.
            loader.skip_vision = !binds_vision(config);
            if loader.skip_vision {
                tracing::info!(
                    "Vision tower: not loaded — the weight loader for model_type '{}' is a \
                     text-only port and binds no vision encoder.",
                    config.model_type,
                );
            }
            loader.defer = defer_hook(config);
            loader.arena = arena_hook(config);
            loader.prefetch_shards = args.fast_load_prefetch_shards
                || std::env::var("METRALE_FAST_LOAD_PREFETCH_SHARDS")
                    .ok()
                    .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
            if loader.prefetch_shards {
                tracing::info!("Fast weight loader shard prefetch/readahead enabled");
            }
            loader
                .load(model_dir, gpu, oom_reserve_bytes)
                .context("Failed to load model weights (fast loader)")?
        }
        #[cfg(not(unix))]
        {
            anyhow::bail!("--fast-load requires a Unix host (needs O_DIRECT / posix_fadvise)");
        }
    } else {
        let mut loader = if ep_size > 1 {
            metrale_model_weights::weights::SafetensorsLoader::with_ep(
                ep_rank,
                ep_size,
                config.num_experts,
            )
        } else {
            metrale_model_weights::weights::SafetensorsLoader::new()
        };
        loader.peak_memory_multiplier = mult;
        loader.skip_activation_scales = skip_activation_scales(config);
        loader.skip_mtp = skip_mtp(config);
        loader.skip_layers_from = skip_mtp_layers_from(args, config);
        loader.defer = defer_hook(config);
        loader
            .load(model_dir, gpu, oom_reserve_bytes)
            .context("Failed to load model weights")?
    };
    tracing::info!("Loaded {} weight tensors", store.len());
    Ok(store)
}

pub(crate) fn load_dflash_drafter(
    args: &cli::ServeArgs,
    target: &ModelConfig,
    ptx_set: &metrale_kernels::TargetPtxSet,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
) -> Result<
    Option<(
        metrale_model_weights::weights::WeightStore,
        metrale_model_arch::weight_loader::DflashConfig,
    )>,
> {
    use metrale_model_weights::weights::WeightLoader;
    if !args.dflash {
        return Ok(None);
    }
    let drafter_id = args
        .draft_model
        .clone()
        .or_else(|| ptx_set.dflash.as_ref().map(|d| d.draft_model.to_string()))
        .context(
            "--dflash set but no drafter HF id provided: pass --draft-model <ID> \
             or use a target whose MODEL.toml has a [dflash] section",
        )?;
    tracing::info!("DFlash: resolving drafter '{drafter_id}'");
    let drafter_dir =
        crate::model_resolver::resolve_model_dir(&drafter_id, args.cache_dir.as_deref())
            .context("Failed to resolve DFlash drafter checkpoint")?;
    let drafter_config_json = std::fs::read_to_string(drafter_dir.join("config.json"))
        .with_context(|| {
            format!(
                "Failed to read drafter config.json at {}",
                drafter_dir.display()
            )
        })?;
    let mut drafter_config =
        metrale_model_arch::weight_loader::dflash_loader::parse_dflash_config(
            &drafter_config_json,
        )?;
    // 2026-10-01: `METRALE_DFLASH_CKPT_ARCH=1` loads the drafter's learned
    // mask embedding (`mask_embedding.pt`) from the drafter directory into
    // the config the head is built from; the summary line names what the
    // engine does with each architecture field, lever on or off.
    {
        use metrale_model_arch::weight_loader::dflash_loader::ckpt_arch;
        let ckpt_arch_on = ckpt_arch::ckpt_arch_enabled();
        if ckpt_arch_on {
            ckpt_arch::attach_mask_embedding(&drafter_dir, &mut drafter_config)
                .context("METRALE_DFLASH_CKPT_ARCH=1: load the drafter's mask embedding")?;
        }
        ckpt_arch::log_arch_summary(&drafter_config, ckpt_arch_on);
    }
    // 2026-09-26: The drafter allocates outside the KV budget, so its whole
    // footprint is estimated from the checkpoint before anything is allocated:
    // safetensors bytes on disk, the drafter KV cache, `fused_kv`, the
    // DFlash2 selector's host copies, the FP8 mirrors (half the store plus
    // twice `vocab x hidden`) unless `METRALE_DFLASH_DRAFTER_FP8=0`, and
    // 300 MiB of scratch.
    let store_bytes_on_disk: u64 = std::fs::read_dir(&drafter_dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        // 2026-09-26: `std::fs::metadata` follows symlinks and
        // `DirEntry::metadata` does not; a Hugging Face snapshot directory
        // holds symlinks into `blobs/`.
        .filter_map(|e| std::fs::metadata(e.path()).ok().map(|m| m.len()))
        .sum();
    // 2026-10-02: The head drafts through the TARGET's embed/lm_head
    // (`install_dflash_drafter`), so when the drafter's own copies have the
    // target's hidden size (and the drafter's own vocab) they are neither
    // loaded nor counted; GLM-5.3 targets only. Any shape or
    // header-read mismatch loads them as before.
    let shared_tables = if target.model_type == "glm5_next" {
        shareable_table_bytes(&drafter_dir, drafter_config.vocab_size, target.hidden_size)
    } else {
        0
    };
    let store_bytes = store_bytes_on_disk.saturating_sub(shared_tables);
    let c = &drafter_config;
    let kv_dim = c.num_key_value_heads * c.head_dim;
    let drafter_kv =
        (args.max_seq_len as u64) * (c.num_hidden_layers as u64) * 2 * (kv_dim as u64) * 2;
    let fused_kv = (c.num_hidden_layers as u64) * 2 * (kv_dim as u64) * (c.hidden_size as u64) * 2;
    let selector_host = c
        .dflash_config
        .as_ref()
        .map(|d| d.selector_rank)
        .filter(|r| *r > 0)
        .map(|r| {
            2 * (c.vocab_size as u64) * (r as u64) * 2 + (r as u64) * (c.hidden_size as u64) * 2
        })
        .unwrap_or(0);
    // 2026-09-26: The predicate `dflash_head/from_weights.rs` allocates the
    // mirrors on: anything but `METRALE_DFLASH_DRAFTER_FP8=0`.
    let fp8_mirrors = if std::env::var("METRALE_DFLASH_DRAFTER_FP8").ok().as_deref() != Some("0") {
        let lm_head = (c.vocab_size as u64) * (c.hidden_size as u64);
        store_bytes / 2 + 2 * lm_head
    } else {
        0
    };
    let scratch_est: u64 = 300 << 20;
    let estimate = store_bytes + drafter_kv + fused_kv + selector_host + fp8_mirrors + scratch_est;

    let free = gpu.free_memory().unwrap_or(0) as u64;
    let total = gpu.total_memory().unwrap_or(0) as u64;
    // 2026-09-26: `total x (1 - util)`, the memory outside the utilization
    // budget, must still be free after the drafter loads.
    let headroom = (total as f64 * (1.0 - args.gpu_memory_utilization)) as u64;
    if free < estimate + headroom {
        anyhow::bail!(
            "DFlash drafter would over-commit unified memory: estimated footprint {:.2} GB              (weights {:.2} + drafter-KV {:.2} + fused_kv {:.2} + selector-host {:.2} +              fp8-mirrors {:.2} + scratch {:.2}) but only {:.2} GB free with {:.2} GB              headroom pledged by --gpu-memory-utilization {:.2}. On GB10 this would SWAP              the host, not error. Lower --max-seq-len, lower --gpu-memory-utilization              pressure elsewhere, or drop METRALE_DFLASH_DRAFTER_FP8.",
            estimate as f64 / 1e9,
            store_bytes as f64 / 1e9,
            drafter_kv as f64 / 1e9,
            fused_kv as f64 / 1e9,
            selector_host as f64 / 1e9,
            fp8_mirrors as f64 / 1e9,
            scratch_est as f64 / 1e9,
            free as f64 / 1e9,
            headroom as f64 / 1e9,
            args.gpu_memory_utilization,
        );
    }
    tracing::info!(
        "DFlash footprint pre-flight: estimate {:.2} GB (weights {:.2}, drafter-KV {:.2},          fp8-mirrors {:.2}, selector-host {:.2}) vs {:.2} GB free, {:.2} GB headroom — OK",
        estimate as f64 / 1e9,
        store_bytes as f64 / 1e9,
        drafter_kv as f64 / 1e9,
        fp8_mirrors as f64 / 1e9,
        selector_host as f64 / 1e9,
        free as f64 / 1e9,
        headroom as f64 / 1e9,
    );

    let mut loader = metrale_model_weights::weights::SafetensorsLoader::new();
    loader.peak_memory_multiplier = None;
    if shared_tables > 0 {
        loader.skip_suffixes = &["embed_tokens.weight", "lm_head.weight"];
        tracing::info!(
            "DFlash drafter: not loading embed_tokens/lm_head ({:.2} GB): same shape as the \
             target's, which the head uses",
            shared_tables as f64 / 1e9
        );
    }
    let mut drafter_store = loader
        .load(&drafter_dir, gpu, 0)
        .context("Failed to load DFlash drafter weights")?;
    // 2026-10-01: Under `METRALE_GLM_DFLASH=1`, free the drafter's own
    // `embed_tokens` / `lm_head`: the head always drafts through the target's
    // tables (`install_dflash_drafter`) and nothing reads these. For
    // canada-quant GLM-5.3-Flash-DFlash2-G they are byte-identical to the
    // target's and 2.54 GB, on every rank.
    if metrale_model_layers::speculative::glm_dflash::glm_dflash_enabled() {
        let (n, bytes) = drafter_store.free_matching(gpu, |name| {
            name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight")
        })?;
        tracing::info!(
            "DFlash drafter: freed {n} unread table(s) (embed_tokens/lm_head), {:.2} GB; \
             the head uses the target's",
            bytes as f64 / 1e9
        );
    }
    tracing::info!(
        "DFlash drafter store: {} tensors, {} bytes",
        drafter_store.len(),
        drafter_store.total_bytes()
    );
    Ok(Some((drafter_store, drafter_config)))
}

/// 2026-09-26: Whether the target checkpoint's LM head (`lm_head.weight`,
/// with or without a `language_model.` or `model.` prefix) is FP8 E4M3. Reads
/// only the safetensors header of the shard holding it; any failure gives
/// `false`.
fn target_ships_native_fp8_lm_head(args: &cli::ServeArgs) -> bool {
    fn inner(args: &cli::ServeArgs) -> Option<bool> {
        let dir = if let Some(p) = &args.model_from_path {
            p.clone()
        } else {
            crate::model_resolver::resolve_model_dir(
                args.model.as_deref()?,
                args.cache_dir.as_deref(),
            )
            .ok()?
        };
        const KEYS: [&str; 3] = [
            "lm_head.weight",
            "language_model.lm_head.weight",
            "model.lm_head.weight",
        ];
        let shard = if let Ok(idx) = std::fs::read(dir.join("model.safetensors.index.json")) {
            let idx: serde_json::Value = serde_json::from_slice(&idx).ok()?;
            let map = idx.get("weight_map")?;
            KEYS.iter()
                .find_map(|k| map.get(*k).and_then(|v| v.as_str()))
                .map(|s| dir.join(s))?
        } else {
            dir.join("model.safetensors")
        };
        use std::io::Read as _;
        let mut f = std::fs::File::open(shard).ok()?;
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8).ok()?;
        let hlen = u64::from_le_bytes(len8);
        if hlen > 64 << 20 {
            return None;
        }
        let mut hdr = vec![0u8; hlen as usize];
        f.read_exact(&mut hdr).ok()?;
        let hdr: serde_json::Value = serde_json::from_slice(&hdr).ok()?;
        let dtype = KEYS
            .iter()
            .find_map(|k| hdr.get(*k))
            .and_then(|t| t.get("dtype"))
            .and_then(|d| d.as_str())?;
        Some(dtype == "F8_E4M3")
    }
    inner(args).unwrap_or(false)
}

/// 2026-09-26: A LoRA adapter loaded at startup: its own `WeightStore` and its
/// parsed PEFT config, one per `--lora-adapter NAME=PATH`.
pub(crate) struct LoraAdapterState {
    pub name: String,
    pub peft_config: metrale_config::PeftAdapterConfig,
    pub store: metrale_model_weights::weights::WeightStore,
}

/// 2026-09-26: Resolve and load every `--lora-adapter` into its own device
/// `WeightStore`, in command-line order; empty when none is given. Errors on
/// more adapters than `--max-loras` before loading any, and on a repeated
/// name or a rank above `--max-lora-rank` (64 when unset) when it is reached.
pub(crate) fn load_lora_adapters(
    args: &cli::ServeArgs,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
) -> Result<Vec<LoraAdapterState>> {
    if args.lora_adapter.is_empty() {
        return Ok(Vec::new());
    }
    if args.lora_adapter.len() > args.max_loras {
        anyhow::bail!(
            "--lora-adapter given {} times but --max-loras={} (pool has {} slots); \
             raise --max-loras or stage the extras on an $METRALE_LORA_PEER",
            args.lora_adapter.len(),
            args.max_loras,
            args.max_loras,
        );
    }
    let mut states: Vec<LoraAdapterState> = Vec::with_capacity(args.lora_adapter.len());
    for (name, spec) in &args.lora_adapter {
        if states.iter().any(|s| &s.name == name) {
            anyhow::bail!("--lora-adapter name '{name}' given twice (names must be unique)");
        }
        tracing::info!("LoRA: resolving adapter '{name}' from '{spec}'");
        let adapter_dir =
            crate::model_resolver::resolve_adapter_dir(spec, args.cache_dir.as_deref())
                .context("Failed to resolve LoRA adapter")?;
        let cfg_path = adapter_dir.join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("Failed to read {}", cfg_path.display()))?;
        // 2026-09-26: Scaling is per adapter: `lora_alpha / r`, or
        // `lora_alpha / sqrt(r)` with `use_rslora` (`PeftAdapterConfig::scaling`).
        let peft_config = metrale_config::parse_peft_adapter_config(&raw)
            .with_context(|| format!("Failed to parse {}", cfg_path.display()))?;
        let rank_ceiling = args.max_lora_rank.unwrap_or(64);
        if peft_config.r > rank_ceiling {
            anyhow::bail!(
                "LoRA adapter '{}' has r={} > --max-lora-rank {} — raise the flag \
                 (slot pool is rank-padded to it) or use a smaller adapter",
                name,
                peft_config.r,
                rank_ceiling,
            );
        }
        let store =
            metrale_model_weights::weights::adapter::load_adapter_safetensors(&adapter_dir, gpu, 0)
                .context("Failed to load LoRA adapter weights")?;
        tracing::info!(
            "LoRA adapter '{}': {} tensors, {} bytes loaded; r={}, alpha={}, \
             use_rslora={}, scaling={:.6}, target_modules={:?}",
            name,
            store.len(),
            store.total_bytes(),
            peft_config.r,
            peft_config.lora_alpha,
            peft_config.use_rslora,
            peft_config.scaling(),
            peft_config.target_modules,
        );
        states.push(LoraAdapterState {
            name: name.clone(),
            peft_config,
            store,
        });
    }
    Ok(states)
}

/// 2026-09-26: Whether the loaders skip the `*.input_scale` activation scales
/// for this model: `qwen4_exp` and `glm5_next`.
///
/// Both loaders skip exactly the names ending in `.input_scale`
/// (`SafetensorsLoader`, `fast_weights::skip`), so `.weight_scale` and
/// `.weight_scale_2` still load. The skip is listed per model because some
/// loaders read `input_scale`: `step3p7` on its own path, and
/// `weight_map/model_a.rs` whenever the tensor is present.
///
/// 2026-10-03: `glm5_next` keeps the names when
/// `METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1`: its loader then DEFERS every `*.input_scale`
/// (`Glm5NextWeightLoader::defer_predicate`), so none is uploaded, and reads the routed
/// experts' from disk at bind time.
fn skip_activation_scales(config: &ModelConfig) -> bool {
    skip_activation_scales_for(
        config.model_type.as_str(),
        metrale_config::glm_moe_prefill_cutlass_w4a4(),
    )
}

/// 2026-10-03: The policy half of [`skip_activation_scales`], with the GLM CUTLASS W4A4 lever
/// passed in.
fn skip_activation_scales_for(model_type: &str, glm_cutlass_w4a4: bool) -> bool {
    match model_type {
        "qwen4_exp" => true,
        "glm5_next" => !glm_cutlass_w4a4,
        _ => false,
    }
}

/// 2026-10-05: Whether the weight store loads at EP 1 for GLM-5.3 expert-TP
/// (`METRALE_GLM_EXPERT_TP=1`); false for every other model and with the lever off.
fn expert_tp_store(config: &ModelConfig) -> bool {
    matches!(config.model_type.as_str(), "glm5_next" | "glm5_next_text")
        && metrale_config::glm_expert_tp()
}

/// 2026-09-26: Whether `mtp.*` is left unloaded: `qwen4_exp`, whose
/// `Qwen4ExpWeightLoader::load_mtp_weights` returns `None`.
fn skip_mtp(config: &ModelConfig) -> bool {
    matches!(config.model_type.as_str(), "qwen4_exp")
}

/// 2026-10-02: With `--dflash` (which excludes `--speculative`) GLM-5.3's MTP
/// block, `layers.{num_hidden_layers}`, is never bound as the proposer, so its
/// weights (~3 GB per rank under EP) are not loaded. `None` for every other
/// case, so serves without `--dflash` are unchanged.
fn skip_mtp_layers_from(args: &cli::ServeArgs, config: &ModelConfig) -> Option<usize> {
    (args.dflash
        && !args.speculative
        && config.model_type == "glm5_next"
        && !config.mtp_layer_types.is_empty())
    .then_some(config.num_hidden_layers)
}

/// 2026-10-02: One safetensors tensor header: name, shape, byte length.
type TensorHeader = (String, Vec<u64>, u64);

/// 2026-10-02: Bytes of the drafter's `embed_tokens.weight` / `lm_head.weight`
/// that can be left unloaded because they are `[vocab, hidden]` with the
/// target's hidden size; 0 when any such tensor differs, or a header cannot be
/// read. Reads only safetensors headers.
fn shareable_table_bytes(dir: &Path, vocab: usize, hidden: usize) -> u64 {
    let mut headers: Vec<TensorHeader> = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    for e in rd.flatten() {
        if !e.path().extension().is_some_and(|x| x == "safetensors") {
            continue;
        }
        match read_safetensors_headers(&e.path()) {
            Some(h) => headers.extend(h),
            None => return 0,
        }
    }
    shareable_bytes_from_headers(&headers, vocab, hidden)
}

fn read_safetensors_headers(path: &Path) -> Option<Vec<TensorHeader>> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8).ok()?;
    let hlen = u64::from_le_bytes(len8);
    if hlen > 64 << 20 {
        return None;
    }
    let mut buf = vec![0u8; hlen as usize];
    f.read_exact(&mut buf).ok()?;
    let hdr: serde_json::Value = serde_json::from_slice(&buf).ok()?;
    let mut out = Vec::new();
    for (name, t) in hdr.as_object()? {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<u64> = t
            .get("shape")?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64())
            .collect();
        let off = t.get("data_offsets")?.as_array()?;
        let bytes = off.get(1)?.as_u64()?.checked_sub(off.first()?.as_u64()?)?;
        out.push((name.clone(), shape, bytes));
    }
    Some(out)
}

/// 2026-10-02: The sum of the table tensors' bytes if every table is
/// `[vocab, hidden]`, else 0 (nothing is shared, everything loads).
fn shareable_bytes_from_headers(headers: &[TensorHeader], vocab: usize, hidden: usize) -> u64 {
    let mut total = 0u64;
    for (name, shape, bytes) in headers {
        if !(name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight")) {
            continue;
        }
        if shape.as_slice() != [vocab as u64, hidden as u64] {
            return 0;
        }
        total += bytes;
    }
    total
}

/// 2026-09-26: Whether the model's weight loader binds a vision encoder;
/// `true` when the model type does not resolve to a loader.
fn binds_vision(config: &metrale_config::ModelConfig) -> bool {
    metrale_model_engine::factory::loader_for_config(config)
        .map(|l| l.binds_vision_encoder())
        .unwrap_or(true)
}

/// 2026-09-26: The tensors the model's weight loader reads from disk itself
/// at bind time (`defer_predicate`); they are recorded with their location,
/// not uploaded. `None` when the model type does not resolve or the loader
/// defers nothing, the trait default. Unlike `skip_activation_scales` and
/// `skip_mtp`, the loader declares this itself, so no model list is kept here.
fn defer_hook(config: &ModelConfig) -> Option<metrale_model_weights::weights::DeferHook> {
    let hook = metrale_model_engine::factory::loader_for_config(config)
        .ok()?
        .defer_predicate(config)?;
    tracing::info!(
        "Weight loader for model_type '{}' defers part of the checkpoint: those tensors are \
         recorded with their on-disk location and read by the model's own loader, never \
         uploaded.",
        config.model_type,
    );
    Some(hook)
}
