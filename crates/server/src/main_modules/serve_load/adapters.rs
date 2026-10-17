// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The adapter steps of `load_model`: the LoRA and DFlash build
//! arguments, the NLLB language pair and adapter, and the stageable-adapter
//! registries that arm demand promotion.
//!
//! Owner: server startup (`met serve`).
//! Invariants: the adapter stores stay owned by `load_model`; these functions
//! only borrow them. The `tracing` events keep `load_model`'s target (`met::main_modules::serve_load`).

use std::sync::Arc;

use anyhow::{Context, Result};
use metrale_config::ModelConfig;

use crate::cli;
use crate::main_modules::serve_phases::LoraAdapterState;

/// 2026-09-26: Pool rank when `--max-lora-rank` is unset and a stageable
/// adapter (peer or disk) is configured, whose rank is not known at startup.
/// It is also the rank ceiling a disk-stageable adapter is checked against.
const DEFAULT_MAX_LORA_RANK: usize = 64;

pub(super) fn lora_build_args<'a>(
    args: &cli::ServeArgs,
    lora_states: &'a [LoraAdapterState],
) -> Option<metrale_model_engine::factory::LoraBuildArgs<'a>> {
    // 2026-09-26: Pool rank: `--max-lora-rank` when set, else the largest
    // resident adapter rank. With a stageable adapter configured it is
    // `DEFAULT_MAX_LORA_RANK`: the pool is sized at startup, and a staged
    // adapter's rank is not known here.
    let max_lora_rank = args.max_lora_rank.unwrap_or_else(|| {
        if !args.lora_stageable.is_empty() || !args.lora_stageable_disk.is_empty() {
            DEFAULT_MAX_LORA_RANK
        } else {
            lora_states
                .iter()
                .map(|l| l.peft_config.r)
                .max()
                .unwrap_or(DEFAULT_MAX_LORA_RANK)
                .max(1)
        }
    });
    if lora_states.is_empty() {
        None
    } else {
        Some(metrale_model_engine::factory::LoraBuildArgs {
            adapters: lora_states
                .iter()
                .map(|l| metrale_model_layers::lora::LoraAdapterInput {
                    name: l.name.clone(),
                    store: &l.store,
                    peft: l.peft_config.clone(),
                })
                .collect(),
            max_lora_rank,
            max_loras: args.max_loras,
        })
    }
}

pub(super) fn dflash_build_args<'a>(
    args: &cli::ServeArgs,
    dflash_drafter_state: &'a mut Option<(
        metrale_model_weights::weights::WeightStore,
        metrale_model_arch::weight_loader::DflashConfig,
    )>,
) -> Option<metrale_model_engine::factory::DflashBuildArgs<'a>> {
    dflash_drafter_state
        .as_mut()
        .map(|(s, c)| metrale_model_engine::factory::DflashBuildArgs {
            drafter_store: s,
            drafter_config: c.clone(),
            gamma: args.dflash_gamma,
            window_size: if args.dflash_window_size > 0 {
                Some(args.dflash_window_size)
            } else {
                None
            },
        })
}

pub(super) fn resolve_nllb_lang(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model_dir: &std::path::Path,
) -> Result<Option<(u32, u32)>> {
    // 2026-09-26: NLLB / M2M-100 language pair, as token ids from the
    // checkpoint's tokenizer; the `ChatTokenizer` is built after the model.
    // Other models pass `None`.
    let nllb_lang: Option<(u32, u32)> = if matches!(config.model_type.as_str(), "m2m_100" | "nllb")
    {
        let src = args.src_lang.as_deref().ok_or_else(|| {
            anyhow::anyhow!("serving an NLLB/M2M-100 checkpoint requires --src-lang")
        })?;
        let tgt = args.tgt_lang.as_deref().ok_or_else(|| {
            anyhow::anyhow!("serving an NLLB/M2M-100 checkpoint requires --tgt-lang")
        })?;
        let tk = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("nllb: load tokenizer for lang resolve: {e}"))?;
        let src_id = tk
            .token_to_id(src)
            .ok_or_else(|| anyhow::anyhow!("unknown --src-lang token '{src}'"))?;
        let tgt_id = tk
            .token_to_id(tgt)
            .ok_or_else(|| anyhow::anyhow!("unknown --tgt-lang token '{tgt}'"))?;
        tracing::info!(target: "met::main_modules::serve_load", "NLLB translation: {src}({src_id}) -> {tgt}({tgt_id})");
        Some((src_id, tgt_id))
    } else {
        None
    };
    Ok(nllb_lang)
}

pub(super) fn resolve_nllb_adapter(
    args: &cli::ServeArgs,
    is_nllb: bool,
) -> Result<(Option<std::path::PathBuf>, Option<String>)> {
    // 2026-09-26: NLLB adapter: the first `--lora-adapter NAME=DIR`, loaded by
    // `NllbGpuModel` itself.
    let nllb_lora_dir: Option<std::path::PathBuf> = if is_nllb {
        match args.lora_adapter.first() {
            Some((_name, spec)) => Some(
                crate::model_resolver::resolve_adapter_dir(spec, args.cache_dir.as_deref())
                    .context("resolving NLLB --lora-adapter")?,
            ),
            None => None,
        }
    } else {
        None
    };
    // 2026-09-26: The NLLB adapter's name, kept apart from the model name;
    // `AppState` advertises it.
    let nllb_adapter_name: Option<String> = if nllb_lora_dir.is_some() {
        args.lora_adapter.first().map(|(n, _)| n.clone())
    } else {
        None
    };
    Ok((nllb_lora_dir, nllb_adapter_name))
}

/// 2026-09-26: The stageable-adapter registries and the promotion manager,
/// as `AppState` takes them.
pub(super) struct LoraPromotion {
    pub lora_peer_addr: Option<String>,
    pub lora_stageable:
        std::collections::HashMap<String, crate::main_modules::promotion::StageableAdapter>,
    pub lora_disk_stageable:
        std::collections::HashMap<String, (std::path::PathBuf, metrale_config::PeftAdapterConfig)>,
    pub promotion: Option<Arc<crate::main_modules::promotion::PromotionManager>>,
}

pub(super) fn build_lora_promotion(
    args: &cli::ServeArgs,
    lora_states: &[LoraAdapterState],
) -> Result<LoraPromotion> {
    // 2026-09-26: Peer-stageable registry, name -> {peer_stage_id, peft}. The
    // peft config is read from each adapter's local `adapter_config.json` now,
    // so a missing or bad one fails the boot.
    let lora_peer_addr = metrale_model_layers::lora::lora_peer_env();
    let mut lora_stageable = std::collections::HashMap::new();
    for (name, peer_id, dir) in &args.lora_stageable {
        let cfg_path = std::path::Path::new(dir).join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path).with_context(|| {
            format!(
                "--lora-stageable '{name}': read peft config {}",
                cfg_path.display()
            )
        })?;
        let peft = metrale_config::parse_peft_adapter_config(&raw)
            .with_context(|| format!("--lora-stageable '{name}': parse {}", cfg_path.display()))?;
        lora_stageable.insert(
            name.clone(),
            crate::main_modules::promotion::StageableAdapter {
                peer_stage_id: peer_id.clone(),
                peft,
            },
        );
    }
    if !lora_stageable.is_empty() && lora_peer_addr.is_none() {
        anyhow::bail!(
            "--lora-stageable given ({} adapter(s)) but $METRALE_LORA_PEER is unset; \
             demand promotion needs a weight peer to RDMA-stage from",
            lora_stageable.len()
        );
    }
    if !lora_stageable.is_empty() && lora_states.is_empty() {
        anyhow::bail!(
            "--lora-stageable needs a resident pool to promote INTO; start with at \
             least one --lora-adapter (and --max-loras > that count for cache headroom)"
        );
    }
    // 2026-09-26: Disk-stageable registry, name -> (resolved dir, peft). The
    // peft config is parsed and rank-checked now; the disk promote reads the
    // adapter's config again from its dir.
    let mut lora_disk_stageable = std::collections::HashMap::new();
    for (name, spec) in &args.lora_stageable_disk {
        if lora_states.iter().any(|s| &s.name == name) || lora_stageable.contains_key(name) {
            anyhow::bail!(
                "--lora-stageable-disk '{name}' collides with a resident/peer-stageable \
                 adapter name"
            );
        }
        let dir = crate::model_resolver::resolve_adapter_dir(spec, args.cache_dir.as_deref())
            .with_context(|| format!("--lora-stageable-disk '{name}': resolve '{spec}'"))?;
        let cfg_path = dir.join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path).with_context(|| {
            format!(
                "--lora-stageable-disk '{name}': read peft config {}",
                cfg_path.display()
            )
        })?;
        let peft = metrale_config::parse_peft_adapter_config(&raw).with_context(|| {
            format!(
                "--lora-stageable-disk '{name}': parse {}",
                cfg_path.display()
            )
        })?;
        let ceiling = args.max_lora_rank.unwrap_or(DEFAULT_MAX_LORA_RANK);
        if peft.r > ceiling {
            anyhow::bail!(
                "--lora-stageable-disk '{name}' r={} > --max-lora-rank {ceiling}",
                peft.r
            );
        }
        lora_disk_stageable.insert(name.clone(), (dir, peft));
    }
    if !lora_disk_stageable.is_empty() && lora_states.is_empty() {
        anyhow::bail!(
            "--lora-stageable-disk needs a resident pool to promote INTO; start with at \
             least one --lora-adapter and --max-loras > that count"
        );
    }
    if !lora_disk_stageable.is_empty()
        && !metrale_model_layers::lora::lora_rotate_env()
        && lora_peer_addr.is_none()
    {
        anyhow::bail!(
            "--lora-stageable-disk needs rotation armed: set METRALE_LORA_ROTATE=1 so decode \
             runs eager and the disk swap can re-point a cache slot"
        );
    }
    // 2026-09-26: Promotion is armed when a LoRA pool is resident and there is
    // a stageable source: a peer registry with a peer address, or a disk
    // registry. Peer and disk misses share one `PromotionManager`.
    let promotion = if (!lora_stageable.is_empty() && lora_peer_addr.is_some()
        || !lora_disk_stageable.is_empty())
        && !lora_states.is_empty()
    {
        Some(Arc::new(
            crate::main_modules::promotion::PromotionManager::default(),
        ))
    } else {
        None
    };
    if promotion.is_some() {
        tracing::info!(target: "met::main_modules::serve_load", "LoRA #27: {} peer + {} disk stageable adapter(s) armed for demand promotion \
             (peer={:?}, cache headroom={} slots)",
            lora_stageable.len(),
            lora_disk_stageable.len(),
            lora_peer_addr,
            args.max_loras.saturating_sub(lora_states.len()),
        );
    }
    Ok(LoraPromotion {
        lora_peer_addr,
        lora_stageable,
        lora_disk_stageable,
        promotion,
    })
}
