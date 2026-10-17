// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The proposers `build_model` installs after
//! `TransformerModel::new` (DeepSeek-V4 MTP, GLM-5.3 MTP, DFlash), and the
//! allocation report it logs last.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - A DeepSeek-V4 or GLM-5.3 proposer build error is logged and not returned;
//!   a DFlash drafter error is returned.
//! - Under `METRALE_HYBRID_DFLASH` (2026-10-10) a drafter that does not install on this rank is
//!   a startup error.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_arch::weight_loader::deepseek_v4::mtp::DeepseekV4MtpModule;
use metrale_model_arch::weight_loader::glm5_next_mtp::Glm5NextMtpModule;
use metrale_model_arch::weight_loader::load_dflash_weights;
use metrale_model_layers::speculative::DraftProposer as _;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight, QuantizedWeight};

use crate::factory::DflashBuildArgs;
use crate::model::TransformerModel;

/// 2026-09-26: Installs the DeepSeek-V4 MTP proposer when the module loaded.
pub(super) fn install_v4_mtp_proposer(
    model: &mut TransformerModel,
    v4_mtp_module: Option<DeepseekV4MtpModule>,
    v4_mtp_embed: DenseWeight,
    v4_mtp_lm_head: DenseWeight,
    mtp_vocab_size: u32,
    max_seq_len: usize,
) {
    if let Some(v4_module) = v4_mtp_module {
        match metrale_model_arch::deepseek_v4_mtp::DeepseekV4MtpHead::new(
            v4_module,
            v4_mtp_embed,
            v4_mtp_lm_head,
            model.config_ref(),
            model.gpu_backend(),
            mtp_vocab_size,
            max_seq_len,
        ) {
            Ok(head) => {
                model.set_dflash_proposer(std::sync::Arc::new(head));
                tracing::info!(target: "metrale_model_engine::factory::build", "DeepSeek-V4 MTP speculative decoding: ENABLED (single-module)");
            }
            Err(e) => {
                tracing::warn!(target: "metrale_model_engine::factory::build", "Failed to build DeepSeek-V4 MTP proposer: {e:#}. Speculative decoding disabled."
                )
            }
        }
    }
}

/// 2026-09-26: Installs the GLM-5.3 MTP proposer when the module loaded.
pub(super) fn install_glm_mtp_proposer(
    model: &mut TransformerModel,
    glm_mtp_module: Option<Glm5NextMtpModule>,
    glm_mtp_embed: DenseWeight,
    glm_mtp_lm_head: DenseWeight,
    max_seq_len: usize,
) {
    if let Some(m) = glm_mtp_module {
        match metrale_model_arch::glm5next_mtp_head::Glm5NextMtpHead::new(
            m,
            glm_mtp_embed,
            glm_mtp_lm_head,
            model.config_ref(),
            model.gpu_backend(),
            max_seq_len,
        ) {
            Ok(head) => {
                model.set_dflash_proposer(std::sync::Arc::new(head));
                tracing::info!(target: "metrale_model_engine::factory::build", "GLM-5.3 MTP speculative decoding: ENABLED");
            }
            Err(e) => {
                tracing::warn!(target: "metrale_model_engine::factory::build", "Failed to build GLM-5.3 MTP proposer: {e:#}. Speculative decoding disabled."
                )
            }
        }
    }
}

/// 2026-09-26: Installs the DFlash drafter when `dflash_args` is given and its
/// store has DFlash weights.
/// 2026-10-14: `dflash_credit` = the BF16 bytes the KV sizing credited
/// (`kv_budget::dflash_bf16_credit_bytes`); the install fails when it frees less.
pub(super) fn install_dflash_drafter(
    model: &mut TransformerModel,
    dflash_args: Option<DflashBuildArgs<'_>>,
    target_embed_for_dflash: DevicePtr,
    target_lm_head_for_dflash: DevicePtr,
    target_lm_head_nvfp4_for_dflash: Option<QuantizedWeight>,
    target_lm_head_native_fp8_for_dflash: Option<(Fp8DenseWeight, usize)>,
    target_hidden_for_dflash: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    dflash_credit: usize,
) -> Result<()> {
    if let Some(args) = dflash_args {
        let weights = load_dflash_weights(
            args.drafter_store,
            &args.drafter_config,
            model.gpu_backend(),
            1,
        )?;
        if let Some(weights) = weights {
            let mut head = metrale_model_arch::dflash_head::BlockDiffusionDraftHead::from_weights(
                weights,
                target_embed_for_dflash,
                target_lm_head_for_dflash,
                target_lm_head_nvfp4_for_dflash,
                target_lm_head_native_fp8_for_dflash,
                target_hidden_for_dflash,
                args.gamma,
                args.window_size,
                model.gpu_backend(),
                max_seq_len,
                max_batch_size,
                model.dflash_tp_rank(),
            )?;
            // 2026-10-10: The BF16 copies of the layer GEMM weights are dead once the head
            // runs on FP8 mirrors (#82 P2); every rank frees its own.
            let released =
                head.release_bf16_layer_weights(args.drafter_store, model.gpu_backend())?;
            if let Some((n, bytes)) = released {
                tracing::info!(target: "metrale_model_engine::factory::build",
                    "DFlash drafter: freed {n} BF16 layer weight tensor(s), {:.2} GB; the head \
                     runs on their FP8 mirrors",
                    bytes as f64 / 1e9
                );
            }
            super::kv_budget::check_dflash_bf16_credit(
                dflash_credit,
                released.map_or(0, |(_, bytes)| bytes),
            )?;
            // 2026-10-14: The mirrors this rank allocated against the reserve (#82 hybrid M1b).
            super::kv_budget::check_dflash_mirror_reserve(
                super::kv_budget::dflash_fp8_mirror_reserve_bytes(
                    &args.drafter_config,
                    model.dflash_tp_rank(),
                ),
                head.fp8_mirror_bytes,
            )?;
            if metrale_model_layers::speculative::hybrid::hybrid_dflash_enabled() {
                // 2026-10-10: Hybrid serve: the drafter goes beside the MTP head, which stays in
                // `proposer` with every buffer it was sized with (`hybrid_spec.rs`).
                let gamma = head.block_gamma();
                model.set_hybrid_dflash_proposer(std::sync::Arc::new(head))?;
                tracing::info!(target: "metrale_model_engine::factory::build",
                    "HYBRID DFLASH: ENGAGED — proposers: MTP head (proposer slot) + DFlash drafter \
                     (γ={gamma:?}, KV window {max_seq_len} tokens); rank 0 picks one per step \
                     (METRALE_HYBRID_DFLASH_DWELL_MS={} ms)",
                    metrale_model_layers::speculative::hybrid::hybrid_dflash_dwell_ms(),
                );
            } else {
                model.set_dflash_proposer(std::sync::Arc::new(head));
                tracing::info!(target: "metrale_model_engine::factory::build", "DFlash drafter installed as the active proposer");
            }
        } else if metrale_model_layers::speculative::hybrid::hybrid_dflash_enabled() {
            // 2026-10-10: Under the hybrid serve a rank without the drafter would answer DFlash
            // steps with the MTP head while the other rank drafts: a startup error, not a warn.
            anyhow::bail!(
                "METRALE_HYBRID_DFLASH=1 but the DFlash drafter store has no fc.weight: the \
                 drafter did not install on this rank, and a hybrid serve needs it on every rank"
            );
        } else {
            super::kv_budget::check_dflash_bf16_credit(dflash_credit, 0)?;
            tracing::warn!(target: "metrale_model_engine::factory::build", "DFlash drafter store had no fc.weight — proposer not installed; \
                 falling back to whatever proposer (if any) the target's MTP path built"
            );
        }
    } else if metrale_model_layers::speculative::hybrid::hybrid_dflash_enabled() {
        anyhow::bail!(
            "METRALE_HYBRID_DFLASH=1 but this rank has no DFlash drafter (--dflash): a hybrid \
             serve needs the drafter installed on every rank"
        );
    }
    Ok(())
}

/// 2026-09-26: Logs the backend's allocation report and whether tracked-live
/// bytes stayed within `total_budget`.
pub(super) fn log_alloc_report(
    model: &TransformerModel,
    total_budget: usize,
    gpu_memory_utilization: f64,
    total_mem: usize,
    gib: impl Fn(usize) -> f64,
) {
    // 2026-09-25: The backend's allocation report, logged at INFO once, after
    // every build allocation.
    if let Some(report) = model.gpu_backend().alloc_report(12, 64) {
        for line in report.lines() {
            tracing::info!(target: "metrale_model_engine::factory::build", "{line}");
        }
    }
    // 2026-09-25: Tracked-live bytes above the util budget mean an allocation
    // the sizing above did not reserve for. Logged as a warning; the build
    // still succeeds.
    if let Some(live) = model.gpu_backend().live_bytes() {
        if live > total_budget {
            tracing::warn!(target: "metrale_model_engine::factory::build", "util pledge exceeded: {:.1} GB tracked live vs {:.1} GB pledged \
                 (--gpu-memory-utilization {:.0}% of {:.1} GB) — an allocation \
                 family above is missing from the preflight reserve",
                gib(live),
                gib(total_budget),
                gpu_memory_utilization * 100.0,
                gib(total_mem),
            );
        } else {
            tracing::info!(target: "metrale_model_engine::factory::build", "util pledge honored: {:.1} GB tracked live within the {:.1} GB \
                 budget ({:.1} GB pledge headroom)",
                gib(live),
                gib(total_budget),
                gib(total_budget - live),
            );
        }
    }
}
