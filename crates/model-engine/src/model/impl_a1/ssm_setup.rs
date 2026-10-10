// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: SSM sizing for `TransformerModel::new`: the verify width the rollback pools
//! are sized for, and the snapshot pool (Marconi region and decode-rollback ring)
//! with its optional spill tier.
//!
//! Owner: model-engine.
//! Invariants:
//! - `has_mtp` holds for every proposer that can reject a draft, so the rollback pools exist whenever a verify can roll back.

use std::sync::Arc;

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layers::MtpQuantization;
use metrale_model_layers::weight_map::{MtpWeights, QuantizedWeight};

use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::ssm_tier::SnapshotBlobStore;

/// 2026-09-26: Returns `(dflash_kgamma, draft_lm_head_nvfp4, has_mtp, mtp_quant_fwd,
/// num_intermediates)`.
pub(super) fn size_rollback_pools(
    config: &ModelConfig,
    mtp_lm_head_nvfp4: Option<QuantizedWeight>,
    lm_head_nvfp4: Option<QuantizedWeight>,
    self_speculative: bool,
    use_speculative: bool,
    mtp_weights: &[MtpWeights],
    mtp_quant: MtpQuantization,
    num_drafts: usize,
) -> (usize, Option<QuantizedWeight>, bool, MtpQuantization, usize) {
    // 2026-09-25: The SSM rollback pools are sized for the widest verify K
    // (`num_intermediates` below): `num_drafts + 1` for MTP, and
    // `dflash_kgamma` = γ + 1 alone when DFlash is on.
    let dflash_kgamma = if !config.dflash_capture_layers.is_empty() {
        // 2026-09-25: The +1 is `last_token` in the verify input
        // `[last_token, draft_0, ..., draft_{γ-1}]`. γ is
        // `config.dflash_gamma`, which the factory sets from `--dflash-gamma`
        // or the drafter checkpoint; an unset γ falls back to 17 rows.
        config.dflash_gamma.map(|g| g + 1).unwrap_or(17)
    } else {
        0
    };
    // 2026-09-25: The MTP proposer drafts through an NVFP4 vocab head: the
    // draft-only head when one was built, else the main NVFP4 head.
    let draft_lm_head_nvfp4 = mtp_lm_head_nvfp4.or(lm_head_nvfp4);
    // 2026-09-25: `has_mtp` sizes the recurrent rollback pools (checkpoints
    // and per-token intermediates), so it must hold for every proposer that
    // can reject a draft. DFlash always needs them: its K=γ verify
    // checkpoints SSM state for partial-accept rollback. A proposer installed
    // after construction (`set_dflash_proposer`) is invisible here, so a
    // checkpoint that declares MTP layers (`mtp_layer_types`, filled by the
    // glm5_next parser) also counts. Without the pools, the first
    // `ssm_pool` checkpoint access indexes an empty vector and panics.
    let checkpoint_declares_mtp = !config.mtp_layer_types.is_empty();
    let has_mtp = self_speculative
        || (use_speculative
            && ((!mtp_weights.is_empty() && draft_lm_head_nvfp4.is_some())
                || checkpoint_declares_mtp))
        || dflash_kgamma > 0;
    // 2026-09-25: The precision the MTP head's forward runs at: a dense-FFN
    // head under `MtpQuantization::Nvfp4` runs the BF16 forward
    // (`MtpQuantization::effective_for_head`).
    let mtp_quant_fwd =
        mtp_quant.effective_for_head(mtp_weights.first().is_some_and(|w| w.dense_ffn.is_some()));
    let num_intermediates = if !has_mtp {
        0
    } else if dflash_kgamma > 0
        && metrale_model_layers::speculative::hybrid::hybrid_dflash_enabled()
    {
        // 2026-10-10: A hybrid serve verifies through both proposers: `num_drafts` is the MTP
        // depth, and the widest verify is the larger of the two.
        dflash_kgamma.max(num_drafts + 1)
    } else if dflash_kgamma > 0 {
        // 2026-09-25: A DFlash serve verifies through the block drafter, so
        // the widest verify is K = γ + 1 and `num_drafts` does not size the
        // pools.
        dflash_kgamma
    } else {
        num_drafts + 1
    };
    (
        dflash_kgamma,
        draft_lm_head_nvfp4,
        has_mtp,
        mtp_quant_fwd,
        num_intermediates,
    )
}

/// 2026-09-26: Returns `(ssm_snapshots, ssm_tier_store)`.
pub(super) fn build_ssm_snapshots(
    config: &ModelConfig,
    ssm_pool: &SsmStatePool,
    use_speculative: bool,
    ssm_cache_slots: usize,
    prefix_cache: &dyn metrale_telemetry::prefix_cache::PrefixCache,
    max_batch_size: usize,
    ssm_checkpoint_interval: usize,
    kv_cache: &PagedKvCache,
    gpu: &dyn GpuBackend,
) -> Result<(SsmSnapshotPool, Option<Arc<dyn SnapshotBlobStore>>)> {
    // 2026-09-25: Fail at startup when an SSM tier variable
    // (`SSM_TIER_VARS`) is set on a model without recurrent state.
    super::super::ssm_tier::ensure_ssm_tier_capability(config)?;

    // 2026-09-25: SSM snapshot pool: the prefix-cache (Marconi) slots and a
    // decode-rollback ring for each of `max_batch_size` sequences; both are
    // empty on a model without SSM layers. The ring depth comes from
    // `ssm_reserve::decode_rollback_ring_slots`, which the server's
    // preflight reserve also calls, so the reservation and this allocation
    // agree. Unless a depth was published or `METRALE_SSM_DECODE_RING`
    // overrides it, the depth is 0 under speculative decode or with
    // watchdogs disabled. The scheduler sizes its rings from
    // `decode_rollback_ring_slots()`, so a 0 turns off save and rollback
    // together.
    let ring = metrale_model_layers::ssm_reserve::decode_rollback_ring_slots(
        ssm_pool.num_ssm_layers,
        use_speculative,
    );
    if let Some(reason) = ring.skip_reason {
        let per_seq = (ssm_pool.h_bytes + ssm_pool.conv_bytes)
            * ssm_pool.num_ssm_layers
            * metrale_kernels::DECODE_ROLLBACK_RING_SLOTS;
        tracing::info!(target: "metrale_model_engine::model::impl_a1", "SSM decode-rollback ring: SKIPPED ({}) — the ring's save/rollback \
             path only runs on plain decode with watchdogs enabled. Saves {:.1} GB \
             ({} seqs x {} slots x full SSM blob). If plain-decode loop re-steer is \
             ever reached it fail-opens to decline; METRALE_SSM_DECODE_RING=1 \
             force-restores the ring.",
            reason,
            (per_seq * max_batch_size) as f64 / 1e9,
            max_batch_size,
            metrale_kernels::DECODE_ROLLBACK_RING_SLOTS,
        );
    }
    let decode_ring_slots = ring.slots;
    // 2026-09-25: Marconi region. `ssm_reserve::marconi_snapshot_slots`
    // makes the same decision the server's preflight reserve made before
    // the weights loaded. Its only reader is a prefix-cache lookup, so it
    // is dropped when the constructed cache is inactive; asking the cache
    // (`is_active`) rather than the CLI flag also covers a config whose
    // flag is set but that installs `NoPrefixCaching`.
    let marconi = metrale_model_layers::ssm_reserve::marconi_snapshot_slots(
        ssm_cache_slots,
        prefix_cache.is_active(),
    );
    if let Some(reason) = marconi.skip_reason {
        tracing::info!(target: "metrale_model_engine::model::impl_a1", "SSM snapshot pool: Marconi region SKIPPED ({}) — {} slot(s) x {} layer(s) \
             = {:.0} MB freed for KV (restore with --enable-prefix-caching, or \
             METRALE_SSM_MARCONI_FULL to allocate anyway)",
            reason,
            ssm_cache_slots,
            ssm_pool.num_ssm_layers,
            (ssm_cache_slots
                * ssm_pool.num_ssm_layers
                * (ssm_pool.h_bytes + ssm_pool.conv_bytes)) as f64
                / (1024.0 * 1024.0),
        );
    }
    let ssm_cache_slots = marconi.slots;
    let ssm_snapshots = SsmSnapshotPool::new(
        ssm_cache_slots,
        ssm_pool.h_bytes,
        ssm_pool.conv_bytes,
        ssm_pool.num_ssm_layers,
        decode_ring_slots,
        max_batch_size,
        // 2026-09-25: Bytes per slot of the last-token hidden snapshot
        // (BF16, `hidden_size` elements), which an exact prefix hit feeds
        // to the lm_head instead of re-running the last token.
        config.hidden_size * 2,
        gpu,
    )?;
    let ssm_tier_store = super::super::impl_a1_init::build_ssm_tier_store(
        config,
        ssm_snapshots.spill_blob_bytes(),
        ssm_pool.num_ssm_layers,
    )?;
    if ssm_checkpoint_interval > 0 && ssm_cache_slots > 0 {
        tracing::info!(target: "metrale_model_engine::model::impl_a1", "Marconi intermediate checkpoints: every {} blocks ({} tokens at block_size={})",
            ssm_checkpoint_interval,
            ssm_checkpoint_interval * kv_cache.block_size(),
            kv_cache.block_size(),
        );
    }
    Ok((ssm_snapshots, ssm_tier_store))
}

/// 2026-09-26: Log the self-speculative layer split when it is on.
pub(super) fn log_self_speculative(config: &ModelConfig, self_speculative: bool) {
    if self_speculative {
        let num_ssm = config.num_ssm_layers();
        let num_attn = config.num_attention_layers();
        tracing::info!(target: "metrale_model_engine::model::impl_a1", "Self-speculative decoding: ENABLED (skipping {} SSM layers, keeping {} attention layers)",
            num_ssm,
            num_attn,
        );
    }
}
