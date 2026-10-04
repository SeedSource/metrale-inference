// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! 2026-09-25: Construction helpers for `TransformerModel::new`: the GDN prefill
//! scratch buffers, the MTP draft proposer and the optional SSM snapshot spill
//! tier.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use std::sync::Arc;

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// 2026-10-04: Allocate the GDN prefill scratch buffers for
/// `min(max_batch_tokens, max_seq_len)` tokens. Returns
/// `(qkv, gate_beta, out, z, gdn_buf_len)`. The buffers are allocated only when
/// the config has linear-attention heads (`conv_dim > 0`) and the GLM-5.3 arena
/// trim does not apply (`glm_arena_trim_active`: GLM-5.3 never runs the
/// two-phase SSM prefill, which needs a single rank, and with NULL buffers
/// `prefill_twophase_dispatch` falls back to `prefill_chunk`); otherwise all
/// four are `DevicePtr::NULL`, which also avoids a zero-byte allocation.
pub(super) fn build_gdn_prefill_buffers(
    config: &ModelConfig,
    max_batch_tokens: usize,
    max_seq_len: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, DevicePtr, DevicePtr, DevicePtr, usize)> {
    let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let nv = config.linear_num_value_heads;
    let conv_dim = key_dim * 2 + value_dim;
    let gdn_buf_len = max_batch_tokens.min(max_seq_len);
    let glm_trim = metrale_gpu_runtime::buffers::glm_arena_trim_active(config);
    let (gdn_qkv, gdn_gate_beta, gdn_out, gdn_z) = if conv_dim > 0 && !glm_trim {
        let qkv = gpu.alloc(gdn_buf_len * conv_dim * 2)?;
        let gb = gpu.alloc(gdn_buf_len * nv * 2 * 4)?;
        let o = gpu.alloc(gdn_buf_len * value_dim * 2)?;
        let z = gpu.alloc(gdn_buf_len * value_dim * 2)?;
        let total_mb =
            (gdn_buf_len * (conv_dim * 2 + nv * 2 * 4 + value_dim * 2 * 2)) / (1024 * 1024);
        tracing::info!(
            "GDN prefill buffers: {total_mb} MB for {gdn_buf_len} tokens (chunked SSM prefill)"
        );
        (qkv, gb, o, z)
    } else {
        if conv_dim > 0 {
            tracing::info!(
                "GDN prefill buffers: not allocated, {} MB saved (METRALE_GLM_ARENA_TRIM: \
                 GLM-5.3 never runs the two-phase SSM prefill)",
                (gdn_buf_len * (conv_dim * 2 + nv * 2 * 4 + value_dim * 2 * 2)) / (1024 * 1024),
            );
        }
        (
            DevicePtr::NULL,
            DevicePtr::NULL,
            DevicePtr::NULL,
            DevicePtr::NULL,
        )
    };
    Ok((gdn_qkv, gdn_gate_beta, gdn_out, gdn_z, gdn_buf_len))
}

/// 2026-09-25: Build the MTP draft proposer when speculative decoding is
/// requested: one `MtpHead` for a single MTP module, a `MultiModuleMtpHead`
/// over one head per module for several.
///
/// Returns `None` when speculative decoding is off, when the checkpoint has no
/// MTP weights, or when no NVFP4 draft head is available. A head that exists
/// but fails to build is an error, so `--speculative` never serves with
/// speculation silently off.
///
/// `lm_head_nvfp4` is the resolved draft head: the main NVFP4 head, or the
/// draft-only NVFP4 head built when the main head is not NVFP4. `MtpHead::new`
/// takes it as a `QuantizedWeight`, which is why an NVFP4 head is required.
pub(super) fn build_mtp_proposer(
    use_speculative: bool,
    mtp_weights: Vec<MtpWeights>,
    embed_tokens: DenseWeight,
    target_final_norm: DenseWeight,
    lm_head_nvfp4: Option<QuantizedWeight>,
    // 2026-09-25: Padded transposed twin of the main head for the batched
    // propose tile GEMM. The caller passes `Some` only when the draft head is
    // the main head (`draft_lm_head_nvfp4_t` in `TransformerModel::new`).
    lm_head_nvfp4_t: Option<(QuantizedWeight, u32)>,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    mtp_quant: metrale_model_layers::layers::MtpQuantization,
    mtp_vocab_size: u32,
    max_seq_len: usize,
    main_kv_blocks: usize,
    levers: &metrale_model_layers::layers::ops::ModelLevers,
) -> Result<Option<Arc<dyn DraftProposer>>> {
    if !use_speculative {
        if !mtp_weights.is_empty() {
            tracing::info!(
                "MTP weights available ({} module(s)) but --speculative not set, skipping MTP head construction",
                mtp_weights.len()
            );
        }
        return Ok(None);
    }
    if mtp_weights.is_empty() {
        return Ok(None);
    }
    let lm_nvfp4 = match lm_head_nvfp4 {
        Some(w) => w,
        None => {
            tracing::warn!(
                "MTP weights found but no NVFP4 LM head — speculative decoding disabled."
            );
            return Ok(None);
        }
    };
    let build_head = |mtp_wts: MtpWeights| {
        metrale_model_layers::layers::MtpHead::new(
            mtp_wts,
            embed_tokens,
            target_final_norm,
            lm_nvfp4,
            lm_head_nvfp4_t,
            config,
            gpu,
            mtp_quant,
            mtp_vocab_size,
            max_seq_len,
            main_kv_blocks,
            levers,
        )
    };
    if mtp_weights.len() == 1 {
        match build_head(mtp_weights.into_iter().next().unwrap()) {
            Ok(head) => {
                tracing::info!("MTP speculative decoding: ENABLED (single-module)");
                Ok(Some(Arc::new(head) as Arc<dyn DraftProposer>))
            }
            Err(e) => Err(e.context(
                "--speculative is set but the MTP head failed to build; refusing to serve \
                 with speculation silently off (change --mtp-quantization, or drop --speculative)",
            )),
        }
    } else {
        let count = mtp_weights.len();
        let heads: Result<Vec<_>> = mtp_weights.into_iter().map(build_head).collect();
        match heads.and_then(metrale_model_layers::layers::mtp_multi::MultiModuleMtpHead::new) {
            Ok(multi) => {
                tracing::info!("MTP speculative decoding: ENABLED (multi-module, {count} heads)");
                Ok(Some(Arc::new(multi) as Arc<dyn DraftProposer>))
            }
            Err(e) => Err(e.context(
                "--speculative is set but the multi-module MTP head failed to build; refusing \
                 to serve with speculation silently off",
            )),
        }
    }
}

/// 2026-09-25: Build the optional SSM snapshot spill tier. Returns `None`
/// unless `METRALE_SSM_TIER` is set and the model has SSM layers; otherwise
/// builds the env-selected store keyed by the model's `ModelFingerprint`.
///
/// `blob_bytes` must be `SsmSnapshotPool::spill_blob_bytes()`: the tier's blobs
/// are fixed-size and must match the spill and fault-in gathers.
pub(super) fn build_ssm_tier_store(
    config: &ModelConfig,
    blob_bytes: usize,
    num_ssm_layers: usize,
) -> Result<Option<Arc<dyn super::ssm_tier::SnapshotBlobStore>>> {
    if super::ssm_tier::ssm_tier_enabled() && num_ssm_layers > 0 {
        let fp = super::ssm_tier::ModelFingerprint::derive(config, blob_bytes)?;
        Ok(Some(super::ssm_tier::build_tier_store(fp, blob_bytes)?))
    } else {
        Ok(None)
    }
}
