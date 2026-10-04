// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The pre-load reserve preflight: the device memory the SSM
//! pools, snapshot regions, decode-rollback ring, buffer arena and CUDA
//! headroom will need, refused before any weight loads when it does not fit.
//! GPU init and the post-load audit are re-exported from sub-modules.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `preflight_reserve` returns `Ok` only when `inference_reserve +
//!   buffer_arena_bytes` is at most `free_mem`.

use anyhow::Result;

use metrale_config::ModelConfig;

use crate::cli;

mod decode_ring;
mod gpu_backend;
mod headroom;
mod per_sequence_state;
mod post_load_audit;
mod refusal;
mod ssm_h_fp16;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) use gpu_backend::init_gpu_backend;
pub(crate) use headroom::PostLoadInputs;
pub(crate) use post_load_audit::post_load_memory_audit;
use {per_sequence_state::per_sequence_reserve, ssm_h_fp16::ssm_h_fp16_preconditions};

pub(crate) struct ReservePreflight {
    pub(crate) inference_reserve: usize,
    pub(crate) buffer_arena_bytes: usize,
    pub(crate) gdn_two_phase_bytes: usize,
    pub(crate) ssm_prefill_chunk: usize,
    pub(crate) max_batch_tokens_pre: usize,
}

pub(crate) fn preflight_reserve(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    free_mem: usize,
    // 2026-09-26: What the decode-ring auto-fit needs to predict post-load KV
    // headroom (`headroom::post_load_yardstick`). The caller gathers it: none
    // of it is in `args` or `config`.
    post_load: &PostLoadInputs<'_>,
) -> Result<ReservePreflight> {
    let h_state_bytes = config.ssm_h_state_bytes();
    let conv_state_bytes = config.ssm_conv_state_bytes();
    // 2026-09-26: `args.dflash` counts as speculative here because
    // `TransformerModel::new` allocates the SSM rollback pools whenever DFlash
    // capture layers exist (`has_mtp` includes `dflash_kgamma > 0`).
    let spec_on_pool =
        args.speculative || args.self_speculative || args.ngram_speculative || args.dflash;
    ssm_h_fp16_preconditions(args, config)?;
    // 2026-09-26: Verify slots are `ssm_reserve::mtp_state_slots`, the count
    // `SsmStatePool::new` also allocates. `METRALE_MTP_POOL_FULL_WIDTH`
    // (present, any value) makes it `max_batch_size`.
    let mtp_state_slots = metrale_model_layers::ssm_reserve::mtp_state_slots(args.max_batch_size);
    // 2026-09-26: The FP32 prefill staging arena of an f16-sized h pool: one
    // blob per slot, shared by all layers, and 0 without `--ssm-h-dtype
    // f16-pool` (`ssm_h_prefill_stage_bytes`). Counted for `max_batch_size`
    // slots, without the pools' dummy slot.
    let ssm_h_stage_bytes = metrale_model_layers::ssm_reserve::ssm_h_prefill_stage_bytes(
        args.max_batch_size,
        h_state_bytes,
        metrale_model_layers::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
    );
    // 2026-09-26: A DFlash serve's verify pools are γ + 1 rows wide on every
    // slot, and the drafter loads after this preflight. γ here is
    // `default_dflash_gamma` of the drafter's `dflash_config.block_size` when
    // its local config.json has one, else `resolved_dflash_gamma(None)`
    // (`--dflash-gamma`, else 16). An unset γ allocates 17 rows
    // (`TransformerModel::new`), so the fallback does not under-reserve.
    let pool_num_drafts = if args.dflash {
        peek_dflash_block_size(args.draft_model.as_deref())
            .map(metrale_model_layers::layers::qwen3_ssm::default_dflash_gamma)
            .unwrap_or_else(|| args.resolved_dflash_gamma(None))
    } else {
        args.resolved_num_drafts()
    };
    let ssm_pool_bytes = metrale_model_layers::ssm_reserve::ssm_pool_reserve_bytes(
        args.max_batch_size,
        config.num_ssm_layers() * h_state_bytes,
        config.num_ssm_layers() * conv_state_bytes,
        spec_on_pool,
        pool_num_drafts,
        mtp_state_slots,
        args.dflash,
        // 2026-09-26: The h-pool narrowing `SsmStatePool::new` applies under
        // `--ssm-h-dtype f16-pool`.
        metrale_model_layers::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
        // 2026-09-26: `--ssm-rollback-mode`, published by `serve_flags` before
        // this runs. Replay mode reserves only the checkpoint blob per verify
        // slot; its input ring is the separate term below.
        metrale_model_layers::ssm_reserve::ssm_rollback_mode(),
    );
    // 2026-09-26: Replay mode's verify-window input ring, sized by
    // `ssm_replay_ring_bytes` as `SsmStatePool::new` sizes it.
    let ssm_replay_ring = if spec_on_pool
        && metrale_model_layers::ssm_reserve::ssm_rollback_mode()
            == metrale_model_layers::ssm_reserve::SsmRollbackMode::Replay
    {
        metrale_model_layers::ssm_reserve::ssm_replay_ring_bytes(
            config.num_ssm_layers(),
            metrale_model_layers::ssm_reserve::ssm_replay_row_bytes(
                config.ssm_qkvz_size(),
                config.linear_num_value_heads,
            ),
            pool_num_drafts + 1,
            mtp_state_slots,
        )
    } else {
        0
    };
    let spec_tokens_pre = spec_reserve_tokens(args);
    // 2026-09-26: An SSM model prefills in chunks of `--max-prefill-tokens`
    // when it is set to anything but 8192 (and above 0), else of 8192; either
    // way at most `max_seq_len`. The chunk bounds `max_batch_tokens_pre`, which
    // sizes the buffer arena and the GDN two-phase term below.
    let ssm_prefill_chunk: usize = if config.num_ssm_layers() > 0 {
        if args.max_prefill_tokens != 8192 && args.max_prefill_tokens > 0 {
            args.max_seq_len.min(args.max_prefill_tokens)
        } else {
            args.max_seq_len.min(8192)
        }
    } else {
        0
    };
    let user_set_prefill_pre = args.max_prefill_tokens != 8192;
    let prefill_budget_pre = if user_set_prefill_pre && args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else if ssm_prefill_chunk > 0 {
        ssm_prefill_chunk
    } else if args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else {
        args.max_seq_len
    };
    let max_batch_tokens_pre = prefill_budget_pre
        .max(spec_tokens_pre)
        .max(args.max_batch_size);
    let buffer_arena_bytes = metrale_gpu_runtime::buffers::BufferSizes::from_config(
        config,
        max_batch_tokens_pre,
        args.max_seq_len,
        args.block_size,
        args.max_batch_size,
    )
    .total_bytes();
    // 2026-09-26: Marconi snapshot slots, from
    // `ssm_reserve::marconi_snapshot_slots`, which `TransformerModel::new` also
    // calls: 0 while prefix caching is inactive, unless
    // `METRALE_SSM_MARCONI_FULL` is present.
    let marconi = metrale_model_layers::ssm_reserve::marconi_snapshot_slots(
        args.ssm_cache_slots,
        metrale_model_layers::ssm_reserve::prefix_caching_active(
            args.prefix_caching_enabled(),
            config.kv_only_prefix_cache_is_safe(),
        ),
    );
    if let Some(reason) = marconi.skip_reason {
        tracing::info!(
            "SSM snapshot pool: Marconi region SKIPPED ({}) — {} slot(s) x {} layer(s) \
             = {} MB not reserved (restore with --enable-prefix-caching, or \
             METRALE_SSM_MARCONI_FULL to over-reserve)",
            reason,
            args.ssm_cache_slots,
            config.num_ssm_layers(),
            (args.ssm_cache_slots * config.num_ssm_layers() * (h_state_bytes + conv_state_bytes))
                / (1024 * 1024),
        );
    }
    // 2026-09-26: One sequence's SSM state across all SSM layers. Marconi
    // reserves one per cache slot; a decode-ring slot is one per batch
    // sequence (`decode_ring::slot_bytes`).
    let per_seq_blob = config.num_ssm_layers() * (h_state_bytes + conv_state_bytes);
    let marconi_bytes = marconi.slots * per_seq_blob;
    let cuda_headroom: usize = if spec_on_pool {
        4 * 1024 * 1024 * 1024
    } else {
        512 * 1024 * 1024
    };
    let gdn_two_phase_bytes: usize = {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        // 2026-10-04: Nothing to reserve when the GLM-5.3 arena trim skips the
        // buffers (`build_gdn_prefill_buffers` in metrale-model-engine).
        if conv_dim > 0
            && config.num_ssm_layers() > 0
            && !metrale_gpu_runtime::buffers::glm_arena_trim_active(config)
        {
            let sl = max_batch_tokens_pre;
            sl * conv_dim * 2 + sl * nv * 2 * 4 + sl * value_dim * 2 + sl * value_dim * 2
        } else {
            0
        }
    };
    // 2026-09-26: Every reserve term except the decode-rollback ring, the only
    // term the auto-fit shrinks.
    let fixed_reserve: usize = ssm_pool_bytes
        + ssm_h_stage_bytes
        + ssm_replay_ring
        + marconi_bytes
        + gdn_two_phase_bytes
        + cuda_headroom
        + per_sequence_reserve(args, config);
    // 2026-09-26: Decode-rollback ring: the requested depth, then the largest
    // depth that fits (`decode_ring::autofit`). A shrunk depth is published
    // (`set_decode_ring_slots`) so `TransformerModel::new` allocates the depth
    // reserved here.
    let ring_requested = decode_ring::requested_slots(args, config);
    let ring_slot_bytes = decode_ring::slot_bytes(args, per_seq_blob);
    // 2026-09-26: The auto-fit's yardstick: predicted post-load KV headroom
    // where the load's residency can be predicted, else pre-load free memory
    // (`headroom.rs`).
    let yardstick =
        headroom::post_load_yardstick(args, config, post_load, fixed_reserve, buffer_arena_bytes);
    let fit = decode_ring::autofit(
        args,
        ring_requested,
        ring_slot_bytes,
        per_seq_blob,
        fixed_reserve + buffer_arena_bytes,
        free_mem,
        &yardstick,
    );
    tracing::info!("{}", fit.decision);
    if let Some(warning) = &fit.warning {
        tracing::warn!("SSM decode-rollback ring auto-fit — {}", warning);
    }
    let ssm_snapshot_bytes = marconi_bytes + fit.slots * ring_slot_bytes;
    let inference_reserve: usize = fixed_reserve + fit.slots * ring_slot_bytes;
    let total_reserve = inference_reserve + buffer_arena_bytes;
    if total_reserve > free_mem {
        return Err(refusal::reserve_refusal(
            args,
            config,
            refusal::Refusal {
                total_reserve,
                free_mem,
                seq_len_independent: ssm_pool_bytes
                    + ssm_h_stage_bytes
                    + ssm_snapshot_bytes
                    + cuda_headroom,
                ring_requested,
                ring_slots: fit.slots,
                per_seq_blob,
                ring_pinned: metrale_model_layers::ssm_reserve::published_decode_ring_slots()
                    .is_some(),
            },
        ));
    }
    tracing::info!(
        "Preflight reserve: inference={} MB, buffer_arena={} MB (pre-load free: {:.1} GB); {}",
        inference_reserve / (1024 * 1024),
        buffer_arena_bytes / (1024 * 1024),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        decode_ring::formula(fit.slots, args.max_batch_size, per_seq_blob),
    );
    let spec_on = spec_on_pool;
    tracing::debug!(
        "Preflight reserve breakdown: \
         ssm_pool={} MB ({} max_batch blobs + {} MTP-covered slots × {} verify blobs, \
         {} ssm_layers × (h+conv)), \
         ssm_snapshot={} MB ({} slots), \
         gdn_two_phase={} MB ({} tokens), \
         cuda_headroom={} MB ({}), \
         spec_on={}, num_drafts={}",
        ssm_pool_bytes / (1024 * 1024),
        args.max_batch_size,
        if spec_on_pool { mtp_state_slots } else { 0 },
        if spec_on_pool {
            args.resolved_num_drafts() + 2
        } else {
            0
        },
        config.num_ssm_layers(),
        ssm_snapshot_bytes / (1024 * 1024),
        marconi.slots,
        gdn_two_phase_bytes / (1024 * 1024),
        max_batch_tokens_pre,
        cuda_headroom / (1024 * 1024),
        if spec_on { "spec/MTP on" } else { "no spec" },
        spec_on,
        if spec_on {
            args.resolved_num_drafts() as i64
        } else {
            -1
        },
    );
    Ok(ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
    })
}

/// 2026-09-26: The DFlash drafter's trained block size,
/// `dflash_config.block_size` in `--draft-model`'s config.json, read without
/// loading the checkpoint. `None` when there is no local config.json with a
/// positive value (an unset flag, an HF id, an absent field).
fn peek_dflash_block_size(draft_model: Option<&str>) -> Option<usize> {
    let dir = std::path::Path::new(draft_model?);
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let g = v.get("dflash_config")?.get("block_size")?.as_u64()? as usize;
    (g > 0).then_some(g)
}

/// 2026-09-26: Rows one sequence's speculative step can occupy.
/// `max_batch_tokens_pre` here and `resolve_prefill_budget` (`kv_cache.rs`)
/// take a max against it.
///
/// DFlash: γ + 1, with γ from `--dflash-gamma`, else `default_dflash_gamma` of
/// the peeked drafter block size, else 16. MTP, self- and n-gram speculation:
/// `num_drafts + 2`. Otherwise 1.
pub(crate) fn spec_reserve_tokens(args: &cli::ServeArgs) -> usize {
    if args.dflash {
        let gamma = args.dflash_gamma.unwrap_or_else(|| {
            peek_dflash_block_size(args.draft_model.as_deref())
                .map(metrale_model_layers::layers::qwen3_ssm::default_dflash_gamma)
                .unwrap_or_else(|| args.resolved_dflash_gamma(None))
        });
        gamma + 1
    } else if args.speculative || args.self_speculative || args.ngram_speculative {
        args.resolved_num_drafts() + 2
    } else {
        1
    }
}
