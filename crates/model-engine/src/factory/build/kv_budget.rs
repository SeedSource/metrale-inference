// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: KV-budget terms for `build_model`: this process's own device
//! footprint, and the reserves for the DFlash drafter and the MTP propose pool,
//! which allocate after the KV cache is sized.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - These functions read and log; none allocates device memory.

use metrale_cache::kv_cache::{KvCacheConfig, PagedKvCache};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layers::MtpQuantization;
use metrale_model_layers::weight_map::MtpWeights;
use metrale_model_weights::weights::WeightStore;

use crate::factory::DflashBuildArgs;

/// 2026-09-26: `used_so_far` as the bytes this process holds, from the first
/// source that applies; the input is total minus free device memory.
pub(super) fn self_relative_used(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    build_entry_free: Option<usize>,
    actual_free: usize,
    mut used_so_far: usize,
    gib: impl Fn(usize) -> f64,
) -> usize {
    // 2026-09-25: `used_so_far` (total − free) also counts other processes on
    // the device. It is replaced by this process's own footprint, from the
    // first source that applies:
    //   1. `METRALE_KV_EXTERNAL_RESERVE_GB` > 0: subtract that many GB from
    //      `used_so_far`, reserving room for co-tenants.
    //   2. The backend's alloc ledger (`gpu.live_bytes()`): bytes this backend
    //      allocated and has not freed. It does not count page cache or
    //      co-tenants; driver context and library workspaces are not ledgered.
    //   3. `baseline_free_bytes()` (free memory at context init) minus free
    //      memory now, capped as described below.
    // The clamp to the free memory left after the reserves applies whichever
    // path set `used_so_far`.
    let manual_reserve_gb = std::env::var("METRALE_KV_EXTERNAL_RESERVE_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&gb| gb > 0.0);
    if let Some(gb) = manual_reserve_gb {
        let ext = (gb * 1024.0 * 1024.0 * 1024.0) as usize;
        let discounted = used_so_far.saturating_sub(ext);
        tracing::info!(target: "metrale_model_engine::factory::build", "METRALE_KV_EXTERNAL_RESERVE_GB={gb} (manual override): discounting \
             external/co-tenant memory from KV budget — used_so_far {:.1} GB → \
             Metrale-own {:.1} GB",
            gib(used_so_far),
            gib(discounted),
        );
        used_so_far = discounted;
    } else if let Some(ledger_live) = gpu.live_bytes() {
        // 2026-09-25: A ledger value between 0 and `used_so_far` is charged
        // as is.
        if ledger_live > 0 && ledger_live <= used_so_far {
            tracing::info!(target: "metrale_model_engine::factory::build", "KV budget self-relative (ledger): Metrale-own {:.1} GB live in \
                 the alloc ledger; {:.1} GB of co-tenant/page-cache use \
                 excluded (set METRALE_KV_EXTERNAL_RESERVE_GB to override)",
                gib(ledger_live),
                gib(used_so_far - ledger_live),
            );
            used_so_far = ledger_live;
        } else if ledger_live > used_so_far {
            // 2026-09-25: Device usage cannot be below what this backend
            // holds, so `free_memory()` over-reported. Charging the larger
            // ledger figure can only shrink the KV budget.
            tracing::warn!(target: "metrale_model_engine::factory::build", "KV budget: free_memory() looks wrong — the alloc ledger holds \
                 {:.1} GB but the device implies only {:.1} GB used, and real \
                 device usage can never be below the ledger. Charging the \
                 ledger's {:.1} GB (the larger, safer figure) instead.",
                gib(ledger_live),
                gib(used_so_far),
                gib(ledger_live),
            );
            used_so_far = ledger_live;
        } else {
            // 2026-09-25: `ledger_live == 0`: nothing ledgered, so the raw
            // figure stays.
            tracing::warn!(target: "metrale_model_engine::factory::build", "KV budget: alloc ledger reports 0 GB live against {:.1} GB \
                 used on the device — using raw used_so_far",
                gib(used_so_far),
            );
        }
    } else if let Some(baseline) = metrale_gpu_runtime::gpu::baseline_free_bytes() {
        // 2026-09-25: Bytes this process consumed since context init.
        let metrale_own = baseline.saturating_sub(actual_free);
        // 2026-09-25: That delta also counts the weight loader's transient
        // memory still resident now. The charge is the smaller of it and the
        // known footprint: `store.total_bytes()` plus what `build_model`
        // allocated since `build_entry_free` (the LoRA pool and the arena).
        let build_own = build_entry_free
            .map(|e| e.saturating_sub(actual_free))
            .unwrap_or(0);
        let known_own = store.total_bytes().saturating_add(build_own);
        let settled = metrale_own.min(known_own);
        // 2026-09-25: A charge that is 0 or above `used_so_far` is
        // implausible; the raw `used_so_far` then stays.
        if settled > 0 && settled <= used_so_far {
            tracing::info!(target: "metrale_model_engine::factory::build", "KV budget self-relative (auto): baseline-free {:.1} GB − free-now \
                 {:.1} GB = {:.1} GB measured; charging settled Metrale-own {:.1} GB \
                 (weights {:.1} GB + build allocs {:.1} GB, loader transient \
                 {:.1} GB released from the charge); co-tenants {:.1} GB excluded \
                 (set METRALE_KV_EXTERNAL_RESERVE_GB to override)",
                gib(baseline),
                gib(actual_free),
                gib(metrale_own),
                gib(settled),
                gib(store.total_bytes()),
                gib(build_own),
                gib(metrale_own.saturating_sub(settled)),
                gib(used_so_far - metrale_own.min(used_so_far)),
            );
            used_so_far = settled;
        } else {
            tracing::warn!(target: "metrale_model_engine::factory::build", "KV budget auto-measure implausible (baseline {:.1} GB, free-now \
                 {:.1} GB, used {:.1} GB) — using raw used_so_far",
                gib(baseline),
                gib(actual_free),
                gib(used_so_far),
            );
        }
    }
    used_so_far
}

/// 2026-09-26: Bytes reserved for the DFlash drafter head, which allocates
/// after KV sizing; `0` without DFlash.
pub(super) fn dflash_reserve_bytes(
    dflash_args: &Option<DflashBuildArgs<'_>>,
    config: &ModelConfig,
    max_seq_len: usize,
    gib: impl Fn(usize) -> f64,
) -> usize {
    let dflash_reserve: usize = dflash_args
        .as_ref()
        .map(|a| {
            let c = &a.drafter_config;
            let kv_dim = c.num_key_value_heads * c.head_dim;
            let drafter_kv = max_seq_len * c.num_hidden_layers * 2 * kv_dim * 2;
            let fused_kv = c.num_hidden_layers * 2 * kv_dim * c.hidden_size * 2;
            let capture = max_seq_len * config.hidden_size * 2;
            // 2026-09-25: The same test as the allocation in
            // dflash_head/from_weights.rs: on unless the variable is `0`.
            let fp8_mirrors =
                if std::env::var("METRALE_DFLASH_DRAFTER_FP8").ok().as_deref() != Some("0") {
                    a.drafter_store.total_bytes() / 2
                } else {
                    0
                };
            drafter_kv + fused_kv + capture + fp8_mirrors + (300 << 20)
        })
        .unwrap_or(0);
    if dflash_reserve > 0 {
        tracing::info!(target: "metrale_model_engine::factory::build", "KV budget: reserving {:.1} GB for post-sizing DFlash drafter allocations",
            gib(dflash_reserve),
        );
    }
    dflash_reserve
}

/// 2026-09-26: Bytes reserved for the MTP head's paged KV pool, which
/// allocates after KV sizing; `0` when no MTP head is built.
pub(super) fn mtp_pool_reserve_bytes(
    use_speculative: bool,
    mtp_weights: &[MtpWeights],
    effective_mtp_quant: MtpQuantization,
    config: &ModelConfig,
    kv_config: &KvCacheConfig,
    kv_budget: usize,
    max_seq_len: usize,
) -> usize {
    if use_speculative && !mtp_weights.is_empty() {
        // 2026-09-25: As the head's kv_config in mtp_head/new.rs: block 16,
        // the target's attention dims, K and V, BF16 for Bf16/Fp8 heads and
        // FP8 for NVFP4.
        let block = 16usize;
        let per_seq_blocks = max_seq_len / block + 1;
        let dense_head = mtp_weights.first().is_some_and(|w| w.dense_ffn.is_some());
        let elem = match effective_mtp_quant.effective_for_head(dense_head) {
            MtpQuantization::Nvfp4 => 1usize,
            MtpQuantization::Fp8 | MtpQuantization::Bf16 => 2,
        };
        let mtp_block_bytes = block * config.num_key_value_heads * config.head_dim * elem * 2;
        let blocks0 = PagedKvCache::compute_num_blocks(kv_config, kv_budget).unwrap_or(0);
        let pool_blocks = per_seq_blocks
            .saturating_mul(metrale_model_layers::speculative::mtp_max_seqs())
            .min(blocks0.max(per_seq_blocks));
        pool_blocks * mtp_block_bytes
    } else {
        0
    }
}

/// 2026-10-01: A59: the GLM-5.3 MTP head allocates its capture, its private KV pool and a
/// drafter indexer cache after the KV pool is sized, unreserved. Under the chunked capture
/// (`METRALE_GLM_MTP_CHUNKED_CAPTURE=1`, `model::mtp_stage`) they are reserved here, priced
/// by `mtp_stage::glm_mtp_reserve_bytes`; with the lever off this returns 0, the old sizing.
pub(super) fn glm_mtp_reserve(
    config: &ModelConfig,
    has_glm_mtp: bool,
    max_seq_len: usize,
    stage_rows: usize,
) -> usize {
    let levers = metrale_model_layers::layers::ops::ModelLevers::get();
    let carry_on = metrale_model_layers::mtp_carry::mtp_carry_drafter_enabled(levers);
    let mt = &config.model_type;
    if !has_glm_mtp || !crate::model::mtp_stage::glm_mtp_chunked_capture(mt, carry_on) {
        return 0;
    }
    let bytes = crate::model::mtp_stage::glm_mtp_reserve_bytes(
        config.hidden_size,
        config.kv_lora_rank,
        config.index_head_dim,
        max_seq_len,
        stage_rows,
        metrale_cache::kv_cache::glm_kv_v_alias(mt),
    );
    tracing::info!(
        "KV budget: reserving {:.2} GB for the GLM MTP head (staging capture of {} rows, \
         drafter KV and indexer; A59)",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        stage_rows.min(max_seq_len),
    );
    bytes
}
