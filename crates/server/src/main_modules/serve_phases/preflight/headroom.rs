// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The yardstick the SSM decode-ring fit is measured against:
//! the predicted post-load KV headroom when it can be predicted, and pre-load
//! free memory otherwise.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `post_load_yardstick` returns `Yardstick::PostLoad` only when
//!   `predicted_derived_bytes` gives a byte count and `checkpoint_bytes`
//!   returns `Some`; otherwise `Yardstick::PreLoadFree` carries the reason.
//!
//! The headroom estimates, before load, the KV budget `factory::build`
//! computes after it:
//!
//! ```text
//! budget   = total_mem x --gpu-memory-utilization
//! pre_kv   = weights + derived + arena + fixed_reserve
//! headroom = budget - pre_kv (saturating)
//! ```
//!
//! On that yardstick the ring is the largest `DECODE_RING_FIT_LADDER` depth
//! with `ring_bytes + kv_floor <= headroom`. `weights` is the checkpoint's
//! on-disk size; `derived` is `predicted_derived_bytes`, which answers only
//! for the native-FP8 dense route.

use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype};
use metrale_config::ModelConfig;

use crate::cli;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// 2026-09-26: Default context length of the KV floor, in tokens per
/// sequence. The floor keeps the fit from giving the ring the room a KV cache
/// needs to admit `--max-batch-size` sequences. `METRALE_KV_FLOOR_TOKENS=<n>`
/// overrides it; 0 disables the floor.
pub(super) const DEFAULT_KV_FLOOR_TOKENS: usize = 4096;

/// 2026-09-26: `METRALE_KV_FLOOR_TOKENS`, or [`DEFAULT_KV_FLOOR_TOKENS`] when
/// unset. An unparseable value logs a warning and takes the default, so a
/// typo does not remove the floor.
pub(super) fn kv_floor_tokens() -> usize {
    match std::env::var("METRALE_KV_FLOOR_TOKENS") {
        Ok(v) => v.trim().parse().unwrap_or_else(|_| {
            tracing::warn!(
                "METRALE_KV_FLOOR_TOKENS='{v}' is not a token count — using the default {}",
                DEFAULT_KV_FLOOR_TOKENS,
            );
            DEFAULT_KV_FLOOR_TOKENS
        }),
        Err(_) => DEFAULT_KV_FLOOR_TOKENS,
    }
}

/// 2026-09-26: What `preflight_reserve` needs from its caller to build the
/// post-load yardstick, beyond `args` and `config`.
pub(crate) struct PostLoadInputs<'a> {
    /// 2026-09-26: `gpu.total_memory()`, the total `factory::build`
    /// multiplies by `--gpu-memory-utilization`.
    pub(crate) total_mem: usize,
    /// 2026-09-26: The resolved checkpoint directory, for the on-disk size.
    pub(crate) model_dir: &'a std::path::Path,
    /// 2026-09-26: The KV dtype, resolved by the caller through
    /// `serve_phases::kv_cache::resolve_kv_dtype_str`.
    pub(crate) kv_dtype: KvCacheDtype,
    /// 2026-09-26: Both `qwen3_attention::W8A8_PREFILL_KERNELS` are loaded.
    /// With the prefill dispatch, it decides whether the loader builds the Q
    /// and O FP8 twins.
    pub(crate) w8a8_prefill_kernels: bool,
}

/// 2026-09-26: Every term of the post-load fit, kept so the decision line can
/// print each one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Headroom {
    /// 2026-09-26: `total_mem x --gpu-memory-utilization`.
    pub(super) budget: usize,
    /// 2026-09-26: Checkpoint bytes on disk.
    pub(super) weights: usize,
    /// 2026-09-26: Predicted bytes of the derived copies the loader builds.
    pub(super) derived: usize,
    /// 2026-09-26: `BufferSizes::from_config(..).total_bytes()`.
    pub(super) arena: usize,
    /// 2026-09-26: The reserve terms that do not scale with ring depth.
    pub(super) fixed: usize,
    /// 2026-09-26: KV bytes for `--max-batch-size` sequences at the floor
    /// context.
    pub(super) kv_floor: usize,
    /// 2026-09-26: `budget - (weights + derived + arena + fixed)`, saturating.
    pub(super) headroom: usize,
}

/// 2026-09-26: What the ring is fitted against.
pub(super) enum Yardstick {
    /// 2026-09-26: Pre-load free memory, for a route whose post-load residency
    /// is not predicted. The string is the reason, for the log.
    PreLoadFree(&'static str),
    /// 2026-09-26: The predicted post-load KV headroom.
    PostLoad(Headroom),
}

impl Yardstick {
    /// 2026-09-26: `(bytes that must fit beside the ring, the limit)`, the
    /// last two quantities `ssm_reserve::fit_decode_ring_slots` takes: the
    /// rest of the reserve and free memory, or the KV floor and the headroom.
    pub(super) fn ladder_basis(
        &self,
        reserve_without_ring: usize,
        free_mem: usize,
    ) -> (usize, usize) {
        match self {
            Self::PreLoadFree(_) => (reserve_without_ring, free_mem),
            Self::PostLoad(h) => (h.kv_floor, h.headroom),
        }
    }

    /// 2026-09-26: The yardstick's name in the shrink warning.
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::PreLoadFree(_) => "pre-load free memory",
            Self::PostLoad(_) => "the predicted post-load KV headroom",
        }
    }

    /// 2026-09-26: The shrink warning's name for the sum checked against the
    /// yardstick.
    pub(super) fn total_label(&self) -> &'static str {
        match self {
            Self::PreLoadFree(_) => "reserve",
            Self::PostLoad(_) => "ring + KV floor",
        }
    }
}

/// 2026-09-26: Build the post-load yardstick, or the pre-load one with the
/// reason. `fixed_reserve` and `arena` are the caller's reserve terms; the
/// weights and derived bytes are estimated here.
pub(super) fn post_load_yardstick(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    inputs: &PostLoadInputs<'_>,
    fixed_reserve: usize,
    arena: usize,
) -> Yardstick {
    let route = metrale_model_arch::weight_loader::predicted_residency::Fp8RouteInputs::from_env(
        config,
        inputs.w8a8_prefill_kernels,
    );
    let estimate = metrale_model_arch::weight_loader::predicted_residency::predicted_derived_bytes(
        config, &route,
    );
    let Some(derived) = estimate.bytes() else {
        return Yardstick::PreLoadFree(
            estimate
                .reason()
                .unwrap_or("the derived-residency prediction declined"),
        );
    };
    let Some(weights) = checkpoint_bytes(inputs.model_dir) else {
        return Yardstick::PreLoadFree("the checkpoint's on-disk size could not be read");
    };
    let budget = (inputs.total_mem as f64 * args.gpu_memory_utilization) as usize;
    let weights = weights as usize;
    let derived = derived as usize;
    let pre_kv = weights
        .saturating_add(derived)
        .saturating_add(arena)
        .saturating_add(fixed_reserve);
    Yardstick::PostLoad(Headroom {
        budget,
        weights,
        derived,
        arena,
        fixed: fixed_reserve,
        kv_floor: kv_floor_bytes(args, config, inputs.kv_dtype),
        headroom: budget.saturating_sub(pre_kv),
    })
}

/// 2026-09-26: KV bytes for `--max-batch-size` sequences at the floor context
/// length, the floor tokens capped at `--max-seq-len`; 0 when either is 0.
pub(super) fn kv_floor_bytes(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    kv_dtype: KvCacheDtype,
) -> usize {
    let tokens = kv_floor_tokens().min(args.max_seq_len);
    if tokens == 0 || args.max_batch_size == 0 {
        return 0;
    }
    args.max_batch_size * tokens * kv_bytes_per_token(args, config, kv_dtype)
}

/// 2026-09-26: Bytes one cached token costs across every attention layer, K
/// and V, priced by `KvCacheConfig::block_bytes_kv_all_layers`, the function
/// `PagedKvCache::compute_num_blocks` divides by. MLA (`kv_lora_rank > 0`)
/// caches one head of `kv_lora_rank + qk_rope_head_dim`, as in
/// `factory::build`'s step 5.
///
/// `layer_dims` is empty because the loader fills `config.kv_layer_dims`
/// after preflight, so every layer takes the global dims. The Qwen3.5-dense
/// loader, the only route with a post-load yardstick, leaves them empty.
pub(super) fn kv_bytes_per_token(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    kv_dtype: KvCacheDtype,
) -> usize {
    let (num_kv_heads, head_dim) = if config.kv_lora_rank > 0 {
        (1, config.kv_lora_rank + config.qk_rope_head_dim)
    } else {
        (config.num_key_value_heads, config.head_dim)
    };
    let block_size = args.block_size.max(1);
    let kv = KvCacheConfig {
        block_size,
        num_kv_heads,
        head_dim,
        num_layers: config.num_attention_layers(),
        dtype: kv_dtype,
        layer_dtypes: Vec::new(),
        layer_dims: Vec::new(),
        cache_blocks_per_seq: None,
        v_aliases_k: metrale_cache::kv_cache::glm_kv_v_alias(&config.model_type),
    };
    kv.block_bytes_kv_all_layers() / block_size
}

/// 2026-09-26: The checkpoint's on-disk byte count: the distinct shards of
/// the safetensors index when there is one, else the `.safetensors` files in
/// the directory, else the `.gguf` files.
///
/// `None` when none of these gives a nonzero total or the directory cannot be
/// read; the caller then uses the pre-load yardstick.
pub(super) fn checkpoint_bytes(model_dir: &std::path::Path) -> Option<u64> {
    if let Some(total) = indexed_shard_bytes(model_dir) {
        return Some(total);
    }
    // 2026-09-26: GGUF files count only when there are no safetensors, so a
    // directory holding both is not counted twice.
    for ext in ["safetensors", "gguf"] {
        let total: u64 = std::fs::read_dir(model_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .is_some_and(|x| x == ext)
            })
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum();
        if total > 0 {
            return Some(total);
        }
    }
    None
}

/// 2026-09-26: Sum of the distinct shard files the index's `weight_map`
/// names; the map names a shard once per tensor. The index is looked up in
/// the order `metrale_model_weights::fast_weights::header::resolve_shards`
/// uses: `model.safetensors.index.json`, then
/// `consolidated.safetensors.index.json`.
fn indexed_shard_bytes(model_dir: &std::path::Path) -> Option<u64> {
    let index = [
        "model.safetensors.index.json",
        "consolidated.safetensors.index.json",
    ]
    .into_iter()
    .map(|n| model_dir.join(n))
    .find(|p| p.exists())?;
    let raw = std::fs::read_to_string(&index).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let map = v.get("weight_map")?.as_object()?;
    let shards: std::collections::HashSet<&str> = map.values().filter_map(|s| s.as_str()).collect();
    let total: u64 = shards
        .iter()
        .filter_map(|s| std::fs::metadata(model_dir.join(s)).ok().map(|m| m.len()))
        .sum();
    (total > 0).then_some(total)
}

impl Headroom {
    /// 2026-09-26: Every term of the decision, in the order the arithmetic
    /// runs.
    pub(super) fn describe(&self, util: f64, ring_bytes: usize, slots: usize) -> String {
        let gb = |b: usize| b as f64 / GIB;
        format!(
            "budget {:.2} GB ({:.0}% util) = weights {:.2} + derived {:.2} + arena {:.2} + \
             fixed reserve {:.2} -> headroom {:.2} GB; ring({}) {:.2} + KV floor {:.2} = {:.2} GB",
            gb(self.budget),
            util * 100.0,
            gb(self.weights),
            gb(self.derived),
            gb(self.arena),
            gb(self.fixed),
            gb(self.headroom),
            slots,
            gb(ring_bytes),
            gb(self.kv_floor),
            gb(ring_bytes + self.kv_floor),
        )
    }
}

#[cfg(test)]
#[path = "headroom_tests.rs"]
mod tests;
