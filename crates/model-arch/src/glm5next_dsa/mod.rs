// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3-Flash DSA (DeepSeek Sparse Attention): the kernel handles, the
//! per-layer geometry and its validation.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - A config returned by `Glm5NextDsaConfig::from_config` has passed `validate`:
//!   `index_topk` is a nonzero multiple of `index_kpool`, `index_kpool` is at most
//!   `KERNEL_MAX_KPOOL`, `kv_lora_rank` equals `KERNEL_KV_LORA_DIM`, and `local_heads` is
//!   nonzero.
//!
//! The checkpoint is the one `kernels/gb10/glm-5.3-flash/MODEL.toml` names
//! (`LibertAIDAI/GLM-5.3-Flash-NVFP4`, revision `9e0d74e3`). The selection kernels are in
//! `kernels/gb10/common/dsa_indexer.cu`, and `examples/dsa_indexer_microtest.rs` runs them
//! on a GPU. [`crate::glm5next_dsa_ref`] is the CPU reference for the equations.
//!
//! # Shape of the pipeline
//!
//! ```text
//! k,gate,valid,ape -> kpool_compress -> pool keys/indices/valid
//!                  -> index_scores  -> [Q, P] scores + candidate validity
//!                  -> topk_pools    -> [Q, select_k] pool ids
//!                  -> expand_selection -> [Q, out_width] token ids (-1 = invalid)
//!                  -> NoPE MLA restricted to those tokens
//! ```
//!
//! # Checkpoint and kernel facts the code depends on
//!
//! * `indexer.k_norm` is a LayerNorm with a bias. `indexer.k_norm.bias` is the only norm
//!   bias among the checkpoint's text tensors
//!   (`crates/model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-structural.txt`).
//! * The pool softmax runs over the pool-slot axis, per channel (`dsa_kpool_compress`).
//! * Pooling starts at `first_key`. A pool counts only if every one of its `index_kpool`
//!   slots is valid, so a trailing partial pool is not a pool.
//! * NoPE: `qk_rope_head_dim` is 0, so there is no rope section and no `wkv_a_rope`
//!   tensor.
//! * `dsa_expand_selection` writes -1 over the whole index row before it stores any index,
//!   so every slot is written.

use anyhow::{Result, bail};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

pub mod attend;
pub mod aux_state;
pub mod binding;
pub mod build;
pub mod layer;
pub mod lazy;
pub mod pool_cache;
pub mod select;
pub mod state;
pub mod tp;

/// 2026-09-25: Module name the DSA selection kernels resolve from. `dsa_indexer.cu` has no
/// entry in `kernels/gb10/common/KERNEL.toml` `[modules]`, so its module is its file stem.
pub const DSA_MODULE: &str = "dsa_indexer";

/// 2026-09-25: Module of `nllb_layernorm_bf16`, the BF16 LayerNorm with a bias that the
/// indexer's `k_norm` needs: `kernels/gb10/common/nllb_encoder.cu`, named by its file stem.
pub const LAYERNORM_MODULE: &str = "nllb_encoder";

/// 2026-09-25: `#define GLM_KV_LORA_DIM` in
/// `kernels/gb10/glm-5.3-flash/nvfp4/glm5next_dsa_mla_decode.cu`, the latent width the
/// decode kernel is compiled for. `Glm5NextDsaConfig::validate` and `attend::decode_attention`
/// refuse a `kv_lora_rank` that differs.
pub const KERNEL_KV_LORA_DIM: usize = 512;

/// 2026-09-25: `float lg[8]` in `dsa_kpool_compress`, the most pool slots the compression
/// kernel holds. The kernel loops `s < KP && s < 8`, so it would drop slots past 8 without
/// an error; config validation refuses a larger `index_kpool` instead.
pub const KERNEL_MAX_KPOOL: usize = 8;

/// 2026-09-25: The DSA selection kernels, the indexer LayerNorm and the two oracle kernels.
///
/// All but `write_geom` and `indexer_store` are resolved with `kernel()`, so a missing
/// entry point fails `resolve`; those two use `try_kernel` and may be `KernelHandle(0)`.
/// 2026-10-01: So do `index_scores_tiled` and `index_scores_tc`.
/// 2026-10-05: So does `grid_stride_marker`.
/// 2026-10-06: So do the five `topk_radix_*` handles.
/// 2026-10-07: So does `index_scores_tc2`.
#[derive(Clone, Copy)]
pub struct Glm5NextDsaKernels {
    pub kpool_compress: KernelHandle,
    pub compact_pools: KernelHandle,
    pub index_scores: KernelHandle,
    /// 2026-10-01: `dsa_index_scores_tiled`: `index_scores` over 32-row x 64-pool tiles,
    /// byte-identical by construction (argument in `dsa_indexer.cu`). `select_tokens`
    /// launches it under `METRALE_GLM_DSA_SCORES_TILED=1` (`select::scores_tiled_for`);
    /// `KernelHandle(0)` when the target lacks it, and `index_scores` runs.
    pub index_scores_tiled: KernelHandle,
    /// 2026-10-01: `dsa_index_scores_tc`: `index_scores` on tensor cores (BF16 MMA, FP32
    /// accumulate; NOT byte-identical, precision chosen by its `mode` argument). `select_tokens`
    /// launches it under `METRALE_GLM_DSA_SCORES_TC` (`select::scores_tc_for`);
    /// `KernelHandle(0)` when the target lacks it, and the FP32 scorer runs.
    pub index_scores_tc: KernelHandle,
    /// 2026-10-07: `dsa_index_scores_tc2`: the bytes of `index_scores_tc` in mode 1 (`bf16`) on
    /// a faster dataflow. `select_tokens` launches it in place of `index_scores_tc` under
    /// `METRALE_GLM_DSA_SCORES_TC2=1` (`select::tc2::scores_tc2_for`); `KernelHandle(0)` when
    /// the target lacks it, and `index_scores_tc` runs.
    pub index_scores_tc2: KernelHandle,
    /// 2026-10-05: `dsa_indexer_grid_stride_v1`, a no-op kernel that `dsa_indexer.cu` defines
    /// iff `dsa_kpool_compress` and `dsa_index_scores` both walk the live pools with a grid
    /// stride. A ceiling `select_tokens` launches the capped stride grid only when this
    /// resolved (`select::grid_stride::stride_mode`); `KernelHandle(0)` when the target's
    /// module lacks it (a copy of the file without the loop, such as the b300 fork), and the
    /// one-block-per-ceiling-pool grids run. Resolved here, at load, not at the first launch:
    /// a failed lookup after the boot audit seals aborts the process
    /// (`metrale_telemetry::kernel_audit`), and one before it is judged by the target's
    /// MODEL.toml `[expected_absent]`, as for `index_scores_tiled`.
    pub grid_stride_marker: KernelHandle,
    pub topk_pools: KernelHandle,
    /// 2026-10-06: `dsa_topk_radix_{init,hist,find,gather,sort}`: the exact radix top-k that
    /// `select_tokens` launches instead of `topk_pools` under `METRALE_GLM_DSA_TOPK_RADIX=1`
    /// above one top-k tile of pools (`select::radix`). `KernelHandle(0)` when the target's
    /// module lacks them (the b300 copy of `dsa_indexer.cu`); the lever is then ignored, with
    /// one warning, and `topk_pools` runs.
    pub topk_radix_init: KernelHandle,
    pub topk_radix_hist: KernelHandle,
    pub topk_radix_find: KernelHandle,
    pub topk_radix_gather: KernelHandle,
    pub topk_radix_sort: KernelHandle,
    pub expand_selection: KernelHandle,
    /// 2026-09-25: `indexer.k_norm`, a LayerNorm with a bias: `nllb_layernorm_bf16`, in place,
    /// taking `(x, weight, bias, rows, dim, eps)`. An RMSNorm kernel would drop both the mean
    /// subtraction and the bias, and the shapes would not show it.
    pub k_norm: KernelHandle,
    /// 2026-09-25: Derives the selector geometry on the device from `seq_len`, so a captured
    /// graph replays over the live context. When it or `indexer_store` is missing,
    /// `Glm5NextDsaLayer::decode_k` does not take the replay-safe path.
    pub write_geom: KernelHandle,
    /// 2026-09-25: Copies the staged indexer row to a device-side position and marks it
    /// valid.
    pub indexer_store: KernelHandle,
    /// 2026-10-06: `METRALE_GLM_DSA_POOL_CACHE=1` kernels (`pool_cache.rs`), resolved only
    /// with the lever on (`KernelHandle(0)` otherwise, so the lever-off load is unchanged):
    /// `dsa_kpool_compress_incr` (compress pools `[pk_start, live)` from the raw-row ring into
    /// the persistent arrays, then set the device `pk_len`), `dsa_write_geom_pk`
    /// (`dsa_write_geom` plus the `DSA_GEOM_PK_START` slot), `dsa_indexer_store_ring`
    /// (`dsa_indexer_store` at ring slot `pos % ring`, clamping the device `pk_len`) and
    /// `dsa_pk_len_clamp` (the host-path clamp after a rewind).
    pub kpool_compress_incr: KernelHandle,
    pub write_geom_pk: KernelHandle,
    pub indexer_store_ring: KernelHandle,
    pub pk_len_clamp: KernelHandle,
    /// 2026-09-25: Oracle only; no code under `src/` launches it. See [`MASKED_ATTN_MAX_KEYS`].
    pub topk_to_mask: KernelHandle,
    /// 2026-09-25: Oracle only; no code under `src/` launches it. See [`MASKED_ATTN_MAX_KEYS`].
    pub mla_masked_attn: KernelHandle,
}

/// 2026-09-25: The most keys `dsa_mla_masked_attn` takes: it stages the `[S]` f32 score row
/// in dynamic shared memory, and `4 · S <= 49,152` B gives 12,288.
///
/// `examples/glm5next_dsa_decode_gate.rs` checks the serve's decode kernel against it, and
/// `examples/dsa_indexer_microtest.rs` launches it with `dsa_topk_to_mask`. The serve's
/// decode is `glm5next_dsa_mla_decode_fp8` (`attend.rs`), which reads the selected
/// tokens through the block table and has no `S`-sized shared-memory row.
pub const MASKED_ATTN_MAX_KEYS: usize = 12_288;

impl Glm5NextDsaKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            kpool_compress: gpu.kernel(DSA_MODULE, "dsa_kpool_compress")?,
            compact_pools: gpu.kernel(DSA_MODULE, "dsa_compact_pools")?,
            index_scores: gpu.kernel(DSA_MODULE, "dsa_index_scores")?,
            index_scores_tiled: metrale_model_layers::layers::try_kernel(
                gpu,
                DSA_MODULE,
                "dsa_index_scores_tiled",
            ),
            index_scores_tc: metrale_model_layers::layers::try_kernel(
                gpu,
                DSA_MODULE,
                "dsa_index_scores_tc",
            ),
            index_scores_tc2: metrale_model_layers::layers::try_kernel(
                gpu,
                DSA_MODULE,
                "dsa_index_scores_tc2",
            ),
            grid_stride_marker: metrale_model_layers::layers::try_kernel(
                gpu,
                DSA_MODULE,
                "dsa_indexer_grid_stride_v1",
            ),
            topk_pools: gpu.kernel(DSA_MODULE, "dsa_topk_pools")?,
            topk_radix_init: metrale_model_layers::layers::try_kernel(
                gpu,
                DSA_MODULE,
                "dsa_topk_radix_init",
            ),
            topk_radix_hist: metrale_model_layers::layers::try_kernel(
                gpu,
                DSA_MODULE,
                "dsa_topk_radix_hist",
            ),
            topk_radix_find: metrale_model_layers::layers::try_kernel(
                gpu,
                DSA_MODULE,
                "dsa_topk_radix_find",
            ),
            topk_radix_gather: metrale_model_layers::layers::try_kernel(
                gpu,
                DSA_MODULE,
                "dsa_topk_radix_gather",
            ),
            topk_radix_sort: metrale_model_layers::layers::try_kernel(
                gpu,
                DSA_MODULE,
                "dsa_topk_radix_sort",
            ),
            expand_selection: gpu.kernel(DSA_MODULE, "dsa_expand_selection")?,
            k_norm: gpu.kernel(LAYERNORM_MODULE, "nllb_layernorm_bf16")?,
            write_geom: metrale_model_layers::layers::try_kernel(gpu, DSA_MODULE, "dsa_write_geom"),
            indexer_store: metrale_model_layers::layers::try_kernel(
                gpu,
                DSA_MODULE,
                "dsa_indexer_store",
            ),
            kpool_compress_incr: pool_cache_kernel(gpu, "dsa_kpool_compress_incr"),
            write_geom_pk: pool_cache_kernel(gpu, "dsa_write_geom_pk"),
            indexer_store_ring: pool_cache_kernel(gpu, "dsa_indexer_store_ring"),
            pk_len_clamp: pool_cache_kernel(gpu, "dsa_pk_len_clamp"),
            topk_to_mask: gpu.kernel(DSA_MODULE, "dsa_topk_to_mask")?,
            mla_masked_attn: gpu.kernel(DSA_MODULE, "dsa_mla_masked_attn")?,
        })
    }
}

/// 2026-10-06: A pool-cache kernel of [`DSA_MODULE`], looked up only with
/// `METRALE_GLM_DSA_POOL_CACHE=1`; `KernelHandle(0)` otherwise or when absent.
fn pool_cache_kernel(gpu: &dyn GpuBackend, name: &str) -> KernelHandle {
    if pool_cache::dsa_pool_cache() {
        metrale_model_layers::layers::try_kernel(gpu, DSA_MODULE, name)
    } else {
        KernelHandle(0)
    }
}

/// 2026-09-25: DSA geometry for one layer, from `ModelConfig`.
///
/// Attention head counts are per rank (`local_heads`); the indexer is replicated, so
/// `index_heads` is the full count. See [`tp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm5NextDsaConfig {
    pub hidden: usize,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_kpool: usize,
    pub index_topk: usize,
    pub always_select_tail: bool,
    /// 2026-09-25: Attention heads this rank owns.
    pub local_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    /// 2026-09-25: 0 on GLM-5.3 (NoPE). `build::absorb_q` and `build::absorb_o` refuse any
    /// other value.
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    /// 2026-09-25: Tokens a sequence's indexer cache is sized for: `--max-seq-len`, or 16,384
    /// when it is unset (`from_config`). [`state::max_dsa_context`] rounds it down to whole
    /// pools.
    pub max_context: usize,
}

impl Glm5NextDsaConfig {
    /// 2026-09-25: `local_heads` is `config.num_attention_heads`, which the serve's
    /// `serve_phases::topology` has already divided by `tp_size`. Returns the result of
    /// `validate`.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        let c = Self {
            hidden: config.hidden_size,
            index_heads: config.index_n_heads,
            index_head_dim: config.index_head_dim,
            index_kpool: config.index_kpool,
            index_topk: config.index_topk,
            always_select_tail: config.index_kpool_always_select_tail,
            local_heads: config.num_attention_heads,
            q_lora_rank: config.q_lora_rank,
            kv_lora_rank: config.kv_lora_rank,
            qk_nope_head_dim: config.qk_nope_head_dim,
            qk_rope_head_dim: config.qk_rope_head_dim,
            v_head_dim: config.v_head_dim,
            // 2026-09-25: `serve_max_seq_len` is `--max-seq-len` (serve_phases::topology); 0
            // means unset (a unit test, a tool), and the cache is then sized for 16,384 tokens
            // rather than the checkpoint's 1,048,576 `max_position_embeddings`.
            max_context: if config.serve_max_seq_len > 0 {
                config.serve_max_seq_len
            } else {
                16_384
            },
        };
        c.validate()?;
        Ok(c)
    }

    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    /// 2026-09-25: True when there is no RoPE section, as on GLM-5.3.
    pub fn is_nope(&self) -> bool {
        self.qk_rope_head_dim == 0
    }
    /// 2026-09-25: Pools selected per query, capped by how many pools exist.
    pub fn select_k(&self, n_pools: usize) -> usize {
        (self.index_topk / self.index_kpool).min(n_pools)
    }
    /// 2026-09-25: Width of the emitted index row: `index_topk`, plus `index_kpool - 1` tail
    /// slots when `always_select_tail`.
    pub fn out_width(&self) -> usize {
        self.index_topk
            + if self.always_select_tail {
                self.index_kpool - 1
            } else {
                0
            }
    }
    /// 2026-09-25: KV latent cache width. Under NoPE this is `kv_lora_rank`, 512 on GLM-5.3;
    /// the DeepSeek-V4-Flash cache carries a rope tail and is 576 wide (`MLA_CACHE_DIM` in
    /// `kernels/gb10/deepseek-v4-flash/nvfp4/mla_paged_decode.cu`).
    pub fn kv_cache_dim(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    pub fn validate(&self) -> Result<()> {
        if self.index_kpool == 0 || self.index_topk == 0 {
            bail!(
                "DSA needs index_kpool>0 and index_topk>0; got {}/{}",
                self.index_kpool,
                self.index_topk
            );
        }
        if !self.index_topk.is_multiple_of(self.index_kpool) {
            bail!(
                "DSA: index_topk ({}) must be a multiple of index_kpool ({}) — \
                 the pool budget is index_topk/index_kpool",
                self.index_topk,
                self.index_kpool
            );
        }
        // 2026-09-25: `dsa_kpool_compress` holds the pool logits in `float lg[8]` and loops
        // `s < KP && s < 8`, but writes `pool_indices`/`pool_valid` for all KP slots, so a
        // larger kpool would pool only the first 8 slots with no error.
        if self.index_kpool > KERNEL_MAX_KPOOL {
            bail!(
                "DSA: index_kpool {} exceeds the {}-slot bound dsa_kpool_compress keeps \
                 in registers (`float lg[8]`); slots past it are silently dropped from \
                 the pooled key while still counting as valid",
                self.index_kpool,
                KERNEL_MAX_KPOOL,
            );
        }
        if self.kv_lora_rank == 0 {
            bail!("DSA is MLA: kv_lora_rank must be > 0");
        }
        // 2026-09-25: `glm5next_dsa_mla_decode.cu` is compiled for `GLM_KV_LORA_DIM` 512 but
        // takes the cache stride as a runtime argument, so a checkpoint with another latent
        // width would be read at the wrong width with no error. Refused here instead.
        if self.kv_lora_rank != KERNEL_KV_LORA_DIM {
            bail!(
                "GLM-5.3 DSA: kv_lora_rank is {}, but glm5next_dsa_mla_decode hardcodes                  GLM_KV_LORA_DIM={}. The decode kernel would read the latent at the                  wrong width. Parameterise GLM_KV_LORA_DIM in kernels/gb10/\
                 glm-5.3-flash/nvfp4/glm5next_dsa_mla_decode.cu before serving this checkpoint.",
                self.kv_lora_rank,
                KERNEL_KV_LORA_DIM,
            );
        }
        if self.local_heads == 0 {
            bail!("DSA: this rank owns zero attention heads");
        }
        Ok(())
    }
}
