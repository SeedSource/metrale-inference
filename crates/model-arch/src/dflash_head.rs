// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: DFlash block-diffusion draft head (arXiv 2602.06036),
//! implementing [`DraftProposer`].
//!
//! A small Qwen3-style transformer drafter that emits a block of draft tokens
//! in one forward pass with bidirectional in-block attention. It is
//! conditioned on target hidden states captured at `target_layer_ids`,
//! projected by one `fc` layer at the drafter's input.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - `block_g()` stays in `2..=gamma.max(2)` (`set_block_g` clamps).

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use metrale_model_layers::speculative::{DraftProposer, ProposerState};
use metrale_model_layers::weight_map::{DenseWeight, QuantizedWeight};

pub use kernels::DflashKernels;
pub use state::{DflashProposerState, DflashScratch};

/// 2026-09-25: Cross-sequence batch for one drafter forward: per sequence its
/// last token, position, drafter block table (a device pointer) and filled
/// ctx slot count.
///
/// Rows are sequence-major: sequence `i` owns rows `[i * w, (i + 1) * w)` of
/// the scratch buffers, `w = block_g()`. Attention, the KV slot writes and
/// the selector's chain seed are per sequence; the weight-bearing ops run
/// once over all `n * w` rows.
pub(super) struct DflashBatch<'a> {
    pub last_tokens: &'a [u32],
    pub positions: &'a [usize],
    pub block_tables: Vec<DevicePtr>,
    pub ctx_counts: Vec<u32>,
}

/// 2026-09-25: Drafter weight precision. `from_weights` builds the head as
/// `Bf16` and switches it to `Fp8Weights` unless `METRALE_DFLASH_DRAFTER_FP8=0`
/// or the `fp8_gemm_n128_row_scaled` kernels are missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DflashQuantization {
    Bf16,
    /// 2026-09-25: Weight-only FP8: the q/k/v/o/gate/up/down weights get FP8
    /// E4M3 copies with per-row f32 scales at load, and the LM head is FP8
    /// (`lm_head_shared_fp8`). The KV cache stays bf16.
    Fp8Weights,
}

/// 2026-09-25: One drafter layer's weights: the bf16 norms (with per-head Q/K
/// RMSNorm), projections and MLP; FP8 copies of the seven GEMM weights when
/// the head is `Fp8Weights` (`None` otherwise); and the DFlash2 conv weights
/// (`attention_conv.*`, `mlp_conv.*`), `None` when the checkpoint has none.
#[allow(dead_code)]
pub struct DflashLayer {
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,

    pub q_proj_fp8: Option<metrale_model_layers::weight_map::Fp8DenseWeight>,
    pub k_proj_fp8: Option<metrale_model_layers::weight_map::Fp8DenseWeight>,
    pub v_proj_fp8: Option<metrale_model_layers::weight_map::Fp8DenseWeight>,
    pub o_proj_fp8: Option<metrale_model_layers::weight_map::Fp8DenseWeight>,
    pub gate_proj_fp8: Option<metrale_model_layers::weight_map::Fp8DenseWeight>,
    pub up_proj_fp8: Option<metrale_model_layers::weight_map::Fp8DenseWeight>,
    pub down_proj_fp8: Option<metrale_model_layers::weight_map::Fp8DenseWeight>,

    pub attention_conv_base: Option<DenseWeight>,
    pub attention_conv_proj: Option<DenseWeight>,
    pub mlp_conv_base: Option<DenseWeight>,
    pub mlp_conv_proj: Option<DenseWeight>,
}

#[allow(dead_code)]
/// 2026-09-25: The propose graphs per block width (see
/// `BlockDiffusionDraftHead::propose_graphs`). `warmup` counts the eager
/// passes run at a width; its capture starts once the count reaches
/// `DFlashLevers::propose_warmup_n` (`METRALE_DFLASH_PROPOSE_WARMUP_N`,
/// default 2).
#[derive(Default)]
pub struct ProposeGraphs {
    pub by_width: std::collections::HashMap<usize, Vec<metrale_gpu_runtime::gpu::GraphHandle>>,
    pub warmup: std::collections::HashMap<usize, usize>,
}

/// 2026-09-25: Block-diffusion draft head; its public API is the
/// [`DraftProposer`] trait. It uses the target's token embedding and LM head
/// (`embed_tokens_shared`, `lm_head_shared` / `lm_head_nvfp4`) alongside the
/// drafter checkpoint's own `fc`, `hidden_norm`, `norm` and layers. The
/// geometry fields come from the drafter's config.
pub struct BlockDiffusionDraftHead {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    /// 2026-09-25: The drafter's widest block (rows per sequence): the launch
    /// `--dflash-gamma`, or `default_dflash_gamma` of the checkpoint's block
    /// size. The gamma-sized buffers (scratch bands, the drafter KV pool) are
    /// sized from it. The block a propose runs is [`Self::block_g`].
    pub gamma: usize,
    /// 2026-09-25: The block width of the propose in flight (rows per
    /// sequence: the anchor plus `block_g - 1` mask rows), set by
    /// `set_block_g` from the scheduler's `num_drafts` and read through
    /// `block_g()`.
    /// provenance-id: 526f6e616c6420522e205374657369616b
    pub(super) block_gamma: std::sync::atomic::AtomicUsize,
    /// 2026-09-25: Widest cross-sequence batch the scratch bands can hold.
    pub(super) max_batch: usize,
    pub mask_token_id: u32,
    /// 2026-10-01: The drafter's learned mask embedding, replicated
    /// `mask_rows_count` times as contiguous BF16 rows `[mask_rows_count,
    /// hidden_size]`. `Some` only under `METRALE_DFLASH_CKPT_ARCH=1` with a
    /// `mask_embedding.pt`; `block_embed` then writes it over each draft
    /// block's mask rows instead of the target's `embed_tokens[mask_token_id]`.
    pub mask_rows: Option<DevicePtr>,
    /// 2026-10-01: Rows in `mask_rows`: `gamma - 1`, at least 1; 0 when
    /// `mask_rows` is `None`.
    pub mask_rows_count: usize,
    pub window_size: Option<usize>,
    /// 2026-09-25: The target layers whose hidden states the drafter is
    /// conditioned on, from the drafter config's `dflash_config.target_layer_ids`.
    pub target_layer_ids: Vec<usize>,
    /// 2026-09-25: The target's hidden size; `fc` takes
    /// `target_layer_ids.len() * target_hidden_size` inputs.
    pub target_hidden_size: usize,

    /// 2026-09-25: The target's token embedding table (the drafter checkpoint
    /// has none).
    pub embed_tokens_shared: DevicePtr,
    /// 2026-09-25: The target's bf16 LM head; not used when `lm_head_nvfp4` is
    /// `Some`.
    pub lm_head_shared: DevicePtr,
    /// 2026-09-25: The target's NVFP4 LM head, when it ships one; the final
    /// logits GEMM then uses it instead of `lm_head_shared`.
    pub lm_head_nvfp4: Option<QuantizedWeight>,
    /// 2026-09-25: The FP8 LM head for `Fp8Weights`: the target checkpoint's
    /// native FP8 head when the loader passes one, else an FP8 copy of
    /// `lm_head_shared`. `None` on the bf16 path.
    pub lm_head_shared_fp8: Option<metrale_model_layers::weight_map::Fp8DenseWeight>,

    // 2026-09-25: From the drafter checkpoint: `hidden_norm` (applied to the
    // projected target context), the final `norm`, and `fc`, which projects
    // the stacked captured target hiddens into the drafter's hidden size.
    pub hidden_norm: DenseWeight,
    pub norm: DenseWeight,
    pub fc: DenseWeight,
    /// 2026-09-25: A draft-vocab to target-vocab id remap; `from_weights` sets
    /// `None`.
    pub draft_id_to_target_id: Option<DevicePtr>,
    pub layers: Vec<DflashLayer>,

    /// 2026-09-25: Every drafter layer's K and V projection weights in one bf16
    /// matrix `[L * 2 * kv_dim, h]`, rows `[K0; V0; K1; V1; ...]`, stitched from
    /// `layers[i]` at construction, so `precompute_ctx_kv` computes every
    /// layer's ctx K/V with one GEMM.
    pub fused_kv_weight: Option<DevicePtr>,

    /// 2026-09-25: The drafter's paged bf16 KV cache: one cache holding all
    /// `num_layers` drafter layers, block-table-keyed like the target's.
    pub kv_cache: Mutex<PagedKvCache>,

    /// 2026-09-25: Per-step scratch buffers, allocated once at construction.
    pub scratch: DflashScratch,

    pub kernels: DflashKernels,

    /// 2026-09-25: The model's `max_seq_len`; with `dflash_ctx_cap()` and the
    /// request's budget it bounds each sequence's ctx accumulator.
    pub max_seq_len: usize,

    /// 2026-09-25: The RoPE inverse-frequency table, f32 `[rotary_dim / 2]` on
    /// the device, YaRN-corrected when the drafter config has `rope_scaling`.
    pub yarn_inv_freq: DevicePtr,

    pub rope_theta: f32,

    /// 2026-09-25: RoPE covers the whole head (`rotary_dim = head_dim`).
    pub rotary_dim: usize,

    /// 2026-09-25: RMSNorm epsilon; `from_weights` sets 1e-6.
    pub rms_norm_eps: f32,

    /// 2026-09-25: Most past target positions the drafter attends per step
    /// (`METRALE_DFLASH_CTX_WINDOW`, default 4096).
    pub ctx_window: usize,

    /// 2026-09-25: The piecewise propose graphs, keyed by block width
    /// (`block_g`): a capture bakes its row count, so each width gets its own,
    /// captured once after the warm-up. Each width holds `2 * num_layers + 1`
    /// handles, `[pre_0, post_0, ..., pre_{N-1}, post_{N-1}, tail]` (the tail
    /// is the final norm, LM head and argmax). Attention runs eagerly between
    /// a layer's halves and is never captured. `GraphHandle(0)` marks an empty
    /// capture; that slot runs eagerly.
    pub propose_graphs: Mutex<ProposeGraphs>,
    /// 2026-09-25: When set, `forward_block` runs eagerly and captures nothing.
    pub suppress_graphs: std::sync::atomic::AtomicBool,
    /// 2026-09-25: Diagnostic and A/B levers, resolved from the environment
    /// once when the head was built; `forward_block` and its per-layer
    /// helpers read these instead of the environment. See
    /// [`levers::DFlashLevers`].
    pub levers: levers::DFlashLevers,
    /// 2026-09-25: Not read; the warm-up is counted per width in
    /// `ProposeGraphs::warmup`.
    pub propose_warmup_count: std::sync::atomic::AtomicUsize,

    pub quant: DflashQuantization,

    /// 2026-09-25: DSpark Markov head rank, 0 when the drafter has no Markov
    /// head (the `markov_w*` tensors absent).
    pub markov_rank: usize,
    pub markov_w1: Option<DenseWeight>,
    pub markov_w2: Option<DenseWeight>,
    pub confidence_proj: Option<DenseWeight>,
    pub confidence_bias: Option<DenseWeight>,
    /// 2026-09-25: The drafter config's `confidence_head_with_markov`.
    pub confidence_with_markov: bool,
    /// 2026-09-25: The shifted-row convention (drafter config
    /// `dflash_config.projector_type == "dspark"`): row j's output is the
    /// token at position j + 1, so the draft vector is rotated right by one.
    /// `METRALE_DSPARK_SHIFT=0|1` overrides it.
    pub shifted_rows: bool,

    // 2026-09-25: DFlash2 geometry from the drafter's `dflash_config` (0 when
    // absent) and its candidate-selector weights (`None` when absent).
    pub conv_kernel_size: usize,
    pub conv_group_size: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub selector_pred: Option<DenseWeight>,
    pub selector_succ: Option<DenseWeight>,
    pub selector_hidden_proj: Option<DenseWeight>,
}

mod dflash2;
/// 2026-09-25: Rows the batched ctx precompute staging holds per step.
pub(super) const PRECOMPUTE_BATCH_ROWS: usize = 256;

/// 2026-09-25: Whether the batched ctx precompute (one fc + fused-KV pass over
/// every sequence's uncommitted ctx rows) runs on the batched propose path:
/// yes unless `METRALE_DFLASH_NO_BATCHED_PRECOMPUTE=1`, which runs the
/// per-sequence loop.
pub(super) fn batched_precompute_enabled() -> bool {
    std::env::var("METRALE_DFLASH_NO_BATCHED_PRECOMPUTE")
        .ok()
        .as_deref()
        != Some("1")
}

/// 2026-09-25: The `METRALE_DFLASH_OPTION_B` predicate (`DFlashLevers::option_b`):
/// on unless the value is `0`. Pure over the raw value, so `option_b_tests`
/// tests it without setting the process environment.
pub(super) fn option_b_from(v: Option<&str>) -> bool {
    v != Some("0")
}

#[cfg(test)]
mod option_b_tests;

mod forward_block;
mod forward_block_layer;
mod forward_block_layer_paged;
mod from_weights;
mod kernels;
pub mod levers;
mod markov;
mod precompute_ctx_kv;
mod precompute_ctx_kv_batched;
mod propose;
mod proposer;
mod state;

/// 2026-09-25: Read once: false only when `METRALE_NO_DFLASH_FP8_RT=1`, so the
/// kernel choice stays fixed across CUDA-graph captures.
pub(crate) fn fp8_rt_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_DFLASH_FP8_RT").as_deref() != Ok("1"))
}

/// 2026-09-25: The DFlash context-window bound, in tokens: the most recent
/// target positions the drafter may accumulate. Both the per-sequence ctx
/// accumulator (`proposer.rs`) and the model's whole-prompt hidden capture
/// (`metrale-model-engine`'s `impl_a1.rs`) are sized from it.
///
/// `METRALE_DFLASH_CTX_CAP=<tokens>` (default 16384); `0` disables the cap.
pub fn dflash_ctx_cap() -> usize {
    std::env::var("METRALE_DFLASH_CTX_CAP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16384)
}

impl BlockDiffusionDraftHead {
    /// 2026-09-25: Block width (rows per sequence) of the propose in flight.
    #[inline]
    pub fn block_g(&self) -> usize {
        self.block_gamma.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// 2026-09-25: Arm the block width for the next propose from the
    /// scheduler's draft count: `num_drafts + 1` rows (anchor + masks),
    /// clamped to `2..=gamma.max(2)`, so a request above the head's sizing
    /// gets the widest block.
    #[inline]
    pub(super) fn set_block_g(&self, num_drafts: usize) {
        let g = (num_drafts + 1).clamp(2, self.gamma.max(2));
        self.block_gamma
            .store(g, std::sync::atomic::Ordering::Relaxed);
    }
}
