// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3-Flash MLP: the dense SwiGLU FFN and the routed NVFP4 MoE, their kernels and per-rank geometry.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `Glm5NextMlpConfig::from_config` returns a config only when the dense and shared widths
//!   divide over `tp_world_size`, `num_experts` divides over `ep_world_size`, and `validate`
//!   passes.
//! - For such configs, the `local_expert_range`s of ranks `0..ep_world_size` partition
//!   `0..num_experts`, and `local_slot` is `None` for every id outside this rank's range.
//!
//! # One all-reduce for both EP and TP
//!
//! With TP or EP above 1, a routed site's output is a partial sum on every rank: the shared
//! expert is TP-sharded (`local_shared_intermediate`) and each rank runs only the experts it
//! owns. `forward::forward_moe` adds the shared output and the routed slots in one combine
//! kernel, and the layer then reduces that output once. The combine has to come before the
//! reduce: adding a TP-sharded shared expert after the reduce would drop the other ranks'
//! part of it.
//!
//! # Numerics the kernels rely on
//!
//! * The SwiGLU clamp is asymmetric: `gate` is bounded above only, `up` on both sides. The
//!   limit is `ModelConfig::swiglu_limit`, which the `glm5_next` parser refuses to default.
//! * The router's correction bias steers selection only. The emitted weight is the chosen
//!   expert's unbiased sigmoid score.
//! * `routed_scaling_factor` multiplies the top-k weights; the shared expert is added unscaled.
//! * Every rank holds the whole router and ranks all `num_experts`, so for the same input every
//!   rank selects the same ids. Sharding the router would give each rank different partial
//!   logits and a different top-k.
//! * `num_experts` is the full routed-expert count; `local_experts` is this rank's share.
//!   `glm5next_router_topk` must be given the full count.

use anyhow::{Result, bail};
use metrale_config::{Glm5NextRouterMode, ModelConfig};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

pub mod build;
pub mod expert_tp;
#[cfg(test)]
mod expert_tp_tests;
pub mod forward;
pub mod forward_prefill_gemm;
pub mod weights;

pub use weights::{Glm5NextDenseMlpWeights, Glm5NextExpertWeights, Glm5NextMoeWeights};

/// 2026-09-25: Module of `kernels/gb10/common/glm5next_ffn.cu`. A `.cu` file not listed in
/// `common/KERNEL.toml`'s `[modules]` resolves under its file stem; the listed ones below do
/// not, so each module name here is checked against that table.
pub const FFN_MODULE: &str = "glm5next_ffn";
/// 2026-09-25: `[modules]`: `dense_gemm_bf16 = "gemm"`.
pub const GEMM_MODULE: &str = "gemm";
/// 2026-09-25: `[modules]`: `w4a16_gemm = "w4a16"`.
pub const W4A16_MODULE: &str = "w4a16";
/// 2026-09-25: Module of `w4a16_gemv.cu` (not in `[modules]`, so its file stem). Its GEMV
/// kernels run every routed-expert projection except the grouped prefill GEMM.
pub const W4A16_GEMV_MODULE: &str = "w4a16_gemv";
/// 2026-09-25: `[modules]`: `moe_permute = "moe"`, the token sort / permute / unpermute kernels.
pub const MOE_MODULE: &str = "moe";
/// 2026-09-25: `[modules]`: `moe_w4a16_grouped_gemm = "moe_w4a16"`, the tensor-core grouped W4A16 GEMM.
pub const MOE_GROUPED_MODULE: &str = "moe_w4a16";

/// 2026-10-05: Under expert-TP (`METRALE_GLM_EXPERT_TP`), the half width `moe_intermediate_size
/// / 2` must be a multiple of this: the CUTLASS W4A4 prefill tile's K and N (128), which also
/// covers the W4A16 MMA prefill (64) and the NVFP4 scale group (16).
pub const EXPERT_TP_WIDTH_ALIGN: usize = 128;

/// 2026-09-25: The most experts `glm5next_router_topk` can select per token: its per-token
/// selection lives in the shared arrays `sel_id[16]` and `sel_w[16]`.
pub const KERNEL_MAX_TOP_K: usize = 16;

/// 2026-09-25: The kernels a GLM MLP site launches.
///
/// `resolve` fails when a `gpu.kernel` entry point is missing (the GEMM/GEMV bases, `w4a16`,
/// `w4a16_gemv` and the three `glm5next_ffn.cu` kernels). The `try_kernel` ones are
/// `KernelHandle(0)` when absent, and the forward takes another path for each.
#[derive(Clone, Copy)]
pub struct Glm5NextMlpKernels {
    /// 2026-09-25: BF16 `C = A @ B^T` tile GEMM: dense FFN and shared expert.
    pub gemm: KernelHandle,
    /// 2026-09-25: The same GEMM with FP32 output, used for the router, because
    /// `glm5next_router_topk` reads FP32 logits.
    pub gemm_f32: KernelHandle,
    /// 2026-09-25: M=1 GEMVs for `gemm` / `gemm_f32`. `gemv_f32` is optional; when it is 0,
    /// the router's M=1 calls run the tile GEMM (`ops::dense_mm_bf16`).
    pub gemv: KernelHandle,
    pub gemv_f32: KernelHandle,
    /// 2026-09-25: `dense_gemv_bf16_batchm`: 2 to `DENSE_GEMV_BATCHM_MAX_M` (16) rows in one
    /// weight sweep, for the dense FFN and shared expert. When 0, those rows run the tile GEMM.
    pub gemv_batchm: KernelHandle,
    /// 2026-09-25: NVFP4 `w4a16_gemm` tile GEMM. Resolved, but not launched by this module.
    pub w4a16: KernelHandle,
    /// 2026-09-25: NVFP4 `C[1, N] = A[1, K] @ B[N, K]^T`, the per-expert decode GEMV.
    ///
    /// Its grid is tied to the kernel's `N_PER_BLOCK`; use `ops::w4a16_gemv_grid_x`.
    pub w4a16_gemv: KernelHandle,
    /// 2026-09-25: Single-warp-per-output variant of `w4a16_gemv`, checked bit-identical to it
    /// by `examples/w4a16_gemv_sw_microtest.rs`. When 0, the base kernel runs.
    ///
    /// Its grid is `ceil(N/8)` (`N_PER_BLOCK_SW`), not the base kernel's `ceil(N/4)`; launch it
    /// through `ops::w4a16_gemv_sw_raw` or `ops::w4a16_decode_gemv`, which pick the grid.
    pub w4a16_gemv_sw: KernelHandle,
    /// 2026-09-25: All `top_k` slots of one row in one launch, grid `(ceil(N/8), top_k, 1)`.
    /// Weights come from the global-id pointer tables indexed by the router's on-device ids.
    /// When 0, the forward reads the ids back to the host and launches per local expert.
    pub w4a16_gemv_sw_moe: KernelHandle,
    /// 2026-09-25: Row-batched `w4a16_gemv_sw_moe`, indexed `[rows - 2]` for rows 2..=8
    /// (`w4a16_gemv_sw_moe_batchm_m2` .. `_m8`). Each expert in the union of the rows'
    /// selections is swept once for all the rows that picked it.
    ///
    /// grid.y is the union entry, not the slot, with extent `rows * top_k`; unfilled entries
    /// return on `u_eid < 0`. Needs [`Self::moe_row_union`].
    pub w4a16_gemv_sw_moe_batchm: [KernelHandle; 7],
    /// 2026-09-25: Builds the union table the batched kernel indexes: one block of
    /// `rows * top_k` threads. `forward_moe` uses it only when `rows * top_k` is at most
    /// `MOE_ROW_UNION_MAX_IDS` (64).
    pub moe_row_union: KernelHandle,
    /// 2026-09-25: `glm5next_swiglu_clamp`, the asymmetric clamped SwiGLU. `moe_silu_mul`
    /// does not clamp.
    pub swiglu: KernelHandle,
    /// 2026-10-06: `glm5next_swiglu_clamp_amax` (`METRALE_GLM_MOE_SWIGLU_AMAX`): [`Self::swiglu`]
    /// plus the max |output| into a device amax slot; 0 when the PTX lacks it (the lever then
    /// stays off).
    pub swiglu_amax: KernelHandle,
    pub router: KernelHandle,
    pub combine: KernelHandle,
    /// 2026-09-25: `moe_sort_by_expert` (`moe_permute.cu`): counting sort of the `[rows, top_k]`
    /// ids into expert-contiguous order, writing `sorted_token_ids`, `expert_offsets` and
    /// `token_to_perm`. When 0, the grouped prefill path is off.
    pub moe_sort_by_expert: KernelHandle,
    /// 2026-09-25: The tensor-core grouped W4A16 GEMM (`mma.sync.aligned.m16n8k16`) for
    /// prefill, at the tile `forward_prefill_gemm::gemm_tile` picks. When 0, the grouped
    /// prefill path is off.
    pub moe_grouped_gemm: KernelHandle,
    /// 2026-09-29: The geometry of the kernel in `moe_grouped_gemm`: `gemm_tile()`, or
    /// `GEMM_TILES[0]` when that tile is missing from the PTX. The dispatch sizes the grid from
    /// it, so a fallback never runs the base kernel on another tile's grid.
    pub(crate) moe_grouped_tile: forward_prefill_gemm::GemmTile,
    /// 2026-09-25: [`Self::combine`] reading the routed rows in expert-sorted order through
    /// `token_to_perm`, with the same accumulation order and single rounding.
    pub combine_indexed: KernelHandle,
    /// 2026-10-06: `glm5next_moe_combine_indexed_local` (`METRALE_GLM_MOE_COMBINE_LOCAL`):
    /// [`Self::combine_indexed`] skipping the slots of experts another EP rank owns; 0 when the
    /// PTX lacks it (the lever then stays off).
    pub combine_indexed_local: KernelHandle,
    /// 2026-09-30: `moe_permute_tokens` (`moe_permute.cu`, same module as
    /// [`Self::moe_sort_by_expert`]): `permuted[row] = hidden_states[sorted_token_ids[row]]`,
    /// the `METRALE_GLM_MOE_PREFILL_PERMUTE=1` gather-once lever for gate/up
    /// (`forward_prefill_gemm::prefill_gemm_permute`). When 0, the dispatch never takes the
    /// lever even if the env var is set.
    pub moe_permute_tokens: KernelHandle,
    /// 2026-10-01: `METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1`: the register-dequant grouped W4A16
    /// prefill (`forward_prefill_gemm/w4a16_mma.rs`). All handles 0 when the lever is off.
    pub(crate) moe_prefill_mma: forward_prefill_gemm::PrefillMmaKernels,
}

impl Glm5NextMlpKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        // 2026-09-25: Each tile is its own entry point, so the tile
        // (`METRALE_GLM_MOE_GEMM_TILE`) is read before resolving. A missing tile
        // other than the base falls back to `GEMM_TILES[0]`, with a warning, and the
        // dispatch then launches with the base tile's geometry.
        let (moe_grouped_gemm, moe_grouped_tile) = {
            let tile = forward_prefill_gemm::gemm_tile();
            let h = metrale_model_layers::layers::try_kernel(gpu, MOE_GROUPED_MODULE, tile.name);
            if h.0 == 0 && tile.name != forward_prefill_gemm::GEMM_TILES[0].name {
                tracing::warn!(
                    "GLM routed-MoE grouped GEMM tile `{}` is not in this target's PTX — \
                     falling back to `{}`",
                    tile.name,
                    forward_prefill_gemm::GEMM_TILES[0].name
                );
                (
                    metrale_model_layers::layers::try_kernel(
                        gpu,
                        MOE_GROUPED_MODULE,
                        forward_prefill_gemm::GEMM_TILES[0].name,
                    ),
                    forward_prefill_gemm::GEMM_TILES[0],
                )
            } else {
                (h, tile)
            }
        };
        Ok(Self {
            gemm: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16")?,
            gemm_f32: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16_f32out")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            gemv_batchm: metrale_model_layers::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            gemv_f32: metrale_model_layers::layers::try_kernel(
                gpu,
                "gemv",
                "dense_gemv_bf16_fp32out",
            ),
            w4a16: gpu.kernel(W4A16_MODULE, "w4a16_gemm")?,
            w4a16_gemv: gpu.kernel(W4A16_GEMV_MODULE, "w4a16_gemv")?,
            w4a16_gemv_sw: metrale_model_layers::layers::try_kernel(
                gpu,
                W4A16_GEMV_MODULE,
                "w4a16_gemv_sw",
            ),
            w4a16_gemv_sw_moe: metrale_model_layers::layers::try_kernel(
                gpu,
                W4A16_GEMV_MODULE,
                "w4a16_gemv_sw_moe",
            ),
            w4a16_gemv_sw_moe_batchm: [
                metrale_model_layers::layers::try_kernel(
                    gpu,
                    W4A16_GEMV_MODULE,
                    "w4a16_gemv_sw_moe_batchm_m2",
                ),
                metrale_model_layers::layers::try_kernel(
                    gpu,
                    W4A16_GEMV_MODULE,
                    "w4a16_gemv_sw_moe_batchm_m3",
                ),
                metrale_model_layers::layers::try_kernel(
                    gpu,
                    W4A16_GEMV_MODULE,
                    "w4a16_gemv_sw_moe_batchm_m4",
                ),
                metrale_model_layers::layers::try_kernel(
                    gpu,
                    W4A16_GEMV_MODULE,
                    "w4a16_gemv_sw_moe_batchm_m5",
                ),
                metrale_model_layers::layers::try_kernel(
                    gpu,
                    W4A16_GEMV_MODULE,
                    "w4a16_gemv_sw_moe_batchm_m6",
                ),
                metrale_model_layers::layers::try_kernel(
                    gpu,
                    W4A16_GEMV_MODULE,
                    "w4a16_gemv_sw_moe_batchm_m7",
                ),
                metrale_model_layers::layers::try_kernel(
                    gpu,
                    W4A16_GEMV_MODULE,
                    "w4a16_gemv_sw_moe_batchm_m8",
                ),
            ],
            moe_row_union: metrale_model_layers::layers::try_kernel(
                gpu,
                W4A16_GEMV_MODULE,
                "glm5next_moe_row_union",
            ),
            swiglu: gpu.kernel(FFN_MODULE, "glm5next_swiglu_clamp")?,
            swiglu_amax: metrale_model_layers::layers::try_kernel(
                gpu,
                FFN_MODULE,
                "glm5next_swiglu_clamp_amax",
            ),
            router: gpu.kernel(FFN_MODULE, "glm5next_router_topk")?,
            combine: gpu.kernel(FFN_MODULE, "glm5next_moe_combine")?,
            moe_sort_by_expert: metrale_model_layers::layers::try_kernel(
                gpu,
                MOE_MODULE,
                "moe_sort_by_expert",
            ),
            moe_grouped_gemm,
            moe_grouped_tile,
            combine_indexed: metrale_model_layers::layers::try_kernel(
                gpu,
                FFN_MODULE,
                "glm5next_moe_combine_indexed",
            ),
            combine_indexed_local: metrale_model_layers::layers::try_kernel(
                gpu,
                FFN_MODULE,
                "glm5next_moe_combine_indexed_local",
            ),
            moe_permute_tokens: metrale_model_layers::layers::try_kernel(
                gpu,
                MOE_MODULE,
                "moe_permute_tokens",
            ),
            moe_prefill_mma: forward_prefill_gemm::PrefillMmaKernels::resolve(gpu),
        })
    }
}

/// 2026-09-25: Which MLP a layer runs; the same variants as [`crate::glm5next_skeleton::Mlp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm5NextMlpKind {
    Dense,
    RoutedMoe,
}

/// 2026-09-25: GLM MLP geometry for one rank, built by [`Self::from_config`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glm5NextMlpConfig {
    pub hidden: usize,
    /// 2026-09-25: `intermediate_size / tp_world_size`: this rank's share of the dense FFN width.
    pub local_dense_intermediate: usize,
    /// 2026-09-25: `moe_intermediate_size`, one routed expert's width. Not divided by TP: an
    /// expert is owned whole by one EP rank.
    /// 2026-10-05: Under expert-TP, `moe_intermediate_size / 2`: this rank's slice of I.
    pub moe_intermediate: usize,
    /// 2026-09-25: `shared_expert_intermediate_size / tp_world_size`: this rank's share of the
    /// shared expert.
    pub local_shared_intermediate: usize,
    /// 2026-09-25: The full routed-expert count, not this rank's share.
    pub num_experts: usize,
    /// 2026-09-25: `num_experts / ep_world_size`; this rank owns ids
    /// `[ep_rank * local_experts, (ep_rank + 1) * local_experts)`.
    /// 2026-10-05: Under expert-TP, `num_experts` (ids `0..num_experts`, see
    /// [`Self::local_expert_range`]).
    pub local_experts: usize,
    pub ep_rank: usize,
    pub top_k: usize,
    /// 2026-09-25: `routed_scaling_factor`. Applied to the top-k weights, not to the shared
    /// expert.
    pub routed_scale: f32,
    /// 2026-09-25: `norm_topk_prob`: divide the top-k weights by their sum plus `1e-20`.
    pub renormalize: bool,
    /// 2026-09-25: The asymmetric SwiGLU clamp bound; see the module header.
    pub swiglu_limit: f32,
    /// 2026-09-25: True for `Glm5NextRouterMode::VllmBf16`: the router kernel then rounds the
    /// scores, the running sum and each weight to BF16, which can change which experts are
    /// selected. The parser yields `HfFp32` (false) when the checkpoint names no router dtype.
    pub router_bf16_ladder: bool,
    /// 2026-09-25: TP ranks the dense/shared FFN is split over. Above 1, the site output is a
    /// partial sum.
    pub tp_world_size: usize,
    /// 2026-09-25: EP ranks the routed experts are split over. Above 1, the routed sum is a
    /// partial sum.
    pub ep_world_size: usize,
}

impl Glm5NextMlpConfig {
    /// 2026-09-25: Divides the global widths by TP and the expert set by EP (a world size of 0
    /// counts as 1), then runs [`Self::validate`]. Errors when a width or the expert count does
    /// not divide evenly.
    /// 2026-10-05: [`Self::from_config_with`] at the process's `METRALE_GLM_EXPERT_TP` setting.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        Self::from_config_with(config, metrale_config::glm_expert_tp())
    }

    /// 2026-10-05: [`Self::from_config`] with the expert layout given: `expert_tp = false` is the
    /// EP split; `true` is expert-TP (`METRALE_GLM_EXPERT_TP`): this rank owns every routed
    /// expert (`local_experts = num_experts`, range `0..num_experts`) at width
    /// `moe_intermediate_size / 2`, its slice of I. Expert-TP is refused unless
    /// `tp_world_size == ep_world_size == 2` on the same ranks (`tp_rank == ep_rank`) and
    /// `moe_intermediate_size / 2` is a multiple of [`EXPERT_TP_WIDTH_ALIGN`].
    pub fn from_config_with(config: &ModelConfig, expert_tp: bool) -> Result<Self> {
        let tp = config.tp_world_size.max(1);
        let ep = config.ep_world_size.max(1);
        if !config.intermediate_size.is_multiple_of(tp) {
            bail!(
                "GLM MLP: intermediate_size {} does not divide over tp_world_size {tp}",
                config.intermediate_size
            );
        }
        if !config.shared_expert_intermediate_size.is_multiple_of(tp) {
            bail!(
                "GLM MLP: shared_expert_intermediate_size {} does not divide over \
                 tp_world_size {tp}",
                config.shared_expert_intermediate_size
            );
        }
        if !config.num_experts.is_multiple_of(ep) {
            bail!(
                "GLM MLP: num_experts {} does not divide over ep_world_size {ep}; a ragged \
                 expert split would leave some ids owned by nobody",
                config.num_experts
            );
        }
        let mut c = Self {
            hidden: config.hidden_size,
            local_dense_intermediate: config.intermediate_size / tp,
            moe_intermediate: config.moe_intermediate_size,
            local_shared_intermediate: config.shared_expert_intermediate_size / tp,
            num_experts: config.num_experts,
            local_experts: config.num_experts / ep,
            ep_rank: config.ep_rank,
            top_k: config.num_experts_per_tok,
            routed_scale: config.routed_scaling_factor as f32,
            renormalize: config.norm_topk_prob,
            swiglu_limit: config.swiglu_limit,
            router_bf16_ladder: matches!(config.glm5next_router_mode, Glm5NextRouterMode::VllmBf16),
            tp_world_size: tp,
            ep_world_size: ep,
        };
        if expert_tp {
            if tp != 2 || ep != 2 || config.tp_rank != config.ep_rank {
                bail!(
                    "METRALE_GLM_EXPERT_TP=1 needs TP2/EP2 on the same two ranks \
                     (tp_world_size 2, ep_world_size 2, tp_rank == ep_rank); this rank has \
                     tp_world_size {tp} (rank {}), ep_world_size {ep} (rank {}). Unset the lever \
                     or serve at --tp-size 2 --ep-size 2",
                    config.tp_rank,
                    config.ep_rank
                );
            }
            let full = config.moe_intermediate_size;
            if !full.is_multiple_of(2) || !(full / 2).is_multiple_of(EXPERT_TP_WIDTH_ALIGN) {
                bail!(
                    "METRALE_GLM_EXPERT_TP=1: moe_intermediate_size {full} does not halve to a \
                     multiple of {EXPERT_TP_WIDTH_ALIGN} (the CUTLASS W4A4 tile; the W4A16 MMA \
                     needs 64), so not every routed-MoE path can run the slice"
                );
            }
            c.moe_intermediate = full / 2;
            c.local_experts = config.num_experts;
        }
        c.validate()?;
        Ok(c)
    }

    /// 2026-10-05: Whether this rank runs expert-TP (`METRALE_GLM_EXPERT_TP`): split over EP
    /// ranks yet owning every routed expert, each at its slice of I.
    pub fn is_expert_tp(&self) -> bool {
        self.ep_world_size > 1 && self.local_experts == self.num_experts
    }

    /// 2026-09-25: The half-open global expert-id range this rank owns.
    /// 2026-10-05: `0..num_experts` when the rank owns every expert (EP 1, or expert-TP).
    pub fn local_expert_range(&self) -> std::ops::Range<usize> {
        let start = if self.local_experts == self.num_experts {
            0
        } else {
            self.ep_rank * self.local_experts
        };
        start..start + self.local_experts
    }

    /// 2026-09-25: Global expert id to local slot, or `None` when another rank owns it. A
    /// remote id contributes zero; `experts` is indexed by this slot, never by the global id.
    pub fn local_slot(&self, global_id: usize) -> Option<usize> {
        let r = self.local_expert_range();
        r.contains(&global_id).then(|| global_id - r.start)
    }

    /// 2026-09-25: Whether the site output leaves this rank as a partial sum needing
    /// `all_reduce(SUM)`.
    pub fn needs_all_reduce(&self) -> bool {
        self.tp_world_size > 1 || self.ep_world_size > 1
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden == 0 {
            bail!("GLM MLP: hidden_size is 0");
        }
        if self.swiglu_limit <= 0.0 {
            bail!(
                "GLM MLP: swiglu_limit is {}. GLM-5.3 clamps its SwiGLU and the clamp is \
                 asymmetric; a zero limit is not 'no clamp', it is a gate forced to <= 0. \
                 The glm5_next parser reads the real value (10.0) and refuses to default it.",
                self.swiglu_limit
            );
        }
        if self.top_k == 0 || self.top_k > KERNEL_MAX_TOP_K {
            bail!(
                "GLM MLP: num_experts_per_tok {} is outside the {}-slot bound \
                 glm5next_router_topk keeps in registers (`float best_w[16]`)",
                self.top_k,
                KERNEL_MAX_TOP_K
            );
        }
        if self.top_k > self.num_experts {
            bail!(
                "GLM MLP: top_k {} exceeds num_experts {}",
                self.top_k,
                self.num_experts
            );
        }
        if self.moe_intermediate == 0 {
            bail!("GLM MLP: moe_intermediate_size is 0 — a routed layer would compute nothing");
        }
        if self.ep_rank >= self.ep_world_size {
            bail!(
                "GLM MLP: ep_rank {} is outside ep_world_size {}",
                self.ep_rank,
                self.ep_world_size
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

/// 2026-10-06: Logs once that `METRALE_GLM_MOE_COMBINE_LOCAL=1` is engaged (the grouped prefill
/// skips the `expert_out` zeroing and the combine skips slots of experts outside `local`).
pub(crate) fn log_combine_local(local: &std::ops::Range<usize>, rows: usize) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_MOE_COMBINE_LOCAL=1: ENGAGED - grouped MoE prefill skips the expert_out \
             zeroing; combine reads local experts {}..{} only (first call: {rows} rows)",
            local.start,
            local.end
        );
    });
}
