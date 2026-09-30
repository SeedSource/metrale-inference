// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host launchers for the GLM-5.3 mHC kernels: expand, pre (the `hc_mix` +
//! `hc_finish` pair), post, and the head mean.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Every kernel resolves from `GLM5NEXT_MHC_MODULE`, none from DeepSeek-V4's
//!   `hyper_connection` module.
//! - `glm_hc_pre` refuses more tokens than `mhc_mix_max_tokens()`, the bound the loader sizes
//!   each site's `mix` scratch from.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-25: Every kernel the GLM-5.3 hyper-connection launches. `Copy`: the loader resolves
/// them once and every layer holds a copy.
#[derive(Clone, Copy)]
pub struct Glm5NextMhcKernels {
    /// 2026-09-25: Broadcast the embedding into the `hc_mult` highway streams; first text layer
    /// only. A GLM target cannot use `hyper_connection::hc_expand`: that kernel is in the
    /// DeepSeek-V4 model directory, and a target compiles only `common/` and its own model
    /// directory.
    pub hc_expand: KernelHandle,
    /// 2026-09-25: The fused `glm5next_hc_pre`, one block per token. `glm_hc_pre` does not launch
    /// it; `examples/glm5next_hc_split_gate.rs` uses it as the oracle for `hc_mix` + `hc_finish`.
    pub hc_pre: KernelHandle,
    /// 2026-09-25: `glm5next_hc_mix`: one block per (token, mixing row), grid `(T, mix_hc)`,
    /// reading an FP32 `hc_fn`.
    pub hc_mix: KernelHandle,
    /// 2026-09-25: `glm5next_hc_mix` reading a BF16 `hc_fn`. Widening BF16 to FP32 is exact, so it
    /// computes the FP32 kernel's result on the widened weights. Optional (`try_kernel`, 0 when
    /// absent); `glm_hc_pre` uses it when the site's `hc_fn_bf16` is set and the handle is not 0.
    pub hc_mix_bf16: KernelHandle,
    /// 2026-09-25: `glm5next_hc_finish`: from the mixes, `post`, `comb` (Sinkhorn) and the
    /// collapsed row `y`.
    pub hc_finish: KernelHandle,
    pub hc_post: KernelHandle,
    pub hc_head: KernelHandle,
}

/// 2026-09-25: The kernel module every GLM mHC kernel resolves from.
pub const GLM5NEXT_MHC_MODULE: &str = "glm5next_mhc";

impl Glm5NextMhcKernels {
    /// 2026-09-25: Resolve the kernels. A missing one is an error, except `hc_mix_bf16`, which is
    /// 0 when absent.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            hc_expand: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_expand")?,
            hc_pre: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_pre")?,
            hc_mix: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_mix")?,
            hc_mix_bf16: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_MODULE,
                "glm5next_hc_mix_bf16",
            ),
            hc_finish: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_finish")?,
            hc_post: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_post")?,
            hc_head: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_head")?,
        })
    }
}

/// 2026-09-25: Broadcast each of `num_tokens` BF16 hidden rows into its `hc_mult` FP32 highway
/// streams.
pub fn glm_hc_expand(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    streams: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(streams)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// 2026-09-25: Collapse the highway to one BF16 row per token: the unweighted mean of the
/// `hc_mult` streams. It takes no weights, unlike DeepSeek-V4's `ops::hc_head`.
pub fn hc_head_mean(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    y_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// 2026-09-25: One mHC site's weights; a layer has one set for the attention site and one for
/// the FFN site.
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextMhcSiteWeights {
    /// 2026-09-25: `[mix_hc, hc_mult * hidden]` (see `mix_hc`): BF16 when `hc_fn_bf16` is set,
    /// FP32 otherwise. The loader uploads it as BF16.
    pub hc_fn: DevicePtr,
    /// 2026-09-25: Whether `hc_fn` is BF16; the loader sets it.
    pub hc_fn_bf16: bool,
    /// 2026-09-25: `[3]` FP32: the logit scales of pre, post and comb, in that order.
    pub hc_scale: DevicePtr,
    /// 2026-09-25: `[mix_hc]` FP32.
    pub hc_base: DevicePtr,
    /// 2026-09-25: `[mhc_mix_max_tokens(), mix_hc]` FP32 scratch that `hc_mix` writes and
    /// `hc_finish` reads; the loader allocates one per site.
    pub mix: DevicePtr,
}

/// 2026-09-25: The lower bound of `mhc_mix_max_tokens()`.
pub const MHC_MIX_MAX_TOKENS: usize = 256;

/// 2026-09-25: Token bound of each site's `mix` scratch, `max(prefill_rows(), MHC_MIX_MAX_TOKENS)`,
/// read once. The loader allocates the scratch from it and `glm_hc_pre` refuses more tokens
/// than it, so the allocation and the check read one value.
pub fn mhc_mix_max_tokens() -> usize {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        // 2026-09-29: `prefill_rows_ffn()` is the staged prefill's FFN window, whose FFN-site
        // `hc_pre` covers that many tokens; it equals `prefill_rows()` unless
        // `METRALE_GLM_PREFILL_STAGED=1`.
        let t = crate::glm5next_layer::prefill_rows()
            .max(crate::glm5next_layer::prefill_rows_ffn())
            .max(MHC_MIX_MAX_TOKENS);
        if t != MHC_MIX_MAX_TOKENS {
            tracing::warn!(
                "GLM mHC `mix` scratch widened to {t} tokens (floor {MHC_MIX_MAX_TOKENS}) to \
                 match METRALE_GLM_PREFILL_ROWS"
            );
        }
        t
    })
}

/// 2026-09-25: Collapse each token's `hc_mult` FP32 streams to one BF16 row in `y_out`, and write
/// this site's `post` and `comb`, by launching `hc_mix` (or `hc_mix_bf16`) then `hc_finish`.
/// `streams` is read, not written, so `glm_hc_post` can take it as its residual. Errors when
/// `num_tokens` exceeds `mhc_mix_max_tokens()`.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_pre(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextMhcKernels,
    streams: DevicePtr,
    w: &Glm5NextMhcSiteWeights,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    let mix_hc = (2 + hc_mult) * hc_mult;
    let mix_cap = mhc_mix_max_tokens();
    if num_tokens as usize > mix_cap {
        anyhow::bail!(
            "glm_hc_pre: {num_tokens} tokens exceeds the {mix_cap}-token `mix` scratch \
             (floor {MHC_MIX_MAX_TOKENS}, widened to METRALE_GLM_PREFILL_ROWS). Do not launch \
             past the allocation."
        );
    }
    // 2026-09-25: Both mix kernels take the same arguments; only the element width of `hc_fn`
    // differs, so `hc_fn_bf16` must describe the pointer.
    let mix_kernel = if w.hc_fn_bf16 && kernels.hc_mix_bf16.0 != 0 {
        kernels.hc_mix_bf16
    } else {
        kernels.hc_mix
    };
    KernelLaunch::new(gpu, mix_kernel)
        .grid([num_tokens, mix_hc, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.hc_fn)
        .arg_ptr(w.mix)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.hc_finish)
        // 2026-09-25: Block `y == 0` computes post, comb and the Sinkhorn; blocks `1..` split the
        // collapse.
        .grid([num_tokens, 1 + collapse_blocks(hidden_size), 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.mix)
        .arg_ptr(w.hc_scale)
        .arg_ptr(w.hc_base)
        .arg_ptr(y_out)
        .arg_ptr(post_out)
        .arg_ptr(comb_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(hc_eps)
        .launch(stream)
}

/// 2026-09-25: `out[j] = post[j] * block_out + Σ_i comb[i][j] * residual[i]` for each token.
/// `out` may alias `residual`: the kernel reads every residual stream of an element before it
/// writes that element.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_post(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, collapse_blocks(hidden_size), 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// 2026-09-25: Blocks along grid y for `hc_finish`'s collapse and for `hc_post`: `ceil(H / 256)`,
/// at least 1, where 256 is the block width both launch at. Each output element is computed on
/// its own, so the block count does not change the result.
const fn collapse_blocks(hidden_size: u32) -> u32 {
    if hidden_size < 256 {
        1
    } else {
        hidden_size.div_ceil(256)
    }
}

/// 2026-09-25: Row count of `hc_fn` and `hc_base`, `(2 + hc_mult) * hc_mult`: `hc_mult` rows
/// each for pre and post, then `hc_mult * hc_mult` for comb.
pub fn mix_hc(hc_mult: usize) -> usize {
    (2 + hc_mult) * hc_mult
}

#[cfg(test)]
mod mhc_shape_tests {
    use super::*;

    /// 2026-09-25: `mix_hc` is the pre, post and comb row counts added, for `hc_mult` 1 to 8.
    #[test]
    fn mix_hc_splits_into_pre_post_and_comb() {
        for hc in 1usize..=8 {
            assert_eq!(mix_hc(hc), hc + hc + hc * hc, "pre + post + comb rows");
        }
        assert_eq!(mix_hc(2), 8);
    }

    /// 2026-09-25: `GLM_HC_MAX_MIX` in `glm5next_mhc.cu` is 24, which is `mix_hc(4)`; `hc_mult` 5
    /// would exceed it.
    #[test]
    fn the_kernels_mix_bound_is_hc_mult_four() {
        assert_eq!(mix_hc(4), 24, "GLM_HC_MAX_MIX in glm5next_mhc.cu");
        assert!(mix_hc(5) > 24, "hc_mult 5 would exceed the kernel's bound");
    }

    /// 2026-09-25: Without `METRALE_GLM_PREFILL_ROWS`, `prefill_rows()` is `PREFILL_ROWS` (16), so
    /// the bound is the 256-token floor.
    #[test]
    fn the_mix_bound_never_goes_below_the_shipped_floor() {
        assert_eq!(mhc_mix_max_tokens(), MHC_MIX_MAX_TOKENS);
        assert!(mhc_mix_max_tokens() >= MHC_MIX_MAX_TOKENS);
    }

    /// 2026-09-25: Total `mix` scratch over 90 sites (45 text layers, 2 sites each) at
    /// `hc_mult` 4, the config fixture's values.
    #[test]
    fn the_mix_scratch_costs_megabytes_not_gigabytes() {
        let total = |tokens: usize| tokens * mix_hc(4) * 4 * 90;
        assert_eq!(
            total(256),
            2_211_840,
            "2.2 MB for the whole model at the floor"
        );
        assert_eq!(total(1024), 8_847_360, "8.8 MB at a 1024-row sub-chunk");
    }
}
