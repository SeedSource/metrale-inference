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
//! - `glm_hc_pre` launches `hc_mix` + `hc_finish` once per `MHC_SLICE_ROWS`-row slice
//!   (`mhc_slices`); at `num_tokens <= MHC_SLICE_ROWS` that is the one unsliced pair, same grids
//!   and pointers. Slicing changes no output bit: see `glm_hc_pre_sliced`.
//! - 2026-10-01: `METRALE_GLM_MHC_TOKMAJOR=1` swaps `hc_mix_bf16` for `hc_mix_bf16_tokmajor`
//!   (one block per token) in calls of at least `MHC_TOKMAJOR_MIN_ROWS` tokens; the mix bytes
//!   are the same (argument in the `.cu`), and `hc_finish` / `hc_post` are unchanged.
//! - 2026-10-06: `glm_hc_post_mix_finish` (`METRALE_GLM_MHC_POST_MIX_FINISH=1`) writes the bytes
//!   of `glm_hc_post_mix` then `glm_hc_pre_part_premixed` of the same site in one launch
//!   (argument in the `.cu`); without its handle the pair runs.

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
    /// 2026-10-01: `glm5next_hc_mix_bf16_tokmajor`: `hc_mix_bf16` with one block per token, grid
    /// `(T, 1)`, same arguments and the same `mix` bytes. Optional (`try_kernel`, 0 when absent);
    /// `glm_hc_pre_sliced` launches it under `METRALE_GLM_MHC_TOKMAJOR=1` (`mhc_tokmajor`).
    pub hc_mix_bf16_tokmajor: KernelHandle,
    /// 2026-09-25: `glm5next_hc_finish`: from the mixes, `post`, `comb` (Sinkhorn) and the
    /// collapsed row `y`.
    pub hc_finish: KernelHandle,
    pub hc_post: KernelHandle,
    pub hc_head: KernelHandle,
    /// 2026-10-06: `glm5next_hc_post_mix_bf16`: `hc_post` fused with the next site's token-major
    /// mix (`glm_hc_post_mix`, `METRALE_GLM_MHC_POST_MIX`). Optional (`try_kernel`, 0 when
    /// absent).
    pub hc_post_mix_bf16: KernelHandle,
    /// 2026-10-06: `glm5next_hc_post_mix_finish_bf16`: `hc_post_mix_bf16` followed by the next
    /// site's `hc_finish` in one launch (`glm_hc_post_mix_finish`,
    /// `METRALE_GLM_MHC_POST_MIX_FINISH`). Optional (`try_kernel`, 0 when absent: the pair runs).
    pub hc_post_mix_finish_bf16: KernelHandle,
}

/// 2026-09-25: The kernel module every GLM mHC kernel resolves from.
pub const GLM5NEXT_MHC_MODULE: &str = "glm5next_mhc";

impl Glm5NextMhcKernels {
    /// 2026-09-25: Resolve the kernels. A missing one is an error, except `hc_mix_bf16`, which is
    /// 0 when absent.
    /// 2026-10-01: `hc_mix_bf16_tokmajor` is optional the same way.
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
            hc_mix_bf16_tokmajor: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_MODULE,
                "glm5next_hc_mix_bf16_tokmajor",
            ),
            hc_finish: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_finish")?,
            hc_post: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_post")?,
            hc_head: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_head")?,
            hc_post_mix_bf16: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_MODULE,
                "glm5next_hc_post_mix_bf16",
            ),
            hc_post_mix_finish_bf16: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_MODULE,
                "glm5next_hc_post_mix_finish_bf16",
            ),
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

/// 2026-09-29: Token rows per `hc_mix` + `hc_finish` launch pair in `glm_hc_pre`.
///
/// `hc_mix` runs grid `(T, mix_hc)`, and the hardware issues blocks in linear block order,
/// `blockIdx.x` (the token) fastest (observed behaviour, not a CUDA guarantee). So each of the
/// `mix_hc` mixing rows is a sweep over all `T` tokens, and every sweep reads each token's whole
/// `hc_mult * hidden` FP32 highway row again (`glm5next_mhc.cu` `glm5next_hc_mix_bf16`:
/// `x = streams + t * hc_dim`, read by the RMS loop and the dot loop). The sweep's working set
/// is `T * hc_mult * hidden * 4` bytes: 16.8 MB at GLM-5.3's 256 x 4 x 4096, which fits GB10's
/// 24 MB L2, and 134 MB at 2048 rows, which does not, so at 2048 rows all 24 sweeps re-read the
/// highway from DRAM (24 x 134 MB = 3.2 GB per call). `hc_finish`, launched after the mix, reads
/// the same highway rows once more, from L2 only while the slice fits. Measured (nsys, pp8192,
/// 2026-09-29): `hc_mix_bf16` 0.495 ms per 256-row call and about 13.5 ms per 2048-row call
/// (derived from the staged profile's totals), 3.4x the linear 3.96 ms; 3.2 GB in 13.5 ms is
/// 239 GB/s, GB10's measured STREAM ceiling (237-240 GB/s). `hc_finish` 0.030 ms and about
/// 0.77 ms, 3.2x. The cause is inferred from the code and those totals, not from an L2 counter.
/// 256 is also the attention-side width every unstaged and staged prefill already launches at.
pub const MHC_SLICE_ROWS: u32 = 256;

/// 2026-09-29: The `(first token, rows)` slices `glm_hc_pre_sliced` launches for `num_tokens`
/// tokens at `slice_rows` (at least 1) rows each: consecutive, tiling `0..num_tokens`, each at
/// most `slice_rows` rows. When `num_tokens <= slice_rows`, 0 included, the only slice is
/// `(0, num_tokens)`, so the launches are exactly the unsliced launcher's.
pub fn mhc_slices(num_tokens: u32, slice_rows: u32) -> impl Iterator<Item = (u32, u32)> {
    let s = slice_rows.max(1);
    let n = num_tokens.div_ceil(s).max(1);
    (0..n).map(move |i| {
        let t = (i as u64 * s as u64) as u32;
        (t, s.min(num_tokens - t))
    })
}

/// 2026-10-01: Fewest tokens a `glm_hc_pre` call needs before `METRALE_GLM_MHC_TOKMAJOR=1` takes
/// effect. `hc_mix_bf16_tokmajor` launches `T` blocks where `hc_mix_bf16` launches `T * mix_hc`,
/// each reading all `mix_hc` BF16 `hc_fn` rows, so a decode (1 row) or verify (a few rows) call
/// would leave most of GB10's SMs idle; those keep `hc_mix_bf16`. The value is PROVISIONAL (not
/// measured): `examples/mhc_tokmajor_bitparity_microtest.rs` times both kernels at 1, 7, 256 and
/// 1000 rows. Every prefill sub-chunk at `METRALE_GLM_PREFILL_ROWS >= 64` (the staged recipe's
/// 256 / 2048) clears it; the default 16-row prefill does not.
pub const MHC_TOKMAJOR_MIN_ROWS: u32 = 64;

/// 2026-10-01: `GLM_HC_MAX_MIX` in `glm5next_mhc.cu`, the accumulator count of
/// `hc_mix_bf16_tokmajor`; a site with more mixing rows keeps `hc_mix_bf16`.
pub const MHC_TOKMAJOR_MAX_MIX: u32 = 24;

/// 2026-10-01: `METRALE_GLM_MHC_TOKMAJOR` as a switch: `1` (surrounding blanks ignored) is on;
/// unset, `0` and anything else are off.
pub(crate) fn parse_mhc_tokmajor(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-01: `METRALE_GLM_MHC_TOKMAJOR=1` runs the mHC mix of every `glm_hc_pre` call of at
/// least `MHC_TOKMAJOR_MIN_ROWS` tokens through `hc_mix_bf16_tokmajor` (one block per token, the
/// highway row read once) instead of `hc_mix_bf16`; byte-identical by construction. Off unless
/// set to `1`; read once.
pub fn mhc_tokmajor() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_MHC_TOKMAJOR").ok();
        let on = parse_mhc_tokmajor(raw.as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_MHC_TOKMAJOR=1 - GLM mHC prefill mix (calls of >= \
                 {MHC_TOKMAJOR_MIN_ROWS} tokens) runs one block per token \
                 (glm5next_hc_mix_bf16_tokmajor; byte-identical by construction)"
            );
        } else if let Some(r) = raw.as_deref().filter(|r| !r.is_empty() && r.trim() != "0") {
            tracing::warn!("METRALE_GLM_MHC_TOKMAJOR={r} is not 0 or 1 - hc_mix_bf16 runs");
        }
        on
    })
}

/// 2026-10-01: Whether a mix launch takes `hc_mix_bf16_tokmajor` for a `requested` one: only for
/// a BF16 `hc_fn`, a resolved handle and at most `MHC_TOKMAJOR_MAX_MIX` mixing rows; otherwise
/// the launcher's usual pick.
pub(crate) fn tokmajor_for(requested: bool, hc_fn_bf16: bool, resolved: bool, mix_hc: u32) -> bool {
    requested && hc_fn_bf16 && resolved && mix_hc <= MHC_TOKMAJOR_MAX_MIX
}

/// 2026-09-25: Collapse each token's `hc_mult` FP32 streams to one BF16 row in `y_out`, and write
/// this site's `post` and `comb`, by launching `hc_mix` (or `hc_mix_bf16`) then `hc_finish`,
/// once per `MHC_SLICE_ROWS`-row slice (`glm_hc_pre_sliced`). `streams` is read, not written, so
/// `glm_hc_post` can take it as its residual. Errors when `num_tokens` exceeds
/// `mhc_mix_max_tokens()`.
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
    glm_hc_pre_sliced(
        gpu,
        kernels,
        streams,
        w,
        y_out,
        post_out,
        comb_out,
        num_tokens,
        hidden_size,
        hc_mult,
        sinkhorn_iters,
        norm_eps,
        hc_eps,
        MHC_SLICE_ROWS,
        stream,
    )
}

/// 2026-09-29: `glm_hc_pre` at `slice_rows` rows per launch pair: for each `mhc_slices` slice
/// `(t0, k)`, `hc_mix` on grid `(k, mix_hc)` then `hc_finish` on grid `(k, 1 + ceil(H / 256))`,
/// every pointer advanced by `t0` rows. `glm_hc_pre` passes `MHC_SLICE_ROWS`; the microtest
/// `glm5next_hc_slice_microtest` sweeps other widths.
///
/// Why every output bit equals the unsliced pair's (grid `(T, ...)`, one launch each):
/// - Both kernels use `blockIdx.x` only as the token index `t`, to address `streams + t * hc *
///   H`, `mix + t * mix_hc`, `y_out + t * H`, `post_out + t * hc` and `comb_out + t * hc * hc`;
///   neither reads `gridDim.x`. `hc_finish` reads `gridDim.y`, which is unchanged. Block
///   `(t - t0, y)` of a slice launched at those pointers advanced by `t0` rows therefore reads
///   the same addresses, runs the same code at the same `blockDim` (256), and so reduces in the
///   same order and writes the same bytes to the same addresses as block `(t, y)` unsliced.
/// - Order on the stream: `hc_finish` of slice `s` reads `mix` rows of slice `s` only, which
///   `hc_mix` of slice `s` wrote before it. Neither kernel writes `streams`, and the later mix
///   launches write only later slices' `mix` rows, so no launch reads a value another launch
///   changes afterwards. `y_out`, `post_out` and `comb_out` are written by `hc_finish` alone,
///   one row per token.
///
/// 2026-10-01: The mix kernel is `hc_mix_bf16_tokmajor` on grid `(k, 1)` when `mhc_tokmajor()`
/// is on and the call has at least `MHC_TOKMAJOR_MIN_ROWS` tokens (`glm_hc_pre_sliced_mix`). It
/// too uses `blockIdx.x` only as `t` and does not read `gridDim`, so the slicing argument above
/// holds for it unchanged.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_pre_sliced(
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
    slice_rows: u32,
    stream: u64,
) -> Result<()> {
    glm_hc_pre_sliced_mix(
        gpu,
        kernels,
        streams,
        w,
        y_out,
        post_out,
        comb_out,
        num_tokens,
        hidden_size,
        hc_mult,
        sinkhorn_iters,
        norm_eps,
        hc_eps,
        slice_rows,
        mix_tokmajor_for_call(num_tokens),
        stream,
    )
}

/// 2026-10-04: The mix kernel request `glm_hc_pre` makes for a call of `call_rows` tokens:
/// `METRALE_GLM_MHC_TOKMAJOR=1` and at least `MHC_TOKMAJOR_MIN_ROWS` tokens (the expression
/// `glm_hc_pre_sliced` has always used, moved here unchanged).
pub fn mix_tokmajor_for_call(call_rows: u32) -> bool {
    mhc_tokmajor() && call_rows >= MHC_TOKMAJOR_MIN_ROWS
}

/// 2026-10-04: `glm_hc_pre` over `num_tokens` tokens that are a part of a `call_rows`-token
/// `glm_hc_pre` call (the sequence-parallel staged prefill runs each rank's rows of a call,
/// `glm5next_layer::seq_parallel`): the same `MHC_SLICE_ROWS` slicing, and the mix kernel the
/// whole call would take (`mix_tokmajor_for_call(call_rows)`), so a narrow part (a tail half
/// below `MHC_TOKMAJOR_MIN_ROWS`) still runs the kernel the call runs. Every row then gets the
/// bytes it gets in the call: see `glm_hc_pre_sliced` (both mix kernels and `hc_finish` use
/// `blockIdx.x` only as the token and never read `gridDim.x`, and the pointers are advanced
/// by the part's first row). At `call_rows == num_tokens` this is `glm_hc_pre`.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_pre_part(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextMhcKernels,
    streams: DevicePtr,
    w: &Glm5NextMhcSiteWeights,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    call_rows: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    glm_hc_pre_sliced_mix(
        gpu,
        kernels,
        streams,
        w,
        y_out,
        post_out,
        comb_out,
        num_tokens,
        hidden_size,
        hc_mult,
        sinkhorn_iters,
        norm_eps,
        hc_eps,
        MHC_SLICE_ROWS,
        mix_tokmajor_for_call(call_rows),
        stream,
    )
}

/// 2026-10-01: `glm_hc_pre_sliced` with the mix kernel chosen by the caller: `tokmajor` asks for
/// `hc_mix_bf16_tokmajor` (grid `(k, 1)` per slice), which runs when `tokmajor_for` admits it;
/// otherwise, and when `tokmajor` is false, the mix is `hc_mix_bf16` (or `hc_mix`) on grid
/// `(k, mix_hc)` as before. A request that falls back is logged once. `hc_finish` is launched
/// the same way either way. `examples/mhc_tokmajor_bitparity_microtest.rs` runs both choices in
/// one process, which the read-once lever cannot.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_pre_sliced_mix(
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
    slice_rows: u32,
    tokmajor: bool,
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
    // 2026-10-01: The token-major mix takes the same arguments as `hc_mix_bf16`; only the grid
    // differs (one block per token).
    let resolved = kernels.hc_mix_bf16_tokmajor.0 != 0;
    let tok = tokmajor_for(tokmajor, w.hc_fn_bf16, resolved, mix_hc);
    if tokmajor && !tok {
        static FELL_BACK: std::sync::Once = std::sync::Once::new();
        FELL_BACK.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_MHC_TOKMAJOR: glm5next_hc_mix_bf16_tokmajor not usable here (hc_fn \
                 BF16: {}, handle resolved: {}, mix_hc {mix_hc} <= {MHC_TOKMAJOR_MAX_MIX}: {}) - \
                 the per-(token, row) mix runs",
                w.hc_fn_bf16,
                resolved,
                mix_hc <= MHC_TOKMAJOR_MAX_MIX
            );
        });
    }
    let (mix_kernel, mix_grid_y) = if tok {
        (kernels.hc_mix_bf16_tokmajor, 1)
    } else {
        (mix_kernel, mix_hc)
    };
    let (h, hc) = (hidden_size as usize, hc_mult as usize);
    for (t0, k) in mhc_slices(num_tokens, slice_rows) {
        let t0 = t0 as usize;
        // 2026-09-29: Row strides in bytes: FP32 highway `[hc, H]`, FP32 `mix` `[mix_hc]`, BF16
        // `y` `[H]`, FP32 `post` `[hc]` and `comb` `[hc, hc]`.
        let s_streams = streams.offset(t0 * hc * h * 4);
        let s_mix = w.mix.offset(t0 * mix_hc as usize * 4);
        KernelLaunch::new(gpu, mix_kernel)
            .grid([k, mix_grid_y, 1])
            .block([256, 1, 1])
            .arg_ptr(s_streams)
            .arg_ptr(w.hc_fn)
            .arg_ptr(s_mix)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(norm_eps)
            .launch(stream)?;
        KernelLaunch::new(gpu, kernels.hc_finish)
            // 2026-09-25: Block `y == 0` computes post, comb and the Sinkhorn; blocks `1..` split
            // the collapse.
            .grid([k, 1 + collapse_blocks(hidden_size), 1])
            .block([256, 1, 1])
            .arg_ptr(s_streams)
            .arg_ptr(s_mix)
            .arg_ptr(w.hc_scale)
            .arg_ptr(w.hc_base)
            .arg_ptr(y_out.offset(t0 * h * 2))
            .arg_ptr(post_out.offset(t0 * hc * 4))
            .arg_ptr(comb_out.offset(t0 * hc * hc * 4))
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_u32(sinkhorn_iters)
            .arg_f32(hc_eps)
            .launch(stream)?;
    }
    Ok(())
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

/// 2026-10-06: `METRALE_GLM_MHC_POST_MIX` is on only for `1`.
pub(crate) fn parse_mhc_post_mix(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-06: `METRALE_GLM_MHC_POST_MIX=1`: the sequence-parallel staged prefill fuses the
/// attention sublayer's `hc_post` with the FFN sublayer's mix (`glm_hc_post_mix`), and the FFN
/// pre then runs `hc_finish` only (`glm_hc_pre_part_premixed`). Byte-identical by construction.
/// Off unless set to `1`; read once.
pub fn mhc_post_mix() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = parse_mhc_post_mix(std::env::var("METRALE_GLM_MHC_POST_MIX").ok().as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_MHC_POST_MIX=1 - GLM mHC: attention hc_post fused with the FFN \
                 site's mix (glm5next_hc_post_mix_bf16; byte-identical by construction)"
            );
        }
        on
    })
}

/// 2026-10-06: Whether `glm5next_hc_post_mix_bf16` can stand in for `hc_post` followed by the
/// token-major mix of `next`: the handle resolved, `next.hc_fn` BF16, and the kernel's fixed
/// GLM-5.3 shape (`hidden` 4096, `hc_mult` 4).
pub fn post_mix_usable(
    kernels: &Glm5NextMhcKernels,
    next: &Glm5NextMhcSiteWeights,
    hidden_size: u32,
    hc_mult: u32,
) -> bool {
    kernels.hc_post_mix_bf16.0 != 0 && next.hc_fn_bf16 && hidden_size == 4096 && hc_mult == 4
}

/// 2026-10-06: `glm_hc_post` of `num_tokens` rows (`out` may alias `residual`) and, from the same
/// pass, the token-major mix of the next site (`next_hc_fn`, BF16) into `mix_out`
/// (`[num_tokens, mix_hc]` FP32): the bytes `glm_hc_post` then that site's `hc_mix_bf16_tokmajor`
/// would write. The caller checks `post_mix_usable`.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_post_mix(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextMhcKernels,
    block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    next_hc_fn: DevicePtr,
    mix_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    if num_tokens == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernels.hc_post_mix_bf16)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_ptr(next_hc_fn)
        .arg_ptr(mix_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .launch(stream)
}

/// 2026-10-06: `METRALE_GLM_MHC_POST_MIX_FINISH` is on only for `1`.
pub(crate) fn parse_mhc_post_mix_finish(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-06: `METRALE_GLM_MHC_POST_MIX_FINISH=1` (with `METRALE_GLM_MHC_POST_MIX=1`): where
/// the sequence-parallel staged prefill fuses a back's `hc_post` with the next site's mix, the
/// same launch also runs that site's `hc_finish` (`glm_hc_post_mix_finish`), and the next front
/// runs the norm only. Byte-identical by construction. Off unless set to `1`; without
/// `METRALE_GLM_MHC_POST_MIX=1` it is ignored (logged). Read once.
pub fn mhc_post_mix_finish() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_MHC_POST_MIX_FINISH").ok();
        let on = parse_mhc_post_mix_finish(raw.as_deref());
        if on && !mhc_post_mix() {
            tracing::warn!(
                "METRALE_GLM_MHC_POST_MIX_FINISH=1 IGNORED: it needs METRALE_GLM_MHC_POST_MIX=1"
            );
            return false;
        }
        if on {
            tracing::warn!(
                "METRALE_GLM_MHC_POST_MIX_FINISH=1 - GLM mHC: the fused post+mix also runs the \
                 next site's hc_finish (glm5next_hc_post_mix_finish_bf16; byte-identical by \
                 construction)"
            );
        }
        on
    })
}

/// 2026-10-06: Whether `glm5next_hc_post_mix_finish_bf16` can stand in for
/// `glm5next_hc_post_mix_bf16` then `hc_finish` of `next`: `post_mix_usable` and its handle
/// resolved.
pub fn post_mix_finish_usable(
    kernels: &Glm5NextMhcKernels,
    next: &Glm5NextMhcSiteWeights,
    hidden_size: u32,
    hc_mult: u32,
) -> bool {
    post_mix_usable(kernels, next, hidden_size, hc_mult) && kernels.hc_post_mix_finish_bf16.0 != 0
}

/// 2026-10-06: `glm_hc_post_mix` (highway into `out`, `out` may alias `residual`; the mix of
/// `next` into `mix_out`) and then, in the same launch, `hc_finish` of `next` over the new
/// highway rows: `y_out` (`[num_tokens, H]` BF16), `post_out` (`[num_tokens, hc]`) and
/// `comb_out` (`[num_tokens, hc, hc]`), the bytes `glm_hc_pre_part_premixed` would write.
/// `y_out` may alias `block_out`, `post_out` `post` and `comb_out` `comb` (the kernel reads a
/// token's inputs before it writes that token's outputs). `next` supplies `hc_fn`, `hc_scale`
/// and `hc_base`. The caller checks `post_mix_finish_usable`.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_post_mix_finish(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextMhcKernels,
    block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    next: &Glm5NextMhcSiteWeights,
    mix_out: DevicePtr,
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
    if num_tokens == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernels.hc_post_mix_finish_bf16)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_ptr(next.hc_fn)
        .arg_ptr(mix_out)
        .arg_ptr(next.hc_scale)
        .arg_ptr(next.hc_base)
        .arg_ptr(y_out)
        .arg_ptr(post_out)
        .arg_ptr(comb_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .launch(stream)
}

/// 2026-10-06: `glm_hc_pre_part` when `w.mix` already holds this call's mix rows (written by
/// `glm_hc_post_mix`): the same `hc_finish` launches over the same `MHC_SLICE_ROWS` slices, no
/// mix launch.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_pre_part_premixed(
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
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    let mix_hc = (2 + hc_mult) * hc_mult;
    let (h, hc) = (hidden_size as usize, hc_mult as usize);
    for (t0, k) in mhc_slices(num_tokens, MHC_SLICE_ROWS) {
        let t0 = t0 as usize;
        KernelLaunch::new(gpu, kernels.hc_finish)
            .grid([k, 1 + collapse_blocks(hidden_size), 1])
            .block([256, 1, 1])
            .arg_ptr(streams.offset(t0 * hc * h * 4))
            .arg_ptr(w.mix.offset(t0 * mix_hc as usize * 4))
            .arg_ptr(w.hc_scale)
            .arg_ptr(w.hc_base)
            .arg_ptr(y_out.offset(t0 * h * 2))
            .arg_ptr(post_out.offset(t0 * hc * 4))
            .arg_ptr(comb_out.offset(t0 * hc * hc * 4))
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_u32(sinkhorn_iters)
            .arg_f32(hc_eps)
            .launch(stream)?;
    }
    Ok(())
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

    /// 2026-09-29: `mhc_slices` tiles `0..T` with consecutive slices of at most the width, and a
    /// call that fits (0 tokens included) is the single unsliced launch `(0, T)`.
    #[test]
    fn slices_tile_the_tokens_and_a_fitting_call_is_one_launch() {
        for t in [
            0u32, 1, 2, 7, 255, 256, 257, 511, 512, 1000, 1280, 2048, 2072, 4096,
        ] {
            for s in [0u32, 1, 3, 64, 128, 256, 512, 4096, u32::MAX] {
                let v: Vec<(u32, u32)> = mhc_slices(t, s).collect();
                if t <= s.max(1) {
                    assert_eq!(v, vec![(0, t)], "T={t} slice={s}: one unsliced launch");
                    continue;
                }
                let mut next = 0u32;
                for (i, &(t0, k)) in v.iter().enumerate() {
                    assert_eq!(t0, next, "T={t} slice={s}: contiguous");
                    assert!(k >= 1 && k <= s.max(1), "T={t} slice={s}: width");
                    if i + 1 < v.len() {
                        assert_eq!(k, s.max(1), "T={t} slice={s}: only the tail is narrower");
                    }
                    next = t0 + k;
                }
                assert_eq!(next, t, "T={t} slice={s}: covers every token");
            }
        }
    }

    /// 2026-09-29: The staged FFN window (2048 rows) runs as eight 256-row pairs; the attention
    /// side (256 rows) and decode (1 row) are single launches, as before.
    #[test]
    fn production_widths_slice_as_expected() {
        let w: Vec<(u32, u32)> = mhc_slices(2048, MHC_SLICE_ROWS).collect();
        assert_eq!(w.len(), 8);
        assert!(w.iter().all(|&(t0, k)| k == 256 && t0 % 256 == 0));
        assert_eq!(
            mhc_slices(256, MHC_SLICE_ROWS).collect::<Vec<_>>(),
            vec![(0, 256)]
        );
        assert_eq!(
            mhc_slices(1, MHC_SLICE_ROWS).collect::<Vec<_>>(),
            vec![(0, 1)]
        );
        assert_eq!(
            mhc_slices(1280, MHC_SLICE_ROWS)
                .map(|s| s.1)
                .collect::<Vec<_>>(),
            vec![256; 5]
        );
        assert_eq!(
            mhc_slices(1000, MHC_SLICE_ROWS).collect::<Vec<_>>(),
            vec![(0, 256), (256, 256), (512, 256), (768, 232)]
        );
    }

    /// 2026-10-06: `METRALE_GLM_MHC_POST_MIX` is on only for `1`.
    #[test]
    fn post_mix_lever_parses_one_as_on_and_everything_else_as_off() {
        assert!(parse_mhc_post_mix(Some("1")));
        assert!(parse_mhc_post_mix(Some(" 1 ")));
        for v in [None, Some(""), Some("0"), Some("2"), Some("on"), Some("01")] {
            assert!(!parse_mhc_post_mix(v), "{v:?}");
        }
    }

    /// 2026-10-06: `METRALE_GLM_MHC_POST_MIX_FINISH` is on only for `1`.
    #[test]
    fn post_mix_finish_lever_parses_one_as_on_and_everything_else_as_off() {
        assert!(parse_mhc_post_mix_finish(Some("1")));
        assert!(parse_mhc_post_mix_finish(Some(" 1 ")));
        for v in [None, Some(""), Some("0"), Some("2"), Some("on"), Some("01")] {
            assert!(!parse_mhc_post_mix_finish(v), "{v:?}");
        }
    }

    /// 2026-10-06: `Glm5NextMhcKernels::resolve` asks for `glm5next_hc_post_mix_finish_bf16`; a
    /// rename in the `.cu` alone fails here instead of silently running the pair at serve.
    #[test]
    fn post_mix_finish_entry_point_matches_the_kernel_file() {
        let cu = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../kernels/gb10/common")
            .join(format!("{GLM5NEXT_MHC_MODULE}.cu"));
        let src = std::fs::read_to_string(&cu).expect("glm5next_mhc.cu readable");
        assert!(
            src.contains("__global__ void __launch_bounds__(GLM_HC_BLOCK) \
                 glm5next_hc_post_mix_finish_bf16("),
            "{cu:?} no longer defines glm5next_hc_post_mix_finish_bf16"
        );
    }

    /// 2026-10-01: `METRALE_GLM_MHC_TOKMAJOR` is on only for `1`.
    #[test]
    fn tokmajor_lever_parses_one_as_on_and_everything_else_as_off() {
        assert!(parse_mhc_tokmajor(Some("1")));
        assert!(parse_mhc_tokmajor(Some(" 1 ")));
        for v in [None, Some(""), Some("0"), Some("2"), Some("on"), Some("true"), Some("01")] {
            assert!(!parse_mhc_tokmajor(v), "{v:?}");
        }
    }

    /// 2026-10-01: The token-major mix runs only when requested, for a BF16 `hc_fn`, with the
    /// handle resolved and at most 24 mixing rows (GLM-5.3's `mix_hc(4)`).
    #[test]
    fn tokmajor_needs_request_bf16_handle_and_the_mix_bound() {
        assert!(tokmajor_for(true, true, true, 24));
        assert!(tokmajor_for(true, true, true, mix_hc(2) as u32));
        assert!(!tokmajor_for(false, true, true, 24), "not requested");
        assert!(!tokmajor_for(true, false, true, 24), "FP32 hc_fn");
        assert!(!tokmajor_for(true, true, false, 24), "handle 0");
        assert!(!tokmajor_for(true, true, true, mix_hc(5) as u32), "past GLM_HC_MAX_MIX");
        assert!(MHC_TOKMAJOR_MIN_ROWS > 1 && MHC_TOKMAJOR_MIN_ROWS <= MHC_SLICE_ROWS);
    }

    /// 2026-10-01: `Glm5NextMhcKernels::resolve` asks for `glm5next_hc_mix_bf16_tokmajor`, and
    /// `MHC_TOKMAJOR_MAX_MIX` mirrors the kernel's `GLM_HC_MAX_MIX`; a rename or a resize in the
    /// `.cu` alone fails here instead of silently falling back at serve.
    #[test]
    fn tokmajor_entry_point_and_mix_bound_match_the_kernel_file() {
        let cu = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../kernels/gb10/common")
            .join(format!("{GLM5NEXT_MHC_MODULE}.cu"));
        let src = std::fs::read_to_string(&cu).expect("glm5next_mhc.cu readable");
        assert!(
            src.contains("extern \"C\" __global__ void glm5next_hc_mix_bf16_tokmajor("),
            "{cu:?} no longer defines glm5next_hc_mix_bf16_tokmajor"
        );
        let define = format!("#define GLM_HC_MAX_MIX {MHC_TOKMAJOR_MAX_MIX}");
        assert!(
            src.lines().any(|l| l.trim() == define),
            "GLM_HC_MAX_MIX in {cu:?} is not MHC_TOKMAJOR_MAX_MIX"
        );
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
