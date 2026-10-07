// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GLM-5.3-Flash KDA attention block: one bound `self_attn` block per KDA layer.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - [`Glm5NextKdaLayer::new`] and [`Glm5NextKdaWorkspace::new`] refuse a config that fails
//!   [`Glm5NextKdaConfig::validate`].
//! - The forward calls (`decode`, `decode_k`, `prefill`) allocate nothing: they write into the
//!   caller's [`Glm5NextKdaWorkspace`] and [`KdaSeqState`].
//!
//! ```text
//! q|k|v_proj -> pack -> conv1d + SiLU -> L2(q,k only) -> kda_gate / sigmoid(b_proj)
//!            -> kda_chunk (prefill) | kda_recurrent (decode)
//!            |  kda_chunk_tc (prefill, opt-in `METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1`)
//!            |  FlashKDA (prefill, opt-in `METRALE_GLM_KDA_PREFILL_FLASHKDA=1`, vendor/flashkda)
//!            -> sigmoid-gated RMSNorm(o_norm, g_b(g_a(h))) -> o_proj
//! ```
//!
//! * The binder accepts only BF16 and F32 tensors ([`binding::KDA_TENSORS`]), and every
//!   projection runs a BF16 kernel, so there is no dequantisation on this path.
//! * The checkpoint stores `q_conv1d`/`k_conv1d`/`v_conv1d` separately; the binder concatenates
//!   them in q, k, v order, the order `front_end` packs the projections in.
//! * `o_norm` gates with a sigmoid, not SiLU, so it runs `kda_o_norm_gated_bf16`
//!   (`kernels/gb10/common/kda_layer_ops.cu`) rather than `gated_rms_norm`.
//! * `dense_mm_bf16` and `cublas_bf16_proj_dense` take no output row stride, so the three q/k/v
//!   projections go to separate `[T, qkv]` buffers and `kda_pack_qkv_bf16` interleaves them.
//! * The conv state keeps `conv_kernel` slots and the conv kernels shift left before convolving;
//!   see [`KdaSeqState`].
//! * Decode fuses conv, SiLU and L2 in one kernel; prefill runs a separate `l2_norm_bf16` over the
//!   q|k channels. V is never normalised.

pub mod binding;
pub mod tp;
pub mod tp_bind;

mod config;
mod decode;
mod kernels;
#[cfg(test)]
mod lever_tests;
mod prefill;
mod prefill_flashkda;
mod prefill_tc;
mod prefill_tc_fuse;
mod snap_fuse;
pub use config::{Glm5NextKdaConfig, Glm5NextKdaWeights};
pub use kernels::Glm5NextKdaKernels;

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

use crate::glm5next_layer::scratch_union::ScratchAlloc;

/// 2026-09-25: Largest shared memory a prefill chunk may need. [`Glm5NextKdaConfig::validate`]
/// refuses a `chunk` whose `smem_prepare` or `smem_scan` exceeds it.
pub const SMEM_CEILING: usize = 49_152;

const BLOCK: u32 = 128;

/// 2026-09-25: V columns one block of the 1R+1W recurrent kernel owns: one warp. At
/// `head_dim = 128` its request is `(3 * 128 + 32 * 129) * 4` = 18,048 B.
const KDA_V_PER_BLOCK: usize = 32;
/// 2026-09-25: Largest shared memory `stateful_row` requests for the 1R+1W kernel; a larger need
/// launches the 2R+2W kernel instead.
const KDA_SMEM_BUDGET: usize = 48 * 1024;

/// 2026-09-25: `METRALE_GLM_KDA_NO_SMEM=1` selects the 2R+2W recurrent kernel. Read once per
/// process; it is checked on every decode row.
fn kda_no_smem() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("METRALE_GLM_KDA_NO_SMEM").as_deref() == Ok("1"))
}

/// 2026-10-01: `METRALE_GLM_KDA_TOKEN_LOOP=1`: a `decode_k` of more than one row that takes no
/// snapshots runs the conv over all rows in one launch, then the recurrence over all rows in one
/// launch, instead of two launches per row. Off by default. Read once per process.
fn kda_token_loop() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("METRALE_GLM_KDA_TOKEN_LOOP").as_deref() == Ok("1"))
}

/// 2026-10-01: Staged elements per thread the generic body of `kda_recurrent_prefill_bf16_pf`
/// holds per prefetched token (`KDA_PF_E_MAX` in kernels/gb10/common/kda_recurrent.cu): the
/// kernel stages `head_dim` elements with `vpb` threads, so it needs `head_dim <= 8 * vpb`.
const KDA_PF_ELEMS_MAX: usize = 8;
/// 2026-10-01: `__launch_bounds__(32)` on `kda_recurrent_prefill_bf16_pf`.
const KDA_PF_THREADS_MAX: usize = 32;

/// 2026-10-01: Whether a `METRALE_GLM_KDA_PREFETCH` value asks for the prefetching kernel: `1`
/// only.
fn prefetch_requested(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-01: `METRALE_GLM_KDA_PREFETCH=1`: the opt-in token loop (`stateful_rows`, under
/// `METRALE_GLM_KDA_TOKEN_LOOP=1`) launches `kda_recurrent_prefill_bf16_pf`, the prefetching
/// twin of `kda_recurrent_prefill_bf16_smem` (byte-identical by construction; checked by
/// `kda_tokenloop_microtest`). Off unless set to `1`. Read once per process.
fn kda_prefetch() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let on = prefetch_requested(std::env::var("METRALE_GLM_KDA_PREFETCH").ok().as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_KDA_PREFETCH=1 - the KDA token loop uses the prefetching recurrent \
                 kernel (byte-identical by construction; see kernels/gb10/common/kda_recurrent.cu)"
            );
        }
        on
    })
}

/// 2026-10-01: Shared memory `kda_recurrent_prefill_bf16_pf` requests at this geometry,
/// `(6 * d + vpb * (d + 1)) * 4` bytes (two staging buffers and the column scratch), or `None`
/// when the geometry breaks its launcher contract or `KDA_SMEM_BUDGET`. At `head_dim = 128`,
/// `vpb = 32`: 19,584 B.
fn kda_pf_smem(d: usize, vpb: usize) -> Option<usize> {
    let smem = (6 * d + vpb * (d + 1)) * 4;
    ((1..=KDA_PF_THREADS_MAX).contains(&vpb)
        && d.is_multiple_of(vpb)
        && d <= KDA_PF_ELEMS_MAX * vpb
        && smem <= KDA_SMEM_BUDGET)
        .then_some(smem)
}

/// 2026-10-01: The one warning when `METRALE_GLM_KDA_PREFETCH=1` cannot be honoured and the
/// token loop keeps `kda_recurrent_prefill_bf16_smem`.
fn kda_prefetch_fallback(why: &str) {
    static W: std::sync::Once = std::sync::Once::new();
    W.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_KDA_PREFETCH=1 ignored ({why}); the KDA token loop keeps \
             kda_recurrent_prefill_bf16_smem"
        )
    });
}

/// 2026-10-01: Whether a `METRALE_GLM_KDA_PREFILL_CHUNKED_TC` value asks for the tensor-core
/// chunked prefill: `1` only.
fn chunked_tc_requested(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-01: `METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1`: a prefill sub-chunk's KDA mixer runs
/// [`Glm5NextKdaLayer::prefill_chunked_tc`] (kernels/gb10/common/kda_chunk_tc.cu) instead of
/// `decode_k`'s per-token recurrence (`glm5next_layer/steps/forward.rs`). The chunked form sums
/// in another order and feeds BF16 tensor-core MMAs, so it is not bit-identical to the per-token
/// walk. Off unless set to `1`. Read once per process.
pub(crate) fn kda_prefill_chunked_tc() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let on = chunked_tc_requested(
            std::env::var("METRALE_GLM_KDA_PREFILL_CHUNKED_TC")
                .ok()
                .as_deref(),
        );
        if on {
            tracing::warn!(
                "METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1 - GLM prefill KDA uses the tensor-core \
                 chunked prefill (kda_chunk_tc). Not bit-identical to the per-token recurrent walk."
            );
        }
        on
    })
}

/// 2026-10-03: Whether a `METRALE_GLM_KDA_PREFILL_FLASHKDA` value asks for the FlashKDA prefill:
/// `1` only.
fn flashkda_requested(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-03: `METRALE_GLM_KDA_PREFILL_FLASHKDA=1`: a prefill sub-chunk of at least
/// `FLASHKDA_MIN_ROWS` rows runs its KDA recurrence through the vendored FlashKDA library
/// ([`Glm5NextKdaLayer::prefill_flashkda`], prefill_flashkda.rs) instead of `decode_k`'s
/// per-token walk (`glm5next_layer/steps/mixer.rs`), ahead of the two other chunked arms. Not
/// bit-identical: FlashKDA sums in chunk order on BF16 tensor-core MMAs and keeps the state in
/// BF16 between its 16-row chunks. Off unless set to `1`. Read once per process.
pub(crate) fn kda_prefill_flashkda() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let on = flashkda_requested(
            std::env::var("METRALE_GLM_KDA_PREFILL_FLASHKDA")
                .ok()
                .as_deref(),
        );
        if on {
            tracing::warn!(
                "METRALE_GLM_KDA_PREFILL_FLASHKDA=1 - GLM prefill KDA uses FlashKDA \
                 (vendor/flashkda, BF16 state between 16-row chunks). Not bit-identical to the \
                 per-token recurrent walk."
            );
        }
        on
    })
}

/// 2026-10-03: The one warning when `METRALE_GLM_KDA_PREFILL_FLASHKDA=1` cannot be honoured and
/// the prefill keeps the other arms.
fn kda_flashkda_fallback(why: &str) {
    static W: std::sync::Once = std::sync::Once::new();
    W.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_KDA_PREFILL_FLASHKDA=1 ignored ({why}); the KDA prefill keeps its \
             other arms"
        )
    });
}

/// 2026-10-01: Chunk width of the tensor-core prefill (`KDA_TC_C` in kda_chunk_tc.cu).
const KDA_TC_C: usize = 16;
/// 2026-10-01: head_dim the tensor-core prefill is compiled for (`KDA_TC_D`).
const KDA_TC_D: usize = 128;
/// 2026-10-01: Rows per `kda_tc_conv_rows` block (`KDA_TC_CONV_ROWS`).
const KDA_TC_CONV_ROWS: usize = 16;
/// 2026-10-01: Warps per `kda_tc_scan` block (`KDA_TC_SCAN_WARPS`); each owns 16 V columns.
const KDA_TC_SCAN_WARPS: usize = 2;
/// 2026-10-01: `kda_tc_scan`'s dynamic shared memory: two 16,128 B stages (`KDA_TC_STAGE`).
const KDA_TC_SCAN_SMEM: usize = 2 * 16_128;
/// 2026-10-01: Lowest `gate_lower_bound` the tensor-core prefill admits. Its prepare kernel
/// factors `exp(gc[i] - gc[j])` through row 7 of a 16-row chunk, so each factor's exponent is
/// below `9 * |bound|`; at -9 that is 81, inside FP32's ~88.7. GLM-5.3-Flash uses -5.
const KDA_TC_GATE_FLOOR: f32 = -9.0;
/// 2026-10-01: Bytes per (16-row chunk, head) record in each of the four workspace buffers the
/// tensor-core prefill borrows (see `prefill_tc.rs`): Q' + W, K'^T, u, M + decay.
const KDA_TC_REC_BYTES: [usize; 4] = [8192, 4096, 8192, 1024];

/// 2026-10-01: Why [`Glm5NextKdaLayer::prefill_chunked_tc`] cannot take `k` rows on this geometry
/// and workspace, or `None` when it can. `t_pad` is the workspace's padded row count; `kernels` is
/// [`Glm5NextKdaKernels::has_chunked_tc`].
fn chunked_tc_refusal(
    cfg: &Glm5NextKdaConfig,
    k: usize,
    t_pad: usize,
    kernels: bool,
) -> Option<&'static str> {
    // 2026-10-01: Each borrowed buffer holds `t_pad * qkv_dim()` FP32 (`Glm5NextKdaWorkspace`); 2026-10-04: the
    // caller passes the padded `chunk_tokens` rows, which is what the chunk buffers were sized to.
    let buf_bytes = t_pad * cfg.qkv_dim() * 4;
    let recs = k.div_ceil(KDA_TC_C) * cfg.heads;
    if !kernels {
        Some("the target lacks a kda_chunk_tc kernel")
    } else if cfg.head_dim != KDA_TC_D {
        Some("head_dim is not 128")
    } else if cfg.conv_kernel > 8 || !cfg.qk_channels().is_multiple_of(256) {
        Some("the conv geometry breaks kda_tc_conv_rows' contract")
    } else if !(KDA_TC_GATE_FLOOR..=0.0).contains(&cfg.gate_lower_bound) {
        Some("gate_lower_bound is outside [-9, 0]")
    } else if k == 0 || KDA_TC_REC_BYTES.iter().any(|b| recs * b > buf_bytes) {
        Some("the workspace cannot hold the chunk records")
    } else {
        None
    }
}

/// 2026-10-01: The one warning when `METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1` cannot be honoured and
/// the prefill takes `decode_k`.
fn kda_chunked_tc_fallback(why: &str) {
    static W: std::sync::Once = std::sync::Once::new();
    W.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1 ignored ({why}); the KDA prefill keeps decode_k"
        )
    });
}

/// 2026-09-25: The per-sequence state a KDA layer carries. The kernels update both buffers in place.
///
/// The conv buffer holds `conv_kernel` slots per channel. The conv kernels shift the window left
/// and write the new input into the last slot before convolving, so slot 0 is shifted out
/// unread. A state kept as `conv_kernel - 1` slots (the golden generators' layout) maps to
/// slots `1..conv_kernel` here, before the shift.
#[derive(Clone, Copy, Debug)]
pub struct KdaSeqState {
    /// 2026-09-25: `[conv_dim, conv_kernel]` FP32.
    pub conv: DevicePtr,
    /// 2026-09-25: `[heads, head_dim, head_dim]` FP32, K-major (`S[k * head_dim + v]`).
    pub recurrent: DevicePtr,
}

/// 2026-09-25: Scratch for the forward path, sized once for `max_tokens` (prefill buffers padded
/// to a whole number of chunks). The loader shares one workspace across the KDA layers.
///
/// The intermediate buffers are `pub` so the numeric tests can compare each stage, not only the
/// layer output.
pub struct Glm5NextKdaWorkspace {
    /// 2026-09-25: `[3, T, qkv]` BF16: the q, k and v projection outputs, before the pack.
    pub qkv_parts: DevicePtr,
    /// 2026-09-25: `[T, conv_dim]` BF16: q|k|v per token, before the conv.
    pub qkv_proj: DevicePtr,
    /// 2026-09-25: `[T, conv_dim]` BF16: after conv + SiLU, with q|k L2-normalised (inside the
    /// conv kernel on decode, by `l2_norm_bf16` in place on prefill).
    pub conv_out: DevicePtr,
    /// 2026-09-25: `[T_pad, qkv]` FP32: q, k and v widened for the chunked prefill. Prefill only.
    pub q_f32: DevicePtr,
    pub k_f32: DevicePtr,
    pub v_f32: DevicePtr,
    /// 2026-09-25: `[T_pad, heads, head_dim]` FP32: the bounded log-decay `kda_gate` writes.
    pub gate: DevicePtr,
    /// 2026-09-25: `[T_pad, heads]` FP32: `sigmoid(b_proj(h))`.
    pub beta: DevicePtr,
    /// 2026-09-25: `[T_pad, heads, head_dim]` FP32: the KDA core output, before `o_norm`.
    pub core: DevicePtr,
    /// 2026-09-25: `[T, qkv]` BF16: `f_b(f_a(h))`, the forget-gate input `kda_gate` reads. It has
    /// the same shape as `out_gate` and must not alias it.
    pub g_raw: DevicePtr,
    /// 2026-09-25: `[T, qkv]` BF16: `g_b(g_a(h))`, the low-rank output gate.
    pub out_gate: DevicePtr,
    /// 2026-09-25: `[T, qkv]` BF16: after the sigmoid-gated RMSNorm.
    pub o_norm_out: DevicePtr,
    /// 2026-09-25: `[T, hidden]` BF16: the block output.
    pub final_out: DevicePtr,
    lowrank: DevicePtr,
    beta_bf16: DevicePtr,
    chunk_gc: DevicePtr,
    chunk_u: DevicePtr,
    chunk_w: DevicePtr,
    max_tokens: usize,
    t_pad: usize,
    /// 2026-10-01: The most tokens the chunked scan (`prefill`) accepts; `q_f32`, `k_f32`,
    /// `v_f32` and `chunk_gc`/`chunk_u`/`chunk_w` are sized for it. `max_tokens` unless the
    /// workspace was built by [`Glm5NextKdaWorkspace::new_split`].
    chunk_tokens: usize,
    /// 2026-10-03: The FlashKDA prefill's own device scratch, allocated only by
    /// [`Glm5NextKdaWorkspace::alloc_flashkda`] (the loader calls it under
    /// `METRALE_GLM_KDA_PREFILL_FLASHKDA=1`); `None` otherwise.
    flashkda: Option<prefill_flashkda::FlashKdaScratch>,
}

impl Glm5NextKdaWorkspace {
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextKdaConfig, max_tokens: usize) -> Result<Self> {
        Self::new_split(gpu, cfg, max_tokens, max_tokens)
    }

    /// 2026-10-01: [`Self::new`] with the chunked scan's six prefill-only buffers (`q_f32`,
    /// `k_f32`, `v_f32`, `chunk_gc`, `chunk_u`, `chunk_w`, 24 bytes per token per `qkv`
    /// channel) sized for `chunk_tokens` (clamped to `1..=max_tokens`) instead of
    /// `max_tokens`. `decode_k` never reads them, so a workspace that only `decode_k` uses at
    /// its full width (`METRALE_GLM_PREFILL_FULLWIDTH_GEMM` without
    /// `METRALE_GLM_KDA_CHUNK_PREFILL`) need not pay for them; `prefill` then refuses more
    /// than `chunk_tokens` tokens. `new_split(.., t, t)` allocates exactly what `new` did.
    pub fn new_split(
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextKdaConfig,
        max_tokens: usize,
        chunk_tokens: usize,
    ) -> Result<Self> {
        Self::new_split_in(cfg, max_tokens, chunk_tokens, &mut |b| gpu.alloc(b))
    }

    /// 2026-10-05: [`Self::new_split`] taking each buffer from `alloc`, in the same order and
    /// sizes (`METRALE_GLM_PREFILL_SCRATCH_UNION` places them in a shared scratch union). It
    /// only allocates: no memset, no upload.
    pub fn new_split_in(
        cfg: &Glm5NextKdaConfig,
        max_tokens: usize,
        chunk_tokens: usize,
        alloc: &mut ScratchAlloc,
    ) -> Result<Self> {
        cfg.validate()?;
        if max_tokens == 0 {
            bail!("workspace needs max_tokens >= 1");
        }
        let (qkv, cd, hd, h) = (cfg.qkv_dim(), cfg.conv_dim(), cfg.head_dim, cfg.heads);
        let t = max_tokens;
        let t_pad = t.div_ceil(cfg.chunk) * cfg.chunk;
        let n = t_pad * qkv;
        let chunk_tokens = chunk_tokens.clamp(1, t);
        // 2026-10-01: Equal to `n` unless `chunk_tokens < max_tokens`.
        let n_chunk = chunk_tokens.div_ceil(cfg.chunk) * cfg.chunk * qkv;
        Ok(Self {
            qkv_parts: alloc(3 * t * qkv * 2)?,
            qkv_proj: alloc(t * cd * 2)?,
            conv_out: alloc(t * cd * 2)?,
            q_f32: alloc(n_chunk * 4)?,
            k_f32: alloc(n_chunk * 4)?,
            v_f32: alloc(n_chunk * 4)?,
            gate: alloc(n * 4)?,
            beta: alloc(t_pad * h * 4)?,
            core: alloc(n * 4)?,
            g_raw: alloc(t * qkv * 2)?,
            out_gate: alloc(t * qkv * 2)?,
            o_norm_out: alloc(t * qkv * 2)?,
            final_out: alloc(t * cfg.hidden * 2)?,
            lowrank: alloc(t * hd * 2)?,
            beta_bf16: alloc(t * h * 2)?,
            chunk_gc: alloc(n_chunk * 4)?,
            chunk_u: alloc(n_chunk * 4)?,
            chunk_w: alloc(n_chunk * 4)?,
            max_tokens: t,
            t_pad,
            chunk_tokens,
            flashkda: None,
        })
    }

    /// 2026-10-01: Device bytes [`Self::new_split`] allocates for these widths, for the
    /// loader's log line.
    pub fn bytes_split(cfg: &Glm5NextKdaConfig, max_tokens: usize, chunk_tokens: usize) -> usize {
        let (qkv, cd, hd, h) = (cfg.qkv_dim(), cfg.conv_dim(), cfg.head_dim, cfg.heads);
        let t = max_tokens.max(1);
        let t_pad = t.div_ceil(cfg.chunk) * cfg.chunk;
        let n_chunk = chunk_tokens.clamp(1, t).div_ceil(cfg.chunk) * cfg.chunk * qkv;
        3 * t * qkv * 2
            + 2 * t * cd * 2
            + 6 * n_chunk * 4
            + 2 * t_pad * qkv * 4
            + t_pad * h * 4
            + 3 * t * qkv * 2
            + t * cfg.hidden * 2
            + t * hd * 2
            + t * h * 2
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    /// 2026-10-01: The most tokens `prefill` (the chunked scan) accepts.
    pub fn chunk_tokens(&self) -> usize {
        self.chunk_tokens
    }
    pub fn t_pad(&self) -> usize {
        self.t_pad
    }
}

/// 2026-09-25: One bound KDA attention block.
pub struct Glm5NextKdaLayer {
    /// 2026-09-25: The checkpoint layer index this block was bound from.
    pub layer_idx: usize,
    pub cfg: Glm5NextKdaConfig,
    pub weights: Glm5NextKdaWeights,
    pub kernels: Glm5NextKdaKernels,
}

impl Glm5NextKdaLayer {
    pub fn new(
        layer_idx: usize,
        cfg: Glm5NextKdaConfig,
        weights: Glm5NextKdaWeights,
        kernels: Glm5NextKdaKernels,
    ) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            layer_idx,
            cfg,
            weights,
            kernels,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &DenseWeight,
        out: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        // 2026-10-03: `METRALE_GLM_DENSE_FP8=1`: a registered weight at <= 16 rows runs on its
        // FP8 copy, and a wider call reads its BF16 dequant (`glm5next_layer::dense_fp8`); off,
        // this returns `weight.weight` without launching.
        let weight = &DenseWeight {
            weight: match crate::glm5next_layer::dense_fp8::route(
                gpu,
                self.kernels.gemv,
                input,
                weight.weight,
                out,
                m,
                n,
                k,
                stream,
            )? {
                crate::glm5next_layer::dense_fp8::Route::Done => return Ok(()),
                crate::glm5next_layer::dense_fp8::Route::Weight(w) => w,
            },
        };
        // 2026-09-25: Only M above `DENSE_GEMV_BATCHM_MAX_M` goes to cuBLASLt. Below it
        // `dense_mm_bf16` runs the M = 1 GEMV or the batched GEMV, whose rows carry the same
        // bits, so the cuBLASLt switch never changes those widths.
        if m > ops::DENSE_GEMV_BATCHM_MAX_M as usize && crate::glm5next_layer::cublas_wide_proj() {
            return ops::cublas_bf16_proj_dense(
                input,
                weight.weight,
                out,
                m as u32,
                n as u32,
                k as u32,
                stream,
            );
        }
        ops::dense_mm_bf16(
            gpu,
            &ops::DenseMmKernels {
                gemm: self.kernels.gemm,
                gemv: self.kernels.gemv,
                batchm: self.kernels.gemv_batchm,
            },
            input,
            weight.weight,
            out,
            m,
            n,
            k,
            stream,
        )
    }

    /// 2026-09-25: Projections, forget gate, beta and output gate, shared by decode and prefill.
    /// All of them read the block's input hidden state, never the post-conv activations.
    fn front_end(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        self.front_end_with(gpu, hidden, t, ws, stream, true)
    }

    /// 2026-10-03: [`Self::front_end`]; with `activate_gates` false it skips the two launches
    /// that activate the forget gate (`kda_gate_bf16` into `ws.gate`) and beta (the sigmoid into
    /// `ws.beta`), leaving the raw `ws.g_raw` and `ws.beta_bf16` the FlashKDA prefill hands to
    /// the library, which activates them itself. Every other launch is the same.
    fn front_end_with(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
        activate_gates: bool,
    ) -> Result<()> {
        self.front_end_opts(gpu, hidden, t, ws, stream, activate_gates, true)
    }

    /// 2026-10-07: [`Self::front_end_with`]; with `pack` false it also skips
    /// `kda_pack_qkv_bf16`, leaving the three projections in `ws.qkv_parts` and `ws.qkv_proj`
    /// unwritten (the fused chunked-TC front, prefill_tc_fuse.rs, reads the parts directly).
    /// Every other launch is the same.
    #[allow(clippy::too_many_arguments)]
    fn front_end_opts(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
        activate_gates: bool,
        pack: bool,
    ) -> Result<()> {
        let c = &self.cfg;
        let (hid, qkv, hd) = (c.hidden, c.qkv_dim(), c.head_dim);
        // 2026-10-06: `METRALE_GLM_DENSE_FP8_W8A8_SHARE_QUANT`: q/k/v/g_a read `hidden`, which
        // nothing below writes (every output is a workspace buffer), so their W8A8 activation
        // quant runs once (`dense_fp8::w8a8_share_input`).
        let _share = crate::glm5next_layer::dense_fp8::w8a8_share_input(hidden, t * hid * 2);

        // 2026-09-25: Three separate `[T, qkv]` projections, then one pack (see the module doc).
        for (i, w) in [
            &self.weights.q_proj,
            &self.weights.k_proj,
            &self.weights.v_proj,
        ]
        .into_iter()
        .enumerate()
        {
            self.gemm(
                gpu,
                hidden,
                w,
                ws.qkv_parts.offset(i * t * qkv * 2),
                t,
                qkv,
                hid,
                stream,
            )?;
        }
        if pack {
            KernelLaunch::new(gpu, self.kernels.pack)
                .grid([div_ceil(qkv as u32, 256), t as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(ws.qkv_parts)
                .arg_ptr(ws.qkv_parts.offset(t * qkv * 2))
                .arg_ptr(ws.qkv_parts.offset(2 * t * qkv * 2))
                .arg_ptr(ws.qkv_proj)
                .arg_u32(t as u32)
                .arg_u32(qkv as u32)
                .launch(stream)?;
        }

        // 2026-09-25: Low-rank forget gate, hidden -> head_dim -> heads * head_dim, then
        // `lower_bound * sigmoid(exp(A_log[h]) * (g[c] + dt_bias[c]))` in `kda_gate_bf16`.
        // `A_log` is per head and `dt_bias` per channel.
        self.gemm(
            gpu,
            hidden,
            &self.weights.f_a,
            ws.lowrank,
            t,
            hd,
            hid,
            stream,
        )?;
        self.gemm(
            gpu,
            ws.lowrank,
            &self.weights.f_b,
            ws.g_raw,
            t,
            qkv,
            hd,
            stream,
        )?;
        if activate_gates {
            KernelLaunch::new(gpu, self.kernels.gate)
                .grid([(t * c.heads) as u32, 1, 1])
                .block([BLOCK, 1, 1])
                .arg_ptr(ws.g_raw)
                .arg_ptr(self.weights.dt_bias)
                .arg_ptr(self.weights.a_log)
                .arg_ptr(ws.gate)
                .arg_u32(t as u32)
                .arg_u32(c.heads as u32)
                .arg_u32(hd as u32)
                .arg_f32(c.gate_lower_bound)
                .launch(stream)?;
        }

        // 2026-09-25: beta = sigmoid(b_proj(hidden)); the KDA kernels read it after the sigmoid.
        self.gemm(
            gpu,
            hidden,
            &self.weights.b_proj,
            ws.beta_bf16,
            t,
            c.heads,
            hid,
            stream,
        )?;
        if activate_gates {
            let n = t * c.heads;
            KernelLaunch::new(gpu, self.kernels.sigmoid)
                .grid([div_ceil(n as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(ws.beta_bf16)
                .arg_ptr(ws.beta)
                .arg_u32(n as u32)
                .launch(stream)?;
        }

        // 2026-09-25: Low-rank output gate; a KDA block has no `Z` tensor (`KDA_TENSORS`).
        self.gemm(
            gpu,
            hidden,
            &self.weights.g_a,
            ws.lowrank,
            t,
            hd,
            hid,
            stream,
        )?;
        self.gemm(
            gpu,
            ws.lowrank,
            &self.weights.g_b,
            ws.out_gate,
            t,
            qkv,
            hd,
            stream,
        )
    }

    /// 2026-09-25: Sigmoid-gated RMSNorm, then `o_proj`.
    fn back_end(
        &self,
        gpu: &dyn GpuBackend,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let fused = self.kernels.o_norm_fp8q;
        self.back_end_with(gpu, t, ws, self.kernels.o_norm, fused, ws.core, stream)
    }

    /// 2026-10-03: [`Self::back_end`] with the norm kernel and its input named: `back_end` passes
    /// `kda_o_norm_gated_bf16` over the FP32 `ws.core`, the FlashKDA prefill
    /// `kda_o_norm_gated_bf16in` over the library's BF16 output. Same launches otherwise.
    ///
    /// 2026-10-07: `fused` is the twin of `o_norm` that also writes the FP8 activation of
    /// `o_proj` (`kda_o_norm_gated_*_fp8q`; `KernelHandle(0)` when absent). With
    /// `METRALE_GLM_NORM_FP8_QUANT_FUSE=1`, a head dim of 128 (one quantizer K-group per head)
    /// and a call the W8A8 arm takes (`dense_fp8::w8a8_fused_input`), the twin replaces the
    /// norm and `o_proj`'s separate quant launch; otherwise this is the unfused pair.
    #[allow(clippy::too_many_arguments)]
    fn back_end_with(
        &self,
        gpu: &dyn GpuBackend,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        o_norm: KernelHandle,
        fused: KernelHandle,
        core: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        if fused.0 != 0
            && c.head_dim == 128
            && crate::glm5next_layer::dense_fp8::w8a8_fused_input(
                gpu,
                self.kernels.gemv,
                ws.o_norm_out,
                self.weights.o_proj.weight,
                ws.final_out,
                (t, c.hidden, c.qkv_dim()),
                stream,
                &mut |a_fp8, a_scale| {
                    // 2026-10-07: Block `row = token * heads + head` is quantizer K-group `head`
                    // of token `token` (head_dim == 128 == the group), so the scale array is
                    // indexed by `row` and the FP8 bytes by `row * 128`, both packed [t, qkv].
                    KernelLaunch::new(gpu, fused)
                        .grid([(t * c.heads) as u32, 1, 1])
                        .block([c.head_dim as u32, 1, 1])
                        .arg_ptr(core)
                        .arg_ptr(ws.out_gate)
                        .arg_ptr(self.weights.o_norm.weight)
                        .arg_ptr(ws.o_norm_out)
                        .arg_ptr(a_fp8)
                        .arg_ptr(a_scale)
                        .arg_u32(c.head_dim as u32)
                        .arg_f32(c.rms_norm_eps)
                        .launch(stream)
                },
            )?
        {
            return Ok(());
        }
        KernelLaunch::new(gpu, o_norm)
            .grid([(t * c.heads) as u32, 1, 1])
            .block([c.head_dim as u32, 1, 1])
            .arg_ptr(core)
            .arg_ptr(ws.out_gate)
            .arg_ptr(self.weights.o_norm.weight)
            .arg_ptr(ws.o_norm_out)
            .arg_u32(c.head_dim as u32)
            .arg_f32(c.rms_norm_eps)
            .launch(stream)?;
        self.gemm(
            gpu,
            ws.o_norm_out,
            &self.weights.o_proj,
            ws.final_out,
            t,
            c.hidden,
            c.qkv_dim(),
            stream,
        )
    }
}
