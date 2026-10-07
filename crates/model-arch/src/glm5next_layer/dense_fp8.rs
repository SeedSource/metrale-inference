// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `METRALE_GLM_DENSE_FP8=1`: FP8 E4M3 weight-only copies of the GLM-5.3
//! non-expert dense projections, the GEMV dispatch that reads them, and the BF16 dequant the
//! wider GEMMs read instead of the (freed) BF16 originals.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Off (the default), nothing is quantized, freed or allocated, and [`route`] returns
//!   `Route::Weight(b)` for the caller's own `b` without launching, so every projection runs
//!   exactly as before, unless `METRALE_GLM_GEMV_TC=1` (next item).
//! - 2026-10-04: `METRALE_GLM_GEMV_TC=1` (`ops::dense_gemv_tcm`, independent of this lever):
//!   [`route`] runs every 1..=16-row call made with the BF16-out `dense_gemv_bf16` handle on
//!   the row-invariant tensor-core GEMV, over the FP8 copy when the weight is registered
//!   here and over the BF16 weight otherwise, and returns `Route::Done`; a shape the TC
//!   entry declines (k not a multiple of 64 / 128) falls through to the rules below.
//! - On, the loader ([`register_layer`]) quantizes, per text layer (0..num_hidden_layers), the
//!   BF16 weights listed in
//!   [`register_layer`] to `[N, K]` E4M3 + one FP32 scale per output row
//!   (`quantize_bf16_to_fp8`, max|row| / 448), FREES the BF16 original, and rewrites the
//!   layer's weight field to the FP8 copy's device pointer, which is the registry key. A
//!   field that names a registered pointer therefore holds FP8 bytes: every read of it must
//!   go through [`route`] (the three GLM `gemm` wrappers: KDA `Glm5NextKdaLayer::gemm`, DSA
//!   `proj_gemm::gemm`, MLP `forward::launch::gemm`; audited 2026-10-03, no other reader).
//!   The one direct BF16 reader, the MTP drafter's `write_kv_rows` (`batchm_rows` on
//!   `kv_a_proj`, `wk`, `compress_gate`), reads only unregistered weights and checks that with
//!   [`ensure_bf16`] (2026-10-04).
//! - 2026-10-05: The MTP block is registered too ([`register_mtp`], called by
//!   `load_glm5next_mtp_module`): its DSA `q_a_proj`, `q_absorb`, `o_absorb`, its shared-expert
//!   `gate_proj`, `up_proj`, `down_proj` and the head's `eh_proj` (`[hidden, 2 * hidden]`, read
//!   only through [`route`] by `glm5next_mtp_head`). Its `kv_a_proj` stays BF16 (`write_kv_rows`
//!   reads it directly). The MTP weights get their own span from arena offset 0 (the drafter
//!   runs between, never inside, a text layer's forward on the one stream), and
//!   [`register_mtp`] re-runs [`finish_load`], which grows the arena if that span is larger than
//!   every text layer's. `eh_proj` is store-owned: its BF16 original is freed by
//!   `Glm5NextWeightLoader::prune_after_load` ([`take_deferred_free`]), not here.
//!   Keys are live allocations that are never freed, so a later allocation can never alias
//!   one; a pointer strictly inside a registered FP8 allocation, or a registered key passed
//!   with another `[n, k]`, is an error, never a silent BF16 read of FP8 bytes.
//! - [`route`] on a registered weight: 1..=16 rows with the caller's GEMV being the BF16-out
//!   `dense_gemv_bf16` run on the FP8 copy (1 row `dense_gemv_fp8w`, 2..=16
//!   `dense_gemv_fp8w_batchm`, rows bit-identical to each other) and return `Route::Done`;
//!   any other row count (prefill) or GEMV returns `Route::Weight(p)`, `p` a BF16 copy
//!   `bf16_rn(fp8 * row_scale)` (`dequant_fp8_rowscale_bf16`) in the dequant arena, and the
//!   caller runs its unchanged BF16 path (cuBLASLt / tile GEMM) on `p`.
//! - Dequant arena: one allocation made at load ([`finish_load`], before the KV pool is
//!   sized) of the largest per-layer sum of converted BF16 bytes; each weight owns a fixed,
//!   256-byte-aligned offset inside its layer's span, so a layer's mixer and MLP weights
//!   coexist and layers reuse the same bytes. All writes and reads are stream-ordered on the
//!   caller's stream. Eager calls cache: a weight whose arena range still holds its own
//!   dequant (same stream, nothing overlapping written since) is not dequantized again, so a
//!   chunk's 256-row sub-chunks of one layer dequantize each weight once. An eager call on a
//!   different stream than the previous dequant first synchronizes that stream. Under CUDA
//!   graph capture the dequant is always captured (never skipped), and the eager cache is
//!   disabled for the rest of the process, since a replay rewrites the arena behind it.
//!   ASSUMED (not enforced): a captured graph that dequantizes (M > 16 on a converted weight,
//!   e.g. batched verify above 16 rows) is replayed on the stream that also runs the eager
//!   GLM forward, or at least never concurrently with it.
//! - 2026-10-05: `METRALE_GLM_DENSE_NVFP4` (a class list, [`parse_nvfp4_classes`]; `1` =
//!   `kda,shared,mlp`; inert, warned once, without `METRALE_GLM_DENSE_FP8=1`): [`register`] also
//!   quantizes a selected weight's BF16 original to NVFP4 (`quantize_to_nvfp4` with
//!   `quantize_bf16_to_nvfp4_mse`: packed E2M1 `[N, K/2]`, one E4M3 scale per 16 K chosen by
//!   squared error, FP32 tensor scale) before freeing it, and keeps the copy in the entry next to
//!   the FP8 copy (the FP8 key stays the field's pointer). [`route`] runs 1..=[`NV4_MAX_M`] rows
//!   on the BF16-out GEMV of such a weight on the CUDA-core NVFP4 tiers (`w4a16_gemv` at 1 row,
//!   `w4a16_gemv_batch{2..8,16}`, each row bit-identical to `w4a16_gemv` on it), ahead of the
//!   `METRALE_GLM_GEMV_TC` FP8 GEMV; wider calls (prefill) keep the FP8 paths. NOT byte-identical
//!   to FP8. Memory: the NVFP4 copies are extra (`n k / 2 + n k / 16` bytes each, allocated at load
//!   before the KV pool is sized). Class split prior art: the SparkGLM decode profile
//!   (architecture read only; `runs/race/sparkglm-prof-L34/SG-VS-C10.md` in spark-bench).
//! - Weights deliberately left BF16: the MoE router (expert selection), the DSA indexer
//!   (`wk`, `compress_gate`, `weights_proj`, `wq_b`: top-k token selection), the KDA
//!   recurrence gates `f_a`, `f_b` (forget-gate decay) and `b_proj` (beta), `embed_tokens`,
//!   `lm_head` (its FP8 option is `--lm-head-dtype fp8`), the mHC `hc_fn` (not a GEMV), the
//!   MTP block's `kv_a_proj`, indexer, router and head, and any DFlash drafter. Together they are < 0.4 GB/rank of decode reads.
//! - 2026-10-05: `METRALE_GLM_DENSE_FP8_W8A8=1` (inert, warned once, without
//!   `METRALE_GLM_DENSE_FP8=1`): [`route`] on a registered weight with more than
//!   `DENSE_GEMV_FP8W_BATCHM_MAX_M` rows, at least [`w8a8_min_rows`], `k % 128 == 0`,
//!   `n % 64 == 0`, a 4-byte-aligned `c`, the caller's GEMV handle the BF16-out
//!   `dense_gemv_bf16` (so its output is BF16) and an activation that fits the W8A8 scratch
//!   (`m * k` FP8 bytes, `m * k / 128` scales) quantizes `a` (`per_token_group_quant_fp8`, per
//!   token per 128-K group) into the scratch, runs `fp8_gemm_t_rowscale` over the FP8 copy into
//!   `c` and returns `Route::Done`; the dequant arena and its cache are not touched. Any other
//!   call takes the dequant path above (each skip reason logged once). NOT byte-identical to
//!   the dequant path (activation quantization). [`route`] writes `c` as packed `[m, n]` BF16
//!   at row stride `n`, the layout the callers' wide arm (cuBLASLt, ld `n`) already writes:
//!   audited 2026-10-05, every wide call of the three wrappers passes a packed `[m, k]` input
//!   and a packed `[m, n]` output (KDA `glm5next_kda/mod.rs` front/back end, `qkv_parts` at
//!   `i * t * qkv`; DSA `wide.rs`, `xseq/group.rs`, `row_batch.rs` arenas; MLP
//!   `forward/dense.rs` row slices at `a * inter` / `a * hidden`), and the FP32-out DSA calls
//!   pass M = 1 on unregistered weights.
//!   W8A8 scratch: one allocation made in [`finish_load`] (before the KV pool is sized) of
//!   `rows * max_k` FP8 bytes + `rows * max_k / 128` FP32 scales + `max_k / 128` FP32 ones
//!   (filled once), `max_k` the largest `k % 128 == 0` registered weight, `rows` from
//!   `METRALE_GLM_DENSE_FP8_W8A8_ROWS` or, unset, the largest row count the loader sizes a
//!   prefill workspace for ([`dense_fp8_w8a8_rows`]). Written and read on the caller's stream;
//!   an eager call on a different stream than the previous one first synchronizes that stream.
//!   Under CUDA graph capture the quant and GEMM are captured; ASSUMED (not enforced), as for
//!   the arena: a captured graph that writes the scratch is never replayed concurrently with
//!   the eager GLM forward.
//! - 2026-10-06: `METRALE_GLM_DENSE_FP8_W8A8_SHARE_QUANT=1` (inert without W8A8): inside a
//!   [`w8a8_share_input`] scope, a W8A8 call whose `a` lies in the scope's byte range and whose
//!   `(a, m, k, stream)` equal those of the quant the scratch holds (made in the same scope)
//!   skips the quant and runs the GEMM on the held FP8 bytes and scales. Those are exactly what
//!   the quant would write again, because the caller promises (the scope's contract) that
//!   nothing writes the range while the scope lives; any other quant into the scratch replaces
//!   the held entry, so a later call re-quantizes. Byte-identical to the lever off. Eager only:
//!   a capturing stream always quantizes and clears the held entry. Callers (2026-10-06): KDA
//!   `front_end_with` (q/k/v/g_a over `hidden`), DSA `decode_k` / `decode_k_wide` (q_a and
//!   kv_a over `hidden`, kv_a issued right after q_a), MLP `forward_dense_sliced` (gate/up
//!   over each row slice).
//! - 2026-10-07: `METRALE_GLM_NORM_FP8_QUANT_FUSE=1` ([`w8a8_fused_input`], module `fused`;
//!   inert without W8A8): a caller whose W8A8 input is made by an RMSNorm-class kernel that
//!   holds each quantizer K-group in registers (today the KDA gated output norm before
//!   `o_proj`) lets that kernel write the BF16 input AND the FP8 bytes and scales into the
//!   W8A8 scratch; the GEMM then runs without the separate quant launch and its BF16 re-read.
//!   Byte-identical to the unfused pair: the kernel quantizes the BF16-rounded value it
//!   stores, with the quantizer's arithmetic. Any call the W8A8 arm would skip returns
//!   `false` without launching, and the caller runs its unfused norm + [`route`].
//! - 2026-10-06: `METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW=1` ([`super::dense_fp8_gw`]; inert
//!   without W8A8, refused without CUTLASS): [`register`] quantizes a weight that also gets an
//!   NVFP4 decode copy (N, K multiples of 128) to 128x128 block-scaled FP8 instead of per-row
//!   (the entry's `bs`; its `w.row_scale` then holds `[N/128, K/128]` block scales). Such an
//!   entry never takes the per-row decode GEMVs (its 1..=16-row calls go to the NVFP4 GEMV or
//!   the dequant), its W8A8 GEMM runs `dense_fp8_gw::gemm` (CUTLASS Sm120 FP8 blockwise) and
//!   its dequant `dequant_fp8_blockscaled_bf16`. Off, no entry has `bs` and nothing changes.

// 2026-10-07: `METRALE_GLM_NORM_FP8_QUANT_FUSE`: the fused norm + quant entry (child module,
// so it reaches this file's private W8A8 state).
mod fused;
pub use fused::{norm_fp8_quant_fuse, w8a8_fused_input};

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight, QuantizedWeight};

/// 2026-10-03: `METRALE_GLM_DENSE_FP8=1` opts in; read once.
pub fn dense_fp8() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_DENSE_FP8").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8=1 - GLM-5.3 non-expert dense projections keep only FP8 \
                 E4M3 copies (per-row scale; BF16 originals freed): <= 16-row GEMVs read FP8, \
                 wider GEMMs a BF16 dequant of it; NOT byte-identical to BF16"
            );
        }
        on
    })
}

/// 2026-10-05: `METRALE_GLM_DENSE_FP8_W8A8=1` opts in (module doc); read once. False, with a
/// warning, when set without `METRALE_GLM_DENSE_FP8=1`.
pub fn dense_fp8_w8a8() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        if std::env::var("METRALE_GLM_DENSE_FP8_W8A8").as_deref() != Ok("1") {
            return false;
        }
        if !dense_fp8() {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8_W8A8=1 is ignored without METRALE_GLM_DENSE_FP8=1 (it only \
                 changes the prefill GEMMs of the FP8 weight copies)"
            );
            return false;
        }
        tracing::warn!(
            "METRALE_GLM_DENSE_FP8_W8A8=1 - GLM-5.3 prefill GEMMs of >= {} rows on \
             FP8 dense weight copies quantize the activations to FP8 E4M3 (per token, per 128-K \
             group) and run the W8A8 fp8_gemm_rowscale_pipe_128x64 instead of dequant + cuBLASLt \
             BF16; NOT byte-identical",
            w8a8_min_rows()
        );
        true
    })
}

/// 2026-10-05: The tensor classes `METRALE_GLM_DENSE_NVFP4` gives an NVFP4 decode copy
/// (module doc). Prior art for the class split: the SparkGLM decode profile (race-decode
/// 2026-10-05, `runs/race/sparkglm-prof-L34/SG-VS-C10.md`, read for architecture only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Nvfp4Classes {
    /// KDA `q_proj`, `k_proj`, `v_proj`, `o_proj` (the output gate `g_a`/`g_b` stays FP8).
    pub kda: bool,
    /// DSA `q_a_proj`, `q_absorb`, `kv_a_proj`, `o_absorb`.
    pub dsa: bool,
    /// 2026-10-05: DSA `o_absorb` only (the attention output projection; the other three
    /// feed attention scores or the cached latent). Implied by `dsa`.
    pub dsa_o: bool,
    /// The MoE layers' shared-expert `gate_proj`, `up_proj`, `down_proj`.
    pub shared: bool,
    /// The dense layers' (0..=2) `gate_proj`, `up_proj`, `down_proj`.
    pub mlp: bool,
    /// The MTP block's converted weights (draft-only numerics).
    pub mtp: bool,
}

impl Nvfp4Classes {
    pub fn any(&self) -> bool {
        self.kda || self.dsa || self.dsa_o || self.shared || self.mlp || self.mtp
    }

    fn has(&self, c: Class) -> bool {
        match c {
            Class::Kda => self.kda,
            Class::Dsa => self.dsa,
            Class::DsaO => self.dsa || self.dsa_o,
            Class::Shared => self.shared,
            Class::Mlp => self.mlp,
            Class::Mtp => self.mtp,
            Class::Keep => false,
        }
    }
}

/// 2026-10-05: The [`Nvfp4Classes`] class of a weight [`register_layer`] / [`register_mtp`]
/// converts; `Keep` never gets an NVFP4 copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Kda,
    Dsa,
    DsaO,
    Shared,
    Mlp,
    Mtp,
    Keep,
}

/// 2026-10-05: Parse a `METRALE_GLM_DENSE_NVFP4` value: unset, empty, `0` or `off` = none;
/// `1` or `on` = `kda,shared,mlp`; else a comma list of `kda`, `dsa`, `dsa_o`, `shared`, `mlp`,
/// `mtp`.
/// An unknown name is an error.
pub fn parse_nvfp4_classes(raw: Option<&str>) -> Result<Nvfp4Classes> {
    let mut c = Nvfp4Classes::default();
    let Some(raw) = raw.map(str::trim) else {
        return Ok(c);
    };
    match raw {
        "" | "0" | "off" => return Ok(c),
        "1" | "on" => {
            c.kda = true;
            c.shared = true;
            c.mlp = true;
            return Ok(c);
        }
        _ => {}
    }
    for name in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match name {
            "kda" => c.kda = true,
            "dsa" => c.dsa = true,
            "dsa_o" => c.dsa_o = true,
            "shared" => c.shared = true,
            "mlp" => c.mlp = true,
            "mtp" => c.mtp = true,
            other => bail!(
                "METRALE_GLM_DENSE_NVFP4: unknown class {other:?} (kda, dsa, dsa_o, shared, mlp, mtp; \
                 or 1 = kda,shared,mlp)"
            ),
        }
    }
    Ok(c)
}

/// 2026-10-05: `METRALE_GLM_DENSE_NVFP4` (module doc); read once. None, with a warning, when
/// set without `METRALE_GLM_DENSE_FP8=1`; none, with an error log, on a value
/// [`parse_nvfp4_classes`] rejects ([`register_layer`] fails the load on it).
pub fn dense_nvfp4() -> Nvfp4Classes {
    static E: OnceLock<Nvfp4Classes> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DENSE_NVFP4").ok();
        let c = match parse_nvfp4_classes(raw.as_deref()) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("{e:#}");
                return Nvfp4Classes::default();
            }
        };
        if !c.any() {
            return c;
        }
        if !dense_fp8() {
            tracing::warn!(
                "METRALE_GLM_DENSE_NVFP4 is ignored without METRALE_GLM_DENSE_FP8=1 (it adds \
                 NVFP4 decode copies to the FP8 dense weights)"
            );
            return Nvfp4Classes::default();
        }
        tracing::warn!(
            "METRALE_GLM_DENSE_NVFP4={} - GLM-5.3 dense classes {c:?} also get an NVFP4 copy \
             (E2M1 + one E4M3 scale per 16 K, MSE-chosen, FP32 tensor scale; quantized from \
             BF16 at load): 1..=16-row GEMVs read it (w4a16_gemv / w4a16_gemv_batch*), prefill \
             keeps the FP8 copy; NOT byte-identical to FP8",
            raw.as_deref().unwrap_or("")
        );
        c
    })
}

/// 2026-10-05: Most rows an NVFP4 decode GEMV takes (`w4a16_gemv_batch16`); wider calls keep
/// the FP8 path.
pub const NV4_MAX_M: usize = 16;

/// 2026-10-05: Fewest rows a W8A8 GEMM takes by default; narrower calls keep the dequant path.
/// PROVISIONAL: not swept; the 128-row tile wastes most of its MMAs below it. 2026-10-06:
/// `METRALE_GLM_DENSE_FP8_W8A8_MIN_ROWS` moves it ([`w8a8_min_rows`]).
pub const W8A8_MIN_ROWS: usize = super::dense_fp8_min_rows::DEFAULT_MIN_ROWS;
pub use super::dense_fp8_min_rows::w8a8_min_rows;

/// 2026-10-05: Rows the W8A8 activation scratch holds at the widest registered `k`:
/// `METRALE_GLM_DENSE_FP8_W8A8_ROWS` (an integer >= 1), else [`w8a8_workspace_rows`]. Read once.
pub fn dense_fp8_w8a8_rows() -> usize {
    static R: OnceLock<usize> = OnceLock::new();
    *R.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DENSE_FP8_W8A8_ROWS").ok();
        let parsed = raw
            .as_deref()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|r| *r >= 1);
        if let (Some(r), None) = (raw.as_deref(), parsed)
            && !r.trim().is_empty()
        {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8_W8A8_ROWS={r} is not an integer >= 1 - using the prefill \
                 workspace rows ({})",
                w8a8_workspace_rows()
            );
        }
        parsed.unwrap_or_else(w8a8_workspace_rows)
    })
}

/// 2026-10-05: The largest row count the GLM-5.3 loader sizes a prefill workspace for, the
/// same terms as `glm5_next_load/loader.rs` (`verify_k`, `kda_rows`, `mlp_rows`, the DSA wide
/// arena): `DENSE_GEMV_BATCHM_MAX_M`, `PREFILL_ROWS`, `prefill_rows()`, `prefill_rows_ffn()`,
/// `fullwidth_rows()` and `batched_verify_rows()`. A GEMM wider than its workspace cannot
/// happen; a wider one anyway (a sizing this list misses) falls back to the dequant path.
pub fn w8a8_workspace_rows() -> usize {
    use crate::glm5next_layer as gl;
    (ops::DENSE_GEMV_BATCHM_MAX_M as usize)
        .max(gl::PREFILL_ROWS)
        .max(gl::prefill_rows())
        .max(gl::prefill_rows_ffn())
        .max(gl::fullwidth_rows().unwrap_or(0))
        .max(gl::levers::batched_verify_rows())
}

/// 2026-10-03: What [`route`] did: `Done` (an FP8 GEMV wrote the output, or there were no
/// rows), or `Weight(p)`: run the caller's BF16 path with `p` as the weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Done,
    Weight(DevicePtr),
}

#[derive(Clone, Copy)]
struct Entry {
    w: Fp8DenseWeight,
    n: usize,
    k: usize,
    /// Byte offset of this weight's BF16 dequant in the arena.
    off: usize,
    /// 2026-10-05: The NVFP4 decode copy (`METRALE_GLM_DENSE_NVFP4`), if this weight has one.
    nv: Option<QuantizedWeight>,
    /// 2026-10-06: `w` is 128x128 block-scaled (`METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW`):
    /// `w.row_scale` holds `[N/128, K/128]` block scales, not per-row ones.
    bs: bool,
}

/// 2026-10-05: The `METRALE_GLM_DENSE_NVFP4` kernels: the load-time quantizer and the decode
/// GEMV tiers (`w4a16_gemv` at 1 row, `w4a16_gemv_batch{2,3}` with M fixed, `batch{4..=8,16}`
/// taking M). Every tier's row bits equal `w4a16_gemv` on that row (`w4a16_gemv.cu`).
#[derive(Clone, Copy)]
struct Nv4Kernels {
    absmax: KernelHandle,
    quant: KernelHandle,
    gemv1: KernelHandle,
    b2: KernelHandle,
    b3: KernelHandle,
    /// 2026-10-07: `w4a16_gemv_batch3_staged` (`METRALE_GLM_NV4_B3_STAGED`); `None` when the
    /// loaded image predates it.
    b3s: Option<KernelHandle>,
    /// `w4a16_gemv_batch4` ..= `batch8` at 0..=4, `batch16` at 5.
    tiers: [KernelHandle; 6],
}

#[derive(Clone, Copy)]
struct Kernels {
    quant: KernelHandle,
    gemv1: KernelHandle,
    batchm: KernelHandle,
    dequant: KernelHandle,
    bf16_gemv: KernelHandle,
}

/// 2026-10-03: Eager dequant cache: `(offset, bytes, key)` of the arena ranges whose last
/// write was that key's dequant on `last_stream`.
struct Cache {
    resident: Vec<(usize, usize, u64)>,
    poisoned: bool,
    last_stream: Option<u64>,
}

static KERNELS: OnceLock<Option<Kernels>> = OnceLock::new();
static MAP: OnceLock<RwLock<BTreeMap<u64, Entry>>> = OnceLock::new();
static ARENA_NEED: AtomicUsize = AtomicUsize::new(0);
static ARENA: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);
static CACHE: Mutex<Cache> = Mutex::new(Cache {
    resident: Vec::new(),
    poisoned: false,
    last_stream: None,
});
/// 2026-10-05: BF16 originals [`register`] quantized but left to their owner to free.
static DEFERRED_FREE: Mutex<Vec<u64>> = Mutex::new(Vec::new());
static HITS: AtomicU64 = AtomicU64::new(0);
static NV4_KERNELS: OnceLock<Option<Nv4Kernels>> = OnceLock::new();
static NV4_HITS: AtomicU64 = AtomicU64::new(0);
static NV4_BYTES: AtomicUsize = AtomicUsize::new(0);
static NV4_WEIGHTS: AtomicUsize = AtomicUsize::new(0);
static NV4_FIRST_HIT: AtomicBool = AtomicBool::new(false);
static DEQUANTS: AtomicU64 = AtomicU64::new(0);
static FIRST_HIT: AtomicBool = AtomicBool::new(false);
static FIRST_DEQUANT: AtomicBool = AtomicBool::new(false);
static HANDLE_MISMATCH: AtomicBool = AtomicBool::new(false);
/// 2026-10-06: Block-scaled weights (`METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW`) and their scale bytes.
static GW_WEIGHTS: AtomicUsize = AtomicUsize::new(0);
static GW_SCALE_BYTES: AtomicUsize = AtomicUsize::new(0);

const ARENA_ALIGN: usize = 256;

/// 2026-10-05: The W8A8 kernels (activation quantizer, per-row-scale GEMM), resolved once.
#[derive(Clone, Copy)]
struct W8a8Kernels {
    quant: ops::Fp8ActQuant,
    gemm: KernelHandle,
}

/// 2026-10-05: The W8A8 activation scratch (module doc): one allocation at `base`.
#[derive(Clone, Copy)]
struct W8a8Scratch {
    base: DevicePtr,
    bytes: usize,
    a_fp8: DevicePtr,
    fp8_cap: usize,
    a_scale: DevicePtr,
    scale_cap: usize,
    /// `ones_len` FP32 1.0 values.
    ones: DevicePtr,
    ones_len: usize,
}

struct W8a8State {
    scratch: Option<W8a8Scratch>,
    /// The stream of the last eager W8A8 launch pair.
    last_stream: Option<u64>,
    /// 2026-10-06: The quant the scratch holds, when it was made inside a share scope
    /// (`METRALE_GLM_DENSE_FP8_W8A8_SHARE_QUANT`); `None` after any other quant.
    held: Option<W8a8Held>,
}

/// 2026-10-06: What the W8A8 scratch holds: the input `a` (device address), `m`, `k`, the
/// stream and the share scope's generation it was quantized under.
#[derive(Clone, Copy, PartialEq, Eq)]
struct W8a8Held {
    a: u64,
    m: usize,
    k: usize,
    stream: u64,
    generation: u64,
}

/// 2026-10-06: A live share scope: the byte range `[lo, hi)` its caller keeps unwritten, and its
/// generation (unique per scope).
#[derive(Clone, Copy)]
struct ShareScope {
    lo: u64,
    hi: u64,
    generation: u64,
}

static SHARE_SCOPE: Mutex<Option<ShareScope>> = Mutex::new(None);
static SHARE_GENERATION: AtomicU64 = AtomicU64::new(0);
static W8A8_REUSED: AtomicU64 = AtomicU64::new(0);
static W8A8_REUSE_FIRST: AtomicBool = AtomicBool::new(false);

/// 2026-10-06: `METRALE_GLM_DENSE_FP8_W8A8_SHARE_QUANT=1`: W8A8 GEMMs inside a
/// [`w8a8_share_input`] scope that read the input the scratch already holds reuse its FP8
/// bytes and scales instead of quantizing again (module doc). Off unless `1`; inert without
/// `METRALE_GLM_DENSE_FP8_W8A8=1`. Read once.
pub fn w8a8_share_quant() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_DENSE_FP8_W8A8_SHARE_QUANT").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8_W8A8_SHARE_QUANT=1 - W8A8 GEMMs that share an input inside \
                 a share scope quantize it once (byte-identical by construction)"
            );
        }
        on && dense_fp8_w8a8()
    })
}

/// 2026-10-06: A share scope over `[a, a + bytes)` (module doc); it ends when dropped. The
/// caller promises that nothing writes that range while the guard lives. Opening a scope ends
/// any open one. Inert (no state) when [`w8a8_share_quant`] is off.
#[must_use = "the scope ends when the guard is dropped"]
pub struct W8a8ShareInput {
    generation: Option<u64>,
}

/// 2026-10-06: Open a share scope over the `bytes` bytes at `a` (see [`W8a8ShareInput`]).
pub fn w8a8_share_input(a: DevicePtr, bytes: usize) -> W8a8ShareInput {
    if !w8a8_share_quant() {
        return W8a8ShareInput { generation: None };
    }
    let generation = SHARE_GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
    *SHARE_SCOPE.lock().unwrap() = Some(ShareScope {
        lo: a.0,
        hi: a.0 + bytes as u64,
        generation,
    });
    W8a8ShareInput {
        generation: Some(generation),
    }
}

impl Drop for W8a8ShareInput {
    fn drop(&mut self) {
        if let Some(g) = self.generation {
            let mut s = SHARE_SCOPE.lock().unwrap();
            if s.is_some_and(|x| x.generation == g) {
                *s = None;
            }
        }
    }
}

/// 2026-10-06: The open scope's generation when `a .. a + bytes` lies inside its range.
fn share_generation_for(a: DevicePtr, bytes: usize) -> Option<u64> {
    let s = (*SHARE_SCOPE.lock().unwrap())?;
    (a.0 >= s.lo && a.0 + bytes as u64 <= s.hi).then_some(s.generation)
}

/// 2026-10-06: W8A8 GEMMs that reused a held quant, for the load summary / tests.
pub fn w8a8_reused_quants() -> u64 {
    W8A8_REUSED.load(Ordering::Relaxed)
}

/// 2026-10-05: Why a wide call on a registered weight skipped W8A8; each is logged once.
#[derive(Clone, Copy)]
enum W8a8Skip {
    FewRows = 0,
    Shape = 1,
    NoKernels = 2,
    NoScratch = 3,
    TooWide = 4,
    NotBf16Out = 5,
}

static W8A8_KERNELS: OnceLock<Option<W8a8Kernels>> = OnceLock::new();
static W8A8: Mutex<W8a8State> = Mutex::new(W8a8State {
    scratch: None,
    last_stream: None,
    held: None,
});
static W8A8_GEMMS: AtomicU64 = AtomicU64::new(0);
static W8A8_FIRST: AtomicBool = AtomicBool::new(false);
static W8A8_SKIP_LOGGED: [AtomicBool; 6] = [
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
];

fn map() -> &'static RwLock<BTreeMap<u64, Entry>> {
    MAP.get_or_init(|| RwLock::new(BTreeMap::new()))
}

/// 2026-10-03: The kernels the lever needs, resolved once; `None` (logged) when any is
/// missing from the target, which leaves the lever inert (nothing quantized or freed).
fn kernels(gpu: &dyn GpuBackend) -> Option<Kernels> {
    *KERNELS.get_or_init(|| {
        let r = (|| -> Result<Kernels> {
            Ok(Kernels {
                quant: gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?,
                gemv1: gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?,
                batchm: gpu.kernel("dense_gemv_fp8w_batchm", "dense_gemv_fp8w_batchm")?,
                dequant: gpu.kernel("dequant_fp8_rowscale_bf16", "dequant_fp8_rowscale_bf16")?,
                bf16_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            })
        })();
        match r {
            Ok(k) => Some(k),
            Err(e) => {
                tracing::warn!("METRALE_GLM_DENSE_FP8: kernels unavailable ({e:#}); staying BF16");
                None
            }
        }
    })
}

/// 2026-10-07: `METRALE_GLM_NV4_B3_STAGED=1` routes the 3-row NVFP4 GEMV through
/// `w4a16_gemv_batch3_staged` (activations staged in shared memory, next weight word
/// prefetched; same arithmetic order, so the same bits as `w4a16_gemv_batch3`). Read once.
pub fn nv4_b3_staged() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_NV4_B3_STAGED").as_deref() == Ok("1"))
}

/// 2026-10-05: The NVFP4 kernels, resolved once; `None` (logged) when any is missing, which
/// leaves the weights FP8-only.
fn nv4_kernels(gpu: &dyn GpuBackend) -> Option<Nv4Kernels> {
    *NV4_KERNELS.get_or_init(|| {
        let r = (|| -> Result<Nv4Kernels> {
            let t = |m: u32| gpu.kernel("w4a16_gemv", &format!("w4a16_gemv_batch{m}"));
            Ok(Nv4Kernels {
                absmax: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
                quant: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_mse")?,
                gemv1: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
                b2: t(2)?,
                b3: t(3)?,
                b3s: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3_staged").ok(),
                tiers: [t(4)?, t(5)?, t(6)?, t(7)?, t(8)?, t(16)?],
            })
        })();
        match r {
            Ok(k) => Some(k),
            Err(e) => {
                tracing::warn!("METRALE_GLM_DENSE_NVFP4: kernels unavailable ({e:#}); staying FP8");
                None
            }
        }
    })
}

/// 2026-10-03: Quantize one BF16 `[n, k]` weight, free the BF16 original and register the
/// FP8 copy with arena offset `off`. Returns the FP8 copy (its `weight` is the new key),
/// or `None` with nothing changed (lever off, kernels missing, empty, or `k % 16 != 0`).
fn register(
    gpu: &dyn GpuBackend,
    bf16: DevicePtr,
    n: usize,
    k: usize,
    off: usize,
    free_original: bool,
    nv4: bool,
) -> Result<Option<Fp8DenseWeight>> {
    if !dense_fp8() || n == 0 || k == 0 || !k.is_multiple_of(16) || bf16.is_null() {
        return Ok(None);
    }
    let Some(kk) = kernels(gpu) else {
        return Ok(None);
    };
    if map().read().unwrap().contains_key(&bf16.0) {
        bail!("METRALE_GLM_DENSE_FP8: {bf16} is already an FP8 copy; registered twice");
    }
    let s = gpu.default_stream();
    // 2026-10-06: `METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW`: a weight that gets an NVFP4 decode
    // copy below is quantized block-scaled (module doc); every other weight as before.
    let bs = super::dense_fp8_gw::cutlass_gw()
        && super::dense_fp8_gw::eligible_for(true, nv4 && nv4_kernels(gpu).is_some(), n, k);
    let w = if bs {
        let w = super::dense_fp8_gw::quantize_block_scaled(gpu, bf16, n, k, s)?;
        GW_WEIGHTS.fetch_add(1, Ordering::Relaxed);
        GW_SCALE_BYTES.fetch_add(n.div_ceil(128) * k.div_ceil(128) * 4, Ordering::Relaxed);
        w
    } else {
        metrale_model_layers::weight_map::quantize_to_fp8(
            &DenseWeight { weight: bf16 },
            n,
            k,
            gpu,
            kk.quant,
            s,
        )?
    };
    // 2026-10-05: The NVFP4 decode copy, from the BF16 original (not from the FP8 copy, which
    // would quantize twice); `quantize_to_nvfp4` synchronizes `s` too.
    let nv = if nv4
        && let Some(q) = nv4_kernels(gpu)
    {
        let w4 = metrale_model_layers::weight_map::quantize_to_nvfp4(
            &DenseWeight { weight: bf16 },
            n,
            k,
            gpu,
            q.absmax,
            q.quant,
            s,
        )?;
        NV4_BYTES.fetch_add(n * k / 2 + n * k / 16, Ordering::Relaxed);
        NV4_WEIGHTS.fetch_add(1, Ordering::Relaxed);
        Some(w4)
    } else {
        None
    };
    // `quantize_to_fp8` synchronized `s`; nothing reads the original after this.
    if free_original {
        gpu.free(bf16)?;
    } else {
        // 2026-10-05: store-owned original (MTP `eh_proj`): the store's `prune_after_load` frees it.
        DEFERRED_FREE.lock().unwrap().push(bf16.0);
    }
    map()
        .write()
        .unwrap()
        .insert(w.weight.0, Entry { w, n, k, off, nv, bs });
    Ok(Some(w))
}

/// 2026-10-03: The entry for `b` when it is a registered key with shape `[n, k]`; `None`
/// when `b` lies outside every FP8 copy; an error when `b` is inside one but is not a key
/// with that shape (a BF16 read there would read FP8 bytes).
fn find(b: DevicePtr, n: usize, k: usize) -> Result<Option<Entry>> {
    let m = map().read().unwrap();
    let Some((&key, e)) = m.range(..=b.0).next_back() else {
        return Ok(None);
    };
    if key == b.0 {
        if e.n == n && e.k == k {
            return Ok(Some(*e));
        }
        bail!(
            "METRALE_GLM_DENSE_FP8: weight {b} is registered as [{}, {}] FP8 but was used as \
             [{n}, {k}]; its BF16 original is freed",
            e.n,
            e.k
        );
    }
    if b.0 < key + (e.n * e.k) as u64 {
        bail!(
            "METRALE_GLM_DENSE_FP8: pointer {b} lies inside the FP8 copy at {:#x} ([{}, {}]); a \
             BF16 read there would read FP8 bytes",
            key,
            e.n,
            e.k
        );
    }
    Ok(None)
}

/// 2026-10-04: An error when `b` is, or lies inside, a registered FP8 copy. For a call site
/// that reads a `[n, k]` BF16 weight directly instead of through [`route`]
/// (`Glm5NextDsaLayer::write_kv_rows`); those weights are not registered today, and this
/// keeps a future registration from turning into a silent BF16 read of FP8 bytes.
pub fn ensure_bf16(b: DevicePtr, n: usize, k: usize, site: &str) -> Result<()> {
    if !dense_fp8() {
        return Ok(());
    }
    if find(b, n, k)?.is_some() {
        bail!(
            "METRALE_GLM_DENSE_FP8: {site} reads weight {b} ([{n}, {k}]) as BF16, but it is \
             registered as an FP8 copy; route that read through dense_fp8::route"
        );
    }
    Ok(())
}

/// 2026-10-03: The FP8 copy registered under key `ptr` with shape `[n, k]`, if any.
/// 2026-10-06: Under `METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW` a weight with an NVFP4 copy may be
/// block-scaled: its `row_scale` then holds `[N/128, K/128]` block scales ([`is_block_scaled`]).
pub fn lookup(ptr: DevicePtr, n: usize, k: usize) -> Option<Fp8DenseWeight> {
    if !dense_fp8() {
        return None;
    }
    let m = map().read().unwrap();
    m.get(&ptr.0).filter(|e| e.n == n && e.k == k).map(|e| e.w)
}

/// 2026-10-06: Whether the copy under key `ptr` (`[n, k]`) is 128x128 block-scaled
/// (`METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW`); false when unregistered or per-row.
pub fn is_block_scaled(ptr: DevicePtr, n: usize, k: usize) -> bool {
    if !dense_fp8() {
        return false;
    }
    let m = map().read().unwrap();
    m.get(&ptr.0).is_some_and(|e| e.n == n && e.k == k && e.bs)
}

/// 2026-10-04: `dense_gemv_bf16` (module `gemv`), the BF16-out GEMV handle the three GLM
/// `gemm` wrappers pass; resolved once, `KernelHandle(0)` when absent.
fn bf16_gemv_handle(gpu: &dyn GpuBackend) -> KernelHandle {
    static H: OnceLock<KernelHandle> = OnceLock::new();
    *H.get_or_init(|| metrale_model_layers::layers::try_kernel(gpu, "gemv", "dense_gemv_bf16"))
}

/// 2026-10-03: The BF16 weight `C[m, n] = A[m, k] @ W^T` must read for weight pointer `b`,
/// running the FP8 GEMV itself when it applies (module doc). Output rows of an FP8 GEMV are
/// packed at stride `n`, as `ops::dense_mm_bf16` writes them.
#[allow(clippy::too_many_arguments)]
pub fn route(
    gpu: &dyn GpuBackend,
    gemv: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<Route> {
    // 2026-10-05: `METRALE_GLM_DENSE_NVFP4`: 1..=16 rows on the BF16-out GEMV of a weight with
    // an NVFP4 copy run the NVFP4 GEMV, ahead of the tensor-core FP8 GEMV below; M = 1 takes
    // it too, so decode and verify rows read the same weights with the same per-row bits.
    if dense_nvfp4().any()
        && (1..=NV4_MAX_M).contains(&m)
        && gemv.0 != 0
        && gemv.0 == bf16_gemv_handle(gpu).0
        && let Some(e) = find(b, n, k)?
        && let Some(q) = e.nv
    {
        nv4_gemv(gpu, a, &q, c, m, n, k, stream)?;
        return Ok(Route::Done);
    }
    // 2026-10-04: `METRALE_GLM_GEMV_TC=1`: 1..=16 rows on the BF16-out GEMV run the
    // row-invariant tensor-core GEMV (`ops::dense_gemv_tcm`) on the FP8 copy when this
    // weight has one, else on the BF16 weight. Every M of a weight takes it, M = 1 included,
    // so decode and verify rows agree; a shape it declines (k not a multiple of the k-block)
    // declines at every M and falls through below.
    if ops::dense_gemv_tcm::gemv_tc_enabled()
        && (1..=ops::dense_gemv_tcm::TCM_MAX_M as usize).contains(&m)
        && gemv.0 != 0
        && gemv.0 == bf16_gemv_handle(gpu).0
    {
        let done = match if dense_fp8() { find(b, n, k)? } else { None } {
            // 2026-10-06: a block-scaled weight has no per-row GEMV; it falls through.
            Some(e) if e.bs => false,
            Some(e) => ops::dense_gemv_tcm::try_fp8(
                gpu, a, &e.w, c, m as u32, n as u32, k as u32, n as u32, stream,
            )?,
            None => ops::dense_gemv_tcm::try_bf16(
                gpu,
                a,
                &DenseWeight { weight: b },
                c,
                m as u32,
                n as u32,
                k as u32,
                n as u32,
                stream,
            )?,
        };
        if done {
            return Ok(Route::Done);
        }
    }
    if !dense_fp8() {
        return Ok(Route::Weight(b));
    }
    let Some(e) = find(b, n, k)? else {
        return Ok(Route::Weight(b));
    };
    if m == 0 {
        return Ok(Route::Done);
    }
    let Some(kk) = kernels(gpu) else {
        bail!("METRALE_GLM_DENSE_FP8: weight {b} is registered but the kernels are missing");
    };
    // 2026-10-06: a block-scaled weight (`e.bs`) has no per-row GEMV; it takes the dequant.
    if m <= ops::DENSE_GEMV_FP8W_BATCHM_MAX_M as usize && !e.bs {
        if gemv.0 == kk.bf16_gemv.0 {
            if m == 1 {
                ops::dense_gemv_fp8w(gpu, kk.gemv1, a, &e.w, c, n as u32, k as u32, stream)?;
            } else {
                ops::dense_gemv_fp8w_batchm(
                    gpu, kk.batchm, a, &e.w, c, m as u32, 1, n as u32, k as u32, n as u32, stream,
                )?;
            }
            HITS.fetch_add(1, Ordering::Relaxed);
            if !FIRST_HIT.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "METRALE_GLM_DENSE_FP8: first FP8 GEMV routed ({m} rows, {n}x{k}); {} weights \
                     registered",
                    map().read().unwrap().len()
                );
            }
            return Ok(Route::Done);
        }
        if !HANDLE_MISMATCH.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8: a registered weight ({n}x{k}) was called with a GEMV other \
                 than dense_gemv_bf16; that call reads a BF16 dequant of the FP8 copy"
            );
        }
    }
    // 2026-10-05: `METRALE_GLM_DENSE_FP8_W8A8=1`: a wide call runs the W8A8 GEMM into `c`
    // when it can (module doc), else falls through to the dequant.
    if dense_fp8_w8a8()
        && m > ops::DENSE_GEMV_FP8W_BATCHM_MAX_M as usize
        && w8a8(gpu, gemv.0 == kk.bf16_gemv.0, a, &e, c, m, stream, None)?
    {
        return Ok(Route::Done);
    }
    let p = dequant(gpu, &kk, b.0, &e, m, stream)?;
    Ok(Route::Weight(p))
}

/// 2026-10-05: `C[m, n] = A[m, k] @ W^T` over the NVFP4 copy `q`, 1..=[`NV4_MAX_M`] packed
/// rows (A at stride `k`, C at stride `n`), on the CUDA-core tier for `m` (never the
/// tensor-core `gemv_tc` route, whose bits depend on M). Grid ceil(n / 4), block 256.
#[allow(clippy::too_many_arguments)]
pub fn nv4_gemv(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    nv4_gemv_uncounted(gpu, a, q, c, m, n, k, stream)?;
    NV4_HITS.fetch_add(1, Ordering::Relaxed);
    if !NV4_FIRST_HIT.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "METRALE_GLM_DENSE_NVFP4: first NVFP4 GEMV routed ({m} rows, {n}x{k}); {} NVFP4 copies \
             ({:.2} GB)",
            NV4_WEIGHTS.load(Ordering::Relaxed),
            NV4_BYTES.load(Ordering::Relaxed) as f64 / 1e9
        );
    }
    Ok(())
}

/// 2026-10-06: [`nv4_gemv`] without the dense lever's hit counter and first-hit log, for an
/// NVFP4 copy that is not in the dense registry (the MTP draft head,
/// `METRALE_GLM_MTP_HEAD_NVFP4`). Same tiers, grid ceil(n / 4) (one block per 4 outputs; the
/// dense microtest's `N % 4 = 1` shape covers a partial last block), block 256, no allocation
/// or sync, so it is capture-safe. Each row's bits equal the M = 1 launch on that row.
#[allow(clippy::too_many_arguments)]
pub fn nv4_gemv_uncounted(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    let Some(kk) = nv4_kernels(gpu) else {
        bail!("METRALE_GLM_DENSE_NVFP4: a weight has an NVFP4 copy but the kernels are missing");
    };
    let (h, takes_m) = match m {
        1 => (kk.gemv1, false),
        2 => (kk.b2, false),
        3 => (
            match kk.b3s {
                Some(h) if nv4_b3_staged() => h,
                _ => kk.b3,
            },
            false,
        ),
        4..=8 => (kk.tiers[m - 4], true),
        9..=NV4_MAX_M => (kk.tiers[5], true),
        _ => bail!("METRALE_GLM_DENSE_NVFP4: {m} rows is outside 1..={NV4_MAX_M}"),
    };
    let mut l = KernelLaunch::new(gpu, h)
        .grid([div_ceil(n as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(q.weight)
        .arg_ptr(q.weight_scale)
        .arg_f32(q.weight_scale_2)
        .arg_ptr(c);
    if takes_m {
        l = l.arg_u32(m as u32);
    }
    l.arg_u32(n as u32).arg_u32(k as u32).launch(stream)
}

/// 2026-10-06: An NVFP4 copy of the BF16 `[n, k]` weight at `bf16`, made by the quantizer the
/// dense lever uses ([`register`]: `quantize_to_nvfp4` with `quantize_bf16_to_nvfp4_mse`;
/// packed E2M1 `[n, k/2]`, one E4M3 scale per 16 K, FP32 tensor scale), in the layout
/// [`nv4_gemv_uncounted`] reads. Outside the registry: nothing is registered, freed or
/// counted, and the BF16 original is untouched. `Ok(None)` when the NVFP4 kernels are missing
/// (logged once) or the shape is empty or `k % 16 != 0`. Synchronizes `stream`.
pub fn quantize_nvfp4_copy(
    gpu: &dyn GpuBackend,
    bf16: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<Option<QuantizedWeight>> {
    if n == 0 || k == 0 || !k.is_multiple_of(16) || bf16.is_null() {
        return Ok(None);
    }
    let Some(q) = nv4_kernels(gpu) else {
        return Ok(None);
    };
    metrale_model_layers::weight_map::quantize_to_nvfp4(
        &DenseWeight { weight: bf16 },
        n,
        k,
        gpu,
        q.absmax,
        q.quant,
        stream,
    )
    .map(Some)
}

/// 2026-10-05: NVFP4 GEMVs [`route`] launched so far (host-side; graph replays not counted).
pub fn nv4_hits() -> u64 {
    NV4_HITS.load(Ordering::Relaxed)
}

/// 2026-10-05: The NVFP4 copy registered under key `ptr` with shape `[n, k]`, if any.
pub fn lookup_nv4(ptr: DevicePtr, n: usize, k: usize) -> Option<QuantizedWeight> {
    if !dense_fp8() {
        return None;
    }
    let m = map().read().unwrap();
    m.get(&ptr.0).filter(|e| e.n == n && e.k == k).and_then(|e| e.nv)
}

/// 2026-10-05: The W8A8 kernels, resolved once; `None` (logged) when the target lacks either.
fn w8a8_kernels(gpu: &dyn GpuBackend) -> Option<W8a8Kernels> {
    *W8A8_KERNELS.get_or_init(|| {
        let quant = ops::Fp8ActQuant::resolve(gpu);
        let gemm = match ops::fp8_gemm_rowscale_kernel(gpu) {
            Ok(Some(h)) => Some(h),
            Ok(None) => None,
            Err(e) => {
                tracing::warn!("METRALE_GLM_DENSE_FP8_W8A8: {e:#}");
                None
            }
        };
        match gemm {
            Some(gemm) if quant.available() => Some(W8a8Kernels { quant, gemm }),
            _ => {
                tracing::warn!(
                    "METRALE_GLM_DENSE_FP8_W8A8: this target lacks {}::{} or \
                     per_token_group_quant_fp8; prefill GEMMs keep the dequant path",
                    ops::FP8_GEMM_PIPE_MODULE,
                    ops::FP8_GEMM_ROWSCALE_ENTRY
                );
                None
            }
        }
    })
}

/// 2026-10-05: Log, once per reason, that a wide call kept the dequant path.
fn w8a8_skip(why: W8a8Skip, m: usize, n: usize, k: usize) -> Result<bool> {
    if !W8A8_SKIP_LOGGED[why as usize].swap(true, Ordering::Relaxed) {
        let text = match why {
            W8a8Skip::FewRows => format!("fewer than {} rows", w8a8_min_rows()),
            W8a8Skip::Shape => {
                "k not a multiple of 128, n not a multiple of 64, or c not 4-byte aligned".into()
            }
            W8a8Skip::NoKernels => "the W8A8 kernels are missing".into(),
            W8a8Skip::NotBf16Out => {
                "the caller's GEMV is not dense_gemv_bf16 (its output may not be BF16)".into()
            }
            W8a8Skip::NoScratch => "no W8A8 scratch (finish_load did not allocate one)".into(),
            W8a8Skip::TooWide => format!(
                "the activation does not fit the W8A8 scratch ({} rows at the widest k; \
                 METRALE_GLM_DENSE_FP8_W8A8_ROWS)",
                dense_fp8_w8a8_rows()
            ),
        };
        if matches!(why, W8a8Skip::FewRows) {
            tracing::info!(
                "METRALE_GLM_DENSE_FP8_W8A8: first {m}-row GEMM ({n}x{k}) kept the dequant \
                 path: {text}"
            );
        } else {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8_W8A8: first {m}-row GEMM ({n}x{k}) kept the dequant \
                 path: {text}"
            );
        }
    }
    Ok(false)
}

/// 2026-10-05: `c[m, n] = W8A8(a, e)` when every condition in the module doc holds (and
/// `bf16_out`: the caller passed the BF16-out `dense_gemv_bf16` handle): quantize
/// `a` into the scratch, then the per-row-scale GEMM over the FP8 copy, both on `stream`.
/// `Ok(false)`, launching nothing, otherwise.
///
/// 2026-10-07: With `fill` (`METRALE_GLM_NORM_FP8_QUANT_FUSE`, [`w8a8_fused_input`]) the quant
/// launch is replaced by `fill(a_fp8, a_scale)`: the caller's kernel, run on `stream`, writes
/// `a` itself AND the FP8 bytes and scales the quantizer would make from it. `fill` runs only
/// when this returns `Ok(true)`; any skip returns `Ok(false)` before calling it. The held
/// share-scope quant is dropped (the scratch changes under it).
#[allow(clippy::too_many_arguments)]
fn w8a8(
    gpu: &dyn GpuBackend,
    bf16_out: bool,
    a: DevicePtr,
    e: &Entry,
    c: DevicePtr,
    m: usize,
    stream: u64,
    fill: Option<&mut dyn FnMut(DevicePtr, DevicePtr) -> Result<()>>,
) -> Result<bool> {
    let (n, k) = (e.n, e.k);
    // The rowscale GEMM writes BF16; a caller passing another GEMV family (the FP32-out
    // `gemv_f32` sites, all M = 1 on unregistered weights today) keeps its own path.
    if !bf16_out {
        return w8a8_skip(W8a8Skip::NotBf16Out, m, n, k);
    }
    // 2026-10-06: A block-scaled weight (`METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW`) keeps the
    // default 64-row floor: its CUTLASS GEMM is gated from 64 rows only.
    if m < w8a8_min_rows() || (e.bs && m < W8A8_MIN_ROWS) {
        return w8a8_skip(W8a8Skip::FewRows, m, n, k);
    }
    if !k.is_multiple_of(ops::FP8_GEMM_PIPE_KGROUP as usize)
        || !n.is_multiple_of(ops::FP8_GEMM_PIPE_BN as usize)
        || !c.0.is_multiple_of(4)
    {
        return w8a8_skip(W8a8Skip::Shape, m, n, k);
    }
    let Some(kk) = w8a8_kernels(gpu) else {
        return w8a8_skip(W8a8Skip::NoKernels, m, n, k);
    };
    let mut st = W8A8.lock().unwrap();
    let Some(s) = st.scratch else {
        return w8a8_skip(W8a8Skip::NoScratch, m, n, k);
    };
    let groups = k / ops::FP8_GEMM_PIPE_KGROUP as usize;
    if m * k > s.fp8_cap || m * groups * 4 > s.scale_cap || groups > s.ones_len {
        return w8a8_skip(W8a8Skip::TooWide, m, n, k);
    }
    let capturing = gpu.stream_is_capturing(stream);
    if !capturing {
        if let Some(prev) = st.last_stream
            && prev != stream
        {
            // Earlier W8A8 GEMMs on `prev` may still read the scratch this quant overwrites.
            gpu.synchronize(prev)?;
        }
        st.last_stream = Some(stream);
    }
    // 2026-10-06: `METRALE_GLM_DENSE_FP8_W8A8_SHARE_QUANT` (module doc): reuse the held quant
    // when this input is the one it was made from, inside the same share scope.
    let generation = if fill.is_none() && w8a8_share_quant() && !capturing {
        share_generation_for(a, m * k * 2)
    } else {
        None
    };
    let want = generation.map(|generation| W8a8Held {
        a: a.0,
        m,
        k,
        stream,
        generation,
    });
    if let Some(fill) = fill {
        // 2026-10-07: The caller's fused kernel writes `a` and the quant; nothing is held.
        fill(s.a_fp8, s.a_scale)?;
        st.held = None;
    } else if want.is_some() && st.held == want {
        W8A8_REUSED.fetch_add(1, Ordering::Relaxed);
        if !W8A8_REUSE_FIRST.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "METRALE_GLM_DENSE_FP8_W8A8_SHARE_QUANT: first reused quant ({m} rows, k {k})"
            );
        }
    } else {
        ops::per_token_group_quant_fp8(
            gpu, kk.quant, a, s.a_fp8, s.a_scale, m as u32, k as u32, stream,
        )?;
        st.held = want;
    }
    if e.bs {
        // 2026-10-06: `METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW` (module doc).
        super::dense_fp8_gw::gemm(gpu, s.a_fp8, s.a_scale, &e.w, c, m, n, k, stream)?;
    } else {
        ops::fp8_gemm_t_rowscale(
            gpu,
            kk.gemm,
            s.a_fp8,
            s.a_scale,
            s.ones,
            e.w.weight,
            e.w.row_scale,
            c,
            m as u32,
            n as u32,
            k as u32,
            stream,
        )?;
    }
    W8A8_GEMMS.fetch_add(1, Ordering::Relaxed);
    if !W8A8_FIRST.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "METRALE_GLM_DENSE_FP8_W8A8: first W8A8 GEMM routed ({m} rows, {n}x{k}); scratch \
             {:.1} MB",
            s.bytes as f64 / 1e6
        );
    }
    Ok(true)
}

/// 2026-10-05: Allocate the W8A8 scratch (module doc) once the weights are registered.
/// Returns its bytes; 0, allocating nothing, when the lever is off, the kernels are missing or
/// no registered weight has `k % 128 == 0` and `n % 64 == 0`.
fn finish_load_w8a8(gpu: &dyn GpuBackend) -> Result<usize> {
    if !dense_fp8_w8a8() || w8a8_kernels(gpu).is_none() {
        return Ok(0);
    }
    let kg = ops::FP8_GEMM_PIPE_KGROUP as usize;
    let max_k = map()
        .read()
        .unwrap()
        .values()
        .filter(|e| e.k.is_multiple_of(kg) && e.n.is_multiple_of(ops::FP8_GEMM_PIPE_BN as usize))
        .map(|e| e.k)
        .max()
        .unwrap_or(0);
    if max_k == 0 {
        return Ok(0);
    }
    let rows = dense_fp8_w8a8_rows();
    let groups = max_k / kg;
    let fp8_cap = rows * max_k;
    let scale_cap = rows * groups * 4;
    let scale_off = fp8_cap.next_multiple_of(ARENA_ALIGN);
    let ones_off = (scale_off + scale_cap).next_multiple_of(ARENA_ALIGN);
    let bytes = ones_off + groups * 4;
    let mut st = W8A8.lock().unwrap();
    if let Some(have) = st.scratch
        && have.fp8_cap >= fp8_cap
        && have.scale_cap >= scale_cap
        && have.ones_len >= groups
    {
        return Ok(have.bytes);
    }
    // 2026-10-06: A new scratch holds no quant.
    st.held = None;
    if let Some(old) = st.scratch.take() {
        gpu.free(old.base)?;
    }
    let base = gpu.alloc(bytes)?;
    let ones: Vec<u8> = (0..groups).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    gpu.copy_h2d(&ones, base.offset(ones_off))?;
    st.scratch = Some(W8a8Scratch {
        base,
        bytes,
        a_fp8: base,
        fp8_cap,
        a_scale: base.offset(scale_off),
        scale_cap,
        ones: base.offset(ones_off),
        ones_len: groups,
    });
    st.last_stream = None;
    tracing::warn!(
        "METRALE_GLM_DENSE_FP8_W8A8: activation scratch {:.1} MB ({rows} rows x K {max_k}: FP8 \
         {:.1} MB + scales {:.1} MB + {groups} ones), allocated at load",
        bytes as f64 / 1e6,
        fp8_cap as f64 / 1e6,
        scale_cap as f64 / 1e6
    );
    Ok(bytes)
}

/// 2026-10-05: W8A8 GEMM launches issued by [`route`] so far (host-side; graph replays are not
/// counted).
pub fn w8a8_gemms() -> u64 {
    W8A8_GEMMS.load(Ordering::Relaxed)
}

/// 2026-10-05: Bytes of the W8A8 scratch [`finish_load`] allocated; 0 when none.
pub fn w8a8_scratch_bytes() -> usize {
    W8A8.lock().unwrap().scratch.map_or(0, |s| s.bytes)
}

/// 2026-10-03: The arena address holding the BF16 dequant of `key`, dequantizing into it
/// unless the eager cache says it is already there (module doc).
fn dequant(
    gpu: &dyn GpuBackend,
    kk: &Kernels,
    key: u64,
    e: &Entry,
    m: usize,
    stream: u64,
) -> Result<DevicePtr> {
    let (base, size) = match *ARENA.lock().unwrap() {
        Some(a) => a,
        None => bail!(
            "METRALE_GLM_DENSE_FP8: a {m}-row GEMM on a converted weight needs the dequant arena, \
             which was not allocated (dense_fp8::finish_load did not run)"
        ),
    };
    let len = e.n * e.k * 2;
    if e.off + len > size {
        bail!(
            "METRALE_GLM_DENSE_FP8: arena of {size} B cannot hold [{}, {}] at offset {}",
            e.n,
            e.k,
            e.off
        );
    }
    let dst = base.offset(e.off);
    let launch = |s: u64| {
        if e.bs {
            super::dense_fp8_gw::dequant_block_scaled(gpu, &e.w, dst, e.n, e.k, s)
        } else {
            ops::dequant_fp8_rowscale_bf16(gpu, kk.dequant, &e.w, dst, e.n as u32, e.k as u32, s)
        }
    };
    let mut c = CACHE.lock().unwrap();
    if gpu.stream_is_capturing(stream) {
        if !c.poisoned {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8: a {m}-row GEMM on a converted weight was captured into a \
                 CUDA graph; the eager dequant cache is off from now on (each wide GEMM \
                 dequantizes its weight)"
            );
        }
        c.poisoned = true;
        c.resident.clear();
        launch(stream)?;
        DEQUANTS.fetch_add(1, Ordering::Relaxed);
        return Ok(dst);
    }
    if !c.poisoned && c.last_stream == Some(stream) && c.resident.iter().any(|r| r.2 == key) {
        return Ok(dst);
    }
    if let Some(prev) = c.last_stream
        && prev != stream
    {
        // Earlier GEMMs on `prev` may still read the arena this launch overwrites.
        gpu.synchronize(prev)?;
    }
    launch(stream)?;
    c.last_stream = Some(stream);
    c.resident
        .retain(|&(o, l, _)| o + l <= e.off || e.off + len <= o);
    if !c.poisoned {
        c.resident.push((e.off, len, key));
    }
    DEQUANTS.fetch_add(1, Ordering::Relaxed);
    if !FIRST_DEQUANT.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "METRALE_GLM_DENSE_FP8: first wide GEMM on a converted weight ({m} rows, {}x{}) reads \
             its BF16 dequant from the {:.1} MB arena",
            e.n,
            e.k,
            size as f64 / 1e6
        );
    }
    Ok(dst)
}

/// 2026-10-03: FP8 GEMV launches issued so far (host-side count; a graph replay is not
/// counted).
pub fn hits() -> u64 {
    HITS.load(Ordering::Relaxed)
}

/// 2026-10-03: Eager or captured dequant launches issued so far (host-side; cache hits and
/// graph replays are not counted).
pub fn dequants() -> u64 {
    DEQUANTS.load(Ordering::Relaxed)
}

/// 2026-10-03: What [`register_layer`] did for one layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerFp8 {
    /// FP8 bytes added (`n * k + 4 n` per weight).
    pub fp8_bytes: usize,
    /// BF16 bytes freed (`2 n k` per weight).
    pub bf16_freed: usize,
    /// This layer's dequant span in the arena, bytes.
    pub arena_bytes: usize,
}

/// 2026-10-03: Convert one text layer's weights (see the module doc for the list and what
/// stays BF16): quantize, free the BF16 originals, rewrite the fields to the FP8 keys and
/// lay the layer's dequant span out from arena offset 0.
pub fn register_layer(
    gpu: &dyn GpuBackend,
    mixer: &mut crate::glm5next_layer::Glm5NextMixer,
    mlp: &mut crate::glm5next_layer::Glm5NextMlpSite,
    mlp_cfg: &crate::glm5next_mlp::Glm5NextMlpConfig,
) -> Result<LayerFp8> {
    use crate::glm5next_layer::{Glm5NextMixer, Glm5NextMlpSite};
    if !dense_fp8() {
        return Ok(LayerFp8::default());
    }
    // 2026-10-05: A bad `METRALE_GLM_DENSE_NVFP4` value fails the load here, not silently.
    parse_nvfp4_classes(std::env::var("METRALE_GLM_DENSE_NVFP4").ok().as_deref())?;
    let nvc = dense_nvfp4();
    let mut list: Vec<(&mut DevicePtr, usize, usize, Class)> = Vec::new();
    match mixer {
        Glm5NextMixer::Kda { layer, cfg, .. } => {
            let w = &mut layer.weights;
            let (hid, qkv, hd) = (cfg.hidden, cfg.qkv_dim(), cfg.head_dim);
            list.push((&mut w.q_proj.weight, qkv, hid, Class::Kda));
            list.push((&mut w.k_proj.weight, qkv, hid, Class::Kda));
            list.push((&mut w.v_proj.weight, qkv, hid, Class::Kda));
            list.push((&mut w.g_a.weight, hd, hid, Class::Keep));
            list.push((&mut w.g_b.weight, qkv, hd, Class::Keep));
            list.push((&mut w.o_proj.weight, hid, qkv, Class::Kda));
        }
        Glm5NextMixer::Dsa(l) => {
            // Shapes as `decode_k` / `decode_k_wide` pass them to `gemm`.
            let l = &mut **l;
            let (c, w) = (&l.cfg, &mut l.weights);
            let heads_lat = c.local_heads * c.kv_lora_rank;
            list.push((&mut w.q_a_proj, c.q_lora_rank, c.hidden, Class::Dsa));
            list.push((&mut w.q_absorb, heads_lat, c.q_lora_rank, Class::Dsa));
            list.push((&mut w.kv_a_proj, c.kv_lora_rank, c.hidden, Class::Dsa));
            list.push((&mut w.o_absorb, c.hidden, heads_lat, Class::DsaO));
        }
    }
    let (w, inter, class) = match mlp {
        Glm5NextMlpSite::Dense(w) => (w, mlp_cfg.local_dense_intermediate, Class::Mlp),
        Glm5NextMlpSite::Moe(w) => (
            &mut w.shared,
            mlp_cfg.local_shared_intermediate,
            Class::Shared,
        ),
    };
    list.push((&mut w.gate_proj, inter, mlp_cfg.hidden, class));
    list.push((&mut w.up_proj, inter, mlp_cfg.hidden, class));
    list.push((&mut w.down_proj, mlp_cfg.hidden, inter, class));
    let mut out = LayerFp8::default();
    for (field, n, k, class) in list {
        convert_weight_with(gpu, field, n, k, &mut out, true, nvc.has(class))?;
    }
    Ok(out)
}

/// 2026-10-03: Convert one BF16 `[n, k]` weight of the layer `acc` describes: quantize it,
/// free the BF16 original, set `*field` to the FP8 copy's pointer (the registry key) and give
/// it the next arena offset of that layer. Returns `false`, changing nothing, when the lever
/// is off, the kernels are missing, the shape is empty or `k % 16 != 0`. [`register_layer`]
/// calls it per weight; `examples/glm5next_dense_fp8_microtest.rs` calls it directly.
pub fn convert_weight(
    gpu: &dyn GpuBackend,
    field: &mut DevicePtr,
    n: usize,
    k: usize,
    acc: &mut LayerFp8,
) -> Result<bool> {
    convert_weight_with(gpu, field, n, k, acc, true, false)
}

/// 2026-10-05: [`convert_weight`] that also makes the NVFP4 decode copy (as
/// `METRALE_GLM_DENSE_NVFP4` does for a selected class); for the microtest
/// (`examples/glm5next_dense_nvfp4_microtest.rs`). [`route`] reads the copy only while
/// [`dense_nvfp4`] selects some class.
pub fn convert_weight_nvfp4(
    gpu: &dyn GpuBackend,
    field: &mut DevicePtr,
    n: usize,
    k: usize,
    acc: &mut LayerFp8,
) -> Result<bool> {
    convert_weight_with(gpu, field, n, k, acc, true, true)
}

/// 2026-10-05: [`convert_weight`], with `free_original = false` for a store-owned BF16 original
/// that the caller must not free (see [`take_deferred_free`]).
fn convert_weight_with(
    gpu: &dyn GpuBackend,
    field: &mut DevicePtr,
    n: usize,
    k: usize,
    acc: &mut LayerFp8,
    free_original: bool,
    nv4: bool,
) -> Result<bool> {
    let off = acc.arena_bytes;
    let Some(w) = register(gpu, *field, n, k, off, free_original, nv4)? else {
        return Ok(false);
    };
    *field = w.weight;
    acc.fp8_bytes += n * k + 4 * n;
    acc.bf16_freed += n * k * 2;
    acc.arena_bytes = (off + n * k * 2).next_multiple_of(ARENA_ALIGN);
    ARENA_NEED.fetch_max(acc.arena_bytes, Ordering::Relaxed);
    Ok(true)
}

/// 2026-10-05: Whether [`register_mtp`] converts the MTP block: `METRALE_GLM_DENSE_FP8=1` and
/// `METRALE_GLM_MTP_DENSE_FP8` not `0` (comb13: its own off switch, so the draft-only MTP part
/// can be turned off without a rebuild; with it off the MTP block stays BF16 as before).
pub fn mtp_dense_fp8() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        dense_fp8() && std::env::var("METRALE_GLM_MTP_DENSE_FP8").as_deref() != Ok("0")
    })
}

/// 2026-10-05: Convert the MTP block's dense weights (module doc) and grow the dequant arena to
/// cover them. Runs from `load_glm5next_mtp_module`, after every text layer is registered and
/// before the KV pool is sized. `eh_proj` is `[hidden, 2 * hidden]` and store-owned (its BF16
/// original is freed later by `prune_after_load`); the block's own weights are freed here.
/// A no-op (`LayerFp8::default()`) with the lever off.
pub fn register_mtp(
    gpu: &dyn GpuBackend,
    layer: &mut crate::glm5next_layer::Glm5NextLayer,
    eh_proj: &mut DenseWeight,
) -> Result<LayerFp8> {
    use crate::glm5next_layer::{Glm5NextMixer, Glm5NextMlpSite};
    if !mtp_dense_fp8() {
        return Ok(LayerFp8::default());
    }
    let hid = layer.mlp_cfg.hidden;
    let nv_mtp = dense_nvfp4().has(Class::Mtp);
    let mut out = LayerFp8::default();
    convert_weight_with(gpu, &mut eh_proj.weight, hid, 2 * hid, &mut out, false, nv_mtp)?;
    let mut list: Vec<(&mut DevicePtr, usize, usize)> = Vec::new();
    match &mut layer.mixer {
        Glm5NextMixer::Dsa(l) => {
            let l = &mut **l;
            let (c, w) = (&l.cfg, &mut l.weights);
            let heads_lat = c.local_heads * c.kv_lora_rank;
            list.push((&mut w.q_a_proj, c.q_lora_rank, c.hidden));
            list.push((&mut w.q_absorb, heads_lat, c.q_lora_rank));
            // `kv_a_proj` stays BF16: `write_kv_rows` reads it directly.
            list.push((&mut w.o_absorb, c.hidden, heads_lat));
        }
        Glm5NextMixer::Kda { .. } => {
            bail!("METRALE_GLM_DENSE_FP8: the MTP block is not a DSA layer")
        }
    }
    let mlp_cfg = &layer.mlp_cfg;
    let (w, inter) = match &mut layer.mlp {
        Glm5NextMlpSite::Dense(w) => (w, mlp_cfg.local_dense_intermediate),
        Glm5NextMlpSite::Moe(w) => (&mut w.shared, mlp_cfg.local_shared_intermediate),
    };
    list.push((&mut w.gate_proj, inter, mlp_cfg.hidden));
    list.push((&mut w.up_proj, inter, mlp_cfg.hidden));
    list.push((&mut w.down_proj, mlp_cfg.hidden, inter));
    for (field, n, k) in list {
        convert_weight_with(gpu, field, n, k, &mut out, true, nv_mtp)?;
    }
    let arena = finish_load(gpu)?;
    tracing::info!(
        "METRALE_GLM_DENSE_FP8: MTP block: {:.1} MB of BF16 dense weights replaced by {:.1} MB of          FP8 copies (eh_proj's {:.1} MB BF16 is freed by prune_after_load); dequant arena {:.1} MB",
        out.bf16_freed as f64 / 1e6,
        out.fp8_bytes as f64 / 1e6,
        (2 * hid * hid * 2) as f64 / 1e6,
        arena as f64 / 1e6,
    );
    Ok(out)
}

/// 2026-10-05: True, once, when `original` is a BF16 pointer [`register_mtp`] quantized and left
/// to its owner to free; the caller then frees it.
pub fn take_deferred_free(original: DevicePtr) -> bool {
    let mut v = DEFERRED_FREE.lock().unwrap();
    match v.iter().position(|&p| p == original.0) {
        Some(i) => {
            v.swap_remove(i);
            true
        }
        None => false,
    }
}

/// 2026-10-03: After every text layer is registered: allocate the dequant arena (the
/// largest per-layer span). Returns its bytes; 0, allocating nothing, when the lever is off
/// or nothing was converted. Must run before the KV pool is sized so the ledger counts it.
/// 2026-10-05: Also allocates the W8A8 activation scratch under `METRALE_GLM_DENSE_FP8_W8A8=1`
/// (module doc; logged with its size, not in the returned bytes, [`w8a8_scratch_bytes`]).
pub fn finish_load(gpu: &dyn GpuBackend) -> Result<usize> {
    // 2026-10-05: Resolve the W8A8 lever here, at load, so `METRALE_GLM_DENSE_FP8_W8A8=1`
    // without `METRALE_GLM_DENSE_FP8=1` is warned about (nothing else reads it then).
    let _ = dense_fp8_w8a8();
    // 2026-10-06: Likewise `METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW` (REFUSED / NOT engaged logs).
    let _ = super::dense_fp8_gw::cutlass_gw();
    if !dense_fp8() {
        return Ok(0);
    }
    let need = ARENA_NEED.load(Ordering::Relaxed);
    if need == 0 {
        return Ok(0);
    }
    let arena = alloc_arena(gpu, need)?;
    finish_load_w8a8(gpu)?;
    // 2026-10-06: `METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW`: warm the CUTLASS workspace (before the
    // KV pool is sized) and log ENGAGED / NOT engaged.
    super::dense_fp8_gw::finish_load(
        GW_WEIGHTS.load(Ordering::Relaxed),
        GW_SCALE_BYTES.load(Ordering::Relaxed),
    )?;
    if dense_nvfp4().any() {
        tracing::info!(
            "METRALE_GLM_DENSE_NVFP4: {} NVFP4 decode copies, {:.2} GB/rank (on top of the FP8 \
             copies, which prefill keeps reading)",
            NV4_WEIGHTS.load(Ordering::Relaxed),
            NV4_BYTES.load(Ordering::Relaxed) as f64 / 1e9
        );
    }
    Ok(arena)
}

/// 2026-10-03: The dequant arena of [`finish_load`], at least `need` bytes (2026-10-05: moved
/// out of `finish_load` unchanged).
fn alloc_arena(gpu: &dyn GpuBackend, need: usize) -> Result<usize> {
    let mut a = ARENA.lock().unwrap();
    if let Some((_, have)) = *a
        && have >= need
    {
        return Ok(have);
    }
    if let Some((old, _)) = a.take() {
        gpu.free(old)?;
    }
    let p = gpu.alloc(need)?;
    *a = Some((p, need));
    *CACHE.lock().unwrap() = Cache {
        resident: Vec::new(),
        poisoned: false,
        last_stream: None,
    };
    Ok(need)
}

/// 2026-10-03: The bytes a decode GEMV on `[n, k]` weight `ptr` actually streams, for the
/// L2 prefetch plan (`METRALE_GLM_DECODE_L2_PREFETCH`): the FP8 copy (`n * k` bytes) when
/// `ptr` is a registered key, else the BF16 matrix.
pub fn decode_span(ptr: DevicePtr, n: usize, k: usize) -> crate::glm5next_layer::L2Span {
    // 2026-10-05: A weight with an NVFP4 copy streams its packed E2M1 bytes in decode.
    if dense_nvfp4().any()
        && let Some(q) = lookup_nv4(ptr, n, k)
    {
        return crate::glm5next_layer::L2Span {
            ptr: q.weight,
            bytes: n * k / 2,
        };
    }
    match lookup(ptr, n, k) {
        Some(w) => crate::glm5next_layer::L2Span {
            ptr: w.weight,
            bytes: n * k,
        },
        None => crate::glm5next_layer::prefetch::bf16_matrix_span(ptr, n, k),
    }
}
