// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `METRALE_GLM_DENSE_FP8=1`: FP8 E4M3 weight-only shadows of the GLM-5.3
//! non-expert decode projections, and the GEMV dispatch that reads them.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Off (the default), nothing is quantized and [`try_gemv`] returns `false` without
//!   launching, so every projection runs exactly as before.
//! - On, the loader ([`register_layer`]) quantizes, per text layer (0..num_hidden_layers;
//!   the MTP block is loaded elsewhere and never registered), the BF16 weights listed in
//!   [`register_layer`] to `[N, K]` E4M3 + one FP32 scale per output row
//!   (`quantize_bf16_to_fp8`, max|row| / 448) and keys them by the BF16 device pointer.
//!   The BF16 weight stays resident: prefill (rows > 16, cuBLASLt / tile GEMM) still reads
//!   it in this first phase, so the FP8 copies are extra memory (N * K + 4 N bytes each).
//! - [`try_gemv`] takes over a projection only when all hold: lever on, 1..=16 rows, the
//!   weight pointer is registered with the same N and K, and the caller's GEMV is the BF16-out
//!   `dense_gemv_bf16` (FP32-out callers keep BF16). 1 row runs `dense_gemv_fp8w`, 2..=16
//!   rows `dense_gemv_fp8w_batchm`, whose rows are bit-identical to `dense_gemv_fp8w` on the
//!   same row (kernel header), so a 1-row decode and a K-row verify produce the same bits for
//!   a row, as the BF16 pair did.
//! - Weights deliberately left BF16: the MoE router (expert selection), the DSA indexer
//!   (`wk`, `compress_gate`, `weights_proj`, `wq_b`: top-k token selection), the KDA
//!   recurrence gates `f_a`, `f_b` (forget-gate decay) and `b_proj` (beta), `embed_tokens`,
//!   `lm_head` (its FP8 option is `--lm-head-dtype fp8`), the mHC `hc_fn` (not a GEMV), the
//!   MTP block and any DFlash drafter. Together they are < 0.4 GB/rank of decode reads.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight};

/// 2026-10-03: `METRALE_GLM_DENSE_FP8=1` opts in; read once.
pub fn dense_fp8() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_DENSE_FP8").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8=1 - GLM-5.3 non-expert decode/verify GEMVs (<= 16 rows) read \
                 FP8 E4M3 weight copies (per-row scale); NOT byte-identical to BF16"
            );
        }
        on
    })
}

#[derive(Clone, Copy)]
struct Entry {
    w: Fp8DenseWeight,
    n: usize,
    k: usize,
}

#[derive(Clone, Copy)]
struct Kernels {
    quant: KernelHandle,
    gemv1: KernelHandle,
    batchm: KernelHandle,
    bf16_gemv: KernelHandle,
}

static KERNELS: OnceLock<Option<Kernels>> = OnceLock::new();
static MAP: OnceLock<RwLock<HashMap<u64, Entry>>> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static FIRST_HIT: AtomicBool = AtomicBool::new(false);
static HANDLE_MISMATCH: AtomicBool = AtomicBool::new(false);

fn map() -> &'static RwLock<HashMap<u64, Entry>> {
    MAP.get_or_init(|| RwLock::new(HashMap::new()))
}

/// 2026-10-03: The kernels the lever needs, resolved once; `None` (logged) when any is
/// missing from the target, which leaves the lever inert.
fn kernels(gpu: &dyn GpuBackend) -> Option<Kernels> {
    *KERNELS.get_or_init(|| {
        let r = (|| -> Result<Kernels> {
            Ok(Kernels {
                quant: gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?,
                gemv1: gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?,
                batchm: gpu.kernel("dense_gemv_fp8w_batchm", "dense_gemv_fp8w_batchm")?,
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

/// 2026-10-03: Quantize one BF16 `[n, k]` weight and register its FP8 copy. Returns the
/// FP8 bytes added (0 when skipped: lever off, kernels missing, or `k % 16 != 0`).
pub fn register(gpu: &dyn GpuBackend, bf16: DevicePtr, n: usize, k: usize) -> Result<usize> {
    if !dense_fp8() || n == 0 || k == 0 || !k.is_multiple_of(16) {
        return Ok(0);
    }
    let Some(kk) = kernels(gpu) else {
        return Ok(0);
    };
    if map().read().unwrap().contains_key(&bf16.0) {
        return Ok(0);
    }
    let w = metrale_model_layers::weight_map::quantize_to_fp8(
        &DenseWeight { weight: bf16 },
        n,
        k,
        gpu,
        kk.quant,
        gpu.default_stream(),
    )?;
    map().write().unwrap().insert(bf16.0, Entry { w, n, k });
    Ok(n * k + n * 4)
}

/// 2026-10-03: The FP8 copy registered for `bf16` with shape `[n, k]`, if any.
pub fn lookup(bf16: DevicePtr, n: usize, k: usize) -> Option<Fp8DenseWeight> {
    if !dense_fp8() {
        return None;
    }
    let m = map().read().unwrap();
    m.get(&bf16.0).filter(|e| e.n == n && e.k == k).map(|e| e.w)
}

/// 2026-10-03: Run `C[m, n] = A[m, k] @ W^T` on the FP8 copy of `b` when the lever applies
/// (module doc); `Ok(false)` means nothing was launched and the caller runs its BF16 path.
/// Output rows are packed at stride `n`, as `ops::dense_mm_bf16` writes them.
#[allow(clippy::too_many_arguments)]
pub fn try_gemv(
    gpu: &dyn GpuBackend,
    gemv: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    if !dense_fp8() || m == 0 || m > ops::DENSE_GEMV_FP8W_BATCHM_MAX_M as usize {
        return Ok(false);
    }
    let Some(w) = lookup(b, n, k) else {
        return Ok(false);
    };
    let Some(kk) = kernels(gpu) else {
        return Ok(false);
    };
    if gemv.0 != kk.bf16_gemv.0 {
        if !HANDLE_MISMATCH.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8: a registered weight ({n}x{k}) was called with a GEMV other \
                 than dense_gemv_bf16; that call stays BF16"
            );
        }
        return Ok(false);
    }
    if m == 1 {
        ops::dense_gemv_fp8w(gpu, kk.gemv1, a, &w, c, n as u32, k as u32, stream)?;
    } else {
        ops::dense_gemv_fp8w_batchm(
            gpu, kk.batchm, a, &w, c, m as u32, 1, n as u32, k as u32, n as u32, stream,
        )?;
    }
    HITS.fetch_add(1, Ordering::Relaxed);
    if !FIRST_HIT.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "METRALE_GLM_DENSE_FP8: first FP8 GEMV routed ({m} rows, {n}x{k}); {} weights registered",
            map().read().unwrap().len()
        );
    }
    Ok(true)
}

/// 2026-10-03: FP8 launches issued so far (host-side count; a graph replay is not counted).
pub fn hits() -> u64 {
    HITS.load(Ordering::Relaxed)
}

/// 2026-10-03: Register one text layer's converted weights (see the module doc for the
/// list and what stays BF16). Returns `(fp8 bytes added, bf16 bytes covered)`.
pub fn register_layer(
    gpu: &dyn GpuBackend,
    mixer: &crate::glm5next_layer::Glm5NextMixer,
    mlp: &crate::glm5next_layer::Glm5NextMlpSite,
    mlp_cfg: &crate::glm5next_mlp::Glm5NextMlpConfig,
) -> Result<(usize, usize)> {
    use crate::glm5next_layer::{Glm5NextMixer, Glm5NextMlpSite};
    if !dense_fp8() {
        return Ok((0, 0));
    }
    let mut list: Vec<(DevicePtr, usize, usize)> = Vec::new();
    match mixer {
        Glm5NextMixer::Kda { layer, cfg, .. } => {
            let w = &layer.weights;
            let (hid, qkv, hd) = (cfg.hidden, cfg.qkv_dim(), cfg.head_dim);
            list.push((w.q_proj.weight, qkv, hid));
            list.push((w.k_proj.weight, qkv, hid));
            list.push((w.v_proj.weight, qkv, hid));
            list.push((w.g_a.weight, hd, hid));
            list.push((w.g_b.weight, qkv, hd));
            list.push((w.o_proj.weight, hid, qkv));
        }
        Glm5NextMixer::Dsa(l) => {
            // Shapes as `decode_k` passes them to its `gemm` calls.
            let (c, w) = (&l.cfg, &l.weights);
            let heads_lat = c.local_heads * c.kv_lora_rank;
            list.push((w.q_a_proj, c.q_lora_rank, c.hidden));
            list.push((w.q_absorb, heads_lat, c.q_lora_rank));
            list.push((w.kv_a_proj, c.kv_lora_rank, c.hidden));
            list.push((w.o_absorb, c.hidden, heads_lat));
        }
    }
    let dense = |w: &crate::glm5next_mlp::weights::Glm5NextDenseMlpWeights, inter: usize| {
        [
            (w.gate_proj, inter, mlp_cfg.hidden),
            (w.up_proj, inter, mlp_cfg.hidden),
            (w.down_proj, mlp_cfg.hidden, inter),
        ]
    };
    match mlp {
        Glm5NextMlpSite::Dense(w) => list.extend(dense(w, mlp_cfg.local_dense_intermediate)),
        Glm5NextMlpSite::Moe(w) => list.extend(dense(&w.shared, mlp_cfg.local_shared_intermediate)),
    }
    let (mut added, mut covered) = (0usize, 0usize);
    for (p, n, k) in list {
        let b = register(gpu, p, n, k)?;
        if b > 0 {
            added += b;
            covered += n * k * 2;
        }
    }
    Ok((added, covered))
}

/// 2026-10-03: The bytes a decode GEMV on `[n, k]` weight `ptr` actually streams, for the
/// L2 prefetch plan (`METRALE_GLM_DECODE_L2_PREFETCH`): the FP8 copy (`n * k` bytes) when
/// one is registered, else the BF16 matrix.
pub fn decode_span(ptr: DevicePtr, n: usize, k: usize) -> crate::glm5next_layer::L2Span {
    match lookup(ptr, n, k) {
        Some(w) => crate::glm5next_layer::L2Span {
            ptr: w.weight,
            bytes: n * k,
        },
        None => crate::glm5next_layer::prefetch::bf16_matrix_span(ptr, n, k),
    }
}
