// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `METRALE_GLM_DENSE_FP8=1`: FP8 E4M3 weight-only copies of the GLM-5.3
//! non-expert dense projections, the GEMV dispatch that reads them, and the BF16 dequant the
//! wider GEMMs read instead of the (freed) BF16 originals.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Off (the default), nothing is quantized, freed or allocated, and [`route`] returns
//!   `Route::Weight(b)` for the caller's own `b` without launching, so every projection runs
//!   exactly as before.
//! - On, the loader ([`register_layer`]) quantizes, per text layer (0..num_hidden_layers;
//!   the MTP block is loaded elsewhere and never registered), the BF16 weights listed in
//!   [`register_layer`] to `[N, K]` E4M3 + one FP32 scale per output row
//!   (`quantize_bf16_to_fp8`, max|row| / 448), FREES the BF16 original, and rewrites the
//!   layer's weight field to the FP8 copy's device pointer, which is the registry key. A
//!   field that names a registered pointer therefore holds FP8 bytes: every read of it must
//!   go through [`route`] (the three GLM `gemm` wrappers: KDA `Glm5NextKdaLayer::gemm`, DSA
//!   `proj_gemm::gemm`, MLP `forward::launch::gemm`; audited 2026-10-03, no other reader).
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
//! - Weights deliberately left BF16: the MoE router (expert selection), the DSA indexer
//!   (`wk`, `compress_gate`, `weights_proj`, `wq_b`: top-k token selection), the KDA
//!   recurrence gates `f_a`, `f_b` (forget-gate decay) and `b_proj` (beta), `embed_tokens`,
//!   `lm_head` (its FP8 option is `--lm-head-dtype fp8`), the mHC `hc_fn` (not a GEMV), the
//!   MTP block and any DFlash drafter. Together they are < 0.4 GB/rank of decode reads.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use anyhow::{Result, bail};
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
                "METRALE_GLM_DENSE_FP8=1 - GLM-5.3 non-expert dense projections keep only FP8 \
                 E4M3 copies (per-row scale; BF16 originals freed): <= 16-row GEMVs read FP8, \
                 wider GEMMs a BF16 dequant of it; NOT byte-identical to BF16"
            );
        }
        on
    })
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
static HITS: AtomicU64 = AtomicU64::new(0);
static DEQUANTS: AtomicU64 = AtomicU64::new(0);
static FIRST_HIT: AtomicBool = AtomicBool::new(false);
static FIRST_DEQUANT: AtomicBool = AtomicBool::new(false);
static HANDLE_MISMATCH: AtomicBool = AtomicBool::new(false);

const ARENA_ALIGN: usize = 256;

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

/// 2026-10-03: Quantize one BF16 `[n, k]` weight, free the BF16 original and register the
/// FP8 copy with arena offset `off`. Returns the FP8 copy (its `weight` is the new key),
/// or `None` with nothing changed (lever off, kernels missing, empty, or `k % 16 != 0`).
fn register(
    gpu: &dyn GpuBackend,
    bf16: DevicePtr,
    n: usize,
    k: usize,
    off: usize,
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
    let w = metrale_model_layers::weight_map::quantize_to_fp8(
        &DenseWeight { weight: bf16 },
        n,
        k,
        gpu,
        kk.quant,
        s,
    )?;
    // `quantize_to_fp8` synchronized `s`; nothing reads the original after this.
    gpu.free(bf16)?;
    map()
        .write()
        .unwrap()
        .insert(w.weight.0, Entry { w, n, k, off });
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

/// 2026-10-03: The FP8 copy registered under key `ptr` with shape `[n, k]`, if any.
pub fn lookup(ptr: DevicePtr, n: usize, k: usize) -> Option<Fp8DenseWeight> {
    if !dense_fp8() {
        return None;
    }
    let m = map().read().unwrap();
    m.get(&ptr.0).filter(|e| e.n == n && e.k == k).map(|e| e.w)
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
    if m <= ops::DENSE_GEMV_FP8W_BATCHM_MAX_M as usize {
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
    let p = dequant(gpu, &kk, b.0, &e, m, stream)?;
    Ok(Route::Weight(p))
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
        ops::dequant_fp8_rowscale_bf16(gpu, kk.dequant, &e.w, dst, e.n as u32, e.k as u32, s)
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
    let mut list: Vec<(&mut DevicePtr, usize, usize)> = Vec::new();
    match mixer {
        Glm5NextMixer::Kda { layer, cfg, .. } => {
            let w = &mut layer.weights;
            let (hid, qkv, hd) = (cfg.hidden, cfg.qkv_dim(), cfg.head_dim);
            list.push((&mut w.q_proj.weight, qkv, hid));
            list.push((&mut w.k_proj.weight, qkv, hid));
            list.push((&mut w.v_proj.weight, qkv, hid));
            list.push((&mut w.g_a.weight, hd, hid));
            list.push((&mut w.g_b.weight, qkv, hd));
            list.push((&mut w.o_proj.weight, hid, qkv));
        }
        Glm5NextMixer::Dsa(l) => {
            // Shapes as `decode_k` / `decode_k_wide` pass them to `gemm`.
            let l = &mut **l;
            let (c, w) = (&l.cfg, &mut l.weights);
            let heads_lat = c.local_heads * c.kv_lora_rank;
            list.push((&mut w.q_a_proj, c.q_lora_rank, c.hidden));
            list.push((&mut w.q_absorb, heads_lat, c.q_lora_rank));
            list.push((&mut w.kv_a_proj, c.kv_lora_rank, c.hidden));
            list.push((&mut w.o_absorb, c.hidden, heads_lat));
        }
    }
    let (w, inter) = match mlp {
        Glm5NextMlpSite::Dense(w) => (w, mlp_cfg.local_dense_intermediate),
        Glm5NextMlpSite::Moe(w) => (&mut w.shared, mlp_cfg.local_shared_intermediate),
    };
    list.push((&mut w.gate_proj, inter, mlp_cfg.hidden));
    list.push((&mut w.up_proj, inter, mlp_cfg.hidden));
    list.push((&mut w.down_proj, mlp_cfg.hidden, inter));
    let mut out = LayerFp8::default();
    for (field, n, k) in list {
        convert_weight(gpu, field, n, k, &mut out)?;
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
    let off = acc.arena_bytes;
    let Some(w) = register(gpu, *field, n, k, off)? else {
        return Ok(false);
    };
    *field = w.weight;
    acc.fp8_bytes += n * k + 4 * n;
    acc.bf16_freed += n * k * 2;
    acc.arena_bytes = (off + n * k * 2).next_multiple_of(ARENA_ALIGN);
    ARENA_NEED.fetch_max(acc.arena_bytes, Ordering::Relaxed);
    Ok(true)
}

/// 2026-10-03: After every text layer is registered: allocate the dequant arena (the
/// largest per-layer span). Returns its bytes; 0, allocating nothing, when the lever is off
/// or nothing was converted. Must run before the KV pool is sized so the ledger counts it.
pub fn finish_load(gpu: &dyn GpuBackend) -> Result<usize> {
    if !dense_fp8() {
        return Ok(0);
    }
    let need = ARENA_NEED.load(Ordering::Relaxed);
    if need == 0 {
        return Ok(0);
    }
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
    match lookup(ptr, n, k) {
        Some(w) => crate::glm5next_layer::L2Span {
            ptr: w.weight,
            bytes: n * k,
        },
        None => crate::glm5next_layer::prefetch::bf16_matrix_span(ptr, n, k),
    }
}
