// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `METRALE_GLM_PREFILL_DENSE_W4A4=<classes>`: GLM-5.3 dense prefill GEMMs as NVFP4
//! activation x NVFP4 weight -> BF16 on the CUTLASS Sm120 block-scaled dense GEMM
//! (`metrale_gpu_runtime::cutlass::nvfp4_dense_w4a4_gemm`, CUTLASS example 79a configuration),
//! over the NVFP4 decode copy `METRALE_GLM_DENSE_NVFP4` already keeps, instead of the FP8 W8A8
//! GEMM of [`super::dense_fp8`].
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Off (unset, empty or `0`, the default), [`classes`] is empty and [`super::dense_fp8::route`]
//!   and `w8a8_fused_input` run exactly as before. Set without `METRALE_GLM_DENSE_FP8=1` it is
//!   ignored and set on a build without CUTLASS it is REFUSED, each logged once.
//! - Which calls (decided in `dense_fp8`, which owns the weight registry): a `route` call with
//!   at least [`MIN_ROWS`] rows, the BF16-out GEMV handle, a registered weight whose class
//!   (`dense_fp8::Class`) the lever lists and which has an NVFP4 copy. The lever never makes an
//!   NVFP4 copy: a listed class without one keeps FP8 (logged at load, [`log_load`]).
//! - Every caller of `route` passes the BF16 activation (audited 2026-10-08: KDA `gemm`, DSA
//!   `proj_gemm::gemm`, MLP `forward::launch::gemm`, MTP head `eh_proj` at <= 16 rows). The
//!   pre-quantized FP8 inputs are W8A8-arm details: `METRALE_GLM_DENSE_FP8_W8A8_SHARE_QUANT`
//!   still hands `route` the BF16 input, and the KDA `o_proj` fused norm + FP8 quant
//!   (`METRALE_GLM_NORM_FP8_QUANT_FUSE`) declines a call this lever takes, so the unfused norm
//!   writes the BF16 input and `route` runs W4A4 (logged once, [`note_fused_bypass`]).
//! - Activation global scale [`ACT_GLOBAL_SCALE`] is a constant, so each row's NVFP4 codes and
//!   scale bytes depend only on that row: an output row's bits do not depend on M or on how
//!   prefill splits rows across calls, and K >= [`CHUNK_MIN_K`] runs in [`CHUNK_ROWS`]-row
//!   launches without changing them.
//! - Weight scales are swizzled per call into the shared CUTLASS workspace (no resident
//!   memory). A call the CUTLASS entry refuses before writing its output (shape, alignment,
//!   workspace, `can_implement`) returns `false` and keeps the FP8 path: counted ([`fallbacks`])
//!   and logged once per `[n, k]` with `PREFILL_DENSE_W4A4 FALLBACK`.
//! - Workspace: shared with the other CUTLASS wrappers and written in stream order on the
//!   caller's stream; ASSUMED (as for `dense_fp8_gw`), no other CUTLASS call runs concurrently
//!   on another stream. Capture-safe (no allocation, host copy or sync).
//! - NOT byte-identical to the FP8 path. GPU gate: `examples/glm_dense_w4a4_prefill_microtest.rs`.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::cutlass::{self, DenseW4a4Args, DenseW4a4Outcome};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::weight_map::QuantizedWeight;

use super::dense_fp8::{Nvfp4Classes, parse_nvfp4_classes};

/// 2026-10-08: The lever's environment variable.
pub const LEVER: &str = "METRALE_GLM_PREFILL_DENSE_W4A4";

/// 2026-10-08: Static NVFP4 activation global scale. Constant on purpose: a per-call amax
/// would make a row's codes depend on the other rows of its call, and prefix-cache restores
/// split prefill at varying M. With 1.0 the UE4M3 block scale amax16 / 6 is exact-range for
/// 16-value maxima of about 0.09 .. 2688 (UE4M3 normal 2^-6 .. 448); smaller blocks get a
/// subnormal (coarser) scale, larger ones saturate. PROVISIONAL: not calibrated per class.
pub const ACT_GLOBAL_SCALE: f32 = 1.0;

/// 2026-10-08: Fewest rows a call takes W4A4 with; narrower calls keep their FP8 path.
/// PROVISIONAL: not swept (the 128-row tile wastes most of its MMAs on short calls).
pub const MIN_ROWS: usize = 256;

/// 2026-10-08: K from which a call runs in [`CHUNK_ROWS`]-row launches: the
/// `dense_fp8_gw` policy (G0 2026-10-06), shared so both CUTLASS prefill paths chunk alike.
pub const CHUNK_MIN_K: usize = super::dense_fp8_gw::GW_CHUNK_MIN_K;

/// 2026-10-08: Rows per launch when K >= [`CHUNK_MIN_K`] (see [`CHUNK_MIN_K`]).
pub const CHUNK_ROWS: usize = super::dense_fp8_gw::GW_CHUNK_ROWS;

/// 2026-10-08: W4A4 GEMMs issued ([`gemms`]).
static GEMMS: AtomicU64 = AtomicU64::new(0);
/// 2026-10-08: Declined calls ([`fallbacks`]).
static FALLBACKS: AtomicU64 = AtomicU64::new(0);
/// 2026-10-08: `[n, k]` shapes whose first fallback was logged.
static FALLBACK_SHAPES: Mutex<BTreeSet<(usize, usize)>> = Mutex::new(BTreeSet::new());

/// 2026-10-08: The raw lever value (trimmed), read once; empty when unset.
pub fn raw() -> &'static str {
    static R: OnceLock<String> = OnceLock::new();
    R.get_or_init(|| std::env::var(LEVER).unwrap_or_default().trim().to_string())
}

/// 2026-10-08: The lever parsed as `METRALE_GLM_DENSE_NVFP4` classes; an unknown class is an
/// error (the load fails on it, [`log_load`]).
pub fn requested() -> Result<Nvfp4Classes> {
    parse_nvfp4_classes(Some(raw()))
}

/// 2026-10-08: The classes the lever engages: [`requested`] when `METRALE_GLM_DENSE_FP8=1` and
/// the build has CUTLASS, else none (logged once). Read once.
pub fn classes() -> Nvfp4Classes {
    static C: OnceLock<Nvfp4Classes> = OnceLock::new();
    *C.get_or_init(|| {
        let c = match requested() {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("{LEVER}={}: {e:#}", raw());
                return Nvfp4Classes::default();
            }
        };
        if !c.any() {
            return c;
        }
        if !cutlass::available() {
            tracing::error!(
                "{LEVER}={} REFUSED: this build has no CUTLASS objects (CUTLASS_HOME unset at \
                 build time) - dense prefill GEMMs keep the FP8 path",
                raw()
            );
            return Nvfp4Classes::default();
        }
        if !super::dense_fp8::dense_fp8() {
            tracing::warn!(
                "{LEVER}={} is ignored without METRALE_GLM_DENSE_FP8=1 (it runs prefill over the \
                 NVFP4 copies METRALE_GLM_DENSE_NVFP4 adds to the FP8 dense weights)",
                raw()
            );
            return Nvfp4Classes::default();
        }
        c
    })
}

/// 2026-10-08: Rows per CUTLASS launch for a `k`-deep call: [`CHUNK_ROWS`] when
/// `k >= CHUNK_MIN_K`, else 0 (one launch).
pub fn rows_per_launch(k: usize) -> usize {
    if k >= CHUNK_MIN_K { CHUNK_ROWS } else { 0 }
}

/// 2026-10-08: W4A4 GEMMs issued (host-side; graph replays not counted).
pub fn gemms() -> u64 {
    GEMMS.load(Ordering::Relaxed)
}

/// 2026-10-08: Calls [`gemm`] declined (each kept the FP8 path).
pub fn fallbacks() -> u64 {
    FALLBACKS.load(Ordering::Relaxed)
}

/// 2026-10-08: `c[m, n]` (BF16, row stride `n`) = W4A4 of the BF16 `a[m, k]` (row stride `k`)
/// and the NVFP4 copy `q` (`dense_fp8::quantize_nvfp4_copy` layout) on `stream`, with the
/// weight scales swizzled per call. `Ok(false)`: refused before writing `c` (counted, logged
/// once per shape); the caller runs its FP8 path. `m == 0` is `Ok(true)` without launching.
pub fn gemm(
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    gemm_with_sfb(a, q, DevicePtr(0), c, m, n, k, stream)
}

/// 2026-10-08: [`gemm`] with an already swizzled SFB (`sfb`, from
/// `cutlass::nvfp4_dense_w4a4_pack_sfb`; null = swizzle per call, what the engine does). The
/// microtest uses it to time the per-call swizzle; the output is the same either way.
#[allow(clippy::too_many_arguments)]
pub fn gemm_with_sfb(
    a: DevicePtr,
    q: &QuantizedWeight,
    sfb: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    if m == 0 {
        return Ok(true);
    }
    let fits = |v: usize| u32::try_from(v).is_ok_and(|x| x <= i32::MAX as u32);
    let outcome = if fits(m) && fits(n) && fits(k) {
        cutlass::nvfp4_dense_w4a4_gemm(
            &DenseW4a4Args {
                act: a.0,
                w_packed: q.weight.0,
                w_scale: q.weight_scale.0,
                w_sfb: sfb.0,
                alpha: q.weight_scale_2 * ACT_GLOBAL_SCALE,
                act_gs: ACT_GLOBAL_SCALE,
                out: c.0,
                m: m as u32,
                n: n as u32,
                k: k as u32,
                rows_per_launch: rows_per_launch(k) as u32,
            },
            stream,
        )?
    } else {
        DenseW4a4Outcome::Declined(-1)
    };
    if let DenseW4a4Outcome::Declined(status) = outcome {
        FALLBACKS.fetch_add(1, Ordering::Relaxed);
        if FALLBACK_SHAPES.lock().unwrap().insert((n, k)) {
            tracing::warn!(
                "PREFILL_DENSE_W4A4 FALLBACK: first {m}-row GEMM on [{n}, {k}] kept the FP8 path \
                 (CUTLASS status {status}: -1 shape, -2 workspace, -3 can_implement, -4 \
                 alignment, -5 scale, -6 launch before output)"
            );
        }
        return Ok(false);
    }
    if GEMMS.fetch_add(1, Ordering::Relaxed) == 0 {
        tracing::warn!(
            "{LEVER}={}: ENGAGED - first NVFP4 W4A4 prefill GEMM ({m} rows, {n}x{k}): classes \
             {:?} with an NVFP4 copy run CUTLASS Sm120 NVFP4 x NVFP4 -> BF16 from {MIN_ROWS} rows \
             (static activation global scale {ACT_GLOBAL_SCALE}, weight scales swizzled per \
             call, K >= {CHUNK_MIN_K} in {CHUNK_ROWS}-row launches) instead of FP8; NOT \
             byte-identical",
            raw(),
            classes()
        );
    }
    Ok(true)
}

/// 2026-10-08: Logged once: the KDA `o_proj` fused norm + FP8 quant
/// (`METRALE_GLM_NORM_FP8_QUANT_FUSE`) declined a call this lever takes.
pub fn note_fused_bypass(m: usize, n: usize, k: usize) {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        tracing::warn!(
            "{LEVER}: the fused norm + FP8 quant (METRALE_GLM_NORM_FP8_QUANT_FUSE, KDA o_proj) is \
             skipped for W4A4 calls (first: {m} rows, {n}x{k}); the unfused norm writes the BF16 \
             input and the GEMM runs W4A4"
        );
    });
}

/// 2026-10-08: At load (from `dense_fp8::finish_load`, after the text layers and again after
/// the MTP block): fail on a bad lever value, and with the lever engaged log how many
/// registered weights per class it will run W4A4 (`engaged`) and which listed classes have
/// registered weights but no NVFP4 copy, so keep FP8 (`no_copy`).
pub fn log_load(engaged: &[(&str, usize)], no_copy: &[&str]) -> Result<()> {
    requested()?;
    if !classes().any() {
        return Ok(());
    }
    tracing::warn!(
        "{LEVER}={}: armed at load - W4A4 prefill weights per class {engaged:?} (>= {MIN_ROWS} \
         rows; ENGAGED is logged at the first GEMM)",
        raw()
    );
    if !no_copy.is_empty() {
        tracing::warn!(
            "{LEVER}={}: listed classes {no_copy:?} have no NVFP4 copy (add them to \
             METRALE_GLM_DENSE_NVFP4); their prefill keeps the FP8 path",
            raw()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-08: K 16384 (DSA o_absorb) runs in 2048-row launches; shallower K in one.
    #[test]
    fn prefill_dense_w4a4_chunks_only_k16384() {
        assert_eq!(rows_per_launch(16384), 2048);
        assert_eq!(rows_per_launch(32768), 2048);
        for k in [1024, 1536, 4096, 6144] {
            assert_eq!(rows_per_launch(k), 0, "{k}");
        }
    }

    /// 2026-10-08: The lever reads the `METRALE_GLM_DENSE_NVFP4` class syntax.
    #[test]
    fn prefill_dense_w4a4_lever_uses_the_nvfp4_class_parser() {
        let all = parse_nvfp4_classes(Some("kda,dsa,shared,mlp,mtp")).unwrap();
        assert!(all.kda && all.dsa && all.shared && all.mlp && all.mtp);
        assert!(!parse_nvfp4_classes(Some("")).unwrap().any());
        assert!(!parse_nvfp4_classes(Some("0")).unwrap().any());
        assert!(parse_nvfp4_classes(Some("kda,bogus")).is_err());
        assert_eq!(ACT_GLOBAL_SCALE, 1.0);
        assert_eq!(MIN_ROWS, 256);
    }
}
