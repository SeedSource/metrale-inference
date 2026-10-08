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
//! - Activation global scale ([`GS_LEVER`], [`GsMode`]): `static` (the default) uses the
//!   constant [`ACT_GLOBAL_SCALE`], so each row's NVFP4 codes and scale bytes depend only on that
//!   row: an output row's bits do not depend on M or on how prefill splits rows across calls,
//!   and K >= [`CHUNK_MIN_K`] runs in [`CHUNK_ROWS`]-row launches without changing them.
//!   `dynamic` (2026-10-08) resolves gs on the device per CUTLASS launch from the amax of that
//!   launch's rows (the routed-MoE W4A4 rule): K >= [`CHUNK_MIN_K`] runs in [`CHUNK_ROWS`]-row
//!   pieces, each with its own amax, and a row's bits depend on the other rows of its piece
//!   (not row-invariant; prefix-cache restores that split prefill can change them).
//! - The first engaged call of each class logs one `PREFILL_DENSE_W4A4 BLOCKSTATS` line
//!   ([`block_stats_once`]: one synchronized D2H of that call's BF16 input, skipped while the
//!   stream is capturing).
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

use anyhow::{Result, bail};
use metrale_gpu_runtime::cutlass::{self, DenseW4a4Args, DenseW4a4Outcome};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::weight_map::QuantizedWeight;

use super::dense_fp8::{Class, Nvfp4Classes, parse_nvfp4_classes};

/// 2026-10-08: The lever's environment variable.
pub const LEVER: &str = "METRALE_GLM_PREFILL_DENSE_W4A4";

/// 2026-10-08: The activation global-scale lever: `static` (default) or `dynamic` ([`GsMode`]).
pub const GS_LEVER: &str = "METRALE_GLM_PREFILL_DENSE_W4A4_GS";

/// 2026-10-08: The NVFP4 activation global scale of `static` mode. Constant on purpose: a
/// per-call amax would make a row's codes depend on the other rows of its call, and
/// prefix-cache restores split prefill at varying M. With 1.0 the UE4M3 block scale amax16 / 6
/// is exact-range for 16-value maxima of about 0.09 .. 2688 (UE4M3 normal 2^-6 .. 448); smaller
/// blocks get a subnormal (coarser) scale, larger ones saturate. PROVISIONAL: not calibrated
/// per class.
pub const ACT_GLOBAL_SCALE: f32 = 1.0;

/// 2026-10-08: How a W4A4 call picks its NVFP4 activation global scale gs ([`GS_LEVER`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GsMode {
    /// [`ACT_GLOBAL_SCALE`] for every call: row-invariant.
    Static,
    /// Per CUTLASS launch, on the device: [`gs_from_amax`] of the amax over that launch's rows
    /// (the routed-MoE W4A4 `act_amax_*` / `resolve_act_gs` rule, the same device helpers).
    /// K >= [`CHUNK_MIN_K`] launches [`CHUNK_ROWS`]-row pieces, each with its own amax. Not
    /// row-invariant by design.
    Dynamic,
}

impl GsMode {
    /// 2026-10-08: The mode's name in the lever syntax and the log lines.
    pub fn name(self) -> &'static str {
        match self {
            GsMode::Static => "static",
            GsMode::Dynamic => "dynamic",
        }
    }
}

/// 2026-10-08: Parse a [`GS_LEVER`] value: unset, empty or `static` = [`GsMode::Static`];
/// `dynamic` = [`GsMode::Dynamic`]; anything else is an error.
pub fn parse_gs_mode(raw: Option<&str>) -> Result<GsMode> {
    match raw.map(str::trim).unwrap_or("") {
        "" | "static" => Ok(GsMode::Static),
        "dynamic" => Ok(GsMode::Dynamic),
        other => bail!("{GS_LEVER}={other}: expected static or dynamic"),
    }
}

/// 2026-10-08: The [`GS_LEVER`] mode, read once; a bad value logs an error and runs `static`
/// (the load fails on it first, [`log_load`]).
pub fn gs_mode() -> GsMode {
    static M: OnceLock<GsMode> = OnceLock::new();
    *M.get_or_init(|| {
        parse_gs_mode(std::env::var(GS_LEVER).ok().as_deref()).unwrap_or_else(|e| {
            tracing::error!("{e:#}");
            GsMode::Static
        })
    })
}

/// 2026-10-08: The dynamic global scale for an amax: amax / (6 * 448), 1.0 when the amax is
/// zero or not finite (host mirror of `act_gs_from_amax_bits` in
/// `cuda/cutlass_nvfp4_w4a4_quant.cuh`, for [`block_stats`]).
pub fn gs_from_amax(amax: f32) -> f32 {
    if amax > 0.0 && amax.is_finite() {
        amax / (6.0 * 448.0)
    } else {
        1.0
    }
}

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
/// 2026-10-08: Classes whose `BLOCKSTATS` line was logged ([`block_stats_once`]).
static STATS_CLASSES: Mutex<BTreeSet<&'static str>> = Mutex::new(BTreeSet::new());

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
/// weight scales swizzled per call and the [`gs_mode`] global scale. `Ok(false)`: refused
/// before writing `c` (counted, logged once per shape); the caller runs its FP8 path.
/// `m == 0` is `Ok(true)` without launching.
pub fn gemm(
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    gemm_ex(a, q, DevicePtr(0), c, m, n, k, gs_mode(), stream)
}

/// 2026-10-08: [`gemm`] with an already swizzled SFB (`sfb`, from
/// `cutlass::nvfp4_dense_w4a4_pack_sfb`; null = swizzle per call, what the engine does) and an
/// explicit [`GsMode`]. The microtest uses it to time the per-call swizzle (the output is the
/// same either way) and to run both modes in one process.
#[allow(clippy::too_many_arguments)]
pub fn gemm_ex(
    a: DevicePtr,
    q: &QuantizedWeight,
    sfb: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    mode: GsMode,
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
                // Dynamic: the entry multiplies alpha by each launch's gs on the device.
                alpha: match mode {
                    GsMode::Static => q.weight_scale_2 * ACT_GLOBAL_SCALE,
                    GsMode::Dynamic => q.weight_scale_2,
                },
                act_gs: ACT_GLOBAL_SCALE,
                dynamic_gs: mode == GsMode::Dynamic,
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
        let gs = match mode {
            GsMode::Static => format!("static activation global scale {ACT_GLOBAL_SCALE}"),
            GsMode::Dynamic => "dynamic activation global scale amax / 2688 per launch (1.0 \
                                for a zero or non-finite amax)"
                .to_string(),
        };
        tracing::warn!(
            "{LEVER}={}: ENGAGED gs={} - first NVFP4 W4A4 prefill GEMM ({m} rows, {n}x{k}): \
             classes {:?} with an NVFP4 copy run CUTLASS Sm120 NVFP4 x NVFP4 -> BF16 from \
             {MIN_ROWS} rows ({gs}, weight scales swizzled per call, K >= {CHUNK_MIN_K} in \
             {CHUNK_ROWS}-row launches) instead of FP8; NOT byte-identical",
            raw(),
            mode.name(),
            classes()
        );
    }
    Ok(true)
}

/// 2026-10-08: NVFP4 activation block statistics of one call ([`block_stats`]): counts of the
/// 16-value blocks whose UE4M3 scale is zero, subnormal or saturated under the gs in use.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BlockStats {
    /// 16-value blocks (`m * k / 16`).
    pub blocks: usize,
    /// Scale byte 0: the block quantizes to all zeros (amax16 / 6 / gs <= 2^-10).
    pub zero: usize,
    /// Subnormal UE4M3 scale (2^-10 < amax16 / 6 / gs < 7.5 * 2^-9): fewer scale bits.
    pub subnormal: usize,
    /// amax16 / 6 / gs > 464 (the midpoint of 448 and the next E4M3 step, 480): the scale
    /// needs more than UE4M3's 448, clamps there, and the block's largest values clip at
    /// 6 * 448 * gs. (Between 448 and 464 round-to-nearest gives 448 either way.)
    pub saturated: usize,
    /// Largest |value| of the call.
    pub amax: f32,
}

impl BlockStats {
    /// 2026-10-08: `count` as a fraction of [`BlockStats::blocks`] (0 for no blocks).
    pub fn frac(&self, count: usize) -> f64 {
        if self.blocks == 0 {
            0.0
        } else {
            count as f64 / self.blocks as f64
        }
    }
}

/// 2026-10-08: [`BlockStats`] of a BF16 `[m, k]` activation (`bf16_le`: little-endian bytes,
/// row stride `k`, `k % 16 == 0`) as the dense W4A4 entry quantizes it in `mode`: static gs =
/// [`ACT_GLOBAL_SCALE`]; dynamic gs = [`gs_from_amax`] per launch piece ([`rows_per_launch`]
/// rows, or all `m`). Each block's scale value is `quant16_gs`'s
/// `(amax16 / 6) * (1 / gs)`, classified by where UE4M3 round-to-nearest-even puts it.
pub fn block_stats(bf16_le: &[u8], m: usize, k: usize, mode: GsMode) -> BlockStats {
    const ZERO_MAX: f32 = 1.0 / 1024.0; // 2^-10: rounds to the 0 scale byte
    const SUBNORMAL_END: f32 = 7.5 / 512.0; // between 7 * 2^-9 and the normal 2^-6
    const SATURATED_FROM: f32 = 464.0; // between 448 and 480
    let val = |i: usize| {
        f32::from_bits(u32::from(u16::from_le_bytes([bf16_le[2 * i], bf16_le[2 * i + 1]])) << 16)
    };
    let mut st = BlockStats {
        blocks: m * (k / 16),
        ..BlockStats::default()
    };
    let step = match (mode, rows_per_launch(k)) {
        (GsMode::Dynamic, r) if r > 0 => r,
        _ => m.max(1),
    };
    let mut r0 = 0;
    while r0 < m {
        let rows = step.min(m - r0);
        let (lo, hi) = (r0 * k, (r0 + rows) * k);
        let piece_amax = (lo..hi).fold(0.0f32, |a, i| a.max(val(i).abs()));
        st.amax = st.amax.max(piece_amax);
        let gs = match mode {
            GsMode::Static => ACT_GLOBAL_SCALE,
            GsMode::Dynamic => gs_from_amax(piece_amax),
        };
        let inv_gs = 1.0 / gs;
        for b in (lo..hi).step_by(16) {
            let amax16 = (b..b + 16).fold(0.0f32, |a, i| a.max(val(i).abs()));
            let sf = (amax16 / 6.0) * inv_gs;
            if sf > SATURATED_FROM {
                st.saturated += 1;
            } else if sf <= ZERO_MAX {
                st.zero += 1;
            } else if sf < SUBNORMAL_END {
                st.subnormal += 1;
            }
        }
        r0 += rows;
    }
    st
}

/// 2026-10-08: The first engaged call of each class (after its GEMM, from `dense_fp8::route`):
/// one synchronized D2H of the BF16 input `a[m, k]` and one
/// `PREFILL_DENSE_W4A4 BLOCKSTATS <class>/<proj> rows= blocks= zero= subnormal= saturated=
/// amax=` line ([`block_stats`] under [`gs_mode`]). Skipped while `stream` is capturing (the
/// next uncaptured call of that class logs instead). Diagnostic only: a copy failure is logged
/// and the serve goes on.
#[allow(clippy::too_many_arguments)]
pub fn block_stats_once(
    gpu: &dyn GpuBackend,
    class: Class,
    proj: &str,
    a: DevicePtr,
    m: usize,
    k: usize,
    stream: u64,
) {
    if gpu.stream_is_capturing(stream) || !STATS_CLASSES.lock().unwrap().insert(class.name()) {
        return;
    }
    let mut buf = vec![0u8; m * k * 2];
    if let Err(e) = gpu.copy_d2h_on_stream(a, &mut buf, stream) {
        tracing::warn!(
            "PREFILL_DENSE_W4A4 BLOCKSTATS {}/{proj}: input copy failed: {e:#}",
            class.name()
        );
        return;
    }
    let st = block_stats(&buf, m, k, gs_mode());
    tracing::warn!(
        "PREFILL_DENSE_W4A4 BLOCKSTATS {}/{proj} rows={m} blocks={} zero={:.6} subnormal={:.6} \
         saturated={:.6} amax={}",
        class.name(),
        st.blocks,
        st.frac(st.zero),
        st.frac(st.subnormal),
        st.frac(st.saturated),
        st.amax
    );
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
/// the MTP block): fail on a bad lever or [`GS_LEVER`] value, and with the lever engaged log
/// how many registered weights per class it will run W4A4 (`engaged`, and the gs mode) and
/// which listed classes have registered weights but no NVFP4 copy, so keep FP8 (`no_copy`).
pub fn log_load(engaged: &[(&str, usize)], no_copy: &[&str]) -> Result<()> {
    requested()?;
    parse_gs_mode(std::env::var(GS_LEVER).ok().as_deref())?;
    if !classes().any() {
        return Ok(());
    }
    tracing::warn!(
        "{LEVER}={}: armed at load - W4A4 prefill weights per class {engaged:?} (>= {MIN_ROWS} \
         rows, gs={}; ENGAGED is logged at the first GEMM)",
        raw(),
        gs_mode().name()
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

    /// 2026-10-08: `METRALE_GLM_PREFILL_DENSE_W4A4_GS` takes static (default) and dynamic only;
    /// the dynamic gs is amax / 2688 with the 1.0 fallback.
    #[test]
    fn prefill_dense_w4a4_gs_mode_parses_and_falls_back() {
        assert_eq!(parse_gs_mode(None).unwrap(), GsMode::Static);
        assert_eq!(parse_gs_mode(Some("")).unwrap(), GsMode::Static);
        assert_eq!(parse_gs_mode(Some("static")).unwrap(), GsMode::Static);
        assert_eq!(parse_gs_mode(Some(" dynamic ")).unwrap(), GsMode::Dynamic);
        assert!(parse_gs_mode(Some("1")).is_err());
        assert_eq!(gs_from_amax(2688.0), 1.0);
        assert_eq!(gs_from_amax(0.0), 1.0);
        assert_eq!(gs_from_amax(f32::INFINITY), 1.0);
        assert_eq!(gs_from_amax(26.88), 26.88 / 2688.0);
    }

    /// 2026-10-08: Block classes under static and dynamic gs, including the per-piece amax of a
    /// chunked K.
    #[test]
    fn prefill_dense_w4a4_block_stats_classify_scales() {
        let bytes = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        };
        // Four blocks of one 64-wide row, static gs 1: zero, subnormal (amax16 0.0625 ->
        // 0.0104), normal (6 -> 1), saturated (5376 -> 896).
        let mut v = vec![0.0f32; 64];
        v[16] = 0.0625;
        v[32] = 6.0;
        v[48] = 5376.0;
        let st = block_stats(&bytes(&v), 1, 64, GsMode::Static);
        assert_eq!(
            (st.blocks, st.zero, st.subnormal, st.saturated),
            (4, 1, 1, 1)
        );
        assert_eq!(st.amax, 5376.0);
        // Dynamic: gs = 5376 / 2688 = 2, so the 5376 block maps to 448 (not saturated) and the
        // 0.0625 block to 0.0052 (still subnormal).
        let st = block_stats(&bytes(&v), 1, 64, GsMode::Dynamic);
        assert_eq!((st.zero, st.subnormal, st.saturated), (1, 1, 0));
        // K = 16384 in dynamic mode: each 2048-row piece has its own amax, so a small first
        // piece is not crushed by a large second one; static gs 1 zeroes the small piece.
        let k = CHUNK_MIN_K;
        let m = CHUNK_ROWS + 1;
        let v: Vec<f32> = (0..m * k)
            .map(|i| if i < CHUNK_ROWS * k { 1e-3 } else { 1e3 })
            .collect();
        let st = block_stats(&bytes(&v), m, k, GsMode::Dynamic);
        assert_eq!((st.zero, st.subnormal, st.saturated), (0, 0, 0));
        let st = block_stats(&bytes(&v), m, k, GsMode::Static);
        assert_eq!(
            (st.zero, st.subnormal, st.saturated),
            (CHUNK_ROWS * k / 16, 0, 0)
        );
    }
}
