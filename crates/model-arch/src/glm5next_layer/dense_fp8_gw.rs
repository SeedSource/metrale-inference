// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: `METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW=1`: the GLM-5.3 dense prefill W8A8 GEMMs
//! of [`super::dense_fp8`] on the CUTLASS Sm120 FP8 blockwise GEMM
//! (`metrale_gpu_runtime::cutlass::fp8_blockwise_gemm_bf16`, after CUTLASS example 87b) with
//! 1x128 activation scales and 128x128 weight block scales.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Engaged ([`cutlass_gw`]) only with `METRALE_GLM_DENSE_FP8_W8A8=1` (hence
//!   `METRALE_GLM_DENSE_FP8=1`) and a CUTLASS build; set without CUTLASS it is REFUSED at load
//!   (logged, the weights stay per-row). Off, nothing here runs and every path is unchanged.
//! - Which weights: at load, `dense_fp8::register` quantizes a weight as 128x128 block-scaled
//!   FP8 E4M3 (`quantize_bf16_to_fp8_blockscaled`, scales `[N/128, K/128]` F32, stored in the
//!   entry's `row_scale` field) INSTEAD of per-row FP8 only when it also gets an NVFP4 decode
//!   copy (`METRALE_GLM_DENSE_NVFP4` selects its class) and N, K are multiples of 128
//!   ([`eligible_for`]). Reader inventory (2026-10-06): the per-row FP8 copy is read by the
//!   1..=16-row decode GEMVs (`dense_gemv_fp8w`, `dense_gemv_fp8w_batchm`, `dense_gemv_tcm`),
//!   which have no block-scale form; a weight with an NVFP4 copy never reaches them (`route`
//!   sends its 1..=16-row BF16-out calls to the NVFP4 GEMV, and every other 1..=16-row call
//!   to the dequant). So a block-scaled weight's FP8 copy is read only by the W8A8 GEMM
//!   ([`gemm`]), the BF16 dequant of the 17..63-row / skipped calls ([`dequant_block_scaled`])
//!   and the L2 prefetch span (bytes only). No weight gets a second FP8 copy: the memory
//!   delta is the scale table only (`4 ceil(N/128) ceil(K/128)` instead of `4 N` bytes).
//!   Weights without an NVFP4 copy (DSA unless `dsa` is selected, KDA g_a/g_b, MTP unless
//!   `mtp`) keep per-row FP8 and the incumbent `fp8_gemm_rowscale_pipe_128x64`.
//! - [`gemm`] runs CUTLASS when the build has it, N and K are multiples of 128 and A, B, C are
//!   16-byte aligned; K >= [`GW_CHUNK_MIN_K`] runs in launches of at most [`GW_CHUNK_ROWS`]
//!   rows (row offsets into A, the activation scales and C); anything else runs the
//!   incumbent block-scaled `fp8_gemm_t_blockscaled` (pipelined 128x64 tile) on the same
//!   operands. NOT byte-identical to the per-row path (weights requantized to block scales,
//!   another accumulation order). GPU gate `examples/dense_fp8_cutlass_gw_microtest.rs`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::Result;
use metrale_gpu_runtime::cutlass::{self, Fp8BlockwiseSchedule};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, WeightQuantFormat};

/// 2026-10-06: Scale block edge (N and K) and activation scale group (K).
pub const GW_BLOCK: usize = 128;

/// 2026-10-06: Smallest K that runs CUTLASS in row chunks. PROVISIONAL (G0 2026-10-06, GB10,
/// cooperative): at K 16384 one 8192-row launch reached 77 TFLOP/s (losing to the incumbent)
/// while 2048 rows reached 122; not swept between K 4096 and 16384.
pub const GW_CHUNK_MIN_K: usize = 16384;

/// 2026-10-06: Rows per CUTLASS launch when K >= [`GW_CHUNK_MIN_K`]. PROVISIONAL (G0
/// 2026-10-06: 2048 rows at K 16384 = 122 TFLOP/s; other sizes not swept).
pub const GW_CHUNK_ROWS: usize = 2048;

/// 2026-10-06: A launch of at most this many rows on a wide-N, short-K weight
/// (N >= [`GW_PINGPONG_MIN_N_OVER_K`] x K) runs the pingpong schedule. PROVISIONAL (G0
/// 2026-10-06: 2048 x 16384 x 1536 pingpong 139 vs cooperative 127 TFLOP/s; the only shape
/// measured both ways).
pub const GW_PINGPONG_MAX_ROWS: usize = 2048;
/// 2026-10-06: See [`GW_PINGPONG_MAX_ROWS`].
pub const GW_PINGPONG_MIN_N_OVER_K: usize = 8;

pub(crate) fn parse_cutlass_gw(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-06: `METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW=1` was set; read once.
pub fn cutlass_gw_requested() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        parse_cutlass_gw(
            std::env::var("METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-10-06: The lever is engaged: requested, `METRALE_GLM_DENSE_FP8_W8A8=1` and a CUTLASS
/// build. Read once; a refusal or a missing prerequisite is logged once.
pub fn cutlass_gw() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        if !cutlass_gw_requested() {
            return false;
        }
        if !cutlass::available() {
            tracing::error!(
                "METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW=1 REFUSED: this build has no CUTLASS \
                 objects (CUTLASS_HOME unset at build time) - dense FP8 weights stay per-row and \
                 prefill keeps fp8_gemm_rowscale_pipe_128x64"
            );
            return false;
        }
        if !super::dense_fp8::dense_fp8_w8a8() {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW=1: NOT engaged - it needs \
                 METRALE_GLM_DENSE_FP8=1 and METRALE_GLM_DENSE_FP8_W8A8=1"
            );
            return false;
        }
        true
    })
}

/// 2026-10-06: Whether a weight is quantized block-scaled (module doc): the lever engaged, the
/// weight has an NVFP4 decode copy, and N, K are multiples of 128.
pub fn eligible_for(engaged: bool, nv4_copy: bool, n: usize, k: usize) -> bool {
    engaged
        && nv4_copy
        && n > 0
        && k > 0
        && n.is_multiple_of(GW_BLOCK)
        && k.is_multiple_of(GW_BLOCK)
}

/// 2026-10-06: The CUTLASS launches of one `[m, n, k]` GEMM: `(first row, rows, schedule)`.
pub fn plan(m: usize, n: usize, k: usize) -> Vec<(usize, usize, Fp8BlockwiseSchedule)> {
    let step = if k >= GW_CHUNK_MIN_K {
        GW_CHUNK_ROWS
    } else {
        m.max(1)
    };
    (0..m)
        .step_by(step)
        .map(|r0| {
            let rows = step.min(m - r0);
            let pp = rows <= GW_PINGPONG_MAX_ROWS && n >= GW_PINGPONG_MIN_N_OVER_K * k;
            let s = if pp {
                Fp8BlockwiseSchedule::Pingpong
            } else {
                Fp8BlockwiseSchedule::Cooperative
            };
            (r0, rows, s)
        })
        .collect()
}

/// 2026-10-06: Which kernel [`gemm`] ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GwPath {
    Cutlass,
    Incumbent,
}

static GEMMS: AtomicU64 = AtomicU64::new(0);
static FALLBACKS: AtomicU64 = AtomicU64::new(0);
static FIRST: AtomicBool = AtomicBool::new(false);
static FIRST_FALLBACK: AtomicBool = AtomicBool::new(false);

/// 2026-10-06: CUTLASS GW GEMMs [`gemm`] issued (host-side; graph replays not counted).
pub fn gw_gemms() -> u64 {
    GEMMS.load(Ordering::Relaxed)
}

/// 2026-10-06: [`gemm`] calls that ran the incumbent block-scaled kernel instead.
pub fn gw_fallbacks() -> u64 {
    FALLBACKS.load(Ordering::Relaxed)
}

fn aligned16(p: DevicePtr) -> bool {
    p.0.is_multiple_of(16)
}

/// 2026-10-06: `c[m, n]` (BF16, row stride `n`) = W8A8 of the quantized activations (`a_fp8`
/// `[m, k]` E4M3, `a_scale` `[m, k/128]` F32) and the block-scaled weight `w` (`[n, k]` E4M3,
/// `w.row_scale` = `[n/128, k/128]` F32), on `stream`: the dispatch the engine uses (module
/// doc). `m == 0` launches nothing.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    gpu: &dyn GpuBackend,
    a_fp8: DevicePtr,
    a_scale: DevicePtr,
    w: &Fp8DenseWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<GwPath> {
    if m == 0 {
        return Ok(GwPath::Cutlass);
    }
    let use_cutlass = cutlass::available()
        && cutlass::fp8_blockwise_shape_ok(n as u32, k as u32)
        && aligned16(a_fp8)
        && aligned16(w.weight)
        && aligned16(c);
    if !use_cutlass {
        let legacy = legacy_kernel(gpu)?;
        ops::fp8_gemm_t_blockscaled(
            gpu,
            legacy,
            a_fp8,
            a_scale,
            w.weight,
            w.row_scale,
            c,
            m as u32,
            n as u32,
            k as u32,
            stream,
        )?;
        FALLBACKS.fetch_add(1, Ordering::Relaxed);
        if !FIRST_FALLBACK.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW: first {m}-row GEMM ({n}x{k}) ran the \
                 incumbent fp8_gemm_t_blockscaled (CUTLASS built {}, shape ok {}, 16-byte \
                 aligned {})",
                cutlass::available(),
                cutlass::fp8_blockwise_shape_ok(n as u32, k as u32),
                aligned16(a_fp8) && aligned16(w.weight) && aligned16(c)
            );
        }
        return Ok(GwPath::Incumbent);
    }
    let groups = k / GW_BLOCK;
    for (r0, rows, schedule) in plan(m, n, k) {
        cutlass::fp8_blockwise_gemm_bf16(
            a_fp8.offset(r0 * k).0,
            a_scale.offset(r0 * groups * 4).0,
            w.weight.0,
            w.row_scale.0,
            c.offset(r0 * n * 2).0,
            rows as u32,
            n as u32,
            k as u32,
            schedule,
            stream,
        )?;
    }
    GEMMS.fetch_add(1, Ordering::Relaxed);
    if !FIRST.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW: first CUTLASS GEMM routed ({m} rows, {n}x{k}, \
             {} launch(es))",
            plan(m, n, k).len()
        );
    }
    Ok(GwPath::Cutlass)
}

/// 2026-10-06: The non-pipelined `fp8_gemm_t_blockscaled` handle `ops::fp8_gemm_t_blockscaled`
/// falls back to when its pipelined twin declines; resolved once.
fn legacy_kernel(gpu: &dyn GpuBackend) -> Result<metrale_gpu_runtime::gpu::KernelHandle> {
    static H: OnceLock<metrale_gpu_runtime::gpu::KernelHandle> = OnceLock::new();
    if let Some(h) = H.get() {
        return Ok(*h);
    }
    let h = gpu.kernel("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled")?;
    Ok(*H.get_or_init(|| h))
}

/// 2026-10-06: Quantize a BF16 `[n, k]` weight to 128x128 block-scaled FP8 E4M3
/// (`quantize_bf16_to_fp8_blockscaled`), synchronizing `stream` so the caller may free the
/// original. The block scales `[ceil(n/128), ceil(k/128)]` F32 go in `row_scale`.
pub fn quantize_block_scaled(
    gpu: &dyn GpuBackend,
    bf16: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<Fp8DenseWeight> {
    let q = gpu.kernel(
        "quantize_bf16_to_fp8_blockscaled",
        "quantize_bf16_to_fp8_blockscaled",
    )?;
    let w = metrale_model_layers::weight_map::quantize_to_fp8_blockscaled(
        &DenseWeight { weight: bf16 },
        n,
        k,
        gpu,
        q,
        stream,
    )?;
    gpu.synchronize(stream)?;
    Ok(Fp8DenseWeight {
        weight: w.weight,
        row_scale: w.row_scale,
    })
}

/// 2026-10-06: `dst` = BF16 dequant of the block-scaled weight `w` (`dequant_fp8_blockscaled_bf16`
/// with 128x128 blocks), on `stream`.
pub fn dequant_block_scaled(
    gpu: &dyn GpuBackend,
    w: &Fp8DenseWeight,
    dst: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    let fw = Fp8Weight {
        weight: w.weight,
        row_scale: w.row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    ops::dequant_fp8_bf16_into(gpu, &fw, dst, stream)
}

/// 2026-10-06: At load (from `dense_fp8::finish_load`, before the KV pool is sized): with
/// `converted` block-scaled weights, warm the shared CUTLASS workspace and log ENGAGED once;
/// with none while engaged, log NOT engaged once. Returns the workspace bytes (0 when none).
pub fn finish_load(converted: usize, scale_bytes: usize) -> Result<usize> {
    if !cutlass_gw() {
        return Ok(0);
    }
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if converted == 0 {
        if !LOGGED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW=1: NOT engaged - no converted dense weight \
                 has an NVFP4 decode copy (METRALE_GLM_DENSE_NVFP4 class) with N and K multiples \
                 of 128, so every weight keeps per-row FP8 and fp8_gemm_rowscale_pipe_128x64"
            );
        }
        return Ok(0);
    }
    let ws = cutlass::warm_workspace()?;
    if !LOGGED.swap(true, Ordering::Relaxed) {
        let k_major = cutlass::fp8_blockwise_scale_k_major().unwrap_or(true);
        tracing::warn!(
            "METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW=1: ENGAGED - {converted} dense weights (those \
             with an NVFP4 decode copy) hold 128x128 block-scaled FP8 instead of per-row \
             ({:.1} MB of block scales); their >= 64-row prefill GEMMs run the CUTLASS Sm120 FP8 \
             blockwise GEMM ({} scale layouts; K >= {GW_CHUNK_MIN_K} in {GW_CHUNK_ROWS}-row \
             launches), other weights keep per-row FP8; CUTLASS workspace {:.1} MB; NOT \
             byte-identical",
            scale_bytes as f64 / 1e6,
            if k_major { "K-major" } else { "MN-major" },
            ws as f64 / 1e6
        );
    }
    Ok(ws)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cutlass_gw_lever_parses_one_as_on_and_everything_else_as_off() {
        assert!(parse_cutlass_gw(Some("1")));
        assert!(parse_cutlass_gw(Some(" 1 ")));
        for v in [None, Some("0"), Some(""), Some("on"), Some("true")] {
            assert!(!parse_cutlass_gw(v), "{v:?}");
        }
    }

    #[test]
    fn cutlass_gw_block_scales_only_engaged_nvfp4_weights_with_128_multiples() {
        assert!(eligible_for(true, true, 4096, 16384));
        assert!(eligible_for(true, true, 1024, 4096));
        assert!(!eligible_for(false, true, 4096, 4096));
        assert!(!eligible_for(true, false, 4096, 4096));
        assert!(!eligible_for(true, true, 4160, 4096));
        assert!(!eligible_for(true, true, 4096, 4160));
        assert!(!eligible_for(true, true, 0, 4096));
    }

    #[test]
    fn cutlass_gw_plan_chunks_k16384_and_picks_pingpong_only_on_wide_short_k() {
        let p = plan(8191, 4096, 16384);
        assert_eq!(p.len(), 4);
        assert_eq!(p[0], (0, 2048, Fp8BlockwiseSchedule::Cooperative));
        assert_eq!(p[3], (6144, 2047, Fp8BlockwiseSchedule::Cooperative));
        assert_eq!(p.iter().map(|x| x.1).sum::<usize>(), 8191);
        assert_eq!(
            plan(8191, 4096, 4096),
            vec![(0, 8191, Fp8BlockwiseSchedule::Cooperative)]
        );
        assert_eq!(
            plan(2048, 16384, 1536),
            vec![(0, 2048, Fp8BlockwiseSchedule::Pingpong)]
        );
        assert_eq!(
            plan(8191, 16384, 1536),
            vec![(0, 8191, Fp8BlockwiseSchedule::Cooperative)]
        );
        assert_eq!(
            plan(2048, 4096, 1024),
            vec![(0, 2048, Fp8BlockwiseSchedule::Cooperative)]
        );
        assert_eq!(
            plan(64, 4096, 16384),
            vec![(0, 64, Fp8BlockwiseSchedule::Cooperative)]
        );
        assert!(plan(0, 4096, 4096).is_empty());
    }
}
