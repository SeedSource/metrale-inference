// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Gate for the GLM-5.3 FP8 dense-weight decode lever
//! (`METRALE_GLM_DENSE_FP8=1`): `dense_gemv_fp8w_batchm` and its FP32-output twin.
//!
//! Owner: model-arch examples.
//! Checks:
//! - (a) Bitwise: every row of `dense_gemv_fp8w_batchm` (M = 1..16, one block row and a
//!   two-way row split) equals `dense_gemv_fp8w` on that row (raw u16 bits), and the
//!   FP32-output twin equals itself at M = 1 (raw u32 bits) and rounds to the BF16 bits.
//! - (b) Numerics against an FP64 host reference over the BF16 weights, per output row:
//!   cosine, max_rel = max|y - ref| / max|ref|, nonfinite count; the same for the BF16
//!   kernel as the incumbent's floor, and the FP8 kernel against FP64 over its own
//!   dequantized weights (kernel arithmetic alone). Tolerances in `TOL_*` below.
//! - (c) Timing: CUDA-graph replay over a cold pool of distinct weights (> 1 GiB of BF16,
//!   so the FP8 pool is > 512 MiB, both far past the 24 MB L2): BF16 incumbent
//!   (`dense_gemv_bf16` at 1 row, `dense_gemv_bf16_batchm` above), FP8 (`dense_gemv_fp8w`
//!   at 1 row, `dense_gemv_fp8w_batchm` above), and the in-tree
//!   `fp8_gemv_rowscale_batch16_rt2` for reference. Activations stay hot, as in decode.
//!
//! - (d) 2026-10-03 phase 2: `dequant_fp8_rowscale_bf16` (the BF16 copy wide GEMMs read once
//!   the BF16 originals are freed) bitwise against a CPU reference (exact E4M3 decode, one f32
//!   multiply by the row scale, round-to-nearest-even to BF16; raw u16 bits) on every shape
//!   above, a zero row, all 254 non-NaN codes x 64 random scales, N = 1 / K = 16, plus an
//!   untouched guard band after the output.
//! - (e) The prefill GEMM (cuBLASLt `cublas_bf16_proj_dense`, the wide-M path) over the
//!   dequantized weight against the same GEMM over the BF16 original, per output row
//!   (cosine, max_rel, nonfinite; tolerance, not exactness), and against the FP8 batchm GEMV
//!   on the first 16 rows (decode/prefill consistency).
//! - (f) `glm5next_layer::dense_fp8` end to end with the lever on: `convert_weight` frees the
//!   BF16 original (alloc-ledger delta), `finish_load` sizes the arena, `route` runs the FP8
//!   GEMV at <= 16 rows (bits == direct launch) and hands wide GEMMs the dequant (bits ==
//!   CPU reference), the eager cache skips repeats and re-dequantizes after an overlapping
//!   layer or a stream change, a captured wide call replays correctly and turns the cache
//!   off, and misuse (interior pointer, wrong shape) is an error.
//! - (g) Timing: the dequant over a cold pool per shape, and the M = 256 cuBLASLt GEMM it
//!   feeds, summed over the converted weights of one rank (one full layer pass).
//!
//! Prints `PASS` iff (a), (b), (d), (e) and (f) pass; timing is reported, never gated here.
//!
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_dense_fp8_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::Fp8DenseWeight;

/// 2026-10-03: Per-rank GLM-5.3 decode GEMV shapes at TP2 (hidden 4096, 64 KDA heads x 128,
/// 64 MLA heads x kv_lora 512, q_lora 1536, dense inter 12288, shared expert 2048) as
/// (N, K, label, launches per decode step per rank) for the weights the lever converts
/// (`glm5next_layer/dense_fp8.rs`). The last row is an N % 4 != 0 edge case, not a GLM shape.
const SHAPES: &[(usize, usize, &str, usize)] = &[
    (4096, 4096, "kda_qkvo", 136),
    (128, 4096, "kda_g_a", 34),
    (4096, 128, "kda_g_b", 34),
    (1536, 4096, "dsa_q_a", 11),
    (16384, 1536, "dsa_q_absorb", 11),
    (512, 4096, "dsa_kv_a", 11),
    (4096, 16384, "dsa_o_absorb", 11),
    (6144, 4096, "dense_gate_up", 6),
    (4096, 6144, "dense_down", 3),
    (1024, 4096, "shexp_gate_up", 84),
    (4096, 1024, "shexp_down", 42),
    // 2026-10-03: 4100 was meant as the N % 4 != 0 case but 4100 % 4 == 0, so the partial
    // last block (n >= N outputs masked) was never run; 4097 and 37 are N % 4 == 1.
    (4097, 1024, "N%4=1", 0),
    (37, 48, "N37_K48", 0),
];
const BIT_MS: &[usize] = &[1, 2, 3, 4, 5, 8, 12, 15, 16];
const TIME_MS: &[usize] = &[1, 3, 4, 8, 12, 16];
const NUM_M: usize = 4;
/// 2026-10-03: FP8 E4M3 keeps 3 mantissa bits: each weight carries a relative error up to
/// 2^-4, about 2^-4/sqrt(3) RMS, so a long random dot product keeps ~3.6 % relative RMS
/// error and cos ~ 0.9994. 0.998 leaves room for heavy-tailed rows; anything a bug
/// produces (wrong scale, wrong element order, dropped vector) is far below it.
const TOL_COS_FP8: f64 = 0.998;
const TOL_MAXREL_FP8: f64 = 0.08;
/// 2026-10-03: The kernel against FP64 over its OWN dequantized weights: FP32
/// accumulation and the BF16 output rounding only.
const TOL_COS_SELF: f64 = 0.99999;
const TOL_MAXREL_SELF: f64 = 0.01;
const POOL_BYTES_BF16: usize = 1 << 30;
/// 2026-10-03: (e) rows; 64 and 256 (the race prefill sub-chunk) take the cuBLASLt arm.
const GEMM_MS: &[usize] = &[64, 256];
/// 2026-10-03: (e) dequant-GEMM against the FP8 GEMV on the same rows: the BF16 rounding of
/// each scaled weight (relative <= 2^-9) and the accumulation order only.
const TOL_COS_DQ_GEMV: f64 = 0.9999;
const TOL_MAXREL_DQ_GEMV: f64 = 0.02;

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Approximately N(0, 1) (Irwin-Hall of 4).
    fn n(&mut self) -> f64 {
        (self.u() + self.u() + self.u() + self.u() - 2.0) * 1.7320508
    }
}

/// 2026-10-03: Weight generator. `heavy`: N(0, 0.02) with per-row gain spread
/// (x0.25..x4) and 0.1 % outliers at x30, the shape of trained projection rows whose max
/// sits far above the bulk (the per-row FP8 scale is set by that max).
fn gen_weight(rng: &mut Lcg, n: usize, k: usize, heavy: bool) -> Vec<bf16> {
    let mut w = Vec::with_capacity(n * k);
    for _ in 0..n {
        let gain = if heavy {
            0.25 * 16f64.powf(rng.u())
        } else {
            1.0
        };
        for _ in 0..k {
            let mut v = rng.n() * 0.02 * gain;
            if heavy && rng.u() < 0.001 {
                v *= 30.0;
            }
            w.push(bf16::from_f64(v));
        }
    }
    w
}

fn gen_act(rng: &mut Lcg, len: usize) -> Vec<bf16> {
    (0..len).map(|_| bf16::from_f64(rng.n())).collect()
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}
fn dn_u32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-03: OCP E4M3 decode (bias 7, subnormal m * 2^-9; 0x7F/0xFF are NaN).
fn e4m3(b: u8) -> f64 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f64;
    if e == 15 && (b & 7) == 7 {
        return f64::NAN;
    }
    if e == 0 {
        s * m * 2f64.powi(-9)
    } else {
        s * (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

struct Kern {
    q: KernelHandle,
    f1: KernelHandle,
    fbm: KernelHandle,
    fbm32: KernelHandle,
    b1: KernelHandle,
    bbm: KernelHandle,
    rt2: Option<KernelHandle>,
    dq: KernelHandle,
}

fn quantize(
    g: &dyn GpuBackend,
    k: &Kern,
    w: DevicePtr,
    n: usize,
    kd: usize,
) -> Result<Fp8DenseWeight> {
    let dw = metrale_model_layers::weight_map::DenseWeight { weight: w };
    metrale_model_layers::weight_map::quantize_to_fp8(&dw, n, kd, g, k.q, g.default_stream())
}

#[allow(clippy::too_many_arguments)]
fn fp8_rows_m1(
    g: &dyn GpuBackend,
    k: &Kern,
    a: DevicePtr,
    w: &Fp8DenseWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    kd: usize,
    s: u64,
) -> Result<()> {
    for t in 0..m {
        ops::dense_gemv_fp8w(
            g,
            k.f1,
            a.offset(t * kd * 2),
            w,
            c.offset(t * n * 2),
            n as u32,
            kd as u32,
            s,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bf16_rows(
    g: &dyn GpuBackend,
    k: &Kern,
    a: DevicePtr,
    w: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kd: usize,
    s: u64,
) -> Result<()> {
    let dw = metrale_model_layers::weight_map::DenseWeight { weight: w };
    if m == 1 {
        ops::dense_gemv(g, k.b1, a, &dw, c, n as u32, kd as u32, s)
    } else {
        ops::dense_gemv_batchm(
            g, k.bbm, a, &dw, c, m as u32, n as u32, kd as u32, n as u32, s,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn fp8_rows(
    g: &dyn GpuBackend,
    k: &Kern,
    a: DevicePtr,
    w: &Fp8DenseWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    kd: usize,
    s: u64,
) -> Result<()> {
    if m == 1 {
        ops::dense_gemv_fp8w(g, k.f1, a, w, c, n as u32, kd as u32, s)
    } else {
        ops::dense_gemv_fp8w_batchm(
            g, k.fbm, a, w, c, m as u32, 1, n as u32, kd as u32, n as u32, s,
        )
    }
}

fn row_stats(y: &[f64], r: &[f64]) -> (f64, f64, usize) {
    let (mut dot, mut yy, mut rr, mut maxd, mut maxr) = (0.0, 0.0, 0.0, 0.0f64, 0.0f64);
    let mut nonfinite = 0;
    for (&a, &b) in y.iter().zip(r) {
        if !a.is_finite() {
            nonfinite += 1;
            continue;
        }
        dot += a * b;
        yy += a * a;
        rr += b * b;
        maxd = maxd.max((a - b).abs());
        maxr = maxr.max(b.abs());
    }
    let cos = if yy > 0.0 && rr > 0.0 {
        dot / (yy.sqrt() * rr.sqrt())
    } else {
        1.0
    };
    (cos, maxd / maxr.max(1e-30), nonfinite)
}

/// (a) Bitwise checks for one shape. Returns the number of failing cases.
fn bitwise(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
) -> Result<usize> {
    let s = g.default_stream();
    let w = gen_weight(rng, n, kd, true);
    let wd = up_bf16(g, &w)?;
    let q = quantize(g, k, wd, n, kd)?;
    let mut fails = 0;
    let maxm = *BIT_MS.iter().max().unwrap();
    let a = gen_act(rng, maxm * kd);
    let ad = up_bf16(g, &a)?;
    let c_ref = g.alloc(maxm * n * 2)?;
    let c_bm = g.alloc(maxm * n * 2)?;
    let c_y = g.alloc(maxm * n * 2)?;
    let c32 = g.alloc(maxm * n * 4)?;
    let c32_1 = g.alloc(maxm * n * 4)?;
    for &m in BIT_MS {
        g.memset(c_bm, 0xAB, maxm * n * 2)?;
        g.memset(c_y, 0xCD, maxm * n * 2)?;
        fp8_rows_m1(g, k, ad, &q, c_ref, m, n, kd, s)?;
        ops::dense_gemv_fp8w_batchm(
            g, k.fbm, ad, &q, c_bm, m as u32, 1, n as u32, kd as u32, n as u32, s,
        )?;
        let y = if m >= 2 { 2 } else { 1 };
        ops::dense_gemv_fp8w_batchm(
            g, k.fbm, ad, &q, c_y, m as u32, y, n as u32, kd as u32, n as u32, s,
        )?;
        ops::dense_gemv_fp8w_batchm(
            g, k.fbm32, ad, &q, c32, m as u32, 1, n as u32, kd as u32, n as u32, s,
        )?;
        for t in 0..m {
            ops::dense_gemv_fp8w_batchm(
                g,
                k.fbm32,
                ad.offset(t * kd * 2),
                &q,
                c32_1.offset(t * n * 4),
                1,
                1,
                n as u32,
                kd as u32,
                n as u32,
                s,
            )?;
        }
        g.synchronize(s)?;
        let r = dn_u16(g, c_ref, m * n)?;
        let b = dn_u16(g, c_bm, m * n)?;
        let yb = dn_u16(g, c_y, m * n)?;
        let f = dn_u32(g, c32, m * n)?;
        let f1 = dn_u32(g, c32_1, m * n)?;
        let d_bm = r.iter().zip(&b).filter(|(x, y)| x != y).count();
        let d_y = r.iter().zip(&yb).filter(|(x, y)| x != y).count();
        let d_32 = f.iter().zip(&f1).filter(|(x, y)| x != y).count();
        let d_rnd = f
            .iter()
            .zip(&r)
            .filter(|(x, y)| bf16::from_f32(f32::from_bits(**x)).to_bits() != **y)
            .count();
        let ok = d_bm == 0 && d_y == 0 && d_32 == 0 && d_rnd == 0;
        if !ok {
            fails += 1;
        }
        println!(
            "BITWISE {label:<12} M={m:<2} batchm-vs-1row diff={d_bm} ysplit({y}) diff={d_y} \
             fp32out-vs-M1 diff={d_32} fp32out->bf16 diff={d_rnd}  {}",
            if ok { "ok" } else { "MISMATCH" }
        );
    }
    for p in [ad, c_ref, c_bm, c_y, c32, c32_1, wd, q.weight, q.row_scale] {
        g.free(p)?;
    }
    Ok(fails)
}

/// (b) Numerics for one shape at NUM_M rows. Returns the number of failing checks.
fn numerics(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
    heavy: bool,
) -> Result<usize> {
    let s = g.default_stream();
    let m = NUM_M;
    let w = gen_weight(rng, n, kd, heavy);
    let wd = up_bf16(g, &w)?;
    let q = quantize(g, k, wd, n, kd)?;
    let a = gen_act(rng, m * kd);
    let ad = up_bf16(g, &a)?;
    let c8 = g.alloc(m * n * 2)?;
    let cb = g.alloc(m * n * 2)?;
    fp8_rows(g, k, ad, &q, c8, m, n, kd, s)?;
    bf16_rows(g, k, ad, wd, cb, m, n, kd, s)?;
    g.synchronize(s)?;
    let y8: Vec<f64> = dn_u16(g, c8, m * n)?
        .iter()
        .map(|&b| bf16::from_bits(b).to_f64())
        .collect();
    let yb: Vec<f64> = dn_u16(g, cb, m * n)?
        .iter()
        .map(|&b| bf16::from_bits(b).to_f64())
        .collect();
    let qb = dn_bytes(g, q.weight, n * kd)?;
    let qs: Vec<f64> = dn_u32(g, q.row_scale, n)?
        .iter()
        .map(|&b| f32::from_bits(b) as f64)
        .collect();
    let wf: Vec<f64> = w.iter().map(|x| x.to_f64()).collect();
    let af: Vec<f64> = a.iter().map(|x| x.to_f64()).collect();
    let mut nan_codes = 0usize;
    let wq: Vec<f64> = qb
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let v = e4m3(b);
            if v.is_nan() {
                nan_codes += 1;
            }
            v * qs[i / kd]
        })
        .collect();
    let mut fails = 0;
    let (mut c8min, mut r8max, mut cbmin, mut rbmax, mut csmin, mut rsmax) =
        (1.0f64, 0.0f64, 1.0f64, 0.0f64, 1.0f64, 0.0f64);
    let mut nf = 0;
    for t in 0..m {
        let mut r = vec![0.0f64; n];
        let mut rq = vec![0.0f64; n];
        for j in 0..n {
            let (mut s1, mut s2) = (0.0, 0.0);
            for i in 0..kd {
                s1 += af[t * kd + i] * wf[j * kd + i];
                s2 += af[t * kd + i] * wq[j * kd + i];
            }
            r[j] = s1;
            rq[j] = s2;
        }
        let (c, mr, nf8) = row_stats(&y8[t * n..(t + 1) * n], &r);
        let (cb, mrb, nfb) = row_stats(&yb[t * n..(t + 1) * n], &r);
        let (cs, mrs, _) = row_stats(&y8[t * n..(t + 1) * n], &rq);
        nf += nf8 + nfb;
        c8min = c8min.min(c);
        r8max = r8max.max(mr);
        cbmin = cbmin.min(cb);
        rbmax = rbmax.max(mrb);
        csmin = csmin.min(cs);
        rsmax = rsmax.max(mrs);
    }
    let ok = c8min >= TOL_COS_FP8
        && r8max <= TOL_MAXREL_FP8
        && csmin >= TOL_COS_SELF
        && rsmax <= TOL_MAXREL_SELF
        && nf == 0
        && nan_codes == 0;
    if !ok {
        fails += 1;
    }
    println!(
        "NUMERICS {label:<12} {} M={m} fp8-vs-fp64(bf16 W): min_cos={c8min:.6} max_rel={r8max:.4} | \
         bf16-kernel-vs-fp64: min_cos={cbmin:.8} max_rel={rbmax:.5} | fp8-kernel-vs-fp64(own W): \
         min_cos={csmin:.8} max_rel={rsmax:.5} | nonfinite={nf} nan_codes={nan_codes}  {}",
        if heavy { "heavy" } else { "gauss" },
        if ok { "ok" } else { "OUT-OF-TOL" }
    );
    for p in [ad, c8, cb, wd, q.weight, q.row_scale] {
        g.free(p)?;
    }
    Ok(fails)
}

/// Replay a captured graph `reps` times; returns ms per graph.
fn time_graph(g: &dyn GpuBackend, s: u64, f: &mut dyn FnMut(u64) -> Result<()>) -> Result<f64> {
    g.begin_capture(s)?;
    if let Err(e) = f(s) {
        g.abort_capture_if_active(s);
        return Err(e);
    }
    let graph = g.end_capture(s)?;
    for _ in 0..2 {
        g.launch_graph(graph, s)?;
    }
    g.synchronize(s)?;
    let reps = 5;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        g.launch_graph(graph, s)?;
    }
    g.synchronize(s)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    g.destroy_graph(graph)?;
    Ok(ms)
}

/// (c) Timing for one shape over a cold pool.
fn timing(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
) -> Result<Vec<(usize, f64, f64)>> {
    let s = g.create_stream()?;
    let bytes = n * kd * 2;
    let pool = POOL_BYTES_BF16.div_ceil(bytes).clamp(4, 1024);
    // One random matrix, copied into every pool slot (contents do not affect timing).
    let w = gen_weight(rng, n, kd, false);
    let mut wb: Vec<DevicePtr> = Vec::with_capacity(pool);
    let mut wq: Vec<Fp8DenseWeight> = Vec::with_capacity(pool);
    let host: Vec<u8> = w.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    for _ in 0..pool {
        let p = g.alloc(bytes)?;
        g.copy_h2d(&host, p)?;
        wq.push(quantize(g, k, p, n, kd)?);
        wb.push(p);
    }
    let maxm = 16;
    let a = gen_act(rng, maxm * kd);
    let ad = up_bf16(g, &a)?;
    let c = g.alloc(maxm * n * 4)?;
    let mut out = Vec::new();
    for &m in TIME_MS {
        let t_bf = time_graph(g, s, &mut |s| {
            for p in &wb {
                bf16_rows(g, k, ad, *p, c, m, n, kd, s)?;
            }
            Ok(())
        })? / pool as f64;
        let t_f8 = time_graph(g, s, &mut |s| {
            for q in &wq {
                fp8_rows(g, k, ad, q, c, m, n, kd, s)?;
            }
            Ok(())
        })? / pool as f64;
        let t_bm1 = if m == 1 {
            Some(
                time_graph(g, s, &mut |s| {
                    for q in &wq {
                        ops::dense_gemv_fp8w_batchm(
                            g, k.fbm, ad, q, c, 1, 1, n as u32, kd as u32, n as u32, s,
                        )?;
                    }
                    Ok(())
                })? / pool as f64,
            )
        } else {
            None
        };
        let t_rt = match k.rt2 {
            Some(h) => Some(
                time_graph(g, s, &mut |s| {
                    for q in &wq {
                        ops::fp8_gemv_rowscale_batch16_rt2(
                            g, h, ad, q, c, m as u32, n as u32, kd as u32, s,
                        )?;
                    }
                    Ok(())
                })? / pool as f64,
            ),
            None => None,
        };
        let gbs = |bytes: usize, ms: f64| bytes as f64 / (ms * 1e-3) / 1e9;
        println!(
            "TIMING {label:<12} M={m:<2} bf16 {:>8.1} us {:>6.1} GB/s | fp8 {:>8.1} us {:>6.1} GB/s | \
             speedup {:>5.2}x{}{}  (pool {pool} x {:.1} MB)",
            t_bf * 1e3,
            gbs(bytes, t_bf),
            t_f8 * 1e3,
            gbs(bytes / 2, t_f8),
            t_bf / t_f8,
            t_bm1
                .map(|t| format!(" | fp8_batchm@M1 {:.1} us", t * 1e3))
                .unwrap_or_default(),
            t_rt.map(|t| format!(" | rt2 {:.1} us {:.2}x", t * 1e3, t_bf / t))
                .unwrap_or_default(),
            bytes as f64 / 1e6,
        );
        out.push((m, t_bf, t_f8));
    }
    for p in wb {
        g.free(p)?;
    }
    for q in wq {
        g.free(q.weight)?;
        g.free(q.row_scale)?;
    }
    g.free(ad)?;
    g.free(c)?;
    Ok(out)
}

// ---------------------------------------------------------------------------------------
// 2026-10-03 phase 2: dequant, prefill GEMM over the dequant, dense_fp8 end to end, timing.
// ---------------------------------------------------------------------------------------

/// 2026-10-03: CPU reference of `dequant_fp8_rowscale_bf16`: exact E4M3 decode (as f32), one
/// f32 multiply by the row scale (IEEE round to nearest even), RNE to BF16. Raw u16 bits.
fn cpu_dequant(q: &[u8], sc: &[f32], kd: usize) -> Vec<u16> {
    q.iter()
        .enumerate()
        .map(|(i, &b)| bf16::from_f32(e4m3(b) as f32 * sc[i / kd]).to_bits())
        .collect()
}

fn up_bytes(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

const GUARD: usize = 256;

/// 2026-10-03: Run the dequant of `q` (`[n, kd]`) into a guarded buffer; returns
/// (bit mismatches vs the CPU reference, guard bytes changed).
fn dequant_check(
    g: &dyn GpuBackend,
    k: &Kern,
    q: &Fp8DenseWeight,
    n: usize,
    kd: usize,
) -> Result<(usize, usize)> {
    let s = g.default_stream();
    let out = g.alloc(n * kd * 2 + GUARD)?;
    g.memset(out, 0xEE, n * kd * 2 + GUARD)?;
    ops::dequant_fp8_rowscale_bf16(g, k.dq, q, out, n as u32, kd as u32, s)?;
    g.synchronize(s)?;
    let got = dn_bytes(g, out, n * kd * 2 + GUARD)?;
    let qb = dn_bytes(g, q.weight, n * kd)?;
    let sc: Vec<f32> = dn_u32(g, q.row_scale, n)?
        .iter()
        .map(|&b| f32::from_bits(b))
        .collect();
    let want = cpu_dequant(&qb, &sc, kd);
    let diff = want
        .iter()
        .enumerate()
        .filter(|(i, w)| u16::from_le_bytes([got[2 * i], got[2 * i + 1]]) != **w)
        .count();
    let guard = got[n * kd * 2..].iter().filter(|&&b| b != 0xEE).count();
    g.free(out)?;
    Ok((diff, guard))
}

/// (d) Dequant bitwise for one shape (weights through the GPU quantizer, row 0 all zero).
fn dequant_bitwise(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
) -> Result<usize> {
    let mut w = gen_weight(rng, n, kd, true);
    if n > 1 {
        for x in &mut w[..kd] {
            *x = bf16::ZERO;
        }
    }
    let wd = up_bf16(g, &w)?;
    let q = quantize(g, k, wd, n, kd)?;
    let (diff, guard) = dequant_check(g, k, &q, n, kd)?;
    let ok = diff == 0 && guard == 0;
    println!(
        "DEQUANT {label:<12} [{n}x{kd}] bits-vs-cpu diff={diff} guard-bytes-touched={guard}  {}",
        if ok { "ok" } else { "MISMATCH" }
    );
    for p in [wd, q.weight, q.row_scale] {
        g.free(p)?;
    }
    Ok(usize::from(!ok))
}

/// (d) Every non-NaN E4M3 code under 64 random row scales (log-uniform 1e-7..10), and the
/// one-vector case N = 1, K = 16.
fn dequant_allcodes(g: &dyn GpuBackend, k: &Kern, rng: &mut Lcg) -> Result<usize> {
    let mut fails = 0;
    for &(n, kd) in &[(64usize, 256usize), (1, 16)] {
        let q: Vec<u8> = (0..n * kd)
            .map(|i| {
                let b = ((i % kd) + (i / kd) * 37) as u8;
                if b & 0x7F == 0x7F { b - 1 } else { b }
            })
            .collect();
        let sc: Vec<f32> = (0..n)
            .map(|_| 10f64.powf(-7.0 + 8.0 * rng.u()) as f32)
            .collect();
        let sb: Vec<u8> = sc.iter().flat_map(|x| x.to_le_bytes()).collect();
        let w = Fp8DenseWeight {
            weight: up_bytes(g, &q)?,
            row_scale: up_bytes(g, &sb)?,
        };
        let (diff, guard) = dequant_check(g, k, &w, n, kd)?;
        let ok = diff == 0 && guard == 0;
        if !ok {
            fails += 1;
        }
        println!(
            "DEQUANT allcodes     [{n}x{kd}] 254 codes x random scales bits-vs-cpu diff={diff} \
             guard-bytes-touched={guard}  {}",
            if ok { "ok" } else { "MISMATCH" }
        );
        g.free(w.weight)?;
        g.free(w.row_scale)?;
    }
    Ok(fails)
}

/// (e) Prefill GEMM over the dequant vs over the BF16 original, and vs the FP8 GEMV.
fn prefill_gemm(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
) -> Result<usize> {
    let s = g.default_stream();
    let w = gen_weight(rng, n, kd, true);
    let wd = up_bf16(g, &w)?;
    let q = quantize(g, k, wd, n, kd)?;
    let deq = g.alloc(n * kd * 2)?;
    ops::dequant_fp8_rowscale_bf16(g, k.dq, &q, deq, n as u32, kd as u32, s)?;
    let mut fails = 0;
    for &m in GEMM_MS {
        let a = gen_act(rng, m * kd);
        let ad = up_bf16(g, &a)?;
        let c_o = g.alloc(m * n * 2)?;
        let c_d = g.alloc(m * n * 2)?;
        let c_v = g.alloc(16 * n * 2)?;
        ops::cublas_bf16_proj_dense(ad, wd, c_o, m as u32, n as u32, kd as u32, s)?;
        ops::cublas_bf16_proj_dense(ad, deq, c_d, m as u32, n as u32, kd as u32, s)?;
        ops::dense_gemv_fp8w_batchm(
            g, k.fbm, ad, &q, c_v, 16, 1, n as u32, kd as u32, n as u32, s,
        )?;
        g.synchronize(s)?;
        let f = |p: DevicePtr, len: usize| -> Result<Vec<f64>> {
            Ok(dn_u16(g, p, len)?
                .iter()
                .map(|&b| bf16::from_bits(b).to_f64())
                .collect())
        };
        let yo = f(c_o, m * n)?;
        let yd = f(c_d, m * n)?;
        let yv = f(c_v, 16 * n)?;
        let (mut cmin, mut rmax, mut nf) = (1.0f64, 0.0f64, 0usize);
        for t in 0..m {
            let (c, r, x) = row_stats(&yd[t * n..(t + 1) * n], &yo[t * n..(t + 1) * n]);
            cmin = cmin.min(c);
            rmax = rmax.max(r);
            nf += x;
        }
        let (mut vmin, mut vmax) = (1.0f64, 0.0f64);
        for t in 0..16 {
            let (c, r, x) = row_stats(&yd[t * n..(t + 1) * n], &yv[t * n..(t + 1) * n]);
            vmin = vmin.min(c);
            vmax = vmax.max(r);
            nf += x;
        }
        let nf_o = yo.iter().filter(|x| !x.is_finite()).count();
        let ok = cmin >= TOL_COS_FP8
            && rmax <= TOL_MAXREL_FP8
            && vmin >= TOL_COS_DQ_GEMV
            && vmax <= TOL_MAXREL_DQ_GEMV
            && nf == 0
            && nf_o == 0;
        if !ok {
            fails += 1;
        }
        println!(
            "PREFILL {label:<12} M={m:<3} cublas(dequant W) vs cublas(bf16 W): min_cos={cmin:.6} \
             max_rel={rmax:.4} | vs fp8 batchm GEMV (rows 0..16): min_cos={vmin:.7} \
             max_rel={vmax:.5} | nonfinite={nf}+{nf_o}  {}",
            if ok { "ok" } else { "OUT-OF-TOL" }
        );
        for p in [ad, c_o, c_d, c_v] {
            g.free(p)?;
        }
    }
    for p in [wd, deq, q.weight, q.row_scale] {
        g.free(p)?;
    }
    Ok(fails)
}

/// (f) `glm5next_layer::dense_fp8` end to end (lever on). Returns the number of failures.
fn route_e2e(g: &dyn GpuBackend, k: &Kern, rng: &mut Lcg) -> Result<usize> {
    use metrale_model_arch::glm5next_layer::dense_fp8::{self as df, LayerFp8, Route};
    let s = g.default_stream();
    let mut fails = 0usize;
    let mut check = |ok: bool, what: &str| {
        if !ok {
            fails += 1;
        }
        println!("ROUTE {what}  {}", if ok { "ok" } else { "FAIL-CHECK" });
    };
    // Layer A: two weights; layer B: one weight overlapping both of A's arena ranges.
    let shapes = [(512usize, 1024usize), (1024, 512), (2048, 1024)];
    let mut ptrs = Vec::new();
    for &(n, kd) in &shapes {
        ptrs.push(up_bf16(g, &gen_weight(rng, n, kd, true))?);
    }
    let live0 = g.live_bytes().unwrap_or(0) as i64;
    let mut la = LayerFp8::default();
    let mut lb = LayerFp8::default();
    let mut conv = true;
    conv &= df::convert_weight(g, &mut ptrs[0], shapes[0].0, shapes[0].1, &mut la)?;
    conv &= df::convert_weight(g, &mut ptrs[1], shapes[1].0, shapes[1].1, &mut la)?;
    conv &= df::convert_weight(g, &mut ptrs[2], shapes[2].0, shapes[2].1, &mut lb)?;
    let live1 = g.live_bytes().unwrap_or(0) as i64;
    let want_delta: i64 = shapes
        .iter()
        .map(|&(n, kd)| (n * kd + 4 * n) as i64 - (2 * n * kd) as i64)
        .sum();
    check(
        conv && live1 - live0 == want_delta,
        &format!(
            "convert_weight: 3 weights converted, alloc-ledger delta {} B (want {want_delta} B = \
             +fp8+scales -bf16)",
            live1 - live0
        ),
    );
    let arena = df::finish_load(g)?;
    let want_arena = la.arena_bytes.max(lb.arena_bytes);
    let live2 = g.live_bytes().unwrap_or(0) as i64;
    check(
        arena == want_arena && live2 - live1 == arena as i64,
        &format!(
            "finish_load: arena {arena} B (want {want_arena} B), ledger +{} B",
            live2 - live1
        ),
    );
    // Host-side reference of each weight's dequant, from the registered FP8 copies.
    let mut refs = Vec::new();
    for (i, &(n, kd)) in shapes.iter().enumerate() {
        let w = df::lookup(ptrs[i], n, kd).expect("registered");
        let qb = dn_bytes(g, w.weight, n * kd)?;
        let sc: Vec<f32> = dn_u32(g, w.row_scale, n)?
            .iter()
            .map(|&b| f32::from_bits(b))
            .collect();
        refs.push((w, cpu_dequant(&qb, &sc, kd)));
    }
    let maxk = 1024;
    let a = gen_act(rng, 64 * maxk);
    let ad = up_bf16(g, &a)?;
    let c1 = g.alloc(64 * 2048 * 2)?;
    let c2 = g.alloc(64 * 2048 * 2)?;
    // <= 16 rows: the FP8 GEMV, bits equal to a direct launch.
    let (n0, k0) = shapes[0];
    let r = df::route(g, k.b1, ad, ptrs[0], c1, 4, n0, k0, s)?;
    ops::dense_gemv_fp8w_batchm(
        g, k.fbm, ad, &refs[0].0, c2, 4, 1, n0 as u32, k0 as u32, n0 as u32, s,
    )?;
    g.synchronize(s)?;
    check(
        r == Route::Done && dn_u16(g, c1, 4 * n0)? == dn_u16(g, c2, 4 * n0)?,
        "route M=4: FP8 batchm GEMV, bits == direct launch",
    );
    let r1 = df::route(g, k.b1, ad, ptrs[0], c1, 1, n0, k0, s)?;
    check(r1 == Route::Done, "route M=1: FP8 GEMV");
    check(
        df::route(g, k.b1, ad, ptrs[0], c1, 0, n0, k0, s)? == Route::Done,
        "route M=0: nothing launched",
    );
    // Wide: the dequant, its bits, and the cache.
    let bits_at = |p: DevicePtr, i: usize| -> Result<bool> {
        g.synchronize(s)?;
        let (n, kd) = shapes[i];
        Ok(dn_u16(g, p, n * kd)? == refs[i].1)
    };
    let wide = |i: usize, st: u64| -> Result<(DevicePtr, u64)> {
        let d0 = df::dequants();
        let (n, kd) = shapes[i];
        match df::route(g, k.b1, ad, ptrs[i], c1, 64, n, kd, st)? {
            Route::Weight(p) => Ok((p, df::dequants() - d0)),
            Route::Done => anyhow::bail!("wide route returned Done"),
        }
    };
    let (pa, d) = wide(0, s)?;
    check(
        d == 1 && bits_at(pa, 0)?,
        "route M=64 A0: dequant into the arena, bits == CPU ref",
    );
    let (pa2, d) = wide(0, s)?;
    check(
        d == 0 && pa2 == pa,
        "route M=64 A0 again: cache hit, no launch",
    );
    let (pb, d) = wide(1, s)?;
    check(
        d == 1 && bits_at(pb, 1)? && pb != pa,
        "route M=64 A1: own arena range, bits ok",
    );
    let (_, d) = wide(0, s)?;
    check(d == 0, "route M=64 A0 after A1: still cached (no overlap)");
    // Exercise the cuBLASLt arm on the routed weight, as the wrappers do.
    ops::cublas_bf16_proj_dense(ad, pa, c2, 64, n0 as u32, k0 as u32, s)?;
    let (pc, d) = wide(2, s)?;
    check(
        d == 1 && bits_at(pc, 2)? && pc == pa,
        "route M=64 B0: offset 0 of the arena, bits ok",
    );
    let (pa3, d) = wide(0, s)?;
    check(
        d == 1 && pa3 == pa && bits_at(pa, 0)?,
        "route M=64 A0 after B0: evicted, dequantized again, bits ok",
    );
    let s2 = g.create_stream()?;
    let (pa4, d) = wide(0, s2)?;
    check(
        d == 1 && pa4 == pa && bits_at(pa, 0)?,
        "route M=64 A0 on a second stream: redone",
    );
    // Misuse is an error, an unregistered pointer passes through.
    let inner = df::route(g, k.b1, ad, ptrs[0].offset(16), c1, 64, n0, k0, s).is_err();
    let shape = df::route(g, k.b1, ad, ptrs[0], c1, 64, n0 + 1, k0, s).is_err();
    let other = g.alloc(64)?;
    let pass = df::route(g, k.b1, ad, other, c1, 64, n0, k0, s)? == Route::Weight(other);
    check(
        inner && shape && pass,
        "route: interior pointer err, wrong shape err, unregistered passes",
    );
    g.free(other)?;
    // Captured wide call: replays the dequant; afterwards the eager cache is off.
    g.synchronize(s)?;
    g.memset(pb, 0, shapes[1].0 * shapes[1].1 * 2)?;
    g.begin_capture(s2)?;
    let cap = df::route(g, k.b1, ad, ptrs[1], c1, 64, shapes[1].0, shapes[1].1, s2);
    let graph = g.end_capture(s2)?;
    let cap_ok = matches!(cap, Ok(Route::Weight(p)) if p == pb);
    let zero_before = dn_u16(g, pb, 8)?.iter().all(|&x| x == 0);
    g.launch_graph(graph, s2)?;
    g.synchronize(s2)?;
    let replay_ok = dn_u16(g, pb, shapes[1].0 * shapes[1].1)? == refs[1].1;
    g.destroy_graph(graph)?;
    check(
        cap_ok && zero_before && replay_ok,
        "route M=64 under capture: captured (not run at capture), replay writes the dequant",
    );
    let (_, d1) = wide(0, s2)?;
    let (_, d2) = wide(0, s2)?;
    check(
        d1 == 1 && d2 == 1 && bits_at(pa, 0)?,
        "after a capture: eager cache off (every call dequantizes)",
    );
    println!(
        "ROUTE stats: fp8 GEMV launches {} dequant launches {}",
        df::hits(),
        df::dequants()
    );
    for p in [ad, c1, c2] {
        g.free(p)?;
    }
    Ok(fails)
}

/// (g) Dequant timing over a cold pool, and the M = 256 cuBLASLt GEMM on the BF16 weight.
/// Returns (dequant ms, gemm ms) per weight.
fn dequant_timing(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
) -> Result<(f64, f64)> {
    let s = g.create_stream()?;
    let bytes = n * kd * 2;
    let pool = POOL_BYTES_BF16.div_ceil(bytes).clamp(4, 256);
    let w = gen_weight(rng, n, kd, false);
    let host: Vec<u8> = w.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let mut wb = Vec::with_capacity(pool);
    let mut wq = Vec::with_capacity(pool);
    for _ in 0..pool {
        let p = g.alloc(bytes)?;
        g.copy_h2d(&host, p)?;
        wq.push(quantize(g, k, p, n, kd)?);
        wb.push(p);
    }
    let out = g.alloc(bytes)?;
    let t_dq = time_graph(g, s, &mut |s| {
        for q in &wq {
            ops::dequant_fp8_rowscale_bf16(g, k.dq, q, out, n as u32, kd as u32, s)?;
        }
        Ok(())
    })? / pool as f64;
    let m = 256;
    let a = gen_act(rng, m * kd);
    let ad = up_bf16(g, &a)?;
    let c = g.alloc(m * n * 2)?;
    // cuBLASLt picks its algorithm on the first call; warm it outside the capture.
    ops::cublas_bf16_proj_dense(ad, wb[0], c, m as u32, n as u32, kd as u32, s)?;
    g.synchronize(s)?;
    let t_mm = time_graph(g, s, &mut |s| {
        for p in &wb {
            ops::cublas_bf16_proj_dense(ad, *p, c, m as u32, n as u32, kd as u32, s)?;
        }
        Ok(())
    })? / pool as f64;
    let gbs = |b: usize, ms: f64| b as f64 / (ms * 1e-3) / 1e9;
    println!(
        "TIMING-DQ {label:<12} dequant {:>8.1} us {:>6.1} GB/s (fp8 read + bf16 write) | cublas \
         M=256 bf16 {:>8.1} us | dequant/gemm {:.2}  (pool {pool} x {:.1} MB)",
        t_dq * 1e3,
        gbs(bytes / 2 * 3, t_dq),
        t_mm * 1e3,
        t_dq / t_mm,
        bytes as f64 / 1e6
    );
    for p in wb {
        g.free(p)?;
    }
    for q in wq {
        g.free(q.weight)?;
        g.free(q.row_scale)?;
    }
    for p in [out, ad, c] {
        g.free(p)?;
    }
    Ok((t_dq, t_mm))
}

fn main() -> Result<()> {
    // SAFETY: single-threaded here, before anything reads the environment. Section (f) needs
    // the lever on; (a)-(e) and (g) call the kernels directly and do not read it.
    unsafe { std::env::set_var("METRALE_GLM_DENSE_FP8", "1") };
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k = Kern {
        q: g.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?,
        f1: g.kernel("gemv_fp8w", "dense_gemv_fp8w")?,
        fbm: g.kernel("dense_gemv_fp8w_batchm", "dense_gemv_fp8w_batchm")?,
        fbm32: g.kernel("dense_gemv_fp8w_batchm", "dense_gemv_fp8w_fp32out_batchm")?,
        b1: g.kernel("gemv", "dense_gemv_bf16")?,
        bbm: g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
        rt2: g
            .kernel("fp8_gemv_rt", "fp8_gemv_rowscale_batch16_rt2")
            .ok(),
        dq: g.kernel("dequant_fp8_rowscale_bf16", "dequant_fp8_rowscale_bf16")?,
    };
    let mut rng = Lcg(0x6c6d_3533_f8f8);

    let mut bit_fail = 0;
    for &(n, kd, label, _) in SHAPES {
        bit_fail += bitwise(g, &k, &mut rng, n, kd, label)?;
    }
    let mut num_fail = 0;
    for &(n, kd, label, _) in SHAPES {
        for heavy in [false, true] {
            num_fail += numerics(g, &k, &mut rng, n, kd, label, heavy)?;
        }
    }
    let mut dq_fail = 0;
    for &(n, kd, label, _) in SHAPES {
        dq_fail += dequant_bitwise(g, &k, &mut rng, n, kd, label)?;
    }
    dq_fail += dequant_allcodes(g, &k, &mut rng)?;
    let mut pf_fail = 0;
    for &(n, kd, label, per_step) in SHAPES {
        if per_step > 0 {
            pf_fail += prefill_gemm(g, &k, &mut rng, n, kd, label)?;
        }
    }
    let (mut dq_pass, mut mm_pass, mut fp8_b, mut bf16_b) = (0.0f64, 0.0f64, 0usize, 0usize);
    for &(n, kd, label, per_step) in SHAPES {
        if per_step == 0 {
            continue;
        }
        let (tdq, tmm) = dequant_timing(g, &k, &mut rng, n, kd, label)?;
        dq_pass += tdq * per_step as f64;
        mm_pass += tmm * per_step as f64;
        fp8_b += (n * kd + 4 * n) * per_step;
        bf16_b += 2 * n * kd * per_step;
    }
    println!(
        "TIMING-DQ SUMMARY one pass over every converted weight of a rank (launch counts above = \
         weights per rank): dequant {dq_pass:.2} ms; cuBLASLt M=256 GEMMs over the same weights \
         {mm_pass:.2} ms. Per 8192-token prefill chunk at 256-row sub-chunks (32 sub-chunks, 1 \
         pass per chunk with the eager cache): +{dq_pass:.2} ms dequant vs {:.1} ms of these \
         GEMMs ({:.2}%); without the cache it would be +{:.1} ms ({:.1}%)",
        mm_pass * 32.0,
        100.0 * dq_pass / (mm_pass * 32.0),
        dq_pass * 32.0,
        100.0 * dq_pass / mm_pass
    );
    println!(
        "MEMORY per rank (shapes x weights per rank above): BF16 originals freed {:.3} GB, FP8 \
         copies + row scales {:.3} GB, net {:.3} GB before the dequant arena",
        bf16_b as f64 / 1e9,
        fp8_b as f64 / 1e9,
        (fp8_b as f64 - bf16_b as f64) / 1e9
    );
    let rt_fail = route_e2e(g, &k, &mut rng)?;
    let mut summary = Vec::new();
    for &(n, kd, label, per_step) in SHAPES {
        if n % 4 != 0 {
            continue;
        }
        for (m, tb, tf) in timing(g, &k, &mut rng, n, kd, label)? {
            summary.push((label, m, tb, tf, per_step));
        }
    }
    for &m in TIME_MS {
        let rows: Vec<_> = summary.iter().filter(|x| x.1 == m).collect();
        let (lo, hi) = rows.iter().fold((f64::MAX, 0.0f64), |a, x| {
            (a.0.min(x.2 / x.3), a.1.max(x.2 / x.3))
        });
        let bf: f64 = rows.iter().map(|x| x.2 * x.4 as f64).sum();
        let f8: f64 = rows.iter().map(|x| x.3 * x.4 as f64).sum();
        println!(
            "TIMING SUMMARY M={m:<2} fp8/bf16 speedup over shapes: min {lo:.2}x max {hi:.2}x | \
             per-step sum over converted GEMVs (launch counts per rank): bf16 {bf:.2} ms -> fp8 \
             {f8:.2} ms, saves {:.2} ms ({:.2}x)",
            bf - f8,
            bf / f8
        );
    }
    println!(
        "tolerances: fp8 vs fp64(bf16 W) cos>={TOL_COS_FP8} max_rel<={TOL_MAXREL_FP8}; \
         fp8 kernel vs fp64(own W) cos>={TOL_COS_SELF} max_rel<={TOL_MAXREL_SELF}; nonfinite=0"
    );
    ensure!(
        bit_fail == 0,
        "bitwise: {bit_fail} shape/M cases differ (see MISMATCH lines)"
    );
    ensure!(
        num_fail == 0,
        "numerics: {num_fail} shape cases out of tolerance (see OUT-OF-TOL lines)"
    );
    ensure!(
        dq_fail == 0,
        "dequant: {dq_fail} cases differ from the CPU reference (see MISMATCH lines)"
    );
    ensure!(
        pf_fail == 0,
        "prefill GEMM over the dequant: {pf_fail} cases out of tolerance (see OUT-OF-TOL lines)"
    );
    ensure!(
        rt_fail == 0,
        "dense_fp8 route/arena: {rt_fail} checks failed (see FAIL-CHECK lines)"
    );
    println!(
        "PASS: dense_gemv_fp8w_batchm bitwise == dense_gemv_fp8w per row (M 1..16, y-split, fp32out, partial last block); numerics within tolerance; dequant_fp8_rowscale_bf16 bitwise == CPU reference; prefill GEMM over the dequant within tolerance; dense_fp8 route/arena/cache checks pass"
    );
    Ok(())
}
