// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Gate 1 for the GLM-5.3 NVFP4 dense decode lever (`METRALE_GLM_DENSE_NVFP4`,
//! `glm5next_layer::dense_fp8`), against the FP8 dense copies it replaces in decode.
//!
//! Owner: model-arch examples.
//! Checks:
//! - (a) Route and bits: with the lever on, `convert_weight_nvfp4` registers a weight with both
//!   copies; `route` at M = 1..=16 on the BF16-out GEMV returns `Done` and writes rows whose raw
//!   u16 bits equal the M = 1 launch on that row (decode and verify rows agree), and the NVFP4
//!   hit counter moves; M = 17 (prefill) returns `Route::Weight` (the FP8 dequant), not `Done`.
//! - (b) Numerics against an FP64 host reference over the BF16 weights, per output row: cosine,
//!   max_rel = max|y - ref| / max|ref|, nonfinite count, for NVFP4 and for the FP8 copy (the
//!   incumbent's floor, `dense_gemv_fp8w`), plus NVFP4 against FP64 over its own dequantized
//!   weights (kernel arithmetic alone). Gaussian and heavy-tailed weights.
//! - (c) Timing: CUDA-graph replay over a cold pool of distinct weights (> 1 GiB BF16), M in
//!   `TIME_MS`: the serve's FP8 decode GEMV (`dense_gemv_tcm` under `METRALE_GLM_GEMV_TC=1`,
//!   else `dense_gemv_fp8w(_batchm)`) against `dense_fp8::nv4_gemv`; GB/s over the bytes each
//!   streams; a per-step sum over the default classes' launches per rank.
//! - (d) `parse_nvfp4_classes`: `1` = kda,shared,mlp; lists; an unknown class is an error.
//!
//! Prints `PASS` iff (a), (b) and (d) pass; timing is reported, never gated here.
//! The tolerances are PROVISIONAL (set 2026-10-05 from the format, not from a sweep); the
//! product gate is the serve's acceptance panel and TEB, not this file.
//!
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_dense_nvfp4_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_layer::dense_fp8 as df;
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight, QuantizedWeight};

/// 2026-10-05: Per-rank GLM-5.3 decode GEMV shapes at TP2 (as in
/// `glm5next_dense_fp8_microtest.rs`) as (N, K, label, launches per decode step per rank,
/// in the default `1` = kda,shared,mlp classes).
const SHAPES: &[(usize, usize, &str, usize, bool)] = &[
    (4096, 4096, "kda_qkvo", 136, true),
    (1024, 4096, "shexp_gate_up", 84, true),
    (4096, 1024, "shexp_down", 42, true),
    (6144, 4096, "dense_gate_up", 6, true),
    (4096, 6144, "dense_down", 3, true),
    (1536, 4096, "dsa_q_a", 11, false),
    (16384, 1536, "dsa_q_absorb", 11, false),
    (512, 4096, "dsa_kv_a", 11, false),
    (4096, 16384, "dsa_o_absorb", 11, false),
    // 2026-10-05: N % 4 == 1 (partial last block).
    (4097, 1024, "N%4=1", 0, false),
];
const BIT_MS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16];
const TIME_MS: &[usize] = &[1, 2, 3, 4, 8];
const NUM_M: usize = 3;
/// 2026-10-05: E2M1 keeps one mantissa bit; with an E4M3 scale per 16 values chosen by squared
/// error, the per-weight relative RMS error is about 8-10 %, so a long random dot product keeps
/// cos ~ 0.995. 0.985 leaves room for heavy-tailed rows; a layout or scale bug lands far below.
/// PROVISIONAL.
const TOL_COS_NV4: f64 = 0.985;
const TOL_MAXREL_NV4: f64 = 0.25;
const TOL_COS_SELF: f64 = 0.99999;
const TOL_MAXREL_SELF: f64 = 0.01;
const POOL_BYTES_BF16: usize = 1 << 30;

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

/// 2026-10-05: Weight generator (as the FP8 microtest): `heavy` = per-row gain x0.25..x4 and
/// 0.1 % outliers at x30.
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
fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn to_f64(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| bf16::from_bits(b).to_f64()).collect()
}

/// 2026-10-05: OCP E4M3 decode (bias 7, subnormal m * 2^-9; 0x7F/0xFF are NaN).
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

const E2M1: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 2026-10-05: The NVFP4 copy's weights as f64 (`w4a16_gemv.cu` layout: low nibble = even K).
fn nv4_dequant(g: &dyn GpuBackend, q: &QuantizedWeight, n: usize, kd: usize) -> Result<Vec<f64>> {
    let pk = dn_bytes(g, q.weight, n * kd / 2)?;
    let sc = dn_bytes(g, q.weight_scale, n * kd / 16)?;
    let s2 = q.weight_scale_2 as f64;
    let mut w = vec![0.0f64; n * kd];
    for r in 0..n {
        for i in 0..kd {
            let b = pk[(r * kd + i) / 2];
            let nib = if i.is_multiple_of(2) { b & 0xF } else { b >> 4 };
            w[r * kd + i] = E2M1[nib as usize] * e4m3(sc[r * (kd / 16) + i / 16]) * s2;
        }
    }
    Ok(w)
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

struct Kern {
    bf16_gemv: KernelHandle,
    fp8_1: KernelHandle,
    fp8_bm: KernelHandle,
}

/// 2026-10-05: The serve's FP8 decode GEMV for `m` rows: the tensor-core `dense_gemv_tcm` when
/// it routes the shape (`METRALE_GLM_GEMV_TC=1`, set in `main`), else `dense_gemv_fp8w`
/// (M = 1) / `dense_gemv_fp8w_batchm`.
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
    if ops::dense_gemv_tcm::try_fp8(g, a, w, c, m as u32, n as u32, kd as u32, n as u32, s)? {
        return Ok(());
    }
    if m == 1 {
        ops::dense_gemv_fp8w(g, k.fp8_1, a, w, c, n as u32, kd as u32, s)
    } else {
        ops::dense_gemv_fp8w_batchm(
            g, k.fp8_bm, a, w, c, m as u32, 1, n as u32, kd as u32, n as u32, s,
        )
    }
}

/// (a) + (b) for one shape: register with both copies, route bits across M, numerics.
/// Returns (bit/route failures, numerics failures).
fn shape_checks(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
    heavy: bool,
) -> Result<(usize, usize)> {
    let s = g.default_stream();
    let w = gen_weight(rng, n, kd, heavy);
    let mut key = up_bf16(g, &w)?;
    let mut acc = df::LayerFp8::default();
    ensure!(
        df::convert_weight_nvfp4(g, &mut key, n, kd, &mut acc)?,
        "{label}: convert_weight_nvfp4 did not convert"
    );
    // The prefill-width (M = 17) route below reads the dequant arena.
    df::finish_load(g)?;
    let q = df::lookup_nv4(key, n, kd)
        .ok_or_else(|| anyhow::anyhow!("{label}: no NVFP4 copy registered"))?;
    let f8 = df::lookup(key, n, kd)
        .ok_or_else(|| anyhow::anyhow!("{label}: no FP8 copy registered"))?;
    let maxm = *BIT_MS.iter().max().unwrap();
    let a = gen_act(rng, maxm.max(17) * kd);
    let ad = up_bf16(g, &a)?;
    let c = g.alloc(maxm.max(17) * n * 2)?;
    let c1 = g.alloc(maxm * n * 2)?;
    let mut bit_fail = 0usize;
    // M = 1 reference, row by row.
    for t in 0..maxm {
        let r = df::route(
            g,
            k.bf16_gemv,
            ad.offset(t * kd * 2),
            key,
            c1.offset(t * n * 2),
            1,
            n,
            kd,
            s,
        )?;
        ensure!(r == df::Route::Done, "{label}: route M=1 did not run the NVFP4 GEMV");
    }
    g.synchronize(s)?;
    let ref1 = dn_u16(g, c1, maxm * n)?;
    for &m in BIT_MS {
        let h0 = df::nv4_hits();
        let r = df::route(g, k.bf16_gemv, ad, key, c, m, n, kd, s)?;
        g.synchronize(s)?;
        let y = dn_u16(g, c, m * n)?;
        let diff = y.iter().zip(&ref1[..m * n]).filter(|(a, b)| a != b).count();
        let ok = r == df::Route::Done && df::nv4_hits() == h0 + 1 && diff == 0;
        if !ok {
            bit_fail += 1;
            println!(
                "MISMATCH {label} M={m}: route={r:?} hits+{} differing elements {diff}",
                df::nv4_hits() - h0
            );
        }
    }
    // Prefill width: not the NVFP4 GEMV.
    let h0 = df::nv4_hits();
    let r17 = df::route(g, k.bf16_gemv, ad, key, c, 17, n, kd, s)?;
    g.synchronize(s)?;
    if !matches!(r17, df::Route::Weight(_)) || df::nv4_hits() != h0 {
        bit_fail += 1;
        println!("MISMATCH {label} M=17: route={r17:?} (want Weight: prefill keeps FP8)");
    }

    // (b) numerics, NUM_M rows.
    let m = NUM_M;
    let cn = g.alloc(m * n * 2)?;
    let c8 = g.alloc(m * n * 2)?;
    df::route(g, k.bf16_gemv, ad, key, cn, m, n, kd, s)?;
    fp8_rows(g, k, ad, &f8, c8, m, n, kd, s)?;
    g.synchronize(s)?;
    let yn = to_f64(&dn_u16(g, cn, m * n)?);
    let y8 = to_f64(&dn_u16(g, c8, m * n)?);
    let wf: Vec<f64> = w.iter().map(|x| x.to_f64()).collect();
    let wq = nv4_dequant(g, &q, n, kd)?;
    let af: Vec<f64> = a.iter().map(|x| x.to_f64()).collect();
    let (mut cn_min, mut rn_max, mut c8_min, mut r8_max, mut cs_min, mut rs_max) =
        (1.0f64, 0.0f64, 1.0f64, 0.0f64, 1.0f64, 0.0f64);
    let (mut en2, mut e82) = (0.0f64, 0.0f64);
    let mut nf = 0usize;
    let nan_scales = dn_bytes(g, q.weight_scale, n * kd / 16)?
        .iter()
        .filter(|&&b| e4m3(b).is_nan())
        .count();
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
        let row = t * n..(t + 1) * n;
        let (c1_, r1_, nf1) = row_stats(&yn[row.clone()], &r);
        let (c2_, r2_, nf2) = row_stats(&y8[row.clone()], &r);
        let (c3_, r3_, _) = row_stats(&yn[row], &rq);
        nf += nf1 + nf2;
        cn_min = cn_min.min(c1_);
        rn_max = rn_max.max(r1_);
        c8_min = c8_min.min(c2_);
        r8_max = r8_max.max(r2_);
        cs_min = cs_min.min(c3_);
        rs_max = rs_max.max(r3_);
        for j in 0..n {
            en2 += (yn[t * n + j] - r[j]).powi(2);
            e82 += (y8[t * n + j] - r[j]).powi(2);
        }
    }
    let ok = cn_min >= TOL_COS_NV4
        && rn_max <= TOL_MAXREL_NV4
        && cs_min >= TOL_COS_SELF
        && rs_max <= TOL_MAXREL_SELF
        && nf == 0
        && nan_scales == 0;
    println!(
        "NUMERICS {label:<14} {} M={m} nvfp4-vs-fp64(bf16 W): min_cos={cn_min:.6} max_rel={rn_max:.4} \
         | fp8-vs-fp64: min_cos={c8_min:.6} max_rel={r8_max:.4} | rms err nvfp4/fp8 {:.2}x | \
         nvfp4-kernel-vs-fp64(own W): min_cos={cs_min:.8} max_rel={rs_max:.5} | nonfinite={nf} \
         nan_scales={nan_scales}  {}",
        if heavy { "heavy" } else { "gauss" },
        (en2 / e82.max(1e-300)).sqrt(),
        if ok { "ok" } else { "OUT-OF-TOL" }
    );
    for p in [ad, c, c1, cn, c8] {
        g.free(p)?;
    }
    // The registered copies stay (registry keys are never freed, as in a serve).
    Ok((bit_fail, usize::from(!ok)))
}

/// Replay a captured graph; returns ms per graph.
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

/// (c) Timing for one shape over a cold pool: (m, fp8 ms, nvfp4 ms) per launch.
fn timing(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
) -> Result<Vec<(usize, f64, f64)>> {
    let s = g.create_stream()?;
    let bf_bytes = n * kd * 2;
    let pool = POOL_BYTES_BF16.div_ceil(bf_bytes).clamp(4, 1024);
    let (aq, qq) = (
        g.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        g.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_mse")?,
    );
    let fq = g.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    let w = gen_weight(rng, n, kd, false);
    let host: Vec<u8> = w.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let mut f8s: Vec<Fp8DenseWeight> = Vec::with_capacity(pool);
    let mut nvs: Vec<QuantizedWeight> = Vec::with_capacity(pool);
    let tmp = g.alloc(bf_bytes)?;
    g.copy_h2d(&host, tmp)?;
    let dw = DenseWeight { weight: tmp };
    for _ in 0..pool {
        f8s.push(metrale_model_layers::weight_map::quantize_to_fp8(
            &dw,
            n,
            kd,
            g,
            fq,
            g.default_stream(),
        )?);
        nvs.push(metrale_model_layers::weight_map::quantize_to_nvfp4(
            &dw,
            n,
            kd,
            g,
            aq,
            qq,
            g.default_stream(),
        )?);
    }
    g.free(tmp)?;
    let maxm = *TIME_MS.iter().max().unwrap();
    let a = gen_act(rng, maxm * kd);
    let ad = up_bf16(g, &a)?;
    let c = g.alloc(maxm * n * 2)?;
    let mut out = Vec::new();
    let (f8_bytes, nv_bytes) = (n * kd + 4 * n, n * kd / 2 + n * kd / 16);
    for &m in TIME_MS {
        let t8 = time_graph(g, s, &mut |s| {
            for w in &f8s {
                fp8_rows(g, k, ad, w, c, m, n, kd, s)?;
            }
            Ok(())
        })? / pool as f64;
        let tn = time_graph(g, s, &mut |s| {
            for q in &nvs {
                df::nv4_gemv(g, ad, q, c, m, n, kd, s)?;
            }
            Ok(())
        })? / pool as f64;
        let gbs = |b: usize, ms: f64| b as f64 / (ms * 1e-3) / 1e9;
        println!(
            "TIMING {label:<14} M={m:<2} fp8 {:>8.1} us {:>6.1} GB/s | nvfp4 {:>8.1} us {:>6.1} GB/s | \
             speedup {:>5.2}x  (pool {pool})",
            t8 * 1e3,
            gbs(f8_bytes, t8),
            tn * 1e3,
            gbs(nv_bytes, tn),
            t8 / tn
        );
        out.push((m, t8, tn));
    }
    for w in f8s {
        g.free(w.weight)?;
        g.free(w.row_scale)?;
    }
    for q in nvs {
        g.free(q.weight)?;
        g.free(q.weight_scale)?;
    }
    g.free(ad)?;
    g.free(c)?;
    Ok(out)
}

/// (d) The class parser.
fn parse_checks() -> usize {
    let mut fails = 0;
    let mut check = |ok: bool, what: &str| {
        if !ok {
            fails += 1;
        }
        println!("PARSE {what}  {}", if ok { "ok" } else { "FAIL-CHECK" });
    };
    let p = |s: Option<&str>| df::parse_nvfp4_classes(s);
    check(p(None).map(|c| !c.any()).unwrap_or(false), "unset = off");
    check(p(Some("0")).map(|c| !c.any()).unwrap_or(false), "0 = off");
    check(
        p(Some("1"))
            .map(|c| c.kda && c.shared && c.mlp && !c.dsa && !c.mtp)
            .unwrap_or(false),
        "1 = kda,shared,mlp",
    );
    check(
        p(Some("kda, dsa"))
            .map(|c| c.kda && c.dsa && !c.shared && !c.mlp && !c.mtp)
            .unwrap_or(false),
        "list kda,dsa",
    );
    check(
        p(Some("dsa_o"))
            .map(|c| c.dsa_o && !c.dsa && !c.kda && !c.shared && !c.mlp && !c.mtp && c.any())
            .unwrap_or(false),
        "list dsa_o",
    );
    check(p(Some("kda,bogus")).is_err(), "unknown class is an error");
    fails
}

fn main() -> Result<()> {
    // SAFETY: single-threaded here, before anything reads the environment.
    unsafe {
        std::env::set_var("METRALE_GLM_DENSE_FP8", "1");
        std::env::set_var("METRALE_GLM_DENSE_NVFP4", "1");
        std::env::set_var("METRALE_GLM_GEMV_TC", "1");
    }
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    ensure!(
        df::dense_nvfp4().any(),
        "METRALE_GLM_DENSE_NVFP4=1 did not resolve to any class"
    );
    let k = Kern {
        bf16_gemv: g.kernel("gemv", "dense_gemv_bf16")?,
        fp8_1: g.kernel("gemv_fp8w", "dense_gemv_fp8w")?,
        fp8_bm: g.kernel("dense_gemv_fp8w_batchm", "dense_gemv_fp8w_batchm")?,
    };
    let mut rng = Lcg(0x6c6d_3533_f4f4);
    let (mut bit_fail, mut num_fail) = (0usize, 0usize);
    for &(n, kd, label, _, _) in SHAPES {
        for heavy in [false, true] {
            let (b, nf) = shape_checks(g, &k, &mut rng, n, kd, label, heavy)?;
            bit_fail += b;
            num_fail += nf;
        }
    }
    let parse_fail = parse_checks();
    let mut summary = Vec::new();
    for &(n, kd, label, per_step, default_class) in SHAPES {
        if per_step == 0 {
            continue;
        }
        for (m, t8, tn) in timing(g, &k, &mut rng, n, kd, label)? {
            summary.push((m, t8, tn, per_step, default_class));
        }
    }
    for &m in TIME_MS {
        let rows: Vec<_> = summary.iter().filter(|x| x.0 == m && x.4).collect();
        let f8: f64 = rows.iter().map(|x| x.1 * x.3 as f64).sum();
        let nv: f64 = rows.iter().map(|x| x.2 * x.3 as f64).sum();
        println!(
            "TIMING SUMMARY M={m:<2} default classes (kda,shared,mlp), launches per rank per step: \
             fp8 {f8:.2} ms -> nvfp4 {nv:.2} ms, saves {:.2} ms ({:.2}x)",
            f8 - nv,
            f8 / nv.max(1e-12)
        );
    }
    println!(
        "tolerances (PROVISIONAL): nvfp4 vs fp64(bf16 W) cos>={TOL_COS_NV4} max_rel<={TOL_MAXREL_NV4}; \
         nvfp4 kernel vs fp64(own W) cos>={TOL_COS_SELF} max_rel<={TOL_MAXREL_SELF}; nonfinite=0"
    );
    ensure!(bit_fail == 0, "route/bits: {bit_fail} cases failed (see MISMATCH lines)");
    ensure!(num_fail == 0, "numerics: {num_fail} cases out of tolerance (see OUT-OF-TOL lines)");
    ensure!(parse_fail == 0, "parse: {parse_fail} checks failed (see FAIL-CHECK lines)");
    println!(
        "PASS: dense NVFP4 route rows bitwise == M=1 per row (M 1..16), prefill keeps FP8, numerics within tolerance, class parser"
    );
    Ok(())
}
