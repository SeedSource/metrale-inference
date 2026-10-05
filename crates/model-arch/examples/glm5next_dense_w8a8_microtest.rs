// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Gate for `METRALE_GLM_DENSE_FP8_W8A8=1` (with `METRALE_GLM_DENSE_FP8=1`): the
//! per-row-scale W8A8 GEMM `fp8_gemm_rowscale_pipe_128x64` (`ops::fp8_gemm_t_rowscale`) and
//! the `glm5next_layer::dense_fp8::route` arm that runs it on GLM-5.3 prefill projections.
//!
//! Owner: model-arch examples.
//! Checks (gated; the run prints `PASS:` only if every one holds, else a `FAIL:` line and a
//! nonzero exit):
//! - (a) The kernel against an FP64 host reference over the SAME quantized operands (decoded
//!   E4M3 A x a_scale per 128-K group, decoded E4M3 W x row scale), per output row on a sample
//!   of rows (tile edges 127/128, the last row, random): cosine >= `TOL_COS_REF`, max_rel =
//!   max|y - ref| / max|ref| <= `TOL_MAXREL_REF`, nonfinite 0; plus no nonfinite value in the
//!   whole output and untouched guard rows after it. M in `GATE_MS`, (N, K) in `SHAPES`.
//! - (b) The kernel against today's prefill path (`dequant_fp8_rowscale_bf16` then cuBLASLt
//!   `cublas_bf16_proj_dense`) on the same BF16 activations and the same FP8 weight copy:
//!   cosine per row >= `TOL_COS_PATH` on every row. That is the activation-quantization error
//!   alone.
//! - (c) `dense_fp8::route` end to end with both levers on (weights registered with
//!   `convert_weight`, scratch from `finish_load`): returns `Done` and `c` is bitwise equal to a
//!   direct quant + rowscale launch; fewer than `W8A8_MIN_ROWS` rows and 1..=16-row GEMVs keep
//!   their paths; a call past the scratch (M x K > rows x max K) falls back to the dequant;
//!   a captured route replays to the same bits. The env levers are OnceLocks, so the
//!   lever-off path, a small scratch (`METRALE_GLM_DENSE_FP8_W8A8_ROWS=512`, plus the shape
//!   skips) and the W8A8-without-DENSE_FP8 warning run in child processes of this binary
//!   (`--phase off|cap|nofp8`), each gated by its exit status.
//! - (d) Timing, reported not gated: per shape at M in `TIME_MS`, `route` on the W8A8 arm
//!   against the dequant path it replaces (`dequant_fp8_rowscale_bf16` + cuBLASLt, launched
//!   directly because `route`'s eager cache would skip the dequant on repeats; a chunk-wide
//!   GEMM dequantizes each weight once per layer), and cuBLASLt alone (a cached dequant) for
//!   reference. Host wall-clock around each launch plus a synchronize; `WARMUP` then `REPS`.
//!
//! Run (GB10):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_dense_w8a8_microtest

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_layer::dense_fp8::{self as df, LayerFp8, Route};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::Fp8DenseWeight;
use std::time::Instant;

/// 2026-10-05: (N, K, label): the GLM-5.3 TP2 dense prefill projections the lever converts,
/// and the MTP `eh_proj` (4096 x 8192) another branch registers.
const SHAPES: [(usize, usize, &str); 6] = [
    (4096, 4096, "kda_qkvo"),
    (1024, 4096, "shexp_gate_up"),
    (4096, 1024, "shexp_down"),
    (16384, 1536, "dsa_q_absorb"),
    (4096, 16384, "dsa_o_absorb"),
    (4096, 8192, "mtp_eh_proj"),
];
const GATE_MS: [usize; 3] = [64, 300, 2048];
const TIME_MS: [usize; 2] = [8192, 2048];
const MAX_M: usize = 8192;
/// 2026-10-05: Rows per (M, shape) checked against the FP64 reference.
const REF_ROWS: usize = 12;
/// 2026-10-05: Guard rows after the output, filled with 0xFF bytes (a BF16 NaN).
const GUARD_ROWS: usize = 2;
/// 2026-10-05: Same operands: FP32 accumulation and the BF16 round only.
const TOL_COS_REF: f64 = 0.99999;
const TOL_MAXREL_REF: f64 = 2e-2;
/// 2026-10-05: Against the dequant path: E4M3 activation rounding (relative <= 2^-4 per value,
/// ~2.7 % RMS) gives cos ~ 0.9996 on random data.
const TOL_COS_PATH: f64 = 0.999;
const WARMUP: usize = 3;
const REPS: usize = 20;
const MAIN_ROWS: usize = 8192;
const CAP_ROWS: usize = 512;
const KG: usize = 128;

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

/// 2026-10-05: Weights: N(0, 0.02) with a per-row gain (x0.25..x4) and 0.1 % outliers at x30,
/// as `glm5next_dense_fp8_microtest`'s heavy generator.
fn gen_weight(rng: &mut Lcg, n: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * k * 2);
    for _ in 0..n {
        let gain = 0.25 * 16f64.powf(rng.u());
        for _ in 0..k {
            let mut v = rng.n() * 0.02 * gain;
            if rng.u() < 0.001 {
                v *= 30.0;
            }
            out.extend_from_slice(&bf16::from_f64(v).to_bits().to_le_bytes());
        }
    }
    out
}

/// 2026-10-05: Activations: uniform in [-1, 1) with 0.1 % outliers at x20 (one outlier sets its
/// 128-group's scale).
fn gen_act(rng: &mut Lcg, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 2);
    for _ in 0..len {
        let mut v = rng.u() * 2.0 - 1.0;
        if rng.u() < 0.001 {
            v *= 20.0;
        }
        out.extend_from_slice(&bf16::from_f64(v).to_bits().to_le_bytes());
    }
    out
}

fn upload(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(16))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn ones(g: &dyn GpuBackend, k: usize) -> Result<DevicePtr> {
    let b: Vec<u8> = (0..k / KG).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    upload(g, &b)
}
fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn dn_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    Ok(dn_bytes(g, p, n * 2)?
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}
fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    Ok(dn_bytes(g, p, n * 4)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn bf(x: u16) -> f64 {
    bf16::from_bits(x).to_f64()
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

fn cosine(a: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na > 0.0 && nb > 0.0 {
        d / (na.sqrt() * nb.sqrt())
    } else {
        f64::NAN
    }
}

struct Kern {
    dq: KernelHandle,
    rs: KernelHandle,
    quant: ops::Fp8ActQuant,
    gemv: KernelHandle,
}

fn kern(g: &dyn GpuBackend) -> Result<Kern> {
    let quant = ops::Fp8ActQuant::resolve(g);
    ensure!(
        quant.available(),
        "per_token_group_quant_fp8 is not in this image"
    );
    Ok(Kern {
        dq: g.kernel("dequant_fp8_rowscale_bf16", "dequant_fp8_rowscale_bf16")?,
        rs: ops::fp8_gemm_rowscale_kernel(g)?.context(
            "this image lacks fp8_gemm_blockscaled_pipe (fp8_gemm_rowscale_pipe_128x64)",
        )?,
        quant,
        gemv: g.kernel("gemv", "dense_gemv_bf16")?,
    })
}

/// 2026-10-05: Per-call scratch for direct (non-route) launches of one shape.
struct Direct {
    a_q: DevicePtr,
    a_s: DevicePtr,
    ones: DevicePtr,
}

impl Direct {
    fn new(g: &dyn GpuBackend, rows: usize, k: usize) -> Result<Self> {
        Ok(Self {
            a_q: g.alloc(rows * k)?,
            a_s: g.alloc(rows * (k / KG) * 4)?,
            ones: ones(g, k)?,
        })
    }
    /// Quant `a` then the rowscale GEMM into `c`, as `route` launches them.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        g: &dyn GpuBackend,
        kn: &Kern,
        a: DevicePtr,
        w: &Fp8DenseWeight,
        c: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        s: u64,
    ) -> Result<()> {
        ops::per_token_group_quant_fp8(g, kn.quant, a, self.a_q, self.a_s, m as u32, k as u32, s)?;
        ops::fp8_gemm_t_rowscale(
            g,
            kn.rs,
            self.a_q,
            self.a_s,
            self.ones,
            w.weight,
            w.row_scale,
            c,
            m as u32,
            n as u32,
            k as u32,
            s,
        )
    }
    fn free(self, g: &dyn GpuBackend) -> Result<()> {
        for p in [self.a_q, self.a_s, self.ones] {
            g.free(p)?;
        }
        Ok(())
    }
}

/// 2026-10-05: The rows (a) checks against FP64: tile edges, the last row, random ones.
fn ref_rows(rng: &mut Lcg, m: usize) -> Vec<usize> {
    let mut r: Vec<usize> = [0, 1, 63, 127, 128, 255, 256, m / 2, m - 1]
        .into_iter()
        .filter(|&x| x < m)
        .collect();
    r.sort_unstable();
    r.dedup();
    while r.len() < REF_ROWS.min(m) {
        let x = (rng.u() * m as f64) as usize % m;
        if !r.contains(&x) {
            r.push(x);
        }
    }
    r.sort_unstable();
    r
}

/// 2026-10-05: (a) and (b) for one shape at one M. Returns the number of failed checks.
#[allow(clippy::too_many_arguments)]
fn gates(
    g: &dyn GpuBackend,
    kn: &Kern,
    rng: &mut Lcg,
    ad: DevicePtr,
    w: &Fp8DenseWeight,
    w_dec: &[f32],
    w_scale: &[f32],
    m: usize,
    n: usize,
    k: usize,
    label: &str,
) -> Result<usize> {
    let s = g.default_stream();
    let dir = Direct::new(g, m, k)?;
    let c_new = g.alloc((m + GUARD_ROWS) * n * 2)?;
    let c_old = g.alloc(m * n * 2)?;
    let deq = g.alloc(n * k * 2)?;
    g.synchronize(s)?;
    g.memset(c_new, 0xFF, (m + GUARD_ROWS) * n * 2)?;
    dir.run(g, kn, ad, w, c_new, m, n, k, s)?;
    ops::dequant_fp8_rowscale_bf16(g, kn.dq, w, deq, n as u32, k as u32, s)?;
    ops::cublas_bf16_proj_dense(ad, deq, c_old, m as u32, n as u32, k as u32, s)?;
    g.synchronize(s)?;
    let y = dn_u16(g, c_new, (m + GUARD_ROWS) * n)?;
    let y_old = dn_u16(g, c_old, m * n)?;
    let aq = dn_bytes(g, dir.a_q, m * k)?;
    let asc = dn_f32(g, dir.a_s, m * (k / KG))?;
    let mut fails = 0usize;

    // (a) FP64 reference over the same quantized operands.
    let nonfinite_all = y[..m * n].iter().filter(|&&v| !bf(v).is_finite()).count();
    let guard_ok = y[m * n..].iter().all(|&v| v == 0xFFFF);
    let groups = k / KG;
    let (mut cos_min, mut rel_max, mut bad_rows) = (1f64, 0f64, 0usize);
    let rows = ref_rows(rng, m);
    let mut a_dec = vec![0f64; k];
    let mut refv = vec![0f64; n];
    for &r in &rows {
        for (d, &b) in a_dec.iter_mut().zip(&aq[r * k..(r + 1) * k]) {
            *d = e4m3(b);
        }
        for (j, out) in refv.iter_mut().enumerate() {
            let wrow = &w_dec[j * k..(j + 1) * k];
            let mut acc = 0f64;
            for gi in 0..groups {
                let part: f64 = a_dec[gi * KG..(gi + 1) * KG]
                    .iter()
                    .zip(&wrow[gi * KG..(gi + 1) * KG])
                    .map(|(x, &y)| x * y as f64)
                    .sum();
                acc += part * asc[r * groups + gi] as f64;
            }
            *out = acc * w_scale[j] as f64;
        }
        let yr: Vec<f64> = y[r * n..(r + 1) * n].iter().map(|&v| bf(v)).collect();
        let cos = cosine(yr.iter().copied().zip(refv.iter().copied()));
        let ref_max = refv.iter().fold(0f64, |a, v| a.max(v.abs()));
        let err_max = yr
            .iter()
            .zip(&refv)
            .fold(0f64, |a, (x, r)| a.max((x - r).abs()));
        let rel = if ref_max > 0.0 {
            err_max / ref_max
        } else {
            f64::INFINITY
        };
        let nonfin = yr.iter().filter(|v| !v.is_finite()).count();
        cos_min = cos_min.min(cos);
        rel_max = rel_max.max(rel);
        // NaN cos or rel fails.
        if cos.is_nan() || rel.is_nan() || cos < TOL_COS_REF || rel > TOL_MAXREL_REF || nonfin > 0 {
            bad_rows += 1;
            println!(
                "OUT-OF-TOL (a) m={m} n={n} k={k} {label} row={r} cos={cos:.7} max_rel={rel:.3e} \
                 nonfinite={nonfin}"
            );
        }
    }
    let ok_a = bad_rows == 0 && nonfinite_all == 0 && guard_ok;
    fails += usize::from(!ok_a);
    println!(
        "GATE-A m={m} n={n} k={k} {label} rows_checked={} cos_min={cos_min:.7} \
         max_rel_max={rel_max:.3e} nonfinite_all={nonfinite_all} guard_untouched={guard_ok}  {}",
        rows.len(),
        if ok_a { "ok" } else { "FAIL-CHECK" }
    );

    // (b) Against the dequant + cuBLASLt path, every row.
    let (mut cos_min_b, mut sum_b, mut bad_b) = (1f64, 0f64, 0usize);
    for r in 0..m {
        let cos = cosine(
            y[r * n..(r + 1) * n]
                .iter()
                .zip(&y_old[r * n..(r + 1) * n])
                .map(|(&a, &b)| (bf(a), bf(b))),
        );
        if cos.is_nan() || cos < TOL_COS_PATH {
            bad_b += 1;
        }
        cos_min_b = cos_min_b.min(cos);
        sum_b += cos;
    }
    let ok_b = bad_b == 0;
    fails += usize::from(!ok_b);
    println!(
        "GATE-B m={m} n={n} k={k} {label} vs dequant+cuBLASLt: cos_min={cos_min_b:.6} \
         cos_mean={:.6} rows_below={bad_b}  {}",
        sum_b / m as f64,
        if ok_b { "ok" } else { "FAIL-CHECK" }
    );
    for p in [c_new, c_old, deq] {
        g.free(p)?;
    }
    dir.free(g)?;
    Ok(fails)
}

/// 2026-10-05: (mean, min) ms over `REPS` launches, each followed by a synchronize, after
/// `WARMUP` untimed ones.
fn time_arm(g: &dyn GpuBackend, s: u64, f: &mut dyn FnMut() -> Result<()>) -> Result<(f64, f64)> {
    for _ in 0..WARMUP {
        f()?;
    }
    g.synchronize(s)?;
    let (mut sum, mut min) = (0f64, f64::MAX);
    for _ in 0..REPS {
        let t = Instant::now();
        f()?;
        g.synchronize(s)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        sum += ms;
        min = min.min(ms);
    }
    Ok((sum / REPS as f64, min))
}

fn report(arm: &str, m: usize, n: usize, k: usize, label: &str, (mean, min): (f64, f64)) {
    let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (mean * 1e-3) / 1e12;
    println!(
        "TIMING m={m} n={n} k={k} {label} arm={arm} ms_mean={mean:.4} ms_min={min:.4} \
         tflops={tflops:.2}"
    );
}

/// 2026-10-05: Upload a BF16 weight and register it (each in its own layer, so the arena is
/// the largest single weight). Returns the registry key and the FP8 copy.
fn register(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    n: usize,
    k: usize,
) -> Result<(DevicePtr, Fp8DenseWeight)> {
    let mut p = upload(g, &gen_weight(rng, n, k))?;
    let mut acc = LayerFp8::default();
    ensure!(
        df::convert_weight(g, &mut p, n, k, &mut acc)?,
        "convert_weight declined [{n}, {k}]"
    );
    let w = df::lookup(p, n, k).context("registered weight not found")?;
    Ok((p, w))
}

fn route_kind(r: Route, b: DevicePtr) -> &'static str {
    match r {
        Route::Done => "Done",
        Route::Weight(p) if p == b => "Weight(own)",
        Route::Weight(_) => "Weight(dequant)",
    }
}

/// 2026-10-05: Main phase: both levers on, `METRALE_GLM_DENSE_FP8_W8A8_ROWS=8192`.
fn main_phase() -> Result<usize> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kn = kern(g)?;
    let s = g.default_stream();
    let mut rng = Lcg(0x7738_6138_f8f8_0510);
    let mut fails = 0usize;
    let mut gate_fails = 0usize;
    let mut check = |ok: bool, what: &str| {
        if !ok {
            fails += 1;
        }
        println!("ROUTE {what}  {}", if ok { "ok" } else { "FAIL-CHECK" });
    };
    check(
        df::dense_fp8() && df::dense_fp8_w8a8() && df::dense_fp8_w8a8_rows() == MAIN_ROWS,
        "levers: DENSE_FP8 on, W8A8 on, scratch rows 8192",
    );

    eprintln!("registering {} weights", SHAPES.len());
    let mut reg = Vec::new();
    for (n, k, _) in SHAPES {
        reg.push(register(g, &mut rng, n, k)?);
    }
    let live0 = g.live_bytes().unwrap_or(0) as i64;
    let arena = df::finish_load(g)?;
    let max_k = SHAPES.iter().map(|s| s.1).max().unwrap();
    let fp8_cap = MAIN_ROWS * max_k;
    let scale_cap = MAIN_ROWS * (max_k / KG) * 4;
    let want = (fp8_cap.next_multiple_of(256) + scale_cap).next_multiple_of(256) + max_k / KG * 4;
    let scratch = df::w8a8_scratch_bytes();
    let live1 = g.live_bytes().unwrap_or(0) as i64;
    check(
        scratch == want && live1 - live0 == (arena + scratch) as i64,
        &format!(
            "finish_load: W8A8 scratch {scratch} B (want {want} B), arena {arena} B, ledger +{} B",
            live1 - live0
        ),
    );

    let mut timing = Vec::new();
    for (i, &(n, k, label)) in SHAPES.iter().enumerate() {
        let (key, w) = reg[i];
        eprintln!("{label}: n={n} k={k}: operands");
        let ad = upload(g, &gen_act(&mut rng, MAX_M * k))?;
        let w_dec: Vec<f32> = dn_bytes(g, w.weight, n * k)?
            .iter()
            .map(|&b| e4m3(b) as f32)
            .collect();
        let w_scale = dn_f32(g, w.row_scale, n)?;

        // (a), (b)
        for m in GATE_MS {
            gate_fails += gates(g, &kn, &mut rng, ad, &w, &w_dec, &w_scale, m, n, k, label)?;
        }
        drop(w_dec);

        // (c) route against a direct launch, and the arms around it.
        let dir = Direct::new(g, 300, k)?;
        let c1 = g.alloc(MAX_M * n * 2)?;
        let c2 = g.alloc(MAX_M * n * 2)?;
        let (w0, d0) = (df::w8a8_gemms(), df::dequants());
        let r = df::route(g, kn.gemv, ad, key, c1, 300, n, k, s)?;
        dir.run(g, &kn, ad, &w, c2, 300, n, k, s)?;
        g.synchronize(s)?;
        let same = dn_u16(g, c1, 300 * n)? == dn_u16(g, c2, 300 * n)?;
        check(
            r == Route::Done && same && df::w8a8_gemms() == w0 + 1 && df::dequants() == d0,
            &format!(
                "{label} M=300: {} (want Done), bits == direct quant+rowscale: {same}, one W8A8 \
                 launch, no dequant",
                route_kind(r, key)
            ),
        );
        let w0 = df::w8a8_gemms();
        let r64 = df::route(g, kn.gemv, ad, key, c1, 64, n, k, s)?;
        let r63 = df::route(g, kn.gemv, ad, key, c1, 63, n, k, s)?;
        let r16 = df::route(g, kn.gemv, ad, key, c1, 16, n, k, s)?;
        check(
            r64 == Route::Done
                && matches!(r63, Route::Weight(p) if p != key)
                && r16 == Route::Done
                && df::w8a8_gemms() == w0 + 1,
            &format!(
                "{label} M=64 {} / M=63 {} / M=16 {}: W8A8 from {} rows, dequant below, GEMV at \
                 <=16 (one W8A8 launch)",
                route_kind(r64, key),
                route_kind(r63, key),
                route_kind(r16, key),
                df::W8A8_MIN_ROWS
            ),
        );
        dir.free(g)?;

        // (d) timing.
        let deq = g.alloc(n * k * 2)?;
        ops::dequant_fp8_rowscale_bf16(g, kn.dq, &w, deq, n as u32, k as u32, s)?;
        for m in TIME_MS {
            let t_w8 = time_arm(
                g,
                s,
                &mut || match df::route(g, kn.gemv, ad, key, c1, m, n, k, s)? {
                    Route::Done => Ok(()),
                    Route::Weight(_) => bail!("route did not take W8A8 at M={m}"),
                },
            )?;
            let t_dq = time_arm(g, s, &mut || {
                ops::dequant_fp8_rowscale_bf16(g, kn.dq, &w, deq, n as u32, k as u32, s)?;
                ops::cublas_bf16_proj_dense(ad, deq, c2, m as u32, n as u32, k as u32, s)
            })?;
            let t_mm = time_arm(g, s, &mut || {
                ops::cublas_bf16_proj_dense(ad, deq, c2, m as u32, n as u32, k as u32, s)
            })?;
            report("w8a8", m, n, k, label, t_w8);
            report("dequant_bf16", m, n, k, label, t_dq);
            report("bf16_gemm_cached", m, n, k, label, t_mm);
            println!(
                "RATIO m={m} n={n} k={k} {label} w8a8_over_dequant_bf16={:.3} \
                 w8a8_over_bf16_gemm_cached={:.3} saved_ms={:.4}",
                t_w8.0 / t_dq.0,
                t_w8.0 / t_mm.0,
                t_dq.0 - t_w8.0
            );
            timing.push((m, t_dq.0 - t_w8.0));
        }
        for p in [ad, c1, c2, deq] {
            g.free(p)?;
        }
    }
    for m in TIME_MS {
        let saved: f64 = timing.iter().filter(|t| t.0 == m).map(|t| t.1).sum();
        println!(
            "TIMING SUMMARY m={m}: one launch per shape above, dequant+cuBLASLt minus W8A8 = \
             {saved:.3} ms (not weighted by launches per layer)"
        );
    }

    // (c) capacity: M x K above rows x max K falls back; equal fits (k = 8192: 16384 rows).
    let (key_o, _) = reg[4];
    let (key_e, _) = reg[5];
    let big = g.alloc(16385 * 8192 * 2)?;
    let cbig = g.alloc(16385 * 4096 * 2)?;
    g.memset(big, 0, 16385 * 8192 * 2)?;
    let w0 = df::w8a8_gemms();
    let r_o = df::route(g, kn.gemv, big, key_o, cbig, MAIN_ROWS + 1, 4096, 16384, s)?;
    let r_e_fit = df::route(g, kn.gemv, big, key_e, cbig, 16384, 4096, 8192, s)?;
    let r_e_over = df::route(g, kn.gemv, big, key_e, cbig, 16385, 4096, 8192, s)?;
    g.synchronize(s)?;
    check(
        matches!(r_o, Route::Weight(p) if p != key_o)
            && r_e_fit == Route::Done
            && matches!(r_e_over, Route::Weight(p) if p != key_e)
            && df::w8a8_gemms() == w0 + 1,
        &format!(
            "capacity: [4096,16384] M=8193 {} (want dequant), [4096,8192] M=16384 {} (want Done), \
             M=16385 {} (want dequant)",
            route_kind(r_o, key_o),
            route_kind(r_e_fit, key_e),
            route_kind(r_e_over, key_e)
        ),
    );
    g.free(big)?;
    g.free(cbig)?;

    // (c) a captured route: nothing runs at capture, the replay writes the same bits.
    let (n, k, _) = SHAPES[0];
    let (key, w) = reg[0];
    let m = 128;
    let ad = upload(g, &gen_act(&mut rng, m * k))?;
    let c1 = g.alloc(m * n * 2)?;
    let c2 = g.alloc(m * n * 2)?;
    let dir = Direct::new(g, m, k)?;
    dir.run(g, &kn, ad, &w, c2, m, n, k, s)?;
    g.synchronize(s)?;
    g.memset(c1, 0, m * n * 2)?;
    let s2 = g.create_stream()?;
    g.begin_capture(s2)?;
    let cap = df::route(g, kn.gemv, ad, key, c1, m, n, k, s2);
    let graph = g.end_capture(s2)?;
    let zero_before = dn_u16(g, c1, m * n)?.iter().all(|&x| x == 0);
    g.launch_graph(graph, s2)?;
    g.synchronize(s2)?;
    let replay_same = dn_u16(g, c1, m * n)? == dn_u16(g, c2, m * n)?;
    g.destroy_graph(graph)?;
    check(
        matches!(cap, Ok(Route::Done)) && zero_before && replay_same,
        "route M=128 under capture: Done, nothing written at capture, replay bits == direct",
    );
    // An eager call on the first stream after the capture.
    let r = df::route(g, kn.gemv, ad, key, c1, m, n, k, s)?;
    g.synchronize(s)?;
    check(
        r == Route::Done && dn_u16(g, c1, m * n)? == dn_u16(g, c2, m * n)?,
        "route M=128 eager after the capture: Done, bits == direct",
    );
    dir.free(g)?;
    for p in [ad, c1, c2] {
        g.free(p)?;
    }
    println!(
        "ROUTE stats: W8A8 launches {} dequant launches {} FP8 GEMV launches {}",
        df::w8a8_gemms(),
        df::dequants(),
        df::hits()
    );
    Ok(fails + gate_fails)
}

/// 2026-10-05: `--phase off`: DENSE_FP8 on, W8A8 unset: a wide call takes the dequant path,
/// no scratch is allocated, no W8A8 launch is made.
fn phase_off() -> Result<usize> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kn = kern(g)?;
    let s = g.default_stream();
    let mut rng = Lcg(0x6f66_66f8);
    let (n, k) = (1024, 4096);
    let (key, w) = register(g, &mut rng, n, k)?;
    df::finish_load(g)?;
    let ad = upload(g, &gen_act(&mut rng, 300 * k))?;
    let c = g.alloc(300 * n * 2)?;
    let deq = g.alloc(n * k * 2)?;
    let r = df::route(g, kn.gemv, ad, key, c, 300, n, k, s)?;
    ops::dequant_fp8_rowscale_bf16(g, kn.dq, &w, deq, n as u32, k as u32, s)?;
    g.synchronize(s)?;
    let same = match r {
        Route::Weight(p) if p != key => dn_u16(g, p, n * k)? == dn_u16(g, deq, n * k)?,
        _ => false,
    };
    let ok =
        !df::dense_fp8_w8a8() && df::w8a8_scratch_bytes() == 0 && df::w8a8_gemms() == 0 && same;
    println!(
        "CHILD off: W8A8 lever {} scratch {} B route M=300 {} dequant bits == direct {same}  {}",
        df::dense_fp8_w8a8(),
        df::w8a8_scratch_bytes(),
        route_kind(r, key),
        if ok { "ok" } else { "FAIL-CHECK" }
    );
    Ok(usize::from(!ok))
}

/// 2026-10-05: `--phase cap`: both levers, `METRALE_GLM_DENSE_FP8_W8A8_ROWS=512`: the scratch
/// size, the byte-capacity rule at two K, and the shape skips (n % 64, k % 128).
fn phase_cap() -> Result<usize> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kn = kern(g)?;
    let s = g.default_stream();
    let mut rng = Lcg(0x6361_70f8);
    let mut fails = 0usize;
    let mut check = |ok: bool, what: &str| {
        if !ok {
            fails += 1;
        }
        println!(
            "CHILD cap: {what}  {}",
            if ok { "ok" } else { "FAIL-CHECK" }
        );
    };
    let (ka, wa) = register(g, &mut rng, 1024, 4096)?;
    let (kb, _) = register(g, &mut rng, 4096, 1024)?;
    let (kc, _) = register(g, &mut rng, 1000, 1024)?;
    let (kd, _) = register(g, &mut rng, 1024, 1040)?;
    df::finish_load(g)?;
    // max_k = 4096 (k = 1040 is not a multiple of 128 and does not count).
    let want = (CAP_ROWS * 4096 + CAP_ROWS * 32 * 4).next_multiple_of(256) + 32 * 4;
    check(
        df::dense_fp8_w8a8_rows() == CAP_ROWS && df::w8a8_scratch_bytes() == want,
        &format!(
            "rows {} scratch {} B (want {CAP_ROWS} rows, {want} B)",
            df::dense_fp8_w8a8_rows(),
            df::w8a8_scratch_bytes()
        ),
    );
    let ad = upload(g, &gen_act(&mut rng, 2049 * 4096))?;
    let c = g.alloc(2049 * 4096 * 2)?;
    let c2 = g.alloc(2049 * 4096 * 2)?;
    let kind = |key: DevicePtr, m: usize, n: usize, k: usize| -> Result<&'static str> {
        Ok(route_kind(
            df::route(g, kn.gemv, ad, key, c, m, n, k, s)?,
            key,
        ))
    };
    let a512 = kind(ka, 512, 1024, 4096)?;
    let a513 = kind(ka, 513, 1024, 4096)?;
    let b2048 = kind(kb, 2048, 4096, 1024)?;
    let b2049 = kind(kb, 2049, 4096, 1024)?;
    let c128 = kind(kc, 128, 1000, 1024)?;
    let d128 = kind(kd, 128, 1024, 1040)?;
    check(
        a512 == "Done"
            && a513 == "Weight(dequant)"
            && b2048 == "Done"
            && b2049 == "Weight(dequant)",
        &format!(
            "capacity in bytes: K=4096 M=512 {a512} M=513 {a513}; K=1024 M=2048 {b2048} M=2049 \
             {b2049}"
        ),
    );
    check(
        c128 == "Weight(dequant)" && d128 == "Weight(dequant)",
        &format!("shape skips: n=1000 {c128}, k=1040 {d128} (want dequant)"),
    );
    let r = df::route(g, kn.gemv, ad, ka, c, 512, 1024, 4096, s)?;
    let dir = Direct::new(g, 512, 4096)?;
    dir.run(g, &kn, ad, &wa, c2, 512, 1024, 4096, s)?;
    g.synchronize(s)?;
    check(
        r == Route::Done && dn_u16(g, c, 512 * 1024)? == dn_u16(g, c2, 512 * 1024)?,
        "K=4096 M=512 (full scratch): bits == direct",
    );
    Ok(fails)
}

/// 2026-10-05: `--phase nofp8`: W8A8 set without DENSE_FP8 is ignored.
fn phase_nofp8() -> Result<usize> {
    let ok = !df::dense_fp8() && !df::dense_fp8_w8a8();
    println!(
        "CHILD nofp8: DENSE_FP8 {} W8A8 {} (want both false)  {}",
        df::dense_fp8(),
        df::dense_fp8_w8a8(),
        if ok { "ok" } else { "FAIL-CHECK" }
    );
    Ok(usize::from(!ok))
}

const LEVERS: [&str; 3] = [
    "METRALE_GLM_DENSE_FP8",
    "METRALE_GLM_DENSE_FP8_W8A8",
    "METRALE_GLM_DENSE_FP8_W8A8_ROWS",
];

/// 2026-10-05: Run this binary as `--phase <name>` with exactly `env` set among [`LEVERS`].
fn child(name: &str, env: &[(&str, &str)]) -> Result<bool> {
    let mut cmd = std::process::Command::new(std::env::current_exe()?);
    cmd.arg("--phase").arg(name);
    for v in LEVERS {
        cmd.env_remove(v);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let st = cmd.status()?;
    println!("CHILD {name}: exit {st}");
    Ok(st.success())
}

fn finish(fails: usize, what: &str) -> Result<()> {
    if fails > 0 {
        println!("FAIL: {what}: {fails} checks failed (see FAIL-CHECK / OUT-OF-TOL lines)");
        bail!("{what}: {fails} checks failed");
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--phase") {
        let phase = args.get(i + 1).map(String::as_str).unwrap_or("");
        let fails = match phase {
            "off" => phase_off()?,
            "cap" => phase_cap()?,
            "nofp8" => phase_nofp8()?,
            other => bail!("unknown phase {other}"),
        };
        return finish(fails, &format!("phase {phase}"));
    }
    let mut child_fail = 0usize;
    child_fail += usize::from(!child("off", &[("METRALE_GLM_DENSE_FP8", "1")])?);
    child_fail += usize::from(!child(
        "cap",
        &[
            ("METRALE_GLM_DENSE_FP8", "1"),
            ("METRALE_GLM_DENSE_FP8_W8A8", "1"),
            ("METRALE_GLM_DENSE_FP8_W8A8_ROWS", "512"),
        ],
    )?);
    child_fail += usize::from(!child("nofp8", &[("METRALE_GLM_DENSE_FP8_W8A8", "1")])?);
    // SAFETY: single-threaded here (the children have exited), before anything in this
    // process reads the levers.
    unsafe {
        std::env::set_var("METRALE_GLM_DENSE_FP8", "1");
        std::env::set_var("METRALE_GLM_DENSE_FP8_W8A8", "1");
        std::env::set_var("METRALE_GLM_DENSE_FP8_W8A8_ROWS", MAIN_ROWS.to_string());
    }
    let fails = main_phase()?;
    println!(
        "tolerances: (a) vs FP64 over the same quantized operands cos>={TOL_COS_REF} \
         max_rel<={TOL_MAXREL_REF} nonfinite=0; (b) vs dequant+cuBLASLt cos>={TOL_COS_PATH} per row"
    );
    finish(child_fail, "child phases (off / cap / nofp8)")?;
    finish(fails, "main phase")?;
    println!(
        "PASS: fp8_gemm_rowscale_pipe_128x64 within tolerance of FP64 over its own quantized \
         operands and of the dequant+cuBLASLt path; dense_fp8::route W8A8 arm bitwise == direct \
         launch, row/shape/capacity fallbacks, capture replay, lever-off and W8A8-without-FP8 \
         paths pass"
    );
    Ok(())
}
