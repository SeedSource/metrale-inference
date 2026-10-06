// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Gate for `METRALE_GLM_DENSE_FP8_W8A8_MIN_ROWS=17`: the W8A8 prefill GEMM
//! (`ops::per_token_group_quant_fp8` + `ops::fp8_gemm_t_rowscale`, run by `dense_fp8::route`) on
//! 17..=63-row GEMMs, which keep the dequant + cuBLASLt path at the default minimum of 64.
//! Owner: model-arch examples.
//! Invariants (`PASS:` only if every gated check holds, else a `FAIL:` line and a nonzero exit;
//! the >= 64-row arm is `glm5next_dense_w8a8_microtest`'s):
//! - (a) Row invariance: for every GLM dense shape and M in `ROW_MS`, the direct W8A8 launch on
//!   the first M of 128 random BF16 rows is bit-identical (raw u16 BF16 patterns) to the first M
//!   rows of the same launch on all 128, so a row's output does not depend on how many rows share
//!   the GEMM. Known-bad: one element of row M/2 changed in the M-row input only must change that
//!   row and no other, so the comparison is not blind. Rows past M stay untouched.
//! - (b) Accuracy: per-row cosine of the W8A8 output against dequant + cuBLASLt on the same BF16
//!   activations and FP8 weight copy >= `TOL_COS_PATH`; no nonfinite value.
//! - (c) Route: `dense_fp8::route` at M = 17 and 63 returns `Done`, adds one `w8a8_gemms()` and
//!   no `dequants()`, and writes the direct launch's bits; at M = 16 it runs the FP8 GEMV tier
//!   (`hits()` + 1), not W8A8.
//! - (d) Timing, reported not gated: per shape, M in `TIME_MS`, median of `REPS` cold launches
//!   (a 256 MiB memset flushes L2 first) of `route` on W8A8 against `dequant_fp8_rowscale_bf16` +
//!   cuBLASLt (dequant launched each time: a chunk-wide GEMM dequantizes each weight once per
//!   layer). Host wall-clock around launch + synchronize (no event elapsed-time in the backend).
//!
//! Run (GB10):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_dense_w8a8_minrows_microtest

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_layer::dense_fp8::{self as df, LayerFp8, Route};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::Fp8DenseWeight;
use std::time::Instant;

/// 2026-10-06: (N, K, label): the template's GLM-5.3 TP2 dense prefill projections.
const SHAPES: [(usize, usize, &str); 6] = [
    (4096, 4096, "kda_qkvo"),
    (1024, 4096, "shexp_gate_up"),
    (4096, 1024, "shexp_down"),
    (16384, 1536, "dsa_q_absorb"),
    (4096, 16384, "dsa_o_absorb"),
    (4096, 8192, "mtp_eh_proj"),
];
const ROW_MS: [usize; 4] = [17, 24, 40, 63];
const TIME_MS: [usize; 4] = [17, 32, 48, 63];
/// 2026-10-06: Rows of the full activation block the M-row runs are compared against.
const A_ROWS: usize = 128;
/// 2026-10-06: Guard rows after each M-row output, filled with 0xFF bytes (a BF16 NaN).
const GUARD_ROWS: usize = 2;
/// 2026-10-06: Against the dequant path (the template's bound; E4M3 rounding gives ~0.9996).
const TOL_COS_PATH: f64 = 0.999;
const WARMUP: usize = 2;
const REPS: usize = 10;
const FLUSH_BYTES: usize = 256 << 20;
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
    fn n(&mut self) -> f64 {
        (self.u() + self.u() + self.u() + self.u() - 2.0) * 1.7320508
    }
}

/// 2026-10-06: Weights: the template's heavy generator (N(0, 0.02), row gain, x30 outliers).
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

/// 2026-10-06: Activations: the template's (uniform [-1, 1), 0.1 % outliers at x20).
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
fn dn_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}
fn bf(x: u16) -> f64 {
    bf16::from_bits(x).to_f64()
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

/// 2026-10-06: Per-shape scratch for direct (non-route) launches of up to `rows` rows.
struct Direct {
    a_q: DevicePtr,
    a_s: DevicePtr,
    ones: DevicePtr,
}

impl Direct {
    fn new(g: &dyn GpuBackend, rows: usize, k: usize) -> Result<Self> {
        let o: Vec<u8> = (0..k / KG).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        Ok(Self {
            a_q: g.alloc(rows * k)?,
            a_s: g.alloc(rows * (k / KG) * 4)?,
            ones: upload(g, &o)?,
        })
    }
    /// Quant `a` (m rows) then the rowscale GEMM into `c`, as `route` launches them.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        g: &dyn GpuBackend,
        kn: &Kern,
        a: DevicePtr,
        w: &Fp8DenseWeight,
        c: DevicePtr,
        (m, n, k): (usize, usize, usize),
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

/// 2026-10-06: Upload a BF16 weight and register it (each in its own layer). Returns the
/// registry key and the FP8 copy.
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

struct Checks(usize);
impl Checks {
    fn line(&mut self, tag: &str, ok: bool, what: &str) {
        self.0 += usize::from(!ok);
        println!("{tag} {what}  {}", if ok { "ok" } else { "FAIL-CHECK" });
    }
}

/// 2026-10-06: Median (ms) over `REPS` cold launches: L2 flushed by a memset, then the launch and
/// a synchronize timed on the host, after `WARMUP` untimed ones.
fn time_cold(
    g: &dyn GpuBackend,
    s: u64,
    flush: DevicePtr,
    f: &mut dyn FnMut() -> Result<()>,
) -> Result<f64> {
    let mut v = Vec::new();
    for i in 0..WARMUP + REPS {
        g.memset(flush, (i % 251) as u8, FLUSH_BYTES)?;
        g.synchronize(s)?;
        let t = Instant::now();
        f()?;
        g.synchronize(s)?;
        if i >= WARMUP {
            v.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Ok(v[REPS / 2])
}

/// 2026-10-06: (a) and (b) for one shape. Returns the direct W8A8 output (as u16 bits) of the
/// first M rows per M in `ROW_MS`, for (c).
#[allow(clippy::too_many_arguments)]
fn gates_ab(
    g: &dyn GpuBackend,
    ck: &mut Checks,
    kn: &Kern,
    w: &Fp8DenseWeight,
    (ad, ab): (DevicePtr, &[u8]),
    (n, k, label): (usize, usize, &str),
    s: u64,
) -> Result<Vec<Vec<u16>>> {
    let dir = Direct::new(g, A_ROWS, k)?;
    let c_full = g.alloc(A_ROWS * n * 2)?;
    let deq = g.alloc(n * k * 2)?;
    dir.run(g, kn, ad, w, c_full, (A_ROWS, n, k), s)?;
    ops::dequant_fp8_rowscale_bf16(g, kn.dq, w, deq, n as u32, k as u32, s)?;
    g.synchronize(s)?;
    let full = dn_u16(g, c_full, A_ROWS * n)?;
    let mut outs = Vec::new();
    for m in ROW_MS {
        let c_m = g.alloc((m + GUARD_ROWS) * n * 2)?;
        let c_old = g.alloc(m * n * 2)?;
        g.memset(c_m, 0xFF, (m + GUARD_ROWS) * n * 2)?;
        dir.run(g, kn, ad, w, c_m, (m, n, k), s)?;
        ops::cublas_bf16_proj_dense(ad, deq, c_old, m as u32, n as u32, k as u32, s)?;
        g.synchronize(s)?;
        let y = dn_u16(g, c_m, (m + GUARD_ROWS) * n)?;
        let y_old = dn_u16(g, c_old, m * n)?;

        // (a) bit-for-bit against the first M rows of the 128-row launch.
        let same = y[..m * n] == full[..m * n];
        let guard_ok = y[m * n..].iter().all(|&v| v == 0xFFFF);
        // Known-bad: bump one element of row M/2 in the M-row input only.
        let r = m / 2;
        let mut bad = ab[..m * k * 2].to_vec();
        let at = (r * k + 7) * 2;
        let old = bf16::from_le_bytes([bad[at], bad[at + 1]]);
        bad[at..at + 2].copy_from_slice(&bf16::from_f32(old.to_f32() + 8.0).to_le_bytes());
        let a_bad = upload(g, &bad)?;
        let c_bad = g.alloc(m * n * 2)?;
        dir.run(g, kn, a_bad, w, c_bad, (m, n, k), s)?;
        g.synchronize(s)?;
        let yb = dn_u16(g, c_bad, m * n)?;
        let row_differs = yb[r * n..(r + 1) * n] != full[r * n..(r + 1) * n];
        let others_same = (0..m)
            .filter(|&i| i != r)
            .all(|i| yb[i * n..(i + 1) * n] == full[i * n..(i + 1) * n]);
        ck.line(
            "GATE-A",
            same && guard_ok && row_differs && others_same,
            &format!(
                "{n}x{k} {label} M={m}: rows == first {m} of {A_ROWS}-row launch bitwise: {same}; \
                 guard rows untouched: {guard_ok}; known-bad (row {r} perturbed) row differs: \
                 {row_differs}, other rows unchanged: {others_same}"
            ),
        );

        // (b) per-row cosine against dequant + cuBLASLt, nonfinite count.
        let nonfinite = y[..m * n].iter().filter(|&&v| !bf(v).is_finite()).count();
        let (mut cos_min, mut sum, mut below) = (1f64, 0f64, 0usize);
        for i in 0..m {
            let cos = cosine(
                y[i * n..(i + 1) * n]
                    .iter()
                    .zip(&y_old[i * n..(i + 1) * n])
                    .map(|(&a, &b)| (bf(a), bf(b))),
            );
            below += usize::from(cos.is_nan() || cos < TOL_COS_PATH);
            cos_min = cos_min.min(cos);
            sum += cos;
        }
        ck.line(
            "GATE-B",
            below == 0 && nonfinite == 0,
            &format!(
                "{n}x{k} {label} M={m} vs dequant+cuBLASLt: cos_min={cos_min:.6} cos_mean={:.6} \
                 rows_below_{TOL_COS_PATH}={below} nonfinite={nonfinite}",
                sum / m as f64
            ),
        );
        outs.push(y[..m * n].to_vec());
        for p in [c_m, c_old, a_bad, c_bad] {
            g.free(p)?;
        }
    }
    for p in [c_full, deq] {
        g.free(p)?;
    }
    dir.free(g)?;
    Ok(outs)
}

/// 2026-10-06: (c) `route` at M = 17 and 63 (W8A8, bits == `direct[i]`) and M = 16 (GEMV tier).
#[allow(clippy::too_many_arguments)]
fn gate_c(
    g: &dyn GpuBackend,
    ck: &mut Checks,
    kn: &Kern,
    (key, ad): (DevicePtr, DevicePtr),
    direct: &[Vec<u16>],
    (n, k, label): (usize, usize, &str),
    s: u64,
) -> Result<()> {
    let c = g.alloc(A_ROWS * n * 2)?;
    // Indices into ROW_MS (and `direct`) of M = 17 and 63.
    for (i, m) in [(0, ROW_MS[0]), (3, ROW_MS[3])] {
        let (w0, d0) = (df::w8a8_gemms(), df::dequants());
        let r = df::route(g, kn.gemv, ad, key, c, m, n, k, s)?;
        g.synchronize(s)?;
        let same = dn_u16(g, c, m * n)? == direct[i];
        ck.line(
            "ROUTE",
            r == Route::Done && same && df::w8a8_gemms() == w0 + 1 && df::dequants() == d0,
            &format!(
                "{n}x{k} {label} M={m}: Done {} (want true), bits == direct launch: {same}, \
                 w8a8_gemms +{} (want 1), dequants +{} (want 0)",
                r == Route::Done,
                df::w8a8_gemms() - w0,
                df::dequants() - d0
            ),
        );
    }
    let (w0, h0) = (df::w8a8_gemms(), df::hits());
    let r16 = df::route(g, kn.gemv, ad, key, c, 16, n, k, s)?;
    g.synchronize(s)?;
    ck.line(
        "ROUTE",
        r16 == Route::Done && df::w8a8_gemms() == w0 && df::hits() == h0 + 1,
        &format!(
            "{n}x{k} {label} M=16: Done {} , w8a8_gemms +{} (want 0), FP8 GEMV hits +{} (want 1)",
            r16 == Route::Done,
            df::w8a8_gemms() - w0,
            df::hits() - h0
        ),
    );
    g.free(c)
}

fn main() -> Result<()> {
    // SAFETY: single-threaded here, before the backend exists and before anything reads a lever
    // (each is a OnceLock). The removals keep the M <= 16 tier on the FP8 GEMV.
    unsafe {
        std::env::set_var("METRALE_GLM_DENSE_FP8", "1");
        std::env::set_var("METRALE_GLM_DENSE_FP8_W8A8", "1");
        std::env::set_var("METRALE_GLM_DENSE_FP8_W8A8_MIN_ROWS", "17");
        std::env::set_var("METRALE_GLM_DENSE_FP8_W8A8_ROWS", A_ROWS.to_string());
        std::env::remove_var("METRALE_GLM_GEMV_TC");
        std::env::remove_var("METRALE_GLM_DENSE_NVFP4");
    }
    if df::w8a8_min_rows() != 17 {
        bail!(
            "w8a8_min_rows() = {} (want 17 from METRALE_GLM_DENSE_FP8_W8A8_MIN_ROWS=17)",
            df::w8a8_min_rows()
        );
    }
    println!("MIN_ROWS w8a8_min_rows() == 17  ok");
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kn = kern(g)?;
    let s = g.default_stream();
    let mut rng = Lcg(0x6d69_6e72_6f77_7331);
    let mut ck = Checks(0);
    ck.line(
        "ROUTE",
        df::dense_fp8() && df::dense_fp8_w8a8() && df::dense_fp8_w8a8_rows() == A_ROWS,
        "levers: DENSE_FP8 on, W8A8 on, scratch rows 128",
    );
    ensure!(
        ops::DENSE_GEMV_FP8W_BATCHM_MAX_M == 16,
        "GEMV tier is not 16 rows: stale M split"
    );

    let mut reg = Vec::new();
    for (n, k, _) in SHAPES {
        reg.push(register(g, &mut rng, n, k)?);
    }
    df::finish_load(g)?;
    ck.line(
        "ROUTE",
        df::w8a8_scratch_bytes() > 0,
        &format!(
            "finish_load: W8A8 scratch {} B (want > 0)",
            df::w8a8_scratch_bytes()
        ),
    );
    let flush = g.alloc(FLUSH_BYTES)?;
    let mut timing: Vec<(usize, f64, f64)> = Vec::new();
    for (i, &(n, k, label)) in SHAPES.iter().enumerate() {
        let (key, w) = reg[i];
        let ab = gen_act(&mut rng, A_ROWS * k);
        let ad = upload(g, &ab)?;
        let direct = gates_ab(g, &mut ck, &kn, &w, (ad, &ab), (n, k, label), s)?;
        gate_c(g, &mut ck, &kn, (key, ad), &direct, (n, k, label), s)?;

        // (d) timing: cold, median of REPS.
        let deq = g.alloc(n * k * 2)?;
        let c = g.alloc(A_ROWS * n * 2)?;
        for m in TIME_MS {
            let t_w8 = time_cold(g, s, flush, &mut || match df::route(
                g, kn.gemv, ad, key, c, m, n, k, s,
            )? {
                Route::Done => Ok(()),
                Route::Weight(_) => bail!("route did not take W8A8 at M={m}"),
            })?;
            let t_dq = time_cold(g, s, flush, &mut || {
                ops::dequant_fp8_rowscale_bf16(g, kn.dq, &w, deq, n as u32, k as u32, s)?;
                ops::cublas_bf16_proj_dense(ad, deq, c, m as u32, n as u32, k as u32, s)
            })?;
            println!(
                "TIMING shape={n}x{k} M={m}: dequant+cublas {t_dq:.4} ms  w8a8 {t_w8:.4} ms  \
                 speedup {:.3}  ({label})",
                t_dq / t_w8
            );
            timing.push((m, t_dq, t_w8));
        }
        for p in [ad, deq, c] {
            g.free(p)?;
        }
    }
    for m in TIME_MS {
        let (dq, w8) = timing
            .iter()
            .filter(|t| t.0 == m)
            .fold((0f64, 0f64), |a, t| (a.0 + t.1, a.1 + t.2));
        println!(
            "TIMING_SUM M={m}: dequant+cublas {dq:.4} ms  w8a8 {w8:.4} ms  speedup {:.3}  \
             (one launch per shape, {} shapes, not weighted by launches per layer)",
            dq / w8,
            SHAPES.len()
        );
    }
    g.free(flush)?;
    if ck.0 > 0 {
        println!("FAIL: {} checks failed (see FAIL-CHECK lines)", ck.0);
        bail!("{} checks failed", ck.0);
    }
    println!(
        "PASS: W8A8 at 17..=63 rows is row-invariant (bitwise == the first M rows of a 128-row \
         launch, known-bad detected), within tolerance of dequant+cuBLASLt, and \
         dense_fp8::route takes it at M=17/63 (bits == direct launch) while M=16 stays on the GEMV"
    );
    Ok(())
}
