// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate for `METRALE_GLM_KDA_FAB_FP8=1`: the GLM-5.3 KDA `f_a` (128 x 4096), `f_b`
//! (4096 x 128) and `b_proj` (32 x 4096) as per-row E4M3 copies on the tensor-core FP8 GEMV
//! (`dense_gemv_tcm_fp8w`), against the BF16 `dense_gemv_tcm_bf16` they run on today.
//!
//! Owner: model-arch examples.
//! Checks, at M = 1, 3, 8, 9, 16 rows (decode, MTP verify, DFlash verify, MB16):
//! - (1) Correctness. The weights are registered the way the loader registers them
//!   (`dense_fp8::convert_weight`: `quantize_bf16_to_fp8`, the quantizer `g_a` / `g_b` and the
//!   `kda_fb:fp8` fake-quant use) and the output is `dense_fp8::route` with both levers on, the
//!   call the KDA `gemm` makes. Against an FP64 host reference over the DEQUANTIZED FP8 weights
//!   (e4m3 * row scale): per output row cosine >= 0.99999 and nonfinite 0, gaussian and
//!   heavy-tailed weights; the route output must equal the direct `dense_gemv_tcm_fp8w` launch
//!   bit for bit. Also printed (INFO, not gated): cosine against the original BF16 weights.
//! - (2) Determinism, on `g_a`'s shape (128 x 4096): the same `dense_gemv_tcm_fp8w` launch five
//!   times into sentinel-filled buffers gives identical u16 bits every time, and nothing outside
//!   the M x N output is written.
//! - (3) Speed (not gated): cold pool of distinct weight copies (>= 256 MB of BF16 per shape,
//!   far past the 24 MB L2), one CUDA graph per row, median of 21 replays, old = BF16
//!   `dense_gemv_tcm_bf16`, new = `dense_gemv_tcm_fp8w`. Prints per row
//!   `M=<m> old_us=<..> new_us=<..> old_GBs=<..> new_GBs=<..>` (us per launch; GB/s over the
//!   bytes the kernel streams: 2 N K for BF16, N K + 4 N for FP8); a rate above 255 GB/s (impossible on
//!   GB10) prints VOID in its place. These shapes are launch-bound, so read the us columns.
//!
//! Final line `PASS all` iff every check passed.
//!
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm_kda_fab_fp8_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_layer::dense_fp8::{self as df, LayerFp8, Route};
use metrale_model_layers::layers::ops::dense_gemv_tcm::{self as tcm, TcmVariant};
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight};

/// (N, K, label): the per-rank KDA shapes at TP2, then `g_a` (the determinism shape).
const SHAPES: &[(usize, usize, &str)] = &[
    (128, 4096, "f_a"),
    (4096, 128, "f_b"),
    (32, 4096, "b_proj"),
    (128, 4096, "g_a"),
];
const MS: &[usize] = &[1, 3, 8, 9, 16];
const MAXM: usize = 16;
const TOL_COS: f64 = 0.99999;
const POOL_BYTES: usize = 256 << 20;
const REPLAYS: usize = 21;
const GBS_MAX: f64 = 255.0;
const SENT: u16 = 0xFFFF;

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

/// N(0, 0.02); `heavy` adds a per-row gain spread (x0.25..x4) and 0.1 % outliers at x30.
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

/// OCP E4M3 decode (bias 7, subnormal m * 2^-9; 0x7F/0xFF are NaN).
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

/// Per output row (one token): (cosine, nonfinite count) of `y` against the FP64 row `r`.
fn row_cos(y: &[f64], r: &[f64]) -> (f64, usize) {
    let (mut dot, mut yy, mut rr) = (0.0, 0.0, 0.0);
    let mut nonfinite = 0;
    for (&a, &b) in y.iter().zip(r) {
        if !a.is_finite() {
            nonfinite += 1;
            continue;
        }
        dot += a * b;
        yy += a * a;
        rr += b * b;
    }
    let cos = if yy > 0.0 && rr > 0.0 {
        dot / (yy.sqrt() * rr.sqrt())
    } else if yy == 0.0 && rr == 0.0 {
        1.0
    } else {
        0.0
    };
    (cos, nonfinite)
}

/// FP64 reference `[m, n]` of `a[m, kd] @ w^T`, `w[j][i]` given by `wf(j, i)`.
fn reference(
    a: &[bf16],
    m: usize,
    n: usize,
    kd: usize,
    wf: &(dyn Fn(usize, usize) -> f64 + Sync),
) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    std::thread::scope(|sc| {
        for (t, row) in out.chunks_mut(n).enumerate() {
            sc.spawn(move || {
                let at: Vec<f64> = a[t * kd..(t + 1) * kd].iter().map(|x| x.to_f64()).collect();
                for (j, o) in row.iter_mut().enumerate() {
                    let mut acc = 0.0f64;
                    for (i, &ai) in at.iter().enumerate() {
                        acc += ai * wf(j, i);
                    }
                    *o = acc;
                }
            });
        }
    });
    out
}

struct Kern {
    q: KernelHandle,
    b1: KernelHandle,
    tc: [KernelHandle; 6],
}

impl Kern {
    fn h(&self, v: TcmVariant) -> KernelHandle {
        self.tc[TcmVariant::ALL.iter().position(|&x| x == v).unwrap()]
    }
}

struct Checks {
    fails: usize,
}
impl Checks {
    fn report(&mut self, ok: bool, what: &str) {
        self.fails += (!ok) as usize;
        println!("{} {what}", if ok { "PASS" } else { "FAIL" });
    }
}

/// Direct `dense_gemv_tcm_fp8w` launch on a registered weight, packed `[m, n]` output.
#[allow(clippy::too_many_arguments)]
fn fp8_direct(
    g: &dyn GpuBackend,
    k: &Kern,
    a: DevicePtr,
    q: &Fp8DenseWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    kd: usize,
    stride: usize,
    s: u64,
) -> Result<()> {
    let v = tcm::variant_for(n as u32, kd as u32, true);
    tcm::launch(
        g,
        k.h(v),
        v,
        a,
        q.weight,
        q.row_scale,
        c,
        m as u32,
        n as u32,
        kd as u32,
        stride as u32,
        s,
    )
}

/// (1) Correctness for one shape and weight family: registers like the loader, runs
/// `route`, compares with FP64 over the dequantized weights.
fn correctness(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    ck: &mut Checks,
    n: usize,
    kd: usize,
    label: &str,
    heavy: bool,
) -> Result<()> {
    let s = g.default_stream();
    let wh = gen_weight(rng, n, kd, heavy);
    let mut field = up_bf16(g, &wh)?;
    let mut acc = LayerFp8::default();
    let conv = df::convert_weight(g, &mut field, n, kd, &mut acc)?;
    ck.report(
        conv,
        &format!("{label} registered as an FP8 copy (convert_weight)"),
    );
    let q = df::lookup(field, n, kd).expect("registered");
    df::finish_load(g)?;
    let qb = dn_bytes(g, q.weight, n * kd)?;
    let qs: Vec<f64> = dn_u32(g, q.row_scale, n)?
        .iter()
        .map(|&b| f32::from_bits(b) as f64)
        .collect();
    let lut: Vec<f64> = (0..256).map(|b| e4m3(b as u8)).collect();
    let nan_codes = qb.iter().filter(|&&b| lut[b as usize].is_nan()).count();
    ck.report(
        nan_codes == 0,
        &format!("{label} FP8 copy holds no NaN code ({nan_codes})"),
    );
    let a = gen_act(rng, MAXM * kd);
    let ad = up_bf16(g, &a)?;
    let c_route = g.alloc(MAXM * n * 2)?;
    let c_direct = g.alloc(MAXM * n * 2)?;
    let r_q = reference(&a, MAXM, n, kd, &|j, i| {
        lut[qb[j * kd + i] as usize] * qs[j]
    });
    let r_b = reference(&a, MAXM, n, kd, &|j, i| wh[j * kd + i].to_f64());
    for &m in MS {
        let r = df::route(g, k.b1, ad, field, c_route, m, n, kd, s)?;
        fp8_direct(g, k, ad, &q, c_direct, m, n, kd, n, s)?;
        g.synchronize(s)?;
        let yr = dn_u16(g, c_route, m * n)?;
        let yd = dn_u16(g, c_direct, m * n)?;
        let f =
            |v: &[u16]| -> Vec<f64> { v.iter().map(|&b| bf16::from_bits(b).to_f64()).collect() };
        let y = f(&yr);
        let (mut min_cos, mut min_cos_bf, mut nf) = (1.0f64, 1.0f64, 0usize);
        for t in 0..m {
            let sl = t * n..(t + 1) * n;
            let (c1, n1) = row_cos(&y[sl.clone()], &r_q[sl.clone()]);
            let (c2, _) = row_cos(&y[sl.clone()], &r_b[sl]);
            min_cos = min_cos.min(c1);
            min_cos_bf = min_cos_bf.min(c2);
            nf += n1;
        }
        println!(
            "INFO {label} {} M={m} cosine vs fp64(original BF16 weights) min={min_cos_bf:.7}",
            if heavy { "heavy" } else { "gauss" }
        );
        ck.report(
            r == Route::Done && min_cos >= TOL_COS && nf == 0,
            &format!(
                "{label} {} M={m}: route Done, min_cos={min_cos:.8} (>= {TOL_COS}) vs fp64(dequantized FP8 W), nonfinite={nf}",
                if heavy { "heavy" } else { "gauss" }
            ),
        );
        ck.report(
            yr == yd,
            &format!(
                "{label} {} M={m}: route bits == direct dense_gemv_tcm_fp8w",
                if heavy { "heavy" } else { "gauss" }
            ),
        );
    }
    for p in [ad, c_route, c_direct] {
        g.free(p)?;
    }
    Ok(())
}

/// (2) Determinism of `dense_gemv_tcm_fp8w` on one shape.
fn determinism(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    ck: &mut Checks,
    n: usize,
    kd: usize,
    label: &str,
) -> Result<()> {
    let s = g.default_stream();
    let w = up_bf16(g, &gen_weight(rng, n, kd, true))?;
    let q = metrale_model_layers::weight_map::quantize_to_fp8(
        &DenseWeight { weight: w },
        n,
        kd,
        g,
        k.q,
        s,
    )?;
    let ad = up_bf16(g, &gen_act(rng, MAXM * kd))?;
    let stride = n + 8;
    let cells = (MAXM + 1) * stride;
    let c = g.alloc(cells * 2)?;
    for &m in MS {
        let mut first: Option<Vec<u16>> = None;
        let (mut diff, mut stray) = (0usize, 0usize);
        for _ in 0..5 {
            g.memset(c, 0xFF, cells * 2)?;
            fp8_direct(g, k, ad, &q, c, m, n, kd, stride, s)?;
            g.synchronize(s)?;
            let o = dn_u16(g, c, cells)?;
            for r in 0..=MAXM {
                for j in 0..stride {
                    if !(r < m && j < n) {
                        stray += (o[r * stride + j] != SENT) as usize;
                    }
                }
            }
            match &first {
                None => first = Some(o),
                Some(f) => {
                    diff += (0..cells)
                        .filter(|&i| i % stride < n && i / stride < m && o[i] != f[i])
                        .count();
                }
            }
        }
        ck.report(
            diff == 0 && stray == 0,
            &format!(
                "{label} M={m}: 5 repeat launches bitwise equal (diff={diff}), stray writes={stray}"
            ),
        );
    }
    for p in [ad, c, w, q.weight, q.row_scale] {
        g.free(p)?;
    }
    Ok(())
}

/// One CUDA graph over `launches`, replayed REPLAYS times; the median ms per graph.
fn time_graph_median(
    g: &dyn GpuBackend,
    s: u64,
    f: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<f64> {
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
    let mut t = Vec::with_capacity(REPLAYS);
    for _ in 0..REPLAYS {
        let t0 = std::time::Instant::now();
        g.launch_graph(graph, s)?;
        g.synchronize(s)?;
        t.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    g.destroy_graph(graph)?;
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(t[REPLAYS / 2])
}

fn rate(bytes: usize, us: f64) -> (f64, String) {
    let gbs = bytes as f64 / (us * 1e-6) / 1e9;
    (
        gbs,
        if gbs > GBS_MAX {
            "VOID".into()
        } else {
            format!("{gbs:.1}")
        },
    )
}

/// (3) Timing for one shape over a cold pool.
fn timing(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
) -> Result<()> {
    let s = g.create_stream()?;
    let bytes_bf = n * kd * 2;
    let bytes_q = n * kd + 4 * n;
    let pool = POOL_BYTES.div_ceil(bytes_bf).clamp(4, 1024);
    let host = gen_weight(rng, n, kd, false);
    let mut ws: Vec<(DevicePtr, Fp8DenseWeight)> = Vec::with_capacity(pool);
    for _ in 0..pool {
        let p = up_bf16(g, &host)?;
        let q = metrale_model_layers::weight_map::quantize_to_fp8(
            &DenseWeight { weight: p },
            n,
            kd,
            g,
            k.q,
            g.default_stream(),
        )?;
        ws.push((p, q));
    }
    let ad = up_bf16(g, &gen_act(rng, MAXM * kd))?;
    let c = g.alloc(MAXM * n * 2)?;
    let vb = tcm::variant_for(n as u32, kd as u32, false);
    let vq = tcm::variant_for(n as u32, kd as u32, true);
    for &m in MS {
        let t_old = time_graph_median(g, s, &mut |s| {
            for (p, _) in &ws {
                tcm::launch(
                    g,
                    k.h(vb),
                    vb,
                    ad,
                    *p,
                    DevicePtr(0),
                    c,
                    m as u32,
                    n as u32,
                    kd as u32,
                    n as u32,
                    s,
                )?;
            }
            Ok(())
        })? * 1e3
            / pool as f64;
        let t_new = time_graph_median(g, s, &mut |s| {
            for (_, q) in &ws {
                tcm::launch(
                    g,
                    k.h(vq),
                    vq,
                    ad,
                    q.weight,
                    q.row_scale,
                    c,
                    m as u32,
                    n as u32,
                    kd as u32,
                    n as u32,
                    s,
                )?;
            }
            Ok(())
        })? * 1e3
            / pool as f64;
        let (go, so) = rate(bytes_bf, t_old);
        let (gn, sn) = rate(bytes_q, t_new);
        println!(
            "TIMING {label:<7} N={n} K={kd} M={m:<2} old_us={t_old:.2} new_us={t_new:.2} old_GBs={so} new_GBs={sn} \
             speedup={:.2}x ({} vs {}; pool {pool} x {:.2} MB BF16, median of {REPLAYS} replays){}",
            t_old / t_new,
            vb.entry(),
            vq.entry(),
            bytes_bf as f64 / 1e6,
            if go > GBS_MAX || gn > GBS_MAX {
                "  VOID"
            } else {
                ""
            }
        );
    }
    for (p, q) in ws {
        g.free(p)?;
        g.free(q.weight)?;
        g.free(q.row_scale)?;
    }
    g.free(ad)?;
    g.free(c)?;
    Ok(())
}

fn main() -> Result<()> {
    // SAFETY: single-threaded here, before anything reads the environment. `route` needs both
    // levers on; the direct launches do not read them.
    unsafe {
        std::env::set_var("METRALE_GLM_GEMV_TC", "1");
        std::env::set_var("METRALE_GLM_DENSE_FP8", "1");
    }
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let tc = tcm::handles(g);
    for (v, h) in TcmVariant::ALL.iter().zip(tc) {
        ensure!(h.0 != 0, "{} did not resolve", v.entry());
    }
    let k = Kern {
        q: g.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?,
        b1: g.kernel("gemv", "dense_gemv_bf16")?,
        tc,
    };
    let mut rng = Lcg(0x6661_6238_3030);
    let mut ck = Checks { fails: 0 };
    for &(n, kd, label) in &SHAPES[..3] {
        ck.report(
            tcm::routes(1, n as u32, kd as u32, true),
            &format!(
                "{label} N={n} K={kd} is a shape dense_gemv_tcm_fp8w accepts (k % {} == 0)",
                tcm::K_STEP_FP8
            ),
        );
    }
    for &(n, kd, label) in SHAPES {
        for heavy in [false, true] {
            correctness(g, &k, &mut rng, &mut ck, n, kd, label, heavy)?;
        }
    }
    let (n, kd, label) = SHAPES[3];
    determinism(g, &k, &mut rng, &mut ck, n, kd, label)?;
    for &(n, kd, label) in SHAPES {
        timing(g, &k, &mut rng, n, kd, label)?;
    }
    ensure!(
        ck.fails == 0,
        "{} checks failed (see the lines starting FAIL)",
        ck.fails
    );
    println!("PASS all");
    Ok(())
}
