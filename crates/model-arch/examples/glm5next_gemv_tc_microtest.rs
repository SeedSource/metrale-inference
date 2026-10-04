// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Gate for `METRALE_GLM_GEMV_TC=1`: the small-M tensor-core dense GEMVs
//! `dense_gemv_tcm_*` (`kernels/gb10/common/dense_gemv_tcm.cu`, launched through
//! `ops::dense_gemv_tcm`), BF16 weights and FP8 E4M3 weight-only (per-row FP32 scale).
//!
//! Owner: model-arch examples.
//! Checks, on every GLM-5.3 per-rank decode shape at TP2, the BF16 LM head's vocab slice
//! (77440 x 4096) and two edge shapes (N % 16 != 0), for both weight kinds:
//! - (a) Row invariance, raw u16 bits: row t at every M = 1..16 equals row t at M = 16; row r
//!   of a 16-row batch whose other rows hold different (and 64x larger) values equals row r
//!   of the base batch; every geometry variant of the kind equals the routed one at M = 16
//!   and M = 3; a second launch equals the first; nothing outside the M x N output is
//!   written (`out_stride = N + 8`, pad columns and row M hold a sentinel).
//! - (b) Numerics against an FP64 host reference, per output row, at M = 16, gaussian and
//!   heavy-tailed weights: BF16 kind against FP64 over the BF16 weights, FP8 kind against
//!   FP64 over the dequantized FP8 weights (e4m3 * row_scale): min cosine >= 0.9999,
//!   max_rel (max|y - ref| / max|ref|) <= 1.5 x the incumbent CUDA-core kernel's max_rel on
//!   the same reference (`dense_gemv_bf16_batchm` / `dense_gemv_fp8w_batchm`), nonfinite 0.
//! - (c) `glm5next_layer::dense_fp8::route` with both levers on: an unregistered BF16 weight
//!   and a registered FP8 weight at M = 1 and 4 run the TC kernel (bits == direct launch);
//!   M = 17 and a non-`dense_gemv_bf16` GEMV handle do not.
//! - (d) Timing (not gated): CUDA-graph replay over a cold pool of distinct weights (> 1 GiB
//!   BF16 per shape, far past the 24 MB L2), activations hot, at M = 1, 3, 4, 8, 11, 12, 16:
//!   the incumbents (BF16: `dense_gemv_bf16` at 1 row, `dense_gemv_bf16_batchm` above; FP8:
//!   `dense_gemv_fp8w` / `dense_gemv_fp8w_batchm`; LM head: `dense_gemv_bf16` /
//!   `dense_gemv_bf16_batchm` on the slice at 1..=8 rows, `dense_gemm_bf16` over the whole
//!   154880-row head at 9..=16 rows, as `lm_head_batched` runs them) against every TC
//!   variant; per-step sums over the GEMVs of one rank.
//!
//! Prints `PASS` iff (a), (b) and (c) pass.
//!
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_gemv_tc_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::layers::ops::dense_gemv_tcm::{self as tcm, TcmVariant};
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight};

/// 2026-10-04: Per-rank GLM-5.3 decode GEMV shapes at TP2 as (N, K, label, launches per
/// decode step per rank), as in glm5next_dense_fp8_microtest; then the two edge shapes.
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
    (4097, 1024, "N4097_K1024", 0),
    (37, 128, "N37_K128", 0),
];
/// 2026-10-04: The BF16 LM head at TP2: vocab 154880 split over two ranks.
const VOCAB: usize = 154880;
const HEAD_N: usize = VOCAB / 2;
const HIDDEN: usize = 4096;
const TIME_MS: &[usize] = &[1, 3, 4, 8, 11, 12, 16];
const MAXM: usize = 16;
const TOL_COS: f64 = 0.9999;
const TOL_REL_RATIO: f64 = 1.5;
const POOL_BYTES_BF16: usize = 1 << 30;
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

/// 2026-10-04: As glm5next_dense_fp8_microtest: N(0, 0.02); `heavy` adds a per-row gain
/// spread (x0.25..x4) and 0.1 % outliers at x30.
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

fn gen_act(rng: &mut Lcg, len: usize, scale: f64) -> Vec<bf16> {
    (0..len).map(|_| bf16::from_f64(rng.n() * scale)).collect()
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

/// 2026-10-04: OCP E4M3 decode (bias 7, subnormal m * 2^-9; 0x7F/0xFF are NaN).
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
    b1: KernelHandle,
    bbm: KernelHandle,
    gemm: KernelHandle,
    tc: [KernelHandle; 6],
}

impl Kern {
    fn h(&self, v: TcmVariant) -> KernelHandle {
        self.tc[TcmVariant::ALL.iter().position(|&x| x == v).unwrap()]
    }
}

fn quantize(
    g: &dyn GpuBackend,
    k: &Kern,
    w: DevicePtr,
    n: usize,
    kd: usize,
) -> Result<Fp8DenseWeight> {
    let dw = DenseWeight { weight: w };
    metrale_model_layers::weight_map::quantize_to_fp8(&dw, n, kd, g, k.q, g.default_stream())
}

/// 2026-10-04: One weight in both kinds: the BF16 weight and its FP8 copy.
struct Wt {
    bf: DevicePtr,
    q: Fp8DenseWeight,
}

#[allow(clippy::too_many_arguments)]
fn tc_launch(
    g: &dyn GpuBackend,
    k: &Kern,
    v: TcmVariant,
    a: DevicePtr,
    w: &Wt,
    c: DevicePtr,
    m: usize,
    n: usize,
    kd: usize,
    stride: usize,
    s: u64,
) -> Result<()> {
    let (wp, sp) = if v.is_fp8() {
        (w.q.weight, w.q.row_scale)
    } else {
        (w.bf, DevicePtr(0))
    };
    tcm::launch(
        g,
        k.h(v),
        v,
        a,
        wp,
        sp,
        c,
        m as u32,
        n as u32,
        kd as u32,
        stride as u32,
        s,
    )
}

/// 2026-10-04: The incumbent CUDA-core GEMV of a kind as the lever-off GLM path runs it.
#[allow(clippy::too_many_arguments)]
fn inc_launch(
    g: &dyn GpuBackend,
    k: &Kern,
    fp8: bool,
    a: DevicePtr,
    w: &Wt,
    c: DevicePtr,
    m: usize,
    n: usize,
    kd: usize,
    s: u64,
) -> Result<()> {
    if fp8 {
        if m == 1 {
            ops::dense_gemv_fp8w(g, k.f1, a, &w.q, c, n as u32, kd as u32, s)
        } else {
            ops::dense_gemv_fp8w_batchm(
                g, k.fbm, a, &w.q, c, m as u32, 1, n as u32, kd as u32, n as u32, s,
            )
        }
    } else {
        let dw = DenseWeight { weight: w.bf };
        if m == 1 {
            ops::dense_gemv(g, k.b1, a, &dw, c, n as u32, kd as u32, s)
        } else {
            ops::dense_gemv_batchm(
                g, k.bbm, a, &dw, c, m as u32, n as u32, kd as u32, n as u32, s,
            )
        }
    }
}

fn kinds() -> [(bool, &'static str, [TcmVariant; 3]); 2] {
    [
        (
            false,
            "bf16",
            [TcmVariant::Bf16, TcmVariant::Bf16Ku4, TcmVariant::Bf16Nt2],
        ),
        (
            true,
            "fp8",
            [TcmVariant::Fp8, TcmVariant::Fp8Ku8, TcmVariant::Fp8Nt2],
        ),
    ]
}

fn mk_weight(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    heavy: bool,
) -> Result<(Vec<bf16>, Wt)> {
    let w = gen_weight(rng, n, kd, heavy);
    let bf = up_bf16(g, &w)?;
    let q = quantize(g, k, bf, n, kd)?;
    Ok((w, Wt { bf, q }))
}

fn free_wt(g: &dyn GpuBackend, w: Wt) -> Result<()> {
    g.free(w.bf)?;
    g.free(w.q.weight)?;
    g.free(w.q.row_scale)?;
    Ok(())
}

/// (a) Row invariance + variants + determinism + no stray writes for one shape, both kinds.
fn invariance(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
) -> Result<usize> {
    let s = g.default_stream();
    let (_, w) = mk_weight(g, k, rng, n, kd, true)?;
    let x = gen_act(rng, MAXM * kd, 1.0);
    let y = gen_act(rng, MAXM * kd, 64.0);
    let xd = up_bf16(g, &x)?;
    let stride = n + 8;
    let cells = (MAXM + 1) * stride;
    let base = g.alloc(cells * 2)?;
    let out = g.alloc(cells * 2)?;
    let mut fails = 0;
    for (fp8, kname, vars) in kinds() {
        let rv = tcm::variant_for(n as u32, kd as u32, fp8);
        g.memset(base, 0xFF, cells * 2)?;
        tc_launch(g, k, rv, xd, &w, base, MAXM, n, kd, stride, s)?;
        g.synchronize(s)?;
        let b = dn_u16(g, base, cells)?;
        // Row-invariance over M, plus the sentinel outside [M, N].
        let (mut d_m, mut stray) = (0usize, 0usize);
        for m in 1..=MAXM {
            g.memset(out, 0xFF, cells * 2)?;
            tc_launch(g, k, rv, xd, &w, out, m, n, kd, stride, s)?;
            g.synchronize(s)?;
            let o = dn_u16(g, out, cells)?;
            for r in 0..=MAXM {
                for j in 0..stride {
                    let i = r * stride + j;
                    if r < m && j < n {
                        d_m += (o[i] != b[i]) as usize;
                    } else {
                        stray += (o[i] != SENT) as usize;
                    }
                }
            }
        }
        // Other rows' values: rows != r replaced by 64x larger, different values.
        let mut d_x = 0usize;
        for r in [0usize, 5, 8, 15] {
            let mut z = y.clone();
            z[r * kd..(r + 1) * kd].copy_from_slice(&x[r * kd..(r + 1) * kd]);
            let zd = up_bf16(g, &z)?;
            tc_launch(g, k, rv, zd, &w, out, MAXM, n, kd, stride, s)?;
            g.synchronize(s)?;
            let o = dn_u16(g, out, cells)?;
            d_x += (0..n)
                .filter(|&j| o[r * stride + j] != b[r * stride + j])
                .count();
            g.free(zd)?;
        }
        // Every variant of the kind == the routed one, at 16 and 3 rows; a repeat == the first.
        let mut d_v = 0usize;
        for &v in &vars {
            for m in [MAXM, 3] {
                tc_launch(g, k, v, xd, &w, out, m, n, kd, stride, s)?;
                g.synchronize(s)?;
                let o = dn_u16(g, out, cells)?;
                d_v += (0..m * stride)
                    .filter(|&i| i % stride < n && o[i] != b[i])
                    .count();
            }
        }
        let ok = d_m == 0 && stray == 0 && d_x == 0 && d_v == 0;
        fails += (!ok) as usize;
        println!(
            "INVARIANCE {label:<14} {kname:<4} routed={} M=1..16 row-vs-M16 diff={d_m} \
             other-rows-changed diff={d_x} variants-vs-routed diff={d_v} stray-writes={stray}  {}",
            rv.entry(),
            if ok { "ok" } else { "MISMATCH" }
        );
    }
    for p in [xd, base, out] {
        g.free(p)?;
    }
    free_wt(g, w)?;
    Ok(fails)
}

/// Per output row: (cos, max_rel, nonfinite) against an FP64 reference row.
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

/// FP64 reference `[m, n]` of `a[m, kd] @ w^T`, `w[j]` given by `wf(j, i)`; one thread per row.
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

/// (b) Numerics for one shape at M = 16. Returns the number of failing kinds.
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
    let m = MAXM;
    let (wh, w) = mk_weight(g, k, rng, n, kd, heavy)?;
    let a = gen_act(rng, m * kd, 1.0);
    let ad = up_bf16(g, &a)?;
    let c_tc = g.alloc(m * n * 2)?;
    let c_in = g.alloc(m * n * 2)?;
    let qb = dn_bytes(g, w.q.weight, n * kd)?;
    let qs: Vec<f64> = dn_u32(g, w.q.row_scale, n)?
        .iter()
        .map(|&b| f32::from_bits(b) as f64)
        .collect();
    let lut: Vec<f64> = (0..256).map(|b| e4m3(b as u8)).collect();
    let nan_codes = qb.iter().filter(|&&b| lut[b as usize].is_nan()).count();
    let r_bf = reference(&a, m, n, kd, &|j, i| wh[j * kd + i].to_f64());
    let r_q = reference(&a, m, n, kd, &|j, i| lut[qb[j * kd + i] as usize] * qs[j]);
    let mut fails = 0;
    for (fp8, kname, _) in kinds() {
        let rv = tcm::variant_for(n as u32, kd as u32, fp8);
        tc_launch(g, k, rv, ad, &w, c_tc, m, n, kd, n, s)?;
        inc_launch(g, k, fp8, ad, &w, c_in, m, n, kd, s)?;
        g.synchronize(s)?;
        let f =
            |v: Vec<u16>| -> Vec<f64> { v.iter().map(|&b| bf16::from_bits(b).to_f64()).collect() };
        let yt = f(dn_u16(g, c_tc, m * n)?);
        let yi = f(dn_u16(g, c_in, m * n)?);
        let r = if fp8 { &r_q } else { &r_bf };
        let (mut ct, mut rt, mut ci, mut ri, mut cb, mut nf) =
            (1.0f64, 0.0f64, 1.0f64, 0.0f64, 1.0f64, 0usize);
        for t in 0..m {
            let sl = t * n..(t + 1) * n;
            let (c1, r1, n1) = row_stats(&yt[sl.clone()], &r[sl.clone()]);
            let (c2, r2, n2) = row_stats(&yi[sl.clone()], &r[sl.clone()]);
            let (c3, _, _) = row_stats(&yt[sl.clone()], &r_bf[sl]);
            ct = ct.min(c1);
            rt = rt.max(r1);
            ci = ci.min(c2);
            ri = ri.max(r2);
            cb = cb.min(c3);
            nf += n1 + n2;
        }
        let ok = ct >= TOL_COS && rt <= TOL_REL_RATIO * ri && nf == 0 && nan_codes == 0;
        fails += (!ok) as usize;
        println!(
            "NUMERICS {label:<14} {kname:<4} {} M={m} vs fp64({}): tc min_cos={ct:.8} max_rel={rt:.6} | \
             incumbent min_cos={ci:.8} max_rel={ri:.6} | ratio {:.3} | tc vs fp64(bf16 W) min_cos={cb:.6} | \
             nonfinite={nf} nan_codes={nan_codes}  {}",
            if heavy { "heavy" } else { "gauss" },
            if fp8 { "own fp8 W" } else { "bf16 W" },
            rt / ri.max(1e-30),
            if ok { "ok" } else { "OUT-OF-TOL" }
        );
    }
    for p in [ad, c_tc, c_in] {
        g.free(p)?;
    }
    free_wt(g, w)?;
    Ok(fails)
}

/// (c) `dense_fp8::route` with both levers on.
fn route_check(g: &dyn GpuBackend, k: &Kern, rng: &mut Lcg) -> Result<usize> {
    use metrale_model_arch::glm5next_layer::dense_fp8::{self as df, LayerFp8, Route};
    let s = g.default_stream();
    let mut fails = 0usize;
    let mut check = |ok: bool, what: &str| {
        fails += (!ok) as usize;
        println!("ROUTE {what}  {}", if ok { "ok" } else { "FAIL-CHECK" });
    };
    let (n, kd) = (1024usize, 2048usize);
    let a = gen_act(rng, 17 * kd, 1.0);
    let ad = up_bf16(g, &a)?;
    let c1 = g.alloc(17 * n * 2)?;
    let c2 = g.alloc(17 * n * 2)?;
    // Unregistered BF16 weight.
    let wb = up_bf16(g, &gen_weight(rng, n, kd, true))?;
    let wt_b = Wt {
        bf: wb,
        q: Fp8DenseWeight {
            weight: DevicePtr(0),
            row_scale: DevicePtr(0),
        },
    };
    let vb = tcm::variant_for(n as u32, kd as u32, false);
    for m in [1usize, 4, 16] {
        let r = df::route(g, k.b1, ad, wb, c1, m, n, kd, s)?;
        tc_launch(g, k, vb, ad, &wt_b, c2, m, n, kd, n, s)?;
        g.synchronize(s)?;
        check(
            r == Route::Done && dn_u16(g, c1, m * n)? == dn_u16(g, c2, m * n)?,
            &format!("BF16 weight M={m}: Done, bits == direct {}", vb.entry()),
        );
    }
    let r17 = df::route(g, k.b1, ad, wb, c1, 17, n, kd, s)?;
    check(
        r17 == Route::Weight(wb),
        "BF16 weight M=17: caller's path (Weight(b))",
    );
    let ro = df::route(g, k.bbm, ad, wb, c1, 4, n, kd, s)?;
    check(
        ro == Route::Weight(wb),
        "BF16 weight, non-dense_gemv_bf16 handle: caller's path",
    );
    // Registered FP8 weight.
    let mut wq = up_bf16(g, &gen_weight(rng, n, kd, true))?;
    let mut acc = LayerFp8::default();
    let conv = df::convert_weight(g, &mut wq, n, kd, &mut acc)?;
    check(conv, "convert_weight registered the FP8 copy");
    let q = df::lookup(wq, n, kd).expect("registered");
    df::finish_load(g)?;
    let wt_q = Wt {
        bf: DevicePtr(0),
        q,
    };
    let vq = tcm::variant_for(n as u32, kd as u32, true);
    for m in [1usize, 4, 16] {
        let r = df::route(g, k.b1, ad, wq, c1, m, n, kd, s)?;
        tc_launch(g, k, vq, ad, &wt_q, c2, m, n, kd, n, s)?;
        g.synchronize(s)?;
        check(
            r == Route::Done && dn_u16(g, c1, m * n)? == dn_u16(g, c2, m * n)?,
            &format!("FP8 weight M={m}: Done, bits == direct {}", vq.entry()),
        );
    }
    let r17q = df::route(g, k.b1, ad, wq, c1, 17, n, kd, s)?;
    check(
        matches!(r17q, Route::Weight(p) if p != wq),
        "FP8 weight M=17: the BF16 dequant (Weight(arena))",
    );
    g.synchronize(s)?;
    for p in [ad, c1, c2, wb] {
        g.free(p)?;
    }
    Ok(fails)
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

/// One timing row: (M, incumbent ms, ms per variant in `kinds()` order).
type TRow = (usize, f64, [f64; 3]);

/// (d) Timing for one shape over a cold pool, both kinds. Returns [bf16 rows, fp8 rows].
fn timing(
    g: &dyn GpuBackend,
    k: &Kern,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
) -> Result<[Vec<TRow>; 2]> {
    let s = g.create_stream()?;
    let bytes = n * kd * 2;
    let pool = POOL_BYTES_BF16.div_ceil(bytes).clamp(4, 1024);
    let w = gen_weight(rng, n, kd, false);
    let host: Vec<u8> = w.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let mut ws = Vec::with_capacity(pool);
    for _ in 0..pool {
        let p = g.alloc(bytes)?;
        g.copy_h2d(&host, p)?;
        let q = quantize(g, k, p, n, kd)?;
        ws.push(Wt { bf: p, q });
    }
    let a = gen_act(rng, MAXM * kd, 1.0);
    let ad = up_bf16(g, &a)?;
    let c = g.alloc(MAXM * n * 2)?;
    let mut out: [Vec<TRow>; 2] = [Vec::new(), Vec::new()];
    for (ki, (fp8, kname, vars)) in kinds().into_iter().enumerate() {
        let wbytes = if fp8 { n * kd + 4 * n } else { bytes };
        let gbs = |ms: f64| wbytes as f64 / (ms * 1e-3) / 1e9;
        for &m in TIME_MS {
            let t_in = time_graph(g, s, &mut |s| {
                for w in &ws {
                    inc_launch(g, k, fp8, ad, w, c, m, n, kd, s)?;
                }
                Ok(())
            })? / pool as f64;
            let mut tv = [0.0f64; 3];
            for (vi, &v) in vars.iter().enumerate() {
                tv[vi] = time_graph(g, s, &mut |s| {
                    for w in &ws {
                        tc_launch(g, k, v, ad, w, c, m, n, kd, n, s)?;
                    }
                    Ok(())
                })? / pool as f64;
            }
            let rv = tcm::variant_for(n as u32, kd as u32, fp8);
            let t_r = tv[vars.iter().position(|&v| v == rv).unwrap()];
            println!(
                "TIMING {label:<14} {kname:<4} M={m:<2} incumbent {:>8.1} us {:>6.1} GB/s | tc(routed {}) \
                 {:>8.1} us {:>6.1} GB/s | speedup {:>5.2}x | variants {} {:.1} / {} {:.1} / {} {:.1} us  \
                 (pool {pool} x {:.1} MB)",
                t_in * 1e3,
                gbs(t_in),
                rv.entry(),
                t_r * 1e3,
                gbs(t_r),
                t_in / t_r,
                vars[0].entry(),
                tv[0] * 1e3,
                vars[1].entry(),
                tv[1] * 1e3,
                vars[2].entry(),
                tv[2] * 1e3,
                wbytes as f64 / 1e6,
            );
            out[ki].push((m, t_in, tv));
        }
    }
    for w in ws {
        free_wt(g, w)?;
    }
    g.free(ad)?;
    g.free(c)?;
    Ok(out)
}

/// (d) LM head timing: whole BF16 head per pool slot; the slice is its first HEAD_N rows.
fn head_timing(g: &dyn GpuBackend, k: &Kern, rng: &mut Lcg) -> Result<Vec<TRow>> {
    let s = g.create_stream()?;
    let (n, kd) = (HEAD_N, HIDDEN);
    let full = VOCAB * kd * 2;
    let pool = 4usize;
    let w = gen_weight(rng, 1024, kd, false);
    let host: Vec<u8> = w.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let mut ws = Vec::with_capacity(pool);
    for _ in 0..pool {
        let p = g.alloc(full)?;
        let mut off = 0;
        while off < full {
            let len = host.len().min(full - off);
            g.copy_h2d(&host[..len], p.offset(off))?;
            off += len;
        }
        ws.push(p);
    }
    let a = gen_act(rng, MAXM * kd, 1.0);
    let ad = up_bf16(g, &a)?;
    let c = g.alloc(MAXM * VOCAB * 2)?;
    let vars = [TcmVariant::Bf16, TcmVariant::Bf16Ku4, TcmVariant::Bf16Nt2];
    let mut out = Vec::new();
    let slice_bytes = (n * kd * 2) as f64;
    for &m in TIME_MS {
        let t_in = time_graph(g, s, &mut |s| {
            for &p in &ws {
                let dw = DenseWeight { weight: p };
                if m == 1 {
                    ops::dense_gemv(g, k.b1, ad, &dw, c, n as u32, kd as u32, s)?;
                } else if m <= ops::DENSE_GEMV_BATCHM_DECODE_MAX_M as usize {
                    ops::dense_gemv_batchm(
                        g,
                        k.bbm,
                        ad,
                        &dw,
                        c,
                        m as u32,
                        n as u32,
                        kd as u32,
                        VOCAB as u32,
                        s,
                    )?;
                } else {
                    ops::dense_gemm(g, k.gemm, ad, &dw, c, m as u32, VOCAB as u32, kd as u32, s)?;
                }
            }
            Ok(())
        })? / pool as f64;
        let mut tv = [0.0f64; 3];
        for (vi, &v) in vars.iter().enumerate() {
            tv[vi] = time_graph(g, s, &mut |s| {
                for &p in &ws {
                    tcm::launch(
                        g,
                        k.h(v),
                        v,
                        ad,
                        p,
                        DevicePtr(0),
                        c,
                        m as u32,
                        n as u32,
                        kd as u32,
                        VOCAB as u32,
                        s,
                    )?;
                }
                Ok(())
            })? / pool as f64;
        }
        let rv = tcm::variant_for(n as u32, kd as u32, false);
        let t_r = tv[vars.iter().position(|&v| v == rv).unwrap()];
        println!(
            "TIMING lm_head_tp2   bf16 M={m:<2} incumbent({}) {:>8.1} us | tc(routed {}, slice) {:>8.1} us \
             {:>6.1} GB/s | speedup {:>5.2}x | variants {:.1} / {:.1} / {:.1} us  (pool {pool} x {:.1} MB head)",
            if m == 1 {
                "dense_gemv_bf16 slice"
            } else if m <= 8 {
                "batchm slice"
            } else {
                "dense_gemm_bf16 whole head"
            },
            t_in * 1e3,
            rv.entry(),
            t_r * 1e3,
            slice_bytes / (t_r * 1e-3) / 1e9,
            t_in / t_r,
            tv[0] * 1e3,
            tv[1] * 1e3,
            tv[2] * 1e3,
            full as f64 / 1e6,
        );
        out.push((m, t_in, tv));
    }
    for p in ws {
        g.free(p)?;
    }
    g.free(ad)?;
    g.free(c)?;
    Ok(out)
}

fn main() -> Result<()> {
    // SAFETY: single-threaded here, before anything reads the environment. (c) needs both
    // levers on; (a), (b) and (d) launch the kernels directly and do not read them.
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
        f1: g.kernel("gemv_fp8w", "dense_gemv_fp8w")?,
        fbm: g.kernel("dense_gemv_fp8w_batchm", "dense_gemv_fp8w_batchm")?,
        b1: g.kernel("gemv", "dense_gemv_bf16")?,
        bbm: g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
        gemm: g.kernel("gemm", "dense_gemm_bf16")?,
        tc,
    };
    let mut rng = Lcg(0x7463_6d35_3367);
    let mut all: Vec<(usize, usize, &str)> =
        SHAPES.iter().map(|&(n, kd, l, _)| (n, kd, l)).collect();
    all.push((HEAD_N, HIDDEN, "lm_head_tp2"));

    let mut inv_fail = 0;
    for &(n, kd, label) in &all {
        inv_fail += invariance(g, &k, &mut rng, n, kd, label)?;
    }
    let mut num_fail = 0;
    for &(n, kd, label) in &all {
        for heavy in [false, true] {
            num_fail += numerics(g, &k, &mut rng, n, kd, label, heavy)?;
        }
    }
    let rt_fail = route_check(g, &k, &mut rng)?;

    // (d) Timing and per-step sums.
    let mut rows: Vec<(usize, [Vec<TRow>; 2])> = Vec::new();
    for &(n, kd, label, per_step) in SHAPES {
        if per_step > 0 {
            rows.push((per_step, timing(g, &k, &mut rng, n, kd, label)?));
        }
    }
    let head = head_timing(g, &k, &mut rng)?;
    for (mi, &m) in TIME_MS.iter().enumerate() {
        for (ki, (_, kname, vars)) in kinds().into_iter().enumerate() {
            let inc: f64 = rows.iter().map(|(ps, r)| *ps as f64 * r[ki][mi].1).sum();
            let routed: f64 = rows
                .iter()
                .zip(SHAPES.iter().filter(|x| x.3 > 0))
                .map(|((ps, r), &(n, kd, _, _))| {
                    let rv = tcm::variant_for(n as u32, kd as u32, ki == 1);
                    *ps as f64 * r[ki][mi].2[vars.iter().position(|&v| v == rv).unwrap()]
                })
                .sum();
            let best: f64 = rows
                .iter()
                .map(|(ps, r)| *ps as f64 * r[ki][mi].2.iter().cloned().fold(f64::MAX, f64::min))
                .sum();
            println!(
                "SUMMARY M={m:<2} {kname:<4} per-step dense GEMVs of one rank (launch counts above): \
                 incumbent {inc:.2} ms -> tc routed {routed:.2} ms (best variant per shape {best:.2} ms), \
                 saves {:.2} ms ({:.2}x)",
                inc - routed,
                inc / routed
            );
        }
        let (hm, hi, hv) = head[mi];
        let hvars = [TcmVariant::Bf16, TcmVariant::Bf16Ku4, TcmVariant::Bf16Nt2];
        let hrv = tcm::variant_for(HEAD_N as u32, HIDDEN as u32, false);
        let hr = hv[hvars.iter().position(|&v| v == hrv).unwrap()];
        println!(
            "SUMMARY M={hm:<2} lm_head one verify head per step: incumbent {:.2} ms -> tc {:.2} ms, saves {:.2} ms",
            hi,
            hr,
            hi - hr
        );
    }
    println!(
        "tolerances: tc vs fp64 (own weights) min_cos>={TOL_COS}, max_rel<={TOL_REL_RATIO} x the \
         incumbent's max_rel on the same reference, nonfinite=0; invariance/variants bitwise (raw u16)"
    );
    ensure!(
        inv_fail == 0,
        "row invariance: {inv_fail} shape/kind cases differ (see MISMATCH lines)"
    );
    ensure!(
        num_fail == 0,
        "numerics: {num_fail} shape/kind cases out of tolerance (see OUT-OF-TOL lines)"
    );
    ensure!(
        rt_fail == 0,
        "dense_fp8::route with METRALE_GLM_GEMV_TC: {rt_fail} checks failed (see FAIL-CHECK lines)"
    );
    println!(
        "PASS: dense_gemv_tcm row-invariant (bitwise, M 1..16, other rows, variants, repeat, no stray writes) on every GLM-5.3 shape + LM head slice, both kinds; numerics within tolerance vs FP64; dense_fp8::route routes 1..16 rows to it"
    );
    Ok(())
}
