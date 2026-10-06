// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Gate for `METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW=1`: the CUTLASS Sm120 FP8
//! blockwise GEMM (`cuda/cutlass_fp8_blockwise_gemm.cu`, after CUTLASS example 87b, NVIDIA,
//! BSD-3-Clause) through the engine's dispatch `dense_fp8_gw::gemm` at the GLM-5.3 TP2 shapes.
//!
//! Owner: model-arch examples.
//! Arms per (N, K) and M, from one BF16 weight and one BF16 activation: (i) cuBLASLt BF16
//! (`ops::cublas_bf16_proj_dense`); (ii) incumbent per-row FP8 + `fp8_gemm_t_rowscale`; (iii)
//! block-scaled FP8 (`dense_fp8_gw::quantize_block_scaled`) + `dense_fp8_gw::gemm` (K 16384 in
//! 2048-row launches; must report `GwPath::Cutlass`). (ii) and (iii) share the
//! `per_token_group_quant_fp8` activations. A case passes when cos(iii, i) and the minimum
//! cosine over 128-column blocks (`blk_min`) are >= `TOL_COS`, cos(iii, i) >= cos(ii, i) -
//! `TOL_VS_INC`, nothing is nonfinite and the guard row after the output is untouched; max_rel
//! is reported. KNOWN_BAD (M 257): block scale (0, 0) doubled must drop blk_min below `TOL_COS`.
//! A child (`--phase-route`, levers on) checks `dense_fp8::route` end to end (block-scaled
//! registration, bits equal to a direct `gemm`, M 1 on the NVFP4 GEMV). Timing (not gated): warm
//! median of `REPS` GEMMs at M 8191, (iii) vs (ii), and totals weighted by calls per forward on
//! one TP2 rank (fixture config: 34 KDA, 11 DSA, 42 shared-expert layers). Last line `PASS:` or
//! `FAIL:` (nonzero exit).
//!
//! Run (GB10, image built with CUTLASS_HOME):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dense_fp8_cutlass_gw_microtest

use anyhow::{Result, bail, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_layer::dense_fp8_gw::{self as gw, GwPath};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight};
use std::time::Instant;

/// 2026-10-06: (N, K, label, calls per forward on one TP2 rank).
const SHAPES: [(usize, usize, &str, usize); 5] = [
    (4096, 4096, "kda_qkvo", 4 * 34),
    (16384, 1536, "dsa_q_absorb", 11),
    (4096, 16384, "dsa_o_absorb", 11),
    (1024, 4096, "shexp_gate_up", 2 * 42),
    (4096, 1024, "shexp_down", 42),
];
const MS: [usize; 4] = [64, 257, 2048, 8191];
const MAX_M: usize = 8191;
const TIME_M: usize = 8191;
const BAD_M: usize = 257;
const TOL_COS: f64 = 0.999;
const TOL_VS_INC: f64 = 1e-4;
const WARMUP: usize = 3;
const REPS: usize = 10;
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

/// 2026-10-06: Weights N(0, 0.02) with a per-row gain (x0.25..x4) and 0.1 % outliers at x30
/// (`glm5next_dense_w8a8_microtest`'s generator).
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

/// 2026-10-06: Activations uniform in [-1, 1) with 0.1 % outliers at x20.
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

/// 2026-10-06: (dot, |x|^2, |y|^2) accumulators turned into a cosine (NaN on a zero norm).
fn cos_of(d: f64, na: f64, nb: f64) -> f64 {
    if na > 0.0 && nb > 0.0 {
        d / (na.sqrt() * nb.sqrt())
    } else {
        f64::NAN
    }
}

/// 2026-10-06: Output metrics of `y` against `r` ([m, n] BF16): (cos, blk_min, max_rel,
/// nonfinite in y).
fn metrics(y: &[u16], r: &[u16], m: usize, n: usize) -> (f64, f64, f64, usize) {
    let blocks = n.div_ceil(KG);
    let mut acc = vec![(0f64, 0f64, 0f64); blocks];
    let (mut err_max, mut ref_max, mut nonfin) = (0f64, 0f64, 0usize);
    for i in 0..m {
        for j in 0..n {
            let (a, b) = (bf(y[i * n + j]), bf(r[i * n + j]));
            if !a.is_finite() {
                nonfin += 1;
                continue;
            }
            let e = &mut acc[j / KG];
            e.0 += a * b;
            e.1 += a * a;
            e.2 += b * b;
            err_max = err_max.max((a - b).abs());
            ref_max = ref_max.max(b.abs());
        }
    }
    let (d, na, nb) = acc
        .iter()
        .fold((0f64, 0f64, 0f64), |s, e| (s.0 + e.0, s.1 + e.1, s.2 + e.2));
    let blk_min = acc
        .iter()
        .map(|e| cos_of(e.0, e.1, e.2))
        .fold(1f64, |a, c| {
            if a.is_nan() || c.is_nan() {
                f64::NAN
            } else {
                a.min(c)
            }
        });
    (cos_of(d, na, nb), blk_min, err_max / ref_max, nonfin)
}

/// 2026-10-06: Warm median ms of `f` over `REPS` launches, each followed by a synchronize.
fn median_ms(g: &dyn GpuBackend, s: u64, f: &mut dyn FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        f()?;
    }
    g.synchronize(s)?;
    let mut t = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = Instant::now();
        f()?;
        g.synchronize(s)?;
        t.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    t.sort_by(f64::total_cmp);
    Ok(t[REPS / 2])
}

struct Kern {
    quant: ops::Fp8ActQuant,
    rs: KernelHandle,
    wq: KernelHandle,
}

/// 2026-10-06: Per-shape device state.
struct Case {
    n: usize,
    k: usize,
    w_bf16: DevicePtr,
    w_row: Fp8DenseWeight,
    w_bs: Fp8DenseWeight,
    a: DevicePtr,
    a_q: DevicePtr,
    a_s: DevicePtr,
    ones: DevicePtr,
}

impl Case {
    fn new(g: &dyn GpuBackend, kn: &Kern, rng: &mut Lcg, n: usize, k: usize) -> Result<Self> {
        let s = g.default_stream();
        let w_bf16 = upload(g, &gen_weight(rng, n, k))?;
        let w_row = metrale_model_layers::weight_map::quantize_to_fp8(
            &DenseWeight { weight: w_bf16 },
            n,
            k,
            g,
            kn.wq,
            s,
        )?;
        let w_bs = gw::quantize_block_scaled(g, w_bf16, n, k, s)?;
        let a = upload(g, &gen_act(rng, MAX_M * k))?;
        let ones: Vec<u8> = (0..k / KG).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        Ok(Self {
            n,
            k,
            w_bf16,
            w_row,
            w_bs,
            a,
            a_q: g.alloc(MAX_M * k)?,
            a_s: g.alloc(MAX_M * (k / KG) * 4)?,
            ones: upload(g, &ones)?,
        })
    }
    fn quant(&self, g: &dyn GpuBackend, kn: &Kern, m: usize) -> Result<()> {
        let (a, q, sc, s) = (self.a, self.a_q, self.a_s, g.default_stream());
        ops::per_token_group_quant_fp8(g, kn.quant, a, q, sc, m as u32, self.k as u32, s)
    }
    fn incumbent(&self, g: &dyn GpuBackend, kn: &Kern, c: DevicePtr, m: usize) -> Result<()> {
        ops::fp8_gemm_t_rowscale(
            g,
            kn.rs,
            self.a_q,
            self.a_s,
            self.ones,
            self.w_row.weight,
            self.w_row.row_scale,
            c,
            m as u32,
            self.n as u32,
            self.k as u32,
            g.default_stream(),
        )
    }
    /// The engine's dispatch (`dense_fp8_gw::gemm`) over weight `w`.
    fn new_path(
        &self,
        g: &dyn GpuBackend,
        w: &Fp8DenseWeight,
        c: DevicePtr,
        m: usize,
    ) -> Result<GwPath> {
        let (q, sc, s) = (self.a_q, self.a_s, g.default_stream());
        gw::gemm(g, q, sc, w, c, m, self.n, self.k, s)
    }
    fn free(self, g: &dyn GpuBackend) -> Result<()> {
        let (wr, wb) = (self.w_row, self.w_bs);
        let ps = [
            self.w_bf16,
            wr.weight,
            wr.row_scale,
            wb.weight,
            wb.row_scale,
            self.a,
        ];
        ps.into_iter()
            .chain([self.a_q, self.a_s, self.ones])
            .try_for_each(|p| g.free(p))
    }
}

/// 2026-10-06: One (shape, M) numerics case. Returns (passed, cos_new, blk_min).
fn check(
    g: &dyn GpuBackend,
    kn: &Kern,
    cs: &Case,
    m: usize,
    label: &str,
) -> Result<(bool, f64, f64)> {
    let (n, k) = (cs.n, cs.k);
    let s = g.default_stream();
    let c_ref = g.alloc(m * n * 2)?;
    let c_old = g.alloc(m * n * 2)?;
    let c_new = g.alloc((m + 1) * n * 2)?;
    g.synchronize(s)?;
    g.memset(c_new, 0xFF, (m + 1) * n * 2)?;
    ops::cublas_bf16_proj_dense(cs.a, cs.w_bf16, c_ref, m as u32, n as u32, k as u32, s)?;
    cs.quant(g, kn, m)?;
    cs.incumbent(g, kn, c_old, m)?;
    let path = cs.new_path(g, &cs.w_bs, c_new, m)?;
    g.synchronize(s)?;
    let r = dn_u16(g, c_ref, m * n)?;
    let y_old = dn_u16(g, c_old, m * n)?;
    let y = dn_u16(g, c_new, (m + 1) * n)?;
    let guard_ok = y[m * n..].iter().all(|&v| v == 0xFFFF);
    let (cos, blk, rel, nonfin) = metrics(&y[..m * n], &r, m, n);
    let (cos_old, _, rel_old, _) = metrics(&y_old, &r, m, n);
    let ok = path == GwPath::Cutlass
        && cos >= TOL_COS
        && blk >= TOL_COS
        && cos >= cos_old - TOL_VS_INC
        && nonfin == 0
        && guard_ok;
    println!(
        "CASE m={m} n={n} k={k} {label} path={path:?} launches={} cos_new={cos:.6} \
         blk_min={blk:.6} max_rel={rel:.3e} cos_incumbent={cos_old:.6} \
         max_rel_incumbent={rel_old:.3e} nonfinite={nonfin} guard={guard_ok}  {}",
        gw::plan(m, n, k).len(),
        if ok { "ok" } else { "FAIL-CHECK" }
    );
    for p in [c_ref, c_old, c_new] {
        g.free(p)?;
    }
    Ok((ok, cos, blk))
}

/// 2026-10-06: KNOWN_BAD: block scale (0, 0) doubled must be caught (blk_min < `TOL_COS`).
fn known_bad(g: &dyn GpuBackend, kn: &Kern, cs: &Case, label: &str) -> Result<bool> {
    let (n, k, m) = (cs.n, cs.k, BAD_M);
    let s = g.default_stream();
    let sb = n.div_ceil(KG) * k.div_ceil(KG) * 4;
    let mut scales = vec![0u8; sb];
    g.copy_d2h(cs.w_bs.row_scale, &mut scales)?;
    let v = f32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]) * 2.0;
    scales[..4].copy_from_slice(&v.to_le_bytes());
    let bad = Fp8DenseWeight {
        weight: cs.w_bs.weight,
        row_scale: upload(g, &scales)?,
    };
    let c_ref = g.alloc(m * n * 2)?;
    let c_bad = g.alloc(m * n * 2)?;
    ops::cublas_bf16_proj_dense(cs.a, cs.w_bf16, c_ref, m as u32, n as u32, k as u32, s)?;
    cs.quant(g, kn, m)?;
    cs.new_path(g, &bad, c_bad, m)?;
    g.synchronize(s)?;
    let (cos, blk, _, _) = metrics(&dn_u16(g, c_bad, m * n)?, &dn_u16(g, c_ref, m * n)?, m, n);
    let caught = blk.is_nan() || blk < TOL_COS;
    let verdict = if caught {
        "caught ok"
    } else {
        "NOT CAUGHT FAIL-CHECK"
    };
    println!(
        "KNOWN_BAD m={m} n={n} k={k} {label} block(0,0) scale x2: cos={cos:.6} blk_min={blk:.6} \
         {verdict}"
    );
    for p in [c_ref, c_bad, bad.row_scale] {
        g.free(p)?;
    }
    Ok(caught)
}

/// 2026-10-06: (new ms, incumbent ms) at `TIME_M`.
fn timing(g: &dyn GpuBackend, kn: &Kern, cs: &Case, label: &str) -> Result<(f64, f64)> {
    let (n, k, m) = (cs.n, cs.k, TIME_M);
    let s = g.default_stream();
    let c = g.alloc(m * n * 2)?;
    cs.quant(g, kn, m)?;
    let t_new = median_ms(g, s, &mut || cs.new_path(g, &cs.w_bs, c, m).map(|_| ()))?;
    let t_old = median_ms(g, s, &mut || cs.incumbent(g, kn, c, m))?;
    let tf = |ms: f64| 2.0 * (m * n * k) as f64 / (ms * 1e-3) / 1e12;
    println!(
        "TIMING m={m} n={n} k={k} {label} new_ms={t_new:.4} ({:.1} TFLOP/s, {} launch(es)) \
         incumbent_ms={t_old:.4} ({:.1} TFLOP/s) speedup={:.3}x",
        tf(t_new),
        gw::plan(m, n, k).len(),
        tf(t_old),
        t_old / t_new
    );
    g.free(c)?;
    Ok((t_new, t_old))
}

fn kern(g: &dyn GpuBackend) -> Result<Kern> {
    let quant = ops::Fp8ActQuant::resolve(g);
    ensure!(
        quant.available(),
        "per_token_group_quant_fp8 is not in this image"
    );
    Ok(Kern {
        quant,
        rs: ops::fp8_gemm_rowscale_kernel(g)?
            .ok_or_else(|| anyhow::anyhow!("this image lacks fp8_gemm_rowscale_pipe_128x64"))?,
        wq: g.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?,
    })
}

/// 2026-10-06: `--phase route` (a child with the levers on): one weight through
/// `dense_fp8::convert_weight_nvfp4` + `finish_load` is block-scaled; `route` at M 257 returns
/// `Done` with bits equal to a direct quant + `dense_fp8_gw::gemm`; M 1 takes the NVFP4 GEMV.
fn phase_route() -> Result<bool> {
    use metrale_model_arch::glm5next_layer::dense_fp8::{self as df, LayerFp8, Route};
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kn = kern(g)?;
    let (n, k, m, s) = (4096usize, 4096usize, 257usize, g.default_stream());
    let mut rng = Lcg(0x5eed_1006);
    let mut p = upload(g, &gen_weight(&mut rng, n, k))?;
    let mut acc = LayerFp8::default();
    ensure!(
        df::convert_weight_nvfp4(g, &mut p, n, k, &mut acc)?,
        "convert declined"
    );
    df::finish_load(g)?;
    let w = df::lookup(p, n, k).ok_or_else(|| anyhow::anyhow!("weight not registered"))?;
    let a = upload(g, &gen_act(&mut rng, m * k))?;
    let (c1, c2) = (g.alloc(m * n * 2)?, g.alloc(m * n * 2)?);
    let (a_q, a_s) = (g.alloc(m * k)?, g.alloc(m * (k / KG) * 4)?);
    let gemv = g.kernel("gemv", "dense_gemv_bf16")?;
    let r = df::route(g, gemv, a, p, c1, m, n, k, s)?;
    ops::per_token_group_quant_fp8(g, kn.quant, a, a_q, a_s, m as u32, k as u32, s)?;
    let path = gw::gemm(g, a_q, a_s, &w, c2, m, n, k, s)?;
    let nv0 = df::nv4_hits();
    let r1 = df::route(g, gemv, a, p, c1.offset(0), 1, n, k, s)?;
    g.synchronize(s)?;
    let same =
        dn_u16(g, c1.offset(n * 2), (m - 1) * n)? == dn_u16(g, c2.offset(n * 2), (m - 1) * n)?;
    let ok = df::is_block_scaled(p, n, k)
        && r == Route::Done
        && path == GwPath::Cutlass
        && same
        && r1 == Route::Done
        && df::nv4_hits() == nv0 + 1;
    println!(
        "ROUTE block_scaled={} m{m}={r:?} bitwise_vs_direct={same} m1={r1:?} nv4_hit={}  {}",
        df::is_block_scaled(p, n, k),
        df::nv4_hits() - nv0,
        if ok { "ok" } else { "FAIL-CHECK" }
    );
    Ok(ok)
}

fn main() -> Result<()> {
    if !metrale_gpu_runtime::cutlass::available() {
        println!("FAIL: this binary was built without CUTLASS (set CUTLASS_HOME at build time)");
        bail!("no CUTLASS");
    }
    if std::env::args().any(|a| a == "--phase-route") {
        return if phase_route()? {
            Ok(())
        } else {
            bail!("route phase failed")
        };
    }
    let route_ok = std::process::Command::new(std::env::current_exe()?)
        .arg("--phase-route")
        .envs([
            ("METRALE_GLM_DENSE_FP8", "1"),
            ("METRALE_GLM_DENSE_FP8_W8A8", "1"),
            ("METRALE_GLM_DENSE_NVFP4", "kda"),
            ("METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW", "1"),
        ])
        .status()?
        .success();
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kn = kern(g)?;
    let k_major = metrale_gpu_runtime::cutlass::fp8_blockwise_scale_k_major()?;
    println!(
        "CUTLASS fp8 blockwise scale layouts: {}",
        if k_major { "K-major" } else { "MN-major" }
    );
    let mut rng = Lcg(0x6677_3838_c0de_1006);
    let (mut fails, mut cases, mut worst_cos, mut worst_blk) = (0usize, 0usize, 1f64, 1f64);
    let (mut tot_new, mut tot_old) = (0f64, 0f64);
    fails += usize::from(!route_ok);
    for (n, k, label, calls) in SHAPES {
        let cs = Case::new(g, &kn, &mut rng, n, k)?;
        for m in MS {
            cases += 1;
            let (ok, cos, blk) = check(g, &kn, &cs, m, label)?;
            fails += usize::from(!ok);
            // A NaN fails through `ok`; the minimum keeps the finite ones.
            worst_cos = worst_cos.min(cos);
            worst_blk = worst_blk.min(blk);
        }
        fails += usize::from(!known_bad(g, &kn, &cs, label)?);
        let (t_new, t_old) = timing(g, &kn, &cs, label)?;
        tot_new += t_new * calls as f64;
        tot_old += t_old * calls as f64;
        cs.free(g)?;
    }
    println!(
        "TIMING-WEIGHTED per forward (one TP2 rank, M {TIME_M}): new {tot_new:.2} ms vs incumbent \
         {tot_old:.2} ms ({:+.2} ms, {:.3}x)",
        tot_new - tot_old,
        tot_old / tot_new
    );
    if fails > 0 {
        println!(
            "FAIL: dense_fp8_cutlass_gw: {fails} checks failed (see FAIL-CHECK lines; route phase ok {route_ok})"
        );
        bail!("{fails} checks failed");
    }
    println!(
        "PASS: dense_fp8_cutlass_gw CUTLASS blockwise W8A8 via dense_fp8_gw::gemm: {cases} cases \
         cos_new >= {TOL_COS} (worst {worst_cos:.6}, worst 128-col block {worst_blk:.6}) and >= \
         incumbent - {TOL_VS_INC}, no nonfinite, guards intact, KNOWN_BAD caught on every shape, \
         route phase ok; weighted GEMM time {tot_new:.2} ms vs incumbent {tot_old:.2} ms"
    );
    Ok(())
}
