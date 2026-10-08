// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Gate for `METRALE_GLM_PREFILL_DENSE_W4A4`: the CUTLASS Sm120 dense NVFP4 W4A4
//! GEMM (`cuda/cutlass_nvfp4_gemm.cu`, CUTLASS example 79a configuration, NVIDIA,
//! BSD-3-Clause) through the engine's dispatch `dense_w4a4::gemm` at the GLM-5.3 TP2 dense
//! prefill shapes.
//!
//! Owner: model-arch examples.
//! Per (class, N, K) and M in {2048, 8192}, from one BF16 weight (N(0, 0.02), per-row gain,
//! 0.1 % x30 outliers) and one BF16 activation (N(0, 1), 0.5 % of channels x20..x50):
//! reference cuBLASLt BF16 (FP32 accumulate); W4A4 = `dense_w4a4::gemm` over the NVFP4 copy the
//! loader makes (`dense_fp8::quantize_nvfp4_copy`); W4A16 = the same copy dequantized on the
//! host to BF16 x the BF16 activation (cuBLASLt); FP8 = the car's incumbent
//! (`per_token_group_quant_fp8` + `dense_fp8_gw::gemm` over `quantize_block_scaled`).
//! Lines: `W4A4|W4A16|FP8 <class>/<proj> M= N= K=: cos= max_rel= nonfinite=`; `ROWS <class>/<proj>
//! M= split=[..]: <n> of <M> rows differ` (pieces through `dense_w4a4::gemm` at row offsets vs
//! the whole call, bitwise); `TIMING <class>/<proj> M= : fp8_ms= w4a4_ms= ratio= w4a4_TF/s=`
//! (ratio = fp8_ms / w4a4_ms, > 1 = W4A4 faster; w4a4_ms = act quant + SFB swizzle + GEMM, the
//! engine call; fp8_ms = quant + GEMM); `SFB_SWIZZLE <class>/<proj> M= : swizzle_ms= share=`
//! (the per-call weight-scale swizzle alone, and w4a4 with a prepacked SFB); then
//! `W4A4_SPEED M= weighted_ratio=` and `W4A4_CLASS <class> ratio=` (weights = calls per forward
//! on one TP2 rank: 34 KDA, 11 DSA, 42 shared-expert, 3 dense-MLP layers), `SFB_RESIDENT_BYTES`
//! (what a load-time pre-swizzled SFB would cost per rank), `ROUTE` / `ENGAGED_SITES` (child
//! `--phase-route`, levers on: class gating, row floor, fused-quant bypass, bits = direct call).
//! Every line above runs the static activation global scale (`GsMode::Static`, the default).
//! 2026-10-08, print only (never FAIL): `ROWS_DYNAMIC <class>/<proj> M= split=[..]: <n> of <M>
//! rows differ` (the same splits under `GsMode::Dynamic`, which is not row-invariant by
//! design) and, per shape at M = 8192, `W4A4_AMP <class>/<proj> amp=<a> gs=<static|dynamic>
//! cos=<> zero_blocks=<frac>`: the whole activation (N(0, 1) with its outlier channels)
//! multiplied by amp in `AMPS`, W4A4 cos against cuBLASLt BF16 on the scaled activation, and
//! the fraction of 16-value blocks whose UE4M3 scale is zero (`dense_w4a4::block_stats`).
//! PASS iff every cos >= `TOL_COS`, no nonfinite, 0 rows differ in every split, 0 fallbacks and
//! the route phase passes (speed is graded by the gate script). Last line `PASS ...` or
//! `FAIL ...` (nonzero exit).
//!
//! Run (GB10, image built with CUTLASS_HOME):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm_dense_w4a4_prefill_microtest

use anyhow::{Result, anyhow, bail, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::cutlass;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_layer::dense_w4a4::GsMode::{Dynamic, Static};
use metrale_model_arch::glm5next_layer::{dense_fp8 as df, dense_fp8_gw as gw, dense_w4a4 as w4};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{Fp8DenseWeight, QuantizedWeight};
use std::time::Instant;

/// 2026-10-08: (class, projection, N, K, calls per forward on one TP2 rank).
const SHAPES: [(&str, &str, usize, usize, usize); 9] = [
    ("kda", "qkvo", 4096, 4096, 4 * 34),
    ("dsa", "q_a", 1536, 4096, 11),
    ("dsa", "kv_a", 512, 4096, 11),
    ("dsa", "q_absorb", 16384, 1536, 11),
    ("dsa", "o_absorb", 4096, 16384, 11),
    ("shared", "gate_up", 1024, 4096, 2 * 42),
    ("shared", "down", 4096, 1024, 42),
    ("mlp", "gate_up", 6144, 4096, 2 * 3),
    ("mlp", "down", 4096, 6144, 3),
];
const CLASSES: [&str; 4] = ["kda", "dsa", "shared", "mlp"];
const MS: [usize; 2] = [2048, 8192];
const MAX_M: usize = 8192;
const SPLITS_8192: [&[usize]; 4] = [
    &[4096, 4096],
    &[2339, 5853],
    &[256, 7936],
    &[2204, 2183, 3805],
];
const SPLITS_2048: [&[usize]; 3] = [&[1024, 1024], &[256, 1792], &[777, 1271]];
const TOL_COS: f64 = 0.99;
/// 2026-10-08: Activation amplitudes of the `W4A4_AMP` sweep (M = 8192, both gs modes).
const AMPS: [f64; 4] = [0.01, 0.1, 1.0, 100.0];
const WARMUP: usize = 5;
const REPS: usize = 20;
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

fn to_bytes(v: impl Iterator<Item = f64>) -> Vec<u8> {
    v.flat_map(|x| bf16::from_f64(x).to_bits().to_le_bytes())
        .collect()
}

/// 2026-10-08: Weights N(0, 0.02) with a per-row gain (x0.25..x4) and 0.1 % outliers at x30
/// (the `dense_fp8_cutlass_gw_microtest` generator).
fn gen_weight(rng: &mut Lcg, n: usize, k: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * k);
    for _ in 0..n {
        let gain = 0.25 * 16f64.powf(rng.u());
        for _ in 0..k {
            let x = rng.n() * 0.02 * gain;
            v.push(if rng.u() < 0.001 { x * 30.0 } else { x });
        }
    }
    to_bytes(v.into_iter())
}

/// 2026-10-08: Activations N(0, 1); 0.5 % of the K channels (fixed per channel) scaled x20..x50.
fn gen_act(rng: &mut Lcg, m: usize, k: usize) -> Vec<u8> {
    let gain: Vec<f64> = (0..k)
        .map(|_| {
            if rng.u() < 0.005 {
                20.0 + 30.0 * rng.u()
            } else {
                1.0
            }
        })
        .collect();
    let mut v = Vec::with_capacity(m * k);
    for _ in 0..m {
        for g in &gain {
            v.push(rng.n() * g);
        }
    }
    to_bytes(v.into_iter())
}

fn upload(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(16))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn dn_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let b = dn(g, p, n * 2)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}
fn bf(x: u16) -> f64 {
    bf16::from_bits(x).to_f64()
}

/// 2026-10-08: OCP E4M3 (bias 7, subnormals, 0x7F/0xFF NaN) to f32.
fn e4m3(b: u8) -> f32 {
    let (s, e, m) = ((b >> 7) & 1, (b >> 3) & 0xF, b & 7);
    let mag = if e == 0xF && m == 7 {
        f32::NAN
    } else if e == 0 {
        m as f32 * 2f32.powi(-9)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    };
    if s == 1 { -mag } else { mag }
}

/// 2026-10-08: Host BF16 dequant of the NVFP4 copy (`[n, k/2]` low nibble first, E4M3
/// `[n, k/16]`, tensor scale2): the W4A16-equivalent weight.
fn dequant_nv4(g: &dyn GpuBackend, q: &QuantizedWeight, n: usize, k: usize) -> Result<Vec<u8>> {
    const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let packed = dn(g, q.weight, n * k / 2)?;
    let scales = dn(g, q.weight_scale, n * k / 16)?;
    let mut out = Vec::with_capacity(n * k * 2);
    for r in 0..n {
        for c in 0..k {
            let byte = packed[r * k / 2 + c / 2];
            let code = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let v = E2M1[(code & 7) as usize] * if code & 8 != 0 { -1.0 } else { 1.0 };
            let x = v * e4m3(scales[r * k / 16 + c / 16]) * q.weight_scale_2;
            out.extend_from_slice(&bf16::from_f32(x).to_bits().to_le_bytes());
        }
    }
    Ok(out)
}

/// 2026-10-08: (cos, max |y - r| / max |r|, nonfinite in y) of `y` against `r`.
fn metrics(y: &[u16], r: &[u16]) -> (f64, f64, usize) {
    let (mut d, mut na, mut nb, mut em, mut rm, mut bad) = (0f64, 0f64, 0f64, 0f64, 0f64, 0);
    for (&a, &b) in y.iter().zip(r) {
        let (a, b) = (bf(a), bf(b));
        if !a.is_finite() {
            bad += 1;
            continue;
        }
        d += a * b;
        na += a * a;
        nb += b * b;
        em = em.max((a - b).abs());
        rm = rm.max(b.abs());
    }
    let cos = if na > 0.0 && nb > 0.0 {
        d / (na.sqrt() * nb.sqrt())
    } else {
        f64::NAN
    };
    (cos, em / rm, bad)
}

/// 2026-10-08: Warm median ms of `f` over `REPS` calls, each followed by a synchronize.
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

/// 2026-10-08: Per-shape device state.
struct Case {
    n: usize,
    k: usize,
    w_bf16: DevicePtr,
    w_deq: DevicePtr,
    nv: QuantizedWeight,
    bs: Fp8DenseWeight,
    a: DevicePtr,
    a_q: DevicePtr,
    a_s: DevicePtr,
    quant: ops::Fp8ActQuant,
}

impl Case {
    fn new(g: &dyn GpuBackend, rng: &mut Lcg, n: usize, k: usize) -> Result<Self> {
        let s = g.default_stream();
        let w_bf16 = upload(g, &gen_weight(rng, n, k))?;
        let nv = df::quantize_nvfp4_copy(g, w_bf16, n, k, s)?
            .ok_or_else(|| anyhow!("the NVFP4 quantizer kernels are not in this image"))?;
        let bs = gw::quantize_block_scaled(g, w_bf16, n, k, s)?;
        let quant = ops::Fp8ActQuant::resolve(g);
        ensure!(
            quant.available(),
            "per_token_group_quant_fp8 is not in this image"
        );
        Ok(Self {
            n,
            k,
            w_bf16,
            w_deq: upload(g, &dequant_nv4(g, &nv, n, k)?)?,
            nv,
            bs,
            a: upload(g, &gen_act(rng, MAX_M, k))?,
            a_q: g.alloc(MAX_M * k)?,
            a_s: g.alloc(MAX_M * (k / KG) * 4)?,
            quant,
        })
    }
    /// 2026-10-08: Rows `r0..r0 + m` through `dense_w4a4::gemm_ex` (the engine's `gemm` with
    /// an explicit gs mode; per-call SFB swizzle as in the engine).
    fn w4a4(
        &self,
        g: &dyn GpuBackend,
        c: DevicePtr,
        r0: usize,
        m: usize,
        mode: w4::GsMode,
    ) -> Result<bool> {
        let a = self.a.offset(r0 * self.k * 2);
        w4::gemm_ex(
            a,
            &self.nv,
            DevicePtr(0),
            c.offset(r0 * self.n * 2),
            m,
            self.n,
            self.k,
            mode,
            g.default_stream(),
        )
    }
    fn fp8(&self, g: &dyn GpuBackend, c: DevicePtr, m: usize) -> Result<()> {
        let (s, k) = (g.default_stream(), self.k as u32);
        ops::per_token_group_quant_fp8(g, self.quant, self.a, self.a_q, self.a_s, m as u32, k, s)?;
        gw::gemm(g, self.a_q, self.a_s, &self.bs, c, m, self.n, self.k, s).map(|_| ())
    }
    fn free(self, g: &dyn GpuBackend) -> Result<()> {
        let (nv, bs) = (self.nv, self.bs);
        [
            self.w_bf16,
            self.w_deq,
            nv.weight,
            nv.weight_scale,
            bs.weight,
            bs.row_scale,
            self.a,
        ]
        .into_iter()
        .chain([self.a_q, self.a_s])
        .try_for_each(|p| g.free(p))
    }
}

/// 2026-10-08: Totals over the run.
#[derive(Default)]
struct Tally {
    fails: usize,
    worst_cos: f64,
    rows_differ: usize,
    /// (class, M, fp8 ms x calls, w4a4 ms x calls)
    time: Vec<(&'static str, usize, f64, f64)>,
    sfb_bytes: usize,
}

/// 2026-10-08: Numerics, row invariance and timing of one shape at one M.
fn run_m(
    g: &dyn GpuBackend,
    cs: &Case,
    m: usize,
    (class, tag): (&'static str, &str),
    calls: usize,
    t: &mut Tally,
) -> Result<f64> {
    let (n, k, s) = (cs.n, cs.k, g.default_stream());
    let bytes = m * n * 2;
    let [c_ref, c_w4, c_w16, c_f8, c_sp] = [0; 5].map(|_| g.alloc(bytes));
    let (c_ref, c_w4, c_w16, c_f8, c_sp) = (c_ref?, c_w4?, c_w16?, c_f8?, c_sp?);
    ops::cublas_bf16_proj_dense(cs.a, cs.w_bf16, c_ref, m as u32, n as u32, k as u32, s)?;
    ops::cublas_bf16_proj_dense(cs.a, cs.w_deq, c_w16, m as u32, n as u32, k as u32, s)?;
    let engaged = cs.w4a4(g, c_w4, 0, m, Static)?;
    cs.fp8(g, c_f8, m)?;
    g.synchronize(s)?;
    let r = dn_u16(g, c_ref, m * n)?;
    let y4 = dn_u16(g, c_w4, m * n)?;
    for (arm, c) in [("W4A4", c_w4), ("W4A16", c_w16), ("FP8", c_f8)] {
        let y = if arm == "W4A4" {
            y4.clone()
        } else {
            dn_u16(g, c, m * n)?
        };
        let (cos, rel, bad) = metrics(&y, &r);
        let ok = cos >= TOL_COS && bad == 0;
        println!(
            "{arm} {tag} M={m} N={n} K={k}: cos={cos:.6} max_rel={rel:.3e} nonfinite={bad}{}",
            if arm == "W4A4" && !ok {
                "  FAIL-CHECK"
            } else {
                ""
            }
        );
        if arm == "W4A4" {
            t.fails += usize::from(!ok || !engaged);
            t.worst_cos = t.worst_cos.min(if cos.is_nan() { -1.0 } else { cos });
        }
    }
    let splits: &[&[usize]] = if m == 8192 {
        &SPLITS_8192
    } else {
        &SPLITS_2048
    };
    for split in splits {
        ensure!(
            split.iter().sum::<usize>() == m,
            "split {split:?} does not sum to {m}"
        );
        g.memset(c_sp, 0xFF, bytes)?;
        let mut r0 = 0;
        for &p in *split {
            ensure!(
                cs.w4a4(g, c_sp, r0, p, Static)?,
                "W4A4 declined a {p}-row piece"
            );
            r0 += p;
        }
        g.synchronize(s)?;
        let ys = dn_u16(g, c_sp, m * n)?;
        let differ = (0..m)
            .filter(|i| ys[i * n..(i + 1) * n] != y4[i * n..(i + 1) * n])
            .count();
        t.rows_differ += differ;
        t.fails += usize::from(differ > 0);
        println!("ROWS {tag} M={m} split={split:?}: {differ} of {m} rows differ");
    }
    // 2026-10-08: The same splits under the dynamic gs: print only (each piece has its own
    // amax, so rows may differ by design).
    ensure!(
        cs.w4a4(g, c_w4, 0, m, Dynamic)?,
        "W4A4 (dynamic gs) declined"
    );
    g.synchronize(s)?;
    let y4d = dn_u16(g, c_w4, m * n)?;
    for split in splits {
        g.memset(c_sp, 0xFF, bytes)?;
        let mut r0 = 0;
        for &p in *split {
            ensure!(
                cs.w4a4(g, c_sp, r0, p, Dynamic)?,
                "W4A4 declined a {p}-row piece"
            );
            r0 += p;
        }
        g.synchronize(s)?;
        let ys = dn_u16(g, c_sp, m * n)?;
        let differ = (0..m)
            .filter(|i| ys[i * n..(i + 1) * n] != y4d[i * n..(i + 1) * n])
            .count();
        println!("ROWS_DYNAMIC {tag} M={m} split={split:?}: {differ} of {m} rows differ");
    }
    let fp8_ms = median_ms(g, s, &mut || cs.fp8(g, c_f8, m))?;
    let w4_ms = median_ms(g, s, &mut || cs.w4a4(g, c_w4, 0, m, Static).map(|_| ()))?;
    let sfb = g.alloc(cutlass::dense_w4a4_sfb_bytes(n, k).max(16))?;
    let (sw, pre) = (cs.nv.weight_scale.0, sfb.0);
    let swz_ms = median_ms(g, s, &mut || {
        cutlass::nvfp4_dense_w4a4_pack_sfb(sw, pre, n as u32, k as u32, s)
    })?;
    let pre_ms = median_ms(g, s, &mut || {
        w4::gemm_ex(cs.a, &cs.nv, sfb, c_w4, m, n, k, Static, s).map(|_| ())
    })?;
    let tf = 2.0 * (m * n * k) as f64 / (w4_ms * 1e-3) / 1e12;
    println!(
        "TIMING {tag} M={m} : fp8_ms={fp8_ms:.4} w4a4_ms={w4_ms:.4} ratio={:.3} \
         w4a4_TF/s={tf:.1}",
        fp8_ms / w4_ms
    );
    println!(
        "SFB_SWIZZLE {tag} M={m} : swizzle_ms={swz_ms:.4} share={:.2}% \
         w4a4_prepacked_sfb_ms={pre_ms:.4}",
        100.0 * swz_ms / w4_ms
    );
    t.time
        .push((class, m, fp8_ms * calls as f64, w4_ms * calls as f64));
    for p in [c_ref, c_w4, c_w16, c_f8, c_sp, sfb] {
        g.free(p)?;
    }
    Ok(swz_ms / w4_ms)
}

/// 2026-10-08: The `W4A4_AMP` sweep of one shape at M = `MAX_M` (module doc): print only.
fn amp_sweep(g: &dyn GpuBackend, cs: &Case, tag: &str) -> Result<()> {
    let (n, k, m, s) = (cs.n, cs.k, MAX_M, g.default_stream());
    let host = dn(g, cs.a, m * k * 2)?;
    let [a, c_ref, c_w4] = [m * k * 2, m * n * 2, m * n * 2].map(|b| g.alloc(b));
    let (a, c_ref, c_w4) = (a?, c_ref?, c_w4?);
    for amp in AMPS {
        let scaled: Vec<u8> = host
            .chunks_exact(2)
            .flat_map(|b| {
                let x = bf(u16::from_le_bytes([b[0], b[1]])) * amp;
                bf16::from_f64(x).to_bits().to_le_bytes()
            })
            .collect();
        g.copy_h2d(&scaled, a)?;
        ops::cublas_bf16_proj_dense(a, cs.w_bf16, c_ref, m as u32, n as u32, k as u32, s)?;
        g.synchronize(s)?;
        let r = dn_u16(g, c_ref, m * n)?;
        for mode in [Static, Dynamic] {
            let ok = w4::gemm_ex(a, &cs.nv, DevicePtr(0), c_w4, m, n, k, mode, s)?;
            ensure!(ok, "W4A4 declined amp {amp} gs={}", mode.name());
            g.synchronize(s)?;
            let (cos, _, _) = metrics(&dn_u16(g, c_w4, m * n)?, &r);
            let st = w4::block_stats(&scaled, m, k, mode);
            println!(
                "W4A4_AMP {tag} amp={amp} gs={} cos={cos:.6} zero_blocks={:.6}",
                mode.name(),
                st.frac(st.zero)
            );
        }
    }
    for p in [a, c_ref, c_w4] {
        g.free(p)?;
    }
    Ok(())
}

/// 2026-10-08: `--phase-route` (levers on; see `main`): class gating, the row floor, the fused
/// quant bypass and bits equal to a direct `dense_w4a4::gemm`, through `dense_fp8::route`.
fn phase_route() -> Result<bool> {
    use df::{Class, LayerFp8, Route};
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let s = g.default_stream();
    let mut rng = Lcg(0x5eed_1008);
    let (n, k, nb, m) = (4096usize, 4096usize, 1024usize, 256usize);
    let mut acc = LayerFp8::default();
    let mut pa = upload(g, &gen_weight(&mut rng, n, k))?;
    let mut pb = upload(g, &gen_weight(&mut rng, nb, k))?;
    ensure!(
        df::convert_weight_nvfp4_class(g, &mut pa, n, k, &mut acc, Class::Kda, "q_proj")?,
        "kda"
    );
    ensure!(
        df::convert_weight_nvfp4_class(g, &mut pb, nb, k, &mut acc, Class::Shared, "gate_proj")?,
        "sh"
    );
    df::finish_load(g)?;
    let q = df::lookup_nv4(pa, n, k).ok_or_else(|| anyhow!("no NVFP4 copy registered"))?;
    let a = upload(g, &gen_act(&mut rng, m, k))?;
    let [c1, c2, c3] = [0; 3].map(|_| g.alloc(m * n * 2));
    let (c1, c2, c3) = (c1?, c2?, c3?);
    let gemv = g.kernel("gemv", "dense_gemv_bf16")?;
    let g0 = w4::gemms();
    let r_a = df::route(g, gemv, a, pa, c1, m, n, k, s)?;
    let hit_a = w4::gemms() - g0;
    ensure!(w4::gemm(a, &q, c2, m, n, k, s)?, "direct W4A4 declined");
    let r_floor = df::route(g, gemv, a, pa, c3, m - 1, n, k, s)?;
    let r_b = df::route(g, gemv, a, pb, c3, m, nb, k, s)?;
    let hit_rest = w4::gemms() - g0 - hit_a - 1;
    let mut filled = false;
    let fused_a = df::w8a8_fused_input(g, gemv, a, pa, c3, (m, n, k), s, &mut |_, _| {
        filled = true;
        Ok(())
    })?;
    let filled_a = filled;
    g.synchronize(s)?;
    let same = dn_u16(g, c1, m * n)? == dn_u16(g, c2, m * n)?;
    let fuse_on = df::norm_fp8_quant_fuse();
    let ok = r_a == Route::Done
        && hit_a == 1
        && same
        && r_floor == Route::Done
        && r_b == Route::Done
        && hit_rest == 0
        && !fused_a
        && !filled_a
        && fuse_on;
    println!(
        "ROUTE kda m{m}={r_a:?} w4a4_hit={hit_a} bitwise_vs_direct={same}; kda m{}={r_floor:?} \
         and shared(unlisted) m{m}={r_b:?} w4a4_hits={hit_rest}; fused_quant(kda m{m})={fused_a} \
         fill_called={filled_a} (fuse lever {fuse_on})  {}",
        m - 1,
        if ok { "ok" } else { "FAIL-CHECK" }
    );
    for p in [a, c1, c2, c3] {
        g.free(p)?;
    }
    Ok(ok)
}

fn main() -> Result<()> {
    if !cutlass::available() {
        println!("FAIL glm_dense_w4a4_prefill_microtest: built without CUTLASS (set CUTLASS_HOME)");
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
            ("METRALE_GLM_DENSE_NVFP4", "kda,shared"),
            ("METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW", "1"),
            ("METRALE_GLM_DENSE_FP8_W8A8_ROWS", "1024"),
            ("METRALE_GLM_NORM_FP8_QUANT_FUSE", "1"),
            (w4::LEVER, "kda"),
        ])
        .status()?
        .success();
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut rng = Lcg(0x6677_4a4a_c0de_1008);
    let mut t = Tally {
        worst_cos: 1.0,
        ..Tally::default()
    };
    let mut worst_share = 0f64;
    for (class, proj, n, k, calls) in SHAPES {
        let cs = Case::new(g, &mut rng, n, k)?;
        let tag = format!("{class}/{proj}");
        for m in MS {
            worst_share = worst_share.max(run_m(g, &cs, m, (class, &tag), calls, &mut t)?);
            if m == MAX_M {
                amp_sweep(g, &cs, &tag)?;
            }
        }
        t.sfb_bytes += cutlass::dense_w4a4_sfb_bytes(n, k) * calls;
        cs.free(g)?;
    }
    for m in MS {
        let rows = t.time.iter().filter(|x| x.1 == m);
        let (f, w) = rows.fold((0f64, 0f64), |a, x| (a.0 + x.2, a.1 + x.3));
        println!(
            "W4A4_SPEED M={m} weighted_ratio={:.3} (fp8 {f:.2} ms, w4a4 {w:.2} ms)",
            f / w
        );
    }
    for class in CLASSES {
        let rows = t.time.iter().filter(|x| x.1 == MAX_M && x.0 == class);
        let (f, w) = rows.fold((0f64, 0f64), |a, x| (a.0 + x.2, a.1 + x.3));
        println!("W4A4_CLASS {class} ratio={:.3} (M={MAX_M})", f / w);
    }
    println!(
        "SFB_RESIDENT_BYTES per_rank={:.1} MB (a load-time pre-swizzled SFB for every listed \
         weight; not allocated: per-call swizzle, worst share {:.2}%)",
        t.sfb_bytes as f64 / 1e6,
        100.0 * worst_share
    );
    println!(
        "ENGAGED_SITES W4A4 (classes listed + NVFP4 copy, >= {} rows, BF16 input): KDA \
         q/k/v/o (o_proj skips the fused norm+FP8 quant), DSA q_a/kv_a/q_absorb/o_absorb, \
         shared gate/up/down, dense MLP gate/up/down, MTP block (class mtp). FP8 kept: KDA \
         g_a/g_b (no NVFP4 copy), calls below {} rows, refused calls (FALLBACK)",
        w4::MIN_ROWS,
        w4::MIN_ROWS
    );
    let fallbacks = w4::fallbacks();
    t.fails += usize::from(fallbacks > 0) + usize::from(!route_ok);
    if t.fails > 0 {
        println!(
            "FAIL glm_dense_w4a4_prefill_microtest: {} checks failed (worst cos {:.6}, {} rows \
             differ, {fallbacks} fallbacks, route phase ok {route_ok})",
            t.fails, t.worst_cos, t.rows_differ
        );
        bail!("{} checks failed", t.fails);
    }
    println!(
        "PASS glm_dense_w4a4_prefill_microtest: {} shapes x M {MS:?}: W4A4 cos >= {TOL_COS} \
         (worst {:.6}), no nonfinite, 0 rows differ in every split, 0 fallbacks, route phase ok",
        SHAPES.len(),
        t.worst_cos
    );
    Ok(())
}
