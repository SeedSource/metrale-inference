// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: SeedHQ/seed-skills#71 S1 gate: the DFlash2 drafter's propose GEMMs on GB10 at
//! the GLM-5.3 drafter's real shapes (8 layers, hidden 4096, intermediate 12288, q 4096,
//! kv 1024, conv 1024, fc K 9 x 4096, fused K/V 16384, selector rank 256, vocab 154880).
//!
//! Owner: model-arch examples (DFlash drafter).
//! Checks, per shape and M in {8, 16}:
//! - old = `dense_gemm_bf16_pipelined` (today's launch), S1a = `dense_gemv_bf16_batchm`
//!   (METRALE_DFLASH_DRAFT_GEMV=1), S1b = `fp8_gemv_rowscale_batch{8,16}_rt2` over the
//!   `quantize_bf16_to_fp8` copy (METRALE_DFLASH_DRAFTER_FP8=1 at <= 16 rows), and the ported
//!   `dflash_fp8_gemm::fp8_gemm_t_row_scaled{,_m16}` (the FP8 path above 16 rows).
//! - Accuracy against an f64 host reference on 64 sampled columns (all rows), and S1a against
//!   old on the whole output: max abs error, max |ref|, relative RMS error. Bounds: BF16
//!   kernels rel-RMS <= 1e-2; S1b (FP8 weights) <= 6e-2; ported W8A8 tiles <= 1e-1.
//! - Timing: CUDA events, median of 15 single launches after 3 warm-ups. Weights rotate over
//!   enough copies to pass ~96 MB per round, so a launch reads cold weights as the propose
//!   does (L2 is 24 MB); GB/s counts weight bytes only.
//! - `STEP` lines sum the per-step launch counts at M = 8 (one sequence, gamma 8):
//!   old, S1a everywhere, and S1a + S1b (FP8 on the seven layer GEMMs and the lm_head).
//! - A line starting `PASS` when every accuracy bound holds; exit 1 on a bound miss, 2 if a
//!   kernel is missing.
//!
//! Run (GB10, under a lane lock):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dflash_propose_gemm_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight};

// CUDA driver event API for kernel-only timing, declared as in `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-08: (N, K, launches per propose step at one sequence, FP8 under S1b, label). Per
/// layer: q, k, v, o, gate, up, down, and the DFlash2 attention + MLP conv projections; then
/// the ctx precompute's fc and fused K/V, the selector's H projection and the lm_head.
const SHAPES: &[(usize, usize, usize, bool, &str)] = &[
    (4096, 4096, 16, true, "q/o"),
    (1024, 4096, 16, true, "k/v"),
    (1024, 4096, 16, false, "dflash2 conv"),
    (12288, 4096, 16, true, "gate/up"),
    (4096, 12288, 8, true, "down"),
    (4096, 36864, 1, false, "fc (precompute)"),
    (16384, 4096, 1, false, "fused K/V (precompute)"),
    (256, 4096, 1, false, "selector H"),
    (154880, 4096, 1, true, "lm_head"),
];
const MS: &[usize] = &[8, 16];
const REPS: usize = 15;
const WARMUP: usize = 3;
const ROTATE_BYTES: usize = 96 << 20;
const SAMPLE_COLS: usize = 64;
const TOL_BF16: f64 = 1e-2;
const TOL_FP8W: f64 = 6e-2;
const TOL_W8A8: f64 = 1e-1;
const ARMS: [&str; 5] = ["old", "S1a", "S1b", "tile16", "tile64"];

struct Rng(u64);
#[rustfmt::skip]
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn u(&mut self) -> f32 { (self.next() & 0xFF_FFFF) as f32 / 16_777_216.0 }
    fn normal(&mut self) -> f32 {
        let (a, b) = (self.u().max(1e-7), self.u());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}

fn up(g: &dyn GpuBackend, v: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Time `f` (enqueued on stream `s`) between two CUDA events; returns milliseconds.
fn time_ms(g: &dyn GpuBackend, s: u64, f: &dyn Fn() -> Result<()>) -> Result<f64> {
    let (mut e0, mut e1) = (0u64, 0u64);
    // SAFETY: plain CUDA driver event calls on handles created here and destroyed below.
    unsafe {
        if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
            bail!("cuEventCreate failed");
        }
        cuEventRecord(e0, s);
    }
    f()?;
    let mut ms = 0f32;
    // SAFETY: as above.
    let rc = unsafe {
        cuEventRecord(e1, s);
        let rc = cuEventSynchronize(e1);
        let rc2 = cuEventElapsedTime(&mut ms, e0, e1);
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
        rc | rc2
    };
    g.synchronize(s)?;
    if rc != 0 {
        bail!("CUDA event timing failed: status {rc}");
    }
    Ok(ms as f64)
}

struct Kernels {
    old: KernelHandle,
    batchm: KernelHandle,
    rt2_8: KernelHandle,
    rt2_16: KernelHandle,
    tile16: KernelHandle,
    tile64: KernelHandle,
    quant: KernelHandle,
}

/// 2026-10-08: One launch of arm `arm` over weight copy `w` / `w8`.
#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    k: &Kernels,
    arm: usize,
    a: DevicePtr,
    w: &DenseWeight,
    w8: &Fp8DenseWeight,
    c: DevicePtr,
    (m, n, kk): (u32, u32, u32),
    s: u64,
) -> Result<()> {
    match arm {
        0 => ops::dense_gemm_bf16_pipelined(g, k.old, a, w, c, m, n, kk, s),
        1 => ops::dense_gemv_batchm(g, k.batchm, a, w, c, m, n, kk, n, s),
        2 if m <= 8 => ops::fp8_gemv_rowscale_batch8_rt2(g, k.rt2_8, a, w8, c, m, n, kk, s),
        2 => ops::fp8_gemv_rowscale_batch16_rt2(g, k.rt2_16, a, w8, c, m, n, kk, s),
        3 => ops::fp8_gemm_n128_row_scaled_m16(g, k.tile16, a, w8, c, m, n, kk, s),
        _ => ops::fp8_gemm_n128_row_scaled(g, k.tile64, a, w8, c, m, n, kk, s),
    }
}

/// 2026-10-08: (max abs error, max |ref|, relative RMS error) of `got` against `want`.
fn err(got: &[f64], want: &[f64]) -> (f64, f64, f64) {
    let (mut max_e, mut max_r, mut se, mut sr) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in got.iter().zip(want) {
        let e = (x - y).abs();
        max_e = max_e.max(e);
        max_r = max_r.max(y.abs());
        se += e * e;
        sr += y * y;
    }
    (max_e, max_r, (se / sr.max(1e-30)).sqrt())
}

/// 2026-10-08: Runs one shape at every M; returns (accuracy misses, [arm][m index] ms).
fn run_shape(
    g: &dyn GpuBackend,
    k: &Kernels,
    rng: &mut Rng,
    (n, kk, label): (usize, usize, &str),
) -> Result<(usize, Vec<Vec<f64>>)> {
    let s = g.create_stream()?;
    // 2026-10-08: Uniform weights of std 0.02 (half-width 0.0346); the lm_head alone is 634M.
    let wh: Vec<bf16> = (0..n * kk)
        .map(|_| bf16::from_f32((rng.u() * 2.0 - 1.0) * 0.0346))
        .collect();
    let w_bytes = n * kk * 2;
    let copies = ROTATE_BYTES.div_ceil(w_bytes).clamp(1, 12);
    let mut ws = vec![DenseWeight {
        weight: up(g, &wh)?,
    }];
    for _ in 1..copies {
        let p = g.alloc(w_bytes)?;
        g.copy_d2d(ws[0].weight, p, w_bytes)?;
        ws.push(DenseWeight { weight: p });
    }
    let mut w8s = Vec::with_capacity(copies);
    for w in &ws {
        w8s.push(w.quantize_to_fp8(g, k.quant, n, kk, s)?);
    }
    g.synchronize(s)?;
    let cols: Vec<usize> = (0..SAMPLE_COLS)
        .map(|j| (j * n / SAMPLE_COLS + j % 7).min(n - 1))
        .collect();
    let mut misses = 0usize;
    let mut ms_table = vec![vec![f64::NAN; MS.len()]; ARMS.len()];
    for (mi, &m) in MS.iter().enumerate() {
        let ah: Vec<bf16> = (0..m * kk).map(|_| bf16::from_f32(rng.normal())).collect();
        let a = up(g, &ah)?;
        let outs: Vec<DevicePtr> = (0..ARMS.len())
            .map(|_| g.alloc(m * n * 2))
            .collect::<Result<_>>()?;
        let dims = (m as u32, n as u32, kk as u32);
        for (arm, &c) in outs.iter().enumerate() {
            launch(g, k, arm, a, &ws[0], &w8s[0], c, dims, s)?;
        }
        g.synchronize(s)?;
        let host: Vec<Vec<f32>> = outs
            .iter()
            .map(|&c| dn(g, c, m * n))
            .collect::<Result<_>>()?;
        // 2026-10-08: f64 host reference on the sampled columns, every row.
        let mut want = Vec::with_capacity(m * cols.len());
        for t in 0..m {
            for &col in &cols {
                let (ar, wr) = (&ah[t * kk..(t + 1) * kk], &wh[col * kk..(col + 1) * kk]);
                want.push(
                    ar.iter()
                        .zip(wr)
                        .map(|(x, y)| x.to_f64() * y.to_f64())
                        .sum(),
                );
            }
        }
        let pick = |v: &[f32]| -> Vec<f64> {
            (0..m)
                .flat_map(|t| cols.iter().map(move |&col| (t, col)))
                .map(|(t, col)| v[t * n + col] as f64)
                .collect()
        };
        for (arm, name) in ARMS.iter().enumerate() {
            let tol = [TOL_BF16, TOL_BF16, TOL_FP8W, TOL_W8A8, TOL_W8A8][arm];
            let (e, r, rel) = err(&pick(&host[arm]), &want);
            let ok = rel <= tol;
            misses += usize::from(!ok);
            println!(
                "ACC {label:<24} M={m:<2} {name:<6} vs host: max_abs {e:.3e} max_ref {r:.3e} \
                 rel_rms {rel:.3e} (<= {tol:.0e}) {}",
                if ok { "ok" } else { "MISS" }
            );
        }
        let full = |v: &[f32]| v.iter().map(|x| *x as f64).collect::<Vec<f64>>();
        let (e, r, rel) = err(&full(&host[1]), &full(&host[0]));
        let ok = rel <= TOL_BF16;
        misses += usize::from(!ok);
        println!(
            "ACC {label:<24} M={m:<2} S1a    vs old (all {} outputs): max_abs {e:.3e} max_ref \
             {r:.3e} rel_rms {rel:.3e} (<= {TOL_BF16:.0e}) {}",
            m * n,
            if ok { "ok" } else { "MISS" }
        );
        for (arm, &c) in outs.iter().enumerate() {
            let mut v = Vec::with_capacity(REPS);
            for rep in 0..WARMUP + REPS {
                let i = rep % copies;
                let f = || launch(g, k, arm, a, &ws[i], &w8s[i], c, dims, s);
                let t = time_ms(g, s, &f)?;
                if rep >= WARMUP {
                    v.push(t);
                }
            }
            let t = median(&mut v);
            ms_table[arm][mi] = t;
            let bytes = if arm >= 2 { n * kk } else { w_bytes };
            println!(
                "TIME {label:<24} M={m:<2} {:<6} {t:>8.4} ms {:>7.1} GB/s (N={n} K={kk})",
                ARMS[arm],
                bytes as f64 / (t * 1e-3) / 1e9
            );
        }
        for c in outs {
            g.free(c)?;
        }
        g.free(a)?;
    }
    for (w, w8) in ws.iter().zip(&w8s) {
        g.free(w.weight)?;
        g.free(w8.weight)?;
        g.free(w8.row_scale)?;
    }
    Ok((misses, ms_table))
}

fn run() -> Result<i32> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let names = [
        ("gemm", "dense_gemm_bf16_pipelined"),
        ("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
        ("fp8_gemv_rt", "fp8_gemv_rowscale_batch8_rt2"),
        ("fp8_gemv_rt", "fp8_gemv_rowscale_batch16_rt2"),
        ("dflash_fp8_gemm", "fp8_gemm_t_row_scaled_m16"),
        ("dflash_fp8_gemm", "fp8_gemm_t_row_scaled"),
        ("gemv_fp8w", "quantize_bf16_to_fp8"),
    ];
    let mut hs = Vec::new();
    for (module, name) in names {
        match g.kernel(module, name) {
            Ok(h) => hs.push(h),
            Err(_) => println!("MISSING kernel entry {module}::{name}"),
        }
    }
    if hs.len() != names.len() {
        println!("FAIL kernel entries missing ({})", names.len() - hs.len());
        return Ok(2);
    }
    let k = Kernels {
        old: hs[0],
        batchm: hs[1],
        rt2_8: hs[2],
        rt2_16: hs[3],
        tile16: hs[4],
        tile64: hs[5],
        quant: hs[6],
    };
    let mut rng = Rng(0x0064_6670_726f_7001);
    let (mut misses, mut step) = (0usize, [0f64; 3]);
    for &(n, kk, count, fp8, label) in SHAPES {
        let (miss, t) = run_shape(g, &k, &mut rng, (n, kk, label))?;
        misses += miss;
        // 2026-10-08: Per-step sums at M = 8 (MS[0]).
        let c = count as f64;
        step[0] += c * t[0][0];
        step[1] += c * t[1][0];
        step[2] += c * if fp8 { t[2][0] } else { t[1][0] };
    }
    println!(
        "STEP propose GEMMs at M=8 (one sequence, gamma 8, precompute at 8 rows): old {:.2} ms, \
         S1a {:.2} ms, S1a+S1b {:.2} ms (nsys 2026-10-08: dense_gemm_bf16_pipelined 37.2 of \
         ~39 ms/step)",
        step[0], step[1], step[2]
    );
    if misses == 0 {
        println!(
            "PASS: dflash_propose_gemm_microtest accuracy bounds hold on {} shapes x {} M",
            SHAPES.len(),
            MS.len()
        );
        Ok(0)
    } else {
        println!("FAIL dflash_propose_gemm_microtest: {misses} accuracy bound(s) missed");
        Ok(1)
    }
}

fn main() {
    match run() {
        Ok(0) => {}
        Ok(c) => std::process::exit(c),
        Err(e) => {
            println!("FAIL error: {e:#}");
            std::process::exit(1);
        }
    }
}
