// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate for `METRALE_DFLASH_NV4_HEAD` / `METRALE_DFLASH_NV4_LAYERS`: the DFlash2
//! drafter's per-rank GEMMs (GLM-5.3, `METRALE_DFLASH_DRAFT_TP2`, gamma 8) on their NVFP4 copy
//! (`dense_fp8::quantize_nvfp4_copy` + `dense_fp8::nv4_gemv_cuda_core`, the levers' path) against
//! today's FP8 row-scaled copy (`quantize_bf16_to_fp8` + `fp8_gemv_rowscale_batch8_rt2`) and an
//! f64 host reference over the BF16 weights. Design reference only: no code copied or ported
//! from TensorFold/Mia (Apache-2.0); kernels are Metrale's own head_nv4 / nv4_gemv_uncounted.
//!
//! Owner: model-arch examples (DFlash drafter).
//! Checks, per shape (lm_head shard [77440, 4096]; per layer q [2048, 4096], k / v
//! [512, 4096], o [4096, 2048] (row-parallel slice), gate / up [6144, 4096], down
//! [4096, 6144] (row-parallel slice)), Gaussian weights (std 0.02) and N(0, 1) activations:
//! - `ACC`: at M in {1, 2, 4, 8}, NVFP4 and FP8 against the host reference on up to 1024
//!   sampled output columns (every row): max relative error (max |err| / max |ref|) and cosine.
//! - `BITWISE-ROWINV`: every row of the NVFP4 M = 8 launch equals, bit for bit, the M = 1 launch
//!   on that row and the same row of the M = 2 and M = 4 launches.
//! - `TIME`: cold L2 (a 72 MiB read-only flush before each launch), CUDA events around one
//!   launch, median of 60 (5 warm-ups dropped), FP8 and NVFP4 interleaved, M = 8: ms and GB/s
//!   over the weight bytes the launch reads (FP8: 1 B/weight + 4 B/row; NVFP4: 0.5625 B/weight).
//!   A `VOID` line when a rate exceeds 255 GB/s (above the GB10 measured max: L2 was not cold).
//! - `ROUND`: the per-propose-round sum (head x1 + each layer GEMM x5), FP8 vs NVFP4.
//! - Final line `PASS dflash_nv4 (...)` when every shape is ROWINV PASS, NVFP4 cosine >= 0.98 at
//!   every M (sanity bound: a broken copy, not the acceptance gate, which is the serve),
//!   NVFP4 is faster in the round sum and no timing is VOID; else `FAIL dflash_nv4 (...)`
//!   (exit 1), or exit 2 when a kernel is missing.
//!
//! Run (GB10, under a lane lock):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dflash_nv4_drafter_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_arch::glm5next_layer::dense_fp8;
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight, QuantizedWeight};

/// (label, N, K, launches per propose round) of one rank under TP2, gamma 8.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("lm_head", 77440, 4096, 1),
    ("q", 2048, 4096, 5),
    ("k", 512, 4096, 5),
    ("v", 512, 4096, 5),
    ("o", 4096, 2048, 5),
    ("gate", 6144, 4096, 5),
    ("up", 6144, 4096, 5),
    ("down", 4096, 6144, 5),
];
const MS: &[usize] = &[1, 2, 4, 8];
const M_TIME: usize = 8;
const SAMPLE_COLS: usize = 1024;
const COS_MIN: f64 = 0.98;
const GBS_VOID: f64 = 255.0;

// CUDA driver event API, declared as in `glm5next_moe_gateup_m3_staged_microtest/timing.rs`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// Timed reps per arm (median) and untimed warm-up reps.
const REPS: usize = 60;
const WARMUP: usize = 5;
/// Flush weight: 4096 x 32768 NVFP4 = 64 MiB packed + 8 MiB scales, all zero, read by
/// `w4a16_gemv` (read-only: a write flush leaves dirty lines whose write-back lands on the
/// timed kernel).
const FLUSH_N: usize = 4096;
const FLUSH_K: usize = 32768;

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} returned status {rc}");
    }
    Ok(())
}

/// Two timing events.
struct Events([u64; 2]);

impl Events {
    fn new() -> Result<Self> {
        let mut e = [0u64; 2];
        for x in e.iter_mut() {
            // SAFETY: plain driver call writing one event handle into `x`; the backend made
            // the context current when it was built.
            check(unsafe { cuEventCreate(x, 0) }, "cuEventCreate")?;
        }
        Ok(Self(e))
    }

    fn rec(&self, i: usize, s: u64) -> Result<()> {
        // SAFETY: records an event this struct created on stream `s`.
        check(unsafe { cuEventRecord(self.0[i], s) }, "cuEventRecord")
    }

    fn us(&self) -> Result<f64> {
        let mut ms: f32 = 0.0;
        // SAFETY: both events were created and recorded by this struct; `ms` outlives the call.
        unsafe {
            check(cuEventSynchronize(self.0[1]), "cuEventSynchronize")?;
            check(
                cuEventElapsedTime(&mut ms, self.0[0], self.0[1]),
                "cuEventElapsedTime",
            )?;
        }
        Ok(ms as f64 * 1e3)
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        for &e in &self.0 {
            // SAFETY: destroys an event this struct created; errors are ignored on drop.
            unsafe {
                cuEventDestroy_v2(e);
            }
        }
    }
}

/// The 72 MiB read-only flush: its buffers and the `w4a16_gemv` entry.
struct Flush {
    k: KernelHandle,
    w: DevicePtr,
    s: DevicePtr,
    a: DevicePtr,
    c: DevicePtr,
}

impl Flush {
    fn new(g: &dyn GpuBackend) -> Result<Self> {
        let f = Self {
            k: g.kernel("w4a16_gemv", "w4a16_gemv")?,
            w: g.alloc(FLUSH_N * FLUSH_K / 2)?,
            s: g.alloc(FLUSH_N * FLUSH_K / 16)?,
            a: g.alloc(FLUSH_K * 2)?,
            c: g.alloc(FLUSH_N * 2)?,
        };
        g.memset(f.w, 0, FLUSH_N * FLUSH_K / 2)?;
        g.memset(f.s, 0, FLUSH_N * FLUSH_K / 16)?;
        g.memset(f.a, 0, FLUSH_K * 2)?;
        g.synchronize(g.default_stream())?;
        Ok(f)
    }

    fn run(&self, g: &dyn GpuBackend, s: u64) -> Result<()> {
        KernelLaunch::new(g, self.k)
            .grid([div_ceil(FLUSH_N as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.a)
            .arg_ptr(self.w)
            .arg_ptr(self.s)
            .arg_f32(1.0)
            .arg_ptr(self.c)
            .arg_u32(FLUSH_N as u32)
            .arg_u32(FLUSH_K as u32)
            .launch(s)
    }

    fn free(self, g: &dyn GpuBackend) -> Result<()> {
        for p in [self.w, self.s, self.a, self.c] {
            g.free(p)?;
        }
        Ok(())
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// Median microseconds of each arm on stream `s`: per rep and arm, the flush (untimed), then
/// the arm alone between two events. Arms are interleaved within a rep; the first WARMUP reps
/// are dropped.
fn time_cold(
    g: &dyn GpuBackend,
    ev: &Events,
    fl: &Flush,
    s: u64,
    arms: &[&dyn Fn(u64) -> Result<()>],
) -> Result<Vec<f64>> {
    let mut t = vec![Vec::with_capacity(REPS); arms.len()];
    for rep in 0..WARMUP + REPS {
        for (i, arm) in arms.iter().enumerate() {
            fl.run(g, s)?;
            ev.rec(0, s)?;
            arm(s)?;
            ev.rec(1, s)?;
            let us = ev.us()?;
            if rep >= WARMUP {
                t[i].push(us);
            }
        }
    }
    Ok(t.into_iter().map(median).collect())
}

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

fn bits(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn to_f64(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&x| bf16::from_bits(x).to_f64()).collect()
}

/// (max |got - want| / max |want|, cosine) of `got` against `want`.
fn rel_cos(got: &[f64], want: &[f64]) -> (f64, f64) {
    let (mut max_e, mut max_r, mut dot, mut gg, mut ww) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (x, y) in got.iter().zip(want) {
        max_e = max_e.max((x - y).abs());
        max_r = max_r.max(y.abs());
        dot += x * y;
        gg += x * x;
        ww += y * y;
    }
    (
        max_e / max_r.max(1e-30),
        dot / (gg.sqrt() * ww.sqrt()).max(1e-30),
    )
}

/// FP8 and NVFP4 bytes one launch over `[n, k]` reads.
fn fp8_bytes(n: usize, k: usize) -> usize {
    n * k + 4 * n
}
fn nv4_bytes(n: usize, k: usize) -> usize {
    n * k / 2 + n * k / 16
}

struct Kernels {
    quant8: KernelHandle,
    rt2_8: KernelHandle,
}

/// One shape's outcome: row-invariance, worst NVFP4 cosine, (FP8, NVFP4) cold ms at M = 8,
/// and whether a rate was VOID.
struct ShapeOut {
    rowinv: bool,
    nv4_cos_min: f64,
    ms: (f64, f64),
    void: bool,
}

fn run_shape(
    g: &dyn GpuBackend,
    k: &Kernels,
    rng: &mut Rng,
    (label, n, kk): (&str, usize, usize),
    (ev, fl): (&Events, &Flush),
) -> Result<ShapeOut> {
    let s = g.create_stream()?;
    let wh: Vec<bf16> = (0..n * kk)
        .map(|_| bf16::from_f32(rng.normal() * 0.02))
        .collect();
    let w = DenseWeight {
        weight: up(g, &wh)?,
    };
    let w8: Fp8DenseWeight = w.quantize_to_fp8(g, k.quant8, n, kk, s)?;
    g.synchronize(s)?;
    let Some(q): Option<QuantizedWeight> = dense_fp8::quantize_nvfp4_copy(g, w.weight, n, kk, s)?
    else {
        bail!("{label}: quantize_nvfp4_copy built nothing (NVFP4 kernels missing?)");
    };
    let mmax = *MS.iter().max().expect("MS");
    let ah: Vec<bf16> = (0..mmax * kk)
        .map(|_| bf16::from_f32(rng.normal()))
        .collect();
    let a = up(g, &ah)?;
    let c8 = g.alloc(mmax * n * 2)?;
    let c4 = g.alloc(mmax * n * 2)?;
    let cols: Vec<usize> = if n <= SAMPLE_COLS {
        (0..n).collect()
    } else {
        (0..SAMPLE_COLS)
            .map(|j| (j * n / SAMPLE_COLS + j % 7).min(n - 1))
            .collect()
    };
    // f64 host reference on the sampled columns, every row of the widest M.
    let mut want = vec![0f64; mmax * cols.len()];
    for t in 0..mmax {
        let ar = &ah[t * kk..(t + 1) * kk];
        for (j, &col) in cols.iter().enumerate() {
            let wr = &wh[col * kk..(col + 1) * kk];
            want[t * cols.len() + j] = ar
                .iter()
                .zip(wr)
                .map(|(x, y)| x.to_f64() * y.to_f64())
                .sum();
        }
    }
    let pick = |v: &[f64], m: usize| -> Vec<f64> {
        (0..m)
            .flat_map(|t| cols.iter().map(move |&col| v[t * n + col]))
            .collect()
    };
    let fp8 = |m: usize, a: DevicePtr, c: DevicePtr, s: u64| {
        ops::fp8_gemv_rowscale_batch8_rt2(g, k.rt2_8, a, &w8, c, m as u32, n as u32, kk as u32, s)
    };
    let nv4 = |m: usize, a: DevicePtr, c: DevicePtr, s: u64| {
        dense_fp8::nv4_gemv_cuda_core(g, a, &q, c, m, n, kk, s)
    };
    let mut nv4_cos_min = f64::INFINITY;
    let mut rows_by_m: Vec<(usize, Vec<u16>)> = Vec::new();
    for &m in MS {
        fp8(m, a, c8, s)?;
        nv4(m, a, c4, s)?;
        g.synchronize(s)?;
        let (h8, h4) = (bits(g, c8, m * n)?, bits(g, c4, m * n)?);
        let w_m = &want[..m * cols.len()];
        let (r8, cos8) = rel_cos(&pick(&to_f64(&h8), m), w_m);
        let (r4, cos4) = rel_cos(&pick(&to_f64(&h4), m), w_m);
        nv4_cos_min = nv4_cos_min.min(cos4);
        println!(
            "ACC {label:<8} [{n}x{kk}] M={m}: NVFP4 max_rel {r4:.3e} cos {cos4:.6} | FP8 max_rel \
             {r8:.3e} cos {cos8:.6} | vs BF16 host ref, {} cols",
            cols.len()
        );
        rows_by_m.push((m, h4));
    }
    // Row invariance: M = 1 per row, and the M = 2 / 4 rows, against the M = 8 launch.
    let full = &rows_by_m.last().expect("MS").1;
    let mut mism = 0usize;
    for r in 0..mmax {
        nv4(1, a.offset(r * kk * 2), c4, s)?;
        g.synchronize(s)?;
        let one = bits(g, c4, n)?;
        mism += usize::from(one[..] != full[r * n..(r + 1) * n]);
    }
    for (m, h) in &rows_by_m {
        for r in 0..*m {
            mism += usize::from(h[r * n..(r + 1) * n] != full[r * n..(r + 1) * n]);
        }
    }
    let rowinv = mism == 0;
    println!(
        "BITWISE-ROWINV {label:<8} [{n}x{kk}] {} ({mism} row mismatches; M=1 x{mmax} rows and \
         M=1,2,4 prefixes vs M={mmax})",
        if rowinv { "PASS" } else { "FAIL" }
    );
    // Cold-L2 timing at M = 8.
    let arm8 = |st: u64| fp8(M_TIME, a, c8, st);
    let arm4 = |st: u64| nv4(M_TIME, a, c4, st);
    let t = time_cold(g, ev, fl, s, &[&arm8, &arm4])?;
    let (ms8, ms4) = (t[0] / 1e3, t[1] / 1e3);
    let gbs = |b: usize, ms: f64| b as f64 / (ms * 1e-3) / 1e9;
    let (g8, g4) = (gbs(fp8_bytes(n, kk), ms8), gbs(nv4_bytes(n, kk), ms4));
    println!(
        "TIME {label:<8} [{n}x{kk}] M={M_TIME} cold-L2 median of {REPS}: FP8 {ms8:.4} ms \
         {g8:.1} GB/s ({:.1} MB) | NVFP4 {ms4:.4} ms {g4:.1} GB/s ({:.1} MB) | NVFP4/FP8 {:.3}x \
         time",
        fp8_bytes(n, kk) as f64 / 1e6,
        nv4_bytes(n, kk) as f64 / 1e6,
        ms4 / ms8
    );
    let mut void = false;
    for (arm, rate) in [("FP8", g8), ("NVFP4", g4)] {
        if rate > GBS_VOID {
            void = true;
            println!(
                "VOID {label} {arm} {rate:.1} GB/s > {GBS_VOID} (above the GB10 measured max: \
                 L2 not cold, timing not valid)"
            );
        }
    }
    for p in [
        a,
        c8,
        c4,
        w.weight,
        w8.weight,
        w8.row_scale,
        q.weight,
        q.weight_scale,
    ] {
        g.free(p)?;
    }
    Ok(ShapeOut {
        rowinv,
        nv4_cos_min,
        ms: (ms8, ms4),
        void,
    })
}

fn run() -> Result<i32> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let names = [
        ("gemv_fp8w", "quantize_bf16_to_fp8"),
        ("fp8_gemv_rt", "fp8_gemv_rowscale_batch8_rt2"),
        ("w4a16_gemv", "w4a16_gemv"),
        ("w4a16_gemv", "w4a16_gemv_batch8"),
        ("quantize_nvfp4", "quantize_bf16_to_nvfp4_mse"),
    ];
    let mut hs = Vec::new();
    for (module, name) in names {
        match g.kernel(module, name) {
            Ok(h) => hs.push(h),
            Err(_) => println!("MISSING kernel entry {module}::{name}"),
        }
    }
    if hs.len() != names.len() {
        println!(
            "FAIL dflash_nv4 (kernel entries missing: {})",
            names.len() - hs.len()
        );
        return Ok(2);
    }
    let k = Kernels {
        quant8: hs[0],
        rt2_8: hs[1],
    };
    let (ev, fl) = (Events::new()?, Flush::new(g)?);
    let mut rng = Rng(0x0064_666e_7634_0001);
    let (mut rowinv_fail, mut cos_fail, mut voids) = (Vec::new(), Vec::new(), 0usize);
    let (mut round8, mut round4) = (0f64, 0f64);
    for &(label, n, kk, count) in SHAPES {
        let o = run_shape(g, &k, &mut rng, (label, n, kk), (&ev, &fl))?;
        if !o.rowinv {
            rowinv_fail.push(label);
        }
        if o.nv4_cos_min < COS_MIN {
            cos_fail.push(label);
        }
        voids += usize::from(o.void);
        round8 += count as f64 * o.ms.0;
        round4 += count as f64 * o.ms.1;
        println!(
            "SAVING {label:<8} x{count}: {:+.4} ms per round (FP8 - NVFP4)",
            count as f64 * (o.ms.0 - o.ms.1)
        );
    }
    fl.free(g)?;
    println!(
        "ROUND per-rank propose GEMMs at M={M_TIME} (head x1 + q,k,v,o,gate,up,down x5, cold L2): \
         FP8 {round8:.3} ms, NVFP4 {round4:.3} ms, saving {:.3} ms (P3 desk projection 1.44-1.75 \
         ms incl. warm-L2 effects)",
        round8 - round4
    );
    let faster = round4 < round8;
    let ok = rowinv_fail.is_empty() && cos_fail.is_empty() && faster && voids == 0;
    let summary = format!(
        "{} shapes x M {:?}; ROWINV fail {:?}; NVFP4 cos < {COS_MIN} {:?}; round FP8 {round8:.3} \
         -> NVFP4 {round4:.3} ms, saving {:.3} ms; VOID timings {voids}",
        SHAPES.len(),
        MS,
        rowinv_fail,
        cos_fail,
        round8 - round4
    );
    if ok {
        println!("PASS dflash_nv4 ({summary})");
        Ok(0)
    } else {
        println!("FAIL dflash_nv4 ({summary})");
        Ok(1)
    }
}

fn main() {
    match run() {
        Ok(0) => {}
        Ok(c) => std::process::exit(c),
        Err(e) => {
            println!("FAIL dflash_nv4 (error: {e:#})");
            std::process::exit(1);
        }
    }
}
