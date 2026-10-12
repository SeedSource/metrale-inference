// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate for `METRALE_DFLASH_NV4_HEAD` / `METRALE_DFLASH_NV4_LAYERS` (and
//! `METRALE_DFLASH_NV4_TC`): the DFlash2 drafter's per-rank GEMMs (GLM-5.3,
//! `METRALE_DFLASH_DRAFT_TP2`, gamma 8) on three routes: today's FP8 row-scaled copy
//! (`quantize_bf16_to_fp8` + `fp8_gemv_rowscale_batch8_rt2`), the NVFP4 copy
//! (`dense_fp8::quantize_nvfp4_copy`) on the CUDA-core tiers (`dense_fp8::nv4_gemv_cuda_core`,
//! "NVFP4-cc") and the same copy on the tensor-core `w4a16_gemv_tc8/tc16`
//! (`dense_fp8::nv4_gemv_tensor_core`, "NVFP4-tc", `METRALE_DFLASH_NV4_TC`), against an f64 host
//! reference over the BF16 weights. Design reference only: no code copied or ported from
//! TensorFold/Mia (Apache-2.0); kernels are Metrale's own head_nv4 / nv4_gemv_uncounted.
//!
//! Owner: model-arch examples (DFlash drafter).
//! Checks, per shape (lm_head shard [77440, 4096]; per layer q [2048, 4096], k / v
//! [512, 4096], o [4096, 2048] (row-parallel slice), gate / up [6144, 4096], down
//! [4096, 6144] (row-parallel slice)), Gaussian weights (std 0.02) and N(0, 1) activations:
//! - `ACC`: at M in {1, 2, 4, 8}, each route against the host reference on up to 1024 sampled
//!   output columns (every row): max relative error (max |err| / max |ref|) and cosine.
//! - `BITWISE-ROWINV <route>`: for NVFP4-cc and NVFP4-tc, every row of the M = 8 launch equals,
//!   bit for bit, the M = 1 launch on that row and the same row of the M = 2 and M = 4 launches.
//!   Information only (the drafter path is acceptance-only): printed honestly, not gated.
//! - `TIME`: cold L2 (a 72 MiB read-only flush before each launch), CUDA events around one
//!   launch, median of 60 (5 warm-ups dropped), the three routes interleaved, M = 8: ms and GB/s
//!   over the weight bytes the launch reads (FP8: 1 B/weight + 4 B/row; NVFP4: 0.5625 B/weight).
//!   A `VOID` line when a rate exceeds 255 GB/s (above the GB10 measured max: L2 was not cold).
//! - `ROUND <route>`: the per-propose-round sum (head x1 + each layer GEMM x5) per route.
//! - Final line `PASS dflash_nv4 (winner <route>, ...)` when the round saving FP8 - best NVFP4
//!   route is >= 0.8 ms, that route's cosine is >= 0.98 at every M and shape, and no timing is
//!   VOID; else `FAIL dflash_nv4 (...)` (exit 1), or exit 2 when a kernel is missing.
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
/// Round saving (FP8 - best NVFP4 route, ms) the gate needs (race-lead 2026-10-12).
const SAVING_MIN_MS: f64 = 0.8;
/// The routes, in column order.
const ROUTES: [&str; 3] = ["FP8", "NVFP4-cc", "NVFP4-tc"];

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

/// One shape's outcome per route ([`ROUTES`] order): row-invariance (FP8 not tested: `true`),
/// worst cosine, cold ms at M = 8; and whether a rate was VOID.
struct ShapeOut {
    rowinv: [bool; 3],
    cos_min: [f64; 3],
    ms: [f64; 3],
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
    if !kk.is_multiple_of(128) {
        bail!("{label}: K {kk} % 128 != 0, the tensor-core route cannot take it");
    }
    let mmax = *MS.iter().max().expect("MS");
    let ah: Vec<bf16> = (0..mmax * kk)
        .map(|_| bf16::from_f32(rng.normal()))
        .collect();
    let a = up(g, &ah)?;
    let outs: Vec<DevicePtr> = (0..ROUTES.len())
        .map(|_| g.alloc(mmax * n * 2))
        .collect::<Result<_>>()?;
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
    let launch = |route: usize, m: usize, a: DevicePtr, c: DevicePtr, s: u64| match route {
        0 => ops::fp8_gemv_rowscale_batch8_rt2(
            g, k.rt2_8, a, &w8, c, m as u32, n as u32, kk as u32, s,
        ),
        1 => dense_fp8::nv4_gemv_cuda_core(g, a, &q, c, m, n, kk, s),
        _ => dense_fp8::nv4_gemv_tensor_core(g, a, &q, c, m, n, kk, s),
    };
    let mut cos_min = [f64::INFINITY; 3];
    // Per route, the output bits at each M.
    let mut rows_by_m: Vec<Vec<(usize, Vec<u16>)>> = vec![Vec::new(); ROUTES.len()];
    for &m in MS {
        let mut line = Vec::new();
        for (route, &c) in outs.iter().enumerate() {
            launch(route, m, a, c, s)?;
            g.synchronize(s)?;
            let h = bits(g, c, m * n)?;
            let (r, cos) = rel_cos(&pick(&to_f64(&h), m), &want[..m * cols.len()]);
            cos_min[route] = cos_min[route].min(cos);
            line.push(format!("{} max_rel {r:.3e} cos {cos:.6}", ROUTES[route]));
            rows_by_m[route].push((m, h));
        }
        println!(
            "ACC {label:<8} [{n}x{kk}] M={m}: {} | vs BF16 host ref, {} cols",
            line.join(" | "),
            cols.len()
        );
    }
    // Row invariance per NVFP4 route: M = 1 per row, and the M = 2 / 4 rows, against M = 8.
    let mut rowinv = [true; 3];
    for route in 1..ROUTES.len() {
        let full = &rows_by_m[route].last().expect("MS").1;
        let c = outs[route];
        let mut mism = 0usize;
        for r in 0..mmax {
            launch(route, 1, a.offset(r * kk * 2), c, s)?;
            g.synchronize(s)?;
            let one = bits(g, c, n)?;
            mism += usize::from(one[..] != full[r * n..(r + 1) * n]);
        }
        for (m, h) in &rows_by_m[route] {
            for r in 0..*m {
                mism += usize::from(h[r * n..(r + 1) * n] != full[r * n..(r + 1) * n]);
            }
        }
        rowinv[route] = mism == 0;
        println!(
            "BITWISE-ROWINV {} {label:<8} [{n}x{kk}] {} ({mism} row mismatches; M=1 x{mmax} rows \
             and M=1,2,4 prefixes vs M={mmax}; information only)",
            ROUTES[route],
            if rowinv[route] { "PASS" } else { "FAIL" }
        );
    }
    // Cold-L2 timing at M = 8, the three routes interleaved.
    let arm0 = |st: u64| launch(0, M_TIME, a, outs[0], st);
    let arm1 = |st: u64| launch(1, M_TIME, a, outs[1], st);
    let arm2 = |st: u64| launch(2, M_TIME, a, outs[2], st);
    let t = time_cold(g, ev, fl, s, &[&arm0, &arm1, &arm2])?;
    let ms = [t[0] / 1e3, t[1] / 1e3, t[2] / 1e3];
    let bytes = [fp8_bytes(n, kk), nv4_bytes(n, kk), nv4_bytes(n, kk)];
    let mut void = false;
    let mut line = Vec::new();
    for route in 0..ROUTES.len() {
        let rate = bytes[route] as f64 / (ms[route] * 1e-3) / 1e9;
        line.push(format!(
            "{} {:.4} ms {rate:.1} GB/s ({:.1} MB)",
            ROUTES[route],
            ms[route],
            bytes[route] as f64 / 1e6
        ));
        if rate > GBS_VOID {
            void = true;
            println!(
                "VOID {label} {} {rate:.1} GB/s > {GBS_VOID} (above the GB10 measured max: \
                 L2 not cold, timing not valid)",
                ROUTES[route]
            );
        }
    }
    println!(
        "TIME {label:<8} [{n}x{kk}] M={M_TIME} cold-L2 median of {REPS}: {} | cc/FP8 {:.3}x \
         tc/FP8 {:.3}x time",
        line.join(" | "),
        ms[1] / ms[0],
        ms[2] / ms[0]
    );
    for p in outs.iter().copied().chain([
        a,
        w.weight,
        w8.weight,
        w8.row_scale,
        q.weight,
        q.weight_scale,
    ]) {
        g.free(p)?;
    }
    Ok(ShapeOut {
        rowinv,
        cos_min,
        ms,
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
        ("w4a16_gemv_tc", "w4a16_gemv_tc8"),
        ("w4a16_gemv_tc", "w4a16_gemv_tc16"),
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
    let mut rowinv_fail: [Vec<&str>; 3] = Default::default();
    let mut cos_fail: [Vec<&str>; 3] = Default::default();
    let (mut voids, mut round) = (0usize, [0f64; 3]);
    for &(label, n, kk, count) in SHAPES {
        let o = run_shape(g, &k, &mut rng, (label, n, kk), (&ev, &fl))?;
        for route in 0..ROUTES.len() {
            if !o.rowinv[route] {
                rowinv_fail[route].push(label);
            }
            if o.cos_min[route] < COS_MIN {
                cos_fail[route].push(label);
            }
            round[route] += count as f64 * o.ms[route];
        }
        voids += usize::from(o.void);
        println!(
            "SAVING {label:<8} x{count}: NVFP4-cc {:+.4} ms, NVFP4-tc {:+.4} ms per round (FP8 - \
             route)",
            count as f64 * (o.ms[0] - o.ms[1]),
            count as f64 * (o.ms[0] - o.ms[2])
        );
    }
    fl.free(g)?;
    for route in 0..ROUTES.len() {
        println!(
            "ROUND {} per-rank propose GEMMs at M={M_TIME} (head x1 + q,k,v,o,gate,up,down x5, \
             cold L2): {:.3} ms, saving vs FP8 {:.3} ms; ROWINV fail {:?}; cos < {COS_MIN} {:?}",
            ROUTES[route],
            round[route],
            round[0] - round[route],
            rowinv_fail[route],
            cos_fail[route]
        );
    }
    let best = if round[2] < round[1] { 2 } else { 1 };
    let saving = round[0] - round[best];
    let ok = saving >= SAVING_MIN_MS && cos_fail[best].is_empty() && voids == 0;
    let summary = format!(
        "winner {}: round FP8 {:.3} -> {:.3} ms, saving {saving:.3} ms (need >= {SAVING_MIN_MS}); \
         NVFP4-cc {:.3} ms, NVFP4-tc {:.3} ms; winner cos < {COS_MIN} {:?}; ROWINV fail cc {:?} \
         tc {:?} (info); VOID timings {voids}; {} shapes x M {:?}",
        ROUTES[best],
        round[0],
        round[best],
        round[1],
        round[2],
        cos_fail[best],
        rowinv_fail[1],
        rowinv_fail[2],
        SHAPES.len(),
        MS
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
