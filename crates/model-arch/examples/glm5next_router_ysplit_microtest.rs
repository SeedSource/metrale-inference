// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate for `METRALE_GLM_ROUTER_YSPLIT` (y-split launches of
//! `dense_gemv_bf16_fp32out_batchm` and the pipelined twin `dense_gemv_bf16_fp32out_batchm_pf`).
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE router).
//! Checks:
//! - (1) Bitwise: the FP32 logits (u32 bits) of every variant against production (the batchm
//!   kernel at gridDim.y = 1; at M = 1 the single-row `dense_gemv_bf16_fp32out`) at
//!   M = 1,2,3,4,5,7,8,9,16, a 288 x 4096 BF16 router weight (N(0, 0.02)), activations normal
//!   and heavy-tailed, two seeds each. One `PASS`/`FAIL bitwise <variant> M=<m>` line per pair.
//! - (2) Speed, cold: 21 weight copies (49.5 MB > the 24 MB L2, and 42 % 21 == 0 so a copy
//!   is not re-read within 21 launches), one CUDA graph of 42 launches per arm (one MoE
//!   layer each), median of 21 replays, two rounds (forward and reverse arm order, averaged),
//!   at M = 1,2,3,5,7,9. Prints per-launch us and GB/s of the router weight (2.36 MB); a row
//!   above 255 GB/s prints `VOID` (impossible on GB10: L2 hit or byte miscount), never FAIL.
//!   The M = 1 production kernel (`dense_gemv_bf16_fp32out`) is printed for reference.
//! - Final `PASS all` only if every bitwise line passed; exit 0 on PASS, 1 on FAIL, 2 if a
//!   kernel is missing.
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_router_ysplit_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const HIDDEN: usize = 4096;
const N: usize = 288;
const COPIES: usize = 21;
const LAUNCHES: usize = 42;
const REPLAYS: usize = 21;
const WEIGHT_BYTES: f64 = (N * HIDDEN * 2) as f64;
const MAX_GBS: f64 = 255.0;
const BIT_MS: [usize; 9] = [1, 2, 3, 4, 5, 7, 8, 9, 16];
const SPEED_MS: [usize; 6] = [1, 2, 3, 5, 7, 9];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn u(&mut self) -> f32 {
        (self.next() & 0xFF_FFFF) as f32 / 16_777_216.0
    }
    fn normal(&mut self) -> f32 {
        let (a, b) = (self.u().max(1e-7), self.u());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}
fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u32>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}
fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

/// A launch variant: which kernel, and gridDim.y (`0` = one row per block row).
#[derive(Clone, Copy)]
struct Variant {
    label: &'static str,
    pf: bool,
    y: u32,
}

const VARIANTS: [Variant; 10] = [
    Variant {
        label: "y1",
        pf: false,
        y: 1,
    },
    Variant {
        label: "y2",
        pf: false,
        y: 2,
    },
    Variant {
        label: "y3",
        pf: false,
        y: 3,
    },
    Variant {
        label: "y4",
        pf: false,
        y: 4,
    },
    Variant {
        label: "y8",
        pf: false,
        y: 8,
    },
    Variant {
        label: "yrows",
        pf: false,
        y: 0,
    },
    Variant {
        label: "pf",
        pf: true,
        y: 1,
    },
    Variant {
        label: "pf2",
        pf: true,
        y: 2,
    },
    Variant {
        label: "pf4",
        pf: true,
        y: 4,
    },
    Variant {
        label: "pfrows",
        pf: true,
        y: 0,
    },
];

struct Kernels {
    single: KernelHandle,
    batchm: KernelHandle,
    pf: KernelHandle,
}

/// One launch. `v == None` is production: the single-row kernel at M = 1, else the batchm
/// kernel with gridDim.y = 1.
#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    k: &Kernels,
    v: Option<Variant>,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    s: u64,
) -> Result<()> {
    let gx = N.div_ceil(4) as u32;
    if m == 1 && v.is_none() {
        return KernelLaunch::new(g, k.single)
            .grid([gx, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(N as u32)
            .arg_u32(HIDDEN as u32)
            .launch(s);
    }
    let (kern, y) = match v {
        None => (k.batchm, 1),
        Some(v) => (
            if v.pf { k.pf } else { k.batchm },
            if v.y == 0 {
                m as u32
            } else {
                v.y.min(m as u32)
            },
        ),
    };
    KernelLaunch::new(g, kern)
        .grid([gx, y, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(N as u32)
        .arg_u32(HIDDEN as u32)
        .arg_u32(N as u32)
        .launch(s)
}

fn activations(rng: &mut Rng, m: usize, heavy: bool) -> Vec<u8> {
    let v: Vec<f32> = (0..m * HIDDEN)
        .map(|_| {
            if heavy {
                // Heavy tails: a normal times exp(2 * normal), clipped to +-3e4.
                (rng.normal() * (2.0 * rng.normal()).exp()).clamp(-3e4, 3e4)
            } else {
                rng.normal()
            }
        })
        .collect();
    bf16_bytes(&v)
}

/// (1) Bitwise; returns the number of FAIL lines.
fn bitwise(g: &dyn GpuBackend, k: &Kernels, rng: &mut Rng) -> Result<usize> {
    let mut fails = 0;
    let weights: Vec<DevicePtr> = (0..2)
        .map(|_| {
            let w: Vec<f32> = (0..N * HIDDEN).map(|_| rng.normal() * 0.02).collect();
            up(g, &bf16_bytes(&w))
        })
        .collect::<Result<_>>()?;
    for m in BIT_MS {
        // (activation kind, weight index) cases.
        let cases = [(false, 0usize), (false, 1), (true, 0), (true, 1)];
        let acts: Vec<DevicePtr> = cases
            .iter()
            .map(|(heavy, _)| up(g, &activations(rng, m, *heavy)))
            .collect::<Result<_>>()?;
        let bytes = m * N * 4;
        let d_c = g.alloc(bytes)?;
        let mut refs = Vec::new();
        for (i, (_, wi)) in cases.iter().enumerate() {
            g.memset_async(d_c, 0xA5, bytes, 0)?;
            launch(g, k, None, acts[i], weights[*wi], d_c, m, 0)?;
            g.synchronize(0)?;
            refs.push(dn(g, d_c, bytes)?);
        }
        for v in VARIANTS {
            let mut diff = 0usize;
            let mut first = None;
            for (i, (_, wi)) in cases.iter().enumerate() {
                g.memset_async(d_c, 0xA5, bytes, 0)?;
                launch(g, k, Some(v), acts[i], weights[*wi], d_c, m, 0)?;
                g.synchronize(0)?;
                let got = dn(g, d_c, bytes)?;
                for (j, (x, y)) in refs[i].iter().zip(&got).enumerate() {
                    if x != y {
                        diff += 1;
                        first.get_or_insert((i, j));
                    }
                }
            }
            if diff == 0 {
                println!(
                    "PASS bitwise {:<7} M={m:<2} ({} cases, {} u32 each)",
                    v.label,
                    cases.len(),
                    m * N
                );
            } else {
                fails += 1;
                println!(
                    "FAIL bitwise {:<7} M={m:<2}: {diff} u32 differ, first (case, idx) {first:?}",
                    v.label
                );
            }
        }
        for p in acts.into_iter().chain([d_c]) {
            g.free(p)?;
        }
    }
    for p in weights {
        g.free(p)?;
    }
    Ok(fails)
}

/// Median per-launch us of a captured graph of `LAUNCHES` launches rotating over `ws`.
#[allow(clippy::too_many_arguments)]
fn time_arm(
    g: &dyn GpuBackend,
    k: &Kernels,
    v: Option<Variant>,
    ws: &[DevicePtr],
    a: DevicePtr,
    c: DevicePtr,
    m: usize,
    s: u64,
) -> Result<f64> {
    g.begin_capture(s)?;
    for i in 0..LAUNCHES {
        if let Err(e) = launch(g, k, v, a, ws[i % ws.len()], c, m, s) {
            g.abort_capture_if_active(s);
            return Err(e);
        }
    }
    let graph = g.end_capture(s)?;
    for _ in 0..3 {
        g.launch_graph(graph, s)?;
    }
    g.synchronize(s)?;
    let mut us = Vec::new();
    for _ in 0..REPLAYS {
        let (mut e0, mut e1) = (0u64, 0u64);
        let mut ms = 0f32;
        // SAFETY: plain CUDA driver event calls on handles created and destroyed here.
        unsafe {
            check(cuEventCreate(&mut e0, 0), "cuEventCreate")?;
            check(cuEventCreate(&mut e1, 0), "cuEventCreate")?;
            cuEventRecord(e0, s);
        }
        g.launch_graph(graph, s)?;
        // SAFETY: as above; the stream is synchronized before the elapsed time is read.
        unsafe { cuEventRecord(e1, s) };
        g.synchronize(s)?;
        // SAFETY: as above.
        unsafe {
            check(cuEventElapsedTime(&mut ms, e0, e1), "cuEventElapsedTime")?;
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        us.push(ms as f64 * 1e3 / LAUNCHES as f64);
    }
    g.destroy_graph(graph)?;
    Ok(median(&mut us))
}

fn speed(g: &dyn GpuBackend, k: &Kernels, rng: &mut Rng) -> Result<()> {
    let s = g.create_stream()?;
    let ws: Vec<DevicePtr> = (0..COPIES)
        .map(|_| {
            let w: Vec<f32> = (0..N * HIDDEN).map(|_| rng.normal() * 0.02).collect();
            up(g, &bf16_bytes(&w))
        })
        .collect::<Result<_>>()?;
    println!(
        "SPEED cold pool: {COPIES} weight copies = {:.1} MB (L2 24 MB), graph of {LAUNCHES} launches, \
         median of {REPLAYS} replays, 2 rounds (fwd + reverse arm order, averaged)",
        COPIES as f64 * WEIGHT_BYTES / 1e6
    );
    let single = {
        let d_a = up(g, &activations(rng, 1, false))?;
        let d_c = g.alloc(N * 4)?;
        let t = time_arm(g, k, None, &ws, d_a, d_c, 1, s)?;
        g.free(d_a)?;
        g.free(d_c)?;
        t
    };
    println!(
        "SPEED M=1 production dense_gemv_bf16_fp32out: {single:.2} us/launch {:.1} GB/s",
        WEIGHT_BYTES / single / 1e3
    );
    for m in SPEED_MS {
        let d_a = up(g, &activations(rng, m, false))?;
        let d_c = g.alloc(m * N * 4)?;
        // Arms: production first, then every variant.
        let arms: Vec<Option<Variant>> = std::iter::once(None)
            .chain(VARIANTS.iter().map(|v| Some(*v)))
            .collect();
        let mut tot = vec![0f64; arms.len()];
        for round in 0..2 {
            let order: Vec<usize> = if round == 0 {
                (0..arms.len()).collect()
            } else {
                (0..arms.len()).rev().collect()
            };
            for i in order {
                tot[i] += time_arm(g, k, arms[i], &ws, d_a, d_c, m, s)? / 2.0;
            }
        }
        let prod = tot[0];
        for (i, arm) in arms.iter().enumerate() {
            let label = arm.map_or("prod", |v| v.label);
            let gbs = WEIGHT_BYTES / tot[i] / 1e3;
            let flag = if gbs > MAX_GBS { " VOID" } else { "" };
            println!(
                "SPEED M={m} {label:<7} us={:.2} GBs={gbs:.1} vs_prod={:.3}x{flag}",
                tot[i],
                prod / tot[i]
            );
        }
        g.free(d_a)?;
        g.free(d_c)?;
    }
    for p in ws {
        g.free(p)?;
    }
    Ok(())
}

fn run() -> Result<i32> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let names = [
        ("gemv", "dense_gemv_bf16_fp32out"),
        ("dense_gemv_bf16_batchm", "dense_gemv_bf16_fp32out_batchm"),
        (
            "dense_gemv_bf16_batchm",
            "dense_gemv_bf16_fp32out_batchm_pf",
        ),
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
        single: hs[0],
        batchm: hs[1],
        pf: hs[2],
    };
    let mut rng = Rng(0x0072_7472_7973_0001);
    let fails = bitwise(g, &k, &mut rng)?;
    speed(g, &k, &mut rng)?;
    if fails == 0 {
        println!("PASS all");
        Ok(0)
    } else {
        println!("FAIL bitwise: {fails} (variant, M) pairs differ from production");
        Ok(1)
    }
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            println!("FAIL microtest error: {e:#}");
            std::process::exit(1);
        }
    }
}
