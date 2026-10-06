// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Correctness and throughput check for `w8a16_gemm` and
//! `w8a16_gemm_pipelined` (FP8 E4M3 weights x BF16 activations, 128 x 128 FP32
//! block scales).
//!
//! The BF16 output is compared with a CPU reference that uses the kernels'
//! two-level FP32 accumulation (inner sum over one 128-wide K block, then
//! `outer += inner * block_scale`). The tensor-core MMA sums within a block in
//! a different order than the CPU loop, so the gate is cosine >= `COSINE_GATE`,
//! not byte equality. Throughput is printed twice: wall-clock around launches
//! that each synchronize, and GPU time from CUDA events around unsynchronized
//! launches.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Usage:
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!       --example w8a16_microtest -- [kernel_name] [M] [N] [K] [hex seed]
//! Defaults: w8a16_gemm 128 512 2048 0x51A7
//!
//! K must be a multiple of 128. Exit code 0 = PASS (cosine >= `COSINE_GATE`), 1 = FAIL.

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

// 2026-09-25: CUDA driver event API, for GPU-only timing: the wall-clock figure
// includes a host synchronize per launch.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-09-25: FP8 block size along both N and K.
const FP8_BLOCK: usize = 128;
/// 2026-09-25: Minimum cosine against the CPU reference for a pass.
const COSINE_GATE: f64 = 0.9995;

// 2026-09-25: splitmix64, seeded from the command line, so a run is reproducible.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// 2026-09-25: f32 to BF16 bits, round to nearest even; NaN stays a quiet NaN.
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// 2026-09-25: E4M3 (e4m3fn) decode: 1 sign, 4 exponent (bias 7), 3 mantissa
/// bits; no Inf, and S.1111.111 is the only NaN. Written out here rather than
/// taken from the kernel's `E4M3_LUT`, so the check does not reuse it.
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2f32.powi(-6)
    } else if exp == 0x0F && mant == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s_to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 2026-09-25: CPU reference with the kernels' two-level FP32 accumulation: inner
/// sums one full 128-wide K block, then `outer += inner * scale`.
fn cpu_reference(
    a_bf16: &[u16],
    b_fp8: &[u8],
    scale: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<u16> {
    let k_blocks = k / FP8_BLOCK;
    let n_blocks = n.div_ceil(FP8_BLOCK);
    let _ = n_blocks;
    let mut out = vec![0u16; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut outer = 0.0f32;
            for kb in 0..k_blocks {
                let mut inner = 0.0f32;
                for kk in 0..FP8_BLOCK {
                    let g_k = kb * FP8_BLOCK + kk;
                    let a = bf16_bits_to_f32(a_bf16[row * k + g_k]);
                    let b = e4m3_to_f32(b_fp8[col * k + g_k]);
                    inner += a * b;
                }
                let scl = scale[(col / FP8_BLOCK) * k_blocks + kb];
                outer += inner * scl;
            }
            out[row * n + col] = f32_to_bf16_bits(outer);
        }
    }
    out
}

/// 2026-09-25: Launch `name` with its geometry and synchronize the stream. Any
/// other kernel name is an error.
fn launch(
    gpu: &dyn GpuBackend,
    name: &str,
    ptrs: [DevicePtr; 4],
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let [a, b, scale, c] = ptrs;
    let handle = gpu.kernel(name, name)?;
    let (grid, block) = match name {
        // 2026-09-25: The geometry of `ops::w8a16_gemm`: a 256x128 (MxN) tile and
        // 512 threads under `metrale_hip`, a 64x64 tile and 128 threads otherwise.
        #[cfg(metrale_hip)]
        "w8a16_gemm" => ([n.div_ceil(128), m.div_ceil(256), 1], [512u32, 1, 1]),
        #[cfg(not(metrale_hip))]
        "w8a16_gemm" => ([n.div_ceil(64), m.div_ceil(64), 1], [128u32, 1, 1]),
        // 2026-09-25: `w8a16_gemm_pipelined`: a 128x32 (MxN) tile, 256 threads.
        "w8a16_gemm_pipelined" => ([n.div_ceil(32), m.div_ceil(128), 1], [256u32, 1, 1]),
        other => bail!("no launch geometry registered for kernel '{other}' — add an arm"),
    };
    KernelLaunch::new(gpu, handle)
        .grid(grid)
        .block(block)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(scale)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(())
}

/// 2026-09-25: `launch` without the synchronize, for the CUDA-event timing loop.
fn launch_no_sync(
    gpu: &dyn GpuBackend,
    name: &str,
    ptrs: [DevicePtr; 4],
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let [a, b, scale, c] = ptrs;
    let handle = gpu.kernel(name, name)?;
    let (grid, block) = match name {
        #[cfg(metrale_hip)]
        "w8a16_gemm" => ([n.div_ceil(128), m.div_ceil(256), 1], [512u32, 1, 1]),
        #[cfg(not(metrale_hip))]
        "w8a16_gemm" => ([n.div_ceil(64), m.div_ceil(64), 1], [128u32, 1, 1]),
        "w8a16_gemm_pipelined" => ([n.div_ceil(32), m.div_ceil(128), 1], [256u32, 1, 1]),
        other => bail!("no launch geometry registered for kernel '{other}' — add an arm"),
    };
    KernelLaunch::new(gpu, handle)
        .grid(grid)
        .block(block)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(scale)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let kernel = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "w8a16_gemm".to_string());
    let m: usize = args.get(2).map_or(128, |s| s.parse().unwrap());
    let n: usize = args.get(3).map_or(512, |s| s.parse().unwrap());
    let k: usize = args.get(4).map_or(2048, |s| s.parse().unwrap());
    let seed: u64 = args.get(5).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });

    if !k.is_multiple_of(FP8_BLOCK) {
        bail!("K ({k}) must be a multiple of FP8_BLOCK ({FP8_BLOCK}) for the clean-block path");
    }
    println!("=== w8a16 microtest: kernel='{kernel}' M={m} N={n} K={k} seed=0x{seed:X} ===");

    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    // 2026-09-25: FP8 weights with exponent field 0..=7: magnitude at most 1.875,
    // and never the NaN code.
    let b_fp8: Vec<u8> = (0..n * k)
        .map(|_| {
            let sign = (rng.next_u64() & 1) as u8;
            let exp = (rng.next_u64() % 8) as u8;
            let mant = (rng.next_u64() % 8) as u8;
            (sign << 7) | (exp << 3) | mant
        })
        .collect();
    let k_blocks = k / FP8_BLOCK;
    let n_blocks = n.div_ceil(FP8_BLOCK);
    let scale: Vec<f32> = (0..n_blocks * k_blocks)
        .map(|_| rng.uniform(0.5, 1.5))
        .collect();

    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
    let b_ptr = upload_bytes(gpu, &b_fp8)?;
    let s_ptr = upload_bytes(gpu, &f32s_to_le(&scale))?;
    let c_ptr = gpu.alloc(m * n * 2)?;
    let ptrs = [a_ptr, b_ptr, s_ptr, c_ptr];

    launch(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream)?;
    let mut c_raw = vec![0u8; m * n * 2];
    gpu.copy_d2h(c_ptr, &mut c_raw)?;
    let c_gpu: Vec<u16> = c_raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let c_cpu = cpu_reference(&a_bf16, &b_fp8, &scale, m, n, k);

    let (mut dot, mut ng, mut nc, mut max_rel, mut sum_rel) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..m * n {
        let g = bf16_bits_to_f32(c_gpu[i]) as f64;
        let c = bf16_bits_to_f32(c_cpu[i]) as f64;
        dot += g * c;
        ng += g * g;
        nc += c * c;
        let rel = (g - c).abs() / c.abs().max(1e-3);
        max_rel = max_rel.max(rel);
        sum_rel += rel;
    }
    let cosine = dot / (ng.sqrt() * nc.sqrt());
    let mean_rel = sum_rel / (m * n) as f64;

    let iters = 50;
    for _ in 0..5 {
        launch(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream)?;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        launch(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream)?;
    }
    let per_iter_s = t0.elapsed().as_secs_f64() / iters as f64;
    let tflops = (2.0 * m as f64 * n as f64 * k as f64) / per_iter_s / 1e12;

    // 2026-09-25: GPU time for `iters` back-to-back unsynchronized launches,
    // between two events recorded on the launch stream.
    let (mut ev_start, mut ev_end): (u64, u64) = (0, 0);
    let rc = unsafe { cuEventCreate(&mut ev_start, 0) };
    if rc != 0 {
        bail!("cuEventCreate(start) failed: status {rc}");
    }
    let rc = unsafe { cuEventCreate(&mut ev_end, 0) };
    if rc != 0 {
        bail!("cuEventCreate(end) failed: status {rc}");
    }
    let rc = unsafe { cuEventRecord(ev_start, stream) };
    if rc != 0 {
        bail!("cuEventRecord(start) failed: status {rc}");
    }
    for _ in 0..iters {
        launch_no_sync(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream)?;
    }
    let rc = unsafe { cuEventRecord(ev_end, stream) };
    if rc != 0 {
        bail!("cuEventRecord(end) failed: status {rc}");
    }
    let rc = unsafe { cuEventSynchronize(ev_end) };
    if rc != 0 {
        bail!("cuEventSynchronize(end) failed: status {rc}");
    }
    let mut elapsed_ms: f32 = 0.0;
    let rc = unsafe { cuEventElapsedTime(&mut elapsed_ms, ev_start, ev_end) };
    if rc != 0 {
        bail!("cuEventElapsedTime failed: status {rc}");
    }
    unsafe {
        cuEventDestroy_v2(ev_start);
        cuEventDestroy_v2(ev_end);
    }
    let kernel_s = (elapsed_ms as f64 / 1e3) / iters as f64;
    let kernel_tflops = (2.0 * m as f64 * n as f64 * k as f64) / kernel_s / 1e12;

    for p in ptrs {
        gpu.free(p).ok();
    }

    println!("cosine={cosine:.6}  mean_rel={mean_rel:.2e}  max_rel={max_rel:.2e}");
    println!(
        "perf: {:.3} ms/iter  ~{tflops:.2} TFLOP/s (wall-clock incl. launch)",
        per_iter_s * 1e3
    );
    println!(
        "kernel-only: {:.4} ms/iter  ~{kernel_tflops:.2} TFLOP/s (CUDA events)",
        kernel_s * 1e3
    );

    if cosine >= COSINE_GATE && cosine.is_finite() {
        println!("RESULT: PASS (cosine {cosine:.6} >= {COSINE_GATE})");
        Ok(())
    } else {
        eprintln!(
            "RESULT: FAIL (cosine {cosine:.6} < {COSINE_GATE}) — layout/dequant/accumulation mismatch"
        );
        std::process::exit(1);
    }
}
