// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Row invariance and speed of `METRALE_CUBLAS_BF16_ALGO_PIN` (one cuBLASLt BF16
//! algorithm per (N, K, out type), chosen at M = 8192) against the default per-M heuristic.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - Bitwise leg: for each shape and arm, one call over all `M_ALL` rows, then each partition in
//!   `SPLITS` as consecutive calls on row slices of the same buffers; every output row of every
//!   piece must carry the whole call's bits (byte compare, so -0.0/+0.0 and NaN payloads count).
//!   The pinned arm must match on every row with zero pin fallbacks; the heuristic arm's row
//!   counts are printed to show the instrument can see a difference.
//! - Speed leg: host wall-clock around one launch + stream synchronize (the launch/sync overhead
//!   is the same for both arms), 3 warm-up then `REPS` timed launches per (shape, M, arm). Per M,
//!   the summed pinned time over the shapes must be <= `SLOW_BIG` x the heuristic's at M >= 2048
//!   and <= `SLOW_SMALL` x at smaller M.
//! - Shapes are the classes the GLM wide projections hit (router logits and indexer weights with
//!   FP32 out; narrow and wide N; K 128 / 1536 / 4096); the serve log prints the real ones
//!   (`cuBLASLt BF16 pin N=.. K=..`).
//! - Prints `PASS ...` only if both legs pass; otherwise `FAIL ...` and an error.
//!
//! Run (GB10): cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!   --example cublas_bf16_pin_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cublaslt;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use std::time::Instant;

const M_ALL: usize = 8192;
/// Row partitions of `M_ALL`; the first is the gridstatic off-grid case (2204-row prefix call,
/// then the 2183-row suffix of a 4387-token prompt).
const SPLITS: [&[usize]; 5] = [
    &[2204, 2183, 3805],
    &[17, 2031, 6144],
    &[4096, 4096],
    &[2339, 5853],
    &[100, 8092],
];
const SPEED_MS: [usize; 5] = [64, 512, 2339, 4387, 8192];
const WARMUP: usize = 3;
const REPS: usize = 20;
const SLOW_BIG: f64 = 1.05;
const SLOW_SMALL: f64 = 1.25;
/// (N, K, FP32 out, class).
const SHAPES: [(usize, usize, bool, &str); 8] = [
    (288, 4096, true, "router_logits"),
    (32, 4096, true, "idx_weights"),
    (4096, 1536, false, "idx_wq_b"),
    (128, 4096, false, "idx_wk"),
    (64, 4096, false, "kda_gate_narrow"),
    (4096, 128, false, "kda_lowrank_up"),
    (4096, 4096, false, "dense_square"),
    (16384, 1536, false, "q_absorb_wide"),
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// BF16 little-endian bytes, uniform in [-amp, amp).
    fn bf16_bytes(&mut self, count: usize, amp: f32) -> Vec<u8> {
        (0..count)
            .flat_map(|_| {
                let u = (self.next() >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
                bf16::from_f32(u * amp).to_bits().to_le_bytes()
            })
            .collect()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

#[allow(clippy::too_many_arguments)]
fn gemm(
    a: DevicePtr,
    w: DevicePtr,
    o: DevicePtr,
    row0: usize,
    m: usize,
    (n, k, f32_out): (usize, usize, bool),
    pin: Option<u32>,
) -> Result<()> {
    let esize = if f32_out { 4 } else { 2 };
    cublaslt::bf16_gemm_act_weight_t_with(
        a.0 + (row0 * k * 2) as u64,
        w.0,
        o.0 + (row0 * n * esize) as u64,
        m as u32,
        n as u32,
        k as u32,
        f32_out,
        pin,
        0,
    )
}

fn time_ms(gpu: &dyn GpuBackend, f: &dyn Fn() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        f()?;
    }
    gpu.synchronize(0)?;
    let mut sum = 0.0;
    for _ in 0..REPS {
        let t = Instant::now();
        f()?;
        gpu.synchronize(0)?;
        sum += t.elapsed().as_secs_f64() * 1e3;
    }
    Ok(sum / REPS as f64)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let pin = Some(cublaslt::BF16_PIN_M);
    let mut rng = Rng(0x5049_4e31_3030_3037);
    let mut failures: Vec<String> = Vec::new();
    let mut heur_rows_total = 0usize;
    let mut t_sum = vec![(0f64, 0f64); SPEED_MS.len()];

    for (n, k, f32_out, name) in SHAPES {
        let shape = (n, k, f32_out);
        let esize = if f32_out { 4 } else { 2 };
        let row_bytes = n * esize;
        let desc = cublaslt::bf16_pin_describe(cublaslt::BF16_PIN_M, n as u32, k as u32, f32_out)?;
        println!(
            "PIN {name} N={n} K={k} out={}: {}",
            if f32_out { "f32" } else { "bf16" },
            desc.as_deref().unwrap_or("NONE (no single-pass candidate)")
        );
        if desc.is_none() {
            failures.push(format!("{name}: no single-pass candidate"));
        }
        let a = upload(&gpu, &rng.bf16_bytes(M_ALL * k, 1.0))?;
        let w = upload(&gpu, &rng.bf16_bytes(n * k, 0.1))?;
        let whole = gpu.alloc(M_ALL * row_bytes)?;
        let parts = gpu.alloc(M_ALL * row_bytes)?;

        for (arm, p) in [("pinned", pin), ("heuristic", None)] {
            gpu.memset(whole, 0, M_ALL * row_bytes)?;
            gemm(a, w, whole, 0, M_ALL, shape, p)?;
            gpu.synchronize(0)?;
            let mut h_whole = vec![0u8; M_ALL * row_bytes];
            gpu.copy_d2h(whole, &mut h_whole)?;
            for split in SPLITS {
                gpu.memset(parts, 0xff, M_ALL * row_bytes)?;
                let mut row0 = 0;
                for &m in split {
                    gemm(a, w, parts, row0, m, shape, p)?;
                    row0 += m;
                }
                gpu.synchronize(0)?;
                let mut h_parts = vec![0u8; M_ALL * row_bytes];
                gpu.copy_d2h(parts, &mut h_parts)?;
                let bad = h_whole
                    .chunks_exact(row_bytes)
                    .zip(h_parts.chunks_exact(row_bytes))
                    .filter(|(x, y)| x != y)
                    .count();
                println!("ROWS {name} arm={arm} split={split:?}: {bad} of {M_ALL} rows differ");
                if arm == "pinned" && bad > 0 {
                    failures.push(format!("{name} split {split:?}: {bad} rows differ"));
                }
                if arm == "heuristic" {
                    heur_rows_total += bad;
                }
            }
        }

        for (i, &m) in SPEED_MS.iter().enumerate() {
            let tp = time_ms(&gpu, &|| gemm(a, w, whole, 0, m, shape, pin))?;
            let th = time_ms(&gpu, &|| gemm(a, w, whole, 0, m, shape, None))?;
            let tf = |ms: f64| 2.0 * (m * n * k) as f64 / (ms * 1e-3) / 1e12;
            println!(
                "TIMING {name} M={m} N={n} K={k}: heuristic {th:.4} ms {:.1} TF/s | pinned \
                 {tp:.4} ms {:.1} TF/s | pinned/heuristic {:.3}",
                tf(th),
                tf(tp),
                tp / th
            );
            t_sum[i].0 += th;
            t_sum[i].1 += tp;
        }
        for p in [a, w, whole, parts] {
            gpu.free(p)?;
        }
    }

    for (i, &m) in SPEED_MS.iter().enumerate() {
        let (th, tp) = t_sum[i];
        let bar = if m >= 2048 { SLOW_BIG } else { SLOW_SMALL };
        let r = tp / th;
        println!(
            "SPEED M={m}: sum heuristic {th:.3} ms, pinned {tp:.3} ms, ratio {r:.3} (bar {bar})"
        );
        if r > bar {
            failures.push(format!("M={m}: pinned/heuristic {r:.3} > {bar}"));
        }
    }
    let fb = cublaslt::bf16_pin_fallbacks();
    println!(
        "FALLBACKS {fb}; heuristic arm rows differing (all shapes and splits): {heur_rows_total}"
    );
    if fb > 0 {
        failures.push(format!("{fb} pin fallbacks"));
    }
    if !failures.is_empty() {
        println!("FAIL cublas_bf16_pin_microtest: {}", failures.join("; "));
        bail!("cublas_bf16_pin_microtest failed");
    }
    println!(
        "PASS cublas_bf16_pin_microtest: pinned rows bitwise M-invariant on {} shapes x {} splits, \
         0 fallbacks, speed within bar",
        SHAPES.len(),
        SPLITS.len()
    );
    Ok(())
}
