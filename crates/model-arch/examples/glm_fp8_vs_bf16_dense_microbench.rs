// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Speed of the in-tree W8A8 block-scaled FP8 GEMM against cuBLASLt BF16 at the
//! GLM-5.3 dense prefill projection shapes, to decide whether an FP8 dense prefill path is
//! worth building.
//!
//! Owner: model-arch examples (GPU microbenches).
//! Invariants:
//! - Timing is the point. The cosine of the FP8 output against the BF16 output is only a
//!   wiring check: the run prints `PASS: timings recorded` only if every non-skipped arm ran
//!   and every cosine is > `COS_GATE`; otherwise it prints a `FAIL` line and returns an error.
//! - Both legs start from the same values: A and W are random BF16; the FP8 weight is those
//!   W values quantized to E4M3 with one FP32 scale per 128x128 block (`[N/128, K/128]`), and
//!   the FP8 activation is `per_token_group_quant_fp8` of the same BF16 A.
//! - A shape the pipelined FP8 kernel's contract rejects (K % 128, N % 64) is skipped with a
//!   `SKIP` line.
//!
//! Arms, C[m,n] = sum_k A[m,k] * W[n,k]:
//! - `bf16_cublaslt`: `ops::cublas_bf16_proj_dense`, the call the GLM prefill makes for dense
//!   projections (it forwards to `cublaslt::bf16_gemm_act_weight_t`).
//! - `fp8_quant`: `ops::per_token_group_quant_fp8` from the BF16 A.
//! - `fp8_gemm`: `ops::fp8_gemm_t_blockscaled`, which launches `fp8_gemm_blockscaled_pipe_128x64`.
//! - `fp8_total`: quant then gemm per launch.
//!
//! Times are host wall-clock around one launch followed by a stream synchronize (the backend
//! has no event-elapsed call), so each carries a launch/sync overhead of tens of microseconds;
//! it is the same for every arm. 3 warm-up launches, then `REPS` timed ones; mean and min.
//!
//! Run (GB10): cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!   --example glm_fp8_vs_bf16_dense_microbench

use anyhow::{Result, bail, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use std::time::Instant;

const WARMUP: usize = 3;
const REPS: usize = 20;
const COS_GATE: f64 = 0.99;
/// Largest finite E4M3 value; the block scale is amax / this.
const E4M3_MAX: f32 = 448.0;
const MS: [usize; 2] = [8192, 2048];
/// (N, K, name): the dense prefill projections of GLM-5.3-Flash.
const SHAPES: [(usize, usize, &str); 5] = [
    (4096, 4096, "kda_qkvo"),
    (4096, 16384, "dsa_o_absorb"),
    (16384, 1536, "dsa_q_absorb"),
    (1024, 4096, "shared_gate_up"),
    (4096, 1024, "shared_down"),
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
    /// BF16 bit patterns, uniform in [-amp, amp).
    fn bf16_bits(&mut self, count: usize, amp: f32) -> Vec<u16> {
        (0..count)
            .map(|_| {
                let u = (self.next() >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
                bf16::from_f32(u * amp).to_bits()
            })
            .collect()
    }
}

fn le_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

/// Value of the non-negative E4M3 code `c` (0x00..=0x7E; 0x7F is NaN).
fn e4m3_value(c: u8) -> f32 {
    let (e, m) = ((c >> 3) & 0xf, (c & 7) as f32);
    if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7)
    }
}

/// Nearest E4M3 code (sign included) to `x`, by search over the monotonic positive table.
fn e4m3_encode(table: &[f32; 127], x: f32) -> u8 {
    let a = x.abs().min(E4M3_MAX);
    let hi = table.partition_point(|&t| t < a).min(126);
    let lo = hi.saturating_sub(1);
    let nearest = if (a - table[lo]).abs() <= (table[hi] - a).abs() {
        lo
    } else {
        hi
    };
    let code = nearest as u8;
    code | if x < 0.0 { 0x80 } else { 0 }
}

/// W `[n, k]` BF16 bits to E4M3 bytes `[n, k]` and FP32 block scales `[ceil(n/128), k/128]`
/// (scale = block amax / 448; dequantized value = code value * scale).
fn quantize_weight(w: &[u16], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let mut table = [0f32; 127];
    for (c, t) in table.iter_mut().enumerate() {
        *t = e4m3_value(c as u8);
    }
    let (nb, kb) = (n.div_ceil(128), k / 128);
    let mut q = vec![0u8; n * k];
    let mut scales = Vec::with_capacity(nb * kb * 4);
    for bn in 0..nb {
        let rows = bn * 128..((bn + 1) * 128).min(n);
        for bk in 0..kb {
            let cols = bk * 128..(bk + 1) * 128;
            let amax = rows
                .clone()
                .flat_map(|r| w[r * k + cols.start..r * k + cols.end].iter())
                .map(|&b| bf16::from_bits(b).to_f32().abs())
                .fold(0f32, f32::max);
            let scale = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
            scales.extend_from_slice(&scale.to_le_bytes());
            for r in rows.clone() {
                for c in cols.clone() {
                    q[r * k + c] =
                        e4m3_encode(&table, bf16::from_bits(w[r * k + c]).to_f32() / scale);
                }
            }
        }
    }
    (q, scales)
}

fn bf16_values(bytes: &[u8]) -> impl Iterator<Item = f64> + '_ {
    bytes
        .chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32() as f64)
}

fn cosine(a: &[u8], b: &[u8]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in bf16_values(a).zip(bf16_values(b)) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

/// (mean, min) milliseconds over `REPS` launches, each followed by a stream synchronize,
/// after `WARMUP` untimed ones.
fn time_arm(gpu: &dyn GpuBackend, f: &dyn Fn() -> Result<()>) -> Result<(f64, f64)> {
    for _ in 0..WARMUP {
        f()?;
    }
    gpu.synchronize(0)?;
    let (mut sum, mut min) = (0f64, f64::MAX);
    for _ in 0..REPS {
        let t = Instant::now();
        f()?;
        gpu.synchronize(0)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        sum += ms;
        min = min.min(ms);
    }
    Ok((sum / REPS as f64, min))
}

fn report(arm: &str, (m, n, k): (usize, usize, usize), (mean, min): (f64, f64), gemm: bool) {
    let tflops = |ms: f64| 2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12;
    if gemm {
        println!(
            "TIMING m={m} n={n} k={k} arm={arm} ms_mean={mean:.4} ms_min={min:.4} tflops={:.2}",
            tflops(mean)
        );
    } else {
        println!("TIMING m={m} n={n} k={k} arm={arm} ms_mean={mean:.4} ms_min={min:.4} tflops=na");
    }
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    ensure!(
        gpu.has_module("fp8_gemm_blockscaled_pipe"),
        "this build lacks the pipelined module; the fp8_gemm arm would time the legacy kernel"
    );
    ensure!(
        std::env::var_os("METRALE_NO_FP8_GEMM_PIPE").is_none(),
        "unset METRALE_NO_FP8_GEMM_PIPE: ops would launch the legacy kernel"
    );
    let legacy = gpu.kernel("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled")?;
    let quant = ops::Fp8ActQuant::resolve(&gpu);
    ensure!(
        quant.available(),
        "`per_token_group_quant_fp8` is not in this image"
    );
    let m_max = *MS.iter().max().unwrap();
    let mut rng = Rng(0x474c_4d35_3320_1004);
    let mut failures: Vec<String> = Vec::new();
    let mut arms_run = 0usize;

    for (n, k, name) in SHAPES {
        if k % 128 != 0 || n % 64 != 0 {
            println!(
                "SKIP m=* n={n} k={k} name={name} reason=FP8 kernel needs K%128==0 and N%64==0"
            );
            continue;
        }
        eprintln!("{name}: generating operands (n={n} k={k}, A rows {m_max})");
        let a_bits = rng.bf16_bits(m_max * k, 1.0);
        let w_bits = rng.bf16_bits(n * k, 0.1);
        let (w_q, w_s) = quantize_weight(&w_bits, n, k);
        let a = upload(&gpu, &le_bytes(&a_bits))?;
        drop(a_bits);
        let w = upload(&gpu, &le_bytes(&w_bits))?;
        drop(w_bits);
        let wq = upload(&gpu, &w_q)?;
        let ws = upload(&gpu, &w_s)?;
        drop((w_q, w_s));
        let a_q = gpu.alloc(m_max * k)?;
        let a_s = gpu.alloc(m_max * (k / 128) * 4)?;
        let out_bytes = m_max * n * 2;
        let out_bf16 = gpu.alloc(out_bytes)?;
        let out_fp8 = gpu.alloc(out_bytes)?;

        for m in MS {
            let id = (m, n, k);
            let (m32, n32, k32) = (m as u32, n as u32, k as u32);
            gpu.memset(out_bf16, 0, out_bytes)?;
            gpu.memset(out_fp8, 0, out_bytes)?;
            let run_bf16 = || ops::cublas_bf16_proj_dense(a, w, out_bf16, m32, n32, k32, 0);
            let run_quant =
                || ops::per_token_group_quant_fp8(&gpu, quant, a, a_q, a_s, m32, k32, 0);
            let run_gemm = || {
                ops::fp8_gemm_t_blockscaled(
                    &gpu, legacy, a_q, a_s, wq, ws, out_fp8, m32, n32, k32, 0,
                )
            };
            let run_total = || -> Result<()> {
                run_quant()?;
                run_gemm()
            };
            // The gemm arm reads a_q / a_s: fill them before timing it.
            run_quant()?;
            gpu.synchronize(0)?;
            let t_bf16 = time_arm(&gpu, &run_bf16)?;
            let t_quant = time_arm(&gpu, &run_quant)?;
            let t_gemm = time_arm(&gpu, &run_gemm)?;
            let t_total = time_arm(&gpu, &run_total)?;
            arms_run += 4;
            report("bf16_cublaslt", id, t_bf16, true);
            report("fp8_quant", id, t_quant, false);
            report("fp8_gemm", id, t_gemm, true);
            report("fp8_total", id, t_total, true);
            println!(
                "RATIO m={m} n={n} k={k} fp8_total_over_bf16={:.3}",
                t_total.0 / t_bf16.0
            );

            // Wiring check on the outputs of the last timed launch of each leg.
            let live = m * n * 2;
            let (mut h_bf16, mut h_fp8) = (vec![0u8; live], vec![0u8; live]);
            gpu.synchronize(0)?;
            gpu.copy_d2h(out_bf16, &mut h_bf16)?;
            gpu.copy_d2h(out_fp8, &mut h_fp8)?;
            let cos = cosine(&h_fp8, &h_bf16);
            println!("COS m={m} n={n} k={k} cos={cos:.6}");
            if cos.is_nan() || cos <= COS_GATE {
                failures.push(format!("m={m} n={n} k={k}: cos {cos:.6} <= {COS_GATE}"));
            }
        }
        for p in [a, w, wq, ws, a_q, a_s, out_bf16, out_fp8] {
            gpu.free(p)?;
        }
    }

    if arms_run == 0 {
        println!("FAIL: every shape was skipped");
        bail!("no shape ran");
    }
    if !failures.is_empty() {
        println!("FAIL: {}", failures.join("; "));
        bail!("cosine wiring check failed");
    }
    println!("PASS: timings recorded");
    Ok(())
}
