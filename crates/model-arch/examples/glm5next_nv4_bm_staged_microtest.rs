// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Gate for `METRALE_GLM_NV4_BATCHM_STAGED`: `w4a16_gemv_batch{4..8,16}_staged`
//! against `w4a16_gemv_batch{4..8,16}` (the 4..=16-row NVFP4 GEMVs of GLM-5.3 C>1 decode and
//! multi-row verify).
//!
//! Owner: model-arch examples.
//! Checks, per per-rank decode shape:
//! - (a) Raw u16 output bits of staged vs unstaged tier are equal on the same inputs at
//!   M = 4, 5, 6, 7, 8, 9, 12, 16 (9..16 on the 16 tier), for three activation sets (random
//!   N(0,1); large magnitude; row 1 all zeros) and two weight sets (Gaussian, heavy-tailed).
//!   Shapes include N % 4 = 1 (partial last block), K = 16384 (many K windows) and
//!   K16 % 64 != 0 (partial last window, odd window count).
//! - (b) Timing at M = 4, 8, 12, 16: CUDA-graph replay over a cold pool of distinct NVFP4
//!   weights (> 256 MB each shape), `TIMING <shape> M=<m> base .. staged .. speedup`.
//! - (c) Per-step weighted sums at M = 4 (C4 serial step) and M = 12 (C4 2-draft verify).
//!
//! Prints `PASS: ...` iff every compare is byte-equal, else `FAIL ...` (exit 1).
//!
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_nv4_bm_staged_microtest

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::weight_map::{DenseWeight, QuantizedWeight};

/// (N, K, label, launches per decode step per rank).
const SHAPES: &[(usize, usize, &str, usize)] = &[
    (4096, 4096, "kda_qkvo", 136),
    (4096, 1024, "shexp_down", 42),
    (4096, 6144, "dense_down", 3),
    (4096, 16384, "dsa_o_absorb", 11),
    (1024, 4096, "shexp_gate_up", 84),
    (16384, 1536, "dsa_q_absorb", 11),
    (6144, 4096, "dense_gate_up", 6),
    (1536, 4096, "dsa_q_a", 11),
    (512, 4096, "dsa_kv_a", 11),
    // Edge shapes (no per-step weight): N % 4 = 1; K16 = 129 (one chunk in the last window).
    (4097, 1024, "N%4=1", 0),
    (4098, 2064, "K16=129", 0),
    (4096, 1040, "K16=65", 0),
];
const POOL_NV4_BYTES: usize = 256 << 20;
/// Rows compared (a) and timed (b).
const CMP_M: &[usize] = &[4, 5, 6, 7, 8, 9, 12, 16];
const TIME_M: &[usize] = &[4, 8, 12, 16];
/// Gate (pre-registered 2026-10-07): staged >= 1.15x at M = 8 on kda_qkvo, else the lever is killed.
const GATE_M8_KDA: f64 = 1.15;

/// Tier index (`dense_fp8` route) for `m` rows.
fn tier(m: usize) -> usize {
    if m <= 8 { m - 4 } else { 5 }
}

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

/// Activation sets: 0 = N(0,1); 1 = large magnitude (x 3000 with 0.1 % x 30 outliers);
/// 2 = N(0,1) with row 1 all zeros.
fn gen_act(rng: &mut Lcg, m: usize, k: usize, set: usize) -> Vec<bf16> {
    let mut a = Vec::with_capacity(m * k);
    for t in 0..m {
        for _ in 0..k {
            let mut v = rng.n();
            if set == 1 {
                v *= 3000.0;
                if rng.u() < 0.001 {
                    v *= 30.0;
                }
            }
            if set == 2 && t == 1 {
                v = 0.0;
            }
            a.push(bf16::from_f64(v));
        }
    }
    a
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

#[allow(clippy::too_many_arguments)]
fn launchm(
    g: &dyn GpuBackend,
    h: KernelHandle,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, h)
        .grid([div_ceil(n as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(q.weight)
        .arg_ptr(q.weight_scale)
        .arg_f32(q.weight_scale_2)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(s)
}

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

struct Kq {
    absmax: KernelHandle,
    quant: KernelHandle,
    base: [KernelHandle; 6],
    staged: [KernelHandle; 6],
}

fn quantize(
    g: &dyn GpuBackend,
    k: &Kq,
    w: &[bf16],
    n: usize,
    kd: usize,
) -> Result<QuantizedWeight> {
    let host: Vec<u8> = w.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let tmp = g.alloc(host.len())?;
    g.copy_h2d(&host, tmp)?;
    let q = metrale_model_layers::weight_map::quantize_to_nvfp4(
        &DenseWeight { weight: tmp },
        n,
        kd,
        g,
        k.absmax,
        k.quant,
        g.default_stream(),
    )?;
    g.synchronize(g.default_stream())?;
    g.free(tmp)?;
    Ok(q)
}

/// (a) Returns the number of failing compares.
fn compare(
    g: &dyn GpuBackend,
    k: &Kq,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
) -> Result<usize> {
    let s = g.default_stream();
    let mut fails = 0;
    for heavy in [false, true] {
        let w = gen_weight(rng, n, kd, heavy);
        let q = quantize(g, k, &w, n, kd)?;
        for (set, &m) in (0..3).flat_map(|s| CMP_M.iter().map(move |m| (s, m))) {
            let ad = up_bf16(g, &gen_act(rng, m, kd, set))?;
            let (cb, cs) = (g.alloc(m * n * 2)?, g.alloc(m * n * 2)?);
            // Poison the outputs so an unwritten element cannot compare equal by accident.
            let poison = vec![0xA5u8; m * n * 2];
            g.copy_h2d(&poison, cb)?;
            g.copy_h2d(&poison, cs)?;
            launchm(g, k.base[tier(m)], ad, &q, cb, m, n, kd, s)?;
            launchm(g, k.staged[tier(m)], ad, &q, cs, m, n, kd, s)?;
            g.synchronize(s)?;
            let (yb, ys) = (dn_bytes(g, cb, m * n * 2)?, dn_bytes(g, cs, m * n * 2)?);
            let diff = yb.iter().zip(&ys).filter(|(a, b)| a != b).count();
            let ok = diff == 0;
            if !ok {
                fails += 1;
            }
            println!(
                "COMPARE {label:<14} {}N={n} K={kd} M={m} acts={} differing bytes {diff}  {}",
                if heavy { "heavy " } else { "gauss " },
                ["random", "large", "zero-row"][set],
                if ok { "byte-equal" } else { "MISMATCH" }
            );
            for p in [ad, cb, cs] {
                g.free(p)?;
            }
        }
        g.free(q.weight)?;
        g.free(q.weight_scale)?;
    }
    Ok(fails)
}

/// (b) (base ms, staged ms) per launch over a cold pool.
fn timing(
    g: &dyn GpuBackend,
    k: &Kq,
    rng: &mut Lcg,
    n: usize,
    kd: usize,
    label: &str,
    m: usize,
) -> Result<(f64, f64)> {
    let s = g.create_stream()?;
    let nv_bytes = n * kd / 2 + n * kd / 16;
    let pool = POOL_NV4_BYTES.div_ceil(nv_bytes).clamp(4, 1024);
    let w = gen_weight(rng, n, kd, false);
    let mut qs = Vec::with_capacity(pool);
    for _ in 0..pool {
        qs.push(quantize(g, k, &w, n, kd)?);
    }
    let ad = up_bf16(g, &gen_act(rng, m, kd, 0))?;
    let c = g.alloc(m * n * 2)?;
    let tb = time_graph(g, s, &mut |s| {
        for q in &qs {
            launchm(g, k.base[tier(m)], ad, q, c, m, n, kd, s)?;
        }
        Ok(())
    })? / pool as f64;
    let ts = time_graph(g, s, &mut |s| {
        for q in &qs {
            launchm(g, k.staged[tier(m)], ad, q, c, m, n, kd, s)?;
        }
        Ok(())
    })? / pool as f64;
    let gbs = |ms: f64| nv_bytes as f64 / (ms * 1e-3) / 1e9;
    println!(
        "TIMING {label:<14} {n}x{kd} M={m} base {:>8.1} us {:>6.1} GB/s staged {:>8.1} us {:>6.1} GB/s speedup {:>5.2}x  (pool {pool})",
        tb * 1e3,
        gbs(tb),
        ts * 1e3,
        gbs(ts),
        tb / ts
    );
    for q in qs {
        g.free(q.weight)?;
        g.free(q.weight_scale)?;
    }
    g.free(ad)?;
    g.free(c)?;
    Ok((tb, ts))
}

fn tiers(g: &dyn GpuBackend, suffix: &str) -> Result<[KernelHandle; 6]> {
    let mut v = Vec::new();
    for m in [4, 5, 6, 7, 8, 16] {
        v.push(g.kernel("w4a16_gemv", &format!("w4a16_gemv_batch{m}{suffix}"))?);
    }
    v.try_into().map_err(|_| anyhow::anyhow!("six tiers"))
}

fn run() -> Result<bool> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k = Kq {
        absmax: g.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quant: g.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_mse")?,
        base: tiers(g, "")?,
        staged: tiers(g, "_staged")?,
    };
    let mut rng = Lcg(0x6e76_3462_6d73);
    let mut fails = 0usize;
    for &(n, kd, label, _) in SHAPES {
        fails += compare(g, &k, &mut rng, n, kd, label)?;
    }
    let mut gate = 0.0f64;
    for &m in TIME_M {
        let (mut base_sum, mut staged_sum) = (0.0f64, 0.0f64);
        for &(n, kd, label, per_step) in SHAPES {
            let (tb, ts) = timing(g, &k, &mut rng, n, kd, label, m)?;
            base_sum += tb * per_step as f64;
            staged_sum += ts * per_step as f64;
            if m == 8 && label == "kda_qkvo" {
                gate = tb / ts;
            }
        }
        println!(
            "TIMING SUMMARY M={m} per step (launches per rank per step weighted, nine production shapes): \
             base {base_sum:.2} ms -> staged {staged_sum:.2} ms, saves {:.2} ms ({:.2}x)",
            base_sum - staged_sum,
            base_sum / staged_sum.max(1e-12)
        );
    }
    let speed_ok = gate >= GATE_M8_KDA;
    println!(
        "GATE kda_qkvo M=8 staged speedup {gate:.2}x (>= {GATE_M8_KDA}: {})",
        if speed_ok { "GO" } else { "KILL" }
    );
    if fails == 0 {
        println!(
            "PASS: w4a16_gemv_batch{{4..8,16}}_staged byte-equal to the unstaged tiers on {} shapes x 2 weight sets x 3 activation sets x M {:?}",
            SHAPES.len(),
            CMP_M
        );
        Ok(true)
    } else {
        println!("FAIL {fails} compares differ (see MISMATCH lines)");
        Ok(false)
    }
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            println!("FAIL error: {e:#}");
            std::process::exit(1);
        }
    }
}
