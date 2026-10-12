// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate for `METRALE_GLM_IDX_WQB_NV4` (`glm5next_dsa/layer/idx_wqb_nv4.rs`): the DSA
//! indexer `wq_b` (4096 x 1536: `index_heads` 32 x `index_head_dim` 128 by `q_lora_rank` 1536,
//! `glm5next_dsa/binding.rs:73`) on an NVFP4 copy with FP32 output
//! (`kernels/gb10/common/w4a16_gemv_f32out.cu`) against the BF16 FP32-out GEMV it replaces.
//!
//! Owner: model-arch examples.
//! Checks (every one prints `PASS ` or `FAIL `; the last line is `PASS all` iff all passed):
//! - (a) Exactness, bitwise (u16 patterns), M = 1..=16: BF16(w4a16_gemv_f32out_b*) equals the BF16
//!   output of the production CUDA-core NVFP4 tier (`w4a16_gemv`, `_batch2`, `_batch3`,
//!   `_batch4..8`, `_batch16`); BF16(w4a16_gemv_tc{8,16}_f32out) equals `w4a16_gemv_tc{8,16}`.
//! - (b) Tolerance against an FP64 host reference on the DEQUANTIZED weights (E2M1 x E4M3 scale x
//!   tensor scale read back from the copy), M = 1,2,3,4,5,7,8,9,16, both new kernels: cosine >=
//!   0.99999, max_rel (max |new - ref| / max |ref| over the output) <= 1e-4, nonfinite 0.
//! - (c) Speed: cold pool (BF16: 4 rotated weight copies = 50 MB; NVFP4: 8 = 28 MB, both above the
//!   24 MB L2), a CUDA graph of 16 launches replayed 25 times, median per launch, at M = 1, 2, 3,
//!   4, 5, 7, 8, 9, 16 for: `old` = the BF16 FP32-out GEMV (`dense_gemv_bf16_fp32out`, M >= 2
//!   `dense_gemv_bf16_fp32out_batchm`), `core` = CUDA-core NVFP4, `tc` = tensor-core NVFP4.
//!   `new` is the faster of core and tc. GB/s = weight (+ scale) bytes / time; > 255 prints
//!   `VOID`. `ROUTE` lines say which kernel won and whether the PROVISIONAL map in
//!   `idx_wqb_nv4.rs` (`BF16_MAX_M`, `CORE_MAX_M`) agrees.
//!
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm_idx_wqb_nv4_microtest

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_arch::glm5next_dsa::layer::{BF16_MAX_M, CORE_MAX_M, WqbRoute, wqb_route};
use metrale_model_layers::layers::ops::{self, DenseMmKernels};
use metrale_model_layers::weight_map::{DenseWeight, QuantizedWeight};

const N: usize = 4096;
const K: usize = 1536;
const MS: [usize; 9] = [1, 2, 3, 4, 5, 7, 8, 9, 16];
const BF16_COPIES: usize = 4;
const NV4_COPIES: usize = 8;
const LAUNCHES: usize = 16;
const REPLAYS: usize = 25;

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

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn bf_bytes(v: &[bf16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn e4m3(b: u8) -> f64 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = ((b >> 3) & 0xF, (b & 7) as f64);
    if e == 0 {
        s * m * 2f64.powi(-9)
    } else {
        s * 2f64.powi(e as i32 - 7) * (1.0 + m / 8.0)
    }
}

const LUT: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

struct Kern {
    absmax: KernelHandle,
    quant: KernelHandle,
    /// Production BF16-out CUDA-core tiers: gemv, b2, b3, b4..=b8, b16.
    prod: Vec<KernelHandle>,
    prod_tc8: KernelHandle,
    prod_tc16: KernelHandle,
    core: Vec<KernelHandle>,
    tc8: KernelHandle,
    tc16: KernelHandle,
    gemm_f32: KernelHandle,
    gemv_f32: KernelHandle,
    batchm_f32: KernelHandle,
}

fn resolve(g: &dyn GpuBackend) -> Result<Kern> {
    let k = |m: &str, n: &str| g.kernel(m, n);
    let mut prod = vec![k("w4a16_gemv", "w4a16_gemv")?];
    for m in [2, 3, 4, 5, 6, 7, 8, 16] {
        prod.push(k("w4a16_gemv", &format!("w4a16_gemv_batch{m}"))?);
    }
    let mut core = Vec::new();
    for m in [1, 2, 3, 4, 5, 6, 7, 8, 16] {
        core.push(k("w4a16_gemv_f32out", &format!("w4a16_gemv_f32out_b{m}"))?);
    }
    Ok(Kern {
        absmax: k("quantize_nvfp4", "nvfp4_global_absmax")?,
        quant: k("quantize_nvfp4", "quantize_bf16_to_nvfp4_mse")?,
        prod,
        prod_tc8: k("w4a16_gemv_tc", "w4a16_gemv_tc8")?,
        prod_tc16: k("w4a16_gemv_tc", "w4a16_gemv_tc16")?,
        core,
        tc8: k("w4a16_gemv_f32out", "w4a16_gemv_tc8_f32out")?,
        tc16: k("w4a16_gemv_f32out", "w4a16_gemv_tc16_f32out")?,
        gemm_f32: k("gemm", "dense_gemm_bf16_f32out")?,
        gemv_f32: k("gemv", "dense_gemv_bf16_fp32out")?,
        batchm_f32: k("dense_gemv_bf16_batchm", "dense_gemv_bf16_fp32out_batchm")?,
    })
}

/// Index into `core` / `prod` (b1/gemv, b2, b3, b4..b8, b16) for `m` rows.
fn tier(m: usize) -> usize {
    if m <= 8 { m - 1 } else { 8 }
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    h: KernelHandle,
    cols: u32,
    takes_m: bool,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    s: u64,
) -> Result<()> {
    let mut l = KernelLaunch::new(g, h)
        .grid([div_ceil(N as u32, cols), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(q.weight)
        .arg_ptr(q.weight_scale)
        .arg_f32(q.weight_scale_2)
        .arg_ptr(c);
    if takes_m {
        l = l.arg_u32(m as u32);
    }
    l.arg_u32(N as u32).arg_u32(K as u32).launch(s)
}

fn new_core(
    g: &dyn GpuBackend,
    k: &Kern,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    s: u64,
) -> Result<()> {
    launch(g, k.core[tier(m)], 4, true, a, q, c, m, s)
}

fn new_tc(
    g: &dyn GpuBackend,
    k: &Kern,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    s: u64,
) -> Result<()> {
    if m <= 8 {
        launch(g, k.tc8, 8, true, a, q, c, m, s)
    } else {
        launch(g, k.tc16, 16, true, a, q, c, m, s)
    }
}

/// The production BF16-out CUDA-core NVFP4 tier (`takes_m` for the 4..=16 tiers only).
fn prod_core(
    g: &dyn GpuBackend,
    k: &Kern,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    s: u64,
) -> Result<()> {
    launch(g, k.prod[tier(m)], 4, m >= 4, a, q, c, m, s)
}

fn prod_tc(
    g: &dyn GpuBackend,
    k: &Kern,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    s: u64,
) -> Result<()> {
    if m <= 8 {
        launch(g, k.prod_tc8, 8, true, a, q, c, m, s)
    } else {
        launch(g, k.prod_tc16, 16, true, a, q, c, m, s)
    }
}

/// The BF16 FP32-out path `wq_b` runs today: `dense_mm_bf16` at M = 1, the FP32-out batched GEMV
/// above (`batchm_rows`).
fn old_bf16(
    g: &dyn GpuBackend,
    k: &Kern,
    a: DevicePtr,
    w: DevicePtr,
    c: DevicePtr,
    m: usize,
    s: u64,
) -> Result<()> {
    if m == 1 {
        ops::dense_mm_bf16(
            g,
            &DenseMmKernels {
                gemm: k.gemm_f32,
                gemv: k.gemv_f32,
                batchm: KernelHandle(0),
            },
            a,
            w,
            c,
            1,
            N,
            K,
            s,
        )
    } else {
        ops::dense_gemv_batchm_fp32out(
            g,
            k.batchm_f32,
            a,
            &DenseWeight { weight: w },
            c,
            m as u32,
            N as u32,
            K as u32,
            N as u32,
            s,
        )
    }
}

fn quantize(g: &dyn GpuBackend, k: &Kern, w: DevicePtr) -> Result<QuantizedWeight> {
    let s = g.default_stream();
    let q = metrale_model_layers::weight_map::quantize_to_nvfp4(
        &DenseWeight { weight: w },
        N,
        K,
        g,
        k.absmax,
        k.quant,
        s,
    )?;
    g.synchronize(s)?;
    Ok(q)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median per-launch microseconds of `LAUNCHES` launches captured in one graph, `REPLAYS` replays.
fn time_graph(
    g: &dyn GpuBackend,
    s: u64,
    f: &mut dyn FnMut(usize, u64) -> Result<()>,
) -> Result<f64> {
    g.begin_capture(s)?;
    for i in 0..LAUNCHES {
        if let Err(e) = f(i, s) {
            g.abort_capture_if_active(s);
            return Err(e);
        }
    }
    let graph = g.end_capture(s)?;
    for _ in 0..3 {
        g.launch_graph(graph, s)?;
    }
    g.synchronize(s)?;
    let mut t = Vec::new();
    for _ in 0..REPLAYS {
        let t0 = std::time::Instant::now();
        g.launch_graph(graph, s)?;
        g.synchronize(s)?;
        t.push(t0.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64);
    }
    g.destroy_graph(graph)?;
    Ok(median(&mut t))
}

struct Report {
    fails: usize,
}

impl Report {
    fn check(&mut self, ok: bool, msg: String) {
        if ok {
            println!("PASS {msg}");
        } else {
            self.fails += 1;
            println!("FAIL {msg}");
        }
    }
}

fn run() -> Result<i32> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k = match resolve(g) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("kernels missing from this target: {e:#}");
            return Ok(2);
        }
    };
    let s = g.default_stream();
    let mut rng = Lcg(0x0077_7162_776e_7634);
    let mut rep = Report { fails: 0 };

    // Weights: N(0, 0.02) BF16 [N, K]; activations N(0, 1) BF16 [16, K] (RMS-normed q_resid scale).
    let w: Vec<bf16> = (0..N * K).map(|_| bf16::from_f64(rng.n() * 0.02)).collect();
    let a_host: Vec<bf16> = (0..16 * K).map(|_| bf16::from_f64(rng.n())).collect();
    let w_dev = up(g, &bf_bytes(&w))?;
    let a_dev = up(g, &bf_bytes(&a_host))?;
    let q = quantize(g, &k, w_dev)?;
    let out32 = g.alloc(16 * N * 4)?;
    let out16 = g.alloc(16 * N * 2)?;

    // FP64 reference on the dequantized weights (read back from the copy).
    let packed = down(g, q.weight, N * K / 2)?;
    let scales = down(g, q.weight_scale, N * K / 16)?;
    let s2 = q.weight_scale_2 as f64;
    let mut dq = vec![0f64; N * K];
    for n in 0..N {
        for j in 0..K {
            let byte = packed[n * (K / 2) + j / 2];
            let nib = if j % 2 == 0 { byte & 0xF } else { byte >> 4 };
            dq[n * K + j] = LUT[nib as usize] * e4m3(scales[n * (K / 16) + j / 16]) * s2;
        }
    }
    let af: Vec<f64> = a_host.iter().map(|x| x.to_f64()).collect();
    let mut refv = vec![0f64; 16 * N];
    for t in 0..16 {
        for n in 0..N {
            let (r, a) = (&dq[n * K..(n + 1) * K], &af[t * K..(t + 1) * K]);
            refv[t * N + n] = r.iter().zip(a).map(|(x, y)| x * y).sum();
        }
    }

    // (a) Exactness, bitwise, M = 1..=16.
    for m in 1..=16usize {
        let mut cmp = |label: &str,
                       new: &dyn Fn() -> Result<()>,
                       old: &dyn Fn() -> Result<()>|
         -> Result<()> {
            g.copy_h2d(&vec![0xA5u8; 16 * N * 4], out32)?;
            g.copy_h2d(&vec![0xA5u8; 16 * N * 2], out16)?;
            new()?;
            old()?;
            g.synchronize(s)?;
            let f = down(g, out32, m * N * 4)?;
            let b = down(g, out16, m * N * 2)?;
            let mut diff = 0usize;
            for (i, c) in f.chunks_exact(4).enumerate() {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let r = bf16::from_f32(v).to_bits();
                let o = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
                if r != o {
                    diff += 1;
                }
            }
            // One row past M must be untouched.
            let past = down(g, out32.offset(m * N * 4), 4)?;
            let clean = m == 16 || past == [0xA5; 4];
            rep.check(diff == 0 && clean, format!("exact {label} M={m}: BF16(f32out) == BF16-out kernel bitwise ({diff} of {} differ), row M untouched {clean}", m * N));
            Ok(())
        };
        cmp("core", &|| new_core(g, &k, a_dev, &q, out32, m, s), &|| {
            prod_core(g, &k, a_dev, &q, out16, m, s)
        })?;
        cmp("tc  ", &|| new_tc(g, &k, a_dev, &q, out32, m, s), &|| {
            prod_tc(g, &k, a_dev, &q, out16, m, s)
        })?;
    }

    // (b) Tolerance against the FP64 reference.
    for &m in &MS {
        for (label, tc) in [("core", false), ("tc  ", true)] {
            g.copy_h2d(&vec![0u8; 16 * N * 4], out32)?;
            if tc {
                new_tc(g, &k, a_dev, &q, out32, m, s)?;
            } else {
                new_core(g, &k, a_dev, &q, out32, m, s)?;
            }
            g.synchronize(s)?;
            let f = down(g, out32, m * N * 4)?;
            let (mut dot, mut nn, mut rr, mut maxd, mut maxr, mut nonfinite) =
                (0f64, 0f64, 0f64, 0f64, 0f64, 0usize);
            for (i, c) in f.chunks_exact(4).enumerate() {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                if !v.is_finite() {
                    nonfinite += 1;
                    continue;
                }
                let (v, r) = (v as f64, refv[i]);
                dot += v * r;
                nn += v * v;
                rr += r * r;
                maxd = maxd.max((v - r).abs());
                maxr = maxr.max(r.abs());
            }
            let cos = dot / (nn.sqrt() * rr.sqrt()).max(1e-300);
            let max_rel = maxd / maxr.max(1e-300);
            rep.check(
                cos >= 0.99999 && max_rel <= 1e-4 && nonfinite == 0,
                format!("tol {label} M={m}: cosine={cos:.9} max_rel(peak-normalized)={max_rel:.3e} nonfinite={nonfinite}"),
            );
        }
    }

    // (c) Cold-pool timing.
    let w_copies: Vec<DevicePtr> = (0..BF16_COPIES)
        .map(|_| up(g, &bf_bytes(&w)))
        .collect::<Result<_>>()?;
    let mut q_copies = Vec::new();
    for _ in 0..NV4_COPIES {
        let t = up(g, &bf_bytes(&w))?;
        q_copies.push(quantize(g, &k, t)?);
        g.free(t)?;
    }
    let bytes_bf16 = (N * K * 2) as f64;
    let bytes_nv4 = (N * K / 2 + N * K / 16) as f64;
    let gbs = |bytes: f64, us: f64| bytes / (us * 1e-6) / 1e9;
    let fmt = |v: f64| {
        if v > 255.0 {
            format!("{v:.0}VOID")
        } else {
            format!("{v:.1}")
        }
    };
    for &m in &MS {
        let old = time_graph(g, s, &mut |i, st| {
            old_bf16(g, &k, a_dev, w_copies[i % BF16_COPIES], out32, m, st)
        })?;
        let core = time_graph(g, s, &mut |i, st| {
            new_core(g, &k, a_dev, &q_copies[i % NV4_COPIES], out32, m, st)
        })?;
        let tc = time_graph(g, s, &mut |i, st| {
            new_tc(g, &k, a_dev, &q_copies[i % NV4_COPIES], out32, m, st)
        })?;
        let (new_us, winner) = if core <= tc {
            (core, "core")
        } else {
            (tc, "tc")
        };
        println!(
            "M={m} old_us={old:.2} new_us={new_us:.2} old_GBs={} new_GBs={} (core_us={core:.2} core_GBs={} tc_us={tc:.2} tc_GBs={})",
            fmt(gbs(bytes_bf16, old)),
            fmt(gbs(bytes_nv4, new_us)),
            fmt(gbs(bytes_nv4, core)),
            fmt(gbs(bytes_nv4, tc)),
        );
        let best = if old <= new_us { "bf16" } else { winner };
        let prov = match wqb_route(m, K) {
            WqbRoute::Bf16 => "bf16",
            WqbRoute::Core => "core",
            WqbRoute::Tc => "tc",
        };
        println!(
            "ROUTE M={m} fastest={best} provisional_map={prov} (BF16_MAX_M={BF16_MAX_M} CORE_MAX_M={CORE_MAX_M}) {}",
            if best == prov {
                "AGREE"
            } else {
                "DIFFERS: retune the constants"
            }
        );
    }

    if rep.fails > 0 {
        println!("FAIL {} checks failed", rep.fails);
        return Ok(1);
    }
    println!("PASS all");
    Ok(0)
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
