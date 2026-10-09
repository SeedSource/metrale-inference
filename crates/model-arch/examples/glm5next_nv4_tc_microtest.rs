// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Gate for `METRALE_GLM_NV4_TC`: `w4a16_gemv_tc8` / `w4a16_gemv_tc16`
//! (`kernels/gb10/common/w4a16_gemv_tc.cu`) against the CUDA-core NVFP4 GEMV tiers of
//! `dense_fp8::nv4_gemv_uncounted` (`w4a16_gemv`, `_batch2`, `_batch3_staged`, `_batch{4..8,16}`).
//!
//! Owner: model-arch examples.
//! Checks, per per-rank decode shape (shapes with `K % 128 != 0` are skipped: the route declines):
//! - (a) Row invariance, bitwise, for 2 weight sets x 3 activation sets: for M = 1..=16 the
//!   lever's kernel (tc8 for M <= 8, tc16 above) on M rows gives, for every row t, the bytes of
//!   the M = 1 tc8 launch on that row alone; for M <= 8 tc16 is byte-equal to tc8; output row M
//!   (one past) is never written.
//! - (b) Accuracy against a CPU f64 reference (64 sampled columns plus first and last, every
//!   row) at M = 1, 3, 8, 12, 16: tc max abs error <= max(1.25 x the CUDA-core tier's, half a
//!   BF16 ulp of the output range).
//! - (c) Timing at M = 1, 2, 3, 4, 8, 12, 16: CUDA-graph replay over a cold pool of distinct
//!   NVFP4 weights, tc against the production CUDA-core route (M = 3: `batch3_staged`, plus a
//!   `b3plain` line), per-step weighted sums.
//! - (d) `GATE` lines: invariance, accuracy, speed (GO: M=8 saves >= 3.0 ms and M=3 delta
//!   <= +0.3 ms; KILL: M=8 saves < 1.5 ms).
//! - (e) 2026-10-09, `METRALE_GLM_NV4_TC_GROUP`: the grouped persistent entries
//!   (`w4a16_gemv_tc8_group` / `_tc16_group`). Bytes: for every `GROUPS` entry, every
//!   production shape above as a one-member group and an edge group, at M = 1..=16, at the
//!   production grid (`gemv_tc::tc_group_launch`: min(tiles, SMs x resident CTAs)) and at a
//!   7-CTA grid, every output byte of every member (rows < M) equals the plain tc8/tc16 launch
//!   and rows >= M stay unwritten; a mismatch is an invariance failure (GATE invariance FAIL).
//!   Timing: per group at M = 2, 3, 4, 8, the plain per-projection tc launches against one
//!   grouped launch over a cold pool, weighted by groups per step:
//!   `TIMING SUMMARY GROUP M=.. per step: tc .. ms -> group .. ms, saves .. ms`.
//!
//! Prints `PASS: ...` and exits 0 iff invariance and accuracy pass, else `FAIL ...` (exit 1);
//! exit 2 when the kernels are missing. Speed only prints its verdict.
//!
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_nv4_tc_microtest

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops::gemv_tc::{self, TcGroupMember};
use metrale_model_layers::weight_map::{DenseWeight, QuantizedWeight, quantize_to_nvfp4};

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
/// 2026-10-09: (label, N of each member, K, groups per decode step per rank), from `SHAPES`:
/// `kda_qkvo` 136 = 34 KDA layers x q/k/v/o (o reads the attention output, so q/k/v group);
/// `shexp_gate_up` 84 = 42 x gate/up; `dense_gate_up` 6 = 3 x gate/up; `dsa_q_a` and
/// `dsa_kv_a` 11 each = 11 DSA layers x one q_a/kv_a pair.
const GROUPS: &[(&str, &[usize], usize, usize)] = &[
    ("kda_qkv", &[4096, 4096, 4096], 4096, 34),
    ("shexp_gate_up", &[1024, 1024], 4096, 42),
    ("dense_gate_up", &[6144, 6144], 4096, 3),
    ("dsa_q_a_kv_a", &[1536, 512], 4096, 11),
    // Edge group (no per-step weight): a partial last tile, an N below one tile, and N % 4 = 1.
    ("edge", &[4097, 5, 1027], 1024, 0),
];
/// 2026-10-09: Rows timed for the grouped launch; the forced small grid of the byte check.
const GROUP_TIME_M: &[usize] = &[2, 3, 4, 8];
const GROUP_SMALL_GRID: u32 = 7;
const POOL_NV4_BYTES: usize = 256 << 20;
/// Rows timed (c) and checked against the CPU reference (b).
const TIME_M: &[usize] = &[1, 2, 3, 4, 8, 12, 16];
const ACC_M: &[usize] = &[1, 3, 8, 12, 16];
/// Gate (pre-registered 2026-10-08): M=8 saves >= GO_SAVE_M8 ms and M=3 delta <= GO_DELTA_M3 ms
/// is GO; M=8 saves < KILL_SAVE_M8 ms is KILL; anything between is GREY.
const GO_SAVE_M8: f64 = 3.0;
const GO_DELTA_M3: f64 = 0.3;
const KILL_SAVE_M8: f64 = 1.5;
#[rustfmt::skip]
const E2M1: [f64; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

struct Lcg(u64);
#[rustfmt::skip]
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn n(&mut self) -> f64 { (self.u() + self.u() + self.u() + self.u() - 2.0) * 1.7320508 }
}

#[rustfmt::skip]
fn gen_weight(rng: &mut Lcg, n: usize, k: usize, heavy: bool) -> Vec<bf16> {
    let mut w = Vec::with_capacity(n * k);
    for _ in 0..n {
        let gain = if heavy { 0.25 * 16f64.powf(rng.u()) } else { 1.0 };
        for _ in 0..k {
            let mut v = rng.n() * 0.02 * gain;
            if heavy && rng.u() < 0.001 { v *= 30.0; }
            w.push(bf16::from_f64(v));
        }
    }
    w
}

/// Activation sets: 0 = N(0,1); 1 = large magnitude (x 3000 with 0.1 % x 30 outliers);
/// 2 = N(0,1) with row 1 all zeros.
#[rustfmt::skip]
fn gen_act(rng: &mut Lcg, m: usize, k: usize, set: usize) -> Vec<bf16> {
    let mut a = Vec::with_capacity(m * k);
    for t in 0..m {
        for _ in 0..k {
            let mut v = rng.n();
            if set == 1 { v *= 3000.0; if rng.u() < 0.001 { v *= 30.0; } }
            if set == 2 && t == 1 { v = 0.0; }
            a.push(bf16::from_f64(v));
        }
    }
    a
}

#[rustfmt::skip]
fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

#[rustfmt::skip]
fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// Kernel handles: quantizer, both tensor-core entries, and the CUDA-core production tiers
/// `cc` = [gemv, batch2, batch3, batch3_staged, batch4..=8, batch16].
struct Kq {
    absmax: KernelHandle,
    quant: KernelHandle,
    tc8: KernelHandle,
    tc16: KernelHandle,
    tc8g: KernelHandle,
    tc16g: KernelHandle,
    cc: Vec<KernelHandle>,
}

impl Kq {
    /// The production CUDA-core kernel for `m` rows (M = 3: `batch3_staged` when `staged`) and
    /// whether it takes the M argument (the 4..=16 tiers do).
    fn base(&self, m: usize, staged: bool) -> (KernelHandle, bool) {
        let i = match m {
            1 | 2 => m - 1,
            3 => 2 + usize::from(staged),
            4..=8 => m,
            _ => 9,
        };
        (self.cc[i], m >= 4)
    }
}

#[rustfmt::skip]
fn resolve(g: &dyn GpuBackend) -> Option<Kq> {
    let k = |m: &str, n: &str| g.kernel(m, n).ok();
    let names = ["", "_batch2", "_batch3", "_batch3_staged", "_batch4", "_batch5"];
    let more = ["_batch6", "_batch7", "_batch8", "_batch16"];
    let cc = names.iter().chain(&more).map(|s| k("w4a16_gemv", &format!("w4a16_gemv{s}")));
    Some(Kq {
        absmax: k("quantize_nvfp4", "nvfp4_global_absmax")?,
        quant: k("quantize_nvfp4", "quantize_bf16_to_nvfp4_mse")?,
        tc8: k("w4a16_gemv_tc", "w4a16_gemv_tc8")?,
        tc16: k("w4a16_gemv_tc", "w4a16_gemv_tc16")?,
        tc8g: k("w4a16_gemv_tc", "w4a16_gemv_tc8_group")?,
        tc16g: k("w4a16_gemv_tc", "w4a16_gemv_tc16_group")?,
        cc: cc.collect::<Option<Vec<_>>>()?,
    })
}

/// One launch: activations `a` (M rows), weight `q`, output `c`, shape N x K.
#[derive(Clone, Copy)]
struct Job<'a> {
    a: DevicePtr,
    q: &'a QuantizedWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    kd: usize,
}

/// Launches `h` with grid `div_ceil(n, cols)`; the M argument goes in only when `takes_m`.
#[rustfmt::skip]
fn launch(g: &dyn GpuBackend, h: KernelHandle, cols: u32, takes_m: bool, j: Job, s: u64) -> Result<()> {
    let mut l = KernelLaunch::new(g, h)
        .grid([div_ceil(j.n as u32, cols), 1, 1]).block([256, 1, 1])
        .arg_ptr(j.a).arg_ptr(j.q.weight).arg_ptr(j.q.weight_scale)
        .arg_f32(j.q.weight_scale_2).arg_ptr(j.c);
    if takes_m {
        l = l.arg_u32(j.m as u32);
    }
    l.arg_u32(j.n as u32).arg_u32(j.kd as u32).launch(s)
}

/// The lever's launch: tc8 for M <= 8, tc16 above (`force16`: tc16 at any M).
fn launch_tc(g: &dyn GpuBackend, k: &Kq, force16: bool, j: Job, s: u64) -> Result<()> {
    if force16 || j.m > 8 {
        launch(g, k.tc16, 16, true, j, s)
    } else {
        launch(g, k.tc8, 8, true, j, s)
    }
}

/// The production CUDA-core route.
fn launch_base(g: &dyn GpuBackend, k: &Kq, staged: bool, j: Job, s: u64) -> Result<()> {
    let (h, takes_m) = k.base(j.m, staged);
    launch(g, h, 4, takes_m, j, s)
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

#[rustfmt::skip]
fn quantize(g: &dyn GpuBackend, k: &Kq, w: &[bf16], n: usize, kd: usize) -> Result<QuantizedWeight> {
    let host: Vec<u8> = w.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let tmp = g.alloc(host.len())?;
    g.copy_h2d(&host, tmp)?;
    let dw = DenseWeight { weight: tmp };
    let s = g.default_stream();
    let q = quantize_to_nvfp4(&dw, n, kd, g, k.absmax, k.quant, s)?;
    g.synchronize(s)?;
    g.free(tmp)?;
    Ok(q)
}

fn free_q(g: &dyn GpuBackend, q: &QuantizedWeight) -> Result<()> {
    g.free(q.weight)?;
    g.free(q.weight_scale)
}

#[rustfmt::skip]
fn e4m3(b: u8) -> f64 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = ((b >> 3) & 0xF, (b & 7) as f64);
    if e == 0 { s * m * 2f64.powi(-9) } else { s * 2f64.powi(e as i32 - 7) * (1.0 + m / 8.0) }
}
#[rustfmt::skip]
fn bf(b: u16) -> f64 { f32::from_bits((b as u32) << 16) as f64 }
#[rustfmt::skip]
fn bf_ulp(v: f64) -> f64 { if v == 0.0 { 0.0 } else { 2f64.powi(v.abs().log2().floor() as i32 - 7) } }
#[rustfmt::skip]
fn to_u16(b: &[u8]) -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() }

/// Counters for (a) and (b); only the first 12 FAIL lines print.
#[derive(Default)]
struct Stats {
    row_cmp: usize,
    tc_cmp: usize,
    grp_cmp: usize,
    inv_fail: usize,
    acc_fail: usize,
}

impl Stats {
    fn inv(&mut self, line: String) {
        self.inv_fail += 1;
        if self.inv_fail <= 12 {
            println!("{line}");
        }
    }
}

/// (a) Row invariance and (b) accuracy for one shape.
#[rustfmt::skip]
fn check(g: &dyn GpuBackend, k: &Kq, rng: &mut Lcg, st: &mut Stats, n: usize, kd: usize, label: &str) -> Result<()> {
    let s = g.default_stream();
    let row = n * 2;
    let (refb, outm, out16) = (g.alloc(16 * row)?, g.alloc(17 * row)?, g.alloc(17 * row)?);
    for heavy in [false, true] {
        let w = gen_weight(rng, n, kd, heavy);
        let q = quantize(g, k, &w, n, kd)?;
        let job = |a, c, m| Job { a, q: &q, c, m, n, kd };
        for set in 0..3 {
            let host = gen_act(rng, 16, kd, set);
            let ad = up_bf16(g, &host)?;
            let tag = format!("{label} {} N={n} K={kd} acts={}", ["gauss", "heavy"][heavy as usize], ["random", "large", "zero-row"][set]);
            // Reference rows: the M = 1 tc8 launch on each row alone.
            for t in 0..16 {
                launch_tc(g, k, false, job(ad.offset(t * kd * 2), refb.offset(t * row), 1), s)?;
            }
            g.synchronize(s)?;
            let rb = dn_bytes(g, refb, 16 * row)?;
            for m in 1..=16usize {
                let poison = vec![0xA5u8; (m + 1) * row];
                g.copy_h2d(&poison, outm)?;
                g.copy_h2d(&poison, out16)?;
                launch_tc(g, k, false, job(ad, outm, m), s)?;
                if m <= 8 {
                    launch_tc(g, k, true, job(ad, out16, m), s)?;
                }
                g.synchronize(s)?;
                let y = dn_bytes(g, outm, (m + 1) * row)?;
                for t in 0..m {
                    st.row_cmp += 1;
                    if y[t * row..(t + 1) * row] != rb[t * row..(t + 1) * row] {
                        st.inv(format!("FAIL invariance {tag} M={m} row={t} differs from the M=1 launch"));
                    }
                }
                if y[m * row..].iter().any(|&b| b != 0xA5) {
                    st.inv(format!("FAIL invariance {tag} M={m} row={m} (one past) was written"));
                }
                if m <= 8 {
                    st.tc_cmp += 1;
                    if dn_bytes(g, out16, (m + 1) * row)? != y {
                        st.inv(format!("FAIL invariance {tag} M={m} row=all tc16 differs from tc8"));
                    }
                }
            }
            if !heavy && set == 0 {
                accuracy(g, k, st, job(ad, outm, 0), &host, label)?;
            }
            g.free(ad)?;
        }
        free_q(g, &q)?;
    }
    for p in [refb, outm, out16] {
        g.free(p)?;
    }
    Ok(())
}

/// (b) tc and the CUDA-core tier against an f64 reference on sampled columns (`j.a` = the
/// 16-row activations `host`; `j.c` is a scratch buffer of 16 rows).
#[rustfmt::skip]
fn accuracy(g: &dyn GpuBackend, k: &Kq, st: &mut Stats, j: Job, host: &[bf16], label: &str) -> Result<()> {
    let (s, n, kd, q) = (g.default_stream(), j.n, j.kd, j.q);
    let wb = dn_bytes(g, q.weight, n * kd / 2)?;
    let sb = dn_bytes(g, q.weight_scale, n * kd / 16)?;
    let cols: Vec<usize> = (0..64).map(|r| (r * 2654435761usize) % n).chain([0, n - 1]).collect();
    let cb = g.alloc(16 * n * 2)?;
    for &m in ACC_M {
        launch_tc(g, k, false, Job { m, ..j }, s)?;
        launch_base(g, k, true, Job { m, c: cb, ..j }, s)?;
        g.synchronize(s)?;
        let tc = to_u16(&dn_bytes(g, j.c, m * n * 2)?);
        let tier = to_u16(&dn_bytes(g, cb, m * n * 2)?);
        let (mut e_tier, mut e_tc, mut range) = (0f64, 0f64, 0f64);
        for r in 0..m {
            for &c in &cols {
                let mut acc = 0f64;
                for kk in 0..kd {
                    let byte = wb[c * kd / 2 + kk / 2];
                    let nib = if kk & 1 == 1 { byte >> 4 } else { byte & 0xF };
                    acc += host[r * kd + kk].to_f64() * E2M1[nib as usize] * e4m3(sb[c * kd / 16 + kk / 16]);
                }
                acc *= q.weight_scale_2 as f64;
                range = range.max(acc.abs());
                e_tier = e_tier.max((bf(tier[r * n + c]) - acc).abs());
                e_tc = e_tc.max((bf(tc[r * n + c]) - acc).abs());
            }
        }
        let ok = e_tc <= (1.25 * e_tier).max(bf_ulp(range) / 2.0);
        println!("ACC {label} M={m} tc {e_tc:.3e} tier {e_tier:.3e} (range {range:.2}) {}", if ok { "ok" } else { "BREACH" });
        st.acc_fail += usize::from(!ok);
    }
    g.free(cb)
}

/// (c) Timing of one shape over a cold pool; adds the per-step weighted ms to `base` / `tc`
/// (indexed like `TIME_M`) and `plain` (M = 3 plain batch3).
#[rustfmt::skip]
fn timing(g: &dyn GpuBackend, k: &Kq, rng: &mut Lcg, sums: &mut [[f64; 7]; 3], shape: (usize, usize, &str, usize)) -> Result<()> {
    let (n, kd, label, per_step) = shape;
    let s = g.create_stream()?;
    let nv_bytes = n * kd / 2 + n * kd / 16;
    let pool = POOL_NV4_BYTES.div_ceil(nv_bytes).clamp(4, 1024);
    let w = gen_weight(rng, n, kd, false);
    let mut qs = Vec::with_capacity(pool);
    for _ in 0..pool {
        qs.push(quantize(g, k, &w, n, kd)?);
    }
    let ad = up_bf16(g, &gen_act(rng, 16, kd, 0))?;
    let c = g.alloc(16 * n * 2)?;
    let t = |m: usize, f: &dyn Fn(Job, u64) -> Result<()>| -> Result<f64> {
        let ms = time_graph(g, s, &mut |s| qs.iter().try_for_each(|q| f(Job { a: ad, q, c, m, n, kd }, s)))?;
        Ok(ms / pool as f64)
    };
    let gbs = |ms: f64| nv_bytes as f64 / (ms * 1e-3) / 1e9;
    for (i, &m) in TIME_M.iter().enumerate() {
        let tb = t(m, &|j, s| launch_base(g, k, true, j, s))?;
        let tt = t(m, &|j, s| launch_tc(g, k, false, j, s))?;
        let line = |name: &str, base: f64| {
            println!(
                "TIMING {label} M={m} {name} {:.1} us {:.1} GB/s tc {:.1} us {:.1} GB/s speedup {:.2}x (pool {pool})",
                base * 1e3, gbs(base), tt * 1e3, gbs(tt), base / tt
            );
        };
        line("base", tb);
        sums[0][i] += tb * per_step as f64;
        sums[1][i] += tt * per_step as f64;
        if m == 3 {
            let tp = t(m, &|j, s| launch_base(g, k, false, j, s))?;
            line("b3plain", tp);
            sums[2][i] += tp * per_step as f64;
        }
    }
    for q in &qs {
        free_q(g, q)?;
    }
    g.free(ad)?;
    g.free(c)
}

/// 2026-10-09: The production persistent CTA cap of a grouped entry (SMs x resident CTAs per
/// SM at 256 threads; 0 when the backend cannot say).
fn group_cap(g: &dyn GpuBackend, h: KernelHandle) -> u32 {
    let sms = g.sm_count().unwrap_or(0);
    sms.saturating_mul(g.max_active_blocks_per_sm(h, 256, 0).unwrap_or(0))
}

/// 2026-10-09: One grouped launch of `qs` / `outs` / `ns` on input `a` at `grid_x` CTAs (the
/// entry's argument order: A, M, K, then three (B_packed, B_scale, scale2, C, N) slots).
#[rustfmt::skip]
#[allow(clippy::too_many_arguments)]
fn launch_group(g: &dyn GpuBackend, k: &Kq, a: DevicePtr, qs: &[&QuantizedWeight], outs: &[DevicePtr], ns: &[usize], m: usize, kd: usize, grid_x: u32, s: u64) -> Result<()> {
    let h = if m > 8 { k.tc16g } else { k.tc8g };
    let mut l = KernelLaunch::new(g, h).grid([grid_x, 1, 1]).block([256, 1, 1])
        .arg_ptr(a).arg_u32(m as u32).arg_u32(kd as u32);
    for i in 0..3 {
        l = match qs.get(i) {
            Some(q) => l.arg_ptr(q.weight).arg_ptr(q.weight_scale).arg_f32(q.weight_scale_2)
                .arg_ptr(outs[i]).arg_u32(ns[i] as u32),
            None => l.arg_ptr(DevicePtr(0)).arg_ptr(DevicePtr(0)).arg_f32(0.0)
                .arg_ptr(DevicePtr(0)).arg_u32(0),
        };
    }
    l.launch(s)
}

/// 2026-10-09: (e) bytes of a group: for M = 1..=16, the production grid (through
/// `gemv_tc::tc_group_launch`) and a `GROUP_SMALL_GRID` grid, every member's output against
/// its plain tc8/tc16 launch, rows >= M poisoned and unwritten.
#[rustfmt::skip]
fn group_check(g: &dyn GpuBackend, k: &Kq, rng: &mut Lcg, st: &mut Stats, grp: (&str, &[usize], usize, usize)) -> Result<()> {
    let (label, ns, kd, _) = grp;
    let s = g.default_stream();
    let mut qs = Vec::new();
    for &n in ns {
        qs.push(quantize(g, k, &gen_weight(rng, n, kd, true), n, kd)?);
    }
    let ad = up_bf16(g, &gen_act(rng, 16, kd, 2))?;
    let refs: Vec<DevicePtr> = ns.iter().map(|&n| g.alloc(17 * n * 2)).collect::<Result<_>>()?;
    let outs: Vec<DevicePtr> = ns.iter().map(|&n| g.alloc(17 * n * 2)).collect::<Result<_>>()?;
    let qr: Vec<&QuantizedWeight> = qs.iter().collect();
    for m in 1..=16usize {
        for (i, &n) in ns.iter().enumerate() {
            launch_tc(g, k, false, Job { a: ad, q: &qs[i], c: refs[i], m, n, kd }, s)?;
        }
        g.synchronize(s)?;
        let want: Vec<Vec<u8>> = ns.iter().enumerate().map(|(i, &n)| dn_bytes(g, refs[i], m * n * 2)).collect::<Result<_>>()?;
        for grid in [None, Some(GROUP_SMALL_GRID)] {
            for (i, &n) in ns.iter().enumerate() {
                g.copy_h2d(&vec![0xA5u8; 17 * n * 2], outs[i])?;
            }
            match grid {
                None => {
                    let mem: Vec<TcGroupMember> = (0..ns.len()).map(|i| TcGroupMember { weight: &qs[i], out: outs[i], n: ns[i] as u32 }).collect();
                    if !gemv_tc::tc_group_launch(g, ad, &mem, m as u32, kd as u32, s)? {
                        st.inv(format!("FAIL group {label} M={m}: tc_group_launch declined"));
                        continue;
                    }
                }
                Some(gx) => launch_group(g, k, ad, &qr, &outs, ns, m, kd, gx, s)?,
            }
            g.synchronize(s)?;
            let gname = grid.map_or("production".to_string(), |x| format!("{x}"));
            for (i, &n) in ns.iter().enumerate() {
                st.grp_cmp += 1;
                let y = dn_bytes(g, outs[i], 17 * n * 2)?;
                if y[..m * n * 2] != want[i][..] {
                    let col = (0..m * n).find(|&e| y[2 * e..2 * e + 2] != want[i][2 * e..2 * e + 2]).unwrap_or(0);
                    st.inv(format!("FAIL group {label} N={n} K={kd} M={m} grid={gname} member={i} differs from plain tc (first row {} col {})", col / n, col % n));
                }
                if y[m * n * 2..].iter().any(|&b| b != 0xA5) {
                    st.inv(format!("FAIL group {label} N={n} M={m} grid={gname} member={i} wrote rows >= M"));
                }
            }
        }
    }
    for p in refs.into_iter().chain(outs) {
        g.free(p)?;
    }
    for q in &qs {
        free_q(g, q)?;
    }
    g.free(ad)
}

/// 2026-10-09: (e) timing of a group over a cold pool of distinct weight sets: per group, the
/// plain per-projection tc launches against one grouped launch at the production grid; adds
/// the per-step weighted ms to `sums[0]` (tc) and `sums[1]` (group), indexed like
/// `GROUP_TIME_M`.
#[rustfmt::skip]
fn group_timing(g: &dyn GpuBackend, k: &Kq, rng: &mut Lcg, sums: &mut [[f64; 4]; 2], grp: (&str, &[usize], usize, usize)) -> Result<()> {
    let (label, ns, kd, per_step) = grp;
    let s = g.create_stream()?;
    let set_bytes: usize = ns.iter().map(|&n| n * kd / 2 + n * kd / 16).sum();
    let pool = POOL_NV4_BYTES.div_ceil(set_bytes).clamp(4, 1024);
    let ws: Vec<Vec<bf16>> = ns.iter().map(|&n| gen_weight(rng, n, kd, false)).collect();
    let mut sets: Vec<Vec<QuantizedWeight>> = Vec::with_capacity(pool);
    for _ in 0..pool {
        sets.push(ns.iter().zip(&ws).map(|(&n, w)| quantize(g, k, w, n, kd)).collect::<Result<_>>()?);
    }
    let ad = up_bf16(g, &gen_act(rng, 16, kd, 0))?;
    let outs: Vec<DevicePtr> = ns.iter().map(|&n| g.alloc(16 * n * 2)).collect::<Result<_>>()?;
    let (cap8, cap16) = (group_cap(g, k.tc8g), group_cap(g, k.tc16g));
    for (i, &m) in GROUP_TIME_M.iter().enumerate() {
        let tt = time_graph(g, s, &mut |s| {
            sets.iter().try_for_each(|set| {
                set.iter().enumerate().try_for_each(|(j, q)| launch_tc(g, k, false, Job { a: ad, q, c: outs[j], m, n: ns[j], kd }, s))
            })
        })? / pool as f64;
        let kind = if m > 8 { gemv_tc::TcKind::M16 } else { gemv_tc::TcKind::M8 };
        let ns32: Vec<u32> = ns.iter().map(|&n| n as u32).collect();
        let tiles = gemv_tc::tc_group_tiles(kind, &ns32);
        let grid = gemv_tc::tc_group_grid(tiles, if m > 8 { cap16 } else { cap8 });
        let tg = time_graph(g, s, &mut |s| {
            sets.iter().try_for_each(|set| {
                let qr: Vec<&QuantizedWeight> = set.iter().collect();
                launch_group(g, k, ad, &qr, &outs, ns, m, kd, grid, s)
            })
        })? / pool as f64;
        let gbs = |ms: f64| set_bytes as f64 / (ms * 1e-3) / 1e9;
        println!(
            "TIMING GROUP {label} M={m} tc {:.1} us {:.1} GB/s ({} launches) group {:.1} us {:.1} GB/s (1 launch, grid {grid} of {tiles} tiles) speedup {:.2}x (pool {pool})",
            tt * 1e3, gbs(tt), ns.len(), tg * 1e3, gbs(tg), tt / tg
        );
        sums[0][i] += tt * per_step as f64;
        sums[1][i] += tg * per_step as f64;
    }
    for set in &sets {
        for q in set {
            free_q(g, q)?;
        }
    }
    for p in outs {
        g.free(p)?;
    }
    g.free(ad)
}

#[rustfmt::skip]
fn run() -> Result<i32> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let Some(k) = resolve(g) else {
        eprintln!("w4a16_gemv_tc8/tc16 or a CUDA-core w4a16_gemv tier is not in this target's module set");
        return Ok(2);
    };
    let mut rng = Lcg(0x6e76_3474_6331);
    let (mut st, mut sums, mut ran) = (Stats::default(), [[0.0f64; 7]; 3], 0usize);
    for &shape in SHAPES {
        let (n, kd, label, _) = shape;
        if kd % 128 != 0 {
            println!("SKIP {label} K % 128 != 0 (route declines; CUDA-core tier at every M)");
            continue;
        }
        ran += 1;
        check(g, &k, &mut rng, &mut st, n, kd, label)?;
        timing(g, &k, &mut rng, &mut sums, shape)?;
    }
    let mut gsums = [[0.0f64; 4]; 2];
    let (mut before, mut after) = (0usize, 0usize);
    for &grp in GROUPS {
        if grp.2 % 128 != 0 {
            println!("SKIP group {} K % 128 != 0", grp.0);
            continue;
        }
        group_check(g, &k, &mut rng, &mut st, grp)?;
        if grp.3 > 0 {
            group_timing(g, &k, &mut rng, &mut gsums, grp)?;
            before += grp.1.len() * grp.3;
            after += grp.3;
        }
    }
    for &shape in SHAPES {
        let (n, kd, label, _) = shape;
        if kd % 128 == 0 {
            group_check(g, &k, &mut rng, &mut st, (label, &[n], kd, 0))?;
        }
    }
    for (i, &m) in GROUP_TIME_M.iter().enumerate() {
        let (t, gr) = (gsums[0][i], gsums[1][i]);
        println!("TIMING SUMMARY GROUP M={m} per step: tc {t:.2} ms -> group {gr:.2} ms, saves {:.2} ms (grouped launches {before} -> {after} per rank per step)", t - gr);
    }
    for (i, &m) in TIME_M.iter().enumerate() {
        let (b, t) = (sums[0][i], sums[1][i]);
        println!(
            "TIMING SUMMARY M={m} per step (launches per rank per step weighted, nine production shapes): base {b:.2} ms -> tc {t:.2} ms, saves {:.2} ms ({:.2}x)",
            b - t,
            b / t.max(1e-12)
        );
    }
    let at = |m: usize| TIME_M.iter().position(|&x| x == m).unwrap_or(0);
    let (i8, i3) = (at(8), at(3));
    println!("TIMING SUMMARY M=3 b3plain {:.2} ms (plain batch3, reference)", sums[2][i3]);
    let (saves8, delta3) = (sums[0][i8] - sums[1][i8], sums[1][i3] - sums[0][i3]);
    let (inv_ok, acc_ok) = (st.inv_fail == 0, st.acc_fail == 0);
    let word = |ok: bool| if ok { "PASS" } else { "FAIL" };
    println!("GATE invariance {} ({} row compares, {} tc8/tc16 compares, {} group member compares)", word(inv_ok), st.row_cmp, st.tc_cmp, st.grp_cmp);
    println!("GATE accuracy {}", word(acc_ok));
    let verdict = if saves8 >= GO_SAVE_M8 && delta3 <= GO_DELTA_M3 {
        "GO"
    } else if saves8 < KILL_SAVE_M8 {
        "KILL"
    } else {
        "GREY"
    };
    println!("GATE speed M=8 saves {saves8:.2} ms, M=3 delta {delta3:+.2} ms (GO needs M=8 saves >= 3.0 and M=3 delta <= +0.3; KILL if M=8 saves < 1.5): {verdict}");
    if inv_ok && acc_ok {
        println!("PASS: tc8/tc16 row-invariant (bitwise, M 1..=16) and within the accuracy bound on {ran} shapes x 2 weight sets x 3 activation sets; grouped entries byte-equal to plain tc on {} member compares", st.grp_cmp);
        Ok(0)
    } else {
        println!("FAIL invariance mismatches {} accuracy breaches {}", st.inv_fail, st.acc_fail);
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
