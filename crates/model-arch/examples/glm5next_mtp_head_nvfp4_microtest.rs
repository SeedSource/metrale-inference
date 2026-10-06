// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Gate 1 for `METRALE_GLM_MTP_HEAD_NVFP4` (`glm5next_mtp_head/head_nv4.rs`): the
//! NVFP4 copy of the GLM-5.3 MTP draft `lm_head` shard against the FP8 copy it replaces.
//!
//! Owner: model-arch examples.
//! Checks, on synthetic head weights at the TP2 shard shape (77428 x 4096) and a smaller
//! heavy-tailed one with `N % 4 = 1` (7741 x 4096):
//! - The copy is built as the head builds it: `dense_fp8::quantize_nvfp4_copy` from the BF16
//!   shard, swept by `dense_fp8::nv4_gemv_uncounted` (the head's `head_sweep_nv4`); the FP8
//!   copy by `quantize_to_fp8` + `dense_gemv_fp8w` (1 row) / `dense_gemv_fp8w_batchm`, as
//!   `init.rs`, `forward_one` and `propose_batch_impl` run it.
//! - (a) Row invariance: at every M in 1..=16 each output row's raw u16 bits equal the M = 1
//!   launch on that row (the batched propose drafts what the per-sequence propose drafts).
//! - (b) Numerics on 16 rows against FP64 host references: over the NVFP4 copy's own weights
//!   (kernel arithmetic; gate cos >= 0.999, max_rel <= 0.01) and over the BF16 weights
//!   (format error; gate cos >= 0.985, the dense NVFP4 gate's PROVISIONAL floor), with the FP8
//!   head's numbers beside them; nonfinite outputs and NaN scales fail.
//! - (c) Top-1 agreement over 256 random hidden states: argmax(NVFP4) == argmax(FP8), and
//!   each against argmax of the BF16-weight head (`dense_gemv_bf16`) as context. Reported,
//!   not gated (random states have near-tied maxima; real logits are peakier). The NVFP4 logits
//!   bitwise equal to the FP8 logits everywhere would mean the comparison is vacuous: fail.
//! - (d) Timing: CUDA events, warm, median of 60 launches, rows 1 and 2, FP8 vs NVFP4 head.
//!
//! Prints a line starting `PASS:` iff (a), (b) and the vacuity check pass, else `FAIL ...`
//! lines; timing is reported, never gated. The product gate is the serve's acceptance panel.
//!
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_mtp_head_nvfp4_microtest

use anyhow::{Context, Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_layer::dense_fp8 as df;
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight, QuantizedWeight};

#[path = "common/mtp_head_nvfp4_ref.rs"]
mod mtp_head_nvfp4_ref;
use mtp_head_nvfp4_ref::*;

// 2026-10-06: CUDA driver event API for kernel-only timing, declared as in
// `dsa_mla_split_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// (N, K, label, heavy-tailed weights).
const SHAPES: &[(usize, usize, &str, bool)] = &[
    (77428, 4096, "head_tp2", false),
    (7741, 4096, "small_heavy", true),
];
const MAX_M: usize = 16;
const TOP1_ROWS: usize = 256;
const COS_OWN_MIN: f64 = 0.999;
const MAXREL_OWN: f64 = 0.01;
/// 2026-10-06: PROVISIONAL, the dense NVFP4 gate's floor (E2M1 with a per-16 E4M3 scale keeps
/// ~8-10 % relative RMS error per weight, so cos ~0.995 against the BF16 weights is expected).
const COS_FMT_MIN: f64 = 0.985;
const WARMUP: usize = 10;
const TIMED: usize = 60;

struct Kern {
    fq: KernelHandle,
    fp8_1: KernelHandle,
    fp8_bm: KernelHandle,
    bf16_1: KernelHandle,
}

fn ck(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what}: status {rc}");
    }
    Ok(())
}

/// 2026-10-06: Median microseconds of `f` on stream `s` over `TIMED` launches, each between two
/// CUDA events, after `WARMUP` launches.
fn time_us(g: &dyn GpuBackend, s: u64, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        f()?;
    }
    g.synchronize(s)?;
    let (mut e0, mut e1): (u64, u64) = (0, 0);
    ck(unsafe { cuEventCreate(&mut e0, 0) }, "cuEventCreate(start)")?;
    ck(unsafe { cuEventCreate(&mut e1, 0) }, "cuEventCreate(end)")?;
    let mut us = Vec::with_capacity(TIMED);
    for _ in 0..TIMED {
        ck(unsafe { cuEventRecord(e0, s) }, "cuEventRecord(start)")?;
        f()?;
        ck(unsafe { cuEventRecord(e1, s) }, "cuEventRecord(end)")?;
        ck(unsafe { cuEventSynchronize(e1) }, "cuEventSynchronize")?;
        let mut ms = 0f32;
        ck(unsafe { cuEventElapsedTime(&mut ms, e0, e1) }, "cuEventElapsedTime")?;
        us.push(ms as f64 * 1e3);
    }
    unsafe {
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    us.sort_by(f64::total_cmp);
    Ok(us[us.len() / 2])
}

/// 2026-10-06: One shape's head copies and buffers.
struct Head<'a> {
    g: &'a dyn GpuBackend,
    k: &'a Kern,
    n: usize,
    kd: usize,
    s: u64,
    w: DevicePtr,
    f8: Fp8DenseWeight,
    q: QuantizedWeight,
}

impl Head<'_> {
    /// The NVFP4 head sweep over `m` rows of `x` (the head's `head_sweep_nv4`).
    fn nv(&self, x: DevicePtr, c: DevicePtr, m: usize) -> Result<()> {
        df::nv4_gemv_uncounted(self.g, x, &self.q, c, m, self.n, self.kd, self.s)
    }
    /// The FP8 head sweep: `dense_gemv_fp8w` at 1 row (`forward_one`), `dense_gemv_fp8w_batchm`
    /// above (`propose_batch_impl`).
    fn fp8(&self, x: DevicePtr, c: DevicePtr, m: usize) -> Result<()> {
        let (n, kd) = (self.n as u32, self.kd as u32);
        if m == 1 {
            ops::dense_gemv_fp8w(self.g, self.k.fp8_1, x, &self.f8, c, n, kd, self.s)
        } else {
            ops::dense_gemv_fp8w_batchm(
                self.g, self.k.fp8_bm, x, &self.f8, c, m as u32, 1, n, kd, n, self.s,
            )
        }
    }
    /// The BF16-weight head, one `dense_gemv_bf16` per row (the unsharded sweep's kernel).
    fn bf16(&self, x: DevicePtr, c: DevicePtr, m: usize) -> Result<()> {
        let w = DenseWeight { weight: self.w };
        for t in 0..m {
            ops::dense_gemv(
                self.g,
                self.k.bf16_1,
                x.offset(t * self.kd * 2),
                &w,
                c.offset(t * self.n * 2),
                self.n as u32,
                self.kd as u32,
                self.s,
            )?;
        }
        Ok(())
    }
}

/// 2026-10-06: (a) + (b) + (c) + (d) for one shape; failures are pushed to `fails`.
#[allow(clippy::too_many_arguments)]
fn run_shape(
    g: &dyn GpuBackend,
    k: &Kern,
    n: usize,
    kd: usize,
    label: &str,
    heavy: bool,
    seed: u64,
    fails: &mut Vec<String>,
) -> Result<()> {
    let s = g.default_stream();
    let t0 = std::time::Instant::now();
    let w_host = gen_weight(seed, n, kd, heavy);
    let w = up_u16(g, &w_host)?;
    let f8 = metrale_model_layers::weight_map::quantize_to_fp8(
        &DenseWeight { weight: w },
        n,
        kd,
        g,
        k.fq,
        s,
    )?;
    let Some(q) = df::quantize_nvfp4_copy(g, w, n, kd, s)? else {
        fails.push(format!("{label}: quantize_nvfp4_copy built no copy (kernels missing?)"));
        return Ok(());
    };
    let hd = Head { g, k, n, kd, s, w, f8, q };
    let mut rng = Lcg(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x6d74_7034);
    let act = gen_act(&mut rng, TOP1_ROWS * kd);
    let ad = up_u16(g, &act)?;
    let (c, c1, cb) = (
        g.alloc(MAX_M * n * 2)?,
        g.alloc(MAX_M * n * 2)?,
        g.alloc(MAX_M * n * 2)?,
    );
    println!(
        "SHAPE {label} [{n} x {kd}] {} weights: fp8 {} MB, nvfp4 {} MB (setup {:.1} s)",
        if heavy { "heavy" } else { "gauss" },
        (n * kd + 4 * n) / (1 << 20),
        (n * kd / 2 + n * kd / 16) / (1 << 20),
        t0.elapsed().as_secs_f64()
    );

    // (a) Row invariance across M.
    for t in 0..MAX_M {
        hd.nv(ad.offset(t * kd * 2), c1.offset(t * n * 2), 1)?;
    }
    g.synchronize(s)?;
    let ref1 = dn_u16(g, c1, MAX_M * n)?;
    let mut bad_m = Vec::new();
    for m in 1..=MAX_M {
        hd.nv(ad, c, m)?;
        g.synchronize(s)?;
        let y = dn_u16(g, c, m * n)?;
        let diff = y.iter().zip(&ref1[..m * n]).filter(|(a, b)| a != b).count();
        if diff != 0 {
            bad_m.push(format!("M={m}: {diff} elements"));
        }
    }
    println!(
        "BITS {label} nvfp4 rows at M=1..={MAX_M} vs the M=1 launch per row: {}",
        if bad_m.is_empty() { "exact".to_string() } else { bad_m.join(", ") }
    );
    if !bad_m.is_empty() {
        fails.push(format!("{label}: row invariance broken ({})", bad_m.join(", ")));
    }

    // (b) Numerics on MAX_M rows: `ref1` is the NVFP4 output; FP8 one row at a time.
    for t in 0..MAX_M {
        hd.fp8(ad.offset(t * kd * 2), c1.offset(t * n * 2), 1)?;
    }
    g.synchronize(s)?;
    let y8: Vec<f64> = dn_u16(g, c1, MAX_M * n)?.into_iter().map(bf).collect();
    let yn: Vec<f64> = ref1.iter().map(|&b| bf(b)).collect();
    let qh = Nv4Host {
        packed: dn_bytes(g, q.weight, n * kd / 2)?,
        scales: dn_bytes(g, q.weight_scale, n * kd / 16)?,
        s2: q.weight_scale_2 as f64,
    };
    let nan_scales = qh.scales.iter().filter(|&&b| e4m3(b).is_nan()).count();
    let tr = std::time::Instant::now();
    let (rb, ro) = fp64_refs(&w_host, &qh, &act, MAX_M, n, kd);
    // (min cos, max max_rel) of nvfp4 vs own-W, nvfp4 vs BF16-W, fp8 vs BF16-W; min nvfp4-vs-fp8 cos.
    let (mut own, mut fmt, mut f8s) = ((1.0f64, 0.0f64), (1.0f64, 0.0f64), (1.0f64, 0.0f64));
    let mut nvf8 = 1.0f64;
    let (mut en2, mut e82, mut nf) = (0.0f64, 0.0f64, 0usize);
    for t in 0..MAX_M {
        let r = t * n..(t + 1) * n;
        let (c_o, m_o, nf1) = row_stats(&yn[r.clone()], &ro[r.clone()]);
        let (c_f, m_f, _) = row_stats(&yn[r.clone()], &rb[r.clone()]);
        let (c_8, m_8, nf2) = row_stats(&y8[r.clone()], &rb[r.clone()]);
        let (c_n8, _, _) = row_stats(&yn[r.clone()], &y8[r.clone()]);
        nf += nf1 + nf2;
        own = (own.0.min(c_o), own.1.max(m_o));
        fmt = (fmt.0.min(c_f), fmt.1.max(m_f));
        f8s = (f8s.0.min(c_8), f8s.1.max(m_8));
        nvf8 = nvf8.min(c_n8);
        for j in r {
            en2 += (yn[j] - rb[j]).powi(2);
            e82 += (y8[j] - rb[j]).powi(2);
        }
    }
    let num_ok = own.0 >= COS_OWN_MIN
        && own.1 <= MAXREL_OWN
        && fmt.0 >= COS_FMT_MIN
        && nf == 0
        && nan_scales == 0;
    println!(
        "NUMERICS {label} rows={MAX_M} (refs {:.1} s): nvfp4-vs-fp64(own W) min_cos={:.8} max_rel={:.5} \
         | nvfp4-vs-fp64(bf16 W) min_cos={:.6} max_rel={:.4} | fp8-vs-fp64(bf16 W) min_cos={:.6} \
         max_rel={:.4} | nvfp4-vs-fp8 min_cos={nvf8:.6} | rms err nvfp4/fp8 {:.2}x | nonfinite={nf} \
         nan_scales={nan_scales}  {}",
        tr.elapsed().as_secs_f64(),
        own.0,
        own.1,
        fmt.0,
        fmt.1,
        f8s.0,
        f8s.1,
        (en2 / e82.max(1e-300)).sqrt(),
        if num_ok { "ok" } else { "OUT-OF-TOL" }
    );
    if !num_ok {
        fails.push(format!("{label}: numerics out of tolerance (see the NUMERICS line)"));
    }
    if fmt.0 < COS_OWN_MIN {
        println!(
            "NOTE {label}: nvfp4 vs fp64 over the BF16 weights is cos {:.6} < {COS_OWN_MIN}: the \
             E2M1 format error, not the kernel (own-weight cos {:.8}); fp8 reaches {:.6}",
            fmt.0, own.0, f8s.0
        );
    }

    // (c) Top-1 agreement over TOP1_ROWS random hidden states.
    let (mut a_n8, mut a_nb, mut a_8b, mut same) = (0usize, 0usize, 0usize, 0usize);
    for b in 0..TOP1_ROWS / MAX_M {
        let x = ad.offset(b * MAX_M * kd * 2);
        hd.nv(x, c, MAX_M)?;
        hd.fp8(x, c1, MAX_M)?;
        hd.bf16(x, cb, MAX_M)?;
        g.synchronize(s)?;
        let (ln, l8, lb) = (
            dn_u16(g, c, MAX_M * n)?,
            dn_u16(g, c1, MAX_M * n)?,
            dn_u16(g, cb, MAX_M * n)?,
        );
        same += ln.iter().zip(&l8).filter(|(a, b)| a == b).count();
        for t in 0..MAX_M {
            let r = t * n..(t + 1) * n;
            let (an, a8, ab) = (argmax(&ln[r.clone()]), argmax(&l8[r.clone()]), argmax(&lb[r]));
            a_n8 += usize::from(an == a8);
            a_nb += usize::from(an == ab);
            a_8b += usize::from(a8 == ab);
        }
    }
    let pct = |x: usize| 100.0 * x as f64 / TOP1_ROWS as f64;
    println!(
        "TOP1 {label} over {TOP1_ROWS} random hidden states: argmax nvfp4==fp8 {:.1} % | context vs \
         the BF16-weight head: nvfp4 {:.1} %, fp8 {:.1} % | logits bitwise nvfp4==fp8 {:.2} %",
        pct(a_n8),
        pct(a_nb),
        pct(a_8b),
        100.0 * same as f64 / (TOP1_ROWS * n) as f64
    );
    if same == TOP1_ROWS * n {
        fails.push(format!("{label}: NVFP4 logits equal FP8 logits everywhere; comparison is vacuous"));
    }

    // (d) Timing, rows 1 and 2.
    let (b8, bn) = (n * kd + 4 * n, n * kd / 2 + n * kd / 16);
    let gbs = |b: usize, us: f64| b as f64 / (us * 1e-6) / 1e9;
    for m in [1usize, 2] {
        let t8 = time_us(g, s, || hd.fp8(ad, c, m))?;
        let tn = time_us(g, s, || hd.nv(ad, c, m))?;
        println!(
            "TIMING {label} rows={m}: fp8 {t8:>8.1} us {:>6.1} GB/s | nvfp4 {tn:>8.1} us {:>6.1} GB/s | \
             saves {:>7.1} us per sweep ({:.2}x)",
            gbs(b8, t8),
            gbs(bn, tn),
            t8 - tn,
            t8 / tn.max(1e-9)
        );
    }

    for p in [ad, c, c1, cb, w, f8.weight, f8.row_scale, q.weight, q.weight_scale] {
        g.free(p).with_context(|| format!("{label}: free"))?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k = Kern {
        fq: g.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?,
        fp8_1: g.kernel("gemv_fp8w", "dense_gemv_fp8w")?,
        fp8_bm: g.kernel("dense_gemv_fp8w_batchm", "dense_gemv_fp8w_batchm")?,
        bf16_1: g.kernel("gemv", "dense_gemv_bf16")?,
    };
    let mut fails = Vec::new();
    for (i, &(n, kd, label, heavy)) in SHAPES.iter().enumerate() {
        run_shape(g, &k, n, kd, label, heavy, 0x6d74_7068_6e76_3400 + i as u64, &mut fails)?;
    }
    println!(
        "bounds: row bits exact at M=1..={MAX_M}; nvfp4 vs fp64(own W) cos>={COS_OWN_MIN} \
         max_rel<={MAXREL_OWN}; nvfp4 vs fp64(bf16 W) cos>={COS_FMT_MIN} (PROVISIONAL); nonfinite=0; \
         top-1 and timing reported, not gated"
    );
    if !fails.is_empty() {
        for f in &fails {
            println!("FAIL {f}");
        }
        std::process::exit(1);
    }
    println!(
        "PASS: MTP draft head NVFP4 rows bitwise == M=1 per row (M 1..={MAX_M}), cos >= {COS_OWN_MIN} \
         vs FP64 over its own weights, >= {COS_FMT_MIN} vs FP64 over the BF16 weights, timing printed"
    );
    Ok(())
}
