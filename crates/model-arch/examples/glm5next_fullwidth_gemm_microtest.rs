// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Numeric gate and timing for `METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1`: every GLM-5.3
//! prefill dense projection the lever widens, at its TP2 rank-local shape, run the way the
//! staged prefill runs it today and as ONE full-width GEMM.
//!
//! * Today, "sliced": cuBLASLt (`ops::cublas_bf16_proj_dense[_f32_out]`, what
//!   `cublas_wide_proj` selects above `DENSE_GEMV_BATCHM_MAX_M`) once per 256-row slice
//!   (`METRALE_GLM_PREFILL_ROWS=256`): the KDA, DSA q/kv/o, router, shared-expert and dense-MLP
//!   projections.
//! * Today, "gemv16": the DSA indexer projections (`wq_b`, `wk`, `compress_gate`,
//!   `weights_proj`) through the batched GEMV (`dense_gemv_bf16[_fp32out]_batchm`), one launch
//!   per 16 rows, as `METRALE_GLM_DSA_ROW_BATCH=1` runs them (each row carries the M = 1 GEMV's
//!   bits, so this is also the default per-row path's output, at a lower launch count).
//! * Full width: one cuBLASLt GEMM over all rows (`decode_k_wide`, `dense_slice = window`).
//!
//! Correctness at 4096 and 4095 rows (the second leaves a 255-row tail slice): cosine between
//! the two outputs over all elements must exceed `COSINE_GATE` (0.9999) and every value must be
//! finite; max abs error and max error relative to the largest |today| value are printed. The
//! two arms sum in different orders, so equality is not expected. A negative control compares
//! the full-width output against today's shifted by one row and must FAIL the gate, or the
//! harness is vacuous.
//!
//! Timing (CUDA events around back-to-back calls, after one warm-up, mean per call) at 4096 and
//! 8192 rows, one TIMING line per projection, then per row count the sum over one prefill of
//! that many tokens, weighting each projection by how often GLM-5.3 runs it (34 KDA layers, 11
//! DSA, 42 routed-MoE, 3 dense). Rank-local GEMM time only; the full-width path's other
//! effects (fewer mHC/norm launches, the DSA metadata uploads) are not timed here. Timing does
//! not decide PASS.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and `PASS` when every cosine clears the gate and the control fires; 1 and `FAIL`
//! otherwise; 2 when a batched GEMV kernel is absent from this target.
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_fullwidth_gemm_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

// 2026-10-01: CUDA driver event API for kernel-only timing, declared as in
// `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-01: Minimum cosine between today's output and the full-width output.
const COSINE_GATE: f64 = 0.9999;
/// 2026-10-01: The served attention sub-chunk (`METRALE_GLM_PREFILL_ROWS=256`), the width of
/// today's sliced GEMMs.
const SLICE: usize = 256;
/// 2026-10-01: `DENSE_GEMV_BATCHM_MAX_M`, the per-launch row cap of today's indexer GEMVs.
const GEMV_ROWS: usize = ops::DENSE_GEMV_BATCHM_MAX_M as usize;
const CHECK_ROWS: &[usize] = &[4096, 4095];
const TIMING_ROWS: &[usize] = &[4096, 8192];
const ITERS: usize = 5;

/// 2026-10-01: How the staged prefill runs a projection today.
#[derive(Clone, Copy, PartialEq)]
enum Today {
    /// 2026-10-01: cuBLASLt per `SLICE` rows.
    Sliced,
    /// 2026-10-01: The batched GEMV per `GEMV_ROWS` rows.
    Gemv16,
}

/// 2026-10-01: One projection: label, `N` (output features), `K` (input features), FP32
/// output, today's path, and how many times one prefilled token runs it across the stack.
struct Proj {
    label: &'static str,
    n: usize,
    k: usize,
    f32_out: bool,
    today: Today,
    per_token: usize,
}

const fn p(
    label: &'static str,
    n: usize,
    k: usize,
    f32_out: bool,
    today: Today,
    per_token: usize,
) -> Proj {
    Proj {
        label,
        n,
        k,
        f32_out,
        today,
        per_token,
    }
}

/// 2026-10-01: GLM-5.3-Flash at TP2, rank-local shapes: hidden 4096; DSA 32 local heads x
/// kv_lora 512 (absorbed q and o), q_lora 1536, indexer 32 x 128 (replicated); KDA 32 local
/// heads x 128, gate rank 128; MoE router 288 experts, shared expert 1024 local; dense MLP
/// 6144 local. Counts: 11 DSA, 34 KDA (q/k/v 3x, f/g low-rank 2x each), 42 MoE (gate+up 2x),
/// 3 dense (gate+up 2x).
const PROJS: &[Proj] = &[
    p("dsa q_a_proj", 1536, 4096, false, Today::Sliced, 11),
    p("dsa q_absorb", 16384, 1536, false, Today::Sliced, 11),
    p("dsa kv_a_proj", 512, 4096, false, Today::Sliced, 11),
    p("dsa o_absorb", 4096, 16384, false, Today::Sliced, 11),
    p("dsa idx wq_b", 4096, 1536, true, Today::Gemv16, 11),
    p("dsa idx wk", 128, 4096, false, Today::Gemv16, 11),
    p("dsa idx comp_gate", 128, 4096, false, Today::Gemv16, 11),
    p("dsa idx weights_proj", 32, 4096, true, Today::Gemv16, 11),
    p("kda q/k/v_proj", 4096, 4096, false, Today::Sliced, 102),
    p("kda f_a/g_a", 128, 4096, false, Today::Sliced, 68),
    p("kda f_b/g_b", 4096, 128, false, Today::Sliced, 68),
    p("kda b_proj", 32, 4096, false, Today::Sliced, 34),
    p("kda o_proj", 4096, 4096, false, Today::Sliced, 34),
    p("moe router", 288, 4096, true, Today::Sliced, 42),
    p("moe shared gate/up", 1024, 4096, false, Today::Sliced, 84),
    p("moe shared down", 4096, 1024, false, Today::Sliced, 42),
    p("dense gate/up", 6144, 4096, false, Today::Sliced, 6),
    p("dense down", 4096, 6144, false, Today::Sliced, 3),
];

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn bf16_bytes(&mut self, n: usize, scale: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                let v = (2.0 * self.f() - 1.0) * scale;
                bf16::from_f32(v).to_bits().to_le_bytes()
            })
            .collect()
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

/// 2026-10-01: `n` output values at `p` as f32 (BF16 widened when `!f32_out`).
fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize, f32_out: bool) -> Result<Vec<f32>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n * if f32_out { 4 } else { 2 }];
    g.copy_d2h(p, &mut b)?;
    Ok(if f32_out {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        b.chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    })
}

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

/// 2026-10-01: Mean ms per call of `f` over `ITERS` back-to-back calls on stream 0, CUDA events
/// around the loop, after one warm-up call.
fn time_ms(g: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    g.synchronize(0)?;
    let (mut e0, mut e1): (u64, u64) = (0, 0);
    // SAFETY: plain CUDA driver calls on events this function creates and destroys; the
    // backend made the context current when it was built.
    unsafe {
        check(cuEventCreate(&mut e0, 0), "cuEventCreate(start)")?;
        check(cuEventCreate(&mut e1, 0), "cuEventCreate(end)")?;
        check(cuEventRecord(e0, 0), "cuEventRecord(start)")?;
    }
    for _ in 0..ITERS {
        f()?;
    }
    let mut ms: f32 = 0.0;
    // SAFETY: as above; `ms` outlives the call.
    unsafe {
        check(cuEventRecord(e1, 0), "cuEventRecord(end)")?;
        check(cuEventSynchronize(e1), "cuEventSynchronize(end)")?;
        check(cuEventElapsedTime(&mut ms, e0, e1), "cuEventElapsedTime")?;
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    Ok(ms as f64 / ITERS as f64)
}

/// 2026-10-01: The batched GEMV kernels today's indexer path launches.
struct Kernels {
    batchm: KernelHandle,
    batchm_f32: KernelHandle,
}

/// 2026-10-01: One cuBLASLt GEMM, `c[m, n] = a[m, k] @ w[n, k]^T`, BF16 or FP32 out.
fn cublas(
    a: DevicePtr,
    w: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    f32_out: bool,
) -> Result<()> {
    let (m, n, k) = (m as u32, n as u32, k as u32);
    if f32_out {
        ops::cublas_bf16_proj_dense_f32_out(a, w, c, m, n, k, 0)
    } else {
        ops::cublas_bf16_proj_dense(a, w, c, m, n, k, 0)
    }
}

/// 2026-10-01: Today's path for `pr` over `rows` rows.
fn run_today(
    g: &dyn GpuBackend,
    ks: &Kernels,
    pr: &Proj,
    a: DevicePtr,
    w: DevicePtr,
    c: DevicePtr,
    rows: usize,
) -> Result<()> {
    let elem = if pr.f32_out { 4 } else { 2 };
    let step = match pr.today {
        Today::Sliced => SLICE,
        Today::Gemv16 => GEMV_ROWS,
    };
    let wt = DenseWeight { weight: w };
    let mut r0 = 0;
    while r0 < rows {
        let m = step.min(rows - r0);
        let (a_r, c_r) = (a.offset(r0 * pr.k * 2), c.offset(r0 * pr.n * elem));
        match pr.today {
            Today::Sliced => cublas(a_r, w, c_r, m, pr.n, pr.k, pr.f32_out)?,
            Today::Gemv16 => {
                let (m, n, k) = (m as u32, pr.n as u32, pr.k as u32);
                if pr.f32_out {
                    let kf = ks.batchm_f32;
                    ops::dense_gemv_batchm_fp32out(g, kf, a_r, &wt, c_r, m, n, k, n, 0)?;
                } else {
                    ops::dense_gemv_batchm(g, ks.batchm, a_r, &wt, c_r, m, n, k, n, 0)?;
                }
            }
        }
        r0 += step;
    }
    Ok(())
}

/// 2026-10-01: Cosine, max abs error and max error relative to max |a| between `a` and `b`;
/// `None` when a value is not finite.
fn compare(a: &[f32], b: &[f32]) -> Option<(f64, f64, f64)> {
    let (mut dot, mut na, mut nb, mut max_err, mut max_a) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        max_err = max_err.max((x - y).abs());
        max_a = max_a.max(x.abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE);
    Some((cos, max_err, max_err / max_a.max(f64::MIN_POSITIVE)))
}

/// 2026-10-01: Per-projection device inputs, sized for the widest row count used.
struct Bufs {
    a: DevicePtr,
    w: DevicePtr,
    c_today: DevicePtr,
    c_full: DevicePtr,
}

impl Bufs {
    fn new(g: &dyn GpuBackend, pr: &Proj, rows: usize, seed: u64) -> Result<Self> {
        let mut rng = Lcg(seed ^ (pr.n * pr.k) as u64);
        let elem = if pr.f32_out { 4 } else { 2 };
        Ok(Self {
            a: up(g, &rng.bf16_bytes(rows * pr.k, 1.0))?,
            w: up(g, &rng.bf16_bytes(pr.n * pr.k, 1.0 / (pr.k as f32).sqrt()))?,
            c_today: g.alloc(rows * pr.n * elem)?,
            c_full: g.alloc(rows * pr.n * elem)?,
        })
    }
    fn free(self, g: &dyn GpuBackend) {
        for p in [self.a, self.w, self.c_today, self.c_full] {
            g.free(p).ok();
        }
    }
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut handles = Vec::new();
    for func in ["dense_gemv_bf16_batchm", "dense_gemv_bf16_fp32out_batchm"] {
        match g.kernel("dense_gemv_bf16_batchm", func) {
            Ok(h) => handles.push(h),
            Err(e) => {
                println!("dense_gemv_bf16_batchm::{func} absent from this target ({e}) - SKIP");
                std::process::exit(2);
            }
        }
    }
    let ks = Kernels {
        batchm: handles[0],
        batchm_f32: handles[1],
    };
    let max_rows = TIMING_ROWS
        .iter()
        .chain(CHECK_ROWS)
        .copied()
        .max()
        .unwrap_or(1);

    let (mut compared, mut failed) = (0usize, 0usize);
    let mut control_fired = None;
    // 2026-10-01: Per timing row count: (today ms, full-width ms) summed over one prefill.
    let mut totals = vec![(0f64, 0f64); TIMING_ROWS.len()];
    for (pi, pr) in PROJS.iter().enumerate() {
        let elem = if pr.f32_out { 4 } else { 2 };
        let bufs = Bufs::new(g, pr, max_rows, 0xF017_D7E5)?;
        let today = match pr.today {
            Today::Sliced => "sliced256",
            Today::Gemv16 => "gemv16",
        };
        for &rows in CHECK_ROWS {
            g.memset(bufs.c_today, 0xA5, rows * pr.n * elem)?;
            g.memset(bufs.c_full, 0x5A, rows * pr.n * elem)?;
            run_today(g, &ks, pr, bufs.a, bufs.w, bufs.c_today, rows)?;
            cublas(bufs.a, bufs.w, bufs.c_full, rows, pr.n, pr.k, pr.f32_out)?;
            let t = down_f32(g, bufs.c_today, rows * pr.n, pr.f32_out)?;
            let f = down_f32(g, bufs.c_full, rows * pr.n, pr.f32_out)?;
            compared += t.len();
            let ok = match compare(&t, &f) {
                Some((cos, abs, rel)) => {
                    let ok = cos > COSINE_GATE;
                    println!(
                        "{:<22} N={:<5} K={:<5} rows={rows:<5} {today:<9} vs full: cos={cos:.8} \
                         max_abs={abs:.3e} max_rel={rel:.3e} {}",
                        pr.label,
                        pr.n,
                        pr.k,
                        if ok { "ok" } else { "BELOW GATE" }
                    );
                    ok
                }
                None => {
                    println!("{:<22} rows={rows}: NON-FINITE output", pr.label);
                    false
                }
            };
            failed += usize::from(!ok);
            // 2026-10-01: Negative control on the first projection: row `r` of the full-width
            // output against row `r + 1` of today's must fall below the gate.
            if pi == 0 && rows == CHECK_ROWS[0] {
                let n = pr.n;
                let shifted = compare(&t[n..], &f[..f.len() - n]);
                let fired = shifted.is_none_or(|(cos, _, _)| cos <= COSINE_GATE);
                println!("CONTROL one-row shift detected={fired}");
                control_fired = Some(fired);
            }
        }
        for (ti, &rows) in TIMING_ROWS.iter().enumerate() {
            let (a, w, c) = (bufs.a, bufs.w, bufs.c_today);
            let old = time_ms(g, || run_today(g, &ks, pr, a, w, c, rows))?;
            let new = time_ms(g, || cublas(a, w, c, rows, pr.n, pr.k, pr.f32_out))?;
            println!(
                "TIMING {:<22} N={:<5} K={:<5} rows={rows:<5}: {today} {old:.4} ms, full-width \
                 {new:.4} ms ({:.2}x), runs {} x per prefill",
                pr.label,
                pr.n,
                pr.k,
                old / new,
                pr.per_token
            );
            totals[ti].0 += old * pr.per_token as f64;
            totals[ti].1 += new * pr.per_token as f64;
        }
        bufs.free(g);
    }
    for (ti, &rows) in TIMING_ROWS.iter().enumerate() {
        let (old, new) = totals[ti];
        println!(
            "TIMING TOTAL rows={rows}: one {rows}-token prefill, rank-local projection GEMMs: \
             today {:.1} ms, full-width {:.1} ms, saved {:.1} ms",
            old,
            new,
            old - new
        );
    }
    // 2026-10-01: The served shape: an 8192-token chunk as two 4096-row windows
    // (`METRALE_GLM_PREFILL_ROWS_FFN=4096`). Today's per-slice cost does not depend on the
    // chunk, so today's 8192-row total stands; full width is twice the 4096-row total.
    if let (Some(i4), Some(i8)) = (
        TIMING_ROWS.iter().position(|&r| r == 4096),
        TIMING_ROWS.iter().position(|&r| r == 8192),
    ) {
        let (old, new) = (totals[i8].0, 2.0 * totals[i4].1);
        println!(
            "TIMING TOTAL 8192 tokens as 2 x 4096-row windows: today {old:.1} ms, full-width \
             {new:.1} ms, saved {:.1} ms",
            old - new
        );
    }

    if compared == 0 {
        println!("FAIL - no element was compared; this run proves nothing.");
        std::process::exit(1);
    }
    if control_fired != Some(true) {
        println!("FAIL - the negative control did not fire; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if failed > 0 {
        println!(
            "FAIL - {failed} leg(s) below cosine {COSINE_GATE} or non-finite. Keep \
             METRALE_GLM_PREFILL_FULLWIDTH_GEMM off on this build."
        );
        std::process::exit(1);
    }
    println!(
        "PASS - {compared} elements: every full-width projection is within cosine \
         {COSINE_GATE} of today's sliced / batched-GEMV output."
    );
    Ok(())
}
