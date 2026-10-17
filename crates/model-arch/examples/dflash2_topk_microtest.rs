// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate for `METRALE_DFLASH_TOPK_REG`: `dflash2_topk16_reg` (one scan, each
//! thread's 16 best in registers, 16 warp-shuffle pops) against `dflash2_topk16` (16 full
//! block-argmax rescans), the DFlash2 selector's per-row top-16 (`dflash2_topk_block`).
//! Design reference only: no code copied or ported from TensorFold/Mia (Apache-2.0).
//!
//! Owner: model-arch examples (DFlash drafter).
//! Checks, per case (rows in {1, 8, 16} x vocab in {77440 (one rank's shard under
//! `METRALE_DFLASH_DRAFT_TP2`), 154880 (one-rank drafter)} x data in {normal: N(0, 4) BF16;
//! ties: round(N(0, 4)), so many rows' maxima tie; flat: every logit 1.0; zeros: +0 / -0 at
//! random, all equal under float compare}):
//! - `BITWISE`: both kernels on copies of the same logits; `top_vals` bits, `top_idx` and the
//!   whole logits buffer afterwards (the -1e30 writes) must be equal.
//! - `TIME`: warm L2 (the serve's logits were just written by the lm_head GEMV), CUDA events
//!   around one launch, median of 60 (5 warm-ups dropped), the logits restored by a D2D copy
//!   outside the events before every launch, the two kernels interleaved.
//! - Final line `PASS dflash2_topk (...)` when every case is bitwise equal and the saving at
//!   8 rows x 77440 (the TP2 propose's block) is >= 0.1 ms; else `FAIL dflash2_topk (...)`
//!   (exit 1), or exit 2 when a kernel is missing.
//!
//! Run (GB10, under a lane lock):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dflash2_topk_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const ROWS: &[usize] = &[1, 8, 16];
const VOCABS: &[usize] = &[77440, 154880];
const DATA: &[&str] = &["normal", "ties", "flat", "zeros"];
/// The timed case: the TP2 propose's block (gamma 8, one sequence) on one rank's shard.
const TIME_CASES: &[(usize, usize)] = &[(8, 77440), (16, 77440), (8, 154880)];
const GATE_CASE: (usize, usize) = (8, 77440);
const SAVING_MIN_MS: f64 = 0.1;
const TOP_K: usize = 16;
const REPS: usize = 60;
const WARMUP: usize = 5;

// CUDA driver event API, declared as in `dflash_nv4_drafter_microtest.rs`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
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

fn logits(rng: &mut Rng, data: &str, len: usize) -> Vec<bf16> {
    (0..len)
        .map(|_| match data {
            "normal" => bf16::from_f32(rng.normal() * 4.0),
            "ties" => bf16::from_f32((rng.normal() * 4.0).round()),
            "flat" => bf16::from_f32(1.0),
            _ => {
                if rng.next() & 1 == 0 {
                    bf16::from_f32(0.0)
                } else {
                    bf16::from_f32(-0.0)
                }
            }
        })
        .collect()
}

fn bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// One kernel's buffers: logits (restored from `src` before each launch), values, ids.
struct Bufs {
    logits: DevicePtr,
    vals: DevicePtr,
    idx: DevicePtr,
}

impl Bufs {
    fn new(g: &dyn GpuBackend, rows: usize, n: usize) -> Result<Self> {
        Ok(Self {
            logits: g.alloc(rows * n * 2)?,
            vals: g.alloc(rows * TOP_K * 4)?,
            idx: g.alloc(rows * TOP_K * 4)?,
        })
    }

    fn free(self, g: &dyn GpuBackend) -> Result<()> {
        for p in [self.logits, self.vals, self.idx] {
            g.free(p)?;
        }
        Ok(())
    }
}

/// The launch `dflash2_topk_block` makes: one 1024-thread block per row.
fn launch(
    g: &dyn GpuBackend,
    k: KernelHandle,
    b: &Bufs,
    rows: usize,
    n: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([rows as u32, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(b.logits)
        .arg_ptr(b.vals)
        .arg_ptr(b.idx)
        .arg_u32(n as u32)
        .launch(s)
}

/// Whether both kernels leave the same `top_vals`, `top_idx` and logits on `host`.
fn bitwise(
    g: &dyn GpuBackend,
    (old, reg): (KernelHandle, KernelHandle),
    host: &[bf16],
    (rows, n): (usize, usize),
    s: u64,
) -> Result<(bool, String)> {
    let raw: Vec<u8> = host
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    let (a, b) = (Bufs::new(g, rows, n)?, Bufs::new(g, rows, n)?);
    for (k, bufs) in [(old, &a), (reg, &b)] {
        g.copy_h2d(&raw, bufs.logits)?;
        g.memset(bufs.vals, 0, rows * TOP_K * 4)?;
        g.memset(bufs.idx, 0, rows * TOP_K * 4)?;
        g.synchronize(s)?;
        launch(g, k, bufs, rows, n, s)?;
        g.synchronize(s)?;
    }
    let v = bytes(g, a.vals, rows * TOP_K * 4)? == bytes(g, b.vals, rows * TOP_K * 4)?;
    let ia = bytes(g, a.idx, rows * TOP_K * 4)?;
    let i = ia == bytes(g, b.idx, rows * TOP_K * 4)?;
    let l = bytes(g, a.logits, rows * n * 2)? == bytes(g, b.logits, rows * n * 2)?;
    // Row 0's first picks, so a reader sees the tie order the case exercised.
    let first: Vec<u32> = ia[..4 * 4]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    a.free(g)?;
    b.free(g)?;
    Ok((
        v && i && l,
        format!("vals {v} idx {i} logits {l}; old row0 idx[..4] {first:?}"),
    ))
}

/// Warm-L2 medians (ms) of the old and the reg kernel on `rows x n` normal logits.
fn time_pair(
    g: &dyn GpuBackend,
    (old, reg): (KernelHandle, KernelHandle),
    rng: &mut Rng,
    (rows, n): (usize, usize),
    (ev, s): (&Events, u64),
) -> Result<(f64, f64)> {
    let host = logits(rng, "normal", rows * n);
    let raw: Vec<u8> = host
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    let src = g.alloc(raw.len())?;
    g.copy_h2d(&raw, src)?;
    let b = Bufs::new(g, rows, n)?;
    let mut t = [Vec::with_capacity(REPS), Vec::with_capacity(REPS)];
    for rep in 0..WARMUP + REPS {
        for (j, k) in [old, reg].into_iter().enumerate() {
            g.copy_d2d_async(src, b.logits, raw.len(), s)?;
            ev.rec(0, s)?;
            launch(g, k, &b, rows, n, s)?;
            ev.rec(1, s)?;
            let us = ev.us()?;
            if rep >= WARMUP {
                t[j].push(us);
            }
        }
    }
    g.free(src)?;
    b.free(g)?;
    let [a, c] = t;
    Ok((median(a) / 1e3, median(c) / 1e3))
}

fn run() -> Result<i32> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let (Ok(old), Ok(reg)) = (
        g.kernel("dflash2", "dflash2_topk16"),
        g.kernel("dflash2", "dflash2_topk16_reg"),
    ) else {
        println!("FAIL dflash2_topk (kernel entry dflash2::dflash2_topk16 or _reg missing)");
        return Ok(2);
    };
    let s = g.create_stream()?;
    let mut rng = Rng(0x0064_6674_6f70_6b01);
    let mut fails = Vec::new();
    let mut cases = 0usize;
    for &n in VOCABS {
        for &rows in ROWS {
            for &data in DATA {
                let host = logits(&mut rng, data, rows * n);
                let (ok, why) = bitwise(g, (old, reg), &host, (rows, n), s)?;
                cases += 1;
                println!(
                    "BITWISE {data:<6} rows {rows:>2} vocab {n}: {} ({why})",
                    if ok { "PASS" } else { "FAIL" }
                );
                if !ok {
                    fails.push(format!("{data}/{rows}/{n}"));
                }
            }
        }
    }
    let ev = Events::new()?;
    let mut gate_saving = f64::NAN;
    for &(rows, n) in TIME_CASES {
        let (t_old, t_reg) = time_pair(g, (old, reg), &mut rng, (rows, n), (&ev, s))?;
        println!(
            "TIME rows {rows:>2} vocab {n} warm-L2 median of {REPS}: dflash2_topk16 {t_old:.4} ms | \
             dflash2_topk16_reg {t_reg:.4} ms | saving {:.4} ms ({:.2}x)",
            t_old - t_reg,
            t_old / t_reg
        );
        if (rows, n) == GATE_CASE {
            gate_saving = t_old - t_reg;
        }
    }
    let ok = fails.is_empty() && gate_saving >= SAVING_MIN_MS;
    let summary = format!(
        "bitwise {}/{cases} cases (fail {fails:?}); saving at {} rows x {} = {gate_saving:.4} ms \
         per propose round (need >= {SAVING_MIN_MS})",
        cases - fails.len(),
        GATE_CASE.0,
        GATE_CASE.1
    );
    if ok {
        println!("PASS dflash2_topk ({summary})");
        Ok(0)
    } else {
        println!("FAIL dflash2_topk ({summary})");
        Ok(1)
    }
}

fn main() {
    match run() {
        Ok(0) => {}
        Ok(c) => std::process::exit(c),
        Err(e) => {
            println!("FAIL dflash2_topk (error: {e:#})");
            std::process::exit(1);
        }
    }
}
