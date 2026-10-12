// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate for `METRALE_DFLASH_NV4_GROUP`: the DFlash2 drafter's per-layer q/k/v
//! (one input) and gate/up (one input) NVFP4 copies as one `gemv_tc::tc_group_launch`
//! (`w4a16_gemv_tc{8,16}_group`) each, against the per-projection tensor-core launches
//! (`dense_fp8::nv4_gemv_tensor_core`, what `METRALE_DFLASH_NV4_TC` runs today). Design
//! reference only: no code copied or ported from TensorFold/Mia (Apache-2.0).
//!
//! Owner: model-arch examples (DFlash drafter).
//! Groups (one rank under `METRALE_DFLASH_DRAFT_TP2`, GLM-5.3): qkv = q [2048, 4096], k / v
//! [512, 4096]; gateup = gate / up [6144, 4096]. Gaussian weights (std 0.02), N(0, 1)
//! activations, copies from `dense_fp8::quantize_nvfp4_copy`.
//! - `BITWISE`: at M in {1, 2, 4, 8, 12, 16}, every member's whole output from the grouped
//!   launch equals the per-projection launch, byte for byte.
//! - `TIME`: cold L2 (a 72 MiB read-only flush before each arm), CUDA events around the arm
//!   (the per-projection launches back to back, or the one grouped launch), median of 60 (5
//!   warm-ups dropped), arms interleaved, M = 8 (the TP2 propose's block). A `VOID` line when
//!   a rate exceeds 255 GB/s.
//! - `ROUND`: (per-projection - grouped) x 5 layers, both groups.
//! - Final line `PASS dflash_nv4_group (...)` when every BITWISE check passes, no timing is
//!   VOID and the round saving is >= 0.1 ms; else `FAIL dflash_nv4_group (...)` (exit 1), or
//!   exit 2 when a kernel is missing.
//!
//! Run (GB10, under a lane lock):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dflash_nv4_group_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_arch::glm5next_layer::dense_fp8;
use metrale_model_layers::layers::ops::gemv_tc::{self, TcGroupMember};
use metrale_model_layers::weight_map::QuantizedWeight;

/// (label, member n's, K, groups per propose round).
const GROUPS: &[(&str, &[usize], usize, usize)] = &[
    ("qkv", &[2048, 512, 512], 4096, 5),
    ("gateup", &[6144, 6144], 4096, 5),
];
const MS: &[usize] = &[1, 2, 4, 8, 12, 16];
const M_TIME: usize = 8;
const GBS_VOID: f64 = 255.0;
const SAVING_MIN_MS: f64 = 0.1;
const REPS: usize = 60;
const WARMUP: usize = 5;
/// Flush weight: 4096 x 32768 NVFP4 = 64 MiB packed + 8 MiB scales, read-only, as in
/// `dflash_nv4_drafter_microtest.rs`.
const FLUSH_N: usize = 4096;
const FLUSH_K: usize = 32768;

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

/// The 72 MiB read-only flush: its buffers and the `w4a16_gemv` entry.
struct Flush {
    k: KernelHandle,
    w: DevicePtr,
    s: DevicePtr,
    a: DevicePtr,
    c: DevicePtr,
}

impl Flush {
    fn new(g: &dyn GpuBackend) -> Result<Self> {
        let f = Self {
            k: g.kernel("w4a16_gemv", "w4a16_gemv")?,
            w: g.alloc(FLUSH_N * FLUSH_K / 2)?,
            s: g.alloc(FLUSH_N * FLUSH_K / 16)?,
            a: g.alloc(FLUSH_K * 2)?,
            c: g.alloc(FLUSH_N * 2)?,
        };
        g.memset(f.w, 0, FLUSH_N * FLUSH_K / 2)?;
        g.memset(f.s, 0, FLUSH_N * FLUSH_K / 16)?;
        g.memset(f.a, 0, FLUSH_K * 2)?;
        g.synchronize(g.default_stream())?;
        Ok(f)
    }

    fn run(&self, g: &dyn GpuBackend, s: u64) -> Result<()> {
        KernelLaunch::new(g, self.k)
            .grid([div_ceil(FLUSH_N as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.a)
            .arg_ptr(self.w)
            .arg_ptr(self.s)
            .arg_f32(1.0)
            .arg_ptr(self.c)
            .arg_u32(FLUSH_N as u32)
            .arg_u32(FLUSH_K as u32)
            .launch(s)
    }

    fn free(self, g: &dyn GpuBackend) -> Result<()> {
        for p in [self.w, self.s, self.a, self.c] {
            g.free(p)?;
        }
        Ok(())
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

fn up(g: &dyn GpuBackend, v: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// One group's outcome: bitwise at every M, ms per arm at M_TIME (per-projection, grouped),
/// and whether a rate was VOID.
struct GroupOut {
    bitwise: bool,
    ms: [f64; 2],
    void: bool,
}

fn run_group(
    g: &dyn GpuBackend,
    rng: &mut Rng,
    (label, ns, kk): (&str, &[usize], usize),
    (ev, fl): (&Events, &Flush),
) -> Result<GroupOut> {
    let s = g.create_stream()?;
    let mmax = *MS.iter().max().expect("MS");
    let mut ws = Vec::new();
    let mut qs: Vec<QuantizedWeight> = Vec::new();
    for &n in ns {
        let wh: Vec<bf16> = (0..n * kk)
            .map(|_| bf16::from_f32(rng.normal() * 0.02))
            .collect();
        let w = up(g, &wh)?;
        let Some(q) = dense_fp8::quantize_nvfp4_copy(g, w, n, kk, s)? else {
            bail!("{label}: quantize_nvfp4_copy built nothing (NVFP4 kernels missing?)");
        };
        ws.push(w);
        qs.push(q);
    }
    let ah: Vec<bf16> = (0..mmax * kk)
        .map(|_| bf16::from_f32(rng.normal()))
        .collect();
    let a = up(g, &ah)?;
    // Outputs: [arm][member], each `[mmax, n]` BF16.
    let outs: Vec<Vec<DevicePtr>> = (0..2)
        .map(|_| ns.iter().map(|&n| g.alloc(mmax * n * 2)).collect())
        .collect::<Result<_>>()?;
    let sep = |m: usize, st: u64| -> Result<()> {
        for ((q, &n), &c) in qs.iter().zip(ns).zip(&outs[0]) {
            dense_fp8::nv4_gemv_tensor_core(g, a, q, c, m, n, kk, st)?;
        }
        Ok(())
    };
    let members: Vec<TcGroupMember> = qs
        .iter()
        .zip(ns)
        .zip(&outs[1])
        .map(|((q, &n), &c)| TcGroupMember {
            weight: q,
            out: c,
            n: n as u32,
        })
        .collect();
    let grp = |m: usize, st: u64| -> Result<()> {
        if !gemv_tc::tc_group_launch(g, a, &members, m as u32, kk as u32, st)? {
            bail!("{label}: tc_group_launch declined {m} rows (group entry missing?)");
        }
        Ok(())
    };
    let mut bitwise = true;
    for &m in MS {
        for (j, &n) in ns.iter().enumerate() {
            g.memset(outs[0][j], 0, mmax * n * 2)?;
            g.memset(outs[1][j], 0xFF, mmax * n * 2)?;
        }
        g.synchronize(s)?;
        sep(m, s)?;
        grp(m, s)?;
        g.synchronize(s)?;
        let mut line = Vec::new();
        for (j, &n) in ns.iter().enumerate() {
            let eq = bytes(g, outs[0][j], m * n * 2)? == bytes(g, outs[1][j], m * n * 2)?;
            bitwise &= eq;
            line.push(format!("n {n} {}", if eq { "equal" } else { "DIFFER" }));
        }
        println!(
            "BITWISE {label:<6} K {kk} M={m}: {} ({})",
            if line.iter().all(|l| l.ends_with("equal")) {
                "PASS"
            } else {
                "FAIL"
            },
            line.join(", ")
        );
    }
    let arm_sep = |st: u64| sep(M_TIME, st);
    let arm_grp = |st: u64| grp(M_TIME, st);
    let arms: [&dyn Fn(u64) -> Result<()>; 2] = [&arm_sep, &arm_grp];
    let mut t = [Vec::with_capacity(REPS), Vec::with_capacity(REPS)];
    for rep in 0..WARMUP + REPS {
        for (i, arm) in arms.iter().enumerate() {
            fl.run(g, s)?;
            ev.rec(0, s)?;
            arm(s)?;
            ev.rec(1, s)?;
            let us = ev.us()?;
            if rep >= WARMUP {
                t[i].push(us);
            }
        }
    }
    let [t0, t1] = t;
    let ms = [median(t0) / 1e3, median(t1) / 1e3];
    let wbytes: usize = ns.iter().map(|&n| n * kk / 2 + n * kk / 16).sum();
    let mut void = false;
    let mut line = Vec::new();
    for (name, &x) in ["per-projection", "grouped"].iter().zip(&ms) {
        let rate = wbytes as f64 / (x * 1e-3) / 1e9;
        line.push(format!("{name} {x:.4} ms {rate:.1} GB/s"));
        if rate > GBS_VOID {
            void = true;
            println!("VOID {label} {name} {rate:.1} GB/s > {GBS_VOID} (L2 not cold)");
        }
    }
    println!(
        "TIME {label:<6} M={M_TIME} cold-L2 median of {REPS} ({:.1} MB): {} | grouped/sep {:.3}x",
        wbytes as f64 / 1e6,
        line.join(" | "),
        ms[1] / ms[0]
    );
    for p in outs.iter().flatten().copied().chain(ws).chain([a]) {
        g.free(p)?;
    }
    for q in &qs {
        g.free(q.weight)?;
        g.free(q.weight_scale)?;
    }
    Ok(GroupOut { bitwise, ms, void })
}

fn run() -> Result<i32> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let names = [
        ("w4a16_gemv", "w4a16_gemv"),
        ("quantize_nvfp4", "quantize_bf16_to_nvfp4_mse"),
        ("w4a16_gemv_tc", "w4a16_gemv_tc8"),
        ("w4a16_gemv_tc", "w4a16_gemv_tc16"),
        ("w4a16_gemv_tc", "w4a16_gemv_tc8_group"),
        ("w4a16_gemv_tc", "w4a16_gemv_tc16_group"),
    ];
    let mut missing = 0usize;
    for (module, name) in names {
        if g.kernel(module, name).is_err() {
            println!("MISSING kernel entry {module}::{name}");
            missing += 1;
        }
    }
    if missing > 0 {
        println!("FAIL dflash_nv4_group (kernel entries missing: {missing})");
        return Ok(2);
    }
    let (ev, fl) = (Events::new()?, Flush::new(g)?);
    let mut rng = Rng(0x0064_666e_7634_6701);
    let (mut fails, mut voids, mut round) = (Vec::new(), 0usize, 0f64);
    for &(label, ns, kk, count) in GROUPS {
        let o = run_group(g, &mut rng, (label, ns, kk), (&ev, &fl))?;
        if !o.bitwise {
            fails.push(label);
        }
        voids += usize::from(o.void);
        let saving = count as f64 * (o.ms[0] - o.ms[1]);
        round += saving;
        println!(
            "SAVING {label:<6} x{count}: {saving:+.4} ms per round (per-projection - grouped)"
        );
    }
    fl.free(g)?;
    println!(
        "ROUND grouped saving {round:.4} ms per propose round (both groups x 5 layers, cold L2)"
    );
    let ok = fails.is_empty() && voids == 0 && round >= SAVING_MIN_MS;
    let summary = format!(
        "round saving {round:.4} ms (need >= {SAVING_MIN_MS}); bitwise fail {fails:?}; VOID \
         timings {voids}; M {MS:?}"
    );
    if ok {
        println!("PASS dflash_nv4_group ({summary})");
        Ok(0)
    } else {
        println!("FAIL dflash_nv4_group ({summary})");
        Ok(1)
    }
}

fn main() {
    match run() {
        Ok(0) => {}
        Ok(c) => std::process::exit(c),
        Err(e) => {
            println!("FAIL dflash_nv4_group (error: {e:#})");
            std::process::exit(1);
        }
    }
}
