// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: GPU microtest for `METRALE_GLM_DECODE_L2_PREFETCH` (`glm5next_l2_prefetch`):
//! does pulling a decode GEMV's weights into L2 during a latency-bound window shorten the GEMV,
//! and is the output unchanged?
//!
//! Each trial reproduces one decode window on stream 0: a read-only L2 flush (the M = 1
//! `dense_gemv_bf16` over a 64 MiB dummy weight, so L2 holds clean lines as it does in decode),
//! then the prefetch launch (prefetch arms only), then `glm5next_l2pf_spin_ns` for W us (a
//! stand-in for the all-reduce + mHC/norm chain), then the consuming `dense_gemv_bf16`. CUDA
//! events time the GEMV alone and the whole window. Shapes: a KDA `q_proj`-like 4096 x 4096
//! (32 MiB), a DSA `q_a_proj`-like 1536 x 4096 (12 MiB) and the MoE router 288 x 4096
//! (2.25 MiB), BF16. Budgets {4, 8, 12, 16, 24} MiB x windows {0, 15, 30, 60, 100} us, each
//! against a no-prefetch control at the same window; medians of `REPS` trials.
//!
//! Also printed: the prefetch kernel's own duration for 12 MiB with no window (if it is close
//! to 12 MiB / 240 GB/s = 52 us the bulk prefetch is NOT asynchronous on this part, and the
//! layer placement would serialise instead of overlap; see the doc's risk list).
//!
//! Identity: every prefetch-arm output is compared byte for byte with the control output of its
//! shape, and the weight and input buffers are read back after all trials and compared with the
//! host bytes (the kernel must write nothing). A negative control (one flipped input bit) must
//! change the output, or the comparison is vacuous.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and `PASS` when every comparison is identical, the negative control fires and the
//! gate cell (32 MiB shape, 12 MiB, 60 us) shortens the GEMV by at least 10 %; 3 and
//! `NO-GAIN` when identical but below the gate (keep the lever off); 1 and `FAIL` on any
//! identity failure; 2 when a kernel is absent from this target.
//!
//! Run (the prefetch kernel is in the glm-5.3-flash model directory):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_l2_prefetch_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_layer::prefetch::{
    L2_PREFETCH_KERNEL, L2_PREFETCH_MODULE, L2Span, launch_l2_prefetch,
};

// 2026-10-01: CUDA driver event API, declared as in `dsa_indexer_tiled_bitparity_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const STREAM: u64 = 0;
/// 2026-10-01: Hidden size and the flush weight's rows: 8192 x 4096 BF16 = 64 MiB, over twice
/// the 24 MB L2.
const K: usize = 4096;
const FLUSH_N: usize = 8192;
/// 2026-10-01: (N, label) of the consuming `[N, 4096]` BF16 GEMV.
const SHAPES: &[(usize, &str)] = &[
    (4096, "KDA q_proj-like 32 MiB"),
    (1536, "DSA q_a_proj-like 12 MiB"),
    (288, "MoE router 2.25 MiB"),
];
const BUDGETS_MIB: &[usize] = &[4, 8, 12, 16, 24];
const WINDOWS_US: &[u64] = &[0, 15, 30, 60, 100];
const REPS: usize = 11;
/// 2026-10-01: GB10 STREAM, 237-240 GB/s measured 2026-09-22 (kernel-engineering ledger); used
/// only for the printed ideal.
const STREAM_GBPS: f64 = 240.0;
/// 2026-10-01: Gate cell: shape index, budget MiB, window us, least GEMV-time gain.
const GATE: (usize, usize, u64, f64) = (0, 12, 60, 0.10);

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn bf16_bytes(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                let v = lo + (hi - lo) * self.f();
                bf16::from_f32(v).to_bits().to_le_bytes()
            })
            .collect()
    }
}

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(STREAM)?;
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-01: Three CUDA events on stream 0: start, before the GEMV, after it.
struct Events([u64; 3]);

impl Events {
    fn new() -> Result<Self> {
        let mut e = [0u64; 3];
        for x in e.iter_mut() {
            // SAFETY: plain CUDA driver call writing one event handle; the backend made the
            // context current when it was built.
            check(unsafe { cuEventCreate(x, 0) }, "cuEventCreate")?;
        }
        Ok(Self(e))
    }

    fn rec(&self, i: usize) -> Result<()> {
        // SAFETY: records an event this struct created on the default stream.
        check(unsafe { cuEventRecord(self.0[i], STREAM) }, "cuEventRecord")
    }

    /// 2026-10-01: Microseconds from event `a` to event `b`, after `b` completes.
    fn us(&self, a: usize, b: usize) -> Result<f64> {
        let mut ms: f32 = 0.0;
        // SAFETY: both events were created and recorded by this struct; `ms` outlives the call.
        unsafe {
            check(cuEventSynchronize(self.0[b]), "cuEventSynchronize")?;
            check(
                cuEventElapsedTime(&mut ms, self.0[a], self.0[b]),
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

struct Kernels {
    prefetch: KernelHandle,
    spin: KernelHandle,
    gemv: KernelHandle,
}

/// 2026-10-01: The M = 1 `dense_gemv_bf16` launch `ops::dense_mm_bf16` issues (4 outputs per
/// 256-thread block): `y[n] = sum_k x[k] * w[n, k]`.
fn gemv(
    g: &dyn GpuBackend,
    ks: &Kernels,
    x: DevicePtr,
    w: DevicePtr,
    y: DevicePtr,
    n: usize,
) -> Result<()> {
    KernelLaunch::new(g, ks.gemv)
        .grid([(n as u32).div_ceil(4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(y)
        .arg_u32(n as u32)
        .arg_u32(K as u32)
        .launch(STREAM)
}

fn spin(g: &dyn GpuBackend, ks: &Kernels, us: u64) -> Result<()> {
    if us == 0 {
        return Ok(());
    }
    KernelLaunch::new(g, ks.spin)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_u64(us * 1000)
        .launch(STREAM)
}

/// 2026-10-01: Device buffers of one shape plus the shared flush weight.
struct Bufs {
    x: DevicePtr,
    w: DevicePtr,
    y: DevicePtr,
    flush_w: DevicePtr,
    flush_y: DevicePtr,
}

/// 2026-10-01: One window: flush, [prefetch `budget` bytes of `w`], spin `window_us`, GEMV.
/// Returns (GEMV us, whole-window us).
fn trial(
    g: &dyn GpuBackend,
    ks: &Kernels,
    ev: &Events,
    b: &Bufs,
    n: usize,
    budget: usize,
    window_us: u64,
) -> Result<(f64, f64)> {
    gemv(g, ks, b.x, b.flush_w, b.flush_y, FLUSH_N)?;
    ev.rec(0)?;
    if budget > 0 {
        launch_l2_prefetch(
            g,
            ks.prefetch,
            &[L2Span {
                ptr: b.w,
                bytes: n * K * 2,
            }],
            budget,
            STREAM,
        )?;
    }
    spin(g, ks, window_us)?;
    ev.rec(1)?;
    gemv(g, ks, b.x, b.w, b.y, n)?;
    ev.rec(2)?;
    Ok((ev.us(1, 2)?, ev.us(0, 2)?))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let lookups = [
        (L2_PREFETCH_MODULE, L2_PREFETCH_KERNEL),
        (L2_PREFETCH_MODULE, "glm5next_l2pf_spin_ns"),
        ("gemv", "dense_gemv_bf16"),
    ];
    let mut h = Vec::new();
    for (module, func) in lookups {
        match g.kernel(module, func) {
            Ok(k) => h.push(k),
            Err(e) => {
                println!("{module}::{func} absent from this target ({e}) - SKIP");
                std::process::exit(2);
            }
        }
    }
    let ks = Kernels {
        prefetch: h[0],
        spin: h[1],
        gemv: h[2],
    };
    let ev = Events::new()?;
    let mut rng = Lcg(0x6c32_7066_2026_1001);

    // 2026-10-01: Async check: the prefetch kernel alone, 12 MiB, no window.
    {
        let w = up(g, &rng.bf16_bytes(12 << 19, -0.05, 0.05))?;
        let mut t = Vec::new();
        for _ in 0..REPS {
            ev.rec(0)?;
            launch_l2_prefetch(
                g,
                ks.prefetch,
                &[L2Span {
                    ptr: w,
                    bytes: 12 << 20,
                }],
                12 << 20,
                STREAM,
            )?;
            ev.rec(2)?;
            t.push(ev.us(0, 2)?);
        }
        let us = median(t);
        let stream_us = (12u64 << 20) as f64 / (STREAM_GBPS * 1e3);
        println!(
            "prefetch kernel alone, 12 MiB: {us:.1} us (12 MiB at STREAM = {stream_us:.1} us) -> \
             {}",
            if us < 0.25 * stream_us {
                "ASYNC (the issuing kernel does not wait for the bytes)"
            } else {
                "NOT ASYNC: the kernel waits for its prefetches; the layer placement would \
                 serialise, not overlap"
            }
        );
        g.free(w)?;
    }

    let x_host = rng.bf16_bytes(K, -1.0, 1.0);
    let flush_w = up(g, &rng.bf16_bytes(FLUSH_N * K, -0.05, 0.05))?;
    let flush_y = g.alloc(FLUSH_N * 2)?;
    let mut identity_ok = true;
    let mut negative_ok = true;
    let mut gate_gain = f64::NAN;

    for (si, &(n, label)) in SHAPES.iter().enumerate() {
        let w_host = rng.bf16_bytes(n * K, -0.05, 0.05);
        let b = Bufs {
            x: up(g, &x_host)?,
            w: up(g, &w_host)?,
            y: g.alloc(n * 2)?,
            flush_w,
            flush_y,
        };
        let mib_w = (n * K * 2) as f64 / (1u64 << 20) as f64;
        println!("\n== {label}: N {n} K {K} ({mib_w:.2} MiB) ==");

        // 2026-10-01: Control output and per-window control timings.
        trial(g, &ks, &ev, &b, n, 0, 0)?;
        let y_ref = down(g, b.y, n * 2)?;
        let mut ctl = Vec::new();
        for &win in WINDOWS_US {
            let mut gt = Vec::new();
            let mut tt = Vec::new();
            for _ in 0..REPS {
                let (a, t) = trial(g, &ks, &ev, &b, n, 0, win)?;
                gt.push(a);
                tt.push(t);
            }
            ctl.push((median(gt), median(tt)));
        }

        println!(
            "budget_MiB window_us  gemv_us(ctl->pf)  window_total_us(ctl->pf)  saved_us  ideal_us"
        );
        for &mib in BUDGETS_MIB {
            for (wi, &win) in WINDOWS_US.iter().enumerate() {
                let mut gt = Vec::new();
                let mut tt = Vec::new();
                for _ in 0..REPS {
                    let (a, t) = trial(g, &ks, &ev, &b, n, mib << 20, win)?;
                    gt.push(a);
                    tt.push(t);
                    if down(g, b.y, n * 2)? != y_ref {
                        identity_ok = false;
                    }
                }
                let (cg, ct) = ctl[wi];
                let (pg, pt) = (median(gt), median(tt));
                let pf_bytes = (mib << 20).min(n * K * 2) as f64;
                let ideal = (pf_bytes / (STREAM_GBPS * 1e3)).min(win as f64);
                println!(
                    "{mib:>10} {win:>9}  {cg:>7.1} -> {pg:>7.1}  {ct:>10.1} -> {pt:>10.1}  {:>8.1}  {ideal:>8.1}",
                    cg - pg
                );
                if (si, mib, win) == (GATE.0, GATE.1, GATE.2) {
                    gate_gain = (cg - pg) / cg;
                }
            }
        }

        // 2026-10-01: The kernel writes nothing: weight and input unchanged.
        if down(g, b.w, n * K * 2)? != w_host || down(g, b.x, K * 2)? != x_host {
            identity_ok = false;
            println!("FAIL: a weight or input byte changed");
        }
        // 2026-10-01: Negative control: flip the top mantissa bit of x[0] (byte 1, bit 6 of the
        // little-endian BF16), a ~1.5x change of that input, which must change y.
        let mut x_flip = x_host.clone();
        x_flip[1] ^= 0x40;
        g.copy_h2d(&x_flip, b.x)?;
        trial(g, &ks, &ev, &b, n, 0, 0)?;
        if down(g, b.y, n * 2)? == y_ref {
            negative_ok = false;
            println!("negative control did NOT change the output for {label}");
        }
        for p in [b.x, b.w, b.y] {
            g.free(p)?;
        }
    }

    println!(
        "\ngate cell (32 MiB shape, {} MiB, {} us): GEMV gain {:.1} % (need {:.0} %)",
        GATE.1,
        GATE.2,
        gate_gain * 100.0,
        GATE.3 * 100.0
    );
    if !identity_ok {
        println!("FAIL - an output, weight or input byte differs with the prefetch on.");
        std::process::exit(1);
    }
    if !negative_ok {
        println!("FAIL - a negative control did not fire; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if gate_gain.is_nan() || gate_gain < GATE.3 {
        println!(
            "NO-GAIN - byte-identical, but the gate cell is below the gain bar; keep the lever off."
        );
        std::process::exit(3);
    }
    println!("PASS - byte-identical, negative control fires, gate cell gain met.");
    Ok(())
}
