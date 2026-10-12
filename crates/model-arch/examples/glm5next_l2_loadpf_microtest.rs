// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: GPU microtest for `METRALE_GLM_DECODE_L2_LOADPF` (`glm5next_l2_loadpf`): does a
//! side-stream REAL-LOAD prefetch, run during a latency-bound window on the main stream, leave
//! an NVFP4 GEMV's weights resident in the 24 MB GB10 L2, and is the output unchanged?
//!
//! Shape: one decode GEMV, M = 1, N 4096 x K 4096 NVFP4 (packed 8 MiB + E4M3 scales 1 MiB: the
//! KDA `o_proj` per rank under TP2), quantized from BF16 by the dense lever's quantizer
//! (`dense_fp8::quantize_nvfp4_copy`) and run by `dense_fp8::nv4_gemv_uncounted` (the decode
//! route's NVFP4 arm). Each trial starts by flushing L2 with WRITES of a 64 MiB scratch buffer
//! (`memset_async`: dirty lines, as the decode step's activations leave them), then:
//!
//! * cold: the GEMV alone;
//! * control: a 90 us spin on the main stream (`glm5next_l2pf_spin_ns`, a stand-in for the KDA
//!   recurrence), then the GEMV;
//! * hot: fork (event on main, side waits), `glm5next_l2_loadpf` over the packed bytes and the
//!   scales on the side stream concurrently with the 90 us spin on main, join (main waits the
//!   side's event), then the GEMV. Swept over CTAs {2, 4, 8, 16, 48}.
//!
//! CUDA events on the main stream time the GEMV alone and the whole window (a window above
//! ~90 us means the load-prefetch outlasted the spin and the join waited). Identity: every
//! hot-arm output equals the cold output byte for byte, and the packed weight and scales are
//! unchanged after all trials (the kernel writes nothing).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and a line starting `PASS` when identical and cold/hot GEMV time >= 1.5 at the
//! default 8 CTAs; 1 and `FAIL` otherwise (the line says which); 2 when a kernel is absent.
//!
//! Run (both kernels are in the glm-5.3-flash model directory):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_l2_loadpf_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_layer::L2Span;
use metrale_model_arch::glm5next_layer::dense_fp8::{nv4_gemv_uncounted, quantize_nvfp4_copy};
use metrale_model_arch::glm5next_layer::loadpf::{
    L2_LOADPF_CTAS, L2_LOADPF_KERNEL, L2_LOADPF_MODULE, launch_l2_loadpf,
};
use metrale_model_arch::glm5next_layer::prefetch::L2_PREFETCH_MODULE;

// 2026-10-11: CUDA driver event API with timing, as in `glm5next_l2_prefetch_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const N: usize = 4096;
const K: usize = 4096;
/// 2026-10-11: 64 MiB of writes, over twice the 24 MB L2.
const FLUSH_BYTES: usize = 64 << 20;
const SPIN_US: u64 = 90;
const CTAS: &[u32] = &[2, 4, 8, 16, 48];
const REPS: usize = 15;
const GATE_SPEEDUP: f64 = 1.5;

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

/// 2026-10-11: Three timing events on the main stream: window start, before the GEMV, after.
struct Events {
    e: [u64; 3],
    stream: u64,
}

impl Events {
    fn new(stream: u64) -> Result<Self> {
        let mut e = [0u64; 3];
        for x in e.iter_mut() {
            // SAFETY: plain CUDA driver call writing one event handle; the backend made the
            // context current when it was built.
            check(unsafe { cuEventCreate(x, 0) }, "cuEventCreate")?;
        }
        Ok(Self { e, stream })
    }

    fn rec(&self, i: usize) -> Result<()> {
        // SAFETY: records an event this struct created on a stream the backend created.
        check(
            unsafe { cuEventRecord(self.e[i], self.stream) },
            "cuEventRecord",
        )
    }

    /// 2026-10-11: Microseconds from event `a` to event `b`, after `b` completes.
    fn us(&self, a: usize, b: usize) -> Result<f64> {
        let mut ms: f32 = 0.0;
        // SAFETY: both events were created and recorded by this struct; `ms` outlives the call.
        unsafe {
            check(cuEventSynchronize(self.e[b]), "cuEventSynchronize")?;
            check(
                cuEventElapsedTime(&mut ms, self.e[a], self.e[b]),
                "cuEventElapsedTime",
            )?;
        }
        Ok(ms as f64 * 1e3)
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        for &e in &self.e {
            // SAFETY: destroys an event this struct created; errors are ignored on drop.
            unsafe {
                cuEventDestroy_v2(e);
            }
        }
    }
}

/// 2026-10-11: Everything one trial touches.
struct Rig<'a> {
    g: &'a dyn GpuBackend,
    loadpf: KernelHandle,
    spin: KernelHandle,
    main: u64,
    side: u64,
    ev_fork: u64,
    ev_join: u64,
    ev: Events,
    flush: DevicePtr,
    x: DevicePtr,
    y: DevicePtr,
    q: metrale_model_layers::weight_map::QuantizedWeight,
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Cold,
    Control,
    Hot(u32),
}

impl Rig<'_> {
    fn spans(&self) -> [L2Span; 2] {
        [
            L2Span {
                ptr: self.q.weight,
                bytes: N * K / 2,
            },
            L2Span {
                ptr: self.q.weight_scale,
                bytes: N * K / 16,
            },
        ]
    }

    fn spin_us(&self, us: u64) -> Result<()> {
        KernelLaunch::new(self.g, self.spin)
            .grid([1, 1, 1])
            .block([32, 1, 1])
            .arg_u64(us * 1000)
            .launch(self.main)
    }

    /// 2026-10-11: One trial; returns (GEMV us, whole-window us).
    fn trial(&self, arm: Arm) -> Result<(f64, f64)> {
        let g = self.g;
        g.memset_async(self.flush, 0, FLUSH_BYTES, self.main)?;
        self.ev.rec(0)?;
        match arm {
            Arm::Cold => {}
            Arm::Control => self.spin_us(SPIN_US)?,
            Arm::Hot(ctas) => {
                g.record_event(self.ev_fork, self.main)?;
                g.stream_wait_event(self.side, self.ev_fork)?;
                launch_l2_loadpf(g, self.loadpf, &self.spans(), ctas, self.side)?;
                g.record_event(self.ev_join, self.side)?;
                self.spin_us(SPIN_US)?;
                g.stream_wait_event(self.main, self.ev_join)?;
            }
        }
        self.ev.rec(1)?;
        nv4_gemv_uncounted(g, self.x, &self.q, self.y, 1, N, K, self.main)?;
        self.ev.rec(2)?;
        Ok((self.ev.us(1, 2)?, self.ev.us(0, 2)?))
    }

    fn y_bytes(&self) -> Result<Vec<u8>> {
        self.g.synchronize(self.main)?;
        let mut b = vec![0u8; N * 2];
        self.g.copy_d2h(self.y, &mut b)?;
        Ok(b)
    }

    fn weight_bytes(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        self.g.synchronize(self.main)?;
        self.g.synchronize(self.side)?;
        let mut p = vec![0u8; N * K / 2];
        let mut s = vec![0u8; N * K / 16];
        self.g.copy_d2h(self.q.weight, &mut p)?;
        self.g.copy_d2h(self.q.weight_scale, &mut s)?;
        Ok((p, s))
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// 2026-10-11: Medians of `REPS` trials of `arm`, plus whether every output equals `want`.
fn arm_medians(rig: &Rig<'_>, arm: Arm, want: &[u8]) -> Result<(f64, f64, bool)> {
    let (mut gt, mut wt, mut same) = (Vec::new(), Vec::new(), true);
    for _ in 0..REPS {
        let (a, w) = rig.trial(arm)?;
        gt.push(a);
        wt.push(w);
        same &= rig.y_bytes()? == want;
    }
    Ok((median(gt), median(wt), same))
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut h = Vec::new();
    for (module, func) in [
        (L2_LOADPF_MODULE, L2_LOADPF_KERNEL),
        (L2_PREFETCH_MODULE, "glm5next_l2pf_spin_ns"),
    ] {
        match g.kernel(module, func) {
            Ok(k) => h.push(k),
            Err(e) => {
                println!("{module}::{func} absent from this target ({e}) - SKIP");
                std::process::exit(2);
            }
        }
    }
    let main_stream = g.create_stream()?;
    let mut rng = Lcg(0x6c70_6620_2026_1011);
    let w_bf16 = g.alloc(N * K * 2)?;
    g.copy_h2d(&rng.bf16_bytes(N * K, -0.05, 0.05), w_bf16)?;
    let Some(q) = quantize_nvfp4_copy(g, w_bf16, N, K, main_stream)? else {
        println!("NVFP4 quantizer or GEMV kernels absent from this target - SKIP");
        std::process::exit(2);
    };
    g.free(w_bf16)?;
    let x = g.alloc(K * 2)?;
    g.copy_h2d(&rng.bf16_bytes(K, -1.0, 1.0), x)?;
    let rig = Rig {
        g,
        loadpf: h[0],
        spin: h[1],
        main: main_stream,
        side: g.create_stream()?,
        ev_fork: g.create_event()?,
        ev_join: g.create_event()?,
        ev: Events::new(main_stream)?,
        flush: g.alloc(FLUSH_BYTES)?,
        x,
        y: g.alloc(N * 2)?,
        q,
    };
    let before = rig.weight_bytes()?;

    // 2026-10-11: Warm-up (module load, clocks), then the reference output.
    for _ in 0..3 {
        rig.trial(Arm::Cold)?;
    }
    let y_ref = rig.y_bytes()?;
    let (cold, cold_w, _) = arm_medians(&rig, Arm::Cold, &y_ref)?;
    let (ctl, ctl_w, _) = arm_medians(&rig, Arm::Control, &y_ref)?;
    println!(
        "NVFP4 GEMV M=1 N={N} K={K} ({:.2} MiB packed+scales), L2 flushed by {} MiB of writes, \
         spin {SPIN_US} us, {REPS} reps (medians)",
        (N * K / 2 + N * K / 16) as f64 / (1u64 << 20) as f64,
        FLUSH_BYTES >> 20
    );
    println!("arm           gemv_us  window_us  cold/arm");
    println!("cold        {cold:>8.1}  {cold_w:>9.1}  {:>8.2}", 1.0);
    println!("control     {ctl:>8.1}  {ctl_w:>9.1}  {:>8.2}", cold / ctl);
    let mut identical = true;
    let mut gate = f64::NAN;
    for &c in CTAS {
        let (hot, hot_w, same) = arm_medians(&rig, Arm::Hot(c), &y_ref)?;
        identical &= same;
        let speedup = cold / hot;
        println!(
            "hot ctas={c:<3} {hot:>8.1}  {hot_w:>9.1}  {speedup:>8.2}{}",
            if hot_w > SPIN_US as f64 + hot + 5.0 {
                "  (join waited: the load-prefetch outlasted the spin)"
            } else {
                ""
            }
        );
        if c == L2_LOADPF_CTAS {
            gate = speedup;
        }
    }
    if rig.weight_bytes()? != before {
        identical = false;
        println!("a packed-weight or scale byte changed");
    }
    println!(
        "gate: cold/hot at {L2_LOADPF_CTAS} CTAs = {gate:.2} (need >= {GATE_SPEEDUP}); \
         outputs and weights identical: {identical}"
    );
    if !identical {
        println!("FAIL - an output or weight byte differs with the load-prefetch on.");
        std::process::exit(1);
    }
    if gate.is_nan() || gate < GATE_SPEEDUP {
        println!(
            "FAIL - byte-identical, but cold/hot {gate:.2} < {GATE_SPEEDUP}: the load-prefetch \
             does not leave the GEMV's weights resident enough."
        );
        std::process::exit(1);
    }
    println!(
        "PASS - byte-identical; cold {cold:.1} us -> hot {:.1} us (x{gate:.2}) at \
         {L2_LOADPF_CTAS} CTAs.",
        cold / gate
    );
    Ok(())
}
