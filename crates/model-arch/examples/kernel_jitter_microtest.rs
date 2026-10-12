// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Does host-side DRAM traffic widen the duration spread of memory-bound decode
//! kernels on GB10's unified memory?
//!
//! Owner: race-memory. Preregistration: SeedHQ/seed-skills#80, comment 6126370952 (arms,
//! counts, metrics and grade bands below are fixed there; change none of them here).
//!
//! What it does: one process, one node, no host settings. Each iteration replays a C = 1
//! decode per-layer slice at the car's shapes, on the default stream, bracketed by two CUDA
//! events: three NVFP4 dense GEMVs at M = 3 rows (`df::nv4_gemv`: 4096x4096 `kda_qkvo`,
//! 1024x4096 `shexp_gate_up`, 4096x1024 `shexp_down`, shapes from
//! `glm5next_dense_nvfp4_microtest.rs`) and one routed-MoE batchm layer (R = 3 rows, union of
//! U = 18 experts, hidden 4096, expert-TP half width mi = 1024: the car `_m3_c8` union sweep for
//! gate, up and down, launched as in `glm5next_moe_fixcost_microtest.rs`). Weights rotate over
//! pools far larger than L2 (dense: 12 distinct copies per shape; MoE: 288 distinct experts per
//! matrix, a fresh U = 18 routing per iteration) so every kernel streams from DRAM.
//! Host load: k threads in this process, each looping `copy_from_slice` between two pre-touched
//! 1 GiB `Vec<u8>` buffers until an `AtomicBool` stops them. Arms, fixed order: Q0a (k = 0),
//! H2 (k = 2), H8 (k = 8), Q0b (k = 0); 3000 timed iterations each after 200 uncounted ones
//! (`JITTER_ITERS` overrides the count for a local smoke run only). Each arm starts its threads,
//! waits 1 s, runs the warm-up, times, then stops and joins.
//!
//! Output: one `ARM` line per arm, `VERDICT hostdram <VOID|KILL|GO|GREY>` and a final `PASS`
//! line when the instrument ran (all four arms completed with n >= 2500 and no CUDA error).
//! The verdict is data, not a pass/fail of the code. `JITTER_CSV=<path>` also writes every raw
//! duration (`arm,k,iter,us`). The durations are device time between the two events, read after
//! one sync at the end of each arm (the host runs ahead, so host launch latency is not in them
//! unless the host falls more than the launch queue behind).
//!
//! Grade (checked in this order): VOID if any arm n < 2500 or |median(Q0a) / median(Q0b) - 1|
//! > 0.05; KILL if quiet S = mean(S(Q0a), S(Q0b)) >= 1.15; GO if quiet S <= 1.06 and
//! S(H8) >= 1.15; else GREY. S = p90 / p10 of the per-iteration durations.
//!
//! What it does NOT prove: that the engine's own host traffic is the jitter source (the load
//! here is a synthetic memcpy, not the serve's host work); anything about C > 1; per-layer
//! burst structure or a partner rank (a 2-rank run adds waits this single-process slice has no
//! analogue of). `memcpy_GBps` is bytes copied per second (each byte is also read once, so DRAM
//! traffic is about twice that).
//!
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example kernel_jitter_microtest

use anyhow::{Result, bail, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_layer::dense_fp8 as df;
use metrale_model_layers::weight_map::{DenseWeight, QuantizedWeight};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

// CUDA driver event API with timing enabled, declared as in `kda_snap_fuse_microtest.rs` (the
// runtime's own events are created with `CU_EVENT_DISABLE_TIMING`).
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const HIDDEN: usize = 4096;
const MI: usize = 1024;
const TOP_K: usize = 8;
const NUM_EXPERTS: usize = 288;
/// Rows per decode step (the C = 1 MTP verify width) and the routed-MoE union size.
const ROWS: usize = 3;
const UNION: usize = 18;
/// Columns per CTA of the car `_c8` entry (8 warps x 8 columns).
const CAR_COLS: usize = 64;
/// Distinct routings cycled through (one per iteration).
const ROUTINGS: usize = 64;
/// Distinct NVFP4 copies per dense shape; 12 x 9.4 MB for the largest, far above L2.
const DENSE_POOL: usize = 12;
/// (n, k, label), the per-rank GLM-5.3 decode GEMV shapes with the most launches per step.
const DENSE_SHAPES: [(usize, usize, &str); 3] = [
    (4096, 4096, "kda_qkvo"),
    (1024, 4096, "shexp_gate_up"),
    (4096, 1024, "shexp_down"),
];
/// NVFP4 bytes per weight: 4-bit value plus one E4M3 scale per 16.
const BYTES_PER_W: f64 = 0.5 + 1.0 / 16.0;

const WARMUP: usize = 200;
const DEFAULT_ITERS: usize = 3000;
const MIN_N: usize = 2500;
const SETTLE: Duration = Duration::from_secs(1);
const BUF_BYTES: usize = 1 << 30;
const CHUNK_BYTES: usize = 64 << 20;
const MAX_K: usize = 8;
/// (name, k) in the fixed order.
const ARMS: [(&str, usize); 4] = [("Q0a", 0), ("H2", 2), ("H8", 8), ("Q0b", 0)];

const MAX_DRIFT: f64 = 0.05;
const KILL_S: f64 = 1.15;
const GO_QUIET_S: f64 = 1.06;
const GO_H8_S: f64 = 1.15;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn u(&mut self) -> f32 {
        (self.next() & 0xFF_FFFF) as f32 / 16_777_216.0
    }
    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn le<T: Copy, const B: usize>(v: &[T], f: impl Fn(T) -> [u8; B]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}

fn bf16_bytes(rng: &mut Rng, n: usize, scale: f32) -> Vec<u8> {
    let v: Vec<bf16> = (0..n)
        .map(|_| bf16::from_f32((rng.u() * 2.0 - 1.0) * scale))
        .collect();
    le(&v, |x| x.to_bits().to_le_bytes())
}

/// Host NVFP4 bytes for an [n, k] expert (random nibbles; E4M3 scale codes 1.0..1.875, never
/// zero, inf or NaN) with 64 spare bytes so each expert's upload starts at its own offset.
struct Host {
    w: Vec<u8>,
    s: Vec<u8>,
    wb: usize,
    sb: usize,
}

fn gen_host(rng: &mut Rng, n: usize, k: usize) -> Host {
    let (wb, sb) = (n * k / 2, n * k / 16);
    Host {
        w: (0..wb + 64).map(|_| rng.next() as u8).collect(),
        s: (0..sb + 64)
            .map(|_| 0x38 | (rng.next() as u8 & 7))
            .collect(),
        wb,
        sb,
    }
}

/// The 288-entry pointer table of one matrix, every entry its own device buffer.
struct Table {
    packed: DevicePtr,
    scale: DevicePtr,
    scale2: DevicePtr,
}

fn make_table(g: &dyn GpuBackend, h: &Host) -> Result<Table> {
    let (mut pp, mut sp, mut s2) = (Vec::new(), Vec::new(), Vec::new());
    for e in 0..NUM_EXPERTS {
        let off = (e * 17) % 64;
        pp.push(up(g, &h.w[off..off + h.wb])?.0);
        sp.push(up(g, &h.s[off..off + h.sb])?.0);
        s2.push(1.0f32 + e as f32 * 0.01);
    }
    Ok(Table {
        packed: up(g, &le(&pp, |x: u64| x.to_le_bytes()))?,
        scale: up(g, &le(&sp, |x: u64| x.to_le_bytes()))?,
        scale2: up(g, &le(&s2, |x: f32| x.to_le_bytes()))?,
    })
}

/// Union tables from `glm5next_moe_row_union` (one block of rows * top_k threads, as
/// `forward/moe_experts.rs` launches it); `union` = live entries.
struct Routing {
    ueid: DevicePtr,
    uslot: DevicePtr,
    union: usize,
}

fn build_routing(g: &dyn GpuBackend, k_union: KernelHandle, ids: &[Vec<i32>]) -> Result<Routing> {
    let rows = ids.len();
    let flat: Vec<i32> = ids.iter().flatten().copied().collect();
    let d_ids = up(g, &le(&flat, |x: i32| x.to_le_bytes()))?;
    let ueid = g.alloc(rows * TOP_K * 4)?;
    let uslot = g.alloc(rows * TOP_K * rows * 4)?;
    KernelLaunch::new(g, k_union)
        .grid([1, 1, 1])
        .block([(rows * TOP_K) as u32, 1, 1])
        .arg_ptr(d_ids)
        .arg_ptr(ueid)
        .arg_ptr(uslot)
        .arg_u32(rows as u32)
        .arg_u32(TOP_K as u32)
        .launch(g.default_stream())?;
    g.synchronize(g.default_stream())?;
    g.free(d_ids)?;
    let mut d: Vec<i32> = flat.into_iter().filter(|&e| e >= 0).collect();
    d.sort_unstable();
    d.dedup();
    Ok(Routing {
        ueid,
        uslot,
        union: d.len(),
    })
}

/// `rows` rows of TOP_K experts each whose union has exactly `u` experts (8 <= u <= rows * 8):
/// the first `u` row-major slots take `u` distinct random experts, each later slot one its row
/// lacks (as `union_ids` in `glm5next_moe_fixcost_microtest.rs`).
fn union_ids(rng: &mut Rng, rows: usize, u: usize) -> Vec<Vec<i32>> {
    let mut pool: Vec<i32> = (0..NUM_EXPERTS as i32).collect();
    for i in 0..u {
        let j = i + rng.below(NUM_EXPERTS - i);
        pool.swap(i, j);
    }
    pool.truncate(u);
    let mut ids = vec![Vec::new(); rows];
    for p in 0..rows * TOP_K {
        let row = &mut ids[p / TOP_K];
        let e = if p < u {
            pool[p]
        } else {
            let free: Vec<i32> = pool.iter().copied().filter(|e| !row.contains(e)).collect();
            free[rng.below(free.len())]
        };
        row.push(e);
    }
    ids
}

/// One union sweep of the car `_m<R>_c8` entry; `down` selects the per-(row, slot) input layout
/// (as `launch` in `glm5next_moe_fixcost_microtest.rs`).
#[allow(clippy::too_many_arguments)]
fn launch_moe(
    g: &dyn GpuBackend,
    h: KernelHandle,
    a: DevicePtr,
    t: &Table,
    c: DevicePtr,
    rt: &Routing,
    n: usize,
    kk: usize,
    down: bool,
    s: u64,
) -> Result<()> {
    let (ars, ass) = if down { (TOP_K * MI, MI) } else { (kk, 0) };
    KernelLaunch::new(g, h)
        .grid([n.div_ceil(CAR_COLS) as u32, (ROWS * TOP_K) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2)
        .arg_ptr(c)
        .arg_ptr(rt.ueid)
        .arg_ptr(rt.uslot)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(ars as u32)
        .arg_u32(ass as u32)
        .arg_u32((TOP_K * n) as u32)
        .launch(s)
}

/// Everything one slice replay touches.
struct Rig<'a> {
    g: &'a dyn GpuBackend,
    s: u64,
    car: KernelHandle,
    gate: Table,
    upm: Table,
    down: Table,
    routs: Vec<Routing>,
    x: DevicePtr,
    act: DevicePtr,
    c_gu: DevicePtr,
    c_dn: DevicePtr,
    dense: Vec<Vec<QuantizedWeight>>,
    d_a: DevicePtr,
    d_c: DevicePtr,
}

impl Rig<'_> {
    /// One decode per-layer slice for iteration `i`: three dense NVFP4 GEMVs at M = 3, then the
    /// MoE layer's gate, up and down union sweeps.
    fn slice(&self, i: usize) -> Result<()> {
        let (g, s) = (self.g, self.s);
        for (si, &(n, k, _)) in DENSE_SHAPES.iter().enumerate() {
            df::nv4_gemv(
                g,
                self.d_a,
                &self.dense[si][i % DENSE_POOL],
                self.d_c,
                ROWS,
                n,
                k,
                s,
            )?;
        }
        let rt = &self.routs[i % ROUTINGS];
        launch_moe(
            g, self.car, self.x, &self.gate, self.c_gu, rt, MI, HIDDEN, false, s,
        )?;
        launch_moe(
            g, self.car, self.x, &self.upm, self.c_gu, rt, MI, HIDDEN, false, s,
        )?;
        launch_moe(
            g, self.car, self.act, &self.down, self.c_dn, rt, HIDDEN, MI, true, s,
        )
    }
}

fn build_rig<'a>(g: &'a dyn GpuBackend, rng: &mut Rng) -> Result<Rig<'a>> {
    let s = g.default_stream();
    let car = g.kernel(
        "w4a16_gemv",
        &format!("w4a16_gemv_sw_moe_batchm_m{ROWS}_c8"),
    )?;
    let k_union = g.kernel("w4a16_gemv", "glm5next_moe_row_union")?;
    let (hgu, hdn) = (gen_host(rng, MI, HIDDEN), gen_host(rng, HIDDEN, MI));
    let gate = make_table(g, &hgu)?;
    let upm = make_table(g, &hgu)?;
    let down = make_table(g, &hdn)?;
    let routs = (0..ROUTINGS)
        .map(|_| build_routing(g, k_union, &union_ids(rng, ROWS, UNION)))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        routs.iter().all(|r| r.union == UNION),
        "routing union is not {UNION}"
    );
    let x = up(g, &bf16_bytes(rng, ROWS * HIDDEN, 1.0))?;
    let act = up(g, &bf16_bytes(rng, ROWS * TOP_K * MI, 1.0))?;
    let c_gu = g.alloc(ROWS * TOP_K * MI * 2)?;
    let c_dn = g.alloc(ROWS * TOP_K * HIDDEN * 2)?;
    // Dense pools: DENSE_POOL separately quantized copies per shape, as the dense microtest's
    // timing pool builds them.
    let (aq, qq) = (
        g.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        g.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_mse")?,
    );
    let mut dense = Vec::new();
    for &(n, k, _) in &DENSE_SHAPES {
        let tmp = up(g, &bf16_bytes(rng, n * k, 0.04))?;
        let dw = DenseWeight { weight: tmp };
        let mut pool = Vec::with_capacity(DENSE_POOL);
        for _ in 0..DENSE_POOL {
            pool.push(metrale_model_layers::weight_map::quantize_to_nvfp4(
                &dw, n, k, g, aq, qq, s,
            )?);
        }
        g.synchronize(s)?;
        g.free(tmp)?;
        dense.push(pool);
    }
    let d_a = up(g, &bf16_bytes(rng, ROWS * 4096, 1.0))?;
    let d_c = g.alloc(ROWS * 4096 * 2)?;
    g.synchronize(s)?;
    Ok(Rig {
        g,
        s,
        car,
        gate,
        upm,
        down,
        routs,
        x,
        act,
        c_gu,
        c_dn,
        dense,
        d_a,
        d_c,
    })
}

/// NVFP4 weight bytes one iteration streams: dense shapes plus 3 matrices x UNION experts.
fn bytes_per_iter() -> (f64, f64) {
    let dense: f64 = DENSE_SHAPES
        .iter()
        .map(|&(n, k, _)| (n * k) as f64 * BYTES_PER_W)
        .sum();
    let moe = (3 * UNION * MI * HIDDEN) as f64 * BYTES_PER_W;
    (dense, moe)
}

fn ck(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what}: CUDA status {rc}");
    }
    Ok(())
}

struct HostBufs {
    src: Vec<u8>,
    dst: Vec<u8>,
}

/// Loops memcpy src -> dst in 64 MiB chunks, counting copied bytes, until `stop`.
fn copier(b: &mut HostBufs, stop: &AtomicBool, bytes: &AtomicU64) {
    while !stop.load(Ordering::Relaxed) {
        for off in (0..BUF_BYTES).step_by(CHUNK_BYTES) {
            let r = off..off + CHUNK_BYTES;
            b.dst[r.clone()].copy_from_slice(&b.src[r]);
            bytes.fetch_add(CHUNK_BYTES as u64, Ordering::Relaxed);
            if stop.load(Ordering::Relaxed) {
                break;
            }
        }
    }
}

struct Arm {
    name: &'static str,
    k: usize,
    us: Vec<f64>,
    gbps: f64,
}

/// Linear-interpolated percentile of a sorted slice.
fn pct(sorted: &[f64], p: f64) -> f64 {
    let x = p * (sorted.len() - 1) as f64;
    let (lo, hi) = (x.floor() as usize, x.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (x - lo as f64)
}

struct Stats {
    median: f64,
    p10r: f64,
    p90r: f64,
    s: f64,
    lag1: f64,
}

fn stats(us: &[f64]) -> Stats {
    let mut v = us.to_vec();
    v.sort_by(f64::total_cmp);
    let (p10, p50, p90) = (pct(&v, 0.1), pct(&v, 0.5), pct(&v, 0.9));
    let mean = us.iter().sum::<f64>() / us.len() as f64;
    let var: f64 = us.iter().map(|x| (x - mean).powi(2)).sum();
    let cov: f64 = us.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum();
    Stats {
        median: p50,
        p10r: p10 / p50,
        p90r: p90 / p50,
        s: p90 / p10,
        lag1: if var > 0.0 { cov / var } else { 0.0 },
    }
}

/// One arm: start `k` copy threads, wait 1 s, warm up 200 iterations, time `iters` iterations
/// each between two events, stop and join. All events are read after one sync at the end.
fn run_arm(
    rig: &Rig,
    bufs: &mut [HostBufs],
    name: &'static str,
    k: usize,
    iters: usize,
) -> Result<Arm> {
    let (stop, bytes) = (AtomicBool::new(false), AtomicU64::new(0));
    let (stop_r, bytes_r) = (&stop, &bytes);
    std::thread::scope(|sc| {
        for b in bufs.iter_mut().take(k) {
            sc.spawn(move || copier(b, stop_r, bytes_r));
        }
        let r = (|| -> Result<Arm> {
            std::thread::sleep(SETTLE);
            for i in 0..WARMUP {
                rig.slice(i)?;
            }
            rig.g.synchronize(rig.s)?;
            let mut ev = Vec::with_capacity(2 * iters);
            let timed = (|| -> Result<(Vec<f64>, f64)> {
                for _ in 0..2 * iters {
                    let mut e = 0u64;
                    ck(unsafe { cuEventCreate(&mut e, 0) }, "cuEventCreate")?;
                    ev.push(e);
                }
                let b0 = bytes_r.load(Ordering::Relaxed);
                let t0 = Instant::now();
                for i in 0..iters {
                    ck(
                        unsafe { cuEventRecord(ev[2 * i], rig.s) },
                        "cuEventRecord(start)",
                    )?;
                    rig.slice(WARMUP + i)?;
                    ck(
                        unsafe { cuEventRecord(ev[2 * i + 1], rig.s) },
                        "cuEventRecord(end)",
                    )?;
                }
                ck(
                    unsafe { cuEventSynchronize(ev[2 * iters - 1]) },
                    "cuEventSynchronize",
                )?;
                let wall = t0.elapsed().as_secs_f64();
                let gbps = (bytes_r.load(Ordering::Relaxed) - b0) as f64 / wall / 1e9;
                let mut us = Vec::with_capacity(iters);
                for i in 0..iters {
                    let mut ms = 0f32;
                    ck(
                        unsafe { cuEventElapsedTime(&mut ms, ev[2 * i], ev[2 * i + 1]) },
                        "cuEventElapsedTime",
                    )?;
                    us.push(ms as f64 * 1e3);
                }
                Ok((us, gbps))
            })();
            for &e in &ev {
                unsafe {
                    cuEventDestroy_v2(e);
                }
            }
            let (us, gbps) = timed?;
            ensure!(us.iter().all(|&x| x > 0.0), "{name}: non-positive duration");
            Ok(Arm { name, k, us, gbps })
        })();
        stop_r.store(true, Ordering::Relaxed);
        r
    })
}

fn run() -> Result<i32> {
    let iters: usize = match std::env::var("JITTER_ITERS") {
        Ok(v) => v.parse()?,
        Err(_) => DEFAULT_ITERS,
    };
    ensure!(iters >= 2, "JITTER_ITERS must be >= 2");
    let mut csv = match std::env::var("JITTER_CSV") {
        Ok(p) if !p.is_empty() => {
            let mut f = std::fs::File::create(p)?;
            writeln!(f, "arm,k,iter,us")?;
            Some(f)
        }
        _ => None,
    };
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut rng = Rng(0x006a_6974_7465_7231);
    let rig = build_rig(g, &mut rng)?;
    let (dense_b, moe_b) = bytes_per_iter();
    println!(
        "CONFIG rows={ROWS} union={UNION} dense_shapes={} moe=gate+up+down mi={MI} hidden={HIDDEN} \
         launches_per_iter={} warmup={WARMUP} iters={iters} \
         bytes_per_iter={:.0} (dense {dense_b:.0} + moe {moe_b:.0}) MB_per_iter={:.1} \
         dense_pool={DENSE_POOL} moe_experts_per_matrix={NUM_EXPERTS} routings={ROUTINGS}",
        DENSE_SHAPES
            .iter()
            .map(|s| format!("{}x{}", s.0, s.1))
            .collect::<Vec<_>>()
            .join("+"),
        DENSE_SHAPES.len() + 3,
        dense_b + moe_b,
        (dense_b + moe_b) / 1e6
    );
    println!(
        "CONFIG host buffers: {MAX_K} pairs x 2 x 1 GiB pre-touched (copy threads use the first k)"
    );
    let mut bufs: Vec<HostBufs> = (0..MAX_K)
        .map(|_| HostBufs {
            src: vec![0xA5u8; BUF_BYTES],
            dst: vec![0x5Au8; BUF_BYTES],
        })
        .collect();
    let mut arms: Vec<Arm> = Vec::new();
    for (name, k) in ARMS {
        let a = run_arm(&rig, &mut bufs, name, k, iters)?;
        let st = stats(&a.us);
        println!(
            "ARM {} k={} n={} median_us={:.2} p10r={:.3} p50r=1.000 p90r={:.3} S={:.3} lag1={:.3} \
             memcpy_GBps={:.1}",
            a.name,
            a.k,
            a.us.len(),
            st.median,
            st.p10r,
            st.p90r,
            st.s,
            st.lag1,
            a.gbps
        );
        if let Some(f) = csv.as_mut() {
            for (i, u) in a.us.iter().enumerate() {
                writeln!(f, "{},{},{},{:.3}", a.name, a.k, i, u)?;
            }
            f.flush()?;
        }
        arms.push(a);
    }
    let st: Vec<Stats> = arms.iter().map(|a| stats(&a.us)).collect();
    let n_ok = arms.iter().all(|a| a.us.len() >= MIN_N);
    let drift = (st[0].median / st[3].median - 1.0).abs();
    let quiet_s = (st[0].s + st[3].s) / 2.0;
    let (s_h2, s_h8) = (st[1].s, st[2].s);
    let verdict = if !n_ok || drift > MAX_DRIFT {
        "VOID"
    } else if quiet_s >= KILL_S {
        "KILL"
    } else if quiet_s <= GO_QUIET_S && s_h8 >= GO_H8_S {
        "GO"
    } else {
        "GREY"
    };
    println!(
        "GRADE n_ok={n_ok} quiet_median_drift={drift:.4} (limit {MAX_DRIFT}) \
         bands: KILL quietS>={KILL_S}, GO quietS<={GO_QUIET_S} and S_H8>={GO_H8_S}"
    );
    println!("VERDICT hostdram {verdict} quietS={quiet_s:.3} S_H2={s_h2:.3} S_H8={s_h8:.3}");
    if n_ok {
        println!("PASS kernel_jitter_microtest ran (instrument complete; hostdram verdict above)");
        Ok(0)
    } else {
        println!("FAIL kernel_jitter_microtest: an arm has n < {MIN_N} (JITTER_ITERS override?)");
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
