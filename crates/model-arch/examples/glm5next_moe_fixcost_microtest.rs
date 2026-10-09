// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Fixed-cost vs per-expert cost of the car's routed-MoE union sweep
//! `w4a16_gemv_sw_moe_batchm_m<R>_c8` on the GLM-5.3 expert-TP per-rank shapes (hidden 4096,
//! 288 experts, top_k 8, mi 1024). Derived from glm5next_moe_tc_microtest.rs (3cf6e764) with
//! every tensor-core entry removed.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Per union size U (U = 0: every u_eid = -1, so each CTA early-exits: pure launch cost): the
//! gate, up and down launches timed separately and as the 3-launch sequence, each as a 42-layer
//! CUDA graph over a cold pool (3 layers x 3 matrices x 288 experts, ~6 GB), median of 9 samples.
//! The runtime has no event-elapsed API, so timing is graph replay wall time, not CUDA events.
//! Check: the car entry's bytes equal the CUDA-core `_m<R>` entry's on gate/up and down shapes
//! (random routing with holes), unused (row, slot)s keep 0xA5.
//!
//! Env: MOE_MT_ROWS (R, 2..=8, default 8), MOE_MT_UNIONS (comma list, clamped to R * 8).
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_moe_fixcost_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const HIDDEN: usize = 4096;
const MI: usize = 1024;
const TOP_K: usize = 8;
const NUM_EXPERTS: usize = 288;
const LAYERS: usize = 3;
const ROUTINGS: usize = 4;
const STEP_LAYERS: usize = 42;
const SAMPLES: usize = 9;
const REPLAYS: usize = 4;
const POISON: u8 = 0xA5;
const BYTES_PER_W: f64 = 0.5 + 1.0 / 16.0;
const DEFAULT_UNIONS: &str = "0,1,2,4,8,12,16,18,20,24,28,32,36,40,44,45,48,52,56,57,60,64";

struct Rng(u64);
#[rustfmt::skip]
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn u(&mut self) -> f32 { (self.next() & 0xFF_FFFF) as f32 / 16_777_216.0 }
    fn below(&mut self, n: usize) -> usize { self.next() as usize % n }
}

#[rustfmt::skip]
fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> { let p = g.alloc(b.len().max(1))?; g.copy_h2d(b, p)?; Ok(p) }
fn le<T: Copy, const B: usize>(v: &[T], f: impl Fn(T) -> [u8; B]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}
#[rustfmt::skip]
fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> { let mut b = vec![0u8; n]; g.copy_d2h(p, &mut b)?; Ok(b) }
#[rustfmt::skip]
fn bf16_bytes(rng: &mut Rng, n: usize) -> Vec<u8> { le(&(0..n).map(|_| bf16::from_f32(rng.u() * 2.0 - 1.0)).collect::<Vec<_>>(), |x| x.to_bits().to_le_bytes()) }

/// Host NVFP4 bytes for an [n, k] expert (random nibbles; E4M3 scales 1.0..1.875) with 64 spare
/// bytes; buffer b of a table starts at offset (b * 17) % 64.
#[rustfmt::skip]
struct Host { w: Vec<u8>, s: Vec<u8>, wb: usize, sb: usize }
#[rustfmt::skip]
fn gen_host(rng: &mut Rng, n: usize, k: usize) -> Host {
    let (wb, sb) = (n * k / 2, n * k / 16);
    Host { w: (0..wb + 64).map(|_| rng.next() as u8).collect(), s: (0..sb + 64).map(|_| 0x38 | (rng.next() as u8 & 7)).collect(), wb, sb }
}

/// The 288-entry pointer table of one matrix; entries cycle through `distinct` real buffers.
#[rustfmt::skip]
struct Table { packed: DevicePtr, scale: DevicePtr, scale2: DevicePtr }
#[rustfmt::skip]
fn make_table(g: &dyn GpuBackend, h: &Host, distinct: usize) -> Result<Table> {
    let (mut pp, mut sp, mut s2) = (Vec::new(), Vec::new(), Vec::new());
    let mut bufs = Vec::new();
    for e in 0..NUM_EXPERTS {
        if e < distinct {
            let off = (e * 17) % 64;
            bufs.push((up(g, &h.w[off..off + h.wb])?.0, up(g, &h.s[off..off + h.sb])?.0));
        }
        let (p, s) = bufs[e % distinct];
        pp.push(p);
        sp.push(s);
        s2.push(1.0f32 + e as f32 * 0.01);
    }
    Ok(Table {
        packed: up(g, &le(&pp, |x: u64| x.to_le_bytes()))?,
        scale: up(g, &le(&sp, |x: u64| x.to_le_bytes()))?,
        scale2: up(g, &le(&s2, |x: f32| x.to_le_bytes()))?,
    })
}

/// Union tables from `glm5next_moe_row_union`; `union` = live entries.
#[rustfmt::skip]
struct Routing { ueid: DevicePtr, uslot: DevicePtr, union: usize }
#[rustfmt::skip]
fn build_routing(g: &dyn GpuBackend, k_union: KernelHandle, ids: &[Vec<i32>]) -> Result<Routing> {
    let rows = ids.len();
    let flat: Vec<i32> = ids.iter().flatten().copied().collect();
    let d_ids = up(g, &le(&flat, |x: i32| x.to_le_bytes()))?;
    let ueid = g.alloc(rows * TOP_K * 4)?;
    let uslot = g.alloc(rows * TOP_K * rows * 4)?;
    KernelLaunch::new(g, k_union)
        .grid([1, 1, 1]).block([(rows * TOP_K) as u32, 1, 1])
        .arg_ptr(d_ids).arg_ptr(ueid).arg_ptr(uslot).arg_u32(rows as u32).arg_u32(TOP_K as u32)
        .launch(0)?;
    g.synchronize(0)?;
    g.free(d_ids)?;
    let mut d: Vec<i32> = flat.into_iter().filter(|&e| e >= 0).collect();
    d.sort_unstable();
    d.dedup();
    Ok(Routing { ueid, uslot, union: d.len() })
}
#[rustfmt::skip]
fn free_routing(g: &dyn GpuBackend, r: &Routing) -> Result<()> { g.free(r.ueid)?; g.free(r.uslot) }

/// `rows` rows of TOP_K distinct random experts each.
#[rustfmt::skip]
fn random_ids(rng: &mut Rng, rows: usize) -> Vec<Vec<i32>> {
    (0..rows).map(|_| {
        let mut p: Vec<i32> = (0..NUM_EXPERTS as i32).collect();
        for i in 0..TOP_K { let j = i + rng.below(NUM_EXPERTS - i); p.swap(i, j); }
        p[..TOP_K].to_vec()
    }).collect()
}

/// `rows` rows whose union has exactly `u` experts. u == 0: every id -1 (empty union). u < 8:
/// row 0 holds u experts and -1s, the other rows -1. u >= 8 (<= rows * 8): the first `u`
/// row-major slots take `u` distinct random experts, each later slot one its row lacks.
#[rustfmt::skip]
fn union_ids(rng: &mut Rng, rows: usize, u: usize) -> Vec<Vec<i32>> {
    let mut pool: Vec<i32> = (0..NUM_EXPERTS as i32).collect();
    for i in 0..u { let j = i + rng.below(NUM_EXPERTS - i); pool.swap(i, j); }
    pool.truncate(u);
    let mut ids = vec![Vec::new(); rows];
    for p in 0..rows * TOP_K {
        let row = &mut ids[p / TOP_K];
        let e = if u < TOP_K { if p < u { pool[p] } else { -1 } } else if p < u { pool[p] } else {
            let free: Vec<i32> = pool.iter().copied().filter(|e| !row.contains(e)).collect();
            free[rng.below(free.len())]
        };
        row.push(e);
    }
    ids
}

/// One union sweep of `h` (a `_m<R>` or `_m<R>_c8` entry; `cols` columns per CTA).
#[allow(clippy::too_many_arguments)]
#[rustfmt::skip]
fn launch(
    g: &dyn GpuBackend, h: KernelHandle, cols: usize, a: DevicePtr, t: &Table, c: DevicePtr, rt: &Routing,
    rows: usize, n: usize, kk: usize, down: bool, s: u64,
) -> Result<()> {
    let (ars, ass) = if down { (TOP_K * MI, MI) } else { (kk, 0) };
    KernelLaunch::new(g, h)
        .grid([n.div_ceil(cols) as u32, (rows * TOP_K) as u32, 1]).block([256, 1, 1])
        .arg_ptr(a).arg_ptr(t.packed).arg_ptr(t.scale).arg_ptr(t.scale2).arg_ptr(c)
        .arg_ptr(rt.ueid).arg_ptr(rt.uslot)
        .arg_u32(n as u32).arg_u32(kk as u32).arg_u32(NUM_EXPERTS as u32)
        .arg_u32(ars as u32).arg_u32(ass as u32).arg_u32((TOP_K * n) as u32)
        .launch(s)
}

#[rustfmt::skip]
struct Kerns { core: KernelHandle, car: KernelHandle, union: KernelHandle }
const CORE_COLS: usize = 8;
const CAR_COLS: usize = 64;

#[rustfmt::skip]
struct LayerTabs { gate: Table, up: Table, down: Table }
#[rustfmt::skip]
struct Bufs { x: DevicePtr, act: DevicePtr, c_gu: DevicePtr, c_dn: DevicePtr }

/// Median ms of one replay of the captured `f` (REPLAYS back to back per sample).
#[rustfmt::skip]
fn time_graph(g: &dyn GpuBackend, s: u64, f: &mut dyn FnMut() -> Result<()>) -> Result<f64> {
    g.begin_capture(s)?;
    if let Err(e) = f() {
        g.abort_capture_if_active(s);
        return Err(e);
    }
    let graph = g.end_capture(s)?;
    for _ in 0..3 { g.launch_graph(graph, s)?; }
    g.synchronize(s)?;
    let mut v = Vec::new();
    for _ in 0..SAMPLES {
        let t0 = std::time::Instant::now();
        for _ in 0..REPLAYS { g.launch_graph(graph, s)?; }
        g.synchronize(s)?;
        v.push(t0.elapsed().as_secs_f64() * 1e3 / REPLAYS as f64);
    }
    g.destroy_graph(graph)?;
    v.sort_by(f64::total_cmp);
    Ok(v[SAMPLES / 2])
}

/// us per layer of [gate, up, down, seq]; seq is the three launches back to back.
#[allow(clippy::too_many_arguments)]
#[rustfmt::skip]
fn time_u(g: &dyn GpuBackend, s: u64, kz: &Kerns, layers: &[LayerTabs], routs: &[Routing], b: &Bufs, rows: usize) -> Result<[f64; 4]> {
    let mut out = [0f64; 4];
    for (mode, o) in out.iter_mut().enumerate() {
        let ms = time_graph(g, s, &mut || {
            for l in 0..STEP_LAYERS {
                let (lt, rt) = (&layers[l % LAYERS], &routs[l % routs.len()]);
                if mode == 0 || mode == 3 { launch(g, kz.car, CAR_COLS, b.x, &lt.gate, b.c_gu, rt, rows, MI, HIDDEN, false, s)?; }
                if mode == 1 || mode == 3 { launch(g, kz.car, CAR_COLS, b.x, &lt.up, b.c_gu, rt, rows, MI, HIDDEN, false, s)?; }
                if mode == 2 || mode == 3 { launch(g, kz.car, CAR_COLS, b.act, &lt.down, b.c_dn, rt, rows, HIDDEN, MI, true, s)?; }
            }
            Ok(())
        })?;
        *o = ms * 1e3 / STEP_LAYERS as f64;
    }
    Ok(out)
}

/// Check: car bytes == `_m<R>` bytes on one gate/up-like and one down-like shape, random
/// routing with holes; returns the number of mismatching shapes.
#[rustfmt::skip]
fn check(g: &dyn GpuBackend, kz: &Kerns, rng: &mut Rng, rows: usize) -> Result<usize> {
    let mut fails = 0;
    for (label, n, k, down) in [("gate/up", MI, HIDDEN, false), ("down", HIDDEN, MI, true)] {
        let t = make_table(g, &gen_host(rng, n, k), 4)?;
        let d_a = up(g, &bf16_bytes(rng, if down { rows * TOP_K * MI } else { rows * HIDDEN }))?;
        let bytes = rows * TOP_K * n * 2;
        let (d_core, d_car) = (g.alloc(bytes)?, g.alloc(bytes)?);
        let mut ids = random_ids(rng, rows);
        for (r, row) in ids.iter_mut().enumerate() { for (s, e) in row.iter_mut().enumerate() { if (r + s) % 5 == 0 { *e = -1; } } }
        let rt = build_routing(g, kz.union, &ids)?;
        let mut y = Vec::new();
        for (h, cols, d_c) in [(kz.core, CORE_COLS, d_core), (kz.car, CAR_COLS, d_car)] {
            g.memset_async(d_c, POISON, bytes, 0)?;
            launch(g, h, cols, d_a, &t, d_c, &rt, rows, n, k, down, 0)?;
            g.synchronize(0)?;
            y.push(dn(g, d_c, bytes)?);
        }
        let (live, same) = (y[0].iter().filter(|&&b| b != POISON).count(), y[0] == y[1]);
        println!("CHECK {label} R={rows} union={} car==core {same} (non-poison bytes {live})", rt.union);
        if !same || live == 0 { fails += 1; }
        free_routing(g, &rt)?;
        for p in [d_a, d_core, d_car, t.packed, t.scale, t.scale2] { g.free(p)?; }
    }
    Ok(fails)
}

/// Least squares y = a + b x; returns (slope, intercept, max |residual|).
fn fit(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let (sx, sy) = (x.iter().sum::<f64>(), y.iter().sum::<f64>());
    let sxx = x.iter().map(|v| v * v).sum::<f64>();
    let sxy = x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let icpt = (sy - slope * sx) / n;
    let res = x
        .iter()
        .zip(y)
        .map(|(a, b)| (b - icpt - slope * a).abs())
        .fold(0.0, f64::max);
    (slope, icpt, res)
}

#[rustfmt::skip]
fn run() -> Result<i32> {
    let rows: usize = std::env::var("MOE_MT_ROWS").ok().map_or(Ok(8), |v| v.trim().parse())?;
    if !(2..=8).contains(&rows) { bail!("MOE_MT_ROWS must be 2..=8, got {rows}"); }
    let spec = std::env::var("MOE_MT_UNIONS").unwrap_or_else(|_| DEFAULT_UNIONS.into());
    let mut unions = Vec::new();
    for t in spec.split(',').filter(|t| !t.trim().is_empty()) { unions.push(t.trim().parse::<usize>()?.min(rows * TOP_K)); }
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let names = [format!("w4a16_gemv_sw_moe_batchm_m{rows}"), format!("w4a16_gemv_sw_moe_batchm_m{rows}_c8"), "glm5next_moe_row_union".into()];
    let mut hs = Vec::new();
    for n in &names {
        match g.kernel("w4a16_gemv", n) { Ok(h) => hs.push(h), Err(_) => { println!("MISSING kernel entry {n}"); return Ok(2); } }
    }
    let kz = Kerns { core: hs[0], car: hs[1], union: hs[2] };
    let mut rng = Rng(0x006d_6f65_6669_7863);
    let bad = check(g, &kz, &mut rng, rows)?;
    // Cold pool: LAYERS layers x (gate, up, down) x 288 distinct experts.
    let (hgu, hdn) = (gen_host(&mut rng, MI, HIDDEN), gen_host(&mut rng, HIDDEN, MI));
    let mut layers = Vec::new();
    for _ in 0..LAYERS {
        let (gate, up) = (make_table(g, &hgu, NUM_EXPERTS)?, make_table(g, &hgu, NUM_EXPERTS)?);
        layers.push(LayerTabs { gate, up, down: make_table(g, &hdn, NUM_EXPERTS)? });
    }
    let s = g.create_stream()?;
    let b = Bufs {
        x: up(g, &bf16_bytes(&mut rng, rows * HIDDEN))?,
        act: up(g, &bf16_bytes(&mut rng, rows * TOP_K * MI))?,
        c_gu: g.alloc(rows * TOP_K * MI * 2)?,
        c_dn: g.alloc(rows * TOP_K * HIDDEN * 2)?,
    };
    let (mut us, mut ts) = (Vec::new(), Vec::new());
    for &u in &unions {
        let routs: Vec<Routing> = (0..ROUTINGS)
            .map(|_| build_routing(g, kz.union, &union_ids(&mut rng, rows, u)))
            .collect::<Result<_>>()?;
        let t = time_u(g, s, &kz, &layers, &routs, &b, rows)?;
        let mu = routs.iter().map(|r| r.union as f64).sum::<f64>() / ROUTINGS as f64;
        let gbs = |us: f64| mu * (MI * HIDDEN) as f64 * BYTES_PER_W / us / 1e3;
        println!("U={u} R={rows} gate_us={:.2} up_us={:.2} down_us={:.2} seq_us={:.2} gate_GBs={:.0} up_GBs={:.0} down_GBs={:.0}",
            t[0], t[1], t[2], t[3], gbs(t[0]), gbs(t[1]), gbs(t[2]));
        us.push(mu);
        ts.push(t);
        for r in &routs { free_routing(g, r)?; }
    }
    println!("FIT least squares over U>=8 (us vs union experts)");
    let sel: Vec<usize> = (0..us.len()).filter(|&i| us[i] >= 8.0).collect();
    if sel.len() >= 2 {
        let x: Vec<f64> = sel.iter().map(|&i| us[i]).collect();
        for (m, name) in ["gate", "up", "down", "seq"].iter().enumerate() {
            let y: Vec<f64> = sel.iter().map(|&i| ts[i][m]).collect();
            let (sl, ic, res) = fit(&x, &y);
            println!("FIT {name}: slope_us_per_expert={sl:.3} intercept_us={ic:.2} resid_max_us={res:.2} (n={})", x.len());
        }
    } else {
        println!("FIT skipped: fewer than 2 unions >= 8");
    }
    if let Some(i) = unions.iter().position(|&u| u == 0) {
        let t = ts[i];
        println!("U0 gate_us={:.2} up_us={:.2} down_us={:.2} seq_us={:.2}", t[0], t[1], t[2], t[3]);
    }
    if bad == 0 { println!("PASS: car _m{rows}_c8 bytes equal _m{rows} on gate/up and down; sweep done"); Ok(0) } else { println!("FAIL: car != core on {bad} shape(s)"); Ok(1) }
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
