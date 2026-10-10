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
//! 2026-10-08: When the image has `w4a16_gemv_sw_moe_batchm_down_m<R>` (METRALE_GLM_MOE_DOWN_FAST),
//! its down launch is timed too (`down_fast_us`, `down_fast_GBs`) and its bytes are compared to
//! the car `_c8` down on every routing of every U, on the check routing with holes and on a
//! ragged one (odd rows all -1, holes in the rest); any mismatch prints a FAIL line.
//! 2026-10-09: When the image has `w4a16_gemv_sw_moe_batchm_gateup_m<R>`
//! (METRALE_GLM_MOE_GATEUP_FAST), its one gate+up launch is timed too (`gateup_fast_us`, the
//! combined time to compare with gate_us + up_us; `gateup_fast_GBs` counts both weight sets) and
//! its gate and up bytes are compared to the car `_c8` gate and up launches on every routing of
//! every U, on the check routing with holes and on a ragged one; any mismatch prints a FAIL line.
//!
//! 2026-10-10: Replay mode (env FIXCOST_REPLAY=<path>|1; R=8 only): times the entries on REAL 8-row
//! routings captured from a serve (examples/data/glm53_route_r8.bin, written by
//! runs/race/mem/route_extract.py; embedded with include_bytes!, `1` or `embedded` selects it) and on
//! synthetic routings at U = round(replay U_mean), each with a clean and a dirty L2 (a memset of
//! FIXCOST_DIRTY_MB, default 6.5, before every timed MoE launch; its own time is removed by
//! subtracting a graph of the memsets alone, since the runtime has no event-elapsed API). The REPLAY /
//! REPLAY_SYNTH / REPLAY2 lines are per-call averages. FIXCOST_REPLAY_CHUNKS (default 24) = chunks of
//! 42 consecutive calls, spread evenly over the file. Unset: behaviour unchanged.
//!
//! 2026-10-10: FIXCOST_SFA=1 (microtest-only falsifier for race #68 "SFB once", seed-skills#68
//! 6096187630): every scale buffer also gets a copy with the same bytes permuted into the CUTLASS
//! Sm1xx SFB atom layout (kernels `*_sfa*` twins in w4a16_gemv.cu, same arithmetic, only the scale
//! address differs). Wherever an entry with a twin is timed (sweep per U, REPLAY/REPLAY_SYNTH/REPLAY2
//! cells) the twin is timed right after with the same routings, graph and samples and printed as
//! `SFA entry=.. R=.. routing=.. l2=.. off_us=.. on_us=.. ratio=..`; every routing that is byte-checked
//! also runs the twin and compares bytes (`SFA BITWISE entry=.. identical|DIFF first_mismatch=..`, a
//! DIFF prints FAIL). A sw_moe R1 cell (w4a16_gemv_sw_moe gate/up/down, one row, top_k 8 slots, as the
//! forward launches it) is timed and compared too. Unset: behaviour and output unchanged.
//! Run: FIXCOST_SFA=1 FIXCOST_REPLAY=1 (add MOE_MT_UNIONS=8,32 to shorten the sweep).
//!
//! Env: MOE_MT_ROWS (R, 2..=8, default 8), MOE_MT_UNIONS (comma list, clamped to R * 8).
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_moe_fixcost_microtest
//! Replay: add FIXCOST_REPLAY=1 (MOE_MT_UNIONS=8 shortens the sweep).

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
struct Host { w: Vec<u8>, s: Vec<u8>, wb: usize, sb: usize, n: usize, k: usize }
#[rustfmt::skip]
fn gen_host(rng: &mut Rng, n: usize, k: usize) -> Host {
    let (wb, sb) = (n * k / 2, n * k / 16);
    Host { w: (0..wb + 64).map(|_| rng.next() as u8).collect(), s: (0..sb + 64).map(|_| 0x38 | (rng.next() as u8 & 7)).collect(), wb, sb, n, k }
}

/// The 288-entry pointer table of one matrix; entries cycle through `distinct` real buffers.
/// `sfa` (FIXCOST_SFA=1): the scale pointer table over copies of the same bytes in the SFB atom layout.
#[derive(Clone, Copy)]
#[rustfmt::skip]
struct Table { packed: DevicePtr, scale: DevicePtr, scale2: DevicePtr, sfa: Option<DevicePtr> }
#[rustfmt::skip]
fn sfa_on() -> bool { std::env::var("FIXCOST_SFA").is_ok_and(|v| v.trim() == "1") }
/// CUTLASS Sm1xx SFB atom byte offset of scale (row n, group g), gg = K / 16 groups per row.
#[rustfmt::skip]
fn sfa_off(n: usize, g: usize, gg: usize) -> usize { ((n >> 7) * (gg >> 2) + (g >> 2)) * 512 + (n & 31) * 16 + ((n >> 5) & 3) * 4 + (g & 3) }
/// Row-major [n, k/16] scale bytes -> the same bytes in the atom layout.
#[rustfmt::skip]
fn to_atom(s: &[u8], n: usize, k: usize) -> Vec<u8> {
    let gg = k / 16;
    assert!(n % 128 == 0 && gg % 4 == 0 && s.len() == n * gg, "SFA needs N % 128 == 0 and (K/16) % 4 == 0 (n {n} k {k})");
    let mut o = vec![0u8; n * gg];
    for r in 0..n { for c in 0..gg { o[sfa_off(r, c, gg)] = s[r * gg + c]; } }
    o
}
/// The table a `_sfa` twin reads: same packed and scale2, scale pointers into the atom copies.
#[rustfmt::skip]
fn twin(t: &Table) -> Table { Table { scale: t.sfa.expect("table has no SFA copy"), sfa: t.sfa, ..*t } }
#[rustfmt::skip]
fn free_table(g: &dyn GpuBackend, t: &Table) -> Result<()> { for p in [t.packed, t.scale, t.scale2].into_iter().chain(t.sfa) { g.free(p)?; } Ok(()) }
#[rustfmt::skip]
fn make_table(g: &dyn GpuBackend, h: &Host, distinct: usize) -> Result<Table> {
    let (mut pp, mut sp, mut s2, mut sa) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut bufs = Vec::new();
    for e in 0..NUM_EXPERTS {
        if e < distinct {
            let off = (e * 17) % 64;
            let atom = if sfa_on() { Some(up(g, &to_atom(&h.s[off..off + h.sb], h.n, h.k))?.0) } else { None };
            bufs.push((up(g, &h.w[off..off + h.wb])?.0, up(g, &h.s[off..off + h.sb])?.0, atom));
        }
        let (p, s, a) = bufs[e % distinct];
        pp.push(p);
        sp.push(s);
        if let Some(a) = a { sa.push(a); }
        s2.push(1.0f32 + e as f32 * 0.01);
    }
    Ok(Table {
        packed: up(g, &le(&pp, |x: u64| x.to_le_bytes()))?,
        scale: up(g, &le(&sp, |x: u64| x.to_le_bytes()))?,
        scale2: up(g, &le(&s2, |x: f32| x.to_le_bytes()))?,
        sfa: if sfa_on() { Some(up(g, &le(&sa, |x: u64| x.to_le_bytes()))?) } else { None },
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
struct Kerns { core: KernelHandle, car: KernelHandle, union: KernelHandle, fast: Option<KernelHandle>, gu: Option<KernelHandle>, car_s: Option<KernelHandle>, fast_s: Option<KernelHandle>, gu_s: Option<KernelHandle>, sw: Option<KernelHandle>, sw_s: Option<KernelHandle> }
const CORE_COLS: usize = 8;
const CAR_COLS: usize = 64;
const FAST_COLS: usize = 128;
const GU_FAST_COLS: usize = 64;

/// 2026-10-09: One `w4a16_gemv_sw_moe_batchm_gateup_m<R>` launch: gate (`tg`) into `cg` and up
/// (`tu`) into `cu`, gate/up shape (N = MI, K = HIDDEN), x shared by every slot.
#[allow(clippy::too_many_arguments)]
#[rustfmt::skip]
fn launch_gu(g: &dyn GpuBackend, h: KernelHandle, x: DevicePtr, tg: &Table, tu: &Table, cg: DevicePtr, cu: DevicePtr, rt: &Routing, rows: usize, s: u64) -> Result<()> {
    KernelLaunch::new(g, h)
        .grid([MI.div_ceil(GU_FAST_COLS) as u32, (rows * TOP_K) as u32, 1]).block([256, 1, 1])
        .arg_ptr(x).arg_ptr(tg.packed).arg_ptr(tg.scale).arg_ptr(tg.scale2)
        .arg_ptr(tu.packed).arg_ptr(tu.scale).arg_ptr(tu.scale2).arg_ptr(cg).arg_ptr(cu)
        .arg_ptr(rt.ueid).arg_ptr(rt.uslot)
        .arg_u32(MI as u32).arg_u32(HIDDEN as u32).arg_u32(NUM_EXPERTS as u32)
        .arg_u32(HIDDEN as u32).arg_u32(0).arg_u32((TOP_K * MI) as u32)
        .launch(s)
}

/// 2026-10-09: Car `_c8` gate and up vs one `gateup_m<R>` launch on `rt`: all four outputs
/// poisoned first; true when gate bytes and up bytes are equal (and, for a non-empty union,
/// some byte was written).
#[allow(clippy::too_many_arguments)]
#[rustfmt::skip]
fn gateup_fast_same(g: &dyn GpuBackend, kz: &Kerns, gu: KernelHandle, x: DevicePtr, tg: &Table, tu: &Table, c: [DevicePtr; 4], rt: &Routing, rows: usize) -> Result<bool> {
    let bytes = rows * TOP_K * MI * 2;
    for d_c in c { g.memset_async(d_c, POISON, bytes, 0)?; }
    launch(g, kz.car, CAR_COLS, x, tg, c[0], rt, rows, MI, HIDDEN, false, 0)?;
    launch(g, kz.car, CAR_COLS, x, tu, c[1], rt, rows, MI, HIDDEN, false, 0)?;
    launch_gu(g, gu, x, tg, tu, c[2], c[3], rt, rows, 0)?;
    g.synchronize(0)?;
    let y: Vec<Vec<u8>> = c.iter().map(|&p| dn(g, p, bytes)).collect::<Result<_>>()?;
    let wrote = rt.union == 0 || (y[0].iter().any(|&b| b != POISON) && y[1].iter().any(|&b| b != POISON));
    Ok(y[0] == y[2] && y[1] == y[3] && wrote)
}

/// 2026-10-08: Car `_c8` down vs `down_m<R>` on `rt`: both outputs poisoned first; true when the
/// bytes are equal (and, for a non-empty union, some byte was written).
#[allow(clippy::too_many_arguments)]
#[rustfmt::skip]
fn down_fast_same(g: &dyn GpuBackend, kz: &Kerns, fast: KernelHandle, a: DevicePtr, t: &Table, c: [DevicePtr; 2], rt: &Routing, rows: usize) -> Result<bool> {
    let bytes = rows * TOP_K * HIDDEN * 2;
    let mut y = Vec::new();
    for (h, cols, d_c) in [(kz.car, CAR_COLS, c[0]), (fast, FAST_COLS, c[1])] {
        g.memset_async(d_c, POISON, bytes, 0)?;
        launch(g, h, cols, a, t, d_c, rt, rows, HIDDEN, MI, true, 0)?;
        g.synchronize(0)?;
        y.push(dn(g, d_c, bytes)?);
    }
    Ok(y[0] == y[1] && (rt.union == 0 || y[0].iter().any(|&b| b != POISON)))
}

#[rustfmt::skip]
#[derive(Clone, Copy)]
struct LayerTabs { gate: Table, up: Table, down: Table }
#[rustfmt::skip]
struct Bufs { x: DevicePtr, act: DevicePtr, c_gu: DevicePtr, c_up: DevicePtr, c_dn: DevicePtr }

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

/// us per layer of the dirty-L2 memset alone (the part subtracted from the dirty arms).
#[rustfmt::skip]
fn time_dirty_base(g: &dyn GpuBackend, s: u64, d: DevicePtr, n: usize) -> Result<f64> {
    let ms = time_graph(g, s, &mut || { for _ in 0..STEP_LAYERS { g.memset_async(d, 0x5A, n, s)?; } Ok(()) })?;
    Ok(ms * 1e3 / STEP_LAYERS as f64)
}

/// us per layer of [gate, up, down, seq, down_fast, gateup_fast]; seq is the three car launches
/// back to back; down_fast / gateup_fast (one launch for both matrices) are NaN without the entry.
#[allow(clippy::too_many_arguments)]
#[rustfmt::skip]
fn time_u(g: &dyn GpuBackend, s: u64, kz: &Kerns, layers: &[LayerTabs], routs: &[Routing], b: &Bufs, rows: usize, dirty: Option<(DevicePtr, usize)>, skip_seq: bool, sfa: bool) -> Result<[f64; 6]> {
    let (car, fast, gu) = if sfa { (kz.car_s.expect("car_s"), kz.fast_s, kz.gu_s) } else { (kz.car, kz.fast, kz.gu) };
    let mut out = [f64::NAN; 6];
    // Dirty L2: one memset before the (single) MoE launch of each layer; modes with 1 launch only.
    let base = match dirty { Some((d, n)) => time_dirty_base(g, s, d, n)?, None => 0.0 };
    for (mode, o) in out.iter_mut().enumerate() {
        if (mode == 4 && kz.fast.is_none()) || (mode == 5 && kz.gu.is_none()) { continue; }
        if mode == 3 && (skip_seq || dirty.is_some()) { continue; }
        let ms = time_graph(g, s, &mut || {
            for l in 0..STEP_LAYERS {
                let (lt, rt) = (&if sfa { LayerTabs { gate: twin(&layers[l % LAYERS].gate), up: twin(&layers[l % LAYERS].up), down: twin(&layers[l % LAYERS].down) } } else { layers[l % LAYERS] }, &routs[l % routs.len()]);
                if let Some((d, n)) = dirty { g.memset_async(d, 0x5A, n, s)?; }
                if mode == 0 || mode == 3 { launch(g, car, CAR_COLS, b.x, &lt.gate, b.c_gu, rt, rows, MI, HIDDEN, false, s)?; }
                if mode == 1 || mode == 3 { launch(g, car, CAR_COLS, b.x, &lt.up, b.c_gu, rt, rows, MI, HIDDEN, false, s)?; }
                if mode == 2 || mode == 3 { launch(g, car, CAR_COLS, b.act, &lt.down, b.c_dn, rt, rows, HIDDEN, MI, true, s)?; }
                if let (4, Some(h)) = (mode, fast) { launch(g, h, FAST_COLS, b.act, &lt.down, b.c_dn, rt, rows, HIDDEN, MI, true, s)?; }
                if let (5, Some(h)) = (mode, gu) { launch_gu(g, h, b.x, &lt.gate, &lt.up, b.c_gu, b.c_up, rt, rows, s)?; }
            }
            Ok(())
        })?;
        *o = ms * 1e3 / STEP_LAYERS as f64 - base;
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
        if let (true, Some(fh)) = (down, kz.fast) {
            let mut rag = random_ids(rng, rows);
            for (r, row) in rag.iter_mut().enumerate() { for (s, e) in row.iter_mut().enumerate() { if r % 2 == 1 || (r + s) % 3 == 0 { *e = -1; } } }
            let rt2 = build_routing(g, kz.union, &rag)?;
            for (name, r) in [("holes", &rt), ("ragged", &rt2)] {
                let ok = down_fast_same(g, kz, fh, d_a, &t, [d_core, d_car], r, rows)?;
                println!("CHECK down_fast {name} R={rows} union={} fast==car {ok}", r.union);
                if !ok { println!("FAIL down_fast {name} R={rows}: bytes differ from car _c8"); fails += 1; }
            }
            free_routing(g, &rt2)?;
        }
        if let (false, Some(gh)) = (down, kz.gu) {
            let tu = make_table(g, &gen_host(rng, n, k), 4)?;
            let c4 = [g.alloc(bytes)?, g.alloc(bytes)?, g.alloc(bytes)?, g.alloc(bytes)?];
            let mut rag = random_ids(rng, rows);
            for (r, row) in rag.iter_mut().enumerate() { for (s, e) in row.iter_mut().enumerate() { if r % 2 == 1 || (r + s) % 3 == 0 { *e = -1; } } }
            let rt2 = build_routing(g, kz.union, &rag)?;
            for (name, r) in [("holes", &rt), ("ragged", &rt2)] {
                let ok = gateup_fast_same(g, kz, gh, d_a, &t, &tu, c4, r, rows)?;
                println!("CHECK gateup_fast {name} R={rows} union={} fast==car {ok}", r.union);
                if !ok { println!("FAIL gateup_fast {name} R={rows}: gate or up bytes differ from car _c8"); fails += 1; }
            }
            free_routing(g, &rt2)?;
            for p in c4 { g.free(p)?; }
            free_table(g, &tu)?;
        }
        free_routing(g, &rt)?;
        for p in [d_a, d_core, d_car] { g.free(p)?; }
        free_table(g, &t)?;
    }
    if kz.car_s.is_some() {
        // 2026-10-10: twin vs original on 4-distinct-expert tables, a random routing with holes and a ragged one.
        let lt = LayerTabs { gate: make_table(g, &gen_host(rng, MI, HIDDEN), 4)?, up: make_table(g, &gen_host(rng, MI, HIDDEN), 4)?, down: make_table(g, &gen_host(rng, HIDDEN, MI), 4)? };
        let (x, act) = (up(g, &bf16_bytes(rng, rows * HIDDEN))?, up(g, &bf16_bytes(rng, rows * TOP_K * MI))?);
        let mut ids = random_ids(rng, rows);
        for (r, row) in ids.iter_mut().enumerate() { for (s, e) in row.iter_mut().enumerate() { if (r + s) % 5 == 0 { *e = -1; } } }
        let mut rag = random_ids(rng, rows);
        for (r, row) in rag.iter_mut().enumerate() { for (s, e) in row.iter_mut().enumerate() { if r % 2 == 1 || (r + s) % 3 == 0 { *e = -1; } } }
        let (r1, r2) = (build_routing(g, kz.union, &ids)?, build_routing(g, kz.union, &rag)?);
        fails += print_bits(rows, "check: holes + ragged routings", &sfa_bits(g, kz, &lt, x, act, &[&r1, &r2], rows)?);
        free_routing(g, &r1)?;
        free_routing(g, &r2)?;
        for t in [&lt.gate, &lt.up, &lt.down] { free_table(g, t)?; }
        g.free(x)?;
        g.free(act)?;
    }
    Ok(fails)
}

/// 2026-10-10: FIXCOST_SFA. Runs `orig` into `nout` poisoned buffers and `twin` into `nout` more, same
/// `bytes` each; None when all outputs are equal (and, if `expect_write`, written), else the first
/// differing byte index over the concatenated outputs (0 when equal but nothing was written).
#[rustfmt::skip]
fn same_outs(g: &dyn GpuBackend, bytes: usize, nout: usize, expect_write: bool, orig: &mut dyn FnMut(&[DevicePtr]) -> Result<()>, twin_f: &mut dyn FnMut(&[DevicePtr]) -> Result<()>) -> Result<Option<usize>> {
    let c: Vec<DevicePtr> = (0..2 * nout).map(|_| g.alloc(bytes)).collect::<Result<_>>()?;
    for &p in &c { g.memset_async(p, POISON, bytes, 0)?; }
    orig(&c[..nout])?;
    twin_f(&c[nout..])?;
    g.synchronize(0)?;
    let y: Vec<Vec<u8>> = c.iter().map(|&p| dn(g, p, bytes)).collect::<Result<_>>()?;
    for &p in &c { g.free(p)?; }
    let (a, b): (Vec<u8>, Vec<u8>) = (y[..nout].concat(), y[nout..].concat());
    if let Some(i) = a.iter().zip(&b).position(|(x, z)| x != z) { return Ok(Some(i)); }
    Ok(if expect_write && a.iter().all(|&x| x == POISON) { Some(0) } else { None })
}

/// 2026-10-10: Twin vs original bytes of every entry that has a twin, on each routing of `rts`:
/// [(entry, first mismatch)] with the first mismatch over the routings (None = identical on all).
#[allow(clippy::too_many_arguments)]
#[rustfmt::skip]
fn sfa_bits(g: &dyn GpuBackend, kz: &Kerns, lt: &LayerTabs, x: DevicePtr, act: DevicePtr, rts: &[&Routing], rows: usize) -> Result<Vec<(&'static str, Option<usize>)>> {
    let tw = LayerTabs { gate: twin(&lt.gate), up: twin(&lt.up), down: twin(&lt.down) };
    let mut out = Vec::new();
    for e in 0..5 {
        let (name, nout, n) = [("batchm_c8_gate", 1, MI), ("batchm_c8_up", 1, MI), ("batchm_c8_down", 1, HIDDEN), ("down_fast", 1, HIDDEN), ("gateup_fast", 2, MI)][e];
        if (e == 3 && kz.fast.is_none()) || (e == 4 && kz.gu.is_none()) { continue; }
        let mut first = None;
        for rt in rts {
            let run = |sfa: bool, c: &[DevicePtr]| -> Result<()> {
                let (t, car) = if sfa { (&tw, kz.car_s.unwrap()) } else { (lt, kz.car) };
                match e {
                    0 => launch(g, car, CAR_COLS, x, &t.gate, c[0], rt, rows, MI, HIDDEN, false, 0),
                    1 => launch(g, car, CAR_COLS, x, &t.up, c[0], rt, rows, MI, HIDDEN, false, 0),
                    2 => launch(g, car, CAR_COLS, act, &t.down, c[0], rt, rows, HIDDEN, MI, true, 0),
                    3 => launch(g, if sfa { kz.fast_s.unwrap() } else { kz.fast.unwrap() }, FAST_COLS, act, &t.down, c[0], rt, rows, HIDDEN, MI, true, 0),
                    _ => launch_gu(g, if sfa { kz.gu_s.unwrap() } else { kz.gu.unwrap() }, x, &t.gate, &t.up, c[0], c[1], rt, rows, 0),
                }
            };
            let m = same_outs(g, rows * TOP_K * n * 2, nout, rt.union > 0, &mut |c| run(false, c), &mut |c| run(true, c))?;
            if first.is_none() { first = m; }
        }
        out.push((name, first));
    }
    Ok(out)
}

/// Prints the SFA BITWISE lines of `res` (tag names the routings); returns the number of DIFFs.
#[rustfmt::skip]
fn print_bits(rows: usize, tag: &str, res: &[(&str, Option<usize>)]) -> usize {
    let mut bad = 0;
    for (name, m) in res {
        match m {
            None => println!("SFA BITWISE entry={name} R={rows} identical ({tag})"),
            Some(i) => { println!("SFA BITWISE entry={name} R={rows} DIFF first_mismatch={i} ({tag})"); println!("FAIL SFA BITWISE entry={name} R={rows} ({tag}): twin bytes differ from original"); bad += 1; }
        }
    }
    bad
}

/// SFA timing lines for the time_u modes (gate, up, down, seq, down_fast, gateup_fast).
#[rustfmt::skip]
fn print_sfa(rows: usize, routing: &str, l2: &str, off: &[f64; 6], on: &[f64; 6]) {
    for (m, name) in ["batchm_c8_gate", "batchm_c8_up", "batchm_c8_down", "batchm_c8_seq", "down_fast", "gateup_fast"].iter().enumerate() {
        if off[m].is_nan() || on[m].is_nan() { continue; }
        println!("SFA entry={name} R={rows} routing={routing} l2={l2} off_us={:.2} on_us={:.2} ratio={:.4}", off[m], on[m], on[m] / off[m]);
    }
}

/// `w4a16_gemv_sw_moe` (or its `_sfa` twin) over one row: grid (ceil(n / 8), TOP_K), the forward's launch.
#[allow(clippy::too_many_arguments)]
#[rustfmt::skip]
fn launch_r1(g: &dyn GpuBackend, h: KernelHandle, a: DevicePtr, t: &Table, c: DevicePtr, ids: DevicePtr, n: usize, kk: usize, stride: usize, s: u64) -> Result<()> {
    KernelLaunch::new(g, h)
        .grid([n.div_ceil(8) as u32, TOP_K as u32, 1]).block([256, 1, 1])
        .arg_ptr(a).arg_ptr(t.packed).arg_ptr(t.scale).arg_ptr(t.scale2).arg_ptr(c).arg_ptr(ids)
        .arg_u32(n as u32).arg_u32(kk as u32).arg_u32(NUM_EXPERTS as u32).arg_u32(stride as u32)
        .launch(s)
}

/// Sw_moe R1 cell: us per layer of [gate, up, down, seq], off then on, 42-layer graphs over the cold pool,
/// ids cycling over 4 random top-8 rows; plus the bitwise compare on a plain and a holed row. Returns DIFFs.
#[rustfmt::skip]
fn sfa_r1(g: &dyn GpuBackend, s: u64, kz: &Kerns, layers: &[LayerTabs], b: &Bufs, rng: &mut Rng) -> Result<usize> {
    let (h, hs) = (kz.sw.unwrap(), kz.sw_s.unwrap());
    let mut rows_ids: Vec<Vec<i32>> = (0..ROUTINGS).map(|_| random_ids(rng, 1).remove(0)).collect();
    let mut holed = random_ids(rng, 1).remove(0);
    holed[2] = -1; holed[5] = -1;
    rows_ids.push(holed);
    let d_ids: Vec<DevicePtr> = rows_ids.iter().map(|r| up(g, &le(r, |x: i32| x.to_le_bytes()))).collect::<Result<_>>()?;
    let mut t = [[f64::NAN; 4]; 2];
    for (sfa, tt) in t.iter_mut().enumerate() {
        for (mode, o) in tt.iter_mut().enumerate() {
            let ms = time_graph(g, s, &mut || {
                for l in 0..STEP_LAYERS {
                    let (lt, ids) = (&layers[l % LAYERS], d_ids[l % ROUTINGS]);
                    let (hh, gt, ut, dt) = if sfa == 1 { (hs, twin(&lt.gate), twin(&lt.up), twin(&lt.down)) } else { (h, lt.gate, lt.up, lt.down) };
                    if mode == 0 || mode == 3 { launch_r1(g, hh, b.x, &gt, b.c_gu, ids, MI, HIDDEN, 0, s)?; }
                    if mode == 1 || mode == 3 { launch_r1(g, hh, b.x, &ut, b.c_gu, ids, MI, HIDDEN, 0, s)?; }
                    if mode == 2 || mode == 3 { launch_r1(g, hh, b.act, &dt, b.c_dn, ids, HIDDEN, MI, MI, s)?; }
                }
                Ok(())
            })?;
            *o = ms * 1e3 / STEP_LAYERS as f64;
        }
    }
    for (m, name) in ["sw_moe_r1_gate", "sw_moe_r1_up", "sw_moe_r1_down", "sw_moe_r1_seq"].iter().enumerate() {
        println!("SFA entry={name} R=1 routing=U=8 l2=clean off_us={:.2} on_us={:.2} ratio={:.4}", t[0][m], t[1][m], t[1][m] / t[0][m]);
    }
    let mut bad = 0;
    for (e, name) in ["sw_moe_r1_gate", "sw_moe_r1_up", "sw_moe_r1_down"].iter().enumerate() {
        let mut first = None;
        for ids in &d_ids {
            let lt = &layers[0];
            let (tb, a, n, kk, stride) = [(&lt.gate, b.x, MI, HIDDEN, 0), (&lt.up, b.x, MI, HIDDEN, 0), (&lt.down, b.act, HIDDEN, MI, MI)][e];
            let tw = twin(tb);
            let m = same_outs(g, TOP_K * n * 2, 1, true,
                &mut |c| launch_r1(g, h, a, tb, c[0], *ids, n, kk, stride, 0),
                &mut |c| launch_r1(g, hs, a, &tw, c[0], *ids, n, kk, stride, 0))?;
            if first.is_none() { first = m; }
        }
        bad += print_bits(1, "sw_moe R1, 4 random rows + 1 holed row", &[(name, first)]);
    }
    for p in d_ids { g.free(p)?; }
    Ok(bad)
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

/// 2026-10-10: Replay file "GR8R" v1: b"GR8R", u32 version, n, rows, top_k, num_experts; then n x
/// (u8 layer, i16 ids[rows * top_k]); all little-endian. Returns the calls' ids (rows x TOP_K each).
static EMBEDDED_REPLAY: &[u8] = include_bytes!("data/glm53_route_r8.bin");
#[rustfmt::skip]
fn parse_replay(b: &[u8]) -> Result<Vec<Vec<Vec<i32>>>> {
    if b.len() < 24 || &b[..4] != b"GR8R" { bail!("replay file: bad magic"); }
    let w = |i: usize| u32::from_le_bytes(b[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
    let (ver, n, rows, k, e) = (w(0), w(1), w(2), w(3), w(4));
    if ver != 1 || rows != 8 || k != TOP_K || e != NUM_EXPERTS { bail!("replay file: unsupported header v{ver} rows {rows} k {k} e {e}"); }
    let rec = 1 + 2 * rows * k;
    if b.len() != 24 + n * rec { bail!("replay file: size {} != {}", b.len(), 24 + n * rec); }
    Ok((0..n).map(|i| {
        let r = &b[24 + i * rec + 1..24 + (i + 1) * rec];
        (0..rows).map(|row| (0..k).map(|j| i16::from_le_bytes([r[2 * (row * k + j)], r[2 * (row * k + j) + 1]]) as i32).collect()).collect()
    }).collect())
}

/// Replay driver: {real, synthetic at U = round(real U_mean)} routings x {clean, dirty} L2, per-call
/// averages over `chunks` chunks of STEP_LAYERS calls (modes: gate, up, down, down_fast, gateup_fast).
/// Returns the number of failed fast-vs-car byte checks.
#[allow(clippy::too_many_arguments)]
#[rustfmt::skip]
fn replay(g: &dyn GpuBackend, s: u64, kz: &Kerns, layers: &[LayerTabs], b: &Bufs, rows: usize, calls: &[Vec<Vec<i32>>], c_cmp: [DevicePtr; 2], c_gu_cmp: [DevicePtr; 4], rng: &mut Rng) -> Result<usize> {
    let want: usize = std::env::var("FIXCOST_REPLAY_CHUNKS").ok().map_or(Ok(24), |v| v.trim().parse())?;
    let dirty_mb: f64 = std::env::var("FIXCOST_DIRTY_MB").ok().map_or(Ok(6.5), |v| v.trim().parse())?;
    let nbytes = (dirty_mb * 1048576.0) as usize;
    let d_dirty = g.alloc(nbytes.max(1))?;
    let per = STEP_LAYERS.min(calls.len());
    let chunks = (calls.len() / per).clamp(1, want);
    let stride = calls.len() / chunks;
    let (mut bad, mut checked) = (0usize, 0usize);
    // acc[routing: 0 real, 1 synth][l2: 0 clean, 1 dirty] -> per-call mean of the 6 time_u modes; um[routing] = U_mean
    let mut acc = [[[0.0f64; 6]; 2]; 2];
    let sfa = kz.car_s.is_some();
    let mut acc_s = [[[0.0f64; 6]; 2]; 2];
    let mut bits: Vec<(&'static str, Option<usize>)> = Vec::new();
    let mut um = [0.0f64; 2];
    let mut u_syn = 0usize;
    for pass in 0..2 {
        for c in 0..chunks {
            let routs: Vec<Routing> = if pass == 0 {
                (0..per).map(|i| build_routing(g, kz.union, &calls[(c * stride + i) % calls.len()])).collect::<Result<_>>()?
            } else {
                (0..per).map(|_| build_routing(g, kz.union, &union_ids(rng, rows, u_syn))).collect::<Result<_>>()?
            };
            um[pass] += routs.iter().map(|r| r.union as f64).sum::<f64>() / (routs.len() * chunks) as f64;
            for (l2, d) in [None, Some((d_dirty, nbytes))].into_iter().enumerate() {
                // 2026-10-10: Under FIXCOST_SFA the twin runs first on odd chunks, so the off/on order
                // alternates and an order bias cancels over the chunks.
                let on_first = sfa && c % 2 == 1;
                if on_first {
                    let t = time_u(g, s, kz, layers, &routs, b, rows, d, true, true)?;
                    for m in 0..6 { acc_s[pass][l2][m] += t[m] / chunks as f64; }
                }
                let t = time_u(g, s, kz, layers, &routs, b, rows, d, true, false)?;
                for m in 0..6 { acc[pass][l2][m] += t[m] / chunks as f64; }
                if sfa && !on_first {
                    let t = time_u(g, s, kz, layers, &routs, b, rows, d, true, true)?;
                    for m in 0..6 { acc_s[pass][l2][m] += t[m] / chunks as f64; }
                }
            }
            if pass == 0 {
                // Byte check on one real call per chunk: fast vs car.
                let rt = &routs[c % routs.len()];
                if sfa {
                    for (i, (name, m)) in sfa_bits(g, kz, &layers[0], b.x, b.act, &[rt], rows)?.into_iter().enumerate() {
                        if bits.len() <= i { bits.push((name, m)); } else if bits[i].1.is_none() { bits[i].1 = m; }
                    }
                }
                if let Some(fh) = kz.fast {
                    checked += 1;
                    if !down_fast_same(g, kz, fh, b.act, &layers[0].down, c_cmp, rt, rows)? { println!("FAIL replay down_fast chunk {c} union={}: bytes differ from car _c8", rt.union); bad += 1; }
                }
                if let Some(gh) = kz.gu && !gateup_fast_same(g, kz, gh, b.x, &layers[0].gate, &layers[0].up, c_gu_cmp, rt, rows)? {
                    println!("FAIL replay gateup_fast chunk {c} union={}: gate or up bytes differ from car _c8", rt.union);
                    bad += 1;
                }
            }
            for r in &routs { free_routing(g, r)?; }
        }
        if pass == 0 { u_syn = um[0].round() as usize; }
    }
    g.free(d_dirty)?;
    let n_calls = chunks * per;
    let line = |tag: &str, u: f64, t: &[f64; 6]| format!("{tag} U_mean={u:.2} car_gateup_us={:.2} fast_gateup_us={:.2} car_down_us={:.2} fast_down_us={:.2}", t[0] + t[1], t[5], t[2], t[4]);
    println!("REPLAY R={rows} calls={n_calls} U_mean={:.2} car_gateup_us={:.2} fast_gateup_us={:.2} car_down_us={:.2} fast_down_us={:.2} fast_gateup_per_U_us={:.3}",
        um[0], acc[0][0][0] + acc[0][0][1], acc[0][0][5], acc[0][0][2], acc[0][0][4], acc[0][0][5] / um[0]);
    println!("REPLAY_SYNTH R={rows} calls={n_calls} U_mean={:.2} car_gateup_us={:.2} fast_gateup_us={:.2} car_down_us={:.2} fast_down_us={:.2} fast_gateup_per_U_us={:.3}",
        um[1], acc[1][0][0] + acc[1][0][1], acc[1][0][5], acc[1][0][2], acc[1][0][4], acc[1][0][5] / um[1]);
    for (pass, rname) in ["real", "synth"].iter().enumerate() {
        for (l2, lname) in ["clean", "dirty"].iter().enumerate() {
            println!("{}", line(&format!("REPLAY2 routing={rname} l2={lname}"), um[pass], &acc[pass][l2]));
        }
    }
    if sfa {
        for (pass, rname) in ["real", "synth"].iter().enumerate() {
            for (l2, lname) in ["clean", "dirty"].iter().enumerate() { print_sfa(rows, rname, lname, &acc[pass][l2], &acc_s[pass][l2]); }
        }
        bad += print_bits(rows, &format!("replay: one real routing per chunk, {chunks} chunks"), &bits);
    }
    println!("CHECK replay fast==car sampled {checked} calls (bad {bad}); dirty_mb={dirty_mb} chunks={chunks}");
    Ok(bad)
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
    let fast = g.kernel("w4a16_gemv", &format!("w4a16_gemv_sw_moe_batchm_down_m{rows}")).ok();
    if fast.is_none() { println!("NOTE no w4a16_gemv_sw_moe_batchm_down_m{rows}: down_fast columns and checks skipped"); }
    let gu = g.kernel("w4a16_gemv", &format!("w4a16_gemv_sw_moe_batchm_gateup_m{rows}")).ok();
    if gu.is_none() { println!("NOTE no w4a16_gemv_sw_moe_batchm_gateup_m{rows}: gateup_fast columns and checks skipped"); }
    let mut hs = Vec::new();
    for n in &names {
        match g.kernel("w4a16_gemv", n) { Ok(h) => hs.push(h), Err(_) => { println!("MISSING kernel entry {n}"); return Ok(2); } }
    }
    let (mut car_s, mut fast_s, mut gu_s, mut sw, mut sw_s) = (None, None, None, None, None);
    if sfa_on() {
        // 2026-10-10: FIXCOST_SFA: the _sfa twins of every entry timed here must exist.
        let look = |n: String| -> Option<KernelHandle> { let h = g.kernel("w4a16_gemv", &n).ok(); if h.is_none() { println!("MISSING kernel entry {n}"); } h };
        car_s = look(format!("w4a16_gemv_sw_moe_batchm_sfa_m{rows}_c8"));
        if fast.is_some() { fast_s = look(format!("w4a16_gemv_sw_moe_batchm_down_sfa_m{rows}")); }
        if gu.is_some() { gu_s = look(format!("w4a16_gemv_sw_moe_batchm_gateup_sfa_m{rows}")); }
        sw = look("w4a16_gemv_sw_moe".into());
        sw_s = look("w4a16_gemv_sw_moe_sfa".into());
        if car_s.is_none() || sw.is_none() || sw_s.is_none() || (fast.is_some() && fast_s.is_none()) || (gu.is_some() && gu_s.is_none()) { return Ok(2); }
        let probe: Vec<u8> = (0..128 * 64).map(|i| i as u8).collect();
        let mut seen = to_atom(&probe, 128, 1024);
        seen.sort_unstable();
        let mut want = probe;
        want.sort_unstable();
        if seen != want { bail!("SFA atom permutation is not a bijection"); }
        println!("SFA atom layout self-test ok (offset formula is a permutation of the 128x64 probe)");
    }
    let kz = Kerns { core: hs[0], car: hs[1], union: hs[2], fast, gu, car_s, fast_s, gu_s, sw, sw_s };
    let mut rng = Rng(0x006d_6f65_6669_7863);
    let mut bad = check(g, &kz, &mut rng, rows)?;
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
        c_up: g.alloc(rows * TOP_K * MI * 2)?,
        c_dn: g.alloc(rows * TOP_K * HIDDEN * 2)?,
    };
    let c_cmp = [g.alloc(rows * TOP_K * HIDDEN * 2)?, g.alloc(rows * TOP_K * HIDDEN * 2)?];
    let gu_bytes = rows * TOP_K * MI * 2;
    let c_gu_cmp = [g.alloc(gu_bytes)?, g.alloc(gu_bytes)?, g.alloc(gu_bytes)?, g.alloc(gu_bytes)?];
    let (mut us, mut ts) = (Vec::new(), Vec::new());
    for &u in &unions {
        let routs: Vec<Routing> = (0..ROUTINGS)
            .map(|_| build_routing(g, kz.union, &union_ids(&mut rng, rows, u)))
            .collect::<Result<_>>()?;
        let t = time_u(g, s, &kz, &layers, &routs, &b, rows, None, false, false)?;
        if kz.car_s.is_some() {
            // 2026-10-10: ABBA (off, on, on, off), each arm the mean of its two runs.
            let t2a = time_u(g, s, &kz, &layers, &routs, &b, rows, None, false, true)?;
            let t2b = time_u(g, s, &kz, &layers, &routs, &b, rows, None, false, true)?;
            let tb = time_u(g, s, &kz, &layers, &routs, &b, rows, None, false, false)?;
            let mut t1 = t;
            let mut t2 = t2a;
            for m in 0..6 { t1[m] = (t[m] + tb[m]) / 2.0; t2[m] = (t2a[m] + t2b[m]) / 2.0; }
            print_sfa(rows, &format!("U={u}"), "clean", &t1, &t2);
            let rr: Vec<&Routing> = routs.iter().collect();
            bad += print_bits(rows, &format!("U={u}, {ROUTINGS} routings"), &sfa_bits(g, &kz, &layers[0], b.x, b.act, &rr, rows)?);
        }
        let mu = routs.iter().map(|r| r.union as f64).sum::<f64>() / ROUTINGS as f64;
        let gbs = |us: f64| mu * (MI * HIDDEN) as f64 * BYTES_PER_W / us / 1e3;
        println!("U={u} R={rows} gate_us={:.2} up_us={:.2} down_us={:.2} seq_us={:.2} gate_GBs={:.0} up_GBs={:.0} down_GBs={:.0} down_fast_us={:.2} down_fast_GBs={:.0} gateup_fast_us={:.2} gateup_fast_GBs={:.0}",
            t[0], t[1], t[2], t[3], gbs(t[0]), gbs(t[1]), gbs(t[2]), t[4], gbs(t[4]), t[5], 2.0 * gbs(t[5]));
        if let Some(fh) = kz.fast {
            let mut all = true;
            for (i, rt) in routs.iter().enumerate() {
                if !down_fast_same(g, &kz, fh, b.act, &layers[0].down, c_cmp, rt, rows)? {
                    println!("FAIL down_fast U={u} R={rows} routing {i} union={}: bytes differ from car _c8", rt.union);
                    bad += 1;
                    all = false;
                }
            }
            println!("CHECK down_fast U={u} R={rows} fast==car {all} ({ROUTINGS} routings)");
        }
        if let Some(gh) = kz.gu {
            let mut all = true;
            for (i, rt) in routs.iter().enumerate() {
                if !gateup_fast_same(g, &kz, gh, b.x, &layers[0].gate, &layers[0].up, c_gu_cmp, rt, rows)? {
                    println!("FAIL gateup_fast U={u} R={rows} routing {i} union={}: gate or up bytes differ from car _c8", rt.union);
                    bad += 1;
                    all = false;
                }
            }
            println!("CHECK gateup_fast U={u} R={rows} fast==car {all} ({ROUTINGS} routings)");
        }
        us.push(mu);
        ts.push(t);
        for r in &routs { free_routing(g, r)?; }
    }
    let mut replay_note = String::new();
    if let Ok(v) = std::env::var("FIXCOST_REPLAY") {
        if rows != 8 { println!("NOTE FIXCOST_REPLAY needs MOE_MT_ROWS=8: replay skipped"); } else {
            let file = if matches!(v.trim(), "" | "1" | "embedded") { EMBEDDED_REPLAY.to_vec() } else { std::fs::read(v.trim())? };
            let calls = parse_replay(&file)?;
            if calls.is_empty() { bail!("replay file has no calls"); }
            bad += replay(g, s, &kz, &layers, &b, rows, &calls, c_cmp, c_gu_cmp, &mut rng)?;
            replay_note = format!("; replay: fast entries equal car _c8 on sampled real routings ({} calls in file)", calls.len());
        }
    }
    let sfa_note = if kz.car_s.is_some() {
        bad += sfa_r1(g, s, &kz, &layers, &b, &mut rng)?;
        if bad == 0 { println!("PASS - SFA: every _sfa twin (batchm _c8 gate/up/down, down_fast, gateup_fast, sw_moe R1) bytes equal the original on every checked routing; timings in the SFA lines above"); }
        "; _sfa twins bitwise equal"
    } else { "" };
    println!("FIT least squares over U>=8 (us vs union experts)");
    let sel: Vec<usize> = (0..us.len()).filter(|&i| us[i] >= 8.0).collect();
    if sel.len() >= 2 {
        let x: Vec<f64> = sel.iter().map(|&i| us[i]).collect();
        for (m, name) in ["gate", "up", "down", "seq", "down_fast", "gateup_fast"].iter().enumerate() {
            if (m == 4 && kz.fast.is_none()) || (m == 5 && kz.gu.is_none()) { continue; }
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
    let mut fp = if kz.fast.is_some() { format!("; down_m{rows} bytes equal car _c8 on every routing") } else { String::new() };
    if kz.gu.is_some() { fp.push_str(&format!("; gateup_m{rows} gate and up bytes equal car _c8 on every routing")); }
    fp.push_str(&replay_note);
    fp.push_str(sfa_note);
    if bad == 0 { println!("PASS: car _m{rows}_c8 bytes equal _m{rows} on gate/up and down{fp}; sweep done"); Ok(0) } else { println!("FAIL: {bad} byte check(s) failed (car vs core, down_fast vs car, gateup_fast vs car)"); Ok(1) }
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
