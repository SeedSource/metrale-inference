// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Gate for `METRALE_GLM_MOE_BATCHM_COLS`: `w4a16_gemv_sw_moe_batchm_m<R>_c<J>`
//! (J = 2, 4, 8 columns per warp) against `w4a16_gemv_sw_moe_batchm_m<R>` (J = 1) on the
//! GLM-5.3 expert-TP per-rank routed-MoE shapes (hidden 4096, 288 experts, top_k 8, mi 1024).
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Checks:
//! - (a) Exactness: for R = 2..=8, J in {2, 4, 8}, gate/up-like (N 1024, K 4096, shared input),
//!   down (N 4096, K 1024, per-(row, slot) input) and a column-tail shape (N 1000, K 4096), over
//!   random top-8 routings (4 seeds), all rows on the same 8 experts and fully disjoint rows:
//!   the union comes from `glm5next_moe_row_union`; both entries write a buffer poisoned with
//!   0xA5 and the whole buffers are compared byte for byte (so unused (row, slot) outputs must
//!   be left untouched by both).
//! - (b) Timing: one layer = gate + up + down launches over a cold pool (3 layers x 3 matrices
//!   x 288 distinct experts, ~6 GB), layers alternate inside a CUDA graph of 42 layers; R in
//!   {3, 8}, random top-8 unions; per-launch and per-layer us for J = 1, 2, 4, 8.
//! - (c) `GATE exact` and `GATE speed` lines, then `PASS:` (exit 0) or `FAIL` (exit 1);
//!   exit 2 if a kernel entry is missing.
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_moe_batchm_cols_microtest

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const HIDDEN: usize = 4096;
const MI: usize = 1024;
const TOP_K: usize = 8;
const NUM_EXPERTS: usize = 288;
const JS: [usize; 4] = [1, 2, 4, 8];
const LAYERS: usize = 3;
const ROUTINGS: usize = 4;
const STEP_LAYERS: usize = 42;
const GO_R8: f64 = 4.0;
const KILL_R8: f64 = 2.0;
const POISON: u8 = 0xA5;

struct Rng(u64);
#[rustfmt::skip]
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn u(&mut self) -> f32 { (self.next() & 0xFF_FFFF) as f32 / 16_777_216.0 }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn le<T: Copy, const B: usize>(v: &[T], f: impl Fn(T) -> [u8; B]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}
fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn bf16_rand(rng: &mut Rng, n: usize) -> Vec<u8> {
    let v: Vec<bf16> = (0..n)
        .map(|_| bf16::from_f32(rng.u() * 2.0 - 1.0))
        .collect();
    le(&v, |x| x.to_bits().to_le_bytes())
}

/// Host NVFP4 weight bytes for an [n, k] expert (random nibbles; E4M3 scale codes 1.0..1.875,
/// never zero, inf or NaN) with 64 spare bytes so uploads can start at distinct offsets.
struct Host {
    w: Vec<u8>,
    s: Vec<u8>,
    wb: usize,
    sb: usize,
}
#[rustfmt::skip]
fn gen_host(rng: &mut Rng, n: usize, k: usize) -> Host {
    let (wb, sb) = (n * k / 2, n * k / 16);
    Host {
        w: (0..wb + 64).map(|_| rng.next() as u8).collect(),
        s: (0..sb + 64).map(|_| 0x38 | (rng.next() as u8 & 7)).collect(),
        wb, sb,
    }
}

/// The 288-entry pointer table of one matrix; entries cycle through `distinct` real buffers.
struct Table {
    packed: DevicePtr,
    scale: DevicePtr,
    scale2: DevicePtr,
}
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

/// Union tables of `ids` ([rows][TOP_K]) built by `glm5next_moe_row_union` (one block of
/// rows * top_k threads, as `forward/moe_experts.rs` launches it).
struct Routing {
    ueid: DevicePtr,
    uslot: DevicePtr,
    union: usize,
}
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
    let mut d = flat;
    d.sort_unstable();
    d.dedup();
    Ok(Routing { ueid, uslot, union: d.len() })
}

fn random_ids(rng: &mut Rng, rows: usize) -> Vec<Vec<i32>> {
    (0..rows)
        .map(|_| {
            let mut p: Vec<i32> = (0..NUM_EXPERTS as i32).collect();
            for i in 0..TOP_K {
                let j = i + rng.next() as usize % (NUM_EXPERTS - i);
                p.swap(i, j);
            }
            p[..TOP_K].to_vec()
        })
        .collect()
}

/// One batchm launch: `n`, `kk` the GEMV shape; `down` selects the per-(row, slot) input layout.
#[allow(clippy::too_many_arguments)]
#[rustfmt::skip]
fn launch(
    g: &dyn GpuBackend, kern: KernelHandle, j: usize, a: DevicePtr, t: &Table, c: DevicePtr,
    rt: &Routing, rows: usize, n: usize, kk: usize, down: bool, s: u64,
) -> Result<()> {
    // gate/up: a_row_stride = hidden, a_slot_stride = 0; down: a_row_stride = top_k * mi,
    // a_slot_stride = mi. c_row_stride = top_k * n in both (gate/up n = mi, down n = hidden).
    let (ars, ass) = if down { (TOP_K * MI, MI) } else { (kk, 0) };
    KernelLaunch::new(g, kern)
        .grid([n.div_ceil(8 * j) as u32, (rows * TOP_K) as u32, 1]).block([256, 1, 1])
        .arg_ptr(a).arg_ptr(t.packed).arg_ptr(t.scale).arg_ptr(t.scale2).arg_ptr(c)
        .arg_ptr(rt.ueid).arg_ptr(rt.uslot)
        .arg_u32(n as u32).arg_u32(kk as u32).arg_u32(NUM_EXPERTS as u32)
        .arg_u32(ars as u32).arg_u32(ass as u32).arg_u32((TOP_K * n) as u32)
        .launch(s)
}

/// `kern[ji][r - 2]` for J = JS[ji].
type Kerns = Vec<Vec<KernelHandle>>;

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
    let reps = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..reps { g.launch_graph(graph, s)?; }
    g.synchronize(s)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    g.destroy_graph(graph)?;
    Ok(ms)
}

#[rustfmt::skip]
struct LayerTabs { gate: Table, up: Table, down: Table }

/// (a) Returns (compares, failures).
#[rustfmt::skip]
fn exactness(g: &dyn GpuBackend, kk: &Kerns, k_union: KernelHandle, rng: &mut Rng) -> Result<(usize, usize)> {
    // (label, n, K, down)
    let shapes = [
        ("gate/up", MI, HIDDEN, false),
        ("down", HIDDEN, MI, true),
        ("tail N=1000", 1000, HIDDEN, false),
    ];
    let (mut total, mut fails) = (0usize, 0usize);
    for (label, n, k, down) in shapes {
        let t = make_table(g, &gen_host(rng, n, k), 4)?;
        for rows in 2..=8usize {
            let mut cases: Vec<(String, Vec<Vec<i32>>)> = (0..4)
                .map(|s| (format!("random{s}"), random_ids(rng, rows)))
                .collect();
            let base = random_ids(rng, 1).remove(0);
            cases.push((
                "same8".into(),
                (0..rows)
                    .map(|r| (0..TOP_K).map(|i| base[(i + r) % TOP_K]).collect())
                    .collect(),
            ));
            cases.push((
                "disjoint".into(),
                (0..rows)
                    .map(|r| {
                        (0..TOP_K)
                            .map(|i| ((5 + r * TOP_K + i) % NUM_EXPERTS) as i32)
                            .collect()
                    })
                    .collect(),
            ));
            let a_elems = if down {
                rows * TOP_K * MI
            } else {
                rows * HIDDEN
            };
            let d_a = up(g, &bf16_rand(rng, a_elems))?;
            let bytes = rows * TOP_K * n * 2;
            let (d_ref, d_new) = (g.alloc(bytes)?, g.alloc(bytes)?);
            let go = |ji: usize, c: DevicePtr, rt: &Routing| {
                launch(g, kk[ji][rows - 2], JS[ji], d_a, &t, c, rt, rows, n, k, down, 0)
            };
            for (tag, ids) in &cases {
                let rt = build_routing(g, k_union, ids)?;
                g.memset_async(d_ref, POISON, bytes, 0)?;
                go(0, d_ref, &rt)?;
                g.synchronize(0)?;
                let r_ref = dn(g, d_ref, bytes)?;
                for (ji, &j) in JS.iter().enumerate().skip(1) {
                    g.memset_async(d_new, POISON, bytes, 0)?;
                    go(ji, d_new, &rt)?;
                    g.synchronize(0)?;
                    let r_new = dn(g, d_new, bytes)?;
                    total += 1;
                    if r_ref != r_new {
                        fails += 1;
                        let diff = r_ref.iter().zip(&r_new).filter(|(a, b)| a != b).count();
                        let first = r_ref.iter().zip(&r_new).position(|(a, b)| a != b);
                        if fails <= 12 {
                            println!(
                                "FAIL exact {label} R={rows} J={j} {tag} union={} \
                                 {diff}/{bytes} bytes differ, first at {first:?}",
                                rt.union
                            );
                        }
                    }
                }
                g.free(rt.ueid)?;
                g.free(rt.uslot)?;
            }
            println!(
                "exact {label:<11} R={rows}: {} routings x J{{2,4,8}} compared (running fails {fails})",
                cases.len()
            );
            for p in [d_a, d_ref, d_new] {
                g.free(p)?;
            }
        }
    }
    Ok((total, fails))
}

/// (b) Returns, for each J in JS, the (gate, up, down, layer) us per layer at `rows` rows,
/// and the mean union size.
fn timing(
    g: &dyn GpuBackend,
    kk: &Kerns,
    k_union: KernelHandle,
    layers: &[LayerTabs],
    rng: &mut Rng,
    rows: usize,
) -> Result<(Vec<[f64; 4]>, f64)> {
    let s = g.create_stream()?;
    let routs: Vec<Routing> = (0..ROUTINGS)
        .map(|_| build_routing(g, k_union, &random_ids(rng, rows)))
        .collect::<Result<_>>()?;
    let mean_union = routs.iter().map(|r| r.union as f64).sum::<f64>() / ROUTINGS as f64;
    let d_x = up(g, &bf16_rand(rng, rows * HIDDEN))?;
    let d_act = up(g, &bf16_rand(rng, rows * TOP_K * MI))?;
    let c_gu = g.alloc(rows * TOP_K * MI * 2)?;
    let c_dn = g.alloc(rows * TOP_K * HIDDEN * 2)?;
    let mut out = Vec::new();
    for (ji, &j) in JS.iter().enumerate() {
        let kern = kk[ji][rows - 2];
        // mode 0 gate, 1 up, 2 down, 3 all three.
        let run = |mode: usize| -> Result<f64> {
            let ms = time_graph(g, s, &mut || {
                for l in 0..STEP_LAYERS {
                    let (lt, rt) = (&layers[l % LAYERS], &routs[l % ROUTINGS]);
                    if mode == 0 || mode == 3 {
                        launch(
                            g, kern, j, d_x, &lt.gate, c_gu, rt, rows, MI, HIDDEN, false, s,
                        )?;
                    }
                    if mode == 1 || mode == 3 {
                        launch(
                            g, kern, j, d_x, &lt.up, c_gu, rt, rows, MI, HIDDEN, false, s,
                        )?;
                    }
                    if mode == 2 || mode == 3 {
                        launch(
                            g, kern, j, d_act, &lt.down, c_dn, rt, rows, HIDDEN, MI, true, s,
                        )?;
                    }
                }
                Ok(())
            })?;
            Ok(ms * 1e3 / STEP_LAYERS as f64)
        };
        out.push([run(0)?, run(1)?, run(2)?, run(3)?]);
    }
    for r in &routs {
        g.free(r.ueid)?;
        g.free(r.uslot)?;
    }
    for p in [d_x, d_act, c_gu, c_dn] {
        g.free(p)?;
    }
    Ok((out, mean_union))
}

fn run() -> Result<i32> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut missing = Vec::new();
    let mut kk: Kerns = Vec::new();
    for &j in &JS {
        let mut v = Vec::new();
        for r in 2..=8 {
            let name = if j == 1 {
                format!("w4a16_gemv_sw_moe_batchm_m{r}")
            } else {
                format!("w4a16_gemv_sw_moe_batchm_m{r}_c{j}")
            };
            match g.kernel("w4a16_gemv", &name) {
                Ok(h) => v.push(h),
                Err(_) => {
                    missing.push(name);
                    v.push(KernelHandle(0));
                }
            }
        }
        kk.push(v);
    }
    let k_union = g.kernel("w4a16_gemv", "glm5next_moe_row_union");
    if k_union.is_err() {
        missing.push("glm5next_moe_row_union".into());
    }
    if !missing.is_empty() {
        for m in &missing {
            println!("MISSING kernel entry {m}");
        }
        println!("FAIL kernel entries missing ({})", missing.len());
        return Ok(2);
    }
    let k_union = k_union?;
    let mut rng = Rng(0x006d_6f65_636f_6c73);

    let (total, fails) = exactness(g, &kk, k_union, &mut rng)?;
    let exact_ok = fails == 0;

    // Cold pool: LAYERS layers x (gate, up, down) x 288 distinct experts.
    let (hgu, hdn) = (
        gen_host(&mut rng, MI, HIDDEN),
        gen_host(&mut rng, HIDDEN, MI),
    );
    let mut layers = Vec::new();
    for _ in 0..LAYERS {
        layers.push(LayerTabs {
            gate: make_table(g, &hgu, NUM_EXPERTS)?,
            up: make_table(g, &hgu, NUM_EXPERTS)?,
            down: make_table(g, &hdn, NUM_EXPERTS)?,
        });
    }
    // saves[ri][ji] = ms/step saved vs J = 1 at R = [3, 8][ri].
    let mut saves = [[0.0f64; 4]; 2];
    for (ri, rows) in [3usize, 8].into_iter().enumerate() {
        let (res, mu) = timing(g, &kk, k_union, &layers, &mut rng, rows)?;
        let base = res[0][3];
        for (ji, &j) in JS.iter().enumerate() {
            let [ga, upv, dw, ly] = res[ji];
            let sv = (base - ly) * STEP_LAYERS as f64 / 1e3;
            saves[ri][ji] = sv;
            println!(
                "TIMING R={rows} union {mu:.1} J={j} gate {ga:.1} up {upv:.1} down {dw:.1} layer {ly:.1} \
                 (x{STEP_LAYERS} = {:.3} ms/step) saves {sv:.3} ms/step vs J=1",
                ly * STEP_LAYERS as f64 / 1e3
            );
        }
    }

    println!(
        "GATE exact {} ({total} compares)",
        if exact_ok { "PASS" } else { "FAIL" }
    );
    // Best J: maximize the R=8 saving among J whose R=3 saving is >= 0; else best by R=8.
    let pick = |only_ok: bool| {
        (1..4)
            .filter(|&ji| !only_ok || saves[0][ji] >= 0.0)
            .max_by(|&a, &b| saves[1][a].total_cmp(&saves[1][b]))
    };
    let (best, qualified) = match pick(true) {
        Some(b) => (b, true),
        None => (pick(false).unwrap_or(1), false),
    };
    let (s8, s3) = (saves[1][best], saves[0][best]);
    let verdict = if s8 < KILL_R8 {
        "KILL"
    } else if qualified && s8 >= GO_R8 && s3 >= 0.0 {
        "GO"
    } else {
        "GREY"
    };
    println!(
        "GATE speed best J={}: R=8 saves {s8:.3} ms/step, R=3 saves {s3:.3} ms/step \
         (GO needs R=8 >= {GO_R8:.1} and R=3 >= 0.0; KILL if R=8 < {KILL_R8:.1}): {verdict}",
        JS[best]
    );
    if exact_ok {
        println!(
            "PASS: batchm _c<J> entries (J 2,4,8; R 2..=8) byte-equal to J=1 on {total} compares \
             (gate/up, down, N=1000 tail)"
        );
        Ok(0)
    } else {
        println!("FAIL exact: {fails}/{total} compares differ");
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
