// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Gate for `METRALE_GLM_MOE_CSF_SCALES` (csfcar step 3): routed-expert block scales
//! held compressed ("scale rows with escape", `glm5next_mlp::scale_rows`) against plain scales,
//! on the GLM-5.3 expert-TP per-rank shapes (hidden 4096, 288 experts, top_k 8, mi 1024:
//! gate/up [N 1024, K 4096], down [N 4096, K 1024]).
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Checks:
//! - (a) Exactness: every kernel family that reads scales through `moe_scale_byte` runs twice on
//!   identical inputs (same packed buffers), once with the plain scale-pointer table and once
//!   with the encoded one (tagged pointers); outputs go to buffers poisoned with 0xA5 and the
//!   whole buffers are compared byte for byte. Families: `w4a16_gemv_sw_moe` (1 row),
//!   `w4a16_gemv_sw_moe_batchm_gateup_m<R>` and `_down_m<R>` (R 2..=8),
//!   `w4a16_gemv_sw_moe_batchm_m<R>` and `_m<R>_c8` (R 2, 4, 8; gate/up-like and down-like
//!   shapes) and the va2 grouped GEMM (`moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_va2`, M rows
//!   per expert 17, 40, 64 over several experts). Random top-8 routings, 4 seeds. Scale bytes are
//!   synthetic but realistic: a per-row band base in 0x58..0x68, values base + (0..13), ~0.6%
//!   outliers anywhere in 0..=0x7E, so escapes occur.
//! - (b) Timing (the stop rule): one decode layer per launch set as production does it: R = 1:
//!   `w4a16_gemv_sw_moe` for gate, up, down; R in {4, 8}: `glm5next_moe_row_union` +
//!   `batchm_gateup_m<R>` + `batchm_down_m<R>`. Cold pool of 3 layers x 288 experts (layers
//!   alternate inside a CUDA graph of 42 layers); plain and encoded graphs are replayed
//!   interleaved; median per-layer us over >= 20 timed replays each.
//! - (c) `GATE exact`, `GATE speed R=1 ratio .. R=4 ratio .. R=8 ratio ..`, then
//!   `PASS: csf not slower at R=1 and R=4` (both median paired ratios <= 1.005, exact PASS;
//!   exit 0) or
//!   `STOP: csf slower per call at R=<..>` (exit 1); exactness FAIL exits 1; a missing kernel
//!   entry exits 2.
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_moe_csf_scales_microtest

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mlp::scale_rows;

const HIDDEN: usize = 4096;
const MI: usize = 1024;
const TOP_K: usize = 8;
const NUM_EXPERTS: usize = 288;
const LAYERS: usize = 3;
const ROUTINGS: usize = 4;
const STEP_LAYERS: usize = 42;
const SEEDS: usize = 4;
const WARMUP: usize = 5;
const REPS: usize = 25;
const POISON: u8 = 0xA5;
/// 2026-10-17: The stop rule's "slower": median paired ratio encoded/plain above this, fixed
/// before any data (0.5% = inside run-to-run graph noise is "not slower").
const SLOWER: f64 = 1.005;
/// Experts a distinct-buffer table cycles through in the exactness runs.
const EXACT_DISTINCT: usize = 16;

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

/// Host packed NVFP4 bytes of an [n, k] expert (random) with 64 spare bytes so uploads can start
/// at distinct offsets.
fn gen_packed(rng: &mut Rng, n: usize, k: usize) -> Vec<u8> {
    (0..n * k / 2 + 64).map(|_| rng.next() as u8).collect()
}

/// Realistic plain scales [n, c]: per row a band base in 0x58..0x68, values base + (0..13), and
/// ~0.6% outliers anywhere in 0..=0x7E (E4M3 positive finite).
fn gen_scales(rng: &mut Rng, n: usize, c: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * c);
    for _ in 0..n {
        let base = 0x58 + (rng.next() % 16) as u8;
        for _ in 0..c {
            let r = rng.next();
            if r % 1000 < 6 {
                v.push(((r >> 10) % 0x7F) as u8);
            } else {
                v.push(base + ((r >> 10) % 14) as u8);
            }
        }
    }
    v
}

/// The 288-entry pointer tables of one matrix, both scale formats over the same packed buffers.
struct Table {
    packed: DevicePtr,
    plain: DevicePtr,
    enc: DevicePtr,
    scale2: DevicePtr,
    /// Scale bytes summed over the 288 entries, plain and encoded.
    plain_bytes: usize,
    enc_bytes: usize,
}
impl Table {
    fn scale(&self, enc: bool) -> DevicePtr {
        if enc { self.enc } else { self.plain }
    }
}

/// Entries cycle through `distinct` real buffers. `packed` is `gen_packed(n, k)`.
fn make_table(
    g: &dyn GpuBackend,
    rng: &mut Rng,
    packed: &[u8],
    n: usize,
    k: usize,
    distinct: usize,
) -> Result<Table> {
    let (wb, c) = (n * k / 2, k / 16);
    let (mut pp, mut sp, mut se, mut s2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut bufs: Vec<(u64, u64, u64, usize, usize)> = Vec::new();
    let (mut plain_bytes, mut enc_bytes) = (0usize, 0usize);
    for e in 0..NUM_EXPERTS {
        if e < distinct {
            let off = (e * 17) % 64;
            let plain = gen_scales(rng, n, c);
            let enc = scale_rows::encode(n, c, &plain)?;
            let pptr = up(g, &packed[off..off + wb])?.0;
            let sptr = up(g, &plain)?.0;
            let eptr = enc.device_ptr(up(g, &enc.bytes)?.0);
            bufs.push((pptr, sptr, eptr, plain.len(), enc.bytes.len()));
        }
        let (p, s, t, pb, eb) = bufs[e % distinct];
        pp.push(p);
        sp.push(s);
        se.push(t);
        plain_bytes += pb;
        enc_bytes += eb;
        s2.push(1.0f32 + e as f32 * 0.01);
    }
    Ok(Table {
        packed: up(g, &le(&pp, |x: u64| x.to_le_bytes()))?,
        plain: up(g, &le(&sp, |x: u64| x.to_le_bytes()))?,
        enc: up(g, &le(&se, |x: u64| x.to_le_bytes()))?,
        scale2: up(g, &le(&s2, |x: f32| x.to_le_bytes()))?,
        plain_bytes,
        enc_bytes,
    })
}

/// Union tables of `ids` ([rows][TOP_K]) built by `glm5next_moe_row_union` (one block of
/// rows * top_k threads, as `forward/moe_experts.rs` launches it).
struct Routing {
    ueid: DevicePtr,
    uslot: DevicePtr,
}
#[rustfmt::skip]
fn build_routing(g: &dyn GpuBackend, k_union: KernelHandle, ids: &[Vec<i32>]) -> Result<Routing> {
    let rows = ids.len();
    let flat: Vec<i32> = ids.iter().flatten().copied().collect();
    let d_ids = up(g, &le(&flat, |x: i32| x.to_le_bytes()))?;
    let ueid = g.alloc(rows * TOP_K * 4)?;
    let uslot = g.alloc(rows * TOP_K * rows * 4)?;
    launch_union(g, k_union, d_ids, ueid, uslot, rows, 0)?;
    g.synchronize(0)?;
    g.free(d_ids)?;
    Ok(Routing { ueid, uslot })
}

/// forward/moe_experts.rs:261-269 (`row_batched_experts`): one block of rows * top_k threads.
fn launch_union(
    g: &dyn GpuBackend,
    k_union: KernelHandle,
    ids: DevicePtr,
    ueid: DevicePtr,
    uslot: DevicePtr,
    rows: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k_union)
        .grid([1, 1, 1])
        .block([(rows * TOP_K) as u32, 1, 1])
        .arg_ptr(ids)
        .arg_ptr(ueid)
        .arg_ptr(uslot)
        .arg_u32(rows as u32)
        .arg_u32(TOP_K as u32)
        .launch(s)
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

struct Kerns {
    sw1: KernelHandle,
    union: KernelHandle,
    /// Indexed `[r - 2]`.
    gateup: Vec<KernelHandle>,
    down: Vec<KernelHandle>,
    /// `bm1[i]` / `bm8[i]` for R = [2, 4, 8][i].
    bm1: Vec<KernelHandle>,
    bm8: Vec<KernelHandle>,
    va2: KernelHandle,
}
const BM_RS: [usize; 3] = [2, 4, 8];

/// launch.rs:152-183 (`w4a16_gemv_moe`): one row, device-dispatched; grid (ceil(n / 8), top_k).
#[allow(clippy::too_many_arguments)]
fn l_sw1(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &Table,
    enc: bool,
    c: DevicePtr,
    ids: DevicePtr,
    n: usize,
    kk: usize,
    input_stride: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(8) as u32, TOP_K as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale(enc))
        .arg_ptr(t.scale2)
        .arg_ptr(c)
        .arg_ptr(ids)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(input_stride as u32)
        .launch(s)
}

/// launch.rs:206-253 (`w4a16_gemv_moe_batchm`): `gx` is the grid x of the entry (n / 8 for
/// `_m<R>`, n / (8 J) for `_c<J>`, n / 128 for `_down_m<R>`); `down` selects the per-(row, slot)
/// input layout (moe_experts.rs:345-363 for down, 274-295 for gate/up strides).
#[allow(clippy::too_many_arguments)]
fn l_batchm(
    g: &dyn GpuBackend,
    k: KernelHandle,
    gx: usize,
    a: DevicePtr,
    t: &Table,
    enc: bool,
    c: DevicePtr,
    rt: &Routing,
    rows: usize,
    n: usize,
    kk: usize,
    down: bool,
    s: u64,
) -> Result<()> {
    let (ars, ass) = if down { (TOP_K * MI, MI) } else { (kk, 0) };
    KernelLaunch::new(g, k)
        .grid([gx as u32, (rows * TOP_K) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale(enc))
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

/// launch.rs:344-391 (`w4a16_gemv_moe_batchm_gateup`), args from moe_experts.rs:274-292: grid x
/// ceil(mi / 64).
#[allow(clippy::too_many_arguments)]
fn l_gateup(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    gate: &Table,
    upt: &Table,
    enc: bool,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    ueid: DevicePtr,
    uslot: DevicePtr,
    rows: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([MI.div_ceil(64) as u32, (rows * TOP_K) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate.packed)
        .arg_ptr(gate.scale(enc))
        .arg_ptr(gate.scale2)
        .arg_ptr(upt.packed)
        .arg_ptr(upt.scale(enc))
        .arg_ptr(upt.scale2)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(ueid)
        .arg_ptr(uslot)
        .arg_u32(MI as u32)
        .arg_u32(HIDDEN as u32)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(HIDDEN as u32)
        .arg_u32(0)
        .arg_u32((TOP_K * MI) as u32)
        .launch(s)
}

/// The down projection's entry for the union sweep: `_down_m<R>` (launch.rs:224-228, grid x
/// n / 128 = MOE_DOWN_FAST_COLS).
#[allow(clippy::too_many_arguments)]
fn l_down(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &Table,
    enc: bool,
    c: DevicePtr,
    ueid: DevicePtr,
    uslot: DevicePtr,
    rows: usize,
    s: u64,
) -> Result<()> {
    let rt = Routing { ueid, uslot };
    l_batchm(
        g,
        k,
        HIDDEN.div_ceil(128),
        a,
        t,
        enc,
        c,
        &rt,
        rows,
        HIDDEN,
        MI,
        true,
        s,
    )
}

/// dispatch.rs:26-54 (`grouped_gemm`) with the va2 tile (tile.rs:159-165: m_tile 128, n_tile 64,
/// 256 threads, not m_fast): grid `[ceil(n / 64), max_m_tiles, num_experts]`.
#[allow(clippy::too_many_arguments)]
fn l_grouped(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &Table,
    enc: bool,
    c: DevicePtr,
    offsets: DevicePtr,
    sorted: DevicePtr,
    n: usize,
    kk: usize,
    max_m_tiles: u32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(64) as u32, max_m_tiles, NUM_EXPERTS as u32])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale(enc))
        .arg_ptr(t.scale2)
        .arg_ptr(c)
        .arg_ptr(offsets)
        .arg_ptr(sorted)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(0)
}

struct Stats {
    total: usize,
    fails: usize,
}

/// Runs `f(enc, outs)` for plain then encoded into poisoned buffers of `sizes` bytes each and
/// compares every buffer byte for byte.
fn pair(
    g: &dyn GpuBackend,
    st: &mut Stats,
    label: &str,
    sizes: &[usize],
    f: &dyn Fn(bool, &[DevicePtr]) -> Result<()>,
) -> Result<()> {
    let mut res: Vec<Vec<Vec<u8>>> = Vec::new();
    for enc in [false, true] {
        let outs: Vec<DevicePtr> = sizes.iter().map(|&b| g.alloc(b)).collect::<Result<_>>()?;
        for (p, &b) in outs.iter().zip(sizes) {
            g.memset_async(*p, POISON, b, 0)?;
        }
        f(enc, &outs)?;
        g.synchronize(0)?;
        let mut r = Vec::new();
        for (p, &b) in outs.iter().zip(sizes) {
            r.push(dn(g, *p, b)?);
            g.free(*p)?;
        }
        res.push(r);
    }
    for (i, (a, b)) in res[0].iter().zip(&res[1]).enumerate() {
        st.total += 1;
        if a != b {
            st.fails += 1;
            if st.fails <= 12 {
                let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
                let first = a.iter().zip(b).position(|(x, y)| x != y);
                println!(
                    "FAIL exact {label} out{i}: {diff}/{} bytes differ, first at {first:?}",
                    a.len()
                );
            }
        }
    }
    Ok(())
}

/// (a) Returns (compares, failures).
fn exactness(g: &dyn GpuBackend, kn: &Kerns, rng: &mut Rng) -> Result<Stats> {
    let mut st = Stats { total: 0, fails: 0 };
    let (pgu, pdn) = (gen_packed(rng, MI, HIDDEN), gen_packed(rng, HIDDEN, MI));
    let gate = make_table(g, rng, &pgu, MI, HIDDEN, EXACT_DISTINCT)?;
    let upt = make_table(g, rng, &pgu, MI, HIDDEN, EXACT_DISTINCT)?;
    let down = make_table(g, rng, &pdn, HIDDEN, MI, EXACT_DISTINCT)?;
    println!(
        "exactness pool: encoded scale bytes {} of {} plain ({:.2}%), gate/up/down x {} distinct",
        gate.enc_bytes + upt.enc_bytes + down.enc_bytes,
        gate.plain_bytes + upt.plain_bytes + down.plain_bytes,
        100.0 * (gate.enc_bytes + upt.enc_bytes + down.enc_bytes) as f64
            / (gate.plain_bytes + upt.plain_bytes + down.plain_bytes) as f64,
        EXACT_DISTINCT
    );

    // 1 row: sw_moe.
    {
        let d_x = up(g, &bf16_rand(rng, HIDDEN))?;
        let d_act = up(g, &bf16_rand(rng, TOP_K * MI))?;
        for seed in 0..SEEDS {
            let ids = random_ids(rng, 1).remove(0);
            let d_ids = up(g, &le(&ids, |x: i32| x.to_le_bytes()))?;
            pair(
                g,
                &mut st,
                &format!("sw_moe gate R=1 seed{seed}"),
                &[TOP_K * MI * 2],
                &|e, o| l_sw1(g, kn.sw1, d_x, &gate, e, o[0], d_ids, MI, HIDDEN, 0, 0),
            )?;
            pair(
                g,
                &mut st,
                &format!("sw_moe down R=1 seed{seed}"),
                &[TOP_K * HIDDEN * 2],
                &|e, o| l_sw1(g, kn.sw1, d_act, &down, e, o[0], d_ids, HIDDEN, MI, MI, 0),
            )?;
            g.free(d_ids)?;
        }
        g.free(d_x)?;
        g.free(d_act)?;
        println!("exact sw_moe R=1: compared (running fails {})", st.fails);
    }

    // Row-batched entries.
    for rows in 2..=8usize {
        let d_x = up(g, &bf16_rand(rng, rows * HIDDEN))?;
        let d_act = up(g, &bf16_rand(rng, rows * TOP_K * MI))?;
        let (gu_b, dn_b) = (rows * TOP_K * MI * 2, rows * TOP_K * HIDDEN * 2);
        for seed in 0..SEEDS {
            let rt = build_routing(g, kn.union, &random_ids(rng, rows))?;
            pair(
                g,
                &mut st,
                &format!("batchm_gateup R={rows} seed{seed}"),
                &[gu_b, gu_b],
                &|e, o| {
                    l_gateup(
                        g,
                        kn.gateup[rows - 2],
                        d_x,
                        &gate,
                        &upt,
                        e,
                        o[0],
                        o[1],
                        rt.ueid,
                        rt.uslot,
                        rows,
                        0,
                    )
                },
            )?;
            pair(
                g,
                &mut st,
                &format!("batchm_down R={rows} seed{seed}"),
                &[dn_b],
                &|e, o| {
                    l_down(
                        g,
                        kn.down[rows - 2],
                        d_act,
                        &down,
                        e,
                        o[0],
                        rt.ueid,
                        rt.uslot,
                        rows,
                        0,
                    )
                },
            )?;
            if let Some(i) = BM_RS.iter().position(|&r| r == rows) {
                for (j, kh) in [(1usize, kn.bm1[i]), (8usize, kn.bm8[i])] {
                    pair(
                        g,
                        &mut st,
                        &format!("batchm_m{rows} J={j} gate-like seed{seed}"),
                        &[gu_b],
                        &|e, o| {
                            l_batchm(
                                g,
                                kh,
                                MI.div_ceil(8 * j),
                                d_x,
                                &gate,
                                e,
                                o[0],
                                &rt,
                                rows,
                                MI,
                                HIDDEN,
                                false,
                                0,
                            )
                        },
                    )?;
                    pair(
                        g,
                        &mut st,
                        &format!("batchm_m{rows} J={j} down-like seed{seed}"),
                        &[dn_b],
                        &|e, o| {
                            l_batchm(
                                g,
                                kh,
                                HIDDEN.div_ceil(8 * j),
                                d_act,
                                &down,
                                e,
                                o[0],
                                &rt,
                                rows,
                                HIDDEN,
                                MI,
                                true,
                                0,
                            )
                        },
                    )?;
                }
            }
            g.free(rt.ueid)?;
            g.free(rt.uslot)?;
        }
        g.free(d_x)?;
        g.free(d_act)?;
        println!(
            "exact batchm entries R={rows}: compared (running fails {})",
            st.fails
        );
    }

    // va2 grouped GEMM: `m` rows on each of 5 experts, expert-sorted rows; gate shape gathers
    // through sorted_token_ids (dispatch.rs gate/up), down shape has a null gather map
    // (dispatch.rs:114-127).
    for m in [17usize, 40, 64] {
        for seed in 0..SEEDS {
            let ex = random_ids(rng, 1).remove(0);
            let ex: Vec<usize> = ex.iter().take(5).map(|&e| e as usize).collect();
            let mut set = ex.clone();
            set.sort_unstable();
            let te = m * set.len();
            let mut offs = vec![0i32; NUM_EXPERTS + 1];
            for e in 0..NUM_EXPERTS {
                offs[e + 1] = offs[e] + if set.contains(&e) { m as i32 } else { 0 };
            }
            let d_off = up(g, &le(&offs, |x: i32| x.to_le_bytes()))?;
            let mut perm: Vec<i32> = (0..te as i32).collect();
            for i in 0..te {
                let j = i + rng.next() as usize % (te - i);
                perm.swap(i, j);
            }
            let d_sorted = up(g, &le(&perm, |x: i32| x.to_le_bytes()))?;
            let d_x = up(g, &bf16_rand(rng, te * HIDDEN))?;
            let d_act = up(g, &bf16_rand(rng, te * MI))?;
            let mt = m.div_ceil(128) as u32;
            pair(
                g,
                &mut st,
                &format!("grouped va2 gate M={m} seed{seed}"),
                &[te * MI * 2],
                &|e, o| {
                    l_grouped(
                        g, kn.va2, d_x, &gate, e, o[0], d_off, d_sorted, MI, HIDDEN, mt,
                    )
                },
            )?;
            pair(
                g,
                &mut st,
                &format!("grouped va2 down M={m} seed{seed}"),
                &[te * HIDDEN * 2],
                &|e, o| {
                    l_grouped(
                        g,
                        kn.va2,
                        d_act,
                        &down,
                        e,
                        o[0],
                        d_off,
                        DevicePtr(0),
                        HIDDEN,
                        MI,
                        mt,
                    )
                },
            )?;
            for p in [d_off, d_sorted, d_x, d_act] {
                g.free(p)?;
            }
        }
        println!(
            "exact grouped va2 M={m}: compared (running fails {})",
            st.fails
        );
    }
    Ok(st)
}

struct LayerTabs {
    gate: Table,
    up: Table,
    down: Table,
}

/// One 42-layer decode graph over the cold pool for `enc`.
#[allow(clippy::too_many_arguments)]
fn build_graph(
    g: &dyn GpuBackend,
    s: u64,
    kn: &Kerns,
    layers: &[LayerTabs],
    rows: usize,
    enc: bool,
    d_ids: &[DevicePtr],
    bufs: &Bufs,
) -> Result<GraphHandle> {
    g.begin_capture(s)?;
    let body = || -> Result<()> {
        for l in 0..STEP_LAYERS {
            let lt = &layers[l % LAYERS];
            let ids = d_ids[l % ROUTINGS];
            if rows == 1 {
                l_sw1(
                    g,
                    kn.sw1,
                    bufs.x,
                    &lt.gate,
                    enc,
                    bufs.c_gate,
                    ids,
                    MI,
                    HIDDEN,
                    0,
                    s,
                )?;
                l_sw1(
                    g, kn.sw1, bufs.x, &lt.up, enc, bufs.c_up, ids, MI, HIDDEN, 0, s,
                )?;
                l_sw1(
                    g, kn.sw1, bufs.act, &lt.down, enc, bufs.c_dn, ids, HIDDEN, MI, MI, s,
                )?;
            } else {
                launch_union(g, kn.union, ids, bufs.ueid, bufs.uslot, rows, s)?;
                l_gateup(
                    g,
                    kn.gateup[rows - 2],
                    bufs.x,
                    &lt.gate,
                    &lt.up,
                    enc,
                    bufs.c_gate,
                    bufs.c_up,
                    bufs.ueid,
                    bufs.uslot,
                    rows,
                    s,
                )?;
                l_down(
                    g,
                    kn.down[rows - 2],
                    bufs.act,
                    &lt.down,
                    enc,
                    bufs.c_dn,
                    bufs.ueid,
                    bufs.uslot,
                    rows,
                    s,
                )?;
            }
        }
        Ok(())
    };
    if let Err(e) = body() {
        g.abort_capture_if_active(s);
        return Err(e);
    }
    g.end_capture(s)
}

struct Bufs {
    x: DevicePtr,
    act: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    c_dn: DevicePtr,
    ueid: DevicePtr,
    uslot: DevicePtr,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// (b) Returns (plain us/layer, encoded us/layer) medians at `rows` rows, and the median of the
/// per-replay encoded/plain ratios (the stop-rule statistic).
fn timing(
    g: &dyn GpuBackend,
    kn: &Kerns,
    layers: &[LayerTabs],
    rng: &mut Rng,
    rows: usize,
) -> Result<(f64, f64, f64)> {
    let s = g.create_stream()?;
    let d_ids: Vec<DevicePtr> = (0..ROUTINGS)
        .map(|_| {
            let flat: Vec<i32> = random_ids(rng, rows).into_iter().flatten().collect();
            up(g, &le(&flat, |x: i32| x.to_le_bytes()))
        })
        .collect::<Result<_>>()?;
    let bufs = Bufs {
        x: up(g, &bf16_rand(rng, rows * HIDDEN))?,
        act: up(g, &bf16_rand(rng, rows * TOP_K * MI))?,
        c_gate: g.alloc(rows * TOP_K * MI * 2)?,
        c_up: g.alloc(rows * TOP_K * MI * 2)?,
        c_dn: g.alloc(rows * TOP_K * HIDDEN * 2)?,
        ueid: g.alloc(rows * TOP_K * 4)?,
        uslot: g.alloc(rows * TOP_K * rows * 4)?,
    };
    let gp = build_graph(g, s, kn, layers, rows, false, &d_ids, &bufs)?;
    let ge = build_graph(g, s, kn, layers, rows, true, &d_ids, &bufs)?;
    let run = |gr: GraphHandle| -> Result<f64> {
        let t0 = std::time::Instant::now();
        g.launch_graph(gr, s)?;
        g.synchronize(s)?;
        Ok(t0.elapsed().as_secs_f64() * 1e6 / STEP_LAYERS as f64)
    };
    for _ in 0..WARMUP {
        run(gp)?;
        run(ge)?;
    }
    let (mut tp, mut te) = (Vec::new(), Vec::new());
    for _ in 0..REPS {
        tp.push(run(gp)?);
        te.push(run(ge)?);
    }
    g.destroy_graph(gp)?;
    g.destroy_graph(ge)?;
    for p in d_ids {
        g.free(p)?;
    }
    for p in [
        bufs.x,
        bufs.act,
        bufs.c_gate,
        bufs.c_up,
        bufs.c_dn,
        bufs.ueid,
        bufs.uslot,
    ] {
        g.free(p)?;
    }
    // 2026-10-17: per-replay pairs (plain then encoded, back to back) for the paired ratio.
    let mut pr: Vec<f64> = te.iter().zip(&tp).map(|(e, p)| e / p).collect();
    let pair = median(&mut pr);
    Ok((median(&mut tp), median(&mut te), pair))
}

fn run() -> Result<i32> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut missing: Vec<String> = Vec::new();
    let mut get = |module: &str, name: &str| -> KernelHandle {
        match g.kernel(module, name) {
            Ok(h) => h,
            Err(_) => {
                missing.push(format!("{module}::{name}"));
                KernelHandle(0)
            }
        }
    };
    let gv = "w4a16_gemv";
    let kn = Kerns {
        sw1: get(gv, "w4a16_gemv_sw_moe"),
        union: get(gv, "glm5next_moe_row_union"),
        gateup: (2..=8)
            .map(|r| get(gv, &format!("w4a16_gemv_sw_moe_batchm_gateup_m{r}")))
            .collect(),
        down: (2..=8)
            .map(|r| get(gv, &format!("w4a16_gemv_sw_moe_batchm_down_m{r}")))
            .collect(),
        bm1: BM_RS
            .iter()
            .map(|r| get(gv, &format!("w4a16_gemv_sw_moe_batchm_m{r}")))
            .collect(),
        bm8: BM_RS
            .iter()
            .map(|r| get(gv, &format!("w4a16_gemv_sw_moe_batchm_m{r}_c8")))
            .collect(),
        va2: get(
            "moe_w4a16",
            "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_va2",
        ),
    };
    if !missing.is_empty() {
        for m in &missing {
            println!("MISSING kernel entry {m}");
        }
        println!("FAIL kernel entries missing ({})", missing.len());
        return Ok(2);
    }
    let mut rng = Rng(0x0063_7366_5f73_636c);

    let st = exactness(g, &kn, &mut rng)?;
    let exact_ok = st.fails == 0;
    println!(
        "GATE exact {} ({} compares, {} differ)",
        if exact_ok { "PASS" } else { "FAIL" },
        st.total,
        st.fails
    );

    // Cold pool: LAYERS layers x (gate, up, down) x 288 distinct experts.
    let (pgu, pdn) = (
        gen_packed(&mut rng, MI, HIDDEN),
        gen_packed(&mut rng, HIDDEN, MI),
    );
    let mut layers = Vec::new();
    for _ in 0..LAYERS {
        layers.push(LayerTabs {
            gate: make_table(g, &mut rng, &pgu, MI, HIDDEN, NUM_EXPERTS)?,
            up: make_table(g, &mut rng, &pgu, MI, HIDDEN, NUM_EXPERTS)?,
            down: make_table(g, &mut rng, &pdn, HIDDEN, MI, NUM_EXPERTS)?,
        });
    }
    let (pb, eb) = layers.iter().fold((0usize, 0usize), |(p, e), l| {
        let t = [&l.gate, &l.up, &l.down];
        (
            p + t.iter().map(|x| x.plain_bytes).sum::<usize>(),
            e + t.iter().map(|x| x.enc_bytes).sum::<usize>(),
        )
    });
    println!(
        "pool scale bytes: plain {pb} ({:.1} MiB), encoded {eb} ({:.1} MiB), encoded/plain {:.4} \
         ({LAYERS} layers x 3 matrices x {NUM_EXPERTS} experts)",
        pb as f64 / 1048576.0,
        eb as f64 / 1048576.0,
        eb as f64 / pb as f64
    );

    let mut ratios = [0.0f64; 3];
    for (i, rows) in [1usize, 4, 8].into_iter().enumerate() {
        let (p, e, pair) = timing(g, &kn, &layers, &mut rng, rows)?;
        ratios[i] = pair;
        println!(
            "TIMING R={rows} layer plain {p:.1} us, encoded {e:.1} us, ratio of medians {:.4}, \
             median paired ratio {pair:.4} (median of {REPS} replays of a {STEP_LAYERS}-layer \
             graph, interleaved)",
            e / p
        );
    }
    println!(
        "GATE speed (median paired encoded/plain; slower above {SLOWER}) R=1 ratio {:.4} R=4 ratio {:.4} R=8 ratio {:.4}",
        ratios[0], ratios[1], ratios[2]
    );
    let slower: Vec<&str> = [(0usize, "1"), (1, "4")]
        .into_iter()
        .filter(|&(i, _)| ratios[i] > SLOWER)
        .map(|(_, r)| r)
        .collect();
    if !exact_ok {
        println!("FAIL exact: {}/{} compares differ", st.fails, st.total);
        if !slower.is_empty() {
            println!("STOP: csf slower per call at R={}", slower.join(","));
        }
        Ok(1)
    } else if slower.is_empty() {
        println!("PASS: csf not slower at R=1 and R=4");
        Ok(0)
    } else {
        println!("STOP: csf slower per call at R={}", slower.join(","));
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
