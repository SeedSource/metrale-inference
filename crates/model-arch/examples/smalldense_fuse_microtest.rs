// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate 0 for fusing the small dense decode GEMVs of the GLM-5.3 verify round
//! (SeedHQ/seed-skills#80; census `.planning/targets/SMALLDENSE-GATE0-20261012.md`). Existing
//! kernels only; every launch below is the production launch (same helper, argument list, grid,
//! block and variant) with the buffers swapped for test data.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Experiments (M = 3 verify rows, distinct random weights per layer):
//! - A, DSA indexer x 11 layers. Parts, per layer, for r in 0..3: `dense_gemv_tcm_bf16` wk
//!   128x4096 and compress_gate 128x4096 (M = 1, hidden row r), `dense_gemv_bf16_fp32out`
//!   weights_proj 32x4096 (hidden row r), fp32out wq_b 4096x1536 (q_resid row r): 12 launches.
//!   Fused: tcm wk M = 3, tcm compress_gate M = 3, `dense_gemv_bf16_fp32out_batchm` (through
//!   `dense_gemv_batchm_split`, y = 1, as `batchm_rows` runs it under
//!   `METRALE_GLM_DSA_GEMV_SPLIT=1`) weights_proj M = 3 and wq_b M = 3: 4 launches.
//! - AD: as A fused, with wk and compress_gate stacked into one [256, 4096] BF16 weight and one
//!   tcm M = 3 launch: 3 launches per layer.
//! - B, KDA x 34 layers. Parts: tcm f_a 128x4096 M = 3 and tcm b_proj 32x4096 M = 3. Fused: one
//!   tcm M = 3 launch over the stacked [160, 4096] weight.
//!
//! Each arm is stream-captured (backend `begin_capture`, relaxed mode as the engine) into one
//! graph over all layers. Timing: a 72 MiB WRITE flush before every replay (a read flush does
//! not evict), 3 rounds x 20 replays after 3 warm-up replays, arm order rotated per round, median
//! per arm. Exactness: every output element of the fused arm against the parts arm, bitwise,
//! each arm's outputs poisoned first (a different byte per arm), element counts asserted > 0,
//! and a flipped weight bit in the fused copy must change the output (negative control).
//!
//! Prints `SDF_ENV`, `SDF_LAUNCH`, `SDF_CMP exp= differ= of=`, `SDF_NEG exp= detected=`,
//! `SDF_T exp= arm= round= ms=` (and the median with implied GB/s), `SDF_SAVING exp=
//! ms_per_round= us_per_layer=`, `SDF_ROW exp=A proj= r= us= gbps=` (eager, events around
//! each launch of the parts' real order after one flush) and `SDF_VOID` when a cold point
//! implies more than 255 GB/s (measured DRAM maximum 249). The last line is `PASS
//! smalldense_fuse (...)` or `FAIL smalldense_fuse: <reason>` (exit 1). Timing is only
//! meaningful alone on the GPU.
//!
//! Run (GPU; build-mt sets the target env):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     METRALE_GLM_GEMV_TC=1 METRALE_GLM_DENSE_FP8=1 METRALE_GLM_DSA_GEMV_SPLIT=1 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example smalldense_fuse_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use metrale_model_layers::layers::ops::{self, dense_gemv_tcm};
use metrale_model_layers::weight_map::DenseWeight;

// CUDA driver event API, declared as in `glm5next_nv4_group_ceiling_microtest/timing.rs`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

fn ck(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

/// Verify rows per launch (MTP K3 verify).
const M: usize = 3;
const HID: usize = 4096;
const QL: usize = 1536;
const IDX_N: usize = 128;
const HEADS: usize = 32;
const NQ: usize = 4096;
const LAYERS_A: usize = 11;
const LAYERS_B: usize = 34;
const ROUNDS: usize = 3;
const REPLAYS: usize = 20;
const WARMUP: usize = 3;
const ROW_REPS: usize = 10;
const FLUSH_BYTES: usize = 72 << 20;
/// Above this a cold point cannot be DRAM (measured maximum 249 GB/s on GB10, 2026-10-11).
const GBPS_CAP: f64 = 255.0;
const POISON_PARTS: u8 = 0xA5;
const POISON_FUSED: u8 = 0x5A;
const POISON_AD: u8 = 0x3C;

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Approximately N(0, 1): the sum of four uniforms, rescaled.
    fn n(&mut self) -> f64 {
        (self.u() + self.u() + self.u() + self.u() - 2.0) * 1.7320508
    }
    fn bf16s(&mut self, count: usize, std: f64) -> Vec<u16> {
        (0..count)
            .map(|_| bf16::from_f64(self.n() * std).to_bits())
            .collect()
    }
}

fn to_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn up16(g: &dyn GpuBackend, v: &[u16]) -> Result<DevicePtr> {
    let p = g.alloc(v.len() * 2)?;
    g.copy_h2d(&to_bytes(v), p)?;
    Ok(p)
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

/// Kernel handles and the launch helpers every arm shares.
struct Ctx<'a> {
    g: &'a dyn GpuBackend,
    stream: u64,
    tcm: KernelHandle,
    var: dense_gemv_tcm::TcmVariant,
    f32_1: KernelHandle,
    f32_m: KernelHandle,
}

impl Ctx<'_> {
    /// `dense_gemv_tcm::launch`, as `dense_fp8::route` -> `try_bf16` calls it: `out_stride = n`.
    fn tcm(
        &self,
        a: DevicePtr,
        w: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        dense_gemv_tcm::launch(
            self.g,
            self.tcm,
            self.var,
            a,
            w,
            DevicePtr(0),
            c,
            m as u32,
            n as u32,
            k as u32,
            n as u32,
            self.stream,
        )
    }

    /// `ops::dense_gemv` on the `dense_gemv_bf16_fp32out` handle: what `gemm` -> `dense_mm_bf16`
    /// runs at M = 1 for `weights_proj` and `wq_b`.
    fn f32_1(&self, a: DevicePtr, w: DevicePtr, c: DevicePtr, n: usize, k: usize) -> Result<()> {
        ops::dense_gemv(
            self.g,
            self.f32_1,
            a,
            &DenseWeight { weight: w },
            c,
            n as u32,
            k as u32,
            self.stream,
        )
    }

    /// `batchm_rows` under `METRALE_GLM_DSA_GEMV_SPLIT=1`: `dense_gemv_batchm_split` with
    /// `y_blocks = ceil(rows / 16) = 1` for 3 rows (grid [ceil(n / 4), 1, 1]).
    fn f32_m(
        &self,
        a: DevicePtr,
        w: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let y = (m as u32).div_ceil(ops::DENSE_GEMV_BATCHM_MAX_M);
        ops::dense_gemv_batchm_split(
            self.g,
            self.f32_m,
            a,
            &DenseWeight { weight: w },
            c,
            m as u32,
            y,
            n as u32,
            k as u32,
            n as u32,
            self.stream,
        )
    }

    fn capture(&self, f: &dyn Fn() -> Result<()>) -> Result<GraphHandle> {
        self.g.synchronize(self.stream)?;
        self.g.begin_capture(self.stream)?;
        if let Err(e) = f() {
            self.g.abort_capture_if_active(self.stream);
            return Err(e);
        }
        self.g.end_capture(self.stream)
    }
}

/// Device output buffers of one arm: (pointer, bytes).
struct Bufs(Vec<(DevicePtr, usize)>);

impl Bufs {
    fn new(g: &dyn GpuBackend, sizes: &[usize]) -> Result<Self> {
        let mut v = Vec::new();
        for &s in sizes {
            v.push((g.alloc(s)?, s));
        }
        Ok(Self(v))
    }
    fn ptr(&self, i: usize) -> DevicePtr {
        self.0[i].0
    }
}

/// Poison every output, replay `graph`, return the outputs' bytes.
fn run_graph(c: &Ctx, graph: GraphHandle, out: &Bufs, poison: u8) -> Result<Vec<Vec<u8>>> {
    for &(p, n) in &out.0 {
        c.g.memset(p, poison, n)?;
    }
    c.g.launch_graph(graph, c.stream)?;
    c.g.synchronize(c.stream)?;
    let mut res = Vec::new();
    for &(p, n) in &out.0 {
        let mut b = vec![0u8; n];
        c.g.copy_d2h(p, &mut b)?;
        res.push(b);
    }
    Ok(res)
}

/// Differing elements of two byte buffers of `esz`-byte elements, and the element count.
fn diff_elems(a: &[u8], b: &[u8], esz: usize) -> (usize, usize) {
    if a.len() != b.len() {
        return (a.len().max(b.len()) / esz, a.len().max(b.len()) / esz);
    }
    let d = a
        .chunks_exact(esz)
        .zip(b.chunks_exact(esz))
        .filter(|(x, y)| x != y)
        .count();
    (d, a.len() / esz)
}

/// Rows `[t][cols]` of a `[M, stride]` BF16 buffer starting at column `c0`, as bytes per row.
fn cols(buf: &[u8], stride: usize, c0: usize, width: usize) -> Vec<u8> {
    let mut v = Vec::new();
    for t in 0..M {
        let s = (t * stride + c0) * 2;
        v.extend_from_slice(&buf[s..s + width * 2]);
    }
    v
}

/// Flip an exponent bit of element 0 (row 0, column 0) of `host` on the device at `ptr`.
fn perturb(g: &dyn GpuBackend, ptr: DevicePtr, host: &[u16]) -> Result<()> {
    let mut h = host.to_vec();
    h[0] ^= 0x4000;
    g.copy_h2d(&to_bytes(&h), ptr)
}

fn restore(g: &dyn GpuBackend, ptr: DevicePtr, host: &[u16]) -> Result<()> {
    g.copy_h2d(&to_bytes(host), ptr)
}

struct Ev([u64; 2]);
impl Ev {
    fn new() -> Result<Self> {
        let mut e = [0u64; 2];
        for x in e.iter_mut() {
            // SAFETY: plain driver call writing one event handle into `x`; the backend made the
            // context current when it was built.
            ck(unsafe { cuEventCreate(x, 0) }, "cuEventCreate")?;
        }
        Ok(Self(e))
    }
}
impl Drop for Ev {
    fn drop(&mut self) {
        for &e in &self.0 {
            // SAFETY: destroys an event this struct created.
            unsafe {
                cuEventDestroy_v2(e);
            }
        }
    }
}

/// Per-arm result of the timing rounds.
struct Timed {
    ms: f64,
}

/// Time two graphs: per replay a 72 MiB write flush (outside the timed region), then the graph
/// between two events. `ROUNDS` x `REPLAYS`, arm order rotated per round. Prints the per-round
/// medians and the overall median with the implied GB/s of `unique_bytes`.
#[allow(clippy::too_many_arguments)]
fn time_pair(
    c: &Ctx,
    flush: DevicePtr,
    ev: &Ev,
    exp: &str,
    names: [&str; 2],
    graphs: [GraphHandle; 2],
    unique_bytes: f64,
    voids: &mut Vec<String>,
) -> Result<[Timed; 2]> {
    let one = |graph: GraphHandle| -> Result<f64> {
        c.g.memset_async(flush, 0x5A, FLUSH_BYTES, c.stream)?;
        // SAFETY: records events this struct created on the backend's stream.
        unsafe { ck(cuEventRecord(ev.0[0], c.stream), "cuEventRecord")? };
        c.g.launch_graph(graph, c.stream)?;
        // SAFETY: as above; the elapsed time is read after the host waited on the second event.
        unsafe {
            ck(cuEventRecord(ev.0[1], c.stream), "cuEventRecord")?;
            ck(cuEventSynchronize(ev.0[1]), "cuEventSynchronize")?;
            let mut ms = 0f32;
            ck(
                cuEventElapsedTime(&mut ms, ev.0[0], ev.0[1]),
                "cuEventElapsedTime",
            )?;
            Ok(ms as f64)
        }
    };
    let mut all: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for round in 0..ROUNDS {
        for j in 0..2 {
            let i = (round + j) % 2;
            for _ in 0..WARMUP {
                one(graphs[i])?;
            }
            let mut t = Vec::with_capacity(REPLAYS);
            for _ in 0..REPLAYS {
                t.push(one(graphs[i])?);
            }
            println!(
                "SDF_T exp={exp} arm={} round={round} ms={:.4}",
                names[i],
                median(t.clone())
            );
            all[i].extend(t);
        }
    }
    let mut out = Vec::new();
    for i in 0..2 {
        let ms = median(all[i].clone());
        let gbps = unique_bytes / (ms * 1e-3) / 1e9;
        println!(
            "SDF_T exp={exp} arm={} median ms={ms:.4} unique_MB={:.2} gbps={gbps:.1}",
            names[i],
            unique_bytes / 1e6
        );
        if gbps > GBPS_CAP {
            println!(
                "SDF_VOID exp={exp} arm={} gbps={gbps:.1} > {GBPS_CAP}",
                names[i]
            );
            voids.push(format!("exp {exp} arm {} implies {gbps:.0} GB/s", names[i]));
        }
        out.push(Timed { ms });
    }
    let (parts, fused) = (out[0].ms, out[1].ms);
    let layers = if exp == "B" { LAYERS_B } else { LAYERS_A };
    println!(
        "SDF_SAVING exp={exp} ms_per_round={:.4} us_per_layer={:.3} layers={layers} (parts {parts:.4} ms, fused {fused:.4} ms)",
        parts - fused,
        (parts - fused) * 1e3 / layers as f64
    );
    let mut it = out.into_iter();
    Ok([it.next().unwrap(), it.next().unwrap()])
}

/// One layer's device weights of experiment A (parts / fused arm copies).
#[derive(Clone, Copy)]
struct AW {
    wk: DevicePtr,
    cg: DevicePtr,
    wp: DevicePtr,
    wqb: DevicePtr,
}

/// One layer's inputs: `hidden` `[3, 4096]` and `q_resid` `[3, 1536]` BF16.
#[derive(Clone, Copy)]
struct Inp {
    hid: DevicePtr,
    qres: DevicePtr,
}

/// One layer of host weights of A.
struct AHost {
    wk: Vec<u16>,
    cg: Vec<u16>,
    wp: Vec<u16>,
    wqb: Vec<u16>,
}

fn up_a(g: &dyn GpuBackend, h: &AHost) -> Result<AW> {
    Ok(AW {
        wk: up16(g, &h.wk)?,
        cg: up16(g, &h.cg)?,
        wp: up16(g, &h.wp)?,
        wqb: up16(g, &h.wqb)?,
    })
}

/// Output buffer sizes of A's parts and fused arm for one layer: wk, cg (BF16 `[3, 128]`),
/// weights_proj (FP32 `[3, 32]`), wq_b (FP32 `[3, 4096]`).
const A_SIZES: [usize; 4] = [M * IDX_N * 2, M * IDX_N * 2, M * HEADS * 4, M * NQ * 4];
const A_ESZ: [usize; 4] = [2, 2, 4, 4];

/// Projection `p` (0 wk, 1 compress_gate, 2 weights_proj, 3 wq_b) of row `r` at M = 1, the
/// production launch of `decode_k` -> `indexer_forward` / `select_row_at`.
fn part_a(c: &Ctx, p: usize, r: usize, w: &AW, i: &Inp, o: &Bufs, l: usize) -> Result<()> {
    let h = i.hid.offset(r * HID * 2);
    let q = i.qres.offset(r * QL * 2);
    match p {
        0 => c.tcm(h, w.wk, o.ptr(4 * l).offset(r * IDX_N * 2), 1, IDX_N, HID),
        1 => c.tcm(
            h,
            w.cg,
            o.ptr(4 * l + 1).offset(r * IDX_N * 2),
            1,
            IDX_N,
            HID,
        ),
        2 => c.f32_1(h, w.wp, o.ptr(4 * l + 2).offset(r * HEADS * 4), HEADS, HID),
        _ => c.f32_1(q, w.wqb, o.ptr(4 * l + 3).offset(r * NQ * 4), NQ, QL),
    }
}

fn parts_a(c: &Ctx, w: &[AW], inp: &[Inp], o: &Bufs) -> Result<()> {
    for l in 0..w.len() {
        for r in 0..M {
            for p in 0..4 {
                part_a(c, p, r, &w[l], &inp[l], o, l)?;
            }
        }
    }
    Ok(())
}

fn fused_a(c: &Ctx, w: &[AW], inp: &[Inp], o: &Bufs) -> Result<()> {
    for l in 0..w.len() {
        c.tcm(inp[l].hid, w[l].wk, o.ptr(4 * l), M, IDX_N, HID)?;
        c.tcm(inp[l].hid, w[l].cg, o.ptr(4 * l + 1), M, IDX_N, HID)?;
        c.f32_m(inp[l].hid, w[l].wp, o.ptr(4 * l + 2), M, HEADS, HID)?;
        c.f32_m(inp[l].qres, w[l].wqb, o.ptr(4 * l + 3), M, NQ, QL)?;
    }
    Ok(())
}

/// AD: `kc` is the stacked `[wk; compress_gate]` `[256, 4096]` weight, output `[3, 256]` BF16.
/// Output buffers per layer: kc, weights_proj, wq_b.
fn fused_ad(c: &Ctx, kc: &[DevicePtr], w: &[AW], inp: &[Inp], o: &Bufs) -> Result<()> {
    for l in 0..w.len() {
        c.tcm(inp[l].hid, kc[l], o.ptr(3 * l), M, 2 * IDX_N, HID)?;
        c.f32_m(inp[l].hid, w[l].wp, o.ptr(3 * l + 1), M, HEADS, HID)?;
        c.f32_m(inp[l].qres, w[l].wqb, o.ptr(3 * l + 2), M, NQ, QL)?;
    }
    Ok(())
}

/// What one exactness leg reports.
struct Leg {
    exp: &'static str,
    differ: usize,
    of: usize,
    neg: bool,
}

const PROJ_A: [&str; 4] = ["wk", "cgate", "wproj", "wq_b"];
const PROJ_A_BYTES: [f64; 4] = [
    (IDX_N * HID * 2) as f64,
    (IDX_N * HID * 2) as f64,
    (HEADS * HID * 2) as f64,
    (NQ * QL * 2) as f64,
];

/// Experiments A and AD.
fn exp_a(
    c: &Ctx,
    flush: DevicePtr,
    ev: &Ev,
    legs: &mut Vec<Leg>,
    voids: &mut Vec<String>,
) -> Result<()> {
    let g = c.g;
    let mut rng = Lcg(0x5344_4655_5345_4130);
    let mut host = Vec::new();
    let mut inp = Vec::new();
    for _ in 0..LAYERS_A {
        host.push(AHost {
            wk: rng.bf16s(IDX_N * HID, 0.02),
            cg: rng.bf16s(IDX_N * HID, 0.02),
            wp: rng.bf16s(HEADS * HID, 0.02),
            wqb: rng.bf16s(NQ * QL, 0.02),
        });
        inp.push(Inp {
            hid: up16(g, &rng.bf16s(M * HID, 1.0))?,
            qres: up16(g, &rng.bf16s(M * QL, 1.0))?,
        });
    }
    // Own copy of the weights per arm, so a flipped bit in the fused copy cannot touch parts.
    let w_parts: Vec<AW> = host.iter().map(|h| up_a(g, h)).collect::<Result<_>>()?;
    let w_fused: Vec<AW> = host.iter().map(|h| up_a(g, h)).collect::<Result<_>>()?;
    let w_ad: Vec<AW> = host.iter().map(|h| up_a(g, h)).collect::<Result<_>>()?;
    let kc_host: Vec<Vec<u16>> = host
        .iter()
        .map(|h| h.wk.iter().chain(h.cg.iter()).copied().collect())
        .collect();
    let kc: Vec<DevicePtr> = kc_host.iter().map(|v| up16(g, v)).collect::<Result<_>>()?;

    let sizes4: Vec<usize> = (0..LAYERS_A).flat_map(|_| A_SIZES).collect();
    let sizes3: Vec<usize> = (0..LAYERS_A)
        .flat_map(|_| [M * 2 * IDX_N * 2, A_SIZES[2], A_SIZES[3]])
        .collect();
    let o_parts = Bufs::new(g, &sizes4)?;
    let o_fused = Bufs::new(g, &sizes4)?;
    let o_ad = Bufs::new(g, &sizes3)?;
    let scratch = Bufs::new(g, &sizes4)?;

    let gp = c.capture(&|| parts_a(c, &w_parts, &inp, &o_parts))?;
    let gf = c.capture(&|| fused_a(c, &w_fused, &inp, &o_fused))?;
    let gad = c.capture(&|| fused_ad(c, &kc, &w_ad, &inp, &o_ad))?;

    let ref_out = run_graph(c, gp, &o_parts, POISON_PARTS)?;
    let f_out = run_graph(c, gf, &o_fused, POISON_FUSED)?;
    let ad_out = run_graph(c, gad, &o_ad, POISON_AD)?;

    // A: every element of the fused arm against the parts arm.
    let (mut differ, mut of) = (0, 0);
    for (i, (r, f)) in ref_out.iter().zip(&f_out).enumerate() {
        let (d, n) = diff_elems(r, f, A_ESZ[i % 4]);
        differ += d;
        of += n;
    }
    // AD: columns 0..128 of the stacked output are wk, 128..256 compress_gate.
    let (mut differ_ad, mut of_ad) = (0, 0);
    for l in 0..LAYERS_A {
        let kcb = &ad_out[3 * l];
        for (pi, c0) in [(0usize, 0usize), (1, IDX_N)] {
            let want = cols(&ref_out[4 * l + pi], IDX_N, 0, IDX_N);
            let got = cols(kcb, 2 * IDX_N, c0, IDX_N);
            let (d, n) = diff_elems(&want, &got, 2);
            differ_ad += d;
            of_ad += n;
        }
        for k in 0..2 {
            let (d, n) = diff_elems(&ref_out[4 * l + 2 + k], &ad_out[3 * l + 1 + k], 4);
            differ_ad += d;
            of_ad += n;
        }
    }

    // Negative controls: one exponent bit of element 0 of each fused weight of layer 0.
    let mut neg_a = true;
    for p in 0..4 {
        let (ptr, h): (DevicePtr, &Vec<u16>) = match p {
            0 => (w_fused[0].wk, &host[0].wk),
            1 => (w_fused[0].cg, &host[0].cg),
            2 => (w_fused[0].wp, &host[0].wp),
            _ => (w_fused[0].wqb, &host[0].wqb),
        };
        perturb(g, ptr, h)?;
        let out = run_graph(c, gf, &o_fused, POISON_FUSED)?;
        restore(g, ptr, h)?;
        let d: usize = ref_out
            .iter()
            .zip(&out)
            .enumerate()
            .map(|(i, (r, f))| diff_elems(r, f, A_ESZ[i % 4]).0)
            .sum();
        println!("SDF_NEG exp=A proj={} differ={d}", PROJ_A[p]);
        neg_a &= d > 0;
    }
    let mut neg_ad = true;
    for p in 0..3 {
        let (ptr, h): (DevicePtr, &Vec<u16>) = match p {
            0 => (kc[0], &kc_host[0]),
            1 => (w_ad[0].wp, &host[0].wp),
            _ => (w_ad[0].wqb, &host[0].wqb),
        };
        perturb(g, ptr, h)?;
        let out = run_graph(c, gad, &o_ad, POISON_AD)?;
        restore(g, ptr, h)?;
        let mut d = 0;
        for l in 0..LAYERS_A {
            for (pi, c0) in [(0usize, 0usize), (1, IDX_N)] {
                let want = cols(&ref_out[4 * l + pi], IDX_N, 0, IDX_N);
                d += diff_elems(&want, &cols(&out[3 * l], 2 * IDX_N, c0, IDX_N), 2).0;
            }
            for k in 0..2 {
                d += diff_elems(&ref_out[4 * l + 2 + k], &out[3 * l + 1 + k], 4).0;
            }
        }
        println!(
            "SDF_NEG exp=AD proj={} differ={d}",
            ["kc", "wproj", "wq_b"][p]
        );
        neg_ad &= d > 0;
    }
    println!("SDF_CMP exp=A differ={differ} of={of}");
    println!("SDF_CMP exp=AD differ={differ_ad} of={of_ad}");
    println!("SDF_NEG exp=A detected={}", neg_a as u8);
    println!("SDF_NEG exp=AD detected={}", neg_ad as u8);
    legs.push(Leg {
        exp: "A",
        differ,
        of,
        neg: neg_a,
    });
    legs.push(Leg {
        exp: "AD",
        differ: differ_ad,
        of: of_ad,
        neg: neg_ad,
    });

    let unique = LAYERS_A as f64 * PROJ_A_BYTES.iter().sum::<f64>();
    time_pair(
        c,
        flush,
        ev,
        "A",
        ["parts", "fused"],
        [gp, gf],
        unique,
        voids,
    )?;
    time_pair(
        c,
        flush,
        ev,
        "AD",
        ["parts", "fused"],
        [gp, gad],
        unique,
        voids,
    )?;

    // Eager per-launch rows of the parts in their real order, after one flush per layer.
    let mut e = [0u64; 13];
    for x in e.iter_mut() {
        // SAFETY: plain driver call writing one event handle into `x`.
        ck(unsafe { cuEventCreate(x, 0) }, "cuEventCreate")?;
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); 12];
    for _ in 0..ROW_REPS {
        for l in 0..LAYERS_A {
            g.memset_async(flush, 0x5A, FLUSH_BYTES, c.stream)?;
            // SAFETY: records events created above on the backend's stream.
            unsafe { ck(cuEventRecord(e[0], c.stream), "cuEventRecord")? };
            for r in 0..M {
                for p in 0..4 {
                    part_a(c, p, r, &w_parts[l], &inp[l], &scratch, l)?;
                    let idx = 1 + r * 4 + p;
                    // SAFETY: as above.
                    unsafe { ck(cuEventRecord(e[idx], c.stream), "cuEventRecord")? };
                }
            }
            // SAFETY: as above; elapsed times are read after the last event completed.
            unsafe {
                ck(cuEventSynchronize(e[12]), "cuEventSynchronize")?;
                for s in 0..12 {
                    let mut ms = 0f32;
                    ck(
                        cuEventElapsedTime(&mut ms, e[s], e[s + 1]),
                        "cuEventElapsedTime",
                    )?;
                    samples[s].push(ms as f64 * 1e3);
                }
            }
        }
    }
    for x in e {
        // SAFETY: destroys events created above.
        unsafe {
            cuEventDestroy_v2(x);
        }
    }
    for p in 0..4 {
        for r in 0..M {
            let us = median(samples[r * 4 + p].clone());
            let gbps = PROJ_A_BYTES[p] / (us * 1e-6) / 1e9;
            let warm = r > 0;
            println!(
                "SDF_ROW exp=A proj={} r={r} us={us:.2} gbps={gbps:.1} warm={}",
                PROJ_A[p], warm as u8
            );
            if !warm && gbps > GBPS_CAP {
                println!(
                    "SDF_VOID exp=A proj={} r=0 gbps={gbps:.1} > {GBPS_CAP}",
                    PROJ_A[p]
                );
                voids.push(format!("row {} r0 implies {gbps:.0} GB/s", PROJ_A[p]));
            }
        }
    }
    Ok(())
}

/// Experiment B.
fn exp_b(
    c: &Ctx,
    flush: DevicePtr,
    ev: &Ev,
    legs: &mut Vec<Leg>,
    voids: &mut Vec<String>,
) -> Result<()> {
    let g = c.g;
    let mut rng = Lcg(0x5344_4655_5345_4231);
    let (mut fa_h, mut bp_h, mut hid) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..LAYERS_B {
        fa_h.push(rng.bf16s(IDX_N * HID, 0.02));
        bp_h.push(rng.bf16s(HEADS * HID, 0.02));
        hid.push(up16(g, &rng.bf16s(M * HID, 1.0))?);
    }
    let st_h: Vec<Vec<u16>> = (0..LAYERS_B)
        .map(|l| fa_h[l].iter().chain(bp_h[l].iter()).copied().collect())
        .collect();
    let fa: Vec<DevicePtr> = fa_h.iter().map(|v| up16(g, v)).collect::<Result<_>>()?;
    let bp: Vec<DevicePtr> = bp_h.iter().map(|v| up16(g, v)).collect::<Result<_>>()?;
    let st: Vec<DevicePtr> = st_h.iter().map(|v| up16(g, v)).collect::<Result<_>>()?;
    let sz_parts: Vec<usize> = (0..LAYERS_B)
        .flat_map(|_| [M * IDX_N * 2, M * HEADS * 2])
        .collect();
    let sz_fused: Vec<usize> = (0..LAYERS_B).map(|_| M * (IDX_N + HEADS) * 2).collect();
    let o_parts = Bufs::new(g, &sz_parts)?;
    let o_fused = Bufs::new(g, &sz_fused)?;

    let gp = c.capture(&|| {
        for l in 0..LAYERS_B {
            c.tcm(hid[l], fa[l], o_parts.ptr(2 * l), M, IDX_N, HID)?;
            c.tcm(hid[l], bp[l], o_parts.ptr(2 * l + 1), M, HEADS, HID)?;
        }
        Ok(())
    })?;
    let gf = c.capture(&|| {
        for l in 0..LAYERS_B {
            c.tcm(hid[l], st[l], o_fused.ptr(l), M, IDX_N + HEADS, HID)?;
        }
        Ok(())
    })?;
    let ref_out = run_graph(c, gp, &o_parts, POISON_PARTS)?;
    let compare = |out: &[Vec<u8>]| -> (usize, usize) {
        let (mut d, mut n) = (0, 0);
        for l in 0..LAYERS_B {
            let a = diff_elems(&ref_out[2 * l], &cols(&out[l], IDX_N + HEADS, 0, IDX_N), 2);
            let b = diff_elems(
                &ref_out[2 * l + 1],
                &cols(&out[l], IDX_N + HEADS, IDX_N, HEADS),
                2,
            );
            d += a.0 + b.0;
            n += a.1 + b.1;
        }
        (d, n)
    };
    let f_out = run_graph(c, gf, &o_fused, POISON_FUSED)?;
    let (differ, of) = compare(&f_out);
    perturb(g, st[0], &st_h[0])?;
    let out = run_graph(c, gf, &o_fused, POISON_FUSED)?;
    restore(g, st[0], &st_h[0])?;
    let (d_neg, _) = compare(&out);
    let neg = d_neg > 0;
    println!("SDF_CMP exp=B differ={differ} of={of}");
    println!("SDF_NEG exp=B differ={d_neg}");
    println!("SDF_NEG exp=B detected={}", neg as u8);
    legs.push(Leg {
        exp: "B",
        differ,
        of,
        neg,
    });

    let unique = LAYERS_B as f64 * ((IDX_N + HEADS) * HID * 2) as f64;
    time_pair(
        c,
        flush,
        ev,
        "B",
        ["parts", "fused"],
        [gp, gf],
        unique,
        voids,
    )?;
    Ok(())
}

fn run() -> Result<(bool, String)> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    // Module strings as production resolves them (`kernels/gb10/common/KERNEL.toml` `[modules]`
    // maps `dense_gemv_bf16 = "gemv"`; an unlisted `.cu` file's module is its file stem):
    //   fp32out M=1   "gemv" / dense_gemv_bf16_fp32out          (glm5next_dsa/layer.rs resolve)
    //   fp32out batchm "dense_gemv_bf16_batchm" / dense_gemv_bf16_fp32out_batchm (the batchm file)
    //   tcm            dense_gemv_tcm::MODULE ("dense_gemv_tcm") / variant entry (handles()).
    let var = dense_gemv_tcm::variant_for(0, 0, false);
    let c = Ctx {
        g,
        stream: g.default_stream(),
        tcm: g.kernel(dense_gemv_tcm::MODULE, var.entry())?,
        var,
        f32_1: g.kernel("gemv", "dense_gemv_bf16_fp32out")?,
        f32_m: g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_fp32out_batchm")?,
    };
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| "unset".into());
    println!(
        "SDF_ENV tcm_variant={:?} entry={} module={} rows_per_cta={} chosen_by=variant_for(n,k,fp8) only, not env; \
         METRALE_GLM_GEMV_TC={} METRALE_GLM_DENSE_FP8={} METRALE_GLM_DSA_GEMV_SPLIT={} (launches call the \
         production helpers directly; the weights are BF16 and unregistered, so dense_fp8 routes them to try_bf16)",
        var,
        var.entry(),
        dense_gemv_tcm::MODULE,
        var.rows_per_cta(),
        env("METRALE_GLM_GEMV_TC"),
        env("METRALE_GLM_DENSE_FP8"),
        env("METRALE_GLM_DSA_GEMV_SPLIT"),
    );
    let tcm_grid = |n: usize| n.div_ceil(var.rows_per_cta() as usize);
    for (name, n, k, kernel, m) in [
        ("A parts wk", IDX_N, HID, "tcm_bf16", 1),
        ("A parts cgate", IDX_N, HID, "tcm_bf16", 1),
        ("A fused wk", IDX_N, HID, "tcm_bf16", M),
        ("A fused cgate", IDX_N, HID, "tcm_bf16", M),
        ("AD fused wk+cgate", 2 * IDX_N, HID, "tcm_bf16", M),
        ("B parts f_a", IDX_N, HID, "tcm_bf16", M),
        ("B parts b_proj", HEADS, HID, "tcm_bf16", M),
        ("B fused f_a+b_proj", IDX_N + HEADS, HID, "tcm_bf16", M),
    ] {
        println!(
            "SDF_LAUNCH {name} kernel={kernel} m={m} n={n} k={k} grid=[{},1,1] block=[256,1,1] smem=0 out_stride={n}",
            tcm_grid(n)
        );
    }
    for (name, n, k, kernel, m, y) in [
        ("A parts wproj", HEADS, HID, "fp32out", 1, 1),
        ("A parts wq_b", NQ, QL, "fp32out", 1, 1),
        ("A fused wproj", HEADS, HID, "fp32out_batchm_split", M, 1),
        ("A fused wq_b", NQ, QL, "fp32out_batchm_split", M, 1),
    ] {
        println!(
            "SDF_LAUNCH {name} kernel={kernel} m={m} n={n} k={k} grid=[{},{y},1] block=[256,1,1] smem=0",
            n.div_ceil(4)
        );
    }

    let flush = g.alloc(FLUSH_BYTES)?;
    g.memset(flush, 0, FLUSH_BYTES)?;
    let ev = Ev::new()?;
    let (mut legs, mut voids) = (Vec::new(), Vec::new());
    exp_a(&c, flush, &ev, &mut legs, &mut voids)?;
    exp_b(&c, flush, &ev, &mut legs, &mut voids)?;

    let mut why: Vec<String> = Vec::new();
    for l in &legs {
        if l.of == 0 {
            why.push(format!("exp {} compared 0 elements", l.exp));
        }
        if l.differ != 0 {
            why.push(format!(
                "exp {} differs in {} of {} elements",
                l.exp, l.differ, l.of
            ));
        }
        if !l.neg {
            why.push(format!("exp {} negative control not detected", l.exp));
        }
    }
    why.extend(voids.into_iter().map(|v| format!("VOID {v}")));
    if why.is_empty() {
        let elems: Vec<String> = legs.iter().map(|l| format!("{}={}", l.exp, l.of)).collect();
        Ok((
            true,
            format!(
                "PASS smalldense_fuse (every fused output bit-identical to parts, elements {}, negative controls detected, no cold point above {GBPS_CAP} GB/s)",
                elems.join(" ")
            ),
        ))
    } else {
        Ok((false, format!("FAIL smalldense_fuse: {}", why.join("; "))))
    }
}

fn main() {
    match run() {
        Ok((ok, line)) => {
            println!("{line}");
            if !ok {
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("FAIL smalldense_fuse: {e:#}");
            std::process::exit(1);
        }
    }
}
