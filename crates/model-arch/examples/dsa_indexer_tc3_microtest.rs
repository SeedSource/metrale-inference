// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Bit-parity and timing gate for `METRALE_GLM_DSA_SCORES_TC3=1`:
//! `dsa_q_to_bf16` + `dsa_index_scores_tc3` against `dsa_index_scores_tc2`, the launch it
//! replaces, both launched as `select_tokens` launches them on an exact launch.
//!
//! Bitwise (hard): H = 32, KP = 4, D = 128 (plus one D = 96 case), Q = 128 at S in {16384,
//! 65536, 131072, 262144} with a causal prefill-tail `q_pos`, and edge cases: Q = 97, P not a
//! multiple of 256 (S = 65536 + 36), a single partial pool tile (P = 250), 32 rows, S = 131072
//! with `q_pos` spread over the whole context (end > q_pos for most pools). Random FP32 q /
//! keys / weights with BF16-rounding stress (exact ties 0x8000, near ties 0x7FFF / 0x8001,
//! magnitudes scaled by 2^+-12, FP32 subnormals), -0.0 and +0.0, zero q heads, zero and
//! duplicated keys, zero / negative weights, `pool_valid` = 0 holes, `valid_keys` = 0 tokens
//! (the case "huge" also puts +-1e30 into q and keys so sums overflow to inf / NaN). Each arm
//! writes into buffers pre-filled with its own sentinel byte (0xA5 tc2, 0x5A tc3), so an
//! element only one arm writes cannot compare equal. `out` is compared as raw u32 and
//! `valid_cand` as raw bytes, never as floats.
//!
//! Negative control (hard): the top mantissa bit of q[Q - 1][0][0] flipped in the tc3 arm's
//! input must change its `out`; a run with no candidate or no positive score fails.
//!
//! Timing (information): Q = 128, D 128, H 32, S in {16384, 65536, 131072, 262144}. Per arm 11
//! distinct buffer sets (one per layer), the 11 calls captured in one CUDA graph, a 64 MiB
//! memset L2 flush before each replay, the median of 9 replays per block, blocks ordered
//! palindromically (ABBA: tc2, replica, tc3, noq, noout, noout, noq, tc3, replica, tc2); an
//! arm's number is the mean of its two block medians. Arms: tc2; tc3 including the prepass;
//! `replica` (the tc2 dataflow as the ablation template instantiates it, bitwise-checked
//! against tc2 on one case, to show the ablations start from tc2's speed); `noq` (the replica
//! with the q global loads removed: q_load leaves zeros, the smem stores stay); `noout` (the
//! replica with the final candidacy / out / valid_cand stores behind a runtime flag that is 0;
//! the MMAs and head sums stay live because the guarded stores read them). TF/s is
//! 2 Q P H D / time.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and a last line starting `PASS` when every case is bitwise identical and the control
//! fires; 1 and `FAIL ...` otherwise; 2 when a kernel is absent from this target.
//!
//! Run (dsa_indexer.cu is a gb10 common kernel):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_indexer_tc3_microtest

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::DSA_MODULE;
use metrale_model_arch::glm5next_dsa::select::tc2::{
    SCORES_TC2_BLOCK, scores_tc2_grid, scores_tc2_smem,
};
use metrale_model_arch::glm5next_dsa::select::tc3::{
    Q_TO_BF16_BLOCK, SCORES_TC3_BLOCK, q_to_bf16_blocks, scores_tc3_grid,
};

// 2026-10-10: CUDA driver event API for kernel-only timing, declared as in
// `dsa_indexer_tc2_bitparity_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-10: GLM-5.3 `index_n_heads` and `index_kpool`; mode 1 is `bf16`.
const H: usize = 32;
const KP: usize = 4;
const MODE: u32 = 1;
const POISON_TC2: u8 = 0xA5;
const POISON_TC3: u8 = 0x5A;
const LAYERS: usize = 11;
const FLUSH_BYTES: usize = 64 << 20;
const REPLAYS: usize = 9;
const TIMING_Q: usize = 128;
const TIMING_SS: &[usize] = &[16_384, 65_536, 131_072, 262_144];

/// 2026-10-10: How `q_pos` is laid out.
#[derive(Clone, Copy, PartialEq)]
enum Qpos {
    /// Causal prefill tail: row r at S - Q + r.
    Tail,
    /// Spread over the whole context: row r at S (r + 1) / Q - 1, so most pools end past it.
    Spread,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    q: usize,
    s: usize,
    d: usize,
    qpos: Qpos,
    huge: bool,
}

const fn case(name: &'static str, q: usize, s: usize, qpos: Qpos) -> Case {
    Case {
        name,
        q,
        s,
        d: 128,
        qpos,
        huge: false,
    }
}

const CASES: &[Case] = &[
    case("Q128_S16384", 128, 16_384, Qpos::Tail),
    case("Q128_S65536", 128, 65_536, Qpos::Tail),
    case("Q128_S131072", 128, 131_072, Qpos::Tail),
    case("Q128_S262144", 128, 262_144, Qpos::Tail),
    case("Q97_S65536", 97, 65_536, Qpos::Spread),
    case("Q128_S65572_Pnot256", 128, 65_536 + 36, Qpos::Tail),
    case("Q97_S65572_Pnot256_spread", 97, 65_536 + 36, Qpos::Spread),
    case("Q128_S131072_spread", 128, 131_072, Qpos::Spread),
    case("Q32_S4096_spread", 32, 4_096, Qpos::Spread),
    case("Q50_P250_onepartialtile", 50, 1_000, Qpos::Spread),
    Case {
        name: "Q128_S32768_huge",
        q: 128,
        s: 32_768,
        d: 128,
        qpos: Qpos::Spread,
        huge: true,
    },
    Case {
        name: "Q33_S8196_D96",
        q: 33,
        s: 8_196,
        d: 96,
        qpos: Qpos::Spread,
        huge: false,
    },
];

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn f(&mut self) -> f32 {
        ((self.u() as f64) / ((1u64 << 53) as f64)) as f32
    }
    /// 2026-10-10: A value in [-1, 1], then one of: unchanged, low 16 bits forced to an exact
    /// BF16 tie or a near tie, scaled by 2^12 or 2^-12, or an FP32 subnormal.
    fn stress(&mut self) -> f32 {
        let x = 2.0 * self.f() - 1.0;
        let b = x.to_bits() & 0xFFFF_0000;
        match self.u() % 16 {
            0 => f32::from_bits(b | 0x8000),
            1 => f32::from_bits(b | 0x7FFF),
            2 => f32::from_bits(b | 0x8001),
            3 => x * 4096.0,
            4 => x / 4096.0,
            5 => {
                let m = self.u() as u32 & 0x007F_FFFF;
                f32::from_bits((x.to_bits() & 0x8000_0000) | m)
            }
            _ => x,
        }
    }
}

struct Inputs {
    q: Vec<f32>,
    keys: Vec<f32>,
    weights: Vec<f32>,
    pool_indices: Vec<i32>,
    pool_valid: Vec<u8>,
    valid_keys: Vec<u8>,
    q_pos: Vec<i32>,
    s: usize,
    p: usize,
}

fn inputs(c: Case, seed: u64) -> Inputs {
    let p = c.s / KP;
    let mut rng = Lcg(seed ^ ((c.q * 131 + p * 7 + c.d) as u64).wrapping_mul(0x9E37_79B9));
    let huge = |i: usize| c.huge && i % 1009 == 0;
    let mut q = vec![0.0f32; c.q * H * c.d];
    for (i, v) in q.iter_mut().enumerate() {
        let zero_head = (i / c.d) % 9 == 4;
        *v = if huge(i) {
            if i % 2 == 0 { 1.0e30 } else { -1.0e30 }
        } else if zero_head {
            0.0
        } else if i % 37 == 11 {
            -0.0
        } else {
            rng.stress()
        };
    }
    let mut keys = vec![0.0f32; p * c.d];
    for pp in 0..p {
        for d in 0..c.d {
            keys[pp * c.d + d] = if huge(pp * c.d + d) {
                1.0e30
            } else if pp % 11 == 5 {
                0.0
            } else if pp % 7 == 3 {
                keys[(pp - 1) * c.d + d]
            } else {
                rng.stress()
            };
        }
    }
    let weights = (0..c.q * H)
        .map(|i| match i % 29 {
            3 => 0.0,
            17 => -0.0,
            _ => 2.0 * rng.f() - 1.0,
        })
        .collect();
    // 2026-10-10: Pool p holds tokens 4p .. 4p + 3; a few ends are out of [0, S).
    let mut pool_indices = vec![0i32; p * KP];
    for pp in 0..p {
        for slot in 0..KP {
            pool_indices[pp * KP + slot] = (pp * KP + slot) as i32;
        }
        let end = &mut pool_indices[pp * KP + KP - 1];
        if pp % 13 == 6 {
            *end = c.s as i32 + 3;
        } else if pp % 17 == 8 {
            *end = -2;
        }
    }
    let pool_valid = (0..p).map(|pp| u8::from(pp % 19 != 7)).collect();
    let valid_keys = (0..c.s).map(|t| u8::from(t % 23 != 11)).collect();
    let q_pos = (0..c.q)
        .map(|r| match c.qpos {
            Qpos::Tail => (c.s as i64 - c.q as i64 + r as i64).max(0) as i32,
            Qpos::Spread => ((c.s * (r + 1)) / c.q) as i32 - 1,
        })
        .collect();
    Inputs {
        q,
        keys,
        weights,
        pool_indices,
        pool_valid,
        valid_keys,
        q_pos,
        s: c.s,
        p,
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

/// 2026-10-10: Device copies of one case's inputs, the BF16 q scratch and one output pair.
struct Bufs {
    q: DevicePtr,
    qbf: DevicePtr,
    keys: DevicePtr,
    weights: DevicePtr,
    pool_indices: DevicePtr,
    pool_valid: DevicePtr,
    valid_keys: DevicePtr,
    q_pos: DevicePtr,
    out: DevicePtr,
    vc: DevicePtr,
    s: usize,
    p: usize,
}

impl Bufs {
    fn new(g: &dyn GpuBackend, i: &Inputs, c: Case) -> Result<Self> {
        Ok(Self {
            q: up(g, &f32_bytes(&i.q))?,
            qbf: g.alloc((c.q * H * c.d * 2).max(1))?,
            keys: up(g, &f32_bytes(&i.keys))?,
            weights: up(g, &f32_bytes(&i.weights))?,
            pool_indices: up(g, &i32_bytes(&i.pool_indices))?,
            pool_valid: up(g, &i.pool_valid)?,
            valid_keys: up(g, &i.valid_keys)?,
            q_pos: up(g, &i32_bytes(&i.q_pos))?,
            out: g.alloc(c.q * i.p * 4)?,
            vc: g.alloc(c.q * i.p)?,
            s: i.s,
            p: i.p,
        })
    }

    fn free(&self, g: &dyn GpuBackend) {
        let all = [
            self.q,
            self.qbf,
            self.keys,
            self.weights,
            self.pool_indices,
            self.pool_valid,
            self.valid_keys,
            self.q_pos,
            self.out,
            self.vc,
        ];
        for p in all {
            g.free(p).ok();
        }
    }
}

/// 2026-10-10: The kernels and which launch each is.
#[derive(Clone, Copy)]
struct Kernels {
    tc2: KernelHandle,
    q_to_bf16: KernelHandle,
    tc3: KernelHandle,
    replica: KernelHandle,
    noq: KernelHandle,
    noout: KernelHandle,
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Tc2,
    Tc3,
    /// tc2's dataflow through the ablation template, nothing removed.
    Replica,
    /// ... without the q global loads.
    NoQ,
    /// ... with the final stores behind a flag that is 0.
    NoOut,
}

/// 2026-10-10: One scores launch on `stream` as `select_tokens` issues it: tc2 (FP32 q), or
/// `dsa_q_to_bf16` then tc3 (BF16 q), or an ablation entry (tc2's grid, block and shared
/// memory, one more argument). `geom` NULL, mode 1.
fn launch(
    g: &dyn GpuBackend,
    k: Kernels,
    arm: Arm,
    b: &Bufs,
    q: DevicePtr,
    d: usize,
    qrows: usize,
    stream: u64,
) -> Result<()> {
    let (handle, grid, block, smem, qptr) = match arm {
        Arm::Tc3 => {
            let n4 = qrows * H * d / 4;
            KernelLaunch::new(g, k.q_to_bf16)
                .grid([q_to_bf16_blocks(n4), 1, 1])
                .block([Q_TO_BF16_BLOCK, 1, 1])
                .arg_ptr(q)
                .arg_ptr(b.qbf)
                .arg_u32(n4 as u32)
                .launch(stream)?;
            (
                k.tc3,
                scores_tc3_grid(qrows, b.p),
                SCORES_TC3_BLOCK,
                scores_tc2_smem(d, H) as u32,
                b.qbf,
            )
        }
        a => (
            match a {
                Arm::Tc2 => k.tc2,
                Arm::Replica => k.replica,
                Arm::NoQ => k.noq,
                _ => k.noout,
            },
            scores_tc2_grid(qrows, b.p),
            SCORES_TC2_BLOCK,
            scores_tc2_smem(d, H) as u32,
            q,
        ),
    };
    let l = KernelLaunch::new(g, handle)
        .grid(grid)
        .block([block, 1, 1])
        .shared_mem(smem)
        .arg_ptr(qptr)
        .arg_ptr(b.keys)
        .arg_ptr(b.weights)
        .arg_ptr(b.pool_indices)
        .arg_ptr(b.pool_valid)
        .arg_ptr(b.valid_keys)
        .arg_ptr(b.q_pos)
        .arg_ptr(b.out)
        .arg_ptr(b.vc)
        .arg_u32(qrows as u32)
        .arg_u32(b.p as u32)
        .arg_u32(H as u32)
        .arg_u32(d as u32)
        .arg_u32(KP as u32)
        .arg_u32(b.s as u32)
        .arg_f32((d as f32).powf(-0.5))
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(MODE);
    // 2026-10-10: The ablation entries take `store_on` last; 0 for noout (stores skipped), 1
    // otherwise.
    let l = if matches!(arm, Arm::Replica | Arm::NoQ | Arm::NoOut) {
        l.arg_u32(u32::from(arm != Arm::NoOut))
    } else {
        l
    };
    l.launch(stream)
}

/// 2026-10-10: (out bytes, valid_cand bytes) of one arm, both buffers poisoned first.
#[allow(clippy::too_many_arguments)]
fn run_arm(
    g: &dyn GpuBackend,
    k: Kernels,
    arm: Arm,
    b: &Bufs,
    q: DevicePtr,
    c: Case,
    poison: u8,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let n = c.q * b.p;
    g.memset(b.out, poison, n * 4)?;
    g.memset(b.vc, poison, n)?;
    launch(g, k, arm, b, q, c.d, c.q, 0)?;
    Ok((down(g, b.out, n * 4)?, down(g, b.vc, n)?))
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// 2026-10-10: Compare `arm` against tc2 on one case; prints the BITWISE line. Returns
/// (identical, candidates, positive scores).
fn bitwise(
    g: &dyn GpuBackend,
    k: Kernels,
    c: Case,
    arm: Arm,
    label: &str,
) -> Result<(bool, usize, usize)> {
    let inp = inputs(c, 0x7C3);
    let b = Bufs::new(g, &inp, c)?;
    let (ro, rv) = run_arm(g, k, Arm::Tc2, &b, b.q, c, POISON_TC2)?;
    let (no, nv) = run_arm(g, k, arm, &b, b.q, c, POISON_TC3)?;
    b.free(g);
    let (rw, nw) = (words(&ro), words(&no));
    let diff_out: Vec<usize> = (0..rw.len()).filter(|&i| rw[i] != nw[i]).collect();
    let diff_vc: Vec<usize> = (0..rv.len()).filter(|&i| rv[i] != nv[i]).collect();
    let ok = diff_out.is_empty() && diff_vc.is_empty();
    let cands = rv.iter().filter(|&&v| v == 1).count();
    let pos = rw.iter().filter(|&&w| f32::from_bits(w) > 0.0).count();
    let nan = rw.iter().filter(|&&w| f32::from_bits(w).is_nan()).count();
    if ok {
        println!(
            "TC3 BITWISE case={}{label} identical (Q={} P={} D={} words={} candidates={cands} \
             positive={pos} nan={nan})",
            c.name,
            c.q,
            inp.p,
            c.d,
            rw.len()
        );
    } else {
        let first = diff_out
            .first()
            .copied()
            .or(diff_vc.first().copied())
            .unwrap_or(0);
        println!(
            "TC3 BITWISE case={}{label} DIFF first={first} ndiff={}",
            c.name,
            diff_out.len() + diff_vc.len()
        );
        if let Some(&i) = diff_out.first() {
            println!(
                "  first out mismatch row {} pool {}: tc2 {:#010x} ({}) new {:#010x} ({}); \
                 out words differing {}, valid_cand bytes differing {}",
                i / inp.p,
                i % inp.p,
                rw[i],
                f32::from_bits(rw[i]),
                nw[i],
                f32::from_bits(nw[i]),
                diff_out.len(),
                diff_vc.len()
            );
        }
    }
    Ok((ok, cands, pos))
}

/// 2026-10-10: The 1-bit flip in q[Q - 1][0][0] fed to the tc3 arm must change its `out`
/// against tc2 on the unflipped input.
fn control(g: &dyn GpuBackend, k: Kernels) -> Result<bool> {
    let c = case("control", 128, 1_024, Qpos::Tail);
    let mut inp = inputs(c, 0xC0DE);
    let b = Bufs::new(g, &inp, c)?;
    let i = (c.q - 1) * H * c.d;
    inp.q[i] = f32::from_bits(inp.q[i].to_bits() ^ (1 << 22));
    let q_pert = up(g, &f32_bytes(&inp.q))?;
    let (ro, _) = run_arm(g, k, Arm::Tc2, &b, b.q, c, POISON_TC2)?;
    let (no, _) = run_arm(g, k, Arm::Tc3, &b, q_pert, c, POISON_TC2)?;
    let fired = ro != no;
    println!(
        "TC3 CONTROL 1-bit flip in q[{}][0][0] detected={fired}",
        c.q - 1
    );
    b.free(g);
    g.free(q_pert).ok();
    Ok(fired)
}

/// 2026-10-10: ms of one replay of `graph` on `stream` after an L2 flush.
fn replay_ms(
    g: &dyn GpuBackend,
    stream: u64,
    flush: DevicePtr,
    graph: metrale_gpu_runtime::gpu::GraphHandle,
    ev: (u64, u64),
) -> Result<f64> {
    g.memset_async(flush, 0x3C, FLUSH_BYTES, stream)?;
    // SAFETY: plain CUDA driver calls on events and a stream this function's caller owns.
    unsafe { check(cuEventRecord(ev.0, stream), "cuEventRecord(start)")? };
    g.launch_graph(graph, stream)?;
    let mut ms: f32 = 0.0;
    // SAFETY: as above; `ms` outlives the call.
    unsafe {
        check(cuEventRecord(ev.1, stream), "cuEventRecord(end)")?;
        check(cuEventSynchronize(ev.1), "cuEventSynchronize(end)")?;
        check(
            cuEventElapsedTime(&mut ms, ev.0, ev.1),
            "cuEventElapsedTime",
        )?;
    }
    Ok(ms as f64)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn timing(g: &dyn GpuBackend, k: Kernels) -> Result<()> {
    let stream = g.create_stream()?;
    let flush = g.alloc(FLUSH_BYTES)?;
    let (mut e0, mut e1): (u64, u64) = (0, 0);
    // SAFETY: events created here and destroyed below.
    unsafe {
        check(cuEventCreate(&mut e0, 0), "cuEventCreate(start)")?;
        check(cuEventCreate(&mut e1, 0), "cuEventCreate(end)")?;
    }
    // 2026-10-10: Palindromic block order: every arm appears twice, mirrored around the middle.
    let order = [
        Arm::Tc2,
        Arm::Replica,
        Arm::Tc3,
        Arm::NoQ,
        Arm::NoOut,
        Arm::NoOut,
        Arm::NoQ,
        Arm::Tc3,
        Arm::Replica,
        Arm::Tc2,
    ];
    for &s in TIMING_SS {
        let c = case("timing", TIMING_Q, s, Qpos::Tail);
        let inp = inputs(c, 0x7177);
        let sets: Vec<Bufs> = (0..LAYERS)
            .map(|_| Bufs::new(g, &inp, c))
            .collect::<Result<_>>()?;
        let mut graphs = Vec::new();
        for arm in [Arm::Tc2, Arm::Tc3, Arm::Replica, Arm::NoQ, Arm::NoOut] {
            g.begin_capture(stream)?;
            for b in &sets {
                if let Err(e) = launch(g, k, arm, b, b.q, c.d, c.q, stream) {
                    g.abort_capture_if_active(stream);
                    return Err(e);
                }
            }
            graphs.push((arm, g.end_capture(stream)?));
        }
        let graph_of = |a: Arm| graphs.iter().find(|(x, _)| *x == a).unwrap().1;
        // 2026-10-10: Burn-in: the first replay of a graph is slow; run each twice untimed.
        for (_, gr) in &graphs {
            for _ in 0..2 {
                g.memset_async(flush, 0x3C, FLUSH_BYTES, stream)?;
                g.launch_graph(*gr, stream)?;
            }
        }
        g.synchronize(stream)?;
        let mut blocks: Vec<(Arm, f64)> = Vec::new();
        for &arm in &order {
            let mut ms = Vec::new();
            for _ in 0..REPLAYS {
                ms.push(replay_ms(g, stream, flush, graph_of(arm), (e0, e1))?);
            }
            blocks.push((arm, median(&mut ms)));
        }
        let per_call_us = |a: Arm| {
            let v: Vec<f64> = blocks
                .iter()
                .filter(|(x, _)| *x == a)
                .map(|(_, m)| *m)
                .collect();
            v.iter().sum::<f64>() / v.len() as f64 * 1e3 / LAYERS as f64
        };
        let block_us = |a: Arm| -> String {
            blocks
                .iter()
                .filter(|(x, _)| *x == a)
                .map(|(_, m)| format!("{:.1}", m * 1e3 / LAYERS as f64))
                .collect::<Vec<_>>()
                .join(",")
        };
        let flops = 2.0 * (c.q * inp.p * H * c.d) as f64;
        let tf = |us: f64| flops / (us * 1e-6) / 1e12;
        let (t2, t3) = (per_call_us(Arm::Tc2), per_call_us(Arm::Tc3));
        println!(
            "TC3 TIME S={s} tc2_us={t2:.1} tc3_us={t3:.1} ratio={:.3} noq_us={:.1} noout_us={:.1} \
             replica_us={:.1} tc2_TFs={:.1} tc3_TFs={:.1} (ABBA per-block us: tc2 {} | replica {} | \
             tc3 {} | noq {} | noout {}; {LAYERS} layers per graph, median of {REPLAYS} replays, 64 \
             MiB flush)",
            t3 / t2,
            per_call_us(Arm::NoQ),
            per_call_us(Arm::NoOut),
            per_call_us(Arm::Replica),
            tf(t2),
            tf(t3),
            block_us(Arm::Tc2),
            block_us(Arm::Replica),
            block_us(Arm::Tc3),
            block_us(Arm::NoQ),
            block_us(Arm::NoOut),
        );
        for (_, gr) in graphs {
            g.destroy_graph(gr)?;
        }
        for b in &sets {
            b.free(g);
        }
    }
    // SAFETY: the events above.
    unsafe {
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    g.free(flush).ok();
    Ok(())
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let get = |func: &str| -> KernelHandle {
        match g.kernel(DSA_MODULE, func) {
            Ok(h) => h,
            Err(e) => {
                println!("{DSA_MODULE}::{func} absent from this target ({e}) - SKIP");
                std::process::exit(2);
            }
        }
    };
    let k = Kernels {
        tc2: get("dsa_index_scores_tc2"),
        q_to_bf16: get("dsa_q_to_bf16"),
        tc3: get("dsa_index_scores_tc3"),
        replica: get("dsa_index_scores_tc3_abl_replica"),
        noq: get("dsa_index_scores_tc3_abl_noq"),
        noout: get("dsa_index_scores_tc3_abl_noout"),
    };
    let (mut failed, mut cands, mut pos) = (Vec::new(), 0usize, 0usize);
    for &c in CASES {
        let (ok, n_c, n_p) = bitwise(g, k, c, Arm::Tc3, "")?;
        cands += n_c;
        pos += n_p;
        if !ok {
            failed.push(c.name.to_string());
        }
    }
    // 2026-10-10: The ablation template's tc2 replica against tc2 itself, so the ablation
    // timings start from a kernel that computes tc2's bytes.
    let rc = case("replica_vs_tc2", 128, 16_384, Qpos::Spread);
    let (ok, _, _) = bitwise(g, k, rc, Arm::Replica, "_replica")?;
    if !ok {
        failed.push("replica".to_string());
    }
    let fired = control(g, k)?;
    timing(g, k)?;
    if cands == 0 || pos == 0 {
        println!("FAIL - TC3: no candidate or no positive score; the cases prove nothing.");
        std::process::exit(1);
    }
    if !fired {
        println!("FAIL - TC3: the negative control did not fire; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if !failed.is_empty() {
        println!(
            "FAIL - TC3: {} case(s) differ ({}); keep METRALE_GLM_DSA_SCORES_TC3 off on this build.",
            failed.len(),
            failed.join(", ")
        );
        std::process::exit(1);
    }
    println!(
        "PASS - TC3 dsa_q_to_bf16 + dsa_index_scores_tc3 bitwise identical to dsa_index_scores_tc2 \
         on {} cases (out u32 bits and valid_cand bytes; {cands} candidates, {pos} positive).",
        CASES.len()
    );
    Ok(())
}
