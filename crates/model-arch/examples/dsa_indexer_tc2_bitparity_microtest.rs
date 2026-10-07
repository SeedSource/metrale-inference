// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Bit-parity and timing gate for `METRALE_GLM_DSA_SCORES_TC2=1`:
//! `dsa_index_scores_tc2` against `dsa_index_scores_tc` in mode 1 (`bf16`), the launch it
//! replaces, both launched as `select_tokens` launches them on an exact launch (directly, so
//! `SCORES_TC_MIN_POOLS` does not apply).
//!
//! Legs (hard): H = 32, (Q, P, D) in `LEGS` (Q 128 at P 256 .. 32768, D 128, plus edge rows,
//! pools and head dims). Random FP32 q / keys / weights in [-1, 1] with BF16-rounding stress
//! (low 16 bits set to exact ties 0x8000 and near ties 0x7FFF / 0x8001, magnitudes scaled by
//! 2^+-12, FP32 subnormals), -0.0 and +0.0, zero q heads, zero and duplicated keys (ties),
//! zero / negative weights, invalid pools, invalid end tokens, pool ends outside [0, S), and a
//! causal prefill-tail `q_pos`. `out` is compared as u32 bit patterns and `valid_cand` as
//! bytes; each arm's buffers start with its own poison byte, so an element only one arm
//! writes cannot compare equal.
//!
//! Negative control (hard): the top mantissa bit of q[Q - 1][0][0] flipped in the tc2 arm's
//! input must change its `out` against the unflipped reference. A run with no candidate or
//! no positive score fails, as it would prove nothing.
//!
//! Timing (information): Q = 128 (the rank's half of a 256-row sub-chunk), D 128, H 32, P in
//! `TIMING_PS`: CUDA events around `ITERS` launches after `WARMUP`; us/call, TF/s at 2QPHD.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and a last line starting `PASS` when every leg is bitwise identical and the control
//! fires; 1 and `FAIL ...` otherwise; 2 when either kernel is absent from this target.
//!
//! Run (dsa_indexer.cu is a gb10 common kernel):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_indexer_tc2_bitparity_microtest

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::DSA_MODULE;
use metrale_model_arch::glm5next_dsa::select::tc2::{
    SCORES_TC2_BLOCK, scores_tc2_grid, scores_tc2_smem,
};
use metrale_model_arch::glm5next_dsa::select::{SCORES_TC_BLOCK, scores_tc_grid};

// 2026-10-07: CUDA driver event API for kernel-only timing, declared as in
// `dsa_indexer_tiled_bitparity_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-07: GLM-5.3 `index_n_heads` and `index_kpool`; mode 1 is `bf16`.
const H: usize = 32;
const KP: usize = 4;
const MODE: u32 = 1;
/// 2026-10-07: (Q, P, D) legs.
const LEGS: &[(usize, usize, usize)] = &[
    (128, 256, 128),
    (128, 2048, 128),
    (128, 8192, 128),
    (128, 32768, 128),
    (1, 320, 128),
    (127, 8191, 128),
    (33, 129, 128),
    (128, 320, 96),
    (127, 2048, 64),
    (17, 200, 16),
];
const TIMING_PS: &[usize] = &[256, 2048, 8192, 32768];
const TIMING_Q: usize = 128;
const POISON_REF: u8 = 0xA5;
const POISON_NEW: u8 = 0x5A;
const WARMUP: usize = 3;
const ITERS: usize = 50;

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
    /// 2026-10-07: A value in [-1, 1], then one of: unchanged, low 16 bits forced to an exact
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

#[derive(Clone, Copy)]
struct Case {
    q: usize,
    p: usize,
    d: usize,
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
}

fn inputs(c: Case, seed: u64) -> Inputs {
    let mut rng = Lcg(seed ^ ((c.q * 131 + c.p * 7 + c.d) as u64).wrapping_mul(0x9E37_79B9));
    let s = c.p * KP + 2;
    let mut q = vec![0.0f32; c.q * H * c.d];
    for (i, v) in q.iter_mut().enumerate() {
        let zero_head = (i / c.d) % 9 == 4;
        *v = if zero_head {
            0.0
        } else if i % 37 == 11 {
            -0.0
        } else {
            rng.stress()
        };
    }
    let mut keys = vec![0.0f32; c.p * c.d];
    for p in 0..c.p {
        for d in 0..c.d {
            keys[p * c.d + d] = if p % 11 == 5 {
                0.0
            } else if p % 7 == 3 {
                keys[(p - 1) * c.d + d]
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
    let mut pool_indices = vec![0i32; c.p * KP];
    for p in 0..c.p {
        for slot in 0..KP {
            pool_indices[p * KP + slot] = (p * KP + slot) as i32;
        }
        let end = &mut pool_indices[p * KP + KP - 1];
        if p % 13 == 6 {
            *end = s as i32 + 3;
        } else if p % 17 == 8 {
            *end = -2;
        }
    }
    let pool_valid = (0..c.p).map(|p| u8::from(p % 19 != 7)).collect();
    let valid_keys = (0..s).map(|t| u8::from(t % 23 != 11)).collect();
    // 2026-10-07: Causal prefill tail: rows S - Q .. S - 1.
    let q_pos = (0..c.q)
        .map(|r| (s as i64 - c.q as i64 + r as i64).max(0) as i32)
        .collect();
    Inputs {
        q,
        keys,
        weights,
        pool_indices,
        pool_valid,
        valid_keys,
        q_pos,
        s,
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

/// 2026-10-07: Device copies of one case's inputs plus one output pair.
struct Bufs {
    q: DevicePtr,
    keys: DevicePtr,
    weights: DevicePtr,
    pool_indices: DevicePtr,
    pool_valid: DevicePtr,
    valid_keys: DevicePtr,
    q_pos: DevicePtr,
    out: DevicePtr,
    vc: DevicePtr,
    s: usize,
}

impl Bufs {
    fn new(g: &dyn GpuBackend, i: &Inputs, c: Case) -> Result<Self> {
        Ok(Self {
            q: up(g, &f32_bytes(&i.q))?,
            keys: up(g, &f32_bytes(&i.keys))?,
            weights: up(g, &f32_bytes(&i.weights))?,
            pool_indices: up(g, &i32_bytes(&i.pool_indices))?,
            pool_valid: up(g, &i.pool_valid)?,
            valid_keys: up(g, &i.valid_keys)?,
            q_pos: up(g, &i32_bytes(&i.q_pos))?,
            out: g.alloc(c.q * c.p * 4)?,
            vc: g.alloc(c.q * c.p)?,
            s: i.s,
        })
    }

    fn free(&self, g: &dyn GpuBackend) {
        let all = [
            self.q,
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

/// 2026-10-07: One launch as `select_tokens` issues it: `dsa_index_scores_tc` (grid
/// `scores_tc_grid`, no shared memory) or `dsa_index_scores_tc2` (grid `scores_tc2_grid`,
/// `scores_tc2_smem` bytes), the same arguments and mode 1 either way, `geom` NULL.
fn launch(
    g: &dyn GpuBackend,
    k: KernelHandle,
    new: bool,
    b: &Bufs,
    q: DevicePtr,
    c: Case,
) -> Result<()> {
    let (grid, block, smem) = if new {
        (
            scores_tc2_grid(c.q, c.p),
            SCORES_TC2_BLOCK,
            scores_tc2_smem(c.d, H) as u32,
        )
    } else {
        (scores_tc_grid(c.q, c.p), SCORES_TC_BLOCK, 0)
    };
    KernelLaunch::new(g, k)
        .grid(grid)
        .block([block, 1, 1])
        .shared_mem(smem)
        .arg_ptr(q)
        .arg_ptr(b.keys)
        .arg_ptr(b.weights)
        .arg_ptr(b.pool_indices)
        .arg_ptr(b.pool_valid)
        .arg_ptr(b.valid_keys)
        .arg_ptr(b.q_pos)
        .arg_ptr(b.out)
        .arg_ptr(b.vc)
        .arg_u32(c.q as u32)
        .arg_u32(c.p as u32)
        .arg_u32(H as u32)
        .arg_u32(c.d as u32)
        .arg_u32(KP as u32)
        .arg_u32(b.s as u32)
        .arg_f32((c.d as f32).powf(-0.5))
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(MODE)
        .launch(0)
}

/// 2026-10-07: (out bytes, valid_cand bytes) of one arm, both buffers poisoned first.
#[allow(clippy::too_many_arguments)]
fn arm(
    g: &dyn GpuBackend,
    k: KernelHandle,
    new: bool,
    b: &Bufs,
    q: DevicePtr,
    c: Case,
    poison: u8,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let n = c.q * c.p;
    g.memset(b.out, poison, n * 4)?;
    g.memset(b.vc, poison, n)?;
    launch(g, k, new, b, q, c)?;
    Ok((down(g, b.out, n * 4)?, down(g, b.vc, n)?))
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// 2026-10-07: Compare one leg; returns (identical, candidates, positive scores).
fn leg(g: &dyn GpuBackend, ks: [KernelHandle; 2], c: Case) -> Result<(bool, usize, usize)> {
    let b = Bufs::new(g, &inputs(c, 0x7C2), c)?;
    let (ro, rv) = arm(g, ks[0], false, &b, b.q, c, POISON_REF)?;
    let (no, nv) = arm(g, ks[1], true, &b, b.q, c, POISON_NEW)?;
    b.free(g);
    let (rw, nw) = (words(&ro), words(&no));
    let diff_out: Vec<usize> = (0..rw.len()).filter(|&i| rw[i] != nw[i]).collect();
    let diff_vc = rv.iter().zip(&nv).filter(|(x, y)| x != y).count();
    let ok = diff_out.is_empty() && diff_vc == 0;
    let cands = rv.iter().filter(|&&v| v == 1).count();
    let pos = rw.iter().filter(|&&w| f32::from_bits(w) > 0.0).count();
    println!(
        "leg Q={:<4} P={:<6} D={:<4} H={H}: out diff_words={:<8} valid_cand diff_bytes={:<8} \
         candidates={cands} positive={pos} {}",
        c.q,
        c.p,
        c.d,
        diff_out.len(),
        diff_vc,
        if ok { "IDENTICAL" } else { "DIFFERENT" }
    );
    if let Some(&i) = diff_out.first() {
        let (r, p) = (i / c.p, i % c.p);
        println!(
            "  first out mismatch row {r} pool {p}: tc {:#010x} ({}) tc2 {:#010x} ({})",
            rw[i],
            f32::from_bits(rw[i]),
            nw[i],
            f32::from_bits(nw[i])
        );
    }
    Ok((ok, cands, pos))
}

/// 2026-10-07: The 1-bit flip in q[Q - 1][0][0] must change tc2's `out` against tc.
fn control(g: &dyn GpuBackend, ks: [KernelHandle; 2]) -> Result<bool> {
    let c = Case {
        q: 128,
        p: 256,
        d: 128,
    };
    let mut inp = inputs(c, 0xC0DE);
    let b = Bufs::new(g, &inp, c)?;
    let i = (c.q - 1) * H * c.d;
    inp.q[i] = f32::from_bits(inp.q[i].to_bits() ^ (1 << 22));
    let q_pert = up(g, &f32_bytes(&inp.q))?;
    let (ro, _) = arm(g, ks[0], false, &b, b.q, c, POISON_REF)?;
    let (no, _) = arm(g, ks[1], true, &b, q_pert, c, POISON_REF)?;
    let fired = ro != no;
    println!(
        "CONTROL 1-bit flip in q[{}][0][0] detected={fired}",
        c.q - 1
    );
    b.free(g);
    g.free(q_pert).ok();
    Ok(fired)
}

/// 2026-10-07: Mean ms per call over `ITERS` back-to-back calls on stream 0 after `WARMUP`.
fn time_ms(g: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        f()?;
    }
    g.synchronize(0)?;
    let (mut e0, mut e1): (u64, u64) = (0, 0);
    // SAFETY: plain CUDA driver calls on events this function creates and destroys; the
    // backend made the context current when it was built.
    unsafe {
        check(cuEventCreate(&mut e0, 0), "cuEventCreate(start)")?;
        check(cuEventCreate(&mut e1, 0), "cuEventCreate(end)")?;
        check(cuEventRecord(e0, 0), "cuEventRecord(start)")?;
    }
    for _ in 0..ITERS {
        f()?;
    }
    let mut ms: f32 = 0.0;
    // SAFETY: as above; `ms` outlives the call.
    unsafe {
        check(cuEventRecord(e1, 0), "cuEventRecord(end)")?;
        check(cuEventSynchronize(e1), "cuEventSynchronize(end)")?;
        check(cuEventElapsedTime(&mut ms, e0, e1), "cuEventElapsedTime")?;
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    Ok(ms as f64 / ITERS as f64)
}

fn timing(g: &dyn GpuBackend, ks: [KernelHandle; 2]) -> Result<()> {
    for &p in TIMING_PS {
        let c = Case {
            q: TIMING_Q,
            p,
            d: 128,
        };
        let b = Bufs::new(g, &inputs(c, 0x7177), c)?;
        let old = time_ms(g, || launch(g, ks[0], false, &b, b.q, c))?;
        let new = time_ms(g, || launch(g, ks[1], true, &b, b.q, c))?;
        let flops = 2.0 * (c.q * c.p * H * c.d) as f64;
        let tf = |ms: f64| flops / (ms * 1e-3) / 1e12;
        println!(
            "TIMING Q={} P={p} D=128 H={H}: dsa_index_scores_tc(bf16) {:.1} us {:.2} TF/s | \
             dsa_index_scores_tc2 {:.1} us {:.2} TF/s | {:.2}x",
            c.q,
            old * 1e3,
            tf(old),
            new * 1e3,
            tf(new),
            old / new
        );
        b.free(g);
    }
    Ok(())
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut ks = [KernelHandle(0); 2];
    let names = ["dsa_index_scores_tc", "dsa_index_scores_tc2"];
    for (slot, func) in names.into_iter().enumerate() {
        match g.kernel(DSA_MODULE, func) {
            Ok(h) => ks[slot] = h,
            Err(e) => {
                println!("{DSA_MODULE}::{func} absent from this target ({e}) - SKIP");
                std::process::exit(2);
            }
        }
    }
    let (mut failed, mut cands, mut pos) = (Vec::new(), 0usize, 0usize);
    for &(q, p, d) in LEGS {
        let c = Case { q, p, d };
        let (ok, n_c, n_p) = leg(g, ks, c)?;
        cands += n_c;
        pos += n_p;
        if !ok {
            failed.push(format!("Q={q} P={p} D={d}"));
        }
    }
    let fired = control(g, ks)?;
    timing(g, ks)?;
    if cands == 0 || pos == 0 {
        println!("FAIL - no candidate or no positive score; the legs prove nothing.");
        std::process::exit(1);
    }
    if !fired {
        println!("FAIL - the negative control did not fire; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if !failed.is_empty() {
        println!(
            "FAIL - {} leg(s) differ ({}); keep METRALE_GLM_DSA_SCORES_TC2 off on this build.",
            failed.len(),
            failed.join(", ")
        );
        std::process::exit(1);
    }
    println!(
        "PASS - {} legs bitwise identical: dsa_index_scores_tc2 matches dsa_index_scores_tc \
         mode 1 (out u32 bits and valid_cand bytes; {cands} candidates, {pos} positive).",
        LEGS.len()
    );
    Ok(())
}
