// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Byte-parity gate (and timing) for two opt-in DSA indexer prefill levers:
//!
//! * `METRALE_GLM_DSA_SCORES_TILED=1`: `dsa_index_scores_tiled` (32 rows x 64 pools per
//!   block since v2) against `dsa_index_scores` (one block per (pool, row)), with the
//!   arguments `select_tokens` passes on an exact launch (the kernels are launched directly,
//!   so `SCORES_TILED_MIN_POOLS` does not apply). Q in {1, 7, 16, 31, 33, 255, 256, 2048}, P in
//!   {1, 63, 64, 65, 2048}, H = 32, D in {128, 96}, two `q_pos` layouts (causal spread and
//!   prefill tail). The data has negative and zero (+0 and -0) weights, zero keys and zero q
//!   heads (+-0 dots), duplicated pool keys (exact ties), invalid pools, invalid end tokens
//!   and pool ends outside [0, S). `out` is compared as u32 bit patterns, `valid_cand` as
//!   bytes.
//! * `METRALE_GLM_DSA_GEMV_SPLIT=1`: one `dense_gemv_batchm_split` launch (grid.y =
//!   ceil(rows / 16)) against one `dense_gemv_batchm[_fp32out]` launch per 16 rows, as
//!   `glm5next_dsa/layer/row_batch.rs` issues them, at rows {1, 15, 16, 17, 255, 256} over
//!   the three indexer shapes (`wq_b` N 4096 K 1536 FP32 out, `weights_proj` N 32 K 4096 FP32
//!   out, `wk` / `compress_gate` N 128 K 4096 BF16 out).
//!
//! Each arm's output starts filled with its own poison byte, so an element one arm never
//! writes cannot compare equal. A negative control per lever (one flipped input bit in the
//! new arm) must be detected, and a run that compared nothing fails. Timing is CUDA events
//! around back-to-back launches (mean per launch, after one warm-up), printed for information;
//! it does not decide PASS. Scores timing sweeps P in {64, 128, 256, 512, 1024, 2048} at
//! Q = 256 (the served `METRALE_GLM_PREFILL_ROWS=256` sub-chunk) and Q = 2048, D 128, to set
//! `SCORES_TILED_MIN_POOLS` (PROVISIONAL).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and `PASS` when every comparison is bitwise identical and both controls fire; 1 and
//! `FAIL` otherwise; 2 when a kernel is absent from this target.
//!
//! Run (both kernel files are gb10 common kernels):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_indexer_tiled_bitparity_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::DSA_MODULE;
use metrale_model_arch::glm5next_dsa::select::{
    SCORES_TILED_BLOCK, scores_tiled_grid, scores_tiled_smem,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

// 2026-10-01: CUDA driver event API for kernel-only timing, declared as in
// `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-01: `SCORES_BLOCK` in `glm5next_dsa/select.rs` (private there): `dsa_index_scores`
/// threads per block, and its least shared memory in bytes.
const OLD_BLOCK: u32 = 128;
/// 2026-10-01: GLM-5.3 `index_n_heads` and `index_kpool`.
const H: usize = 32;
const KP: usize = 4;
const QS: &[usize] = &[1, 7, 16, 31, 33, 255, 256, 2048];
const PS: &[usize] = &[1, 63, 64, 65, 2048];
const DS: &[usize] = &[128, 96];
/// 2026-10-01: (N, K, FP32 out, label): the GLM-5.3 indexer GEMVs of the DSA row batch
/// (hidden 4096, q_lora_rank 1536, index_heads 32, index_head_dim 128).
const GEMV_SHAPES: &[(usize, usize, bool, &str)] = &[
    (4096, 1536, true, "wq_b"),
    (32, 4096, true, "weights_proj"),
    (128, 4096, false, "wk / compress_gate"),
];
const GEMV_ROWS: &[usize] = &[1, 15, 16, 17, 255, 256];
/// 2026-10-01: `DENSE_GEMV_BATCHM_MAX_M`, the per-launch row cap of the per-16-row loop.
const MAX_M: usize = ops::DENSE_GEMV_BATCHM_MAX_M as usize;
const POISON_REF: u8 = 0xA5;
const POISON_NEW: u8 = 0x5A;
const ITERS: usize = 20;
/// 2026-10-01: Scores timing grid (rows per selector call, pools).
const TIMING_QS: &[usize] = &[256, 2048];
const TIMING_PS: &[usize] = &[64, 128, 256, 512, 1024, 2048];

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
    fn bf16_bytes(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f32(self.r(lo, hi)).to_bits().to_le_bytes())
            .collect()
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

fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

/// 2026-10-01: Mean ms per call of `f` over `ITERS` back-to-back calls on stream 0, CUDA
/// events around the loop, after one warm-up call.
fn time_ms(g: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
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

/// 2026-10-01: Tally of one run: compared elements and differing legs.
#[derive(Default)]
struct Tally {
    compared: usize,
    failed: usize,
}

impl Tally {
    fn leg(&mut self, label: &str, elems: usize, a: &[u8], b: &[u8]) {
        let same = a == b;
        self.compared += elems;
        if !same {
            self.failed += 1;
        }
        let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
        println!("{label:<52} elems={elems:<8} byte-identical={same:<5} diff_bytes={diff}");
    }
}

struct Kernels {
    scores: KernelHandle,
    tiled: KernelHandle,
    batchm: KernelHandle,
    batchm_f32: KernelHandle,
}

/// 2026-10-01: One scores case: Q rows, P pools, head dim D, `q_pos` layout.
#[derive(Clone, Copy)]
struct Case {
    q: usize,
    p: usize,
    d: usize,
    tail: bool,
}

/// 2026-10-01: Host inputs of one case, `select_tokens`' argument layout.
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

fn scores_inputs(c: Case, seed: u64) -> Inputs {
    let mut rng = Lcg(seed ^ ((c.q * 131 + c.p * 7 + c.d) as u64).wrapping_mul(0x9E37_79B9));
    let s = c.p * KP + 2;
    // 2026-10-01: q: zero heads (+-0 dots) and -0.0 elements among random values.
    let mut q = vec![0.0f32; c.q * H * c.d];
    for r in 0..c.q {
        for h in 0..H {
            let zero_head = (r * H + h) % 9 == 4;
            for d in 0..c.d {
                let i = (r * H + h) * c.d + d;
                q[i] = if zero_head {
                    0.0
                } else if i % 37 == 11 {
                    -0.0
                } else {
                    rng.r(-1.0, 1.0)
                };
            }
        }
    }
    // 2026-10-01: Keys: zero pools, and pools that duplicate the previous one (exact ties).
    let mut keys = vec![0.0f32; c.p * c.d];
    for p in 0..c.p {
        for d in 0..c.d {
            keys[p * c.d + d] = if p % 11 == 5 {
                0.0
            } else if p % 7 == 3 {
                keys[(p - 1) * c.d + d]
            } else {
                rng.r(-1.0, 1.0)
            };
        }
    }
    let weights = (0..c.q * H)
        .map(|i| match i % 29 {
            3 => 0.0,
            17 => -0.0,
            _ => rng.r(-1.0, 1.0),
        })
        .collect();
    // 2026-10-01: Pool p holds tokens p * KP .. p * KP + KP - 1; some ends lie outside
    // [0, S), which the kernels clamp.
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
    // 2026-10-01: Causal: a prefill tail (rows S - Q .. S - 1), or rows spread over [0, S).
    let q_pos = (0..c.q)
        .map(|r| {
            if c.tail {
                (s as i64 - c.q as i64 + r as i64).max(0) as i32
            } else {
                ((r + 1) * s / c.q) as i32 - 1
            }
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
        s,
    }
}

/// 2026-10-01: Device copies of one case's inputs.
struct Bufs {
    q: DevicePtr,
    keys: DevicePtr,
    weights: DevicePtr,
    pool_indices: DevicePtr,
    pool_valid: DevicePtr,
    valid_keys: DevicePtr,
    q_pos: DevicePtr,
    s: usize,
}

impl Bufs {
    fn new(g: &dyn GpuBackend, i: &Inputs) -> Result<Self> {
        Ok(Self {
            q: up(g, &f32_bytes(&i.q))?,
            keys: up(g, &f32_bytes(&i.keys))?,
            weights: up(g, &f32_bytes(&i.weights))?,
            pool_indices: up(g, &i32_bytes(&i.pool_indices))?,
            pool_valid: up(g, &i.pool_valid)?,
            valid_keys: up(g, &i.valid_keys)?,
            q_pos: up(g, &i32_bytes(&i.q_pos))?,
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
        ];
        for p in all {
            g.free(p).ok();
        }
    }
}

/// 2026-10-01: One scores launch as `select_tokens` issues it on an exact launch: the tiled
/// kernel on its grid / block / shared memory, or `dsa_index_scores` on grid (P, Q), block
/// `OLD_BLOCK`, `max(OLD_BLOCK, 4 * H)` bytes; `geom` NULL either way.
#[allow(clippy::too_many_arguments)]
fn launch_scores(
    g: &dyn GpuBackend,
    ks: &Kernels,
    tiled: bool,
    b: &Bufs,
    q: DevicePtr,
    out: DevicePtr,
    vc: DevicePtr,
    c: Case,
) -> Result<()> {
    let (h, grid, block, smem) = if tiled {
        let (grid, smem) = (scores_tiled_grid(c.q, c.p), scores_tiled_smem(c.d) as u32);
        (ks.tiled, grid, SCORES_TILED_BLOCK, smem)
    } else {
        let smem = OLD_BLOCK.max((H * 4) as u32);
        (ks.scores, [c.p as u32, c.q as u32, 1], OLD_BLOCK, smem)
    };
    KernelLaunch::new(g, h)
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
        .arg_ptr(out)
        .arg_ptr(vc)
        .arg_u32(c.q as u32)
        .arg_u32(c.p as u32)
        .arg_u32(H as u32)
        .arg_u32(c.d as u32)
        .arg_u32(KP as u32)
        .arg_u32(b.s as u32)
        .arg_f32((c.d as f32).powf(-0.5))
        .arg_ptr(DevicePtr::NULL)
        .launch(0)
}

/// 2026-10-01: (out bytes, valid_cand bytes) of one arm, each buffer pre-filled with
/// `poison`; `q` may differ from `b.q` (the negative control).
fn scores_arm(
    g: &dyn GpuBackend,
    ks: &Kernels,
    tiled: bool,
    b: &Bufs,
    q: DevicePtr,
    c: Case,
    poison: u8,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let n = c.q * c.p;
    let (out, vc) = (g.alloc(n * 4)?, g.alloc(n)?);
    g.memset(out, poison, n * 4)?;
    g.memset(vc, poison, n)?;
    launch_scores(g, ks, tiled, b, q, out, vc, c)?;
    let res = (down(g, out, n * 4)?, down(g, vc, n)?);
    g.free(out).ok();
    g.free(vc).ok();
    Ok(res)
}

fn scores_legs(g: &dyn GpuBackend, ks: &Kernels, tally: &mut Tally) -> Result<()> {
    let (mut cands, mut positive) = (0usize, 0usize);
    for &d in DS {
        for &q in QS {
            for &p in PS {
                for tail in [false, true] {
                    let c = Case { q, p, d, tail };
                    let b = Bufs::new(g, &scores_inputs(c, 0x5EED))?;
                    let (ro, rv) = scores_arm(g, ks, false, &b, b.q, c, POISON_REF)?;
                    let (no, nv) = scores_arm(g, ks, true, &b, b.q, c, POISON_NEW)?;
                    let lay = if tail { "tail" } else { "spread" };
                    let name = format!("scores D={d} Q={q} P={p} {lay}");
                    tally.leg(&format!("{name} out"), q * p, &ro, &no);
                    tally.leg(&format!("{name} valid_cand"), q * p, &rv, &nv);
                    cands += rv.iter().filter(|&&v| v == 1).count();
                    positive += ro
                        .chunks_exact(4)
                        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                        .filter(|&s| s > 0.0)
                        .count();
                    b.free(g);
                }
            }
        }
    }
    println!("scores: {cands} candidate outputs, {positive} with a positive score");
    if cands == 0 || positive == 0 {
        println!("scores: no candidate or no positive score - the legs prove nothing");
        tally.failed += 1;
    }
    Ok(())
}

/// 2026-10-01: Negative control: the top mantissa bit of q[Q - 1][0][0] flipped in the tiled
/// arm's input must change its scores against the unflipped reference (row Q - 1 sees every
/// pool under the tail layout).
fn scores_control(g: &dyn GpuBackend, ks: &Kernels) -> Result<bool> {
    let c = Case {
        q: 16,
        p: 64,
        d: 128,
        tail: true,
    };
    let mut inp = scores_inputs(c, 0xC0DE);
    let b = Bufs::new(g, &inp)?;
    let i = (c.q - 1) * H * c.d;
    inp.q[i] = f32::from_bits(inp.q[i].to_bits() ^ (1 << 22));
    let q_pert = up(g, &f32_bytes(&inp.q))?;
    let (ro, _) = scores_arm(g, ks, false, &b, b.q, c, POISON_REF)?;
    let (no, _) = scores_arm(g, ks, true, &b, q_pert, c, POISON_REF)?;
    let fired = ro != no;
    println!("scores CONTROL 1-bit flip in q detected={fired}");
    b.free(g);
    g.free(q_pert).ok();
    Ok(fired)
}

/// 2026-10-01: Timing sweep; `TILED>=old` marks where the tiled kernel is no faster.
fn scores_timing(g: &dyn GpuBackend, ks: &Kernels) -> Result<()> {
    for (q, p) in TIMING_QS
        .iter()
        .flat_map(|&q| TIMING_PS.iter().map(move |&p| (q, p)))
    {
        let c = Case {
            q,
            p,
            d: 128,
            tail: true,
        };
        let b = Bufs::new(g, &scores_inputs(c, 0x7177))?;
        let n = c.q * c.p;
        let (out, vc) = (g.alloc(n * 4)?, g.alloc(n)?);
        let old = time_ms(g, || launch_scores(g, ks, false, &b, b.q, out, vc, c))?;
        let new = time_ms(g, || launch_scores(g, ks, true, &b, b.q, out, vc, c))?;
        let flag = if new >= old { "  TILED>=old" } else { "" };
        println!(
            "TIMING scores Q={q} P={p} D=128 H={H}: dsa_index_scores {old:.4} ms, \
             dsa_index_scores_tiled {new:.4} ms ({:.2}x){flag}",
            old / new
        );
        g.free(out).ok();
        g.free(vc).ok();
        b.free(g);
    }
    Ok(())
}

/// 2026-10-01: `batchm_rows` with the lever off: one launch per `MAX_M` rows.
#[allow(clippy::too_many_arguments)]
fn gemv_loop(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    f32_out: bool,
    a: DevicePtr,
    w: &DenseWeight,
    c: DevicePtr,
    rows: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let elem = if f32_out { 4 } else { 2 };
    let mut r0 = 0;
    while r0 < rows {
        let m = MAX_M.min(rows - r0) as u32;
        let (a_r, c_r) = (a.offset(r0 * k * 2), c.offset(r0 * n * elem));
        let (n32, k32) = (n as u32, k as u32);
        if f32_out {
            ops::dense_gemv_batchm_fp32out(g, kernel, a_r, w, c_r, m, n32, k32, n32, 0)?;
        } else {
            ops::dense_gemv_batchm(g, kernel, a_r, w, c_r, m, n32, k32, n32, 0)?;
        }
        r0 += MAX_M;
    }
    Ok(())
}

/// 2026-10-01: `batchm_rows` with the lever on: one launch, grid.y = ceil(rows / 16).
#[allow(clippy::too_many_arguments)]
fn gemv_split(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w: &DenseWeight,
    c: DevicePtr,
    rows: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let (m, y) = (rows as u32, rows.div_ceil(MAX_M) as u32);
    let (n32, k32) = (n as u32, k as u32);
    ops::dense_gemv_batchm_split(g, kernel, a, w, c, m, y, n32, k32, n32, 0)
}

/// 2026-10-01: (loop bytes, split bytes) for `rows` rows; the split arm reads `a_new`.
#[allow(clippy::too_many_arguments)]
fn gemv_pair(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    f32_out: bool,
    a: DevicePtr,
    a_new: DevicePtr,
    w: &DenseWeight,
    rows: usize,
    n: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let bytes = rows * n * if f32_out { 4 } else { 2 };
    let (c_ref, c_new) = (g.alloc(bytes)?, g.alloc(bytes)?);
    g.memset(c_ref, POISON_REF, bytes)?;
    g.memset(c_new, POISON_NEW, bytes)?;
    gemv_loop(g, kernel, f32_out, a, w, c_ref, rows, n, k)?;
    gemv_split(g, kernel, a_new, w, c_new, rows, n, k)?;
    let out = (down(g, c_ref, bytes)?, down(g, c_new, bytes)?);
    g.free(c_ref).ok();
    g.free(c_new).ok();
    Ok(out)
}

/// 2026-10-01: Every rows x shape leg, the negative control (one ULP flipped in row 16 of the
/// split arm's input at 17 rows, on the FP32-out `wq_b` shape) and the timing.
fn gemv_legs(g: &dyn GpuBackend, ks: &Kernels, tally: &mut Tally) -> Result<bool> {
    let max_rows = *GEMV_ROWS.iter().max().unwrap_or(&1);
    let mut control_ok = true;
    for (si, &(n, k, f32_out, label)) in GEMV_SHAPES.iter().enumerate() {
        let mut rng = Lcg(0xD5A ^ (n * k) as u64);
        let a_bytes = rng.bf16_bytes(max_rows * k, -1.5, 1.5);
        let a = up(g, &a_bytes)?;
        let wt = DenseWeight {
            weight: up(g, &rng.bf16_bytes(n * k, -0.08, 0.08))?,
        };
        let kernel = if f32_out { ks.batchm_f32 } else { ks.batchm };
        let kind = if f32_out { "f32" } else { "bf16" };
        for &rows in GEMV_ROWS {
            let (r, b) = gemv_pair(g, kernel, f32_out, a, a, &wt, rows, n, k)?;
            let name = format!("gemv {label} N={n} K={k} {kind} rows={rows}");
            tally.leg(&name, rows * n, &r, &b);
        }
        if si == 0 {
            let mut pert = a_bytes.clone();
            pert[2 * (16 * k + 3)] ^= 1;
            let a_pert = up(g, &pert)?;
            let (r, b) = gemv_pair(g, kernel, f32_out, a, a_pert, &wt, 17, n, k)?;
            let fired = r != b;
            control_ok &= fired;
            println!("gemv CONTROL 1-ULP flip in row 16 detected={fired}");
            g.free(a_pert).ok();
        }
        for rows in [17usize, 256] {
            let bytes = rows * n * if f32_out { 4 } else { 2 };
            let c = g.alloc(bytes)?;
            let old = time_ms(g, || gemv_loop(g, kernel, f32_out, a, &wt, c, rows, n, k))?;
            let new = time_ms(g, || gemv_split(g, kernel, a, &wt, c, rows, n, k))?;
            println!(
                "TIMING gemv {label} N={n} K={k} rows={rows}: per-16 loop {old:.4} ms, \
                 split {new:.4} ms ({:.2}x)",
                old / new
            );
            g.free(c).ok();
        }
        g.free(a).ok();
        g.free(wt.weight).ok();
    }
    Ok(control_ok)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let lookups = [
        (DSA_MODULE, "dsa_index_scores"),
        (DSA_MODULE, "dsa_index_scores_tiled"),
        ("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
        ("dense_gemv_bf16_batchm", "dense_gemv_bf16_fp32out_batchm"),
    ];
    let mut handles = Vec::new();
    for (module, func) in lookups {
        match g.kernel(module, func) {
            Ok(h) => handles.push(h),
            Err(e) => {
                println!("{module}::{func} absent from this target ({e}) - SKIP");
                std::process::exit(2);
            }
        }
    }
    let ks = Kernels {
        scores: handles[0],
        tiled: handles[1],
        batchm: handles[2],
        batchm_f32: handles[3],
    };

    let mut tally = Tally::default();
    scores_legs(g, &ks, &mut tally)?;
    let scores_control_ok = scores_control(g, &ks)?;
    let gemv_control_ok = gemv_legs(g, &ks, &mut tally)?;
    scores_timing(g, &ks)?;

    if tally.compared == 0 {
        println!("FAIL - no element was compared; this run proves nothing.");
        std::process::exit(1);
    }
    if !scores_control_ok || !gemv_control_ok {
        println!("FAIL - a negative control did not fire; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if tally.failed > 0 {
        println!(
            "FAIL - {} leg(s) differ. Keep METRALE_GLM_DSA_SCORES_TILED / \
             METRALE_GLM_DSA_GEMV_SPLIT off on this build.",
            tally.failed
        );
        std::process::exit(1);
    }
    println!(
        "PASS - {} elements byte-identical: dsa_index_scores_tiled matches dsa_index_scores \
         (out bits and valid_cand) and the y-split indexer GEMVs match the per-16-row loop.",
        tally.compared
    );
    Ok(())
}
