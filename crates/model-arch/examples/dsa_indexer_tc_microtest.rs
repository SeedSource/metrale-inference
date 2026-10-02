// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Accuracy and timing gate for `METRALE_GLM_DSA_SCORES_TC`
//! (`dsa_index_scores_tc`, tensor-core DSA indexer scores) against `dsa_index_scores`, the
//! FP32 scorer it replaces, launched as `select_tokens` launches them on an exact launch
//! (directly, so `SCORES_TC_MIN_POOLS` does not apply).
//!
//! The tensor-core kernel is NOT byte-identical by design, so this is not a bit-parity gate.
//! It checks, per precision mode (1 bf16, 2 split2, 3 split3):
//!
//! * Exact legs (every mode, hard): `valid_cand` byte-identical to the reference, and every
//!   non-candidate score the exact `-FLT_MAX` bit pattern. Shapes Q {1, 7, 16, 17, 256} x
//!   P {1, 63, 65, 512, 2048} x D {128, 96, 64}, H 32, causal-spread and prefill-tail
//!   `q_pos`, with the tiled gate's edge data (zero / negative weights, zero keys and q heads,
//!   duplicated keys, invalid pools, invalid and out-of-range pool ends).
//! * Error (hard for split3, information for bf16 / split2): the largest |tc - ref| over
//!   candidates divided by its bound `scale * sum_h |w_h| ||q_h|| * ||k_p||` (see `Bound`;
//!   independent of cancellation inside a score); split3 must stay at or under `SPLIT3_TOL`,
//!   bf16 / split2 are flagged above `INFO_TOL` (2^-7) for information only.
//! * Selection recall at 32K tokens (Q 256 prefill tail, P 8192 pools, uniform random data,
//!   the worst case for near ties): per row, the top `select_k` = 512 pools by (score desc,
//!   pool asc) among candidates, as `dsa_topk_pools` orders them; recall = |tc n ref| / k.
//!   Mean recall gates split3 at `SPLIT3_MIN_RECALL`; bf16 / split2 are information.
//! * Planted needles (every mode, hard): 32 pools aligned with a direction every query head
//!   leans toward, positive weights; every needle must be in every row's top 512 for the
//!   reference (else the leg is malformed) and for every mode.
//! * Negative control: split3 on a q perturbed by 1 % must exceed `SPLIT3_TOL`, or the error
//!   check is vacuous.
//!
//! Timing (information; CUDA events, mean per launch after one warm-up): for prompts of 32K
//! and 64K tokens at `METRALE_GLM_PREFILL_ROWS` 16 (default) and 256 (served), one selection
//! pass at the prompt's average pool count (N / 8 at index_kpool 4) for `dsa_index_scores`,
//! `dsa_index_scores_tiled` (when present, D 128) and the three tensor-core modes, plus
//! `dsa_kpool_compress` and `dsa_topk_pools` at the same pool count; the projected per-prompt
//! indexer seconds are passes (N / rows) x per-pass ms x 11 DSA layers.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and `PASS` when every hard check holds and the control fires; 1 and `FAIL`
//! otherwise; 2 when `dsa_index_scores` or `dsa_index_scores_tc` is absent from this target.
//!
//! Run (dsa_indexer.cu is a gb10 common kernel):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_indexer_tc_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::DSA_MODULE;
use metrale_model_arch::glm5next_dsa::select::{
    SCORES_TC_BLOCK, SCORES_TILED_BLOCK, scores_tc_grid, scores_tiled_grid, scores_tiled_smem,
    topk_smem_for_tile, topk_tile,
};

// 2026-10-01: CUDA driver event API for kernel-only timing, declared as in
// `dsa_indexer_tiled_bitparity_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-01: `SCORES_BLOCK` in `glm5next_dsa/select.rs` (private there).
const OLD_BLOCK: u32 = 128;
/// 2026-10-01: GLM-5.3 `index_n_heads`, `index_kpool`, `index_topk / index_kpool`, and DSA
/// layers.
const H: usize = 32;
const KP: usize = 4;
const SELECT_K: usize = 512;
const DSA_LAYERS: f64 = 11.0;
const QS: &[usize] = &[1, 7, 16, 17, 256];
const PS: &[usize] = &[1, 63, 65, 512, 2048];
const DS: &[usize] = &[128, 96, 64];
const MODES: &[(u32, &str)] = &[(1, "bf16"), (2, "split2"), (3, "split3")];
/// 2026-10-01: split3 error and recall gates (PROVISIONAL, from the error estimate in the
/// kernel header: about 2^-16 of the bound per dot, 2^-8 for bf16 / split2). bf16 / split2
/// flags are information.
const SPLIT3_TOL: f64 = 1e-4;
const SPLIT3_MIN_RECALL: f64 = 0.999;
const INFO_TOL: f64 = 8e-3;
const NEEDLES: usize = 32;
const POISON: u8 = 0xA5;
const ITERS: usize = 10;
/// 2026-10-01: Prompt lengths and selector rows per pass for the timing projection.
const TIMING_NS: &[usize] = &[32_768, 65_536];
const TIMING_QS: &[usize] = &[16, 256];
const NEG_FLT_MAX_BITS: u32 = 0xff7f_ffff;

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

fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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

struct Kernels {
    scores: KernelHandle,
    tc: KernelHandle,
    /// 2026-10-01: Timing only; `KernelHandle(0)` when absent.
    tiled: KernelHandle,
    compress: KernelHandle,
    topk: KernelHandle,
}

/// 2026-10-01: Which scorer a launch runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Ref,
    Tiled,
    Tc(u32),
}

/// 2026-10-01: Data layout of one case.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Data {
    /// 2026-10-01: The tiled gate's edge data (zeros, -0, ties, invalid pools / ends).
    Edge,
    /// 2026-10-01: Clean uniform data, every pool valid: the recall worst case.
    Uniform,
    /// 2026-10-01: Uniform noise plus `NEEDLES` planted pools, positive weights.
    Needle,
}

#[derive(Clone, Copy)]
struct Case {
    q: usize,
    p: usize,
    d: usize,
    tail: bool,
    data: Data,
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
    needles: Vec<usize>,
}

fn inputs(c: Case, seed: u64) -> Inputs {
    let mut rng = Lcg(seed ^ ((c.q * 131 + c.p * 7 + c.d) as u64).wrapping_mul(0x9E37_79B9));
    let s = c.p * KP + 2;
    let edge = c.data == Data::Edge;
    // 2026-10-01: Needle direction u; every q head leans toward it, needle keys carry it.
    let u: Vec<f32> = (0..c.d).map(|_| rng.r(-1.0, 1.0)).collect();
    let needles: Vec<usize> = if c.data == Data::Needle {
        (0..NEEDLES).map(|i| (i * 2 + 1) * c.p / (2 * NEEDLES + 2)).collect()
    } else {
        Vec::new()
    };
    let mut q = vec![0.0f32; c.q * H * c.d];
    for r in 0..c.q {
        for h in 0..H {
            let zero_head = edge && (r * H + h) % 9 == 4;
            for d in 0..c.d {
                let i = (r * H + h) * c.d + d;
                q[i] = if zero_head {
                    0.0
                } else if edge && i % 37 == 11 {
                    -0.0
                } else if c.data == Data::Needle {
                    0.5 * u[d] + rng.r(-0.5, 0.5)
                } else {
                    rng.r(-1.0, 1.0)
                };
            }
        }
    }
    let mut keys = vec![0.0f32; c.p * c.d];
    for p in 0..c.p {
        for d in 0..c.d {
            keys[p * c.d + d] = if edge && p % 11 == 5 {
                0.0
            } else if edge && p % 7 == 3 && p > 0 {
                keys[(p - 1) * c.d + d]
            } else if needles.contains(&p) {
                2.0 * u[d] + rng.r(-0.1, 0.1)
            } else {
                rng.r(-1.0, 1.0)
            };
        }
    }
    let weights = (0..c.q * H)
        .map(|i| match (c.data, i % 29) {
            (Data::Needle, _) => rng.r(0.1, 1.0),
            (Data::Edge, 3) => 0.0,
            (Data::Edge, 17) => -0.0,
            _ => rng.r(-1.0, 1.0),
        })
        .collect();
    let mut pool_indices = vec![0i32; c.p * KP];
    for p in 0..c.p {
        for slot in 0..KP {
            pool_indices[p * KP + slot] = (p * KP + slot) as i32;
        }
        let end = &mut pool_indices[p * KP + KP - 1];
        if edge && p % 13 == 6 {
            *end = s as i32 + 3;
        } else if edge && p % 17 == 8 {
            *end = -2;
        }
    }
    let pool_valid = (0..c.p).map(|p| u8::from(!edge || p % 19 != 7)).collect();
    let valid_keys = (0..s).map(|t| u8::from(!edge || t % 23 != 11)).collect();
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
        needles,
    }
}

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
            out: g.alloc((c.q * c.p * 4).max(1))?,
            vc: g.alloc((c.q * c.p).max(1))?,
            s: i.s,
        })
    }

    fn free(&self, g: &dyn GpuBackend) {
        for p in [
            self.q,
            self.keys,
            self.weights,
            self.pool_indices,
            self.pool_valid,
            self.valid_keys,
            self.q_pos,
            self.out,
            self.vc,
        ] {
            g.free(p).ok();
        }
    }
}

/// 2026-10-01: One scores launch as `select_tokens` issues it on an exact launch (`geom`
/// NULL), reading `q` (which may differ from `b.q`, the negative control).
fn launch_scores(
    g: &dyn GpuBackend,
    ks: &Kernels,
    arm: Arm,
    b: &Bufs,
    q: DevicePtr,
    c: Case,
) -> Result<()> {
    let (h, grid, block, smem) = match arm {
        Arm::Ref => {
            let smem = OLD_BLOCK.max((H * 4) as u32);
            (ks.scores, [c.p as u32, c.q as u32, 1], OLD_BLOCK, smem)
        }
        Arm::Tiled => {
            let smem = scores_tiled_smem(c.d) as u32;
            (ks.tiled, scores_tiled_grid(c.q, c.p), SCORES_TILED_BLOCK, smem)
        }
        Arm::Tc(_) => (ks.tc, scores_tc_grid(c.q, c.p), SCORES_TC_BLOCK, 0),
    };
    let l = KernelLaunch::new(g, h)
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
        .arg_ptr(DevicePtr::NULL);
    let l = if let Arm::Tc(mode) = arm { l.arg_u32(mode) } else { l };
    l.launch(0)
}

/// 2026-10-01: (scores, valid_cand) of one arm, both buffers poisoned first.
fn run_arm(
    g: &dyn GpuBackend,
    ks: &Kernels,
    arm: Arm,
    b: &Bufs,
    q: DevicePtr,
    c: Case,
) -> Result<(Vec<f32>, Vec<u8>)> {
    g.memset_async(b.out, POISON, c.q * c.p * 4, 0)?;
    g.memset_async(b.vc, POISON, c.q * c.p, 0)?;
    launch_scores(g, ks, arm, b, q, c)?;
    let out = as_f32(&down(g, b.out, c.q * c.p * 4)?);
    let vc = down(g, b.vc, c.q * c.p)?;
    Ok((out, vc))
}

/// 2026-10-01: Top `k` candidate pools of one row, ordered as `dsa_topk_pools` orders them
/// (score descending, then pool ascending).
fn top_k(row: &[f32], vc: &[u8], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).filter(|&p| vc[p] != 0).collect();
    idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// 2026-10-01: Result of comparing one arm against the reference on one case.
#[derive(Default)]
struct Cmp {
    exact_ok: bool,
    max_rel: f64,
    recall_mean: f64,
    recall_min: f64,
    rows_same_set: usize,
    rows_ranked: usize,
    needles_missed: usize,
}

/// 2026-10-01: Per-row and per-pool factors of the error bound. Since relu is 1-Lipschitz,
/// |tc - ref| <= sum_h |w_h| * scale * |d dot_h| and |d dot_h| <= eps * ||q_h|| * ||k_p||
/// (Cauchy-Schwarz over the per-element errors), so the error is measured against
/// `scale * a[r] * kn[p]` with a[r] = sum_h |w_rh| * ||q_rh||: about 2^-16 for split3 and 2^-8
/// for bf16 / split2, independent of cancellation inside a score.
struct Bound {
    a: Vec<f64>,
    kn: Vec<f64>,
    scale: f64,
}

impl Bound {
    fn new(c: Case, i: &Inputs) -> Self {
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        let a = (0..c.q)
            .map(|r| {
                (0..H)
                    .map(|h| {
                        let qh = &i.q[(r * H + h) * c.d..(r * H + h + 1) * c.d];
                        (i.weights[r * H + h] as f64).abs() * norm(qh)
                    })
                    .sum()
            })
            .collect();
        let kn = (0..c.p).map(|p| norm(&i.keys[p * c.d..(p + 1) * c.d])).collect();
        Self {
            a,
            kn,
            scale: (c.d as f64).powf(-0.5),
        }
    }
}

fn compare(
    c: Case,
    bound: &Bound,
    r: &(Vec<f32>, Vec<u8>),
    t: &(Vec<f32>, Vec<u8>),
    needles: &[usize],
) -> Cmp {
    let mut m = Cmp {
        exact_ok: r.1 == t.1,
        recall_min: 1.0,
        ..Cmp::default()
    };
    let mut recall_sum = 0.0;
    for row in 0..c.q {
        let lo = row * c.p;
        let (rs, ts) = (&r.0[lo..lo + c.p], &t.0[lo..lo + c.p]);
        let rv = &r.1[lo..lo + c.p];
        for p in 0..c.p {
            if rv[p] == 1 {
                let diff = (ts[p] as f64 - rs[p] as f64).abs();
                let den = bound.scale * bound.a[row] * bound.kn[p];
                let e = if diff == 0.0 {
                    0.0
                } else if den > 0.0 && diff.is_finite() {
                    diff / den
                } else {
                    f64::INFINITY
                };
                m.max_rel = m.max_rel.max(e);
            } else if ts[p].to_bits() != NEG_FLT_MAX_BITS || rs[p].to_bits() != NEG_FLT_MAX_BITS {
                m.exact_ok = false;
            }
        }
        let n_cand = rv.iter().filter(|&&v| v == 1).count();
        if n_cand > SELECT_K {
            let a = top_k(rs, rv, SELECT_K);
            let b = top_k(ts, rv, SELECT_K);
            let hit = b.iter().filter(|p| a.contains(p)).count();
            let rec = hit as f64 / SELECT_K as f64;
            recall_sum += rec;
            m.recall_min = m.recall_min.min(rec);
            m.rows_ranked += 1;
            if hit == SELECT_K {
                m.rows_same_set += 1;
            }
            for n in needles {
                if rv[*n] == 1 && !b.contains(n) {
                    m.needles_missed += 1;
                }
            }
        }
    }
    m.recall_mean = if m.rows_ranked > 0 {
        recall_sum / m.rows_ranked as f64
    } else {
        1.0
    };
    m
}

/// 2026-10-01: Needles the reference itself misses (a malformed needle leg when nonzero).
fn ref_needles_missed(c: Case, r: &(Vec<f32>, Vec<u8>), needles: &[usize]) -> usize {
    let mut missed = 0;
    for row in 0..c.q {
        let lo = row * c.p;
        let (rs, rv) = (&r.0[lo..lo + c.p], &r.1[lo..lo + c.p]);
        let top = top_k(rs, rv, SELECT_K);
        missed += needles.iter().filter(|n| rv[**n] == 1 && !top.contains(n)).count();
    }
    missed
}

#[derive(Default)]
struct Verdict {
    legs: usize,
    failed: Vec<String>,
}

fn case_label(c: Case) -> String {
    let data = match c.data {
        Data::Edge => "edge",
        Data::Uniform => "uniform",
        Data::Needle => "needle",
    };
    format!(
        "Q={:<4} P={:<5} D={:<3} {} {data}",
        c.q,
        c.p,
        c.d,
        if c.tail { "tail  " } else { "spread" }
    )
}

/// 2026-10-01: Run every mode on one case and record the hard checks.
fn run_case(g: &dyn GpuBackend, ks: &Kernels, c: Case, v: &mut Verdict) -> Result<()> {
    let i = inputs(c, 0xD5A7C);
    let b = Bufs::new(g, &i, c)?;
    let bound = Bound::new(c, &i);
    let r = run_arm(g, ks, Arm::Ref, &b, b.q, c)?;
    if c.data == Data::Needle {
        let miss = ref_needles_missed(c, &r, &i.needles);
        if miss > 0 {
            let tag = case_label(c);
            v.failed.push(format!("{tag} reference misses {miss} needles: malformed leg"));
        }
    }
    for &(mode, name) in MODES {
        let t = run_arm(g, ks, Arm::Tc(mode), &b, b.q, c)?;
        let m = compare(c, &bound, &r, &t, &i.needles);
        v.legs += 1;
        let tol = if mode == 3 { SPLIT3_TOL } else { INFO_TOL };
        let tol_flag = if m.max_rel <= tol { "" } else { " OVER-TOL" };
        println!(
            "{} {name:<6} exact={:<5} max_rel={:.3e}{tol_flag} recall mean={:.5} min={:.4} \
             same_set={}/{} needles_missed={}",
            case_label(c),
            m.exact_ok,
            m.max_rel,
            m.recall_mean,
            m.recall_min,
            m.rows_same_set,
            m.rows_ranked,
            m.needles_missed
        );
        let tag = format!("{} {name}", case_label(c));
        if !m.exact_ok {
            v.failed.push(format!("{tag}: candidacy / -FLT_MAX differs"));
        }
        if m.needles_missed > 0 {
            v.failed.push(format!("{tag}: {} needles missed", m.needles_missed));
        }
        if mode == 3 && m.max_rel > SPLIT3_TOL {
            v.failed.push(format!("{tag}: max_rel {:.3e} > {SPLIT3_TOL:e}", m.max_rel));
        }
        if mode == 3 && c.data == Data::Uniform && m.recall_mean < SPLIT3_MIN_RECALL {
            v.failed.push(format!("{tag}: recall {:.5} < {SPLIT3_MIN_RECALL}", m.recall_mean));
        }
    }
    b.free(g);
    Ok(())
}

/// 2026-10-01: split3 on a q scaled by 1.01 must exceed `SPLIT3_TOL` against the reference.
fn control(g: &dyn GpuBackend, ks: &Kernels) -> Result<bool> {
    let c = Case {
        q: 16,
        p: 512,
        d: 128,
        tail: true,
        data: Data::Uniform,
    };
    let i = inputs(c, 0xC0);
    let b = Bufs::new(g, &i, c)?;
    let q2: Vec<f32> = i.q.iter().map(|x| x * 1.01).collect();
    let q2p = up(g, &f32_bytes(&q2))?;
    let r = run_arm(g, ks, Arm::Ref, &b, b.q, c)?;
    let t = run_arm(g, ks, Arm::Tc(3), &b, q2p, c)?;
    let m = compare(c, &Bound::new(c, &i), &r, &t, &[]);
    let fired = m.max_rel > SPLIT3_TOL;
    println!(
        "control: split3 on q x 1.01 max_rel={:.3e} -> {}",
        m.max_rel,
        if fired { "detected" } else { "NOT detected" }
    );
    g.free(q2p).ok();
    b.free(g);
    Ok(fired)
}

/// 2026-10-01: Per-pass timing at the average pool count of 32K / 64K prompts and the
/// projected per-prompt indexer seconds over 11 DSA layers.
fn timing(g: &dyn GpuBackend, ks: &Kernels) -> Result<()> {
    println!(
        "\ntiming (ms per selection pass at the prompt's average pool count; projected s per \
         prompt x {DSA_LAYERS} layers):"
    );
    for &n in TIMING_NS {
        let p = n / (2 * KP);
        let s_tok = p * KP;
        // 2026-10-01: Compress and top-k at this pool count (independent of the scorer).
        let d = 128usize;
        let mut rng = Lcg(n as u64);
        let kb: Vec<u8> = (0..s_tok * d)
            .flat_map(|_| bf16::from_f32(rng.r(-1.0, 1.0)).to_bits().to_le_bytes())
            .collect();
        let gate: Vec<u8> = (0..s_tok * d)
            .flat_map(|_| bf16::from_f32(rng.r(-1.0, 1.0)).to_bits().to_le_bytes())
            .collect();
        let kdev = up(g, &kb)?;
        let gdev = up(g, &gate)?;
        let valid = up(g, &vec![1u8; s_tok])?;
        let ape = up(g, &f32_bytes(&(0..KP * d).map(|_| rng.r(-0.1, 0.1)).collect::<Vec<_>>()))?;
        let pk = g.alloc(p * d * 4)?;
        let pi = g.alloc(p * KP * 4)?;
        let pv = g.alloc(p)?;
        let compress_ms = time_ms(g, || {
            KernelLaunch::new(g, ks.compress)
                .grid([p as u32, 1, 1])
                .block([d as u32, 1, 1])
                .arg_ptr(kdev)
                .arg_ptr(gdev)
                .arg_ptr(valid)
                .arg_ptr(ape)
                .arg_ptr(pk)
                .arg_ptr(pi)
                .arg_ptr(pv)
                .arg_u32(s_tok as u32)
                .arg_u32(d as u32)
                .arg_u32(KP as u32)
                .arg_i32(0)
                .arg_ptr(DevicePtr::NULL)
                .launch(0)
        })?;
        for p_free in [kdev, gdev, valid, ape, pk, pi, pv] {
            g.free(p_free).ok();
        }
        for &qr in TIMING_QS {
            let passes = (n / qr) as f64;
            let proj = |ms: f64| passes * ms * DSA_LAYERS / 1000.0;
            let c = Case {
                q: qr,
                p,
                d,
                tail: true,
                data: Data::Uniform,
            };
            let i = inputs(c, 0x7111);
            let b = Bufs::new(g, &i, c)?;
            let mut arms: Vec<(String, Arm)> = vec![("dsa_index_scores".into(), Arm::Ref)];
            if ks.tiled.0 != 0 {
                arms.push(("dsa_index_scores_tiled".into(), Arm::Tiled));
            }
            for &(mode, name) in MODES {
                arms.push((format!("dsa_index_scores_tc {name}"), Arm::Tc(mode)));
            }
            let mut base = 0.0;
            for (label, arm) in arms {
                let ms = time_ms(g, || launch_scores(g, ks, arm, &b, b.q, c))?;
                if arm == Arm::Ref {
                    base = ms;
                }
                println!(
                    "  N={n:<6} rows={qr:<4} P={p:<5} {label:<28} {ms:>9.4} ms ({:>6.2}x)  -> \
                     {:>8.2} s/prompt",
                    base / ms.max(1e-9),
                    proj(ms)
                );
            }
            let np2 = p.next_power_of_two().max(2).min(topk_tile());
            let sel = g.alloc(qr * SELECT_K * 4)?;
            let topk_ms = time_ms(g, || {
                KernelLaunch::new(g, ks.topk)
                    .grid([qr as u32, 1, 1])
                    .block([256, 1, 1])
                    .shared_mem(topk_smem_for_tile(np2) as u32)
                    .arg_ptr(b.out)
                    .arg_ptr(sel)
                    .arg_u32(qr as u32)
                    .arg_u32(p as u32)
                    .arg_u32(np2 as u32)
                    .arg_u32(SELECT_K.min(p) as u32)
                    .arg_ptr(DevicePtr::NULL)
                    .launch(0)
            })?;
            g.free(sel).ok();
            println!(
                "  N={n:<6} rows={qr:<4} P={p:<5} {:<28} {topk_ms:>9.4} ms           -> \
                 {:>8.2} s/prompt",
                "dsa_topk_pools",
                proj(topk_ms)
            );
            println!(
                "  N={n:<6} rows={qr:<4} P={p:<5} {:<28} {compress_ms:>9.4} ms           -> \
                 {:>8.2} s/prompt",
                "dsa_kpool_compress",
                proj(compress_ms)
            );
            b.free(g);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut handles = Vec::new();
    let funcs = [
        "dsa_index_scores",
        "dsa_index_scores_tc",
        "dsa_kpool_compress",
        "dsa_topk_pools",
    ];
    for func in funcs {
        match g.kernel(DSA_MODULE, func) {
            Ok(h) => handles.push(h),
            Err(e) => {
                println!("{DSA_MODULE}::{func} absent from this target ({e}) - SKIP");
                std::process::exit(2);
            }
        }
    }
    let ks = Kernels {
        scores: handles[0],
        tc: handles[1],
        tiled: g
            .kernel(DSA_MODULE, "dsa_index_scores_tiled")
            .unwrap_or(KernelHandle(0)),
        compress: handles[2],
        topk: handles[3],
    };

    let mut v = Verdict::default();
    for &d in DS {
        for &q in QS {
            for &p in PS {
                for tail in [false, true] {
                    let c = Case {
                        q,
                        p,
                        d,
                        tail,
                        data: Data::Edge,
                    };
                    run_case(g, &ks, c, &mut v)?;
                }
            }
        }
    }
    // 2026-10-01: 32K-token legs: recall on clean uniform data, then planted needles.
    for data in [Data::Uniform, Data::Needle] {
        let c = Case {
            q: 256,
            p: 8192,
            d: 128,
            tail: true,
            data,
        };
        run_case(g, &ks, c, &mut v)?;
    }
    let control_ok = control(g, &ks)?;
    timing(g, &ks)?;

    if v.legs == 0 {
        println!("FAIL - no leg ran; this run proves nothing.");
        std::process::exit(1);
    }
    if !control_ok {
        println!("FAIL - the negative control did not fire; the error check is VACUOUS.");
        std::process::exit(1);
    }
    if !v.failed.is_empty() {
        for f in &v.failed {
            println!("  failed: {f}");
        }
        println!("FAIL - {} of {} legs failed a hard check.", v.failed.len(), v.legs);
        std::process::exit(1);
    }
    println!(
        "PASS - {} legs: candidacy and -FLT_MAX exact in every mode, split3 max_rel <= \
         {SPLIT3_TOL:e} and 32K recall >= {SPLIT3_MIN_RECALL}, every planted needle \
         selected; control fired.",
        v.legs
    );
    Ok(())
}
