// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Byte-parity gate (and timing) for `METRALE_GLM_DSA_SCORES_DECODE=1`:
//! `dsa_index_scores_decode` against the plain `dsa_index_scores`, both launched as
//! `select_tokens` launches them on a ceiling (graph-replay decode) selection: `q_rows = 1`, the
//! production 512K ceiling (`--max-seq-len` 540,672, 135,168 pools) as the scalar arguments, a
//! device `geom` filled by the production `dsa_write_geom` from each row's `seq_len`, the plain
//! kernel on its production grid (the grid-stride grid when the module has the marker), the
//! decode kernel on `scores_decode_grid_x` with `scores_decode_smem`.
//!
//! * Live contexts S: 131,072 and 262,144 (the car's), a ragged 131,072 + 4 x 37 + 3, and small
//!   edges (0 pools, 1 pool, 31 / 32 / 33 pools, 129 pools + 1 token). Per S, the three verify
//!   rows of a K = 3 step (row r: `seq_len` S + r, `q_pos` S + r - 1, so rows 1 and 2 carry a
//!   partial trailing pool) and a fourth row whose `q_pos` is S / 2 (half the pools are not
//!   visible: the candidacy path), each with its own q and weights.
//! * Data: random keys with zero pools and duplicated pools (exact ties); q with zero heads and
//!   -0.0 elements; weights with +0, -0 and negatives; invalid pools (`pool_valid` 0), invalid
//!   end tokens (every 23rd token), pool ends past S and below 0 (clamped by the kernels).
//! * Compare: each arm's `out` / `valid_cand` start filled with its own poison byte; the live
//!   pools `[0, P)` are compared as raw bytes (`out` as u32 bit patterns), and past P both arms
//!   must have left their poison untouched. A negative control (one bit of q flipped in the
//!   decode arm's input) must be detected; a run that compared nothing fails.
//! * Timing: per S in {131,072; 262,144; ragged}, a captured graph of 33 launches (11 layers,
//!   each its own key buffer, x 3 verify rows), CUDA events around each replay, median of 21
//!   replays after 3 warm-ups: `TIMING scores_plain S=.. ms_per_step=..` and
//!   `TIMING scores_decode S=.. ms_per_step=..`. Timing never decides the verdict.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and a final line starting `PASS` when every compare is byte-identical and the
//! control fires; 1 and a `FAIL` line otherwise; 2 when a kernel is absent from this target.
//!
//! Run (gb10 common kernels):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_scores_decode_bitparity_microtest
//! Needs about 0.8 GB of device memory (11 layers' pool keys at the 512K ceiling).

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::DSA_MODULE;
use metrale_model_arch::glm5next_dsa::select::grid_stride::{
    GRID_STRIDE_MARKER, ceiling_grids, dsa_grid_stride,
};
use metrale_model_arch::glm5next_dsa::select::scores_decode::{
    SCORES_DECODE_BLOCK, scores_decode_grid_x, scores_decode_smem,
};
use metrale_model_arch::glm5next_dsa::select::topk_tile;

// 2026-10-08: CUDA driver event API for kernel-only timing, declared as in
// `dsa_depth_decode_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-08: GLM-5.3 `index_n_heads`, `index_head_dim`, `index_kpool`, `index_topk`.
const H: usize = 32;
const D: usize = 128;
const KP: usize = 4;
const TOPK: usize = 2048;
/// 2026-10-08: `--max-seq-len` of the production 512K row and its pool capacity, as
/// `dsa_depth_rig` (`contiguous_pool_count(KP, MAX_SEQ)`).
const MAX_SEQ: usize = 540_672;
const MAX_POOLS: usize = MAX_SEQ / KP;
/// 2026-10-08: `SCORES_BLOCK` in `glm5next_dsa/select.rs` (private there).
const PLAIN_BLOCK: u32 = 128;
/// 2026-10-08: DSA text layers in a decode step and verify rows per layer at K = 3; row
/// case `ROWS` is the extra half-`q_pos` row of the parity legs.
const LAYERS: usize = 11;
const ROWS: usize = 3;
const ROW_CASES: usize = ROWS + 1;
const RAGGED: usize = 131_072 + 4 * 37 + 3;
const PARITY_S: &[usize] = &[
    3,
    4,
    31 * KP,
    32 * KP,
    33 * KP + 2,
    129 * KP + 1,
    131_072,
    RAGGED,
    262_144,
];
const TIMING_S: &[usize] = &[131_072, 262_144, RAGGED];
const REPLAYS: usize = 21;
const POISON_REF: u8 = 0xA5;
const POISON_NEW: u8 = 0x5A;

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

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

struct Kernels {
    plain: KernelHandle,
    decode: KernelHandle,
    write_geom: KernelHandle,
}

/// 2026-10-08: Device inputs: per-layer pool keys (layer 0 is the parity data), per-row-case q
/// and weights, the shared pool indices / validity / token validity, and per-row-case
/// `seq_len`, `q_pos` and `geom`.
struct Fixture {
    keys: Vec<DevicePtr>,
    q: DevicePtr,
    q_flip: DevicePtr,
    weights: DevicePtr,
    pool_indices: DevicePtr,
    pool_valid: DevicePtr,
    valid_keys: DevicePtr,
    seq_len: Vec<DevicePtr>,
    q_pos: Vec<DevicePtr>,
    geom: Vec<DevicePtr>,
    /// 2026-10-08: Grid x of the plain and the decode kernel at the ceiling.
    grids: (usize, usize),
}

impl Fixture {
    fn new(g: &dyn GpuBackend, marker: bool) -> Result<Self> {
        let mut rng = Lcg(0xDEC0_DE5C);
        // 2026-10-08: Keys: zero pools, and pools that duplicate the previous one (exact ties).
        let mut keys = vec![0.0f32; MAX_POOLS * D];
        for p in 0..MAX_POOLS {
            for d in 0..D {
                keys[p * D + d] = if p % 11 == 5 {
                    0.0
                } else if p % 7 == 3 {
                    keys[(p - 1) * D + d]
                } else {
                    rng.r(-1.0, 1.0)
                };
            }
        }
        let key_bytes = f32_bytes(&keys);
        drop(keys);
        let keys = (0..LAYERS)
            .map(|_| up(g, &key_bytes))
            .collect::<Result<Vec<_>>>()?;
        // 2026-10-08: q: zero heads (+-0 dots) and -0.0 elements among random values.
        let mut q = vec![0.0f32; ROW_CASES * H * D];
        for (i, x) in q.iter_mut().enumerate() {
            let head = i / D;
            *x = if head % 9 == 4 {
                0.0
            } else if i % 37 == 11 {
                -0.0
            } else {
                rng.r(-1.0, 1.0)
            };
        }
        let mut q_flip = q.clone();
        q_flip[3] = f32::from_bits(q_flip[3].to_bits() ^ (1 << 22));
        let weights: Vec<f32> = (0..ROW_CASES * H)
            .map(|i| match i % 29 {
                3 => 0.0,
                17 => -0.0,
                _ => rng.r(-1.0, 1.0),
            })
            .collect();
        // 2026-10-08: Pool p holds tokens p * KP .. p * KP + KP - 1; some ends lie past any S
        // or below 0, which the kernels clamp.
        let mut pool_indices = vec![0i32; MAX_POOLS * KP];
        for p in 0..MAX_POOLS {
            for slot in 0..KP {
                pool_indices[p * KP + slot] = (p * KP + slot) as i32;
            }
            let end = &mut pool_indices[p * KP + KP - 1];
            if p % 13 == 6 {
                *end = (MAX_SEQ + 3) as i32;
            } else if p % 17 == 8 {
                *end = -2;
            }
        }
        let pool_valid: Vec<u8> = (0..MAX_POOLS).map(|p| u8::from(p % 19 != 7)).collect();
        let valid_keys: Vec<u8> = (0..MAX_SEQ + 8).map(|t| u8::from(t % 23 != 11)).collect();
        let sms = g.sm_count().map_or(48, |n| n as usize).max(1);
        let stride = dsa_grid_stride() && marker;
        let mut per_row = |_| -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
            Ok((g.alloc(4)?, g.alloc(4)?, g.alloc(6 * 4)?))
        };
        let rows = (0..ROW_CASES)
            .map(&mut per_row)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            keys,
            q: up(g, &f32_bytes(&q))?,
            q_flip: up(g, &f32_bytes(&q_flip))?,
            weights: up(g, &f32_bytes(&weights))?,
            pool_indices: up(g, &i32_bytes(&pool_indices))?,
            pool_valid: up(g, &pool_valid)?,
            valid_keys: up(g, &valid_keys)?,
            seq_len: rows.iter().map(|r| r.0).collect(),
            q_pos: rows.iter().map(|r| r.1).collect(),
            geom: rows.iter().map(|r| r.2).collect(),
            grids: (
                ceiling_grids(MAX_POOLS, if stride { sms } else { 0 }, stride).1,
                scores_decode_grid_x(MAX_POOLS, sms),
            ),
        })
    }

    /// 2026-10-08: (`seq_len`, `q_pos`) of row case `r` at live context `s`.
    fn row_at(s: usize, r: usize) -> (usize, i32) {
        if r < ROWS {
            (s + r, (s + r) as i32 - 1)
        } else {
            (s, (s / 2) as i32)
        }
    }

    /// 2026-10-08: Live context `s`: each row case's `seq_len` and `q_pos`, then the production
    /// `dsa_write_geom` fills its `geom` (1 thread; `seq_len`, `geom`, `index_kpool`,
    /// `index_topk`, `topk_tile()`), as `layer/decode_k.rs` launches it.
    fn set_context(&self, g: &dyn GpuBackend, ks: &Kernels, s: usize) -> Result<()> {
        for r in 0..ROW_CASES {
            let (sl, qp) = Self::row_at(s, r);
            g.copy_h2d(&i32_bytes(&[sl as i32]), self.seq_len[r])?;
            g.copy_h2d(&i32_bytes(&[qp]), self.q_pos[r])?;
            KernelLaunch::new(g, ks.write_geom)
                .grid([1, 1, 1])
                .block([1, 1, 1])
                .arg_ptr(self.seq_len[r])
                .arg_ptr(self.geom[r])
                .arg_u32(KP as u32)
                .arg_u32(TOPK as u32)
                .arg_u32(topk_tile() as u32)
                .launch(0)?;
        }
        g.synchronize(0)
    }

    /// 2026-10-08: One scores launch of layer `l`, row case `r`, as `select_tokens` issues it
    /// on a ceiling launch (`select/launch.rs`): scalars at the ceiling, `geom` from the device.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        g: &dyn GpuBackend,
        ks: &Kernels,
        decode: bool,
        q: DevicePtr,
        l: usize,
        r: usize,
        out: DevicePtr,
        vc: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (handle, grid_x, block, smem) = if decode {
            (
                ks.decode,
                self.grids.1,
                SCORES_DECODE_BLOCK,
                scores_decode_smem(H) as u32,
            )
        } else {
            (
                ks.plain,
                self.grids.0,
                PLAIN_BLOCK,
                PLAIN_BLOCK.max((H * 4) as u32),
            )
        };
        KernelLaunch::new(g, handle)
            .grid([grid_x as u32, 1, 1])
            .block([block, 1, 1])
            .shared_mem(smem)
            .arg_ptr(q.offset(r * H * D * 4))
            .arg_ptr(self.keys[l])
            .arg_ptr(self.weights.offset(r * H * 4))
            .arg_ptr(self.pool_indices)
            .arg_ptr(self.pool_valid)
            .arg_ptr(self.valid_keys)
            .arg_ptr(self.q_pos[r])
            .arg_ptr(out)
            .arg_ptr(vc)
            .arg_u32(1)
            .arg_u32(MAX_POOLS as u32)
            .arg_u32(H as u32)
            .arg_u32(D as u32)
            .arg_u32(KP as u32)
            .arg_u32(MAX_SEQ as u32)
            .arg_f32((D as f32).powf(-0.5))
            .arg_ptr(self.geom[r])
            .launch(stream)
    }

    /// 2026-10-08: (out bytes, valid_cand bytes) over the whole ceiling of one arm on layer 0,
    /// each buffer pre-filled with `poison`.
    fn arm(
        &self,
        g: &dyn GpuBackend,
        ks: &Kernels,
        decode: bool,
        q: DevicePtr,
        r: usize,
        poison: u8,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let (out, vc) = (g.alloc(MAX_POOLS * 4)?, g.alloc(MAX_POOLS)?);
        g.memset(out, poison, MAX_POOLS * 4)?;
        g.memset(vc, poison, MAX_POOLS)?;
        self.launch(g, ks, decode, q, 0, r, out, vc, 0)?;
        let res = (down(g, out, MAX_POOLS * 4)?, down(g, vc, MAX_POOLS)?);
        g.free(out).ok();
        g.free(vc).ok();
        Ok(res)
    }
}

#[derive(Default)]
struct Tally {
    compared: usize,
    failed: usize,
    cands: usize,
    positive: usize,
}

/// 2026-10-08: Every S x row case: live pools byte-compared, the rest left at each arm's poison.
fn parity(g: &dyn GpuBackend, ks: &Kernels, f: &Fixture, t: &mut Tally) -> Result<()> {
    for &s in PARITY_S {
        f.set_context(g, ks, s)?;
        for r in 0..ROW_CASES {
            let p = Fixture::row_at(s, r).0 / KP;
            let (ro, rv) = f.arm(g, ks, false, f.q, r, POISON_REF)?;
            let (no, nv) = f.arm(g, ks, true, f.q, r, POISON_NEW)?;
            let same = ro[..p * 4] == no[..p * 4] && rv[..p] == nv[..p];
            let tails = ro[p * 4..].iter().all(|&b| b == POISON_REF)
                && rv[p..].iter().all(|&b| b == POISON_REF)
                && no[p * 4..].iter().all(|&b| b == POISON_NEW)
                && nv[p..].iter().all(|&b| b == POISON_NEW);
            let diff = ro[..p * 4]
                .chunks_exact(4)
                .zip(no[..p * 4].chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count();
            t.compared += p;
            t.cands += rv[..p].iter().filter(|&&v| v == 1).count();
            t.positive += ro[..p * 4]
                .chunks_exact(4)
                .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .filter(|&x| x > 0.0)
                .count();
            let ok = same && tails;
            if !ok {
                t.failed += 1;
            }
            let lay = if r < ROWS { "verify" } else { "half-qpos" };
            println!(
                "parity S={s} row={r} ({lay}) P={p} byte-identical={same} tails_untouched={tails} \
                 diff_scores={diff}"
            );
        }
    }
    Ok(())
}

/// 2026-10-08: Negative control: one mantissa bit of q (row case 0, head 0) flipped in the
/// decode arm's input must change its scores against the unflipped plain arm.
fn control(g: &dyn GpuBackend, ks: &Kernels, f: &Fixture) -> Result<bool> {
    let s = 129 * KP + 1;
    f.set_context(g, ks, s)?;
    let p = s / KP;
    let (ro, _) = f.arm(g, ks, false, f.q, 0, POISON_REF)?;
    let (no, _) = f.arm(g, ks, true, f.q_flip, 0, POISON_REF)?;
    let fired = ro[..p * 4] != no[..p * 4];
    println!("CONTROL 1-bit flip in q detected={fired}");
    Ok(fired)
}

/// 2026-10-08: Median ms of one replay of a graph of the 33 launches of a decode step
/// (call i: layer i / 3, verify row i % 3), CUDA events around each replay.
fn time_step(
    g: &dyn GpuBackend,
    ks: &Kernels,
    f: &Fixture,
    decode: bool,
    st: u64,
    out: DevicePtr,
    vc: DevicePtr,
) -> Result<f64> {
    g.begin_capture(st)?;
    let body = (0..LAYERS * ROWS)
        .try_for_each(|i| f.launch(g, ks, decode, f.q, i / ROWS, i % ROWS, out, vc, st));
    if let Err(e) = body {
        g.abort_capture_if_active(st);
        return Err(e);
    }
    let graph = g.end_capture(st)?;
    for _ in 0..3 {
        g.launch_graph(graph, st)?;
    }
    g.synchronize(st)?;
    let mut ev = vec![(0u64, 0u64); REPLAYS];
    // SAFETY: plain CUDA driver event calls; the events are created, used and destroyed here,
    // and the backend made the context current when it was built.
    unsafe {
        for e in ev.iter_mut() {
            check(cuEventCreate(&mut e.0, 0), "cuEventCreate")?;
            check(cuEventCreate(&mut e.1, 0), "cuEventCreate")?;
        }
    }
    for e in &ev {
        // SAFETY: as above.
        unsafe { check(cuEventRecord(e.0, st), "cuEventRecord(start)")? };
        g.launch_graph(graph, st)?;
        // SAFETY: as above.
        unsafe { check(cuEventRecord(e.1, st), "cuEventRecord(end)")? };
    }
    g.synchronize(st)?;
    let mut ms = Vec::with_capacity(REPLAYS);
    for e in &ev {
        let mut t: f32 = 0.0;
        // SAFETY: as above; `t` outlives the call and both events have completed.
        unsafe {
            check(cuEventElapsedTime(&mut t, e.0, e.1), "cuEventElapsedTime")?;
            cuEventDestroy_v2(e.0);
            cuEventDestroy_v2(e.1);
        }
        ms.push(f64::from(t));
    }
    g.destroy_graph(graph)?;
    ms.sort_by(|x, y| x.total_cmp(y));
    Ok(ms[ms.len() / 2])
}

fn timing(g: &dyn GpuBackend, ks: &Kernels, f: &Fixture) -> Result<()> {
    let st = g.create_stream()?;
    let (out, vc) = (g.alloc(MAX_POOLS * 4)?, g.alloc(MAX_POOLS)?);
    for &s in TIMING_S {
        f.set_context(g, ks, s)?;
        let plain = time_step(g, ks, f, false, st, out, vc)?;
        let dec = time_step(g, ks, f, true, st, out, vc)?;
        println!(
            "TIMING scores_plain S={s} ms_per_step={plain:.3} grid={} ({LAYERS} layers x {ROWS} \
             rows)",
            f.grids.0
        );
        println!(
            "TIMING scores_decode S={s} ms_per_step={dec:.3} grid={} speedup={:.2}x",
            f.grids.1,
            plain / dec
        );
    }
    g.free(out).ok();
    g.free(vc).ok();
    Ok(())
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut handles = Vec::new();
    for func in [
        "dsa_index_scores",
        "dsa_index_scores_decode",
        "dsa_write_geom",
    ] {
        match g.kernel(DSA_MODULE, func) {
            Ok(h) => handles.push(h),
            Err(e) => {
                println!("{DSA_MODULE}::{func} absent from this target ({e}) - SKIP");
                std::process::exit(2);
            }
        }
    }
    let ks = Kernels {
        plain: handles[0],
        decode: handles[1],
        write_geom: handles[2],
    };
    let marker = g.kernel(DSA_MODULE, GRID_STRIDE_MARKER).is_ok();
    let f = Fixture::new(g, marker)?;
    println!(
        "ceiling {MAX_POOLS} pools ({MAX_SEQ} tokens); grids plain {} (grid stride {}) decode {} \
         blocks x {SCORES_DECODE_BLOCK} threads, {} B shared",
        f.grids.0,
        dsa_grid_stride() && marker,
        f.grids.1,
        scores_decode_smem(H)
    );

    let mut t = Tally::default();
    parity(g, &ks, &f, &mut t)?;
    let control_ok = control(g, &ks, &f)?;
    timing(g, &ks, &f)?;
    println!(
        "{} live pools compared, {} candidates, {} positive scores, {} failing legs",
        t.compared, t.cands, t.positive, t.failed
    );
    if t.compared == 0 || t.cands == 0 || t.positive == 0 {
        println!(
            "FAIL - nothing, or no candidate / positive score, was compared; this run proves nothing."
        );
        std::process::exit(1);
    }
    if !control_ok {
        println!("FAIL - the negative control did not fire; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if t.failed > 0 {
        println!(
            "FAIL - {} leg(s) differ. Keep METRALE_GLM_DSA_SCORES_DECODE off on this build.",
            t.failed
        );
        std::process::exit(1);
    }
    println!(
        "PASS - {} live pools byte-identical: dsa_index_scores_decode matches dsa_index_scores \
         (out bits and valid_cand) under the ceiling launch at {} contexts x {ROW_CASES} rows.",
        t.compared,
        PARITY_S.len()
    );
    Ok(())
}
