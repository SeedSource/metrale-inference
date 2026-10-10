// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: GATE for `METRALE_GLM_DSA_EXPAND_TAIL_COUNT=1` (`select::tail_count`): the
//! always-tail branch of `dsa_expand_selection` against `dsa_valid_chunk_counts` +
//! `dsa_expand_selection_cc`, at the prefill shape of `dsa_topk_chain_prefill_microtest`
//! (Q = 128 rows a rank, P = S / 4 pools, select_k 512, width 2051, always_tail = 1, host
//! geometry, `launch.rs` argument list).
//!
//! BITWISE. For S in {16,384; 65,536; 131,072; 262,144} (and a ragged S = 16,900 that is not a
//! chunk multiple), the full `[Q, width]` output of the lever-off and lever-on launches is
//! compared as raw i32 bits over six variants: (0) the rig (q_pos = S - 128 + r, every 23rd key
//! invalid, no masked row); then edge rows (q_pos -5, -1, 0, 1, 3, 4, 63, 64, 1,023, 1,024,
//! 1,025, 2,047, 2,048, S / 2, S - 1, S, S + 5, 2 S, masked rows among them, the rest random)
//! over (1) every 23rd key invalid, (2) long runs of invalid keys (whole chunks, key 0
//! invalid), (3) about half the keys invalid at random with `first_key` 64 (so q_pos 0 to 63
//! is below `first_key`), (4) no key valid, (5) every key valid. A tail that never wrote a
//! valid index would make the comparison vacuous, so the line reports the valid tail slots the
//! lever-off run wrote.
//!
//! TIME. Per S, the rig variant: lever off (`dsa_expand_selection`) against on
//! (`dsa_valid_chunk_counts` + `dsa_expand_selection_cc`), each as a CUDA graph of 11
//! launches (one per layer, distinct `selected`, `valid_cand` and `out` buffers), CUDA events
//! around ONE replay, median of 9 replays after 1 warm-up, a 64 MiB memset L2 flush before
//! every replay (outside the events), per-layer microseconds.
//!
//! Output lines: `EXPCNT BITWISE S=<S> identical=<true|false> ...`, `EXPCNT TIME S=<S>
//! off_us=<x> on_us=<y> ...`; the last line is `PASS - EXPCNT` (exit 0) when every BITWISE line
//! is identical, else `FAIL - EXPCNT` (exit 1); exit 2 when the target lacks the entry points.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types. Needs about 0.2 GB of device memory at S = 262,144.
//!
//! Run (gb10 common kernels):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_expand_tail_count_microtest

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::Glm5NextDsaKernels;
use metrale_model_arch::glm5next_dsa::select::tail_count::{CHUNK_COUNTS_BLOCK, chunk_count};

// 2026-10-10: CUDA driver event API, as in `dsa_topk_chain_prefill_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const Q: usize = 128;
const KP: usize = 4;
const SELECT_K: usize = 512;
const WIDTH: usize = 2051;
const LAYERS: usize = 11;
const CONTEXTS: &[usize] = &[16_384, 65_536, 131_072, 262_144];
/// 2026-10-10: Not a multiple of the 1,024-key chunk (bitwise only).
const RAGGED: usize = 16_900;
const REPS: usize = 9;
const FLUSH_BYTES: usize = 64 << 20;
const POISON: u8 = 0xA5;

/// 2026-10-10: splitmix64, for the deterministic inputs.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn le_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(4))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// 2026-10-10: Per-layer buffers.
struct Layer {
    selected: DevicePtr,
    valid_cand: DevicePtr,
    out: DevicePtr,
}

struct Rig<'a> {
    g: &'a dyn GpuBackend,
    k: &'a Glm5NextDsaKernels,
    st: u64,
    s: usize,
    p: usize,
    layers: Vec<Layer>,
    pool_indices: DevicePtr,
    valid_keys: DevicePtr,
    counts: DevicePtr,
    q_pos: DevicePtr,
    q_mask: DevicePtr,
    flush: DevicePtr,
}

impl<'a> Rig<'a> {
    fn new(g: &'a dyn GpuBackend, k: &'a Glm5NextDsaKernels, st: u64, s: usize) -> Result<Self> {
        let p = s / KP;
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let mut rng = Rng(((s as u64) << 16) | l as u64);
            let mut selected = Vec::with_capacity(Q * SELECT_K);
            let mut valid_cand = Vec::with_capacity(Q * p);
            for _ in 0..Q {
                for j in 0..SELECT_K {
                    selected.push(if j % 37 == 36 {
                        -1
                    } else {
                        rng.below(p) as i32
                    });
                }
                for _ in 0..p {
                    valid_cand.push(u8::from(rng.below(6) != 0));
                }
            }
            layers.push(Layer {
                selected: up(g, &le_i32(&selected))?,
                valid_cand: up(g, &valid_cand)?,
                out: g.alloc(Q * WIDTH * 4)?,
            });
        }
        let pool_idx: Vec<i32> = (0..(p * KP) as i32).collect();
        Ok(Rig {
            g,
            k,
            st,
            s,
            p,
            layers,
            pool_indices: up(g, &le_i32(&pool_idx))?,
            valid_keys: g.alloc(s)?,
            counts: g.alloc(chunk_count(s) * 4)?,
            q_pos: g.alloc(Q * 4)?,
            q_mask: g.alloc(Q)?,
            flush: g.alloc(FLUSH_BYTES)?,
        })
    }

    fn free(self) -> Result<()> {
        for l in &self.layers {
            for b in [l.selected, l.valid_cand, l.out] {
                self.g.free(b)?;
            }
        }
        for b in [
            self.pool_indices,
            self.valid_keys,
            self.counts,
            self.q_pos,
            self.q_mask,
            self.flush,
        ] {
            self.g.free(b)?;
        }
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        self.g.synchronize(0)?;
        self.g.synchronize(self.st)
    }

    /// 2026-10-10: Upload one variant's `valid_keys`, `q_pos` and `q_mask`.
    fn set_inputs(&self, valid: &[u8], q_pos: &[i32], mask: &[u8]) -> Result<()> {
        self.g.copy_h2d(valid, self.valid_keys)?;
        self.g.copy_h2d(&le_i32(q_pos), self.q_pos)?;
        self.g.copy_h2d(mask, self.q_mask)?;
        self.sync()
    }

    /// 2026-10-10: `dsa_expand_selection` (`cc` false) or the chunk counts + `_cc` entry point
    /// (`cc` true), with the argument list of `select/launch.rs`.
    fn expand(&self, l: usize, cc: bool, first_key: i32, stream: u64) -> Result<()> {
        let b = &self.layers[l];
        let (g, k) = (self.g, self.k);
        if cc {
            KernelLaunch::new(g, k.valid_chunk_counts)
                .grid([chunk_count(self.s) as u32, 1, 1])
                .block([CHUNK_COUNTS_BLOCK, 1, 1])
                .arg_ptr(self.valid_keys)
                .arg_ptr(self.counts)
                .arg_u32(self.s as u32)
                .launch(stream)?;
        }
        let launch = KernelLaunch::new(
            g,
            if cc {
                k.expand_selection_cc
            } else {
                k.expand_selection
            },
        )
        .grid([Q as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b.selected)
        .arg_ptr(self.pool_indices)
        .arg_ptr(b.valid_cand)
        .arg_ptr(self.valid_keys)
        .arg_ptr(self.q_pos)
        .arg_ptr(self.q_mask)
        .arg_ptr(b.out)
        .arg_u32(Q as u32)
        .arg_u32(self.p as u32)
        .arg_u32(KP as u32)
        .arg_u32(self.s as u32)
        .arg_u32(SELECT_K as u32)
        .arg_u32(WIDTH as u32)
        .arg_i32(first_key)
        .arg_i32(1)
        .arg_ptr(DevicePtr::NULL);
        let launch = if cc {
            launch.arg_ptr(self.counts)
        } else {
            launch
        };
        launch.launch(stream)
    }

    /// 2026-10-10: Layer 0's output of one launch (poisoned first), downloaded.
    fn run_once(&self, cc: bool, first_key: i32) -> Result<Vec<u8>> {
        self.g.memset(self.layers[0].out, POISON, Q * WIDTH * 4)?;
        self.expand(0, cc, first_key, self.st)?;
        self.sync()?;
        let mut v = vec![0u8; Q * WIDTH * 4];
        self.g.copy_d2h(self.layers[0].out, &mut v)?;
        Ok(v)
    }

    /// 2026-10-10: Median per-layer us of the 11-launch graph, L2 flushed before each replay.
    fn time_graph(&self, cc: bool) -> Result<f64> {
        let (g, st) = (self.g, self.st);
        g.begin_capture(st)?;
        for l in 0..LAYERS {
            if let Err(e) = self.expand(l, cc, 0, st) {
                g.abort_capture_if_active(st);
                return Err(e);
            }
        }
        let graph = g.end_capture(st)?;
        g.launch_graph(graph, st)?; // warm-up
        let (mut e0, mut e1) = (0u64, 0u64);
        // SAFETY: plain CUDA driver event calls; created, used and destroyed in this function
        // and the backend made the context current when it was built.
        unsafe {
            check(cuEventCreate(&mut e0, 0), "cuEventCreate")?;
            check(cuEventCreate(&mut e1, 0), "cuEventCreate")?;
        }
        let mut ms = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            self.sync()?;
            g.memset(self.flush, 0, FLUSH_BYTES)?;
            self.sync()?;
            // SAFETY: as above; both events have completed before the elapsed read.
            unsafe { check(cuEventRecord(e0, st), "cuEventRecord")? };
            g.launch_graph(graph, st)?;
            unsafe { check(cuEventRecord(e1, st), "cuEventRecord")? };
            g.synchronize(st)?;
            let mut t: f32 = 0.0;
            unsafe { check(cuEventElapsedTime(&mut t, e0, e1), "cuEventElapsedTime")? };
            ms.push(f64::from(t));
        }
        // SAFETY: as above.
        unsafe {
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        g.destroy_graph(graph)?;
        Ok(median(ms) * 1000.0 / LAYERS as f64)
    }
}

/// 2026-10-10: Edge `q_pos` rows (masked flags alongside) for context `s`, the rest random.
fn edge_rows(s: usize, seed: u64) -> (Vec<i32>, Vec<u8>) {
    let si = s as i32;
    let fixed: [(i32, u8); 22] = [
        (-5, 1),
        (-1, 1),
        (0, 1),
        (1, 1),
        (3, 1),
        (4, 1),
        (63, 1),
        (64, 1),
        (1023, 1),
        (1024, 1),
        (1025, 1),
        (2047, 1),
        (2048, 1),
        (si / 2, 1),
        (si - 1, 1),
        (si, 1),
        (si + 5, 1),
        (2 * si, 1),
        (0, 0),
        (si - 1, 0),
        (-1, 0),
        (1500, 0),
    ];
    let mut rng = Rng(seed);
    let mut q_pos: Vec<i32> = fixed.iter().map(|f| f.0).collect();
    let mut mask: Vec<u8> = fixed.iter().map(|f| f.1).collect();
    while q_pos.len() < Q {
        q_pos.push(rng.below(s) as i32);
        mask.push(u8::from(rng.below(9) != 4));
    }
    (q_pos, mask)
}

/// 2026-10-10: A variant: name, `valid_keys`, `q_pos`, `q_mask`, `first_key`.
type Variant = (&'static str, Vec<u8>, Vec<i32>, Vec<u8>, i32);

fn variants(s: usize) -> Vec<Variant> {
    let every23: Vec<u8> = (0..s).map(|t| u8::from(t % 23 != 22)).collect();
    let rig_q: Vec<i32> = (0..Q).map(|r| (s - Q + r) as i32).collect();
    let (eq, em) = edge_rows(s, s as u64);
    let long_runs: Vec<u8> = (0..s)
        .map(|t| {
            let hole = t == 0
                || (1024..3072).contains(&t)
                || (s / 2..s / 2 + 1500).contains(&t)
                || t >= s.saturating_sub(700);
            u8::from(!hole)
        })
        .collect();
    let mut rng = Rng(0xC0FFEE ^ s as u64);
    let half: Vec<u8> = (0..s)
        .map(|_| (rng.below(2)) as u8 * (1 + rng.below(3) as u8))
        .collect();
    vec![
        ("rig", every23.clone(), rig_q, vec![1u8; Q], 0),
        ("edge_every23", every23, eq.clone(), em.clone(), 0),
        ("edge_long_runs", long_runs, eq.clone(), em.clone(), 0),
        ("edge_half_fk64", half, eq.clone(), em.clone(), 64),
        ("edge_none_valid", vec![0u8; s], eq.clone(), em.clone(), 0),
        ("edge_all_valid", vec![1u8; s], eq, em, 0),
    ]
}

/// 2026-10-10: Valid (non -1) tail-or-selection slots in a lever-off output, a vacuity check.
fn valid_slots(out: &[u8]) -> usize {
    out.chunks_exact(4)
        .filter(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]) >= 0)
        .count()
}

/// 2026-10-10: BITWISE line of context `s`: all variants, off against on. Returns identical.
fn bitwise(rig: &Rig<'_>) -> Result<bool> {
    let mut all = true;
    let mut names = Vec::new();
    let mut valid_off = 0usize;
    let mut tail_slots_rig = 0usize;
    for (i, (name, valid, q_pos, mask, fk)) in variants(rig.s).iter().enumerate() {
        rig.set_inputs(valid, q_pos, mask)?;
        let off = rig.run_once(false, *fk)?;
        let on = rig.run_once(true, *fk)?;
        let same = off == on;
        all &= same;
        if i == 0 {
            // 2026-10-10: Slots at and after the selection prefix hold only the tail here; the
            // rig rows have row_select_k * KP = 2048 selected slots, so the tail is the last 3.
            tail_slots_rig = off
                .chunks_exact(WIDTH * 4)
                .map(|row| valid_slots(&row[2048 * 4..]))
                .sum();
        }
        valid_off += valid_slots(&off);
        names.push(format!("{name}={same}"));
    }
    println!(
        "EXPCNT BITWISE S={} identical={all} variants=[{}] rig_valid_tail_slots={tail_slots_rig} \
         valid_slots_off_total={valid_off}",
        rig.s,
        names.join(",")
    );
    Ok(all)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k = Glm5NextDsaKernels::resolve(g)?;
    if k.expand_selection_cc.0 == 0 || k.valid_chunk_counts.0 == 0 {
        println!(
            "FAIL - EXPCNT: dsa_expand_selection_cc / dsa_valid_chunk_counts are absent from \
             this target's dsa_indexer module"
        );
        std::process::exit(2);
    }
    let st = g.create_stream()?;
    println!(
        "Q={Q} rows, select_k {SELECT_K}, width {WIDTH}, {LAYERS} layers, chunk {} keys; \
         median of {REPS} replays, {} MiB L2 flush before each, per-layer us",
        metrale_model_arch::glm5next_dsa::select::tail_count::EXPAND_TAIL_COUNT_CHUNK,
        FLUSH_BYTES >> 20
    );
    let mut ok = true;
    for &s in CONTEXTS.iter().chain(std::iter::once(&RAGGED)) {
        let rig = Rig::new(g, &k, st, s)?;
        ok &= bitwise(&rig)?;
        if s != RAGGED {
            let (_, valid, q_pos, mask, _) = variants(s).swap_remove(0);
            rig.set_inputs(&valid, &q_pos, &mask)?;
            let off = rig.time_graph(false)?;
            let on = rig.time_graph(true)?;
            println!(
                "EXPCNT TIME S={s} off_us={off:.2} on_us={on:.2} speedup={:.2}x",
                off / on.max(1e-9)
            );
        }
        rig.free()?;
    }
    if ok {
        println!("PASS - EXPCNT");
        Ok(())
    } else {
        println!("FAIL - EXPCNT: a BITWISE line differs");
        std::process::exit(1);
    }
}
