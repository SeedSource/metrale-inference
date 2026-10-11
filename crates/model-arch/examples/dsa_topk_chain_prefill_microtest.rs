// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: GATE 0 (measurement only, no production behaviour change) for the GLM-5.3 prefill
//! DSA selection chain: per layer and per 256-row sub-chunk (Q = 128 rows a rank) the radix
//! top-k (`select::radix::launch_topk_radix`: init, hist + find for passes 0 to 2, gather, sort)
//! followed by `dsa_expand_selection` (always_tail = 1), timed on rig-shaped inputs.
//!
//! Inputs: Q = 128 rows, S in {16,384; 65,536; 131,072; 262,144}, P = S / 4 pools,
//! select_k 512, `rig_scores` rows (about 4 in 23 pools -FLT_MAX, the rest sums of 8
//! `w * relu(x)`), `valid_cand` 1 exactly where the score is not -FLT_MAX, q_pos = S - 128 + r,
//! valid_keys 1 except every 23rd token, pool_indices 4p..4p+3, q_mask all 1, geometry from the
//! host (no `geom`). 11 distinct buffer sets (one per layer), so no layer re-reads another's
//! scores.
//!
//! What is launched. The FULL CHAIN numbers (`graph_us`, `chain_us`, and the chunk-sweep
//! outputs) go through `launch_topk_radix`, the function `select_tokens` calls, followed by an
//! expand launch with the argument list of `select/launch.rs` (that launch is inline in
//! `select_tokens`, there is no helper to call, so it is repeated here). The PER-KERNEL numbers
//! need one kernel at a time, so they use a launch of each stage with the same arguments as
//! `launch_topk_radix` (chunk count from `radix_chunks`, so the test-only override applies);
//! the staged chain's `selected` and expand rows are compared bitwise with the production
//! launcher's (`staged_matches_production`) and the verdict fails if they differ.
//!
//! How the time is taken. Every measurement is a CUDA graph of 11 launches (one per layer),
//! CUDA events around ONE replay, median of 9 replays after 1 warm-up, reported per layer in
//! microseconds (so it includes the graph node overhead of a back-to-back sequence, a few us a
//! node). Before every measured replay: (1) the work buffers are restored from a snapshot taken
//! just before that stage (so a stage sees exactly the state the chain gives it, e.g. gather
//! sees the pass-2 tie bases), and (2) a 64 MiB memset flushes L2 (the scores kernel writes
//! 8 to 32 MB a layer, so the selection chain starts from DRAM or a partly warm L2, not an
//! all-warm one). The restore and flush are outside the events. Consequence: the per-kernel
//! numbers are COLD-L2 per kernel, so hist passes 1, 2 and gather (which re-read the scores
//! that pass 0 just touched) are overstated against the in-chain case; `graph_us` (whole chain
//! in one graph, flushed once before the replay) is the in-chain figure and `total_us` the sum
//! of the cold per-kernel figures. Compare the two before attributing a share.
//!
//! Output lines (per S): `CHAIN KERNELS`, `CHAIN CHUNKS` (c in 1, 2, 4, 8, 16 via
//! `set_radix_chunks_override`), `CHAIN EXPAND` (always_tail 1 vs 0), `CHAIN BIN` (pass-0 bin
//! that holds the select_k-th key: its element count, read from hist0 after find pass 0).
//!
//! 2026-10-10: `CHAIN COMPACT S=<S> compact_us=<x> gather_us=<y> ratio=<x/y> b0_median=<n>`: the
//! prototype `dsa_topk_radix_compact` (one read of the row after find pass 0: keys above the
//! pass-0 threshold bin to the candidate slots, bin-b0 pairs to a per-row buffer capped at 2,048)
//! timed in the same way as the stages above (state after init, hist 0, find 0; cold L2) next to
//! the production gather (state after find 2). `b0_median` is the median row count of bin b0
//! as counted by the compact kernel. A `CHAIN COMPACT CHECK` line compares those counts with the
//! `dsa_topk_radix_binstat` kernel's. Timing only: compact's output is not a selection.
//!
//! 2026-10-10: `dsa_topk_radix_compact` is now the production kernel of
//! `METRALE_GLM_DSA_TOPK_RADIX_COMPACT=1` (`select::radix_compact`): it only touches rows whose
//! bin b0 holds at most 2,048 keys (COMPACT), so the CHECK line expects its count to equal
//! binstat's on those rows and 0 on the others, and `b0_median` is binstat's median.
//!
//! 2026-10-10: RCOMPACT (the lever's gate). `launch_topk_radix` with the pair buffer set (lever
//! on) against the same call without it (lever off), on the rig and on edge cases:
//! - `RCOMPACT BITWISE S=<S> case=<name> identical=<bool>`: `selected` (raw i32 bits) of the
//!   lever-on call equals the lever-off call's (rig: and the expand rows too); edge cases also
//!   equal a host sort (`host_reference`, the `dsa_topk_pools` order; select_k 0: untouched
//!   poison). Edge cases: heavy -FLT_MAX ties with the -FLT_MAX bin at or below and above the
//!   cap, all-equal scores below and above the cap, select_k equal to the live pools (and
//!   below 512), select_k 0, need0 equal to the bin's count, COMPACT and FALLBACK rows mixed in
//!   one call, ragged P, and device-geometry (`geom`) launches with a larger scalar ceiling,
//!   each at the production chunk count and at 5 chunks (`/c5`). S is 4 x P.
//! - `RCOMPACT ROWS S=<S> case=<name> compact=<n> fallback=<m>`: rows select0 finished (mode
//!   word 1) and rows left to the passes, so a case's coverage is visible.
//! - `RCOMPACT TIME S=<S> off_us=<x> on_us=<y> ratio=<y/x>`: the full production chain (radix +
//!   expand) of the rig, 11 layers in one graph, flushed, median of 9, per layer, lever off and
//!   on (as `graph_us`).
//! - `PASS - RCOMPACT` only when every BITWISE line is identical.
//!
//! Exit: 0 and `PASS - CHAIN gate0 ...` when every chunk output is bitwise identical to c = 1
//! (and the staged chain matches production); 1 with `FAIL`; 2 when the radix entry points are
//! absent from this target. 2026-10-10: also 1 when an RCOMPACT line differs, 2 when the
//! compaction entry points are absent.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types. Needs about 0.6 GB of device memory at S = 262,144.
//!
//! Run (gb10 common kernels):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_topk_chain_prefill_microtest

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::Glm5NextDsaKernels;
use metrale_model_arch::glm5next_dsa::select::radix::{
    RADIX_BINS, RADIX_PASSES, RADIX_STATE, RADIX_THREADS, RadixTopk, launch_topk_radix,
    radix_chunks, radix_resolved, radix_row_bytes, set_radix_chunks_override,
};
use metrale_model_arch::glm5next_dsa::select::binstat::launch_binstat;
use metrale_model_arch::glm5next_dsa::select::radix_compact::{
    COMPACT_CAP, RS_MODE, compact_bytes, radix_compact_resolved,
};

// 2026-10-10: this example uses the score generator and the host order of the shared helpers.
#[allow(dead_code)]
#[path = "common/dsa_topk_radix_common.rs"]
mod dsa_topk_radix_common;
use dsa_topk_radix_common::{SELECT_K, host_reference, rig_scores};

// 2026-10-10: CUDA driver event API, declared as in `dsa_topk_radix_common` (its copy is private).
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-10: Query rows of a rank's 256-row sub-chunk.
const Q: usize = 128;
/// 2026-10-10: GLM-5.3 `index_kpool`, candidate capacity (`radix_kcap`), DSA text layers.
const KP: usize = 4;
const KCAP: usize = 512;
const LAYERS: usize = 11;
/// 2026-10-10: Output row width: index_topk 2048 + index_kpool - 1.
const WIDTH: usize = 2051;
/// 2026-10-10: Contexts (tokens) and chunk counts swept.
const CONTEXTS: &[usize] = &[16_384, 65_536, 131_072, 262_144];
const SWEEP: &[usize] = &[1, 2, 4, 8, 16];
/// 2026-10-10: Measured replays per number (median), L2 flush size, fill byte of outputs.
const REPS: usize = 9;
const FLUSH_BYTES: usize = 64 << 20;
const POISON: u8 = 0xA5;

/// 2026-10-10: One launch of the chain, for the per-kernel timing.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Stage {
    Init,
    Hist(u32),
    Find(u32),
    Gather,
    Sort,
    /// 2026-10-10: `dsa_expand_selection` with `always_tail` set to the flag.
    Expand(bool),
    /// 2026-10-10: `dsa_topk_radix_compact` prototype (not part of [`STAGES`]).
    Compact,
}

/// 2026-10-10: The chain in launch order.
const STAGES: [Stage; 10] = [
    Stage::Init,
    Stage::Hist(0),
    Stage::Find(0),
    Stage::Hist(1),
    Stage::Find(1),
    Stage::Hist(2),
    Stage::Find(2),
    Stage::Gather,
    Stage::Sort,
    Stage::Expand(true),
];

/// 2026-10-10: Per-layer device buffers.
struct Layer {
    scores: DevicePtr,
    valid_cand: DevicePtr,
    selected: DevicePtr,
    out: DevicePtr,
    work: DevicePtr,
}

/// 2026-10-10: Everything one context `s` needs.
struct Rig<'a> {
    g: &'a dyn GpuBackend,
    k: &'a Glm5NextDsaKernels,
    st: u64,
    s: usize,
    p: usize,
    layers: Vec<Layer>,
    work_all: DevicePtr,
    work_bytes: usize,
    snap: DevicePtr,
    flush: DevicePtr,
    pool_indices: DevicePtr,
    valid_keys: DevicePtr,
    q_pos: DevicePtr,
    q_mask: DevicePtr,
    /// 2026-10-10: Per-row bin-b0 buffers of the compact prototype (shared by all layers).
    bins: DevicePtr,
}

/// 2026-10-10: Words of one row's compact buffer (`DSA_COMPACT_CAP` keys, then indices).
const BIN_ROW_WORDS: usize = 2 * 2048;

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

impl<'a> Rig<'a> {
    /// 2026-10-10: Generate and upload the 11 layers' inputs for context `s`.
    fn new(g: &'a dyn GpuBackend, k: &'a Glm5NextDsaKernels, st: u64, s: usize) -> Result<Self> {
        let p = s / KP;
        let work_bytes = Q * radix_row_bytes(KCAP);
        let work_all = g.alloc(LAYERS * work_bytes)?;
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let mut flat: Vec<f32> = Vec::with_capacity(Q * p);
            for r in 0..Q {
                flat.extend(rig_scores(
                    ((s as u64) << 16) | ((l as u64) << 8) | r as u64,
                    p,
                ));
            }
            let valid: Vec<u8> = flat.iter().map(|&v| u8::from(v != -f32::MAX)).collect();
            let fb: Vec<u8> = flat.iter().flat_map(|x| x.to_le_bytes()).collect();
            layers.push(Layer {
                scores: up(g, &fb)?,
                valid_cand: up(g, &valid)?,
                selected: g.alloc(Q * SELECT_K * 4)?,
                out: g.alloc(Q * WIDTH * 4)?,
                work: work_all.offset(l * work_bytes),
            });
        }
        let pool_idx: Vec<i32> = (0..(p * KP) as i32).collect();
        let valid_keys: Vec<u8> = (0..s).map(|t| u8::from(t % 23 != 22)).collect();
        let q_pos: Vec<i32> = (0..Q).map(|r| (s - Q + r) as i32).collect();
        let rig = Rig {
            g,
            k,
            st,
            s,
            p,
            layers,
            work_all,
            work_bytes,
            snap: g.alloc(LAYERS * work_bytes)?,
            flush: g.alloc(FLUSH_BYTES)?,
            pool_indices: up(g, &le_i32(&pool_idx))?,
            valid_keys: up(g, &valid_keys)?,
            q_pos: up(g, &le_i32(&q_pos))?,
            q_mask: up(g, &[1u8; Q])?,
            bins: g.alloc(Q * BIN_ROW_WORDS * 4)?,
        };
        g.memset(rig.work_all, POISON, LAYERS * work_bytes)?;
        Ok(rig)
    }

    fn free(self) -> Result<()> {
        for l in &self.layers {
            for b in [l.scores, l.valid_cand, l.selected, l.out] {
                self.g.free(b)?;
            }
        }
        for b in [
            self.work_all,
            self.snap,
            self.flush,
            self.pool_indices,
            self.valid_keys,
            self.q_pos,
            self.q_mask,
            self.bins,
        ] {
            self.g.free(b)?;
        }
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        self.g.synchronize(0)?;
        self.g.synchronize(self.st)
    }

    /// 2026-10-10: `dsa_expand_selection` with the argument list of `select/launch.rs`.
    fn expand(&self, l: usize, tail: bool, stream: u64) -> Result<()> {
        let b = &self.layers[l];
        KernelLaunch::new(self.g, self.k.expand_selection)
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
            .arg_i32(0)
            .arg_i32(i32::from(tail))
            .arg_ptr(DevicePtr::NULL)
            .launch(stream)
    }

    /// 2026-10-10: The production chain of layer `l`: `launch_topk_radix` then the expand.
    /// With `compact` (`METRALE_GLM_DSA_TOPK_RADIX_COMPACT` on) the rig's pair buffers (`bins`,
    /// one per row, shared by the layers, which run in stream order) are passed to it.
    fn chain_with(&self, l: usize, stream: u64, compact: bool) -> Result<()> {
        let b = &self.layers[l];
        let t = RadixTopk {
            scores: b.scores,
            selected: b.selected,
            work: b.work,
            q_rows: Q,
            n_pools: self.p,
            select_k: SELECT_K,
            kcap: KCAP,
            geom_dev: DevicePtr::NULL,
            bins: if compact { self.bins } else { DevicePtr::NULL },
        };
        launch_topk_radix(self.g, self.k, &t, stream)?;
        self.expand(l, true, stream)
    }

    /// 2026-10-10: One stage of layer `l`, with the arguments `launch_topk_radix` passes.
    fn stage(&self, stg: Stage, l: usize, stream: u64) -> Result<()> {
        let b = &self.layers[l];
        let (rows, chunks) = (Q as u32, radix_chunks(Q) as u32);
        let block = [RADIX_THREADS, 1, 1];
        let (p, selk, kcap) = (self.p as u32, SELECT_K as u32, KCAP as u32);
        let gd = DevicePtr::NULL;
        let (g, k) = (self.g, self.k);
        match stg {
            Stage::Init => KernelLaunch::new(g, k.topk_radix_init)
                .grid([rows, 1, 1])
                .block(block)
                .arg_ptr(b.work)
                .arg_u32(selk)
                .arg_u32(kcap)
                .arg_ptr(gd)
                .launch(stream),
            Stage::Hist(pass) => KernelLaunch::new(g, k.topk_radix_hist)
                .grid([chunks, rows, 1])
                .block(block)
                .arg_ptr(b.scores)
                .arg_ptr(b.work)
                .arg_u32(p)
                .arg_u32(selk)
                .arg_u32(kcap)
                .arg_u32(pass)
                .arg_ptr(gd)
                .launch(stream),
            Stage::Find(pass) => KernelLaunch::new(g, k.topk_radix_find)
                .grid([rows, 1, 1])
                .block(block)
                .arg_ptr(b.work)
                .arg_u32(selk)
                .arg_u32(kcap)
                .arg_u32(pass)
                .arg_u32(chunks)
                .arg_ptr(gd)
                .launch(stream),
            Stage::Gather => KernelLaunch::new(g, k.topk_radix_gather)
                .grid([chunks, rows, 1])
                .block(block)
                .arg_ptr(b.scores)
                .arg_ptr(b.work)
                .arg_u32(p)
                .arg_u32(selk)
                .arg_u32(kcap)
                .arg_ptr(gd)
                .launch(stream),
            Stage::Sort => KernelLaunch::new(g, k.topk_radix_sort)
                .grid([rows, 1, 1])
                .block(block)
                .arg_ptr(b.work)
                .arg_ptr(b.selected)
                .arg_u32(selk)
                .arg_u32(kcap)
                .arg_ptr(gd)
                .launch(stream),
            Stage::Expand(tail) => self.expand(l, tail, stream),
            Stage::Compact => KernelLaunch::new(g, k.topk_radix_compact)
                .grid([chunks, rows, 1])
                .block(block)
                .arg_ptr(b.scores)
                .arg_ptr(b.work)
                .arg_ptr(self.bins)
                .arg_u32(p)
                .arg_u32(selk)
                .arg_u32(kcap)
                .arg_ptr(gd)
                .launch(stream),
        }
    }

    /// 2026-10-10: Poison both outputs of every layer.
    fn poison_outputs(&self) -> Result<()> {
        for b in &self.layers {
            self.g.memset(b.selected, POISON, Q * SELECT_K * 4)?;
            self.g.memset(b.out, POISON, Q * WIDTH * 4)?;
        }
        Ok(())
    }

    /// 2026-10-10: Production chain on every layer (outputs poisoned first), then both outputs
    /// downloaded.
    fn run_production(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        self.run_production_with(false)
    }

    /// 2026-10-10: [`Self::run_production`] with the compaction on or off.
    fn run_production_with(&self, compact: bool) -> Result<(Vec<u8>, Vec<u8>)> {
        self.poison_outputs()?;
        for l in 0..LAYERS {
            self.chain_with(l, self.st, compact)?;
        }
        self.dump()
    }

    fn dump(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        self.sync()?;
        let (mut sel, mut out) = (Vec::new(), Vec::new());
        for b in &self.layers {
            let mut v = vec![0u8; Q * SELECT_K * 4];
            self.g.copy_d2h(b.selected, &mut v)?;
            sel.extend(v);
            let mut v = vec![0u8; Q * WIDTH * 4];
            self.g.copy_d2h(b.out, &mut v)?;
            out.extend(v);
        }
        Ok((sel, out))
    }

    /// 2026-10-10: Capture `f` on the stream, replay it `REPS` times and return the median
    /// replay in us per layer. Before each replay: restore the work buffers from `snap` and
    /// flush L2 with a 64 MiB memset (both outside the events).
    fn time_graph(&self, restore: bool, f: &mut dyn FnMut(u64) -> Result<()>) -> Result<f64> {
        let (g, st) = (self.g, self.st);
        g.begin_capture(st)?;
        if let Err(e) = f(st) {
            g.abort_capture_if_active(st);
            return Err(e);
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
            if restore {
                g.copy_d2d(self.snap, self.work_all, LAYERS * self.work_bytes)?;
            }
            g.memset(self.flush, 0, FLUSH_BYTES)?;
            self.sync()?;
            // SAFETY: as above; `t` outlives the call and both events have completed.
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

    /// 2026-10-10: Per-layer us of `stg` alone. The work buffers must hold the state just
    /// before `stg` (and `selected` must be valid for Expand); they are left unchanged.
    fn time_stage(&self, stg: Stage) -> Result<f64> {
        self.sync()?;
        self.g
            .copy_d2d(self.work_all, self.snap, LAYERS * self.work_bytes)?;
        let us = self.time_graph(true, &mut |sm| {
            (0..LAYERS).try_for_each(|l| self.stage(stg, l, sm))
        })?;
        self.sync()?;
        self.g
            .copy_d2d(self.snap, self.work_all, LAYERS * self.work_bytes)?;
        Ok(us)
    }

    /// 2026-10-10: Walk the chain stage by stage on all layers (staged launches): time each
    /// stage in the state the chain gives it, then run it for real to advance. Returns the 10
    /// per-layer times in [`STAGES`] order.
    fn staged_times(&self) -> Result<[f64; 10]> {
        let mut t = [0.0; 10];
        self.poison_outputs()?;
        for (i, &stg) in STAGES.iter().enumerate() {
            t[i] = self.time_stage(stg)?;
            for l in 0..LAYERS {
                self.stage(stg, l, self.st)?;
            }
            self.sync()?;
        }
        Ok(t)
    }

    /// 2026-10-10: Whole production chain of 11 layers in one graph, flushed once before each
    /// replay: per-layer us.
    fn chain_us(&self) -> Result<f64> {
        self.chain_us_with(false)
    }

    /// 2026-10-10: [`Self::chain_us`] with the compaction on or off.
    fn chain_us_with(&self, compact: bool) -> Result<f64> {
        self.time_graph(false, &mut |sm| {
            (0..LAYERS).try_for_each(|l| self.chain_with(l, sm, compact))
        })
    }
}

/// 2026-10-10: Pass-0 threshold-bin occupancy of layer 0: after init, hist pass 0 and find
/// pass 0, row r's state prefix holds the winning bin in its top 12 bits and hist0 its count
/// (hist0 is overwritten by pass 2's per-chunk histograms, so it is read here).
fn bin_occupancy(rig: &Rig<'_>) -> Result<(Vec<u32>, usize)> {
    for stg in [Stage::Init, Stage::Hist(0), Stage::Find(0)] {
        rig.stage(stg, 0, rig.st)?;
    }
    rig.sync()?;
    let mut raw = vec![0u8; rig.work_bytes];
    rig.g.copy_d2h(rig.layers[0].work, &mut raw)?;
    let words: Vec<u32> = raw
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let row_words = radix_row_bytes(KCAP) / 4;
    let counts = (0..Q)
        .map(|r| {
            let w = &words[r * row_words..(r + 1) * row_words];
            let bin = (w[RADIX_STATE] >> 20) as usize;
            assert!(bin < RADIX_BINS);
            w[bin]
        })
        .collect();
    Ok((counts, rig.p))
}

/// 2026-10-10: State word offsets (`DSA_RS_*`) the compact and binstat kernels write.
const RS_B0CNT: usize = 4;
const RS_BINCNT: usize = 6;
// 2026-10-10: the binstat slot is checked against the crate's constant.
const _: () = assert!(RS_B0CNT == metrale_model_arch::glm5next_dsa::select::binstat::RS_B0CNT);

/// 2026-10-10: The COMPACT arm for the rig's context: with every layer at the state after init,
/// hist 0 and find 0 (and the binstat kernel run), time the compact prototype per layer, then
/// read the bin-b0 counts it and the binstat kernel produced for layer 0. Returns
/// `(compact_us, median count, binstat_matches_compact)`.
fn compact_arm(rig: &Rig<'_>) -> Result<(f64, u32, bool)> {
    for stg in [Stage::Init, Stage::Hist(0), Stage::Find(0)] {
        for l in 0..LAYERS {
            rig.stage(stg, l, rig.st)?;
        }
    }
    for l in 0..LAYERS {
        launch_binstat(
            rig.g,
            rig.k,
            rig.layers[l].work,
            Q as u32,
            rig.p as u32,
            SELECT_K as u32,
            KCAP as u32,
            DevicePtr::NULL,
            rig.st,
        )?;
    }
    rig.sync()?;
    let us = rig.time_stage(Stage::Compact)?;
    // One real run on the (restored) post-find0 state, then read layer 0's counters.
    for l in 0..LAYERS {
        rig.stage(Stage::Compact, l, rig.st)?;
    }
    rig.sync()?;
    let mut raw = vec![0u8; rig.work_bytes];
    rig.g.copy_d2h(rig.layers[0].work, &mut raw)?;
    let row_words = radix_row_bytes(KCAP) / 4;
    let word = |r: usize, i: usize| {
        let o = 4 * (r * row_words + RADIX_STATE + i);
        u32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]])
    };
    let compact: Vec<u32> = (0..Q).map(|r| word(r, RS_BINCNT)).collect();
    let stat: Vec<u32> = (0..Q).map(|r| word(r, RS_B0CNT)).collect();
    // 2026-10-10: the production kernel counts only COMPACT rows (bin b0 <= cap).
    let want: Vec<u32> = stat
        .iter()
        .map(|&c| if c as usize <= COMPACT_CAP { c } else { 0 })
        .collect();
    let mut sorted = stat.clone();
    sorted.sort_unstable();
    Ok((us, sorted[sorted.len() / 2], compact == want))
}

/// 2026-10-10: Mode words (`DSA_RS_MODE`) of the first `rows` rows of a work buffer: how many
/// rows select0 finished.
fn compact_rows(g: &dyn GpuBackend, work: DevicePtr, rows: usize) -> Result<usize> {
    let row_bytes = radix_row_bytes(KCAP);
    let mut n = 0;
    for r in 0..rows {
        let mut w = [0u8; 4];
        g.copy_d2h(
            work.offset(r * row_bytes + (RADIX_STATE + RS_MODE) * 4),
            &mut w,
        )?;
        n += usize::from(u32::from_le_bytes(w) == 1);
    }
    Ok(n)
}

/// 2026-10-10: One RCOMPACT edge case: its rows (equal length P), select_k, and whether the
/// launch reads P and select_k from a device `geom` (scalars then a larger ceiling).
struct Edge {
    name: String,
    rows: Vec<Vec<f32>>,
    select_k: usize,
    geom: bool,
}

/// 2026-10-10: Run `e` lever off and on (work and `selected` poisoned before each, so init's
/// re-zeroing is exercised) at chunk override `chunks`, print its BITWISE and ROWS lines and
/// return whether it is identical (on == off == host order).
fn run_edge(
    g: &dyn GpuBackend,
    k: &Glm5NextDsaKernels,
    st: u64,
    e: &Edge,
    chunks: Option<usize>,
) -> Result<bool> {
    let (q, p) = (e.rows.len(), e.rows[0].len());
    // 2026-10-10: the kernels read row r at r * P; a shorter or longer row would be read shifted
    // while the host reference sorts the row as given (need0_eq_bin had 9000 and 9001, gx2 F).
    anyhow::ensure!(
        e.rows.iter().all(|r| r.len() == p),
        "edge case {}: rows of unequal length",
        e.name
    );
    let flat: Vec<u8> = e
        .rows
        .iter()
        .flat_map(|r| r.iter().flat_map(|x| x.to_le_bytes()))
        .collect();
    let scores = up(g, &flat)?;
    let sel_bytes = (q * e.select_k * 4).max(4);
    let selected = g.alloc(sel_bytes)?;
    let work_bytes = q * radix_row_bytes(KCAP);
    let work = g.alloc(work_bytes + compact_bytes(q))?;
    let bins = work.offset(work_bytes);
    // 2026-10-10: geom slots S, pools incl. partial, pools, select_k, np2 (as dsa_write_geom).
    let (geom_dev, n_pools, select_k) = if e.geom {
        let np2 = p.next_power_of_two().clamp(2, 2048) as i32;
        let gv = [(p * KP) as i32, p as i32, p as i32, e.select_k as i32, np2];
        (up(g, &le_i32(&gv))?, p + 4096, SELECT_K)
    } else {
        (DevicePtr::NULL, p, e.select_k)
    };
    set_radix_chunks_override(chunks);
    let mut got = Vec::new();
    for compact in [false, true] {
        g.memset(selected, POISON, sel_bytes)?;
        g.memset(work, POISON, work_bytes + compact_bytes(q))?;
        let t = RadixTopk {
            scores,
            selected,
            work,
            q_rows: q,
            n_pools,
            select_k,
            kcap: KCAP,
            geom_dev,
            bins: if compact { bins } else { DevicePtr::NULL },
        };
        launch_topk_radix(g, k, &t, st)?;
        g.synchronize(st)?;
        let mut v = vec![0u8; sel_bytes];
        g.copy_d2h(selected, &mut v)?;
        got.push(v);
    }
    set_radix_chunks_override(None);
    let n_compact = if e.select_k == 0 {
        0
    } else {
        compact_rows(g, work, q)?
    };
    let want: Vec<u8> = if e.select_k == 0 {
        vec![POISON; sel_bytes]
    } else {
        e.rows
            .iter()
            .flat_map(|r| le_i32(&host_reference(r, e.select_k)))
            .collect()
    };
    let identical = got[0] == got[1] && got[1] == want;
    let suffix = chunks.map_or(String::new(), |c| format!("/c{c}"));
    // 2026-10-10: on_eq_off is the lever's exactness; off_eq_host checks the case and the old path.
    println!(
        "RCOMPACT BITWISE S={} case={}{suffix} identical={identical} on_eq_off={} off_eq_host={}",
        p * KP,
        e.name,
        got[0] == got[1],
        got[0] == want
    );
    println!(
        "RCOMPACT ROWS S={} case={}{suffix} compact={n_compact} fallback={}",
        p * KP,
        e.name,
        q - n_compact
    );
    for b in [scores, selected, work] {
        g.free(b)?;
    }
    if !geom_dev.is_null() {
        g.free(geom_dev)?;
    }
    Ok(identical)
}

/// 2026-10-10: The RCOMPACT edge cases (see the header).
fn edges() -> Vec<Edge> {
    let m = -f32::MAX;
    let rig = |seed: u64, p: usize| rig_scores(0xED6E_0000 ^ seed, p);
    // 2026-10-10: `real` rig scores then `neg` -FLT_MAX, interleaved by a fixed stride.
    let ties = |seed: u64, real: usize, neg: usize| -> Vec<f32> {
        let mut v = vec![m; real + neg];
        let src = rig_scores(seed, real * 2);
        let mut placed = 0;
        for (i, &x) in src.iter().filter(|&&x| x != m).take(real).enumerate() {
            v[(i * 7 + 3) % (real + neg)] = x;
            placed += 1;
        }
        // 2026-10-10: top up with distinct values if the generator gave too few.
        let mut j = 0;
        while placed < real {
            if v[j] == m {
                v[j] = 0.01 * j as f32 + 3.0;
                placed += 1;
            }
            j += 1;
        }
        v
    };
    // 2026-10-10: need0 == hist0[b0]: 500 keys above [1.0, 1.125), 12 distinct keys in it,
    // the rest 0.1.
    let exact = |p: usize| -> Vec<f32> {
        let mut v = vec![0.1f32; p];
        for i in 0..500 {
            v[(i * 17 + 5) % p] = 2.0 + i as f32 / 256.0;
        }
        let mut placed = 0;
        for i in (0..p).rev() {
            if v[i] == 0.1 && placed < 12 {
                v[i] = 1.0 + placed as f32 / 128.0;
                placed += 1;
            }
        }
        v
    };
    let e = |name: &str, rows: Vec<Vec<f32>>, select_k: usize, geom: bool| Edge {
        name: name.to_string(),
        rows,
        select_k,
        geom,
    };
    vec![
        e(
            "neg_max_bin_le_cap",
            (0..4).map(|r| ties(r, 300, 2_000)).collect(),
            SELECT_K,
            false,
        ),
        e(
            "neg_max_bin_gt_cap",
            (0..4).map(|r| ties(r, 300, 2_249)).collect(),
            SELECT_K,
            false,
        ),
        e(
            "all_neg_max_p2048",
            vec![vec![m; 2_048]; 2],
            SELECT_K,
            false,
        ),
        e(
            "all_equal_le_cap",
            vec![vec![0.75; 2_000]; 2],
            SELECT_K,
            false,
        ),
        e(
            "all_equal_gt_cap",
            vec![vec![0.75; 3_000]; 2],
            SELECT_K,
            false,
        ),
        e(
            "selk_eq_pools_512",
            (0..4).map(|r| rig(r, 512)).collect(),
            SELECT_K,
            false,
        ),
        e(
            "selk_eq_pools_300",
            (0..4).map(|r| rig(r, 300)).collect(),
            300,
            false,
        ),
        e(
            "selk_zero",
            (0..4).map(|r| rig(r, 4_096)).collect(),
            0,
            false,
        ),
        e(
            "need0_eq_bin",
            // 2026-10-10: equal-length rows (the runner strides by rows[0].len()); the second is the
            // first reversed, so its 12 bin keys sit at the row start instead of the end.
            vec![exact(9_000), exact(9_000).into_iter().rev().collect()],
            SELECT_K,
            false,
        ),
        e(
            "mixed",
            vec![
                rig(1, 8_192),
                ties(2, 300, 7_892),
                vec![0.75; 8_192],
                exact(8_192),
                rig(3, 8_192),
                (0..8_192).map(|i| [0.1, 0.2, 0.3, 0.4][i % 4]).collect(),
            ],
            SELECT_K,
            false,
        ),
        e(
            "ragged_p70001",
            (0..4).map(|r| rig(r, 70_001)).collect(),
            SELECT_K,
            false,
        ),
        e(
            "ragged_p9001",
            (0..8).map(|r| rig(r, 9_001)).collect(),
            SELECT_K,
            false,
        ),
        e("geom_1row", vec![rig(9, 33_001)], SELECT_K, true),
        e(
            "geom_ties",
            (0..4).map(|r| ties(r, 300, 2_000)).collect(),
            SELECT_K,
            true,
        ),
    ]
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k = Glm5NextDsaKernels::resolve(g)?;
    if !radix_resolved(&k) {
        println!(
            "FAIL: dsa_topk_chain_prefill_microtest - the dsa_topk_radix_* entry points are \
             absent from this target's dsa_indexer module"
        );
        std::process::exit(2);
    }
    let st = g.create_stream()?;
    println!(
        "Q={Q} rows, select_k {SELECT_K}, width {WIDTH}, {LAYERS} layers, production chunks \
         {} per row, {RADIX_PASSES} radix passes; median of {REPS} replays, {} MiB L2 flush \
         before each, per-layer us",
        radix_chunks(Q),
        FLUSH_BYTES >> 20
    );
    let mut all_identical = true;
    let mut staged_ok = true;
    let mut binstat_ok = true;
    // 2026-10-10: RCOMPACT verdict; the compaction needs five more entry points.
    let rc_ok = radix_compact_resolved(&k);
    let mut rcompact_ok = true;

    for &s in CONTEXTS {
        set_radix_chunks_override(None);
        let rig = Rig::new(g, &k, st, s)?;
        let p = rig.p;

        // 2026-10-10: Production chain output, the reference for the sweep.
        let (sel0, out0) = rig.run_production()?;

        // a) per-kernel times at the production chunk policy.
        let t = rig.staged_times()?;
        let (sel_s, out_s) = rig.dump()?;
        let matches = sel_s == sel0 && out_s == out0;
        staged_ok &= matches;
        println!("STAGED S={s} staged_matches_production={matches}");
        let find = t[2] + t[4] + t[6];
        let total: f64 = t.iter().sum();
        let graph_us = rig.chain_us()?;
        println!(
            "CHAIN KERNELS S={s} init_us={:.2} hist0_us={:.2} hist1_us={:.2} hist2_us={:.2} \
             find_us={find:.2} gather_us={:.2} sort_us={:.2} expand_us={:.2} total_us={total:.2} \
             graph_us={graph_us:.2}",
            t[0], t[1], t[3], t[5], t[7], t[8], t[9]
        );

        // b) chunk sweep.
        for &c in SWEEP {
            set_radix_chunks_override(Some(c));
            let (sel, out) = rig.run_production()?;
            let identical = sel == sel0 && out == out0;
            all_identical &= identical;
            let chain = rig.chain_us()?;
            let tc = rig.staged_times()?;
            let hist_gather = tc[1] + tc[3] + tc[5] + tc[7];
            println!(
                "CHAIN CHUNKS S={s} c={c} chain_us={chain:.2} hist_gather_us={hist_gather:.2} \
                 identical={identical}"
            );
        }
        set_radix_chunks_override(None);

        // c) expand ablation (state is the end of the chain: selected valid, work untouched).
        rig.run_production()?;
        let on = rig.time_stage(Stage::Expand(true))?;
        let off = rig.time_stage(Stage::Expand(false))?;
        println!(
            "CHAIN EXPAND S={s} tail_on_us={on:.2} tail_off_us={off:.2} tail_share={:.3}",
            (on - off) / on.max(1e-9)
        );

        // d) pass-0 bin occupancy.
        let (mut counts, _) = bin_occupancy(&rig)?;
        counts.sort_unstable();
        let med = counts[counts.len() / 2];
        let max = counts[counts.len() - 1];
        println!(
            "CHAIN BIN S={s} median={med} ({:.3}% of P) max={max} (P={p}, {:.3}% of P)",
            100.0 * f64::from(med) / p as f64,
            100.0 * f64::from(max) / p as f64
        );
        // e) compact prototype vs gather (gather_us is the pass-2-state stage time above).
        if k.topk_radix_compact.0 != 0 && k.topk_radix_binstat.0 != 0 {
            let (compact_us, b0_med, matches) = compact_arm(&rig)?;
            binstat_ok &= matches;
            println!(
                "CHAIN COMPACT S={s} compact_us={compact_us:.2} gather_us={:.2} ratio={:.3} \
                 b0_median={b0_med}",
                t[7],
                compact_us / t[7].max(1e-9)
            );
            println!("CHAIN COMPACT CHECK S={s} binstat_matches_compact={matches}");
        } else {
            println!("CHAIN COMPACT S={s} skipped: dsa_topk_radix_compact/binstat absent");
        }
        // f) RCOMPACT on the rig: lever on vs the lever-off production outputs, and chain time.
        if rc_ok {
            set_radix_chunks_override(None);
            let (sel_on, out_on) = rig.run_production_with(true)?;
            let identical = sel_on == sel0 && out_on == out0;
            rcompact_ok &= identical;
            println!("RCOMPACT BITWISE S={s} case=rig identical={identical}");
            let n = compact_rows(g, rig.layers[0].work, Q)?;
            println!(
                "RCOMPACT ROWS S={s} case=rig compact={n} fallback={}",
                Q - n
            );
            let off_us = rig.chain_us_with(false)?;
            let on_us = rig.chain_us_with(true)?;
            println!(
                "RCOMPACT TIME S={s} off_us={off_us:.2} on_us={on_us:.2} ratio={:.3}",
                on_us / off_us.max(1e-9)
            );
        }
        rig.free()?;
    }

    // 2026-10-10: RCOMPACT edge cases, at the production chunk count and at 5 chunks.
    if rc_ok {
        for e in edges() {
            for chunks in [None, Some(5)] {
                rcompact_ok &= run_edge(g, &k, st, &e, chunks)?;
            }
        }
        if rcompact_ok {
            println!("PASS - RCOMPACT: every RCOMPACT BITWISE line identical");
        } else {
            println!("FAIL - RCOMPACT: an RCOMPACT BITWISE line differs");
        }
    } else {
        println!(
            "FAIL - RCOMPACT: the compaction entry points (dsa_topk_radix_compact, _select0, \
             _hist_rc, _find_rc, _gather_rc) are absent from this target"
        );
    }

    if !binstat_ok {
        println!("FAIL - CHAIN COMPACT: binstat counts differ from the compact kernel's");
        std::process::exit(1);
    }
    if all_identical && staged_ok {
        println!("PASS - CHAIN gate0: chunk outputs identical in every S and c");
        if !rc_ok {
            std::process::exit(2);
        }
        if !rcompact_ok {
            std::process::exit(1);
        }
        Ok(())
    } else {
        println!(
            "FAIL - CHAIN gate0: chunk outputs identical={all_identical}, \
             staged_matches_production={staged_ok}"
        );
        std::process::exit(1);
    }
}
