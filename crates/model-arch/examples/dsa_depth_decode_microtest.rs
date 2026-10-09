// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Where a GLM-5.3 decode step's time goes at long context, for the DSA selection
//! kernels (and, if its kernel is present, the MLA decode attend). At 532K context the race
//! measured ~390-450 ms a step against ~75 ms at short context; this times the selection's share.
//!
//! Per decode step at K = 3 each of the 11 DSA text layers runs `select_tokens` once per verify
//! row: 33 calls, each a `q_rows = 1` ceiling launch (`DsaSelectLaunch::Ceiling`) over the
//! production 512K capacity (`--max-seq-len` 540,672, 135,168 pools) with a device `geom` that
//! the production `dsa_write_geom` kernel fills from the row's `seq_len` (`layer/decode_k.rs`).
//! The kernels walk the live pools, so their cost grows with the live context S while the grid
//! stays at the production grid-stride size.
//!
//! * Live context S in {131,072; 262,144; 532,480} tokens (row `r` of the step has `S + r`
//!   tokens, so rows 1 and 2 carry a partial trailing pool), distinct per-layer caches (k, gate,
//!   valid; every 23rd token invalid), production `DsaSelectScratch`, production grids.
//! * Timing: CUDA events around a replay of a captured 33-call graph, median of 51 replays after
//!   3 warm-ups, ms per decode step. One `DEPTH` line per S: the full `select_tokens`, then
//!   `dsa_kpool_compress`, `dsa_index_scores`, `dsa_topk_pools` and `dsa_expand_selection` each
//!   launched alone with exactly the arguments `select_tokens` passes (`examples/common/
//!   dsa_depth_rig.rs` cites the lines mirrored), and a `SUM` line of the stages against the full
//!   selector. Each stage graph runs on the buffers the staged sanity pass left, so a stage that
//!   reads another's output reads live data.
//! * Sanity: at each S, for every layer and verify row, the token ids `select_tokens` writes
//!   equal those of the four stage launches, compared as raw bytes of the whole `[3, out_width]`
//!   buffer after both were filled with a poison byte; a run whose tokens are all `-1` or poison
//!   fails. Timing never decides the verdict.
//! * Optional `ATTEND` line: the MLA decode attend (`attend::decode_attention`, the production
//!   dispatcher, so the `METRALE_GLM_DSA_MLA_*` levers in the environment apply) over the
//!   selected tokens the selector just produced, gathered from a paged FP8 latent cache of S
//!   tokens per layer (64-token blocks, 512 latent dims, 32 heads, three rows in one launch per
//!   layer), 11 layers a step. Random cache data; timing only, no correctness check (the output
//!   is only checked to be non-zero, printed as `out_nonzero`).
//!
//! 2026-10-08: Production dispatch of the car (c26). Contexts default to 131,072 and 262,144
//! (`DEPTH_MT_CONTEXTS=a,b,c` overrides; 532,480 is available). With `METRALE_GLM_DSA_POOL_CACHE=1`
//! the fixture keeps per-layer persistent pool arrays warmed to S, the `full` line is
//! `dsa_write_geom_pk` + `select_tokens` with the cache (only the new pools compress), and the
//! compress stage is `dsa_write_geom_pk` + `dsa_kpool_compress_incr` at the ceiling grid. The
//! topk stage follows `METRALE_GLM_DSA_TOPK_RADIX`; the scores stage is the plain scorer a
//! ceiling launch always takes (TC/TC2 need exact host geometry). One line per stage and S:
//! `STAGE <name> S=<s> ms_per_step=<x> impl=<which>`. With the cache on, the sanity pass also
//! compares its tokens with the cache-less production `select_tokens`.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and a `PASS` line when every full-vs-staged token compare is byte-equal; 1 otherwise;
//! 2 when a selection kernel (or `dsa_write_geom`) is absent from this target.
//!
//! Run (gb10 common kernels):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_depth_decode_microtest
//! Needs about 7 GB of device memory (11 layers' indexer caches, plus 11 latent caches at a time
//! for the attend), so stop a running serve on the node first.

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_dsa::Glm5NextDsaKernels;
use metrale_model_arch::glm5next_dsa::attend::{
    DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaDecodeKernel, decode_attention,
};
use metrale_model_arch::glm5next_dsa::pool_cache::dsa_pool_cache;
use metrale_model_arch::glm5next_dsa::select::DsaSelectGeometry;

#[path = "common/dsa_depth_rig.rs"]
mod dsa_depth_rig;
use dsa_depth_rig::*;

// 2026-10-06: CUDA driver event API for kernel-only timing, declared as in
// `dsa_grid_stride_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-08: Live contexts timed (tokens): `DEPTH_MT_CONTEXTS` (comma list), default the
/// car's 131,072 and 262,144; 532,480 is available through the variable.
fn contexts() -> Vec<usize> {
    let raw = std::env::var("DEPTH_MT_CONTEXTS").unwrap_or_else(|_| "131072,262144".into());
    raw.split(',')
        .map(|t| t.trim().parse().expect("DEPTH_MT_CONTEXTS: comma list of token counts"))
        .collect()
}
/// 2026-10-06: Selector calls in a K = 3 decode step: one per layer per verify row.
const CALLS: usize = LAYERS * ROWS;
const REPLAYS: usize = 51;
/// 2026-10-06: Paged latent cache of the attend arm: 64-token blocks of 512 FP8 latent dims.
const BLOCK: usize = 64;
const KVL: usize = 512;
const HEADS: usize = 32;
const K_SCALE: f32 = 0.0173;

/// 2026-10-06: Median ms of one replay of the graph `f` captures on `s`, over `REPLAYS`
/// replays each bracketed by CUDA events, after 3 warm-up replays.
fn time_graph(g: &dyn GpuBackend, s: u64, f: &mut dyn FnMut(u64) -> Result<()>) -> Result<f64> {
    g.begin_capture(s)?;
    if let Err(e) = f(s) {
        g.abort_capture_if_active(s);
        return Err(e);
    }
    let graph = g.end_capture(s)?;
    for _ in 0..3 {
        g.launch_graph(graph, s)?;
    }
    g.synchronize(s)?;
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
        unsafe { check(cuEventRecord(e.0, s), "cuEventRecord(start)")? };
        g.launch_graph(graph, s)?;
        // SAFETY: as above.
        unsafe { check(cuEventRecord(e.1, s), "cuEventRecord(end)")? };
    }
    g.synchronize(s)?;
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

/// 2026-10-06: Compare, for every layer and verify row, the tokens `select_tokens` writes with
/// those of the four stage launches. Returns (layers compared, equal, tokens selected in row 0
/// of the last layer, i.e. entries that are neither `-1` nor the poison).
fn sanity(g: &dyn GpuBackend, f: &Fixture, s: usize) -> Result<(usize, usize, usize)> {
    let (mut legs, mut equal, mut live) = (0, 0, 0);
    let mut nc_diff = 0;
    for l in 0..LAYERS {
        f.poison_tokens(g)?;
        for r in 0..ROWS {
            f.full(g, s, l, r, 0)?;
            f.staged(g, l, r, 0)?;
        }
        let (a, b) = (
            down(g, f.scratch.tokens(), f.token_bytes())?,
            down(g, f.st.tokens, f.token_bytes())?,
        );
        legs += 1;
        equal += usize::from(a == b);
        if f.pool_cache {
            // 2026-10-08: The pool-cache selection against the production cache-less
            // `select_tokens` (full compress into a scratch) on the same geometry.
            for r in 0..ROWS {
                f.full(g, s, l, r, 0)?;
                f.select(g, s, l, r, false, 0)?;
            }
            let c = down(g, f.scratch_nc.tokens(), f.token_bytes())?;
            let a2 = down(g, f.scratch.tokens(), f.token_bytes())?;
            legs += 1;
            equal += usize::from(a2 == c);
            nc_diff += usize::from(a2 != c);
        }
        let poison = i32::from_le_bytes([POISON; 4]);
        live += a
            .chunks_exact(4)
            .map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .filter(|&t| t >= 0 && t != poison)
            .count();
    }
    if nc_diff > 0 {
        println!("pool-cache select_tokens differs from cache-less select_tokens in {nc_diff} layers");
    }
    Ok((legs, equal, live))
}

/// 2026-10-06: The MLA decode attend over the selected tokens now in `f.scratch`, per decode
/// step: one `decode_attention` launch per layer (three rows, one launch), each layer with its own
/// paged FP8 latent cache of `s + ROWS` tokens. Returns (ms per step, output non-zero).
fn attend(
    g: &dyn GpuBackend,
    f: &Fixture,
    kernel: Glm5NextDsaDecodeKernel,
    s: usize,
    st: u64,
) -> Result<(f64, bool)> {
    let cfg = &f.cfg;
    let blocks = (s + ROWS).div_ceil(BLOCK);
    let table_w = MAX_SEQ.div_ceil(BLOCK);
    // 2026-10-06: Random E4M3 codes with the two NaN codes (0x7F, 0xFF) excluded, a 16 MiB pattern
    // tiled over the cache.
    let mut x = 0x5EED_u64;
    let pat: Vec<u8> = (0..16usize << 20)
        .map(|_| loop {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = (x >> 33) as u8;
            if (b & 0x7F) != 0x7F {
                break b;
            }
        })
        .collect();
    let cache_bytes = blocks * BLOCK * KVL;
    let host: Vec<u8> = pat.iter().copied().cycle().take(cache_bytes).collect();
    let caches = (0..LAYERS)
        .map(|_| up(g, &host))
        .collect::<Result<Vec<_>>>()?;
    // 2026-10-06: Identity block table per row, past the live blocks pointing at block 0.
    let table: Vec<i32> = (0..ROWS)
        .flat_map(|_| (0..table_w).map(|b| if b < blocks { b as i32 } else { 0 }))
        .collect();
    let tables = up(g, &i32_bytes(&table))?;
    let seq: Vec<i32> = (0..ROWS).map(|r| (s + r) as i32).collect();
    let seq_lens = up(g, &i32_bytes(&seq))?;
    let q = up(g, &vec![0x3Cu8; ROWS * HEADS * KVL * 2])?;
    let out = g.alloc(ROWS * HEADS * KVL * 2)?;
    g.memset(out, 0, ROWS * HEADS * KVL * 2)?;
    let paging = DsaDecodePaging {
        num_seqs: ROWS,
        num_q_heads: HEADS,
        num_kv_heads: 1,
        max_blocks_per_seq: table_w,
        block_size: BLOCK,
        cache_stride_bytes: (BLOCK * KVL) as u64,
    };
    let geom = DsaSelectGeometry::plan(cfg, s, ROWS)?;
    let inputs = |c| DsaDecodeInputs {
        q,
        k_cache: c,
        v_cache: c,
        out,
        block_tables: tables,
        seq_lens,
        sel_indices: f.scratch.tokens(),
        k_scale: K_SCALE,
        v_scale: K_SCALE,
    };
    let ms = time_graph(g, st, &mut |stream| {
        for &c in &caches {
            decode_attention(g, kernel, cfg, &geom, &paging, &inputs(c), stream)?;
        }
        Ok(())
    })?;
    let nonzero = down(g, out, ROWS * HEADS * KVL * 2)?.iter().any(|&b| b != 0);
    for p in caches.into_iter().chain([tables, seq_lens, q, out]) {
        g.free(p).ok();
    }
    Ok((ms, nonzero))
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kernels = match Glm5NextDsaKernels::resolve(g) {
        Ok(k) if k.write_geom.0 != 0 => k,
        r => {
            println!(
                "a DSA selection kernel or dsa_write_geom is absent from this target ({}) - SKIP",
                r.err().map_or("write_geom".into(), |e| e.to_string())
            );
            std::process::exit(2);
        }
    };
    if dsa_pool_cache() && (kernels.kpool_compress_incr.0 == 0 || kernels.write_geom_pk.0 == 0) {
        println!("METRALE_GLM_DSA_POOL_CACHE=1 but dsa_kpool_compress_incr / dsa_write_geom_pk are absent - SKIP");
        std::process::exit(2);
    }
    let f = Fixture::new(g, kernels)?;
    let (ic, isc, itk) = f.impls();
    println!(
        "pool_cache={} radix_topk={} capacity {MAX_POOLS} pools ({MAX_SEQ} tokens); grids compress \
         {} scores {} blocks; {LAYERS} layers x {ROWS} rows = {CALLS} calls a step",
        f.pool_cache, f.radix, f.grids.0, f.grids.1
    );
    let attend_kernel = Glm5NextDsaDecodeKernel::resolve_for(g, &f.cfg).ok();
    if attend_kernel.is_none() {
        println!("glm5next_dsa_mla_decode_fp8 absent from this target - ATTEND arm skipped");
    }
    let st = g.create_stream()?;
    let ctxs = contexts();
    let (mut legs, mut equal, mut live) = (0, 0, 0);
    for &s in &ctxs {
        f.set_context(g, s)?;
        let (l, e, n) = sanity(g, &f, s)?;
        (legs, equal, live) = (legs + l, equal + e, live + n);
        // 2026-10-06: Every graph is 33 calls: call `i` is layer `i / 3`, verify row `i % 3`.
        let each = |g: &dyn GpuBackend, body: &dyn Fn(usize, usize, u64) -> Result<()>| {
            time_graph(g, st, &mut |stream| {
                (0..CALLS).try_for_each(|i| body(i / ROWS, i % ROWS, stream))
            })
        };
        let full = each(g, &|l, r, sm| f.full(g, s, l, r, sm))?;
        let compress = each(g, &|l, r, sm| f.compress(g, l, r, sm))?;
        let scores = each(g, &|l, r, sm| f.scores(g, l, r, sm))?;
        let topk = each(g, &|_, r, sm| f.topk(g, r, sm))?;
        let expand = each(g, &|l, r, sm| f.expand(g, l, r, sm))?;
        let cname = if f.pool_cache { "compress_incr" } else { "compress" };
        let lines = [
            (cname, compress, ic.as_str()),
            ("scores", scores, isc.as_str()),
            ("topk", topk, itk.as_str()),
            ("expand", expand, "dsa_expand_selection"),
            ("full", full, "select_tokens(Ceiling), production dispatch"),
        ];
        for (name, ms, imp) in lines {
            println!("STAGE {name} S={s} ms_per_step={ms:.3} impl={imp}");
        }
        let sum = compress + scores + topk + expand;
        println!(
            "SUM S={s} sum_of_stages={sum:.3} full={full:.3} sum/full={:.3} ({CALLS} calls a step)",
            sum / full
        );
        if let Some(k) = attend_kernel {
            // 2026-10-06: Tokens for the attend: the real selection of the last layer.
            for r in 0..ROWS {
                f.full(g, s, 0, r, 0)?;
            }
            g.synchronize(0)?;
            let (ms, nonzero) = attend(g, &f, k, s, st)?;
            let env = |k: &str| std::env::var(k).unwrap_or_else(|_| "unset".into());
            println!(
                "STAGE attend S={s} ms_per_step={ms:.3} impl=decode_attention(MLA_SPLIT={} \
                 MLA_HEADGROUP={}; {LAYERS} launches x {ROWS} rows; out_nonzero={nonzero})",
                env("METRALE_GLM_DSA_MLA_SPLIT"),
                env("METRALE_GLM_DSA_MLA_HEADGROUP")
            );
        }
    }

    println!("{equal}/{legs} layer compares byte-equal; {live} selected token ids in all");
    if legs == 0 || live == 0 {
        println!("FAIL - nothing, or only empty selections, was compared; this run proves nothing.");
        std::process::exit(1);
    }
    if equal != legs {
        println!("FAIL - select_tokens and the stage-by-stage launches wrote different tokens.");
        std::process::exit(1);
    }
    println!(
        "PASS - select_tokens tokens are byte-equal to the stage launches at {} contexts x \
         {LAYERS} layers x {ROWS} rows.",
        ctxs.len()
    );
    Ok(())
}
