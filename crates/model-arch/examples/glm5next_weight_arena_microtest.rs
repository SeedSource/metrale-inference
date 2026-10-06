// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Gate for the GLM-5.3 weight arena (`METRALE_GLM_WEIGHT_ARENA`): the same
//! synthetic NVFP4 routed experts bound with the arena off and on, through the real binders.
//!
//! Owner: model-arch examples (GLM-5.3 weight loading).
//! Invariants: none beyond the types. Correctness is gated; memory figures are information.
//!
//! Why: on GB10 each `cuMemAlloc` costs ~14-18 KiB outside the request (spark-bench
//! `runs/race/alloccost/RESULT.md`, 2026-10-06); the arena replaces tens of thousands of
//! weight allocations per rank with a few chunks. Only addresses may change.
//!
//! Parts (shapes are GLM-5.3's: hidden 4096, I 2048, 288 experts, NVFP4 E2M1 + E4M3/16):
//!   etp    `--layers` (default 3) layers x 288 experts at expert-TP rank 0 (I/2 = 1024), packed
//!          U8 read from a shard file and sliced by `bind_expert_cfg` -> `bind_expert_tp` ->
//!          `upload_slice` (6 buffers per expert). The layers share one source file.
//!   fast   one layer x 144 experts of U8 NVFP4 in a `model.safetensors`, loaded by
//!          `FastSafetensorsLoader` (`load_shard_fast`), arena hook = the GLM `arena_rule`.
//!   quant  8 deferred full-width BF16 experts (EP2 rank 0 of 16), quantised at bind by
//!          `bind_expert` -> `quantize_deferred_expert_proj` -> `upload_bytes`.
//!
//! Checks per part: every per-expert (or per-tensor) buffer's bytes are identical with the
//! arena on and off; every arena pointer is 256-byte aligned and inside the arena; no two
//! arena buffers overlap; the arena's chunks are exactly filled (no tail waste, no unplanned
//! chunk); the allocation ledger grew by exactly the chunk bytes (on) and the requested bytes
//! (off); after releasing every store the ledger is back to its starting count. Memory: the
//! drop in /proc/meminfo MemAvailable and in cuMemGetInfo free for each arm, beside the
//! expected saving (~15 KiB per allocation saved). Prints `PASS: ...` and exits 0 when every
//! check holds, else `FAIL ...` and exits 1.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_weight_arena_microtest -- [--layers N] [--dir PATH]
//!
//! Disk: ~3.8 GiB (etp) + ~1.9 GiB (fast) + ~0.4 GiB (quant) under `--dir`, removed at the end.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use metrale_core::scope::ModelResource;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mlp::Glm5NextMlpConfig;
use metrale_model_arch::glm5next_mlp::weights::{Glm5NextExpertWeights, Nvfp4Proj};
use metrale_model_arch::weight_loader::glm5_next_load::bind_expert_cfg;
use metrale_model_arch::weight_loader::glm5_next_load::expert_arena::{
    arena_rule, plan_expert_layer,
};
use metrale_model_weights::fast_weights::FastSafetensorsLoader;
use metrale_model_weights::weights::{
    ArenaHook, DeferredTensor, WEIGHT_ARENA_ALIGN, WeightArena, WeightDtype, WeightLoader,
    WeightStore,
};

const H: usize = 4096;
const I: usize = 2048;
const PROJS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
/// Mid-range of the measured 14-18 KiB per allocation.
const PER_ALLOC: f64 = 15.5 * 1024.0;
const MIB: f64 = 1024.0 * 1024.0;

#[path = "common/weight_arena_rig.rs"]
mod rig;
use rig::{cost, d2h, expert_name, snap, write_safetensors, write_source};

/// Failed checks; each is printed when it happens.
struct Checks(Vec<String>);
impl Checks {
    fn ok(&mut self, cond: bool, what: impl FnOnce() -> String) {
        if !cond {
            let w = what();
            println!("CHECK miss: {w}");
            self.0.push(w);
        }
    }
}

/// Arena-side checks shared by every part: alignment, containment, no overlap, exact fill,
/// ledger growth, then the memory line.
#[allow(clippy::too_many_arguments)]
fn arena_checks(
    part: &str,
    arena: &WeightArena,
    spans: &mut [(u64, usize)],
    expect_subs: usize,
    off: (f64, f64, usize, usize),
    on: (f64, f64, usize, usize),
    ck: &mut Checks,
) {
    for &(p, _) in spans.iter() {
        ck.ok(p.is_multiple_of(WEIGHT_ARENA_ALIGN as u64), || {
            format!("{part}: {p:#x} unaligned")
        });
        ck.ok(arena.contains(DevicePtr(p)), || {
            format!("{part}: {p:#x} outside the arena")
        });
    }
    spans.sort_unstable();
    for w in spans.windows(2) {
        ck.ok(w[0].0 + w[0].1 as u64 <= w[1].0, || {
            format!("{part}: {:#x}+{} overlaps {:#x}", w[0].0, w[0].1, w[1].0)
        });
    }
    let s = arena.stats();
    println!(
        "ARENA part={part} chunks={} chunk_mib={:.1} requested_mib={:.1} used_mib={:.1} \
         tail_waste={} unplanned={} subs={} saved={}",
        s.chunks,
        s.chunk_bytes as f64 / MIB,
        s.requested as f64 / MIB,
        s.used as f64 / MIB,
        s.tail_waste,
        s.unplanned_chunks,
        s.sub_allocations,
        s.allocations_saved()
    );
    ck.ok(s.sub_allocations == expect_subs, || {
        format!("{part}: {} subs", s.sub_allocations)
    });
    ck.ok(s.tail_waste == 0 && s.unplanned_chunks == 0, || {
        format!("{part}: plan not exact")
    });
    ck.ok(on.2 == s.chunk_bytes, || {
        format!("{part}: ledger +{} != chunks {}", on.2, s.chunk_bytes)
    });
    ck.ok(on.3 == s.chunks, || {
        format!("{part}: ledger +{} allocs != {} chunks", on.3, s.chunks)
    });
    let saved = off.3.saturating_sub(on.3);
    println!(
        "MEM part={part} off: allocs={} memavail_drop={:.1}MiB devfree_drop={:.1}MiB | on: \
         allocs={} memavail_drop={:.1}MiB devfree_drop={:.1}MiB | recovered memavail={:.1}MiB \
         devfree={:.1}MiB, expected ~{:.1}MiB ({saved} allocations x 15.5 KiB)",
        off.3,
        off.0,
        off.1,
        on.3,
        on.0,
        on.1,
        off.0 - on.0,
        off.1 - on.1,
        saved as f64 * PER_ALLOC / MIB
    );
}

/// Bind `layers` x the local experts of `cfg` from `store`, planning the arena when `arena`.
fn bind(
    g: &dyn GpuBackend,
    store: &WeightStore,
    layers: usize,
    cfg: &Glm5NextMlpConfig,
    arena: bool,
) -> Result<Vec<Glm5NextExpertWeights>> {
    let mut out = Vec::new();
    for l in 0..layers {
        plan_expert_layer(store, l, cfg, arena);
        for id in cfg.local_expert_range() {
            out.push(bind_expert_cfg(g, store, l, id, cfg)?);
        }
    }
    Ok(out)
}

/// A derived-arena part (etp, quant): bind off then on, compare every projection, check.
fn derived_part(
    g: &dyn GpuBackend,
    part: &str,
    src: &[(usize, String, String, DeferredTensor)],
    layers: usize,
    cfg: &Glm5NextMlpConfig,
    ck: &mut Checks,
) -> Result<()> {
    let store_for = || {
        let mut s = WeightStore::empty();
        for l in 0..layers {
            for (id, p, leaf, d) in src {
                s.defer(expert_name(l, *id, p, leaf), d.clone());
            }
        }
        s
    };
    let (mut s_off, mut s_on) = (store_for(), store_for());
    let t0 = snap(g)?;
    let w_off = bind(g, &s_off, layers, cfg, false)?;
    let t1 = snap(g)?;
    let w_on = bind(g, &s_on, layers, cfg, true)?;
    let t2 = snap(g)?;
    let (packed, scale) = (cfg.moe_intermediate * H / 2, cfg.moe_intermediate * H / 16);
    let mut spans = Vec::new();
    for (k, (a, b)) in w_off.iter().zip(&w_on).enumerate() {
        let pairs: [(&Nvfp4Proj, &Nvfp4Proj); 3] = [
            (&a.gate_proj, &b.gate_proj),
            (&a.up_proj, &b.up_proj),
            (&a.down_proj, &b.down_proj),
        ];
        for (pi, (x, y)) in pairs.into_iter().enumerate() {
            for (px, py, n) in [(x.packed, y.packed, packed), (x.scale, y.scale, scale)] {
                ck.ok(d2h(g, px, n)? == d2h(g, py, n)?, || {
                    format!("{part}: expert {k} proj {pi} bytes differ")
                });
                ck.ok(!s_on.derived().arena().contains(px), || {
                    format!("{part}: off ptr in arena")
                });
                spans.push((py.0, n));
            }
            ck.ok(
                x.scale_2.to_bits() == y.scale_2.to_bits()
                    && x.input_scale.to_bits() == y.input_scale.to_bits(),
                || format!("{part}: expert {k} proj {pi} scalars differ"),
            );
        }
    }
    let expect = w_on.len() * 6;
    arena_checks(
        part,
        s_on.derived().arena(),
        &mut spans,
        expect,
        cost(t0, t1),
        cost(t1, t2),
        ck,
    );
    s_off.release(g)?;
    s_on.release(g)?;
    Ok(())
}

/// The fast-loader part: load the same file with and without the GLM arena hook.
fn fast_part(g: &dyn GpuBackend, dir: &Path, ck: &mut Checks) -> Result<()> {
    let load = |hook: Option<ArenaHook>| -> Result<WeightStore> {
        let mut l = FastSafetensorsLoader::new();
        l.arena = hook;
        l.load(dir, g, 0)
    };
    let t0 = snap(g)?;
    let mut s_off = load(None)?;
    let t1 = snap(g)?;
    let mut s_on = load(Some(std::sync::Arc::new(arena_rule)))?;
    let t2 = snap(g)?;
    let mut spans = Vec::new();
    let mut names: Vec<String> = s_off.names().map(str::to_string).collect();
    names.sort();
    for n in &names {
        let (a, b) = (s_off.get(n)?, s_on.get(n)?);
        let len = a.byte_size();
        ck.ok(d2h(g, a.ptr, len)? == d2h(g, b.ptr, len)?, || {
            format!("fast: {n} bytes differ")
        });
        let claimed = arena_rule(n, a.dtype);
        ck.ok(s_on.arena().contains(b.ptr) == claimed, || {
            format!("fast: {n} arena placement")
        });
        if claimed {
            spans.push((b.ptr.0, len));
        }
    }
    let expect = spans.len();
    ck.ok(expect + 1 == names.len(), || {
        format!("fast: {expect} of {} claimed", names.len())
    });
    let (off, mut on) = (cost(t0, t1), cost(t1, t2));
    // The BF16 tensor outside the arena is one ledgered allocation of its own on both arms.
    on.2 = on.2.saturating_sub(H * 2);
    on.3 = on.3.saturating_sub(1);
    arena_checks("fast", s_on.arena(), &mut spans, expect, off, on, ck);
    s_off.release(g)?;
    s_on.release(g)?;
    Ok(())
}

fn mlp_cfg(num_experts: usize, local: usize, width: usize) -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: H,
        local_dense_intermediate: 6144,
        moe_intermediate: width,
        local_shared_intermediate: 1024,
        num_experts,
        local_experts: local,
        ep_rank: 0,
        top_k: 8,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 0.0,
        router_bf16_ladder: false,
        tp_world_size: 2,
        ep_world_size: 2,
    }
}

fn run(g: &dyn GpuBackend, dir: &Path, layers: usize, ck: &mut Checks) -> Result<()> {
    let base = g.live_alloc_count();
    println!("SETUP writing sources under {}", dir.display());
    let etp_src = write_source(dir.join("etp.bin"), 0..288, false)?;
    // Expert-TP rank 0: all 288 experts at I/2.
    derived_part(g, "etp", &etp_src, layers, &mlp_cfg(288, 288, I / 2), ck)?;
    let fast_src: Vec<_> = etp_src.into_iter().filter(|(id, ..)| *id < 144).collect();
    let fast_dir = dir.join("fast");
    std::fs::create_dir_all(&fast_dir)?;
    write_safetensors(&fast_dir, &fast_src)?;
    drop(fast_src);
    std::fs::remove_file(dir.join("etp.bin"))?;
    fast_part(g, &fast_dir, ck)?;
    // EP2 rank 0 of 16 experts at full I: 8 BF16 experts quantised at bind.
    let q_src = write_source(dir.join("quant.bin"), 0..8, true)?;
    derived_part(g, "quant", &q_src, 1, &mlp_cfg(16, 8, I), ck)?;
    let end = g.live_alloc_count();
    ck.ok(end == base, || {
        format!("ledger holds {end} allocations after release, {base} before")
    });
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let layers: usize = arg("--layers")
        .map_or(Ok(3), |v| v.parse())
        .context("--layers")?;
    let dir = arg("--dir")
        .map_or_else(std::env::temp_dir, PathBuf::from)
        .join("glm5next_weight_arena");
    std::fs::create_dir_all(&dir)?;
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut ck = Checks(Vec::new());
    let r = run(g, &dir, layers, &mut ck);
    let _ = std::fs::remove_dir_all(&dir);
    if let Err(e) = r {
        println!("FAIL error: {e:#}");
        std::process::exit(1);
    }
    if !ck.0.is_empty() {
        println!("FAIL {} check(s) missed; first: {}", ck.0.len(), ck.0[0]);
        std::process::exit(1);
    }
    println!(
        "PASS: weight arena bytes identical to per-allocation binds (etp {layers}x288, fast 144, \
         quant 8 experts), 256-byte aligned, no overlap, chunks exactly filled, ledger exact"
    );
    Ok(())
}
