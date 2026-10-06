// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Stage-5 gate of the DSA pool cache (`METRALE_GLM_DSA_POOL_CACHE=1`, race #79):
//! the cached path's pool keys, pool ids, pool validity and selected token ids are
//! byte-identical to today's full recompute at every selection, across every write pattern
//! production uses. Written 2026-10-06, NOT yet run (needs a GB10).
//!
//! Two copies of one GLM-5.3 DSA indexer cache get the same random rows (`examples/common/
//! pool_cache_parity_rig.rs`): A on the old path (full-length keys, `dsa_kpool_compress` over
//! every pool into the select scratch each pass), B a pool-cache `Glm5NextDsaState` (8,448-row
//! ring, persistent pools, `dsa_kpool_compress_incr` from the watermark, a select scratch
//! without pool regions). Per context target S_max (139,264 tokens, then 532,480) the script:
//! 1. Two full-width prefill windows: 8,192 rows written, then 32 sub-chunk selections of 256
//!    query rows each (`decode_k_wide`).
//! 2. MTP drafter context rows to S_max - 64 in 256-row tiles with no selection, each followed
//!    by the compress-only pass (`write_kv_rows` / `pool_compress_written`), a write far longer
//!    than the ring; then one selection.
//! 3. Eight decode appends, one row and one selection each.
//! 4. Eight K = 3 verify steps with 0..3 rows rejected in turn, alternating the host path
//!    (exact launches, explicit rewind, the host `dsa_pk_len_clamp`) and graph replay
//!    (captured once: `dsa_indexer_store_ring` lowering the device watermark,
//!    `dsa_write_geom_pk`, ceiling selections; `replay_room` before, `sync_to` after).
//! 5. Aux blob v2 snapshot, restore into a fresh state, then four decode appends and four
//!    verify steps on the restored state (graph recaptured).
//! At every selection: the pools over `[0, floor(S / 4))` and the selected token ids, A vs B,
//! as raw bytes (a selection of nothing fails the run). Graph steps compare each row's tokens
//! and the pools after the step.
//! Controls, which must fire: one flipped bit in B's ring makes the next compare differ, and a
//! rewind deeper than the ring returns `Err` and moves nothing.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and a `PASS` line only when every compare is equal and both controls fire; 1
//! otherwise; 2 when a kernel it needs is absent from this target.
//!
//! Run (gb10 common kernels; sets `METRALE_GLM_DSA_POOL_CACHE=1` itself):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_pool_cache_parity_microtest
//! About 1 GB of device memory; minutes, dominated by the full recompute and the compares.

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_dsa::Glm5NextDsaKernels;

#[path = "common/pool_cache_parity_rig.rs"]
mod pool_cache_parity_rig;
use pool_cache_parity_rig::*;

/// 2026-10-06: Context targets: about 140K, and one near 532K.
const TARGETS: &[usize] = &[139_264, 532_480];

/// 2026-10-06: One target's script; returns whether both controls fired.
fn run(p: &mut Pair, s_max: usize) -> Result<bool> {
    p.wide(32)?;
    p.wide(32)?;
    println!("  wide windows: S = {}, {} compares", p.a_len, p.compared);
    p.drafter_ctx(s_max - 64)?;
    p.select(1, "after drafter context")?;
    println!("  drafter context: S = {}", p.a_len);
    let steps = |p: &mut Pair, appends: usize, verifies: usize| -> Result<()> {
        for _ in 0..appends {
            p.write(1)?;
            p.select(1, "decode append")?;
        }
        for i in 0..verifies {
            let seq = p.a_len - i % 4;
            if i % 2 == 0 {
                p.verify_exact(seq)?;
            } else {
                p.verify_graph(seq)?;
            }
        }
        Ok(())
    };
    // 2026-10-06: The first verify starts at the current length; each later one after
    // rejecting `i % 4` of the previous step's three rows.
    steps(p, 8, 8)?;
    println!(
        "  decode + verify: S = {}, {} compares",
        p.a_len, p.compared
    );
    let bytes = p.aux_round_trip()?;
    println!(
        "  aux v2: {bytes} B at S = {} ({:.1} B/token)",
        p.a_len,
        bytes as f64 / p.a_len as f64
    );
    steps(p, 4, 4)?;
    let flip = p.control_flip()?;
    let deep = p.control_deep_rewind();
    println!("  controls: flipped ring bit caught {flip}, over-deep rewind refused {deep}");
    Ok(flip && deep)
}

fn main() -> Result<()> {
    // SAFETY: set before any thread exists and before the lever is first read (the kernel
    // resolve below reads it once).
    unsafe { std::env::set_var("METRALE_GLM_DSA_POOL_CACHE", "1") };
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let all_present = |k: &Glm5NextDsaKernels| {
        let pool = [k.kpool_compress_incr, k.write_geom_pk, k.indexer_store_ring];
        [k.write_geom, k.pk_len_clamp]
            .iter()
            .chain(&pool)
            .all(|h| h.0 != 0)
    };
    let kernels = match Glm5NextDsaKernels::resolve(g) {
        Ok(k) if all_present(&k) => k,
        r => {
            let why = r
                .err()
                .map_or("a pool-cache entry point".into(), |e| e.to_string());
            println!("a DSA kernel is absent from this target ({why}) - SKIP");
            std::process::exit(2);
        }
    };
    let (mut compared, mut differed, mut controls) = (0, 0, true);
    let mut first = None;
    for (i, &s_max) in TARGETS.iter().enumerate() {
        println!("target S = {s_max}");
        let mut p = Pair::new(g, kernels, 0x9A11_C0DE + i as u64)?;
        controls &= run(&mut p, s_max)?;
        compared += p.compared;
        differed += p.differed;
        first = first.or(p.first_diff.take());
        p.free()?;
    }
    if differed == 0 && controls && compared > 0 {
        println!(
            "PASS dsa_pool_cache_parity: {compared} selections byte-equal (pools and tokens), \
             controls fired"
        );
        Ok(())
    } else {
        println!(
            "FAIL dsa_pool_cache_parity: {differed} of {compared} selections differ (first: \
             {}), controls fired {controls}",
            first.unwrap_or_else(|| "none".into())
        );
        std::process::exit(1);
    }
}
