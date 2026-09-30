// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Byte-identity gate for the `METRALE_GLM_MOE_PREFILL_PERMUTE=1` lever
//! (`forward_prefill_gemm::prefill_gemm_permute`): gathering `x` into expert-sorted order once
//! through `moe_permute_tokens`, then running the gate/up grouped GEMM with `sorted_token_ids`
//! NULL, must produce the exact same bytes as the production path, which gathers each GEMM's own
//! A rows through `sorted_token_ids` directly.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants:
//! - For every `(tile, top_k, rows)` in `TILES x TOPK_CFGS x ROW_COUNTS`, the "gathered" arm
//!   (lever off: A = raw token order, `sorted_token_ids` from the sort) and the "permuted" arm
//!   (lever on: A = `moe_permute_tokens` output, `sorted_token_ids` NULL) write byte-identical
//!   `C`. Any difference is fatal.
//! - `top_k = 1`: a slot is a row and `sorted_token_ids[pos]` is the row `moe_sort_by_expert`
//!   placed at sorted position `pos`, `total_expanded == rows` (16 experts, all local — the
//!   original single-tile shape this gate covered).
//! - `top_k = 8`: the production/bench routing shape — each row picks 8 distinct experts out of
//!   288, `total_expanded == rows * 8`, and `sorted_token_ids[pos]` is the TOKEN ROW
//!   `i / top_k` that landed at sorted position `pos` (`forward_prefill_gemm::tests::sort_ref`'s
//!   model of `moe_sort_by_expert`). Half the experts (`e % EP != 0`, `EP = 2`) carry a NULL
//!   weight pointer, exactly as `glm5next_moe_grouped_tile_bench` builds its production-shape
//!   routing and pointer table, so a CTA whose expert is remote early-exits without writing —
//!   in both arms, from the same pointer table.
//! - `moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128` is the production default
//!   (`DEFAULT_GEMM_TILE` in `forward_prefill_gemm/tile.rs`); an unresolved kernel for it is
//!   skipped with a message, not fatal. `moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64` is the
//!   tile production selects via `METRALE_GLM_MOE_GEMM_TILE=bt_m128_k64`; an unresolved kernel
//!   for it fails the run.
//! - The permute arm calls the exact wrapper `forward_prefill_gemm::dispatch` calls,
//!   `metrale_model_layers::layers::ops::moe_permute_tokens`, with `total_expanded` rows.
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_moe_prefill_permute_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops::moe_permute_tokens;

/// 2026-09-30: One grouped-GEMM tile entry point under test, and the launch geometry it needs —
/// the same fields `forward_prefill_gemm::tile::GemmTile` carries for these two (both
/// `m_fast: false`, so the grid is always `[n_tiles, m_tiles, num_experts]`,
/// `GemmTile::grid_dims`'s non-`_mfast` branch).
struct MicroTile {
    /// Kernel entry point in the `moe_w4a16` module.
    name: &'static str,
    /// Rows per CTA (`GemmTile::m_tile`); the grid height is counted in it.
    m_tile: usize,
    /// Output columns per CTA; grid.x is `ceil(N / n_tile)`.
    n_tile: u32,
    /// Threads per block.
    threads: u32,
    /// An unresolved kernel fails the run instead of being skipped with a message.
    required: bool,
}

/// 2026-09-30: `forward_prefill_gemm::tile::DEFAULT_GEMM_TILE` (unset `METRALE_GLM_MOE_GEMM_TILE`)
/// and the tile production selects with `METRALE_GLM_MOE_GEMM_TILE=bt_m128_k64` — field values
/// from `GEMM_TILES` in `forward_prefill_gemm/tile.rs`.
const TILES: &[MicroTile] = &[
    MicroTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128",
        m_tile: 16,
        n_tile: 64,
        threads: 128,
        required: false,
    },
    MicroTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64",
        m_tile: 128,
        n_tile: 64,
        threads: 256,
        required: true,
    },
];

/// 2026-09-30: One `top_k` shape this gate covers: the routing width, the expert count routing
/// draws from, and the EP divisor that decides which experts are local (`e % EP == 0`). `EP = 1`
/// makes every expert local (no NULL pointers) — the original `top_k = 1` shape.
struct TopKCfg {
    top_k: usize,
    num_experts: usize,
    ep: usize,
}

const TOPK_CFGS: &[TopKCfg] = &[
    // 2026-09-30: The gate's original shape: 16 experts, all local, one slot per row.
    TopKCfg {
        top_k: 1,
        num_experts: 16,
        ep: 1,
    },
    // 2026-09-30: The production/bench shape (`glm5next_moe_grouped_tile_bench`): 288 experts,
    // top_k = 8, half local.
    TopKCfg {
        top_k: 8,
        num_experts: 288,
        ep: 2,
    },
];

/// 2026-09-30: One expert's output width: four `N_TILE = 64` tiles.
const N: usize = 256;
/// 2026-09-30: `K/16 = 32` scale groups and `K/2 = 256` packed bytes per output row.
const K: usize = 512;

/// 2026-09-30: `rows` values the lever must be byte-identical at, per the task packet: 1 (a
/// single sorted row), 17 (odd, spans two 16-row tiles), 256 and 1000 (mid-size, uneven expert
/// split), 2048 (the production staged window) and 4096 (the widest window the moeprobe RESULT
/// measured the embedded gather's cost at).
const ROW_COUNTS: &[usize] = &[1, 17, 256, 1000, 2048, 4096];

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn up_u64(g: &dyn GpuBackend, v: &[u64]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}

fn up_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}

fn up_i32(g: &dyn GpuBackend, v: &[i32]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}

fn up_bf16(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter()
            .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn dn_i32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<i32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// 2026-09-30: One expert's NVFP4 weight, the same layout
/// `glm5next_moe_grouped_prefill_microtest::make_expert` uses: packed `[N, K/2]` (even `k` in the
/// low nibble), E4M3 block scales `[N, K/16]`, one `f32` `scale2`.
fn make_expert(s: &mut u64) -> (Vec<u8>, Vec<u8>, f32) {
    let mut packed = vec![0u8; N * K / 2];
    for b in packed.iter_mut() {
        *b = lcg(s) as u8;
    }
    let mut scale = vec![0u8; N * K / 16];
    for b in scale.iter_mut() {
        // 2026-09-30: E4M3 codes 0x30..=0x47 decode to 0.5..=3.75: no zero, no NaN.
        *b = 0x30 + (lcg(s) % 0x18) as u8;
    }
    (packed, scale, 0.5 + (lcg(s) % 64) as f32 / 64.0)
}

/// 2026-09-30: A pointer table over `num_experts` experts, built with `build_experts`: the
/// device buffers this run owns (only the local ones), and the three tables the GEMM reads.
struct ExpertsSet {
    owned: Vec<DevicePtr>,
    d_packed_ptrs: DevicePtr,
    d_scale_ptrs: DevicePtr,
    d_scale2: DevicePtr,
}

impl ExpertsSet {
    fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in
            self.owned
                .into_iter()
                .chain([self.d_packed_ptrs, self.d_scale_ptrs, self.d_scale2])
        {
            gpu.free(p)?;
        }
        Ok(())
    }
}

/// 2026-09-30: Expert `e` is local when `e % ep == 0`; every other expert keeps a NULL
/// (`DevicePtr(0)`) weight pointer and a zero `scale2`, exactly as
/// `glm5next_moe_grouped_tile_bench::main` builds its pointer table (`e % EP != 0` skips
/// allocation). `ep = 1` makes every expert local.
fn build_experts(
    gpu: &dyn GpuBackend,
    s: &mut u64,
    num_experts: usize,
    ep: usize,
) -> Result<ExpertsSet> {
    let mut owned = Vec::new();
    let mut packed_ptrs = vec![0u64; num_experts];
    let mut scale_ptrs = vec![0u64; num_experts];
    let mut scale2 = vec![0f32; num_experts];
    for e in 0..num_experts {
        if e % ep != 0 {
            continue;
        }
        let (packed, scale, s2) = make_expert(s);
        let dp = up(gpu, &packed)?;
        let ds = up(gpu, &scale)?;
        packed_ptrs[e] = dp.0;
        scale_ptrs[e] = ds.0;
        scale2[e] = s2;
        owned.push(dp);
        owned.push(ds);
    }
    Ok(ExpertsSet {
        owned,
        d_packed_ptrs: up_u64(gpu, &packed_ptrs)?,
        d_scale_ptrs: up_u64(gpu, &scale_ptrs)?,
        d_scale2: up_f32(gpu, &scale2)?,
    })
}

/// 2026-09-30: A row's `top_k` ids are distinct, as `glm5next_router_topk` produces them — the
/// same draw-until-distinct loop `glm5next_moe_grouped_tile_bench::main` uses (`main.rs:~255-265`)
/// generalised to any `top_k`.
fn make_routing(s: &mut u64, rows: usize, top_k: usize, num_experts: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(rows * top_k);
    for _ in 0..rows {
        let mut picked: Vec<u32> = Vec::with_capacity(top_k);
        while picked.len() < top_k {
            let e = (lcg(s) as usize % num_experts) as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }
    ids
}

#[allow(clippy::too_many_arguments)]
fn launch_tile(
    gpu: &dyn GpuBackend,
    tile: &MicroTile,
    k: KernelHandle,
    a: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2: DevicePtr,
    c: DevicePtr,
    off: DevicePtr,
    stid: DevicePtr,
    num_experts: usize,
    m_tiles: u32,
) -> Result<()> {
    let n_tiles = (N as u32).div_ceil(tile.n_tile);
    KernelLaunch::new(gpu, k)
        .grid([n_tiles, m_tiles, num_experts as u32])
        .block([tile.threads, 1, 1])
        .arg_ptr(a)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2)
        .arg_ptr(c)
        .arg_ptr(off)
        .arg_ptr(stid)
        .arg_u32(num_experts as u32)
        .arg_u32(N as u32)
        .arg_u32(K as u32)
        .launch(0)
}

fn main() -> Result<()> {
    // 2026-09-30: Same module-selection dance as `glm5next_moe_grouped_prefill_microtest`: a
    // build of every gb10 model resolves `ptx_modules()` to deepseek-v4-flash's set, whose
    // `moe_w4a16_grouped_gemm.cu` lacks the tile variants this test needs.
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let g = MetraleCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &g;

    // 2026-09-30: Module names from the gb10 common `KERNEL.toml` `[modules]` table:
    // `moe_permute = "moe"` (both `moe_sort_by_expert` and `moe_permute_tokens` live there).
    let k_sort: KernelHandle = gpu.kernel("moe", "moe_sort_by_expert")?;
    let k_permute: KernelHandle = gpu.kernel("moe", "moe_permute_tokens")?;

    let mut s = 0x5EED_5A17_u64;

    // 2026-09-30: One pointer table per `top_k` shape, shared across tiles and row counts —
    // experts and locality do not depend on either.
    let mut expert_sets: Vec<ExpertsSet> = Vec::with_capacity(TOPK_CFGS.len());
    for cfg in TOPK_CFGS {
        expert_sets.push(build_experts(gpu, &mut s, cfg.num_experts, cfg.ep)?);
    }

    let mut failures = 0usize;
    let mut tested = 0usize;

    for tile in TILES {
        let k_gemm: KernelHandle = match gpu.kernel("moe_w4a16", tile.name) {
            Ok(k) => k,
            Err(e) => {
                if tile.required {
                    bail!("tile `{}` did not resolve: {e}", tile.name);
                }
                println!("tile `{}`: UNRESOLVED, skipping ({e})", tile.name);
                continue;
            }
        };

        for (cfg, experts) in TOPK_CFGS.iter().zip(&expert_sets) {
            for &rows in ROW_COUNTS {
                let te = rows * cfg.top_k;
                let ids = make_routing(&mut s, rows, cfg.top_k, cfg.num_experts);
                let a: Vec<f32> = (0..rows * K)
                    .map(|_| (lcg(&mut s) % 2001) as f32 / 1000.0 - 1.0)
                    .collect();
                let d_a = up_bf16(gpu, &a)?;
                let d_ids = up_i32(gpu, &ids.iter().map(|x| *x as i32).collect::<Vec<_>>())?;

                let d_stid = gpu.alloc(te * 4)?;
                let d_seid = gpu.alloc(te * 4)?;
                let d_off = gpu.alloc((cfg.num_experts + 1) * 4)?;
                let d_t2p = gpu.alloc(te * 4)?;
                KernelLaunch::new(gpu, k_sort)
                    .grid([1, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_ids)
                    .arg_ptr(d_stid)
                    .arg_ptr(d_seid)
                    .arg_ptr(d_off)
                    .arg_ptr(d_t2p)
                    .arg_u32(te as u32)
                    .arg_u32(cfg.num_experts as u32)
                    .arg_u32(cfg.top_k as u32)
                    .launch(0)?;
                gpu.synchronize(0)?;

                let off = dn_i32(gpu, d_off, cfg.num_experts + 1)?;
                if off[0] != 0 || off[cfg.num_experts] != te as i32 {
                    bail!(
                        "tile={} top_k={} rows={rows}: expert_offsets does not span [0, {te}): {} .. {}",
                        tile.name,
                        cfg.top_k,
                        off[0],
                        off[cfg.num_experts]
                    );
                }
                let busiest = (0..cfg.num_experts)
                    .map(|e| off[e + 1] - off[e])
                    .max()
                    .unwrap_or(0)
                    .max(0) as u32;
                let m_tiles = busiest.div_ceil(tile.m_tile as u32).max(1);

                // 2026-09-30: Arm A, lever OFF: the GEMM gathers A through `sorted_token_ids` —
                // today's production path.
                let c_bytes = te * N * 2;
                let d_c_gathered = gpu.alloc(c_bytes)?;
                gpu.memset_async(d_c_gathered, 0xFF, c_bytes, 0)?;
                launch_tile(
                    gpu,
                    tile,
                    k_gemm,
                    d_a,
                    experts.d_packed_ptrs,
                    experts.d_scale_ptrs,
                    experts.d_scale2,
                    d_c_gathered,
                    d_off,
                    d_stid,
                    cfg.num_experts,
                    m_tiles,
                )?;
                gpu.synchronize(0)?;
                let mut want = vec![0u8; c_bytes];
                gpu.copy_d2h(d_c_gathered, &mut want)?;

                // 2026-09-30: Arm B, lever ON: `moe_permute_tokens` gathers x into expert-sorted
                // order once, through the exact wrapper `forward_prefill_gemm::dispatch` calls,
                // then the same GEMM runs with `sorted_token_ids` NULL.
                let d_perm = gpu.alloc(te * K * 2)?;
                gpu.memset_async(d_perm, 0xAA, te * K * 2, 0)?;
                moe_permute_tokens(gpu, k_permute, d_a, d_perm, d_stid, K as u32, te as u32, 0)?;
                let d_c_permuted = gpu.alloc(c_bytes)?;
                gpu.memset_async(d_c_permuted, 0xFF, c_bytes, 0)?;
                launch_tile(
                    gpu,
                    tile,
                    k_gemm,
                    d_perm,
                    experts.d_packed_ptrs,
                    experts.d_scale_ptrs,
                    experts.d_scale2,
                    d_c_permuted,
                    d_off,
                    DevicePtr(0),
                    cfg.num_experts,
                    m_tiles,
                )?;
                gpu.synchronize(0)?;
                let mut got = vec![0u8; c_bytes];
                gpu.copy_d2h(d_c_permuted, &mut got)?;

                tested += 1;
                let diffs = want.iter().zip(&got).filter(|(x, y)| x != y).count();
                match want.iter().zip(&got).position(|(x, y)| x != y) {
                    None => println!(
                        "  tile={:<45} top_k={:<2} rows={rows:5} busiest={busiest:5} m_tiles={m_tiles:3}  IDENTICAL ({c_bytes} bytes)",
                        tile.name, cfg.top_k
                    ),
                    Some(i) => {
                        failures += 1;
                        let elem = i / 2;
                        println!(
                            "  tile={:<45} top_k={:<2} rows={rows:5} busiest={busiest:5} m_tiles={m_tiles:3}  🔴 first-mismatch: \
                             {diffs} bytes differ; first at sorted row {}, n {}: gathered {:02x?} permuted {:02x?}",
                            tile.name,
                            cfg.top_k,
                            elem / N,
                            elem % N,
                            &want[i & !1..(i & !1) + 2],
                            &got[i & !1..(i & !1) + 2]
                        );
                    }
                }

                for p in [
                    d_a,
                    d_ids,
                    d_stid,
                    d_seid,
                    d_off,
                    d_t2p,
                    d_c_gathered,
                    d_perm,
                    d_c_permuted,
                ] {
                    gpu.free(p)?;
                }
            }
        }
    }

    for set in expert_sets {
        set.free(gpu)?;
    }

    if tested == 0 {
        bail!("no tile resolved — nothing was tested");
    }
    if failures > 0 {
        bail!(
            "{failures}/{tested} (tile, top_k, rows) combination(s) are NOT byte-identical between \
             the gathered and permuted arms — METRALE_GLM_MOE_PREFILL_PERMUTE=1 is not \
             byte-identical by construction"
        );
    }
    println!(
        "gate/up through moe_permute_tokens + null sorted_token_ids is byte-identical to \
         gathering through sorted_token_ids directly, at every (tile, top_k, rows) combination \
         tested ({tested} total)."
    );
    println!("PASS");
    Ok(())
}
