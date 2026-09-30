// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Byte-identity gate for the `METRALE_GLM_MOE_PREFILL_PERMUTE=1` lever
//! (`forward_prefill_gemm::prefill_gemm_permute`): gathering `x` into expert-sorted order once
//! through `moe_permute_tokens`, then running the gate/up grouped GEMM with `sorted_token_ids`
//! NULL, must produce the exact same bytes as the production path, which gathers each GEMM's own
//! A rows through `sorted_token_ids` directly.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants:
//! - For every row count in `ROW_COUNTS`, the "gathered" arm (lever off: A = raw token order, `sorted_token_ids`
//!   from the sort) and the "permuted" arm (lever on: A = `moe_permute_tokens` output, `sorted_token_ids`
//!   NULL) write byte-identical `C`. Any difference is fatal.
//! - `top_k = 1` here, so a slot is a row and `sorted_token_ids[pos]` is the row `moe_sort_by_expert`
//!   placed at sorted position `pos` — the same shape `forward_moe_grouped_prefill` runs the permute
//!   lever over, with `total_expanded == rows`.
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_moe_prefill_permute_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-30: The production default tile (`DEFAULT_GEMM_TILE` in
/// `glm5next_mlp/forward_prefill_gemm/tile.rs`), the one the permute lever actually feeds in
/// production.
const REF_TILE: &str = "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128";
const REF_M_TILE: u32 = 16;
const REF_N_TILE: u32 = 64;
const REF_THREADS: u32 = 128;

const NUM_EXPERTS: usize = 16;
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

#[allow(clippy::too_many_arguments)]
fn launch_ref_tile(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2: DevicePtr,
    c: DevicePtr,
    off: DevicePtr,
    stid: DevicePtr,
    m_tiles: u32,
) -> Result<()> {
    let n_tiles = (N as u32).div_ceil(REF_N_TILE);
    KernelLaunch::new(gpu, k)
        .grid([n_tiles, m_tiles, NUM_EXPERTS as u32])
        .block([REF_THREADS, 1, 1])
        .arg_ptr(a)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2)
        .arg_ptr(c)
        .arg_ptr(off)
        .arg_ptr(stid)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(N as u32)
        .arg_u32(K as u32)
        .launch(0)
}

fn main() -> Result<()> {
    // 2026-09-30: Same module-selection dance as `glm5next_moe_grouped_prefill_microtest`: a
    // build of every gb10 model resolves `ptx_modules()` to deepseek-v4-flash's set, whose
    // `moe_w4a16_grouped_gemm.cu` lacks the `bt_m16_k128` tile variant this test needs.
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let g = MetraleCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &g;

    // 2026-09-30: Module names from the gb10 common `KERNEL.toml` `[modules]` table:
    // `moe_permute = "moe"` (both `moe_sort_by_expert` and `moe_permute_tokens` live there),
    // `moe_w4a16_grouped_gemm = "moe_w4a16"`.
    let k_sort: KernelHandle = gpu.kernel("moe", "moe_sort_by_expert")?;
    let k_permute: KernelHandle = gpu.kernel("moe", "moe_permute_tokens")?;
    let k_gemm: KernelHandle = gpu.kernel("moe_w4a16", REF_TILE)?;

    // 2026-09-30: Experts are fixed across every row count; only the routing and A change.
    let mut s = 0x5EED_5A17_u64;
    let experts: Vec<(Vec<u8>, Vec<u8>, f32)> =
        (0..NUM_EXPERTS).map(|_| make_expert(&mut s)).collect();
    let packed: Vec<DevicePtr> = experts
        .iter()
        .map(|(p, _, _)| up(gpu, p))
        .collect::<Result<_>>()?;
    let scales: Vec<DevicePtr> = experts
        .iter()
        .map(|(_, sc, _)| up(gpu, sc))
        .collect::<Result<_>>()?;
    let d_packed_ptrs = up_u64(gpu, &packed.iter().map(|p| p.0).collect::<Vec<_>>())?;
    let d_scale_ptrs = up_u64(gpu, &scales.iter().map(|p| p.0).collect::<Vec<_>>())?;
    let d_scale2 = up_f32(
        gpu,
        &experts.iter().map(|(_, _, s2)| *s2).collect::<Vec<_>>(),
    )?;

    let mut failures = 0usize;

    for &rows in ROW_COUNTS {
        // 2026-09-30: `top_k = 1`: a slot is a row, so `total_expanded == rows`, the same shape
        // `forward_moe_grouped_prefill` runs the permute lever over.
        let te = rows;
        let ids: Vec<u32> = (0..rows)
            .map(|_| (lcg(&mut s) as usize % NUM_EXPERTS) as u32)
            .collect();
        let a: Vec<f32> = (0..rows * K)
            .map(|_| (lcg(&mut s) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let d_a = up_bf16(gpu, &a)?;
        let d_ids = up_i32(gpu, &ids.iter().map(|x| *x as i32).collect::<Vec<_>>())?;

        let d_stid = gpu.alloc(te * 4)?;
        let d_seid = gpu.alloc(te * 4)?;
        let d_off = gpu.alloc((NUM_EXPERTS + 1) * 4)?;
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
            .arg_u32(NUM_EXPERTS as u32)
            .arg_u32(1u32) // top_k
            .launch(0)?;
        gpu.synchronize(0)?;

        let off = dn_i32(gpu, d_off, NUM_EXPERTS + 1)?;
        if off[0] != 0 || off[NUM_EXPERTS] != te as i32 {
            bail!(
                "rows={rows}: expert_offsets does not span [0, {te}): {} .. {}",
                off[0],
                off[NUM_EXPERTS]
            );
        }
        let busiest = (0..NUM_EXPERTS)
            .map(|e| off[e + 1] - off[e])
            .max()
            .unwrap_or(0)
            .max(0) as u32;
        let m_tiles = busiest.div_ceil(REF_M_TILE).max(1);

        // 2026-09-30: Arm A, lever OFF: the GEMM gathers A through `sorted_token_ids` — today's
        // production path.
        let c_bytes = te * N * 2;
        let d_c_gathered = gpu.alloc(c_bytes)?;
        gpu.memset_async(d_c_gathered, 0xFF, c_bytes, 0)?;
        launch_ref_tile(
            gpu,
            k_gemm,
            d_a,
            d_packed_ptrs,
            d_scale_ptrs,
            d_scale2,
            d_c_gathered,
            d_off,
            d_stid,
            m_tiles,
        )?;
        gpu.synchronize(0)?;
        let mut want = vec![0u8; c_bytes];
        gpu.copy_d2h(d_c_gathered, &mut want)?;

        // 2026-09-30: Arm B, lever ON: `moe_permute_tokens` gathers x into expert-sorted order
        // once, then the same GEMM runs with `sorted_token_ids` NULL
        // (`forward_prefill_gemm::prefill_gemm_permute`'s exact sequence).
        let d_perm = gpu.alloc(te * K * 2)?;
        gpu.memset_async(d_perm, 0xAA, te * K * 2, 0)?;
        KernelLaunch::new(gpu, k_permute)
            .grid([te as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_a)
            .arg_ptr(d_perm)
            .arg_ptr(d_stid)
            .arg_u32(K as u32)
            .arg_u32(te as u32)
            .launch(0)?;
        let d_c_permuted = gpu.alloc(c_bytes)?;
        gpu.memset_async(d_c_permuted, 0xFF, c_bytes, 0)?;
        launch_ref_tile(
            gpu,
            k_gemm,
            d_perm,
            d_packed_ptrs,
            d_scale_ptrs,
            d_scale2,
            d_c_permuted,
            d_off,
            DevicePtr(0),
            m_tiles,
        )?;
        gpu.synchronize(0)?;
        let mut got = vec![0u8; c_bytes];
        gpu.copy_d2h(d_c_permuted, &mut got)?;

        let diffs = want.iter().zip(&got).filter(|(x, y)| x != y).count();
        match want.iter().zip(&got).position(|(x, y)| x != y) {
            None => println!(
                "  rows={rows:5} busiest={busiest:5} m_tiles={m_tiles:3}  BYTE-IDENTICAL ({c_bytes} bytes)"
            ),
            Some(i) => {
                failures += 1;
                let elem = i / 2;
                println!(
                    "  rows={rows:5} busiest={busiest:5} m_tiles={m_tiles:3}  🔴 {diffs} bytes differ; \
                     first at sorted row {}, n {}: gathered {:02x?} permuted {:02x?}",
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

    for p in packed
        .into_iter()
        .chain(scales)
        .chain([d_packed_ptrs, d_scale_ptrs, d_scale2])
    {
        gpu.free(p)?;
    }

    if failures > 0 {
        bail!(
            "{failures}/{} row count(s) are NOT byte-identical between the gathered and permuted \
             arms — METRALE_GLM_MOE_PREFILL_PERMUTE=1 is not byte-identical by construction",
            ROW_COUNTS.len()
        );
    }
    println!(
        "PASS — gate/up through moe_permute_tokens + null sorted_token_ids is byte-identical to \
         gathering through sorted_token_ids directly, at every row count in {ROW_COUNTS:?}."
    );
    Ok(())
}
