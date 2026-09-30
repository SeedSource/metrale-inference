// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Layout gate for the GLM routed-MoE prefill grouped GEMM: it must read the NVFP4
//! experts the way the routed GEMV does.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants:
//! - The run fails if the device sort breaks its contract (offsets, a slot mapped twice or out
//!   of range, a sorted row carrying the wrong token or expert).
//! - It fails if the GEMV or the grouped GEMM peak error exceeds `BAR` of the reference's peak
//!   magnitude, or if the GEMM is further from the BF16-weight reference than from the exact
//!   one.
//!
//! `moe_w4a16_grouped_gemm_ptrtable` (tensor core, `mma.sync.aligned.m16n8k16`) and
//! `w4a16_gemv_sw_moe_batchm_m8` (software dequant) read the same NVFP4 bytes: packed
//! `[N, K/2]` with even `k` in the low nibble, E4M3 block scales per 16 values, and a
//! per-expert `scale2`. One set of random experts runs through both kernels over the same
//! routing, and each is scored against an FP32 host reference that dequantises the same bytes.
//!
//! Byte equality is not the bar. The GEMV sums two interleaved FP32 `fmaf` chains per lane and
//! combines them with a warp shuffle tree; the GEMM accumulates `K_STEP = 16` `mma.sync` steps.
//! Same operands, different association. `forward_moe` takes the grouped GEMM only when
//! `rows > MOE_ROW_BATCH_MAX_ROWS`, among other conditions.
//!
//! The sort is checked separately from the arithmetic: a sort that dropped or duplicated a slot
//! could still produce a plausible GEMM output.
//!
//! 2026-09-29: Tile identity (whole-chunk prefill M1). After the layout gate, every M1 tile
//! (`M1_TILES`) runs on the same routing, weights and inputs as `bt_m16_k128`, the production
//! default, and must write byte-identical output: skewed routing (one expert over 2 x 128
//! rows, one with a handful, a partial last M tile everywhere), one remote expert with a NULL
//! weight pointer, the gathered (gate/up) and direct (down) A paths, at the small shape and at
//! the two production shapes. The output buffer starts as 0xFF so an unwritten element shows.
//! Any mismatch, or an M1 tile or the reference missing from the PTX, exits nonzero.
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_moe_grouped_prefill_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use half::bf16;
use metrale_cache::kv_dequant::{NVFP4_E2M1_LUT, NVFP4_GROUP_SIZE, e4m3_lut};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-25: 64 tokens at `top_k = 8` give 512 sorted rows over 16 experts, about 32 per
/// expert, so each expert's `M_TILE = 64` tile is partly filled.
const M: usize = 64;
const TOP_K: usize = 8;
const NUM_EXPERTS: usize = 16;
/// 2026-09-25: One expert's output width: four `N_TILE = 64` tiles.
const N: usize = 256;
/// 2026-09-25: Input width: `K/16 = 32` scale groups and `K/2 = 256` packed bytes per output
/// row.
const K: usize = 512;
/// 2026-09-25: Equal to `glm5next_mlp::forward::MOE_ROW_BATCH_MAX_ROWS`; the GEMV arm runs the
/// `_m8` kernel over row groups of this size.
const GEMV_TIER: usize = 8;

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

fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

fn dn_i32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<i32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// 2026-09-25: One expert's NVFP4 weight in the layout both kernels read: packed `[N, K/2]`
/// (even `k` in the low nibble), E4M3 block scales `[N, K/16]`, one `f32` `scale2`.
struct Expert {
    packed: Vec<u8>,
    scale: Vec<u8>,
    scale2: f32,
}

fn make_expert(s: &mut u64) -> Expert {
    let mut packed = vec![0u8; N * K / 2];
    for b in packed.iter_mut() {
        *b = lcg(s) as u8;
    }
    let mut scale = vec![0u8; N * K / NVFP4_GROUP_SIZE];
    for b in scale.iter_mut() {
        // 2026-09-25: E4M3 codes 0x30..=0x47, which decode to 0.5..=3.75: no zero, no NaN.
        *b = 0x30 + (lcg(s) % 0x18) as u8;
    }
    Expert {
        packed,
        scale,
        scale2: 0.5 + (lcg(s) % 64) as f32 / 64.0,
    }
}

/// 2026-09-25: `dequant(W)[n, k]` as `lut * e4m3 * scale2`, the grouped GEMM's order.
fn w(e: &Expert, n: usize, k: usize) -> f32 {
    let byte = e.packed[n * (K / 2) + k / 2];
    let nib = if k % 2 == 1 { byte >> 4 } else { byte & 0xF };
    let sb = e.scale[n * (K / NVFP4_GROUP_SIZE) + k / NVFP4_GROUP_SIZE];
    NVFP4_E2M1_LUT[nib as usize] * e4m3_lut()[sb as usize] * e.scale2
}

/// 2026-09-25: Host counting sort. Only its `expert_offsets` are compared: the row order inside
/// one expert follows the device kernel's atomic order, so the device rows are checked against
/// the sort's contract instead.
fn sort_host(ids: &[u32]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let te = ids.len();
    let mut counts = vec![0i32; NUM_EXPERTS];
    for &e in ids {
        counts[e as usize] += 1;
    }
    let mut offsets = vec![0i32; NUM_EXPERTS + 1];
    for e in 0..NUM_EXPERTS {
        offsets[e + 1] = offsets[e] + counts[e];
    }
    let mut cur: Vec<i32> = offsets[..NUM_EXPERTS].to_vec();
    let mut stid = vec![-1i32; te];
    let mut t2p = vec![-1i32; te];
    for (i, &e) in ids.iter().enumerate() {
        let p = cur[e as usize];
        cur[e as usize] += 1;
        stid[p as usize] = (i / TOP_K) as i32;
        t2p[i] = p;
    }
    (stid, offsets, t2p)
}

struct Err2 {
    abs: f32,
    rel: f32,
}

/// 2026-09-25: Peak absolute error, and peak relative error with the denominator floored at 1%
/// of the output scale. The outputs are `K = 512` dot products of signed terms, so some land
/// near zero by cancellation, and an unfloored ratio would turn a rounding error into a large
/// "relative error".
fn score(got: &[f32], want: &[f32], scale: f32) -> Err2 {
    let floor = 0.01 * scale;
    let mut abs = 0.0f32;
    let mut rel = 0.0f32;
    for (g, r) in got.iter().zip(want) {
        let a = (g - r).abs();
        abs = abs.max(a);
        rel = rel.max(a / r.abs().max(floor));
    }
    Err2 { abs, rel }
}

fn main() -> Result<()> {
    // 2026-09-29: The (glm-5.3-flash, nvfp4) set when the build has it, as the tile bench does:
    // in a build of every gb10 model, `ptx_modules()` is deepseek-v4-flash's set, whose own
    // `moe_w4a16_grouped_gemm.cu` lacks the tile variants the M1 identity check needs.
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let g = MetraleCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &g;
    // 2026-09-25: Module names come from the `[modules]` table of the gb10 common KERNEL.toml
    // (`moe_permute = "moe"`, `moe_w4a16_grouped_gemm = "moe_w4a16"`); `w4a16_gemv` has no
    // entry and keeps its file stem.
    let k_sort: KernelHandle = gpu.kernel("moe", "moe_sort_by_expert")?;
    let k_gemm: KernelHandle = gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")?;
    let k_union: KernelHandle = gpu.kernel("w4a16_gemv", "glm5next_moe_row_union")?;
    let k_batchm: KernelHandle = gpu.kernel("w4a16_gemv", "w4a16_gemv_sw_moe_batchm_m8")?;

    let mut s = 0x5EED_1234_u64;

    let mut ids: Vec<u32> = Vec::with_capacity(M * TOP_K);
    for _ in 0..M {
        let mut picked: Vec<u32> = Vec::with_capacity(TOP_K);
        while picked.len() < TOP_K {
            let e = (lcg(&mut s) as usize % NUM_EXPERTS) as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }
    let te = M * TOP_K;

    let experts: Vec<Expert> = (0..NUM_EXPERTS).map(|_| make_expert(&mut s)).collect();
    let a_host: Vec<f32> = (0..M * K)
        .map(|_| (lcg(&mut s) % 2001) as f32 / 1000.0 - 1.0)
        .collect();
    // 2026-09-25: The kernels read BF16 activations, so the reference reads the same rounded
    // values; otherwise the input's rounding would count as kernel error.
    let a_bf: Vec<f32> = a_host.iter().map(|x| bf16::from_f32(*x).to_f32()).collect();

    let d_a = up_bf16(gpu, &a_host)?;
    let packed: Vec<DevicePtr> = experts
        .iter()
        .map(|e| up(gpu, &e.packed))
        .collect::<Result<_>>()?;
    let scales: Vec<DevicePtr> = experts
        .iter()
        .map(|e| up(gpu, &e.scale))
        .collect::<Result<_>>()?;
    let d_packed_ptrs = up_u64(gpu, &packed.iter().map(|p| p.0).collect::<Vec<_>>())?;
    let d_scale_ptrs = up_u64(gpu, &scales.iter().map(|p| p.0).collect::<Vec<_>>())?;
    let d_scale2 = up_f32(gpu, &experts.iter().map(|e| e.scale2).collect::<Vec<_>>())?;

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
        .arg_u32(TOP_K as u32)
        .launch(0)?;
    gpu.synchronize(0)?;

    let stid = dn_i32(gpu, d_stid, te)?;
    let seid = dn_i32(gpu, d_seid, te)?;
    let off = dn_i32(gpu, d_off, NUM_EXPERTS + 1)?;
    let t2p = dn_i32(gpu, d_t2p, te)?;

    let (_, off_ref, _) = sort_host(&ids);
    if off != off_ref {
        bail!("expert_offsets disagree with the host counting sort:\n{off:?}\n{off_ref:?}");
    }
    let mut seen = vec![false; te];
    for (i, &p) in t2p.iter().enumerate() {
        if p < 0 || p as usize >= te || seen[p as usize] {
            bail!("token_to_perm[{i}] = {p} is out of range or a duplicate");
        }
        seen[p as usize] = true;
        if stid[p as usize] != (i / TOP_K) as i32 {
            bail!(
                "slot {i} maps to sorted row {p}, which carries token {}",
                stid[p as usize]
            );
        }
        if seid[p as usize] != ids[i] as i32 {
            bail!(
                "slot {i} maps to sorted row {p}, which carries expert {}",
                seid[p as usize]
            );
        }
    }
    let busiest = (0..NUM_EXPERTS)
        .map(|e| off[e + 1] - off[e])
        .max()
        .unwrap_or(0);
    println!(
        "sort OK: {te} slots, {NUM_EXPERTS} experts, busiest expert {busiest} rows, \
         max_m_tiles {}",
        (busiest as u32).div_ceil(64).max(1)
    );
    let max_m_tiles = (busiest as u32).div_ceil(64).max(1);

    // 2026-09-25: Two host references per sorted row. `want_sorted` uses the exact FP32
    // dequantised weight; `want_bf16w` rounds that weight to BF16 first.
    // `moe_w4a16_grouped_gemm_ptrtable` stages its B tile as
    // `__float2bfloat16(E2M1 * fp8 * scale2)`, because `mma.sync ... .bf16.bf16.f32` takes
    // BF16 operands, so it carries a weight rounding the GEMV does not. The second reference
    // separates that operand precision from a layout error.
    let mut want_sorted = vec![0.0f32; te * N];
    let mut want_bf16w = vec![0.0f32; te * N];
    for p in 0..te {
        let tok = stid[p] as usize;
        let e = &experts[seid[p] as usize];
        for n in 0..N {
            let mut acc = 0.0f64;
            let mut acc_b = 0.0f64;
            for kk in 0..K {
                let a = a_bf[tok * K + kk];
                let wv = w(e, n, kk);
                acc += (a * wv) as f64;
                acc_b += (a * bf16::from_f32(wv).to_f32()) as f64;
            }
            want_sorted[p * N + n] = acc as f32;
            want_bf16w[p * N + n] = acc_b as f32;
        }
    }

    let d_c_gemm = gpu.alloc(te * N * 2)?;
    gpu.memset_async(d_c_gemm, 0, te * N * 2, 0)?;
    KernelLaunch::new(gpu, k_gemm)
        .grid([(N as u32).div_ceil(64), max_m_tiles, NUM_EXPERTS as u32])
        .block([128, 1, 1])
        .arg_ptr(d_a)
        .arg_ptr(d_packed_ptrs)
        .arg_ptr(d_scale_ptrs)
        .arg_ptr(d_scale2)
        .arg_ptr(d_c_gemm)
        .arg_ptr(d_off)
        .arg_ptr(d_stid)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(N as u32)
        .arg_u32(K as u32)
        .launch(0)?;
    gpu.synchronize(0)?;
    let got_gemm = dn_bf16(gpu, d_c_gemm, te * N)?;

    let d_c_gemv = gpu.alloc(te * N * 2)?;
    gpu.memset_async(d_c_gemv, 0, te * N * 2, 0)?;
    let d_ueid = gpu.alloc(GEMV_TIER * TOP_K * 4)?;
    let d_uslot = gpu.alloc(GEMV_TIER * TOP_K * GEMV_TIER * 4)?;
    for r0 in (0..M).step_by(GEMV_TIER) {
        KernelLaunch::new(gpu, k_union)
            .grid([1, 1, 1])
            .block([(GEMV_TIER * TOP_K) as u32, 1, 1])
            .arg_ptr(d_ids.offset(r0 * TOP_K * 4))
            .arg_ptr(d_ueid)
            .arg_ptr(d_uslot)
            .arg_u32(GEMV_TIER as u32)
            .arg_u32(TOP_K as u32)
            .launch(0)?;
        KernelLaunch::new(gpu, k_batchm)
            .grid([(N as u32).div_ceil(8), (GEMV_TIER * TOP_K) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(d_a.offset(r0 * K * 2))
            .arg_ptr(d_packed_ptrs)
            .arg_ptr(d_scale_ptrs)
            .arg_ptr(d_scale2)
            .arg_ptr(d_c_gemv.offset(r0 * TOP_K * N * 2))
            .arg_ptr(d_ueid)
            .arg_ptr(d_uslot)
            .arg_u32(N as u32)
            .arg_u32(K as u32)
            .arg_u32(NUM_EXPERTS as u32)
            .arg_u32(K as u32)
            .arg_u32(0)
            .arg_u32((TOP_K * N) as u32)
            .launch(0)?;
    }
    gpu.synchronize(0)?;
    let got_gemv_slot = dn_bf16(gpu, d_c_gemv, te * N)?;
    // 2026-09-25: Re-index the GEMV's slot-major output into sorted order so it lines up with
    // the references and the GEMM.
    let mut got_gemv = vec![0.0f32; te * N];
    for i in 0..te {
        let p = t2p[i] as usize;
        got_gemv[p * N..(p + 1) * N].copy_from_slice(&got_gemv_slot[i * N..(i + 1) * N]);
    }

    let mut worst: Vec<(f32, usize)> = (0..te * N)
        .map(|i| ((got_gemm[i] - want_sorted[i]).abs(), i))
        .collect();
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("  worst |gemm - ref| elements (sorted_row, n): ref / gemm / gemv");
    for &(_, i) in worst.iter().take(5) {
        println!(
            "    ({:4}, {:3}) expert {:2}  {:12.5} / {:12.5} / {:12.5}",
            i / N,
            i % N,
            seid[i / N],
            want_sorted[i],
            got_gemm[i],
            got_gemv[i]
        );
    }

    let scale: f32 = want_sorted.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let e_gemm = score(&got_gemm, &want_sorted, scale);
    let e_gemv = score(&got_gemv, &want_sorted, scale);
    let e_pair = score(&got_gemm, &got_gemv, scale);
    let e_gemm_b = score(&got_gemm, &want_bf16w, scale);

    println!("M={M} top_k={TOP_K} experts={NUM_EXPERTS} N={N} K={K}  |ref|max = {scale:.4}");
    println!(
        "  (max_rel floors the denominator at 1% of |ref|max = {:.3})",
        0.01 * scale
    );
    for (label, e) in [
        ("grouped GEMM vs exact FP32 ref ", &e_gemm),
        ("GEMV (production) vs same ref  ", &e_gemv),
        ("grouped GEMM vs BF16-weight ref", &e_gemm_b),
        ("grouped GEMM vs GEMV           ", &e_pair),
    ] {
        println!(
            "  {label}: max_abs {:9.6}  max_abs/scale {:9.6}  max_rel {:9.6}",
            e.abs,
            e.abs / scale,
            e.rel
        );
    }

    // 2026-09-25: The bar is on the peak error divided by the reference's peak magnitude, not
    // on the per-element relative error (see `score`). A layout error (wrong nibble half,
    // wrong scale stride or group size, a missing `scale2`) decorrelates the output from the
    // reference rather than adding rounding-sized error.
    const BAR: f32 = 0.05;
    if e_gemv.abs / scale > BAR {
        bail!(
            "the PRODUCTION GEMV missed its own reference by {:.4} of scale — the harness is \
             wrong, not the grouped kernel",
            e_gemv.abs / scale
        );
    }
    if e_gemm.abs / scale > BAR {
        bail!(
            "grouped GEMM peak error {:.4} of scale exceeds {BAR}: the NVFP4 layout is NOT shared",
            e_gemm.abs / scale
        );
    }
    // 2026-09-25: The GEMM must also be at least as close to the BF16-weight reference as to
    // the exact one: evidence that its remaining gap is the BF16 weight operand.
    if e_gemm_b.abs > e_gemm.abs {
        bail!(
            "grouped GEMM is FURTHER from the BF16-weight reference ({:.6}) than from the exact \
             one ({:.6}) — the gap is not operand rounding; do not attribute it to precision",
            e_gemm_b.abs,
            e_gemm.abs
        );
    }
    println!(
        "PASS — the grouped GEMM reads GLM's NVFP4 experts correctly; residual gap is BF16 \
         operand precision ({:.2}x closer to the BF16-weight reference).",
        e_gemm.abs / e_gemm_b.abs.max(f32::MIN_POSITIVE)
    );

    tile_identity(gpu, k_sort)?;
    Ok(())
}

/// 2026-09-29: One grouped-GEMM tile: entry point and launch geometry (`GemmTile` in
/// `glm5next_mlp/forward_prefill_gemm/tile.rs`).
struct TileDef {
    kernel: &'static str,
    m_tile: u32,
    n_tile: u32,
    threads: u32,
    m_fast: bool,
}

/// 2026-09-29: The byte-identity reference, the production default tile.
const REF_TILE: TileDef = TileDef {
    kernel: "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128",
    m_tile: 16,
    n_tile: 64,
    threads: 128,
    m_fast: false,
};

/// 2026-09-29: The whole-chunk prefill M1 tiles, each required to match `REF_TILE` exactly.
const M1_TILES: &[TileDef] = &[
    TileDef {
        kernel: "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128_mfast",
        m_tile: 16,
        n_tile: 64,
        threads: 128,
        m_fast: true,
    },
    TileDef {
        kernel: "moe_w4a16_grouped_gemm_ptrtable_bt_k128",
        m_tile: 64,
        n_tile: 64,
        threads: 128,
        m_fast: false,
    },
    TileDef {
        kernel: "moe_w4a16_grouped_gemm_ptrtable_bt_m64_k128_mfast",
        m_tile: 64,
        n_tile: 64,
        threads: 128,
        m_fast: true,
    },
    TileDef {
        kernel: "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64",
        m_tile: 128,
        n_tile: 64,
        threads: 256,
        m_fast: false,
    },
    TileDef {
        kernel: "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_mfast",
        m_tile: 128,
        n_tile: 64,
        threads: 256,
        m_fast: true,
    },    // 2026-09-30: VEC_A (coalesced 16-byte A staging), same smem_A bits as `bt_m128_k64`.
    TileDef {
        kernel: "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_va",
        m_tile: 128,
        n_tile: 64,
        threads: 256,
        m_fast: false,
    },
];

/// 2026-09-29: Tokens of the identity routing; at `top_k = 8` over 16 experts, 3072 slots.
const IDENT_TOKENS: usize = 384;
/// 2026-09-29: This expert gets a NULL weight pointer, as a remote EP expert does.
const IDENT_REMOTE_EXPERT: usize = 5;

#[allow(clippy::too_many_arguments)]
fn launch_tile(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    t: &TileDef,
    a: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2: DevicePtr,
    c: DevicePtr,
    off: DevicePtr,
    stid: DevicePtr,
    busiest: u32,
    n: usize,
    kk: usize,
) -> Result<()> {
    let m_tiles = busiest.div_ceil(t.m_tile).max(1);
    let n_tiles = (n as u32).div_ceil(t.n_tile);
    let grid = if t.m_fast {
        [m_tiles, n_tiles, NUM_EXPERTS as u32]
    } else {
        [n_tiles, m_tiles, NUM_EXPERTS as u32]
    };
    KernelLaunch::new(gpu, k)
        .grid(grid)
        .block([t.threads, 1, 1])
        .arg_ptr(a)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2)
        .arg_ptr(c)
        .arg_ptr(off)
        .arg_ptr(stid)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(0)
}

/// 2026-09-29: Byte equality of every `M1_TILES` entry with `REF_TILE`; see the module doc.
fn tile_identity(gpu: &dyn GpuBackend, k_sort: KernelHandle) -> Result<()> {
    println!("\n--- M1 tile identity vs {} ---", REF_TILE.kernel);
    let k_ref = gpu
        .kernel("moe_w4a16", REF_TILE.kernel)
        .map_err(|e| anyhow::anyhow!("reference tile {} unresolved: {e}", REF_TILE.kernel))?;
    let mut tiles: Vec<(&TileDef, KernelHandle)> = Vec::new();
    for t in M1_TILES {
        match gpu.kernel("moe_w4a16", t.kernel) {
            Ok(k) => tiles.push((t, k)),
            Err(e) => bail!("M1 tile {} is not in this build's PTX: {e}", t.kernel),
        }
    }

    let mut s = 0x1DE7_7117_u64;
    // 2026-09-29: Expert e is drawn with probability (2e + 1) / 256, so expert 15 takes about
    // 12 % of the slots (~360 rows: two full 128-row tiles and a partial one) and expert 0
    // about 0.4 % (~12 rows).
    let te = IDENT_TOKENS * TOP_K;
    let mut ids: Vec<u32> = Vec::with_capacity(te);
    for _ in 0..IDENT_TOKENS {
        let mut picked: Vec<u32> = Vec::with_capacity(TOP_K);
        while picked.len() < TOP_K {
            let r = (lcg(&mut s) % 256) as usize;
            let e = (0..NUM_EXPERTS).find(|e| (e + 1) * (e + 1) > r).unwrap() as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }
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
        .arg_u32(TOP_K as u32)
        .launch(0)?;
    gpu.synchronize(0)?;
    let off = dn_i32(gpu, d_off, NUM_EXPERTS + 1)?;
    let counts: Vec<i32> = (0..NUM_EXPERTS).map(|e| off[e + 1] - off[e]).collect();
    let busiest = counts.iter().copied().max().unwrap_or(0) as u32;
    println!(
        "  routing: {te} slots, rows per expert {counts:?}, expert {IDENT_REMOTE_EXPERT} remote"
    );

    let mut failures = 0usize;
    for (n, kk) in [(N, K), (2048usize, 4096usize), (4096, 2048)] {
        let packed_bytes = n * kk / 2;
        let scale_bytes = n * kk / NVFP4_GROUP_SIZE;
        let mut packed_ptrs = vec![0u64; NUM_EXPERTS];
        let mut scale_ptrs = vec![0u64; NUM_EXPERTS];
        let mut scale2 = vec![0f32; NUM_EXPERTS];
        let mut owned: Vec<DevicePtr> = Vec::new();
        for e in 0..NUM_EXPERTS {
            let pb: Vec<u8> = (0..packed_bytes).map(|_| lcg(&mut s) as u8).collect();
            // 2026-09-29: E4M3 codes 0x30..=0x47, as in `make_expert`.
            let sb: Vec<u8> = (0..scale_bytes)
                .map(|_| 0x30 + (lcg(&mut s) % 0x18) as u8)
                .collect();
            scale2[e] = 0.5 + (lcg(&mut s) % 64) as f32 / 64.0;
            if e == IDENT_REMOTE_EXPERT {
                continue;
            }
            let pp = up(gpu, &pb)?;
            let sp = up(gpu, &sb)?;
            packed_ptrs[e] = pp.0;
            scale_ptrs[e] = sp.0;
            owned.push(pp);
            owned.push(sp);
        }
        let d_pp = up_u64(gpu, &packed_ptrs)?;
        let d_sp = up_u64(gpu, &scale_ptrs)?;
        let d_s2 = up_f32(gpu, &scale2)?;
        // 2026-09-29: `te` rows of `kk` BF16 values in [-1, 1]; the gathered path reads the
        // first IDENT_TOKENS of them.
        let a: Vec<f32> = (0..te * kk)
            .map(|_| (lcg(&mut s) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let d_a = up_bf16(gpu, &a)?;
        let c_bytes = te * n * 2;
        let d_c = gpu.alloc(c_bytes)?;

        for (path, stid) in [("gathered", d_stid), ("direct", DevicePtr(0))] {
            gpu.memset_async(d_c, 0xFF, c_bytes, 0)?;
            launch_tile(
                gpu, k_ref, &REF_TILE, d_a, d_pp, d_sp, d_s2, d_c, d_off, stid, busiest, n, kk,
            )?;
            gpu.synchronize(0)?;
            let mut want = vec![0u8; c_bytes];
            gpu.copy_d2h(d_c, &mut want)?;
            // 2026-09-29: Sanity of the reference: exactly the remote expert's rows are left
            // at the 0xFFFF fill (a NaN no finite product produces).
            let unwritten = want
                .chunks_exact(2)
                .filter(|h| h[0] == 0xFF && h[1] == 0xFF)
                .count();
            let remote_elems = counts[IDENT_REMOTE_EXPERT] as usize * n;
            if unwritten != remote_elems {
                bail!(
                    "reference {} left {unwritten} elements unwritten at N={n} K={kk} {path}, \
                     expected {remote_elems} (the remote expert's rows)",
                    REF_TILE.kernel
                );
            }
            for (t, k) in &tiles {
                gpu.memset_async(d_c, 0xFF, c_bytes, 0)?;
                launch_tile(
                    gpu, *k, t, d_a, d_pp, d_sp, d_s2, d_c, d_off, stid, busiest, n, kk,
                )?;
                gpu.synchronize(0)?;
                let mut got = vec![0u8; c_bytes];
                gpu.copy_d2h(d_c, &mut got)?;
                let diffs = want
                    .chunks_exact(2)
                    .zip(got.chunks_exact(2))
                    .filter(|(x, y)| x != y)
                    .count();
                match want.iter().zip(&got).position(|(x, y)| x != y) {
                    None => println!(
                        "  N={n:4} K={kk:4} {path:8} {:<52} BYTE-IDENTICAL",
                        t.kernel
                    ),
                    Some(i) => {
                        failures += 1;
                        let e = i / 2;
                        println!(
                            "  N={n:4} K={kk:4} {path:8} {:<52} 🔴 {diffs} elements differ; first at \
                             sorted row {}, n {}: want {:02x?} got {:02x?}",
                            t.kernel,
                            e / n,
                            e % n,
                            &want[e * 2..e * 2 + 2],
                            &got[e * 2..e * 2 + 2]
                        );
                    }
                }
            }
        }
        for p in owned.into_iter().chain([d_pp, d_sp, d_s2, d_a, d_c]) {
            gpu.free(p)?;
        }
    }
    if failures > 0 {
        bail!(
            "{failures} M1 tile run(s) are NOT byte-identical to {}",
            REF_TILE.kernel
        );
    }
    println!(
        "PASS — every M1 tile is byte-identical to {}",
        REF_TILE.kernel
    );
    Ok(())
}
