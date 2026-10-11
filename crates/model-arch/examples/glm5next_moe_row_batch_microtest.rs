// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bit-identity gate: the row-batched `w4a16_gemv_sw_moe_batchm_m<R>` against the
//! per-row `w4a16_gemv_sw_moe`, for every (row, slot) either computes.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants:
//! - The run fails (panics) if the union table claims a (row, slot) twice or never, maps a slot
//!   to the wrong expert, or its size differs from the distinct id count; it fails if any arm's
//!   output is not byte-identical to the per-row path.
//!
//! The batched kernel loads an expert's weights once for all the rows that selected it; each
//! row still gets the per-row kernel's `w4a16_gemv_partial` walk, shuffle tree and two-term
//! combine, so the assert is byte equality.
//!
//! Covered, because each is a way the union table can be wrong rather than imprecise:
//!   * rows that share experts and rows that share none;
//!   * remote experts (`packed_ptrs == 0`), whose slots neither path writes;
//!   * experts selected by one row and not another;
//!   * the shared-input (gate/up, `a_slot_stride = 0`) and slot-major (down) input layouts;
//!   * every tier 2..=8 at `top_k = 4` and `top_k = 8`, up to `rows * top_k = 64`, the most
//!     ids `forward_moe` gives the single `glm5next_moe_row_union` block.
//!
//! 2026-10-11: The fast entries (`METRALE_GLM_MOE_GATEUP_FAST` `_gateup_m<R>`,
//! `METRALE_GLM_MOE_DOWN_FAST` `_down_m<R>`, and for 9..=16 rows their `_m8x2` pair entries, #80
//! m16 v2) are checked against the per-row path too: gate+up on the gate/up layout (both outputs),
//! down on a K = 1024 slot-major layout ("down1k"; the fast down entry needs K <= 1024).
//!
//! The union table is checked separately from the arithmetic: a table that dropped ids could
//! still produce a plausible output.
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_moe_row_batch_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const N: usize = 2048;
const K: usize = 1024;
/// 2026-09-25: At `rows = 8`, `top_k = 8` the disjoint routing case needs 64 distinct ids.
/// 2026-10-11: 136: at `rows = 16`, `top_k = 8` it needs 128 (the METRALE_GLM_MOE_UNION_ROWS tiers).
const NUM_EXPERTS: usize = 136;
/// 2026-09-25: 8 is GLM-5.3's routed top-k.
const TOP_KS: [usize; 2] = [4, 8];
/// 2026-09-25: The widest `w4a16_gemv_sw_moe_batchm_m<R>` that `w4a16_gemv.cu` compiles.
/// 2026-10-11: 16, with the `_m9` .. `_m16` union-sweep entries (METRALE_GLM_MOE_UNION_ROWS).
const MAX_ROWS: usize = 16;
/// 2026-09-25: `MOE_ROW_UNION_MAX_IDS` in `glm5next_mlp/forward.rs`: the most ids
/// `forward_moe` gives the one-block `glm5next_moe_row_union`.
/// 2026-10-11: `MOE_ROW_UNION_WIDE_MAX_IDS` (128) for the 9..=16-row tiers.
const MAX_UNION_IDS: usize = 128;
/// 2026-10-11: The car's `METRALE_GLM_MOE_BATCHM_COLS`; its `_m<R>_c8` entries are checked too.
const COLS_J: usize = 8;

/// 2026-09-25: Deterministic random bytes. A realistic NVFP4 packing is irrelevant to a
/// bit-equality gate.
fn lcg(seed: &mut u64) -> u8 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

struct Table {
    packed: DevicePtr,
    scale: DevicePtr,
    scale2: DevicePtr,
}

#[allow(clippy::too_many_arguments)]
fn per_row(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &Table,
    c: DevicePtr,
    ids: DevicePtr,
    n: usize,
    kk: usize,
    top_k: usize,
    input_stride: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(8) as u32, top_k as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2)
        .arg_ptr(c)
        .arg_ptr(ids)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(input_stride as u32)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn batched(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &Table,
    c: DevicePtr,
    u_eid: DevicePtr,
    u_slot: DevicePtr,
    n: usize,
    kk: usize,
    rows: usize,
    top_k: usize,
    a_row_stride: usize,
    a_slot_stride: usize,
    c_row_stride: usize,
    j: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(8 * j) as u32, (rows * top_k) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2)
        .arg_ptr(c)
        .arg_ptr(u_eid)
        .arg_ptr(u_slot)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(a_row_stride as u32)
        .arg_u32(a_slot_stride as u32)
        .arg_u32(c_row_stride as u32)
        .launch(0)
}

/// 2026-10-11: One fast gate+up (`pair`: `_gateup_m8x2`, which takes `rows` last) or fast down
/// launch, grid x = ceil(n / cols).
#[allow(clippy::too_many_arguments)]
fn fast(
    g: &dyn GpuBackend,
    k: KernelHandle,
    gateup: bool,
    pair: bool,
    a: DevicePtr,
    t: &Table,
    c: [DevicePtr; 2],
    u_eid: DevicePtr,
    u_slot: DevicePtr,
    n: usize,
    kk: usize,
    rows: usize,
    top_k: usize,
    strides: [usize; 3],
) -> Result<()> {
    let cols = if gateup { 64 } else { 128 };
    let mut l = KernelLaunch::new(g, k)
        .grid([n.div_ceil(cols) as u32, (rows * top_k) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2);
    if gateup {
        l = l.arg_ptr(t.packed).arg_ptr(t.scale).arg_ptr(t.scale2).arg_ptr(c[0]).arg_ptr(c[1]);
    } else {
        l = l.arg_ptr(c[0]);
    }
    l = l
        .arg_ptr(u_eid)
        .arg_ptr(u_slot)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(strides[0] as u32)
        .arg_u32(strides[1] as u32)
        .arg_u32(strides[2] as u32);
    if pair {
        l = l.arg_u32(rows as u32);
    }
    l.launch(0)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k_row = gpu.kernel("w4a16_gemv", "w4a16_gemv_sw_moe")?;
    let k_union = gpu.kernel("w4a16_gemv", "glm5next_moe_row_union")?;
    // 2026-10-11: The car's union kernel (METRALE_GLM_MOE_UNION_SCAN=1); its tables must match.
    let k_scan = gpu.kernel("w4a16_gemv", "glm5next_moe_row_union_scan")?;
    let k_b: Vec<KernelHandle> = (2..=MAX_ROWS)
        .map(|r| gpu.kernel("w4a16_gemv", &format!("w4a16_gemv_sw_moe_batchm_m{r}")))
        .collect::<Result<_, _>>()?;
    let k_bc: Vec<KernelHandle> = (2..=MAX_ROWS)
        .map(|r| gpu.kernel("w4a16_gemv", &format!("w4a16_gemv_sw_moe_batchm_m{r}_c{COLS_J}")))
        .collect::<Result<_, _>>()?;
    // 2026-10-11: The fast entries per tier 2..=MAX_ROWS: `_<kind>_m<R>` up to 8 rows, the
    // `_<kind>_m8x2` pair entry above (flag true).
    let fast_tiers = |kind: &str| -> Result<Vec<(KernelHandle, bool)>> {
        (2..=MAX_ROWS)
            .map(|r| {
                let (name, pair) = if r <= 8 {
                    (format!("w4a16_gemv_sw_moe_batchm_{kind}_m{r}"), false)
                } else {
                    (format!("w4a16_gemv_sw_moe_batchm_{kind}_m8x2"), true)
                };
                Ok((gpu.kernel("w4a16_gemv", &name)?, pair))
            })
            .collect()
    };
    let k_gu = fast_tiers("gateup")?;
    let k_dn = fast_tiers("down")?;

    // 2026-09-25: Experts 3 and 11 are remote: NULL pointers.
    let mut seed = 0x51ed_5eedu64;
    let mut packed_ptrs = Vec::new();
    let mut scale_ptrs = Vec::new();
    let mut scale2 = Vec::new();
    for e in 0..NUM_EXPERTS {
        let remote = e == 3 || e == 11;
        if remote {
            packed_ptrs.push(0u64);
            scale_ptrs.push(0u64);
            scale2.push(0.0f32);
            continue;
        }
        let w: Vec<u8> = (0..N * K / 2).map(|_| lcg(&mut seed)).collect();
        // 2026-09-25: E4M3 codes 0x38..=0x3F (1.0..=1.875), so no scale is zero, inf or NaN.
        let s: Vec<u8> = (0..N * (K / 16))
            .map(|_| 0x38 | (lcg(&mut seed) & 0x07))
            .collect();
        packed_ptrs.push(up(&gpu, &w)?.0);
        scale_ptrs.push(up(&gpu, &s)?.0);
        scale2.push(1.0 + (e as f32) * 0.01);
    }
    let t = Table {
        packed: up(
            &gpu,
            &packed_ptrs
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        scale: up(
            &gpu,
            &scale_ptrs
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        scale2: up(
            &gpu,
            &scale2
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
    };

    // 2026-09-25: Routing cases per (rows, top_k). `partial+remote` puts the remote experts 3
    // and 11 in every row, so a tier that wrote a remote slot would show up as a diff.
    fn cases(rows: usize, top_k: usize) -> Vec<(String, Vec<Vec<i32>>)> {
        let m = NUM_EXPERTS as i32;
        let disjoint: Vec<Vec<i32>> = (0..rows)
            .map(|r| (0..top_k).map(|i| ((r * top_k + i) as i32) % m).collect())
            .collect();
        let full: Vec<Vec<i32>> = (0..rows)
            .map(|_| (0..top_k).map(|i| (i as i32 * 2) % m).collect())
            .collect();
        let partial: Vec<Vec<i32>> = (0..rows)
            .map(|r| {
                let mut v = vec![3i32, 11];
                let mut n = 0i32;
                while v.len() < top_k {
                    let c = ((r as i32 * 5) + n * 3 + 20) % m;
                    if !v.contains(&c) {
                        v.push(c);
                    }
                    n += 1;
                }
                v.truncate(top_k);
                v
            })
            .collect();
        // 2026-09-25: Half of each row's ids shared by every row, half private.
        let heavy: Vec<Vec<i32>> = (0..rows)
            .map(|r| {
                let mut v: Vec<i32> = (0..top_k / 2).map(|i| i as i32).collect();
                let mut n = 0i32;
                while v.len() < top_k {
                    let c = (30 + r as i32 * 7 + n) % m;
                    if !v.contains(&c) {
                        v.push(c);
                    }
                    n += 1;
                }
                v.truncate(top_k);
                v
            })
            .collect();
        vec![
            (format!("{rows}r k{top_k} disjoint"), disjoint),
            (format!("{rows}r k{top_k} full overlap"), full),
            (format!("{rows}r k{top_k} partial+remote"), partial),
            (format!("{rows}r k{top_k} heavy overlap"), heavy),
        ]
    }

    let all: Vec<(String, Vec<Vec<i32>>, usize)> = TOP_KS
        .iter()
        .flat_map(|&tk| {
            (2..=MAX_ROWS)
                .filter(move |r| r * tk <= MAX_UNION_IDS)
                .flat_map(move |r| cases(r, tk).into_iter().map(move |(t, c)| (t, c, tk)))
        })
        .collect();

    let mut failures = 0usize;
    for (tag, ids, top_k) in &all {
        let (rows, top_k) = (ids.len(), *top_k);
        let flat: Vec<i32> = ids.iter().flatten().copied().collect();
        let d_ids = up(
            &gpu,
            &flat
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let d_ueid = gpu.alloc(rows * top_k * 4)?;
        let d_uslot = gpu.alloc(rows * top_k * rows * 4)?;

        KernelLaunch::new(&gpu, k_union)
            .grid([1, 1, 1])
            .block([(rows * top_k) as u32, 1, 1])
            .arg_ptr(d_ids)
            .arg_ptr(d_ueid)
            .arg_ptr(d_uslot)
            .arg_u32(rows as u32)
            .arg_u32(top_k as u32)
            .launch(0)?;
        gpu.synchronize(0)?;
        // 2026-10-11: The scan variant's tables, byte-compared with the base kernel's.
        {
            let d_ueid2 = gpu.alloc(rows * top_k * 4)?;
            let d_uslot2 = gpu.alloc(rows * top_k * rows * 4)?;
            KernelLaunch::new(&gpu, k_scan)
                .grid([1, 1, 1])
                .block([(rows * top_k) as u32, 1, 1])
                .arg_ptr(d_ids)
                .arg_ptr(d_ueid2)
                .arg_ptr(d_uslot2)
                .arg_u32(rows as u32)
                .arg_u32(top_k as u32)
                .launch(0)?;
            gpu.synchronize(0)?;
            let same = dn(&gpu, d_ueid, rows * top_k * 4)? == dn(&gpu, d_ueid2, rows * top_k * 4)?
                && dn(&gpu, d_uslot, rows * top_k * rows * 4)?
                    == dn(&gpu, d_uslot2, rows * top_k * rows * 4)?;
            if !same {
                println!("  FAIL  {tag:28} union_scan tables differ from glm5next_moe_row_union");
                failures += 1;
            }
            gpu.free(d_ueid2)?;
            gpu.free(d_uslot2)?;
        }

        let ueid: Vec<i32> = dn(&gpu, d_ueid, rows * top_k * 4)?
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let uslot: Vec<i32> = dn(&gpu, d_uslot, rows * top_k * rows * 4)?
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut seen = vec![vec![false; top_k]; rows];
        for (u, &e) in ueid.iter().enumerate() {
            if e < 0 {
                continue;
            }
            for r in 0..rows {
                let s = uslot[u * rows + r];
                if s < 0 {
                    continue;
                }
                assert_eq!(
                    ids[r][s as usize], e,
                    "{tag}: union entry {u} claims row {r} slot {s}"
                );
                assert!(
                    !seen[r][s as usize],
                    "{tag}: row {r} slot {s} claimed twice"
                );
                seen[r][s as usize] = true;
            }
        }
        for r in 0..rows {
            for s in 0..top_k {
                assert!(seen[r][s], "{tag}: row {r} slot {s} never claimed");
            }
        }
        let n_union = ueid.iter().filter(|e| **e >= 0).count();
        let distinct = {
            let mut v: Vec<i32> = flat.clone();
            v.sort_unstable();
            v.dedup();
            v.len()
        };
        assert_eq!(n_union, distinct, "{tag}: union size");

        for (layout, kk, nn, a_slot_stride) in
            [("gate/up", K, N, 0usize), ("down", N, K, N), ("down1k", K, N, K)]
        {
            let a_row_stride = if a_slot_stride == 0 { kk } else { top_k * kk };
            let a: Vec<u8> = (0..rows * a_row_stride * 2)
                .map(|_| lcg(&mut seed))
                .collect();
            let d_a = up(&gpu, &a)?;

            let bytes = rows * top_k * nn * 2;
            let d_ref = gpu.alloc(bytes)?;
            let d_new = gpu.alloc(bytes)?;
            gpu.memset_async(d_ref, 0, bytes, 0)?;
            gpu.memset_async(d_new, 0, bytes, 0)?;

            for r in 0..rows {
                per_row(
                    &gpu,
                    k_row,
                    d_a.offset(r * a_row_stride * 2),
                    &t,
                    d_ref.offset(r * top_k * nn * 2),
                    d_ids.offset(r * top_k * 4),
                    nn,
                    kk,
                    top_k,
                    a_slot_stride,
                )?;
            }
            batched(
                &gpu,
                k_b[rows - 2],
                d_a,
                &t,
                d_new,
                d_ueid,
                d_uslot,
                nn,
                kk,
                rows,
                top_k,
                a_row_stride,
                a_slot_stride,
                top_k * nn,
                1,
            )?;
            // 2026-10-11: The `_c8` entry into its own buffer.
            let d_cols = gpu.alloc(bytes)?;
            gpu.memset_async(d_cols, 0, bytes, 0)?;
            batched(
                &gpu,
                k_bc[rows - 2],
                d_a,
                &t,
                d_cols,
                d_ueid,
                d_uslot,
                nn,
                kk,
                rows,
                top_k,
                a_row_stride,
                a_slot_stride,
                top_k * nn,
                COLS_J,
            )?;
            gpu.synchronize(0)?;

            let r_ref = dn(&gpu, d_ref, bytes)?;
            let r_new = dn(&gpu, d_new, bytes)?;
            if dn(&gpu, d_cols, bytes)? != r_ref {
                println!("  FAIL  {tag:28} [{layout:7}] _c{COLS_J} differs from the per-row path");
                failures += 1;
            }
            gpu.free(d_cols)?;
            // 2026-10-11: The fast entry for this layout (gate+up writes both outputs; the
            // table serves as gate and up), each output into its own poisoned buffer.
            let fe = match layout {
                "gate/up" => Some((k_gu[rows - 2], true)),
                "down1k" => Some((k_dn[rows - 2], false)),
                _ => None,
            };
            if let Some(((kf, pair), gateup)) = fe {
                let c = [gpu.alloc(bytes)?, gpu.alloc(bytes)?];
                for p in c {
                    gpu.memset_async(p, 0, bytes, 0)?;
                }
                fast(
                    &gpu,
                    kf,
                    gateup,
                    pair,
                    d_a,
                    &t,
                    c,
                    d_ueid,
                    d_uslot,
                    nn,
                    kk,
                    rows,
                    top_k,
                    [a_row_stride, a_slot_stride, top_k * nn],
                )?;
                gpu.synchronize(0)?;
                let outs = if gateup { 2 } else { 1 };
                for (i, &p) in c.iter().enumerate().take(outs) {
                    if dn(&gpu, p, bytes)? != r_ref {
                        let kind = if gateup { "gateup" } else { "down" };
                        let e = if pair { "m8x2".to_string() } else { format!("m{rows}") };
                        println!(
                            "  FAIL  {tag:28} [{layout:7}] _{kind}_{e} output {i} differs from the per-row path"
                        );
                        failures += 1;
                    }
                }
                for p in c {
                    gpu.free(p)?;
                }
            }
            if r_ref == r_new {
                println!(
                    "  PASS  {tag:28} [{layout:7}] rows={rows} union={n_union}/{}",
                    rows * top_k
                );
            } else {
                let diff = r_ref.iter().zip(&r_new).filter(|(a, b)| a != b).count();
                println!("  FAIL  {tag:28} [{layout:7}] {diff}/{bytes} bytes differ");
                failures += 1;
            }
        }
    }

    if failures > 0 {
        bail!("{failures} arm(s) are not bit-identical to the per-row path");
    }
    println!(
        "\nPASS: row-batched MoE (and _c{COLS_J}, the fast gateup/down entries incl. _m8x2) is \
         bit-identical to the per-row path on every arm, tiers 2..={MAX_ROWS}."
    );
    Ok(())
}
