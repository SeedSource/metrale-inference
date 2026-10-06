// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Bitwise gate for `glm5next_moe_combine_indexed_local`
//! (`METRALE_GLM_MOE_COMBINE_LOCAL`) against `glm5next_moe_combine_indexed` over a zeroed
//! `expert_out`, the pair the grouped routed-MoE prefill switches between.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - Exits 0 only if, for every case, the local kernel's whole BF16 output equals the reference's
//!   bit for bit while the local arm's remote-expert rows hold random bytes (NaN patterns
//!   included) instead of zeros, and a known-bad arm (local span shifted by one expert) differs.
//!
//! Cases: GLM-5.3 shapes (288 experts, top_k 8, hidden 4096) at 8191 and 7 tokens, as EP rank 0
//! (experts 0..144) and rank 1 (144..288), plus a 2-expert-per-rank toy. Routing weights include
//! negative values, so the reference adds `-0.0f` products. Timing (8191 tokens): reference =
//! memset of `rows * top_k * hidden` BF16 + combine, local = combine only; median of 10.
//!
//! Run: `cargo run -p metrale-model-arch --release --features cuda,gpu-examples --example
//! moe_combine_local_bitparity_microtest`

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mlp::FFN_MODULE;

// 2026-10-06: CUDA driver event API for timing, declared as in `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 24) as f32
    }
}

fn upload(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn bf16_bytes(v: f32) -> [u8; 2] {
    half::bf16::from_f32(v).to_le_bytes()
}

/// 2026-10-06: Host routing for `t` tokens: distinct experts per token, rows sorted by expert
/// (slot order within an expert), as `moe_sort_by_expert` lays them out.
struct Routing {
    offsets: Vec<i32>,
    token_to_perm: Vec<i32>,
    expert_of_row: Vec<usize>,
}

fn route(rng: &mut Lcg, t: usize, top_k: usize, experts: usize) -> Routing {
    let mut ids = Vec::with_capacity(t * top_k);
    for _ in 0..t {
        let mut chosen: Vec<usize> = Vec::with_capacity(top_k);
        while chosen.len() < top_k {
            let e = rng.next() as usize % experts;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        ids.extend(chosen);
    }
    let mut counts = vec![0usize; experts];
    for &e in &ids {
        counts[e] += 1;
    }
    let mut offsets = vec![0i32; experts + 1];
    for e in 0..experts {
        offsets[e + 1] = offsets[e] + counts[e] as i32;
    }
    let mut fill: Vec<i32> = offsets[..experts].to_vec();
    let mut token_to_perm = vec![0i32; ids.len()];
    let mut expert_of_row = vec![0usize; ids.len()];
    for (slot, &e) in ids.iter().enumerate() {
        token_to_perm[slot] = fill[e];
        expert_of_row[fill[e] as usize] = e;
        fill[e] += 1;
    }
    Routing {
        offsets,
        token_to_perm,
        expert_of_row,
    }
}

struct Bufs {
    eo_ref: DevicePtr,
    eo_new: DevicePtr,
    perm: DevicePtr,
    wts: DevicePtr,
    shared: DevicePtr,
    offsets: DevicePtr,
    out: DevicePtr,
    t: usize,
    hidden: usize,
    top_k: usize,
}

fn launch_ref(g: &dyn GpuBackend, k: KernelHandle, b: &Bufs) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([b.t as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b.eo_ref)
        .arg_ptr(b.perm)
        .arg_ptr(b.wts)
        .arg_ptr(b.shared)
        .arg_ptr(b.out)
        .arg_u32(b.hidden as u32)
        .arg_u32(b.top_k as u32)
        .launch(0)
}

fn launch_local(
    g: &dyn GpuBackend,
    k: KernelHandle,
    b: &Bufs,
    first: usize,
    end: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([b.t as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b.eo_new)
        .arg_ptr(b.perm)
        .arg_ptr(b.wts)
        .arg_ptr(b.shared)
        .arg_ptr(b.out)
        .arg_u32(b.hidden as u32)
        .arg_u32(b.top_k as u32)
        .arg_ptr(b.offsets)
        .arg_u32(first as u32)
        .arg_u32(end as u32)
        .launch(0)
}

fn read_out(g: &dyn GpuBackend, b: &Bufs) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut v = vec![0u8; b.t * b.hidden * 2];
    g.copy_d2h(b.out, &mut v)?;
    Ok(v)
}

fn time_ms(g: &dyn GpuBackend, reps: usize, mut f: impl FnMut() -> Result<()>) -> Result<f32> {
    let mut v = Vec::with_capacity(reps);
    for _ in 0..reps {
        let (mut e0, mut e1, mut ms) = (0u64, 0u64, 0f32);
        unsafe {
            if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
                bail!("cuEventCreate returned an error");
            }
            cuEventRecord(e0, 0);
        }
        f()?;
        unsafe {
            cuEventRecord(e1, 0);
            cuEventSynchronize(e1);
            cuEventElapsedTime(&mut ms, e0, e1);
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        v.push(ms);
    }
    g.synchronize(0)?;
    v.sort_by(f32::total_cmp);
    Ok(v[v.len() / 2])
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let Some(glm) = sets.iter().find(|s| s.target.model == "glm-5.3-flash") else {
        bail!("moe_combine_local_bitparity_microtest: no glm-5.3-flash PTX set in this build");
    };
    let backend = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &backend;
    let k_ref = g.kernel(FFN_MODULE, "glm5next_moe_combine_indexed")?;
    let k_loc = g.kernel(FFN_MODULE, "glm5next_moe_combine_indexed_local")?;
    let mut bad = 0usize;

    // 2026-10-06: (tokens, experts, top_k, hidden, local first, local end, time it).
    let cases = [
        (
            8191usize, 288usize, 8usize, 4096usize, 0usize, 144usize, true,
        ),
        (8191, 288, 8, 4096, 144, 288, false),
        (7, 288, 8, 4096, 0, 144, false),
        (7, 288, 8, 4096, 144, 288, false),
        (33, 4, 2, 1000, 2, 4, false),
    ];
    for (ci, &(t, experts, top_k, hidden, first, end, timed)) in cases.iter().enumerate() {
        let mut rng = Lcg(0xC0B1 + ci as u64);
        let r = route(&mut rng, t, top_k, experts);
        let te = t * top_k;
        let (lo, hi) = (r.offsets[first] as usize, r.offsets[end] as usize);
        // 2026-10-06: Local rows random BF16 in both arms; remote rows zero (reference) or random
        // bytes, NaN patterns included (local arm).
        let mut eo_ref = vec![0u8; te * hidden * 2];
        let mut eo_new = vec![0u8; te * hidden * 2];
        for row in 0..te {
            let local = (first..end).contains(&r.expert_of_row[row]);
            debug_assert_eq!(local, (lo..hi).contains(&row));
            for d in 0..hidden {
                let i = (row * hidden + d) * 2;
                if local {
                    let v = bf16_bytes(rng.unit() * 4.0 - 2.0);
                    eo_ref[i..i + 2].copy_from_slice(&v);
                    eo_new[i..i + 2].copy_from_slice(&v);
                } else {
                    let x = rng.next();
                    eo_new[i..i + 2].copy_from_slice(&(x as u16).to_le_bytes());
                }
            }
        }
        // 2026-10-06: Weights in (-0.5, 1): negative ones make the reference add -0.0f products.
        let wts: Vec<u8> = (0..te)
            .flat_map(|_| (rng.unit() * 1.5 - 0.5).to_le_bytes())
            .collect();
        let shared: Vec<u8> = (0..t * hidden)
            .flat_map(|_| bf16_bytes(rng.unit() * 2.0 - 1.0))
            .collect();
        let perm: Vec<u8> = r
            .token_to_perm
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let offs: Vec<u8> = r.offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b = Bufs {
            eo_ref: upload(g, &eo_ref)?,
            eo_new: upload(g, &eo_new)?,
            perm: upload(g, &perm)?,
            wts: upload(g, &wts)?,
            shared: upload(g, &shared)?,
            offsets: upload(g, &offs)?,
            out: g.alloc(t * hidden * 2)?,
            t,
            hidden,
            top_k,
        };
        launch_ref(g, k_ref, &b)?;
        let want = read_out(g, &b)?;
        launch_local(g, k_loc, &b, first, end)?;
        let got = read_out(g, &b)?;
        let diff = want
            .chunks_exact(2)
            .zip(got.chunks_exact(2))
            .filter(|(x, y)| x != y)
            .count();
        println!(
            "case tokens={t} experts={experts} top_k={top_k} hidden={hidden} local={first}..{end} \
             rows {lo}..{hi} of {te}: {diff} differing BF16 outputs of {}",
            t * hidden
        );
        if diff != 0 {
            println!("MISMATCH: tokens={t} local={first}..{end}");
            bad += 1;
        }
        // 2026-10-06: Known-bad: the span starting one past its first expert that has rows
        // (one expert later is blind when that expert is empty, as at 7 tokens) must change the
        // output.
        let busy = (first..end)
            .find(|&e| r.offsets[e + 1] > r.offsets[e])
            .unwrap_or(first);
        let shifted = (busy + 1).min(experts);
        launch_local(g, k_loc, &b, shifted, (end + 1).min(experts))?;
        let kb = read_out(g, &b)?;
        let kd = want
            .chunks_exact(2)
            .zip(kb.chunks_exact(2))
            .filter(|(x, y)| x != y)
            .count();
        if kd == 0 {
            println!("KNOWN_BAD not detected: tokens={t} (the comparison is blind)");
            bad += 1;
        } else {
            println!("KNOWN_BAD detected: shifted span differs in {kd} outputs");
        }
        if timed {
            let zero_bytes = te * hidden * 2;
            let t_ref = time_ms(g, 10, || {
                g.memset_async(b.eo_ref, 0, zero_bytes, 0)?;
                launch_ref(g, k_ref, &b)
            })?;
            let t_loc = time_ms(g, 10, || launch_local(g, k_loc, &b, first, end))?;
            println!(
                "TIMING tokens={t}: memset+combine {t_ref:.3} ms  local combine {t_loc:.3} ms  \
                 speedup {:.2}x",
                t_ref / t_loc
            );
        }
        for p in [
            b.eo_ref, b.eo_new, b.perm, b.wts, b.shared, b.offsets, b.out,
        ] {
            g.free(p)?;
        }
    }
    if bad != 0 {
        bail!("moe_combine_local_bitparity_microtest: {bad} check(s) did not hold");
    }
    println!(
        "PASS: glm5next_moe_combine_indexed_local equals glm5next_moe_combine_indexed over a zeroed \
         expert_out bit for bit in every case (remote rows random); known-bad detected"
    );
    Ok(())
}
