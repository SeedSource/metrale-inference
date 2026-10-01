// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Byte-parity gate for the head-grouped DSA MLA attention
//! (`glm5next_dsa_mla_decode_fp8_hg{2,4,8}`, `METRALE_GLM_DSA_MLA_HEADGROUP`) against the
//! per-head `glm5next_dsa_mla_decode_fp8`, both launched through
//! `attend::decode_attention_headgroup`.
//!
//! Inputs: 32 heads, random BF16 q, a random FP8 pool (no NaN codes) with a non-power-of-two
//! k_scale, random paged block tables into a shared pool, and selections of width W in
//! {2051, 257, 9, 1} over {1, 7, 256} rows that carry `-1` holes, other negative and
//! past-`seq_len` indices, and duplicates. Row 1 (when present) has `seq_len` 0 and must stay
//! untouched; row 2 has no valid index and must come out zero. Two extra cases take the
//! kernels' other branches: a pool at a 1-byte offset (byte loads instead of 16-byte loads) and
//! a distinct V buffer and scale.
//!
//! Every comparison is bitwise. Each arm's output starts filled with its own poison byte, so an
//! element one arm never writes cannot compare equal. A negative control (one flipped q bit) must
//! be detected, and a run that compared no element fails.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 when every case is byte-identical, 1 on any difference, a vacuous comparison or a
//! negative control that does not fire, 2 when a kernel is absent from this target.
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_mla_headgroup_bitparity_microtest

use anyhow::{Context, Result};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::Glm5NextDsaConfig;
use metrale_model_arch::glm5next_dsa::attend::{
    DSA_MLA_HEADGROUPS, DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaDecodeKernel,
    decode_attention_headgroup,
};
use metrale_model_arch::glm5next_dsa::select::DsaSelectGeometry;

const HEADS: usize = 32;
const KVL: usize = 512;
const BLOCK: usize = 64;
const MAX_CTX: usize = 4096;
const MAX_BLOCKS: usize = MAX_CTX / BLOCK;
const N_PHYS: usize = 96;
const WIDTHS: &[usize] = &[2051, 257, 9, 1];
const ROWS: &[usize] = &[1, 7, 256];
const K_SCALE: f32 = 0.0173;
const V_SCALE: f32 = 0.0291;
const POISON_REF: u8 = 0xA5;
const POISON_NEW: u8 = 0x5A;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn f(&mut self) -> f32 {
        ((self.next() as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn bf16_bytes(&mut self, n: usize, amp: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f32(amp * (2.0 * self.f() - 1.0)).to_bits().to_le_bytes())
            .collect()
    }
    /// 2026-10-01: Random E4M3 codes, the two NaN codes (0x7F, 0xFF) excluded.
    fn fp8_bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| loop {
                let b = (self.next() & 0xFF) as u8;
                if (b & 0x7F) != 0x7F {
                    break b;
                }
            })
            .collect()
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: HEADS,
        q_lora_rank: 1536,
        kv_lora_rank: KVL,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: MAX_CTX,
    }
}

/// 2026-10-01: `decode_attention` reads only `q_rows` and `out_width` from the geometry.
fn geom(rows: usize, width: usize) -> DsaSelectGeometry {
    DsaSelectGeometry {
        seq: MAX_CTX,
        q_rows: rows,
        n_pools_full: 0,
        n_pools: 0,
        select_k: 0,
        out_width: width,
        topk_np2: 0,
        topk_smem: 0,
        index_head_dim: 128,
        index_heads: 32,
        index_kpool: 4,
    }
}

fn paging(rows: usize) -> DsaDecodePaging {
    DsaDecodePaging {
        num_seqs: rows,
        num_q_heads: HEADS,
        num_kv_heads: 1,
        max_blocks_per_seq: MAX_BLOCKS,
        block_size: BLOCK,
        cache_stride_bytes: (BLOCK * KVL) as u64,
    }
}

/// 2026-10-01: One case's device inputs and the host copies the checks need.
struct Case {
    rows: usize,
    width: usize,
    seq_lens: Vec<i32>,
    q_bytes: Vec<u8>,
    inputs: DsaDecodeInputs,
    owned: Vec<DevicePtr>,
}

#[derive(Clone, Copy)]
enum Pool {
    Aligned,
    /// 2026-10-01: K = V at a 1-byte offset: the head-grouped kernels' byte-load branch.
    Misaligned,
    /// 2026-10-01: A separate V buffer and scale: the `same_kv == false` branch.
    DistinctV,
}

fn build_case(
    g: &dyn GpuBackend,
    rows: usize,
    width: usize,
    pool: Pool,
    seed: u64,
) -> Result<Case> {
    let mut rng = Lcg(seed);
    let q_amp = if seed & 1 == 0 { 1.0 } else { 4.0 };
    let q_bytes = rng.bf16_bytes(rows * HEADS * KVL, q_amp);

    let mut seq_lens = Vec::with_capacity(rows);
    let mut sel = Vec::with_capacity(rows * width);
    let mut bt = Vec::with_capacity(rows * MAX_BLOCKS);
    for r in 0..rows {
        let sl = if r == 1 { 0 } else { 1 + rng.below(MAX_CTX) };
        seq_lens.push(sl as i32);
        for _ in 0..MAX_BLOCKS {
            bt.push(rng.below(N_PHYS) as i32);
        }
        let row_start = sel.len();
        for k in 0..width {
            let roll = rng.below(100);
            let t: i32 = if r == 2 {
                // 2026-10-01: No valid index: the row must come out zero.
                match roll % 3 {
                    0 => -1,
                    1 => sl as i32 + rng.below(64) as i32,
                    _ => -(2 + rng.below(9) as i32),
                }
            } else if roll < 8 {
                -1
            } else if roll < 10 {
                -(2 + rng.below(9) as i32)
            } else if roll < 12 {
                sl as i32 + rng.below(64) as i32
            } else if roll < 18 && k > 0 {
                sel[row_start + rng.below(k)]
            } else {
                rng.below(sl.max(1)) as i32
            };
            sel.push(t);
        }
    }

    let pool_bytes = N_PHYS * BLOCK * KVL;
    let fp8 = rng.fp8_bytes(pool_bytes);
    let mut owned = Vec::new();
    let (k_cache, v_cache, v_scale) = match pool {
        Pool::Aligned => {
            let p = up(g, &fp8)?;
            owned.push(p);
            (p, p, K_SCALE)
        }
        Pool::Misaligned => {
            let base = g.alloc(pool_bytes + 16)?;
            owned.push(base);
            g.copy_h2d(&fp8, base.offset(1))?;
            (base.offset(1), base.offset(1), K_SCALE)
        }
        Pool::DistinctV => {
            let k = up(g, &fp8)?;
            let v = up(g, &rng.fp8_bytes(pool_bytes))?;
            owned.push(k);
            owned.push(v);
            (k, v, V_SCALE)
        }
    };
    let q = up(g, &q_bytes)?;
    let out = g.alloc(rows * HEADS * KVL * 2)?;
    let block_tables = up(g, &i32_bytes(&bt))?;
    let seq_lens_d = up(g, &i32_bytes(&seq_lens))?;
    let sel_indices = up(g, &i32_bytes(&sel))?;
    owned.extend([q, out, block_tables, seq_lens_d, sel_indices]);
    Ok(Case {
        rows,
        width,
        seq_lens,
        q_bytes,
        inputs: DsaDecodeInputs {
            q,
            k_cache,
            v_cache,
            out,
            block_tables,
            seq_lens: seq_lens_d,
            sel_indices,
            k_scale: K_SCALE,
            v_scale,
        },
        owned,
    })
}

/// 2026-10-01: One arm: the output poisoned with `poison`, head group `hg` (0 = per head).
fn run(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
    q: DevicePtr,
    hg: usize,
    poison: u8,
) -> Result<Vec<u8>> {
    let bytes = case.rows * HEADS * KVL * 2;
    g.memset(case.inputs.out, poison, bytes)?;
    let inputs = DsaDecodeInputs { q, ..case.inputs };
    let (c, gm, pg) = (cfg(), geom(case.rows, case.width), paging(case.rows));
    decode_attention_headgroup(g, kernel, hg, &c, &gm, &pg, &inputs, 0)?;
    down(g, case.inputs.out, bytes)
}

/// 2026-10-01: Rows with `seq_len` 0 must keep each arm's own poison; every other row must
/// match byte for byte and must have been written. Returns (compared elements, pass, differing
/// bytes).
fn compare(case: &Case, r: &[u8], n: &[u8]) -> (usize, bool, usize) {
    let row_bytes = HEADS * KVL * 2;
    let (mut compared, mut ok, mut diff) = (0usize, true, 0usize);
    for (row, &sl) in case.seq_lens.iter().enumerate() {
        let (a, b) = (
            &r[row * row_bytes..(row + 1) * row_bytes],
            &n[row * row_bytes..(row + 1) * row_bytes],
        );
        if sl == 0 {
            ok &= a.iter().all(|&x| x == POISON_REF) && b.iter().all(|&x| x == POISON_NEW);
            continue;
        }
        compared += HEADS * KVL;
        diff += a.iter().zip(b).filter(|(x, y)| x != y).count();
        ok &= a == b;
        // 2026-10-01: A written row cannot still be all poison.
        ok &= !a.iter().all(|&x| x == POISON_REF);
    }
    (compared, ok, diff)
}

fn main() -> Result<()> {
    // 2026-10-01: The kernel is in the glm-5.3-flash target; `ptx_modules()` aliases the first
    // compiled target, so the backend is built from that set (as `glm5next_dsa_decode_gate`).
    let sets = metrale_kernels::all_ptx_sets();
    let Some(glm) = sets.iter().find(|s| s.target.model == "glm-5.3-flash") else {
        println!("glm-5.3-flash kernel target not built - SKIP");
        std::process::exit(2);
    };
    let backend = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &backend;
    let kernel = Glm5NextDsaDecodeKernel::resolve(g).context("Glm5NextDsaDecodeKernel")?;
    for hg in DSA_MLA_HEADGROUPS {
        if !kernel.has_headgroup(hg) {
            println!("glm5next_dsa_mla_decode_fp8_hg{hg} absent from this target - SKIP");
            std::process::exit(2);
        }
    }

    let mut plan: Vec<(usize, usize, Pool)> = Vec::new();
    for &w in WIDTHS {
        for &rows in ROWS {
            plan.push((rows, w, Pool::Aligned));
        }
    }
    plan.push((7, 257, Pool::Misaligned));
    plan.push((7, 257, Pool::DistinctV));

    let (mut compared, mut failed) = (0usize, 0usize);
    for (i, &(rows, w, pool)) in plan.iter().enumerate() {
        let case = build_case(g, rows, w, pool, 0x5EED_0000 + i as u64)?;
        let r = run(g, kernel, &case, case.inputs.q, 0, POISON_REF)?;
        for hg in DSA_MLA_HEADGROUPS {
            let n = run(g, kernel, &case, case.inputs.q, hg, POISON_NEW)?;
            let (c, ok, diff) = compare(&case, &r, &n);
            compared += c;
            if !ok {
                failed += 1;
            }
            let kind = match pool {
                Pool::Aligned => "aligned",
                Pool::Misaligned => "misaligned",
                Pool::DistinctV => "distinct-V",
            };
            println!(
                "W={w:<5} rows={rows:<4} {kind:<10} hg{hg}  elems={c:<9} byte-identical={ok:<5} \
                 diff_bytes={diff}"
            );
        }
        for p in &case.owned {
            g.free(*p).ok();
        }
    }

    // 2026-10-01: Negative control: one flipped q bit (row 0, head 5) must change the hg4
    // output against the unflipped per-head reference.
    let case = build_case(g, 7, 257, Pool::Aligned, 0xC047_8001)?;
    let r = run(g, kernel, &case, case.inputs.q, 0, POISON_REF)?;
    let mut pert = case.q_bytes.clone();
    pert[2 * (5 * KVL + 17)] ^= 0x10;
    let q_pert = up(g, &pert)?;
    let n = run(g, kernel, &case, q_pert, 4, POISON_NEW)?;
    let (_, same, _) = compare(&case, &r, &n);
    let fired = !same;
    println!("CONTROL 1-bit q flip detected={fired}");
    g.free(q_pert).ok();
    for p in &case.owned {
        g.free(*p).ok();
    }

    // 2026-10-01: Rough timing at the prefill shape (rows 256, W 2051, 32 heads): host clock
    // around 20 launches after 3 warm-up launches, synchronised at both ends.
    let case = build_case(g, 256, 2051, Pool::Aligned, 0x7135)?;
    let launch = |hg: usize| -> Result<()> {
        let (c, gm, pg) = (cfg(), geom(case.rows, case.width), paging(case.rows));
        decode_attention_headgroup(g, kernel, hg, &c, &gm, &pg, &case.inputs, 0)
    };
    let mut base_us = 0.0;
    for hg in std::iter::once(0).chain(DSA_MLA_HEADGROUPS) {
        for _ in 0..3 {
            launch(hg)?;
        }
        g.synchronize(0)?;
        let t = std::time::Instant::now();
        for _ in 0..20 {
            launch(hg)?;
        }
        g.synchronize(0)?;
        let us = t.elapsed().as_secs_f64() * 1e6 / 20.0;
        if hg == 0 {
            base_us = us;
        }
        let name = if hg == 0 {
            "per-head".to_string()
        } else {
            format!("hg{hg}")
        };
        println!(
            "TIMING rows=256 W=2051 heads=32 {name:<8} {us:>10.1} us/launch  x{:.2} vs per-head",
            base_us / us
        );
    }
    for p in &case.owned {
        g.free(*p).ok();
    }

    if compared == 0 {
        println!("FAIL - no element was compared; this run proves nothing.");
        std::process::exit(1);
    }
    if !fired {
        println!("FAIL - the negative control did not fire; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if failed > 0 {
        println!(
            "FAIL - {failed} case(s) differ. METRALE_GLM_DSA_MLA_HEADGROUP is NOT byte-identical \
             to the per-head kernel on this build; keep it off."
        );
        std::process::exit(1);
    }
    println!(
        "PASS - {compared} elements byte-identical: glm5next_dsa_mla_decode_fp8_hg2/4/8 match \
         the per-head kernel in every case."
    );
    Ok(())
}
