// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Accuracy and timing gate for the tensor-core DSA MLA prefill kernel
//! (`glm5next_dsa_mla_prefill_tc_fp8`, `METRALE_GLM_MLA_PREFILL_TC=1`) against the decode kernel
//! prefill runs today (`glm5next_dsa_mla_decode_fp8`, per head; its head-grouped variants are
//! byte-identical to it), both through the `attend` launchers.
//!
//! The tensor-core kernel is NOT byte-identical by design (BF16 P, a different FP32 accumulation
//! order), so the gate is numeric: over every row with `seq_len > 0`, cosine similarity above
//! `COS_MIN` and a maximum absolute error at most `MAX_ERR_REL` of the reference's largest
//! magnitude, no non-finite output, rows with `seq_len` 0 left untouched by both kernels, and rows
//! with no valid index written as zeros. Max and mean absolute error are printed for every case.
//!
//! Inputs, GLM-5.3 TP2 shapes (32 heads per rank, 512-wide latent, selection 2051, pages of 64):
//! random BF16 q, a random FP8 pool (no NaN codes) with a non-power-of-two scale, and
//! * `edge`: 7 and 64 rows over random contexts up to 16K with per-row block tables, selections
//!   carrying `-1` holes, other negative and past-`seq_len` indices and duplicates; row 1 has
//!   `seq_len` 0 and row 2 no valid index;
//! * `prefill`: the rows of one causal prefill from position 0 (4096 and 8192 rows, `seq_len` =
//!   position + 1) on one shared block table, as `decode_rows_batched` launches them, each row
//!   selecting whole 4-token pools plus the 3 tail tokens (every position while `seq_len` <= 2051).
//!   These two are also timed: host clock around `ITERS` launches after 2 warm-ups.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 on PASS, 1 on FAIL, 2 when a kernel is absent from this target.
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_mla_prefill_tc_microtest

use anyhow::{Context, Result};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::Glm5NextDsaConfig;
use metrale_model_arch::glm5next_dsa::attend::{
    DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaDecodeKernel, decode_attention_headgroup,
    prefill_attention_tc,
};
use metrale_model_arch::glm5next_dsa::select::DsaSelectGeometry;

const HEADS: usize = 32;
const KVL: usize = 512;
const BLOCK: usize = 64;
const EDGE_CTX: usize = 16_384;
const KPOOL: usize = 4;
const TAIL: usize = 3;
const WIDTH: usize = 2051;
const K_SCALE: f32 = 0.0173;
const POISON_REF: u8 = 0xA5;
const POISON_TC: u8 = 0x5A;
const COS_MIN: f64 = 0.9995;
const MAX_ERR_REL: f64 = 0.02;
const ITERS: usize = 5;

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
        index_kpool: KPOOL,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: HEADS,
        q_lora_rank: 1536,
        kv_lora_rank: KVL,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: EDGE_CTX,
    }
}

/// 2026-10-01: The attention launchers read only `q_rows` and `out_width` from the geometry.
fn geom(rows: usize, width: usize) -> DsaSelectGeometry {
    DsaSelectGeometry {
        seq: EDGE_CTX,
        q_rows: rows,
        n_pools_full: 0,
        n_pools: 0,
        select_k: 0,
        out_width: width,
        topk_np2: 0,
        topk_smem: 0,
        index_head_dim: 128,
        index_heads: 32,
        index_kpool: KPOOL,
    }
}

/// 2026-10-01: One case's device inputs and launch geometry, with the host copies the checks
/// need.
struct Case {
    name: String,
    rows: usize,
    width: usize,
    seq_lens: Vec<i32>,
    paging: DsaDecodePaging,
    inputs: DsaDecodeInputs,
    owned: Vec<DevicePtr>,
}

/// 2026-10-01: Uploads a case: `bt` is either `rows * max_blocks` entries or one shared table
/// (`max_blocks == 0`), `n_phys` physical pages in the pool.
#[allow(clippy::too_many_arguments)]
fn upload(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    name: String,
    seq_lens: Vec<i32>,
    sel: &[i32],
    bt: &[i32],
    max_blocks: usize,
    n_phys: usize,
    q_amp: f32,
) -> Result<Case> {
    let rows = seq_lens.len();
    let width = sel.len() / rows;
    let q = up(g, &rng.bf16_bytes(rows * HEADS * KVL, q_amp))?;
    let pool = up(g, &rng.fp8_bytes(n_phys * BLOCK * KVL))?;
    let out = g.alloc(rows * HEADS * KVL * 2)?;
    let block_tables = up(g, &i32_bytes(bt))?;
    let seq_lens_d = up(g, &i32_bytes(&seq_lens))?;
    let sel_indices = up(g, &i32_bytes(sel))?;
    Ok(Case {
        name,
        rows,
        width,
        seq_lens,
        paging: DsaDecodePaging {
            num_seqs: rows,
            num_q_heads: HEADS,
            num_kv_heads: 1,
            max_blocks_per_seq: max_blocks,
            block_size: BLOCK,
            cache_stride_bytes: (BLOCK * KVL) as u64,
        },
        inputs: DsaDecodeInputs {
            q,
            k_cache: pool,
            v_cache: pool,
            out,
            block_tables,
            seq_lens: seq_lens_d,
            sel_indices,
            k_scale: K_SCALE,
            v_scale: K_SCALE,
        },
        owned: vec![q, pool, out, block_tables, seq_lens_d, sel_indices],
    })
}

/// 2026-10-01: Random contexts with per-row tables and every kind of invalid index, as the
/// head-group parity harness builds them.
fn edge_case(g: &dyn GpuBackend, rows: usize, width: usize, seed: u64) -> Result<Case> {
    let mut rng = Lcg(seed);
    let max_blocks = EDGE_CTX / BLOCK;
    let n_phys = max_blocks + 32;
    let (mut seq_lens, mut sel, mut bt) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..rows {
        let sl = if r == 1 { 0 } else { 1 + rng.below(EDGE_CTX) };
        seq_lens.push(sl as i32);
        bt.extend((0..max_blocks).map(|_| rng.below(n_phys) as i32));
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
    let amp = if seed & 1 == 0 { 1.0 } else { 4.0 };
    let name = format!("edge rows={rows} W={width}");
    upload(g, &mut rng, name, seq_lens, &sel, &bt, max_blocks, n_phys, amp)
}

/// 2026-10-01: One causal prefill of `rows` tokens from position 0 on a shared table. Row r
/// (`seq_len` r + 1) selects every position while `seq_len` <= `WIDTH`, else whole random pools
/// (duplicates possible) and the last `TAIL` positions.
fn prefill_case(g: &dyn GpuBackend, rows: usize, seed: u64) -> Result<Case> {
    let mut rng = Lcg(seed);
    let n_phys = rows.div_ceil(BLOCK) + 8;
    // 2026-10-01: A random page order, so neighbouring blocks are not neighbouring pages.
    let mut bt: Vec<i32> = (0..n_phys as i32).collect();
    for i in (1..bt.len()).rev() {
        bt.swap(i, rng.below(i + 1));
    }
    let mut seq_lens = Vec::with_capacity(rows);
    let mut sel = Vec::with_capacity(rows * WIDTH);
    for r in 0..rows {
        let sl = r + 1;
        seq_lens.push(sl as i32);
        if sl <= WIDTH {
            sel.extend((0..WIDTH).map(|k| if k < sl { k as i32 } else { -1 }));
            continue;
        }
        let n_pools = sl.div_ceil(KPOOL);
        let mut k = 0;
        while k < WIDTH - TAIL {
            let base = rng.below(n_pools) * KPOOL;
            let take = KPOOL.min(WIDTH - TAIL - k);
            sel.extend((base..base + take).map(|t| if t < sl { t as i32 } else { -1 }));
            k += take;
        }
        sel.extend((0..TAIL).map(|d| (sl - TAIL + d) as i32));
    }
    let name = format!("prefill rows={rows} W={WIDTH}");
    upload(g, &mut rng, name, seq_lens, &sel, &bt, 0, n_phys, 1.0)
}

/// 2026-10-01: One launch of one arm: `tc` selects the tensor-core kernel, else the decode
/// kernel per head (`hg` 0) or per head group.
fn launch(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
    tc: bool,
    hg: usize,
) -> Result<()> {
    let (c, gm) = (cfg(), geom(case.rows, case.width));
    if tc {
        prefill_attention_tc(g, kernel, &c, &gm, &case.paging, &case.inputs, 0)
    } else {
        decode_attention_headgroup(g, kernel, hg, &c, &gm, &case.paging, &case.inputs, 0)
    }
}

/// 2026-10-01: One arm's output, poisoned with `poison` before the launch.
fn run(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
    tc: bool,
    poison: u8,
) -> Result<Vec<u8>> {
    let bytes = case.rows * HEADS * KVL * 2;
    g.memset(case.inputs.out, poison, bytes)?;
    launch(g, kernel, case, tc, 0)?;
    down(g, case.inputs.out, bytes)
}

fn bf16_at(b: &[u8], i: usize) -> f64 {
    bf16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f64()
}

/// 2026-10-01: Compares the two arms; returns whether the case passes.
fn compare(case: &Case, r: &[u8], t: &[u8]) -> bool {
    let row_elems = HEADS * KVL;
    let (mut dot, mut nr, mut nt) = (0f64, 0f64, 0f64);
    let (mut max_err, mut sum_err, mut ref_max) = (0f64, 0f64, 0f64);
    let (mut compared, mut nonfinite, mut layout_ok) = (0usize, 0usize, true);
    for (row, &sl) in case.seq_lens.iter().enumerate() {
        let span = row * row_elems * 2..(row + 1) * row_elems * 2;
        if sl == 0 {
            layout_ok &= r[span.clone()].iter().all(|&x| x == POISON_REF)
                && t[span].iter().all(|&x| x == POISON_TC);
            continue;
        }
        let mut ref_zero = true;
        for i in row * row_elems..(row + 1) * row_elems {
            let (a, b) = (bf16_at(r, i), bf16_at(t, i));
            if !a.is_finite() || !b.is_finite() {
                nonfinite += 1;
                continue;
            }
            ref_zero &= a == 0.0;
            dot += a * b;
            nr += a * a;
            nt += b * b;
            let e = (a - b).abs();
            max_err = max_err.max(e);
            sum_err += e;
            ref_max = ref_max.max(a.abs());
            compared += 1;
        }
        // 2026-10-01: A row the reference wrote as zeros (no valid index) must be zeros here too.
        if ref_zero {
            let tc_row = &t[span];
            layout_ok &= tc_row.chunks(2).all(|h| (u16::from_le_bytes([h[0], h[1]]) & 0x7FFF) == 0);
        }
    }
    let cos = if nr > 0.0 && nt > 0.0 {
        dot / (nr.sqrt() * nt.sqrt())
    } else {
        0.0
    };
    let mean_err = sum_err / compared.max(1) as f64;
    let rel = max_err / ref_max.max(f64::MIN_POSITIVE);
    let pass = compared > 0 && nonfinite == 0 && layout_ok && cos > COS_MIN && rel <= MAX_ERR_REL;
    println!(
        "CASE {:<24} elems={compared:<10} cos={cos:.7} max_abs_err={max_err:.3e} \
         mean_abs_err={mean_err:.3e} ref_max={ref_max:.3e} max_err/ref_max={rel:.2e} \
         nonfinite={nonfinite} layout_ok={layout_ok} -> {}",
        case.name,
        if pass { "ok" } else { "BAD" }
    );
    pass
}

/// 2026-10-01: Mean microseconds per launch of one arm.
fn time_arm(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
    tc: bool,
    hg: usize,
) -> Result<f64> {
    for _ in 0..2 {
        launch(g, kernel, case, tc, hg)?;
    }
    g.synchronize(0)?;
    let t = std::time::Instant::now();
    for _ in 0..ITERS {
        launch(g, kernel, case, tc, hg)?;
    }
    g.synchronize(0)?;
    Ok(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
}

fn main() -> Result<()> {
    // 2026-10-01: The kernels are in the glm-5.3-flash target; `ptx_modules()` aliases the first
    // compiled target, so the backend is built from that set (as `glm5next_dsa_decode_gate`).
    let sets = metrale_kernels::all_ptx_sets();
    let Some(glm) = sets.iter().find(|s| s.target.model == "glm-5.3-flash") else {
        println!("glm-5.3-flash kernel target not built - SKIP");
        std::process::exit(2);
    };
    let backend = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &backend;
    let kernel = Glm5NextDsaDecodeKernel::resolve(g).context("Glm5NextDsaDecodeKernel")?;
    if !kernel.has_prefill_tc() {
        println!("glm5next_dsa_mla_prefill_tc_fp8 absent from this target - SKIP");
        std::process::exit(2);
    }

    let mut failed = 0usize;
    let cases = [
        edge_case(g, 7, 257, 0x5EED_0001)?,
        edge_case(g, 64, WIDTH, 0x5EED_0002)?,
        edge_case(g, 64, WIDTH, 0x5EED_0003)?,
    ];
    for case in cases {
        let r = run(g, kernel, &case, false, POISON_REF)?;
        let t = run(g, kernel, &case, true, POISON_TC)?;
        failed += usize::from(!compare(&case, &r, &t));
        for p in &case.owned {
            g.free(*p).ok();
        }
    }

    for (rows, seed) in [(4096usize, 0x7135_4096u64), (8192, 0x7135_8192)] {
        let case = prefill_case(g, rows, seed)?;
        let r = run(g, kernel, &case, false, POISON_REF)?;
        let t = run(g, kernel, &case, true, POISON_TC)?;
        failed += usize::from(!compare(&case, &r, &t));
        let per_head = time_arm(g, kernel, &case, false, 0)?;
        let label = |name: &str, us: f64| {
            println!(
                "TIMING rows={rows} W={WIDTH} heads={HEADS} {name:<9} {us:>12.1} us/launch  \
                 x{:.2} vs per-head",
                per_head / us
            );
        };
        label("per-head", per_head);
        // 2026-10-01: hg8, the fastest decode-kernel arm, when this target has it.
        if kernel.has_headgroup(8) {
            label("hg8", time_arm(g, kernel, &case, false, 8)?);
        }
        label("tc", time_arm(g, kernel, &case, true, 0)?);
        for p in &case.owned {
            g.free(*p).ok();
        }
    }

    if failed > 0 {
        println!(
            "FAIL - {failed} case(s) outside cos > {COS_MIN} / max_err <= {MAX_ERR_REL} x ref_max \
             (or non-finite, or a seq_len-0 / empty row written wrong). Keep \
             METRALE_GLM_MLA_PREFILL_TC off on this build."
        );
        std::process::exit(1);
    }
    println!(
        "PASS - glm5next_dsa_mla_prefill_tc_fp8 matches the decode kernel within cos > {COS_MIN} \
         and max_err <= {MAX_ERR_REL} x ref_max in every case (numerics differ by design; \
         model-level quality is gated separately)."
    );
    Ok(())
}
