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
//! 2026-10-08: And the exact rewrite `glm5next_dsa_mla_prefill_tc2_fp8`
//! (`METRALE_GLM_MLA_PREFILL_TC2=1`) with its old-converter arm `_tc2_hwcvt_fp8`, both BITWISE
//! against the tensor-core kernel: the whole output buffer as raw u16, each arm run twice with
//! two sentinel fills (so written and unwritten elements must agree too). Cases: every case above
//! (`edge` also at 33 rows, the odd count), `nan` (an edge case whose pool holds the E4M3 NaN
//! codes on a few pages, so the converter's NaN fallback runs in place), and `ctx32k` / `ctx131k`:
//! the last 256 rows of a 32,768- and a 131,072-token prefill on one shared block table whose
//! pages are spread over a pool larger than GB10's 24 MB L2 (48 MB / 64 MB), each row selecting
//! 512 whole 4-token pools plus the 3 tail tokens, neighbouring rows sharing about 95% of their
//! pools. Also: all 256 E4M3 codes through the old chain, the TC2 converter and its fast path on
//! the device (`glm5next_dsa_mla_prefill_tc2_cvt_check`); a poisoned control (one cache byte of a
//! selected token changed for a TC2 run only must change the output); and CUDA-event timing of
//! the ctx cases (median of 15 after a warm call, arms alternated), printed as
//! `GATE timing tc2 131k old X ms new Y ms ratio R; 32k old X ms new Y ms ratio R` with
//! R = new / old (below 1 is faster). The final line starts `PASS` only when every check passed.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 on PASS, 1 on FAIL, 2 when the tensor-core kernel is absent from this target (an absent
//! TC2 entry point is a FAIL).
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_mla_prefill_tc_microtest

use anyhow::{Context, Result};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::Glm5NextDsaConfig;
use metrale_model_arch::glm5next_dsa::attend::{
    DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaDecodeKernel, MLA_PREFILL_TC2_CVT_CHECK_ENTRY,
    MLA_PREFILL_TC2_MODULE, decode_attention_headgroup, prefill_attention_tc,
    prefill_attention_tc2, prefill_attention_tc2_hwcvt,
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
// 2026-10-08: The second sentinel of the bitwise runs.
const POISON_ALT: u8 = 0xC3;
// 2026-10-08: Rows of the ctx cases (a production prefill sub-chunk) and CUDA-event samples.
const CTX_ROWS: usize = 256;
const EVENT_ITERS: usize = 15;
// 2026-10-08: Pool floor of the ctx cases: twice GB10's 24 MB L2.
const CTX_POOL_MIN_BYTES: usize = 48 << 20;

// 2026-10-08: CUDA driver event API for kernel-only timing, declared as in
// `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

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
    /// 2026-10-08: Random E4M3 codes, all 256 (NaN codes included).
    fn fp8_bytes_any(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() & 0xFF) as u8).collect()
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
    /// 2026-10-08: Host copies for the poisoned control: the selection rows and the block
    /// table(s) (`paging.max_blocks_per_seq` 0 = one shared table).
    sel: Vec<i32>,
    bt: Vec<i32>,
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
    // 2026-10-08: q first, then the pool, the draw order these seeds always had.
    let q_bytes = rng.bf16_bytes(seq_lens.len() * HEADS * KVL, q_amp);
    let pool_bytes = rng.fp8_bytes(n_phys * BLOCK * KVL);
    upload_pool(
        g,
        name,
        seq_lens,
        sel,
        bt,
        max_blocks,
        &q_bytes,
        &pool_bytes,
    )
}

/// 2026-10-08: [`upload`] with the q bytes (`rows * HEADS * KVL` BF16) and the pool bytes given.
#[allow(clippy::too_many_arguments)]
fn upload_pool(
    g: &dyn GpuBackend,
    name: String,
    seq_lens: Vec<i32>,
    sel: &[i32],
    bt: &[i32],
    max_blocks: usize,
    q_bytes: &[u8],
    pool_bytes: &[u8],
) -> Result<Case> {
    let rows = seq_lens.len();
    let width = sel.len() / rows;
    let q = up(g, q_bytes)?;
    let pool = up(g, pool_bytes)?;
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
        sel: sel.to_vec(),
        bt: bt.to_vec(),
    })
}

/// 2026-10-01: Random contexts with per-row tables and every kind of invalid index, as the
/// head-group parity harness builds them.
fn edge_case(g: &dyn GpuBackend, rows: usize, width: usize, seed: u64) -> Result<Case> {
    edge_case_nan(g, rows, width, seed, false)
}

/// 2026-10-08: [`edge_case`]; with `nan`, physical pages 0..`NAN_PAGES` hold all 256 E4M3 codes
/// (NaN codes included), so rows whose table maps one of them attend NaN keys and the TC2
/// converter's NaN fallback runs in place. Bitwise TC2 vs TC only.
fn edge_case_nan(
    g: &dyn GpuBackend,
    rows: usize,
    width: usize,
    seed: u64,
    nan: bool,
) -> Result<Case> {
    const NAN_PAGES: usize = 4;
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
    if !nan {
        let name = format!("edge rows={rows} W={width}");
        return upload(
            g, &mut rng, name, seq_lens, &sel, &bt, max_blocks, n_phys, amp,
        );
    }
    let page = BLOCK * KVL;
    let q = rng.bf16_bytes(rows * HEADS * KVL, amp);
    let mut pool = rng.fp8_bytes(n_phys * page);
    let any = rng.fp8_bytes_any(NAN_PAGES * page);
    pool[..NAN_PAGES * page].copy_from_slice(&any);
    let name = format!("nan rows={rows} W={width}");
    upload_pool(g, name, seq_lens, &sel, &bt, max_blocks, &q, &pool)
}

/// 2026-10-08: The last `CTX_ROWS` rows of a `seq`-token prefill (row r: `seq_len` =
/// seq - CTX_ROWS + r + 1) on one shared block table that maps each logical block to a distinct
/// random page of a pool of at least `CTX_POOL_MIN_BYTES`. Row 0 selects 512 random distinct
/// whole pools; each next row keeps the previous row's pools in place except about 5%, replaced
/// by new random pools (neighbouring rows overlap, as top-k sets do); every row then lists the
/// pools' 4 tokens and its last `TAIL` positions (`WIDTH` = 2051 entries).
fn ctx_case(g: &dyn GpuBackend, seq: usize, seed: u64) -> Result<Case> {
    let mut rng = Lcg(seed);
    let page = BLOCK * KVL;
    let n_logical = seq.div_ceil(BLOCK);
    let n_phys = n_logical.max(CTX_POOL_MIN_BYTES / page) + 8;
    let mut phys: Vec<i32> = (0..n_phys as i32).collect();
    for i in (1..phys.len()).rev() {
        phys.swap(i, rng.below(i + 1));
    }
    let bt: Vec<i32> = phys[..n_logical].to_vec();
    let n_sel_pools = (WIDTH - TAIL) / KPOOL;
    let first_sl = seq - CTX_ROWS + 1;
    let n_pools0 = (first_sl - TAIL) / KPOOL;
    let mut pools: Vec<usize> = Vec::with_capacity(n_sel_pools);
    let mut taken = std::collections::HashSet::new();
    while pools.len() < n_sel_pools {
        let p = rng.below(n_pools0);
        if taken.insert(p) {
            pools.push(p);
        }
    }
    let (mut seq_lens, mut sel) = (Vec::with_capacity(CTX_ROWS), Vec::new());
    for r in 0..CTX_ROWS {
        let sl = first_sl + r;
        seq_lens.push(sl as i32);
        if r > 0 {
            let n_pools = (sl - TAIL) / KPOOL;
            for slot in 0..n_sel_pools {
                if rng.below(100) < 5 {
                    let p = loop {
                        let p = rng.below(n_pools);
                        if !taken.contains(&p) {
                            break p;
                        }
                    };
                    taken.remove(&pools[slot]);
                    taken.insert(p);
                    pools[slot] = p;
                }
            }
        }
        for &p in &pools {
            sel.extend((p * KPOOL..p * KPOOL + KPOOL).map(|t| t as i32));
        }
        sel.extend((0..TAIL).map(|d| (sl - TAIL + d) as i32));
    }
    let q = rng.bf16_bytes(CTX_ROWS * HEADS * KVL, 1.0);
    let pool = rng.fp8_bytes(n_phys * page);
    let name = format!("ctx{}k rows={CTX_ROWS} W={WIDTH}", seq / 1024);
    upload_pool(g, name, seq_lens, &sel, &bt, 0, &q, &pool)
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

/// 2026-10-08: The kernels a launch can take.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// 2026-10-08: The decode kernel per head (`hg` 0) or per head group.
    Decode,
    /// 2026-10-08: `glm5next_dsa_mla_prefill_tc_fp8`.
    Tc,
    /// 2026-10-08: `glm5next_dsa_mla_prefill_tc2_fp8`.
    Tc2,
    /// 2026-10-08: `glm5next_dsa_mla_prefill_tc2_hwcvt_fp8`.
    Tc2Hw,
}

/// 2026-10-01: One launch of one arm (2026-10-08: as [`Arm`]).
fn launch_arm(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
    arm: Arm,
    hg: usize,
) -> Result<()> {
    let (c, gm) = (cfg(), geom(case.rows, case.width));
    let (p, i) = (&case.paging, &case.inputs);
    match arm {
        Arm::Decode => decode_attention_headgroup(g, kernel, hg, &c, &gm, p, i, 0),
        Arm::Tc => prefill_attention_tc(g, kernel, &c, &gm, p, i, 0),
        Arm::Tc2 => prefill_attention_tc2(g, kernel, &c, &gm, p, i, 0),
        Arm::Tc2Hw => prefill_attention_tc2_hwcvt(g, kernel, &c, &gm, p, i, 0),
    }
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
    launch_arm(g, kernel, case, if tc { Arm::Tc } else { Arm::Decode }, hg)
}

/// 2026-10-01: One arm's output, poisoned with `poison` before the launch.
fn run(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
    tc: bool,
    poison: u8,
) -> Result<Vec<u8>> {
    run_arm(
        g,
        kernel,
        case,
        if tc { Arm::Tc } else { Arm::Decode },
        poison,
    )
}

/// 2026-10-08: [`run`] for any [`Arm`].
fn run_arm(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
    arm: Arm,
    poison: u8,
) -> Result<Vec<u8>> {
    let bytes = case.rows * HEADS * KVL * 2;
    g.memset(case.inputs.out, poison, bytes)?;
    launch_arm(g, kernel, case, arm, 0)?;
    down(g, case.inputs.out, bytes)
}

/// 2026-10-08: Raw u16 comparison of two whole output buffers; prints one line, `BITWISE ... ok`
/// or `FAIL BITWISE ...` with the first difference (row, head, dim) and the count.
fn bitwise(case: &Case, label: &str, want: &[u8], got: &[u8]) -> bool {
    let n = want.len() / 2;
    let u = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let mut first = None;
    let mut diff = 0usize;
    for i in 0..n {
        if u(want, i) != u(got, i) {
            diff += 1;
            first.get_or_insert(i);
        }
    }
    match first {
        None if want.len() == got.len() => {
            println!(
                "BITWISE {:<24} {label:<22} identical ({n} u16) -> ok",
                case.name
            );
            true
        }
        None => {
            println!("FAIL BITWISE {} {label}: lengths differ", case.name);
            false
        }
        Some(i) => {
            let (row, head, dim) = (i / (HEADS * KVL), (i / KVL) % HEADS, i % KVL);
            println!(
                "FAIL BITWISE {:<24} {label:<22} {diff} of {n} u16 differ; first at row {row} \
                 head {head} dim {dim}: want 0x{:04x} got 0x{:04x}",
                case.name,
                u(want, i),
                u(got, i)
            );
            false
        }
    }
}

/// 2026-10-08: TC2 (and the hwcvt arm, when resolved) bitwise against the tensor-core kernel,
/// each with both sentinels; returns the number of failed comparisons.
fn tc2_bitwise(g: &dyn GpuBackend, kernel: Glm5NextDsaDecodeKernel, case: &Case) -> Result<usize> {
    let mut failed = 0usize;
    for poison in [POISON_TC, POISON_ALT] {
        let want = run_arm(g, kernel, case, Arm::Tc, poison)?;
        let got = run_arm(g, kernel, case, Arm::Tc2, poison)?;
        failed += usize::from(!bitwise(
            case,
            &format!("tc2 vs tc fill={poison:#04x}"),
            &want,
            &got,
        ));
        if kernel.has_prefill_tc2_hwcvt() {
            let got = run_arm(g, kernel, case, Arm::Tc2Hw, poison)?;
            let label = format!("tc2hw vs tc fill={poison:#04x}");
            failed += usize::from(!bitwise(case, &label, &want, &got));
        }
    }
    Ok(failed)
}

/// 2026-10-08: Poisoned control: the byte at dim 0 of row 0's first selected token is changed
/// (to 0x00, or 0x38 = 1.0 when it was 0x00) for one TC2 run only; that output must differ from
/// the clean tensor-core output somewhere, else the bitwise compare could be vacuous. The byte
/// is restored after. Returns whether the control behaved.
fn poisoned_control(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
) -> Result<bool> {
    let t = case.sel[0];
    let sl = case.seq_lens[0];
    if t < 0 || t >= sl {
        println!(
            "FAIL CONTROL {}: row 0's first selection {t} is not a valid token",
            case.name
        );
        return Ok(false);
    }
    let t = t as usize;
    let page = case.bt[t / BLOCK] as usize;
    let at = case
        .inputs
        .k_cache
        .offset(page * BLOCK * KVL + (t % BLOCK) * KVL);
    let mut orig = [0u8; 1];
    g.synchronize(0)?;
    g.copy_d2h(at, &mut orig)?;
    let flipped = if orig[0] == 0 { 0x38u8 } else { 0x00 };
    let clean = run_arm(g, kernel, case, Arm::Tc, POISON_TC)?;
    g.copy_h2d(&[flipped], at)?;
    let poisoned = run_arm(g, kernel, case, Arm::Tc2, POISON_TC);
    g.copy_h2d(&orig, at)?;
    g.synchronize(0)?;
    let poisoned = poisoned?;
    let differ = clean
        .chunks(2)
        .zip(poisoned.chunks(2))
        .filter(|(a, b)| a != b)
        .count();
    let ok = differ > 0;
    println!(
        "{}CONTROL {:<24} cache byte of token {t} dim 0 0x{:02x} -> 0x{flipped:02x} for TC2 only: \
         {differ} u16 differ from the clean TC output -> {}",
        if ok { "" } else { "FAIL " },
        case.name,
        orig[0],
        if ok {
            "ok (compare is not vacuous)"
        } else {
            "BAD (compare would be vacuous)"
        }
    );
    Ok(ok)
}

/// 2026-10-08: The E4M3 value of `c` (no NaN code) as an f32, from the format's definition.
fn e4m3_value(c: u8) -> f32 {
    let sign = if c & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let (e, m) = (((c >> 3) & 0xF) as i32, (c & 7) as f32);
    let v = if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    sign * v
}

/// 2026-10-08: All 256 E4M3 codes on the device through the old chain, the TC2 converter and its
/// fast path. Passes when the TC2 converter equals the old chain for all 256 (NaN codes
/// included), the fast path equals it for the 254 non-NaN codes, and the old chain gives each
/// non-NaN code's exact value (a check of the reference itself). Prints the NaN codes' bits.
fn converter_check(g: &dyn GpuBackend) -> Result<bool> {
    let Ok(h) = g.kernel(MLA_PREFILL_TC2_MODULE, MLA_PREFILL_TC2_CVT_CHECK_ENTRY) else {
        println!("FAIL CVT {MLA_PREFILL_TC2_CVT_CHECK_ENTRY} absent from this target");
        return Ok(false);
    };
    let out = g.alloc(384 * 4)?;
    g.memset(out, 0xEE, 384 * 4)?;
    KernelLaunch::new(g, h)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(out)
        .launch(0)?;
    let b = down(g, out, 384 * 4)?;
    g.free(out).ok();
    let half = |arr: usize, code: usize| {
        let w = arr * 128 + code / 2;
        let word = u32::from_le_bytes([b[4 * w], b[4 * w + 1], b[4 * w + 2], b[4 * w + 3]]);
        (word >> (16 * (code % 2))) as u16
    };
    let (mut bad_new, mut bad_fast, mut bad_ref) = (0usize, 0usize, 0usize);
    for code in 0..256usize {
        let (old, new, fast) = (half(0, code), half(1, code), half(2, code));
        let nan = code & 0x7F == 0x7F;
        if new != old {
            bad_new += 1;
            println!("FAIL CVT code 0x{code:02x}: tc2 0x{new:04x} old 0x{old:04x}");
        }
        if nan {
            println!(
                "CVT NaN code 0x{code:02x}: old chain 0x{old:04x}, tc2 converter 0x{new:04x} \
                 (fallback), fast path alone 0x{fast:04x} (not used for NaN groups)"
            );
            continue;
        }
        if fast != old {
            bad_fast += 1;
            println!("FAIL CVT code 0x{code:02x}: fast path 0x{fast:04x} old 0x{old:04x}");
        }
        let want = bf16::from_f32(e4m3_value(code as u8)).to_bits();
        if old != want {
            bad_ref += 1;
            println!("FAIL CVT code 0x{code:02x}: old chain 0x{old:04x}, exact value 0x{want:04x}");
        }
    }
    let ok = bad_new == 0 && bad_fast == 0 && bad_ref == 0;
    println!(
        "{}CVT all 256 E4M3 codes: tc2 converter vs old chain {bad_new} differ, fast path vs old \
         {bad_fast} of 254 differ, old vs exact value {bad_ref} of 254 differ -> {}",
        if ok { "" } else { "FAIL " },
        if ok { "ok" } else { "BAD" }
    );
    Ok(ok)
}

/// 2026-10-08: Kernel milliseconds of one launch of each arm by CUDA events: one warm call of
/// each, then `EVENT_ITERS` rounds alternating the arms; the median per arm.
fn event_times(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
    arms: &[Arm],
) -> Result<Vec<f64>> {
    let (mut e0, mut e1) = (0u64, 0u64);
    let rc = unsafe { cuEventCreate(&mut e0, 0) } | unsafe { cuEventCreate(&mut e1, 0) };
    if rc != 0 {
        anyhow::bail!("cuEventCreate failed: status {rc}");
    }
    for &arm in arms {
        launch_arm(g, kernel, case, arm, 0)?;
    }
    g.synchronize(0)?;
    let mut samples = vec![Vec::with_capacity(EVENT_ITERS); arms.len()];
    for it in 0..EVENT_ITERS {
        for k in 0..arms.len() {
            // 2026-10-08: Alternate which arm goes first, so neither always follows the other.
            let a = if it % 2 == 0 { k } else { arms.len() - 1 - k };
            let mut ms = 0f32;
            let rc = unsafe { cuEventRecord(e0, 0) };
            launch_arm(g, kernel, case, arms[a], 0)?;
            let rc = rc
                | unsafe { cuEventRecord(e1, 0) }
                | unsafe { cuEventSynchronize(e1) }
                | unsafe { cuEventElapsedTime(&mut ms, e0, e1) };
            if rc != 0 {
                anyhow::bail!("CUDA event timing failed: status {rc}");
            }
            samples[a].push(ms as f64);
        }
    }
    unsafe {
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    Ok(samples
        .into_iter()
        .map(|mut v| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        })
        .collect())
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

/// 2026-10-08: [`time_arm`] for any [`Arm`] (host clock, mean of `ITERS` after 2 warm-ups).
fn time_arm_of(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
    arm: Arm,
) -> Result<f64> {
    for _ in 0..2 {
        launch_arm(g, kernel, case, arm, 0)?;
    }
    g.synchronize(0)?;
    let t = std::time::Instant::now();
    for _ in 0..ITERS {
        launch_arm(g, kernel, case, arm, 0)?;
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
    let kernel = Glm5NextDsaDecodeKernel::resolve(g)
        .context("Glm5NextDsaDecodeKernel")?
        .with_prefill_tc2(g);
    if !kernel.has_prefill_tc() {
        println!("glm5next_dsa_mla_prefill_tc_fp8 absent from this target - SKIP");
        std::process::exit(2);
    }
    if !kernel.has_prefill_tc2() {
        println!(
            "FAIL - glm5next_dsa_mla_prefill_tc2_fp8 absent from this target (module \
             {MLA_PREFILL_TC2_MODULE} not built or the entry did not resolve)"
        );
        std::process::exit(1);
    }
    if !kernel.has_prefill_tc2_hwcvt() {
        println!("NOTE glm5next_dsa_mla_prefill_tc2_hwcvt_fp8 absent: its arm is skipped");
    }

    let mut failed = 0usize;
    // 2026-10-08: Bitwise failures and control failures are counted apart from the numeric ones.
    let mut bit_failed = 0usize;
    bit_failed += usize::from(!converter_check(g)?);

    let cases = [
        edge_case(g, 7, 257, 0x5EED_0001)?,
        edge_case(g, 64, WIDTH, 0x5EED_0002)?,
        edge_case(g, 64, WIDTH, 0x5EED_0003)?,
        edge_case(g, 33, WIDTH, 0x5EED_0004)?,
    ];
    for case in cases {
        let r = run(g, kernel, &case, false, POISON_REF)?;
        let t = run(g, kernel, &case, true, POISON_TC)?;
        failed += usize::from(!compare(&case, &r, &t));
        bit_failed += tc2_bitwise(g, kernel, &case)?;
        for p in &case.owned {
            g.free(*p).ok();
        }
    }

    // 2026-10-08: NaN codes in the pool: TC2 vs TC bitwise only (both write NaN rows).
    {
        let case = edge_case_nan(g, 64, WIDTH, 0x5EED_0005, true)?;
        bit_failed += tc2_bitwise(g, kernel, &case)?;
        for p in &case.owned {
            g.free(*p).ok();
        }
    }

    // 2026-10-08: The long-context sub-chunks: bitwise, the poisoned control (on 32K), timing.
    let mut gate = Vec::new();
    for (seq, seed) in [(131_072usize, 0xC7C_0131u64), (32_768, 0xC7C_0032)] {
        let case = ctx_case(g, seq, seed)?;
        bit_failed += tc2_bitwise(g, kernel, &case)?;
        if seq == 32_768 {
            bit_failed += usize::from(!poisoned_control(g, kernel, &case)?);
        }
        let mut arms = vec![Arm::Tc, Arm::Tc2];
        if kernel.has_prefill_tc2_hwcvt() {
            arms.push(Arm::Tc2Hw);
        }
        let ms = event_times(g, kernel, &case, &arms)?;
        println!(
            "TIMING {:<24} CUDA events, median of {EVENT_ITERS}: tc {:.4} ms  tc2 {:.4} ms{}",
            case.name,
            ms[0],
            ms[1],
            ms.get(2)
                .map_or(String::new(), |h| format!("  tc2hw {h:.4} ms"))
        );
        gate.push((seq / 1024, ms[0], ms[1]));
        for p in &case.owned {
            g.free(*p).ok();
        }
    }
    let part = |&(k, old, new): &(usize, f64, f64)| {
        format!(
            "{k}k old {old:.4} ms new {new:.4} ms ratio {:.4}",
            new / old
        )
    };
    println!("GATE timing tc2 {}; {}", part(&gate[0]), part(&gate[1]));
    println!("(ratio = new / old; below 1 is faster)");

    for (rows, seed) in [(4096usize, 0x7135_4096u64), (8192, 0x7135_8192)] {
        let case = prefill_case(g, rows, seed)?;
        let r = run(g, kernel, &case, false, POISON_REF)?;
        let t = run(g, kernel, &case, true, POISON_TC)?;
        failed += usize::from(!compare(&case, &r, &t));
        drop((r, t));
        bit_failed += tc2_bitwise(g, kernel, &case)?;
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
        label("tc2", time_arm_of(g, kernel, &case, Arm::Tc2)?);
        for p in &case.owned {
            g.free(*p).ok();
        }
    }

    if bit_failed > 0 {
        println!(
            "FAIL - {bit_failed} TC2 bitwise / converter / control check(s) failed. Keep \
             METRALE_GLM_MLA_PREFILL_TC2 off on this build."
        );
    }
    if failed > 0 {
        println!(
            "FAIL - {failed} case(s) outside cos > {COS_MIN} / max_err <= {MAX_ERR_REL} x ref_max \
             (or non-finite, or a seq_len-0 / empty row written wrong). Keep \
             METRALE_GLM_MLA_PREFILL_TC off on this build."
        );
    }
    if failed > 0 || bit_failed > 0 {
        std::process::exit(1);
    }
    println!(
        "PASS - glm5next_dsa_mla_prefill_tc2_fp8 (and the hwcvt arm when built) is bitwise \
         identical to glm5next_dsa_mla_prefill_tc_fp8 in every case, the converter matches the \
         old chain on all 256 codes, the poisoned control differs; and \
         glm5next_dsa_mla_prefill_tc_fp8 matches the decode kernel within cos > {COS_MIN} and \
         max_err <= {MAX_ERR_REL} x ref_max in every case (numerics differ by design; \
         model-level quality is gated separately)."
    );
    Ok(())
}
