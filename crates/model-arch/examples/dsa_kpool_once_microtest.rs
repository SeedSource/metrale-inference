// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Byte-parity gate for `METRALE_GLM_DSA_KPOOL_ONCE`: compressing the DSA indexer
//! pools once per prefill window (over the window's end length, its keys all valid) against
//! compressing them from key 0 for every `core_rows` selection sub-chunk (over that
//! sub-chunk's end length, the keys past it not yet valid), as `decode_k_wide` does without
//! the lever.
//!
//! * Pool leg: `dsa_kpool_compress` launched directly with `select_tokens`' arguments. For every
//!   sub-chunk `j` (end length `S_j`, `P_j = S_j / 4` complete pools) the `pool_keys` (f32, as
//!   u32 bit patterns), `pool_indices` and `pool_valid` of the first `P_j` pools of the
//!   once-per-window compress are compared byte for byte with the per-sub-chunk compress's. The
//!   trailing partial pool of the per-sub-chunk compress is not compared: it is written
//!   invalid, and no later kernel reads it (`P = n_pools`). Both arms start from a poison fill,
//!   so a pool an arm leaves unwritten cannot match.
//! * Selection leg: the production entry points. Per sub-chunk `select_tokens` (compress
//!   inside) against `compress_window` once + `select_tokens_pooled` per sub-chunk, same
//!   scratch, `[n, out_width]` token ids compared byte for byte (the window's keys are valid
//!   before the first selection in the once arm, and only through `S_j` in the other, so a
//!   read of a key past the row would show).
//! * Windows (prior length, rows): (0, 8192), (8192, 8192), (16421, 8192) (a prior length that
//!   is not a multiple of the pool width, so a pool straddles two windows), (8192, 1003) and
//!   (123, 777), `core_rows` 256, GLM-5.3 H = 32, D = 128, KP = 4. Rows below the prior length
//!   include invalid keys (every 211th), so some pools are invalid.
//! * Controls that must fire: one flipped key bit in the once arm's input (pool leg and
//!   selection leg inputs are compared; the pool comparator must see it), and a run that
//!   compared nothing, or whose selections hold no token, fails.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and a final `PASS` line when everything is identical and the control fires; 1 and
//! `FAIL` otherwise.
//!
//! Run (the kernel is a gb10 common kernel):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_kpool_once_microtest

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::select::pool_once::{compress_window, select_tokens_pooled};
use metrale_model_arch::glm5next_dsa::select::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, DsaSelectScratch, select_tokens,
};
use metrale_model_arch::glm5next_dsa::{DSA_MODULE, Glm5NextDsaConfig, Glm5NextDsaKernels};

const H: usize = 32;
const D: usize = 128;
const KP: usize = 4;
const CORE: usize = 256;
const MAX_ROWS: usize = 8192;
/// 2026-10-07: Indexer cache rows: the longest window end (16,421 + 8,192) with room to spare.
const MAX_CTX: usize = 32_768;
/// 2026-10-07: (prior length, window rows).
const WINDOWS: &[(usize, usize)] = &[
    (0, 8192),
    (8192, 8192),
    (16_421, 8192),
    (8192, 1003),
    (123, 777),
];
const POISON_A: u8 = 0xA5;
const POISON_B: u8 = 0x5A;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
    fn bf16_bytes(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f32(self.r(lo, hi)).to_bits().to_le_bytes())
            .collect()
    }
    fn f32_bytes(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u8> {
        (0..n).flat_map(|_| self.r(lo, hi).to_le_bytes()).collect()
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
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
        index_heads: H,
        index_head_dim: D,
        index_kpool: KP,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 32,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: MAX_CTX,
    }
}

/// 2026-10-07: The three pool regions at capacity `pools`: keys f32, indices i32, validity u8.
#[derive(Clone, Copy)]
struct Pools {
    keys: DevicePtr,
    indices: DevicePtr,
    valid: DevicePtr,
    cap: usize,
}

impl Pools {
    fn new(g: &dyn GpuBackend, cap: usize) -> Result<Self> {
        Ok(Self {
            keys: g.alloc(cap * D * 4)?,
            indices: g.alloc(cap * KP * 4)?,
            valid: g.alloc(cap)?,
            cap,
        })
    }
    fn fill(&self, g: &dyn GpuBackend, byte: u8) -> Result<()> {
        g.memset_async(self.keys, byte, self.cap * D * 4, 0)?;
        g.memset_async(self.indices, byte, self.cap * KP * 4, 0)?;
        g.memset_async(self.valid, byte, self.cap, 0)
    }
    /// 2026-10-07: The first `p` pools' bytes: (keys, indices, validity).
    fn read(&self, g: &dyn GpuBackend, p: usize) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        Ok((
            down(g, self.keys, p * D * 4)?,
            down(g, self.indices, p * KP * 4)?,
            down(g, self.valid, p)?,
        ))
    }
}

/// 2026-10-07: Everything the legs read.
struct Rig {
    compress: KernelHandle,
    ape: DevicePtr,
    k_host: Vec<u8>,
    k: DevicePtr,
    gate: DevicePtr,
    valid: DevicePtr,
    q: DevicePtr,
    weights: DevicePtr,
    q_pos: DevicePtr,
    q_mask: DevicePtr,
}

impl Rig {
    /// 2026-10-07: `dsa_kpool_compress` as `select_tokens` launches it on an exact launch.
    fn compress(&self, g: &dyn GpuBackend, k: DevicePtr, out: &Pools, seq: usize) -> Result<()> {
        KernelLaunch::new(g, self.compress)
            .grid([seq.div_ceil(KP) as u32, 1, 1])
            .block([D as u32, 1, 1])
            .arg_ptr(k)
            .arg_ptr(self.gate)
            .arg_ptr(self.valid)
            .arg_ptr(self.ape)
            .arg_ptr(out.keys)
            .arg_ptr(out.indices)
            .arg_ptr(out.valid)
            .arg_u32(seq as u32)
            .arg_u32(D as u32)
            .arg_u32(KP as u32)
            .arg_i32(0)
            .arg_ptr(DevicePtr::NULL)
            .launch(0)
    }

    /// 2026-10-07: `valid` holds the invalid-key pattern through `end` and zeros past it.
    fn set_valid(&self, g: &dyn GpuBackend, pat: &[u8], end: usize) -> Result<()> {
        let mut v = vec![0u8; MAX_CTX];
        v[..end].copy_from_slice(&pat[..end]);
        g.copy_h2d(&v, self.valid)
    }

    fn inputs(&self, t0: usize, k: DevicePtr) -> DsaSelectInputs {
        DsaSelectInputs {
            k_normed: k,
            gate: self.gate,
            valid: self.valid,
            ape: self.ape,
            q: self.q.offset(t0 * H * D * 4),
            weights: self.weights.offset(t0 * H * 4),
            q_pos: self.q_pos.offset(t0 * 4),
            q_mask: self.q_mask,
            first_key: 0,
            geom_dev: DevicePtr::NULL,
            pool_cache: None,
        }
    }
}

#[derive(Default)]
struct Tally {
    bytes: usize,
    mismatches: usize,
    selected: usize,
}

impl Tally {
    fn same(&mut self, a: &[u8], b: &[u8]) -> bool {
        self.bytes += a.len().min(b.len());
        let ok = a == b;
        if !ok {
            self.mismatches += 1;
        }
        ok
    }
}

fn subs(rows: usize) -> Vec<(usize, usize)> {
    (0..rows)
        .step_by(CORE)
        .map(|t| (t, CORE.min(rows - t)))
        .collect()
}

/// 2026-10-07: One window, both legs. Returns whether everything matched.
fn window(
    g: &dyn GpuBackend,
    r: &Rig,
    kern: &Glm5NextDsaKernels,
    scratch: &DsaSelectScratch,
    pat: &[u8],
    (l0, rows): (usize, usize),
    t: &mut Tally,
) -> Result<bool> {
    let c = cfg();
    let end = l0 + rows;
    let cap = end.div_ceil(KP);
    let mut ok = true;

    // Pool leg. Once arm: the window's keys valid, one compress over the window's end.
    let once = Pools::new(g, cap)?;
    once.fill(g, POISON_B)?;
    r.set_valid(g, pat, end)?;
    r.compress(g, r.k, &once, end)?;
    let all = once.read(g, cap)?;
    let per = Pools::new(g, cap)?;
    let mut legs = 0;
    for &(t0, n) in &subs(rows) {
        let sj = l0 + t0 + n;
        let pj = sj / KP;
        per.fill(g, POISON_A)?;
        r.set_valid(g, pat, sj)?;
        r.compress(g, r.k, &per, sj)?;
        let (pk, pi, pv) = per.read(g, pj)?;
        ok &= t.same(&pk, &all.0[..pj * D * 4]);
        ok &= t.same(&pi, &all.1[..pj * KP * 4]);
        ok &= t.same(&pv, &all.2[..pj]);
        legs += 1;
    }

    // Selection leg. Per sub-chunk `select_tokens` against `compress_window` + `_pooled`.
    let width = c.out_width();
    let mut want = Vec::new();
    for &(t0, n) in &subs(rows) {
        let sj = l0 + t0 + n;
        r.set_valid(g, pat, sj)?;
        let geom = DsaSelectGeometry::plan(&c, sj, n)?;
        g.memset_async(scratch.tokens(), POISON_A, n * width * 4, 0)?;
        let inp = r.inputs(t0, r.k);
        select_tokens(g, kern, &c, &geom, &inp, scratch, DsaSelectLaunch::Exact, 0)?;
        want.push(down(g, scratch.tokens(), n * width * 4)?);
    }
    r.set_valid(g, pat, end)?;
    let wg = DsaSelectGeometry::plan(&c, end, 1)?;
    compress_window(g, kern, &c, &wg, &r.inputs(0, r.k), scratch, 0)?;
    for (&(t0, n), w) in subs(rows).iter().zip(&want) {
        let sj = l0 + t0 + n;
        let geom = DsaSelectGeometry::plan(&c, sj, n)?;
        g.memset_async(scratch.tokens(), POISON_B, n * width * 4, 0)?;
        select_tokens_pooled(g, kern, &c, &geom, &r.inputs(t0, r.k), scratch, 0)?;
        let got = down(g, scratch.tokens(), n * width * 4)?;
        ok &= t.same(w, &got);
        t.selected += got
            .chunks_exact(4)
            .filter(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) >= 0)
            .count();
    }
    println!(
        "window prior={l0} rows={rows}: {legs} sub-chunks, {cap} pools at the end: {}",
        if ok { "identical" } else { "MISMATCH" }
    );
    Ok(ok)
}

/// 2026-10-07: The comparator must see one flipped bit in one pool's key: compress the window
/// with the bit flipped and compare it with the clean once-per-window pools.
fn control(g: &dyn GpuBackend, r: &Rig, pat: &[u8]) -> Result<bool> {
    let end = 8192 + 256;
    let cap = end / KP;
    r.set_valid(g, pat, end)?;
    let clean = Pools::new(g, cap)?;
    clean.fill(g, POISON_A)?;
    r.compress(g, r.k, &clean, end)?;
    let mut flipped = r.k_host.clone();
    flipped[(1000 * D + 3) * 2] ^= 0x40;
    let k2 = up(g, &flipped)?;
    let bad = Pools::new(g, cap)?;
    bad.fill(g, POISON_A)?;
    r.compress(g, k2, &bad, end)?;
    let (a, b) = (clean.read(g, cap)?, bad.read(g, cap)?);
    let fired = a.0 != b.0;
    println!("control (one flipped key bit in pool 250): detected={fired}");
    Ok(fired)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kern = Glm5NextDsaKernels::resolve(g)?;
    let c = cfg();
    let mut rng = Lcg(0x4B50_4F4F_4C31);
    let k_host = rng.bf16_bytes(MAX_CTX * D, -2.0, 2.0);
    let rig = Rig {
        compress: g.kernel(DSA_MODULE, "dsa_kpool_compress")?,
        ape: up(g, &rng.f32_bytes(KP * D, -0.5, 0.5))?,
        k: up(g, &k_host)?,
        k_host,
        gate: up(g, &rng.bf16_bytes(MAX_CTX * D, -2.0, 2.0))?,
        valid: g.alloc(MAX_CTX)?,
        q: up(g, &rng.f32_bytes(MAX_ROWS * H * D, -1.0, 1.0))?,
        weights: up(g, &rng.f32_bytes(MAX_ROWS * H, -1.0, 1.0))?,
        q_pos: g.alloc(MAX_ROWS * 4)?,
        q_mask: {
            let p = g.alloc(MAX_ROWS)?;
            g.memset_async(p, 1, MAX_ROWS, 0)?;
            p
        },
    };
    let geom = DsaSelectGeometry::plan(&c, MAX_CTX, CORE)?;
    let scratch = DsaSelectScratch::alloc(g, &c, &geom)?;
    // 2026-10-07: Keys below the longest prior length: every 211th invalid.
    let pat: Vec<u8> = (0..MAX_CTX)
        .map(|i| u8::from(!(i < 16_421 && i % 211 == 5)))
        .collect();

    let mut t = Tally::default();
    let mut all_ok = true;
    for &(l0, rows) in WINDOWS {
        let q_pos: Vec<u8> = (0..MAX_ROWS)
            .flat_map(|i| ((l0 + i) as i32).to_le_bytes())
            .collect();
        g.copy_h2d(&q_pos, rig.q_pos)?;
        all_ok &= window(g, &rig, &kern, &scratch, &pat, (l0, rows), &mut t)?;
    }
    let fired = control(g, &rig, &pat)?;
    println!(
        "{} bytes compared, {} mismatching comparisons, {} selected tokens",
        t.bytes, t.mismatches, t.selected
    );
    if t.bytes == 0 || t.selected == 0 {
        println!("FAIL: dsa_kpool_once_microtest compared nothing or selected nothing");
        std::process::exit(1);
    }
    if !fired {
        println!("FAIL: dsa_kpool_once_microtest control did not fire (comparator is blind)");
        std::process::exit(1);
    }
    if !all_ok {
        println!("FAIL: dsa_kpool_once_microtest once-per-window pools differ");
        std::process::exit(1);
    }
    println!("PASS dsa_kpool_once_microtest: once-per-window == per-sub-chunk, byte for byte");
    Ok(())
}
