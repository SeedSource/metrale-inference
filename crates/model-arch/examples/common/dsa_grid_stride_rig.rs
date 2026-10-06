// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Inputs, uploads, kernel launches, buffers and the comparison tally shared by
//! `dsa_grid_stride_microtest`: GLM-5.3's DSA indexer shapes (H = 32, D = 128, KP = 4, one
//! query row), per-layer indexer caches, and `dsa_kpool_compress` / `dsa_index_scores`
//! launched as `select_tokens` launches them under a ceiling launch (S and the pool counts
//! read from a device `geom`).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-05: GLM-5.3 `index_n_heads`, `index_head_dim` and `index_kpool`.
const H: usize = 32;
pub(crate) const D: usize = 128;
pub(crate) const KP: usize = 4;
/// 2026-10-05: `SCORES_BLOCK` in `glm5next_dsa/select.rs` (private there): `dsa_index_scores`
/// threads per block and its least shared memory in bytes; compress runs `D` threads.
const BLOCK: u32 = 128;
/// 2026-10-05: Tokens the layer caches hold: the most live pools, plus a partial pool.
const MAX_TOKENS: usize = 16_384 * KP + 3;
/// 2026-10-05: DSA layers in a decode step; their caches are distinct, so timed replays do not
/// serve every call from the L2.
pub(crate) const LAYERS: usize = 11;
pub(crate) const POISON_A: u8 = 0xA5;
pub(crate) const POISON_B: u8 = 0x5A;

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
}

pub(crate) fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

pub(crate) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

pub(crate) fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

/// 2026-10-05: One layer's indexer cache: `[token, D]` BF16 keys and gate, `[token]` validity.
#[derive(Clone, Copy)]
pub(crate) struct Layer {
    pub(crate) k: DevicePtr,
    pub(crate) gate: DevicePtr,
    pub(crate) valid: DevicePtr,
}

/// 2026-10-05: Compress outputs at capacity (`max_pools + 1` pools): keys f32, indices i32,
/// validity u8.
#[derive(Clone, Copy)]
pub(crate) struct Pools {
    pub(crate) keys: DevicePtr,
    pub(crate) indices: DevicePtr,
    pub(crate) valid: DevicePtr,
}

/// 2026-10-05: Scores outputs at capacity (`max_pools`, one row): scores f32, candidacy u8.
#[derive(Clone, Copy)]
pub(crate) struct Scored {
    pub(crate) out: DevicePtr,
    pub(crate) cand: DevicePtr,
}

/// 2026-10-05: Everything the launches read, uploaded once.
pub(crate) struct Dev {
    pub(crate) compress: KernelHandle,
    pub(crate) scores: KernelHandle,
    pub(crate) sms: usize,
    pub(crate) ape: DevicePtr,
    pub(crate) q: DevicePtr,
    pub(crate) weights: DevicePtr,
    pub(crate) q_pos: DevicePtr,
    pub(crate) layers: Vec<Layer>,
    pub(crate) k_host: Vec<u8>,
    pub(crate) q_host: Vec<f32>,
}

impl Dev {
    pub(crate) fn new(
        g: &dyn GpuBackend,
        compress: KernelHandle,
        scores: KernelHandle,
    ) -> Result<Self> {
        let mut rng = Lcg(0x05EE_DD5A);
        let k_host = rng.bf16_bytes(MAX_TOKENS * D, -1.0, 1.0);
        let gate = rng.bf16_bytes(MAX_TOKENS * D, -2.0, 2.0);
        // 2026-10-05: Every 23rd token invalid, so pools are partly or wholly invalid.
        let valid: Vec<u8> = (0..MAX_TOKENS).map(|t| u8::from(t % 23 != 11)).collect();
        let mut layers = Vec::new();
        for _ in 0..LAYERS {
            layers.push(Layer {
                k: up(g, &k_host)?,
                gate: up(g, &gate)?,
                valid: up(g, &valid)?,
            });
        }
        let q_host: Vec<f32> = (0..H * D).map(|_| rng.r(-1.0, 1.0)).collect();
        // 2026-10-05: Head weights with a +0 and a -0 among random values.
        let weights: Vec<f32> = (0..H)
            .map(|i| match i % 29 {
                3 => 0.0,
                17 => -0.0,
                _ => rng.r(-1.0, 1.0),
            })
            .collect();
        let ape: Vec<f32> = (0..KP * D).map(|_| rng.r(-0.5, 0.5)).collect();
        Ok(Self {
            compress,
            scores,
            sms: g.sm_count().map_or(48, |n| n as usize),
            ape: up(g, &f32_bytes(&ape))?,
            q: up(g, &f32_bytes(&q_host))?,
            weights: up(g, &f32_bytes(&weights))?,
            q_pos: g.alloc(4)?,
            layers,
            k_host,
            q_host,
        })
    }
}

/// 2026-10-05: `dsa_write_geom`'s five slots for `s` live tokens, as the host would write them:
/// S, pools with the partial one, complete pools, select_k, top-k tile width.
pub(crate) fn geom_bytes(s: usize) -> Vec<u8> {
    let np = s / KP;
    let mut np2 = 2usize;
    while np2 < np && np2 < 2048 {
        np2 <<= 1;
    }
    let sel = np.min(2048 / KP);
    i32_bytes(&[
        s as i32,
        s.div_ceil(KP) as i32,
        np as i32,
        sel as i32,
        np2 as i32,
    ])
}

/// 2026-10-05: The query's position for `s` live tokens: a few tokens behind the end, so the
/// last pools are not visible to it.
pub(crate) fn q_pos_of(s: usize) -> i32 {
    if s > 16 { s as i32 - 6 } else { s as i32 - 1 }
}

/// 2026-10-05: A `dsa_kpool_compress` launch as `select_tokens` issues a ceiling launch over
/// `m` pools, with `grid` blocks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compress(
    g: &dyn GpuBackend,
    d: &Dev,
    m: usize,
    grid: usize,
    l: Layer,
    p: Pools,
    geom: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(g, d.compress)
        .grid([grid as u32, 1, 1])
        .block([D as u32, 1, 1])
        .arg_ptr(l.k)
        .arg_ptr(l.gate)
        .arg_ptr(l.valid)
        .arg_ptr(d.ape)
        .arg_ptr(p.keys)
        .arg_ptr(p.indices)
        .arg_ptr(p.valid)
        .arg_u32((m * KP) as u32)
        .arg_u32(D as u32)
        .arg_u32(KP as u32)
        .arg_i32(0)
        .arg_ptr(geom)
        .launch(stream)
}

/// 2026-10-05: A `dsa_index_scores` launch as `select_tokens` issues a ceiling launch over `m`
/// pools (one query row), with `grid` blocks; `q` may differ from `d.q` (a control).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scores(
    g: &dyn GpuBackend,
    d: &Dev,
    m: usize,
    grid: usize,
    (l, q): (Layer, DevicePtr),
    p: Pools,
    o: Scored,
    geom: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(g, d.scores)
        .grid([grid as u32, 1, 1])
        .block([BLOCK, 1, 1])
        .shared_mem(BLOCK.max((H * 4) as u32))
        .arg_ptr(q)
        .arg_ptr(p.keys)
        .arg_ptr(d.weights)
        .arg_ptr(p.indices)
        .arg_ptr(p.valid)
        .arg_ptr(l.valid)
        .arg_ptr(d.q_pos)
        .arg_ptr(o.out)
        .arg_ptr(o.cand)
        .arg_u32(1)
        .arg_u32(m as u32)
        .arg_u32(H as u32)
        .arg_u32(D as u32)
        .arg_u32(KP as u32)
        .arg_u32((m * KP) as u32)
        .arg_f32((D as f32).powf(-0.5))
        .arg_ptr(geom)
        .launch(stream)
}

/// 2026-10-05: Buffers for one capacity `m`.
pub(crate) struct Arena {
    pub(crate) m: usize,
    pub(crate) r: Pools,
    pub(crate) n: Pools,
    pub(crate) o_ref: Scored,
    pub(crate) o_new: Scored,
}

impl Arena {
    pub(crate) fn new(g: &dyn GpuBackend, m: usize) -> Result<Self> {
        let pools = || -> Result<Pools> {
            Ok(Pools {
                keys: g.alloc((m + 1) * D * 4)?,
                indices: g.alloc((m + 1) * KP * 4)?,
                valid: g.alloc(m + 1)?,
            })
        };
        let scored = || -> Result<Scored> {
            Ok(Scored {
                out: g.alloc(m * 4)?,
                cand: g.alloc(m)?,
            })
        };
        Ok(Self {
            m,
            r: pools()?,
            n: pools()?,
            o_ref: scored()?,
            o_new: scored()?,
        })
    }

    pub(crate) fn fill(&self, g: &dyn GpuBackend, p: Pools, o: Scored, v: u8) -> Result<()> {
        let m = self.m;
        g.memset(p.keys, v, (m + 1) * D * 4)?;
        g.memset(p.indices, v, (m + 1) * KP * 4)?;
        g.memset(p.valid, v, m + 1)?;
        g.memset(o.out, v, m * 4)?;
        g.memset(o.cand, v, m)
    }

    pub(crate) fn read_pools(&self, g: &dyn GpuBackend, p: Pools) -> Result<Vec<Vec<u8>>> {
        let m = self.m;
        Ok(vec![
            down(g, p.keys, (m + 1) * D * 4)?,
            down(g, p.indices, (m + 1) * KP * 4)?,
            down(g, p.valid, m + 1)?,
        ])
    }

    pub(crate) fn read_scored(&self, g: &dyn GpuBackend, o: Scored) -> Result<Vec<Vec<u8>>> {
        Ok(vec![down(g, o.out, self.m * 4)?, down(g, o.cand, self.m)?])
    }
}

/// 2026-10-05: Comparisons made, bytes compared, comparisons that differed.
#[derive(Default)]
pub(crate) struct Tally {
    pub(crate) legs: usize,
    pub(crate) bytes: usize,
    pub(crate) failed: usize,
    pub(crate) candidates: usize,
    pub(crate) positive: usize,
}

impl Tally {
    pub(crate) fn same(&mut self, a: &[Vec<u8>], b: &[Vec<u8>]) -> bool {
        let ok = a == b;
        self.legs += 1;
        self.bytes += a.iter().map(Vec::len).sum::<usize>();
        self.failed += usize::from(!ok);
        ok
    }
}
