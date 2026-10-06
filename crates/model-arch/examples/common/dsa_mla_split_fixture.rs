// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Fixture of `dsa_mla_split_microtest`: the inputs of
//! `dsa_mla_headgroup_bitparity_microtest` (32 heads, random BF16 q, a random FP8 pool without
//! NaN codes, random paged block tables, selections with holes, out-of-range indices and
//! duplicates) plus a valid-prefix selection layout, built on the device.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::Glm5NextDsaConfig;
use metrale_model_arch::glm5next_dsa::attend::{DsaDecodeInputs, DsaDecodePaging};
use metrale_model_arch::glm5next_dsa::select::DsaSelectGeometry;

pub(crate) const HEADS: usize = 32;
pub(crate) const KVL: usize = 512;
const BLOCK: usize = 64;
const MAX_CTX: usize = 4096;
const MAX_BLOCKS: usize = MAX_CTX / BLOCK;
const N_PHYS: usize = 96;
const K_SCALE: f32 = 0.0173;
const V_SCALE: f32 = 0.0291;

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
    /// 2026-10-05: Random E4M3 codes, the two NaN codes (0x7F, 0xFF) excluded.
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

pub(crate) fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

pub(crate) fn cfg() -> Glm5NextDsaConfig {
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

/// 2026-10-05: The launchers read only `q_rows` and `out_width` from the geometry.
pub(crate) fn geom(rows: usize, width: usize) -> DsaSelectGeometry {
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

pub(crate) fn paging(rows: usize) -> DsaDecodePaging {
    DsaDecodePaging {
        num_seqs: rows,
        num_q_heads: HEADS,
        num_kv_heads: 1,
        max_blocks_per_seq: MAX_BLOCKS,
        block_size: BLOCK,
        cache_stride_bytes: (BLOCK * KVL) as u64,
    }
}

/// 2026-10-05: How a case's selection rows are filled.
#[derive(Clone, Copy)]
pub(crate) enum Sel {
    /// 2026-10-05: Holes, other negatives, past-`seq_len`, duplicates; rows 1 and 2 special.
    Mixed,
    /// 2026-10-05: The first `n` slots valid, the rest `-1` (dsa_expand_selection's shape).
    Prefix(usize),
}

#[derive(Clone, Copy)]
pub(crate) enum Pool {
    Aligned,
    /// 2026-10-05: K = V at a 1-byte offset: the byte-load branch.
    Misaligned,
    /// 2026-10-05: A separate V buffer and scale: the `same_kv == false` branch.
    DistinctV,
}

/// 2026-10-05: One case's device inputs and the host copies the checks need.
pub(crate) struct Case {
    pub(crate) rows: usize,
    pub(crate) width: usize,
    pub(crate) seq_lens: Vec<i32>,
    pub(crate) invalid_row: Option<usize>,
    pub(crate) q_bytes: Vec<u8>,
    pub(crate) inputs: DsaDecodeInputs,
    pub(crate) owned: Vec<DevicePtr>,
}

pub(crate) fn build_case(
    g: &dyn GpuBackend,
    rows: usize,
    width: usize,
    sel_kind: Sel,
    pool: Pool,
    seed: u64,
) -> Result<Case> {
    let mut rng = Lcg(seed);
    let q_amp = if seed & 1 == 0 { 1.0 } else { 4.0 };
    let q_bytes = rng.bf16_bytes(rows * HEADS * KVL, q_amp);
    let special = matches!(sel_kind, Sel::Mixed) && rows >= 3;

    let mut seq_lens = Vec::with_capacity(rows);
    let mut sel = Vec::with_capacity(rows * width);
    let mut bt = Vec::with_capacity(rows * MAX_BLOCKS);
    for r in 0..rows {
        let sl = match sel_kind {
            Sel::Mixed if special && r == 1 => 0,
            Sel::Mixed => 1 + rng.below(MAX_CTX),
            Sel::Prefix(_) => MAX_CTX,
        };
        seq_lens.push(sl as i32);
        for _ in 0..MAX_BLOCKS {
            bt.push(rng.below(N_PHYS) as i32);
        }
        let row_start = sel.len();
        for k in 0..width {
            let roll = rng.below(100);
            let t: i32 = match sel_kind {
                Sel::Prefix(n) if k < n => rng.below(sl) as i32,
                Sel::Prefix(_) => -1,
                // 2026-10-05: No valid index: the row must come out zero.
                Sel::Mixed if special && r == 2 => match roll % 3 {
                    0 => -1,
                    1 => sl as i32 + rng.below(64) as i32,
                    _ => -(2 + rng.below(9) as i32),
                },
                Sel::Mixed if roll < 8 => -1,
                Sel::Mixed if roll < 10 => -(2 + rng.below(9) as i32),
                Sel::Mixed if roll < 12 => sl as i32 + rng.below(64) as i32,
                Sel::Mixed if roll < 18 && k > 0 => sel[row_start + rng.below(k)],
                Sel::Mixed => rng.below(sl.max(1)) as i32,
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
        invalid_row: special.then_some(2),
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

pub(crate) fn free_case(g: &dyn GpuBackend, case: &Case) {
    for p in &case.owned {
        g.free(*p).ok();
    }
}
