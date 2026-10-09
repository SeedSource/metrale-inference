// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Fixture and per-stage launches shared by `dsa_depth_decode_microtest`: GLM-5.3's
//! DSA indexer shapes (H = 32, D = 128, KP = 4, one query row per call, three verify rows), 11
//! layers' indexer caches sized for the production 512K row (`--max-seq-len` 540,672), the
//! device `geom` written by the production `dsa_write_geom` kernel, a production
//! `DsaSelectScratch`, and the four selection kernels launched one at a time with exactly the
//! arguments `select_tokens` passes on a ceiling launch (`select/launch.rs`, cited per launch).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::pool_cache::dsa_pool_cache;
use metrale_model_arch::glm5next_dsa::select::grid_stride::{ceiling_grids, dsa_grid_stride};
use metrale_model_arch::glm5next_dsa::select::radix::{self, RadixMode, RadixTopk};
use metrale_model_arch::glm5next_dsa::select::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, DsaSelectScratch, select_tokens,
    topk_smem_for_tile, topk_tile,
};
use metrale_model_arch::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};

/// 2026-10-06: GLM-5.3 `index_n_heads`, `index_head_dim`, `index_kpool`, `index_topk`.
const H: usize = 32;
pub(crate) const D: usize = 128;
pub(crate) const KP: usize = 4;
const TOPK: usize = 2048;
/// 2026-10-06: `--max-seq-len` of the production 512K row.
pub(crate) const MAX_SEQ: usize = 540_672;
/// 2026-10-06: Capacity in pools, `contiguous_pool_count(KP, MAX_SEQ)` as `select_row_at` passes it
/// (`layer/rows.rs` lines 96-102).
pub(crate) const MAX_POOLS: usize = MAX_SEQ / KP;
/// 2026-10-06: DSA text layers in a decode step, and verify rows per layer at K = 3.
pub(crate) const LAYERS: usize = 11;
pub(crate) const ROWS: usize = 3;
/// 2026-10-06: `SCORES_BLOCK` and `ROW_BLOCK` in `glm5next_dsa/select.rs` (lines 48 and 50;
/// private there): `dsa_index_scores` and `dsa_topk_pools` / `dsa_expand_selection` threads per
/// block. A change there must be mirrored here; the token byte compare would not notice a
/// different block size, only the timing would drift.
const SCORES_BLOCK: u32 = 128;
const ROW_BLOCK: u32 = 256;
pub(crate) const POISON: u8 = 0xA5;

#[path = "dsa_depth_cache.rs"]
mod cache;

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

pub(crate) fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
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

/// 2026-10-06: GLM-5.3 DSA config as the other DSA examples build it
/// (`dsa_mla_split_fixture::cfg`), with the production 512K ceiling and `always_select_tail`.
pub(crate) fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: H,
        index_head_dim: D,
        index_kpool: KP,
        index_topk: TOPK,
        always_select_tail: true,
        local_heads: 32,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: MAX_SEQ,
    }
}

/// 2026-10-06: One layer's indexer cache: `[token, D]` BF16 keys and gate, `[token]` validity.
#[derive(Clone, Copy)]
pub(crate) struct Layer {
    pub(crate) k: DevicePtr,
    pub(crate) gate: DevicePtr,
    pub(crate) valid: DevicePtr,
    /// 2026-10-08: The persistent pool arrays with `METRALE_GLM_DSA_POOL_CACHE=1` (see
    /// `dsa_depth_cache.rs`); all NULL with the lever off.
    pub(crate) pc: PoolArrays,
}

/// 2026-10-08: One layer's pool-cache arrays: `pk` `[pools + 1, D]` f32, `pidx` `[pools + 1, KP]`
/// i32, `pvalid` `[pools + 1]` u8, `pk_len` one i32 (the state's `pk_len_dev`).
#[derive(Clone, Copy)]
pub(crate) struct PoolArrays {
    pub(crate) pk: DevicePtr,
    pub(crate) pidx: DevicePtr,
    pub(crate) pvalid: DevicePtr,
    pub(crate) pk_len: DevicePtr,
}

/// 2026-10-06: One verify row's device vectors: `seq_len` (one i32, the input of
/// `dsa_write_geom`), the `[5]` i32 `geom` it fills, and the query position.
#[derive(Clone, Copy)]
pub(crate) struct Row {
    seq_len: DevicePtr,
    geom: DevicePtr,
    q_pos: DevicePtr,
}

/// 2026-10-06: Buffers the stage-by-stage launches write, sized as the scratch of a
/// `DsaSelectScratch` for one query row at `MAX_POOLS` (`select.rs` `scratch_bytes`).
pub(crate) struct Stage {
    pool_keys: DevicePtr,
    pool_indices: DevicePtr,
    pool_valid: DevicePtr,
    scores: DevicePtr,
    valid_cand: DevicePtr,
    selected: DevicePtr,
    /// 2026-10-06: `[ROWS, out_width]` i32 token ids, row `r` at `r * out_width`.
    pub(crate) tokens: DevicePtr,
    /// 2026-10-08: Radix top-k work buffer for one row (`METRALE_GLM_DSA_TOPK_RADIX=1`).
    work: DevicePtr,
}

pub(crate) struct Fixture {
    pub(crate) cfg: Glm5NextDsaConfig,
    pub(crate) kernels: Glm5NextDsaKernels,
    pub(crate) layers: Vec<Layer>,
    ape: DevicePtr,
    q: DevicePtr,
    weights: DevicePtr,
    q_mask: DevicePtr,
    rows: Vec<Row>,
    pub(crate) scratch: DsaSelectScratch,
    /// 2026-10-08: A second scratch planned without pool regions, for the cache-less
    /// `select_tokens` the pool-cache run is compared with.
    pub(crate) scratch_nc: DsaSelectScratch,
    pub(crate) st: Stage,
    /// 2026-10-08: `METRALE_GLM_DSA_POOL_CACHE=1` is on (production `dsa_pool_cache()`), and the
    /// radix top-k engages at the ceiling pool count.
    pub(crate) pool_cache: bool,
    pub(crate) radix: bool,
    /// 2026-10-06: Grids `(compress, scores)` a ceiling launch gets, as `select_tokens` computes
    /// them (`select/launch.rs` lines 85-88, `grid_stride::ceiling_launch_grids_for`).
    pub(crate) grids: (usize, usize),
}

impl Fixture {
    pub(crate) fn new(g: &dyn GpuBackend, kernels: Glm5NextDsaKernels) -> Result<Self> {
        let cfg = cfg();
        cfg.validate()?;
        let pool_cache = dsa_pool_cache();
        let mut rng = Lcg(0x0DE9_7D5A);
        let tokens = MAX_POOLS * KP + 3;
        let k_host = rng.bf16_bytes(tokens * D, -1.0, 1.0);
        let gate = rng.bf16_bytes(tokens * D, -2.0, 2.0);
        // 2026-10-06: Every 23rd token invalid, as the grid-stride rig: a few invalid tokens.
        let valid: Vec<u8> = (0..tokens).map(|t| u8::from(t % 23 != 11)).collect();
        let mut layers = Vec::new();
        for _ in 0..LAYERS {
            layers.push(Layer {
                k: up(g, &k_host)?,
                gate: up(g, &gate)?,
                valid: up(g, &valid)?,
                pc: if pool_cache {
                    PoolArrays::alloc(g)?
                } else {
                    PoolArrays::null()
                },
            });
        }
        let q: Vec<f32> = (0..H * D).map(|_| rng.r(-1.0, 1.0)).collect();
        let weights: Vec<f32> = (0..H).map(|_| rng.r(0.0, 0.3)).collect();
        let ape: Vec<f32> = (0..KP * D).map(|_| rng.r(-0.5, 0.5)).collect();
        let q_mask = up(g, &[1u8; ROWS])?;
        let mut rows = Vec::new();
        for _ in 0..ROWS {
            rows.push(Row {
                seq_len: g.alloc(4)?,
                geom: g.alloc(6 * 4)?,
                q_pos: g.alloc(4)?,
            });
        }
        // 2026-10-06: The scratch is planned at the ceiling for the three verify rows, as the
        // workspace plans it (`Glm5NextDsaWorkspace::new`).
        let plan_geom = DsaSelectGeometry::plan(&cfg, MAX_SEQ, ROWS)?;
        let scratch = DsaSelectScratch::alloc(g, &cfg, &plan_geom)?;
        let scratch_nc = DsaSelectScratch::alloc_sized(
            g,
            DsaSelectScratch::plan_bytes_levers(
                &cfg,
                &[plan_geom],
                radix::dsa_topk_radix(),
                false,
            ),
        )?;
        let radix_on = radix::radix_mode(
            radix::dsa_topk_radix(),
            radix::radix_resolved(&kernels),
            true,
            MAX_POOLS,
        ) == RadixMode::Engaged;
        let m = MAX_POOLS;
        let st = Stage {
            pool_keys: g.alloc((m + 1) * D * 4)?,
            pool_indices: g.alloc((m + 1) * KP * 4)?,
            pool_valid: g.alloc(m + 1)?,
            scores: g.alloc(m * 4)?,
            valid_cand: g.alloc(m)?,
            selected: g.alloc(cfg.select_k(m) * 4)?,
            tokens: g.alloc(ROWS * cfg.out_width() * 4)?,
            work: g.alloc(radix::radix_work_bytes(&cfg, 1).max(1))?,
        };
        let sms = g.sm_count().map_or(48, |n| n as usize).max(1);
        let stride = dsa_grid_stride() && kernels.grid_stride_marker.0 != 0;
        Ok(Self {
            grids: ceiling_grids(m, if stride { sms } else { 0 }, stride),
            cfg,
            kernels,
            layers,
            ape: up(g, &f32_bytes(&ape))?,
            q: up(g, &f32_bytes(&q))?,
            weights: up(g, &f32_bytes(&weights))?,
            q_mask,
            rows,
            scratch,
            scratch_nc,
            st,
            pool_cache,
            radix: radix_on,
        })
    }

    /// 2026-10-06: Live context `s`: verify row `r` has `seq_len = s + r` and its query sits at
    /// the last position, `s + r - 1`, as `decode_k` places them (`meta.seq_len` row `r`,
    /// `meta.positions` row `r`). Then the production `dsa_write_geom` fills each row's `geom`
    /// from its `seq_len`, exactly as `layer/decode_k.rs` lines 391-402 launch it (1 block of
    /// 1 thread; arguments `seq_len`, `geom`, `index_kpool`, `index_topk`, `topk_tile()`).
    pub(crate) fn set_context(&self, g: &dyn GpuBackend, s: usize) -> Result<()> {
        for (r, row) in self.rows.iter().enumerate() {
            g.copy_h2d(&i32_bytes(&[(s + r) as i32]), row.seq_len)?;
            g.copy_h2d(&i32_bytes(&[(s + r) as i32 - 1]), row.q_pos)?;
            if self.pool_cache {
                continue; // `geom_pk` writes the geometry per call, from each layer's pk_len
            }
            KernelLaunch::new(g, self.kernels.write_geom)
                .grid([1, 1, 1])
                .block([1, 1, 1])
                .arg_ptr(row.seq_len)
                .arg_ptr(row.geom)
                .arg_u32(KP as u32)
                .arg_u32(TOPK as u32)
                .arg_u32(topk_tile() as u32)
                .launch(0)?;
        }
        if self.pool_cache {
            self.warm(g, s)?;
        }
        g.synchronize(0)
    }

    /// 2026-10-06: The host geometry of row `r` at live context `s`: `state.geometry(cfg, 1)`
    /// in `select_row_at` (`layer/rows.rs` line 50).
    pub(crate) fn geometry(&self, s: usize, r: usize) -> Result<DsaSelectGeometry> {
        DsaSelectGeometry::plan(&self.cfg, s + r, 1)
    }

    /// 2026-10-06: The full production selector for layer `l`, row `r`: `select_tokens` with
    /// `DsaSelectLaunch::Ceiling { max_pools }`, inputs and a per-row scratch as `select_row_at`
    /// passes them (`layer/rows.rs` lines 76-120).
    pub(crate) fn full(
        &self,
        g: &dyn GpuBackend,
        s: usize,
        l: usize,
        r: usize,
        stream: u64,
    ) -> Result<()> {
        if self.pool_cache {
            // 2026-10-08: `decode_k_rows` writes the geometry per layer and row first.
            self.geom_pk(g, l, r, stream)?;
        }
        self.select(g, s, l, r, self.pool_cache, stream)
    }

    /// 2026-10-08: `select_tokens` for layer `l`, row `r`, with the layer's pool arrays
    /// (`cached`) or without (full compress into the scratch). The cache-less call needs the
    /// geometry `write_geom` / `write_geom_pk` left in the row's `geom`.
    pub(crate) fn select(
        &self,
        g: &dyn GpuBackend,
        s: usize,
        l: usize,
        r: usize,
        cached: bool,
        stream: u64,
    ) -> Result<()> {
        let (layer, row) = (self.layers[l], self.rows[r]);
        let inputs = DsaSelectInputs {
            k_normed: layer.k,
            gate: layer.gate,
            valid: layer.valid,
            ape: self.ape,
            q: self.q,
            weights: self.weights,
            q_pos: row.q_pos,
            q_mask: self.q_mask.offset(r),
            first_key: 0,
            geom_dev: row.geom,
            pool_cache: cached.then(|| self.pc_args(l)),
        };
        select_tokens(
            g,
            &self.kernels,
            &self.cfg,
            &self.geometry(s, r)?,
            &inputs,
            &if cached { &self.scratch } else { &self.scratch_nc }.row(r, &self.cfg),
            DsaSelectLaunch::Ceiling { max_pools: MAX_POOLS },
            stream,
        )
    }

    /// 2026-10-06: Ceiling-launch scalars `select_tokens` derives (`select/launch.rs` lines
    /// 58-66): `(seq_a, npools_a, np2_a, selk_a)`.
    pub(super) fn scalars(&self) -> (usize, usize, usize, usize) {
        let m = MAX_POOLS;
        (
            m * KP,
            m,
            m.next_power_of_two().max(2).min(topk_tile()),
            self.cfg.select_k(m),
        )
    }

    /// 2026-10-06: `dsa_kpool_compress` alone, as `select/launch.rs` lines 92-107.
    pub(crate) fn compress(&self, g: &dyn GpuBackend, l: usize, r: usize, stream: u64) -> Result<()> {
        if self.pool_cache {
            self.geom_pk(g, l, r, stream)?;
            return self.compress_incr(g, l, r, stream);
        }
        let (layer, row, st) = (self.layers[l], self.rows[r], &self.st);
        let (seq_a, ..) = self.scalars();
        KernelLaunch::new(g, self.kernels.kpool_compress)
            .grid([self.grids.0 as u32, 1, 1])
            .block([D.min(1024) as u32, 1, 1])
            .arg_ptr(layer.k)
            .arg_ptr(layer.gate)
            .arg_ptr(layer.valid)
            .arg_ptr(self.ape)
            .arg_ptr(st.pool_keys)
            .arg_ptr(st.pool_indices)
            .arg_ptr(st.pool_valid)
            .arg_u32(seq_a as u32)
            .arg_u32(D as u32)
            .arg_u32(KP as u32)
            .arg_i32(0)
            .arg_ptr(row.geom)
            .launch(stream)
    }

    /// 2026-10-06: `dsa_index_scores` alone (the plain kernel: a ceiling launch never takes the
    /// tiled or tensor-core scorer, `select/launch.rs` lines 118-137), as lines 138-185.
    pub(crate) fn scores(&self, g: &dyn GpuBackend, l: usize, r: usize, stream: u64) -> Result<()> {
        let (layer, row, st) = (self.layers[l], self.rows[r], &self.st);
        let (seq_a, npools_a, ..) = self.scalars();
        let (pk, pidx, pvalid) = self.pools(l);
        KernelLaunch::new(g, self.kernels.index_scores)
            .grid([self.grids.1 as u32, 1, 1])
            .block([SCORES_BLOCK, 1, 1])
            .shared_mem(SCORES_BLOCK.max((H * 4) as u32))
            .arg_ptr(self.q)
            .arg_ptr(pk)
            .arg_ptr(self.weights)
            .arg_ptr(pidx)
            .arg_ptr(pvalid)
            .arg_ptr(layer.valid)
            .arg_ptr(row.q_pos)
            .arg_ptr(st.scores)
            .arg_ptr(st.valid_cand)
            .arg_u32(1)
            .arg_u32(npools_a as u32)
            .arg_u32(H as u32)
            .arg_u32(D as u32)
            .arg_u32(KP as u32)
            .arg_u32(seq_a as u32)
            .arg_f32((D as f32).powf(-0.5))
            .arg_ptr(row.geom)
            .launch(stream)
    }

    /// 2026-10-06: `dsa_topk_pools` alone, as `select/launch.rs` lines 189-200.
    pub(crate) fn topk(&self, g: &dyn GpuBackend, r: usize, stream: u64) -> Result<()> {
        let (row, st) = (self.rows[r], &self.st);
        let (_, npools_a, np2_a, selk_a) = self.scalars();
        if self.radix {
            // 2026-10-08: `select/launch.rs` lines 213-228: the radix select at the ceiling.
            let t = RadixTopk {
                scores: st.scores,
                selected: st.selected,
                work: st.work,
                q_rows: 1,
                n_pools: npools_a,
                select_k: selk_a,
                kcap: radix::radix_kcap(&self.cfg),
                geom_dev: row.geom,
            };
            return radix::launch_topk_radix(g, &self.kernels, &t, stream);
        }
        KernelLaunch::new(g, self.kernels.topk_pools)
            .grid([1, 1, 1])
            .block([ROW_BLOCK, 1, 1])
            .shared_mem(topk_smem_for_tile(np2_a) as u32)
            .arg_ptr(st.scores)
            .arg_ptr(st.selected)
            .arg_u32(1)
            .arg_u32(npools_a as u32)
            .arg_u32(np2_a as u32)
            .arg_u32(selk_a as u32)
            .arg_ptr(row.geom)
            .launch(stream)
    }

    /// 2026-10-06: `dsa_expand_selection` alone, as `select/launch.rs` lines 203-222, writing
    /// row `r` of the stage token buffer.
    pub(crate) fn expand(&self, g: &dyn GpuBackend, l: usize, r: usize, stream: u64) -> Result<()> {
        let (layer, row, st) = (self.layers[l], self.rows[r], &self.st);
        let (seq_a, npools_a, _, selk_a) = self.scalars();
        let (_, pidx, _) = self.pools(l);
        let w = self.cfg.out_width();
        KernelLaunch::new(g, self.kernels.expand_selection)
            .grid([1, 1, 1])
            .block([ROW_BLOCK, 1, 1])
            .arg_ptr(st.selected)
            .arg_ptr(pidx)
            .arg_ptr(st.valid_cand)
            .arg_ptr(layer.valid)
            .arg_ptr(row.q_pos)
            .arg_ptr(self.q_mask.offset(r))
            .arg_ptr(st.tokens.offset(r * w * 4))
            .arg_u32(1)
            .arg_u32(npools_a as u32)
            .arg_u32(KP as u32)
            .arg_u32(seq_a as u32)
            .arg_u32(selk_a as u32)
            .arg_u32(w as u32)
            .arg_i32(0)
            .arg_i32(i32::from(self.cfg.always_select_tail))
            .arg_ptr(row.geom)
            .launch(stream)
    }
}
