// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Fixture of `dsa_ceiling_rows_microtest`: two pool-cache GLM-5.3 DSA indexer
//! states (`METRALE_GLM_DSA_POOL_CACHE=1`, as the car) holding the same random context, and
//! the two ways of running one sequence's `k` verify-row selections on them.
//!
//! * Arm A, today's replay-safe path (`decode_k_rows`, decode_k.rs:259-436 at 198cb05d): per
//!   row, `dsa_indexer_store_ring` at `positions[r]` (`store_pre_indexer_row`),
//!   `dsa_write_geom_pk` from `seq_len[r]` into the one workspace geometry
//!   (`pool_write_geom_pk`), then a `q_rows = 1` ceiling selection into row `r` of the output
//!   (`select_row_at`, `DsaSelectLaunch::Ceiling`).
//! * Arm B: all `k` stores, ONE `dsa_write_geom_pk` from `seq_len[k - 1]`, then ONE ceiling
//!   selection with `q_rows = k` (`DsaSelectLaunch::CeilingRows`) over the contiguous `q`,
//!   head weights and `positions` of the rows. Every input is a device buffer the step's
//!   metadata already holds, so nothing is copied from the host inside the step.
//!
//! The step's per-row inputs (staged key/gate rows, `q`, head weights, positions, `seq_len`)
//! are uploaded before a step, as the batched-verify metadata is before a graph replay.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::select::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, DsaSelectScratch, PoolRows,
    compress_pools_only, contiguous_pool_count, select_tokens, topk_tile,
};
use metrale_model_arch::glm5next_dsa::state::{Glm5NextDsaState, max_dsa_context};
use metrale_model_arch::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};

pub(crate) const H: usize = 32;
pub(crate) const D: usize = 128;
pub(crate) const KP: usize = 4;
const TOPK: usize = 2048;
/// 2026-10-12: `--max-seq-len` of the car's 512K row (CAR.json profile 512k, 540,672).
const MAX_SEQ: usize = 540_672;
/// 2026-10-12: Most verify rows of one sequence tried.
pub(crate) const KMAX: usize = 8;
/// 2026-10-12: Context rows written per host tile while filling (under the 8,448-row ring).
const TILE: usize = 256;

pub(crate) struct Lcg(pub(crate) u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn bf16_bytes(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                bf16::from_f32(lo + (hi - lo) * self.f())
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    }
    fn f32_bytes(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| (lo + (hi - lo) * self.f()).to_le_bytes())
            .collect()
    }
}

fn i32s(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    v.into_iter()
        .flat_map(|x| (x as i32).to_le_bytes())
        .collect()
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn cfg() -> Glm5NextDsaConfig {
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

/// 2026-10-12: One arm's state, its select scratch (planned for `KMAX` rows at the ceiling,
/// as `Glm5NextDsaWorkspace` plans `max_rows`) and its device geometry (6 slots).
pub(crate) struct Side {
    pub(crate) st: Glm5NextDsaState,
    pub(crate) scr: DsaSelectScratch,
    geom: DevicePtr,
}

/// 2026-10-12: Which selection a step runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Arm {
    /// 2026-10-12: Arm A on side A: `k` single-row chains.
    PerRow,
    /// 2026-10-12: Arm B on side B: one `k`-row chain.
    Rows,
    /// 2026-10-12: Control on side B: arm B with the geometry written from row 0's `seq_len`
    /// (the wrong length); it must differ from arm A when a later row needs a pool or a tail
    /// key past row 0's length.
    RowsFirstLen,
}

pub(crate) struct Rig<'g> {
    pub(crate) g: &'g dyn GpuBackend,
    pub(crate) s: u64,
    pub(crate) cfg: Glm5NextDsaConfig,
    k: Glm5NextDsaKernels,
    pub(crate) a: Side,
    pub(crate) b: Side,
    ape: DevicePtr,
    q: DevicePtr,
    w: DevicePtr,
    q_mask: DevicePtr,
    stage_k: DevicePtr,
    stage_g: DevicePtr,
    /// 2026-10-12: `[KMAX]` i32 positions (`AttnMetadataDev::positions`; store row and `q_pos`).
    pos: DevicePtr,
    /// 2026-10-12: `[KMAX]` i32 per-row `seq_len` (`AttnMetadataDev::seq_len`).
    sl: DevicePtr,
    rng: Lcg,
    /// 2026-10-12: Rows of context before the step (both states' host length).
    pub(crate) seq: usize,
    pub(crate) max_pools: usize,
}

impl<'g> Rig<'g> {
    pub(crate) fn new(g: &'g dyn GpuBackend, k: Glm5NextDsaKernels, seed: u64) -> Result<Self> {
        let cfg = cfg();
        cfg.validate()?;
        let cap = max_dsa_context(&cfg);
        let mut rng = Lcg(seed);
        let plan = DsaSelectGeometry::plan(&cfg, cap, KMAX)?;
        let side = |g: &dyn GpuBackend| -> Result<Side> {
            Ok(Side {
                st: Glm5NextDsaState::alloc_pool_cache(g, &cfg, 0, None)?,
                scr: DsaSelectScratch::alloc_sized(
                    g,
                    DsaSelectScratch::plan_bytes_with(&cfg, &[plan], true),
                )?,
                geom: g.alloc(6 * 4)?,
            })
        };
        Ok(Self {
            g,
            s: g.create_stream()?,
            cfg,
            k,
            a: side(g)?,
            b: side(g)?,
            ape: up(g, &rng.f32_bytes(KP * D, -0.5, 0.5))?,
            q: g.alloc(KMAX * H * D * 4)?,
            w: g.alloc(KMAX * H * 4)?,
            q_mask: up(g, &[1u8; KMAX])?,
            stage_k: g.alloc(KMAX * D * 2)?,
            stage_g: g.alloc(KMAX * D * 2)?,
            pos: g.alloc(KMAX * 4)?,
            sl: g.alloc(KMAX * 4)?,
            rng,
            seq: 0,
            max_pools: contiguous_pool_count(KP, cap),
        })
    }

    pub(crate) fn sync(&self) -> Result<()> {
        self.g.synchronize(self.s)
    }

    /// 2026-10-12: Both states to `seq` rows of the same random context. The steps run since
    /// the last fill left the device watermark at `(seq_old + k) / KP` while the host length
    /// stayed `seq_old`, so it is clamped back first (`dsa_pk_len_clamp`, the host path's
    /// lowering after a rewind); then `TILE`-row writes, each followed by the compress-only
    /// pass the MTP drafter context rows take (`write_kv_rows` / `pool_compress_written`), so
    /// both states end with pools `[0, seq / KP)` final and the watermark at `seq / KP`.
    pub(crate) fn fill_to(&mut self, seq: usize) -> Result<()> {
        if seq < self.seq {
            bail!("fill_to {seq}: the context only grows (at {})", self.seq);
        }
        self.sync()?;
        let (g, cfg, s) = (self.g, self.cfg, self.s);
        for side in [&self.a, &self.b] {
            let pc = side.st.pool_cache().expect("pool cache");
            KernelLaunch::new(g, self.k.pk_len_clamp)
                .grid([1, 1, 1])
                .block([1, 1, 1])
                .arg_ptr(pc.pk_len_dev)
                .arg_i32((self.seq / KP) as i32)
                .launch(s)?;
        }
        while self.seq < seq {
            let n = TILE.min(seq - self.seq);
            let kb = self.rng.bf16_bytes(n * D, -1.0, 1.0);
            let gb = self.rng.bf16_bytes(n * D, -2.0, 2.0);
            self.sync()?;
            let pos0 = self.seq;
            for side in [&mut self.a, &mut self.b] {
                side.st.ensure_room(n)?;
                let row = D * 2;
                for (r0, m) in side.st.ring_runs(pos0, n) {
                    let off = side.st.row_offset(pos0 + r0);
                    g.copy_h2d(&kb[r0 * row..(r0 + m) * row], side.st.k_normed.offset(off))?;
                    g.copy_h2d(&gb[r0 * row..(r0 + m) * row], side.st.gate.offset(off))?;
                }
                g.copy_h2d(&vec![1u8; n], side.st.valid.offset(pos0))?;
                side.st.advance(n)?;
                if side.st.pool_compress_due() {
                    let pc = side.st.pool_select_args()?.expect("pool cache");
                    let rows = PoolRows {
                        k_normed: side.st.k_normed,
                        gate: side.st.gate,
                        valid: side.st.valid,
                        ape: self.ape,
                        first_key: 0,
                    };
                    let len = side.st.len();
                    compress_pools_only(g, &self.k, &cfg, &rows, &pc, len, s)?;
                    side.st.note_selected();
                }
            }
            self.seq += n;
        }
        self.sync()
    }

    /// 2026-10-12: A new random step of `k` rows at `seq`: staged key/gate rows, the selector
    /// query `[k, H, D]` f32 and head weights `[k, H]`, positions `seq + r`, `seq_len`
    /// `seq + r + 1`.
    pub(crate) fn draw(&mut self, k: usize) -> Result<()> {
        self.sync()?;
        let g = self.g;
        g.copy_h2d(&self.rng.bf16_bytes(k * D, -1.0, 1.0), self.stage_k)?;
        g.copy_h2d(&self.rng.bf16_bytes(k * D, -2.0, 2.0), self.stage_g)?;
        g.copy_h2d(&self.rng.f32_bytes(k * H * D, -1.0, 1.0), self.q)?;
        g.copy_h2d(&self.rng.f32_bytes(k * H, 0.0, 0.3), self.w)?;
        g.copy_h2d(&i32s(self.seq..self.seq + k), self.pos)?;
        g.copy_h2d(&i32s(self.seq + 1..self.seq + k + 1), self.sl)
    }

    /// 2026-10-12: `dsa_indexer_store_ring` of step row `r` into `side` (`store_indexer_row_from`).
    fn store(&self, side: &Side, r: usize) -> Result<()> {
        let pc = side.st.pool_cache().expect("pool cache");
        KernelLaunch::new(self.g, self.k.indexer_store_ring)
            .grid([1, 1, 1])
            .block([D as u32, 1, 1])
            .arg_ptr(self.stage_k.offset(r * D * 2))
            .arg_ptr(self.stage_g.offset(r * D * 2))
            .arg_ptr(self.pos.offset(r * 4))
            .arg_ptr(side.st.k_normed)
            .arg_ptr(side.st.gate)
            .arg_ptr(side.st.valid)
            .arg_u32(D as u32)
            .arg_u32(pc.book.ring_rows() as u32)
            .arg_u32(KP as u32)
            .arg_ptr(pc.pk_len_dev)
            .launch(self.s)
    }

    /// 2026-10-12: `dsa_write_geom_pk` from `seq_len[r]` into `side`'s geometry
    /// (`pool_write_geom_pk`).
    fn geom(&self, side: &Side, r: usize) -> Result<()> {
        let pc = side.st.pool_cache().expect("pool cache");
        KernelLaunch::new(self.g, self.k.write_geom_pk)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(self.sl.offset(r * 4))
            .arg_ptr(side.geom)
            .arg_u32(KP as u32)
            .arg_u32(TOPK as u32)
            .arg_u32(topk_tile() as u32)
            .arg_ptr(pc.pk_len_dev)
            .launch(self.s)
    }

    /// 2026-10-12: The ceiling selection of rows `[r0, r0 + rows)` on `side` into row `r0` of its
    /// output (`select_row_at` for one row; the `CeilingRows` pass for several).
    fn select(&self, side: &Side, r0: usize, rows: usize) -> Result<()> {
        let cfg = self.cfg;
        let mut pc = side.st.pool_select_args()?.expect("pool cache");
        // 2026-10-12: Unused on a ceiling launch (the start comes from the geometry slot).
        pc.pk_start = 0;
        let inputs = DsaSelectInputs {
            k_normed: side.st.k_normed,
            gate: side.st.gate,
            valid: side.st.valid,
            ape: self.ape,
            q: self.q.offset(r0 * H * D * 4),
            weights: self.w.offset(r0 * H * 4),
            q_pos: self.pos.offset(r0 * 4),
            q_mask: self.q_mask,
            first_key: 0,
            geom_dev: side.geom,
            pool_cache: Some(pc),
        };
        let max_pools = self.max_pools;
        let launch = if rows == 1 {
            DsaSelectLaunch::Ceiling { max_pools }
        } else {
            DsaSelectLaunch::CeilingRows { max_pools }
        };
        let geom = DsaSelectGeometry::plan(&cfg, max_dsa_context(&cfg), rows)?;
        let scr = side.scr.row(r0, &cfg);
        select_tokens(self.g, &self.k, &cfg, &geom, &inputs, &scr, launch, self.s)
    }

    /// 2026-10-12: Enqueue one step of `k` rows of `arm` on the rig's stream (capturable: no
    /// host copy, no synchronisation).
    pub(crate) fn step(&self, arm: Arm, k: usize) -> Result<()> {
        match arm {
            Arm::PerRow => {
                for r in 0..k {
                    self.store(&self.a, r)?;
                    self.geom(&self.a, r)?;
                    self.select(&self.a, r, 1)?;
                }
            }
            Arm::Rows | Arm::RowsFirstLen => {
                for r in 0..k {
                    self.store(&self.b, r)?;
                }
                let r_len = if arm == Arm::Rows { k - 1 } else { 0 };
                self.geom(&self.b, r_len)?;
                self.select(&self.b, 0, k)?;
            }
        }
        Ok(())
    }

    /// 2026-10-12: `[k, out_width]` i32 token rows of `side`'s output, as bytes.
    pub(crate) fn tokens(&self, side: &Side, k: usize) -> Result<Vec<u8>> {
        self.sync()?;
        let mut v = vec![0u8; k * self.cfg.out_width() * 4];
        self.g.copy_d2h(side.scr.tokens(), &mut v)?;
        Ok(v)
    }

    /// 2026-10-12: Pool keys, ids and validity over `[0, ceil(end / KP))` plus the device
    /// watermark of `side`, as bytes.
    pub(crate) fn pools(&self, side: &Side, end: usize) -> Result<Vec<u8>> {
        self.sync()?;
        let pc = side.st.pool_cache().expect("pool cache");
        let n = end.div_ceil(KP);
        let mut out = Vec::new();
        for (p, bytes) in [
            (pc.pk, n * D * 4),
            (pc.pidx, n * KP * 4),
            (pc.pvalid, n),
            (pc.pk_len_dev, 4),
        ] {
            let mut v = vec![0u8; bytes];
            self.g.copy_d2h(p, &mut v)?;
            out.extend(v);
        }
        Ok(out)
    }

    pub(crate) fn free(mut self) -> Result<()> {
        let g = self.g;
        for side in [&mut self.a, &mut self.b] {
            side.st.free(g)?;
            side.scr.free(g)?;
            g.free(side.geom)?;
        }
        for p in [
            self.ape,
            self.q,
            self.w,
            self.q_mask,
            self.stage_k,
            self.stage_g,
            self.pos,
            self.sl,
        ] {
            g.free(p)?;
        }
        Ok(())
    }
}

/// 2026-10-12: Per-row count of selected token ids (`>= 0`) in `[k, out_width]` token bytes.
pub(crate) fn valid_counts(tok: &[u8], ow: usize) -> Vec<usize> {
    tok.chunks_exact(ow * 4)
        .map(|row| {
            row.chunks_exact(4)
                .filter(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]) >= 0)
                .count()
        })
        .collect()
}

/// 2026-10-12: Host reference of each row's layout, independent of either arm: with every
/// key valid, row `r` (`q_pos = seq + r`, `vis = q_pos + 1` visible keys) holds
/// `row_select_k = min(vis / KP, 512)` pools' keys (all `< vis - vis % KP`) in slots
/// `[0, row_select_k * KP)`, then its tail keys `vis - vis % KP ..` in the next `vis % KP`
/// slots, then `-1`. `None` when every row matches, else the first mismatch.
pub(crate) fn placement_error(tok: &[u8], seq: usize, ow: usize) -> Option<String> {
    let id = |row: &[u8], i: usize| i32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap());
    for (r, row) in tok.chunks_exact(ow * 4).enumerate() {
        let vis = seq + r + 1;
        let sel = (vis / KP).min(TOPK / KP);
        let (tail, start) = (vis % KP, vis - vis % KP);
        let base = sel * KP;
        if let Some(i) = (0..base).find(|&i| !(0..start as i32).contains(&id(row, i))) {
            return Some(format!("row {r}: pool slot {i} = {}", id(row, i)));
        }
        for t in 0..KP - 1 {
            let want = if t < tail { (start + t) as i32 } else { -1 };
            if id(row, base + t) != want {
                return Some(format!(
                    "row {r}: tail slot {} = {}, want {want}",
                    base + t,
                    id(row, base + t)
                ));
            }
        }
        if let Some(i) = (base + KP - 1..ow).find(|&i| id(row, i) != -1) {
            return Some(format!("row {r}: slot {i} past the tail = {}", id(row, i)));
        }
    }
    None
}
