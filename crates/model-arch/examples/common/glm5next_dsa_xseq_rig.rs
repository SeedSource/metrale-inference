// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The rig `glm5next_dsa_xseq_microtest` drives: one DSA layer, its sequences, the
//! per-case metadata, the poison/capture helpers and the two arms (OFF: the call sites'
//! per-sequence `decode_k` loop; ON: `decode_xseq`).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

#![allow(dead_code)]

use super::glm5next_multiseq_stack as st;
use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::layer::{DsaXseqArena, Glm5NextDsaLayer};
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, LayerState, MoeLoraRoute};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use std::sync::Arc;

/// 2026-10-03: Rows per prefill call (the workspace's `max_rows` is at least this).
pub(crate) const CHUNK: usize = 16;
/// 2026-10-03: Most metadata rows a case uploads.
pub(crate) const META_ROWS: usize = 32;
pub(crate) const POISON_OFF: u8 = 0xA5;
pub(crate) const POISON_ON: u8 = 0x5A;

pub(crate) struct Env<'g> {
    pub(crate) gpu: &'g dyn GpuBackend,
    pub(crate) config: ModelConfig,
    pub(crate) buffers: BufferArena,
    pub(crate) dispatch: GemmDispatch,
    pub(crate) derived: DerivedWeights,
    pub(crate) levers: ModelLevers,
    pub(crate) stats: ModelStats,
}

impl Env<'_> {
    pub(crate) fn ctx(
        &self,
        attn_metadata: Option<AttnMetadataDev>,
        decode_step: bool,
    ) -> ForwardContext<'_> {
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata,
            profile: false,
            comm: None,
            graph_capture: false,
            decode_step,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: MoeLoraRoute::Fold,
        }
    }
}

pub(crate) struct Seq {
    pub(crate) len: usize,
    pub(crate) blocks: Vec<u32>,
    pub(crate) state: Box<dyn LayerState>,
}

impl Seq {
    pub(crate) fn dsa(&mut self) -> &mut Glm5NextDsaState {
        self.state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .expect("DSA state")
    }
    pub(crate) fn slot(&self, p: usize) -> usize {
        self.blocks[p / st::BLOCK] as usize * st::BLOCK + p % st::BLOCK
    }
}

/// 2026-10-03: The metadata arrays one case uploads, `META_ROWS` rows.
pub(crate) struct Meta {
    pub(crate) pos: DevicePtr,
    pub(crate) slot: DevicePtr,
    pub(crate) sl: DevicePtr,
    pub(crate) bt: DevicePtr,
}

pub(crate) struct Rig<'g> {
    pub(crate) env: Env<'g>,
    pub(crate) layer: Glm5NextDsaLayer,
    pub(crate) arena: Arc<DsaXseqArena>,
    pub(crate) kv: PagedKvCache,
    pub(crate) meta: Meta,
    pub(crate) hid: DevicePtr,
    pub(crate) h: usize,
    pub(crate) kv_lora: usize,
    pub(crate) d: usize,
}

/// 2026-10-03: Which per-sequence loop the OFF arm mirrors.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Mode {
    /// `multi_seq.rs`: one row per sequence, `row_view(i)`.
    Decode,
    /// `verify_multi.rs`: `ks[i]` rows, `verify_rows_view(m, off[i], ks[i])`.
    Verify,
}

impl Rig<'_> {
    pub(crate) fn g(&self) -> &dyn GpuBackend {
        self.env.gpu
    }

    /// 2026-10-03: Rows `seq.len + r`, r < k, of every sequence, sequence-major, as the
    /// engine's batch metadata lays them out.
    pub(crate) fn upload_meta(&self, seqs: &[Seq], ks: &[usize]) -> Result<AttnMetadataDev> {
        let (mut pos, mut slot, mut sl) = (Vec::new(), Vec::new(), Vec::new());
        let rows: usize = ks.iter().sum();
        if rows > META_ROWS {
            bail!("{rows} metadata rows over {META_ROWS}");
        }
        let mut bt = vec![0u8; rows * st::MAX_BLOCKS * 4];
        let mut r = 0;
        for (s, &k) in seqs.iter().zip(ks) {
            for j in 0..k {
                let p = s.len + j;
                pos.extend((p as i32).to_le_bytes());
                slot.extend((s.slot(p) as i64).to_le_bytes());
                sl.extend(((p + 1) as i32).to_le_bytes());
                for (b, id) in s.blocks.iter().enumerate() {
                    let o = (r * st::MAX_BLOCKS + b) * 4;
                    bt[o..o + 4].copy_from_slice(&id.to_le_bytes());
                }
                r += 1;
            }
        }
        let g = self.g();
        g.synchronize(g.default_stream())?;
        g.copy_h2d(&pos, self.meta.pos)?;
        g.copy_h2d(&slot, self.meta.slot)?;
        g.copy_h2d(&sl, self.meta.sl)?;
        g.copy_h2d(&bt, self.meta.bt)?;
        Ok(AttnMetadataDev {
            positions: self.meta.pos,
            positions_h: DevicePtr::NULL,
            positions_w: DevicePtr::NULL,
            slot: self.meta.slot,
            seq_len: self.meta.sl,
            block_table: self.meta.bt,
            max_blocks_per_seq: st::MAX_BLOCKS as u32,
            num_seqs: rows as u32,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        })
    }

    /// 2026-10-03: The metadata each sequence's `decode_k` gets in the loop.
    pub(crate) fn seq_metas(
        m: &AttnMetadataDev,
        ks: &[usize],
        mode: Mode,
    ) -> Vec<Option<AttnMetadataDev>> {
        let mut off = 0;
        ks.iter()
            .enumerate()
            .map(|(i, &k)| {
                let v = match mode {
                    Mode::Decode => m.row_view(i),
                    Mode::Verify => AttnMetadataDev {
                        num_seqs: k as u32,
                        ..m.row_view(off)
                    },
                };
                off += k;
                Some(v)
            })
            .collect()
    }

    pub(crate) fn prefill(&mut self, seq: &mut Seq, len: usize, rng: &mut st::Lcg) -> Result<()> {
        let mut t = 0;
        while t < len {
            let k = CHUNK.min(len - t);
            self.g().synchronize(self.g().default_stream())?;
            self.g()
                .copy_h2d(&st::bf16_bytes(&rng.vec(k * self.h, 1.0)), self.hid)?;
            let ctx = self.env.ctx(None, false);
            let mut bt = seq.blocks.clone();
            self.layer.decode_k(
                self.hid,
                k,
                seq.state.as_mut(),
                &mut self.kv,
                t,
                &mut bt,
                &ctx,
                self.env.gpu.default_stream(),
                true,
            )?;
            t += k;
        }
        seq.len = len;
        Ok(())
    }

    /// 2026-10-03: Fill every byte a step over `ks` writes outside `hid` with `poison`.
    pub(crate) fn poison(&self, seqs: &mut [Seq], ks: &[usize], poison: u8) -> Result<()> {
        let g = self.env.gpu;
        g.synchronize(g.default_stream())?;
        let (kv_lora, d) = (self.kv_lora, self.d);
        for (s, &k) in seqs.iter_mut().zip(ks) {
            for j in 0..k {
                let p = s.len + j;
                g.copy_h2d(
                    &vec![poison; kv_lora],
                    self.kv.k_pool_ptr(0).offset(s.slot(p) * kv_lora),
                )?;
                let ds = s.dsa();
                let off = ds.row_offset(p);
                g.copy_h2d(&vec![poison; d * 2], ds.k_normed.offset(off))?;
                g.copy_h2d(&vec![poison; d * 2], ds.gate.offset(off))?;
                g.copy_h2d(&[poison], ds.valid.offset(p))?;
            }
        }
        Ok(())
    }

    /// 2026-10-03: Per sequence: its output rows, then for each new row its KV slot, its
    /// indexer key, gate and validity byte.
    pub(crate) fn capture(
        &self,
        seqs: &mut [Seq],
        ks: &[usize],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let g = self.env.gpu;
        let (kv_lora, d, h) = (self.kv_lora, self.d, self.h);
        let mut out = Vec::new();
        let mut o = 0;
        for (s, &k) in seqs.iter_mut().zip(ks) {
            let rows = st::down(g, self.hid.offset(o * h * 2), k * h * 2)?;
            let mut side = Vec::new();
            for j in 0..k {
                let p = s.len + j;
                side.extend(st::down(
                    g,
                    self.kv.k_pool_ptr(0).offset(s.slot(p) * kv_lora),
                    kv_lora,
                )?);
                let ds = s.dsa();
                let off = ds.row_offset(p);
                side.extend(st::down(g, ds.k_normed.offset(off), d * 2)?);
                side.extend(st::down(g, ds.gate.offset(off), d * 2)?);
                side.extend(st::down(g, ds.valid.offset(p), 1)?);
            }
            out.push((rows, side));
            o += k;
        }
        Ok(out)
    }

    /// 2026-10-03: The OFF arm: the call sites' per-sequence loop.
    pub(crate) fn run_off(
        &mut self,
        seqs: &mut [Seq],
        ks: &[usize],
        metas: &[Option<AttnMetadataDev>],
        decode_step: bool,
    ) -> Result<()> {
        self.layer.workspace.set_xseq(None);
        let mut o = 0;
        for (i, s) in seqs.iter_mut().enumerate() {
            let ctx = self.env.ctx(metas[i], decode_step);
            let mut bt = s.blocks.clone();
            self.layer.decode_k(
                self.hid.offset(o * self.h * 2),
                ks[i],
                s.state.as_mut(),
                &mut self.kv,
                s.len,
                &mut bt,
                &ctx,
                self.env.gpu.default_stream(),
                false,
            )?;
            o += ks[i];
        }
        Ok(())
    }

    /// 2026-10-03: The ON arm: `decode_xseq` with the arena attached; errors if it declines.
    pub(crate) fn run_on(
        &mut self,
        seqs: &mut [Seq],
        ks: &[usize],
        metas: &[Option<AttnMetadataDev>],
        decode_step: bool,
    ) -> Result<()> {
        self.layer.workspace.set_xseq(Some(self.arena.clone()));
        let ctx = self.env.ctx(None, decode_step);
        let lens: Vec<usize> = seqs.iter().map(|s| s.len).collect();
        let bts: Vec<Vec<u32>> = seqs.iter().map(|s| s.blocks.clone()).collect();
        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
            seqs.iter_mut().map(|s| &mut *s.state).collect();
        let engaged = self.layer.decode_xseq(
            self.hid,
            ks,
            &mut refs,
            &mut self.kv,
            &lens,
            &bts,
            metas,
            &ctx,
            self.env.gpu.default_stream(),
        )?;
        self.layer.workspace.set_xseq(None);
        if !engaged {
            bail!("decode_xseq declined ks={ks:?}");
        }
        Ok(())
    }
}

pub(crate) fn bf16_diff(a: &[u8], b: &[u8]) -> (usize, f32) {
    let (fa, fb) = (st::row_f32(a), st::row_f32(b));
    let mut n = 0;
    let mut mx = 0f32;
    for (x, y) in fa.iter().zip(&fb) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            mx = mx.max((x - y).abs());
        }
    }
    (n, mx)
}
