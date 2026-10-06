// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The graph-replay verify step of `dsa_pool_cache_parity_microtest`, split out of
//! `pool_cache_parity_rig.rs` (500-line cap): both sides' K = 3 step graphs, captured once per
//! B state and replayed per step as `decode_k_rows` replays them.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GraphHandle;
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::select::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, contiguous_pool_count, select_tokens,
    topk_tile,
};
use metrale_model_arch::glm5next_dsa::state::max_dsa_context;

use super::{D, KP, Pair, TOPK, VROWS, i32s};

impl Pair<'_> {
    /// 2026-10-06: A K = 3 verify step by graph replay (`decode_k_rows` replay-safe, captured
    /// once): B's counter may still be ahead of `seq` (a rejected draft); `replay_room` checks
    /// it first, the replay's `dsa_indexer_store_ring` lowers the device watermark, each row's
    /// `dsa_write_geom_pk` and ceiling select run, then `sync_to` reconciles the host.
    pub(crate) fn verify_graph(&mut self, seq: usize) -> Result<()> {
        let g = self.g;
        self.b.replay_room(seq, VROWS)?;
        self.sync()?;
        let (kb, gb) = self.rows_random(VROWS);
        let row = D * 2;
        // 2026-10-06: A's rows at their absolute positions; B's through the stage buffers.
        g.copy_h2d(&kb, self.a_k.offset(seq * row))?;
        g.copy_h2d(&gb, self.a_g.offset(seq * row))?;
        g.copy_h2d(&[1u8; VROWS], self.a_valid.offset(seq))?;
        for r in 0..VROWS {
            let gr = self.rows[r];
            g.copy_h2d(&kb[r * row..(r + 1) * row], gr.stage_k)?;
            g.copy_h2d(&gb[r * row..(r + 1) * row], gr.stage_g)?;
            g.copy_h2d(&i32s([seq + r]), gr.pos)?;
            g.copy_h2d(&i32s([seq + r + 1]), gr.sl)?;
            g.copy_h2d(&i32s([seq + r]), gr.q_pos)?;
        }
        if self.graphs.is_none() {
            self.graphs = Some(self.capture()?);
        }
        let (ga, gb_) = self.graphs.expect("captured");
        g.launch_graph(ga, self.s)?;
        g.launch_graph(gb_, self.s)?;
        self.b.sync_to(seq, VROWS)?;
        self.a_len = seq + VROWS;
        let rows: Vec<(usize, usize)> = (0..VROWS).map(|r| (r, 1)).collect();
        self.compare(seq + VROWS, &rows, "verify (graph replay)")?;
        Ok(())
    }

    /// 2026-10-06: Capture A's (`dsa_write_geom` + ceiling select per row) and B's
    /// (`dsa_indexer_store_ring` + `dsa_write_geom_pk` + ceiling select per row) step graphs,
    /// arguments as `store_indexer_row`, `decode_k_rows` and `select_row_at` pass them.
    fn capture(&self) -> Result<(GraphHandle, GraphHandle)> {
        let (g, cfg, s) = (self.g, self.cfg, self.s);
        let cap = max_dsa_context(&cfg);
        let launch = DsaSelectLaunch::Ceiling {
            max_pools: contiguous_pool_count(KP, cap),
        };
        let geom = DsaSelectGeometry::plan(&cfg, cap, 1)?;
        let pc = self.b.pool_cache().expect("pool cache");
        let ring = pc.book.ring_rows();
        let mut b_args = pc.select_args();
        b_args.pk_start = 0;
        let one = |b: bool| -> Result<()> {
            for (r, gr) in self.rows.iter().enumerate() {
                let geom_dev = if b { gr.geom_b } else { gr.geom_a };
                if b {
                    KernelLaunch::new(g, self.k.indexer_store_ring)
                        .grid([1, 1, 1])
                        .block([D as u32, 1, 1])
                        .arg_ptr(gr.stage_k)
                        .arg_ptr(gr.stage_g)
                        .arg_ptr(gr.pos)
                        .arg_ptr(self.b.k_normed)
                        .arg_ptr(self.b.gate)
                        .arg_ptr(self.b.valid)
                        .arg_u32(D as u32)
                        .arg_u32(ring as u32)
                        .arg_u32(KP as u32)
                        .arg_ptr(pc.pk_len_dev)
                        .launch(s)?;
                }
                let wg = if b {
                    self.k.write_geom_pk
                } else {
                    self.k.write_geom
                };
                let mut l = KernelLaunch::new(g, wg)
                    .grid([1, 1, 1])
                    .block([1, 1, 1])
                    .arg_ptr(gr.sl)
                    .arg_ptr(geom_dev)
                    .arg_u32(KP as u32)
                    .arg_u32(TOPK as u32)
                    .arg_u32(topk_tile() as u32);
                if b {
                    l = l.arg_ptr(pc.pk_len_dev);
                }
                l.launch(s)?;
                let inputs = DsaSelectInputs {
                    k_normed: if b { self.b.k_normed } else { self.a_k },
                    gate: if b { self.b.gate } else { self.a_g },
                    valid: if b { self.b.valid } else { self.a_valid },
                    ape: self.ape,
                    q: self.q,
                    weights: self.w,
                    q_pos: gr.q_pos,
                    q_mask: self.q_mask,
                    first_key: 0,
                    geom_dev,
                    pool_cache: if b { Some(b_args) } else { None },
                };
                let scr = if b { &self.b_scr } else { &self.a_scr };
                select_tokens(
                    g,
                    &self.k,
                    &cfg,
                    &geom,
                    &inputs,
                    &scr.row(r, &cfg),
                    launch,
                    s,
                )?;
            }
            Ok(())
        };
        let mut out = [GraphHandle(0); 2];
        for (i, b) in [false, true].into_iter().enumerate() {
            g.begin_capture(s)?;
            if let Err(e) = one(b) {
                g.abort_capture_if_active(s);
                return Err(e);
            }
            out[i] = g.end_capture(s)?;
        }
        Ok((out[0], out[1]))
    }
}
