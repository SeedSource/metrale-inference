// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_DFLASH_TP_GRAPHS=1` (default off): CUDA graphs for the two-rank
//! (`METRALE_DFLASH_DRAFT_TP2`) DFlash propose, captured between its collectives.
//!
//! Why: the TP propose runs fully eagerly (`forward_block`'s single-rank graph path is off
//! under TP), ~150 launches per C=1 round with ~0.67 ms of GPU idle between them (rank-0
//! nsys, cardf arm, PROPOSE-SPLIT 2026-10-12).
//!
//! What is captured: the propose's segment plan ([`tp_segment_plan`]) is the eager TP
//! forward cut at every collective and at attention. Per layer: pre-attention, o_proj,
//! the MLP up to down_proj, and the residual tail (`PostAttnPart`); then the lm_head with
//! the top-16, and the selector projection with the walk: `4 * num_layers + 2` graphs per
//! block width. Attention (eager in the single-rank graph path too), the two row-parallel
//! all-reduces per layer, the top-16 all-gather with its host merge, and everything
//! before the layers (ctx precompute with its `fc` all-gather, embed, slot fill, header
//! broadcast) stay eager.
//!
//! No NCCL operation is ever captured. NCCL requires a captured operation to be captured
//! on every participating rank and launched from the graph of that collective capture
//! ("Using NCCL with CUDA Graphs", NCCL user guide); keeping every collective eager means
//! the ranks never have to agree on graph state, so rank 1's `tp_dummy` path, a capture
//! failure on one rank, or different warm-up counts cannot desynchronise the collectives.
//! Both ranks still key their graphs by the header's block width.
//!
//! Exactness: a replay launches the kernels its capture recorded, with the arguments they
//! had, in stream order. Every captured segment is device launches into fixed buffers
//! (scratch, weights, the paged pool) whose scalar arguments depend only on the block
//! width, the key; the per-propose host inputs (positions, token ids, attention's
//! indirect args, the merged top-16) are written to fixed device buffers by eager steps.
//! The eager TP forward (`forward_block_layer_pre_attn`, attention, `post_attn_layer`,
//! `block_tail_select`) runs exactly this plan's segments in this order, so the graphed
//! propose issues the same kernels on the same inputs in the same order: byte-identical
//! drafts.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - No `TpSeg` that issues a collective or a host sync is captured ([`TpSeg::graphed`]).
//! - A segment whose capture fails runs eagerly at once (a capture records without
//!   executing) and keeps a zero handle, so it runs eagerly on every replay.

use std::sync::Once;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GraphHandle};
use metrale_model_layers::layer::ForwardContext;

use super::dims::BlockDims;
use crate::dflash_head::BlockDiffusionDraftHead;
use crate::dflash_head::forward_block_layer_paged::{PagedLayerArgs, PostAttnPart};

/// 2026-10-12: The engaged line, printed once per process (one per rank).
static ENGAGED: Once = Once::new();

/// 2026-10-12: One step of the TP propose's layers and tail, in eager order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TpSeg {
    /// 2026-10-12: `forward_block_layer_pre_attn` of layer `l`.
    Pre(usize),
    /// 2026-10-12: Paged attention of layer `l` (eager, as on the single-rank path).
    Attn(usize),
    /// 2026-10-12: One piece of layer `l`'s post-attention call.
    Post(usize, PostAttnPart),
    /// 2026-10-12: The row-parallel all-reduce of `stream_acc` after o_proj or down_proj.
    Reduce(usize),
    /// 2026-10-12: Final norm, lm_head and the per-row top-16.
    TailLogits,
    /// 2026-10-12: The top-16 all-gather, D2H, host merge and H2D (`tp_merge_top16`).
    Merge,
    /// 2026-10-12: Selector projection and chain walk.
    TailPick,
}

impl TpSeg {
    /// 2026-10-12: Captured in a graph: device launches only, no collective, no host sync.
    pub(super) fn graphed(self) -> bool {
        matches!(
            self,
            TpSeg::Pre(_) | TpSeg::Post(..) | TpSeg::TailLogits | TpSeg::TailPick
        )
    }

    /// 2026-10-12: Issues an NCCL collective.
    #[cfg(test)]
    pub(super) fn collective(self) -> bool {
        matches!(self, TpSeg::Reduce(_) | TpSeg::Merge)
    }
}

/// 2026-10-12: The eager TP forward of `num_layers` layers and the DFlash2 tail, as segments.
pub(super) fn tp_segment_plan(num_layers: usize) -> Vec<TpSeg> {
    let mut plan = Vec::with_capacity(num_layers * 7 + 3);
    for l in 0..num_layers {
        plan.extend([
            TpSeg::Pre(l),
            TpSeg::Attn(l),
            TpSeg::Post(l, PostAttnPart::OProj),
            TpSeg::Reduce(l),
            TpSeg::Post(l, PostAttnPart::Mlp),
            TpSeg::Reduce(l),
            TpSeg::Post(l, PostAttnPart::Out),
        ]);
    }
    plan.extend([TpSeg::TailLogits, TpSeg::Merge, TpSeg::TailPick]);
    plan
}

/// 2026-10-12: Graphs per block width: `4 * num_layers + 2`.
pub(super) fn tp_graph_slots(num_layers: usize) -> usize {
    tp_segment_plan(num_layers)
        .into_iter()
        .filter(|s| s.graphed())
        .count()
}

impl BlockDiffusionDraftHead {
    /// 2026-10-12: Boot-time note when the lever is set on a head that will not use it.
    pub(in crate::dflash_head) fn log_tp_graphs_lever(&self) {
        if self.levers.tp_graphs && self.tp.is_none() {
            tracing::warn!(
                "METRALE_DFLASH_TP_GRAPHS=1: IDLE (no two-rank drafter: needs \
                 METRALE_DFLASH_DRAFT_TP2=1 on a two-rank serve); single-rank propose graphs \
                 are unchanged"
            );
        }
    }

    /// 2026-10-12: Run one segment eagerly. `args` is layer `l`'s `PagedLayerArgs`.
    #[allow(clippy::too_many_arguments)]
    fn run_tp_seg(
        &self,
        seg: TpSeg,
        d: &BlockDims<'_>,
        ctx: &ForwardContext,
        stream: u64,
        make_paged_args: &impl Fn(usize) -> Option<PagedLayerArgs>,
        stream_noise: DevicePtr,
        norm_noise: DevicePtr,
    ) -> Result<()> {
        let args = |l: usize| make_paged_args(l).expect("option_b args available");
        match seg {
            TpSeg::Pre(l) => {
                self.forward_block_layer_pre_attn(&self.layers[l], &args(l), ctx)?;
            }
            TpSeg::Attn(l) => {
                let (k_pool, v_pool) = {
                    let cache = self.kv_cache.lock();
                    (cache.k_pool_ptr(l), cache.v_pool_ptr(l))
                };
                self.forward_block_layer_attention(&args(l), ctx, k_pool, v_pool)?;
            }
            TpSeg::Post(l, part) => {
                self.post_attn_part(&self.layers[l], &args(l), ctx, part)?;
            }
            TpSeg::Reduce(_) => {
                self.tp_reduce_rows(ctx, self.scratch.stream_acc, d.g, stream)?;
            }
            TpSeg::TailLogits => {
                self.block_tail_logits(d, ctx, stream, stream_noise, norm_noise)?;
            }
            TpSeg::Merge => {
                self.tp_merge_top16(ctx, d.g as usize, stream)?;
            }
            TpSeg::TailPick => {
                self.block_tail_pick(d, ctx, stream, norm_noise)?;
            }
        }
        Ok(())
    }

    /// 2026-10-12: Capture one graphed segment (`run(seg)`) and launch it. A failed capture
    /// is released, the segment runs eagerly, and the slot keeps `GraphHandle(0)`.
    fn capture_tp_seg(
        &self,
        seg: TpSeg,
        ctx: &ForwardContext,
        stream: u64,
        run: &impl Fn(TpSeg) -> Result<()>,
    ) -> Result<GraphHandle> {
        let gpu = ctx.gpu;
        let captured = gpu.begin_capture(stream).and_then(|()| {
            run(seg)?;
            gpu.end_capture(stream)
        });
        match captured {
            Ok(h) if h.0 != 0 => {
                gpu.launch_graph(h, stream)?;
                Ok(h)
            }
            other => {
                if let Err(e) = &other {
                    gpu.abort_capture_if_active(stream);
                    tracing::warn!(
                        "METRALE_DFLASH_TP_GRAPHS: capture of {seg:?} failed ({e:#}); it runs \
                         eagerly"
                    );
                }
                run(seg)?;
                Ok(GraphHandle(0))
            }
        }
    }

    /// 2026-10-12: The TP propose's layers and tail under `METRALE_DFLASH_TP_GRAPHS`:
    /// `propose_warmup_n` eager passes per block width, one capture pass, then replays.
    pub(super) fn forward_block_tp_graphed(
        &self,
        d: &BlockDims<'_>,
        ctx: &ForwardContext,
        stream: u64,
        make_paged_args: &impl Fn(usize) -> Option<PagedLayerArgs>,
        stream_noise: DevicePtr,
        norm_noise: DevicePtr,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let width = d.width;
        let run = |seg: TpSeg| {
            self.run_tp_seg(
                seg,
                d,
                ctx,
                stream,
                make_paged_args,
                stream_noise,
                norm_noise,
            )
        };
        let plan = tp_segment_plan(self.layers.len());
        let slots = tp_graph_slots(self.layers.len());
        let mut g = self.propose_graphs.lock();
        let ready = matches!(g.tp_by_width.get(&width), Some(v) if v.len() == slots);
        if ready {
            let graphs = g.tp_by_width.get(&width).expect("checked above");
            let mut slot = 0;
            for seg in plan {
                if seg.graphed() {
                    let h = graphs[slot];
                    slot += 1;
                    if h.0 != 0 {
                        gpu.launch_graph(h, stream)?;
                        continue;
                    }
                }
                run(seg)?;
            }
            return Ok(());
        }
        let warmed = g.tp_warmup.get(&width).copied().unwrap_or(0);
        if warmed < self.levers.propose_warmup_n {
            g.tp_warmup.insert(width, warmed + 1);
            return plan.into_iter().try_for_each(run);
        }
        let mut handles = Vec::with_capacity(slots);
        for seg in plan {
            if seg.graphed() {
                handles.push(self.capture_tp_seg(seg, ctx, stream, &run)?);
            } else {
                run(seg)?;
            }
        }
        let captured = handles.iter().filter(|h| h.0 != 0).count();
        let rank = self.tp.as_ref().map_or(0, |t| t.rank());
        tracing::info!(
            "DFlash TP piecewise capture: rank {rank}, block width {width}: {captured}/{slots} \
             segments captured"
        );
        ENGAGED.call_once(|| {
            tracing::warn!(
                "METRALE_DFLASH_TP_GRAPHS=1: ENGAGED (rank {rank}, width {width}, \
                 {captured}/{slots} segments graphed; attention, {} all-reduces and the top-16 \
                 merge eager)",
                2 * self.layers.len()
            );
        });
        g.tp_by_width.insert(width, handles);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_keeps_every_collective_out_of_the_graphs() {
        for layers in [1usize, 5, 8] {
            let plan = tp_segment_plan(layers);
            assert!(plan.iter().all(|s| !(s.graphed() && s.collective())));
            assert_eq!(
                plan.iter()
                    .filter(|s| matches!(s, TpSeg::Reduce(_)))
                    .count(),
                2 * layers,
                "two all-reduces per layer"
            );
            assert_eq!(plan.iter().filter(|s| **s == TpSeg::Merge).count(), 1);
            assert!(
                plan.iter()
                    .all(|s| !(s.graphed() && matches!(s, TpSeg::Attn(_)))),
                "attention stays eager"
            );
            assert_eq!(tp_graph_slots(layers), 4 * layers + 2);
        }
    }

    /// 2026-10-12: The plan is the eager TP forward's order: per layer pre, attention,
    /// o_proj, reduce, MLP, reduce, out; then logits, merge, pick. A collective always
    /// sits between two graphed pieces, never at the start of the layer loop.
    #[test]
    fn plan_order_matches_the_eager_forward() {
        let plan = tp_segment_plan(2);
        let want = vec![
            TpSeg::Pre(0),
            TpSeg::Attn(0),
            TpSeg::Post(0, PostAttnPart::OProj),
            TpSeg::Reduce(0),
            TpSeg::Post(0, PostAttnPart::Mlp),
            TpSeg::Reduce(0),
            TpSeg::Post(0, PostAttnPart::Out),
            TpSeg::Pre(1),
            TpSeg::Attn(1),
            TpSeg::Post(1, PostAttnPart::OProj),
            TpSeg::Reduce(1),
            TpSeg::Post(1, PostAttnPart::Mlp),
            TpSeg::Reduce(1),
            TpSeg::Post(1, PostAttnPart::Out),
            TpSeg::TailLogits,
            TpSeg::Merge,
            TpSeg::TailPick,
        ];
        assert_eq!(plan, want);
    }
}
