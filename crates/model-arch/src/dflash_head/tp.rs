// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Two-rank tensor-parallel DFlash drafter, `METRALE_DFLASH_DRAFT_TP2=1`
//! (default off).
//!
//! Why: at C=1 the GLM-5.3 target runs EP2 over two GB10s, and the drafter propose
//! (~10.9 ms/step, bandwidth-bound) ran on rank 0 alone while rank 1 waited in an NCCL
//! broadcast (nsys, SeedHQ/seed-skills#82). Megatron sharding halves each rank's weight
//! traffic:
//! - q/k/v and gate/up column-parallel (whole KV-head groups per rank); o and down
//!   row-parallel, each followed by one BF16 all-reduce of `[width, hidden]`;
//! - `fc` column-parallel, its two halves joined by one all-gather per precompute chunk;
//! - lm_head vocab-parallel: each rank takes its shard's top-16 per row, one all-gather of
//!   16 (value, id) pairs per row, then a host merge (`shard::merge_top16_rows`).
//! - The drafter KV cache is head-sharded: each rank caches its own KV heads.
//!
//! Collectives per single-sequence propose (on top of the EP preamble and the four
//! `EP_CMD_MTP_PROPOSE` words): 1 header broadcast (`hdr_bytes + (gamma+1)` ctx rows), one
//! `fc` all-gather per precompute chunk (1 in steady decode), `2 * num_layers` all-reduces,
//! 1 top-16 all-gather.
//!
//! Lockstep: the engine sends `EP_CMD_MTP_PROPOSE` because `needs_comm()` is true, and the
//! worker enters `propose` (`impl_a2.rs`). Rank 1 holds the prompt's ctx rows from its own
//! (ungated) prefill capture; every later row, and every value that steers its control flow,
//! comes in rank 0's header (`header.rs`). Rank 1 never takes a branch rank 0 did not
//! announce: when it cannot apply the header it still issues the header's collectives on
//! scratch and reports the desync in the top-16 all-gather, and rank 0 resends every row
//! on its next propose.
//!
//! Numerics: the row-parallel sums and the BF16 all-reduce round differently from one
//! rank's single GEMM, so logits can differ in the last bits; that changes proposals
//! (acceptance) only, never the verified output.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Single sequence only: `propose_batch` declines and `propose_batch_max` is 1.
//! - The propose runs on the communicator's legacy stream (the GPU default stream), so
//!   collectives and kernels are ordered on one stream. It runs eagerly unless
//!   `METRALE_DFLASH_TP_GRAPHS=1`, which graphs only the pieces between collectives
//!   (`forward_block/tp_graphs.rs`); no collective is ever captured.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::weight_map::DenseWeight;

use super::BlockDiffusionDraftHead;
use crate::weight_loader::DflashWeights;

mod collectives;
mod follower;
pub(super) mod header;
mod leader;
pub(super) mod shard;
pub(super) mod shard_weights;
#[cfg(test)]
mod tests;

use shard::{RankLayout, Split, TpGeometry, WeightPart};

/// 2026-10-09: What `from_weights` builds with: the plan (`None` = one rank), this rank's
/// `(num_q_heads, num_kv_heads, intermediate_size, vocab)` and its BF16 lm_head rows.
pub struct TpPrep {
    pub plan: Option<TpPlan>,
    pub dims: (usize, usize, usize, usize),
    pub lm_head: DevicePtr,
}

/// 2026-10-09: The construction-time split: this rank, the full geometry and its slices.
#[derive(Clone, Debug)]
pub struct TpPlan {
    pub rank: usize,
    pub world: usize,
    pub geom: TpGeometry,
    pub layout: RankLayout,
}

impl TpPlan {
    /// 2026-10-09: `Some` when the lever is on and this is a two-rank serve. Errors (fail
    /// fast) when the lever is on and the drafter cannot be split as built: no paged cache,
    /// an NVFP4 lm_head, a DSpark Markov head, or a geometry `TpGeometry::check` rejects.
    pub fn decide(
        lever: bool,
        rank_world: Option<(usize, usize)>,
        weights: &DflashWeights,
        target_hidden_size: usize,
        option_b: bool,
        nvfp4_lm_head: bool,
    ) -> Result<Option<Self>> {
        if !lever {
            return Ok(None);
        }
        let Some((rank, world)) = rank_world.filter(|&(_, w)| w == 2) else {
            tracing::warn!(
                "METRALE_DFLASH_DRAFT_TP2=1 needs a two-rank serve (got {rank_world:?}); \
                 the drafter runs on rank 0 alone"
            );
            return Ok(None);
        };
        let c = &weights.config;
        ensure!(
            option_b,
            "METRALE_DFLASH_DRAFT_TP2 needs the paged drafter cache (METRALE_DFLASH_OPTION_B)"
        );
        ensure!(
            !nvfp4_lm_head,
            "METRALE_DFLASH_DRAFT_TP2 does not split an NVFP4 lm_head"
        );
        ensure!(
            weights.markov_w1.is_none() && weights.selector_pred.is_some(),
            "METRALE_DFLASH_DRAFT_TP2 needs a DFlash2 selector drafter without a Markov head"
        );
        let geom = TpGeometry {
            world,
            q_heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
            hidden: c.hidden_size,
            inter: c.intermediate_size,
            vocab: c.vocab_size,
            fc_in: c
                .dflash_config
                .as_ref()
                .map_or(0, |d| d.target_layer_ids.len())
                * target_hidden_size,
        };
        let layout = geom.layout(rank)?;
        Ok(Some(Self {
            rank,
            world,
            geom,
            layout,
        }))
    }

    /// 2026-10-09: This rank's `(num_q_heads, num_kv_heads, intermediate_size, vocab)`.
    pub fn local_dims(&self) -> (usize, usize, usize, usize) {
        (
            self.layout.q_heads.1,
            self.layout.kv_heads.1,
            self.layout.inter.1,
            self.layout.vocab.1,
        )
    }

    /// 2026-10-09: Point the column-parallel weights at this rank's rows (q/k/v, gate/up,
    /// `fc`); BF16 rows are contiguous, so each is a pointer offset. The row-parallel o/down
    /// stay full here and are sliced at FP8 quantization (`shard_weights::quantize_cols`).
    pub fn shard_row_views(&self, weights: &mut DflashWeights) {
        let view = |w: &DenseWeight, name: &str| {
            let p = self.part(name);
            debug_assert_eq!(p.split, Split::Rows);
            DenseWeight {
                weight: w.weight.offset(p.start * p.k * 2),
            }
        };
        for layer in weights.layers.iter_mut() {
            layer.q_proj = view(&layer.q_proj, "q_proj");
            layer.k_proj = view(&layer.k_proj, "k_proj");
            layer.v_proj = view(&layer.v_proj, "v_proj");
            layer.gate_proj = view(&layer.gate_proj, "gate_proj");
            layer.up_proj = view(&layer.up_proj, "up_proj");
        }
        weights.fc = view(&weights.fc, "fc");
    }

    /// 2026-10-09: This rank's part of the weight `name` (`shard::weight_parts`, whose
    /// coverage `tests.rs` checks); every slice applied on the device comes from here.
    pub fn part(&self, name: &str) -> WeightPart {
        let parts = shard::weight_parts(&self.geom, self.rank).expect("plan geometry checked");
        *parts
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("no TP weight part {name}"))
    }

    /// 2026-10-09: [`Self::decide`]; with a plan, shard the column-parallel views
    /// ([`Self::shard_row_views`]) and return this rank's dims and lm_head rows.
    pub fn prepare(
        levers: &super::levers::DFlashLevers,
        rank_world: Option<(usize, usize)>,
        weights: &mut DflashWeights,
        target_hidden_size: usize,
        (lm_head, nvfp4_lm_head): (DevicePtr, bool),
    ) -> Result<TpPrep> {
        let c = &weights.config;
        let full = (
            c.num_attention_heads,
            c.num_key_value_heads,
            c.intermediate_size,
            c.vocab_size,
        );
        let plan = Self::decide(
            levers.draft_tp2,
            rank_world,
            weights,
            target_hidden_size,
            levers.option_b,
            nvfp4_lm_head,
        )?;
        let Some(p) = plan else {
            return Ok(TpPrep {
                plan: None,
                dims: full,
                lm_head,
            });
        };
        p.shard_row_views(weights);
        Ok(TpPrep {
            dims: p.local_dims(),
            lm_head: p.lm_head_rows(lm_head),
            plan: Some(p),
        })
    }

    /// 2026-10-09: This rank's rows of the BF16 lm_head `[vocab, hidden]`.
    pub fn lm_head_rows(&self, lm_head: DevicePtr) -> DevicePtr {
        let p = self.part("lm_head");
        lm_head.offset(p.start * p.k * 2)
    }
}

/// 2026-10-09: The live TP state of a drafter head: the plan and the device scratch of the
/// header transfers, the `fc` all-gather and the top-16 all-gather.
pub struct DraftTp {
    pub plan: TpPlan,
    /// 2026-10-09: Every rank's first vocab id, by rank.
    pub vocab_starts: Vec<u32>,
    /// 2026-10-09: `fc` output columns per rank (`hidden / world`).
    pub fc_len: usize,
    /// 2026-10-09: Ctx rows one header transfer carries (`gamma + 1`).
    pub inline_cap: usize,
    pub hdr_bytes: usize,
    pub ctx_slot_bytes: usize,
    /// 2026-10-09: `[hdr_bytes][inline_cap ctx rows]`, one transfer.
    pub stage: DevicePtr,
    pub stage_bytes: usize,
    /// 2026-10-09: `[ctx_window, fc_len]` BF16, this rank's `fc` output.
    pub fc_local: DevicePtr,
    /// 2026-10-09: `[world, ctx_window, fc_len]` BF16, the gathered halves.
    pub fc_gather: DevicePtr,
    /// 2026-10-09: Top-16 all-gather send block and receive (`world` blocks).
    pub top_send: DevicePtr,
    pub top_recv: DevicePtr,
    /// 2026-10-09: Rows the top-16 buffers hold (`gamma`).
    pub top_rows: usize,
    /// 2026-10-09: Rank 0: rank 1 reported it could not apply a header.
    pub peer_desync: AtomicBool,
    /// 2026-10-09: The status byte this rank sends in the next top-16 all-gather: 1 when its
    /// state matches the header of the propose in flight.
    pub local_status: AtomicU32,
}

impl DraftTp {
    /// 2026-10-09: Allocate the TP scratch for `head` (already built with local geometry).
    pub fn new(plan: TpPlan, head: &BlockDiffusionDraftHead, gpu: &dyn GpuBackend) -> Result<Self> {
        let world = plan.world;
        let vocab_starts = (0..world)
            .map(|r| plan.geom.layout(r).map(|l| l.vocab.0 as u32))
            .collect::<Result<Vec<_>>>()?;
        let fc_len = plan.layout.fc_out.1;
        let inline_cap = head.gamma + 1;
        let hdr_bytes = header::hdr_bytes(inline_cap);
        let ctx_slot_bytes = head.target_layer_ids.len() * head.target_hidden_size * 2;
        let stage_bytes = hdr_bytes + inline_cap * ctx_slot_bytes;
        let top_rows = head.gamma.max(2);
        let top_block = top_block_bytes(top_rows);
        let fc_bytes = head.ctx_window * fc_len * 2;
        tracing::info!(
            "METRALE_DFLASH_DRAFT_TP2: rank {}/{world} heads q{:?} kv{:?} inter{:?} vocab{:?} \
             fc{:?}; scratch {:.1} MB",
            plan.rank,
            plan.layout.q_heads,
            plan.layout.kv_heads,
            plan.layout.inter,
            plan.layout.vocab,
            plan.layout.fc_out,
            (stage_bytes + fc_bytes * (1 + world) + top_block * (1 + world)) as f64 / 1e6,
        );
        Ok(Self {
            vocab_starts,
            fc_len,
            inline_cap,
            hdr_bytes,
            ctx_slot_bytes,
            stage: gpu.alloc(stage_bytes)?,
            stage_bytes,
            fc_local: gpu.alloc(fc_bytes)?,
            fc_gather: gpu.alloc(fc_bytes * world)?,
            top_send: gpu.alloc(top_block)?,
            top_recv: gpu.alloc(top_block * world)?,
            top_rows,
            peer_desync: AtomicBool::new(false),
            local_status: AtomicU32::new(1),
            plan,
        })
    }

    pub fn rank(&self) -> usize {
        self.plan.rank
    }

    pub(super) fn set_status(&self, ok: bool) {
        self.local_status.store(u32::from(ok), Ordering::Relaxed);
    }
}

/// 2026-10-09: Bytes of one rank's top-16 all-gather block for `rows` rows: values (f32),
/// ids (u32), then 16 status bytes.
pub fn top_block_bytes(rows: usize) -> usize {
    rows * shard::TOP_K * 8 + 16
}

impl BlockDiffusionDraftHead {
    /// 2026-10-09: Checks after the head is built: the TP propose needs the FP8 weights
    /// (row-parallel o/down exist only as FP8 slices) and the DFlash2 selector tail.
    pub(super) fn tp_validate(&self) -> Result<()> {
        if self.tp.is_none() {
            return Ok(());
        }
        ensure!(
            matches!(self.quant, super::DflashQuantization::Fp8Weights),
            "METRALE_DFLASH_DRAFT_TP2 needs the FP8 drafter (METRALE_DFLASH_DRAFTER_FP8 \
             not 0, and its kernels present)"
        );
        ensure!(
            self.dflash2_active(),
            "METRALE_DFLASH_DRAFT_TP2 needs the DFlash2 selector path (kernels present, \
             METRALE_DFLASH2 not 0)"
        );
        Ok(())
    }
}
