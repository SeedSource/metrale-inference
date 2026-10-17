// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Rank 0's side of the two-rank DFlash propose, the header transfers, and the
//! precompute-plus-forward both ranks run once their ctx rows agree.
//!
//! Order on rank 0: every step that can fail without a collective (stream check, ctx
//! append, block table, keep/resync bookkeeping) runs first; a failure there still sends a
//! header, flagged abort, so rank 1 (already inside its propose) returns instead of
//! waiting. After the header, the collective schedule is fixed by the header alone.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Rank 0 sends exactly one main transfer per propose that reaches `propose_tp`.
//! - `tp_synced_pos` advances only after a full forward, so a failed propose resends its
//!   rows next time.

use anyhow::Result;
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::speculative::ProposerState;

use super::DraftTp;
use super::collectives::tp_comm;
use super::header::{self, FLAG_RESYNC, TpHeader};
use super::shard::{leader_keep, precompute_chunks};
use crate::dflash_head::propose_prep::DRAFT_BLOCK_SIZE;
use crate::dflash_head::{BlockDiffusionDraftHead, DflashProposerState};

fn dflash_state(state: &mut dyn ProposerState) -> Result<&mut DflashProposerState> {
    state
        .as_any_mut()
        .downcast_mut::<DflashProposerState>()
        .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))
}

impl BlockDiffusionDraftHead {
    /// 2026-10-09: `DraftProposer::propose` under `METRALE_DFLASH_DRAFT_TP2`. Rank 0 returns
    /// the drafts; rank 1 (`tp_follow`) returns none.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::dflash_head) fn propose_tp(
        &self,
        last_token: u32,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let tp = self.tp.as_ref().expect("propose_tp needs TP state");
        let comm = tp_comm(ctx, tp)?;
        if tp.rank() != 0 {
            return self.tp_follow(state, ctx, comm, stream);
        }
        let prep = self.tp_lead_prep(
            tp,
            last_token,
            position,
            num_drafts,
            state,
            ctx,
            stream,
            target_hidden_stack,
        );
        let hdr = match prep {
            Ok(h) => h,
            Err(e) => {
                self.tp_send(ctx, comm, tp, &TpHeader::abort(), DevicePtr::NULL, stream)?;
                return Err(e);
            }
        };
        let dstate = dflash_state(state)?;
        self.tp_send(ctx, comm, tp, &hdr, dstate.ctx_hidden_acc, stream)?;
        tp.set_status(true);
        let drafts = self.tp_run_block(dstate, &hdr, ctx, stream)?;
        dstate.tp_synced_pos = dstate.ctx_positions.last().copied();
        // 2026-10-09: As `propose_drafts`: drop the anchor row, then the cap.
        let cap = self.levers.draft_cap.unwrap_or(self.block_g());
        let drafts = if self.mask_token_id != 0 && drafts.len() > 1 {
            drafts[1..].to_vec()
        } else {
            drafts
        };
        let drafts: Vec<u32> = drafts.into_iter().take(cap).collect();
        dstate.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    /// 2026-10-09: Rank 0's prep, with no collective: the ctx append `propose_drafts` does,
    /// the block table, and the header. `keep` is the leading rows rank 1 already holds
    /// (`leader_keep`); none (first sync, or rank 1 reported a desync) resends every row
    /// and recomputes every row's K/V on both ranks.
    #[allow(clippy::too_many_arguments)]
    fn tp_lead_prep(
        &self,
        tp: &DraftTp,
        last_token: u32,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<TpHeader> {
        anyhow::ensure!(
            stream == ctx.gpu.default_stream(),
            "TP drafter propose must run on the default (communicator) stream"
        );
        self.set_block_g(num_drafts);
        let dstate = dflash_state(state)?;
        let eagle_skip = dstate.skip_next_decode_append;
        dstate.skip_next_decode_append = false;
        if !self.levers.no_decode_append
            && !eagle_skip
            && let Some(latest_ctx) = target_hidden_stack
        {
            // 2026-10-17 (A201): a full window slides (A200) instead of skipping the append;
            // the slide sets `ctx_committed = 0`, and the follower conforms from the header.
            if let Some(s) = dstate.decode_append(
                ctx.gpu,
                latest_ctx,
                position.saturating_sub(1) as i32,
                stream,
            )? {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::info!(
                        "DFlash ctx slide (decode append, TP leader): dropped {} oldest, keep {} (window {}); first at this site",
                        s.drop,
                        s.keep,
                        dstate.max_ctx_len
                    )
                });
            }
        }
        anyhow::ensure!(
            dstate.ctx_positions.len() == dstate.ctx_len
                && dstate.ctx_slot_bytes == tp.ctx_slot_bytes,
            "TP drafter: {} positions for {} ctx rows",
            dstate.ctx_positions.len(),
            dstate.ctx_len
        );
        self.ensure_block_table(dstate, ctx.gpu)?;
        let resync = tp
            .peer_desync
            .swap(false, std::sync::atomic::Ordering::Relaxed);
        let synced = if resync { None } else { dstate.tp_synced_pos };
        let keep = leader_keep(&dstate.ctx_positions, synced);
        let committed = if keep == 0 || self.levers.full_precompute {
            0
        } else {
            dstate.ctx_committed.min(keep)
        };
        dstate.ctx_committed = committed;
        let l0 = dstate.ctx_len;
        if resync {
            tracing::warn!("DFlash TP2: rank 1 reported a desync; resending {l0} ctx rows");
        }
        Ok(TpHeader {
            flags: if keep == 0 && l0 > 0 { FLAG_RESYNC } else { 0 },
            last_token,
            position: position as u32,
            width: self.block_g() as u32,
            ctx_len: l0 as u32,
            committed: committed as u32,
            chunk: self.ctx_window as u32,
            keep: keep as u32,
            first_pos: dstate.ctx_positions.first().copied().unwrap_or(-1),
            keep_last_pos: if keep > 0 {
                dstate.ctx_positions[keep - 1]
            } else {
                -1
            },
            n_new: (l0 - keep) as u32,
            positions: dstate.ctx_positions[keep..].to_vec(),
        })
    }

    /// 2026-10-09: Broadcast the main transfer and any chunk transfers. Rows
    /// `[keep, ctx_len)` of `acc` ride along (`acc` is unused for an abort header).
    fn tp_send(
        &self,
        ctx: &ForwardContext,
        comm: &dyn CommBackend,
        tp: &DraftTp,
        hdr: &TpHeader,
        acc: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let cap = tp.inline_cap;
        let slot = tp.ctx_slot_bytes;
        let rows = header::transfer_rows(hdr.n_new as usize, cap);
        let mut row = hdr.keep as usize;
        let mut done = 0usize;
        for t in 0..rows.len().max(1) {
            let n = rows.get(t).copied().unwrap_or(0);
            let words = if t == 0 {
                hdr.encode_main(cap)
            } else {
                header::encode_chunk(&hdr.positions[done..done + n], cap)
            };
            ctx.gpu
                .copy_h2d(&header::words_to_bytes(&words), tp.stage)?;
            if n > 0 {
                ctx.gpu.copy_d2d_async(
                    acc.offset(row * slot),
                    tp.stage.offset(tp.hdr_bytes),
                    n * slot,
                    stream,
                )?;
            }
            comm.broadcast(tp.stage.0, tp.stage_bytes, 0)?;
            row += n;
            done += n;
        }
        Ok(())
    }

    /// 2026-10-09: Both ranks, once their ctx rows match the header: the K/V precompute of
    /// rows `[committed, ctx_len)` in `chunk`-row pieces (one `fc` all-gather each), then the
    /// block forward (two all-reduces per layer, one top-16 all-gather). Returns every row's
    /// token, as `forward_block` does.
    pub(super) fn tp_run_block(
        &self,
        dstate: &mut DflashProposerState,
        hdr: &TpHeader,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let l0 = hdr.ctx_len as usize;
        let bt = dstate
            .block_table_dev
            .ok_or_else(|| anyhow::anyhow!("TP drafter: no block table"))?;
        let slots = self.scratch.slot_mapping_dev;
        for (s, n) in precompute_chunks(hdr.committed as usize, l0, hdr.chunk as usize) {
            metrale_model_layers::layers::ops::fill_slots_from_block_table(
                ctx.gpu,
                self.kernels.fill_slots,
                slots,
                bt,
                s as u32,
                n as u32,
                DRAFT_BLOCK_SIZE as u32,
                stream,
            )?;
            self.precompute_ctx_kv(
                dstate.ctx_hidden_acc,
                s,
                n,
                &dstate.ctx_positions[s..s + n],
                slots,
                ctx,
                stream,
                true,
            )?;
        }
        dstate.ctx_committed = l0;
        dstate.ctx_count_drafter = l0;
        let ctx_count = if self.levers.option_b_no_ctx {
            0
        } else {
            l0 as u32
        };
        self.forward_block(
            hdr.last_token,
            hdr.position as usize,
            ctx,
            stream,
            (l0 > 0).then_some((dstate.ctx_hidden_acc, l0)),
            Some((bt, ctx_count)),
            None,
        )
    }
}
