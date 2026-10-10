// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Rank 1's side of the two-rank DFlash propose. It receives rank 0's header,
//! makes its ctx rows equal to rank 0's (drop, keep, append the carried rows), then runs the
//! same precompute and forward; its drafts are discarded.
//!
//! When the header cannot be applied (no DFlash state, a position mismatch, no room, no
//! drafter blocks), rank 1 still issues every collective the header implies, on scratch
//! (`tp_dummy`), and sends status 0 in the top-16 all-gather; rank 0 then resends every row
//! on its next propose. Rank 0's drafts for that one propose are garbage (rank 1's partial
//! sums are), which costs acceptance only.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Every collective count and size here is a function of the header and the head's
//!   fixed geometry, never of rank 1's own state.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use metrale_comm::CommBackend;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::speculative::ProposerState;

use super::DraftTp;
use super::header::{self, TpHeader};
use super::shard::{Conform, plan_conform, precompute_chunks};
use crate::dflash_head::{BlockDiffusionDraftHead, DflashProposerState};

/// 2026-10-09: Desyncs seen; the first 8 and every 1000th are logged.
static DESYNCS: AtomicU64 = AtomicU64::new(0);

impl BlockDiffusionDraftHead {
    /// 2026-10-09: Receive one transfer into the stage and return its header words.
    fn tp_recv(
        &self,
        ctx: &ForwardContext,
        comm: &dyn CommBackend,
        tp: &DraftTp,
    ) -> Result<Vec<u32>> {
        comm.broadcast(tp.stage.0, tp.stage_bytes, 0)?;
        let mut b = vec![0u8; tp.hdr_bytes];
        ctx.gpu.copy_d2h(tp.stage, &mut b)?;
        Ok(header::bytes_to_words(&b))
    }

    /// 2026-10-09: Copy the stage's `n` rows to the end of rank 1's ctx rows.
    fn tp_install(
        &self,
        ctx: &ForwardContext,
        tp: &DraftTp,
        d: &mut DflashProposerState,
        positions: &[i32],
        stream: u64,
    ) -> Result<()> {
        let n = positions.len();
        anyhow::ensure!(d.ctx_len + n <= d.max_ctx_len, "TP follower: ctx full");
        if n > 0 {
            ctx.gpu.copy_d2d_async(
                tp.stage.offset(tp.hdr_bytes),
                d.ctx_hidden_acc.offset(d.ctx_len * d.ctx_slot_bytes),
                n * d.ctx_slot_bytes,
                stream,
            )?;
        }
        d.ctx_positions.extend_from_slice(positions);
        d.ctx_len += n;
        Ok(())
    }

    /// 2026-10-09: Drop rank 1's first `c.drop` rows and keep the next `c.keep`. The move
    /// goes in pieces of `c.drop` rows, so no piece's source overlaps its destination, and
    /// later pieces read rows no earlier piece wrote.
    fn tp_conform_rows(
        &self,
        ctx: &ForwardContext,
        d: &mut DflashProposerState,
        c: Conform,
        stream: u64,
    ) -> Result<()> {
        let slot = d.ctx_slot_bytes;
        if c.drop > 0 {
            let mut moved = 0;
            while moved < c.keep {
                let n = (c.keep - moved).min(c.drop);
                ctx.gpu.copy_d2d_async(
                    d.ctx_hidden_acc.offset((c.drop + moved) * slot),
                    d.ctx_hidden_acc.offset(moved * slot),
                    n * slot,
                    stream,
                )?;
                moved += n;
            }
            d.ctx_positions.drain(..c.drop);
        }
        d.ctx_positions.truncate(c.keep);
        d.ctx_len = c.keep;
        Ok(())
    }

    /// 2026-10-09: Rank 1's propose. Returns no drafts.
    pub(super) fn tp_follow(
        &self,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        comm: &dyn CommBackend,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let tp = self.tp.as_ref().expect("tp_follow needs TP state");
        let hdr = TpHeader::decode_main(&self.tp_recv(ctx, comm, tp)?, tp.inline_cap)?;
        if hdr.is_abort() {
            return Ok(Vec::new());
        }
        // 2026-10-09: These size the collectives; rank 0 built them from the same head
        // geometry and env, so a mismatch is a misconfigured serve.
        let width = hdr.width as usize;
        anyhow::ensure!(
            (2..=self.gamma.max(2)).contains(&width) && width <= tp.top_rows,
            "TP follower: header width {width}"
        );
        anyhow::ensure!(
            hdr.chunk as usize <= self.ctx_window
                && (hdr.chunk > 0 || hdr.committed == hdr.ctx_len),
            "TP follower: header chunk {} vs ctx_window {}",
            hdr.chunk,
            self.ctx_window
        );
        self.block_gamma.store(width, Ordering::Relaxed);

        let mut why: Option<String> = None;
        let mut dstate = state.as_any_mut().downcast_mut::<DflashProposerState>();
        if dstate.is_none() {
            why = Some("no DFlash proposer state".into());
        }
        if let Some(d) = dstate.as_deref_mut() {
            let plan = plan_conform(
                &d.ctx_positions,
                d.max_ctx_len,
                hdr.keep as usize,
                hdr.first_pos,
                hdr.keep_last_pos,
                hdr.ctx_len as usize,
            )
            .and_then(|c| {
                anyhow::ensure!(
                    d.ctx_slot_bytes == tp.ctx_slot_bytes,
                    "ctx row size {} vs {}",
                    d.ctx_slot_bytes,
                    tp.ctx_slot_bytes
                );
                // 2026-10-09: Rows moved, so their cached K/V is at the wrong slots
                // unless the header recomputes it (rank 0 resets `committed` on a slide).
                anyhow::ensure!(
                    c.drop == 0 || hdr.committed == 0,
                    "drop {} with committed {}",
                    c.drop,
                    hdr.committed
                );
                Ok(c)
            });
            match plan {
                Ok(c) => self.tp_conform_rows(ctx, d, c, stream)?,
                Err(e) => why = Some(format!("{e:#}")),
            }
        }
        // 2026-10-09: Every transfer is received whatever the outcome; rows are installed
        // only while rank 1 still matches.
        let rows = header::transfer_rows(hdr.n_new as usize, tp.inline_cap);
        for (t, &n) in rows.iter().enumerate() {
            let positions = if t == 0 {
                Ok(hdr.positions.clone())
            } else {
                header::decode_chunk(&self.tp_recv(ctx, comm, tp)?, n, tp.inline_cap)
            };
            if why.is_some() {
                continue;
            }
            let d = dstate.as_deref_mut().expect("state checked above");
            if let Err(e) = positions.and_then(|p| self.tp_install(ctx, tp, d, &p, stream)) {
                why = Some(format!("{e:#}"));
            }
        }
        if why.is_none() {
            let d = dstate.as_deref_mut().expect("state checked above");
            if d.ctx_len != hdr.ctx_len as usize {
                why = Some(format!(
                    "ctx_len {} after install, header {}",
                    d.ctx_len, hdr.ctx_len
                ));
            } else if let Err(e) = self.ensure_block_table(d, ctx.gpu) {
                why = Some(format!("{e:#}"));
            } else {
                d.ctx_committed = hdr.committed as usize;
            }
        }
        tp.set_status(why.is_none());
        match (why, dstate) {
            (None, Some(d)) => {
                self.tp_run_block(d, &hdr, ctx, stream)?;
            }
            (why, _) => {
                let n = DESYNCS.fetch_add(1, Ordering::Relaxed);
                if n < 8 || n.is_multiple_of(1000) {
                    tracing::warn!(
                        "DFlash TP2 rank 1: cannot apply header (ctx_len {}, keep {}): {} — \
                         collectives on scratch, rank 0 resends next propose (desync #{})",
                        hdr.ctx_len,
                        hdr.keep,
                        why.unwrap_or_default(),
                        n + 1
                    );
                }
                self.tp_dummy(&hdr, ctx, comm, stream)?;
            }
        }
        Ok(Vec::new())
    }

    /// 2026-10-09: The header's collectives with scratch contents, in `tp_run_block`'s order:
    /// one `fc` all-gather per precompute chunk, two all-reduces per layer, then the top-16
    /// all-gather (status 0).
    fn tp_dummy(
        &self,
        hdr: &TpHeader,
        ctx: &ForwardContext,
        comm: &dyn CommBackend,
        stream: u64,
    ) -> Result<()> {
        let tp = self.tp.as_ref().expect("tp_dummy needs TP state");
        let chunks = precompute_chunks(
            hdr.committed as usize,
            hdr.ctx_len as usize,
            hdr.chunk as usize,
        );
        for (_, n) in chunks {
            comm.all_gather(tp.fc_local.0, tp.fc_gather.0, n * tp.fc_len * 2)?;
        }
        let width = hdr.width as usize;
        for _ in 0..2 * self.layers.len() {
            comm.all_reduce_async(
                self.scratch.stream_acc.0,
                width * self.hidden_size * 2,
                stream,
            )?;
        }
        tp.set_status(false);
        self.tp_merge_top16(ctx, width, stream)
    }
}
