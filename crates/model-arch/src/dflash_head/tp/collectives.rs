// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The collectives inside the two-rank DFlash propose: the `fc` all-gather,
//! the row-parallel all-reduce and the top-16 all-gather plus host merge. Each is a no-op
//! (or the single-rank op) when the head has no TP state, so the shared forward code calls
//! them unconditionally.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - `stream` is the communicator's legacy stream (the GPU default stream; `propose_tp`
//!   checks it), so `all_gather`, which runs there, is ordered with the kernels around it.

use anyhow::Result;
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::ForwardContext;

use super::shard::{RankTop, TOP_K, merge_top16_rows};
use super::{DraftTp, top_block_bytes};
use crate::dflash_head::BlockDiffusionDraftHead;

/// 2026-10-09: The communicator of a TP propose, checked against the head's plan.
pub(super) fn tp_comm<'a>(ctx: &ForwardContext<'a>, tp: &DraftTp) -> Result<&'a dyn CommBackend> {
    let comm = ctx.comm.ok_or_else(|| {
        anyhow::anyhow!("METRALE_DFLASH_DRAFT_TP2 propose without the communicator")
    })?;
    anyhow::ensure!(
        comm.world_size() == tp.plan.world && comm.rank() == tp.plan.rank,
        "TP drafter built for rank {}/{}, communicator is rank {}/{}",
        tp.plan.rank,
        tp.plan.world,
        comm.rank(),
        comm.world_size()
    );
    Ok(comm)
}

impl BlockDiffusionDraftHead {
    /// 2026-10-09: Ctx precompute step 1, `fc_proj[n, h] = src[n, fc_in] @ fc^T`. With TP each
    /// rank computes its `fc_len` output columns, one all-gather joins them, and two pitched
    /// copies interleave them into `fc_proj` (a byte copy: the same values one rank's GEMM
    /// would write, when the GEMM's per-column results do not depend on N).
    pub(in crate::dflash_head) fn fc_project(
        &self,
        ctx: &ForwardContext,
        src: DevicePtr,
        n: u32,
        fc_in: u32,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden_size;
        // 2026-10-12: METRALE_DFLASH_AUX_FP8: up to 16 rows on the FP8 mirror; BF16 kept.
        let fc = (&self.fc, self.fc_fp8.as_ref(), true);
        let Some(tp) = self.tp.as_ref() else {
            return self.draft_gemm_aux(
                gpu,
                src,
                fc,
                self.scratch.fc_proj,
                n,
                h as u32,
                fc_in,
                stream,
            );
        };
        let comm = tp_comm(ctx, tp)?;
        anyhow::ensure!(n as usize <= self.ctx_window, "fc_project: {n} rows");
        self.draft_gemm_aux(
            gpu,
            src,
            fc,
            tp.fc_local,
            n,
            tp.fc_len as u32,
            fc_in,
            stream,
        )?;
        let part = n as usize * tp.fc_len * 2;
        comm.all_gather(tp.fc_local.0, tp.fc_gather.0, part)?;
        for r in 0..tp.plan.world {
            gpu.copy_d2d_2d_async(
                tp.fc_gather.offset(r * part),
                tp.fc_len * 2,
                self.scratch.fc_proj.offset(r * tp.fc_len * 2),
                h * 2,
                tp.fc_len * 2,
                n as usize,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-10-09: Sum a row-parallel output `[rows, hidden]` BF16 across ranks in place
    /// (after o_proj and after down_proj). No-op without TP.
    pub(in crate::dflash_head) fn tp_reduce_rows(
        &self,
        ctx: &ForwardContext,
        ptr: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let Some(tp) = self.tp.as_ref() else {
            return Ok(());
        };
        let bytes = rows as usize * self.hidden_size * 2;
        tp_comm(ctx, tp)?.all_reduce_async(ptr.0, bytes, stream)
    }

    /// 2026-10-09: After `dflash2_topk16` on this rank's vocab shard: gather every rank's
    /// `[rows, 16]` values and shard-local ids plus a status byte, merge on the host
    /// (`merge_top16_rows`) and write the global top-16 back into `sel_vals`/`sel_idx`, so
    /// the selector walk sees the same candidates on every rank. Rank 0 records a peer's
    /// status 0 in `peer_desync`. No-op without TP.
    pub(in crate::dflash_head) fn tp_merge_top16(
        &self,
        ctx: &ForwardContext,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(tp) = self.tp.as_ref() else {
            return Ok(());
        };
        let comm = tp_comm(ctx, tp)?;
        let gpu = ctx.gpu;
        anyhow::ensure!(rows <= tp.top_rows, "tp_merge_top16: {rows} rows");
        let vb = rows * TOP_K * 4;
        gpu.copy_d2d_async(self.scratch.sel_vals, tp.top_send, vb, stream)?;
        gpu.copy_d2d_async(self.scratch.sel_idx, tp.top_send.offset(vb), vb, stream)?;
        let status = tp.local_status.load(std::sync::atomic::Ordering::Relaxed) as u8;
        gpu.memset_async(tp.top_send.offset(2 * vb), status, 16, stream)?;
        let part = top_block_bytes(rows);
        comm.all_gather(tp.top_send.0, tp.top_recv.0, part)?;
        let mut host = vec![0u8; part * tp.plan.world];
        gpu.copy_d2h(tp.top_recv, &mut host)?;
        let mut vals: Vec<Vec<f32>> = Vec::with_capacity(tp.plan.world);
        let mut ids: Vec<Vec<u32>> = Vec::with_capacity(tp.plan.world);
        for r in 0..tp.plan.world {
            let b = &host[r * part..(r + 1) * part];
            vals.push(
                b[..vb]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            );
            ids.push(
                b[vb..2 * vb]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            );
            if r != tp.plan.rank && b[2 * vb] != 1 {
                tp.peer_desync
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let parts: Vec<RankTop<'_>> = (0..tp.plan.world)
            .map(|r| RankTop {
                vals: &vals[r],
                ids: &ids[r],
                vocab_start: tp.vocab_starts[r],
            })
            .collect();
        let (mv, mi) = merge_top16_rows(rows, &parts)?;
        let vbytes: Vec<u8> = mv.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ibytes: Vec<u8> = mi.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d(&vbytes, self.scratch.sel_vals)?;
        gpu.copy_h2d(&ibytes, self.scratch.sel_idx)?;
        Ok(())
    }
}
