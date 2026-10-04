// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The batched MTP propose that mirrors the per-sequence one
//! (`DraftProposer::propose_batch_mirrors_serial`, `METRALE_GLM_MTP_BATCH_DRAFT=1`): the same
//! forward context as `run_mtp_propose_inner` (the communicator when `needs_comm`), and on a
//! multi-rank serve the worker command that runs the same batch on every rank.
//!
//! Wire, rank 0 -> every worker, as `ep_send_propose_batch` sends and
//! `ep_worker_propose_batch` receives it:
//!
//! ```text
//! preamble seq_id = 0, cmd = EP_CMD_MTP_PROPOSE_BATCH   (v2 shape)
//! n                    (u32)
//! slots[n] ++ tokens[n] ++ positions[n] ++ stash_idx[n] ++ [num_drafts]
//!                      (one bulk broadcast, `encode_propose_batch`)
//! slots vote           (`ep_gather_u32`, every rank: 1 = every slot allocated with drafter
//!                       state and the stash present; any 0 ends the command on every rank)
//!   ... both ranks run `ensure_drafter_context` per sequence, in batch order ...
//! ready vote           (`ep_gather_u32`: `propose_batch_ready` on every rank; any 0 ends it)
//!   ... both ranks run `propose_batch` (identical collectives); the worker's drafts are
//!   discarded ...
//! ```
//!
//! The drafter input of row i is verify stash slot `stash_idx[i]` on both ranks. Rank 0 sends
//! the command only while its stash is the one the worker also wrote
//! (`ep_stash_mirrored`: set by the batched verify's failure gather, cleared by every later
//! rank-0 stash, such as the batched bootstrap's, which the worker does not mirror).
//!
//! Owner: model-engine (speculative propose).
//! Invariants:
//! - Lever off (`propose_batch_mirrors_serial` false) nothing here runs.
//! - A refused vote ends the command on every rank before any drafter collective of the batch;
//!   rank 0 then proposes per sequence. A failure inside `propose_batch` after both votes
//!   passed desyncs the ranks, as a failed per-sequence propose does.

use anyhow::{Result, bail, ensure};

use super::super::types::TransformerModel;
use super::verify_ep::merge_ok_votes;
use crate::traits::SequenceState;
use metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS;
use metrale_model_layers::speculative::{DraftProposer, ProposerState};

/// 2026-10-04: EP worker command: run the batched MTP propose rank 0 is about to run (module
/// docs).
pub(in crate::model) const EP_CMD_MTP_PROPOSE_BATCH: u32 = 0xFFFF_FFFA;

// 2026-10-04: Above the decode-token range and distinct from every other worker opcode.
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH > 0xFFFF_FFEF,
    "would decode as a token id"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != 0xFFFF_FFE0,
    "collides with batched decode"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != 0xFFFF_FFF0,
    "collides with prefill chunk"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != 0xFFFF_FFF1,
    "collides with alloc-slot"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != 0xFFFF_FFF2,
    "collides with verify K=2"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != 0xFFFF_FFF3,
    "collides with verify K=3"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != 0xFFFF_FFF4,
    "collides with verify K=4"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != metrale_model_layers::speculative::EP_CMD_MTP_PROPOSE,
    "collides with MTP propose"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != metrale_model_layers::speculative::glm_dflash::EP_CMD_VERIFY_KGAMMA,
    "collides with the DFlash K=gamma verify"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != 0xFFFF_FFF7,
    "reserved: DFlash ctx-commit"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != super::decode_checkpoint::EP_CMD_DECODE_CKPT,
    "collides with the decode checkpoint"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != super::verify_ep::EP_CMD_VERIFY_BATCH,
    "collides with the batched verify"
);
const _: () = assert!(
    EP_CMD_MTP_PROPOSE_BATCH != 0xFFFF_FFFF,
    "collides with shutdown"
);

/// 2026-10-04: One decoded `EP_CMD_MTP_PROPOSE_BATCH` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::model) struct ProposeBatchMeta {
    pub slots: Vec<usize>,
    pub tokens: Vec<u32>,
    pub positions: Vec<usize>,
    pub stash_idx: Vec<usize>,
    pub num_drafts: usize,
}

/// 2026-10-04: `slots[n] ++ tokens[n] ++ positions[n] ++ stash_idx[n] ++ [num_drafts]` as u32
/// words. Errors when a slice length differs from `slots.len()` or a value does not fit a u32.
pub(in crate::model) fn encode_propose_batch(
    slots: &[usize],
    tokens: &[u32],
    positions: &[usize],
    stash_idx: &[usize],
    num_drafts: usize,
) -> Result<Vec<u32>> {
    let n = slots.len();
    ensure!(
        tokens.len() == n && positions.len() == n && stash_idx.len() == n,
        "propose batch meta: {n} slots, {} tokens, {} positions, {} stash rows",
        tokens.len(),
        positions.len(),
        stash_idx.len()
    );
    let word = |v: usize| -> Result<u32> {
        u32::try_from(v).map_err(|_| anyhow::anyhow!("propose batch meta: {v} exceeds u32"))
    };
    let mut out = Vec::with_capacity(4 * n + 1);
    for &s in slots {
        out.push(word(s)?);
    }
    out.extend_from_slice(tokens);
    for &p in positions {
        out.push(word(p)?);
    }
    for &i in stash_idx {
        out.push(word(i)?);
    }
    out.push(word(num_drafts)?);
    Ok(out)
}

/// 2026-10-04: The inverse of [`encode_propose_batch`] for `n` sequences, validated: `2 <= n`,
/// slots distinct, stash rows distinct and below `VERIFY_WY_TABLE_SEQS`, `num_drafts >= 1`.
pub(in crate::model) fn decode_propose_batch(words: &[u32], n: usize) -> Result<ProposeBatchMeta> {
    ensure!(
        n >= 2 && words.len() == 4 * n + 1,
        "propose batch meta: {} words for n={n}",
        words.len()
    );
    let col = |c: usize| words[c * n..(c + 1) * n].iter().map(|&w| w as usize);
    let meta = ProposeBatchMeta {
        slots: col(0).collect(),
        tokens: words[n..2 * n].to_vec(),
        positions: col(2).collect(),
        stash_idx: col(3).collect(),
        num_drafts: words[4 * n] as usize,
    };
    ensure!(meta.num_drafts >= 1, "propose batch meta: num_drafts 0");
    for (i, &s) in meta.stash_idx.iter().enumerate() {
        ensure!(
            s < VERIFY_WY_TABLE_SEQS,
            "propose batch meta: stash row {s} >= {VERIFY_WY_TABLE_SEQS}"
        );
        ensure!(
            !meta.stash_idx[..i].contains(&s),
            "propose batch meta: stash row {s} repeated"
        );
    }
    for (i, &s) in meta.slots.iter().enumerate() {
        ensure!(
            !meta.slots[..i].contains(&s),
            "propose batch meta: slot {s} repeated"
        );
    }
    Ok(meta)
}

impl TransformerModel {
    /// 2026-10-04: The batched propose for a proposer whose `propose_batch` mirrors its
    /// `propose` (module docs). `Ok(None)`: rank 0 proposes per sequence (and so does every
    /// rank, through the per-sequence worker handshake).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_mtp_propose_batched_mirrored(
        &self,
        proposer: &dyn DraftProposer,
        tokens: &[u32],
        positions: &[usize],
        stash_idx: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let n = seqs.len();
        if n < 2
            || num_drafts == 0
            || tokens.len() != n
            || positions.len() != n
            || stash_idx.len() != n
            || stash_idx.iter().any(|&i| i >= VERIFY_WY_TABLE_SEQS)
            || seqs.iter().any(|s| s.proposer_state.is_none())
        {
            return Ok(None);
        }
        let ep = self.multi_rank_protocol_active() && proposer.needs_comm();
        if ep {
            if !self.ep_protocol_v2
                || !self
                    .ep_stash_mirrored
                    .load(std::sync::atomic::Ordering::Relaxed)
            {
                return Ok(None);
            }
            let slots: Vec<usize> = seqs.iter().map(|s| s.slot_idx).collect();
            let words = encode_propose_batch(&slots, tokens, positions, stash_idx, num_drafts)?;
            decode_propose_batch(&words, n)?;
            self.ep_broadcast_seq_and_cmd(0, EP_CMD_MTP_PROPOSE_BATCH, true)?;
            self.ep_broadcast_u32(n as u32)?;
            self.ep_broadcast_tokens(&words)?;
            if !merge_ok_votes(&self.ep_gather_u32(1)?) {
                tracing::debug!("EP batched propose: a worker refused the slots ({slots:?})");
                return Ok(None);
            }
        }
        let stream = self.gpu.default_stream();
        let ctx = self.mtp_serial_propose_ctx(proposer);
        for seq in seqs.iter_mut() {
            self.ensure_drafter_context(proposer, seq, &ctx, stream);
        }
        let h = self.config.hidden_size;
        let hiddens: Vec<metrale_gpu_runtime::gpu::DevicePtr> = stash_idx
            .iter()
            .map(|&i| self.verify_hidden_stash.offset(i * h * 2))
            .collect();
        let mut states: Vec<&mut dyn ProposerState> = Vec::with_capacity(n);
        for seq in seqs.iter_mut() {
            if let Some(s) = seq.proposer_state.as_mut() {
                states.push(s.as_mut());
            }
        }
        let ready = proposer.propose_batch_ready(positions, num_drafts, &mut states);
        let all_ready = if ep {
            merge_ok_votes(&self.ep_gather_u32(u32::from(ready))?)
        } else {
            ready
        };
        if !all_ready {
            return Ok(None);
        }
        let out = proposer.propose_batch(
            tokens,
            &hiddens,
            positions,
            num_drafts,
            &mut states,
            &ctx,
            stream,
            out_conf,
        )?;
        if ep && out.is_none() {
            // 2026-10-04: `propose_batch` returns `Ok(None)` only when `propose_batch_ready`
            // is false, which every rank just voted against; reaching here is a contract break.
            bail!("EP batched propose: propose_batch declined after every rank voted ready");
        }
        Ok(out)
    }

    /// 2026-10-04: The forward context `run_mtp_propose_inner` builds: `mtp_propose_ctx` with
    /// the communicator when the proposer `needs_comm`.
    pub(super) fn mtp_serial_propose_ctx(
        &self,
        proposer: &dyn DraftProposer,
    ) -> metrale_model_layers::layer::ForwardContext<'_> {
        let mut ctx = self.mtp_propose_ctx();
        if proposer.needs_comm() {
            ctx.comm = self.comm_ref();
        }
        ctx
    }

    /// 2026-10-04: Worker side of `EP_CMD_MTP_PROPOSE_BATCH` (module docs). Always reads the
    /// whole payload and makes both votes it reaches; the drafts are discarded. Errors after a
    /// refused vote are returned (request-scoped, as `ep_worker_verify_batch`'s).
    pub(in crate::model) fn ep_worker_propose_batch(
        &self,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)? as usize;
        ensure!(
            (2..=VERIFY_WY_TABLE_SEQS).contains(&n),
            "ep_worker_propose_batch: n={n} outside 2..={VERIFY_WY_TABLE_SEQS}"
        );
        let words = self.ep_broadcast_tokens(&vec![0u32; 4 * n + 1])?;
        let pre = decode_propose_batch(&words, n).and_then(|meta| {
            let proposer = self
                .proposer
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("ep_worker_propose_batch: no proposer"))?;
            ensure!(
                !self.verify_hidden_stash.is_null(),
                "ep_worker_propose_batch: verify stash not allocated"
            );
            let refs = propose_batch_refs(slots, &meta.slots)?;
            Ok((meta, proposer, refs))
        });
        let all_ok = merge_ok_votes(&self.ep_gather_u32(u32::from(pre.is_ok()))?);
        let (meta, proposer, mut refs) = match pre {
            Ok(p) if all_ok => p,
            Err(e) => return Err(e),
            Ok(_) => return Ok(true),
        };
        let stream = self.gpu.default_stream();
        let ctx = self.mtp_serial_propose_ctx(proposer);
        for seq in refs.iter_mut() {
            self.ensure_drafter_context(proposer, seq, &ctx, stream);
        }
        let h = self.config.hidden_size;
        let hiddens: Vec<metrale_gpu_runtime::gpu::DevicePtr> = meta
            .stash_idx
            .iter()
            .map(|&i| self.verify_hidden_stash.offset(i * h * 2))
            .collect();
        let mut states: Vec<&mut dyn ProposerState> = Vec::with_capacity(n);
        for seq in refs.iter_mut() {
            if let Some(s) = seq.proposer_state.as_mut() {
                states.push(s.as_mut());
            }
        }
        let ready = states.len() == n
            && proposer.propose_batch_ready(&meta.positions, meta.num_drafts, &mut states);
        if !merge_ok_votes(&self.ep_gather_u32(u32::from(ready))?) {
            return Ok(true);
        }
        if let Err(e) = proposer.propose_batch(
            &meta.tokens,
            &hiddens,
            &meta.positions,
            meta.num_drafts,
            &mut states,
            &ctx,
            stream,
            None,
        ) {
            // 2026-10-04: As the per-sequence worker arm: logged, not returned; rank 0 decides
            // what is verified.
            tracing::warn!("EP worker batched MTP propose failed (continuing): {e:#}");
        }
        Ok(true)
    }
}

/// 2026-10-04: Disjoint `&mut` refs to the addressed slots, ordered as rank 0's `slots[]`, each
/// with drafter state (the `swap_remove` walk of `verify_batch_preflight`).
fn propose_batch_refs<'a>(
    slots: &'a mut [Option<SequenceState>],
    slot_ids: &[usize],
) -> Result<Vec<&'a mut SequenceState>> {
    let mut slot_refs: Vec<(usize, &mut SequenceState)> = slots
        .iter_mut()
        .enumerate()
        .filter_map(|(i, opt)| opt.as_mut().map(|s| (i, s)))
        .collect();
    let mut refs: Vec<&mut SequenceState> = Vec::with_capacity(slot_ids.len());
    for &idx in slot_ids {
        let pos = slot_refs
            .iter()
            .position(|(i, _)| *i == idx)
            .ok_or_else(|| anyhow::anyhow!("ep_worker_propose_batch: slot {idx} not allocated"))?;
        let seq = slot_refs.swap_remove(pos).1;
        ensure!(
            seq.proposer_state.is_some(),
            "ep_worker_propose_batch: slot {idx} has no drafter state"
        );
        refs.push(seq);
    }
    Ok(refs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propose_batch_meta_round_trips_in_batch_order() {
        let w = encode_propose_batch(&[5, 1, 3], &[10, 11, 12], &[100, 2300, 7], &[2, 0, 1], 2)
            .unwrap();
        assert_eq!(w.len(), 13);
        let m = decode_propose_batch(&w, 3).unwrap();
        assert_eq!(m.slots, vec![5, 1, 3]);
        assert_eq!(m.tokens, vec![10, 11, 12]);
        assert_eq!(m.positions, vec![100, 2300, 7]);
        assert_eq!(m.stash_idx, vec![2, 0, 1]);
        assert_eq!(m.num_drafts, 2);
    }

    #[test]
    fn propose_batch_meta_refuses_bad_batches() {
        let enc = |slots: &[usize], stash: &[usize], nd| {
            let n = slots.len();
            encode_propose_batch(slots, &vec![0; n], &vec![0; n], stash, nd).unwrap()
        };
        // 2026-10-04: one sequence, a repeated slot, a repeated or out-of-range stash row,
        // zero drafts, a short payload.
        assert!(decode_propose_batch(&enc(&[0], &[0], 1), 1).is_err());
        assert!(decode_propose_batch(&enc(&[4, 4], &[0, 1], 1), 2).is_err());
        assert!(decode_propose_batch(&enc(&[0, 1], &[1, 1], 1), 2).is_err());
        assert!(decode_propose_batch(&enc(&[0, 1], &[0, VERIFY_WY_TABLE_SEQS], 1), 2).is_err());
        assert!(decode_propose_batch(&enc(&[0, 1], &[0, 1], 0), 2).is_err());
        assert!(decode_propose_batch(&enc(&[0, 1], &[0, 1], 1)[..8], 2).is_err());
        assert!(encode_propose_batch(&[0, 1], &[0], &[0, 0], &[0, 1], 1).is_err());
    }
}
