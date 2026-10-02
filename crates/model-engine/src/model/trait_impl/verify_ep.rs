// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The multi-rank batched MTP verify (`METRALE_GLM_BATCHED_VERIFY=1` with
//! `METRALE_EP_PROTOCOL=v2`): the worker command that carries the whole batch in one message,
//! rank 0's send, the worker's mirror, and the pure wire and bookkeeping helpers.
//!
//! Wire, rank 0 -> every worker, as `ep_send_verify_batch` sends and `ep_worker_verify_batch`
//! receives it:
//!
//! ```text
//! preamble seq_id = 0, cmd = EP_CMD_VERIFY_BATCH   (v2 shape)
//! n                    (u32)
//! slots[n] ++ ks[n]    (one bulk broadcast, `encode_verify_batch_meta`)
//! tokens[Σ ks]         (one bulk broadcast, sequence-major)
//!   ... both ranks run `decode_verify_batched_dispatch` (identical collectives) ...
//! accepted[n]          (one bulk broadcast from the scheduler, or n x EP_VERIFY_BATCH_ABORT)
//! ```
//!
//! After the verdict the worker stashes each sequence's accepted hidden row and applies the
//! per-sequence rewind, proposer trim and SSM rollback the per-sequence K=4 worker arm applies
//! (`impl_a2.rs` `0xFFFFFFF4`). The proposes that follow arrive as ordinary
//! `EP_CMD_MTP_PROPOSE` commands whose hidden index carries `MTP_HIDDEN_FROM_STASH`.
//!
//! Owner: model-engine (speculative verify).
//! Invariants:
//! - Rank 0 sends only after its own validation of the batch passes, so every word the worker
//!   reads was sent; once sent, rank 0 always sends the verdict words (the scheduler on `Ok`,
//!   `decode_verify_batched` on `Err`). The abort is clean only for a failure both ranks hit at
//!   the same collective (the shape-determined refusals `verify_n_seqs` raises before any launch
//!   do); a rank-0-only failure mid-forward desyncs the ranks, as on the per-sequence path.
//! - The worker orders its sequences as `slots[]`, so batch row `i` is the same sequence on
//!   both ranks.

use anyhow::{Result, bail, ensure};

use super::super::types::TransformerModel;
use super::verify_e2::VERIFY_ROW_CAP;
use crate::traits::{ModelDraft, ModelSsmState, SequenceState, VerifyBatchedOpts};
use metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS;

/// 2026-10-02: EP worker command: run the batched verify rank 0 is about to run (module docs).
pub(in crate::model) const EP_CMD_VERIFY_BATCH: u32 = 0xFFFF_FFF9;

/// 2026-10-02: Verdict word meaning "rank 0's verify failed after the command went out": the
/// worker then skips the bookkeeping. Never a valid accepted count (`verify_batch_commit_plan`
/// refuses any count `>= ks[i]`).
pub(in crate::model) const EP_VERIFY_BATCH_ABORT: u32 = u32::MAX;

/// 2026-10-02: Flag in the `EP_CMD_MTP_PROPOSE` hidden-index word: the head loaded the drafter
/// input from verify stash slot `idx & !flag` (`save_hidden_for_mtp_from_stash`), so the worker
/// loads the same slot. Hidden rows stay below `VERIFY_ROW_CAP` and stash slots below
/// `VERIFY_WY_TABLE_SEQS`, so the flag never meets a real index.
pub(in crate::model) const MTP_HIDDEN_FROM_STASH: u32 = 0x8000_0000;

// 2026-10-02: Above the decode-token range and distinct from every other worker opcode (the
// list `decode_checkpoint/plan.rs` keeps).
const _: () = assert!(EP_CMD_VERIFY_BATCH > 0xFFFF_FFEF, "would decode as a token id");
const _: () = assert!(EP_CMD_VERIFY_BATCH != 0xFFFF_FFE0, "collides with batched decode");
const _: () = assert!(EP_CMD_VERIFY_BATCH != 0xFFFF_FFF0, "collides with prefill chunk");
const _: () = assert!(EP_CMD_VERIFY_BATCH != 0xFFFF_FFF1, "collides with alloc-slot");
const _: () = assert!(EP_CMD_VERIFY_BATCH != 0xFFFF_FFF2, "collides with verify K=2");
const _: () = assert!(EP_CMD_VERIFY_BATCH != 0xFFFF_FFF3, "collides with verify K=3");
const _: () = assert!(EP_CMD_VERIFY_BATCH != 0xFFFF_FFF4, "collides with verify K=4");
const _: () = assert!(
    EP_CMD_VERIFY_BATCH != metrale_model_layers::speculative::EP_CMD_MTP_PROPOSE,
    "collides with MTP propose"
);
const _: () = assert!(
    EP_CMD_VERIFY_BATCH != metrale_model_layers::speculative::glm_dflash::EP_CMD_VERIFY_KGAMMA,
    "collides with the DFlash K=gamma verify"
);
const _: () = assert!(EP_CMD_VERIFY_BATCH != 0xFFFF_FFF7, "reserved: DFlash ctx-commit");
const _: () = assert!(
    EP_CMD_VERIFY_BATCH != super::decode_checkpoint::EP_CMD_DECODE_CKPT,
    "collides with the decode checkpoint"
);
const _: () = assert!(EP_CMD_VERIFY_BATCH != 0xFFFF_FFFF, "collides with shutdown");
const _: () = assert!(
    (VERIFY_ROW_CAP as u64) < MTP_HIDDEN_FROM_STASH as u64
        && (VERIFY_WY_TABLE_SEQS as u64) < MTP_HIDDEN_FROM_STASH as u64,
    "the stash flag would meet a real hidden index"
);

/// 2026-10-02: One sequence's verdict as the worker applies it: `accepted` drafts kept, and
/// `rewind` rejected rows dropped from `seq_len` and `tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::model) struct SeqVerdict {
    pub accepted: usize,
    pub rewind: usize,
}

/// 2026-10-02: `slots[n] ++ ks[n]` as u32 words.
pub(in crate::model) fn encode_verify_batch_meta(slots: &[u32], ks: &[usize]) -> Vec<u32> {
    slots
        .iter()
        .copied()
        .chain(ks.iter().map(|&k| k as u32))
        .collect()
}

/// 2026-10-02: The inverse of [`encode_verify_batch_meta`], validated against the envelope
/// `can_batch_verify_dispatch` admits without DFlash: `2 <= n <= VERIFY_WY_TABLE_SEQS`, every
/// k in 2..=4, `Σ ks <= VERIFY_ROW_CAP`, every slot below `slot_cap` and none repeated.
pub(in crate::model) fn decode_verify_batch_meta(
    words: &[u32],
    slot_cap: usize,
) -> Result<(Vec<usize>, Vec<usize>)> {
    ensure!(
        words.len().is_multiple_of(2),
        "verify batch meta: odd word count {}",
        words.len()
    );
    let n = words.len() / 2;
    ensure!(
        (2..=VERIFY_WY_TABLE_SEQS).contains(&n),
        "verify batch meta: n={n} outside 2..={VERIFY_WY_TABLE_SEQS}"
    );
    let slots: Vec<usize> = words[..n].iter().map(|&w| w as usize).collect();
    let ks: Vec<usize> = words[n..].iter().map(|&w| w as usize).collect();
    ensure!(
        ks.iter().all(|k| (2..=4).contains(k)) && ks.iter().sum::<usize>() <= VERIFY_ROW_CAP,
        "verify batch meta: ks={ks:?} outside 2..=4 or above {VERIFY_ROW_CAP} rows"
    );
    for (i, &s) in slots.iter().enumerate() {
        ensure!(s < slot_cap, "verify batch meta: slot {s} >= capacity {slot_cap}");
        ensure!(!slots[..i].contains(&s), "verify batch meta: slot {s} repeated");
    }
    Ok((slots, ks))
}

/// 2026-10-02: Each sequence's verdict from rank 0's accepted counts: `accepted = words[i]`,
/// which must be below `ks[i]` (at most `ks[i] - 1` drafts), and `rewind = ks[i] - 1 -
/// accepted`. Refuses a length mismatch and any out-of-range count, the abort word included.
pub(in crate::model) fn verify_batch_commit_plan(
    ks: &[usize],
    words: &[u32],
) -> Result<Vec<SeqVerdict>> {
    ensure!(
        words.len() == ks.len(),
        "verify batch verdict: {} words for {} sequences",
        words.len(),
        ks.len()
    );
    ks.iter()
        .zip(words)
        .map(|(&k, &w)| {
            let accepted = w as usize;
            ensure!(
                k >= 1 && accepted < k,
                "verify batch verdict: {accepted} accepted of a {k}-row verify"
            );
            Ok(SeqVerdict {
                accepted,
                rewind: k - 1 - accepted,
            })
        })
        .collect()
}

impl TransformerModel {
    /// 2026-10-02: Whether any layer drives its own verify state
    /// (`decode_verify_multi_own_states`).
    pub(super) fn any_verify_own_states(&self) -> bool {
        self.layers
            .iter()
            .any(|l| l.decode_verify_multi_own_states())
    }

    /// 2026-10-02: Whether the batched verify may run with the comm backend: the multi-rank
    /// protocol live in its v2 shape, no DFlash drafter (its capture is not mirrored), and every
    /// layer driving its own verify state, so the forward's collectives depend only on what
    /// `EP_CMD_VERIFY_BATCH` carries.
    pub(super) fn batched_verify_ep_ok(&self) -> bool {
        self.multi_rank_protocol_active()
            && self.ep_protocol_v2
            && self.dflash_hidden_save.is_none()
            && !self.layers.is_empty()
            && self
                .layers
                .iter()
                .all(|l| l.decode_verify_multi_own_states())
    }

    /// 2026-10-02: Rank 0: send `EP_CMD_VERIFY_BATCH` with the batch (module docs) and return
    /// true. Returns false, sending nothing, without the multi-rank protocol or on a worker.
    /// Validates the batch first, so a refusal sends nothing.
    pub(super) fn ep_send_verify_batch(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &[&mut SequenceState],
    ) -> Result<bool> {
        let Some(comm) = self.comm.as_ref() else {
            return Ok(false);
        };
        if !self.multi_rank_protocol_active() || comm.rank() != 0 {
            return Ok(false);
        }
        ensure!(
            self.ep_protocol_v2,
            "multi-rank batched verify needs METRALE_EP_PROTOCOL=v2"
        );
        ensure!(
            seqs.len() == ks.len() && tokens.len() == ks.iter().sum::<usize>(),
            "multi-rank batched verify: {} seqs, ks={ks:?}, {} tokens",
            seqs.len(),
            tokens.len()
        );
        let slots: Vec<u32> = seqs.iter().map(|s| s.slot_idx as u32).collect();
        let meta = encode_verify_batch_meta(&slots, ks);
        decode_verify_batch_meta(&meta, usize::MAX)?;
        self.ep_broadcast_seq_and_cmd(0, EP_CMD_VERIFY_BATCH, true)?;
        self.ep_broadcast_u32(ks.len() as u32)?;
        self.ep_broadcast_tokens(&meta)?;
        self.ep_broadcast_tokens(tokens)?;
        tracing::debug!("EP batched verify sent: slots={slots:?} ks={ks:?}");
        Ok(true)
    }

    /// 2026-10-02: Rank 0, after a sent batch failed: the verdict words the worker waits for,
    /// all `EP_VERIFY_BATCH_ABORT`. A send failure is logged; the error being returned is the
    /// verify's.
    pub(super) fn ep_send_verify_batch_abort(&self, n: usize) {
        if let Err(e) = self.ep_broadcast_tokens(&vec![EP_VERIFY_BATCH_ABORT; n]) {
            tracing::error!("EP batched verify abort words: {e:#}");
        }
    }

    /// 2026-10-02: Worker side of `EP_CMD_VERIFY_BATCH` (module docs). Reads the batch, runs the
    /// verify rank 0 runs over the addressed slots in rank 0's order, always reads the verdict
    /// words, then applies them. An error before the forward (bad meta, an unallocated slot) is
    /// returned at once, as `ep_worker_decode_batch` does.
    pub(in crate::model) fn ep_worker_verify_batch(
        &self,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)? as usize;
        ensure!(
            (2..=VERIFY_WY_TABLE_SEQS).contains(&n),
            "ep_worker_verify_batch: n={n} outside 2..={VERIFY_WY_TABLE_SEQS}"
        );
        let meta = self.ep_broadcast_tokens(&vec![0u32; 2 * n])?;
        let (slot_ids, ks) = decode_verify_batch_meta(&meta, slots.len())?;
        let r_total: usize = ks.iter().sum();
        let tokens = self.ep_broadcast_tokens(&vec![0u32; r_total])?;

        // 2026-10-02: Disjoint `&mut` refs ordered as the head's `slots[]`, by the
        // `swap_remove` walk `ep_worker_decode_batch` uses.
        let mut slot_refs: Vec<(usize, &mut SequenceState)> = slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, opt)| opt.as_mut().map(|s| (i, s)))
            .collect();
        let mut refs: Vec<&mut SequenceState> = Vec::with_capacity(n);
        for &idx in &slot_ids {
            let pos = slot_refs
                .iter()
                .position(|(i, _)| *i == idx)
                .ok_or_else(|| {
                    anyhow::anyhow!("ep_worker_verify_batch: slot {idx} not allocated")
                })?;
            refs.push(slot_refs.swap_remove(pos).1);
        }

        // 2026-10-02: As the K=4 worker arm: order the previous step's rollback copies first.
        self.sync_secondary()?;
        let opts = VerifyBatchedOpts {
            write_on_accept: true,
        };
        let fwd = self
            .ssm_pool
            .require_verify_rollback_supported()
            .and_then(|()| self.decode_verify_batched_dispatch(&tokens, &ks, &mut refs, 0, opts));
        let fwd = self.release_verify_capture_on_err(fwd);
        // 2026-10-02: Read the verdict whatever the forward returned: rank 0 sends it either way,
        // and leaving it unread would make it the next command.
        let words = self.ep_broadcast_tokens(&vec![0u32; n])?;
        fwd?;
        if words.iter().all(|&w| w == EP_VERIFY_BATCH_ABORT) {
            bail!("ep_worker_verify_batch: rank 0 abandoned the batch (slots={slot_ids:?})");
        }
        let plan = verify_batch_commit_plan(&ks, &words)?;

        for (seq, v) in refs.iter_mut().zip(&plan) {
            self.trim_proposer_state(seq, v.accepted, 0)?;
            if v.rewind == 0 {
                self.start_checkpoint_async(seq)?;
            } else {
                seq.seq_len -= v.rewind;
                for _ in 0..v.rewind {
                    seq.tokens.pop();
                }
                self.start_rollback_and_checkpoint_async(seq, v.accepted + 1)?;
            }
        }
        // 2026-10-02: The accepted hidden rows into stash slots `0..n` before any propose
        // overwrites them; the head's `step_verify_k4_batched` stashes the same rows. After the
        // bookkeeping, so a stash error cannot leave this rank's lengths behind rank 0's.
        let mut row = 0usize;
        let mut stash_rows = Vec::with_capacity(n);
        for (k, v) in ks.iter().zip(&plan) {
            stash_rows.push(row + v.accepted);
            row += k;
        }
        self.stash_verify_hidden_rows_dispatch(&stash_rows, 0)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_round_trips_in_batch_order() {
        let w = encode_verify_batch_meta(&[5, 0, 3], &[4, 2, 3]);
        assert_eq!(w, [5, 0, 3, 4, 2, 3]);
        let (slots, ks) = decode_verify_batch_meta(&w, 8).expect("valid");
        assert_eq!(slots, [5, 0, 3]);
        assert_eq!(ks, [4, 2, 3]);
    }

    #[test]
    fn meta_refuses_what_can_batch_verify_refuses() {
        // 2026-10-02: one sequence, odd length, k outside 2..=4, slot past capacity, repeat.
        assert!(decode_verify_batch_meta(&[1, 4], 8).is_err());
        assert!(decode_verify_batch_meta(&[1, 2, 4], 8).is_err());
        assert!(decode_verify_batch_meta(&encode_verify_batch_meta(&[0, 1], &[4, 5]), 8).is_err());
        assert!(decode_verify_batch_meta(&encode_verify_batch_meta(&[0, 1], &[1, 4]), 8).is_err());
        assert!(decode_verify_batch_meta(&encode_verify_batch_meta(&[0, 8], &[4, 4]), 8).is_err());
        assert!(decode_verify_batch_meta(&encode_verify_batch_meta(&[2, 2], &[4, 4]), 8).is_err());
        let wide = vec![0u32; 2 * (VERIFY_WY_TABLE_SEQS + 1)];
        assert!(decode_verify_batch_meta(&wide, usize::MAX).is_err());
    }

    #[test]
    fn commit_plan_rewinds_the_rejected_rows_per_sequence() {
        let plan = verify_batch_commit_plan(&[4, 4, 3, 2], &[3, 0, 1, 1]).expect("valid");
        assert_eq!(
            plan,
            [
                SeqVerdict { accepted: 3, rewind: 0 },
                SeqVerdict { accepted: 0, rewind: 3 },
                SeqVerdict { accepted: 1, rewind: 1 },
                SeqVerdict { accepted: 1, rewind: 0 },
            ]
        );
    }

    #[test]
    fn commit_plan_refuses_overcounts_mismatch_and_the_abort_word() {
        assert!(verify_batch_commit_plan(&[4, 4], &[4, 0]).is_err());
        assert!(verify_batch_commit_plan(&[4, 4], &[1]).is_err());
        assert!(verify_batch_commit_plan(&[4, 4], &[EP_VERIFY_BATCH_ABORT; 2]).is_err());
    }
}
