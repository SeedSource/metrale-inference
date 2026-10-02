// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelEp`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use crate::traits::SequenceState;
use anyhow::Result;

// 2026-09-26: Only the `ep_worker_step` doc link names it.
#[cfg(doc)]
use super::EpCommandFailed;

/// 2026-09-26: The multi-rank (EP) command protocol.
pub trait ModelEp {
    /// 2026-09-25: EP worker step, on a rank above 0: receive a `(seq_id, cmd)` from rank 0 and
    /// run the command in slot `seq_id`. Returns `Ok(false)` when the worker should shut down.
    ///
    /// An `Err` carrying [`EpCommandFailed`] means the command ran and failed, and the worker
    /// stays up; any other `Err` means the receive failed, and the worker exits.
    ///
    /// `slots` is sized to `--max-batch-size`; a `seq_id >= slots.len()` fails. Default:
    /// `Ok(true)`.
    fn ep_worker_step(&self, _slots: &mut [Option<SequenceState>]) -> Result<bool> {
        Ok(true)
    }

    /// 2026-09-25: Whether the model runs the multi-rank command protocol. The scheduler then
    /// turns off its single-GPU fused paths, the mixed forward among them. Default `false`.
    fn is_ep(&self) -> bool {
        false
    }

    /// 2026-09-25: Send a command word to every worker rank. Default: `Ok(())`.
    fn ep_broadcast_cmd(&self, _cmd: u32) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Send a `(seq_id, cmd)` pair to every worker rank, as the first broadcast of a
    /// command; its follow-up broadcasts use [`Self::ep_broadcast_cmd`]. When
    /// [`Self::ep_protocol_v2`] is false, `seq_id` is not sent. Default: `Ok(())`.
    fn ep_broadcast_cmd_for_seq(&self, _seq_id: u32, _cmd: u32) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Whether the EP protocol carries a `seq_id` with each command
    /// (`METRALE_EP_PROTOCOL=v2`). Default `false`.
    fn ep_protocol_v2(&self) -> bool {
        false
    }

    /// 2026-09-25: Broadcast a token array to every worker rank in one broadcast. Default: an
    /// empty `Vec`.
    fn ep_broadcast_tokens(&self, _tokens: &[u32]) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }

    /// 2026-10-02: Batched-verify commit agreement (`verify_ep.rs`): one gather of every rank's
    /// failure mask (bit i = sequence i's commit failed), returning their OR. Call it only after
    /// the verdict words went out, on every path, as the workers do. Default: `local`.
    fn ep_agree_verify_failures(&self, local: u32) -> Result<u32> {
        Ok(local)
    }
}
