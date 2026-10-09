// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The model-side pieces of the GLM-5.3 DFlash batched verify
//! (`METRALE_GLM_DFLASH_BATCHED_VERIFY`, model-layers `glm_dflash_batch.rs`): the DFlash batch
//! shape `can_batch_verify_dispatch` admits, and the worker's verify hidden stash.
//!
//! The wire is `EP_CMD_VERIFY_BATCH` unchanged (`verify_ep.rs`): with a DFlash drafter the
//! forward's tap capture runs on rank 0 alone and only rank 0's `commit_ctx` reads it, so the
//! worker mirrors the forward, the verdict and the commit exactly as for MTP.
//!
//! Owner: model-engine (speculative verify).
//! Invariants:
//! - With a comm backend a DFlash batch is admitted only at k <= 4, the widths
//!   `decode_verify_batch_meta` carries, so `ep_send_verify_batch` never refuses an admitted
//!   batch at send.
//! - The worker skips the stash only when no worker propose can read it (lever on, proposer
//!   on rank 0 alone); otherwise the stash and its fail-every-sequence error path are as before.

use anyhow::Result;
use metrale_model_layers::speculative::glm_dflash_batch as gdb;

use super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-10-09: The DFlash clause of `can_batch_verify_dispatch` (`gdb::dflash_batch_shape_ok`).
    pub(super) fn dflash_batch_shape_ok(&self, ks: &[usize]) -> bool {
        gdb::dflash_batch_shape_ok(
            ks,
            self.dflash_kgamma,
            self.dflash_hidden_save_rows,
            self.comm.is_some(),
        )
    }

    /// 2026-10-09: The worker's stash of each sequence's accepted hidden row after a batched
    /// verify (`ep_worker_verify_batch`). Skipped under the DFlash batched lever when the
    /// proposer runs on rank 0 alone (`gdb::worker_skips_verify_stash`).
    pub(super) fn ep_worker_stash_verify_rows(&self, rows: &[usize]) -> Result<()> {
        let needs_comm = self.proposer.as_ref().map(|p| p.needs_comm());
        if gdb::worker_skips_verify_stash(gdb::glm_dflash_batched_verify(), needs_comm) {
            return Ok(());
        }
        self.stash_verify_hidden_rows_dispatch(rows, 0)
    }
}
