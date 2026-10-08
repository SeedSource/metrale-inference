// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The EP worker's SSM state handling after an MTP verify verdict (`impl_a2.rs`
//! `0xFFFFFFF2/3/4`, `verify_ep.rs` `EP_CMD_VERIFY_BATCH`).
//!
//! By default the worker keeps its historical semantics: a full accept checkpoints `h_state`
//! and `conv_state` (`start_checkpoint_async`), and a partial accept restores the intermediate
//! after the last accepted row and checkpoints that (`start_rollback_and_checkpoint_async`).
//!
//! `METRALE_EP_WORKER_PREFIX_COMMIT=1` gives the worker rank 0's semantics instead: the
//! scheduler's `commit_accepted_prefix` (full accept: nothing; partial: the same intermediate
//! restore). The live `h_state` / `conv_state` a verdict leaves are the same bytes either way;
//! only the copy into `h_state_checkpoint` / `conv_state_checkpoint` is dropped. Those buffers
//! are read only by a rollback to `num_accepted == 0` (`start_rollback_and_checkpoint_async`,
//! `rollback_ssm_states`), which no caller issues: every verdict commits at least the anchor
//! row.
//!
//! Owner: model-engine (speculative verify, EP worker).
//! Invariants:
//! - `rows` is the committed row count, anchor included: `1..=k`; `rows == k` is a full accept.
//! - Rank 0 never calls this; its behaviour does not depend on the lever.

use anyhow::Result;

use super::super::types::TransformerModel;
use crate::traits::{ModelSsmState, SequenceState};

/// 2026-10-08: `METRALE_EP_WORKER_PREFIX_COMMIT=1` (module docs). Read once per process.
fn worker_prefix_commit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_EP_WORKER_PREFIX_COMMIT").as_deref() == Ok("1"))
}

impl TransformerModel {
    /// 2026-10-08: Commit `rows` of a `k`-row verify on the worker (module docs). The caller
    /// has already rewound `seq_len` / `tokens` and trimmed the proposer, as before.
    pub(in crate::model) fn ep_worker_commit_rows(
        &self,
        seq: &mut SequenceState,
        rows: usize,
        k: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            (1..=k).contains(&rows),
            "ep_worker_commit_rows: {rows} rows of a {k}-row verify"
        );
        if !worker_prefix_commit() {
            return if rows == k {
                self.start_checkpoint_async(seq)
            } else {
                self.start_rollback_and_checkpoint_async(seq, rows)
            };
        }
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| tracing::info!("METRALE_EP_WORKER_PREFIX_COMMIT: ENGAGED"));
        // 2026-10-08: The flush the `start_*` arms run first; a no-op unless a carried verify
        // left rows pending (never on GLM-5.3, whose layers do not carry).
        self.gdn_carry_flush_pending()?;
        self.commit_accepted_prefix_dispatch(seq, rows, k)
    }
}
