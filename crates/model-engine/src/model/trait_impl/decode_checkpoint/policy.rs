// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Which Marconi SSM snapshot families this process saves: all of them (the
//! default), or the prefill ones only (`METRALE_MARCONI_PREFILL_ONLY=1`).
//!
//! Prefill-only turns off the decode checkpoints (`decode_marconi_checkpoint_dispatch`) and the
//! finish-leaf snapshot (`finish_leaf_snapshot`). The prefill snapshots stay on: interval
//! checkpoints, the prompt-tail checkpoint, the exact leaf and the mid-chunk tail. The reasons
//! are reasoned from source, not measured on GPU (PROVISIONAL):
//! - A decode-time snapshot covers generated tokens. The next turn can restore it only when its
//!   match reaches past the previous prompt. Two things stop the match there. A chat template
//!   that drops earlier turns' reasoning re-renders the assistant turn with tokens other than
//!   the generated ones. And on a multi-rank world the EP worker never inserts generated tokens
//!   into its radix tree, because `insert_intermediate_snapshot` is index-only and
//!   `cache_sequence` runs only in rank 0's scheduler, so the F83 minimum caps the match at the
//!   previous prompt. Decode checkpoints still take a pool slot every
//!   `METRALE_DECODE_CKPT_BLOCKS` blocks (default 4), and recency eviction drops the older
//!   prompt-tail checkpoint first: the one checkpoint the next turn can restore.
//! - Only rank 0 saves the finish leaf. On a multi-rank world it gives rank 0 one more pool
//!   entry and one more eviction than the worker, so the pools drift, and the A100 vote
//!   (`snap_agree`) then refuses restores both ranks could have served.
//!
//! The prefill snapshots are saved on every rank from the same tokens, at the same chunk ends.
//!
//! Owner: model-engine prefix cache.
//! Invariants:
//! - With the variable unset, or set to anything but `1`, every decision here is the one the
//!   code made before this lever existed.
//! - Only rank 0 consults the policy for a decode checkpoint. The worker saves what rank 0
//!   commands (`decode_marconi_checkpoint_worker`), so a decode checkpoint is never saved on
//!   one rank only because the ranks' environments differ.

use std::sync::OnceLock;

/// 2026-10-01: The snapshot families this process saves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::model) enum SnapshotPolicy {
    /// 2026-10-01: Every family; the default.
    All,
    /// 2026-10-01: Prefill snapshots only (`METRALE_MARCONI_PREFILL_ONLY=1`).
    PrefillOnly,
}

impl SnapshotPolicy {
    /// 2026-10-01: The policy for a raw `METRALE_MARCONI_PREFILL_ONLY` value: `1` selects
    /// prefill-only. Anything else, unset included, selects `All`.
    pub(in crate::model) fn from_env_value(value: Option<&str>) -> Self {
        if value == Some("1") {
            Self::PrefillOnly
        } else {
            Self::All
        }
    }

    /// 2026-10-01: Whether rank 0 saves decode-time checkpoints.
    pub(in crate::model) fn saves_decode_checkpoints(self) -> bool {
        self == Self::All
    }

    /// 2026-10-01: Whether the retire path saves the finish-leaf snapshot.
    pub(in crate::model) fn saves_finish_leaf(self) -> bool {
        self == Self::All
    }
}

/// 2026-10-01: This process's policy, read once and cached.
pub(in crate::model) fn snapshot_policy() -> SnapshotPolicy {
    static POLICY: OnceLock<SnapshotPolicy> = OnceLock::new();
    *POLICY.get_or_init(|| {
        let raw = std::env::var("METRALE_MARCONI_PREFILL_ONLY").ok();
        let policy = SnapshotPolicy::from_env_value(raw.as_deref());
        if policy == SnapshotPolicy::PrefillOnly {
            tracing::warn!(
                "METRALE_MARCONI_PREFILL_ONLY=1: Marconi decode checkpoints and finish-leaf \
                 snapshots are off; only prefill snapshots are saved"
            );
        }
        policy
    })
}

/// 2026-10-01: `CkptInputs::enabled` for a decode checkpoint: the snapshot pool and the prefix
/// cache are on, and `policy` saves decode checkpoints. With `SnapshotPolicy::All` this is the
/// expression the decode path used before the lever existed.
pub(in crate::model) fn decode_ckpt_enabled(
    snapshots_enabled: bool,
    prefix_cache_active: bool,
    policy: SnapshotPolicy,
) -> bool {
    snapshots_enabled && prefix_cache_active && policy.saves_decode_checkpoints()
}

#[cfg(test)]
#[path = "policy_tests.rs"]
mod tests;
