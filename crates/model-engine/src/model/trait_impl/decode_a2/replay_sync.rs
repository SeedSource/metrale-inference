// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Per-row layer hooks around a batched multi-sequence decode graph replay, as
//! `decode_a`'s single-sequence replay runs them: `check_replay_room` on every row before the
//! launch (GLM-5.3's DSA indexer row is written from a device position, so a replay past its
//! ceiling would write past the buffer) and `sync_replayed_step` after it (a replay runs kernels
//! only; GLM-5.3 advances its DSA indexer's host length here). Both default to no-ops, so other
//! models are unaffected.
//!
//! Owner: model-engine (decode).
//! Invariants: called with each row's pre-step `seq_len`, before the caller advances it.

use anyhow::Result;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-10-07: `check_replay_room` for every layer of every row.
    pub(super) fn ms_replay_check_room(&self, seqs: &[&mut SequenceState]) -> Result<()> {
        for seq in seqs.iter() {
            for (i, layer) in self.layers.iter().enumerate() {
                layer.check_replay_room(&*seq.layer_states[i], seq.seq_len, 1)?;
            }
        }
        Ok(())
    }

    /// 2026-10-07: `sync_replayed_step` for every layer of every row; logs the first replay.
    pub(super) fn ms_replay_sync(&self, seqs: &mut [&mut SequenceState]) -> Result<()> {
        for seq in seqs.iter_mut() {
            let seq_len = seq.seq_len;
            for (i, layer) in self.layers.iter().enumerate() {
                layer.sync_replayed_step(seq.layer_states[i].as_mut(), seq_len, 1)?;
            }
        }
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::warn!(
                "multi-seq decode graph REPLAYED: first replay of a captured {}-row step",
                seqs.len()
            );
        });
        Ok(())
    }
}
