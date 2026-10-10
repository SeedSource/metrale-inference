// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl ModelSsmState for TransformerModel`, mostly delegating to `<method>_dispatch`
//! helpers in the sibling modules.
//!
//! Owner: model-engine.
//! Invariants: the ones in `trait_impl/mod.rs`.

use anyhow::Result;

use crate::model::types::TransformerModel;
use crate::traits::{ModelSsmState, SequenceState};

impl ModelSsmState for TransformerModel {
    fn checkpoint_ssm_states(&self, seq: &mut SequenceState) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        self.checkpoint_ssm_states_dispatch(seq)
    }

    fn rollback_ssm_states(&self, seq: &mut SequenceState, num_accepted: usize) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        self.rollback_ssm_states_dispatch(seq, num_accepted)
    }

    fn has_ssm_layers(&self) -> bool {
        self.ssm_pool.num_ssm_layers > 0
    }

    fn decode_rollback_unsupported(&self) -> bool {
        self.layers.iter().any(|l| l.decode_rollback_unsupported())
    }

    fn decode_rollback_ring_slots(&self) -> usize {
        if self.ssm_snapshots.decode_rollback_enabled() {
            self.ssm_snapshots.decode_ring_slots
        } else {
            0
        }
    }

    fn save_decode_ssm_snapshot(&self, seq: &SequenceState, ring_slot: usize) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        self.save_decode_ssm_snapshot_dispatch(seq, ring_slot)
    }

    fn restore_decode_ssm_snapshot(&self, seq: &SequenceState, ring_slot: usize) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        self.restore_decode_ssm_snapshot_dispatch(seq, ring_slot)
    }

    fn decode_marconi_checkpoint(&self, seq: &mut SequenceState) {
        if let Err(e) = self.gdn_carry_flush_pending() {
            tracing::error!(
                "gdn carry flush before {}: {e:#}",
                "decode_marconi_checkpoint"
            );
        }
        if let Err(e) = self.kda_defer_flush(self.gpu.default_stream()) {
            tracing::error!("KDA deferred-commit flush before decode_marconi_checkpoint: {e:#}");
        }
        self.decode_marconi_checkpoint_dispatch(seq)
    }

    fn ssm_snapshot_occupancy(&self) -> Option<(u32, u32)> {
        let (used, total) = self.ssm_snapshots.occupancy();
        (total > 0).then_some((used as u32, total as u32))
    }

    fn start_checkpoint_async(&self, seq: &mut SequenceState) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        self.start_checkpoint_async_dispatch(seq)
    }

    fn start_rollback_and_checkpoint_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        self.start_rollback_and_checkpoint_async_dispatch(seq, num_accepted)
    }

    fn sync_secondary(&self) -> Result<()> {
        self.sync_secondary_dispatch()
    }

    fn commit_accepted_prefix(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        self.commit_accepted_prefix_dispatch(seq, num_accepted, k)
    }
}
