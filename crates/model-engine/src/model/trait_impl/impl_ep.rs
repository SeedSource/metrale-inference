// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl ModelEp for TransformerModel`, mostly delegating to `<method>_dispatch`
//! helpers in the sibling modules.
//!
//! Owner: model-engine.
//! Invariants: the ones in `trait_impl/mod.rs`.

use anyhow::Result;

use crate::model::types::TransformerModel;
use crate::traits::{ModelEp, SequenceState};

impl ModelEp for TransformerModel {
    fn ep_worker_step(&self, slots: &mut [Option<SequenceState>]) -> Result<bool> {
        self.gdn_carry_flush_pending()?;
        self.ep_worker_step_dispatch(slots)
    }

    fn is_ep(&self) -> bool {
        self.is_ep_dispatch()
    }

    fn ep_broadcast_cmd(&self, cmd: u32) -> Result<()> {
        self.ep_broadcast_cmd_dispatch(cmd)
    }

    fn ep_broadcast_cmd_for_seq(&self, seq_id: u32, cmd: u32) -> Result<()> {
        // 2026-09-25: `ep_protocol_v2` is set at construction from
        // `METRALE_EP_PROTOCOL=v2`.
        self.ep_broadcast_seq_and_cmd(seq_id, cmd, self.ep_protocol_v2)
    }

    fn ep_protocol_v2(&self) -> bool {
        self.ep_protocol_v2
    }

    fn ep_broadcast_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        self.ep_broadcast_tokens_dispatch(tokens)
    }

    fn ep_agree_verify_failures(&self, local: u32) -> Result<u32> {
        self.ep_agree_verify_failures_impl(local)
    }
}
