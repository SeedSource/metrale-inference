// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Hybrid speculative decode (`METRALE_HYBRID_DFLASH=1`, issue #82): the routing that
//! lets one model hold two proposers. `proposer` keeps the GLM MTP head and `dflash_proposer` the
//! DFlash drafter; each sequence holds a state for each (`proposer_state`, `dflash_state`). Rank 0
//! sets the step's proposer (`set_spec_step_mode`) and carries it to the workers in
//! `EP_CMD_SPEC_MODE`; every propose and verify trim then reaches the step's proposer through
//! [`TransformerModel::step_proposer`] and [`TransformerModel::step_state_mut`].
//!
//! Owner: model-engine (speculative).
//! Invariants:
//! - Without `dflash_proposer` (every serve but the hybrid one) each helper returns exactly
//!   what the code it replaced read: `proposer`, `proposer_state`, `dflash_hidden_save.is_some()`.
//! - A worker's mode changes only in [`TransformerModel::ep_worker_set_spec_mode`].
//! - The DFlash context is appended only through `commit_ctx` on a hybrid serve; the older
//!   append paths ([`TransformerModel::legacy_dflash_state`]) are inert there, so no row is
//!   appended twice.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use metrale_model_layers::speculative::hybrid::{self as hy, SpecStepMode};
use metrale_model_layers::speculative::{DraftProposer, ProposerState};

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-10-10: Whether this is a hybrid serve (a DFlash drafter beside the MTP head).
    #[inline]
    pub(in crate::model) fn hybrid_on(&self) -> bool {
        self.dflash_proposer.is_some()
    }

    /// 2026-10-10: The current step's mode (hybrid serves only; `Mtp` until rank 0 sets one).
    #[inline]
    pub(in crate::model) fn spec_mode_now(&self) -> SpecStepMode {
        SpecStepMode::from_u8(self.spec_mode.load(Ordering::Relaxed)).unwrap_or(SpecStepMode::Mtp)
    }

    /// 2026-10-10: Whether the current step runs the DFlash drafter: on a hybrid serve its mode is
    /// `Dflash`; on any other serve, whether the model captures DFlash taps (a DFlash serve), the
    /// serve-wide test this replaces.
    #[inline]
    pub(in crate::model) fn dflash_step(&self) -> bool {
        if self.hybrid_on() {
            self.spec_mode_now() == SpecStepMode::Dflash
        } else {
            self.dflash_hidden_save.is_some()
        }
    }

    /// 2026-10-10: The proposer the current step runs: the DFlash drafter in a hybrid `Dflash`
    /// step, else `proposer`.
    #[inline]
    pub(in crate::model) fn step_proposer(&self) -> Option<&Arc<dyn DraftProposer>> {
        if self.hybrid_on() && self.spec_mode_now() == SpecStepMode::Dflash {
            self.dflash_proposer.as_ref()
        } else {
            self.proposer.as_ref()
        }
    }

    /// 2026-10-10: `seq`'s state for [`Self::step_proposer`].
    #[inline]
    pub(in crate::model) fn step_state_mut<'a>(
        &self,
        seq: &'a mut SequenceState,
    ) -> Option<&'a mut Box<dyn ProposerState>> {
        if self.hybrid_on() && self.spec_mode_now() == SpecStepMode::Dflash {
            seq.dflash_state.as_mut()
        } else {
            seq.proposer_state.as_mut()
        }
    }

    /// 2026-10-10: The DFlash drafter whatever the step: `dflash_proposer` on a hybrid serve,
    /// else `proposer` (a DFlash drafter only on a DFlash serve).
    #[inline]
    pub(in crate::model) fn dflash_head(&self) -> Option<&Arc<dyn DraftProposer>> {
        if self.hybrid_on() {
            self.dflash_proposer.as_ref()
        } else {
            self.proposer.as_ref()
        }
    }

    /// 2026-10-10: `seq`'s DFlash drafter state whatever the step, for the tap capture and
    /// `commit_ctx`: `dflash_state` on a hybrid serve, else `proposer_state` (the callers
    /// downcast and return when it is not a DFlash state).
    #[inline]
    pub(in crate::model) fn dflash_ctx_state<'a>(
        &self,
        seq: &'a mut SequenceState,
    ) -> Option<&'a mut Box<dyn ProposerState>> {
        if self.hybrid_on() {
            seq.dflash_state.as_mut()
        } else {
            seq.proposer_state.as_mut()
        }
    }

    /// 2026-10-10: `seq`'s DFlash state for the pre-unified append paths
    /// (`dflash_accept_append`, `dflash_eagle_*_append`, `dflash_serial_ctx_append`): `None` on a
    /// hybrid serve, which appends through `commit_ctx` only, else `proposer_state`.
    #[inline]
    pub(in crate::model) fn legacy_dflash_state<'a>(
        &self,
        seq: &'a mut SequenceState,
    ) -> Option<&'a mut Box<dyn ProposerState>> {
        if self.hybrid_on() {
            None
        } else {
            seq.proposer_state.as_mut()
        }
    }

    /// 2026-10-10: `Model::set_spec_step_mode` on rank 0. A no-op on a non-hybrid serve. On a
    /// change it first sends the workers `EP_CMD_SPEC_MODE` (multi-rank only), then switches this
    /// rank, so no worker command of the new step can arrive before the mode does.
    pub(super) fn set_spec_step_mode_dispatch(&self, mode: SpecStepMode) -> Result<()> {
        if !self.hybrid_on() {
            return Ok(());
        }
        if self.spec_mode.load(Ordering::Relaxed) == mode as u8 {
            return Ok(());
        }
        if let Some(c) = self.comm.as_ref()
            && c.rank() != 0
        {
            bail!(
                "set_spec_step_mode on rank {}: only rank 0 decides the step mode",
                c.rank()
            );
        }
        if self.multi_rank_protocol_active() {
            self.ep_broadcast_seq_and_cmd(0, hy::EP_CMD_SPEC_MODE, self.ep_protocol_v2)?;
            self.ep_broadcast_u32(hy::encode_spec_mode(mode))?;
        }
        self.spec_mode.store(mode as u8, Ordering::Relaxed);
        Ok(())
    }

    /// 2026-10-10: Worker side of `EP_CMD_SPEC_MODE`: read the mode word and switch this rank.
    /// Errors on a word that is not a mode word, or on a worker started without the hybrid lever
    /// (rank 0 would otherwise route proposes this rank cannot mirror).
    pub(in crate::model) fn ep_worker_set_spec_mode(&self) -> Result<bool> {
        let word = self.ep_broadcast_u32(0)?;
        let mode = hy::decode_spec_mode(word)?;
        if !self.hybrid_on() {
            bail!(
                "EP_CMD_SPEC_MODE ({}) on a worker without the hybrid drafter: set \
                 METRALE_HYBRID_DFLASH=1 (with --dflash --speculative) on every rank",
                mode.name()
            );
        }
        self.spec_mode.store(mode as u8, Ordering::Relaxed);
        Ok(true)
    }

    /// 2026-10-10: `Model::hybrid_dflash_ready`: a hybrid serve, `seq` holds a DFlash state, and
    /// its SSM slot's verify pools hold a K=gamma verify (`ssm_reserve::HYBRID_DFLASH_SLOT`).
    pub(super) fn hybrid_dflash_ready_dispatch(&self, seq: &SequenceState) -> bool {
        self.hybrid_on()
            && seq.dflash_state.is_some()
            && self.ssm_pool.verify_draft_capacity(seq.slot_idx)
                >= self.dflash_kgamma.saturating_sub(1).max(1)
    }

    /// 2026-10-10: Install the DFlash drafter of a hybrid serve beside the MTP head already in
    /// `proposer`. The MTP head keeps every buffer `set_dflash_proposer` sized for it.
    pub fn set_hybrid_dflash_proposer(&mut self, drafter: Arc<dyn DraftProposer>) -> Result<()> {
        if self.proposer.is_none() {
            bail!(
                "METRALE_HYBRID_DFLASH=1: no MTP head was built, so there is nothing to pair the \
                 DFlash drafter with (--speculative needs the checkpoint's MTP weights)"
            );
        }
        if drafter.block_gamma().is_none() {
            bail!("METRALE_HYBRID_DFLASH=1: the second proposer is not a block (DFlash) drafter");
        }
        self.dflash_proposer = Some(drafter);
        self.spec_mode
            .store(SpecStepMode::Mtp as u8, Ordering::Relaxed);
        Ok(())
    }

    /// 2026-10-10: The SP half of the hybrid prefill capture (`try_dflash_prefill_capture_layer`,
    /// `impl_b3.rs`): the chunk rows the capture keeps, the window `PrefillCtxWindow` (the last
    /// `max_ctx_len` prompt positions; `tap_window_rows`), swapped between the ranks by the layer when its SP prefill split
    /// them. A no-op off a hybrid multi-rank serve and for a sequence without a DFlash state.
    /// Every input is the same on both ranks: slot 0 holds a DFlash state of the same window on
    /// both (`meta.rs`), and both see the same chunk.
    pub(in crate::model) fn hybrid_sp_tap_gather(
        &self,
        seq: &mut crate::traits::SequenceState,
        layer_idx: usize,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.hybrid_on() {
            return Ok(());
        }
        let Some(comm) = self.comm.as_deref() else {
            return Ok(());
        };
        let (prompt_len, run_start) = (seq.prompt_len, seq.prefill_run_start);
        let Some(max_ctx) = self.dflash_ctx_state(seq).and_then(|ps| {
            ps.as_any_mut()
                .downcast_mut::<metrale_model_arch::dflash_head::DflashProposerState>()
                .map(|d| d.max_ctx_len)
        }) else {
            return Ok(());
        };
        let win = metrale_model_arch::dflash_head::PrefillCtxWindow::new(
            prompt_len,
            chunk_start + proc_count,
            max_ctx,
            run_start,
        );
        let rows = metrale_model_arch::glm5next_layer::seq_parallel::tap_window_rows(
            chunk_start,
            proc_count,
            win.span(),
        );
        if rows.1 == 0 {
            return Ok(());
        }
        self.layers[layer_idx].prefill_tap_gather(
            comm,
            self.buffers.hidden_states(),
            chunk_start,
            proc_count,
            rows,
            stream,
        )?;
        Ok(())
    }
}
