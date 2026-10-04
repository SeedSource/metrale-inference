// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl ModelDraft for TransformerModel`, mostly delegating to `<method>_dispatch`
//! helpers in the sibling modules.
//!
//! The DFlash context-append methods (`dflash_accept_append`,
//! `dflash_eagle_accept_append`, `dflash_eagle_kgamma_append`, `commit_ctx`,
//! `dflash_serial_ctx_append`) are implemented here.
//!
//! Owner: model-engine.
//! Invariants: the ones in `trait_impl/mod.rs`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use crate::model::types::TransformerModel;
use crate::traits::{ModelDraft, SequenceState};

impl ModelDraft for TransformerModel {
    fn mtp_slot_draft_capacity(&self, slot_idx: usize) -> usize {
        self.ssm_pool.verify_draft_capacity(slot_idx)
    }

    fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        params: &metrale_sampling::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        self.gdn_carry_flush_pending()?;
        self.generate_speculative_dispatch(prompt_tokens, params, num_drafts)
    }

    fn has_proposer(&self) -> bool {
        self.has_proposer_dispatch()
    }

    fn dflash_gamma(&self) -> Option<usize> {
        self.proposer.as_ref().and_then(|p| p.block_gamma())
    }

    fn has_self_speculative(&self) -> bool {
        self.has_self_speculative_dispatch()
    }

    fn decode_draft(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.gdn_carry_flush_pending()?;
        self.decode_draft_dispatch(token, seq, stream)
    }

    fn stash_verify_hidden_rows(&self, rows: &[usize], _stream: u64) -> Result<()> {
        self.stash_verify_hidden_rows_dispatch(rows, _stream)
    }

    fn save_hidden_for_mtp_from_stash(&self, idx: usize, _stream: u64) -> Result<()> {
        self.save_hidden_for_mtp_from_stash_dispatch(idx, _stream)
    }

    fn stash_verify_catchup_rows(&self, slot_rows: &[(usize, usize)]) -> Result<()> {
        self.stash_verify_catchup_rows_dispatch(slot_rows)
    }

    fn run_mtp_catchup_batched(
        &self,
        tokens: &[Vec<u32>],
        first_slot: &[usize],
        first_pos: &[usize],
        seqs: &mut [&mut SequenceState],
    ) -> Result<usize> {
        self.run_mtp_catchup_batched_dispatch(tokens, first_slot, first_pos, seqs)
    }

    fn run_mtp_propose_batched(
        &self,
        tokens: &[u32],
        positions: &[usize],
        stash_idx: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        _stream: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        self.run_mtp_propose_batched_dispatch(
            tokens, positions, stash_idx, num_drafts, seqs, out_conf,
        )
    }

    fn mtp_propose_batch_max(&self) -> usize {
        match &self.proposer {
            Some(p) => p.propose_batch_max(&self.buffers, &self.config),
            None => 1,
        }
    }

    fn mtp_propose_batch_mirrors_serial(&self) -> bool {
        self.proposer
            .as_ref()
            .is_some_and(|p| p.propose_batch_mirrors_serial())
    }

    fn save_hidden_for_catchup(&self, token_idx: usize, pos: usize) -> Result<()> {
        self.save_hidden_for_catchup_dispatch(token_idx, pos)
    }

    fn save_hidden_for_mtp(&self, token_idx: usize, _stream: u64) -> Result<()> {
        self.save_hidden_for_mtp_dispatch(token_idx, _stream)
    }

    fn save_dflash_hidden_for_propose(&self, token_idx: usize, _stream: u64) -> Result<()> {
        self.save_dflash_hidden_dispatch(token_idx, _stream)
    }

    fn dflash_accept_append(&self, seq: &mut SequenceState) -> Result<()> {
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let prop = match seq.proposer_state.as_mut() {
            Some(p) => p.as_mut(),
            None => return Ok(()),
        };
        let d = prop
            .as_any_mut()
            .downcast_mut::<metrale_model_arch::dflash_head::DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not DFlash proposer state"))?;
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let save_1 = base.offset(ctx_slot_bytes);
        let dst = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(save_1, dst, ctx_slot_bytes, self.gpu.default_stream())?;
        d.ctx_positions.push((seq.seq_len as i32).saturating_sub(1));
        d.ctx_len += 1;
        Ok(())
    }

    fn dflash_eagle_accept_append(&self, seq: &mut SequenceState) -> Result<()> {
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let prop = match seq.proposer_state.as_mut() {
            Some(p) => p.as_mut(),
            None => return Ok(()),
        };
        let d = prop
            .as_any_mut()
            .downcast_mut::<metrale_model_arch::dflash_head::DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not DFlash proposer state"))?;
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let stream = self.gpu.default_stream();
        let pos_row0 = (seq.seq_len as i32).saturating_sub(2);
        let pos_row1 = (seq.seq_len as i32).saturating_sub(1);
        let save_0 = base;
        let dst_0 = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(save_0, dst_0, ctx_slot_bytes, stream)?;
        d.ctx_positions.push(pos_row0);
        d.ctx_len += 1;
        let save_1 = base.offset(ctx_slot_bytes);
        let dst_1 = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(save_1, dst_1, ctx_slot_bytes, stream)?;
        d.ctx_positions.push(pos_row1);
        d.ctx_len += 1;
        d.skip_next_decode_append = true;
        Ok(())
    }

    fn dflash_eagle_kgamma_append(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        base_pos: usize,
    ) -> Result<()> {
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let prop = match seq.proposer_state.as_mut() {
            Some(p) => p.as_mut(),
            None => return Ok(()),
        };
        let d = prop
            .as_any_mut()
            .downcast_mut::<metrale_model_arch::dflash_head::DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not DFlash proposer state"))?;
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let stream = self.gpu.default_stream();
        for t in 0..=num_accepted {
            let row = base.offset(t * ctx_slot_bytes);
            let dst = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
            self.gpu.copy_d2d_async(row, dst, ctx_slot_bytes, stream)?;
            let pos = (base_pos + t) as i32;
            d.ctx_positions.push(pos);
            d.ctx_len += 1;
        }
        d.skip_next_decode_append = true;
        Ok(())
    }

    fn dflash_capture_band(&self) -> usize {
        self.dflash_kgamma
    }

    fn commit_ctx(
        &self,
        seq: &mut SequenceState,
        num_committed: usize,
        base_pos: usize,
        scratch_row: usize,
    ) -> Result<()> {
        if num_committed == 0 {
            return Ok(());
        }
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let prop = match seq.proposer_state.as_mut() {
            Some(p) => p.as_mut(),
            None => return Ok(()),
        };
        let d = match prop
            .as_any_mut()
            .downcast_mut::<metrale_model_arch::dflash_head::DflashProposerState>()
        {
            Some(d) => d,
            None => return Ok(()),
        };
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let stream = self.gpu.default_stream();

        // 2026-09-25: Capture writes stop at `dflash_hidden_save_rows`
        // (`try_dflash_capture_all_at` caps its row count), so a row past it
        // was never captured. Appending it would put a stale hidden into the
        // context, so skip the commit with a warning instead.
        if scratch_row + num_committed > self.dflash_hidden_save_rows {
            tracing::warn!(target: "metrale_model_engine::model::trait_impl", "commit_ctx: scratch rows {}..{} exceed capture capacity {} — skipping (ctx hole)",
                scratch_row,
                scratch_row + num_committed,
                self.dflash_hidden_save_rows,
            );
            return Ok(());
        }

        // 2026-09-25: If the incoming rows would overflow the accumulator,
        // slide first: keep the newest `keep` rows and drop the oldest.
        // `ctx_committed = 0` makes the next propose recompute K/V for every
        // kept row, and `ctx_positions` keeps absolute positions. `drop_n >=
        // keep` holds only when `ctx_len >= 2 * keep`; when `ctx_len` is below
        // that, the source and destination ranges of the single copy overlap.
        if d.ctx_len + num_committed > d.max_ctx_len {
            let keep = (d.max_ctx_len / 2).min(d.max_ctx_len.saturating_sub(num_committed));
            let drop_n = d.ctx_len.saturating_sub(keep);
            if drop_n > 0 {
                let src = d.ctx_hidden_acc.offset(drop_n * ctx_slot_bytes);
                let dst0 = d.ctx_hidden_acc.offset(0);
                self.gpu
                    .copy_d2d_async(src, dst0, keep * ctx_slot_bytes, stream)?;
                d.ctx_positions.drain(..drop_n);
                d.ctx_len = keep;
                d.ctx_committed = 0;
                tracing::info!(target: "metrale_model_engine::model::trait_impl", "DFlash UNIFIED_CTX watermark: slid ctx window (dropped {} oldest, keep {})",
                    drop_n,
                    keep,
                );
            }
        }

        // 2026-09-25: Append at the tail. The destination row is indexed by
        // `ctx_len`, while `ctx_positions` records the absolute position
        // `base_pos + t`. The two differ once a slide has run.
        debug_assert_eq!(d.ctx_positions.len(), d.ctx_len);
        for t in 0..num_committed {
            let row = base.offset((scratch_row + t) * ctx_slot_bytes);
            let dst = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
            self.gpu.copy_d2d_async(row, dst, ctx_slot_bytes, stream)?;
            d.ctx_positions.push((base_pos + t) as i32);
            d.ctx_len += 1;
        }
        // 2026-09-25: The latest capture is in the context now, so the next
        // propose skips its own decode-append of it.
        d.skip_next_decode_append = true;

        tracing::debug!(target: "metrale_model_engine::model::trait_impl", "CTX_COMMIT slot={} rows={} base_pos={} ctx_len_after={}",
            seq.slot_idx,
            num_committed,
            base_pos,
            d.ctx_len,
        );
        if self.stats.once("log:dflash_unified_ctx") {
            tracing::info!(target: "metrale_model_engine::model::trait_impl", "DFlash UNIFIED_CTX ACTIVE: first commit_ctx rows={} base_pos={} ctx_len={}",
                num_committed,
                base_pos,
                d.ctx_len,
            );
        }
        Ok(())
    }

    fn dflash_serial_ctx_append(&self, seq: &mut SequenceState) -> Result<()> {
        // 2026-09-25: Append the hidden captured for the token just decoded
        // serially. The decode layer loop (`try_dflash_capture(i, 0, ..)` in
        // decode_a3.rs) writes it to row 0 of `dflash_hidden_save`, one slot per
        // capture layer, which is the layout of one accumulator row.
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let prop = match seq.proposer_state.as_mut() {
            Some(p) => p.as_mut(),
            None => return Ok(()),
        };
        // 2026-09-25: A non-DFlash proposer state returns `Ok` here, whereas
        // `dflash_eagle_accept_append` returns an error for one.
        let d = match prop
            .as_any_mut()
            .downcast_mut::<metrale_model_arch::dflash_head::DflashProposerState>()
        {
            Some(d) => d,
            None => return Ok(()),
        };
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let stream = self.gpu.default_stream();
        // 2026-09-25: When the accumulator is full, slide the window: keep the
        // newest `max_ctx_len / 2` rows and drop the oldest. Here `ctx_len >=
        // max_ctx_len`, so `drop_n >= keep` and the single copy's source and
        // destination ranges do not overlap. `ctx_committed = 0` makes the next
        // propose recompute K/V for every kept row, and `ctx_positions` keeps
        // absolute positions.
        if d.ctx_len >= d.max_ctx_len {
            let keep = d.max_ctx_len / 2;
            let drop_n = d.ctx_len - keep;
            let src = d.ctx_hidden_acc.offset(drop_n * ctx_slot_bytes);
            let dst0 = d.ctx_hidden_acc.offset(0);
            self.gpu
                .copy_d2d_async(src, dst0, keep * ctx_slot_bytes, stream)?;
            d.ctx_positions.drain(..drop_n);
            d.ctx_len = keep;
            d.ctx_committed = 0;
            tracing::info!(target: "metrale_model_engine::model::trait_impl", "DFlash SERIAL_APPEND watermark: slid ctx window (dropped {} oldest, keep {})",
                drop_n,
                keep,
            );
        }
        let dst = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
        self.gpu.copy_d2d_async(base, dst, ctx_slot_bytes, stream)?;
        if self.stats.once("log:dflash_serial_append") {
            tracing::info!(target: "metrale_model_engine::model::trait_impl", "DFlash SERIAL_APPEND ACTIVE: first serial ctx append at ctx_len={} pos={}",
                d.ctx_len,
                seq.seq_len.saturating_sub(1),
            );
        }
        // 2026-09-25: Decode has already advanced `seq_len` past this token, so
        // its position is `seq_len - 1`.
        debug_assert_eq!(d.ctx_positions.len(), d.ctx_len);
        d.ctx_positions.push(seq.seq_len.saturating_sub(1) as i32);
        d.ctx_len += 1;
        // 2026-09-25: The latest capture is in the context now, so the next
        // propose skips its own decode-append of it.
        d.skip_next_decode_append = true;
        Ok(())
    }

    fn run_mtp_propose(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        self.run_mtp_propose_dispatch(token, position, seq, _stream)
    }

    fn run_mtp_propose_multi(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        _stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.run_mtp_propose_multi_dispatch(
            token,
            position,
            num_drafts,
            seq,
            _stream,
            grammar_bitmask,
        )
    }

    fn read_deferred_draft_token(&self) -> Result<u32> {
        self.read_deferred_draft_token_dispatch()
    }

    fn trim_proposer_state(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        self.trim_proposer_state_dispatch(seq, num_accepted, _stream)
    }
}
