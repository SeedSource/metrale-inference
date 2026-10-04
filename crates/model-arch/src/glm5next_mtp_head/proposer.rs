// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `impl DraftProposer for Glm5NextMtpHead`: state lifetime, context prefill, propose and rollback.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - `after_verify` never trims the first row of a propose.
//! - A second `free_state` on the same state is a no-op.

use super::*;

impl DraftProposer for Glm5NextMtpHead {
    /// 2026-09-25: `self.max_seq_len` is already capped at `max_dsa_context` by `new`, so the
    /// model's capture buffer is no longer than the drafter's own row space.
    fn prefill_hidden_rows(&self, max_seq_len: usize) -> usize {
        max_seq_len.min(self.max_seq_len)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        self.alloc_state_rows(gpu, self.max_seq_len)
    }

    /// 2026-10-04: Under `METRALE_GLM_MTP_SEQ_KV` the sequence's own pool is sized to its reach
    /// (`budget_tokens`, plus 16 rows for drafts written past it), capped at `max_seq_len`; a
    /// draft past its pool fails the DSA block-table lookup and never writes outside it. Off,
    /// this is `alloc_state`.
    fn alloc_state_for(
        &self,
        gpu: &dyn GpuBackend,
        budget_tokens: usize,
    ) -> Result<Box<dyn ProposerState>> {
        if !self.seq_kv {
            return self.alloc_state(gpu);
        }
        let rows = budget_tokens.saturating_add(16).min(self.max_seq_len);
        self.alloc_state_rows(gpu, rows)
    }

    /// 2026-10-04: `batch_rows_max` under `METRALE_GLM_MTP_BATCH_DRAFT`, else 1.
    fn propose_batch_max(
        &self,
        buffers: &metrale_gpu_runtime::buffers::BufferArena,
        _config: &metrale_config::ModelConfig,
    ) -> usize {
        self.batch_rows_max(buffers)
    }

    /// 2026-10-04: True under `METRALE_GLM_MTP_BATCH_DRAFT` (with its kernels): `propose_batch`
    /// is `propose` for each sequence, batched, and wants the same forward context.
    fn propose_batch_mirrors_serial(&self) -> bool {
        self.batch.is_some()
    }

    fn propose_batch_ready(
        &self,
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
    ) -> bool {
        self.batch_ready(positions, num_drafts, states)
    }

    /// 2026-10-04: `Ok(None)` unless `propose_batch_ready`; otherwise `propose` for every
    /// sequence, batched per draft step (`batch.rs`). No confidences: `out_conf` is cleared,
    /// as the per-sequence propose leaves none.
    #[allow(clippy::too_many_arguments)]
    fn propose_batch(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        if !self.batch_ready(positions, num_drafts, states) {
            return Ok(None);
        }
        if let Some(c) = out_conf {
            c.clear();
        }
        self.propose_batch_impl(
            last_tokens,
            target_hiddens,
            positions,
            num_drafts,
            states,
            ctx,
            stream,
        )
        .map(Some)
    }

    /// 2026-09-25: Release everything `alloc_state` allocated: the DSA indexer cache and the
    /// scratch buffers. `DevicePtr` has no `Drop`, so without this override they would leak.
    /// A second call is a no-op (`released`).
    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid GLM MTP proposer state"))?;
        if st.released {
            return Ok(());
        }
        st.released = true;
        st.dsa.free(gpu)?;
        for p in [st.concat, st.x, st.logits, st.arg, st.head_xchg] {
            gpu.free(p)?;
        }
        // 2026-10-04: The sequence's own latent pool (`METRALE_GLM_MTP_SEQ_KV`): one layer, its
        // K pool, and its V pool unless V aliases K (`drafter_kv_config`).
        if let Some(own) = st.own_kv.take() {
            let (k, v) = (own.k_pool_ptr(0), own.v_pool_ptr(0));
            gpu.free(k)?;
            if v.0 != k.0 {
                gpu.free(v)?;
            }
        }
        // 2026-09-25: The drafter's private pool is claimed whole by `alloc_state`
        // (`(0..blocks).collect()`), not drawn from an allocator, so there is
        // nothing to hand back; clearing it stops a freed state from looking live.
        st.block_table.clear();
        st.seq_len = 0;
        Ok(())
    }

    /// 2026-09-25: The block's routed experts are split across EP ranks and its DSA `o_proj`
    /// is row-parallel, so its forward needs the communicator. On unless
    /// `METRALE_NO_MTP_EP_PROPOSE=1` (`mtp_ep_propose_enabled`).
    fn needs_comm(&self) -> bool {
        metrale_model_layers::speculative::mtp_ep_propose_enabled()
    }

    /// 2026-09-25: True: a context row normalises into `ctx.buffers` (`drafter_write_kv_row`).
    fn prefill_uses_shared_buffers(&self) -> bool {
        true
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .map_or(0, |st| st.seq_len)
    }

    /// 2026-09-25: Dense row space: the newest row's slot is its pair key.
    fn last_pair_key(&self, state: &mut dyn ProposerState) -> Option<usize> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .and_then(|st| st.seq_len.checked_sub(1))
    }

    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let t0 = std::time::Instant::now();
        let rows = self.rows_impl(prompt_tokens, hiddens, 0, state, ctx, stream)?;
        // 2026-09-25: `rows_impl` returns 0 without work once the drafter has rows, so only a
        // call that wrote rows logs.
        if rows > 0 {
            tracing::info!(
                "GLM MTP drafter prefill: {rows} rows ({} prompt tokens) in {:.1} ms",
                prompt_tokens.len(),
                t0.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(rows)
    }

    /// 2026-09-25: `pos_base` is unused: this drafter's RoPE position is its KV slot (see
    /// `rows_impl`). A feed that does not start exactly at `drafter_rows()` writes nothing.
    fn catchup_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        _pos_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.rows_impl(tokens, hiddens, row_base, state, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        _grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not a GLM MTP proposer state"))?;
        // 2026-09-25: A drafter ahead of the target's `position` is rewound to it, so its
        // indexer does not select over rows past the target's context.
        if st.seq_len > position {
            st.dsa.rewind_to(position)?;
            st.seq_len = position;
        }
        if metrale_model_layers::speculative::mtp_refeed_debug() {
            let fp = metrale_model_layers::speculative::hidden_fingerprint(
                ctx.gpu,
                target_hidden,
                self.hidden,
            );
            tracing::info!(
                "GLM_MTP_DBG propose position={position} drafter_rows={} tok={last_token} \
                 fp_target={fp:016x}",
                st.seq_len,
            );
        }
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut token = last_token;
        let mut hidden = target_hidden;
        for i in 0..num_drafts {
            let d = self.forward_one(token, hidden, position + i, st, ctx, stream)?;
            drafts.push(d);
            token = d;
            // 2026-09-25: Draft 1 consumes the target's verified hidden; every later draft
            // consumes the drafter's own block output.
            hidden = st.x;
        }
        st.last_drafted = drafts.len();
        Ok(drafts)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not a GLM MTP proposer state"))?;
        // 2026-09-25: Rejected rows are unreachable: the indexer reads `[0, len)` and the next
        // propose writes from `seq_len`, so rolling the counters back is the whole rollback.
        //
        // The first row of a propose is always kept. It pairs the last committed token with the
        // target's own hidden, both known at propose time, so a rejected draft does not make it
        // wrong. Only later rows depend on a draft having been accepted, and trimming the first
        // would drop a real row from the dense row space and shift every later RoPE position.
        // At `num_drafts = 1` nothing is trimmed.
        let keep = st.last_drafted.min(num_accepted + 1);
        let trim = st.last_drafted - keep;
        if trim > 0 {
            st.seq_len = st.seq_len.saturating_sub(trim);
            st.dsa.rewind_to(st.seq_len)?;
        }
        Ok(())
    }
}

impl Glm5NextMtpHead {
    /// 2026-10-04: A drafter state whose block table covers `rows` rows. Without
    /// `METRALE_GLM_MTP_SEQ_KV`, `rows` is `max_seq_len` and the table is the whole of the
    /// head's shared pool, as before the lever; with it, the table indexes a pool of this
    /// state's own of the same block count.
    fn alloc_state_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
    ) -> Result<Box<dyn ProposerState>> {
        let dsa = match &self.module.layer.mixer {
            // 2026-10-03: The drafter writes rows past the target's KV grid, so its cache maps
            // `PROPOSER_LOOKAHEAD_ROWS` ahead when lazily mapped (`METRALE_DSA_INDEXER_LAZY`).
            crate::glm5next_layer::Glm5NextMixer::Dsa(l) => Glm5NextDsaState::alloc_with_lookahead(
                gpu,
                &l.cfg,
                crate::glm5next_dsa::lazy::PROPOSER_LOOKAHEAD_ROWS,
            )?,
            _ => bail!("GLM MTP block is not a DSA layer"),
        };
        let h = self.hidden;
        // 2026-09-25: Claims every block of the private pool, which `new` sizes for one
        // sequence of `max_seq_len` rows.
        let blocks = (rows / 16 + 2) as u32;
        let own_kv = if self.seq_kv {
            Some(PagedKvCache::new(
                init::drafter_kv_config(self.kv_lora_rank),
                blocks as usize,
                gpu,
            )?)
        } else {
            None
        };
        Ok(Box::new(Glm5NextMtpProposerState {
            dsa,
            own_kv,
            seq_len: 0,
            block_table: (0..blocks).collect(),
            last_drafted: 0,
            concat: gpu.alloc(2 * h * 2)?,
            x: gpu.alloc(h * 2)?,
            logits: gpu.alloc(self.vocab * 2)?,
            arg: gpu.alloc(4)?,
            head_xchg: gpu.alloc(16)?,
            released: false,
        }))
    }
}
