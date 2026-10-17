// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `DraftProposer` for `BlockDiffusionDraftHead`, and the allocation of
//! its per-sequence state.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - `propose_batch` declines with `Ok(None)`, never an error, so the caller
//!   falls back to the per-sequence loop.

use super::*;

impl DraftProposer for BlockDiffusionDraftHead {
    fn block_gamma(&self) -> Option<usize> {
        Some(self.gamma)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        self.alloc_state_windowed(gpu, usize::MAX)
    }

    fn alloc_state_for(
        &self,
        gpu: &dyn GpuBackend,
        budget_tokens: usize,
    ) -> Result<Box<dyn ProposerState>> {
        self.alloc_state_windowed(gpu, budget_tokens)
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: metrale_gpu_runtime::gpu::DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &metrale_model_layers::layer::ForwardContext,
        stream: u64,
        draft_embed_target: Option<metrale_gpu_runtime::gpu::DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<metrale_gpu_runtime::gpu::DevicePtr>,
    ) -> Result<Vec<u32>> {
        if self.tp.is_some() {
            return self.propose_tp(
                last_token,
                position,
                num_drafts,
                state,
                ctx,
                stream,
                target_hidden_stack,
            );
        }
        self.propose_drafts(
            last_token,
            target_hidden,
            position,
            num_drafts,
            state,
            ctx,
            stream,
            draft_embed_target,
            grammar_bitmask,
            target_hidden_stack,
            None,
        )
    }

    /// 2026-09-25: 1 without the DFlash2 path or when
    /// `METRALE_DFLASH_BATCH_PROPOSE` is below 2; otherwise that width, capped
    /// at the scratch bands (`max_batch`).
    fn propose_batch_max(
        &self,
        _buffers: &metrale_gpu_runtime::buffers::BufferArena,
        _config: &metrale_config::ModelConfig,
    ) -> usize {
        // 2026-10-09: The two-rank drafter proposes one sequence at a time.
        if !self.dflash2_active() || self.tp.is_some() {
            return 1;
        }
        let want = self.levers.batch_propose_width;
        if want < 2 {
            return 1;
        }
        want.min(self.max_batch.max(1))
    }

    /// 2026-09-25: One drafter forward over `n * block_g()` rows instead of n
    /// forwards. The per-sequence prep runs through `propose_drafts` with a
    /// `collect_prep` sink; the batched ctx precompute and the forward then run
    /// once for all n sequences.
    ///
    /// Returns `Ok(None)` to decline: fewer than 2 or more than `max_batch`
    /// sequences, no DFlash2 path, a failed prep, precompute or forward, or a
    /// forward that returned fewer than `n * block_g()` rows.
    fn propose_batch(
        &self,
        last_tokens: &[u32],
        _target_hiddens: &[metrale_gpu_runtime::gpu::DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        ctx: &metrale_model_layers::layer::ForwardContext,
        stream: u64,
        _out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let n = last_tokens.len();
        if n < 2
            || n > self.max_batch
            || positions.len() != n
            || states.len() != n
            || !self.dflash2_active()
            || self.tp.is_some()
        {
            return Ok(None);
        }

        // 2026-09-25: Per-sequence prep, collecting each sequence's paged
        // descriptor. One failed prep declines the whole batch.
        self.set_block_g(num_drafts);
        let mut prep: Vec<(metrale_gpu_runtime::gpu::DevicePtr, u32)> = Vec::with_capacity(n);
        for (i, st) in states.iter_mut().enumerate() {
            let before = prep.len();
            match self.propose_drafts(
                last_tokens[i],
                metrale_gpu_runtime::gpu::DevicePtr::NULL,
                positions[i],
                num_drafts,
                *st,
                ctx,
                stream,
                None,
                None,
                None,
                Some(&mut prep),
            ) {
                Ok(_) if prep.len() == before + 1 => {}
                Ok(_) => return Ok(None),
                Err(e) => {
                    tracing::warn!("DFlash batched propose prep (seq {i}): {e:#} — per-seq path");
                    return Ok(None);
                }
            }
        }

        // 2026-09-25: One ctx precompute over every sequence's uncommitted
        // rows, which the prep deferred. A failure leaves `ctx_committed`
        // unchanged for every tail not yet written, so the per-sequence path
        // recomputes those.
        if batched_precompute_enabled()
            && let Err(e) = self.precompute_ctx_kv_batched(states, ctx, stream)
        {
            tracing::warn!("DFlash batched ctx precompute: {e:#} — per-seq path");
            return Ok(None);
        }

        let batch = DflashBatch {
            last_tokens,
            positions,
            block_tables: prep.iter().map(|p| p.0).collect(),
            ctx_counts: prep.iter().map(|p| p.1).collect(),
        };
        let all = match self.forward_block(
            last_tokens[0],
            positions[0],
            ctx,
            stream,
            None,
            Some(prep[0]),
            Some(&batch),
        ) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("DFlash batched forward_block: {e:#} — falling back to per-seq");
                return Ok(None);
            }
        };
        let g = self.block_g();
        if all.len() < n * g {
            tracing::warn!(
                "DFlash batched forward returned {} rows, expected {} — per-seq path",
                all.len(),
                n * g
            );
            return Ok(None);
        }

        // 2026-09-25: Split the bands. As in `propose_drafts`, a drafter with a
        // mask token drops each band's row 0 (the anchor).
        let cap = self.levers.draft_cap.unwrap_or(g);
        let mut out: Vec<Vec<u32>> = Vec::with_capacity(n);
        for (i, st) in states.iter_mut().enumerate() {
            let band = &all[i * g..(i + 1) * g];
            let drafts: Vec<u32> = if self.mask_token_id != 0 {
                band.iter().skip(1).copied().take(cap).collect()
            } else {
                band.iter().copied().take(cap).collect()
            };
            if let Some(d) = st.as_any_mut().downcast_mut::<DflashProposerState>() {
                d.last_num_drafted = drafts.len();
            }
            out.push(drafts);
        }
        Ok(Some(out))
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
        // 2026-09-25: `ctx_committed` counts the ctx rows whose K/V is in the
        // paged cache. The paths that shrink `ctx_len`, the window slides in
        // model-engine's `commit_ctx` and `dflash_serial_ctx_append`, reset it
        // to 0; `propose_drafts` also clamps it to `ctx_len`.
        let _ = num_accepted;
        dstate.last_num_drafted = 0;
        Ok(())
    }

    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        // 2026-09-25: Return the drafter's paged blocks to the shared pool.
        let dstate = match state.as_any_mut().downcast_mut::<DflashProposerState>() {
            Some(s) => s,
            None => return Ok(()),
        };
        if !dstate.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&dstate.block_table);
            dstate.block_table.clear();
        }
        // 2026-09-25: Free the ctx accumulator (`max_ctx_len` rows of
        // `ctx_slot_bytes`); `DevicePtr` is `Copy` and frees nothing on drop.
        // The null check and reset make a second `free_state` skip it.
        if dstate.ctx_hidden_acc.0 != 0 {
            gpu.free(dstate.ctx_hidden_acc)?;
            dstate.ctx_hidden_acc = DevicePtr(0);
        }
        if let Some(bt) = dstate.block_table_dev.take() {
            gpu.free(bt)?;
        }
        // 2026-09-25: Counters and flags back to their `alloc_state` values.
        dstate.max_ctx_count_drafter = 0;
        dstate.ctx_count_drafter = 0;
        dstate.ctx_committed = 0;
        dstate.ctx_positions.clear();
        dstate.seq_len = 0;
        dstate.ctx_len = 0;
        dstate.prefill_done = false;
        dstate.last_num_drafted = 0;
        dstate.last_num_accepted = 0;
        dstate.skip_next_decode_append = false;
        dstate.tp_synced_pos = None;
        Ok(())
    }

    /// 2026-10-09: True with the two-rank drafter (`tp.rs`): the engine then sends the worker
    /// `EP_CMD_MTP_PROPOSE` before every propose and hands the forward the communicator.
    fn needs_comm(&self) -> bool {
        self.tp.is_some()
    }
}

impl BlockDiffusionDraftHead {
    /// 2026-09-25: Allocate proposer state with a ctx accumulator of `window`
    /// rows: the smallest of `budget_tokens + gamma + 1`,
    /// `METRALE_DFLASH_CTX_CAP` (unless `0`) and the head's `max_seq_len`.
    fn alloc_state_windowed(
        &self,
        gpu: &dyn GpuBackend,
        budget_tokens: usize,
    ) -> Result<Box<dyn ProposerState>> {
        let bf16 = 2usize;
        let ctx_slot_bytes = self.target_layer_ids.len() * self.target_hidden_size * bf16;
        // 2026-09-25: The accumulator is per sequence, `window *
        // ctx_slot_bytes` bytes. `METRALE_DFLASH_CTX_CAP` (default 16384, `0`
        // uncapped) bounds it below the head's `max_seq_len`. When it fills,
        // model-engine's `commit_ctx` and `dflash_serial_ctx_append` slide it,
        // keeping the newest rows and their positions.
        let cap = dflash_ctx_cap();
        let ceiling = if cap == 0 {
            self.max_seq_len
        } else {
            self.max_seq_len.min(cap)
        };
        // 2026-09-25: `budget_tokens` is the prompt plus `max_tokens` when the
        // caller knows it, `usize::MAX` otherwise.
        let window = ceiling.min(budget_tokens.saturating_add(self.gamma + 1));
        if ceiling < self.max_seq_len {
            static LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(
                    "DFlash ctx window capped to {} of --max-seq-len {} ({} MB/seq instead of \
                     {} MB): the accumulator is PER SEQUENCE, so the uncapped size is what \
                     OOMs a high-concurrency long-context serve. Override with \
                     METRALE_DFLASH_CTX_CAP=<tokens> (0 = uncapped).",
                    ceiling,
                    self.max_seq_len,
                    ceiling * ctx_slot_bytes / (1024 * 1024),
                    self.max_seq_len * ctx_slot_bytes / (1024 * 1024),
                );
            }
        }
        let total = window * ctx_slot_bytes;
        let ctx_hidden_acc = gpu.alloc(total)?;
        // 2026-09-25: Zeroed, so no earlier allocation's data is visible.
        gpu.memset(ctx_hidden_acc, 0, total)?;
        Ok(Box::new(DflashProposerState {
            block_table: Vec::with_capacity(64),
            seq_len: 0,
            last_num_drafted: 0,
            prefill_done: false,
            ctx_hidden_acc,
            ctx_len: 0,
            last_num_accepted: 0,
            skip_next_decode_append: false,
            max_ctx_len: window,
            ctx_slot_bytes,
            // 2026-09-25: Allocated by the first propose with the paged cache on.
            block_table_dev: None,
            ctx_count_drafter: 0,
            max_ctx_count_drafter: 0,
            ctx_committed: 0,
            ctx_positions: Vec::new(),
            tp_synced_pos: None,
        }))
    }
}
