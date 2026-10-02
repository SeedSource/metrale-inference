// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The sequence-parallel staged prefill (`METRALE_GLM_PREFILL_SEQ_PARALLEL=1`):
//! `prefill_staged_run`'s two passes with the replicated row-local work split by rows across
//! the two tensor-parallel ranks (ownership: `glm5next_layer::seq_parallel::SpPlan`, rank 0
//! the first half of every sub-chunk, rank 1 the second).
//!
//! Per pass (attention, then FFN):
//! 1. Front, owned rows only (half of each sub-chunk): (`hc_expand` on layer 0,) `hc_pre`,
//!    the norm into those rows of `norm_output` (whole-chunk sized; checked by the caller).
//! 2. Grouped send/recv: each rank sends its normed rows, receives the peer's.
//! 3. The mixer per sub-chunk (attention) or the MLP per FFN window, over ALL rows, exactly
//!    the launches `prefill_staged_run` issues, reading the same normed rows.
//! 4. Per item, a reduce-scatter in place of the all-reduce: the partial rows the peer owns
//!    go to the peer; the peer's partial of my rows arrives in my (dead) `hidden` rows, and I
//!    add my partial into it with `bf16_add_inplace`.
//! 5. Back, owned rows only, after the pass: `hc_post` from `hidden` (and on the last
//!    layer `hc_head_mean`). The last layer then swaps the final `hidden` rows, so both ranks
//!    leave with the whole output, as today.
//!
//! Why this is byte-identical to `prefill_staged_run` (its module notes give the base case):
//! - `hc_expand`, `hc_pre`, the RMSNorm, `hc_post` and `hc_head_mean` are one block (or one
//!   grid row) per token and read only that token's highway slot, `post`/`comb` and row, so
//!   running them for a subset of rows (half-sub-chunk launches) writes those rows' bytes as
//!   the full launch does (`hc_pre`'s two mix kernels are byte-identical at any row count). The
//!   highway was identical on both ranks before (every rank ran the same replicated launches
//!   on the same all-reduced values), so the normed rows a rank receives are the bytes it would
//!   have computed. Each rank's highway is valid only on its own rows from layer 0 on, and only
//!   its own rows are read until the final swap.
//! - The mixer and MLP launches, widths and order are unchanged; only the address of their
//!   (identical) normed input moves within `norm_output`, and DSA writes its partial over the
//!   sub-chunk's own rows there.
//! - The all-reduce leaves `__hadd(own, peer)` on every element (`bf16_add_inplace` with
//!   `dst` = own partial); here the owner computes `__hadd(peer, own)`, the same bits because
//!   IEEE addition commutes. Only the backend whose all-reduce is that exchange and add takes
//!   this path (`CommBackend::all_reduce_is_send_recv_add`).
//! - Moving the owned `hc_post`s after the pass changes no input: a mixer reads no highway
//!   slot, and the FFN front reads slots only after the attention back has run.
//!
//! Between layers `hidden` holds scratch in either path, but different scratch here (a
//! DFlash capture of a mid-stack layer's `hidden` would see it change); the final layer's
//! `hidden` and every highway slot a rank reads are as today.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Both ranks issue the same collectives in the same order (the plan is a function of the
//!   sub-chunk width alone); an empty send or receive is skipped on both sides.

use super::*;
use crate::glm5next_layer::seq_parallel::{SpPlan, Span, reduce_item, swap_owned};

impl Glm5NextLayer {
    /// 2026-10-01: The plan when this staged prefill of `num_tokens` rows in `subs` takes the
    /// sequence-parallel path: lever on, eager, a two-rank communicator whose all-reduce is
    /// the send/recv + add exchange, the add kernel loaded, the chunk within `norm_output`.
    /// Every input is the same on both ranks.
    pub(in crate::glm5next_layer) fn sp_plan(
        &self,
        num_tokens: usize,
        subs: &[(usize, usize)],
        ctx: &ForwardContext,
    ) -> Option<SpPlan> {
        let comm = ctx.comm?;
        let ok = prefill_seq_parallel()
            && !ctx.graph_capture
            && comm.world_size() == 2
            && comm.all_reduce_is_send_recv_add()
            && self.add_k.0 != 0
            && self.mhc.is_some()
            && num_tokens * self.hidden * 2 <= ctx.buffers.norm_output_bytes();
        if !ok {
            static SKIPPED: std::sync::Once = std::sync::Once::new();
            if prefill_seq_parallel() {
                SKIPPED.call_once(|| {
                    tracing::warn!(
                        "METRALE_GLM_PREFILL_SEQ_PARALLEL=1: NOT engaged on a prefill of \
                         {num_tokens} rows (needs eager, a two-rank send/recv all-reduce, \
                         bf16_add_inplace and a chunk within norm_output)"
                    );
                });
            }
            return None;
        }
        let plan = SpPlan::new(subs)?;
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_PREFILL_SEQ_PARALLEL=1: ENGAGED (first prefill: {num_tokens} rows, \
                 each rank owns half of every {}-row sub-chunk)",
                plan.width
            );
        });
        Some(plan)
    }

    /// 2026-10-01: `prefill_staged_run` over `subs` under `plan` (module notes).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn prefill_staged_sp(
        &self,
        hidden: DevicePtr,
        subs: &[(usize, usize)],
        plan: SpPlan,
        rows: usize,
        rows_ffn: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(comm) = ctx.comm else {
            bail!(
                "GLM layer {}: sequence-parallel prefill without a communicator",
                self.layer_idx
            );
        };
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let (gpu, add_k, rank) = (ctx.gpu, self.add_k, comm.rank());
        let (h, rb) = (self.hidden, self.hidden * 2);
        let normed = ctx.buffers.norm_output();
        let chunk = subs.last().map_or((0, 0), |&(t, k)| (0, t + k));
        let mine = plan.spans(rank, chunk);

        // 2026-10-01: Attention pass.
        let (attn, input_norm) = (&mhc.attn, self.input_norm);
        for &s in &mine {
            self.sp_front(hidden, s, attn, input_norm, self.is_first, ctx, stream)?;
        }
        let t_x = profile::start();
        swap_owned(comm, plan, normed, chunk, rb, stream)?;
        profile::end(profile::REDUCE_ATTN, t_x, gpu, stream);
        for &(t, k) in subs {
            let p = self.attn_mixer(
                normed.offset(t * rb),
                k,
                state,
                kv_cache,
                seq_len_start + t,
                block_table,
                ctx,
                stream,
                // 2026-10-01: No KDA snapshots and a prefill sub-chunk, as
                // `prefill_staged_run` passes.
                false,
                true,
            )?;
            let t_x = profile::start();
            let reduce = self.mixer_all_reduce;
            reduce_item(
                gpu,
                add_k,
                comm,
                plan,
                p,
                hidden,
                (t, k),
                h,
                reduce,
                stream,
            )?;
            profile::end(profile::REDUCE_ATTN, t_x, gpu, stream);
        }
        for &s in &mine {
            self.sp_back(hidden, s, false, ctx, stream)?;
        }

        // 2026-10-01: FFN pass, over the windows `prefill_staged_run` uses.
        let (ffn, post_norm) = (&mhc.ffn, self.post_attn_norm);
        for &s in &mine {
            self.sp_front(hidden, s, ffn, post_norm, false, ctx, stream)?;
        }
        let t_x = profile::start();
        swap_owned(comm, plan, normed, chunk, rb, stream)?;
        profile::end(profile::REDUCE_MLP, t_x, gpu, stream);
        let ffn_out = ctx.buffers.moe_output();
        let reduce = self.mlp_cfg.needs_all_reduce();
        let wins = ffn_windows(subs, rows_ffn, prefill_tail_merge(), |k| self.ffn_mergeable(k));
        for (t, k) in wins {
            self.mlp_compute(normed.offset(t * rb), ffn_out, k, rows, ctx, stream)?;
            let t_x = profile::start();
            reduce_item(
                gpu,
                add_k,
                comm,
                plan,
                ffn_out,
                hidden,
                (t, k),
                h,
                reduce,
                stream,
            )?;
            profile::end(profile::REDUCE_MLP, t_x, gpu, stream);
        }
        for &s in &mine {
            self.sp_back(hidden, s, self.is_last, ctx, stream)?;
        }
        if self.is_last {
            // 2026-10-01: Both ranks leave with every final row, as `prefill_staged_run` does.
            swap_owned(comm, plan, hidden, chunk, rb, stream)?;
            profile::step();
        }
        Ok(())
    }

    /// 2026-10-01: The front of owned rows `[t, t + k)`: `hc_expand` when `expand`,
    /// `hc_pre` of `site` into `hidden` rows, then the norm with `norm_w` into the same rows of
    /// `norm_output`, the launches `attn_half` / `ffn_half` issue before the mixer / MLP.
    #[allow(clippy::too_many_arguments)]
    fn sp_front(
        &self,
        hidden: DevicePtr,
        (t, k): Span,
        site: &Glm5NextMhcSiteWeights,
        norm_w: DevicePtr,
        expand: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let hc = mhc.hc_mult;
        let streams = ctx.buffers.hc_streams().offset(t * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(t * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(t * hc * hc * 4);
        let x = hidden.offset(t * h * 2);
        let (kt, ht, hct) = (k as u32, h as u32, hc as u32);
        let t_mhc = profile::start();
        if expand {
            let expand_k = mhc.kernels.hc_expand;
            glm_hc_expand(gpu, expand_k, x, streams, kt, ht, hct, stream)?;
        }
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            site,
            x,
            post,
            comb,
            kt,
            ht,
            hct,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        let out = ctx.buffers.norm_output().offset(t * h * 2);
        self.norm(gpu, x, norm_w, out, k, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        Ok(())
    }

    /// 2026-10-01: The back of owned rows `[t, t + k)`: `hc_post` of the reduced partial
    /// in its `hidden` rows, then (`last`) `hc_head_mean` into the same rows.
    fn sp_back(
        &self,
        hidden: DevicePtr,
        (t, k): Span,
        last: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let hc = mhc.hc_mult;
        let streams = ctx.buffers.hc_streams().offset(t * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(t * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(t * hc * hc * 4);
        let x = hidden.offset(t * h * 2);
        let (kt, ht, hct) = (k as u32, h as u32, hc as u32);
        let t_post = profile::start();
        let post_k = mhc.kernels.hc_post;
        glm_hc_post(gpu, post_k, x, streams, post, comb, streams, kt, ht, hct, stream)?;
        if last {
            let head_k = mhc.kernels.hc_head;
            hc_head_mean(gpu, head_k, streams, x, kt, ht, hct, stream)?;
        }
        profile::end(profile::MHC_POST, t_post, gpu, stream);
        Ok(())
    }
}
