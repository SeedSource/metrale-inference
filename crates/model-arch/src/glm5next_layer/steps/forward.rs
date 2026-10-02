// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The mHC layer bodies: `forward_one` (one token) and `forward_k` (K rows of one
//! sequence in one call).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `forward_one` uses highway slot `slot`; `forward_k` uses slots `slot_base..slot_base + k`.

use super::*;

impl Glm5NextLayer {
    /// 2026-09-25: One token through the whole layer, using highway slot `slot`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_one(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        slot: usize,
        st: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_offloaded: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let Some(mhc) = self.mhc.as_ref() else {
            return self.forward_one_plain(
                hidden,
                residual,
                st,
                kv_cache,
                seq_len,
                block_table,
                disk_block_ids,
                disk_offloaded,
                ctx,
                stream,
            );
        };
        let hc = mhc.hc_mult;
        let streams = ctx.buffers.hc_streams().offset(slot * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();

        let t_mhc = profile::start();
        if self.is_first {
            glm_hc_expand(
                gpu,
                mhc.kernels.hc_expand,
                hidden,
                streams,
                1,
                h as u32,
                hc as u32,
                stream,
            )?;
        }

        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.attn,
            hidden,
            post,
            comb,
            1,
            h as u32,
            hc as u32,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.input_norm, normed, 1, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        let attn_out = self.mixer_forward(
            normed,
            residual,
            st,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_offloaded,
            ctx,
            stream,
        )?;
        // 2026-10-01: Decode L2 prefetch of this layer's FFN head (FFN-site `hc_fn`, router or
        // dense `gate_proj`) ahead of the attention all-reduce; no-op unless
        // `METRALE_GLM_DECODE_L2_PREFETCH=1`.
        self.l2_prefetch(&self.prefetch.ffn_head, 1, ctx, stream)?;
        if self.mixer_all_reduce {
            self.reduce_probe(profile::REDUCE_ATTN_BAR, "attn", ctx, stream);
            let t = profile::start_hot();
            self.reduce_partial(attn_out, 1, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_ATTN_ENQ, t);
            let t = profile::start_hot();
            profile::end(profile::REDUCE_ATTN, t, ctx.gpu, stream);
        }
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            attn_out,
            streams,
            post,
            comb,
            streams,
            1,
            h as u32,
            hc as u32,
            stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);

        let t_mhc = profile::start();
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.ffn,
            hidden,
            post,
            comb,
            1,
            h as u32,
            hc as u32,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.post_attn_norm, normed, 1, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        self.mlp_forward(normed, ffn_out, 1, 1, ctx, stream)?;
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            ffn_out,
            streams,
            post,
            comb,
            streams,
            1,
            h as u32,
            hc as u32,
            stream,
        )?;

        if self.is_last {
            hc_head_mean(
                gpu,
                mhc.kernels.hc_head,
                streams,
                hidden,
                1,
                h as u32,
                hc as u32,
                stream,
            )?;
        }
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        if self.is_last {
            profile::step();
        }
        Ok(())
    }

    /// 2026-09-25: `k` rows of one sequence through the layer in one call. Row `t` uses highway
    /// slot `slot_base + t` and position `seq_len + t`. Each mHC and norm launch covers all `k`
    /// rows and computes every row on its own, so a row's result equals a single-row launch.
    /// KDA's `decode_k` runs its recurrence row by row. Errors on a layer without a
    /// hyper-connection.
    ///
    /// With `take_snapshots`, a KDA layer writes its state after row `t` to intermediate `t`
    /// for `t < k - 1`.
    ///
    /// 2026-09-29: The body is `attn_half` then `ffn_half` over the same `k` rows, the launch
    /// sequence it has always issued; the staged prefill (`prefill_staged_run`) calls the two
    /// halves at different widths.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_k(
        &self,
        hidden: DevicePtr,
        k: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        take_snapshots: bool,
        slot_base: usize,
        is_prefill: bool,
    ) -> Result<()> {
        self.attn_half(
            hidden,
            k,
            state,
            kv_cache,
            seq_len,
            block_table,
            ctx,
            stream,
            take_snapshots,
            slot_base,
            is_prefill,
            // 2026-10-01: The DSA core over all `k` rows, as before the full-width lever.
            k,
        )?;
        // 2026-09-29: One slice as wide as the call: the MLP's dense GEMMs run over all `k` rows
        // at once, as before the split.
        self.ffn_half(hidden, k, ctx, stream, slot_base, k)
    }

    /// 2026-09-29: The attention half of `forward_k` over rows `slot_base..slot_base + k`: the
    /// optional `hc_expand` (layer 0), the attention site's `hc_pre`, the input norm, the mixer,
    /// its all-reduce and `hc_post`. It writes highway slots `slot_base..slot_base + k`, their
    /// `post`/`comb`, rows `0..k` of `hidden`, the mixer state and KV, and nothing the FFN half
    /// of any other row reads.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn attn_half(
        &self,
        hidden: DevicePtr,
        k: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        take_snapshots: bool,
        slot_base: usize,
        is_prefill: bool,
        // 2026-10-01: The DSA selection and attend width (`attn_half_inner`).
        core_rows: usize,
    ) -> Result<()> {
        self.attn_half_inner(
            hidden,
            k,
            state,
            kv_cache,
            seq_len,
            block_table,
            ctx,
            stream,
            take_snapshots,
            slot_base,
            is_prefill,
            core_rows,
            None,
        )
    }

    /// 2026-10-01: `attn_half`, or with `defer = Some(slot)` and a mixer all-reduce, its front
    /// only: everything up to the mixer, then the mixer output copied into rows `0..k` of
    /// `hidden` (dead after the input norm read them) and its all-reduce issued there as a
    /// deferred all-reduce under `slot` (`reduce_deferred`); `attn_finish` later joins it and runs
    /// `hc_post`. Without a mixer all-reduce, `defer` is ignored and `attn_finish` does nothing.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn attn_half_inner(
        &self,
        hidden: DevicePtr,
        k: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        take_snapshots: bool,
        slot_base: usize,
        // 2026-09-25: True only for a prefill sub-chunk (`Glm5NextLayer::prefill`). It selects
        // the DSA batched selector (`batch_select_enabled`) and, with
        // `METRALE_GLM_KDA_CHUNK_PREFILL=1`, the KDA chunked scan, both prefill-only.
        is_prefill: bool,
        // 2026-10-01: Rows per DSA selection and gather-attend. `k` (every caller but the
        // full-width staged pass) runs `decode_k` over all rows as before; fewer, on a prefill,
        // runs `decode_k_wide`: the projections over all `k` rows, the selection and attend per
        // `core_rows` rows (`METRALE_GLM_PREFILL_FULLWIDTH_GEMM`). KDA ignores it.
        core_rows: usize,
        defer: Option<usize>,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let hc = mhc.hc_mult;
        // 2026-09-25: `slot_base` is the first row's token index within the whole forward, so
        // the sub-chunks of one prefill use disjoint highway slots.
        let streams = ctx.buffers.hc_streams().offset(slot_base * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot_base * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot_base * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let (kt, ht, hct) = (k as u32, h as u32, hc as u32);

        let t_mhc = profile::start();
        if self.is_first {
            glm_hc_expand(
                gpu,
                mhc.kernels.hc_expand,
                hidden,
                streams,
                kt,
                ht,
                hct,
                stream,
            )?;
        }

        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.attn,
            hidden,
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
        self.norm(gpu, hidden, self.input_norm, normed, k, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        let attn_out = self.attn_mixer(
            normed,
            k,
            state,
            kv_cache,
            seq_len,
            block_table,
            ctx,
            stream,
            take_snapshots,
            is_prefill,
            core_rows,
        )?;
        // 2026-10-01: The verify path's FFN-head prefetch (as in `forward_one`); never in
        // prefill, whose sub-chunks are compute-bound.
        if !is_prefill {
            self.l2_prefetch(&self.prefetch.ffn_head, k, ctx, stream)?;
        }
        if let (Some(slot), true) = (defer, self.mixer_all_reduce) {
            return self.reduce_deferred(attn_out, hidden, k, slot, ctx, stream);
        }
        if self.mixer_all_reduce {
            self.reduce_probe(profile::REDUCE_ATTN_BAR, "attn", ctx, stream);
            let t = profile::start_hot();
            self.reduce_partial(attn_out, k, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_ATTN_ENQ, t);
            let t = profile::start_hot();
            profile::end(profile::REDUCE_ATTN, t, ctx.gpu, stream);
        }
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            attn_out,
            streams,
            post,
            comb,
            streams,
            kt,
            ht,
            hct,
            stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        Ok(())
    }

    /// 2026-10-01: The back of an `attn_half_inner(.., Some(slot))` over rows
    /// `slot_base..slot_base + k`: join `slot`, then `hc_post` of the reduced mixer output, which
    /// sits in rows `0..k` of `hidden`. Exactly the launches `attn_half` runs after its
    /// all-reduce, minus the profiling probes. Nothing when the layer has no mixer all-reduce.
    pub(in crate::glm5next_layer) fn attn_finish(
        &self,
        hidden: DevicePtr,
        k: usize,
        slot_base: usize,
        slot: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if !self.mixer_all_reduce {
            return Ok(());
        }
        let gpu = ctx.gpu;
        let h = self.hidden;
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let hc = mhc.hc_mult;
        let streams = ctx.buffers.hc_streams().offset(slot_base * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot_base * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot_base * hc * hc * 4);
        let (kt, ht, hct) = (k as u32, h as u32, hc as u32);
        self.reduce_join(slot, ctx, stream)?;
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            hidden,
            streams,
            post,
            comb,
            streams,
            kt,
            ht,
            hct,
            stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        Ok(())
    }
}
