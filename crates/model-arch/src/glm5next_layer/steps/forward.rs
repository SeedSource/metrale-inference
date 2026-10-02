// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The mHC layer bodies: `forward_one` (one token) and `forward_k` (K rows of one
//! sequence in one call).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `forward_one` uses highway slot `slot`; `forward_k` uses slots `slot_base..slot_base + k`.

use super::*;

/// 2026-09-25: `METRALE_GLM_KDA_CHUNK_PREFILL=1` sends a prefill sub-chunk's KDA mixer through
/// the chunked scan (`Glm5NextKdaLayer::prefill`) instead of `decode_k`'s per-token recurrence.
/// Off unless set to `1`; read once. The chunked scan computes the recurrence chunk by chunk, in
/// a different order, so its output is not bit-identical to the per-token walk.
fn kda_chunk_prefill() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_KDA_CHUNK_PREFILL").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_KDA_CHUNK_PREFILL=1 - GLM prefill KDA uses the CHUNKED scan \
                 (kda_chunk_prepare + kda_chunk_scan). Not bit-identical to the per-token \
                 recurrent walk."
            );
        }
        on
    })
}

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

        // 2026-10-01: A DFlash tap layer (`dflash_tap`) collapses its completed highway into
        // `hidden` the same way, so the engine's capture after this layer reads the drafter's
        // tap (the unweighted mean of the `hc_mult` streams) instead of the FFN-site `hc_pre`
        // scratch. Nothing reads `hidden` between here and the next layer's `hc_pre`.
        if self.is_last || self.dflash_tap {
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
        // 2026-09-25: True only for a prefill sub-chunk (`Glm5NextLayer::prefill`). It selects
        // the DSA batched selector (`batch_select_enabled`) and, with
        // `METRALE_GLM_KDA_CHUNK_PREFILL=1`, the KDA chunked scan, both prefill-only.
        is_prefill: bool,
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

        let kda_ctx = match &self.mixer {
            Glm5NextMixer::Kda { ws, .. } => {
                if k > ws.max_tokens() {
                    bail!(
                        "GLM layer {}: a {k}-token verify exceeds the KDA workspace built for {}",
                        self.layer_idx,
                        ws.max_tokens()
                    );
                }
                let st = self.kda_state(state)?;
                // 2026-09-25: Intermediates only when `take_snapshots` (a verify); prefill passes
                // none. `decode_k` looks each row up with `get`, so an empty list takes none.
                let snaps: Vec<(DevicePtr, DevicePtr)> = if take_snapshots {
                    (0..k.saturating_sub(1))
                        .map(|t| (st.h_state_intermediates[t], st.conv_state_intermediates[t]))
                        .collect()
                } else {
                    Vec::new()
                };
                Some((
                    KdaSeqState {
                        conv: st.conv_state,
                        recurrent: st.h_state,
                    },
                    snaps,
                ))
            }
            Glm5NextMixer::Dsa(_) => None,
        };

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
        let t = profile::start();
        let attn_out = match (&self.mixer, &kda_ctx) {
            (Glm5NextMixer::Kda { layer, ws, .. }, Some((kda, snaps))) => {
                // 2026-09-25: The chunked scan runs only for a prefill sub-chunk of more than
                // one row that asked for no intermediates: it never materialises the state
                // after an interior row, and its order differs from `decode_k`'s, which a
                // verify must match.
                if is_prefill && k > 1 && snaps.is_empty() && kda_chunk_prefill() {
                    layer.prefill(gpu, normed, k, kda, ws, stream)?;
                } else {
                    layer.decode_k(gpu, normed, k, kda, ws, snaps, stream)?;
                }
                ws.final_out
            }
            (Glm5NextMixer::Dsa(layer), _) => {
                // 2026-09-25: DSA's `decode_k` writes its output projection over its input
                // buffer.
                layer.decode_k(
                    normed,
                    k,
                    state,
                    kv_cache,
                    seq_len,
                    block_table,
                    ctx,
                    stream,
                    is_prefill,
                )?;
                normed
            }
            (Glm5NextMixer::Kda { .. }, None) => {
                bail!("GLM layer {}: KDA mixer without KDA state", self.layer_idx)
            }
        };
        profile::end(profile::KDA, t, gpu, stream);
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

    /// 2026-09-29: The FFN half of `forward_k` over rows `slot_base..slot_base + k`: the FFN
    /// site's `hc_pre`, the post-attention norm, the MLP and its all-reduce, `hc_post`, and on
    /// the last layer `hc_head_mean`. Row `t` reads only highway slot `slot_base + t` (and the
    /// layer's weights); every launch here is row-local except the MLP's dense GEMMs, which run
    /// in consecutive slices of `dense_slice` rows (`mlp_forward`). Errors on a layer without a
    /// hyper-connection.
    pub(in crate::glm5next_layer) fn ffn_half(
        &self,
        hidden: DevicePtr,
        k: usize,
        ctx: &ForwardContext,
        stream: u64,
        slot_base: usize,
        dense_slice: usize,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let hc = mhc.hc_mult;
        let streams = ctx.buffers.hc_streams().offset(slot_base * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot_base * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot_base * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();
        let (kt, ht, hct) = (k as u32, h as u32, hc as u32);

        let t_mhc = profile::start();
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.ffn,
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
        self.norm(gpu, hidden, self.post_attn_norm, normed, k, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        self.mlp_forward(normed, ffn_out, k, dense_slice, ctx, stream)?;
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            ffn_out,
            streams,
            post,
            comb,
            streams,
            kt,
            ht,
            hct,
            stream,
        )?;
        // 2026-10-01: DFlash tap layer: the same collapse over rows `0..k` (see `forward_one`).
        // Row-local, so the staged prefill's FFN windows collapse exactly their own rows.
        if self.is_last || self.dflash_tap {
            hc_head_mean(
                gpu,
                mhc.kernels.hc_head,
                streams,
                hidden,
                kt,
                ht,
                hct,
                stream,
            )?;
        }
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        if self.is_last {
            profile::step();
        }
        Ok(())
    }
}
