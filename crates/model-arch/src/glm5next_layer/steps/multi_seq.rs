// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The batched multi-sequence decode: N sequences, one token each, through the layer
//! in one call. Ported from rsafier's Atlas research/glm-exl3 branch (e69446eee: per-row highway
//! slot and metadata row, then one weight sweep over N rows; acf792e28: KDA projections batched
//! across sequences; 04beaac1f: per-sequence selection keeps long contexts batched).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Row `i` uses highway slot `ctx.hc_row_offset + i` and `attn_metadata.row_view(i)`, never
//!   slot 0 or metadata row 0 (the two aliases that made `decode_multi_seq_unsupported` true).
//! - The mHC sites, both norms, the MLP/MoE, the all-reduces and the KDA projections run once
//!   over all N rows; the KDA recurrence and the DSA attention run per sequence, each against
//!   that sequence's own state, page table and length.
//! - The highway slots are contiguous from the base: a decode step rebuilds the highway at the
//!   first layer (`hc_expand`) and collapses it at the last, so a row's slot never has to agree
//!   with a scheduler slot or outlive the step.

use super::*;

impl Glm5NextLayer {
    /// 2026-10-01: `TransformerLayer::decode_multi_seq` for this layer. `hidden` and `residual`
    /// are `[num_seqs, hidden]` BF16; `states[i]`, `seq_lens[i]` and `block_tables[i]` belong to
    /// row `i`. One row, or the MTP block (no hyper-connection, so no highway to alias), runs
    /// `forward_one` per row with that row's slot and metadata; otherwise `forward_n_seqs`.
    /// Errors, before any launch, when a slice is shorter than `num_seqs`: serving the rows that
    /// happen to be there would answer some sequences with another's activations.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn decode_n_seqs<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _tc = crate::glm5next_layer::dense_nv4_tc::MultiSeqScope::enter(num_seqs);
        if states.len() < num_seqs || seq_lens.len() < num_seqs || block_tables.len() < num_seqs {
            bail!(
                "GLM layer {}: decode_multi_seq of {num_seqs} sequences got states={} \
                 seq_lens={} block_tables={}",
                self.layer_idx,
                states.len(),
                seq_lens.len(),
                block_tables.len()
            );
        }
        let cap = ctx.buffers.max_batch_tokens();
        if ctx.hc_row_offset + num_seqs > cap {
            bail!(
                "GLM layer {}: decode_multi_seq rows {}..{} exceed the {cap}-slot mHC highway",
                self.layer_idx,
                ctx.hc_row_offset,
                ctx.hc_row_offset + num_seqs
            );
        }
        if self.mhc.is_none() || num_seqs == 1 {
            let h = self.hidden;
            for (i, state) in states.iter_mut().enumerate().take(num_seqs) {
                let off = i * h * 2;
                let row_ctx = ForwardContext {
                    attn_metadata: ctx.attn_metadata.as_ref().map(|m| m.row_view(i)),
                    ..*ctx
                };
                let mut bt = block_tables[i].clone();
                let mut no_disk = Vec::<u32>::new();
                let mut no_offloaded = Vec::<u32>::new();
                self.forward_one(
                    hidden.offset(off),
                    residual.offset(off),
                    ctx.hc_row_offset + i,
                    &mut **state,
                    kv_cache,
                    seq_lens[i],
                    &mut bt,
                    &mut no_disk,
                    &mut no_offloaded,
                    &row_ctx,
                    stream,
                )?;
            }
            return Ok(());
        }
        self.forward_n_seqs(
            hidden,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx.hc_row_offset,
            ctx,
            stream,
        )
    }

    /// 2026-10-01: `n` decode rows, one per sequence, with ONE sweep over the layer's weights:
    /// `forward_k`'s attention half with the row axis read as the sequence axis, then `ffn_half`
    /// unchanged. Row `i` uses highway slot `slot_base + i`.
    ///
    /// The mixer is the only per-sequence part. KDA runs `decode_n_seqs` (projections over all
    /// rows, each row's recurrence on its own state); DSA runs its single-row decode per row with
    /// `row_view(i)` metadata, so each row writes its own KV slot and indexer row and selects
    /// from its own indexer cache (`decode_multi_seq_selection_per_seq`). DSA writes its output
    /// projection over the row it was handed, so the rows land contiguous in `normed`.
    #[allow(clippy::too_many_arguments)]
    fn forward_n_seqs<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        n: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        slot_base: usize,
        ctx: &ForwardContext,
        stream: u64,
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
        let (nt, ht, hct) = (n as u32, h as u32, hc as u32);

        let t_mhc = profile::start();
        if self.is_first {
            glm_hc_expand(
                gpu,
                mhc.kernels.hc_expand,
                hidden,
                streams,
                nt,
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
            nt,
            ht,
            hct,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.input_norm, normed, n, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);

        let t = profile::start();
        let attn_out = match &self.mixer {
            Glm5NextMixer::Kda { layer, ws, .. } => {
                let mut seq_states = Vec::with_capacity(n);
                for state in states.iter_mut().take(n) {
                    let st = self.kda_state(&mut **state)?;
                    seq_states.push(KdaSeqState {
                        conv: st.conv_state,
                        recurrent: st.h_state,
                    });
                }
                layer.decode_n_seqs(gpu, normed, n, &seq_states, ws, stream)?;
                ws.final_out
            }
            // 2026-10-03: `METRALE_GLM_DSA_XSEQ_BATCH`: the projections once over all `n`
            // rows (`glm5next_dsa/layer/xseq.rs`), each row's metadata the loop's
            // `row_view(i)`; `decode_xseq` returns false, launching nothing, when it does not
            // engage, and the per-sequence loop below runs.
            Glm5NextMixer::Dsa(layer)
                if layer.xseq_attached() && {
                    let metas: Vec<_> = (0..n)
                        .map(|i| ctx.attn_metadata.as_ref().map(|m| m.row_view(i)))
                        .collect();
                    layer.decode_xseq(
                        normed,
                        &vec![1; n],
                        &mut states[..n],
                        kv_cache,
                        &seq_lens[..n],
                        &block_tables[..n],
                        &metas,
                        ctx,
                        stream,
                    )?
                } =>
            {
                normed
            }
            Glm5NextMixer::Dsa(layer) => {
                for (i, state) in states.iter_mut().enumerate().take(n) {
                    let row_ctx = ForwardContext {
                        attn_metadata: ctx.attn_metadata.as_ref().map(|m| m.row_view(i)),
                        ..*ctx
                    };
                    let mut bt = block_tables[i].clone();
                    layer.decode_k(
                        normed.offset(i * h * 2),
                        1,
                        &mut **state,
                        kv_cache,
                        seq_lens[i],
                        &mut bt,
                        &row_ctx,
                        stream,
                        // 2026-10-01: `is_prefill` false, as `Glm5NextDsaLayer::decode` passes.
                        false,
                    )?;
                }
                normed
            }
        };
        profile::end(profile::KDA, t, gpu, stream);
        if self.mixer_all_reduce {
            self.reduce_probe(profile::REDUCE_ATTN_BAR, "attn", ctx, stream);
            let t = profile::start_hot();
            self.reduce_partial(attn_out, n, ctx, stream)?;
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
            nt,
            ht,
            hct,
            stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        // 2026-10-01: The FFN site over all `n` rows: one MoE / dense MLP call, one all-reduce,
        // and on the last layer `hc_head_mean` of every row.
        self.ffn_half(hidden, n, ctx, stream, slot_base, n)
    }
}
