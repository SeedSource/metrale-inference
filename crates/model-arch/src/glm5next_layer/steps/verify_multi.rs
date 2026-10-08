// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The batched MTP verify (`METRALE_GLM_BATCHED_VERIFY=1`): `ks[i]` draft rows for
//! each of N sequences through the layer in one call, `TransformerLayer::decode_verify_multi_seqs`
//! for this layer. The shape of `multi_seq.rs`'s `forward_n_seqs` with the row axis read as
//! `R = Σ ks` sequence-major verify rows.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Sequence `i` owns rows `off_i..off_i + ks[i]` (`verify_row_offsets`), highway slots
//!   `ctx.hc_row_offset + off_i ..`, and the metadata rows `verify_rows_view(m, off_i, ks[i])`.
//! - The mHC sites, both norms, the MLP/MoE, the all-reduces and the KDA projections run once
//!   over all R rows; the KDA recurrence (with its per-row snapshots) and the DSA attention run
//!   per sequence, each against that sequence's own state, page table and length, exactly as
//!   that sequence's `decode_batched` (`forward_k`) runs them.
//! - The launch and collective sequence depends only on `ks`, the per-sequence lengths and the
//!   levers, so two ranks handed the same batch issue the same collectives in the same order.

use super::*;
use metrale_model_layers::layer::AttnMetadataDev;

/// 2026-10-02: Prefix sums of `ks`: entry `i` is sequence `i`'s first row, the last entry the
/// row total `R`. Length `ks.len() + 1`.
pub(in crate::glm5next_layer) fn verify_row_offsets(ks: &[usize]) -> Vec<usize> {
    let mut off = Vec::with_capacity(ks.len() + 1);
    let mut acc = 0usize;
    for &k in ks {
        off.push(acc);
        acc += k;
    }
    off.push(acc);
    off
}

/// 2026-10-02: Rows `base..base + k` of a batched verify's metadata as one sequence's `k`-row
/// metadata: `row_view(base)` with `num_seqs = k`, so `Glm5NextDsaLayer::decode_k` reads every
/// row's position, KV slot, length and block table from the device (its `rowwise_meta` arm
/// takes metadata whose `num_seqs` equals the row count) instead of the host fallback.
pub(in crate::glm5next_layer) fn verify_rows_view(
    m: &AttnMetadataDev,
    base: usize,
    k: usize,
) -> AttnMetadataDev {
    AttnMetadataDev {
        num_seqs: k as u32,
        ..m.row_view(base)
    }
}

impl Glm5NextLayer {
    /// 2026-10-02: Most verify rows one `verify_n_seqs` call takes: the MLP workspace rows and,
    /// on a KDA layer, the KDA workspace rows. The loader sizes both for at least
    /// `max(16, METRALE_GLM_PREFILL_ROWS)` rows, and 64 under the lever (`batched_verify_rows`).
    pub(in crate::glm5next_layer) fn verify_rows_cap(&self) -> usize {
        let mlp = self.mlp_ws.max_rows();
        match &self.mixer {
            Glm5NextMixer::Kda { ws, .. } => mlp.min(ws.max_tokens()),
            Glm5NextMixer::Dsa(_) => mlp,
        }
    }

    /// 2026-10-02: `decode_verify_multi_seqs` for this layer. `hidden` holds the `R = Σ ks`
    /// verify rows sequence-major; `states[i]`, `seq_lens[i]` (the pre-verify length) and
    /// `block_tables[i]` belong to sequence `i`. A KDA layer writes the state after row `t` of
    /// sequence `i` to its intermediate `t` for `t < ks[i] - 1`, as `decode_batched` does, so
    /// `commit_accepted_prefix` and the rollback restore per sequence as before. Errors, before
    /// any launch, on a layer without a hyper-connection, a slice shorter than `ks`, a zero
    /// `ks[i]`, rows past the mHC highway or `verify_rows_cap`, or a KDA pool with too few
    /// intermediates.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn verify_n_seqs<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        ks: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let n = ks.len();
        let _tc = crate::glm5next_layer::dense_nv4_tc::MultiSeqScope::enter(n);
        let off = verify_row_offsets(ks);
        let r = off[n];
        if n == 0
            || ks.contains(&0)
            || states.len() < n
            || seq_lens.len() < n
            || block_tables.len() < n
        {
            bail!(
                "GLM layer {}: batched verify of ks={ks:?} got states={} seq_lens={} \
                 block_tables={}",
                self.layer_idx,
                states.len(),
                seq_lens.len(),
                block_tables.len()
            );
        }
        let cap = ctx.buffers.max_batch_tokens();
        let slot_base = ctx.hc_row_offset;
        if slot_base + r > cap {
            bail!(
                "GLM layer {}: batched verify rows {slot_base}..{} exceed the {cap}-slot mHC \
                 highway",
                self.layer_idx,
                slot_base + r
            );
        }
        if r > self.mlp_ws.max_rows() {
            bail!(
                "GLM layer {}: a {r}-row batched verify exceeds the MLP workspace built for {}",
                self.layer_idx,
                self.mlp_ws.max_rows()
            );
        }
        let gpu = ctx.gpu;
        let h = self.hidden;
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        // 2026-10-02: KDA inputs gathered (and the pool depth checked, as `decode_batched` does)
        // before the first launch, so a refusal launches nothing.
        let kda_inputs = match &self.mixer {
            Glm5NextMixer::Kda { ws, .. } => {
                if r > ws.max_tokens() {
                    bail!(
                        "GLM layer {}: a {r}-row batched verify exceeds the KDA workspace \
                         built for {}",
                        self.layer_idx,
                        ws.max_tokens()
                    );
                }
                let mut seq_states = Vec::with_capacity(n);
                let mut snaps = Vec::with_capacity(n);
                for (i, state) in states.iter_mut().enumerate().take(n) {
                    let st = self.kda_state(&mut **state)?;
                    let k = ks[i];
                    if st.h_state_intermediates.len() + 1 < k
                        || st.conv_state_intermediates.len() + 1 < k
                    {
                        bail!(
                            "GLM layer {}: a {k}-row verify of batch row {i} needs {} \
                             per-token state snapshots but the pool has h={} conv={}",
                            self.layer_idx,
                            k - 1,
                            st.h_state_intermediates.len(),
                            st.conv_state_intermediates.len(),
                        );
                    }
                    seq_states.push(KdaSeqState {
                        conv: st.conv_state,
                        recurrent: st.h_state,
                    });
                    snaps.push(
                        (0..k - 1)
                            .map(|t| (st.h_state_intermediates[t], st.conv_state_intermediates[t]))
                            .collect::<Vec<_>>(),
                    );
                }
                Some((seq_states, snaps))
            }
            Glm5NextMixer::Dsa(_) => None,
        };

        let hc = mhc.hc_mult;
        let streams = ctx.buffers.hc_streams().offset(slot_base * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot_base * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot_base * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let (rt, ht, hct) = (r as u32, h as u32, hc as u32);

        let t_mhc = profile::start();
        if self.is_first {
            glm_hc_expand(
                gpu,
                mhc.kernels.hc_expand,
                hidden,
                streams,
                rt,
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
            rt,
            ht,
            hct,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.input_norm, normed, r, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);

        let t = profile::start();
        let attn_out = match (&self.mixer, kda_inputs) {
            (Glm5NextMixer::Kda { layer, ws, .. }, Some((seq_states, snaps))) => {
                layer.decode_verify_n_seqs(gpu, normed, ks, &seq_states, &snaps, ws, stream)?;
                ws.final_out
            }
            // 2026-10-03: `METRALE_GLM_DSA_XSEQ_BATCH`: the projections once over all R rows
            // (`glm5next_dsa/layer/xseq.rs`), each sequence's metadata the loop's
            // `verify_rows_view`; `decode_xseq` returns false, launching nothing, when it does
            // not engage, and the per-sequence loop below runs.
            (Glm5NextMixer::Dsa(layer), _)
                if layer.xseq_attached() && {
                    let metas: Vec<_> = (0..n)
                        .map(|i| {
                            ctx.attn_metadata
                                .as_ref()
                                .map(|m| verify_rows_view(m, off[i], ks[i]))
                        })
                        .collect();
                    layer.decode_xseq(
                        normed,
                        ks,
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
            (Glm5NextMixer::Dsa(layer), _) => {
                for (i, state) in states.iter_mut().enumerate().take(n) {
                    let row_ctx = ForwardContext {
                        attn_metadata: ctx
                            .attn_metadata
                            .as_ref()
                            .map(|m| verify_rows_view(m, off[i], ks[i])),
                        ..*ctx
                    };
                    let mut bt = block_tables[i].clone();
                    layer.decode_k(
                        normed.offset(off[i] * h * 2),
                        ks[i],
                        &mut **state,
                        kv_cache,
                        seq_lens[i],
                        &mut bt,
                        &row_ctx,
                        stream,
                        // 2026-10-02: `is_prefill` false: a speculative verify, as
                        // `decode_batched` passes it.
                        false,
                    )?;
                }
                normed
            }
            (Glm5NextMixer::Kda { .. }, None) => {
                bail!("GLM layer {}: KDA mixer without KDA state", self.layer_idx)
            }
        };
        profile::end(profile::KDA, t, gpu, stream);
        // 2026-10-02: The verify path's FFN-head prefetch, as `attn_half_inner` issues it; a
        // no-op unless `METRALE_GLM_DECODE_L2_PREFETCH=1`.
        self.l2_prefetch(&self.prefetch.ffn_head, r, ctx, stream)?;
        if self.mixer_all_reduce {
            self.reduce_probe(profile::REDUCE_ATTN_BAR, "attn", ctx, stream);
            let t = profile::start_hot();
            self.reduce_partial(attn_out, r, ctx, stream)?;
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
            rt,
            ht,
            hct,
            stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        // 2026-10-02: The FFN site over all R rows: one MoE / dense MLP call, one all-reduce,
        // and on the last layer `hc_head_mean` of every row.
        self.ffn_half(hidden, r, ctx, stream, slot_base, r)
    }
}

#[cfg(test)]
mod tests {
    use super::{verify_row_offsets, verify_rows_view};
    use metrale_gpu_runtime::gpu::DevicePtr;
    use metrale_model_layers::layer::AttnMetadataDev;

    #[test]
    fn row_offsets_are_the_prefix_sums_of_ks() {
        assert_eq!(verify_row_offsets(&[4, 4, 4, 4]), [0, 4, 8, 12, 16]);
        assert_eq!(verify_row_offsets(&[4, 2, 3]), [0, 4, 6, 9]);
        assert_eq!(verify_row_offsets(&[]), [0]);
    }

    /// 2026-10-02: A sequence's view starts at its first row at every array's element width and
    /// describes exactly its `k` rows; null arrays stay null.
    #[test]
    fn rows_view_advances_each_array_and_narrows_to_k_rows() {
        let m = AttnMetadataDev {
            positions: DevicePtr(0x1000),
            positions_h: DevicePtr(0x1000),
            positions_w: DevicePtr(0x1000),
            slot: DevicePtr(0x2000),
            seq_len: DevicePtr(0x3000),
            block_table: DevicePtr(0x4000),
            max_blocks_per_seq: 10,
            num_seqs: 16,
            seq_slot: DevicePtr(0),
            moe_row_adapter: DevicePtr(0),
        };
        let v = verify_rows_view(&m, 6, 3);
        assert_eq!(v.num_seqs, 3);
        assert_eq!(v.positions.0, 0x1000 + 6 * 4);
        assert_eq!(v.slot.0, 0x2000 + 6 * 8);
        assert_eq!(v.seq_len.0, 0x3000 + 6 * 4);
        assert_eq!(v.block_table.0, 0x4000 + 6 * 10 * 4);
        assert_eq!(v.max_blocks_per_seq, 10);
        assert_eq!(v.seq_slot.0, 0);
        assert_eq!(v.moe_row_adapter.0, 0);
    }
}
