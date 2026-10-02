// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The attention site's mixer call (`attn_mixer`), moved out of
//! `forward.rs::attn_half_inner` unchanged so the sequence-parallel staged prefill
//! (`staged/sp.rs`) runs the very same dispatch.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: `attn_mixer` issues the launches `attn_half_inner` issued between the input
//! norm and the all-reduce, in the same order.

use super::*;

impl Glm5NextLayer {
    /// 2026-10-01: The mixer over `k` rows of `normed` (positions from `seq_len`): KDA
    /// (`decode_k`, or the chunked scan under `METRALE_GLM_KDA_CHUNK_PREFILL=1` for a prefill
    /// sub-chunk) leaving its partial in the shared workspace's `final_out`, or DSA writing its
    /// partial over `normed` (`decode_k_wide` when `core_rows < k` on a prefill). Returns where
    /// the partial is.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn attn_mixer(
        &self,
        normed: DevicePtr,
        k: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        take_snapshots: bool,
        is_prefill: bool,
        // 2026-10-01: The DSA selection and attend width; `k` runs `decode_k` as before (see
        // `attn_half_inner`). KDA ignores it.
        core_rows: usize,
    ) -> Result<DevicePtr> {
        let gpu = ctx.gpu;
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
        let t = profile::start();
        let attn_out = match (&self.mixer, &kda_ctx) {
            (Glm5NextMixer::Kda { layer, ws, .. }, Some((kda, snaps))) => {
                // 2026-09-25: The chunked scan runs only for a prefill sub-chunk of more than
                // one row that asked for no intermediates: it never materialises the state
                // after an interior row, and its order differs from `decode_k`'s, which a
                // verify must match.
                // 2026-10-01: `METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1` takes the same sub-chunks
                // through the tensor-core chunked prefill, ahead of the FP32 chunked scan when
                // both levers are set; it falls back to `decode_k` itself when it cannot run.
                let chunkable = is_prefill && k > 1 && snaps.is_empty();
                if chunkable && crate::glm5next_kda::kda_prefill_chunked_tc() {
                    layer.prefill_chunked_tc(gpu, normed, k, kda, ws, stream)?;
                } else if chunkable && kda_chunk_prefill() {
                    layer.prefill(gpu, normed, k, kda, ws, stream)?;
                } else {
                    layer.decode_k(gpu, normed, k, kda, ws, snaps, stream)?;
                }
                ws.final_out
            }
            (Glm5NextMixer::Dsa(layer), _) => {
                // 2026-09-25: DSA's `decode_k` writes its output projection over its input
                // buffer.
                // 2026-10-01: So does `decode_k_wide`, over all `k` rows.
                if is_prefill && core_rows < k {
                    layer.decode_k_wide(
                        normed,
                        k,
                        core_rows,
                        state,
                        kv_cache,
                        seq_len,
                        block_table,
                        ctx,
                        stream,
                    )?;
                } else {
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
                }
                normed
            }
            (Glm5NextMixer::Kda { .. }, None) => {
                bail!("GLM layer {}: KDA mixer without KDA state", self.layer_idx)
            }
        };
        profile::end(profile::KDA, t, gpu, stream);
        Ok(attn_out)
    }
}
