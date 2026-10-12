// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The attention site's mixer call (`attn_mixer`), moved out of
//! `forward.rs::attn_half_inner` unchanged so the sequence-parallel staged prefill
//! (`staged/sp.rs`) runs the very same dispatch.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: `attn_mixer` issues the launches `attn_half_inner` issued between the input
//! norm and the all-reduce, in the same order.
//! 2026-10-12: Under `METRALE_GLM_KDA_STATE_F16=1` (an FP16 slot, `h_is_f16`) a KDA prefill call
//! runs every arm on the slot's FP32 staging blob: the state is widened into it before the arm
//! and narrowed back (round to nearest even) after, so the prefill kernels see FP32 as before.
//! A verify or decode passes the FP16 slot itself (glm5next_kda/state_f16.rs). Off, `h_is_f16`
//! is false on every state the KDA mixer accepts (`kda_state`), so no launch changes.

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
            Glm5NextMixer::Kda { layer, ws, .. } => {
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
                // 2026-10-03: An in-pass snapshot capture (`METRALE_GLM_SSM_INPASS_CAPTURE=1`)
                // whose point falls in this prefill call's rows; `None` on every other call and
                // whenever the pass plans no in-pass capture.
                let capture = if is_prefill && !take_snapshots {
                    ctx.midchunk_capture
                        .as_ref()
                        .and_then(|c| c.inpass_split(seq_len, k, st.h_state))
                } else {
                    None
                };
                // 2026-10-09: `METRALE_GLM_KDA_DEFER_COMMIT=1`: the model sets `kda_defer` before
                // a single-sequence verify; only a snapshot-taking walk reads it.
                let defer = if take_snapshots { st.kda_defer } else { None };
                // 2026-10-12: `METRALE_GLM_KDA_STATE_F16=1`: a prefill call runs on the FP32
                // staging blob (module header); `(slot, stage)` to narrow back after the arm.
                let staged = match st.h_prefill_stage {
                    Some(stage) if st.h_is_f16 && is_prefill && !take_snapshots => {
                        layer.widen_state(gpu, st.h_state, stage, stream)?;
                        Some((st.h_state, stage))
                    }
                    _ => None,
                };
                let kda = match staged {
                    Some((_, stage)) => KdaSeqState {
                        conv: st.conv_state,
                        recurrent: stage,
                        recurrent_f16: false,
                    },
                    None => KdaSeqState {
                        conv: st.conv_state,
                        recurrent: st.h_state,
                        recurrent_f16: st.h_is_f16,
                    },
                };
                Some((kda, snaps, capture, defer, staged))
            }
            Glm5NextMixer::Dsa(_) => None,
        };
        let t = profile::start();
        let attn_out = match (&self.mixer, &kda_ctx) {
            (Glm5NextMixer::Kda { layer, ws, .. }, Some((kda, snaps, capture, defer, staged))) => {
                // 2026-09-25: The chunked scan runs only for a prefill sub-chunk of more than
                // one row that asked for no intermediates: it never materialises the state
                // after an interior row, and its order differs from `decode_k`'s, which a
                // verify must match.
                // 2026-10-01: `METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1` takes the same sub-chunks
                // through the tensor-core chunked prefill, ahead of the FP32 chunked scan when
                // both levers are set; it falls back to `decode_k` itself when it cannot run.
                let chunkable = is_prefill && k > 1 && snaps.is_empty();
                // 2026-10-03: `METRALE_GLM_KDA_PREFILL_FLASHKDA=1` goes first. It returns false
                // with nothing launched for a sub-chunk below `FLASHKDA_MIN_ROWS` or a build or
                // geometry it cannot take, and the arms below run as before.
                let flashkda = chunkable
                    && crate::glm5next_kda::kda_prefill_flashkda()
                    && layer.prefill_flashkda(gpu, normed, k, kda, ws, stream)?;
                if flashkda {
                    // 2026-10-03: FlashKDA ran the recurrence; the partial is in `final_out`.
                } else if chunkable && crate::glm5next_kda::kda_prefill_chunked_tc() {
                    // 2026-10-06: The TC prefill takes the in-pass capture too (at a 16-row
                    // chunk boundary or the call's end; through `decode_k_capture` when it
                    // falls back) and says whether it copied; counted as the arm below counts.
                    let cap = capture.as_ref();
                    if layer.prefill_chunked_tc(gpu, normed, k, kda, ws, cap, stream)?
                        && let Some(mc) = ctx.midchunk_capture.as_ref()
                    {
                        mc.captured.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                } else if chunkable && kda_chunk_prefill() {
                    layer.prefill(gpu, normed, k, kda, ws, stream)?;
                } else if let (Some(cap), Some(mc)) = (capture, ctx.midchunk_capture.as_ref()) {
                    // 2026-10-03: The same launches as `decode_k` below, with the state copied
                    // into the reserved snapshot slot after row `cap.row - 1`. FlashKDA and the
                    // FP32 chunked scan above never capture; the model enables the in-pass
                    // capture only when neither can run (`inpass_ssm_capture_supported`), and
                    // without this count the engine registers nothing. 2026-10-06: the TC arm
                    // above captures and counts itself.
                    layer.decode_k_capture(gpu, normed, k, kda, ws, cap, stream)?;
                    mc.captured.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                } else if let Some(d) = defer.as_ref().filter(|_| !snaps.is_empty()) {
                    // 2026-10-09: The fused walk with row 0 reading the state the last commit
                    // left in its verify intermediate (`glm5next_kda/snap_fuse/defer.rs`).
                    layer.decode_k_defer(gpu, normed, k, kda, ws, snaps, d, stream)?;
                } else {
                    layer.decode_k(gpu, normed, k, kda, ws, snaps, stream)?;
                }
                // 2026-10-12: The staged prefill's state back into the FP16 slot.
                if let Some((slot, stage)) = *staged {
                    layer.narrow_state(gpu, stage, slot, stream)?;
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
