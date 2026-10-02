// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The FFN half of a GLM-5.3 layer (`ffn_half`, moved here unchanged from
//! `forward.rs` to keep that file under the 500-line cap), plus its overlapped form for the
//! staged prefill (`ffn_half_inner(.., Some(slot))` then `ffn_finish`,
//! `METRALE_GLM_PREFILL_COMM_OVERLAP=1`) and the shared tail `ffn_back`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: `ffn_half` issues the launch sequence it always has: `hc_pre`, the norm,
//! `mlp_forward` (MLP then all-reduce), then `ffn_back` (`hc_post`, `hc_head_mean` on the last
//! layer).

use super::*;

impl Glm5NextLayer {
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
        self.ffn_half_inner(hidden, k, ctx, stream, slot_base, dense_slice, None)
    }

    /// 2026-10-01: `ffn_half`, or with `defer = Some(slot)` and an MLP all-reduce, its front
    /// only: `hc_pre`, the norm and the MLP (`mlp_compute`), then the partial copied into rows
    /// `0..k` of `hidden` (dead after the norm read them) and its all-reduce issued there as a
    /// deferred all-reduce under `slot`; `ffn_finish` later joins it and runs the back
    /// (`ffn_back`). Without an MLP all-reduce, `defer` is ignored and `ffn_finish` does nothing.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn ffn_half_inner(
        &self,
        hidden: DevicePtr,
        k: usize,
        ctx: &ForwardContext,
        stream: u64,
        slot_base: usize,
        dense_slice: usize,
        defer: Option<usize>,
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
        if let (Some(slot), true) = (defer, self.mlp_cfg.needs_all_reduce()) {
            self.mlp_compute(normed, ffn_out, k, dense_slice, ctx, stream)?;
            return self.reduce_deferred(ffn_out, hidden, k, slot, ctx, stream);
        }
        self.mlp_forward(normed, ffn_out, k, dense_slice, ctx, stream)?;
        self.ffn_back(ffn_out, hidden, k, slot_base, ctx, stream)
    }

    /// 2026-10-01: The back of an `ffn_half_inner(.., Some(slot))` over rows
    /// `slot_base..slot_base + k`: join `slot`, then `ffn_back` with the reduced MLP output,
    /// which sits in rows `0..k` of `hidden`. Nothing when the layer has no MLP all-reduce.
    pub(in crate::glm5next_layer) fn ffn_finish(
        &self,
        hidden: DevicePtr,
        k: usize,
        slot_base: usize,
        slot: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if !self.mlp_cfg.needs_all_reduce() {
            return Ok(());
        }
        self.reduce_join(slot, ctx, stream)?;
        self.ffn_back(hidden, hidden, k, slot_base, ctx, stream)
    }

    /// 2026-10-01: The tail of the FFN half, split out of `ffn_half` unchanged: `hc_post` of the
    /// (reduced) MLP output `x`, and on the last layer `hc_head_mean` into rows `0..k` of
    /// `hidden`. `x` may be `hidden` itself: `hc_post` reads it before `hc_head_mean` writes it,
    /// in stream order.
    fn ffn_back(
        &self,
        x: DevicePtr,
        hidden: DevicePtr,
        k: usize,
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
        let (kt, ht, hct) = (k as u32, h as u32, hc as u32);
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            x,
            streams,
            post,
            comb,
            streams,
            kt,
            ht,
            hct,
            stream,
        )?;
        if self.is_last {
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
