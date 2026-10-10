// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Phase 3 of the two-phase and batched GDN prefill
//! (`prefill_phase3_inner`): gated norm, out_proj, TP reduce, post-norm, FFN
//! and residual adds; and the non-pooled `alloc_state_inner`.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - A state `alloc_state_inner` returns has zeroed h and conv buffers.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase3_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;

        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let value_dim = nv * vd;

        // 2026-09-25: Read the GDN output and Z at `token_offset` in `gdn_bufs`;
        // both are `value_dim` wide per token.
        let gdn_out_chunk = gdn_bufs.output.offset(token_offset * value_dim * bf16);
        let z_chunk = gdn_bufs.z.offset(token_offset * value_dim * bf16);

        // 2026-09-25: The gated-norm output reuses `ssm_qkvz`, as
        // `prefill_block` does.
        let normed_out_buf = ctx.buffers.ssm_qkvz();
        ops::gated_rms_norm_prefill(
            ctx.gpu,
            self.gated_rms_norm_prefill_k,
            gdn_out_chunk,
            z_chunk,
            &self.ssm.norm,
            normed_out_buf,
            nv as u32,
            vd as u32,
            eps,
            k,
            value_dim as u32,
            value_dim as u32,
            stream,
        )?;

        let out_proj_buf = ctx.buffers.moe_output();
        // 2026-09-25: The same `prefill_out_proj_dispatch` as the
        // single-stream prefill.
        self.prefill_out_proj_dispatch(ctx, normed_out_buf, out_proj_buf, k, h, value_dim, stream)?;
        // 2026-09-25: Sum the out_proj partials across TP ranks and apply the
        // out_proj LoRA delta.
        self.ssm_tp_all_reduce(out_proj_buf, normed_out_buf, num_tokens, ctx, stream)?;

        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            out_proj_buf,
            &self.post_attn_norm,
            ctx.buffers.norm_output(),
            residual,
            num_tokens as u32,
            h as u32,
            eps,
            stream,
        )?;
        self.ffn
            .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)?;
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            ctx.buffers.moe_output(),
            (num_tokens * h) as u32,
            stream,
        )?;

        Ok(())
    }

    pub(super) fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let h_state = gpu.alloc(self.h_state_bytes)?;
        gpu.memset(h_state, 0, self.h_state_bytes)?;
        let conv_state = gpu.alloc(self.conv_state_bytes)?;
        gpu.memset(conv_state, 0, self.conv_state_bytes)?;
        Ok(Box::new(SsmLayerState {
            kda_defer: None,
            h_state,
            conv_state,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
            h_is_f16: false,
            // 2026-09-25: This state owns a private FP32 `h_state_bytes` blob,
            // not a pool slot, so there is no staging blob to widen into.
            h_prefill_stage: None,
            ple: None,
        }))
    }
}
