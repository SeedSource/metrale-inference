// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `impl TransformerLayer for NemotronMamba2Layer`: the decode
//! steps listed in the parent module, prefill (`prefill.rs`) and the per-layer
//! SSM state.
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants:
//! - `alloc_state` returns zeroed FP32 `h_state` and `conv_state` buffers.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::NemotronMamba2Layer;
use metrale_model_layers::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use metrale_model_layers::layer::{
    LayerAuxState, LayerCapabilities, LayerGraphHooks, LayerSplitPrefill, LayerWeightSetup,
    LayerWriteOnAccept,
};
use metrale_model_layers::layers::ops;

impl TransformerLayer for NemotronMamba2Layer {
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut metrale_cache::kv_cache::PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;

        let proj = ctx.buffers.ssm_qkvz();
        // 2026-09-25: Projection precedence, here and for out_proj: native
        // BF16, then native FP8 (`w8a16_gemv`), then the NVFP4 GEMV.
        if let Some(ref w) = self.in_proj_bf16 {
            // 2026-09-25: Native BF16: `ssm.in_proj` is NULL here (never quantized).
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                normed,
                w,
                proj,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        } else if let Some(ref fp8w) = self.in_proj_fp8 {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                proj,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        } else {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                ctx.levers.gemv_sw,
                normed,
                &self.ssm.in_proj,
                proj,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        }

        let z_ptr = proj;
        let xbc_ptr = proj.offset(self.d_inner * 2);
        let dt_ptr = proj.offset((self.d_inner + self.d_xbc) * 2);

        let xbc_out = ctx.buffers.ssm_deinterleaved();
        self.conv1d_update_biased(
            ctx.gpu,
            ssm_state.conv_state,
            xbc_ptr,
            xbc_out,
            self.d_xbc as u32,
            self.d_conv as u32,
            1,
            stream,
        )?;

        let x_ptr = xbc_out;
        let gs = self.n_groups * self.state_size;
        let b_ptr = xbc_out.offset(self.d_inner * 2);
        let c_ptr = xbc_out.offset((self.d_inner + gs) * 2);

        let y_ptr = ctx.buffers.attn_output();
        self.ssm_decode(
            ctx.gpu,
            ssm_state.h_state,
            x_ptr,
            b_ptr,
            c_ptr,
            dt_ptr,
            y_ptr,
            1,
            stream,
        )?;

        let gated_out = ctx.buffers.norm_output();
        let group_size = (self.d_inner / self.n_groups) as u32;
        ops::gated_rms_norm(
            ctx.gpu,
            self.gated_rms_norm_k,
            y_ptr,
            z_ptr,
            &self.ssm.ssm_norm,
            gated_out,
            1,
            self.d_inner as u32,
            self.in_proj_size as u32,
            eps,
            group_size,
            stream,
        )?;

        let out = ctx.buffers.qkv_output();
        if let Some(ref w) = self.out_proj_bf16 {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                gated_out,
                w,
                out,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else if let Some(ref fp8w) = self.out_proj_fp8 {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                gated_out,
                fp8w.weight,
                fp8w.row_scale,
                out,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                ctx.levers.gemv_sw,
                gated_out,
                &self.ssm.out_proj,
                out,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        }

        // 2026-09-25: `hidden` still holds the layer input (`rms_norm_residual`
        // only reads it), so this adds the mixer output to it.
        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, out, h as u32, stream)?;

        Ok(())
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_ssm(hidden, residual, num_tokens, state, ctx, stream)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
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
            h_prefill_stage: None,
            ple: None,
        }))
    }
}

impl LayerCapabilities for NemotronMamba2Layer {}
impl LayerWeightSetup for NemotronMamba2Layer {}
impl LayerWriteOnAccept for NemotronMamba2Layer {}
impl LayerGraphHooks for NemotronMamba2Layer {}
impl LayerAuxState for NemotronMamba2Layer {}
impl LayerSplitPrefill for NemotronMamba2Layer {}
