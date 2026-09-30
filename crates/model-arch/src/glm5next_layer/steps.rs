// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The building blocks of `Glm5NextLayer`: norm, mixer, MLP, all-reduce, the MTP
//! block's plain residual path, and the state accessors.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;

mod drafter;
mod forward;
pub(super) mod staged;

impl Glm5NextLayer {
    /// 2026-09-25: `rms_norm_vanilla` over `rows` contiguous `[hidden]` rows in one launch. The
    /// kernel runs one block per row (`token = blockIdx.x`) and the blocks share nothing, so the
    /// result equals `rows` single-row launches.
    fn norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.rms_norm_k)
            .grid([rows as u32, 1, 1])
            .block([(self.hidden.min(1024)) as u32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(self.hidden as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        Ok(())
    }

    /// 2026-09-25: Run the mixer on `normed`; returns the pointer that holds its output.
    #[allow(clippy::too_many_arguments)]
    fn mixer_forward(
        &self,
        normed: DevicePtr,
        residual: DevicePtr,
        st: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_offloaded: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match &self.mixer {
            Glm5NextMixer::Kda { layer, ws, .. } => {
                let ssm = self.kda_state(st)?;
                let kda = KdaSeqState {
                    conv: ssm.conv_state,
                    recurrent: ssm.h_state,
                };
                let t = profile::start();
                layer.decode(ctx.gpu, normed, &kda, ws, stream)?;
                profile::end(profile::KDA, t, ctx.gpu, stream);
                Ok(ws.final_out)
            }
            Glm5NextMixer::Dsa(layer) => {
                let dsa: &mut Glm5NextDsaState = self.dsa_state(st)?;
                layer.decode(
                    normed,
                    residual,
                    dsa,
                    kv_cache,
                    seq_len,
                    block_table,
                    disk_block_ids,
                    disk_offloaded,
                    ctx,
                    stream,
                )?;
                // 2026-09-25: DSA's `decode` writes its output projection over its input buffer.
                Ok(normed)
            }
        }
    }

    /// 2026-09-25: While profiling, a 2-byte all-reduce timed into bucket `bar`, then the
    /// route-trace line for `site`. No-op when profiling is off or there is no communicator.
    fn reduce_probe(&self, bar: usize, site: &str, ctx: &ForwardContext, stream: u64) {
        if !profile::on() {
            return;
        }
        let Some(comm) = ctx.comm else { return };
        let t = profile::start_hot();
        let p = profile::probe_buf(ctx.gpu);
        if p != 0 {
            let _ = comm.all_reduce_async(p, 2, stream);
        }
        let us = profile::end_us(bar, t, ctx.gpu, stream);
        profile::trace_bar(
            site,
            self.layer_idx,
            matches!(self.mlp, Glm5NextMlpSite::Moe(_)),
            us,
        );
    }

    /// 2026-09-25: All-reduce `rows` contiguous `[hidden]` BF16 rows in one collective when
    /// `ctx.comm` is set: blocking under graph capture, asynchronous otherwise. No-op without a
    /// communicator.
    fn reduce_partial(
        &self,
        p: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(comm) = ctx.comm {
            let bytes = rows * self.hidden * 2;
            if ctx.graph_capture {
                comm.all_reduce(p.0, bytes)?;
            } else {
                comm.all_reduce_async(p.0, bytes, stream)?;
            }
        }
        Ok(())
    }

    /// 2026-09-25: The MLP over `rows` rows of `normed` into `out`, then one all-reduce of all
    /// rows when `Glm5NextMlpConfig::needs_all_reduce`.
    ///
    /// 2026-09-29: The dense GEMMs (router, shared expert, dense MLP) run in consecutive slices of
    /// `dense_slice` rows (`forward_moe_sliced`, `forward_dense_sliced`); `dense_slice >= rows` is
    /// one slice, the unsliced launch.
    fn mlp_forward(
        &self,
        normed: DevicePtr,
        out: DevicePtr,
        rows: usize,
        dense_slice: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let t_dense = matches!(self.mlp, Glm5NextMlpSite::Dense(_))
            .then(profile::start)
            .flatten();
        match &self.mlp {
            Glm5NextMlpSite::Dense(w) => forward_dense_sliced(
                ctx.gpu,
                &self.mlp_kernels,
                &self.mlp_cfg,
                w,
                self.mlp_cfg.local_dense_intermediate,
                normed,
                out,
                rows,
                dense_slice,
                &self.mlp_ws,
                stream,
            )?,
            Glm5NextMlpSite::Moe(w) => forward_moe_sliced(
                ctx.gpu,
                &self.mlp_kernels,
                &self.mlp_cfg,
                w,
                normed,
                out,
                rows,
                dense_slice,
                &self.mlp_ws,
                stream,
            )?,
        }
        profile::end(profile::MLP_DENSE, t_dense, ctx.gpu, stream);
        // 2026-09-25: One all-reduce covers both partial sums, the EP-split routed experts and
        // the TP-split shared expert: `forward_moe` adds them together before returning.
        if self.mlp_cfg.needs_all_reduce() {
            self.reduce_probe(profile::REDUCE_MLP_BAR, "mlp", ctx, stream);
            let t = profile::start_hot();
            self.reduce_partial(out, rows, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_MLP_ENQ, t);
            // 2026-09-25: This span times only the `synchronize` in `profile::end`.
            let t = profile::start_hot();
            profile::end(profile::REDUCE_MLP, t, ctx.gpu, stream);
        }
        Ok(())
    }

    /// 2026-09-25: `dst += src` over `n` BF16 elements (`bf16_add_inplace`); errors when the
    /// target lacks the kernel.
    fn add_inplace(
        &self,
        gpu: &dyn GpuBackend,
        dst: DevicePtr,
        src: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        if self.add_k.0 == 0 {
            bail!(
                "GLM layer {}: bf16_add_inplace is not loaded on this target; the MTP layer's \
                 plain residual path needs it",
                self.layer_idx
            );
        }
        KernelLaunch::new(gpu, self.add_k)
            .grid([(n as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_i32(n as i32)
            .launch(stream)
    }

    /// 2026-09-25: One token through a block with no hyper-connection (`mhc: None`, the MTP
    /// block): `x += mixer(norm(x))`, then `x += mlp(norm(x))`, each all-reduce before its add.
    #[allow(clippy::too_many_arguments)]
    fn forward_one_plain(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
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
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();

        self.norm(gpu, hidden, self.input_norm, normed, 1, stream)?;
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
            self.reduce_partial(attn_out, 1, ctx, stream)?;
        }
        self.add_inplace(gpu, hidden, attn_out, h, stream)?;

        self.norm(gpu, hidden, self.post_attn_norm, normed, 1, stream)?;
        self.mlp_forward(normed, ffn_out, 1, 1, ctx, stream)?;
        self.add_inplace(gpu, hidden, ffn_out, h, stream)
    }

    /// 2026-09-25: This KDA layer's state as the SSM pool's `SsmLayerState`. Errors when the
    /// state is another type, or when its recurrent state is FP16: `h_is_f16` is set, or an FP32
    /// staging blob is attached, which only an FP16-sized pool has.
    pub(super) fn kda_state<'a>(
        &self,
        state: &'a mut dyn LayerState,
    ) -> Result<&'a mut SsmLayerState> {
        let st = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a KDA mixer was handed state that is not an SsmLayerState",
                    self.layer_idx
                )
            })?;
        if st.h_is_f16 || st.h_prefill_stage.is_some() {
            bail!(
                "GLM layer {}: KDA recurrent state is FP32-only; --ssm-h-dtype f16 narrowed it",
                self.layer_idx
            );
        }
        Ok(st)
    }

    /// 2026-09-25: This DSA layer's `Glm5NextDsaState`; errors on any other state type.
    pub(super) fn dsa_state<'a>(
        &self,
        state: &'a mut dyn LayerState,
    ) -> Result<&'a mut Glm5NextDsaState> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a DSA mixer was handed state that is not a Glm5NextDsaState",
                    self.layer_idx
                )
            })
    }
}
