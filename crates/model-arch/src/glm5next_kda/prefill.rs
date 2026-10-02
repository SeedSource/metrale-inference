// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The KDA chunked prefill (`prefill`, `prefill_with_pad_fill`).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - A call covers `1..=ws.max_tokens()` tokens; any other count is refused before a launch.
//!   2026-10-01: also any count above `ws.chunk_tokens()`, which equals `max_tokens()` unless
//!   the workspace came from `Glm5NextKdaWorkspace::new_split`.

use super::*;

impl Glm5NextKdaLayer {
    /// 2026-09-25: Chunked prefill over `t` tokens, starting from the carried state.
    pub fn prefill(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        self.prefill_with_pad_fill(gpu, hidden, t, state, ws, 0.0, stream)
    }

    /// 2026-09-25: [`Self::prefill`] with the padded q/k/v tails filled with `pad_fill`.
    ///
    /// [`Self::prefill`] passes zero. The layer harness passes other values to show that
    /// `kda_chunk_*` ignores positions at or past `t` in-kernel, which a zeroed tail cannot show.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_with_pad_fill(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        pad_fill: f32,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, cd, hd) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        if t == 0 || t > ws.max_tokens {
            bail!(
                "prefill of {t} tokens does not fit a workspace built for {}",
                ws.max_tokens
            );
        }
        // 2026-10-01: The chunked-scan buffers may be narrower than the rest (`new_split`).
        if t > ws.chunk_tokens {
            bail!(
                "chunked KDA prefill of {t} tokens does not fit the {}-token chunk buffers \
                 (METRALE_GLM_PREFILL_FULLWIDTH_GEMM sizes them only for \
                 METRALE_GLM_KDA_CHUNK_PREFILL=1)",
                ws.chunk_tokens
            );
        }
        let nchunks = t.div_ceil(c.chunk);
        let tp = nchunks * c.chunk;
        // 2026-09-25: The same three profile buckets as `decode_k`.
        let t_front = crate::glm5next_layer::profile::start();
        self.front_end(gpu, hidden, t, ws, stream)?;
        crate::glm5next_layer::profile::end(
            crate::glm5next_layer::profile::KDA_FRONT,
            t_front,
            gpu,
            stream,
        );
        let t_recur = crate::glm5next_layer::profile::start();

        // 2026-09-25: The prefill conv is conv + SiLU only; L2 is a separate launch over q|k.
        KernelLaunch::new(gpu, self.kernels.conv_prefill)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state.conv)
            .arg_ptr(ws.qkv_proj)
            .arg_ptr(self.weights.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(ws.conv_out)
            .arg_u32(cd as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(t as u32)
            .arg_u32(cd as u32)
            .arg_u32(cd as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.kernels.l2)
            .grid([(c.qk_channels() / hd) as u32, t as u32, 1])
            .block([hd as u32, 1, 1])
            .arg_ptr(ws.conv_out)
            .arg_u32(hd as u32)
            .arg_f32(c.l2_eps)
            .arg_u32(cd as u32)
            .launch(stream)?;

        // 2026-09-25: Zero the gate and beta pad tails, so only the q/k/v tails carry `pad_fill`.
        for (buf, real, padded) in [
            (ws.gate, t * qkv, tp * qkv),
            (ws.beta, t * c.heads, tp * c.heads),
        ] {
            if padded == real {
                continue;
            }
            KernelLaunch::new(gpu, self.kernels.fill)
                .grid([div_ceil((padded - real) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(buf.offset(real * 4))
                .arg_u32((padded - real) as u32)
                .arg_f32(0.0)
                .launch(stream)?;
        }
        for p in [ws.q_f32, ws.k_f32, ws.v_f32] {
            KernelLaunch::new(gpu, self.kernels.fill)
                .grid([div_ceil((tp * qkv) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(p)
                .arg_u32((tp * qkv) as u32)
                .arg_f32(pad_fill)
                .launch(stream)?;
        }
        KernelLaunch::new(gpu, self.kernels.split_widen)
            .grid([div_ceil(qkv as u32, 256), t as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(ws.conv_out)
            .arg_ptr(ws.q_f32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.v_f32)
            .arg_u32(t as u32)
            .arg_u32(qkv as u32)
            .launch(stream)?;

        KernelLaunch::new(gpu, self.kernels.chunk_prepare)
            .grid([nchunks as u32, c.heads as u32, 1])
            .block([BLOCK, 1, 1])
            .shared_mem(c.smem_prepare() as u32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.v_f32)
            .arg_ptr(ws.gate)
            .arg_ptr(ws.beta)
            .arg_ptr(ws.chunk_gc)
            .arg_ptr(ws.chunk_u)
            .arg_ptr(ws.chunk_w)
            .arg_u32(c.heads as u32)
            .arg_u32(hd as u32)
            .arg_u32(c.chunk as u32)
            .arg_u32(t as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.kernels.chunk_scan)
            .grid([c.heads as u32, 1, 1])
            .block([BLOCK, 1, 1])
            .shared_mem(c.smem_scan() as u32)
            .arg_ptr(ws.q_f32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.chunk_gc)
            .arg_ptr(ws.chunk_u)
            .arg_ptr(ws.chunk_w)
            .arg_ptr(state.recurrent)
            .arg_ptr(ws.core)
            .arg_u32(c.heads as u32)
            .arg_u32(hd as u32)
            .arg_u32(c.chunk as u32)
            .arg_u32(nchunks as u32)
            .arg_u32(t as u32)
            .arg_f32(1.0 / (hd as f32).sqrt())
            .launch(stream)?;
        crate::glm5next_layer::profile::end(
            crate::glm5next_layer::profile::KDA_RECUR,
            t_recur,
            gpu,
            stream,
        );

        let t_back = crate::glm5next_layer::profile::start();
        let r = self.back_end(gpu, t, ws, stream);
        crate::glm5next_layer::profile::end(
            crate::glm5next_layer::profile::KDA_BACK,
            t_back,
            gpu,
            stream,
        );
        r
    }
}
