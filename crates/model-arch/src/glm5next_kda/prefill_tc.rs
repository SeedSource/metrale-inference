// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The opt-in tensor-core chunked KDA prefill (`prefill_chunked_tc`,
//! `METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1`).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - A call covers `1..=ws.max_tokens()` tokens; any other count is refused before a launch.
//! - Outputs land where `decode_k`'s do (`ws.core`, then `ws.final_out` through `back_end`), and
//!   both states end in the layouts the decode kernels read: the recurrent state FP32
//!   `[heads, head_dim, head_dim]` K-major, the conv state as `causal_conv1d_update_l2norm_rows`
//!   leaves it.
//! - When [`chunked_tc_refusal`] names a reason, the call is exactly `decode_k` with no
//!   snapshots, so the fallback is the default path.
//!
//! The four launches (kernels/gb10/common/kda_chunk_tc.cu):
//!
//! ```text
//! kda_tc_conv_rows        conv + SiLU + L2, every row at once (bits of the token loop's conv)
//! kda_tc_conv_state_tail  the conv state after the last row
//! kda_tc_prepare          per (16-row chunk, head), FP32: (I + L)^-1, u, W, Q', K', M, decay
//! kda_tc_scan             per (head, 16 V columns) warp: the state across chunks, BF16 MMAs
//! ```
//!
//! The chunk records go into workspace buffers that only the FP32 chunked scan
//! (`METRALE_GLM_KDA_CHUNK_PREFILL`) otherwise uses, so the lever allocates nothing: Q' + W in
//! `q_f32`, K'^T in `k_f32`, u in `v_f32`, M + decay in `chunk_gc` (`KDA_TC_REC_BYTES` per
//! record; [`chunked_tc_refusal`] checks they fit).
//!
//! 2026-10-07: Under `METRALE_GLM_KDA_FRONT_FUSE=1` the first three launches and the front
//! end's pack, gate and beta sigmoid become two (prefill_tc_fuse.rs); the scan is unchanged.

use super::prefill_tc_fuse::{front_fuse_refusal, kda_front_fuse, kda_front_fuse_fallback};
use super::*;

impl Glm5NextKdaLayer {
    /// 2026-10-01: `k` prefill tokens of one sequence through the tensor-core chunked recurrence,
    /// starting from the carried state; the projections around it are `decode_k`'s
    /// (`front_end`, `back_end`). Not bit-identical to `decode_k`: the recurrence sums in chunk
    /// order and its MMAs take BF16 operands (kda_chunk_tc.cu). Takes no snapshots.
    ///
    /// Falls back to `decode_k` (warning once) when the target lacks a kernel or the geometry or
    /// workspace is outside what the kernels take ([`chunked_tc_refusal`]).
    pub fn prefill_chunked_tc(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        self.prefill_chunked_tc_arm(gpu, hidden, k, state, ws, kda_front_fuse(), stream)
    }

    /// 2026-10-07: [`Self::prefill_chunked_tc`] with `METRALE_GLM_KDA_FRONT_FUSE` passed in as
    /// `front_fuse` instead of read from the environment, so `kda_front_fuse_microtest` runs
    /// both arms in one process. `front_fuse` true still falls back to the unfused launches
    /// (warning once) when [`front_fuse_refusal`] names a reason.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunked_tc_arm(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        front_fuse: bool,
        stream: u64,
    ) -> Result<()> {
        if k == 0 || k > ws.max_tokens {
            bail!(
                "KDA chunked-TC prefill of {k} tokens does not fit a workspace built for {}",
                ws.max_tokens
            );
        }
        let kernels = self.kernels.has_chunked_tc();
        // 2026-10-04: the borrowed q/k/v_f32 + chunk_* buffers hold `chunk_tokens` rows (padded), not
        // `t_pad`: under METRALE_GLM_PREFILL_FULLWIDTH_GEMM the loader sized them to verify_k, so checking
        // t_pad let an 8192-row prefill write the chunk records past their end.
        let chunk_pad = ws.chunk_tokens.div_ceil(self.cfg.chunk) * self.cfg.chunk;
        if let Some(why) = chunked_tc_refusal(&self.cfg, k, chunk_pad, kernels) {
            kda_chunked_tc_fallback(why);
            return self.decode_k(gpu, hidden, k, state, ws, &[], stream);
        }
        // 2026-10-07: METRALE_GLM_KDA_FRONT_FUSE=1: the front end skips the pack, the gate and
        // the beta sigmoid, and the recurrence computes them inside its prepare
        // (prefill_tc_fuse.rs); byte-identical.
        let fuse = front_fuse
            && match front_fuse_refusal(&self.cfg, self.kernels.has_front_fuse()) {
                Some(why) => {
                    kda_front_fuse_fallback(why);
                    false
                }
                None => true,
            };
        use crate::glm5next_layer::profile;
        // 2026-10-01: The same three profile buckets as `decode_k`.
        let t_front = profile::start();
        self.front_end_opts(gpu, hidden, k, ws, stream, !fuse, !fuse)?;
        profile::end(profile::KDA_FRONT, t_front, gpu, stream);
        let t_recur = profile::start();
        if fuse {
            self.stateful_chunked_tc_fused(gpu, k, state, ws, stream)?;
        } else {
            self.stateful_chunked_tc(gpu, k, state, ws, stream)?;
        }
        profile::end(profile::KDA_RECUR, t_recur, gpu, stream);
        let t_back = profile::start();
        let r = self.back_end(gpu, k, ws, stream);
        profile::end(profile::KDA_BACK, t_back, gpu, stream);
        r
    }

    /// 2026-10-01: The four kda_chunk_tc launches over rows `0..k` of the workspace: conv state
    /// and recurrent state in `state` advance in place, the core output lands in `ws.core`. The
    /// caller has checked [`chunked_tc_refusal`].
    fn stateful_chunked_tc(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, cd, d) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        let nchunks = k.div_ceil(KDA_TC_C);

        // 2026-10-01: The token loop's conv (`stateful_rows`) with its rows spread over grid.y;
        // it reads the conv state, and the tail launch then writes it.
        KernelLaunch::new(gpu, self.kernels.tc_conv)
            .grid([
                div_ceil(cd as u32, 256),
                div_ceil(k as u32, KDA_TC_CONV_ROWS as u32),
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(state.conv)
            .arg_ptr(ws.qkv_proj)
            .arg_ptr(self.weights.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(ws.conv_out)
            .arg_u32(k as u32)
            .arg_u32(cd as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(c.qk_channels() as u32)
            .arg_u32(d as u32)
            .arg_f32(c.l2_eps)
            .arg_u32(cd as u32)
            .arg_u32(cd as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.kernels.tc_conv_tail)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state.conv)
            .arg_ptr(ws.qkv_proj)
            .arg_u32(k as u32)
            .arg_u32(cd as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(cd as u32)
            .launch(stream)?;

        // 2026-10-01: q, k and v are the conv output's three `qkv`-wide column groups.
        KernelLaunch::new(gpu, self.kernels.tc_prepare)
            .grid([nchunks as u32, c.heads as u32, 1])
            .block([KDA_TC_D as u32, 1, 1])
            .arg_ptr(ws.conv_out)
            .arg_ptr(ws.conv_out.offset(qkv * 2))
            .arg_ptr(ws.conv_out.offset(qkv * 4))
            .arg_ptr(ws.gate)
            .arg_ptr(ws.beta)
            .arg_ptr(ws.q_f32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.v_f32)
            .arg_ptr(ws.chunk_gc)
            .arg_u32(c.heads as u32)
            .arg_u32(k as u32)
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_u32(c.heads as u32)
            .arg_f32(1.0 / (d as f32).sqrt())
            .launch(stream)?;
        self.launch_tc_scan(gpu, k, state, ws, stream)
    }

    /// 2026-10-07: `kda_tc_scan` over the chunk records of rows `0..k` (moved out of
    /// `stateful_chunked_tc` unchanged; the fused front launches it too).
    pub(super) fn launch_tc_scan(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, d) = (c.qkv_dim(), c.head_dim);
        let nchunks = k.div_ceil(KDA_TC_C);
        KernelLaunch::new(gpu, self.kernels.tc_scan)
            .grid([c.heads as u32, (d / (16 * KDA_TC_SCAN_WARPS)) as u32, 1])
            .block([(32 * KDA_TC_SCAN_WARPS) as u32, 1, 1])
            .shared_mem(KDA_TC_SCAN_SMEM as u32)
            .arg_ptr(ws.q_f32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.v_f32)
            .arg_ptr(ws.chunk_gc)
            .arg_ptr(state.recurrent)
            .arg_ptr(ws.core)
            .arg_u32(c.heads as u32)
            .arg_u32(k as u32)
            .arg_u32(nchunks as u32)
            .arg_u32(qkv as u32)
            .launch(stream)?;
        Ok(())
    }
}
