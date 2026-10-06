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
//!   snapshots, so the fallback is the default path; 2026-10-06: with an in-pass capture it is
//!   `decode_k_capture`, which captures at any row.
//! - 2026-10-06: Without an in-pass capture the launches are the ones this file has always made.
//!   With one, outputs and both final states are unchanged bit for bit: the capture only splits
//!   `kda_tc_scan` at a 16-row chunk boundary (it keeps the state FP32 in registers, reads it once
//!   and writes it once, so a launch boundary changes no value) and adds a conv-tail launch on
//!   the snapshot slot. `kda_tc_capture_microtest` checks both claims.
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
//! 2026-10-06: An in-pass capture (`METRALE_GLM_SSM_INPASS_CAPTURE=1`, [`InpassSplit`]) after
//! `row` rows of a `k`-row call ([`tc_capture_plan`]):
//!
//! ```text
//! row < k, row % 16 == 0  conv_rows (all rows); copy conv state -> conv_dst;
//!                         conv_state_tail(row) on conv_dst; conv_state_tail(k) on the state;
//!                         prepare; scan chunks [0, row/16); copy recurrent state -> h_dst;
//!                         scan chunks [row/16, n)
//! row == k                the uncaptured launches, then both copies
//! any other row           the uncaptured launches, no copy (the engine then drops the slot)
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
use metrale_model_layers::layer::InpassSplit;

/// 2026-10-06: How the tensor-core prefill takes an in-pass capture after `row` rows of a
/// `k`-row call ([`tc_capture_plan`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TcCapture {
    /// 2026-10-06: No copy: no capture asked for, or `row` is inside a 16-row chunk.
    NoCapture,
    /// 2026-10-06: `row = c0 * KDA_TC_C < k`: the scan runs as chunks `[0, c0)`, then the copy,
    /// then chunks `[c0, n)`.
    Split { c0: usize },
    /// 2026-10-06: `row == k`: the copies follow the uncaptured launches.
    After,
}

/// 2026-10-06: The capture a `k`-row call can take after `row` rows: `After` at `row == k`,
/// `Split` at a chunk boundary strictly inside the call, otherwise `NoCapture`.
fn tc_capture_plan(row: usize, k: usize) -> TcCapture {
    if k > 0 && row == k {
        TcCapture::After
    } else if row > 0 && row < k && row.is_multiple_of(KDA_TC_C) {
        TcCapture::Split { c0: row / KDA_TC_C }
    } else {
        TcCapture::NoCapture
    }
}

/// 2026-10-06: Byte offsets of a `kda_tc_scan` launch that starts at chunk `c0`: into each record
/// buffer (`q_f32`, `k_f32`, `v_f32`, `chunk_gc`; records at `rec = c * heads + h`,
/// `KDA_TC_REC_BYTES` each) and into `ws.core` (FP32 rows of `qkv`). All zero at `c0 = 0`.
fn tc_scan_offsets(c0: usize, heads: usize, qkv: usize) -> ([usize; 4], usize) {
    (
        KDA_TC_REC_BYTES.map(|b| c0 * heads * b),
        c0 * KDA_TC_C * qkv * 4,
    )
}

impl Glm5NextKdaLayer {
    /// 2026-10-01: `k` prefill tokens of one sequence through the tensor-core chunked recurrence,
    /// starting from the carried state; the projections around it are `decode_k`'s
    /// (`front_end`, `back_end`). Not bit-identical to `decode_k`: the recurrence sums in chunk
    /// order and its MMAs take BF16 operands (kda_chunk_tc.cu). Takes no snapshots.
    ///
    /// Falls back to `decode_k` (warning once) when the target lacks a kernel or the geometry or
    /// workspace is outside what the kernels take ([`chunked_tc_refusal`]).
    ///
    /// 2026-10-06: `capture` is an in-pass snapshot (`METRALE_GLM_SSM_INPASS_CAPTURE=1`): the
    /// states after `capture.row` rows are copied to its slot when that row is a 16-row chunk
    /// boundary or the end of the call (module doc), and through `decode_k_capture` on the
    /// fallback. Returns whether the copies were issued; the caller counts the layer as captured
    /// only then. `None` makes the launches this method has always made, and returns false.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunked_tc(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        capture: Option<&InpassSplit>,
        stream: u64,
    ) -> Result<bool> {
        self.prefill_chunked_tc_inner(gpu, hidden, k, state, ws, kda_front_fuse(), capture, stream)
    }

    /// 2026-10-07: [`Self::prefill_chunked_tc`] with `METRALE_GLM_KDA_FRONT_FUSE` passed in as
    /// `front_fuse` instead of read from the environment, so `kda_front_fuse_microtest` runs
    /// both arms in one process, and with no in-pass capture. `front_fuse` true still falls back
    /// to the unfused launches (warning once) when [`front_fuse_refusal`] names a reason.
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
        self.prefill_chunked_tc_inner(gpu, hidden, k, state, ws, front_fuse, None, stream)
            .map(|_| ())
    }

    /// 2026-10-10: The body of [`Self::prefill_chunked_tc`] and [`Self::prefill_chunked_tc_arm`]:
    /// the front-fuse switch and the in-pass capture together. A capture that splits the scan at
    /// a chunk boundary ([`TcCapture::Split`]) needs the unfused front (its conv-state tail
    /// reads `ws.qkv_proj`, which the fused front never writes), so it runs unfused, warning
    /// once; a capture at the end of the call (or none) is compatible with the fused front.
    #[allow(clippy::too_many_arguments)]
    fn prefill_chunked_tc_inner(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        front_fuse: bool,
        capture: Option<&InpassSplit>,
        stream: u64,
    ) -> Result<bool> {
        if k == 0 || k > ws.max_tokens {
            bail!(
                "KDA chunked-TC prefill of {k} tokens does not fit a workspace built for {}",
                ws.max_tokens
            );
        }
        if let Some(why) = self.chunked_tc_refusal_ws(k, ws) {
            kda_chunked_tc_fallback(why);
            if let Some(cap) = capture {
                self.decode_k_capture(gpu, hidden, k, state, ws, cap, stream)?;
                return Ok(true);
            }
            self.decode_k(gpu, hidden, k, state, ws, &[], stream)?;
            return Ok(false);
        }
        // 2026-10-07: METRALE_GLM_KDA_FRONT_FUSE=1: the front end skips the pack, the gate and
        // the beta sigmoid, and the recurrence computes them inside its prepare
        // (prefill_tc_fuse.rs); byte-identical.
        let split = matches!(
            capture.map(|cap| tc_capture_plan(cap.row, k)),
            Some(TcCapture::Split { .. })
        );
        if front_fuse && split {
            kda_front_fuse_split_capture();
        }
        let fuse = front_fuse
            && !split
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
        let captured = if fuse {
            self.stateful_chunked_tc_fused(gpu, k, state, ws, stream)?;
            // 2026-10-10: `fuse` implies the plan is not `Split`: the only capture left is the
            // end-of-call copy, the same two copies `stateful_chunked_tc` issues after its scan.
            match capture.map(|cap| (cap, tc_capture_plan(cap.row, k))) {
                Some((cap, TcCapture::After)) => {
                    gpu.copy_d2d_async(state.recurrent, cap.h_dst, cap.h_bytes, stream)?;
                    gpu.copy_d2d_async(state.conv, cap.conv_dst, cap.conv_bytes, stream)?;
                    true
                }
                _ => false,
            }
        } else {
            self.stateful_chunked_tc(gpu, k, state, ws, capture, stream)?
        };
        profile::end(profile::KDA_RECUR, t_recur, gpu, stream);
        let t_back = profile::start();
        let r = self.back_end(gpu, k, ws, stream);
        profile::end(profile::KDA_BACK, t_back, gpu, stream);
        r.map(|()| captured)
    }

    /// 2026-10-06: Microtest arm (`kda_tc_capture_microtest`): the four kda_chunk_tc launches
    /// over workspace rows `0..k` as a previous call's front end left them, with no front or
    /// back end, advancing `state` in place. Refuses where [`Self::prefill_chunked_tc`] would
    /// take `decode_k`, so a test cannot pass on the fallback.
    pub fn chunked_tc_recurrence_arm(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        if k == 0 || k > ws.max_tokens {
            bail!("KDA chunked-TC arm of {k} rows: outside 1..={}", ws.max_tokens);
        }
        if let Some(why) = self.chunked_tc_refusal_ws(k, ws) {
            bail!("KDA chunked-TC arm of {k} rows refused: {why}");
        }
        self.stateful_chunked_tc(gpu, k, state, ws, None, stream)
            .map(|_| ())
    }

    /// 2026-10-06: [`chunked_tc_refusal`] for `k` rows of this layer on `ws`.
    fn chunked_tc_refusal_ws(&self, k: usize, ws: &Glm5NextKdaWorkspace) -> Option<&'static str> {
        let kernels = self.kernels.has_chunked_tc();
        // 2026-10-04: the borrowed q/k/v_f32 + chunk_* buffers hold `chunk_tokens` rows (padded),
        // not `t_pad`: under METRALE_GLM_PREFILL_FULLWIDTH_GEMM the loader sized them to
        // verify_k, so checking t_pad let an 8192-row prefill write the chunk records past their
        // end.
        let chunk_pad = ws.chunk_tokens.div_ceil(self.cfg.chunk) * self.cfg.chunk;
        chunked_tc_refusal(&self.cfg, k, chunk_pad, kernels)
    }

    /// 2026-10-01: The four kda_chunk_tc launches over rows `0..k` of the workspace: conv state
    /// and recurrent state in `state` advance in place, the core output lands in `ws.core`. The
    /// caller has checked [`chunked_tc_refusal`].
    ///
    /// 2026-10-06: With `capture`, the module doc's capture launches ([`tc_capture_plan`]); a
    /// `Split` also needs the slot's `conv_bytes` to hold the whole conv state, which the tail
    /// launch writes in place there. Returns whether the copies were issued.
    fn stateful_chunked_tc(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        capture: Option<&InpassSplit>,
        stream: u64,
    ) -> Result<bool> {
        let c = &self.cfg;
        let (qkv, cd, d) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        let nchunks = k.div_ceil(KDA_TC_C);
        let plan = match capture.map(|cap| (cap, tc_capture_plan(cap.row, k))) {
            Some((cap, TcCapture::Split { .. })) if cap.conv_bytes < c.conv_state_elems() * 4 => {
                TcCapture::NoCapture
            }
            Some((_, p)) => p,
            None => TcCapture::NoCapture,
        };

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
        // 2026-10-06: A split capture's conv state: the incoming state (the launch above only
        // read it) copied to the slot and advanced there by `row` rows, the launch below's
        // arithmetic on the same inputs.
        if let (Some(cap), TcCapture::Split { .. }) = (capture, plan) {
            gpu.copy_d2d_async(state.conv, cap.conv_dst, cap.conv_bytes, stream)?;
            self.tc_conv_tail_launch(gpu, cap.conv_dst, cap.row, ws, stream)?;
        }
        self.tc_conv_tail_launch(gpu, state.conv, k, ws, stream)?;

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
        // 2026-10-06: The scan reads the recurrent state once and writes it once, FP32, so two
        // launches split at a chunk boundary carry it exactly; the copy between them is the
        // state after `row` rows.
        match (capture, plan) {
            (Some(cap), TcCapture::Split { c0 }) => {
                self.tc_scan_launch(gpu, (0, c0), cap.row, state, ws, stream)?;
                gpu.copy_d2d_async(state.recurrent, cap.h_dst, cap.h_bytes, stream)?;
                self.tc_scan_launch(gpu, (c0, nchunks), k - cap.row, state, ws, stream)?;
            }
            _ => self.tc_scan_launch(gpu, (0, nchunks), k, state, ws, stream)?,
        }
        if let (Some(cap), TcCapture::After) = (capture, plan) {
            gpu.copy_d2d_async(state.recurrent, cap.h_dst, cap.h_bytes, stream)?;
            gpu.copy_d2d_async(state.conv, cap.conv_dst, cap.conv_bytes, stream)?;
        }
        Ok(plan != TcCapture::NoCapture)
    }

    /// 2026-10-06: `kda_tc_conv_state_tail` on `conv` (a `[conv_dim, conv_kernel]` FP32 state)
    /// after `rows` rows of `ws.qkv_proj`. At `(state.conv, k)` these are the arguments
    /// `stateful_chunked_tc` has always passed.
    fn tc_conv_tail_launch(
        &self,
        gpu: &dyn GpuBackend,
        conv: DevicePtr,
        rows: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let cd = c.conv_dim();
        KernelLaunch::new(gpu, self.kernels.tc_conv_tail)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(conv)
            .arg_ptr(ws.qkv_proj)
            .arg_u32(rows as u32)
            .arg_u32(cd as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(cd as u32)
            .launch(stream)?;
        Ok(())
    }

    /// 2026-10-06: `kda_tc_scan` over chunks `chunks.0..chunks.1` (`rows` rows from row
    /// `16 * chunks.0`), the record and output pointers advanced by [`tc_scan_offsets`]. At
    /// `(0, nchunks)` and `rows = k` these are the arguments `stateful_chunked_tc` has always
    /// passed (every offset is zero).
    fn tc_scan_launch(
        &self,
        gpu: &dyn GpuBackend,
        chunks: (usize, usize),
        rows: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, d) = (c.qkv_dim(), c.head_dim);
        let (rec, out) = tc_scan_offsets(chunks.0, c.heads, qkv);
        KernelLaunch::new(gpu, self.kernels.tc_scan)
            .grid([c.heads as u32, (d / (16 * KDA_TC_SCAN_WARPS)) as u32, 1])
            .block([(32 * KDA_TC_SCAN_WARPS) as u32, 1, 1])
            .shared_mem(KDA_TC_SCAN_SMEM as u32)
            .arg_ptr(ws.q_f32.offset(rec[0]))
            .arg_ptr(ws.k_f32.offset(rec[1]))
            .arg_ptr(ws.v_f32.offset(rec[2]))
            .arg_ptr(ws.chunk_gc.offset(rec[3]))
            .arg_ptr(state.recurrent)
            .arg_ptr(ws.core.offset(out))
            .arg_u32(c.heads as u32)
            .arg_u32(rows as u32)
            .arg_u32((chunks.1 - chunks.0) as u32)
            .arg_u32(qkv as u32)
            .launch(stream)?;
        Ok(())
    }

    /// 2026-10-07: `kda_tc_scan` over the chunk records of rows `0..k` (the fused front launches
    /// it too). 2026-10-10: now the uncaptured `tc_scan_launch` over every chunk, the same
    /// arguments as before.
    pub(super) fn launch_tc_scan(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        self.tc_scan_launch(gpu, (0, k.div_ceil(KDA_TC_C)), k, state, ws, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-06: Only a chunk boundary strictly inside the call splits; the end of the call
    /// copies after; every other row (and an empty call) takes no capture.
    #[test]
    fn capture_plan_splits_only_on_chunk_boundaries() {
        assert_eq!(tc_capture_plan(16, 2085), TcCapture::Split { c0: 1 });
        assert_eq!(tc_capture_plan(2048, 2085), TcCapture::Split { c0: 128 });
        assert_eq!(tc_capture_plan(8176, 8192), TcCapture::Split { c0: 511 });
        assert_eq!(tc_capture_plan(8192, 8192), TcCapture::After);
        assert_eq!(tc_capture_plan(2085, 2085), TcCapture::After, "end of a partial chunk");
        assert_eq!(tc_capture_plan(16, 16), TcCapture::After);
        for (row, k) in [(17, 2085), (2050, 2085), (8, 16), (0, 16), (0, 0), (32, 16)] {
            assert_eq!(tc_capture_plan(row, k), TcCapture::NoCapture, "row {row} of {k}");
        }
    }

    /// 2026-10-06: Launch B's pointers land on record `c0 * heads` of each buffer and on row
    /// `16 * c0` of `core`; launch A's (and the uncaptured launch's) are unmoved.
    #[test]
    fn scan_offsets_start_at_chunk_c0() {
        let (heads, qkv) = (32, 32 * 128);
        assert_eq!(tc_scan_offsets(0, heads, qkv), ([0; 4], 0));
        let (rec, out) = tc_scan_offsets(128, heads, qkv);
        assert_eq!(rec, [128 * 32 * 8192, 128 * 32 * 4096, 128 * 32 * 8192, 128 * 32 * 1024]);
        assert_eq!(out, 2048 * qkv * 4);
        // 2026-10-06: The u record is the kernel's `rec * (KDA_TC_C * KDA_TC_D)` floats.
        assert_eq!(KDA_TC_REC_BYTES[2], KDA_TC_C * KDA_TC_D * 4);
    }
}
