// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The KDA recurrent path: single-token `decode` and K-row `decode_k`.
//! 2026-10-01: Plus `decode_n_seqs`, one token for each of N sequences.
//! 2026-10-02: Plus `decode_verify_n_seqs`, `ks[i]` verify rows for each of N sequences.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - The recurrent state advances one row at a time, in row order (`stateful_row`).
//! - The opt-in token loop (`stateful_rows`) advances it over all rows in two launches, rows still
//!   in order, and only for a `decode_k` that takes no snapshots.
//! - `METRALE_GLM_KDA_PREFETCH=1` changes only which recurrent kernel the token loop launches,
//!   never whether the loop runs.

use super::*;

impl Glm5NextKdaLayer {
    /// 2026-09-25: The stateful half of one KDA token: the conv window update (SiLU and L2 fused),
    /// then one recurrent step, both on row `row` of the workspace, updating `state` in place.
    ///
    /// The state after row `t + 1` depends on the state after row `t`, so [`Self::decode_k`] calls
    /// this once per row, in order, and batches only the projections around it. The chunked
    /// [`Self::prefill`] computes the same recurrence in a different order, so its results are not
    /// guaranteed to match this path bit for bit.
    ///
    /// `q`/`k` reach `kda_recurrent` already L2-normalised by the conv, which is the input that
    /// kernel expects; normalising them again would change the bf16-rounded values.
    fn stateful_row(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let qkv = c.qkv_dim();
        let cd = c.conv_dim();

        ops::conv1d_update_l2norm(
            gpu,
            self.kernels.conv_decode,
            state.conv,
            ws.qkv_proj.offset(row * cd * 2),
            &self.weights.conv,
            ws.conv_out.offset(row * cd * 2),
            c.conv_dim() as u32,
            c.conv_kernel as u32,
            1,
            c.qk_channels() as u32,
            c.head_dim as u32,
            c.l2_eps,
            stream,
        )?;

        let d = c.head_dim;
        // 2026-09-25: 1R+1W when the target has the shared-memory kernel. Each block owns `vpb`
        // V columns and grid.y covers the rest; the request is `3 * d` floats plus
        // `vpb * (d + 1)` floats of column scratch.
        let vpb = KDA_V_PER_BLOCK.min(d);
        let smem_smem = (3 * d + vpb * (d + 1)) * 4;
        if self.kernels.recurrent_smem.0 != 0
            && d.is_multiple_of(vpb)
            && smem_smem <= KDA_SMEM_BUDGET
            && !kda_no_smem()
        {
            KernelLaunch::new(gpu, self.kernels.recurrent_smem)
                .grid([c.heads as u32, (d / vpb) as u32, 1])
                .block([vpb as u32, 1, 1])
                .shared_mem(smem_smem as u32)
                .arg_ptr(ws.conv_out.offset(row * cd * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 4))
                .arg_ptr(ws.gate.offset(row * qkv * 4))
                .arg_ptr(ws.beta.offset(row * c.heads * 4))
                .arg_ptr(state.recurrent)
                .arg_ptr(ws.core.offset(row * qkv * 4))
                .arg_u32(c.heads as u32)
                .arg_u32(d as u32)
                .arg_f32(1.0 / (d as f32).sqrt())
                .arg_u32(vpb as u32)
                .launch(stream)?;
        } else {
            KernelLaunch::new(gpu, self.kernels.recurrent)
                .grid([c.heads as u32, 1, 1])
                .block([BLOCK.min(d as u32), 1, 1])
                .shared_mem((3 * d * 4) as u32)
                .arg_ptr(ws.conv_out.offset(row * cd * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 4))
                .arg_ptr(ws.gate.offset(row * qkv * 4))
                .arg_ptr(ws.beta.offset(row * c.heads * 4))
                .arg_ptr(state.recurrent)
                .arg_ptr(ws.core.offset(row * qkv * 4))
                .arg_u32(c.heads as u32)
                .arg_u32(d as u32)
                .arg_f32(1.0 / (d as f32).sqrt())
                .launch(stream)?;
        }

        Ok(())
    }

    /// 2026-10-01: The opt-in token loop (`METRALE_GLM_KDA_TOKEN_LOOP=1`): rows `0..k` of
    /// [`Self::stateful_row`] in two launches instead of `2 * k`. Returns `Ok(false)`, having
    /// launched nothing, when the target lacks either kernel or the shared-memory recurrent
    /// kernel would not be chosen for a single row; the caller then walks the rows.
    ///
    /// The conv over every row runs first, then the recurrence over every row. That equals the
    /// interleaved walk because conv row `t + 1` reads only `qkv_proj` row `t + 1` and the conv
    /// state, neither of which the recurrence touches, and recurrent row `t` reads only conv row
    /// `t`. Each kernel computes its rows in order with the per-row kernel's arithmetic
    /// (`causal_conv1d_update_l2norm_rows`, `kda_recurrent_prefill_bf16_smem`), so outputs and
    /// both final states match the walk bit for bit; `kda_tokenloop_microtest` checks that.
    /// Takes no snapshots: it never materialises the state after an interior row.
    ///
    /// 2026-10-01: Under `METRALE_GLM_KDA_PREFETCH=1` the recurrence launches
    /// `kda_recurrent_prefill_bf16_pf` instead (same arguments, same per-element arithmetic, inputs
    /// prefetched two tokens ahead); the microtest checks it against the same walk.
    fn stateful_rows(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<bool> {
        let c = &self.cfg;
        let qkv = c.qkv_dim();
        let cd = c.conv_dim();
        let d = c.head_dim;
        let vpb = KDA_V_PER_BLOCK.min(d);
        let smem_smem = (3 * d + vpb * (d + 1)) * 4;
        // 2026-10-01: Only where `stateful_row` would launch the shared-memory recurrent kernel,
        // the kernel whose arithmetic the row kernel copies. The conv-rows window holds 8 taps.
        if self.kernels.conv_rows.0 == 0
            || self.kernels.recurrent_rows.0 == 0
            || self.kernels.recurrent_smem.0 == 0
            || !d.is_multiple_of(vpb)
            || smem_smem > KDA_SMEM_BUDGET
            || kda_no_smem()
            || c.conv_kernel > 8
        {
            return Ok(false);
        }

        KernelLaunch::new(gpu, self.kernels.conv_rows)
            .grid([div_ceil(cd as u32, 256), 1, 1])
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

        // 2026-10-01: `METRALE_GLM_KDA_PREFETCH=1` swaps in the prefetching twin, same arguments,
        // when the target has it and the geometry meets its contract; otherwise the row kernel,
        // with one warning per process.
        let (rows_kernel, rows_smem) = match self.prefetch_rows(d, vpb) {
            Some(smem) => (self.kernels.recurrent_pf, smem),
            None => (self.kernels.recurrent_rows, smem_smem),
        };
        KernelLaunch::new(gpu, rows_kernel)
            .grid([c.heads as u32, (d / vpb) as u32, 1])
            .block([vpb as u32, 1, 1])
            .shared_mem(rows_smem as u32)
            .arg_ptr(ws.conv_out)
            .arg_ptr(ws.conv_out.offset(qkv * 2))
            .arg_ptr(ws.conv_out.offset(qkv * 4))
            .arg_ptr(ws.gate)
            .arg_ptr(ws.beta)
            .arg_ptr(state.recurrent)
            .arg_ptr(ws.core)
            .arg_u32(c.heads as u32)
            .arg_u32(d as u32)
            .arg_f32(1.0 / (d as f32).sqrt())
            .arg_u32(vpb as u32)
            .arg_u32(k as u32)
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_u32(c.heads as u32)
            .arg_u32(qkv as u32)
            .launch(stream)?;
        Ok(true)
    }

    /// 2026-10-01: Shared memory for `kda_recurrent_prefill_bf16_pf` when `stateful_rows` should
    /// launch it: `METRALE_GLM_KDA_PREFETCH=1`, the handle resolved, and the geometry inside its
    /// launcher contract (`kda_pf_smem`). `None` keeps `kda_recurrent_prefill_bf16_smem`; with the
    /// lever on, that is warned about once.
    fn prefetch_rows(&self, d: usize, vpb: usize) -> Option<usize> {
        if !kda_prefetch() {
            return None;
        }
        if self.kernels.recurrent_pf.0 == 0 {
            kda_prefetch_fallback("the target has no kda_recurrent_prefill_bf16_pf");
            return None;
        }
        let smem = kda_pf_smem(d, vpb);
        if smem.is_none() {
            kda_prefetch_fallback("head_dim / V-per-block outside the kernel's launcher contract");
        }
        smem
    }

    /// 2026-09-25: Single-token decode, carrying both states. The result lands in `ws.final_out`;
    /// `state` is updated in place.
    pub fn decode(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        self.front_end(gpu, hidden, 1, ws, stream)?;
        self.stateful_row(gpu, 0, state, ws, stream)?;
        self.back_end(gpu, 1, ws, stream)
    }

    /// 2026-09-25: `k` tokens of one sequence: the projections batched, the recurrence one row at a
    /// time. Used for the speculative verify, and for prefill sub-chunks that do not take the
    /// chunked arm (`glm5next_layer/steps/mixer.rs`).
    ///
    /// `front_end` and `back_end` run once over all `k` rows. For `2 <= k <=
    /// DENSE_GEMV_BATCHM_MAX_M` on a target with `dense_gemv_bf16_batchm`, the output matches `k`
    /// serial [`Self::decode`] calls bit for bit: the batched GEMV gives each row the bits of the
    /// M = 1 GEMV, the pack, gate, sigmoid and `o_norm` kernels treat rows independently, and
    /// `stateful_row` walks the state one row at a time as `decode` does.
    ///
    /// `snapshots[t]`, `(h_dst, conv_dst)`, receives the state after row `t`; rows past
    /// `snapshots.len()` take no snapshot. Refuses `k == 0` or `k` above the workspace.
    pub fn decode_k(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        snapshots: &[(DevicePtr, DevicePtr)],
        stream: u64,
    ) -> Result<()> {
        if k == 0 || k > ws.max_tokens {
            bail!(
                "KDA decode_k of {k} tokens does not fit a workspace built for {}",
                ws.max_tokens
            );
        }
        use crate::glm5next_layer::profile;
        let c = &self.cfg;
        let (h_bytes, conv_bytes) = (c.recurrent_state_elems() * 4, c.conv_state_elems() * 4);
        // 2026-09-25: Three profile buckets (front, recurrence, back), each closed where its span
        // ends.
        let t_front = profile::start();
        self.front_end(gpu, hidden, k, ws, stream)?;
        profile::end(profile::KDA_FRONT, t_front, gpu, stream);
        let t_recur = profile::start();
        // 2026-10-01: Opt-in (`METRALE_GLM_KDA_TOKEN_LOOP=1`): two launches for all rows when no
        // snapshot is asked for; otherwise, or when the target lacks the row kernels, the walk.
        let looped = snapshots.is_empty()
            && k > 1
            && kda_token_loop()
            && self.stateful_rows(gpu, k, state, ws, stream)?;
        if !looped {
            for row in 0..k {
                self.stateful_row(gpu, row, state, ws, stream)?;
                if let Some((h_dst, conv_dst)) = snapshots.get(row) {
                    gpu.copy_d2d_async(state.recurrent, *h_dst, h_bytes, stream)?;
                    gpu.copy_d2d_async(state.conv, *conv_dst, conv_bytes, stream)?;
                }
            }
        }
        profile::end(profile::KDA_RECUR, t_recur, gpu, stream);
        let t_back = profile::start();
        let r = self.back_end(gpu, k, ws, stream);
        profile::end(profile::KDA_BACK, t_back, gpu, stream);
        r
    }

    /// 2026-10-01: One token for each of `n` sequences: `front_end` and `back_end` once over all
    /// `n` rows (the KDA projection weights read once per step instead of once per sequence),
    /// and `stateful_row(i, &states[i])` for row `i`. The rows' recurrences are independent, so
    /// the order of the loop cannot matter; the token-axis ordering argument on `stateful_row`
    /// does not apply across sequences. Row `i` of `ws.final_out` holds sequence `i`'s output.
    ///
    /// For `2 <= n <= DENSE_GEMV_BATCHM_MAX_M` on a target with `dense_gemv_bf16_batchm` the
    /// output and every state match `n` separate [`Self::decode`] calls bit for bit, by the
    /// argument on [`Self::decode_k`]. Takes no snapshots: plain decode is never rolled back.
    /// Refuses `n == 0`, `n` above the workspace, or `states.len() != n`. Ported from rsafier's
    /// Atlas `Glm5NextKdaLayer::decode_n_seqs` (Atlas acf792e28).
    pub fn decode_n_seqs(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        n: usize,
        states: &[KdaSeqState],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        if n == 0 || n > ws.max_tokens {
            bail!(
                "KDA decode_n_seqs of {n} sequences does not fit a workspace built for {}",
                ws.max_tokens
            );
        }
        // 2026-10-01: A short slice means the caller and this layer disagree about the batch;
        // advancing another sequence's state would be a silent, compounding wrong answer.
        if states.len() != n {
            bail!("KDA decode_n_seqs: {n} sequences but {} states", states.len());
        }
        use crate::glm5next_layer::profile;
        let t_front = profile::start();
        self.front_end(gpu, hidden, n, ws, stream)?;
        profile::end(profile::KDA_FRONT, t_front, gpu, stream);
        let t_recur = profile::start();
        for (row, state) in states.iter().enumerate() {
            self.stateful_row(gpu, row, state, ws, stream)?;
        }
        profile::end(profile::KDA_RECUR, t_recur, gpu, stream);
        let t_back = profile::start();
        let r = self.back_end(gpu, n, ws, stream);
        profile::end(profile::KDA_BACK, t_back, gpu, stream);
        r
    }

    /// 2026-10-02: The batched MTP verify of `ks.len()` sequences (`METRALE_GLM_BATCHED_VERIFY`):
    /// `front_end` and `back_end` once over all `R = Σ ks` rows, and for sequence `i` the
    /// [`Self::decode_k`] walk over its rows `off_i..off_i + ks[i]` (sequence-major, `off_i` the
    /// prefix sum of `ks`) on `states[i]`, writing the state after its row `t` to
    /// `snapshots[i][t]` for `t < snapshots[i].len()`. Each sequence's walk is the one
    /// `decode_k` runs for it alone, and the sequences' recurrences are independent; only the
    /// projection launches change width, so up to `DENSE_GEMV_BATCHM_MAX_M` rows the outputs match
    /// per-sequence `decode_k` calls by the argument on [`Self::decode_k`], and above it the
    /// projections take another kernel (numerics-changing). Refuses an empty batch, a zero
    /// `ks[i]`, `R` above the workspace, or a slice length that disagrees with `ks`.
    pub fn decode_verify_n_seqs(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        ks: &[usize],
        states: &[KdaSeqState],
        snapshots: &[Vec<(DevicePtr, DevicePtr)>],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let r: usize = ks.iter().sum();
        if ks.is_empty() || ks.contains(&0) || r > ws.max_tokens {
            bail!(
                "KDA decode_verify_n_seqs of ks={ks:?} does not fit a workspace built for {}",
                ws.max_tokens
            );
        }
        if states.len() != ks.len() || snapshots.len() != ks.len() {
            bail!(
                "KDA decode_verify_n_seqs: {} sequences but {} states and {} snapshot lists",
                ks.len(),
                states.len(),
                snapshots.len()
            );
        }
        use crate::glm5next_layer::profile;
        let c = &self.cfg;
        let (h_bytes, conv_bytes) = (c.recurrent_state_elems() * 4, c.conv_state_elems() * 4);
        let t_front = profile::start();
        self.front_end(gpu, hidden, r, ws, stream)?;
        profile::end(profile::KDA_FRONT, t_front, gpu, stream);
        let t_recur = profile::start();
        let mut base = 0usize;
        for ((&k, state), snaps) in ks.iter().zip(states).zip(snapshots) {
            for t in 0..k {
                self.stateful_row(gpu, base + t, state, ws, stream)?;
                if let Some((h_dst, conv_dst)) = snaps.get(t) {
                    gpu.copy_d2d_async(state.recurrent, *h_dst, h_bytes, stream)?;
                    gpu.copy_d2d_async(state.conv, *conv_dst, conv_bytes, stream)?;
                }
            }
            base += k;
        }
        profile::end(profile::KDA_RECUR, t_recur, gpu, stream);
        let t_back = profile::start();
        let res = self.back_end(gpu, r, ws, stream);
        profile::end(profile::KDA_BACK, t_back, gpu, stream);
        res
    }
}
