// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The sequence-parallel staged prefill (`METRALE_GLM_PREFILL_SEQ_PARALLEL=1`):
//! `prefill_staged_run`'s two passes with the replicated row-local work split by rows across
//! the two tensor-parallel ranks (ownership: `glm5next_layer::seq_parallel::SpPlan`, rank 0
//! the first half of every sub-chunk, rank 1 the second).
//!
//! Per pass (attention, then FFN; the body is `seq_parallel::sp_pass`):
//! 1. Front, owned rows only (half of each sub-chunk): (`hc_expand` on layer 0,) `hc_pre`,
//!    the norm into those rows of `norm_output` (whole-chunk sized; checked by the caller).
//! 2. Grouped send/recv: each rank sends its normed rows, receives the peer's.
//! 3. The mixer per attention call (attention) or the MLP per FFN window, over ALL rows,
//!    exactly the launches `prefill_staged_run` issues, reading the same normed rows.
//! 4. Per item, a reduce-scatter in place of the all-reduce: the partial rows the peer owns
//!    go to the peer; the peer's partial of my rows arrives in my (dead) `hidden` rows, and I
//!    add my partial into it with `bf16_add_inplace`.
//! 5. Back, owned rows only, after the pass: `hc_post` from `hidden` (and on the last
//!    layer `hc_head_mean`). The last layer then swaps the final `hidden` rows, so both ranks
//!    leave with the whole output, as today.
//!
//! 2026-10-04: With `METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1` (`wide`) the passes issue the
//! full-width arm's calls: the attention calls are the `sub_chunks(num_tokens, rows_ffn)`
//! windows with the DSA core at `rows`, and each FFN window's MLP runs its dense GEMMs as one
//! slice as wide as the window. Ownership stays per sub-chunk (`SpPlan::spans` clips it to a
//! window), and each owned `hc_pre` takes the mix kernel of the call it belongs to.
//! 2026-10-05: With `METRALE_GLM_PREFILL_SP_WINDOW_OWNER=1` ownership is cut per `rows_ffn`
//! window instead (`seq_parallel::owner_chunks`): one span per rank per call. Nothing below
//! depends on where the ownership cuts fall.
//!
//! Why this is byte-identical to `prefill_staged_run` (its module notes give the base case):
//! - `hc_expand`, `hc_pre`, the RMSNorm, `hc_post` and `hc_head_mean` are one block (or one
//!   grid row) per token and read only that token's highway slot, `post`/`comb` and row
//!   (`blockIdx.x` is the token; none reads `gridDim.x`; `glm5next_mhc.cu`,
//!   `rms_norm_vanilla.cu`), so running them for a subset of rows (half-sub-chunk launches)
//!   writes those rows' bytes as the full launch does. The one choice that depends on a
//!   launch's row count, `METRALE_GLM_MHC_TOKMAJOR`'s `>= 64` rows, is made from the enclosing
//!   call's width (`glm_hc_pre_part`), so every row runs the mix kernel it runs today. The
//!   highway was identical on both ranks before (every rank ran the same replicated launches
//!   on the same all-reduced values), so the normed rows a rank receives are the bytes it
//!   would have computed. Each rank's highway is valid only on its own rows from layer 0 on,
//!   and only its own rows are read until the final swap.
//! - The mixer and MLP launches, widths and order are unchanged (the same `attn_mixer` /
//!   `mlp_compute` arguments `prefill_staged_run` passes, full width or not); only the
//!   address of their (identical) normed input moves within `norm_output`, by whole rows
//!   (`hidden * 2` bytes), and DSA writes its partial over the call's own rows there. No
//!   kernel choice reads that address (cuBLASLt plans from the shapes and types alone,
//!   `gpu-runtime/src/cublaslt.rs` `gemm_act_weight_t_out`).
//! - The all-reduce leaves `__hadd(own, peer)` on every element (`bf16_add_inplace` with
//!   `dst` = own partial); here the owner computes `__hadd(peer, own)`, the same bits because
//!   IEEE addition commutes. Only the backend whose all-reduce is that exchange and add takes
//!   this path (`CommBackend::all_reduce_is_send_recv_add`).
//! - Moving the owned `hc_post`s after the pass changes no input: a mixer reads no highway
//!   slot, and the FFN front reads slots only after the attention back has run. Under the
//!   full-width arm the same holds per window: a window's mixer reads only its normed rows,
//!   its mixer state and the KV / indexer rows earlier calls wrote.
//!
//! Between layers `hidden` holds scratch in either path, but different scratch here (a
//! DFlash capture of a mid-stack layer's `hidden` would see it change); the final layer's
//! `hidden` and every highway slot a rank reads are as today.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Both ranks issue the same collectives in the same order (the plan and the calls are a
//!   function of the chunk length, the widths and the levers alone); an empty send or receive
//!   is skipped on both sides.

use super::*;
use crate::glm5next_layer::seq_parallel::{SpPlan, SpRows, SpSite, sp_pass, swap_owned};

impl Glm5NextLayer {
    /// 2026-10-01: The plan when this staged prefill of `num_tokens` rows in `subs` takes the
    /// sequence-parallel path: lever on, eager, a two-rank communicator whose all-reduce is
    /// the send/recv + add exchange, the add kernel loaded, the chunk within `norm_output`.
    /// Every input is the same on both ranks.
    pub(in crate::glm5next_layer) fn sp_plan(
        &self,
        num_tokens: usize,
        subs: &[(usize, usize)],
        ctx: &ForwardContext,
    ) -> Option<SpPlan> {
        let comm = ctx.comm?;
        let ok = prefill_seq_parallel()
            && !ctx.graph_capture
            && comm.world_size() == 2
            && comm.all_reduce_is_send_recv_add()
            && self.add_k.0 != 0
            && self.mhc.is_some()
            && num_tokens * self.hidden * 2 <= ctx.buffers.norm_output_bytes();
        if !ok {
            static SKIPPED: std::sync::Once = std::sync::Once::new();
            if prefill_seq_parallel() {
                SKIPPED.call_once(|| {
                    tracing::warn!(
                        "METRALE_GLM_PREFILL_SEQ_PARALLEL=1: NOT engaged on a prefill of \
                         {num_tokens} rows (needs eager, a two-rank send/recv all-reduce, \
                         bf16_add_inplace and a chunk within norm_output)"
                    );
                });
            }
            return None;
        }
        let plan = SpPlan::new(subs)?;
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_PREFILL_SEQ_PARALLEL=1: ENGAGED (first prefill: {num_tokens} rows, \
                 each rank owns half of every {}-row ownership chunk; full-width arm {})",
                plan.width,
                prefill_fullwidth_gemm()
            );
        });
        Some(plan)
    }

    /// 2026-10-01: `prefill_staged_run` over `subs` under `plan` (module notes).
    /// 2026-10-04: `wide` is `prefill_staged_run`'s full-width arm: the attention calls are the
    /// `rows_ffn` windows (DSA core at `rows`) and the FFN windows' dense slices are the
    /// windows themselves.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn prefill_staged_sp(
        &self,
        hidden: DevicePtr,
        subs: &[(usize, usize)],
        plan: SpPlan,
        rows: usize,
        rows_ffn: usize,
        wide: bool,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(comm) = ctx.comm else {
            bail!(
                "GLM layer {}: sequence-parallel prefill without a communicator",
                self.layer_idx
            );
        };
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let gpu = ctx.gpu;
        let num_tokens = plan.total;
        let lanes = SpRows {
            mhc: &mhc.kernels,
            rms_norm: self.rms_norm_k,
            hidden: self.hidden,
            hc_mult: mhc.hc_mult,
            sinkhorn_iters: mhc.sinkhorn_iters as u32,
            rms_eps: self.rms_eps,
            hc_eps: mhc.hc_eps,
            streams: ctx.buffers.hc_streams(),
            post: ctx.buffers.hc_post(),
            comb: ctx.buffers.hc_comb(),
            normed: ctx.buffers.norm_output(),
        };

        // 2026-10-01: Attention pass.
        // 2026-10-04: Over the calls `prefill_staged_run` makes: the sub-chunks, or under the
        // full-width arm the `rows_ffn` windows with the DSA core at `rows`.
        let attn_calls = if wide {
            sub_chunks(num_tokens, rows_ffn)
        } else {
            subs.to_vec()
        };
        let attn = SpSite {
            weights: &mhc.attn,
            norm: self.input_norm,
            expand: self.is_first,
            last: false,
            reduce: self.mixer_all_reduce,
            attn: true,
        };
        let mixer = |(t, k): (usize, usize), x: DevicePtr| {
            self.attn_mixer(
                x,
                k,
                state,
                kv_cache,
                seq_len_start + t,
                block_table,
                ctx,
                stream,
                // 2026-10-01: No KDA snapshots and a prefill sub-chunk, as
                // `prefill_staged_run` passes.
                false,
                true,
                // 2026-10-04: The DSA core: all `k` rows, or `rows` under the full-width
                // arm, as `prefill_staged_run` passes.
                if wide { rows.min(k) } else { k },
            )
        };
        let add_k = self.add_k;
        sp_pass(gpu, comm, add_k, plan, &lanes, hidden, &attn_calls, attn, mixer, stream)?;

        // 2026-10-01: FFN pass, over the windows `prefill_staged_run` uses.
        let ffn_out = ctx.buffers.moe_output();
        let wins = ffn_windows(subs, rows_ffn, prefill_tail_merge(), |k| self.ffn_mergeable(k));
        let ffn = SpSite {
            weights: &mhc.ffn,
            norm: self.post_attn_norm,
            expand: false,
            last: self.is_last,
            reduce: self.mlp_cfg.needs_all_reduce(),
            attn: false,
        };
        let mlp = |(_, k): (usize, usize), x: DevicePtr| {
            // 2026-10-04: Full width: the window's dense GEMMs in one slice, as
            // `prefill_staged_run` passes.
            let dense_slice = if wide { k } else { rows };
            self.mlp_compute(x, ffn_out, k, dense_slice, ctx, stream)
                .map(|()| ffn_out)
        };
        sp_pass(gpu, comm, add_k, plan, &lanes, hidden, &wins, ffn, mlp, stream)?;
        if self.is_last {
            // 2026-10-01: Both ranks leave with every final row, as `prefill_staged_run` does.
            swap_owned(comm, plan, hidden, (0, num_tokens), self.hidden * 2, stream)?;
            profile::step();
        }
        Ok(())
    }
}
