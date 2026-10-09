// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `Glm5NextDsaLayer::xseq_group`, one group of `decode_xseq` (module notes in
//! `xseq.rs`): the projections and the indexer projections over the group's rows, the
//! per-sequence body (`decode_k_rows`) over each sequence's rows, and the output projection.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Every sequence's lockstep, room and metadata checks run before the group's first launch.

use anyhow::{Result, bail, ensure};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, LayerState};

use super::super::row_batch::batchm_rows;
use super::super::row_src::{PreIndexer, RowSrc};
use super::super::{Glm5NextDsaLayer, gemm};
use super::{DsaXseqArena, dsa_state};

impl Glm5NextDsaLayer {
    /// 2026-10-03: One group of `decode_xseq` (at most `arena.max_rows` rows).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn xseq_group(
        &self,
        a: &DsaXseqArena,
        hidden: DevicePtr,
        ks: &[usize],
        states: &mut [&mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        seq_meta: &[Option<AttnMetadataDev>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::glm5next_layer::profile;
        let gpu = ctx.gpu;
        let c = &self.cfg;
        let r: usize = ks.iter().sum();
        ensure!(
            r <= a.max_rows,
            "DSA xseq group of {r} rows over {}",
            a.max_rows
        );
        // 2026-10-03: Every sequence's checks before the group's first launch: the lockstep
        // check (`decode_k`'s first step), room for its rows (the loop checks one row at a
        // time), and `decode_k`'s metadata rule, from which its row-wise metadata follows.
        let mut rowwise: Vec<Option<AttnMetadataDev>> = Vec::with_capacity(ks.len());
        for (i, state) in states.iter_mut().enumerate() {
            let st = dsa_state(&mut **state)?;
            self.check_lockstep(st, seq_lens[i])?;
            st.ensure_room(ks[i])?;
            let k = ks[i];
            if k > 1
                && let Some(m) = seq_meta[i].as_ref()
                && m.num_seqs as usize != k
                && ctx.decode_step
            {
                bail!(
                    "DSA layer {}: a {k}-row pass cannot share attn_metadata describing {} \
                     token(s) — its position and KV slot describe a single token",
                    self.layer_idx,
                    m.num_seqs
                );
            }
            rowwise.push(
                (k > 1)
                    .then_some(seq_meta[i])
                    .flatten()
                    .filter(|m| m.num_seqs as usize == k),
            );
        }

        let (hid, ql, kvl) = (c.hidden, c.q_lora_rank, c.kv_lora_rank);
        let lat = c.local_heads * kvl;
        let (d, heads) = (c.index_head_dim, c.index_heads);
        let kn = &self.kernels;
        let wt = &self.weights;
        let t_proj = profile::start();
        // 2026-10-09: `METRALE_GLM_NV4_TC_GROUP=1`: q_a and kv_a both read `hidden`, and the q
        // chain below neither reads `a.kv_a` nor writes `hidden`, so kv_a may run with q_a as
        // one grouped launch here (`dense_nv4_group`, same bytes) and is then skipped below.
        let grouped = crate::glm5next_layer::dense_nv4_group::enabled()
            && self.try_group_q_a_kv_a(gpu, hidden, a.q_a, a.kv_a, r, stream)?;
        if !grouped {
            gemm(
                gpu,
                kn.gemm,
                kn.gemv,
                kn.gemv_batchm,
                hidden,
                wt.q_a_proj,
                a.q_a,
                r,
                ql,
                hid,
                stream,
            )?;
        }
        // 2026-10-03: One block per row, as in `decode_k`.
        KernelLaunch::new(gpu, kn.rms_norm)
            .grid([r as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a.q_a)
            .arg_ptr(wt.q_a_layernorm)
            .arg_ptr(a.q_resid)
            .arg_u32(ql as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        gemm(
            gpu,
            kn.gemm,
            kn.gemv,
            kn.gemv_batchm,
            a.q_resid,
            wt.q_absorb,
            a.q_abs,
            r,
            lat,
            ql,
            stream,
        )?;
        if !grouped {
            gemm(
                gpu,
                kn.gemm,
                kn.gemv,
                kn.gemv_batchm,
                hidden,
                wt.kv_a_proj,
                a.kv_a,
                r,
                kvl,
                hid,
                stream,
            )?;
        }
        profile::end(profile::DSA_PROJ, t_proj, gpu, stream);

        // 2026-10-03: The indexer projections of every row, as `indexer_rows_batched` and
        // `qidx_rows_batched` run them, into the arena instead of the caches.
        let t = profile::start();
        // 2026-10-05: `wk` and `compress_gate` go through `gemm`, the dispatch the per-sequence
        // decode uses (`dense_fp8::route`: the row-invariant TC GEMV under
        // `METRALE_GLM_GEMV_TC=1`), so a sequence's indexer keys do not depend on whether it
        // decodes alone or in a cross-sequence group. `batchm_rows` on `dense_gemv_bf16_batchm`
        // matched the per-sequence path only with TC off (race-dec-lmhbf-L34: C=2/4 texts
        // diverged from C=1 under the ship env).
        gemm(
            gpu,
            kn.gemm,
            kn.gemv,
            kn.gemv_batchm,
            hidden,
            wt.wk,
            a.k_normed,
            r,
            d,
            hid,
            stream,
        )?;
        // 2026-10-03: `nllb_layernorm_bf16` normalises row `blockIdx.x` in place; every block
        // does the same arithmetic on its own row at any `rows` (`indexer_rows_batched`).
        KernelLaunch::new(gpu, self.select_kernels.k_norm)
            .grid([r as u32, 1, 1])
            .block([d.min(1024) as u32, 1, 1])
            .shared_mem((d.min(1024) * 4) as u32)
            .arg_ptr(a.k_normed)
            .arg_ptr(wt.k_norm_weight)
            .arg_ptr(wt.k_norm_bias)
            .arg_u32(r as u32)
            .arg_u32(d as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        gemm(
            gpu,
            kn.gemm,
            kn.gemv,
            kn.gemv_batchm,
            hidden,
            wt.compress_gate,
            a.gate,
            r,
            d,
            hid,
            stream,
        )?;
        let f32k = a.gemv_batchm_f32;
        let wp = wt.weights_proj;
        batchm_rows(
            gpu,
            f32k,
            4,
            hidden,
            wp,
            a.head_weights,
            r,
            heads,
            hid,
            stream,
        )?;
        let nq = heads * d;
        batchm_rows(gpu, f32k, 4, a.q_resid, wt.wq_b, a.q_idx, r, nq, ql, stream)?;
        profile::end(profile::DSA_INDEXER, t, gpu, stream);

        // 2026-10-03: Per sequence, the loop's own body over the sequence's arena rows.
        let mut o = 0;
        for (i, state) in states.iter_mut().enumerate() {
            let k = ks[i];
            let st = dsa_state(&mut **state)?;
            let src = RowSrc {
                kv_a: a.kv_a.offset(o * kvl * 2),
                q_resid: a.q_resid.offset(o * ql * 2),
                q_abs: a.q_abs.offset(o * lat * 2),
                attn_out: a.attn_out.offset(o * lat * 2),
                pre: Some(PreIndexer {
                    k_normed: a.k_normed.offset(o * d * 2),
                    gate: a.gate.offset(o * d * 2),
                    head_weights: a.head_weights.offset(o * heads * 4),
                    q_idx: a.q_idx.offset(o * nq * 4),
                }),
            };
            let row_ctx = ForwardContext {
                attn_metadata: seq_meta[i],
                ..*ctx
            };
            let mut bt = block_tables[i].clone();
            self.decode_k_rows(
                &src,
                hidden.offset(o * hid * 2),
                k,
                st,
                kv_cache,
                seq_lens[i],
                &mut bt,
                &row_ctx,
                rowwise[i],
                profile::start(),
                stream,
                // 2026-10-03: `is_prefill` false: a decode step or a speculative verify, as
                // both per-sequence loops pass it.
                false,
            )?;
            o += k;
        }

        let t_proj = profile::start();
        gemm(
            gpu,
            kn.gemm,
            kn.gemv,
            kn.gemv_batchm,
            a.attn_out,
            wt.o_absorb,
            hidden,
            r,
            hid,
            lat,
            stream,
        )?;
        profile::end(profile::DSA_PROJ, t_proj, gpu, stream);
        Ok(())
    }
}
