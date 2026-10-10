// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Ctx K/V precompute for the paged drafter cache: project new ctx rows
//! and derive their K/V for every drafter layer in one pass.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - A call covers at most `ctx_window` rows; a wider call returns an error
//!   before any launch.
//! - K gets the layer's `k_norm` and RoPE at the row's `slot_positions` entry;
//!   V gets neither.
//!
//! With the paged cache on, `forward_block` runs its layers over the γ rows only
//! (`eff_ctx = 0`) and attends to the ctx K/V written here.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::weight_map::DenseWeight;

impl BlockDiffusionDraftHead {
    /// 2026-09-25: Project ctx rows through `fc` and `hidden_norm`, derive K/V for
    /// every drafter layer, apply per-layer `k_norm` and RoPE to K, and with
    /// `commit` write K/V into each layer's paged cache.
    ///
    /// `ctx_base_ptr`: base of a `[rows, L_t * h_t]` BF16 slab of captured
    ///   target hiddens.
    /// `start_slot`: first row of that slab to project.
    /// `new_ctx_count`: contiguous rows from `start_slot`; at most `ctx_window`.
    /// `slot_positions`: one RoPE position per row (`new_ctx_count` entries).
    /// `slot_mapping_dev`: `i64[new_ctx_count]` paged-cache slot indices.
    /// `commit`: write the K/V into the paged cache.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn precompute_ctx_kv(
        &self,
        ctx_base_ptr: DevicePtr,
        start_slot: usize,
        new_ctx_count: usize,
        slot_positions: &[i32],
        slot_mapping_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        commit: bool,
    ) -> Result<()> {
        use metrale_model_layers::layers::ops;

        if new_ctx_count == 0 {
            return Ok(());
        }
        // 2026-09-25: `fc_proj`, `fused_kv_out` and `slot_mapping_dev` hold
        // `ctx_window` rows; a wider call would write past them.
        anyhow::ensure!(
            new_ctx_count <= self.ctx_window,
            "DFlash precompute_ctx_kv: {new_ctx_count} rows exceed the {}-row ctx window",
            self.ctx_window
        );

        let Some(fused_kv) = self.fused_kv_weight else {
            anyhow::bail!(
                "DFlash precompute_ctx_kv called without fused_kv_weight — build-order bug"
            );
        };

        let gpu = ctx.gpu;
        let bf16 = 2usize;
        let h = self.hidden_size as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let n = new_ctx_count as u32;
        let l_total = self.num_layers;
        let target_hidden_dim = self.target_layer_ids.len() * self.target_hidden_size;
        let ctx_slot_bytes = target_hidden_dim * bf16;
        let kv_slab_bytes = (kv_dim as usize) * bf16;
        let row_stride = l_total * 2 * kv_slab_bytes;

        // 2026-09-25: `METRALE_DFLASH_PRECOMPUTE_DUMP=1`, one-shot per
        // `ctx.stats` (`ModelStats::dumped`).
        let dump = self.levers.precompute_dump && ctx.stats.dumped.keyed("dflash_precompute");
        let dump_buf = |label: &str, ptr: DevicePtr, bytes: usize| -> Result<()> {
            if !dump {
                return Ok(());
            }
            let mut buf = vec![0u8; bytes];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ptr, &mut buf)?;
            let path = format!("/tmp/metrale_precompute_{label}.bin");
            if let Err(e) = std::fs::write(&path, &buf) {
                tracing::warn!("precompute dump {label} write failed: {e}");
            } else {
                tracing::info!("precompute dump {label}: {} bytes → {}", bytes, path);
            }
            Ok(())
        };

        // 2026-09-25: Step 1: `fc` maps `[n, L_t * h_t]` to `[n, h]` (2026-10-09: split
        // across ranks under `METRALE_DFLASH_DRAFT_TP2`, see `tp/collectives.rs`).
        let src = ctx_base_ptr.offset(start_slot * ctx_slot_bytes);
        self.fc_project(ctx, src, n, target_hidden_dim as u32, stream)?;
        dump_buf(
            "fc_proj",
            self.scratch.fc_proj,
            new_ctx_count * self.hidden_size * bf16,
        )?;

        // 2026-09-25: Step 2: `hidden_norm` (RMS) in place.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.fc_proj,
            &self.hidden_norm,
            self.scratch.fc_proj,
            n,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        dump_buf(
            "fc_proj_normed",
            self.scratch.fc_proj,
            new_ctx_count * self.hidden_size * bf16,
        )?;

        // 2026-09-25: Step 3: one KV GEMM for all layers,
        // `[n, h] × [h, L * 2 * kv_dim]`. Each output row is
        // `[K_0 | V_0 | K_1 | V_1 | … | K_{L-1} | V_{L-1}]`.
        let fused_w = DenseWeight { weight: fused_kv };
        let fused_n_cols = (l_total as u32) * 2 * kv_dim;
        self.draft_gemm_bf16(
            gpu,
            self.scratch.fc_proj,
            &fused_w,
            self.scratch.fused_kv_out,
            n,
            fused_n_cols,
            h,
            stream,
        )?;
        dump_buf(
            "fused_kv_out",
            self.scratch.fused_kv_out,
            new_ctx_count * l_total * 2 * kv_slab_bytes,
        )?;

        // 2026-09-25: Step 4: `slot_positions` repeated L times, as i32, into
        // `norm_buf`. Every caller runs the γ-block layer loop, which reuses
        // `norm_buf`, only after this returns.
        debug_assert_eq!(slot_positions.len(), new_ctx_count);
        {
            let repeated_bytes: Vec<u8> = slot_positions
                .iter()
                .cloned()
                .cycle()
                .take(l_total * new_ctx_count)
                .flat_map(|p: i32| p.to_le_bytes())
                .collect();
            gpu.copy_h2d(&repeated_bytes, self.scratch.norm_buf)?;
        }

        // 2026-09-25: Step 5: copy every layer's K into a contiguous
        // `[L, n, kv_dim]` stage in `mlp_intermediate`, which the layer loop
        // also reuses only after this returns. The copies are async on
        // `stream`, where every later step of this function runs.
        let all_k_stage = self.scratch.mlp_intermediate;
        for l in 0..l_total {
            for row in 0..new_ctx_count {
                let k_src = self
                    .scratch
                    .fused_kv_out
                    .offset(row * row_stride + l * 2 * kv_slab_bytes);
                let k_dst = all_k_stage.offset((l * new_ctx_count + row) * kv_slab_bytes);
                gpu.copy_d2d_async(k_src, k_dst, kv_slab_bytes, stream)?;
            }
        }

        // 2026-09-25: Step 6: per-layer `k_norm`, per head: each layer's
        // `[n, kv_dim]` block is normed as `[n * num_kv_heads, head_dim]`.
        for l in 0..l_total {
            let k_l = all_k_stage.offset(l * new_ctx_count * kv_slab_bytes);
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                k_l,
                &self.layers[l].k_norm,
                k_l,
                n * self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rms_norm_eps,
                stream,
            )?;
        }

        // 2026-09-25: Step 7: one RoPE launch over all `L * n` K rows at the
        // step-4 positions. `num_q_heads = 0` makes the grid K-only, so the Q
        // pointer is never read.
        ops::rope_yarn(
            gpu,
            self.kernels.rope_qwen3,
            all_k_stage,
            all_k_stage,
            self.scratch.norm_buf,
            l_total as u32 * n,
            0,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            stream,
        )?;

        if dump {
            dump_buf(
                "layer0_k_post_rope",
                all_k_stage,
                new_ctx_count * kv_slab_bytes,
            )?;
        }

        // 2026-09-25: Step 8: per layer, compact V into `v_buf` and, with
        // `commit`, write that layer's K stage and V into its paged cache.
        let v_stage = self.scratch.v_buf;
        for l in 0..l_total {
            let k_l = all_k_stage.offset(l * new_ctx_count * kv_slab_bytes);

            for row in 0..new_ctx_count {
                let v_src = self
                    .scratch
                    .fused_kv_out
                    .offset(row * row_stride + l * 2 * kv_slab_bytes + kv_slab_bytes);
                let v_dst = v_stage.offset(row * kv_slab_bytes);
                gpu.copy_d2d_async(v_src, v_dst, kv_slab_bytes, stream)?;
            }

            if dump && l == 0 {
                dump_buf("layer0_v", v_stage, new_ctx_count * kv_slab_bytes)?;
            }

            if commit {
                let (k_pool, v_pool) = {
                    let cache = self.kv_cache.lock();
                    (cache.k_pool_ptr(l), cache.v_pool_ptr(l))
                };
                ops::reshape_and_cache(
                    gpu,
                    self.kernels.reshape_cache_bf16,
                    k_l,
                    v_stage,
                    k_pool,
                    v_pool,
                    slot_mapping_dev,
                    n,
                    self.num_kv_heads as u32,
                    self.head_dim as u32,
                    16, // 2026-09-25: the drafter cache's block size (from_weights.rs).
                    kv_dim,
                    kv_dim,
                    0,
                    stream,
                )?;
            }
        }

        Ok(())
    }
}
