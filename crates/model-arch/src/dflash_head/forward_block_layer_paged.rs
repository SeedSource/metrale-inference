// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The paged drafter layer body. The block rows attend over the ctx K/V
//! already in the drafter's paged BF16 cache (written by `precompute_ctx_kv`) and the
//! block K/V each layer writes; no ctx rows run through the layers.
//!
//! Each layer is three calls. Every op except attention covers all
//! `n_seq * block_g()` rows at once.
//! - `forward_block_layer_pre_attn`: input_layernorm, DFlash2 attention conv prepare,
//!   q_proj and q_norm, k_proj and k_norm, v_proj, RoPE, and the block K/V write into
//!   the paged cache.
//! - `forward_block_layer_attention`: non-causal paged attention, one launch per
//!   sequence.
//! - `forward_block_layer_post_attn`: o_proj, DFlash2 attention conv finish, residual
//!   add, post_attention_layernorm, DFlash2 MLP conv prepare, gate/up, SiLU-mul, down,
//!   DFlash2 MLP conv finish, residual add.
//!
//! When graphs are on, `forward_block` captures the first and third calls as separate
//! subgraphs and runs attention eagerly between them.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{BlockDiffusionDraftHead, DflashLayer};
use metrale_model_layers::layer::ForwardContext;

mod contig_attn;
mod post_attn;
mod pre_attn;

pub(super) use post_attn::PostAttnPart;

/// 2026-09-25: Inputs to the paged layer body, built by `forward_block` for each layer.
#[allow(clippy::too_many_arguments)]
pub(super) struct PagedLayerArgs {
    pub layer_idx: usize,
    /// 2026-09-25: Sequence 0's count of ctx slots already in the paged cache. The
    /// contiguous attention path and the OPTION_B diagnostic read it; the indirect
    /// attention reads its values from `option_b_indirect_args_dev` instead.
    pub ctx_count: u32,
    pub h: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub inter: u32,
    pub inv_sqrt_d: f32,
    /// 2026-09-25: The i64 cache slot of each block row, `n_seq * block_g()` entries
    /// packed seq-major. Built once per forward and shared by every layer.
    pub slot_mapping_gamma: DevicePtr,
    /// 2026-09-25: Sequence 0's drafter block table, which the single-sequence
    /// attention reads (a batch uses `seq_block_tables`).
    pub block_table_dev: DevicePtr,
    pub stream: u64,
    /// 2026-09-25: This forward is the armed one-shot per-layer dump
    /// (`METRALE_DFLASH_BLOCK_DUMP=1`): each layer writes its block intermediates to
    /// /tmp/metrale_blk_L{layer}_{stage}.bin. That lever also keeps graph capture off,
    /// so the dump's syncs and copies always run eagerly.
    pub block_dump: bool,
    /// 2026-09-25: Sequences packed into this forward, seq-major: sequence i owns rows
    /// `[i * block_g(), (i + 1) * block_g())`. Every weight-bearing op runs over all
    /// `n_seq * block_g()` rows at once, so the drafter weights are read once per
    /// forward rather than once per sequence. `1` on the single-sequence path.
    pub n_seq: u32,
    /// 2026-09-25: Per-sequence drafter block tables, `n_seq` long; empty on the
    /// single-sequence path, which uses `block_table_dev`. Only attention reads them:
    /// it reads each sequence's own KV pages, one launch per sequence. Attention reads
    /// no weights, so the per-sequence launches do not re-read any weight.
    pub seq_block_tables: Vec<DevicePtr>,
}

impl BlockDiffusionDraftHead {
    /// 2026-09-25: Runs the three calls of one layer in order. Nothing calls it:
    /// `forward_block` calls the three itself so it can capture them separately.
    #[allow(dead_code)]
    pub(super) fn forward_block_layer_paged(
        &self,
        layer: &DflashLayer,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let (k_pool, v_pool) = self.forward_block_layer_pre_attn(layer, args, ctx)?;
        self.forward_block_layer_attention(args, ctx, k_pool, v_pool)?;
        self.forward_block_layer_post_attn(layer, args, ctx)?;
        Ok(())
    }

    /// 2026-09-25: Write `rows * cols` BF16 values from `src` to
    /// `/tmp/metrale_blk_L{layer_idx}_{stage}.bin` after syncing `stream`. A failed
    /// file write is logged, not returned. Called only when `block_dump` is set.
    pub(super) fn block_dump_buf(
        &self,
        ctx: &ForwardContext,
        src: DevicePtr,
        layer_idx: usize,
        stage: &str,
        rows: u32,
        cols: u32,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let bf16 = 2usize;
        let n_bytes = rows as usize * cols as usize * bf16;
        gpu.synchronize(stream)?;
        let mut buf = vec![0u8; n_bytes];
        gpu.copy_d2h(src, &mut buf)?;
        let path = format!("/tmp/metrale_blk_L{layer_idx}_{stage}.bin");
        if let Err(e) = std::fs::write(&path, &buf) {
            tracing::warn!("DFLASH BLOCK_DUMP per-layer: write {path} failed: {e}");
        } else if layer_idx == 0 {
            tracing::info!("DFLASH BLOCK_DUMP per-layer: wrote {path} ({rows}x{cols} BF16)");
        }
        Ok(())
    }

    /// 2026-09-25: Pre-attention call: input_layernorm, DFlash2 attention conv prepare,
    /// q_proj and q_norm, k_proj and k_norm, v_proj, RoPE, and the block K/V write into
    /// this layer's paged cache. Returns this layer's `(k_pool, v_pool)`.
    ///
    /// `forward_block` captures it when graphs are on. Its host copies and syncs run
    /// only under `METRALE_DFLASH_OPTION_B_DIAG` or the block dump, and both levers keep
    /// capture off (`GRAPH_SUPPRESSING_DIAGNOSTICS` in levers.rs).
    pub(super) fn forward_block_layer_pre_attn(
        &self,
        layer: &DflashLayer,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
    ) -> Result<(DevicePtr, DevicePtr)> {
        self.pre_attn_layer(layer, args, ctx)
    }

    /// 2026-09-25: Non-causal paged attention of each sequence's block rows over that
    /// sequence's ctx and block K/V in this layer's pool, one launch per sequence.
    /// `forward_block` never captures it. `METRALE_DFLASH_CONTIG_ATTN=1` runs
    /// `forward_block_layer_attention_contig` instead.
    pub(super) fn forward_block_layer_attention(
        &self,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
    ) -> Result<()> {
        use metrale_model_layers::layers::ops;

        if ctx.levers.dflash_contig_attn {
            return self.forward_block_layer_attention_contig(args, ctx, k_pool, v_pool);
        }

        let PagedLayerArgs {
            block_table_dev,
            stream,
            inv_sqrt_d,
            ..
        } = *args;
        let gpu = ctx.gpu;
        let g = self.block_g() as u32;

        // 2026-09-25: One launch per sequence over its band of `g` rows; `g` is the
        // per-sequence width (q_len) here. The kernel reads the sequence's kv_len,
        // q_offset and q_rope_pos from its 12-byte slot of
        // `option_b_indirect_args_dev`, which `forward_block` wrote for this propose.
        let n_seq = args.n_seq.max(1) as usize;
        let q_dim_bytes = (self.num_q_heads * self.head_dim) * 2;
        for i in 0..n_seq {
            let (bt_i, args_i) = if n_seq == 1 {
                (block_table_dev, self.scratch.option_b_indirect_args_dev)
            } else {
                (
                    *args.seq_block_tables.get(i).ok_or_else(|| {
                        anyhow::anyhow!("dflash attn: no block table for seq {i}")
                    })?,
                    self.scratch.option_b_indirect_args_dev.offset(i * 12),
                )
            };
            ops::prefill_attention_paged_dflash_bf16_indirect(
                gpu,
                self.kernels.prefill_attn_dflash_bf16_indirect,
                self.scratch.q_buf.offset(i * g as usize * q_dim_bytes),
                k_pool,
                v_pool,
                self.scratch.attn_out.offset(i * g as usize * q_dim_bytes),
                bt_i,
                g,
                args_i,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                16,
                0,
                inv_sqrt_d,
                stream,
            )?;
        }

        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.attn_out,
                args.layer_idx,
                "attn_out",
                g,
                args.q_dim,
                stream,
            )?;
        }

        Ok(())
    }

    /// 2026-09-25: Post-attention call: o_proj, DFlash2 attention conv finish, residual
    /// add, post_attention_layernorm, DFlash2 MLP conv prepare, gate/up, SiLU-mul,
    /// down, DFlash2 MLP conv finish, residual add. It copies to the host and syncs
    /// only for the block dump, which keeps capture off, so `forward_block` can
    /// capture it.
    pub(super) fn forward_block_layer_post_attn(
        &self,
        layer: &DflashLayer,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.post_attn_layer(layer, args, ctx)
    }

    /// 2026-09-26: The `METRALE_DFLASH_OPTION_B_DIAG=1` log of `forward_block_layer_pre_attn`,
    /// run after the block K/V write with that call's `k_pool` and `kv_len`.
    fn option_b_diag(
        &self,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
        k_pool: DevicePtr,
        kv_len: u32,
    ) -> Result<()> {
        let PagedLayerArgs {
            layer_idx,
            ctx_count,
            slot_mapping_gamma,
            block_table_dev,
            stream,
            ..
        } = *args;
        let gpu = ctx.gpu;
        // 2026-09-25: METRALE_DFLASH_OPTION_B_DIAG=1, once per model: log the first 8
        // BF16 values of layer 0's first block K row from `k_buf` and from its cache
        // slot, the first 4 block-table entries, and the first ctx K row.
        if layer_idx == 0 && self.levers.option_b_diag {
            // 2026-09-25: `ctx.stats` is this model's `ModelStats`, so each model logs once.
            if ctx.stats.dumped.keyed("dflash_option_b") {
                gpu.synchronize(stream)?;

                // 2026-09-25: The cache slot of block row 0 (i64).
                let mut slot0_bytes = [0u8; 8];
                gpu.copy_d2h(slot_mapping_gamma, &mut slot0_bytes)?;
                let slot0 = i64::from_le_bytes(slot0_bytes);
                let block_size: usize = 16;
                let phys_block = slot0 / block_size as i64;
                let block_off = slot0 % block_size as i64;

                // 2026-09-25: Its address, in BF16 elements from k_pool:
                //   phys_block * (block_size * num_kv_heads * head_dim)
                //   + block_off * (num_kv_heads * head_dim)
                let n_elems = self.num_kv_heads * self.head_dim;
                let block_stride_bytes = block_size * n_elems * 2;
                let row_stride_bytes = n_elems * 2;
                let cache_row_ptr = k_pool.offset(
                    (phys_block as usize) * block_stride_bytes
                        + (block_off as usize) * row_stride_bytes,
                );

                let read8 = |p: metrale_gpu_runtime::gpu::DevicePtr| -> Result<Vec<f32>> {
                    let mut b = [0u8; 16];
                    gpu.copy_d2h(p, &mut b)?;
                    Ok(b.chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            f32::from_bits((bits as u32) << 16)
                        })
                        .collect())
                };
                let src = read8(self.scratch.k_buf)?;
                let cached = read8(cache_row_ptr)?;
                tracing::info!(
                    "DFLASH OPTION_B DIAG: γ K layer0 slot0={} phys_block={} off={} \
                     src[0..8]={:?} cached[0..8]={:?}",
                    slot0,
                    phys_block,
                    block_off,
                    src,
                    cached,
                );

                let mut bt_bytes = [0u8; 16];
                gpu.copy_d2h(block_table_dev, &mut bt_bytes)?;
                let bt: Vec<u32> = bt_bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                tracing::info!(
                    "DFLASH OPTION_B DIAG: ctx_count={} block_table[0..4]={:?} kv_len={}",
                    ctx_count,
                    bt,
                    kv_len,
                );

                // 2026-09-25: ctx slot 0 is row 0 of physical block `block_table[0]`,
                // written by `precompute_ctx_kv`; all zeros means the ctx write missed.
                if ctx_count > 0 {
                    let ctx0_ptr = k_pool;
                    let ctx0_phys_block = bt[0] as usize;
                    let ctx0_addr = k_pool.offset(ctx0_phys_block * block_stride_bytes);
                    let ctx0 = read8(ctx0_addr)?;
                    let ctx0_max = ctx0.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                    tracing::info!(
                        "DFLASH OPTION_B DIAG: ctx K layer0 slot0 (phys_block={}) values={:?} max_abs={:.4}",
                        ctx0_phys_block,
                        ctx0,
                        ctx0_max,
                    );
                    let _ = ctx0_ptr;
                }
            }
        }
        Ok(())
    }
}
