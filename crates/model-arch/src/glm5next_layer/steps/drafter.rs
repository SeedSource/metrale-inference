// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The MTP drafter's entry points into its `Glm5NextLayer`: one decode token, and
//! one context row's DSA cache writes. 2026-10-04: and one token for each of `n` sequences in
//! one call (`decode_rows_for_drafter`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;

impl Glm5NextLayer {
    /// 2026-09-25: One token through this block for the MTP drafter; errors on a block with a
    /// hyper-connection. `hidden` is both input and residual and is updated in place. No disk
    /// tiers: the disk block lists passed down are empty.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_one_for_drafter(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.mhc.is_some() {
            bail!(
                "GLM layer {}: decode_one_for_drafter is the MTP block's path; this layer has a \
                 hyper-connection",
                self.layer_idx
            );
        }
        let (mut disk_a, mut disk_b) = (Vec::new(), Vec::new());
        self.forward_one_plain(
            hidden,
            hidden,
            state,
            kv_cache,
            seq_len,
            block_table,
            &mut disk_a,
            &mut disk_b,
            ctx,
            stream,
        )
    }

    /// 2026-10-04: `n` drafter rows, one per sequence, through this block in one call
    /// (`METRALE_GLM_MTP_BATCH_DRAFT=1`, `Glm5NextMtpHead::propose_batch`): `forward_one_plain`
    /// with the row axis read as the sequence axis. `hidden` is `[n, hidden]` BF16, input and
    /// residual, updated in place; row `i` belongs to `states[i]`, `kv_caches[i]`,
    /// `seq_lens[i]` and `block_tables[i]`.
    ///
    /// The norms, the adds, the mixer all-reduce and the MLP (router, routed experts, shared
    /// expert, its all-reduce) run once over all `n` rows; the DSA mixer runs per row through
    /// the same `mixer_forward` call `forward_one_plain` makes, against that row's own state,
    /// KV pool, length and block table. Every batched launch here is row-local (the BF16 GEMVs
    /// at 2..=16 rows are `dense_gemv_bf16_batchm`, bit-identical per row to the one-row
    /// `dense_gemv_bf16`; the MoE combine is per row and slot), so each row's bytes equal what
    /// `decode_one_for_drafter` writes for that row alone. Errors on a block with a
    /// hyper-connection, `n` of 0, or a slice shorter than `n`, before any launch.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_rows_for_drafter(
        &self,
        hidden: DevicePtr,
        n: usize,
        states: &mut [&mut dyn LayerState],
        kv_caches: &mut [&mut PagedKvCache],
        seq_lens: &[usize],
        block_tables: &mut [&mut Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.mhc.is_some() {
            bail!(
                "GLM layer {}: decode_rows_for_drafter is the MTP block's path; this layer has \
                 a hyper-connection",
                self.layer_idx
            );
        }
        if n == 0
            || states.len() < n
            || kv_caches.len() < n
            || seq_lens.len() < n
            || block_tables.len() < n
        {
            bail!(
                "GLM layer {}: {n} drafter rows got states={} kv={} seq_lens={} block_tables={}",
                self.layer_idx,
                states.len(),
                kv_caches.len(),
                seq_lens.len(),
                block_tables.len()
            );
        }
        let gpu = ctx.gpu;
        let h = self.hidden;
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();

        self.norm(gpu, hidden, self.input_norm, normed, n, stream)?;
        for i in 0..n {
            let (mut disk_a, mut disk_b) = (Vec::new(), Vec::new());
            let row = normed.offset(i * h * 2);
            let out = self.mixer_forward(
                row,
                hidden.offset(i * h * 2),
                &mut *states[i],
                &mut *kv_caches[i],
                seq_lens[i],
                &mut *block_tables[i],
                &mut disk_a,
                &mut disk_b,
                ctx,
                stream,
            )?;
            // 2026-10-04: DSA writes its output projection over the row it was handed, so the
            // rows land contiguous in `normed`, where the batched all-reduce and add read them.
            if out.0 != row.0 {
                bail!(
                    "GLM layer {}: the drafter block's mixer did not write in place",
                    self.layer_idx
                );
            }
        }
        if self.mixer_all_reduce {
            self.reduce_partial(normed, n, ctx, stream)?;
        }
        self.add_inplace(gpu, hidden, normed, n * h, stream)?;

        self.norm(gpu, hidden, self.post_attn_norm, normed, n, stream)?;
        self.mlp_forward(normed, ffn_out, n, n, ctx, stream)?;
        self.add_inplace(gpu, hidden, ffn_out, n * h, stream)
    }

    /// 2026-09-25: One drafter context row: `input_norm`, then `Glm5NextDsaLayer::write_kv_row`,
    /// which writes the row's KV latent and indexer entry without computing the block output.
    /// Errors on a block whose mixer is not DSA.
    #[allow(clippy::too_many_arguments)]
    pub fn drafter_write_kv_row(
        &self,
        x: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let layer = match &self.mixer {
            Glm5NextMixer::Dsa(l) => l,
            _ => bail!(
                "GLM layer {}: drafter_write_kv_row is the MTP block's path; this layer is not \
                 a DSA layer",
                self.layer_idx
            ),
        };
        let normed = ctx.buffers.norm_output();
        self.norm(ctx.gpu, x, self.input_norm, normed, 1, stream)?;
        layer.write_kv_row(normed, state, kv_cache, seq_len, block_table, ctx, stream)
    }

    /// 2026-10-03: `drafter_write_kv_row` for `k` consecutive rows: `input_norm` over all
    /// `k` rows of `x` (`[k, hidden]`) into `normed`, then `Glm5NextDsaLayer::write_kv_rows`.
    /// `normed` and `kv_a` are caller scratch (`[k, hidden]`, `[k, kv_lora_rank]` BF16), `slots`
    /// a device `[k]` i64 buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn drafter_write_kv_rows(
        &self,
        x: DevicePtr,
        k: usize,
        normed: DevicePtr,
        kv_a: DevicePtr,
        slots: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let layer = match &self.mixer {
            Glm5NextMixer::Dsa(l) => l,
            _ => bail!(
                "GLM layer {}: drafter_write_kv_rows is the MTP block's path; this layer is not                  a DSA layer",
                self.layer_idx
            ),
        };
        self.norm(ctx.gpu, x, self.input_norm, normed, k, stream)?;
        layer.write_kv_rows(
            ctx.gpu, normed, k, kv_a, slots, state, kv_cache, seq_len, block_table, stream,
        )
    }

    /// 2026-10-03: Whether `drafter_write_kv_rows` can run on this block.
    pub fn can_drafter_write_kv_rows(&self) -> bool {
        matches!(&self.mixer, Glm5NextMixer::Dsa(l) if l.can_write_kv_rows())
    }
}
