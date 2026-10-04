// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The MTP drafter's entry points into its `Glm5NextLayer`: one decode token, and
//! one context row's DSA cache writes.
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
