// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `Glm5NextDsaLayer::write_kv_rows`: `write_kv_row` for `k` consecutive MTP
//! drafter context rows at once (`METRALE_GLM_MTP_CTX_ROWBATCH=1`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Row `r` of the call lands exactly where `write_kv_row` at `seq_len + r` puts it: KV slot
//!   from the block table, indexer row `state.len() + r`, and the same bytes, because the
//!   projections run through the batched GEMV twins that are bit-identical per row to the
//!   M = 1 GEMV (see `row_batch.rs`) and the norms are one block per row.
//! - No collective is issued, as in `write_kv_row`.
//! - Not written: the selector head weights (`weights_proj`). `write_kv_row` leaves them in
//!   workspace scratch that a context row never reads, so no carried state differs.

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layer::LayerState;

use super::super::state::Glm5NextDsaState;
use super::Glm5NextDsaLayer;
use super::row_batch::batchm_rows;

impl Glm5NextDsaLayer {
    /// 2026-10-03: Whether `write_kv_rows` can run: the batched BF16 GEMV is resolved.
    pub fn can_write_kv_rows(&self) -> bool {
        self.kernels.gemv_batchm.0 != 0
    }

    /// 2026-10-03: Write `k` context rows. `hidden` is `[k, hidden]` BF16 (already
    /// input-normed), `kv_a` a caller scratch of `[k, kv_lora_rank]` BF16, `slots` a device
    /// `[k]` i64 buffer this call fills. `seq_len` is the first row's position.
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv_rows(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        kv_a: DevicePtr,
        slots: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &[u32],
        stream: u64,
    ) -> Result<()> {
        if k == 0 {
            return Ok(());
        }
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?;
        match st.len().cmp(&seq_len) {
            std::cmp::Ordering::Greater => st.rewind_to(seq_len)?,
            std::cmp::Ordering::Less => bail!(
                "DSA layer {}: indexer cache holds {} rows but the drafter is at {seq_len} — \
                 rows are MISSING, not merely stale.",
                self.layer_idx,
                st.len()
            ),
            std::cmp::Ordering::Equal => {}
        }
        // Checked before any write, as `indexer_forward`'s `ensure_room(1)` is per row.
        st.ensure_room(k)?;
        let bm = self.kernels.gemv_batchm;
        let (kvr, h, d) = (
            self.cfg.kv_lora_rank,
            self.cfg.hidden,
            self.cfg.index_head_dim,
        );
        batchm_rows(gpu, bm, 2, hidden, self.weights.kv_a_proj, kv_a, k, kvr, h, stream)?;
        let block_size = kv_cache.config().block_size;
        let mut host = Vec::with_capacity(k * 8);
        for r in 0..k {
            let pos = seq_len + r;
            let physical = *block_table.get(pos / block_size).ok_or_else(|| {
                anyhow::anyhow!(
                    "DSA layer {}: block table has {} entries, needs logical block {} for \
                     drafter row {pos}",
                    self.layer_idx,
                    block_table.len(),
                    pos / block_size
                )
            })? as usize;
            let slot = (physical * block_size + pos % block_size) as i64;
            host.extend_from_slice(&slot.to_le_bytes());
        }
        // Blocking copy, like the row loop's. The caller synchronises `stream` before
        // reusing `slots` for the next tile.
        gpu.copy_h2d(&host, slots)?;
        KernelLaunch::new(gpu, self.kernels.latent_write)
            .grid([k as u32, 1, 1])
            .block([kvr as u32, 1, 1])
            .arg_ptr(kv_a)
            .arg_ptr(self.weights.kv_a_layernorm)
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_layer_idx))
            .arg_ptr(slots)
            .arg_u32(kvr as u32)
            .arg_f32(self.rms_eps)
            .arg_f32(1.0 / self.kv_scale)
            .launch(stream)?;
        // Indexer key side for all k rows, as `indexer_rows_batched` writes it.
        let pos0 = st.len();
        let off = st.row_offset(pos0);
        let k_rows = st.k_normed.offset(off);
        batchm_rows(gpu, bm, 2, hidden, self.weights.wk, k_rows, k, d, h, stream)?;
        KernelLaunch::new(gpu, self.select_kernels.k_norm)
            .grid([k as u32, 1, 1])
            .block([d.min(1024) as u32, 1, 1])
            .shared_mem((d.min(1024) * 4) as u32)
            .arg_ptr(k_rows)
            .arg_ptr(self.weights.k_norm_weight)
            .arg_ptr(self.weights.k_norm_bias)
            .arg_u32(k as u32)
            .arg_u32(d as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        let gate = st.gate.offset(off);
        batchm_rows(gpu, bm, 2, hidden, self.weights.compress_gate, gate, k, d, h, stream)?;
        gpu.memset_async(st.valid.offset(pos0), 1, k, stream)?;
        st.advance(k)
    }
}
