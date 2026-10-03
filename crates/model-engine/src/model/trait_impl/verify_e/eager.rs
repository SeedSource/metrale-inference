// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The eager (or captured) body of the batched verify forward: the layer loop
//! over all R rows and the per-row argmax.
//!
//! Owner: model-engine (speculative verify).
//! Invariants:
//! - The argmax writes rows `0..r_total` to the mapped blob's device alias, or to scratch.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::LayerType;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;
use metrale_model_layers::layer::{ForwardContext, LayerState};
use metrale_model_layers::layers::ops;

impl TransformerModel {
    /// 2026-09-26: Run every layer over the R rows: attention through `decode_multi_seq`
    /// with `attn_dummy_states`, every other layer through `decode_verify_multi`.
    /// 2026-10-02: A layer that owns its verify state runs `decode_verify_multi_seqs`.
    pub(super) fn run_verify_layers(
        &self,
        attn_dummy_states: &mut [Vec<Box<dyn LayerState>>],
        seqs: &mut [&mut SequenceState],
        kv_cache: &mut PagedKvCache,
        seq_lens_vec: &[usize],
        block_tables_vec: &[Vec<u32>],
        hidden: DevicePtr,
        residual: DevicePtr,
        r_total: usize,
        n: usize,
        ks: &[usize],
        off: &[usize],
        wy_tables_base: DevicePtr,
        k4_diag: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut attn_idx = 0usize;
        let mut ssm_idx = 0usize;
        // 2026-10-02: Each sequence's pre-verify length and page table, for the layers that
        // drive their own verify state (`decode_verify_multi_own_states`).
        let (own_lens, own_bts): (Vec<usize>, Vec<Vec<u32>>) = if self.any_verify_own_states() {
            seqs.iter().map(|s| (s.seq_len, s.block_table.clone())).unzip()
        } else {
            (Vec::new(), Vec::new())
        };
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_type = self.config.layer_type(layer_idx);

            if layer.decode_verify_multi_own_states() {
                // 2026-10-02: Whatever its `LayerType`; the counters stay aligned for the rest.
                attn_idx += usize::from(layer_type == LayerType::FullAttention);
                ssm_idx += usize::from(layer_type == LayerType::LinearAttention);
                let mut state_refs: Vec<&mut (dyn LayerState + 'static)> = seqs
                    .iter_mut()
                    .map(|s| s.layer_states[layer_idx].as_mut())
                    .collect();
                layer.decode_verify_multi_seqs(
                    hidden,
                    residual,
                    ks,
                    &mut state_refs,
                    kv_cache,
                    &own_lens,
                    &own_bts,
                    ctx,
                    stream,
                )?;
            } else if layer_type == LayerType::FullAttention {
                let mut refs: Vec<&mut (dyn LayerState + 'static)> = attn_dummy_states[attn_idx]
                    .iter_mut()
                    .map(|s| s.as_mut())
                    .collect();
                attn_idx += 1;
                layer.decode_multi_seq(
                    hidden,
                    residual,
                    r_total,
                    &mut refs,
                    kv_cache,
                    seq_lens_vec,
                    block_tables_vec,
                    ctx,
                    stream,
                )?;
            } else {
                let mut wy_slice = DevicePtr::NULL;
                if layer_type == LayerType::LinearAttention {
                    if !wy_tables_base.is_null() {
                        wy_slice = wy_tables_base.offset(
                            ssm_idx * metrale_model_layers::layer::VERIFY_WY_LAYER_STRIDE_BYTES,
                        );
                    }
                    ssm_idx += 1;
                }
                let mut state_refs: Vec<&mut (dyn LayerState + 'static)> = seqs
                    .iter_mut()
                    .map(|s| s.layer_states[layer_idx].as_mut())
                    .collect();
                layer.decode_verify_multi(
                    hidden,
                    residual,
                    n,
                    ks,
                    &mut state_refs,
                    kv_cache,
                    wy_slice,
                    ctx,
                    stream,
                )?;
            }

            // 2026-09-25: DFlash capture: sequence i's rows start at
            // `off[i]` and go to its own band at `i * dflash_kgamma`, the
            // `scratch_row` the scheduler passes to `commit_ctx`.
            // `try_dflash_capture_all_at` returns at once when DFlash is
            // off or this is not a capture layer.
            if self.dflash_hidden_save.is_some() {
                for i in 0..n {
                    self.try_dflash_capture_all_at(
                        layer_idx,
                        off[i],
                        ks[i],
                        i * self.dflash_kgamma,
                        stream,
                    )?;
                }
            }

            if k4_diag && let Err(e) = self.gpu.synchronize(stream) {
                anyhow::bail!(
                    "K4_DIAG(batched): CUDA error after layer {layer_idx} ({layer_type:?}): {e:#}"
                );
            }
        }
        Ok(())
    }

    /// 2026-09-26: Argmax of each of the `r_total` logits rows.
    pub(super) fn verify_rows_argmax(
        &self,
        mapped_argmax: Option<(*mut u8, DevicePtr)>,
        r_total: usize,
        bf16: usize,
        stream: u64,
    ) -> Result<()> {
        let vocab = self.config.vocab_size;
        // 2026-09-25: With a mapped blob (`mapped_argmax_host_dev`) the
        // argmax kernel writes its 4-byte rows straight into page-locked
        // host memory through the blob's device alias, so the step ends
        // with a kernel and no device-to-host copy. Otherwise it writes to
        // scratch and a copy follows.
        let argmax_out = match mapped_argmax {
            Some((_, dev)) => dev,
            None => self.buffers.scratch(),
        };
        // 2026-09-25: One launch with one block per row; one launch per row
        // when the batched kernel is absent.
        if self.argmax_batch_kernel.0 != 0 {
            ops::argmax_bf16_batch(
                self.gpu.as_ref(),
                self.argmax_batch_kernel,
                self.buffers.logits(),
                argmax_out,
                vocab as u32,
                r_total as u32,
                vocab as u32,
                stream,
            )?;
        } else {
            for r in 0..r_total {
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    self.buffers.logits().offset(r * vocab * bf16),
                    argmax_out.offset(r * 4),
                    vocab as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
