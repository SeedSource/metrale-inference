// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Assembles the decode rows' `seq_lens`, `block_tables` and layer states for the fused mixed forward.
//!
//! Owner: model-engine decode.
//! Invariants:
//! - Padding rows never use a claimable SSM slot: they point at `SsmStatePool::dummy_slot`,
//!   which is outside the pool's free list.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;
use metrale_config::LayerType;
use metrale_model_layers::layer::{LayerState, SsmLayerState};

impl TransformerModel {
    /// 2026-09-25: Build the decode portion's `(seq_lens, block_tables, all_layer_states)`
    /// for the fused mixed forward, `padded_n` rows long. Real decode sequences contribute their
    /// own seq_len / block_table / moved-out layer_states; padding slots get
    /// `seq_len=0`, the dummy KV block, and freshly-built dummy layer states
    /// (SSM layers point at the pool's `dummy_slot`).
    pub(super) fn mixed_build_decode_layer_states(
        &self,
        decode_seqs: &mut [&mut SequenceState],
        padded_n: usize,
        n_decode: usize,
    ) -> Result<(Vec<usize>, Vec<Vec<u32>>, Vec<Vec<Box<dyn LayerState>>>)> {
        let seq_lens: Vec<usize> = (0..padded_n)
            .map(|i| {
                if i < n_decode {
                    decode_seqs[i].seq_len
                } else {
                    0
                }
            })
            .collect();
        let block_tables: Vec<Vec<u32>> = (0..padded_n)
            .map(|i| {
                if i < n_decode {
                    decode_seqs[i].block_table.clone()
                } else {
                    vec![self.dummy_kv_block]
                }
            })
            .collect();

        let mut all_layer_states: Vec<Vec<Box<dyn LayerState>>> = decode_seqs
            .iter_mut()
            .map(|s| std::mem::take(&mut s.layer_states))
            .collect();

        // 2026-09-25: Padding rows use the pool's `dummy_slot()`, which `claim_slot` never
        // hands out, so pad SSM kernel writes cannot touch a claimed sequence.
        let dummy_ssm_slot = self.ssm_pool.dummy_slot();
        for _pad_pos in n_decode..padded_n {
            let mut dummy: Vec<Box<dyn LayerState>> = Vec::with_capacity(self.layers.len());
            let mut ssm_idx = 0usize;
            for (li, layer) in self.layers.iter().enumerate() {
                if self.config.layer_type(li) == LayerType::LinearAttention {
                    dummy.push(Box::new(SsmLayerState {
                        kda_defer: None,
                        h_state: self.ssm_pool.h_state(ssm_idx, dummy_ssm_slot),
                        conv_state: self.ssm_pool.conv_state(ssm_idx, dummy_ssm_slot),
                        h_state_checkpoint: None,
                        conv_state_checkpoint: None,
                        h_state_intermediates: Vec::new(),
                        conv_state_intermediates: Vec::new(),
                        // 2026-09-25: Tagged with the active h dtype: under FP16 h-state the
                        // batched decode refuses any row whose state is not tagged FP16.
                        h_is_f16: metrale_model_layers::layers::qwen3_ssm::ssm_h_fp16_enabled(),
                        // 2026-09-25: Padding rows are never prefilled. The staging pointer is
                        // carried so the row matches a real slot of an f16-sized pool.
                        h_prefill_stage: self.ssm_pool.h_prefill_stage(dummy_ssm_slot),
                        ple: None,
                    }));
                    ssm_idx += 1;
                } else {
                    dummy.push(layer.alloc_state(self.gpu.as_ref())?);
                }
            }
            all_layer_states.push(dummy);
        }

        Ok((seq_lens, block_tables, all_layer_states))
    }
}
