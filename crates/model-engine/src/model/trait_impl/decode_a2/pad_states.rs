// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The padding rows' layer states for `decode_batch_compute_main_with`.
//!
//! Owner: model-engine (decode).
//! Invariants: padding rows' SSM layers point at `SsmStatePool::dummy_slot`, which is outside
//! the pool's free list.

use anyhow::Result;
use metrale_config::LayerType;
use metrale_model_layers::layer::{LayerState, SsmLayerState};

use super::super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-09-26: Append one freshly built layer-state row to `all_layer_states` for each
    /// padding position in `n..padded_n`.
    pub(super) fn decode_batch_push_pad_states(
        &self,
        all_layer_states: &mut Vec<Vec<Box<dyn LayerState>>>,
        n: usize,
        padded_n: usize,
    ) -> Result<()> {
        // 2026-09-25: Padding rows use the dedicated `dummy_slot()`, so their SSM
        // writes never land in a claimed sequence's pool slot.
        let dummy_ssm_slot = self.ssm_pool.dummy_slot();
        for _pad_pos in n..padded_n {
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
                        h_is_f16: metrale_model_layers::layers::qwen3_ssm::ssm_h_fp16_enabled(),
                        // 2026-09-25: Never prefilled; set so the dummy row
                        // matches a real row's geometry.
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
        Ok(())
    }
}
