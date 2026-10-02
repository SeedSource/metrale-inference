// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn test_empty_layer_state_downcast() {
    let state: Box<dyn LayerState> = Box::new(EmptyLayerState);
    assert!(state.as_any().downcast_ref::<EmptyLayerState>().is_some());
    assert!(state.as_any().downcast_ref::<SsmLayerState>().is_none());
}

#[test]
fn test_ssm_layer_state_downcast() {
    let state: Box<dyn LayerState> = Box::new(SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    });
    let ssm = state.as_any().downcast_ref::<SsmLayerState>().unwrap();
    assert_eq!(ssm.h_state.0, 0x1000);
    assert_eq!(ssm.conv_state.0, 0x2000);
}

#[test]
fn test_ssm_layer_state_mut() {
    let mut state: Box<dyn LayerState> = Box::new(SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    });
    let ssm = state.as_any_mut().downcast_mut::<SsmLayerState>().unwrap();
    ssm.h_state = DevicePtr(0x3000);
    assert_eq!(ssm.h_state.0, 0x3000);
}

/// 2026-10-01: `AttnMetadataDev::row_view` advances each per-row array by its own element width
/// and leaves null pointers null (the batched GLM decode hands row `i` `row_view(i)`).
#[test]
fn attn_metadata_row_view_uses_each_arrays_own_stride() {
    let m = AttnMetadataDev {
        positions: DevicePtr(0x1000),
        positions_h: DevicePtr(0x1000),
        positions_w: DevicePtr(0x1000),
        slot: DevicePtr(0x2000),
        seq_len: DevicePtr(0x3000),
        block_table: DevicePtr(0x4000),
        max_blocks_per_seq: 7,
        num_seqs: 4,
        seq_slot: DevicePtr(0),
        moe_row_adapter: DevicePtr(0x5000),
    };
    let r = m.row_view(3);
    assert_eq!(r.positions.0, 0x1000 + 3 * 4);
    assert_eq!(r.positions_h.0, 0x1000 + 3 * 4);
    assert_eq!(r.positions_w.0, 0x1000 + 3 * 4);
    assert_eq!(r.slot.0, 0x2000 + 3 * 8);
    assert_eq!(r.seq_len.0, 0x3000 + 3 * 4);
    assert_eq!(r.block_table.0, 0x4000 + 3 * 7 * 4);
    assert_eq!(r.max_blocks_per_seq, 7);
    assert_eq!(r.num_seqs, 1);
    assert_eq!(r.seq_slot.0, 0, "a null array stays null");
    assert_eq!(r.moe_row_adapter.0, 0x5000 + 3 * 4);
    let r0 = m.row_view(0);
    assert_eq!(
        (r0.positions.0, r0.slot.0, r0.block_table.0, r0.num_seqs),
        (0x1000, 0x2000, 0x4000, 4)
    );
    assert_eq!(m.row_view(9).num_seqs, 0, "past the end saturates at zero rows");
}
