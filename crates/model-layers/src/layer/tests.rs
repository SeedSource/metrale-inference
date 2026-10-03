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

/// 2026-10-03: `inpass_split_row`: the one call whose positions end at or after the capture
/// point and start before it gets the split; the rows before it are `cut - seq_len`.
#[test]
fn inpass_split_row_places_the_cut_in_exactly_one_call() {
    assert_eq!(inpass_split_row(640, 512, 256), Some(128));
    assert_eq!(inpass_split_row(768, 512, 256), Some(256), "cut at the call's end");
    assert_eq!(inpass_split_row(512, 512, 256), None, "cut at the call's start");
    assert_eq!(inpass_split_row(769, 512, 256), None);
    assert_eq!(inpass_split_row(1, 0, 1), Some(1));
    assert_eq!(inpass_split_row(5, 3, 0), None, "an empty call never splits");
    // 2026-10-03: Sub-chunks tiling a pass from position 100: exactly one call splits.
    for cut in 101..=100 + 1000 {
        let hits: Vec<usize> = (0..1000)
            .step_by(256)
            .filter_map(|t| inpass_split_row(cut, 100 + t, 256usize.min(1000 - t)))
            .collect();
        assert_eq!(hits.len(), 1, "cut {cut}");
    }
}

/// 2026-10-03: `MidchunkCapture::inpass_split` finds the layer's ordinal by its live h_state
/// address, and declines a tail mid-chunk plan (empty `live_h`) or an unknown address.
#[test]
fn inpass_split_matches_the_layer_by_pool_address() {
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let captured = std::sync::atomic::AtomicUsize::new(0);
    let h_dsts = [DevicePtr(0xA000), DevicePtr(0xB000)];
    let conv_dsts = [DevicePtr(0xC000), DevicePtr(0xD000)];
    let live = [DevicePtr(0x1000), DevicePtr(0x2000)];
    let cap = MidchunkCapture {
        cap_local: 300,
        h_dsts: &h_dsts,
        conv_dsts: &conv_dsts,
        h_bytes: 64,
        conv_bytes: 16,
        ssm_layer_counter: &counter,
        cap_local_early: None,
        h_dsts_early: &[],
        conv_dsts_early: &[],
        seq_pos_start: 40,
        live_h: &live,
        captured: &captured,
    };
    // 2026-10-03: Capture point 340; the call over [256, 512) holds it at row 84.
    let s = cap.inpass_split(256, 256, DevicePtr(0x2000)).unwrap();
    assert_eq!(
        s,
        InpassSplit {
            row: 84,
            h_dst: DevicePtr(0xB000),
            conv_dst: DevicePtr(0xD000),
            h_bytes: 64,
            conv_bytes: 16,
        }
    );
    assert!(cap.inpass_split(0, 256, DevicePtr(0x2000)).is_none());
    assert!(cap.inpass_split(256, 256, DevicePtr(0x3000)).is_none());
    let tail_plan = MidchunkCapture {
        live_h: &[],
        ..cap
    };
    assert!(tail_plan.inpass_split(256, 256, DevicePtr(0x2000)).is_none());
}
