// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the DSA aux-state codec: round-trip exactness and every
//! refusal.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.
//!
//! CPU-only: `MockGpuBackend` keeps real bytes behind its `DevicePtr`s, so these
//! check the blob layout and the refusals, not how kernels read the rows.

use super::*;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use crate::glm5next_dsa::Glm5NextDsaConfig;
use crate::glm5next_dsa::state::{Glm5NextDsaState, indexer_state_bytes};

/// 2026-09-25: A GLM-5.3-shaped config with a small `max_context`, so the
/// tests stay cheap.
fn cfg(max_context: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context,
    }
}

/// 2026-09-25: Fill the three buffers with a position-dependent pattern and set
/// the cursor to `len`, standing in for `indexer_forward`.
fn ingest(gpu: &MockGpuBackend, st: &mut Glm5NextDsaState, len: usize, salt: u8) {
    let d = st.index_head_dim();
    let keys: Vec<u8> = (0..len * d * 2)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(salt))
        .collect();
    let gates: Vec<u8> = (0..len * d * 2)
        .map(|i| {
            (i as u8)
                .wrapping_mul(17)
                .wrapping_add(salt)
                .wrapping_add(7)
        })
        .collect();
    let valid: Vec<u8> = (0..len).map(|i| ((i % 251) as u8) ^ salt).collect();
    gpu.copy_h2d(&keys, st.k_normed).unwrap();
    gpu.copy_h2d(&gates, st.gate).unwrap();
    gpu.copy_h2d(&valid, st.valid).unwrap();
    st.rewind_to(0).unwrap();
    st.advance(len).unwrap();
}

/// 2026-09-25: Read back the reachable `[0, len)` rows as `(k_normed, gate,
/// valid)`.
fn readback(gpu: &MockGpuBackend, st: &Glm5NextDsaState) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (len, d) = (st.len(), st.index_head_dim());
    let mut k = vec![0u8; len * d * 2];
    let mut g = vec![0u8; len * d * 2];
    let mut v = vec![0u8; len];
    gpu.copy_d2h(st.k_normed, &mut k).unwrap();
    gpu.copy_d2h(st.gate, &mut g).unwrap();
    gpu.copy_d2h(st.valid, &mut v).unwrap();
    (k, g, v)
}

/// 2026-09-25: A snapshot taken at N tokens, restored into a state that another
/// sequence has since overwritten, reproduces all three buffers byte for byte.
#[test]
fn snapshot_mutate_restore_is_byte_exact() {
    let gpu = MockGpuBackend::new();
    let c = cfg(4_096);
    let mut st = Glm5NextDsaState::alloc(&gpu, &c).unwrap();

    ingest(&gpu, &mut st, 1_000, 0xA5);
    let before = readback(&gpu, &st);
    let blob = st.snapshot_blob(&gpu, 0).unwrap();

    // 2026-09-25: Another sequence overwrites this state with different content
    // and length.
    ingest(&gpu, &mut st, 2_048, 0x5A);
    assert_ne!(
        readback(&gpu, &st).0,
        before.0,
        "mutation must actually move"
    );

    st.restore_blob(&blob, &gpu, 0).unwrap();
    assert_eq!(st.len(), 1_000, "cursor restored");
    assert_eq!(
        readback(&gpu, &st),
        before,
        "k_normed / gate / valid all exact"
    );
}

/// 2026-09-25: Round trip at several prefix lengths, including empty and a full
/// reservation.
#[test]
fn round_trip_holds_at_every_prefix_length() {
    let gpu = MockGpuBackend::new();
    let c = cfg(4_096);
    let mut st = Glm5NextDsaState::alloc(&gpu, &c).unwrap();

    for len in [0usize, 1, 4, 255, 256, 1_023, 4_096] {
        ingest(&gpu, &mut st, len, len as u8);
        let want = readback(&gpu, &st);
        let blob = st.snapshot_blob(&gpu, 0).unwrap();
        assert_eq!(
            blob.len(),
            16 + len * (c.index_head_dim * 4 + 1),
            "blob carries len rows, not capacity, at len={len}"
        );

        ingest(&gpu, &mut st, 4_096, 0xFF);
        st.restore_blob(&blob, &gpu, 0).unwrap();
        assert_eq!(st.len(), len);
        assert_eq!(readback(&gpu, &st), want, "exact at len={len}");
    }
}

/// 2026-09-25: The blob size depends on the written prefix, not on the
/// reservation.
#[test]
fn blob_scales_with_prefix_not_with_reservation() {
    let gpu = MockGpuBackend::new();
    let mut small = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    let mut huge = Glm5NextDsaState::alloc(&gpu, &cfg(131_072)).unwrap();
    ingest(&gpu, &mut small, 1_000, 1);
    ingest(&gpu, &mut huge, 1_000, 1);
    assert_eq!(
        small.snapshot_blob(&gpu, 0).unwrap().len(),
        huge.snapshot_blob(&gpu, 0).unwrap().len(),
        "same prefix, same cost, whatever --max-seq-len claims"
    );
    assert!(indexer_state_bytes(huge.capacity(), 128) > indexer_state_bytes(small.capacity(), 128));
}

/// 2026-09-25: A truncated blob is an error and leaves the cursor alone. Cuts
/// inside the header and inside the body are both covered; only the size check
/// catches the second.
#[test]
fn a_truncated_blob_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    ingest(&gpu, &mut st, 64, 3);
    let blob = st.snapshot_blob(&gpu, 0).unwrap();

    for cut in [0usize, 8, 15, 16, blob.len() - 1] {
        let err = st
            .restore_blob(&blob[..cut], &gpu, 0)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("truncated") || err.contains("size mismatch"),
            "cut to {cut} must be refused, got: {err}"
        );
    }
    assert_eq!(
        st.len(),
        64,
        "a refused restore leaves the cursor where it was"
    );
}

/// 2026-09-25: A header with a different `index_head_dim` is refused before the
/// size check, which a different `(len, index_head_dim)` pair of equal byte
/// count would pass.
#[test]
fn a_wrong_geometry_blob_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    ingest(&gpu, &mut st, 32, 9);
    let mut blob = st.snapshot_blob(&gpu, 0).unwrap();
    blob[8..16].copy_from_slice(&64u64.to_le_bytes());

    let err = st.restore_blob(&blob, &gpu, 0).unwrap_err().to_string();
    assert!(err.contains("index_head_dim"), "got: {err}");
    assert_eq!(st.len(), 32, "refused restores do not move the cursor");
}

/// 2026-09-25: A header `len` that disagrees with the body length is refused, in
/// both directions.
#[test]
fn a_len_body_disagreement_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    ingest(&gpu, &mut st, 100, 4);
    let blob = st.snapshot_blob(&gpu, 0).unwrap();

    for claimed in [99u64, 101, 0] {
        let mut bad = blob.clone();
        bad[..8].copy_from_slice(&claimed.to_le_bytes());
        let err = st.restore_blob(&bad, &gpu, 0).unwrap_err().to_string();
        assert!(err.contains("size mismatch"), "claimed {claimed}: {err}");
    }
}

/// 2026-09-25: A snapshot longer than this sequence's reservation is refused, not
/// clamped, and nothing is applied.
#[test]
fn a_blob_longer_than_the_reservation_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut big = Glm5NextDsaState::alloc(&gpu, &cfg(8_192)).unwrap();
    ingest(&gpu, &mut big, 8_192, 2);
    let blob = big.snapshot_blob(&gpu, 0).unwrap();

    let mut small = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    let err = small.restore_blob(&blob, &gpu, 0).unwrap_err().to_string();
    assert!(err.contains("exceeds"), "got: {err}");
    assert_eq!(small.len(), 0, "nothing was applied");
}

/// 2026-09-25: `rewind_to` moves only the cursor: re-advancing over `[n, len)`
/// without rewriting reproduces the original bytes.
#[test]
fn rewind_is_exact_within_a_live_sequence() {
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    ingest(&gpu, &mut st, 500, 11);
    let full = readback(&gpu, &st);

    st.rewind_to(300).unwrap();
    let at_300 = st.snapshot_blob(&gpu, 0).unwrap();
    assert_eq!(at_300.len(), 16 + 300 * (128 * 4 + 1));

    st.advance(200).unwrap();
    assert_eq!(
        readback(&gpu, &st),
        full,
        "rewind moved the cursor, not the rows"
    );

    // 2026-09-25: The 300-token blob's key rows are the first 300 key rows of
    // the 500-token blob.
    let at_500 = st.snapshot_blob(&gpu, 0).unwrap();
    assert_eq!(
        &at_300[16..16 + 300 * 128 * 2],
        &at_500[16..16 + 300 * 128 * 2]
    );
}

/// 2026-09-25: `rewind_to` still refuses to move the cursor forward.
#[test]
fn the_cursor_cannot_be_moved_forward_by_a_restore() {
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    ingest(&gpu, &mut st, 10, 1);
    assert!(st.rewind_to(11).is_err(), "forward rewind is still a bug");
}

/// 2026-10-03: `snapshot_blob_prefix(n)` on a state holding more rows is byte-identical to the
/// `snapshot_blob` of the same state with its cursor at `n`, restores to exactly the first `n`
/// rows, and refuses a prefix longer than the state.
#[test]
fn a_prefix_blob_equals_the_blob_taken_at_that_length() {
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    ingest(&gpu, &mut st, 2_000, 0x3C);
    let full_len = st.len();
    for n in [0usize, 1, 640, 1_999, 2_000] {
        let prefix = st.snapshot_blob_prefix(n, &gpu, 0).unwrap();
        assert_eq!(st.len(), full_len, "taking a prefix leaves the cursor alone");
        st.rewind_to(n).unwrap();
        let at_n = st.snapshot_blob(&gpu, 0).unwrap();
        st.advance(full_len - n).unwrap();
        assert_eq!(prefix, at_n, "prefix blob == blob at cursor {n}");

        let mut other = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
        ingest(&gpu, &mut other, 3_000, 0x77);
        other.restore_blob(&prefix, &gpu, 0).unwrap();
        assert_eq!(other.len(), n);
        assert_eq!(other.snapshot_blob(&gpu, 0).unwrap(), at_n, "round trip at {n}");
    }
    assert_eq!(
        st.snapshot_blob_prefix(full_len, &gpu, 0).unwrap(),
        st.snapshot_blob(&gpu, 0).unwrap(),
        "the full prefix is snapshot_blob"
    );
    assert!(st.snapshot_blob_prefix(full_len + 1, &gpu, 0).is_err());
}
