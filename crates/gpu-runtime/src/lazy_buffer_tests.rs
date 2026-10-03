// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: CPU tests for the lazy-buffer mapping schedule and budget, on the eager kind
//! (the mock backend), which runs the same charge/refusal/high-water logic as VMM.

use std::sync::Arc;

use super::*;
use crate::gpu::mock::MockGpuBackend;

const G: usize = DEFAULT_GRANULE;

/// 2026-10-03: The capture set is process-wide, so the test that opens a capture must not
/// overlap the tests that map.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

#[test]
fn granule_rounding() {
    assert_eq!(round_up_to_granule(0, G), 0);
    assert_eq!(round_up_to_granule(1, G), G);
    assert_eq!(round_up_to_granule(G, G), G);
    assert_eq!(round_up_to_granule(G + 1, G), 2 * G);
    assert_eq!(round_up_to_granule(5, 0), 5);
}

#[test]
fn mapped_target_rounds_up_and_caps_at_the_reservation() {
    let reserved = 16 * G;
    assert_eq!(mapped_target(0, reserved, G), 0);
    assert_eq!(mapped_target(1, reserved, G), G);
    assert_eq!(mapped_target(8192 * 256, reserved, G), G); // 8192 rows of 256 B = 1 granule
    assert_eq!(mapped_target(8193 * 256, reserved, G), 2 * G);
    assert_eq!(mapped_target(usize::MAX / 2, reserved, G), reserved);
}

#[test]
fn budget_charges_refunds_and_refuses_without_side_effects() {
    let b = MapBudget::new("test", 3 * G);
    b.try_charge(2 * G).unwrap();
    assert_eq!(b.used(), 2 * G);
    let e = b.try_charge(2 * G).unwrap_err().to_string();
    assert!(e.starts_with("KV cache exhausted"), "{e}");
    assert_eq!(b.used(), 2 * G, "a refused charge changes nothing");
    b.try_charge(G).unwrap();
    assert_eq!(b.available(), 0);
    b.refund(3 * G);
    assert_eq!(b.used(), 0);
    b.refund(G);
    assert_eq!(b.used(), 0, "refund saturates");
}

#[test]
fn ensure_mapped_grows_by_whole_granules_and_never_shrinks() {
    let _serial = SERIAL.lock();
    let gpu = MockGpuBackend::new();
    let budget = Arc::new(MapBudget::new("test", 64 * G));
    let buf = gpu.alloc_lazy(10 * G + 3, Some(budget.clone())).unwrap();
    assert_eq!(buf.reserved_bytes(), 11 * G);
    assert_eq!(buf.mapped_bytes(), 0);
    assert_eq!(budget.used(), 0, "reserving charges nothing");

    buf.ensure_mapped(1).unwrap();
    assert_eq!(buf.mapped_bytes(), G);
    buf.ensure_mapped(G).unwrap();
    assert_eq!(buf.mapped_bytes(), G, "within the high-water mark: no-op");
    buf.ensure_mapped(3 * G + 1).unwrap();
    assert_eq!(buf.mapped_bytes(), 4 * G);
    buf.ensure_mapped(2 * G).unwrap();
    assert_eq!(buf.mapped_bytes(), 4 * G, "a smaller request never shrinks");
    buf.ensure_mapped(usize::MAX).unwrap();
    assert_eq!(buf.mapped_bytes(), 11 * G, "capped at the reservation");
    assert_eq!(budget.used(), 11 * G);

    buf.release(&gpu).unwrap();
    assert_eq!(budget.used(), 0, "release refunds the mapped bytes");
    buf.release(&gpu).unwrap();
    assert_eq!(
        gpu.live_bytes(),
        Some(0),
        "the eager allocation is freed once"
    );
    assert!(buf.ensure_mapped(1).is_err(), "no mapping after release");
}

#[test]
fn a_full_budget_refuses_and_leaves_the_mapping_unchanged() {
    let _serial = SERIAL.lock();
    let gpu = MockGpuBackend::new();
    let budget = Arc::new(MapBudget::new("pool", 3 * G));
    let a = gpu.alloc_lazy(8 * G, Some(budget.clone())).unwrap();
    let b = gpu.alloc_lazy(8 * G, Some(budget.clone())).unwrap();
    a.ensure_mapped(2 * G).unwrap();
    b.ensure_mapped(G).unwrap();
    let e = b.ensure_mapped(2 * G).unwrap_err().to_string();
    assert!(
        e.contains("KV cache exhausted") && e.contains("pool"),
        "{e}"
    );
    assert_eq!(b.mapped_bytes(), G);
    assert_eq!(budget.used(), 3 * G);
    // 2026-10-03: Releasing one buffer makes room for the other.
    a.release(&gpu).unwrap();
    b.ensure_mapped(3 * G).unwrap();
    assert_eq!(budget.used(), 3 * G);
    b.release(&gpu).unwrap();
    assert_eq!(budget.used(), 0);
}

#[test]
fn mapping_is_refused_inside_a_capture() {
    let _serial = SERIAL.lock();
    let gpu = MockGpuBackend::new();
    let buf = gpu.alloc_lazy(4 * G, None).unwrap();
    buf.ensure_mapped(G).unwrap();
    note_capture_begin(0xC0FFEE);
    assert!(capture_active());
    buf.ensure_mapped(G).unwrap(); // already mapped: allowed during capture
    let e = buf.ensure_mapped(2 * G).unwrap_err().to_string();
    assert!(e.contains("inside a stream capture"), "{e}");
    note_capture_end(0xBAD); // not the capturing stream: no change
    assert!(capture_active());
    note_capture_end(0xC0FFEE);
    assert!(!capture_active());
    buf.ensure_mapped(2 * G).unwrap();
    assert_eq!(buf.mapped_bytes(), 2 * G);
    buf.release(&gpu).unwrap();
}
