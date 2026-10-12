// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: CPU tests for `loadpf.rs`: lever parsing, span extents and clipping, the grid,
//! and the fork/join launch shape on the mock backend.

use super::*;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

fn sp(p: u64, b: usize) -> L2Span {
    L2Span {
        ptr: DevicePtr(p),
        bytes: b,
    }
}

/// 2026-10-11: Only a trimmed `1` turns a switch on; CTAs take 1..=48 and nothing else.
#[test]
fn levers_parse() {
    for (raw, want) in [
        (None, false),
        (Some(""), false),
        (Some("0"), false),
        (Some("1"), true),
        (Some(" 1 "), true),
        (Some("on"), false),
        (Some("2"), false),
    ] {
        assert_eq!(parse_switch(raw), want, "{raw:?}");
    }
    for (raw, want) in [
        (None, None),
        (Some(""), None),
        (Some("0"), None),
        (Some("1"), Some(1)),
        (Some(" 8 "), Some(8)),
        (Some("48"), Some(48)),
        (Some("49"), None),
        (Some("-3"), None),
        (Some("x"), None),
    ] {
        assert_eq!(parse_ctas(raw), want, "{raw:?}");
    }
}

/// 2026-10-11: NVFP4 is packed `n k / 2` + E4M3 scales `n k / 16`; FP8 is `n k` + F32 scales
/// per row or per 128x128 block (rounded up). KDA `o_proj` per rank on the car: [4096, 4096].
#[test]
fn span_extents() {
    let [p, s] = nv4_spans(DevicePtr(0x100), DevicePtr(0x200), 4096, 4096);
    assert_eq!((p.ptr, p.bytes), (DevicePtr(0x100), 4096 * 4096 / 2));
    assert_eq!((s.ptr, s.bytes), (DevicePtr(0x200), 4096 * 4096 / 16));
    // 2026-10-11: 9.0 MiB: under the 16 MiB launch budget, so nothing is clipped.
    assert_eq!(p.bytes + s.bytes, 9_437_184);
    assert_eq!(clip_spans(&[p, s], L2_LOADPF_MAX_BYTES), vec![p, s]);
    let [w, r] = fp8_spans(DevicePtr(8), DevicePtr(16), 300, 4096, false);
    assert_eq!((w.bytes, r.bytes), (300 * 4096, 300 * 4));
    let [_, b] = fp8_spans(DevicePtr(8), DevicePtr(16), 300, 4096, true);
    assert_eq!(b.bytes, 3 * 32 * 4);
}

/// 2026-10-11: The grid is the lever's CTAs, clamped to [1, 48], but never more than the
/// sectors need (256 threads x 8 sectors = 64 KiB per CTA).
#[test]
fn grid_follows_ctas_and_sectors() {
    let nine = [sp(1, 9 << 20)];
    assert_eq!(loadpf_grid(&nine, 8), 8);
    assert_eq!(loadpf_grid(&nine, 0), 1);
    assert_eq!(loadpf_grid(&nine, 500), L2_LOADPF_MAX_CTAS);
    assert_eq!(loadpf_grid(&[sp(1, 16)], 8), 1);
    assert_eq!(loadpf_grid(&[sp(1, 64 << 10), sp(2, 64 << 10)], 8), 2);
    assert_eq!(loadpf_grid(&[], 8), 1);
}

fn lane(kernel: u64) -> LoadPfLane {
    LoadPfLane {
        kernel: KernelHandle(kernel),
        side: 7,
        ev_fork: 11,
        ev_join: 12,
    }
}

/// 2026-10-11: A site keeps the clipped spans (budget, 16-byte rounding, null spans dropped)
/// and is absent without a kernel or without spans.
#[test]
fn site_clips_and_needs_kernel_and_spans() {
    let s = LoadPfSite::new(lane(5), &[sp(0, 64), sp(0x40, 20 << 20), sp(0x80, 64)], 8).unwrap();
    assert_eq!(s.spans(), &[sp(0x40, L2_LOADPF_MAX_BYTES)]);
    let s = LoadPfSite::new(lane(5), &[sp(0x40, 1000), sp(0x80, 64)], 8).unwrap();
    assert_eq!(s.spans(), &[sp(0x40, 992), sp(0x80, 64)]);
    assert!(LoadPfSite::new(lane(0), &[sp(0x40, 1000)], 8).is_none());
    assert!(LoadPfSite::new(lane(5), &[sp(0, 1000), sp(0x40, 8)], 8).is_none());
}

/// 2026-10-11: A fork launches once on the SIDE stream with the kernel's argument list (four
/// span slots, unused ones null/0, then a null sink); steps of 0 or more than 16 rows launch
/// nothing and leave nothing to join.
#[test]
fn fork_launches_on_the_side_stream_within_the_row_window() {
    let gpu = MockGpuBackend::new();
    let s = LoadPfSite::new(lane(5), &[sp(0x40, 4096), sp(0x2000, 512)], 3).unwrap();
    assert!(!s.fork(&gpu, 0, 1).unwrap());
    assert!(!s.fork(&gpu, L2_LOADPF_MAX_ROWS + 1, 1).unwrap());
    assert!(gpu.launches_snapshot().is_empty());
    assert!(!s.pending.load(Ordering::Acquire));
    for rows in [1, 3, L2_LOADPF_MAX_ROWS] {
        assert!(s.fork(&gpu, rows, 1).unwrap());
        assert!(s.pending.load(Ordering::Acquire));
        s.join(&gpu, 1).unwrap();
        assert!(!s.pending.load(Ordering::Acquire));
    }
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 3);
    let first = &l[0];
    assert_eq!((first.func, first.stream), (5, 7));
    assert_eq!(first.grid, [1, 1, 1]);
    assert_eq!(first.block, [L2_LOADPF_BLOCK, 1, 1]);
    assert_eq!(first.args.len(), 2 * L2_PREFETCH_MAX_SPANS + 1);
    assert_eq!(first.args[0], MockArg::Buffer(DevicePtr(0x40)));
    assert_eq!(
        first.args[1],
        MockArg::Bytes(4096u64.to_le_bytes().to_vec())
    );
    assert_eq!(first.args[2], MockArg::Buffer(DevicePtr(0x2000)));
    assert_eq!(first.args[4], MockArg::Buffer(DevicePtr(0)));
    assert_eq!(first.args[5], MockArg::Bytes(0u64.to_le_bytes().to_vec()));
    assert_eq!(first.args[8], MockArg::Buffer(DevicePtr(0)));
    // 2026-10-11: A join with nothing pending, and the free functions on `None`, are no-ops.
    s.join(&gpu, 1).unwrap();
    fork(None, &gpu, 1, 1).unwrap();
    join(None, &gpu, 1).unwrap();
    assert_eq!(gpu.launches_snapshot().len(), 3);
}
