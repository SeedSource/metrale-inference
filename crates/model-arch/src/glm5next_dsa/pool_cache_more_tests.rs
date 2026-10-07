// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Tests of the DSA pool cache (`METRALE_GLM_DSA_POOL_CACHE`) added with stage 4
//! and the lead review: the prefix-cache aux blob v2 round trip (on and off the pool grid,
//! prefix cuts, off-grid tails), the v1/v2 format refusals, the drafter context-row compress
//! that keeps a long drafter prefill within the ring, and the select scratch without its pool
//! regions.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.
//!
//! CPU-only (mock backend). The GPU parity of the same paths is
//! `examples/dsa_pool_cache_parity_microtest.rs` (stage 5).

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::*;
use crate::glm5next_dsa::Glm5NextDsaConfig;
use crate::glm5next_dsa::aux_state::V2_TAG;
use crate::glm5next_dsa::state::{Glm5NextDsaState, max_dsa_context};

const R: usize = POOL_CACHE_RING_ROWS;
const KP: usize = 4;
const D: usize = 128;

fn cfg(max_context: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: D,
        index_kpool: KP,
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

/// 2026-10-06: Deterministic filler, distinct per `salt`.
fn bytes(n: usize, salt: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(salt) ^ (i >> 8) as u8)
        .collect()
}

/// 2026-10-06: Write rows `[len, len + n)` as a writer would (ring slots, `valid`), then
/// record a selection when `select` (the compress writes the new pools; here the pool arrays
/// get filler bytes for them).
fn append(gpu: &MockGpuBackend, st: &mut Glm5NextDsaState, n: usize, select: bool, salt: u8) {
    st.ensure_room(n).unwrap();
    let pos0 = st.len();
    for (r0, m) in st.ring_runs(pos0, n) {
        let off = st.row_offset(pos0 + r0);
        gpu.copy_h2d(&bytes(m * D * 2, salt), st.k_normed.offset(off))
            .unwrap();
        gpu.copy_h2d(&bytes(m * D * 2, salt ^ 0x55), st.gate.offset(off))
            .unwrap();
    }
    gpu.copy_h2d(&vec![1u8; n], st.valid.offset(pos0)).unwrap();
    st.advance(n).unwrap();
    if select {
        let pc = st.pool_cache().unwrap();
        let (from, to) = (pc.book.pk_len(), st.len() / KP);
        if to > from {
            gpu.copy_h2d(
                &bytes((to - from) * D * 4, salt),
                pc.pk.offset(from * D * 4),
            )
            .unwrap();
            gpu.copy_h2d(
                &bytes((to - from) * KP * 4, salt),
                pc.pidx.offset(from * KP * 4),
            )
            .unwrap();
            gpu.copy_h2d(&bytes(to - from, salt), pc.pvalid.offset(from))
                .unwrap();
        }
        st.note_selected();
    }
}

/// 2026-10-06: Everything a v2 restore must reproduce: pools `[0, pk)`, the tail rows' ring
/// slots, `valid`, the device and host watermarks, the cursor and the ring floor.
fn observed(gpu: &MockGpuBackend, st: &Glm5NextDsaState) -> (Vec<u8>, usize, usize, usize) {
    let pc = st.pool_cache().unwrap();
    let pk = pc.book.pk_len();
    let read = |p: DevicePtr, n: usize| {
        let mut v = vec![0u8; n];
        gpu.copy_d2h(p, &mut v).unwrap();
        v
    };
    let mut out = read(pc.pk, pk * D * 4);
    out.extend(read(pc.pidx, pk * KP * 4));
    out.extend(read(pc.pvalid, pk));
    for r in KP * pk..st.len() {
        out.extend(read(st.k_normed.offset(st.row_offset(r)), D * 2));
        out.extend(read(st.gate.offset(st.row_offset(r)), D * 2));
    }
    out.extend(read(st.valid, st.len()));
    out.extend(read(pc.pk_len_dev, 4));
    (out, st.len(), pk, pc.book.ring_lo())
}

/// 2026-10-06: Aux v2 round trip, bit-exact, at `len % 4 == 0` and `!= 0`, for a full and a
/// prefix cut, and with an off-grid tail longer than a pool (rows written since the last
/// selection, read from the ring). The restored state then keeps appending and selecting.
#[test]
fn aux_v2_round_trips_bit_exact_on_and_off_the_pool_grid() {
    let gpu = MockGpuBackend::new();
    let c = cfg(65_536);
    // 2026-10-06: (rows written with a selection each 256, rows appended after without one,
    // the prefix cut) -> expected pools and tail rows in the blob.
    for (sel, extra, cut, pk, tail) in [
        (20_000usize, 0usize, None, 5_000usize, 0usize),
        (20_000, 3, None, 5_000, 3),
        (20_000, 256, None, 5_000, 256),
        (20_224, 3, Some(19_999), 4_999, 3),
        (20_224, 0, Some(8_192), 2_048, 0),
    ] {
        let mut a = Glm5NextDsaState::alloc_pool_cache(&gpu, &c, 0, None).unwrap();
        let mut at = 0;
        while at < sel {
            let n = 256.min(sel - at);
            append(&gpu, &mut a, n, true, at as u8);
            at += n;
        }
        if extra > 0 {
            append(&gpu, &mut a, extra, false, 0xA5);
        }
        let rows = cut.unwrap_or(a.len());
        let blob = a.snapshot_blob_prefix(rows, &gpu, 0).unwrap();
        assert_eq!(
            blob.len(),
            a.blob_bytes_for_rows(rows),
            "size agrees ({rows})"
        );
        let word = |i: usize| u64::from_le_bytes(blob[i * 8..i * 8 + 8].try_into().unwrap());
        assert_eq!(word(0), V2_TAG);
        assert_eq!(
            (word(1), word(2), word(3)),
            (rows as u64, D as u64, pk as u64)
        );
        assert_eq!(
            blob.len(),
            32 + pk * (D * 4 + KP * 4 + 1) + tail * D * 4 + rows,
            "layout ({rows})"
        );

        let mut b = Glm5NextDsaState::alloc_pool_cache(&gpu, &c, 0, None).unwrap();
        b.restore_blob(&blob, &gpu, 0).unwrap();
        let (got, len, got_pk, lo) = observed(&gpu, &b);
        assert_eq!(
            (len, got_pk, lo),
            (rows, pk, KP * pk),
            "cursor and book ({rows})"
        );
        assert_eq!(
            &got[got.len() - 4..],
            &(pk as i32).to_le_bytes(),
            "device pk_len"
        );
        assert_eq!(
            b.snapshot_blob(&gpu, 0).unwrap(),
            blob,
            "re-snapshot ({rows})"
        );
        if cut.is_none() {
            // 2026-10-06: All but the device watermark, which only a kernel writes in `a`.
            let (want, ..) = observed(&gpu, &a);
            let n = got.len() - 4;
            assert_eq!(got[..n], want[..n], "bytes ({rows})");
        }
        // 2026-10-06: The restored state goes on: the next compress starts at `pk`.
        assert_eq!(b.pool_select_args().unwrap().unwrap().pk_start, pk);
        for _ in 0..(2 * R / 256) {
            append(&gpu, &mut b, 256, true, 7);
        }
        a.free(&gpu).unwrap();
        b.free(&gpu).unwrap();
    }
}

/// 2026-10-06: A v1 blob is refused by a pool-cache state and a v2 blob by a v1 state, before
/// any copy; a truncated or resized v2 blob is refused; a prefix whose tail left the ring is
/// refused at snapshot.
#[test]
fn aux_formats_do_not_cross_and_bad_v2_blobs_are_refused() {
    let gpu = MockGpuBackend::new();
    let c = cfg(65_536);
    let mut v1 = Glm5NextDsaState::alloc(&gpu, &c).unwrap();
    assert!(!v1.is_pool_cache(), "lever off in tests");
    v1.advance(8).unwrap();
    let blob1 = v1.snapshot_blob(&gpu, 0).unwrap();
    let mut pc = Glm5NextDsaState::alloc_pool_cache(&gpu, &c, 0, None).unwrap();
    append(&gpu, &mut pc, 8, true, 1);
    let blob2 = pc.snapshot_blob(&gpu, 0).unwrap();

    let h = gpu.h2d_count();
    let e = pc.restore_blob(&blob1, &gpu, 0).unwrap_err().to_string();
    assert!(e.contains("v1 but this state uses the pool cache"), "{e}");
    let e = v1.restore_blob(&blob2, &gpu, 0).unwrap_err().to_string();
    assert!(e.contains("v2 (pool cache)"), "{e}");
    let e = pc
        .restore_blob(&blob2[..20], &gpu, 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("truncated"), "{e}");
    let mut longer = blob2.clone();
    longer.push(0);
    assert!(pc.restore_blob(&longer, &gpu, 0).is_err());
    assert_eq!(gpu.h2d_count(), h, "no copy before a refusal");
    assert_eq!((pc.len(), v1.len()), (8, 8), "nothing moved");

    // 2026-10-06: A prefix cut far behind the ring: whole pools are final and travel, but a
    // tail row below the ring floor is gone, so that cut is refused.
    let mut far = Glm5NextDsaState::alloc_pool_cache(&gpu, &c, 0, None).unwrap();
    for _ in 0..(3 * R / 256) {
        append(&gpu, &mut far, 256, true, 2);
    }
    assert!(far.snapshot_blob_prefix(100, &gpu, 0).is_ok());
    let e = far
        .snapshot_blob_prefix(102, &gpu, 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("Refused"), "{e}");
    for st in [v1, pc, far].iter_mut() {
        st.free(&gpu).unwrap();
    }
}

/// 2026-10-06: The v2 blob is about 134 B per token per layer at a grid point (v1: 513).
#[test]
fn aux_v2_size_at_a_grid_point() {
    let gpu = MockGpuBackend::new();
    let c = cfg(65_536);
    let mut st = Glm5NextDsaState::alloc_pool_cache(&gpu, &c, 0, None).unwrap();
    for _ in 0..(16_384 / 256) {
        append(&gpu, &mut st, 256, true, 9);
    }
    let b = st.blob_bytes_for_rows(16_384);
    assert_eq!(b, 32 + 4_096 * (512 + 16 + 1) + 16_384);
    assert!((b as f64 / 16_384.0 - 133.25).abs() < 0.01);
    st.free(&gpu).unwrap();
}

/// 2026-10-06: A drafter prefill longer than the ring: context rows are written in 256-row
/// tiles (or one at a time) with no selection; the compress-only pass after each write keeps
/// the watermark at `len / kpool`, so every `check_write` passes. Without it the ring is full
/// after `R` rows.
#[test]
fn a_drafter_prefill_longer_than_the_ring_passes_check_write() {
    let gpu = MockGpuBackend::new();
    let c = cfg(65_536);
    let mut st = Glm5NextDsaState::alloc_pool_cache(&gpu, &c, 0, None).unwrap();
    let mut written = 0;
    while written < 3 * R + 77 {
        let t = 256.min(3 * R + 77 - written);
        st.ensure_room(t).unwrap();
        st.advance(t).unwrap();
        // 2026-10-06: What `pool_compress_written` does after `write_kv_rows`.
        if st.pool_compress_due() {
            st.pool_select_args().unwrap().unwrap();
            st.note_selected();
        }
        written += t;
    }
    for _ in 0..9 {
        st.ensure_room(1).unwrap();
        st.advance(1).unwrap();
        if st.pool_compress_due() {
            st.note_selected();
        }
    }
    assert!(!st.pool_compress_due());
    assert_eq!(st.pool_cache().unwrap().book.pk_len(), st.len() / KP);
    st.free(&gpu).unwrap();

    let mut no = Glm5NextDsaState::alloc_pool_cache(&gpu, &c, 0, None).unwrap();
    no.advance(R).unwrap();
    assert!(
        no.ensure_room(1).is_err(),
        "without the compress the ring is full"
    );
    no.free(&gpu).unwrap();

    // 2026-10-06: Both context writers end with the compress-only pass, and the batched one
    // splits calls longer than `ring_rows - kpool`.
    let ctx = include_str!("layer/ctx_rows.rs");
    let rows = include_str!("layer/rows.rs");
    assert!(ctx.contains("self.pool_compress_written(gpu, st, stream)"));
    assert!(ctx.contains("pc.book.ring_rows() - pc.book.kpool()"));
    assert!(rows.contains("self.pool_compress_written(gpu, st, stream)"));
}

/// 2026-10-06: `compress_pools_only` is the exact `dsa_kpool_compress_incr` launch a selection
/// makes, from the watermark, with no other kernel.
#[test]
fn the_compress_only_pass_is_one_exact_incremental_launch() {
    use crate::glm5next_dsa::Glm5NextDsaKernels;
    use crate::glm5next_dsa::select::{PoolRows, compress_pools_only};
    let gpu = MockGpuBackend::new();
    let c = cfg(65_536);
    let mut kernels = Glm5NextDsaKernels::resolve(&gpu).unwrap();
    kernels.kpool_compress_incr = KernelHandle(0xBEEF);
    let pc = DsaPoolCacheArgs {
        pk: DevicePtr(0xA000),
        pidx: DevicePtr(0xB000),
        pvalid: DevicePtr(0xC000),
        pk_len_dev: DevicePtr(0xD000),
        pk_start: 2_000,
        ring_rows: R,
    };
    let rows = PoolRows {
        k_normed: DevicePtr(0x1000),
        gate: DevicePtr(0x2000),
        valid: DevicePtr(0x3000),
        ape: DevicePtr(0x4000),
        first_key: 0,
    };
    compress_pools_only(&gpu, &kernels, &c, &rows, &pc, 8_256, 0).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    assert_eq!(l[0].func, 0xBEEF);
    assert_eq!(l[0].grid, [64, 1, 1], "pools [2000, 2064)");
    let u = |v: u32| MockArg::Bytes(v.to_le_bytes().to_vec());
    assert_eq!(l[0].args[7], u(8_256));
    assert_eq!(l[0].args[12], u(2_000));
    assert_eq!(
        l[0].args[14],
        MockArg::Buffer(DevicePtr::NULL),
        "exact: no geom"
    );
    kernels.kpool_compress_incr = KernelHandle(0);
    assert!(compress_pools_only(&gpu, &kernels, &c, &rows, &pc, 8_256, 0).is_err());
}

/// 2026-10-06: With the cache the select scratch plans and allocates no pool regions; a pass
/// without the cache's arrays is then refused; the shared plan shrinks by exactly those bytes.
#[test]
fn the_select_scratch_drops_its_pool_regions_with_the_cache() {
    use crate::glm5next_dsa::select::shared::SharedPlan;
    use crate::glm5next_dsa::select::{DsaSelectGeometry, DsaSelectScratch};
    let c = cfg(557_056);
    let seq = max_dsa_context(&c);
    let g = DsaSelectGeometry::plan(&c, seq, 256).unwrap();
    let (off, t_off) = DsaSelectScratch::plan_bytes_with(&c, &[g], false);
    let (on, t_on) = DsaSelectScratch::plan_bytes_with(&c, &[g], true);
    let pools = seq.div_ceil(KP);
    assert_eq!(off[..3], [pools * D * 4, pools * KP * 4, pools]);
    assert_eq!(on[..3], [0, 0, 0]);
    assert_eq!(on[3..], off[3..]);
    assert_eq!(t_on, t_off);

    let gpu = MockGpuBackend::new();
    let n0 = gpu.live_alloc_count();
    let s = DsaSelectScratch::alloc_sized(&gpu, (on, t_on)).unwrap();
    assert_eq!(
        gpu.live_alloc_count(),
        n0 + 4,
        "scores, candidacy, selected, tokens"
    );
    assert!(s.fits_with(&c, &g, true).is_ok());
    assert!(s.fits_with(&c, &g, false).is_err());
    s.free(&gpu).unwrap();
    assert_eq!(gpu.live_alloc_count(), n0);

    let p_off = SharedPlan::new_with(&c, 256, false).unwrap();
    let p_on = SharedPlan::new_with(&c, 256, true).unwrap();
    let pool_bytes = pools * (D * 4 + KP * 4 + 1);
    assert_eq!(p_off.shared_bytes() - p_on.shared_bytes(), pool_bytes);
    assert_eq!(p_off.layer_bytes - p_on.layer_bytes, pool_bytes);
    assert_eq!(p_off.mtp_bytes - p_on.mtp_bytes, pool_bytes);
}

/// 2026-10-06 (comb21 port): The pool-cache and radix top-k levers plan the select scratch
/// independently: the pool cache zeroes regions 0 to 2 only, the radix lever sizes region 6
/// only, and with both on the scratch allocates scores, candidacy, selected, tokens and the
/// radix work buffer, and still refuses a pass without the cache's arrays.
#[test]
fn the_pool_cache_and_radix_levers_plan_the_scratch_independently() {
    use crate::glm5next_dsa::select::radix::radix_region_bytes;
    use crate::glm5next_dsa::select::{DsaSelectGeometry, DsaSelectScratch};
    let c = cfg(557_056);
    let g = DsaSelectGeometry::plan(&c, max_dsa_context(&c), 1).unwrap();
    let radix = radix_region_bytes(&g, &c, true);
    assert!(radix > 0, "the ceiling pass has more than one top-k tile of pools");
    let plan = |r: bool, pc: bool| DsaSelectScratch::plan_bytes_levers(&c, &[g], r, pc);
    let (off, t_off) = plan(false, false);
    assert_eq!(off[6], 0);
    for (r, pc) in [(true, false), (false, true), (true, true)] {
        let (p, t) = plan(r, pc);
        assert_eq!(t, t_off);
        assert_eq!(p[3..6], off[3..6], "radix {r} pool cache {pc}");
        let pools: &[usize] = if pc { &[0, 0, 0] } else { &off[..3] };
        assert_eq!(&p[..3], pools, "radix {r} pool cache {pc}");
        assert_eq!(p[6], if r { radix } else { 0 }, "radix {r} pool cache {pc}");
    }
    assert_eq!(DsaSelectScratch::plan_bytes_with(&c, &[g], true), plan(false, true));
    assert_eq!(DsaSelectScratch::plan_bytes_for(&c, &[g], true), plan(true, false));

    let gpu = MockGpuBackend::new();
    let n0 = gpu.live_alloc_count();
    let s = DsaSelectScratch::alloc_sized(&gpu, plan(true, true)).unwrap();
    assert_eq!(
        gpu.live_alloc_count(),
        n0 + 5,
        "scores, candidacy, selected, tokens, radix work"
    );
    assert!(s.fits_with(&c, &g, true).is_ok());
    assert!(s.fits_with(&c, &g, false).is_err());
    s.free(&gpu).unwrap();
    assert_eq!(gpu.live_alloc_count(), n0);
}

/// 2026-10-06 (comb21 port): Source guard. In the select launcher the scratch's pool regions
/// appear only in the full-compress (`pool_cache: None`) arm; every scores variant (plain,
/// tiled, tensor-core: one launch with the handle swapped) reads the `pool_keys` /
/// `pool_indices` / `pool_valid` the cache match chose, and so does the expand; the radix
/// top-k, the index split and the grid-stride helpers never name a pool region.
#[test]
fn every_select_launch_reads_the_pools_the_cache_match_chose() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/glm5next_dsa/select");
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
    let src = read("launch.rs");
    let arm = src.find("match inputs.pool_cache {").expect("the cache match");
    let none = src[arm..].find("None => {").expect("the full-compress arm") + arm;
    let end = src[none..].find("if has_pools {").expect("the scores block") + none;
    // 2026-10-07 (comb23, idx merge): the full compress moved into `launch_kpool_compress`,
    // which writes the three scratch regions. It is called from the None arm here and from
    // `pool_once::compress_window`, which bails when the pool cache is on.
    let helper = src.find("pub(super) fn launch_kpool_compress(").expect("the compress helper");
    let helper_end = src[helper..].find(".launch(stream)").unwrap() + helper;
    let hits: Vec<usize> = src.match_indices("scratch.pool_").map(|(i, _)| i).collect();
    assert_eq!(hits.len(), 6, "the helper's three compress outputs and the three returned regions");
    let inside = |i: usize| (none < i && i < end) || (helper < i && i < helper_end);
    assert!(hits.iter().all(|&i| inside(i)), "a scratch pool region outside the None arm");
    let calls: Vec<usize> = src
        .match_indices("launch_kpool_compress(")
        .map(|(i, _)| i)
        .filter(|&i| !(helper..helper_end).contains(&i))
        .collect();
    assert!(!calls.is_empty(), "the None arm compresses through the helper");
    assert!(calls.iter().all(|&i| none < i && i < end), "a compress call outside the None arm");
    let once = read("pool_once.rs");
    let window = &once[once.find("pub fn compress_window(").expect("compress_window")..];
    let bail = window.find("dsa_pool_cache()").expect("compress_window checks the pool cache");
    assert!(bail < window.find("launch_kpool_compress(").unwrap(), "the check precedes the compress");
    let scores = &src[end..src[end..].find(".launch(stream)").unwrap() + end];
    for p in [".arg_ptr(pool_keys)", ".arg_ptr(pool_indices)", ".arg_ptr(pool_valid)"] {
        assert!(scores.contains(p), "scores launch lacks {p}");
    }
    let expand = &src[src.find("kernels.expand_selection").unwrap()..];
    let expand = &expand[..expand.find(".launch(stream)").unwrap()];
    assert!(expand.contains(".arg_ptr(pool_indices)"));
    for f in ["radix.rs", "split.rs", "grid_stride.rs", "shared.rs"] {
        let s = read(f);
        for field in [".pool_keys", ".pool_indices", ".pool_valid", "pool_regions("] {
            assert!(!s.contains(field), "{f} names {field}");
        }
    }
}
