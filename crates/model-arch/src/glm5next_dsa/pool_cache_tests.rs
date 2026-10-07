// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Tests of the DSA pool cache (`METRALE_GLM_DSA_POOL_CACHE`), stages 1-3: the
//! lever entry, the ring bookkeeping and its three release-mode checks on both sides of each
//! bound, the state layout on the mock backend (ring offsets, allocation equal to the reserve,
//! lazy mapping, release), the replay pre-check, the reserve and pool sizing, and the launches
//! `select_tokens` makes with the cache set. Stage 4 (aux v2) and the lead-review items are in
//! `pool_cache_more_tests.rs`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.
//!
//! CPU-only. The kernels themselves (`dsa_kpool_compress_incr` and its three helpers) are
//! checked on a GPU by the stage-5 parity microtest, not here.

use std::sync::Arc;

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::lazy_buffer::{DEFAULT_GRANULE, MapBudget};
use metrale_model_layers::layer::LayerState;

use super::*;
use crate::glm5next_dsa::Glm5NextDsaConfig;
use crate::glm5next_dsa::state::{Glm5NextDsaState, max_dsa_context, ring_runs_for};

const R: usize = POOL_CACHE_RING_ROWS;
const KP: usize = 4;

fn cfg(max_context: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
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

/// 2026-10-06: A book after rows `[0, len)` were written in appends of at most `step` rows,
/// each followed by a selection (the decode and prefill pattern).
fn book_after(len: usize, step: usize) -> RingBook {
    let mut b = RingBook::new(R, KP);
    let mut at = 0;
    while at < len {
        let end = (at + step).min(len);
        b.check_write(at, end).unwrap();
        b.note_write(end);
        b.note_compressed(end);
        at = end;
    }
    b
}

#[test]
fn the_lever_is_declared_default_off_and_read_here() {
    use metrale_config::levers::{Class, Ty, lookup};
    let l = lookup("METRALE_GLM_DSA_POOL_CACHE").expect("declared in the lever table");
    assert_eq!(l.default, "off");
    assert_eq!((l.ty, l.class), (Ty::Switch, Class::Runtime));
    assert_eq!(l.reader, "crates/model-arch/src/glm5next_dsa/pool_cache.rs");
    assert!(include_str!("pool_cache.rs").contains(l.env));
    assert!(l.doc.contains("8,448-row ring"), "{}", l.doc);
    assert_eq!(R, 8_448, "the table text names the ring size");
}

#[test]
fn the_ring_holds_a_full_width_window_plus_slack() {
    // 2026-10-06: The widest write before a selection is the 8,192-row full-width window; up to
    // `KP - 1` rows of a partial pool sit below it.
    const { assert!(8_192 + KP - 1 <= R) };
    assert_eq!(ring_rows_for(1 << 20), R);
    assert_eq!(ring_rows_for(4_096), 4_096, "a small cache is its own ring");
    assert_eq!(ring_rows_for(0), 1);
}

/// 2026-10-06: Check 1 (write): `end - kpool * pk_len <= ring_rows`, exactly at the bound and
/// one past it, fresh and after a compress.
#[test]
fn a_write_is_refused_one_row_past_the_ring() {
    let b = RingBook::new(R, KP);
    b.check_write(0, R).unwrap();
    let e = b.check_write(0, R + 1).unwrap_err().to_string();
    assert!(e.contains("pool cache") && e.contains("Refused"), "{e}");

    let b = book_after(10_001, 256);
    assert_eq!(b.pk_len(), 2_500);
    b.check_write(10_001, 10_000 + R).unwrap();
    assert!(b.check_write(10_001, 10_000 + R + 1).is_err());
    // 2026-10-06: The full-width window after a window that ended mid-pool still fits.
    b.check_write(10_001, 10_001 + 8_192).unwrap();
}

/// 2026-10-06: Writes without a selection (drafter context rows) run out of ring after
/// `ring_rows` rows: refused, not wrapped over.
#[test]
fn appends_without_a_selection_are_refused_at_the_ring() {
    let mut b = RingBook::new(R, KP);
    let mut len = 0;
    while len < R {
        b.check_write(len, len + 1).unwrap();
        b.note_write(len + 1);
        len += 1;
    }
    assert!(b.check_write(len, len + 1).is_err());
}

/// 2026-10-06: Check 2 (rewind): the rows the next compress needs, `[kpool * min(pk_len,
/// n / kpool), n)`, must be in the ring. In steady state `ring_lo = len - ring_rows`.
#[test]
fn a_rewind_is_refused_once_its_recompute_rows_left_the_ring() {
    let len = 100_000;
    let b = book_after(len, 1);
    assert_eq!(b.ring_lo(), len - R);
    // 2026-10-06: Deepest legal rewind: the first recomputed pool starts exactly at ring_lo.
    let ok = len - R;
    assert_eq!(ok % KP, 0);
    b.check_rewind(len, ok + 1).unwrap();
    b.check_rewind(len, ok).unwrap();
    // 2026-10-06: One row lower needs pool (ok - 1) / 4, whose first row left the ring.
    let e = b.check_rewind(len, ok - 1).unwrap_err().to_string();
    assert!(e.contains("rewind") && e.contains("Refused"), "{e}");
    // 2026-10-06: A short rewind (a rejected draft) is always fine.
    b.check_rewind(len, len - 3).unwrap();
    // 2026-10-06: Forward is not a rewind.
    assert!(b.check_rewind(len, len + 1).is_err());
}

/// 2026-10-06: A rewind to a pool boundary at or below the watermark needs no rows; the floor
/// drops to the new length, so later appends are judged from there.
#[test]
fn a_rewind_onto_a_pool_boundary_needs_no_rows() {
    let len = 100_000;
    let mut b = book_after(len, 1);
    b.check_rewind(len, 0).unwrap();
    b.note_rewind(0);
    assert_eq!((b.pk_len(), b.ring_lo()), (0, 0));
    b.check_write(0, R).unwrap();
    // 2026-10-06: The same rewind into the middle of a pool is refused (its first rows left).
    let b = book_after(len, 1);
    assert!(b.check_rewind(len, 2).is_err());
}

/// 2026-10-06: Check 3 (read): a compress from `pk_start` reads `[kpool * pk_start, len)`,
/// never below the watermark or the ring floor, never more than the ring.
#[test]
fn a_compress_never_reads_outside_the_watermark_and_the_ring() {
    let mut b = book_after(50_000, 7);
    b.check_read(50_000, b.pk_len()).unwrap();
    assert!(
        b.check_read(50_000, b.pk_len() + 1).is_err(),
        "past the watermark"
    );
    // 2026-10-06: A start below the ring floor is refused.
    let below = b.ring_lo() / KP - 1;
    assert!(b.check_read(50_000, below).is_err());
    // 2026-10-06: After appends that reach the ring bound the read is exactly the ring.
    let first = KP * b.pk_len();
    b.check_write(50_000, first + R).unwrap();
    b.note_write(first + R);
    b.check_read(first + R, b.pk_len()).unwrap();
    assert!(
        b.check_read(first + R + 1, b.pk_len()).is_err(),
        "more rows than the ring"
    );
}

/// 2026-10-06: The device watermark is clamped by a host write only after a rewind.
#[test]
fn the_device_watermark_is_clamped_only_after_a_rewind() {
    let mut b = book_after(100, 1);
    assert_eq!(b.dev_clamp_for(100), None, "append: nothing to do");
    b.check_rewind(100, 90).unwrap();
    b.note_rewind(90);
    assert_eq!(b.dev_clamp_for(90), Some(22));
    b.note_dev_clamp(22);
    assert_eq!(b.dev_clamp_for(90), None);
    b.note_compressed(91);
    assert_eq!(b.dev_clamp_for(91), None);
}

#[test]
fn ring_runs_split_only_at_the_wrap() {
    assert_eq!(ring_runs_for(R, 0, 256), vec![(0, 256)]);
    assert_eq!(ring_runs_for(R, R - 256, 256), vec![(0, 256)]);
    assert_eq!(ring_runs_for(R, R - 100, 256), vec![(0, 100), (100, 156)]);
    assert_eq!(ring_runs_for(R, 3 * R + 5, 8_192), vec![(0, 8_192)]);
    assert_eq!(
        ring_runs_for(R, 3 * R + 300, 8_192),
        vec![(0, 8_148), (8_148, 44)]
    );
    assert_eq!(ring_runs_for(R, 7, 0), vec![(0, 0)]);
}

/// 2026-10-06: The eager layout: what is allocated equals the reserve, row offsets are ring
/// slots, the checks fire through the state API, and release returns every byte.
#[test]
fn the_eager_state_allocates_the_reserve_and_releases_it() {
    let gpu = MockGpuBackend::new();
    let c = cfg(131_072);
    let cap = max_dsa_context(&c);
    let (n0, b0) = (gpu.live_alloc_count(), gpu.live_bytes().unwrap());
    let mut st = Glm5NextDsaState::alloc_pool_cache(&gpu, &c, 0, None).unwrap();
    assert!(st.is_pool_cache());
    // 2026-10-06: pk, pidx, pvalid, pk_len_dev, the two rings, valid.
    assert_eq!(gpu.live_alloc_count(), n0 + 7);
    assert_eq!(
        gpu.live_bytes().unwrap() - b0,
        pool_cache_state_bytes(cap, 128, KP)
    );
    assert_eq!(st.row_offset(0), 0);
    assert_eq!(st.row_offset(R + 3), 3 * 256);
    assert_eq!(st.mapped_rows(), None);

    // 2026-10-06: Appends with a selection after each keep going past the ring.
    for _ in 0..(2 * R / 256) {
        st.ensure_room(256).unwrap();
        st.advance(256).unwrap();
        st.pool_select_args().unwrap().expect("pool cache on");
        st.note_selected();
    }
    let args = st.pool_select_args().unwrap().unwrap();
    assert_eq!(args.pk_start, st.len() / KP);
    assert_eq!(args.ring_rows, R);
    // 2026-10-06: Appends without a selection stop at the ring, before anything moves.
    let len = st.len();
    st.ensure_room(R).unwrap();
    assert!(st.ensure_room(R + 1).is_err());
    assert!(st.advance(R + 1).is_err());
    assert_eq!(st.len(), len);
    // 2026-10-06: A rejected draft rewinds fine; a rewind deeper than the ring is refused and
    // moves nothing.
    st.advance(3).unwrap();
    st.rewind_to(len + 1).unwrap();
    assert!(st.rewind_to(len - R - 1).is_err());
    assert_eq!(st.len(), len + 1);

    st.free(&gpu).unwrap();
    assert_eq!(gpu.live_alloc_count(), n0);
    assert_eq!(gpu.live_bytes().unwrap(), b0);
    st.free(&gpu).unwrap();
}

/// 2026-10-06: Lazily mapped `pk`/`pidx`: nothing mapped at creation, whole pools as rows are
/// made room for, charged to the pool, refunded on release.
#[test]
fn the_lazy_state_maps_pool_arrays_on_demand() {
    let gpu = MockGpuBackend::new();
    let pool = Arc::new(MapBudget::new("test", usize::MAX));
    let c = cfg(1 << 20);
    let mut st = Glm5NextDsaState::alloc_pool_cache(&gpu, &c, 0, Some(pool.clone())).unwrap();
    assert_eq!(st.mapped_rows(), Some(0));
    assert_eq!(pool.used(), 0);
    st.ensure_room(1).unwrap();
    // 2026-10-06: One pk granule holds 4,096 pools (16,384 rows); one pidx granule 131,072.
    assert_eq!(st.mapped_rows(), Some(16_384));
    assert_eq!(pool.used(), 2 * DEFAULT_GRANULE);
    assert_eq!(st.rows_backed_through(), Some(16_384));
    st.map_rows_through(16_385).unwrap();
    assert_eq!(st.mapped_rows(), Some(32_768));
    st.free(&gpu).unwrap();
    assert_eq!(pool.used(), 0);
}

/// 2026-10-06: The replay pre-check runs the rewind and write checks the replay will meet,
/// and `sync_to` leaves the watermark where the replayed compress left the device's.
#[test]
fn the_replay_pre_check_and_sync_follow_the_ring() {
    let gpu = MockGpuBackend::new();
    let c = cfg(131_072);
    let mut st = Glm5NextDsaState::alloc_pool_cache(&gpu, &c, 0, None).unwrap();
    st.advance(20_000).unwrap_err();
    for _ in 0..(20_000 / 250) {
        st.advance(250).unwrap();
        st.note_selected();
    }
    // 2026-10-06: A verify of 4 after a rejected draft of 3: rewind 3, write 4.
    st.advance(3).unwrap();
    st.replay_room(20_000, 4).unwrap();
    st.sync_to(20_000, 4).unwrap();
    assert_eq!(st.len(), 20_004);
    let args = st.pool_select_args().unwrap().unwrap();
    assert_eq!(args.pk_start, 20_004 / KP);
    // 2026-10-06: A replay rewinding past the ring is refused before the launch.
    // 2026-10-06: (A rewind onto a pool boundary needs no ring rows, so the refused case is
    // one row past one.)
    assert!(st.replay_room(20_004 - R - 5, 1).is_err());
    st.replay_room(20_004 - R - 4, 1).unwrap();
    st.free(&gpu).unwrap();
}

/// 2026-10-06: `select_tokens` with the cache set: the incremental compress from the
/// watermark, then scores, top-k and expand on the persistent arrays (the scratch's pool
/// regions untouched).
#[test]
fn select_tokens_compresses_from_the_watermark_into_the_persistent_arrays() {
    use crate::glm5next_dsa::Glm5NextDsaKernels;
    use crate::glm5next_dsa::select::{
        DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, DsaSelectScratch, select_tokens,
    };
    let gpu = MockGpuBackend::new();
    let c = cfg(16_384);
    let big = DsaSelectGeometry::plan(&c, max_dsa_context(&c), 1).unwrap();
    let scratch = DsaSelectScratch::alloc(&gpu, &c, &big).unwrap();
    let mut kernels = Glm5NextDsaKernels::resolve(&gpu).unwrap();
    let incr = KernelHandle(0xBEEF);
    kernels.kpool_compress_incr = incr;
    let pc = DsaPoolCacheArgs {
        pk: DevicePtr(0xA000),
        pidx: DevicePtr(0xB000),
        pvalid: DevicePtr(0xC000),
        pk_len_dev: DevicePtr(0xD000),
        pk_start: 2_000,
        ring_rows: R,
    };
    let p = |n: u64| DevicePtr(0x1000 * n);
    let inputs = DsaSelectInputs {
        k_normed: p(1),
        gate: p(2),
        valid: p(3),
        ape: p(4),
        q: p(5),
        weights: p(6),
        q_pos: p(7),
        q_mask: p(8),
        first_key: 0,
        geom_dev: DevicePtr::NULL,
        pool_cache: Some(pc),
    };
    let geom = DsaSelectGeometry::plan(&c, 8_006, 1).unwrap();
    let exact = DsaSelectLaunch::Exact;
    select_tokens(&gpu, &kernels, &c, &geom, &inputs, &scratch, exact, 0).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 4, "compress, scores, top-k, expand");
    assert_eq!(l[0].func, incr.0);
    // 2026-10-06: Pools [2000, 2002): 8,006 rows are 2,001 complete pools plus a partial one.
    assert_eq!(l[0].grid, [2, 1, 1]);
    let u = |v: u32| MockArg::Bytes(v.to_le_bytes().to_vec());
    assert_eq!(l[0].args[4], MockArg::Buffer(pc.pk));
    assert_eq!(l[0].args[5], MockArg::Buffer(pc.pidx));
    assert_eq!(l[0].args[6], MockArg::Buffer(pc.pvalid));
    assert_eq!(l[0].args[7], u(8_006));
    assert_eq!(l[0].args[11], u(R as u32));
    assert_eq!(l[0].args[12], u(2_000));
    assert_eq!(l[0].args[13], MockArg::Buffer(pc.pk_len_dev));
    // 2026-10-06: Scores read the persistent keys, ids and validity; expand the ids.
    assert_eq!(l[1].args[1], MockArg::Buffer(pc.pk));
    assert_eq!(l[1].args[3], MockArg::Buffer(pc.pidx));
    assert_eq!(l[1].args[4], MockArg::Buffer(pc.pvalid));
    assert_eq!(l[3].args[1], MockArg::Buffer(pc.pidx));

    // 2026-10-06: Nothing due (watermark at the last complete pool, length on a pool
    // boundary): still one block, which publishes the device watermark.
    let at = DsaPoolCacheArgs {
        pk_start: 2_000,
        ..pc
    };
    let geom = DsaSelectGeometry::plan(&c, 8_000, 1).unwrap();
    let inputs = DsaSelectInputs {
        pool_cache: Some(at),
        ..inputs
    };
    select_tokens(&gpu, &kernels, &c, &geom, &inputs, &scratch, exact, 0).unwrap();
    assert_eq!(gpu.launches_snapshot()[4].grid, [1, 1, 1]);

    // 2026-10-06: Refused before any launch: the kernel absent, or a nonzero first key.
    let n = gpu.launches_snapshot().len();
    let mut no_incr = kernels;
    no_incr.kpool_compress_incr = KernelHandle(0);
    let r = select_tokens(&gpu, &no_incr, &c, &geom, &inputs, &scratch, exact, 0);
    assert!(r.is_err());
    let shifted = DsaSelectInputs {
        first_key: 1,
        ..inputs
    };
    let r = select_tokens(&gpu, &kernels, &c, &geom, &shifted, &scratch, exact, 0);
    assert!(r.is_err());
    assert_eq!(gpu.launches_snapshot().len(), n);
}

/// 2026-10-06: The kernel file defines every pool-cache entry point the host resolves, and the
/// incremental compress keeps `dsa_kpool_compress`'s per-pool arithmetic.
#[test]
fn the_kernel_file_defines_the_pool_cache_kernels() {
    let cu = include_str!("../../../../kernels/gb10/common/dsa_indexer.cu");
    for k in [
        "dsa_kpool_compress_incr",
        "dsa_write_geom_pk",
        "dsa_indexer_store_ring",
        "dsa_pk_len_clamp",
    ] {
        assert!(cu.contains(&format!("__global__ void {k}(")), "{k}");
    }
    let body = |name: &str| {
        let i = cu.find(&format!("__global__ void {name}(")).unwrap();
        let j = i + cu[i..].find("pool_keys[p * D + d] = acc;").unwrap();
        cu[i..j].to_string()
    };
    let (full, incr) = (body("dsa_kpool_compress"), body("dsa_kpool_compress_incr"));
    for line in [
        "lg[s] = (lg[s] == -CUDART_INF_F) ? 0.0f : __expf(lg[s] - mx);",
        "float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;",
        "mx = fmaxf(mx, lg[s]);",
        "if (tid == 0) pool_indices[p * KP + s] = ok ? (int)raw : DSA_INVALID;",
        "if (tid == 0) pool_valid[p] = all_valid ? 1 : 0;",
    ] {
        assert!(full.contains(line) && incr.contains(line), "{line}");
    }
    assert!(incr.contains("raw % ring_rows"));
    assert!(incr.contains("pk_start + blockIdx.x"));
}

/// 2026-10-06: The reserve with the cache: per token and layer 133.25 B of pool arrays and
/// `valid` (was 513), plus the per-sequence rings; lever off unchanged.
#[test]
fn the_reserve_charges_pool_arrays_and_rings() {
    let cap = 540_672;
    let per = pool_cache_state_bytes(cap, 128, KP);
    let rings = 2 * R * 128 * 2 + 4;
    assert_eq!(per - rings, cap / 4 * (512 + 16 + 1) + cap);
    assert_eq!((per - rings) * 4, cap * 533);
    assert_eq!(rings, 4_325_380);
    // 2026-10-06: 12 caches (11 DSA + MTP) at 540,672 rows: 0.85 GiB against 3.10 GiB.
    let old = 12 * crate::glm5next_dsa::state::indexer_state_bytes(cap, 128);
    assert_eq!(old, 3_328_376_832);
    assert_eq!(12 * per, 916_439_088);
}

/// 2026-10-06: The lazily mapped pool with the cache: pool keys map per 16,384 tokens and ids
/// per 524,288, so a sequence of `t` tokens takes `ceil(t / 16,384) + ceil(t / 524,288)`
/// granules per cache, and `free_tokens` inverts that exactly.
#[test]
fn the_lazy_shape_with_the_cache_counts_pool_granules() {
    use crate::glm5next_dsa::lazy::LazyShape;
    let g = DEFAULT_GRANULE;
    let shape = LazyShape {
        dsa_layers: 11,
        proposer: false,
        index_head_dim: 128,
        capacity: 1 << 20,
        pool_cache: true,
        index_kpool: KP,
    };
    assert_eq!(shape.seq_mapped_bytes(1, g), 11 * 2 * g);
    assert_eq!(shape.seq_mapped_bytes(16_384, g), 11 * 2 * g);
    assert_eq!(shape.seq_mapped_bytes(16_385, g), 11 * 3 * g);
    assert_eq!(shape.seq_mapped_bytes(524_288, g), 11 * 33 * g);
    assert_eq!(shape.seq_mapped_bytes(524_289, g), 11 * 35 * g);
    // 2026-10-06: Budgets below the capacity (`free_tokens` does not cap at it, lever off
    // neither).
    for granules in [0usize, 1, 2, 3, 32, 33, 34, 35] {
        let t = shape.free_tokens(11 * granules * g, g);
        assert!(
            t == 0 || shape.seq_mapped_bytes(t, g) <= 11 * granules * g,
            "{granules}: {t}"
        );
        let more = shape.seq_mapped_bytes(t + 16_384, g);
        assert!(more > 11 * granules * g, "{granules}: {t} is the most");
    }
    assert_eq!(shape.free_tokens(11 * 33 * g, g), 524_288);
    let off = LazyShape {
        pool_cache: false,
        ..shape
    };
    assert_eq!(off.free_tokens(11 * 2 * g, g), 8_192, "lever off unchanged");
}
