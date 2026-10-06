// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Host tests of `METRALE_GLM_DSA_GRID_STRIDE`: the switch and its table entry,
//! the stride-grid arithmetic, a host replay of the kernels' grid-stride walk, and the grids
//! `select_tokens` gives the pool-indexed kernels on the mock backend, for a ceiling and for
//! an exact launch.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::super::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, DsaSelectScratch, ROW_BLOCK, SCORES_BLOCK,
    contiguous_pool_count, select_tokens,
};
use super::*;
use crate::glm5next_dsa::state::max_dsa_context;
use crate::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

/// 2026-10-05: GLM-5.3 DSA geometry (`select/tests.rs`) at `max_context` tokens.
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

/// 2026-10-05: The mock backend's SM count (a GB10's, `kernels/gb10/HARDWARE.toml`).
const MOCK_SMS: usize = 48;

/// 2026-10-05: A `u32` kernel argument as the mock records it.
fn u32_arg(v: usize) -> MockArg {
    MockArg::Bytes((v as u32).to_le_bytes().to_vec())
}

/// 2026-10-05: Dummy device addresses for the selection inputs; the mock launches do not
/// dereference them.
fn inputs(geom_dev: DevicePtr) -> DsaSelectInputs {
    let p = |n: u64| DevicePtr(0x1000 * n);
    DsaSelectInputs {
        k_normed: p(1),
        gate: p(2),
        valid: p(3),
        ape: p(4),
        q: p(5),
        weights: p(6),
        q_pos: p(7),
        q_mask: p(8),
        first_key: 0,
        geom_dev,
    }
}

/// 2026-10-05: One single-row `select_tokens` on a mock GB10: `(geometry, launches in dispatch
/// order)`. Each pass is compress, scores, top-k, expand. `None` is a ceiling launch over the
/// pools of `max_context` tokens (planned at that context, as `Glm5NextDsaWorkspace` does,
/// with a non-null `geom_dev`); `Some(seq)` is an exact launch over `seq` tokens with none.
fn run(max_context: usize, exact_seq: Option<usize>) -> (DsaSelectGeometry, Vec<MockLaunch>) {
    let gpu = MockGpuBackend::new();
    let c = cfg(max_context);
    let biggest = DsaSelectGeometry::plan(&c, max_dsa_context(&c), 1).unwrap();
    let scratch = DsaSelectScratch::alloc(&gpu, &c, &biggest).unwrap();
    let kernels = Glm5NextDsaKernels::resolve(&gpu).unwrap();
    let (geom, geom_dev, launch) = match exact_seq {
        None => {
            let max_pools = contiguous_pool_count(c.index_kpool, max_dsa_context(&c));
            (
                biggest,
                DevicePtr(0x9000),
                DsaSelectLaunch::Ceiling { max_pools },
            )
        }
        Some(seq) => (
            DsaSelectGeometry::plan(&c, seq, 1).unwrap(),
            DevicePtr::NULL,
            DsaSelectLaunch::Exact,
        ),
    };
    select_tokens(
        &gpu,
        &kernels,
        &c,
        &geom,
        &inputs(geom_dev),
        &scratch,
        launch,
        0,
    )
    .unwrap();
    (geom, gpu.launches_snapshot())
}

/// 2026-10-05: The kernels' loop on the host: block `b` of a `grid`-block launch takes pools
/// `b`, `b + grid`, ... below `live`.
fn pools_of_block(b: usize, grid: usize, live: usize) -> impl Iterator<Item = usize> {
    (b..live).step_by(grid)
}

#[test]
fn the_switch_is_on_unless_zero() {
    assert!(parse_grid_stride(None));
    for on in ["", " ", "1", "2"] {
        assert!(parse_grid_stride(Some(on)), "{on:?}");
    }
    for off in ["0", " 0", "0 ", " 0 "] {
        assert!(!parse_grid_stride(Some(off)), "{off:?}");
    }
}

#[test]
fn the_lever_is_declared_default_on_and_read_here() {
    use metrale_config::levers::{Class, Ty, lookup};
    let l = lookup("METRALE_GLM_DSA_GRID_STRIDE").expect("declared in the lever table");
    assert_eq!(l.default, "on");
    assert_eq!((l.ty, l.class), (Ty::Switch, Class::Runtime));
    assert_eq!(
        l.reader,
        "crates/model-arch/src/glm5next_dsa/select/grid_stride.rs"
    );
    assert!(include_str!("grid_stride.rs").contains(l.env));
    // 2026-10-05: The table text names the factor, so it is changed with the constant.
    assert!(
        l.doc
            .contains(&format!("{STRIDE_BLOCKS_PER_SM} blocks per SM")),
        "{}",
        l.doc
    );
}

#[test]
fn the_stride_grid_is_a_few_waves_capped_at_the_ceiling() {
    let g = stride_blocks(MOCK_SMS);
    assert_eq!(g, MOCK_SMS * STRIDE_BLOCKS_PER_SM);
    assert_eq!(stride_blocks(0), STRIDE_BLOCKS_PER_SM, "0 SMs counts as 1");
    // 2026-10-05: At `--max-seq-len` 65,536 and 786,432 (`max_pools` 16,384 and 196,608) the
    // ceiling grids are 16,385 + 16,384 and 196,609 + 196,608 blocks; the stride grid does not
    // depend on the context.
    for m in [16_384usize, 196_608] {
        assert_eq!(ceiling_grids(m, MOCK_SMS, true), (g, g), "m {m}");
        assert_eq!(ceiling_grids(m, MOCK_SMS, false), (m + 1, m), "m {m}");
    }
    // 2026-10-05: A ceiling that already fits in one wave is the earlier grid, lever on or off;
    // `0` stays `0`, as it was.
    for m in [0usize, 1, 37, g - 1] {
        assert_eq!(ceiling_grids(m, MOCK_SMS, true), (m + 1, m), "m {m}");
        assert_eq!(ceiling_grids(m, MOCK_SMS, false), (m + 1, m), "m {m}");
    }
    assert_eq!(ceiling_grids(g, MOCK_SMS, true), (g, g));
    assert_eq!(ceiling_grid_x(g + 1, MOCK_SMS, true), g);
    assert_eq!(ceiling_grid_x(g + 1, MOCK_SMS, false), g + 1);
    assert_eq!(ceiling_grid_x(0, MOCK_SMS, true), 0);
}

/// 2026-10-05: The invariant the kernels rely on: any grid of one block or more takes every
/// live pool exactly once, and a grid of `live` blocks or more is the earlier map, block `b`
/// on pool `b`.
#[test]
fn a_stride_grid_covers_every_live_pool_exactly_once() {
    let g = stride_blocks(MOCK_SMS);
    for live in [0usize, 1, 37, 1_024, 16_384, 196_608] {
        for grid in [1usize, 7, g, g + 1, live.max(1), live + 1, 16_385] {
            let mut hits = vec![0u32; live];
            for b in 0..grid {
                for p in pools_of_block(b, grid, live) {
                    hits[p] += 1;
                }
            }
            assert!(hits.iter().all(|&h| h == 1), "live {live}, grid {grid}");
            if grid >= live {
                for b in 0..live {
                    let taken: Vec<usize> = pools_of_block(b, grid, live).collect();
                    assert_eq!(taken, [b], "live {live}, grid {grid}, block {b}");
                }
            }
        }
    }
}

/// 2026-10-05: A ceiling launch gives compress and the scores the grids `ceiling_grids` names
/// for the lever's state, and changes nothing else: block sizes, shared memory, the scalar
/// arguments (set from the ceiling, so every capture passes the same ones), and the top-k and
/// expand launches.
#[test]
fn a_ceiling_launch_gets_the_stride_grid_and_nothing_else_changes() {
    let stride = dsa_grid_stride();
    for max_context in [65_536usize, 1_024] {
        let (geom, l) = run(max_context, None);
        let c = cfg(max_context);
        let m = contiguous_pool_count(c.index_kpool, max_dsa_context(&c));
        assert_eq!(m, geom.n_pools);
        let (compress_x, scores_x) = ceiling_grids(m, MOCK_SMS, stride);
        assert_eq!(l.len(), 4, "compress, scores, top-k, expand");
        assert_eq!(l[0].grid, [compress_x as u32, 1, 1], "compress, m {m}");
        assert_eq!(l[1].grid, [scores_x as u32, 1, 1], "scores, m {m}");
        assert_eq!(l[0].block, [128, 1, 1]);
        assert_eq!(l[1].block, [SCORES_BLOCK, 1, 1]);
        assert_eq!(l[1].shared_mem, SCORES_BLOCK.max(32 * 4));
        assert_eq!(l[2].grid, [1, 1, 1]);
        assert_eq!(l[3].grid, [1, 1, 1]);
        assert_eq!(l[3].block, [ROW_BLOCK, 1, 1]);
        // 2026-10-05: compress takes S, D, KP after its 7 buffers; the scores take Q, P, H, D,
        // KP, S after their 9.
        let kp = c.index_kpool;
        assert_eq!(l[0].args[7], u32_arg(m * kp));
        assert_eq!(l[0].args[8], u32_arg(128));
        assert_eq!(l[0].args[9], u32_arg(kp));
        assert_eq!(l[1].args[10], u32_arg(m));
        assert_eq!(l[1].args[14], u32_arg(m * kp));
    }
    if stride {
        let (_, l) = run(65_536, None);
        assert_eq!(l[0].grid, [384, 1, 1], "8 blocks x 48 SMs, not 16,385");
        assert_eq!(l[1].grid, [384, 1, 1], "8 blocks x 48 SMs, not 16,384");
    }
}

/// 2026-10-05: An exact launch keeps one block per pool: `n_pools_full` for compress (the
/// trailing partial pool included) and `n_pools` for the scores, whatever the lever says.
#[test]
fn an_exact_launch_keeps_one_block_per_pool() {
    for (seq, full, whole) in [(1_000usize, 250usize, 250usize), (1_002, 251, 250)] {
        let (geom, l) = run(65_536, Some(seq));
        assert_eq!((geom.n_pools_full, geom.n_pools), (full, whole));
        assert_eq!(l.len(), 4);
        assert_eq!(l[0].grid, [full as u32, 1, 1], "seq {seq}");
        assert_eq!(l[1].grid, [whole as u32, 1, 1], "seq {seq}");
    }
}
