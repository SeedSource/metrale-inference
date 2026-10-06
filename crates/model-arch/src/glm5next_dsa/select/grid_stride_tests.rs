// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Host tests of `METRALE_GLM_DSA_GRID_STRIDE`: the switch and its table entry,
//! the stride-grid arithmetic, a host replay of the kernels' grid-stride walk, the marker gate
//! (`dsa_indexer_grid_stride_v1`: which kernel files define it, when it is looked up, the mode
//! and its log line), and the grids `select_tokens` gives the pool-indexed kernels on the mock
//! backend, for a ceiling launch with and without the marker and for an exact launch.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::super::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, DsaSelectScratch, ROW_BLOCK, SCORES_BLOCK,
    contiguous_pool_count, select_tokens,
};
use super::*;
use crate::glm5next_dsa::state::max_dsa_context;
use crate::glm5next_dsa::{DSA_MODULE, Glm5NextDsaConfig, Glm5NextDsaKernels};
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
/// `marker` is whether the kernel module defines `dsa_indexer_grid_stride_v1`: `false` denies
/// that lookup, as a module without the stride loop would.
fn run(
    max_context: usize,
    exact_seq: Option<usize>,
    marker: bool,
) -> (DsaSelectGeometry, Vec<MockLaunch>) {
    let gpu = MockGpuBackend::new();
    if !marker {
        gpu.deny_kernel(DSA_MODULE, GRID_STRIDE_MARKER);
    }
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

/// 2026-10-05: With the marker resolved, a ceiling launch gives compress and the scores the
/// grids `ceiling_grids` names for the lever's state, and changes nothing else: block sizes,
/// shared memory, the scalar arguments (set from the ceiling, so every capture passes the same
/// ones), and the top-k and expand launches.
#[test]
fn a_ceiling_launch_gets_the_stride_grid_and_nothing_else_changes() {
    let stride = dsa_grid_stride();
    for max_context in [65_536usize, 1_024] {
        let (geom, l) = run(max_context, None, true);
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
        let (_, l) = run(65_536, None, true);
        assert_eq!(l[0].grid, [384, 1, 1], "8 blocks x 48 SMs, not 16,385");
        assert_eq!(l[1].grid, [384, 1, 1], "8 blocks x 48 SMs, not 16,384");
    }
}

/// 2026-10-05: A module without `dsa_indexer_grid_stride_v1` has kernels that are one block per
/// pool, so a ceiling launch gets the ceiling grids (`m + 1` for compress, `m` for the scores)
/// with the lever on or off, and everything else about the launches is as with the marker.
#[test]
fn a_ceiling_launch_without_the_marker_gets_the_ceiling_grids() {
    for max_context in [65_536usize, 1_024] {
        let c = cfg(max_context);
        let m = contiguous_pool_count(c.index_kpool, max_dsa_context(&c));
        let (_, l) = run(max_context, None, false);
        let (_, with_marker) = run(max_context, None, true);
        assert_eq!(l.len(), 4, "compress, scores, top-k, expand");
        assert_eq!(l[0].grid, [(m + 1) as u32, 1, 1], "compress, m {m}");
        assert_eq!(l[1].grid, [m as u32, 1, 1], "scores, m {m}");
        for (i, (a, b)) in l.iter().zip(&with_marker).enumerate() {
            assert_eq!(
                (a.block, a.shared_mem),
                (b.block, b.shared_mem),
                "launch {i}"
            );
            assert_eq!(a.args, b.args, "launch {i}");
            if i >= 2 {
                assert_eq!(a.grid, b.grid, "launch {i}");
            }
        }
    }
}

/// 2026-10-05: An exact launch keeps one block per pool: `n_pools_full` for compress (the
/// trailing partial pool included) and `n_pools` for the scores, whatever the lever says.
#[test]
fn an_exact_launch_keeps_one_block_per_pool() {
    for (seq, full, whole) in [(1_000usize, 250usize, 250usize), (1_002, 251, 250)] {
        for marker in [true, false] {
            let (geom, l) = run(65_536, Some(seq), marker);
            assert_eq!((geom.n_pools_full, geom.n_pools), (full, whole));
            assert_eq!(l.len(), 4);
            assert_eq!(l[0].grid, [full as u32, 1, 1], "seq {seq}, marker {marker}");
            assert_eq!(
                l[1].grid,
                [whole as u32, 1, 1],
                "seq {seq}, marker {marker}"
            );
        }
    }
}

/// 2026-10-05: The stride grid needs the lever AND the marker; any other state is the
/// one-block-per-ceiling-pool grids. Each lever state is passed in, since the process's own
/// is read once from the environment.
#[test]
fn the_stride_grid_needs_the_lever_and_the_marker() {
    let gpu = MockGpuBackend::new();
    let g = stride_blocks(MOCK_SMS);
    for m in [16_384usize, 196_608] {
        for (lever, marker, want) in [
            (true, true, (g, g)),
            (true, false, (m + 1, m)),
            (false, true, (m + 1, m)),
            (false, false, (m + 1, m)),
        ] {
            assert_eq!(
                ceiling_launch_grids_for(lever, &gpu, marker, m),
                want,
                "lever {lever}, marker {marker}, m {m}"
            );
        }
    }
}

/// 2026-10-05: The mode, and the one line logged for it on the first ceiling launch.
#[test]
fn the_mode_and_the_log_line_follow_the_lever_and_the_marker() {
    let g = stride_blocks(MOCK_SMS);
    assert_eq!(stride_mode(true, true), StrideMode::Engaged);
    assert_eq!(stride_mode(true, false), StrideMode::MarkerMissing);
    assert_eq!(stride_mode(false, true), StrideMode::LeverOff);
    assert_eq!(stride_mode(false, false), StrideMode::LeverOff);
    assert!(StrideMode::Engaged.engaged());
    assert!(!StrideMode::LeverOff.engaged() && !StrideMode::MarkerMissing.engaged());
    let line = |lever, marker| grid_stride_log_line(stride_mode(lever, marker), g);
    assert_eq!(line(true, true), "GRID_STRIDE: ENGAGED (G=384)");
    assert_eq!(line(false, true), "GRID_STRIDE: OFF (lever=0)");
    assert_eq!(line(false, false), "GRID_STRIDE: OFF (lever=0)");
    assert_eq!(
        line(true, false),
        "GRID_STRIDE: OFF (kernel module lacks dsa_indexer_grid_stride_v1)"
    );
}

/// 2026-10-05: The marker is looked up when the kernels are resolved, not at the first
/// launch: a lookup that fails after the boot audit seals aborts the process. It is optional:
/// a module without it still resolves, with a zero handle.
#[test]
fn the_marker_is_resolved_with_the_kernels_and_is_optional() {
    let gpu = MockGpuBackend::new();
    let k = Glm5NextDsaKernels::resolve(&gpu).unwrap();
    assert_ne!(k.grid_stride_marker.0, 0);
    let want = (DSA_MODULE.to_string(), GRID_STRIDE_MARKER.to_string());
    assert!(gpu.kernel_lookups_snapshot().contains(&want));

    let gpu = MockGpuBackend::new();
    gpu.deny_kernel(DSA_MODULE, GRID_STRIDE_MARKER);
    let k = Glm5NextDsaKernels::resolve(&gpu).expect("a module without the marker resolves");
    assert_eq!(k.grid_stride_marker.0, 0);
    assert_ne!(k.kpool_compress.0, 0);
    assert_ne!(k.index_scores.0, 0);
}

/// 2026-10-05: `.cu` text with `//` comments removed and whitespace runs collapsed to one
/// space, so prose that names a kernel or the loop is not read as code.
fn code_of(src: &str) -> String {
    src.lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .flat_map(str::split_whitespace)
        .collect::<Vec<_>>()
        .join(" ")
        .replace("for(", "for (")
}

/// 2026-10-05: Whether `body` holds a `for` whose index starts at `blockIdx.x` and steps by
/// `gridDim.x`: the walk over the live pools.
fn has_stride_loop(body: &str) -> bool {
    body.split("for (").skip(1).any(|rest| {
        let mut parts = rest.splitn(3, ';');
        match (parts.next(), parts.next(), parts.next()) {
            (Some(init), Some(_), Some(step)) => {
                init.contains("= blockIdx.x")
                    && step
                        .split(')')
                        .next()
                        .is_some_and(|s| s.contains("+= gridDim.x"))
            }
            _ => false,
        }
    })
}

/// 2026-10-05: What a `dsa_indexer.cu` declares: `(dsa_kpool_compress has the stride loop,
/// dsa_index_scores has it, the marker is defined)`. A kernel's text runs to the next entry
/// point. Panics on a missing kernel, so a rename fails the scan.
fn stride_facts(src: &str) -> (bool, bool, bool) {
    let code = code_of(src);
    let has_loop = |kernel: &str| {
        let head = format!("__global__ void {kernel}(");
        let at = code.find(&head).unwrap_or_else(|| panic!("no `{head}`"));
        let rest = &code[at + head.len()..];
        has_stride_loop(rest.split("__global__").next().unwrap_or(rest))
    };
    (
        has_loop("dsa_kpool_compress"),
        has_loop("dsa_index_scores"),
        code.contains(&format!("__global__ void {GRID_STRIDE_MARKER}(")),
    )
}

/// 2026-10-05: The host launches the capped stride grid when the module defines the marker, so
/// a `dsa_indexer.cu` defines it iff `dsa_kpool_compress` and `dsa_index_scores` both carry the
/// loop. A copy without the loop that defined it would run on a grid that covers the first
/// few pools only; one with both loops that did not would never get the stride grid. Every
/// hardware tree's copy is read: b300 forks the file and has neither today.
#[test]
fn every_dsa_indexer_defines_the_marker_iff_both_kernels_carry_the_loop() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    let mut seen = Vec::new();
    for hw in std::fs::read_dir(&root).expect("kernels/ readable") {
        let hw = hw.expect("a kernels/ entry").path();
        let cu = hw.join("common/dsa_indexer.cu");
        if !cu.is_file() {
            continue;
        }
        let src = std::fs::read_to_string(&cu).unwrap_or_else(|e| panic!("{cu:?}: {e}"));
        let (compress, scores, marker) = stride_facts(&src);
        assert_eq!(
            marker,
            compress && scores,
            "{cu:?}: marker defined {marker}, stride loop in dsa_kpool_compress {compress}, \
             in dsa_index_scores {scores}"
        );
        seen.push((
            hw.file_name().unwrap().to_string_lossy().into_owned(),
            marker,
        ));
    }
    // 2026-10-05: gb10 is the origin and carries the loops; a scan that missed it, or found
    // its marker gone, would leave the stride grid off everywhere without failing.
    assert!(
        seen.contains(&("gb10".to_string(), true)),
        "gb10's dsa_indexer.cu must carry both loops and the marker: {seen:?}"
    );
}

/// 2026-10-05: A `.cu` text with the two kernels, each with or without the loop, and the
/// marker or not. Without the loop, the loop is still written in a comment, and the marker is
/// written in a comment either way; neither may count.
fn synthetic_cu(compress_loop: bool, scores_loop: bool, marker: bool) -> String {
    let body = |stride: bool| {
        if stride {
            "for (unsigned int p = blockIdx.x; p < live; p += gridDim.x) { work(p); }"
        } else {
            "unsigned int p = blockIdx.x; // for (unsigned int p = blockIdx.x; p < n; p += \
             gridDim.x)\n work(p);"
        }
    };
    let define = if marker {
        format!("extern \"C\" __global__ void {GRID_STRIDE_MARKER}() {{}}")
    } else {
        String::new()
    };
    format!(
        "// extern \"C\" __global__ void {GRID_STRIDE_MARKER}() {{}}\n\
         extern \"C\" __global__ void dsa_kpool_compress(int a) {{ {} }}\n\
         extern \"C\" __global__ void dsa_index_scores(int a) {{ {} }}\n{define}\n",
        body(compress_loop),
        body(scores_loop),
    )
}

/// 2026-10-05: The scan has teeth: each of the eight loop / marker combinations is read back as
/// written (so a loop in the second kernel is not credited to the first), and the rule rejects
/// exactly the four that break it.
#[test]
fn the_marker_rule_tells_a_consistent_file_from_a_broken_one() {
    let broken = [
        (true, true, false),
        (false, false, true),
        (true, false, true),
        (false, true, true),
    ];
    for bits in 0..8u8 {
        let (c, s, m) = (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
        assert_eq!(
            stride_facts(&synthetic_cu(c, s, m)),
            (c, s, m),
            "bits {bits}"
        );
        assert_eq!(m != (c && s), broken.contains(&(c, s, m)), "bits {bits}");
    }
}
