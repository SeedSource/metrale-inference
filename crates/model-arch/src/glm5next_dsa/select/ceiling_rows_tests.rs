// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Host tests of `DsaSelectLaunch::CeilingRows` on the mock backend: the
//! single-row `Ceiling` refusal of `q_rows > 1` is unchanged, `CeilingRows` needs a
//! `geom_dev` and a scratch that fits its rows at the ceiling, it launches the ceiling chain
//! with the row axis widened, and nothing in production constructs it.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::super::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, DsaSelectScratch, contiguous_pool_count,
    select_tokens,
};
use crate::glm5next_dsa::state::max_dsa_context;
use crate::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::{MockGpuBackend, MockLaunch};

/// 2026-10-12: GLM-5.3 DSA geometry (`select/tests.rs`) at 65,536 tokens.
fn cfg() -> Glm5NextDsaConfig {
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
        max_context: 65_536,
    }
}

/// 2026-10-12: Dummy device addresses; the mock launches do not dereference them.
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
        pool_cache: None,
    }
}

/// 2026-10-12: One `select_tokens` at the ceiling of `cfg()`: `q_rows` rows, a scratch planned
/// for `scratch_rows` rows at the ceiling, `rows` choosing `CeilingRows` over `Ceiling`.
fn run(
    q_rows: usize,
    scratch_rows: usize,
    rows: bool,
    geom_dev: DevicePtr,
) -> anyhow::Result<Vec<MockLaunch>> {
    let gpu = MockGpuBackend::new();
    let c = cfg();
    let cap = max_dsa_context(&c);
    let plan = DsaSelectGeometry::plan(&c, cap, scratch_rows)?;
    let scratch = DsaSelectScratch::alloc(&gpu, &c, &plan)?;
    let kernels = Glm5NextDsaKernels::resolve(&gpu)?;
    let geom = DsaSelectGeometry::plan(&c, cap, q_rows)?;
    let max_pools = contiguous_pool_count(c.index_kpool, cap);
    let launch = if rows {
        DsaSelectLaunch::CeilingRows { max_pools }
    } else {
        DsaSelectLaunch::Ceiling { max_pools }
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
    )?;
    Ok(gpu.launches_snapshot())
}

const GEOM: DevicePtr = DevicePtr(0x9000);

#[test]
fn single_row_ceiling_still_refuses_rows() {
    let e = run(3, 3, false, GEOM).unwrap_err().to_string();
    assert!(e.contains("decode-only"), "{e}");
}

#[test]
fn ceiling_rows_needs_a_device_geometry() {
    let e = run(3, 3, true, DevicePtr::NULL).unwrap_err().to_string();
    assert!(e.contains("geom_dev"), "{e}");
}

#[test]
fn ceiling_rows_needs_its_rows_at_the_ceiling() {
    let e = run(3, 1, true, GEOM).unwrap_err().to_string();
    assert!(e.contains("scratch region"), "{e}");
    assert!(run(3, 3, true, GEOM).is_ok());
}

/// 2026-10-12: The same kernels in the same order as a single-row ceiling pass, with the row
/// axis (scores grid y, top-k and expand grid x) at `q_rows`, and the same scalar arguments.
#[test]
fn ceiling_rows_is_the_ceiling_chain_with_the_row_axis_widened() {
    let one = run(1, 3, false, GEOM).unwrap();
    let three = run(3, 3, true, GEOM).unwrap();
    assert_eq!(one.len(), three.len());
    for (i, (a, b)) in one.iter().zip(&three).enumerate() {
        assert_eq!(a.func, b.func, "launch {i}");
        assert_eq!(
            (a.block, a.shared_mem),
            (b.block, b.shared_mem),
            "launch {i}"
        );
    }
    // 2026-10-12: compress, scores, top-k, expand (both levers off on the mock).
    assert_eq!(one.len(), 4);
    assert_eq!(three[0].grid, one[0].grid, "compress has no row axis");
    assert_eq!(
        three[1].grid,
        [one[1].grid[0], 3, 1],
        "scores grid y = rows"
    );
    assert_eq!(three[2].grid, [3, 1, 1], "top-k one block per row");
    assert_eq!(three[3].grid, [3, 1, 1], "expand one block per row");
    // 2026-10-12: Only the q_rows argument (`Q`) differs.
    for (i, (a, b)) in one.iter().zip(&three).enumerate().skip(1) {
        let diff: Vec<usize> = (0..a.args.len())
            .filter(|&j| a.args[j] != b.args[j])
            .collect();
        assert!(diff.len() <= 1, "launch {i}: args {diff:?} differ");
    }
}

/// 2026-10-12: No production file constructs `CeilingRows`: it is reachable only from the
/// microtest until a lever-gated caller is reviewed in.
#[test]
fn no_production_caller_constructs_ceiling_rows() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let allowed = [
        "glm5next_dsa/select.rs",
        "glm5next_dsa/select/launch.rs",
        "glm5next_dsa/select/ceiling_rows.rs",
        "glm5next_dsa/select/ceiling_rows_tests.rs",
    ];
    let mut pending = vec![root.clone()];
    let mut scanned = 0;
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("readable src dir") {
            let path = entry.expect("readable entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|x| x == "rs") {
                scanned += 1;
                let rel = path.strip_prefix(&root).unwrap().to_string_lossy();
                let src = std::fs::read_to_string(&path).unwrap();
                if src.contains("CeilingRows") {
                    assert!(allowed.contains(&rel.as_ref()), "{rel} uses CeilingRows");
                }
            }
        }
    }
    assert!(scanned > 100, "scanned only {scanned} files");
}
