// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pins the `.param` count of each kernel in `PINS` against the PTX
//! compiled into this binary, for every target that has the kernel. The test is
//! `#[ignore]`d: it needs a build with nvcc and `METRALE_SKIP_BUILD` unset.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! When a kernel gains a parameter, update its launcher in
//! metrale-model-layers and its pin here in the same commit.

/// 2026-09-25: (module, kernel, expected `.param` count). A target whose PTX
/// lacks a pair is skipped: presence is the kernel audit's job, this test
/// checks the arity of what is present.
const PINS: &[(&str, &str, usize)] = &[
    ("w4a16", "w4a16_gemm", 8),
    ("w4a16", "w4a16_gemm_t", 9),
    ("w4a16", "w4a16_gemm_t_p3", 9),
    // 2026-09-25: The deep-K twins take no stride, yet dense_ffn's small-M arm
    // launches `w4a16_gemm_t_k64` through `ops::w4a16_gemm_n128`, which packs
    // 9 arguments (`ldb = n`). Pinned at 8, so a stride added to them fails
    // here.
    ("w4a16", "w4a16_gemm_t_k64", 8),
    ("w4a16", "w4a16_gemm_t_k64_p3", 8),
    ("w4a16", "w4a16_gemm_t_k64_n64_p3", 8),
    ("w4a16", "w4a16_gemm_t_m128", 8),
    ("w4a16", "w4a16_gemm_t_m128_bf16", 8),
    ("w4a16", "w4a16_gemm_t_m128_bf16_v2", 9),
    ("w4a16_v2", "w4a16_gemm_t_m128_v2", 8),
    ("w4a16_v3", "w4a16_gemm_t_m128_v3", 8),
    // 2026-09-25: Load-time weight transpose; `ops::transpose_u8` packs 4.
    ("transpose_u8", "transpose_u8", 4),
    // 2026-09-25: The 32-row W8A16 tile. `ops::w8a16_gemm_pipelined_m32_strided`
    // packs 9, and the contiguous entry passes K and N as the row pitches.
    ("w8a16_gemm_pipelined_m32", "w8a16_gemm_pipelined_m32", 9),
    // 2026-10-05: The per-row-scale W8A8 GEMM; `ops::fp8_gemm_t_rowscale` packs 9
    // (A, a_scale, ones, B, w_row_scale, C, M, N, K).
    (
        "fp8_gemm_blockscaled_pipe",
        "fp8_gemm_rowscale_pipe_128x64",
        9,
    ),
];

/// 2026-09-25: The arity a target's copy of a kernel must have. No target
/// differs from the family pin, so it returns the pin; this is the place for a
/// per-target exception. `w4a16_gemm_t_ldb_drift_is_exactly_the_known_set`
/// (metrale-model-layers `ops/gemm_dense_tests.rs`) checks that every copy of
/// the `LDB_KERNELS` declares `ldb`. Derive any exception's arity from the
/// `.cu` tree.
fn expected_arity(model: &str, module: &str, kernel: &str, family_pin: usize) -> usize {
    let _ = (model, module, kernel);
    family_pin
}

/// 2026-09-25: Counts the `.param` declarations between `.entry <kernel>(` and
/// the next `)`, or `None` if the entry is absent.
fn ptx_param_count(ptx: &str, kernel: &str) -> Option<usize> {
    let needle = format!(".entry {kernel}(");
    let start = ptx.find(&needle)?;
    let body = &ptx[start..];
    let close = body.find(')')?;
    Some(body[..close].matches(".param").count())
}

#[test]
#[ignore = "requires nvcc and METRALE_SKIP_BUILD unset"]
fn w4a16_launch_family_arity_pins() {
    // 2026-09-25: Under `METRALE_SKIP_BUILD=1` build.rs emits a stub
    // `target_ptx.rs` with no modules, so there is no PTX to read: report and
    // return. With compiled PTX, the `checked >= 4` floor below applies.
    if metrale_kernels::available_targets()
        .iter()
        .all(|s| s.modules.is_empty())
    {
        eprintln!("no compiled PTX in this binary (stub build) — arity pins skipped");
        return;
    }
    let mut checked = 0usize;
    for set in metrale_kernels::available_targets() {
        for (module, blob) in &set.modules {
            let Ok(ptx) = std::str::from_utf8(blob) else {
                continue;
            };
            for &(pin_module, kernel, family_pin) in PINS {
                if *module != pin_module {
                    continue;
                }
                if let Some(count) = ptx_param_count(ptx, kernel) {
                    let want = expected_arity(set.target.model, module, kernel, family_pin);
                    assert_eq!(
                        count, want,
                        "PTX arity drift: {}::{} on target {} has {} params, launcher family \
                         pins {} — update the launcher AND this pin together",
                        module, kernel, set.target.model, count, want
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked >= 4,
        "arity test checked only {checked} kernels — PTX sets missing? (wildcard build expected)"
    );
}
