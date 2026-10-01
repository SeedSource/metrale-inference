// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pins GLM-5.3's DSA decode to the kernel file
//! `Glm5NextDsaDecodeKernel::resolve` loads, and keeps a `common/` copy of the
//! `mla_paged_decode{,_fp8}.cu` kernels from shadowing a model directory's.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! GLM's DSA layer resolves its own selected-index decode
//! (`glm5next_dsa_mla_decode`), not the MLA paged-decode kernels, which only
//! `deepseek-v4-flash/nvfp4/` holds.

use std::path::{Path, PathBuf};

use metrale_model_arch::glm5next_dsa::KERNEL_KV_LORA_DIM;
use metrale_model_arch::glm5next_dsa::attend::{DSA_DECODE_MODULE, DSA_MLA_HEADGROUPS};

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/model-engine is two levels below the workspace root")
        .join("kernels")
}

/// 2026-09-25: The integer of `#define <name> <integer>`, ignoring any trailing
/// comment; panics when the define is missing or not an integer.
fn define(text: &str, name: &str, whose: &str) -> usize {
    let needle = format!("#define {name} ");
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with(&needle))
        .unwrap_or_else(|| panic!("{whose} no longer defines {name}"));
    line.trim_start()[needle.len()..]
        .split_whitespace()
        .next()
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("{name} in {whose} is not a plain integer: {line}"))
}

/// 2026-09-25: `Glm5NextDsaDecodeKernel::resolve` asks module `DSA_DECODE_MODULE`
/// for `glm5next_dsa_mla_decode_fp8`, and an unlisted `.cu` file's module name is
/// its file stem. A rename or move out of the GLM target without the resolve site
/// would fail at model load; this test fails first.
#[test]
fn the_dsa_decode_entry_point_exists_in_the_glm_target() {
    // 2026-09-25: The path is derived from the constant the resolve site uses.
    let cu = kernels_root().join(format!("gb10/glm-5.3-flash/nvfp4/{DSA_DECODE_MODULE}.cu"));
    let src = std::fs::read_to_string(&cu).unwrap_or_else(|e| {
        panic!("GLM's DSA decode kernel is missing at {cu:?} ({e}); `Glm5NextDsaDecodeKernel::resolve` would fail at load")
    });
    assert!(
        src.contains("glm5next_dsa_mla_decode_fp8"),
        "{cu:?} no longer defines the entry point `Glm5NextDsaDecodeKernel::resolve` asks \
         for (`glm5next_dsa_mla_decode_fp8`). The module name is the file stem, so a rename \
         here is a load-time failure at serve."
    );
}

/// 2026-10-01: The head-grouped entry points `Glm5NextDsaDecodeKernel::resolve` looks up
/// (optional, 0 when absent) live in the same file, so `METRALE_GLM_DSA_MLA_HEADGROUP` does not
/// silently fall back to the per-head kernel after a rename.
#[test]
fn the_dsa_decode_headgroup_entry_points_exist_in_the_glm_target() {
    let cu = kernels_root().join(format!("gb10/glm-5.3-flash/nvfp4/{DSA_DECODE_MODULE}.cu"));
    let src = std::fs::read_to_string(&cu).expect("GLM DSA decode kernel readable");
    for g in DSA_MLA_HEADGROUPS {
        let entry = format!("glm5next_dsa_mla_decode_fp8_hg{g}");
        assert!(
            src.contains(&format!("{entry},")),
            "{cu:?} no longer instantiates `{entry}`, which `Glm5NextDsaDecodeKernel::resolve` \
             asks for when METRALE_GLM_DSA_MLA_HEADGROUP={g}"
        );
    }
}

/// 2026-09-25: `KERNEL_KV_LORA_DIM`, which `Glm5NextDsaConfig::validate` checks
/// `kv_lora_rank` against, mirrors the kernel's `GLM_KV_LORA_DIM`. If only the
/// kernel changed, validation would admit a checkpoint the kernel reads at the
/// wrong width.
#[test]
fn rust_kv_lora_mirror_matches_the_glm_kernel_define() {
    let cu = kernels_root().join(format!("gb10/glm-5.3-flash/nvfp4/{DSA_DECODE_MODULE}.cu"));
    let src = std::fs::read_to_string(&cu).expect("GLM DSA decode kernel readable");
    assert_eq!(
        define(&src, "GLM_KV_LORA_DIM", "glm5next_dsa_mla_decode.cu"),
        KERNEL_KV_LORA_DIM,
        "GLM_KV_LORA_DIM and KERNEL_KV_LORA_DIM disagree; Glm5NextDsaConfig::validate would \
         admit a checkpoint the kernel reads at the wrong width."
    );
}

/// 2026-09-25: No `common/` copy of the MLA paged-decode kernels while a model
/// directory also carries one. The build merges `common/` and then the model
/// directory, so the model's file shadows the common one, and nothing keeps the
/// two in step.
///
/// Copies in several model directories are allowed: each target builds from its
/// own directory, so they shadow nothing.
#[test]
fn no_common_copy_shadows_a_model_mla_paged_decode_kernel() {
    let root = kernels_root();
    for stem in ["mla_paged_decode.cu", "mla_paged_decode_fp8.cu"] {
        let mut in_common: Vec<String> = Vec::new();
        let mut in_model: Vec<String> = Vec::new();
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else {
                continue;
            };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().and_then(|s| s.to_str()) == Some(stem) {
                    let rel = p.strip_prefix(&root).unwrap_or(&p).display().to_string();
                    if rel.split('/').any(|seg| seg == "common") {
                        in_common.push(rel);
                    } else {
                        in_model.push(rel);
                    }
                }
            }
        }
        in_common.sort();
        in_model.sort();
        assert!(
            in_common.is_empty() || in_model.is_empty(),
            "{stem} exists BOTH in common ({in_common:?}) and in a model directory \
             ({in_model:?}). `common/` merges into every target, so the model's copy \
             wins and the two silently diverge. Promote or delete — do not fork."
        );
        assert!(
            in_common.len() <= 1,
            "{stem} appears more than once under common/: {in_common:?}"
        );
    }
}
