// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Checks the checked-in hopper and b200 trees, which inherit
//! gb10's kernels through `[hardware] inherits = "gb10"`: their HARDWARE.toml
//! keys, their `common/` role, their model directories and leaves, and their
//! `kernel_source` redirects.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! The resolver (`metrale_closure::layout`) reads every file an overlay does
//! not hold from gb10, and a file the overlay holds that replaces a gb10 file
//! must be declared in `[shadow]` of the `KERNEL.toml` beside it. CI runs
//! `cargo test` with `METRALE_SKIP_BUILD=1`, where build.rs returns before it
//! resolves any target, so these tests are what grade the trees there. Each
//! test loops over `INHERITED` and names the hardware set in its failure
//! message. The `METRALE_NO_WARP_BLOCKSCALE_MMA` and `[expected_absent]`
//! checks are in `inherited_targets_w4a4.rs`.

#[path = "support/inherited.rs"]
mod inherited;

use inherited::{INHERITED, gb10_dir, hardware_toml, hw_dir, workspace_root};
use metrale_closure::layout::{Layout, Role, Target, Tier, discover, walk};

fn resolve(hw: &str, model: &str, quant: &str) -> Layout {
    let t = Target {
        hardware: hw.into(),
        model: model.into(),
        quant: quant.into(),
    };
    discover(&workspace_root(), &t).unwrap_or_else(|e| panic!("{t}: {e}"))
}

/// 2026-09-25: Each overlay's `[hardware]` states its own name, vendor, arch,
/// compute capability and `inherits = "gb10"`. build.rs's `resolve_targets`
/// passes `hardware.arch` to nvcc and reads `hardware.vendor` for the flag key.
#[test]
fn every_inherited_hardware_toml_declares_its_own_nvidia_arch() {
    for t in INHERITED {
        let toml = hardware_toml(t.hw);
        let hw = toml
            .get("hardware")
            .unwrap_or_else(|| panic!("kernels/{}: no [hardware] table", t.hw));
        assert_eq!(
            hw.get("name").and_then(|v| v.as_str()),
            Some(t.hw),
            "kernels/{}: [hardware].name must match the directory",
            t.hw
        );
        assert_eq!(
            hw.get("vendor").and_then(|v| v.as_str()),
            Some("nvidia"),
            "kernels/{}: vendor picks the compiler in \
             build_target::resolve_compute_target",
            t.hw
        );
        assert_eq!(
            hw.get("arch").and_then(|v| v.as_str()),
            Some(t.arch),
            "kernels/{}: arch is what reaches nvcc -arch=",
            t.hw
        );
        assert_eq!(
            hw.get("compute_capability").and_then(|v| v.as_str()),
            Some(t.cc),
            "kernels/{}: compute_capability is read by tests/target_hints.rs",
            t.hw
        );
        assert_eq!(
            hw.get("inherits").and_then(|v| v.as_str()),
            Some("gb10"),
            "kernels/{}: an overlay of gb10 says so in HARDWARE.toml",
            t.hw
        );
    }
}

/// 2026-09-25: Each overlay's `[hardware]` table has gb10's keys plus
/// `inherits`.
#[test]
fn every_inherited_hardware_toml_carries_the_same_key_set_as_gb10() {
    let gb10_path = gb10_dir().join("HARDWARE.toml");
    let gb10: toml::Value =
        toml::from_str(&std::fs::read_to_string(&gb10_path).expect("gb10 HARDWARE.toml"))
            .expect("valid TOML");
    let keys = |v: &toml::Value| -> std::collections::BTreeSet<String> {
        v.get("hardware")
            .and_then(|h| h.as_table())
            .expect("[hardware] table")
            .keys()
            .cloned()
            .collect()
    };
    let mut expected = keys(&gb10);
    expected.insert("inherits".into());
    for t in INHERITED {
        assert_eq!(
            keys(&hardware_toml(t.hw)),
            expected,
            "kernels/{}/HARDWARE.toml must declare the same keys as the other \
             NVIDIA targets, plus `inherits`",
            t.hw
        );
    }
}

/// 2026-09-25: Each overlay's `[defaults]` names the same levers as gb10's,
/// even where the value agrees. A key left out would take its value from
/// `build_defaults::baseline`, which the file would not show. The values are
/// asserted in `tests/target_defaults.rs`.
#[test]
fn every_inherited_hardware_toml_declares_the_same_serving_levers_as_gb10() {
    let gb10_path = gb10_dir().join("HARDWARE.toml");
    let gb10: toml::Value =
        toml::from_str(&std::fs::read_to_string(&gb10_path).expect("gb10 HARDWARE.toml"))
            .expect("valid TOML");
    let levers = |v: &toml::Value| -> std::collections::BTreeSet<String> {
        v.get("defaults")
            .and_then(|d| d.as_table())
            .expect("[defaults] table")
            .keys()
            .cloned()
            .collect()
    };
    for t in INHERITED {
        assert_eq!(
            levers(&hardware_toml(t.hw)),
            levers(&gb10),
            "kernels/{}/HARDWARE.toml [defaults] must state every lever \
             explicitly, so the file answers 'what does this target serve \
             with' on its own",
            t.hw
        );
    }
}

/// 2026-09-25: Each overlay's common role holds every `.cu`, `.cuh` and `.h`
/// of `kernels/gb10/common`, plus the files in its own `common/`, where a file
/// that replaces a gb10 one is declared in `[shadow]` and an addition is not.
/// gb10's `common/KERNEL.toml` is among its configs.
#[test]
fn every_inherited_common_is_gb10s_common_plus_its_own_declared_files() {
    let gb10_common = gb10_dir().join("common");
    for t in INHERITED {
        let l = resolve(t.hw, t.models[0], "nvfp4");
        for entry in std::fs::read_dir(&gb10_common).unwrap().flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !(name.ends_with(".cu") || name.ends_with(".cuh") || name.ends_with(".h")) {
                continue;
            }
            assert!(
                l.common.contains_key(&name),
                "kernels/{}: gb10/common/{name} does not reach this overlay",
                t.hw
            );
        }
        let own_common = hw_dir(t.hw).join("common");
        for (name, e) in &l.common {
            let layer = &l.layers[e.layer];
            if layer.tier == Tier::Own {
                assert_eq!(e.source, own_common.join(name), "kernels/{}", t.hw);
                let replaces = gb10_common.join(name).is_file();
                let declared = l
                    .shadows
                    .iter()
                    .any(|s| s.role == Role::Common && &s.name == name);
                assert_eq!(
                    replaces, declared,
                    "kernels/{}/common/{name}: a replacement is declared in [shadow], an addition is not",
                    t.hw
                );
            } else {
                assert_eq!(e.source, gb10_common.join(name), "kernels/{}", t.hw);
            }
        }
        assert!(
            l.configs().contains(&gb10_common.join("KERNEL.toml")),
            "kernels/{}: gb10's common/KERNEL.toml is the flag base",
            t.hw
        );
        assert!(
            l.common.keys().filter(|n| n.ends_with(".cu")).count() > 100,
            "kernels/{}: suspiciously few .cu sources resolved",
            t.hw
        );
    }
}

/// 2026-09-25: The models `metrale_closure::layout::walk` lists for `hw`, which
/// build.rs also uses to expand `METRALE_TARGET_MODEL=*`.
fn model_dirs(hw: &str) -> Vec<String> {
    let mut names: Vec<String> = walk(&workspace_root())
        .expect("the tree resolves")
        .into_iter()
        .filter(|t| t.hardware == hw)
        .map(|t| t.model)
        .collect();
    names.sort();
    names.dedup();
    names
}

// 2026-09-25: b200 also has kimi-k3, which is not in `P0_MODELS`;
// b200_kimi_target.rs checks its sources.
fn declared_models(t: &inherited::Inherited) -> Vec<&str> {
    let mut models = t.models.to_vec();
    if t.hw == "b200" {
        models.push("kimi-k3");
        models.sort_unstable();
    }
    models
}

/// 2026-09-25: The models `walk` finds for each overlay are the declared ones.
#[test]
fn the_p0_model_targets_are_the_ones_declared() {
    for t in INHERITED {
        assert_eq!(model_dirs(t.hw), declared_models(t), "kernels/{}", t.hw);
    }
}

/// 2026-09-25: Each declared model's MODEL.toml is a regular file, not a
/// symlink, its `[model].name` is its directory name, and its first six lines
/// carry the target's provenance note and name `--check-kernels`.
#[test]
fn every_model_toml_names_its_directory_and_records_its_provenance() {
    for t in INHERITED {
        for model in t.models {
            let path = hw_dir(t.hw).join(model).join("MODEL.toml");
            assert!(
                std::fs::read_link(&path).is_err(),
                "{}/{model}/MODEL.toml is a symlink",
                t.hw
            );
            let text = std::fs::read_to_string(&path).unwrap();
            let toml: toml::Value = toml::from_str(&text)
                .unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()));
            assert_eq!(
                toml.get("model")
                    .and_then(|m| m.get("name"))
                    .and_then(|v| v.as_str()),
                Some(*model),
                "kernels/{}/{model}",
                t.hw
            );
            let head: String = text.lines().take(6).collect::<Vec<_>>().join("\n");
            assert!(
                head.contains(t.provenance),
                "{}/{model}/MODEL.toml does not open with the inheritance note {:?}:\n{head}",
                t.hw,
                t.provenance
            );
            assert!(
                head.contains("--check-kernels"),
                "{}/{model}/MODEL.toml does not say expected_absent is unharvested on this hardware",
                t.hw
            );
        }
    }
}

/// 2026-09-25: Each declared model's nvfp4 leaf role is gb10's leaf, file for
/// file: the overlay holds no `<model>/nvfp4` directory of its own.
///
/// Hopper uses an nvfp4 leaf too, because serve accepts an nvfp4 kernel set
/// for FP8 and BF16 checkpoints (`quant_pair_compatible` in
/// crates/server/src/main_modules/serve_quant.rs).
#[test]
fn every_model_leaf_is_gb10s_leaf() {
    for t in INHERITED {
        for model in t.models {
            let l = resolve(t.hw, model, "nvfp4");
            let origin = gb10_dir().join(&l.source_model).join("nvfp4");
            assert!(
                !hw_dir(t.hw).join(&l.source_model).join("nvfp4").exists(),
                "kernels/{}/{}/nvfp4 exists: a per-model fork on an overlay",
                t.hw,
                l.source_model
            );
            assert_eq!(l.layers[1].dir, origin, "{}/{model}: parent leaf", t.hw);
            for (name, e) in &l.leaf {
                // 2026-09-25: Held by gb10's leaf, or `use`d by it from another
                // gb10 model.
                assert_eq!(
                    l.layers[e.layer].tier,
                    Tier::Parent,
                    "{}/{model}: {name}",
                    t.hw
                );
                assert!(e.source.starts_with(gb10_dir()), "{}/{model}: {name}", t.hw);
            }
            let gb10 = resolve("gb10", model, "nvfp4");
            assert_eq!(
                l.leaf.keys().collect::<Vec<_>>(),
                gb10.leaf.keys().collect::<Vec<_>>(),
                "{}/{model}: the leaf role differs from gb10's",
                t.hw
            );
        }
    }
}

/// 2026-09-25: Every model of each overlay resolves at that hardware's arch
/// with more than 100 modules.
#[test]
fn a_wildcard_build_resolves_declared_targets_at_the_declared_arch() {
    for t in INHERITED {
        let names = model_dirs(t.hw);
        assert_eq!(names, declared_models(t), "kernels/{}", t.hw);
        for model in &names {
            let l = resolve(t.hw, model, "nvfp4");
            assert_eq!(l.hardware.arch.as_deref(), Some(t.arch), "{}/{model}", t.hw);
            assert!(
                l.modules().len() > 100,
                "{}/{model}: {} modules",
                t.hw,
                l.modules().len()
            );
        }
    }
}

/// 2026-09-25: Two `kernel_source` redirects, each within the same hardware
/// tree: hopper's qwen3.8-27b reads qwen3.6-27b, and deepseek-v4.1-flash reads
/// deepseek-v4-flash. Every other model reads its own directory.
#[test]
fn inherited_redirects_resolve_only_to_the_declared_same_hardware_source() {
    for t in INHERITED {
        for model in t.models {
            let l = resolve(t.hw, model, "nvfp4");
            let source = match (t.hw, *model) {
                ("hopper", "qwen3.8-27b") => "qwen3.6-27b",
                (_, "deepseek-v4.1-flash") => "deepseek-v4-flash",
                _ => model,
            };
            assert_eq!(
                l.source_model, source,
                "{}/{model}: kernel_source redirect",
                t.hw
            );
            assert_eq!(
                l.layers[0].dir,
                hw_dir(t.hw).join(source).join("nvfp4"),
                "{}/{model}",
                t.hw
            );
        }
    }
}

/// 2026-10-05: gb10's glm-5.3-flash target compiles the Hopper FP8 activation-quantizer
/// twin through `[sources] use` in its KERNEL.toml, beside the shared quantizer it is
/// byte-identical to. No other gb10 target stages it, so their `fp8_act_quant_hopper`
/// handle stays 0.
#[test]
fn gb10_glm53_stages_the_fp8_act_quant_twin_and_other_gb10_targets_do_not() {
    let twin = resolve("gb10", "glm-5.3-flash", "nvfp4");
    let modules = twin.modules();
    let find = |m: &[(String, &metrale_closure::layout::Entry)], stem: &str| {
        m.iter().find(|(s, _)| s == stem).map(|(_, e)| e.source.clone())
    };
    let src = find(&modules, "fp8_act_quant_hopper").expect("glm-5.3-flash stages the twin");
    assert_eq!(
        src,
        workspace_root().join("kernels/hopper/common/fp8_act_quant_hopper.cu")
    );
    assert!(twin.sources().contains(&src), "the build compiles it");
    assert!(
        find(&modules, "per_token_group_quant_fp8").is_some(),
        "the shared quantizer it is compared against stays"
    );
    let other = resolve("gb10", "qwen3.6-27b", "nvfp4");
    assert!(find(&other.modules(), "fp8_act_quant_hopper").is_none());
    let hopper = resolve("hopper", "qwen3.8-27b", "nvfp4");
    assert!(find(&hopper.modules(), "fp8_act_quant_hopper").is_some());
}
