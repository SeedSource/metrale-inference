// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Parses checked-in MODEL.toml `[behavior]` tables with the build
//! script's own `build_parse_behavior.rs` and checks the values the compiled
//! `ModelBehavior` gets from them.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! An integration test because cargo does not run a build script's own unit
//! tests, and because `METRALE_SKIP_BUILD=1` leaves the compiled target
//! registry empty, so the per-target `ModelBehavior` cannot be read there.

#![allow(dead_code)]

#[path = "../build_parse_behavior.rs"]
mod build_parse_behavior;

use build_parse_behavior::{DEFAULT_MASK_STRAY_TOOL_CALL_END, parse_behavior};
use std::path::PathBuf;

fn model_dir(hw: &str, model: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/kernels is two levels below the workspace root")
        .join("kernels")
        .join(hw)
        .join(model)
}

/// 2026-10-15: GLM-5.3 turns the stray `</tool_call>` mask on; the default is
/// off, in the build-time parse and in `ModelBehavior::default()` alike.
#[test]
fn glm53_masks_a_stray_tool_call_end_and_the_default_is_off() {
    let dir = model_dir("gb10", "glm-5.3-flash");
    assert!(dir.join("MODEL.toml").exists(), "{dir:?}/MODEL.toml");
    let glm = parse_behavior(&dir);
    assert!(
        glm.mask_stray_tool_call_end,
        "kernels/gb10/glm-5.3-flash sets it"
    );
    // 2026-10-15: the sibling key in the same table still parses.
    assert!(glm.effort_prompt_only);

    assert!(!DEFAULT_MASK_STRAY_TOOL_CALL_END);
    assert!(!metrale_kernels::DEFAULT_MASK_STRAY_TOOL_CALL_END);
    assert!(!metrale_kernels::ModelBehavior::default().mask_stray_tool_call_end);

    // 2026-10-15: a model that does not set the key gets the default.
    let other = parse_behavior(&model_dir("gb10", "qwen3.5-397b-a17b"));
    assert!(!other.mask_stray_tool_call_end);
    // 2026-10-15: a directory with no MODEL.toml gets the default.
    let none = parse_behavior(&model_dir("gb10", "no-such-model"));
    assert!(!none.mask_stray_tool_call_end);
}
