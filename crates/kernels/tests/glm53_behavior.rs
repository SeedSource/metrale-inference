// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The `[behavior]` values `kernels/gb10/glm-5.3-flash/MODEL.toml` gives through
//! the build script's own parser (`build_parse_behavior.rs`, included here), and that a
//! model without those keys keeps every default.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.

#[allow(dead_code)]
#[path = "../build_parse_behavior.rs"]
mod parse_behavior;

use parse_behavior::{ParsedBehavior, parse_behavior};

fn gb10(model: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10")
        .join(model)
}

#[test]
fn glm53_model_toml_sets_its_parity_flags() {
    let b = parse_behavior(&gb10("glm-5.3-flash"));
    assert!(b.effort_prompt_only);
    assert!(b.response_format_with_tools);
    assert!(!b.enable_think_loop_watchdog);
    assert!(!b.tool_call_start_bias);
}

#[test]
fn a_model_without_the_keys_keeps_the_defaults() {
    let d = ParsedBehavior::default();
    let lib = metrale_kernels::ModelBehavior::default();
    assert!(!d.response_format_with_tools);
    assert_eq!(d.response_format_with_tools, lib.response_format_with_tools);
    let b = parse_behavior(&gb10("qwen3.5-397b-a17b"));
    assert!(!b.response_format_with_tools);
    assert!(b.enable_think_loop_watchdog);
    assert_eq!(d.enable_think_loop_watchdog, lib.enable_think_loop_watchdog);
    assert!(b.tool_call_start_bias);
    assert_eq!(d.tool_call_start_bias, lib.tool_call_start_bias);
}
