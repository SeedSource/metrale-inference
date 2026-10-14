// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests that the checked-in GLM-5.3-Flash MODEL.toml gives a
//! chat request that omits its sampling fields the official serve's values.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! The official values: nvidia/GLM-5.3-Flash-NVFP4 generation_config.json sets
//! temperature 1.0 and top_p 0.95; vLLM v0.30.0 fills the rest with top_k 0
//! (all tokens), min_p 0.0, repetition_penalty 1.0, presence and frequency
//! 0.0, and has no top-n-sigma. `build_parse::parse_sampling_presets` reads
//! each key below as is; a key left out would take its parse default (top_k
//! 20, min_p and top_n_sigma `None` = the server's `--default-min-p 0.08` and
//! `--default-top-n-sigma 1.0`), so every key must be present.

use std::path::PathBuf;

fn glm_model_toml() -> toml::Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/glm-5.3-flash/MODEL.toml");
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    toml::from_str(&s).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// 2026-10-14: The presets, not the server-wide defaults, feed the core sampler.
#[test]
fn glm53_uses_its_presets_for_the_core_sampler() {
    let t = glm_model_toml();
    assert_eq!(
        t["behavior"]["use_sampling_presets_for_core"].as_bool(),
        Some(true)
    );
}

/// 2026-10-14: Every chat category declares the official values, with no
/// top-k, min-p or top-n-sigma filter and neutral penalties.
#[test]
fn every_glm53_category_declares_the_official_values() {
    let t = glm_model_toml();
    for cat in ["thinking_text", "thinking_coding", "non_thinking", "tools"] {
        let c = &t["sampling"][cat];
        let f = |k: &str| c.get(k).and_then(|v| v.as_float());
        assert_eq!(f("temperature"), Some(1.0), "{cat}");
        assert_eq!(f("top_p"), Some(0.95), "{cat}");
        assert_eq!(
            c.get("top_k").and_then(|v| v.as_integer()),
            Some(0),
            "{cat}"
        );
        assert_eq!(f("min_p"), Some(0.0), "{cat}");
        assert_eq!(f("top_n_sigma"), Some(0.0), "{cat}");
        for k in [
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "dry_multiplier",
            "lz_penalty",
        ] {
            assert!(
                c.get(k).is_none(),
                "{cat}.{k}: the parse default is the neutral value"
            );
        }
    }
}
