// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host-only tests of the GLM-5.3 layer wiring, on the skeleton built from the
//! checkpoint config fixture.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_config::parse_config;

use crate::glm5next_skeleton::{Glm5NextTextSkeleton, Mixer, Mlp, ResidualStep, Site};

const CONFIG: &str =
    include_str!("../../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json");

fn skeleton() -> Glm5NextTextSkeleton {
    Glm5NextTextSkeleton::from_config(&parse_config(CONFIG).expect("real config parses"))
        .expect("skeleton builds")
}

/// 2026-09-25: The skeleton's residual plan for layer 0 is `hc_pre -> norm -> sublayer -> hc_post`
/// at the attention site and again at the FFN site, the order `forward_one` runs.
#[test]
fn the_layer_executes_the_skeletons_residual_plan() {
    let sk = skeleton();
    let l = sk.layers[0];
    assert_eq!(
        sk.residual_plan(&l),
        vec![
            ResidualStep::SaveResidual,
            ResidualStep::HcPre(Site::Attn),
            ResidualStep::Norm("input_layernorm.weight"),
            ResidualStep::Mixer,
            ResidualStep::HcPost(Site::Attn),
            ResidualStep::SaveResidual,
            ResidualStep::HcPre(Site::Ffn),
            ResidualStep::Norm("post_attention_layernorm.weight"),
            ResidualStep::Mlp,
            ResidualStep::HcPost(Site::Ffn),
        ],
        "forward_one runs hc_pre -> norm -> sublayer -> hc_post twice, in this order"
    );
}

/// 2026-09-25: The layer census of the config fixture: 45 text layers, 34 KDA and 11 DSA,
/// 3 dense and 42 routed, with a hyper-connection on every text layer and none on the MTP layer.
#[test]
fn the_stack_the_composite_must_cover() {
    let sk = skeleton();
    assert_eq!(sk.layers.len(), 45);
    assert_eq!(
        sk.layers.iter().filter(|l| l.mixer == Mixer::Kda).count(),
        34
    );
    assert_eq!(
        sk.layers.iter().filter(|l| l.mixer == Mixer::Dsa).count(),
        11
    );
    assert_eq!(sk.layers.iter().filter(|l| l.mlp == Mlp::Dense).count(), 3);
    assert_eq!(
        sk.layers.iter().filter(|l| l.mlp == Mlp::RoutedMoe).count(),
        42
    );
    assert!(sk.layers.iter().all(|l| l.hyper_connection));
    assert!(!sk.mtp.expect("layer 45 exists").hyper_connection);
}

/// 2026-09-25: The fixture has KDA+dense, KDA+routed and DSA+routed layers, and no DSA+dense
/// layer.
#[test]
fn every_dispatch_arm_is_reachable() {
    let sk = skeleton();
    let combos: std::collections::BTreeSet<(bool, bool)> = sk
        .layers
        .iter()
        .map(|l| (l.mixer == Mixer::Kda, l.mlp == Mlp::Dense))
        .collect();
    assert!(combos.contains(&(true, true)), "KDA + dense");
    assert!(combos.contains(&(true, false)), "KDA + routed");
    assert!(combos.contains(&(false, false)), "DSA + routed");
    assert!(!combos.contains(&(false, true)), "no DSA + dense today");
}

/// 2026-10-01: `METRALE_GLM_DSA_SCORES_TILED` is on only for `1`.
#[test]
fn dsa_tiled_lever_parses_one_as_on_and_everything_else_as_off() {
    use super::levers::parse_dsa_switch;
    assert!(parse_dsa_switch(Some("1")));
    assert!(parse_dsa_switch(Some(" 1 ")));
    for v in [None, Some(""), Some("0"), Some("2"), Some("on"), Some("true"), Some("01")] {
        assert!(!parse_dsa_switch(v), "{v:?}");
    }
}

/// 2026-10-01: `METRALE_GLM_DSA_GEMV_SPLIT` shares the switch parser: on only for `1`.
#[test]
fn dsa_tiled_gemv_split_lever_shares_the_switch_parser() {
    let src = include_str!("levers.rs");
    let start = src
        .find("pub(crate) fn dsa_gemv_split()")
        .expect("dsa_gemv_split defined");
    let body = &src[start..];
    let body = &body[..body.find("\n}\n").expect("fn closes")];
    assert!(body.contains("std::env::var(\"METRALE_GLM_DSA_GEMV_SPLIT\")"));
    assert!(body.contains("parse_dsa_switch(raw.as_deref())"));
}

/// 2026-10-01: `METRALE_GLM_DSA_SCORES_TC` maps `split3`/`1` to 3, `split2` to 2, `bf16` to 1
/// (blanks and case ignored) and everything else, unset included, to 0 (off).
#[test]
fn dsa_scores_tc_lever_parses_modes_and_defaults_off() {
    use super::levers::parse_dsa_scores_tc;
    for (v, want) in [
        (Some("1"), 3),
        (Some("split3"), 3),
        (Some(" SPLIT3 "), 3),
        (Some("split2"), 2),
        (Some("bf16"), 1),
        (Some("BF16"), 1),
    ] {
        assert_eq!(parse_dsa_scores_tc(v), want, "{v:?}");
    }
    let off = [None, Some(""), Some("0"), Some("off")];
    let unknown = [Some("2"), Some("3"), Some("fp8"), Some("on")];
    for v in off.into_iter().chain(unknown) {
        assert_eq!(parse_dsa_scores_tc(v), 0, "{v:?}");
    }
}
