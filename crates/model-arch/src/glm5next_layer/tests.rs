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

/// 2026-10-01: `METRALE_GLM_PREFILL_FULLWIDTH_GEMM` is on only for `1`, and the staged pass
/// takes its full-width arm only when the FFN window is wider than the attention sub-chunk.
#[test]
fn fullwidth_lever_parses_one_as_on_and_needs_a_wider_window() {
    use super::levers::parse_fullwidth_switch;
    use super::steps::staged::full_width_attn;
    assert!(parse_fullwidth_switch(Some("1")));
    assert!(parse_fullwidth_switch(Some(" 1 ")));
    for v in [None, Some(""), Some("0"), Some("2"), Some("on"), Some("true"), Some("01")] {
        assert!(!parse_fullwidth_switch(v), "{v:?}");
    }
    assert!(full_width_attn(true, 256, 4096));
    assert!(full_width_attn(true, 256, 512));
    assert!(!full_width_attn(true, 256, 256), "equal widths: nothing to widen");
    assert!(!full_width_attn(false, 256, 4096), "lever off: the sliced pass");
}

/// 2026-10-01: The full-width lever needs the staged prefill: its reader requires
/// `METRALE_GLM_PREFILL_STAGED=1` before it reports on.
#[test]
fn fullwidth_lever_is_inert_without_staging() {
    let src = include_str!("levers.rs");
    let start = src
        .find("pub fn prefill_fullwidth_gemm()")
        .expect("prefill_fullwidth_gemm defined");
    let body = &src[start..];
    let body = &body[..body.find("\n}\n").expect("fn closes")];
    assert!(body.contains("std::env::var(\"METRALE_GLM_PREFILL_FULLWIDTH_GEMM\")"));
    assert!(
        body.contains("std::env::var(\"METRALE_GLM_PREFILL_STAGED\").as_deref() == Ok(\"1\")")
    );
}

/// 2026-10-01: `Glm5NextKdaWorkspace::bytes_split` at equal widths is the unsplit size, and
/// narrower chunk buffers save exactly 24 bytes per padded token per `qkv` channel (GLM-5.3
/// TP2 geometry: 32 heads x 128, hidden 4096, chunk 32).
#[test]
fn kda_split_workspace_saves_only_the_chunk_buffers() {
    use crate::glm5next_kda::{Glm5NextKdaConfig, Glm5NextKdaWorkspace};
    let cfg = Glm5NextKdaConfig {
        hidden: 4096,
        heads: 32,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-5,
        l2_eps: 1e-6,
        chunk: 32,
    };
    let qkv = cfg.qkv_dim();
    let full = Glm5NextKdaWorkspace::bytes_split(&cfg, 4096, 4096);
    let split = Glm5NextKdaWorkspace::bytes_split(&cfg, 4096, 256);
    assert_eq!(full - split, 24 * (4096 - 256) * qkv);
    assert_eq!(
        Glm5NextKdaWorkspace::bytes_split(&cfg, 256, 4096),
        Glm5NextKdaWorkspace::bytes_split(&cfg, 256, 256),
        "chunk_tokens is clamped to max_tokens"
    );
}
