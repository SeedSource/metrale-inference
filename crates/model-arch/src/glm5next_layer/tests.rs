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

/// 2026-10-01: `METRALE_GLM_DECODE_L2_PREFETCH` shares the switch parser (on only for `1`), and
/// `METRALE_GLM_DECODE_L2_PREFETCH_MIB` accepts integers 1..=64 only.
#[test]
fn decode_l2_prefetch_levers_parse() {
    use super::levers::parse_l2_prefetch_mib;
    let src = include_str!("levers.rs");
    let start = src
        .find("pub(crate) fn decode_l2_prefetch()")
        .expect("decode_l2_prefetch defined");
    let body = &src[start..];
    let body = &body[..body.find("\n}\n").expect("fn closes")];
    assert!(body.contains("std::env::var(\"METRALE_GLM_DECODE_L2_PREFETCH\")"));
    assert!(body.contains("parse_dsa_switch(raw.as_deref())"));
    assert_eq!(parse_l2_prefetch_mib(Some("12")), Some(12));
    assert_eq!(parse_l2_prefetch_mib(Some(" 1 ")), Some(1));
    assert_eq!(parse_l2_prefetch_mib(Some("64")), Some(64));
    for v in [None, Some(""), Some("0"), Some("65"), Some("-1"), Some("8M"), Some("1.5")] {
        assert_eq!(parse_l2_prefetch_mib(v), None, "{v:?}");
    }
}

/// 2026-10-01: The batched multi-sequence decode keeps both row aliases removed: row `i` takes
/// highway slot `ctx.hc_row_offset + i` and `row_view(i)` metadata in the per-row arm, and the
/// DSA arm of `forward_n_seqs` hands each row its own metadata row. The route stays default-off
/// behind `METRALE_GLM_DECODE_MULTI_SEQ`.
#[test]
fn multi_seq_decode_indexes_each_row_and_stays_behind_its_lever() {
    let ms = include_str!("steps/multi_seq.rs");
    assert!(ms.contains("ctx.hc_row_offset + i,"), "per-row highway slot");
    // 2026-10-03: Three: the two per-row arms and the `METRALE_GLM_DSA_XSEQ_BATCH` arm.
    assert_eq!(
        ms.matches("m.row_view(i)").count(),
        3,
        "every per-row arm reads its own metadata row"
    );
    assert!(ms.contains("layer.decode_n_seqs(gpu, normed, n, &seq_states, ws, stream)"));
    let m = include_str!("mod.rs");
    let start = m
        .find("fn decode_multi_seq_unsupported(&self) -> bool {")
        .expect("override present");
    let body = &m[start..start + 120];
    assert!(body.contains("!levers::decode_multi_seq()"), "{body}");
    let l = include_str!("levers.rs");
    assert!(l.contains("std::env::var(\"METRALE_GLM_DECODE_MULTI_SEQ\").as_deref() == Ok(\"1\")"));
}

/// 2026-10-03: The cross-sequence DSA projections stay default-off behind
/// `METRALE_GLM_DSA_XSEQ_BATCH`: the loader attaches the arena only under the lever, both
/// call sites try `decode_xseq` only with the arena attached and keep their per-sequence loop
/// as the fallback arm, and each sequence gets the loop's own metadata rows.
#[test]
fn dsa_xseq_batch_stays_behind_its_lever() {
    let l = include_str!("../glm5next_dsa/layer/xseq.rs");
    assert!(l.contains("std::env::var(\"METRALE_GLM_DSA_XSEQ_BATCH\").as_deref() == Ok(\"1\")"));
    let ld = include_str!("../weight_loader/glm5_next_load/loader.rs");
    assert!(ld.contains("let dsa_xseq = if crate::glm5next_dsa::layer::dsa_xseq_batch() {"));
    assert_eq!(ld.matches("ws.with_xseq(").count(), 1);
    for (file, src, meta) in [
        (
            "multi_seq.rs",
            include_str!("steps/multi_seq.rs"),
            "ctx.attn_metadata.as_ref().map(|m| m.row_view(i))",
        ),
        (
            "verify_multi.rs",
            include_str!("steps/verify_multi.rs"),
            ".map(|m| verify_rows_view(m, off[i], ks[i]))",
        ),
    ] {
        let arm = src
            .split_once("if layer.xseq_attached() && {")
            .unwrap_or_else(|| panic!("{file}: xseq arm present"))
            .1;
        let arm = &arm[..arm.find("=>").expect("arm ends")];
        assert!(arm.contains(meta), "{file}: the loop's metadata rows");
        assert!(arm.contains("layer.decode_xseq("), "{file}");
        assert!(
            src.matches("layer.decode_k(").count() == 1,
            "{file}: the per-sequence loop stays as the fallback"
        );
    }
}

/// 2026-10-02: The batched MTP verify stays default-off behind `METRALE_GLM_BATCHED_VERIFY`:
/// with the lever off, `decode_verify_multi_unsupported` answers true and the engine never
/// batches a GLM verify. On, each sequence keeps its own metadata rows and its own KDA walk.
#[test]
fn batched_verify_stays_behind_its_lever_and_indexes_each_sequence() {
    let m = include_str!("mod.rs");
    let start = m
        .find("fn decode_verify_multi_unsupported(&self) -> bool {")
        .expect("override present");
    let body = &m[start..start + 110];
    assert!(body.contains("!levers::batched_verify()"), "{body}");
    let l = include_str!("levers.rs");
    assert!(l.contains("std::env::var(\"METRALE_GLM_BATCHED_VERIFY\").as_deref() == Ok(\"1\")"));
    let v = include_str!("steps/verify_multi.rs");
    assert!(v.contains("verify_rows_view(m, off[i], ks[i])"), "per-sequence metadata rows");
    assert!(v.contains("layer.decode_verify_n_seqs(gpu, normed, ks,"), "per-sequence KDA walk");
    // 2026-10-02: The engine's row gate reads the same workspace rows the layer checks, and the
    // loader widens those workspaces only through `batched_verify_rows` (0 with the lever off).
    let cap = m
        .find("fn decode_verify_multi_max_rows(&self) -> usize {")
        .expect("row cap override present");
    assert!(m[cap..cap + 90].contains("self.verify_rows_cap()"));
    assert!(l.contains("if batched_verify() { 64 } else { 0 }"));
    let ld = include_str!("../weight_loader/glm5_next_load/loader.rs");
    assert!(ld.contains(".max(wide_rows.unwrap_or(0)).max(bv_rows);"), "KDA rows");
    assert!(
        ld.contains(".max(crate::glm5next_layer::prefill_rows_ffn()).max(bv_rows);"),
        "MLP rows"
    );
}
