// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Glm5NextTextSkeleton` checked against the GLM-5.3 reference checkpoint:
//! topology, residual wiring, state plan, structural weight accounting, stop tokens, and
//! the weight and state budgets.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! The fixtures come from `LibertAIDAI/GLM-5.3-Flash-NVFP4` snapshot `9e0d74e3…`:
//!   * `…-config.json`: the checkpoint's config, parsed by the `glm5_next` parser.
//!   * `…-structural.txt`: every non-MLP text tensor name, with real layer indices,
//!     1,047 rows.
//!   * `…-families.tsv`: tensor count and bytes per tensor family.

use std::collections::BTreeSet;

use metrale_config::{LayerType, parse_config};
use metrale_model_arch::glm5next_skeleton::{
    FinalStep, Glm5NextTextSkeleton, Mixer, Mlp, ResidualStep, Site, StateKind,
};

const CONFIG: &str = include_str!("fixtures/glm53-nvfp4-9e0d74e3-config.json");
const STRUCTURAL: &str = include_str!("fixtures/glm53-nvfp4-9e0d74e3-structural.txt");

const EXPECTED_STRUCTURAL: usize = 1_047;
const N_KDA: usize = 34;
const N_DSA: usize = 11;

fn skeleton() -> Glm5NextTextSkeleton {
    let cfg = parse_config(CONFIG).expect("the real checkpoint config parses");
    Glm5NextTextSkeleton::from_config(&cfg).expect("skeleton builds from the real config")
}

fn available() -> BTreeSet<String> {
    STRUCTURAL
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

#[test]
fn fixture_is_the_reference_checkpoint() {
    assert_eq!(
        available().len(),
        EXPECTED_STRUCTURAL,
        "structural fixture drifted"
    );
}

/// 2026-09-25: 34 KDA and 11 DSA text layers, with layer 45 (MTP) held outside the text
/// stack.
#[test]
fn topology_is_34_kda_11_dsa_and_mtp_is_not_in_the_text_stack() {
    let s = skeleton();
    assert_eq!(s.layers.len(), 45, "the text stack is 45 layers, not 46");
    let kda = s.layers.iter().filter(|l| l.mixer == Mixer::Kda).count();
    let dsa = s.layers.iter().filter(|l| l.mixer == Mixer::Dsa).count();
    assert_eq!((kda, dsa), (N_KDA, N_DSA));
    assert!(s.layers.iter().all(|l| !l.is_mtp));

    let mtp = s.mtp.expect("num_nextn_predict_layers = 1");
    assert_eq!(mtp.index, 45);
    assert!(mtp.is_mtp);
    // 2026-09-25: The MTP layer's mixer is DSA.
    assert_eq!(mtp.mixer, Mixer::Dsa);
}

/// 2026-09-25: The sparse layers are the ones the checkpoint config lists.
#[test]
fn sparse_layers_are_the_checkpoint_s_own_list() {
    let s = skeleton();
    let dsa: Vec<usize> = s
        .layers
        .iter()
        .filter(|l| l.mixer == Mixer::Dsa)
        .map(|l| l.index)
        .collect();
    assert_eq!(dsa, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]);
}

/// 2026-09-25: `first_k_dense_replace = 3` makes layers 0..=2 dense; the other 42 text
/// layers and the MTP layer route to experts.
#[test]
fn layers_0_to_2_are_dense_and_everything_after_routes() {
    let s = skeleton();
    let dense: Vec<usize> = s
        .layers
        .iter()
        .filter(|l| l.mlp == Mlp::Dense)
        .map(|l| l.index)
        .collect();
    assert_eq!(dense, vec![0, 1, 2]);
    assert_eq!(
        s.layers.iter().filter(|l| l.mlp == Mlp::RoutedMoe).count(),
        42
    );
    assert_eq!(s.mtp.unwrap().mlp, Mlp::RoutedMoe);
}

/// 2026-09-25: Layer 45 carries no `hc_*` tensor; all 270 are on layers 0..=44.
#[test]
fn hyper_connection_is_on_every_text_layer_and_on_no_mtp_layer() {
    let s = skeleton();
    assert!(s.layers.iter().all(|l| l.hyper_connection));
    assert!(!s.mtp.unwrap().hyper_connection);

    let hc = available().iter().filter(|n| n.contains(".hc_")).count();
    assert_eq!(hc, 270, "6 mHC tensors x 45 text layers");
    assert!(
        !available()
            .iter()
            .any(|n| n.starts_with("model.language_model.layers.45.hc_")),
        "the MTP layer must not carry a hyper-connection"
    );
}

/// 2026-09-25: The residual path of a text layer, in order: save residual → `hc_pre` → norm
/// → sublayer → `hc_post`, once per site.
#[test]
fn residual_wiring_matches_the_reference_decoder_layer() {
    let s = skeleton();
    let l = s.layers[0];
    assert_eq!(
        s.residual_plan(&l),
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
        ]
    );
    // 2026-09-25: Every text layer has the same 10 steps, whatever its mixer or MLP kind.
    for l in &s.layers {
        assert_eq!(s.residual_plan(l).len(), 10, "layer {}", l.index);
    }
}

/// 2026-09-25: The MTP layer has no hyper-connection, so its plan has no mHC steps.
#[test]
fn mtp_residual_path_has_no_mhc_steps() {
    let s = skeleton();
    let plan = s.residual_plan(&s.mtp.unwrap());
    assert!(
        !plan
            .iter()
            .any(|st| matches!(st, ResidualStep::HcPre(_) | ResidualStep::HcPost(_))),
        "{plan:?}"
    );
    assert_eq!(plan.len(), 6);
}

/// 2026-09-25: The final collapse is an unweighted mean with no parameters. The DeepSeek-V4
/// `hc_head` kernel is a learned weighted sum that reads `hc_head.*` weights, and this
/// checkpoint has none.
#[test]
fn final_collapse_is_a_parameterless_mean_then_norm() {
    let s = skeleton();
    assert_eq!(
        s.final_plan(),
        [
            FinalStep::HyperHeadMean,
            FinalStep::Norm("model.language_model.norm.weight"),
            FinalStep::LmHead,
        ]
    );
    assert!(
        !available().iter().any(|n| n.contains("hc_head")),
        "a learned final collapse would need hc_head weights; the checkpoint has none"
    );
}

/// 2026-09-25: Every layer has exactly one state kind: 34 KDA layers carry recurrent state,
/// and 12 (11 text DSA layers and the MTP layer) use KV blocks.
#[test]
fn state_plan_splits_recurrent_from_paged_and_covers_every_layer() {
    let s = skeleton();
    let plan = s.state_plan();
    assert_eq!(plan.len(), 46, "45 text layers + the MTP layer");
    assert_eq!(s.kda_state_layers().len(), N_KDA);
    assert_eq!(s.kv_cache_layers().len(), N_DSA + 1);
    assert_eq!(plan[&45], StateKind::SparseKv);
    assert_eq!(plan[&0], StateKind::KdaRecurrent);
    assert_eq!(plan[&3], StateKind::SparseKv);
    assert_eq!(
        s.kda_state_layers().len() + s.kv_cache_layers().len(),
        plan.len()
    );
}

/// 2026-09-25: The structural contract matches the checkpoint exactly: nothing missing and
/// nothing unexpected. MLP/MoE and vision names are counted as deferred.
#[test]
fn structural_binding_closes_exactly() {
    let s = skeleton();
    let acc = s.account(&available());
    assert!(
        acc.missing.is_empty(),
        "{} structural tensor(s) absent from the checkpoint, first 10: {:?}",
        acc.missing.len(),
        &acc.missing[..acc.missing.len().min(10)]
    );
    assert!(
        acc.unexpected.is_empty(),
        "{} text tensor(s) the skeleton was never taught, first 10: {:?}",
        acc.unexpected.len(),
        &acc.unexpected[..acc.unexpected.len().min(10)]
    );
    assert_eq!(acc.bound, acc.required);
    assert!(acc.is_complete());
    // 2026-09-25: The fixture holds only non-MLP text tensors, so nothing is deferred.
    assert_eq!(acc.deferred, 0);
    assert_eq!(acc.required, EXPECTED_STRUCTURAL);
}

/// 2026-09-25: Per-signature counts, so a drift names the family that changed:
/// 34 x 23 + 11 x 22 + 1 x 20 + 3 non-layer = 1,047.
#[test]
fn per_layer_structural_counts_match_the_three_measured_signatures() {
    let s = skeleton();
    for l in &s.layers {
        let n = s.structural_tensors(l).len();
        let want = match l.mixer {
            Mixer::Kda => 23,
            Mixer::Dsa => 22,
        };
        assert_eq!(n, want, "layer {} ({:?})", l.index, l.mixer);
    }
    assert_eq!(s.structural_tensors(&s.mtp.unwrap()).len(), 20);
    assert_eq!(
        N_KDA * 23 + N_DSA * 22 + 20 + 3,
        EXPECTED_STRUCTURAL,
        "the three signatures must account for the whole structural surface"
    );
}

/// 2026-09-25: A missing structural tensor, here the indexer `k_norm.bias`, makes the
/// accounting incomplete.
#[test]
fn a_dropped_indexer_bias_is_a_hard_failure_not_a_skip() {
    let s = skeleton();
    let mut avail = available();
    assert!(avail.remove("model.language_model.layers.3.self_attn.indexer.k_norm.bias"));
    let acc = s.account(&avail);
    assert!(!acc.is_complete());
    assert_eq!(acc.missing.len(), 1);
}

/// 2026-09-25: An unknown text tensor is reported as unexpected. Only `.mlp.` and vision
/// names are deferred.
#[test]
fn an_untaught_text_tensor_is_refused_not_absorbed() {
    let s = skeleton();
    let mut avail = available();
    avail.insert("model.language_model.layers.9.self_attn.wat".to_string());
    let acc = s.account(&avail);
    assert!(!acc.is_complete());
    assert_eq!(acc.unexpected.len(), 1);

    // 2026-09-25: An MoE tensor is deferred and counted.
    let mut avail2 = available();
    avail2.insert("model.language_model.layers.9.mlp.experts.0.down_proj.weight".to_string());
    let acc2 = s.account(&avail2);
    assert!(acc2.is_complete());
    assert_eq!(acc2.deferred, 1);
}

/// 2026-09-25: `from_config` refuses a layer type GLM-5.3 does not have.
#[test]
fn an_unexpected_layer_type_is_refused() {
    let mut cfg = parse_config(CONFIG).expect("parses");
    cfg.layer_types[5] = LayerType::SlidingAttention;
    assert!(Glm5NextTextSkeleton::from_config(&cfg).is_err());
}

/// 2026-09-25: The config's three `eos_token_id` entries all survive parsing, and the first,
/// 154820, is the primary.
#[test]
fn all_three_glm_stop_tokens_survive_parsing() {
    let cfg = parse_config(CONFIG).expect("parses");
    assert_eq!(cfg.eos_token_id, 154820, "<|endoftext|> is the primary");
    assert_eq!(cfg.eos_ids(), vec![154820, 154827, 154829]);
    for id in [154820u32, 154827, 154829] {
        assert!(cfg.is_eos(id), "generation must stop on {id}");
    }
    assert!(!cfg.is_eos(154828));
}

const FAMILIES: &str = include_str!("fixtures/glm53-nvfp4-9e0d74e3-families.tsv");

fn families() -> Vec<(String, usize, usize)> {
    FAMILIES
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut p = l.split('\t');
            (
                p.next().unwrap().to_string(),
                p.next().unwrap().parse().unwrap(),
                p.next().unwrap().parse().unwrap(),
            )
        })
        .collect()
}

/// 2026-09-25: The family counts add up to the checkpoint's 113,074 tensors, and the families
/// that are not loaded are named with `NOT LOADED`.
#[test]
fn the_whole_checkpoint_is_accounted_for_and_the_excluded_set_is_explicit() {
    let f = families();
    let total: usize = f.iter().map(|(_, c, _)| c).sum();
    assert_eq!(total, 113_074, "checkpoint tensor count drifted");
    let excluded: Vec<&str> = f
        .iter()
        .filter(|(n, ..)| n.contains("NOT LOADED"))
        .map(|(n, ..)| n.as_str())
        .collect();
    // 2026-09-25: The excluded families are the vision tower and the six layer-45 families.
    assert!(excluded.iter().any(|n| n.starts_with("vision")));
    assert_eq!(excluded.len(), 7, "excluded families: {excluded:?}");
    let excl_t: usize = f
        .iter()
        .filter(|(n, ..)| n.contains("NOT LOADED"))
        .map(|(_, c, _)| c)
        .sum();
    assert_eq!(excl_t, 347 + 2592 + 14 + 4 + 3 + 2 + 2);
}

/// 2026-09-25: Text weight memory and the EP=2 split, from the fixture's byte counts.
#[test]
fn text_model_weight_memory_and_the_ep2_split() {
    let f = families();
    let text: usize = f
        .iter()
        .filter(|(n, ..)| !n.contains("NOT LOADED"))
        .map(|(_, _, b)| b)
        .sum();
    let experts: usize = f
        .iter()
        .filter(|(n, ..)| n == "moe_experts_NVFP4")
        .map(|(_, _, b)| b)
        .sum();
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    assert!((gib(text) - 176.086).abs() < 0.01, "{}", gib(text));
    assert!((gib(experts) - 159.469).abs() < 0.01, "{}", gib(experts));
    // 2026-09-25: At EP=2 only the routed experts shard; everything else replicates.
    let per_rank = experts / 2 + (text - experts);
    assert!((gib(per_rank) - 96.352).abs() < 0.01, "{}", gib(per_rank));
    assert!(gib(text) > 121.0, "EP=1 would fit and this test is stale");
    assert!(gib(per_rank) < 112.0, "EP=2 no longer fits either");
}

/// 2026-09-25: The per-sequence state budget, split into what is fixed and what grows.
#[test]
fn per_sequence_state_budget_separates_fixed_from_growing() {
    let cfg = parse_config(CONFIG).expect("parses");
    let s = skeleton();
    let b = s.state_budget(&cfg, 0);
    // 2026-09-25: `state_budget` sizes the KDA recurrent state at 4 bytes per element.
    assert_eq!(b.kda_recurrent % 4, 0);
    assert!(b.fixed() > 0 && b.per_token() > 0);
    // 2026-09-25: Growth per token is the MLA KV plus the indexer cache of the 11 sparse text
    // layers, and the KV part is the larger.
    assert!(b.dsa_kv_per_token > b.dsa_indexer_per_token);
    let short = b.for_sequence(1);
    let long = b.for_sequence(4096);
    assert_eq!(long - short, 4095 * b.per_token());
    // 2026-09-25: `per_rank(2)` halves the KDA state and leaves the mHC highway whole.
    let r = b.per_rank(2);
    assert_eq!(r.kda_recurrent, b.kda_recurrent / 2);
    assert_eq!(r.mhc_highway_per_token, b.mhc_highway_per_token);
}

/// 2026-10-03: The preflight charge at msl 131,072, max batch 4, MTP on, from the real config:
/// eager (lever off) 4 x 807,213,588 B, the figure the 2026-10-03 longctx boot logged as
/// "3079 MB"; lazy (`METRALE_DSA_INDEXER_LAZY=1`, default pool) a 784 MiB shared pool plus the
/// eager `valid` flags and proposer scratch per sequence, 791 MiB, 2.40 GB more for the KV pool.
#[test]
fn lazy_indexer_reserve_at_131k_mb4_frees_2_4_gb_for_kv() {
    use metrale_model_arch::seq_state_reserve::{lazy_indexer_reserve, per_sequence_state_bytes};
    let cfg = parse_config(CONFIG).expect("parses");
    let eager = per_sequence_state_bytes(&cfg, 131_072, true)
        .unwrap()
        .for_batch(4);
    assert_eq!(eager, 3_228_854_352);
    assert_eq!(eager >> 20, 3079);

    let lazy = lazy_indexer_reserve(&cfg, 131_072, true, 4, None)
        .unwrap()
        .expect("glm5_next");
    assert_eq!(lazy.pool.shape.bufs_per_seq(), 24);
    assert_eq!(lazy.pool.limit_bytes, 784 << 20);
    assert_eq!(lazy.eager_per_seq, 12 * 131_072 + 334_356);
    assert_eq!(lazy.for_batch(4), 829_712_464);
    assert_eq!(eager - lazy.for_batch(4), 2_399_141_888);

    // 2026-10-03: An explicit pool is honoured up to what 4 full-length sequences can map.
    let big = lazy_indexer_reserve(&cfg, 131_072, true, 4, Some(1.6))
        .unwrap()
        .unwrap();
    assert_eq!(big.pool.limit_bytes, (1.6 * (1u64 << 30) as f64) as usize);
    let huge = lazy_indexer_reserve(&cfg, 131_072, true, 4, Some(64.0))
        .unwrap()
        .unwrap();
    assert_eq!(huge.pool.limit_bytes, 4 * 24 * (32 << 20));
    // 2026-10-03: MTP off: no proposer buffers, no look-ahead granule.
    let nospec = lazy_indexer_reserve(&cfg, 131_072, false, 4, None)
        .unwrap()
        .unwrap();
    assert_eq!(nospec.pool.shape.bufs_per_seq(), 22);
    assert_eq!(nospec.pool.limit_bytes, 4 * 22 * 4 * (2 << 20));
}
