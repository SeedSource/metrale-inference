// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Source-level tests of the per-sequence release contract: one release hook
//! on `TransformerLayer`, and a release chokepoint with no `env::var` call and no
//! call-site filter.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

// 2026-09-26: `sequence.rs` then `sequence_compact.rs`, so the text after
// `fn free_sequence_dispatch` also covers slot reuse and compaction.
// `the_sequence_split_is_fully_scanned` pins this list to the directory.
const SEQUENCE_RS: &str = concat!(
    include_str!("../../../model-engine/src/model/trait_impl/sequence.rs"),
    include_str!("../../../model-engine/src/model/trait_impl/sequence_compact.rs"),
);
// 2026-09-26: `TransformerLayer` and its supertraits, one file each, scanned as one text.
const TRANSFORMER_LAYER_RS: &str = concat!(
    include_str!("transformer_layer.rs"),
    include_str!("transformer_layer/aux_state.rs"),
    include_str!("transformer_layer/capabilities.rs"),
    include_str!("transformer_layer/graph_hooks.rs"),
    include_str!("transformer_layer/split_prefill.rs"),
    include_str!("transformer_layer/weight_setup.rs"),
    include_str!("transformer_layer/write_on_accept.rs"),
);

/// 2026-09-26: `TransformerLayer` and its supertraits declare `release_state` once and no
/// `free_state`, so layer state has one release hook. `DraftProposer::free_state` belongs to
/// another trait and is outside the scanned files.
#[test]
fn transformer_layer_has_exactly_one_release_hook() {
    assert_eq!(
        TRANSFORMER_LAYER_RS.matches("fn release_state").count(),
        1,
        "TransformerLayer must declare release_state exactly once"
    );
    assert_eq!(
        TRANSFORMER_LAYER_RS.matches("fn free_state").count(),
        0,
        "TransformerLayer::free_state was retired in favour of #821's release_state; \
         reintroducing it recreates the two-owner split"
    );
}

/// 2026-09-26: No `env::var` call appears from `fn free_sequence_dispatch` to the end of
/// model-engine `trait_impl/sequence_compact.rs`, and `METRALE_GLM_DSA_STATE_LEAK` does not
/// occur in either file.
#[test]
fn the_release_chokepoint_has_no_env_escape_hatch() {
    let dispatch = SEQUENCE_RS
        .split_once("fn free_sequence_dispatch")
        .expect("free_sequence_dispatch exists")
        .1;
    assert!(
        !dispatch.contains("env::var"),
        "free_sequence_dispatch must not read an environment variable: a release that can be \
         switched off is not an invariant"
    );
    assert_eq!(
        SEQUENCE_RS.matches("METRALE_GLM_DSA_STATE_LEAK").count(),
        0,
        "the DSA leak switch must stay deleted"
    );
}

/// 2026-09-25: The chokepoint calls `release_state` for every layer with no
/// `uses_ssm_pool()` filter at the call site; each impl decides by the state's type. The
/// `.free_state(` calls are proposers' (`DraftProposer`): the proposer slot's and, on a hybrid
/// MTP + DFlash serve (2026-10-10), the DFlash drafter's.
#[test]
fn the_chokepoint_releases_every_layer_without_a_site_filter() {
    let dispatch = SEQUENCE_RS
        .split_once("fn free_sequence_dispatch")
        .expect("free_sequence_dispatch exists")
        .1;
    assert!(
        dispatch.contains("release_state("),
        "the chokepoint must call release_state"
    );
    let frees: Vec<&str> = dispatch
        .match_indices(".free_state(")
        .map(|(i, _)| {
            dispatch[..i]
                .rsplit(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or("")
        })
        .collect();
    assert!(
        !frees.is_empty() && frees.iter().all(|r| matches!(*r, "proposer" | "drafter")),
        "only DraftProposer::free_state calls (on `proposer` / the hybrid `drafter`) may remain in \
         the chokepoint, a different trait from the layers' release_state; found receivers {frees:?}"
    );
    assert!(
        !dispatch.contains("uses_ssm_pool()"),
        "no call-site pooled-ness filter: impls refuse by type"
    );
}

/// 2026-09-26: `SEQUENCE_RS` holds every `sequence*.rs` file of model-engine
/// `trait_impl/`. A further split of `sequence.rs` into a sibling file fails
/// here until the new file is added to the `concat!`.
#[test]
fn the_sequence_split_is_fully_scanned() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../model-engine/src/model/trait_impl");
    let mut found: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("readable trait_impl entry").file_name())
        .filter_map(|n| n.into_string().ok())
        .filter(|n| n.starts_with("sequence") && n.ends_with(".rs"))
        .collect();
    found.sort();
    assert_eq!(found, ["sequence.rs", "sequence_compact.rs"]);
}
