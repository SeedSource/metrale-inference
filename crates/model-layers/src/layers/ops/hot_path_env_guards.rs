// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Source-level tests that confine environment reads on per-token, per-layer and per-forward paths to named functions.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types. The tests check that in each file of `GUARDED`,
//! `std::env::var` appears in code only inside the functions its allow list names, and that
//! only the files in `ALLOWED` contain the call text for `ModelLevers::from_env`.
//!
//! The checks read source text because "who may read the environment" is not observable at run
//! time. To guard a new file, move its reads into [`super::ModelLevers`] (or the module's own
//! levers struct) and read the resolved field, rather than extending an allow list.

/// 2026-09-25: Files on a per-token, per-layer or per-forward path, each with the functions in it
/// that may read the environment. An empty list means the file may not read it at all. Paths are
/// relative to `crates/model-layers/src`.
const GUARDED: [(&str, &[&str]); 53] = [
    ("layers/dense_ffn_weights.rs", &[]),
    ("layers/dense_ffn_init.rs", &[]),
    ("layers/dense_ffn_load.rs", &[]),
    ("layers/dense_ffn_overlays.rs", &[]),
    ("layers/dense_ffn_decode.rs", &[]),
    ("layers/dense_ffn_decode_batch.rs", &[]),
    ("layers/dense_ffn_prefill.rs", &[]),
    ("layers/dense_ffn_prefill_fp8.rs", &[]),
    ("layers/dense_ffn_prefill_nvfp4.rs", &[]),
    ("layers/dense_ffn_nvfp4_plan.rs", &[]),
    (
        "layers/dense_ffn.rs",
        &[
            // 2026-09-25: Kernel-choice helpers cached in a `OnceLock`, so the choice is fixed after
            // the first read, as CUDA graph capture needs.
            "mmq_small_tile_enabled",
            "mmq_tile64_enabled",
            // 2026-09-25: Called by the weight loaders at load time.
            "finalize_q4k_load",
            "finalize_nvfp4_mmq_load",
        ],
    ),
    ("layers/moe/forward_prefill_routed.rs", &[]),
    (
        "../../model-engine/src/model/trait_impl/decode_a.rs",
        &[
            // 2026-09-25: Both cache their read in a `OnceLock`, so it happens once per process. They
            // are named one by one because the scanner does not recognise a `OnceLock`; a
            // reviewer checks that and records it here.
            "redzone_range_file",
            "redzone_every",
        ],
    ),
    ("layers/moe/forward.rs", &[]),
    ("layers/moe/forward/debug_dumps.rs", &[]),
    ("layers/moe/forward/ep_reduce.rs", &[]),
    ("layers/moe/forward/route.rs", &[]),
    ("layers/moe/forward/zero_expert.rs", &[]),
    ("layers/moe/forward_batched_gate.rs", &[]),
    ("layers/moe/forward_k2.rs", &[]),
    ("layers/mtp_head/forward.rs", &[]),
    ("layers/mtp_head/forward/attend.rs", &[]),
    ("layers/mtp_head/forward/host_logits.rs", &[]),
    (
        "layers/mtp_head/draft_proposer.rs",
        &[
            // 2026-09-25: Reads `METRALE_MTP_DRAFT_CONF` through `draft_conf_tau`. Its only caller,
            // `run_mtp_propose_inner` (model-engine impl_b3.rs), calls it only when the resolved
            // `draft_conf_tau` lever is above 0.
            "last_confidence",
        ],
    ),
    ("../../model-engine/src/model/impl_b3.rs", &[]),
    (
        "../../model-engine/src/model/trait_impl/decode_a2.rs",
        &[
            // 2026-09-25: Caches its read of `METRALE_NO_DECODE_GRAPHS_MULTISEQ` in a `OnceLock`, so
            // it happens once per process.
            "multiseq_graphs_enabled",
        ],
    ),
    (
        "../../model-engine/src/model/trait_impl/decode_a2/pad_states.rs",
        &[],
    ),
    (
        "../../model-engine/src/model/trait_impl/decode_a2/perseq.rs",
        &[],
    ),
    ("../../model-arch/src/nemotron_mamba2/prefill.rs", &[]),
    ("../../model-arch/src/nemotron_moe/prefill_sorted.rs", &[]),
    (
        "../../model-arch/src/nemotron_moe/prefill_shared_up.rs",
        &[],
    ),
    // 2026-09-25: The `ModelLevers` resolution. It is listed, with `from_env` allowed, so that a
    // read anywhere else in the file, a second resolution path, fails the test.
    ("layers/ops/model_levers_resolve.rs", &["from_env"]),
    // 2026-09-25: DFlash drafter forward files. `dflash_head/from_weights.rs` is not listed: it
    // builds the head, which is where its reads belong.
    ("../../model-arch/src/dflash_head/forward_block.rs", &[]),
    (
        "../../model-arch/src/dflash_head/forward_block/ctx_prologue.rs",
        &[],
    ),
    (
        "../../model-arch/src/dflash_head/forward_block/dims.rs",
        &[],
    ),
    (
        "../../model-arch/src/dflash_head/forward_block/embed.rs",
        &[],
    ),
    (
        "../../model-arch/src/dflash_head/forward_block/graphs.rs",
        &[],
    ),
    (
        "../../model-arch/src/dflash_head/forward_block/host_drafts.rs",
        &[],
    ),
    (
        "../../model-arch/src/dflash_head/forward_block/tail.rs",
        &[],
    ),
    (
        "../../model-arch/src/dflash_head/forward_block/tests.rs",
        &[],
    ),
    (
        "../../model-arch/src/dflash_head/forward_block_layer.rs",
        &[],
    ),
    (
        "../../model-arch/src/dflash_head/forward_block_layer_paged.rs",
        &[],
    ),
    (
        "../../model-arch/src/dflash_head/forward_block_layer_paged/contig_attn.rs",
        &[],
    ),
    (
        "../../model-arch/src/dflash_head/forward_block_layer_paged/post_attn.rs",
        &[],
    ),
    (
        "../../model-arch/src/dflash_head/forward_block_layer_paged/pre_attn.rs",
        &[],
    ),
    ("../../model-arch/src/dflash_head/propose.rs", &[]),
    ("../../model-arch/src/dflash_head/markov.rs", &[]),
    ("../../model-arch/src/dflash_head/dflash2.rs", &[]),
    ("../../model-arch/src/dflash_head/precompute_ctx_kv.rs", &[]),
    // 2026-10-09: The two-rank drafter's propose path (METRALE_DFLASH_DRAFT_TP2).
    ("../../model-arch/src/dflash_head/propose_prep.rs", &[]),
    ("../../model-arch/src/dflash_head/tp/collectives.rs", &[]),
    ("../../model-arch/src/dflash_head/tp/follower.rs", &[]),
    ("../../model-arch/src/dflash_head/tp/leader.rs", &[]),
];

fn src_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// 2026-09-25: Every environment read in `text`, as `(line number, fn name)`.
///
/// Each line is cut at its first `//`, so text in a comment is not counted as a read.
///
/// A read is charged to the most recent line above it that starts with a `fn` declaration.
/// Closures do not declare `fn`, so a read inside one is charged to the enclosing function.
fn env_reads(text: &str) -> Vec<(usize, String)> {
    let mut current = "<file scope>".to_string();
    let mut found = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        for prefix in ["pub(crate) fn ", "pub(super) fn ", "pub fn ", "fn "] {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                current = rest
                    .split(['(', '<'])
                    .next()
                    .unwrap_or("?")
                    .trim()
                    .to_string();
                break;
            }
        }
        let code = line.split("//").next().unwrap_or("");
        if code.contains("std::env::var") {
            found.push((i + 1, current.clone()));
        }
    }
    found
}

/// 2026-09-25: A read in a guarded file, in a function its allow list does not name, fails this
/// test.
#[test]
fn the_environment_is_read_only_where_it_is_allowed() {
    let root = src_root();
    let mut offenders = Vec::new();
    for (rel, allowed) in GUARDED {
        let path = root.join(rel);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{} is a guarded path: {e}", path.display()));
        for (line, func) in env_reads(&text) {
            if !allowed.contains(&func.as_str()) {
                offenders.push(format!("{rel}:{line} in `{func}`"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "environment reads on a per-token / per-layer / per-forward path. Each \
         takes the process-wide env lock, which serialises concurrent decode \
         threads. Resolve the variable ONCE into a levers struct and read the \
         field instead — do not extend the allow list: {offenders:?}"
    );
}

/// 2026-09-25: The scanner tells a call from a comment about a call, and charges each read to its
/// own function. Without this the test above could pass while measuring nothing.
#[test]
fn the_scanner_discriminates() {
    let sample = "\
fn cold() {
    let a = std::env::var(\"X\").ok();
}
pub fn hot() {
    // this used to call std::env::var(\"Y\") and no longer does
    let b = 1;
}
pub(super) fn hotter(&self) {
    let c = std::env::var_os(\"Z\").is_some();
}
";
    let reads = env_reads(sample);
    assert_eq!(
        reads,
        vec![(2, "cold".to_string()), (9, "hotter".to_string())],
        "the comment inside `hot` must not count, and `var_os` inside \
         `hotter` must"
    );
}

/// 2026-09-25: Only the files in `ALLOWED` may contain the call text for `ModelLevers::from_env`.
/// `ModelLevers::get` (model_levers_resolve.rs) caches the resolution in a `OnceLock`; every
/// other reader uses `get()` or the `levers` its context carries. The search covers every `.rs`
/// file under `crates/`, skipping `target` directories.
#[test]
fn from_env_is_called_only_where_it_is_allowed() {
    const ALLOWED: [&str; 3] = [
        // 2026-09-25: Holds no such call. `ModelLevers::get` lives in model_levers_resolve.rs and
        // passes `Self::from_env`, which this text search does not match.
        "crates/model-layers/src/layers/ops/model_levers.rs",
        // 2026-09-25: The model build resolves a fresh copy so it can overwrite `max_decode_seqs`.
        "crates/model-engine/src/model/impl_a1.rs",
        // 2026-09-25: This file, whose assert message contains the searched text.
        "crates/model-layers/src/layers/ops/hot_path_env_guards.rs",
    ];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let mut offenders = Vec::new();
    let mut stack = vec![root.join("crates")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "target") {
                    continue;
                }
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                if !text.contains("ModelLevers::from_env()") {
                    continue;
                }
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                if !ALLOWED.contains(&rel.as_str()) {
                    offenders.push(rel);
                }
            }
        }
    }
    offenders.sort();
    assert!(
        offenders.is_empty(),
        "ModelLevers::from_env() re-reads ~40 env vars under a global lock. \
         These call it instead of the once-resolved ModelLevers::get(): {offenders:?}"
    );
}
