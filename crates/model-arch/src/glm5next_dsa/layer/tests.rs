// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host tests of `layer.rs`: kernel choices and guard ordering checked by reading
//! the source (`include_str!`), and the `batch_select_enabled` table. The decode numerics
//! are checked on a GPU by `examples/glm5next_dsa_decode_gate.rs`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 16_384,
    }
}

/// 2026-09-25: `rms_norm` computes `x * rms * (1 + w)` and `rms_norm_vanilla` computes
/// `x * rms * w`, with the same signature. GLM needs the vanilla one, and the shapes cannot
/// tell them apart, so the source is checked for the resolve call.
#[test]
fn the_layer_takes_the_vanilla_rmsnorm_not_the_plus_one_variant() {
    let src = include_str!("../layer.rs");
    assert!(
        src.contains(r#"gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")"#),
        "GLM uses x*rms*w; `rms_norm` applies x*rms*(1+w) and would shift every norm"
    );
    for (file, text) in [
        ("layer.rs", src),
        ("layer/decode_k.rs", include_str!("decode_k.rs")),
        ("layer/rows.rs", include_str!("rows.rs")),
        ("layer/workspace.rs", include_str!("workspace.rs")),
        ("layer/proj_gemm.rs", include_str!("proj_gemm.rs")),
        ("layer/row_batch.rs", include_str!("row_batch.rs")),
        ("layer/wide.rs", include_str!("wide.rs")),
        ("layer/xseq.rs", include_str!("xseq.rs")),
    ] {
        assert!(
            !text.contains(r#""rms_norm", "rms_norm""#),
            "the +1-offset RMSNorm must not appear in a GLM path ({file})"
        );
    }
}

/// 2026-10-01: `METRALE_GLM_DSA_ROW_BATCH` is reachable only where the batched selector runs,
/// and its indexer projections stay on the batched GEMVs: `gemm` (whose arm above
/// `DENSE_GEMV_BATCHM_MAX_M` is cuBLASLt) appears once in `row_batch.rs`, for `o_absorb`, which
/// the row loop runs through the same call.
#[test]
fn the_row_batch_is_prefill_only_and_off_the_cublas_arm() {
    let entry = "let select_rows =
            batch_select_enabled(w.q_idx_rows.0 != 0, is_prefill, ctx.graph_capture, k);
        if let Some(rb) = self.row_batch_ready(gpu, select_rows, stream) {";
    assert!(
        include_str!("decode_k.rs").contains(entry),
        "the row batch must sit behind batch_select_enabled"
    );
    let rb = include_str!("row_batch.rs");
    let gate = "let ok = batch_select && self.persist_bt && kernels && same_stream;";
    assert!(rb.contains(gate), "row_batch_ready lost a condition");
    assert_eq!(rb.matches("gemm(").count(), 1, "only o_absorb uses gemm");
    assert!(rb.contains("self.weights.o_absorb"));
}

/// 2026-09-25: The indexer's `k_norm` is a LayerNorm with a bias: the layer launches the
/// `k_norm` kernel handle and passes `k_norm_bias`.
#[test]
fn the_indexer_norm_passes_a_bias() {
    let src = include_str!("../layer.rs");
    assert!(
        src.contains("k_norm_bias"),
        "indexer.k_norm.bias is REQUIRED; a weight-only bind drops mean subtraction too"
    );
    assert!(src.contains("self.select_kernels.k_norm"));
}

/// 2026-09-25: Q reaches the decode kernel in latent space. A raw `q_b_proj` is a well-formed
/// tensor of the wrong width per head (256 against 512) in the wrong space.
#[test]
fn q_is_absorbed_to_the_latent_width() {
    let c = cfg();
    assert_eq!(c.kv_lora_rank, 512);
    assert_ne!(
        c.qk_nope_head_dim, c.kv_lora_rank,
        "if these were equal the absorption mistake would be undetectable by shape"
    );
    let src = include_str!("../layer.rs");
    assert!(src.contains("q_absorb"));
}

/// 2026-09-25: `decode_k` rewinds an indexer cache that is ahead of the sequence (a rejected
/// draft) and refuses one that is behind (rows never written).
#[test]
fn a_lockstep_drift_is_refused_behind_and_rewound_ahead() {
    let src = include_str!("../layer.rs");
    assert!(
        src.contains("must advance in lockstep"),
        "the drift guard must state why it exists"
    );
    assert!(
        src.contains("st.len().cmp(&seq_len)"),
        "the guard must branch on the DIRECTION of the drift, not merely on inequality"
    );
    assert!(
        src.contains("Ordering::Greater => st.rewind_to(seq_len)?"),
        "AHEAD must rewind — this is the verify-reject path"
    );
    assert!(
        src.contains("rows are MISSING, not merely stale"),
        "BEHIND must still refuse, and say why it is the unrecoverable direction"
    );
}

/// 2026-09-25: `rewind_to` refuses to grow the cache.
#[test]
fn the_indexer_rewind_only_shrinks() {
    let src = include_str!("../state.rs");
    assert!(src.contains("rewind only shrinks"));
}

/// 2026-09-25: The decode takes the same latent pool for K and V: absorbed NoPE MLA caches
/// one latent per token.
#[test]
fn k_and_v_are_the_same_latent_pool() {
    let src = include_str!("../layer.rs");
    assert!(src.contains("v_cache: pool"));
    assert!(src.contains("K and V are the same latent"));
}

/// 2026-09-25: At `max_context` 16,384 and `index_kpool` 4, `max_dsa_context` is 16,384
/// tokens and `plan` gives 4,096 pools there; `Glm5NextDsaWorkspace::new` plans the selection
/// scratch at `max_dsa_context`.
#[test]
fn the_workspace_is_sized_at_the_context_cap() {
    let c = cfg();
    let cap = super::super::state::max_dsa_context(&c);
    assert_eq!(cap, 16_384);
    let geom = super::super::select::DsaSelectGeometry::plan(&c, cap, 1).unwrap();
    assert_eq!(
        geom.n_pools, 4_096,
        "the cap is the largest plannable pool axis"
    );
}

/// 2026-09-25: `indexer_forward` calls `ensure_room(1)` before its first GEMM. `advance(1)` at
/// its end runs after the writes to row `capacity`, which is past the buffers.
#[test]
fn the_indexer_checks_capacity_before_it_writes() {
    let src = include_str!("../layer.rs");
    let body = src
        .split_once("pub fn indexer_forward")
        .expect("indexer_forward must exist")
        .1
        .split_once("fn attend_rows")
        .expect("attend_rows follows indexer_forward")
        .0;
    let guard = body
        .find("state.ensure_room(1)?")
        .expect("indexer_forward must precheck capacity");
    let first_write = body.find("gemm(").expect("indexer_forward writes via gemm");
    assert!(
        guard < first_write,
        "the capacity check must come before the first device write, not after"
    );
}

/// 2026-09-25: In `decode_a`, `verify_b`, `verify_c` and `verify_c2`, the first
/// `check_replay_room` call comes before the first `launch_graph`, and `sync_replayed_step`
/// comes after it. A replay writes indexer rows with no host code in the loop, so the room
/// check has to run before the launch.
#[test]
fn every_graph_replay_checks_room_before_it_launches() {
    let paths: [(&str, &str); 4] = [
        (
            "decode_a",
            include_str!("../../../../model-engine/src/model/trait_impl/decode_a.rs"),
        ),
        (
            "verify_b",
            include_str!("../../../../model-engine/src/model/trait_impl/verify_b.rs"),
        ),
        (
            "verify_c",
            include_str!("../../../../model-engine/src/model/trait_impl/verify_c.rs"),
        ),
        (
            "verify_c2",
            include_str!("../../../../model-engine/src/model/trait_impl/verify_c2.rs"),
        ),
    ];
    for (name, src) in paths {
        let guard = src
            .find("layer.check_replay_room(")
            .unwrap_or_else(|| panic!("{name}: the replay branch must precheck capacity"));
        let launch = src
            .find("self.gpu.launch_graph(")
            .unwrap_or_else(|| panic!("{name}: expected a graph replay"));
        assert!(
            guard < launch,
            "{name}: the room check must precede the FIRST launch_graph — after it, the \
             write has already happened"
        );
        let sync = src
            .find("layer.sync_replayed_step(")
            .unwrap_or_else(|| panic!("{name}: the reconcile must still be there"));
        assert!(
            launch < sync,
            "{name}: the reconcile stays AFTER the launch — moving it would change the \
             A56 rewind semantics this fix must not touch"
        );
    }
}

/// 2026-09-25: Both the composite `Glm5NextLayer`, which the loaded model holds, and
/// `Glm5NextDsaLayer` implement `check_replay_room`.
#[test]
fn the_composite_layer_implements_the_room_check_too() {
    let composite = include_str!("../../glm5next_layer/mod.rs");
    assert!(
        composite.contains("fn check_replay_room"),
        "the COMPOSITE layer must implement it — the inner impl is never reached"
    );
    assert!(
        include_str!("../layer.rs").contains("fn check_replay_room"),
        "the inner DSA layer implements it as well"
    );
}

/// 2026-09-25: The replay pre-check adds the context "DSA replay pre-check" to its error in both
/// layers, so a log shows which check refused; the eager check raises the same base text.
#[test]
fn the_replay_refusal_is_distinguishable_from_the_prefill_one() {
    for src in [
        include_str!("../layer.rs"),
        include_str!("../../glm5next_layer/mod.rs"),
    ] {
        assert!(
            src.contains("DSA replay pre-check"),
            "the replay guard must tag its refusal so a log can name the route"
        );
    }
}

// 2026-09-25: Which passes may take the batched selector (`batch_select_enabled`): a prefill
// sub-chunk of more than one row, never a decode step or a speculative verify, eager or graphed.
// `verify_a` sets `graph_capture` false, so `!graph_capture` alone would let an eager verify in.

use super::super::layer::batch_select_enabled;

#[test]
fn only_a_multi_row_eager_prefill_takes_the_batched_selector() {
    // 2026-09-25: (phase, workspace_ready, is_prefill, graph_capture, k, expected)
    let cases: &[(&str, bool, bool, bool, usize, bool)] = &[
        (
            "prefill sub-chunk, PREFILL_ROWS=16",
            true,
            true,
            false,
            16,
            true,
        ),
        ("prefill tail sub-chunk, k=2", true, true, false, 2, true),
        ("prefill tail sub-chunk, k=1", true, true, false, 1, false),
        ("chunked prefill, second chunk", true, true, false, 16, true),
        ("decode step, k=1, graphed", true, false, true, 1, false),
        ("decode step, k=1, eager", true, false, false, 1, false),
        ("graphed verify K=3", true, false, true, 3, false),
        ("graphed verify K=4", true, false, true, 4, false),
        (
            "EAGER verify K=3 (METRALE_GLM_VERIFY_GRAPHS=0)",
            true,
            false,
            false,
            3,
            false,
        ),
        (
            "EAGER verify K=2 (verify_b, HSS engaged)",
            true,
            false,
            false,
            2,
            false,
        ),
        (
            "EAGER verify K=4 (verify_c2, LoRA eager)",
            true,
            false,
            false,
            4,
            false,
        ),
        (
            "verify_a generic N-token (graph_capture hard false)",
            true,
            false,
            false,
            5,
            false,
        ),
        (
            "kill-switch METRALE_DSA_SELECT_ROWS=0, prefill",
            false,
            true,
            false,
            16,
            false,
        ),
        ("kill-switch, eager verify", false, false, false, 3, false),
    ];
    for &(phase, ws, pf, gc, k, want) in cases {
        assert_eq!(
            batch_select_enabled(ws, pf, gc, k),
            want,
            "{phase}: workspace={ws} is_prefill={pf} graph_capture={gc} k={k}"
        );
    }
}

/// 2026-09-26: `forward_k` is `pub(in crate::glm5next_layer)`, so every caller is in a
/// `glm5next_layer/` file. It has two call sites, both in `mod.rs`: the prefill sub-chunk
/// loop (`is_prefill` true) and the speculative verify (`false`). A third caller in any
/// module file fails this test and has to choose.
#[test]
fn forward_k_has_two_callers_and_the_verify_one_is_not_prefill() {
    let src = include_str!("../../glm5next_layer/mod.rs");
    let module = concat!(
        include_str!("../../glm5next_layer/comm_overlap.rs"),
        include_str!("../../glm5next_layer/dense_fp8.rs"),
        include_str!("../../glm5next_layer/dense_fp8_gw.rs"),
        include_str!("../../glm5next_layer/dense_fp8_min_rows.rs"),
        include_str!("../../glm5next_layer/mod.rs"),
        include_str!("../../glm5next_layer/levers.rs"),
        include_str!("../../glm5next_layer/levers_comm.rs"),
        include_str!("../../glm5next_layer/prefetch.rs"),
        include_str!("../../glm5next_layer/profile.rs"),
        include_str!("../../glm5next_layer/scratch_union.rs"),
        include_str!("../../glm5next_layer/seq_parallel.rs"),
        include_str!("../../glm5next_layer/seq_parallel/premix.rs"),
        include_str!("../../glm5next_layer/state.rs"),
        include_str!("../../glm5next_layer/steps.rs"),
        include_str!("../../glm5next_layer/steps/drafter.rs"),
        include_str!("../../glm5next_layer/steps/ffn_half.rs"),
        include_str!("../../glm5next_layer/steps/forward.rs"),
        include_str!("../../glm5next_layer/steps/mixer.rs"),
        include_str!("../../glm5next_layer/steps/multi_seq.rs"),
        include_str!("../../glm5next_layer/steps/staged.rs"),
        include_str!("../../glm5next_layer/steps/staged/sp.rs"),
        include_str!("../../glm5next_layer/steps/verify_multi.rs"),
        include_str!("../../glm5next_layer/types.rs"),
    );
    assert_eq!(
        glm5next_layer_files(),
        [
            "comm_overlap.rs",
            "dense_fp8.rs",
            "dense_fp8_gw.rs",
            "dense_fp8_min_rows.rs",
            "levers.rs",
            "levers_comm.rs",
            "mod.rs",
            "prefetch.rs",
            "profile.rs",
            "scratch_union.rs",
            "scratch_union_tests.rs",
            "seq_parallel.rs",
            "seq_parallel/premix.rs",
            "seq_parallel/tests.rs",
            "state.rs",
            "steps.rs",
            "steps/drafter.rs",
            "steps/ffn_half.rs",
            "steps/forward.rs",
            "steps/mixer.rs",
            "steps/multi_seq.rs",
            "steps/staged.rs",
            "steps/staged/sp.rs",
            "steps/verify_multi.rs",
            "tests.rs",
            "types.rs",
        ],
        "a new glm5next_layer file must join the scanned `concat!` above"
    );
    assert_eq!(
        module.matches("self.forward_k(").count(),
        2,
        "a new forward_k caller must decide its own `is_prefill`, not inherit one"
    );
    assert!(
        src.contains(
            "// This IS the prefill sub-chunk caller.
                    true,"
        ),
        "the prefill sub-chunk must pass is_prefill = true"
    );
    assert!(
        src.contains("// A speculative verify, NOT a prefill sub-chunk"),
        "the speculative verify must pass is_prefill = false, and say why"
    );
    // 2026-09-29: `attn_half` takes the same `is_prefill`. Its two callers are `forward_k`,
    // which forwards its own argument, and the staged prefill's attention pass (`true`).
    assert_eq!(
        module.matches("self.attn_half(").count(),
        2,
        "a new attn_half caller must decide its own `is_prefill`, not inherit one"
    );
    let staged = include_str!("../../glm5next_layer/steps/staged.rs");
    assert!(
        staged.contains(
            "// This IS a prefill sub-chunk (staged attention pass).
                true,"
        ),
        "the staged attention pass must pass is_prefill = true"
    );
    // 2026-10-01: `attn_mixer`: `attn_half_inner` forwards its own; `staged/sp.rs` passes true.
    assert_eq!(
        module.matches("self.attn_mixer(").count(),
        2,
        "a new attn_mixer caller must decide its own `is_prefill`, not inherit one"
    );
}

/// 2026-09-25: `dsa_expand_selection` puts the tail at `row_select_k * KP`, with
/// `row_select_k` clamped to the row's own pool count. A batched pass plans `select_k` from
/// the group's final length, so without the clamp an earlier row's tail would sit later than
/// in a single-row pass. `glm5next_dsa_mla_decode_fp8` splits the row into `NUM_WARPS` (8)
/// slices and merges their softmaxes, so a moved tail token changes the summation order.
/// Measured 2026-09-06 on that kernel without the clamp: 14 of 18 configurations with a tail
/// token across a slice boundary differed, by up to 2 BF16 ulp.
#[test]
fn expand_selection_clamps_select_k_to_the_row() {
    let src = include_str!("../../../../../kernels/gb10/common/dsa_indexer.cu");
    assert!(
        src.contains("const unsigned int row_pools = (unsigned int)(q_pos[r] + 1) / KP;"),
        "the row's own pool count must come from its own q_pos"
    );
    assert!(
        src.contains("unsigned int base = row_select_k * KP;"),
        "the tail base must be the ROW's select_k, never the pass's"
    );
    assert!(
        !src.contains("unsigned int base = select_k * KP;"),
        "the pass-scalar tail base is the defect; it must not come back"
    );
}

/// 2026-09-26: Every `.rs` file under `glm5next_layer/`, relative and sorted.
fn glm5next_layer_files() -> Vec<String> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/glm5next_layer");
    let mut files = Vec::new();
    let mut pending = vec![root.clone()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let path = entry.expect("readable glm5next_layer entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|x| x == "rs") {
                let rel = path.strip_prefix(&root).expect("under the root");
                let parts: Vec<_> = rel.iter().map(|c| c.to_string_lossy()).collect();
                files.push(parts.join("/"));
            }
        }
    }
    files.sort();
    files
}

/// 2026-10-01: A `METRALE_GLM_DSA_GEMV_SPLIT` launch gives every block row at most
/// `DENSE_GEMV_BATCHM_MAX_M` rows and leaves none empty, so it passes
/// `dense_gemv_batchm_split`'s check and the kernel's `MAX_M` clamp never drops a row.
#[test]
fn dsa_tiled_gemv_split_block_rows_stay_within_max_m() {
    let max = metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize;
    for rows in 1..=8192usize {
        let y = row_batch::gemv_split_blocks(rows) as usize;
        let per = rows.div_ceil(y);
        assert!((1..=rows).contains(&y), "rows {rows}: y {y}");
        assert!(per <= max, "rows {rows}: {per} rows per block row");
        assert!((y - 1) * per < rows, "rows {rows}: an empty block row");
    }
    assert_eq!(row_batch::gemv_split_blocks(16), 1);
    assert_eq!(row_batch::gemv_split_blocks(17), 2);
    assert_eq!(row_batch::gemv_split_blocks(256), 16);
}

/// 2026-10-01: Both batched GEMV entry points carry the y-split prologue the split launch
/// relies on, and `batchm_rows` takes the split only behind the lever.
#[test]
fn dsa_tiled_gemv_split_kernels_carry_the_y_split() {
    let cu = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common/dense_gemv_bf16_batchm.cu");
    let src = std::fs::read_to_string(&cu).expect("dense_gemv_bf16_batchm.cu readable");
    for entry in ["dense_gemv_bf16_batchm(", "dense_gemv_bf16_fp32out_batchm("] {
        let head = format!("extern \"C\" __global__ void {entry}");
        assert!(src.contains(&head), "{cu:?} lacks {entry}");
    }
    let prologue = "const unsigned int r0 = blockIdx.y * rows_per_y;";
    assert_eq!(src.matches(prologue).count(), 2, "{cu:?}: y-split prologue per entry");
    let rb = include_str!("row_batch.rs");
    assert!(rb.contains("if rows > 0 && crate::glm5next_layer::levers::dsa_gemv_split() {"));
}
