// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3-Flash layer launch levers: the prefill sub-chunk width, cuBLASLt for
//! wide projections, and the batched DSA indexer query.
//! 2026-10-01: Plus the batched multi-sequence decode (`decode_multi_seq`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Each function reads its environment variable once per process and caches the result.

// 2026-10-01: The two-rank communication levers live in `levers_comm.rs` (500-line cap).
#[path = "levers_comm.rs"]
mod comm;
pub use comm::{dsa_index_split, dsa_index_split_wide, prefill_comm_overlap, prefill_seq_parallel};

/// 2026-09-25: Default tokens per batched prefill sub-chunk, the width `Glm5NextLayer::prefill`
/// hands `Glm5NextLayer::forward_k` (overridable through [`prefill_rows`]).
///
/// 16 is `DENSE_GEMV_BATCHM_MAX_M`, the widest M that `ops::dense_mm_bf16` sends to the batched
/// GEMV (`dense_gemv_bf16_batchm`), whose rows carry the same bits as the M = 1 GEMV. A wider
/// sub-chunk moves the dense projections to cuBLASLt or the tile GEMM, which accumulate in a
/// different order. Measured 2026-09-02: the 12 GLM prefill projection shapes cost 1.36-1.98x less
/// per token at M = 16 than at M = 8 in 11 of 12 (N4096 K128: 0.77x).
///
/// The routed MoE does not follow the width: `glm5next_mlp::forward::forward_moe` splits a wider
/// row group into even sub-groups of at most `row_batch_max()`, whose default and ceiling is
/// `MOE_ROW_BATCH_MAX_ROWS` (8).
pub(crate) const PREFILL_ROWS: usize = 16;

/// 2026-09-25: Send GLM projections with M above `DENSE_GEMV_BATCHM_MAX_M` to cuBLASLt BF16
/// instead of the tile GEMM. Read by the KDA, DSA and MLP blocks, always behind a
/// `> DENSE_GEMV_BATCHM_MAX_M` row test, so the M = 1 GEMV and the batched GEMV are never
/// replaced. On unless `METRALE_GLM_CUBLAS_PROJ=0`.
pub(crate) fn cublas_wide_proj() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_CUBLAS_PROJ").as_deref() != Ok("0");
        tracing::warn!(
            "METRALE_GLM_CUBLAS_PROJ: wide GLM projections (M > {}) use {}",
            metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M,
            if on {
                "cuBLASLt BF16"
            } else {
                "the scalar tile GEMM"
            }
        );
        on
    })
}

/// 2026-09-25: Compute the DSA indexer `wq_b` projection for all prefill rows in one cuBLASLt
/// GEMM instead of one M = 1 GEMV per row. Off unless `METRALE_GLM_DSA_BATCH_QIDX=1`.
///
/// Prefill only: it is read in `select_rows_batched`, which runs only when
/// `batch_select_enabled` holds (`is_prefill && k > 1`). Decode and the speculative verify keep
/// the per-row GEMV. The GEMM accumulates in a different order from the GEMV.
pub(crate) fn dsa_batch_qidx() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_DSA_BATCH_QIDX").as_deref() == Ok("1");
        tracing::warn!(
            "METRALE_GLM_DSA_BATCH_QIDX: prefill DSA indexer q uses {}",
            if on {
                "ONE batched cuBLASLt GEMM"
            } else {
                "one M=1 GEMV per row"
            }
        );
        on
    })
}

/// 2026-10-01: `METRALE_GLM_DSA_ROW_BATCH=1` runs the per-row part of a prefill sub-chunk's DSA
/// block (`Glm5NextDsaLayer::decode_k`) for all rows at once: one metadata upload from pinned
/// staging, one latent-write launch, the indexer projections through the batched GEMVs and
/// one `k_norm` launch. Bit-identical to the per-row walk by construction
/// (`glm5next_dsa/layer/row_batch.rs`, `docs/dsa-rowbatch-NOTES.md`). Off unless set to `1`;
/// read once. Prefill only: it applies only where `batch_select_enabled` holds.
pub(crate) fn dsa_row_batch() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_DSA_ROW_BATCH").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_DSA_ROW_BATCH=1 - prefill DSA latent/indexer writes run once per \
                 sub-chunk instead of once per row"
            );
        }
        on
    })
}

/// 2026-10-01: `METRALE_GLM_DSA_SCORES_TILED` and `METRALE_GLM_DSA_GEMV_SPLIT` as a switch: `1`
/// (surrounding blanks ignored) is on; unset, `0` and anything else are off.
pub(crate) fn parse_dsa_switch(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-01: Warn when a DSA switch holds something other than unset, empty, `0` or `1`.
fn warn_unparsed_dsa_switch(name: &str, raw: Option<&str>) {
    if let Some(r) = raw.filter(|r| !r.is_empty() && r.trim() != "0" && r.trim() != "1") {
        tracing::warn!("{name}={r} is not 0 or 1 - treated as off");
    }
}

/// 2026-10-01: `METRALE_GLM_DSA_SCORES_TILED=1` scores an exact (host-geometry) DSA selection
/// with `dsa_index_scores_tiled`, one block per 32-row x 64-pool tile, instead of
/// `dsa_index_scores`, one block per (pool, row); byte-identical by construction (argument in
/// `kernels/gb10/common/dsa_indexer.cu`, GPU gate
/// `examples/dsa_indexer_tiled_bitparity_microtest.rs`). The ceiling (graph-replay decode)
/// launch, fewer than `SCORES_TILED_MIN_POOLS` pools (PROVISIONAL), an unresolved entry
/// point or a shape outside the tiled envelope keeps `dsa_index_scores`
/// (`glm5next_dsa::select::scores_tiled_for`, logged once). Off unless set
/// to `1`; read once.
pub(crate) fn dsa_scores_tiled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_SCORES_TILED").ok();
        let on = parse_dsa_switch(raw.as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_DSA_SCORES_TILED=1 - exact DSA selections score pools with \
                 dsa_index_scores_tiled (32 rows x 64 pools per block; byte-identical by \
                 construction)"
            );
        }
        warn_unparsed_dsa_switch("METRALE_GLM_DSA_SCORES_TILED", raw.as_deref());
        on
    })
}

/// 2026-10-01: `METRALE_GLM_DSA_SCORES_TC` as a precision mode for `dsa_index_scores_tc`
/// (surrounding blanks and case ignored): `split3` or `1` is 3 (q and keys split into BF16
/// hi + lo, FP32-class dots), `split2` is 2 (q split only), `bf16` is 1 (one BF16 MMA);
/// unset, empty, `0`, `off` and anything else are 0 (off).
pub(crate) fn parse_dsa_scores_tc(v: Option<&str>) -> u32 {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("1") | Some("split3") => 3,
        Some("split2") => 2,
        Some("bf16") => 1,
        _ => 0,
    }
}

/// 2026-10-01: `METRALE_GLM_DSA_SCORES_TC=<mode>` scores an exact (host-geometry) DSA
/// selection of at least `SCORES_TC_MIN_POOLS` pools (PROVISIONAL) with `dsa_index_scores_tc`
/// on tensor cores (mma.sync m16n8k16 BF16, FP32 accumulate) instead of the FP32 scorers, for
/// long prompts where the O(rows x pools) indexer dominates prefill. NOT byte-identical: the
/// scores carry about 2^-16 (`split3`) or 2^-9 (`split2`, `bf16`) relative error, so near-tie
/// pools at the top-k boundary can swap. GPU gate: `examples/dsa_indexer_tc_microtest.rs`.
/// Takes precedence over `METRALE_GLM_DSA_SCORES_TILED`. Off (0) unless set; read once.
pub(crate) fn dsa_scores_tc() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_SCORES_TC").ok();
        let mode = parse_dsa_scores_tc(raw.as_deref());
        let t = raw.as_deref().map(str::trim).unwrap_or_default();
        if mode != 0 {
            tracing::warn!(
                "METRALE_GLM_DSA_SCORES_TC={t} - exact DSA selections score pools on tensor \
                 cores, mode {mode} (3 split3, 2 split2, 1 bf16; NOT byte-identical)"
            );
        } else if !t.is_empty() && t != "0" && !t.eq_ignore_ascii_case("off") {
            tracing::warn!(
                "METRALE_GLM_DSA_SCORES_TC={t} is not split3, split2, bf16, 1, 0 or off - \
                 treated as off"
            );
        }
        mode
    })
}

/// 2026-10-01: `METRALE_GLM_DSA_GEMV_SPLIT=1` runs each batched indexer GEMV of the DSA row
/// batch (`glm5next_dsa/layer/row_batch.rs`: `wk`, `compress_gate`, `weights_proj`, `wq_b`)
/// as ONE launch over all `k` rows, `ceil(k / 16)` block rows of at most 16 rows each
/// (`dense_gemv_batchm_split`), instead of one launch per 16 rows. Every row gets the same
/// arithmetic in any split (`dense_gemv_bf16_batchm.cu` header), so the bytes do not change.
/// Only reachable under `METRALE_GLM_DSA_ROW_BATCH=1`. Off unless set to `1`; read once.
pub(crate) fn dsa_gemv_split() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_GEMV_SPLIT").ok();
        let on = parse_dsa_switch(raw.as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_DSA_GEMV_SPLIT=1 - DSA row-batch indexer GEMVs run one y-split \
                 launch per projection instead of one per 16 rows (byte-identical by \
                 construction)"
            );
        }
        warn_unparsed_dsa_switch("METRALE_GLM_DSA_GEMV_SPLIT", raw.as_deref());
        on
    })
}

/// 2026-10-01: `METRALE_GLM_DECODE_L2_PREFETCH=1`: a decode or verify step of at most
/// `prefetch::L2_PREFETCH_MAX_ROWS` rows enqueues one `glm5next_l2_prefetch` launch before each
/// all-reduce, pulling the weights read right after the latency-bound chain into L2
/// (`glm5next_layer/prefetch.rs`). Byte-identical by construction: the kernel writes nothing.
/// Off unless set to `1`; read once.
pub(crate) fn decode_l2_prefetch() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DECODE_L2_PREFETCH").ok();
        let on = parse_dsa_switch(raw.as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_DECODE_L2_PREFETCH=1 - decode/verify steps prefetch the next \
                 weights into L2 before each all-reduce, up to {} MiB per launch \
                 (byte-identical by construction)",
                decode_l2_prefetch_bytes() >> 20
            );
        }
        warn_unparsed_dsa_switch("METRALE_GLM_DECODE_L2_PREFETCH", raw.as_deref());
        on
    })
}

/// 2026-10-01: `METRALE_GLM_DECODE_MULTI_SEQ=1` lets a multi-sequence decode step run as ONE
/// batched forward (`Glm5NextLayer::decode_multi_seq`: mHC, norms, MLP/MoE and the KDA
/// projections over all N rows; the KDA recurrence and the DSA attention per sequence) instead
/// of one full forward per sequence. It flips `decode_multi_seq_unsupported` to false, which is
/// the only routing input; the batched override itself runs whenever the dispatcher selects it.
/// Eager and unpadded (`decode_multi_seq_eager_only`); the batched MTP verify has its own lever,
/// [`batched_verify`]. Off unless set to `1`; read once. Port of rsafier's Atlas
/// research/glm-exl3 multi-sequence decode (e69446eee, acf792e28, 04beaac1f).
pub fn decode_multi_seq() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_DECODE_MULTI_SEQ").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_DECODE_MULTI_SEQ=1 - GLM multi-sequence decode runs one batched \
                 forward per step (eager, unpadded); KDA recurrence and DSA attention stay per \
                 sequence"
            );
        }
        on
    })
}

/// 2026-10-02: `METRALE_GLM_BATCHED_VERIFY=1` lets the MTP verify of several sequences run as
/// ONE forward over all `R = Σ ks` rows (`Glm5NextLayer::verify_n_seqs`: mHC, norms, MLP/MoE
/// and the KDA projections over all rows; the KDA recurrence with its snapshots and the DSA
/// attention per sequence) instead of one verify forward per sequence. It flips
/// `decode_verify_multi_unsupported` to false, which admits the batch on one GPU and, with
/// `METRALE_EP_PROTOCOL=v2`, on two ranks (model-engine `verify_ep.rs`). Eager. Off unless set
/// to `1`; read once; off, every verify takes the per-sequence path it took before.
pub fn batched_verify() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_BATCHED_VERIFY").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_BATCHED_VERIFY=1 - GLM MTP verify of several sequences runs one \
                 batched forward per step (eager); KDA recurrence and DSA attention stay per \
                 sequence"
            );
        }
        on
    })
}

/// 2026-10-02: Rows the KDA and MLP workspaces hold under `batched_verify`: the widest batch the
/// default draft ladder builds (8 x 4, 16 x 3 or 32 x 2 rows). 0, no change, when it is off.
pub fn batched_verify_rows() -> usize {
    if batched_verify() { 64 } else { 0 }
}

/// 2026-10-01: Default MiB per `glm5next_l2_prefetch` launch: 12, half the 24 MB GB10 L2.
/// PROVISIONAL: sized from a ~50 us window at ~240 GB/s, not measured yet
/// (`examples/glm5next_l2_prefetch_microtest.rs` sweeps it).
pub(crate) const DECODE_L2_PREFETCH_MIB: usize = 12;

/// 2026-10-01: The MiB value of `METRALE_GLM_DECODE_L2_PREFETCH_MIB`: an integer in 1..=64
/// (surrounding blanks ignored); `None` for unset or anything else.
pub(crate) fn parse_l2_prefetch_mib(v: Option<&str>) -> Option<usize> {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|m| (1..=64).contains(m))
}

/// 2026-10-01: Bytes per prefetch launch: `METRALE_GLM_DECODE_L2_PREFETCH_MIB` (1..=64) MiB,
/// else [`DECODE_L2_PREFETCH_MIB`]. Read once, only when the prefetch lever is on.
pub(crate) fn decode_l2_prefetch_bytes() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DECODE_L2_PREFETCH_MIB").ok();
        let parsed = parse_l2_prefetch_mib(raw.as_deref());
        if let (Some(r), None) = (raw.as_deref(), parsed)
            && !r.trim().is_empty()
        {
            tracing::warn!(
                "METRALE_GLM_DECODE_L2_PREFETCH_MIB={r} is not an integer in 1..=64 - using \
                 {DECODE_L2_PREFETCH_MIB}"
            );
        }
        parsed.unwrap_or(DECODE_L2_PREFETCH_MIB) << 20
    })
}

/// 2026-09-25: `PREFILL_ROWS`, overridable at launch with `METRALE_GLM_PREFILL_ROWS` (values
/// below 1 or unparsable are ignored). `1` selects the per-token walk: `Glm5NextLayer::prefill`
/// takes the batched sub-chunk path only when `rows > 1`. A width above
/// `DENSE_GEMV_BATCHM_MAX_M` changes the projection numerics as well as the speed.
pub fn prefill_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        let r = std::env::var("METRALE_GLM_PREFILL_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|r| *r >= 1)
            .unwrap_or(PREFILL_ROWS);
        if r != PREFILL_ROWS {
            tracing::warn!("GLM prefill sub-chunk overridden to {r} rows (default {PREFILL_ROWS})");
        }
        r
    })
}

/// 2026-09-29: Ceiling of the staged prefill's FFN window (`prefill_rows_ffn`), in rows.
/// 2026-10-01: Raised from 4096 to 8192 (one window per 8192-token server prefill chunk); a
/// request at or below 4096 resolves as before. The MLP scratch and the mHC `mix` scratch are
/// sized from `prefill_rows_ffn()`, so only a request above 4096 allocates more.
pub const PREFILL_ROWS_FFN_MAX: usize = 8192;

/// 2026-09-29: `METRALE_GLM_PREFILL_STAGED=1` runs a prefill chunk as two passes per layer: the
/// attention half over every `prefill_rows()` sub-chunk, then the FFN half over windows of up to
/// `prefill_rows_ffn()` rows (`Glm5NextLayer::prefill_staged_run`). Off unless set to `1`; read
/// once. Eager prefill only: a call under graph capture takes the unstaged loop.
pub fn prefill_staged() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_PREFILL_STAGED").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_PREFILL_STAGED=1 - GLM prefill runs attention at {} rows, then the \
                 FFN in windows of up to {} rows (byte-identical to the unstaged loop by \
                 construction; see steps/staged.rs)",
                prefill_rows(),
                prefill_rows_ffn()
            );
        }
        on
    })
}

/// 2026-09-29: The FFN window for attention width `attn` and a requested width: the request
/// (default `attn`) capped at `PREFILL_ROWS_FFN_MAX`, rounded down to a multiple of `attn`, and
/// never below `attn`.
pub(crate) fn resolve_rows_ffn(attn: usize, requested: Option<usize>) -> usize {
    let attn = attn.max(1);
    let want = requested
        .filter(|r| *r >= 1)
        .unwrap_or(attn)
        .min(PREFILL_ROWS_FFN_MAX);
    ((want / attn) * attn).max(attn)
}

/// 2026-09-29: The staged prefill's FFN window, from `METRALE_GLM_PREFILL_ROWS_FFN` through
/// `resolve_rows_ffn`; `prefill_rows()` when `prefill_staged()` is off, so every sizing that
/// reads it is unchanged then. Read once.
pub fn prefill_rows_ffn() -> usize {
    static R: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let attn = prefill_rows();
        let staged = std::env::var("METRALE_GLM_PREFILL_STAGED").as_deref() == Ok("1");
        if !staged {
            return attn;
        }
        let req = std::env::var("METRALE_GLM_PREFILL_ROWS_FFN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let r = resolve_rows_ffn(attn, req);
        if req.is_some_and(|q| q != r) {
            tracing::warn!(
                "METRALE_GLM_PREFILL_ROWS_FFN={} resolved to {r} (a multiple of the {attn}-row \
                 attention width, at most {PREFILL_ROWS_FFN_MAX})",
                req.unwrap_or_default()
            );
        }
        r
    })
}

/// 2026-10-01: `METRALE_GLM_PREFILL_TAIL_MERGE=1` (with `METRALE_GLM_PREFILL_STAGED=1`): the
/// staged FFN pass lets a mergeable tail sub-chunk (narrower than `prefill_rows()`) join the
/// window before it instead of taking a window, a routed grouped GEMM and an all-reduce of its
/// own (`ffn_windows`). Byte-identical by the same argument as the staged pass. Off unless set
/// to `1`; read once; inert when staging is off.
pub fn prefill_tail_merge() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_PREFILL_TAIL_MERGE").as_deref() == Ok("1")
            && std::env::var("METRALE_GLM_PREFILL_STAGED").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_PREFILL_TAIL_MERGE=1 - a mergeable prefill tail joins the last                  staged FFN window (byte-identical by construction; see steps/staged.rs)"
            );
        }
        on
    })
}

/// 2026-09-25: `METRALE_GLM_KDA_CHUNK_PREFILL=1` sends a prefill sub-chunk's KDA mixer through
/// the chunked scan (`Glm5NextKdaLayer::prefill`) instead of `decode_k`'s per-token recurrence.
/// Off unless set to `1`; read once. The chunked scan computes the recurrence chunk by chunk, in
/// a different order, so its output is not bit-identical to the per-token walk.
///
/// 2026-10-01: Moved here from `steps/forward.rs`; the loader also reads it, to size the KDA
/// workspace's chunked-scan buffers under `METRALE_GLM_PREFILL_FULLWIDTH_GEMM`.
pub(crate) fn kda_chunk_prefill() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_KDA_CHUNK_PREFILL").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_KDA_CHUNK_PREFILL=1 - GLM prefill KDA uses the CHUNKED scan \
                 (kda_chunk_prepare + kda_chunk_scan). Not bit-identical to the per-token \
                 recurrent walk."
            );
        }
        on
    })
}

/// 2026-10-01: Whether a `METRALE_GLM_PREFILL_FULLWIDTH_GEMM` value turns the lever on: `1`
/// (surrounding blanks ignored) only.
pub(crate) fn parse_fullwidth_switch(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-01: `METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1` (with `METRALE_GLM_PREFILL_STAGED=1`):
/// the staged prefill runs its dense projections as one GEMM per window of
/// `prefill_rows_ffn()` rows instead of one per `prefill_rows()` sub-chunk
/// (`Glm5NextLayer::prefill_staged_run`):
/// - the attention pass covers a whole window per call, so the mHC, the norm, every KDA
///   projection and the DSA q/kv/o and indexer (`wq_b`, `wk`, `compress_gate`,
///   `weights_proj`) projections see M = the window; the DSA selection and gather-attend
///   still run per `prefill_rows()` sub-chunk (`Glm5NextDsaLayer::decode_k_wide`);
/// - the FFN pass hands the MLP `dense_slice = window`, so the router, shared-expert and
///   dense-MLP GEMMs run once per window.
///
/// NOT byte-identical: cuBLASLt picks its algorithm per M and the indexer projections leave
/// the per-row GEMV for a tensor-core GEMM, so the sums are taken in another order. Quality is
/// gated at model level. Off unless set to `1`; read once; inert when staging is off. The
/// loader sizes the KDA workspace and the shared DSA wide arena from the window only when
/// this is on.
pub fn prefill_fullwidth_gemm() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_PREFILL_FULLWIDTH_GEMM").ok();
        let asked = parse_fullwidth_switch(raw.as_deref());
        let on = asked && std::env::var("METRALE_GLM_PREFILL_STAGED").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1 - staged GLM prefill projections run as \
                 one GEMM per {}-row window (DSA selection and attend still per {} rows); NOT \
                 byte-identical to the sliced path",
                prefill_rows_ffn(),
                prefill_rows()
            );
        } else if asked {
            tracing::warn!(
                "METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1 ignored: it needs \
                 METRALE_GLM_PREFILL_STAGED=1"
            );
        }
        warn_unparsed_dsa_switch("METRALE_GLM_PREFILL_FULLWIDTH_GEMM", raw.as_deref());
        on
    })
}

/// 2026-10-01: The window the full-width GEMMs cover, `prefill_rows_ffn()`, when
/// [`prefill_fullwidth_gemm`] is on; `None` otherwise. The loader sizes the KDA workspace and
/// the DSA wide arena from it.
pub fn fullwidth_rows() -> Option<usize> {
    prefill_fullwidth_gemm().then(prefill_rows_ffn)
}

/// 2026-09-29: The env-read inputs that decide which sub-chunks the staged FFN pass merges
/// (`grouped_prefill_selected`): the grouped-GEMM minimum rows and its switch, forced host
/// dispatch and route tracing, packed into one value for the rank-agreement check. A skew
/// would give the ranks different window counts, and each window issues its own all-reduce.
pub fn staged_merge_signature() -> u64 {
    use crate::glm5next_mlp::forward_prefill_gemm::{prefill_gemm_enabled, prefill_gemm_min_rows};
    ((prefill_gemm_min_rows() as u64) << 3)
        | (u64::from(prefill_gemm_enabled()) << 2)
        | (u64::from(crate::glm5next_mlp::forward::host_dispatch_forced()) << 1)
        | u64::from(super::profile::trace_on())
}
