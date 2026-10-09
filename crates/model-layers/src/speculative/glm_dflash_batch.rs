// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: GLM-5.3 DFlash at C >= 2: the batched-verify lever
//! (`METRALE_GLM_DFLASH_BATCHED_VERIFY`), its per-sequence control arm
//! (`METRALE_GLM_DFLASH_PERSEQ_K4`), the C >= 2 verify width they pin, and the pure predicates
//! the model engine's multi-rank batched verify (`verify_ep.rs`) is admitted by. Host-side and
//! pure apart from the two lever reads, so it is CPU-tested (`glm_dflash_batch_tests.rs`).
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - Both levers are read once per process; each logs `ENGAGED` once when on.
//! - [`glm_dflash_multi_k`] pins only the C >= 2 width; the C = 1 width and the serve's gamma
//!   (`--dflash-gamma`, pool sizing, capture bands) are untouched.
//! - Under a communicator a DFlash batch is admitted only at widths the multi-rank batch meta
//!   carries ([`EP_BATCH_VERIFY_MAX_K`]), so a batch rank 0 admits is never refused at send.

/// 2026-10-09: The C >= 2 verify width (rows per sequence: 3 drafts + 1) the GLM DFlash levers
/// pin in v1. A later lever may pick K per step in 2..=4; nothing downstream assumes 4.
pub const GLM_DFLASH_MULTI_K: usize = 4;

/// 2026-10-09: Widest per-sequence k the multi-rank batched verify carries: the bound
/// `decode_verify_batch_meta` (model-engine `verify_ep.rs`) enforces on both ranks.
pub const EP_BATCH_VERIFY_MAX_K: usize = 4;

/// 2026-10-09: `METRALE_GLM_DFLASH_BATCHED_VERIFY=1`: on a GLM-5.3 DFlash serve, C >= 2
/// verifies all sequences in one batched forward (`step_verify_dflash_batched`) at
/// [`GLM_DFLASH_MULTI_K`], also on two ranks (EP protocol v2 with
/// `METRALE_GLM_BATCHED_VERIFY=1`). Read once; logs `ENGAGED` once when on.
pub fn glm_dflash_batched_verify() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = super::glm_dflash::parse_switch(
            std::env::var("METRALE_GLM_DFLASH_BATCHED_VERIFY")
                .ok()
                .as_deref(),
        );
        if on {
            tracing::info!(
                "METRALE_GLM_DFLASH_BATCHED_VERIFY: ENGAGED (GLM DFlash C>=2: one batched verify \
                 at K={GLM_DFLASH_MULTI_K}; C=1 unchanged)"
            );
        }
        on
    })
}

/// 2026-10-09: `METRALE_GLM_DFLASH_PERSEQ_K4=1` (test-only control arm): the same C >= 2 width
/// as the batched lever, verified one sequence at a time, for the batched vs per-sequence A/B.
/// Read once; logs `ENGAGED` once when on.
pub fn glm_dflash_perseq_k4() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = super::glm_dflash::parse_switch(
            std::env::var("METRALE_GLM_DFLASH_PERSEQ_K4")
                .ok()
                .as_deref(),
        );
        if on {
            tracing::info!(
                "METRALE_GLM_DFLASH_PERSEQ_K4: ENGAGED (GLM DFlash C>=2: per-sequence verify at \
                 K={GLM_DFLASH_MULTI_K}; control arm)"
            );
        }
        on
    })
}

/// 2026-10-09: The C >= 2 DFlash verify width to pin for `model_type`, or `None`: Some only for
/// a GLM-5.3 target with either lever on.
pub fn glm_dflash_multi_k(model_type: &str, batched: bool, perseq_k4: bool) -> Option<usize> {
    let glm = matches!(model_type, "glm5_next" | "glm5_next_text");
    (glm && (batched || perseq_k4)).then_some(glm_dflash_multi_k_env())
}

/// 2026-10-09: `METRALE_GLM_DFLASH_MULTI_K=<2..=4>` (test lever, default [`GLM_DFLASH_MULTI_K`]):
/// the C >= 2 width under either lever, for measuring the batched verify cost per width. A value
/// outside 2..=4 keeps the default and warns once. Read once per process.
pub fn glm_dflash_multi_k_env() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| match std::env::var("METRALE_GLM_DFLASH_MULTI_K") {
        Err(_) => GLM_DFLASH_MULTI_K,
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(k) if (2..=4).contains(&k) => {
                tracing::warn!("METRALE_GLM_DFLASH_MULTI_K={k}: ENGAGED - GLM DFlash C>=2 width K={k}");
                k
            }
            _ => {
                tracing::warn!(
                    "METRALE_GLM_DFLASH_MULTI_K={v:?} outside 2..=4; keeping K={GLM_DFLASH_MULTI_K}"
                );
                GLM_DFLASH_MULTI_K
            }
        },
    })
}

/// 2026-10-09: Whether the batched verify may run with a comm backend: the multi-rank protocol
/// live in its v2 shape, every layer driving its own verify state (`layers_own_states`: non-empty
/// and all `decode_verify_multi_own_states`), and no DFlash drafter unless the batched lever is
/// on. The DFlash capture is rank 0's alone (`try_dflash_capture_all_at` no-ops on a worker) and
/// only rank 0's `commit_ctx` reads it, so nothing of it is mirrored.
pub fn batched_verify_ep_admits(
    protocol_active: bool,
    protocol_v2: bool,
    dflash: bool,
    dflash_batched: bool,
    layers_own_states: bool,
) -> bool {
    protocol_active && protocol_v2 && (!dflash || dflash_batched) && layers_own_states
}

/// 2026-10-09: DFlash batch shape: one k for the whole batch, `2 <= k <= kgamma`, and at most
/// [`EP_BATCH_VERIFY_MAX_K`] with a comm backend (`multi_rank`); `kgamma >= 2`; a full capture
/// band of `kgamma` rows per sequence within `save_rows`.
pub fn dflash_batch_shape_ok(
    ks: &[usize],
    kgamma: usize,
    save_rows: usize,
    multi_rank: bool,
) -> bool {
    let k_hi = if multi_rank {
        kgamma.min(EP_BATCH_VERIFY_MAX_K)
    } else {
        kgamma
    };
    kgamma >= 2
        && ks.iter().all(|&k| (2..=k_hi).contains(&k))
        && ks.iter().all(|&k| k == ks[0])
        && ks.len() * kgamma <= save_rows
}

/// 2026-10-09: Whether a worker may skip the batched verify's hidden stash: under the batched
/// lever with a proposer that runs on rank 0 alone (`needs_comm == Some(false)`, the DFlash
/// drafter), no worker propose ever reads the stash. An EP-sharded proposer (GLM MTP) or no
/// proposer keeps the stash and its error path.
pub fn worker_skips_verify_stash(dflash_batched: bool, proposer_needs_comm: Option<bool>) -> bool {
    dflash_batched && proposer_needs_comm == Some(false)
}

/// 2026-10-09: The verdict words rank 0 broadcasts after a batched verify: sequence i's accepted
/// draft count, clamped to `ks[i] - 1` (the bound `verify_batch_commit_plan` enforces).
pub fn batch_verdict_words(accepted: &[usize], ks: &[usize]) -> Vec<u32> {
    accepted
        .iter()
        .zip(ks)
        .map(|(&na, &k)| na.min(k.saturating_sub(1)) as u32)
        .collect()
}

#[cfg(test)]
#[path = "glm_dflash_batch_tests.rs"]
mod tests;
