// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Load-time warm-up: one throwaway prefill before the listener
//! binds, so the first real request does not pay the engine's first-use cost.
//!
//! Why: on the GLM-5.3 TP2 car (race-prefill #68, d4c 2026-10-08) the first
//! 32K request of a serve took 1.32-1.35 s longer (server TTFT) than the
//! second, mostly in its first 8192-token chunk, with +0.2-0.4 s in each
//! later chunk. The TTFT targets are cold first requests.
//!
//! `METRALE_SERVE_WARMUP_TOKENS=N` (unset or 0 = off) submits one blocking
//! request of N synthetic tokens through the scheduler channel the API uses,
//! with `max_tokens` 1, greedy, thinking off and MTP off, and waits for it.
//!
//! Owner: server (startup).
//! Invariants:
//! - Runs on the head rank only, after the scheduler starts and before the
//!   listener binds, so no HTTP request can race it and the readiness line
//!   comes after it.
//! - The synthetic prompt starts with an ordinary token, never the chat
//!   template's first token, so no real prompt shares a prefix with it: the
//!   blocks and SSM snapshot it leaves in the prefix cache are never matched
//!   and age out by LRU.
//! - A failed warm-up is logged and serving continues.

use std::sync::Arc;
use std::time::Instant;

use crate::api::InferenceRequest;
use crate::main_modules::AppState;

const LEVER: &str = "METRALE_SERVE_WARMUP_TOKENS";

/// 2026-10-08: The token count `METRALE_SERVE_WARMUP_TOKENS` asks for, or
/// `None` when the warm-up is off (unset, empty, 0 or not a number; a bad value
/// is warned once).
pub(crate) fn warmup_tokens() -> Option<usize> {
    let raw = std::env::var(LEVER).ok()?;
    match raw.trim().parse::<usize>() {
        Ok(0) => None,
        Ok(n) => Some(n),
        Err(_) => {
            tracing::warn!("{LEVER}={raw:?} is not a token count; the load-time warm-up is off");
            None
        }
    }
}

/// 2026-10-08: `n` ordinary token ids from a fixed LCG, all in 1000..100000
/// (inside the GLM-5.3 text vocabulary, below every special token), so the
/// prompt is the same on every serve and starts with no template token.
pub(crate) fn synthetic_prompt(n: usize) -> Vec<u32> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            1000 + ((x >> 33) % 99_000) as u32
        })
        .collect()
}

/// 2026-10-08: Run the warm-up when the lever asks for one. Holds `state` only
/// for the call, so a later model swap is not blocked by a live sender.
pub(crate) async fn run_if_enabled(state: &Arc<AppState>) {
    let Some(n) = warmup_tokens() else { return };
    let tokens = synthetic_prompt(n);
    let (response_tx, rx) = tokio::sync::oneshot::channel();
    let req = InferenceRequest::Blocking {
        session_hash: crate::session_manager::compute_session_hash(&tokens),
        prompt_tokens: Arc::new(tokens),
        adapter_slot: -1,
        src_lang_id: 0,
        tgt_lang_id: 0,
        num_beams: 1,
        length_penalty: 1.0,
        early_stopping: false,
        image_pixels: vec![],
        max_tokens: 1,
        min_tokens: 0,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        lz_penalty: 0.0,
        logit_bias: vec![],
        stop_tokens: vec![],
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        require_tool_call: false,
        tools_present: false,
        suppress_tool_call: false,
        disable_mtp: true,
        grammar_spec: None,
        seed: Some(42),
        top_logprobs: None,
        prompt_logprobs: None,
        echo: false,
        timeout_at: None,
        response_tx,
    };
    tracing::warn!("{LEVER}={n}: ENGAGED - one {n}-token throwaway prefill before the listener binds");
    let t0 = Instant::now();
    if state.request_tx.send(req).await.is_err() {
        tracing::warn!("{LEVER}: the scheduler channel is closed; warm-up skipped");
        return;
    }
    match rx.await {
        Ok(Ok(r)) => tracing::warn!(
            "{LEVER}: warm-up done in {:.0} ms ({n} prompt tokens, {} cached, finish {})",
            t0.elapsed().as_secs_f64() * 1e3,
            r.cached_prompt_tokens,
            r.finish_reason
        ),
        Ok(Err(e)) => tracing::warn!("{LEVER}: warm-up failed ({e:#}); serving anyway"),
        Err(_) => tracing::warn!("{LEVER}: warm-up reply dropped; serving anyway"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_prompt_is_fixed_and_in_range() {
        let a = synthetic_prompt(4096);
        assert_eq!(a, synthetic_prompt(4096));
        assert!(a.iter().all(|&t| (1000..100_000).contains(&t)));
        assert_eq!(&synthetic_prompt(8)[..], &a[..8]);
    }
}
