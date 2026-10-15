// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Request fixtures for the scheduler tests.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: A minimal blocking request whose response receiver is dropped, for tests
/// that never read the reply.
pub(in crate::scheduler) fn blocking_request(
    grammar_spec: Option<crate::api::GrammarSpec>,
) -> crate::api::InferenceRequest {
    let (response_tx, _rx) = tokio::sync::oneshot::channel();
    crate::api::InferenceRequest::Blocking {
        prompt_tokens: std::sync::Arc::new(vec![0]),
        session_hash: 0,
        cache_salt_hash: 0,
        adapter_slot: -1,
        src_lang_id: 0,
        tgt_lang_id: 0,
        num_beams: 1,
        length_penalty: 1.0,
        early_stopping: false,
        image_pixels: vec![],
        max_tokens: 2,
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
        tools_present: grammar_spec.is_some(),
        suppress_tool_call: false,
        disable_mtp: true,
        grammar_spec,
        seed: Some(42),
        top_logprobs: None,
        prompt_logprobs: None,
        echo: false,
        timeout_at: None,
        response_tx,
    }
}
