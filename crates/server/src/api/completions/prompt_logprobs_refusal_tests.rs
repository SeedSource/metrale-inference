// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17 (A191): Tests that `/v1/completions` refuses prompt logprobs on a multi-rank serve.
//!
//! Owner: server completions API.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn multi_rank_echo_with_logprobs_is_refused() {
    let msg = multi_rank_prompt_logprobs_refusal(2, true, true).expect("refused at world_size 2");
    assert!(
        msg.contains("world_size=2") && msg.contains("rank 0 only"),
        "{msg}"
    );
}

#[test]
fn single_rank_or_no_prompt_logprobs_is_allowed() {
    assert_eq!(multi_rank_prompt_logprobs_refusal(1, true, true), None);
    assert_eq!(
        multi_rank_prompt_logprobs_refusal(2, true, false),
        None,
        "echo alone: no prompt logprobs"
    );
    assert_eq!(
        multi_rank_prompt_logprobs_refusal(2, false, true),
        None,
        "logprobs alone: generated tokens only"
    );
}

#[test]
fn request_json_reaches_the_refusal_inputs() {
    let req: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m", "prompt": "hi", "echo": true, "logprobs": 20, "max_tokens": 1,
    }))
    .unwrap();
    assert!(multi_rank_prompt_logprobs_refusal(2, req.echo, req.logprobs.is_some()).is_some());
}
