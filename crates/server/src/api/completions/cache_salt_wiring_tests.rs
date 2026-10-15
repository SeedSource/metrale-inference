// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests that a request's `cache_salt` reaches the scheduler (A188):
//! `/v1/completions` carries its hash on the `InferenceRequest`, and chat lowers
//! it into the IR that the chat handlers hash.
//!
//! Owner: server completions API.
//! Invariants: none beyond the types.

use super::*;
use metrale_cache::radix_tree::cache_salt_hash;

fn streaming(body: serde_json::Value) -> InferenceRequest {
    let req: CompletionRequest = serde_json::from_value(body).unwrap();
    let (token_tx, _token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(16);
    build_streaming_request(
        &req,
        crate::api::completions_exec::CompletionParams::test(),
        std::sync::Arc::new(vec![1, 2, 3]),
        42,
        None,
        token_tx,
    )
}

#[test]
fn completions_carry_cache_salt_hash() {
    let a = streaming(serde_json::json!({"model": "m", "prompt": "hi", "cache_salt": "tenant-a"}));
    let b = streaming(serde_json::json!({"model": "m", "prompt": "hi", "cache_salt": "tenant-b"}));
    let none = streaming(serde_json::json!({"model": "m", "prompt": "hi"}));
    assert_eq!(a.cache_salt_hash(), cache_salt_hash(Some("tenant-a")));
    assert_ne!(a.cache_salt_hash(), b.cache_salt_hash());
    assert_ne!(a.cache_salt_hash(), 0);
    assert_eq!(none.cache_salt_hash(), 0);
}

#[test]
fn chat_lowers_cache_salt_into_ir() {
    let body = |salt: Option<&str>| {
        let mut v =
            serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        if let Some(s) = salt {
            v["cache_salt"] = serde_json::json!(s);
        }
        let req: crate::openai::ChatCompletionRequest = serde_json::from_value(v).unwrap();
        crate::ir::ChatRequest::from(req)
    };
    assert_eq!(
        body(Some("tenant-a")).cache_salt.as_deref(),
        Some("tenant-a")
    );
    assert_eq!(body(None).cache_salt, None);
}
