// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests that a prefill failing after the response sink was
//! taken reaches the request: an `Error` frame on a stream, an `Err` on a
//! blocking call. `handle_prefill_start_error` has no access to the failing
//! request's sink, so `start_chunked_prefill` must deliver it; these tests
//! call it with a model whose `prefill_chunk` fails.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use futures::StreamExt;

use super::test_support::PreemptStubModel;
use crate::api::InferenceRequest;
use crate::api::inference_types::StreamEvent;

/// 2026-09-25: One request body; only `$variant` and its sink fields vary.
macro_rules! request {
    ($variant:ident { $($sink:tt)* }) => {
        InferenceRequest::$variant {
            prompt_tokens: Arc::new(vec![1, 2, 3]),
            session_hash: 0,
            adapter_slot: -1,
            src_lang_id: 0,
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
            image_pixels: vec![],
            max_tokens: 4,
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
            $($sink)*
        }
    };
}

/// 2026-09-25: Call `start_chunked_prefill`, not deferred, with a model
/// whose `prefill_chunk` fails.
fn start(req: InferenceRequest) -> anyhow::Result<()> {
    let mut engine = None;
    super::prefill_a_step::start_chunked_prefill(
        &super::sched_ctx::SchedCtx::for_test(),
        None,
        None,
        None,
        None,
        &PreemptStubModel::default(),
        req,
        &[],
        64,
        0,
        0,
        &mut engine,
        0,
        false,
        None,
        None,
    )
    .map(|_| ())
}

#[tokio::test]
async fn a_streaming_prefill_start_error_reaches_the_client_as_an_error_frame() {
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(16);
    let err = start(request!(Streaming {
        token_tx,
        cancel_flag: Arc::new(AtomicBool::new(false)),
    }))
    .expect_err("the stub's prefill_chunk refuses");
    assert!(
        format!("{err:#}").contains("unused in preempt tests"),
        "{err:#}"
    );

    // 2026-09-25: Read the channel through `stream_terminal::terminated`,
    // as the SSE layer does, so the test sees the frames the client sees.
    let events: Vec<StreamEvent> = crate::api::stream_terminal::terminated(
        tokio_stream::wrappers::ReceiverStream::new(token_rx),
    )
    .collect()
    .await;
    let [StreamEvent::Error(msg)] = events.as_slice() else {
        panic!(
            "expected exactly one Error frame, got {} events",
            events.len()
        );
    };
    assert!(
        msg.starts_with("prefill_chunk failed:") && msg.contains("unused in preempt tests"),
        "the frame must carry the real reason, not the no-result backstop: {msg}"
    );
}

#[tokio::test]
async fn a_blocking_prefill_start_error_reaches_the_caller_as_an_err() {
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    start(request!(Blocking { response_tx })).expect_err("the stub's prefill_chunk refuses");
    let got = response_rx
        .await
        .expect("the sink must be answered, not dropped (a drop reads as 'Inference cancelled')");
    let Err(e) = got else {
        panic!("a failed prefill must not produce a response");
    };
    assert!(
        format!("{e:#}").starts_with("prefill_chunk failed:"),
        "{e:#}"
    );
}

/// 2026-10-03: A prompt whose K/V can never fit the whole pool is refused before any
/// allocation or prefill, with the reason on the sink (race #79: it used to be
/// force-admitted and fail after its full prefill). The 3-token prompt plus its first
/// decode position needs 4 one-token blocks; the pool holds 3. With 4 it is not
/// refused by this check and reaches the stub's failing `prefill_chunk`.
#[test]
fn a_prompt_larger_than_the_kv_pool_is_refused_before_prefill() {
    let run = |total_blocks: usize| {
        let model = PreemptStubModel {
            total_blocks,
            block_size: Some(1),
            ..Default::default()
        };
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        let mut engine = None;
        let err = super::prefill_a_step::start_chunked_prefill(
            &super::sched_ctx::SchedCtx::for_test(),
            None,
            None,
            None,
            None,
            &model,
            request!(Blocking { response_tx }),
            &[],
            64,
            0,
            0,
            &mut engine,
            0,
            false,
            None,
            None,
        )
        .map(|_| ())
        .expect_err("both pools end in an error");
        let sent = response_rx
            .try_recv()
            .expect("the sink must be answered, not dropped");
        (
            format!("{err:#}"),
            format!("{:#}", sent.err().expect("an error, not a response")),
            model.prefilled.lock().unwrap().len(),
        )
    };

    let (err, sent, prefilled) = run(3);
    assert!(err.starts_with("prompt too long for the KV cache"), "{err}");
    assert_eq!(sent, err, "the caller sees the same reason");
    assert_eq!(prefilled, 0, "nothing was prefilled");

    let (err, _, _) = run(4);
    assert!(
        !err.contains("prompt too long"),
        "a prompt that fits is not refused: {err}"
    );
}
