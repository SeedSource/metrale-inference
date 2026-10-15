// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Encoder from `ir::StreamDelta`s to OpenAI
//! `chat.completion.chunk` SSE events.
//!
//! Owner: server (OpenAI adapter).
//! Invariants:
//! - Every `chat.completion.chunk` of one response carries the same id.
//! - The role chunk comes first and `[DONE]` last.

use axum::response::sse::Event;

use crate::ir::StreamDelta;

use super::{ChatCompletionChunk, Usage};

/// 2026-09-26: Encode one delta into its OpenAI SSE event(s).
/// `include_usage` is the request's `stream_options.include_usage`: it
/// chooses between a usage-only chunk followed by a finish chunk without
/// usage, and one finish chunk carrying usage.
pub(crate) fn delta_to_chunk_events(
    d: &StreamDelta,
    model: &str,
    id: &str,
    include_usage: bool,
) -> Vec<Event> {
    delta_to_payloads(d, model, id, include_usage)
        .into_iter()
        .map(|payload| Event::default().data(payload))
        .collect()
}

/// 2026-09-26: The `data:` payloads for one delta, in emission order. A
/// separate function so tests can read the chunk JSON; axum's `Event`
/// cannot be read back.
fn delta_to_payloads(d: &StreamDelta, model: &str, id: &str, include_usage: bool) -> Vec<String> {
    match d {
        StreamDelta::Content { text, token_ids } => vec![chunk_json(
            ChatCompletionChunk::content_chunk(model, id, text.clone())
                .with_token_ids(token_ids.clone()),
        )],
        StreamDelta::Reasoning { text, token_ids } => vec![chunk_json(
            ChatCompletionChunk::reasoning_chunk(model, id, text.clone())
                .with_token_ids(token_ids.clone()),
        )],
        StreamDelta::ToolCallStart {
            index,
            id: call_id,
            name,
        } => {
            // 2026-09-26: `tool_call_start_chunk` takes a whole `ToolCall`;
            // the delta carries only id and name, and the start chunk's
            // arguments are empty.
            let tc = crate::tool_parser::ToolCall {
                id: call_id.clone(),
                call_type: "function".to_string(),
                function: crate::tool_parser::FunctionCall {
                    name: name.clone(),
                    arguments: String::new(),
                },
            };
            vec![chunk_json(ChatCompletionChunk::tool_call_start_chunk(
                model, id, &tc, *index,
            ))]
        }
        StreamDelta::ToolCallArgs {
            index,
            fragment,
            token_ids,
        } => vec![chunk_json(
            ChatCompletionChunk::tool_call_args_fragment(model, id, *index, fragment)
                .with_token_ids(token_ids.clone()),
        )],
        StreamDelta::Refusal { text } => vec![chunk_json(ChatCompletionChunk::refusal_chunk(
            model,
            id,
            text.clone(),
        ))],
        StreamDelta::Finish {
            reason,
            usage,
            token_ids,
        } => {
            let wire_usage = wire_usage(usage);
            if include_usage {
                // 2026-09-26: The usage-only chunk (`choices: []`) comes
                // first, then the finish chunk without usage. The residual
                // token ids go on the finish chunk: the usage-only chunk
                // has no choice to carry them.
                vec![
                    chunk_json(ChatCompletionChunk::usage_only_chunk(model, id, wire_usage)),
                    chunk_json(
                        ChatCompletionChunk::final_chunk_no_usage(model, id, reason.as_wire())
                            .with_stop_reason(reason.stop_reason())
                            .with_token_ids(token_ids.clone()),
                    ),
                ]
            } else {
                vec![chunk_json(
                    ChatCompletionChunk::done_chunk(model, id, reason.as_wire(), wire_usage)
                        .with_stop_reason(reason.stop_reason())
                        .with_token_ids(token_ids.clone()),
                )]
            }
        }
        // 2026-09-26: The error payload is already wire JSON; it is sent
        // unchanged.
        StreamDelta::Error { message } => vec![message.clone()],
    }
}

fn chunk_json(chunk: ChatCompletionChunk) -> String {
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// 2026-09-26: The same `From<&ir::Usage>` mapping the blocking encoder
/// uses.
fn wire_usage(u: &crate::ir::Usage) -> Usage {
    Usage::from(u)
}

/// 2026-09-26: The streamed `/v1/chat/completions` response: the role
/// chunk, the encoded deltas, then `[DONE]`, with SSE keep-alive. It mints
/// the chunk id that every chunk of the response shares.
pub(crate) fn encode_sse_response(
    deltas: crate::ir::DeltaStream,
    model: String,
    include_usage: bool,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use axum::response::sse::{Event, KeepAlive, Sse};
    use futures::StreamExt;

    let chunk_id = super::new_chunk_id();
    let role_chunk = super::ChatCompletionChunk::role_chunk(&model, &chunk_id);
    let role_json = serde_json::to_string(&role_chunk).unwrap_or_default();
    let role_event = futures::stream::once(async move {
        Ok::<_, std::convert::Infallible>(Event::default().data(role_json))
    });
    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    // 2026-09-26: Logged once, when the first delta reaches this encoder;
    // compare with `handle_token`'s first-delta log line.
    let first_event_logged = std::cell::Cell::new(false);
    let token_stream = deltas.flat_map(move |d| {
        if !first_event_logged.replace(true) {
            tracing::debug!("stream: first delta reaches the SSE encoder");
        }
        let events: Vec<Result<Event, std::convert::Infallible>> =
            delta_to_chunk_events(&d, &model, &chunk_id, include_usage)
                .into_iter()
                .map(Ok)
                .collect();
        futures::stream::iter(events)
    });
    let full_stream = role_event.chain(token_stream).chain(done_event);
    Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{FinishReason, StreamDelta};

    /// 2026-09-26: Parse a chunk payload and zero `created`, so two chunks
    /// built across a second boundary still compare equal.
    fn norm(payload: &str) -> serde_json::Value {
        let mut v: serde_json::Value = serde_json::from_str(payload).expect("valid chunk JSON");
        v["created"] = serde_json::json!(0);
        v
    }

    #[test]
    fn content_delta_encodes_via_content_chunk() {
        let d = StreamDelta::Content {
            text: "hi".into(),
            token_ids: Vec::new(),
        };
        let p = delta_to_payloads(&d, "m", "chatcmpl-x", false);
        assert_eq!(p.len(), 1);
        assert!(p[0].contains("\"content\":\"hi\""), "payload: {}", p[0]);
        assert!(
            !p[0].contains("token_ids"),
            "opt-out payload must omit token_ids: {}",
            p[0]
        );
        let expected = serde_json::to_string(&ChatCompletionChunk::content_chunk(
            "m",
            "chatcmpl-x",
            "hi".to_string(),
        ))
        .unwrap();
        assert_eq!(norm(&p[0]), norm(&expected));

        let d = StreamDelta::Content {
            text: "hi".into(),
            token_ids: vec![1, 2],
        };
        let p = delta_to_payloads(&d, "m", "chatcmpl-x", false);
        assert!(p[0].contains("\"token_ids\":[1,2]"), "payload: {}", p[0]);
    }

    #[test]
    fn reasoning_delta_encodes_via_reasoning_chunk() {
        let d = StreamDelta::Reasoning {
            text: "mull".into(),
            token_ids: vec![9],
        };
        let p = delta_to_payloads(&d, "m", "id-1", false);
        assert_eq!(p.len(), 1);
        assert!(
            p[0].contains("\"reasoning_content\":\"mull\""),
            "payload: {}",
            p[0]
        );
        let expected = serde_json::to_string(
            &ChatCompletionChunk::reasoning_chunk("m", "id-1", "mull".to_string())
                .with_token_ids(vec![9]),
        )
        .unwrap();
        assert_eq!(norm(&p[0]), norm(&expected));
    }

    #[test]
    fn tool_call_start_and_args_deltas_encode_openai_tool_chunks() {
        let start = StreamDelta::ToolCallStart {
            index: 3,
            id: "call_abc".into(),
            name: "get_weather".into(),
        };
        let p = delta_to_payloads(&start, "m", "id-1", false);
        assert_eq!(p.len(), 1);
        assert!(p[0].contains("\"role\":\"assistant\""), "payload: {}", p[0]);
        assert!(
            p[0].contains(
                "\"tool_calls\":[{\"index\":3,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]"
            ),
            "payload: {}",
            p[0]
        );

        let args = StreamDelta::ToolCallArgs {
            index: 3,
            fragment: "{\"city\":".into(),
            token_ids: Vec::new(),
        };
        let p = delta_to_payloads(&args, "m", "id-1", false);
        assert_eq!(p.len(), 1);
        let expected = serde_json::to_string(&ChatCompletionChunk::tool_call_args_fragment(
            "m",
            "id-1",
            3,
            "{\"city\":",
        ))
        .unwrap();
        assert_eq!(norm(&p[0]), norm(&expected));
    }

    /// 2026-09-26: The usage-only chunk carries the timing fields that
    /// `Usage::from(&ir::Usage)` produces.
    #[test]
    fn finish_usage_chunk_carries_the_raw_timing_components() {
        let usage = crate::ir::Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            cached_prompt_tokens: 0,
            reasoning_tokens: 0,
            accepted_prediction_tokens: 0,
            time_to_first_token_ms: 12.5,
            decode_time_ms: 100.0,
            response_tokens_per_second: 40.0,
        };
        let d = StreamDelta::Finish {
            reason: FinishReason::Stop,
            usage,
            token_ids: Vec::new(),
        };
        let p = delta_to_payloads(&d, "m", "id-1", true);
        let v: serde_json::Value = serde_json::from_str(&p[0]).unwrap();
        assert_eq!(v["usage"]["decode_time_ms"], 100.0, "{}", p[0]);
        assert_eq!(v["usage"]["total_time_ms"], 112.5, "{}", p[0]);
        assert_eq!(v["usage"]["time_to_first_token_ms"], 12.5, "{}", p[0]);
        assert_eq!(v["usage"]["response_token/s"], 40.0, "{}", p[0]);
    }

    /// 2026-10-14: A repetition guard cut finishes with `finish_reason` "stop" and
    /// `stop_reason` "repetition_detected" in both framings; any other finish has no
    /// `stop_reason` key (#82 guard fix).
    #[test]
    fn repetition_finish_carries_stop_reason_on_both_framings() {
        let usage = || crate::ir::Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            cached_prompt_tokens: 0,
            reasoning_tokens: 0,
            accepted_prediction_tokens: 0,
            time_to_first_token_ms: 1.0,
            decode_time_ms: 1.0,
            response_tokens_per_second: 1.0,
        };
        for include_usage in [true, false] {
            let d = StreamDelta::Finish {
                reason: FinishReason::Repetition,
                usage: usage(),
                token_ids: Vec::new(),
            };
            let p = delta_to_payloads(&d, "m", "id-1", include_usage);
            let v: serde_json::Value = serde_json::from_str(p.last().unwrap()).unwrap();
            assert_eq!(v["choices"][0]["finish_reason"], "stop", "{p:?}");
            assert_eq!(
                v["choices"][0]["stop_reason"], "repetition_detected",
                "{p:?}"
            );
            let d = StreamDelta::Finish {
                reason: FinishReason::Stop,
                usage: usage(),
                token_ids: Vec::new(),
            };
            let p = delta_to_payloads(&d, "m", "id-1", include_usage);
            assert!(!p.last().unwrap().contains("stop_reason"), "{p:?}");
        }
    }

    #[test]
    fn finish_framing_matches_include_usage_modes() {
        let usage = crate::ir::Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            cached_prompt_tokens: 2,
            reasoning_tokens: 3,
            accepted_prediction_tokens: 4,
            time_to_first_token_ms: 12.5,
            decode_time_ms: 100.0,
            response_tokens_per_second: 40.0,
        };
        let d = StreamDelta::Finish {
            reason: FinishReason::Stop,
            usage,
            token_ids: vec![7],
        };

        let p = delta_to_payloads(&d, "m", "id-1", true);
        assert_eq!(p.len(), 2);
        assert!(p[0].contains("\"choices\":[]"), "payload: {}", p[0]);
        assert!(p[0].contains("\"prompt_tokens\":10"), "payload: {}", p[0]);
        assert!(
            p[0].contains("\"completion_tokens\":5"),
            "payload: {}",
            p[0]
        );
        assert!(p[0].contains("\"total_tokens\":15"), "payload: {}", p[0]);
        assert!(p[0].contains("\"cached_tokens\":2"), "payload: {}", p[0]);
        assert!(p[0].contains("\"reasoning_tokens\":3"), "payload: {}", p[0]);
        assert!(
            p[0].contains("\"accepted_prediction_tokens\":4"),
            "payload: {}",
            p[0]
        );
        assert!(
            p[0].contains("\"rejected_prediction_tokens\":0"),
            "payload: {}",
            p[0]
        );
        assert!(
            p[0].contains("\"time_to_first_token_ms\":12.5"),
            "payload: {}",
            p[0]
        );
        assert!(
            p[0].contains("\"response_token/s\":40.0"),
            "payload: {}",
            p[0]
        );
        assert!(
            p[1].contains("\"finish_reason\":\"stop\""),
            "payload: {}",
            p[1]
        );
        assert!(
            !p[1].contains("\"usage\""),
            "final chunk must omit usage when include_usage=true: {}",
            p[1]
        );
        assert!(p[1].contains("\"token_ids\":[7]"), "payload: {}", p[1]);

        let p = delta_to_payloads(&d, "m", "id-1", false);
        assert_eq!(p.len(), 1);
        assert!(
            p[0].contains("\"finish_reason\":\"stop\""),
            "payload: {}",
            p[0]
        );
        assert!(p[0].contains("\"usage\":{"), "payload: {}", p[0]);
        assert!(p[0].contains("\"total_tokens\":15"), "payload: {}", p[0]);
        assert!(p[0].contains("\"token_ids\":[7]"), "payload: {}", p[0]);
    }

    #[test]
    fn error_delta_payload_is_verbatim() {
        let msg = r#"{"error":{"message":"boom","type":"server_error","code":500}}"#;
        let d = StreamDelta::Error {
            message: msg.to_string(),
        };
        let p = delta_to_payloads(&d, "m", "id-1", false);
        assert_eq!(p, vec![msg.to_string()]);
    }
}
