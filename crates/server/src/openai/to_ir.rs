// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Lowers the OpenAI Chat Completions wire request into the chat IR:
//! `IncomingMessage` into `ir::Message`, and `ChatCompletionRequest` into
//! `ir::ChatRequest`. String-keyed `logit_bias`, the `logprobs`/`top_logprobs` pair,
//! `response_format: text` and the thinking channels are resolved here.
//!
//! Owner: server (OpenAI API layer).
//! Invariants:
//! - Both conversions are infallible: a `logit_bias` key that is not a token id is
//!   dropped, and tool-call arguments that are not JSON become `{}`.

use super::{ChatCompletionRequest, IncomingMessage, ResponseFormat};
use crate::ir;
use crate::ir::message::{ContentPart, ImageData, ImageSource, Message, Reasoning, Role, ToolCall};

impl From<&IncomingMessage> for Message {
    fn from(m: &IncomingMessage) -> Self {
        // 2026-09-26: Media parts first, in the order of `m.content.media` (images and
        // videos interleaved as the client sent them), then one text part: the wire parse
        // has already joined all text into `m.content.text`. The chat path reads this order
        // back through `Message::media_kinds`.
        let mut content: Vec<ContentPart> = Vec::new();
        for item in &m.content.media {
            let data = ImageData::from_uri(item.uri.clone());
            content.push(match item.kind {
                crate::ir::MediaKind::Image => ContentPart::Image(ImageSource { data }),
                crate::ir::MediaKind::Video => ContentPart::Video(crate::ir::VideoSource { data }),
            });
        }
        if !m.content.text.is_empty() {
            content.push(ContentPart::Text(m.content.text.clone()));
        }

        // 2026-09-26: Tool-call `arguments` that do not parse as JSON become `{}`; a missing
        // id becomes the empty string.
        let tool_calls: Vec<ToolCall> = m
            .tool_calls
            .as_ref()
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| ToolCall {
                        id: tc.id.clone().unwrap_or_default(),
                        name: tc.function.name.clone(),
                        arguments: serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| serde_json::Value::Object(Default::default())),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Message {
            role: Role::from(m.role.as_str()),
            content,
            tool_calls,
            tool_call_id: m.tool_call_id.clone(),
            name: m.name.clone(),
            reasoning: m.reasoning_content.clone().map(|text| Reasoning { text }),
            tool_error: false,
        }
    }
}

impl From<ChatCompletionRequest> for ir::ChatRequest {
    /// 2026-09-26: Lower the parsed wire request into the [`ir::ChatRequest`] envelope.
    /// Range checks run later, on the envelope (`chat_phases::validate_input`).
    ///
    /// The echo-only fields `service_tier`, `store`, `metadata` and `stream_options` are
    /// not lowered; the chat handler copies them into a `ResponseEcho` first.
    fn from(req: ChatCompletionRequest) -> Self {
        let thinking = req.client_thinking_directive();
        let reasoning_effort = req.client_reasoning_effort();
        let preserve_thinking = req
            .chat_template_kwargs
            .as_ref()
            .and_then(|kw| kw.preserve_thinking);
        let top_logprobs = resolve_top_logprobs(req.logprobs, req.top_logprobs);
        // 2026-09-26: Keys that do not parse as `u32` token ids are dropped.
        let logit_bias: Vec<(u32, f32)> = req.logit_bias.as_ref().map_or(Vec::new(), |map| {
            map.iter()
                .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
                .collect()
        });
        let response_format = match req.response_format {
            None | Some(ResponseFormat::Text) => None,
            Some(ResponseFormat::JsonObject) => Some(ir::ResponseFormat::JsonObject),
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                Some(ir::ResponseFormat::JsonSchema {
                    name: json_schema.name,
                    schema: json_schema.schema,
                    strict: json_schema.strict,
                })
            }
        };
        ir::ChatRequest {
            model: req.model,
            messages: req.messages.iter().map(Into::into).collect(),
            tools: req.tools.unwrap_or_default(),
            tool_choice: req.tool_choice,
            sampling: ir::SamplingParams {
                temperature: req.temperature,
                top_k: req.top_k,
                top_p: req.top_p,
                top_n_sigma: req.top_n_sigma,
                min_p: req.min_p,
                repetition_penalty: req.repetition_penalty,
                presence_penalty: req.presence_penalty,
                frequency_penalty: req.frequency_penalty,
            },
            max_tokens: req.max_tokens,
            min_tokens: req.min_tokens,
            ignore_eos: req.ignore_eos,
            stop: req.stop,
            stream: req.stream,
            n: req.n,
            response_format,
            thinking,
            reasoning_effort,
            preserve_thinking,
            repetition_detection: req.repetition_detection,
            adapter: req.adapter,
            src_lang: req.src_lang,
            tgt_lang: req.tgt_lang,
            num_beams: req.num_beams,
            length_penalty: req.length_penalty,
            early_stopping: req.early_stopping,
            logit_bias,
            top_logprobs,
            seed: req.seed,
            timeout_secs: req.timeout,
            return_token_ids: req.return_token_ids,
        }
    }
}

/// 2026-09-26: Resolve the chat logprobs parameters. An explicit `top_logprobs` count wins,
/// capped at 20, whatever `logprobs` says; `logprobs: true` alone gives 0 alternatives
/// (sampled-token logprobs only); otherwise `None`.
pub(crate) fn resolve_top_logprobs(logprobs: Option<bool>, top_logprobs: Option<u8>) -> Option<u8> {
    match (logprobs, top_logprobs) {
        (_, Some(n)) => Some(n.min(20)),
        (Some(true), None) => Some(0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::MediaKind;
    use crate::ir::message::{ContentPart, ImageData, ImageSource, Reasoning};
    use crate::openai::{IncomingMessage, MediaRef, ParsedContent};
    use crate::tool_parser::{IncomingFunction, IncomingToolCall};

    fn image(uri: &str) -> MediaRef {
        MediaRef {
            kind: MediaKind::Image,
            uri: uri.to_string(),
        }
    }

    fn video(uri: &str) -> MediaRef {
        MediaRef {
            kind: MediaKind::Video,
            uri: uri.to_string(),
        }
    }

    fn msg(role: &str) -> IncomingMessage {
        IncomingMessage {
            role: role.to_string(),
            content: ParsedContent::default(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn text_only_user_message() {
        let mut m = msg("user");
        m.content.text = "hi".into();
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::User);
        assert_eq!(ir.content, vec![ContentPart::Text("hi".into())]);
        assert!(ir.media_kinds().is_empty());
        assert!(ir.tool_calls.is_empty());
        assert!(ir.reasoning.is_none());
        assert!(!ir.tool_error);
    }

    #[test]
    fn empty_content_yields_no_parts() {
        let m = msg("user");
        let ir: Message = (&m).into();
        assert!(ir.content.is_empty());
        assert_eq!(ir.text(), "");
    }

    #[test]
    fn images_precede_text_and_are_preserved_verbatim() {
        let mut m = msg("user");
        m.content.text = "see".into();
        m.content.media = vec![image("data:image/png;base64,AAA")];
        let ir: Message = (&m).into();
        assert_eq!(
            ir.content,
            vec![
                ContentPart::Image(ImageSource {
                    data: ImageData::Base64("data:image/png;base64,AAA".into()),
                }),
                ContentPart::Text("see".into()),
            ]
        );
        assert_eq!(ir.media_kinds(), vec![MediaKind::Image]);
    }

    #[test]
    fn media_order_survives_the_ir_conversion() {
        let mut m = msg("user");
        m.content.text = "which came first?".into();
        m.content.media = vec![video("vvv"), image("iii"), video("www")];
        let ir: Message = (&m).into();
        assert_eq!(
            ir.content,
            vec![
                ContentPart::Video(crate::ir::VideoSource {
                    data: ImageData::Base64("vvv".into()),
                }),
                ContentPart::Image(ImageSource {
                    data: ImageData::Base64("iii".into()),
                }),
                ContentPart::Video(crate::ir::VideoSource {
                    data: ImageData::Base64("www".into()),
                }),
                ContentPart::Text("which came first?".into()),
            ]
        );
        assert_eq!(
            ir.media_kinds(),
            vec![MediaKind::Video, MediaKind::Image, MediaKind::Video],
            "media_kinds is what drives the markers, the pad counts and the encoder items"
        );
    }

    #[test]
    fn remote_url_image_classified_as_url_variant() {
        // 2026-09-26: `http://` and `https://` URIs become `ImageData::Url`; any other string
        // is `ImageData::Base64`.
        let mut m = msg("user");
        m.content.media = vec![
            image("https://example.com/cat.png"),
            image("data:image/png;base64,AAA"),
        ];
        let ir: Message = (&m).into();
        assert_eq!(
            ir.content,
            vec![
                ContentPart::Image(ImageSource {
                    data: ImageData::Url("https://example.com/cat.png".into()),
                }),
                ContentPart::Image(ImageSource {
                    data: ImageData::Base64("data:image/png;base64,AAA".into()),
                }),
            ]
        );
    }

    #[test]
    fn assistant_tool_calls_parse_arguments_to_json() {
        let mut m = msg("assistant");
        m.tool_calls = Some(vec![IncomingToolCall {
            id: Some("call_1".into()),
            function: IncomingFunction {
                name: "get_weather".into(),
                arguments: r#"{"city":"SF"}"#.into(),
            },
        }]);
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::Assistant);
        assert_eq!(ir.tool_calls.len(), 1);
        assert_eq!(ir.tool_calls[0].id, "call_1");
        assert_eq!(ir.tool_calls[0].name, "get_weather");
        assert_eq!(
            ir.tool_calls[0].arguments,
            serde_json::json!({"city": "SF"})
        );
    }

    #[test]
    fn malformed_tool_args_default_to_empty_object() {
        let mut m = msg("assistant");
        m.tool_calls = Some(vec![IncomingToolCall {
            id: None,
            function: IncomingFunction {
                name: "f".into(),
                arguments: "not json".into(),
            },
        }]);
        let ir: Message = (&m).into();
        assert_eq!(ir.tool_calls[0].id, "");
        assert_eq!(ir.tool_calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn reasoning_content_maps_to_first_class_reasoning() {
        let mut m = msg("assistant");
        m.reasoning_content = Some("ponder".into());
        let ir: Message = (&m).into();
        assert_eq!(
            ir.reasoning,
            Some(Reasoning {
                text: "ponder".into()
            })
        );

        let m2 = msg("assistant");
        let ir2: Message = (&m2).into();
        assert!(ir2.reasoning.is_none());
    }

    #[test]
    fn tool_message_preserves_call_id_and_name() {
        let mut m = msg("tool");
        m.content.text = "exit 0".into();
        m.tool_call_id = Some("call_1".into());
        m.name = Some("bash".into());
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::Tool);
        assert_eq!(ir.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(ir.name.as_deref(), Some("bash"));
        assert_eq!(ir.text(), "exit 0");
    }

    #[test]
    fn unknown_role_is_preserved_losslessly() {
        let m = msg("developer");
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::Other("developer".into()));
        assert_eq!(ir.role.as_wire(), "developer");
    }

    fn wire(body: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(body).expect("valid chat request")
    }

    #[test]
    fn envelope_lowers_scalars_and_parses_logit_bias() {
        let req = wire(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "temperature": 0.5,
            "logit_bias": {"42": 1.5, "not-a-token": -1.0},
            "logprobs": true,
            "stop": ["END"],
            "seed": 7,
            "n": 2
        }));
        let ir = ir::ChatRequest::from(req);
        assert_eq!(ir.model, "m");
        assert_eq!(ir.messages.len(), 1);
        assert_eq!(ir.max_tokens, 64);
        assert_eq!(ir.sampling.temperature, Some(0.5));
        assert_eq!(ir.logit_bias, vec![(42, 1.5)]);
        assert_eq!(ir.top_logprobs, Some(0));
        assert_eq!(ir.stop, vec!["END".to_string()]);
        assert_eq!(ir.seed, Some(7));
        assert_eq!(ir.n, 2);
        assert!(ir.tools.is_empty());
        assert!(ir.response_format.is_none());
    }

    #[test]
    fn envelope_maps_text_response_format_to_none() {
        let req = wire(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "text"}
        }));
        assert!(ir::ChatRequest::from(req).response_format.is_none());

        let req = wire(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object"}}}
        }));
        match ir::ChatRequest::from(req).response_format {
            Some(ir::ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            }) => {
                assert_eq!(name, "s");
                assert_eq!(schema, serde_json::json!({"type": "object"}));
                assert!(strict);
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }

    #[test]
    fn resolve_top_logprobs_matrix() {
        assert_eq!(resolve_top_logprobs(Some(true), None), Some(0));
        assert_eq!(resolve_top_logprobs(None, Some(5)), Some(5));
        assert_eq!(resolve_top_logprobs(Some(false), Some(3)), Some(3));
        assert_eq!(resolve_top_logprobs(Some(true), Some(99)), Some(20));
        assert_eq!(resolve_top_logprobs(None, None), None);
        assert_eq!(resolve_top_logprobs(Some(false), None), None);
    }
}
