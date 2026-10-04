// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `strip_bare_role_literal` with the streaming tool detector.
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

use super::strip_bare_role_literal;
use crate::tool_parser::{DetectorOutput, StreamingToolDetector};

/// 2026-09-26: The two content-phase steps of `handle_token_inner` under test: the
/// role-literal strip (given the detector's `inside_tool_call()`), then the detector.
/// Returns every `ToolCallStart` name, including those from the final flush.
fn stream_names(chunks: &[&str]) -> Vec<String> {
    let mut det = StreamingToolDetector::new();
    let mut names = Vec::new();
    for &c in chunks {
        let mut delta = c.to_string();
        // 2026-09-26: As in `handle_token_inner`, read the in-body flag before feeding
        // this delta.
        let inside_tool_call = det.inside_tool_call();
        strip_bare_role_literal(&mut delta, inside_tool_call);
        if delta.is_empty() {
            continue;
        }
        for o in det.process(&delta) {
            if let DetectorOutput::ToolCallStart { name, .. } = o {
                names.push(name);
            }
        }
    }
    for o in det.flush() {
        if let DetectorOutput::ToolCallStart { name, .. } = o {
            names.push(name);
        }
    }
    names
}

/// 2026-09-26: A `tool_search` name whose leading `tool` arrives as its own fragment
/// streams intact, not as `_search`.
#[test]
fn tool_search_name_split_after_tool_streams_intact() {
    let names = stream_names(&[
        "<tool_call>\n{\"name\": \"",
        "tool",
        "_search\", \"arguments\": {\"query\": \"CRM\"}}",
        "\n</tool_call>",
    ]);
    assert_eq!(names, vec!["tool_search".to_string()]);
}

#[test]
fn tool_call_name_split_after_tool_streams_intact() {
    let names = stream_names(&[
        "<tool_call>\n{\"name\": \"",
        "tool",
        "_call\", \"arguments\": {}}",
        "\n</tool_call>",
    ]);
    assert_eq!(names, vec!["tool_call".to_string()]);
}

#[test]
fn tool_describe_name_split_after_tool_streams_intact() {
    let names = stream_names(&[
        "<tool_call>\n{\"name\": \"",
        "tool",
        "_describe\", \"arguments\": {\"id\": 7}}",
        "\n</tool_call>",
    ]);
    assert_eq!(names, vec!["tool_describe".to_string()]);
}

#[test]
fn ordinary_name_streams_intact() {
    let names = stream_names(&[
        "<tool_call>\n{\"name\": \"get",
        "_weather\", \"arguments\": {\"city\": \"NYC\"}}",
        "\n</tool_call>",
    ]);
    assert_eq!(names, vec!["get_weather".to_string()]);
}

/// 2026-09-26: Outside a tool call, each bare role literal (also padded with spaces) is
/// cleared.
#[test]
fn bare_role_literal_still_stripped_outside_tool_call() {
    for lit in ["user", "assistant", "tool", "  tool  "] {
        let mut d = lit.to_string();
        strip_bare_role_literal(&mut d, false);
        assert!(
            d.is_empty(),
            "bare role literal {lit:?} must be stripped in content"
        );
    }
}

#[test]
fn bare_role_literal_preserved_inside_tool_call() {
    for lit in ["user", "assistant", "tool"] {
        let mut d = lit.to_string();
        strip_bare_role_literal(&mut d, true);
        assert_eq!(d, lit, "fragment {lit:?} must survive inside a tool call");
    }
}

#[test]
fn dsml_split_opener_preserves_tool_fragment_and_streams_call() {
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "\n\n",
        "<｜DSM",
        "L",
        "｜",
        "tool",
        "_calls>\n<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    ];
    let mut outputs = Vec::new();
    for chunk in chunks {
        let mut delta = chunk.to_string();
        strip_bare_role_literal(&mut delta, det.inside_tool_call());
        if !delta.is_empty() {
            outputs.extend(det.process(&delta));
        }
    }
    outputs.extend(det.flush());

    let mut content = String::new();
    let mut calls = Vec::new();
    for output in outputs {
        match output {
            DetectorOutput::Content(text) => content.push_str(&text),
            DetectorOutput::ToolCall(call, index) => calls.push((index, call)),
            _ => {}
        }
    }
    assert!(content.is_empty(), "DSML envelope leaked: {content:?}");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, 0);
    assert_eq!(calls[0].1.function.name, "get_weather");
    assert_eq!(calls[0].1.function.arguments, r#"{"city":"Paris"}"#);
}

#[test]
fn ordinary_content_untouched() {
    for inside in [false, true] {
        let mut d = "the tool ran".to_string();
        strip_bare_role_literal(&mut d, inside);
        assert_eq!(d, "the tool ran");
    }
}

/// 2026-10-03: E1/L17. Replays the content-phase strip over token-sized deltas
/// (as `handle_token_inner` feeds them: `delta` = newly decoded bytes of one token).
fn replay(chunks: &[&str], inside: bool) -> String {
    let mut out = String::new();
    for &c in chunks {
        let mut d = c.to_string();
        strip_bare_role_literal(&mut d, inside);
        out.push_str(&d);
    }
    out
}

#[test]
fn role_word_split_outside_envelope_loses_word_before_fix() {
    let whole = replay(&["a tool."], false);
    let split = replay(&["a", " tool", "."], false);
    println!("E1 L17 whole={whole:?} split={split:?}");
    assert_eq!(whole, "a tool.");
    assert_eq!(split, "a."); // FIXFLIP: want "a tool."
    assert_eq!(replay(&["the", " user", " asked"], false), "the asked"); // FIXFLIP
    assert_eq!(replay(&["I", " am", " an", " assistant"], false), "I am an"); // FIXFLIP
}

#[test]
fn role_word_split_inside_envelope_is_preserved() {
    assert_eq!(replay(&["a", " tool", "."], true), "a tool.");
}
