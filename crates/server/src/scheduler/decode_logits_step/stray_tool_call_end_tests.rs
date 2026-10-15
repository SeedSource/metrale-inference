// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The decode device-argmax fallback for a stray `</tool_call>`
//! (`decode_argmax_hits_stray_tool_call_end`), and that such a row keeps the
//! device argmax otherwise (`decode_row_uses_gpu_argmax`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{decode_argmax_hits_stray_tool_call_end, decode_row_uses_gpu_argmax};
use crate::scheduler::test_support::test_seq;
use crate::scheduler::types::ActiveSeq;

const TOOL_CALL_END: u32 = 9;
const HELLO: u32 = 3;

/// 2026-10-15: a greedy plain-chat row outside thinking, no tools, no grammar.
fn plain_row() -> ActiveSeq {
    let (mut a, _rx) = test_seq(vec![HELLO], 100, None, 4);
    a.finished = false;
    a.temperature = 0.0;
    a.inside_thinking = false;
    a.think_ended = false;
    a.tools_present = false;
    a.grammar_state = None;
    a.inside_tool_body = false;
    a.top_logprobs = None;
    a.min_tokens = 0;
    a.tool_call_end_token = Some(TOOL_CALL_END);
    a
}

#[test]
fn stray_end_argmax_falls_back_only_where_the_mask_applies() {
    let mut a = plain_row();
    let t = Some(TOOL_CALL_END);
    assert!(decode_argmax_hits_stray_tool_call_end(
        true,
        &a,
        t,
        TOOL_CALL_END
    ));
    assert!(!decode_argmax_hits_stray_tool_call_end(true, &a, t, HELLO));
    assert!(
        !decode_argmax_hits_stray_tool_call_end(false, &a, t, TOOL_CALL_END),
        "flag off: the device argmax is emitted as before"
    );
    assert!(!decode_argmax_hits_stray_tool_call_end(
        true,
        &a,
        None,
        TOOL_CALL_END
    ));
    a.tools_present = true;
    assert!(!decode_argmax_hits_stray_tool_call_end(
        true,
        &a,
        t,
        TOOL_CALL_END
    ));
    a.tools_present = false;
    a.inside_tool_body = true;
    assert!(
        !decode_argmax_hits_stray_tool_call_end(true, &a, t, TOOL_CALL_END),
        "closing an opened <tool_call> is legal"
    );
}

#[test]
fn a_row_the_mask_applies_to_keeps_the_device_argmax() {
    // 2026-10-15: like `think_ended` rows for the two think ids, the row stays
    // eligible and falls back only on a hit, instead of taking the host every step.
    let a = plain_row();
    assert!(decode_row_uses_gpu_argmax(&a, a.output_tokens.len(), true));
    assert!(decode_row_uses_gpu_argmax(&a, a.output_tokens.len(), false));
}
