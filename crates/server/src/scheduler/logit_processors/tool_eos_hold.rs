// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The post-think tool-turn EOS hold of spec-off decode, and its
//! pre-sample form behind `METRALE_TOOL_EOS_HOLD_MASK` (default off).
//!
//! The hold (`decode_logits_step/per_token.rs`) drops a sampled EOS on a
//! tool-armed turn until `POST_THINK_MIN_CONTENT` output tokens follow the
//! thinking ones, so the model has room to open a tool call. The dropped
//! token is still the next input (`last_token`), the A103 feed: after an
//! end-of-turn `<|user|>` GLM-5.3 writes a user turn of its own (TEB TC-45).
//! With the lever on, [`mask_stops`] removes the stop tokens from the logits
//! while the hold applies, so the sampler picks another token and nothing is
//! dropped or fed; the post-sample hold is then off. `emit_token` (the
//! speculative commit) has no hold, so the mask runs on spec-off decode only
//! (`PositionKind::FinalDecode`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use crate::scheduler::ActiveSeq;

/// 2026-10-04: Output tokens past the thinking ones before a tool-armed
/// turn's EOS is honoured.
pub(crate) const POST_THINK_MIN_CONTENT: u32 = 16;

/// 2026-10-04: Whether the hold applies to the next token: a tool-armed turn
/// (`require_tool_call` or `tool_request`) after `</think>`, with fewer than
/// `POST_THINK_MIN_CONTENT` output tokens past the thinking ones.
pub(crate) fn applies(a: &ActiveSeq) -> bool {
    let content = (a.output_tokens.len() as u32).saturating_sub(a.thinking_tokens);
    (a.require_tool_call || a.tool_request) && a.think_ended && content < POST_THINK_MIN_CONTENT
}

/// 2026-10-04: Set `a.eos_tokens` to `-inf` while the hold applies, unless no
/// other token would stay finite (a grammar that allows only a stop): that
/// stop then ends the turn. Returns whether it masked.
pub(crate) fn mask_stops(logits: &mut [f32], a: &ActiveSeq) -> bool {
    if !applies(a) {
        return false;
    }
    let is_stop = |i: usize| a.eos_tokens.iter().any(|&e| e as usize == i);
    if !logits
        .iter()
        .enumerate()
        .any(|(i, v)| v.is_finite() && !is_stop(i))
    {
        return false;
    }
    for &e in &a.eos_tokens {
        if let Some(v) = logits.get_mut(e as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
    true
}
