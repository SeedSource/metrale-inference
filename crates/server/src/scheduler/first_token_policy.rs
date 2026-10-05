// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Token-0 grammar policy, applied by `sample_step::sample_first_token`.
//!
//! Later tokens skip the grammar while inside `<think>`
//! (`logit_processors/grammar_bitmask.rs`, `decode_logits_step/per_token.rs`,
//! `emit_step/token.rs`). Token 0 is sampled before
//! `ActiveSeq::inside_thinking` exists, so the same rule is applied here
//! from [`born_inside_thinking`].
//!
//! 2026-10-04: A95: a token 0 that is the model's `</think>` ends thinking
//! at birth ([`end_thinking_at_token0`]), as a later sampled `</think>`
//! does.
//!
//! Owner: scheduler.
//! Invariants:
//! - With a grammar, [`first_token_with`] either hands it to the sampler
//!   and, when sampling succeeds, advances it past token 0, or, when
//!   `policy.grammar_suspended`, leaves it untouched.
//! - 2026-10-04: [`end_thinking_at_token0`] resets the same fields as the
//!   `</think>` branches of `decode_logits_step/per_token.rs`,
//!   `emit_step/token.rs` and `verify_pipeline_helper/pick_positions.rs`.

use crate::grammar::GrammarState;
use crate::scheduler::ActiveSeq;
use anyhow::Result;

/// 2026-09-25: Whether a new sequence starts inside `<think>`: thinking is
/// enabled and the model has a `</think>` token.
///
/// The prefill steps set `ActiveSeq::inside_thinking` from it (or'd with a
/// spontaneous `<think>` first token), and [`FirstTokenPolicy::for_birth`]
/// uses it.
pub(super) fn born_inside_thinking(enable_thinking: bool, think_end_token: Option<u32>) -> bool {
    enable_thinking && think_end_token.is_some()
}

/// 2026-09-25: What token 0 may do with an armed grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FirstTokenPolicy {
    /// 2026-09-25: Token 0 is sampled inside `<think>`: no bitmask and no
    /// `accept_token`, so the matcher is untouched for the first
    /// post-think token.
    pub grammar_suspended: bool,
    /// 2026-09-25: The `<tool_call>` id, suppressed at token 0 while a
    /// grammar is armed and suspended. Later tokens inside `<think>` get it
    /// masked by `ToolCallDuringThinkingMask`.
    pub tool_call_start: Option<u32>,
}

impl FirstTokenPolicy {
    /// 2026-09-25: Policy for a sequence about to be born from these
    /// request facts.
    pub(super) fn for_birth(
        enable_thinking: bool,
        think_end_token: Option<u32>,
        tool_call_start: Option<u32>,
    ) -> Self {
        Self {
            grammar_suspended: born_inside_thinking(enable_thinking, think_end_token),
            tool_call_start,
        }
    }
}

/// 2026-09-25: Core of `sample_first_token`, generic over the
/// model-dependent sampler.
///
/// `sample(suppress_ids, grammar)` is called exactly once. It receives the
/// grammar only when the policy lets the grammar act; then the matcher is
/// advanced past the returned token. With no grammar the call is a plain
/// pass-through.
pub(super) fn first_token_with<F>(
    policy: FirstTokenPolicy,
    suppress_ids: &[u32],
    grammar_state: Option<&mut GrammarState>,
    sample: F,
) -> Result<u32>
where
    F: FnOnce(&[u32], Option<&mut GrammarState>) -> Result<u32>,
{
    let Some(gs) = grammar_state else {
        return sample(suppress_ids, None);
    };
    if policy.grammar_suspended {
        // 2026-09-25: grammar armed, sequence born inside `<think>`: the
        // matcher does not see this token, and `<tool_call>` is suppressed.
        let mut ids = suppress_ids.to_vec();
        if let Some(t) = policy.tool_call_start
            && !ids.contains(&t)
        {
            ids.push(t);
        }
        return sample(&ids, None);
    }
    let tok = sample(suppress_ids, Some(&mut *gs))?;
    gs.accept_token(tok);
    Ok(tok)
}

/// 2026-10-04: A95: when a sequence born inside `<think>` samples the
/// model's `</think>` as token 0, end thinking now, with the same field
/// resets as a later sampled `</think>`: `inside_thinking` false,
/// `think_ended` and the one-shot `think_just_ended` set,
/// `thinking_tokens` left at 0. The stream has already switched to content
/// on that token; without this the scheduler stayed inside thinking,
/// counted the answer against the thinking budget, held back EOS, and
/// forced a second `</think>` mid-answer.
///
/// `enabled` is `SchedLevers::think_end_at_token0`
/// (`METRALE_THINK_END_AT_TOKEN0`, default on; `0` keeps the sequence
/// inside thinking, the behaviour before 2026-10-04). Token 0 was sampled
/// with the grammar suspended, so the matcher has not seen it, as with a
/// later `</think>`. Logs at info like `emit_token`'s close, so a serve log
/// shows when it fires.
pub(super) fn end_thinking_at_token0(a: &mut ActiveSeq, first: u32, enabled: bool) {
    if !enabled || !a.inside_thinking || a.think_end_token != Some(first) {
        return;
    }
    a.inside_thinking = false;
    a.force_end_thinking = false;
    a.sentence_defer_count = 0;
    a.consecutive_confident = 0;
    a.in_code_fence = false;
    a.think_ended = true;
    a.think_just_ended = true;
    tracing::info!("Thinking ended at token 0 (budget={:?})", a.thinking_budget);
}
