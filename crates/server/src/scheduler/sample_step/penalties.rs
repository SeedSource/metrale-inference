// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Penalty and bias parameters for one decode position: the
//! position kind, the penalty-history scope, the in-tool opener-bias strip,
//! `effective_min_p` and the `SamplingParams` builder.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Which position a [`penalty_params_for`] or
/// [`crate::scheduler::logit_processors::process_position_logits`] call is
/// for: the final decode position (`decode_logits_seq::process_seq_logits`)
/// or an MTP verify or bootstrap position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::scheduler) enum PositionKind {
    FinalDecode,
    Verify,
}

impl PositionKind {
    /// 2026-09-25: The path label written into the AdaDec diagnostic record
    /// (`METRALE_ADADEC_DIAGNOSTIC`).
    pub(in crate::scheduler) fn adadec_label(self) -> &'static str {
        match self {
            PositionKind::FinalDecode => "decode",
            PositionKind::Verify => "verify",
        }
    }
}

/// 2026-09-25: The token history the penalties see: the tokens after the
/// last `tool_call_end_token` in `output_tokens`, or all of `output_tokens`
/// when there is no such token or no end token is configured.
///
/// The repetition penalty is applied once per occurrence in the history
/// (`apply_penalties_and_bias`), so without the cut every completed
/// parallel call would penalise the next call's scaffold tokens once more.
pub(in crate::scheduler) fn penalty_history_scope(
    output_tokens: &[u32],
    tool_call_end_token: Option<u32>,
) -> &[u32] {
    match tool_call_end_token.and_then(|t| output_tokens.iter().rposition(|&x| x == t)) {
        Some(p) => &output_tokens[p + 1..],
        None => output_tokens,
    }
}

/// 2026-09-25: Inside a tool body, drop positive bias entries on the
/// `<tool_call>` opener. `sampling_setup` gives the opener a positive bias
/// on tools-active requests to start a call; inside a body that bias can
/// only push toward a spurious re-open. Negative entries are kept.
pub(in crate::scheduler) fn strip_in_tool_opener_bias(
    logit_bias: &mut Vec<(u32, f32)>,
    in_tool: bool,
    opener: Option<u32>,
) {
    if !in_tool {
        return;
    }
    let Some(tc_open) = opener else {
        return;
    };
    logit_bias.retain(|&(id, delta)| id != tc_open || delta <= 0.0);
}

/// 2026-09-25: The min_p handed to the sampler: `min_p`, or `0.0` when
/// the run's `mtp_minp` lever is off. The lever is on unless
/// `METRALE_NO_MTP_MINP=1`.
pub(in crate::scheduler) fn effective_min_p(
    min_p: f32,
    levers: &crate::scheduler::logit_processors::SamplingLevers,
) -> f32 {
    if levers.mtp_minp { min_p } else { 0.0 }
}

/// 2026-09-25: Build the penalty and bias [`SamplingParams`] for one
/// position of `a`. Both position kinds use it: the final decode position
/// and the MTP verify and bootstrap positions.
///
/// - `lz_penalty` is 0 while a grammar is active.
/// - `dry_multiplier` is 0 inside a tool body (outside thinking).
/// - Positive `<tool_call>` opener bias is dropped inside a tool body.
/// - The minimum-reasoning floor below is appended to `logit_bias`.
///
/// `FinalDecode` callers pass the step's temperature, seed and the
/// sequence's `logit_bias`. `Verify` callers pass temperature 0.0 and no
/// seed; a debug build asserts this. 2026-09-29: their base bias is the one
/// decode would apply at the same position ([`speculative_base_logit_bias`],
/// A144): the sequence's `logit_bias` (the server's `<tool_call>` nudge
/// included) when decode runs the host pipeline for the row, empty when
/// decode's device-argmax path would skip it.
pub(in crate::scheduler) fn penalty_params_for(
    a: &ActiveSeq,
    kind: PositionKind,
    temperature: f32,
    seed: Option<u64>,
    base_logit_bias: Vec<(u32, f32)>,
    // 2026-09-25: MODEL.toml `[behavior].min_reasoning_floor_tokens`,
    // carried on `WatchdogParams`: 16 when unset, 0 disables the floor.
    min_reasoning_floor: u32,
) -> SamplingParams {
    // 2026-09-29: the base bias is no longer pinned empty for `Verify`
    // (A144): verify must carry the `logit_bias` decode applies, or spec-on
    // diverges from spec-off on every tools-present request.
    debug_assert!(
        kind != PositionKind::Verify || (temperature == 0.0 && seed.is_none()),
        "Verify positions must pass temperature=0.0, seed=None"
    );
    let in_tool = a.inside_tool_body && !a.inside_thinking;
    let mut logit_bias = base_logit_bias;

    strip_in_tool_opener_bias(&mut logit_bias, in_tool, a.tool_call_start_token);

    // 2026-09-25: Minimum-reasoning floor: while inside thinking with
    // fewer than `floor` thinking tokens, bias `</think>` by -8.0 (a strong
    // push, not a mask). A `thinking_budget` below the floor turns it off.
    let floor = min_reasoning_floor;
    if floor > 0
        && a.inside_thinking
        && a.thinking_tokens < floor
        && a.thinking_budget.unwrap_or(floor) >= floor
        && let Some(end_tok) = a.think_end_token
    {
        logit_bias.push((end_tok, -8.0f32));
    }

    SamplingParams {
        temperature,
        top_k: a.top_k,
        top_p: a.top_p,
        top_n_sigma: a.top_n_sigma,
        min_p: a.min_p,
        logit_bias,
        // 2026-09-25: The repetition, presence and frequency penalties
        // apply inside a tool body too; only DRY is turned off there.
        repetition_penalty: a.repetition_penalty,
        repetition_penalty_window: a.repetition_penalty_window,
        presence_penalty: a.presence_penalty,
        frequency_penalty: a.frequency_penalty,
        lz_penalty: if a.grammar_state.is_some() {
            0.0
        } else {
            a.lz_penalty
        },
        // 2026-09-25: DRY penalises repeated n-grams, and JSON tool
        // arguments repeat short ones such as `","` legitimately.
        dry_multiplier: if in_tool { 0.0 } else { a.dry_multiplier },
        dry_base: a.dry_base,
        dry_allowed_length: a.dry_allowed_length,
        dry_sequence_breakers: a.dry_sequence_breakers.clone(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed,
    }
}

/// 2026-09-29: A144: the base `logit_bias` a speculative position (MTP and
/// DFlash verify, MTP bootstrap) hands to [`penalty_params_for`] so its pick
/// matches what the single-row decode path would emit at the same position.
///
/// Decode applies `a.logit_bias` when it runs the host pipeline for the row.
/// Its device-argmax path skips the bias
/// (`decode_logits_step::decode_row_uses_gpu_argmax`), except when that
/// argmax lands on a post-think `</think>`/`<think>` id, where it redoes the
/// step on the host with the bias. Parity with decode, not "always apply",
/// is the contract, so all three cases are mirrored:
///  * bias empty: empty;
///  * decode would run the host pipeline at `a.output_tokens.len() +
///    verify_pos`: `a.logit_bias`;
///  * decode would take the device argmax: empty, unless `a.think_ended` and
///    the position's raw argmax (`raw_argmax`, evaluated only in this case)
///    is `think_end_token` or `a.think_start_token`.
///
/// `admit_think_ended` is `think_ended_gpu_argmax`
/// (`METRALE_NO_THINKENDED_GPU_ARGMAX` turns it off). `a` must reflect the
/// position's state: the verify loop advances the think and tool-body flags
/// per position (`pick_positions_from_host`). The in-tool-body opener strip
/// is not done here; [`penalty_params_for`] applies it from the same `a`.
pub(in crate::scheduler) fn speculative_base_logit_bias(
    a: &ActiveSeq,
    verify_pos: usize,
    think_end_token: Option<u32>,
    admit_think_ended: bool,
    raw_argmax: impl FnOnce() -> u32,
) -> Vec<(u32, f32)> {
    if a.logit_bias.is_empty() {
        return Vec::new();
    }
    if !crate::scheduler::decode_logits_step::decode_row_uses_gpu_argmax(
        a,
        a.output_tokens.len() + verify_pos,
        admit_think_ended,
    ) {
        return a.logit_bias.clone();
    }
    if a.think_ended {
        let tok = raw_argmax();
        if Some(tok) == think_end_token || Some(tok) == a.think_start_token {
            return a.logit_bias.clone();
        }
    }
    Vec::new()
}

/// 2026-09-29: A144: true when a speculative device-argmax shortcut (the
/// verify grammar and grammarless fast paths, the masked chat fast path, the
/// DFlash raw-argmax verdict, the DFlash batched verify) must not be taken
/// because decode would apply a non-empty `logit_bias` to this row: the bias
/// can raise a competitor above the raw argmax, and those shortcuts never
/// see it. When decode would itself take its device argmax (bias skipped),
/// the shortcut stays legal; the post-think structural-id exception is left
/// to the callers' existing structural fallbacks.
///
/// Evaluated on the step-start state. Sound for the fast paths because they
/// require `!inside_thinking` (no `</think>` can flip the think flags inside
/// the window) and the `min_tokens` term only relaxes with position;
/// conservative (forces host) otherwise.
pub(in crate::scheduler) fn speculative_bias_forces_host(
    a: &ActiveSeq,
    admit_think_ended: bool,
) -> bool {
    !a.logit_bias.is_empty()
        && !crate::scheduler::decode_logits_step::decode_row_uses_gpu_argmax(
            a,
            a.output_tokens.len(),
            admit_think_ended,
        )
}
