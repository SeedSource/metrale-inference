// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: the verify-time token pick: each speculative verify position's
//! logits go through `logit_processors::process_position_logits`, the same
//! per-position function the non-MTP decode path calls.
//!
//! Owner: scheduler.
//! Invariants:
//! - `verify_pick_all_with_pipeline` returns with the grammar matcher at the
//!   history depth it entered with, and with `inside_thinking`,
//!   `think_ended`, `think_just_ended` and (2026-09-29, A144)
//!   `inside_tool_body` unchanged. Speculative `accept_token` advances are
//!   rolled back by history delta, and the flags are restored (pick_all.rs,
//!   pick_positions.rs).
//! - 2026-09-29: A146: `pick_positions_from_host` pushes each position's
//!   pick onto `a.output_tokens` before the next position (and truncates on
//!   exit), so every position sees the committed history plus the window's
//!   earlier picks, as decode would; it passes `verify_pos` 0. The
//!   `min_tokens` checks (`MinTokensEosMask`, `ForcedTokenFastPath`) and the
//!   sampling seed add `verify_pos` for callers that do not push.

mod argmax;
mod fast_masked;
mod pick_all;
mod pick_positions;
#[cfg(test)]
mod pick_positions_tests;
mod scratch;

use crate::scheduler::ActiveSeq;
use crate::scheduler::helpers::bf16_to_f32;
use crate::scheduler::logit_processors::LogitsContext;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::Model;

/// 2026-09-25: pick the token for one verify position.
///
/// Dequantises `logits_bytes` (`vocab_size` BF16 values, or FP32 when
/// `is_fp32`) and runs `process_position_logits` with `PositionKind::Verify`
/// penalties. Returns the first of:
/// - the token `process_position_logits` returns (the
///   `METRALE_FORCE_TEMP_ZERO` raw argmax, or a forced grammar token);
/// - a sample from the processed logits, when `mtp_verify_sample` is on and
///   the sequence's temperature is above 0;
/// - the last-index argmax of the processed logits (2026-09-29, A144b:
///   decode's tie-break).
///
/// The pipeline stages mutate `a` (for example the F2ConfidenceEarlyStop
/// streak and `sentence_defer_count`). `verify_pos` is this position's index
/// in the verify span; it offsets the sampling seed and the token count of
/// the `min_tokens` checks.
pub fn verify_pick_with_pipeline(
    logits_bytes: &[u8],
    is_fp32: bool,
    vocab_size: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    verify_pos: usize,
) -> u32 {
    use crate::scheduler::mtp_timing::Phase;
    // 2026-09-25: dequantise into the reused thread-local buffer
    // (`scratch.rs`). `clear` then `extend` writes all `vocab_size` entries
    // before any read.
    let t_dequant = ctx.clock.now();
    let mut f32_logits = scratch::DEQUANT_SCRATCH.with(|s| std::mem::take(&mut *s.borrow_mut()));
    f32_logits.clear();
    f32_logits.reserve(vocab_size);
    if is_fp32 {
        f32_logits.extend((0..vocab_size).map(|j| {
            let off = j * 4;
            f32::from_le_bytes([
                logits_bytes[off],
                logits_bytes[off + 1],
                logits_bytes[off + 2],
                logits_bytes[off + 3],
            ])
        }));
    } else {
        f32_logits.extend((0..vocab_size).map(|j| {
            let lo = logits_bytes[j * 2];
            let hi = logits_bytes[j * 2 + 1];
            bf16_to_f32(lo, hi)
        }));
    }
    ctx.tel.mark(Phase::Dequant, t_dequant);
    // 2026-09-25: the guard hands the buffer back to `DEQUANT_SCRATCH` on
    // every return below.
    let mut f32_logits = scratch::ScratchGuard(f32_logits);

    // 2026-09-25: verify positions take the sequence's repetition, presence,
    // frequency, LZ and DRY penalties and the min-reasoning `</think>` floor
    // bias, with temperature 0 and no seed (`penalty_params_for`). Built
    // before `a` is borrowed mutably below.
    //
    // 2026-09-29: A144: the base bias is the one decode would apply at this
    // position (`speculative_base_logit_bias`). It was empty, so the
    // server's tools-active `<tool_call>` +3.0 nudge (and any client
    // `logit_bias`) never reached verified tokens and spec-on diverged from
    // spec-off on tool-bearing requests. `a` carries this position's think
    // and tool-body state (advanced per position by
    // `pick_positions_from_host`), so the in-tool-body opener strip in
    // `penalty_params_for` is per position too. The raw-argmax probe runs
    // only in decode's device-argmax regime, on a `think_ended` row with a
    // non-empty bias.
    let base_bias = crate::scheduler::sample_step::speculative_base_logit_bias(
        a,
        verify_pos,
        ctx.think_end_token,
        ctx.sampling.think_ended_gpu_argmax,
        || argmax::argmax_first_wins(&f32_logits),
    );
    let penalties = crate::scheduler::sample_step::penalty_params_for(
        a,
        crate::scheduler::sample_step::PositionKind::Verify,
        0.0,
        None,
        base_bias,
        ctx.watchdog.min_reasoning_floor,
    );

    // 2026-09-25: a `Some(tok)` from `process_position_logits` is the
    // force-temp-zero argmax or a forced grammar token, emitted as is. That
    // call never advances the grammar matcher; the position loop in
    // `pick_positions.rs` does.
    let t_proc = ctx.clock.now();
    // 2026-09-25: the `min_tokens` checks count
    // `output_tokens.len() + verify_pos`.
    let pos_ctx = LogitsContext {
        verify_pos,
        ..ctx.clone()
    };
    if let Some(tok) = crate::scheduler::logit_processors::process_position_logits(
        &mut f32_logits,
        a,
        &pos_ctx,
        &penalties,
        crate::scheduler::sample_step::PositionKind::Verify,
    ) {
        ctx.tel.mark(Phase::PipelineProc, t_proc);
        return tok;
    }
    ctx.tel.mark(Phase::PipelineProc, t_proc);

    // 2026-09-25: sample instead of taking the argmax when
    // `mtp_verify_sample` is on (`METRALE_NO_MTP_VERIFY_SAMPLE=1` turns it
    // off), the temperature is above 0 and `force_temp_zero` is off. The
    // penalties were applied in place above, so the sampler gets neutral
    // penalty fields, and masked tokens are already at -inf. min_p is
    // `effective_min_p` (0.0 under `METRALE_NO_MTP_MINP=1`). The seed offset
    // `output_tokens.len() + verify_pos` is the one the non-MTP path
    // (`decode_logits_seq::process_seq_logits`) uses for the same emitted
    // position.
    if ctx.sampling.mtp_verify_sample && a.temperature > 0.0 && !ctx.sampling.force_temp_zero {
        let t_sample = ctx.clock.now();
        let step_seed = a
            .seed
            .map(|s| s.wrapping_add((a.output_tokens.len() + verify_pos) as u64));
        let sampler_shape = metrale_sampling::SamplingParams {
            temperature: a.temperature,
            top_k: a.top_k,
            top_p: a.top_p,
            top_n_sigma: a.top_n_sigma,
            min_p: crate::scheduler::sample_step::effective_min_p(a.min_p, &ctx.sampling),
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            repetition_penalty_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            lz_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: penalties.dry_base,
            dry_allowed_length: penalties.dry_allowed_length,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: step_seed,
        };
        // 2026-09-25: SAFETY: `f32_logits` holds exactly `vocab_size`
        // initialised f32s, so `vocab_size * 4` bytes are in bounds, and u8
        // has no alignment requirement.
        let f32_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
        let sampled = metrale_sampling::sample_with_params_history(f32_bytes, &sampler_shape, &[]);
        // 2026-09-25: timed as `Phase::Argmax` because the sample takes the
        // argmax's place.
        ctx.tel.mark(Phase::Argmax, t_sample);
        return sampled;
    }

    // 2026-09-29: A144b: this is decode's host greedy pick for this position
    // (temperature 0 reaches here only when `process_position_logits`
    // returned no forced token), so it uses decode's tie-break, last index
    // wins (`greedy_pick_last_wins`), not first-index-wins
    // `argmax_first_wins`. The first-wins pick was the A144b root cause: on
    // quantised checkpoints exact logit ties are common, and spec-off decode
    // and K3 verify emitted different tied ids (54/60 divergent TEB
    // transcripts at temperature 0). `argmax_first_wins` stays where it
    // mirrors the device argmax instead (`speculative_base_logit_bias`'s
    // raw-argmax probe above), whose tie order is a different, unverified
    // one.
    let t_argmax = ctx.clock.now();
    let best_id = argmax::greedy_pick_last_wins(&f32_logits);
    ctx.tel.mark(Phase::Argmax, t_argmax);
    best_id
}

/// 2026-09-29: A146: the committed history followed by the window's
/// positions `0..K-1` (`argmax_ids[..K-1]`; the last position is never
/// history for another). Pair with [`position_history`]: position `i` must
/// be judged against the committed tokens PLUS picks `0..i-1`, which is what
/// decode (which has committed them) and the host path
/// (`pick_positions_from_host` pushes them) see. Before this the fast paths
/// tested immunity against the committed history only, so `[X, X]` with X
/// new passed position 1 unpenalised while the host path and decode
/// penalised it.
pub(crate) fn window_penalty_history(a: &ActiveSeq, argmax_ids: &[u32]) -> Vec<u32> {
    let prefix = &argmax_ids[..argmax_ids.len().saturating_sub(1)];
    let mut h = Vec::with_capacity(a.output_tokens.len() + prefix.len());
    h.extend_from_slice(&a.output_tokens);
    h.extend_from_slice(prefix);
    h
}

/// 2026-09-29: A146: position `i`'s penalty history (scoped like the
/// pipeline's `penalty_history_scope`) out of a [`window_penalty_history`]
/// buffer whose committed part is `base_len` long.
pub(crate) fn position_history<'h>(
    h: &'h [u32],
    base_len: usize,
    i: usize,
    ctx: &LogitsContext,
) -> &'h [u32] {
    crate::scheduler::sample_step::penalty_history_scope(
        &h[..(base_len + i).min(h.len())],
        ctx.tool_call_end_token,
    )
}

/// 2026-10-04: Whether an MTP bootstrap row picks its token through
/// [`pick_decode_row_with_pipeline`] instead of the penalties-only sampler:
/// inside `<think>` (A146), or, with `first_after_close`
/// (`METRALE_THINK_END_AT_TOKEN0`, default on), on the first token after a
/// `</think>` (`think_just_ended`). A95: a token-0 close always reaches its
/// first answer token here, with no drafts yet. That token is the one
/// `PinToToolCallStart` acts on, and `emit_token` clears the flag on it;
/// `PostCloseThinkMask` and `MinTokensEosMask` act there too, as in decode
/// and in the verify window. `=0` keeps the thinking-only rule.
pub fn bootstrap_takes_pipeline(a: &ActiveSeq, first_after_close: bool) -> bool {
    a.inside_thinking || (first_after_close && a.think_just_ended)
}

/// 2026-09-29: A146, spec-in-think parity: pick ONE decode row (the MTP
/// bootstrap token) through the full host pipeline, as `process_decode_logits` does for every thinking row.
/// The bootstrap's `sample_token_with_grammar` applies penalties and bias
/// only (no forced `</think>` injection, mid-word mask, F2 or pin), so a
/// bootstrap inside `<think>` could emit a token spec-off never would.
/// `None` on a D2H failure (the caller fails the step as before).
pub fn pick_decode_row_with_pipeline(
    model: &dyn Model,
    row_logits: DevicePtr,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> Option<u32> {
    let vocab = model.vocab_size();
    let is_fp32 = model.decode_logits_fp32();
    let mut buf = vec![0u8; vocab * if is_fp32 { 4 } else { 2 }];
    model.copy_logits_to_host(row_logits, &mut buf).ok()?;
    // 2026-09-29: the pipeline mutates the accumulators on `a` directly here
    // (decode semantics); a stale verify-window trail must not overwrite
    // them.
    a.spec_think_trail.clear();
    Some(verify_pick_with_pipeline(&buf, is_fp32, vocab, a, ctx, 0))
}

pub use pick_all::verify_pick_all_with_pipeline;
