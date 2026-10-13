// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The content leg of `handle_token`: the SimHash and token loop
//! watchdogs over a sanitized content chunk, and the detector's `Content` arm.
//!
//! Owner: server streaming API.
//! Invariants:
//! - Every `cancel_flag` store in this file sets `guard_stop` first
//!   (`cancel_guard_tests` checks it).

use crate::api::sanitizer::sanitize_content_chunk;
use crate::api::stream_guards::check_loop_watchdog;
use crate::ir::StreamDelta;

use super::super::ctx::StreamCtx;
use super::super::state::StreamState;
use super::{
    DeltaVec, forced_len_suppresses, guard_honor_forced_len, simhash_loop_enabled,
    watchdogs_disabled,
};

/// 2026-10-12: A content guard fired (`semantic` names which). Under
/// `METRALE_GUARD_HONOR_FORCED_LEN` (`lever_on`) a request that forces its length
/// (`ignore_eos`, or fewer than `min_tokens` tokens so far) only counts the fire and
/// returns `false`. Otherwise name the guard, cancel the request and return `true`.
pub(super) fn apply_guard_trip(
    state: &mut StreamState,
    lever_on: bool,
    ignore_eos: bool,
    min_tokens: usize,
    semantic: bool,
) -> bool {
    if forced_len_suppresses(lever_on, ignore_eos, min_tokens, state.tokens_seen) {
        state.guard_suppressed = state.guard_suppressed.saturating_add(1);
        return false;
    }
    if semantic {
        tracing::warn!(target: "met::api::chat_stream::handle_token", ring_len = state.simhash_guard.len(),
            "SimHash semantic-loop watchdog fired (paraphrased sentence repeat)"
        );
    }
    state.loop_watchdog_triggered = true;
    state.stop_string_triggered = true;
    state.guard_stop = Some(if semantic {
        "simhash_semantic_loop"
    } else {
        "token_loop_watchdog"
    });
    state
        .cancel_flag
        .store(true, std::sync::atomic::Ordering::Release);
    true
}

/// 2026-09-26: Run the SimHash and token loop watchdogs over an already-sanitized
/// content chunk (both callers sanitize first). On a trip: name the guard, cancel, and
/// return `Some` with no deltas. Otherwise `Some` with one `Content` delta for non-empty
/// text (also kept in `refusal_scan_buf`, up to 16 KiB), or `None` for empty text.
pub(super) fn process_detector_content(
    state: &mut StreamState,
    ctx: &StreamCtx,
    sanitized_or_raw: &str,
) -> Option<DeltaVec> {
    let sanitized = sanitized_or_raw;

    // 2026-09-26: SimHash semantic-loop guard (`loop_simhash`), checked at each sentence
    // boundary or every 1024 pending bytes. One chunk at bigram Jaccard >= 0.55 against
    // any of the last 16 trips it and ends the stream, which repetitive structured
    // output can reach; `METRALE_SIMHASH_LOOP=0` turns it off.
    let semantic_trip = if simhash_loop_enabled() && !state.loop_watchdog_triggered {
        state.simhash_pending.push_str(sanitized);
        let mut dup = false;
        if crate::loop_simhash::ends_at_sentence_boundary(&state.simhash_pending).is_some()
            || state.simhash_pending.len() >= 1024
        {
            dup = state.simhash_guard.check(&state.simhash_pending);
            state.simhash_pending.clear();
        }
        if state.simhash_pending.len() > 4096 {
            let drop_to = state.simhash_pending.len() / 2;
            state.simhash_pending.drain(..drop_to);
        }
        dup
    } else {
        false
    };

    let token_trip = !watchdogs_disabled()
        && check_loop_watchdog(
            sanitized,
            &mut state.loop_scan_buf,
            state.loop_watchdog_triggered,
        );

    if (semantic_trip || token_trip)
        && apply_guard_trip(
            state,
            guard_honor_forced_len(),
            ctx.ignore_eos,
            ctx.min_tokens,
            semantic_trip,
        )
    {
        // 2026-09-26: A trip emits nothing further for this chunk.
        return Some(DeltaVec::new());
    }

    if !sanitized.is_empty() {
        if state.refusal_scan_buf.len() < 16_384 {
            state.refusal_scan_buf.push_str(sanitized);
        }
        let out: DeltaVec = vec![StreamDelta::Content {
            text: sanitized.to_string(),
            token_ids: state.take_ids_if(ctx.req_return_token_ids),
        }];
        return Some(out);
    }
    None
}

/// 2026-09-26: The detector branch's `Content(text)` arm: sanitize, then
/// `process_detector_content`.
pub(super) fn detector_content_arm(
    state: &mut StreamState,
    ctx: &StreamCtx,
    text: &str,
) -> Option<DeltaVec> {
    let sanitized = sanitize_content_chunk(
        text,
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &mut state.inside_envelope,
        &ctx.leak_markers,
    );
    process_detector_content(state, ctx, &sanitized)
}

#[cfg(test)]
#[path = "guard_forced_len_tests.rs"]
mod guard_forced_len_tests;
