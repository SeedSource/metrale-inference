// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Guards on the outbound text of a chat stream; none of them changes the
//! prompt.
//!
//! - [`bump_f12_tool_call_count`]: counts tool calls and sets the stop flag past the cap.
//! - [`check_loop_watchdog`]: reports a repeated line or phrase in streamed content.
//! - [`flush_content_sanitizer`]: drains the content sanitizer's held-back tail, at the
//!   end of the stream and before each tool-call event.
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

use crate::tool_parser;

/// 2026-09-26: Add one to `count`; when `count` exceeds `max` and `stop` is not yet set,
/// set it and log a warning. The chat stream passes `max` from
/// `METRALE_MAX_TOOL_CALLS_PER_RESPONSE`, 12 when unset or unparsable
/// (`chat_stream/mod.rs`).
pub fn bump_f12_tool_call_count(count: &mut usize, max: usize, stop: &mut bool) {
    *count += 1;
    if *count > max && !*stop {
        tracing::warn!(
            emitted = *count,
            max,
            "tool-call cap reached; ending response"
        );
        *stop = true;
    }
}

/// 2026-10-14: Lines (`p == 1`) a single repeated line needs, with the default when
/// `METRALE_LOOP_LINE_REPEATS_SINGLE` is unset or not an integer >= 2.
const DEFAULT_LINE_REPEATS_SINGLE: usize = 5;
/// 2026-10-14: Block repeats a block of 2 or more lines needs, with the default when
/// `METRALE_LOOP_LINE_REPEATS` is unset or not an integer >= 2.
const DEFAULT_LINE_REPEATS: usize = 3;
/// 2026-10-14: Longest block, in non-blank lines, the periodic-tail rule looks for.
const LINE_MAX_PERIOD: usize = 32;
/// 2026-10-14: Alphanumeric chars a repeating block must carry; box-drawing, punctuation
/// and whitespace count for nothing, so diagram borders and `}` runs never qualify.
const LINE_MIN_BLOCK_ALNUM: usize = 16;
/// 2026-10-14: Phrase rule (one long line, no newline): shortest period in bytes that may
/// fire. A tail that is periodic at a shorter period (`0, 0, 0`, `na na`, `=-=-`) is
/// padding or a short token run, not a phrase loop, and never fires.
const PHRASE_MIN_PERIOD: usize = 8;
/// 2026-10-14: Longest phrase period, in bytes.
const PHRASE_MAX_PERIOD: usize = 512;
/// 2026-10-14: Copies of the phrase the tail must hold (the compared span is at least
/// `PHRASE_REPEATS * q` bytes).
const PHRASE_REPEATS: usize = 3;
/// 2026-10-14: Shortest compared span in bytes, so a short period also needs a long run.
const PHRASE_MIN_SPAN: usize = 96;
/// 2026-10-14: Percent of the span's bytes that must equal the byte `q` earlier
/// (ASCII case folded). 97, not 95: a one-line JSON array of objects that differ in one
/// digit per ~24 bytes matches at ~96%.
const PHRASE_MATCH_PCT: usize = 97;
/// 2026-10-14: Alphanumeric chars the compared span must carry.
const PHRASE_MIN_SPAN_ALNUM: usize = 48;

/// 2026-10-14: Read once per process: `METRALE_LOOP_LINE_REPEATS` (block of 2 or more
/// lines) and `METRALE_LOOP_LINE_REPEATS_SINGLE` (one line); an unset, unparsable or
/// below-2 value keeps the default.
fn line_repeats() -> (usize, usize) {
    static R: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let read = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&n| n >= 2)
                .unwrap_or(default)
        };
        (
            read("METRALE_LOOP_LINE_REPEATS", DEFAULT_LINE_REPEATS),
            read(
                "METRALE_LOOP_LINE_REPEATS_SINGLE",
                DEFAULT_LINE_REPEATS_SINGLE,
            ),
        )
    })
}

/// 2026-09-26: Trimmed, ASCII-lowercased, whitespace runs collapsed to one space.
fn norm_line(s: &str) -> String {
    let lowered = s.trim().to_ascii_lowercase();
    let mut out = String::with_capacity(lowered.len());
    let mut prev_space = false;
    for ch in lowered.chars() {
        if ch.is_ascii_whitespace() {
            if !prev_space && !out.is_empty() {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

fn alnum_chars(s: &str) -> usize {
    s.chars().filter(|c| c.is_alphanumeric()).count()
}

/// 2026-10-14: Periodic-tail rule over the complete lines of `complete` (blank lines
/// dropped, the rest normalised by `norm_line`). `Some((p, repeats))` for the smallest
/// period `p <= LINE_MAX_PERIOD` such that the newest `repeats * p` lines satisfy
/// `line[i] == line[i - p]`, with `repeats` = `r_single` for `p == 1` and `r_block`
/// otherwise, and the newest block carries `LINE_MIN_BLOCK_ALNUM` alphanumeric chars.
pub(crate) fn periodic_line_tail(
    complete: &str,
    r_block: usize,
    r_single: usize,
) -> Option<(usize, usize)> {
    let max_lines = (LINE_MAX_PERIOD * r_block).max(r_single);
    let mut lines: Vec<String> = complete
        .lines()
        .rev()
        .map(norm_line)
        .filter(|l| !l.is_empty())
        .take(max_lines)
        .collect();
    lines.reverse();
    let n = lines.len();
    for p in 1..=LINE_MAX_PERIOD {
        let repeats = if p == 1 { r_single } else { r_block };
        let need = repeats * p;
        if need > n {
            if p > 1 {
                break;
            }
            continue;
        }
        let periodic = (n - need + p..n).all(|i| lines[i] == lines[i - p]);
        if periodic
            && lines[n - p..].iter().map(|l| alnum_chars(l)).sum::<usize>() >= LINE_MIN_BLOCK_ALNUM
        {
            return Some((p, repeats));
        }
    }
    None
}

/// 2026-10-14: Phrase rule over `line`, the unfinished last line: `Some(q)` when the
/// smallest period `q` (bytes, `1..=PHRASE_MAX_PERIOD`) at which the newest
/// `max(PHRASE_REPEATS * q, PHRASE_MIN_SPAN)` bytes match the byte `q` earlier in at
/// least `PHRASE_MATCH_PCT` percent of places is `>= PHRASE_MIN_PERIOD`, and that span
/// carries `PHRASE_MIN_SPAN_ALNUM` alphanumeric chars. A smaller smallest period (zero
/// arrays, separator runs) returns `None`.
pub(crate) fn periodic_phrase_tail(line: &str) -> Option<usize> {
    let b = line.as_bytes();
    let n = b.len();
    if n < PHRASE_MIN_SPAN {
        return None;
    }
    for q in 1..=PHRASE_MAX_PERIOD {
        let span = (PHRASE_REPEATS * q).max(PHRASE_MIN_SPAN);
        if span > n {
            break;
        }
        let compared = span - q;
        let allowed = compared * (100 - PHRASE_MATCH_PCT) / 100;
        let mut miss = 0usize;
        for i in n - compared..n {
            if !b[i].eq_ignore_ascii_case(&b[i - q]) {
                miss += 1;
                if miss > allowed {
                    break;
                }
            }
        }
        if miss > allowed {
            continue;
        }
        if q < PHRASE_MIN_PERIOD {
            return None;
        }
        let mut start = n - span;
        while !line.is_char_boundary(start) {
            start += 1;
        }
        return (alnum_chars(&line[start..]) >= PHRASE_MIN_SPAN_ALNUM).then_some(q);
    }
    None
}

/// 2026-10-14 (was: 4 equal lines, or 4 copies of a 30-byte needle, anywhere in the
/// buffer; that cut ordinary code, markdown and diagram answers, #82): append `text` to
/// `loop_scan_buf` and report a repetition loop. Once the buffer passes 10,240 bytes it is
/// cut back to its last 8,192 bytes at most. A loop is a periodic tail, not a repeat
/// anywhere: legit structure repeats a line or a template with new content in between.
/// - Line rule (`periodic_line_tail`), when `text` holds a newline: the newest complete
///   lines are a block of `p` lines repeated back to back, 3 times for `p >= 2`
///   (`METRALE_LOOP_LINE_REPEATS`) and 5 times for one line
///   (`METRALE_LOOP_LINE_REPEATS_SINGLE`), the block carrying 16 alphanumeric chars.
/// - Phrase rule (`periodic_phrase_tail`): the unfinished last line ends in a phrase of 8
///   to 512 bytes repeated 3 times at 97% byte match over at least 96 bytes.
///
/// Returns false, without touching the buffer, when `already_triggered` or `text` is
/// empty.
pub fn check_loop_watchdog(
    text: &str,
    loop_scan_buf: &mut String,
    already_triggered: bool,
) -> bool {
    if already_triggered || text.is_empty() {
        return false;
    }
    loop_scan_buf.push_str(text);
    if loop_scan_buf.len() > 10_240 {
        let drop = loop_scan_buf.len() - 8_192;
        let cut = loop_scan_buf
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= drop)
            .unwrap_or(drop);
        loop_scan_buf.drain(..cut);
    }
    let last_nl = loop_scan_buf.rfind('\n');
    if let Some(nl) = last_nl.filter(|_| text.contains('\n')) {
        let (r_block, r_single) = line_repeats();
        if let Some((period, repeats)) = periodic_line_tail(&loop_scan_buf[..nl], r_block, r_single)
        {
            tracing::warn!(
                period,
                repeats,
                "loop watchdog fired — periodic tail of repeated lines in post-detector content"
            );
            return true;
        }
    }
    let partial = match last_nl {
        Some(nl) => &loop_scan_buf[nl + 1..],
        None => loop_scan_buf.as_str(),
    };
    if let Some(period) = periodic_phrase_tail(partial) {
        tracing::warn!(
            period,
            "loop watchdog fired — periodic phrase in one line of post-detector content"
        );
        return true;
    }
    false
}

/// 2026-09-26: Take the sanitizer's held-back tail and return what may still be emitted.
/// When suppression is active the tail is dropped and the flag cleared. Otherwise
/// complete tags are scrubbed (unless the markers declare envelopes), a trailing partial
/// close marker is cut, and the rest is returned, unless it is a lone partial tag: it
/// starts with `<`, holds no whitespace after trimming its end, and is shorter than the
/// longest `orphan_open`/`close` marker.
pub fn flush_content_sanitizer(
    tag_scan_buf: &mut String,
    suppressing_param_leak: &mut bool,
    markers: &tool_parser::LeakMarkers,
) -> String {
    if *suppressing_param_leak {
        tag_scan_buf.clear();
        *suppressing_param_leak = false;
        return String::new();
    }
    if tag_scan_buf.is_empty() {
        return String::new();
    }
    let tag_max: usize = markers
        .orphan_open
        .iter()
        .chain(markers.close.iter())
        .map(|t| t.len())
        .max()
        .unwrap_or(0);
    let mut final_text = std::mem::take(tag_scan_buf);
    // 2026-09-26: Remove complete tool-call tags from the tail. Skipped when the markers
    // declare envelopes: `sanitize_content_chunk` passes the tags inside an envelope
    // through as content, and the scrub would remove them.
    if markers.envelope_open.is_empty() {
        final_text = super::scrub::scrub_tool_tags(&final_text, markers);
    }
    // 2026-09-26: Cut a trailing partial close marker: when the text from the last `<` to
    // the end is a strict prefix of a `close` marker, drop it and keep the text before
    // it. The partial-tag check below only catches a tail that starts with `<` and holds
    // no whitespace, so it misses a partial close after other text, such as `\n\n</_call`.
    if let Some(lt) = final_text.rfind('<') {
        let suffix = &final_text[lt..];
        let is_partial_close = markers
            .close
            .iter()
            .any(|t| t.len() > suffix.len() && t.starts_with(suffix));
        if is_partial_close {
            final_text.truncate(lt);
        }
    }
    let looks_like_partial_tag = {
        let t = final_text.trim_end();
        tag_max > 0 && t.starts_with('<') && !t.contains(char::is_whitespace) && t.len() < tag_max
    };
    if looks_like_partial_tag {
        String::new()
    } else {
        final_text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f12_under_cap_does_not_stop() {
        let (mut count, mut stop) = (0usize, false);
        bump_f12_tool_call_count(&mut count, 12, &mut stop);
        assert_eq!(count, 1);
        assert!(!stop);
    }

    #[test]
    fn f12_at_cap_does_not_stop() {
        let (mut count, mut stop) = (11usize, false);
        bump_f12_tool_call_count(&mut count, 12, &mut stop);
        assert_eq!(count, 12);
        assert!(!stop);
    }

    #[test]
    fn f12_over_cap_trips_stop() {
        let (mut count, mut stop) = (12usize, false);
        bump_f12_tool_call_count(&mut count, 12, &mut stop);
        assert_eq!(count, 13);
        assert!(stop);
    }

    #[test]
    fn watchdog_already_triggered_returns_false() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog("anything", &mut buf, true));
    }

    #[test]
    fn watchdog_empty_text_returns_false() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog("", &mut buf, false));
    }

    #[test]
    fn watchdog_five_identical_lines_fire_four_do_not() {
        // 2026-10-14: one repeated line (p == 1) needs 5 back-to-back copies.
        let mut buf = String::new();
        let line = "Running cargo test on the project\n";
        for _ in 0..4 {
            assert!(!check_loop_watchdog(line, &mut buf, false));
        }
        assert!(check_loop_watchdog(line, &mut buf, false));
    }

    #[test]
    fn periodic_line_tail_needs_back_to_back_blocks() {
        let block = "alpha beta gamma delta\nepsilon zeta eta theta\n";
        assert_eq!(periodic_line_tail(&block.repeat(3), 3, 5), Some((2, 3)));
        assert_eq!(periodic_line_tail(&block.repeat(2), 3, 5), None);
        // A different line between the copies breaks the period.
        let mut s = String::new();
        for i in 0..6 {
            s.push_str(block);
            s.push_str(&format!("note number {i}\n"));
        }
        assert_eq!(periodic_line_tail(&s, 3, 5), None);
        // Blank lines do not count, so a paragraph loop has period 1.
        let para = "This exact paragraph keeps coming back again.\n\n";
        assert_eq!(periodic_line_tail(&para.repeat(5), 3, 5), Some((1, 5)));
        assert_eq!(periodic_line_tail(&para.repeat(4), 3, 5), None);
    }

    #[test]
    fn periodic_line_tail_ignores_content_free_blocks() {
        let rows = "│                    │\n├────────────────────┤\n";
        assert_eq!(periodic_line_tail(&rows.repeat(10), 3, 5), None);
        assert_eq!(periodic_line_tail(&"}\n".repeat(40), 3, 5), None);
        assert_eq!(periodic_line_tail(&"| -- | -- |\n".repeat(20), 3, 5), None);
    }

    #[test]
    fn periodic_phrase_tail_finds_a_phrase_loop_in_one_line() {
        let phrase = "and then the answer goes back to the start ";
        let line = format!("Intro text. {}", phrase.repeat(3));
        assert_eq!(periodic_phrase_tail(&line), Some(phrase.len()));
        let two = format!("Intro text. {}", phrase.repeat(2));
        assert_eq!(periodic_phrase_tail(&two), None);
    }

    #[test]
    fn periodic_phrase_tail_skips_short_periods_and_prose() {
        let zeros = format!("let a = [{}];", "0, ".repeat(80));
        assert_eq!(periodic_phrase_tail(&zeros), None);
        assert_eq!(periodic_phrase_tail(&"=-".repeat(100)), None);
        let prose = "The quick brown fox jumps over the lazy dog while the band plays \
                     a slow tune, and nobody in the room notices the clock on the wall \
                     ticking past midnight as the rain keeps falling outside.";
        assert_eq!(periodic_phrase_tail(prose), None);
        let mut json = String::from("[");
        for i in 0..9 {
            json.push_str(&format!("{{\"type\":\"point\",\"x\":{i}}},"));
        }
        assert_eq!(
            periodic_phrase_tail(&json),
            None,
            "one digit in 24 bytes differs"
        );
    }

    #[test]
    fn watchdog_buffer_caps_at_10kb() {
        let mut buf = String::new();
        let big = "x".repeat(5000);
        check_loop_watchdog(&big, &mut buf, false);
        check_loop_watchdog(&big, &mut buf, false);
        check_loop_watchdog(&big, &mut buf, false);
        assert!(
            buf.len() <= 10_240,
            "buffer should self-trim, got {}",
            buf.len()
        );
    }
}
