// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Source-scanning tests for the scheduler's finish sites.
//!
//! A finish with no `a.guard_stop`, whose last token is neither an EOS nor
//! the tool-call end token, is reported by `derive_finish_reason` as
//! `"stop"` while budget remains. Two checks read the scheduler sources as
//! text:
//! - every non-exempt control-token hard stop (anchored on its
//!   `hard-stop fired` log line) sets `a.guard_stop`;
//! - per source, the number of `finished = true` code lines, and how many
//!   of them name a guard nearby, is pinned.
//!
//! `api/chat_stream/cancel_guard_tests.rs` covers the `cancel_flag` stores
//! in `api/chat_stream`; this file covers `finished` in the scheduler.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: Scheduler files carrying control-token hard stops.
const HARD_STOP_FILES: &[(&str, &str)] = &[
    (
        "decode_logits_step.rs + decode_logits_step/{host_sample,per_token,content_emit}.rs",
        concat!(
            include_str!("decode_logits_step.rs"),
            include_str!("decode_logits_step/host_sample.rs"),
            include_str!("decode_logits_step/per_token.rs"),
            include_str!("decode_logits_step/content_emit.rs")
        ),
    ),
    (
        "emit_step/{token,grammar_close,tool_param}.rs",
        concat!(
            include_str!("emit_step/token.rs"),
            include_str!("emit_step/grammar_close.rs"),
            include_str!("emit_step/tool_param.rs")
        ),
    ),
];

/// 2026-09-25: Text of the log line each hard-stop site emits next to its
/// `finished` assignment; the scan's anchor.
const HARD_STOP_ANCHOR: &str = "hard-stop fired";

/// 2026-09-25: Text that shows a site names its guard.
const NAMED: &str = "guard_stop = Some(";

/// 2026-09-25: Hard stops exempt from naming a guard.
///
/// `tokenizer_runtime.rs` adds `<|im_start|>` to `eos_tokens`, and the site
/// pushes the token, so `derive_finish_reason`'s EOS rung reports `"stop"`
/// whether or not a guard is named. `<tool_response>` is not added to
/// `eos_tokens`, so it needs a name.
///
/// Matched against the anchor line only, not the window, so a comment that
/// mentions `<|im_start|>` near another site's log line does not exempt
/// that site.
const INTENTIONAL_STOP: &[&str] = &["<|im_start|>"];

/// 2026-09-25: Lines searched before an anchor (`scan`), and on both sides
/// of a finish site (`scan_finish_ledger`).
const WINDOW: usize = 12;

fn scan(src: &str) -> (usize, Vec<usize>) {
    let lines: Vec<&str> = src.lines().collect();
    let mut sites = 0usize;
    let mut bare = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !line.contains(HARD_STOP_ANCHOR) {
            continue;
        }
        let from = i.saturating_sub(WINDOW);
        let window = lines[from..=i].join("\n");
        if !window.contains("finished = true") {
            // 2026-09-25: an anchor with no finish next to it is not a cut
            // site.
            continue;
        }
        if INTENTIONAL_STOP.iter().any(|t| line.contains(t)) {
            continue;
        }
        sites += 1;
        if !window.contains(NAMED) {
            bare.push(i + 1);
        }
    }
    (sites, bare)
}

#[test]
fn every_control_token_hard_stop_names_its_guard() {
    let mut total = 0usize;
    let mut bare: Vec<String> = Vec::new();
    for (name, src) in HARD_STOP_FILES {
        let (n, lines) = scan(src);
        total += n;
        bare.extend(lines.iter().map(|l| format!("{name}:{l}")));
    }

    // 2026-09-25: floor: the `<tool_response>` stop exists in both sources.
    // If the scan stops matching, this fails instead of passing on zero
    // sites.
    assert!(
        total >= 2,
        "found only {total} non-exempt hard-stop sites — the scan stopped \
         matching. Fix the detection before trusting a green here."
    );

    assert!(
        bare.is_empty(),
        "a control-token hard stop ended a turn without naming its guard at: \
         {bare:?}\n\
         An unnamed guard cut wires finish_reason \"stop\", claiming the model \
         finished when the server truncated it. Set \
         `a.guard_stop = Some(GUARD_STOP_*)` at the site, or add the token to \
         INTENTIONAL_STOP with the reason it is genuinely a natural end."
    );
}

#[test]
fn the_scan_flags_a_bare_stop_and_clears_a_named_one() {
    // 2026-09-25: negative half: the detector flags a bare site. Without
    // it, the test above would also pass on a scan that matches nothing.
    let bare = "\
        if tok == trs {\n\
        \x20   a.output_tokens.push(tok);\n\
        \x20   a.finished = true;\n\
        \x20   tracing::debug!(\"<tool_response> hard-stop fired (id={trs})\");\n\
        }\n";
    let (n, flagged) = scan(bare);
    assert_eq!(n, 1, "the site should be counted");
    assert_eq!(flagged, vec![4], "a bare hard stop must be flagged");

    // 2026-09-25: positive half: naming it clears the finding.
    let named = bare.replace(
        "    a.finished = true;",
        "    a.finished = true;\n    a.guard_stop = Some(GUARD_STOP_TOOL_RESPONSE);",
    );
    let (n, flagged) = scan(&named);
    assert_eq!(n, 1);
    assert!(flagged.is_empty(), "a named hard stop must not be flagged");
}

/// 2026-09-25: The finish-site ledger. `scan` sees only sites that log the
/// anchor; this counts every code line containing `finished = true`, and
/// how many have a `guard_stop = Some(` within +/-`WINDOW` lines. It does
/// not classify the unnamed sites: adding a site, or removing a name, moves
/// a count and fails the pin.
fn scan_finish_ledger(src: &str) -> (usize, usize) {
    // 2026-09-25: lines starting with `//` are blanked, so comment prose is
    // neither a site nor a name.
    let lines: Vec<&str> = src
        .lines()
        .map(|l| {
            if l.trim_start().starts_with("//") {
                ""
            } else {
                l
            }
        })
        .collect();
    let mut total = 0usize;
    let mut named = 0usize;
    for (i, line) in lines.iter().enumerate() {
        if !line.contains("finished = true") {
            continue;
        }
        total += 1;
        let from = i.saturating_sub(WINDOW);
        let to = (i + WINDOW).min(lines.len() - 1);
        if lines[from..=to].join("\n").contains(NAMED) {
            named += 1;
        }
    }
    (total, named)
}

#[test]
fn every_finish_site_is_a_recorded_decision() {
    // 2026-09-25: (file, source, total `finished = true` sites, sites
    // naming a guard).
    //
    // Named: decode_logits_step: `<tool_response>`, think-skip,
    // fuzzy_repetition. emit_step: `<tool_response>`, think-skip,
    // post-think cap, content loop, inter-tool prose, tool_envelope_stuck.
    // decode_logits_content: post-think cap, content loop, inter-tool prose.
    //
    // Unnamed: emit_step: cancel flag, `<|im_start|>`, EOS, failed stream
    // send, length stop, and (2026-09-29, A143 part B) `</tool_call>` with no
    // grammar and no tools declared, the speculative twin of decode's
    // unnamed site: a content-based natural stop, not a server cut.
    // decode_logits_step: post-completion tool-open cap, two failed stream
    // sends, `</tool_call>` with no grammar and no tools declared, EOS,
    // `remaining == 0`, `max_seq_len` ceiling, terminated grammar.
    const LEDGER: &[(&str, &str, usize, usize)] = &[
        (
            "decode_logits_step.rs + decode_logits_step/{host_sample,per_token,content_emit}.rs",
            concat!(
                include_str!("decode_logits_step.rs"),
                include_str!("decode_logits_step/host_sample.rs"),
                include_str!("decode_logits_step/per_token.rs"),
                include_str!("decode_logits_step/content_emit.rs")
            ),
            11,
            3,
        ),
        (
            "emit_step/{token,grammar_close,tool_param}.rs",
            concat!(
                include_str!("emit_step/token.rs"),
                include_str!("emit_step/grammar_close.rs"),
                include_str!("emit_step/tool_param.rs")
            ),
            12,
            6,
        ),
        (
            "decode_logits_content.rs",
            include_str!("decode_logits_content.rs"),
            3,
            3,
        ),
    ];
    for (name, src, want_total, want_named) in LEDGER {
        let (total, named) = scan_finish_ledger(src);
        assert_eq!(
            (total, named),
            (*want_total, *want_named),
            "{name}: finish-site ledger drift — found {total} sites / {named} named, \
             pinned {want_total} / {want_named}. If you ADDED a site: a cut the \
             SERVER decides (watchdog, control token, degeneration) must set \
             `a.guard_stop = Some(GUARD_STOP_*)` at the site — unnamed it wires \
             finish_reason \"stop\" and agentic clients treat the truncation as a \
             finished turn. A NATURAL stop (model EOS / budget / client cancel) \
             updates this pin AND the rationale comment above it. If a count \
             DROPPED, a site or its guard name was removed — verify that was the \
             intent."
        );
    }
}

#[test]
fn the_ledger_counts_sites_credits_naming_and_ignores_prose() {
    // 2026-09-25: a bare site counts unnamed, a named site is credited, and
    // comment prose is not a site.
    let bare = "\
        if a.think_skip_count >= 50 {\n\
        \x20   a.finished = true;\n\
        }\n";
    assert_eq!(
        scan_finish_ledger(bare),
        (1, 0),
        "a bare finish site is 1 site, 0 named"
    );

    let named = "\
        if a.think_skip_count >= 50 {\n\
        \x20   a.finished = true;\n\
        \x20   a.guard_stop = Some(GUARD_STOP_THINK_SKIP);\n\
        }\n";
    assert_eq!(
        scan_finish_ledger(named),
        (1, 1),
        "a guard named in the window must be credited"
    );

    let prose = "// Previously we set `a.finished = true` here -- a\n";
    assert_eq!(
        scan_finish_ledger(prose),
        (0, 0),
        "prose mentioning the mutation is not a cut site"
    );
}

#[test]
fn the_intentional_stop_exemption_is_load_bearing_and_narrow() {
    // 2026-09-25: `<|im_start|>` is exempt.
    let im_start = "\
        if tok == ims {\n\
        \x20   a.output_tokens.push(tok);\n\
        \x20   a.finished = true;\n\
        \x20   tracing::debug!(\"<|im_start|> hard-stop fired (id={ims})\");\n\
        }\n";
    let (n, flagged) = scan(im_start);
    assert_eq!(n, 0, "<|im_start|> must be exempt — it is eos-registered");
    assert!(flagged.is_empty());

    // 2026-09-25: the exemption must not cover a neighbouring bare site.
    // The two are more than WINDOW lines apart, so the exempt site's text
    // is outside the other's window.
    let gap = "\n".repeat(WINDOW + 2);
    let both = format!(
        "{im_start}{gap}if tok == trs {{\n    a.finished = true;\n    \
         tracing::debug!(\"<tool_response> hard-stop fired\");\n}}\n"
    );
    let (n, flagged) = scan(&both);
    assert_eq!(n, 1, "the <tool_response> site must still be counted");
    assert_eq!(flagged.len(), 1, "and still flagged when bare");
}
