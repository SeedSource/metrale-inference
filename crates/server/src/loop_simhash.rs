// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Sentence-level loop guard for the chat stream. A sentence is a
//! repeat when its set of word-bigram hashes has a Jaccard similarity at or
//! above the threshold with one of the last `cap` sentences stored. Since
//! 2026-10-14 (#82 guard fix) the guard trips only on a sustained run: `run`
//! (default 3, `METRALE_SIMHASH_RUN`) checked sentences in a row that are each
//! a repeat that adds no word bigram absent from the ring. One matching
//! sentence (a code line beside a similar one, a templated heading) no longer
//! ends the stream, nor does a list whose items share a template but each name
//! something new: a loop's vocabulary closes, a list's keeps growing. Despite
//! the type's name, no SimHash is computed.
//!
//! Owner: server chat stream.
//! Invariants: the ring holds at most `cap` sets, and `cap` is at least 1.
//!
//! Before hashing, a sentence is lowercased and every byte that is not an
//! ASCII letter or digit is dropped, except that runs of spaces, tabs and
//! newlines become one space; case and punctuation variants therefore
//! collide.

use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

const DEFAULT_RING_CAP: usize = 16;
const DEFAULT_JACCARD_THRESHOLD_PCT: u32 = 55;
const WORD_SHINGLE_LEN: usize = 2;
const MIN_NORMALIZED_LEN: usize = 24;
const MIN_WORDS: usize = 3;
/// 2026-10-14: Consecutive repeat sentences that trip the guard, when
/// `METRALE_SIMHASH_RUN` is unset or not an integer >= 1.
const DEFAULT_RUN: usize = 3;
/// 2026-10-14: A sentence with fewer word bigrams than this is compared at
/// `SHORT_THRESHOLD_PCT` (when that is above the guard's threshold): a 3-bigram
/// set reaches 55% by sharing 2 bigrams.
const SHORT_BIGRAMS: usize = 6;
const SHORT_THRESHOLD_PCT: u32 = 75;

/// 2026-10-14: Read once per process: `METRALE_SIMHASH_RUN`, the consecutive
/// repeat sentences that trip the guard (default `DEFAULT_RUN`; an unset,
/// unparsable or zero value keeps the default).
fn default_run() -> usize {
    static RUN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *RUN.get_or_init(|| {
        std::env::var("METRALE_SIMHASH_RUN")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(DEFAULT_RUN)
    })
}

/// 2026-09-26: The word-bigram sets of the last `cap` sentences stored.
#[derive(Debug)]
pub struct SimHashLoopGuard {
    ring: VecDeque<BTreeSet<u64>>,
    cap: usize,
    threshold_pct: u32,
    /// 2026-10-14: Repeat sentences in a row needed to trip.
    run: usize,
    /// 2026-10-14: Repeat sentences in a row so far.
    streak: usize,
}

impl Default for SimHashLoopGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl SimHashLoopGuard {
    /// 2026-09-26: `DEFAULT_RING_CAP` sentences and a threshold of
    /// `DEFAULT_JACCARD_THRESHOLD_PCT` percent.
    pub fn new() -> Self {
        Self::with_params(DEFAULT_RING_CAP, DEFAULT_JACCARD_THRESHOLD_PCT)
    }

    /// 2026-09-26: A ring of `cap` sets (at least 1) and a Jaccard threshold
    /// in whole percent (55 means 0.55); the run length is the process default.
    pub fn with_params(cap: usize, threshold_pct: u32) -> Self {
        Self::with_run(cap, threshold_pct, default_run())
    }

    /// 2026-10-14: As `with_params`, tripping after `run` (at least 1)
    /// consecutive repeat sentences.
    pub fn with_run(cap: usize, threshold_pct: u32, run: usize) -> Self {
        Self {
            ring: VecDeque::with_capacity(cap.max(1)),
            cap: cap.max(1),
            threshold_pct,
            run: run.max(1),
            streak: 0,
        }
    }

    /// 2026-10-14: True when this sentence completes a run of `run` checked
    /// sentences in a row that each reach the threshold (`jaccard_pct`; at
    /// least `SHORT_THRESHOLD_PCT` for a set under `SHORT_BIGRAMS` bigrams)
    /// against some set in the ring and whose bigrams all appear somewhere in
    /// the ring. Any other checked sentence ends the run. The set then enters the ring, evicting the oldest one past `cap`. A
    /// sentence shorter than `MIN_NORMALIZED_LEN` bytes once normalized, or
    /// with fewer than `MIN_WORDS` words, returns false, is not stored and
    /// leaves the run as it was.
    pub fn check(&mut self, sentence: &str) -> bool {
        let normalized = normalize(sentence);
        if normalized.len() < MIN_NORMALIZED_LEN {
            return false;
        }
        let shingles = bigram_set(&normalized);
        if shingles.is_empty() {
            return false;
        }
        let threshold = if shingles.len() < SHORT_BIGRAMS {
            self.threshold_pct.max(SHORT_THRESHOLD_PCT)
        } else {
            self.threshold_pct
        };
        let dup = self
            .ring
            .iter()
            .any(|prev| jaccard_pct(&shingles, prev) >= threshold);
        let closed = dup
            && shingles
                .iter()
                .all(|h| self.ring.iter().any(|prev| prev.contains(h)));
        self.streak = if closed { self.streak + 1 } else { 0 };
        if self.ring.len() >= self.cap {
            self.ring.pop_front();
        }
        self.ring.push_back(shingles);
        self.streak >= self.run
    }

    pub fn reset(&mut self) {
        self.ring.clear();
        self.streak = 0;
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

/// 2026-09-26: The normalisation described in the module docs, with no
/// leading or trailing space.
fn normalize(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut last_was_space = true;
    for &b in s.as_bytes() {
        let c = b.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_was_space = false;
        } else if (c == b' ' || c == b'\t' || c == b'\n' || c == b'\r') && !last_was_space {
            out.push(b' ');
            last_was_space = true;
        }
    }
    if out.last() == Some(&b' ') {
        out.pop();
    }
    out
}

/// 2026-09-26: The set of `DefaultHasher` hashes of each pair of adjacent
/// words joined by one space; empty below `MIN_WORDS` words.
fn bigram_set(bytes: &[u8]) -> BTreeSet<u64> {
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return BTreeSet::new(),
    };
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < MIN_WORDS {
        return BTreeSet::new();
    }
    let mut set = BTreeSet::new();
    for window in words.windows(WORD_SHINGLE_LEN) {
        let mut h = DefaultHasher::new();
        for (i, w) in window.iter().enumerate() {
            if i > 0 {
                h.write_u8(b' ');
            }
            h.write(w.as_bytes());
        }
        set.insert(h.finish());
    }
    set
}

/// 2026-09-26: `|A ∩ B| × 100 / |A ∪ B|`, rounded down; 0 when either set is
/// empty.
fn jaccard_pct(a: &BTreeSet<u64>, b: &BTreeSet<u64>) -> u32 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let inter = a.intersection(b).count() as u64;
    let union = (a.len() + b.len()) as u64 - inter;
    if union == 0 {
        return 0;
    }
    ((inter * 100) / union) as u32
}

/// 2026-09-26: `Some(buffer.len())` when the trailing spaces, tabs and
/// newlines hold at least two newlines, or when the text before them ends
/// with `.`, `!`, `?`, `:` or a closing triple-backtick fence; `None`
/// otherwise. The chat stream
/// (`api/chat_stream/handle_token/detector_content.rs`) passes its pending
/// text to `check` when this returns `Some`.
pub fn ends_at_sentence_boundary(buffer: &str) -> Option<usize> {
    let bytes = buffer.as_bytes();
    let n = bytes.len();
    if n == 0 {
        return None;
    }
    let mut tail = n;
    let mut newlines = 0u32;
    while tail > 0 {
        let b = bytes[tail - 1];
        if b == b'\n' {
            newlines += 1;
            tail -= 1;
        } else if b == b' ' || b == b'\t' || b == b'\r' {
            tail -= 1;
        } else {
            break;
        }
    }
    if newlines >= 2 {
        return Some(n);
    }
    // 2026-09-26: A closing fence ends a unit too, so a code block that ends
    // without `.!?` or a blank line reaches `check` before the chat stream's
    // 1024-byte flush.
    if tail >= 3 && &bytes[tail - 3..tail] == b"```" {
        return Some(n);
    }
    if tail == 0 {
        return None;
    }
    let last = bytes[tail - 1];
    // 2026-09-26: `:` ends a sentence as well; `check` ignores short ones
    // (`MIN_NORMALIZED_LEN`, `MIN_WORDS`), so a heading such as `Steps:`
    // cannot trip the guard.
    if last == b'.' || last == b'!' || last == b'?' || last == b':' {
        Some(n)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paraphrase_with_case_change_triggers_on_third_repeat() {
        let s1 = "I'll create a proper Rust Axum server with the /echo endpoint.";
        let s2 = "I'll create a proper Rust axum server with an /echo endpoint and tests.";
        let s3 = "I'll create a proper Rust axum server with the /echo endpoint.";
        // 2026-10-14 (#82 guard fix): the guard trips on the third repeat in a row
        // (`DEFAULT_RUN`) whose words are all already in the ring. s2 is a repeat but
        // brings new bigrams (`with an`, `and tests`), so it does not count; s3, s4 and
        // s5 do, and s5 trips. Before the fix s2 tripped.
        let s4 = "I'll create a proper Rust Axum server with an /echo endpoint.";
        let s5 = "I'll create a proper Rust axum server with the /echo endpoint and tests.";
        let mut g = SimHashLoopGuard::with_run(DEFAULT_RING_CAP, DEFAULT_JACCARD_THRESHOLD_PCT, 3);
        assert!(!g.check(s1), "first sentence is novel");
        assert!(
            !g.check(s2),
            "second sentence (case+wording paraphrase) is a repeat with new words"
        );
        assert!(!g.check(s3), "third sentence: closed repeat, run 1");
        assert!(!g.check(s4), "fourth sentence: closed repeat, run 2");
        assert!(g.check(s5), "fifth sentence: closed repeat, run 3, trips");
    }

    #[test]
    fn a_templated_list_that_names_something_new_each_item_does_not_trip() {
        let mut g = SimHashLoopGuard::with_run(DEFAULT_RING_CAP, DEFAULT_JACCARD_THRESHOLD_PCT, 3);
        for svc in [
            "nginx",
            "postgres",
            "redis",
            "kafka",
            "grafana",
            "prometheus",
            "jenkins",
            "rabbitmq",
            "elasticsearch",
            "memcached",
            "haproxy",
            "vault",
        ] {
            let item =
                format!("- Restart the {svc} service and verify that it comes back up cleanly.");
            assert!(!g.check(&item), "{item}");
        }
    }

    #[test]
    fn a_novel_sentence_ends_the_run() {
        let mut g = SimHashLoopGuard::with_run(DEFAULT_RING_CAP, DEFAULT_JACCARD_THRESHOLD_PCT, 3);
        let s = "I will create the Rust axum server with proper tests.";
        assert!(!g.check(s));
        assert!(!g.check(s));
        assert!(!g.check(s));
        assert!(!g.check("Next we wire the router into main and bind the listener."));
        assert!(!g.check(s), "run restarts at 1 after a novel sentence");
        assert!(!g.check(s));
        assert!(g.check(s));
    }

    #[test]
    fn short_sentences_need_the_short_threshold() {
        // 2026-10-14: 4-bigram sets sharing 3 bigrams are 60% similar: a repeat at
        // 55%, not at the 75% a set under `SHORT_BIGRAMS` bigrams needs.
        let mut g = SimHashLoopGuard::with_run(DEFAULT_RING_CAP, DEFAULT_JACCARD_THRESHOLD_PCT, 1);
        assert!(!g.check("elif scale_setting matches the Standard:"));
        assert!(!g.check("elif scale_setting matches the Premium:"));
        assert!(!g.check("elif scale_setting matches the Economy:"));
        assert!(
            g.check("elif scale_setting matches the Economy:"),
            "an exact repeat still counts"
        );
    }

    #[test]
    fn legitimate_distinct_prose_does_not_trigger() {
        let mut g = SimHashLoopGuard::new();
        let sentences = [
            "First, we set up the Rust workspace and add axum as a dependency.",
            "Then we define a handler function that echoes the request body.",
            "After that we wire the handler into the router at /echo.",
            "Finally we add an integration test using tower's ServiceExt.",
            "Run the test suite with cargo test to confirm everything passes.",
        ];
        let mut any_dup = false;
        for s in sentences {
            if g.check(s) {
                any_dup = true;
            }
        }
        assert!(!any_dup, "five distinct prose sentences must not collide");
    }

    #[test]
    fn short_sentences_do_not_trigger() {
        let mut g = SimHashLoopGuard::new();
        for _ in 0..10 {
            assert!(!g.check("Yes."));
            assert!(!g.check("Done."));
        }
    }

    #[test]
    fn enumeration_with_template_does_not_trigger_immediately() {
        let mut g = SimHashLoopGuard::new();
        assert!(!g.check("Test 1: verify the echo handler returns the body unchanged."));
        assert!(!g.check("Test 2: verify the router rejects malformed POST bodies."));
        assert!(!g.check("Test 3: verify the server gracefully shuts down on signal."));
    }

    #[test]
    fn reset_clears_ring() {
        let mut g = SimHashLoopGuard::new();
        let s = "I'll create a proper Rust Axum server with the /echo endpoint.";
        g.check(s);
        assert_eq!(g.len(), 1);
        g.reset();
        assert_eq!(g.len(), 0);
        assert!(!g.check(s), "after reset the same sentence is novel again");
    }

    #[test]
    fn boundary_detection_period_then_space() {
        assert!(ends_at_sentence_boundary("Hello world. ").is_some());
        assert!(ends_at_sentence_boundary("Hello world! ").is_some());
        assert!(ends_at_sentence_boundary("Hello world? ").is_some());
        assert!(ends_at_sentence_boundary("Hello world").is_none());
        assert!(ends_at_sentence_boundary("Hello world.").is_some());
        assert!(ends_at_sentence_boundary("Hello world,").is_none());
    }

    #[test]
    fn f47_colon_is_sentence_boundary() {
        assert!(
            ends_at_sentence_boundary(
                "Let me try a different approach - let me use the cargo bin directly:"
            )
            .is_some()
        );
        assert!(ends_at_sentence_boundary("Step one:").is_some());
        assert!(ends_at_sentence_boundary("Step one: ").is_some());
    }

    #[test]
    fn f47_cc_session_23x_phrase_loop_trips() {
        let mut g = SimHashLoopGuard::with_run(DEFAULT_RING_CAP, DEFAULT_JACCARD_THRESHOLD_PCT, 3);
        let s =
            "I see the issue. Let me try a different approach - let me use the cargo bin directly:";
        assert!(!g.check(s), "first emit must be novel");
        assert!(!g.check(s), "first repeat is run 1");
        assert!(!g.check(s), "second repeat is run 2");
        assert!(g.check(s), "third repeat in a row must trip");
    }

    #[test]
    fn boundary_detection_double_newline() {
        assert!(ends_at_sentence_boundary("paragraph\n\n").is_some());
        assert!(ends_at_sentence_boundary("paragraph\n").is_none());
    }

    #[test]
    fn ring_capacity_is_respected() {
        let mut g = SimHashLoopGuard::with_params(4, 55);
        for i in 0..10 {
            g.check(&format!(
                "this is sentence number {} with sufficiently long content for hashing",
                i
            ));
        }
        assert_eq!(g.len(), 4, "ring length capped at cap=4");
    }

    #[test]
    fn identical_sentences_trip_on_the_third_repeat() {
        let mut g = SimHashLoopGuard::with_run(DEFAULT_RING_CAP, DEFAULT_JACCARD_THRESHOLD_PCT, 3);
        let s = "I will create the Rust axum server with proper tests.";
        assert!(!g.check(s), "first emit is novel");
        assert!(
            !g.check(s),
            "one exact repeat no longer trips (#82 guard fix)"
        );
        assert!(!g.check(s));
        assert!(g.check(s), "third exact repeat in a row trips");
    }

    #[test]
    fn related_topics_with_distinct_action_do_not_trip() {
        let mut g = SimHashLoopGuard::new();
        assert!(!g.check("The axum router accepts incoming HTTP requests."));
        assert!(!g.check("The axum handler returns a JSON response body."));
        assert!(!g.check("The axum extractor parses the query parameters."));
    }
}
