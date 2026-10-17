// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17 (A200): Source scan of `impl_draft.rs`: every function that appends to the
//! DFlash ctx accumulator (`ctx_hidden_acc`) calls the shared slide
//! (`make_ctx_room` / `DflashProposerState::make_room_for`) before its first row copy. The
//! three legacy appends wrote at `ctx_len` with no bound before 2026-10-17, and a K=2 accept
//! at a full window wrote past the allocation (fp4lbdf: both ranks hung). The slide itself is
//! tested on `MockGpuBackend` in model-arch (`dflash_head/ctx_slide_tests.rs`).
//!
//! Owner: model-engine.

const SRC: &str = include_str!("impl_draft.rs");

/// 2026-10-17 (A200): The body of `fn <name>(` up to the next `fn ` at the same indent.
fn body(name: &str) -> &'static str {
    let start = SRC
        .find(&format!("    fn {name}("))
        .unwrap_or_else(|| panic!("fn {name} not found in impl_draft.rs"));
    let rest = &SRC[start + 4..];
    let end = rest.find("\n    fn ").map_or(rest.len(), |e| e + 1);
    &SRC[start..start + 4 + end]
}

#[test]
fn every_ctx_append_makes_room_before_its_first_write() {
    for f in [
        "dflash_accept_append",
        "dflash_eagle_accept_append",
        "dflash_eagle_kgamma_append",
        "commit_ctx",
        "dflash_serial_ctx_append",
    ] {
        let b = body(f);
        let write = b
            .find("d.ctx_hidden_acc.offset(d.ctx_len")
            .unwrap_or_else(|| panic!("{f}: no append at ctx_len"));
        let room = ["make_ctx_room(", "make_room_for("]
            .iter()
            .filter_map(|k| b.find(k))
            .min()
            .unwrap_or_else(|| panic!("{f}: appends without making room"));
        assert!(room < write, "{f}: makes room after its first write");
    }
}

#[test]
fn no_hand_rolled_slide_is_left() {
    // 2026-10-17 (A200): the slides `commit_ctx` and `dflash_serial_ctx_append` did inline
    // (one possibly overlapping copy) are gone; only the shared helper moves rows.
    assert!(
        !SRC.contains("d.ctx_positions.drain("),
        "a slide outside make_room_for"
    );
    assert!(
        !SRC.contains("let drop_n ="),
        "a slide outside make_room_for"
    );
}
