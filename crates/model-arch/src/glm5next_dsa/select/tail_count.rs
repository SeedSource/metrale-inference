// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_GLM_DSA_EXPAND_TAIL_COUNT=1` (default off): the always-tail branch of
//! `dsa_expand_selection` gets each row's `vis_count` from per-chunk counts of `valid_keys`
//! instead of every row scanning all `S` bytes.
//!
//! The original (`kernels/gb10/common/dsa_indexer.cu`, `dsa_expand_selection`) counts, in each
//! of its `Q` blocks, `t` in `[0, S)` with `t <= q_pos[r]` and `valid_keys[t] != 0`: `Q x S`
//! byte reads a call. `valid_keys` is one `[seq]` u8 array per call, shared by all rows and
//! indexed by absolute `t` (`first_key` is not subtracted from it), so those counts share
//! their prefix. `dsa_valid_chunk_counts` makes one integer count per 1,024-byte chunk, one
//! launch a call; `dsa_expand_selection_cc` then adds the whole-chunk counts below
//! `lim = min(q_pos[r] + 1, S)` (0 for a negative `q_pos`) and scans the bytes of the one
//! partial chunk. The same integer over the same bytes, so `vis_count`, the tail and the whole
//! output row are bit-identical. Gate: `examples/dsa_topk_chain_prefill_microtest.rs`
//! (`EXPCNT BITWISE` / `EXPCNT TIME` lines).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`EXPAND_TAIL_COUNT_CHUNK`] equals `DSA_CC_CHUNK` in `dsa_indexer.cu` (checked by a test).
//! - The chunk counts live in the select scratch's `scores` region, dead from the end of the
//!   top-k to the next pass's scores kernel (`dsa_expand_selection` does not read it): no new
//!   allocation. A pass whose region is too small, a ceiling launch, a launch with device
//!   geometry or a module without both entry points keeps `dsa_expand_selection`.
//! - Lever off: `select_tokens` launches exactly what it did before.

use std::sync::OnceLock;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_dsa::Glm5NextDsaKernels;

/// 2026-10-10: Bytes of `valid_keys` one chunk count covers (`DSA_CC_CHUNK`).
pub const EXPAND_TAIL_COUNT_CHUNK: usize = 1024;
/// 2026-10-10: Threads per `dsa_valid_chunk_counts` block (its shared array holds 8 warps).
pub const CHUNK_COUNTS_BLOCK: u32 = 256;
/// 2026-10-10: Logged once, on the first launch that takes `dsa_expand_selection_cc`.
pub const EXPAND_TAIL_COUNT_ENGAGED_LINE: &str = "METRALE_GLM_DSA_EXPAND_TAIL_COUNT ENGAGED";

/// 2026-10-10: Chunk counts for `seq` keys.
pub fn chunk_count(seq: usize) -> usize {
    seq.div_ceil(EXPAND_TAIL_COUNT_CHUNK)
}

/// 2026-10-10: `1` (surrounding blanks ignored) is on; unset, `0` and anything else are off.
pub fn parse_expand_tail_count(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-10: `METRALE_GLM_DSA_EXPAND_TAIL_COUNT`, read once; a value other than unset, empty,
/// `0` or `1` is off, with one warning.
pub fn dsa_expand_tail_count() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_EXPAND_TAIL_COUNT").ok();
        let odd = |r: &&str| !r.is_empty() && r.trim() != "0" && r.trim() != "1";
        if let Some(r) = raw.as_deref().filter(odd) {
            tracing::warn!("METRALE_GLM_DSA_EXPAND_TAIL_COUNT={r} is not 0 or 1 - treated as off");
        }
        parse_expand_tail_count(raw.as_deref())
    })
}

/// 2026-10-10: Whether an expand launch takes the chunk-count path: the lever on, both entry
/// points resolved, an exact launch with host geometry (`S` is the scalar the counts are made
/// for), the always-tail branch (the only one that counts), at least one key, and a `scores`
/// region of at least `chunk_count(seq)` i32.
pub fn expand_tail_count_for(
    requested: bool,
    resolved: bool,
    exact_host_geom: bool,
    always_tail: bool,
    seq: usize,
    scores_bytes: usize,
) -> bool {
    requested
        && resolved
        && exact_host_geom
        && always_tail
        && seq > 0
        && scores_bytes >= chunk_count(seq) * 4
}

/// 2026-10-10: Launch `dsa_valid_chunk_counts` over `valid[0..seq)` into `counts`.
pub(super) fn launch_chunk_counts(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    valid: DevicePtr,
    counts: DevicePtr,
    seq: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.valid_chunk_counts)
        .grid([chunk_count(seq) as u32, 1, 1])
        .block([CHUNK_COUNTS_BLOCK, 1, 1])
        .arg_ptr(valid)
        .arg_ptr(counts)
        .arg_u32(seq as u32)
        .launch(stream)
}

/// 2026-10-10: Log once that the lever engaged, and once why a launch with the lever on did not.
pub(super) fn log_expand_tail_count(requested: bool, engaged: bool, seq: usize) {
    if engaged {
        static ENGAGED: OnceLock<()> = OnceLock::new();
        ENGAGED.get_or_init(|| {
            tracing::warn!(
                "{EXPAND_TAIL_COUNT_ENGAGED_LINE} - dsa_expand_selection_cc takes each row's \
                 visible-key count from dsa_valid_chunk_counts ({EXPAND_TAIL_COUNT_CHUNK}-key \
                 chunks; first engaged launch {seq} keys; the integer count of the byte scan)"
            );
        });
    } else if requested {
        static FELL_BACK: OnceLock<()> = OnceLock::new();
        FELL_BACK.get_or_init(|| {
            tracing::warn!(
                "METRALE_GLM_DSA_EXPAND_TAIL_COUNT=1: NOT engaged on this launch, \
                 dsa_expand_selection runs (a ceiling or device-geometry launch, no always-tail \
                 selection, an unresolved entry point, or a scores region too small for the \
                 chunk counts)"
            );
        });
    }
}

#[cfg(test)]
#[path = "tail_count_tests.rs"]
mod tests;
