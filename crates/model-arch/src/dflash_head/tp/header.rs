// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The propose header rank 0 broadcasts to rank 1 at the start of every
//! two-rank DFlash propose (`METRALE_DFLASH_DRAFT_TP2`), and its codec.
//!
//! One transfer is a fixed-size block: `HDR_FIXED_WORDS + inline_cap` u32 words, padded to
//! [`hdr_bytes`], then `inline_cap` ctx rows. The first transfer (the main header) carries
//! every value rank 1 needs to take the same branches and issue the same collectives as rank
//! 0: block width, token, position, the ctx rows to keep, the new rows' positions and the
//! precompute range. When more new rows exist than one transfer holds, chunk transfers of
//! the same size follow ([`transfer_rows`]).
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - `decode_main(encode_main(h))` returns `h` with the first transfer's positions
//!   (`tests.rs`).
//! - Every field that changes rank 1's control flow is in the header; rank 1 reads nothing
//!   from its own scheduler state.

use anyhow::{Result, ensure};

/// 2026-10-09: "DFTP".
pub const HDR_MAGIC: u32 = 0x4446_5450;
/// 2026-10-09: "DFTC", a chunk transfer.
pub const CHUNK_MAGIC: u32 = 0x4446_5443;
pub const HDR_VERSION: u32 = 1;
/// 2026-10-09: Fixed words before the inline positions.
pub const HDR_FIXED_WORDS: usize = 16;
/// 2026-10-09: The leader could not prepare; rank 1 returns without collectives.
pub const FLAG_ABORT: u32 = 1;
/// 2026-10-09: The leader resent every row (first sync, or a reported desync).
pub const FLAG_RESYNC: u32 = 2;

/// 2026-10-09: Bytes of a transfer's header region: the words, rounded up to 256 so the
/// rows that follow stay aligned.
pub fn hdr_bytes(inline_cap: usize) -> usize {
    ((HDR_FIXED_WORDS + inline_cap) * 4).div_ceil(256) * 256
}

/// 2026-10-09: Rows per transfer for `n_new` new rows: the main header's share first, then
/// full chunks, then the rest. Empty when `n_new == 0` (the main header still goes).
pub fn transfer_rows(n_new: usize, inline_cap: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut left = n_new;
    while left > 0 && inline_cap > 0 {
        let n = left.min(inline_cap);
        out.push(n);
        left -= n;
    }
    out
}

/// 2026-10-09: The main header. `positions` holds every new row's RoPE position
/// (`n_new` entries) on the leader; after `decode_main` it holds the first transfer's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TpHeader {
    pub flags: u32,
    pub last_token: u32,
    pub position: u32,
    /// 2026-10-09: Block width (`block_g`) of this propose.
    pub width: u32,
    /// 2026-10-09: Rank 0's `ctx_len` after its prep.
    pub ctx_len: u32,
    /// 2026-10-09: First ctx row whose K/V this propose precomputes.
    pub committed: u32,
    /// 2026-10-09: Precompute chunk rows (rank 0's `ctx_window`).
    pub chunk: u32,
    /// 2026-10-09: Leading rows (rank-0 indexing) rank 1 already holds.
    pub keep: u32,
    /// 2026-10-09: Rank 0's position of ctx row 0, `-1` when `ctx_len == 0`.
    pub first_pos: i32,
    /// 2026-10-09: Rank 0's position of ctx row `keep - 1`, `-1` when `keep == 0`.
    pub keep_last_pos: i32,
    /// 2026-10-09: New rows `[keep, ctx_len)` the transfers carry.
    pub n_new: u32,
    pub positions: Vec<i32>,
}

fn checksum(words: &[u32]) -> u32 {
    words
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 14)
        .fold(0x9E37_79B9u32, |acc, (_, &w)| acc.rotate_left(5) ^ w)
}

impl TpHeader {
    /// 2026-10-09: The header of a propose the leader abandoned before any collective.
    pub fn abort() -> Self {
        Self {
            flags: FLAG_ABORT,
            first_pos: -1,
            keep_last_pos: -1,
            ..Self::default()
        }
    }

    pub fn is_abort(&self) -> bool {
        self.flags & FLAG_ABORT != 0
    }

    /// 2026-10-09: The main transfer's words (`HDR_FIXED_WORDS + inline_cap`).
    pub fn encode_main(&self, inline_cap: usize) -> Vec<u32> {
        let mut w = vec![0u32; HDR_FIXED_WORDS + inline_cap];
        let first = (self.n_new as usize).min(inline_cap);
        w[0] = HDR_MAGIC;
        w[1] = HDR_VERSION;
        w[2] = self.flags;
        w[3] = self.last_token;
        w[4] = self.position;
        w[5] = self.width;
        w[6] = self.ctx_len;
        w[7] = self.committed;
        w[8] = self.chunk;
        w[9] = self.keep;
        w[10] = self.first_pos as u32;
        w[11] = self.keep_last_pos as u32;
        w[12] = self.n_new;
        w[13] = first as u32;
        for (i, &p) in self.positions.iter().take(first).enumerate() {
            w[HDR_FIXED_WORDS + i] = p as u32;
        }
        w[14] = checksum(&w);
        w
    }

    /// 2026-10-09: Decode and validate a main transfer. Errors on a bad magic, version or
    /// checksum, or counts that do not add up.
    pub fn decode_main(w: &[u32], inline_cap: usize) -> Result<Self> {
        ensure!(
            w.len() >= HDR_FIXED_WORDS + inline_cap,
            "TP header: {} words, want {}",
            w.len(),
            HDR_FIXED_WORDS + inline_cap
        );
        let w = &w[..HDR_FIXED_WORDS + inline_cap];
        ensure!(w[0] == HDR_MAGIC, "TP header: magic {:#x}", w[0]);
        ensure!(w[1] == HDR_VERSION, "TP header: version {}", w[1]);
        ensure!(w[14] == checksum(w), "TP header: checksum mismatch");
        let h = Self {
            flags: w[2],
            last_token: w[3],
            position: w[4],
            width: w[5],
            ctx_len: w[6],
            committed: w[7],
            chunk: w[8],
            keep: w[9],
            first_pos: w[10] as i32,
            keep_last_pos: w[11] as i32,
            n_new: w[12],
            positions: Vec::new(),
        };
        let first = w[13] as usize;
        ensure!(
            first == (h.n_new as usize).min(inline_cap),
            "TP header: {first} inline rows for {} new",
            h.n_new
        );
        ensure!(
            h.keep as u64 + h.n_new as u64 == h.ctx_len as u64,
            "TP header: keep {} + new {} != ctx_len {}",
            h.keep,
            h.n_new,
            h.ctx_len
        );
        ensure!(h.committed <= h.ctx_len, "TP header: committed > ctx_len");
        let positions = w[HDR_FIXED_WORDS..HDR_FIXED_WORDS + first]
            .iter()
            .map(|&p| p as i32)
            .collect();
        Ok(Self { positions, ..h })
    }
}

/// 2026-10-09: A chunk transfer's words: magic, row count, checksum, then the positions.
pub fn encode_chunk(positions: &[i32], inline_cap: usize) -> Vec<u32> {
    let mut w = vec![0u32; HDR_FIXED_WORDS + inline_cap];
    w[0] = CHUNK_MAGIC;
    w[1] = HDR_VERSION;
    w[13] = positions.len() as u32;
    for (i, &p) in positions.iter().take(inline_cap).enumerate() {
        w[HDR_FIXED_WORDS + i] = p as u32;
    }
    w[14] = checksum(&w);
    w
}

/// 2026-10-09: Decode a chunk transfer expected to carry `rows` rows.
pub fn decode_chunk(w: &[u32], rows: usize, inline_cap: usize) -> Result<Vec<i32>> {
    ensure!(w.len() >= HDR_FIXED_WORDS + inline_cap, "TP chunk: short");
    let w = &w[..HDR_FIXED_WORDS + inline_cap];
    ensure!(
        w[0] == CHUNK_MAGIC && w[1] == HDR_VERSION,
        "TP chunk: bad magic"
    );
    ensure!(w[14] == checksum(w), "TP chunk: checksum mismatch");
    ensure!(
        w[13] as usize == rows && rows <= inline_cap,
        "TP chunk: {} rows, want {rows}",
        w[13]
    );
    Ok(w[HDR_FIXED_WORDS..HDR_FIXED_WORDS + rows]
        .iter()
        .map(|&p| p as i32)
        .collect())
}

/// 2026-10-09: Host words to bytes (little endian) and back.
pub fn words_to_bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn bytes_to_words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
