// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Lossless compressed routed-expert block scales held on the device
//! (`METRALE_GLM_MOE_CSF_SCALES=1`, csfcar): the "scale rows with escape" format, v1.
//!
//! A plain NVFP4 block scale is `[rows, c]` E4M3 bytes (`c = K / 16`), read by the MoE kernels as
//! `S[n * c + kk]`. This format keeps, per row, one base byte and `c` 4-bit codes, so a reader
//! that knows `(n, c, kk)` still finds its byte with one load:
//! `code = nibble kk of row n; byte = base[n] + code` for codes 0..=14; code 15 ([`ESCAPE`]) means
//! the byte is in the row's exception list. It is a row-wise variant of NVIDIA's csf_32x2_v1
//! (same base + 4-bit code idea; csf_32x2_v1 walks 32x2 blocks with per-block exception offsets,
//! which a GEMV lane cannot address directly). Measured on 24 official NVFP4 expert projections
//! (runs/race/pf-fp4l/csfsize/csfrow-20261017.txt): 53.50% of plain per rank after the expert-TP
//! cut (csf_32x2_v1: 54.27%), exceptions 0.58% of scales, at most 16 in one row.
//!
//! Buffer layout (`bytes`, little-endian):
//! - `[0, 2 (rows + 1))`: `row_off` u16, a prefix sum: row `n`'s exceptions are records
//!   `row_off[n]..row_off[n + 1]`.
//! - then the exception records, u16 each: `kk << 8 | true byte`, ascending `kk` inside a row.
//! - zero padding to a multiple of 4, then two u32: the head size `H` (bytes from the buffer start
//!   to the row section) and `rows`. So the row section starts at `H`, and `H - 8` and `H - 4`
//!   hold those two words; `H` is a multiple of 4.
//! - `[H, H + rows (1 + c / 2))`: per row, the base byte then `c / 2` code bytes; code `kk` is the
//!   low nibble of byte `kk / 2` for even `kk`, the high nibble for odd `kk`.
//!
//! The device pointer a kernel receives is `buffer + H` with bit 0 set ([`tag`]): a reader tells
//! the formats apart by that bit (plain buffers are at least 2-byte aligned) and, on an escape,
//! finds the header at `ptr - 8` without being told `rows`.
//!
//! Owner: model-arch GLM-5.3 MoE.
//! Invariants: [`decode`] of [`encode`] is the input, byte for byte; [`scale_byte`] is the host
//! mirror of the device reader `moe_scale_byte` (kernels/gb10/common/moe_scale_rows.cuh) and
//! agrees with [`decode`] on every position; encode refuses (never truncates) what the format
//! cannot hold: odd or zero `c`, `c > 256` (`kk` is one byte), more than 65535 exceptions.

use anyhow::{Result, bail, ensure};

/// The code meaning "look the byte up in the row's exception list".
pub const ESCAPE: u8 = 15;
/// Bytes a code covers: codes 0..=14 are `base + code`.
const WINDOW: usize = ESCAPE as usize;

/// 2026-10-17: One encoded projection scale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScaleRows {
    /// The whole buffer, uploaded as is.
    pub bytes: Vec<u8>,
    /// `H`: the row section's offset into `bytes` (a multiple of 4).
    pub head: usize,
    /// Exception records stored.
    pub exceptions: usize,
}

impl ScaleRows {
    /// The device pointer the MoE kernels take for a buffer uploaded at `base`.
    pub fn device_ptr(&self, base: u64) -> u64 {
        tag(base + self.head as u64)
    }
}

/// 2026-10-17: Bit 0 set: the reader's mark for this format.
pub fn tag(ptr: u64) -> u64 {
    ptr | 1
}

/// The row-section stride: one base byte and `c / 2` code bytes.
fn row_stride(c: usize) -> usize {
    1 + c / 2
}

/// 2026-10-17: The base of one row: the start `b` of the 15-wide window `[b, b + 14]` holding the
/// most of the row's bytes (the smallest such `b`), so the fewest bytes become exceptions.
fn row_base(row: &[u8]) -> u8 {
    let mut h = [0u32; 256];
    for &v in row {
        h[v as usize] += 1;
    }
    let mut win: u32 = h[..WINDOW].iter().sum();
    let (mut best, mut best_b) = (win, 0usize);
    for b in 1..=(256 - WINDOW) {
        win = win + h[b + WINDOW - 1] - h[b - 1];
        if win > best {
            (best, best_b) = (win, b);
        }
    }
    best_b as u8
}

/// 2026-10-17: Encode plain `[rows, c]` E4M3 scale bytes.
pub fn encode(rows: usize, c: usize, plain: &[u8]) -> Result<ScaleRows> {
    ensure!(
        rows > 0 && c > 0 && c % 2 == 0 && c <= 256,
        "scale rows: [{rows}, {c}] needs rows > 0 and an even c in 2..=256"
    );
    ensure!(
        plain.len() == rows * c,
        "scale rows: {} bytes for [{rows}, {c}]",
        plain.len()
    );
    let stride = row_stride(c);
    let mut body = vec![0u8; rows * stride];
    let mut row_off: Vec<u16> = Vec::with_capacity(rows + 1);
    let mut exc: Vec<u16> = Vec::new();
    row_off.push(0);
    for (n, row) in plain.chunks_exact(c).enumerate() {
        let base = row_base(row);
        let out = &mut body[n * stride..(n + 1) * stride];
        out[0] = base;
        for (kk, &v) in row.iter().enumerate() {
            let d = v.wrapping_sub(base);
            let code = if v >= base && (d as usize) < WINDOW {
                d
            } else {
                exc.push(((kk as u16) << 8) | v as u16);
                ESCAPE
            };
            out[1 + kk / 2] |= if kk % 2 == 0 { code } else { code << 4 };
        }
        if exc.len() > u16::MAX as usize {
            bail!(
                "scale rows: more than {} exceptions in a [{rows}, {c}] scale",
                u16::MAX
            );
        }
        row_off.push(exc.len() as u16);
    }
    let head = (2 * (rows + 1) + 2 * exc.len()).div_ceil(4) * 4 + 8;
    let mut bytes = Vec::with_capacity(head + body.len());
    for w in row_off.iter().chain(exc.iter()) {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes.resize(head - 8, 0);
    bytes.extend_from_slice(&(head as u32).to_le_bytes());
    bytes.extend_from_slice(&(rows as u32).to_le_bytes());
    bytes.extend_from_slice(&body);
    Ok(ScaleRows {
        bytes,
        head,
        exceptions: exc.len(),
    })
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// 2026-10-17: Byte `(n, kk)` of an encoded scale whose row section starts at `head`, the way the
/// device reader computes it (row section first; on an escape, the header at `head - 8`).
pub fn scale_byte(bytes: &[u8], head: usize, c: usize, n: usize, kk: usize) -> u8 {
    let row = head + n * row_stride(c);
    let b = bytes[row + 1 + kk / 2];
    let code = if kk % 2 == 0 { b & 0x0F } else { b >> 4 };
    if code != ESCAPE {
        return bytes[row].wrapping_add(code);
    }
    let (h, rows) = (
        u32_at(bytes, head - 8) as usize,
        u32_at(bytes, head - 4) as usize,
    );
    let start = head - h;
    let exc0 = start + 2 * (rows + 1);
    let (lo, hi) = (
        u16_at(bytes, start + 2 * n) as usize,
        u16_at(bytes, start + 2 * (n + 1)) as usize,
    );
    for i in lo..hi {
        let r = u16_at(bytes, exc0 + 2 * i);
        if (r >> 8) as usize == kk {
            return r as u8;
        }
    }
    // An escape with no record: the device reader returns 0 too; decode refuses it.
    0
}

/// 2026-10-17: Decode an encoded scale back to plain `[rows, c]` bytes, checking every structural
/// fact (header words, prefix sums, every record used once and inside its row, no escape without
/// a record).
pub fn decode(rows: usize, c: usize, s: &ScaleRows) -> Result<Vec<u8>> {
    let (b, head) = (&s.bytes, s.head);
    ensure!(
        head >= 8 && head % 4 == 0 && b.len() == head + rows * row_stride(c),
        "scale rows: buffer {} bytes, head {head}, for [{rows}, {c}]",
        b.len()
    );
    ensure!(
        u32_at(b, head - 8) as usize == head && u32_at(b, head - 4) as usize == rows,
        "scale rows: header words do not match head {head} / rows {rows}"
    );
    let exc0 = 2 * (rows + 1);
    let total = u16_at(b, 2 * rows) as usize;
    ensure!(
        u16_at(b, 0) == 0 && exc0 + 2 * total <= head - 8,
        "scale rows: exception table overruns the header"
    );
    let mut out = Vec::with_capacity(rows * c);
    for n in 0..rows {
        let (lo, hi) = (u16_at(b, 2 * n) as usize, u16_at(b, 2 * (n + 1)) as usize);
        ensure!(lo <= hi, "scale rows: row {n} exception range {lo}..{hi}");
        let mut used = 0usize;
        for kk in 0..c {
            let rb = b[head + n * row_stride(c) + 1 + kk / 2];
            let code = if kk % 2 == 0 { rb & 0x0F } else { rb >> 4 };
            if code == ESCAPE {
                ensure!(
                    (lo..hi).any(|i| (u16_at(b, exc0 + 2 * i) >> 8) as usize == kk),
                    "scale rows: escape at ({n}, {kk}) has no record"
                );
                used += 1;
            }
            out.push(scale_byte(b, head, c, n, kk));
        }
        ensure!(
            used == hi - lo,
            "scale rows: row {n} has {} records for {used} escapes",
            hi - lo
        );
    }
    Ok(out)
}

/// 2026-10-17: Encoded size of a `[rows, c]` scale with `exceptions` records, without encoding
/// (tests and the arena plan's cross-check).
pub fn encoded_len(rows: usize, c: usize, exceptions: usize) -> usize {
    (2 * (rows + 1) + 2 * exceptions).div_ceil(4) * 4 + 8 + rows * row_stride(c)
}

#[cfg(test)]
#[path = "scale_rows_tests.rs"]
mod tests;
