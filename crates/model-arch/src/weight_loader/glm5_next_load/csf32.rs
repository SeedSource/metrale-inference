// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `csf_32x2_v1` compressed expert block scales (nvidia/GLM-5.3-Flash-FP4-Local)
//! decoded to the plain row-major `[rows, C]` E4M3 bytes the NVFP4 expert path reads (C = K/16).
//!
//! Re-implemented from the format description in local-inference-lab/flashinfer
//! `docs/csf-scale-blocks.md` and its reference codec `scripts/encode_csf.py` (Apache-2.0); no
//! code copied. Lossless by the spec ("exact scale bytes").
//!
//! Five tensors replace `<proj>.weight_scale`, named `<proj>.weight_scale.csf32_<part>`:
//! - `bases` U8 `[rows]`: per-row base byte.
//! - `codes` U8 `[rows*C/2]`: 4-bit codes, low nibble first, in block order below.
//! - `offsets` U16 `[blocks + pages]`: per page, a prefix sum over its blocks (first entry 0) of
//!   the block's exception count.
//! - `page_bases` U32 `[pages]`: the page's byte origin into `exceptions` (2 bytes a record).
//! - `exceptions` U16: low byte = 2*local_row + local_col, high byte = the true scale byte.
//!
//! Geometry: blocks of 32 rows x 2 columns; groups of up to 4 row-blocks x 2 column-pairs, walked
//! row-major over (row-block step 4, column-pair step 2); inside a group, column pair 0's row
//! blocks come first, then pair 1's; fields inside a block are row-major; pages are 32 groups.
//! Decode: `scale[pos] = bases[row] + code`, then each block's exceptions overwrite positions.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants: every structural fact the format implies is checked (offsets length, page prefixes
//! start at 0, every code and exception consumed exactly, base + code <= 255); a violation is an
//! error naming the tensor, never a partially decoded scale.

use anyhow::{Result, bail};

const BR: usize = 32; // rows per block
const BC: usize = 2; // columns per block
const GROUP_RB: usize = 4; // row blocks per group
const GROUP_CP: usize = 2; // column pairs per group
const PAGE_GROUPS: usize = 32;

/// The five decoded-from-disk parts of one projection's compressed scale.
pub(super) struct Csf32Parts<'a> {
    pub bases: &'a [u8],
    pub codes: &'a [u8],
    pub offsets: &'a [u16],
    pub page_bases: &'a [u32],
    pub exceptions: &'a [u16],
}

/// Block ids (row_block * CP + col_pair) in encoding order, grouped.
fn groups(rows: usize, cols: usize) -> Vec<Vec<usize>> {
    let (rb, cp) = (rows / BR, cols.div_ceil(BC));
    let mut out = Vec::new();
    for r in (0..rb).step_by(GROUP_RB) {
        for c in (0..cp).step_by(GROUP_CP) {
            let mut g = Vec::with_capacity(GROUP_RB * GROUP_CP);
            for cc in c..(c + GROUP_CP).min(cp) {
                for rr in r..(r + GROUP_RB).min(rb) {
                    g.push(rr * cp + cc);
                }
            }
            out.push(g);
        }
    }
    out
}

/// 2026-10-16: Decode one projection's `csf_32x2_v1` scale to plain `[rows, cols]` E4M3 bytes.
pub(super) fn decode(name: &str, rows: usize, cols: usize, p: &Csf32Parts) -> Result<Vec<u8>> {
    if rows == 0 || !rows.is_multiple_of(BR) {
        bail!("{name}: csf_32x2_v1 needs rows a multiple of {BR}, got {rows}");
    }
    if p.bases.len() != rows {
        bail!("{name}: csf32_bases has {} entries for {rows} rows", p.bases.len());
    }
    let n = rows * cols;
    if p.codes.len() != n.div_ceil(2) {
        bail!("{name}: csf32_codes has {} bytes for {n} 4-bit codes", p.codes.len());
    }
    let gs = groups(rows, cols);
    let cp = cols.div_ceil(BC);
    let nblocks: usize = gs.iter().map(Vec::len).sum();
    let pages = gs.len().div_ceil(PAGE_GROUPS);
    if p.offsets.len() != nblocks + pages || p.page_bases.len() != pages {
        bail!(
            "{name}: csf32_offsets {} / page_bases {} entries, expected {} / {pages} ({nblocks} blocks, {pages} pages)",
            p.offsets.len(),
            p.page_bases.len(),
            nblocks + pages
        );
    }
    let mut out = vec![0u8; n];
    let (mut fi, mut before, mut consumed) = (0usize, 0usize, 0usize);
    for (pg, glist) in gs.chunks(PAGE_GROUPS).enumerate() {
        let nb: usize = glist.iter().map(Vec::len).sum();
        let o = &p.offsets[pg + before..pg + before + nb + 1];
        if o[0] != 0 {
            bail!("{name}: page {pg} exception prefix starts at {}, not 0", o[0]);
        }
        if p.page_bases[pg] % 2 != 0 {
            bail!("{name}: page {pg} byte origin {} is not record-aligned", p.page_bases[pg]);
        }
        let begin = p.page_bases[pg] as usize / 2;
        for (k, &b) in glist.iter().flatten().enumerate() {
            let (brow, bcol) = (b / cp, b % cp);
            let first = brow * BR * cols + bcol * BC;
            let w = BC.min(cols - bcol * BC);
            for lr in 0..BR {
                for lc in 0..w {
                    let pos = first + lr * cols + lc;
                    let byte = p.codes[fi >> 1];
                    let code = if fi & 1 == 1 { byte >> 4 } else { byte & 0x0F };
                    fi += 1;
                    let v = p.bases[pos / cols] as usize + code as usize;
                    if v > 255 {
                        bail!("{name}: base {} + code {code} overflows a byte at {pos}", p.bases[pos / cols]);
                    }
                    out[pos] = v as u8;
                }
            }
            let (lo, hi) = (o[k] as usize, o[k + 1] as usize);
            if hi < lo || begin + hi > p.exceptions.len() {
                bail!("{name}: page {pg} block {k} exception range {lo}..{hi} from {begin} outside {} records", p.exceptions.len());
            }
            for &rec in &p.exceptions[begin + lo..begin + hi] {
                let local = (rec & 0xFF) as usize;
                let (lr, lc) = (local >> 1, local & 1);
                if lc >= w || lr >= BR {
                    bail!("{name}: exception {rec:#06x} outside its {BR}x{w} block");
                }
                out[first + lr * cols + lc] = (rec >> 8) as u8;
            }
        }
        before += nb;
        consumed += o[nb] as usize;
    }
    if fi != n || consumed != p.exceptions.len() {
        bail!("{name}: consumed {fi}/{n} codes and {consumed}/{} exceptions", p.exceptions.len());
    }
    Ok(out)
}

/// 2026-10-16: Test oracle: encode plain `[rows, cols]` scale bytes (per-row base = row minimum,
/// codes 0..=15, anything further an exception). Pages are packed back to back. Owned parts in
/// the on-disk element types.
#[cfg(test)]
pub(super) fn encode(rows: usize, cols: usize, s: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u16>, Vec<u32>, Vec<u16>) {
    let bases: Vec<u8> = (0..rows).map(|r| *s[r * cols..(r + 1) * cols].iter().min().unwrap()).collect();
    let gs = groups(rows, cols);
    let cp = cols.div_ceil(BC);
    let (mut codes, mut offsets, mut page_bases, mut exc) = (vec![0u8; (rows * cols).div_ceil(2)], vec![], vec![], vec![]);
    let mut fi = 0usize;
    for glist in gs.chunks(PAGE_GROUPS) {
        page_bases.push((exc.len() * 2) as u32);
        let start = exc.len();
        offsets.push(0u16);
        for &b in glist.iter().flatten() {
            let (brow, bcol) = (b / cp, b % cp);
            let first = brow * BR * cols + bcol * BC;
            let w = BC.min(cols - bcol * BC);
            for lr in 0..BR {
                for lc in 0..w {
                    let pos = first + lr * cols + lc;
                    let d = s[pos] - bases[pos / cols];
                    let code = if d <= 15 { d } else { 0 };
                    if d > 15 {
                        exc.push(((s[pos] as u16) << 8) | (2 * lr + lc) as u16);
                    }
                    codes[fi >> 1] |= if fi & 1 == 1 { code << 4 } else { code };
                    fi += 1;
                }
            }
            offsets.push((exc.len() - start) as u16);
        }
    }
    (bases, codes, offsets, page_bases, exc)
}
