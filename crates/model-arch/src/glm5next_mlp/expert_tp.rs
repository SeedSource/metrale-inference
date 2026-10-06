// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Host-side slicing of one routed NVFP4 expert projection for expert-TP
//! (`METRALE_GLM_EXPERT_TP`): rank `r` of `world` keeps its share of the intermediate
//! dimension I.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The layout is the checkpoint's and `w4a16_gemv.cu`'s: packed `[rows, cols / 2]` (two E2M1
//!   codes per byte, low nibble = even column) and E4M3 block scales `[rows, cols / 16]`, both
//!   row-major. `cols` is the logical K.
//! - gate_proj/up_proj (`[I, H]`) are cut by rows ([`Cut::Rows`]: column-parallel); down_proj
//!   (`[H, I]`) by columns ([`Cut::Cols`]: row-parallel), on whole 16-element scale groups. The
//!   per-tensor `scale_2` and `input_scale` are kept whole, so the slices of every rank
//!   reassemble into the original bytes.
//!
//! Design basis: the standard tensor-parallel MoE (Megatron-LM expert TP; vLLM FusedMoE shards
//! `intermediate_size` over TP ranks). SwiGLU is element-wise over I, so each rank's partial
//! `down_r @ (silu(gate_r) * up_r)` sums to the full expert output in the post-MoE all-reduce.

use anyhow::{Result, bail};

/// 2026-10-05: Which dimension of a `[rows, cols]` projection a rank keeps a share of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cut {
    /// 2026-10-05: Rows `[r * rows / world, (r + 1) * rows / world)`: gate_proj and up_proj.
    Rows,
    /// 2026-10-05: Columns `[r * cols / world, (r + 1) * cols / world)` of every row: down_proj.
    Cols,
}

impl Cut {
    /// 2026-10-05: The cut of a routed expert projection by its leaf name: `down_proj` is cut by
    /// columns, every other (`gate_proj`, `up_proj`) by rows.
    pub fn of(proj: &str) -> Self {
        if proj == "down_proj" {
            Self::Cols
        } else {
            Self::Rows
        }
    }
}

/// 2026-10-05: One NVFP4 projection on the host.
#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4Host {
    /// 2026-10-05: `[rows, cols / 2]` packed E2M1 codes.
    pub packed: Vec<u8>,
    /// 2026-10-05: `[rows, cols / 16]` E4M3 block scales.
    pub scale: Vec<u8>,
    /// 2026-10-05: The per-tensor F32 global scale.
    pub scale_2: f32,
    /// 2026-10-05: The static F32 activation scale, 0.0 when absent.
    pub input_scale: f32,
}

/// 2026-10-05: The byte range of rank `rank`'s rows `[rank * rows / world, ...)` in a row-major
/// buffer of `total` bytes and `rows` rows. Errors when `rows` does not split over `world` or
/// `total` over `rows`.
pub fn row_range(
    total: usize,
    rows: usize,
    rank: usize,
    world: usize,
) -> Result<std::ops::Range<usize>> {
    if world == 0 || rank >= world {
        bail!("expert-TP: rank {rank} is outside world {world}");
    }
    if rows == 0 || !rows.is_multiple_of(world) || !total.is_multiple_of(rows) {
        bail!("expert-TP: {total} bytes in {rows} rows do not split over {world} ranks by row");
    }
    let row_bytes = total / rows;
    let part = rows / world;
    Ok(rank * part * row_bytes..(rank + 1) * part * row_bytes)
}

/// 2026-10-05: Rank `rank`'s share of every row of a row-major buffer of `rows` rows: bytes
/// `[rank * w, (rank + 1) * w)` of each row, `w = row_bytes / world`.
pub fn col_part(b: &[u8], rows: usize, rank: usize, world: usize) -> Result<Vec<u8>> {
    if world == 0 || rank >= world {
        bail!("expert-TP: rank {rank} is outside world {world}");
    }
    if rows == 0 || !b.len().is_multiple_of(rows) {
        bail!("expert-TP: {} bytes are not {rows} whole rows", b.len());
    }
    let row_bytes = b.len() / rows;
    if !row_bytes.is_multiple_of(world) {
        bail!("expert-TP: a {row_bytes}-byte row does not split over {world} ranks");
    }
    let w = row_bytes / world;
    let mut out = Vec::with_capacity(rows * w);
    for row in b.chunks_exact(row_bytes) {
        out.extend_from_slice(&row[rank * w..(rank + 1) * w]);
    }
    Ok(out)
}

/// 2026-10-05: Rank `rank` of `world`'s expert-TP slice of the `[rows, cols]` projection `full`,
/// with `scale_2` and `input_scale` copied whole. [`Cut::Cols`] needs `cols / world` to be a
/// multiple of 16 (whole scale groups). Errors when the byte counts do not match `rows` and
/// `cols` or the cut does not split evenly.
pub fn slice_nvfp4(
    full: &Nvfp4Host,
    rows: usize,
    cols: usize,
    cut: Cut,
    rank: usize,
    world: usize,
) -> Result<Nvfp4Host> {
    if !cols.is_multiple_of(16) {
        bail!("expert-TP: {cols} columns are not whole 16-element scale groups");
    }
    if full.packed.len() != rows * cols / 2 || full.scale.len() != rows * cols / 16 {
        bail!(
            "expert-TP: a [{rows}, {cols}] NVFP4 projection needs {} packed and {} scale bytes, \
             got {} and {}",
            rows * cols / 2,
            rows * cols / 16,
            full.packed.len(),
            full.scale.len()
        );
    }
    let (packed, scale) = match cut {
        Cut::Rows => (
            full.packed[row_range(full.packed.len(), rows, rank, world)?].to_vec(),
            full.scale[row_range(full.scale.len(), rows, rank, world)?].to_vec(),
        ),
        Cut::Cols => {
            if world == 0 || !cols.is_multiple_of(world) || !(cols / world).is_multiple_of(16) {
                bail!(
                    "expert-TP: {cols} columns do not split over {world} ranks on whole \
                     16-element scale groups"
                );
            }
            (
                col_part(&full.packed, rows, rank, world)?,
                col_part(&full.scale, rows, rank, world)?,
            )
        }
    };
    Ok(Nvfp4Host {
        packed,
        scale,
        scale_2: full.scale_2,
        input_scale: full.input_scale,
    })
}
