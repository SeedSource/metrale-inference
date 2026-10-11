// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: `METRALE_CUTLASS_W4A4_GROUPED_TILE` (named `METRALE_CUTLASS_W4A4_GROUPED_TILE` on the
//! 10-07 branch seed/race-pf-moetile; renamed 2026-10-11): which instantiation of the CUTLASS
//! Sm120 NVFP4 grouped GEMM the W4A4 MoE prefill entries launch (`W4a4TileVariant` in
//! `cuda/cutlass_nvfp4_grouped_gemm.cu`, which cites where each tile / schedule comes from).
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - The env is read once, by the first W4A4 grouped call ([`w4a4_gemm_tile_lever`]), logged
//!   once at WARN when it selects a variant or is not understood, and handed to C before that
//!   call launches. Unset, empty, `pp128` or `default` selects the default kernel, which
//!   launches exactly what it launched before the lever existed.
//! - Every variant has the default's strides, scale-factor layouts and epilogue (static_assert
//!   on the C side), so it reads the same packed A / SFA and weight SFB cache. It is meant
//!   bitwise identical to the default; `glm_moe_w4a4_tile_microtest` (in-process, every
//!   variant) and the `HASH` lines of `glm_moe_w4a4_cutlass_microtest` (one variant per run)
//!   check that.
//! - A variant that does not fit the CUTLASS workspace or fails `can_implement` launches
//!   nothing and that GEMM runs the default instead (counted in [`w4a4_gemm_tile_stats`],
//!   logged once by the C side).
//! - 2026-10-11: [`set_w4a4_gemm_timing`] / [`w4a4_gemm_timing_take`]: CUDA-event time of the
//!   grouped GEMM launches alone (per projection), for microtests.

use std::sync::OnceLock;

#[cfg(metrale_cutlass)]
use super::{
    metrale_cutlass_set_w4a4_gemm_skip, metrale_cutlass_set_w4a4_gemm_tile,
    metrale_cutlass_set_w4a4_gemm_tile_override, metrale_cutlass_set_w4a4_gemm_timing,
    metrale_cutlass_w4a4_gemm_tile_stats, metrale_cutlass_w4a4_gemm_timing_take,
};

/// 2026-10-07: One grouped GEMM instantiation; `id` is the C side's variant number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum W4a4GemmTile {
    /// 2026-10-07: 128x128x128, ptr-array pingpong (the kernel before the lever).
    Default,
    /// 2026-10-07: 128x128x128, ptr-array cooperative.
    Coop,
    /// 2026-10-07: 128x128x256, ptr-array pingpong.
    K256,
    /// 2026-10-07: 128x128x256, ptr-array cooperative.
    CoopK256,
    /// 2026-10-11: 256x128x128, ptr-array cooperative.
    Co256,
    /// 2026-10-11: 128x256x128, ptr-array cooperative.
    Co128N256,
}

impl W4a4GemmTile {
    pub const ALL: [Self; 6] = [
        Self::Default,
        Self::Coop,
        Self::K256,
        Self::CoopK256,
        Self::Co256,
        Self::Co128N256,
    ];

    pub fn id(self) -> i32 {
        match self {
            Self::Default => 0,
            Self::Coop => 1,
            Self::K256 => 2,
            Self::CoopK256 => 3,
            Self::Co256 => 4,
            Self::Co128N256 => 5,
        }
    }

    pub fn from_id(id: i64) -> Option<Self> {
        Self::ALL.into_iter().find(|t| i64::from(t.id()) == id)
    }

    /// 2026-10-07: The lever value that selects it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Default => "pp128",
            Self::Coop => "co128",
            Self::K256 => "pp128k256",
            Self::CoopK256 => "co128k256",
            Self::Co256 => "co256",
            Self::Co128N256 => "co128n256",
        }
    }

    pub fn desc(self) -> &'static str {
        match self {
            Self::Default => "128x128x128 ptr-array pingpong (default)",
            Self::Coop => "128x128x128 ptr-array cooperative",
            Self::K256 => "128x128x256 ptr-array pingpong",
            Self::CoopK256 => "128x128x256 ptr-array cooperative",
            Self::Co256 => "256x128x128 ptr-array cooperative",
            Self::Co128N256 => "128x256x128 ptr-array cooperative",
        }
    }

    /// 2026-10-07: A lever value (name, case-insensitive, surrounding blanks ignored); empty or
    /// `default` is the default. `None` for anything else.
    pub fn parse(v: &str) -> Option<Self> {
        let v = v.trim();
        if v.is_empty() || v.eq_ignore_ascii_case("default") {
            return Some(Self::Default);
        }
        Self::ALL
            .into_iter()
            .find(|t| t.name().eq_ignore_ascii_case(v))
    }
}

/// 2026-10-07: The tile a raw lever value selects, and the WARN line to log (none for the
/// default). An unknown value selects the default.
fn resolve(raw: &str) -> (W4a4GemmTile, Option<String>) {
    match W4a4GemmTile::parse(raw) {
        Some(W4a4GemmTile::Default) => (W4a4GemmTile::Default, None),
        Some(t) => (
            t,
            Some(format!(
                "METRALE_CUTLASS_W4A4_GROUPED_TILE={raw}: ENGAGED - CUTLASS W4A4 grouped GEMM tile {}",
                t.desc()
            )),
        ),
        None => (
            W4a4GemmTile::Default,
            Some(format!(
                "METRALE_CUTLASS_W4A4_GROUPED_TILE={raw}: unknown value (want pp128, co128, pp128k256, \
                 co128k256, co256 or co128n256); the CUTLASS W4A4 grouped GEMM keeps the default tile {}",
                W4a4GemmTile::Default.desc()
            )),
        ),
    }
}

/// 2026-10-07: The lever's tile, read and logged once; the first call also hands it to the C
/// side. The W4A4 grouped wrappers call this before every launch.
pub fn w4a4_gemm_tile_lever() -> W4a4GemmTile {
    static V: OnceLock<W4a4GemmTile> = OnceLock::new();
    *V.get_or_init(|| {
        let raw = std::env::var("METRALE_CUTLASS_W4A4_GROUPED_TILE").unwrap_or_default();
        let (tile, line) = resolve(&raw);
        if let Some(line) = line {
            tracing::warn!("{line}");
        }
        #[cfg(metrale_cutlass)]
        unsafe {
            metrale_cutlass_set_w4a4_gemm_tile(tile.id());
        }
        tile
    })
}

/// 2026-10-07: Test hook (`glm_moe_w4a4_tile_microtest`): `Some(t)` makes every later W4A4
/// grouped GEMM in this process launch `t`; `None` follows the lever again.
pub fn set_w4a4_gemm_tile_override(tile: Option<W4a4GemmTile>) {
    #[cfg(metrale_cutlass)]
    unsafe {
        metrale_cutlass_set_w4a4_gemm_tile_override(tile.map_or(-1, W4a4GemmTile::id));
    }
    #[cfg(not(metrale_cutlass))]
    let _ = tile;
}

/// 2026-10-07: Timing hook (`glm_moe_w4a4_tile_microtest`): `true` makes every later W4A4
/// grouped call stage its A side and arrays but launch no GEMM (outputs are not written), so
/// the A-side prep can be timed alone; `false` restores normal calls.
pub fn set_w4a4_gemm_skip(on: bool) {
    #[cfg(metrale_cutlass)]
    unsafe {
        metrale_cutlass_set_w4a4_gemm_skip(i32::from(on));
    }
    #[cfg(not(metrale_cutlass))]
    let _ = on;
}

/// 2026-10-07: (the variant the last W4A4 grouped GEMM launched, `None` before any; how many
/// W4A4 GEMMs fell back to the default because the selected variant did not fit the workspace
/// or failed `can_implement`). Diagnostic: concurrent callers may interleave. `(None, 0)`
/// without CUTLASS.
pub fn w4a4_gemm_tile_stats() -> (Option<W4a4GemmTile>, u64) {
    #[cfg(metrale_cutlass)]
    {
        let mut v = [0u64; 2];
        unsafe { metrale_cutlass_w4a4_gemm_tile_stats(v.as_mut_ptr(), v.len() as i32) };
        (W4a4GemmTile::from_id(v[0] as i64), v[1])
    }
    #[cfg(not(metrale_cutlass))]
    (None, 0)
}

/// 2026-10-11: GEMM-only time of the W4A4 grouped GEMM launches of one projection slot since the
/// last [`w4a4_gemm_timing_take`] (CUDA events around `gemm.run` on the call's stream).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct W4a4GemmTiming {
    pub count: u64,
    pub sum_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

impl W4a4GemmTiming {
    pub fn mean_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_ms / self.count as f64
        }
    }
}

/// 2026-10-11: Timing hook (`glm_moe_w4a4_cutlass_microtest` RPE sweep): `true` makes every later
/// W4A4 grouped GEMM record a CUDA event pair around its launch; `false` stops. Microtests only
/// (not for concurrent callers); no effect without CUTLASS.
pub fn set_w4a4_gemm_timing(on: bool) {
    #[cfg(metrale_cutlass)]
    unsafe {
        metrale_cutlass_set_w4a4_gemm_timing(i32::from(on));
    }
    #[cfg(not(metrale_cutlass))]
    let _ = on;
}

/// 2026-10-11: Waits for the recorded pairs and returns their times per slot `[gate, up, down]`,
/// clearing them. All zero without CUTLASS or when nothing was recorded.
pub fn w4a4_gemm_timing_take() -> anyhow::Result<[W4a4GemmTiming; 3]> {
    #[cfg(metrale_cutlass)]
    {
        let mut v = [0f64; 12];
        let rc = unsafe { metrale_cutlass_w4a4_gemm_timing_take(v.as_mut_ptr(), 3) };
        if rc != 0 {
            anyhow::bail!("W4A4 GEMM timing events failed (CUDA error {rc})");
        }
        Ok(std::array::from_fn(|i| W4a4GemmTiming {
            count: v[i * 4] as u64,
            sum_ms: v[i * 4 + 1],
            min_ms: v[i * 4 + 2],
            max_ms: v[i * 4 + 3],
        }))
    }
    #[cfg(not(metrale_cutlass))]
    Ok([W4a4GemmTiming::default(); 3])
}

#[cfg(test)]
mod tests {
    use super::{W4a4GemmTile, resolve};

    #[test]
    fn names_ids_and_unknown_values() {
        for t in W4a4GemmTile::ALL {
            assert_eq!(W4a4GemmTile::parse(t.name()), Some(t));
            assert_eq!(W4a4GemmTile::from_id(i64::from(t.id())), Some(t));
        }
        assert_eq!(W4a4GemmTile::parse(" CO128 "), Some(W4a4GemmTile::Coop));
        assert_eq!(W4a4GemmTile::parse(""), Some(W4a4GemmTile::Default));
        assert_eq!(W4a4GemmTile::parse("default"), Some(W4a4GemmTile::Default));
        assert_eq!(W4a4GemmTile::parse("co256"), Some(W4a4GemmTile::Co256));
        assert_eq!(W4a4GemmTile::parse("coop"), None);
        assert_eq!(W4a4GemmTile::from_id(-1), None);
        assert_eq!(resolve("default"), (W4a4GemmTile::Default, None));
        assert_eq!(resolve("pp128"), (W4a4GemmTile::Default, None));
        let (t, line) = resolve("pp128k256");
        assert_eq!(t, W4a4GemmTile::K256);
        assert!(line.is_some_and(|l| l.contains("ENGAGED")));
        let (t, line) = resolve("256x128");
        assert_eq!(t, W4a4GemmTile::Default);
        assert!(line.is_some_and(|l| l.contains("unknown value")));
    }
}
