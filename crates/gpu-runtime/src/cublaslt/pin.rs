// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: `METRALE_CUBLAS_BF16_ALGO_PIN=1`: run every cuBLASLt BF16
//! `act @ weightᵀ` GEMM of one (N, K, out type) on one algorithm, so a row's
//! output bits do not depend on how many rows (M) share its call.
//!
//! Why: by default `gemm_act_weight_t_out` asks `cublasLtMatmulAlgoGetHeuristic`
//! for each call's own M, and the heuristic picks different kernels (tiles,
//! split-K) for different M. A prefix-cache restore off the 8192 grid then
//! runs the suffix at a different M than the cold prefill, and the GLM wide
//! projections (router logits, KDA gates, DSA indexer) give different bits
//! (race-prefill gridfals 2026-10-07: turn 1 differs with cuBLASLt, equal with
//! `METRALE_GLM_CUBLAS_PROJ=0`).
//!
//! Owner: gpu-runtime (cuBLASLt wrapper).
//! Invariants:
//! - The algorithm for a key is the first heuristic result at M = `pin_m`
//!   whose state is success, whose workspace fits, and whose config has
//!   SPLITK_NUM <= 1 and REDUCTION_SCHEME NONE: one dot-product sequence per
//!   output element, independent of M.
//! - The choice is cached per (pin M, N, K, out type) for the process; the
//!   heuristic is deterministic for one device, library and workspace size,
//!   so every rank and every run picks the same algorithm.
//! - A call whose M the pinned algorithm rejects (`cublasLtMatmulAlgoCheck`)
//!   falls back to the heuristic at its own M and logs one warning per key:
//!   that call is not row-invariant.

use super::{CUDA_R_32F, Ctx, Descs, HeuristicResult, ctx, cublasLtMatmulAlgoConfigGetAttribute};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// 2026-10-07: The M the pinned algorithm is chosen at: the GLM full-width
/// prefill window (`METRALE_GLM_PREFILL_FULLWIDTH_GEMM`), the M that carries
/// most of the prefill FLOPs.
pub const BF16_PIN_M: u32 = 8192;

/// 2026-10-07: Heuristic results inspected when choosing.
const CANDIDATES: usize = 16;

const CFG_ID: u32 = 0;
const CFG_TILE_ID: u32 = 1;
const CFG_SPLITK_NUM: u32 = 2;
const CFG_REDUCTION_SCHEME: u32 = 3;
const CFG_CTA_SWIZZLING: u32 = 4;
const CFG_CUSTOM_OPTION: u32 = 5;
const CFG_STAGES_ID: u32 = 6;

type Key = (u32, u32, u32, i32);

/// 2026-10-07: Chosen algorithm per key; `None` = no candidate qualified, so
/// calls of that key keep the per-M heuristic.
static CACHE: OnceLock<Mutex<HashMap<Key, Option<[u64; 8]>>>> = OnceLock::new();
static FALLBACK_LOGGED: OnceLock<Mutex<HashSet<Key>>> = OnceLock::new();
static FALLBACKS: AtomicUsize = AtomicUsize::new(0);

/// 2026-10-07: `METRALE_CUBLAS_BF16_ALGO_PIN=1`, read once; logs ENGAGED.
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("METRALE_CUBLAS_BF16_ALGO_PIN").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_CUBLAS_BF16_ALGO_PIN=1: ENGAGED - one cuBLASLt BF16 algorithm per \
                 (N, K, out type), chosen at M = {BF16_PIN_M}, no split-K"
            );
        }
        on
    })
}

fn cfg_u32(algo: &[u64; 8], attr: u32) -> Option<u32> {
    let mut v = [0u8; 4];
    let mut written = 0usize;
    let st = unsafe {
        cublasLtMatmulAlgoConfigGetAttribute(
            algo.as_ptr() as *const c_void,
            attr,
            v.as_mut_ptr() as *mut c_void,
            4,
            &mut written,
        )
    };
    (st == 0 && written == 4).then(|| u32::from_le_bytes(v))
}

fn describe(algo: &[u64; 8]) -> String {
    let f = |a| cfg_u32(algo, a).map_or("?".to_string(), |v| (v as i32).to_string());
    format!(
        "algo {} tile {} stages {} splitk {} reduction {} swizzle {} custom {}",
        f(CFG_ID),
        f(CFG_TILE_ID),
        f(CFG_STAGES_ID),
        f(CFG_SPLITK_NUM),
        f(CFG_REDUCTION_SCHEME),
        f(CFG_CTA_SWIZZLING),
        f(CFG_CUSTOM_OPTION),
    )
}

/// 2026-10-07: The first qualifying heuristic result at M = `pin_m` (see the
/// module invariants), logged with its config.
fn choose(ctx: &Ctx, pin_m: u32, n: u32, k: u32, out_dtype: i32) -> Result<Option<[u64; 8]>> {
    let d = Descs::new(pin_m, n, k, out_dtype, ctx.ws_size)?;
    let mut res = [HeuristicResult::ZERO; CANDIDATES];
    let got = d.heuristic(ctx, &mut res)?;
    let pick = res[..got].iter().position(|r| {
        r.state == 0
            && r.workspace_size <= ctx.ws_size
            && cfg_u32(&r.algo, CFG_SPLITK_NUM).is_some_and(|s| s <= 1)
            && cfg_u32(&r.algo, CFG_REDUCTION_SCHEME) == Some(0)
    });
    let out = if out_dtype == CUDA_R_32F {
        "f32"
    } else {
        "bf16"
    };
    match pick {
        Some(i) => {
            tracing::info!(
                "cuBLASLt BF16 pin N={n} K={k} out={out} at M={pin_m}: heuristic #{i} of {got}, {}",
                describe(&res[i].algo)
            );
            Ok(Some(res[i].algo))
        }
        None => {
            tracing::warn!(
                "cuBLASLt BF16 pin N={n} K={k} out={out} at M={pin_m}: none of {got} candidates \
                 is single-pass; this shape keeps the per-M heuristic (not row-invariant)"
            );
            Ok(None)
        }
    }
}

/// 2026-10-07: The pinned algorithm for (N, K, `out_dtype`) at M = `pin_m`,
/// chosen on first use and cached.
pub(super) fn algo(
    ctx: &Ctx,
    pin_m: u32,
    n: u32,
    k: u32,
    out_dtype: i32,
) -> Result<Option<[u64; 8]>> {
    let key = (pin_m, n, k, out_dtype);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(a) = cache.lock().unwrap().get(&key) {
        return Ok(*a);
    }
    let a = choose(ctx, pin_m, n, k, out_dtype)?;
    cache.lock().unwrap().insert(key, a);
    Ok(a)
}

/// 2026-10-07: One warning per key when `cublasLtMatmulAlgoCheck` rejects the
/// pinned algorithm at some M (or it needs more workspace than there is).
pub(super) fn note_fallback(m: u32, n: u32, k: u32, out_dtype: i32, status: i32, ws: usize) {
    FALLBACKS.fetch_add(1, Ordering::Relaxed);
    let set = FALLBACK_LOGGED.get_or_init(|| Mutex::new(HashSet::new()));
    if set.lock().unwrap().insert((m, n, k, out_dtype)) {
        tracing::warn!(
            "cuBLASLt BF16 pin FALLBACK M={m} N={n} K={k} out_dtype={out_dtype}: AlgoCheck status \
             {status}, workspace {ws}; this call uses the per-M heuristic (not row-invariant)"
        );
    }
}

/// 2026-10-07: Config text of the algorithm pinned for (N, K, out type) at
/// M = `pin_m`, or `None` when no candidate qualifies. For microtests.
pub fn bf16_pin_describe(pin_m: u32, n: u32, k: u32, out_f32: bool) -> Result<Option<String>> {
    let ctx = ctx()?;
    let dtype = if out_f32 {
        CUDA_R_32F
    } else {
        super::CUDA_R_16BF
    };
    Ok(algo(ctx, pin_m, n, k, dtype)?.map(|a| describe(&a)))
}

/// 2026-10-07: Calls so far that fell back from a pinned algorithm to the
/// per-M heuristic. For microtests.
pub fn bf16_pin_fallbacks() -> usize {
    FALLBACKS.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_result_matches_cublaslt_h_layout() {
        assert_eq!(std::mem::size_of::<HeuristicResult>(), 96);
        assert_eq!(std::mem::offset_of!(HeuristicResult, workspace_size), 64);
        assert_eq!(std::mem::offset_of!(HeuristicResult, state), 72);
    }

    #[test]
    fn pin_m_is_the_fullwidth_window() {
        assert_eq!(BF16_PIN_M, 8192);
    }
}
