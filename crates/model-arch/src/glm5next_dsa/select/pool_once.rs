// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: `METRALE_GLM_DSA_KPOOL_ONCE=1` (default off): the full-width staged prefill
//! (`Glm5NextDsaLayer::decode_k_wide`) compresses the indexer pools once per window instead of
//! once per `core_rows` sub-chunk selection.
//!
//! Why: `dsa_kpool_compress` writes pools `[0, ceil(S / kpool))` from key 0 on every
//! `select_tokens`, so the 256-row sub-chunks of one 8,192-row window compress the same early
//! pools 32 times (measured 1.147 s per 131K prefill, 0.082 s at 32K, rank 0, 2026-10-07).
//!
//! Why the bytes do not change: pool `p` is a function of keys `[p * kpool, (p + 1) * kpool)`
//! (their BF16 key, gate and validity bytes) and the APE table, and nothing else. `S` enters only
//! as the range test `raw < S`, which every key of a complete pool passes, and validity is read
//! only for those keys, which are valid in the sliced path once their sub-chunk's rows are
//! marked and in the once-per-window path because the window's rows are marked before the
//! compress. `pool_keys[p]`, `pool_indices[p * kpool ..]` and `pool_valid[p]` are therefore
//! the same bytes whichever `S >= (p + 1) * kpool` the launch is given. A sub-chunk reads only
//! its first `geom.n_pools = S_j / kpool` pools (`dsa_index_scores`, `dsa_topk_pools` and
//! `dsa_expand_selection` all take `P = n_pools`), each complete at `S_j`. The one region that
//! differs is the trailing partial pool of the sliced path (written invalid at `S_j`), which
//! no later kernel reads. GPU gate: `examples/dsa_kpool_once_microtest.rs`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `compress_window` and every `select_tokens_pooled` that follows it run on one stream with
//!   the same `scratch`, with nothing else writing the pool regions in between. The scratch
//!   may be shared by every DSA layer (`METRALE_GLM_DSA_SELECT_SCRATCH_SHARED`), so the
//!   window's sub-chunks must all be issued before the next layer's selection; `decode_k_wide`
//!   does so (one layer's loop, one stream).
//! - The window's keys, gates and validity bytes are all written before `compress_window`.

use std::sync::OnceLock;

use super::launch::{launch_kpool_compress, select_tokens_with};
use super::*;

/// 2026-10-07: Whether `METRALE_GLM_DSA_KPOOL_ONCE=1` is set; read once. A value other than
/// unset, empty, `0` or `1` is warned about and treated as off.
pub fn dsa_kpool_once() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_KPOOL_ONCE").ok();
        let (on, odd) = parse_kpool_once(raw.as_deref());
        if let Some(r) = odd {
            tracing::warn!("METRALE_GLM_DSA_KPOOL_ONCE={r} is not 0 or 1 - treated as off");
        }
        if on {
            tracing::warn!(
                "METRALE_GLM_DSA_KPOOL_ONCE=1 - full-width DSA prefill compresses the indexer \
                 pools once per window, not per sub-chunk (byte-identical by construction)"
            );
        }
        on
    })
}

/// 2026-10-07: (on, the unparsed value to warn about): only `1` (blanks ignored) is on.
pub(crate) fn parse_kpool_once(v: Option<&str>) -> (bool, Option<String>) {
    match v.map(str::trim) {
        Some("1") => (true, None),
        Some("") | Some("0") | None => (false, None),
        Some(r) => (false, Some(r.to_string())),
    }
}

/// 2026-10-07: Compress every pool of `geom.seq` tokens (`geom` planned at the window's end
/// length) into `scratch`'s pool regions: the one launch `select_tokens` makes per pass,
/// issued once for the window. Fails when `scratch` cannot hold `geom`'s pools or `inputs` is
/// a device-geometry (ceiling) launch, which has no window.
pub fn compress_window(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    stream: u64,
) -> Result<()> {
    if inputs.geom_dev.0 != 0 {
        bail!("DSA pool compress: the once-per-window compress takes host geometry only");
    }
    scratch.fits(cfg, geom)?;
    // 2026-10-07: logged once so a serve shows the lever reached the prefill (gate scripts grep it).
    static ENGAGED: OnceLock<()> = OnceLock::new();
    ENGAGED.get_or_init(|| {
        tracing::warn!(
            "METRALE_GLM_DSA_KPOOL_ONCE=1: ENGAGED - dsa_kpool_compress once per full-width window"
        );
    });
    let (d, kp) = (geom.index_head_dim, geom.index_kpool);
    launch_kpool_compress(
        gpu,
        kernels,
        inputs,
        scratch,
        (geom.n_pools_full, geom.seq),
        (d, kp),
        stream,
    )
}

/// 2026-10-07: [`select_tokens`] on an exact launch without its `dsa_kpool_compress`: the pool
/// regions of `scratch` hold [`compress_window`]'s pools for a window ending at or after
/// `geom.seq`, whose first `geom.n_pools` pools are what this pass would have compressed.
#[allow(clippy::too_many_arguments)]
pub fn select_tokens_pooled(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    stream: u64,
) -> Result<()> {
    select_tokens_with(
        gpu,
        kernels,
        cfg,
        geom,
        inputs,
        scratch,
        DsaSelectLaunch::Exact,
        false,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_turns_it_on() {
        assert_eq!(parse_kpool_once(Some("1")), (true, None));
        assert_eq!(parse_kpool_once(Some(" 1 ")), (true, None));
        for off in [None, Some(""), Some("0")] {
            assert_eq!(parse_kpool_once(off), (false, None));
        }
        assert_eq!(
            parse_kpool_once(Some("true")),
            (false, Some("true".to_string()))
        );
    }
}
