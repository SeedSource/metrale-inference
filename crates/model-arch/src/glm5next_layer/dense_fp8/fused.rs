// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: `METRALE_GLM_NORM_FP8_QUANT_FUSE=1`: a W8A8 prefill GEMM whose BF16 input is
//! written by an RMSNorm-class kernel skips its separate `per_token_group_quant_fp8` launch,
//! because that kernel also writes the FP8 bytes and scales into the W8A8 scratch.
//!
//! Owner: model-arch (GLM-5.3). Child of [`super`] (`dense_fp8`): it reads the W8A8 scratch,
//! the weight map and `w8a8` through the parent's private items.
//!
//! Invariants:
//! - Engaged ([`norm_fp8_quant_fuse`]) only with `METRALE_GLM_DENSE_FP8_W8A8=1` (hence
//!   `METRALE_GLM_DENSE_FP8=1`). Off, nothing here runs and every path is unchanged.
//! - [`w8a8_fused_input`] returns `Ok(false)`, launching nothing and NOT calling `fill`, for
//!   every call the plain [`super::route`] would not send to the W8A8 arm (a call of at most 16
//!   rows, which takes a GEMV; an unregistered weight; a non-BF16-out GEMV handle; too few
//!   rows; a shape, scratch or kernel the arm declines; 2026-10-08: a call that
//!   `METRALE_GLM_PREFILL_DENSE_W4A4` selects, which `route` sends to W4A4 first). The caller
//!   then runs its unfused norm and `route`, which makes the same decision again.
//! - On `Ok(true)` the GEMM has been issued on `stream`: `fill` ran first (it writes the
//!   caller's BF16 input and the W8A8 scratch's `a_fp8 [m, k]` and `a_scale [m, k / 128]`), then
//!   the same GEMM `route` would issue. Exact when `fill` writes the bytes the unfused norm and
//!   `per_token_group_quant_fp8` would (the caller's contract; its microtest is the gate).
//! - The scratch is single-slot: the share-scope held quant is dropped, and an eager call on
//!   another stream than the previous W8A8 call first synchronizes that stream (as `w8a8`).

use super::*;

/// 2026-10-07: `METRALE_GLM_NORM_FP8_QUANT_FUSE` parsed: `1` is on, anything else (or unset) off.
pub(crate) fn parse_norm_fp8_quant_fuse(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-07: `METRALE_GLM_NORM_FP8_QUANT_FUSE=1` was set and `METRALE_GLM_DENSE_FP8_W8A8=1`
/// is engaged; read once. Set without W8A8 it is ignored, with one warning.
pub fn norm_fp8_quant_fuse() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_NORM_FP8_QUANT_FUSE").ok();
        let asked = parse_norm_fp8_quant_fuse(raw.as_deref());
        if !asked {
            return false;
        }
        if !dense_fp8_w8a8() {
            tracing::warn!(
                "METRALE_GLM_NORM_FP8_QUANT_FUSE=1 is ignored without \
                 METRALE_GLM_DENSE_FP8_W8A8=1 (it only removes a W8A8 quant launch)"
            );
            return false;
        }
        tracing::warn!(
            "METRALE_GLM_NORM_FP8_QUANT_FUSE=1 - the KDA gated output norm also writes the FP8 \
             activation of o_proj's W8A8 GEMM (byte-identical to the separate quant launch)"
        );
        true
    })
}

/// 2026-10-07: `c[m, n] = W8A8(a, weight b)` with the quant fused into the kernel that makes
/// `a` (module doc). `fill(a_fp8, a_scale)` must launch that kernel on `stream`. `Ok(true)`:
/// `fill` and the GEMM were issued; `Ok(false)`: nothing was, run the unfused path.
#[allow(clippy::too_many_arguments)]
pub fn w8a8_fused_input(
    gpu: &dyn GpuBackend,
    gemv: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    (m, n, k): (usize, usize, usize),
    stream: u64,
    fill: &mut dyn FnMut(DevicePtr, DevicePtr) -> Result<()>,
) -> Result<bool> {
    if !norm_fp8_quant_fuse() || !dense_fp8() {
        return Ok(false);
    }
    // `route` sends 1..=16 rows to a GEMV (NVFP4, tensor-core or FP8) ahead of the W8A8 arm;
    // only a wider call reaches it, and only that call may skip its input's separate quant.
    let narrow = NV4_MAX_M
        .max(ops::dense_gemv_tcm::TCM_MAX_M as usize)
        .max(ops::DENSE_GEMV_FP8W_BATCHM_MAX_M as usize);
    if m <= narrow {
        return Ok(false);
    }
    let Some(e) = find(b, n, k)? else {
        return Ok(false);
    };
    let Some(kk) = kernels(gpu) else {
        return Ok(false);
    };
    // 2026-10-08: `METRALE_GLM_PREFILL_DENSE_W4A4`: a call `route` runs as W4A4 needs the BF16
    // input, not the FP8 quant; decline it so the caller runs its unfused norm and `route`.
    if w4a4_copy(&e, m, gemv.0 == kk.bf16_gemv.0).is_some() {
        super::super::dense_w4a4::note_fused_bypass(m, n, k);
        return Ok(false);
    }
    // 2026-10-07: logged once when the fused norm + quant first replaces a quant launch.
    static ENGAGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ENGAGED.get_or_init(|| tracing::warn!("METRALE_GLM_NORM_FP8_QUANT_FUSE=1: ENGAGED"));
    w8a8(
        gpu,
        gemv.0 == kk.bf16_gemv.0,
        a,
        &e,
        c,
        m,
        stream,
        Some(fill),
    )
}

#[cfg(test)]
mod tests {
    use super::parse_norm_fp8_quant_fuse;

    /// 2026-10-07: Only `1` (trimmed) turns the lever on.
    #[test]
    fn norm_fp8_quant_fuse_lever_parses_one_as_on_and_everything_else_as_off() {
        assert!(parse_norm_fp8_quant_fuse(Some("1")));
        assert!(parse_norm_fp8_quant_fuse(Some(" 1 ")));
        for v in [None, Some(""), Some("0"), Some("true"), Some("11")] {
            assert!(!parse_norm_fp8_quant_fuse(v), "{v:?}");
        }
    }
}
