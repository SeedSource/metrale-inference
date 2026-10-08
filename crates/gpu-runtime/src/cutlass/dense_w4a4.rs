// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The CUTLASS Sm120 dense NVFP4 W4A4 GEMM with BF16 output
//! (`metrale_cutlass_nvfp4_dense_w4a4_gemm` in `cuda/cutlass_nvfp4_gemm.cu`, the CUTLASS example
//! 79a configuration), for `METRALE_GLM_PREFILL_DENSE_W4A4`:
//!
//!   out[m, n] = bf16(alpha * Σ_k e2m1(A)[m, k] sfa[m, k/16] · e2m1(W)[n, k] sfb[n, k/16])
//!
//! with A quantized per call from BF16 under a global scale `gs`: the caller's static value with
//! `alpha = weight scale2 * gs`, or (2026-10-08, [`DenseW4a4Args::dynamic_gs`]) one resolved on
//! the device per launch from the amax of that launch's rows (`amax / (6 * 448)`, 1.0 for a zero
//! or non-finite amax; the routed-MoE W4A4 rule) with the caller's `alpha = weight scale2`.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - The weight is the GLM dense NVFP4 copy: packed E2M1 `[N, K/2]` (low nibble first), E4M3
//!   scales `[N, K/16]` row-major; N and K are positive multiples of 128 ([`dense_w4a4_shape_ok`]);
//!   activation, weight, output and a prepacked SFB are 16-byte aligned.
//! - [`nvfp4_dense_w4a4_gemm`] returns `Ok(Declined(status))` only for a refusal made before
//!   anything wrote the output (shape, alignment, scale, workspace, `can_implement`, or a failure
//!   before the first GEMM launch): the caller may then run another path on the same output.
//!   Any later failure is an `Err`.
//! - Static `gs`: each row's activation codes and scales depend only on that row and `gs`, and
//!   each launch covers whole rows, so the output rows do not depend on `m` or on
//!   `rows_per_launch` (`glm_dense_w4a4_prefill_microtest` checks the bits). Dynamic `gs` is not
//!   row-invariant: a row's result depends on the other rows of its launch.

use anyhow::{Result, bail};

#[cfg(metrale_cutlass)]
use std::ffi::c_void;

#[cfg(metrale_cutlass)]
use super::*;

/// 2026-10-08: What [`nvfp4_dense_w4a4_gemm`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseW4a4Outcome {
    /// 2026-10-08: The GEMM ran; `out` holds the result.
    Done,
    /// 2026-10-08: Refused before writing `out` (C status: -1 shape, -2 workspace, -3
    /// `can_implement`, -4 alignment, -5 scale, -6 a failure before the first GEMM launch,
    /// -120 an object built without SM120/121 support).
    Declined(i32),
}

/// 2026-10-08: Whether the entry takes an `[n, k]` weight: both positive multiples of 128.
pub fn dense_w4a4_shape_ok(n: usize, k: usize) -> bool {
    n > 0 && k > 0 && n.is_multiple_of(128) && k.is_multiple_of(128)
}

/// 2026-10-08: The swizzled SFB bytes of an `[n, k]` weight (the CUTLASS layout's size); 0 for a
/// shape [`dense_w4a4_shape_ok`] refuses or a build without CUTLASS.
pub fn dense_w4a4_sfb_bytes(n: usize, k: usize) -> usize {
    #[cfg(metrale_cutlass)]
    {
        if !dense_w4a4_shape_ok(n, k) || n > i32::MAX as usize || k > i32::MAX as usize {
            return 0;
        }
        unsafe { metrale_cutlass_nvfp4_dense_w4a4_sfb_bytes(n as i32, k as i32) as usize }
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (n, k);
        0
    }
}

/// 2026-10-08: Swizzle an `[n, k/16]` E4M3 scale array into `out` (16-byte aligned, at least
/// [`dense_w4a4_sfb_bytes`] bytes) on `stream`: the SFB the GEMM reads. The GEMM does this itself
/// into the shared workspace when it gets no prepacked SFB.
pub fn nvfp4_dense_w4a4_pack_sfb(
    scale_nk: u64,
    out: u64,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(metrale_cutlass)]
    {
        let status = unsafe {
            metrale_cutlass_nvfp4_dense_w4a4_pack_sfb(
                scale_nk as *const c_void,
                out as *mut c_void,
                n as i32,
                k as i32,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS dense W4A4 SFB pack failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (scale_nk, out, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-10-08: The operands of one [`nvfp4_dense_w4a4_gemm`] call (device addresses).
#[derive(Clone, Copy, Debug)]
pub struct DenseW4a4Args {
    /// BF16 `[m, k]`, row stride `k`.
    pub act: u64,
    /// Packed E2M1 `[n, k/2]`.
    pub w_packed: u64,
    /// E4M3 `[n, k/16]` row-major (read when `w_sfb` is 0).
    pub w_scale: u64,
    /// The swizzled SFB ([`nvfp4_dense_w4a4_pack_sfb`]), or 0 to swizzle per call.
    pub w_sfb: u64,
    /// Epilogue scale: weight scale2 * `act_gs` (static), weight scale2 alone (dynamic).
    pub alpha: f32,
    /// Static activation global scale (> 0); unused when `dynamic_gs`.
    pub act_gs: f32,
    /// 2026-10-08: Resolve the global scale per launch from that launch's rows' amax.
    pub dynamic_gs: bool,
    /// BF16 `[m, n]`, row stride `n`.
    pub out: u64,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// Rows per launch; 0 = all rows in one launch.
    pub rows_per_launch: u32,
}

/// 2026-10-08: `out = W4A4(act; weight)` on `stream` in the shared CUTLASS workspace (module doc).
pub fn nvfp4_dense_w4a4_gemm(a: &DenseW4a4Args, stream: u64) -> Result<DenseW4a4Outcome> {
    #[cfg(metrale_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            metrale_cutlass_nvfp4_dense_w4a4_gemm(
                a.act as *const c_void,
                a.w_packed as *const c_void,
                a.w_scale as *const c_void,
                a.w_sfb as *const c_void,
                a.alpha,
                a.act_gs,
                i32::from(a.dynamic_gs),
                a.out as *mut c_void,
                a.m as i32,
                a.n as i32,
                a.k as i32,
                a.rows_per_launch as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        match status {
            0 => Ok(DenseW4a4Outcome::Done),
            -6..=-1 | -120 => Ok(DenseW4a4Outcome::Declined(status)),
            _ => bail!(
                "CUTLASS dense W4A4 GEMM failed after writing part of its output: status {status} \
                 for {}x{}x{}",
                a.m,
                a.n,
                a.k
            ),
        }
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (a, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

#[cfg(test)]
mod tests {
    use super::dense_w4a4_shape_ok;

    /// 2026-10-08: Every GLM-5.3 TP2 dense prefill shape is accepted; a non-multiple of 128 is not.
    #[test]
    fn dense_w4a4_takes_the_glm_shapes_and_refuses_unaligned_ones() {
        for (n, k) in [
            (4096, 4096),
            (1536, 4096),
            (512, 4096),
            (16384, 1536),
            (4096, 16384),
            (1024, 4096),
            (4096, 1024),
            (6144, 4096),
            (4096, 6144),
        ] {
            assert!(dense_w4a4_shape_ok(n, k), "{n}x{k}");
        }
        for (n, k) in [(0, 4096), (4096, 0), (4160, 4096), (4096, 4160), (64, 4096)] {
            assert!(!dense_w4a4_shape_ok(n, k), "{n}x{k}");
        }
    }
}
