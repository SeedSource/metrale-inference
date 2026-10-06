// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The CUTLASS Sm120 FP8 blockwise-scaled W8A8 GEMM with BF16 output
//! (`cuda/cutlass_fp8_blockwise_gemm.cu`, after CUTLASS example 87b):
//!
//!   out[m, n] = bf16( Σ_g (Σ_{k∈g} A[m, k] · B[n, k]) · a_scale[m, g] · b_scale[n / 128, g] )
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - `a_scale` is row-major `[M, K/128]` F32 (`per_token_group_quant_fp8`), `b_scale` row-major
//!   `[N/128, K/128]` F32 (`quantize_bf16_to_fp8_blockscaled`), whatever majorness the object
//!   was built with (the C side transposes when it is MN-major).
//! - N and K are positive multiples of [`FP8_BLOCKWISE_BLOCK`]; A, B and `out` are 16-byte
//!   aligned. One call is one launch; M chunking is the caller's.

use anyhow::{Result, bail};

#[cfg(metrale_cutlass)]
use std::ffi::c_void;

#[cfg(metrale_cutlass)]
use super::*;

/// 2026-10-06: Scale block edge on N and K (and the CUTLASS tile K).
pub const FP8_BLOCKWISE_BLOCK: u32 = 128;

/// 2026-10-06: The kernel schedule: cooperative 128x128x128 (`KernelScheduleSm120Blockwise`)
/// or pingpong 64x128x128 (`KernelTmaWarpSpecializedBlockwisePingpongSm120`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8BlockwiseSchedule {
    Cooperative = 0,
    Pingpong = 1,
}

/// 2026-10-06: Whether `[n, k]` is a shape the wrapper takes (both positive multiples of 128).
pub fn fp8_blockwise_shape_ok(n: u32, k: u32) -> bool {
    n > 0 && k > 0 && n.is_multiple_of(FP8_BLOCKWISE_BLOCK) && k.is_multiple_of(FP8_BLOCKWISE_BLOCK)
}

/// 2026-10-06: Whether the CUTLASS object was built with K-major scale layouts (`Ok(false)`:
/// MN-major, the C side transposes the scales per call). Errors without CUTLASS.
pub fn fp8_blockwise_scale_k_major() -> Result<bool> {
    #[cfg(metrale_cutlass)]
    {
        Ok(unsafe { metrale_cutlass_fp8_blockwise_scale_k_major() } != 0)
    }
    #[cfg(not(metrale_cutlass))]
    {
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-10-06: `out[m, n] = W8A8(a_fp8, a_scale; b_fp8, b_scale)` (module doc), one launch on
/// `stream`, using the shared CUTLASS workspace. `m == 0` launches nothing.
#[allow(clippy::too_many_arguments)]
pub fn fp8_blockwise_gemm_bf16(
    a_fp8: u64,
    a_scale: u64,
    b_fp8: u64,
    b_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    schedule: Fp8BlockwiseSchedule,
    stream: u64,
) -> Result<()> {
    if m == 0 {
        return Ok(());
    }
    if !fp8_blockwise_shape_ok(n, k) {
        bail!("CUTLASS fp8 blockwise GEMM: [{n}, {k}] needs N and K positive multiples of 128");
    }
    #[cfg(metrale_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            metrale_cutlass_fp8_blockwise_gemm_bf16(
                a_fp8 as *const c_void,
                a_scale as *const c_void,
                b_fp8 as *const c_void,
                b_scale as *const c_void,
                out as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                schedule as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!(
                "CUTLASS fp8 blockwise GEMM failed: status {status} for {m}x{n}x{k} ({schedule:?}; \
                 -2 workspace too small, -4 misaligned pointer)"
            );
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (a_fp8, a_scale, b_fp8, b_scale, out, schedule, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_blockwise_shape_ok_needs_both_dims_multiples_of_128() {
        assert!(fp8_blockwise_shape_ok(4096, 16384));
        assert!(fp8_blockwise_shape_ok(128, 128));
        assert!(!fp8_blockwise_shape_ok(0, 128));
        assert!(!fp8_blockwise_shape_ok(4096, 0));
        assert!(!fp8_blockwise_shape_ok(4160, 4096));
        assert!(!fp8_blockwise_shape_ok(4096, 4160));
        assert_eq!(Fp8BlockwiseSchedule::Cooperative as i32, 0);
        assert_eq!(Fp8BlockwiseSchedule::Pingpong as i32, 1);
    }
}
