// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CUTLASS NVFP4 weight pack, scale swizzle and transpose wrappers.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};

#[cfg(metrale_cutlass)]
use std::ffi::c_void;

#[cfg(metrale_cutlass)]
use super::*;

/// 2026-09-25: Swizzle an E4M3 weight scale into the CUTLASS SFB layout
/// (`tile_atom_to_shape_SFB`, ue4m3) that the grouped GEMM reads. The layout
/// depends only on `n` and `k`, so it is built once per expert at load
/// (`MoeLayer::build_cutlass_grouped_sfb`). `scale_out` must hold the whole
/// swizzled region.
///
/// `src_n_major` selects the source layout: `false` reads `[K/16,N]`, `true`
/// reads `[N,K/16]`. The output is the same either way.
pub fn pack_weight_sfb(
    scale_in: u64,
    scale_out: u64,
    n: u32,
    k: u32,
    src_n_major: bool,
    stream: u64,
) -> Result<()> {
    #[cfg(metrale_cutlass)]
    {
        let status = unsafe {
            metrale_cutlass_pack_weight_sfb(
                scale_in as *const c_void,
                scale_out as *mut c_void,
                n as i32,
                k as i32,
                i32::from(src_n_major),
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS weight SFB pack failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (scale_in, scale_out, n, k, src_n_major, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-10-03: Bytes of one expert's swizzled SFB for an `n x k` projection:
/// `round_up(n, 128) * round_up(k / 16, 4)` (the Sm1xx 128 x 4 scale atom; the formula
/// `MoeLayer::build_cutlass_grouped_sfb` sizes its buffers with).
/// [`pack_weight_sfb_batched`] checks it against the CUTLASS layout and refuses a smaller
/// stride.
pub fn sfb_bytes(n: usize, k: usize) -> usize {
    n.div_ceil(128) * 128 * (k / 16).div_ceil(4) * 4
}

/// 2026-10-03: [`pack_weight_sfb`] for `count` experts in one launch. `scale_ptrs_dev` is a
/// DEVICE table of `u64` scale pointers; slot `s` swizzles entry `first + s` (a null entry is
/// skipped) into `out_base + s * out_stride`. `out_stride` must be at least
/// [`sfb_bytes`]`(n, k)`.
#[allow(clippy::too_many_arguments)]
pub fn pack_weight_sfb_batched(
    scale_ptrs_dev: u64,
    first: u32,
    count: u32,
    out_base: u64,
    out_stride: usize,
    n: u32,
    k: u32,
    src_n_major: bool,
    stream: u64,
) -> Result<()> {
    if out_stride < sfb_bytes(n as usize, k as usize) {
        bail!(
            "CUTLASS batched SFB pack: stride {out_stride} < {} bytes for {n}x{k}",
            sfb_bytes(n as usize, k as usize)
        );
    }
    #[cfg(metrale_cutlass)]
    {
        let status = unsafe {
            metrale_cutlass_pack_weight_sfb_batched(
                scale_ptrs_dev as *const u64,
                first as i32,
                count as i32,
                out_base as *mut c_void,
                out_stride as u64,
                n as i32,
                k as i32,
                i32::from(src_n_major),
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS batched SFB pack failed: status {status} for {count} x {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (scale_ptrs_dev, first, count, out_base, n, k, src_n_major, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-09-25: Pack a row-major BF16 weight `[N,K]` into CUTLASS NVFP4: packed
/// `[N,K/2]` (K-contiguous) and E4M3 scales `[K/16,N]`, each scale the group's
/// max magnitude / 6. The scales carry no second-level factor, so pass
/// `weight_scale_2 = 1.0` to `nvfp4_gemm_bf16_act_weight_t`.
pub fn pack_bf16_weight_to_nvfp4_t(
    weight_bf16: u64,
    packed_t: u64,
    scale_t: u64,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(metrale_cutlass)]
    {
        let status = unsafe {
            metrale_cutlass_pack_bf16_weight_to_nvfp4_t(
                weight_bf16 as *const c_void,
                packed_t as *mut c_void,
                scale_t as *mut c_void,
                n as i32,
                k as i32,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS BF16->NVFP4 weight pack failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (weight_bf16, packed_t, scale_t, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-09-25: Transpose a packed NVFP4 weight from `[K/2, N]` into the CUTLASS
/// `[N, K/2]` layout that `nvfp4_gemm_bf16_act_weight_t` reads. A byte
/// transpose: the two nibbles of each byte stay together. `dst_packed` must hold
/// `N * K/2` bytes.
pub fn transpose_nvfp4_packed_kton(
    src_packed_t: u64,
    dst_packed: u64,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(metrale_cutlass)]
    {
        let status = unsafe {
            metrale_cutlass_transpose_nvfp4_packed_kton(
                src_packed_t as *const c_void,
                dst_packed as *mut c_void,
                n as i32,
                k as i32,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS NVFP4 weight transpose failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (src_packed_t, dst_packed, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

#[cfg(test)]
mod sfb_bytes_tests {
    use super::sfb_bytes;

    /// 2026-10-03: GLM-5.3 shapes and the padding of both dimensions.
    #[test]
    fn pads_n_to_128_and_k_groups_to_4() {
        assert_eq!(sfb_bytes(2048, 4096), 2048 * 256);
        assert_eq!(sfb_bytes(4096, 2048), 4096 * 128);
        assert_eq!(sfb_bytes(1, 16), 128 * 4);
        assert_eq!(sfb_bytes(129, 80), 256 * 8);
    }
}
