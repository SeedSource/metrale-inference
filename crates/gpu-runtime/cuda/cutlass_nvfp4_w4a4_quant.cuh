// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-08: Device helpers shared by the CUTLASS Sm120 NVFP4 W4A4 wrappers: the RNE E2M1
// code, one 16-value block of the global-scale activation quantizer, and one 128 x 64 tile of
// the weight-scale (SFB) swizzle. Moved verbatim out of cutlass_nvfp4_grouped_gemm.cu (routed
// MoE W4A4) so the dense prefill W4A4 entry in cutlass_nvfp4_gemm.cu
// (METRALE_GLM_PREFILL_DENSE_W4A4) quantizes with the same code.
// Owner: gpu-runtime (CUTLASS reference objects).
// Invariants:
// - Include only inside a CUTLASS_ARCH_MMA_SM120/SM121_SUPPORTED block, after the CUTLASS
//   headers. Header-only, all __forceinline__: each includer compiles its own copy.
// - The quantizer is a pure function of (16 values, gs): no state, no M dependence, so a row's
//   codes and scale do not depend on how rows are batched.
// - CUTLASS (BSD-3-Clause, NVIDIA) headers only; see THIRD_PARTY_NOTICES.md.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cutlass/float8.h"

// 2026-10-03: Round-to-nearest-even E2M1 code, saturating at 6: sign bit 8, magnitude code
// 0..7 = {0, 0.5, 1, 1.5, 2, 3, 4, 6}. This is the rounding of PTX `cvt.rn.satfinite.e2m1x2.f32`
// (ties to the even code: 0.75 -> 1.0, 1.75 -> 2.0, 3.5 -> 4.0), where float_to_e2m1_g
// (cutlass_nvfp4_grouped_gemm.cu) rounds those three ties down. A NaN input gives code 7.
__device__ __forceinline__ unsigned char float_to_e2m1_rne(float x) {
  unsigned char sign = (x < 0.0f) ? 8u : 0u;
  float ax = fabsf(x);
  unsigned char mag;
  if (ax <= 0.25f) {
    mag = 0;
  } else if (ax < 0.75f) {
    mag = 1;
  } else if (ax <= 1.25f) {
    mag = 2;
  } else if (ax < 1.75f) {
    mag = 3;
  } else if (ax <= 2.5f) {
    mag = 4;
  } else if (ax < 3.5f) {
    mag = 5;
  } else if (ax <= 5.0f) {
    mag = 6;
  } else {
    mag = 7;
  }
  return sign | mag;
}

// 2026-10-06: METRALE_CUTLASS_W4A4_PACK_ONCE (gate/up, gathered A). One 16-value block of
// pack_act_grouped_gs: the same expressions in the same order (keep the two in step), so the
// 8 code bytes and the block-scale byte are a pure function of (row values, gs). codes8 gets
// the 8 bytes pack_act_grouped_gs stores at packed[row * (k/2) + base/2 ..].
// 2026-10-08: The dense W4A4 activation pack calls it on 16 values it vector-loaded into a
// local array (base 0); the arithmetic is this function's, unchanged.
__device__ __forceinline__ void quant16_gs(
    const __nv_bfloat16* __restrict__ arow,
    int base,
    float gs,
    unsigned char* __restrict__ codes8,
    unsigned char* __restrict__ sf_out) {
  const float inv_gs = 1.0f / gs;
  float v[16];
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    v[i] = __bfloat162float(arow[base + i]);
    max_abs = fmaxf(max_abs, fabsf(v[i]));
  }
  float sf_val = fminf((max_abs / 6.0f) * inv_gs, 448.0f);
  cutlass::float_ue4m3_t sf(sf_val);
  *sf_out = *reinterpret_cast<unsigned char*>(&sf);
  float dec = static_cast<float>(sf);
  float out_scale = dec > 0.0f ? 1.0f / (dec * gs) : 0.0f;
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    codes8[i / 2] = static_cast<unsigned char>(
        float_to_e2m1_rne(v[i] * out_scale) | (float_to_e2m1_rne(v[i + 1] * out_scale) << 4));
  }
}

// 2026-10-08: One 128 x 64 SFB tile of a [N, K/16] E4M3 scale array (row-major, `groups` =
// K/16 bytes per row) for lane `lane` of its warp: rows nb*128 + j*32 + lane (j < 4), groups
// kb*4 .. kb*4 + 3, each byte converted E4M3 -> float -> UE4M3; words[j] holds row j*32 + lane's
// 4 bytes, the 16 bytes the lane stores at tile base + lane * 16 (see
// pack_weight_sfb_batched_tiled_k for the atom layout). `aligned`: src and groups are multiples
// of 4, so each row's 4 bytes are one 4-byte load. The body of pack_weight_sfb_batched_tiled_k
// moved here unchanged.
__device__ __forceinline__ void sfb_tile_words(
    const unsigned char* __restrict__ src,
    int groups,
    int nb,
    int kb,
    int lane,
    bool aligned,
    unsigned int (&words)[4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned long long row = (unsigned long long)(nb * 128 + j * 32 + lane);
    const unsigned long long at = row * groups + kb * 4;
    unsigned int in4;
    if (aligned) {
      in4 = *reinterpret_cast<const unsigned int*>(src + at);
    } else {
      in4 = (unsigned int)src[at] | ((unsigned int)src[at + 1] << 8) |
            ((unsigned int)src[at + 2] << 16) | ((unsigned int)src[at + 3] << 24);
    }
    unsigned int w = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      __nv_fp8_e4m3 in;
      *reinterpret_cast<unsigned char*>(&in) = (unsigned char)((in4 >> (8 * i)) & 0xFFu);
      cutlass::float_ue4m3_t sf(static_cast<float>(in));
      w |= (unsigned int)(*reinterpret_cast<unsigned char*>(&sf)) << (8 * i);
    }
    words[j] = w;
  }
}
