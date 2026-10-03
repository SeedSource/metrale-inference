// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-03: FP8 E4M3 weight with one FP32 scale per output row -> BF16 copy:
// out[n, k] = bf16_rn(float(fp8(in[n, k])) * row_scale[n]).
//
// Owner: gb10 kernels.
// Invariants:
// - in is [N, K] FP8 E4M3 bytes row-major and row_scale is [N] FP32, as quantize_bf16_to_fp8
//   (dense_gemv_fp8w.cu) writes them; out is [N, K] BF16 row-major.
// - Assumes K % 16 == 0 (the host wrapper refuses other K), so every 16-byte vector lies in
//   one row and in, out are 16-byte aligned per vector.
// - Launch: block (256, 1, 1), any grid.x >= 1 (grid-stride over the N * K / 16 vectors).
// - Value: the E4M3 decode is the one dense_gemv_fp8w / dense_gemv_fp8w_batchm use
//   (__nv_fp8_e4m3 -> float, exact; the software decode on SCALE/HIP builds), times the row
//   scale in ONE FP32 multiply (round to nearest even; no FMA is possible), then
//   __float2bfloat16_rn (round to nearest even). A CPU reference doing the same three steps
//   (exact decode, f32 multiply, RNE to BF16) gets the same bits. The GEMV kernels instead
//   keep the decoded weight in FP32 and apply the scale to each thread's partial sum, so a
//   GEMM over this BF16 copy differs from them by the BF16 rounding of each scaled weight
//   (relative <= 2^-9) and by accumulation order; it is not bit-identical to the GEMVs.
// - The quantizer clamps to +-448 and never writes the NaN codes 0x7F / 0xFF; they decode to
//   NaN here on CUDA (0 on SCALE/HIP), as in the GEMVs.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define DQ8_BLOCK 256

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// 2026-10-03: Same software decode as fp8bm_dec in dense_gemv_fp8w_batchm.cu.
__device__ __forceinline__ float dq8_dec(unsigned int b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)                  v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                          v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float dq8_dec(unsigned int b) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = (unsigned char)(b & 0xFFu);
    return (float)f;
}
#endif

// 2026-10-03: Two scaled weights (bytes at bit 0 and bit 8 of b) packed as BF16 lo | hi.
__device__ __forceinline__ unsigned int dq8_pair(unsigned int b, float s) {
    const __nv_bfloat16 lo = __float2bfloat16_rn(dq8_dec(b) * s);
    const __nv_bfloat16 hi = __float2bfloat16_rn(dq8_dec(b >> 8) * s);
    return (unsigned int)(*(const unsigned short*)&lo) |
           ((unsigned int)(*(const unsigned short*)&hi) << 16);
}

extern "C" __global__ void __launch_bounds__(DQ8_BLOCK) dequant_fp8_rowscale_bf16(
    const uint4* __restrict__ in,
    const float* __restrict__ row_scale,
    uint4* __restrict__ out,
    unsigned int N,
    unsigned int K
) {
    const unsigned long long vec_per_row = K / 16u;
    const unsigned long long total = (unsigned long long)N * vec_per_row;
    const unsigned long long stride = (unsigned long long)gridDim.x * DQ8_BLOCK;
    for (unsigned long long v = (unsigned long long)blockIdx.x * DQ8_BLOCK + threadIdx.x;
         v < total; v += stride) {
        const float s = row_scale[v / vec_per_row];
        const uint4 b = in[v];
        const unsigned int w[4] = {b.x, b.y, b.z, b.w};
        unsigned int o[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            o[2 * i] = dq8_pair(w[i], s);
            o[2 * i + 1] = dq8_pair(w[i] >> 16, s);
        }
        out[2 * v] = make_uint4(o[0], o[1], o[2], o[3]);
        out[2 * v + 1] = make_uint4(o[4], o[5], o[6], o[7]);
    }
}
