// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-03: Layout glue around the vendored FlashKDA forward (vendor/flashkda) for the opt-in
// GLM-5.3 KDA prefill (METRALE_GLM_KDA_PREFILL_FLASHKDA=1,
// crates/model-arch/src/glm5next_kda/prefill_flashkda.rs). Our own code; FlashKDA itself is
// compiled by crates/gpu-runtime/build.rs, not here.
//
// Owner: gb10 kernels.
// Invariants:
// - Pure data movement except kda_o_norm_gated_bf16in, which is kda_o_norm_gated_bf16
//   (kda_layer_ops.cu) reading a BF16 input instead of FP32: the same operations in the same
//   order on the widened value, so for an input that is a BF16 value widened to FP32 the two
//   kernels write the same bits.
// - kda_flk_state_t: src and dst must not alias.

#include <cuda_bf16.h>

// 2026-10-03: kda_flk_beta_t: beta logits [rows, heads] (row stride `heads`) to the
// head-major [heads, rows] FlashKDA reads. Grid (ceil(rows / 256), heads), block 256.
extern "C" __global__ void kda_flk_beta_t(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned int rows,
    unsigned int heads
) {
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int h = blockIdx.y;
    if (t >= rows || h >= heads) return;
    dst[(size_t)h * rows + t] = src[(size_t)t * heads + h];
}

#define KDA_FLK_TILE 32u
#define KDA_FLK_ROWS 8u

// 2026-10-03: kda_flk_state_t: per head, dst[h][j][i] = src[h][i][j] over a d x d FP32 matrix.
// Converts our K-major recurrent state [heads, k, v] to FlashKDA's [heads, v, k] and back (the
// same kernel both ways). Grid (d / 32, d / 32, heads), block (32, 8); d a multiple of 32.
extern "C" __global__ void kda_flk_state_t(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int d
) {
    __shared__ float tile[KDA_FLK_TILE][KDA_FLK_TILE + 1];
    const size_t base = (size_t)blockIdx.z * d * d;
    const unsigned int c0 = blockIdx.x * KDA_FLK_TILE;
    const unsigned int r0 = blockIdx.y * KDA_FLK_TILE;
    for (unsigned int r = threadIdx.y; r < KDA_FLK_TILE; r += KDA_FLK_ROWS) {
        tile[r][threadIdx.x] = src[base + (size_t)(r0 + r) * d + c0 + threadIdx.x];
    }
    __syncthreads();
    for (unsigned int r = threadIdx.y; r < KDA_FLK_TILE; r += KDA_FLK_ROWS) {
        dst[base + (size_t)(c0 + r) * d + r0 + threadIdx.x] = tile[threadIdx.x][r];
    }
}

// 2026-10-03: kda_o_norm_gated_bf16in: kda_o_norm_gated_bf16 with a BF16 input (FlashKDA's
// output) instead of FP32. One block per row of head_dim elements, block = head_dim (a whole
// number of warps, at most 1024); out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i] *
// sigmoid(gate[i]).
extern "C" __global__ void kda_o_norm_gated_bf16in(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int head_dim,
    float eps
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* x = input + (size_t)row * head_dim;
    float acc = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
        float f = __bfloat162float(x[i]);
        acc += f * f;
    }
    __shared__ float red[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < ((blockDim.x + 31) / 32)) ? red[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) red[0] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float)head_dim + eps);
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
        const float g = __bfloat162float(gate[(size_t)row * head_dim + i]);
        const float s = 1.0f / (1.0f + __expf(-g));
        output[(size_t)row * head_dim + i] =
            __float2bfloat16(__bfloat162float(x[i]) * inv * __bfloat162float(weight[i]) * s);
    }
}
