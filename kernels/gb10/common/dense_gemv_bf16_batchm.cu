// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Dense BF16 GEMV over M activation rows in one pass over the weight:
// C[t, n] = sum_k A[t, k] * B[n, k] for t in [0, M).
//
// Owner: gb10 kernels.
// Invariants:
// - A is [M, K] contiguous, B is [N, K] row-major, row t of C starts at
//   C + t * out_stride (BF16 elements).
// - Launch: grid (ceil(N / 4), Y, 1), block (256, 1, 1); 64 threads (2 warps) per output.
//   2026-09-27: Y > 1 splits the rows over block rows of ceil(M / Y) each.
// - Only the first min(ceil(M / Y), MAX_M) rows of a block row are computed; the host
//   wrappers refuse more.
// - Assumes K % 8 == 0: rows of A and B start at byte 2 * row * K, which is 16-byte
//   aligned for the uint4 loads only then. The scalar tail covers K % 8, not that alignment.
// - Each row's result is bit-identical to dense_gemv_bf16 on that row: the same kv order
//   (stride 64), the same lo-then-hi add order, the same warp and cross-warp reduction,
//   and the common build passes --fmad=false. Staging A through shared memory changes
//   where the operands are read from, not their values or order.
//
// The four output groups of a block walk the same kv sequence, so each 64-vector slab of
// every A row is staged once per block in shared memory and read by all four groups.




































#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 8
// 2026-09-25: Compile-time cap on batched rows, mirrored by DENSE_GEMV_BATCHM_MAX_M in
// the host wrapper. acc[t] is an independent FP32 chain per row and m enters no row's
// operand order, so every M up to the cap gives each row the same bits.
// Shared memory: As is MAX_M * 64 * 16 B = 16 KB, plus 512 B for the fold.















#define MAX_M 16

extern "C" __global__ void dense_gemv_bf16_batchm(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    // 2026-09-27: With gridDim.y > 1, block row y takes rows [y * R, min((y + 1) * R, M)),
    // R = ceil(M / gridDim.y); each row's arithmetic is the same in every block.
    const unsigned int rows_per_y = (M + gridDim.y - 1) / gridDim.y;
    const unsigned int r0 = blockIdx.y * rows_per_y;
    if (r0 >= M) return;
    A += (unsigned long long)r0 * K;
    C += (unsigned long long)r0 * out_stride;
    M = min(rows_per_y, M - r0);
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    // 2026-09-25: A mask, not a return: every thread of a partial last block has to reach
    // the __syncthreads() calls in the staging loop.
    const bool active = (n < N);

    const unsigned int m = (M > MAX_M) ? MAX_M : M;

    float acc[MAX_M];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)(active ? n : 0) * K);

    // 2026-09-25: One 64-vector slab of every A row, shared by the four output groups.

    __shared__ uint4 As[MAX_M][BLOCK_SIZE / N_PER_BLOCK];

    for (unsigned int base = 0; base < K_VEC; base += threads_per_out) {

        for (unsigned int idx = threadIdx.x; idx < m * threads_per_out; idx += BLOCK_SIZE) {
            const unsigned int t = idx / threads_per_out;
            const unsigned int l = idx % threads_per_out;
            const unsigned int kv = base + l;
            if (kv < K_VEC) {
                As[t][l] = ((const uint4*)(A + (unsigned long long)t * K))[kv];
            }
        }
        __syncthreads();

        const unsigned int kv = base + lane;
        if (kv < K_VEC && active) {

            uint4 b_data = B_vec[kv];
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

            float bf[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 b_lo, b_hi;
                *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                bf[2 * i] = __bfloat162float(b_lo);
                bf[2 * i + 1] = __bfloat162float(b_hi);
            }

            for (unsigned int t = 0; t < m; t++) {
                uint4 a_data = As[t][lane];
                const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
                float a = acc[t];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat16 a_lo, a_hi;
                    *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                    *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);

                    a += __bfloat162float(a_lo) * bf[2 * i];
                    a += __bfloat162float(a_hi) * bf[2 * i + 1];
                }
                acc[t] = a;
            }
        }
        // 2026-09-25: Before the next slab overwrites what the compute above still reads.
        __syncthreads();
    }

    // 2026-09-25: Scalar tail for the last K % 8 elements.




    if (active) {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            const float bfv = __bfloat162float(B_row[k]);
            for (unsigned int t = 0; t < m; t++) {
                acc[t] += __bfloat162float(A[(unsigned long long)t * K + k]) * bfv;
            }
        }
    }

    if (!active) return;

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    for (unsigned int t = 0; t < m; t++) {
        float a = acc[t];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        acc[t] = a;
    }

    // 2026-09-25: Two warps per output: add the warp partials through shared memory, per row.
    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];

    if (warp_lane == 0) {
        const unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        for (unsigned int t = 0; t < m; t++) smem[t][smem_idx] = acc[t];
    }
    __syncthreads();

    if (lane == 0) {
        for (unsigned int t = 0; t < m; t++) {
            const float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * out_stride + n] = __float2bfloat16(r);
        }
    }
}

// 2026-10-01: FP32-output twin of dense_gemv_bf16_batchm, as dense_gemv_bf16_fp32out is of
// dense_gemv_bf16: the body above copied token for token (same kv stride-64 slabs, same
// lo-then-hi add order, same shuffle tree, same smem partial order, same y-split), except
// that C is float and the FP32 sum is stored without rounding to BF16. So each row's result
// is bit-identical to dense_gemv_bf16_fp32out on that row, by the same argument as the
// header gives for the BF16 pair. out_stride counts FP32 elements. Resolved with try_kernel
// by the GLM DSA indexer row batch (METRALE_GLM_DSA_ROW_BATCH=1); gated by
// crates/model-arch/examples/dsa_rowbatch_bitparity_microtest.rs.

extern "C" __global__ void dense_gemv_bf16_fp32out_batchm(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    // 2026-09-27: With gridDim.y > 1, block row y takes rows [y * R, min((y + 1) * R, M)),
    // R = ceil(M / gridDim.y); each row's arithmetic is the same in every block.
    const unsigned int rows_per_y = (M + gridDim.y - 1) / gridDim.y;
    const unsigned int r0 = blockIdx.y * rows_per_y;
    if (r0 >= M) return;
    A += (unsigned long long)r0 * K;
    C += (unsigned long long)r0 * out_stride;
    M = min(rows_per_y, M - r0);
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    // 2026-09-25: A mask, not a return: every thread of a partial last block has to reach
    // the __syncthreads() calls in the staging loop.
    const bool active = (n < N);

    const unsigned int m = (M > MAX_M) ? MAX_M : M;

    float acc[MAX_M];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)(active ? n : 0) * K);

    // 2026-09-25: One 64-vector slab of every A row, shared by the four output groups.

    __shared__ uint4 As[MAX_M][BLOCK_SIZE / N_PER_BLOCK];

    for (unsigned int base = 0; base < K_VEC; base += threads_per_out) {

        for (unsigned int idx = threadIdx.x; idx < m * threads_per_out; idx += BLOCK_SIZE) {
            const unsigned int t = idx / threads_per_out;
            const unsigned int l = idx % threads_per_out;
            const unsigned int kv = base + l;
            if (kv < K_VEC) {
                As[t][l] = ((const uint4*)(A + (unsigned long long)t * K))[kv];
            }
        }
        __syncthreads();

        const unsigned int kv = base + lane;
        if (kv < K_VEC && active) {

            uint4 b_data = B_vec[kv];
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

            float bf[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 b_lo, b_hi;
                *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                bf[2 * i] = __bfloat162float(b_lo);
                bf[2 * i + 1] = __bfloat162float(b_hi);
            }

            for (unsigned int t = 0; t < m; t++) {
                uint4 a_data = As[t][lane];
                const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
                float a = acc[t];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat16 a_lo, a_hi;
                    *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                    *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);

                    a += __bfloat162float(a_lo) * bf[2 * i];
                    a += __bfloat162float(a_hi) * bf[2 * i + 1];
                }
                acc[t] = a;
            }
        }
        // 2026-09-25: Before the next slab overwrites what the compute above still reads.
        __syncthreads();
    }

    // 2026-09-25: Scalar tail for the last K % 8 elements.




    if (active) {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            const float bfv = __bfloat162float(B_row[k]);
            for (unsigned int t = 0; t < m; t++) {
                acc[t] += __bfloat162float(A[(unsigned long long)t * K + k]) * bfv;
            }
        }
    }

    if (!active) return;

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    for (unsigned int t = 0; t < m; t++) {
        float a = acc[t];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        acc[t] = a;
    }

    // 2026-09-25: Two warps per output: add the warp partials through shared memory, per row.
    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];

    if (warp_lane == 0) {
        const unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        for (unsigned int t = 0; t < m; t++) smem[t][smem_idx] = acc[t];
    }
    __syncthreads();

    if (lane == 0) {
        for (unsigned int t = 0; t < m; t++) {
            const float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * out_stride + n] = r;
        }
    }
}
