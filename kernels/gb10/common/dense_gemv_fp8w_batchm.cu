// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-03: FP8-weight GEMV over M activation rows in one pass over the weight:
// C[t, n] = row_scale[n] * sum_k A[t, k] * fp8(B[n, k]) for t in [0, M).
//
// Owner: gb10 kernels.
// Invariants:
// - A is [M, K] BF16 contiguous, B is [N, K] FP8 E4M3 bytes row-major (made by
//   quantize_bf16_to_fp8 in dense_gemv_fp8w.cu), row_scale is [N] FP32, row t of C starts
//   at C + t * out_stride elements (BF16 for dense_gemv_fp8w_batchm, FP32 for
//   dense_gemv_fp8w_fp32out_batchm).
// - Launch: grid (ceil(N / 4), Y, 1), block (256, 1, 1), dynamic shared memory
//   min(ceil(M / Y), MAX_M) * 2048 bytes; 64 threads (2 warps) per output. Y > 1 splits the
//   rows over block rows of ceil(M / Y) each, as dense_gemv_bf16_batchm does.
// - Only the first min(ceil(M / Y), MAX_M) rows of a block row are computed; the host
//   wrappers refuse more.
// - Assumes K % 16 == 0 (no scalar tail, as dense_gemv_fp8w): rows of B start 16-byte
//   aligned and rows of A 32-byte aligned only then. The host wrappers refuse other K.
// - Each row's BF16 result is bit-identical to dense_gemv_fp8w on that row: the same kv
//   order (16-weight vectors, stride 64), the same in-vector order (elements 0..15, each a
//   rounded product then a rounded add; the build passes --fmad=false), the scale applied
//   once to each thread's partial sum before the reduction, the same warp shuffle tree and
//   the same two-warp shared-memory fold. Decoding a weight once and reusing the float for
//   every row does not change its value (E4M3 -> FP32 is exact), and staging A in shared
//   memory changes where an operand is read from, not its value or order.
// - The FP32-output twin stores the same FP32 sum without rounding, so rounding its result
//   to BF16 gives the BF16 kernel's bits.
//
// The four output groups of a block walk the same kv sequence, so each 64-vector slab of
// every A row (64 x 16 BF16 = 2 KB per row) is staged once per block and read by all four.
// The slab's weight vector is loaded before the staging barrier so its DRAM latency
// overlaps the staging copy.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define FP8BM_BLOCK_SIZE 256
#define FP8BM_N_PER_BLOCK 4
#define FP8BM_WARP_SIZE 32
#define FP8BM_THREADS_PER_OUT (FP8BM_BLOCK_SIZE / FP8BM_N_PER_BLOCK)
// 2026-10-03: Compile-time cap on batched rows, mirrored by DENSE_GEMV_FP8W_BATCHM_MAX_M
// in the host wrapper. acc[t] is an independent FP32 chain per row and m enters no row's
// operand order, so every M up to the cap gives each row the same bits.
#define FP8BM_MAX_M 16

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// 2026-10-03: Same software decode as scl_fp8 in dense_gemv_fp8w.cu (each .cu is its own
// module, so the helper is repeated rather than shared).
__device__ __forceinline__ float fp8bm_dec(unsigned int b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)                  v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                          v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float fp8bm_dec(unsigned int b) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = (unsigned char)(b & 0xFFu);
    return (float)f;
}
#endif

__device__ __forceinline__ float fp8bm_bf_lo(unsigned int x) {
    return __uint_as_float(x << 16);
}
__device__ __forceinline__ float fp8bm_bf_hi(unsigned int x) {
    return __uint_as_float(x & 0xFFFF0000u);
}

template <typename OutT>
__device__ __forceinline__ OutT fp8bm_store(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 fp8bm_store<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ float fp8bm_store<float>(float v) {
    return v;
}

template <typename OutT>
__device__ __forceinline__ void fp8bm_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    OutT* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    // 2026-10-03: As[t * 128 + h * 64 + l]: half h (activations 8h..8h+7) of slab vector l
    // of row t, halves stored apart so a warp's 16-byte reads of one half are contiguous
    // (no bank conflict). Sized by the launch to m rows.
    extern __shared__ uint4 fp8bm_As[];

    const unsigned int rows_per_y = (M + gridDim.y - 1) / gridDim.y;
    const unsigned int r0 = blockIdx.y * rows_per_y;
    if (r0 >= M) return;
    A += (unsigned long long)r0 * K;
    C += (unsigned long long)r0 * out_stride;
    M = min(rows_per_y, M - r0);

    const unsigned int local_out = threadIdx.x / FP8BM_THREADS_PER_OUT;
    const unsigned int lane = threadIdx.x % FP8BM_THREADS_PER_OUT;
    const unsigned int n = blockIdx.x * FP8BM_N_PER_BLOCK + local_out;
    // A mask, not a return: every thread of a partial last block reaches the barriers.
    const bool active = (n < N);
    const unsigned int m = (M > FP8BM_MAX_M) ? FP8BM_MAX_M : M;

    float acc[FP8BM_MAX_M];
    #pragma unroll
    for (int t = 0; t < FP8BM_MAX_M; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / 16;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)(active ? n : 0) * K);

    for (unsigned int base = 0; base < K_VEC; base += FP8BM_THREADS_PER_OUT) {
        const unsigned int kv = base + lane;
        const bool do_kv = active && kv < K_VEC;
        uint4 b_data = make_uint4(0u, 0u, 0u, 0u);
        if (do_kv) b_data = B_vec[kv];

        for (unsigned int idx = threadIdx.x; idx < m * 2 * FP8BM_THREADS_PER_OUT; idx += FP8BM_BLOCK_SIZE) {
            const unsigned int t = idx / (2 * FP8BM_THREADS_PER_OUT);
            const unsigned int j = idx % (2 * FP8BM_THREADS_PER_OUT);
            if (base * 2 + j < K_VEC * 2) {
                fp8bm_As[t * 2 * FP8BM_THREADS_PER_OUT + (j & 1u) * FP8BM_THREADS_PER_OUT + (j >> 1)] =
                    ((const uint4*)(A + (unsigned long long)t * K))[base * 2 + j];
            }
        }
        __syncthreads();

        if (do_kv) {
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
            float wf[16];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                wf[4 * i + 0] = fp8bm_dec(b_raw[i]);
                wf[4 * i + 1] = fp8bm_dec(b_raw[i] >> 8);
                wf[4 * i + 2] = fp8bm_dec(b_raw[i] >> 16);
                wf[4 * i + 3] = fp8bm_dec(b_raw[i] >> 24);
            }

            for (unsigned int t = 0; t < m; t++) {
                const uint4 a0 = fp8bm_As[t * 2 * FP8BM_THREADS_PER_OUT + lane];
                const uint4 a1 = fp8bm_As[t * 2 * FP8BM_THREADS_PER_OUT + FP8BM_THREADS_PER_OUT + lane];
                const unsigned int a_raw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                float a = acc[t];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    a += fp8bm_bf_lo(a_raw[i]) * wf[2 * i];
                    a += fp8bm_bf_hi(a_raw[i]) * wf[2 * i + 1];
                }
                acc[t] = a;
            }
        }
        // Before the next slab overwrites what the compute above still reads.
        __syncthreads();
    }

    if (!active) return;

    const float scale = row_scale[n];
    const unsigned int warp_lane = threadIdx.x % FP8BM_WARP_SIZE;

    for (unsigned int t = 0; t < m; t++) {
        float a = acc[t] * scale;
        #pragma unroll
        for (int offset = FP8BM_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        acc[t] = a;
    }

    __shared__ float smem[FP8BM_MAX_M][FP8BM_N_PER_BLOCK * 2];
    if (warp_lane == 0) {
        const unsigned int smem_idx = local_out * 2 + (lane / FP8BM_WARP_SIZE);
        for (unsigned int t = 0; t < m; t++) smem[t][smem_idx] = acc[t];
    }
    __syncthreads();

    if (lane == 0) {
        for (unsigned int t = 0; t < m; t++) {
            const float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * out_stride + n] = fp8bm_store<OutT>(r);
        }
    }
}

extern "C" __global__ void __launch_bounds__(FP8BM_BLOCK_SIZE) dense_gemv_fp8w_batchm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    fp8bm_body<__nv_bfloat16>(A, B, row_scale, C, M, N, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(FP8BM_BLOCK_SIZE) dense_gemv_fp8w_fp32out_batchm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    fp8bm_body<float>(A, B, row_scale, C, M, N, K, out_stride);
}
