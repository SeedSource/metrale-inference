// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Dense BF16 GEMV for one activation row: C[n] = sum_k A[k] * B[n, k].
//
// Owner: gb10 kernels.
// Invariants:
// - A is [1, K], B is [N, K] row-major (one weight row per output), C is [1, N].
// - Launch: grid (ceil(N / 4), 1, 1), block (256, 1, 1). Each block computes
//   N_PER_BLOCK = 4 outputs with 64 threads (2 warps) per output.
// - Assumes K % 8 == 0: row n starts at byte 2 * n * K, which is 16-byte aligned for
//   the uint4 (8 x BF16) loads only then. The scalar tail covers the K % 8 elements,
//   not that alignment.







#ifndef F32O_BODY_PASS

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 8







extern "C" __global__ void dense_gemv_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    float acc = 0.0f;


    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* A_vec = (const uint4*)A;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 a_data = A_vec[kv];
        uint4 b_data = B_vec[kv];

        // 2026-09-25: A uint4 holds four uint32 words of two BF16 each, low half first.
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }


    {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            acc += __bfloat162float(A[k]) * __bfloat162float(B_row[k]);
        }
    }

    // 2026-09-25: Reduce within each warp, then add the two warp partials of each output
    // through shared memory.
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }


    __shared__ float smem[N_PER_BLOCK * 2];

    if (warp_lane == 0) {

        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();


    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// 2026-09-25: FP32-output twin of dense_gemv_bf16: identical arithmetic, but the FP32
// accumulator is stored without rounding to BF16. Resolved with try_kernel by the
// glm5next DSA layer and MLP, which fall back to the tile GEMM when it is absent.








// 2026-10-12: The entry points. Each includes this file once more with F32O_BODY_PASS set, which
// compiles the kernel text at the end of the file under that entry's name (F32O_V: 0 production,
// 1 _pdl, 2 _pdl_touch), so the production kernel is the body written out directly.
#define F32O_BODY_PASS 1
#define F32O_NAME dense_gemv_bf16_fp32out
#define F32O_V 0
#include "dense_gemv_bf16.cu"
#undef F32O_NAME
#undef F32O_V
#if !defined(__SCALE__) && !defined(__HIP_PLATFORM_AMD__) && defined(__CUDACC__)
#define F32O_PDL_TRIGGER() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
#define F32O_PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#define F32O_NAME dense_gemv_bf16_fp32out_pdl
#define F32O_V 1
#include "dense_gemv_bf16.cu"
#undef F32O_NAME
#undef F32O_V
#define F32O_NAME dense_gemv_bf16_fp32out_pdl_touch
#define F32O_V 2
#include "dense_gemv_bf16.cu"
#undef F32O_NAME
#undef F32O_V
#endif
#undef F32O_BODY_PASS

#else  // F32O_BODY_PASS

// 2026-10-12: the dense_gemv_bf16_fp32out body (see the entry points above).
extern "C" __global__ void F32O_NAME(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
#if F32O_V != 0
    // 2026-10-12: PDL variants (see the entry points). P0: trigger at start.
    F32O_PDL_TRIGGER();
#endif
#if F32O_V == 1
    F32O_PDL_WAIT();  // _pdl: nothing is read before the producer is complete
#endif
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

#if F32O_V == 2
    // _pdl_touch: this thread's first weight vector (real load, kept by an empty asm) is issued
    // before the wait (ld.volatile, so it is not dead-code eliminated); the activation is first
    // read after it.
    {
        if (n < N && lane < K / VEC_SIZE) {
            // PTX-scoped destination registers: no C++ variable is set and left unused
            // (nvcc --Werror all-warnings rejects #550-D).
            const uint4* tp = ((const uint4*)(B + (unsigned long long)n * K)) + lane;
            asm volatile("{ .reg .b32 t0, t1, t2, t3;\n\t"
                         "ld.volatile.global.v4.u32 {t0, t1, t2, t3}, [%0]; }"
                         :: "l"(tp) : "memory");
        }
        F32O_PDL_WAIT();
    }
#endif

    float acc = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* A_vec = (const uint4*)A;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 a_data = A_vec[kv];
        uint4 b_data = B_vec[kv];

        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }

    {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            acc += __bfloat162float(A[k]) * __bfloat162float(B_row[k]);
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float smem[N_PER_BLOCK * 2];

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = result;
    }
}

#endif  // F32O_BODY_PASS
