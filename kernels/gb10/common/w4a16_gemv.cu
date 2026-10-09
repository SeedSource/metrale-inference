// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: NVFP4-weight GEMVs with BF16 activations for decode and small row counts:
// M = 1 (w4a16_gemv, _sw, _logits), M up to 32 (w4a16_gemv_batch*), the GEMVs that store
// their output deinterleaved (w4a16_gemv_qg*, _qkvz) or run two weights (_dual_batch*), and
// the routed-expert GEMVs (w4a16_gemv_sw_moe*, glm5next_moe_row_union).
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
//
// B_packed is [N, K/2]: byte j of row n holds W[n, 2j] in its low nibble and W[n, 2j + 1] in
// its high nibble. B_scale is [N, K/16] E4M3, one byte per 16 K values, and scale2 is FP32
// per tensor. A weight is E2M1_LUT[nibble] * e4m3(scale byte) * scale2. Kernels that walk K
// in 16-value chunks need K to be a multiple of 16, the 8-value ones (qg, qkvz, dual) a
// multiple of 8; a K tail is not read.
//
// The strix, strix-hip and b300 common KERNEL.toml files compile this same file.






#include <cuda_bf16.h>
#include <cuda_fp8.h>

// 2026-09-25: Software E4M3 decode for the SCALE and HIP builds: exponent 0 is subnormal
// (m * 2^-9), and the NaN code (exponent 15, mantissa 7) decodes to 0. CUDA builds use the
// __nv_fp8_e4m3 conversion.


#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#endif

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16


__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// 2026-09-25: The M = 1 partials read E2M1 values from a shared-memory copy of E2M1_LUT,
// not from the __constant__ table: the index is a data-dependent nibble, and a __constant__
// read is serialized across the distinct addresses of a warp. The copy holds the same FP32
// values. gemv_sw_tests.rs (decode_gemv_partials_index_a_shared_staged_lut) checks it.
//
// With METRALE_WARP_LUT_STAGED 0 (the SCALE and HIP builds) the one-warp-per-output kernels
// stage nothing and pass the __constant__ table; the block-staged kernels stage on every
// target.
























#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
#define METRALE_WARP_LUT_STAGED 0
#else
#define METRALE_WARP_LUT_STAGED 1
#endif

// 2026-09-25: Lanes 0..15 fill this warp's own copy and __syncwarp publishes it. It serves
// the one-warp-per-output kernels, which return early on n >= N for a whole warp, where a
// block-wide __syncthreads would be a divergent barrier.

__device__ __forceinline__ void stage_e2m1_lut_warp(float* s_lut, unsigned int lane) {
#if METRALE_WARP_LUT_STAGED
    if (lane < 16u) s_lut[lane] = E2M1_LUT[lane];
    __syncwarp();
#else
    (void)s_lut; (void)lane;
#endif
}

// 2026-09-25: w4a16_gemv_partial: the partial sum of lane orig_lane (0..63) of a 64-thread
// output. It takes the 16-value chunks kk = 2 * orig_lane + c + 128j (c = 0, 1) into two
// accumulators: per chunk, part is a fmaf chain over the 16 values, then
// acc_c = fmaf(scale, part, acc_c). It returns acc0 + acc1. lut is the staged E2M1 copy.
//
// w4a16_gemv: M = 1, four outputs per 256-thread block (64 threads each), grid ceil(N/4)
// (W4A16_GEMV_OUTS_PER_BLOCK in model-layers ops/gemv_sw.rs). Each output's two warps
// reduce by shuffle, and the output is smem[2 * out] + smem[2 * out + 1].









__device__ __forceinline__ float w4a16_gemv_partial(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K16, unsigned int orig_lane,
    const float* __restrict__ lut)
{
    float acc0 = 0.0f, acc1 = 0.0f;
    const unsigned int stride2 = 128u;
    for (unsigned int k16 = orig_lane * 2u; k16 < K16 + 1u; k16 += stride2) {
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            const unsigned int kk = k16 + (unsigned int)c;
            if (kk >= K16) break;

            uint4 a_lo = ((const uint4*)A)[kk * 2];
            uint4 a_hi = ((const uint4*)A)[kk * 2 + 1];
            const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            unsigned long long packed8 = *(const unsigned long long*)(
                B_packed + (unsigned long long)n * half_K + kk * 8);
            unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            float part = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&a_raw[b]);
                part = fmaf(af.x, lut[byte_val & 0xF], part);
                part = fmaf(af.y, lut[byte_val >> 4], part);
            }
            if (c == 0) acc0 = fmaf(scale, part, acc0);
            else        acc1 = fmaf(scale, part, acc1);
        }
    }
    return acc0 + acc1;
}

extern "C" __global__ void w4a16_gemv(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    // 2026-09-25: No early return: threads of outputs past N must still reach the barrier.
    float acc = 0.0f;
    if (n < N) {
        acc = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane, s_lut);
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0 && n < N) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// 2026-09-25: w4a16_gemv_sw: w4a16_gemv with one warp per output, eight outputs per block,
// grid ceil(N/8) (W4A16_GEMV_SW_OUTS_PER_BLOCK). Lane l computes the partials of orig lanes
// l and l + 32, reduces each in w4a16_gemv's shuffle tree and adds the two: the same
// operations in the same order as w4a16_gemv. METRALE_NO_GEMV_SW=1 selects w4a16_gemv instead
// (gemv_sw.rs gemv_sw_from).










#define N_PER_BLOCK_SW 8

extern "C" __global__ void w4a16_gemv_sw(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int local_out = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;



    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_warp(s_lut[local_out], lane);
#if METRALE_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT;
#endif



    float acc_a = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane, warp_lut);
    float acc_b = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane + 32u, warp_lut);


    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }



    if (lane == 0) {
        float result = acc_a + acc_b;
        C[n] = __float2bfloat16(result);
    }
}

// 2026-09-25: w4a16_gemv_sw_moe: w4a16_gemv_sw for every routed slot in one launch, grid
// (ceil(N/8), top_k). Slot s runs expert expert_ids[s] through the per-expert pointer
// tables and writes row s of C; its input is A + s * input_stride (0 shares one row). A
// slot whose id is negative or >= num_experts, or whose packed_ptrs entry is 0 (an expert
// this rank does not own), writes nothing, so the caller's pre-zeroed row stays. Per slot
// the arithmetic is that of w4a16_gemv_sw.













extern "C" __global__ void w4a16_gemv_sw_moe(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_ids,
    unsigned int N,
    unsigned int K,
    unsigned int num_experts,
    unsigned int input_stride
) {
    const unsigned int slot = blockIdx.y;
    const int eid = expert_ids[slot];

    if (eid < 0 || (unsigned int)eid >= num_experts) return;
    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[eid];
    if (B_packed == 0) return;
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[eid];
    const float scale2 = scale2_vals[eid];

    const __nv_bfloat16* __restrict__ Ain =
        A + (unsigned long long)slot * (unsigned long long)input_stride;
    __nv_bfloat16* __restrict__ Cout = C + (unsigned long long)slot * (unsigned long long)N;

    const unsigned int local_out = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_warp(s_lut[local_out], lane);
#if METRALE_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT;
#endif

    float acc_a = w4a16_gemv_partial(Ain, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane, warp_lut);
    float acc_b = w4a16_gemv_partial(Ain, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane + 32u, warp_lut);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }

    if (lane == 0) {
        float result = acc_a + acc_b;
        Cout[n] = __float2bfloat16(result);
    }
}

// 2026-09-25: w4a16_gemv with an FP32 C, same grid. Its K walk differs: chunk k16 = lane +
// 64j into one accumulator, with the scale multiplied into each weight, so its sums are not
// bit-identical to w4a16_gemv's.



extern "C" __global__ void w4a16_gemv_logits(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    float* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
        unsigned long long packed8 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo_bf, a_hi_bf;
            *(unsigned short*)&a_lo_bf = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi_bf = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo_bf) * w_lo;
            acc += __bfloat162float(a_hi_bf) * w_hi;
        }
    }
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();
    if (lane == 0) {
        C[n] = smem[local_out * 2] + smem[local_out * 2 + 1];
    }
}





















// 2026-09-25: w4a16_gemv_batchm_impl<MAX_M>: w4a16_gemv for M <= MAX_M rows of A ([M, K],
// contiguous) with one weight read: each chunk's weight bytes and scale are loaded and
// unpacked once for every row. C is [M, N]; grid ceil(N/4), block 256.
//
// Each row's output is bit-identical to w4a16_gemv on that row. Physical thread p (0..63)
// walks the chunks p + 64 * phase + 128j, which are the chunks of reference lane
// p/2 + 32 * phase, accumulator p % 2, in the same order:
//   p + 64 * phase + 128j == 2 * (p/2 + 32 * phase) + p % 2 + 128j
// The two phases run one after the other on one acc[] array. Per chunk the row's part is
// the same fmaf chain, then acc = fmaf(scale, part, acc). Threads p and p ^ 1 then hold
// accumulators 0 and 1 of one reference lane; one __shfl_xor_sync adds them (FP32 addition
// commutes), the even thread stores the sum in s_vl at virtual lane p/2 + 32 * phase, and
// s_vl is reduced by w4a16_gemv's two-warp shuffle tree and smem sum.
//
// This walk keeps a warp's activation loads on consecutive chunks, 32 B apart; w4a16_gemv's
// own mapping (chunk 2l + c for lane l) would put them 64 B apart.







































































template <int MAX_M>
__device__ __forceinline__ void w4a16_gemv_batchm_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;




    float acc[MAX_M];
    __shared__ float s_vl[MAX_M][N_PER_BLOCK][2 * WARP_SIZE];

    #pragma unroll 1
    for (unsigned int phase = 0; phase < 2u; phase++) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;




        for (unsigned int kk = lane + phase * threads_per_out; kk < K16;
             kk += threads_per_out * 2u) {



            unsigned long long packed8 =
                *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk * 8);
            unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif




            float wl[16];
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                wl[b * 2]     = s_lut[byte_val & 0xF];
                wl[b * 2 + 1] = s_lut[byte_val >> 4];
            }

            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                const __nv_bfloat16* At = A + (unsigned long long)t * K;
                uint4 a_lo = ((const uint4*)At)[kk * 2];
                uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
                const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};


                float part = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    // 2026-09-25: ar[b] << 16 and ar[b] & 0xFFFF0000 are the FP32
                    // values of the low and high BF16 halves, the same values
                    // __bfloat1622float2 returns.














                    const float ax = __uint_as_float(ar[b] << 16);
                    const float ay = __uint_as_float(ar[b] & 0xFFFF0000u);
                    part = fmaf(ax, wl[b * 2], part);
                    part = fmaf(ay, wl[b * 2 + 1], part);
                }
                acc[t] = fmaf(scale, part, acc[t]);
            }
        }




        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            const float v = acc[t] + __shfl_xor_sync(0xFFFFFFFF, acc[t], 1);
            if ((lane & 1u) == 0u) {
                s_vl[t][local_out][phase * WARP_SIZE + (lane >> 1)] = v;
            }
        }
    }
    __syncthreads();

    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        float a = s_vl[t][local_out][lane];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        if (lane % WARP_SIZE == 0) smem[t][local_out * 2 + warp_in_out] = a;
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}












extern "C" __global__ void w4a16_gemv_batch2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<2>(A, B_packed, B_scale, scale2, C, 2u, N, K);
}




extern "C" __global__ void w4a16_gemv_batch3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<3>(A, B_packed, B_scale, scale2, C, 3u, N, K);
}

// 2026-10-07: METRALE_GLM_NV4_B3_STAGED. Same arithmetic as w4a16_gemv_batchm_impl<3> (so the
// same bits as w4a16_gemv_batch3 and, per row, the M = 1 kernel) with two changes that move no
// FP operation:
//  (1) The three BF16 activation rows are staged in shared memory once per block, one K window
//      at a time (NV4B3_WCH k16 chunks = 16 * NV4B3_WCH K), by cooperative uint4 loads. Each
//      thread then reads its 2 x uint4 per row from shared memory instead of global memory.
//      The window is a static 12 KiB x (NV4B3_WCH / 128) buffer, so no dynamic-smem opt-in is
//      needed for any K (K = 16384 is 128 chunks x 8 windows).
//  (2) The next iteration's 8-byte weight word and scale byte are loaded into registers before
//      the current iteration's FMAs.
// Order kept from the base kernel: thread (output, lane) walks chunks kk = lane + 64 * phase +
// 128 * j ascending, per phase its own accumulators acc[phase][t] (fmaf(scale, part, acc)),
// the per-chunk `part` chain over 16 elements in index order, then the shfl_xor / s_vl /
// shfl_down / smem reduction verbatim. NV4B3_WCH must be a multiple of 128 so a window holds
// whole strides of the chunk walk (a thread's chunk sequence per phase is unchanged).
// Threads with n >= N take no part in the math but reach every __syncthreads.
#ifndef NV4B3_WCH
#define NV4B3_WCH 128
#endif
#define NV4B3_JW (NV4B3_WCH / 128)

extern "C" __global__ void w4a16_gemv_batch3_staged(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    static_assert(NV4B3_WCH % 128 == 0, "window must hold whole 128-chunk strides");
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;  // uniform per warp (64 threads share an output)

    __shared__ float s_lut[16];
    __shared__ uint4 s_a[3][2][NV4B3_WCH];  // [row][lo/hi uint4 of the chunk][chunk in window]
    __shared__ float s_vl[3][N_PER_BLOCK][2 * WARP_SIZE];
    __shared__ float smem[3][N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;
    const unsigned int nwin = (K16 + NV4B3_WCH - 1) / NV4B3_WCH;

    const unsigned char* wrow = B_packed + (unsigned long long)(valid ? n : 0u) * half_K;
    const unsigned char* srow = B_scale + (unsigned long long)(valid ? n : 0u) * num_groups;

    // Register-held weight word and scale byte of the current iteration; first = (win 0, ph 0, j 0).
    unsigned long long cur_w = 0ull;
    unsigned char cur_s = 0;
    if (valid && lane < K16) {
        cur_w = *(const unsigned long long*)(wrow + (unsigned long long)lane * 8);
        cur_s = srow[lane];
    }

    float acc[2][3];
    #pragma unroll
    for (int p = 0; p < 2; p++) {
        #pragma unroll
        for (int t = 0; t < 3; t++) acc[p][t] = 0.0f;
    }

    #pragma unroll 1
    for (unsigned int win = 0; win < nwin; win++) {
        const unsigned int base = win * NV4B3_WCH;
        const unsigned int cnt = min((unsigned int)NV4B3_WCH, K16 - base);
        // Cooperative staging: coalesced uint4 loads, 2 uint4 (= 16 BF16) per chunk per row.
        for (unsigned int u = threadIdx.x; u < cnt * 2u; u += BLOCK_SIZE) {
            #pragma unroll
            for (int t = 0; t < 3; t++) {
                const uint4 v = ((const uint4*)(A + (unsigned long long)t * K))[(unsigned long long)base * 2 + u];
                s_a[t][u & 1u][u >> 1] = v;
            }
        }
        __syncthreads();  // staged window (and s_lut on the first pass) visible

        if (valid) {
            #pragma unroll
            for (int ph = 0; ph < 2; ph++) {
                #pragma unroll
                for (int j = 0; j < NV4B3_JW; j++) {
                    const unsigned int kk = base + lane + ph * threads_per_out + 128u * j;
                    // Prefetch the next iteration (next slot, or slot 0 of the next window).
                    const int ns = ph * NV4B3_JW + j + 1;
                    unsigned int nk;
                    if (ns < 2 * NV4B3_JW) nk = base + lane + (ns / NV4B3_JW) * threads_per_out + 128u * (ns % NV4B3_JW);
                    else                   nk = base + NV4B3_WCH + lane;
                    unsigned long long nxt_w = 0ull;
                    unsigned char nxt_s = 0;
                    if (nk < K16) {
                        nxt_w = *(const unsigned long long*)(wrow + (unsigned long long)nk * 8);
                        nxt_s = srow[nk];
                    }
                    if (kk < K16) {
                        __nv_fp8_e4m3 fp8;
                        *(unsigned char*)&fp8 = cur_s;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
                        float scale = scl_fp8(cur_s) * scale2;
#else
                        float scale = (float)fp8 * scale2;
#endif
                        float wl[16];
                        #pragma unroll
                        for (int b = 0; b < 8; b++) {
                            unsigned char byte_val = (unsigned char)(cur_w >> (b * 8));
                            wl[b * 2]     = s_lut[byte_val & 0xF];
                            wl[b * 2 + 1] = s_lut[byte_val >> 4];
                        }
                        const unsigned int c = kk - base;
                        #pragma unroll
                        for (int t = 0; t < 3; t++) {
                            const uint4 a_lo = s_a[t][0][c];
                            const uint4 a_hi = s_a[t][1][c];
                            const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
                            float part = 0.0f;
                            #pragma unroll
                            for (int b = 0; b < 8; b++) {
                                const float ax = __uint_as_float(ar[b] << 16);
                                const float ay = __uint_as_float(ar[b] & 0xFFFF0000u);
                                part = fmaf(ax, wl[b * 2], part);
                                part = fmaf(ay, wl[b * 2 + 1], part);
                            }
                            acc[ph][t] = fmaf(scale, part, acc[ph][t]);
                        }
                    }
                    cur_w = nxt_w;
                    cur_s = nxt_s;
                }
            }
        }
        __syncthreads();  // window consumed before the next staging overwrites it
    }

    if (valid) {
        #pragma unroll
        for (int ph = 0; ph < 2; ph++) {
            #pragma unroll
            for (int t = 0; t < 3; t++) {
                const float v = acc[ph][t] + __shfl_xor_sync(0xFFFFFFFF, acc[ph][t], 1);
                if ((lane & 1u) == 0u) {
                    s_vl[t][local_out][ph * WARP_SIZE + (lane >> 1)] = v;
                }
            }
        }
    }
    __syncthreads();

    const unsigned int warp_in_out = lane / WARP_SIZE;
    if (valid) {
        #pragma unroll
        for (int t = 0; t < 3; t++) {
            float a = s_vl[t][local_out][lane];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                a += __shfl_down_sync(0xFFFFFFFF, a, offset);
            }
            if (lane % WARP_SIZE == 0) smem[t][local_out * 2 + warp_in_out] = a;
        }
    }
    __syncthreads();

    if (valid && lane == 0) {
        #pragma unroll
        for (int t = 0; t < 3; t++) {
            float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}


// 2026-10-07: METRALE_GLM_NV4_BATCHM_STAGED. w4a16_gemv_batchm_impl<MAX_M> (rows t < M) with the
// BF16 activation rows staged in shared memory, as w4a16_gemv_batch3_staged does for 3 rows; same
// FP operations in the same order, so each row's bits equal the tier's (w4a16_gemv_batch{4..8,16})
// and the M = 1 kernel's. Without staging, every thread reloads M x 32 B of activations per 9 B of
// weight, so the tiers slow with M (ra-t22 dense microtest, KDA 4096^2: M=1 42.6 us, M=8 72.2 us;
// in the C4 serve batch16 at 12 rows averages 88.6 us vs batch4 47.8 us).
// The window is one 64-chunk phase slice: window (pair pr, phase ph) holds chunks
// [128 pr + 64 ph, +64), and thread `lane` of an output takes chunk 128 pr + 64 ph + lane, which
// is the base kernel's walk kk = lane + 64 phase + 128 j (j = pr) in ascending j into the phase's
// own accumulators. The next chunk's weight word and scale byte are loaded before the current
// FMAs. The windows (MAX_M x 2 KiB) and the reduction scratch s_vl (MAX_M x 1 KiB) share one
// buffer, which is dead after the walk's last barrier: 32 KiB at MAX_M = 16, no dynamic-smem
// opt-in. Threads with n >= N take no part in the math but reach every __syncthreads.
#define NV4BM_WCH 64
template <int MAX_M>
__device__ __forceinline__ void w4a16_gemv_batchm_staged_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64 = NV4BM_WCH
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;  // uniform per warp (64 threads share an output)
    const unsigned int rows = M < (unsigned int)MAX_M ? M : (unsigned int)MAX_M;

    constexpr unsigned int A_BYTES = MAX_M * 2 * NV4BM_WCH * 16;
    constexpr unsigned int VL_BYTES = MAX_M * N_PER_BLOCK * 2 * WARP_SIZE * 4;
    __shared__ __align__(16) unsigned char s_buf[A_BYTES > VL_BYTES ? A_BYTES : VL_BYTES];
    uint4 (*s_a)[2][NV4BM_WCH] = reinterpret_cast<uint4 (*)[2][NV4BM_WCH]>(s_buf);
    float (*s_vl)[N_PER_BLOCK][2 * WARP_SIZE] =
        reinterpret_cast<float (*)[N_PER_BLOCK][2 * WARP_SIZE]>(s_buf);
    __shared__ float s_lut[16];
    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;
    const unsigned int npair = (K16 + 2u * NV4BM_WCH - 1u) / (2u * NV4BM_WCH);
    const unsigned char* wrow = B_packed + (unsigned long long)(valid ? n : 0u) * half_K;
    const unsigned char* srow = B_scale + (unsigned long long)(valid ? n : 0u) * num_groups;

    unsigned long long cur_w = 0ull;
    unsigned char cur_s = 0;
    if (valid && lane < K16) {
        cur_w = *(const unsigned long long*)(wrow + (unsigned long long)lane * 8);
        cur_s = srow[lane];
    }
    float acc[2][MAX_M];
    #pragma unroll
    for (int p = 0; p < 2; p++) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) acc[p][t] = 0.0f;
    }

    #pragma unroll 1
    for (unsigned int pr = 0; pr < npair; pr++) {
        #pragma unroll
        for (int ph = 0; ph < 2; ph++) {
            const unsigned int base = pr * (2u * NV4BM_WCH) + (unsigned int)ph * NV4BM_WCH;
            if (base >= K16) break;  // uniform: the last pair may have one window
            const unsigned int cnt = min((unsigned int)NV4BM_WCH, K16 - base);
            const unsigned int per = cnt * 2u;  // uint4 per row in this window
            // Cooperative staging: coalesced uint4 loads, 2 uint4 (= 16 BF16) per chunk per row.
            for (unsigned int u = threadIdx.x; u < rows * per; u += BLOCK_SIZE) {
                const unsigned int t = u / per;
                const unsigned int r = u - t * per;
                s_a[t][r & 1u][r >> 1] =
                    ((const uint4*)(A + (unsigned long long)t * K))[(unsigned long long)base * 2 + r];
            }
            __syncthreads();  // staged window (and s_lut on the first pass) visible
            const unsigned int kk = base + lane;
            const unsigned int nk = kk + NV4BM_WCH;  // this thread's next chunk (next window)
            unsigned long long nxt_w = 0ull;
            unsigned char nxt_s = 0;
            if (valid && nk < K16) {
                nxt_w = *(const unsigned long long*)(wrow + (unsigned long long)nk * 8);
                nxt_s = srow[nk];
            }
            if (valid && kk < K16) {
                __nv_fp8_e4m3 fp8;
                *(unsigned char*)&fp8 = cur_s;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
                float scale = scl_fp8(cur_s) * scale2;
#else
                float scale = (float)fp8 * scale2;
#endif
                float wl[16];
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    unsigned char byte_val = (unsigned char)(cur_w >> (b * 8));
                    wl[b * 2]     = s_lut[byte_val & 0xF];
                    wl[b * 2 + 1] = s_lut[byte_val >> 4];
                }
                #pragma unroll
                for (int t = 0; t < MAX_M; t++) {
                    if ((unsigned int)t >= M) continue;
                    const uint4 a_lo = s_a[t][0][lane];
                    const uint4 a_hi = s_a[t][1][lane];
                    const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                                a_hi.x, a_hi.y, a_hi.z, a_hi.w};
                    float part = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < 8; b++) {
                        const float ax = __uint_as_float(ar[b] << 16);
                        const float ay = __uint_as_float(ar[b] & 0xFFFF0000u);
                        part = fmaf(ax, wl[b * 2], part);
                        part = fmaf(ay, wl[b * 2 + 1], part);
                    }
                    acc[ph][t] = fmaf(scale, part, acc[ph][t]);
                }
            }
            cur_w = nxt_w;
            cur_s = nxt_s;
            __syncthreads();  // window consumed before the next staging (or s_vl) overwrites it
        }
    }

    if (valid) {
        #pragma unroll
        for (int ph = 0; ph < 2; ph++) {
            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                const float v = acc[ph][t] + __shfl_xor_sync(0xFFFFFFFF, acc[ph][t], 1);
                if ((lane & 1u) == 0u) {
                    s_vl[t][local_out][ph * WARP_SIZE + (lane >> 1)] = v;
                }
            }
        }
    }
    __syncthreads();
    const unsigned int warp_in_out = lane / WARP_SIZE;
    if (valid) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float a = s_vl[t][local_out][lane];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                a += __shfl_down_sync(0xFFFFFFFF, a, offset);
            }
            if (lane % WARP_SIZE == 0) smem[t][local_out * 2 + warp_in_out] = a;
        }
    }
    __syncthreads();
    if (valid && lane == 0) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}

// 2026-10-07: The staged tiers; same arguments as w4a16_gemv_batch{4..8,16}.
#define METRALE_NV4BM_STAGED_ENTRY(R)                                                \
extern "C" __global__ void w4a16_gemv_batch##R##_staged(                           \
    const __nv_bfloat16* __restrict__ A,                                           \
    const unsigned char* __restrict__ B_packed,                                    \
    const unsigned char* __restrict__ B_scale,                                     \
    const float scale2,                                                            \
    __nv_bfloat16* __restrict__ C,                                                 \
    unsigned int M,                                                                \
    unsigned int N,                                                                \
    unsigned int K                                                                 \
) {                                                                                \
    w4a16_gemv_batchm_staged_impl<R>(A, B_packed, B_scale, scale2, C, M, N, K);    \
}

METRALE_NV4BM_STAGED_ENTRY(4)
METRALE_NV4BM_STAGED_ENTRY(5)
METRALE_NV4BM_STAGED_ENTRY(6)
METRALE_NV4BM_STAGED_ENTRY(7)
METRALE_NV4BM_STAGED_ENTRY(8)
METRALE_NV4BM_STAGED_ENTRY(16)

extern "C" __global__ void w4a16_gemv_batch4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<4>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// 2026-09-25: Exact-M tiers 5, 6 and 7. MAX_M sizes acc[], s_vl[] and smem[], and with the
// row loop unrolled it also sizes the code: the t >= M guard skips a dead row's work but
// not its instructions. The model-layers w4a16_gemv_tiers.rs picks the narrowest resolved
// tier >= M; METRALE_NO_GEMV_EXACT_M_TIERS, set to any value, removes 5, 6 and 7 from the
// choice. Per row the arithmetic does not depend on MAX_M, so every tier gives the same
// bits at the same M.







































extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch5(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<5>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch6(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<6>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch7(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<7>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// 2026-09-25: M <= 8. For width 8, W4a16BatchmTiers::resolve takes
// w4a16_gemv_batch8_rt2 when it resolves (gemv_tier.rs batch8_kernel;
// METRALE_NO_BATCH8_RT=1 declines it), and this kernel otherwise.













extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch8(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<8>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// 2026-09-25: Weight-prefetch variant of w4a16_gemv_batchm_impl.
// provenance-id: 526f6e616c6420522e205374657369616b
//
// The next chunk's weight bytes and scale are loaded before the current chunk's FMAs. The
// chunk order, per-chunk FMA chain, fold and reduction are those of
// w4a16_gemv_batchm_impl, so each output has the same bits. Only batchm_bench
// (model-arch examples) launches w4a16_gemv_batch8_pf and w4a16_gemv_batch8_pf_free.








template <int MAX_M>
__device__ __forceinline__ void w4a16_gemv_batchm_impl_pf(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    float acc[MAX_M];
    __shared__ float s_vl[MAX_M][N_PER_BLOCK][2 * WARP_SIZE];

    #pragma unroll 1
    for (unsigned int phase = 0; phase < 2u; phase++) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;



        unsigned int kk = lane + phase * threads_per_out;
        unsigned long long packed8 = 0;
        unsigned char scale_byte = 0;
        if (kk < K16) {
            packed8 =
                *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk * 8);
            scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
        }

        while (kk < K16) {


            const unsigned int kn = kk + threads_per_out * 2u;
            unsigned long long packed8_n = 0;
            unsigned char scale_byte_n = 0;
            if (kn < K16) {
                packed8_n = *(const unsigned long long*)(B_packed +
                                                         (unsigned long long)n * half_K + kn * 8);
                scale_byte_n = B_scale[(unsigned long long)n * num_groups + kn];
            }

            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            float wl[16];
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                wl[b * 2]     = s_lut[byte_val & 0xF];
                wl[b * 2 + 1] = s_lut[byte_val >> 4];
            }

            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                const __nv_bfloat16* At = A + (unsigned long long)t * K;
                uint4 a_lo = ((const uint4*)At)[kk * 2];
                uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
                const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};

                float part = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&ar[b]);
                    part = fmaf(af.x, wl[b * 2], part);
                    part = fmaf(af.y, wl[b * 2 + 1], part);
                }
                acc[t] = fmaf(scale, part, acc[t]);
            }

            kk = kn;
            packed8 = packed8_n;
            scale_byte = scale_byte_n;
        }


        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            const float v = acc[t] + __shfl_xor_sync(0xFFFFFFFF, acc[t], 1);
            if ((lane & 1u) == 0u) {
                s_vl[t][local_out][phase * WARP_SIZE + (lane >> 1)] = v;
            }
        }
    }
    __syncthreads();

    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        float a = s_vl[t][local_out][lane];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        if (lane % WARP_SIZE == 0) smem[t][local_out * 2 + warp_in_out] = a;
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}




extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch8_pf(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl_pf<8>(A, B_packed, B_scale, scale2, C, M, N, K);
}


extern "C" __global__ void w4a16_gemv_batch8_pf_free(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl_pf<8>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// 2026-09-25: Activation row-ahead variant of w4a16_gemv_batchm_impl.
// provenance-id: 526f6e616c6420522e205374657369616b
//
// While row t's FMA chain runs, row t + 1's two activation loads are issued; row 0's loads
// go out before the E2M1 unpack. WPF adds the weight prefetch of
// w4a16_gemv_batchm_impl_pf. The per-row arithmetic is unchanged, so each output has the
// same bits as w4a16_gemv_batchm_impl. Only batchm_bench launches w4a16_gemv_batch8_pf2
// and w4a16_gemv_batch8_pf3.









template <int MAX_M, bool WPF>
__device__ __forceinline__ void w4a16_gemv_batchm_impl_apf(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    float acc[MAX_M];
    __shared__ float s_vl[MAX_M][N_PER_BLOCK][2 * WARP_SIZE];

    #pragma unroll 1
    for (unsigned int phase = 0; phase < 2u; phase++) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

        unsigned long long packed8 = 0;
        unsigned char scale_byte = 0;
        if (WPF) {
            const unsigned int kk0 = lane + phase * threads_per_out;
            if (kk0 < K16) {
                packed8 = *(const unsigned long long*)(B_packed +
                                                       (unsigned long long)n * half_K + kk0 * 8);
                scale_byte = B_scale[(unsigned long long)n * num_groups + kk0];
            }
        }

        for (unsigned int kk = lane + phase * threads_per_out; kk < K16;
             kk += threads_per_out * 2u) {
            if (WPF) {

            } else {
                packed8 = *(const unsigned long long*)(B_packed +
                                                       (unsigned long long)n * half_K + kk * 8);
                scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
            }


            uint4 a_lo_c = ((const uint4*)A)[kk * 2];
            uint4 a_hi_c = ((const uint4*)A)[kk * 2 + 1];

            unsigned long long packed8_n = 0;
            unsigned char scale_byte_n = 0;
            if (WPF) {
                const unsigned int kn = kk + threads_per_out * 2u;
                if (kn < K16) {
                    packed8_n = *(const unsigned long long*)(B_packed +
                                                             (unsigned long long)n * half_K +
                                                             kn * 8);
                    scale_byte_n = B_scale[(unsigned long long)n * num_groups + kn];
                }
            }

            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            float wl[16];
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                wl[b * 2]     = s_lut[byte_val & 0xF];
                wl[b * 2 + 1] = s_lut[byte_val >> 4];
            }

            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;


                uint4 a_lo_n = a_lo_c;
                uint4 a_hi_n = a_hi_c;
                if (t + 1 < MAX_M && (unsigned int)(t + 1) < M) {
                    const __nv_bfloat16* At_n = A + (unsigned long long)(t + 1) * K;
                    a_lo_n = ((const uint4*)At_n)[kk * 2];
                    a_hi_n = ((const uint4*)At_n)[kk * 2 + 1];
                }
                const unsigned int ar[8] = {a_lo_c.x, a_lo_c.y, a_lo_c.z, a_lo_c.w,
                                            a_hi_c.x, a_hi_c.y, a_hi_c.z, a_hi_c.w};

                float part = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&ar[b]);
                    part = fmaf(af.x, wl[b * 2], part);
                    part = fmaf(af.y, wl[b * 2 + 1], part);
                }
                acc[t] = fmaf(scale, part, acc[t]);
                a_lo_c = a_lo_n;
                a_hi_c = a_hi_n;
            }

            if (WPF) {
                packed8 = packed8_n;
                scale_byte = scale_byte_n;
            }
        }


        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            const float v = acc[t] + __shfl_xor_sync(0xFFFFFFFF, acc[t], 1);
            if ((lane & 1u) == 0u) {
                s_vl[t][local_out][phase * WARP_SIZE + (lane >> 1)] = v;
            }
        }
    }
    __syncthreads();

    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        float a = s_vl[t][local_out][lane];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        if (lane % WARP_SIZE == 0) smem[t][local_out * 2 + warp_in_out] = a;
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}

// 2026-09-25: Register-tiled variant of w4a16_gemv_batchm_impl.
// provenance-id: 526f6e616c6420522e205374657369616b
//
// Each 64-thread group computes T adjacent outputs n0 .. n0 + T - 1,
// n0 = (blockIdx.x * 4 + group) * T, so one activation load per (chunk, row) feeds T FMA
// chains. Each output's chunk walk, FMA chain, fold and reduction are those of
// w4a16_gemv_batchm_impl, so its bits are the same. A block covers 4 * T outputs;
// ops::w4a16_gemv_batchm launches w4a16_gemv_batch8_rt2 with grid ceil(N/4), and the
// surplus blocks return on n0 >= N. Only batchm_bench launches w4a16_gemv_batch8_rt4.






template <int MAX_M, int T>
__device__ __forceinline__ void w4a16_gemv_batchm_impl_rt(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n0 = (blockIdx.x * N_PER_BLOCK + local_out) * T;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    if (n0 >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    float acc[T][MAX_M];


    __shared__ float s_vl[MAX_M][N_PER_BLOCK][T][2 * WARP_SIZE];

    #pragma unroll 1
    for (unsigned int phase = 0; phase < 2u; phase++) {
        #pragma unroll
        for (int o = 0; o < T; o++)
            #pragma unroll
            for (int t = 0; t < MAX_M; t++) acc[o][t] = 0.0f;

        for (unsigned int kk = lane + phase * threads_per_out; kk < K16;
             kk += threads_per_out * 2u) {

            unsigned long long packed8[T];
            float scale[T];
            #pragma unroll
            for (int o = 0; o < T; o++) {
                const unsigned long long n = n0 + o;
                if (n < N) {
                    packed8[o] = *(const unsigned long long*)(B_packed + n * half_K + kk * 8);
                    const unsigned char sb = B_scale[n * num_groups + kk];
                    __nv_fp8_e4m3 fp8;
                    *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
                    scale[o] = scl_fp8(sb) * scale2;
#else
                    scale[o] = (float)fp8 * scale2;
#endif
                } else {
                    packed8[o] = 0ull;
                    scale[o] = 0.0f;
                }
            }
            float wl[T][16];
            #pragma unroll
            for (int o = 0; o < T; o++) {
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    unsigned char byte_val = (unsigned char)(packed8[o] >> (b * 8));
                    wl[o][b * 2]     = s_lut[byte_val & 0xF];
                    wl[o][b * 2 + 1] = s_lut[byte_val >> 4];
                }
            }

            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                const __nv_bfloat16* At = A + (unsigned long long)t * K;

                uint4 a_lo = ((const uint4*)At)[kk * 2];
                uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
                const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};
                #pragma unroll
                for (int o = 0; o < T; o++) {

                    float part = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < 8; b++) {
                        float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&ar[b]);
                        part = fmaf(af.x, wl[o][b * 2], part);
                        part = fmaf(af.y, wl[o][b * 2 + 1], part);
                    }
                    acc[o][t] = fmaf(scale[o], part, acc[o][t]);
                }
            }
        }


        #pragma unroll
        for (int o = 0; o < T; o++) {
            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                const float v = acc[o][t] + __shfl_xor_sync(0xFFFFFFFF, acc[o][t], 1);
                if ((lane & 1u) == 0u) {
                    s_vl[t][local_out][o][phase * WARP_SIZE + (lane >> 1)] = v;
                }
            }
        }
    }
    __syncthreads();

    __shared__ float smem[MAX_M][N_PER_BLOCK * T * 2];
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int o = 0; o < T; o++) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float a = s_vl[t][local_out][o][lane];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                a += __shfl_down_sync(0xFFFFFFFF, a, offset);
            }
            if (lane % WARP_SIZE == 0)
                smem[t][(local_out * T + o) * 2 + warp_in_out] = a;
        }
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int o = 0; o < T; o++) {
            const unsigned int n = n0 + o;
            if (n >= N) continue;
            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                float r = smem[t][(local_out * T + o) * 2]
                        + smem[t][(local_out * T + o) * 2 + 1];
                C[(unsigned long long)t * N + n] = __float2bfloat16(r);
            }
        }
    }
}


extern "C" __global__ void w4a16_gemv_batch8_rt2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl_rt<8, 2>(A, B_packed, B_scale, scale2, C, M, N, K);
}


extern "C" __global__ void w4a16_gemv_batch8_rt4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl_rt<8, 4>(A, B_packed, B_scale, scale2, C, M, N, K);
}


extern "C" __global__ void w4a16_gemv_batch8_pf2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl_apf<8, false>(A, B_packed, B_scale, scale2, C, M, N, K);
}


extern "C" __global__ void w4a16_gemv_batch8_pf3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl_apf<8, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}


extern "C" __global__ void w4a16_gemv_batch16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<16>(A, B_packed, B_scale, scale2, C, M, N, K);
}







extern "C" __global__ void w4a16_gemv_batch32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<32>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// 2026-09-25: w4a16_gemv_qg: M = 1 over a weight whose rows interleave Q and gate per head,
// [Q_h0, G_h0, Q_h1, ...], each head_dim rows; output n is stored at its place in
// [Q of all heads | G of all heads], N = num_heads * head_dim * 2. Four outputs per block,
// grid ceil(N/4). Its K walk is the 8-value chunks k8 = lane + 64j with
// acc += a * (lut * scale) per value, not w4a16_gemv's.







extern "C" __global__ void w4a16_gemv_qg(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / WARP_SIZE)] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];



        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx] = __float2bfloat16(result);
    }
}

// 2026-09-25: w4a16_gemv_qkvz: the K walk of w4a16_gemv_qg, with output n stored at its place
// in [Q | K | V | Z]. The weight rows are num_groups groups of (Q_g and K_g of head_k_dim,
// V_g and Z_g of vheads_per_group * head_v_dim), the layout deinterleave_qkvz
// (ssm_preprocess.cu) takes. Grid ceil(N/4).








extern "C" __global__ void w4a16_gemv_qkvz(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,

    unsigned int num_groups,
    unsigned int head_k_dim,
    unsigned int vheads_per_group,
    unsigned int head_v_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups_k = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups_k + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / WARP_SIZE)] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];


        unsigned int v_group_size = vheads_per_group * head_v_dim;
        unsigned int group_dim = 2 * head_k_dim + 2 * v_group_size;
        unsigned int g = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_groups * head_k_dim;
        unsigned int k_total = num_groups * head_k_dim;

        unsigned int out_idx;
        if (idx < head_k_dim) {
            out_idx = g * head_k_dim + idx;
        } else if (idx < 2 * head_k_dim) {
            out_idx = q_total + g * head_k_dim + (idx - head_k_dim);
        } else if (idx < 2 * head_k_dim + v_group_size) {
            out_idx = q_total + k_total + g * v_group_size + (idx - 2 * head_k_dim);
        } else {
            out_idx = q_total + k_total + num_groups * v_group_size
                    + g * v_group_size + (idx - 2 * head_k_dim - v_group_size);
        }
        C[out_idx] = __float2bfloat16(result);
    }
}

// 2026-09-25: w4a16_gemv_qg for two rows (A [2, K], C [2, N]) over one weight read. Per row
// the operations and their order are those of w4a16_gemv_qg (the same k8 walk,
// w = lut * scale and two separate acc += per byte), so on gb10, which builds this file
// with --fmad=false (kernels/gb10/common/KERNEL.toml), each row matches w4a16_gemv_qg bit
// for bit. Grid ceil(N/4).




















extern "C" __global__ void w4a16_gemv_qg_batch2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    __nv_bfloat16* __restrict__ C1 = C + N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];


        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx]  = __float2bfloat16(result0);
        C1[out_idx] = __float2bfloat16(result1);
    }
}

// 2026-09-25: w4a16_gemv_dual_batch2: two weights over the same two-row input A [2, K_in];
// blockIdx.z picks weight 0 or 1, and each writes its own C [2, N]. Grid (ceil(N/4), 1, 2).
// Per row it is w4a16_gemv_qg's arithmetic without the deinterleave.







extern "C" __global__ void w4a16_gemv_dual_batch2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B0_packed,
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,
    unsigned int N,
    unsigned int K_in
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = (proj == 0) ? B0_packed : B1_packed;
    const unsigned char* B_scale = (proj == 0) ? B0_scale : B1_scale;
    float s2 = (proj == 0) ? B0_scale2 : B1_scale2;
    __nv_bfloat16* C_out = (proj == 0) ? C0 : C1;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K8 = K_in / 8;

    const __nv_bfloat16* A1 = A + K_in;
    __nv_bfloat16* C_out1 = C_out + N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(sb) * s2;
#else
        float scale = (float)fp8 * s2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];
        C_out[n]  = __float2bfloat16(result0);
        C_out1[n] = __float2bfloat16(result1);
    }
}




















// 2026-09-25: w4a16_gemv_qg_batch2 for three rows (A [3, K], C [3, N]).










extern "C" __global__ void w4a16_gemv_qg_batch3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    const __nv_bfloat16* __restrict__ A2 = A + 2 * K;
    __nv_bfloat16* __restrict__ C1 = C + N;
    __nv_bfloat16* __restrict__ C2 = C + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo;
            acc2 += __bfloat162float(a2_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];

        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx]  = __float2bfloat16(result0);
        C1[out_idx] = __float2bfloat16(result1);
        C2[out_idx] = __float2bfloat16(result2);
    }
}

// 2026-09-25: w4a16_gemv_dual_batch2 for three rows (A [3, K_in], each C [3, N]).









extern "C" __global__ void w4a16_gemv_dual_batch3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B0_packed,
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,
    unsigned int N,
    unsigned int K_in
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = (proj == 0) ? B0_packed : B1_packed;
    const unsigned char* B_scale = (proj == 0) ? B0_scale : B1_scale;
    float s2 = (proj == 0) ? B0_scale2 : B1_scale2;
    __nv_bfloat16* C_out = (proj == 0) ? C0 : C1;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K8 = K_in / 8;

    const __nv_bfloat16* A1 = A + K_in;
    const __nv_bfloat16* A2 = A + 2 * K_in;
    __nv_bfloat16* C_out1 = C_out + N;
    __nv_bfloat16* C_out2 = C_out + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(sb) * s2;
#else
        float scale = (float)fp8 * s2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo;
            acc2 += __bfloat162float(a2_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];
        C_out[n]  = __float2bfloat16(result0);
        C_out1[n] = __float2bfloat16(result1);
        C_out2[n] = __float2bfloat16(result2);
    }
}

// 2026-09-25: Routed experts for several rows with one sweep per expert: for the rows handled
// together, glm5next_moe_row_union builds the union of the experts they selected, on
// device, and w4a16_gemv_sw_moe_batchm_m<R> reads each union expert's weights once for all
// of the R rows that chose it. Per (row, slot) the arithmetic is that of
// w4a16_gemv_sw_moe (the same partial per orig lane, shuffle tree and two-term sum); only
// the weight load is shared.

















// 2026-09-26: Union of ids [rows, top_k]: u_eid[u] is the expert of union entry u (-1 when
// unused), and u_slot[u * rows + r] the slot row r gave it (-1 when row r did not select
// it). Entries follow the first appearance of each id in row-major order. It needs one
// block of at least rows * top_k threads; glm5next_mlp/forward/moe_experts.rs launches exactly
// that, and forward.rs takes this path only while rows * top_k <= MOE_ROW_UNION_MAX_IDS (64).


extern "C" __global__ void glm5next_moe_row_union(
    const int* __restrict__ ids,
    int* __restrict__ u_eid,
    int* __restrict__ u_slot,
    unsigned int rows,
    unsigned int top_k
) {
    const unsigned int T = rows * top_k;
    const unsigned int t = threadIdx.x;

    // 2026-09-25: Clear the tables before the barrier. Every thread must reach it, so the range
    // check is a predicate here and the early return comes after.

    if (t < T) {
        u_eid[t] = -1;
        for (unsigned int r = 0; r < rows; r++) u_slot[t * rows + r] = -1;
    }
    __syncthreads();
    if (t >= T) return;

    const int eid = ids[t];
    if (eid < 0) return;

    // 2026-09-25: Only the thread of an id's first occurrence writes its entry.
    for (unsigned int tp = 0; tp < t; tp++) {
        if (ids[tp] == eid) return;
    }

    // 2026-09-25: Its union index is the number of first occurrences before t.
    int uidx = 0;
    for (unsigned int tp = 0; tp < t; tp++) {
        const int e2 = ids[tp];
        if (e2 < 0) continue;
        bool owner2 = true;
        for (unsigned int tq = 0; tq < tp; tq++) {
            if (ids[tq] == e2) { owner2 = false; break; }
        }
        if (owner2) uidx++;
    }

    u_eid[uidx] = eid;

    for (unsigned int tp = t; tp < T; tp++) {
        if (ids[tp] == eid) u_slot[uidx * rows + tp / top_k] = (int)(tp % top_k);
    }
}

// 2026-10-08: METRALE_GLM_MOE_UNION_SCAN=1: glm5next_moe_row_union with the same launch shape
// and the same u_eid / u_slot, entry for entry. ids are staged in shared memory once, each
// position's first-occurrence flag is computed from them, and a first occurrence's union index
// is the count of flags before it (the old kernel recomputed every earlier flag per thread
// from global memory, O(T^3) loads at T = 64). T = rows * top_k <= 64.

#define MOE_ROW_UNION_SCAN_MAX 64

extern "C" __global__ void glm5next_moe_row_union_scan(
    const int* __restrict__ ids,
    int* __restrict__ u_eid,
    int* __restrict__ u_slot,
    unsigned int rows,
    unsigned int top_k
) {
    __shared__ int s_ids[MOE_ROW_UNION_SCAN_MAX];
    __shared__ int s_first[MOE_ROW_UNION_SCAN_MAX];
    const unsigned int T = rows * top_k;
    const unsigned int t = threadIdx.x;

    if (t < T) {
        s_ids[t] = ids[t];
        u_eid[t] = -1;
        for (unsigned int r = 0; r < rows; r++) u_slot[t * rows + r] = -1;
    }
    __syncthreads();

    int eid = -1;
    int first = 0;
    if (t < T) {
        eid = s_ids[t];
        first = eid >= 0;
        for (unsigned int tp = 0; tp < t && first; tp++) {
            if (s_ids[tp] == eid) first = 0;
        }
        s_first[t] = first;
    }
    __syncthreads();
    if (!first) return;

    int uidx = 0;
    for (unsigned int tp = 0; tp < t; tp++) uidx += s_first[tp];

    u_eid[uidx] = eid;
    for (unsigned int tp = t; tp < T; tp++) {
        if (s_ids[tp] == eid) u_slot[uidx * rows + tp / top_k] = (int)(tp % top_k);
    }
}

// 2026-09-25: w4a16_gemv_partial for R rows over one weight read. Aptr[r] == nullptr skips
// row r (not read, contributes nothing); every other row gets w4a16_gemv_partial's result.

template <int R>
__device__ __forceinline__ void w4a16_gemv_partial_rows(
    const __nv_bfloat16* const* __restrict__ Aptr,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K16, unsigned int orig_lane,
    const float* __restrict__ lut,
    float* __restrict__ out)
{
    float acc0[R], acc1[R];
    #pragma unroll
    for (int r = 0; r < R; r++) { acc0[r] = 0.0f; acc1[r] = 0.0f; }

    const unsigned int stride2 = 128u;
    for (unsigned int k16 = orig_lane * 2u; k16 < K16 + 1u; k16 += stride2) {
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            const unsigned int kk = k16 + (unsigned int)c;
            if (kk >= K16) break;


            unsigned long long packed8 = *(const unsigned long long*)(
                B_packed + (unsigned long long)n * half_K + kk * 8);
            unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            #pragma unroll
            for (int r = 0; r < R; r++) {
                if (Aptr[r] == nullptr) continue;
                uint4 a_lo = ((const uint4*)Aptr[r])[kk * 2];
                uint4 a_hi = ((const uint4*)Aptr[r])[kk * 2 + 1];
                const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                               a_hi.x, a_hi.y, a_hi.z, a_hi.w};
                float part = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                    float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&a_raw[b]);
                    part = fmaf(af.x, lut[byte_val & 0xF], part);
                    part = fmaf(af.y, lut[byte_val >> 4], part);
                }
                if (c == 0) acc0[r] = fmaf(scale, part, acc0[r]);
                else        acc1[r] = fmaf(scale, part, acc1[r]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) out[r] = acc0[r] + acc1[r];
}

// 2026-09-25: Grid (ceil(N/8), rows * top_k), block 256; blockIdx.y is the union entry. R
// must equal the union's rows: u_slot is read as [u * R + r].
// 2026-10-08: J output columns per warp (METRALE_GLM_MOE_BATCHM_COLS, entries _c<J>): grid x is
// ceil(N / (8 J)) and warp w of block bx computes columns bx * 8J + 8j + w, j = 0..J-1, each with
// the J = 1 arithmetic (same partial chains, shuffle tree and acc_a + acc_b), so every output
// is bit-identical to the J = 1 entry; only the per-block setup is shared by J columns.
template <int R, int J = 1>
__device__ __forceinline__ void w4a16_gemv_sw_moe_batchm_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ u_eid,
    const int* __restrict__ u_slot,
    unsigned int N, unsigned int K, unsigned int num_experts,
    unsigned int a_row_stride,
    unsigned int a_slot_stride,
    unsigned int c_row_stride)
{
    const unsigned int u = blockIdx.y;
    const int eid = u_eid[u];
    if (eid < 0 || (unsigned int)eid >= num_experts) return;
    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[eid];
    if (B_packed == 0) return;
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[eid];
    const float scale2 = scale2_vals[eid];

    int slot[R];
    const __nv_bfloat16* Aptr[R];
    bool any = false;
    #pragma unroll
    for (int r = 0; r < R; r++) {
        slot[r] = u_slot[u * R + r];
        if (slot[r] < 0) { Aptr[r] = nullptr; continue; }
        Aptr[r] = A + (unsigned long long)r * a_row_stride
                    + (unsigned long long)slot[r] * a_slot_stride;
        any = true;
    }
    if (!any) return;

    const unsigned int local_out = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    if (blockIdx.x * (N_PER_BLOCK_SW * J) + local_out >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_warp(s_lut[local_out], lane);
#if METRALE_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT;
#endif

    #pragma unroll 1
    for (int j = 0; j < J; j++) {
        const unsigned int n =
            blockIdx.x * (N_PER_BLOCK_SW * J) + (unsigned int)j * N_PER_BLOCK_SW + local_out;
        if (n >= N) break;
        float acc_a[R], acc_b[R];
        w4a16_gemv_partial_rows<R>(Aptr, B_packed, B_scale, scale2, n, half_K,
                                   num_groups, K16, lane, warp_lut, acc_a);
        w4a16_gemv_partial_rows<R>(Aptr, B_packed, B_scale, scale2, n, half_K,
                                   num_groups, K16, lane + 32u, warp_lut, acc_b);

        #pragma unroll
        for (int r = 0; r < R; r++) {
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                acc_a[r] += __shfl_down_sync(0xFFFFFFFF, acc_a[r], offset);
                acc_b[r] += __shfl_down_sync(0xFFFFFFFF, acc_b[r], offset);
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < R; r++) {
                if (slot[r] < 0) continue;
                float result = acc_a[r] + acc_b[r];
                C[(unsigned long long)r * c_row_stride
                  + (unsigned long long)slot[r] * N + n] = __float2bfloat16(result);
            }
        }
    }
}

#define METRALE_MOE_BATCHM_ENTRY(R)                                                  \
extern "C" __global__ void w4a16_gemv_sw_moe_batchm_m##R(                          \
    const __nv_bfloat16* __restrict__ A,                                           \
    const unsigned long long* __restrict__ packed_ptrs,                            \
    const unsigned long long* __restrict__ scale_ptrs,                             \
    const float* __restrict__ scale2_vals,                                         \
    __nv_bfloat16* __restrict__ C,                                                 \
    const int* __restrict__ u_eid,                                                 \
    const int* __restrict__ u_slot,                                                \
    unsigned int N, unsigned int K, unsigned int num_experts,                      \
    unsigned int a_row_stride, unsigned int a_slot_stride, unsigned int c_row_stride) \
{                                                                                  \
    w4a16_gemv_sw_moe_batchm_body<R>(A, packed_ptrs, scale_ptrs, scale2_vals, C,   \
        u_eid, u_slot, N, K, num_experts, a_row_stride, a_slot_stride, c_row_stride); \
}

METRALE_MOE_BATCHM_ENTRY(2)
METRALE_MOE_BATCHM_ENTRY(3)
METRALE_MOE_BATCHM_ENTRY(4)





METRALE_MOE_BATCHM_ENTRY(5)
METRALE_MOE_BATCHM_ENTRY(6)
METRALE_MOE_BATCHM_ENTRY(7)
METRALE_MOE_BATCHM_ENTRY(8)

// 2026-10-08: METRALE_GLM_MOE_BATCHM_COLS entries: J columns per warp, grid x ceil(N / (8 J)),
// same arguments as w4a16_gemv_sw_moe_batchm_m<R>, bit-identical outputs.
#define METRALE_MOE_BATCHM_COLS_ENTRY(R, J)                                          \
extern "C" __global__ void w4a16_gemv_sw_moe_batchm_m##R##_c##J(                   \
    const __nv_bfloat16* __restrict__ A,                                           \
    const unsigned long long* __restrict__ packed_ptrs,                            \
    const unsigned long long* __restrict__ scale_ptrs,                             \
    const float* __restrict__ scale2_vals,                                         \
    __nv_bfloat16* __restrict__ C,                                                 \
    const int* __restrict__ u_eid,                                                 \
    const int* __restrict__ u_slot,                                                \
    unsigned int N, unsigned int K, unsigned int num_experts,                      \
    unsigned int a_row_stride, unsigned int a_slot_stride, unsigned int c_row_stride) \
{                                                                                  \
    w4a16_gemv_sw_moe_batchm_body<R, J>(A, packed_ptrs, scale_ptrs, scale2_vals, C, \
        u_eid, u_slot, N, K, num_experts, a_row_stride, a_slot_stride, c_row_stride); \
}
#define METRALE_MOE_BATCHM_COLS_ENTRIES(J)                                           \
    METRALE_MOE_BATCHM_COLS_ENTRY(2, J) METRALE_MOE_BATCHM_COLS_ENTRY(3, J)          \
    METRALE_MOE_BATCHM_COLS_ENTRY(4, J) METRALE_MOE_BATCHM_COLS_ENTRY(5, J)          \
    METRALE_MOE_BATCHM_COLS_ENTRY(6, J) METRALE_MOE_BATCHM_COLS_ENTRY(7, J)          \
    METRALE_MOE_BATCHM_COLS_ENTRY(8, J)
METRALE_MOE_BATCHM_COLS_ENTRIES(2)
METRALE_MOE_BATCHM_COLS_ENTRIES(4)
METRALE_MOE_BATCHM_COLS_ENTRIES(8)

// 2026-10-08: METRALE_GLM_MOE_DOWN_FAST. w4a16_gemv_sw_moe_batchm_down_m<R>: the down-projection
// union sweep (K <= 1024, K16 = K / 16 <= 64) with output bytes identical to
// w4a16_gemv_sw_moe_batchm_m<R>_c8 (and so to _m<R>). Same arguments; grid
// (ceil(N / 128), rows * top_k), block 256. Warp w of block bx computes columns
// bx * 128 + 8 j + w, j = 0..15, two at a time.
//
// Bit-equality. In w4a16_gemv_partial_rows<R> with K16 <= 64, orig lane l < 32 (chain a) runs
// exactly chunks kk = 2l (acc0) and 2l + 1 (acc1), each when kk < K16: part = the 16-step fmaf
// chain from 0.0f over elements kk*16 + e, e = 0..15, with weight LUT[nibble e of the 8-byte
// word] (low nibble of byte b is e = 2b, high is 2b + 1), then acc = fmaf(scale, part, 0.0f)
// with scale = fp8 * scale2, out = acc0 + acc1. Orig lanes 32..63 (chain b) run no chunk, so
// every acc_b is +0.0 and its shuffle tree gives +0.0 exactly. This kernel evaluates those same
// fmaf chains with the same operands in the same order (activations are the exact BF16 -> FP32
// values, staged once per block in shared memory; the weight values come from the same LUT,
// dequantized once per column and reused across rows), the same scale mul and fmaf, the same
// acc0 + acc1, and stores bf16(acc_a + 0.0f). The tree for acc_a is evaluated node by node:
// node s_o[i] = s_{2o}[i] + s_{2o}[i + o] for o = 16, 8, 4, 2, 1, i < o, exactly the nodes lane 0
// of the shuffle-down tree depends on. The first log2(V) levels are computed "transposed"
// (lanes L and L ^ o each finish half of the values, swapping the other half with one
// __shfl_xor_sync), the rest with a plain xor butterfly. Each node is the same two FP32
// operands added once (FP32 addition commutes, so which lane holds the left operand does not
// change the bits). With --fmad=false no mul/add pair is contracted, and fmaf is fused in both.
#define MOE_DOWN_FAST_COLS_PER_WARP 16
#define MOE_DOWN_FAST_LANE_STRIDE 36

template <int R>
__device__ __forceinline__ void w4a16_gemv_sw_moe_batchm_down_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ u_eid,
    const int* __restrict__ u_slot,
    unsigned int N, unsigned int K, unsigned int num_experts,
    unsigned int a_row_stride,
    unsigned int a_slot_stride,
    unsigned int c_row_stride)
{
    // 2026-10-08: RP rows padded to a power of two; V = 2 * RP values (two columns) per tree.
    constexpr int RP = (R <= 2) ? 2 : ((R <= 4) ? 4 : 8);
    constexpr int V = 2 * RP;
    constexpr int LOGV = (V == 4) ? 2 : ((V == 8) ? 3 : 4);
    constexpr int LS = MOE_DOWN_FAST_LANE_STRIDE;
    // 2026-10-08: Row r, lane l's 32 activations (chunks 2l, 2l + 1) at s_a[r][l * 36 ..];
    // the 4-float pad keeps a quarter-warp's float4 reads on distinct banks.
    __shared__ __align__(16) float s_a[R * 32 * LS];
    __shared__ float s_lut[16];

    const unsigned int u = blockIdx.y;
    const int eid = u_eid[u];
    if (eid < 0 || (unsigned int)eid >= num_experts) return;
    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[eid];
    if (B_packed == 0) return;
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[eid];
    const float scale2 = scale2_vals[eid];
    const unsigned int K16 = K / 16;
    // 2026-10-08: The host launches this entry only for K <= 1024.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
    if (K16 > 64u) return;
#else
    if (K16 > 64u) __trap();
#endif

    bool live[R];
    bool any = false;
    #pragma unroll
    for (int r = 0; r < R; r++) {
        live[r] = u_slot[u * R + r] >= 0;
        any = any || live[r];
    }
    if (!any) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int w = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int kk0 = 2u * lane, kk1 = 2u * lane + 1u;
    const bool v0 = kk0 < K16, v1 = kk1 < K16;
    const unsigned int nb = blockIdx.x * (N_PER_BLOCK_SW * MOE_DOWN_FAST_COLS_PER_WARP) + w;

    // 2026-10-08: First column pair's weights, issued before the staging barrier.
    unsigned long long q[2][2];
    unsigned char sb[2][2];
    #pragma unroll
    for (int c = 0; c < 2; c++) {
        const unsigned int n = nb + (unsigned int)c * N_PER_BLOCK_SW;
        const bool ok = n < N;
        q[c][0] = (ok && v0) ? *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk0 * 8) : 0ull;
        q[c][1] = (ok && v1) ? *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk1 * 8) : 0ull;
        sb[c][0] = (ok && v0) ? B_scale[(unsigned long long)n * num_groups + kk0] : (unsigned char)0;
        sb[c][1] = (ok && v1) ? B_scale[(unsigned long long)n * num_groups + kk1] : (unsigned char)0;
    }

    // 2026-10-08: Stage the live rows' activations as FP32 (exact) and the E2M1 table.
#if METRALE_WARP_LUT_STAGED
    if (tid < 16u) s_lut[tid] = E2M1_LUT[tid];
    const float* __restrict__ lut = s_lut;
#else
    const float* __restrict__ lut = E2M1_LUT;
#endif
    const unsigned int CH = K16 * 2u;  // 8-element (uint4) chunks per row
    #pragma unroll
    for (int r = 0; r < R; r++) {
        if (!live[r]) continue;
        const int sl = u_slot[u * R + r];
        const uint4* ar = (const uint4*)(A + (unsigned long long)r * a_row_stride
                                           + (unsigned long long)sl * a_slot_stride);
        for (unsigned int ch = tid; ch < CH; ch += blockDim.x) {
            const uint4 x = ar[ch];
            const unsigned int e = ch * 8u;
            float* d = s_a + r * 32 * LS + (e >> 5) * LS + (e & 31u);
            const float2 f0 = __bfloat1622float2(*(const __nv_bfloat162*)&x.x);
            const float2 f1 = __bfloat1622float2(*(const __nv_bfloat162*)&x.y);
            const float2 f2 = __bfloat1622float2(*(const __nv_bfloat162*)&x.z);
            const float2 f3 = __bfloat1622float2(*(const __nv_bfloat162*)&x.w);
            ((float4*)d)[0] = make_float4(f0.x, f0.y, f1.x, f1.y);
            ((float4*)d)[1] = make_float4(f2.x, f2.y, f3.x, f3.y);
        }
    }
    __syncthreads();
    // 2026-10-08: No block barrier below; a warp past N leaves on its own.

    // 2026-10-08: The value this lane stores after the tree: m = lane / (32 / V) when
    // lane % (32 / V) == 0; column c = m / RP of the pair, row r = m % RP.
    const unsigned int m = lane >> (5 - LOGV);
    const bool holder = (lane & ((32u >> LOGV) - 1u)) == 0u;
    const unsigned int my_c = m / RP, my_r = m % RP;
    int my_slot = -1;
    if (holder && my_r < (unsigned int)R) my_slot = u_slot[u * R + my_r];

    #pragma unroll 1
    for (int p = 0; p < MOE_DOWN_FAST_COLS_PER_WARP / 2; p++) {
        const unsigned int n0 = nb + (unsigned int)p * (2 * N_PER_BLOCK_SW);
        if (n0 >= N) break;

        // 2026-10-08: Dequantize this pair (same LUT values and scale product as the car).
        float wv[2][32];
        float sc[2][2];
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                __nv_fp8_e4m3 fp8;
                *(unsigned char*)&fp8 = sb[c][h];
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
                sc[c][h] = scl_fp8(sb[c][h]) * scale2;
#else
                sc[c][h] = (float)fp8 * scale2;
#endif
                #pragma unroll
                for (int e = 0; e < 16; e++)
                    wv[c][h * 16 + e] = lut[(unsigned int)(q[c][h] >> (4 * e)) & 0xFu];
            }
        }

        // 2026-10-08: Prefetch the next pair's weights.
        {
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                const unsigned int n = n0 + 2u * N_PER_BLOCK_SW + (unsigned int)c * N_PER_BLOCK_SW;
                const bool ok = (p + 1 < MOE_DOWN_FAST_COLS_PER_WARP / 2) && n < N;
                q[c][0] = (ok && v0) ? *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk0 * 8) : 0ull;
                q[c][1] = (ok && v1) ? *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk1 * 8) : 0ull;
                sb[c][0] = (ok && v0) ? B_scale[(unsigned long long)n * num_groups + kk0] : (unsigned char)0;
                sb[c][1] = (ok && v1) ? B_scale[(unsigned long long)n * num_groups + kk1] : (unsigned char)0;
            }
        }

        // 2026-10-08: Per row, both columns' chains over one read of the activations.
        float v[V];
        #pragma unroll
        for (int i = 0; i < V; i++) v[i] = 0.0f;
        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (!live[r]) continue;
            const float4* ar = (const float4*)(s_a + r * 32 * LS + lane * LS);
            float pt[2][2];
            #pragma unroll
            for (int c = 0; c < 2; c++) { pt[c][0] = 0.0f; pt[c][1] = 0.0f; }
            #pragma unroll
            for (int f = 0; f < 8; f++) {
                const float4 x = ar[f];
                const int h = f >> 2;
                const int e = (f & 3) * 4;
                #pragma unroll
                for (int c = 0; c < 2; c++) {
                    pt[c][h] = fmaf(x.x, wv[c][h * 16 + e + 0], pt[c][h]);
                    pt[c][h] = fmaf(x.y, wv[c][h * 16 + e + 1], pt[c][h]);
                    pt[c][h] = fmaf(x.z, wv[c][h * 16 + e + 2], pt[c][h]);
                    pt[c][h] = fmaf(x.w, wv[c][h * 16 + e + 3], pt[c][h]);
                }
            }
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                const float acc0 = v0 ? fmaf(sc[c][0], pt[c][0], 0.0f) : 0.0f;
                const float acc1 = v1 ? fmaf(sc[c][1], pt[c][1], 0.0f) : 0.0f;
                v[c * RP + r] = __fadd_rn(acc0, acc1);
            }
        }

        // 2026-10-08: The shuffle-down tree's nodes. Transposed levels: at offset o the lane
        // with bit o clear keeps values [0, h) and the other keeps [h, 2h); each finishes its
        // half with the partner's copy.
        #pragma unroll
        for (int s = 0; s < LOGV; s++) {
            const unsigned int o = 16u >> s;
            const int hh = V >> (s + 1);
            const bool hi = (lane & o) != 0u;
            #pragma unroll
            for (int k = 0; k < hh; k++) {
                const float keep = hi ? v[k + hh] : v[k];
                const float send = hi ? v[k] : v[k + hh];
                const float recv = __shfl_xor_sync(0xFFFFFFFF, send, o);
                v[k] = __fadd_rn(keep, recv);
            }
        }
        float t = v[0];
        #pragma unroll
        for (unsigned int o = 16u >> LOGV; o > 0u; o >>= 1) {
            t = __fadd_rn(t, __shfl_xor_sync(0xFFFFFFFF, t, o));
        }

        // 2026-10-08: result = acc_a + acc_b with acc_b = +0.0 (chain b ran no chunk).
        const unsigned int n = n0 + my_c * N_PER_BLOCK_SW;
        if (my_slot >= 0 && n < N) {
            const float result = __fadd_rn(t, 0.0f);
            C[(unsigned long long)my_r * c_row_stride
              + (unsigned long long)my_slot * N + n] = __float2bfloat16(result);
        }
    }
}

#define METRALE_MOE_BATCHM_DOWN_ENTRY(R)                                             \
extern "C" __global__ __launch_bounds__(256, 2) void w4a16_gemv_sw_moe_batchm_down_m##R( \
    const __nv_bfloat16* __restrict__ A,                                           \
    const unsigned long long* __restrict__ packed_ptrs,                            \
    const unsigned long long* __restrict__ scale_ptrs,                             \
    const float* __restrict__ scale2_vals,                                         \
    __nv_bfloat16* __restrict__ C,                                                 \
    const int* __restrict__ u_eid,                                                 \
    const int* __restrict__ u_slot,                                                \
    unsigned int N, unsigned int K, unsigned int num_experts,                      \
    unsigned int a_row_stride, unsigned int a_slot_stride, unsigned int c_row_stride) \
{                                                                                  \
    w4a16_gemv_sw_moe_batchm_down_body<R>(A, packed_ptrs, scale_ptrs, scale2_vals, C, \
        u_eid, u_slot, N, K, num_experts, a_row_stride, a_slot_stride, c_row_stride); \
}
METRALE_MOE_BATCHM_DOWN_ENTRY(2)
METRALE_MOE_BATCHM_DOWN_ENTRY(3)
METRALE_MOE_BATCHM_DOWN_ENTRY(4)
METRALE_MOE_BATCHM_DOWN_ENTRY(5)
METRALE_MOE_BATCHM_DOWN_ENTRY(6)
METRALE_MOE_BATCHM_DOWN_ENTRY(7)
METRALE_MOE_BATCHM_DOWN_ENTRY(8)

// 2026-10-09: METRALE_GLM_MOE_GATEUP_FAST. w4a16_gemv_sw_moe_batchm_gateup_m<R>: the gate AND up
// union sweeps of one row group in ONE launch, each output byte-identical to
// w4a16_gemv_sw_moe_batchm_m<R> (and so to _m<R>_c<J>) on that matrix. Arguments: A, the gate
// table (packed, scale, scale2), the up table, C_gate, C_up, then u_eid .. c_row_stride as in
// _m<R> (N, K and the strides are shared by both matrices). Grid (ceil(N / 64), rows * top_k),
// block 256; warp w of block bx computes column n = bx * 64 + 8 j + w, j = 0..7, of BOTH
// matrices.
//
// Work sharing. Per column, each (chunk, row) activation load (two uint4, the reference's own
// global loads) feeds both the gate and the up chain, and each chunk's 16 weights per matrix
// are dequantized once (LUT reads) and reused across the R rows. The car issues the row loads
// once per matrix and 16 LUT reads per row per chunk per matrix.
//
// Bit-equality, any K (K16 = K / 16; GLM-5.3 gate/up: K = hidden = 4096, K16 = 256). In
// w4a16_gemv_partial_rows<R>, orig lane o (chain a: o = lane, chain b: o = lane + 32) runs
// chunks kk = 2o + 128 i + h for i = 0, 1, .. while kk < K16 (the loop's `K16 + 1` bound only
// adds an iteration that breaks before any arithmetic), h = 0 into acc0 and h = 1 into acc1,
// each acc = fmaf(scale, part, acc) from 0.0f in increasing i, with part the 16-step fmaf chain
// from 0.0f over elements kk*16 + e (weight LUT[nibble e of the 8-byte word], activation the
// exact BF16 -> FP32 value, element pairs from __bfloat1622float2) and scale = fp8 * scale2;
// out = acc0 + acc1 (a lane with no chunk gives +0.0). acc_a and acc_b each go through the
// 5-level shuffle-down tree and lane 0 stores bf16(acc_a + acc_b). This kernel runs, per lane,
// per chain and per h, exactly those fmaf chains in increasing i with the same operands, then
// acc0 + acc1. Each chain's tree is evaluated node by node like the down entry: V = 2 * RP
// values per lane (index matrix * RP + row, RP = R padded to 2, 4 or 8), log2(V) transposed
// levels (at offset o the lane with bit o clear keeps values [0, h), the other [h, 2h), each
// adds its partner's copy, one __shfl_xor_sync per pair), then a plain xor butterfly over the
// remaining offsets. Each tree node is the same two FP32 operands added once (addition
// commutes). Both chains' trees leave value m on the same lane, which stores
// bf16(t_a + t_b). --fmad=false (common/KERNEL.toml): no mul/add pair is contracted.
#define MOE_GATEUP_FAST_COLS_PER_WARP 8

// 2026-10-09: One chain's shuffle-down tree over V values, transposed then butterfly; returns
// the node value this lane holds (meaningful on holder lanes, see the caller).
template <int V, int LOGV>
__device__ __forceinline__ float moe_gateup_tree(float (&v)[V], unsigned int lane)
{
    #pragma unroll
    for (int s = 0; s < LOGV; s++) {
        const unsigned int o = 16u >> s;
        const int hh = V >> (s + 1);
        const bool hi = (lane & o) != 0u;
        #pragma unroll
        for (int k = 0; k < hh; k++) {
            const float keep = hi ? v[k + hh] : v[k];
            const float send = hi ? v[k] : v[k + hh];
            const float recv = __shfl_xor_sync(0xFFFFFFFF, send, o);
            v[k] = __fadd_rn(keep, recv);
        }
    }
    float t = v[0];
    #pragma unroll
    for (unsigned int o = 16u >> LOGV; o > 0u; o >>= 1) {
        t = __fadd_rn(t, __shfl_xor_sync(0xFFFFFFFF, t, o));
    }
    return t;
}

template <int R>
__device__ __forceinline__ void w4a16_gemv_sw_moe_batchm_gateup_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ g_packed_ptrs,
    const unsigned long long* __restrict__ g_scale_ptrs,
    const float* __restrict__ g_scale2_vals,
    const unsigned long long* __restrict__ u_packed_ptrs,
    const unsigned long long* __restrict__ u_scale_ptrs,
    const float* __restrict__ u_scale2_vals,
    __nv_bfloat16* __restrict__ C_gate,
    __nv_bfloat16* __restrict__ C_up,
    const int* __restrict__ u_eid,
    const int* __restrict__ u_slot,
    unsigned int N, unsigned int K, unsigned int num_experts,
    unsigned int a_row_stride,
    unsigned int a_slot_stride,
    unsigned int c_row_stride)
{
    constexpr int RP = (R <= 2) ? 2 : ((R <= 4) ? 4 : 8);
    constexpr int V = 2 * RP;
    constexpr int LOGV = (V == 4) ? 2 : ((V == 8) ? 3 : 4);
    __shared__ float s_lut[16];

    const unsigned int u = blockIdx.y;
    const int eid = u_eid[u];
    if (eid < 0 || (unsigned int)eid >= num_experts) return;
    const unsigned char* Bp0 = (const unsigned char*)g_packed_ptrs[eid];
    const unsigned char* Bp1 = (const unsigned char*)u_packed_ptrs[eid];
    // 2026-10-09: The car's launch for a matrix returns on a null packed pointer; here that
    // matrix's outputs stay unwritten and the other matrix still runs.
    const bool ok0 = Bp0 != 0, ok1 = Bp1 != 0;
    if (!ok0 && !ok1) return;
    const unsigned char* Bs0 = ok0 ? (const unsigned char*)g_scale_ptrs[eid] : nullptr;
    const unsigned char* Bs1 = ok1 ? (const unsigned char*)u_scale_ptrs[eid] : nullptr;
    const float s20 = ok0 ? g_scale2_vals[eid] : 0.0f;
    const float s21 = ok1 ? u_scale2_vals[eid] : 0.0f;

    // 2026-10-09: Live rows as a bit mask and 32-bit element offsets into A.
    unsigned int aoff[R];
    unsigned int live = 0u;
    #pragma unroll
    for (int r = 0; r < R; r++) {
        const int sl = u_slot[u * R + r];
        aoff[r] = 0u;
        if (sl < 0) continue;
        aoff[r] = (unsigned int)r * a_row_stride + (unsigned int)sl * a_slot_stride;
        live |= 1u << r;
    }
    if (live == 0u) return;

    const unsigned int tid = threadIdx.x;
#if METRALE_WARP_LUT_STAGED
    if (tid < 16u) s_lut[tid] = E2M1_LUT[tid];
    __syncthreads();
    const float* __restrict__ lut = s_lut;
#else
    const float* __restrict__ lut = E2M1_LUT;
#endif
    // 2026-10-09: No block barrier below; a warp past N leaves on its own.

    const unsigned int w = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    // 2026-10-09: The value this lane stores after the trees: m = lane / (32 / V) when
    // lane % (32 / V) == 0; matrix m / RP (0 gate, 1 up), row m % RP.
    const unsigned int m = lane >> (5 - LOGV);
    const bool holder = (lane & ((32u >> LOGV) - 1u)) == 0u;
    const unsigned int my_mat = m / RP, my_r = m % RP;
    int my_slot = -1;
    if (holder && my_r < (unsigned int)R && (my_mat ? ok1 : ok0)) my_slot = u_slot[u * R + my_r];
    __nv_bfloat16* __restrict__ Cm = my_mat ? C_up : C_gate;

    #pragma unroll 1
    for (int j = 0; j < MOE_GATEUP_FAST_COLS_PER_WARP; j++) {
        const unsigned int n =
            blockIdx.x * (N_PER_BLOCK_SW * MOE_GATEUP_FAST_COLS_PER_WARP)
            + (unsigned int)j * N_PER_BLOCK_SW + w;
        if (n >= N) break;
        const unsigned long long wo = (unsigned long long)n * half_K;
        const unsigned long long so = (unsigned long long)n * num_groups;

        float t[2];
        #pragma unroll
        for (int ch = 0; ch < 2; ch++) {
            // 2026-10-09: v[mat * RP + r] = this chain's acc0 + acc1 for (mat, r).
            float v[V];
            #pragma unroll
            for (int i = 0; i < V; i++) v[i] = 0.0f;
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                float acc0[R], acc1[R];
                #pragma unroll
                for (int r = 0; r < R; r++) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
                #pragma unroll 1
                for (unsigned int kk = 2u * (lane + 32u * (unsigned int)ch) + (unsigned int)h;
                     kk < K16; kk += 128u) {
                    const unsigned long long q0 = ok0 ? *(const unsigned long long*)(Bp0 + wo + kk * 8) : 0ull;
                    const unsigned long long q1 = ok1 ? *(const unsigned long long*)(Bp1 + wo + kk * 8) : 0ull;
                    const unsigned char b0 = ok0 ? Bs0[so + kk] : (unsigned char)0;
                    const unsigned char b1 = ok1 ? Bs1[so + kk] : (unsigned char)0;
                    __nv_fp8_e4m3 f0, f1;
                    *(unsigned char*)&f0 = b0;
                    *(unsigned char*)&f1 = b1;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
                    const float sc0 = scl_fp8(b0) * s20;
                    const float sc1 = scl_fp8(b1) * s21;
#else
                    const float sc0 = (float)f0 * s20;
                    const float sc1 = (float)f1 * s21;
#endif
                    float w0[16], w1[16];
                    #pragma unroll
                    for (int e = 0; e < 16; e++) {
                        w0[e] = lut[(unsigned int)(q0 >> (4 * e)) & 0xFu];
                        w1[e] = lut[(unsigned int)(q1 >> (4 * e)) & 0xFu];
                    }
                    #pragma unroll
                    for (int r = 0; r < R; r++) {
                        if (!(live & (1u << r))) continue;
                        const uint4* ar = (const uint4*)(A + aoff[r]);
                        const uint4 a_lo = ar[kk * 2];
                        const uint4 a_hi = ar[kk * 2 + 1];
                        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                                       a_hi.x, a_hi.y, a_hi.z, a_hi.w};
                        float pg = 0.0f, pu = 0.0f;
                        #pragma unroll
                        for (int b = 0; b < 8; b++) {
                            const float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&a_raw[b]);
                            pg = fmaf(af.x, w0[2 * b], pg);
                            pg = fmaf(af.y, w0[2 * b + 1], pg);
                            pu = fmaf(af.x, w1[2 * b], pu);
                            pu = fmaf(af.y, w1[2 * b + 1], pu);
                        }
                        acc0[r] = fmaf(sc0, pg, acc0[r]);
                        acc1[r] = fmaf(sc1, pu, acc1[r]);
                    }
                }
                // 2026-10-09: acc0/acc1 here are the gate/up accumulators of chunk parity h.
                // h = 0 parks them in v; h = 1 adds: out = acc(h=0) + acc(h=1), as the car.
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    if (h == 0) { v[r] = acc0[r]; v[RP + r] = acc1[r]; }
                    else {
                        v[r] = __fadd_rn(v[r], acc0[r]);
                        v[RP + r] = __fadd_rn(v[RP + r], acc1[r]);
                    }
                }
            }
            t[ch] = moe_gateup_tree<V, LOGV>(v, lane);
        }

        // 2026-10-09: result = acc_a + acc_b, both trees' lane-0 nodes for (mat, row).
        if (my_slot >= 0) {
            const float result = __fadd_rn(t[0], t[1]);
            Cm[(unsigned long long)my_r * c_row_stride
               + (unsigned long long)my_slot * N + n] = __float2bfloat16(result);
        }
    }
}

#define METRALE_MOE_BATCHM_GATEUP_ENTRY(R)                                           \
extern "C" __global__ __launch_bounds__(256, 2) void w4a16_gemv_sw_moe_batchm_gateup_m##R( \
    const __nv_bfloat16* __restrict__ A,                                           \
    const unsigned long long* __restrict__ g_packed_ptrs,                          \
    const unsigned long long* __restrict__ g_scale_ptrs,                           \
    const float* __restrict__ g_scale2_vals,                                       \
    const unsigned long long* __restrict__ u_packed_ptrs,                          \
    const unsigned long long* __restrict__ u_scale_ptrs,                           \
    const float* __restrict__ u_scale2_vals,                                       \
    __nv_bfloat16* __restrict__ C_gate,                                            \
    __nv_bfloat16* __restrict__ C_up,                                              \
    const int* __restrict__ u_eid,                                                 \
    const int* __restrict__ u_slot,                                                \
    unsigned int N, unsigned int K, unsigned int num_experts,                      \
    unsigned int a_row_stride, unsigned int a_slot_stride, unsigned int c_row_stride) \
{                                                                                  \
    w4a16_gemv_sw_moe_batchm_gateup_body<R>(A, g_packed_ptrs, g_scale_ptrs,        \
        g_scale2_vals, u_packed_ptrs, u_scale_ptrs, u_scale2_vals, C_gate, C_up,   \
        u_eid, u_slot, N, K, num_experts, a_row_stride, a_slot_stride, c_row_stride); \
}
METRALE_MOE_BATCHM_GATEUP_ENTRY(2)
METRALE_MOE_BATCHM_GATEUP_ENTRY(3)
METRALE_MOE_BATCHM_GATEUP_ENTRY(4)
METRALE_MOE_BATCHM_GATEUP_ENTRY(5)
METRALE_MOE_BATCHM_GATEUP_ENTRY(6)
METRALE_MOE_BATCHM_GATEUP_ENTRY(7)
METRALE_MOE_BATCHM_GATEUP_ENTRY(8)
