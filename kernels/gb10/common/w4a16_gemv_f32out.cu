// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-12: NVFP4-weight GEMVs with BF16 activations and FP32 OUTPUT, for the GLM-5.3 DSA
// indexer's `wq_b` (METRALE_GLM_IDX_WQB_NV4). Two families, both copies of production kernels with
// only the store changed (no BF16 rounding of the result):
// - w4a16_gemv_f32out_b{1,2,3,4,5,6,7,8,16}: w4a16_gemv_batchm_impl<R> of w4a16_gemv.cu (CUDA
//   cores). Every row is bit-identical to w4a16_gemv on it, and BF16(result) is bit-identical to
//   the BF16 output of w4a16_gemv / w4a16_gemv_batch* (same FMA and reduction order; the store is
//   the only difference). All take M (1 <= M <= R) after scale2/C, then N, K; grid ceil(N/4),
//   block 256, C [M, N] FP32.
// - w4a16_gemv_tc{8,16}_f32out: w4a16_gemv_tc_tile of w4a16_gemv_tc.cu (mma.sync, flat in M);
//   BF16(result) is bit-identical to w4a16_gemv_tc8 / tc16. Needs K % 128 == 0; grid
//   ceil(N / (8 * NT)) (tc8 NT 1, tc16 NT 2), block 256; C must be 8-byte aligned.
// Weight layout and scales as in w4a16_gemv.cu (B_packed [N, K/2], B_scale [N, K/16] E4M3).
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16
#define TC_WARPS 8

__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

template <int MAX_M>
__device__ __forceinline__ void w4a16_gemv_f32out_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    float* __restrict__ C,
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
            float scale = (float)fp8 * scale2;
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
            C[(unsigned long long)t * N + n] = r;
        }
    }
}

__device__ __forceinline__ void w4tc_mma(float (&d)[4], uint32_t a0, uint32_t a1,
                                         uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 2026-09-25: Nibbles at bits 12..15 and 28..31 of q -> BF16x2 {lo, hi} = e2m1 * 2^-126, by
// bit manipulation alone, so the hopper and b200 targets, which inherit gb10/common, compile it.
// Its name avoids the tokens crates/kernels/tests/blockscale_mma_guard.rs looks for.
__device__ __forceinline__ uint32_t w4tc_fp4pair(uint32_t q) {
    return (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
}
__device__ __forceinline__ uint32_t w4tc_bmul2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
// 2026-09-25: E4M3 scale byte -> BF16 (scale * 2^100) in both halves. Exact.
__device__ __forceinline__ uint32_t w4tc_scale_x2(uint32_t sb) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = (unsigned char)sb;
    const float v = (float)f * 0x1p100f;
    const __nv_bfloat16 h = __float2bfloat16_rn(v);
    const uint32_t u = (uint32_t)(*(const unsigned short*)&h);
    return u | (u << 16);
}
__device__ __forceinline__ void w4tc_store2(float* p, float a, float b, bool paired,
                                            bool b_live, bool a_live) {
    if (paired) {
        *(float2*)p = make_float2(a, b);
    } else {
        if (a_live) p[0] = a;
        if (b_live) p[1] = b;
    }
}
// 2026-10-09: One 8 * NT column tile starting at column n0 of C. Nothing in it reads blockIdx or
// gridDim, so the per-CTA entries (n0 = blockIdx.x * 8 * NT) and the grouped persistent entries
// below run the same instructions on the same operands for a given tile: same bits.
template <int MT, int NT, int KU>
__device__ __forceinline__ void w4a16_gemv_tc_tile(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    float* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, const unsigned int n0)
{
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int half_K = K >> 1;
    const unsigned int num_groups = K >> 4;
    const unsigned int num_kb = K >> 7;
    const bool lo_live = g < M;
    const bool hi_live = (MT > 8) && (g + 8u < M);
    const uint4* a_lo_row = (const uint4*)(A + (unsigned long long)g * K);
    const uint4* a_hi_row = (const uint4*)(A + (unsigned long long)(g + 8u) * K);
    // 2026-09-25: A weight row n >= N, in a last tile that N does not fill, loads zeros and is
    // never stored.
    bool tile_live[NT];
    const unsigned char* wrow[NT];
    const unsigned char* srow[NT];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        const unsigned int n = n0 + (unsigned int)i * 8u + g;
        tile_live[i] = n < N;
        wrow[i] = B_packed + (unsigned long long)n * half_K + t * 16u;
        srow[i] = B_scale + (unsigned long long)n * num_groups + t * 2u;
    }
    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) acc[i][c] = 0.0f;
    }
    // 2026-09-25: Each trip covers KU k-blocks and issues all of their weight and scale loads,
    // KU * NT 16-byte weight loads per thread, before the first dequant.
    for (unsigned int kb0 = warp; kb0 < num_kb; kb0 += TC_WARPS * KU) {
        uint4 w[KU][NT];
        uint32_t sc[KU][NT];
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * TC_WARPS;
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                if (tile_live[i] && kb < num_kb) {
                    w[u][i] = *(const uint4*)(wrow[i] + kb * 64u);
                    sc[u][i] = *(const unsigned short*)(srow[i] + kb * 8u);
                } else {
                    w[u][i] = make_uint4(0u, 0u, 0u, 0u);
                    sc[u][i] = 0u;
                }
            }
        }
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * TC_WARPS;
            if (kb >= num_kb) break;
            // 2026-09-25: This thread's 32 activations of row g (and g + 8), k = kb * 128 + t * 32 on; abase counts uint4s of 8 BF16.
            uint4 al[4], ah[4];
            const unsigned int abase = kb * 16u + t * 4u;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                al[j] = lo_live ? a_lo_row[abase + j] : make_uint4(0u, 0u, 0u, 0u);
                if (MT > 8) ah[j] = hi_live ? a_hi_row[abase + j] : make_uint4(0u, 0u, 0u, 0u);
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                // 2026-09-25: Activation pairs for P0..P3 of weight word j.
                const uint32_t l37 = __byte_perm(al[j].y, al[j].w, 0x7632);
                const uint32_t l26 = __byte_perm(al[j].y, al[j].w, 0x5410);
                const uint32_t l15 = __byte_perm(al[j].x, al[j].z, 0x7632);
                const uint32_t l04 = __byte_perm(al[j].x, al[j].z, 0x5410);
                uint32_t h37 = 0u, h26 = 0u, h15 = 0u, h04 = 0u;
                if (MT > 8) {
                    h37 = __byte_perm(ah[j].y, ah[j].w, 0x7632);
                    h26 = __byte_perm(ah[j].y, ah[j].w, 0x5410);
                    h15 = __byte_perm(ah[j].x, ah[j].z, 0x7632);
                    h04 = __byte_perm(ah[j].x, ah[j].z, 0x5410);
                }
                #pragma unroll
                for (int i = 0; i < NT; i++) {
                    const uint32_t q = (j == 0) ? w[u][i].x : (j == 1) ? w[u][i].y
                                     : (j == 2) ? w[u][i].z : w[u][i].w;
                    const uint32_t s = w4tc_scale_x2((sc[u][i] >> ((j >> 1) * 8)) & 0xFFu);
                    const uint32_t p0 = w4tc_bmul2(w4tc_fp4pair(q), s);
                    const uint32_t p1 = w4tc_bmul2(w4tc_fp4pair(q << 4), s);
                    const uint32_t p2 = w4tc_bmul2(w4tc_fp4pair(q << 8), s);
                    const uint32_t p3 = w4tc_bmul2(w4tc_fp4pair(q << 12), s);
                    w4tc_mma(acc[i], l37, h37, l26, h26, p0, p1);
                    w4tc_mma(acc[i], l15, h15, l04, h04, p2, p3);
                }
            }
        }
    }
    // 2026-09-25: Split-K reduction across the CTA's warps, in warp order.
    __shared__ float red[TC_WARPS][NT][4][32];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) red[warp][i][c][lane] = acc[i][c];
    }
    __syncthreads();
    const float sfin = scale2 * 0x1p26f;
    for (unsigned int i = warp; i < (unsigned int)NT; i += TC_WARPS) {
        if (n0 + i * 8u >= N) continue;
        float r[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float v = red[0][i][c][lane];
            #pragma unroll
            for (int ww = 1; ww < TC_WARPS; ww++) v += red[ww][i][c][lane];
            r[c] = v * sfin;
        }
        const unsigned int col = n0 + i * 8u + t * 2u;
        // 2026-09-25: Paired 4-byte store only when it is aligned (N even) and in range.
        const bool paired = ((N & 1u) == 0u) && (col + 1u < N);
        if (g < M) w4tc_store2(C + (unsigned long long)g * N + col, r[0], r[1], paired, col + 1u < N, col < N);
        if (MT > 8 && g + 8u < M)
            w4tc_store2(C + (unsigned long long)(g + 8u) * N + col, r[2], r[3], paired, col + 1u < N, col < N);
    }
}
template <int MT, int NT, int KU>
__device__ __forceinline__ void w4a16_gemv_tc_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    float* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K)
{
    w4a16_gemv_tc_tile<MT, NT, KU>(A, B_packed, B_scale, scale2, C, M, N, K,
                                   blockIdx.x * (8u * NT));
}

#define F32OUT_ENTRY(NAME, R)                                                          \
    extern "C" __global__ void NAME(                                                    \
        const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed, \
        const unsigned char* __restrict__ B_scale, const float scale2,                  \
        float* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {        \
        w4a16_gemv_f32out_impl<R>(A, B_packed, B_scale, scale2, C, M, N, K);            \
    }

F32OUT_ENTRY(w4a16_gemv_f32out_b1, 1)
F32OUT_ENTRY(w4a16_gemv_f32out_b2, 2)
F32OUT_ENTRY(w4a16_gemv_f32out_b3, 3)
F32OUT_ENTRY(w4a16_gemv_f32out_b4, 4)
F32OUT_ENTRY(w4a16_gemv_f32out_b5, 5)
F32OUT_ENTRY(w4a16_gemv_f32out_b6, 6)
F32OUT_ENTRY(w4a16_gemv_f32out_b7, 7)
F32OUT_ENTRY(w4a16_gemv_f32out_b8, 8)
F32OUT_ENTRY(w4a16_gemv_f32out_b16, 16)

#define TC_F32OUT_ENTRY(NAME, MT, NT, KU)                                               \
    extern "C" __global__ __launch_bounds__(TC_WARPS * 32) void NAME(                    \
        const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,  \
        const unsigned char* __restrict__ B_scale, const float scale2,                    \
        float* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {          \
        w4a16_gemv_tc_impl<MT, NT, KU>(A, B_packed, B_scale, scale2, C, M, N, K);        \
    }

TC_F32OUT_ENTRY(w4a16_gemv_tc8_f32out, 8, 1, 8)
TC_F32OUT_ENTRY(w4a16_gemv_tc16_f32out, 16, 2, 2)
