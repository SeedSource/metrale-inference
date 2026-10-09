// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-08: Row-scaled FP8 GEMMs for the DFlash drafter on GLM-5.3 (module `dflash_fp8_gemm`,
// SeedHQ/seed-skills#71 S1b):
//   C[M, N] (bf16) = (e4m3(A[M, K]) . B_fp8[N, :]) x row_scale[N]
// Ported from Metrale's own kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu (fp8_gemm_t_row_scaled
// and fp8_gemm_t_row_scaled_m16, with the helpers they call), unchanged except that the
// __SCALE__ / HIP branches are dropped: this file is compiled by nvcc for gb10 only.
// Not copied from any third-party source.
//
// Owner: gb10 kernels (glm-5.3-flash).
// Invariants:
// - B_fp8 is [N, K] E4M3 with one f32 scale per row (quantize_bf16_to_fp8's layout); A and C are
//   row-major BF16. A is converted to E4M3 (cvt.rn.satfinite, unscaled) inside the MMA tile, so
//   this is W8A8: |A| above 448 saturates and small |A| lose precision. Drafter only.
// - fp8_gemm_t_row_scaled: grid (ceil(N / 128), ceil(M / 64)), block 128.
//   fp8_gemm_t_row_scaled_m16: grid (ceil(N / 128)), block 32, M <= 16 (rows past 16 are not
//   computed; ops::fp8_gemm_n128_row_scaled_m16's callers keep M <= 16).
// - K % 32 == 0 (a partial last K step is zero-filled by the predicated loads, so a tail is
//   dropped, not read out of bounds). Every drafter projection has K a multiple of 1024.
// - The drafter takes these only above 16 rows, or with METRALE_NO_DFLASH_FP8_RT=1; at <= 16
//   rows its FP8 projections run fp8_gemv_rowscale_batch{8,16}_rt2 (common/fp8_gemv_rt.cu).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define DFP8_M_TILE 64
#define DFP8_N_TILE 128
#define DFP8_K_STEP 32
#define DFP8_PAD 8   // 2026-10-08: (32 + 8) BF16 = 80-byte rows keep cp.async destinations 16-byte aligned.

// 2026-10-08: Two floats to packed E4M3 (cvt.rn.satfinite.e4m3x2.f32): the high byte is e4m3(a_hi),
// the low byte e4m3(b_lo).
__device__ __forceinline__ unsigned short dfp8_cvt_e4m3x2_f32(float a_hi, float b_lo) {
    unsigned short d;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(a_hi), "f"(b_lo));
    return d;
}

// 2026-10-08: One m16n8k32 E4M3 MMA with FP32 accumulation.
__device__ __forceinline__ void dfp8_mma_e4m3(float* acc,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
}

__device__ __forceinline__ void dfp8_cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void dfp8_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void dfp8_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

// 2026-10-08: Four BF16 values in shared memory to four E4M3 bytes, the first in the low byte.
__device__ __forceinline__ unsigned int dfp8_bf16x4_to_e4m3x4(const unsigned short* src) {
    unsigned int p0 = *(const unsigned int*)src;
    unsigned int p1 = *(const unsigned int*)(src + 2);
    unsigned short bf0 = (unsigned short)(p0 & 0xFFFFu);
    unsigned short bf1 = (unsigned short)(p0 >> 16);
    unsigned short bf2 = (unsigned short)(p1 & 0xFFFFu);
    unsigned short bf3 = (unsigned short)(p1 >> 16);
    float f0, f1, f2, f3;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f0) : "h"(bf0));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f1) : "h"(bf1));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f2) : "h"(bf2));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f3) : "h"(bf3));
    unsigned short h0, h1;
    h0 = dfp8_cvt_e4m3x2_f32(f1, f0);
    h1 = dfp8_cvt_e4m3x2_f32(f3, f2);
    return ((unsigned int)h1 << 16) | (unsigned int)h0;
}

// 2026-10-08: 64 x 128 tile, 4 warps of 16 rows, K step 32, A and B double-buffered with cp.async.
extern "C" __global__ void fp8_gemm_t_row_scaled(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * DFP8_N_TILE;
    const unsigned int cta_m = blockIdx.y * DFP8_M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][DFP8_M_TILE][DFP8_K_STEP + DFP8_PAD];
    __shared__ unsigned char smem_B[2][DFP8_N_TILE][DFP8_K_STEP];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = DFP8_K_STEP + DFP8_PAD;

    #define DFP8_LOADS_RS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                dfp8_cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            dfp8_cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            dfp8_cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    #define DFP8_COMPUTE_RS(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = dfp8_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = dfp8_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = dfp8_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = dfp8_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            dfp8_mma_e4m3(acc[nt], a0, a1, a2, a3, b0, b1); \
        } \
    } while(0)

    DFP8_LOADS_RS(0, 0);
    dfp8_cp_async_commit();
    dfp8_cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = DFP8_K_STEP; k_base < K; k_base += DFP8_K_STEP) {
        int nxt = 1 - cur;
        DFP8_LOADS_RS(nxt, k_base);
        dfp8_cp_async_commit();
        DFP8_COMPUTE_RS(cur, cur);
        dfp8_cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    DFP8_COMPUTE_RS(cur, cur);

    #undef DFP8_LOADS_RS
    #undef DFP8_COMPUTE_RS

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        float sc0 = (c0 < N) ? row_scale[c0] : 0.0f;
        float sc1 = (c1 < N) ? row_scale[c1] : 0.0f;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0] * sc0);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1] * sc1);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2] * sc0);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3] * sc1);
    }
}

// 2026-10-08: The single-warp 16-row twin: one warp computes rows 0..15 of a 128-column tile. Two
// rounds of 16-byte copies fill the 16 x 32 BF16 A tile (thread t loads row t / 2, columns
// ((t & 1) << 4) + 8r in round r); four rounds fill the 128 x 32 B tile.
extern "C" __global__ void fp8_gemm_t_row_scaled_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * DFP8_N_TILE;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][16][DFP8_K_STEP + DFP8_PAD];
    __shared__ unsigned char smem_B[2][DFP8_N_TILE][DFP8_K_STEP];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = DFP8_K_STEP + DFP8_PAD;

    #define DFP8_LOADS_M16(buf, kb) do { \
        _Pragma("unroll") \
        for (int ar = 0; ar < 2; ar++) { \
            unsigned int row = threadIdx.x >> 1; \
            unsigned int a_col = ((threadIdx.x & 1) << 4) + ar * 8; \
            unsigned int gc = (kb) + a_col; \
            unsigned int gr = row; \
            dfp8_cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                &A[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 7 < K)); \
        } \
        { \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int my_n = rnd * 32 + threadIdx.x; \
                unsigned int gn = cta_n + my_n; \
                bool valid = (gn < N) && ((kb) + 31 < K); \
                dfp8_cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                    &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
                dfp8_cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                    &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
            } \
        } \
    } while(0)

    #define DFP8_COMPUTE_M16(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = group_id, fr1 = fr0 + 8; \
        unsigned int a0 = dfp8_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = dfp8_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = dfp8_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = dfp8_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            dfp8_mma_e4m3(acc[nt], a0, a1, a2, a3, b0, b1); \
        } \
    } while(0)

    DFP8_LOADS_M16(0, 0);
    dfp8_cp_async_commit();
    dfp8_cp_async_wait_all();
    __syncwarp();

    int cur = 0;
    for (unsigned int k_base = DFP8_K_STEP; k_base < K; k_base += DFP8_K_STEP) {
        int nxt = 1 - cur;
        DFP8_LOADS_M16(nxt, k_base);
        dfp8_cp_async_commit();
        DFP8_COMPUTE_M16(cur, cur);
        dfp8_cp_async_wait_all();
        __syncwarp();
        cur = nxt;
    }
    DFP8_COMPUTE_M16(cur, cur);

    #undef DFP8_LOADS_M16
    #undef DFP8_COMPUTE_M16

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = group_id;
        unsigned int r1 = r0 + 8;
        float sc0 = (c0 < N) ? row_scale[c0] : 0.0f;
        float sc1 = (c1 < N) ? row_scale[c1] : 0.0f;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0] * sc0);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1] * sc1);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2] * sc0);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3] * sc1);
    }
}
