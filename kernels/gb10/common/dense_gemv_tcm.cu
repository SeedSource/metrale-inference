// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-04: Small-M dense GEMV on tensor cores, for BF16 weights and for FP8 E4M3
// weight-only (one FP32 scale per output row), BF16 activations and output:
//   C[t, n] = sum_k A[t, k] * W[n, k]                    (dense_gemv_tcm_bf16*)
//   C[t, n] = row_scale[n] * sum_k A[t, k] * fp8(W[n, k]) (dense_gemv_tcm_fp8w*)
// for t in [0, M), M <= 16 (TCM_MAX_M).
//
// Owner: gb10 kernels.
// Invariants:
// - A is [M, K] BF16 contiguous, W is [N, K] row-major (BF16, or E4M3 bytes made by
//   quantize_bf16_to_fp8), row_scale is [N] FP32 (ignored by the BF16 entries, may be null),
//   row t of C starts at C + t * out_stride BF16 elements.
// - K % 64 == 0 for the BF16 entries, K % 128 == 0 for the FP8 entries (one k-block is 128
//   weight bytes per row in both); the host wrapper refuses any other K. Any N: a weight row
//   past N loads zeros and is never stored. Rows of A past M are never read or stored.
// - Launch: grid (ceil(N / (16 * NT)), 1, 1), block (256, 1, 1), no dynamic shared memory.
// - ROW-INVARIANT: row t's bits depend only on A[t], W, row_scale, N and K - never on M or
//   on any other row's values. Tokens sit on the 8-wide N side of mma.m16n8k16 (two token
//   tiles, 0..7 and 8..15); each D column is its own dot product, a token tile past M is
//   skipped warp-uniformly, the k-block order of every warp and the fixed warp order of the
//   reduction do not depend on M, and an absent token row feeds zeros only into its own
//   column. So one M = 1 decode and a 16-row verify give each row the same bits.
// - Not bit-identical to the CUDA-core GEMVs (dense_gemv_bf16*, dense_gemv_fp8w*):
//   accumulation happens inside the MMA, in another order.
// - Deterministic: no atomics; the 8 warps split K (interleaved k-blocks) and reduce
//   through shared memory in warp order 0..7.
//
// Operand placement (as dense_gemv_bf16_tc.cu): weights on the 16-row A side, tokens on the
// B side, so one MMA covers 16 weight rows x 16 k for 8 tokens.
//
// K permutation (no repack). Thread (g = lane / 4, t = lane % 4) loads, for weight rows g
// and g + 8 of its tile, two 16-byte pieces of each 128-byte k-block: bytes 16t (piece 0)
// and 64 + 16t (piece 1), so each load instruction of a warp reads 64 contiguous bytes per
// row. Any bijection between a k16 MMA's 16 logical k slots and 16 physical k is exact as
// long as the A and B fragments use the same one; here the slot -> physical map depends
// only on t, the piece and the MMA index:
// - BF16 (64 k per block, 4 MMAs): piece p holds k = 32p + 8t + (0..7). MMA j takes words
//   (2j)&3 and (2j+1)&3 of piece j>>1 in k slots {2t, 2t+1} and {2t+8, 2t+9}. The B
//   fragment of token g is built from the same 16 k of activation row g, loaded at k
//   offsets 8t and 32 + 8t.
// - FP8 (128 k per block, 8 MMAs): piece p holds k = 64p + 16t + (0..15). MMA (p, j) takes
//   bytes 4j..4j+3 of piece p, physical k = 64p + 16t + 4j + (0..3): bytes 4j, 4j+1 in slots
//   {2t, 2t+1}, bytes 4j+2, 4j+3 in {2t+8, 2t+9}, each pair converted E4M3 -> BF16 (exact:
//   every E4M3 value has <= 4 significant bits and an exponent inside BF16's range). The B
//   fragment of token g is words 2j and 2j+1 of activation row g's k = 64p + 16t + (0..15),
//   loaded as two 16-byte vectors.
// The FP8 row scale multiplies the FP32 dot product once, after the warp reduction, then
// the result is rounded to BF16 (RNE).
//
// Weights are read once per launch with ld.global.nc.L1::no_allocate, leaving L1 to the
// activation rows every CTA re-reads. KU k-blocks per warp per trip are all issued before
// any MMA (KU * NT * 2 rows * 32 B of weights in flight per thread).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define TCM_WARPS 8
#define TCM_THREADS (TCM_WARPS * 32)
// 2026-10-04: Token tiles of 8: rows 0..7 and 8..15. Mirrored by TCM_MAX_M in
// crates/model-layers/src/layers/ops/dense_gemv_tcm.rs.
#define TCM_NB 2

__device__ __forceinline__ void tcm_mma(float (&d)[4], uint32_t a0, uint32_t a1,
                                        uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 tcm_ld_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t tcm_word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// 2026-10-04: Software E4M3 decode, as fp8bm_dec in dense_gemv_fp8w_batchm.cu.
__device__ __forceinline__ float tcm_e4m3_f32(uint32_t b) {
    uint32_t s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u;
    float v;
    if (e == 0u)                  v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                          v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
__device__ __forceinline__ uint32_t tcm_e4m3x2_bf16x2(uint32_t pair) {
    __nv_bfloat162 r = __floats2bfloat162_rn(tcm_e4m3_f32(pair & 0xFFu),
                                             tcm_e4m3_f32((pair >> 8) & 0xFFu));
    return *reinterpret_cast<uint32_t*>(&r);
}
#else
// 2026-10-04: Two E4M3 bytes (low byte = lower k) -> BF16x2 (low half = lower k).
// cvt.rn.f16x2.e4m3x2 is exact (every E4M3 value is an FP16 normal or zero), FP16 -> FP32
// is exact, and FP32 -> BF16 RN is exact for values with <= 4 significant bits.
__device__ __forceinline__ uint32_t tcm_e4m3x2_bf16x2(uint32_t pair) {
    const __half2_raw hr =
        __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(pair & 0xFFFFu), __NV_E4M3);
    const float2 f = __half22float2(__half2(hr));
    __nv_bfloat162 r = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<uint32_t*>(&r);
}
#endif

template <bool FP8, int NT, int KU>
__device__ __forceinline__ void tcm_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride)
{
    // 2026-10-04: Elements per k-block: 128 weight bytes per row in both kinds.
    constexpr unsigned int KBE = FP8 ? 128u : 64u;
    constexpr unsigned int WBYTES = FP8 ? 1u : 2u;

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int n0 = blockIdx.x * (16u * NT);
    const unsigned int num_kb = K / KBE;
    const unsigned long long row_bytes = (unsigned long long)K * WBYTES;

    // Weight rows g and g + 8 of each tile; a row past N loads zeros.
    bool live[NT][2];
    const unsigned char* wrow[NT][2];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned int n = n0 + (unsigned int)i * 16u + (unsigned int)h * 8u + g;
            live[i][h] = n < N;
            wrow[i][h] = W + (unsigned long long)(live[i][h] ? n : 0u) * row_bytes + t * 16u;
        }
    }
    // Token tile b is on iff its first row is < M (warp-uniform); inside an on tile a row
    // past M feeds zeros into its own column only.
    bool tile_on[TCM_NB];
    bool tok_live[TCM_NB];
    const __nv_bfloat16* arow[TCM_NB];
    #pragma unroll
    for (int b = 0; b < TCM_NB; b++) {
        const unsigned int tok = (unsigned int)b * 8u + g;
        tile_on[b] = (unsigned int)b * 8u < M;
        tok_live[b] = tok < M;
        arow[b] = A + (unsigned long long)(tok_live[b] ? tok : 0u) * K + (FP8 ? t * 16u : t * 8u);
    }

    float acc[NT][TCM_NB][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int b = 0; b < TCM_NB; b++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[i][b][c] = 0.0f;
        }
    }

    for (unsigned int kb0 = warp; kb0 < num_kb; kb0 += TCM_WARPS * KU) {
        uint4 w[KU][NT][2][2];
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * TCM_WARPS;
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    if (live[i][h] && kb < num_kb) {
                        const unsigned char* p = wrow[i][h] + (unsigned long long)kb * 128u;
                        w[u][i][h][0] = tcm_ld_stream(p);
                        w[u][i][h][1] = tcm_ld_stream(p + 64);
                    } else {
                        w[u][i][h][0] = make_uint4(0u, 0u, 0u, 0u);
                        w[u][i][h][1] = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            }
        }
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * TCM_WARPS;
            if (kb >= num_kb) break;
            #pragma unroll
            for (int p = 0; p < (FP8 ? 2 : 1); p++) {
                // Activation pieces of each on tile for this block (FP8: piece p only).
                uint4 av[TCM_NB][2];
                #pragma unroll
                for (int b = 0; b < TCM_NB; b++) {
                    if (tile_on[b] && tok_live[b]) {
                        const uint4* ap = (const uint4*)(arow[b] + kb * KBE + (FP8 ? p * 64u : 0u));
                        av[b][0] = ap[0];
                        av[b][1] = FP8 ? ap[1] : ap[4];   // FP8: +8 k; BF16: +32 k
                    } else {
                        av[b][0] = make_uint4(0u, 0u, 0u, 0u);
                        av[b][1] = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint32_t a0[NT], a1[NT], a2[NT], a3[NT];
                    #pragma unroll
                    for (int i = 0; i < NT; i++) {
                        if (FP8) {
                            const uint32_t q0 = tcm_word(w[u][i][0][p], j);
                            const uint32_t q1 = tcm_word(w[u][i][1][p], j);
                            a0[i] = tcm_e4m3x2_bf16x2(q0);
                            a2[i] = tcm_e4m3x2_bf16x2(q0 >> 16);
                            a1[i] = tcm_e4m3x2_bf16x2(q1);
                            a3[i] = tcm_e4m3x2_bf16x2(q1 >> 16);
                        } else {
                            a0[i] = tcm_word(w[u][i][0][j >> 1], (2 * j) & 3);
                            a2[i] = tcm_word(w[u][i][0][j >> 1], (2 * j + 1) & 3);
                            a1[i] = tcm_word(w[u][i][1][j >> 1], (2 * j) & 3);
                            a3[i] = tcm_word(w[u][i][1][j >> 1], (2 * j + 1) & 3);
                        }
                    }
                    #pragma unroll
                    for (int b = 0; b < TCM_NB; b++) {
                        if (!tile_on[b]) continue;
                        const uint32_t b0 = tcm_word(av[b][j >> 1], (2 * j) & 3);
                        const uint32_t b1 = tcm_word(av[b][j >> 1], (2 * j + 1) & 3);
                        #pragma unroll
                        for (int i = 0; i < NT; i++) {
                            tcm_mma(acc[i][b], a0[i], a1[i], a2[i], a3[i], b0, b1);
                        }
                    }
                }
            }
        }
    }

    // Fixed-order split-K reduction across the CTA's warps.
    __shared__ float red[TCM_WARPS][NT * TCM_NB][4][32];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int b = 0; b < TCM_NB; b++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) red[warp][i * TCM_NB + b][c][lane] = acc[i][b][c];
        }
    }
    __syncthreads();

    // D fragment: d0, d1 = (weight row g, tokens 2t, 2t+1); d2, d3 = row g + 8.
    for (unsigned int f = warp; f < (unsigned int)(NT * TCM_NB); f += TCM_WARPS) {
        const unsigned int i = f / TCM_NB;
        const unsigned int b = f % TCM_NB;
        if (b * 8u >= M) continue;
        const unsigned int tok0 = b * 8u + t * 2u;
        const unsigned int n_lo = n0 + i * 16u + g;
        const unsigned int n_hi = n_lo + 8u;
        float s_lo = 1.0f, s_hi = 1.0f;
        if (FP8) {
            s_lo = n_lo < N ? row_scale[n_lo] : 0.0f;
            s_hi = n_hi < N ? row_scale[n_hi] : 0.0f;
        }
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float v = red[0][f][c][lane];
            #pragma unroll
            for (int ww = 1; ww < TCM_WARPS; ww++) v += red[ww][f][c][lane];
            const unsigned int tok = tok0 + (unsigned int)(c & 1);
            const unsigned int n = (c < 2) ? n_lo : n_hi;
            if (FP8) v *= (c < 2) ? s_lo : s_hi;
            if (tok < M && n < N)
                C[(unsigned long long)tok * out_stride + n] = __float2bfloat16_rn(v);
        }
    }
}

#define TCM_ENTRY(NAME, FP8, NT, KU, MINB)                                               \
    extern "C" __global__ __launch_bounds__(TCM_THREADS, MINB) void NAME(                  \
        const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,          \
        const float* __restrict__ row_scale, __nv_bfloat16* __restrict__ C,                \
        unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {         \
        tcm_impl<FP8, NT, KU>(A, W, row_scale, C, M, N, K, out_stride);                    \
    }

// 2026-10-04: Geometry variants measured by crates/model-arch/examples/
// glm5next_gemv_tc_microtest.rs; ops::dense_gemv_tcm picks one per weight kind from N and K
// only, never from M. Variants of one kind are bit-identical to each other: k-block kb always
// goes to warp kb % TCM_WARPS, each warp adds its blocks in increasing kb order whatever KU
// is, and NT only changes which CTA owns a weight row, not that row's arithmetic.
// minBlocks 1: up to 255 registers, one CTA per SM; minBlocks 2: at most 128, two per SM.
TCM_ENTRY(dense_gemv_tcm_bf16, false, 1, 8, 1)
TCM_ENTRY(dense_gemv_tcm_bf16_ku4, false, 1, 4, 2)
TCM_ENTRY(dense_gemv_tcm_bf16_nt2, false, 2, 4, 1)
TCM_ENTRY(dense_gemv_tcm_fp8w, true, 1, 4, 2)
TCM_ENTRY(dense_gemv_tcm_fp8w_ku8, true, 1, 8, 1)
TCM_ENTRY(dense_gemv_tcm_fp8w_nt2, true, 2, 4, 1)
