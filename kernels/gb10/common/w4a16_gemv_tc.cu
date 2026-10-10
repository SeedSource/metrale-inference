// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: NVFP4 W4A16 GEMV for 1 to 16 rows on tensor cores (`mma.sync.m16n8k16`, BF16
// operands, FP32 accumulate): C[m, n] = sum_k A[m, k] * dequant(B[n, k]). The entries take the
// arguments of the CUDA-core `w4a16_gemv_batch*` kernels in w4a16_gemv.cu, and
// `ops::w4a16_gemv_batchm` launches them in their place when gemv_tc::tc_kernel routes the shape.
//
// Owner: gb10 kernels.
// Invariants:
// - A [M, K] BF16, B_packed [N, K / 2] E2M1 pairs, B_scale [N, K / 16] E4M3, scale2 FP32,
//   C [M, N] BF16. The caller guarantees K % 128 == 0 and 1 <= M <= MT (8 for tc8, 16 for
//   tc16); gemv_tc::tc_route checks both. Any N: a weight row n >= N loads zeros.
// - Grid (ceil(N / (8 * NT)), 1, 1), block TC_WARPS * 32 = 256. Only C[m, n] with m < M and
//   n < N is written. The `_group` entries (2026-10-09) take any grid and loop tiles; see
//   w4a16_gemv_tc_group_impl.
// - The TC_WARPS warps of a CTA take interleaved 128-value K blocks for the same 8 * NT
//   columns and are summed through shared memory in warp order, so a result does not depend on
//   scheduling.
//
// The work per weight byte does not depend on M: 8 E2M1 nibbles become four BF16x2 MMA
// operands, and one MMA covers all rows. Measured 2026-09-23 on GB10: the CUDA-core tiers drew
// 72-86 W on the GPU rail against about 50 W for tensor-core paths streaming the same bytes.
//
// Dequant: an E2M1 nibble s|e1 e0|m placed at BF16 bits 15 | 8..7 | 6 reads as e2m1 * 2^-126
// (e >= 1 is a normal with exponent field e; e = 0, m = 1 is the BF16 subnormal 2^-127). The
// E4M3 group scale becomes BF16 scale * 2^100, exact because E4M3 has 3 mantissa bits and every
// value stays normal. One BF16x2 multiply forms w' = e2m1 * scale * 2^-26, also exact: a 2-bit
// times a 4-bit significand fits BF16's 8 bits, and the product is normal. The MMA accumulates
// A * w' in FP32 and the epilogue multiplies by scale2 * 2^26. The result differs from the
// CUDA-core tiers only through the order and grouping of the FP32 products and sums; the
// model-arch example w4a16_gemv_tc_oracle bounds that difference.
//
// K permutation, so the checkpoint's [N, K / 2] layout needs no repack: a dot product does not
// change when the same permutation of k is applied to both operands. Thread (g = lane / 4,
// t = lane % 4) reads 16 contiguous weight bytes (32 k values) of row n = n0 + 8 * i + g; a quad
// covers 128 contiguous k, 64 contiguous bytes. Each 32-bit word q (k0..k7) yields the operand
// pairs P0 = (k3, k7), P1 = (k2, k6), P2 = (k1, k5), P3 = (k0, k4), and the activation operand
// for the same slot is built by the matching byte permutation of the thread's own row-g
// activations. The MMA's k slots 2t, 2t + 1, 2t + 8, 2t + 9 belong to lane % 4 == t in both the
// A and B fragments, so the two permutations agree.



















#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define TC_WARPS 8

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

__device__ __forceinline__ void w4tc_store2(__nv_bfloat16* p, float a, float b, bool paired,
                                            bool b_live, bool a_live) {
    if (paired) {
        *(__nv_bfloat162*)p = __floats2bfloat162_rn(a, b);
    } else {
        if (a_live) p[0] = __float2bfloat16_rn(a);
        if (b_live) p[1] = __float2bfloat16_rn(b);
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
    __nv_bfloat16* __restrict__ C,
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
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K)
{
    w4a16_gemv_tc_tile<MT, NT, KU>(A, B_packed, B_scale, scale2, C, M, N, K,
                                   blockIdx.x * (8u * NT));
}

// 2026-10-09: METRALE_GLM_NV4_TC_GROUP. Up to three projections that read the same A [M, K]
// (weights g = 0..2 of N_g columns each; N_g = 0 drops entry g) in one persistent launch. The
// tiles are numbered entry by entry: entry 0's ceil(N_0 / (8 * NT)) tiles, then entry 1's, then
// entry 2's. CTA b runs tiles b, b + gridDim.x, ... with w4a16_gemv_tc_tile on the entry's own
// weight, scale, scale2, output and N, at its local column, exactly the arguments the per-CTA
// entry's block for that column gets, so every output byte equals the per-projection launch.
// The barrier after a tile keeps its split-K reduction reads of `red` ahead of the next tile's
// writes. Grid: any size >= 1 (the host passes min(tiles, SMs x resident CTAs per SM)).
template <int MT, int NT, int KU>
__device__ __forceinline__ void w4a16_gemv_tc_group_impl(
    const __nv_bfloat16* __restrict__ A, unsigned int M, unsigned int K,
    const unsigned char* Bp0, const unsigned char* Bs0, float s0, __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* Bp1, const unsigned char* Bs1, float s1, __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* Bp2, const unsigned char* Bs2, float s2, __nv_bfloat16* C2, unsigned int N2)
{
    const unsigned int cols = 8u * NT;
    const unsigned int e0 = (N0 + cols - 1u) / cols;
    const unsigned int e1 = e0 + (N1 + cols - 1u) / cols;
    const unsigned int e2 = e1 + (N2 + cols - 1u) / cols;
    for (unsigned int tile = blockIdx.x; tile < e2; tile += gridDim.x) {
        const unsigned char* bp;
        const unsigned char* bs;
        float sc;
        __nv_bfloat16* c;
        unsigned int n, local;
        if (tile < e0) {
            bp = Bp0; bs = Bs0; sc = s0; c = C0; n = N0; local = tile;
        } else if (tile < e1) {
            bp = Bp1; bs = Bs1; sc = s1; c = C1; n = N1; local = tile - e0;
        } else {
            bp = Bp2; bs = Bs2; sc = s2; c = C2; n = N2; local = tile - e1;
        }
        w4a16_gemv_tc_tile<MT, NT, KU>(A, bp, bs, sc, c, M, n, K, local * cols);
        __syncthreads();
    }
}

#define W4TC_ENTRY(NAME, MT, NT, KU)                                                   \
    extern "C" __global__ __launch_bounds__(TC_WARPS * 32) void NAME(                    \
        const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,  \
        const unsigned char* __restrict__ B_scale, const float scale2,                    \
        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {  \
        w4a16_gemv_tc_impl<MT, NT, KU>(A, B_packed, B_scale, scale2, C, M, N, K);        \
    }

// 2026-09-25: tc8 (MT 8) serves M <= 8, and its A fragment's rows 8..15 are zero; tc16 (MT 16)
// serves M <= 16. gemv_tc::tc_route picks tc8 whenever M <= 8 and tc8 resolved.




W4TC_ENTRY(w4a16_gemv_tc8, 8, 1, 8)
W4TC_ENTRY(w4a16_gemv_tc16, 16, 2, 2)

#define W4TC_GROUP_ENTRY(NAME, MT, NT, KU)                                              \
    extern "C" __global__ __launch_bounds__(TC_WARPS * 32) void NAME(                    \
        const __nv_bfloat16* __restrict__ A, unsigned int M, unsigned int K,              \
        const unsigned char* Bp0, const unsigned char* Bs0, float s0,                     \
        __nv_bfloat16* C0, unsigned int N0,                                               \
        const unsigned char* Bp1, const unsigned char* Bs1, float s1,                     \
        __nv_bfloat16* C1, unsigned int N1,                                               \
        const unsigned char* Bp2, const unsigned char* Bs2, float s2,                     \
        __nv_bfloat16* C2, unsigned int N2) {                                             \
        w4a16_gemv_tc_group_impl<MT, NT, KU>(A, M, K, Bp0, Bs0, s0, C0, N0, Bp1, Bs1, s1, \
                                             C1, N1, Bp2, Bs2, s2, C2, N2);              \
    }

// 2026-10-09: The grouped persistent twins of tc8 / tc16 (same MT, NT, KU).
W4TC_GROUP_ENTRY(w4a16_gemv_tc8_group, 8, 1, 8)
W4TC_GROUP_ENTRY(w4a16_gemv_tc16_group, 16, 2, 2)
