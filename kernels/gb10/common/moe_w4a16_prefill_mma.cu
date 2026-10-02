// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-01: Grouped W4A16 expert GEMMs for the GLM-5.3 routed-MoE PREFILL, behind
// METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1 (model-arch glm5next_mlp/forward_prefill_gemm/w4a16_mma.rs).
// BF16 activations times NVFP4 weights (E2M1 nibbles, low nibble = even k, one E4M3 scale per 16 k,
// one f32 scale2 per expert and projection), mma.sync m16n8k16 BF16 with F32 accumulators.
//
// Design (written for this file; the register-dequant idea is the one vLLM's Marlin kernels use,
// no code is taken from them):
//   - Rows arrive sorted by expert (moe_sort_by_expert). `moe_w4a16_prefill_tile_list` turns
//     expert_offsets into a compact list of (expert, M tile) work items over the LOCAL experts
//     only, so no CTA is launched for an empty or remote expert's tile slot.
//   - Grid (N tiles, tile-list capacity). blockIdx.x (fast) is the N tile, so the CTAs that
//     share one A tile run back to back, and the next M tile of the same expert follows at once,
//     re-reading that expert's weights from L2 rather than DRAM.
//   - A, the packed weight bytes and the E4M3 scale bytes all travel global -> shared by
//     cp.async in a STAGES-deep ring (one __syncthreads per 64-wide K step). The weights stay
//     packed in shared memory; each warp decodes its own B fragments in registers (E2M1 via two
//     byte permutes, times e4m3 * scale2 in F32, one cvt.rn.bf16x2), so no dequantised tile
//     ever goes back through shared memory.
//   - Within each 32-wide k block a thread owns 8 CONSECUTIVE k values of a B row (one 32-bit
//     shared load) and the matching 8 k of its two A rows (one 16-byte shared load each). The two
//     MMAs of the block see k in a fixed permuted order; a dot product does not depend on the
//     order of its terms mathematically, but the F32 sums are not bit-identical to the
//     production `moe_w4a16_grouped_gemm_ptrtable_*` kernels.
//   - Each dequantised weight is __float2bfloat16(e2m1 * (e4m3 * scale2)), the same F32 product
//     and rounding as the production `ARITH_LUT` tiles (`e2m1_decode(n) * sc`), so the BF16 B
//     values are bit-identical to theirs; only the accumulation order differs.
//   - `_gateup_silu`: one CTA computes gate AND up for the same 64 output columns (shared-memory
//     B rows 0..63 gate, 64..127 up). A thread holds gate and up accumulators for the same
//     (row, col), so GLM's clamped SwiGLU runs in the epilogue on the BF16-rounded gate and up,
//     with glm5next_swiglu_clamp's exact expression, and only the activation is written.
//   - `_down`: the plain grouped GEMM (128 output columns per CTA), A already in sorted order.
//
// Shape contract (the host checks every item before launching, else takes the production path):
// K % 64 == 0; N % 64 == 0 (gateup) or N % 128 == 0 (down); A 16-byte aligned; each local
// expert's packed weights 4-byte aligned (16-byte aligned takes the fast copy) and its scales
// 1-byte aligned (4-byte aligned takes the async copy); num_experts <= 1024; tile-list capacity
// >= the real tile count (each local expert adds at most one partial tile, so
// ceil(rows / MT) + local experts suffices).
//
// Owner: gb10 kernels.
// Invariants:
// - A row's output depends only on its A row, its expert's weights and the fixed k order, never
//   on the other rows in its tile, the tile list, or the grid.
// - Rows past an expert's count are never stored; an expert with a NULL weight pointer gets no
//   tile, so its rows are left untouched (EP: the caller pre-zeroes the routed output).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define PMMA_KS 64
#define PMMA_MAX_EXPERTS 1024

// 2026-10-01: Two E2M1 values (one packed byte x: low nibble first) as a BF16 pair, unscaled.
// Magnitude idx 0..7 = {0, 0.5, 1, 1.5, 2, 3, 4, 6}; their BF16 bit patterns are
// 0x0000 0x3F00 0x3F80 0x3FC0 0x4000 0x4040 0x4080 0x40C0. The low bytes {00 00 80 C0 00 40 80 C0}
// and high bytes {00 3F 3F 3F 40 40 40 40} are 8-byte tables for __byte_perm; table byte 0 is
// 0x00 in both, which fills the unused byte slots. Bit 3 of each nibble is the sign (bit 15 of
// its BF16). Every value is a normal BF16 or a signed zero, so the F32 widening below is exact
// and FTZ cannot touch it; it equals e2m1_decode(n) (moe_w4a16_grouped_gemm.cu) for all 16 codes.
__device__ __forceinline__ unsigned int pmma_e2m1_pair_bf16(unsigned int x) {
    const unsigned int sel = (x & 0x7u) | ((x << 4) & 0x700u);
    const unsigned int lo = __byte_perm(0xC0800000u, 0xC0804000u, sel);
    const unsigned int hi = __byte_perm(0x3F3F3F00u, 0x40404040u, sel << 4);
    const unsigned int sgn = ((x & 0x08u) << 12) | ((x & 0x80u) << 24);
    return lo | hi | sgn;
}

// 2026-10-01: One packed byte -> the BF16x2 B-fragment register: low half = even k. Each half is
// __float2bfloat16(e2m1 * sc) with sc = e4m3 * scale2, as in the production ARITH_LUT staging.
__device__ __forceinline__ unsigned int pmma_dq2(unsigned int x, float sc) {
    const unsigned int v = pmma_e2m1_pair_bf16(x);
    const float f0 = __uint_as_float(v << 16);
    const float f1 = __uint_as_float(v & 0xFFFF0000u);
    __nv_bfloat162 r = __floats2bfloat162_rn(f0 * sc, f1 * sc);
    return *reinterpret_cast<unsigned int*>(&r);
}

__device__ __forceinline__ void pmma_mma(float* c, unsigned int a0, unsigned int a1,
                                         unsigned int a2, unsigned int a3,
                                         unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void pmma_cp16(void* dst, const void* src) {
    const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src) : "memory");
}

__device__ __forceinline__ void pmma_cp4(void* dst, const void* src) {
    const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(d), "l"(src) : "memory");
}

__device__ __forceinline__ void pmma_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void pmma_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

// 2026-10-01: The activation glm5next_swiglu_clamp (glm5next_ffn.cu) computes, on this thread's
// F32 accumulators rounded to BF16 first, exactly as that kernel reads a_gate / a_up.
__device__ __forceinline__ float pmma_swiglu(float gacc, float uacc, float limit) {
    float g = __bfloat162float(__float2bfloat16(gacc));
    float u = __bfloat162float(__float2bfloat16(uacc));
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    const float s = g / (1.0f + expf(-g));
    return s * u;
}

// 2026-10-01: Compact (expert, M tile) list. tiles[0] = number of entries (clamped to cap);
// tiles[1 + i] = (expert << 16) | m_tile, in ascending expert then m_tile order. An expert gets
// ceil(rows / m_tile) entries when its packed pointer is non-NULL, none otherwise. One block;
// any blockDim. Refuses (writes count 0) when num_experts > PMMA_MAX_EXPERTS or m_tile == 0.
extern "C" __global__ void moe_w4a16_prefill_tile_list(
    const int* __restrict__ expert_offsets,             // 2026-10-01: [num_experts + 1]
    const unsigned long long* __restrict__ packed_ptrs, // 2026-10-01: [num_experts], NULL = remote
    unsigned int* __restrict__ tiles,                   // 2026-10-01: [1 + cap]
    unsigned int num_experts,
    unsigned int m_tile,
    unsigned int cap
) {
    __shared__ unsigned int s_start[PMMA_MAX_EXPERTS];
    __shared__ unsigned int s_total;
    if (num_experts > PMMA_MAX_EXPERTS || m_tile == 0u) {
        if (threadIdx.x == 0) tiles[0] = 0u;
        return;
    }
    for (unsigned int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        const int rows = expert_offsets[e + 1] - expert_offsets[e];
        s_start[e] = (packed_ptrs[e] != 0ull && rows > 0)
            ? ((unsigned int)rows + m_tile - 1u) / m_tile : 0u;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int run = 0u;
        for (unsigned int e = 0; e < num_experts; e++) {
            const unsigned int c = s_start[e];
            s_start[e] = run;
            run += c;
        }
        s_total = run;
    }
    __syncthreads();
    for (unsigned int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        const int rows = expert_offsets[e + 1] - expert_offsets[e];
        const unsigned int c = (packed_ptrs[e] != 0ull && rows > 0)
            ? ((unsigned int)rows + m_tile - 1u) / m_tile : 0u;
        const unsigned int start = s_start[e];
        for (unsigned int i = 0; i < c; i++) {
            if (start + i < cap) tiles[1u + start + i] = (e << 16) | i;
        }
    }
    if (threadIdx.x == 0) tiles[0] = s_total < cap ? s_total : cap;
}

// 2026-10-01: The GEMM core. MT rows per CTA (64 or 128): MT / 64 warp rows times 4 warp
// columns, each warp a 64 x 32 tile of B rows (4 m16 x 4 n8 accumulators). STAGES-deep cp.async
// ring of 64-wide K steps in dynamic shared memory; per stage:
//   A  [2 k32 blocks][MT rows][32 BF16]  (MT * 128 B)
//   B  [2 k32 blocks][128 rows][16 B]    (4096 B, packed E2M1)
//   S  [128 rows][4 E4M3]                (512 B)
// FUSED: B rows 0..63 are gate rows n0..n0+63 and 64..127 up rows n0..n0+63, N is the
// activation width and n0 = blockIdx.x * 64. Otherwise B rows are n0..n0+127, n0 =
// blockIdx.x * 128.
template <int MT, int STAGES, bool FUSED>
__device__ __forceinline__ void pmma_core(
    const __nv_bfloat16* __restrict__ A,
    const int* __restrict__ sorted_token_ids,
    const unsigned long long* __restrict__ p0,
    const unsigned long long* __restrict__ sp0,
    const float* __restrict__ s2v0,
    const unsigned long long* __restrict__ p1,
    const unsigned long long* __restrict__ sp1,
    const float* __restrict__ s2v1,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const unsigned int* __restrict__ tiles,
    unsigned int N,
    unsigned int K,
    float limit
) {
    constexpr int WARPS_M = MT / 64;
    constexpr int THREADS = WARPS_M * 4 * 32;
    constexpr int A_STAGE = MT * PMMA_KS * 2;
    constexpr int B_STAGE = 128 * (PMMA_KS / 2);
    constexpr int S_STAGE = 128 * (PMMA_KS / 16);
    constexpr int STAGE = A_STAGE + B_STAGE + S_STAGE;
    constexpr int B_CPT = 256 / THREADS;
    static_assert(MT == 64 || MT == 128, "MT is 64 or 128");
    static_assert(MT * 8 == 4 * THREADS, "four 16-byte A chunks per thread per stage");
    static_assert(STAGES >= 2, "a ring needs two stages");
    static_assert(STAGE % 16 == 0 && A_STAGE % 16 == 0 && B_STAGE % 16 == 0,
                  "every stage region must stay 16-byte aligned for cp.async");

    extern __shared__ __align__(16) unsigned char pmma_smem[];

    const unsigned int count = tiles[0];
    if (blockIdx.y >= count) return;
    const unsigned int entry = tiles[1u + blockIdx.y];
    const unsigned int e = entry >> 16;
    const int m_start = expert_offsets[e] + (int)((entry & 0xFFFFu) * (unsigned int)MT);
    const int m_left = expert_offsets[e + 1] - m_start;
    const int m_rows = m_left < MT ? m_left : MT;
    if (m_rows <= 0) return;

    const unsigned char* B0 = (const unsigned char*)p0[e];
    const unsigned char* S0 = (const unsigned char*)sp0[e];
    const float sc2_0 = s2v0[e];
    const unsigned char* B1 = FUSED ? (const unsigned char*)p1[e] : B0;
    const unsigned char* S1 = FUSED ? (const unsigned char*)sp1[e] : S0;
    const float sc2_1 = FUSED ? s2v1[e] : sc2_0;
    // 2026-10-01: The tile list never names a remote expert; this only guards a caller error.
    if (B0 == 0 || B1 == 0) return;

    const unsigned int half_K = K >> 1;
    const unsigned int num_groups = K >> 4;
    const unsigned int n0 = blockIdx.x * (FUSED ? 64u : 128u);

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int warp_m = (warp >> 2) * 64;
    const int wn = warp & 3;
    const bool warp_active = warp_m < m_rows;

    // 2026-10-01: A staging. Chunk c = tid + j * THREADS (j < 4) of a stage is k32 block
    // c / (MT * 4), row (c % (MT * 4)) / 4, quarter c % 4: rows tid / 4 and tid / 4 + MT / 2,
    // quarter tid % 4, both k32 blocks. Eight consecutive threads fill 128 contiguous shared
    // bytes. A row past the expert's count re-reads the tile's first row: its products land
    // only in rows that are never stored.
    const int a_q = tid & 3;
    const int a_r0 = tid >> 2;
    const __nv_bfloat16* a_src[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = a_r0 + i * (MT / 2);
        const int grow = m_start + (r < m_rows ? r : 0);
        const unsigned int arow = sorted_token_ids
            ? (unsigned int)sorted_token_ids[grow] : (unsigned int)grow;
        a_src[i] = A + (unsigned long long)arow * K + (unsigned int)(a_q * 8);
    }

    // 2026-10-01: B / S staging sources: shared row r (0..127) of this CTA.
    auto b_row = [&](int r) -> const unsigned char* {
        if (FUSED) {
            return (r < 64) ? (B0 + (unsigned long long)(n0 + (unsigned int)r) * half_K)
                            : (B1 + (unsigned long long)(n0 + (unsigned int)(r - 64)) * half_K);
        }
        return B0 + (unsigned long long)(n0 + (unsigned int)r) * half_K;
    };
    auto s_row = [&](int r) -> const unsigned char* {
        if (FUSED) {
            return (r < 64) ? (S0 + (unsigned long long)(n0 + (unsigned int)r) * num_groups)
                            : (S1 + (unsigned long long)(n0 + (unsigned int)(r - 64)) * num_groups);
        }
        return S0 + (unsigned long long)(n0 + (unsigned int)r) * num_groups;
    };
    const bool b16 = ((((unsigned long long)B0) | ((unsigned long long)B1)) & 15ull) == 0ull;
    const bool s4 = ((((unsigned long long)S0) | ((unsigned long long)S1)) & 3ull) == 0ull;

    auto load_stage = [&](int kt, int slot) {
        unsigned char* st = pmma_smem + slot * STAGE;
        const unsigned int k0 = (unsigned int)kt * PMMA_KS;
        #pragma unroll
        for (int kb = 0; kb < 2; kb++) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int r = a_r0 + i * (MT / 2);
                pmma_cp16(st + kb * (MT * 64) + r * 64 + a_q * 16, a_src[i] + k0 + kb * 32);
            }
        }
        unsigned char* sb = st + A_STAGE;
        #pragma unroll
        for (int j = 0; j < B_CPT; j++) {
            const int c = tid + j * THREADS;
            const int kb = c >> 7;
            const int r = c & 127;
            const unsigned char* src = b_row(r) + (k0 >> 1) + kb * 16;
            unsigned char* dst = sb + kb * 2048 + r * 16;
            if (b16) {
                pmma_cp16(dst, src);
            } else {
                #pragma unroll
                for (int q = 0; q < 4; q++) pmma_cp4(dst + q * 4, src + q * 4);
            }
        }
        unsigned char* ss = sb + B_STAGE;
        if (tid < 128) {
            const unsigned char* src = s_row(tid) + (k0 >> 4);
            if (s4) {
                pmma_cp4(ss + tid * 4, src);
            } else {
                // 2026-10-01: Unaligned scales: plain loads. The slot is free (the barrier of
                // this iteration) and is read only after a later barrier.
                #pragma unroll
                for (int q = 0; q < 4; q++) ss[tid * 4 + q] = src[q];
            }
        }
    };

    float acc[4][4][4];
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            acc[mi][nt][0] = 0.0f; acc[mi][nt][1] = 0.0f;
            acc[mi][nt][2] = 0.0f; acc[mi][nt][3] = 0.0f;
        }
    }

    // 2026-10-01: Shared B row of n8 tile nt for this thread (fragment column g).
    int brow[4];
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        brow[nt] = FUSED ? ((nt >> 1) * 64 + wn * 16 + (nt & 1) * 8 + g)
                         : (wn * 32 + nt * 8 + g);
    }

    auto compute = [&](int slot) {
        const unsigned char* st = pmma_smem + slot * STAGE;
        const unsigned char* sA = st;
        const unsigned char* sB = st + A_STAGE;
        const unsigned char* sS = sB + B_STAGE;
        #pragma unroll
        for (int kb = 0; kb < 2; kb++) {
            // 2026-10-01: This thread's B: physical k 8t..8t+7 of the k32 block (bytes 4t..4t+3
            // of the row's 16), all in scale group t / 2. mma #0 takes bytes 0, 1 (k 8t..8t+3),
            // mma #1 bytes 2, 3 (k 8t+4..8t+7); A below supplies the same k in the same slots.
            unsigned int bq[4][4];
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const unsigned int w =
                    *(const unsigned int*)(sB + kb * 2048 + brow[nt] * 16 + t * 4);
                __nv_fp8_e4m3 f8;
                *(unsigned char*)&f8 = sS[brow[nt] * 4 + kb * 2 + (t >> 1)];
                const float sc = (float)f8 * ((FUSED && nt >= 2) ? sc2_1 : sc2_0);
                bq[nt][0] = pmma_dq2(w & 0xFFu, sc);
                bq[nt][1] = pmma_dq2((w >> 8) & 0xFFu, sc);
                bq[nt][2] = pmma_dq2((w >> 16) & 0xFFu, sc);
                bq[nt][3] = pmma_dq2(w >> 24, sc);
            }
            // 2026-10-01: Rows of a 16-row slice past the expert's count hold copies of the tile's
            // first row (always staged), so the loads are safe; only the MMAs are skipped.
            uint4 lo[4], hi[4];
            #pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                const int ar = warp_m + mi * 16 + g;
                lo[mi] = *(const uint4*)(sA + kb * (MT * 64) + ar * 64 + t * 16);
                hi[mi] = *(const uint4*)(sA + kb * (MT * 64) + (ar + 8) * 64 + t * 16);
            }
            // 2026-10-01: All mma #0 first, then all mma #1: the two MMAs that chain on one
            // accumulator are 16 independent MMAs apart instead of back to back (the asm is
            // volatile, so the compiler keeps this order).
            #pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                if (warp_m + mi * 16 >= m_rows) continue;
                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    pmma_mma(acc[mi][nt], lo[mi].x, hi[mi].x, lo[mi].y, hi[mi].y,
                             bq[nt][0], bq[nt][1]);
                }
            }
            #pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                if (warp_m + mi * 16 >= m_rows) continue;
                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    pmma_mma(acc[mi][nt], lo[mi].z, hi[mi].z, lo[mi].w, hi[mi].w,
                             bq[nt][2], bq[nt][3]);
                }
            }
        }
    };

    // 2026-10-01: The ring. Before iteration kt, STAGES - 1 + kt groups are committed (one per
    // prologue slot, one per iteration, empty ones included), so wait_group STAGES - 2 means
    // stage kt has landed for this thread; the barrier then publishes every thread's copies AND
    // proves every warp finished computing stage kt - 1, whose slot the new load reuses.
    const int KT = (int)(K / PMMA_KS);
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < KT) load_stage(s, s);
        pmma_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        pmma_wait<STAGES - 2>();
        __syncthreads();
        const int nk = kt + STAGES - 1;
        if (nk < KT) load_stage(nk, nk % STAGES);
        pmma_commit();
        if (warp_active) compute(kt % STAGES);
    }
    pmma_wait<0>();

    if (!warp_active) return;

    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        const int r0 = warp_m + mi * 16 + g;
        const int r1 = r0 + 8;
        if (FUSED) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                const unsigned int col = n0 + (unsigned int)(wn * 16 + j * 8 + t * 2);
                if (r0 < m_rows) {
                    __nv_bfloat162 o = __floats2bfloat162_rn(
                        pmma_swiglu(acc[mi][j][0], acc[mi][j + 2][0], limit),
                        pmma_swiglu(acc[mi][j][1], acc[mi][j + 2][1], limit));
                    *(__nv_bfloat162*)(C + (unsigned long long)(m_start + r0) * N + col) = o;
                }
                if (r1 < m_rows) {
                    __nv_bfloat162 o = __floats2bfloat162_rn(
                        pmma_swiglu(acc[mi][j][2], acc[mi][j + 2][2], limit),
                        pmma_swiglu(acc[mi][j][3], acc[mi][j + 2][3], limit));
                    *(__nv_bfloat162*)(C + (unsigned long long)(m_start + r1) * N + col) = o;
                }
            }
        } else {
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const unsigned int col = n0 + (unsigned int)(wn * 32 + nt * 8 + t * 2);
                if (r0 < m_rows) {
                    *(__nv_bfloat162*)(C + (unsigned long long)(m_start + r0) * N + col) =
                        __floats2bfloat162_rn(acc[mi][nt][0], acc[mi][nt][1]);
                }
                if (r1 < m_rows) {
                    *(__nv_bfloat162*)(C + (unsigned long long)(m_start + r1) * N + col) =
                        __floats2bfloat162_rn(acc[mi][nt][2], acc[mi][nt][3]);
                }
            }
        }
    }
}

// 2026-10-01: Dynamic shared memory per launch = STAGES * (MT * 128 + 4608) bytes:
// m128 (4 stages) 83,968 B; m64 (3 stages) 38,400 B. The host passes exactly this
// (w4a16_mma.rs `PrefillMmaTile::smem_bytes`).
#define PMMA_GATEUP(SUFFIX, MT, STAGES)                                                   \
extern "C" __global__ __launch_bounds__((MT / 64) * 128)                                  \
void moe_w4a16_prefill_mma_gateup_silu_##SUFFIX(                                          \
    const __nv_bfloat16* __restrict__ A, const int* __restrict__ sorted_token_ids,        \
    const unsigned long long* __restrict__ gate_ptrs,                                     \
    const unsigned long long* __restrict__ gate_scale_ptrs,                               \
    const float* __restrict__ gate_scale2,                                                \
    const unsigned long long* __restrict__ up_ptrs,                                       \
    const unsigned long long* __restrict__ up_scale_ptrs,                                 \
    const float* __restrict__ up_scale2,                                                  \
    __nv_bfloat16* __restrict__ act, const int* __restrict__ expert_offsets,              \
    const unsigned int* __restrict__ tiles, unsigned int N, unsigned int K, float limit)  \
{                                                                                         \
    pmma_core<MT, STAGES, true>(A, sorted_token_ids, gate_ptrs, gate_scale_ptrs,          \
        gate_scale2, up_ptrs, up_scale_ptrs, up_scale2, act, expert_offsets, tiles,       \
        N, K, limit);                                                                     \
}

#define PMMA_DOWN(SUFFIX, MT, STAGES)                                                     \
extern "C" __global__ __launch_bounds__((MT / 64) * 128)                                  \
void moe_w4a16_prefill_mma_down_##SUFFIX(                                                 \
    const __nv_bfloat16* __restrict__ A, const int* __restrict__ sorted_token_ids,        \
    const unsigned long long* __restrict__ ptrs,                                          \
    const unsigned long long* __restrict__ scale_ptrs,                                    \
    const float* __restrict__ scale2,                                                     \
    __nv_bfloat16* __restrict__ C, const int* __restrict__ expert_offsets,                \
    const unsigned int* __restrict__ tiles, unsigned int N, unsigned int K)               \
{                                                                                         \
    pmma_core<MT, STAGES, false>(A, sorted_token_ids, ptrs, scale_ptrs, scale2,           \
        ptrs, scale_ptrs, scale2, C, expert_offsets, tiles, N, K, 0.0f);                  \
}

PMMA_GATEUP(m128, 128, 4)
PMMA_GATEUP(m64, 64, 3)
PMMA_DOWN(m128, 128, 4)
PMMA_DOWN(m64, 64, 3)
