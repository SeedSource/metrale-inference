// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GLM-5.3-Flash DSA (DeepSeek Sparse Attention) kernels: the kpool indexer and
// the NoPE MLA oracle.
//
// Owner: gb10 kernels.
// Invariants:
// - Pipeline order: dsa_kpool_compress (pool keys, token ids, validity) -> dsa_index_scores
//   (per-(query, pool) score and candidacy) -> dsa_topk_pools (deterministic top-k) ->
//   dsa_expand_selection (raw token ids, tail appended, -1 padded). dsa_topk_to_mask and
//   dsa_mla_masked_attn are the oracle's dense path; dsa_compact_pools gathers kept pools.
// - -1 (DSA_INVALID) marks an invalid index. dsa_expand_selection writes -1 over its whole
//   output row and synchronises before it stores any real index, so every slot of the row
//   is written on every path, including a masked query.
// - dsa_mla_masked_attn has no rope section: q and k share one head dim qd, and `scale` is
//   an argument, never derived here.







#include <cuda_bf16.h>
#include <math_constants.h>
#include <float.h>
#include <limits.h>

#define DSA_INVALID (-1)

__device__ __forceinline__ float dsa_block_sum(float v, float* smem, unsigned tid, unsigned nthreads) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if ((tid & 31u) == 0u) smem[tid >> 5] = v;
    __syncthreads();
    if (tid < 32u) {
        float x = (tid < ((nthreads + 31u) / 32u)) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffff, x, off);
        if (tid == 0u) smem[0] = x;
    }
    __syncthreads();
    return smem[0];
}

// 2026-09-25: 1. kpool compression. One block per pool; threads stride over channels. The
// softmax runs over the pool-slot axis, independently per channel. Pool p covers tokens
// first_key + p*KP + s, so left padding before first_key is skipped. A pool is valid only
// if every slot is in range and valid, so a trailing partial pool is never valid. KP must
// be <= 8 (lg[8]); config validation refuses a larger index_kpool (KERNEL_MAX_KPOOL).




// 2026-09-25: Replay-safe geometry. A CUDA graph fixes every scalar argument at capture time,
// but S, the pool counts, the top-k tile and select_k grow with the context. Each selector
// kernel therefore takes `geom`, a 5-int device vector that dsa_write_geom fills once per
// step from seq_len. When non-null it overrides the scalar arguments; when null the scalars
// are used as passed. Pool-indexed blocks past the live pool count return at once, so a
// ceiling launch can fix the grid at the context ceiling. Slots: S (tokens in the cache),
// pools including the trailing partial one, complete pools, select_k, top-k tile width.






#define DSA_GEOM_S        0
#define DSA_GEOM_NPOOLS_F 1
#define DSA_GEOM_NPOOLS   2
#define DSA_GEOM_SELECT_K 3
#define DSA_GEOM_NP2      4

// 2026-09-25: One thread. Fills the geom slots from seq_len[0]. `tile` is the top-k tile width
// (a power of two); np2 = min(max(2, next power of two >= complete pools), tile).
extern "C" __global__ void dsa_write_geom(
    const int* __restrict__ seq_len,
    int* __restrict__ geom,
    unsigned int KP,
    unsigned int topk,
    unsigned int tile
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int S = seq_len[0];
    const int np = S / (int)KP;


    int np2 = 2;
    while (np2 < np && np2 < (int)tile) np2 <<= 1;
    const int cap = (int)(topk / KP);
    geom[DSA_GEOM_S] = S;
    geom[DSA_GEOM_NPOOLS_F] = (S + (int)KP - 1) / (int)KP;
    geom[DSA_GEOM_NPOOLS] = np;
    geom[DSA_GEOM_SELECT_K] = np < cap ? np : cap;
    geom[DSA_GEOM_NP2] = np2;
}

// 2026-09-25: Copies one staged indexer row (k and gate) into the cache at the device-side
// row pos[0] and marks it valid, so a replayed graph writes the live row.



extern "C" __global__ void dsa_indexer_store(
    const __nv_bfloat16* __restrict__ stage_k,
    const __nv_bfloat16* __restrict__ stage_gate,
    const int* __restrict__ pos,
    __nv_bfloat16* __restrict__ k_normed,
    __nv_bfloat16* __restrict__ gate,
    unsigned char* __restrict__ valid,
    unsigned int D
) {
    const size_t base = (size_t)pos[0] * D;
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        k_normed[base + d] = stage_k[d];
        gate[base + d] = stage_gate[d];
    }
    if (threadIdx.x == 0) valid[pos[0]] = 1;
}

extern "C" __global__ void dsa_kpool_compress(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ gate,
    const unsigned char* __restrict__ valid,
    const float* __restrict__ ape,
    float* __restrict__ pool_keys,
    int* __restrict__ pool_indices,
    unsigned char* __restrict__ pool_valid,
    unsigned int S,
    unsigned int D,
    unsigned int KP,
    int first_key,
    const int* __restrict__ geom
) {
    const unsigned int p = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        // 2026-09-25: Under a ceiling launch this pool may not be live yet.
        if (p >= (unsigned int)geom[DSA_GEOM_NPOOLS_F]) return;
    }

    // 2026-09-25: Slot bookkeeping is the same for every channel, so thread 0 writes it.
    bool all_valid = true;
    for (unsigned int s = 0; s < KP; ++s) {
        long long raw = (long long)first_key + (long long)p * KP + s;
        bool in_range = raw >= 0 && raw < (long long)S;
        bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
        all_valid &= ok;
        if (tid == 0) pool_indices[p * KP + s] = ok ? (int)raw : DSA_INVALID;
    }
    if (tid == 0) pool_valid[p] = all_valid ? 1 : 0;

    for (unsigned int d = tid; d < D; d += blockDim.x) {
        float mx = -CUDART_INF_F;
        float lg[8];
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            long long raw = (long long)first_key + (long long)p * KP + s;
            bool in_range = raw >= 0 && raw < (long long)S;
            bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
            lg[s] = ok ? (__bfloat162float(gate[(size_t)raw * D + d]) + ape[s * D + d])
                       : -CUDART_INF_F;
            mx = fmaxf(mx, lg[s]);
        }
        float sum = 0.0f;
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            lg[s] = (lg[s] == -CUDART_INF_F) ? 0.0f : __expf(lg[s] - mx);
            sum += lg[s];
        }
        // 2026-09-25: A pool with no valid slot has sum 0, so its weights and key are 0.
        float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        float acc = 0.0f;
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            long long raw = (long long)first_key + (long long)p * KP + s;
            bool in_range = raw >= 0 && raw < (long long)S;
            bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
            if (ok) acc += lg[s] * inv * __bfloat162float(k[(size_t)raw * D + d]);
        }
        pool_keys[p * D + d] = acc;
    }
}

// 2026-09-25: 2. Per-(query, pool) index score, one block per (pool, query). `weights` must
// already carry the index_heads^-0.5 factor; the kernel applies only `scale` (the host
// passes index_head_dim^-0.5). score = sum_h weights[h] * relu(scale * dot(q_h, key)).

extern "C" __global__ void dsa_index_scores(
    const float* __restrict__ q,
    const float* __restrict__ pool_keys,
    const float* __restrict__ weights,
    const int* __restrict__ pool_indices,
    const unsigned char* __restrict__ pool_valid,
    const unsigned char* __restrict__ valid_keys,
    const int* __restrict__ q_pos,
    float* __restrict__ out,
    unsigned char* __restrict__ valid_cand,
    unsigned int Q,
    unsigned int P,
    unsigned int H,
    unsigned int D,
    unsigned int KP,
    unsigned int S,
    float scale,
    const int* __restrict__ geom
) {
    const unsigned int p = blockIdx.x;
    const unsigned int r = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    extern __shared__ float sh[];
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        // 2026-09-25: `P` is also the row stride of out / valid_cand, so it may vary with geom
        // only because the device-geometry path is decode-only (Q == 1, every r * P is 0).
        // The launcher refuses a ceiling launch with Q > 1.
        if (p >= P) return;
    }

    // 2026-09-25: A pool is a candidate only when it is complete and its last token, clamped
    // to [0, S-1], is at or before this query's position and valid. Others score -FLT_MAX.
    int end = pool_indices[p * KP + KP - 1];
    int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
    bool vis = (end_c <= q_pos[r]) && (valid_keys[end_c] != 0);
    bool cand = (pool_valid[p] != 0) && vis;
    if (tid == 0) valid_cand[(size_t)r * P + p] = cand ? 1 : 0;
    if (!cand) {
        if (tid == 0) out[(size_t)r * P + p] = -FLT_MAX;
        return;
    }

    // 2026-09-25: One warp per head: each lane accumulates every 32nd product, a shuffle tree
    // reduces them, and the head's term lands in sh[h]. Thread 0 then sums sh[0..H) in head
    // order. sh must hold H floats: the host requests max(SCORES_BLOCK, 4 * H) bytes.

















    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int nwarps = (blockDim.x + 31u) / 32u;
    const float* __restrict__ pk = pool_keys + (size_t)p * D;
    for (unsigned int h = warp; h < H; h += nwarps) {
        const float* __restrict__ qh = q + ((size_t)r * H + h) * D;
        float dot = 0.0f;
        for (unsigned int d = lane; d < D; d += 32u) dot += qh[d] * pk[d];
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) sh[h] = weights[(size_t)r * H + h] * fmaxf(scale * dot, 0.0f);
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int h = 0; h < H; ++h) acc += sh[h];
        out[(size_t)r * P + p] = acc;
    }
}

// 2026-10-01: 2b. dsa_index_scores_tiled: the same scores and candidacy as dsa_index_scores,
// byte-identical by construction, with one block per DSA_TILE_ROWS x DSA_TILE_POOLS tile
// instead of one per (pool, query), so a row's q is read once per tile, not once per pool.
// Opt-in (METRALE_GLM_DSA_SCORES_TILED=1); resolved with try_kernel; gated on a GPU by
// crates/model-arch/examples/dsa_indexer_tiled_bitparity_microtest.rs.
//
// Why the bits match (the common build passes --fmad=false, so a * b + c is a rounded
// multiply then a rounded add, as in dsa_index_scores):
// - In dsa_index_scores, lane l of a head's warp sums qh[d] * pk[d] over d = l, l + 32, ...
//   (d < D), ascending, from 0.0f. dsa_chain4 runs exactly that chain for lane l.
// - __shfl_down_sync with offsets 16, 8, 4, 2, 1 leaves at lane 0 the tree
//   x1[i] = x0[i] + x0[i + 16], x2[i] = x1[i] + x1[i + 8], ..., x5[0] = x4[0] + x4[1], x0
//   the lane chains, each add with the lower lane on the left. (A lane above 31 - off reads
//   its own value, but lane 0's result never depends on such a lane.) DsaTree4<I, K> is
//   x_K[I] = x_(K-1)[I] + x_(K-1)[I + (32 >> K)], and DsaTree4<0, 5> is x5[0].
// - The head term is weights[r * H + h] * fmaxf(scale * x5[0], 0.0f), and the score is
//   0.0f + term(0) + term(1) + ... in ascending h, as thread 0 sums sh[] there.
// - Candidacy, valid_cand and the -FLT_MAX store are dsa_index_scores' code, per output.
// Staging q and the pool keys in shared memory changes where operands are read from, not
// their values or the order they are combined in.
//
// Layout: DSA_TILE_THREADS threads (8 warps). Warp w owns tile rows 2w and 2w + 1, lane l
// owns tile pools l and l + 32: four outputs per thread, which share every q and key load.
// The tile's pool keys are staged once with row stride D + 1 (lane l reads bank
// (l + d) % 32: conflict-free); q is staged per head (rows are warp-uniform, so its reads
// broadcast). Dynamic shared memory: (DSA_TILE_POOLS * (D + 1) + DSA_TILE_ROWS * D) * 4
// bytes, 41,216 at D = 128. The depth-first tree keeps at most five partial quads plus the
// running chain live: about 24 accumulator registers for the four outputs (estimate, not a
// ptxas report).
//
// Not supported: the device-geometry (ceiling) path dsa_index_scores serves from `geom` at
// Q == 1. The host never selects this kernel with a non-NULL geom; a launch that does traps
// instead of reading frozen scalars.

#define DSA_TILE_ROWS 16u
#define DSA_TILE_POOLS 64u
#define DSA_TILE_THREADS 256u

// 2026-10-01: Four (row, pool) values: rows 0/1 x pools 0/1 of one thread.
struct DsaQuad {
    float r0p0, r0p1, r1p0, r1p1;
};

// 2026-10-01: dsa_index_scores lane l's chain for four (row, pool) pairs at once: ascending
// d from 0.0f, each product rounded before its add.
__device__ __forceinline__ DsaQuad dsa_chain4(
    const float* q0, const float* q1, const float* k0, const float* k1,
    unsigned int l, unsigned int D
) {
    DsaQuad c = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int d = l; d < D; d += 32u) {
        const float a0 = q0[d];
        const float a1 = q1[d];
        const float b0 = k0[d];
        const float b1 = k1[d];
        c.r0p0 += a0 * b0;
        c.r0p1 += a0 * b1;
        c.r1p0 += a1 * b0;
        c.r1p1 += a1 * b1;
    }
    return c;
}

// 2026-10-01: Node x_K[I] of the shfl_down tree (see the 2b header), left subtree first.
template <unsigned int I, unsigned int K>
struct DsaTree4 {
    static __device__ __forceinline__ DsaQuad run(
        const float* q0, const float* q1, const float* k0, const float* k1, unsigned int D
    ) {
        const DsaQuad lo = DsaTree4<I, K - 1u>::run(q0, q1, k0, k1, D);
        const DsaQuad hi = DsaTree4<I + (32u >> K), K - 1u>::run(q0, q1, k0, k1, D);
        DsaQuad s;
        s.r0p0 = lo.r0p0 + hi.r0p0;
        s.r0p1 = lo.r0p1 + hi.r0p1;
        s.r1p0 = lo.r1p0 + hi.r1p0;
        s.r1p1 = lo.r1p1 + hi.r1p1;
        return s;
    }
};

template <unsigned int I>
struct DsaTree4<I, 0u> {
    static __device__ __forceinline__ DsaQuad run(
        const float* q0, const float* q1, const float* k0, const float* k1, unsigned int D
    ) {
        return dsa_chain4(q0, q1, k0, k1, I, D);
    }
};

extern "C" __global__ void dsa_index_scores_tiled(
    const float* __restrict__ q,
    const float* __restrict__ pool_keys,
    const float* __restrict__ weights,
    const int* __restrict__ pool_indices,
    const unsigned char* __restrict__ pool_valid,
    const unsigned char* __restrict__ valid_keys,
    const int* __restrict__ q_pos,
    float* __restrict__ out,
    unsigned char* __restrict__ valid_cand,
    unsigned int Q,
    unsigned int P,
    unsigned int H,
    unsigned int D,
    unsigned int KP,
    unsigned int S,
    float scale,
    const int* __restrict__ geom
) {
    // 2026-10-01: No device-geometry path (see the 2b header); the host never passes one.
    if (geom) __trap();
    extern __shared__ float tsh[];
    float* k_s = tsh;
    float* q_s = tsh + DSA_TILE_POOLS * (D + 1u);
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int p_base = blockIdx.x * DSA_TILE_POOLS;
    const unsigned int r_base = blockIdx.y * DSA_TILE_ROWS;

    // 2026-10-01: This thread's outputs: tile rows (2 * warp, 2 * warp + 1) x tile pools
    // (lane, lane + 32). Candidacy and its stores are dsa_index_scores' code, per output.
    const unsigned int rows[2] = {r_base + 2u * warp, r_base + 2u * warp + 1u};
    const unsigned int pools[2] = {p_base + lane, p_base + lane + 32u};
    bool cand[2][2];
    bool mine_any = false;
    #pragma unroll
    for (int a = 0; a < 2; ++a) {
        #pragma unroll
        for (int b = 0; b < 2; ++b) {
            const unsigned int r = rows[a];
            const unsigned int p = pools[b];
            cand[a][b] = false;
            if (r >= Q || p >= P) continue;
            int end = pool_indices[p * KP + KP - 1];
            int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
            bool vis = (end_c <= q_pos[r]) && (valid_keys[end_c] != 0);
            bool c = (pool_valid[p] != 0) && vis;
            cand[a][b] = c;
            mine_any = mine_any || c;
            valid_cand[(size_t)r * P + p] = c ? 1 : 0;
            if (!c) out[(size_t)r * P + p] = -FLT_MAX;
        }
    }
    // 2026-10-01: A tile with no candidate (e.g. wholly past the causal diagonal) is done.
    if (!__syncthreads_or(mine_any ? 1 : 0)) return;

    // 2026-10-01: The tile's pool keys, row stride D + 1; pools past P are staged as 0 and
    // never stored.
    const unsigned int ks = D + 1u;
    for (unsigned int i = tid; i < DSA_TILE_POOLS * D; i += DSA_TILE_THREADS) {
        const unsigned int pl = i / D;
        const unsigned int d = i - pl * D;
        const unsigned int p = p_base + pl;
        k_s[pl * ks + d] = (p < P) ? pool_keys[(size_t)p * D + d] : 0.0f;
    }

    const float* q0 = q_s + (2u * warp) * D;
    const float* q1 = q_s + (2u * warp + 1u) * D;
    const float* k0 = k_s + lane * ks;
    const float* k1 = k_s + (lane + 32u) * ks;
    const bool r0_ok = rows[0] < Q;
    const bool r1_ok = rows[1] < Q;
    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    for (unsigned int h = 0; h < H; ++h) {
        // 2026-10-01: Every read of the previous head's q_s (and, at h = 0, every key store)
        // completes before q_s is rewritten.
        __syncthreads();
        for (unsigned int i = tid; i < DSA_TILE_ROWS * D; i += DSA_TILE_THREADS) {
            const unsigned int rl = i / D;
            const unsigned int d = i - rl * D;
            const unsigned int r = r_base + rl;
            q_s[rl * D + d] = (r < Q) ? q[((size_t)r * H + h) * D + d] : 0.0f;
        }
        __syncthreads();
        if (mine_any) {
            const DsaQuad x = DsaTree4<0u, 5u>::run(q0, q1, k0, k1, D);
            const float w0 = r0_ok ? weights[(size_t)rows[0] * H + h] : 0.0f;
            const float w1 = r1_ok ? weights[(size_t)rows[1] * H + h] : 0.0f;
            acc00 += w0 * fmaxf(scale * x.r0p0, 0.0f);
            acc01 += w0 * fmaxf(scale * x.r0p1, 0.0f);
            acc10 += w1 * fmaxf(scale * x.r1p0, 0.0f);
            acc11 += w1 * fmaxf(scale * x.r1p1, 0.0f);
        }
    }
    if (cand[0][0]) out[(size_t)rows[0] * P + pools[0]] = acc00;
    if (cand[0][1]) out[(size_t)rows[0] * P + pools[1]] = acc01;
    if (cand[1][0]) out[(size_t)rows[1] * P + pools[0]] = acc10;
    if (cand[1][1]) out[(size_t)rows[1] * P + pools[1]] = acc11;
}

// 2026-09-25: 3. Deterministic top-k over pools, one block per query. A tiled bitonic
// select: the pool axis is walked in tiles of NP2 and a running best-NP2 list is kept in
// shared memory (two tiles of [f32, i32], 16 * NP2 bytes, whatever the context). The order
// is score descending, then pool index ascending. That comparator is a total order over
// unique indices, so the top-select_k prefix is unique and the tiled result equals a
// whole-axis sort. select_k must be <= NP2; DsaSelectGeometry::plan refuses more. Slots
// never filled (index INT_MAX) are written as -1.














extern "C" __global__ void dsa_topk_pools(
    const float* __restrict__ scores,
    int* __restrict__ selected,
    unsigned int Q,
    unsigned int P,
    unsigned int NP2,
    unsigned int select_k,
    const int* __restrict__ geom
) {
    if (geom) {
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        NP2 = (unsigned int)geom[DSA_GEOM_NP2];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    // 2026-09-25: Under a ceiling launch the dynamic shared memory is sized for the ceiling;
    // the walk still runs over this step's NP2 and P.
    extern __shared__ char raw_sh[];
    const unsigned int T = NP2;
    float* sv = (float*)raw_sh;
    int*   si = (int*)(raw_sh + (size_t)(2 * T) * sizeof(float));
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    // 2026-09-25: Running best list, descending; starts as padding that sorts last on both keys.
    for (unsigned int i = tid; i < T; i += blockDim.x) {
        sv[i] = -FLT_MAX;
        si[i] = INT_MAX;
    }
    __syncthreads();

#define DSA_TOPK_GT(a, b) ((sv[(a)] > sv[(b)]) || (sv[(a)] == sv[(b)] && si[(a)] < si[(b)]))
#define DSA_TOPK_SWAP(a, b)                                                                  \
    do {                                                                                     \
        float tv_ = sv[(a)]; sv[(a)] = sv[(b)]; sv[(b)] = tv_;                               \
        int   ti_ = si[(a)]; si[(a)] = si[(b)]; si[(b)] = ti_;                               \
    } while (0)

    for (unsigned int base = 0; base < P; base += T) {
        // 2026-09-25: Candidate tile into [T, 2T); a short tile pads with the same sentinel.
        for (unsigned int i = tid; i < T; i += blockDim.x) {
            unsigned int idx = base + i;
            sv[T + i] = (idx < P) ? scores[(size_t)r * P + idx] : -FLT_MAX;
            si[T + i] = (idx < P) ? (int)idx : INT_MAX;
        }
        __syncthreads();

        // 2026-09-25: Bitonic sort of the candidate tile, descending.
        for (unsigned int k = 2; k <= T; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                for (unsigned int i = tid; i < T; i += blockDim.x) {
                    unsigned int l = i ^ j;
                    if (l > i) {
                        bool gt = DSA_TOPK_GT(T + i, T + l);
                        bool want_desc = ((i & k) == 0);
                        if (want_desc != gt) DSA_TOPK_SWAP(T + i, T + l);
                    }
                }
                __syncthreads();
            }
        }

        // 2026-09-25: Half-cleaner across the two descending runs: pairing best[i] with
        // cand[T-1-i] leaves the top T of both in [0, T), bitonic but not yet sorted.
        for (unsigned int i = tid; i < T; i += blockDim.x) {
            unsigned int a = i, b = T + (T - 1 - i);
            if (!DSA_TOPK_GT(a, b)) DSA_TOPK_SWAP(a, b);
        }
        __syncthreads();

        // 2026-09-25: Bitonic merge restores descending order over [0, T).
        for (unsigned int j = T >> 1; j > 0; j >>= 1) {
            for (unsigned int i = tid; i < T; i += blockDim.x) {
                unsigned int l = i ^ j;
                if (l > i && !DSA_TOPK_GT(i, l)) DSA_TOPK_SWAP(i, l);
            }
            __syncthreads();
        }
    }

#undef DSA_TOPK_GT
#undef DSA_TOPK_SWAP

    for (unsigned int i = tid; i < select_k; i += blockDim.x)
        selected[(size_t)r * select_k + i] = (si[i] == INT_MAX) ? DSA_INVALID : si[i];
}

// 2026-09-25: 4. Expand the selected pools into raw token ids, one block per query, into a
// row of `width` slots (index_topk, plus index_kpool - 1 when always_tail). The row is
// filled with -1 and synchronised before any real index is written, so short rows, invalid
// pools and a missing tail all leave the sentinel.
extern "C" __global__ void dsa_expand_selection(
    const int* __restrict__ selected,
    const int* __restrict__ pool_indices,
    const unsigned char* __restrict__ valid_cand,
    const unsigned char* __restrict__ valid_keys,
    const int* __restrict__ q_pos,
    const unsigned char* __restrict__ q_mask,
    int* __restrict__ out,
    unsigned int Q,
    unsigned int P,
    unsigned int KP,
    unsigned int S,
    unsigned int select_k,
    unsigned int width,
    int first_key,
    int always_tail,
    const int* __restrict__ geom
) {
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    int* row = out + (size_t)r * width;

    for (unsigned int i = tid; i < width; i += blockDim.x) row[i] = DSA_INVALID;
    __syncthreads();
    if (q_mask[r] == 0) return;   // 2026-09-25: a masked query selects nothing; the row stays all -1

    // 2026-09-25: Per-row select_k. The scalar select_k is the pass's, planned from the pass's
    // cache length, and a multi-row pass plans it from the group's final length. Clamping it
    // to this row's own pool count (q_pos[r] + 1) / KP gives the row the select_k, and so the
    // tail base below, that a single-row pass at that row's length plans. `selected` keeps
    // the pass stride; only the count is per row. The tail slot matters because
    // glm5next_dsa_mla_decode_fp8 splits the row into NUM_WARPS = 8 slices merged across
    // warps. Measured 2026-09-06 on that kernel without the clamp: 14 of 18 configurations
    // where a tail token crossed a slice boundary differed, by up to 2 BF16 ulp.












    const unsigned int row_pools = (unsigned int)(q_pos[r] + 1) / KP;
    const unsigned int row_select_k = (row_pools < select_k) ? row_pools : select_k;

    for (unsigned int j = tid; j < row_select_k; j += blockDim.x) {
        int p = selected[(size_t)r * select_k + j];
        bool ok = (p >= 0) && (valid_cand[(size_t)r * P + p] != 0);
        for (unsigned int s = 0; s < KP; ++s) {
            unsigned int w = j * KP + s;
            if (w < width) row[w] = ok ? pool_indices[(size_t)p * KP + s] : DSA_INVALID;
        }
    }

    if (always_tail) {
        // 2026-09-25: The in-progress pool as raw indices. vis_count, the number of valid keys
        // at or before q_pos[r], is an integer count split across all threads, so it is exact.
        // It is not taken as q_pos[r] + 1 - first_key, because valid_keys may mark keys at or
        // below q_pos[r] invalid. always_tail is kernel-uniform and r is blockIdx.x, so every
        // thread reaches the barrier below; the q_mask return above is whole-block as well.













        __shared__ int vis_warp[32];
        const int qp = q_pos[r];
        int local = 0;
        for (unsigned int t = tid; t < S; t += blockDim.x)
            if ((int)t <= qp && valid_keys[t] != 0) ++local;
        for (int off = 16; off > 0; off >>= 1)
            local += __shfl_down_sync(0xffffffffu, local, off);
        const unsigned int lane = tid & 31u;
        const unsigned int warp = tid >> 5;
        if (lane == 0) vis_warp[warp] = local;
        __syncthreads();
        if (tid != 0) return;
        const unsigned int nwarps = (blockDim.x + 31u) / 32u;
        int vis_count = 0;
        for (unsigned int w = 0; w < nwarps; ++w) vis_count += vis_warp[w];

        int tail_count = vis_count % (int)KP;
        int tail_start = first_key + vis_count - tail_count;
        unsigned int base = row_select_k * KP;
        for (unsigned int t = 0; t + 1 < KP; ++t) {
            long long idx = (long long)tail_start + t;
            bool ok = ((int)t < tail_count) && idx >= 0 && idx < (long long)S
                      && ((int)idx <= q_pos[r]) && valid_keys[(unsigned)idx] != 0;
            if (base + t < width) row[base + t] = ok ? (int)idx : DSA_INVALID;
        }
    }
}

// 2026-09-25: 5. Index row -> visibility mask (oracle). Duplicates collapse, so a repeated
// token is attended once; out-of-range and -1 entries are dropped.

extern "C" __global__ void dsa_topk_to_mask(
    const int* __restrict__ topk,
    unsigned char* __restrict__ mask,
    unsigned int Q,
    unsigned int width,
    unsigned int S
) {
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    unsigned char* row = mask + (size_t)r * S;
    for (unsigned int i = tid; i < S; i += blockDim.x) row[i] = 0;
    __syncthreads();
    for (unsigned int j = tid; j < width; j += blockDim.x) {
        int i = topk[(size_t)r * width + j];
        if (i >= 0 && i < (int)S) row[i] = 1;
    }
}

// 2026-09-25: 6. NoPE MLA over the selected tokens (oracle), one block per (query, head).
// Scores are parallel over keys and the value sum over dims, with the score row staged in
// shared memory in between, so there is no barrier per key. The score row is S floats of
// dynamic shared memory: at 49,152 B that caps S at 12,288 keys (MASKED_ATTN_MAX_KEYS in
// glm5next_dsa/mod.rs).









extern "C" __global__ void dsa_mla_masked_attn(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const unsigned char* __restrict__ mask,
    float* __restrict__ out,
    unsigned int Q,
    unsigned int S,
    unsigned int H,
    unsigned int qd,
    unsigned int vd,
    float scale,
    // 2026-09-25: Nonzero rounds each scaled score to BF16 before the softmax.



    unsigned int round_scores_bf16
) {
    extern __shared__ float sc[];
    __shared__ float red[32];
    __shared__ float s_m, s_l;
    const unsigned int r = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* qrow = q + ((size_t)r * H + h) * qd;
    const unsigned char* mrow = mask + (size_t)r * S;


    float local_max = -CUDART_INF_F;
    for (unsigned int t = tid; t < S; t += blockDim.x) {
        if (mrow[t] == 0) { sc[t] = -CUDART_INF_F; continue; }
        float dot = 0.0f;
        const __nv_bfloat16* krow = k + ((size_t)t * H + h) * qd;
        for (unsigned int d = 0; d < qd; ++d)
            dot += __bfloat162float(qrow[d]) * __bfloat162float(krow[d]);
        float sv = dot * scale;
        if (round_scores_bf16) sv = __bfloat162float(__float2bfloat16(sv));
        sc[t] = sv;
        local_max = fmaxf(local_max, sv);
    }
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    if ((tid & 31u) == 0u) red[tid >> 5] = local_max;
    __syncthreads();
    if (tid < 32u) {
        float x = (tid < ((blockDim.x + 31u) / 32u)) ? red[tid] : -CUDART_INF_F;
        for (int off = 16; off > 0; off >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, off));
        if (tid == 0u) s_m = x;
    }
    __syncthreads();


    const float m = s_m;
    float local_sum = 0.0f;
    for (unsigned int t = tid; t < S; t += blockDim.x) {
        float e = (sc[t] == -CUDART_INF_F) ? 0.0f : __expf(sc[t] - m);
        sc[t] = e;
        local_sum += e;
    }
    local_sum = dsa_block_sum(local_sum, red, tid, blockDim.x);
    if (tid == 0) s_l = local_sum;
    __syncthreads();


    const float inv = (s_l > 0.0f) ? (1.0f / s_l) : 0.0f;
    for (unsigned int d = tid; d < vd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int t = 0; t < S; ++t) {
            float p = sc[t];
            if (p != 0.0f) acc += p * __bfloat162float(v[((size_t)t * H + h) * vd + d]);
        }
        out[((size_t)r * H + h) * vd + d] = acc * inv;
    }
}

// 2026-09-25: 1b. Pool-axis compaction: gathers the pools listed in `keep` (original pool
// ids) into dense arrays; the caller computes `keep`. select_tokens does not launch it:
// over a contiguous cache the kept pools are the prefix 0 .. S / KP.






extern "C" __global__ void dsa_compact_pools(
    const float* __restrict__ keys_in,
    const int* __restrict__ idx_in,
    const unsigned char* __restrict__ valid_in,
    const int* __restrict__ keep,
    float* __restrict__ keys_out,
    int* __restrict__ idx_out,
    unsigned char* __restrict__ valid_out,
    unsigned int P_kept,
    unsigned int D,
    unsigned int KP
) {
    const unsigned int p = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const int src = keep[p];
    for (unsigned int d = tid; d < D; d += blockDim.x)
        keys_out[(size_t)p * D + d] = keys_in[(size_t)src * D + d];
    for (unsigned int s = tid; s < KP; s += blockDim.x)
        idx_out[(size_t)p * KP + s] = idx_in[(size_t)src * KP + s];
    if (tid == 0) valid_out[p] = valid_in[src];
}
