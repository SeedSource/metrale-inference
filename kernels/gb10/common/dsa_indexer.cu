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
// - 2026-10-06: dsa_topk_radix_{init,hist,find,gather,sort} (3b, METRALE_GLM_DSA_TOPK_RADIX=1)
//   write the same `selected` bytes as dsa_topk_pools, for pool counts above one top-k tile.
// - 2026-10-06: METRALE_GLM_DSA_POOL_CACHE=1 (pool cache, host side in
//   crates/model-arch/src/glm5next_dsa/pool_cache.rs) replaces the first stage with
//   dsa_kpool_compress_incr: pools [pk_start, live) only, into persistent per-state arrays,
//   reading k/gate from a ring at row % ring_rows, each pool by the per-pool body of
//   dsa_kpool_compress unchanged. The later stages read the persistent arrays. The kernels
//   above it (dsa_write_geom, dsa_indexer_store, dsa_kpool_compress) are untouched; the
//   lever-off path launches only them.
// - 2026-10-08: dsa_index_scores_decode (2e, METRALE_GLM_DSA_SCORES_DECODE=1) writes the same
//   out / valid_cand bytes as dsa_index_scores on the ceiling (decode) launch.







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
// 2026-10-05: dsa_kpool_compress and dsa_index_scores walk the live pools with a grid
// stride (block b takes pools b, b + gridDim.x, ... below the live count), so a ceiling
// launch need not be one block per ceiling pool: the host launches a few waves of blocks
// and a graph replay pays for those, not for the context ceiling. With
// METRALE_GLM_DSA_GRID_STRIDE=0 the host launches the ceiling grid again. It does so too when
// the module lacks the marker kernel dsa_indexer_grid_stride_v1 (defined below
// dsa_index_scores), so a copy of this file without the loop must not define it.

#define DSA_GEOM_S        0
#define DSA_GEOM_NPOOLS_F 1
#define DSA_GEOM_NPOOLS   2
#define DSA_GEOM_SELECT_K 3
#define DSA_GEOM_NP2      4
// 2026-10-06: Pool cache only: first pool the incremental compress recomputes. Written by
// dsa_write_geom_pk into a 6-slot geom; dsa_write_geom (5 slots) never touches it.
#define DSA_GEOM_PK_START 5

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
    const unsigned int tid = threadIdx.x;
    // 2026-10-05: Grid-stride walk over the live pools: block b takes pools b, b + gridDim.x,
    // ... below `live`, so any grid of one block or more covers every live pool exactly once,
    // each by one block running the per-pool code below unchanged (same thread mapping, same
    // reduction order), and a grid of `live` blocks or more is one pool per block, this
    // kernel's earlier form. `live` counts the pools including the trailing partial one: geom's
    // under a ceiling launch, whose grid the host sets (METRALE_GLM_DSA_GRID_STRIDE), else the
    // grid itself (an exact launch has one block per pool).
    unsigned int live = gridDim.x;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        live = (unsigned int)geom[DSA_GEOM_NPOOLS_F];
    }
    for (unsigned int p = blockIdx.x; p < live; p += gridDim.x) {
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
}

// 2026-10-06: Pool cache (METRALE_GLM_DSA_POOL_CACHE=1). The device copy of a state's pool
// watermark, pk_len_dev[0], counts the pools whose keys in the persistent arrays are final.
// It may only be lowered by a write (a rewritten row's pool must be recomputed) and is set by
// the compress. Four kernels:
// - dsa_write_geom_pk: dsa_write_geom's five slots, plus DSA_GEOM_PK_START =
//   min(pk_len_dev, (S - 1) / KP), the first pool the replayed row's compress recomputes.
// - dsa_indexer_store_ring: dsa_indexer_store at ring slot pos % ring_rows; valid stays
//   full length (valid[pos]); thread 0 lowers pk_len_dev to pos / KP.
// - dsa_pk_len_clamp: the host path's lowering after a rewind (pk_len_dev = min(., pools)).
// - dsa_kpool_compress_incr: below.

// 2026-10-06: One thread. dsa_write_geom's slots from seq_len[0] (same arithmetic), and the
// pool-cache start slot. pk_len_dev may be ahead of S after a rejected draft; the store of
// row S - 1 (dsa_indexer_store_ring) already lowered it, and the min here bounds it again.
extern "C" __global__ void dsa_write_geom_pk(
    const int* __restrict__ seq_len,
    int* __restrict__ geom,
    unsigned int KP,
    unsigned int topk,
    unsigned int tile,
    const int* __restrict__ pk_len_dev
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
    const int last = S > 0 ? (S - 1) / (int)KP : 0;
    const int pk = pk_len_dev[0];
    geom[DSA_GEOM_PK_START] = pk < last ? pk : last;
}

// 2026-10-06: dsa_indexer_store with the k/gate rows in a ring of ring_rows rows: the staged
// row lands at slot pos[0] % ring_rows, valid[pos[0]] is set at the absolute position, and the
// pool watermark drops to the written row's pool.
extern "C" __global__ void dsa_indexer_store_ring(
    const __nv_bfloat16* __restrict__ stage_k,
    const __nv_bfloat16* __restrict__ stage_gate,
    const int* __restrict__ pos,
    __nv_bfloat16* __restrict__ k_ring,
    __nv_bfloat16* __restrict__ gate_ring,
    unsigned char* __restrict__ valid,
    unsigned int D,
    unsigned int ring_rows,
    unsigned int KP,
    int* __restrict__ pk_len_dev
) {
    const int p = pos[0];
    const size_t base = (size_t)((unsigned int)p % ring_rows) * D;
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        k_ring[base + d] = stage_k[d];
        gate_ring[base + d] = stage_gate[d];
    }
    if (threadIdx.x == 0) {
        valid[p] = 1;
        const int pool = p / (int)KP;
        if (pk_len_dev[0] > pool) pk_len_dev[0] = pool;
    }
}

// 2026-10-06: One thread. Lowers the pool watermark to `pools` (host-path write after a rewind).
extern "C" __global__ void dsa_pk_len_clamp(int* __restrict__ pk_len_dev, int pools) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (pk_len_dev[0] > pools) pk_len_dev[0] = pools;
}

// 2026-10-06: Incremental kpool compression into persistent per-state arrays. Pools
// [pk_start, live) only, walked with the grid stride of dsa_kpool_compress (block b takes
// pk_start + b, pk_start + b + gridDim.x, ...). The per-pool body is dsa_kpool_compress's,
// line for line (same slot loop order, same __expf, same FP32 accumulation order); the one
// difference is the k/gate index, ring slot raw % ring_rows instead of raw, so each pool key is
// bit-identical to the full recompute's as long as the ring still holds rows
// [pk_start * KP, S) (the host checks it before every launch). valid is full length.
// Exact launch (geom NULL): S, pk_start from the arguments, live = ceil(S / KP). Ceiling launch:
// S, live and pk_start from geom (dsa_write_geom_pk). Afterwards block 0 sets pk_len_dev to
// S / KP (complete pools; the trailing partial pool is rewritten and stays invalid until it
// completes). No block reads pk_len_dev here (the start is the argument or the geom slot), so
// that store races with nothing.
extern "C" __global__ void dsa_kpool_compress_incr(
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
    unsigned int ring_rows,
    unsigned int pk_start,
    int* __restrict__ pk_len_dev,
    const int* __restrict__ geom
) {
    const unsigned int tid = threadIdx.x;
    unsigned int live = (S + KP - 1) / KP;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        live = (unsigned int)geom[DSA_GEOM_NPOOLS_F];
        pk_start = (unsigned int)geom[DSA_GEOM_PK_START];
    }
    for (unsigned int p = pk_start + blockIdx.x; p < live; p += gridDim.x) {
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
                // 2026-10-06: Ring slot of row raw (raw >= 0 whenever ok).
                size_t slot = ok ? (size_t)((unsigned long long)raw % ring_rows) : 0;
                lg[s] = ok ? (__bfloat162float(gate[slot * D + d]) + ape[s * D + d])
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
                if (ok) {
                    size_t slot = (size_t)((unsigned long long)raw % ring_rows);
                    acc += lg[s] * inv * __bfloat162float(k[slot * D + d]);
                }
            }
            pool_keys[p * D + d] = acc;
        }
    }
    if (blockIdx.x == 0 && tid == 0) pk_len_dev[0] = (int)(S / KP);
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
    const unsigned int r = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    extern __shared__ float sh[];
    // 2026-10-05: Grid-stride walk over the live pools, as in dsa_kpool_compress: block (b, r)
    // takes pools b, b + gridDim.x, ... below `live`, each computed exactly as the
    // one-pool-per-block form computed it, and a grid of `live` blocks or more is that form.
    // `live` is geom's complete-pool count under a ceiling launch, else the grid width (an
    // exact launch has one block per pool and no early exit).
    unsigned int live = gridDim.x;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        // 2026-09-25: `P` is also the row stride of out / valid_cand, so it may vary with geom
        // only because the device-geometry path is decode-only (Q == 1, every r * P is 0).
        // The launcher refuses a ceiling launch with Q > 1.
        live = P;
    }
    for (unsigned int p = blockIdx.x; p < live; p += gridDim.x) {
        // 2026-09-25: A pool is a candidate only when it is complete and its last token, clamped
        // to [0, S-1], is at or before this query's position and valid. Others score -FLT_MAX.
        int end = pool_indices[p * KP + KP - 1];
        int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
        bool vis = (end_c <= q_pos[r]) && (valid_keys[end_c] != 0);
        bool cand = (pool_valid[p] != 0) && vis;
        if (tid == 0) valid_cand[(size_t)r * P + p] = cand ? 1 : 0;
        if (!cand) {
            if (tid == 0) out[(size_t)r * P + p] = -FLT_MAX;
            continue;
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
        // 2026-10-05: This block's next pool rewrites sh[0..H): every thread waits here until
        // thread 0 has summed it. All threads reach this barrier or none do (cand depends on
        // the pool and the row only), so it is safe inside the loop.
        __syncthreads();
    }
}

// 2026-10-05: No-op marker: it exists iff both kernels above carry the grid-stride loop.
extern "C" __global__ void dsa_indexer_grid_stride_v1() {}

// 2026-10-01: 2b. dsa_index_scores_tiled: the same scores and candidacy as dsa_index_scores,
// byte-identical by construction, with one block per DSA_TILE_ROWS x DSA_TILE_POOLS tile
// instead of one per (pool, query), so a row's q is read once per tile, not once per pool.
// Opt-in (METRALE_GLM_DSA_SCORES_TILED=1); resolved with try_kernel; gated on a GPU by
// crates/model-arch/examples/dsa_indexer_tiled_bitparity_microtest.rs.
// 2026-10-01 (v2): 32 x 64 tiles, 128 threads, 4 rows x 4 pools per thread, D/32 unrolled
// at compile time, q of the next head prefetched into registers during this head's compute.
// v1 (16 x 64 tiles, 2 x 2 per thread) was byte-identical but 0.27x-1.31x of
// dsa_index_scores on GB10 (Q = 256, P = 64..2048).
//
// Why the bits match (the common build passes --fmad=false, so a * b + c is a rounded
// multiply then a rounded add, as in dsa_index_scores):
// - In dsa_index_scores, lane l of a head's warp sums qh[d] * pk[d] over d = l, l + 32, ...
//   (d < D), ascending, from 0.0f. dsa_leaf16 runs exactly that chain (x0[l]) for each of the
//   thread's 16 (row, pool) outputs; the 16 chains are independent and never mix.
// - __shfl_down_sync with offsets 16, 8, 4, 2, 1 leaves at lane 0 the tree
//   x1[i] = x0[i] + x0[i + 16], x2[i] = x1[i] + x1[i + 8], x3[i] = x2[i] + x2[i + 4],
//   x4[i] = x3[i] + x3[i + 2], x5[0] = x4[0] + x4[1], each add with the lower lane on the
//   left. (A lane above 31 - off reads its own value, but lane 0's result never depends on
//   such a lane.) dsa_x1 / dsa_x2 / dsa_x3 are those nodes for one i, left operand first;
//   the head loop forms x3[0], x3[2], x3[1], x3[3] in that order and combines them as
//   x4[0] = x3[0] + x3[2], x4[1] = x3[1] + x3[3], x5[0] = x4[0] + x4[1].
// - The head term is weights[r * H + h] * fmaxf(scale * x5[0], 0.0f), and the score is
//   0.0f + term(0) + term(1) + ... in ascending h, as thread 0 sums sh[] there. Each thread
//   owns its 16 outputs for every head, so no sum is split across threads.
// - Candidacy, valid_cand and the -FLT_MAX store are dsa_index_scores' code, per output.
// Staging q and the pool keys in (swizzled) shared memory changes where operands are read
// from, not their values or the order they are combined in.
//
// Layout: DSA_TILE_THREADS threads (4 warps). Warp w owns tile rows [8w, 8w + 8); lane
// (rs, ps) = (lane >> 4, lane & 15) owns rows 8w + 4 rs + a and pools ps + 16 b (a, b < 4).
// Shared memory (dynamic, (DSA_TILE_POOLS + DSA_TILE_ROWS) * D * 4 bytes, exactly 49,152 at
// D = 128): the tile's pool keys [64][D], staged once, and one head's q [32][D], restaged
// per head. Element (row, d) of either sits at row * D + (d ^ (row & 31)) (D is a multiple
// of 32, so the XOR stays inside d's 32-aligned group): the 16 pools a warp reads at one d
// fall in 16 distinct banks, and its two q rows (4 apart) in two, so every load is one
// wavefront. Per lane chain and warp: 16 + 16 loads for 64 * D / 32 MACs.
// Registers (estimate, not a ptxas report): 16 accumulators, a 3-deep tree (16 floats per
// level) plus the 16 current chains and the x4[0] / x3 holders, 8 operands, and 8 * D / 32
// prefetched q floats: about 170 at D = 128, under the 255 cap; two blocks per SM fit both
// the register file and shared memory.
//
// Not supported: the device-geometry (ceiling) path dsa_index_scores serves from `geom` at
// Q == 1, and D that is not a multiple of 32 up to 128. The host never selects this kernel
// there; a launch that does traps instead of computing anything.

#define DSA_TILE_ROWS 32u
#define DSA_TILE_POOLS 64u
#define DSA_TILE_THREADS 128u

// 2026-10-01: Shared-memory slot of element (row, d) of a [rows][D] tile (see 2b).
__device__ __forceinline__ unsigned int dsa_swz(
    unsigned int row, unsigned int d, unsigned int D
) {
    return row * D + (d ^ (row & 31u));
}

// 2026-10-01: x0[l] for the thread's 4 x 4 outputs: d = l + 32 j for j < NJ, ascending,
// from 0.0f, each product rounded before its add. qrow / kpool are tile-local.
template <int NJ>
__device__ __forceinline__ void dsa_leaf16(
    float (&c)[4][4], const float* q_s, const float* k_s,
    const unsigned int (&qrow)[4], const unsigned int (&kpool)[4], unsigned int l
) {
    const unsigned int D = 32u * NJ;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) c[a][b] = 0.0f;
    }
    #pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const unsigned int d = l + 32u * j;
        float x[4], y[4];
        #pragma unroll
        for (int a = 0; a < 4; ++a) x[a] = q_s[dsa_swz(qrow[a], d, D)];
        #pragma unroll
        for (int b = 0; b < 4; ++b) y[b] = k_s[dsa_swz(kpool[b], d, D)];
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) c[a][b] += x[a] * y[b];
        }
    }
}

// 2026-10-01: s = lo + hi per output, lo on the left.
__device__ __forceinline__ void dsa_add16(
    float (&s)[4][4], const float (&lo)[4][4], const float (&hi)[4][4]
) {
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) s[a][b] = lo[a][b] + hi[a][b];
    }
}

// 2026-10-01: x1[i] = x0[i] + x0[i + 16].
template <int NJ>
__device__ __forceinline__ void dsa_x1(
    float (&s)[4][4], const float* q_s, const float* k_s,
    const unsigned int (&qrow)[4], const unsigned int (&kpool)[4], unsigned int i
) {
    float lo[4][4], hi[4][4];
    dsa_leaf16<NJ>(lo, q_s, k_s, qrow, kpool, i);
    dsa_leaf16<NJ>(hi, q_s, k_s, qrow, kpool, i + 16u);
    dsa_add16(s, lo, hi);
}

// 2026-10-01: x2[i] = x1[i] + x1[i + 8].
template <int NJ>
__device__ __forceinline__ void dsa_x2(
    float (&s)[4][4], const float* q_s, const float* k_s,
    const unsigned int (&qrow)[4], const unsigned int (&kpool)[4], unsigned int i
) {
    float lo[4][4], hi[4][4];
    dsa_x1<NJ>(lo, q_s, k_s, qrow, kpool, i);
    dsa_x1<NJ>(hi, q_s, k_s, qrow, kpool, i + 8u);
    dsa_add16(s, lo, hi);
}

// 2026-10-01: x3[i] = x2[i] + x2[i + 4].
template <int NJ>
__device__ __forceinline__ void dsa_x3(
    float (&s)[4][4], const float* q_s, const float* k_s,
    const unsigned int (&qrow)[4], const unsigned int (&kpool)[4], unsigned int i
) {
    float lo[4][4], hi[4][4];
    dsa_x2<NJ>(lo, q_s, k_s, qrow, kpool, i);
    dsa_x2<NJ>(hi, q_s, k_s, qrow, kpool, i + 4u);
    dsa_add16(s, lo, hi);
}

// 2026-10-01: Head h's q rows of the tile, this thread's share (8 * NJ floats), into
// registers; rows past Q read as 0 (never stored).
template <int NJ>
__device__ __forceinline__ void dsa_q_fetch(
    float (&pf)[8 * NJ], const float* __restrict__ q, unsigned int r_base, unsigned int Q,
    unsigned int H, unsigned int h, unsigned int tid
) {
    const unsigned int D = 32u * NJ;
    #pragma unroll
    for (int m = 0; m < 8 * NJ; ++m) {
        const unsigned int i = tid + DSA_TILE_THREADS * m;
        const unsigned int rl = i / D;
        const unsigned int d = i - rl * D;
        const unsigned int r = r_base + rl;
        pf[m] = (r < Q) ? q[((size_t)r * H + h) * D + d] : 0.0f;
    }
}

template <int NJ>
__device__ __forceinline__ void dsa_scores_tile(
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
    unsigned int KP,
    unsigned int S,
    float scale,
    float* tsh
) {
    const unsigned int D = 32u * NJ;
    float* k_s = tsh;
    float* q_s = tsh + DSA_TILE_POOLS * D;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int p_base = blockIdx.x * DSA_TILE_POOLS;
    const unsigned int r_base = blockIdx.y * DSA_TILE_ROWS;
    unsigned int qrow[4], kpool[4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) qrow[a] = 8u * warp + 4u * (lane >> 4) + a;
    #pragma unroll
    for (int b = 0; b < 4; ++b) kpool[b] = (lane & 15u) + 16u * b;

    // 2026-10-01: Candidacy and its stores are dsa_index_scores' code, per output; bit
    // 4 a + b of cmask is output (qrow[a], kpool[b]).
    unsigned int cmask = 0u;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const unsigned int r = r_base + qrow[a];
            const unsigned int p = p_base + kpool[b];
            if (r >= Q || p >= P) continue;
            int end = pool_indices[p * KP + KP - 1];
            int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
            bool vis = (end_c <= q_pos[r]) && (valid_keys[end_c] != 0);
            bool cand = (pool_valid[p] != 0) && vis;
            if (cand) cmask |= 1u << (4 * a + b);
            valid_cand[(size_t)r * P + p] = cand ? 1 : 0;
            if (!cand) out[(size_t)r * P + p] = -FLT_MAX;
        }
    }
    // 2026-10-01: A tile with no candidate (e.g. wholly past the causal diagonal) is done.
    if (!__syncthreads_or(cmask != 0u ? 1 : 0)) return;

    // 2026-10-01: The tile's pool keys; pools past P are staged as 0 and never stored.
    for (unsigned int i = tid; i < DSA_TILE_POOLS * D; i += DSA_TILE_THREADS) {
        const unsigned int pl = i / D;
        const unsigned int d = i - pl * D;
        const unsigned int p = p_base + pl;
        k_s[dsa_swz(pl, d, D)] = (p < P) ? pool_keys[(size_t)p * D + d] : 0.0f;
    }

    float pf[8 * NJ];
    dsa_q_fetch<NJ>(pf, q, r_base, Q, H, 0u, tid);
    float acc[4][4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0f;
    }
    for (unsigned int h = 0; h < H; ++h) {
        // 2026-10-01: Every read of the previous head's q_s (and, at h = 0, every key store)
        // completes before q_s is rewritten.
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < 8 * NJ; ++m) {
            const unsigned int i = tid + DSA_TILE_THREADS * m;
            const unsigned int rl = i / D;
            q_s[dsa_swz(rl, i - rl * D, D)] = pf[m];
        }
        __syncthreads();
        // 2026-10-01: The next head's q is in flight while this head computes.
        if (h + 1u < H) dsa_q_fetch<NJ>(pf, q, r_base, Q, H, h + 1u, tid);
        if (cmask == 0u) continue;
        // 2026-10-01: x3[0], x3[2], x3[1], x3[3] (t = 0..3), then x4[0], x4[1], x5[0].
        float x3[4][4], hold[4][4], x4lo[4][4], x5[4][4];
        #pragma unroll 1
        for (unsigned int t = 0; t < 4u; ++t) {
            const unsigned int i = (t >> 1) | ((t & 1u) << 1);
            dsa_x3<NJ>(x3, q_s, k_s, qrow, kpool, i);
            if (t == 0u || t == 2u) {
                #pragma unroll
                for (int a = 0; a < 4; ++a) {
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) hold[a][b] = x3[a][b];
                }
            } else if (t == 1u) {
                dsa_add16(x4lo, hold, x3);
            } else {
                float x4hi[4][4];
                dsa_add16(x4hi, hold, x3);
                dsa_add16(x5, x4lo, x4hi);
            }
        }
        float w[4];
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            const unsigned int r = r_base + qrow[a];
            w[a] = (r < Q) ? weights[(size_t)r * H + h] : 0.0f;
        }
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) acc[a][b] += w[a] * fmaxf(scale * x5[a][b], 0.0f);
        }
    }
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            if ((cmask >> (4 * a + b)) & 1u) {
                const unsigned int r = r_base + qrow[a];
                const unsigned int p = p_base + kpool[b];
                out[(size_t)r * P + p] = acc[a][b];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(DSA_TILE_THREADS) dsa_index_scores_tiled(
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
    // 2026-10-01: No device-geometry path and no D outside {32, 64, 96, 128} (see 2b); the
    // host never launches either.
    if (geom || (D & 31u) != 0u) __trap();
    extern __shared__ float tsh[];
    switch (D >> 5) {
    case 1u:
        dsa_scores_tile<1>(q, pool_keys, weights, pool_indices, pool_valid, valid_keys, q_pos,
                           out, valid_cand, Q, P, H, KP, S, scale, tsh);
        break;
    case 2u:
        dsa_scores_tile<2>(q, pool_keys, weights, pool_indices, pool_valid, valid_keys, q_pos,
                           out, valid_cand, Q, P, H, KP, S, scale, tsh);
        break;
    case 3u:
        dsa_scores_tile<3>(q, pool_keys, weights, pool_indices, pool_valid, valid_keys, q_pos,
                           out, valid_cand, Q, P, H, KP, S, scale, tsh);
        break;
    case 4u:
        dsa_scores_tile<4>(q, pool_keys, weights, pool_indices, pool_valid, valid_keys, q_pos,
                           out, valid_cand, Q, P, H, KP, S, scale, tsh);
        break;
    default:
        __trap();
    }
}

// 2026-10-01: 2c. dsa_index_scores_tc: the per-(row, pool) scores of dsa_index_scores on
// tensor cores (mma.sync m16n8k16, BF16 operands, FP32 accumulate), for long prompts where
// the O(rows x pools) indexer dominates prefill. Opt-in (METRALE_GLM_DSA_SCORES_TC, default
// off); resolved with try_kernel; gated on a GPU by
// crates/model-arch/examples/dsa_indexer_tc_microtest.rs (accuracy against dsa_index_scores,
// selection recall, planted-needle recall, timing at 32K / 64K).
//
// NOT byte-identical to dsa_index_scores: the dots are formed from BF16 pieces and summed in
// the tensor core's order. The precision is chosen by `mode` (the q split is the in-repo
// qsa_score_rows_tc pattern, kernels/gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu):
//   mode 1 ("bf16"):   q.k ~ bf16(q).bf16(k)                                  1 MMA
//   mode 2 ("split2"): q ~ q_hi + q_lo;  q.k ~ q_hi.k_hi + q_lo.k_hi            2 MMAs
//   mode 3 ("split3"): k ~ k_hi + k_lo too; q.k ~ q_hi.k_hi + q_lo.k_hi + q_hi.k_lo  3 MMAs
// with x_hi = bf16(x), x_lo = bf16(x - x_hi). split3 drops only q_lo.k_lo and the BF16
// rounding of the lo parts (both about 2^-16 relative to the dot), so it is FP32-class;
// bf16 and split2 are about 2^-9 relative. Everything after the dot is dsa_index_scores'
// arithmetic per output: term = weights[r * H + h] * fmaxf(scale * dot, 0), summed over
// ascending h from 0.0f; candidacy, valid_cand and the -FLT_MAX store are its code.
//
// Layout: DSA_TC_THREADS threads (4 warps); a block covers DSA_TC_ROWS (16) rows x
// DSA_TC_POOLS (64) pools, warp w the pools [16 w, 16 w + 16) as two 8-pool n-tiles. The
// MMA is M = rows, N = pools, K = head dims; lane (g, t) = (lane >> 2, lane & 3) owns
// C elements (row g | g + 8, pool 2 t | 2 t + 1) of each n-tile, so a (row, pool) score
// accumulates across heads in one thread's registers. The warp's pool keys are read once
// from global (float2, FP32) into B fragments that stay in registers for every head (32
// registers hi, 32 lo at D = 128); q is read per head straight into A fragments (the four
// warps of a block read the same q rows, so they hit L1). No shared memory, no barriers;
// a warp whose pools all lie past P returns at once.
// Registers (estimate, not a ptxas report): 64 B-fragment + 8 accumulator + 8 score + 16
// transient A = about 110.
//
// Not supported: the device-geometry (ceiling) path, D not a multiple of 16, D > 128, mode
// outside 1..3. The host never selects this kernel there; a launch that does traps.

#define DSA_TC_ROWS 16u
#define DSA_TC_POOLS 64u
#define DSA_TC_THREADS 128u
#define DSA_TC_MAX_KSTEP 8u

// 2026-10-01: Two BF16 values as one 32-bit MMA operand register, `lo` in the low half (the
// lower K index), as the PTX fragment layout requires.
__device__ __forceinline__ unsigned int dsa_tc_pack(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (unsigned int)__bfloat16_as_ushort(lo) | ((unsigned int)__bfloat16_as_ushort(hi) << 16);
}

// 2026-10-01: x_hi = bf16(x), x_lo = bf16(x - x_hi) for a K-adjacent pair (x, y).
__device__ __forceinline__ void dsa_tc_split(float x, float y, unsigned int& hi, unsigned int& lo) {
    const __nv_bfloat16 hx = __float2bfloat16_rn(x);
    const __nv_bfloat16 hy = __float2bfloat16_rn(y);
    const __nv_bfloat16 lx = __float2bfloat16_rn(x - __bfloat162float(hx));
    const __nv_bfloat16 ly = __float2bfloat16_rn(y - __bfloat162float(hy));
    hi = dsa_tc_pack(hx, hy);
    lo = dsa_tc_pack(lx, ly);
}

// 2026-10-01: c += A(16x16, row) . B(16x8, col), BF16 in, FP32 accumulate.
__device__ __forceinline__ void dsa_tc_mma(float (&c)[4], unsigned int a0, unsigned int a1,
                                           unsigned int a2, unsigned int a3, unsigned int b0,
                                           unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 2026-10-01: (x, y) = src[k], src[k + 1] when `ok`, else zeros. `src + k` is 8-byte aligned:
// k is even, D is a multiple of 16, and the buffers are allocation-aligned.
__device__ __forceinline__ float2 dsa_tc_ld2(const float* __restrict__ src, unsigned int k, bool ok) {
    if (!ok) return make_float2(0.0f, 0.0f);
    return *reinterpret_cast<const float2*>(src + k);
}

extern "C" __global__ void __launch_bounds__(DSA_TC_THREADS) dsa_index_scores_tc(
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
    const int* __restrict__ geom,
    unsigned int mode
) {
    if (geom || D == 0u || (D & 15u) != 0u || D > 16u * DSA_TC_MAX_KSTEP || mode < 1u || mode > 3u)
        __trap();
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int pw = blockIdx.x * DSA_TC_POOLS + warp * 16u;
    // 2026-10-01: Warp-uniform, and nothing below synchronises the block.
    if (pw >= P) return;
    const unsigned int r0 = blockIdx.y * DSA_TC_ROWS;
    const unsigned int ra = r0 + g;
    const unsigned int rb = r0 + g + 8u;
    const bool va = ra < Q;
    const bool vb = rb < Q;
    const unsigned int nks = D >> 4;

    // 2026-10-01: B fragments of the warp's two n-tiles for every K step: register b of
    // n-tile j at step s holds K indices 16 s + 8 b + 2 t, + 1 of pool pw + 8 j + g.
    unsigned int bh[2][DSA_TC_MAX_KSTEP][2];
    unsigned int bl[2][DSA_TC_MAX_KSTEP][2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const unsigned int p = pw + 8u * (unsigned int)j + g;
        const bool vp = p < P;
        const float* __restrict__ pk = pool_keys + (size_t)(vp ? p : 0u) * D;
#pragma unroll
        for (int s = 0; s < (int)DSA_TC_MAX_KSTEP; ++s) {
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const unsigned int k = 16u * (unsigned int)s + 8u * (unsigned int)b + 2u * t;
                const float2 v = dsa_tc_ld2(pk, k, vp && (unsigned int)s < nks);
                dsa_tc_split(v.x, v.y, bh[j][s][b], bl[j][s][b]);
            }
        }
    }

    float sum[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        sum[j][0] = 0.0f; sum[j][1] = 0.0f; sum[j][2] = 0.0f; sum[j][3] = 0.0f;
    }

    for (unsigned int h = 0; h < H; ++h) {
        float acc[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            acc[j][0] = 0.0f; acc[j][1] = 0.0f; acc[j][2] = 0.0f; acc[j][3] = 0.0f;
        }
        const float* __restrict__ qa = q + ((size_t)(va ? ra : 0u) * H + h) * D;
        const float* __restrict__ qb = q + ((size_t)(vb ? rb : 0u) * H + h) * D;
#pragma unroll
        for (int s = 0; s < (int)DSA_TC_MAX_KSTEP; ++s) {
            if ((unsigned int)s >= nks) break;
            // 2026-10-01: A fragment: a0 (row g, K 16 s + 2 t), a1 (row g + 8, same K),
            // a2 (row g, K + 8), a3 (row g + 8, K + 8).
            const unsigned int k0 = 16u * (unsigned int)s + 2u * t;
            const unsigned int k1 = k0 + 8u;
            const float2 x0 = dsa_tc_ld2(qa, k0, va);
            const float2 x1 = dsa_tc_ld2(qb, k0, vb);
            const float2 x2 = dsa_tc_ld2(qa, k1, va);
            const float2 x3 = dsa_tc_ld2(qb, k1, vb);
            unsigned int ah0, ah1, ah2, ah3, al0, al1, al2, al3;
            dsa_tc_split(x0.x, x0.y, ah0, al0);
            dsa_tc_split(x1.x, x1.y, ah1, al1);
            dsa_tc_split(x2.x, x2.y, ah2, al2);
            dsa_tc_split(x3.x, x3.y, ah3, al3);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                dsa_tc_mma(acc[j], ah0, ah1, ah2, ah3, bh[j][s][0], bh[j][s][1]);
                if (mode >= 2u) dsa_tc_mma(acc[j], al0, al1, al2, al3, bh[j][s][0], bh[j][s][1]);
                if (mode == 3u) dsa_tc_mma(acc[j], ah0, ah1, ah2, ah3, bl[j][s][0], bl[j][s][1]);
            }
        }
        const float wa = va ? weights[(size_t)ra * H + h] : 0.0f;
        const float wb = vb ? weights[(size_t)rb * H + h] : 0.0f;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            sum[j][0] += wa * fmaxf(scale * acc[j][0], 0.0f);
            sum[j][1] += wa * fmaxf(scale * acc[j][1], 0.0f);
            sum[j][2] += wb * fmaxf(scale * acc[j][2], 0.0f);
            sum[j][3] += wb * fmaxf(scale * acc[j][3], 0.0f);
        }
    }

    // 2026-10-01: Candidacy and stores, dsa_index_scores' code per owned (row, pool).
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const unsigned int r = (e < 2) ? ra : rb;
            const unsigned int p = pw + 8u * (unsigned int)j + 2u * t + (unsigned int)(e & 1);
            if (r >= Q || p >= P) continue;
            const int end = pool_indices[(size_t)p * KP + KP - 1];
            const int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
            const bool vis = (end_c <= q_pos[r]) && (valid_keys[end_c] != 0);
            const bool cand = (pool_valid[p] != 0) && vis;
            valid_cand[(size_t)r * P + p] = cand ? 1 : 0;
            out[(size_t)r * P + p] = cand ? sum[j][e] : -FLT_MAX;
        }
    }
}

// 2026-10-07: 2d. dsa_index_scores_tc2: dsa_index_scores_tc in mode 1 ("bf16") with the
// same bytes (out as bit patterns, valid_cand) and a faster dataflow. Opt-in
// (METRALE_GLM_DSA_SCORES_TC2=1, default off), and only where METRALE_GLM_DSA_SCORES_TC=bf16
// would launch dsa_index_scores_tc; resolved with try_kernel; gated on a GPU by
// crates/model-arch/examples/dsa_indexer_tc2_bitparity_microtest.rs (bitwise against
// dsa_index_scores_tc mode 1, and timing).
//
// Why 2c is slow (measured 13 TF/s at P >= 8192, 142 registers, 25% occupancy): every warp
// re-reads q from global in FP32 for every head and K step and converts it, and splits its
// keys into hi + lo though mode 1 uses only hi.
//
// Layout: DSA_TC2_THREADS threads (8 warps); a block covers DSA_TC2_ROWS (32) rows x
// DSA_TC2_POOLS (128) pools, warp w the pools [16 w, 16 w + 16) as two 8-pool n-tiles for
// both 16-row m-tiles. Keys: as in 2c, the warp's pool keys are read once from global (FP32,
// float2) into BF16 B fragments that stay in registers for every head (hi only). q: per head,
// the block's 32 rows are read once from global (FP32, float4), rounded to BF16 and stored
// in shared memory (row stride D + 8 BF16, so the 8 rows of an ldmatrix phase hit distinct
// bank groups); the A fragments come from there with ldmatrix.x4. The next head's q is
// loaded into registers before this head's MMAs and stored into the other buffer after them,
// with one __syncthreads per head. The block's weights sit in shared memory as
// [32][H + 1] floats (rows past Q as 0.0f). Dynamic shared memory: 2 x 32 x (D + 8) x 2 B of
// q plus 32 x (H + 1) x 4 B of weights (21,632 B at D = 128, H = 32).
//
// Why the bytes equal 2c mode 1 for every (row r, pool p) it stores:
// - Operands: q and key elements are __float2bfloat16_rn of the same FP32 values (q rows past
//   Q and keys of pools past P are 0, as in 2c, and are never stored). The A fragment of
//   K step s holds row r's K indices 16 s .. 16 s + 15 and the B fragment pool p's same 16
//   indices, packed lower K in the low half, as dsa_tc_pack does; ldmatrix.x4 matrices
//   (rows 0-7 | 8-15) x (K 0-7 | 8-15) give exactly the a0..a3 registers 2c loads.
// - Dot: per head, acc starts at 0.0f and takes one mma.sync.m16n8k16 BF16 -> FP32 per K step
//   s = 0 .. D / 16 - 1 in ascending order, as 2c's mode-1 chain. Each MMA even has the
//   same whole operands as one of 2c's: m-tiles start at multiples of 16 rows and n-tiles at
//   multiples of 8 pools in both kernels, rows past Q and pools past P are zeros in both, so
//   (r, p) is tile row r % 16, column p % 8 of the same 16 x 8 x 16 product with the same C
//   (the microtest checks the bytes).
// - Head sum: sum += w * fmaxf(scale * acc, 0.0f) from 0.0f in ascending h, the same source
//   expression in the same file, so the same nvcc flags (--fmad=false, KERNEL.toml): one
//   rounded multiply, max, multiply, add, as in 2c. The weight is weights[r * H + h], read
//   through shared memory instead of global (the same float).
// - Candidacy, valid_cand and the -FLT_MAX store are 2c's code.
// ptxas (CUDA 13.0, sm_121f, -O3 --fmad=false, Tower, 2026-10-07): 122 registers, no spills,
// so two 256-thread blocks per SM (__launch_bounds__(256, 2)); 2c: 142 registers.
//
// Not supported (traps): device geometry (geom != NULL), D not a multiple of 16 or above 128,
// mode other than 1. The host also caps H at SCORES_TC2_MAX_H (64), bounding the weights in
// shared memory. The host never selects it outside that envelope.

#define DSA_TC2_ROWS 32u
#define DSA_TC2_POOLS 128u
#define DSA_TC2_THREADS 256u
#define DSA_TC2_WARPS 8u

// 2026-10-07: d0..d3 = the four 8x8 BF16 matrices whose row addresses lanes 0-7, 8-15, 16-23,
// 24-31 pass; register j of a lane holds row lane / 4, columns 2 (lane % 4), + 1 of matrix j.
__device__ __forceinline__ void dsa_tc2_ldsm_x4(unsigned int addr, unsigned int& d0,
                                                unsigned int& d1, unsigned int& d2,
                                                unsigned int& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(addr)
                 : "memory");
}

// 2026-10-07: One block, NKS = D / 16 K steps. `qs` is two [32][D + 8] BF16 q buffers, `ws`
// [32][H + 1] floats, both in dynamic shared memory.
template <int NKS>
__device__ __forceinline__ void dsa_scores_tc2_tile(
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
    unsigned int KP,
    unsigned int S,
    float scale,
    __nv_bfloat16* qs,
    float* ws
) {
    constexpr unsigned int D = 16u * (unsigned int)NKS;
    constexpr unsigned int LD = D + 8u;                 // q row stride, BF16 elements
    constexpr unsigned int BUF = DSA_TC2_ROWS * LD;     // one q buffer, BF16 elements
    constexpr unsigned int F4_ROW = D / 4u;             // float4 per q row
    constexpr unsigned int F4_ALL = DSA_TC2_ROWS * F4_ROW;
    constexpr int F4_IT = (int)((F4_ALL + DSA_TC2_THREADS - 1u) / DSA_TC2_THREADS);

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int r0 = blockIdx.y * DSA_TC2_ROWS;
    const unsigned int pw = blockIdx.x * DSA_TC2_POOLS + warp * 16u;
    // 2026-10-07: Warp-uniform. A warp past P still stages q and meets every barrier.
    const bool active = pw < P;
    const unsigned int WS = H + 1u;

    // 2026-10-07: Weights of the block's rows, 0.0f past Q (2c's `va ? ... : 0.0f`).
    for (unsigned int e = tid; e < DSA_TC2_ROWS * H; e += DSA_TC2_THREADS) {
        const unsigned int rr = e / H;
        const unsigned int hh = e - rr * H;
        const unsigned int r = r0 + rr;
        ws[rr * WS + hh] = r < Q ? weights[(size_t)r * H + hh] : 0.0f;
    }

    // 2026-10-07: B fragments, 2c's layout and rounding (hi only): register b of n-tile j at
    // step s holds K indices 16 s + 8 b + 2 t, + 1 of pool pw + 8 j + g.
    unsigned int bk[2][NKS][2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const unsigned int p = pw + 8u * (unsigned int)j + g;
        const bool vp = active && p < P;
        const float* __restrict__ pk = pool_keys + (size_t)(vp ? p : 0u) * D;
#pragma unroll
        for (int s = 0; s < NKS; ++s) {
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const unsigned int k = 16u * (unsigned int)s + 8u * (unsigned int)b + 2u * t;
                const float2 v = dsa_tc_ld2(pk, k, vp);
                bk[j][s][b] = dsa_tc_pack(__float2bfloat16_rn(v.x), __float2bfloat16_rn(v.y));
            }
        }
    }

    // 2026-10-07: q staging: float4 e of the block's head-h tile is row e / F4_ROW, elements
    // 4 (e % F4_ROW) .. + 3; rows past Q are zeros (2c's dsa_tc_ld2 with ok = false).
    float4 pf[F4_IT];
    auto q_load = [&](unsigned int h) {
#pragma unroll
        for (int it = 0; it < F4_IT; ++it) {
            const unsigned int e = tid + (unsigned int)it * DSA_TC2_THREADS;
            const unsigned int rr = e / F4_ROW;
            const unsigned int c4 = e - rr * F4_ROW;
            const unsigned int r = r0 + rr;
            pf[it] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (e < F4_ALL && r < Q)
                pf[it] = *reinterpret_cast<const float4*>(q + ((size_t)r * H + h) * D + 4u * c4);
        }
    };
    auto q_store = [&](unsigned int buf) {
#pragma unroll
        for (int it = 0; it < F4_IT; ++it) {
            const unsigned int e = tid + (unsigned int)it * DSA_TC2_THREADS;
            if (e >= F4_ALL) continue;
            const unsigned int rr = e / F4_ROW;
            const unsigned int c4 = e - rr * F4_ROW;
            uint2 v;
            v.x = dsa_tc_pack(__float2bfloat16_rn(pf[it].x), __float2bfloat16_rn(pf[it].y));
            v.y = dsa_tc_pack(__float2bfloat16_rn(pf[it].z), __float2bfloat16_rn(pf[it].w));
            *reinterpret_cast<uint2*>(qs + buf * BUF + rr * LD + 4u * c4) = v;
        }
    };

    // 2026-10-07: ldmatrix row address of lane l for m-tile i at K step 0: matrix l / 8 is
    // (rows + 8 ((l / 8) & 1)) x (K + 8 (l / 16)), row l % 8 of it.
    const unsigned int qs_base = (unsigned int)__cvta_generic_to_shared(qs);
    unsigned int a_off[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const unsigned int row = 16u * (unsigned int)i + (lane & 7u) + ((lane >> 3) & 1u) * 8u;
        a_off[i] = (row * LD + (lane >> 4) * 8u) * 2u;
    }

    float sum[2][2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            sum[i][j][0] = 0.0f; sum[i][j][1] = 0.0f; sum[i][j][2] = 0.0f; sum[i][j][3] = 0.0f;
        }
    }

    if (H > 0u) {
        q_load(0u);
        q_store(0u);
    }
    __syncthreads();

    for (unsigned int h = 0; h < H; ++h) {
        const unsigned int cur = h & 1u;
        if (h + 1u < H) q_load(h + 1u);
        if (active) {
            float acc[2][2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    acc[i][j][0] = 0.0f; acc[i][j][1] = 0.0f;
                    acc[i][j][2] = 0.0f; acc[i][j][3] = 0.0f;
                }
            }
            const unsigned int qb = qs_base + cur * BUF * 2u;
#pragma unroll
            for (int s = 0; s < NKS; ++s) {
                unsigned int a[2][4];
#pragma unroll
                for (int i = 0; i < 2; ++i)
                    dsa_tc2_ldsm_x4(qb + a_off[i] + 32u * (unsigned int)s, a[i][0], a[i][1],
                                    a[i][2], a[i][3]);
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int j = 0; j < 2; ++j)
                        dsa_tc_mma(acc[i][j], a[i][0], a[i][1], a[i][2], a[i][3], bk[j][s][0],
                                   bk[j][s][1]);
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const float wa = ws[(16u * (unsigned int)i + g) * WS + h];
                const float wb = ws[(16u * (unsigned int)i + g + 8u) * WS + h];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    sum[i][j][0] += wa * fmaxf(scale * acc[i][j][0], 0.0f);
                    sum[i][j][1] += wa * fmaxf(scale * acc[i][j][1], 0.0f);
                    sum[i][j][2] += wb * fmaxf(scale * acc[i][j][2], 0.0f);
                    sum[i][j][3] += wb * fmaxf(scale * acc[i][j][3], 0.0f);
                }
            }
        }
        // 2026-10-07: The other buffer was last read for head h - 1, before the barrier that
        // ended that iteration; this head's buffer is read again only after the next barrier.
        if (h + 1u < H) q_store(cur ^ 1u);
        __syncthreads();
    }

    if (!active) return;
    // 2026-10-07: Candidacy and stores, 2c's code per owned (row, pool).
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const unsigned int r = r0 + 16u * (unsigned int)i + g + ((e < 2) ? 0u : 8u);
                const unsigned int p = pw + 8u * (unsigned int)j + 2u * t + (unsigned int)(e & 1);
                if (r >= Q || p >= P) continue;
                const int end = pool_indices[(size_t)p * KP + KP - 1];
                const int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
                const bool vis = (end_c <= q_pos[r]) && (valid_keys[end_c] != 0);
                const bool cand = (pool_valid[p] != 0) && vis;
                valid_cand[(size_t)r * P + p] = cand ? 1 : 0;
                out[(size_t)r * P + p] = cand ? sum[i][j][e] : -FLT_MAX;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(DSA_TC2_THREADS, 2) dsa_index_scores_tc2(
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
    const int* __restrict__ geom,
    unsigned int mode
) {
    if (geom || D == 0u || (D & 15u) != 0u || D > 16u * DSA_TC_MAX_KSTEP || mode != 1u)
        __trap();
    extern __shared__ uint4 dsa_tc2_smem[];
    __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(dsa_tc2_smem);
    float* ws = reinterpret_cast<float*>(qs + 2u * DSA_TC2_ROWS * (D + 8u));
#define DSA_TC2_CASE(N)                                                                      \
    case N:                                                                                  \
        dsa_scores_tc2_tile<N>(q, pool_keys, weights, pool_indices, pool_valid, valid_keys,  \
                               q_pos, out, valid_cand, Q, P, H, KP, S, scale, qs, ws);       \
        break;
    switch (D >> 4) {
        DSA_TC2_CASE(1)
        DSA_TC2_CASE(2)
        DSA_TC2_CASE(3)
        DSA_TC2_CASE(4)
        DSA_TC2_CASE(5)
        DSA_TC2_CASE(6)
        DSA_TC2_CASE(7)
        DSA_TC2_CASE(8)
    default:
        __trap();
    }
#undef DSA_TC2_CASE
}

// 2026-10-08: 2e. dsa_index_scores_decode: the scores and candidacy of dsa_index_scores,
// byte-identical by construction, for the decode (ceiling, graph-replay) launch, which the
// tiled and tensor-core scorers do not serve. Opt-in (METRALE_GLM_DSA_SCORES_DECODE=1, host
// side crates/model-arch/src/glm5next_dsa/select/scores_decode.rs); resolved with try_kernel;
// gated on a GPU by crates/model-arch/examples/dsa_scores_decode_bitparity_microtest.rs.
//
// Why: under a ceiling launch dsa_index_scores runs one pool per block iteration, one warp
// per head, and re-reads the row's whole q (H x D FP32, 16 KB at H 32, D 128) from global
// for every pool, so it measured 20.2 / 41.9 ms a decode step (11 layers x 3 rows) at 131K /
// 262K against a ~4.4 ms key-byte floor at 262K (2026-10-08, one GB10).
//
// Geometry: the same arguments as dsa_index_scores. With `geom` non-null (the ceiling path)
// S and P come from geom[DSA_GEOM_S] / geom[DSA_GEOM_NPOOLS] exactly as dsa_index_scores
// reads them, and the live pool count is that P; the grid is fixed (the host sizes it from
// the SM count, never from the context) and the warps walk the live pools with a grid
// stride, so a captured graph replays over any live context. With `geom` null the scalars
// are used and every one of the P pools is scored. `P` is the row stride of out /
// valid_cand, as in dsa_index_scores, so a ceiling launch must have Q == 1 (the launcher
// refuses Q > 1 there).
//
// Layout: DSA_DEC_THREADS threads (4 warps), blockIdx.y = row r. Shared memory (dynamic,
// (H * D + H + DSA_DEC_WARPS * 32 * DSA_DEC_LD) * 4 bytes, 33,408 at H 32): the row's q
// [H][D] and weights [H], staged once per block, and per warp a [32][DSA_DEC_LD] key
// transpose buffer. Warp-tile t covers pools [32 t, 32 t + 32), lane l owns pool 32 t + l;
// a block's warps take tiles blockIdx.x * 4 + w, then + gridDim.x * 4, ... below the live
// count. Keys reach the lane's registers (k[0..D)) through the transpose buffer in four
// 32-dim chunks: each warp load instruction is 8 lanes x 16 B over 4 pools' contiguous
// 128-byte chunks (whole sectors, coalesced), stored at row * 33 + col (conflict-free), then
// lane l reads row l (bank (l + j) % 32, conflict-free). q is read from shared memory as
// float4 broadcasts (all lanes read the same address).
//
// Why the bits match (the common build passes --fmad=false, KERNEL.toml, so in
// dsa_index_scores a * b + c is a rounded multiply then a rounded add; here every multiply
// and add is written __fmul_rn / __fadd_rn, which are never contracted, so this kernel
// computes that same sequence whatever the fmad setting):
// - In dsa_index_scores, lane l of head h's warp sums qh[d] * pk[d] over d = l, l + 32, ...
//   (d < D), ascending, from 0.0f: x0[l]. Here the owning thread forms the same chain for
//   every l (dsa_dec_x0: 0.0f + q[l] k[l], then + q[l + 32] k[l + 32], ...), from the same
//   values (shared/register copies of q and the pool key, not recomputed).
// - The shuffle_down tree with offsets 16, 8, 4, 2, 1 leaves at lane 0
//   x1[i] = x0[i] + x0[i + 16], x2[i] = x1[i] + x1[i + 8], x3[i] = x2[i] + x2[i + 4],
//   x4[i] = x3[i] + x3[i + 2], x5 = x4[0] + x4[1], lower lane on the left (as the 2b argument
//   above); dsa_dec_tree evaluates exactly those nodes in place, left operand first.
// - The head term is weights[r * H + h] * fmaxf(scale * x5, 0.0f) (the same fmaxf operand
//   order), and the score is 0.0f + term(0) + term(1) + ... in ascending h, as thread 0 of
//   dsa_index_scores sums sh[] there; one thread owns a pool's whole sum, so nothing is split.
// - Candidacy, valid_cand and the -FLT_MAX store are dsa_index_scores' code, per pool.
//   Keys of a non-candidate pool are loaded (all below the live count, so in bounds) but its
//   score is never stored.
//
// Registers (estimate, not a ptxas report): 128 key floats, 32 chain/tree floats, 16 q
// floats in flight and addressing, about 190, under the 255 that __launch_bounds__(128, 2)
// allows. Not supported: D != 128 or blockDim.x != DSA_DEC_THREADS; the host never selects
// this kernel there, and a launch that does traps instead of computing anything.

#define DSA_DEC_THREADS 128u
#define DSA_DEC_WARPS (DSA_DEC_THREADS / 32u)
#define DSA_DEC_D 128u
#define DSA_DEC_NJ (DSA_DEC_D / 32u)
#define DSA_DEC_LD 33u

// 2026-10-08: x0[l0 .. l0 + 3] (see 2e) for one head: d = l + 32 j, j ascending, from 0.0f,
// each product rounded before its add. qh is the head's [D] q in shared memory (16-byte
// aligned), k the lane's pool key.
__device__ __forceinline__ void dsa_dec_x0x4(
    const float* __restrict__ qh, const float (&k)[DSA_DEC_D], unsigned int l0, float (&x)[32]
) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
    for (unsigned int j = 0; j < DSA_DEC_NJ; ++j) {
        const float4 qv = *reinterpret_cast<const float4*>(qh + l0 + 32u * j);
        a0 = __fadd_rn(a0, __fmul_rn(qv.x, k[l0 + 32u * j + 0u]));
        a1 = __fadd_rn(a1, __fmul_rn(qv.y, k[l0 + 32u * j + 1u]));
        a2 = __fadd_rn(a2, __fmul_rn(qv.z, k[l0 + 32u * j + 2u]));
        a3 = __fadd_rn(a3, __fmul_rn(qv.w, k[l0 + 32u * j + 3u]));
    }
    x[l0 + 0u] = a0;
    x[l0 + 1u] = a1;
    x[l0 + 2u] = a2;
    x[l0 + 3u] = a3;
}

// 2026-10-08: x5 from x0[0..32): the shuffle_down(16, 8, 4, 2, 1) tree's lane-0 value (2e).
__device__ __forceinline__ float dsa_dec_tree(float (&x)[32]) {
#pragma unroll
    for (unsigned int i = 0; i < 16u; ++i) x[i] = __fadd_rn(x[i], x[i + 16u]);
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) x[i] = __fadd_rn(x[i], x[i + 8u]);
#pragma unroll
    for (unsigned int i = 0; i < 4u; ++i) x[i] = __fadd_rn(x[i], x[i + 4u]);
#pragma unroll
    for (unsigned int i = 0; i < 2u; ++i) x[i] = __fadd_rn(x[i], x[i + 2u]);
    return __fadd_rn(x[0], x[1]);
}

extern "C" __global__ void __launch_bounds__(DSA_DEC_THREADS, 2) dsa_index_scores_decode(
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
    if (D != DSA_DEC_D || blockDim.x != DSA_DEC_THREADS) __trap();
    (void)Q;
    const unsigned int r = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
    }
    const unsigned int live = P;

    extern __shared__ __align__(16) float dsa_dec_sh[];
    float* qs = dsa_dec_sh;
    float* ws = qs + (size_t)H * DSA_DEC_D;
    float* st = ws + H + warp * 32u * DSA_DEC_LD;

    // 2026-10-08: The row's q and weights, once per block. The only block-wide barrier: after
    // it every warp runs on its own (warp-uniform loop, __syncwarp only).
    const float* __restrict__ qrow = q + (size_t)r * H * DSA_DEC_D;
    for (unsigned int i = tid; i < H * DSA_DEC_D; i += DSA_DEC_THREADS) qs[i] = qrow[i];
    for (unsigned int i = tid; i < H; i += DSA_DEC_THREADS) ws[i] = weights[(size_t)r * H + i];
    __syncthreads();

    const unsigned int step = gridDim.x * DSA_DEC_WARPS;
    for (unsigned int t = blockIdx.x * DSA_DEC_WARPS + warp; (size_t)t * 32u < (size_t)live;
         t += step) {
        const unsigned int p0 = t * 32u;
        const unsigned int p = p0 + lane;
        const bool in = p < live;

        // 2026-10-08: dsa_index_scores' candidacy and stores, per pool.
        bool cand = false;
        if (in) {
            int end = pool_indices[p * KP + KP - 1];
            int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
            bool vis = (end_c <= q_pos[r]) && (valid_keys[end_c] != 0);
            cand = (pool_valid[p] != 0) && vis;
            valid_cand[(size_t)r * P + p] = cand ? 1 : 0;
            if (!cand) out[(size_t)r * P + p] = -FLT_MAX;
        }

        // 2026-10-08: The tile's keys into registers, chunk c = dims [32 c, 32 c + 32).
        float k[DSA_DEC_D];
#pragma unroll
        for (unsigned int c = 0; c < DSA_DEC_NJ; ++c) {
#pragma unroll
            for (unsigned int i = 0; i < 8u; ++i) {
                const unsigned int row = i * 4u + (lane >> 3);
                const unsigned int col = (lane & 7u) * 4u;
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (p0 + row < live) {
                    v = *reinterpret_cast<const float4*>(
                        pool_keys + (size_t)(p0 + row) * DSA_DEC_D + c * 32u + col);
                }
                float* s = st + row * DSA_DEC_LD + col;
                s[0] = v.x;
                s[1] = v.y;
                s[2] = v.z;
                s[3] = v.w;
            }
            __syncwarp();
#pragma unroll
            for (unsigned int j = 0; j < 32u; ++j) k[c * 32u + j] = st[lane * DSA_DEC_LD + j];
            __syncwarp();
        }

        // 2026-10-08: The score, every head in ascending order (see 2e). Lanes past the live
        // count or not candidates compute on zero / unused keys and store nothing.
        float acc = 0.0f;
        for (unsigned int h = 0; h < H; ++h) {
            const float* qh = qs + (size_t)h * DSA_DEC_D;
            float x[32];
#pragma unroll
            for (unsigned int l0 = 0; l0 < 32u; l0 += 4u) dsa_dec_x0x4(qh, k, l0, x);
            const float dot = dsa_dec_tree(x);
            const float term = __fmul_rn(ws[h], fmaxf(__fmul_rn(scale, dot), 0.0f));
            acc = __fadd_rn(acc, term);
        }
        if (in && cand) out[(size_t)r * P + p] = acc;
    }
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

// 2026-10-06: 3b. Exact radix top-k over pools (METRALE_GLM_DSA_TOPK_RADIX=1, opt-in): the
// same output as dsa_topk_pools, byte for byte, without its per-tile bitonic sorts. The host
// (crates/model-arch/src/glm5next_dsa/select/radix.rs) takes it only when the pool count it
// dispatches on (n_pools, or the ceiling pool count of a graph launch) exceeds topk_tile();
// below that dsa_topk_pools sorts one tile and is already cheap.
//
// Order (the contract of dsa_topk_pools): score descending, then pool index ascending, with
// float compare semantics: -0.0 ties +0.0, and -FLT_MAX (a non-candidate) is an ordinary
// value that ties with itself. dsa_radix_key maps a score to a uint32 that orders exactly so:
// -0.0 is canonicalised to +0.0 first, then a non-negative float gets its sign bit set and a
// negative one is bit-inverted. NaN is out of scope (scores are finite sums of relu terms).
//
// Sequence per call, all on one stream (9 launches; every block of a launch takes the same
// early exit when select_k is 0, so no barrier is skipped by part of a block):
//   init        (grid Q)     zero the two global histograms and the state words of each row.
//   hist pass 0 (grid C x Q) histogram of key bits 31..20 (4096 bins) over the row's chunk,
//                            warp-aggregated into shared memory, then atomicAdd into hist0.
//   find pass 0 (grid Q)     scan the bins from the top until the count reaches select_k:
//                            that bin is the threshold's top 12 bits; `need` becomes the count
//                            still wanted inside it.
//   hist/find pass 1         bits 19..8 (4096 bins, hist1) over keys matching the prefix.
//   hist/find pass 2         bits 7..0 (256 bins) over keys matching the 24-bit prefix; each
//                            chunk stores its own 256-bin histogram (no atomics), so find can
//                            read the per-chunk count of the threshold key tau. After it:
//                            tau, m = ties at tau to take (>= 1), n_gt = select_k - m =
//                            count(key > tau), and tie_base[c] = ties at tau in chunks < c.
//   gather      (grid C x Q) keys > tau go to candidate slots [0, n_gt) through an atomic
//                            counter (any order); a key == tau at global tie rank t (index
//                            order: tie_base[c] + rank within the chunk from a block scan) goes
//                            to slot n_gt + t when t < m. So exactly select_k candidates, and
//                            threshold ties are the m lowest indices, deterministically, with no
//                            tie buffer however many -FLT_MAX ties there are.
//   sort        (grid Q)     bitonic sort of the select_k [key, index] pairs, key descending
//                            then index ascending (a total order over unique indices), and the
//                            indices written to selected[r * select_k + i].
// The candidate set equals the first select_k pools of the total order, and the sort orders it
// by that order, so `selected` equals dsa_topk_pools'. The row stride of `scores` is P and that
// of `selected` is select_k, both from `geom` when it is non-null, as in dsa_topk_pools.
//
// Ceiling launch: P and select_k come from geom; the grids are fixed (C chunks per row) and the
// chunk width ceil(P / C) follows the live P, so a chunk past the live pools is empty (it still
// stores its zero pass-2 histogram and gathers nothing). init re-zeroes the global scratch in
// every call, inside the captured sequence.
//
// Work buffer per row (u32 words, row stride DSA_RADIX_CAND + 2 * kcap):
//   [0, 4096) hist0, [4096, 8192) hist1; pass 2's per-chunk histograms [c * 256, c * 256 + 256)
//   reuse [0, C * 256) (C <= 32, so at most both) once find pass 1 has read hist1.
//   [8192, 8208) state, [8208, 8240) tie_base per chunk, then kcap candidate keys and kcap
//   candidate indices.

#define DSA_RADIX_THREADS 256u
#define DSA_RADIX_MAX_CHUNKS 32u
#define DSA_RADIX_BINS 4096u
#define DSA_RADIX_BINS_LAST 256u
#define DSA_RADIX_STATE 8192u
#define DSA_RADIX_STATE_WORDS 16u
#define DSA_RADIX_TIE 8208u
#define DSA_RADIX_CAND 8240u
#define DSA_RADIX_SORT_MAX 2048u

// 2026-10-06: State word slots (offsets from DSA_RADIX_STATE).
#define DSA_RS_PREFIX 0u
#define DSA_RS_NEED   1u
#define DSA_RS_GT_CNT 2u
#define DSA_RS_NGT    3u

// 2026-10-06: Monotone key: a > b as floats (with -0.0 == +0.0) iff key(a) > key(b).
__device__ __forceinline__ unsigned int dsa_radix_key(float v) {
    unsigned int u = __float_as_uint(v);
    if (u == 0x80000000u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// 2026-10-06: Whether `key` matches the threshold prefix fixed by the passes before `pass`.
__device__ __forceinline__ bool dsa_radix_match(unsigned int key, unsigned int prefix,
                                                unsigned int pass) {
    if (pass == 0u) return true;
    if (pass == 1u) return (key >> 20) == (prefix >> 20);
    return (key >> 8) == (prefix >> 8);
}

// 2026-10-06: The bin of `key` in `pass`: bits 31..20, 19..8, then 7..0.
__device__ __forceinline__ unsigned int dsa_radix_digit(unsigned int key, unsigned int pass) {
    if (pass == 0u) return key >> 20;
    if (pass == 1u) return (key >> 8) & 0xFFFu;
    return key & 0xFFu;
}

// 2026-10-06: Exclusive block scan of v over threadIdx.x (blockDim.x a multiple of 32, at most
// 1024); `*total` gets the block sum. Every thread of the block must call it (it has barriers);
// `ws` is a 32-word shared scratch, free again when it returns.
__device__ __forceinline__ unsigned int dsa_radix_excl_scan(unsigned int v, unsigned int* ws,
                                                            unsigned int* total) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int nwarps = blockDim.x >> 5;
    unsigned int x = v;
    for (unsigned int off = 1; off < 32u; off <<= 1) {
        const unsigned int y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= off) x += y;
    }
    if (lane == 31u) ws[warp] = x;
    __syncthreads();
    if (warp == 0u) {
        unsigned int s = (lane < nwarps) ? ws[lane] : 0u;
        for (unsigned int off = 1; off < 32u; off <<= 1) {
            const unsigned int y = __shfl_up_sync(0xffffffffu, s, off);
            if (lane >= off) s += y;
        }
        ws[lane] = s;
    }
    __syncthreads();
    const unsigned int base = (warp == 0u) ? 0u : ws[warp - 1u];
    *total = ws[nwarps - 1u];
    __syncthreads();
    return base + x - v;
}

// 2026-10-06: One row's work buffer.
__device__ __forceinline__ unsigned int* dsa_radix_row(unsigned int* work, unsigned int r,
                                                       unsigned int kcap) {
    return work + (size_t)r * (DSA_RADIX_CAND + 2u * kcap);
}

// 2026-10-06: Zero hist0, hist1 and the state words of row blockIdx.x.
extern "C" __global__ void __launch_bounds__(DSA_RADIX_THREADS) dsa_topk_radix_init(
    unsigned int* __restrict__ work,
    unsigned int select_k,
    unsigned int kcap,
    const int* __restrict__ geom
) {
    if (geom) select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    if (select_k == 0u) return;
    // 2026-10-06: The host sizes kcap >= every select_k it plans; a larger one would overrun
    // the candidate buffer and the sort's shared arrays.
    if (select_k > kcap || kcap > DSA_RADIX_SORT_MAX) __trap();
    unsigned int* w = dsa_radix_row(work, blockIdx.x, kcap);
    for (unsigned int i = threadIdx.x; i < DSA_RADIX_STATE + DSA_RADIX_STATE_WORDS; i += blockDim.x)
        w[i] = 0u;
}

// 2026-10-06: Histogram pass `pass` over chunk blockIdx.x of row blockIdx.y.
extern "C" __global__ void __launch_bounds__(DSA_RADIX_THREADS) dsa_topk_radix_hist(
    const float* __restrict__ scores,
    unsigned int* __restrict__ work,
    unsigned int P,
    unsigned int select_k,
    unsigned int kcap,
    unsigned int pass,
    const int* __restrict__ geom
) {
    if (geom) {
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    if (select_k == 0u) return;
    if (gridDim.x > DSA_RADIX_MAX_CHUNKS) __trap();
    __shared__ unsigned int sh[DSA_RADIX_BINS];
    const unsigned int tid = threadIdx.x;
    const unsigned int c = blockIdx.x;
    const unsigned int r = blockIdx.y;
    unsigned int* w = dsa_radix_row(work, r, kcap);
    const unsigned int nbins = (pass == 2u) ? DSA_RADIX_BINS_LAST : DSA_RADIX_BINS;
    for (unsigned int b = tid; b < nbins; b += blockDim.x) sh[b] = 0u;
    __syncthreads();

    const unsigned int prefix = (pass == 0u) ? 0u : w[DSA_RADIX_STATE + DSA_RS_PREFIX];
    const unsigned int chunk = (P + gridDim.x - 1u) / gridDim.x;
    const unsigned int lo = min(c * chunk, P);
    const unsigned int hi = min(lo + chunk, P);
    const float* row = scores + (size_t)r * P;
    const unsigned int lane = tid & 31u;
    // 2026-10-06: The trip count depends on lo, hi and blockDim only, so every lane of every
    // warp reaches the ballot; lanes past `hi` vote false.
    for (unsigned int base = lo; base < hi; base += blockDim.x) {
        const unsigned int i = base + tid;
        bool live = i < hi;
        unsigned int digit = 0u;
        if (live) {
            const unsigned int key = dsa_radix_key(row[i]);
            live = dsa_radix_match(key, prefix, pass);
            digit = dsa_radix_digit(key, pass);
        }
        const unsigned int active = __ballot_sync(0xffffffffu, live);
        if (live) {
            // 2026-10-06: One shared atomic per distinct bin in the warp (thousands of
            // -FLT_MAX ties fall in one bin).
            const unsigned int peers = __match_any_sync(active, digit);
            if (lane == (unsigned int)(__ffs(peers) - 1))
                atomicAdd(&sh[digit], (unsigned int)__popc(peers));
        }
    }
    __syncthreads();

    if (pass < 2u) {
        unsigned int* g = w + pass * DSA_RADIX_BINS;
        for (unsigned int b = tid; b < DSA_RADIX_BINS; b += blockDim.x)
            if (sh[b] != 0u) atomicAdd(&g[b], sh[b]);
    } else {
        // 2026-10-06: Every chunk, empty or not, stores all its bins.
        unsigned int* g = w + c * DSA_RADIX_BINS_LAST;
        for (unsigned int b = tid; b < DSA_RADIX_BINS_LAST; b += blockDim.x) g[b] = sh[b];
    }
}

// 2026-10-06: Count of bin `b` in `pass`: the global histogram, or (pass 2) the per-chunk
// histograms summed over the `chunks` chunks.
__device__ __forceinline__ unsigned int dsa_radix_bin_count(const unsigned int* w,
                                                            unsigned int pass, unsigned int b,
                                                            unsigned int chunks) {
    if (pass < 2u) return w[pass * DSA_RADIX_BINS + b];
    unsigned int s = 0u;
    for (unsigned int c = 0; c < chunks; ++c) s += w[c * DSA_RADIX_BINS_LAST + b];
    return s;
}

// 2026-10-06: Find pass `pass` for row blockIdx.x: the bin where the count from the top reaches
// `need`. Thread t owns bins [nbins - (t + 1) * per, nbins - t * per), walked from the top, so
// the exclusive scan over t is the count above its bins. Exactly one thread has
// excl < need <= excl + local (need >= 1 and the bins hold at least `need` keys); it extends the
// prefix and, after pass 2, writes n_gt and the per-chunk tie bases. `chunks` is the grid x of
// the hist launches.
extern "C" __global__ void __launch_bounds__(DSA_RADIX_THREADS) dsa_topk_radix_find(
    unsigned int* __restrict__ work,
    unsigned int select_k,
    unsigned int kcap,
    unsigned int pass,
    unsigned int chunks,
    const int* __restrict__ geom
) {
    if (geom) select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    if (select_k == 0u) return;
    // 2026-10-06: The bin split assumes DSA_RADIX_THREADS threads (16 bins, then 1, each).
    if (blockDim.x != DSA_RADIX_THREADS || chunks > DSA_RADIX_MAX_CHUNKS) __trap();
    __shared__ unsigned int ws[32];
    const unsigned int tid = threadIdx.x;
    unsigned int* w = dsa_radix_row(work, blockIdx.x, kcap);
    const unsigned int nbins = (pass == 2u) ? DSA_RADIX_BINS_LAST : DSA_RADIX_BINS;
    const unsigned int per = nbins / blockDim.x;
    const unsigned int need = (pass == 0u) ? select_k : w[DSA_RADIX_STATE + DSA_RS_NEED];
    const unsigned int prefix = (pass == 0u) ? 0u : w[DSA_RADIX_STATE + DSA_RS_PREFIX];
    const unsigned int top = nbins - 1u - tid * per;
    unsigned int local = 0u;
    for (unsigned int j = 0; j < per; ++j) local += dsa_radix_bin_count(w, pass, top - j, chunks);
    unsigned int total;
    const unsigned int excl = dsa_radix_excl_scan(local, ws, &total);
    // 2026-10-06: The scan's barriers order every read of the state above before the write
    // below.
    if (excl < need && need <= excl + local) {
        unsigned int cum = excl;
        unsigned int b = top;
        for (unsigned int j = 0; j < per; ++j) {
            b = top - j;
            const unsigned int h = dsa_radix_bin_count(w, pass, b, chunks);
            if (cum + h >= need) break;
            cum += h;
        }
        const unsigned int shift = (pass == 0u) ? 20u : ((pass == 1u) ? 8u : 0u);
        const unsigned int left = need - cum;
        w[DSA_RADIX_STATE + DSA_RS_PREFIX] = prefix | (b << shift);
        w[DSA_RADIX_STATE + DSA_RS_NEED] = left;
        if (pass == 2u) {
            w[DSA_RADIX_STATE + DSA_RS_NGT] = select_k - left;
            unsigned int run = 0u;
            for (unsigned int c = 0; c < chunks; ++c) {
                w[DSA_RADIX_TIE + c] = run;
                run += w[c * DSA_RADIX_BINS_LAST + b];
            }
        }
    }
}

// 2026-10-06: Gather the select_k candidates of chunk blockIdx.x of row blockIdx.y (see 3b).
extern "C" __global__ void __launch_bounds__(DSA_RADIX_THREADS) dsa_topk_radix_gather(
    const float* __restrict__ scores,
    unsigned int* __restrict__ work,
    unsigned int P,
    unsigned int select_k,
    unsigned int kcap,
    const int* __restrict__ geom
) {
    if (geom) {
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    if (select_k == 0u) return;
    __shared__ unsigned int ws[32];
    const unsigned int tid = threadIdx.x;
    const unsigned int c = blockIdx.x;
    const unsigned int r = blockIdx.y;
    unsigned int* w = dsa_radix_row(work, r, kcap);
    const unsigned int tau = w[DSA_RADIX_STATE + DSA_RS_PREFIX];
    const unsigned int m = w[DSA_RADIX_STATE + DSA_RS_NEED];
    const unsigned int ngt = w[DSA_RADIX_STATE + DSA_RS_NGT];
    unsigned int* cand_key = w + DSA_RADIX_CAND;
    unsigned int* cand_idx = w + DSA_RADIX_CAND + kcap;

    const unsigned int chunk = (P + gridDim.x - 1u) / gridDim.x;
    const unsigned int lo = min(c * chunk, P);
    const unsigned int hi = min(lo + chunk, P);
    const float* row = scores + (size_t)r * P;
    // 2026-10-06: Ties at tau in chunks before this one; the loop adds this chunk's in index
    // order.
    unsigned int run = w[DSA_RADIX_TIE + c];
    for (unsigned int base = lo; base < hi; base += blockDim.x) {
        const unsigned int i = base + tid;
        const bool live = i < hi;
        const unsigned int key = live ? dsa_radix_key(row[i]) : 0u;
        if (live && key > tau) {
            const unsigned int slot = atomicAdd(&w[DSA_RADIX_STATE + DSA_RS_GT_CNT], 1u);
            if (slot < ngt) {
                cand_key[slot] = key;
                cand_idx[slot] = i;
            }
        }
        const unsigned int tie = (live && key == tau) ? 1u : 0u;
        unsigned int total;
        const unsigned int rank = run + dsa_radix_excl_scan(tie, ws, &total);
        if (tie != 0u && rank < m) {
            cand_key[ngt + rank] = key;
            cand_idx[ngt + rank] = i;
        }
        run += total;
    }
}

// 2026-10-06: Sort row blockIdx.x's select_k candidates (key descending, then index ascending)
// and write their pool indices.
extern "C" __global__ void __launch_bounds__(DSA_RADIX_THREADS) dsa_topk_radix_sort(
    const unsigned int* __restrict__ work,
    int* __restrict__ selected,
    unsigned int select_k,
    unsigned int kcap,
    const int* __restrict__ geom
) {
    if (geom) select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    if (select_k == 0u) return;
    __shared__ unsigned int sk[DSA_RADIX_SORT_MAX];
    __shared__ int si[DSA_RADIX_SORT_MAX];
    const unsigned int tid = threadIdx.x;
    const unsigned int r = blockIdx.x;
    const unsigned int* w = work + (size_t)r * (DSA_RADIX_CAND + 2u * kcap);
    const unsigned int* cand_key = w + DSA_RADIX_CAND;
    const unsigned int* cand_idx = w + DSA_RADIX_CAND + kcap;
    unsigned int n2 = 2u;
    while (n2 < select_k) n2 <<= 1;
    // 2026-10-06: Padding sorts last on both keys (key 0 is below every finite score's key).
    for (unsigned int i = tid; i < n2; i += blockDim.x) {
        sk[i] = (i < select_k) ? cand_key[i] : 0u;
        si[i] = (i < select_k) ? (int)cand_idx[i] : INT_MAX;
    }
    __syncthreads();

#define DSA_RADIX_GT(a, b) ((sk[(a)] > sk[(b)]) || (sk[(a)] == sk[(b)] && si[(a)] < si[(b)]))
    for (unsigned int k = 2; k <= n2; k <<= 1) {
        for (unsigned int j = k >> 1; j > 0; j >>= 1) {
            for (unsigned int i = tid; i < n2; i += blockDim.x) {
                const unsigned int l = i ^ j;
                if (l > i) {
                    const bool gt = DSA_RADIX_GT(i, l);
                    const bool want_desc = ((i & k) == 0);
                    if (want_desc != gt) {
                        const unsigned int tk = sk[i]; sk[i] = sk[l]; sk[l] = tk;
                        const int ti = si[i]; si[i] = si[l]; si[l] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
#undef DSA_RADIX_GT

    for (unsigned int i = tid; i < select_k; i += blockDim.x)
        selected[(size_t)r * select_k + i] = si[i];
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
