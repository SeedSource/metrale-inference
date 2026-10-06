// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GLM-5.3-Flash selected-index MLA decode over the paged FP8 latent cache
// (module `glm5next_dsa_mla_decode`, entry `glm5next_dsa_mla_decode_fp8`).
// Owner: gb10 kernels (glm-5.3-flash).
// Invariants: none beyond the launch contract below.
//
// NoPE: qk_rope_head_dim is 0, so a cache token is the latent alone and there is no rope
// arm. The MLA decode kernels in deepseek-v4-flash/nvfp4/ assume a 64-dim rope tail
// (`ROPE_DIM 64`), so GLM does not use them.
//
// One block per (q_head, row), 8 warps (blockDim 256). The warps split the row's selection
// `sel_indices[row, 0..sel_width)` and gather each selected token through the block table,
// one token at a time, with a per-warp online softmax and a cross-warp merge.
//
// Launch contract:
//   * kv_lora_dim == GLM_KV_LORA_DIM (512).
//   * A selection entry of -1, or any index outside [0, seq_len), is skipped.
//   * Nothing deduplicates: an index listed twice is attended twice.
//   * A row with no valid index writes zeros; a row whose seq_len is 0 writes nothing
//     to O.







































#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define VEC_BF16 16
#define VEC_U32  8
#define NUM_WARPS 8

// 2026-09-25: The latent width, fixed at compile time: 32 lanes * VEC_BF16 (16) covers
// exactly 512. The host refuses a kv_lora_rank that differs (Glm5NextDsaConfig::validate,
// KERNEL_KV_LORA_DIM).
#define GLM_KV_LORA_DIM 512

#define DSA_INVALID (-1)

__device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}


// 2026-09-25: Decode this lane's VEC_BF16 FP8 bytes of one cache token, times `scale`.
__device__ __forceinline__ void load_kv_fp8(
    const unsigned char* __restrict__ token_base,
    unsigned int lane_offset,
    float scale,
    float* __restrict__ out
) {
    const unsigned char* p = token_base + lane_offset;
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++)
        out[i] = fp8e4m3_to_f32((__nv_fp8_storage_t)p[i]) * scale;
}

extern "C" __global__ void glm5next_dsa_mla_decode_fp8(
    const __nv_bfloat16* __restrict__ Q,           // 2026-09-25: [rows, num_q_heads, kv_lora_dim] bf16
    const unsigned char* __restrict__ K_cache,     // 2026-09-25: FP8 latent cache
    const unsigned char* __restrict__ V_cache,     // 2026-09-25: the caller passes the K buffer
    __nv_bfloat16* __restrict__ O,                 // 2026-09-25: [rows, num_q_heads, kv_lora_dim] bf16
    const int* __restrict__ block_tables,          // 2026-09-25: row r at r * max_blocks_per_seq
    const int* __restrict__ seq_lens,              // 2026-09-25: [rows]
    const int* __restrict__ sel_indices,           // 2026-09-25: [rows, sel_width] i32, -1 = unused
    const unsigned int sel_width,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int kv_lora_dim,                // 2026-09-25: latent width; a token is num_kv_heads * kv_lora_dim bytes
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned long long cache_stride_bytes
) {
    const unsigned int q_head  = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int lane_offset = lane_id * VEC_BF16;

    // 2026-09-25: No rope term in the token stride (NoPE).
    const unsigned int token_stride = num_kv_heads * kv_lora_dim;

    const int* my_block_table = block_tables + (size_t)seq_idx * max_blocks_per_seq;
    const int* my_sel         = sel_indices  + (size_t)seq_idx * sel_width;

    // 2026-09-25: Q and O are [rows, num_q_heads, kv_lora_dim], and blockIdx.y is the row.
    // The layer passes all verify rows of a step as rows of one launch
    // (glm5next_dsa/layer.rs `attend_rows`).

    const unsigned long long row_off = (unsigned long long)seq_idx * num_q_heads * kv_lora_dim;


    const unsigned int* q32 =
        (const unsigned int*)(Q + row_off + (unsigned long long)q_head * kv_lora_dim + lane_offset);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unsigned int v = q32[i];
        q_reg[2*i]     = __bfloat162float(__ushort_as_bfloat16((unsigned short)(v & 0xFFFF)));
        q_reg[2*i + 1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(v >> 16)));
    }

    // 2026-09-25: Each warp takes a contiguous chunk of the selection row. A warp with no
    // valid index ends with l == 0 and contributes nothing to the merge.
    const unsigned int chunk = (sel_width + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int j     = warp_id * chunk;
    unsigned int j_end = j + chunk;
    if (j_end > sel_width) j_end = sel_width;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    for (; j < j_end; j++) {
        const int t = my_sel[j];
        // 2026-09-25: Skipped rather than clamped: a clamped index would attend a real but
        // wrong token.
        if (t == DSA_INVALID || t < 0 || (unsigned int)t >= seq_len) continue;

        const unsigned int logical_block = (unsigned int)t / block_size;
        const unsigned int p             = (unsigned int)t % block_size;
        const unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned char* k_tok =
            K_cache + (unsigned long long)physical_block * cache_stride_bytes + p * token_stride;
        const unsigned char* v_tok =
            V_cache + (unsigned long long)physical_block * cache_stride_bytes + p * token_stride;

        float k_tmp[VEC_BF16];
        load_kv_fp8(k_tok, lane_offset, k_scale, k_tmp);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            if (lane_offset + i < kv_lora_dim) dot += q_reg[i] * k_tmp[i];
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);

        const float score   = dot * inv_sqrt_d;
        const float m_new   = fmaxf(m, score);
        const float exp_old = __expf(m - m_new);
        const float exp_new = __expf(score - m_new);
        l = l * exp_old + exp_new;

        // 2026-09-25: Absorbed MLA: the layer passes one pool as both K and V and one scale
        // as both scales (glm5next_dsa/layer.rs `attend_rows`), so V is the K values just
        // decoded, bit for bit. The copy saves the second load; `__restrict__` on both
        // pointers stops the compiler from merging the loads itself. Distinct K/V buffers or
        // scales take the second load. Both operands are kernel arguments, so the branch is
        // uniform.











        const bool same_kv = (K_cache == V_cache) && (k_scale == v_scale);
        float v_tmp[VEC_BF16];
        if (same_kv) {
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) v_tmp[i] = k_tmp[i];
        } else {
            load_kv_fp8(v_tok, lane_offset, v_scale, v_tmp);
        }

        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            o_reg[i] = o_reg[i] * exp_old + exp_new * v_tmp[i];
        m = m_new;
    }

    // 2026-09-25: Cross-warp merge of (m, l, o); no attention-sink term.
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][GLM_KV_LORA_DIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++)
        if (lane_offset + i < GLM_KV_LORA_DIM) smem_o[warp_id][lane_offset + i] = o_reg[i];
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            const unsigned int other = warp_id + stride;
            const float lw = smem_l[other];
            if (lw > 0.0f) {
                const float mw     = smem_m[other];
                const float my_m   = smem_m[warp_id];
                const float my_l   = smem_l[warp_id];
                const float m_new  = fmaxf(my_m, mw);
                const float sc_me  = __expf(my_m - m_new);
                const float sc_w   = __expf(mw - m_new);
                smem_l[warp_id] = my_l * sc_me + lw * sc_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < GLM_KV_LORA_DIM; i++)
                    smem_o[warp_id][i] = smem_o[warp_id][i] * sc_me + smem_o[other][i] * sc_w;
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        const float final_l = smem_l[0];
        // 2026-09-25: final_l == 0 means no valid index in the row: write zeros rather than
        // divide by zero.
        const float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 =
            (unsigned int*)(O + row_off + (unsigned long long)q_head * kv_lora_dim + lane_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            const float v0 = smem_o[0][lane_offset + 2*i]     * inv_l;
            const float v1 = smem_o[0][lane_offset + 2*i + 1] * inv_l;
            const unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            const unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// 2026-10-01: Head-grouped variant: entries `glm5next_dsa_mla_decode_fp8_hg2`, `_hg4` and
// `_hg8`, opt-in through METRALE_GLM_DSA_MLA_HEADGROUP (glm5next_dsa/attend.rs); the kernel
// above stays the default. One block per (G consecutive q_heads, row), grid
// [num_q_heads / G, rows], blockDim 256. Every head-block of a row gathers the same keys, so a
// warp here gathers and decodes each selected token once and runs the G heads' updates on it.
//
// 2026-10-01: Byte-identical to the kernel above for every (row, head), argued against what
// one output element depends on:
//   (a) the key split: the same chunk = ceil(sel_width / 8), warp w owning
//       [w * chunk, min((w + 1) * chunk, sel_width));
//   (b) the online-softmax order: each warp walks its valid keys in increasing j, one head's
//       m/l/o updated per key in that order. Only the integer work (index read, validity,
//       block-table read, address) runs 32 keys at a time, one key per lane, and loads are
//       issued one key ahead of the math;
//   (c) the per-lane dot: lane L still owns dims 16L..16L+15 of q and of the token, summed
//       i = 0..15 with the same guard, mul then add (--fmad=false, KERNEL.toml), then the same
//       xor butterfly 16, 8, 4, 2, 1. The token bytes are decoded by fp8e4m3_to_f32 times
//       k_scale per byte, as load_kv_fp8 does; only the load width differs (one 16-byte load
//       when every address term is a multiple of 16, checked per launch, else byte loads);
//   (d) the merge: per head, the same 4/2/1 tree with the `lw > 0` guard, __expf,
//       1 / final_l and bf16 store. The tree's element updates are independent, so they are
//       spread over the warp's lanes instead of repeated by all 32.
// Which block or warp computes a given (row, head, slice) does not enter any of these.
//
// 2026-10-01: q for the G heads sits in registers (G * 16 floats per lane) for G = 2 and 4.
// For G = 8 that and o (8 * 16 floats) would not fit 255 registers, so q is held as floats in
// smem_o, which is free until the merge, in a swizzled layout that keeps each lane's 16-byte
// reads in distinct banks. bf16 -> f32 is exact, so where q lives does not change its value.

// 2026-10-01: 16 FP8 bytes at `p`: one 16-byte load when `aligned`, else byte loads. Either
// way word k holds bytes 4k..4k+3, low byte first.
__device__ __forceinline__ uint4 load_fp8x16(const unsigned char* __restrict__ p, bool aligned) {
    if (aligned) return *reinterpret_cast<const uint4*>(p);
    unsigned int w[4];
    #pragma unroll
    for (int k = 0; k < 4; k++)
        w[k] = (unsigned int)p[4 * k] | ((unsigned int)p[4 * k + 1] << 8)
             | ((unsigned int)p[4 * k + 2] << 16) | ((unsigned int)p[4 * k + 3] << 24);
    return make_uint4(w[0], w[1], w[2], w[3]);
}

// 2026-10-01: load_kv_fp8 on bytes already loaded: byte i decoded, times `scale`.
__device__ __forceinline__ void decode_fp8x16(uint4 raw, float scale, float* __restrict__ out) {
    const unsigned int w[4] = {raw.x, raw.y, raw.z, raw.w};
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++)
        out[i] = fp8e4m3_to_f32((__nv_fp8_storage_t)((w[i >> 2] >> (8 * (i & 3))) & 0xFFu)) * scale;
}

// 2026-10-01: Float4 slot of q[g][16L + 4c .. +3] when q lives in shared memory (Q_SMEM).
// The rotation by L >> 1 puts the 8 lanes of a 16-byte-load phase in 8 distinct bank groups.
__device__ __forceinline__ unsigned int q_smem_slot(int g, unsigned int lane, int c) {
    return (unsigned int)g * (GLM_KV_LORA_DIM / 4) + 4u * lane + (((unsigned int)c + (lane >> 1)) & 3u);
}

// 2026-10-01: One warp's slice [j, j_end) for G heads. SAME_KV is the kernel above's
// `same_kv` branch, hoisted out of the loop.
// 2026-10-05: `key_stride` walks the keys j, j + key_stride, j + 2 * key_stride, ... below
// j_end: lane k of a window resolves key j0 + k * key_stride. The head-grouped kernels pass 1,
// which is the contiguous walk exactly (integer index math only; every float operation and its
// order are unchanged). The split kernel passes num_splits * NUM_WARPS (see dsa_mla_decode_hg).
template <int G, bool Q_SMEM, bool SAME_KV>
__device__ __forceinline__ void dsa_hg_slice(
    const float (&q_reg)[G][VEC_BF16],
    const float4* __restrict__ q_s,
    float (&m)[G],
    float (&l)[G],
    float (&o_reg)[G][VEC_BF16],
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    const int* __restrict__ my_block_table,
    const int* __restrict__ my_sel,
    unsigned int j,
    const unsigned int j_end,
    const unsigned int seq_len,
    const unsigned int block_size,
    const unsigned int token_stride,
    const unsigned int kv_lora_dim,
    const unsigned int lane_id,
    const unsigned int lane_offset,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned long long cache_stride_bytes,
    const bool aligned,
    const unsigned int key_stride
) {
    for (unsigned int j0 = j; j0 < j_end; j0 += WARP_SIZE * key_stride) {
        // 2026-10-01: Lane k resolves key j0 + k: the same validity test, block-table read and
        // byte offset the kernel above computes per key.
        // 2026-10-05: Key j0 + k * key_stride (j0 + k at key_stride 1).
        const unsigned int jl = j0 + lane_id * key_stride;
        unsigned long long tok_off = 0ull;
        bool ok = false;
        if (jl < j_end) {
            const int t = my_sel[jl];
            if (!(t == DSA_INVALID || t < 0 || (unsigned int)t >= seq_len)) {
                const unsigned int logical_block  = (unsigned int)t / block_size;
                const unsigned int p              = (unsigned int)t % block_size;
                const unsigned int physical_block = (unsigned int)my_block_table[logical_block];
                tok_off = (unsigned long long)physical_block * cache_stride_bytes + p * token_stride;
                ok = true;
            }
        }
        // 2026-10-01: Valid keys of the window, consumed lowest bit first (increasing j).
        unsigned int pending = __ballot_sync(0xffffffffu, ok);
        if (pending == 0u) continue;

        int src = __ffs((int)pending) - 1;
        pending &= pending - 1u;
        unsigned long long off = __shfl_sync(0xffffffffu, tok_off, src);
        uint4 k_raw = load_fp8x16(K_cache + off + lane_offset, aligned);
        uint4 v_raw = make_uint4(0u, 0u, 0u, 0u);
        if constexpr (!SAME_KV) v_raw = load_fp8x16(V_cache + off + lane_offset, aligned);

        for (;;) {
            const uint4 k_cur = k_raw;
            const uint4 v_cur = v_raw;
            // 2026-10-01: The next valid key's bytes load before this key's math.
            const bool more = pending != 0u;
            if (more) {
                src = __ffs((int)pending) - 1;
                pending &= pending - 1u;
                off = __shfl_sync(0xffffffffu, tok_off, src);
                k_raw = load_fp8x16(K_cache + off + lane_offset, aligned);
                if constexpr (!SAME_KV) v_raw = load_fp8x16(V_cache + off + lane_offset, aligned);
            }

            float k_tmp[VEC_BF16];
            decode_fp8x16(k_cur, k_scale, k_tmp);

            float score[G];
            #pragma unroll
            for (int g = 0; g < G; g++) {
                float dot = 0.0f;
                if constexpr (Q_SMEM) {
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        const float4 qa = q_s[q_smem_slot(g, lane_id, c)];
                        const float qc[4] = {qa.x, qa.y, qa.z, qa.w};
                        #pragma unroll
                        for (int e = 0; e < 4; e++)
                            if (lane_offset + (unsigned int)(4 * c + e) < kv_lora_dim)
                                dot += qc[e] * k_tmp[4 * c + e];
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++)
                        if (lane_offset + i < kv_lora_dim) dot += q_reg[g][i] * k_tmp[i];
                }
                #pragma unroll
                for (int sh = WARP_SIZE / 2; sh > 0; sh >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, sh);
                score[g] = dot * inv_sqrt_d;
            }

            // 2026-10-01: SAME_KV: V is the K values just decoded, as in the kernel above.
            float v_own[VEC_BF16];
            if constexpr (!SAME_KV) decode_fp8x16(v_cur, v_scale, v_own);

            #pragma unroll
            for (int g = 0; g < G; g++) {
                const float m_new   = fmaxf(m[g], score[g]);
                const float exp_old = __expf(m[g] - m_new);
                const float exp_new = __expf(score[g] - m_new);
                l[g] = l[g] * exp_old + exp_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    const float v_i = SAME_KV ? k_tmp[i] : v_own[i];
                    o_reg[g][i] = o_reg[g][i] * exp_old + exp_new * v_i;
                }
                m[g] = m_new;
            }

            if (!more) break;
        }
    }
}

// 2026-10-05: SPLIT (entry `glm5next_dsa_mla_decode_fp8_hg8_split`, opt-in through
// METRALE_GLM_DSA_MLA_SPLIT, glm5next_dsa/attend.rs) is the same block with the row's
// selection shared by num_splits blocks along grid z, grid [num_q_heads / G, rows, num_splits].
// The grid of the plain kernel is [num_q_heads / 8, rows] = 12 blocks at 32 heads and 3 verify
// rows on a 48-SM GB10, and each warp walks up to ceil(sel_width / 8) keys one after another,
// so the launch is bound by that serial chain, not by bandwidth.
//   * Keys: the T = num_splits * NUM_WARPS warps of a (head group, row) take the selection
//     interleaved, warp t = split * NUM_WARPS + warp_id owning j = t, t + T, t + 2T, ... The
//     selection is `-1`-padded past the row's valid prefix (dsa_expand_selection), so a
//     contiguous split would leave the valid keys on the first blocks; interleaved, every warp
//     gets about valid / T of them wherever they sit.
//   * Each block runs the same per-warp online softmax and the same cross-warp tree, then
//     writes, per head, its UNNORMALISED partial instead of O: o[512] and (m, l), FP32, to
//     `partials`. Layout, P = rows * num_q_heads * num_splits partials, partial
//     p = (row * num_q_heads + head) * num_splits + split:
//       o  at partials[p * 512 .. p * 512 + 512)
//       ml at partials[P * 512 + 2p], [P * 512 + 2p + 1]   (m, l)
//     A block whose keys are all invalid writes l = 0 (and o = 0); the merge skips it.
//   * glm5next_dsa_mla_split_merge (below) combines a (row, head)'s num_splits partials with
//     the log-sum-exp rescale and writes BF16 O under this file's launch contract.
// Not byte-identical to the plain kernel: the keys reach the softmax in another order and the
// partials merge in another tree. FP32 throughout, --fmad=false as for the whole module.
template <int G, bool Q_SMEM, bool SPLIT>
__device__ __forceinline__ void dsa_mla_decode_hg(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int* __restrict__ sel_indices,
    const unsigned int sel_width,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int kv_lora_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned long long cache_stride_bytes,
    float* __restrict__ partials,
    const unsigned int num_splits
) {
    // 2026-10-01: Q_SMEM keeps G * 512 floats of q in smem_o, which holds NUM_WARPS * 512.
    static_assert(G >= 1 && G <= NUM_WARPS, "q for G heads must fit smem_o");

    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ __align__(16) float smem_o[NUM_WARPS][GLM_KV_LORA_DIM];

    const unsigned int head0   = blockIdx.x * G;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    // 2026-10-01: The host launches only when G divides num_q_heads.
    if (head0 + G > num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int lane_offset  = lane_id * VEC_BF16;
    const unsigned int token_stride = num_kv_heads * kv_lora_dim;

    const int* my_block_table = block_tables + (size_t)seq_idx * max_blocks_per_seq;
    const int* my_sel         = sel_indices  + (size_t)seq_idx * sel_width;

    const unsigned long long row_off = (unsigned long long)seq_idx * num_q_heads * kv_lora_dim;

    // 2026-10-01: q, converted from bf16 exactly as the kernel above does.
    float q_reg[G][VEC_BF16];
    float4* q_s = reinterpret_cast<float4*>(&smem_o[0][0]);
    if constexpr (Q_SMEM) {
        for (unsigned int g = warp_id; g < (unsigned int)G; g += NUM_WARPS) {
            const unsigned int* q32 = (const unsigned int*)(
                Q + row_off + (unsigned long long)(head0 + g) * kv_lora_dim + lane_offset);
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const unsigned int v0 = q32[2 * c];
                const unsigned int v1 = q32[2 * c + 1];
                q_s[q_smem_slot((int)g, lane_id, c)] = make_float4(
                    __bfloat162float(__ushort_as_bfloat16((unsigned short)(v0 & 0xFFFF))),
                    __bfloat162float(__ushort_as_bfloat16((unsigned short)(v0 >> 16))),
                    __bfloat162float(__ushort_as_bfloat16((unsigned short)(v1 & 0xFFFF))),
                    __bfloat162float(__ushort_as_bfloat16((unsigned short)(v1 >> 16))));
            }
        }
        __syncthreads();
    } else {
        #pragma unroll
        for (int g = 0; g < G; g++) {
            const unsigned int* q32 = (const unsigned int*)(
                Q + row_off + (unsigned long long)(head0 + g) * kv_lora_dim + lane_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                unsigned int v = q32[i];
                q_reg[g][2*i]     = __bfloat162float(__ushort_as_bfloat16((unsigned short)(v & 0xFFFF)));
                q_reg[g][2*i + 1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(v >> 16)));
            }
        }
    }

    // 2026-10-01: The kernel above's split, unchanged.
    // 2026-10-05: SPLIT: this warp's interleaved keys (see above the template).
    unsigned int j, j_end, key_stride;
    if constexpr (SPLIT) {
        key_stride = num_splits * NUM_WARPS;
        j      = blockIdx.z * NUM_WARPS + warp_id;
        j_end  = sel_width;
    } else {
        const unsigned int chunk = (sel_width + NUM_WARPS - 1) / NUM_WARPS;
        j      = warp_id * chunk;
        j_end  = j + chunk;
        if (j_end > sel_width) j_end = sel_width;
        key_stride = 1u;
    }

    float m[G];
    float l[G];
    float o_reg[G][VEC_BF16];
    #pragma unroll
    for (int g = 0; g < G; g++) {
        m[g] = -1e30f;
        l[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[g][i] = 0.0f;
    }

    // 2026-10-01: Both operands are kernel arguments, so both tests are uniform. `aligned`
    // covers every term of a token address: base, physical_block * cache_stride_bytes,
    // p * token_stride and lane_offset (16 * lane).
    const bool same_kv = (K_cache == V_cache) && (k_scale == v_scale);
    const bool aligned =
        ((((unsigned long long)K_cache) | ((unsigned long long)V_cache) | cache_stride_bytes
          | (unsigned long long)token_stride) & 15ull) == 0ull;

    if (same_kv) {
        dsa_hg_slice<G, Q_SMEM, true>(q_reg, q_s, m, l, o_reg, K_cache, V_cache, my_block_table,
            my_sel, j, j_end, seq_len, block_size, token_stride, kv_lora_dim, lane_id,
            lane_offset, inv_sqrt_d, k_scale, v_scale, cache_stride_bytes, aligned, key_stride);
    } else {
        dsa_hg_slice<G, Q_SMEM, false>(q_reg, q_s, m, l, o_reg, K_cache, V_cache, my_block_table,
            my_sel, j, j_end, seq_len, block_size, token_stride, kv_lora_dim, lane_id,
            lane_offset, inv_sqrt_d, k_scale, v_scale, cache_stride_bytes, aligned, key_stride);
    }

    // 2026-10-01: Per head, the kernel above's cross-warp merge and store on one 16 KB
    // buffer. The barrier at the top of each head orders it after the q reads (Q_SMEM) and
    // after warp 0's store of the previous head.
    float4* o_s4 = reinterpret_cast<float4*>(&smem_o[0][0]);
    #pragma unroll
    for (int g = 0; g < G; g++) {
        __syncthreads();
        if (lane_id == 0) {
            smem_m[warp_id] = m[g];
            smem_l[warp_id] = l[g];
        }
        #pragma unroll
        for (int c = 0; c < 4; c++)
            o_s4[warp_id * (GLM_KV_LORA_DIM / 4) + lane_id * 4 + c] = make_float4(
                o_reg[g][4 * c], o_reg[g][4 * c + 1], o_reg[g][4 * c + 2], o_reg[g][4 * c + 3]);
        __syncthreads();

        #pragma unroll
        for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
            if (warp_id < (unsigned int)stride) {
                const unsigned int other = warp_id + stride;
                const float lw = smem_l[other];
                if (lw > 0.0f) {
                    const float mw     = smem_m[other];
                    const float my_m   = smem_m[warp_id];
                    const float my_l   = smem_l[warp_id];
                    const float m_new  = fmaxf(my_m, mw);
                    const float sc_me  = __expf(my_m - m_new);
                    const float sc_w   = __expf(mw - m_new);
                    const float l_new  = my_l * sc_me + lw * sc_w;
                    // 2026-10-01: Element e = i * 32 + lane: the same update the kernel
                    // above repeats on all 32 lanes, done once, in distinct banks.
                    #pragma unroll
                    for (int i = 0; i < GLM_KV_LORA_DIM / WARP_SIZE; i++) {
                        const unsigned int e = (unsigned int)i * WARP_SIZE + lane_id;
                        smem_o[warp_id][e] = smem_o[warp_id][e] * sc_me + smem_o[other][e] * sc_w;
                    }
                    // 2026-10-01: Every lane has read smem_m/l[warp_id] before lane 0 writes.
                    __syncwarp();
                    if (lane_id == 0) {
                        smem_l[warp_id] = l_new;
                        smem_m[warp_id] = m_new;
                    }
                }
            }
            __syncthreads();
        }

        if constexpr (SPLIT) {
            // 2026-10-05: This block's partial for head head0 + g, unnormalised (layout above
            // the template). The barrier at the top of the next head orders warp 0's reads of
            // smem_o[0] before the next head overwrites it, as for the store below.
            if (warp_id == 0) {
                const unsigned long long n_part =
                    (unsigned long long)gridDim.y * num_q_heads * num_splits;
                const unsigned long long p =
                    ((unsigned long long)seq_idx * num_q_heads + head0 + g) * num_splits
                    + blockIdx.z;
                float4* po =
                    reinterpret_cast<float4*>(partials + p * GLM_KV_LORA_DIM + lane_offset);
                #pragma unroll
                for (int c = 0; c < 4; c++) po[c] = o_s4[lane_id * 4 + c];
                if (lane_id == 0)
                    reinterpret_cast<float2*>(partials + n_part * GLM_KV_LORA_DIM)[p] =
                        make_float2(smem_m[0], smem_l[0]);
            }
        } else {
            if (warp_id == 0) {
                const float final_l = smem_l[0];
                const float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
                unsigned int* o32 = (unsigned int*)(
                    O + row_off + (unsigned long long)(head0 + g) * kv_lora_dim + lane_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    const float v0 = smem_o[0][lane_offset + 2*i]     * inv_l;
                    const float v1 = smem_o[0][lane_offset + 2*i + 1] * inv_l;
                    const unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
                    const unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
                    o32[i] = lo | (hi << 16);
                }
            }
        }
    }
}

// 2026-10-01: The entry points, one per G, with the kernel above's argument list.
// 2026-10-05: SPLIT false: no partials, num_splits 1 (both unread).
#define DSA_MLA_HG_ENTRY(NAME, G, Q_SMEM, MIN_BLOCKS)                                         \
    extern "C" __global__ void __launch_bounds__(NUM_WARPS * WARP_SIZE, MIN_BLOCKS) NAME(     \
        const __nv_bfloat16* __restrict__ Q, const unsigned char* __restrict__ K_cache,      \
        const unsigned char* __restrict__ V_cache, __nv_bfloat16* __restrict__ O,            \
        const int* __restrict__ block_tables, const int* __restrict__ seq_lens,              \
        const int* __restrict__ sel_indices, const unsigned int sel_width,                   \
        const unsigned int max_blocks_per_seq, const unsigned int num_q_heads,               \
        const unsigned int num_kv_heads, const unsigned int kv_lora_dim,                     \
        const unsigned int block_size, const float inv_sqrt_d, const float k_scale,          \
        const float v_scale, const unsigned long long cache_stride_bytes) {                  \
        dsa_mla_decode_hg<G, Q_SMEM, false>(Q, K_cache, V_cache, O, block_tables, seq_lens,   \
            sel_indices, sel_width, max_blocks_per_seq, num_q_heads, num_kv_heads,           \
            kv_lora_dim, block_size, inv_sqrt_d, k_scale, v_scale, cache_stride_bytes,       \
            nullptr, 1u);                                                                     \
    }

// 2026-10-01: G = 2 is capped at 128 registers for two blocks per SM; G = 4 and 8 take one.
DSA_MLA_HG_ENTRY(glm5next_dsa_mla_decode_fp8_hg2, 2, false, 2)
DSA_MLA_HG_ENTRY(glm5next_dsa_mla_decode_fp8_hg4, 4, false, 1)
DSA_MLA_HG_ENTRY(glm5next_dsa_mla_decode_fp8_hg8, 8, true, 1)

// 2026-10-05: The split entry (METRALE_GLM_DSA_MLA_SPLIT): G = 8 with q in shared memory, as
// `_hg8`, one block per (8 heads, row, split), grid [num_q_heads / 8, rows, num_splits]. The
// head-grouped argument list with O replaced by `partials` and `num_splits` appended; the host
// sizes `partials` for rows * num_q_heads * num_splits * (512 + 2) floats.
extern "C" __global__ void __launch_bounds__(NUM_WARPS * WARP_SIZE, 1)
glm5next_dsa_mla_decode_fp8_hg8_split(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    float* __restrict__ partials,                  // 2026-10-05: layout above dsa_mla_decode_hg
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int* __restrict__ sel_indices,
    const unsigned int sel_width,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int kv_lora_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned long long cache_stride_bytes,
    const unsigned int num_splits                  // 2026-10-05: == gridDim.z
) {
    dsa_mla_decode_hg<8, true, true>(Q, K_cache, V_cache, nullptr, block_tables, seq_lens,
        sel_indices, sel_width, max_blocks_per_seq, num_q_heads, num_kv_heads, kv_lora_dim,
        block_size, inv_sqrt_d, k_scale, v_scale, cache_stride_bytes, partials, num_splits);
}

// 2026-10-05: Threads of the merge: four latent dims each.
#define SPLIT_MERGE_THREADS (GLM_KV_LORA_DIM / 4)

// 2026-10-05: Merge of the split partials: one block per (head, row), grid
// [num_q_heads, rows], blockDim SPLIT_MERGE_THREADS (128), launched on the split launch's
// stream right after it with the same rows, heads and num_splits (the partial layout depends on
// all three). For one (row, head), over the partials s with l_s > 0:
//   M = max m_s,  L = sum l_s * exp(m_s - M),  o = sum o_s * exp(m_s - M),  O = bf16(o * (1 / L))
// with __expf, and 1 / L applied as the plain kernel applies 1 / final_l. The launch contract:
//   * seq_len 0: the split blocks wrote nothing and this writes nothing to O;
//   * no valid index in the row: every l_s is 0, so L = 0, 1 / L is taken as 0 and o stays
//     0.0f, and the row is written as +0.0 bf16, the bits the plain kernel writes;
//   * skipped and duplicated indices were handled by the split blocks exactly as by the plain
//     kernel (dsa_hg_slice).
extern "C" __global__ void __launch_bounds__(SPLIT_MERGE_THREADS)
glm5next_dsa_mla_split_merge(
    const float* __restrict__ partials,
    __nv_bfloat16* __restrict__ O,                 // 2026-10-05: [rows, num_q_heads, kv_lora_dim] bf16
    const int* __restrict__ seq_lens,              // 2026-10-05: [rows]
    const unsigned int num_q_heads,
    const unsigned int kv_lora_dim,
    const unsigned int num_splits
) {
    const unsigned int head    = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid     = threadIdx.x;

    if (head >= num_q_heads) return;
    if (seq_lens[seq_idx] == 0) return;

    const unsigned long long n_part = (unsigned long long)gridDim.y * num_q_heads * num_splits;
    const unsigned long long p0 = ((unsigned long long)seq_idx * num_q_heads + head) * num_splits;
    const float2* ml = reinterpret_cast<const float2*>(partials + n_part * GLM_KV_LORA_DIM);

    float m_max = -1e30f;
    for (unsigned int s = 0; s < num_splits; s++) {
        const float2 v = ml[p0 + s];
        if (v.y > 0.0f) m_max = fmaxf(m_max, v.x);
    }

    float L = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int s = 0; s < num_splits; s++) {
        const float2 v = ml[p0 + s];
        if (!(v.y > 0.0f)) continue;
        const float w = __expf(v.x - m_max);
        L = L + v.y * w;
        const float4 o =
            reinterpret_cast<const float4*>(partials + (p0 + s) * GLM_KV_LORA_DIM)[tid];
        acc[0] = acc[0] + o.x * w;
        acc[1] = acc[1] + o.y * w;
        acc[2] = acc[2] + o.z * w;
        acc[3] = acc[3] + o.w * w;
    }

    const float inv_l = (L > 0.0f) ? (1.0f / L) : 0.0f;
    unsigned int* o32 = (unsigned int*)(
        O + (unsigned long long)seq_idx * num_q_heads * kv_lora_dim
          + (unsigned long long)head * kv_lora_dim + 4u * tid);
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const unsigned int lo =
            (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc[2*i] * inv_l));
        const unsigned int hi =
            (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc[2*i + 1] * inv_l));
        o32[i] = lo | (hi << 16);
    }
}
