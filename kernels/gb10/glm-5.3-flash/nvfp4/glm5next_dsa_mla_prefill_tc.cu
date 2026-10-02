// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-01: GLM-5.3-Flash selected-index MLA attention for PREFILL rows on tensor cores
// (module `glm5next_dsa_mla_prefill_tc`, entry `glm5next_dsa_mla_prefill_tc_fp8`), opt-in through
// METRALE_GLM_MLA_PREFILL_TC=1 (glm5next_dsa/attend/prefill_tc.rs). The default path is
// `glm5next_dsa_mla_decode_fp8` (glm5next_dsa_mla_decode.cu), which runs the FP32 dot and the
// P*V update on CUDA cores, one block per (head or head group, row).
//
// Same inputs and output layout as `glm5next_dsa_mla_decode_fp8`: Q and O [rows, num_q_heads, 512]
// BF16, the paged FP8 E4M3 latent cache (NoPE: a token is the 512-wide latent, one KV head), the
// row's block table at row * max_blocks_per_seq, seq_lens[row], and the DSA selection
// sel_indices[row, 0..sel_width) with -1 holes. K and V are the same pool and the same scale
// (absorbed MLA); the host launches only then.
//
// Organization: one CTA of 8 warps per (32 consecutive heads, row), grid [num_q_heads / 32, rows].
// The 32 heads of a row share the row's selection, so the heads are the MMA M dimension and every
// selected token is gathered and dequantized once per 32 heads:
//   1. The row's valid indices (0 <= t < seq_len) are compacted into shared memory in selection
//      order (block-wide stable compaction), so key tiles are dense and a short row (an early
//      prefill position) runs only ceil(valid / 32) tiles.
//   2. Per tile of TC_NK (32) keys: each warp loads 4 keys' 512 FP8 bytes (16 B per lane, one
//      coalesced 512 B row per warp instruction), converts them to BF16 (exact: every E4M3 value
//      is a BF16 value; the scale is applied in FP32 later) and stores them XOR-swizzled.
//      The loads for tile t + 1 are issued right after tile t's tile is published, and the block
//      table reads for tile t + 2 right after them, so neither latency is on the critical path.
//   3. S = Q K^T: warp w takes heads (w & 1) * 16 and dims (w >> 1) * 128 of the 512, all 32 keys
//      (8 k-steps of bf16 m16n8k16, 4 n-tiles); the four dim-quarter partials go to shared memory
//      as FP32 and are summed in a fixed order, so the result does not depend on scheduling.
//   4. Online softmax in the exp2 domain, 8 threads per head: scale k_scale * inv_sqrt_d * log2 e,
//      running max and sum in FP32, P rounded to BF16 for the MMA (the sum uses the FP32 p).
//   5. O += P V with V = the same BF16 key tile read through ldmatrix.trans: warp w owns heads
//      (w & 1) * 16 and dims (w >> 1) * 128 of O, 64 FP32 accumulators per thread, rescaled by the
//      softmax correction before each tile's MMAs.
//   6. O * (v_scale / l) in BF16. A row with no valid index writes zeros; a row whose seq_len is 0
//      writes nothing (both as the decode kernel).
//
// Numerics differ from `glm5next_dsa_mla_decode_fp8` by construction (BF16 P, FP32 MMA
// accumulation order, exp2f instead of __expf, scales applied once outside the dot products); the
// GPU microtest `examples/dsa_mla_prefill_tc_microtest.rs` bounds the difference (cosine and max
// error) and model-level quality is gated separately. Written for this repository from the
// standard mma.sync / ldmatrix fragment layouts (PTX ISA), in the style of
// kernels/gb10/common/attn_prefill_fa128.cu; no third-party code is vendored.
//
// Owner: gb10 kernels (glm-5.3-flash).
// Invariants:
// - Block 256 threads; dynamic shared memory TC_SMEM_BYTES; grid [num_q_heads / TC_HEADS, rows].
// - kv_lora_dim == TC_D (512), one KV head, sel_width <= TC_MAX_SEL, and the cache base,
//   cache_stride_bytes, Q and O 16-byte aligned; the host checks every one and keeps the decode
//   kernel otherwise (glm5next_dsa/attend/prefill_tc.rs `prefill_tc_refusal`).
// - Indices outside [0, seq_len) are skipped, -1 included; duplicates are attended twice.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define TC_WARP_SIZE 32
#define TC_WARPS 8
#define TC_THREADS 256
// 2026-10-01: Heads per CTA (the MMA M dimension, two m16 tiles). Mirrored by
// MLA_PREFILL_TC_HEADS (glm5next_dsa/attend/prefill_tc.rs).
#define TC_HEADS 32
// 2026-10-01: The latent width; 64 chunks of 16 B per BF16 row.
#define TC_D 512
// 2026-10-01: Keys per tile.
#define TC_NK 32
// 2026-10-01: Widest selection row the compacted index list holds (production: 2051). Mirrored by
// MLA_PREFILL_TC_MAX_SEL.
#define TC_MAX_SEL 2560
// 2026-10-01: Row strides of the FP32 S partials and the BF16 P tile, padded so the partial
// stores and the P ldmatrix rows fall in distinct bank groups.
#define TC_SP_LD 40
#define TC_PS_LD 40

// 2026-10-01: Dynamic shared-memory layout, byte offsets.
#define TC_OFF_Q 0                                               // [32][512] BF16, swizzled
#define TC_OFF_K (TC_OFF_Q + TC_HEADS * TC_D * 2)                // [32][512] BF16, swizzled
#define TC_OFF_SP (TC_OFF_K + TC_NK * TC_D * 2)                  // [4][32][TC_SP_LD] FP32
#define TC_OFF_P (TC_OFF_SP + 4 * TC_HEADS * TC_SP_LD * 4)       // [32][TC_PS_LD] BF16
#define TC_OFF_SEL (TC_OFF_P + TC_HEADS * TC_PS_LD * 2)          // [TC_MAX_SEL] i32
#define TC_OFF_ALPHA (TC_OFF_SEL + TC_MAX_SEL * 4)               // [32] FP32
#define TC_OFF_LINV (TC_OFF_ALPHA + TC_HEADS * 4)                // [32] FP32
#define TC_OFF_CNT (TC_OFF_LINV + TC_HEADS * 4)                  // [8] i32
// 2026-10-01: The launch's dynamic shared memory, a plain integer so the host mirror
// (MLA_PREFILL_TC_SMEM_BYTES) can be checked against it; the static_assert ties it to the layout.
#define TC_SMEM_BYTES 99104
static_assert(TC_OFF_CNT + TC_WARPS * 4 == TC_SMEM_BYTES, "TC_SMEM_BYTES is the layout's size");
static_assert(TC_HEADS * TC_D * 2 / 16 % TC_THREADS == 0, "Q loads split evenly");
static_assert(TC_NK * TC_D / 16 == 4 * TC_THREADS, "each thread loads 4 key chunks per tile");

// 2026-10-01: Offset of a key that is past the compacted list: nothing is loaded for it.
#define TC_NO_KEY (~0ull)

__device__ __forceinline__ float tc_neg_inf() { return __int_as_float((int)0xff800000u); }

__device__ __forceinline__ unsigned tc_smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

// 2026-10-01: Byte offset of 16-byte chunk `ch` (0..63) of row `row` in a [rows][512] BF16 tile.
// The XOR with the row's low 3 bits puts the 8 rows of an ldmatrix phase (8 consecutive rows, one
// chunk column) in 8 distinct bank groups; it only permutes chunks within an aligned group of 8.
__device__ __forceinline__ unsigned tc_swz(unsigned row, unsigned ch) {
    return row * (TC_D * 2) + ((ch ^ (row & 7u)) << 4);
}

__device__ __forceinline__ void tc_ldsm_x4(unsigned a, unsigned& d0, unsigned& d1, unsigned& d2,
                                           unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a));
}

__device__ __forceinline__ void tc_ldsm_x4_t(unsigned a, unsigned& d0, unsigned& d1, unsigned& d2,
                                             unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a));
}

__device__ __forceinline__ void tc_mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ unsigned tc_pack_bf16(float lo, float hi) {
    const unsigned l = __bfloat16_as_ushort(__float2bfloat16(lo));
    const unsigned h = __bfloat16_as_ushort(__float2bfloat16(hi));
    return l | (h << 16);
}

// 2026-10-01: 16 FP8 E4M3 bytes (byte i = dim i) to 16 BF16 values, packed two per word with the
// lower dim in the low half (memory order). E4M3 -> FP16 -> FP32 -> BF16 is exact for every
// finite E4M3 code, so the tile holds the unscaled cache values.
__device__ __forceinline__ void tc_fp8x16_to_bf16(uint4 raw, unsigned (&out)[8]) {
    const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(
                (__nv_fp8x2_storage_t)((w[i] >> (16 * h)) & 0xFFFFu), __NV_E4M3);
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
            out[2 * i + h] = tc_pack_bf16(f.x, f.y);
        }
    }
}

// 2026-10-01: Byte offset in the cache of compacted key `j`, or TC_NO_KEY past the list. The same
// page arithmetic as the decode kernel: block table entry t / block_size, slot t % block_size.
__device__ __forceinline__ unsigned long long tc_key_off(
    const int* sel_s,
    unsigned j,
    unsigned n_valid,
    const int* __restrict__ block_table,
    unsigned block_size,
    unsigned long long cache_stride_bytes
) {
    if (j >= n_valid) return TC_NO_KEY;
    const unsigned t = (unsigned)sel_s[j];
    const unsigned physical_block = (unsigned)block_table[t / block_size];
    return (unsigned long long)physical_block * cache_stride_bytes
         + (unsigned long long)(t % block_size) * TC_D;
}

// 2026-10-01: This lane's 16 bytes of the key at `off`; zeros for TC_NO_KEY (a zero BF16 row,
// masked out of the softmax, so it adds exactly nothing).
__device__ __forceinline__ uint4 tc_load_key(const unsigned char* __restrict__ kv,
                                             unsigned long long off, unsigned lane) {
    if (off == TC_NO_KEY) return make_uint4(0u, 0u, 0u, 0u);
    return __ldg(reinterpret_cast<const uint4*>(kv + off) + lane);
}

extern "C" __global__ void __launch_bounds__(TC_THREADS, 1) glm5next_dsa_mla_prefill_tc_fp8(
    const __nv_bfloat16* __restrict__ Q,           // 2026-10-01: [rows, num_q_heads, 512] bf16
    const unsigned char* __restrict__ KV_cache,    // 2026-10-01: FP8 latent cache, K == V
    __nv_bfloat16* __restrict__ O,                 // 2026-10-01: [rows, num_q_heads, 512] bf16
    const int* __restrict__ block_tables,          // 2026-10-01: row r at r * max_blocks_per_seq
    const int* __restrict__ seq_lens,              // 2026-10-01: [rows]
    const int* __restrict__ sel_indices,           // 2026-10-01: [rows, sel_width] i32, -1 = unused
    const unsigned int sel_width,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float kv_scale,
    const unsigned long long cache_stride_bytes
) {
    extern __shared__ __align__(16) unsigned char tc_smem[];
    unsigned char* q_s = tc_smem + TC_OFF_Q;
    unsigned char* k_s = tc_smem + TC_OFF_K;
    float* sp_s        = reinterpret_cast<float*>(tc_smem + TC_OFF_SP);
    unsigned char* p_s = tc_smem + TC_OFF_P;
    int* sel_s         = reinterpret_cast<int*>(tc_smem + TC_OFF_SEL);
    float* alpha_s     = reinterpret_cast<float*>(tc_smem + TC_OFF_ALPHA);
    float* linv_s      = reinterpret_cast<float*>(tc_smem + TC_OFF_LINV);
    int* cnt_s         = reinterpret_cast<int*>(tc_smem + TC_OFF_CNT);

    const unsigned head0 = blockIdx.x * TC_HEADS;
    const unsigned row   = blockIdx.y;
    const unsigned tid   = threadIdx.x;
    const unsigned warp  = tid / TC_WARP_SIZE;
    const unsigned lane  = tid % TC_WARP_SIZE;

    // 2026-10-01: The host launches only when TC_HEADS divides num_q_heads.
    if (head0 + TC_HEADS > num_q_heads) return;

    const unsigned seq_len = (unsigned)seq_lens[row];
    if (seq_len == 0) return;

    const int* my_block_table = block_tables + (size_t)row * max_blocks_per_seq;
    const int* my_sel         = sel_indices  + (size_t)row * sel_width;
    const unsigned long long q_off = ((unsigned long long)row * num_q_heads + head0) * TC_D;

    // 2026-10-01: The 32 heads' q rows (32 KB, contiguous in Q) into the swizzled Q tile.
    {
        const uint4* q4 = reinterpret_cast<const uint4*>(Q + q_off);
        #pragma unroll
        for (int i = 0; i < TC_HEADS * TC_D * 2 / 16 / TC_THREADS; i++) {
            const unsigned c = tid + (unsigned)i * TC_THREADS;
            *reinterpret_cast<uint4*>(q_s + tc_swz(c >> 6, c & 63u)) = q4[c];
        }
    }

    // 2026-10-01: Stable compaction of the valid indices, 256 at a time: a warp's ballot gives
    // each lane its rank among the warp's valid entries, and the warp counts give the warp's
    // base. The barrier at the end of each pass also publishes the Q tile.
    unsigned n_valid = 0;
    for (unsigned j0 = 0; j0 < sel_width; j0 += TC_THREADS) {
        const unsigned j = j0 + tid;
        int t = -1;
        bool ok = false;
        if (j < sel_width) {
            t = my_sel[j];
            ok = !(t < 0 || (unsigned)t >= seq_len);
        }
        const unsigned ball = __ballot_sync(0xffffffffu, ok);
        if (lane == 0) cnt_s[warp] = __popc(ball);
        __syncthreads();
        unsigned base = n_valid;
        unsigned total = 0;
        #pragma unroll
        for (unsigned w = 0; w < TC_WARPS; w++) {
            const unsigned c = (unsigned)cnt_s[w];
            if (w < warp) base += c;
            total += c;
        }
        if (ok) sel_s[base + __popc(ball & ((1u << lane) - 1u))] = t;
        n_valid += total;
        __syncthreads();
    }

    const unsigned n_tiles = (n_valid + TC_NK - 1) / TC_NK;

    // 2026-10-01: This thread loads 16 bytes (lane) of keys warp + 8 i, i = 0..3, of every tile.
    uint4 raw[4];
    unsigned long long off_nxt[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned kk = warp + 8u * (unsigned)i;
        raw[i] = tc_load_key(KV_cache,
                             tc_key_off(sel_s, kk, n_valid, my_block_table, block_size,
                                        cache_stride_bytes),
                             lane);
        off_nxt[i] = tc_key_off(sel_s, TC_NK + kk, n_valid, my_block_table, block_size,
                                cache_stride_bytes);
    }

    // 2026-10-01: Warp roles. QK: heads mt * 16.., dims qd * 128.. (a quarter of the dot).
    // PV: heads mt * 16.., output dims qd * 128... Softmax: head sm_row, keys sm_c4..sm_c4 + 3.
    const unsigned mt     = warp & 1u;
    const unsigned qd     = warp >> 1;
    const unsigned g      = lane >> 2;
    const unsigned t4     = lane & 3u;
    const unsigned sm_row = tid >> 3;
    const unsigned sm_c4  = (tid & 7u) * 4u;
    // 2026-10-01: ldmatrix lane addressing: A-style (rows +8 on lanes 8-15/24-31, chunk +1 on
    // lanes 16-31) and the non-transposed B pair (keys +8 on lanes 16-31, chunk +1 on lanes 8-15).
    const unsigned a_row = (lane & 7u) + ((lane >> 3) & 1u) * 8u;
    const unsigned a_ch  = lane >> 4;
    const unsigned b_row = (lane & 7u) + ((lane >> 4) & 1u) * 8u;
    const unsigned b_ch  = (lane >> 3) & 1u;

    const float qk_scale = kv_scale * inv_sqrt_d * 1.4426950408889634f;
    float m_run = tc_neg_inf();
    float l_run = 0.0f;

    float o_acc[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) o_acc[nt][e] = 0.0f;
    }

    for (unsigned tile = 0; tile < n_tiles; tile++) {
        // 2026-10-01: (1) This tile's bytes, loaded one tile ago, into the BF16 key tile.
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned pk[8];
            tc_fp8x16_to_bf16(raw[i], pk);
            const unsigned kk = warp + 8u * (unsigned)i;
            *reinterpret_cast<uint4*>(k_s + tc_swz(kk, 2u * lane)) =
                make_uint4(pk[0], pk[1], pk[2], pk[3]);
            *reinterpret_cast<uint4*>(k_s + tc_swz(kk, 2u * lane + 1u)) =
                make_uint4(pk[4], pk[5], pk[6], pk[7]);
        }
        __syncthreads();

        // 2026-10-01: (2) The next tile's bytes, then the block-table reads for the one after.
        if (tile + 1 < n_tiles) {
            #pragma unroll
            for (int i = 0; i < 4; i++) raw[i] = tc_load_key(KV_cache, off_nxt[i], lane);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            off_nxt[i] = tc_key_off(sel_s, (tile + 2u) * TC_NK + warp + 8u * (unsigned)i, n_valid,
                                    my_block_table, block_size, cache_stride_bytes);
        }

        // 2026-10-01: (3) S partial over dims qd * 128..+127 for heads mt * 16.., all 32 keys.
        float s_acc[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) s_acc[nt][e] = 0.0f;
        }
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const unsigned kc = qd * 16u + 2u * (unsigned)ks;
            unsigned a[4];
            tc_ldsm_x4(tc_smem_addr(q_s + tc_swz(mt * 16u + a_row, kc + a_ch)), a[0], a[1], a[2],
                       a[3]);
            #pragma unroll
            for (int np = 0; np < 2; np++) {
                unsigned b0, b1, b2, b3;
                tc_ldsm_x4(tc_smem_addr(k_s + tc_swz((unsigned)np * 16u + b_row, kc + b_ch)), b0,
                           b1, b2, b3);
                tc_mma_bf16(s_acc[2 * np], a, b0, b1);
                tc_mma_bf16(s_acc[2 * np + 1], a, b2, b3);
            }
        }
        {
            float* sp = sp_s + qd * (TC_HEADS * TC_SP_LD);
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const unsigned key = (unsigned)nt * 8u + 2u * t4;
                *reinterpret_cast<float2*>(sp + (mt * 16u + g) * TC_SP_LD + key) =
                    make_float2(s_acc[nt][0], s_acc[nt][1]);
                *reinterpret_cast<float2*>(sp + (mt * 16u + g + 8u) * TC_SP_LD + key) =
                    make_float2(s_acc[nt][2], s_acc[nt][3]);
            }
        }
        __syncthreads();

        // 2026-10-01: (4) Online softmax: the four partials summed in quarter order, scaled into
        // the exp2 domain, keys past the compacted list masked. The tile's first key is always
        // valid, so m_new is finite and the first tile's alpha is exp2(-inf) = 0.
        {
            float s[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) s[e] = 0.0f;
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                const float4 v = *reinterpret_cast<const float4*>(
                    sp_s + q * (TC_HEADS * TC_SP_LD) + sm_row * TC_SP_LD + sm_c4);
                s[0] += v.x;
                s[1] += v.y;
                s[2] += v.z;
                s[3] += v.w;
            }
            const unsigned key0 = tile * TC_NK + sm_c4;
            float mx = tc_neg_inf();
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                s[e] = (key0 + (unsigned)e < n_valid) ? s[e] * qk_scale : tc_neg_inf();
                mx = fmaxf(mx, s[e]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            const float m_new = fmaxf(m_run, mx);
            const float alpha = exp2f(m_run - m_new);
            float p[4];
            float sum = 0.0f;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                p[e] = exp2f(s[e] - m_new);
                sum += p[e];
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            sum += __shfl_xor_sync(0xffffffffu, sum, 4);
            l_run = l_run * alpha + sum;
            m_run = m_new;
            *reinterpret_cast<uint2*>(p_s + (sm_row * TC_PS_LD + sm_c4) * 2u) =
                make_uint2(tc_pack_bf16(p[0], p[1]), tc_pack_bf16(p[2], p[3]));
            if ((tid & 7u) == 0u) alpha_s[sm_row] = alpha;
        }
        __syncthreads();

        // 2026-10-01: (5) O = O * alpha + P V over this tile's 32 keys (two k16 steps).
        {
            const float a_lo = alpha_s[mt * 16u + g];
            const float a_hi = alpha_s[mt * 16u + g + 8u];
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                o_acc[nt][0] *= a_lo;
                o_acc[nt][1] *= a_lo;
                o_acc[nt][2] *= a_hi;
                o_acc[nt][3] *= a_hi;
            }
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                unsigned a[4];
                tc_ldsm_x4(tc_smem_addr(p_s) + (mt * 16u + a_row) * (TC_PS_LD * 2u)
                               + (2u * (unsigned)ks + a_ch) * 16u,
                           a[0], a[1], a[2], a[3]);
                #pragma unroll
                for (int np = 0; np < 8; np++) {
                    unsigned b0, b1, b2, b3;
                    tc_ldsm_x4_t(tc_smem_addr(k_s + tc_swz((unsigned)ks * 16u + a_row,
                                                           qd * 16u + 2u * (unsigned)np + a_ch)),
                                 b0, b1, b2, b3);
                    tc_mma_bf16(o_acc[2 * np], a, b0, b1);
                    tc_mma_bf16(o_acc[2 * np + 1], a, b2, b3);
                }
            }
        }
        // 2026-10-01: The next tile overwrites the key tile, P and alpha.
        __syncthreads();
    }

    // 2026-10-01: (6) O * v_scale / l, zeros when the row had no valid index.
    if ((tid & 7u) == 0u) linv_s[sm_row] = (l_run > 0.0f) ? (kv_scale / l_run) : 0.0f;
    __syncthreads();
    {
        const float s_lo = linv_s[mt * 16u + g];
        const float s_hi = linv_s[mt * 16u + g + 8u];
        unsigned* o_lo = reinterpret_cast<unsigned*>(
            O + q_off + (unsigned long long)(mt * 16u + g) * TC_D);
        unsigned* o_hi = reinterpret_cast<unsigned*>(
            O + q_off + (unsigned long long)(mt * 16u + g + 8u) * TC_D);
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            const unsigned d2 = (qd * 128u + (unsigned)nt * 8u + 2u * t4) / 2u;
            o_lo[d2] = tc_pack_bf16(o_acc[nt][0] * s_lo, o_acc[nt][1] * s_lo);
            o_hi[d2] = tc_pack_bf16(o_acc[nt][2] * s_hi, o_acc[nt][3] * s_hi);
        }
    }
}
