// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: Phase-ablation variants of glm5next_dsa_mla_prefill_tc2_fp8 (module
// `glm5next_dsa_mla_prefill_ablate`), for TIMING ONLY. No lever and no production path resolves this
// module; only crates/model-arch/examples/dsa_mla_prefill_ablation_microtest.rs launches it.
//
// Why: METRALE_GLM_MLA_PREFILL_TC2 was bit-exact but only x0.95 of the tensor-core kernel
// (race-pf-mlatc2-L12, 2026-10-09), so halving the barriers was not the limiter; ncu is not
// available on the cluster. Each arm below removes one phase of TC2 and keeps the rest of the
// schedule (both barriers per tile, the compaction, the shared-memory layout), so the time an arm
// saves is that phase's share of the critical path.
//
// One templated body (ab_body), the same as glm5next_dsa_mla_prefill_tc2.cu's tc2_body<true> with
// the phases behind compile-time flags; the control arm (all phases, NK 32, compaction) writes the
// same bits as glm5next_dsa_mla_prefill_tc2_fp8 (the microtest checks it). Arms:
//   a  ab_full        control
//   b  ab_nogather    no block-table / key loads: a per-tile register pattern is converted
//   c  ab_noconvert   key bytes stored as they are (16 raw bytes written twice per lane)
//   d  ab_noqk        no QK MMAs: the S partials are a cheap per-tile value
//   e  ab_nosoftmax   no softmax: P is a constant BF16 tile and alpha 1, written once
//   f  ab_nopv        no rescale and no PV MMAs
//   g  ab_mmaonly     QK and PV MMAs on whatever the buffers hold; no gather, convert, key store or
//                     softmax
//   h  ab_nk16_occ2   16-key tiles, no compaction (the selection read from global), 46,880 B of
//                     shared memory, __launch_bounds__(256, 2): two CTAs per SM if registers allow
//   h1 ab_nk16_occ1   the same as h with __launch_bounds__(256, 1)
//   i  ab_nocompact   the control without compaction (separates that change from h's NK)
// Arms other than a write different numbers by construction; every arm writes its accumulators
// to O so no phase it keeps is dead code.
//
// Owner: gb10 kernels (glm-5.3-flash), microtest support.
// Invariants:
// - Same arguments, block (256) and grid ([num_q_heads / 32, rows]) as the TC2 entry; dynamic shared
//   memory AB_SMEM_NK32 (99,104) or AB_SMEM_NK16 (46,880), mirrored in the microtest.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define AB_THREADS 256
#define AB_WARPS 8
#define AB_HEADS 32
#define AB_D 512
#define AB_MAX_SEL 2560
#define AB_NO_KEY (~0ull)
#define AB_SMEM_NK32 99104
#define AB_SMEM_NK16 46880

// 2026-10-09: Shared-memory layout of a variant with NK-key tiles, with or without the compacted
// index list. The Q tile (32 KB) is staged in the second key buffer at NK 32 and across both at
// NK 16.
template <int NK, bool COMPACT>
struct AbLayout {
    static constexpr int KBUF = NK * AB_D * 2;
    static constexpr int SP_LD = NK + 8;
    static constexpr int PS_LD = NK + 8;
    static constexpr int OFF_K0 = 0;
    static constexpr int OFF_Q = (NK == 32) ? KBUF : 0;
    static constexpr int OFF_SP = 2 * KBUF;
    static constexpr int OFF_P = OFF_SP + 4 * AB_HEADS * SP_LD * 4;
    static constexpr int OFF_SEL = OFF_P + AB_HEADS * PS_LD * 2;
    static constexpr int OFF_ALPHA = OFF_SEL + (COMPACT ? AB_MAX_SEL * 4 : 0);
    static constexpr int OFF_LINV = OFF_ALPHA + AB_HEADS * 4;
    static constexpr int OFF_CNT = OFF_LINV + AB_HEADS * 4;
    static constexpr int BYTES = OFF_CNT + AB_WARPS * 4;
};
static_assert(AbLayout<32, true>::BYTES == AB_SMEM_NK32, "NK 32 layout is TC2's");
static_assert(AbLayout<16, false>::BYTES == AB_SMEM_NK16, "NK 16 layout");
static_assert(AbLayout<32, false>::BYTES <= AB_SMEM_NK32, "the no-compaction arm fits the NK 32 launch");

// ---- Helpers copied from glm5next_dsa_mla_prefill_tc2.cu (tc2_ -> ab_). ----

__device__ __forceinline__ float ab_neg_inf() { return __int_as_float((int)0xff800000u); }

__device__ __forceinline__ unsigned ab_smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ unsigned ab_swz(unsigned row, unsigned ch) {
    return row * (AB_D * 2) + ((ch ^ (row & 7u)) << 4);
}

__device__ __forceinline__ void ab_ldsm_x4(unsigned a, unsigned& d0, unsigned& d1, unsigned& d2,
                                           unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a));
}

__device__ __forceinline__ void ab_ldsm_x4_t(unsigned a, unsigned& d0, unsigned& d1, unsigned& d2,
                                             unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a));
}

__device__ __forceinline__ void ab_mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ unsigned ab_pack_bf16(float lo, float hi) {
    const unsigned l = __bfloat16_as_ushort(__float2bfloat16(lo));
    const unsigned h = __bfloat16_as_ushort(__float2bfloat16(hi));
    return l | (h << 16);
}

__device__ __forceinline__ void ab_fp8x16_to_bf16(uint4 raw, unsigned (&out)[8]) {
    const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(
                (__nv_fp8x2_storage_t)((w[i] >> (16 * h)) & 0xFFFFu), __NV_E4M3);
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
            out[2 * i + h] = ab_pack_bf16(f.x, f.y);
        }
    }
}

__device__ __forceinline__ void ab_e4m3x4_to_bf16_bits(unsigned w, unsigned& lo, unsigned& hi) {
    const unsigned mag = w & 0x7F7F7F7Fu;
    const unsigned sgn = w ^ mag;
    const unsigned lo_bits = (__byte_perm(mag, 0u, 0x4140u) << 4) | __byte_perm(sgn, 0u, 0x1404u);
    const unsigned hi_bits = (__byte_perm(mag, 0u, 0x4342u) << 4) | __byte_perm(sgn, 0u, 0x3424u);
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(lo) : "r"(lo_bits), "r"(0x7B807B80u),
        "r"(0x80008000u));
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(hi) : "r"(hi_bits), "r"(0x7B807B80u),
        "r"(0x80008000u));
}

__device__ __forceinline__ void ab_fp8x16_to_bf16_bits(uint4 raw, unsigned (&out)[8]) {
    const unsigned nan = (((raw.x & 0x7F7F7F7Fu) + 0x01010101u) | ((raw.y & 0x7F7F7F7Fu) + 0x01010101u)
                          | ((raw.z & 0x7F7F7F7Fu) + 0x01010101u)
                          | ((raw.w & 0x7F7F7F7Fu) + 0x01010101u))
                       & 0x80808080u;
    if (nan != 0u) {
        ab_fp8x16_to_bf16(raw, out);
    } else {
        ab_e4m3x4_to_bf16_bits(raw.x, out[0], out[1]);
        ab_e4m3x4_to_bf16_bits(raw.y, out[2], out[3]);
        ab_e4m3x4_to_bf16_bits(raw.z, out[4], out[5]);
        ab_e4m3x4_to_bf16_bits(raw.w, out[6], out[7]);
    }
}

// 2026-10-09: Byte offset of key j: from the compacted list (COMPACT) or straight from the row's
// global selection (an invalid entry gives AB_NO_KEY, a zero row).
template <bool COMPACT>
__device__ __forceinline__ unsigned long long ab_key_off(
    const int* sel_s, const int* __restrict__ my_sel, unsigned j, unsigned n_valid,
    unsigned seq_len, const int* __restrict__ block_table, unsigned block_size,
    unsigned long long cache_stride_bytes
) {
    if (j >= n_valid) return AB_NO_KEY;
    int ti;
    if constexpr (COMPACT) {
        ti = sel_s[j];
    } else {
        ti = my_sel[j];
        if (ti < 0 || (unsigned)ti >= seq_len) return AB_NO_KEY;
    }
    const unsigned t = (unsigned)ti;
    const unsigned physical_block = (unsigned)block_table[t / block_size];
    return (unsigned long long)physical_block * cache_stride_bytes
         + (unsigned long long)(t % block_size) * AB_D;
}

__device__ __forceinline__ uint4 ab_load_key(const unsigned char* __restrict__ kv,
                                             unsigned long long off, unsigned lane) {
    if (off == AB_NO_KEY) return make_uint4(0u, 0u, 0u, 0u);
    return __ldg(reinterpret_cast<const uint4*>(kv + off) + lane);
}

// 2026-10-09: The phases an arm keeps.
template <int NK, bool GATHER, bool CONVERT, bool STOREK, bool QK, bool SOFTMAX, bool PV, bool COMPACT>
struct AbArm {
    static constexpr int nk = NK;
    static constexpr bool gather = GATHER, convert = CONVERT, storek = STOREK, qk = QK,
                          softmax = SOFTMAX, pv = PV, compact = COMPACT;
};

// 2026-10-09: The tile's raw key chunks of this thread (keys warp + 8 i): loaded (GATHER) or a
// per-tile register pattern of non-NaN codes (bytes 0x28..0x3F).
template <class A>
__device__ __forceinline__ void ab_fetch(uint4 (&raw)[A::nk / 8], unsigned long long (&off)[A::nk / 8],
                                         const unsigned char* __restrict__ kv, unsigned lane,
                                         unsigned warp, unsigned tile) {
    #pragma unroll
    for (int i = 0; i < A::nk / 8; i++) {
        if constexpr (A::gather) {
            raw[i] = ab_load_key(kv, off[i], lane);
        } else {
            const unsigned v = 0x30303030u ^ (((tile + (unsigned)i) & 7u) * 0x01010101u);
            raw[i] = make_uint4(v, v ^ (lane & 7u), v ^ warp, v ^ 0x08080808u);
        }
    }
}

template <class A>
__device__ __forceinline__ void ab_offsets(unsigned long long (&off)[A::nk / 8], const int* sel_s,
                                           const int* __restrict__ my_sel, unsigned first,
                                           unsigned warp, unsigned n_valid, unsigned seq_len,
                                           const int* __restrict__ bt, unsigned block_size,
                                           unsigned long long stride) {
    if constexpr (A::gather) {
        #pragma unroll
        for (int i = 0; i < A::nk / 8; i++) {
            off[i] = ab_key_off<A::compact>(sel_s, my_sel, first + warp + 8u * (unsigned)i,
                                            n_valid, seq_len, bt, block_size, stride);
        }
    }
}

template <class A>
__device__ __forceinline__ void ab_store_tile(unsigned char* kb, const uint4 (&raw)[A::nk / 8],
                                              unsigned warp, unsigned lane) {
    if constexpr (A::storek) {
    #pragma unroll
    for (int i = 0; i < A::nk / 8; i++) {
        unsigned pk[8];
        if constexpr (A::convert) {
            ab_fp8x16_to_bf16_bits(raw[i], pk);
        } else {
            pk[0] = raw[i].x; pk[1] = raw[i].y; pk[2] = raw[i].z; pk[3] = raw[i].w;
            pk[4] = raw[i].x; pk[5] = raw[i].y; pk[6] = raw[i].z; pk[7] = raw[i].w;
        }
        const unsigned kk = warp + 8u * (unsigned)i;
        *reinterpret_cast<uint4*>(kb + ab_swz(kk, 2u * lane)) =
            make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *reinterpret_cast<uint4*>(kb + ab_swz(kk, 2u * lane + 1u)) =
            make_uint4(pk[4], pk[5], pk[6], pk[7]);
    }
    }
}

template <class A>
__device__ __forceinline__ void ab_qk_tile(const unsigned (&qf)[8][4], unsigned kb, float* sp_s,
                                           unsigned mt, unsigned qd, unsigned g, unsigned t4,
                                           unsigned b_row, unsigned b_ch, unsigned tile) {
    constexpr int NT = A::nk / 8;
    constexpr int SP_LD = AbLayout<A::nk, A::compact>::SP_LD;
    float s_acc[NT][4];
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            s_acc[nt][e] = A::qk ? 0.0f : (float)((tile + (unsigned)(nt + e)) & 3u) * 0.25f;
        }
    }
    if constexpr (A::qk) {
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const unsigned kc = qd * 16u + 2u * (unsigned)ks;
            #pragma unroll
            for (int np = 0; np < A::nk / 16; np++) {
                unsigned b0, b1, b2, b3;
                ab_ldsm_x4(kb + ab_swz((unsigned)np * 16u + b_row, kc + b_ch), b0, b1, b2, b3);
                ab_mma_bf16(s_acc[2 * np], qf[ks], b0, b1);
                ab_mma_bf16(s_acc[2 * np + 1], qf[ks], b2, b3);
            }
        }
    }
    float* sp = sp_s + qd * (AB_HEADS * SP_LD);
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const unsigned key = (unsigned)nt * 8u + 2u * t4;
        *reinterpret_cast<float2*>(sp + (mt * 16u + g) * SP_LD + key) =
            make_float2(s_acc[nt][0], s_acc[nt][1]);
        *reinterpret_cast<float2*>(sp + (mt * 16u + g + 8u) * SP_LD + key) =
            make_float2(s_acc[nt][2], s_acc[nt][3]);
    }
}

template <class A>
__device__ __forceinline__ void ab_body(
    unsigned char* smem,
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ KV_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int* __restrict__ sel_indices,
    const unsigned int sel_width,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float kv_scale,
    const unsigned long long cache_stride_bytes
) {
    using L = AbLayout<A::nk, A::compact>;
    constexpr int NK = A::nk;
    constexpr int KPW = NK / 8;  // keys per warp per tile, and keys per softmax thread
    unsigned char* k0_s = smem + L::OFF_K0;
    unsigned char* q_s  = smem + L::OFF_Q;
    float* sp_s         = reinterpret_cast<float*>(smem + L::OFF_SP);
    unsigned char* p_s  = smem + L::OFF_P;
    int* sel_s          = reinterpret_cast<int*>(smem + L::OFF_SEL);
    float* alpha_s      = reinterpret_cast<float*>(smem + L::OFF_ALPHA);
    float* linv_s       = reinterpret_cast<float*>(smem + L::OFF_LINV);
    int* cnt_s          = reinterpret_cast<int*>(smem + L::OFF_CNT);

    const unsigned head0 = blockIdx.x * AB_HEADS;
    const unsigned row   = blockIdx.y;
    const unsigned tid   = threadIdx.x;
    const unsigned warp  = tid / 32u;
    const unsigned lane  = tid % 32u;

    if (head0 + AB_HEADS > num_q_heads) return;
    const unsigned seq_len = (unsigned)seq_lens[row];
    if (seq_len == 0) return;

    const int* my_block_table = block_tables + (size_t)row * max_blocks_per_seq;
    const int* my_sel         = sel_indices  + (size_t)row * sel_width;
    const unsigned long long q_off = ((unsigned long long)row * num_q_heads + head0) * AB_D;

    {
        const uint4* q4 = reinterpret_cast<const uint4*>(Q + q_off);
        #pragma unroll
        for (int i = 0; i < AB_HEADS * AB_D * 2 / 16 / AB_THREADS; i++) {
            const unsigned c = tid + (unsigned)i * AB_THREADS;
            *reinterpret_cast<uint4*>(q_s + ab_swz(c >> 6, c & 63u)) = q4[c];
        }
    }

    unsigned n_valid = 0;
    if constexpr (A::compact) {
        for (unsigned j0 = 0; j0 < sel_width; j0 += AB_THREADS) {
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
            for (unsigned w = 0; w < AB_WARPS; w++) {
                const unsigned c = (unsigned)cnt_s[w];
                if (w < warp) base += c;
                total += c;
            }
            if (ok) sel_s[base + __popc(ball & ((1u << lane) - 1u))] = t;
            n_valid += total;
            __syncthreads();
        }
        if (sel_width == 0) __syncthreads();
    } else {
        n_valid = sel_width;
        __syncthreads();
    }

    const unsigned n_tiles = (n_valid + NK - 1) / NK;

    uint4 raw[KPW];
    unsigned long long off_nxt[KPW];
    #pragma unroll
    for (int i = 0; i < KPW; i++) off_nxt[i] = AB_NO_KEY;
    ab_offsets<A>(off_nxt, sel_s, my_sel, 0u, warp, n_valid, seq_len, my_block_table, block_size,
                  cache_stride_bytes);
    ab_fetch<A>(raw, off_nxt, KV_cache, lane, warp, 0u);
    ab_offsets<A>(off_nxt, sel_s, my_sel, (unsigned)NK, warp, n_valid, seq_len, my_block_table,
                  block_size, cache_stride_bytes);

    const unsigned mt     = warp & 1u;
    const unsigned qd     = warp >> 1;
    const unsigned g      = lane >> 2;
    const unsigned t4     = lane & 3u;
    const unsigned sm_row = tid >> 3;
    const unsigned sm_c   = (tid & 7u) * (unsigned)KPW;
    const unsigned a_row = (lane & 7u) + ((lane >> 3) & 1u) * 8u;
    const unsigned a_ch  = lane >> 4;
    const unsigned b_row = (lane & 7u) + ((lane >> 4) & 1u) * 8u;
    const unsigned b_ch  = (lane >> 3) & 1u;

    unsigned qf[8][4];
    #pragma unroll
    for (int ks = 0; ks < 8; ks++) {
        const unsigned kc = qd * 16u + 2u * (unsigned)ks;
        ab_ldsm_x4(ab_smem_addr(q_s + ab_swz(mt * 16u + a_row, kc + a_ch)), qf[ks][0], qf[ks][1],
                   qf[ks][2], qf[ks][3]);
    }
    // 2026-10-09: At NK 16 the Q tile spans both key buffers: every warp reads its fragments
    // before tile 0 is stored.
    if constexpr (NK != 32) __syncthreads();

    const float qk_scale = kv_scale * inv_sqrt_d * 1.4426950408889634f;
    float m_run = ab_neg_inf();
    float l_run = 0.0f;

    float o_acc[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) o_acc[nt][e] = 0.0f;
    }

    // 2026-10-09: Without softmax, P is a constant BF16 tile and alpha 1, written once.
    if constexpr (!A::softmax) {
        for (unsigned i = tid; i < (unsigned)(AB_HEADS * L::PS_LD / 2); i += AB_THREADS) {
            reinterpret_cast<unsigned*>(p_s)[i] = 0x3C003C00u;
        }
        if (tid < AB_HEADS) alpha_s[tid] = 1.0f;
    }

    ab_store_tile<A>(k0_s, raw, warp, lane);
    ab_fetch<A>(raw, off_nxt, KV_cache, lane, warp, 1u);
    ab_offsets<A>(off_nxt, sel_s, my_sel, 2u * NK, warp, n_valid, seq_len, my_block_table,
                  block_size, cache_stride_bytes);
    __syncthreads();

    const unsigned kb0 = ab_smem_addr(k0_s);
    ab_qk_tile<A>(qf, kb0, sp_s, mt, qd, g, t4, b_row, b_ch, 0u);

    for (unsigned tile = 0; tile < n_tiles; tile++) {
        const unsigned kcur = kb0 + (tile & 1u) * (unsigned)L::KBUF;
        const unsigned nbuf = (tile + 1u) & 1u;
        const unsigned knxt = kb0 + nbuf * (unsigned)L::KBUF;

        __syncthreads();  // B1

        ab_store_tile<A>(smem + nbuf * L::KBUF, raw, warp, lane);
        ab_fetch<A>(raw, off_nxt, KV_cache, lane, warp, tile + 2u);
        ab_offsets<A>(off_nxt, sel_s, my_sel, (tile + 3u) * NK, warp, n_valid, seq_len,
                      my_block_table, block_size, cache_stride_bytes);

        if constexpr (A::softmax) {
            float s[KPW];
            #pragma unroll
            for (int e = 0; e < KPW; e++) s[e] = 0.0f;
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                const float* src = sp_s + q * (AB_HEADS * L::SP_LD) + sm_row * L::SP_LD + sm_c;
                if constexpr (KPW == 4) {
                    const float4 v = *reinterpret_cast<const float4*>(src);
                    s[0] += v.x;
                    s[1] += v.y;
                    s[2] += v.z;
                    s[3] += v.w;
                } else {
                    const float2 v = *reinterpret_cast<const float2*>(src);
                    s[0] += v.x;
                    s[1] += v.y;
                }
            }
            const unsigned key0 = tile * NK + sm_c;
            float mx = ab_neg_inf();
            #pragma unroll
            for (int e = 0; e < KPW; e++) {
                s[e] = (key0 + (unsigned)e < n_valid) ? s[e] * qk_scale : ab_neg_inf();
                mx = fmaxf(mx, s[e]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            const float m_new = fmaxf(m_run, mx);
            const float alpha = exp2f(m_run - m_new);
            float p[KPW];
            float sum = 0.0f;
            #pragma unroll
            for (int e = 0; e < KPW; e++) {
                p[e] = exp2f(s[e] - m_new);
                sum += p[e];
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            sum += __shfl_xor_sync(0xffffffffu, sum, 4);
            l_run = l_run * alpha + sum;
            m_run = m_new;
            if constexpr (KPW == 4) {
                *reinterpret_cast<uint2*>(p_s + (sm_row * L::PS_LD + sm_c) * 2u) =
                    make_uint2(ab_pack_bf16(p[0], p[1]), ab_pack_bf16(p[2], p[3]));
            } else {
                *reinterpret_cast<unsigned*>(p_s + (sm_row * L::PS_LD + sm_c) * 2u) =
                    ab_pack_bf16(p[0], p[1]);
            }
            if ((tid & 7u) == 0u) alpha_s[sm_row] = alpha;
        }

        __syncthreads();  // B2

        {
            const float a_lo = alpha_s[mt * 16u + g];
            const float a_hi = alpha_s[mt * 16u + g + 8u];
            if constexpr (A::pv) {
                #pragma unroll
                for (int nt = 0; nt < 16; nt++) {
                    o_acc[nt][0] *= a_lo;
                    o_acc[nt][1] *= a_lo;
                    o_acc[nt][2] *= a_hi;
                    o_acc[nt][3] *= a_hi;
                }
                #pragma unroll
                for (int ks = 0; ks < NK / 16; ks++) {
                    unsigned a[4];
                    ab_ldsm_x4(ab_smem_addr(p_s) + (mt * 16u + a_row) * (L::PS_LD * 2u)
                                   + (2u * (unsigned)ks + a_ch) * 16u,
                               a[0], a[1], a[2], a[3]);
                    #pragma unroll
                    for (int np = 0; np < 8; np++) {
                        unsigned b0, b1, b2, b3;
                        ab_ldsm_x4_t(kcur + ab_swz((unsigned)ks * 16u + a_row,
                                                   qd * 16u + 2u * (unsigned)np + a_ch),
                                     b0, b1, b2, b3);
                        ab_mma_bf16(o_acc[2 * np], a, b0, b1);
                        ab_mma_bf16(o_acc[2 * np + 1], a, b2, b3);
                    }
                }
            } else {
                // 2026-10-09: Keep the alpha reads live without the rescale.
                o_acc[0][0] += a_lo;
                o_acc[0][2] += a_hi;
            }
        }

        ab_qk_tile<A>(qf, knxt, sp_s, mt, qd, g, t4, b_row, b_ch, tile + 1u);
    }

    if constexpr (!A::softmax) l_run = 1.0f;
    if ((tid & 7u) == 0u) linv_s[sm_row] = (l_run > 0.0f) ? (kv_scale / l_run) : 0.0f;
    __syncthreads();
    {
        const float s_lo = linv_s[mt * 16u + g];
        const float s_hi = linv_s[mt * 16u + g + 8u];
        unsigned* o_lo = reinterpret_cast<unsigned*>(
            O + q_off + (unsigned long long)(mt * 16u + g) * AB_D);
        unsigned* o_hi = reinterpret_cast<unsigned*>(
            O + q_off + (unsigned long long)(mt * 16u + g + 8u) * AB_D);
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            const unsigned d2 = (qd * 128u + (unsigned)nt * 8u + 2u * t4) / 2u;
            o_lo[d2] = ab_pack_bf16(o_acc[nt][0] * s_lo, o_acc[nt][1] * s_lo);
            o_hi[d2] = ab_pack_bf16(o_acc[nt][2] * s_hi, o_acc[nt][3] * s_hi);
        }
    }
}

#define AB_PARAMS                                                                                 \
    const __nv_bfloat16* __restrict__ Q, const unsigned char* __restrict__ KV_cache,             \
        __nv_bfloat16* __restrict__ O, const int* __restrict__ block_tables,                     \
        const int* __restrict__ seq_lens, const int* __restrict__ sel_indices,                   \
        const unsigned int sel_width, const unsigned int max_blocks_per_seq,                     \
        const unsigned int num_q_heads, const unsigned int block_size, const float inv_sqrt_d,  \
        const float kv_scale, const unsigned long long cache_stride_bytes
#define AB_ARGS                                                                                   \
    Q, KV_cache, O, block_tables, seq_lens, sel_indices, sel_width, max_blocks_per_seq,          \
        num_q_heads, block_size, inv_sqrt_d, kv_scale, cache_stride_bytes

// 2026-10-09: One extern "C" entry per arm. Flags: NK, gather, convert, store K, QK, softmax, PV,
// compaction.
#define AB_ENTRY(name, minb, ...)                                                                 \
    extern "C" __global__ void __launch_bounds__(AB_THREADS, minb) name(AB_PARAMS) {             \
        extern __shared__ __align__(16) unsigned char name##_smem[];                              \
        ab_body<AbArm<__VA_ARGS__>>(name##_smem, AB_ARGS);                                        \
    }

AB_ENTRY(glm5next_dsa_mla_prefill_ablate_full, 1, 32, true, true, true, true, true, true, true)
AB_ENTRY(glm5next_dsa_mla_prefill_ablate_nogather, 1, 32, false, true, true, true, true, true, true)
AB_ENTRY(glm5next_dsa_mla_prefill_ablate_noconvert, 1, 32, true, false, true, true, true, true, true)
AB_ENTRY(glm5next_dsa_mla_prefill_ablate_noqk, 1, 32, true, true, true, false, true, true, true)
AB_ENTRY(glm5next_dsa_mla_prefill_ablate_nosoftmax, 1, 32, true, true, true, true, false, true, true)
AB_ENTRY(glm5next_dsa_mla_prefill_ablate_nopv, 1, 32, true, true, true, true, true, false, true)
AB_ENTRY(glm5next_dsa_mla_prefill_ablate_mmaonly, 1, 32, false, false, false, true, false, true, true)
AB_ENTRY(glm5next_dsa_mla_prefill_ablate_nk16_occ2, 2, 16, true, true, true, true, true, true, false)
AB_ENTRY(glm5next_dsa_mla_prefill_ablate_nk16_occ1, 1, 16, true, true, true, true, true, true, false)
AB_ENTRY(glm5next_dsa_mla_prefill_ablate_nocompact, 1, 32, true, true, true, true, true, true, false)
