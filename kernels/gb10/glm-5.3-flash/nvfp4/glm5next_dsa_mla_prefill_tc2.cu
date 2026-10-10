// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-08: GLM-5.3-Flash DSA MLA prefill attention on tensor cores, second dataflow (module
// `glm5next_dsa_mla_prefill_tc2`, entries `glm5next_dsa_mla_prefill_tc2_fp8` and
// `glm5next_dsa_mla_prefill_tc2_hwcvt_fp8`), opt-in through METRALE_GLM_MLA_PREFILL_TC2=1 where
// METRALE_GLM_MLA_PREFILL_TC=1 takes `glm5next_dsa_mla_prefill_tc_fp8`
// (glm5next_dsa/attend/prefill_tc.rs). It writes the same output bits as that kernel: same
// inputs, same grid [num_q_heads / 32, rows], block 256, same dynamic shared memory size.
//
// What is the same as glm5next_dsa_mla_prefill_tc.cu (the helpers below are copied verbatim with
// a tc2_ prefix, and this directory compiles with --fmad=false, so the same expressions give the
// same instructions' results):
//   - the stable compaction of the selection, the per-warp key loads (4 keys x 16 B per lane),
//     the block-table arithmetic, zero rows past the compacted list;
//   - the BF16 key tile bits and its XOR swizzle;
//   - the Q A-fragments (the same swizzled staging and the same ldmatrix.x4 addresses);
//   - S = Q K^T per warp (16 heads, one 128-dim quarter): the same m16n8k16 instructions, the same
//     operands, the same per-accumulator order (k-steps 0..7); the quarter partials summed from
//     0.0f in quarter order 0..3;
//   - the online softmax and the epilogue (verbatim);
//   - O += P V: alpha rescale, then the same two k16 steps over the same P and key-tile bits.
//
// What changes (exact by construction):
//   A1 (`_fp8` entry only) the FP8 E4M3 -> BF16 conversion is integer bit placement plus one
//      exact BF16 multiply by 2^120 per pair (tc2_fp8x16_to_bf16_bits) instead of
//      E4M3 -> FP16 -> FP32 -> BF16 conversions. For every non-NaN code the result is the code's
//      value, which the old chain also produces (every E4M3 value is a BF16 value). A 16-byte
//      group holding a NaN code (0x7F / 0xFF) goes through the old chain itself, so NaN bits
//      match whatever the old chain gives. `_hwcvt_fp8` keeps the old chain everywhere (A2 + A3
//      only), for timing the converter separately.
//   A2 the Q A-fragments live in registers (32 per thread), loaded once per CTA, so the Q tile no
//      longer occupies shared memory during the key loop; Q is staged in the second key buffer.
//   A3 the BF16 key tile is double-buffered and each tile runs 2 barriers instead of 4:
//        B1 | convert tile t+1 into the other buffer, issue tile t+2 loads, softmax(t)
//        B2 | PV(t) on this buffer, then QK(t+1) on the other buffer (S partials of t+1)
//      Hazards: B1 orders PV(t-1)'s reads of buffer (t+1)&1, P and alpha before they are
//      rewritten, and QK(t)'s S partials before softmax(t) reads them; B2 orders softmax(t)'s
//      reads of the partials before QK(t+1) rewrites them, and publishes P, alpha and tile t+1.
//      The last tile's QK(t+1) runs on a stale or zero buffer and its partials are never read.
//   A5 (2026-10-09) the block-table reads for tile t+3 are issued at tile t and their pages kept in
//      registers; the byte offsets are formed at tile t+1, where the key loads use them. The old
//      schedule used each page right after its load, so each tile waited for four block-table
//      round trips in a row behind B1 (cuobjdump -sass of the _fp8 entry). The offsets are
//      the same expression (tc2_page_off = tc2_key_off's arithmetic), so the loads are unchanged.
//   A6 (2026-10-09) after the compaction each compacted token index t is replaced in place by its
//      cache row block_table[t / block_size] * block_size + t % block_size, once per CTA (one
//      thread per index); the key loop forms each offset as row * TC2_D. The host takes this
//      entry only where cache_stride_bytes == block_size * TC2_D, so row * TC2_D equals the old
//      page * cache_stride_bytes + (t % block_size) * TC2_D, and the loads are unchanged. A row
//      fits 32 bits while the cache is below 2 TiB. Before A6 every lane of a warp ran the
//      runtime-divisor division and the block-table load for each of its 4 keys per tile
//      (~130 of the loop's 910 SASS instructions, cuobjdump -sass of the _fp8 entry, nvcc 13.2).
//   A7 (2026-10-09, microtest arm `_tc2_pf3_fp8` only, not launched by the lever) key loads three
//      tiles ahead instead of two, for no-reuse long selections where A6's shorter tile hides less
//      DRAM latency (keyload `dram` 131K: A6 0.893 ms vs A5 0.819 ms). 225 vs 213 registers.
//   A4 (persistent grid) is not used: at 99 KB of shared memory one CTA fits per SM, and rows past
//      2,051 tokens all run 65 tiles, so a pull scheduler runs the same ceil(rows / SMs) rounds.
//
// Written for this repository from the PTX ISA fragment layouts, as glm5next_dsa_mla_prefill_tc.cu;
// no third-party code is vendored. GPU gate: crates/model-arch/examples/dsa_mla_prefill_tc_microtest.rs
// (bitwise against glm5next_dsa_mla_prefill_tc_fp8, all 256 codes through both converters).
//
// Owner: gb10 kernels (glm-5.3-flash).
// Invariants:
// - Block 256 threads; dynamic shared memory TC2_SMEM_BYTES; grid [num_q_heads / TC2_HEADS, rows].
// - The host launches only where glm5next_dsa_mla_prefill_tc_fp8 would launch (same refusals).
// - TC2_HEADS == TC2_NK: the Q tile is staged in the second key buffer.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define TC2_WARP_SIZE 32
#define TC2_WARPS 8
#define TC2_THREADS 256
// 2026-10-08: Heads per CTA. Mirrored by MLA_PREFILL_TC2_HEADS (glm5next_dsa/attend/prefill_tc.rs).
#define TC2_HEADS 32
#define TC2_D 512
#define TC2_NK 32
// 2026-10-08: Widest selection row (as TC_MAX_SEL). Mirrored by MLA_PREFILL_TC_MAX_SEL.
#define TC2_MAX_SEL 2560
#define TC2_SP_LD 40
#define TC2_PS_LD 40

// 2026-10-08: Dynamic shared-memory layout, byte offsets: the old layout with the Q tile replaced
// by the second key buffer.
#define TC2_KBUF_BYTES (TC2_NK * TC2_D * 2)
#define TC2_OFF_K0 0                                               // [32][512] BF16, swizzled
#define TC2_OFF_K1 (TC2_OFF_K0 + TC2_KBUF_BYTES)                   // [32][512] BF16, swizzled
#define TC2_OFF_SP (TC2_OFF_K1 + TC2_KBUF_BYTES)                   // [4][32][TC2_SP_LD] FP32
#define TC2_OFF_P (TC2_OFF_SP + 4 * TC2_HEADS * TC2_SP_LD * 4)     // [32][TC2_PS_LD] BF16
#define TC2_OFF_SEL (TC2_OFF_P + TC2_HEADS * TC2_PS_LD * 2)        // [TC2_MAX_SEL] i32
#define TC2_OFF_ALPHA (TC2_OFF_SEL + TC2_MAX_SEL * 4)              // [32] FP32
#define TC2_OFF_LINV (TC2_OFF_ALPHA + TC2_HEADS * 4)               // [32] FP32
#define TC2_OFF_CNT (TC2_OFF_LINV + TC2_HEADS * 4)                 // [8] i32
// 2026-10-08: Mirrored by MLA_PREFILL_TC2_SMEM_BYTES.
#define TC2_SMEM_BYTES 99104
static_assert(TC2_OFF_CNT + TC2_WARPS * 4 == TC2_SMEM_BYTES, "TC2_SMEM_BYTES is the layout's size");
static_assert(TC2_HEADS == TC2_NK, "the Q tile is staged in the second key buffer");
static_assert(TC2_KBUF_BYTES == 32768, "the buffer select is a shift by 15");
static_assert(TC2_HEADS * TC2_D * 2 / 16 % TC2_THREADS == 0, "Q loads split evenly");
static_assert(TC2_NK * TC2_D / 16 == 4 * TC2_THREADS, "each thread loads 4 key chunks per tile");

#define TC2_NO_KEY (~0ull)

// ---- Helpers copied verbatim from glm5next_dsa_mla_prefill_tc.cu (tc_ -> tc2_). ----

__device__ __forceinline__ float tc2_neg_inf() { return __int_as_float((int)0xff800000u); }

__device__ __forceinline__ unsigned tc2_smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ unsigned tc2_swz(unsigned row, unsigned ch) {
    return row * (TC2_D * 2) + ((ch ^ (row & 7u)) << 4);
}

__device__ __forceinline__ void tc2_ldsm_x4(unsigned a, unsigned& d0, unsigned& d1, unsigned& d2,
                                            unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a));
}

__device__ __forceinline__ void tc2_ldsm_x4_t(unsigned a, unsigned& d0, unsigned& d1, unsigned& d2,
                                              unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a));
}

__device__ __forceinline__ void tc2_mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ unsigned tc2_pack_bf16(float lo, float hi) {
    const unsigned l = __bfloat16_as_ushort(__float2bfloat16(lo));
    const unsigned h = __bfloat16_as_ushort(__float2bfloat16(hi));
    return l | (h << 16);
}

// 2026-10-08: The old chain (tc_fp8x16_to_bf16): E4M3 -> FP16 -> FP32 -> BF16, two dims per word,
// the lower dim in the low half.
__device__ __forceinline__ void tc2_fp8x16_to_bf16(uint4 raw, unsigned (&out)[8]) {
    const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(
                (__nv_fp8x2_storage_t)((w[i] >> (16 * h)) & 0xFFFFu), __NV_E4M3);
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
            out[2 * i + h] = tc2_pack_bf16(f.x, f.y);
        }
    }
}

__device__ __forceinline__ unsigned long long tc2_key_off(
    const int* sel_s,
    unsigned j,
    unsigned n_valid,
    const int* __restrict__ block_table,
    unsigned block_size,
    unsigned long long cache_stride_bytes
) {
    if (j >= n_valid) return TC2_NO_KEY;
    const unsigned t = (unsigned)sel_s[j];
    const unsigned physical_block = (unsigned)block_table[t / block_size];
    return (unsigned long long)physical_block * cache_stride_bytes
         + (unsigned long long)(t % block_size) * TC2_D;
}

// 2026-10-09: A6: sel_s holds cache rows (tc2_resolve_rows); a key past the list is TC2_NO_ROW
// (no row: the cache is below 2 TiB) and loads zeros. tc2_row_off(r) == tc2_key_off for the
// same key where cache_stride_bytes == block_size * TC2_D.
#define TC2_NO_ROW (~0u)
__device__ __forceinline__ unsigned tc2_key_row(const int* sel_s, unsigned j, unsigned n_valid) {
    return j < n_valid ? (unsigned)sel_s[j] : TC2_NO_ROW;
}

__device__ __forceinline__ unsigned long long tc2_row_off(unsigned row) {
    if (row == TC2_NO_ROW) return TC2_NO_KEY;
    return (unsigned long long)row * TC2_D;
}

// 2026-10-09: A6: sel_s[j] = t -> block_table[t / block_size] * block_size + t % block_size for
// j < n_valid, one thread per index; all block-table loads are issued before any store.
#define TC2_RESOLVE (TC2_MAX_SEL / TC2_THREADS)
static_assert(TC2_MAX_SEL % TC2_THREADS == 0, "each thread resolves TC2_RESOLVE indices");
__device__ __forceinline__ void tc2_resolve_rows(int* sel_s, unsigned n_valid, unsigned tid,
                                                 const int* __restrict__ block_table,
                                                 unsigned block_size) {
    unsigned row[TC2_RESOLVE];
    #pragma unroll
    for (int i = 0; i < TC2_RESOLVE; i++) {
        const unsigned j = tid + (unsigned)i * TC2_THREADS;
        row[i] = 0u;
        if (j < n_valid) {
            const unsigned t = (unsigned)sel_s[j];
            row[i] = (unsigned)block_table[t / block_size] * block_size + t % block_size;
        }
    }
    #pragma unroll
    for (int i = 0; i < TC2_RESOLVE; i++) {
        const unsigned j = tid + (unsigned)i * TC2_THREADS;
        if (j < n_valid) sel_s[j] = (int)row[i];
    }
}

__device__ __forceinline__ uint4 tc2_load_key(const unsigned char* __restrict__ kv,
                                              unsigned long long off, unsigned lane) {
    if (off == TC2_NO_KEY) return make_uint4(0u, 0u, 0u, 0u);
    return __ldg(reinterpret_cast<const uint4*>(kv + off) + lane);
}

// ---- A1: the bit-placement converter. ----

// 2026-10-08: BF16 2^120 in both halves, and -0.0 in both halves (the FMA addend, so x * 2^120
// + (-0) is x * 2^120 for every x, signed zeros included).
#define TC2_BF16X2_2P120 0x7B807B80u
#define TC2_BF16X2_NEG0 0x80008000u

// 2026-10-08: Two BF16 words for the 4 E4M3 codes of `w` (no code 0x7F / 0xFF), byte k = dim k:
// `lo` = dims 0, 1 and `hi` = dims 2, 3, the lower dim in the low half. A code c = s eeee mmm is
// placed as the BF16 s 0000eeee mmm0000 (sign to bit 15, the 7 magnitude bits shifted left by 4),
// whose value is c's value times 2^-120: a normal code (eeee != 0) gets the exponent field
// eeee + 0 against bias 127 instead of 7, and a subnormal code (eeee == 0, value m * 2^-9)
// becomes the BF16 subnormal m * 2^-129. One BF16 FMA by 2^120 (plus -0) scales both back
// exactly: the product is a BF16 value (|c| <= 448, no rounding) and BF16 FMA takes subnormal
// operands (PTX: no .ftz form for bf16). Zero codes keep their sign (+0 * 2^120 + -0 = +0,
// -0 * 2^120 + -0 = -0), as the old chain does (0x80 -> -0).
__device__ __forceinline__ void tc2_e4m3x4_to_bf16_bits(unsigned w, unsigned& lo, unsigned& hi) {
    const unsigned mag = w & 0x7F7F7F7Fu;
    const unsigned sgn = w ^ mag;
    // 2026-10-08: __byte_perm(x, 0, s): result byte n is byte s[n] of {x, 0} (4..7 select 0).
    // 0x4140 puts bytes 0, 1 at bytes 0, 2; 0x1404 puts bytes 0, 1 at bytes 1, 3 (bits 15, 31).
    const unsigned lo_bits = (__byte_perm(mag, 0u, 0x4140u) << 4) | __byte_perm(sgn, 0u, 0x1404u);
    const unsigned hi_bits = (__byte_perm(mag, 0u, 0x4342u) << 4) | __byte_perm(sgn, 0u, 0x3424u);
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(lo) : "r"(lo_bits), "r"(TC2_BF16X2_2P120),
        "r"(TC2_BF16X2_NEG0));
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(hi) : "r"(hi_bits), "r"(TC2_BF16X2_2P120),
        "r"(TC2_BF16X2_NEG0));
}

// 2026-10-08: The fast path alone over 16 codes (NaN codes give a wrong value here; the caller
// guarantees there are none, and the microtest's code check calls it directly).
__device__ __forceinline__ void tc2_fp8x16_to_bf16_fast(uint4 raw, unsigned (&out)[8]) {
    tc2_e4m3x4_to_bf16_bits(raw.x, out[0], out[1]);
    tc2_e4m3x4_to_bf16_bits(raw.y, out[2], out[3]);
    tc2_e4m3x4_to_bf16_bits(raw.z, out[4], out[5]);
    tc2_e4m3x4_to_bf16_bits(raw.w, out[6], out[7]);
}

// 2026-10-08: A1 for 16 codes: the fast path unless one of them is a NaN code, in which case the
// whole group takes the old chain. A byte's magnitude m <= 0x7F, so m + 1 sets bit 7 of the byte
// exactly when m == 0x7F and never carries into the next byte.
__device__ __forceinline__ void tc2_fp8x16_to_bf16_bits(uint4 raw, unsigned (&out)[8]) {
    const unsigned nan = (((raw.x & 0x7F7F7F7Fu) + 0x01010101u) | ((raw.y & 0x7F7F7F7Fu) + 0x01010101u)
                          | ((raw.z & 0x7F7F7F7Fu) + 0x01010101u)
                          | ((raw.w & 0x7F7F7F7Fu) + 0x01010101u))
                       & 0x80808080u;
    if (nan != 0u) {
        tc2_fp8x16_to_bf16(raw, out);
    } else {
        tc2_fp8x16_to_bf16_fast(raw, out);
    }
}

template <bool kBitCvt>
__device__ __forceinline__ void tc2_convert(uint4 raw, unsigned (&out)[8]) {
    if (kBitCvt) {
        tc2_fp8x16_to_bf16_bits(raw, out);
    } else {
        tc2_fp8x16_to_bf16(raw, out);
    }
}

// 2026-10-08: This thread's 4 key chunks (keys warp + 8 i, lane's 16 bytes) into the key buffer at
// `kb`, the old step (1) verbatim apart from the converter.
template <bool kBitCvt>
__device__ __forceinline__ void tc2_store_tile(unsigned char* kb, const uint4 (&raw)[4],
                                               unsigned warp, unsigned lane) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned pk[8];
        tc2_convert<kBitCvt>(raw[i], pk);
        const unsigned kk = warp + 8u * (unsigned)i;
        *reinterpret_cast<uint4*>(kb + tc2_swz(kk, 2u * lane)) =
            make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *reinterpret_cast<uint4*>(kb + tc2_swz(kk, 2u * lane + 1u)) =
            make_uint4(pk[4], pk[5], pk[6], pk[7]);
    }
}

// 2026-10-08: The old step (3) with the A fragments from registers: S partial over dims
// qd * 128.. for heads mt * 16.., all 32 keys of the buffer at shared address `kb`, stored to the
// quarter's FP32 partials.
__device__ __forceinline__ void tc2_qk_tile(const unsigned (&qf)[8][4], unsigned kb, float* sp_s,
                                            unsigned mt, unsigned qd, unsigned g, unsigned t4,
                                            unsigned b_row, unsigned b_ch) {
    float s_acc[4][4];
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) s_acc[nt][e] = 0.0f;
    }
    #pragma unroll
    for (int ks = 0; ks < 8; ks++) {
        const unsigned kc = qd * 16u + 2u * (unsigned)ks;
        #pragma unroll
        for (int np = 0; np < 2; np++) {
            unsigned b0, b1, b2, b3;
            tc2_ldsm_x4(kb + tc2_swz((unsigned)np * 16u + b_row, kc + b_ch), b0, b1, b2, b3);
            tc2_mma_bf16(s_acc[2 * np], qf[ks], b0, b1);
            tc2_mma_bf16(s_acc[2 * np + 1], qf[ks], b2, b3);
        }
    }
    float* sp = sp_s + qd * (TC2_HEADS * TC2_SP_LD);
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        const unsigned key = (unsigned)nt * 8u + 2u * t4;
        *reinterpret_cast<float2*>(sp + (mt * 16u + g) * TC2_SP_LD + key) =
            make_float2(s_acc[nt][0], s_acc[nt][1]);
        *reinterpret_cast<float2*>(sp + (mt * 16u + g + 8u) * TC2_SP_LD + key) =
            make_float2(s_acc[nt][2], s_acc[nt][3]);
    }
}

// 2026-10-09: kPf3 (A7, microtest arm `_tc2_pf3_fp8`): key loads issued three tiles ahead instead
// of two (a second register set raw1); the same bytes reach the same stores, so the output bits
// are unchanged.
template <bool kBitCvt, bool kPf3 = false>
__device__ __forceinline__ void tc2_body(
    unsigned char* tc2_smem,
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
    unsigned char* k0_s = tc2_smem + TC2_OFF_K0;
    unsigned char* k1_s = tc2_smem + TC2_OFF_K1;
    float* sp_s         = reinterpret_cast<float*>(tc2_smem + TC2_OFF_SP);
    unsigned char* p_s  = tc2_smem + TC2_OFF_P;
    int* sel_s          = reinterpret_cast<int*>(tc2_smem + TC2_OFF_SEL);
    float* alpha_s      = reinterpret_cast<float*>(tc2_smem + TC2_OFF_ALPHA);
    float* linv_s       = reinterpret_cast<float*>(tc2_smem + TC2_OFF_LINV);
    int* cnt_s          = reinterpret_cast<int*>(tc2_smem + TC2_OFF_CNT);

    const unsigned head0 = blockIdx.x * TC2_HEADS;
    const unsigned row   = blockIdx.y;
    const unsigned tid   = threadIdx.x;
    const unsigned warp  = tid / TC2_WARP_SIZE;
    const unsigned lane  = tid % TC2_WARP_SIZE;

    if (head0 + TC2_HEADS > num_q_heads) return;

    const unsigned seq_len = (unsigned)seq_lens[row];
    if (seq_len == 0) return;

    const int* my_block_table = block_tables + (size_t)row * max_blocks_per_seq;
    const int* my_sel         = sel_indices  + (size_t)row * sel_width;
    const unsigned long long q_off = ((unsigned long long)row * num_q_heads + head0) * TC2_D;

    // 2026-10-08: The 32 heads' q rows into the second key buffer, swizzled as the old Q tile.
    {
        const uint4* q4 = reinterpret_cast<const uint4*>(Q + q_off);
        #pragma unroll
        for (int i = 0; i < TC2_HEADS * TC2_D * 2 / 16 / TC2_THREADS; i++) {
            const unsigned c = tid + (unsigned)i * TC2_THREADS;
            *reinterpret_cast<uint4*>(k1_s + tc2_swz(c >> 6, c & 63u)) = q4[c];
        }
    }

    // 2026-10-08: Stable compaction, verbatim; its barriers also publish the staged Q.
    unsigned n_valid = 0;
    for (unsigned j0 = 0; j0 < sel_width; j0 += TC2_THREADS) {
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
        for (unsigned w = 0; w < TC2_WARPS; w++) {
            const unsigned c = (unsigned)cnt_s[w];
            if (w < warp) base += c;
            total += c;
        }
        if (ok) sel_s[base + __popc(ball & ((1u << lane) - 1u))] = t;
        n_valid += total;
        __syncthreads();
    }
    // 2026-10-08: The old kernel publishes Q with the compaction's last barrier; a selection of
    // width 0 runs no pass, so one more barrier keeps the staged Q published in that case too.
    if (sel_width == 0) __syncthreads();

    // 2026-10-09: A6: token indices -> cache rows; the barrier publishes them.
    tc2_resolve_rows(sel_s, n_valid, tid, my_block_table, block_size);
    __syncthreads();

    const unsigned n_tiles = (n_valid + TC2_NK - 1) / TC2_NK;

    // 2026-10-08: Tile 0's bytes and tile 1's offsets (A6: from the rows).
    uint4 raw[4];
    unsigned long long off_nxt[4];
    unsigned row_nxt[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned kk = warp + 8u * (unsigned)i;
        raw[i] = tc2_load_key(KV_cache, tc2_row_off(tc2_key_row(sel_s, kk, n_valid)), lane);
        off_nxt[i] = tc2_row_off(tc2_key_row(sel_s, TC2_NK + kk, n_valid));
    }

    const unsigned mt     = warp & 1u;
    const unsigned qd     = warp >> 1;
    const unsigned g      = lane >> 2;
    const unsigned t4     = lane & 3u;
    const unsigned sm_row = tid >> 3;
    const unsigned sm_c4  = (tid & 7u) * 4u;
    const unsigned a_row = (lane & 7u) + ((lane >> 3) & 1u) * 8u;
    const unsigned a_ch  = lane >> 4;
    const unsigned b_row = (lane & 7u) + ((lane >> 4) & 1u) * 8u;
    const unsigned b_ch  = (lane >> 3) & 1u;

    // 2026-10-08: A2: this warp's 8 Q A-fragments (heads mt * 16.., dims qd * 128..), the same
    // ldmatrix.x4 addresses the old kernel issues every tile.
    unsigned qf[8][4];
    #pragma unroll
    for (int ks = 0; ks < 8; ks++) {
        const unsigned kc = qd * 16u + 2u * (unsigned)ks;
        tc2_ldsm_x4(tc2_smem_addr(k1_s + tc2_swz(mt * 16u + a_row, kc + a_ch)), qf[ks][0],
                    qf[ks][1], qf[ks][2], qf[ks][3]);
    }

    const float qk_scale = kv_scale * inv_sqrt_d * 1.4426950408889634f;
    float m_run = tc2_neg_inf();
    float l_run = 0.0f;

    float o_acc[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) o_acc[nt][e] = 0.0f;
    }

    // 2026-10-08: Tile 0 into buffer 0, tile 1's loads, tile 2's offsets (A7: tile 2's loads
    // into raw1 and tile 3's rows).
    tc2_store_tile<kBitCvt>(k0_s, raw, warp, lane);
    #pragma unroll
    for (int i = 0; i < 4; i++) raw[i] = tc2_load_key(KV_cache, off_nxt[i], lane);
    uint4 raw1[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned kk = warp + 8u * (unsigned)i;
        if (kPf3) {
            raw1[i] = tc2_load_key(KV_cache, tc2_row_off(tc2_key_row(sel_s, 2u * TC2_NK + kk,
                                                                     n_valid)), lane);
            row_nxt[i] = tc2_key_row(sel_s, 3u * TC2_NK + kk, n_valid);
        } else {
            row_nxt[i] = tc2_key_row(sel_s, 2u * TC2_NK + kk, n_valid);
        }
    }
    // 2026-10-08: Publishes buffer 0; every warp has read its Q fragments out of buffer 1.
    __syncthreads();

    const unsigned kb0 = tc2_smem_addr(k0_s);
    tc2_qk_tile(qf, kb0, sp_s, mt, qd, g, t4, b_row, b_ch);

    for (unsigned tile = 0; tile < n_tiles; tile++) {
        const unsigned kcur = kb0 + ((tile & 1u) << 15);
        const unsigned nbuf = (tile + 1u) & 1u;
        const unsigned knxt = kb0 + (nbuf << 15);

        // 2026-10-08: B1. S partials of this tile published; the previous PV is done with the
        // other buffer, P and alpha.
        __syncthreads();

        // 2026-10-08: Tile + 1's bytes (loaded one tile ago) into the other buffer, then the
        // loads for tile + 2 and the row reads for tile + 3 (A6). Past the list the offsets
        // are TC2_NO_KEY and the loads return zeros without touching memory.
        tc2_store_tile<kBitCvt>(tc2_smem + (nbuf << 15), raw, warp, lane);
        if (kPf3) {
            // 2026-10-09: A7: tile + 2's bytes move up, tile + 3's loads, tile + 4's rows.
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                raw[i] = raw1[i];
                raw1[i] = tc2_load_key(KV_cache, tc2_row_off(row_nxt[i]), lane);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                raw[i] = tc2_load_key(KV_cache, tc2_row_off(row_nxt[i]), lane);
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            row_nxt[i] = tc2_key_row(sel_s, (tile + (kPf3 ? 4u : 3u)) * TC2_NK + warp
                                                + 8u * (unsigned)i, n_valid);
        }

        // 2026-10-08: Online softmax, verbatim (old step 4).
        {
            float s[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) s[e] = 0.0f;
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                const float4 v = *reinterpret_cast<const float4*>(
                    sp_s + q * (TC2_HEADS * TC2_SP_LD) + sm_row * TC2_SP_LD + sm_c4);
                s[0] += v.x;
                s[1] += v.y;
                s[2] += v.z;
                s[3] += v.w;
            }
            const unsigned key0 = tile * TC2_NK + sm_c4;
            float mx = tc2_neg_inf();
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                s[e] = (key0 + (unsigned)e < n_valid) ? s[e] * qk_scale : tc2_neg_inf();
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
            *reinterpret_cast<uint2*>(p_s + (sm_row * TC2_PS_LD + sm_c4) * 2u) =
                make_uint2(tc2_pack_bf16(p[0], p[1]), tc2_pack_bf16(p[2], p[3]));
            if ((tid & 7u) == 0u) alpha_s[sm_row] = alpha;
        }

        // 2026-10-08: B2. P, alpha and tile + 1's buffer published; softmax is done with the S
        // partials.
        __syncthreads();

        // 2026-10-08: O = O * alpha + P V over this tile's buffer (old step 5, verbatim apart from
        // the buffer address).
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
                tc2_ldsm_x4(tc2_smem_addr(p_s) + (mt * 16u + a_row) * (TC2_PS_LD * 2u)
                                + (2u * (unsigned)ks + a_ch) * 16u,
                            a[0], a[1], a[2], a[3]);
                #pragma unroll
                for (int np = 0; np < 8; np++) {
                    unsigned b0, b1, b2, b3;
                    tc2_ldsm_x4_t(kcur + tc2_swz((unsigned)ks * 16u + a_row,
                                                 qd * 16u + 2u * (unsigned)np + a_ch),
                                  b0, b1, b2, b3);
                    tc2_mma_bf16(o_acc[2 * np], a, b0, b1);
                    tc2_mma_bf16(o_acc[2 * np + 1], a, b2, b3);
                }
            }
        }

        // 2026-10-08: S partials of tile + 1 from the other buffer (unused after the last tile).
        tc2_qk_tile(qf, knxt, sp_s, mt, qd, g, t4, b_row, b_ch);
    }

    // 2026-10-08: O * v_scale / l, verbatim. linv_s is not touched inside the loop, so warps still
    // finishing the last tile do not race this store.
    if ((tid & 7u) == 0u) linv_s[sm_row] = (l_run > 0.0f) ? (kv_scale / l_run) : 0.0f;
    __syncthreads();
    {
        const float s_lo = linv_s[mt * 16u + g];
        const float s_hi = linv_s[mt * 16u + g + 8u];
        unsigned* o_lo = reinterpret_cast<unsigned*>(
            O + q_off + (unsigned long long)(mt * 16u + g) * TC2_D);
        unsigned* o_hi = reinterpret_cast<unsigned*>(
            O + q_off + (unsigned long long)(mt * 16u + g + 8u) * TC2_D);
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            const unsigned d2 = (qd * 128u + (unsigned)nt * 8u + 2u * t4) / 2u;
            o_lo[d2] = tc2_pack_bf16(o_acc[nt][0] * s_lo, o_acc[nt][1] * s_lo);
            o_hi[d2] = tc2_pack_bf16(o_acc[nt][2] * s_hi, o_acc[nt][3] * s_hi);
        }
    }
}

// 2026-10-08: A1 + A2 + A3 (the entry METRALE_GLM_MLA_PREFILL_TC2=1 launches).
extern "C" __global__ void __launch_bounds__(TC2_THREADS, 1) glm5next_dsa_mla_prefill_tc2_fp8(
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
    extern __shared__ __align__(16) unsigned char tc2_smem_a[];
    tc2_body<true>(tc2_smem_a, Q, KV_cache, O, block_tables, seq_lens, sel_indices, sel_width,
                   max_blocks_per_seq, num_q_heads, block_size, inv_sqrt_d, kv_scale,
                   cache_stride_bytes);
}

// 2026-10-08: A2 + A3 with the old converter (microtest timing arm; same output bits).
extern "C" __global__ void __launch_bounds__(TC2_THREADS, 1) glm5next_dsa_mla_prefill_tc2_hwcvt_fp8(
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
    extern __shared__ __align__(16) unsigned char tc2_smem_b[];
    tc2_body<false>(tc2_smem_b, Q, KV_cache, O, block_tables, seq_lens, sel_indices, sel_width,
                    max_blocks_per_seq, num_q_heads, block_size, inv_sqrt_d, kv_scale,
                    cache_stride_bytes);
}

// 2026-10-09: A7 microtest arm: the production entry with key loads three tiles ahead (kPf3).
extern "C" __global__ void __launch_bounds__(TC2_THREADS, 1) glm5next_dsa_mla_prefill_tc2_pf3_fp8(
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
    extern __shared__ __align__(16) unsigned char tc2_smem_c[];
    tc2_body<true, true>(tc2_smem_c, Q, KV_cache, O, block_tables, seq_lens, sel_indices,
                         sel_width, max_blocks_per_seq, num_q_heads, block_size, inv_sqrt_d,
                         kv_scale, cache_stride_bytes);
}

// 2026-10-08: Converter check for the microtest, one block of 16 threads: thread i converts the
// 16 codes 16 i .. 16 i + 15 (byte j = code 16 i + j) with the old chain (out[0..128)), the A1
// converter (out[128..256)) and the A1 fast path alone (out[256..384)), 8 words each.
extern "C" __global__ void glm5next_dsa_mla_prefill_tc2_cvt_check(unsigned* __restrict__ out) {
    const unsigned i = threadIdx.x;
    if (i >= 16u) return;
    unsigned w[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const unsigned c = 16u * i + 4u * (unsigned)k;
        w[k] = c | ((c + 1u) << 8) | ((c + 2u) << 16) | ((c + 3u) << 24);
    }
    const uint4 raw = make_uint4(w[0], w[1], w[2], w[3]);
    unsigned a[8], b[8], c[8];
    tc2_fp8x16_to_bf16(raw, a);
    tc2_fp8x16_to_bf16_bits(raw, b);
    tc2_fp8x16_to_bf16_fast(raw, c);
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[8u * i + k] = a[k];
        out[128u + 8u * i + k] = b[k];
        out[256u + 8u * i + k] = c[k];
    }
}

// 2026-10-08: L2 eviction for the microtest's cold-L2 timing: reads `n16` 16-byte chunks of `src`
// grid-stride (plain loads, which allocate in L2) and folds them into one word, stored only when
// it equals an arbitrary constant so the loads cannot be dropped. Read-only, so the lines it
// leaves in L2 are clean and the next kernel pays no write-back for them.
extern "C" __global__ void glm5next_dsa_mla_prefill_tc2_l2_flush(const uint4* __restrict__ src,
                                                                 const unsigned long long n16,
                                                                 unsigned* __restrict__ sink) {
    unsigned acc = 0u;
    const unsigned long long step = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16;
         i += step) {
        const uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9E3779B9u) sink[0] = acc;
}
