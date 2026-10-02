// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-01: Chunked KDA (Kimi Delta Attention) prefill on tensor cores, the opt-in
// `METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1` arm (crates/model-arch/src/glm5next_kda/prefill_tc.rs).
// Four kernels, launched in this order on one stream:
//   1. kda_tc_conv_rows: conv + SiLU + L2 for every row at once, one block per (256 channels,
//      KDA_TC_CONV_ROWS rows). It reads the conv state and does not write it.
//   2. kda_tc_conv_state_tail: the conv state after the last row.
//   3. kda_tc_prepare: everything that does not depend on the recurrent state, one block per
//      (chunk of 16 rows, head), in FP32.
//   4. kda_tc_scan: the recurrent state carried across chunks, one warp per (head, 16 V
//      columns), with BF16 m16n8k16 MMAs and FP32 accumulation.
//
// Owner: gb10 kernels.
// Invariants:
// - head_dim is the compile-time KDA_TC_D = 128 and the chunk the compile-time KDA_TC_C = 16;
//   the host refuses any other geometry before a launch.
// - Numerics differ from the per-token recurrence (kda_recurrent.cu): sums run in another
//   order and the MMA operands are rounded to BF16 (the state as a BF16 hi + lo pair). Only
//   kda_tc_conv_rows and kda_tc_conv_state_tail are bit-identical to their token-loop twin
//   causal_conv1d_update_l2norm_rows; kda_chunk_tc_microtest checks both claims.
// - A position t >= T contributes nothing: prepare reads q, k, v, gate and beta there as zero,
//   so its rows of u, W and K' are zero and it moves no state; scan never writes its out row.
//
// The math, per head, over a chunk of C = 16 positions (i the query row, j the key row, d a
// channel, S the [k][v] state at the chunk start; gate is a log-decay, applied before each
// token's delta-rule update, as in kda_recurrent_prefill_bf16_smem):
//   gc[i,d]  = sum_{p<=i} gate[p,d]
//   L[i,j]   = beta[i] sum_d k[i,d] k[j,d] exp(gc[i,d] - gc[j,d])            for j < i
//   Tb       = (I + L)^-1 diag(beta)
//   u        = Tb v,   W = Tb (k * exp(gc))
//   Q'[i,d]  = scale q[i,d] exp(gc[i,d]),   K'[i,d] = k[i,d] exp(gc[C-1,d] - gc[i,d])
//   M[i,j]   = scale sum_d q[i,d] k[j,d] exp(gc[i,d] - gc[j,d])               for j <= i
//   v_new    = u - W S
//   out      = Q' S + M v_new
//   S       <- diag(exp(gc[C-1])) S + K'^T v_new
// This is kda_chunk.cu's formulation at C = 16 instead of 32.
//
// Why C = 16 (the FlashKDA v1 choice; see THIRD_PARTY_NOTICES.md, design reference only, no
// code taken): with GLM's gate_lower_bound = -5 every gate is in (-5, 0), so within 16 rows
// |gc[i] - gc[j]| < 80 and exp() of it stays inside the FP32/BF16 exponent range. prepare
// factors exp(gc[i] - gc[j]) through the chunk's row 7 (KDA_TC_REF), which halves that range
// to < 45, so the host admits any lower bound >= -9. Every operand that reaches an MMA is
// bounded by |q|, |k|, |u| or 1 (exp of a non-positive number), so BF16 rounding there costs
// relative precision only.
//
// The scan keeps each warp's 16 V columns of S transposed (St[v][k], 16 x 128 FP32) in the
// MMA accumulator layout, which is also the layout of an m16n8k16 A operand, so St feeds
// St @ W^T and St @ Q'^T and takes v_new^T @ K' without a shared-memory round trip. St is split
// into BF16 hi + lo for the two products against it, and v_new likewise for the two products
// against it, so those MMAs see ~16 mantissa bits of the FP32 values.
//
// Records prepare writes for chunk c, head h, at rec = c * H + h (separate buffers so each one
// fits a workspace buffer the token loop does not use; see prefill_tc.rs):
//   qw  [rec][2][16][128] BF16: Q' then W, row-major.
//   kpt [rec][128][16]    BF16: K' transposed.
//   u   [rec][16][128]    FP32.
//   md  [rec][256]        FP32 words: M as BF16 [16][16] (masked to j <= i) in words 0..127,
//                         exp(gc[C-1]) [128] in words 128..255.

#include <cuda_bf16.h>
#include <math.h>

#define KDA_TC_D 128u
#define KDA_TC_C 16u
// 2026-10-01: The reference row prepare factors exp(gc[i] - gc[j]) through.
#define KDA_TC_REF 7u
// 2026-10-01: Rows per kda_tc_conv_rows block.
#define KDA_TC_CONV_ROWS 16u
// 2026-10-01: Warps per kda_tc_scan block; each owns 16 V columns, so grid.y = D / (16 * W).
#define KDA_TC_SCAN_WARPS 2u

// 2026-10-01: kda_tc_scan's double-buffered stage, in bytes. Rows are padded so the MMA
// fragment loads (8 rows x 4 words per warp) hit distinct banks: Q' and W rows are 128 + 8
// BF16, K'^T and M rows 16 + 8 BF16.
#define KDA_TC_QW_ROW 272u
#define KDA_TC_KM_ROW 48u
#define KDA_TC_ST_Q 0u
#define KDA_TC_ST_W (KDA_TC_ST_Q + 16u * KDA_TC_QW_ROW)
#define KDA_TC_ST_K (KDA_TC_ST_W + 16u * KDA_TC_QW_ROW)
#define KDA_TC_ST_M (KDA_TC_ST_K + 128u * KDA_TC_KM_ROW)
#define KDA_TC_ST_DEC (KDA_TC_ST_M + 16u * KDA_TC_KM_ROW)
#define KDA_TC_STAGE (KDA_TC_ST_DEC + 512u)
// 2026-10-01: 16-byte copies per stage: Q' 256, W 256, K'^T 256, M 32, decay 32.
#define KDA_TC_STAGE_CHUNKS 832u

namespace kdatc {
__device__ __forceinline__ unsigned sa(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa(dst)), "l"(src));
}
__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]),
          "f"(c[2]), "f"(c[3]));
}
// 2026-10-01: Two consecutive BF16 from shared memory as one MMA operand register (the lower
// address in the low half).
__device__ __forceinline__ unsigned ld2(const __nv_bfloat16* p) {
    return *reinterpret_cast<const unsigned*>(p);
}
__device__ __forceinline__ unsigned pack(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (unsigned)__bfloat16_as_ushort(lo) | ((unsigned)__bfloat16_as_ushort(hi) << 16);
}
// 2026-10-01: x ~= hi + lo with both BF16: hi = bf16(x), lo = bf16(x - hi). Packs the pair
// (x0, x1) into one hi and one lo operand register.
__device__ __forceinline__ void split2(float x0, float x1, unsigned& hi, unsigned& lo) {
    const __nv_bfloat16 h0 = __float2bfloat16_rn(x0);
    const __nv_bfloat16 h1 = __float2bfloat16_rn(x1);
    const __nv_bfloat16 l0 = __float2bfloat16_rn(x0 - __bfloat162float(h0));
    const __nv_bfloat16 l1 = __float2bfloat16_rn(x1 - __bfloat162float(h1));
    hi = pack(h0, h1);
    lo = pack(l0, l1);
}
}  // namespace kdatc

// 2026-10-01: kda_tc_conv_rows: the outputs of causal_conv1d_update_l2norm_rows
// (causal_conv1d.cu) for `rows` rows of one sequence, with the rows spread over grid.y instead
// of walked by one thread. Same arguments; grid (ceil(dim / 256), ceil(rows /
// KDA_TC_CONV_ROWS)), block 256; the same L2 contract (256 threads, head_dim 128, qk_channels a
// multiple of 256); d_conv <= 8.
//
// Row t's window is the inputs t - d_conv + 1 .. t, with the positions before row 0 taken from
// the incoming conv state (slot d_conv + s for position s < 0; slot 0 is shifted out unread, as
// in the row kernel). Those are the FP32 values the row kernel's shifted window holds at row t,
// and every later operation is transcribed from it in the same order (bias or 0, the serial sum
// over k, SiLU with __expf, the __shfl_down tree, the four warp partials in warp order,
// rsqrtf), under the tree's --fmad=false, so each output is the same bits. The conv state is
// read only; kda_tc_conv_state_tail writes it after this kernel.
extern "C" __global__ void kda_tc_conv_rows(
    const float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);
    const bool valid = (ch < dim);

    float wcoef[8];
    if (valid) {
        const __nv_bfloat16* w = weight + ch * d_conv;
        for (unsigned int k = 0; k < d_conv; k++) wcoef[k] = (float)w[k];
    }

    __shared__ float warp_sums[8];

    const unsigned int r0 = blockIdx.y * KDA_TC_CONV_ROWS;
    const unsigned int r1 = min(rows, r0 + KDA_TC_CONV_ROWS);
    for (unsigned int t = r0; t < r1; t++) {
        float silu = 0.0f;

        if (valid) {
            float win[8];
            for (unsigned int i = 0; i < d_conv; i++) {
                const int s = (int)t - (int)(d_conv - 1u) + (int)i;
                win[i] = s >= 0 ? (float)new_input[(size_t)s * input_stride + ch]
                                : conv_state[ch * d_conv + (unsigned int)((int)d_conv + s)];
            }

            float acc = (bias != nullptr) ? bias[ch] : 0.0f;
            for (unsigned int k = 0; k < d_conv; k++)
                acc += win[k] * wcoef[k];

            float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
            silu = acc * sigmoid_acc;
        }

        if (block_needs_l2) {
            float sq = valid ? (silu * silu) : 0.0f;

            const unsigned int warp_id = tid / 32;
            const unsigned int lane = tid % 32;
            for (int offset = 16; offset >= 1; offset >>= 1)
                sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);

            if (lane == 0) warp_sums[warp_id] = sq;
            __syncthreads();

            const unsigned int head_in_block = tid / head_dim;
            const unsigned int base_warp = head_in_block * (head_dim / 32);

            if (tid == 0 || tid == head_dim) {
                float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                            + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
                warp_sums[base_warp] = rsqrtf(total + l2_eps);
            }
            __syncthreads();

            if (valid) {
                silu *= warp_sums[base_warp];
            }
            // 2026-10-01: Keeps the next row's lane-0 write to warp_sums behind this read.
            __syncthreads();
        }

        if (valid) {
            output[(size_t)t * output_stride + ch] = __float2bfloat16(silu);
        }
    }
}

// 2026-10-01: kda_tc_conv_state_tail: the conv state causal_conv1d_update_l2norm_rows leaves
// after `rows` rows: slot i holds position rows - d_conv + i, an input row widened to FP32 when
// that position is >= 0, otherwise the incoming slot rows + i. Each thread owns one channel's
// slots and writes them in ascending order, reading only slots above the one it writes, so the
// update is safe in place. Grid (ceil(dim / 256)), block 256. Launch after kda_tc_conv_rows,
// which reads the incoming state.
extern "C" __global__ void kda_tc_conv_state_tail(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    unsigned int rows,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int input_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    float* state = conv_state + ch * d_conv;
    for (unsigned int i = 0; i < d_conv; i++) {
        const int s = (int)rows - (int)d_conv + (int)i;
        state[i] = s >= 0 ? (float)new_input[(size_t)s * input_stride + ch] : state[rows + i];
    }
}

// 2026-10-01: kda_tc_prepare: the state-independent half of one chunk, in FP32. Grid (num_chunks,
// H), block KDA_TC_D (thread d owns channel d). q, k and v are the L2-normalised BF16 conv
// outputs (row stride qkv_row_stride elements), gate the FP32 log-decay (row stride
// gate_row_stride), beta FP32 (row stride beta_row_stride); head h's channels start at h * D.
// Writes the record described at the top of this file.
//
// L and M come from the factored form exp(gc[i] - gc[j]) = exp(gc[i] - gc[r]) exp(gc[r] - gc[j])
// with r = KDA_TC_REF: two exps per (row, channel) instead of one per (i, j, channel). The
// inverse is a forward substitution down each column, one thread per column, column entries in
// registers.
extern "C" __global__ void __launch_bounds__(128) kda_tc_prepare(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ qw_out,
    __nv_bfloat16* __restrict__ kpt_out,
    float* __restrict__ u_out,
    float* __restrict__ md_out,
    unsigned int H,
    unsigned int T,
    unsigned int qkv_row_stride,
    unsigned int gate_row_stride,
    unsigned int beta_row_stride,
    float scale
) {
    // 2026-10-01: Rows of kp / kn / qp padded to D + 1 floats: the dot products below read
    // kn[j][d] for 16 values of j at once.
    __shared__ float kp[KDA_TC_C][KDA_TC_D + 1];
    __shared__ float kn[KDA_TC_C][KDA_TC_D + 1];
    __shared__ float qp[KDA_TC_C][KDA_TC_D + 1];
    __shared__ float sh_l[KDA_TC_C][KDA_TC_C + 1];
    __shared__ float sh_m[KDA_TC_C][KDA_TC_C + 1];
    __shared__ float sh_t[KDA_TC_C][KDA_TC_C + 1];
    __shared__ float sh_beta[KDA_TC_C];

    const unsigned int c = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const size_t hd = (size_t)h * KDA_TC_D;
    const unsigned int rec = c * H + h;

    if (d < KDA_TC_C) {
        const unsigned int t = c * KDA_TC_C + d;
        sh_beta[d] = t < T ? beta[(size_t)t * beta_row_stride + h] : 0.0f;
    }

    float gc[KDA_TC_C], kf[KDA_TC_C], vf[KDA_TC_C], qf[KDA_TC_C];
    float acc = 0.0f;
    #pragma unroll
    for (unsigned int i = 0; i < KDA_TC_C; ++i) {
        const unsigned int t = c * KDA_TC_C + i;
        const bool live = t < T;
        const size_t qo = (size_t)t * qkv_row_stride + hd + d;
        acc += live ? gate[(size_t)t * gate_row_stride + hd + d] : 0.0f;
        gc[i] = acc;
        kf[i] = live ? __bfloat162float(k[qo]) : 0.0f;
        vf[i] = live ? __bfloat162float(v[qo]) : 0.0f;
        qf[i] = live ? __bfloat162float(q[qo]) : 0.0f;
    }

    const float gr = gc[KDA_TC_REF];
    #pragma unroll
    for (unsigned int i = 0; i < KDA_TC_C; ++i) {
        const float ep = expf(gc[i] - gr);
        const float en = expf(gr - gc[i]);
        kp[i][d] = kf[i] * ep;
        kn[i][d] = kf[i] * en;
        qp[i][d] = qf[i] * ep;
    }
    __syncthreads();

    // 2026-10-01: L (strictly lower, beta applied) and M (lower with the diagonal, scale
    // applied): 256 entries, two per thread.
    for (unsigned int p = d; p < KDA_TC_C * KDA_TC_C; p += blockDim.x) {
        const unsigned int i = p / KDA_TC_C, j = p % KDA_TC_C;
        float dk = 0.0f, dq = 0.0f;
        if (j <= i) {
            for (unsigned int e = 0; e < KDA_TC_D; ++e) {
                const float b = kn[j][e];
                dk += kp[i][e] * b;
                dq += qp[i][e] * b;
            }
        }
        sh_l[i][j] = j < i ? sh_beta[i] * dk : 0.0f;
        sh_m[i][j] = j <= i ? scale * dq : 0.0f;
    }
    __syncthreads();

    // 2026-10-01: Column j of (I + L)^-1: col[i] = -sum_{m<i} L[i][m] col[m] for i > j, 1 at
    // i = j, 0 above. Entries m < j of col are zero, so the sum over all m < i is the sum over
    // j <= m < i. Then column j of Tb = column j * beta[j].
    if (d < KDA_TC_C) {
        const unsigned int j = d;
        float col[KDA_TC_C];
        #pragma unroll
        for (unsigned int m = 0; m < KDA_TC_C; ++m) col[m] = (m == j) ? 1.0f : 0.0f;
        #pragma unroll
        for (unsigned int i = 1; i < KDA_TC_C; ++i) {
            float s = 0.0f;
            #pragma unroll
            for (unsigned int m = 0; m < i; ++m) s += sh_l[i][m] * col[m];
            if (i > j) col[i] = -s;
        }
        const float bj = sh_beta[j];
        #pragma unroll
        for (unsigned int i = 0; i < KDA_TC_C; ++i) sh_t[i][j] = col[i] * bj;
    }
    __syncthreads();

    // 2026-10-01: u = Tb v, W = Tb (k * exp(gc)), Q', K' and the decay, channel d of each row.
    float kg[KDA_TC_C];
    #pragma unroll
    for (unsigned int j = 0; j < KDA_TC_C; ++j) kg[j] = kf[j] * expf(gc[j]);
    const float glast = gc[KDA_TC_C - 1];
    __nv_bfloat16* q_rec = qw_out + (size_t)rec * (2u * KDA_TC_C * KDA_TC_D);
    __nv_bfloat16* w_rec = q_rec + KDA_TC_C * KDA_TC_D;
    float* u_rec = u_out + (size_t)rec * (KDA_TC_C * KDA_TC_D);
    __nv_bfloat16 kcol[KDA_TC_C];
    #pragma unroll
    for (unsigned int i = 0; i < KDA_TC_C; ++i) {
        float au = 0.0f, aw = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j <= i; ++j) {
            const float tb = sh_t[i][j];
            au += tb * vf[j];
            aw += tb * kg[j];
        }
        u_rec[i * KDA_TC_D + d] = au;
        w_rec[i * KDA_TC_D + d] = __float2bfloat16_rn(aw);
        q_rec[i * KDA_TC_D + d] = __float2bfloat16_rn(scale * qf[i] * expf(gc[i]));
        kcol[i] = __float2bfloat16_rn(kf[i] * expf(glast - gc[i]));
    }
    __nv_bfloat16* k_rec = kpt_out + (size_t)rec * (KDA_TC_D * KDA_TC_C) + (size_t)d * KDA_TC_C;
    #pragma unroll
    for (unsigned int i = 0; i < KDA_TC_C; ++i) k_rec[i] = kcol[i];

    float* md = md_out + (size_t)rec * 256u;
    md[128u + d] = expf(glast);
    __nv_bfloat16* m_rec = reinterpret_cast<__nv_bfloat16*>(md);
    for (unsigned int p = d; p < KDA_TC_C * KDA_TC_C; p += blockDim.x) {
        m_rec[p] = __float2bfloat16_rn(sh_m[p / KDA_TC_C][p % KDA_TC_C]);
    }
}

// 2026-10-01: Stage chunk record `rec` into shared memory with 16-byte cp.async copies, every
// thread of the block taking a share; the caller commits the group.
__device__ __forceinline__ void kda_tc_stage(
    unsigned char* st,
    const unsigned char* qw,
    const unsigned char* kpt,
    const unsigned char* md,
    unsigned int rec
) {
    const unsigned char* gq = qw + (size_t)rec * 8192u;
    const unsigned char* gk = kpt + (size_t)rec * 4096u;
    const unsigned char* gm = md + (size_t)rec * 1024u;
    for (unsigned int x = threadIdx.x; x < KDA_TC_STAGE_CHUNKS; x += blockDim.x) {
        if (x < 256u) {
            const unsigned int r = x >> 4, ch = x & 15u;
            kdatc::cp16(st + KDA_TC_ST_Q + r * KDA_TC_QW_ROW + ch * 16u, gq + r * 256u + ch * 16u);
        } else if (x < 512u) {
            const unsigned int y = x - 256u, r = y >> 4, ch = y & 15u;
            kdatc::cp16(st + KDA_TC_ST_W + r * KDA_TC_QW_ROW + ch * 16u,
                        gq + 4096u + r * 256u + ch * 16u);
        } else if (x < 768u) {
            const unsigned int y = x - 512u, r = y >> 1, ch = y & 1u;
            kdatc::cp16(st + KDA_TC_ST_K + r * KDA_TC_KM_ROW + ch * 16u, gk + r * 32u + ch * 16u);
        } else if (x < 800u) {
            const unsigned int y = x - 768u, r = y >> 1, ch = y & 1u;
            kdatc::cp16(st + KDA_TC_ST_M + r * KDA_TC_KM_ROW + ch * 16u, gm + r * 32u + ch * 16u);
        } else {
            const unsigned int y = x - 800u;
            kdatc::cp16(st + KDA_TC_ST_DEC + y * 16u, gm + 512u + y * 16u);
        }
    }
}

// 2026-10-01: kda_tc_scan: the state carried across chunks. Grid (H, D / (16 *
// KDA_TC_SCAN_WARPS)), block 32 * KDA_TC_SCAN_WARPS, dynamic shared memory 2 * KDA_TC_STAGE
// (32,256 B). Warp w of block (h, y) owns V columns v0 = (y * KDA_TC_SCAN_WARPS + w) * 16 ..
// v0 + 15 of head h and walks the chunks in order; warps never exchange data, they only share
// the staged records.
//
// Fragment layout (g = lane / 4, tq = lane % 4): st[nt][0..3] hold St[v][k] at (v0 + g, k),
// (v0 + g, k + 1), (v0 + g + 8, k), (v0 + g + 8, k + 1) with k = nt * 8 + 2 * tq; the products
// over the chunk rows use the same layout with rows in place of k. The state in global memory
// is FP32 [H, D, D], K-major (S[k * D + v]), the layout the decode kernels read; it is read
// before the first chunk and written after the last. out is FP32 with row stride
// out_row_stride.
//
// Per chunk: wait for the stage, one __syncthreads (publishes it and retires the buffer the
// next stage overwrites), issue the next stage, then:
//   P = St (W^T), O = St (Q'^T)   (St as hi + lo, separate accumulators so each MMA chain is
//                                  8 deep)
//   vn = u^T - P;  O += vn (M^T)  (vn as hi + lo);  store O for rows < T
//   St = St * decay[k] + vn K'    (vn as hi + lo)
extern "C" __global__ void __launch_bounds__(32 * KDA_TC_SCAN_WARPS) kda_tc_scan(
    const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kpt,
    const float* __restrict__ u,
    const float* __restrict__ md,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int T,
    unsigned int num_chunks,
    unsigned int out_row_stride
) {
    extern __shared__ __align__(16) unsigned char kda_tc_smem[];
    const unsigned int h = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2, tq = lane & 3u;
    const unsigned int v0 = (blockIdx.y * KDA_TC_SCAN_WARPS + warp) * 16u;
    float* S = state + (size_t)h * KDA_TC_D * KDA_TC_D;

    float st[16][4];
    #pragma unroll
    for (unsigned int nt = 0; nt < 16; ++nt) {
        const unsigned int kk = nt * 8u + 2u * tq;
        st[nt][0] = S[(size_t)kk * KDA_TC_D + v0 + g];
        st[nt][1] = S[(size_t)(kk + 1u) * KDA_TC_D + v0 + g];
        st[nt][2] = S[(size_t)kk * KDA_TC_D + v0 + g + 8u];
        st[nt][3] = S[(size_t)(kk + 1u) * KDA_TC_D + v0 + g + 8u];
    }

    const unsigned char* qwb = reinterpret_cast<const unsigned char*>(qw);
    const unsigned char* kpb = reinterpret_cast<const unsigned char*>(kpt);
    const unsigned char* mdb = reinterpret_cast<const unsigned char*>(md);

    kda_tc_stage(kda_tc_smem, qwb, kpb, mdb, h);
    asm volatile("cp.async.commit_group;\n" ::);

    for (unsigned int c = 0; c < num_chunks; ++c) {
        const unsigned int rec = c * H + h;

        // 2026-10-01: This chunk's u in the vn layout, issued before the wait so the loads
        // overlap it. Rows past T are zero in the record.
        const float* ur = u + (size_t)rec * (KDA_TC_C * KDA_TC_D);
        float uu[2][4];
        #pragma unroll
        for (unsigned int nt = 0; nt < 2; ++nt) {
            const unsigned int r = nt * 8u + 2u * tq;
            uu[nt][0] = ur[r * KDA_TC_D + v0 + g];
            uu[nt][1] = ur[(r + 1u) * KDA_TC_D + v0 + g];
            uu[nt][2] = ur[r * KDA_TC_D + v0 + g + 8u];
            uu[nt][3] = ur[(r + 1u) * KDA_TC_D + v0 + g + 8u];
        }

        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        if (c + 1u < num_chunks) {
            kda_tc_stage(kda_tc_smem + ((c + 1u) & 1u) * KDA_TC_STAGE, qwb, kpb, mdb, rec + H);
            asm volatile("cp.async.commit_group;\n" ::);
        }

        const unsigned char* sb = kda_tc_smem + (c & 1u) * KDA_TC_STAGE;
        const __nv_bfloat16* sq = reinterpret_cast<const __nv_bfloat16*>(sb + KDA_TC_ST_Q);
        const __nv_bfloat16* sw = reinterpret_cast<const __nv_bfloat16*>(sb + KDA_TC_ST_W);
        const __nv_bfloat16* sk = reinterpret_cast<const __nv_bfloat16*>(sb + KDA_TC_ST_K);
        const __nv_bfloat16* sm = reinterpret_cast<const __nv_bfloat16*>(sb + KDA_TC_ST_M);
        const float* sd = reinterpret_cast<const float*>(sb + KDA_TC_ST_DEC);
        // 2026-10-01: Row strides in BF16 elements.
        const unsigned int qwr = KDA_TC_QW_ROW / 2u, kmr = KDA_TC_KM_ROW / 2u;

        float ph[2][4] = {}, pl[2][4] = {}, oh[2][4] = {}, ol[2][4] = {};
        #pragma unroll
        for (unsigned int s = 0; s < KDA_TC_D / 16u; ++s) {
            unsigned int ah[4], al[4];
            kdatc::split2(st[2 * s][0], st[2 * s][1], ah[0], al[0]);
            kdatc::split2(st[2 * s][2], st[2 * s][3], ah[1], al[1]);
            kdatc::split2(st[2 * s + 1][0], st[2 * s + 1][1], ah[2], al[2]);
            kdatc::split2(st[2 * s + 1][2], st[2 * s + 1][3], ah[3], al[3]);
            const unsigned int kk = s * 16u + 2u * tq;
            #pragma unroll
            for (unsigned int nt = 0; nt < 2; ++nt) {
                const unsigned int r = (nt * 8u + g) * qwr + kk;
                const unsigned int bw0 = kdatc::ld2(sw + r), bw1 = kdatc::ld2(sw + r + 8u);
                const unsigned int bq0 = kdatc::ld2(sq + r), bq1 = kdatc::ld2(sq + r + 8u);
                kdatc::mma16816(ph[nt], ah, bw0, bw1);
                kdatc::mma16816(pl[nt], al, bw0, bw1);
                kdatc::mma16816(oh[nt], ah, bq0, bq1);
                kdatc::mma16816(ol[nt], al, bq0, bq1);
            }
        }

        // 2026-10-01: vn = u^T - W S, as the A operand over the chunk rows (hi + lo).
        float vn[2][4], o[2][4];
        #pragma unroll
        for (unsigned int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (unsigned int e = 0; e < 4; ++e) {
                vn[nt][e] = uu[nt][e] - (ph[nt][e] + pl[nt][e]);
                o[nt][e] = oh[nt][e] + ol[nt][e];
            }
        }
        unsigned int vh[4], vl[4];
        kdatc::split2(vn[0][0], vn[0][1], vh[0], vl[0]);
        kdatc::split2(vn[0][2], vn[0][3], vh[1], vl[1]);
        kdatc::split2(vn[1][0], vn[1][1], vh[2], vl[2]);
        kdatc::split2(vn[1][2], vn[1][3], vh[3], vl[3]);

        // 2026-10-01: out^T = O + vn (M^T); B[j][i] = M[i][j].
        #pragma unroll
        for (unsigned int nt = 0; nt < 2; ++nt) {
            const unsigned int r = (nt * 8u + g) * kmr + 2u * tq;
            const unsigned int b0 = kdatc::ld2(sm + r), b1 = kdatc::ld2(sm + r + 8u);
            kdatc::mma16816(o[nt], vh, b0, b1);
            kdatc::mma16816(o[nt], vl, b0, b1);
            #pragma unroll
            for (unsigned int e = 0; e < 4; ++e) {
                const unsigned int t = c * KDA_TC_C + nt * 8u + 2u * tq + (e & 1u);
                const unsigned int vc = v0 + g + ((e & 2u) ? 8u : 0u);
                if (t < T) out[(size_t)t * out_row_stride + (size_t)h * KDA_TC_D + vc] = o[nt][e];
            }
        }

        // 2026-10-01: St = St * decay[k] + vn K'; B[j][k] = K'[j][k] = K'^T[k][j].
        #pragma unroll
        for (unsigned int nt = 0; nt < 16; ++nt) {
            const unsigned int kk = nt * 8u + 2u * tq;
            const float d0 = sd[kk], d1 = sd[kk + 1u];
            st[nt][0] *= d0;
            st[nt][1] *= d1;
            st[nt][2] *= d0;
            st[nt][3] *= d1;
            const unsigned int r = (nt * 8u + g) * kmr + 2u * tq;
            const unsigned int b0 = kdatc::ld2(sk + r), b1 = kdatc::ld2(sk + r + 8u);
            kdatc::mma16816(st[nt], vh, b0, b1);
            kdatc::mma16816(st[nt], vl, b0, b1);
        }
    }

    #pragma unroll
    for (unsigned int nt = 0; nt < 16; ++nt) {
        const unsigned int kk = nt * 8u + 2u * tq;
        S[(size_t)kk * KDA_TC_D + v0 + g] = st[nt][0];
        S[(size_t)(kk + 1u) * KDA_TC_D + v0 + g] = st[nt][1];
        S[(size_t)kk * KDA_TC_D + v0 + g + 8u] = st[nt][2];
        S[(size_t)(kk + 1u) * KDA_TC_D + v0 + g + 8u] = st[nt][3];
    }
}
