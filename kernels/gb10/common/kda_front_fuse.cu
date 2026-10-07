// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: The fused front of the tensor-core chunked KDA prefill
// (METRALE_GLM_KDA_FRONT_FUSE=1, crates/model-arch/src/glm5next_kda/prefill_tc_fuse.rs).
//
// The incumbent chunked-TC prefill (kda_chunk_tc.cu, prefill_tc.rs) runs, per layer call:
//   kda_pack_qkv_bf16      q|k|v projections [3][T][qkv] -> qkv_proj [T][3 qkv] (R+W 6 T qkv B)
//   kda_gate_bf16          g_raw BF16 -> gate FP32                              (R 2, W 4 T qkv B)
//   kda_sigmoid_bf16_f32   beta_bf16 -> beta FP32                               (tiny)
//   kda_tc_conv_rows       qkv_proj -> conv_out (conv + SiLU + L2 on q, k)     (R+W 6 T qkv B)
//   kda_tc_conv_state_tail qkv_proj -> conv state                              (tiny)
//   kda_tc_prepare         conv_out + gate + beta -> chunk records             (R 10 T qkv B)
// This file replaces those six launches with two:
//   kda_ff_prepare         the projections, g_raw and beta_bf16 -> chunk records, computing the
//                          conv, SiLU, L2, gate and beta sigmoid in registers
//   kda_ff_conv_state_tail the conv state after the last row, read from the projections
// so qkv_proj, conv_out, gate and beta are never written or read. kda_tc_scan then consumes
// the records unchanged.
//
// Owner: gb10 kernels.
// Invariants:
// - Byte-identical to the incumbent sequence: every value is the same expression on the same
//   floats in the same order, transcribed from its source kernel, and this file is built with
//   the same flags as theirs (common/KERNEL.toml: --fmad=false, so no multiply-add is
//   contracted in either). Per source:
//   * pack: a copy, so reading the projection at [part][t][col] reads the bits qkv_proj held.
//   * conv (kda_tc_conv_rows): window inputs widened BF16 -> FP32, positions before row 0 from
//     the incoming conv state at slot d_conv + s; acc = bias or 0, then acc += win[k] * w[k]
//     for k ascending; SiLU acc * (1 / (1 + __expf(-acc))); for q and k the sum of squares as
//     the same __shfl_down tree per warp (prepare's thread d is conv_rows' head-local thread
//     d: one head = 4 warps of 32 consecutive channels in both), the four warp partials added
//     in warp order, rsqrtf(total + l2_eps), one multiply; then __float2bfloat16 and back
//     to FP32 (the conv_out store and prepare's load). v takes no L2 (qk_channels is 2 qkv and
//     a multiple of 256, which the host checks, so conv_rows' block_needs_l2 is exactly q|k).
//   * gate (kda_gate_bf16): lower_bound * (1 / (1 + expf(-(expf(A_log[h]) * (g + dt_bias))))),
//     the same kda_gate_scalar body; prepare accumulated the stored FP32, which is this value.
//   * beta (kda_sigmoid_bf16_f32): 1 / (1 + __expf(-x)) on the widened BF16.
//   * prepare (kda_tc_prepare): everything after the loads is copied unchanged.
// - Geometry: head_dim KDA_FF_D = 128, chunk KDA_FF_C = 16, conv taps KDA_FF_DCONV = 4
//   (compile time; the host refuses others). Grid (num_chunks, H), block 128, as kda_tc_prepare.
// - kda_ff_prepare reads the conv state and does not write it; kda_ff_conv_state_tail writes it
//   and must run after kda_ff_prepare (same stream), as kda_tc_conv_state_tail runs after
//   kda_tc_conv_rows.
// - q, k and v are the three [T][qkv] projections, row stride in_row_stride elements (the
//   host's qkv_parts, so in_row_stride = qkv = H * 128); g_raw is [T][H * 128] BF16, beta_raw
//   [T][H] BF16, dt_bias [H * 128] FP32, A_log [H] FP32, conv weight [3 qkv][4] BF16, conv
//   state [3 qkv][4] FP32 (channel order q | k | v, as qkv_proj's columns).

#include <cuda_bf16.h>
#include <math.h>

#define KDA_FF_D 128u
#define KDA_FF_C 16u
// 2026-10-07: KDA_TC_REF in kda_chunk_tc.cu.
#define KDA_FF_REF 7u
#define KDA_FF_DCONV 4u
// 2026-10-07: Window registers per channel: the chunk's rows plus the d_conv - 1 before them.
#define KDA_FF_WIN (KDA_FF_C + KDA_FF_DCONV - 1u)

// 2026-10-07: kda_gate_scalar (kda_gate.cu), the same body.
__device__ __forceinline__ float kda_ff_gate_scalar(float g_raw, float dt_bias,
                                                    float decay, float lower_bound) {
    return lower_bound * (1.0f / (1.0f + expf(-(decay * (g_raw + dt_bias)))));
}

// 2026-10-07: kda_tc_conv_rows' window value at position s for one channel: the input row
// widened to FP32 when s >= 0, otherwise the incoming conv state's slot d_conv + s.
__device__ __forceinline__ float kda_ff_pos(
    const __nv_bfloat16* __restrict__ src,
    const float* __restrict__ conv_state,
    unsigned int ch,
    int s,
    unsigned int in_row_stride
) {
    return s >= 0 ? (float)src[(size_t)s * in_row_stride]
                  : conv_state[ch * KDA_FF_DCONV + (unsigned int)((int)KDA_FF_DCONV + s)];
}

// 2026-10-07: kda_tc_conv_rows' conv + SiLU for one channel at one row, window win[0..3]
// (oldest first).
__device__ __forceinline__ float kda_ff_conv_silu(const float* win, const float* wcoef,
                                                  float bias0) {
    float acc = bias0;
    #pragma unroll
    for (unsigned int k = 0; k < KDA_FF_DCONV; k++)
        acc += win[k] * wcoef[k];
    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    return acc * sigmoid_acc;
}

// 2026-10-07: kda_tc_conv_rows' per-warp sum of squares: lane 0 ends with the tree sum.
__device__ __forceinline__ float kda_ff_warp_sq(float silu) {
    float sq = silu * silu;
    for (int offset = 16; offset >= 1; offset >>= 1)
        sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
    return sq;
}

// 2026-10-07: kda_ff_prepare: kda_tc_prepare with its inputs computed from the projections,
// g_raw and beta_raw in place of conv_out, gate and beta (see the file comment). Same grid,
// block, record layout and outputs.
extern "C" __global__ void __launch_bounds__(128) kda_ff_prepare(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const __nv_bfloat16* __restrict__ g_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ beta_raw,
    __nv_bfloat16* __restrict__ qw_out,
    __nv_bfloat16* __restrict__ kpt_out,
    float* __restrict__ u_out,
    float* __restrict__ md_out,
    unsigned int H,
    unsigned int T,
    unsigned int in_row_stride,
    float l2_eps,
    float lower_bound,
    float scale
) {
    __shared__ float kp[KDA_FF_C][KDA_FF_D + 1];
    __shared__ float kn[KDA_FF_C][KDA_FF_D + 1];
    __shared__ float qp[KDA_FF_C][KDA_FF_D + 1];
    __shared__ float sh_l[KDA_FF_C][KDA_FF_C + 1];
    __shared__ float sh_m[KDA_FF_C][KDA_FF_C + 1];
    __shared__ float sh_t[KDA_FF_C][KDA_FF_C + 1];
    __shared__ float sh_beta[KDA_FF_C];
    // 2026-10-07: Per row, the four warp partials of the q and of the k sum of squares
    // (kda_tc_conv_rows' warp_sums for the head).
    __shared__ float l2p[KDA_FF_C][2][4];

    const unsigned int c = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const unsigned int warp = d >> 5;
    const unsigned int lane = d & 31u;
    const unsigned int qkv = H * KDA_FF_D;
    const size_t hd = (size_t)h * KDA_FF_D;
    const unsigned int rec = c * H + h;
    const unsigned int r0 = c * KDA_FF_C;

    if (d < KDA_FF_C) {
        const unsigned int t = r0 + d;
        float b = 0.0f;
        if (t < T) {
            float x = __bfloat162float(beta_raw[(size_t)t * H + h]);
            b = 1.0f / (1.0f + __expf(-x));
        }
        sh_beta[d] = b;
    }

    // 2026-10-07: This thread's three conv channels (q, k, v of head channel d).
    const unsigned int chq = (unsigned int)hd + d;
    const unsigned int chk = qkv + chq;
    const unsigned int chv = 2u * qkv + chq;
    float wq[KDA_FF_DCONV], wk[KDA_FF_DCONV], wv[KDA_FF_DCONV];
    #pragma unroll
    for (unsigned int i = 0; i < KDA_FF_DCONV; i++) {
        wq[i] = (float)conv_weight[chq * KDA_FF_DCONV + i];
        wk[i] = (float)conv_weight[chk * KDA_FF_DCONV + i];
        wv[i] = (float)conv_weight[chv * KDA_FF_DCONV + i];
    }
    const float bq = (conv_bias != nullptr) ? conv_bias[chq] : 0.0f;
    const float bk = (conv_bias != nullptr) ? conv_bias[chk] : 0.0f;
    const float bv = (conv_bias != nullptr) ? conv_bias[chv] : 0.0f;

    const __nv_bfloat16* sq_src = q + hd + d;
    const __nv_bfloat16* sk_src = k + hd + d;
    const __nv_bfloat16* sv_src = v + hd + d;

    // 2026-10-07: Window slot j holds position r0 - (d_conv - 1) + j; the first d_conv - 1 come
    // before the chunk (from the previous chunk's rows or the conv state). Chunk c has a live
    // row r0 < T, so every position read here is below T.
    float xq[KDA_FF_WIN], xk[KDA_FF_WIN], xv[KDA_FF_WIN];
    #pragma unroll
    for (unsigned int j = 0; j < KDA_FF_DCONV - 1u; j++) {
        const int s = (int)r0 - (int)(KDA_FF_DCONV - 1u) + (int)j;
        xq[j] = kda_ff_pos(sq_src, conv_state, chq, s, in_row_stride);
        xk[j] = kda_ff_pos(sk_src, conv_state, chk, s, in_row_stride);
        xv[j] = kda_ff_pos(sv_src, conv_state, chv, s, in_row_stride);
    }

    const float decay = expf(A_log[h]);
    const float dtb = dt_bias[hd + d];
    const __nv_bfloat16* g_src = g_raw + hd + d;

    float gc[KDA_FF_C], kf[KDA_FF_C], vf[KDA_FF_C], qf[KDA_FF_C];
    float acc = 0.0f;
    #pragma unroll
    for (unsigned int i = 0; i < KDA_FF_C; ++i) {
        const unsigned int t = r0 + i;
        const bool live = t < T;
        float g = 0.0f;
        if (live) {
            g = kda_ff_gate_scalar(__bfloat162float(g_src[(size_t)t * qkv]), dtb, decay,
                                   lower_bound);
        }
        acc += live ? g : 0.0f;
        gc[i] = acc;
        float sq = 0.0f, sk = 0.0f, sv = 0.0f;
        // 2026-10-07: `live` is the same for the whole block, so the shuffles below run with
        // every lane of every warp.
        if (live) {
            const unsigned int j = i + KDA_FF_DCONV - 1u;
            xq[j] = (float)sq_src[(size_t)t * in_row_stride];
            xk[j] = (float)sk_src[(size_t)t * in_row_stride];
            xv[j] = (float)sv_src[(size_t)t * in_row_stride];
            sq = kda_ff_conv_silu(&xq[i], wq, bq);
            sk = kda_ff_conv_silu(&xk[i], wk, bk);
            sv = kda_ff_conv_silu(&xv[i], wv, bv);
            const float pq = kda_ff_warp_sq(sq);
            const float pk = kda_ff_warp_sq(sk);
            if (lane == 0) {
                l2p[i][0][warp] = pq;
                l2p[i][1][warp] = pk;
            }
        }
        qf[i] = sq;
        kf[i] = sk;
        vf[i] = live ? __bfloat162float(__float2bfloat16(sv)) : 0.0f;
    }
    __syncthreads();

    // 2026-10-07: The L2 scale, then the BF16 round trip through conv_out.
    #pragma unroll
    for (unsigned int i = 0; i < KDA_FF_C; ++i) {
        if (r0 + i < T) {
            const float tq = l2p[i][0][0] + l2p[i][0][1] + l2p[i][0][2] + l2p[i][0][3];
            const float tk = l2p[i][1][0] + l2p[i][1][1] + l2p[i][1][2] + l2p[i][1][3];
            float nq = qf[i];
            float nk = kf[i];
            nq *= rsqrtf(tq + l2_eps);
            nk *= rsqrtf(tk + l2_eps);
            qf[i] = __bfloat162float(__float2bfloat16(nq));
            kf[i] = __bfloat162float(__float2bfloat16(nk));
        }
    }

    // 2026-10-07: From here on, kda_tc_prepare unchanged.
    const float gr = gc[KDA_FF_REF];
    #pragma unroll
    for (unsigned int i = 0; i < KDA_FF_C; ++i) {
        const float ep = expf(gc[i] - gr);
        const float en = expf(gr - gc[i]);
        kp[i][d] = kf[i] * ep;
        kn[i][d] = kf[i] * en;
        qp[i][d] = qf[i] * ep;
    }
    __syncthreads();

    for (unsigned int p = d; p < KDA_FF_C * KDA_FF_C; p += blockDim.x) {
        const unsigned int i = p / KDA_FF_C, j = p % KDA_FF_C;
        float dk = 0.0f, dq = 0.0f;
        if (j <= i) {
            for (unsigned int e = 0; e < KDA_FF_D; ++e) {
                const float b = kn[j][e];
                dk += kp[i][e] * b;
                dq += qp[i][e] * b;
            }
        }
        sh_l[i][j] = j < i ? sh_beta[i] * dk : 0.0f;
        sh_m[i][j] = j <= i ? scale * dq : 0.0f;
    }
    __syncthreads();

    if (d < KDA_FF_C) {
        const unsigned int j = d;
        float col[KDA_FF_C];
        #pragma unroll
        for (unsigned int m = 0; m < KDA_FF_C; ++m) col[m] = (m == j) ? 1.0f : 0.0f;
        #pragma unroll
        for (unsigned int i = 1; i < KDA_FF_C; ++i) {
            float s = 0.0f;
            #pragma unroll
            for (unsigned int m = 0; m < i; ++m) s += sh_l[i][m] * col[m];
            if (i > j) col[i] = -s;
        }
        const float bj = sh_beta[j];
        #pragma unroll
        for (unsigned int i = 0; i < KDA_FF_C; ++i) sh_t[i][j] = col[i] * bj;
    }
    __syncthreads();

    float kg[KDA_FF_C];
    #pragma unroll
    for (unsigned int j = 0; j < KDA_FF_C; ++j) kg[j] = kf[j] * expf(gc[j]);
    const float glast = gc[KDA_FF_C - 1];
    __nv_bfloat16* q_rec = qw_out + (size_t)rec * (2u * KDA_FF_C * KDA_FF_D);
    __nv_bfloat16* w_rec = q_rec + KDA_FF_C * KDA_FF_D;
    float* u_rec = u_out + (size_t)rec * (KDA_FF_C * KDA_FF_D);
    __nv_bfloat16 kcol[KDA_FF_C];
    #pragma unroll
    for (unsigned int i = 0; i < KDA_FF_C; ++i) {
        float au = 0.0f, aw = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j <= i; ++j) {
            const float tb = sh_t[i][j];
            au += tb * vf[j];
            aw += tb * kg[j];
        }
        u_rec[i * KDA_FF_D + d] = au;
        w_rec[i * KDA_FF_D + d] = __float2bfloat16_rn(aw);
        q_rec[i * KDA_FF_D + d] = __float2bfloat16_rn(scale * qf[i] * expf(gc[i]));
        kcol[i] = __float2bfloat16_rn(kf[i] * expf(glast - gc[i]));
    }
    __nv_bfloat16* k_rec = kpt_out + (size_t)rec * (KDA_FF_D * KDA_FF_C) + (size_t)d * KDA_FF_C;
    #pragma unroll
    for (unsigned int i = 0; i < KDA_FF_C; ++i) k_rec[i] = kcol[i];

    float* md = md_out + (size_t)rec * 256u;
    md[128u + d] = expf(glast);
    __nv_bfloat16* m_rec = reinterpret_cast<__nv_bfloat16*>(md);
    for (unsigned int p = d; p < KDA_FF_C * KDA_FF_C; p += blockDim.x) {
        m_rec[p] = __float2bfloat16_rn(sh_m[p / KDA_FF_C][p % KDA_FF_C]);
    }
}

// 2026-10-07: kda_ff_conv_state_tail: kda_tc_conv_state_tail reading channel ch from its
// projection (q, k or v by ch / qkv, column ch % qkv, row stride in_row_stride) instead of the
// packed qkv_proj; the same values, the same slot order. Grid (ceil(3 qkv / 256)), block 256.
// Launch after kda_ff_prepare, which reads the incoming state.
extern "C" __global__ void kda_ff_conv_state_tail(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    unsigned int rows,
    unsigned int qkv,
    unsigned int d_conv,
    unsigned int in_row_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= 3u * qkv) return;
    const unsigned int part = ch / qkv;
    const unsigned int col = ch - part * qkv;
    const __nv_bfloat16* src = part == 0u ? q : (part == 1u ? k : v);
    float* state = conv_state + ch * d_conv;
    for (unsigned int i = 0; i < d_conv; i++) {
        const int s = (int)rows - (int)d_conv + (int)i;
        state[i] = s >= 0 ? (float)src[(size_t)s * in_row_stride + col] : state[rows + i];
    }
}
