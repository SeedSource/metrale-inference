// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: KDA (Kimi Delta Attention) recurrent decode: one token, per head h:
//   S[k][v] <- S[k][v] * exp(gate[k])
//   delta[v] = (v[v] - sum_k S[k][v] k[k]) * beta
//   S[k][v] <- S[k][v] + k[k] delta[v]
//   out[v]   = sum_k S[k][v] q[k] * scale
//
// Owner: gb10 kernels.
// Invariants:
// - q, k, v and gate are [H, D] and beta is [H]; the state is FP32 [H, D, D], indexed
//   [k][v] and updated in place; out is FP32 [H, D]. Gate, beta, state and out are FP32 in
//   every entry point.
// - blockIdx.x is the head. The two-pass kernels stride threads over v; the _smem kernel
//   owns VPB columns per block, starting at blockIdx.y * VPB.
// - The caller supplies q and k already L2-normalised (the decode conv,
//   causal_conv1d_update_l2norm, applies it), beta already sigmoided, and
//   scale = 1/sqrt(D).
//
// gate is kda_gate's output, a log-decay, and this kernel applies exp() to it.
// compute_gdn_gates (ssm_preprocess.cu) stores its gate already exponentiated and one per
// head, so a GDN gate passed here would be exponentiated twice, with no error raised.
















































#include <cuda_bf16.h>
#include <math.h>

// 2026-09-25: Dynamic shared memory: 3 * D floats, holding exp(gate), k and q * scale.
#define KDA_REC_BODY(LOAD_QKV)                                                        \
    extern __shared__ float sh[];                                                     \
    const unsigned int h = blockIdx.x;                                                \
    if (h >= H) return;                                                               \
    float* sh_decay = sh;                                                             \
    float* sh_k = sh + D;                                                             \
    float* sh_q = sh + 2u * D;                                                        \
    const size_t hd = (size_t)h * D;                                                  \
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {                       \
        sh_decay[i] = expf(gate[hd + i]);                                             \
        sh_k[i] = LOAD_QKV(k[hd + i]);                                                \
        sh_q[i] = LOAD_QKV(q[hd + i]) * scale;                                        \
    }                                                                                 \
    __syncthreads();                                                                  \
    const float b = beta[h];                                                          \
    float* S = state + hd * D;                                                        \
    for (unsigned int vi = threadIdx.x; vi < D; vi += blockDim.x) {                    \
        float kv = 0.0f;                                                              \
        /* 2026-09-25: Pass 1 decays column vi of S and accumulates kv = sum_k S[k][vi] k[k];\
           pass 2 adds k[k] * delta and accumulates o = sum_k S[k][vi] q[k] * scale.  \
           One thread per column: a warp reads consecutive vi for a fixed kk, so the  \
           state accesses are coalesced. unroll 8 keeps eight independent state loads \
           in flight per thread; it does not reorder the kv and o sums, which run     \
           kk = 0..D-1 in both passes.                                                \
                                                                                      \
                                                                                      \
           */                                                                         \
        _Pragma("unroll 8")                                                           \
        for (unsigned int kk = 0; kk < D; ++kk) {                                      \
            const size_t idx = (size_t)kk * D + vi;                                   \
            const float s = S[idx] * sh_decay[kk];                                    \
            S[idx] = s;                                                               \
            kv += s * sh_k[kk];                                                       \
        }                                                                             \
        const float delta = (LOAD_QKV(v[hd + vi]) - kv) * b;                          \
        float o = 0.0f;                                                               \
        _Pragma("unroll 8")                                                           \
        for (unsigned int kk = 0; kk < D; ++kk) {                                      \
            const size_t idx = (size_t)kk * D + vi;                                   \
            const float s = S[idx] + sh_k[kk] * delta;                                \
            S[idx] = s;                                                               \
            o += s * sh_q[kk];                                                        \
        }                                                                             \
        out[hd + vi] = o;                                                             \
    }

#define KDA_REC_IDENT(x) (x)
#define KDA_REC_BF16(x) __bfloat162float(x)

// 2026-09-25: FP32-input twin, used by the kda_recurrent and kda_chunk microtest examples.
extern "C" __global__ void kda_recurrent_decode_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale
) {
    KDA_REC_BODY(KDA_REC_IDENT)
}

// 2026-09-25: BF16 q, k and v. glm5next_kda launches this when it does not launch the
// _smem kernel below, with block = min(128, D) and 3 * D floats of shared memory.


extern "C" __global__ void kda_recurrent_decode_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale
) {
    KDA_REC_BODY(KDA_REC_BF16)
}

// 2026-09-25: Single-pass-over-global variant: the same expressions in the same order as
// kda_recurrent_decode_bf16 (decay, kv over kk = 0..D-1, delta, update, o over kk = 0..D-1),
// but each thread keeps its decayed column in shared memory between the two passes, so the
// state is read once and written once from global memory.
//
// The v axis has no cross-thread dependency (kv, delta and o are per (h, vi)), so a block
// owns VPB columns and grid.y covers D / VPB. Shared memory: 3 * D + VPB * (D + 1) floats.
// The launcher must make VPB divide D and launch blockDim.x == VPB; otherwise columns are
// dropped with no error. glm5next_kda checks both, uses this kernel only when the target
// has it and the request fits KDA_SMEM_BUDGET, and skips it when
// METRALE_GLM_KDA_NO_SMEM=1.















extern "C" __global__ void kda_recurrent_decode_bf16_smem(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale,
    unsigned int VPB
) {
    extern __shared__ float sh[];
    const unsigned int h = blockIdx.x;
    if (h >= H) return;
    const unsigned int v0 = blockIdx.y * VPB;
    if (v0 >= D) return;

    float* sh_decay = sh;
    float* sh_k = sh + D;
    float* sh_q = sh + 2u * D;
// 2026-09-25: [VPB, D + 1]: column threadIdx.x of this block's slice, k-major. The +1 pad
// avoids bank conflicts: at a stride of D = 128 floats every thread of a warp would hit
// the same bank (128 % 32 == 0); a stride of D + 1 moves each thread to the next bank.



    float* sh_s = sh + 3u * D;
    const unsigned int col_stride = D + 1u;

    const size_t hd = (size_t)h * D;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        sh_decay[i] = expf(gate[hd + i]);
        sh_k[i] = __bfloat162float(k[hd + i]);
        sh_q[i] = __bfloat162float(q[hd + i]) * scale;
    }
    __syncthreads();

    const float b = beta[h];
    float* S = state + hd * D;
    const unsigned int vi = v0 + threadIdx.x;
    if (threadIdx.x >= VPB || vi >= D) return;
    float* col = sh_s + (size_t)threadIdx.x * col_stride;

    float kv = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = S[(size_t)kk * D + vi] * sh_decay[kk];
        col[kk] = s;
        kv += s * sh_k[kk];
    }
    const float delta = (__bfloat162float(v[hd + vi]) - kv) * b;
    float o = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = col[kk] + sh_k[kk] * delta;
        S[(size_t)kk * D + vi] = s;
        o += s * sh_q[kk];
    }
    out[hd + vi] = o;
}

// 2026-10-01: Token-loop form of kda_recurrent_decode_bf16_smem: T tokens of one sequence in
// one launch, instead of T launches. Same grid (H, D / VPB), same block (VPB), same shared
// memory request (3 * D + VPB * (D + 1) floats), same launcher contract (VPB divides D and
// equals blockDim.x).
//
// Each thread loads its state column from global once, keeps it in its shared-memory column
// across all T tokens, and writes it back once at the end. The state is FP32 in global and
// FP32 in shared memory, so holding it on chip changes no value. Per token the arithmetic is
// kda_recurrent_decode_bf16_smem's, expression for expression and in the same order (decay,
// kv over kk = 0..D-1, delta, update, o over kk = 0..D-1); the only operand that differs is
// where the pre-decay state value is read from (col[kk] here, S[kk * D + vi] there), which
// holds the same float. The gb10 tree builds with --fmad=false (common/KERNEL.toml), so no
// multiply-add is contracted in either kernel.
//
// Token t reads q/k/v at t * qkv_row_stride (BF16 elements; k and v are passed as their own
// row-0 pointers, as stateful_row passes them), gate at t * gate_row_stride, beta at
// t * beta_row_stride and writes out at t * out_row_stride (FP32 elements): exactly the
// per-row pointers glm5next_kda's stateful_row hands the decode kernel.
//
// Synchronisation: a __syncthreads before each token's staging (t > 0) keeps every thread's
// reads of sh_decay / sh_k / sh_q for token t - 1 ahead of the overwrite; one after the
// staging publishes it. Threads past VPB (none under the launcher contract) take no column
// but still reach every barrier.
extern "C" __global__ void kda_recurrent_prefill_bf16_smem(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale,
    unsigned int VPB,
    unsigned int T,
    unsigned int qkv_row_stride,
    unsigned int gate_row_stride,
    unsigned int beta_row_stride,
    unsigned int out_row_stride
) {
    extern __shared__ float sh[];
    const unsigned int h = blockIdx.x;
    if (h >= H) return;
    const unsigned int v0 = blockIdx.y * VPB;
    if (v0 >= D) return;

    float* sh_decay = sh;
    float* sh_k = sh + D;
    float* sh_q = sh + 2u * D;
    float* sh_s = sh + 3u * D;
    const unsigned int col_stride = D + 1u;

    const size_t hd = (size_t)h * D;
    float* S = state + hd * D;
    const unsigned int vi = v0 + threadIdx.x;
    const bool owns = (threadIdx.x < VPB && vi < D);
    float* col = sh_s + (size_t)threadIdx.x * col_stride;

    if (owns) {
        for (unsigned int kk = 0; kk < D; ++kk)
            col[kk] = S[(size_t)kk * D + vi];
    }

    for (unsigned int t = 0; t < T; ++t) {
        const __nv_bfloat16* qt = q + (size_t)t * qkv_row_stride;
        const __nv_bfloat16* kt = k + (size_t)t * qkv_row_stride;
        const __nv_bfloat16* vt = v + (size_t)t * qkv_row_stride;
        const float* gt = gate + (size_t)t * gate_row_stride;
        const float* bt = beta + (size_t)t * beta_row_stride;
        float* ot = out + (size_t)t * out_row_stride;

        if (t > 0) __syncthreads();
        for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
            sh_decay[i] = expf(gt[hd + i]);
            sh_k[i] = __bfloat162float(kt[hd + i]);
            sh_q[i] = __bfloat162float(qt[hd + i]) * scale;
        }
        __syncthreads();

        if (owns) {
            const float b = bt[h];
            float kv = 0.0f;
            #pragma unroll 8
            for (unsigned int kk = 0; kk < D; ++kk) {
                const float s = col[kk] * sh_decay[kk];
                col[kk] = s;
                kv += s * sh_k[kk];
            }
            const float delta = (__bfloat162float(vt[hd + vi]) - kv) * b;
            float o = 0.0f;
            #pragma unroll 8
            for (unsigned int kk = 0; kk < D; ++kk) {
                const float s = col[kk] + sh_k[kk] * delta;
                col[kk] = s;
                o += s * sh_q[kk];
            }
            ot[hd + vi] = o;
        }
    }

    if (owns) {
        for (unsigned int kk = 0; kk < D; ++kk)
            S[(size_t)kk * D + vi] = col[kk];
    }
}

// 2026-10-01: Prefetching twin of kda_recurrent_prefill_bf16_smem: the same arguments, grid
// (H, D / VPB) and block (VPB), and the same per-element arithmetic in the same order, with the
// serial global-load latency taken off the token loop.
//
// What changes, per thread:
// - Prefetch. The raw inputs a thread stages for token t (its gate, k and q elements
//   i = tid, tid + VPB, ...) and the raw v and beta its column reads are loaded into registers
//   KDA_PF_DIST tokens ahead (KdaPfRaw, one set per in-flight token), so token t + 1's loads are
//   in flight while token t computes. Loading earlier changes no value: nothing writes q, k, v,
//   gate or beta during the launch.
// - Staging. The staged arrays (expf(gate), k, q * scale) are double-buffered, and token t + 1
//   is staged right after token t computes, from its prefetched raw values. The conversions are
//   kda_recurrent_prefill_bf16_smem's (expf, __bfloat162float, * scale) applied to the same
//   input values, so the staged floats are the same. v is widened at staging time instead of
//   after pass 1: the same op on the same bf16.
// - One __syncthreads per token instead of two. Token t reads buffer t & 1; staging t + 1
//   writes buffer (t + 1) & 1, whose last readers (token t - 1) finished before the barrier
//   that ended token t - 1; the barrier after staging publishes token t + 1.
// - At D == 128 and VPB == blockDim.x == 32 (the GLM-5.3 geometry), each thread holds its state
//   column in 128 registers instead of shared memory, with both kk loops fully unrolled and the
//   staged decay, k and q read as float4 broadcasts. Every other geometry keeps the
//   shared-memory column (kda_pf_run<0, KDA_PF_E_MAX>).
//
// What does not change, per (h, vi) and per token: pass 1 is s = col[kk] * decay[kk], store,
// kv += s * k[kk] for kk = 0..D-1 in order; delta = (v - kv) * b; pass 2 is
// s = col[kk] + k[kk] * delta, store, o += s * q[kk] for kk = 0..D-1 in order; out = o; the
// column is read from global once before token 0 and written back once after the last token.
// Full unrolling keeps each sum a dependent FADD chain in ascending kk (nvcc does not
// reassociate floating-point adds), and the gb10 tree builds with --fmad=false
// (common/KERNEL.toml), so no multiply-add is contracted.
//
// Launcher contract (glm5next_kda's stateful_rows checks it): VPB divides D, blockDim.x == VPB
// <= 32 (__launch_bounds__), D <= KDA_PF_E_MAX * VPB, and shared memory
// (6 * D + VPB * (D + 1)) floats. Columns or staged elements outside the contract are dropped
// with no error, as in the _smem kernels.
#define KDA_PF_DIST 2u
#define KDA_PF_E_MAX 8u

// 2026-10-01: One token's raw inputs for one thread: E staged elements of gate, k and q, and
// the v element and beta of the thread's column.
template <unsigned int E>
struct KdaPfRaw {
    float g[E];
    __nv_bfloat16 k[E];
    __nv_bfloat16 q[E];
    __nv_bfloat16 v;
    float b;
};

struct KdaPfArgs {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* gate;
    const float* beta;
    unsigned int D;
    unsigned int T;
    unsigned int qkv_row_stride;
    unsigned int gate_row_stride;
    unsigned int beta_row_stride;
    size_t hd;
    unsigned int h;
    unsigned int vi;
    bool owns;
};

// 2026-10-01: Token t's raw inputs, at the addresses kda_recurrent_prefill_bf16_smem reads
// (gt[hd + i], kt[hd + i], qt[hd + i], vt[hd + vi], bt[h]). DC != 0 fixes D = DC and 32
// threads per block.
template <unsigned int DC, unsigned int E>
__device__ __forceinline__ void kda_pf_load(KdaPfRaw<E>& r, const KdaPfArgs& a, unsigned int t) {
    const unsigned int Dn = DC ? DC : a.D;
    const unsigned int nt = DC ? 32u : blockDim.x;
    const size_t qo = (size_t)t * a.qkv_row_stride + a.hd;
    const size_t go = (size_t)t * a.gate_row_stride + a.hd;
    #pragma unroll
    for (unsigned int j = 0; j < E; ++j) {
        const unsigned int i = threadIdx.x + j * nt;
        if (i < Dn) {
            r.g[j] = a.gate[go + i];
            r.k[j] = a.k[qo + i];
            r.q[j] = a.q[qo + i];
        }
    }
    if (a.owns) {
        r.v = a.v[qo + a.vi];
        r.b = a.beta[(size_t)t * a.beta_row_stride + a.h];
    }
}

// 2026-10-01: Stage one token from its raw inputs into dst ([decay | k | q], D floats each)
// with kda_recurrent_prefill_bf16_smem's staging expressions, and widen the column's v.
template <unsigned int DC, unsigned int E>
__device__ __forceinline__ void kda_pf_stage(
    const KdaPfRaw<E>& r, float* dst, const KdaPfArgs& a, float scale, float& vf, float& b
) {
    const unsigned int Dn = DC ? DC : a.D;
    const unsigned int nt = DC ? 32u : blockDim.x;
    #pragma unroll
    for (unsigned int j = 0; j < E; ++j) {
        const unsigned int i = threadIdx.x + j * nt;
        if (i < Dn) {
            dst[i] = expf(r.g[j]);
            dst[Dn + i] = __bfloat162float(r.k[j]);
            dst[2u * Dn + i] = __bfloat162float(r.q[j]) * scale;
        }
    }
    if (a.owns) {
        vf = __bfloat162float(r.v);
        b = r.b;
    }
}

// 2026-10-01: The token loop. DC == 0: runtime D, column in shared memory (stride D + 1, as in
// kda_recurrent_prefill_bf16_smem). DC == 128: column in registers; requires blockDim.x == 32.
// sh: [2][3][D] staged floats, then (DC == 0 only) [VPB][D + 1] columns.
template <unsigned int DC, unsigned int E>
__device__ __forceinline__ void kda_pf_run(
    const KdaPfArgs& a, float* __restrict__ state, float* __restrict__ out,
    unsigned int out_row_stride, float scale, float* sh
) {
    const unsigned int Dn = DC ? DC : a.D;
    float* S = state + a.hd * Dn;
    float* col = sh + 6u * Dn + (size_t)threadIdx.x * (Dn + 1u);
    float colr[DC ? DC : 1];

    if (a.owns) {
        if constexpr (DC != 0) {
            #pragma unroll
            for (unsigned int kk = 0; kk < DC; ++kk)
                colr[kk] = S[(size_t)kk * DC + a.vi];
        } else {
            for (unsigned int kk = 0; kk < Dn; ++kk)
                col[kk] = S[(size_t)kk * Dn + a.vi];
        }
    }

    // 2026-10-01: Prologue: raw[p] <- token p; stage token 0 into buffer 0; raw[0] <- token
    // KDA_PF_DIST. From then on raw[j] holds the next token congruent to j mod KDA_PF_DIST.
    // Each refill is issued after a barrier, never just before one, so a barrier that waited on
    // outstanding loads would still find them covered by a token's compute.
    KdaPfRaw<E> raw[KDA_PF_DIST];
    #pragma unroll
    for (unsigned int p = 0; p < KDA_PF_DIST; ++p)
        if (p < a.T) kda_pf_load<DC, E>(raw[p], a, p);
    float vf = 0.0f, b = 0.0f;
    if (a.T > 0) kda_pf_stage<DC, E>(raw[0], sh, a, scale, vf, b);
    __syncthreads();
    if (KDA_PF_DIST < a.T) kda_pf_load<DC, E>(raw[0], a, KDA_PF_DIST);

    for (unsigned int t0 = 0; t0 < a.T; t0 += KDA_PF_DIST) {
        #pragma unroll
        for (unsigned int p = 0; p < KDA_PF_DIST; ++p) {
            const unsigned int t = t0 + p;
            // 2026-10-01: Block-uniform, so every thread reaches the same barriers.
            if (t >= a.T) break;
            const float* c_decay = sh + (size_t)(t & 1u) * 3u * Dn;
            const float* c_k = c_decay + Dn;
            const float* c_q = c_decay + 2u * Dn;

            if (a.owns) {
                float kv = 0.0f;
                float o = 0.0f;
                if constexpr (DC != 0) {
                    // 2026-10-01: The float4 reads deliver the same staged floats; each sum still
                    // takes its terms one at a time in ascending kk.
                    #pragma unroll
                    for (unsigned int kk = 0; kk < DC; kk += 4) {
                        const float4 dd = *reinterpret_cast<const float4*>(c_decay + kk);
                        const float4 k4 = *reinterpret_cast<const float4*>(c_k + kk);
                        float s;
                        s = colr[kk] * dd.x;     colr[kk] = s;     kv += s * k4.x;
                        s = colr[kk + 1] * dd.y; colr[kk + 1] = s; kv += s * k4.y;
                        s = colr[kk + 2] * dd.z; colr[kk + 2] = s; kv += s * k4.z;
                        s = colr[kk + 3] * dd.w; colr[kk + 3] = s; kv += s * k4.w;
                    }
                    const float delta = (vf - kv) * b;
                    #pragma unroll
                    for (unsigned int kk = 0; kk < DC; kk += 4) {
                        const float4 k4 = *reinterpret_cast<const float4*>(c_k + kk);
                        const float4 q4 = *reinterpret_cast<const float4*>(c_q + kk);
                        float s;
                        s = colr[kk] + k4.x * delta;     colr[kk] = s;     o += s * q4.x;
                        s = colr[kk + 1] + k4.y * delta; colr[kk + 1] = s; o += s * q4.y;
                        s = colr[kk + 2] + k4.z * delta; colr[kk + 2] = s; o += s * q4.z;
                        s = colr[kk + 3] + k4.w * delta; colr[kk + 3] = s; o += s * q4.w;
                    }
                } else {
                    #pragma unroll 8
                    for (unsigned int kk = 0; kk < Dn; ++kk) {
                        const float s = col[kk] * c_decay[kk];
                        col[kk] = s;
                        kv += s * c_k[kk];
                    }
                    const float delta = (vf - kv) * b;
                    #pragma unroll 8
                    for (unsigned int kk = 0; kk < Dn; ++kk) {
                        const float s = col[kk] + c_k[kk] * delta;
                        col[kk] = s;
                        o += s * c_q[kk];
                    }
                }
                out[(size_t)t * out_row_stride + a.hd + a.vi] = o;
            }

            // 2026-10-01: Stage token t + 1 into the other buffer; after the barrier, refill its
            // raw slot with token t + 1 + KDA_PF_DIST.
            const unsigned int nx = (p + 1u) % KDA_PF_DIST;
            if (t + 1u < a.T)
                kda_pf_stage<DC, E>(
                    raw[nx], sh + (size_t)((t + 1u) & 1u) * 3u * Dn, a, scale, vf, b
                );
            __syncthreads();
            if (t + 1u + KDA_PF_DIST < a.T)
                kda_pf_load<DC, E>(raw[nx], a, t + 1u + KDA_PF_DIST);
        }
    }

    if (a.owns) {
        if constexpr (DC != 0) {
            #pragma unroll
            for (unsigned int kk = 0; kk < DC; ++kk)
                S[(size_t)kk * DC + a.vi] = colr[kk];
        } else {
            for (unsigned int kk = 0; kk < Dn; ++kk)
                S[(size_t)kk * Dn + a.vi] = col[kk];
        }
    }
}

#define KDA_PF_PARAMS                                                                         \
    const __nv_bfloat16* __restrict__ q,                                                      \
    const __nv_bfloat16* __restrict__ k,                                                      \
    const __nv_bfloat16* __restrict__ v,                                                      \
    const float* __restrict__ gate,                                                           \
    const float* __restrict__ beta,                                                           \
    float* __restrict__ state,                                                                \
    float* __restrict__ out,                                                                  \
    unsigned int H,                                                                           \
    unsigned int D,                                                                           \
    float scale,                                                                              \
    unsigned int VPB,                                                                         \
    unsigned int T,                                                                           \
    unsigned int qkv_row_stride,                                                              \
    unsigned int gate_row_stride,                                                             \
    unsigned int beta_row_stride,                                                             \
    unsigned int out_row_stride

// 2026-10-01: Block prologue shared by the two entry points: the _smem kernels' early returns
// (whole blocks only) and the per-thread KdaPfArgs.
#define KDA_PF_PROLOGUE                                                                       \
    extern __shared__ __align__(16) float sh_pf[];                                            \
    const unsigned int h = blockIdx.x;                                                        \
    if (h >= H) return;                                                                       \
    const unsigned int v0 = blockIdx.y * VPB;                                                 \
    if (v0 >= D) return;                                                                      \
    KdaPfArgs a;                                                                              \
    a.q = q;                                                                                  \
    a.k = k;                                                                                  \
    a.v = v;                                                                                  \
    a.gate = gate;                                                                            \
    a.beta = beta;                                                                            \
    a.D = D;                                                                                  \
    a.T = T;                                                                                  \
    a.qkv_row_stride = qkv_row_stride;                                                        \
    a.gate_row_stride = gate_row_stride;                                                      \
    a.beta_row_stride = beta_row_stride;                                                      \
    a.hd = (size_t)h * D;                                                                     \
    a.h = h;                                                                                  \
    a.vi = v0 + threadIdx.x;                                                                  \
    a.owns = (threadIdx.x < VPB && a.vi < D);

// 2026-10-01: The opt-in prefill token loop under METRALE_GLM_KDA_PREFETCH=1 (glm5next_kda's
// stateful_rows). Picks the register-column body at D == 128 with VPB == blockDim.x == 32 and
// the shared-memory-column body otherwise; the test is block-uniform.
extern "C" __global__ void __launch_bounds__(32) kda_recurrent_prefill_bf16_pf(KDA_PF_PARAMS) {
    KDA_PF_PROLOGUE
    if (D == 128u && VPB == 32u && blockDim.x == 32u)
        kda_pf_run<128u, 4u>(a, state, out, out_row_stride, scale, sh_pf);
    else
        kda_pf_run<0u, KDA_PF_E_MAX>(a, state, out, out_row_stride, scale, sh_pf);
}

// 2026-10-01: The shared-memory-column body at every geometry, so kda_tokenloop_microtest can
// check it at D == 128 too. Not launched by the model.
extern "C" __global__ void __launch_bounds__(32) kda_recurrent_prefill_bf16_pf_smem(
    KDA_PF_PARAMS
) {
    KDA_PF_PROLOGUE
    kda_pf_run<0u, KDA_PF_E_MAX>(a, state, out, out_row_stride, scale, sh_pf);
}
