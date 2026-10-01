// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GLM-5.3 mHC (manifold-constrained hyper-connection) kernels: expand, pre (fused,
// and split into mix + finish), post, and the head mean.
//
// Owner: gb10 kernels.
// Invariants:
// - glm5next_hc_pre, and glm5next_hc_mix (or _bf16) followed by glm5next_hc_finish, end the
//   Sinkhorn on its last column normalisation, which divides by (sum + hc_eps). They do not
//   add the exact column projection hyper_connection::hc_pre ends with
//   (kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu).
// - The split pair reproduces glm5next_hc_pre byte for byte, and glm5next_hc_post reproduces
//   glm5next_hc_post_ref; glm5next_hc_split_gate and glm5next_hc_post_gate (model-arch
//   examples) check both.
// - glm5next_hc_post_ref and glm5next_hc_expand have the arithmetic of hyper_connection's
//   hc_post and hc_expand.
// - Every kernel assumes blockDim == GLM_HC_BLOCK (256) and hc_mult <= GLM_HC_MAX_MULT (4);
//   the shared and local arrays are sized for that.
// - 2026-10-01: glm5next_hc_mix_bf16_tokmajor (one block per token) writes the same mix bytes
//   as glm5next_hc_mix_bf16 (argument above the kernel; checked on a GPU by model-arch
//   examples/mhc_tokmajor_bitparity_microtest.rs).
//
// All of them resolve from the module glm5next_mhc (GLM5NEXT_MHC_MODULE in model-arch
// glm5next_mhc.rs), so a target without the DeepSeek-V4 model directory has every GLM mHC
// kernel: a target compiles common/ plus its own model directory (crates/kernels/
// build_stage.rs). The host launches mix, finish, post, head and expand through glm_hc_pre,
// glm_hc_post, hc_head_mean and glm_hc_expand. glm5next_hc_pre and glm5next_hc_post_ref
// are the oracles of the gates above; mhc_microtest launches glm5next_hc_pre through
// ops::hc_pre, whose argument list it shares.










#include <cuda_bf16.h>

#define GLM_HC_BLOCK 256
#define GLM_HC_MAX_MULT 4
#define GLM_HC_MAX_MIX 24

// 2026-09-25: In-place tree sum of red[0..GLM_HC_BLOCK); every thread of the block calls it.
__device__ __forceinline__ float glm_hc_block_reduce(float* red, unsigned int tid) {
    for (unsigned int s = GLM_HC_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    return red[0];
}


// 2026-09-25: glm5next_hc_pre: streams [T, hc, H] -> y_out [T, H] (BF16), post_out [T, hc],
// comb_out [T, hc, hc]. Grid (T, 1, 1), one block per token.
//   flat  = streams[t] / sqrt(mean(streams[t]^2) + norm_eps)     (over all hc*H values)
//   mixes = hc_fn @ flat                                         -> [pre | post | comb]
//   pre   = sigmoid(pre * scale0 + base) + hc_eps
//   post  = 2 * sigmoid(post * scale1 + base)
//   comb  = softmax(comb * scale2 + base, over j) + hc_eps, then a column normalisation,
//           then (sinkhorn_iters - 1) x (row, column) normalisations, each / (sum + hc_eps)
//   y     = sum_i pre[i] * streams[t, i]
// hc_fn is [mix_hc, hc*H] FP32 with mix_hc = (2 + hc) * hc; hc_scale is [3], hc_base
// [mix_hc].
extern "C" __global__ void glm5next_hc_pre(
    const float* __restrict__ streams,
    const float* __restrict__ hc_fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float norm_eps,
    const float hc_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int hc_dim = hc * H;
    const unsigned int mix_hc = (2 + hc) * hc;

    const float* x = streams + (size_t)t * hc_dim;

    __shared__ float red[GLM_HC_BLOCK];
    __shared__ float s_rsqrt;
    __shared__ float s_mix[GLM_HC_MAX_MIX];
    __shared__ float s_pre[GLM_HC_MAX_MULT];


    float ss = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
        float v = (float)x[k];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    float ssum = glm_hc_block_reduce(red, tid);
    if (tid == 0) s_rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    __syncthreads();
    const float rsqrt = s_rsqrt;

    // 2026-09-25: mixes[m] = rsqrt * sum_k fn[m, k] * x[k]: the norm is a per-token scalar,
    // so it is applied once after the dot.
    for (unsigned int m = 0; m < mix_hc; ++m) {
        const float* fn_row = hc_fn + (size_t)m * hc_dim;
        float acc = 0.f;
        for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
            acc += fn_row[k] * (float)x[k];
        }
        red[tid] = acc;
        __syncthreads();
        float r = glm_hc_block_reduce(red, tid);
        if (tid == 0) s_mix[m] = r * rsqrt;
        __syncthreads();
    }


    if (tid == 0) {
        float comb[GLM_HC_MAX_MULT * GLM_HC_MAX_MULT];
        for (unsigned int i = 0; i < hc; ++i) {
            float pr = s_mix[i] * hc_scale[0] + hc_base[i];
            s_pre[i] = 1.f / (1.f + expf(-pr)) + hc_eps;
            float po = s_mix[hc + i] * hc_scale[1] + hc_base[hc + i];
            post_out[(size_t)t * hc + i] = 2.f * (1.f / (1.f + expf(-po)));
        }
        for (unsigned int i = 0; i < hc; ++i)
            for (unsigned int j = 0; j < hc; ++j)
                comb[i * hc + j] =
                    s_mix[2 * hc + i * hc + j] * hc_scale[2] + hc_base[2 * hc + i * hc + j];

        for (unsigned int i = 0; i < hc; ++i) {
            float mx = -1e30f;
            for (unsigned int j = 0; j < hc; ++j) mx = fmaxf(mx, comb[i * hc + j]);
            float sum = 0.f;
            for (unsigned int j = 0; j < hc; ++j) {
                float e = expf(comb[i * hc + j] - mx);
                comb[i * hc + j] = e;
                sum += e;
            }
            for (unsigned int j = 0; j < hc; ++j) comb[i * hc + j] = comb[i * hc + j] / sum + hc_eps;
        }

        for (unsigned int j = 0; j < hc; ++j) {
            float c = hc_eps;
            for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
            for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
        }

        for (unsigned int it = 0; it + 1 < sinkhorn_iters; ++it) {
            for (unsigned int i = 0; i < hc; ++i) {
                float r = hc_eps;
                for (unsigned int j = 0; j < hc; ++j) r += comb[i * hc + j];
                for (unsigned int j = 0; j < hc; ++j) comb[i * hc + j] /= r;
            }
            for (unsigned int j = 0; j < hc; ++j) {
                float c = hc_eps;
                for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
                for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
            }
        }



        // 2026-09-25: No exact column projection after the loop; see the file header.
        for (unsigned int i = 0; i < hc; ++i)
            for (unsigned int j = 0; j < hc; ++j)
                comb_out[(size_t)t * hc * hc + i * hc + j] = comb[i * hc + j];
    }
    __syncthreads();


    for (unsigned int d = tid; d < H; d += GLM_HC_BLOCK) {
        float acc = 0.f;
        for (unsigned int i = 0; i < hc; ++i) acc += s_pre[i] * (float)x[i * H + d];
        y_out[(size_t)t * H + d] = __float2bfloat16(acc);
    }
}


















// 2026-09-25: glm5next_hc_mix + glm5next_hc_finish: glm5next_hc_pre split in two.
// glm5next_hc_mix runs one block per (token, mixing row) instead of one block per token
// walking all mix_hc rows. Each block recomputes the RMS over the token's hc*H stream
// values in the fused kernel's order and reduces its one row the way the fused kernel
// does, so every mix equals the fused kernel's. Recomputing the RMS per block keeps the
// blocks free of any cross-block dependency.

// 2026-09-25: streams [T, hc, H] -> mix_out [T, mix_hc]; hc_fn is [mix_hc, hc*H] FP32.
extern "C" __global__ void glm5next_hc_mix(
    const float* __restrict__ streams,
    const float* __restrict__ hc_fn,
    float* __restrict__ mix_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const float norm_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc_dim = hc_mult * hidden_size;
    const unsigned int mix_hc = (2 + hc_mult) * hc_mult;

    const float* x = streams + (size_t)t * hc_dim;
    __shared__ float red[GLM_HC_BLOCK];


    float ss = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
        float v = (float)x[k];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    const float ssum = glm_hc_block_reduce(red, tid);
    const float rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    // 2026-09-25: Every thread has read red[0], and `red` is reused below.
    __syncthreads();


    const float* fn_row = hc_fn + (size_t)m * hc_dim;
    float acc = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
        acc += fn_row[k] * (float)x[k];
    }
    red[tid] = acc;
    __syncthreads();
    const float r = glm_hc_block_reduce(red, tid);
    if (tid == 0) mix_out[(size_t)t * mix_hc + m] = r * rsqrt;
}




































// 2026-09-25: glm5next_hc_mix_bf16: glm5next_hc_mix reading hc_fn as BF16 [mix_hc, hc*H].
// Widening BF16 to FP32 is exact, so each product and the accumulation order are
// glm5next_hc_mix's on the widened weights. glm_hc_pre (model-arch glm5next_mhc.rs) picks it
// when the site's hc_fn is BF16 (Glm5NextMhcSiteWeights::hc_fn_bf16) and the handle is
// linked. Grid (T, mix_hc, 1).
extern "C" __global__ void glm5next_hc_mix_bf16(
    const float* __restrict__ streams,
    const __nv_bfloat16* __restrict__ hc_fn,
    float* __restrict__ mix_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const float norm_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc_dim = hc_mult * hidden_size;
    const unsigned int mix_hc = (2 + hc_mult) * hc_mult;

    const float* x = streams + (size_t)t * hc_dim;
    __shared__ float red[GLM_HC_BLOCK];


    float ss = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
        float v = (float)x[k];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    const float ssum = glm_hc_block_reduce(red, tid);
    const float rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    // 2026-09-25: Every thread has read red[0], and `red` is reused below.
    __syncthreads();


    const __nv_bfloat16* fn_row = hc_fn + (size_t)m * hc_dim;
    float acc = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
        acc += __bfloat162float(fn_row[k]) * (float)x[k];
    }
    red[tid] = acc;
    __syncthreads();
    const float r = glm_hc_block_reduce(red, tid);
    if (tid == 0) mix_out[(size_t)t * mix_hc + m] = r * rsqrt;
}


// 2026-10-01: glm5next_hc_mix_bf16_tokmajor: glm5next_hc_mix_bf16 with one block per token
// instead of one per (token, mixing row), so each token's hc*H highway row is read from memory
// once instead of 2 * mix_hc times (the RMS loop and the dot loop of each of the mix_hc
// blocks). Same arguments and the same mix_out [T, mix_hc]; grid (T, 1, 1). glm_hc_pre_sliced
// (model-arch glm5next_mhc.rs) launches it under METRALE_GLM_MHC_TOKMAJOR=1. Every output bit
// equals glm5next_hc_mix_bf16's:
// - Ownership: thread tid handles k = tid + 256 j, j ascending, as there; it reads x[k] once and
//   feeds the same value to the RMS chain and to every row's dot chain.
// - Chains: the RMS partial is ss += v * v from 0.f, and row m's dot partial is
//   acc[m] += __bfloat162float(hc_fn[m, k]) * v from 0.f, each in ascending j, one rounded
//   multiply then one rounded add per term (common/ builds with --fmad=false, KERNEL.toml).
//   The rows' chains are independent registers, so interleaving them changes no chain.
// - Tree: red[a][e] += red[a][e + s] for e < s, s = 128 .. 1, one barrier per level, then
//   red[a][0]: glm_hc_block_reduce's pairing on each of the 1 + mix_hc arrays. Which thread
//   performs an add does not change its operands.
// - RMS once: each glm5next_hc_mix_bf16 block computes ssum and rsqrt from the same x with the
//   same code, so all mix_hc of them hold the same bits; computing it once is exact.
// - Epilogue: mix_out[t, m] = r * rsqrtf(ssum / (float)hc_dim + norm_eps), the same expression,
//   written once per (t, m).
// Requires mix_hc <= GLM_HC_MAX_MIX (hc_mult <= 4); the host falls back past it. Registers:
// GLM_HC_MAX_MIX accumulators plus the RMS partial, x and the loop state; x is not held across
// iterations. Shared: (1 + GLM_HC_MAX_MIX) * GLM_HC_BLOCK floats, 25.6 KB.
extern "C" __global__ void glm5next_hc_mix_bf16_tokmajor(
    const float* __restrict__ streams,
    const __nv_bfloat16* __restrict__ hc_fn,
    float* __restrict__ mix_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const float norm_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc_dim = hc_mult * hidden_size;
    const unsigned int mix_hc = (2 + hc_mult) * hc_mult;

    const float* x = streams + (size_t)t * hc_dim;
    // 2026-10-01: Array 0 is the RMS partial, array 1 + m row m's dot partial.
    __shared__ float red[1 + GLM_HC_MAX_MIX][GLM_HC_BLOCK];


    float ss = 0.f;
    float acc[GLM_HC_MAX_MIX];
#pragma unroll
    for (unsigned int m = 0; m < GLM_HC_MAX_MIX; ++m) acc[m] = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
        const float v = (float)x[k];
        ss += v * v;
#pragma unroll
        for (unsigned int m = 0; m < GLM_HC_MAX_MIX; ++m) {
            if (m < mix_hc) acc[m] += __bfloat162float(hc_fn[(size_t)m * hc_dim + k]) * v;
        }
    }
    red[0][tid] = ss;
#pragma unroll
    for (unsigned int m = 0; m < GLM_HC_MAX_MIX; ++m) {
        if (m < mix_hc) red[1 + m][tid] = acc[m];
    }
    __syncthreads();


    // 2026-10-01: glm_hc_block_reduce on 1 + mix_hc arrays at once; at level s the block
    // shares the (1 + mix_hc) * s adds. red[a][e] (e < s) is written by one thread and
    // red[a][e + s] is not written at this level, so no add races.
    const unsigned int nred = 1 + mix_hc;
    for (unsigned int s = GLM_HC_BLOCK / 2; s > 0; s >>= 1) {
        for (unsigned int i = tid; i < nred * s; i += GLM_HC_BLOCK) {
            const unsigned int a = i / s;
            const unsigned int e = i - a * s;
            red[a][e] += red[a][e + s];
        }
        __syncthreads();
    }


    if (tid < mix_hc) {
        const float ssum = red[0][0];
        const float rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
        const float r = red[1 + tid][0];
        mix_out[(size_t)t * mix_hc + tid] = r * rsqrt;
    }
}















// 2026-09-25: glm5next_hc_finish: mix [T, mix_hc] -> y_out [T, H], post_out [T, hc],
// comb_out [T, hc, hc]; the part of glm5next_hc_pre after the mixes, read from global
// memory. Grid (T, NB, 1): block y == 0 computes post, comb and the Sinkhorn, and the
// collapse y[d] = sum_i pre[i] * x[i, d] is split over blocks 1..NB-1, or done by block 0
// when NB == 1. glm_hc_pre launches NB = 1 + collapse_blocks(H) (model-arch
// glm5next_mhc.rs). Every block computes `pre` itself from the same mix row, so the
// collapse blocks do not depend on block 0. Each output element's arithmetic and order are
// the fused kernel's, whatever NB is.
extern "C" __global__ void glm5next_hc_finish(
    const float* __restrict__ streams,
    const float* __restrict__ mix,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float hc_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int by = blockIdx.y;
    const unsigned int nb = gridDim.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int mix_hc = (2 + hc) * hc;

    const float* x = streams + (size_t)t * hc * H;
    const float* s_mix = mix + (size_t)t * mix_hc;
    __shared__ float s_pre[GLM_HC_MAX_MULT];




    // 2026-09-25: `comb` lives in shared memory: indexed with the runtime `hc`, a local array
    // cannot be register-allocated and would sit in local memory.
    __shared__ float comb[GLM_HC_MAX_MULT * GLM_HC_MAX_MULT];






    // 2026-09-25: One thread per row, then one per column. Each thread sums in the fused
    // kernel's order (row i over j ascending, column j over i ascending, both starting from
    // hc_eps) and divides, so the results equal the fused kernel's. A division is never
    // replaced by a multiply with 1 / r: that rounds differently.
    const bool lane = tid < hc;


    if (lane) {
        const unsigned int i = tid;
        float pr = s_mix[i] * hc_scale[0] + hc_base[i];
        s_pre[i] = 1.f / (1.f + expf(-pr)) + hc_eps;
    }


    if (by == 0) {
        if (lane) {
            const unsigned int i = tid;
            float po = s_mix[hc + i] * hc_scale[1] + hc_base[hc + i];
            post_out[(size_t)t * hc + i] = 2.f * (1.f / (1.f + expf(-po)));
            for (unsigned int j = 0; j < hc; ++j)
                comb[i * hc + j] =
                    s_mix[2 * hc + i * hc + j] * hc_scale[2] + hc_base[2 * hc + i * hc + j];
        }
        __syncthreads();


        if (lane) {
            const unsigned int i = tid;
            float mx = -1e30f;
            for (unsigned int j = 0; j < hc; ++j) mx = fmaxf(mx, comb[i * hc + j]);
            float sum = 0.f;
            for (unsigned int j = 0; j < hc; ++j) {
                float e = expf(comb[i * hc + j] - mx);
                comb[i * hc + j] = e;
                sum += e;
            }
            for (unsigned int j = 0; j < hc; ++j)
                comb[i * hc + j] = comb[i * hc + j] / sum + hc_eps;
        }
        __syncthreads();


        if (lane) {
            const unsigned int j = tid;
            float c = hc_eps;
            for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
            for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
        }
        __syncthreads();


        for (unsigned int it = 0; it + 1 < sinkhorn_iters; ++it) {
            if (lane) {
                const unsigned int i = tid;
                float r = hc_eps;
                for (unsigned int j = 0; j < hc; ++j) r += comb[i * hc + j];
                for (unsigned int j = 0; j < hc; ++j) comb[i * hc + j] /= r;
            }
            __syncthreads();
            if (lane) {
                const unsigned int j = tid;
                float c = hc_eps;
                for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
                for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
            }
            __syncthreads();
        }
        // 2026-09-25: No exact column projection after the loop, as in glm5next_hc_pre.
        for (unsigned int k = tid; k < hc * hc; k += GLM_HC_BLOCK)
            comb_out[(size_t)t * hc * hc + k] = comb[k];
    }
    __syncthreads();




    if (nb == 1 || by > 0) {
        const unsigned int slot = (nb == 1) ? 0 : by - 1;
        const unsigned int nslot = (nb == 1) ? 1 : nb - 1;
        for (unsigned int d = slot * GLM_HC_BLOCK + tid; d < H; d += nslot * GLM_HC_BLOCK) {
            float acc = 0.f;
            for (unsigned int i = 0; i < hc; ++i) acc += s_pre[i] * (float)x[i * H + d];
            y_out[(size_t)t * H + d] = __float2bfloat16(acc);
        }
    }
}















// 2026-09-25: glm5next_hc_post_ref: out[t, j, d] = post[t, j] * block_out[t, d]
// + sum_i comb[t, i, j] * residual[t, i, d]. Grid (T, 1, 1). `out` may alias `residual`:
// the thread that owns column d reads all hc residual values of d before it writes d.
// block_out is [T, H] BF16; residual and out are [T, hc, H], post [T, hc], comb [T, hc, hc],
// all FP32. Same arithmetic as hyper_connection::hc_post. The host launches
// glm5next_hc_post instead; this kernel is the oracle of glm5next_hc_post_gate.
extern "C" __global__ void glm5next_hc_post_ref(
    const __nv_bfloat16* __restrict__ block_out,
    const float* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    float* __restrict__ out,
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    const __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    float* o = out + (size_t)t * hc * H;

    for (unsigned int d = tid; d < H; d += GLM_HC_BLOCK) {
        float xd = (float)x[d];
        float rv[GLM_HC_MAX_MULT];
        for (unsigned int i = 0; i < hc; ++i) rv[i] = res[i * H + d];
        for (unsigned int j = 0; j < hc; ++j) {
            float acc = p[j] * xd;
            for (unsigned int i = 0; i < hc; ++i) acc += c[i * hc + j] * rv[i];
            o[j * H + d] = acc;
        }
    }
}










// 2026-09-25: glm5next_hc_post: glm5next_hc_post_ref with two changes that keep every value:
// - Loops run to the compile-time GLM_HC_MAX_MULT with an `i < hc` guard, so `rv` can be
//   register-allocated; the executed arithmetic and its order (i ascending) are the
//   reference's.
// - Grid (T, NB, 1): columns d are spread over blockIdx.y. Aliasing `out` with `residual`
//   stays safe: the thread for column d reads res[i*H + d] and writes o[j*H + d] only, and
//   those addresses meet only for the same d. glm_hc_post launches NB = collapse_blocks(H)
//   (model-arch glm5next_mhc.rs).
extern "C" __global__ void glm5next_hc_post(
    const __nv_bfloat16* __restrict__ block_out,
    const float* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    float* __restrict__ out,
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    const __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    float* o = out + (size_t)t * hc * H;


    // 2026-09-25: post and comb are read into shared memory once per block.
    __shared__ float s_p[GLM_HC_MAX_MULT];
    __shared__ float s_c[GLM_HC_MAX_MULT * GLM_HC_MAX_MULT];
    if (tid < hc) s_p[tid] = p[tid];
    if (tid < hc * hc) s_c[tid] = c[tid];
    __syncthreads();

    const unsigned int stride = gridDim.y * GLM_HC_BLOCK;
    for (unsigned int d = blockIdx.y * GLM_HC_BLOCK + tid; d < H; d += stride) {
        float xd = (float)x[d];
        float rv[GLM_HC_MAX_MULT];
#pragma unroll
        for (unsigned int i = 0; i < GLM_HC_MAX_MULT; ++i)
            if (i < hc) rv[i] = res[i * H + d];
#pragma unroll
        for (unsigned int j = 0; j < GLM_HC_MAX_MULT; ++j) {
            if (j < hc) {
                float acc = s_p[j] * xd;
#pragma unroll
                for (unsigned int i = 0; i < GLM_HC_MAX_MULT; ++i)
                    if (i < hc) acc += s_c[i * hc + j] * rv[i];
                o[j * H + d] = acc;
            }
        }
    }
}











// 2026-09-25: glm5next_hc_head: y_out[t, d] = (sum_i streams[t, i, d]) * (1 / hc), an
// unweighted mean with no weight arguments; the sum is scaled once, by the reciprocal.
// hyper_connection::hc_head is a learned weighted sum reading head_fn, head_scale and
// head_base. Grid (T, 1, 1).
extern "C" __global__ void glm5next_hc_head(
    const float* __restrict__ streams,
    __nv_bfloat16* __restrict__ y_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    const float* x = streams + (size_t)t * hc * H;
    const float inv = 1.0f / (float)hc;

    for (unsigned int d = tid; d < H; d += GLM_HC_BLOCK) {
        float acc = 0.f;
        for (unsigned int i = 0; i < hc; ++i) acc += x[i * H + d];
        y_out[(size_t)t * H + d] = __float2bfloat16(acc * inv);
    }
}










// 2026-09-25: glm5next_hc_expand: streams[t, i, d] = hidden[t, d] for every i < hc_mult,
// widening the BF16 embedding to the FP32 highway. Same broadcast as
// hyper_connection::hc_expand, which lives in kernels/gb10/deepseek-v4-flash/nvfp4/, a model
// directory a GLM target does not compile. Grid (T, 1, 1).
extern "C" __global__ void glm5next_hc_expand(
    const __nv_bfloat16* __restrict__ hidden,
    float* __restrict__ streams,
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const __nv_bfloat16* x = hidden + (size_t)t * H;
    float* s = streams + (size_t)t * hc_mult * H;
    for (unsigned int d = tid; d < H; d += GLM_HC_BLOCK) {
        float v = (float)x[d];
        for (unsigned int i = 0; i < hc_mult; ++i) s[i * H + d] = v;
    }
}
