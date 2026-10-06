// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GLM-5.3 MoE FFN kernels: clamped SwiGLU, router top-k and the expert combines.
//
// Owner: gb10 kernels.
// Invariants:
// - The SwiGLU clamp is asymmetric: the gate is bounded above only and `up` on both sides,
//   out = silu(min(gate, limit)) * clamp(up, -limit, limit), computed in FP32.
// - glm5next_router_topk ranks experts by sigmoid(logit) + bias but weights them by the
//   unbiased sigmoid(logit); an exact tie goes to the lower expert id.
// - The combines add the shared expert unscaled; routed_scale is already in the weights.
//
// moe_silu_mul (moe_silu_mul.cu) does not clamp, so GLM has its own entry points. The host
// resolves them with kernel(), not try_kernel, and Glm5NextMlpConfig::validate refuses a
// swiglu_limit <= 0 (model-arch glm5next_mlp/mod.rs).











#include <cuda_bf16.h>

// 2026-09-26: One thread per element; the host launches ceil(n / 256) blocks of 256
// (swiglu in model-arch glm5next_mlp/forward/launch.rs).
extern "C" __global__ void glm5next_swiglu_clamp(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    const unsigned int n,
    const float limit
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = (float)gate[i];
    float u = (float)up[i];
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    float s = g / (1.0f + expf(-g));
    out[i] = __float2bfloat16(s * u);
}


// 2026-09-25: glm5next_swiglu_clamp with an FP32 output. No host code launches it.
extern "C" __global__ void glm5next_swiglu_clamp_f32out(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    float* __restrict__ out,
    const unsigned int n,
    const float limit
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = (float)gate[i];
    float u = (float)up[i];
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    float s = g / (1.0f + expf(-g));
    out[i] = s * u;
}




















// 2026-09-25: glm5next_router_topk, one block per token t:
//   s[e]       = sigmoid(logits[t, e])
//   choice[e]  = s[e] + bias[e]                  (selection only)
//   ids        = the top_k experts by choice, an exact tie to the lower id
//   weights[k] = s[ids[k]], divided by (sum + 1e-20) if `renormalize`, times routed_scale
// logits is [T, num_experts] FP32, bias [num_experts], ids and weights [T, top_k].
// bf16_ladder != 0 rounds s, choice, the running sum and each weight step to BF16. The
// model passes 1 only for Glm5NextRouterMode::VllmBf16; the config parser's default is
// HfFp32 (config parsers/glm5_next/parse.rs), which passes 0.
// n_group != 1 returns without writing anything. The config parser refuses such a
// checkpoint and the host passes 1.
// Otherwise every one of the token's top_k slots is written; a slot no expert fills gets
// id -1 and weight 0. The per-slot sum runs sequentially in thread 0, in slot order.
// Block: a power of two, at most 1024 threads (the tree reduction halves blockDim). The
// host launches 256 threads (ACT_BLOCK) and Glm5NextMlpConfig::validate keeps
// 1 <= top_k <= 16 and top_k <= num_experts.
extern "C" __global__ void glm5next_router_topk(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    const unsigned int num_experts,
    const unsigned int top_k,
    const unsigned int n_group,
    const float routed_scale,
    const unsigned int renormalize,
    const unsigned int bf16_ladder
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (n_group != 1) return;

    const float* row = logits + (size_t)t * num_experts;
    int* ids = topk_ids + (size_t)t * top_k;
    float* wts = topk_weights + (size_t)t * top_k;

    // 2026-09-25: Reduction scratch for up to 1024 threads. A thread without a candidate
    // reports id INT_MAX, which loses every tie-break against a real expert.
    __shared__ float red_c[1024];
    __shared__ int   red_i[1024];
    // 2026-09-25: Up to 16 slots; the host refuses top_k > 16 (KERNEL_MAX_TOP_K).
    __shared__ int   sel_id[16];
    __shared__ float sel_w[16];


    if (tid == 0) {
        for (unsigned int k = 0; k < top_k; ++k) {
            ids[k] = -1;
            wts[k] = 0.0f;
        }
    }
    __syncthreads();

    for (unsigned int k = 0; k < top_k; ++k) {
        float best = -1e30f;
        int arg = 2147483647;
        for (unsigned int e = tid; e < num_experts; e += blockDim.x) {
            bool taken = false;
            for (unsigned int j = 0; j < k; ++j)
                if (sel_id[j] == (int)e) { taken = true; break; }
            if (taken) continue;
            float s = 1.0f / (1.0f + expf(-row[e]));
            if (bf16_ladder) s = (float)__float2bfloat16(s);
            float c = s + bias[e];
            if (bf16_ladder) c = (float)__float2bfloat16(c);
            if (c > best || (c == best && (int)e < arg)) { best = c; arg = (int)e; }
        }
        red_c[tid] = best;
        red_i[tid] = arg;
        __syncthreads();
        for (unsigned int w = blockDim.x >> 1; w > 0; w >>= 1) {
            if (tid < w) {
                const float oc = red_c[tid + w];
                const int   oi = red_i[tid + w];
                if (oc > red_c[tid] || (oc == red_c[tid] && oi < red_i[tid])) {
                    red_c[tid] = oc;
                    red_i[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int win = red_i[0];
            if (win == 2147483647) {
                // 2026-09-25: No candidate left; unreachable while top_k <= num_experts.

                sel_id[k] = -1;
                sel_w[k] = 0.0f;
            } else {
                sel_id[k] = win;

                float s = 1.0f / (1.0f + expf(-row[win]));
                if (bf16_ladder) s = (float)__float2bfloat16(s);
                sel_w[k] = s;
            }
        }
        __syncthreads();
    }

    if (tid != 0) return;

    float sum = 0.0f;
    for (unsigned int k = 0; k < top_k; ++k) {
        sum += sel_w[k];
        if (bf16_ladder) sum = (float)__float2bfloat16(sum);
    }
    for (unsigned int k = 0; k < top_k; ++k) {
        float w = sel_w[k];
        if (renormalize) {
            w = w / (sum + 1e-20f);
            if (bf16_ladder) w = (float)__float2bfloat16(w);
        }
        w *= routed_scale;
        if (bf16_ladder) w = (float)__float2bfloat16(w);
        ids[k] = sel_id[k];
        wts[k] = w;
    }
}





// 2026-09-25: glm5next_moe_combine: out[t, d] = shared[t, d] + sum_k weights[t, k] *
// expert_out[t, k, d], one FP32 accumulator in slot order and one rounding to BF16.
// expert_out is [T, top_k, hidden], weights [T, top_k], shared and out [T, hidden].
// Grid (T, 1, 1); the block's threads stride over hidden.
extern "C" __global__ void glm5next_moe_combine(
    const __nv_bfloat16* __restrict__ expert_out,
    const float* __restrict__ weights,
    const __nv_bfloat16* __restrict__ shared,
    __nv_bfloat16* __restrict__ out,
    const unsigned int hidden,
    const unsigned int top_k
) {
    const unsigned int t = blockIdx.x;
    const __nv_bfloat16* eo = expert_out + (size_t)t * top_k * hidden;
    const float* w = weights + (size_t)t * top_k;
    for (unsigned int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < top_k; ++k) acc += w[k] * (float)eo[k * hidden + d];
        acc += (float)shared[(size_t)t * hidden + d];
        out[(size_t)t * hidden + d] = __float2bfloat16(acc);
    }
}








// 2026-09-25: glm5next_moe_combine_indexed: glm5next_moe_combine over expert-sorted routed
// outputs. Slot k of token t is read from row token_to_perm[t * top_k + k] of expert_out
// ([total_expanded, hidden]) instead of row t * top_k + k. The accumulation order and the
// single rounding are glm5next_moe_combine's, so the two differ only in where they read.
// Grid (T, 1, 1); the block's threads stride over hidden.
extern "C" __global__ void glm5next_moe_combine_indexed(
    const __nv_bfloat16* __restrict__ expert_out,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ weights,
    const __nv_bfloat16* __restrict__ shared,
    __nv_bfloat16* __restrict__ out,
    const unsigned int hidden,
    const unsigned int top_k
) {
    const unsigned int t = blockIdx.x;
    const int* __restrict__ perm = token_to_perm + (size_t)t * top_k;
    const float* w = weights + (size_t)t * top_k;
    for (unsigned int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < top_k; ++k)
            acc += w[k] * (float)expert_out[(size_t)perm[k] * hidden + d];
        acc += (float)shared[(size_t)t * hidden + d];
        out[(size_t)t * hidden + d] = __float2bfloat16(acc);
    }
}

// 2026-10-06: glm5next_moe_combine_indexed_local: glm5next_moe_combine_indexed that skips slot k
// when its sorted row perm[k] lies outside this rank's experts, [expert_offsets[local_first],
// expert_offsets[local_end]) (rows are sorted by expert id, so that span is contiguous), so the
// caller need not zero expert_out for the experts another EP rank owns
// (METRALE_GLM_MOE_COMBINE_LOCAL, model-arch glm5next_mlp/forward.rs). Same output bits as
// glm5next_moe_combine_indexed over a zeroed expert_out:
// - A skipped slot there adds w[k] * 0.0f = +-0.0f (w finite). acc starts at +0.0f, and under
//   round-to-nearest a sum is -0.0f only when both addends are -0.0f, so acc is never -0.0f and
//   acc + (+-0.0f) == acc bit for bit. The other slots run the same adds in the same k order,
//   then the shared add and the single rounding.
// - Local slots read the same rows (every grouped down path writes every row of every local
//   expert).
// Grid (T, 1, 1); the block's threads stride over hidden.
extern "C" __global__ void glm5next_moe_combine_indexed_local(
    const __nv_bfloat16* __restrict__ expert_out,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ weights,
    const __nv_bfloat16* __restrict__ shared,
    __nv_bfloat16* __restrict__ out,
    const unsigned int hidden,
    const unsigned int top_k,
    const int* __restrict__ expert_offsets,
    const unsigned int local_first,
    const unsigned int local_end
) {
    const unsigned int t = blockIdx.x;
    const int* __restrict__ perm = token_to_perm + (size_t)t * top_k;
    const float* w = weights + (size_t)t * top_k;
    const int lo = expert_offsets[local_first];
    const int hi = expert_offsets[local_end];
    for (unsigned int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < top_k; ++k) {
            const int r = perm[k];
            if (r < lo || r >= hi) continue;
            acc += w[k] * (float)expert_out[(size_t)r * hidden + d];
        }
        acc += (float)shared[(size_t)t * hidden + d];
        out[(size_t)t * hidden + d] = __float2bfloat16(acc);
    }
}
