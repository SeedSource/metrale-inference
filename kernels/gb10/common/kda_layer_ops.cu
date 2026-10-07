// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Glue kernels for the KDA layer: the sigmoid-gated output RMSNorm, the q|k|v
// pack and split, the beta sigmoid, and a device-side fill.
//
// Owner: gb10 kernels.
// Invariants:
// - kda_o_norm_gated_*: one block per row of head_dim elements (blockIdx.x is the row);
//   out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i] * sigmoid(gate[i]). The block must
//   be a whole number of warps and at most 1024 threads: the reductions shuffle with a
//   full mask into red[32]. The host launches block = head_dim, which
//   Glm5NextKdaConfig::validate requires to be 128.
// - kda_split_widen and kda_pack_qkv_bf16: blockIdx.y is the token and each thread one
//   channel; tokens t >= T are not written.
//
// The norm does not reuse gated_rms_norm_f32_input (rms_norm.cu): that kernel applies
// SiLU to the gate, g / (1 + expf(-g)), and this one applies sigmoid.


#include <cuda_bf16.h>
#include <cuda_fp8.h>





#define KDA_ONORM_BODY(GATE_LD, W_LD, OUT_ST)                                                 \
    const unsigned int row = blockIdx.x;                                                      \
    const unsigned int tid = threadIdx.x;                                                     \
    const float* x = input + (size_t)row * head_dim;                                          \
    float acc = 0.0f;                                                                         \
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) { float f = x[i]; acc += f * f; }\
    __shared__ float red[32];                                                                 \
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);     \
    if ((tid & 31) == 0) red[tid >> 5] = acc;                                                 \
    __syncthreads();                                                                          \
    if (tid < 32) {                                                                           \
        float v = (tid < ((blockDim.x + 31) / 32)) ? red[tid] : 0.0f;                          \
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);      \
        if (tid == 0) red[0] = v;                                                             \
    }                                                                                         \
    __syncthreads();                                                                          \
    const float inv = rsqrtf(red[0] / (float)head_dim + eps);                                 \
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {                                \
        const float g = (GATE_LD);                                                            \
        const float s = 1.0f / (1.0f + __expf(-g));                    \
        OUT_ST(x[i] * inv * (W_LD) * s);                                                      \
    }

// 2026-09-25: The entry point glm5next_kda resolves: FP32 input; BF16 gate, weight and output.
extern "C" __global__ void kda_o_norm_gated_bf16(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int head_dim,
    float eps
) {
#define KDA_ONORM_OUT_BF16(val) output[(size_t)row * head_dim + i] = __float2bfloat16(val)
    KDA_ONORM_BODY(__bfloat162float(gate[(size_t)row * head_dim + i]),
                   __bfloat162float(weight[i]),
                   KDA_ONORM_OUT_BF16)
#undef KDA_ONORM_OUT_BF16
}


extern "C" __global__ void kda_o_norm_gated_f32(
    const float* __restrict__ input,
    const float* __restrict__ gate,
    const float* __restrict__ weight,
    float* __restrict__ output,
    unsigned int head_dim,
    float eps
) {
#define KDA_ONORM_OUT_F32(val) output[(size_t)row * head_dim + i] = (val)
    KDA_ONORM_BODY(gate[(size_t)row * head_dim + i], weight[i], KDA_ONORM_OUT_F32)
#undef KDA_ONORM_OUT_F32
}

// 2026-10-07: METRALE_GLM_NORM_FP8_QUANT_FUSE: kda_o_norm_gated_*_fp8q write what
// kda_o_norm_gated_bf16 / kda_o_norm_gated_bf16in write PLUS the FP8 E4M3 activation and FP32
// scale that per_token_group_quant_fp8 (kernels/gb10/common/per_token_group_quant_fp8.cu, and
// its Hopper twin) would make from that BF16 output, so the o_proj W8A8 GEMM skips the
// quantizer launch and its BF16 re-read of the whole [T, qkv] activation.
// Invariants (the fused kernels):
// - blockDim.x == head_dim == 128: one thread per element and one block per quantizer
//   K-group, so block `row` (= token * heads + head) is group `row % heads` of token
//   `row / heads`. The quantizer's [M, K/128] scale array (K = heads * 128) is then indexed by
//   `row`, and its [M, K] bytes by `row * 128 + tid`.
// - The BF16 output is computed by the same statements as the unfused kernel (same reduction
//   order, same `x * inv * w * s` association), and the quantizer then runs on the BF16-ROUNDED
//   value (`__float2bfloat16`, widened back), exactly the bytes a separate launch would read.
//   The scale (`amax / 448.0f`, 1e-12f floor), the per-element division by the scale, the
//   clamp and the saturating `__nv_cvt_float_to_fp8` are the quantizer's. `fmaxf` is exact and
//   drops a NaN operand and the amax is seeded with 0.0f, as there, so the reduction tree
//   cannot change the result.
// - Gate: crates/model-arch/examples/kda_o_norm_fp8q_microtest.rs (bitwise).
#define KDA_FP8Q_MAX 448.0f

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// 2026-10-07: SCALE and HIP builds encode E4M3 in software, as per_token_group_quant_fp8.cu does.
__device__ __forceinline__ unsigned char kda_fp8q_encode(float v) {
    if (v != v) return 0x7F;
    unsigned int bb = __float_as_uint(v); unsigned int sign = (bb >> 31) & 1u;
    int e = (int)((bb >> 23) & 0xFF) - 127; unsigned int man = bb & 0x7FFFFFu;
    int ee = e + 7; unsigned int em;
    if (ee < 1) { ee = 0; em = 0; if (e >= -10) { float a = v < 0 ? -v : v; em = (unsigned int)(a / 0.001953125f + 0.5f); if (em > 7u) em = 7u; } }
    else if (ee > 15) { ee = 15; em = 6; }
    else { em = (man + (1u << 19)) >> 20; if (em > 7u) { em = 0; ee++; if (ee > 15) { ee = 15; em = 6; } } }
    return (unsigned char)((sign << 7) | ((unsigned)ee << 3) | em);
}
#else
__device__ __forceinline__ unsigned char kda_fp8q_encode(float v) {
    return (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
#endif

// 2026-10-07: Quantize the block's 128 BF16-rounded values `v` (thread `tid` holds element
// `tid`) as one K-group: scale into `scale[row]`, byte into `fp8[row * 128 + tid]`. Every
// thread of the 128-thread block must call it (it synchronizes).
__device__ __forceinline__ void kda_onorm_fp8_group(
    float v,
    unsigned int row,
    unsigned char* __restrict__ fp8,
    float* __restrict__ scale
) {
    __shared__ float qred[4];
    const unsigned int tid = threadIdx.x;
    float a = fmaxf(0.0f, fabsf(v));
    for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));
    if ((tid & 31) == 0) qred[tid >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(fmaxf(qred[0], qred[1]), fmaxf(qred[2], qred[3]));
    float sc = amax / KDA_FP8Q_MAX;
    if (sc < 1e-12f) sc = 1e-12f;
    if (tid == 0) scale[row] = sc;
    float q = v / sc;
    q = fmaxf(fminf(q, KDA_FP8Q_MAX), -KDA_FP8Q_MAX);
    fp8[(size_t)row * 128 + tid] = kda_fp8q_encode(q);
}

// 2026-10-07: KDA_ONORM_BODY's reduction and per-element expression, one element per thread
// (blockDim.x == head_dim), ending in the quantizer instead of the plain store. X_LD, GATE_LD
// and W_LD read element `tid` of the input, gate and weight as float.
#define KDA_ONORM_FP8Q_BODY(X_LD, GATE_LD, W_LD)                                              \
    const unsigned int row = blockIdx.x;                                                      \
    const unsigned int tid = threadIdx.x;                                                     \
    const float xv = (X_LD);                                                                  \
    float acc = 0.0f;                                                                         \
    acc += xv * xv;                                                                           \
    __shared__ float red[32];                                                                 \
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);     \
    if ((tid & 31) == 0) red[tid >> 5] = acc;                                                 \
    __syncthreads();                                                                          \
    if (tid < 32) {                                                                           \
        float v = (tid < ((blockDim.x + 31) / 32)) ? red[tid] : 0.0f;                          \
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);      \
        if (tid == 0) red[0] = v;                                                             \
    }                                                                                         \
    __syncthreads();                                                                          \
    const float inv = rsqrtf(red[0] / (float)head_dim + eps);                                 \
    const float g = (GATE_LD);                                                                \
    const float s = 1.0f / (1.0f + __expf(-g));                                               \
    const __nv_bfloat16 ob = __float2bfloat16(xv * inv * (W_LD) * s);                          \
    output[(size_t)row * head_dim + tid] = ob;                                                \
    kda_onorm_fp8_group(__bfloat162float(ob), row, out_fp8, out_scale);

// 2026-10-07: kda_o_norm_gated_bf16 (FP32 input) plus the quantizer's FP8 bytes and scales.
extern "C" __global__ void kda_o_norm_gated_bf16_fp8q(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned char* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    unsigned int head_dim,
    float eps
) {
    KDA_ONORM_FP8Q_BODY(input[(size_t)row * head_dim + tid],
                        __bfloat162float(gate[(size_t)row * head_dim + tid]),
                        __bfloat162float(weight[tid]))
}

// 2026-10-07: kda_o_norm_gated_bf16in (BF16 input, the FlashKDA prefill's) plus the same.
extern "C" __global__ void kda_o_norm_gated_bf16in_fp8q(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned char* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    unsigned int head_dim,
    float eps
) {
    KDA_ONORM_FP8Q_BODY(__bfloat162float(input[(size_t)row * head_dim + tid]),
                        __bfloat162float(gate[(size_t)row * head_dim + tid]),
                        __bfloat162float(weight[tid]))
}

// 2026-09-25: Splits the [T, 3 * qkv] BF16 q|k|v rows into three FP32 [T_pad, qkv] buffers;
// BF16 to FP32 is exact. Tokens t >= T are not written, so the pad tail keeps what the
// caller put there: zero in production, or a poison value from the pad-guard test through
// prefill_with_pad_fill.




extern "C" __global__ void kda_split_widen(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    unsigned int T,
    unsigned int qkv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int t = blockIdx.y;
    if (ch >= qkv || t >= T) return;
    const size_t s = (size_t)t * 3 * qkv + ch;
    const size_t d = (size_t)t * qkv + ch;
    q[d] = __bfloat162float(src[s]);
    k[d] = __bfloat162float(src[s + qkv]);
    v[d] = __bfloat162float(src[s + 2 * qkv]);
}



extern "C" __global__ void kda_fill_f32(float* __restrict__ dst, unsigned int n, float value) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}

// 2026-09-25: beta = sigmoid(b_proj(hidden)), BF16 in and FP32 out; the KDA kernels take
// beta already sigmoided.
extern "C" __global__ void kda_sigmoid_bf16_f32(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(src[i]);
        dst[i] = 1.0f / (1.0f + __expf(-x));
    }
}

// 2026-09-25: Interleaves three [T, qkv] projections into the [T, 3 * qkv] q|k|v layout the
// depthwise conv reads. The projections are separate GEMMs because dense_gemm_bf16 writes
// C[row * N + col] with no output stride: aimed at offsets inside one [T, 3 * qkv] buffer
// they would overwrite each other for T > 1, and agree at T = 1, where a decode-only test
// would not see it.


extern "C" __global__ void kda_pack_qkv_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ dst,
    unsigned int T,
    unsigned int qkv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int t = blockIdx.y;
    if (ch >= qkv || t >= T) return;
    const size_t s = (size_t)t * qkv + ch;
    const size_t d = (size_t)t * 3 * qkv + ch;
    dst[d] = q[s];
    dst[d + qkv] = k[s];
    dst[d + 2 * qkv] = v[s];
}
