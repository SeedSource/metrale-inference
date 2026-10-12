// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-12: FP16-storage twins of the three KDA decode-row recurrent kernels the GLM-5.3 car
// launches on decode and verify, for METRALE_GLM_KDA_STATE_F16=1
// (crates/model-arch/src/glm5next_kda/state_f16.rs):
//   kda_recurrent_decode_bf16_smem_f16        <- kda_recurrent_decode_bf16_smem (kda_recurrent.cu)
//   kda_recurrent_decode_bf16_smem_io_f16     <- kda_recurrent_decode_bf16_smem_io (kda_snap_fuse.cu)
//   kda_recurrent_decode_bf16_smem_defer_f16  <- kda_recurrent_decode_bf16_smem_defer (kda_snap_fuse.cu)
//
// Owner: gb10 kernels.
// Invariants:
// - Only the recurrent state's storage changes: [H, D, D] __half, K-major (S[k * D + v]), the
//   FP32 parents' layout element for element. q, k, v (BF16), gate, beta and out (FP32), the grid
//   (H, D / VPB), the block (VPB), the shared memory (3 * D + VPB * (D + 1) floats) and the
//   launcher contract (VPB divides D, blockDim.x == VPB) are the parents'.
// - Every state element is loaded with gdn_f16_load (exactly __half2float) and then goes through
//   the parent's expressions, in the parent's order, in FP32 registers and FP32 shared memory;
//   `out` is computed from the FP32 value before it is stored. Each updated element is stored
//   once, with gdn_f16_store (gdn_f16_state.cuh: clamp to +-65504, then round to nearest even).
//   So, inside the FP16 range, a launch equals its FP32 parent run on the widened state followed
//   by ssm_h_state_f32_to_f16 (ssm_h_dtype.cu) of the written state, bit for bit, and `out`
//   equals the parent's `out` bit for bit (glm_kda_state_f16_microtest checks both).
// - A K-row walk therefore rounds the state to FP16 after every row, before the next row reads
//   it, whether the rows run in place (_f16), ping-pong through snapshot slots (_io_f16) or start
//   from a device-chosen intermediate (_defer_f16): the fused walk equals the walk of single-row
//   launches with copies, by construction.
// - Same build flags as every common/ kernel (--fmad=false, common/KERNEL.toml): no multiply-add
//   is contracted here or in the parents.
// - _io_f16: state_in and state_out must not overlap (__restrict__), as in the parent.
// - _defer_f16: the aliasing argument of kda_recurrent_decode_bf16_smem_defer holds unchanged
//   (thread (h, vi) alone touches column vi of head h, and reads all of it before writing any of
//   it), so state_out may equal the chosen input; state_inter_stride is in __half elements.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

#include "gdn_f16_state.cuh"

// 2026-10-12: kda_recurrent_decode_bf16_smem with an FP16 state, updated in place.
extern "C" __global__ void kda_recurrent_decode_bf16_smem_f16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __half* __restrict__ state,
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
    __half* S = state + hd * D;
    const unsigned int vi = v0 + threadIdx.x;
    if (threadIdx.x >= VPB || vi >= D) return;
    float* col = sh_s + (size_t)threadIdx.x * col_stride;

    float kv = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = gdn_f16_load(&S[(size_t)kk * D + vi]) * sh_decay[kk];
        col[kk] = s;
        kv += s * sh_k[kk];
    }
    const float delta = (__bfloat162float(v[hd + vi]) - kv) * b;
    float o = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = col[kk] + sh_k[kk] * delta;
        S[(size_t)kk * D + vi] = gdn_f16_store(s);
        o += s * sh_q[kk];
    }
    out[hd + vi] = o;
}

// 2026-10-12: kda_recurrent_decode_bf16_smem_io with an FP16 state: pass 1 reads state_in, pass 2
// writes state_out.
extern "C" __global__ void kda_recurrent_decode_bf16_smem_io_f16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __half* __restrict__ state_in,
    __half* __restrict__ state_out,
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
    const __half* S_in = state_in + hd * D;
    __half* S_out = state_out + hd * D;
    const unsigned int vi = v0 + threadIdx.x;
    if (threadIdx.x >= VPB || vi >= D) return;
    float* col = sh_s + (size_t)threadIdx.x * col_stride;

    float kv = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = gdn_f16_load(&S_in[(size_t)kk * D + vi]) * sh_decay[kk];
        col[kk] = s;
        kv += s * sh_k[kk];
    }
    const float delta = (__bfloat162float(v[hd + vi]) - kv) * b;
    float o = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = col[kk] + sh_k[kk] * delta;
        S_out[(size_t)kk * D + vi] = gdn_f16_store(s);
        o += s * sh_q[kk];
    }
    out[hd + vi] = o;
}

// 2026-10-12: kda_recurrent_decode_bf16_smem_defer with an FP16 state: the input is intermediate
// *defer_idx (state_inter0 + idx * state_inter_stride, in __half elements) when it is >= 0, else
// the live state; the output goes to state_out. No __restrict__ on the aliasing trio.
extern "C" __global__ void kda_recurrent_decode_bf16_smem_defer_f16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __half* state_live,
    const __half* state_inter0,
    unsigned long long state_inter_stride,
    const int* __restrict__ defer_idx,
    __half* state_out,
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
    const int di = *defer_idx;
    const __half* state_in =
        (di >= 0) ? state_inter0 + (size_t)di * state_inter_stride : state_live;

    float* sh_decay = sh;
    float* sh_k = sh + D;
    float* sh_q = sh + 2u * D;
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
    const __half* S_in = state_in + hd * D;
    __half* S_out = state_out + hd * D;
    const unsigned int vi = v0 + threadIdx.x;
    if (threadIdx.x >= VPB || vi >= D) return;
    float* col = sh_s + (size_t)threadIdx.x * col_stride;

    float kv = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = gdn_f16_load(&S_in[(size_t)kk * D + vi]) * sh_decay[kk];
        col[kk] = s;
        kv += s * sh_k[kk];
    }
    const float delta = (__bfloat162float(v[hd + vi]) - kv) * b;
    float o = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = col[kk] + sh_k[kk] * delta;
        S_out[(size_t)kk * D + vi] = gdn_f16_store(s);
        o += s * sh_q[kk];
    }
    out[hd + vi] = o;
}
