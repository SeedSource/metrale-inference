// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-06: Out-of-place twins of the two KDA decode-row kernels, for the fused verify
// snapshot (METRALE_GLM_KDA_SNAP_FUSE=1, crates/model-arch/src/glm5next_kda/snap_fuse.rs).
//
// The K-row MTP verify walks the KDA state one row at a time and, after each interior row,
// used to copy the recurrent state (FP32 [H, D, D]) and the conv state (FP32 [dim, d_conv])
// into that row's snapshot slot with two device-to-device memcpys. With these kernels the
// row reads the state from `*_in` and writes the advanced state to `*_out`, so an interior
// row writes its snapshot slot directly, the next row reads it from there, and the last row
// writes the live state. The memcpys disappear.
//
// Owner: gb10 kernels.
// Invariants:
// - causal_conv1d_update_l2norm_io is causal_conv1d_update_l2norm (causal_conv1d.cu) with the
//   state pointer split into conv_state_in / conv_state_out: the same expressions in the same
//   order on the same float values. Its window is formed in conv_state_out and the convolution
//   reads it back from there, exactly as the in-place kernel reads its updated window.
// - kda_recurrent_decode_bf16_smem_io is kda_recurrent_decode_bf16_smem (kda_recurrent.cu)
//   with the state pointer split into state_in (pass 1 reads) / state_out (pass 2 writes).
//   In the in-place kernel every state element is read by pass 1 before pass 2 of the same
//   thread overwrites it and no other thread touches it, so reading from a separate buffer
//   gives the same floats.
// - Both are built with --fmad=false (common/KERNEL.toml), as their in-place originals are,
//   so no multiply-add is contracted in either; the outputs and the written state are
//   bit-identical to the in-place kernel's (kda_snap_fuse_microtest checks it).
// - `*_in` and `*_out` must not overlap (__restrict__). snap_fuse.rs launches the in-place
//   originals whenever the two pointers are equal.
// - Launch contracts are the originals': the conv takes grid (ceil(dim / 256), batch), 256
//   threads, qk_channels % 256 == 0, head_dim 128; the recurrence takes grid (H, D / VPB),
//   VPB threads, VPB dividing D, 3 * D + VPB * (D + 1) floats of shared memory.

#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void causal_conv1d_update_l2norm_io(
    const float* __restrict__ conv_state_in,
    float* __restrict__ conv_state_out,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);

    const bool valid = (ch < dim && b < batch);
    float silu = 0.0f;

    if (valid) {
        const float* state_in = conv_state_in + (b * dim + ch) * d_conv;
        float* state = conv_state_out + (b * dim + ch) * d_conv;

        // 2026-10-06: The in-place kernel's shift, `state[i] = state[i + 1]` for i ascending,
        // reads each slot before it is overwritten, so it equals copying slot i + 1 of the
        // old window into slot i of the new one.
        for (unsigned int i = 0; i < d_conv - 1; i++)
            state[i] = state_in[i + 1];
        state[d_conv - 1] = (float)new_input[b * dim + ch];

        const __nv_bfloat16* w = weight + ch * d_conv;
        float acc = (bias != nullptr) ? bias[ch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++)
            acc += state[k] * (float)w[k];

        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        silu = acc * sigmoid_acc;
    }

    if (block_needs_l2) {
        float sq = valid ? (silu * silu) : 0.0f;

        const unsigned int warp_id = tid / 32;
        const unsigned int lane = tid % 32;
        for (int offset = 16; offset >= 1; offset >>= 1)
            sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);

        __shared__ float warp_sums[8];
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
    }

    if (valid) {
        output[b * dim + ch] = __float2bfloat16(silu);
    }
}

extern "C" __global__ void kda_recurrent_decode_bf16_smem_io(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
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
    // 2026-10-06: [VPB, D + 1] column scratch, padded as in kda_recurrent_decode_bf16_smem.
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
    const float* S_in = state_in + hd * D;
    float* S_out = state_out + hd * D;
    const unsigned int vi = v0 + threadIdx.x;
    if (threadIdx.x >= VPB || vi >= D) return;
    float* col = sh_s + (size_t)threadIdx.x * col_stride;

    float kv = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = S_in[(size_t)kk * D + vi] * sh_decay[kk];
        col[kk] = s;
        kv += s * sh_k[kk];
    }
    const float delta = (__bfloat162float(v[hd + vi]) - kv) * b;
    float o = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = col[kk] + sh_k[kk] * delta;
        S_out[(size_t)kk * D + vi] = s;
        o += s * sh_q[kk];
    }
    out[hd + vi] = o;
}
