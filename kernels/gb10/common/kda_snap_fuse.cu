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

// 2026-10-09: Row 0 of a verify walk whose INPUT state is chosen on the device, for the deferred
// commit (METRALE_GLM_KDA_DEFER_COMMIT=1, crates/model-arch/src/glm5next_kda/snap_fuse/defer.rs).
//
// A partial-accept commit used to copy verify intermediate `a - 1` into the live state (68 MiB
// + 6.4 MiB per step on GLM-5.3 TP2). Under the lever it leaves `a - 1` in a per-slot device
// word instead, and the next verify's row 0 reads its input from there:
// - *defer_idx < 0: the input is the live state, and the kernel is exactly the `_io` twin above
//   (the same expressions in the same order on the same floats).
// - *defer_idx = i >= 0: the input is intermediate i of this slot, `*_inter0 + i * stride`,
//   which holds the bytes the dropped copy would have written into the live state.
// Either way the output is written to `*_out` (snapshot slot 0), as the `_io` twin's row 0 does.
// The index is read from device memory, not passed as a host value or a host-built pointer, so
// a verify captured into a CUDA graph stays correct on every replay.
//
// Hazards (why these kernels may alias their input and output; no `__restrict__` on that pair):
// - With i == 0 the input and the output are the same buffer (intermediate 0 is snapshot 0).
// - Conv: each thread owns one channel's d_conv-float window (`state_in` / `state` below, offset
//   (b * dim + ch) * d_conv) and no other thread reads or writes it. The shift reads slot i + 1
//   before writing slot i, for i ascending, so in place it is the in-place kernel's own shift
//   (`state[i] = state[i + 1]`, causal_conv1d.cu); slot i + 1 is overwritten only on the next
//   iteration, after it was read. The convolution then reads the new window back from `state`,
//   as the `_io` twin does. Without `__restrict__` the compiler must keep that load/store order.
// - Recurrence: thread (h, vi) is the only thread that touches column vi of head h (blocks split
//   heads and VPB-column groups, threads split columns). Its first loop reads the whole column
//   from `S_in` into shared memory before its second loop writes any of it to `S_out`, and every
//   store depends on `delta`, which depends on every load; so in place the column is fully read
//   before it is overwritten, as in kda_recurrent_decode_bf16_smem.
// - i >= 1: input and output are different intermediates; rows >= 1 then read the output, so the
//   source slot is consumed by row 0 before row (i) overwrites it (stream order, kernel boundary).
// Same build flags as the file (--fmad=false): bit-identical to the copy followed by row 0.

extern "C" __global__ void causal_conv1d_update_l2norm_defer(
    const float* conv_state_live,
    const float* conv_inter0,
    unsigned long long conv_inter_stride,
    const int* __restrict__ defer_idx,
    float* conv_state_out,
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
    const int di = *defer_idx;
    const float* conv_state_in =
        (di >= 0) ? conv_inter0 + (size_t)di * conv_inter_stride : conv_state_live;

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

extern "C" __global__ void kda_recurrent_decode_bf16_smem_defer(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const float* state_live,
    const float* state_inter0,
    unsigned long long state_inter_stride,
    const int* __restrict__ defer_idx,
    float* state_out,
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
    const float* state_in =
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
