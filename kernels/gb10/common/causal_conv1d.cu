// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Causal depthwise conv1d + SiLU kernels for SSM layers: a full-sequence prefill, decode updates of a
// per-channel FP32 window, and decode variants fused with a per-head L2 norm.
//
// The conv is SiLU(bias + sum_k x[t - (d_conv-1) + k] * weight[ch, k]), so weight[ch, 0] multiplies the oldest input
// and weight[ch, d_conv-1] the newest. Weights are BF16 [dim, d_conv]; bias is FP32 [dim] or null.
// A window state conv_state[.., ch, 0..d_conv) is FP32, oldest first.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.









#include <cuda_bf16.h>

// 2026-09-25: causal_conv1d_fwd: input and output are [batch, dim, seq_len]; inputs before position 0 count as zero.
// One block per (channel, batch), grid (dim, batch); its threads stride over the positions. At most 8 taps are used.





extern "C" __global__ void causal_conv1d_fwd(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch,
    unsigned int dim,
    unsigned int seq_len,
    unsigned int d_conv
) {
    const unsigned int ch = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (ch >= dim || b >= batch) return;


    const __nv_bfloat16* in_ptr = input + (b * dim + ch) * seq_len;
    __nv_bfloat16* out_ptr = output + (b * dim + ch) * seq_len;
    const __nv_bfloat16* w = weight + ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;


    float w_reg[8];
    for (unsigned int i = 0; i < d_conv && i < 8; i++) {
        w_reg[i] = (float)w[i];
    }


    for (unsigned int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float acc = b_val;





        for (unsigned int k = 0; k < d_conv && k < 8; k++) {
            int idx = (int)t - (int)(d_conv - 1) + (int)k;
            if (idx >= 0) {
                acc += (float)in_ptr[idx] * w_reg[k];
            }
        }


        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        float silu = acc * sigmoid_acc;

        out_ptr[t] = __float2bfloat16(silu);
    }
}

// 2026-09-25: causal_conv1d_update: one decode token per sequence. conv_state is [batch, dim, d_conv]; new_input and
// output are [batch, dim]. Each channel's window shifts left by one and takes the new value last. One thread per
// channel; the host launches grid (ceil(dim/256), batch) with 256 threads.














extern "C" __global__ void causal_conv1d_update(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    if (ch >= dim || b >= batch) return;


    float* state = conv_state + (b * dim + ch) * d_conv;


    for (unsigned int i = 0; i < d_conv - 1; i++) {
        state[i] = state[i + 1];
    }


    state[d_conv - 1] = (float)new_input[b * dim + ch];


    const __nv_bfloat16* w = weight + ch * d_conv;
    float acc = (bias != nullptr) ? bias[ch] : 0.0f;
    for (unsigned int k = 0; k < d_conv; k++) {
        acc += state[k] * (float)w[k];
    }


    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    float silu = acc * sigmoid_acc;

    output[b * dim + ch] = __float2bfloat16(silu);
}

// 2026-09-25: causal_conv1d_update with an FP32 output.




extern "C" __global__ void causal_conv1d_update_f32(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    if (ch >= dim || b >= batch) return;

    float* state = conv_state + (b * dim + ch) * d_conv;

    for (unsigned int i = 0; i < d_conv - 1; i++) {
        state[i] = state[i + 1];
    }
    state[d_conv - 1] = (float)new_input[b * dim + ch];

    const __nv_bfloat16* w = weight + ch * d_conv;
    float acc = (bias != nullptr) ? bias[ch] : 0.0f;
    for (unsigned int k = 0; k < d_conv; k++) {
        acc += state[k] * (float)w[k];
    }

    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    float silu = acc * sigmoid_acc;

    output[b * dim + ch] = silu;
}

// 2026-09-25: causal_conv1d_update_prefill: `seq_len` tokens of one sequence, one thread per channel walking the tokens
// in order with the window in registers; conv_state [dim, d_conv] is read at the start and written at the end. Token t
// is read from input[t * input_stride + ch] and written to output[t * output_stride + ch] (strides in BF16 elements).
// The window is s[0..3], so d_conv must be 4. With METRALE_CONV1D_TP=0 the host launches grid ceil(dim/256), block 256.







extern "C" __global__ void causal_conv1d_update_prefill(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int seq_len,
    unsigned int input_stride,
    unsigned int output_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;

    float* state = conv_state + ch * d_conv;
    const __nv_bfloat16* w = weight + ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;


    float w_reg[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        w_reg[k] = (float)w[k];
    }


    float s[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        s[k] = state[k];
    }


    for (unsigned int t = 0; t < seq_len; t++) {
        float new_val = (float)input[(unsigned long long)t * input_stride + ch];


        s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = new_val;


        float acc = b_val + s[0]*w_reg[0] + s[1]*w_reg[1] + s[2]*w_reg[2] + s[3]*w_reg[3];


        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        output[(unsigned long long)t * output_stride + ch] = __float2bfloat16(acc * sigmoid_acc);
    }


    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        state[k] = s[k];
    }
}

// 2026-09-25: causal_conv1d_update_chunk2: two decode tokens per sequence in one launch; new_input and output are
// [batch, 2, dim]. The window after the first token is also written to conv_state_intermediate [batch, dim, d_conv],
// and conv_state ends after the second. The window is four wide, so d_conv must be 4. Host grid (ceil(dim/256), batch),
// block 256.







extern "C" __global__ void causal_conv1d_update_chunk2(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ conv_state_intermediate,
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    if (ch >= dim || b >= batch) return;

    float* state = conv_state + (b * dim + ch) * d_conv;
    float* state_inter = conv_state_intermediate + (b * dim + ch) * d_conv;
    const __nv_bfloat16* w = weight + ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;


    float w_reg[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        w_reg[k] = (float)w[k];
    }


    float s[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        s[k] = state[k];
    }


    float in0 = (float)new_input[b * 2 * dim + ch];

    float s0_0 = s[1], s0_1 = s[2], s0_2 = s[3], s0_3 = in0;


    state_inter[0] = s0_0;
    state_inter[1] = s0_1;
    state_inter[2] = s0_2;
    state_inter[3] = s0_3;


    float acc0 = b_val + s0_0*w_reg[0] + s0_1*w_reg[1] + s0_2*w_reg[2] + s0_3*w_reg[3];
    float sig0 = 1.0f / (1.0f + __expf(-acc0));
    output[b * 2 * dim + ch] = __float2bfloat16(acc0 * sig0);


    float in1 = (float)new_input[(b * 2 + 1) * dim + ch];
    float s1_0 = s0_1, s1_1 = s0_2, s1_2 = s0_3, s1_3 = in1;


    state[0] = s1_0;
    state[1] = s1_1;
    state[2] = s1_2;
    state[3] = s1_3;


    float acc1 = b_val + s1_0*w_reg[0] + s1_1*w_reg[1] + s1_2*w_reg[2] + s1_3*w_reg[3];
    float sig1 = 1.0f / (1.0f + __expf(-acc1));
    output[(b * 2 + 1) * dim + ch] = __float2bfloat16(acc1 * sig1);
}

// 2026-09-25: causal_conv1d_update_l2norm: causal_conv1d_update, then, in blocks that start below `qk_channels`, an L2
// norm of each head's SiLU outputs: x * rsqrt(sum(x^2) + l2_eps). Channels from qk_channels on get SiLU only.
// The reduction assumes 256 threads and head_dim 128 (two heads per block, four warps each), and qk_channels must be a
// multiple of 256 so that no block holds both kinds of channel.












extern "C" __global__ void causal_conv1d_update_l2norm(
    float* __restrict__ conv_state,
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
        float* state = conv_state + (b * dim + ch) * d_conv;

        for (unsigned int i = 0; i < d_conv - 1; i++)
            state[i] = state[i + 1];
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

// 2026-10-01: causal_conv1d_update_l2norm_rows: `rows` consecutive tokens of ONE sequence in one
// launch, in token order, with the same results as `rows` causal_conv1d_update_l2norm launches at
// batch 1 (outputs and the final conv state). Grid (ceil(dim / 256), 1, 1), block 256; the same
// L2 contract (256 threads, head_dim 128, qk_channels a multiple of 256). d_conv <= 8.
//
// Per token the arithmetic is causal_conv1d_update_l2norm's in the same order: window shift,
// insert of the new BF16 input widened to FP32, bias (or 0) then the serial sum over k of
// window[k] * weight[k], SiLU with __expf, then the per-warp __shfl_down tree (offsets 16..1),
// the four warp partials added in warp order and rsqrtf(total + l2_eps). Thread ch keeps its
// window in registers across tokens and writes it back once at the end; the window is FP32 in
// both places, and the weights are widened once instead of per token, which gives the same
// floats, so no value changes. The gb10 tree builds with --fmad=false (common/KERNEL.toml).
// gdn_verify_fused_conv_kn (gdn_verify_fused_conv_kn.cu) is the same structure, checked byte
// for byte against the per-token kernel by gdn_conv_kn_microtest.
//
// A third __syncthreads after the L2 apply keeps each token's read of warp_sums[base_warp] ahead
// of the next token's lane-0 write to the same slot. Token t reads new_input at
// t * input_stride and writes output at t * output_stride (BF16 elements).
extern "C" __global__ void causal_conv1d_update_l2norm_rows(
    float* __restrict__ conv_state,
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

    float win[8];
    float wcoef[8];
    if (valid) {
        const float* state = conv_state + ch * d_conv;
        for (unsigned int i = 0; i < d_conv; i++) win[i] = state[i];
        const __nv_bfloat16* w = weight + ch * d_conv;
        for (unsigned int k = 0; k < d_conv; k++) wcoef[k] = (float)w[k];
    }

    __shared__ float warp_sums[8];

    for (unsigned int t = 0; t < rows; t++) {
        float silu = 0.0f;

        if (valid) {
            for (unsigned int i = 0; i < d_conv - 1; i++)
                win[i] = win[i + 1];
            win[d_conv - 1] = (float)new_input[(size_t)t * input_stride + ch];

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
            // 2026-10-01: Keeps the next token's lane-0 write to warp_sums behind this read.
            __syncthreads();
        }

        if (valid) {
            output[(size_t)t * output_stride + ch] = __float2bfloat16(silu);
        }
    }

    if (valid) {
        float* state = conv_state + ch * d_conv;
        for (unsigned int i = 0; i < d_conv; i++) state[i] = win[i];
    }
}

// 2026-09-25: causal_conv1d_update_l2norm with an FP32 output.



extern "C" __global__ void causal_conv1d_update_l2norm_f32(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
        float* state = conv_state + (b * dim + ch) * d_conv;
        for (unsigned int i = 0; i < d_conv - 1; i++)
            state[i] = state[i + 1];
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
        output[b * dim + ch] = silu;
    }
}

// 2026-09-25: causal_conv1d_update_l2norm_f32 with the per-sequence row strides of new_input (BF16 elements) and output
// (FP32 elements) passed in, instead of both being `dim`. The multi-sequence decode path reads QKVZ projection rows
// `qkvz_size` apart and writes `conv_dim`-strided rows (qwen3_ssm/trait_decode_multi_seq/ssm_batched_recurrent.rs).
// conv_state keeps the (b * dim + ch) * d_conv layout, so the sequences' state slots must be contiguous.


















extern "C" __global__ void causal_conv1d_update_l2norm_f32_strided(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);
    const bool valid = (ch < dim && b < batch);
    float silu = 0.0f;

    if (valid) {
        float* state = conv_state + (b * dim + ch) * d_conv;
        for (unsigned int i = 0; i < d_conv - 1; i++)
            state[i] = state[i + 1];
        state[d_conv - 1] = (float)new_input[b * input_stride + ch];
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
        output[b * output_stride + ch] = silu;
    }
}

// 2026-09-25: causal_conv1d_update_prefill_tp: causal_conv1d_update_prefill, parallel over tokens as well as channels.
// Output t depends only on inputs t-3..t (for t < 3 partly on the incoming conv_state), never on earlier outputs, so
// each thread computes 8 consecutive tokens of one channel with a rolling window: 11 input reads for 8 outputs. The host
// launches block (32, 8), so a warp spans 32 channels and the [t * stride + ch] loads coalesce, and grid
// (ceil(dim/32), ceil(seq_len/64)). The thread owning the last token writes the new conv_state. d_conv must be 4.

















extern "C" __global__ void __launch_bounds__(256, 4)
causal_conv1d_update_prefill_tp(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int seq_len,
    unsigned int input_stride,
    unsigned int output_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    const unsigned int t0 = (blockIdx.y * blockDim.y + threadIdx.y) * 8u;
    if (t0 >= seq_len) return;

    const float* state = conv_state + (unsigned long long)ch * d_conv;
    const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    float w_reg[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (unsigned int k = 0; k < 4; k++)
        if (k < d_conv) w_reg[k] = __bfloat162float(w[k]);

    // 2026-09-25: x(t) for t < 0 comes from the incoming conv_state (x(-1) is its last element), as in the serial
    // kernel; t >= seq_len reads as 0.
    auto xin = [&] (long long t) -> float {
        if (t >= 0) {
            return (t < (long long)seq_len)
                ? __bfloat162float(input[(unsigned long long)t * input_stride + ch])
                : 0.0f;
        }
        const long long idx = (long long)d_conv + t;
        return (idx >= 0) ? state[idx] : 0.0f;
    };



    float s0 = xin((long long)t0 - 3);
    float s1 = xin((long long)t0 - 2);
    float s2 = xin((long long)t0 - 1);
    #pragma unroll
    for (unsigned int i = 0; i < 8; i++) {
        const unsigned int t = t0 + i;
        if (t >= seq_len) break;
        const float s3 = xin((long long)t);
        const float acc = b_val + s0 * w_reg[0] + s1 * w_reg[1] + s2 * w_reg[2] + s3 * w_reg[3];
        const float sig = 1.0f / (1.0f + __expf(-acc));
        output[(unsigned long long)t * output_stride + ch] = __float2bfloat16(acc * sig);
        s0 = s1; s1 = s2; s2 = s3;
    }

    // 2026-09-25: The new conv_state is the last d_conv inputs; only the thread owning the last token writes it.

    if (t0 + 8u >= seq_len) {
        float* st = conv_state + (unsigned long long)ch * d_conv;
        #pragma unroll
        for (unsigned int k = 0; k < 4; k++)
            if (k < d_conv)
                st[k] = xin((long long)seq_len - (long long)d_conv + (long long)k);
    }
}
