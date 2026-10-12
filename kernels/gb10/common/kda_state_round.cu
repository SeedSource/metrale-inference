// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-12: METRALE_GLM_KDA_STATE_ROUND=bf16|fp16 (quality-test lever, default off): round a KDA
// recurrent state in place to BF16 or FP16 precision and back to FP32, track the range of the
// values it rounded, and log that range from the device.
//
// Owner: gb10 kernels.
// Invariants:
// - state is FP32, `n` elements ([heads, head_dim, head_dim] of one sequence's layer state);
//   every element is overwritten with the round-to-nearest-even value of its own old value
//   (__float2bfloat16_rn / __float2half_rn, then back to FP32). Nothing else is written. The
//   production recurrent kernels are untouched: the host launches this kernel on the same
//   stream right after the kernel that wrote the state, so it is captured in a CUDA graph
//   like its neighbour.
// - mode 0 = BF16, 1 = FP16. Any n; grid-stride. Launch: block (256, 1, 1), any grid.x >= 1.
//   Every thread of every warp reaches the shuffles (no early return before them).
// - slot is 16 u32 of device memory, zeroed at allocation and owned by one layer (launches on
//   one slot are stream-ordered, so the slot is never used by two launches at once):
//     [0] decode max |v| over the FINITE pre-rounding values, as FP32 bits (a non-negative float
//         orders like its bits, so atomicMax on the bits is a float max);
//     [1] decode count of finite values whose rounded value is infinite (FP16: |v| at or above
//         65520; BF16 practically never);
//     [2] decode count of non-finite pre-rounding values (NaN or inf; they stay as they are);
//     [3] decode launches since the last decode log line;
//     [4]..[6] the same three statistics for prefill launches;
//     [7] prefill launches since the last prefill log line;
//     [8] block-completion ticket (reset by the last block of every launch);
//     [9]..[15] unused.
// - phase 0 = decode, 1 = prefill. log_every = 0 never prints and never resets the statistics
//   (the host microtest reads them from the slot). Otherwise the LAST block of each launch (the
//   one whose ticket is gridDim.x - 1, after __threadfence) does:
//     phase 1: counts the launch only;
//     phase 0: if the slot holds prefill statistics, prints them as one line with phase=prefill
//       (step = prefill launches) and clears them (so a prefill is reported once, at the first
//       decode row after it); then counts the launch and, every log_every-th decode launch,
//       prints the decode statistics (phase=decode, step = decode launches) and clears them.
//   Printing is device printf, so it needs no host readback and works inside a CUDA graph; the
//   text reaches the process stdout when the host next synchronizes.
//   Line: KDA_STATE_ROUND fmt=<bf16|fp16> step=<n> absmax=<v> overflow=<count> phase=<..>
//         layer=<idx> nonfinite=<count>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define KSR_BLOCK 256
#define KSR_DEC 0
#define KSR_PF 4
#define KSR_TICKET 8

__device__ __forceinline__ unsigned int ksr_read(unsigned int* p) { return atomicAdd(p, 0u); }

__device__ void ksr_report(unsigned int* s, unsigned int mode, unsigned int layer, unsigned int pf) {
    const unsigned int base = pf ? KSR_PF : KSR_DEC;
    const float absmax = __uint_as_float(ksr_read(&s[base]));
    const unsigned int ovf = ksr_read(&s[base + 1]);
    const unsigned int nonfinite = ksr_read(&s[base + 2]);
    const unsigned int step = ksr_read(&s[base + 3]);
    printf("KDA_STATE_ROUND fmt=%s step=%u absmax=%.6g overflow=%u phase=%s layer=%u nonfinite=%u\n",
           mode == 0u ? "bf16" : "fp16", step, (double)absmax, ovf, pf ? "prefill" : "decode",
           layer, nonfinite);
    atomicExch(&s[base], 0u);
    atomicExch(&s[base + 1], 0u);
    atomicExch(&s[base + 2], 0u);
    atomicExch(&s[base + 3], 0u);
}

extern "C" __global__ void __launch_bounds__(KSR_BLOCK) kda_state_round(
    float* __restrict__ state,
    unsigned long long n,
    unsigned int mode,
    unsigned int* slot,
    unsigned int phase,
    unsigned int log_every,
    unsigned int layer
) {
    float amax = 0.0f;
    unsigned int ovf = 0u;
    unsigned int nonfinite = 0u;
    const unsigned long long stride = (unsigned long long)gridDim.x * KSR_BLOCK;
    for (unsigned long long i = (unsigned long long)blockIdx.x * KSR_BLOCK + threadIdx.x; i < n;
         i += stride) {
        const float v = state[i];
        const float r = (mode == 0u) ? __bfloat162float(__float2bfloat16_rn(v))
                                     : __half2float(__float2half_rn(v));
        if (isfinite(v)) {
            amax = fmaxf(amax, fabsf(v));
            if (!isfinite(r)) ovf += 1u;
        } else {
            nonfinite += 1u;
        }
        state[i] = r;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xFFFFFFFFu, amax, off));
        ovf += __shfl_down_sync(0xFFFFFFFFu, ovf, off);
        nonfinite += __shfl_down_sync(0xFFFFFFFFu, nonfinite, off);
    }
    const unsigned int base = phase ? KSR_PF : KSR_DEC;
    if ((threadIdx.x & 31u) == 0u) {
        atomicMax(&slot[base], __float_as_uint(amax));
        if (ovf != 0u) atomicAdd(&slot[base + 1], ovf);
        if (nonfinite != 0u) atomicAdd(&slot[base + 2], nonfinite);
    }
    if (log_every == 0u) return;

    __shared__ unsigned int is_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        is_last = (atomicAdd(&slot[KSR_TICKET], 1u) == gridDim.x - 1u) ? 1u : 0u;
    }
    __syncthreads();
    if (is_last == 0u || threadIdx.x != 0) return;
    __threadfence();
    if (phase != 0u) {
        atomicAdd(&slot[KSR_PF + 3], 1u);
    } else {
        if (ksr_read(&slot[KSR_PF + 3]) != 0u) ksr_report(slot, mode, layer, 1u);
        const unsigned int calls = atomicAdd(&slot[KSR_DEC + 3], 1u) + 1u;
        if (calls % log_every == 0u) ksr_report(slot, mode, layer, 0u);
    }
    atomicExch(&slot[KSR_TICKET], 0u);
}
