// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-01: GLM-5.3-Flash decode L2 weight prefetch (module `glm5next_l2_prefetch`).
// Owner: gb10 kernels (glm-5.3-flash).
// Invariants:
// - glm5next_l2_prefetch writes no memory. It only issues `cp.async.bulk.prefetch.L2.global`
//   hints for up to four read-only byte spans, so every value the model computes is the same
//   with or without it (byte-identical by construction). A hint may be dropped by the
//   hardware; that costs speed, never correctness.
// - Each span starts 16-byte aligned and its length is a multiple of 16 (the host rounds the
//   length DOWN, glm5next_layer/prefetch.rs `clip_spans`), so every bulk prefetch stays inside
//   the allocation the span names: the host builds spans only from the exact `n * k * elem`
//   extent the consuming GEMV reads.
// - glm5next_l2pf_spin_ns is a microtest helper (a stand-in for a latency-bound window such as
//   an all-reduce); the model never launches it.
//
// Why: in single-stream decode the DRAM sits nearly idle during the latency-bound chain between
// two weight-streaming GEMVs (all-reduce, hc_post, hc_mix, hc_finish, rms_norm, router top-k).
// The next GEMV's weights are static, so their first bytes can be pulled into the 24 MB L2
// during that chain. This is the inter-instruction weight pipelining of persistent decode
// megakernels (Hazy Research "Look Ma, No Bubbles", 2025-05-27; Mirage Persistent Kernel,
// arXiv 2512.22219), done here without a persistent kernel: the prefetch is one small kernel
// enqueued before the collective. No code from either project is used.
//
// Launch contract: blockDim.x == GLM_L2PF_BLOCK (128); any grid (the host uses
// ceil(chunks / 128) blocks, at most GLM_L2PF_MAX_BLOCKS). Unused span slots pass n == 0.





#include <stdint.h>

#define GLM_L2PF_BLOCK 128u
#define GLM_L2PF_CHUNK 16384ull

__device__ __forceinline__ void glm_l2pf_bulk(const char* addr, unsigned int bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // 2026-10-01: TMA bulk prefetch into L2 (PTX ISA 8.0, sm_90+; GB10 is sm_121). No
    // completion mechanism and no shared-memory destination, so the issuing thread does not
    // wait for the bytes.
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(addr), "r"(bytes)
                 : "memory");
#else
    for (unsigned int o = 0; o < bytes; o += 128u) {
        asm volatile("prefetch.global.L2 [%0];\n" ::"l"(addr + o));
    }
#endif
}

// 2026-10-01: Thread c (grid-stride) prefetches chunk c of the concatenation of the four spans,
// GLM_L2PF_CHUNK bytes each (the last chunk of a span is shorter, still a multiple of 16).
extern "C" __global__ void glm5next_l2_prefetch(
    const char* __restrict__ p0, const unsigned long long n0,
    const char* __restrict__ p1, const unsigned long long n1,
    const char* __restrict__ p2, const unsigned long long n2,
    const char* __restrict__ p3, const unsigned long long n3
) {
    const char* p[4] = {p0, p1, p2, p3};
    const unsigned long long n[4] = {n0, n1, n2, n3};
    unsigned long long c0[5];
    c0[0] = 0;
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        c0[s + 1] = c0[s] + (n[s] + GLM_L2PF_CHUNK - 1) / GLM_L2PF_CHUNK;
    }
    const unsigned long long total = c0[4];
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long c = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         c < total; c += stride) {
        int s = 0;
        while (s < 3 && c >= c0[s + 1]) ++s;
        const unsigned long long off = (c - c0[s]) * GLM_L2PF_CHUNK;
        const unsigned long long left = n[s] - off;
        const unsigned int bytes =
            (unsigned int)(left < GLM_L2PF_CHUNK ? left : GLM_L2PF_CHUNK);
        if (bytes >= 16u) glm_l2pf_bulk(p[s] + off, bytes & ~15u);
    }
}

// 2026-10-01: Microtest helper: thread 0 of block 0 spins on %globaltimer for `ns`
// nanoseconds; every other thread exits. Grid (1, 1, 1), any blockDim.
extern "C" __global__ void glm5next_l2pf_spin_ns(const unsigned long long ns) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}
