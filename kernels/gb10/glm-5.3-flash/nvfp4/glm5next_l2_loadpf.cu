// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-11: GLM-5.3-Flash decode L2 load-prefetch (module `glm5next_l2_loadpf`), the kernel
// behind `METRALE_GLM_DECODE_L2_LOADPF` (host: glm5next_layer/loadpf.rs).
// Owner: gb10 kernels (glm-5.3-flash).
// Invariants:
// - glm5next_l2_loadpf writes no memory in practice. It REALLY LOADS one u32 from every
//   32-byte sector of up to four read-only spans (`ld.global.cg`: cached in L2, not in L1) and
//   sums them into a register. The only store is guarded by `sink != nullptr && acc ==
//   GLM_L2LPF_MAGIC`, and the host always passes `sink = nullptr`, so the store never runs;
//   the guard exists so the compiler cannot prove the loads dead (the loads are also `asm
//   volatile`). Every value the model computes is the same with or without it.
// - Each span starts at least 4-byte aligned and its length is a multiple of 16 (the host rounds
//   DOWN, prefetch.rs `clip_spans`), so a sector offset `i * 32 < n` always leaves >= 16 bytes,
//   and the 4-byte load at that offset stays inside the span.
//
// Why: on GB10 (24 MB L2) prefetch HINTS (`prefetch.global.L2`, `cp.async.bulk.prefetch`) do not
// reliably leave the lines resident; the hint-based METRALE_GLM_DECODE_L2_PREFETCH measured
// +3.47 ms/step slower. A real load does allocate the sector in L2. The host runs this kernel on
// a SIDE stream during a latency-bound window of the main stream (the KDA recurrence), with a
// small grid (METRALE_GLM_DECODE_L2_LOADPF_CTAS, default 8) so it co-runs with the latency-bound
// kernels instead of taking the GPU, and the main stream joins it before the consuming GEMV.
//
// Launch contract: blockDim.x == GLM_L2LPF_BLOCK (256); any grid. Unused span slots pass n == 0.

#include <stdint.h>

#define GLM_L2LPF_BLOCK 256u
#define GLM_L2LPF_SECTOR 32ull
// 2026-10-11: Independent loads in flight per thread per iteration (memory-level parallelism:
// 8 CTAs x 256 threads x 8 = 16K sectors = 512 KiB outstanding, above GB10's ~240 GB/s x ~1 us
// bandwidth-latency product).
#define GLM_L2LPF_UNROLL 8
#define GLM_L2LPF_MAGIC 0x7fffffffu

__device__ __forceinline__ unsigned int glm_l2lpf_ld(const char* addr) {
    unsigned int v;
    // 2026-10-11: Cache-global load: allocates in L2, bypasses L1, so it does not evict the
    // co-running kernels' L1 working sets.
    asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(addr));
    return v;
}

// 2026-10-11: Thread t (grid-stride, GLM_L2LPF_UNROLL sectors per step) loads sector c of the
// concatenation of the four spans' sectors.
extern "C" __global__ void glm5next_l2_loadpf(
    const char* __restrict__ p0, const unsigned long long n0,
    const char* __restrict__ p1, const unsigned long long n1,
    const char* __restrict__ p2, const unsigned long long n2,
    const char* __restrict__ p3, const unsigned long long n3,
    unsigned int* sink
) {
    const char* p[4] = {p0, p1, p2, p3};
    const unsigned long long n[4] = {n0, n1, n2, n3};
    unsigned long long c0[5];
    c0[0] = 0;
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        c0[s + 1] = c0[s] + (n[s] + GLM_L2LPF_SECTOR - 1) / GLM_L2LPF_SECTOR;
    }
    const unsigned long long total = c0[4];
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned int acc = 0;
    for (unsigned long long c = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         c < total; c += stride * GLM_L2LPF_UNROLL) {
        unsigned int v[GLM_L2LPF_UNROLL];
        // 2026-10-11: All loads first, then the sums, so the loads are in flight together.
        #pragma unroll
        for (int u = 0; u < GLM_L2LPF_UNROLL; ++u) {
            const unsigned long long cc = c + (unsigned long long)u * stride;
            v[u] = 0;
            if (cc < total) {
                int s = 0;
                while (s < 3 && cc >= c0[s + 1]) ++s;
                v[u] = glm_l2lpf_ld(p[s] + (cc - c0[s]) * GLM_L2LPF_SECTOR);
            }
        }
        #pragma unroll
        for (int u = 0; u < GLM_L2LPF_UNROLL; ++u) acc += v[u];
    }
    if (sink != nullptr && acc == GLM_L2LPF_MAGIC) *sink = acc;
}
