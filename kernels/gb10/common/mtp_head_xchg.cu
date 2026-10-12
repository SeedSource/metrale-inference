// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-12: The GLM-5.3 MTP draft head's cross-rank pick packet, written on
// the device (`METRALE_GLM_MTP_XCHG_DEVICE=1`, model-arch
// glm5next_mtp_head/xchg_dev.rs). The host packer it replaces is
// `forward_one` in glm5next_mtp_head.rs; this kernel writes the same 16 bytes
// per row, so the all-reduce that follows it and the host decode that reads
// its result are unchanged.
//
// Owner: gb10 kernels.
// Invariants:
// - Row i (one thread) writes 8 BF16 lanes at xchg[8 i .. 8 i + 8]: lane
//   `rank` is the raw BF16 logit at the row's local argmax, lanes
//   2 + 3 rank + d (d = 0..2) are base-256 digit d of the global index
//   v0 + arg[i], each a small integer as BF16 (the top 16 bits of its f32,
//   exact for 0..255, as the host's `bf` does); every other lane is zero.
// - Reads arg[i] (u32, local index < nv) and logits[i * nv + arg[i]] only.
// - rank is 0 or 1 (8 lanes hold two ranks).

#include <stdint.h>

extern "C" __global__ void mtp_head_xchg_pack(
    const uint32_t* __restrict__ arg,
    const uint16_t* __restrict__ logits,
    uint16_t* __restrict__ xchg,
    uint32_t nv,
    uint32_t v0,
    uint32_t rank,
    uint32_t n_rows)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) {
        return;
    }
    const uint32_t l = arg[i];
    uint16_t* row = xchg + 8u * i;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        row[j] = 0;
    }
    row[rank] = logits[(size_t)i * nv + l];
    const uint32_t g = v0 + l;
#pragma unroll
    for (uint32_t d = 0; d < 3; ++d) {
        const float digit = (float)((g >> (8u * d)) & 0xFFu);
        row[2u + rank * 3u + d] = (uint16_t)(__float_as_uint(digit) >> 16);
    }
}
