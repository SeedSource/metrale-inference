// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-03: C ABI over FlashKDA's forward launcher (vendor/flashkda, MIT, unmodified) for one
// sequence with an FP32 recurrent state read and written in place.
//
// Owner: gpu-runtime (FlashKDA bridge).
// Invariants:
// - Only `launch_fwd<128, true, true, true, false>` is called: head dim 128, FP32 state in and
//   out, no cu_seqlens (N = 1). Upstream's PyTorch binding (csrc/flash_kda.cpp, not vendored)
//   picks the same instantiation for that call shape.
// - The workspace size is upstream's formula for the non-varlen case (`get_workspace_size` in
//   csrc/flash_kda.cpp counts one extra tile per sequence for varlen; non-varlen launches use
//   exactly ceil(rows / 16) tiles, which is what `total_tiles` is set to here).
// - Every pointer is a device pointer; the call validates sizes but not residency.
//
// Layouts (all row-major, contiguous):
//   q, k, v, g, out   BF16 [rows, heads, 128]      (g is the gate BEFORE activation)
//   beta_t            BF16 [heads, rows]           (beta logits, sigmoid applied inside)
//   state             FP32 [heads, 128(v), 128(k)] (read at the start, written at the end)
//   a_log             FP32 [heads]
//   dt_bias           FP32 [heads, 128]

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "fwd.h"

namespace {
constexpr int kChunk = 16;
constexpr int kD = 128;
// 2026-10-03: Upstream per-(tile, head) workspace record: three CHUNK x D BF16 tiles (k decayed,
// q decayed, k restored), a D-long FP32 total gate, and two CHUNK x CHUNK BF16 matrices (the
// inverse and the masked q.k^T).
constexpr int64_t kPerTileBytes = 3 * kChunk * kD * 2 + kD * 4 + 2 * kChunk * kChunk * 2;

int64_t tiles_for(int64_t rows) { return (rows + kChunk - 1) / kChunk; }
}  // namespace

// 2026-10-03: Device bytes `metrale_flashkda_fwd_fp32_state` needs for `rows` rows of one
// sequence with `heads` heads: the per-tile records plus the 128-byte tile-prefix trailer
// (N + 1 = 2 int32, rounded up to 128 B). Returns 0 for a non-positive argument.
extern "C" int64_t metrale_flashkda_workspace_bytes(int64_t rows, int64_t heads) {
    if (rows <= 0 || heads <= 0) return 0;
    const int64_t tile_prefix_bytes = ((1 + 1) * 4 + 127) / 128 * 128;
    return heads * tiles_for(rows) * kPerTileBytes + tile_prefix_bytes;
}

// 2026-10-03: One FlashKDA forward over `rows` rows of one sequence. `state` is both the initial
// and the final state (each (head) CTA loads its slice before its first chunk and stores it
// after its last; no other CTA touches that slice). `scale` multiplies q (upstream rounds it to
// BF16 inside); `lower_bound` is the KDA gate lower bound (upstream documents -5..0).
// Returns 0, or a cudaError_t (cudaErrorInvalidValue for a bad argument or a small workspace).
extern "C" int metrale_flashkda_fwd_fp32_state(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta_t,
    void* state,
    float scale,
    void* out,
    void* workspace,
    int64_t workspace_bytes,
    int rows,
    int heads,
    const float* a_log,
    const float* dt_bias,
    float lower_bound,
    void* stream
) {
    if (rows <= 0 || heads <= 0 || q == nullptr || k == nullptr || v == nullptr ||
        g == nullptr || beta_t == nullptr || state == nullptr || out == nullptr ||
        workspace == nullptr || a_log == nullptr || dt_bias == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (workspace_bytes < metrale_flashkda_workspace_bytes(rows, heads)) {
        return (int)cudaErrorInvalidValue;
    }
    using BF16 = cutlass::bfloat16_t;
    // 2026-10-03: Upstream's binding passes the gate bound pre-multiplied by log2(e): the kernels
    // work in exp2.
    const float gate_scale = (float)((double)lower_bound * 1.4426950408889634);
    launch_fwd<kD, true, true, true, false>(
        static_cast<BF16 const*>(q),
        static_cast<BF16 const*>(k),
        static_cast<BF16 const*>(v),
        static_cast<BF16 const*>(g),
        static_cast<BF16 const*>(beta_t),
        state,
        scale,
        state,
        static_cast<BF16*>(out),
        workspace,
        (int)tiles_for(rows),
        rows,
        heads,
        1,
        nullptr,
        a_log,
        dt_bias,
        gate_scale,
        static_cast<cudaStream_t>(stream));
    return (int)cudaGetLastError();
}
