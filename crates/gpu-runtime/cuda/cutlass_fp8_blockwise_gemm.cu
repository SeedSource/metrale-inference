// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-06: CUTLASS Sm120 FP8 blockwise-scaled GEMM host wrapper (BF16 out), for the GLM-5.3
// dense prefill W8A8 GEMMs under METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW=1:
//   D[m, n] = bf16( sum_g ( sum_{k in g} A[m, k] * B[n, k] ) * SFA[m, g] * SFB[n / 128, g] ),  g = 128-K groups
// The kernel configuration follows CUTLASS v4.6.0 examples/87_blackwell_geforce_gemm_blockwise/
// 87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu (BSD-3-Clause, NVIDIA; see THIRD_PARTY_NOTICES.md):
// A e4m3 RowMajor [M, K], B e4m3 ColumnMajor (= a row-major [N, K] weight, K contiguous), C/D bf16
// RowMajor, FP32 accumulate, scale granularity (M, N, K) = (1, 128, 128), cooperative 128x128x128
// (KernelScheduleSm120Blockwise) and pingpong 64x128x128 (KernelTmaWarpSpecializedBlockwisePingpongSm120).
// Owner: gpu-runtime (CUTLASS reference objects).
// Invariants:
// - The CALLER's scale layouts are always K-major (row-major): a_scale [M, K/128] F32 (as
//   per_token_group_quant_fp8 writes it) and b_scale [N/128, K/128] F32 (as
//   quantize_bf16_to_fp8_blockscaled writes it).
// - METRALE_FP8BW_SF_K_MAJOR selects the majorness CUTLASS is instantiated with (both SFA and SFB
//   the same: smem_atom_layoutSFB keys on majorSFA in blockwise_scale_layout.hpp). 1 (default):
//   K-major, the caller's buffers are passed through. 0: MN-major (87b's default, the only one 87b
//   exercises); the wrapper then transposes both scale buffers into the workspace before the GEMM
//   (an extra M * K/128 + N/128 * K/128 F32 of workspace and two small launches per call).
// - Every extern "C" entry returns 0 on success; -1 bad shape (m <= 0, n or k not a positive
//   multiple of 128); -2 workspace too small; -3 unknown schedule; -4 a misaligned pointer
//   (A, B, D need 16 bytes, the scales 4); -120 built without SM120/SM121 MMA support; otherwise
//   a nonzero cutlass::Status or a negated CUDA error.
// - beta is 0: D is passed as C too and never read.

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/float8.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

// 2026-10-06: The one-line fallback switch (file header). K-major is untested by 87b; MN-major is
// what 87b runs.
#ifndef METRALE_FP8BW_SF_K_MAJOR
#define METRALE_FP8BW_SF_K_MAJOR 1
#endif

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace metrale_fp8bw {

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD = ElementC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute = float;

using CooperativeTile = Shape<_128, _128, _128>;
using PingpongTile = Shape<_64, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;
constexpr cute::UMMA::Major kMajorSF =
    METRALE_FP8BW_SF_K_MAJOR ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
    ScaleGranularityM, ScaleGranularityN, ScaleGranularityK, kMajorSF, kMajorSF>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

template <class TileShape>
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutC, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

template <class TileShape, class Schedule>
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue<TileShape>::SharedStorage))>,
    Schedule>::CollectiveOp;

template <class TileShape, class Schedule>
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop<TileShape, Schedule>,
    CollectiveEpilogue<TileShape>,
    void>;

using CooperativeGemm = cutlass::gemm::device::GemmUniversalAdapter<
    GemmKernel<CooperativeTile, cutlass::gemm::KernelScheduleSm120Blockwise>>;
using PingpongGemm = cutlass::gemm::device::GemmUniversalAdapter<
    GemmKernel<PingpongTile, cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>>;

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

#if !METRALE_FP8BW_SF_K_MAJOR
// 2026-10-06: Row-major [rows, cols] F32 -> row-major [cols, rows] (the MN-major scale layouts:
// SFA[g * M + m], SFB[g * (N/128) + nb]).
__global__ void transpose_f32(const float* __restrict__ src, float* __restrict__ dst, int rows, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols) {
    dst[(size_t)c * rows + r] = src[(size_t)r * cols + c];
  }
}

inline cudaError_t launch_transpose(const float* src, float* dst, int rows, int cols, cudaStream_t s) {
  dim3 block(32, 8);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  transpose_f32<<<grid, block, 0, s>>>(src, dst, rows, cols);
  return cudaGetLastError();
}
#endif

template <class Gemm>
int run(const void* a_fp8, const float* a_scale, const void* b_fp8, const float* b_scale, void* out_bf16,
        int m, int n, int k, void* workspace, size_t workspace_size, cudaStream_t stream) {
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  const float* sfa = a_scale;
  const float* sfb = b_scale;
  size_t scratch = 0;
#if !METRALE_FP8BW_SF_K_MAJOR
  const int kg = k / ScaleGranularityK;
  const int nb = n / ScaleGranularityN;
  size_t sfa_bytes = align_up(static_cast<size_t>(m) * kg * sizeof(float), 256);
  size_t sfb_bytes = align_up(static_cast<size_t>(nb) * kg * sizeof(float), 256);
  scratch = sfa_bytes + sfb_bytes;
  float* sfa_mn = static_cast<float*>(workspace);
  float* sfb_mn = reinterpret_cast<float*>(static_cast<unsigned char*>(workspace) + sfa_bytes);
  sfa = sfa_mn;
  sfb = sfb_mn;
#endif

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<ElementA const*>(a_fp8), stride_a,
       reinterpret_cast<ElementB const*>(b_fp8), stride_b,
       sfa, layout_sfa,
       sfb, layout_sfb},
      {{},
       reinterpret_cast<ElementC const*>(out_bf16), stride_c,
       reinterpret_cast<ElementD*>(out_bf16), stride_d}};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;

  size_t gemm_ws = Gemm::get_workspace_size(args);
  if (scratch + gemm_ws > workspace_size) {
    return -2;
  }

#if !METRALE_FP8BW_SF_K_MAJOR
  cudaError_t err = launch_transpose(a_scale, sfa_mn, m, kg, stream);
  if (err != cudaSuccess) {
    return -static_cast<int>(err);
  }
  err = launch_transpose(b_scale, sfb_mn, nb, kg, stream);
  if (err != cudaSuccess) {
    return -static_cast<int>(err);
  }
#endif

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  status = gemm.initialize(args, static_cast<unsigned char*>(workspace) + scratch, stream);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  status = gemm.run(stream);
  return static_cast<int>(status);
}

}  // namespace metrale_fp8bw

#endif

// 2026-10-06: D[m, n] (BF16, row stride n) = blockwise W8A8 GEMM of A [m, k] e4m3 (a_scale
// [m, k/128] F32) and the row-major [n, k] e4m3 weight B (b_scale [n/128, k/128] F32). schedule
// 0 = cooperative 128x128x128, 1 = pingpong 64x128x128. One launch; M chunking is the caller's.
extern "C" int metrale_cutlass_fp8_blockwise_gemm_bf16(
    const void* a_fp8,
    const void* a_scale,
    const void* b_fp8,
    const void* b_scale,
    void* out_bf16,
    int m,
    int n,
    int k,
    int schedule,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (m <= 0 || n <= 0 || k <= 0 || (n % 128) != 0 || (k % 128) != 0) {
    return -1;
  }
  auto misaligned = [](const void* p, uintptr_t a) {
    return (reinterpret_cast<uintptr_t>(p) % a) != 0;
  };
  if (misaligned(a_fp8, 16) || misaligned(b_fp8, 16) || misaligned(out_bf16, 16) ||
      misaligned(a_scale, 4) || misaligned(b_scale, 4) || misaligned(workspace, 256)) {
    return -4;
  }
  const float* sa = static_cast<const float*>(a_scale);
  const float* sb = static_cast<const float*>(b_scale);
  switch (schedule) {
    case 0:
      return metrale_fp8bw::run<metrale_fp8bw::CooperativeGemm>(
          a_fp8, sa, b_fp8, sb, out_bf16, m, n, k, workspace, workspace_size, stream);
    case 1:
      return metrale_fp8bw::run<metrale_fp8bw::PingpongGemm>(
          a_fp8, sa, b_fp8, sb, out_bf16, m, n, k, workspace, workspace_size, stream);
    default:
      return -3;
  }
#else
  (void)a_fp8;
  (void)a_scale;
  (void)b_fp8;
  (void)b_scale;
  (void)out_bf16;
  (void)m;
  (void)n;
  (void)k;
  (void)schedule;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  return -120;
#endif
}

// 2026-10-06: 1 when this object was built with K-major scale layouts (METRALE_FP8BW_SF_K_MAJOR),
// 0 when MN-major (the wrapper transposes the caller's K-major scales into the workspace).
extern "C" int metrale_cutlass_fp8_blockwise_scale_k_major() {
  return METRALE_FP8BW_SF_K_MAJOR ? 1 : 0;
}
