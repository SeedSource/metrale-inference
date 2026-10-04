// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: CUTLASS Sm120 NVFP4 MoE GEMMs as one GemmUniversalMode::kGrouped launch per
// projection over all active experts, plus the load-time SFB swizzle.
// Owner: gpu-runtime (CUTLASS reference objects).
// Invariants:
// - Built with CUTLASS_ARCH_MMA_SM120/SM121 support, every extern "C" entry returns -1 for
//   n <= 0, k <= 0 or k % 16 != 0 (the GEMMs also for num_experts <= 0); built without
//   it, -120. Any other nonzero return is a failure; the cudaMemcpyAsync uploads and the
//   batched A-pack launch are not checked, so their failures are not reported.
// - A group's B and SFB pointers are checked for null before its GEMM launches (-140).
// - 2026-10-03: The `_w4a4` entry points (global activation scale, GLM-5.3 prefill lever
//   METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4) and metrale_cutlass_pack_weight_sfb_batched are
//   additions; the pre-existing entry points launch exactly what they launched before (the
//   new prep_grouped_a / launch_projection parameters default to the old behaviour).
// - CUTLASS (BSD-3-Clause, NVIDIA) headers only; see THIRD_PARTY_NOTICES.md.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

// 2026-09-25: Types, layouts, alignments and tile shape as in the dense GEMM (cutlass_nvfp4_gemm.cu).
using ElementInput = cutlass::float_e2m1_t;
using ElementA = cutlass::nv_float4_t<ElementInput>;
using ElementB = cutlass::nv_float4_t<ElementInput>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementAccumulator = float;
using ElementCompute = float;


using GmemLayoutA = cutlass::layout::RowMajor;
using GmemLayoutB = cutlass::layout::ColumnMajor;
using GmemLayoutC = cutlass::layout::RowMajor;
using GmemLayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

// 2026-09-25: Epilogue D = alpha[g] * acc, beta 0; alpha[g] is group g's scale2.
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutC*,
    AlignmentC,
    ElementD,
    GmemLayoutD*,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;


using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    GmemLayoutA*,
    AlignmentA,
    ElementB,
    GmemLayoutB*,
    AlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,


    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>,
    CollectiveMainloop,
    CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;


using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ProblemShape = cute::Shape<int, int, int>;

static inline size_t align_up_(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}

// 2026-09-25: The rounding of float_to_e2m1 in cutlass_nvfp4_gemm.cu.
__device__ __forceinline__ unsigned char float_to_e2m1_g(float x) {
  unsigned char sign = (x < 0.0f) ? 8u : 0u;
  float ax = fabsf(x);
  unsigned char mag;
  if (ax <= 0.25f) {
    mag = 0;
  } else if (ax <= 0.75f) {
    mag = 1;
  } else if (ax <= 1.25f) {
    mag = 2;
  } else if (ax <= 1.75f) {
    mag = 3;
  } else if (ax <= 2.5f) {
    mag = 4;
  } else if (ax <= 3.5f) {
    mag = 5;
  } else if (ax <= 5.0f) {
    mag = 6;
  } else {
    mag = 7;
  }
  return sign | mag;
}

// 2026-09-25: metrale_cutlass_pack_bf16_act_nvfp4 (cutlass_nvfp4_gemm.cu) for one group with a
// gather: local row r reads token sorted_token_ids[ms + r] of token-major act_global [*, k] (token
// ms + r when the ids are null). Not launched here; prep_grouped_a uses pack_act_grouped_batched.
template <class LayoutSFA_t>
__global__ void pack_act_group(
    const __nv_bfloat16* __restrict__ act_global,
    const int* __restrict__ sorted_token_ids,
    int ms,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ scales,
    int m,
    int k,
    LayoutSFA_t layout_sfa) {
  int row = blockIdx.x;
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  int groups = k / 16;
  if (row >= m || group >= groups) {
    return;
  }


  int gid = ms + row;
  int tok = sorted_token_ids ? sorted_token_ids[gid] : gid;
  const __nv_bfloat16* arow = act_global + (unsigned long long)tok * k;
  int base = group * 16;
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float v = __bfloat162float(arow[base + i]);
    max_abs = fmaxf(max_abs, fabsf(v));
  }
  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  cutlass::float_ue4m3_t sf(scale);
  scales[layout_sfa(row, base, 0)] = *reinterpret_cast<unsigned char*>(&sf);
  float dec = static_cast<float>(sf);
  float inv = dec > 0.0f ? 1.0f / dec : 0.0f;
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    float v0 = __bfloat162float(arow[base + i]) * inv;
    float v1 = __bfloat162float(arow[base + i + 1]) * inv;
    packed[(unsigned long long)row * (k / 2) + base / 2 + i / 2] =
        static_cast<unsigned char>(float_to_e2m1_g(v0) | (float_to_e2m1_g(v1) << 4));
  }
}









// 2026-09-25: pack_act_group for every group in one launch: blockIdx.z is the group,
// and ms_arr / m_arr / packed_arr / scales_arr ([G], device) carry each group's first
// sorted row, row count, packed-A region and SFA region. grid.x is max(m_e); blocks
// past a group's m_e return. layout_sfa_dummy is unused.
template <class LayoutSFA_t>
__global__ void pack_act_grouped_batched(
    const __nv_bfloat16* __restrict__ act_global,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ ms_arr,
    const int* __restrict__ m_arr,
    unsigned char* const* __restrict__ packed_arr,
    unsigned char* const* __restrict__ scales_arr,
    int k,
    LayoutSFA_t layout_sfa_dummy) {
  const int e = blockIdx.z;
  const int m_e = m_arr[e];
  int row = blockIdx.x;
  if (row >= m_e) {
    return;
  }
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  const int groups = k / 16;
  if (group >= groups) {
    return;
  }

  // 2026-09-25: The SFA layout depends on the group's m_e, so it is built per group.
  auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m_e, 1, k, 1));
  (void)layout_sfa_dummy;

  unsigned char* packed = packed_arr[e];
  unsigned char* scales = scales_arr[e];
  const int ms = ms_arr[e];

  int gid = ms + row;
  int tok = sorted_token_ids ? sorted_token_ids[gid] : gid;
  const __nv_bfloat16* arow = act_global + (unsigned long long)tok * k;
  int base = group * 16;
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float v = __bfloat162float(arow[base + i]);
    max_abs = fmaxf(max_abs, fabsf(v));
  }
  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  cutlass::float_ue4m3_t sf(scale);
  scales[layout_sfa(row, base, 0)] = *reinterpret_cast<unsigned char*>(&sf);
  float dec = static_cast<float>(sf);
  float inv = dec > 0.0f ? 1.0f / dec : 0.0f;
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    float v0 = __bfloat162float(arow[base + i]) * inv;
    float v1 = __bfloat162float(arow[base + i + 1]) * inv;
    packed[(unsigned long long)row * (k / 2) + base / 2 + i / 2] =
        static_cast<unsigned char>(float_to_e2m1_g(v0) | (float_to_e2m1_g(v1) << 4));
  }
}


// 2026-09-25: Swizzle one expert's E4M3 weight scales into the SFB layout. The source is
// [K/16, N] when src_n_major is 0 and [N, K/16] otherwise; the output does not depend
// on it.
template <class LayoutSFB_t>
__global__ void pack_weight_sfb_group(
    const unsigned char* __restrict__ metrale_scales,
    unsigned char* __restrict__ cutlass_scales,
    int n,
    int k,
    int src_n_major,
    LayoutSFB_t layout_sfb) {
  int col = blockIdx.x;
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  int groups = k / 16;
  if (col >= n || group >= groups) {
    return;
  }



  unsigned char metrale_scale =
      src_n_major ? metrale_scales[(unsigned long long)col * groups + group]
                  : metrale_scales[(unsigned long long)group * n + col];
  __nv_fp8_e4m3 in;
  *reinterpret_cast<unsigned char*>(&in) = metrale_scale;
  float scale = static_cast<float>(in);
  cutlass::float_ue4m3_t sf(scale);
  cutlass_scales[layout_sfb(col, group * 16, 0)] = *reinterpret_cast<unsigned char*>(&sf);
}

// 2026-10-03: W4A4 with a global activation scale (the `_w4a4` entry points below; GLM-5.3
// routed-MoE prefill, METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1). The kernels in this block are
// launched only by those entry points and metrale_cutlass_pack_weight_sfb_batched; the kernels
// above and the pre-existing entry points are unchanged.

// 2026-10-03: Round-to-nearest-even E2M1 code, saturating at 6: sign bit 8, magnitude code
// 0..7 = {0, 0.5, 1, 1.5, 2, 3, 4, 6}. This is the rounding of PTX `cvt.rn.satfinite.e2m1x2.f32`
// (ties to the even code: 0.75 -> 1.0, 1.75 -> 2.0, 3.5 -> 4.0), where float_to_e2m1_g above
// rounds those three ties down. A NaN input gives magnitude code 7.
__device__ __forceinline__ unsigned char float_to_e2m1_rne(float x) {
  unsigned char sign = (x < 0.0f) ? 8u : 0u;
  float ax = fabsf(x);
  unsigned char mag;
  if (ax <= 0.25f) {
    mag = 0;
  } else if (ax < 0.75f) {
    mag = 1;
  } else if (ax <= 1.25f) {
    mag = 2;
  } else if (ax < 1.75f) {
    mag = 3;
  } else if (ax <= 2.5f) {
    mag = 4;
  } else if (ax < 3.5f) {
    mag = 5;
  } else if (ax <= 5.0f) {
    mag = 6;
  } else {
    mag = 7;
  }
  return sign | mag;
}

// 2026-10-03: pack_act_grouped_batched with an NVFP4 global scale per group, gs_arr[g] > 0
// (the checkpoint's `input_scale`, i.e. calibrated amax / (6 * 448), or the dynamic value
// resolve_act_gs wrote). Per 16 values: block scale sf = UE4M3(min(amax / 6 / gs, 448)), codes
// E2M1_rne(v / (sf * gs)); the GEMM epilogue multiplies by alpha = weight_scale_2 * gs. The
// two-level recipe of the ModelOpt/TensorRT NVFP4 activation quantizer (vLLM passes
// 1 / input_scale as its `a1_gscale`). layout_sfa_dummy is unused.
template <class LayoutSFA_t>
__global__ void pack_act_grouped_gs(
    const __nv_bfloat16* __restrict__ act_global,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ ms_arr,
    const int* __restrict__ m_arr,
    unsigned char* const* __restrict__ packed_arr,
    unsigned char* const* __restrict__ scales_arr,
    const float* __restrict__ gs_arr,
    int k,
    LayoutSFA_t layout_sfa_dummy) {
  const int e = blockIdx.z;
  const int m_e = m_arr[e];
  int row = blockIdx.x;
  if (row >= m_e) {
    return;
  }
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  const int groups = k / 16;
  if (group >= groups) {
    return;
  }
  auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m_e, 1, k, 1));
  (void)layout_sfa_dummy;

  unsigned char* packed = packed_arr[e];
  unsigned char* scales = scales_arr[e];
  const int ms = ms_arr[e];
  const float gs = gs_arr[e];
  const float inv_gs = 1.0f / gs;

  int gid = ms + row;
  int tok = sorted_token_ids ? sorted_token_ids[gid] : gid;
  const __nv_bfloat16* arow = act_global + (unsigned long long)tok * k;
  int base = group * 16;
  float v[16];
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    v[i] = __bfloat162float(arow[base + i]);
    max_abs = fmaxf(max_abs, fabsf(v[i]));
  }
  float sf_val = fminf((max_abs / 6.0f) * inv_gs, 448.0f);
  cutlass::float_ue4m3_t sf(sf_val);
  scales[layout_sfa(row, base, 0)] = *reinterpret_cast<unsigned char*>(&sf);
  float dec = static_cast<float>(sf);
  float out_scale = dec > 0.0f ? 1.0f / (dec * gs) : 0.0f;
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    packed[(unsigned long long)row * (k / 2) + base / 2 + i / 2] = static_cast<unsigned char>(
        float_to_e2m1_rne(v[i] * out_scale) | (float_to_e2m1_rne(v[i + 1] * out_scale) << 4));
  }
}

// 2026-10-03: Dynamic per-tensor amax for the groups without a static scale (gs_arr[g] not
// > 0): the max |value| over every row those groups read, as float bits in *amax_bits (zeroed
// by the caller; non-negative floats order like their bits). Grid (max m_e, 1, G), 256 threads.
__global__ void act_amax_grouped(
    const __nv_bfloat16* __restrict__ act_global,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ ms_arr,
    const int* __restrict__ m_arr,
    const float* __restrict__ gs_arr,
    int k,
    unsigned int* __restrict__ amax_bits) {
  const int e = blockIdx.z;
  if (gs_arr[e] > 0.0f) {
    return;
  }
  const int row = blockIdx.x;
  if (row >= m_arr[e]) {
    return;
  }
  const int gid = ms_arr[e] + row;
  const int tok = sorted_token_ids ? sorted_token_ids[gid] : gid;
  const __nv_bfloat16* arow = act_global + (unsigned long long)tok * k;
  float m = 0.0f;
  for (int c = threadIdx.x; c < k; c += blockDim.x) {
    m = fmaxf(m, fabsf(__bfloat162float(arow[c])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  }
  if ((threadIdx.x & 31) == 0) {
    atomicMax(amax_bits, __float_as_uint(m));
  }
}

// 2026-10-03: Every gs[g] that is not > 0 becomes amax / (6 * 448) from act_amax_grouped, the
// value a calibrated input_scale has for that amax; 1.0 when that amax is zero or not finite.
__global__ void resolve_act_gs(
    float* __restrict__ gs, int G, const unsigned int* __restrict__ amax_bits) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= G) {
    return;
  }
  if (!(gs[g] > 0.0f)) {
    const float amax = __uint_as_float(*amax_bits);
    gs[g] = (amax > 0.0f && isfinite(amax)) ? amax / (6.0f * 448.0f) : 1.0f;
  }
}

// 2026-10-03: alpha[g] = s2[g] * gs[g], the epilogue scale of a W4A4 projection.
__global__ void make_alpha_w4a4(
    const float* __restrict__ gs, const float* __restrict__ s2, float* __restrict__ alpha, int G) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g < G) {
    alpha[g] = s2[g] * gs[g];
  }
}

// 2026-10-03: pack_weight_sfb_group for `count` experts in one launch: slot s swizzles
// src_ptrs[first + s] (skipped when null) into out_base + s * out_stride. Grid
// (ceil(n * k/16 / 256), count), 256 threads, one scale byte per thread.
template <class LayoutSFB_t>
__global__ void pack_weight_sfb_batched_k(
    const unsigned long long* __restrict__ src_ptrs,
    int first,
    unsigned char* __restrict__ out_base,
    unsigned long long out_stride,
    int n,
    int k,
    int src_n_major,
    LayoutSFB_t layout_sfb) {
  const int slot = blockIdx.y;
  const unsigned char* src = reinterpret_cast<const unsigned char*>(src_ptrs[first + slot]);
  if (src == nullptr) {
    return;
  }
  const int groups = k / 16;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)n * groups) {
    return;
  }
  const int col = (int)(idx / groups);
  const int group = (int)(idx % groups);
  unsigned char metrale_scale =
      src_n_major ? src[(unsigned long long)col * groups + group]
                  : src[(unsigned long long)group * n + col];
  __nv_fp8_e4m3 in;
  *reinterpret_cast<unsigned char*>(&in) = metrale_scale;
  cutlass::float_ue4m3_t sf(static_cast<float>(in));
  out_base[(unsigned long long)slot * out_stride + layout_sfb(col, group * 16, 0)] =
      *reinterpret_cast<unsigned char*>(&sf);
}

#endif


// 2026-09-25: Load-time SFB swizzle of one expert's weight scales (scale_in [K/16, N]
// E4M3, or [N, K/16] when src_n_major) into scale_out. The SFB layout is built here with
// M = 1 and in launch_projection with each group's M, which assumes it does not depend
// on M. Returns -cudaError when the launch fails.
extern "C" int metrale_cutlass_pack_weight_sfb(
    const void* scale_in,
    void* scale_out,
    int n,
    int k,
    int src_n_major,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (n <= 0 || k <= 0 || (k % 16) != 0) {
    return -1;
  }

  auto layout_sfb =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(1, n, k, 1));
  dim3 block(256);
  dim3 grid(n, (k / 16 + block.x - 1) / block.x);
  pack_weight_sfb_group<<<grid, block, 0, stream>>>(
      static_cast<const unsigned char*>(scale_in),
      static_cast<unsigned char*>(scale_out),
      n,
      k,
      src_n_major,
      layout_sfb);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : -static_cast<int>(err);
#else
  (void)scale_in;
  (void)scale_out;
  (void)n;
  (void)k;
  (void)src_n_major;
  (void)stream;
  return -120;
#endif
}

// 2026-10-03: The load/prefill-time SFB swizzle of `count` experts in one launch: slot s reads
// the E4M3 scales at scale_ptrs_dev[first + s] (a DEVICE pointer table; a null entry is skipped)
// and writes the swizzled SFB at out_base + s * out_stride. Same source layouts and the same
// M-independence assumption as metrale_cutlass_pack_weight_sfb. Returns -1 for bad arguments,
// -3 when out_stride is smaller than one expert's SFB (size(filter_zeros(layout))), else
// -cudaError of the launch.
extern "C" int metrale_cutlass_pack_weight_sfb_batched(
    const unsigned long long* scale_ptrs_dev,
    int first,
    int count,
    void* out_base,
    unsigned long long out_stride,
    int n,
    int k,
    int src_n_major,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (scale_ptrs_dev == nullptr || out_base == nullptr || first < 0 || count <= 0 ||
      count > 65535 || n <= 0 || k <= 0 || (k % 16) != 0) {
    return -1;
  }
  auto layout_sfb =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(1, n, k, 1));
  if ((unsigned long long)size(filter_zeros(layout_sfb)) > out_stride) {
    return -3;
  }
  const long long elems = (long long)n * (k / 16);
  dim3 block(256);
  dim3 grid((unsigned int)((elems + 255) / 256), (unsigned int)count);
  pack_weight_sfb_batched_k<<<grid, block, 0, stream>>>(
      scale_ptrs_dev,
      first,
      static_cast<unsigned char*>(out_base),
      out_stride,
      n,
      k,
      src_n_major,
      layout_sfb);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : -static_cast<int>(err);
#else
  (void)scale_ptrs_dev;
  (void)first;
  (void)count;
  (void)out_base;
  (void)out_stride;
  (void)n;
  (void)k;
  (void)src_n_major;
  (void)stream;
  return -120;
#endif
}

// 2026-09-25: prep_grouped_a gathers and packs A once per call; launch_projection runs one
// kGrouped GEMM against it. gate_up shares one A between gate and up; down packs its own.
// Workspace, in order: packed A | SFA | the A-pack's [G] arrays | the A-side argument
// arrays | the B-side argument arrays and the CUTLASS workspace (per projection).
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

// 2026-09-25: A packed for G groups; group g: host_shapes[g] = {m_e, n, k}, ms[g] its first sorted
// row and C row offset, me[g] = m_e, eidx[g] its expert. cursor: first free byte after the A side.
struct GroupedAPrep {
  int status = 0;
  int G = 0;
  std::vector<ProblemShape> host_shapes;
  std::vector<int> ms;
  std::vector<int> me;
  std::vector<int> eidx;
  ProblemShape* dShapes = nullptr;
  const ElementA::DataType** dA = nullptr;
  const ElementSF** dSFA = nullptr;
  StrideA* dsA = nullptr;
  LayoutSFA* dlSFA = nullptr;
  size_t cursor = 0;
  // 2026-10-03: W4A4 global-scale mode only (act_gs_host non-null): device alpha arrays,
  // alpha[j][g] = (j-th projection's scale2 of group g's expert) * (group g's global scale).
  float* dAlpha[2] = {nullptr, nullptr};
};











// 2026-09-25: Gather and pack A for every active group and upload the A-side argument
// arrays. An expert is active when it has rows and, unless b_valid_ptrs is null or the
// null guard is off, a nonzero b_valid_ptrs[e]. A 0 entry is an expert whose weights a
// loader left null (deepseek_v4 does so for experts a rank does not hold): it gets no
// group, and its C rows keep their contents (forward_prefill_routed zeroes them first
// when `comm` is set).
// 2026-10-03: With act_gs_host null (every pre-existing caller) nothing below changes. With it
// set (the `_w4a4` entry points): act_gs_host[e] is expert e's NVFP4 global activation scale
// (> 0 static, else dynamic: amax / (6 * 448) over the rows of the dynamic groups), A is packed
// by pack_act_grouped_gs, and for j < n_alpha (at most 2) p.dAlpha[j][g] = alpha_s2_host[j][e]
// * gs. Before any launch the A side is checked against workspace_size: on overflow p.status =
// -2 and p.G = 0, nothing launched.
static GroupedAPrep prep_grouped_a(
    const __nv_bfloat16* A_global,
    const int* sorted_token_ids,
    const int* expert_offsets_host,
    const unsigned long long* b_valid_ptrs,
    int num_experts,
    int n,
    int k,
    unsigned char* ws,
    cudaStream_t stream,
    const float* act_gs_host = nullptr,
    int n_alpha = 0,
    const float* const* alpha_s2_host = nullptr,
    size_t workspace_size = 0) {


  // 2026-09-25: METRALE_CUTLASS_EP_NULL_GUARD starting with '0' turns the guard off: every
  // expert with rows is grouped, and a null B then makes launch_projection return -140.
  static const bool null_guard = [] {
    const char* v = getenv("METRALE_CUTLASS_EP_NULL_GUARD");
    return !(v != nullptr && v[0] == '0');
  }();

  // 2026-09-25: Both passes use this predicate, so a_grp_off / sfa_grp_off (indexed by
  // `gi`) line up with the groups the second pass builds.
  auto group_active = [&](int e) -> bool {
    if (expert_offsets_host[e + 1] - expert_offsets_host[e] <= 0) {
      return false;
    }
    return !null_guard || b_valid_ptrs == nullptr || b_valid_ptrs[e] != 0;
  };
  GroupedAPrep p;
  std::vector<const ElementA::DataType*> hA;
  std::vector<const ElementSF*> hSFA;
  std::vector<StrideA> sA;

  // 2026-09-25: First pass: each group's 256-byte-aligned packed-A and SFA offsets.
  std::vector<size_t> a_grp_off;
  std::vector<size_t> sfa_grp_off;
  size_t a_acc = 0;
  size_t sfa_acc = 0;
  for (int e = 0; e < num_experts; ++e) {
    int m_e = expert_offsets_host[e + 1] - expert_offsets_host[e];
    if (!group_active(e)) {
      continue;
    }
    auto lsa =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m_e, n, k, 1));
    a_grp_off.push_back(a_acc);
    sfa_grp_off.push_back(sfa_acc);
    a_acc += align_up_((size_t)m_e * (k / 2), 256);
    sfa_acc += align_up_((size_t)size(filter_zeros(lsa)), 256);
  }
  size_t a_off = 0;
  size_t sfa_off = align_up_(a_acc, 256);
  size_t cursor = align_up_(sfa_off + sfa_acc, 256);

  // 2026-10-03: W4A4 mode: the whole A side (packed A, SFA, every [G] array below) plus one
  // projection's B-side [G] arrays (launch_projection writes those before its own room check),
  // each array 256-aligned, must fit before anything is written.
  if (act_gs_host != nullptr) {
    const size_t g_est = a_grp_off.size();
    const size_t per_group = 2 * sizeof(int) + 4 * sizeof(void*) +
                             (2 + 2 * (size_t)n_alpha) * sizeof(float) + sizeof(ProblemShape) +
                             sizeof(StrideA) + sizeof(LayoutSFA) + 5 * sizeof(void*) +
                             sizeof(StrideB) + sizeof(StrideC) + sizeof(StrideD) +
                             sizeof(LayoutSFB) + sizeof(float);
    const size_t arrays = g_est * per_group + 48 * 256;
    if (cursor + arrays > workspace_size) {
      p.status = -2;
      p.cursor = cursor;
      return p;
    }
  }

  // 2026-09-25: Second pass: collect each group's A-pack scalars and A-side arguments;
  // the A-pack is one launch after the loop.
  std::vector<int> h_ms, h_me;
  std::vector<unsigned char*> h_apk, h_sfa;
  int max_me = 0;
  int gi = 0;
  for (int e = 0; e < num_experts; ++e) {
    int ms = expert_offsets_host[e];
    int m_e = expert_offsets_host[e + 1] - ms;
    if (!group_active(e)) {
      continue;
    }
    auto lsa =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m_e, n, k, 1));
    unsigned char* a_e = ws + a_off + a_grp_off[gi];
    unsigned char* sfa_e = ws + sfa_off + sfa_grp_off[gi];

    dim3 blk(256);
    dim3 grd(m_e, (k / 16 + blk.x - 1) / blk.x);



    h_ms.push_back(ms);
    h_me.push_back(m_e);
    h_apk.push_back(a_e);
    h_sfa.push_back(sfa_e);
    if (m_e > max_me) max_me = m_e;
    (void)lsa;

    p.host_shapes.push_back(ProblemShape{m_e, n, k});
    p.ms.push_back(ms);
    p.me.push_back(m_e);
    p.eidx.push_back(e);
    hA.push_back(reinterpret_cast<const ElementA::DataType*>(a_e));
    hSFA.push_back(reinterpret_cast<const ElementSF*>(sfa_e));
    sA.push_back(cutlass::make_cute_packed_stride(StrideA{}, {m_e, k, 1}));
    ++gi;
  }


  // 2026-09-25: One A-pack launch for every active group.
  if (!h_me.empty()) {
    const int G = (int)h_me.size();
    size_t ms_b = align_up_((size_t)G * sizeof(int), 256);
    size_t pk_b = align_up_((size_t)G * sizeof(void*), 256);
    unsigned char* d_ms = ws + cursor;
    unsigned char* d_me = d_ms + ms_b;
    unsigned char* d_apk = d_me + ms_b;
    unsigned char* d_sfa = d_apk + pk_b;
    cursor = align_up_(cursor + 2 * ms_b + 2 * pk_b, 256);
    cudaMemcpyAsync(d_ms, h_ms.data(), G * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_me, h_me.data(), G * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_apk, h_apk.data(), G * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sfa, h_sfa.data(), G * sizeof(void*), cudaMemcpyHostToDevice, stream);
    auto lsa0 = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(max_me, n, k, 1));
    dim3 blk(256);
    dim3 grd(max_me, (k / 16 + blk.x - 1) / blk.x, G);
    if (act_gs_host == nullptr) {
      pack_act_grouped_batched<<<grd, blk, 0, stream>>>(
          A_global, sorted_token_ids, (const int*)d_ms, (const int*)d_me,
          (unsigned char* const*)d_apk, (unsigned char* const*)d_sfa, k, lsa0);
    } else {
      // 2026-10-03: Per-group global scales (eidx is complete: the loop above pushed one
      // entry per group), the dynamic amax when a group has no static scale, the pack, then
      // each projection's alpha.
      const size_t f_b = align_up_((size_t)G * sizeof(float), 256);
      std::vector<float> h_gs(G);
      bool any_dyn = false;
      for (int g = 0; g < G; ++g) {
        const float v = act_gs_host[p.eidx[g]];
        const bool ok = v > 0.0f && v < 3.0e38f;
        h_gs[g] = ok ? v : 0.0f;
        any_dyn = any_dyn || !ok;
      }
      float* d_gs = reinterpret_cast<float*>(ws + cursor);
      unsigned int* d_amax = reinterpret_cast<unsigned int*>(ws + cursor + f_b);
      cursor = align_up_(cursor + f_b + 256, 256);
      cudaMemcpyAsync(d_gs, h_gs.data(), G * sizeof(float), cudaMemcpyHostToDevice, stream);
      if (any_dyn) {
        cudaMemsetAsync(d_amax, 0, sizeof(unsigned int), stream);
        dim3 ablk(256);
        dim3 agrd(max_me, 1, G);
        act_amax_grouped<<<agrd, ablk, 0, stream>>>(
            A_global, sorted_token_ids, (const int*)d_ms, (const int*)d_me, d_gs, k, d_amax);
        resolve_act_gs<<<(G + 255) / 256, 256, 0, stream>>>(d_gs, G, d_amax);
      }
      pack_act_grouped_gs<<<grd, blk, 0, stream>>>(
          A_global, sorted_token_ids, (const int*)d_ms, (const int*)d_me,
          (unsigned char* const*)d_apk, (unsigned char* const*)d_sfa, d_gs, k, lsa0);
      for (int j = 0; j < n_alpha && j < 2; ++j) {
        std::vector<float> h_s2(G);
        for (int g = 0; g < G; ++g) {
          h_s2[g] = alpha_s2_host[j][p.eidx[g]];
        }
        float* d_s2 = reinterpret_cast<float*>(ws + cursor);
        cursor = align_up_(cursor + f_b, 256);
        float* d_alpha = reinterpret_cast<float*>(ws + cursor);
        cursor = align_up_(cursor + f_b, 256);
        cudaMemcpyAsync(d_s2, h_s2.data(), G * sizeof(float), cudaMemcpyHostToDevice, stream);
        make_alpha_w4a4<<<(G + 255) / 256, 256, 0, stream>>>(d_gs, d_s2, d_alpha, G);
        p.dAlpha[j] = d_alpha;
      }
    }
  }

  p.G = (int)p.host_shapes.size();
  if (p.G == 0) {
    p.cursor = cursor;
    return p;
  }

  auto put = [&](const void* src, size_t bytes) -> void* {
    void* dst = ws + cursor;
    cursor = align_up_(cursor + bytes, 256);
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    return dst;
  };
  p.dShapes = (ProblemShape*)put(p.host_shapes.data(), p.G * sizeof(ProblemShape));
  p.dA = (const ElementA::DataType**)put(hA.data(), p.G * sizeof(void*));
  p.dSFA = (const ElementSF**)put(hSFA.data(), p.G * sizeof(void*));
  p.dsA = (StrideA*)put(sA.data(), p.G * sizeof(StrideA));
  {
    std::vector<LayoutSFA> lSFA(p.G);
    for (int g = 0; g < p.G; ++g) {
      lSFA[g] =
          Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(p.me[g], n, k, 1));
    }
    p.dlSFA = (LayoutSFA*)put(lSFA.data(), p.G * sizeof(LayoutSFA));
  }
  p.cursor = cursor;
  return p;
}

// 2026-09-25: One kGrouped GEMM over a's groups, arrays and CUTLASS workspace from cursor_start.
// Status: -140 null B/SFB pointer, -2 no room, tag - 50 can_implement, else tag + CUTLASS status.
static int launch_projection(
    const GroupedAPrep& a,
    const unsigned long long* packed_ptrs,
    const unsigned long long* sfb_ptrs,
    const float* scale2_vals,
    __nv_bfloat16* C_bf16,
    int n,
    int k,
    unsigned char* ws,
    size_t cursor_start,
    size_t workspace_size,
    cudaStream_t stream,
    int tag,
    const float* d_alpha = nullptr) {
  int G = a.G;
  if (G == 0) {
    return 0;
  }
  std::vector<const ElementB::DataType*> hB(G);
  std::vector<const ElementSF*> hSFB(G);
  std::vector<const ElementC*> hC(G);
  std::vector<ElementD*> hD(G);
  std::vector<StrideB> sB(G);
  std::vector<StrideC> sC(G);
  std::vector<StrideD> sD(G);
  std::vector<LayoutSFB> lSFB(G);
  std::vector<float> alpha_host(G);
  for (int g = 0; g < G; ++g) {
    int e = a.eidx[g];
    int m_e = a.me[g];
    size_t ms = (size_t)a.ms[g];


    // 2026-09-25: A null B or SFB pointer returns -140 before any launch, instead of
    // an illegal-address fault inside the GEMM.
    if (packed_ptrs[e] == 0 || sfb_ptrs[e] == 0) {
      return -140;
    }
    hB[g] = reinterpret_cast<const ElementB::DataType*>(packed_ptrs[e]);
    hSFB[g] = reinterpret_cast<const ElementSF*>(sfb_ptrs[e]);
    hC[g] = reinterpret_cast<const ElementC*>(C_bf16 + ms * n);
    hD[g] = reinterpret_cast<ElementD*>(C_bf16 + ms * n);
    sB[g] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    sC[g] = cutlass::make_cute_packed_stride(StrideC{}, {m_e, n, 1});
    sD[g] = cutlass::make_cute_packed_stride(StrideD{}, {m_e, n, 1});
    lSFB[g] =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m_e, n, k, 1));
    alpha_host[g] = scale2_vals[e];
  }

  size_t cursor = cursor_start;
  auto put = [&](const void* src, size_t bytes) -> void* {
    void* dst = ws + cursor;
    cursor = align_up_(cursor + bytes, 256);
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    return dst;
  };
  auto* dB = (const ElementB::DataType**)put(hB.data(), G * sizeof(void*));
  auto* dSFB = (const ElementSF**)put(hSFB.data(), G * sizeof(void*));
  auto* dC = (const ElementC**)put(hC.data(), G * sizeof(void*));
  auto* dD = (ElementD**)put(hD.data(), G * sizeof(void*));
  auto* dsB = (StrideB*)put(sB.data(), G * sizeof(StrideB));
  auto* dsC = (StrideC*)put(sC.data(), G * sizeof(StrideC));
  auto* dsD = (StrideD*)put(sD.data(), G * sizeof(StrideD));
  auto* dlSFB = (LayoutSFB*)put(lSFB.data(), G * sizeof(LayoutSFB));
  // 2026-10-03: d_alpha (W4A4 mode: per-group device alphas from prep_grouped_a) replaces the
  // host scale2 upload; null (every pre-existing caller) uploads alpha_host as before.
  const float* dAlpha = d_alpha;
  if (dAlpha == nullptr) {
    dAlpha = (const float*)put(alpha_host.data(), G * sizeof(float));
  }

  // 2026-09-25: Group g reads its alpha through alpha_ptr_array[g] = &dAlpha[g].
  std::vector<const float*> hAlphaPtr(G);
  for (int g = 0; g < G; ++g) {
    hAlphaPtr[g] = dAlpha + g;
  }
  auto* dAlphaPtr = (const float**)put(hAlphaPtr.data(), G * sizeof(const float*));

  cutlass::KernelHardwareInfo hw{};
  hw.device_id = 0;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  typename Gemm::GemmKernel::CollectiveMainloop::Arguments mainloop_args{
      a.dA, a.dsA, dB, dsB, a.dSFA, a.dlSFA, dSFB, dlSFB};

  typename Gemm::GemmKernel::CollectiveEpilogue::Arguments epi_args{};
  epi_args.thread.alpha = 1.0f;
  epi_args.thread.beta = 0.0f;
  epi_args.thread.alpha_ptr_array = dAlphaPtr;
  epi_args.ptr_C = dC;
  epi_args.dC = dsC;
  epi_args.ptr_D = dD;
  epi_args.dD = dsD;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {G, a.dShapes, const_cast<ProblemShape*>(a.host_shapes.data())},
      mainloop_args,
      epi_args,
      hw};

  Gemm gemm;
  size_t need = Gemm::get_workspace_size(args);
  if (cursor + need > workspace_size) {
    return -2;
  }
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    return tag + (-50);
  }
  cutlass::Status st = gemm.initialize(args, ws + cursor, stream);
  if (st != cutlass::Status::kSuccess) {
    return tag + static_cast<int>(st);
  }
  st = gemm.run(stream);
  return st == cutlass::Status::kSuccess ? 0 : tag + static_cast<int>(st);
}
#endif



// 2026-09-25: Grouped gate/up. A_bf16 is token-major [num_tokens, K]; sorted row r reads
// token sorted_token_ids[r]. *_packed_ptrs[e] points at [N, K/2] E2M1, *_sfb_ptrs[e] at the
// swizzled SFB; the pointer tables, *_scale2_vals and expert_offsets_host are host arrays.
// C_gate / C_up are [M_total, N] in sorted-row order. Status tags: 100000 gate, 200000 up.
extern "C" int metrale_cutlass_nvfp4_grouped_gate_up_fused(
    const void* A_bf16,
    const int* sorted_token_ids,
    const unsigned long long* gate_packed_ptrs,
    const unsigned long long* gate_sfb_ptrs,
    const float* gate_scale2_vals,
    const unsigned long long* up_packed_ptrs,
    const unsigned long long* up_sfb_ptrs,
    const float* up_scale2_vals,
    void* C_gate_bf16,
    void* C_up_bf16,
    const int* expert_offsets_host,
    int num_experts,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (n <= 0 || k <= 0 || (k % 16) != 0 || num_experts <= 0) {
    return -1;
  }
  unsigned char* ws = static_cast<unsigned char*>(workspace);

  // 2026-09-25: One A for gate and up, filtered by gate's pointer table; a group whose up
  // pointer is null then fails with -140.
  GroupedAPrep a = prep_grouped_a(static_cast<const __nv_bfloat16*>(A_bf16),
                                  sorted_token_ids, expert_offsets_host,
                                  gate_packed_ptrs, num_experts,
                                  n, k, ws, stream);
  if (a.G == 0) {
    return 0;
  }
  // 2026-09-25: Both projections carve from a.cursor. Up's uploads overwrite gate's
  // arrays only after gate's GEMM, because both are ordered on `stream`.
  int rc = launch_projection(a, gate_packed_ptrs, gate_sfb_ptrs, gate_scale2_vals,
                             static_cast<__nv_bfloat16*>(C_gate_bf16), n, k, ws,
                             a.cursor, workspace_size, stream, 100000);
  if (rc) {
    return rc;
  }
  rc = launch_projection(a, up_packed_ptrs, up_sfb_ptrs, up_scale2_vals,
                         static_cast<__nv_bfloat16*>(C_up_bf16), n, k, ws, a.cursor,
                         workspace_size, stream, 200000);
  return rc;
#else
  (void)A_bf16;
  (void)sorted_token_ids;
  (void)gate_packed_ptrs;
  (void)gate_sfb_ptrs;
  (void)gate_scale2_vals;
  (void)up_packed_ptrs;
  (void)up_sfb_ptrs;
  (void)up_scale2_vals;
  (void)C_gate_bf16;
  (void)C_up_bf16;
  (void)expert_offsets_host;
  (void)num_experts;
  (void)n;
  (void)k;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  return -120;
#endif
}



// 2026-09-25: Grouped down. A_bf16 is [M_total, K] and already in sorted-row order (no
// gather); packed_ptrs[e] points at [N, K/2] E2M1. Status tag 300000.
extern "C" int metrale_cutlass_nvfp4_grouped_down(
    const void* A_bf16,
    const unsigned long long* packed_ptrs,
    const unsigned long long* sfb_ptrs,
    const float* scale2_vals,
    void* C_bf16,
    const int* expert_offsets_host,
    int num_experts,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (n <= 0 || k <= 0 || (k % 16) != 0 || num_experts <= 0) {
    return -1;
  }
  unsigned char* ws = static_cast<unsigned char*>(workspace);
  GroupedAPrep a = prep_grouped_a(static_cast<const __nv_bfloat16*>(A_bf16), nullptr,
                                  expert_offsets_host, packed_ptrs, num_experts, n, k,
                                  ws, stream);
  if (a.G == 0) {
    return 0;
  }
  return launch_projection(a, packed_ptrs, sfb_ptrs, scale2_vals,
                           static_cast<__nv_bfloat16*>(C_bf16), n, k, ws, a.cursor,
                           workspace_size, stream, 300000);
#else
  (void)A_bf16;
  (void)packed_ptrs;
  (void)sfb_ptrs;
  (void)scale2_vals;
  (void)C_bf16;
  (void)expert_offsets_host;
  (void)num_experts;
  (void)n;
  (void)k;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  return -120;
#endif
}

// 2026-10-03: W4A4 grouped gate/up with an NVFP4 global activation scale: as
// metrale_cutlass_nvfp4_grouped_gate_up_fused, except that A is quantized against
// act_gscale_vals[e] (host, per expert; > 0 = static, the checkpoint's input_scale, shared by
// gate and up; otherwise dynamic per-tensor amax / (6 * 448) over those experts' rows), and
// gate's / up's epilogue alpha is scale2[e] * that scale. Status: as the fused entry, plus -2
// when the A side does not fit the workspace (nothing launched). Tags 100000 gate, 200000 up.
extern "C" int metrale_cutlass_nvfp4_grouped_gate_up_w4a4(
    const void* A_bf16,
    const int* sorted_token_ids,
    const unsigned long long* gate_packed_ptrs,
    const unsigned long long* gate_sfb_ptrs,
    const float* gate_scale2_vals,
    const unsigned long long* up_packed_ptrs,
    const unsigned long long* up_sfb_ptrs,
    const float* up_scale2_vals,
    const float* act_gscale_vals,
    void* C_gate_bf16,
    void* C_up_bf16,
    const int* expert_offsets_host,
    int num_experts,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (n <= 0 || k <= 0 || (k % 16) != 0 || num_experts <= 0 || act_gscale_vals == nullptr) {
    return -1;
  }
  unsigned char* ws = static_cast<unsigned char*>(workspace);
  const float* s2[2] = {gate_scale2_vals, up_scale2_vals};
  GroupedAPrep a = prep_grouped_a(static_cast<const __nv_bfloat16*>(A_bf16),
                                  sorted_token_ids, expert_offsets_host,
                                  gate_packed_ptrs, num_experts,
                                  n, k, ws, stream, act_gscale_vals, 2, s2, workspace_size);
  if (a.status != 0) {
    return a.status;
  }
  if (a.G == 0) {
    return 0;
  }
  int rc = launch_projection(a, gate_packed_ptrs, gate_sfb_ptrs, gate_scale2_vals,
                             static_cast<__nv_bfloat16*>(C_gate_bf16), n, k, ws,
                             a.cursor, workspace_size, stream, 100000, a.dAlpha[0]);
  if (rc) {
    return rc;
  }
  return launch_projection(a, up_packed_ptrs, up_sfb_ptrs, up_scale2_vals,
                           static_cast<__nv_bfloat16*>(C_up_bf16), n, k, ws, a.cursor,
                           workspace_size, stream, 200000, a.dAlpha[1]);
#else
  (void)A_bf16;
  (void)sorted_token_ids;
  (void)gate_packed_ptrs;
  (void)gate_sfb_ptrs;
  (void)gate_scale2_vals;
  (void)up_packed_ptrs;
  (void)up_sfb_ptrs;
  (void)up_scale2_vals;
  (void)act_gscale_vals;
  (void)C_gate_bf16;
  (void)C_up_bf16;
  (void)expert_offsets_host;
  (void)num_experts;
  (void)n;
  (void)k;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  return -120;
#endif
}

// 2026-10-03: W4A4 grouped down with an NVFP4 global activation scale: as
// metrale_cutlass_nvfp4_grouped_down (A in sorted-row order, no gather), with act_gscale_vals
// and alpha as in metrale_cutlass_nvfp4_grouped_gate_up_w4a4. Status tag 300000; -2 when the
// A side does not fit the workspace (nothing launched).
extern "C" int metrale_cutlass_nvfp4_grouped_down_w4a4(
    const void* A_bf16,
    const unsigned long long* packed_ptrs,
    const unsigned long long* sfb_ptrs,
    const float* scale2_vals,
    const float* act_gscale_vals,
    void* C_bf16,
    const int* expert_offsets_host,
    int num_experts,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (n <= 0 || k <= 0 || (k % 16) != 0 || num_experts <= 0 || act_gscale_vals == nullptr) {
    return -1;
  }
  unsigned char* ws = static_cast<unsigned char*>(workspace);
  const float* s2[1] = {scale2_vals};
  GroupedAPrep a = prep_grouped_a(static_cast<const __nv_bfloat16*>(A_bf16), nullptr,
                                  expert_offsets_host, packed_ptrs, num_experts, n, k,
                                  ws, stream, act_gscale_vals, 1, s2, workspace_size);
  if (a.status != 0) {
    return a.status;
  }
  if (a.G == 0) {
    return 0;
  }
  return launch_projection(a, packed_ptrs, sfb_ptrs, scale2_vals,
                           static_cast<__nv_bfloat16*>(C_bf16), n, k, ws, a.cursor,
                           workspace_size, stream, 300000, a.dAlpha[0]);
#else
  (void)A_bf16;
  (void)packed_ptrs;
  (void)sfb_ptrs;
  (void)scale2_vals;
  (void)act_gscale_vals;
  (void)C_bf16;
  (void)expert_offsets_host;
  (void)num_experts;
  (void)n;
  (void)k;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  return -120;
#endif
}
