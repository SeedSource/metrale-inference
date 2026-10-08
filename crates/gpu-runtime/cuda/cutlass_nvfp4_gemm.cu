// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: CUTLASS Sm120 NVFP4 GEMM host wrappers: the dense GEMM, a per-expert grouped
// gate/up loop over it, and the weight pack and transpose helpers.
// Owner: gpu-runtime (CUTLASS reference objects).
// Invariants:
// - Every extern "C" entry returns 0 on success and a nonzero status otherwise.
// - Built with CUTLASS_ARCH_MMA_SM120/SM121 support, every entry returns -1 for k <= 0 or
//   k % 16 != 0; built without it, every entry but the transpose returns -120.
// - 2026-10-08: The dense W4A4 entries at the end (METRALE_GLM_PREFILL_DENSE_W4A4) reuse the
//   dense GEMM type above and take n, k positive multiples of 128 (-1 otherwise; the SFB size
//   query returns 0 bytes instead); the entries before them launch exactly what they launched
//   before.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ThreadBlockShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutCTag,
    AlignmentC,
    ElementD,
    LayoutDTag,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutATag,
    AlignmentA,
    ElementB,
    LayoutBTag,
    AlignmentB,
    ElementAccumulator,
    ThreadBlockShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

template <typename T>
__device__ __forceinline__ unsigned char byte_of(T v) {
  return *reinterpret_cast<unsigned char*>(&v);
}

__device__ __forceinline__ unsigned char float_to_e2m1(float x) {
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

__device__ __forceinline__ float fp8_e4m3_to_float(unsigned char byte) {
  __nv_fp8_e4m3 v;
  *reinterpret_cast<unsigned char*>(&v) = byte;
  return static_cast<float>(v);
}

template <class Layout>
__global__ void metrale_cutlass_pack_bf16_act_nvfp4(
    const __nv_bfloat16* __restrict__ act,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ scales,
    int m,
    int k,
    Layout layout_sfa) {
  int row = blockIdx.x;
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  int groups = k / 16;
  if (row >= m || group >= groups) {
    return;
  }

  int base = group * 16;
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float v = __bfloat162float(act[(unsigned long long)row * k + base + i]);
    max_abs = fmaxf(max_abs, fabsf(v));
  }

  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  cutlass::float_ue4m3_t sf(scale);
  scales[layout_sfa(row, base, 0)] = byte_of(sf);
  float decoded_scale = static_cast<float>(sf);
  float inv_scale = decoded_scale > 0.0f ? 1.0f / decoded_scale : 0.0f;

#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    float v0 = __bfloat162float(act[(unsigned long long)row * k + base + i]) * inv_scale;
    float v1 = __bfloat162float(act[(unsigned long long)row * k + base + i + 1]) * inv_scale;
    packed[(unsigned long long)row * (k / 2) + base / 2 + i / 2] =
        static_cast<unsigned char>(float_to_e2m1(v0) | (float_to_e2m1(v1) << 4));
  }
}

template <class Layout>
__global__ void metrale_cutlass_pack_nvfp4_weight_scales_t(
    const unsigned char* __restrict__ metrale_scales_t,
    unsigned char* __restrict__ cutlass_scales,
    float scale2,
    int n,
    int k,
    Layout layout_sfb) {
  int col = blockIdx.x;
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  int groups = k / 16;
  if (col >= n || group >= groups) {
    return;
  }
  unsigned char metrale_scale = metrale_scales_t[(unsigned long long)group * n + col];
  float scale = fp8_e4m3_to_float(metrale_scale);
  cutlass::float_ue4m3_t sf(scale);
  cutlass_scales[layout_sfb(col, group * 16, 0)] = byte_of(sf);
}

__global__ void metrale_cutlass_pack_bf16_weight_nvfp4_t(
    const __nv_bfloat16* __restrict__ weight,
    unsigned char* __restrict__ packed_t,
    unsigned char* __restrict__ scale_t,
    int n,
    int k) {
  int col = blockIdx.x;
  int group = blockIdx.y * blockDim.x + threadIdx.x;
  int groups = k / 16;
  if (col >= n || group >= groups) {
    return;
  }

  int base = group * 16;
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float v = __bfloat162float(weight[(unsigned long long)col * k + base + i]);
    max_abs = fmaxf(max_abs, fabsf(v));
  }

  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  __nv_fp8_e4m3 fp8_scale(scale);
  scale_t[(unsigned long long)group * n + col] = byte_of(fp8_scale);
  float decoded_scale = static_cast<float>(fp8_scale);
  float inv_scale = decoded_scale > 0.0f ? 1.0f / decoded_scale : 0.0f;

#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    float v0 = __bfloat162float(weight[(unsigned long long)col * k + base + i]) * inv_scale;
    float v1 = __bfloat162float(weight[(unsigned long long)col * k + base + i + 1]) * inv_scale;
    // 2026-09-25: CUTLASS ColumnMajor B is K-contiguous, [N, K/2] bytes, not
    // the [K/2, N] layout metrale_cutlass_transpose_nvfp4_packed_kton reads.
    packed_t[(unsigned long long)col * (k / 2) + base / 2 + i / 2] =
        static_cast<unsigned char>(float_to_e2m1(v0) | (float_to_e2m1(v1) << 4));
  }
}

size_t align_up(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}

#endif

// 2026-09-25: One NVFP4 GEMM call. Packs the activation and the weight scales into
// `workspace`. Status: -1 bad shape, -2 when the packed operands plus the CUTLASS
// workspace do not fit, -10 - err / -1000 - err when the activation / scale pack
// launch fails, else the CUTLASS status.
static int nvfp4_gemm_single_tile(
    const void* act_bf16,
    const void* weight_packed_t,
    const void* weight_scale_t,
    float weight_scale_2,
    void* out_bf16,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (m <= 0 || n <= 0 || k <= 0 || (k % 16) != 0) {
    return -1;
  }

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  LayoutA layout_a = make_layout(make_shape(m, k, 1), stride_a);
  auto layout_sfa = CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, n, k, 1));
  auto layout_sfb = CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, n, k, 1));

  size_t a_bytes = align_up(static_cast<size_t>(size(layout_a)), 256);
  size_t sfa_bytes = align_up(static_cast<size_t>(size(filter_zeros(layout_sfa))), 256);
  size_t sfb_bytes = align_up(static_cast<size_t>(size(filter_zeros(layout_sfb))), 256);

  Gemm gemm;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {
          reinterpret_cast<ElementA::DataType const*>(workspace),
          stride_a,
          reinterpret_cast<ElementB::DataType const*>(weight_packed_t),
          stride_b,
          reinterpret_cast<ElementA::ScaleFactorType const*>(
              static_cast<unsigned char*>(workspace) + a_bytes),
          layout_sfa,
          reinterpret_cast<ElementB::ScaleFactorType const*>(
              static_cast<unsigned char*>(workspace) + a_bytes + sfa_bytes),
          layout_sfb,
      },
      {
          {weight_scale_2, 0.0f},
          reinterpret_cast<ElementC const*>(out_bf16),
          stride_c,
          reinterpret_cast<ElementD*>(out_bf16),
          stride_d,
      }};

  size_t gemm_workspace_size = Gemm::get_workspace_size(args);
  size_t gemm_workspace_off = align_up(a_bytes + sfa_bytes + sfb_bytes, 256);
  if (gemm_workspace_off + gemm_workspace_size > workspace_size) {
    return -2;
  }

  dim3 block(256);
  dim3 grid_act(m, (k / 16 + block.x - 1) / block.x);
  metrale_cutlass_pack_bf16_act_nvfp4<<<grid_act, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(act_bf16),
      static_cast<unsigned char*>(workspace),
      static_cast<unsigned char*>(workspace) + a_bytes,
      m,
      k,
      layout_sfa);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return -10 - static_cast<int>(err);
  }

  dim3 grid_sfb(n, (k / 16 + block.x - 1) / block.x);
  metrale_cutlass_pack_nvfp4_weight_scales_t<<<grid_sfb, block, 0, stream>>>(
      static_cast<const unsigned char*>(weight_scale_t),
      static_cast<unsigned char*>(workspace) + a_bytes + sfa_bytes,
      weight_scale_2,
      n,
      k,
      layout_sfb);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return -1000 - static_cast<int>(err);
  }

  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  status = gemm.initialize(
      args,
      static_cast<unsigned char*>(workspace) + gemm_workspace_off,
      stream);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  status = gemm(stream);
  return static_cast<int>(status);
#else
  (void)act_bf16;
  (void)weight_packed_t;
  (void)weight_scale_t;
  (void)weight_scale_2;
  (void)out_bf16;
  (void)m;
  (void)n;
  (void)k;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  return -120;
#endif
}


// 2026-09-25: out[m, n] = nvfp4(act[m, k]) @ weight_t[n, k]^T, BF16 out. The packed
// activation grows with M, so M is split into calls of at most 4096 rows that share
// the weight; the first failing call's status is returned. m <= 4096 is one call.
// weight_scale_2 is applied as the epilogue alpha (beta 0).
extern "C" int metrale_cutlass_nvfp4_gemm_bf16_act_weight_t(
    const void* act_bf16,
    const void* weight_packed_t,
    const void* weight_scale_t,
    float weight_scale_2,
    void* out_bf16,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  const int M_TILE = 4096;
  if (m <= M_TILE) {
    return nvfp4_gemm_single_tile(act_bf16, weight_packed_t, weight_scale_t,
        weight_scale_2, out_bf16, m, n, k, workspace, workspace_size, stream);
  }
  for (int m0 = 0; m0 < m; m0 += M_TILE) {
    int sub_m = (m - m0 < M_TILE) ? (m - m0) : M_TILE;
    const void* a_off = static_cast<const void*>(
        static_cast<const __nv_bfloat16*>(act_bf16) + static_cast<size_t>(m0) * k);
    void* c_off = static_cast<void*>(
        static_cast<__nv_bfloat16*>(out_bf16) + static_cast<size_t>(m0) * n);
    int rc = nvfp4_gemm_single_tile(a_off, weight_packed_t, weight_scale_t,
        weight_scale_2, c_off, sub_m, n, k, workspace, workspace_size, stream);
    if (rc != 0) {
      return rc;
    }
  }
  return 0;
}

// 2026-09-25: Byte transpose of a packed NVFP4 weight from [K/2, N] (N-contiguous)
// to CUTLASS's [N, K/2] (K-contiguous), the B layout
// metrale_cutlass_nvfp4_gemm_bf16_act_weight_t reads. The two nibbles of each byte
// are not reordered.
__global__ void metrale_cutlass_transpose_nvfp4_packed(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int n,
    int k) {
  int half = k / 2;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (c >= n || h >= half) {
    return;
  }

  dst[(unsigned long long)c * half + h] = src[(unsigned long long)h * n + c];
}

extern "C" int metrale_cutlass_transpose_nvfp4_packed_kton(
    const void* src_packed_t,
    void* dst_packed,
    int n,
    int k,
    cudaStream_t stream) {
  if (n <= 0 || k <= 0 || (k % 16) != 0) {
    return -1;
  }
  dim3 block(32, 8);
  dim3 grid((n + block.x - 1) / block.x, (k / 2 + block.y - 1) / block.y);
  metrale_cutlass_transpose_nvfp4_packed<<<grid, block, 0, stream>>>(
      static_cast<const unsigned char*>(src_packed_t),
      static_cast<unsigned char*>(dst_packed),
      n,
      k);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : -static_cast<int>(err);
}





// 2026-09-25: Grouped NVFP4 gate/up as a loop of metrale_cutlass_nvfp4_gemm_bf16_act_weight_t
// calls, one gate and one up call per expert with rows.
//   A_bf16            [M_total, K] BF16; expert e owns rows
//                     [expert_offsets_host[e], expert_offsets_host[e+1]).
//   *_packed_ptrs[e]  device pointer to [N, K/2] packed E2M1;
//   *_scale_ptrs[e]   device pointer to [K/16, N] E4M3 scales.
//   C_gate / C_up     [M_total, N] BF16, rows as in A.
//   *_scale2_vals[e]  expert e's weight_scale_2.
// The pointer tables, *_scale2_vals and expert_offsets_host ([num_experts + 1]) are
// read on the host. Experts with no rows are skipped. A failing call returns
// 100000 + status (gate) or 200000 + status (up).
extern "C" int metrale_cutlass_nvfp4_grouped_gate_up(
    const void* A_bf16,
    const unsigned long long* gate_packed_ptrs,
    const unsigned long long* gate_scale_ptrs,
    const float* gate_scale2_vals,
    const unsigned long long* up_packed_ptrs,
    const unsigned long long* up_scale_ptrs,
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
  const __nv_bfloat16* A = static_cast<const __nv_bfloat16*>(A_bf16);
  __nv_bfloat16* Cg = static_cast<__nv_bfloat16*>(C_gate_bf16);
  __nv_bfloat16* Cu = static_cast<__nv_bfloat16*>(C_up_bf16);

  for (int e = 0; e < num_experts; ++e) {
    int m_start = expert_offsets_host[e];
    int m_end = expert_offsets_host[e + 1];
    int m_e = m_end - m_start;
    if (m_e <= 0) {
      continue;
    }
    const __nv_bfloat16* A_e = A + (size_t)m_start * k;

    int rc = metrale_cutlass_nvfp4_gemm_bf16_act_weight_t(
        A_e,
        reinterpret_cast<const void*>(gate_packed_ptrs[e]),
        reinterpret_cast<const void*>(gate_scale_ptrs[e]),
        gate_scale2_vals[e],
        Cg + (size_t)m_start * n,
        m_e, n, k, workspace, workspace_size, stream);
    if (rc != 0) {
      return 100000 + rc;
    }

    rc = metrale_cutlass_nvfp4_gemm_bf16_act_weight_t(
        A_e,
        reinterpret_cast<const void*>(up_packed_ptrs[e]),
        reinterpret_cast<const void*>(up_scale_ptrs[e]),
        up_scale2_vals[e],
        Cu + (size_t)m_start * n,
        m_e, n, k, workspace, workspace_size, stream);
    if (rc != 0) {
      return 200000 + rc;
    }
  }
  return 0;
#else
  (void)A_bf16; (void)gate_packed_ptrs; (void)gate_scale_ptrs; (void)gate_scale2_vals;
  (void)up_packed_ptrs; (void)up_scale_ptrs; (void)up_scale2_vals;
  (void)C_gate_bf16; (void)C_up_bf16; (void)expert_offsets_host;
  (void)num_experts; (void)n; (void)k; (void)workspace; (void)workspace_size; (void)stream;
  return -120;
#endif
}

extern "C" int metrale_cutlass_pack_bf16_weight_to_nvfp4_t(
    const void* weight_bf16,
    void* packed_t,
    void* scale_t,
    int n,
    int k,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (n <= 0 || k <= 0 || (k % 16) != 0) {
    return -1;
  }
  dim3 block(256);
  dim3 grid(n, (k / 16 + block.x - 1) / block.x);
  metrale_cutlass_pack_bf16_weight_nvfp4_t<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(weight_bf16),
      static_cast<unsigned char*>(packed_t),
      static_cast<unsigned char*>(scale_t),
      n,
      k);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : -static_cast<int>(err);
#else
  (void)weight_bf16;
  (void)packed_t;
  (void)scale_t;
  (void)n;
  (void)k;
  (void)stream;
  return -120;
#endif
}

// =============================================================================================
// 2026-10-08: METRALE_GLM_PREFILL_DENSE_W4A4 (GLM-5.3 dense prefill projections): NVFP4
// activation x NVFP4 weight -> BF16 on the dense GEMM above (the CUTLASS example 79a
// configuration: 128x128x128 tile, 1x1x1 cluster, KernelScheduleAuto), with
// - the activation quantized per call by w4a4_dense_pack_act_k with a caller-given STATIC
//   global scale gs (quant16_gs from cutlass_nvfp4_w4a4_quant.cuh, the routed-MoE W4A4
//   quantizer: sf = UE4M3(min(amax16 / 6 / gs, 448)), codes E2M1_rne(v / (sf * gs))), so a row's
//   codes and scale bytes depend only on that row;
// - the weight as the GLM dense NVFP4 copy: packed E2M1 [N, K/2] (low nibble first, the
//   ColumnMajor-B byte layout), E4M3 scales [N, K/16] row-major, swizzled per call into the
//   workspace (w4a4_dense_pack_sfb_k, the tiled SFB pack of the MoE path) unless the caller
//   passes an already swizzled SFB;
// - epilogue D = alpha * acc (alpha = weight scale2 * gs), beta 0.
// Status: 0 ok. Nothing launched (the caller may run another path): -1 shape (m <= 0, n or k
// not a positive multiple of 128), -2 workspace too small, -3 can_implement refused, -4 a
// misaligned pointer (act, weight, out, sfb 16 B; workspace 256 B) or no scale source, -5 gs
// not > 0 / alpha not finite. -6: a failure before the first GEMM launch (only the workspace
// was written; the output is untouched). -7 / -8: a later initialize / GEMM launch failed (the
// output may be partly written). -120: built without SM120/121 support.

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

#include <cmath>

#include "cutlass_nvfp4_w4a4_quant.cuh"

// 2026-10-08: One thread per (row, 16-value group) of the [m, k] BF16 activation, flat
// row-major index: two 16-byte loads (act 16-byte aligned, k % 128 == 0), quant16_gs on the
// 16 values, 8 code bytes stored at packed[row * k/2 + group * 8] and the scale byte at
// layout_sfa(row, group * 16). Grid ceil(m * k/16 / 256), 256 threads.
template <class LayoutSFA_t>
__global__ void w4a4_dense_pack_act_k(
    const __nv_bfloat16* __restrict__ act,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ sfa,
    int m,
    int k,
    float gs,
    LayoutSFA_t layout_sfa) {
  const int groups = k / 16;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)m * groups) {
    return;
  }
  const int row = (int)(idx / groups);
  const int group = (int)(idx - (long long)row * groups);
  const uint4* src =
      reinterpret_cast<const uint4*>(act + (unsigned long long)row * k + group * 16);
  __align__(16) __nv_bfloat16 v[16];
  *reinterpret_cast<uint4*>(&v[0]) = src[0];
  *reinterpret_cast<uint4*>(&v[8]) = src[1];
  unsigned char c[8];
  unsigned char sf;
  quant16_gs(v, 0, gs, c, &sf);
  sfa[layout_sfa(row, group * 16, 0)] = sf;
  uint2 w;
  w.x = (unsigned int)c[0] | ((unsigned int)c[1] << 8) | ((unsigned int)c[2] << 16) |
        ((unsigned int)c[3] << 24);
  w.y = (unsigned int)c[4] | ((unsigned int)c[5] << 8) | ((unsigned int)c[6] << 16) |
        ((unsigned int)c[7] << 24);
  *reinterpret_cast<uint2*>(packed + (unsigned long long)row * (k / 2) + group * 8) = w;
}

// 2026-10-08: The SFB swizzle of one weight's [N, K/16] E4M3 scales (n % 128 == 0,
// k % 64 == 0): one warp per 128 x 64 tile (sfb_tile_words, the body of the MoE path's
// pack_weight_sfb_batched_tiled_k), one 16-byte store per lane. Grid ceil(tiles / 8), 256
// threads; out 16-byte aligned.
template <class LayoutSFB_t>
__global__ void w4a4_dense_pack_sfb_k(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ out,
    int n,
    int k,
    LayoutSFB_t layout_sfb) {
  const int groups = k / 16;
  const int k_tiles = k / 64;
  const long long tiles = (long long)(n / 128) * k_tiles;
  const long long tile = (long long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (tile >= tiles) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const int nb = (int)(tile / k_tiles);
  const int kb = (int)(tile % k_tiles);
  const bool aligned =
      ((reinterpret_cast<unsigned long long>(src) | (unsigned long long)groups) & 3ull) == 0;
  unsigned int words[4];
  sfb_tile_words(src, groups, nb, kb, lane, aligned, words);
  unsigned char* dst = out + layout_sfb(nb * 128, kb * 64, 0) + (unsigned long long)lane * 16;
  *reinterpret_cast<uint4*>(dst) = make_uint4(words[0], words[1], words[2], words[3]);
}

// 2026-10-08: Whether `p` is not a multiple of `a` (a power of two).
static bool w4a4_misaligned(const void* p, unsigned long long a) {
  return (reinterpret_cast<unsigned long long>(p) & (a - 1)) != 0;
}

// 2026-10-08: The dense GEMM's SFB layout for an [n, k] weight (it does not depend on M).
static auto w4a4_layout_sfb(int n, int k) {
  return CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(1, n, k, 1));
}

// 2026-10-08: Bytes of that SFB layout.
static size_t w4a4_sfb_bytes(int n, int k) {
  return static_cast<size_t>(size(filter_zeros(w4a4_layout_sfb(n, k))));
}

// 2026-10-08: The GEMM arguments of one launch over `rows` rows.
static typename Gemm::Arguments w4a4_args(
    int rows, int n, int k, const unsigned char* a, const unsigned char* sfa,
    const unsigned char* sfb, const void* w_packed, float alpha, void* out) {
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {rows, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {rows, n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {rows, n, 1});
  auto layout_sfa = CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(rows, n, k, 1));
  auto layout_sfb = CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(rows, n, k, 1));
  return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {rows, n, k, 1},
      {
          reinterpret_cast<ElementA::DataType const*>(a),
          stride_a,
          reinterpret_cast<ElementB::DataType const*>(w_packed),
          stride_b,
          reinterpret_cast<ElementA::ScaleFactorType const*>(sfa),
          layout_sfa,
          reinterpret_cast<ElementB::ScaleFactorType const*>(sfb),
          layout_sfb,
      },
      {
          {alpha, 0.0f},
          reinterpret_cast<ElementC const*>(out),
          stride_c,
          reinterpret_cast<ElementD*>(out),
          stride_d,
      }};
}

// 2026-10-08: Workspace offsets of one launch of `rows` rows after `base` bytes (the SFB):
// packed A, SFA, then the CUTLASS GEMM workspace (its size depends on the arguments).
struct W4a4Ws {
  size_t a_off, sfa_off, gemm_off;
};

static W4a4Ws w4a4_ws(size_t base, int rows, int n, int k) {
  auto layout_sfa = CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(rows, n, k, 1));
  W4a4Ws w;
  w.a_off = base;
  w.sfa_off = align_up(w.a_off + static_cast<size_t>(rows) * (k / 2), 256);
  w.gemm_off = align_up(w.sfa_off + static_cast<size_t>(size(filter_zeros(layout_sfa))), 256);
  return w;
}

#endif

// 2026-10-08: Bytes of the swizzled SFB of an [n, k] weight (what
// metrale_cutlass_nvfp4_dense_w4a4_pack_sfb writes); 0 for a shape the entry refuses (n, k not
// positive multiples of 128) or a build without SM120/121 support.
extern "C" unsigned long long metrale_cutlass_nvfp4_dense_w4a4_sfb_bytes(int n, int k) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (n <= 0 || k <= 0 || (n % 128) != 0 || (k % 128) != 0) {
    return 0;
  }
  return static_cast<unsigned long long>(w4a4_sfb_bytes(n, k));
#else
  (void)n;
  (void)k;
  return 0;
#endif
}

// 2026-10-08: Swizzle one weight's [n, k/16] E4M3 scales into `out` (16-byte aligned, at least
// metrale_cutlass_nvfp4_dense_w4a4_sfb_bytes(n, k) bytes) on `stream`: the SFB the dense W4A4
// GEMM reads. -1 bad shape / pointer, -cudaError on a launch failure, -120 without SM120/121.
extern "C" int metrale_cutlass_nvfp4_dense_w4a4_pack_sfb(
    const void* scale_nk, void* out, int n, int k, cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (scale_nk == nullptr || out == nullptr || w4a4_misaligned(out, 16) || n <= 0 || k <= 0 ||
      (n % 128) != 0 || (k % 128) != 0) {
    return -1;
  }
  const long long tiles = (long long)(n / 128) * (k / 64);
  w4a4_dense_pack_sfb_k<<<(unsigned int)((tiles + 7) / 8), 256, 0, stream>>>(
      static_cast<const unsigned char*>(scale_nk), static_cast<unsigned char*>(out), n, k,
      w4a4_layout_sfb(n, k));
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : -static_cast<int>(err);
#else
  (void)scale_nk;
  (void)out;
  (void)n;
  (void)k;
  (void)stream;
  return -120;
#endif
}

// 2026-10-08: out[m, n] (BF16, row stride n) = alpha * nvfp4(act[m, k]; gs) @ W[n, k]^T (see the
// section comment for the operands and the status codes). `w_sfb` null: the scales at
// `w_scale_nk` are swizzled into the workspace once per call; else `w_sfb` is the swizzled SFB
// and `w_scale_nk` is unused. rows_per_launch > 0 runs the rows in launches of at most that
// many rows (each launch quantizes its own rows, then runs its GEMM; the workspace is reused
// in stream order); <= 0 is one launch. No allocation, host sync or host copy: capture-safe.
extern "C" int metrale_cutlass_nvfp4_dense_w4a4_gemm(
    const void* act_bf16,
    const void* w_packed,
    const void* w_scale_nk,
    const void* w_sfb,
    float alpha,
    float act_gs,
    void* out_bf16,
    int m,
    int n,
    int k,
    int rows_per_launch,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (m <= 0 || n <= 0 || k <= 0 || (n % 128) != 0 || (k % 128) != 0) {
    return -1;
  }
  if (act_bf16 == nullptr || w_packed == nullptr || out_bf16 == nullptr ||
      workspace == nullptr || (w_sfb == nullptr && w_scale_nk == nullptr) ||
      w4a4_misaligned(act_bf16, 16) || w4a4_misaligned(w_packed, 16) ||
      w4a4_misaligned(out_bf16, 16) || (w_sfb != nullptr && w4a4_misaligned(w_sfb, 16)) ||
      w4a4_misaligned(workspace, 256)) {
    return -4;
  }
  if (!(act_gs > 0.0f) || !std::isfinite(act_gs) || !std::isfinite(alpha)) {
    return -5;
  }
  unsigned char* ws = static_cast<unsigned char*>(workspace);
  const size_t sfb_ws = w_sfb == nullptr ? align_up(w4a4_sfb_bytes(n, k), 256) : 0;
  const unsigned char* sfb =
      w_sfb != nullptr ? static_cast<const unsigned char*>(w_sfb) : ws;
  const int step = (rows_per_launch > 0 && rows_per_launch < m) ? rows_per_launch : m;
  const int last = m - ((m - 1) / step) * step;

  // Everything that can refuse is checked before anything launches.
  Gemm gemm;
  for (int rows : {step, last}) {
    const W4a4Ws w = w4a4_ws(sfb_ws, rows, n, k);
    auto args = w4a4_args(rows, n, k, ws + w.a_off, ws + w.sfa_off, sfb, w_packed, alpha,
                          out_bf16);
    if (w.gemm_off + Gemm::get_workspace_size(args) > workspace_size) {
      return -2;
    }
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
      return -3;
    }
  }

  if (w_sfb == nullptr) {
    const long long tiles = (long long)(n / 128) * (k / 64);
    w4a4_dense_pack_sfb_k<<<(unsigned int)((tiles + 7) / 8), 256, 0, stream>>>(
        static_cast<const unsigned char*>(w_scale_nk), ws, n, k, w4a4_layout_sfb(n, k));
    if (cudaGetLastError() != cudaSuccess) {
      return -6;
    }
  }
  for (int r0 = 0; r0 < m; r0 += step) {
    const int rows = (m - r0 < step) ? (m - r0) : step;
    const W4a4Ws w = w4a4_ws(sfb_ws, rows, n, k);
    auto layout_sfa = CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(rows, n, k, 1));
    const long long items = (long long)rows * (k / 16);
    w4a4_dense_pack_act_k<<<(unsigned int)((items + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(act_bf16) + (unsigned long long)r0 * k,
        ws + w.a_off, ws + w.sfa_off, rows, k, act_gs, layout_sfa);
    if (cudaGetLastError() != cudaSuccess) {
      return r0 == 0 ? -6 : -7;
    }
    void* out = static_cast<__nv_bfloat16*>(out_bf16) + (unsigned long long)r0 * n;
    auto args = w4a4_args(rows, n, k, ws + w.a_off, ws + w.sfa_off, sfb, w_packed, alpha, out);
    if (gemm.initialize(args, ws + w.gemm_off, stream) != cutlass::Status::kSuccess) {
      return r0 == 0 ? -6 : -7;
    }
    if (gemm.run(stream) != cutlass::Status::kSuccess) {
      return -8;
    }
  }
  return 0;
#else
  (void)act_bf16;
  (void)w_packed;
  (void)w_scale_nk;
  (void)w_sfb;
  (void)alpha;
  (void)act_gs;
  (void)out_bf16;
  (void)m;
  (void)n;
  (void)k;
  (void)rows_per_launch;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  return -120;
#endif
}
