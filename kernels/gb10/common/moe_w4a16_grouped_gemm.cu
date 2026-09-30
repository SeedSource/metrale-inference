// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Grouped W4A16 GEMM kernels for MoE, all experts in one launch: BF16 activations times NVFP4 weights (E2M1
// nibbles, low nibble first, with one FP8 E4M3 scale per GROUP_SIZE values of K and a per-tensor or per-expert
// scale2), dequantised to BF16 in shared memory and multiplied by mma m16n8k16 into F32 accumulators.
//
// Expert e owns output rows expert_offsets[e] .. expert_offsets[e + 1]. Grid (ceil(N / tile N), max_m_tiles,
// num_experts): blockIdx.x is the N tile, blockIdx.y the M tile within the expert, blockIdx.z the expert.
//
// Entry points: `moe_w4a16_grouped_gemm` (all experts' weights in one buffer); `_ptrtable` (per-expert pointer
// tables, A rows gathered through sorted_token_ids); `_ptrtable_t` (the same with weights stored [K/2, N]); the
// `_ptrtable_<suffix>` tile variants from P3B_GROUPED_VARIANT; `moe_w4a16_grouped_stream_probe`; and (bench-only,
// GLM_TILE_BENCH_DIAG=1) the `_ptrtable_bt_m128_k64_diag_{nodeq,nomma,noload}` pipeline-stage isolators from
// P3B_GROUPED_VARIANT_DIAG.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.







#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_MOE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void moe_w4a16_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,        // 2026-09-25: [total_tokens, K], rows in expert order
    const unsigned char* __restrict__ B_packed,  // 2026-09-25: [num_experts, N, K/2] E2M1
    const unsigned char* __restrict__ B_scale,   // 2026-09-25: [num_experts, N, K/GROUP_SIZE] E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // 2026-09-25: [total_tokens, N]
    const int* __restrict__ expert_offsets,       // 2026-09-25: [num_experts + 1] prefix sum
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {

    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;


    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;


    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;


    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE;

    // 2026-09-25: Each expert's weights are N-major: B [N, K/2], scales [N, K/GROUP_SIZE].
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int weight_stride_packed = N * half_K;
    const unsigned int scale_stride = N * num_groups;
    const unsigned char* B_expert = B_packed + expert_id * weight_stride_packed;
    const unsigned char* S_expert = B_scale + expert_id * scale_stride;


    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;


    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];


    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE + PAD;


    const unsigned int M_eff = (unsigned int)M_expert;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {

        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;

                bool valid = (cta_m_local + row) < M_eff && gc < K;
                smem_A[row][col] = valid ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }


        {
            const unsigned int elems_per_thread = (K_STEP * N_TILE) / 128;
            unsigned int scale_group = k_base / GROUP_SIZE;

            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    // 2026-09-25: B_packed[gn, gk / 2]; an odd gk is the high nibble.
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_expert[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);


                    unsigned char scale_byte = S_expert[(unsigned long long)gn * num_groups + scale_group];
                    float fp8_val;
                    {
                        __nv_fp8_e4m3 fp8;
                        *(unsigned char*)&fp8 = scale_byte;
                        fp8_val = (float)fp8;
                    }

                    float dequant_val = E2M1_LUT_MOE[nibble] * fp8_val * scale2;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();


        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;

        unsigned int frag_r0 = warp_m_offset + group_id;
        unsigned int frag_r1 = warp_m_offset + group_id + 8;
        unsigned int frag_c0 = tid * 2;
        unsigned int frag_c1 = tid * 2 + 8;

        unsigned int a0 = ((unsigned int)sA[frag_r0 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = ((unsigned int)sA[frag_r1 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = ((unsigned int)sA[frag_r0 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = ((unsigned int)sA[frag_r1 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = tid * 2;
            unsigned int k1 = tid * 2 + 8;

            unsigned int b0 = ((unsigned int)sB[(k0 + 1) * b_stride + n_col] << 16) |
                              (unsigned int)sB[k0 * b_stride + n_col];
            unsigned int b1 = ((unsigned int)sB[(k1 + 1) * b_stride + n_col] << 16) |
                              (unsigned int)sB[k1 * b_stride + n_col];

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(acc[n_tile][0]), "=f"(acc[n_tile][1]),
                  "=f"(acc[n_tile][2]), "=f"(acc[n_tile][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(acc[n_tile][0]), "f"(acc[n_tile][1]),
                  "f"(acc[n_tile][2]), "f"(acc[n_tile][3])
            );
        }

        __syncthreads();
    }


    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        bool row0_valid = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool row1_valid = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;

        if (row0_valid && col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[n_tile][0]);
        if (row0_valid && col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[n_tile][1]);
        if (row1_valid && col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[n_tile][2]);
        if (row1_valid && col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[n_tile][3]);
    }
}

// 2026-09-25: `moe_w4a16_grouped_gemm_ptrtable`: per-expert weight, scale and scale2 pointers from device tables,
// with A rows gathered through sorted_token_ids, or taken in order when it is NULL. An expert whose weight
// pointer is NULL returns without writing its rows. Block 128.








extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable(
    const __nv_bfloat16* __restrict__ A,           // 2026-09-25: [num_tokens, K], rows read through sorted_token_ids
    const unsigned long long* __restrict__ B_packed_ptrs, // 2026-09-25: [num_experts] -> [N, K/2] E2M1
    const unsigned long long* __restrict__ B_scale_ptrs,  // 2026-09-25: [num_experts] -> [N, K/GROUP_SIZE] E4M3
    const float* __restrict__ scale2_vals,         // 2026-09-25: [num_experts]
    __nv_bfloat16* __restrict__ C,                  // 2026-09-25: [total_expanded, N]
    const int* __restrict__ expert_offsets,          // 2026-09-25: [num_experts + 1] prefix sum
    const int* __restrict__ sorted_token_ids,       // 2026-09-25: [total_expanded] -> row of A, or NULL
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE;


    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];


    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE + PAD;
    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {

        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gc = k_base + col;
                bool valid = (cta_m_local + row) < M_eff && gc < K;
                if (valid) {
                    unsigned int a_row = sorted_token_ids
                        ? (unsigned int)sorted_token_ids[cta_m + row]
                        : (cta_m + row);
                    smem_A[row][col] = A[a_row * K + gc];
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }


        {
            const unsigned int elems_per_thread = (K_STEP * N_TILE) / 128;
            unsigned int scale_group = k_base / GROUP_SIZE;

            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    // 2026-09-25: B_packed[gn, gk / 2]; an odd gk is the high nibble.
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_expert[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);


                    unsigned char scale_byte = S_expert[(unsigned long long)gn * num_groups + scale_group];
                    float fp8_val;
                    {
                        __nv_fp8_e4m3 fp8;
                        *(unsigned char*)&fp8 = scale_byte;
                        fp8_val = (float)fp8;
                    }

                    float dequant_val = E2M1_LUT_MOE[nibble] * fp8_val * scale2;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();


        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;

        unsigned int frag_r0 = warp_m_offset + group_id;
        unsigned int frag_r1 = warp_m_offset + group_id + 8;
        unsigned int frag_c0 = tid * 2;
        unsigned int frag_c1 = tid * 2 + 8;

        unsigned int a0 = ((unsigned int)sA[frag_r0 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = ((unsigned int)sA[frag_r1 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = ((unsigned int)sA[frag_r0 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = ((unsigned int)sA[frag_r1 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = tid * 2;
            unsigned int k1 = tid * 2 + 8;

            unsigned int b0 = ((unsigned int)sB[(k0 + 1) * b_stride + n_col] << 16) |
                              (unsigned int)sB[k0 * b_stride + n_col];
            unsigned int b1 = ((unsigned int)sB[(k1 + 1) * b_stride + n_col] << 16) |
                              (unsigned int)sB[k1 * b_stride + n_col];

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(acc[n_tile][0]), "=f"(acc[n_tile][1]),
                  "=f"(acc[n_tile][2]), "=f"(acc[n_tile][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(acc[n_tile][0]), "f"(acc[n_tile][1]),
                  "f"(acc[n_tile][2]), "f"(acc[n_tile][3])
            );
        }

        __syncthreads();
    }


    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;
        bool row0_valid = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool row1_valid = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;

        if (row0_valid && col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[n_tile][0]);
        if (row0_valid && col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[n_tile][1]);
        if (row1_valid && col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[n_tile][2]);
        if (row1_valid && col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[n_tile][3]);
    }
}

// 2026-09-25: `moe_w4a16_grouped_gemm_ptrtable_t`: `_ptrtable` with each expert's weights stored [K/2, N] and its
// scales [K/GROUP_SIZE, N], so threads with adjacent n read adjacent bytes.





extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE + PAD;
    const unsigned int M_eff = (unsigned int)M_expert;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gc = k_base + col;
                bool valid = (cta_m_local + row) < M_eff && gc < K;
                if (valid) {
                    unsigned int a_row = sorted_token_ids
                        ? (unsigned int)sorted_token_ids[cta_m + row]
                        : (cta_m + row);
                    smem_A[row][col] = A[a_row * K + gc];
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }


        {
            const unsigned int elems_per_thread = (K_STEP * N_TILE) / 128;
            unsigned int scale_group = k_base / GROUP_SIZE;

            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_expert[(unsigned long long)k_pair * N + gn];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);

                    unsigned char scale_byte = S_expert[(unsigned long long)scale_group * N + gn];
                    float fp8_val;
                    {
                        __nv_fp8_e4m3 fp8;
                        *(unsigned char*)&fp8 = scale_byte;
                        fp8_val = (float)fp8;
                    }

                    float dequant_val = E2M1_LUT_MOE[nibble] * fp8_val * scale2;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;

        unsigned int frag_r0 = warp_m_offset + group_id;
        unsigned int frag_r1 = warp_m_offset + group_id + 8;
        unsigned int frag_c0 = tid * 2;
        unsigned int frag_c1 = tid * 2 + 8;

        unsigned int a0 = ((unsigned int)sA[frag_r0 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = ((unsigned int)sA[frag_r1 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = ((unsigned int)sA[frag_r0 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = ((unsigned int)sA[frag_r1 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = tid * 2;
            unsigned int k1 = tid * 2 + 8;

            unsigned int b0 = ((unsigned int)sB[(k0 + 1) * b_stride + n_col] << 16) |
                              (unsigned int)sB[k0 * b_stride + n_col];
            unsigned int b1 = ((unsigned int)sB[(k1 + 1) * b_stride + n_col] << 16) |
                              (unsigned int)sB[k1 * b_stride + n_col];

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(acc[n_tile][0]), "=f"(acc[n_tile][1]),
                  "=f"(acc[n_tile][2]), "=f"(acc[n_tile][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(acc[n_tile][0]), "f"(acc[n_tile][1]),
                  "f"(acc[n_tile][2]), "f"(acc[n_tile][3])
            );
        }

        __syncthreads();
    }

    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;
        bool row0_valid = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool row1_valid = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;

        if (row0_valid && col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[n_tile][0]);
        if (row0_valid && col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[n_tile][1]);
        if (row1_valid && col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[n_tile][2]);
        if (row1_valid && col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[n_tile][3]);
    }
}

// 2026-09-25: Tile-geometry variants of `moe_w4a16_grouped_gemm_ptrtable`: one entry point per P3B_GROUPED_VARIANT
// line below, each a `moe_w4a16_grouped_core` instance. They compute the same product with these changes:
//   - One thread stages a whole GROUP_SIZE-wide k run for one n: its 8 packed bytes as two uint32 loads and
//     its scale byte once.
//   - SPLIT_N = false: the warps split M (MT = WARPS * 16), and a warp whose 16 rows all lie past the expert's
//     rows skips its MMAs. SPLIT_N = true: MT = 16 and the warps split N.
//   - KS values of K are staged per shared-memory round trip; the MMAs take them 16 at a time, ascending.
//   - KMAJOR: consecutive threads take consecutive k groups of one n instead of consecutive n.
//   - ARITH_LUT: E2M1 is decoded by e2m1_decode instead of read from E2M1_LUT_MOE.
//   - BT: the staged B tile is [NTILE][KS + 8] instead of [KS][NTILE + PAD].
// A staged weight here is lut * (e4m3 * scale2), where the base kernels compute (lut * e4m3) * scale2. The
// two can round differently, so a variant's output is not guaranteed to match the base kernel's bit for bit.
//
// A variant's grid counts M tiles in its own MT and N tiles in its own NTILE. A grid computed for a larger M
// tile drops rows, and one computed for a larger N tile leaves columns unwritten.












































// 2026-09-25: Template parameters of `moe_w4a16_grouped_core`:
//   MT       rows per CTA tile: WARPS * 16 when the warps split M, 16 when they split N.
//   NTILE    output columns per CTA; grid x counts these.
//   KS       K staged per shared-memory round trip, a multiple of GROUP_SIZE. MT * KS and
//            (KS / GROUP_SIZE) * NTILE must be multiples of the thread count (static_asserts).
//   WARPS    warps per CTA; the block is WARPS * 32 threads.
//   SPLIT_N  false: each warp takes 16 rows and the whole N tile; true: every warp takes the same 16 rows
//            and NTILE / WARPS columns.
//   KMAJOR, ARITH_LUT, BT: see the variants comment above.
//
// e2m1_decode: the E2M1 value of nibble n built from bits, the same float as E2M1_LUT_MOE[n] for all 16 codes.
// idx = n & 7 gives {0, 0.5, 1, 1.5, 2, 3, 4, 6}: exponent 126 + (idx >> 1), mantissa bit 22 for an odd
// idx >= 2, and all-zero bits for idx 0. Bit 3 of n is the sign, so nibble 8 gives -0.0.









__device__ __forceinline__ float e2m1_decode(unsigned int n) {
    unsigned int idx = n & 7u;
    unsigned int mant = (idx >= 2u) ? ((idx & 1u) << 22) : 0u;
    unsigned int bits = (idx == 0u) ? 0u : (((126u + (idx >> 1)) << 23) | mant);
    bits |= (n & 8u) << 28;
    return __int_as_float(bits);
}

// 2026-09-30: DIAG (bench-only; every production/default entry point instantiates DIAG = 0,
// byte-identical to before this parameter existed). Used only by the `_diag_*` entry points
// below `bt_m128_k64` (examples/glm5next_moe_grouped_tile_bench, GLM_TILE_BENCH_DIAG=1):
//   0  normal.
//   1  NODEQ: keeps the two weight-byte loads (w0, w1) and the scale-byte load, skips the
//      E2M1 LUT/arith decode and the two multiplies, and stores a BF16 folded from the loaded
//      bytes (XOR) so the compiler cannot prove the loads dead.
//   2  NOMMA: keeps the loads, dequant, smem_B stores and both __syncthreads(), skips
//      mma.sync, and accumulates a value folded (XOR) from the same fragment registers the
//      mma would have consumed, so they cannot be proven dead.
//   3  NOLOAD-W: keeps dequant and mma unchanged, but every (gk, gn) reads the same ~8 weight
//      bytes and 1 scale byte from a tiny CTA-shared cache loaded once at CTA entry, instead
//      of walking B_expert/S_expert in DRAM — negligible DRAM weight traffic per CTA.
template <int MT, int NTILE, int KS, int WARPS, bool SPLIT_N, bool KMAJOR, bool ARITH_LUT,
          bool BT, bool MFAST = false, int DIAG = 0>
__device__ __forceinline__ void moe_w4a16_grouped_core(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    constexpr int THREADS = WARPS * 32;
    constexpr int NT = SPLIT_N ? (NTILE / WARPS / 8) : (NTILE / 8);
    static_assert(SPLIT_N ? (MT == 16) : (MT == WARPS * 16), "MT must match the warp split");
    static_assert(NT >= 1, "each warp needs at least one 8-wide n subtile");
    static_assert((MT * KS) % THREADS == 0, "A tile must divide evenly across the block");
    static_assert(((KS / GROUP_SIZE) * NTILE) % THREADS == 0,
                  "B scale-groups must divide evenly across the block");
    static_assert(KS % GROUP_SIZE == 0, "KS must be a whole number of scale groups");

    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    // 2026-09-29: MFAST swaps grid x and y: blockIdx.x is the M tile and blockIdx.y the N tile, so
    // the CTAs of one (expert, N tile), which read the same weight columns, are adjacent in launch
    // order. Only the block-to-tile mapping changes; each CTA computes exactly what it would
    // with MFAST = false.
    const unsigned int m_blk = MFAST ? blockIdx.x : blockIdx.y;
    const unsigned int n_blk = MFAST ? blockIdx.y : blockIdx.x;
    const int cta_m_local = (int)m_blk * MT;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = n_blk * NTILE;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    // 2026-09-25: A NULL weight pointer returns without writing this expert's rows.
    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    const unsigned int warp_m_offset = SPLIT_N ? 0u : warp_id * 16u;
    const unsigned int warp_n_offset = SPLIT_N ? warp_id * (NTILE / WARPS) : 0u;

    // 2026-09-25: The staged B tile. BT = false: [KS][NTILE + PAD], as in the base kernel. BT = true: [NTILE][KS + 8], so
    // one thread's 16 values for one n are contiguous and each (k, k + 1) MMA fragment pair is one aligned 32-bit
    // read. (KS + 8) * 2 is a multiple of 16 for every KS the BT variants use (64, 128, 256), so each row stays
    // 16-byte aligned.








    constexpr int B_STRIDE = BT ? (KS + 8) : (NTILE + PAD);
    constexpr int B_ELEMS = BT ? (NTILE * B_STRIDE) : (KS * B_STRIDE);
    __shared__ __nv_bfloat16 smem_A[MT][KS + PAD];
    __shared__ __align__(16) __nv_bfloat16 smem_B[B_ELEMS];

    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = KS + PAD;
    const unsigned int b_stride = (unsigned int)B_STRIDE;
    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

    // 2026-09-25: A warp whose 16 rows all lie past M_expert skips its MMAs and its stores; it still takes part in
    // the cooperative loads and every barrier.


    const bool warp_has_rows = (cta_m_local + (int)warp_m_offset) < M_expert;

    // 2026-09-30: DIAG == 3 (NOLOAD-W) only. One thread reads the expert's first 8 weight
    // bytes and first scale byte into a CTA-shared cache; every (gk, gn) below then decodes
    // from this cache instead of B_expert/S_expert, so the CTA's DRAM weight traffic is this
    // one small read, not the full N * K/2 + N * K/GROUP_SIZE sweep.
    __shared__ unsigned int diag_w0, diag_w1;
    __shared__ unsigned char diag_scale;
    if (DIAG == 3) {
        if (threadIdx.x == 0) {
            diag_w0 = *(const unsigned int*)(B_expert);
            diag_w1 = *(const unsigned int*)(B_expert + 4);
            diag_scale = S_expert[0];
        }
        __syncthreads();
    }

    for (unsigned int k_base = 0; k_base < K; k_base += KS) {

        {
            constexpr unsigned int ept = (unsigned int)((MT * KS) / THREADS);
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int row = idx / KS;
                unsigned int col = idx % KS;
                unsigned int gc = k_base + col;
                bool valid = (cta_m_local + row) < M_eff && gc < K;
                if (valid) {
                    unsigned int a_row = sorted_token_ids
                        ? (unsigned int)sorted_token_ids[cta_m + row]
                        : (cta_m + row);
                    smem_A[row][col] = A[a_row * K + gc];
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }


        {
            constexpr unsigned int gpt =
                (unsigned int)(((KS / GROUP_SIZE) * NTILE) / THREADS);
            #pragma unroll
            for (unsigned int i = 0; i < gpt; i++) {
                unsigned int g = threadIdx.x * gpt + i;
                // 2026-09-25: KMAJOR = false: consecutive threads take consecutive n, whose packed bytes are K / 2 apart.
                // KMAJOR = true: consecutive threads take consecutive k groups of the same n, so adjacent lanes read
                // adjacent bytes of one weight row. The staged values and their slots are the same either way.






                constexpr unsigned int KG = (unsigned int)(KS / GROUP_SIZE);
                unsigned int kg = KMAJOR ? ((g % KG) * GROUP_SIZE) : ((g / NTILE) * GROUP_SIZE);
                unsigned int n  = KMAJOR ? (g / KG) : (g % NTILE);
                unsigned int gk = k_base + kg;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    unsigned int w0, w1;
                    float sc;
                    if (DIAG == 3) {
                        // 2026-09-30: NOLOAD-W — every (gk, gn) reads the same CTA-cached
                        // bytes instead of B_expert/S_expert; no per-iteration DRAM weight
                        // traffic.
                        w0 = diag_w0;
                        w1 = diag_w1;
                        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = diag_scale;
                        sc = (float)fp8 * scale2;
                    } else {
                        const unsigned char* bp =
                            B_expert + (unsigned long long)gn * half_K + (gk / 2);
                        unsigned char sb =
                            S_expert[(unsigned long long)gn * num_groups + (gk / GROUP_SIZE)];
                        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
                        sc = (float)fp8 * scale2;
                        w0 = *(const unsigned int*)(bp);
                        w1 = *(const unsigned int*)(bp + 4);
                    }

                    __nv_bfloat16* st = smem_B + (unsigned int)(n * B_STRIDE) + kg;
                    if (DIAG == 1) {
                        // 2026-09-30: NODEQ — w0, w1 and sc above are still real loads (kept
                        // live by the XOR below); skip the LUT/arith decode and the two
                        // multiplies per element.
                        __nv_bfloat16 dv = __float2bfloat16(
                            (float)((w0 ^ w1 ^ __float_as_uint(sc)) & 0xFFu));
                        #pragma unroll
                        for (int j = 0; j < GROUP_SIZE; j++) {
                            if (BT) st[j] = dv;
                            else smem_B[(kg + j) * B_STRIDE + n] = dv;
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            unsigned char c0 = (unsigned char)((w0 >> (j * 8)) & 0xFF);
                            unsigned char c1 = (unsigned char)((w1 >> (j * 8)) & 0xFF);
                            float v0 = ARITH_LUT ? e2m1_decode(c0 & 0xF) : E2M1_LUT_MOE[c0 & 0xF];
                            float v1 = ARITH_LUT ? e2m1_decode(c0 >> 4)   : E2M1_LUT_MOE[c0 >> 4];
                            float v2 = ARITH_LUT ? e2m1_decode(c1 & 0xF) : E2M1_LUT_MOE[c1 & 0xF];
                            float v3 = ARITH_LUT ? e2m1_decode(c1 >> 4)   : E2M1_LUT_MOE[c1 >> 4];
                            if (BT) {
                                st[j * 2]         = __float2bfloat16(v0 * sc);
                                st[j * 2 + 1]     = __float2bfloat16(v1 * sc);
                                st[8 + j * 2]     = __float2bfloat16(v2 * sc);
                                st[8 + j * 2 + 1] = __float2bfloat16(v3 * sc);
                            } else {
                                smem_B[(kg + j * 2) * B_STRIDE + n]         = __float2bfloat16(v0 * sc);
                                smem_B[(kg + j * 2 + 1) * B_STRIDE + n]     = __float2bfloat16(v1 * sc);
                                smem_B[(kg + 8 + j * 2) * B_STRIDE + n]     = __float2bfloat16(v2 * sc);
                                smem_B[(kg + 8 + j * 2 + 1) * B_STRIDE + n] = __float2bfloat16(v3 * sc);
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < GROUP_SIZE; j++) {
                        if (BT) smem_B[(unsigned int)(n * B_STRIDE) + kg + j] = __float2bfloat16(0.0f);
                        else smem_B[(kg + j) * B_STRIDE + n] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        __syncthreads();

        if (warp_has_rows) {
            const unsigned short* sA = (const unsigned short*)smem_A;
            const unsigned short* sB = (const unsigned short*)smem_B;
            unsigned int fr0 = warp_m_offset + group_id;
            unsigned int fr1 = fr0 + 8;

            // 2026-09-25: The staged tile holds KS / 16 MMA fragments, consumed in ascending K.


            #pragma unroll
            for (unsigned int kf = 0; kf < (unsigned int)KS; kf += 16) {
                unsigned int fc0 = kf + tid * 2, fc1 = fc0 + 8;
                unsigned int a0 = ((unsigned int)sA[fr0 * a_stride + fc0 + 1] << 16) |
                                  (unsigned int)sA[fr0 * a_stride + fc0];
                unsigned int a1 = ((unsigned int)sA[fr1 * a_stride + fc0 + 1] << 16) |
                                  (unsigned int)sA[fr1 * a_stride + fc0];
                unsigned int a2 = ((unsigned int)sA[fr0 * a_stride + fc1 + 1] << 16) |
                                  (unsigned int)sA[fr0 * a_stride + fc1];
                unsigned int a3 = ((unsigned int)sA[fr1 * a_stride + fc1 + 1] << 16) |
                                  (unsigned int)sA[fr1 * a_stride + fc1];

                #pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    unsigned int nc = warp_n_offset + nt * 8 + group_id;
                    unsigned int k0 = kf + tid * 2, k1 = k0 + 8;
                    unsigned int b0, b1;
                    if (BT) {
                        // 2026-09-25: k and k + 1 are adjacent: one aligned 32-bit read each.
                        const unsigned int* r = (const unsigned int*)(sB + nc * b_stride);
                        b0 = r[k0 >> 1];
                        b1 = r[k1 >> 1];
                    } else {
                        b0 = ((unsigned int)sB[(k0 + 1) * b_stride + nc] << 16) |
                             (unsigned int)sB[k0 * b_stride + nc];
                        b1 = ((unsigned int)sB[(k1 + 1) * b_stride + nc] << 16) |
                             (unsigned int)sB[k1 * b_stride + nc];
                    }
                    if (DIAG == 2) {
                        // 2026-09-30: NOMMA — a0..a3, b0, b1 are the same fragment registers
                        // the mma.sync below would consume (the smem reads, stores and both
                        // barriers above are unchanged); fold them with XOR instead of issuing
                        // the mma, so the registers cannot be proven dead.
                        unsigned int mix = a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1;
                        float mixed = (float)(mix & 0xFFFFu) * 1.0e-6f;
                        acc[nt][0] += mixed;
                        acc[nt][1] += mixed;
                        acc[nt][2] += mixed;
                        acc[nt][3] += mixed;
                    } else {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                            : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                              "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
                    }
                }
            }
        }

        __syncthreads();
    }

    if (!warp_has_rows) return;

    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        unsigned int c0 = cta_n + warp_n_offset + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3]);
    }
}

#define P3B_GROUPED_VARIANT(SUFFIX, MT, NTILE, KS, WARPS, SPLIT_N, KMAJOR, ALUT, BT) \
extern "C" __global__ __launch_bounds__((WARPS) * 32)                         \
void moe_w4a16_grouped_gemm_ptrtable_##SUFFIX(                                \
    const __nv_bfloat16* __restrict__ A,                                      \
    const unsigned long long* __restrict__ B_packed_ptrs,                     \
    const unsigned long long* __restrict__ B_scale_ptrs,                      \
    const float* __restrict__ scale2_vals,                                    \
    __nv_bfloat16* __restrict__ C,                                            \
    const int* __restrict__ expert_offsets,                                   \
    const int* __restrict__ sorted_token_ids,                                 \
    unsigned int num_experts,                                                 \
    unsigned int N,                                                           \
    unsigned int K                                                            \
) {                                                                           \
    moe_w4a16_grouped_core<MT, NTILE, KS, WARPS, SPLIT_N, KMAJOR, ALUT, BT>(  \
        A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,                       \
        expert_offsets, sorted_token_ids, num_experts, N, K);                 \
}

// 2026-09-29: `P3B_GROUPED_VARIANT` with MFAST = true: the launch grid is
// (max_m_tiles, ceil(N / NTILE), num_experts), M tile in x. A grid in the default (N tile in x)
// order computes the wrong tiles.
#define P3B_GROUPED_VARIANT_MFAST(SUFFIX, MT, NTILE, KS, WARPS, SPLIT_N, KMAJOR, ALUT, BT) \
extern "C" __global__ __launch_bounds__((WARPS) * 32)                         \
void moe_w4a16_grouped_gemm_ptrtable_##SUFFIX(                                \
    const __nv_bfloat16* __restrict__ A,                                      \
    const unsigned long long* __restrict__ B_packed_ptrs,                     \
    const unsigned long long* __restrict__ B_scale_ptrs,                      \
    const float* __restrict__ scale2_vals,                                    \
    __nv_bfloat16* __restrict__ C,                                            \
    const int* __restrict__ expert_offsets,                                   \
    const int* __restrict__ sorted_token_ids,                                 \
    unsigned int num_experts,                                                 \
    unsigned int N,                                                           \
    unsigned int K                                                            \
) {                                                                           \
    moe_w4a16_grouped_core<MT, NTILE, KS, WARPS, SPLIT_N, KMAJOR, ALUT, BT, true>( \
        A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,                       \
        expert_offsets, sorted_token_ids, num_experts, N, K);                 \
}

// 2026-09-30: `P3B_GROUPED_VARIANT` with an explicit DIAG mode (1 NODEQ, 2 NOMMA, 3
// NOLOAD-W; see `moe_w4a16_grouped_core`'s DIAG doc comment). Bench-only
// (examples/glm5next_moe_grouped_tile_bench, GLM_TILE_BENCH_DIAG=1); never used in
// production. Same MFAST = false launch/grid convention as `P3B_GROUPED_VARIANT`.
#define P3B_GROUPED_VARIANT_DIAG(SUFFIX, MT, NTILE, KS, WARPS, SPLIT_N, KMAJOR, ALUT, BT, DIAGMODE) \
extern "C" __global__ __launch_bounds__((WARPS) * 32)                         \
void moe_w4a16_grouped_gemm_ptrtable_##SUFFIX(                                \
    const __nv_bfloat16* __restrict__ A,                                      \
    const unsigned long long* __restrict__ B_packed_ptrs,                     \
    const unsigned long long* __restrict__ B_scale_ptrs,                      \
    const float* __restrict__ scale2_vals,                                    \
    __nv_bfloat16* __restrict__ C,                                            \
    const int* __restrict__ expert_offsets,                                   \
    const int* __restrict__ sorted_token_ids,                                 \
    unsigned int num_experts,                                                 \
    unsigned int N,                                                           \
    unsigned int K                                                            \
) {                                                                           \
    moe_w4a16_grouped_core<MT, NTILE, KS, WARPS, SPLIT_N, KMAJOR, ALUT, BT, false, DIAGMODE>( \
        A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,                       \
        expert_offsets, sorted_token_ids, num_experts, N, K);                 \
}

// 2026-09-25: suffix, MT, NTILE, KS, WARPS, SPLIT_N, KMAJOR, ARITH_LUT, BT.
// MT 64 with 4 warps over M; only KS differs from the base kernel's K_STEP.
P3B_GROUPED_VARIANT(k32,            64,    64,  32,    4, false, false, false, false)
P3B_GROUPED_VARIANT(k64,            64,    64,  64,    4, false, false, false, false)
P3B_GROUPED_VARIANT(k128,           64,    64, 128,    4, false, false, false, false)
// 2026-09-25: MT 16, four warps splitting the 64-wide N tile; max_m_tiles counts 16-row tiles.
P3B_GROUPED_VARIANT(m16_k32,        16,    64,  32,    4, true,  false, false, false)
P3B_GROUPED_VARIANT(m16_k64,        16,    64,  64,    4, true,  false, false, false)
P3B_GROUPED_VARIANT(m16_k128,       16,    64, 128,    4, true,  false, false, false)
P3B_GROUPED_VARIANT(m16_k256,       16,    64, 256,    4, true,  false, false, false)
// 2026-09-25: MT 16, NTILE 128, eight warps; grid x counts 128-column tiles.
P3B_GROUPED_VARIANT(m16_n128_k32,   16,   128,  32,    8, true,  false, false, false)
P3B_GROUPED_VARIANT(m16_n128_k64,   16,   128,  64,    8, true,  false, false, false)
P3B_GROUPED_VARIANT(m16_n128_k128,  16,   128, 128,    8, true,  false, false, false)

P3B_GROUPED_VARIANT(km_k64,         64,    64,  64,    4, false, true, false, false)
P3B_GROUPED_VARIANT(km_m16_k32,     16,    64,  32,    4, true,  true, false, false)
P3B_GROUPED_VARIANT(km_m16_k64,     16,    64,  64,    4, true,  true, false, false)
P3B_GROUPED_VARIANT(km_m16_k128,    16,    64, 128,    4, true,  true, false, false)
P3B_GROUPED_VARIANT(km_m16_n128_k64, 16,  128,  64,    8, true,  true, false, false)
P3B_GROUPED_VARIANT(km_m16_n128_k128,16,  128, 128,    8, true,  true, false, false)

// 2026-09-25: `moe_w4a16_grouped_stream_probe`: reads each expert's B_packed and B_scale through the same pointer
// tables in coalesced uint4 loads, with no dequant, shared memory, barriers or MMA, as a bandwidth reference
// for the GEMMs (examples/glm5next_moe_grouped_tile_bench). The loads feed an XOR that is stored only if it
// is 0xFFFFFFFF, so the compiler keeps them. A trailing partial 16 bytes is not read.
// Launch: grid (any x, 1, num_experts), block 256; the blocks stride over the bytes.









extern "C" __global__ __launch_bounds__(256)
void moe_w4a16_grouped_stream_probe(
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    unsigned int* __restrict__ sink,
    unsigned int num_experts,
    unsigned int packed_bytes,   // 2026-09-25: N * K / 2
    unsigned int scale_bytes     // 2026-09-25: N * K / GROUP_SIZE
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;
    const uint4* bp = (const uint4*)B_packed_ptrs[expert_id];
    const uint4* bs = (const uint4*)B_scale_ptrs[expert_id];
    if (bp == 0) return;

    const unsigned int p_vec = packed_bytes / 16;
    const unsigned int s_vec = scale_bytes / 16;
    const unsigned int stride = gridDim.x * blockDim.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint4 acc = make_uint4(0, 0, 0, 0);
    for (unsigned int i = idx; i < p_vec; i += stride) {
        uint4 v = bp[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    for (unsigned int i = idx; i < s_vec; i += stride) {
        uint4 v = bs[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    unsigned int r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0xFFFFFFFFu) sink[expert_id] = r;  // 2026-09-25: keeps the loads live
}


P3B_GROUPED_VARIANT(al_k64,          64,    64,  64,    4, false, false, true, false)
P3B_GROUPED_VARIANT(al_m16_k32,      16,    64,  32,    4, true,  false, true, false)
P3B_GROUPED_VARIANT(al_m16_k64,      16,    64,  64,    4, true,  false, true, false)
P3B_GROUPED_VARIANT(al_m16_k128,     16,    64, 128,    4, true,  false, true, false)
P3B_GROUPED_VARIANT(al_m16_n128_k64, 16,   128,  64,    8, true,  false, true, false)
P3B_GROUPED_VARIANT(alkm_m16_k32,    16,    64,  32,    4, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_m16_k64,    16,    64,  64,    4, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_m16_k128,   16,    64, 128,    4, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_m16_k256,   16,    64, 256,    4, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_m16_n128_k64,  16, 128,  64,    8, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_m16_n128_k128, 16, 128, 128,    8, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_k64,        64,    64,  64,    4, false, true,  true, false)
P3B_GROUPED_VARIANT(alkm_k128,       64,    64, 128,    4, false, true,  true, false)


P3B_GROUPED_VARIANT(bt_m16_k64,      16,    64,  64,    4, true,  true,  true, true)
P3B_GROUPED_VARIANT(bt_m16_k128,     16,    64, 128,    4, true,  true,  true, true)
P3B_GROUPED_VARIANT(bt_m16_k256,     16,    64, 256,    4, true,  true,  true, true)
P3B_GROUPED_VARIANT(bt_m16_n128_k128, 16,  128, 128,    8, true,  true,  true, true)
P3B_GROUPED_VARIANT(bt_k128,         64,    64, 128,    4, false, true,  true, true)

// 2026-09-29: Whole-chunk prefill M1 (spark-bench .planning/METRALE-WHOLE-CHUNK-PREFILL-DESIGN-20260929.md).
// Every tile below stages the same BF16 A and B values as `bt_m16_k128` (same KMAJOR, ARITH_LUT
// and BT staging, so each B value is __float2bfloat16(e2m1_decode(nibble) * (e4m3 * scale2))) and
// feeds each output element's accumulator the same m16n8k16 MMAs in the same ascending
// 16-wide k order, whatever KS is: the K loop visits k_base = 0, KS, 2*KS, ... and inside it
// kf = 0, 16, ..., KS - 16. An element's result depends only on its A row, its B column and
// that order, so M tile, warp split, KS and grid order change which CTA or warp computes an
// element, not its bits (INFERRED; the tile bench and the prefill microtest assert it).
// `bt_k128` above is already the M64 tile (4 warps over M, KS 128). The M128 tile uses KS 64:
// at KS 128 its static shared memory would be 128 * 130 * 2 + 64 * 136 * 2 = 50,688 B, over
// the 48 KB static limit; at KS 64 it is 128 * 66 * 2 + 64 * 72 * 2 = 26,112 B.
P3B_GROUPED_VARIANT_MFAST(bt_m16_k128_mfast, 16, 64, 128,    4, true,  true,  true, true)
P3B_GROUPED_VARIANT_MFAST(bt_m64_k128_mfast, 64, 64, 128,    4, false, true,  true, true)
P3B_GROUPED_VARIANT(bt_m128_k64,            128, 64,  64,    8, false, true,  true, true)
P3B_GROUPED_VARIANT_MFAST(bt_m128_k64_mfast, 128, 64, 64,    8, false, true,  true, true)

// 2026-09-30: Bench-only diagnostic variants of `bt_m128_k64`, isolating one pipeline stage
// each (met-moeprobe, .planning residual-limiter classification). Never used in production;
// gated at the bench layer behind GLM_TILE_BENCH_DIAG=1
// (examples/glm5next_moe_grouped_tile_bench). Same MT/NTILE/KS/WARPS/grid/block/smem
// footprint as `bt_m128_k64`; only the DIAG mode differs.
P3B_GROUPED_VARIANT_DIAG(bt_m128_k64_diag_nodeq,  128, 64, 64, 8, false, true, true, true, 1)
P3B_GROUPED_VARIANT_DIAG(bt_m128_k64_diag_nomma,  128, 64, 64, 8, false, true, true, true, 2)
P3B_GROUPED_VARIANT_DIAG(bt_m128_k64_diag_noload, 128, 64, 64, 8, false, true, true, true, 3)
