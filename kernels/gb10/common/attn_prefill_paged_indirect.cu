// SPDX-License-Identifier: MIT OR Apache-2.0


// 2026-09-26: Kernels `attn_prefill_paged_indirect` and `_64`: attn_prefill_paged.cu with `kv_len`, `q_offset` and
// `q_rope_pos` read from device memory.
//
// KERNEL_PREAMBLE reads the three u32 values from the pointer arguments at kernel entry and overwrites the scalar
// `kv_len` and `q_offset` arguments with them. The DFlash draft head writes the values for each propose
// (dflash_head/forward_block/embed.rs) and launches the kernel eagerly, outside its CUDA graphs. In the body, `q_offset`
// caps the KV blocks scanned at the tile's last query row, and `q_rope_pos` is the absolute query position of the masks.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.



#include <cuda_bf16.h>

// 2026-09-25: Same loader as attn_prefill_paged.cu: one 32-row K or V tile by 16-byte cp.async, with rows at or past
// `kv_l` zero-filled.


#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned long long _ps = (unsigned long long)cache_block_size * num_kv_heads * head_dim; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const void* _gm = (const void*)( \
                    (cache) + _pb * _ps + _bo * _rs + (kvh) * head_dim + _col); \
                metrale_cp16(&(smem)[_row][_col], _gm); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

// 2026-10-10: Overridable so attn_prefill_paged_indirect_h128.cu can build this body under its own symbol (A93).
#ifndef KERNEL_NAME
#define KERNEL_NAME attn_prefill_paged_indirect
#endif
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d,                          \
                           const unsigned int* __restrict__ kv_len_ptr,        \
                           const unsigned int* __restrict__ q_offset_ptr,      \
                           const unsigned int* __restrict__ q_rope_pos_ptr
// 2026-09-25: KERNEL_PREAMBLE declares `q_rope_pos`, so prefill_paged_compute.cuh must skip its default
// `q_rope_pos = q_offset`.
#define Q_ROPE_POS_OVERRIDE
#define KERNEL_PREAMBLE                                                         \
                                                                                \
                                                                                \
    __shared__ unsigned int s_indirect[3];                                      \
    if (threadIdx.x == 0) {                                                     \
        s_indirect[0] = *kv_len_ptr;                                            \
        s_indirect[1] = *q_offset_ptr;                                          \
        s_indirect[2] = *q_rope_pos_ptr;                                        \
    }                                                                           \
    __syncthreads();                                                            \
    kv_len = s_indirect[0];                                                     \
    q_offset = s_indirect[1];                                                   \
    unsigned int q_rope_pos = s_indirect[2];

#include "prefill_paged_compute.cuh"
