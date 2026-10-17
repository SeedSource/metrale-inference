// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-17: The device reader of routed-expert block scales held as "scale rows with escape"
// (METRALE_GLM_MOE_CSF_SCALES=1, csfcar). Format and host mirror: model-arch
// glm5next_mlp/scale_rows.rs (`scale_byte` there is this function line for line; its tests check
// it against the decoder at every position).
//
// Owner: gb10 kernels.
// Invariants: a plain scale pointer (bit 0 clear) reads exactly S[n * num_groups + kk], as
// before; a tagged pointer (bit 0 set) points at the row section of an encoded buffer: per row a
// base byte and num_groups 4-bit codes (code 15 = look the byte up in the row's exception list),
// with the header words (head size, rows) in the 8 bytes before it. num_groups is even and at
// most 256 for a tagged pointer (the encoder refuses anything else).

#pragma once

// 2026-10-17: The escape path: row n's exception records (kk << 8 | byte), found through the
// header at R - 8 (R 4-byte aligned). Rare (0.6% of scales on the official experts), so kept out
// of line.
static __device__ __noinline__ unsigned char moe_scale_rows_exception(
    const unsigned char* R, unsigned long long n, unsigned int kk)
{
    const unsigned int head = *(const unsigned int*)(R - 8);
    const unsigned int rows = *(const unsigned int*)(R - 4);
    const unsigned short* off = (const unsigned short*)(R - head);
    const unsigned short* exc = off + rows + 1u;
    const unsigned int hi = off[n + 1ull];
    for (unsigned int i = off[n]; i < hi; i++) {
        const unsigned short r = exc[i];
        if ((unsigned int)(r >> 8) == kk) return (unsigned char)(r & 0xFFu);
    }
    return (unsigned char)0;
}

// 2026-10-17: Scale byte (n, kk) of a [N, num_groups] block scale, plain or tagged.
static __device__ __forceinline__ unsigned char moe_scale_byte(
    const unsigned char* S, unsigned long long n, unsigned int num_groups, unsigned int kk)
{
    const unsigned long long t = (unsigned long long)S;
    if ((t & 1ull) == 0ull) return S[n * num_groups + kk];
    const unsigned char* R = (const unsigned char*)(t ^ 1ull);
    const unsigned char* row = R + n * (1ull + (unsigned long long)(num_groups >> 1));
    const unsigned char b = row[1u + (kk >> 1)];
    const unsigned int code = (kk & 1u) ? (unsigned int)(b >> 4) : (unsigned int)(b & 0x0Fu);
    if (code != 15u) return (unsigned char)(row[0] + code);
    return moe_scale_rows_exception(R, n, kk);
}
