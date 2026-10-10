// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: Kernels `attn_prefill_paged_indirect_h128` and `_64`: attn_prefill_paged_indirect.cu built with HDIM 128,
// for the DFlash drafter's γ-block attention when the drafter's head_dim is 128 (incoai GLM-5.3-Flash-DFlash2).
//
// The common module is compiled with the .cuh default HDIM 256. Fed a head_dim-128 drafter, every tile row loads 256
// elements at head_dim strides, so heads h and h+1 share one tile and each head's scores become
// Q_h·K_h + Q_{h+1}·K_{h+1}: in bounds, wrong weights (A93, measured on GB10 2026-09-07). The drafter picks its module
// by width in `dflash_head::attn_width`, which refuses a width without a build.
//
// Owner: gb10 kernels.
// Invariants: head_dim must equal HDIM (128).

#define HDIM 128
#define KERNEL_NAME attn_prefill_paged_indirect_h128
#include "attn_prefill_paged_indirect.cu"
