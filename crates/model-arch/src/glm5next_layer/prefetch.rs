// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Decode L2 weight prefetch across latency windows
//! (`METRALE_GLM_DECODE_L2_PREFETCH=1`, default off).
//!
//! A decode or verify step of 1..=`L2_PREFETCH_MAX_ROWS` rows streams one weight matrix after
//! another, and between two of them runs a latency-bound chain that leaves the DRAM almost
//! idle: the TP/EP all-reduce (including the EP rank-imbalance wait), `hc_post`, `hc_mix`,
//! `hc_finish` and `rms_norm`. The weights read right after each chain are static, so a layer
//! enqueues ONE `glm5next_l2_prefetch` launch just before each of its two all-reduces:
//!
//! * before the attention all-reduce: this layer's FFN head ([`Glm5NextPrefetch::ffn_head`]),
//!   the FFN-site `hc_fn` and the router (MoE) or the dense `gate_proj`;
//! * before the FFN all-reduce: the NEXT layer's attention head
//!   ([`Glm5NextPrefetch::next_attn_head`]), its attention-site `hc_fn` then its first mixer
//!   projection (KDA `q_proj`, DSA `q_a_proj`). Never further ahead: the projection after
//!   each of these streams more than the 24 MB L2 and would evict it.
//!
//! Each launch is clipped to `decode_l2_prefetch_bytes()` (default 12 MiB, half the 24 MB GB10
//! L2; PROVISIONAL until the microtest and a serve A/B size it).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Byte-identical by construction: the kernel writes no device memory; it issues L2 prefetch
//!   hints only. Every GEMV, norm and collective computes the same bits with the lever on or off.
//! - A span is a weight pointer plus exactly the `n * k * elem` extent its consuming launch
//!   reads (`bf16_matrix_span`, `mhc_site_span`), rounded DOWN to 16 bytes, so a prefetch never
//!   leaves the allocation.
//! - The launch is a plain kernel node with static arguments, so it is graph-capture and
//!   replay safe (decode `METRALE_EP_GRAPHS`, verify graphs).
//! - Lever off, an unresolved kernel, no spans or more than `L2_PREFETCH_MAX_ROWS` rows: no
//!   launch, and the step is the one it has always been.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_mhc::{Glm5NextMhcSiteWeights, mix_hc};

/// 2026-10-01: The module (`kernels/gb10/glm-5.3-flash/nvfp4/glm5next_l2_prefetch.cu`).
pub const L2_PREFETCH_MODULE: &str = "glm5next_l2_prefetch";
/// 2026-10-01: The entry point the layer launches.
pub const L2_PREFETCH_KERNEL: &str = "glm5next_l2_prefetch";
/// 2026-10-01: Span slots per launch (the kernel's argument list).
pub const L2_PREFETCH_MAX_SPANS: usize = 4;
/// 2026-10-01: `GLM_L2PF_BLOCK` and `GLM_L2PF_CHUNK` in the kernel.
pub const L2_PREFETCH_BLOCK: u32 = 128;
pub const L2_PREFETCH_CHUNK: usize = 16384;
/// 2026-10-01: Grid cap: one block per GB10 SM (48). A 12 MiB launch needs 768 chunks = 6
/// blocks, so the cap only matters for budgets above ~96 MiB.
pub const L2_PREFETCH_MAX_BLOCKS: u32 = 48;
/// 2026-10-01: Widest step that prefetches: decode (1) and every verify width in use (K3 = 3,
/// DFlash up to 8). Prefill sub-chunks (16..8192 rows) are compute-bound and skip it; a
/// prefill tail of at most 8 rows would also prefetch, which changes no value.
pub const L2_PREFETCH_MAX_ROWS: usize = 8;

/// 2026-10-01: One read-only byte range to pull into L2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2Span {
    pub ptr: DevicePtr,
    pub bytes: usize,
}

/// 2026-10-01: A layer's prefetch plan, built by the loader. Empty lists (the MTP block, the
/// last text layer's `next_attn_head`) launch nothing.
#[derive(Debug, Clone)]
pub struct Glm5NextPrefetch {
    /// 2026-10-01: `glm5next_l2_prefetch`, `KernelHandle(0)` when the target lacks it.
    pub kernel: KernelHandle,
    /// 2026-10-01: Issued before this layer's attention all-reduce.
    pub ffn_head: Vec<L2Span>,
    /// 2026-10-01: Issued before this layer's FFN all-reduce: the next layer's attention head.
    pub next_attn_head: Vec<L2Span>,
}

impl Default for Glm5NextPrefetch {
    fn default() -> Self {
        Self {
            kernel: KernelHandle(0),
            ffn_head: Vec::new(),
            next_attn_head: Vec::new(),
        }
    }
}

/// 2026-10-01: A BF16 `[n, k]` weight as one span (`n * k * 2` bytes).
pub fn bf16_matrix_span(ptr: DevicePtr, n: usize, k: usize) -> L2Span {
    L2Span {
        ptr,
        bytes: n * k * 2,
    }
}

/// 2026-10-01: A mHC site's `hc_fn`, `[mix_hc(hc_mult), hc_mult * hidden]`, BF16 or FP32
/// (`hc_fn_bf16`): the matrix `hc_mix` streams first at every site.
pub fn mhc_site_span(w: &Glm5NextMhcSiteWeights, hc_mult: usize, hidden: usize) -> L2Span {
    let elem = if w.hc_fn_bf16 { 2 } else { 4 };
    L2Span {
        ptr: w.hc_fn,
        bytes: mix_hc(hc_mult) * hc_mult * hidden * elem,
    }
}

/// 2026-10-01: The spans one launch covers: in order, null and empty spans dropped, each
/// length rounded down to 16 bytes, at most `L2_PREFETCH_MAX_SPANS` spans, the running total
/// clipped to `budget` bytes (the span that crosses it is cut, the rest dropped).
pub fn clip_spans(spans: &[L2Span], budget: usize) -> Vec<L2Span> {
    let mut out = Vec::with_capacity(L2_PREFETCH_MAX_SPANS);
    let mut left = budget;
    for s in spans {
        if out.len() == L2_PREFETCH_MAX_SPANS || left < 16 {
            break;
        }
        if s.ptr.0 == 0 {
            continue;
        }
        let bytes = s.bytes.min(left) & !15usize;
        if bytes == 0 {
            continue;
        }
        out.push(L2Span { ptr: s.ptr, bytes });
        left -= bytes;
    }
    out
}

/// 2026-10-01: Blocks for `bytes` total: one thread per `L2_PREFETCH_CHUNK` chunk (counted per
/// span, as the kernel does), at least 1, at most `L2_PREFETCH_MAX_BLOCKS`.
pub fn prefetch_blocks(spans: &[L2Span]) -> u32 {
    let chunks: usize = spans
        .iter()
        .map(|s| s.bytes.div_ceil(L2_PREFETCH_CHUNK))
        .sum();
    (chunks.div_ceil(L2_PREFETCH_BLOCK as usize) as u32).clamp(1, L2_PREFETCH_MAX_BLOCKS)
}

/// 2026-10-01: One `glm5next_l2_prefetch` launch over `clip_spans(spans, budget)` on `stream`.
/// No launch (and `Ok`) when the kernel is unresolved or nothing is left after clipping.
pub fn launch_l2_prefetch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    spans: &[L2Span],
    budget: usize,
    stream: u64,
) -> Result<()> {
    if kernel.0 == 0 {
        return Ok(());
    }
    let s = clip_spans(spans, budget);
    if s.is_empty() {
        return Ok(());
    }
    let mut l = KernelLaunch::new(gpu, kernel)
        .grid([prefetch_blocks(&s), 1, 1])
        .block([L2_PREFETCH_BLOCK, 1, 1]);
    for i in 0..L2_PREFETCH_MAX_SPANS {
        let (p, n) = s
            .get(i)
            .map_or((DevicePtr(0), 0u64), |x| (x.ptr, x.bytes as u64));
        l = l.arg_ptr(p).arg_u64(n);
    }
    l.launch(stream)
}

#[cfg(test)]
mod prefetch_tests {
    use super::*;

    fn sp(p: u64, b: usize) -> L2Span {
        L2Span {
            ptr: DevicePtr(p),
            bytes: b,
        }
    }

    /// 2026-10-01: The budget cuts the crossing span and drops the rest; lengths stay multiples
    /// of 16 and never exceed the span they came from.
    #[test]
    fn clip_cuts_at_budget_and_rounds_down() {
        let s = [sp(0x1000, 1000), sp(0x2000, 50_000), sp(0x3000, 4096)];
        let c = clip_spans(&s, 20_000);
        assert_eq!(c, vec![sp(0x1000, 992), sp(0x2000, 19_008)]);
        assert!(c.iter().zip(&s).all(|(a, b)| a.bytes <= b.bytes && a.bytes % 16 == 0));
        assert!(c.iter().map(|x| x.bytes).sum::<usize>() <= 20_000);
    }

    /// 2026-10-01: Null pointers and spans shorter than 16 bytes are skipped; at most four
    /// spans survive.
    #[test]
    fn clip_skips_null_and_tiny_and_caps_slots() {
        let s = [
            sp(0, 4096),
            sp(0x10, 8),
            sp(0x100, 32),
            sp(0x200, 32),
            sp(0x300, 32),
            sp(0x400, 32),
            sp(0x500, 32),
        ];
        let c = clip_spans(&s, usize::MAX);
        assert_eq!(c.len(), L2_PREFETCH_MAX_SPANS);
        assert_eq!(c[0], sp(0x100, 32));
        assert!(clip_spans(&s, 15).is_empty());
        assert!(clip_spans(&[], 1 << 20).is_empty());
    }

    /// 2026-10-01: Blocks follow the per-span chunk count, clamped to [1, 48].
    #[test]
    fn blocks_follow_chunk_count() {
        assert_eq!(prefetch_blocks(&[sp(1, 16)]), 1);
        // 2026-10-01: 12 MiB = 768 chunks = 6 blocks of 128 threads.
        assert_eq!(prefetch_blocks(&[sp(1, 12 << 20)]), 6);
        assert_eq!(prefetch_blocks(&[sp(1, 1 << 30)]), L2_PREFETCH_MAX_BLOCKS);
        // 2026-10-01: Chunks are counted per span: two 1-byte-over spans need 4 chunks.
        let two = [
            sp(1, L2_PREFETCH_CHUNK + 16),
            sp(2, L2_PREFETCH_CHUNK + 16),
        ];
        assert_eq!(prefetch_blocks(&two), 1);
    }

    /// 2026-10-01: Span extents are the consuming launch's `n * k * elem`.
    #[test]
    fn span_extents() {
        assert_eq!(bf16_matrix_span(DevicePtr(8), 288, 4096).bytes, 288 * 4096 * 2);
        let w = Glm5NextMhcSiteWeights {
            hc_fn: DevicePtr(16),
            hc_fn_bf16: true,
            hc_scale: DevicePtr(0),
            hc_base: DevicePtr(0),
            mix: DevicePtr(0),
        };
        assert_eq!(mhc_site_span(&w, 4, 4096).bytes, 24 * 4 * 4096 * 2);
        let w32 = Glm5NextMhcSiteWeights {
            hc_fn_bf16: false,
            ..w
        };
        assert_eq!(mhc_site_span(&w32, 4, 4096).bytes, 24 * 4 * 4096 * 4);
    }
}
