// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weight loading from safetensors files (SBIO IORouter for filesystem I/O).

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use std::collections::HashMap;
use std::path::Path;

/// Advise the OS to evict a file's pages from the page cache.
///
/// On GB10 (unified memory), mmap'd safetensors share the GPU memory pool.
/// After copying tensors to GPU, the mmap pages linger in the page cache,
/// consuming memory that should be available for KV cache and inference buffers.
/// This function tells the kernel those pages are no longer needed.
#[cfg(target_os = "linux")]
pub(crate) fn evict_page_cache(file: &std::fs::File) {
    use std::os::unix::io::AsRawFd;
    // POSIX_FADV_DONTNEED = 4 on Linux (POSIX standard).
    // macOS lacks posix_fadvise — see the non-linux branch below.
    const POSIX_FADV_DONTNEED: libc::c_int = 4;
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED);
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn evict_page_cache(_file: &std::fs::File) {
    // No-op: macOS/BSD have no posix_fadvise. Apple Silicon UMA already
    // shares page cache with the GPU pool, so eviction is unnecessary.
}

/// Data type of a weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    BF16,
    FP32,
    FP8E4M3,
    FP8E8M0,
    UInt8,
    Int64,
    /// Keep-packed PrismML ternary Q2_0 (ggml id 42): raw on-disk blocks stay
    /// 2-bit in VRAM (fp16 scale + 2-bit codes per group of `group` elements),
    /// dequantized in-kernel by the native `q2_0_gemv` decode path. Only
    /// produced by the GGUF loader under `METRALE_GGUF_NATIVE_Q2=1`. Its byte
    /// footprint is NOT a per-element size (2-bit codes + an inline scale per
    /// group), so [`WeightDtype::byte_size`] returns 0 for this variant and the
    /// real size is computed in [`WeightTensor::byte_size`] (shape + group).
    PackedQ2_0 {
        group: u16,
    },
    /// Raw GGUF `Q2_K` blocks kept on the device (84 bytes per 256 weights,
    /// `[N, K]` row-major in blocks) for the K-quant GEMV/MMQ kernels.
    Q2K,
    /// Raw GGUF `Q3_K` blocks kept on the device (110 bytes per 256 weights).
    Q3K,
    /// Raw GGUF `Q6_K` blocks kept on the device (210 bytes per 256 weights):
    /// the DeepSeek-V4.1 output head, run by the K-quant GEMV instead of a
    /// bf16 expansion (0.82 instead of 2 bytes a weight read every token).
    Q6K,
}

impl WeightDtype {
    /// Bytes per element for the fixed-width dtypes. Returns 0 for the
    /// block-based [`WeightDtype::PackedQ2_0`] — [`WeightTensor::byte_size`]
    /// handles that variant directly, and no caller multiplies its numel by this.
    pub fn byte_size(self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::FP32 => 4,
            Self::FP8E4M3 => 1,
            Self::FP8E8M0 => 1,
            Self::UInt8 => 1,
            Self::Int64 => 8,
            Self::PackedQ2_0 { .. } => 0,
            Self::Q2K => 0,
            Self::Q3K => 0,
            Self::Q6K => 0,
        }
    }

    fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self> {
        match dtype {
            safetensors::Dtype::BF16 => Ok(Self::BF16),
            safetensors::Dtype::F32 => Ok(Self::FP32),
            safetensors::Dtype::U8 => Ok(Self::UInt8),
            // I8: raw 1-byte container for 4-bit-packed NVFP4 (DeepSeek-V4 MTP
            // experts). Treat as UInt8 — signedness is irrelevant for packed FP4.
            safetensors::Dtype::I8 => Ok(Self::UInt8),
            safetensors::Dtype::F8_E4M3 => Ok(Self::FP8E4M3),
            safetensors::Dtype::F8_E8M0 => Ok(Self::FP8E8M0),
            safetensors::Dtype::I64 => Ok(Self::Int64),
            other => bail!("Unsupported safetensors dtype: {other:?}"),
        }
    }

    /// Map a raw safetensors header dtype STRING (as it appears in the JSON
    /// header, e.g. `"BF16"`, `"F8_E4M3"`) to a [`WeightDtype`], factored out
    /// so the RDMA weight loader (which receives dtype as a wire string in the
    /// peer manifest, not a `safetensors::Dtype`) resolves it identically to
    /// the disk loaders — byte-identity depends on the two ends agreeing.
    pub fn from_safetensors_str(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => Self::FP32,
            "BF16" => Self::BF16,
            "U8" => Self::UInt8,
            // I8 is a 1-byte raw container (packed NVFP4); signedness is
            // irrelevant, treat as raw bytes exactly like the disk path.
            "I8" => Self::UInt8,
            "F8_E4M3" => Self::FP8E4M3,
            "F8_E8M0" => Self::FP8E8M0,
            "I64" => Self::Int64,
            other => bail!("Unsupported safetensors dtype '{other}'"),
        })
    }
}

/// Convert a little-endian IEEE-754 half-precision (F16) tensor byte buffer
/// to BF16 bytes. F16 and BF16 are both 2 bytes/element but have different
/// bit layouts (5-bit vs 8-bit exponent), so the bytes cannot be
/// reinterpreted — each value goes f16 → f32 (exact) → bf16
/// (round-to-nearest-even). Shared by both disk loaders so F16 checkpoints
/// (e.g. centml modelopt W4A4 exports, which ship all unquantized tensors as
/// F16) land in the store as BF16; [`WeightDtype`] itself stays closed to
/// store-legal dtypes and F16 can never appear on the RDMA wire.
pub(crate) fn f16_to_bf16_bytes(src: &[u8]) -> Vec<u8> {
    use half::{bf16, f16};
    debug_assert_eq!(src.len() % 2, 0, "F16 tensor byte length must be even");
    let mut out = Vec::with_capacity(src.len());
    for pair in src.chunks_exact(2) {
        let h = f16::from_le_bytes([pair[0], pair[1]]);
        out.extend_from_slice(&bf16::from_f32(h.to_f32()).to_le_bytes());
    }
    out
}

/// A weight tensor on the GPU.
pub struct WeightTensor {
    pub ptr: DevicePtr,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

impl WeightTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        match self.dtype {
            // Packed Q2_0: `n_blocks = numel / group` blocks of
            // `2 + group/4` bytes (34 @ g128, 18 @ g64) — the on-disk footprint.
            WeightDtype::PackedQ2_0 { group } => {
                let g = group as usize;
                debug_assert!(g == 128 || g == 64, "unexpected Q2_0 group {g}");
                let n_blocks = self.num_elements() / g.max(1);
                n_blocks * (2 + g / 4)
            }
            WeightDtype::Q2K => self.num_elements() / 256 * 84,
            WeightDtype::Q3K => self.num_elements() / 256 * 110,
            WeightDtype::Q6K => self.num_elements() / 256 * 210,
            d => self.num_elements() * d.byte_size(),
        }
    }

    /// The Q2_0 group size if this tensor is keep-packed ternary, else `None`.
    pub fn q2_group(&self) -> Option<u16> {
        match self.dtype {
            WeightDtype::PackedQ2_0 { group } => Some(group),
            _ => None,
        }
    }

    /// True if this tensor holds keep-packed ternary Q2_0 blocks (id 42).
    pub fn is_packed_q2(&self) -> bool {
        matches!(self.dtype, WeightDtype::PackedQ2_0 { .. })
    }
}

/// All model weights loaded onto the GPU, keyed by HuggingFace name.
pub struct WeightStore {
    weights: HashMap<String, WeightTensor>,
    prepartitioned_tp: Option<(usize, usize)>,
    /// Buffers a loader derived from these tensors — fused concats, transposed
    /// twins, requants. Owned here so teardown RELEASES them instead of the
    /// backend sweep reclaiming them unowned (#736, #915); see `derived.rs`.
    derived: DerivedStore,
    /// Tensors deliberately NOT uploaded, with where they live on disk.
    ///
    /// The n-gram embedding tables of the LongCat / Qwen3.8-Flash-Next family
    /// are 63 GB (LongCat-Lite) to ~102 GB (Flash-Next) of BF16. Uploading
    /// them through the generic path would exhaust a 121 GB unified box
    /// before any quantization could run — and on GB10 the fallback is
    /// `alloc_managed`, i.e. Linux swap, i.e. the documented kernel freeze.
    /// They are skipped at load and served either by streaming per-table
    /// quantize-on-load or straight off NVMe by `NgramRowCache`, both of
    /// which need only this (path, offset) locator.
    deferred: HashMap<String, DeferredTensor>,
    /// 2026-10-06: Where the checkpoint tensors an [`ArenaHook`] claimed live (the fast loader,
    /// `METRALE_GLM_WEIGHT_ARENA=1`); disabled and empty otherwise. `free_matching` and
    /// `release` never pass one of its pointers to `GpuBackend::free`; see `arena.rs`.
    arena: WeightArena,
}

mod arena;
pub use arena::{
    ArenaHook, ArenaStats, WEIGHT_ARENA_ALIGN, WEIGHT_ARENA_CHUNK_CAP, WeightArena,
    arena_footprint, pack_chunks,
};
mod deferred;
pub use deferred::{DeferHook, DeferredTensor};
mod store;

/// SBIO IORouter trait for weight loading.
pub trait WeightLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore>;
}

/// Loads weights from safetensors files using mmap.
pub struct SafetensorsLoader {
    /// EP rank (0-based). Only used when ep_world_size > 1.
    pub ep_rank: usize,
    /// EP world size. When > 1, remote expert tensors are skipped.
    pub ep_world_size: usize,
    /// Total number of MoE experts in the model (for EP partitioning).
    pub num_experts: usize,
    /// Override for the peak memory multiplier in the pre-flight OOM check.
    /// Set from QuantFormat::peak_memory_multiplier() in the caller.
    /// When None, the pre-flight uses its own heuristic (1.3x NVFP4 / 1.5x FP8).
    pub peak_memory_multiplier: Option<f64>,
    /// Skip the W4A4 `*.input_scale` activation scales at load.
    ///
    /// ModelOpt NVFP4 checkpoints ship one 0-dim F32 scalar per quantized
    /// projection. On a 512-expert model that is ~74k four-byte allocations,
    /// each taking a full allocation granule — GBs of padding for values
    /// Metrale Engine never reads, because it serves w4a16 (BF16 activations) and the
    /// NVFP4 loader already treats the key as optional.
    ///
    /// OPT-IN: `step3p7` reads this key on its own path, so it must stay off
    /// unless the model's loader is known not to need it.
    pub skip_activation_scales: bool,
    /// Skip `mtp.*` tensors at load.
    ///
    /// For models whose loader deliberately does not build an MTP head,
    /// uploading its weights is pure waste — on Qwen3.8-Flash-Next that is a
    /// 1.49 GB expert shard plus the MTP backbone, held resident while the KV
    /// cache goes without.
    ///
    /// OPT-IN: a model that DOES build an MTP head must keep them, so this is
    /// set only where `load_mtp_weights` is known to return `None`.
    pub skip_mtp: bool,
    /// 2026-10-02: Skip the text stack's layers numbered `first` and above
    /// (`model.language_model.layers.N.*`, `model.layers.N.*`): GLM-5.3's MTP
    /// block sits at `layers.{num_hidden_layers}`. Set only when no MTP head is
    /// requested (`--dflash` without `--speculative`). `None` (default) keeps
    /// every layer.
    pub skip_layers_from: Option<usize>,
    /// 2026-10-02: Skip tensors whose name ends with any of these (the DFlash
    /// drafter's `embed_tokens.weight` / `lm_head.weight`, which the head
    /// replaces with the target's). Empty (default) skips nothing.
    pub skip_suffixes: &'static [&'static str],
    /// Tensors the MODEL's weight loader will read from disk itself, so this
    /// loader must record their location instead of uploading them. See
    /// [`DeferHook`] for the contract and why a loader asks for it.
    pub defer: Option<DeferHook>,
}

impl Default for SafetensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetensorsLoader {
    /// Create a loader with no expert parallelism (loads all tensors).
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            peak_memory_multiplier: None,
            skip_activation_scales: false,
            skip_mtp: false,
            skip_layers_from: None,
            skip_suffixes: &[],
            defer: None,
        }
    }

    /// Create a loader with EP-aware filtering.
    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
            skip_activation_scales: false,
            skip_mtp: false,
            skip_layers_from: None,
            skip_suffixes: &[],
            defer: None,
        }
    }

    /// Does the model's loader claim this tensor? `false` when no hook is set,
    /// which is every model but the ones that opt in.
    ///
    /// 🪤 Consulted only for tensors `should_skip_tensor` KEPT — deferring a
    /// tensor this rank was never going to load would put a remote expert's
    /// location in the store and invite a binder to read it.
    pub fn is_deferred(&self, name: &str, dtype: WeightDtype) -> bool {
        self.defer.as_ref().is_some_and(|f| f(name, dtype))
    }

    /// Check if a tensor should be skipped under EP.
    /// Skips `*.experts.{E}.*` tensors where E is not in local range.
    /// MTP head experts are never skipped (small, fully replicated).
    ///
    /// 🪤 The MTP exemption keys on a leading `mtp.` — a DeepSeek-style name.
    /// GLM-5.3 puts its MTP head at `model.language_model.layers.45.*` with no
    /// `mtp.` prefix, so that layer's routed experts ARE sharded on GLM. Fine
    /// while the MTP head is out of scope; revisit before enabling it.
    ///
    /// `pub` so residency can be PROVEN against a real checkpoint index
    /// without collectives (see `metrale-model-layers/tests/glm53_ep_residency.rs`).
    pub fn should_skip_tensor(&self, name: &str) -> bool {
        // MTP head weights for a model whose loader does not build one.
        if self.skip_mtp && name.starts_with("mtp.") {
            return true;
        }
        if let Some(first) = self.skip_layers_from
            && crate::mtp_layout::is_text_layer_at_or_above(name, first)
        {
            return true;
        }
        if self.skip_suffixes.iter().any(|s| name.ends_with(s)) {
            return true;
        }
        // W4A4 activation scales: never read on the w4a16 path (the NVFP4
        // loader falls back to `DevicePtr::NULL`), and 4-byte allocations are
        // almost pure granule padding at expert scale.
        if self.skip_activation_scales && name.ends_with(".input_scale") {
            return true;
        }
        if self.ep_world_size <= 1 {
            return false;
        }
        // MTP head experts are small — always replicate, never shard.
        if name.starts_with("mtp.") {
            return false;
        }
        // Parse expert index from patterns like "*.experts.42.gate_proj*"
        if let Some(idx) = parse_expert_index(name) {
            let per_rank = self.num_experts / self.ep_world_size;
            let local_start = self.ep_rank * per_rank;
            let local_end = if self.ep_rank == self.ep_world_size - 1 {
                self.num_experts
            } else {
                local_start + per_rank
            };
            idx < local_start || idx >= local_end
        } else {
            false // Non-expert tensors are always loaded (replicated)
        }
    }
}

/// Split a tensor name into (everything but its last numeric path segment,
/// that segment as a number) so names sort NUMERICALLY on the index.
/// `embedders.2` must precede `embedders.10`; a plain lexicographic sort puts
/// `10` first and silently mis-maps every table after the ninth.
pub mod adapter;
mod derived;
pub use derived::DerivedStore;
mod gguf;
mod k3;
mod loader;
pub use gguf::dequant_cpu;
pub use gguf::expert_stream;
pub use gguf::{GgufLoader, GgufShardSet, config_from_gguf_dir, find_gguf, find_gguf_shards};
pub use k3::K3SafetensorsLoader;
pub(crate) use loader::estimate_load_bytes;
// Platform-independent: consumed by the unix-only fast-weights (O_DIRECT) path
// AND by the GGUF loader, which builds everywhere. Gating this on `unix` broke
// the Windows CUDA build the moment `gguf.rs` started using it.
pub(crate) use loader::check_oom_guard;
// Consumed by the unix-only fast-weights (O_DIRECT) loader path.
#[cfg(unix)]
pub(crate) use loader::estimate_has_fp8;

mod name_utils;
pub(crate) use name_utils::split_trailing_index;
pub use name_utils::{is_ngram_table, parse_expert_index};

#[cfg(test)]
mod packed_q2_tests;
mod prefix_detect;
pub use prefix_detect::auto_detect_weight_prefix;

mod release;

#[cfg(test)]
mod teardown_tests;
