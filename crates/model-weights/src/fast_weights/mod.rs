// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast safetensors loader (InstantTensor-style) — pure Rust.
//!
//! Two wins over the mmap-based loader in [`crate::weights`]:
//!
//! 1. **`O_DIRECT`** reads. Bypasses the OS page cache, so the bytes never
//!    compete with GPU allocations on GB10 unified memory. The mmap path
//!    already works around this with `POSIX_FADV_DONTNEED` post-load; here
//!    we avoid the pollution in the first place.
//! 2. **Pipelined read/copy**. One background reader thread fetches the
//!    next tensor while the main thread does `copy_h2d` for the current
//!    one. Overlaps disk I/O with the host→device memcpy.
//!
//! Behavioural parity with [`crate::weights::SafetensorsLoader`] is
//! preserved — same EP filtering, same OOM pre-flight, same UVM fallback
//! on GPU allocation failure, same extra-weights handling.

use crate::weights::{
    WeightLoader, WeightStore, WeightTensor, check_oom_guard, estimate_has_fp8,
    estimate_load_bytes, evict_page_cache, f16_to_bf16_bytes,
};
use anyhow::{Context, Result, bail};
use metrale_gpu_runtime::gpu::GpuBackend;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;

mod arena_plan;
mod direct_io;
mod header;

use header::{parse_header, resolve_shards};

/// Pure-Rust InstantTensor-style loader. Same public shape as
/// [`crate::weights::SafetensorsLoader`].
pub struct FastSafetensorsLoader {
    pub ep_rank: usize,
    pub ep_world_size: usize,
    pub num_experts: usize,
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
    /// (GLM-5.3's MTP block at `layers.{num_hidden_layers}`); `None` keeps all.
    pub skip_layers_from: Option<usize>,
    /// When true (default), attempt `O_DIRECT`; fall back to buffered reads if
    /// the filesystem rejects it (tmpfs, overlayfs, some FUSE backends).
    pub try_direct_io: bool,
    /// Per-shard heuristic cap: if a shard's tensor count exceeds this,
    /// we skip `O_DIRECT` for that shard and fall back to buffered +
    /// pipelined reads even when [`Self::try_direct_io`] is `true`.
    ///
    /// Motivation: `O_DIRECT`'s 4 KiB-aligned per-tensor `pread` has a
    /// fixed syscall + copy overhead that kernel readahead amortises for
    /// free on the buffered path. Benchmarks on GB10 showed buffered wins
    /// above ~5k tensors/shard; O_DIRECT wins below. Set to [`usize::MAX`]
    /// to disable.
    pub direct_io_tensor_cap: usize,
    /// When true, advise the kernel to read a whole buffered shard
    /// sequentially before the per-tensor copy loop starts. This helps NFS
    /// mounts where many small tensor reads defeat normal readahead.
    pub prefetch_shards: bool,
    /// Skip a multimodal checkpoint's vision tower.
    ///
    /// Set by the caller from `ModelWeightLoader::binds_vision_encoder()`:
    /// false by default, true only when the model's loader is a text-only
    /// port that will never bind the tower. Reading it anyway costs the full
    /// tower in unified memory (1.05 GiB/rank on GLM-5.3's checkpoint) from
    /// load time until `build_model` frees it — which is after the inference
    /// -buffer preflight has already refused the serve.
    pub skip_vision: bool,
    /// Tensors the MODEL's weight loader will read from disk itself, so this
    /// loader records their location instead of uploading them.
    ///
    /// Set by the caller from `ModelWeightLoader::defer_predicate()`, the same
    /// way `skip_vision` comes from `binds_vision_encoder()`. `None` for every
    /// model that does not opt in. See [`crate::weights::DeferHook`] for the
    /// contract — including the half that says the pre-flight must agree.
    pub defer: Option<crate::weights::DeferHook>,
    /// 2026-10-06: Tensors the MODEL's weight loader keeps until teardown and never frees
    /// alone, so they are uploaded into one [`crate::weights::WeightArena`] instead of one
    /// allocation each (`ModelWeightLoader::arena_predicate`, `METRALE_GLM_WEIGHT_ARENA=1`).
    /// `None` (the default) uploads every tensor as before.
    pub arena: Option<crate::weights::ArenaHook>,
}

/// Is this tensor part of a multimodal checkpoint's vision tower?
///
/// Same three spellings `build_model`'s unbound-tower reclaim matches, kept
/// here so the load-time skip and the post-bind free can never disagree.
pub fn is_vision_tensor(name: &str) -> bool {
    name.starts_with("model.visual.")
        || name.starts_with("model.vision")
        || name.starts_with("visual.")
}

/// Default tensor-count cap for per-shard `O_DIRECT`. Above this, the fast
/// loader uses buffered reads even when `try_direct_io = true`. See the
/// field doc on [`FastSafetensorsLoader::direct_io_tensor_cap`].
pub const DEFAULT_DIRECT_IO_TENSOR_CAP: usize = 5000;

impl Default for FastSafetensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

mod load;
#[path = "skip.rs"]
mod skip;

impl FastSafetensorsLoader {
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            peak_memory_multiplier: None,
            skip_activation_scales: false,
            skip_mtp: false,
            skip_layers_from: None,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
            prefetch_shards: false,
            skip_vision: false,
            defer: None,
            arena: None,
        }
    }

    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
            skip_activation_scales: false,
            skip_mtp: false,
            skip_layers_from: None,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
            prefetch_shards: false,
            skip_vision: false,
            defer: None,
            arena: None,
        }
    }
}

/// Load a single shard with O_DIRECT + pipelined read/copy.
///
/// Pipeline:
///   reader thread: pread tensor N into aligned buffer → sync_channel ──▶
///   main thread:   recv → copy_h2d → store tensor
///
/// The channel has capacity 1, so at any time the reader is ≤1 tensor
/// ahead of the copier. Memory overhead per shard: 2 × max_tensor_bytes
/// (rounded up to O_DIRECT alignment).
#[allow(clippy::too_many_arguments)]
fn load_shard_fast(
    shard_path: &Path,
    tensor_filter: Option<&[String]>,
    gpu: &dyn GpuBackend,
    skip_fn: &dyn Fn(&str) -> bool,
    defer_fn: &dyn Fn(&str, crate::weights::WeightDtype) -> bool,
    try_direct_io: bool,
    direct_io_tensor_cap: usize,
    prefetch_shards: bool,
    out: &mut HashMap<String, WeightTensor>,
    deferred_out: &mut HashMap<String, crate::weights::DeferredTensor>,
    offload_logged: &mut bool,
    arena: arena_plan::ArenaSel<'_>,
) -> Result<()> {
    // Header parsing uses a buffered fd — header is a few KB, cache pollution
    // is negligible and buffered I/O handles short reads cleanly.
    let mut meta_file = File::open(shard_path)
        .with_context(|| format!("Failed to open {}", shard_path.display()))?;
    let mut tensors = parse_header(&mut meta_file)?;
    let file_size = meta_file.metadata()?.len();

    // Filter down to tensors we actually want (index filter + EP filter).
    if let Some(allow) = tensor_filter {
        let allow_set: std::collections::HashSet<&str> = allow.iter().map(|s| s.as_str()).collect();
        tensors.retain(|t| allow_set.contains(t.name.as_str()));
    }
    // The n-gram embedding TABLES are never uploaded with the checkpoint —
    // 63 GB (LongCat-Lite) to ~102 GB (Flash-Next) of BF16 would exhaust a
    // 121 GB unified box before any quantization could run, and the fallback
    // on GB10 is managed memory, i.e. Linux swap, i.e. a kernel freeze. They
    // are recorded with their on-disk location and served either by streaming
    // per-table quantize-on-load or straight off NVMe by the row cache.
    //
    // The MODEL's loader can claim tensors too (`defer_fn`), for the other
    // reason a tensor must not be swept: the bytes that reach the device are
    // not the bytes on disk. `nvidia/GLM-5.3-Flash-NVFP4` ships the MTP
    // block's routed experts full-width BF16 against a w4a16-only forward, so
    // every one of them would be uploaded, read back, quantised and freed —
    // ~7.25 GB per EP=2 rank resident across the whole 45-layer build.
    let mut deferred_here: Vec<(String, crate::weights::DeferredTensor)> = Vec::new();
    let mut ngram_count = 0usize;
    #[allow(clippy::items_after_statements)]
    tensors.retain(|t| {
        let mut defer = |t: &header::TensorMeta| {
            deferred_here.push((
                t.name.clone(),
                crate::weights::DeferredTensor {
                    path: shard_path.to_path_buf(),
                    offset: t.abs_offset,
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                },
            ));
        };
        if crate::weights::is_ngram_table(&t.name) {
            defer(t);
            ngram_count += 1;
            return false;
        }
        // 🪤 The model hook is asked only about tensors this rank KEEPS. Asked
        // first, it would record a remote expert's on-disk location and invite
        // a binder to read weights EP gave to another rank.
        if skip_fn(&t.name) {
            return false;
        }
        // 🪤 `from_f16` is not deferrable: the store's BF16 is a REWRITE of the
        // disk bytes, so a (path, offset) locator would hand its reader F16.
        // `estimate_load_bytes` refuses the same case, so the two agree.
        if !t.from_f16 && defer_fn(&t.name, t.dtype) {
            defer(t);
            return false;
        }
        true
    });
    if !deferred_here.is_empty() {
        tracing::info!(
            "Deferred {} tensor(s) in {} ({} n-gram table(s), {} claimed by the model's \
             weight loader) — served from disk, not uploaded",
            deferred_here.len(),
            shard_path.display(),
            ngram_count,
            deferred_here.len() - ngram_count,
        );
        deferred_out.extend(deferred_here);
    }

    // Per-shard heuristic: above `direct_io_tensor_cap` tensors, O_DIRECT's
    // per-tensor syscall + 4 KiB alignment overhead costs more than kernel
    // readahead on the buffered path saves. Skip the direct-open attempt
    // entirely in that case — keeps the log clean and avoids a wasted fd.
    let wants_direct = try_direct_io && tensors.len() <= direct_io_tensor_cap;
    if try_direct_io && !wants_direct {
        tracing::info!(
            "  Shard has {} tensors (> {} cap) — using buffered+pipelined path",
            tensors.len(),
            direct_io_tensor_cap
        );
    }

    // File for data reads. Try O_DIRECT; if it fails, fall through to buffered.
    let (direct_file, using_direct) = match wants_direct
        .then(|| direct_io::open_direct(shard_path))
        .transpose()
    {
        Ok(Some(f)) => (Some(f), true),
        Ok(None) => (None, false),
        Err(e) => {
            tracing::warn!(
                "O_DIRECT open failed for {} ({e}); falling back to buffered reads",
                shard_path.display()
            );
            (None, false)
        }
    };
    let buffered_file = File::open(shard_path)?;
    let data_fd = direct_file.as_ref().unwrap_or(&buffered_file);
    if prefetch_shards && !using_direct {
        advise_prefetch_shard(&buffered_file, shard_path, file_size);
    }

    // Pipelined reader: sends (tensor_index, aligned_buffer, slice_start) to main.
    type ReadMsg = (usize, direct_io::AlignedBuffer, usize);
    let (tx, rx) = sync_channel::<Result<ReadMsg>>(1);
    let tensors_for_reader: Vec<(u64, usize)> =
        tensors.iter().map(|t| (t.abs_offset, t.len)).collect();
    let raw_fd = {
        use std::os::unix::io::AsRawFd;
        data_fd.as_raw_fd()
    };

    let _ = file_size; // retained for future use (tail-fragment buffered read)
    // 2026-10-06: Plan the weight arena for this shard's claimed tensors (all false without one).
    let in_arena = arena_plan::plan_shard(arena, &tensors);
    let reader_handle = std::thread::spawn(move || {
        for (idx, (abs_offset, len)) in tensors_for_reader.iter().enumerate() {
            let msg = direct_io::read_tensor_aligned(raw_fd, *abs_offset, *len, using_direct)
                .map(|(buf, slice_start)| (idx, buf, slice_start));
            if tx.send(msg).is_err() {
                break; // receiver dropped
            }
        }
    });

    // Copier: drains the channel, does gpu alloc + copy_h2d, inserts into the map.
    for result in rx {
        let (idx, buf, slice_start) = result?;
        let meta = &tensors[idx];
        let raw = &buf.as_slice()[slice_start..slice_start + meta.len];
        // F16 shards: convert bytes to BF16 before upload (same length,
        // different bit layout — meta.dtype is already staged as BF16).
        let converted: Vec<u8>;
        let src: &[u8] = if meta.from_f16 {
            converted = f16_to_bf16_bytes(raw);
            &converted
        } else {
            raw
        };

        // 2026-10-06: An arena tensor's bytes go into its arena slot; every other tensor (and
        // every tensor without an arena) is allocated alone, as before.
        let from_arena = arena_plan::alloc(arena, in_arena[idx], gpu, meta.len)?;
        let ptr = match from_arena.map_or_else(|| gpu.alloc(meta.len), Ok) {
            Ok(p) => {
                gpu.copy_h2d(src, p)?;
                p
            }
            Err(_) => {
                if !*offload_logged {
                    tracing::warn!(
                        "GPU alloc failed for {} ({} bytes) — switching to managed (UVM) memory",
                        meta.name,
                        meta.len
                    );
                    *offload_logged = true;
                }
                let p = gpu.alloc_managed(meta.len)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), p.0 as *mut u8, meta.len);
                }
                p
            }
        };

        out.insert(
            meta.name.clone(),
            WeightTensor {
                ptr,
                shape: meta.shape.clone(),
                dtype: meta.dtype,
            },
        );
    }

    reader_handle
        .join()
        .map_err(|_| anyhow::anyhow!("reader thread panicked"))?;

    // Release file handles, then advise the kernel to drop any pages we did
    // end up caching on the buffered fallback path. O_DIRECT reads never hit
    // the page cache, so the posix_fadvise is a no-op there but cheap.
    drop(direct_file);
    evict_page_cache(&buffered_file);
    drop(buffered_file);
    Ok(())
}

#[cfg(target_os = "linux")]
fn advise_prefetch_shard(file: &File, shard_path: &Path, file_size: u64) {
    use std::os::unix::io::AsRawFd;

    let fd = file.as_raw_fd();
    let seq_rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL) };
    let willneed_rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_WILLNEED) };
    if seq_rc == 0 && willneed_rc == 0 {
        tracing::info!(
            "  NFS/shard prefetch requested for {} ({:.2} GB)",
            shard_path.display(),
            file_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        tracing::warn!(
            "  NFS/shard prefetch hint failed for {}: sequential_rc={}, willneed_rc={}",
            shard_path.display(),
            seq_rc,
            willneed_rc
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_prefetch_shard(_file: &File, _shard_path: &Path, _file_size: u64) {}

#[cfg(test)]
mod skip_vision_tests {
    use super::{FastSafetensorsLoader, is_vision_tensor};

    fn loader(skip_vision: bool, ep: usize) -> FastSafetensorsLoader {
        let mut l = FastSafetensorsLoader::with_ep(0, ep, 288);
        l.skip_vision = skip_vision;
        l
    }

    #[test]
    fn vision_names_are_recognised() {
        assert!(is_vision_tensor("model.visual.blocks.0.attn.proj.weight"));
        assert!(is_vision_tensor(
            "model.vision_tower.encoder.layer.0.weight"
        ));
        assert!(is_vision_tensor("visual.merger.proj.weight"));
        assert!(!is_vision_tensor(
            "model.language_model.layers.45.eh_proj.weight"
        ));
        // The trap: a text tensor whose name merely CONTAINS "vision".
        assert!(!is_vision_tensor(
            "model.language_model.layers.3.mlp.revision.weight"
        ));
    }

    #[test]
    fn skip_vision_drops_only_the_tower() {
        let l = loader(true, 2);
        assert!(l.should_skip_tensor("model.visual.blocks.0.attn.proj.weight"));
        assert!(!l.should_skip_tensor("model.language_model.layers.45.eh_proj.weight"));
        assert!(!l.should_skip_tensor("lm_head.weight"));
    }

    #[test]
    fn skip_vision_applies_without_ep() {
        // The EP short-circuit must not swallow the vision rule at ep=1.
        let l = loader(true, 1);
        assert!(l.should_skip_tensor("model.visual.blocks.0.attn.proj.weight"));
        assert!(!l.should_skip_tensor("model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn default_loader_keeps_the_tower() {
        let l = loader(false, 2);
        assert!(!l.should_skip_tensor("model.visual.blocks.0.attn.proj.weight"));
        assert!(!FastSafetensorsLoader::new().skip_vision);
    }

    #[test]
    fn ep_expert_filtering_is_unchanged_by_the_vision_rule() {
        let l = loader(true, 2); // ep_rank 0 of 2, 288 experts -> keeps 0..143
        assert!(
            !l.should_skip_tensor("model.language_model.layers.4.mlp.experts.7.up_proj.weight")
        );
        assert!(
            l.should_skip_tensor("model.language_model.layers.4.mlp.experts.200.up_proj.weight")
        );
    }
}
