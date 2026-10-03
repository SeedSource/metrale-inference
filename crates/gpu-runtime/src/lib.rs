// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU runtime: the `GpuBackend` trait with its CUDA, Metal and mock
//! implementations, device buffers, streams, the kernel registry, the cuBLASLt,
//! CUTLASS, FlashInfer and FlashKDA bridges, and the process-wide SSM tail-snapshot and
//! hermetic switches.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

#![deny(warnings)]
#![deny(clippy::all)]

pub mod buffers;
#[cfg(feature = "cuda")]
pub mod cublaslt;
// 2026-09-25: Without the `cuda` feature these modules come from the
// `*_metal_stub.rs` files, which keep the entry points metrale-model-layers calls
// without a `cfg`; their GPU entry points are `unreachable!`.
#[cfg(not(feature = "cuda"))]
#[path = "cublaslt_metal_stub.rs"]
pub mod cublaslt;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(feature = "cuda")]
pub mod cutlass;
#[cfg(not(feature = "cuda"))]
#[path = "cutlass_metal_stub.rs"]
pub mod cutlass;
#[cfg(feature = "cuda")]
pub mod flashinfer;
#[cfg(not(feature = "cuda"))]
#[path = "flashinfer_metal_stub.rs"]
pub mod flashinfer;
// 2026-10-03: No stub needed: without `cfg(metrale_flashkda)` (never set without the `cuda`
// feature) the module only reports itself unavailable.
pub mod flashkda;
pub mod gpu;
#[path = "gpu_args.rs"]
mod gpu_args;
pub mod host_heap;
pub mod kernel_args;
pub mod lazy_buffer;
#[cfg(feature = "metal")]
pub mod metal_backend;
pub mod op_cache;
pub mod pinned_hosts;
pub mod timing;

/// 2026-09-25: The last multiple of `block_size` strictly below `total_tokens`, or
/// `None` when `block_size` is 0 or `total_tokens <= block_size`.
///
/// The SSM tail snapshot is placed here: in-pass by `prepare_midchunk_capture`
/// (metrale-model-engine), or, with `ssm_tail_ckpt_enabled` on and
/// `ssm_tail_midchunk_enabled` off, by ending a prefill chunk here
/// (`run_standard` and `run_batched_mixed` in metrale-server).
pub fn ssm_tail_boundary(total_tokens: usize, block_size: usize) -> Option<usize> {
    if block_size == 0 || total_tokens <= block_size {
        return None;
    }
    let boundary = ((total_tokens - 1) / block_size) * block_size;
    (boundary > 0).then_some(boundary)
}

/// 2026-09-25: Tail-checkpoint chunk split, on only with `METRALE_SSM_TAIL_CKPT=1`
/// (see `ssm_tail_boundary`). Resolved on first call and cached.
///
/// Off by default: an A/B measured 2026-07-10 (174 samples per arm) found it
/// perf-neutral, because the extra chunk split (median 868 ms) cancelled the SSM
/// replay it saved (about 1374 ms).
pub fn ssm_tail_ckpt_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("METRALE_SSM_TAIL_CKPT").as_deref(), Ok("1")))
}

/// 2026-09-25: Publish `met serve --hermetic`, before the first `hermetic_enabled`
/// call, which caches its answer. Only `true` is published, so without the flag
/// `METRALE_HERMETIC=1` still decides.
pub fn set_hermetic(on: bool) {
    if on {
        let _ = HERMETIC.set(true);
    }
}

static HERMETIC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// 2026-09-25: Whether hermetic mode is on: `set_hermetic(true)` was called or
/// `METRALE_HERMETIC=1`. Resolved on first call and cached; read by the SSM
/// snapshot lookups in metrale-cache (`snapshot.rs`, `snapshot_tier.rs`).
pub fn hermetic_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        HERMETIC
            .get()
            .copied()
            .unwrap_or_else(|| matches!(std::env::var("METRALE_HERMETIC").as_deref(), Ok("1")))
    })
}

/// 2026-09-25: Publish `--no-ssm-tail-midchunk`: the serve plan passes
/// `Some(false)` when the flag is given and `None` otherwise. `None` publishes
/// nothing, so `METRALE_SSM_TAIL_MIDCHUNK` still decides. The cell is set once:
/// the first published value, or the first `ssm_tail_midchunk_enabled` read, wins.
pub fn set_ssm_tail_midchunk(on: Option<bool>) {
    if let Some(on) = on {
        let _ = SSM_TAIL_MIDCHUNK.set(on);
    }
}

static SSM_TAIL_MIDCHUNK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// 2026-09-25: Mid-chunk SSM tail capture: on unless `--no-ssm-tail-midchunk` or
/// `METRALE_SSM_TAIL_MIDCHUNK=0`. While it is on the scheduler does not end chunks
/// at `ssm_tail_boundary`, and `prepare_midchunk_capture` (metrale-model-engine)
/// plans an in-pass capture there, which it does only on `metrale_scale` builds.
/// Resolved on first call and cached.
pub fn ssm_tail_midchunk_enabled() -> bool {
    *SSM_TAIL_MIDCHUNK.get_or_init(|| {
        !matches!(
            std::env::var("METRALE_SSM_TAIL_MIDCHUNK").as_deref(),
            Ok("0")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absent_flag_does_not_seal_the_midchunk_cell() {
        // 2026-09-25: `None` must leave the cell unset so a later `Some` applies.
        // The cell is process-global with no reset, so no other test in this
        // crate may touch it.
        for _ in 0..3 {
            set_ssm_tail_midchunk(None);
        }
        set_ssm_tail_midchunk(Some(false));
        assert!(
            !ssm_tail_midchunk_enabled(),
            "an absent flag must leave the cell open for the next writer"
        );
        set_ssm_tail_midchunk(Some(true));
        assert!(!ssm_tail_midchunk_enabled(), "and a SET one is final");
    }

    #[test]
    fn the_tail_boundary_is_the_last_block_strictly_below_the_prompt() {
        // 2026-09-25: No boundary gives `None`, not 0: a snapshot at token 0
        // would restore nothing.
        assert_eq!(ssm_tail_boundary(0, 16), None);
        assert_eq!(ssm_tail_boundary(16, 16), None, "not the prompt's own end");
        assert_eq!(ssm_tail_boundary(17, 16), Some(16));
        assert_eq!(ssm_tail_boundary(32, 16), Some(16), "strictly below");
        assert_eq!(ssm_tail_boundary(33, 16), Some(32));
        assert_eq!(ssm_tail_boundary(100, 0), None, "no division by zero");
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_host;
#[cfg(feature = "cuda")]
pub mod kernel;
pub mod mlx_int8;
#[cfg(feature = "cuda")]
pub mod registry;
#[cfg(feature = "cuda")]
pub mod stream;
