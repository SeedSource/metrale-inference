// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `METRALE_GLM_NV4_TC`: GLM NVFP4 GEMVs on the tensor-core kernels.
//!
//! Owner: model-arch (GLM-5.3).
//! `METRALE_GLM_NV4_TC=1` routes every `dense_fp8::nv4_gemv_uncounted` launch (1..=16 rows,
//! `K % 128 == 0`) through `w4a16_gemv_tc8` (M <= 8) or `w4a16_gemv_tc16`
//! (`kernels/gb10/common/w4a16_gemv_tc.cu`, routed by `ops::gemv_tc::tc_kernel`) instead of the
//! CUDA-core `w4a16_gemv` / `w4a16_gemv_batch*` tiers. The tensor-core work per weight byte does
//! not depend on M. Each token is its own MMA row, and the k-block order and warp reduction are
//! the same at every M, so a row's bits do not depend on M or on the other rows (gate
//! `examples/glm5next_nv4_tc_microtest.rs`). NOT byte-identical to the CUDA-core tiers: the
//! FP32 sums run in another order. A shape the route declines (`K % 128 != 0`,
//! `METRALE_NO_W4A16_TC`, kernels missing) keeps its CUDA-core tier at every M. Read once.
//!
//! 2026-10-08: values. An integer `1..=16` is the fewest rows routed (`=1`: every launch; it
//! lost 2 T=0 hardmode rows, c23dtqual2 154 vs 156). `multi` routes only launches inside a
//! multi-sequence decode or batched verify step of at least 2 sequences ([`MultiSeqScope`],
//! `steps/multi_seq.rs`, `steps/verify_multi.rs`): a single sequence's decode, verify and
//! prefill keep the car's CUDA-core bits, and only C>1 batched rows move (their bits then
//! depend on the batch). The scope is read when a kernel is launched or captured.

use std::cell::Cell;
use std::sync::{Once, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops::gemv_tc;
use metrale_model_layers::weight_map::QuantizedWeight;

/// What `METRALE_GLM_NV4_TC` routes (read once): `Floor(m)` for an integer `m` in 1..=16
/// (launches of at least `m` rows), `Multi` for `multi`; `None` when unset or any other value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nv4Tc {
    Floor(usize),
    Multi,
}

/// The parsed `METRALE_GLM_NV4_TC` (see [`Nv4Tc`]).
pub fn nv4_tc() -> Option<Nv4Tc> {
    static E: OnceLock<Option<Nv4Tc>> = OnceLock::new();
    *E.get_or_init(|| parse(std::env::var("METRALE_GLM_NV4_TC").ok().as_deref()))
}

fn parse(v: Option<&str>) -> Option<Nv4Tc> {
    match v?.trim() {
        "multi" => Some(Nv4Tc::Multi),
        x => x
            .parse::<usize>()
            .ok()
            .filter(|m| (1..=16).contains(m))
            .map(Nv4Tc::Floor),
    }
}

thread_local! {
    static IN_MULTI: Cell<bool> = const { Cell::new(false) };
}

/// Marks a multi-sequence step of `seqs` sequences on this thread for [`Nv4Tc::Multi`] (only
/// when `seqs >= 2`); the previous value comes back on drop, so nesting is safe.
pub struct MultiSeqScope(bool);

impl MultiSeqScope {
    pub fn enter(seqs: usize) -> Self {
        Self(IN_MULTI.with(|c| c.replace(c.get() || seqs >= 2)))
    }
}

impl Drop for MultiSeqScope {
    fn drop(&mut self) {
        IN_MULTI.with(|c| c.set(self.0));
    }
}

/// Launches `C[m, n] = A[m, k] @ dequant(q)^T` on the tensor-core kernel when the lever takes
/// this launch ([`Nv4Tc`]) and the route takes the shape: `Ok(true)` when it launched,
/// `Ok(false)` to keep the CUDA-core tier. No allocation or sync, so it is capture-safe.
#[allow(clippy::too_many_arguments)]
pub fn try_launch(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    let take = match nv4_tc() {
        None => false,
        Some(Nv4Tc::Floor(min_m)) => m >= min_m,
        Some(Nv4Tc::Multi) => IN_MULTI.with(Cell::get),
    };
    if !take {
        return Ok(false);
    }
    let Some((h, grid_x)) = gemv_tc::tc_kernel(gpu, m as u32, n as u32, k as u32) else {
        return Ok(false);
    };
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_NV4_TC={}: ENGAGED - GLM NVFP4 GEMVs (K % 128 == 0) run \
             w4a16_gemv_tc8/tc16 (first: {m} rows, {n}x{k}); NOT byte-identical to the \
             CUDA-core tiers",
            std::env::var("METRALE_GLM_NV4_TC").unwrap_or_default()
        );
    });
    KernelLaunch::new(gpu, h)
        .grid([grid_x, 1, 1])
        .block([gemv_tc::TC_BLOCK, 1, 1])
        .arg_ptr(a)
        .arg_ptr(q.weight)
        .arg_ptr(q.weight_scale)
        .arg_f32(q.weight_scale_2)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_parse_and_the_scope_needs_two_sequences() {
        assert_eq!(parse(Some("1")), Some(Nv4Tc::Floor(1)));
        assert_eq!(parse(Some("4")), Some(Nv4Tc::Floor(4)));
        assert_eq!(parse(Some("multi")), Some(Nv4Tc::Multi));
        assert_eq!(parse(Some("0")), None);
        assert_eq!(parse(Some("17")), None);
        assert_eq!(parse(None), None);
        let get = || IN_MULTI.with(Cell::get);
        {
            let _one = MultiSeqScope::enter(1);
            assert!(!get());
            {
                let _two = MultiSeqScope::enter(2);
                assert!(get());
                let _inner = MultiSeqScope::enter(1);
                assert!(get());
            }
            assert!(!get());
        }
        assert!(!get());
    }
}
