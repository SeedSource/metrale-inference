// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The full-width prefill window (`wide.rs`) with the DSA pool cache
//! (`METRALE_GLM_DSA_POOL_CACHE=1`): where its `k`/`gate` rows go. Split out of `wide.rs`
//! (500-line cap).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - The window's rows go to ring slots; `ensure_room(k)` (run by `decode_k_wide` first)
//!   checked the ring holds them with every row the next compress needs.
//! - When the ring wraps inside the window the keys and gates are produced into the staging
//!   buffer (`arena.q_idx`, free until `wq_b` writes it, on the same stream) and copied to
//!   their two ring runs, so the GEMM runs at the same M and its bytes do not change.
//! - Lever off: no clamp, no staging; the cache rows at `row_offset(len)` as before.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::state::Glm5NextDsaState;
use super::Glm5NextDsaLayer;

/// 2026-10-06: Where one window writes its keys and gates, and how they reach the ring.
pub(super) struct PoolWindow {
    pub(super) k_rows: DevicePtr,
    pub(super) gate_rows: DevicePtr,
    pos0: usize,
    /// 2026-10-06: Ring runs `(first row, rows)` when staged (the ring wraps); empty otherwise.
    staged_runs: Vec<(usize, usize)>,
}

impl Glm5NextDsaLayer {
    /// 2026-10-06: Before the window's projections: clamp the device watermark after a rewind,
    /// record the ring write through `len + k`, and pick the destinations (`stage` holds at
    /// least `2 * k * index_head_dim * 2` bytes).
    pub(super) fn pool_window_begin(
        &self,
        gpu: &dyn GpuBackend,
        st: &mut Glm5NextDsaState,
        stage: DevicePtr,
        k: usize,
        stream: u64,
    ) -> Result<PoolWindow> {
        let d = self.cfg.index_head_dim;
        let pos0 = st.len();
        self.pool_clamp_before_write(gpu, st, pos0, stream)?;
        st.note_ring_write(pos0 + k);
        let runs = st.ring_runs(pos0, k);
        let off = st.row_offset(pos0);
        Ok(if runs.len() > 1 {
            PoolWindow {
                k_rows: stage,
                gate_rows: stage.offset(k * d * 2),
                pos0,
                staged_runs: runs,
            }
        } else {
            PoolWindow {
                k_rows: st.k_normed.offset(off),
                gate_rows: st.gate.offset(off),
                pos0,
                staged_runs: Vec::new(),
            }
        })
    }
}

impl PoolWindow {
    /// 2026-10-06: After the projections: copy staged rows to their ring runs (no-op unless
    /// staged).
    pub(super) fn scatter(
        &self,
        gpu: &dyn GpuBackend,
        st: &Glm5NextDsaState,
        d: usize,
        stream: u64,
    ) -> Result<()> {
        for &(r0, n) in &self.staged_runs {
            let dst = st.row_offset(self.pos0 + r0);
            let (src_k, src_g) = (
                self.k_rows.offset(r0 * d * 2),
                self.gate_rows.offset(r0 * d * 2),
            );
            gpu.copy_d2d_async(src_k, st.k_normed.offset(dst), n * d * 2, stream)?;
            gpu.copy_d2d_async(src_g, st.gate.offset(dst), n * d * 2, stream)?;
        }
        Ok(())
    }
}
