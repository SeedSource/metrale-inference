// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Host-side FFI to FlashKDA's chunked KDA forward (vendor/flashkda, MIT) through
//! our C-ABI wrapper `cuda/flashkda_kda_fwd.cu`: one sequence, head dim 128, FP32 recurrent
//! state read and written in place.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - The FFI exists only under `cfg(metrale_flashkda)`, which `build.rs` sets only when
//!   `FLASHKDA_CUTLASS_HOME` is set at build time. Without it [`available`] is false,
//!   [`workspace_bytes`] still answers (it is pure arithmetic) and [`fwd_fp32_state`] returns an
//!   error.
//! - The only caller is the opt-in GLM-5.3 KDA prefill
//!   (`METRALE_GLM_KDA_PREFILL_FLASHKDA=1`, metrale-model-arch `glm5next_kda/prefill_flashkda.rs`).
//!
//! Numerics (upstream's, not ours): the library keeps the recurrent state in BF16 shared memory
//! between its 16-row chunks (FP32 only at load and store), L2-normalises q and k itself
//! (`rsqrt(sum + 1e-6)`), applies the gate and beta sigmoids with `tanh.approx`, rounds `scale`
//! to BF16, and writes a BF16 output.

use anyhow::{Result, bail};

/// 2026-10-03: The head dim FlashKDA is built for (`launch_fwd<128, ..>`).
pub const HEAD_DIM: usize = 128;
/// 2026-10-03: FlashKDA's chunk length.
pub const CHUNK: usize = 16;
/// 2026-10-03: Upstream's per-(16-row tile, head) workspace record in bytes (see
/// `cuda/flashkda_kda_fwd.cu`).
const PER_TILE_BYTES: usize = 3 * CHUNK * HEAD_DIM * 2 + HEAD_DIM * 4 + 2 * CHUNK * CHUNK * 2;
/// 2026-10-03: The tile-prefix trailer upstream appends (2 int32, rounded up to 128 B).
const TRAILER_BYTES: usize = 128;

#[cfg(metrale_flashkda)]
unsafe extern "C" {
    fn metrale_flashkda_workspace_bytes(rows: i64, heads: i64) -> i64;
    #[allow(clippy::too_many_arguments)]
    fn metrale_flashkda_fwd_fp32_state(
        q: *const std::ffi::c_void,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        g: *const std::ffi::c_void,
        beta_t: *const std::ffi::c_void,
        state: *mut std::ffi::c_void,
        scale: f32,
        out: *mut std::ffi::c_void,
        workspace: *mut std::ffi::c_void,
        workspace_bytes: i64,
        rows: i32,
        heads: i32,
        a_log: *const f32,
        dt_bias: *const f32,
        lower_bound: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

/// 2026-10-03: Whether FlashKDA was compiled in (`FLASHKDA_CUTLASS_HOME` was set at build time).
pub fn available() -> bool {
    cfg!(metrale_flashkda)
}

/// 2026-10-03: Device bytes of workspace one [`fwd_fp32_state`] call over `rows` rows with
/// `heads` heads needs; 0 when either is 0. Mirrors `metrale_flashkda_workspace_bytes` (checked
/// against it when the library is built).
pub fn workspace_bytes(rows: usize, heads: usize) -> usize {
    if rows == 0 || heads == 0 {
        return 0;
    }
    let ours = heads * rows.div_ceil(CHUNK) * PER_TILE_BYTES + TRAILER_BYTES;
    #[cfg(metrale_flashkda)]
    {
        // SAFETY: pure arithmetic in the wrapper; no device access.
        let theirs = unsafe { metrale_flashkda_workspace_bytes(rows as i64, heads as i64) };
        debug_assert_eq!(theirs as usize, ours, "FlashKDA workspace formula drifted");
    }
    ours
}

/// 2026-10-03: Device pointers and geometry for one [`fwd_fp32_state`] call. All buffers are
/// row-major and contiguous:
/// `q`, `k`, `v`, `g`, `out`: BF16 `[rows, heads, 128]` (`g` before the gate activation);
/// `beta_t`: BF16 `[heads, rows]` (logits); `state`: FP32 `[heads, 128 (v), 128 (k)]`, read and
/// written in place; `a_log`: FP32 `[heads]`; `dt_bias`: FP32 `[heads, 128]`.
#[derive(Clone, Copy, Debug)]
pub struct FwdArgs {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub g: u64,
    pub beta_t: u64,
    pub state: u64,
    pub out: u64,
    pub workspace: u64,
    pub workspace_bytes: usize,
    pub rows: usize,
    pub heads: usize,
    pub a_log: u64,
    pub dt_bias: u64,
    pub scale: f32,
    pub lower_bound: f32,
}

/// 2026-10-03: One FlashKDA forward on `stream` (a raw `CUstream`). Errors when the library is
/// not built, an argument is out of range, or the launch fails.
pub fn fwd_fp32_state(a: &FwdArgs, stream: u64) -> Result<()> {
    if a.rows == 0 || a.rows > i32::MAX as usize || a.heads == 0 || a.heads > i32::MAX as usize {
        bail!("FlashKDA: rows {} / heads {} out of range", a.rows, a.heads);
    }
    let need = workspace_bytes(a.rows, a.heads);
    if a.workspace_bytes < need {
        bail!(
            "FlashKDA: workspace of {} B is smaller than the {need} B {} rows x {} heads need",
            a.workspace_bytes,
            a.rows,
            a.heads
        );
    }
    #[cfg(metrale_flashkda)]
    {
        use std::ffi::c_void;
        let p = |x: u64| x as usize as *const c_void;
        let m = |x: u64| x as usize as *mut c_void;
        // SAFETY: the caller passes device pointers sized as `FwdArgs` documents; the wrapper
        // checks the scalar arguments and the workspace size again.
        let rc = unsafe {
            metrale_flashkda_fwd_fp32_state(
                p(a.q),
                p(a.k),
                p(a.v),
                p(a.g),
                p(a.beta_t),
                m(a.state),
                a.scale,
                m(a.out),
                m(a.workspace),
                a.workspace_bytes as i64,
                a.rows as i32,
                a.heads as i32,
                a.a_log as usize as *const f32,
                a.dt_bias as usize as *const f32,
                a.lower_bound,
                stream as usize as *mut c_void,
            )
        };
        if rc != 0 {
            bail!("FlashKDA forward failed: cudaError {rc}");
        }
        Ok(())
    }
    #[cfg(not(metrale_flashkda))]
    {
        let _ = stream;
        bail!("FlashKDA was not built; set FLASHKDA_CUTLASS_HOME at build time")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-03: Upstream's per-tile record is 13,824 B; GLM-5.3 TP2 (32 heads) at 4,096 rows
    /// needs 256 tiles per head.
    #[test]
    fn workspace_matches_upstream_formula() {
        assert_eq!(PER_TILE_BYTES, 13_824);
        assert_eq!(workspace_bytes(0, 32), 0);
        assert_eq!(workspace_bytes(16, 1), 13_824 + 128);
        assert_eq!(workspace_bytes(17, 1), 2 * 13_824 + 128);
        assert_eq!(workspace_bytes(4096, 32), 32 * 256 * 13_824 + 128);
    }

    #[test]
    fn refuses_bad_shapes_before_any_launch() {
        let a = FwdArgs {
            q: 0,
            k: 0,
            v: 0,
            g: 0,
            beta_t: 0,
            state: 0,
            out: 0,
            workspace: 0,
            workspace_bytes: 0,
            rows: 0,
            heads: 32,
            a_log: 0,
            dt_bias: 0,
            scale: 1.0,
            lower_bound: -5.0,
        };
        assert!(fwd_fp32_state(&a, 0).is_err());
        let a = FwdArgs { rows: 64, ..a };
        // 2026-10-03: A zero-byte workspace is refused before the FFI is reached.
        assert!(fwd_fp32_state(&a, 0).is_err());
    }
}
