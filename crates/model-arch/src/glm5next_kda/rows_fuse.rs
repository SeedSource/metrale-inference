// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `METRALE_GLM_KDA_ROWS_FUSE=1`: the fused verify walk ([`Glm5NextKdaLayer::
//! snap_walk`], `METRALE_GLM_KDA_SNAP_FUSE=1`) launches its `k` recurrent rows as one
//! `kda_recurrent_decode_bf16_smem_rows` instead of one launch per row.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Off (the default), or when the walk's plan has a copy tail, `k` is outside `1..=4` or the
//!   PTX lacks the kernel, nothing here runs and `snap_walk` is unchanged.
//! - On, the `k` conv rows launch first, in row order, exactly as in the walk (each conv row
//!   reads only the previous conv row's state, never a recurrent output), then one recurrent
//!   launch reads the plan's row-0 input state once and writes each row's state to that row's
//!   plan output, with the per-row kernel's expressions on the same floats. So every snapshot
//!   slot, the live state and every output end bit-identical to the walk; the state is read
//!   from global once per layer instead of `k` times, and `k - 1` launches are gone.
//!   `kda_rows_fuse_microtest` checks it at every accept outcome.
//! - Fixed pointers, no host synchronisation, no allocation: valid under CUDA graph capture.
//!
//! Why (c24 C=1 nsys, SeedHQ/seed-skills#80, 2026-10-08): the three per-row launches take
//! 17.9 + 10.2 + 7.4 us per KDA layer and read 3 x 2.1 MB of FP32 state.

use std::sync::atomic::{AtomicU8, Ordering};

use super::*;

/// 2026-10-08: 0 = read `METRALE_GLM_KDA_ROWS_FUSE` once, 1 = forced off, 2 = forced on
/// ([`Glm5NextKdaLayer::force_rows_fuse`], microtests only).
static FORCE: AtomicU8 = AtomicU8::new(0);

/// 2026-10-08: Whether a `METRALE_GLM_KDA_ROWS_FUSE` value asks for the fused rows: `1` only.
fn rows_fuse_requested(v: Option<&str>) -> bool {
    v == Some("1")
}

fn kda_rows_fuse() -> bool {
    match FORCE.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        rows_fuse_requested(std::env::var("METRALE_GLM_KDA_ROWS_FUSE").ok().as_deref())
    })
}

/// 2026-10-08: The most rows one launch takes (the kernel's `state_out0..3`).
const MAX_ROWS: usize = 4;

impl Glm5NextKdaLayer {
    /// 2026-10-08: Forces `METRALE_GLM_KDA_ROWS_FUSE` on or off for this process, so
    /// `kda_rows_fuse_microtest` can run both arms; `None` returns to the environment.
    pub fn force_rows_fuse(on: Option<bool>) {
        FORCE.store(on.map_or(0, |b| if b { 2 } else { 1 }), Ordering::Relaxed);
    }

    /// 2026-10-08: The fused-rows walk for `plan`, or `Ok(false)` having launched nothing when
    /// the lever is off or the plan is outside its contract (the caller then walks per row).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rows_walk(
        &self,
        gpu: &dyn GpuBackend,
        row0: usize,
        plan: &[SnapRowIo],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<bool> {
        let k = plan.len();
        if !kda_rows_fuse()
            || self.kernels.recurrent_rows_io.0 == 0
            || !(1..=MAX_ROWS).contains(&k)
            || plan.iter().any(|io| io.copy_to.is_some())
        {
            return Ok(false);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_KDA_ROWS_FUSE: ENGAGED - the KDA verify walk runs its recurrent rows \
                 in one launch (bit-identical; kernels/gb10/common/kda_snap_fuse.cu)"
            );
        });
        for (t, io) in plan.iter().enumerate() {
            self.conv_row_io(gpu, row0 + t, io.c_in, io.c_out, ws, stream)?;
        }
        let c = &self.cfg;
        let (qkv, cd, d) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        let vpb = KDA_V_PER_BLOCK.min(d);
        let smem = (3 * d + vpb * (d + 1)) * 4;
        let outs: Vec<DevicePtr> = (0..MAX_ROWS).map(|t| plan[t.min(k - 1)].h_out).collect();
        KernelLaunch::new(gpu, self.kernels.recurrent_rows_io)
            .grid([c.heads as u32, (d / vpb) as u32, 1])
            .block([vpb as u32, 1, 1])
            .shared_mem(smem as u32)
            .arg_ptr(ws.conv_out.offset(row0 * cd * 2))
            .arg_ptr(ws.gate.offset(row0 * qkv * 4))
            .arg_ptr(ws.beta.offset(row0 * c.heads * 4))
            .arg_ptr(plan[0].h_in)
            .arg_ptr(outs[0])
            .arg_ptr(outs[1])
            .arg_ptr(outs[2])
            .arg_ptr(outs[3])
            .arg_ptr(ws.core.offset(row0 * qkv * 4))
            .arg_u32(c.heads as u32)
            .arg_u32(d as u32)
            .arg_f32(1.0 / (d as f32).sqrt())
            .arg_u32(vpb as u32)
            .arg_u32(k as u32)
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .launch(stream)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lever_is_on_only_for_one() {
        assert!(rows_fuse_requested(Some("1")));
        for v in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some(" 1"),
            Some("2"),
        ] {
            assert!(!rows_fuse_requested(v), "{v:?}");
        }
    }
}
