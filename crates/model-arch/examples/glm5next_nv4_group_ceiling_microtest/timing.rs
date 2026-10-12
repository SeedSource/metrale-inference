// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Launch timing for `glm5next_nv4_group_ceiling_microtest`: CUDA events around one
//! arm after an untimed prelude (L2 flush or a chained GEMV), medians over interleaved arms.
//!
//! Owner: model-arch examples.

use anyhow::{Result, bail};

// CUDA driver event API, declared as in `glm5next_l2_prefetch_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// Timed reps per arm (median) and untimed warm-up reps.
pub const REPS: usize = 60;
pub const WARMUP: usize = 5;

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

/// Two timing events.
pub struct Events([u64; 2]);

impl Events {
    pub fn new() -> Result<Self> {
        let mut e = [0u64; 2];
        for x in e.iter_mut() {
            // SAFETY: plain driver call writing one event handle into `x`; the backend made
            // the context current when it was built.
            check(unsafe { cuEventCreate(x, 0) }, "cuEventCreate")?;
        }
        Ok(Self(e))
    }

    fn rec(&self, i: usize, s: u64) -> Result<()> {
        // SAFETY: records an event this struct created on stream `s`.
        check(unsafe { cuEventRecord(self.0[i], s) }, "cuEventRecord")
    }

    fn us(&self) -> Result<f64> {
        let mut ms: f32 = 0.0;
        // SAFETY: both events were created and recorded by this struct; `ms` outlives the call.
        unsafe {
            check(cuEventSynchronize(self.0[1]), "cuEventSynchronize")?;
            check(
                cuEventElapsedTime(&mut ms, self.0[0], self.0[1]),
                "cuEventElapsedTime",
            )?;
        }
        Ok(ms as f64 * 1e3)
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        for &e in &self.0 {
            // SAFETY: destroys an event this struct created; errors are ignored on drop.
            unsafe {
                cuEventDestroy_v2(e);
            }
        }
    }
}

pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// Median microseconds of each arm on stream `s`: per rep and arm, `pre` (untimed: an L2
/// flush or a chained GEMV), then the arm between two events. Arms are interleaved per rep.
pub fn time_arms(
    ev: &Events,
    s: u64,
    pre: &dyn Fn(u64) -> Result<()>,
    arms: &[&dyn Fn(u64) -> Result<()>],
) -> Result<Vec<f64>> {
    let mut t = vec![Vec::with_capacity(REPS); arms.len()];
    for rep in 0..WARMUP + REPS {
        for (i, arm) in arms.iter().enumerate() {
            pre(s)?;
            ev.rec(0, s)?;
            arm(s)?;
            ev.rec(1, s)?;
            let us = ev.us()?;
            if rep >= WARMUP {
                t[i].push(us);
            }
        }
    }
    Ok(t.into_iter().map(median).collect())
}
