// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Helpers of `dsa_topk_radix_microtest`: rig-shaped score rows, the adversarial
//! sets, the host order the top-k selects in, and CUDA-event graph timing.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::GpuBackend;

// 2026-10-06: CUDA driver event API for kernel-only timing, declared as in
// `dsa_depth_decode_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-06: Graph replays timed per measurement.
const REPLAYS: usize = 51;

/// 2026-10-06: A CUDA driver status as a `Result`.
fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

/// 2026-10-06: GLM-5.3 `index_topk / index_kpool`, the select_k cap.
pub(crate) const SELECT_K: usize = 512;

/// 2026-10-06: Deterministic generator.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn pick(&mut self, from: &[f32]) -> f32 {
        from[((self.f() * from.len() as f32) as usize).min(from.len() - 1)]
    }
}

/// 2026-10-06: Rig-shaped scores for `p` pools: about 4 in 23 pools -FLT_MAX (not candidates),
/// the others `0 + w0 relu(x0) + ... + w7 relu(x7)` in f32, so a pool whose terms are all 0
/// scores exactly +0.0.
pub(crate) fn rig_scores(seed: u64, p: usize) -> Vec<f32> {
    let mut g = Lcg::new(seed);
    (0..p)
        .map(|_| {
            if g.f() < 4.0 / 23.0 {
                return -f32::MAX;
            }
            let mut acc = 0.0f32;
            for _ in 0..8 {
                let w = g.f() * 0.3;
                let x = g.f() * 2.0 - 1.0;
                acc += w * x.max(0.0);
            }
            acc
        })
        .collect()
}

/// 2026-10-06: A generator of one score row from a seed.
pub(crate) type Gen = Box<dyn Fn(u64) -> Vec<f32>>;

/// 2026-10-06: One named adversarial set.
fn case(name: &'static str, f: impl Fn(u64) -> Vec<f32> + 'static) -> (&'static str, Gen) {
    (name, Box::new(f))
}

/// 2026-10-06: The adversarial sets: name and row generator.
pub(crate) fn adversarial() -> Vec<(&'static str, Gen)> {
    let m = -f32::MAX;
    vec![
        case("all equal", |_| vec![0.75f32; 9_000]),
        case("all -FLT_MAX", move |_| vec![m; 7_000]),
        case("half -FLT_MAX", move |s| {
            let mut g = Lcg::new(s);
            (0..20_000)
                .map(|_| if g.f() < 0.5 { m } else { g.f() })
                .collect()
        }),
        case("+-0.0 mix", move |s| {
            let mut g = Lcg::new(s);
            (0..12_000).map(|_| g.pick(&[0.0, -0.0, 1e-3, m])).collect()
        }),
        // 2026-10-06: About 100 positive pools, so the threshold falls inside the zeros.
        case("+-0.0 at the threshold", move |s| {
            let mut g = Lcg::new(s);
            (0..12_000)
                .map(|i| {
                    if i % 120 == 7 {
                        0.5 + g.f()
                    } else {
                        g.pick(&[0.0, -0.0, m])
                    }
                })
                .collect()
        }),
        case("exactly select_k candidates", move |s| {
            let mut g = Lcg::new(s);
            let mut v = vec![m; 30_000];
            for i in 0..SELECT_K {
                v[(i * 58 + (s as usize % 50)) % 30_000] = 0.25 + g.f();
            }
            v
        }),
        case("P = select_k", |s| {
            let mut g = Lcg::new(s);
            (0..SELECT_K).map(|_| g.f()).collect()
        }),
        case("P = select_k + 1", |s| {
            let mut g = Lcg::new(s);
            (0..SELECT_K + 1).map(|_| g.f()).collect()
        }),
        case("P = 2049", |s| {
            let mut g = Lcg::new(s);
            (0..2_049).map(|_| g.pick(&[0.5, 0.25, 0.125, 1.0])).collect()
        }),
        case("P = 100003 (not a chunk multiple)", |s| rig_scores(s, 100_003)),
        case("threshold duplicates", move |s| {
            let mut g = Lcg::new(s);
            (0..50_000)
                .map(|_| g.pick(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, m]))
                .collect()
        }),
        case("denormals", |s| {
            let mut g = Lcg::new(s);
            (0..6_001)
                .map(|_| g.pick(&[1e-45, -1e-45, 3e-45, 0.0, -0.0, -3e-45]))
                .collect()
        }),
        case("wide spread, both signs", |s| {
            let mut g = Lcg::new(s);
            (0..133_120).map(|_| (g.f() - 0.5) * 1.0e6).collect()
        }),
    ]
}

/// 2026-10-06: Score descending (-0.0 == +0.0), then index ascending: the first `k` indices.
pub(crate) fn host_reference(scores: &[f32], k: usize) -> Vec<i32> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&i, &j| {
        scores[j]
            .partial_cmp(&scores[i])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });
    idx.into_iter().take(k).map(|i| i as i32).collect()
}

/// 2026-10-06: Median ms of one replay of the graph `f` captures on `s`, over `REPLAYS`
/// replays each bracketed by CUDA events, after 3 warm-up replays (as
/// `dsa_depth_decode_microtest`).
pub(crate) fn time_graph(
    g: &dyn GpuBackend,
    s: u64,
    f: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<f64> {
    g.begin_capture(s)?;
    if let Err(e) = f(s) {
        g.abort_capture_if_active(s);
        return Err(e);
    }
    let graph = g.end_capture(s)?;
    for _ in 0..3 {
        g.launch_graph(graph, s)?;
    }
    g.synchronize(s)?;
    let mut ev = vec![(0u64, 0u64); REPLAYS];
    // SAFETY: plain CUDA driver event calls; the events are created, used and destroyed here,
    // and the backend made the context current when it was built.
    unsafe {
        for e in ev.iter_mut() {
            check(cuEventCreate(&mut e.0, 0), "cuEventCreate")?;
            check(cuEventCreate(&mut e.1, 0), "cuEventCreate")?;
        }
    }
    for e in &ev {
        // SAFETY: as above.
        unsafe { check(cuEventRecord(e.0, s), "cuEventRecord(start)")? };
        g.launch_graph(graph, s)?;
        // SAFETY: as above.
        unsafe { check(cuEventRecord(e.1, s), "cuEventRecord(end)")? };
    }
    g.synchronize(s)?;
    let mut ms = Vec::with_capacity(REPLAYS);
    for e in &ev {
        let mut t: f32 = 0.0;
        // SAFETY: as above; `t` outlives the call and both events have completed.
        unsafe {
            check(cuEventElapsedTime(&mut t, e.0, e.1), "cuEventElapsedTime")?;
            cuEventDestroy_v2(e.0);
            cuEventDestroy_v2(e.1);
        }
        ms.push(f64::from(t));
    }
    g.destroy_graph(graph)?;
    ms.sort_by(|x, y| x.total_cmp(y));
    Ok(ms[ms.len() / 2])
}
