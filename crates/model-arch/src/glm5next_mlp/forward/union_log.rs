//! 2026-10-10: `METRALE_GLM_MOE_UNION_LOG=N` (observe only): how much expert-weight re-reading the
//! row-batched routed MoE does once a call has more rows than one sweep holds
//! (`MOE_ROW_BATCH_MAX_ROWS`). Every Nth such call copies the routed expert ids to the host
//! (synchronous D2H, debug only) and counts, over the experts this rank owns, the distinct
//! experts of each dispatched sub-group (`U_a`, `U_b`, ...) and of all rows (`U_all`). An expert
//! picked in two sub-groups is streamed once per sweep; `reread_frac = 1 - U_all / sum(U_sub)`.
//! Never touches a device computation; skipped under graph capture.

use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{Glm5NextMlpConfig, MOE_ROW_BATCH_MAX_ROWS};

/// Logged calls between two `MOE UNION LOG` lines.
const LINE_EVERY: u64 = 64;

/// `METRALE_GLM_MOE_UNION_LOG=N`: log every Nth wide call; `None` when unset, `0` or unparsable.
fn every() -> Option<u64> {
    static E: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("METRALE_GLM_MOE_UNION_LOG")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
    })
}

/// Per-sub-group unions (owned experts only), their sum, the union over all rows, and
/// `reread_frac = 1 - union / sum` (0 when the sum is 0). `ids[r]` is row `r`'s top-k expert ids;
/// `groups` are `(start, width)` row ranges as dispatched.
pub(super) fn union_stats(
    ids: &[Vec<u32>],
    groups: &[(usize, usize)],
    owns: impl Fn(u32) -> bool,
) -> (Vec<usize>, usize, usize, f64) {
    let mut subs = Vec::with_capacity(groups.len());
    let mut all = BTreeSet::new();
    for &(start, width) in groups {
        let mut s = BTreeSet::new();
        for row in ids.iter().skip(start).take(width) {
            for &e in row.iter().filter(|&&e| owns(e)) {
                s.insert(e);
                all.insert(e);
            }
        }
        subs.push(s.len());
    }
    let sum: usize = subs.iter().sum();
    let frac = if sum == 0 {
        0.0
    } else {
        1.0 - all.len() as f64 / sum as f64
    };
    (subs, sum, all.len(), frac)
}

#[derive(Default)]
struct Acc {
    logged: u64,
    rows: u64,
    sum_sub: u64,
    union: u64,
    hist: std::collections::BTreeMap<usize, u64>,
}

static CALLS: AtomicU64 = AtomicU64::new(0);
static ACC: Mutex<Acc> = Mutex::new(Acc {
    logged: 0,
    rows: 0,
    sum_sub: 0,
    union: 0,
    hist: std::collections::BTreeMap::new(),
});

/// Hook, called right after the router top-k. No-op (no allocation, no sync) when the lever is
/// unset, `rows <= MOE_ROW_BATCH_MAX_ROWS`, or `stream` is capturing a graph. `ids` holds
/// `rows * top_k` I32 expert ids; `groups` are the sub-groups `forward_moe` dispatches.
pub(super) fn maybe_log(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    ids: DevicePtr,
    rows: usize,
    groups: &[(usize, usize)],
    stream: u64,
) {
    let Some(n) = every() else { return };
    if rows <= MOE_ROW_BATCH_MAX_ROWS || gpu.stream_is_capturing(stream) {
        return;
    }
    if !CALLS.fetch_add(1, Ordering::Relaxed).is_multiple_of(n) {
        return;
    }
    let mut raw = vec![0u8; rows * cfg.top_k * 4];
    if gpu.synchronize(stream).is_err() || gpu.copy_d2h(ids, &mut raw).is_err() {
        return;
    }
    let per_row: Vec<Vec<u32>> = raw
        .chunks_exact(cfg.top_k * 4)
        .map(|r| {
            r.chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
        .collect();
    // Ownership as `local_expert_range` decides it (the range `u_eid` is built over).
    let own = cfg.local_expert_range();
    let (_, sum, all, _) = union_stats(&per_row, groups, |e| own.contains(&(e as usize)));
    let Ok(mut a) = ACC.lock() else { return };
    a.logged += 1;
    a.rows += rows as u64;
    a.sum_sub += sum as u64;
    a.union += all as u64;
    *a.hist.entry(rows).or_default() += 1;
    if a.logged % LINE_EVERY == 0 {
        let c = a.logged as f64;
        let (s, u) = (a.sum_sub as f64 / c, a.union as f64 / c);
        let frac = if s > 0.0 { 1.0 - u / s } else { 0.0 };
        tracing::info!(
            "MOE UNION LOG: calls={} rows_mean={:.2} sum_sub_U_mean={:.2} union_U_mean={:.2} \
             reread_frac={:.4} rows_hist={:?}",
            a.logged,
            a.rows as f64 / c,
            s,
            u,
            frac,
            a.hist
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_stats_table() {
        // (name, ids, groups, owns-below, want subs, want sum, want union, want frac)
        let cases: Vec<(
            &str,
            Vec<Vec<u32>>,
            Vec<(usize, usize)>,
            u32,
            Vec<usize>,
            usize,
            usize,
            f64,
        )> = vec![
            (
                "disjoint",
                vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
                vec![(0, 2), (2, 2)],
                99,
                vec![4, 4],
                8,
                8,
                0.0,
            ),
            (
                "identical",
                vec![vec![1, 2]; 4],
                vec![(0, 2), (2, 2)],
                99,
                vec![2, 2],
                4,
                2,
                0.5,
            ),
            (
                "partial",
                vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]],
                vec![(0, 2), (2, 2)],
                99,
                vec![3, 3],
                6,
                5,
                1.0 - 5.0 / 6.0,
            ),
            (
                "ownership",
                vec![vec![0, 10], vec![0, 11], vec![1, 10], vec![1, 12]],
                vec![(0, 2), (2, 2)],
                5,
                vec![1, 1],
                2,
                2,
                0.0,
            ),
            (
                "none_owned",
                vec![vec![9, 10]],
                vec![(0, 1)],
                5,
                vec![0],
                0,
                0,
                0.0,
            ),
            (
                "three_groups",
                vec![vec![1]; 3],
                vec![(0, 1), (1, 1), (2, 1)],
                99,
                vec![1, 1, 1],
                3,
                1,
                2.0 / 3.0,
            ),
        ];
        for (name, ids, groups, below, subs, sum, un, frac) in cases {
            let owns = |e: u32| e < below;
            let (s, sm, u, f) = union_stats(&ids, &groups, owns);
            assert_eq!((s, sm, u), (subs, sum, un), "{name}");
            assert!((f - frac).abs() < 1e-12, "{name}: {f} vs {frac}");
        }
    }
}
