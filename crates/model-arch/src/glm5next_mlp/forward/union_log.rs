// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_GLM_MOE_UNION_LOG=N` (observe only): how much expert-weight re-reading the
//! row-batched routed MoE does once a call has more rows than one sweep holds
//! (`MOE_ROW_BATCH_MAX_ROWS`). Every Nth such call copies the routed expert ids to the host
//! (synchronous D2H, debug only) and counts, over the experts this rank owns, the distinct
//! experts of each dispatched sub-group (`U_a`, `U_b`, ...) and of all rows (`U_all`). An expert
//! picked in two sub-groups is streamed once per sweep; `reread_frac = 1 - U_all / sum(U_sub)`.
//! Never touches a device computation; skipped under graph capture.
//!
//! 2026-10-11: `METRALE_GLM_MOE_UNION_LOG_MIN_ROWS=k`: calls of at least `k` rows qualify too
//! (e.g. a 3- or 8-row single-sequence verify), and every logged call also prints one
//! `MOE UNION CALL:` line: layer, rows, slots (`rows * top_k`), `u_local` (distinct experts this
//! rank owns), `u_global` (distinct experts over all ranks: the router's ids are global ids,
//! `0..num_experts`), and `bytes_rank = u_local * expert_bytes(cfg)` (one expert's NVFP4
//! gate+up+down as stored on this rank). The layer index comes from [`set_layer`], which the
//! GLM layer calls before its MoE site. A graph replay never runs this host hook.

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

/// 2026-10-11: A row count of at least 1, else `None` (unset, `0`, unparsable).
fn parse_min_rows(v: Option<&str>) -> Option<usize> {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&k| k > 0)
}

/// 2026-10-11: `METRALE_GLM_MOE_UNION_LOG_MIN_ROWS=k`, read once.
fn min_rows() -> Option<usize> {
    static M: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        parse_min_rows(
            std::env::var("METRALE_GLM_MOE_UNION_LOG_MIN_ROWS")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-10-11: Whether a call of `rows` rows is counted: wider than one batchm sweep (the
/// original rule), or at least `min_rows` when that is set.
fn qualifies(rows: usize, min_rows: Option<usize>) -> bool {
    rows > MOE_ROW_BATCH_MAX_ROWS || min_rows.is_some_and(|k| rows >= k)
}

/// 2026-10-11: One routed expert's bytes as stored on this rank: gate, up (`[I, hidden]`) and
/// down (`[hidden, I]`) NVFP4, each packed E2M1 (`I * hidden / 2`) plus E4M3 block scales
/// (`I * hidden / 16`), at this rank's `moe_intermediate` (half the checkpoint's under
/// `METRALE_GLM_EXPERT_TP=1`).
pub(super) fn expert_bytes(cfg: &Glm5NextMlpConfig) -> usize {
    let e = cfg.moe_intermediate * cfg.hidden;
    3 * (e / 2 + e / 16)
}

/// 2026-10-11: `(u_local, u_global)`: distinct ids this rank owns, distinct ids overall.
pub(super) fn call_unions(ids: &[Vec<u32>], owns: impl Fn(u32) -> bool) -> (usize, usize) {
    let all: BTreeSet<u32> = ids.iter().flatten().copied().collect();
    let local = all.iter().filter(|&&e| owns(e)).count();
    (local, all.len())
}

thread_local! {
    static LAYER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// 2026-10-11: The layer whose MoE site runs next on this thread, for the `MOE UNION CALL`
/// line. No-op unless the union log is on.
pub(crate) fn set_layer(idx: usize) {
    if every().is_some() {
        LAYER.with(|l| l.set(Some(idx)));
    }
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
    let min = min_rows();
    if !qualifies(rows, min) || gpu.stream_is_capturing(stream) {
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
    if min.is_some() {
        let (u_local, u_global) = call_unions(&per_row, |e| own.contains(&(e as usize)));
        let eb = expert_bytes(cfg);
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| {
            tracing::info!(
                "MOE UNION LOG: bytes_per_expert_rank={eb} (NVFP4 gate+up+down packed + E4M3 \
                 scales at moe_intermediate={} hidden={}); u_local over this rank's experts \
                 {:?}, u_global over all {} (ids are global)",
                cfg.moe_intermediate,
                cfg.hidden,
                own,
                cfg.num_experts
            );
        });
        let layer = LAYER
            .with(|l| l.get())
            .map_or("?".to_string(), |l| l.to_string());
        tracing::info!(
            "MOE UNION CALL: layer={layer} rows={rows} slots={} u_local={u_local} \
             u_global={u_global} sum_sub_U={sum} bytes_rank={}",
            rows * cfg.top_k,
            u_local * eb
        );
    }
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

    /// 2026-10-11: `MIN_ROWS` parsing and which calls qualify.
    #[test]
    fn min_rows_parse_and_qualify() {
        for (raw, want) in [
            (None, None),
            (Some(""), None),
            (Some("0"), None),
            (Some("x"), None),
            (Some(" 3 "), Some(3)),
            (Some("1"), Some(1)),
        ] {
            assert_eq!(parse_min_rows(raw), want, "{raw:?}");
        }
        let wide = MOE_ROW_BATCH_MAX_ROWS + 1;
        assert!(!qualifies(3, None));
        assert!(qualifies(wide, None));
        assert!(qualifies(3, Some(3)));
        assert!(!qualifies(2, Some(3)));
        assert!(qualifies(wide, Some(64)));
        assert!(qualifies(1, Some(1)));
    }

    /// 2026-10-11: u_local counts distinct owned ids, u_global distinct ids over all rows.
    #[test]
    fn call_unions_fixture() {
        // 2026-10-11: A 3-row verify, top_k 4, rank owns ids < 8 (the EP lower half).
        let ids = vec![vec![1, 2, 9, 10], vec![2, 3, 10, 11], vec![1, 3, 9, 12]];
        assert_eq!(call_unions(&ids, |e| e < 8), (3, 7));
        assert_eq!(call_unions(&ids, |_| true), (7, 7));
        assert_eq!(call_unions(&ids, |_| false), (0, 7));
        assert_eq!(call_unions(&[], |_| true), (0, 0));
        // 2026-10-11: Eight identical rows collapse to top_k.
        assert_eq!(call_unions(&vec![vec![4, 5, 6, 7]; 8], |_| true), (4, 4));
    }

    /// 2026-10-11: One expert as stored per rank: 3 x I x hidden x 9/16 bytes. Expert-TP
    /// (I = 2048 / 2 = 1024, hidden 4096): 7,077,888 B.
    #[test]
    fn expert_bytes_from_shapes() {
        let cfg = Glm5NextMlpConfig {
            hidden: 4096,
            local_dense_intermediate: 0,
            moe_intermediate: 1024,
            local_shared_intermediate: 0,
            num_experts: 288,
            local_experts: 288,
            ep_rank: 0,
            top_k: 8,
            routed_scale: 1.0,
            renormalize: true,
            swiglu_limit: 0.0,
            router_bf16_ladder: false,
            tp_world_size: 2,
            ep_world_size: 2,
        };
        assert_eq!(expert_bytes(&cfg), 7_077_888);
        let full = Glm5NextMlpConfig {
            moe_intermediate: 2048,
            ..cfg
        };
        assert_eq!(expert_bytes(&full), 2 * 7_077_888);
    }

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
