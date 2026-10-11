// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Cache key and LRU for multi-sequence decode CUDA graphs.
//!
//! A batched-decode graph bakes each row's SSM `h_state` / `conv_state` pointers, which are a
//! function of the row's SSM pool slot, so the key is the slot vector in batch order. A batch
//! can be a subset of the active set (the MTP bootstrap passes only the draftless sequences),
//! and rows `n..padded_n` use the dummy slot; keying on the slot vector keeps a graph captured
//! for one composition from replaying for another. A layer whose per-sequence state lives
//! outside the pool returns true from `graph_stale_on_new_sequence()`, and
//! `free_sequence_dispatch` then drops every batched graph whose key holds the freed slot.
//!
//! Multi-seq decode graphs are on unless `METRALE_NO_DECODE_GRAPHS_MULTISEQ=1` (decode_a2.rs).
//! Measured 2026-07-27: C=8 65.75 -> 67.6 (+2.8%), C=16 92.6 -> 95.6 (+3.2%), emitted-text SHA
//! unchanged, 2 reps per cell.
//!
//! Owner: model-engine decode graphs.
//! Invariants:
//! - A key from `batch_decode_graph_key` has one entry per row, `padded_n` long, or is
//!   `[padded_n]` for a model with no SSM layers.
//! - `insert_batch_decode_graph` never evicts a key pinned by an in-flight fed step.

use metrale_gpu_runtime::gpu::{GpuBackend, GraphHandle};
use std::collections::HashMap;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// 2026-09-25: Bound on cached batched-decode graphs (one per distinct slot vector): 16 plus
/// the decode-meta rows. At the cap a new key evicts the least-recently-used unpinned graph,
/// so the cap bounds graph memory and a new composition is still captured.
pub(super) fn batch_decode_graph_cap(decode_meta_rows: usize) -> usize {
    16 + decode_meta_rows
}

/// 2026-09-25: Insert `graph` at `key`, evicting the LRU entry when at `cap` and the key
/// is new. Returns handles the caller must `destroy_graph` (evicted LRU
/// and/or the previous occupant of `key`).
pub(super) fn lru_insert_graph(
    cache: &mut (HashMap<Vec<u32>, (GraphHandle, u64)>, u64),
    cap: usize,
    key: Vec<u32>,
    graph: GraphHandle,
) -> Vec<GraphHandle> {
    lru_insert_graph_pinned(cache, cap, key, graph, &[])
}

/// 2026-09-25: [`lru_insert_graph`] that never evicts a `pinned` key, the graph of a fed decode
/// step that may still be executing on the stream. With every entry pinned, the cache grows
/// past `cap`.
pub(super) fn lru_insert_graph_pinned(
    cache: &mut (HashMap<Vec<u32>, (GraphHandle, u64)>, u64),
    cap: usize,
    key: Vec<u32>,
    graph: GraphHandle,
    pinned: &[Vec<u32>],
) -> Vec<GraphHandle> {
    let mut drop = Vec::new();
    if cache.0.len() >= cap
        && !cache.0.contains_key(&key)
        && let Some(evict) = cache
            .0
            .iter()
            .filter(|(k, _)| !pinned.contains(k))
            .min_by_key(|(_, entry)| entry.1)
            .map(|(k, _)| k.clone())
        && let Some((old, _)) = cache.0.remove(&evict)
    {
        drop.push(old);
    }
    cache.1 += 1;
    let tick = cache.1;
    if let Some((old, _)) = cache.0.insert(key, (graph, tick)) {
        drop.push(old);
    }
    drop
}

/// 2026-09-25: Whether batches other than slots `[0..n)` with `n == padded_n` (an MTP bootstrap
/// subset, any `n < padded_n` batch) may be graphed. On unless `METRALE_NO_MTP_BOOT_GRAPH` is
/// set to any value, `0` included; off, those batches run eager. Read once per process.
pub(super) fn boot_graph_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_MTP_BOOT_GRAPH").is_none())
}

impl TransformerModel {
    /// 2026-09-25: Batched-decode graph key: each row's SSM pool slot in batch order,
    /// `padded_n` entries long (rows `n..padded_n` carry the dummy slot). A model with no
    /// SSM layers gets `[padded_n]`.
    ///
    /// `None` means run eager: a row has no SSM pool slot, or [`boot_graph_enabled`] is off
    /// and the batch is not slots `[0..n)` with `n == padded_n`.
    pub(super) fn batch_decode_graph_key(
        &self,
        seqs: &[&mut SequenceState],
        padded_n: usize,
    ) -> Option<Vec<u32>> {
        if self.config.num_ssm_layers() == 0 {
            return Some(vec![padded_n as u32]);
        }
        let n = seqs.len();
        let mut key: Vec<u32> = Vec::with_capacity(padded_n);
        for s in seqs.iter() {
            match s.ssm_slot_idx() {
                Some(idx) => key.push(idx as u32),
                None => {
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| {
                        tracing::warn!(
                            "multi-seq decode graphs OFF for this step: a sequence has no SSM \
                             pool slot, so its baked layer-state pointers are per-sequence"
                        );
                    });
                    return None;
                }
            }
        }
        let dummy = self.ssm_pool.dummy_slot() as u32;
        for _ in n..padded_n {
            key.push(dummy);
        }
        if !boot_graph_enabled()
            && (n != padded_n || key.iter().enumerate().any(|(i, &s)| s != i as u32))
        {
            return None;
        }
        Some(key)
    }

    /// 2026-09-25: Insert a freshly captured graph, evicting the least-recently-used
    /// entry at [`batch_decode_graph_cap`]. Keys pinned by an in-flight fed step
    /// (`pinned_graph_keys`) are never evicted.
    pub(super) fn insert_batch_decode_graph(
        &self,
        cache: &mut (HashMap<Vec<u32>, (GraphHandle, u64)>, u64),
        key: Vec<u32>,
        graph: GraphHandle,
    ) {
        let cap = batch_decode_graph_cap(self.buffers.decode_meta().rows());
        let pinned = self.pinned_graph_keys();
        for old in lru_insert_graph_pinned(cache, cap, key, graph, &pinned) {
            if let Err(e) = self.gpu.destroy_graph(old) {
                tracing::warn!("batched-decode graph evict: {e:#}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_gpu_runtime::gpu::GraphHandle;

    fn empty_cache() -> (HashMap<Vec<u32>, (GraphHandle, u64)>, u64) {
        (HashMap::new(), 0)
    }

    #[test]
    fn lru_insert_below_cap_drops_nothing() {
        let mut cache = empty_cache();
        let drop = lru_insert_graph(&mut cache, 2, vec![0], GraphHandle(1));
        assert!(drop.is_empty());
        assert_eq!(cache.0.len(), 1);
        let (stored, tick) = cache.0.get(&vec![0]).unwrap();
        assert_eq!(stored.0, 1);
        assert_eq!(*tick, 1);
        assert_eq!(cache.1, 1);
    }

    #[test]
    fn lru_insert_at_cap_evicts_oldest_not_the_new_key() {
        let mut cache = empty_cache();
        lru_insert_graph(&mut cache, 2, vec![0], GraphHandle(10));
        lru_insert_graph(&mut cache, 2, vec![1], GraphHandle(11));
        let drop = lru_insert_graph(&mut cache, 2, vec![2], GraphHandle(12));
        assert_eq!(drop.iter().map(|g| g.0).collect::<Vec<_>>(), vec![10]);
        assert!(cache.0.contains_key(&vec![1]));
        assert!(cache.0.contains_key(&vec![2]));
        assert!(!cache.0.contains_key(&vec![0]));
    }

    #[test]
    fn replacing_an_existing_key_destroys_the_old_handle_without_evicting_peers() {
        let mut cache = empty_cache();
        lru_insert_graph(&mut cache, 2, vec![0], GraphHandle(10));
        lru_insert_graph(&mut cache, 2, vec![1], GraphHandle(11));
        let drop = lru_insert_graph(&mut cache, 2, vec![0], GraphHandle(99));
        assert_eq!(drop.iter().map(|g| g.0).collect::<Vec<_>>(), vec![10]);
        assert_eq!(cache.0.len(), 2);
        assert_eq!(cache.0.get(&vec![0]).unwrap().0.0, 99);
        assert_eq!(cache.0.get(&vec![1]).unwrap().0.0, 11);
        assert_eq!(cache.0.get(&vec![0]).unwrap().1, 3, "replacement is MRU");

        let drop = lru_insert_graph(&mut cache, 2, vec![2], GraphHandle(12));
        assert_eq!(
            drop.iter().map(|handle| handle.0).collect::<Vec<_>>(),
            vec![11],
            "the untouched peer is LRU"
        );
        assert_eq!(cache.0.get(&vec![0]).unwrap().0.0, 99);
        assert_eq!(cache.0.get(&vec![2]).unwrap().0.0, 12);
        assert!(!cache.0.contains_key(&vec![1]));
    }

    #[test]
    fn a_pinned_key_is_never_the_eviction_victim() {
        let mut cache = empty_cache();
        lru_insert_graph(&mut cache, 2, vec![0], GraphHandle(10));
        lru_insert_graph(&mut cache, 2, vec![1], GraphHandle(11));
        // 2026-09-25: [0] is the LRU, but it is in flight: the next-oldest goes instead.
        let drop = lru_insert_graph_pinned(&mut cache, 2, vec![2], GraphHandle(12), &[vec![0]]);
        assert_eq!(drop.iter().map(|g| g.0).collect::<Vec<_>>(), vec![11]);
        assert!(cache.0.contains_key(&vec![0]));
        assert!(cache.0.contains_key(&vec![2]));
        // 2026-09-25: Every entry pinned: nothing is evicted and the cache grows past cap
        // rather than destroying a graph the stream may still be replaying.
        let drop =
            lru_insert_graph_pinned(&mut cache, 2, vec![3], GraphHandle(13), &[vec![0], vec![2]]);
        assert!(drop.is_empty());
        assert_eq!(cache.0.len(), 3);
    }

    #[test]
    fn cap_is_headroom_over_decode_meta_rows() {
        assert_eq!(batch_decode_graph_cap(0), 16);
        assert_eq!(batch_decode_graph_cap(1), 17);
        assert_eq!(batch_decode_graph_cap(32), 48);
        assert_eq!(batch_decode_graph_cap(64), 80);
    }

    /// 2026-09-25: `free_sequence_dispatch` touches the slot-keyed decode graphs only behind
    /// `graph_stale_on_new_sequence()`, never drains or clears them wholesale, and drops the
    /// LoRA-baked `verify_kgamma_graph` / `fused_graph` entries of the slot. Removing the guard,
    /// or replacing the per-slot `remove` with `drain`/`clear`, fails this test.
    /// 2026-10-10: The drops live in `drop_slot_graphs` / `drop_slot_kgamma_graphs`
    /// (`sequence_compact.rs`, shared with the `METRALE_MTP_POOL_CAP` slot migration); the
    /// wholesale checks read those helpers, the guard check reads `free_sequence_dispatch`.
    #[test]
    fn free_sequence_only_drops_slot_graphs_a_layer_calls_stale() {
        let read = |f: &str| {
            std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("src/model/trait_impl")
                    .join(f),
            )
            .unwrap()
        };
        let fn_body = |src: &str, name: &str| -> String {
            let start = src.find(name).unwrap_or_else(|| panic!("{name}"));
            let body = &src[start..];
            let end = body.find("\n    pub(super) fn ").unwrap_or(body.len());
            body[..end].to_string()
        };
        let seq = read("sequence.rs");
        let compact = read("sequence_compact.rs");
        let body = fn_body(&seq, "fn free_sequence_dispatch");
        let helpers = fn_body(&compact, "fn drop_slot_graphs")
            + &fn_body(&compact, "fn drop_slot_kgamma_graphs");
        for src in [&body, &helpers] {
            for bad in ["decode_graph.lock().drain()", "decode_graph.lock().clear()"] {
                assert!(
                    !src.contains(bad),
                    "free_sequence must not invalidate decode graphs wholesale ({bad})"
                );
            }
            assert!(
                !src.contains("batch.0.drain()") && !src.contains("batch.0.clear()"),
                "free_sequence must not invalidate batch decode graphs wholesale"
            );
        }
        assert!(
            helpers.contains("decode_graph")
                && helpers.contains(".remove(&slot_idx)")
                && helpers.contains("self.batch_decode_graphs.lock()"),
            "drop_slot_graphs drops the slot-keyed decode graphs"
        );
        let touches_graphs = body.contains("self.decode_graph.lock()")
            || body.contains("self.batch_decode_graphs.lock()")
            || body.contains("self.drop_slot_graphs(");
        assert_eq!(
            touches_graphs,
            body.contains("graph_stale_on_new_sequence"),
            "free_sequence may drop slot graphs ONLY behind graph_stale_on_new_sequence()"
        );
        assert!(
            body.contains("self.drop_slot_kgamma_graphs(")
                && helpers.contains("verify_kgamma_graph")
                && helpers.contains("fused_graph"),
            "LoRA-baked graphs still drop"
        );
    }
}
