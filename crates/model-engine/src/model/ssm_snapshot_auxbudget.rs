// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Host byte budget for the aux blobs (`SsmSnapshotPool::aux_blobs`) that
//! intermediate grid checkpoints carry.
//!
//! GLM-5.3's DSA indexer blob is ~5,643 B per prefix token, copied to a fresh host `Vec` at
//! every 8K grid save. With 4 snapshot slots a single 135K prompt kept ~4 large blobs live and
//! drove host MemAvailable from 5.5 GB to 2.2 GB (OOM watchdog at 2 GB). Before a grid save the
//! pool now checks `live + new <= budget`, evicts to make room, or the save is skipped.
//!
//! Owner: model-engine SSM snapshot pool.
//! Invariants:
//! - `AuxMeta::live` equals the sum of the blob bytes of the slots in `aux_blobs`: `set_aux*`
//!   replaces a slot's bytes and `clear_slot_bookkeeping` removes them, so every free and
//!   reuse path keeps it exact.
//! - The decision reads only token counts, layer geometry and this pool's bookkeeping (never
//!   MemAvailable), so ranks that run the same prefill decide the same.
//! - Budget 0 is the old behaviour: `make_room_for_aux` returns `true` and evicts nothing.

#![allow(dead_code)]

use std::collections::HashMap;

use metrale_telemetry::prefix_cache::PrefixCache;

use super::ssm_snapshot::SsmSnapshotPool;

/// 2026-10-03: Default `METRALE_PREFIX_AUX_BUDGET_MB`.
pub(super) const DEFAULT_AUX_BUDGET_MB: usize = 1536;

/// 2026-10-03: The host aux byte budget: `METRALE_PREFIX_AUX_BUDGET_MB` (default 1536;
/// `0` = unlimited), read once per process.
pub(super) fn aux_budget_bytes() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("METRALE_PREFIX_AUX_BUDGET_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_AUX_BUDGET_MB)
            .saturating_mul(1024 * 1024)
    })
}

/// 2026-10-03: FNV-1a over a whole prompt: the owner key of the grid checkpoints one prefill
/// saves. Deterministic across ranks.
pub(super) fn owner_key(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 2026-10-03: Per-slot aux bytes, the live total, and each grid checkpoint's owner.
#[derive(Default)]
pub(super) struct AuxMeta {
    live: usize,
    /// slot -> (bytes, Some((owner, tokens)) for a grid checkpoint)
    slots: HashMap<usize, (usize, Option<(u64, usize)>)>,
}

impl AuxMeta {
    pub(super) fn set(&mut self, slot: usize, bytes: usize, owner: Option<(u64, usize)>) {
        if let Some((old, _)) = self.slots.insert(slot, (bytes, owner)) {
            self.live -= old;
        }
        self.live += bytes;
    }

    pub(super) fn remove(&mut self, slot: usize) {
        if let Some((old, _)) = self.slots.remove(&slot) {
            self.live -= old;
        }
    }
}

impl SsmSnapshotPool {
    /// 2026-10-03: Host bytes of aux blobs currently held.
    pub(super) fn aux_live_bytes(&self) -> usize {
        self.aux_meta.lock().live
    }

    /// 2026-10-03: Make `live + new_bytes <= budget` by evicting snapshots, or return `false`
    /// (nothing evicted when `new_bytes` alone exceeds the budget). Victims: first `owner`'s own
    /// earlier grid checkpoints, oldest (fewest tokens) first, since the newest grid point is
    /// the one a follow-up turn restores from; then the global LRU
    /// (`PrefixCache::evict_snapshot_lru`). Every victim goes through the index removal and
    /// `free`, so the radix entry, the slot and its aux blob are cleared together. `budget == 0`
    /// returns `true` without touching anything.
    pub(super) fn make_room_for_aux(
        &self,
        cache: &dyn PrefixCache,
        owner: u64,
        new_bytes: usize,
        budget: usize,
    ) -> bool {
        if budget == 0 {
            return true;
        }
        if new_bytes > budget {
            return false;
        }
        if self.aux_live_bytes() + new_bytes <= budget {
            return true;
        }
        let mut own: Vec<(usize, usize)> = {
            let m = self.aux_meta.lock();
            m.slots
                .iter()
                .filter_map(|(&slot, &(_, o))| match o {
                    Some((ow, tok)) if ow == owner => Some((tok, slot)),
                    _ => None,
                })
                .collect()
        };
        own.sort_unstable();
        for (_, slot) in own {
            if self.aux_live_bytes() + new_bytes <= budget {
                return true;
            }
            if cache.evict_snapshot_slot(slot) {
                self.free(slot);
            }
        }
        while self.aux_live_bytes() + new_bytes > budget {
            match cache.evict_snapshot_lru() {
                Some(slot) => self.free(slot),
                None => return false,
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_cache::radix_tree::RadixTree;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

    fn pool(gpu: &MockGpuBackend, slots: usize) -> SsmSnapshotPool {
        SsmSnapshotPool::new(slots, 32, 16, 2, 0, 0, 8, gpu).unwrap()
    }

    /// Claim a slot, give it a `bytes` blob owned by `owner` at `tok` tokens, register it.
    fn grid_save(
        p: &SsmSnapshotPool,
        c: &RadixTree,
        owner: u64,
        tok: usize,
        bytes: usize,
    ) -> usize {
        let s = p.try_pop_free_slot().expect("free slot");
        p.set_aux_owned(s, vec![(0, vec![0u8; bytes])], owner, tok);
        let toks: Vec<u32> = (0..tok as u32).map(|t| t ^ owner as u32).collect();
        assert!(
            c.insert_intermediate_snapshot(&toks, &[], &[], 16, s, 0, tok, 0)
                .is_none()
        );
        s
    }

    #[test]
    fn counter_tracks_set_replace_free_and_reuse() {
        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, 3);
        let a = p.try_pop_free_slot().unwrap();
        p.set_aux(a, vec![(0, vec![0; 100]), (1, vec![0; 50])]);
        assert_eq!(p.aux_live_bytes(), 150);
        p.set_aux_owned(a, vec![(0, vec![0; 40])], 1, 8);
        assert_eq!(
            p.aux_live_bytes(),
            40,
            "replacing a slot's blob replaces its bytes"
        );
        let b = p.try_pop_free_slot().unwrap();
        p.set_aux(b, vec![(0, vec![0; 10])]);
        assert_eq!(p.aux_live_bytes(), 50);
        p.free(a);
        p.free(b);
        assert_eq!(p.aux_live_bytes(), 0);
        let again = p.try_pop_free_slot().unwrap();
        assert_eq!(p.aux_live_bytes(), 0);
        p.free(again);
    }

    #[test]
    fn budget_zero_is_a_no_op() {
        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, 2);
        let c = RadixTree::new();
        grid_save(&p, &c, 1, 16, 1000);
        assert!(p.make_room_for_aux(&c, 1, usize::MAX / 2, 0));
        assert_eq!(p.aux_live_bytes(), 1000);
        assert_eq!(c.snapshot_count(), 1);
    }

    #[test]
    fn evicts_own_oldest_first_then_stops_when_it_fits() {
        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, 4);
        let c = RadixTree::new();
        let other = grid_save(&p, &c, 99, 16, 100); // another sequence, oldest by recency
        let s1 = grid_save(&p, &c, 1, 32, 100);
        let s2 = grid_save(&p, &c, 1, 48, 100);
        assert_eq!(p.aux_live_bytes(), 300);
        // budget 350, new 100: need to drop 50+ -> exactly one victim, owner's oldest (32 tok),
        // even though `other` is globally older.
        assert!(p.make_room_for_aux(&c, 1, 100, 350));
        assert_eq!(p.aux_live_bytes(), 200);
        assert!(p.has_aux(other) && p.has_aux(s2) && !p.has_aux(s1));
        assert_eq!(c.snapshot_count(), 2, "radix entry removed with the blob");
        assert_eq!(p.occupancy().0, 2, "slot returned to the free list");
        // Now need to drop 2: own s2, then global LRU takes `other`.
        assert!(p.make_room_for_aux(&c, 1, 300, 350));
        assert_eq!(p.aux_live_bytes(), 0);
        assert_eq!(c.snapshot_count(), 0);
        assert_eq!(p.occupancy().0, 0);
    }

    #[test]
    fn skips_when_it_cannot_fit_and_evicts_nothing() {
        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, 2);
        let c = RadixTree::new();
        grid_save(&p, &c, 1, 16, 100);
        assert!(
            !p.make_room_for_aux(&c, 1, 500, 400),
            "bigger than the whole budget"
        );
        assert_eq!(p.aux_live_bytes(), 100);
        assert_eq!(
            c.snapshot_count(),
            1,
            "no eviction for a save that can never fit"
        );
    }

    /// The 2026-10-03 failure: GLM-5.3 5,643 B/token, grid 8192, 4 slots, budget 1536 MiB.
    /// Returns (peak live aux bytes, live at the end, skipped saves).
    fn simulate(prompt: usize) -> (usize, usize, usize) {
        const BPT: usize = 5643;
        let budget = 1536 * 1024 * 1024;
        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, 4);
        let c = RadixTree::new();
        let (mut peak, mut skipped) = (0usize, 0usize);
        let mut tok = 8192;
        while tok < prompt {
            let bytes = BPT * tok + 16;
            if p.make_room_for_aux(&c, 7, bytes, budget) {
                // account the real size without allocating it
                // pool full: the real path frees the LRU (`reclaim_from_cache`)
                if p.occupancy().0 == p.occupancy().1 {
                    let v = c.evict_snapshot_lru().expect("a victim");
                    p.free(v);
                }
                let s = p.try_pop_free_slot().expect("slot");
                p.aux_meta.lock().set(s, bytes, Some((7, tok)));
                let toks: Vec<u32> = (0..tok as u32).collect();
                c.insert_intermediate_snapshot(&toks, &[], &[], 16, s, 0, tok, 0);
            } else {
                skipped += 1;
            }
            peak = peak.max(p.aux_live_bytes());
            tok += 8192;
        }
        (peak, p.aux_live_bytes(), skipped)
    }

    #[test]
    fn grid_schedule_peak_for_135k_and_204k_prompts() {
        let budget = 1536 * 1024 * 1024;
        for prompt in [135_000usize, 204_000] {
            let (peak, end, skipped) = simulate(prompt);
            eprintln!(
                "prompt {prompt}: peak {} MiB, end {} MiB, skipped {skipped}",
                peak >> 20,
                end >> 20
            );
            assert_eq!(skipped, 0);
            assert!(peak <= budget);
        }
        // 204K: last grid point 196,608 tok = 1,058 MiB; the previous 188,416 tok (1,014 MiB)
        // cannot coexist with it, so exactly one large blob is live at the end.
        let (_, end, _) = simulate(204_000);
        assert!(
            end < 1100 << 20,
            "only the newest large blob remains: {} MiB",
            end >> 20
        );
    }
}
