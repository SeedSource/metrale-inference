// SPDX-License-Identifier: MIT OR Apache-2.0

//! `WeightStore` methods: lookup, deferral, derived tensors and the byte and
//! FP8 accounting over the loaded set.

use super::*;

impl WeightStore {
    /// Create an empty weight store (for testing).
    pub fn empty() -> Self {
        Self {
            weights: HashMap::new(),
            prepartitioned_tp: None,
            deferred: HashMap::new(),
            derived: DerivedStore::default(),
            arena: WeightArena::default(),
        }
    }

    /// Record a tensor that was skipped at load, with its on-disk location.
    pub fn defer(&mut self, name: String, t: DeferredTensor) {
        self.deferred.insert(name, t);
    }

    /// Look up a deferred (not-uploaded) tensor's on-disk location.
    pub fn deferred(&self, name: &str) -> Option<&DeferredTensor> {
        self.deferred.get(name)
    }

    /// Every deferred tensor, name-sorted (NUMERIC on a trailing index, so
    /// `embedders.10` sorts after `embedders.2` — a lexicographic sort here
    /// silently mis-maps the n-gram tables, which cost a real debugging
    /// session the first time).
    pub fn deferred_sorted(&self) -> Vec<(&String, &DeferredTensor)> {
        let mut v: Vec<_> = self.deferred.iter().collect();
        v.sort_by_key(|(n, _)| split_trailing_index(n));
        v
    }

    /// Wrap a pre-built map. Used by alternate loaders (e.g.
    /// `fast_weights::FastSafetensorsLoader`, and the RDMA weight loader in
    /// `metrale-storage`, which lives in a different crate and so needs this pub).
    pub fn from_map(weights: HashMap<String, WeightTensor>) -> Self {
        Self {
            weights,
            prepartitioned_tp: None,
            deferred: HashMap::new(),
            derived: DerivedStore::default(),
            arena: WeightArena::default(),
        }
    }

    /// Get a weight tensor by name. Fails fast if not found.
    pub fn get(&self, name: &str) -> Result<&WeightTensor> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Weight '{name}' not found in store"))
    }

    /// Check if a weight exists.
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// Number of loaded weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// True if no weights are loaded.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Device bytes the store still holds. Not the on-disk load estimate:
    /// this shrinks as `free_matching` drops tensors the binders replaced.
    pub fn resident_bytes(&self) -> usize {
        self.weights.values().map(|t| t.byte_size()).sum()
    }

    /// Iterator over all weight names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weights.keys().map(|s| s.as_str())
    }

    /// Free and forget every tensor whose name matches `pred`. Returns
    /// `(tensors freed, bytes freed)`.
    ///
    /// For loaders that do NOT bind zero-copy from the store's device pointers:
    /// they upload their own copy, so the original is dead weight the moment the
    /// binder returns, and on a unified-memory GB10 that duplicate is the
    /// difference between fitting a KV cache and not.
    ///
    /// 🪤 The caller owns the "is it dead?" question. A tensor bound zero-copy
    /// (every routed expert, and the fused per-expert views in
    /// `weight_loader/step3p7.rs`) is still live in a layer struct — freeing it
    /// here is a use-after-free with no diagnostic. Match narrowly.
    ///
    /// Per-entry free is sound for the same reason `release` gives below: the
    /// loaders allocate one `gpu.alloc` per tensor, and no loader inserts an
    /// `.offset()` view of a shared block into this map — except the weight
    /// arena's sub-allocations (2026-10-06), which are recognised by
    /// `WeightArena::contains` and never freed here.
    pub fn free_matching(
        &mut self,
        gpu: &dyn GpuBackend,
        pred: impl Fn(&str) -> bool,
    ) -> Result<(usize, usize)> {
        let doomed: Vec<String> = self.weights.keys().filter(|n| pred(n)).cloned().collect();
        let (mut count, mut bytes) = (0usize, 0usize);
        let (mut kept, mut kept_bytes) = (0usize, 0usize);
        for name in doomed {
            // `remove` before `free`: the map must never hold a pointer to
            // memory that is gone, even if the free below fails.
            let Some(t) = self.weights.remove(&name) else {
                continue;
            };
            // 2026-10-06: An arena tensor cannot be freed alone; its bytes stay until the
            // arena is released at teardown. Not counted as freed, and reported below.
            if self.arena.contains(t.ptr) {
                kept += 1;
                kept_bytes += t.byte_size();
                continue;
            }
            bytes += t.byte_size();
            gpu.free(t.ptr)
                .map_err(|e| e.context(format!("freeing weight {name}")))?;
            count += 1;
        }
        if kept > 0 {
            tracing::warn!(
                "weight arena: {kept} tensor(s) ({:.1} MiB) were released by name but live in the \
                 weight arena, so their memory stays resident until teardown; the model's arena \
                 predicate claims a tensor its loader frees",
                kept_bytes as f64 / (1024.0 * 1024.0)
            );
        }
        Ok((count, bytes))
    }

    /// 2026-10-06: Install the arena the fast loader filled (replacing the empty default).
    pub fn set_arena(&mut self, arena: WeightArena) {
        self.arena = arena;
    }

    /// 2026-10-06: The arena holding the tensors an [`ArenaHook`] claimed at load.
    pub fn arena(&self) -> &WeightArena {
        &self.arena
    }

    /// The owner for buffers a loader derives from these tensors.
    ///
    /// `&self` because `ModelWeightLoader::load_layers` takes `&WeightStore`;
    /// the interior `Mutex` is the whole reason `DerivedStore` exists as a
    /// type rather than a `Vec` field. See `weights/derived.rs`.
    pub fn derived(&self) -> &DerivedStore {
        &self.derived
    }

    /// Total bytes across all weight tensors on the GPU.
    pub fn total_bytes(&self) -> usize {
        self.weights.values().map(|w| w.byte_size()).sum()
    }

    /// Check if any tensor has FP8 dtype.
    pub fn has_fp8_weights(&self) -> bool {
        self.weights
            .values()
            .any(|w| matches!(w.dtype, WeightDtype::FP8E4M3))
    }

    /// Number of per-layer FP8 KV-cache scale tensors (`*.k_scale`) the
    /// checkpoint ships. `>0` means the model carries calibrated KV scales, so
    /// FP8 KV needs no online calibration; `0` means the scales default to 1.0
    /// (which clips BF16 into E4M3 range), so online calibration or a non-FP8 KV
    /// dtype is required. Used to log the right guidance at serve time.
    pub fn fp8_kv_scale_count(&self) -> usize {
        self.names().filter(|n| n.ends_with(".k_scale")).count()
    }
}
