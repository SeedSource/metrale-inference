// SPDX-License-Identifier: MIT OR Apache-2.0

//! `WeightLoader` for `FastSafetensorsLoader`: the shard walk that drives
//! `load_shard_fast`.

use super::*;

impl WeightLoader for FastSafetensorsLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        let skip_fn = |name: &str| self.should_skip_tensor(name);
        let defer_fn =
            |name: &str, dtype: crate::weights::WeightDtype| self.is_deferred(name, dtype);

        // Resolve shard list (sharded index, single file, or unindexed shards).
        let (shard_files, tensor_to_shard): (Vec<PathBuf>, Option<HashMap<String, String>>) =
            resolve_shards(model_dir)?;

        // Pre-flight OOM estimate (identical to SafetensorsLoader).
        //
        // Deferred tensors are DEFERRED further down — they are never
        // uploaded, so counting them here refuses a model that fits. On
        // LongCat-Flash-Lite the n-gram tables are 62.8 of the checkpoint's
        // 138 GB, which is the difference between a 167 GB "peak" and a 98 GB
        // one; on `nvidia/GLM-5.3-Flash-NVFP4` the MTP block's full-width
        // routed experts are ~7.25 GB per EP=2 rank.
        //
        // 🔴 These two closures and the retain loop below MUST stay the same
        // rule. A tensor counted here but deferred there inflates the peak;
        // one deferred here but swept there is the OOM this pre-flight exists
        // to prevent.
        let preflight_skip = |name: &str| skip_fn(name) || crate::weights::is_ngram_table(name);
        {
            let estimated = estimate_load_bytes(&shard_files, &preflight_skip, &defer_fn)?;
            let has_fp8 = estimate_has_fp8(&shard_files, &preflight_skip, &defer_fn)?;
            let mult = self
                .peak_memory_multiplier
                .unwrap_or(if has_fp8 { 1.5 } else { 1.3 });
            let peak = (estimated as f64 * mult) as usize;
            let free = gpu.free_memory()?;
            let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!(
                "Fast-load pre-flight: {:.2} GB on-disk, {:.1}x overhead = {:.2} GB peak, \
                 {:.2} GB free, {:.1} GB reserve (FP8: {})",
                gib(estimated),
                mult,
                gib(peak),
                gib(free),
                gib(oom_reserve_bytes),
                has_fp8,
            );
            metrale_telemetry::progress::preflight(gib(estimated), gib(free));
            if peak + oom_reserve_bytes > free {
                bail!(
                    "OOM pre-flight: peak {:.2} GB + {:.2} GB reserve exceeds {:.2} GB free. \
                     Use a smaller quantization or add more GPUs for EP.",
                    gib(peak),
                    gib(oom_reserve_bytes),
                    gib(free),
                );
            }
        }

        // Load each shard. Loaded tensors filtered by EP rules upstream.
        let mut weights: HashMap<String, WeightTensor> = HashMap::new();
        // Locations of tensors deliberately NOT uploaded (the n-gram tables).
        let mut deferred: HashMap<String, crate::weights::DeferredTensor> = HashMap::new();
        let total_shards = shard_files.len();
        let initial_free = gpu.free_memory()?;
        let mut offload_logged = false;
        // 2026-10-06: The weight arena for the tensors the model's arena hook claims; with no
        // hook it is never planned and every tensor is allocated alone, as before.
        let arena = crate::weights::WeightArena::default();
        let arena_pick = |name: &str, dtype: crate::weights::WeightDtype| {
            self.arena.as_ref().is_some_and(|h| h(name, dtype))
        };
        let pick: &dyn Fn(&str, crate::weights::WeightDtype) -> bool = &arena_pick;
        let arena_sel: arena_plan::ArenaSel<'_> = self.arena.as_ref().map(|_| (&arena, pick));

        for (i, shard_path) in shard_files.iter().enumerate() {
            // When an index is present, only load the tensors it routes here;
            // otherwise load everything in the shard. `None` means "load all".
            let shard_name = shard_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            let tensor_filter: Option<Vec<String>> = tensor_to_shard.as_ref().map(|map| {
                map.iter()
                    .filter(|(_, s)| *s == shard_name)
                    .map(|(t, _)| t.clone())
                    .collect()
            });

            tracing::info!(
                "Fast-loading shard {}/{}: {}{}",
                i + 1,
                total_shards,
                shard_name,
                tensor_filter
                    .as_ref()
                    .map(|v| format!(" ({} tensors)", v.len()))
                    .unwrap_or_default(),
            );
            metrale_telemetry::progress::shard_start(i + 1, total_shards, shard_name);

            load_shard_fast(
                shard_path,
                tensor_filter.as_deref(),
                gpu,
                &skip_fn,
                &defer_fn,
                self.try_direct_io,
                self.direct_io_tensor_cap,
                self.prefetch_shards,
                &mut weights,
                &mut deferred,
                &mut offload_logged,
                arena_sel,
            )?;

            let free_now = gpu.free_memory().unwrap_or(0);
            let used = initial_free.saturating_sub(free_now);
            tracing::info!(
                "  Shard {}/{} done — GPU memory: {:.2} GB used, {:.2} GB free",
                i + 1,
                total_shards,
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                free_now as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            metrale_telemetry::progress::shard_done(
                i + 1,
                total_shards,
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                free_now as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            if !offload_logged {
                check_oom_guard(
                    gpu,
                    oom_reserve_bytes,
                    &format!("fast weight loading (shard {}/{})", i + 1, total_shards),
                )?;
            }
        }

        // Extra weights (e.g. MTP grafted from another quantization).
        let no_skip = |_: &str| false;
        let no_defer = |_: &str, _: crate::weights::WeightDtype| false;
        let extra = model_dir.join("extra_weights.safetensors");
        if extra.exists() {
            tracing::info!("Fast-loading extra_weights.safetensors");
            let mut extra_offload = false;
            load_shard_fast(
                &extra,
                None,
                gpu,
                &no_skip,
                &no_defer,
                self.try_direct_io,
                self.direct_io_tensor_cap,
                self.prefetch_shards,
                &mut weights,
                &mut deferred,
                &mut extra_offload,
                None,
            )?;
        }

        tracing::info!("Fast-loaded {} weight tensors", weights.len());
        let mut store = WeightStore::from_map(weights);
        for (name, d) in deferred {
            store.defer(name, d);
        }
        arena.log_summary();
        store.set_arena(arena);
        Ok(store)
    }
}
