// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The device-allocation ledger's entries and reports: size and
//! allocating call site of every live allocation.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants: none beyond the types.

use super::MetraleCudaBackend;

/// 2026-09-25: One live device allocation: its size in bytes and the call
/// site of `GpuBackend::alloc` or `alloc_managed` that made it (both are
/// `#[track_caller]` on the trait and on the CUDA impl).
#[derive(Clone, Copy)]
pub(super) struct AllocRecord {
    pub(super) bytes: usize,
    pub(super) site: &'static std::panic::Location<'static>,
}

impl MetraleCudaBackend {
    /// 2026-09-25: Enter an allocation in the ledger. `site` is the caller of
    /// `GpuBackend::alloc` or `alloc_managed`.
    pub(crate) fn record_alloc(
        &self,
        ptr: crate::gpu::DevicePtr,
        bytes: usize,
        site: &'static std::panic::Location<'static>,
    ) {
        self.live_allocs
            .lock()
            .insert(ptr.0, AllocRecord { bytes, site });
    }

    pub(crate) fn forget_alloc(&self, ptr: crate::gpu::DevicePtr) {
        self.live_allocs.lock().remove(&ptr.0);
    }

    /// 2026-09-25: Total bytes on the ledger, plus (2026-10-03) the bytes VMM lazy buffers
    /// have mapped (`lazy_buffer::vmm_mapped_bytes`), which have no per-site entry.
    pub fn live_bytes(&self) -> usize {
        self.live_allocs
            .lock()
            .values()
            .map(|r| r.bytes)
            .sum::<usize>()
            + crate::lazy_buffer::vmm_mapped_bytes()
    }

    /// 2026-09-25: A text report of the ledger: the total, then up to `top_n`
    /// call sites of at least `min_mb` MiB, largest first, then (if any are
    /// left) one line summing the other sites, then up to `top_n` source
    /// files, largest first.
    pub fn alloc_report(&self, top_n: usize, min_mb: usize) -> String {
        use std::collections::HashMap;
        let mut by_site: HashMap<String, (usize, usize)> = HashMap::new();
        let mut total = 0usize;
        for rec in self.live_allocs.lock().values() {
            total += rec.bytes;
            let key = format!("{}:{}", rec.site.file(), rec.site.line());
            let e = by_site.entry(key).or_insert((0, 0));
            e.0 += rec.bytes;
            e.1 += 1;
        }
        let mut rows: Vec<(String, usize, usize)> =
            by_site.into_iter().map(|(k, v)| (k, v.0, v.1)).collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));

        let mut out = format!(
            "GPU allocation ledger: {:.2} GB live across {} sites\n",
            total as f64 / 1e9,
            rows.len()
        );
        let vmm = crate::lazy_buffer::vmm_mapped_bytes();
        if vmm > 0 {
            out.push_str(&format!(
                "  {:>9.1} MB  VMM lazy buffers (mapped granules, not in the total above)\n",
                vmm as f64 / (1024.0 * 1024.0)
            ));
        }
        let mut shown = 0usize;
        let mut folded_bytes = 0usize;
        let mut folded_sites = 0usize;
        for (site, bytes, count) in rows {
            if shown < top_n && bytes >= min_mb * 1024 * 1024 {
                out.push_str(&format!(
                    "  {:>9.1} MB  x{:<5} {}\n",
                    bytes as f64 / (1024.0 * 1024.0),
                    count,
                    site
                ));
                shown += 1;
            } else {
                folded_bytes += bytes;
                folded_sites += 1;
            }
        }
        if folded_sites > 0 {
            out.push_str(&format!(
                "  {:>9.1} MB  across {} smaller sites\n",
                folded_bytes as f64 / (1024.0 * 1024.0),
                folded_sites
            ));
        }

        // 2026-09-25: Per-file rollup: a file that allocates from many lines,
        // each under the cut above, still shows as one entry here.
        let mut by_file: HashMap<&str, (usize, usize)> = HashMap::new();
        for rec in self.live_allocs.lock().values() {
            let e = by_file.entry(rec.site.file()).or_insert((0, 0));
            e.0 += rec.bytes;
            e.1 += 1;
        }
        let mut frows: Vec<(&str, usize, usize)> =
            by_file.into_iter().map(|(k, v)| (k, v.0, v.1)).collect();
        frows.sort_by(|a, b| b.1.cmp(&a.1));
        out.push_str("  ── by file ──\n");
        for (file, bytes, count) in frows.into_iter().take(top_n) {
            out.push_str(&format!(
                "  {:>9.1} MB  x{:<5} {}\n",
                bytes as f64 / (1024.0 * 1024.0),
                count,
                file
            ));
        }
        out
    }
}
