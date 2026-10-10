// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: `DsaSelectScratch` planning, allocation and the `fits` checks, split out of
//! `select.rs` (500-line cap) when the pool cache (`METRALE_GLM_DSA_POOL_CACHE=1`) made the
//! pool regions optional.
//! 2026-10-06 (comb21 port): seven regions, the seventh the radix top-k work buffer
//! (`METRALE_GLM_DSA_TOPK_RADIX`, [`radix::regions`]); the two levers plan independently.
//! 2026-10-10: eight regions, the eighth the BF16 q copy of `METRALE_GLM_DSA_SCORES_TC3`
//! ([`super::tc3::qbf_region_bytes`]), 0 bytes and not allocated with the lever off.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: those of `select.rs` (a pass larger than the allocation is an `Err`; with the
//! pool cache the pool regions are planned at 0 bytes, left null, and a pass that would write
//! them is refused).

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{DsaSelectGeometry, DsaSelectScratch, radix};
use crate::glm5next_dsa::Glm5NextDsaConfig;

impl DsaSelectScratch {
    /// 2026-09-25: Allocate for `geom`, the largest pass the caller will run
    /// (`Glm5NextDsaWorkspace::new` plans it at `max_dsa_context`).
    pub fn alloc(
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextDsaConfig,
        geom: &DsaSelectGeometry,
    ) -> Result<Self> {
        Self::alloc_sized(gpu, Self::plan_bytes(cfg, std::slice::from_ref(geom)))
    }

    /// 2026-10-05: Region bytes (field order) and `tokens` bytes that cover every pass in
    /// `geoms`: the per-region maximum. For one geometry, exactly what [`Self::alloc`] has
    /// always reserved. `METRALE_GLM_DSA_SELECT_SCRATCH_SHARED` (`shared.rs`) sizes the one
    /// scratch every DSA workspace shares with it.
    /// 2026-10-06: With `METRALE_GLM_DSA_POOL_CACHE=1` the three pool regions are planned at 0
    /// bytes and not allocated ([`Self::plan_bytes_with`]).
    /// 2026-10-06: Seven regions; the seventh is the radix top-k work buffer (0 with
    /// `METRALE_GLM_DSA_TOPK_RADIX` off).
    /// 2026-10-10: Eight; the eighth is the BF16 q copy (0 with `METRALE_GLM_DSA_SCORES_TC3` off).
    pub fn plan_bytes(cfg: &Glm5NextDsaConfig, geoms: &[DsaSelectGeometry]) -> ([usize; 8], usize) {
        let pool_cache = crate::glm5next_dsa::pool_cache::dsa_pool_cache();
        Self::plan_bytes_levers(cfg, geoms, radix::dsa_topk_radix(), pool_cache)
    }

    /// 2026-10-06: [`Self::plan_bytes`] for an explicit `METRALE_GLM_DSA_TOPK_RADIX` state
    /// (the pool-cache lever from the environment).
    pub fn plan_bytes_for(
        cfg: &Glm5NextDsaConfig,
        geoms: &[DsaSelectGeometry],
        lever: bool,
    ) -> ([usize; 8], usize) {
        let pool_cache = crate::glm5next_dsa::pool_cache::dsa_pool_cache();
        Self::plan_bytes_levers(cfg, geoms, lever, pool_cache)
    }

    /// 2026-10-06: [`Self::plan_bytes`] with the pool-cache lever given: `pool_cache` plans the
    /// pool key/index/validity regions at 0 bytes (every pass then reads the state's
    /// persistent arrays). Pure, for tests.
    pub fn plan_bytes_with(
        cfg: &Glm5NextDsaConfig,
        geoms: &[DsaSelectGeometry],
        pool_cache: bool,
    ) -> ([usize; 8], usize) {
        Self::plan_bytes_levers(cfg, geoms, radix::dsa_topk_radix(), pool_cache)
    }

    /// 2026-10-06: The plan with both levers given: `radix_lever` (`METRALE_GLM_DSA_TOPK_RADIX`,
    /// region 6) and `pool_cache` (`METRALE_GLM_DSA_POOL_CACHE`, regions 0 to 2 at 0 bytes).
    pub fn plan_bytes_levers(
        cfg: &Glm5NextDsaConfig,
        geoms: &[DsaSelectGeometry],
        radix_lever: bool,
        pool_cache: bool,
    ) -> ([usize; 8], usize) {
        let mut capacity = [0usize; 8];
        let mut tokens_bytes = 0;
        for g in geoms {
            for (c, w) in capacity
                .iter_mut()
                .zip(radix::regions(g, cfg, radix_lever, pool_cache))
            {
                *c = (*c).max(w);
            }
            tokens_bytes = tokens_bytes.max(g.q_rows * cfg.out_width() * 4);
        }
        (capacity, tokens_bytes)
    }

    /// 2026-10-05: Total bytes of a scratch planned by [`Self::plan_bytes`].
    pub fn planned_total(plan: &([usize; 8], usize)) -> usize {
        plan.0.iter().sum::<usize>() + plan.1
    }

    /// 2026-10-05: Allocate `plan` (from [`Self::plan_bytes`]), in field order.
    /// 2026-10-06: A pool region planned at 0 bytes (`METRALE_GLM_DSA_POOL_CACHE=1`) is left
    /// null and not allocated; `fits` then refuses any pass that would write it.
    pub fn alloc_sized(gpu: &dyn GpuBackend, plan: ([usize; 8], usize)) -> Result<Self> {
        let (capacity, tokens_bytes) = plan;
        let pool_region = |bytes: usize| -> Result<DevicePtr> {
            if bytes == 0 {
                Ok(DevicePtr(0))
            } else {
                gpu.alloc(bytes)
            }
        };
        Ok(Self {
            pool_keys: pool_region(capacity[0])?,
            pool_indices: pool_region(capacity[1])?,
            pool_valid: pool_region(capacity[2])?,
            scores: gpu.alloc(capacity[3])?,
            valid_cand: gpu.alloc(capacity[4])?,
            selected: gpu.alloc(capacity[5])?,
            tokens: gpu.alloc(tokens_bytes)?,
            radix: radix::alloc_region(gpu, capacity[6])?,
            qbf: radix::alloc_region(gpu, capacity[7])?,
            capacity,
            tokens_bytes,
        })
    }

    /// 2026-10-05: Bytes this scratch reserved, all regions and `tokens`.
    pub fn bytes(&self) -> usize {
        Self::planned_total(&(self.capacity, self.tokens_bytes))
    }

    /// 2026-10-06: The pool key, pool index and pool validity regions the last full compress
    /// wrote (null when planned without them, `METRALE_GLM_DSA_POOL_CACHE=1`). For parity
    /// microtests (`examples/dsa_pool_cache_parity_microtest.rs`).
    pub fn pool_regions(&self) -> [DevicePtr; 3] {
        [self.pool_keys, self.pool_indices, self.pool_valid]
    }

    /// 2026-09-25: `[q_rows, out_width]` i32 selection produced by the last pass.
    pub fn tokens(&self) -> DevicePtr {
        self.tokens
    }

    /// 2026-09-25: The same scratch with `tokens` pointing at row `row`.
    ///
    /// The per-row selector (`q_rows == 1`) writes each row's result into its own slot of
    /// the `[max_rows, out_width]` output, which the attention reads in one launch. The
    /// other regions are temporaries of one pass and are shared.
    pub fn row(&self, row: usize, cfg: &Glm5NextDsaConfig) -> Self {
        Self {
            tokens: self.tokens.offset(row * cfg.out_width() * 4),
            tokens_bytes: self.tokens_bytes - row * cfg.out_width() * 4,
            ..*self
        }
    }

    /// 2026-09-25: Whether `geom` fits what was allocated. `select_tokens` checks it on every
    /// pass, so a pass larger than the allocation is an error rather than an overrun.
    /// 2026-10-06: Under `METRALE_GLM_DSA_POOL_CACHE=1` the pool regions are not needed
    /// ([`Self::fits_with`]).
    pub fn fits(&self, cfg: &Glm5NextDsaConfig, geom: &DsaSelectGeometry) -> Result<()> {
        self.fits_with(cfg, geom, crate::glm5next_dsa::pool_cache::dsa_pool_cache())
    }

    /// 2026-10-06: [`Self::fits`] for a pass that does (`pool_cache`) or does not compress into
    /// the state's persistent pool arrays. `select_tokens` passes whether its inputs carry
    /// them, so a pass without them on a scratch planned for the cache is refused.
    pub fn fits_with(
        &self,
        cfg: &Glm5NextDsaConfig,
        geom: &DsaSelectGeometry,
        pool_cache: bool,
    ) -> Result<()> {
        let want = radix::regions(geom, cfg, radix::dsa_topk_radix(), pool_cache);
        for (i, (w, c)) in want.iter().zip(self.capacity.iter()).enumerate() {
            if w > c {
                bail!(
                    "DSA select: scratch region {i} needs {w} B but only {c} B was \
                     reserved ({} tokens, {} pools, {} query rows)",
                    geom.seq,
                    geom.n_pools,
                    geom.q_rows
                );
            }
        }
        let want_tokens = geom.q_rows * cfg.out_width() * 4;
        if want_tokens > self.tokens_bytes {
            bail!(
                "DSA select: selection output needs {want_tokens} B but only {} B was \
                 reserved",
                self.tokens_bytes
            );
        }
        Ok(())
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.pool_keys,
            self.pool_indices,
            self.pool_valid,
            self.scores,
            self.valid_cand,
            self.selected,
            self.tokens,
            self.radix,
            self.qbf,
        ] {
            gpu.free(p)?;
        }
        Ok(())
    }
}
