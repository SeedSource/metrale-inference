// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The `METRALE_GLM_DSA_POOL_CACHE=1` side of `dsa_depth_rig` (split out for the
//! 500-line cap): per-layer persistent pool arrays, the warm-up that fills them for a live
//! context, `dsa_write_geom_pk`, and the incremental compress launched as the ceiling decode
//! launches it. Steady state is the production one: pools `[0, S / kpool)` are final in `pk`,
//! so a step compresses only from the geometry's start slot (the device `pk_len`) on.
//!
//! The raw-row "ring" is the flat full-length `k`/`gate` cache (`ring_rows` = its row count, so
//! slot = row): the kernel reads the same rows from the same slots arithmetic, but not from the
//! production 8,448-row ring's memory footprint (INFERRED: a few cache lines per step either way).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::pool_cache::DsaPoolCacheArgs;
use metrale_model_arch::glm5next_dsa::select::{PoolRows, compress_pools_only, topk_tile};

use super::{D, Fixture, KP, MAX_POOLS, POISON, PoolArrays, ROWS, TOPK};

impl PoolArrays {
    pub(super) fn null() -> Self {
        let n = DevicePtr(0);
        Self { pk: n, pidx: n, pvalid: n, pk_len: n }
    }

    /// 2026-10-08: Arrays for `MAX_POOLS` pools plus the trailing partial pool's slot
    /// (`PoolCache::alloc` sizes them `capacity / kpool`).
    pub(super) fn alloc(g: &dyn GpuBackend) -> Result<Self> {
        let m = MAX_POOLS + 1;
        let pk_len = g.alloc(4)?;
        g.memset(pk_len, 0, 4)?;
        Ok(Self {
            pk: g.alloc(m * D * 4)?,
            pidx: g.alloc(m * KP * 4)?,
            pvalid: g.alloc(m)?,
            pk_len,
        })
    }
}

impl Fixture {
    /// 2026-10-08: Rows the raw-row cache holds (its slot of row `r` is `r % ring_rows`).
    fn ring_rows(&self) -> usize {
        MAX_POOLS * KP + 3
    }

    /// 2026-10-08: `DsaPoolCacheArgs` of layer `l` as `Glm5NextDsaState::pool_select_args`
    /// returns them on a replay-safe path (`pk_start` unused under a ceiling launch).
    pub(super) fn pc_args(&self, l: usize) -> DsaPoolCacheArgs {
        let p = self.layers[l].pc;
        DsaPoolCacheArgs {
            pk: p.pk,
            pidx: p.pidx,
            pvalid: p.pvalid,
            pk_len_dev: p.pk_len,
            pk_start: 0,
            ring_rows: self.ring_rows(),
        }
    }

    /// 2026-10-08: The pool key, index and validity arrays the scores / expand kernels read:
    /// the layer's persistent ones with the cache, else the stage scratch.
    pub(super) fn pools(&self, l: usize) -> (DevicePtr, DevicePtr, DevicePtr) {
        if self.pool_cache {
            let p = self.layers[l].pc;
            (p.pk, p.pidx, p.pvalid)
        } else {
            (self.st.pool_keys, self.st.pool_indices, self.st.pool_valid)
        }
    }

    /// 2026-10-08: Warm every layer's cache for live context `s`: device `pk_len` to 0, then one
    /// exact incremental compress over `[0, s)` (`compress_pools_only`), which leaves pools
    /// `[0, s / kpool)` final and `pk_len = s / kpool`, the state after a prefill to `s`.
    pub(super) fn warm(&self, g: &dyn GpuBackend, s: usize) -> Result<()> {
        for l in 0..self.layers.len() {
            let layer = self.layers[l];
            g.copy_h2d(&[0u8; 4], layer.pc.pk_len)?;
            let rows = PoolRows {
                k_normed: layer.k,
                gate: layer.gate,
                valid: layer.valid,
                ape: self.ape,
                first_key: 0,
            };
            compress_pools_only(g, &self.kernels, &self.cfg, &rows, &self.pc_args(l), s, 0)?;
        }
        Ok(())
    }

    /// 2026-10-08: `dsa_write_geom_pk` for layer `l`, row `r`: `dsa_write_geom` plus the start
    /// slot from the layer's device `pk_len`, as `layer/rows.rs` `pool_write_geom_pk`
    /// (lines 184-211) launches it.
    pub(crate) fn geom_pk(&self, g: &dyn GpuBackend, l: usize, r: usize, stream: u64) -> Result<()> {
        KernelLaunch::new(g, self.kernels.write_geom_pk)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(self.rows[r].seq_len)
            .arg_ptr(self.rows[r].geom)
            .arg_u32(KP as u32)
            .arg_u32(TOPK as u32)
            .arg_u32(topk_tile() as u32)
            .arg_ptr(self.layers[l].pc.pk_len)
            .launch(stream)
    }

    /// 2026-10-08: `dsa_kpool_compress_incr` alone, the ceiling launch `select/launch.rs`
    /// `launch_compress_incr` (lines 317-343) makes through `launch_incr_raw` (lines 380-410):
    /// the ceiling grid, `seq = MAX_POOLS * kpool`, start 0 (the real start is the geom slot).
    pub(crate) fn compress_incr(&self, g: &dyn GpuBackend, l: usize, r: usize, stream: u64) -> Result<()> {
        let (layer, p) = (self.layers[l], self.layers[l].pc);
        KernelLaunch::new(g, self.kernels.kpool_compress_incr)
            .grid([self.grids.0 as u32, 1, 1])
            .block([D.min(1024) as u32, 1, 1])
            .arg_ptr(layer.k)
            .arg_ptr(layer.gate)
            .arg_ptr(layer.valid)
            .arg_ptr(self.ape)
            .arg_ptr(p.pk)
            .arg_ptr(p.pidx)
            .arg_ptr(p.pvalid)
            .arg_u32((MAX_POOLS * KP) as u32)
            .arg_u32(D as u32)
            .arg_u32(KP as u32)
            .arg_i32(0)
            .arg_u32(self.ring_rows() as u32)
            .arg_u32(0)
            .arg_ptr(p.pk_len)
            .arg_ptr(self.rows[r].geom)
            .launch(stream)
    }

    /// 2026-10-08: Which implementation each selection stage runs, as the production dispatch
    /// picks it for a ceiling (graph-replay) launch with the current environment: `(compress,
    /// scores, topk)`. A ceiling launch never takes `dsa_index_scores_tc` / `_tc2`
    /// (`scores_tc_for` needs exact host geometry, `select/launch.rs` lines 116-123), so the
    /// plain scorer runs whatever `METRALE_GLM_DSA_SCORES_TC` / `_TC2` say.
    pub(crate) fn impls(&self) -> (String, String, String) {
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| "unset".into());
        (
            if self.pool_cache {
                "dsa_write_geom_pk+dsa_kpool_compress_incr(ceiling grid)".into()
            } else {
                "dsa_kpool_compress(full,POOL_CACHE off)".into()
            },
            format!(
                "dsa_index_scores(plain; ceiling launch never engages TC/TC2: SCORES_TC={} \
                 SCORES_TC2={})",
                env("METRALE_GLM_DSA_SCORES_TC"),
                env("METRALE_GLM_DSA_SCORES_TC2")
            ),
            if self.radix {
                "dsa_topk_radix(9 launches)".into()
            } else {
                format!("dsa_topk_pools(TOPK_RADIX={})", env("METRALE_GLM_DSA_TOPK_RADIX"))
            },
        )
    }

    /// 2026-10-06: Stage-by-stage selection of layer `l`, row `r`, in `select_tokens` order.
    pub(crate) fn staged(&self, g: &dyn GpuBackend, l: usize, r: usize, stream: u64) -> Result<()> {
        self.compress(g, l, r, stream)?;
        self.scores(g, l, r, stream)?;
        self.topk(g, r, stream)?;
        self.expand(g, l, r, stream)
    }

    /// 2026-10-06: Bytes of the `[ROWS, out_width]` token buffers.
    pub(crate) fn token_bytes(&self) -> usize {
        ROWS * self.cfg.out_width() * 4
    }

    /// 2026-10-06: Fill both token buffers with the poison byte, so a slot a path does not
    /// write cannot match the other path's output.
    pub(crate) fn poison_tokens(&self, g: &dyn GpuBackend) -> Result<()> {
        g.memset(self.scratch.tokens(), POISON, self.token_bytes())?;
        g.memset(self.scratch_nc.tokens(), POISON, self.token_bytes())?;
        g.memset(self.st.tokens, POISON, self.token_bytes())
    }
}
