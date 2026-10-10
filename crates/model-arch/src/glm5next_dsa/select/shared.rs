// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: `METRALE_GLM_DSA_SELECT_SCRATCH_SHARED=1`: ONE [`DsaSelectScratch`] serves every
//! GLM-5.3 DSA workspace on the rank (the text stack's DSA layers and the MTP head) instead of
//! one per workspace. Each is planned at `max_dsa_context` tokens, so at `--max-seq-len 786432`
//! the twelve copies took ~3.9 GB per rank (2026-10-05 alloc ledger, TP2, 256-row prefill
//! sub-chunks); shared, eleven of them go to the KV cache, which is sized after load.
//!
//! Why the workspaces may share it (code read, 2026-10-05):
//! - Every region (pool keys/indices/validity, scores, candidacy, selected pools, `tokens`) is
//!   written and fully consumed inside ONE layer call: `select_row_at` / `select_rows_batched`
//!   / `select_tokens_split` / `decode_k_wide` write it, and that same call's `attend_rows_at`
//!   reads `tokens()` before it returns. Nothing reads it after the layer returns: there is no
//!   cross-layer or target-to-MTP selection reuse (shared-indexer configs are refused,
//!   `config::parsers::glm5_next::refuse_shared_indexer`), and no host read-back.
//! - The DSA layers and the MTP head run one after another on one stream: TP/EP runs force the
//!   default stream for prefill (`prefill_b.rs` / `prefill_c.rs`, `multi_rank_protocol_active`),
//!   decode/verify/propose run on it, the full-width and row-batch paths require it, and the
//!   index-split token exchange is a send/recv on the compute stream. A captured decode or
//!   draft graph is one stream's chain, so its replay keeps the layers in order too.
//! - Allocated at load and never moved or freed, so graphs that bake its pointers stay valid.
//!   Sized to the per-region maximum over every user's geometry (layers at `verify_k` rows,
//!   the MTP head at [`MTP_ROWS`]), so no workspace's `fits` check is weaker than before.
//!
//! Not covered: a single-rank (no communicator) GLM serve, where the scheduler's separate
//! prefill stream could overlap a decode. That case already shares the KDA and MLP
//! workspaces across layers and each DSA layer's own workspace between prefill and decode;
//! sharing widens the overlap from "same layer" to "any DSA layer".
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Lever off: [`install`] is not called and [`for_mtp`] returns `None`, so every workspace
//!   allocates its own scratch exactly as before.
//! - Lever on: one allocation, held here for the life of the process, handed out by copy.
//! - 2026-10-06: With `METRALE_GLM_DSA_POOL_CACHE=1` the shared plan (like every
//!   per-workspace one) has no pool key/index/validity regions.

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::{DsaSelectGeometry, DsaSelectScratch};
use crate::glm5next_dsa::Glm5NextDsaConfig;
use crate::glm5next_dsa::state::max_dsa_context;

/// 2026-10-05: Query rows of the MTP head's DSA workspace: the drafter runs that layer one row
/// at a time (`load_glm5next_mtp_module`). The shared scratch plans this geometry too.
pub const MTP_ROWS: usize = 1;

/// 2026-10-05: `METRALE_GLM_DSA_SELECT_SCRATCH_SHARED=1` (blanks ignored) turns sharing on; off
/// otherwise. Read once.
pub fn dsa_select_scratch_shared() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_SELECT_SCRATCH_SHARED").ok();
        crate::glm5next_layer::scratch_union::parse_switch(raw.as_deref())
    })
}

/// 2026-10-05: The shared scratch, set by [`install`].
static SHARED: Mutex<Option<DsaSelectScratch>> = Mutex::new(None);

/// 2026-10-05: The sizes behind the shared scratch: what it reserves and what the per-workspace
/// allocations it replaces would have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedPlan {
    /// 2026-10-05: Region and `tokens` bytes of the one shared scratch.
    /// 2026-10-06: Seven regions; the seventh is the radix top-k work buffer (0 with
    /// `METRALE_GLM_DSA_TOPK_RADIX` off).
    pub plan: ([usize; 8], usize),
    /// 2026-10-05: Bytes one text-layer workspace allocates on its own.
    pub layer_bytes: usize,
    /// 2026-10-05: Bytes the MTP head's workspace allocates on its own.
    pub mtp_bytes: usize,
}

impl SharedPlan {
    /// 2026-10-05: Plan for text layers at `layer_rows` rows and the MTP head at [`MTP_ROWS`],
    /// each at `max_dsa_context(cfg)` tokens, as `Glm5NextDsaWorkspace::new` plans them.
    /// 2026-10-06: With `METRALE_GLM_DSA_POOL_CACHE=1` neither the shared nor the per-workspace
    /// plans carry the pool regions ([`Self::new_with`]).
    pub fn new(cfg: &Glm5NextDsaConfig, layer_rows: usize) -> Result<Self> {
        let pool_cache = crate::glm5next_dsa::pool_cache::dsa_pool_cache();
        Self::new_with(cfg, layer_rows, pool_cache)
    }

    /// 2026-10-06: [`Self::new`] with the pool-cache lever given. Pure, for tests.
    pub fn new_with(cfg: &Glm5NextDsaConfig, layer_rows: usize, pool_cache: bool) -> Result<Self> {
        let seq = max_dsa_context(cfg);
        let layer = DsaSelectGeometry::plan(cfg, seq, layer_rows.max(1))?;
        let mtp = DsaSelectGeometry::plan(cfg, seq, MTP_ROWS)?;
        let one = |g: &DsaSelectGeometry| {
            DsaSelectScratch::planned_total(&DsaSelectScratch::plan_bytes_with(
                cfg,
                std::slice::from_ref(g),
                pool_cache,
            ))
        };
        Ok(Self {
            plan: DsaSelectScratch::plan_bytes_with(cfg, &[layer, mtp], pool_cache),
            layer_bytes: one(&layer),
            mtp_bytes: one(&mtp),
        })
    }

    /// 2026-10-05: Bytes of the shared scratch.
    pub fn shared_bytes(&self) -> usize {
        DsaSelectScratch::planned_total(&self.plan)
    }

    /// 2026-10-05: Bytes saved against `layers` per-layer workspaces plus, when `mtp`, the MTP
    /// head's: the per-workspace sum minus the one shared scratch (0 with no user).
    pub fn saved_bytes(&self, layers: usize, mtp: bool) -> usize {
        let sum = layers * self.layer_bytes + if mtp { self.mtp_bytes } else { 0 };
        sum.saturating_sub(self.shared_bytes())
    }
}

/// 2026-10-05: Allocate the shared scratch (the loader, before the text layers are built) for
/// `layers` DSA layers at `layer_rows` rows and the MTP head, log the saving, and keep it for
/// [`for_mtp`]. Called only with the lever on.
pub fn install(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextDsaConfig,
    layer_rows: usize,
    layers: usize,
) -> Result<DsaSelectScratch> {
    let p = SharedPlan::new(cfg, layer_rows)?;
    let scratch = DsaSelectScratch::alloc_sized(gpu, p.plan)?;
    let mb = |b: usize| b as f64 / 1e6;
    tracing::info!(
        "GLM DSA select scratch SHARED (METRALE_GLM_DSA_SELECT_SCRATCH_SHARED): 1 x {:.1} MB at \
         {} tokens for {layers} DSA layers ({layer_rows} rows, {:.1} MB each) and the MTP head \
         ({MTP_ROWS} row, {:.1} MB); saves {:.1} MB ({:.1} MB without an MTP head){}",
        mb(p.shared_bytes()),
        max_dsa_context(cfg),
        mb(p.layer_bytes),
        mb(p.mtp_bytes),
        mb(p.saved_bytes(layers, true)),
        mb(p.saved_bytes(layers, false)),
        if crate::glm5next_dsa::pool_cache::dsa_pool_cache() {
            "; pool regions not allocated (METRALE_GLM_DSA_POOL_CACHE)"
        } else {
            ""
        },
    );
    *SHARED.lock().unwrap_or_else(|e| e.into_inner()) = Some(scratch);
    Ok(scratch)
}

/// 2026-10-05: The shared scratch for the MTP head's `rows`-row workspace, or `None` (it then
/// allocates its own): lever off, nothing installed, or the installed scratch does not fit
/// that geometry (logged).
pub fn for_mtp(cfg: &Glm5NextDsaConfig, rows: usize) -> Option<DsaSelectScratch> {
    if !dsa_select_scratch_shared() {
        return None;
    }
    let s = (*SHARED.lock().unwrap_or_else(|e| e.into_inner()))?;
    let fit = DsaSelectGeometry::plan(cfg, max_dsa_context(cfg), rows.max(1))
        .and_then(|g| s.fits(cfg, &g));
    match fit {
        Ok(()) => Some(s),
        Err(e) => {
            tracing::warn!(
                "METRALE_GLM_DSA_SELECT_SCRATCH_SHARED: the MTP head allocates its own DSA \
                 select scratch ({e:#})"
            );
            None
        }
    }
}

#[cfg(test)]
#[path = "shared_tests.rs"]
mod tests;
