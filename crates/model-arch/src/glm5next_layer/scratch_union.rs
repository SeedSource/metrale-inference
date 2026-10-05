// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: `METRALE_GLM_PREFILL_SCRATCH_UNION=1`: the three load-time, per-rank GLM scratch
//! workspaces that every layer shares (the KDA workspace, the shared MLP workspace and the DSA
//! wide arena) start at the same base inside ONE device allocation sized to the largest of
//! them, instead of three allocations. Same kernels, same math; only the addresses change.
//! GLM-5.3 TP2 at `METRALE_GLM_PREFILL_ROWS_FFN=8192`: KDA 1169.7 MB, MLP 1422.4 MB, DSA wide
//! 731.0 MB -> one 1422 MB union, ~1.9 GB per rank more for the KV cache (sized after load).
//!
//! Why the members may alias (code read, 2026-10-05):
//! - All three are pure per-call scratch, used strictly one after another on one stream; no
//!   contents carry across sublayers, windows, chunks, layers or requests. The attention
//!   partial (KDA `final_out`, DSA over its input) is copied/reduced into `hidden` before the
//!   FFN pass; the MLP workspace is written and read inside one `mlp_compute` call; the DSA
//!   wide arena is used only by eager `decode_k_wide` on the default stream (`meta` uploaded
//!   per call).
//! - Decode/verify CUDA graphs bake pointers into the KDA and MLP workspaces, so the union is
//!   allocated at load (where the members were) and never moved, freed or resized. The
//!   constructors only allocate (no memset, no upload), and `cuMemAlloc` memory is not zeroed
//!   either way.
//!
//! MUST stay OUT of the union (persistent or differently scoped contents): the FlashKDA scratch
//! (transposed recurrent state), the CUTLASS W4A4 workspace and SFB cache, the W8A8 scratch
//! (`dense_fp8`), the DSA xseq arena, every per-layer `Glm5NextDsaWorkspace` (persistent
//! `bt`/`sl`), the MTP head's own MLP/DSA workspaces, `norm_output`/`moe_output`/`hidden` and
//! the buffer arena. A per-layer MLP workspace (`METRALE_GLM_MLP_WS_SHARED=0`) is not a member.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Lever off: no code path here runs; each member is built by its plain constructor, whose
//!   `gpu.alloc` calls are those it made before (`*_in` with `gpu.alloc`).
//! - Each sub-allocation is `ALIGN`-aligned from the union base; a member's placing pass ends
//!   exactly where its measuring pass did.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::glm5next_dsa::Glm5NextDsaConfig;
use crate::glm5next_dsa::layer::DsaWideArena;
use crate::glm5next_kda::{Glm5NextKdaConfig, Glm5NextKdaWorkspace};
use crate::glm5next_mlp::Glm5NextMlpConfig;
use crate::glm5next_mlp::forward::Glm5NextMlpWorkspace;

/// 2026-10-05: The allocator the `*_in` workspace constructors take.
pub type ScratchAlloc<'a> = dyn FnMut(usize) -> Result<DevicePtr> + 'a;

/// 2026-10-05: Alignment of every sub-allocation, from the union base (itself `cuMemAlloc`ed).
pub const ALIGN: usize = 256;

/// 2026-10-05: `METRALE_GLM_PREFILL_SCRATCH_UNION=1` (blanks ignored) turns the union on; off
/// otherwise. Read once.
pub fn scratch_union() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_PREFILL_SCRATCH_UNION").ok();
        parse_switch(raw.as_deref())
    })
}

/// 2026-10-05: `1` (surrounding blanks ignored) only.
pub(crate) fn parse_switch(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-05: A bump sub-allocator over `[base, ..)`: each call returns `base + offset` with
/// the offset rounded up to `ALIGN`. With base 0 it measures (the pointers are offsets).
pub struct Bump {
    base: u64,
    cursor: usize,
}

impl Bump {
    pub fn new(base: DevicePtr) -> Self {
        Self {
            base: base.0,
            cursor: 0,
        }
    }

    pub fn alloc(&mut self, bytes: usize) -> Result<DevicePtr> {
        let off = self.cursor.div_ceil(ALIGN) * ALIGN;
        self.cursor = off + bytes;
        Ok(DevicePtr(self.base + off as u64))
    }

    /// 2026-10-05: The high-water mark: end of the last sub-allocation, in bytes from base.
    pub fn used(&self) -> usize {
        self.cursor
    }
}

/// 2026-10-05: Bytes a member takes in the union: run its `*_in` constructor against a
/// measuring `Bump` and drop the result (the workspaces own no resources; no `Drop`).
pub fn measure<T>(build: impl FnOnce(&mut ScratchAlloc) -> Result<T>) -> Result<usize> {
    let mut b = Bump::new(DevicePtr(0));
    drop(build(&mut |n| b.alloc(n))?);
    Ok(b.used())
}

/// 2026-10-05: Member names, the `place` keys and the log's labels.
pub const KDA: &str = "KDA ws";
pub const MLP: &str = "shared MLP ws";
pub const DSA_WIDE: &str = "DSA wide arena";

/// 2026-10-05: The loader's union: measures the KDA workspace (`kda` = config, rows, chunk
/// rows, as `new_split`), the shared MLP workspace (`mlp` = config, rows, `u_slot` rows, as
/// `new_sized`; `None` when per-layer) and the DSA wide arena (`dsa` = config, rows; `None`
/// without one), then [`ScratchUnion::alloc`].
pub fn plan_glm(
    gpu: &dyn GpuBackend,
    kda: (&Glm5NextKdaConfig, usize, usize),
    mlp: Option<(&Glm5NextMlpConfig, usize, usize)>,
    dsa: Option<(&Glm5NextDsaConfig, usize)>,
) -> Result<Option<ScratchUnion>> {
    let (kc, kr, kch) = kda;
    let mut m = vec![(
        KDA,
        measure(|a| Glm5NextKdaWorkspace::new_split_in(kc, kr, kch, a))?,
    )];
    if let Some((c, r, ur)) = mlp {
        m.push((
            MLP,
            measure(|a| Glm5NextMlpWorkspace::new_sized_in(c, r, ur, a))?,
        ));
    }
    if let Some((c, r)) = dsa {
        m.push((DSA_WIDE, measure(|a| DsaWideArena::new_in(c, r, a))?));
    }
    ScratchUnion::alloc(gpu, m)
}

/// 2026-10-05: One device allocation that its members share from offset 0.
pub struct ScratchUnion {
    base: DevicePtr,
    members: Vec<(&'static str, usize)>,
}

impl ScratchUnion {
    /// 2026-10-05: Allocate `max(member bytes)` once and log the saving. With fewer than two
    /// members there is nothing to share: logs and returns `None` (separate allocations).
    pub fn alloc(
        gpu: &dyn GpuBackend,
        members: Vec<(&'static str, usize)>,
    ) -> Result<Option<Self>> {
        let mb = |b: usize| b as f64 / 1e6;
        let list = members
            .iter()
            .map(|(n, b)| format!("{n} {:.1} MB", mb(*b)))
            .collect::<Vec<_>>()
            .join(", ");
        if members.len() < 2 {
            tracing::warn!(
                "METRALE_GLM_PREFILL_SCRATCH_UNION=1 but only {} member(s) ({list}); separate \
                 allocations kept",
                members.len()
            );
            return Ok(None);
        }
        let bytes = members.iter().map(|m| m.1).max().unwrap_or(0);
        let sum: usize = members.iter().map(|m| m.1).sum();
        let base = gpu.alloc(bytes)?;
        tracing::warn!(
            "GLM prefill scratch union (METRALE_GLM_PREFILL_SCRATCH_UNION): 1 x {:.1} MB shared \
             by {list}; saves {:.1} MB against separate allocations",
            mb(bytes),
            mb(sum - bytes)
        );
        Ok(Some(Self { base, members }))
    }

    /// 2026-10-05: Build member `name` from the union base; errors if `name` is not a member
    /// or its placing pass ends anywhere but its measured size.
    pub fn place<T>(
        &self,
        name: &str,
        build: impl FnOnce(&mut ScratchAlloc) -> Result<T>,
    ) -> Result<T> {
        let Some(&(_, want)) = self.members.iter().find(|m| m.0 == name) else {
            bail!("scratch union has no member {name:?}");
        };
        let mut b = Bump::new(self.base);
        let out = build(&mut |n| b.alloc(n))?;
        if b.used() != want {
            bail!(
                "scratch union member {name:?}: placed {} B, measured {want} B",
                b.used()
            );
        }
        Ok(out)
    }

    pub fn base(&self) -> DevicePtr {
        self.base
    }
}

#[cfg(test)]
#[path = "scratch_union_tests.rs"]
mod tests;
