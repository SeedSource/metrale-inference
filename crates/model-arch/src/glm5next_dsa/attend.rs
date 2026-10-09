// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3 DSA attention: the launcher for the selected-index NoPE MLA
//! paged decode.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `decode_attention` launches only after `DsaDecodePaging::validate`
//!   passes, the selection has one query row per decoded sequence, and
//!   `kv_lora_rank` equals `KERNEL_KV_LORA_DIM`.
//!
//! It consumes what [`super::select`] produced: a `[q_rows, out_width]` i32 row
//! of token ids with `-1` holes, and gathers exactly those tokens from the
//! paged FP8 latent cache. `dsa_mla_masked_attn` is an oracle, not this path
//! (see [`super::MASKED_ATTN_MAX_KEYS`]). When the selection row has no
//! duplicates, the gather attends to the same tokens as the reference's masked
//! attention, whose mask (`glm5next_dsa_ref::topk_to_mask`) is 0/1 set
//! membership.
//!
//! # NoPE, and why neither DeepSeek-V4 MLA decode kernel would do
//!
//! GLM-5.3 is NoPE: the `glm5_next` config parser refuses `qk_rope_head_dim != 0`,
//! so the latent is the whole cache token. `deepseek-v4-flash/nvfp4/mla_paged_decode.cu`
//! declares `kv_cache_dim` and never reads it (it hardcodes `ROPE_DIM 64`);
//! `mla_paged_decode_fp8.cu` in the same directory uses the runtime stride but
//! then overwrites dims 448–511 with rope bytes read past the latent, which under
//! NoPE belong to the next token. Hence a GLM-target kernel with no rope arm.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{Glm5NextDsaConfig, select::DsaSelectGeometry};

pub mod prefill_tc;
pub use prefill_tc::{
    MLA_PREFILL_TC_ENTRY, MLA_PREFILL_TC_HEADS, MLA_PREFILL_TC_MAX_SEL, MLA_PREFILL_TC_MODULE,
    MLA_PREFILL_TC_SMEM_BYTES, MLA_PREFILL_TC2_CVT_CHECK_ENTRY, MLA_PREFILL_TC2_ENTRY,
    MLA_PREFILL_TC2_HEADS, MLA_PREFILL_TC2_HWCVT_ENTRY, MLA_PREFILL_TC2_L2_FLUSH_ENTRY,
    MLA_PREFILL_TC2_MODULE,
    MLA_PREFILL_TC2_SMEM_BYTES, attention, prefill_attention_tc, prefill_attention_tc2,
    prefill_attention_tc2_hwcvt,
};
pub mod split;
pub use split::{
    DSA_MLA_SPLIT_ENTRY, DSA_MLA_SPLIT_HEADGROUP, DSA_MLA_SPLIT_MAX, DSA_MLA_SPLIT_MAX_ROWS,
    DSA_MLA_SPLIT_MERGE_ENTRY, DSA_MLA_SPLIT_MIN_KEYS, DSA_MLA_SPLIT_PARTIAL_FLOATS, MlaSplit,
    decode_attention_split, mla_split, split_count, split_scratch_bytes,
};

/// 2026-09-25: Module name the DSA decode kernel resolves from. An unlisted `.cu`
/// takes its file stem, and this one lives in the `glm-5.3-flash` target.
pub const DSA_DECODE_MODULE: &str = "glm5next_dsa_mla_decode";

/// 2026-09-25: Threads per block: `NUM_WARPS * WARP_SIZE` (8 × 32) in the kernel.
const DECODE_BLOCK: u32 = 256;

/// 2026-09-29 (A153, CPU-confirmed): `METRALE_GLM_MLA_SCALE_AUTHOR=1` switches the NoPE MLA
/// softmax scale from `kv_lora_rank^-0.5` (what this kernel shipped with) to
/// `qk_head_dim^-0.5` (`qk_nope_head_dim + qk_rope_head_dim`), the convention the model author
/// uses and `glm5next_dsa_ref::mla::mla_masked_attention` already implements. On GLM-5.3
/// (NoPE, `qk_rope_head_dim == 0`) the two differ: 1/sqrt(512) vs 1/sqrt(256). Off by default;
/// read once per process, like [`crate::glm5next_layer::prefill_staged`].
pub fn mla_scale_author() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_MLA_SCALE_AUTHOR").as_deref() == Ok("1"))
}

/// 2026-09-29: The NoPE MLA softmax scale for `cfg`, given the resolved
/// `METRALE_GLM_MLA_SCALE_AUTHOR` choice. Split out from [`mla_scale`] so tests can check both
/// arms without touching the process-wide env lever.
pub(crate) fn mla_scale_for(cfg: &Glm5NextDsaConfig, author: bool) -> f32 {
    if author {
        ((cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) as f32).powf(-0.5)
    } else {
        (cfg.kv_lora_rank as f32).powf(-0.5)
    }
}

/// 2026-09-29: The NoPE MLA softmax scale for `cfg`: `kv_lora_rank^-0.5` (A153's default,
/// unchanged) unless `METRALE_GLM_MLA_SCALE_AUTHOR=1`, in which case `qk_head_dim^-0.5`.
pub fn mla_scale(cfg: &Glm5NextDsaConfig) -> f32 {
    mla_scale_for(cfg, mla_scale_author())
}

/// 2026-10-01: Head-group sizes with a `glm5next_dsa_mla_decode_fp8_hg{G}` entry point: one
/// block per G consecutive heads of a row, byte-identical to the per-head kernel (argument in
/// the `.cu`; checked on a GPU by `examples/dsa_mla_headgroup_bitparity_microtest.rs`).
pub const DSA_MLA_HEADGROUPS: [usize; 3] = [2, 4, 8];

/// 2026-10-01: `METRALE_GLM_DSA_MLA_HEADGROUP` as a head-group size: `2`, `4` or `8` select
/// that variant; unset, `0` and anything else select the per-head kernel (0).
pub(crate) fn parse_mla_headgroup(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(t) => t
            .parse::<usize>()
            .ok()
            .filter(|g| DSA_MLA_HEADGROUPS.contains(g))
            .unwrap_or(0),
        None => 0,
    }
}

/// 2026-10-01: `METRALE_GLM_DSA_MLA_HEADGROUP=2|4|8` launches the DSA MLA attention one block
/// per G heads of a row instead of one per head, so each selected token is gathered and
/// decoded once per G heads; byte-identical by construction. Every `decode_attention` call
/// (prefill and decode) follows it. Off (0) unless set to 2, 4 or 8; read once.
pub fn mla_headgroup() -> usize {
    static E: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_MLA_HEADGROUP").ok();
        let g = parse_mla_headgroup(raw.as_deref());
        if g != 0 {
            tracing::warn!(
                "METRALE_GLM_DSA_MLA_HEADGROUP={g} - DSA MLA attention runs one block per {g} \
                 heads (glm5next_dsa_mla_decode_fp8_hg{g}; byte-identical by construction)"
            );
        } else if let Some(r) = raw.as_deref().filter(|r| !r.is_empty() && r.trim() != "0") {
            tracing::warn!(
                "METRALE_GLM_DSA_MLA_HEADGROUP={r} is not 0, 2, 4 or 8 - the per-head kernel runs"
            );
        }
        g
    })
}

/// 2026-10-01: The head-group size a launch takes for a `requested` one: `requested` when it
/// divides `num_q_heads` and its entry point resolved, else 0 (the per-head kernel).
pub(crate) fn headgroup_for(requested: usize, num_q_heads: usize, resolved: bool) -> usize {
    if requested != 0 && num_q_heads.is_multiple_of(requested) && resolved {
        requested
    } else {
        0
    }
}

/// 2026-09-25: The selected-index MLA decode entry point.
/// 2026-10-01: With the head-grouped variants, `KernelHandle(0)` where one did not resolve.
/// 2026-10-01: And the tensor-core prefill kernel (`METRALE_GLM_MLA_PREFILL_TC`, see
/// [`prefill_tc`]), `KernelHandle(0)` when it did not resolve.
/// 2026-10-08: And its exact rewrite (`METRALE_GLM_MLA_PREFILL_TC2`): looked up only with the
/// lever on (or by [`Self::with_prefill_tc2`]), `KernelHandle(0)` otherwise.
/// 2026-10-05: And the split-key decode (`METRALE_GLM_DSA_MLA_SPLIT`, see [`split`]): its two
/// entry points (`KernelHandle(0)` when absent), the device's SM count the split rule reads,
/// and the partials scratch the launch writes (NULL, 0 partials, unless [`Self::resolve_for`]
/// installed it with the lever on or [`Self::with_split_scratch`] attached one).
#[derive(Clone, Copy)]
pub struct Glm5NextDsaDecodeKernel {
    base: KernelHandle,
    headgroup: [KernelHandle; DSA_MLA_HEADGROUPS.len()],
    prefill_tc: KernelHandle,
    prefill_tc2: KernelHandle,
    prefill_tc2_hwcvt: KernelHandle,
    split: KernelHandle,
    split_merge: KernelHandle,
    sm_count: u32,
    split_scratch: DevicePtr,
    /// 2026-10-05: Capacity of `split_scratch` in partials of
    /// [`DSA_MLA_SPLIT_PARTIAL_FLOATS`] floats each (rows x heads x splits).
    split_partials: usize,
}

impl Glm5NextDsaDecodeKernel {
    /// 2026-09-25: Resolved with `kernel()`, not `try_kernel`: a missing entry
    /// point is an error, with no dense fallback.
    /// 2026-10-01: The head-grouped entry points are optional (`try_kernel`, 0 when absent).
    /// 2026-10-01: So is the tensor-core prefill kernel.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let base = gpu.kernel(DSA_DECODE_MODULE, "glm5next_dsa_mla_decode_fp8")?;
        let headgroup = DSA_MLA_HEADGROUPS.map(|g| {
            metrale_model_layers::layers::try_kernel(
                gpu,
                DSA_DECODE_MODULE,
                &format!("glm5next_dsa_mla_decode_fp8_hg{g}"),
            )
        });
        let prefill_tc = metrale_model_layers::layers::try_kernel(
            gpu,
            MLA_PREFILL_TC_MODULE,
            MLA_PREFILL_TC_ENTRY,
        );
        // 2026-10-08: The TC2 rewrite only with its lever on, so lever off looks up exactly
        // what it did before.
        let prefill_tc2 = if prefill_tc::mla_prefill_tc2() {
            metrale_model_layers::layers::try_kernel(
                gpu,
                MLA_PREFILL_TC2_MODULE,
                MLA_PREFILL_TC2_ENTRY,
            )
        } else {
            KernelHandle(0)
        };
        // 2026-10-05: The split pair is optional too; `sm_count` is read once here, as
        // `GpuBackend::sm_count` asks (the CUDA backend asks the driver, the mock says 48).
        let split =
            metrale_model_layers::layers::try_kernel(gpu, DSA_DECODE_MODULE, DSA_MLA_SPLIT_ENTRY);
        let split_merge = metrale_model_layers::layers::try_kernel(
            gpu,
            DSA_DECODE_MODULE,
            DSA_MLA_SPLIT_MERGE_ENTRY,
        );
        let sm_count = gpu.sm_count().unwrap_or(split::FALLBACK_SM_COUNT);
        Ok(Self {
            base,
            headgroup,
            prefill_tc,
            prefill_tc2,
            prefill_tc2_hwcvt: KernelHandle(0),
            split,
            split_merge,
            sm_count,
            split_scratch: DevicePtr(0),
            split_partials: 0,
        })
    }

    /// 2026-10-05: [`Self::resolve`] for a layer of `cfg`; with `METRALE_GLM_DSA_MLA_SPLIT` on
    /// (and both split entry points resolved) it also attaches the rank's one shared split
    /// scratch (`split::shared_scratch`), sized for `cfg.local_heads` and allocated by the
    /// first layer that asks, at load. Lever off: exactly `resolve`, no allocation.
    pub fn resolve_for(gpu: &dyn GpuBackend, cfg: &Glm5NextDsaConfig) -> Result<Self> {
        let mut k = Self::resolve(gpu)?;
        if mla_split() != MlaSplit::Off && k.has_split_kernels() {
            let (ptr, partials) = split::shared_scratch(gpu, cfg.local_heads)?;
            k.split_scratch = ptr;
            k.split_partials = partials;
        }
        Ok(k)
    }

    /// 2026-10-05: This kernel with a split scratch of its own, sized for
    /// [`DSA_MLA_SPLIT_MAX_ROWS`] rows of `num_q_heads` heads at [`DSA_MLA_SPLIT_MAX`] splits
    /// ([`split_scratch_bytes`]), whatever the lever says. For the microtest, which runs the
    /// split arm directly; the caller owns (and may free) the returned pointer.
    pub fn with_split_scratch(
        mut self,
        gpu: &dyn GpuBackend,
        num_q_heads: usize,
    ) -> Result<(Self, DevicePtr)> {
        let p = gpu.alloc(split_scratch_bytes(num_q_heads))?;
        self.split_scratch = p;
        self.split_partials = split::scratch_partials(num_q_heads);
        Ok((self, p))
    }

    /// 2026-10-05: Whether both split entry points resolved.
    pub fn has_split_kernels(&self) -> bool {
        self.split.0 != 0 && self.split_merge.0 != 0
    }

    /// 2026-10-05: Whether a split launch can run: both entry points and a scratch.
    pub fn has_split(&self) -> bool {
        self.has_split_kernels() && self.split_scratch.0 != 0 && self.split_partials != 0
    }

    /// 2026-10-01: Whether `glm5next_dsa_mla_prefill_tc_fp8` resolved.
    pub fn has_prefill_tc(&self) -> bool {
        self.prefill_tc.0 != 0
    }

    /// 2026-10-08: This kernel with both TC2 entry points looked up, whatever the lever says
    /// (`KernelHandle(0)` for one that is absent). For the microtest.
    pub fn with_prefill_tc2(mut self, gpu: &dyn GpuBackend) -> Self {
        let get = |entry: &str| {
            metrale_model_layers::layers::try_kernel(gpu, MLA_PREFILL_TC2_MODULE, entry)
        };
        self.prefill_tc2 = get(MLA_PREFILL_TC2_ENTRY);
        self.prefill_tc2_hwcvt = get(MLA_PREFILL_TC2_HWCVT_ENTRY);
        self
    }

    /// 2026-10-08: Whether `glm5next_dsa_mla_prefill_tc2_fp8` resolved.
    pub fn has_prefill_tc2(&self) -> bool {
        self.prefill_tc2.0 != 0
    }

    /// 2026-10-08: Whether `glm5next_dsa_mla_prefill_tc2_hwcvt_fp8` resolved (only
    /// [`Self::with_prefill_tc2`] looks it up).
    pub fn has_prefill_tc2_hwcvt(&self) -> bool {
        self.prefill_tc2_hwcvt.0 != 0
    }

    /// 2026-10-08: The tensor-core prefill handle (`KernelHandle(0)` when absent), for the
    /// microtest's launch-only timing.
    pub fn prefill_tc_handle(&self) -> KernelHandle {
        self.prefill_tc
    }

    /// 2026-10-08: The TC2 handle: the old-converter arm when `hwcvt`.
    pub fn prefill_tc2_handle(&self, hwcvt: bool) -> KernelHandle {
        if hwcvt {
            self.prefill_tc2_hwcvt
        } else {
            self.prefill_tc2
        }
    }

    /// 2026-10-01: The `_hg{g}` handle, `None` for a `g` without an entry point or one that
    /// did not resolve.
    fn headgroup_handle(&self, g: usize) -> Option<KernelHandle> {
        let i = DSA_MLA_HEADGROUPS.iter().position(|&x| x == g)?;
        Some(self.headgroup[i]).filter(|h| h.0 != 0)
    }

    /// 2026-10-01: Whether the `_hg{g}` entry point resolved.
    pub fn has_headgroup(&self, g: usize) -> bool {
        self.headgroup_handle(g).is_some()
    }
}

/// 2026-09-25: Everything the decode reads, all caller-owned.
#[derive(Debug, Clone, Copy)]
pub struct DsaDecodeInputs {
    /// 2026-09-25: `[num_q_heads * kv_lora_rank]` BF16: this rank's absorbed
    /// queries.
    pub q: DevicePtr,
    /// 2026-09-25: FP8 paged latent cache. In absorbed NoPE MLA, K and V are the
    /// same buffer; both are taken so a caller may pass them separately.
    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    /// 2026-09-25: `[num_q_heads * kv_lora_rank]` BF16 output.
    pub out: DevicePtr,
    /// 2026-09-25: `[num_seqs, max_blocks_per_seq]` i32.
    pub block_tables: DevicePtr,
    /// 2026-09-25: `[num_seqs]` i32.
    pub seq_lens: DevicePtr,
    /// 2026-09-25: `[num_seqs, out_width]` i32:
    /// [`super::select::DsaSelectScratch::tokens`].
    pub sel_indices: DevicePtr,
    pub k_scale: f32,
    pub v_scale: f32,
}

/// 2026-09-25: Paging geometry the decode needs and the selection does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsaDecodePaging {
    pub num_seqs: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub max_blocks_per_seq: usize,
    pub block_size: usize,
    pub cache_stride_bytes: u64,
}

impl DsaDecodePaging {
    /// 2026-09-25: Refuses zero sequences, heads or `block_size`, a `block_size`
    /// that is not a multiple of `index_kpool`, and any `num_kv_heads` but 1.
    ///
    /// The `index_kpool` rule is the selector's, not this kernel's: pools are
    /// built over absolute positions, so a block that straddles a pool boundary
    /// splits a pool across two pages. The per-token gather would not notice.
    pub fn validate(&self, cfg: &Glm5NextDsaConfig) -> Result<()> {
        if self.num_seqs == 0 || self.num_q_heads == 0 {
            bail!(
                "DSA decode: degenerate launch ({} seqs, {} heads)",
                self.num_seqs,
                self.num_q_heads
            );
        }
        if self.block_size == 0 {
            bail!("DSA decode: block_size must be > 0");
        }
        if !self.block_size.is_multiple_of(cfg.index_kpool) {
            bail!(
                "DSA decode: block_size {} is not a multiple of index_kpool {} — a pool \
                 would straddle a page boundary",
                self.block_size,
                cfg.index_kpool
            );
        }
        if self.num_kv_heads != 1 {
            bail!(
                "DSA decode: MLA carries a single latent KV head, got {}",
                self.num_kv_heads
            );
        }
        Ok(())
    }
}

/// 2026-09-25: Launch the selected-index decode. Enqueued on `stream`, not
/// synchronised.
///
/// `geom.q_rows` must equal `paging.num_seqs`: one query row per sequence. A
/// mismatch would index the selection rows with the wrong stride, so it is an
/// error.
///
/// 2026-10-01: The kernel is the per-head one unless `METRALE_GLM_DSA_MLA_HEADGROUP` selects a
/// head group that divides `num_q_heads` and resolved; otherwise the per-head kernel runs and
/// the fallback is logged once.
///
/// 2026-10-05: When that is head group 8, `METRALE_GLM_DSA_MLA_SPLIT` may take the split-key
/// pair instead (`split::split_for`). [`attention`] reaches it only for decode and verify
/// rows (`is_prefill` false, `decode_attention_with`).
pub fn decode_attention(
    gpu: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
) -> Result<()> {
    decode_attention_with(gpu, kernel, cfg, geom, paging, inputs, true, stream)
}

/// 2026-10-05: [`decode_attention`], with `allow_split` false keeping the split-key path off
/// for this launch whatever `METRALE_GLM_DSA_MLA_SPLIT` says. [`attention`] passes
/// `!is_prefill`: prefill rows keep the head-grouped kernel, so the rank's one shared split
/// scratch is only ever written by the decode / verify / draft stream (a single-rank serve may
/// run a prefill on a second stream; see [`split`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_attention_with(
    gpu: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    allow_split: bool,
    stream: u64,
) -> Result<()> {
    let requested = mla_headgroup();
    let g = headgroup_for(requested, paging.num_q_heads, kernel.has_headgroup(requested));
    if g != requested {
        static FELL_BACK: std::sync::Once = std::sync::Once::new();
        FELL_BACK.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_DSA_MLA_HEADGROUP={requested}: {} - the per-head kernel runs",
                if kernel.has_headgroup(requested) {
                    format!(
                        "{} q heads per rank is not a multiple of {requested}",
                        paging.num_q_heads
                    )
                } else {
                    format!("glm5next_dsa_mla_decode_fp8_hg{requested} did not resolve")
                }
            );
        });
    }
    // 2026-10-05: `METRALE_GLM_DSA_MLA_SPLIT` (off by default) on the head-group-8 path only.
    let splits = if allow_split && g == DSA_MLA_SPLIT_HEADGROUP {
        split::split_for(&kernel, paging, geom)
    } else {
        None
    };
    if let Some(s) = splits {
        return decode_attention_split(gpu, kernel, s, cfg, geom, paging, inputs, stream);
    }
    decode_attention_headgroup(gpu, kernel, g, cfg, geom, paging, inputs, stream)
}

/// 2026-10-01: [`decode_attention`] with the head-group size fixed by the caller: 0 for the
/// per-head kernel, else one of [`DSA_MLA_HEADGROUPS`] that resolved and divides
/// `num_q_heads` (anything else is an error). The bit-parity microtest runs both arms
/// through it.
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_headgroup(
    gpu: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    g: usize,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
) -> Result<()> {
    paging.validate(cfg)?;
    if geom.q_rows != paging.num_seqs {
        bail!(
            "DSA decode: selection has {} query rows but {} sequences are being decoded; \
             the selection row stride would be wrong",
            geom.q_rows,
            paging.num_seqs
        );
    }

    // 2026-09-25: The kernel tiles 512 latent dims across 32 lanes at 16 each.
    // `Glm5NextDsaConfig::validate` checks the same bound; it is repeated here
    // because the kernel itself would not notice a mismatch.
    if cfg.kv_lora_rank != super::KERNEL_KV_LORA_DIM {
        bail!(
            "DSA decode: kv_lora_rank {} != kernel tiling {}",
            cfg.kv_lora_rank,
            super::KERNEL_KV_LORA_DIM
        );
    }

    // 2026-10-01: The head-grouped kernels take the same arguments; only the grid's x shrinks.
    let (handle, grid_x) = if g == 0 {
        (kernel.base, paging.num_q_heads)
    } else {
        let Some(h) = kernel.headgroup_handle(g) else {
            bail!("DSA decode: head group {g} has no resolved glm5next_dsa_mla_decode_fp8_hg{g}");
        };
        if !paging.num_q_heads.is_multiple_of(g) {
            bail!(
                "DSA decode: head group {g} does not divide {} q heads",
                paging.num_q_heads
            );
        }
        (h, paging.num_q_heads / g)
    };

    KernelLaunch::new(gpu, handle)
        .grid([grid_x as u32, paging.num_seqs as u32, 1])
        .block([DECODE_BLOCK, 1, 1])
        .arg_ptr(inputs.q)
        .arg_ptr(inputs.k_cache)
        .arg_ptr(inputs.v_cache)
        .arg_ptr(inputs.out)
        .arg_ptr(inputs.block_tables)
        .arg_ptr(inputs.seq_lens)
        .arg_ptr(inputs.sel_indices)
        .arg_u32(geom.out_width as u32)
        .arg_u32(paging.max_blocks_per_seq as u32)
        .arg_u32(paging.num_q_heads as u32)
        .arg_u32(paging.num_kv_heads as u32)
        .arg_u32(cfg.kv_lora_rank as u32)
        .arg_u32(paging.block_size as u32)
        // 2026-09-25: NoPE: the score scale is over the latent width, which is the
        // whole cache token (A153: `METRALE_GLM_MLA_SCALE_AUTHOR=1` switches this to the
        // model author's `qk_head_dim^-0.5`; see `mla_scale`).
        .arg_f32(mla_scale(cfg))
        .arg_f32(inputs.k_scale)
        .arg_f32(inputs.v_scale)
        .arg_u64(paging.cache_stride_bytes)
        .launch(stream)?;
    Ok(())
}

#[cfg(test)]
mod tests;
