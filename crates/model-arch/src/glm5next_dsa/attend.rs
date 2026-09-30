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

/// 2026-09-25: The selected-index MLA decode entry point.
#[derive(Clone, Copy)]
pub struct Glm5NextDsaDecodeKernel(KernelHandle);

impl Glm5NextDsaDecodeKernel {
    /// 2026-09-25: Resolved with `kernel()`, not `try_kernel`: a missing entry
    /// point is an error, with no dense fallback.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self(gpu.kernel(
            DSA_DECODE_MODULE,
            "glm5next_dsa_mla_decode_fp8",
        )?))
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
pub fn decode_attention(
    gpu: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
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

    KernelLaunch::new(gpu, kernel.0)
        .grid([paging.num_q_heads as u32, paging.num_seqs as u32, 1])
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
