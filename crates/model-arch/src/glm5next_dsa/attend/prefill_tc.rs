// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: `METRALE_GLM_MLA_PREFILL_TC=1`: the DSA MLA attention of a prefill sub-chunk on
//! tensor cores, `glm5next_dsa_mla_prefill_tc_fp8` (module `glm5next_dsa_mla_prefill_tc`,
//! `kernels/gb10/glm-5.3-flash/nvfp4/glm5next_dsa_mla_prefill_tc.cu`), instead of the decode
//! kernel per row. One block per (32 heads, row); the heads are the MMA M dimension, each selected
//! token is gathered and converted to BF16 once per 32 heads, and S = QK^T and O += PV run as BF16
//! m16n8k16 MMAs with FP32 accumulation. Same inputs, same output layout.
//!
//! NOT byte-identical to [`super::decode_attention`]: P is rounded to BF16 for the MMA and the
//! accumulation order differs. The GPU microtest `examples/dsa_mla_prefill_tc_microtest.rs`
//! bounds the difference; model-level quality is gated separately before the lever is used.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Off by default: with the lever unset, [`attention`] is exactly [`super::decode_attention`].
//! - Decode and verify launches (`is_prefill == false`) never take the tensor-core kernel.
//! - The kernel launches only when `prefill_tc_refusal` returns `None`; otherwise the decode
//!   kernel runs and the reason is logged once.
//!
//! 2026-10-08: `METRALE_GLM_MLA_PREFILL_TC2=1` (default off): where the tensor-core kernel would
//! launch, `glm5next_dsa_mla_prefill_tc2_fp8` (module `glm5next_dsa_mla_prefill_tc2`) launches
//! instead, with the same grid, block, shared memory and arguments, writing the same output bits
//! (exact rewrite: bit-placement FP8 conversion, Q fragments in registers, double-buffered key
//! tile with 2 barriers per tile; argument in the `.cu` header). GPU gate:
//! `examples/dsa_mla_prefill_tc_microtest.rs` (bitwise against the tensor-core kernel).
//! - TC2 off: [`attention`] makes exactly the launches it made before, and
//!   [`Glm5NextDsaDecodeKernel::resolve`] looks up no TC2 entry point.
//! - TC2 on: it replaces only `glm5next_dsa_mla_prefill_tc_fp8` launches; with
//!   `METRALE_GLM_MLA_PREFILL_TC` off, a refused launch or an unresolved entry point it is ignored,
//!   logged once (`METRALE_GLM_MLA_PREFILL_TC2=1 IGNORED: <reason>`).

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{
    DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaDecodeKernel, decode_attention_with, mla_scale,
};
use crate::glm5next_dsa::{Glm5NextDsaConfig, KERNEL_KV_LORA_DIM, select::DsaSelectGeometry};

/// 2026-10-01: Module of the tensor-core prefill kernel (the `.cu` file stem).
pub const MLA_PREFILL_TC_MODULE: &str = "glm5next_dsa_mla_prefill_tc";

/// 2026-10-01: Its entry point.
pub const MLA_PREFILL_TC_ENTRY: &str = "glm5next_dsa_mla_prefill_tc_fp8";

/// 2026-10-01: Heads per block, the kernel's `TC_HEADS`; the per-rank head count must be a
/// multiple of it.
pub const MLA_PREFILL_TC_HEADS: usize = 32;

/// 2026-10-01: Widest selection row the kernel compacts in shared memory, its `TC_MAX_SEL`
/// (GLM-5.3 selects 2051).
pub const MLA_PREFILL_TC_MAX_SEL: usize = 2560;

/// 2026-10-01: Dynamic shared memory per block, the kernel's `TC_SMEM_BYTES` (the registry
/// raises the function's limit for a launch above 48 KB).
pub const MLA_PREFILL_TC_SMEM_BYTES: u32 = 99_104;

/// 2026-10-01: Threads per block: 8 warps.
const MLA_PREFILL_TC_BLOCK: u32 = 256;

/// 2026-10-08: Module of the exact rewrite (`METRALE_GLM_MLA_PREFILL_TC2`), the `.cu` file stem.
pub const MLA_PREFILL_TC2_MODULE: &str = "glm5next_dsa_mla_prefill_tc2";

/// 2026-10-08: Its entry point the lever launches (A1 bit-placement converter + A2 + A3).
pub const MLA_PREFILL_TC2_ENTRY: &str = "glm5next_dsa_mla_prefill_tc2_fp8";

/// 2026-10-08: The same dataflow with the old FP8 conversion chain (A2 + A3 only); the microtest
/// times it beside [`MLA_PREFILL_TC2_ENTRY`]. Same output bits.
pub const MLA_PREFILL_TC2_HWCVT_ENTRY: &str = "glm5next_dsa_mla_prefill_tc2_hwcvt_fp8";

/// 2026-10-08: The converter check kernel of the same module (microtest only): all 256 E4M3
/// codes through the old chain, the A1 converter and its fast path alone.
pub const MLA_PREFILL_TC2_CVT_CHECK_ENTRY: &str = "glm5next_dsa_mla_prefill_tc2_cvt_check";

/// 2026-10-08: Heads per block of the rewrite, its `TC2_HEADS` (the same as the tensor-core
/// kernel's, so the same refusals apply).
pub const MLA_PREFILL_TC2_HEADS: usize = 32;

/// 2026-10-08: Its dynamic shared memory, `TC2_SMEM_BYTES`: the tensor-core kernel's layout with
/// the Q tile replaced by a second key buffer, the same size.
pub const MLA_PREFILL_TC2_SMEM_BYTES: u32 = 99_104;

/// 2026-10-08: Logged once, on the first launch that takes the rewrite.
pub const MLA_PREFILL_TC2_ENGAGED_LINE: &str = "METRALE_GLM_MLA_PREFILL_TC2 ENGAGED";

/// 2026-10-08: `METRALE_GLM_MLA_PREFILL_TC2=1`: where `METRALE_GLM_MLA_PREFILL_TC=1` launches
/// `glm5next_dsa_mla_prefill_tc_fp8`, launch `glm5next_dsa_mla_prefill_tc2_fp8` instead (same
/// output bits). Off unless set to `1`; read once.
pub fn mla_prefill_tc2() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_MLA_PREFILL_TC2").ok();
        let on = crate::glm5next_layer::levers::parse_dsa_switch(raw.as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_MLA_PREFILL_TC2=1 - where METRALE_GLM_MLA_PREFILL_TC=1 takes \
                 {MLA_PREFILL_TC_ENTRY}, {MLA_PREFILL_TC2_ENTRY} runs (the same output bits)"
            );
        } else if let Some(r) = raw.as_deref().filter(|r| !r.is_empty() && r.trim() != "0") {
            tracing::warn!("METRALE_GLM_MLA_PREFILL_TC2={r} is not 0 or 1 - treated as off");
        }
        on
    })
}

/// 2026-10-08: Why a prefill launch with `METRALE_GLM_MLA_PREFILL_TC2=1` does not take the
/// rewrite, `None` when it does: `tc_on` is `METRALE_GLM_MLA_PREFILL_TC`, `tc_refusal` that
/// kernel's [`prefill_tc_refusal`] and `tc2_resolved` whether the rewrite's entry point resolved.
pub(crate) fn prefill_tc2_ignored(
    tc_on: bool,
    tc_refusal: Option<&str>,
    tc2_resolved: bool,
) -> Option<String> {
    if !tc_on {
        return Some(
            "METRALE_GLM_MLA_PREFILL_TC is off (TC2 replaces only that kernel's launches)"
                .to_string(),
        );
    }
    if let Some(why) = tc_refusal {
        return Some(format!(
            "{MLA_PREFILL_TC_ENTRY} does not take this launch ({why})"
        ));
    }
    if !tc2_resolved {
        return Some(format!("{MLA_PREFILL_TC2_ENTRY} did not resolve"));
    }
    None
}

/// 2026-10-08: The one `IGNORED` line of the process.
fn log_tc2_ignored(why: &str) {
    static IGNORED: std::sync::Once = std::sync::Once::new();
    IGNORED.call_once(|| {
        tracing::warn!("METRALE_GLM_MLA_PREFILL_TC2=1 IGNORED: {why}");
    });
}

/// 2026-10-01: `METRALE_GLM_MLA_PREFILL_TC=1` runs the DSA MLA attention of prefill sub-chunks on
/// tensor cores (`glm5next_dsa_mla_prefill_tc_fp8`); NOT byte-identical to the decode kernel.
/// Decode and verify keep the decode kernel. Off unless set to `1`; read once.
pub fn mla_prefill_tc() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_MLA_PREFILL_TC").ok();
        let on = crate::glm5next_layer::levers::parse_dsa_switch(raw.as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_MLA_PREFILL_TC=1 - prefill DSA MLA attention runs on tensor cores \
                 (glm5next_dsa_mla_prefill_tc_fp8; BF16 MMA, NOT byte-identical to the decode \
                 kernel)"
            );
        } else if let Some(r) = raw.as_deref().filter(|r| !r.is_empty() && r.trim() != "0") {
            tracing::warn!("METRALE_GLM_MLA_PREFILL_TC={r} is not 0 or 1 - treated as off");
        }
        on
    })
}

/// 2026-10-01: Why the tensor-core kernel cannot take this launch, `None` when it can. `resolved`
/// is whether its entry point resolved. Every refusal is a shape or layout the kernel does not
/// handle; the caller then runs the decode kernel.
pub(crate) fn prefill_tc_refusal(
    resolved: bool,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
) -> Option<String> {
    if !resolved {
        return Some(format!("{MLA_PREFILL_TC_ENTRY} did not resolve"));
    }
    if cfg.kv_lora_rank != KERNEL_KV_LORA_DIM {
        return Some(format!("kv_lora_rank {} != {KERNEL_KV_LORA_DIM}", cfg.kv_lora_rank));
    }
    if paging.num_kv_heads != 1 {
        return Some(format!("{} KV heads, the kernel takes 1", paging.num_kv_heads));
    }
    if paging.num_q_heads == 0 || !paging.num_q_heads.is_multiple_of(MLA_PREFILL_TC_HEADS) {
        return Some(format!(
            "{} q heads per rank is not a multiple of {MLA_PREFILL_TC_HEADS}",
            paging.num_q_heads
        ));
    }
    if geom.out_width > MLA_PREFILL_TC_MAX_SEL {
        return Some(format!("selection width {} > {MLA_PREFILL_TC_MAX_SEL}", geom.out_width));
    }
    // 2026-10-01: Absorbed MLA passes one pool and one scale as K and V; the kernel reads V from
    // the K tile it already holds.
    if inputs.k_cache != inputs.v_cache || inputs.k_scale.to_bits() != inputs.v_scale.to_bits() {
        return Some("distinct K and V caches or scales".to_string());
    }
    // 2026-10-01: 16-byte loads and stores: the cache base and page stride (a token is 512 bytes
    // and a lane's slice 16), and Q and O (rows and heads are multiples of 1024 bytes).
    if (inputs.k_cache.0 | inputs.q.0 | inputs.out.0 | paging.cache_stride_bytes) & 15 != 0 {
        return Some("cache, page stride, Q or O not 16-byte aligned".to_string());
    }
    None
}

/// 2026-10-01: The DSA MLA attention of `paging.num_seqs` query rows: the tensor-core kernel when
/// `is_prefill`, `METRALE_GLM_MLA_PREFILL_TC=1` and `prefill_tc_refusal` passes, else
/// [`super::decode_attention`] (which follows `METRALE_GLM_DSA_MLA_HEADGROUP`). Engagement and the
/// first fallback are logged once. 2026-10-05: The fallback runs with the split-key path
/// (`METRALE_GLM_DSA_MLA_SPLIT`) allowed only when `is_prefill` is false.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    gpu: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    is_prefill: bool,
    stream: u64,
) -> Result<()> {
    // 2026-10-08: Read only for prefill launches; off, nothing below changes.
    let tc2 = is_prefill && mla_prefill_tc2();
    if is_prefill && mla_prefill_tc() {
        let refusal = prefill_tc_refusal(kernel.has_prefill_tc(), cfg, geom, paging, inputs);
        if tc2 {
            match prefill_tc2_ignored(true, refusal.as_deref(), kernel.has_prefill_tc2()) {
                None => {
                    static TC2_ENGAGED: std::sync::Once = std::sync::Once::new();
                    TC2_ENGAGED.call_once(|| {
                        tracing::warn!(
                            "{MLA_PREFILL_TC2_ENGAGED_LINE}: {MLA_PREFILL_TC2_ENTRY} attends \
                             prefill rows in place of {MLA_PREFILL_TC_ENTRY} ({} q heads per \
                             rank, selection width {}; same output bits)",
                            paging.num_q_heads,
                            geom.out_width
                        );
                    });
                    return prefill_attention_tc2(gpu, kernel, cfg, geom, paging, inputs, stream);
                }
                Some(why) => log_tc2_ignored(&why),
            }
        }
        match refusal {
            None => {
                static ENGAGED: std::sync::Once = std::sync::Once::new();
                ENGAGED.call_once(|| {
                    tracing::warn!(
                        "METRALE_GLM_MLA_PREFILL_TC=1: ENGAGED - {MLA_PREFILL_TC_ENTRY} attends \
                         prefill rows ({} q heads per rank, selection width {})",
                        paging.num_q_heads,
                        geom.out_width
                    );
                });
                return prefill_attention_tc(gpu, kernel, cfg, geom, paging, inputs, stream);
            }
            Some(why) => {
                static FELL_BACK: std::sync::Once = std::sync::Once::new();
                FELL_BACK.call_once(|| {
                    tracing::warn!(
                        "METRALE_GLM_MLA_PREFILL_TC=1: NOT engaged ({why}) - the decode kernel \
                         attends prefill rows"
                    );
                });
            }
        }
    } else if tc2 && let Some(why) = prefill_tc2_ignored(false, None, kernel.has_prefill_tc2()) {
        log_tc2_ignored(&why);
    }
    // 2026-10-05: `METRALE_GLM_DSA_MLA_SPLIT` applies to decode and verify rows only.
    decode_attention_with(gpu, kernel, cfg, geom, paging, inputs, !is_prefill, stream)
}

/// 2026-10-01: Launch `glm5next_dsa_mla_prefill_tc_fp8` over `paging.num_seqs` rows, grid
/// `[num_q_heads / 32, rows]`. Enqueued on `stream`, not synchronised. The same checks as
/// [`super::decode_attention_headgroup`], and a launch `prefill_tc_refusal` refuses is an error
/// here (the microtest calls this directly; [`attention`] falls back instead).
pub fn prefill_attention_tc(
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
            "DSA MLA prefill TC: selection has {} query rows but {} rows are attended; the \
             selection row stride would be wrong",
            geom.q_rows,
            paging.num_seqs
        );
    }
    if let Some(why) = prefill_tc_refusal(kernel.has_prefill_tc(), cfg, geom, paging, inputs) {
        bail!("DSA MLA prefill TC: {why}");
    }

    KernelLaunch::new(gpu, kernel.prefill_tc)
        .grid([
            (paging.num_q_heads / MLA_PREFILL_TC_HEADS) as u32,
            paging.num_seqs as u32,
            1,
        ])
        .block([MLA_PREFILL_TC_BLOCK, 1, 1])
        .shared_mem(MLA_PREFILL_TC_SMEM_BYTES)
        .arg_ptr(inputs.q)
        .arg_ptr(inputs.k_cache)
        .arg_ptr(inputs.out)
        .arg_ptr(inputs.block_tables)
        .arg_ptr(inputs.seq_lens)
        .arg_ptr(inputs.sel_indices)
        .arg_u32(geom.out_width as u32)
        .arg_u32(paging.max_blocks_per_seq as u32)
        .arg_u32(paging.num_q_heads as u32)
        .arg_u32(paging.block_size as u32)
        // 2026-10-01: The decode kernel's score scale (A153 lever included) and the one cache
        // scale; the kernel folds log2(e) in itself.
        .arg_f32(mla_scale(cfg))
        .arg_f32(inputs.k_scale)
        .arg_u64(paging.cache_stride_bytes)
        .launch(stream)?;
    Ok(())
}

/// 2026-10-08: Launch `glm5next_dsa_mla_prefill_tc2_fp8` exactly as [`prefill_attention_tc`]
/// launches the tensor-core kernel (same checks, grid, block, shared memory and arguments).
/// An unresolved entry point or a launch `prefill_tc_refusal` refuses is an error here.
pub fn prefill_attention_tc2(
    gpu: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
) -> Result<()> {
    let handle = kernel.prefill_tc2_handle(false);
    launch_tc2(
        gpu,
        handle,
        MLA_PREFILL_TC2_ENTRY,
        cfg,
        geom,
        paging,
        inputs,
        stream,
    )
}

/// 2026-10-08: [`prefill_attention_tc2`] on `glm5next_dsa_mla_prefill_tc2_hwcvt_fp8` (the old
/// FP8 conversion chain); microtest timing arm.
pub fn prefill_attention_tc2_hwcvt(
    gpu: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
) -> Result<()> {
    let handle = kernel.prefill_tc2_handle(true);
    launch_tc2(
        gpu,
        handle,
        MLA_PREFILL_TC2_HWCVT_ENTRY,
        cfg,
        geom,
        paging,
        inputs,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_tc2(
    gpu: &dyn GpuBackend,
    handle: metrale_gpu_runtime::gpu::KernelHandle,
    entry: &str,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
) -> Result<()> {
    paging.validate(cfg)?;
    if geom.q_rows != paging.num_seqs {
        bail!(
            "DSA MLA prefill TC2: selection has {} query rows but {} rows are attended; the \
             selection row stride would be wrong",
            geom.q_rows,
            paging.num_seqs
        );
    }
    if handle.0 == 0 {
        bail!("DSA MLA prefill TC2: {entry} did not resolve");
    }
    // 2026-10-08: The rewrite takes exactly the launches the tensor-core kernel takes; its
    // resolution is not required for that check.
    if let Some(why) = prefill_tc_refusal(true, cfg, geom, paging, inputs) {
        bail!("DSA MLA prefill TC2: {why}");
    }

    KernelLaunch::new(gpu, handle)
        .grid([
            (paging.num_q_heads / MLA_PREFILL_TC2_HEADS) as u32,
            paging.num_seqs as u32,
            1,
        ])
        .block([MLA_PREFILL_TC_BLOCK, 1, 1])
        .shared_mem(MLA_PREFILL_TC2_SMEM_BYTES)
        .arg_ptr(inputs.q)
        .arg_ptr(inputs.k_cache)
        .arg_ptr(inputs.out)
        .arg_ptr(inputs.block_tables)
        .arg_ptr(inputs.seq_lens)
        .arg_ptr(inputs.sel_indices)
        .arg_u32(geom.out_width as u32)
        .arg_u32(paging.max_blocks_per_seq as u32)
        .arg_u32(paging.num_q_heads as u32)
        .arg_u32(paging.block_size as u32)
        .arg_f32(mla_scale(cfg))
        .arg_f32(inputs.k_scale)
        .arg_u64(paging.cache_stride_bytes)
        .launch(stream)?;
    Ok(())
}
