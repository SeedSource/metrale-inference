// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: `METRALE_GLM_DSA_MLA_SPLIT`: the split-key (flash-decoding) form of the
//! head-group-8 DSA MLA decode. The selection row is shared by S blocks along grid z
//! (`glm5next_dsa_mla_decode_fp8_hg8_split`, grid `[num_q_heads / 8, rows, S]`); each block
//! writes per-head FP32 partials (o[512], m, l) to a preallocated scratch, and
//! `glm5next_dsa_mla_split_merge` (grid `[num_q_heads, rows]`) combines them with the
//! log-sum-exp rescale into the BF16 output. The kernel file carries the layout and the
//! launch contract (`kernels/gb10/glm-5.3-flash/nvfp4/glm5next_dsa_mla_decode.cu`).
//!
//! Why: the plain `_hg8` launch is 4 x 3 = 12 blocks at 32 heads per rank and 3 verify rows
//! on a 48-SM GB10, and each of its warps walks up to `ceil(sel_width / 8)` keys in series;
//! comb16 rank 0 measured ~150 us per launch, 13 launches per step (2026-10-05). The split
//! spreads the row's keys over `S * 8` warps instead of 8.
//!
//! NOT byte-identical to `_hg8`: the keys reach the softmax in another order and the partials
//! merge in another tree. FP32 throughout, `--fmad=false` like the whole module (KERNEL.toml).
//! Gate: `crates/model-arch/examples/dsa_mla_split_microtest.rs`.
//!
//! # The split rule (`split_count`)
//!
//! `1` (auto) picks, per launch, `S = min(16, ceil(2 * sm_count / ((heads / 8) * rows)),
//! floor(sel_width / 64))`, then caps it by the scratch: enough blocks for two waves of the
//! device (`_hg8` is built `__launch_bounds__(256, 1)`, so one block per SM is assumed;
//! PROVISIONAL, its register count is not measured), but no block below
//! [`DSA_MLA_SPLIT_MIN_KEYS`] selection slots. An `S` below 2 keeps `_hg8`.
//! GLM-5.3 TP2 (32 heads, `sel_width` 2051, 48 SMs): rows 1 -> 16, rows 3 -> 8, rows 12 or
//! 16 -> 2. `sel_width` counts slots, not valid keys (the row is `-1`-padded past its valid
//! prefix, `dsa_expand_selection`); the kernel interleaves keys across the `S * 8` warps, so
//! the valid prefix is spread over every block wherever it ends. An integer `n` >= 2 forces
//! `S = min(n, 16)` (still capped by the scratch).
//!
//! # Graph capture
//!
//! `S` depends only on rows, heads, `sel_width` (`cfg.out_width()`, fixed), the SM count read
//! at resolve and the scratch capacity, so a captured (rows) shape replays the grid it was
//! captured with. No allocation and no host sync at launch: the scratch is allocated at load
//! ([`super::Glm5NextDsaDecodeKernel::resolve_for`]) and never moved or freed.
//!
//! # One scratch per rank
//!
//! As `select::shared` argues for the select scratch: the DSA layers and the MTP head run one
//! after another on one stream, and a split launch's partials are written and fully consumed
//! by the merge enqueued right after it on the same stream; nothing reads them later. The
//! split path is taken only for decode, verify and draft rows (`super::decode_attention_with`
//! with `allow_split = !is_prefill`), so a single-rank serve's separate prefill stream never
//! writes it. One rank per process, as for `select::shared`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Lever off: `split_for` returns `None` and `resolve_for` allocates nothing, so every
//!   launch is the head-grouped (or per-head) one it was.
//! - A split launch writes `rows * heads * S` partials, at most the scratch's capacity
//!   ([`decode_attention_split`] refuses more).

use std::sync::{Mutex, OnceLock};

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{DECODE_BLOCK, DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaDecodeKernel, mla_scale};
use crate::glm5next_dsa::select::DsaSelectGeometry;
use crate::glm5next_dsa::{Glm5NextDsaConfig, KERNEL_KV_LORA_DIM};

/// 2026-10-05: The split entry point (module [`super::DSA_DECODE_MODULE`]).
pub const DSA_MLA_SPLIT_ENTRY: &str = "glm5next_dsa_mla_decode_fp8_hg8_split";

/// 2026-10-05: The merge entry point (same module).
pub const DSA_MLA_SPLIT_MERGE_ENTRY: &str = "glm5next_dsa_mla_split_merge";

/// 2026-10-05: The head group the split kernel is built for (`_hg8`'s, q in shared memory).
pub const DSA_MLA_SPLIT_HEADGROUP: usize = 8;

/// 2026-10-05: The largest S a launch takes, auto or forced.
pub const DSA_MLA_SPLIT_MAX: usize = 16;

/// 2026-10-05: The most rows a split launch takes: the scratch is sized for it. A launch of
/// more rows (prefill sub-chunks) keeps `_hg8`, whose grid is already wide.
pub const DSA_MLA_SPLIT_MAX_ROWS: usize = 16;

/// 2026-10-05: Fewest selection slots auto S leaves a block (8 per warp), so q staging and
/// the partial write stay small next to the key walk. PROVISIONAL: chosen, not measured.
pub const DSA_MLA_SPLIT_MIN_KEYS: usize = 64;

/// 2026-10-05: FP32 values per partial: o over the latent width, then m and l.
pub const DSA_MLA_SPLIT_PARTIAL_FLOATS: usize = KERNEL_KV_LORA_DIM + 2;

/// 2026-10-05: Threads per merge block: `SPLIT_MERGE_THREADS` (`GLM_KV_LORA_DIM / 4`) in the
/// kernel, four latent dims each.
const SPLIT_MERGE_BLOCK: u32 = (KERNEL_KV_LORA_DIM / 4) as u32;

/// 2026-10-05: Waves of the device auto S aims for (see the module header).
const SPLIT_WAVES: usize = 2;

/// 2026-10-05: SM count assumed when the backend cannot say (Metal): GB10's 48.
pub(super) const FALLBACK_SM_COUNT: u32 = 48;

/// 2026-10-05: `METRALE_GLM_DSA_MLA_SPLIT` resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlaSplit {
    /// 2026-10-05: Unset, `0` or not a number: `_hg8` as before.
    Off,
    /// 2026-10-05: `1`: S from `split_count`'s rule per launch.
    Auto,
    /// 2026-10-05: An integer `n` >= 2: S = `min(n, DSA_MLA_SPLIT_MAX)` per launch.
    Fixed(usize),
}

/// 2026-10-05: Parse `METRALE_GLM_DSA_MLA_SPLIT` (blanks ignored): unset, empty, `0` or
/// anything that is not a non-negative integer is off, `1` is auto, `n` >= 2 forces
/// `min(n, 16)`.
pub(crate) fn parse_mla_split(v: Option<&str>) -> MlaSplit {
    match v.map(str::trim).map(str::parse::<usize>) {
        Some(Ok(1)) => MlaSplit::Auto,
        Some(Ok(n)) if n >= 2 => MlaSplit::Fixed(n.min(DSA_MLA_SPLIT_MAX)),
        _ => MlaSplit::Off,
    }
}

/// 2026-10-05: `METRALE_GLM_DSA_MLA_SPLIT`, read once. Default off. It applies only where
/// `METRALE_GLM_DSA_MLA_HEADGROUP=8` already selects `_hg8`, to decode / verify / draft
/// launches of at most [`DSA_MLA_SPLIT_MAX_ROWS`] rows.
pub fn mla_split() -> MlaSplit {
    static E: OnceLock<MlaSplit> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_MLA_SPLIT").ok();
        let m = parse_mla_split(raw.as_deref());
        let r = raw.as_deref().unwrap_or("").trim();
        match m {
            MlaSplit::Off if !r.is_empty() && r != "0" => tracing::warn!(
                "METRALE_GLM_DSA_MLA_SPLIT={r} is not 0, 1 or an integer - the split-key DSA MLA \
                 decode stays off"
            ),
            MlaSplit::Off => {}
            MlaSplit::Auto => tracing::warn!(
                "METRALE_GLM_DSA_MLA_SPLIT=1 - head-group-8 DSA MLA decode rows split the \
                 selection over S blocks (auto S, at most {DSA_MLA_SPLIT_MAX}; \
                 {DSA_MLA_SPLIT_ENTRY} + {DSA_MLA_SPLIT_MERGE_ENTRY}; NOT byte-identical)"
            ),
            MlaSplit::Fixed(n) => tracing::warn!(
                "METRALE_GLM_DSA_MLA_SPLIT={r} - head-group-8 DSA MLA decode rows split the \
                 selection over S = {n} blocks (at most {DSA_MLA_SPLIT_MAX}; \
                 {DSA_MLA_SPLIT_ENTRY} + {DSA_MLA_SPLIT_MERGE_ENTRY}; NOT byte-identical)"
            ),
        }
        m
    })
}

/// 2026-10-05: S for one launch of `rows` rows of `num_q_heads` heads over a `sel_width`-slot
/// selection, on `sm_count` SMs with a scratch of `cap_partials` partials (module header).
/// 1 means "do not split". Public for the microtest, which prints the auto S it times.
pub fn split_count(
    mode: MlaSplit,
    rows: usize,
    num_q_heads: usize,
    sel_width: usize,
    sm_count: u32,
    cap_partials: usize,
) -> usize {
    if rows == 0 || num_q_heads == 0 {
        return 1;
    }
    let want = match mode {
        MlaSplit::Off => return 1,
        MlaSplit::Fixed(n) => n,
        MlaSplit::Auto => {
            let base = (num_q_heads / DSA_MLA_SPLIT_HEADGROUP).max(1) * rows;
            let fill = (SPLIT_WAVES * sm_count.max(1) as usize).div_ceil(base);
            fill.min(sel_width / DSA_MLA_SPLIT_MIN_KEYS)
        }
    };
    let cap = cap_partials / (rows * num_q_heads);
    want.min(DSA_MLA_SPLIT_MAX).min(cap).max(1)
}

/// 2026-10-05: S for this launch when the split path should take it (`>= 2`), else `None`:
/// lever off, the entry points or the scratch missing (logged once), more than
/// [`DSA_MLA_SPLIT_MAX_ROWS`] rows, or `split_count` says 1. Engagement is logged once.
pub(crate) fn split_for(
    kernel: &Glm5NextDsaDecodeKernel,
    paging: &DsaDecodePaging,
    geom: &DsaSelectGeometry,
) -> Option<usize> {
    let mode = mla_split();
    if mode == MlaSplit::Off {
        return None;
    }
    if !kernel.has_split() {
        static MISSING: std::sync::Once = std::sync::Once::new();
        MISSING.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_DSA_MLA_SPLIT: NOT engaged ({}) - glm5next_dsa_mla_decode_fp8_hg8 \
                 runs",
                if kernel.has_split_kernels() {
                    "no split scratch: this decode kernel was resolved without `resolve_for`"
                } else {
                    "the split entry points did not resolve"
                }
            );
        });
        return None;
    }
    if paging.num_seqs > DSA_MLA_SPLIT_MAX_ROWS {
        return None;
    }
    let s = split_count(
        mode,
        paging.num_seqs,
        paging.num_q_heads,
        geom.out_width,
        kernel.sm_count,
        kernel.split_partials,
    );
    if s < 2 {
        return None;
    }
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_DSA_MLA_SPLIT: ENGAGED - {DSA_MLA_SPLIT_ENTRY} (first launch: {} rows, \
             {} q heads per rank, selection width {}, S = {s}, {} SMs)",
            paging.num_seqs,
            paging.num_q_heads,
            geom.out_width,
            kernel.sm_count
        );
    });
    Some(s)
}

/// 2026-10-05: Partials a scratch for `num_q_heads` heads holds:
/// `DSA_MLA_SPLIT_MAX_ROWS * num_q_heads * DSA_MLA_SPLIT_MAX`.
pub(super) fn scratch_partials(num_q_heads: usize) -> usize {
    DSA_MLA_SPLIT_MAX_ROWS * num_q_heads * DSA_MLA_SPLIT_MAX
}

/// 2026-10-05: Bytes of a split scratch for `num_q_heads` heads: 16 rows x heads x 16 splits x
/// (512 + 2) FP32. GLM-5.3 TP2 (32 heads per rank): 16,842,752 B (16.8 MB); TP1 (64):
/// 33,685,504 B.
pub fn split_scratch_bytes(num_q_heads: usize) -> usize {
    scratch_partials(num_q_heads) * DSA_MLA_SPLIT_PARTIAL_FLOATS * std::mem::size_of::<f32>()
}

/// 2026-10-05: The rank's one split scratch and its capacity in partials.
static SHARED: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);

/// 2026-10-05: The rank's shared split scratch for `num_q_heads` heads: the installed one when
/// it is large enough, else a new one (allocated here, logged, kept for the life of the
/// process; a smaller earlier one stays allocated for the kernels that hold it). Called at load
/// by [`super::Glm5NextDsaDecodeKernel::resolve_for`] with the lever on.
pub(super) fn shared_scratch(
    gpu: &dyn GpuBackend,
    num_q_heads: usize,
) -> Result<(DevicePtr, usize)> {
    let need = scratch_partials(num_q_heads);
    let mut slot = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(hit) = (*slot).filter(|&(_, cap)| cap >= need) {
        return Ok(hit);
    }
    let bytes = split_scratch_bytes(num_q_heads);
    let p = gpu.alloc(bytes)?;
    tracing::info!(
        "GLM DSA MLA split scratch (METRALE_GLM_DSA_MLA_SPLIT): 1 x {:.1} MB ({} rows x \
         {num_q_heads} heads x {} splits x {} FP32), shared by the DSA layers and the MTP head",
        bytes as f64 / 1e6,
        DSA_MLA_SPLIT_MAX_ROWS,
        DSA_MLA_SPLIT_MAX,
        DSA_MLA_SPLIT_PARTIAL_FLOATS
    );
    *slot = Some((p, need));
    Ok((p, need))
}

/// 2026-10-05: Launch the split pair over `paging.num_seqs` rows with `s` splits: the split
/// kernel, grid `[num_q_heads / 8, rows, s]`, then the merge, grid `[num_q_heads, rows]`, both
/// on `stream`, not synchronised. The same checks as [`super::decode_attention_headgroup`],
/// plus: both entry points and a scratch present, `s` in `1..=DSA_MLA_SPLIT_MAX`, 8 dividing
/// the heads, and `rows * heads * s` within the scratch. `s` = 1 is a valid launch (one block
/// per head group and row, keys interleaved over its 8 warps); the lever never takes it. The
/// microtest calls this directly.
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_split(
    gpu: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    s: usize,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
) -> Result<()> {
    paging.validate(cfg)?;
    if geom.q_rows != paging.num_seqs {
        bail!(
            "DSA MLA split: selection has {} query rows but {} sequences are being decoded; \
             the selection row stride would be wrong",
            geom.q_rows,
            paging.num_seqs
        );
    }
    if cfg.kv_lora_rank != KERNEL_KV_LORA_DIM {
        bail!(
            "DSA MLA split: kv_lora_rank {} != kernel tiling {KERNEL_KV_LORA_DIM}",
            cfg.kv_lora_rank
        );
    }
    if !kernel.has_split() {
        bail!(
            "DSA MLA split: {DSA_MLA_SPLIT_ENTRY} / {DSA_MLA_SPLIT_MERGE_ENTRY} did not resolve \
             or no split scratch is attached"
        );
    }
    if s == 0 || s > DSA_MLA_SPLIT_MAX {
        bail!("DSA MLA split: S = {s} is outside 1..={DSA_MLA_SPLIT_MAX}");
    }
    if !paging.num_q_heads.is_multiple_of(DSA_MLA_SPLIT_HEADGROUP) {
        bail!(
            "DSA MLA split: head group {DSA_MLA_SPLIT_HEADGROUP} does not divide {} q heads",
            paging.num_q_heads
        );
    }
    let partials = paging.num_seqs * paging.num_q_heads * s;
    if partials > kernel.split_partials {
        bail!(
            "DSA MLA split: {} rows x {} heads x S = {s} needs {partials} partials, the scratch \
             holds {}",
            paging.num_seqs,
            paging.num_q_heads,
            kernel.split_partials
        );
    }

    let rows = paging.num_seqs as u32;
    // 2026-10-05: `_hg8`'s argument list with O replaced by the scratch and S appended.
    KernelLaunch::new(gpu, kernel.split)
        .grid([
            (paging.num_q_heads / DSA_MLA_SPLIT_HEADGROUP) as u32,
            rows,
            s as u32,
        ])
        .block([DECODE_BLOCK, 1, 1])
        .arg_ptr(inputs.q)
        .arg_ptr(inputs.k_cache)
        .arg_ptr(inputs.v_cache)
        .arg_ptr(kernel.split_scratch)
        .arg_ptr(inputs.block_tables)
        .arg_ptr(inputs.seq_lens)
        .arg_ptr(inputs.sel_indices)
        .arg_u32(geom.out_width as u32)
        .arg_u32(paging.max_blocks_per_seq as u32)
        .arg_u32(paging.num_q_heads as u32)
        .arg_u32(paging.num_kv_heads as u32)
        .arg_u32(cfg.kv_lora_rank as u32)
        .arg_u32(paging.block_size as u32)
        // 2026-10-05: The decode kernel's score scale, A153 lever included (`mla_scale`).
        .arg_f32(mla_scale(cfg))
        .arg_f32(inputs.k_scale)
        .arg_f32(inputs.v_scale)
        .arg_u64(paging.cache_stride_bytes)
        .arg_u32(s as u32)
        .launch(stream)?;
    // 2026-10-05: Same rows, heads and S as the split launch: the partial layout depends on
    // all three.
    KernelLaunch::new(gpu, kernel.split_merge)
        .grid([paging.num_q_heads as u32, rows, 1])
        .block([SPLIT_MERGE_BLOCK, 1, 1])
        .arg_ptr(kernel.split_scratch)
        .arg_ptr(inputs.out)
        .arg_ptr(inputs.seq_lens)
        .arg_u32(paging.num_q_heads as u32)
        .arg_u32(cfg.kv_lora_rank as u32)
        .arg_u32(s as u32)
        .launch(stream)?;
    Ok(())
}
