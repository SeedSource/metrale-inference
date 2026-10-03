// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The opt-in FlashKDA KDA prefill (`prefill_flashkda`,
//! `METRALE_GLM_KDA_PREFILL_FLASHKDA=1`): the recurrence of a prefill sub-chunk runs in
//! MoonshotAI's FlashKDA (vendor/flashkda, MIT, unmodified; FFI in
//! `metrale_gpu_runtime::flashkda`).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - A call covers `1..=ws.max_tokens()` tokens; more is an error before any launch.
//! - When [`flashkda_refusal`] names a reason, or the call has fewer than [`FLASHKDA_MIN_ROWS`]
//!   rows, nothing is launched and the call returns `false`: the caller runs its other arms.
//! - Both states end in the layouts the decode kernels read: the conv state as
//!   `kda_tc_conv_state_tail` leaves it (the token loop's), the recurrent state FP32
//!   `[heads, head_dim (k), head_dim (v)]` K-major.
//! - Conv1d and the output gating are the existing kernels; only the recurrence moves.
//!
//! ```text
//! front_end (no gate/beta activation)  q|k|v, g_raw (f_b(f_a(h))), beta_bf16 (b_proj(h)), out_gate
//! kda_tc_conv_rows x3 (qk_channels 0)  conv + SiLU, no L2, into q | k | v contiguous (qkv_parts)
//! kda_tc_conv_state_tail               the conv state after the last row
//! kda_flk_state_t                      state [h, k, v] -> scratch [h, v, k]   (FlashKDA's layout)
//! per piece of <= piece_rows rows:
//!   kda_flk_beta_t                     beta logits [rows, h] -> [h, rows]     (into ws.beta)
//!   FlashKDA fwd                       L2(q, k), gates, chunked delta rule, state in place,
//!                                      BF16 output into conv_out
//! kda_flk_state_t                      scratch [h, v, k] -> state [h, k, v]
//! back_end (kda_o_norm_gated_bf16in)   sigmoid-gated RMSNorm on the BF16 output, then o_proj
//! ```
//!
//! The state transposes are where a per-sequence state slot is addressed: they read and write
//! `state.recurrent` wherever the caller's slot lives, so the library only ever sees the
//! workspace's own contiguous scratch (no source change to FlashKDA's TMA state descriptors).
//!
//! Buffers borrowed from the workspace (dead on this path): `qkv_parts` (the projections, dead
//! after the pack) holds the contiguous conv outputs, `ws.beta` (the FP32 sigmoid output this
//! path never writes) the transposed beta logits, `conv_out` the library's BF16 output.
//! FlashKDA's workspace and the transposed state are the workspace's own
//! [`FlashKdaScratch`], allocated only when the lever is on.
//!
//! Numerics (FlashKDA's; why this is not bit-identical to `decode_k`): the state is BF16 between
//! 16-row chunks inside a call (FP32 only at its load and store), q and k are L2-normalised
//! inside the library with the same `rsqrt(sum + 1e-6)` form, the gate and beta sigmoids use
//! `tanh.approx`, `scale` is rounded to BF16, and the core output is BF16 (FP32 on the token
//! loop). Quality is gated at model level; `flashkda_prefill_microtest` measures the drift.

use super::*;
use metrale_gpu_runtime::flashkda;

/// 2026-10-03: Fewest rows a call sends to FlashKDA; smaller calls (decode-sized tails) keep
/// the other arms. PROVISIONAL: a guess at where a 16-row-chunk library stops paying for its two
/// state transposes and host-side TMA descriptor setup; `flashkda_prefill_microtest` times 64,
/// 256, 2,048 and 8,192 rows to set it.
pub(crate) const FLASHKDA_MIN_ROWS: usize = 64;
/// 2026-10-03: Most rows one FlashKDA call takes; a longer sub-chunk is split into near-equal
/// pieces of at most this many rows (multiples of 16), the state carried in place between them.
/// Bounds the library's workspace to `flashkda::workspace_bytes(4096, heads)` (113.2 MB at 32
/// heads). PROVISIONAL.
pub(crate) const FLASHKDA_PIECE_ROWS: usize = 4096;
/// 2026-10-03: The lowest `gate_lower_bound` upstream documents for the library (`-5.0 .. 0`).
/// GLM-5.3-Flash uses -5.
const FLASHKDA_GATE_FLOOR: f32 = -5.0;
/// 2026-10-03: Most conv taps `kda_tc_conv_rows` takes (its window registers hold 8).
const FLASHKDA_CONV_TAPS_MAX: usize = 8;

/// 2026-10-03: The FlashKDA prefill's own device scratch (see the module doc).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlashKdaScratch {
    /// 2026-10-03: FlashKDA's workspace, `flashkda::workspace_bytes(piece_rows, heads)` bytes.
    ws: DevicePtr,
    ws_bytes: usize,
    /// 2026-10-03: FP32 `[heads, head_dim (v), head_dim (k)]`: the state in FlashKDA's layout.
    state_t: DevicePtr,
    /// 2026-10-03: Most rows one library call takes with this workspace.
    piece_rows: usize,
}

impl Glm5NextKdaWorkspace {
    /// 2026-10-03: Rows one FlashKDA call takes with a workspace of `max_tokens` rows:
    /// `min(max_tokens, FLASHKDA_PIECE_ROWS)` rounded up to a multiple of 16.
    fn flashkda_piece_rows(max_tokens: usize) -> usize {
        max_tokens
            .clamp(1, FLASHKDA_PIECE_ROWS)
            .div_ceil(flashkda::CHUNK)
            * flashkda::CHUNK
    }

    /// 2026-10-03: Device bytes [`Self::alloc_flashkda`] allocates for this geometry.
    pub fn bytes_flashkda(cfg: &Glm5NextKdaConfig, max_tokens: usize) -> usize {
        flashkda::workspace_bytes(Self::flashkda_piece_rows(max_tokens), cfg.heads)
            + cfg.recurrent_state_elems() * 4
    }

    /// 2026-10-03: Allocate the FlashKDA prefill's scratch (the library workspace for
    /// `min(max_tokens, FLASHKDA_PIECE_ROWS)` rows and one transposed recurrent state) and
    /// return its size in bytes. Allocates nothing and returns 0 when the library was not built
    /// (`flashkda::available()`), when the geometry is outside what the path takes, or when the
    /// scratch already exists. The loader calls it under `METRALE_GLM_KDA_PREFILL_FLASHKDA=1`
    /// only, so with the lever off the workspace is exactly what `new_split` allocated.
    pub fn alloc_flashkda(
        &mut self,
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextKdaConfig,
    ) -> Result<usize> {
        if self.flashkda.is_some() || !flashkda::available() || cfg.head_dim != flashkda::HEAD_DIM {
            return Ok(0);
        }
        let piece_rows = Self::flashkda_piece_rows(self.max_tokens);
        let ws_bytes = flashkda::workspace_bytes(piece_rows, cfg.heads);
        let state_bytes = cfg.recurrent_state_elems() * 4;
        self.flashkda = Some(FlashKdaScratch {
            ws: gpu.alloc(ws_bytes)?,
            ws_bytes,
            state_t: gpu.alloc(state_bytes)?,
            piece_rows,
        });
        Ok(ws_bytes + state_bytes)
    }

    /// 2026-10-03: Whether [`Self::alloc_flashkda`] allocated the FlashKDA scratch.
    pub fn has_flashkda(&self) -> bool {
        self.flashkda.is_some()
    }

    /// 2026-10-03: `[T, heads]` BF16 `b_proj(h)`, the beta logits before the sigmoid (what the
    /// FlashKDA prefill transposes for the library), for the microtest's timing.
    pub fn beta_bf16_ptr(&self) -> DevicePtr {
        self.beta_bf16
    }
}

/// 2026-10-03: The `(start, rows)` pieces a `k`-row call is split into for FlashKDA calls of at
/// most `cap` rows (`cap` a positive multiple of 16): the fewest pieces, near-equal, each a
/// multiple of 16 rows except possibly the last.
pub(crate) fn flashkda_pieces(k: usize, cap: usize) -> Vec<(usize, usize)> {
    debug_assert!(cap >= flashkda::CHUNK && cap.is_multiple_of(flashkda::CHUNK));
    if k == 0 {
        return Vec::new();
    }
    let n = k.div_ceil(cap);
    let step = k.div_ceil(n).div_ceil(flashkda::CHUNK) * flashkda::CHUNK;
    (0..k).step_by(step).map(|s| (s, step.min(k - s))).collect()
}

/// 2026-10-03: Why [`Glm5NextKdaLayer::prefill_flashkda`] cannot run on this build, geometry and
/// workspace, or `None` when it can. `lib` is `flashkda::available()`, `kernels`
/// [`Glm5NextKdaKernels::has_flashkda_glue`], `scratch` [`Glm5NextKdaWorkspace::has_flashkda`].
pub(crate) fn flashkda_refusal(
    cfg: &Glm5NextKdaConfig,
    lib: bool,
    kernels: bool,
    scratch: bool,
) -> Option<&'static str> {
    if !lib {
        Some("FlashKDA was not built: FLASHKDA_CUTLASS_HOME was unset at build time")
    } else if !kernels {
        Some("the target lacks a kda_flashkda_glue or kda_chunk_tc kernel")
    } else if cfg.head_dim != flashkda::HEAD_DIM {
        Some("head_dim is not 128")
    } else if cfg.conv_kernel > FLASHKDA_CONV_TAPS_MAX {
        Some("conv_kernel is above kda_tc_conv_rows' 8 taps")
    } else if !(FLASHKDA_GATE_FLOOR..=0.0).contains(&cfg.gate_lower_bound) {
        Some("gate_lower_bound is outside FlashKDA's [-5, 0]")
    } else if !scratch {
        Some("the KDA workspace has no FlashKDA scratch")
    } else {
        None
    }
}

impl Glm5NextKdaLayer {
    /// 2026-10-03: `k` prefill tokens of one sequence through FlashKDA, starting from the
    /// carried state; the projections, conv and output gating around it are the existing
    /// kernels. Returns `Ok(false)` with nothing launched when `k < FLASHKDA_MIN_ROWS` or
    /// `flashkda_refusal` names a reason (warned once), so the caller falls back. Not
    /// bit-identical to `decode_k` (module doc). Takes no snapshots.
    pub fn prefill_flashkda(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<bool> {
        if k == 0 || k > ws.max_tokens {
            bail!(
                "KDA FlashKDA prefill of {k} tokens does not fit a workspace built for {}",
                ws.max_tokens
            );
        }
        if k < FLASHKDA_MIN_ROWS {
            return Ok(false);
        }
        let refusal = flashkda_refusal(
            &self.cfg,
            flashkda::available(),
            self.kernels.has_flashkda_glue(),
            ws.flashkda.is_some(),
        );
        let scratch = match (refusal, ws.flashkda) {
            (None, Some(s)) => s,
            (why, _) => {
                kda_flashkda_fallback(why.unwrap_or("the KDA workspace has no FlashKDA scratch"));
                return Ok(false);
            }
        };
        use crate::glm5next_layer::profile;
        // 2026-10-03: The same three profile buckets as `decode_k`.
        let t_front = profile::start();
        self.front_end_with(gpu, hidden, k, ws, stream, false)?;
        profile::end(profile::KDA_FRONT, t_front, gpu, stream);
        let t_recur = profile::start();
        self.stateful_flashkda(gpu, k, state, ws, &scratch, stream)?;
        profile::end(profile::KDA_RECUR, t_recur, gpu, stream);
        let t_back = profile::start();
        // 2026-10-03: The library's BF16 output, `[k, heads, head_dim]` rows of `qkv`, sits at
        // the start of `conv_out`.
        let r = self.back_end_with(gpu, k, ws, self.kernels.flk_o_norm, ws.conv_out, stream);
        profile::end(profile::KDA_BACK, t_back, gpu, stream);
        r.map(|()| true)
    }

    /// 2026-10-03: Conv, conv-state tail, state transposes and the FlashKDA pieces over rows
    /// `0..k` (see the module doc). The caller has checked [`flashkda_refusal`].
    fn stateful_flashkda(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        scratch: &FlashKdaScratch,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, cd, d, h) = (c.qkv_dim(), c.conv_dim(), c.head_dim, c.heads);
        let taps = c.conv_kernel;

        // 2026-10-03: The conv without L2 (`qk_channels` 0: no block normalises; FlashKDA does
        // it) once per q, k and v column group, so each lands contiguous `[k, qkv]` in
        // `qkv_parts` (part i at `i * k * qkv` elements, the offsets `front_end` used). The
        // conv state and the weight are `[conv_dim, taps]`, so a group's slice starts at
        // `i * qkv * taps`. All three read the incoming conv state; the tail then writes it.
        for i in 0..3 {
            KernelLaunch::new(gpu, self.kernels.tc_conv)
                .grid([
                    div_ceil(qkv as u32, 256),
                    div_ceil(k as u32, KDA_TC_CONV_ROWS as u32),
                    1,
                ])
                .block([256, 1, 1])
                .arg_ptr(state.conv.offset(i * qkv * taps * 4))
                .arg_ptr(ws.qkv_proj.offset(i * qkv * 2))
                .arg_ptr(self.weights.conv.weight.offset(i * qkv * taps * 2))
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(ws.qkv_parts.offset(i * k * qkv * 2))
                .arg_u32(k as u32)
                .arg_u32(qkv as u32)
                .arg_u32(taps as u32)
                .arg_u32(0)
                .arg_u32(d as u32)
                .arg_f32(c.l2_eps)
                .arg_u32(cd as u32)
                .arg_u32(qkv as u32)
                .launch(stream)?;
        }
        KernelLaunch::new(gpu, self.kernels.tc_conv_tail)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state.conv)
            .arg_ptr(ws.qkv_proj)
            .arg_u32(k as u32)
            .arg_u32(cd as u32)
            .arg_u32(taps as u32)
            .arg_u32(cd as u32)
            .launch(stream)?;

        self.flk_state_transpose(gpu, state.recurrent, scratch.state_t, stream)?;
        let (q, kk, v) = (
            ws.qkv_parts,
            ws.qkv_parts.offset(k * qkv * 2),
            ws.qkv_parts.offset(2 * k * qkv * 2),
        );
        for (s, rows) in flashkda_pieces(k, scratch.piece_rows) {
            KernelLaunch::new(gpu, self.kernels.flk_beta_t)
                .grid([div_ceil(rows as u32, 256), h as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(ws.beta_bf16.offset(s * h * 2))
                .arg_ptr(ws.beta)
                .arg_u32(rows as u32)
                .arg_u32(h as u32)
                .launch(stream)?;
            let row = s * qkv * 2;
            flashkda::fwd_fp32_state(
                &flashkda::FwdArgs {
                    q: q.offset(row).0,
                    k: kk.offset(row).0,
                    v: v.offset(row).0,
                    g: ws.g_raw.offset(row).0,
                    beta_t: ws.beta.0,
                    state: scratch.state_t.0,
                    out: ws.conv_out.offset(row).0,
                    workspace: scratch.ws.0,
                    workspace_bytes: scratch.ws_bytes,
                    rows,
                    heads: h,
                    a_log: self.weights.a_log.0,
                    dt_bias: self.weights.dt_bias.0,
                    scale: 1.0 / (d as f32).sqrt(),
                    lower_bound: c.gate_lower_bound,
                },
                stream,
            )?;
        }
        self.flk_state_transpose(gpu, scratch.state_t, state.recurrent, stream)
    }

    /// 2026-10-03: Per head, `dst = src^T` over the `head_dim x head_dim` FP32 state.
    fn flk_state_transpose(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let d = self.cfg.head_dim as u32;
        KernelLaunch::new(gpu, self.kernels.flk_state_t)
            .grid([d / 32, d / 32, self.cfg.heads as u32])
            .block([32, 8, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(d)
            .launch(stream)
    }
}
