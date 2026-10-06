// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Glm5NextDsaLayer`, one GLM-5.3 DSA block: the projections, the indexer
//! cache write, token selection and the NoPE MLA gather-attend over the selected tokens.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `decode_k` fails before its first launch when the indexer cache is behind the sequence
//!   (`len < seq_len`) or `k` is 0 or above the workspace's `max_rows`. An indexer cache
//!   ahead of the sequence is rewound to `seq_len` first.
//! - `indexer_forward` checks that the indexer cache has room (`ensure_room`) before it
//!   writes.
//! - 2026-10-06: With `METRALE_GLM_DSA_POOL_CACHE=1` that check includes the ring bound, and
//!   `check_lockstep`'s rewind the rewind bound (`pool_cache::RingBook`); the replay pre-check
//!   (`check_replay_room`) runs both for the rewind and write a replay will do.
//!
//! Decode, end to end:
//!
//! ```text
//! hidden ─┬─ q_a_proj ─ RMSNorm ─┬─ q_absorb ────────────── Q (latent space)
//!         │                      └─ indexer.wq_b ────────── q_idx  ─┐
//!         ├─ indexer.wk ─ LayerNorm(w,b) ─ state.k_normed ──────────┤
//!         ├─ compress_gate ─────────────── state.gate ──────────────┼─ select_tokens
//!         ├─ weights_proj ──────────────── head weights ────────────┘        │
//!         └─ kv_a_proj ─ RMSNorm ─ FP8 ─── paged latent cache                │
//!                                                                            ▼
//!                                            glm5next_dsa_mla_decode_fp8 (gather)
//! ```
//!
//! # Kernel choices the shapes do not check
//!
//! * The norms use `rms_norm_vanilla`, `x * rms * w`. The `rms_norm` kernel (module `norm`)
//!   computes `x * rms * (1 + w)` with the same signature.
//! * `indexer.k_norm` is a LayerNorm with a bias: `nllb_layernorm_bf16(x, w, b, …)`.
//! * `weights_proj` carries `index_heads^-0.5` from load ([`Glm5NextDsaWeights`]), because
//!   `dsa_index_scores` does not apply it.
//! * Q reaches the decode kernel as `q_absorb`: `q_b_proj` multiplied through `kv_b_proj`'s
//!   K half, so it is in the `kv_lora_rank`-wide latent space the kernel dots against. The
//!   raw `q_b_proj` is `qk_head_dim` (256) per head.

use anyhow::{Context, Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::attend::{DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaDecodeKernel, attention};
use super::state::Glm5NextDsaState;
use super::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use metrale_model_layers::layer::{ForwardContext, LayerState, TransformerLayer};

mod ctx_rows;
mod decode_k;
mod proj_gemm;
mod row_batch;
mod row_src;
mod rows;
mod wide;
mod workspace;
mod xseq;

use proj_gemm::gemm;
pub use wide::DsaWideArena;
pub use workspace::Glm5NextDsaWorkspace;
pub(crate) use workspace::batch_select_enabled;
pub use xseq::{DsaXseqArena, dsa_xseq_batch};

/// 2026-09-25: The projection, norm and latent-write kernels of a DSA block. The selection
/// kernels are in `Glm5NextDsaKernels`, the decode kernel in `attend`.
#[derive(Clone, Copy)]
pub struct Glm5NextDsaLayerKernels {
    /// 2026-09-25: `dense_gemm_bf16`, `C = A @ B^T`, BF16 out.
    pub gemm: KernelHandle,
    /// 2026-09-25: The same with FP32 out, for the selector's `q_idx` and head weights.
    pub gemm_f32: KernelHandle,
    /// 2026-09-25: M = 1 kernels for `gemm` and `gemm_f32`. `gemv_f32` is `KernelHandle(0)`
    /// on a target without `dense_gemv_bf16_fp32out`, and `dense_mm_bf16` then uses the tile
    /// GEMM.
    pub gemv: KernelHandle,
    pub gemv_f32: KernelHandle,
    /// 2026-09-25: `dense_gemv_bf16_batchm`: 2..=`DENSE_GEMV_BATCHM_MAX_M` rows in one pass
    /// over the weight; `KernelHandle(0)` when the target lacks it.
    pub gemv_batchm: KernelHandle,
    /// 2026-09-25: `rms_norm_vanilla`; see the module header.
    pub rms_norm: KernelHandle,
    /// 2026-09-25: `glm5next_mla_latent_write_fp8`: RMSNorm, FP8 quantisation and the paged
    /// slot write of the KV latent.
    pub latent_write: KernelHandle,
}

impl Glm5NextDsaLayerKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            // 2026-09-25: `kernels/gb10/common/KERNEL.toml` `[modules]` maps
            // `dense_gemm_bf16 = "gemm"` and `dense_gemv_bf16 = "gemv"`; an unlisted `.cu`
            // file's module is its file stem.
            gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            gemm_f32: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            gemv_f32: metrale_model_layers::layers::try_kernel(
                gpu,
                "gemv",
                "dense_gemv_bf16_fp32out",
            ),
            gemv_batchm: metrale_model_layers::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            latent_write: gpu
                .kernel("glm5next_mla_latent_write", "glm5next_mla_latent_write_fp8")?,
        })
    }
}

/// 2026-09-25: One DSA block's weights on the device, already sharded for this rank.
pub struct Glm5NextDsaWeights {
    pub q_a_proj: DevicePtr,
    pub q_a_layernorm: DevicePtr,
    /// 2026-09-25: `[local_heads * kv_lora_rank, q_lora_rank]` BF16: `q_b_proj` absorbed
    /// through `kv_b_proj`'s K half, so Q arrives in latent space.
    pub q_absorb: DevicePtr,
    pub kv_a_proj: DevicePtr,
    pub kv_a_layernorm: DevicePtr,
    /// 2026-09-25: `[hidden, local_heads * kv_lora_rank]` BF16, row-parallel; the caller
    /// all-reduces the output.
    ///
    /// `o_proj` with `kv_b_proj`'s V half folded in (`build::absorb_o`), because the decode
    /// kernel's output is in latent space, `kv_lora_rank` per head. The checkpoint `o_proj`
    /// is `v_head_dim` per head (256 against 512 on GLM-5.3).
    pub o_absorb: DevicePtr,
    // 2026-09-25: The indexer weights below are replicated on every rank (`tp.rs`).
    pub wk: DevicePtr,
    pub k_norm_weight: DevicePtr,
    pub k_norm_bias: DevicePtr,
    pub compress_gate: DevicePtr,
    pub wq_b: DevicePtr,
    /// 2026-09-25: Multiplied by `index_heads^-0.5` at load; `dsa_index_scores` does not
    /// apply it.
    pub weights_proj: DevicePtr,
    /// 2026-09-25: `[index_kpool, index_head_dim]` FP32; the checkpoint stores BF16.
    pub ape: DevicePtr,
}

pub struct Glm5NextDsaLayer {
    pub cfg: Glm5NextDsaConfig,
    pub weights: Glm5NextDsaWeights,
    pub kernels: Glm5NextDsaLayerKernels,
    pub select_kernels: Glm5NextDsaKernels,
    pub decode_kernel: Glm5NextDsaDecodeKernel,
    pub workspace: Glm5NextDsaWorkspace,
    /// 2026-09-25: Index in the model's layer stack; used only in error messages.
    pub layer_idx: usize,
    /// 2026-09-25: Index into the KV pool (`kv_cache.k_pool_ptr`): the loader counts DSA
    /// layers only, so it runs 0..11 on GLM-5.3 while `layer_idx` runs to 45. The MTP drafter
    /// uses 0.
    pub attn_layer_idx: usize,
    pub rms_eps: f32,
    /// 2026-09-25: FP8 latent-cache scale. The decode reads with it and the latent write
    /// takes `1/scale`.
    pub kv_scale: f32,
    /// 2026-09-25: Use the workspace's `bt`/`sl` rather than allocating them per call. The
    /// loaders set it true unless `METRALE_GLM_DSA_ALLOC_PER_STEP=1`.
    pub persist_bt: bool,
}

impl Glm5NextDsaLayer {
    /// 2026-09-25: Project `hidden` into indexer cache row `state.len()`, then advance by one.
    ///
    /// With `pos_dev`, `k_normed` and `gate` go to the workspace staging rows and
    /// `dsa_indexer_store` places them at the device-side position; without it they are
    /// written straight into the cache row. Also leaves this row's selector head weights in
    /// the workspace. Fails before any write when the cache is full.
    pub fn indexer_forward(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        state: &mut Glm5NextDsaState,
        pos_dev: Option<DevicePtr>,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Checked before any write: everything below writes row `state.len()`,
        // which is past the end of the buffers when the cache is full.
        state.ensure_room(1)?;
        let d = self.cfg.index_head_dim;
        let pos = state.len();
        let off = state.row_offset(pos);
        let w = &self.workspace;
        // 2026-10-06: Pool cache on, `off` is the row's ring slot (`row_offset`); a host-path
        // write first clamps the device `pk_len` after a rewind (`pool_clamp_before_write`).
        let (k_dst, gate_dst) = match pos_dev {
            Some(_) => (w.stage_k, w.stage_gate),
            None => {
                self.pool_clamp_before_write(gpu, state, pos, stream)?;
                (state.k_normed.offset(off), state.gate.offset(off))
            }
        };

        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.wk,
            k_dst,
            1,
            d,
            self.cfg.hidden,
            stream,
        )?;
        KernelLaunch::new(gpu, self.select_kernels.k_norm)
            .grid([1, 1, 1])
            .block([d.min(1024) as u32, 1, 1])
            .shared_mem((d.min(1024) * 4) as u32)
            .arg_ptr(k_dst)
            .arg_ptr(self.weights.k_norm_weight)
            .arg_ptr(self.weights.k_norm_bias)
            .arg_u32(1)
            .arg_u32(d as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;

        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.compress_gate,
            gate_dst,
            1,
            d,
            self.cfg.hidden,
            stream,
        )?;

        // 2026-09-25: Selector head weights from the layer input, FP32 out. `weights_proj` is
        // `[index_heads, hidden]`; the reference computes
        // `weights_proj(hidden) * index_heads**-0.5`
        // (`glm5next_dsa_ref/gen_dsa_indexer_golden.py`), and the scale is already in the
        // weight (`build.rs` transform 2).
        gemm(
            gpu,
            self.kernels.gemm_f32,
            self.kernels.gemv_f32,
            // 2026-09-25: No FP32-out batched GEMV kernel exists.
            KernelHandle(0),
            hidden,
            self.weights.weights_proj,
            self.workspace.head_weights,
            1,
            self.cfg.index_heads,
            self.cfg.hidden,
            stream,
        )?;

        self.store_indexer_row(gpu, state, pos_dev, pos, d, stream)
    }

    /// 2026-09-25: The gather-attend for all `rows` query rows in one launch.
    ///
    /// Grid y is the row. Each row reads the paged latent cache through its own `q_abs` row,
    /// selection row and `seq_lens` entry, and the block-table row at stride
    /// `max_blocks_per_seq` (0 when the rows share one table). 2026-10-01: `is_prefill` lets
    /// `METRALE_GLM_MLA_PREFILL_TC=1` take the tensor-core kernel (`attend::attention`).
    #[allow(clippy::too_many_arguments)]
    fn attend_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        state: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        block_table_dev: DevicePtr,
        seq_lens_dev: DevicePtr,
        paging: &DsaDecodePaging,
        is_prefill: bool,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        self.attend_rows_at(
            gpu,
            rows,
            state,
            kv_cache,
            block_table_dev,
            seq_lens_dev,
            paging,
            is_prefill,
            w.q_abs,
            w.attn_out,
            stream,
        )
    }

    /// 2026-10-03: `attend_rows` reading the absorbed queries from `q` and writing `out`
    /// (`[rows, local_heads * kv_lora_rank]` BF16 each), the selection from the workspace.
    #[allow(clippy::too_many_arguments)]
    fn attend_rows_at(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        state: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        block_table_dev: DevicePtr,
        seq_lens_dev: DevicePtr,
        paging: &DsaDecodePaging,
        is_prefill: bool,
        q: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        use crate::glm5next_layer::profile;
        let w = &self.workspace;
        // 2026-09-25: `decode_attention` reads only `out_width` and `q_rows` from it.
        let geom = state.geometry(&self.cfg, rows)?;
        let paging = DsaDecodePaging {
            num_seqs: rows,
            ..*paging
        };
        let t = profile::start();
        let pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        attention(
            gpu,
            self.decode_kernel,
            &self.cfg,
            &geom,
            &paging,
            &DsaDecodeInputs {
                q,
                k_cache: pool,
                v_cache: pool, // 2026-09-25: absorbed NoPE MLA: K and V are the same latent
                out,
                block_tables: block_table_dev,
                seq_lens: seq_lens_dev,
                sel_indices: w.select.tokens(),
                k_scale: self.kv_scale,
                v_scale: self.kv_scale,
            },
            is_prefill,
            stream,
        )?;
        profile::end(profile::DSA_ATTEND, t, gpu, stream);
        Ok(())
    }

    /// 2026-09-26: `decode_k`'s check that the indexer cache is in lockstep with the KV
    /// cache: rewinds `st` when it is ahead of `seq_len`, fails when it is behind.
    fn check_lockstep(&self, st: &mut Glm5NextDsaState, seq_len: usize) -> Result<()> {
        // 2026-09-25: The indexer cache must advance in lockstep with the KV cache.
        //
        // * Ahead (`len > seq_len`) follows a rejected speculative draft: rewinding to
        //   `seq_len` makes the rows past it unreachable (the selector reads `[0, len)`) and
        //   the next write overwrites them.
        // * Behind (`len < seq_len`) means rows were never written, which is an error.
        match st.len().cmp(&seq_len) {
            std::cmp::Ordering::Greater => st.rewind_to(seq_len)?,
            std::cmp::Ordering::Less => bail!(
                "DSA layer {}: indexer cache holds {} tokens but the sequence is at {} — \
                 rows are MISSING, not merely stale. The indexer stream must advance in \
                 lockstep with the KV cache.",
                self.layer_idx,
                st.len(),
                seq_len
            ),
            std::cmp::Ordering::Equal => {}
        }
        Ok(())
    }
}

impl TransformerLayer for Glm5NextDsaLayer {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(Glm5NextDsaState::alloc(gpu, &self.cfg)?))
    }

    /// 2026-09-25: Frees a `Glm5NextDsaState`; any other state type is left alone. The
    /// composite `Glm5NextLayer` has its own override that does the same.
    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(dsa) = state.as_any_mut().downcast_mut::<Glm5NextDsaState>() {
            dsa.free(gpu)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_k(
            hidden,
            1,
            state,
            kv_cache,
            seq_len,
            block_table,
            ctx,
            stream,
            // 2026-09-25: `is_prefill`: a single-token decode is not a prefill sub-chunk.
            false,
        )
    }
}

impl metrale_model_layers::layer::LayerCapabilities for Glm5NextDsaLayer {}
impl metrale_model_layers::layer::LayerWeightSetup for Glm5NextDsaLayer {}
impl metrale_model_layers::layer::LayerWriteOnAccept for Glm5NextDsaLayer {}

impl metrale_model_layers::layer::LayerGraphHooks for Glm5NextDsaLayer {
    /// 2026-09-25: Fails when a replay writing indexer rows up to `seq_len + k` would pass the
    /// cache capacity; the error carries the context "DSA replay pre-check".
    fn check_replay_room(&self, state: &dyn LayerState, seq_len: usize, k: usize) -> Result<()> {
        state
            .as_any()
            .downcast_ref::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?
            .replay_room(seq_len, k)
            .with_context(|| {
                format!("DSA replay pre-check (before launch_graph, seq_len {seq_len} + k {k})")
            })
    }

    /// 2026-09-25: A replayed graph writes the indexer rows (positions from device memory)
    /// without running `decode`, so this moves the host-side row counter to where `decode_k`
    /// would have left it (`Glm5NextDsaState::sync_to`).
    fn sync_replayed_step(
        &self,
        state: &mut dyn LayerState,
        seq_len: usize,
        k: usize,
    ) -> Result<()> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?
            .sync_to(seq_len, k)
    }
}

impl metrale_model_layers::layer::LayerAuxState for Glm5NextDsaLayer {}
impl metrale_model_layers::layer::LayerSplitPrefill for Glm5NextDsaLayer {}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod bt_trim_tests;
