// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The synthetic GLM-5.3 layer stack `glm5next_multiseq_decode_microtest` drives:
//! three real `Glm5NextLayer`s (KDA + dense MLP, DSA + routed MoE, KDA + routed MoE) at the
//! rank-0 geometry of the TP=2/EP=2 serve, with random weights, plus the KV pool, per-sequence
//! layer states and host/device helpers.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

#![allow(dead_code)]

use anyhow::Result;
use half::bf16;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::attend::Glm5NextDsaDecodeKernel;
use metrale_model_arch::glm5next_dsa::layer::{
    Glm5NextDsaLayer, Glm5NextDsaLayerKernels, Glm5NextDsaWeights, Glm5NextDsaWorkspace,
};
use metrale_model_arch::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace,
};
use metrale_model_arch::glm5next_layer::{
    Glm5NextLayer, Glm5NextMhc, Glm5NextMixer, Glm5NextMlpSite, prefill_rows, prefill_rows_ffn,
};
use metrale_model_arch::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, mhc_mix_max_tokens, mix_hc,
};
use metrale_model_arch::glm5next_mlp::build::{build_dense_mlp, build_moe};
use metrale_model_arch::glm5next_mlp::forward::Glm5NextMlpWorkspace;
use metrale_model_arch::glm5next_mlp::weights::{Glm5NextExpertWeights, Nvfp4Proj};
use metrale_model_arch::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use metrale_model_layers::layer::{AttnMetadataDev, LayerState, TransformerLayer};
use metrale_model_layers::weight_map::DenseWeight;
use std::sync::Arc;

/// 2026-10-01: Rows the KDA, DSA and MLP workspaces hold, as the loader sizes them: the largest
/// of 16 (`PREFILL_ROWS`), `DENSE_GEMV_BATCHM_MAX_M` and `prefill_rows()`.
pub(crate) fn rows() -> usize {
    (metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize)
        .max(16)
        .max(prefill_rows())
}
/// 2026-10-01: KV block size and the per-row block-table stride of the decode metadata.
pub(crate) const BLOCK: usize = 64;
pub(crate) const MAX_BLOCKS: usize = 128;
const FULL_DENSE: usize = 12288;
const FULL_SHARED: usize = 2048;

pub(crate) struct Lcg(pub(crate) u64);
impl Lcg {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    /// 2026-10-01: Uniform in [-amp, amp].
    pub(crate) fn vec(&mut self, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() % 2001) as f32 / 1000.0 - 1.0) * amp)
            .collect()
    }
    pub(crate) fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

pub(crate) fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
pub(crate) fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect()
}
pub(crate) fn up_bf16(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(g, &bf16_bytes(v))
}
pub(crate) fn up_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(g, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}
pub(crate) fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn dw(g: &dyn GpuBackend, s: &mut Lcg, n: usize, amp: f32) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: up_bf16(g, &s.vec(n, amp))?,
    })
}
/// 2026-10-01: A norm weight around 1.0, BF16.
fn norm_w(g: &dyn GpuBackend, s: &mut Lcg, n: usize) -> Result<DevicePtr> {
    let v: Vec<f32> = s.vec(n, 0.1).iter().map(|x| 1.0 + x).collect();
    up_bf16(g, &v)
}

/// 2026-10-01: The per-rank configs: KDA at half the heads, DSA at half the attention heads,
/// the MLP at TP=2/EP=2 rank 0 (as `glm5next_ffn_staged_microtest`).
pub(crate) struct Geometry {
    pub(crate) config: ModelConfig,
    pub(crate) kda: Glm5NextKdaConfig,
    pub(crate) dsa: Glm5NextDsaConfig,
    pub(crate) mlp: Glm5NextMlpConfig,
}

pub(crate) fn geometry(config_json: &str) -> Result<Geometry> {
    let mut config = metrale_config::parse_config(config_json)?;
    // 2026-10-01: Rank 0 of the TP=2/EP=2 serve; the head counts below are halved explicitly.
    config.tp_world_size = 2;
    config.ep_world_size = 2;
    let kda = Glm5NextKdaConfig {
        hidden: config.hidden_size,
        heads: config.linear_num_value_heads / 2,
        head_dim: config.linear_value_head_dim,
        conv_kernel: config.linear_conv_kernel_dim,
        gate_lower_bound: config.linear_gate_lower_bound,
        rms_norm_eps: config.rms_norm_eps as f32,
        l2_eps: 1e-6,
        chunk: 32,
    };
    kda.validate()?;
    let mut dsa = Glm5NextDsaConfig::from_config(&config)?;
    dsa.local_heads /= 2;
    let mlp = Glm5NextMlpConfig {
        hidden: config.hidden_size,
        local_dense_intermediate: FULL_DENSE / 2,
        moe_intermediate: 2048,
        local_shared_intermediate: FULL_SHARED / 2,
        num_experts: 288,
        local_experts: 144,
        ep_rank: 0,
        top_k: 8,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 2,
        ep_world_size: 2,
    };
    mlp.validate()?;
    Ok(Geometry {
        config,
        kda,
        dsa,
        mlp,
    })
}

fn nvfp4(g: &dyn GpuBackend, s: &mut Lcg, out: usize, inn: usize) -> Result<Nvfp4Proj> {
    let packed = s.bytes(out * inn / 2);
    let scale: Vec<u8> = (0..out * inn / 16)
        .map(|_| 0x28 + (s.next() % 16) as u8)
        .collect();
    Ok(Nvfp4Proj {
        packed: up(g, &packed)?,
        scale: up(g, &scale)?,
        scale_2: 0.02,
    })
}

fn kda_weights(
    g: &dyn GpuBackend,
    c: &Glm5NextKdaConfig,
    s: &mut Lcg,
) -> Result<Glm5NextKdaWeights> {
    let (hid, qkv, d, h) = (c.hidden, c.qkv_dim(), c.head_dim, c.heads);
    Ok(Glm5NextKdaWeights {
        q_proj: dw(g, s, qkv * hid, 0.02)?,
        k_proj: dw(g, s, qkv * hid, 0.02)?,
        v_proj: dw(g, s, qkv * hid, 0.02)?,
        conv: dw(g, s, c.conv_dim() * c.conv_kernel, 0.5)?,
        f_a: dw(g, s, d * hid, 0.02)?,
        f_b: dw(g, s, qkv * d, 0.05)?,
        dt_bias: up_f32(g, &s.vec(qkv, 0.5))?,
        a_log: up_f32(g, &s.vec(h, 0.5))?,
        b_proj: dw(g, s, h * hid, 0.02)?,
        g_a: dw(g, s, d * hid, 0.02)?,
        g_b: dw(g, s, qkv * d, 0.05)?,
        o_norm: DenseWeight {
            weight: norm_w(g, s, d)?,
        },
        o_proj: dw(g, s, hid * qkv, 0.02)?,
    })
}

/// 2026-10-01: Post-transform DSA weights at this rank's shapes (`build_dsa_weights` output):
/// `q_absorb` `[local_heads * kv_lora_rank, q_lora_rank]`, `o_absorb`
/// `[hidden, local_heads * kv_lora_rank]`, the indexer replicated.
fn dsa_weights(
    g: &dyn GpuBackend,
    c: &Glm5NextDsaConfig,
    s: &mut Lcg,
) -> Result<Glm5NextDsaWeights> {
    let (hid, ql, kvl, lh) = (c.hidden, c.q_lora_rank, c.kv_lora_rank, c.local_heads);
    let (ih, d) = (c.index_heads, c.index_head_dim);
    Ok(Glm5NextDsaWeights {
        q_a_proj: up_bf16(g, &s.vec(ql * hid, 0.02))?,
        q_a_layernorm: norm_w(g, s, ql)?,
        q_absorb: up_bf16(g, &s.vec(lh * kvl * ql, 0.02))?,
        kv_a_proj: up_bf16(g, &s.vec(kvl * hid, 0.02))?,
        kv_a_layernorm: norm_w(g, s, kvl)?,
        o_absorb: up_bf16(g, &s.vec(hid * lh * kvl, 0.01))?,
        wk: up_bf16(g, &s.vec(d * hid, 0.02))?,
        k_norm_weight: norm_w(g, s, d)?,
        k_norm_bias: up_bf16(g, &s.vec(d, 0.05))?,
        compress_gate: up_bf16(g, &s.vec(d * hid, 0.02))?,
        wq_b: up_bf16(g, &s.vec(ih * d * ql, 0.02))?,
        weights_proj: up_bf16(g, &s.vec(ih * hid, 0.02))?,
        ape: up_f32(g, &s.vec(c.index_kpool * d, 0.1))?,
    })
}

fn mhc_site(
    g: &dyn GpuBackend,
    hc: usize,
    hid: usize,
    s: &mut Lcg,
) -> Result<Glm5NextMhcSiteWeights> {
    let m = mix_hc(hc);
    let scale: Vec<f32> = s.vec(3, 0.5).iter().map(|x| 1.0 + x).collect();
    Ok(Glm5NextMhcSiteWeights {
        hc_fn: up_bf16(g, &s.vec(m * hc * hid, 0.02))?,
        hc_fn_bf16: true,
        hc_scale: up_f32(g, &scale)?,
        hc_base: up_f32(g, &s.vec(m, 0.1))?,
        mix: g.alloc(mhc_mix_max_tokens() * m * 4)?,
    })
}

/// 2026-10-01: Layer 0 KDA + dense MLP (`is_first`), layer 1 DSA + routed MoE, layer 2 KDA +
/// routed MoE (`is_last`). The two MoE layers share `expert_sets` distinct NVFP4 expert weight
/// sets, cycled over the 144 local experts (144 = every expert its own weights).
pub(crate) fn build_stack(
    g: &dyn GpuBackend,
    geo: &Geometry,
    expert_sets: usize,
    s: &mut Lcg,
) -> Result<Vec<Glm5NextLayer>> {
    let cfg = &geo.config;
    let (h, hc) = (cfg.hidden_size, cfg.hc_mult);
    let kda_k = Glm5NextKdaKernels::resolve(g)?;
    let dsa_k = Glm5NextDsaKernels::resolve(g)?;
    let dsa_lk = Glm5NextDsaLayerKernels::resolve(g)?;
    let mlp_k = Glm5NextMlpKernels::resolve(g)?;
    let mhc_k = Glm5NextMhcKernels::resolve(g)?;
    let rms_norm_k = g.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
    let add_k = metrale_model_layers::layers::try_kernel(g, "bf16_add", "bf16_add_inplace");
    let rows = rows();
    let kda_ws = Arc::new(Glm5NextKdaWorkspace::new(g, &geo.kda, rows)?);
    let mlp_rows = rows.max(prefill_rows_ffn());
    let mlp_ws = Arc::new(Glm5NextMlpWorkspace::new_sized(g, &geo.mlp, mlp_rows, rows)?);

    let c = &geo.mlp;
    let mut host = std::collections::HashMap::<String, Vec<f32>>::new();
    host.insert("mlp.gate.weight".into(), s.vec(c.num_experts * h, 0.05));
    host.insert(
        "mlp.gate.e_score_correction_bias".into(),
        s.vec(c.num_experts, 0.01),
    );
    for (prefix, full) in [("mlp.shared_experts", FULL_SHARED), ("mlp", FULL_DENSE)] {
        for n in ["gate_proj", "up_proj", "down_proj"] {
            host.insert(format!("{prefix}.{n}.weight"), s.vec(full * h, 0.02));
        }
    }
    let load = |n: &str| -> Result<Vec<f32>> {
        host.get(n)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no synthetic tensor {n}"))
    };
    let mi = c.moe_intermediate;
    let sets: Vec<Glm5NextExpertWeights> = (0..expert_sets.clamp(1, c.local_experts))
        .map(|_| {
            Ok(Glm5NextExpertWeights {
                gate_proj: nvfp4(g, s, mi, h)?,
                up_proj: nvfp4(g, s, mi, h)?,
                down_proj: nvfp4(g, s, h, mi)?,
            })
        })
        .collect::<Result<_>>()?;
    let expert = |id: usize| -> Result<Glm5NextExpertWeights> { Ok(sets[id % sets.len()]) };

    let mut layers = Vec::with_capacity(3);
    for idx in 0..3usize {
        let mixer = if idx == 1 {
            Glm5NextMixer::Dsa(Box::new(Glm5NextDsaLayer {
                persist_bt: true,
                cfg: geo.dsa,
                weights: dsa_weights(g, &geo.dsa, s)?,
                kernels: dsa_lk,
                select_kernels: dsa_k,
                decode_kernel: Glm5NextDsaDecodeKernel::resolve(g)?,
                workspace: Glm5NextDsaWorkspace::new(g, &geo.dsa, rows)?,
                layer_idx: idx,
                attn_layer_idx: 0,
                rms_eps: cfg.rms_norm_eps as f32,
                kv_scale: 1.0,
            }))
        } else {
            Glm5NextMixer::Kda {
                layer: Box::new(Glm5NextKdaLayer::new(
                    idx,
                    geo.kda,
                    kda_weights(g, &geo.kda, s)?,
                    kda_k,
                )?),
                ws: kda_ws.clone(),
                cfg: geo.kda,
            }
        };
        let mlp = if idx == 0 {
            Glm5NextMlpSite::Dense(build_dense_mlp(g, c, 0, FULL_DENSE, "mlp", &load)?)
        } else {
            Glm5NextMlpSite::Moe(Box::new(build_moe(g, c, 0, FULL_SHARED, &load, &expert)?))
        };
        layers.push(Glm5NextLayer {
            layer_idx: idx,
            mixer,
            mlp,
            mlp_cfg: *c,
            mlp_kernels: mlp_k,
            mlp_ws: mlp_ws.clone(),
            mhc: Some(Glm5NextMhc {
                kernels: mhc_k,
                attn: mhc_site(g, hc, h, s)?,
                ffn: mhc_site(g, hc, h, s)?,
                hc_mult: hc,
                sinkhorn_iters: cfg.hc_sinkhorn_iters,
                hc_eps: cfg.hc_eps,
            }),
            input_norm: norm_w(g, s, h)?,
            post_attn_norm: norm_w(g, s, h)?,
            rms_norm_k,
            add_k,
            rms_eps: cfg.rms_norm_eps as f32,
            hidden: h,
            mixer_all_reduce: false,
            is_first: idx == 0,
            is_last: idx == 2,
            prefetch: Default::default(),
            dflash_tap: false,
        });
    }
    Ok(layers)
}

/// 2026-10-01: One FP8 latent pool (the single DSA layer, `attn_layer_idx` 0).
pub(crate) fn kv_cache(
    g: &dyn GpuBackend,
    dsa: &Glm5NextDsaConfig,
    blocks: usize,
) -> Result<PagedKvCache> {
    let c = KvCacheConfig {
        block_size: BLOCK,
        num_kv_heads: 1,
        head_dim: dsa.kv_lora_rank,
        num_layers: 1,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: Vec::new(),
        layer_dims: Vec::new(),
        cache_blocks_per_seq: None,
    };
    PagedKvCache::new(c, blocks, g)
}

/// 2026-10-01: One sequence: its length so far, its KV blocks and one state per layer, from
/// `alloc_state` (KDA: a zeroed, pool-free `SsmLayerState`; DSA: a `Glm5NextDsaState`).
pub(crate) struct Seq {
    pub(crate) len: usize,
    pub(crate) blocks: Vec<u32>,
    pub(crate) states: Vec<Box<dyn LayerState>>,
}

pub(crate) fn new_seq(
    g: &dyn GpuBackend,
    stack: &[Glm5NextLayer],
    first_block: u32,
    n_blocks: usize,
) -> Result<Seq> {
    let states = stack
        .iter()
        .map(|l| l.alloc_state(g))
        .collect::<Result<Vec<_>>>()?;
    Ok(Seq {
        len: 0,
        blocks: (first_block..first_block + n_blocks as u32).collect(),
        states,
    })
}

pub(crate) fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
pub(crate) fn env_list(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

pub(crate) fn row_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}
pub(crate) fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// 2026-10-01: Device arrays for up to `rows` decode rows of metadata.
pub(crate) struct MetaBufs {
    pos: DevicePtr,
    slot: DevicePtr,
    sl: DevicePtr,
    bt: DevicePtr,
}

impl MetaBufs {
    pub(crate) fn new(g: &dyn GpuBackend, rows: usize) -> Result<Self> {
        Ok(Self {
            pos: g.alloc(rows * 4)?,
            slot: g.alloc(rows * 8)?,
            sl: g.alloc(rows * 4)?,
            bt: g.alloc(rows * MAX_BLOCKS * 4)?,
        })
    }

    /// 2026-10-01: Row `r` is the next token of `seqs[r]`: position `len`, its paged slot,
    /// `seq_len` `len + 1` and its block-table row (stride `MAX_BLOCKS`), as
    /// `upload_batch_metadata_fixed` lays them out.
    pub(crate) fn upload(&self, g: &dyn GpuBackend, seqs: &[&Seq]) -> Result<AttnMetadataDev> {
        let (mut pos, mut slot, mut sl) = (Vec::new(), Vec::new(), Vec::new());
        let mut bt = vec![0u8; seqs.len() * MAX_BLOCKS * 4];
        for (r, s) in seqs.iter().enumerate() {
            let p = s.len;
            let phys = s.blocks[p / BLOCK] as usize;
            pos.extend((p as i32).to_le_bytes());
            slot.extend(((phys * BLOCK + p % BLOCK) as i64).to_le_bytes());
            sl.extend(((p + 1) as i32).to_le_bytes());
            for (j, b) in s.blocks.iter().enumerate() {
                let o = (r * MAX_BLOCKS + j) * 4;
                bt[o..o + 4].copy_from_slice(&b.to_le_bytes());
            }
        }
        g.synchronize(0)?;
        g.copy_h2d(&pos, self.pos)?;
        g.copy_h2d(&slot, self.slot)?;
        g.copy_h2d(&sl, self.sl)?;
        g.copy_h2d(&bt, self.bt)?;
        Ok(AttnMetadataDev {
            positions: self.pos,
            positions_h: DevicePtr::NULL,
            positions_w: DevicePtr::NULL,
            slot: self.slot,
            seq_len: self.sl,
            block_table: self.bt,
            max_blocks_per_seq: MAX_BLOCKS as u32,
            num_seqs: seqs.len() as u32,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        })
    }
}
