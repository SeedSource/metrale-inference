// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bind one GLM-5.3 MLP site for a rank: TP-slice the dense FFN or shared expert,
//! and build the routed experts' pointer tables over GLOBAL expert ids.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - TP: `gate_proj`/`up_proj` (`[inter, hidden]`) are split by row and `down_proj`
//!   (`[hidden, inter]`) by column, so the dense output is a partial sum.
//! - EP: an expert is owned whole by one rank. In each pointer table, an id another rank owns
//!   keeps a null `packed` pointer.
//! - The router weight and bias are loaded unsliced on every rank.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextMlpConfig;
use super::weights::{
    Glm5NextDenseMlpWeights, Glm5NextExpertPtrTable, Glm5NextExpertWeights, Glm5NextMoePtrTables,
    Glm5NextMoeWeights, Nvfp4Proj,
};

/// 2026-09-25: A BF16 or F32 tensor as host `f32`, by layer-relative name.
pub type LoadFn<'a> = &'a dyn Fn(&str) -> Result<Vec<f32>>;
/// 2026-09-25: One routed expert's weights, by GLOBAL id. Experts are not sliced, so the
/// loader's closure (`bind_expert`) can return NVFP4 pointers straight from the weight store.
pub type ExpertFn<'a> = &'a dyn Fn(usize) -> Result<Glm5NextExpertWeights>;

/// 2026-09-25: Rows `[start, end)` of a `[rows, row_elems]` row-major tensor, for a
/// column-parallel projection.
fn row_slice(v: &[f32], row_elems: usize, start: usize, end: usize) -> Vec<f32> {
    v[start * row_elems..end * row_elems].to_vec()
}

/// 2026-09-25: Columns `[start, end)` of every row, for the row-parallel `down_proj`.
fn col_slice(v: &[f32], row_elems: usize, start: usize, end: usize) -> Vec<f32> {
    v.chunks(row_elems)
        .flat_map(|r| r[start..end].iter().copied())
        .collect()
}

fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

fn up_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

/// 2026-09-25: TP-slice and upload one BF16 SwiGLU MLP, a dense layer or a routed layer's
/// shared expert. `full_inter` is the unsliced width; the rank keeps `full_inter / tp`. Errors
/// when `full_inter` does not divide over TP or a tensor has the wrong element count.
pub fn build_dense_mlp(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    tp_rank: usize,
    full_inter: usize,
    prefix: &str,
    load: LoadFn<'_>,
) -> Result<Glm5NextDenseMlpWeights> {
    let tp = cfg.tp_world_size;
    if !full_inter.is_multiple_of(tp) {
        bail!("GLM MLP {prefix}: intermediate {full_inter} does not divide over tp {tp}");
    }
    let local = full_inter / tp;
    let lo = tp_rank * local;

    let get = |n: &str| -> Result<Vec<f32>> { load(&format!("{prefix}.{n}")) };

    let expect = |name: &str, v: &[f32], want: usize| -> Result<()> {
        if v.len() != want {
            bail!(
                "GLM MLP {prefix}.{name}: {} elements, expected {want}",
                v.len()
            );
        }
        Ok(())
    };

    let gate = get("gate_proj.weight")?;
    expect("gate_proj.weight", &gate, full_inter * cfg.hidden)?;
    let up = get("up_proj.weight")?;
    expect("up_proj.weight", &up, full_inter * cfg.hidden)?;
    let down = get("down_proj.weight")?;
    expect("down_proj.weight", &down, cfg.hidden * full_inter)?;

    Ok(Glm5NextDenseMlpWeights {
        gate_proj: up_bf16(gpu, &row_slice(&gate, cfg.hidden, lo, lo + local))?,
        up_proj: up_bf16(gpu, &row_slice(&up, cfg.hidden, lo, lo + local))?,
        down_proj: up_bf16(gpu, &col_slice(&down, full_inter, lo, lo + local))?,
    })
}

/// 2026-09-25: One projection's pointer table over all `num_experts` GLOBAL ids. An id another
/// rank owns keeps a null `packed` pointer, which the expert kernels skip.
fn build_expert_ptr_table(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    experts: &[Glm5NextExpertWeights],
    proj: impl Fn(&Glm5NextExpertWeights) -> Nvfp4Proj,
) -> Result<Glm5NextExpertPtrTable> {
    let n = cfg.num_experts;
    let mut packed = vec![0u8; n * 8];
    let mut scale = vec![0u8; n * 8];
    let mut scale2 = vec![0u8; n * 4];
    for id in 0..n {
        let Some(local) = cfg.local_slot(id) else {
            continue;
        };
        let p = proj(&experts[local]);
        packed[id * 8..id * 8 + 8].copy_from_slice(&p.packed.0.to_le_bytes());
        scale[id * 8..id * 8 + 8].copy_from_slice(&p.scale.0.to_le_bytes());
        scale2[id * 4..id * 4 + 4].copy_from_slice(&p.scale_2.to_le_bytes());
    }
    let packed_ptrs = gpu.alloc(packed.len())?;
    gpu.copy_h2d(&packed, packed_ptrs)?;
    let scale_ptrs = gpu.alloc(scale.len())?;
    gpu.copy_h2d(&scale, scale_ptrs)?;
    let scale2_vals = gpu.alloc(scale2.len())?;
    gpu.copy_h2d(&scale2, scale2_vals)?;
    Ok(Glm5NextExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

/// 2026-09-25: Bind one routed MoE site for this rank: the router and bias unsliced, the shared
/// expert TP-sliced, and the `local_experts` routed experts this EP rank owns.
pub fn build_moe(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    tp_rank: usize,
    full_shared_inter: usize,
    load: LoadFn<'_>,
    expert: ExpertFn<'_>,
) -> Result<Glm5NextMoeWeights> {
    // 2026-09-25: The router weight and bias are loaded whole on every rank, so every rank
    // selects the same experts.
    let router = load("mlp.gate.weight")?;
    if router.len() != cfg.num_experts * cfg.hidden {
        bail!(
            "GLM MoE mlp.gate.weight: {} elements, expected {} ({} experts x {} hidden)",
            router.len(),
            cfg.num_experts * cfg.hidden,
            cfg.num_experts,
            cfg.hidden
        );
    }
    let bias = load("mlp.gate.e_score_correction_bias")?;
    if bias.len() != cfg.num_experts {
        bail!(
            "GLM MoE e_score_correction_bias: {} entries, expected {}",
            bias.len(),
            cfg.num_experts
        );
    }

    let shared = build_dense_mlp(
        gpu,
        cfg,
        tp_rank,
        full_shared_inter,
        "mlp.shared_experts",
        load,
    )?;

    // 2026-09-25: Ascending GLOBAL id: slot `i` holds id `local_expert_range().start + i`, the
    // inverse of `Glm5NextMlpConfig::local_slot`.
    let mut experts = Vec::with_capacity(cfg.local_experts);
    for id in cfg.local_expert_range() {
        experts.push(expert(id)?);
    }

    let ptrs = Glm5NextMoePtrTables {
        gate: build_expert_ptr_table(gpu, cfg, &experts, |e| e.gate_proj)?,
        up: build_expert_ptr_table(gpu, cfg, &experts, |e| e.up_proj)?,
        down: build_expert_ptr_table(gpu, cfg, &experts, |e| e.down_proj)?,
    };

    Ok(Glm5NextMoeWeights {
        router: up_bf16(gpu, &router)?,
        // 2026-09-25: FP32: `glm5next_router_topk` reads `bias` as `const float*`.
        router_bias: up_f32(gpu, &bias)?,
        shared,
        experts,
        ptrs,
        loadpf_shared: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: `row_slice` takes rows and `col_slice` takes columns, on tensors whose values
    /// encode their index.
    #[test]
    fn dense_slicing_takes_rows_for_gate_and_columns_for_down() {
        let gate: Vec<f32> = (0..4)
            .flat_map(|r| (0..3).map(move |c| (r * 10 + c) as f32))
            .collect();
        assert_eq!(
            row_slice(&gate, 3, 2, 4),
            vec![20., 21., 22., 30., 31., 32.]
        );

        let down: Vec<f32> = (0..3)
            .flat_map(|r| (0..4).map(move |c| (r * 10 + c) as f32))
            .collect();
        assert_eq!(col_slice(&down, 4, 2, 4), vec![2., 3., 12., 13., 22., 23.]);
        // 2026-09-25: On a square tensor the two slices have the same length and different
        // values, so only the values tell them apart.
        let sq: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let by_row = row_slice(&sq, 4, 2, 4);
        let by_col = col_slice(&sq, 4, 2, 4);
        assert_eq!(by_row.len(), by_col.len());
        assert_ne!(by_row, by_col);
    }
}
