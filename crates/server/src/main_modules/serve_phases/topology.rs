// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Serve topology: TP and EP sizes and ranks, the TP-local head
//! counts, and the NCCL communicator.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - A `Topology` from `resolve_topology` has `world_size == tp_size * ep_size`,
//!   or `world_size == tp_size == ep_size > 1`.

#[cfg(feature = "nccl")]
use anyhow::Context;
use anyhow::Result;

use metrale_config::ModelConfig;

use crate::cli;

pub(crate) struct Topology {
    pub(crate) world_size: usize,
    pub(crate) tp_size: usize,
    pub(crate) ep_size: usize,
    pub(crate) tp_rank: usize,
    pub(crate) ep_rank: usize,
}

pub(crate) fn resolve_topology(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
) -> Result<Topology> {
    let (tp_size, ep_size) = if args.tp_size == 1 && args.ep_size == 1 && args.world_size > 1 {
        (1usize, args.world_size)
    } else {
        (args.tp_size.max(1), args.ep_size.max(1))
    };
    let derived_world = if tp_size == ep_size {
        tp_size
    } else {
        tp_size * ep_size
    };
    let world_size = if args.world_size <= 1 && (tp_size > 1 || ep_size > 1) {
        tracing::info!(
            "Auto-derived world_size={} from --tp-size {} --ep-size {} (rule: \
             tp==ep → overlapping = tp; else orthogonal = tp×ep). Pass \
             --world-size to override.",
            derived_world,
            tp_size,
            ep_size,
        );
        derived_world
    } else {
        args.world_size
    };
    let (tp_rank, ep_rank) = if tp_size == ep_size && tp_size == world_size && tp_size > 1 {
        (args.rank, args.rank)
    } else if world_size == tp_size * ep_size {
        (args.rank % tp_size, args.rank / tp_size)
    } else {
        anyhow::bail!(
            "Invalid parallelism topology: world_size={} but tp_size={} × ep_size={} = {}. \
             Either use orthogonal mesh (world = tp × ep) or overlapping groups \
             (world = tp = ep, used for 2-GPU TP+EP composition).",
            world_size,
            tp_size,
            ep_size,
            tp_size * ep_size,
        );
    };
    config.tp_rank = tp_rank;
    config.tp_world_size = tp_size;
    // 2026-09-26: Loaders size per-sequence state from `serve_max_seq_len`,
    // e.g. the GLM-5.3 DSA indexer cache (`glm5next_dsa/mod.rs`).
    config.serve_max_seq_len = args.max_seq_len;
    config.ep_rank = ep_rank;
    config.ep_world_size = ep_size;
    if tp_size > 1 {
        let loader = metrale_model_engine::factory::loader_for_config(config)?;
        if !loader.supports_tp() {
            anyhow::bail!(
                "TP (--tp-size > 1) is not supported by the {} weight loader. \
                 Run with --tp-size 1 (EP-only). To extend TP to this architecture, \
                 wire `crate::tp_shard::slice_for_rank` per attention/MoE/SSM \
                 tensor in the loader and override `ModelWeightLoader::supports_tp()` \
                 to return true. See `weight_loader/minimax.rs` as the reference.",
                config.model_type,
            );
        }
        drop(loader);
        if !config.num_attention_heads.is_multiple_of(tp_size) {
            anyhow::bail!(
                "TP requires num_attention_heads ({}) divisible by tp_size ({})",
                config.num_attention_heads,
                tp_size,
            );
        }
        if !config.num_key_value_heads.is_multiple_of(tp_size) {
            anyhow::bail!(
                "TP requires num_key_value_heads ({}) divisible by tp_size ({})",
                config.num_key_value_heads,
                tp_size,
            );
        }
        config.num_attention_heads /= tp_size;
        config.num_key_value_heads /= tp_size;
        if config.linear_num_key_heads > 0 || config.linear_num_value_heads > 0 {
            if !config.linear_num_key_heads.is_multiple_of(tp_size) {
                anyhow::bail!(
                    "TP requires linear_num_key_heads ({}) divisible by tp_size ({})",
                    config.linear_num_key_heads,
                    tp_size,
                );
            }
            if !config.linear_num_value_heads.is_multiple_of(tp_size) {
                anyhow::bail!(
                    "TP requires linear_num_value_heads ({}) divisible by tp_size ({})",
                    config.linear_num_value_heads,
                    tp_size,
                );
            }
            config.linear_num_key_heads /= tp_size;
            config.linear_num_value_heads /= tp_size;
        }
        tracing::info!(
            "TP-local head counts: num_attention_heads={}, num_key_value_heads={}, \
             linear_num_key_heads={}, linear_num_value_heads={}",
            config.num_attention_heads,
            config.num_key_value_heads,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
        );
    }
    if world_size > 1 {
        let (start, end) = config.local_expert_range();
        tracing::info!(
            "Parallelism: global rank {}/{} (tp_rank={}/{}, ep_rank={}/{}), local experts [{}, {})",
            args.rank,
            world_size,
            tp_rank,
            tp_size,
            ep_rank,
            ep_size,
            start,
            end,
        );
    }
    Ok(Topology {
        world_size,
        tp_size,
        ep_size,
        tp_rank,
        ep_rank,
    })
}

/// 2026-09-26: The NCCL communicator, or `None` when `world_size <= 1`. The
/// receive buffer holds `max_batch_tokens x max(hidden_size, vocab_size)`
/// elements of `ALL_REDUCE_DTYPE_BYTES` (`required_model_recv_bytes`).
#[cfg(feature = "nccl")]
pub(crate) fn init_nccl_comm(
    args: &cli::ServeArgs,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    world_size: usize,
    max_batch_tokens: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<Option<std::sync::Arc<dyn metrale_comm::CommBackend>>> {
    use metrale_comm::CommBackend;
    if world_size <= 1 {
        return Ok(None);
    }
    // 2026-10-03: `METRALE_RECV_BUFFER_TIGHT=1` (race #79, default off) sizes for
    // activations plus a few logits rows instead of `max_batch_tokens` logits rows.
    let tight = std::env::var("METRALE_RECV_BUFFER_TIGHT").as_deref() == Ok("1");
    let recv_capacity = if tight {
        metrale_comm::nccl_backend::required_model_recv_bytes_tight(
            max_batch_tokens,
            hidden_size,
            vocab_size,
        )
    } else {
        metrale_comm::nccl_backend::required_model_recv_bytes(
            max_batch_tokens,
            hidden_size,
            vocab_size,
        )
    }
    .context("Failed to size the NCCL receive buffer")?;
    if tight {
        tracing::info!(
            "METRALE_RECV_BUFFER_TIGHT=1: recv_buffer = max(max_batch_tokens × hidden_size, \
             {} × vocab_size) × {} B",
            metrale_comm::nccl_backend::TIGHT_LOGIT_ROWS,
            metrale_comm::nccl_backend::ALL_REDUCE_DTYPE_BYTES,
        );
    }
    tracing::info!(
        "Initializing NCCL: rank {}/{}, master {}:{}, recv_buffer {} MiB \
         (max_batch_tokens={} × max(hidden_size,vocab_size)={} × {} B)",
        args.rank,
        world_size,
        args.master_addr,
        args.master_port,
        recv_capacity / (1024 * 1024),
        max_batch_tokens,
        hidden_size.max(vocab_size),
        metrale_comm::nccl_backend::ALL_REDUCE_DTYPE_BYTES,
    );
    let cuda_stream = gpu.default_stream();
    let backend = metrale_comm::NcclBackend::new(
        args.rank,
        world_size,
        &args.master_addr,
        args.master_port,
        cuda_stream,
        recv_capacity,
    )
    .context("Failed to initialize NCCL")?;
    tracing::info!("NCCL initialized: rank {}", backend.rank());
    Ok(Some(
        std::sync::Arc::new(backend) as std::sync::Arc<dyn metrale_comm::CommBackend>
    ))
}

/// 2026-09-26: A `cuda` build without `nccl` has no collectives:
/// `world_size > 1` is an error, and otherwise there is no communicator.
#[cfg(all(feature = "cuda", not(feature = "nccl")))]
pub(crate) fn init_nccl_comm(
    _args: &cli::ServeArgs,
    _gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    world_size: usize,
    _max_batch_tokens: usize,
    _hidden_size: usize,
    _vocab_size: usize,
) -> Result<Option<std::sync::Arc<dyn metrale_comm::CommBackend>>> {
    if world_size > 1 {
        anyhow::bail!(
            "multi-rank NCCL is not available in this build (cuda feature \
             without nccl — SCALE/AMD gfx1151 has no NCCL library); \
             single-device only"
        );
    }
    Ok(None)
}

/// 2026-09-26: A `metal` build has no collectives: `world_size > 1` is an
/// error, and otherwise there is no communicator.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_nccl_comm(
    _args: &cli::ServeArgs,
    _gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    world_size: usize,
    _max_batch_tokens: usize,
    _hidden_size: usize,
    _vocab_size: usize,
) -> Result<Option<std::sync::Arc<dyn metrale_comm::CommBackend>>> {
    if world_size > 1 {
        anyhow::bail!(
            "multi-rank NCCL is not available on Apple Silicon (metal feature); \
             single-device only"
        );
    }
    Ok(None)
}
