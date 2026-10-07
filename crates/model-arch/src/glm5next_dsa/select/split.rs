// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: `METRALE_GLM_DSA_INDEX_SPLIT=1`: the two tensor-parallel ranks split the query
//! rows of one exact batched DSA selection (a prefill sub-chunk's `k` rows) instead of both
//! selecting all of them. Rank 0 selects rows `[0, ceil(k / 2))`, rank 1 the rest; each then
//! sends its rows of the `[k, out_width]` token output to the other, so both hold the full
//! output the replicated pass would have written.
//!
//! Why the bytes do not change: every selection kernel indexes its query row by block index
//! (`dsa_index_scores` grid `(P, Q)`, `dsa_index_scores_tiled` tiles of 32 rows whose outputs
//! never mix, `dsa_topk_pools` and `dsa_expand_selection` one block per row) and uses
//! `q_rows` only as a bound and as the stride of the per-pass temporaries (scores,
//! candidacy, selected pools), which no row reads across. The key side (pool compression)
//! is computed in full on both ranks from the same replicated indexer cache, and the
//! geometry (`S`, pool counts, `select_k`, `out_width`) depends on the context, not on
//! `q_rows`. Offsetting `q`, `weights`, `q_pos`, `q_mask` and `tokens` by the first owned
//! row therefore hands each row the inputs and output slot it had in the full pass. The
//! exchange is a byte copy. GPU gate: `examples/glm5next_dsa_index_split_2rank.rs`.
//!
//! 2026-10-04: The full-width staged prefill (`decode_k_wide`) splits each `core_rows`
//! sub-chunk's selection the same way under `METRALE_GLM_DSA_INDEX_SPLIT_WIDE=1`
//! (`row_split_for`). Only the selection kernels are split there: the query-side projections
//! (`wq_b`, `weights_proj`) run once over the whole window on cuBLASLt (the default,
//! `METRALE_GLM_CUBLAS_PROJ`), whose per-row bytes depend on M (it picks its algorithm per
//! shape, `gpu-runtime/src/cublaslt.rs`), so both ranks keep computing them for every row.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Both ranks call `select_tokens_split` for the same `k` at the same point, or neither
//!   does: the exchange is a grouped send/recv pair. The lever is in the startup
//!   rank-agreement check.

use anyhow::{Result, ensure};
use metrale_comm::CommBackend;

use super::*;

/// 2026-10-01: The rows of a `k`-row pass that `rank` (of two) selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowSplit {
    pub rank: usize,
    pub k: usize,
    /// 2026-10-01: First owned row.
    pub r0: usize,
    /// 2026-10-01: Owned row count, at least 1.
    pub rows: usize,
}

impl RowSplit {
    /// 2026-10-01: `None` below two rows (one rank would own nothing) or for a rank other
    /// than 0 or 1.
    pub fn new(k: usize, rank: usize) -> Option<Self> {
        let a = k.div_ceil(2);
        match rank {
            _ if k < 2 => None,
            0 => Some(Self {
                rank,
                k,
                r0: 0,
                rows: a,
            }),
            1 => Some(Self {
                rank,
                k,
                r0: a,
                rows: k - a,
            }),
            _ => None,
        }
    }

    /// 2026-10-01: The other rank's rows of the same pass.
    pub fn peer(&self) -> Self {
        let a = self.k.div_ceil(2);
        let rank = 1 - self.rank;
        let (r0, rows) = if rank == 0 { (0, a) } else { (a, self.k - a) };
        Self {
            rank,
            k: self.k,
            r0,
            rows,
        }
    }

    /// 2026-10-01: `inputs` with every per-query array starting at the first owned row.
    pub fn inputs(&self, cfg: &Glm5NextDsaConfig, inputs: &DsaSelectInputs) -> DsaSelectInputs {
        let heads = cfg.index_heads;
        DsaSelectInputs {
            q: inputs.q.offset(self.r0 * heads * cfg.index_head_dim * 4),
            weights: inputs.weights.offset(self.r0 * heads * 4),
            q_pos: inputs.q_pos.offset(self.r0 * 4),
            q_mask: inputs.q_mask.offset(self.r0),
            ..*inputs
        }
    }
}

/// 2026-10-04: This rank's split of an exact `k`-row pass, with the communicator to swap
/// over, when `on` and `comm` has two ranks; `None` otherwise and below two rows. Both ranks
/// get `Some` for the same `k` or neither does (`on` is rank-agreed at startup).
pub fn row_split_for(
    comm: Option<&dyn CommBackend>,
    on: bool,
    k: usize,
) -> Option<(RowSplit, &dyn CommBackend)> {
    let comm = comm.filter(|c| on && c.world_size() == 2)?;
    RowSplit::new(k, comm.rank()).map(|s| (s, comm))
}

/// 2026-10-01: Swap the two ranks' rows of a row-major buffer in place: send this rank's
/// `split.rows` rows of `row_bytes` at `base`, receive the peer's into their own rows. One
/// NCCL group on `stream`; the group is closed even when posting fails.
pub fn exchange_rows(
    comm: &dyn CommBackend,
    base: DevicePtr,
    row_bytes: usize,
    split: RowSplit,
    stream: u64,
) -> Result<()> {
    let peer = split.peer();
    let (mine, theirs) = (base.offset(split.r0 * row_bytes), base.offset(peer.r0 * row_bytes));
    comm.group_start()?;
    let posted = comm
        .send_to(mine.0, split.rows * row_bytes, peer.rank, stream)
        .and_then(|()| comm.recv_from(theirs.0, peer.rows * row_bytes, peer.rank, stream));
    let closed = comm.group_end();
    posted.and(closed)
}

/// 2026-10-01: `select_tokens` for this rank's rows of the exact `geom.q_rows`-row pass,
/// then the token exchange, leaving all `[q_rows, out_width]` rows in `scratch.tokens()`
/// on both ranks, byte for byte what the full pass writes (module doc).
#[allow(clippy::too_many_arguments)]
pub fn select_tokens_split(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    split: RowSplit,
    comm: &dyn CommBackend,
    stream: u64,
) -> Result<()> {
    ensure!(
        geom.q_rows == split.k && inputs.geom_dev.0 == 0,
        "DSA index split: a {}-row split of a {}-row pass (device geometry {}); the split \
         takes exact host-geometry passes only",
        split.k,
        geom.q_rows,
        inputs.geom_dev.0 != 0
    );
    let mine = DsaSelectGeometry {
        q_rows: split.rows,
        ..*geom
    };
    let inp = split.inputs(cfg, inputs);
    let out = scratch.row(split.r0, cfg);
    select_tokens(gpu, kernels, cfg, &mine, &inp, &out, DsaSelectLaunch::Exact, stream)?;
    // 2026-10-01: Under `METRALE_GLM_PREFILL_COMM_OVERLAP` a deferred all-reduce may still be
    // in flight on the comm stream. Waiting for both slots first keeps this communicator to
    // one operation in flight at a time, in issue order, on both ranks. A slot with nothing
    // outstanding (or never recorded) costs nothing.
    for slot in 0..metrale_comm::ALL_REDUCE_DEFERRED_SLOTS {
        comm.all_reduce_join(stream, slot)?;
    }
    exchange_rows(comm, scratch.tokens(), cfg.out_width() * 4, split, stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_cover_every_row_once() {
        for k in 2..=513 {
            let (a, b) = (RowSplit::new(k, 0).unwrap(), RowSplit::new(k, 1).unwrap());
            assert_eq!(a.r0, 0);
            assert_eq!(a.r0 + a.rows, b.r0);
            assert_eq!(b.r0 + b.rows, k);
            assert!(a.rows >= b.rows && b.rows >= 1 && a.rows - b.rows <= 1);
            assert_eq!(a.peer(), b);
            assert_eq!(b.peer(), a);
        }
    }

    #[test]
    fn refuses_one_row_and_other_ranks() {
        assert_eq!(RowSplit::new(1, 0), None);
        assert_eq!(RowSplit::new(0, 1), None);
        assert_eq!(RowSplit::new(8, 2), None);
    }

    #[test]
    fn inputs_offset_only_the_query_side() {
        let cfg = Glm5NextDsaConfig {
            hidden: 4096,
            index_heads: 32,
            index_head_dim: 128,
            index_kpool: 4,
            index_topk: 2048,
            always_select_tail: true,
            local_heads: 32,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 256,
            qk_rope_head_dim: 0,
            v_head_dim: 256,
            max_context: 8192,
        };
        let full = DsaSelectInputs {
            k_normed: DevicePtr(1 << 40),
            gate: DevicePtr(2 << 40),
            valid: DevicePtr(3 << 40),
            ape: DevicePtr(4 << 40),
            q: DevicePtr(5 << 40),
            weights: DevicePtr(6 << 40),
            q_pos: DevicePtr(7 << 40),
            q_mask: DevicePtr(8 << 40),
            first_key: 0,
            geom_dev: DevicePtr::NULL,
            pool_cache: None,
        };
        let s = RowSplit::new(256, 1).unwrap();
        let i = s.inputs(&cfg, &full);
        assert_eq!(i.q.0, (5 << 40) + 128 * 32 * 128 * 4);
        assert_eq!(i.weights.0, (6 << 40) + 128 * 32 * 4);
        assert_eq!(i.q_pos.0, (7 << 40) + 128 * 4);
        assert_eq!(i.q_mask.0, (8 << 40) + 128);
        let keys = |x: &DsaSelectInputs| (x.k_normed, x.gate, x.valid, x.ape);
        assert_eq!(keys(&i), keys(&full));
    }
}
