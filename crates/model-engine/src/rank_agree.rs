// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Startup check that every rank holds the same values for the env-read scalars
//! that shape the collective schedule.
//!
//! Such a scalar is read independently on each rank. For example,
//! [`metrale_model_arch::glm5next_layer::prefill_rows`] (`METRALE_GLM_PREFILL_ROWS`) sets the row
//! count of each prefill sub-chunk, and each sub-chunk issues its own `reduce_partial`, so a skew
//! between ranks is a hang or a reduce over the wrong extent rather than a perf difference.
//! Rank 0 broadcasts its values, every rank compares them with its own, and a mismatch fails
//! model construction with an error naming each disagreeing entry. The caller,
//! `TransformerModel::new`, chooses the list.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::GpuBackend;

/// 2026-09-25: Broadcast rank 0's `items` and bail if this rank's own values differ.
///
/// `items` is `(name, value)`; the name appears only in the error message. A single-rank run or
/// an empty list returns `Ok(())` without communicating.
///
/// This is a collective: every rank must call it with the same `items.len()` at the same point.
/// The one caller is `TransformerModel::new`, which every rank runs.
pub(crate) fn assert_ranks_agree(
    gpu: &dyn GpuBackend,
    comm: &dyn CommBackend,
    items: &[(&str, u64)],
) -> Result<()> {
    if comm.world_size() < 2 || items.is_empty() {
        return Ok(());
    }
    let bytes = items.len() * 8;
    let buf = gpu.alloc(bytes)?;

    let gathered = (|| -> Result<Vec<u64>> {
        let mut host: Vec<u8> = items.iter().flat_map(|(_, v)| v.to_le_bytes()).collect();
        gpu.copy_h2d(&host, buf)?;
        comm.broadcast(buf.0, bytes, 0)?;
        gpu.copy_d2h(buf, &mut host)?;
        Ok(host
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
            .collect())
    })();

    // 2026-09-25: The scratch is freed before `gathered?`, so a failed broadcast does not leak
    // it; a failed free is logged and does not fail the check.
    if let Err(e) = gpu.free(buf) {
        tracing::warn!("rank-agreement scratch free failed (non-fatal): {e}");
    }

    let root = gathered?;
    let rank = comm.rank();
    let bad: Vec<String> = items
        .iter()
        .zip(&root)
        .filter(|((_, mine), theirs)| mine != *theirs)
        .map(|((name, mine), theirs)| {
            format!("{name}: rank {rank} has {mine}, rank 0 has {theirs}")
        })
        .collect();
    if !bad.is_empty() {
        bail!(
            "ranks disagree on collective-shaping config — this is a hang or a wrong-extent \
             reduce, not a perf skew. Set the same value on EVERY rank: {}",
            bad.join("; ")
        );
    }
    tracing::info!(
        "rank-agreement OK on {} collective-shaping scalar(s): {}",
        items.len(),
        items
            .iter()
            .map(|(n, v)| format!("{n}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok(())
}

/// 2026-10-14 (race-memory, A183): The minimum of `mine` over every rank: each rank in turn
/// broadcasts its value. A single-rank run returns `mine` without communicating. A collective:
/// every rank calls it at the same point (the one caller is `TransformerModel::new`, for the KV
/// pool block count).
pub(crate) fn agree_min_u64(
    gpu: &dyn GpuBackend,
    comm: &dyn CommBackend,
    mine: u64,
) -> Result<u64> {
    if comm.world_size() < 2 {
        return Ok(mine);
    }
    let buf = gpu.alloc(8)?;
    let gathered = (|| -> Result<u64> {
        let mut min = mine;
        for root in 0..comm.world_size() {
            let mut host = mine.to_le_bytes();
            gpu.copy_h2d(&host, buf)?;
            comm.broadcast(buf.0, 8, root)?;
            gpu.copy_d2h(buf, &mut host)?;
            min = min.min(u64::from_le_bytes(host));
        }
        Ok(min)
    })();
    if let Err(e) = gpu.free(buf) {
        tracing::warn!("rank-agreement scratch free failed (non-fatal): {e}");
    }
    gathered
}

#[cfg(test)]
mod tests {
    //! 2026-09-25: The collective needs 2 ranks, so these tests cover only the comparison, on
    //! `mismatches`, a copy of the filter in `assert_ranks_agree`.

    fn mismatches(items: &[(&str, u64)], root: &[u64], rank: usize) -> Vec<String> {
        items
            .iter()
            .zip(root)
            .filter(|((_, mine), theirs)| mine != *theirs)
            .map(|((name, mine), theirs)| {
                format!("{name}: rank {rank} has {mine}, rank 0 has {theirs}")
            })
            .collect()
    }

    #[test]
    fn agreement_is_silent() {
        let items = [("METRALE_GLM_PREFILL_ROWS", 8u64), ("ep_protocol_v2", 0)];
        assert!(mismatches(&items, &[8, 0], 1).is_empty());
    }

    #[test]
    fn a_single_disagreement_names_the_lever_and_both_values() {
        let items = [("METRALE_GLM_PREFILL_ROWS", 4u64), ("ep_protocol_v2", 0)];
        let bad = mismatches(&items, &[8, 0], 1);
        assert_eq!(
            bad,
            vec!["METRALE_GLM_PREFILL_ROWS: rank 1 has 4, rank 0 has 8"]
        );
    }

    #[test]
    fn every_disagreement_is_reported_not_just_the_first() {
        let items = [("METRALE_GLM_PREFILL_ROWS", 4u64), ("ep_protocol_v2", 1)];
        assert_eq!(mismatches(&items, &[8, 0], 1).len(), 2);
    }
}
