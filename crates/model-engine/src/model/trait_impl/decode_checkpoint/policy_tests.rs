// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Tests for the Marconi snapshot policy (`METRALE_MARCONI_PREFILL_ONLY`): the
//! value parse, that the default reproduces the pre-lever decisions, that prefill-only sends
//! no decode checkpoint and saves no finish leaf, and a multi-turn chat on a real
//! `RadixTree` with a 16-slot pool.
//!
//! Owner: model-engine prefix cache.
//! Invariants: none beyond the types.

use super::*;
use crate::model::trait_impl::decode_checkpoint::{CkptInputs, decode_ckpt_plan};
use crate::prefill_plan::{is_checkpoint_chunk_end, plan_chunk_len, tail_split_point};
use metrale_cache::radix_tree::RadixTree;
use metrale_telemetry::prefix_cache::PrefixCache;

/// 2026-10-01: KV block size of the GLM-5.3 serve.
const BS: usize = 16;

#[test]
fn only_an_explicit_one_selects_prefill_only() {
    assert_eq!(SnapshotPolicy::from_env_value(None), SnapshotPolicy::All);
    for other in ["0", "", "true", "on", "yes", "2", " 1", "1 "] {
        assert_eq!(
            SnapshotPolicy::from_env_value(Some(other)),
            SnapshotPolicy::All,
            "{other:?} must keep every snapshot family"
        );
    }
    assert_eq!(SnapshotPolicy::from_env_value(Some("1")), SnapshotPolicy::PrefillOnly);
}

#[test]
fn the_default_policy_reproduces_the_pre_lever_decisions() {
    // 2026-10-01: Before the lever, the decode path computed
    // `ssm_snapshots.is_enabled() && prefix_cache.is_active()` and always saved the finish leaf.
    for snapshots in [false, true] {
        for cache in [false, true] {
            assert_eq!(
                decode_ckpt_enabled(snapshots, cache, SnapshotPolicy::All),
                snapshots && cache
            );
            assert!(!decode_ckpt_enabled(snapshots, cache, SnapshotPolicy::PrefillOnly));
        }
    }
    assert!(SnapshotPolicy::All.saves_decode_checkpoints());
    assert!(SnapshotPolicy::All.saves_finish_leaf());
    assert!(!SnapshotPolicy::PrefillOnly.saves_decode_checkpoints());
    assert!(!SnapshotPolicy::PrefillOnly.saves_finish_leaf());
}

/// 2026-10-01: The decode-checkpoint inputs at `tokens_len`, with the default interval of 4
/// blocks and the enable bit `decode_ckpt_enabled` gives `policy`.
fn ckpt_inputs(policy: SnapshotPolicy, tokens_len: usize, last_ckpt_block: usize) -> CkptInputs {
    CkptInputs {
        enabled: decode_ckpt_enabled(true, true, policy),
        num_ssm_layers: 34,
        hss_window_start: 0,
        slot_idx: 0,
        tokens_len,
        block_size: BS,
        block_table_len: tokens_len.div_ceil(BS),
        last_ckpt_block,
        interval: 4,
    }
}

#[test]
fn prefill_only_never_fires_a_decode_checkpoint_or_its_ep_command() {
    // 2026-10-01: `decode_ckpt_plan` returning `None` means no save on rank 0 and no
    // `EP_CMD_DECODE_CKPT` on the wire. Sweep every decode length across 40 blocks.
    let prompt = 30_007;
    let mut fired_default = 0;
    let mut last = 0;
    for len in prompt..prompt + 40 * BS {
        if let Some(plan) = decode_ckpt_plan(&ckpt_inputs(SnapshotPolicy::All, len, last)) {
            fired_default += 1;
            last = plan.end_block;
        }
        assert_eq!(
            decode_ckpt_plan(&ckpt_inputs(SnapshotPolicy::PrefillOnly, len, 0)),
            None,
            "prefill-only fired a decode checkpoint at {len} tokens"
        );
    }
    assert_eq!(fired_default, 10, "the default fires every 4 blocks");
}

/// 2026-10-01: A Marconi slot pool of `n` slots. `acquire` mirrors `SsmSnapshotPool::save`
/// failing on a full pool, then `reclaim_from_cache` (no spill tier) freeing the
/// `evict_snapshot_lru` victim, then the retried save taking it.
struct Pool {
    free: Vec<usize>,
}

impl Pool {
    fn new(n: usize) -> Self {
        Self {
            free: (0..n).rev().collect(),
        }
    }

    fn acquire(&mut self, tree: &RadixTree) -> usize {
        if let Some(slot) = self.free.pop() {
            return slot;
        }
        tree.evict_snapshot_lru()
            .expect("a full pool holds a registered snapshot to evict")
    }

    fn give_back(&mut self, displaced: Option<usize>) {
        if let Some(slot) = displaced {
            self.free.push(slot);
        }
    }
}

fn blocks(n_tokens: usize) -> Vec<u32> {
    (0..n_tokens.div_ceil(BS) as u32).collect()
}

/// 2026-10-01: The chunk ends at which a prefill of `prompt` saves an intermediate checkpoint
/// (`prefill_b_save_checkpoint`), for the defaults `--max-prefill-tokens 8192` and
/// `--ssm-checkpoint-interval 256`. Chunks follow `plan_chunk_len`, and a last chunk that
/// spans the tail split point is split there (`prefill_chunk_dispatch`). A warm prefill that
/// restored at `from` is modelled as saving only the ends past `from`.
fn prefill_checkpoint_ends(prompt: usize, from: usize) -> Vec<usize> {
    let cut = tail_split_point(prompt, BS);
    let mut ends = Vec::new();
    let mut offset = 0;
    while offset < prompt {
        let len = plan_chunk_len(offset, prompt, (prompt - offset).min(8192), Some(BS), cut);
        let mut end = offset + len;
        if end >= prompt {
            match cut {
                Some(c) if c > offset => end = c,
                _ => break,
            }
        }
        if end > from && is_checkpoint_chunk_end(end, prompt, BS, 256) {
            ends.push(end);
        }
        offset = end;
    }
    ends
}

/// 2026-10-01: One turn on one rank: the prefill checkpoints and the prompt insert, a decode
/// checkpoint whenever `decode_ckpt_plan` fires, then `cache_sequence` (the finish leaf when
/// `policy` saves it, else a plain insert). Session 0 and adapter 0 throughout.
fn run_turn(
    tree: &RadixTree,
    pool: &mut Pool,
    policy: SnapshotPolicy,
    prompt: &[u32],
    generated: &[u32],
    restored_at: usize,
) {
    for end in prefill_checkpoint_ends(prompt.len(), restored_at) {
        let slot = pool.acquire(tree);
        tree.insert(&prompt[..end], &blocks(end), &[], BS, end, 0);
        let displaced =
            tree.insert_intermediate_snapshot(&prompt[..end], &[], &[], BS, slot, 0, end, 0);
        pool.give_back(displaced);
    }
    let p = prompt.len();
    tree.insert(prompt, &blocks(p), &[], BS, restored_at, 0);
    let seq: Vec<u32> = prompt.iter().chain(generated).copied().collect();
    let mut last_ckpt_block = 0;
    for len in p + 1..=seq.len() {
        if let Some(plan) = decode_ckpt_plan(&ckpt_inputs(policy, len, last_ckpt_block)) {
            let slot = pool.acquire(tree);
            let n = plan.snap_tokens;
            let displaced =
                tree.insert_intermediate_snapshot(&seq[..n], &[], &[], BS, slot, 0, n, 0);
            pool.give_back(displaced);
            last_ckpt_block = plan.end_block;
        }
    }
    if policy.saves_finish_leaf() {
        let slot = pool.acquire(tree);
        let (displaced, _) =
            tree.insert_with_snapshot(&seq, &blocks(seq.len()), &[], BS, slot, 0, p, 0);
        pool.give_back(displaced);
    } else {
        tree.insert(&seq, &blocks(seq.len()), &[], BS, p, 0);
    }
}

/// 2026-10-01: Four turns of a ~30K-token chat whose template drops each answer's reasoning:
/// the next prompt is the previous prompt, then a re-rendered answer (tokens other than the
/// generated ones), then a new user message. For each turn after the first, returns the depth
/// of the snapshot the lookup would restore (`None`: recompute everything) and the tail split
/// point of the previous prompt.
fn chat(policy: SnapshotPolicy) -> Vec<(Option<usize>, usize)> {
    let tree = RadixTree::new();
    let mut pool = Pool::new(16);
    let mut prompt: Vec<u32> = (0..30_007).collect();
    let mut prev_cut = 0;
    let mut out = Vec::new();
    for turn in 0..4u32 {
        let mut restored_at = 0;
        if turn > 0 {
            let m = tree.lookup(&prompt, BS, 0, 0);
            let depth = m.ssm_snapshot.map(|_| m.ssm_snapshot_tokens);
            restored_at = depth.unwrap_or(0);
            out.push((depth, prev_cut));
        }
        // 2026-10-01: 1,200 generated tokens (reasoning and answer); 400 re-rendered answer
        // tokens and 150 user tokens extend the prompt.
        let generated: Vec<u32> = (0..1_200).map(|i| 1_000_000 + turn * 10_000 + i).collect();
        run_turn(&tree, &mut pool, policy, &prompt, &generated, restored_at);
        prev_cut = tail_split_point(prompt.len(), BS).expect("a 30K prompt has a tail split");
        prompt.extend((0..400).map(|i| 2_000_000 + turn * 10_000 + i));
        prompt.extend((0..150).map(|i| 3_000_000 + turn * 10_000 + i));
    }
    out
}

#[test]
fn prefill_only_keeps_the_prompt_tail_checkpoint_the_next_turn_restores() {
    // 2026-10-01: Each warm turn restores at the previous prompt's tail split point and
    // replays only the re-rendered answer, the new message and at most two blocks.
    let turns = chat(SnapshotPolicy::PrefillOnly);
    assert_eq!(turns.len(), 3);
    for (i, (depth, prev_cut)) in turns.into_iter().enumerate() {
        assert_eq!(
            depth,
            Some(prev_cut),
            "turn {}: expected a restore at the previous prompt's tail split",
            i + 2
        );
    }
}

#[test]
fn the_default_loses_that_checkpoint_to_decode_checkpoint_churn() {
    // 2026-10-01: Characterises the default on the same chat: 4 prefill checkpoints, then 19
    // decode checkpoints and a finish leaf into 16 slots. Recency eviction drops the prefill
    // checkpoints first, and no decode-time snapshot lies on the next prompt, so every warm
    // turn recomputes from token 0. A change to the default should change this test.
    let turns = chat(SnapshotPolicy::All);
    assert_eq!(turns.len(), 3);
    for (i, (depth, _)) in turns.into_iter().enumerate() {
        assert_eq!(depth, None, "turn {}", i + 2);
    }
}
