// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU reserve terms for SSM (linear-attention) state: verify-pool
//! slot counts and capacities, pool bytes, Marconi snapshot slots, and the
//! decode-rollback ring depth (`decode_ring`).
//!
//! Owner: model-layers (SSM reserve).
//! Invariants: none beyond the types.
//!
//! The server's `preflight_reserve`, which runs before the weights load, and the
//! allocating code (`SsmStatePool::new`, `TransformerModel::new`) call the same
//! functions here. If the two computed different sizes, a serve would fail a
//! CUDA allocation after load, or refuse a configuration it could run.

mod decode_ring;
pub use decode_ring::{
    DECODE_RING_FIT_LADDER, DecodeRingDecision, decode_rollback_ring_slots,
    decode_rollback_ring_slots_with, fit_decode_ring_slots, parse_decode_ring_slots,
    published_decode_ring_slots, set_decode_ring_slots, watchdogs_disabled_from_value,
};

mod rollback;
pub use rollback::{
    SsmRollbackMode, set_ssm_rollback_mode, ssm_replay_ring_bytes, ssm_replay_row_bytes,
    ssm_rollback_mode,
};

/// 2026-09-25: Number of SSM-pool slots the verify pools (per-token intermediates and
/// the pre-verify checkpoint) cover.
///
/// Three callers use it: the server's `preflight_reserve` sizes the pre-load
/// reserve with it, `SsmStatePool::new` allocates that many verify slots plus a
/// dummy, and for a model with SSM layers the scheduler takes the plain-decode
/// branch while any active slot is at or above it (`spec_slot_cap`,
/// `lane_decode.rs`). Retirement compacts survivors onto low slots, except
/// under EP protocol v2 (`retire_finished_sequences`).
///
/// It is `max_batch_size` when that is at most 32 (`layer::VERIFY_WY_TABLE_SEQS`,
/// the floor) or when [`mtp_pool_full_width`] holds; otherwise
/// `max(mtp_max_seqs(), 32)`, capped at `max_batch_size`. The scheduler never
/// speculates with more than `mtp_max_seqs()` active sequences.
pub fn mtp_state_slots(max_batch_size: usize) -> usize {
    if mtp_pool_cap() {
        return mtp_state_slots_capped(max_batch_size, crate::speculative::mtp_max_seqs());
    }
    mtp_state_slots_with(
        max_batch_size,
        crate::speculative::mtp_max_seqs(),
        mtp_pool_full_width(),
    )
}

/// 2026-09-25: True when `METRALE_MTP_POOL_FULL_WIDTH` is present (any value, `0`
/// included) or `METRALE_EP_PROTOCOL` is `v2`. Then [`mtp_state_slots`] is
/// `max_batch_size` and [`verify_slot_drafts`] is `num_drafts` for every slot.
pub fn mtp_pool_full_width() -> bool {
    std::env::var_os("METRALE_MTP_POOL_FULL_WIDTH").is_some()
        || matches!(std::env::var("METRALE_EP_PROTOCOL").as_deref(), Ok("v2"))
}

/// 2026-10-10: `METRALE_MTP_POOL_CAP=1` (exactly `1`) and `METRALE_MTP_POOL_FULL_WIDTH`
/// absent. Read once per process. Then [`mtp_state_slots`] is
/// [`mtp_state_slots_capped`]: `min(max_batch_size, mtp_max_seqs())` with no 32 floor,
/// also under EP protocol v2. The scheduler never speculates with more than
/// `mtp_max_seqs()` active sequences, so wider pools are never used at once.
/// [`verify_slot_drafts`] is unchanged: under v2 every covered slot keeps `num_drafts`.
///
/// v2 never compacts survivors onto low slots (`retire_finished_sequences`), so with the
/// cap `SsmStatePool` claims the lowest free slot on both ranks (the worker mirrors the
/// head's claims, `ep_worker_step`). A sequence on a slot at or above the cap can only
/// have been claimed while at least `mtp_max_seqs()` other slots were held. Until it
/// finishes, a batch holding it plain-decodes; the scheduler counts those steps
/// ([`note_mtp_uncovered_step`]). Exact: pool layout and slot choice only.
pub fn mtp_pool_cap() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("METRALE_MTP_POOL_CAP").as_deref() == Ok("1")
            && std::env::var_os("METRALE_MTP_POOL_FULL_WIDTH").is_none()
    })
}

/// 2026-10-10: Env-free core of [`mtp_state_slots`] under [`mtp_pool_cap`]:
/// `min(max_batch_size, spec_dispatch_cap)`, at least 1.
pub fn mtp_state_slots_capped(max_batch_size: usize, spec_dispatch_cap: usize) -> usize {
    max_batch_size.min(spec_dispatch_cap.max(1))
}

/// 2026-10-10: Decode steps on which MTP would have run but an active sequence sat on a
/// slot at or above the verify-pool cover ([`mtp_pool_cap`]); reported at shutdown.
static MTP_UNCOVERED_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 2026-10-10: Count one MTP step lost to an uncovered slot (`lane_decode.rs`) and return
/// the new total.
pub fn note_mtp_uncovered_step() -> u64 {
    MTP_UNCOVERED_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
}

/// 2026-10-10: MTP steps lost to uncovered slots so far in this process.
pub fn mtp_uncovered_steps() -> u64 {
    MTP_UNCOVERED_STEPS.load(std::sync::atomic::Ordering::Relaxed)
}

/// 2026-09-25: Env-free core of [`mtp_state_slots`]. The `VERIFY_WY_TABLE_SEQS` floor
/// (32) covers every slot when `max_batch_size` is at most 32, also when
/// `METRALE_NO_MTP_K_LADDER` lowers the dispatch cap to 4.
pub fn mtp_state_slots_with(
    max_batch_size: usize,
    spec_dispatch_cap: usize,
    full_width: bool,
) -> usize {
    if full_width {
        return max_batch_size;
    }
    max_batch_size.min(spec_dispatch_cap.max(crate::layer::VERIFY_WY_TABLE_SEQS))
}

/// 2026-09-25: Verify draft capacity of pool slot `slot_idx`: the largest count
/// `drafts_at(n)` gives for a width `n` in `slot_idx + 1..=max(dispatch_cap,
/// slot_idx + 1)`, clamped to `1..=num_drafts`; 0 when `num_drafts` is 0.
///
/// Retirement compacts active sequences onto slots `0..n`
/// (`retire_finished_sequences`), so a sequence on slot `slot_idx` usually runs
/// with at least `slot_idx + 1` active, hence the lower bound. When it does not,
/// the scheduler clamps the step's draft count to the smallest capacity among
/// the active slots (`spec_capacity::clamp_drafts_to_slot_capacity`).
///
/// With the default ladder (`4:3,8:3,16:1,32:1`), `--num-drafts 3` and a
/// dispatch cap of 32, slots 0..8 get 3 drafts and slots from 8 on get 1. So
/// `adaptive_rung`'s two-draft lift at widths 9..=16 is clamped back to one
/// draft at every such width: nine or more active sequences hold as many
/// distinct slots, one of them at index 8 or above. [`mtp_pool_full_width`]
/// gives every slot `num_drafts`, which keeps the lift.
pub fn verify_slot_drafts_with(
    slot_idx: usize,
    dispatch_cap: usize,
    num_drafts: usize,
    drafts_at: impl Fn(usize) -> usize,
) -> usize {
    if num_drafts == 0 {
        return 0;
    }
    let hi = dispatch_cap.max(slot_idx + 1);
    ((slot_idx + 1)..=hi)
        .map(&drafts_at)
        .max()
        .unwrap_or(num_drafts)
        .clamp(1, num_drafts)
}

/// 2026-09-25: [`verify_slot_drafts_with`] with the dispatch cap `mtp_max_seqs()` and
/// the ladder `mtp_ladder_drafts`. Every slot gets `num_drafts` when
/// [`mtp_pool_full_width`] holds or the ladder is disabled.
pub fn verify_slot_drafts(slot_idx: usize, num_drafts: usize) -> usize {
    if mtp_pool_full_width() {
        return num_drafts;
    }
    verify_slot_drafts_with(
        slot_idx,
        crate::speculative::mtp_max_seqs(),
        num_drafts,
        |n| crate::speculative::mtp_ladder_drafts(n, num_drafts),
    )
}

/// 2026-09-25: H-state intermediates the verify pools hold for pool slot `slot_idx`:
/// the slot's draft capacity, which is K-1 for a K-row verify. With
/// `uniform_verify` (set by the server's preflight for DFlash) every slot gets
/// `num_drafts`.
///
/// K-1 suffices because the state after the last verified row stays in
/// `h_state`: `commit_accepted_prefix` returns early on a full accept and
/// otherwise reads index `num_accepted - 1 <= K - 2` (`async_chkpt.rs`), and
/// the batched WY path requires only k-1 h intermediates
/// (`trait_decode_batched_conv_gdn_multi.rs`).
///
/// Only the H side is tiered. The conv intermediates stay at `num_drafts + 1`
/// per slot: `gdn_verify_fused_conv_kn_batched` writes all K, and the batched
/// path declines unless every sequence holds k conv intermediates and the
/// per-sequence regions are evenly spaced (`trait_decode_batched_conv_gdn_multi.rs`).
pub fn verify_slot_h_intermediates(
    slot_idx: usize,
    num_drafts: usize,
    uniform_verify: bool,
) -> usize {
    if uniform_verify {
        return num_drafts;
    }
    verify_slot_drafts(slot_idx, num_drafts)
}

/// 2026-09-25: Bytes of one stored h-state blob: half of `h_f32_bytes` when `f16_pool`
/// (FP16 elements), else `h_f32_bytes`. `SsmStatePool::new`, the preflight
/// reserve ([`ssm_pool_reserve_bytes`]) and the SSM layer's slot stride
/// (`ssm_h_fp16.rs`) all take the stored width from here.
///
/// `f16_pool` is `ssm_h_f16_pool_enabled()` (`--ssm-h-dtype f16-pool`) at the
/// production call sites; it is a parameter so pool geometry is testable
/// without the process-global flag.
///
/// Panics when `h_f32_bytes` is not a multiple of 4.
pub fn ssm_h_stored_bytes(h_f32_bytes: usize, f16_pool: bool) -> usize {
    assert!(
        h_f32_bytes.is_multiple_of(4),
        "h-state blobs are FP32-element sized"
    );
    if f16_pool {
        h_f32_bytes / 2
    } else {
        h_f32_bytes
    }
}

/// 2026-09-25: Bytes of the FP32 h-state prefill staging arena: `slots *
/// h_layer_f32_bytes` when `f16_pool`, else 0.
///
/// On an f16-sized pool the layer widens a slot's h-state into its FP32
/// staging blob before the GDN prefill kernels run and narrows it back after
/// (`ssm_h_fp16::prefill_h_begin` / `prefill_h_end`). There is one blob per
/// slot, shared by all layers, so `h_layer_f32_bytes` is one layer's FP32 h
/// blob, not the across-layers total the other reserve terms use.
///
/// `SsmStatePool::new` passes its slot count including the dummy slot
/// (`max_slots + 1`); the server's preflight passes `max_batch_size`.
pub fn ssm_h_prefill_stage_bytes(slots: usize, h_layer_f32_bytes: usize, f16_pool: bool) -> usize {
    if f16_pool {
        slots * h_layer_f32_bytes
    } else {
        0
    }
}

/// 2026-10-10: `METRALE_SSM_NO_CHECKPOINT=1` (exactly `1`). Read once per process.
///
/// When on, `SsmStatePool::new` allocates no per-slot pre-verify checkpoint pools
/// (`h_checkpoint_pools` / `conv_checkpoint_pools`) and the pre-load preflight reserve
/// ([`ssm_pool_reserve_bytes`]) drops the same share, so the KV budget receives it.
/// The checkpoint is read only by a rollback to `num_accepted == 0`, which the MTP path
/// never issues (ngram and self-speculative do; `validate_serve_args` refuses them with
/// the lever on). Exact: no numeric change. Both the reserve and the pool read this one
/// function so they agree; the pure helpers below take the value as a `bool`.
pub fn ssm_no_checkpoint() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_SSM_NO_CHECKPOINT").as_deref() == Ok("1"))
}

/// 2026-10-10: Process-wide count of checkpoint reads/writes refused under
/// `METRALE_SSM_NO_CHECKPOINT=1` (a num_accepted == 0 rollback, a lazy checkpoint). The
/// gate expects 0 on every MTP serve.
static CHECKPOINT_READS_REFUSED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// 2026-10-10: Count one refused checkpoint use and return the new total. Call before
/// bailing; put the total in the error text.
pub fn note_ssm_checkpoint_read_refused() -> u64 {
    CHECKPOINT_READS_REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
}

/// 2026-10-10: Refused checkpoint uses so far in this process.
pub fn ssm_checkpoint_reads_refused() -> u64 {
    CHECKPOINT_READS_REFUSED.load(std::sync::atomic::Ordering::Relaxed)
}

/// 2026-10-10: Bytes of the per-verify-slot pre-verify checkpoint (h and conv, all SSM
/// layers) the pool allocates for `slots` slots: `slots * blob`, or 0 when
/// `no_checkpoint` ([`ssm_no_checkpoint`]). `SsmStatePool::new` and
/// [`ssm_pool_reserve_bytes_with`] both size the checkpoint through this.
pub fn ssm_checkpoint_bytes(slots: usize, blob_bytes: usize, no_checkpoint: bool) -> usize {
    if no_checkpoint { 0 } else { slots * blob_bytes }
}

/// 2026-09-25: SSM state-pool bytes for the pre-load preflight reserve, following the
/// layout `SsmStatePool::new` allocates, without the pools' dummy slots:
///
/// * base: `max_batch_size` blobs (h and conv across all SSM layers);
/// * with `spec_on`, per verify slot (`mtp_state_slots` of them): under
///   `SsmRollbackMode::Snapshot`, [`verify_slot_h_intermediates`] h blobs,
///   `num_drafts + 1` conv blobs and one pre-verify checkpoint blob (h and
///   conv); under `SsmRollbackMode::Replay`, the checkpoint blob only.
///
/// `h_blob_bytes` and `conv_blob_bytes` are per-sequence totals across all SSM
/// layers at FP32 width; `h_f16_pool` narrows the h terms through
/// [`ssm_h_stored_bytes`].
///
/// 2026-10-10: The checkpoint blob is left out when `METRALE_SSM_NO_CHECKPOINT=1`
/// ([`ssm_no_checkpoint`]); see [`ssm_pool_reserve_bytes_with`].
pub fn ssm_pool_reserve_bytes(
    max_batch_size: usize,
    h_blob_bytes: usize,
    conv_blob_bytes: usize,
    spec_on: bool,
    num_drafts: usize,
    mtp_state_slots: usize,
    uniform_verify: bool,
    h_f16_pool: bool,
    rollback: SsmRollbackMode,
) -> usize {
    ssm_pool_reserve_bytes_with(
        max_batch_size,
        h_blob_bytes,
        conv_blob_bytes,
        spec_on,
        num_drafts,
        mtp_state_slots,
        uniform_verify,
        h_f16_pool,
        rollback,
        ssm_no_checkpoint(),
    )
}

/// 2026-10-10: Env-free core of [`ssm_pool_reserve_bytes`]. `no_checkpoint` drops the
/// pre-verify checkpoint blob from every verify slot in both rollback modes (Replay's
/// whole per-slot term was that blob, so it drops to 0).
pub fn ssm_pool_reserve_bytes_with(
    max_batch_size: usize,
    h_blob_bytes: usize,
    conv_blob_bytes: usize,
    spec_on: bool,
    num_drafts: usize,
    mtp_state_slots: usize,
    uniform_verify: bool,
    h_f16_pool: bool,
    rollback: SsmRollbackMode,
    no_checkpoint: bool,
) -> usize {
    let h_blob_bytes = ssm_h_stored_bytes(h_blob_bytes, h_f16_pool);
    let blob = h_blob_bytes + conv_blob_bytes;
    let base = max_batch_size * blob;
    if !spec_on {
        return base;
    }
    let ckpt = ssm_checkpoint_bytes(1, blob, no_checkpoint);
    let verify: usize = (0..mtp_state_slots)
        .map(|slot| match rollback {
            SsmRollbackMode::Snapshot => {
                verify_slot_h_intermediates(slot, num_drafts, uniform_verify) * h_blob_bytes
                    + (num_drafts + 1) * conv_blob_bytes
                    + ckpt
            }
            // 2026-09-25: Replay's verify-window input ring is a separate
            // term (`ssm_replay_ring_bytes`), sized by activation rows.
            SsmRollbackMode::Replay => ckpt,
        })
        .sum();
    base + verify
}

/// 2026-10-10: The SSM-pool slot whose verify pools a hybrid serve (`METRALE_HYBRID_DFLASH=1`)
/// sizes for the DFlash K=gamma verify. Only a sequence on this slot is DFlash-eligible; under
/// the hybrid lever slots are claimed lowest index first, so a request that starts alone gets it.
pub const HYBRID_DFLASH_SLOT: usize = 0;

/// 2026-10-10: Hybrid serve: H intermediates of pool slot `slot_idx`. Every slot keeps its MTP
/// capacity ([`verify_slot_drafts`] of `mtp_drafts`); [`HYBRID_DFLASH_SLOT`] alone is widened to
/// `dflash_drafts` (gamma), so the K=gamma verify fits it and nowhere else.
pub fn hybrid_slot_h_intermediates(
    slot_idx: usize,
    mtp_drafts: usize,
    dflash_drafts: usize,
) -> usize {
    hybrid_slot_h_intermediates_with(
        slot_idx,
        verify_slot_drafts(slot_idx, mtp_drafts),
        dflash_drafts,
    )
}

/// 2026-10-10: Env-free core of [`hybrid_slot_h_intermediates`], given the slot's MTP capacity.
pub fn hybrid_slot_h_intermediates_with(
    slot_idx: usize,
    mtp_capacity: usize,
    dflash_drafts: usize,
) -> usize {
    if slot_idx == HYBRID_DFLASH_SLOT {
        mtp_capacity.max(dflash_drafts)
    } else {
        mtp_capacity
    }
}

/// 2026-10-10: [`ssm_pool_reserve_bytes_with`] for a hybrid serve (snapshot rollback only): per
/// verify slot, [`hybrid_slot_h_intermediates_with`] h blobs, `max(mtp_drafts, dflash_drafts) + 1`
/// conv blobs (the conv intermediates do not tier) and the checkpoint blob.
#[allow(clippy::too_many_arguments)]
pub fn ssm_pool_reserve_bytes_hybrid_with(
    max_batch_size: usize,
    h_blob_bytes: usize,
    conv_blob_bytes: usize,
    mtp_drafts: usize,
    dflash_drafts: usize,
    mtp_state_slots: usize,
    h_f16_pool: bool,
    no_checkpoint: bool,
    slot_capacity: impl Fn(usize) -> usize,
) -> usize {
    let h_blob_bytes = ssm_h_stored_bytes(h_blob_bytes, h_f16_pool);
    let blob = h_blob_bytes + conv_blob_bytes;
    let base = max_batch_size * blob;
    let ckpt = ssm_checkpoint_bytes(1, blob, no_checkpoint);
    let conv_k = mtp_drafts.max(dflash_drafts) + 1;
    let verify: usize = (0..mtp_state_slots)
        .map(|slot| {
            hybrid_slot_h_intermediates_with(slot, slot_capacity(slot), dflash_drafts)
                * h_blob_bytes
                + conv_k * conv_blob_bytes
                + ckpt
        })
        .sum();
    base + verify
}

/// 2026-09-25: Outcome of the Marconi snapshot-slot decision. `skip_reason` is `Some`
/// only when the slots are dropped because prefix caching is inactive, so the
/// allocating call site can log the saving.
pub struct MarconiSlotDecision {
    pub slots: usize,
    pub skip_reason: Option<&'static str>,
}

/// 2026-09-25: Number of Marconi SSM-snapshot slots to reserve and allocate:
/// `requested` (`--ssm-cache-slots`, default 16), or 0 when prefix caching is
/// inactive.
///
/// The server's `preflight_reserve` and `TransformerModel::new` (`impl_a1.rs`)
/// both call this. A slot is restored from a prefix-cache match
/// (`PrefixMatch::ssm_snapshot`); with prefix caching inactive the server
/// installs `NoPrefixCaching` (`build_prefix_cache`), whose lookup returns an
/// empty match.
///
/// `METRALE_SSM_MARCONI_FULL`, present with any value (`0` included), keeps
/// `requested` regardless.
pub fn marconi_snapshot_slots(
    requested: usize,
    prefix_caching_active: bool,
) -> MarconiSlotDecision {
    marconi_snapshot_slots_with(requested, prefix_caching_active, marconi_reserve_full())
}

/// 2026-09-25: Whether `METRALE_SSM_MARCONI_FULL` is present, with any value.
pub fn marconi_reserve_full() -> bool {
    std::env::var_os("METRALE_SSM_MARCONI_FULL").is_some()
}

/// 2026-09-25: Env-free core of [`marconi_snapshot_slots`].
pub fn marconi_snapshot_slots_with(
    requested: usize,
    prefix_caching_active: bool,
    full_reserve: bool,
) -> MarconiSlotDecision {
    if requested == 0 || prefix_caching_active || full_reserve {
        return MarconiSlotDecision {
            slots: requested,
            skip_reason: None,
        };
    }
    MarconiSlotDecision {
        slots: 0,
        skip_reason: Some("prefix caching inactive — Marconi snapshot slots are unreachable"),
    }
}

/// 2026-09-25: Whether the prefix cache the server will install is a real cache: the
/// flag and `ModelConfig::kv_only_prefix_cache_is_safe`, the predicate
/// `build_prefix_cache` uses. Preflight calls this before the cache exists;
/// `TransformerModel::new` asks the constructed cache (`PrefixCache::is_active`).
pub fn prefix_caching_active(enable_flag: bool, kv_only_prefix_cache_is_safe: bool) -> bool {
    enable_flag && kv_only_prefix_cache_is_safe
}

#[cfg(test)]
#[path = "ssm_reserve_tests.rs"]
mod mtp_state_slot_tests;
