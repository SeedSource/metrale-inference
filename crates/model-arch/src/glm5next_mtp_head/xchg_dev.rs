// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_MTP_XCHG_DEVICE=1` (race-memory #80, C1 round): the vocab-sharded draft
//! head's cross-rank pick packet is written on the device (`kernels/gb10/common/mtp_head_xchg.cu`)
//! right behind the argmax, so the all-reduce is enqueued without a host round trip. The host path
//! it replaces (`forward_one`) syncs, reads the local argmax and its logit back, packs on the host and
//! copies the packet up: one sync, two D2H and one H2D per draft step, on both ranks.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - Off unless the variable is exactly `1` (after trimming); read once per process. Off, the kernel
//!   is not looked up and `forward_one` runs the host path unchanged.
//! - The kernel writes the bytes the host packer writes (same lanes, same BF16 bits), and
//!   [`pick_from_row`] is the host decode `forward_one` and `exchange_picks` run, so the pick is
//!   unchanged. Checked bitwise against the host packer by `mtp_head_xchg_microtest`.
//! - A missing kernel (handle 0) keeps the host path, warning once.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-12: Whether the variable's text turns the device packet on.
pub(super) fn parse_xchg_device(v: Option<&str>) -> bool {
    v.is_some_and(|s| s.trim() == "1")
}

/// 2026-10-12: The process-wide setting.
pub(super) fn xchg_device() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_xchg_device(std::env::var("METRALE_GLM_MTP_XCHG_DEVICE").ok().as_deref())
    })
}

/// 2026-10-12: The kernel handle when the lever is on and the target ships it, else handle 0.
/// Logs the ENGAGED line (or the fallback warning) once per process.
pub(super) fn resolve(gpu: &dyn GpuBackend) -> KernelHandle {
    if !xchg_device() {
        return KernelHandle(0);
    }
    let k = metrale_model_layers::layers::try_kernel(gpu, "mtp_head_xchg", "mtp_head_xchg_pack");
    if k.0 == 0 {
        tracing::warn!(
            "METRALE_GLM_MTP_XCHG_DEVICE=1 ignored: kernel mtp_head_xchg_pack missing; host packet"
        );
    } else {
        tracing::warn!(
            "METRALE_GLM_MTP_XCHG_DEVICE=1: ENGAGED - draft head pick packet written on the device"
        );
    }
    k
}

/// 2026-10-12: Write `n_rows` 16-byte pick rows into `xchg` from `arg` (`[n_rows]` u32 local
/// indices) and `logits` (`[n_rows, nv]` BF16), for vocab shard `v0` of rank `rank`.
#[allow(clippy::too_many_arguments)]
pub fn launch_pack(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    arg: DevicePtr,
    logits: DevicePtr,
    xchg: DevicePtr,
    nv: u32,
    v0: u32,
    rank: u32,
    n_rows: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(rank < 2, "mtp_head_xchg_pack: rank {rank} outside 0..2");
    KernelLaunch::new(gpu, kernel)
        .grid([n_rows.div_ceil(32), 1, 1])
        .block([32, 1, 1])
        .arg_ptr(arg)
        .arg_ptr(logits)
        .arg_ptr(xchg)
        .arg_u32(nv)
        .arg_u32(v0)
        .arg_u32(rank)
        .arg_u32(n_rows)
        .launch(stream)
}

/// 2026-10-12: The host packer's 16 bytes for one row (`forward_one`): this rank's raw BF16
/// logit `lb` in lane `rank`, the three base-256 digits of `g` in lanes `2 + 3 rank + d`.
pub fn host_row(rank: usize, lb: [u8; 2], g: usize) -> [u8; 16] {
    let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes();
    let mut pack = [0u8; 16];
    pack[rank * 2..][..2].copy_from_slice(&lb);
    for d in 0..3 {
        let digit = ((g >> (8 * d)) & 0xFF) as f32;
        pack[4 + (rank * 3 + d) * 2..][..2].copy_from_slice(&bf(digit));
    }
    pack
}

/// 2026-10-12: The pick from one all-reduced row: the higher logit's rank, the lower rank on a
/// tie (`>=`), then its three digits. The decode of `forward_one` and `exchange_picks`.
pub(super) fn pick_from_row(row: &[u8]) -> u32 {
    let lane =
        |j: usize| f32::from_bits((u16::from_le_bytes([row[j * 2], row[j * 2 + 1]]) as u32) << 16);
    let win = if lane(0) >= lane(1) { 0 } else { 1 };
    (0..3).fold(0usize, |a, d| {
        a + ((lane(2 + win * 3 + d) as usize) << (8 * d))
    }) as u32
}

#[cfg(test)]
mod tests {
    use super::{host_row, parse_xchg_device, pick_from_row};

    #[test]
    fn parse_is_off_unless_one() {
        assert!(!parse_xchg_device(None));
        assert!(!parse_xchg_device(Some("0")));
        assert!(!parse_xchg_device(Some("true")));
        assert!(parse_xchg_device(Some("1")));
        assert!(parse_xchg_device(Some(" 1\n")));
    }

    /// The SUM of two ranks' rows (each the other's lanes zero) decodes to the higher logit's
    /// global index, the lower rank on a tie.
    #[test]
    fn summed_rows_pick_the_winner() {
        let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes();
        let sum = |a: [u8; 16], b: [u8; 16]| {
            let mut o = [0u8; 16];
            for j in 0..8 {
                let f = |r: &[u8; 16]| {
                    f32::from_bits((u16::from_le_bytes([r[2 * j], r[2 * j + 1]]) as u32) << 16)
                };
                o[2 * j..][..2].copy_from_slice(&bf(f(&a) + f(&b)));
            }
            o
        };
        for (l0, l1, g0, g1, want) in [
            (3.5f32, 2.0f32, 17usize, 154_000usize, 17u32),
            (-1.0, 2.0, 0, 154_879, 154_879),
            (7.25, 7.25, 77_440, 77_441, 77_440),
            (0.0, -0.0, 255, 65_536, 255),
        ] {
            let r = sum(host_row(0, bf(l0), g0), host_row(1, bf(l1), g1));
            assert_eq!(pick_from_row(&r), want, "{l0} {l1} {g0} {g1}");
        }
    }
}
