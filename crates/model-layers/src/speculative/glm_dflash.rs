// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: GLM-5.3 (`glm5_next`) DFlash bring-up: the opt-in lever, the EP worker command
//! for the K=γ verify, the tap-layer flags, the worker's keep-length arithmetic and the
//! acceptance accounting. Everything here is host-side and pure, so it is CPU-tested
//! (`glm_dflash_tests.rs`).
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - [`glm_dflash_tap_flags`] refuses a tap outside the text stack instead of dropping it: a
//!   silently missing tap leaves the drafter reading zeros and the serve still lossless, so
//!   only acceptance would show it.
//! - [`kgamma_keep_len`] never keeps more rows than the verify wrote, nor fewer than the
//!   pre-verify prefix plus the anchor row.

use anyhow::{Result, bail};

/// 2026-10-01: EP worker command: run the K=γ DFlash verify alongside rank 0. Payload: `k`,
/// then the `k` verify tokens in one bulk broadcast; after rank 0's accept walk, `num_accepted`
/// (drafts accepted, `0..k`). The value was reserved for this in
/// `decode_checkpoint/plan.rs` (A113).
pub const EP_CMD_VERIFY_KGAMMA: u32 = 0xFFFF_FFF6;

/// 2026-10-01: The widest K=γ verify the worker accepts. The kgamma verify's seq-slot buffer
/// holds K <= 32 rows (`verify_d.rs`); a larger word is a desynchronised stream, not a request.
pub const KGAMMA_MAX_K: usize = 32;

/// 2026-10-01: `METRALE_GLM_DFLASH=1` allows a DFlash drafter on a `glm5_next` target: mHC tap
/// collapse at the drafter's tap layers, the K=γ verify on every rank (EP worker command), and
/// CUDA-graph verify under EP. Off by default; a GLM serve given a drafter without it refuses
/// to build ([`glm_dflash_gate`]). Read once per process.
pub fn glm_dflash_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse_switch(std::env::var("METRALE_GLM_DFLASH").ok().as_deref()))
}

/// 2026-10-01: The lever predicate: exactly `1` arms it.
pub fn parse_switch(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-01: Build-time gate. A `glm5_next` target with a DFlash drafter needs the lever;
/// any other model, or no drafter, passes untouched.
pub fn glm_dflash_gate(model_type: &str, has_drafter: bool, lever_on: bool) -> Result<()> {
    let glm = matches!(model_type, "glm5_next" | "glm5_next_text");
    if glm && has_drafter && !lever_on {
        bail!(
            "DFlash on {model_type} is opt-in and unvalidated: set METRALE_GLM_DFLASH=1 \
             (mHC tap collapse, EP K=γ verify) or drop --dflash. Without it the drafter \
             would read mHC scratch instead of its taps and a 2-rank verify would hang."
        );
    }
    Ok(())
}

/// 2026-10-01: Per-layer tap flags for a text stack of `num_layers` layers: `flags[i]` is true
/// when layer `i` is one of `capture_layers` (the drafter's `target_layer_ids`, 0-based, tap =
/// the layer's completed output). Errors on an out-of-range or duplicated tap.
pub fn glm_dflash_tap_flags(capture_layers: &[usize], num_layers: usize) -> Result<Vec<bool>> {
    let mut flags = vec![false; num_layers];
    for &l in capture_layers {
        if l >= num_layers {
            bail!(
                "DFlash tap layer {l} is outside the {num_layers}-layer text stack \
                 (drafter target_layer_ids must be 0-based target layer indices)"
            );
        }
        if flags[l] {
            bail!("DFlash tap layer {l} is listed twice in target_layer_ids");
        }
        flags[l] = true;
    }
    Ok(flags)
}

/// 2026-10-01: Rows a sequence keeps after a K-row verify. `seq_len_after` is the length the
/// verify left (pre-verify length + `k`); the sequence keeps the pre-verify prefix, the anchor
/// row and the `num_accepted` accepted drafts. Errors on a payload that cannot describe a
/// verify of `k` rows: `k == 0`, `num_accepted >= k` (the bonus row is not a draft), or a
/// length shorter than `k`.
pub fn kgamma_keep_len(seq_len_after: usize, k: usize, num_accepted: usize) -> Result<usize> {
    if k == 0 || k > KGAMMA_MAX_K {
        bail!("K=γ verify width {k} outside 1..={KGAMMA_MAX_K}");
    }
    if num_accepted >= k {
        bail!(
            "K=γ verify: num_accepted {num_accepted} >= k {k}; at most k-1 drafts can be \
             accepted (row 0 is the anchor). A bonus off-by-one on the head?"
        );
    }
    if seq_len_after < k {
        bail!("K=γ verify: seq_len {seq_len_after} is shorter than the {k} verified rows");
    }
    Ok(seq_len_after - k + num_accepted + 1)
}

/// 2026-10-01: Running acceptance for one serve: verify steps, drafts offered and accepted,
/// and a histogram of accepted drafts per step (index = accepted, length = widest K seen).
#[derive(Debug, Default, Clone, PartialEq)]
pub struct AcceptStats {
    pub steps: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub hist: Vec<u64>,
}

impl AcceptStats {
    /// 2026-10-01: Record one verify that offered `drafted` drafts and accepted `accepted`.
    /// `accepted` is clamped to `drafted`.
    pub fn record(&mut self, drafted: usize, accepted: usize) {
        let a = accepted.min(drafted);
        self.steps += 1;
        self.drafted += drafted as u64;
        self.accepted += a as u64;
        if self.hist.len() <= drafted {
            self.hist.resize(drafted + 1, 0);
        }
        self.hist[a] += 1;
    }

    /// 2026-10-01: Mean accepted drafts per step (0 before any step).
    pub fn mean_accepted(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.accepted as f64 / self.steps as f64
        }
    }

    /// 2026-10-01: Mean tokens emitted per verify step: accepted drafts plus the bonus.
    pub fn mean_len(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.mean_accepted() + 1.0
        }
    }

    /// 2026-10-01: One log line, `steps=.. mean_accepted=.. mean_len=.. rate=.. hist=[..]`,
    /// the format `scripts/race/dflash-glm-check.sh` parses.
    pub fn summary(&self) -> String {
        let rate = if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        };
        format!(
            "steps={} mean_accepted={:.3} mean_len={:.3} rate={:.3} hist={:?}",
            self.steps,
            self.mean_accepted(),
            self.mean_len(),
            rate,
            self.hist
        )
    }
}

#[cfg(test)]
#[path = "glm_dflash_tests.rs"]
mod tests;
