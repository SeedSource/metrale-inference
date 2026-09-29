// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`SchedLimits`]: the run's hard stops, from this model's tokenizer and the CLI.
//!
//! `serve_load` builds the value for a run: the token ids come from
//! `TokenizerRuntime` and `max_seq_len` from `--max-seq-len`. The token ids are
//! valid only for the tokenizer that resolved them. 2026-09-29: A146 adds the
//! code-fence token, which is not a stop (see its field).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: Hard output/length limits for one scheduler run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedLimits {
    /// 2026-09-25: `<|im_start|>` as a single token id, when it encodes to one.
    /// Emitting it means the model has begun a new ChatML turn on its own, so
    /// the sequence is stopped there. `None` → no such hard stop.
    pub im_start_hard_stop: Option<u32>,
    /// 2026-09-25: `<tool_response>` as a single token id, when it encodes to one.
    /// Emitting it stops the sequence only while `SchedLevers::tool_response_stop`
    /// is set (on unless `METRALE_TOOL_RESPONSE_STOP` is `0` or `false`).
    pub tool_response_hard_stop: Option<u32>,
    /// 2026-09-25: Served-context ceiling (`--max-seq-len`), checked per decode step.
    /// `0` means no ceiling: `helpers::seqlen_force_stop` then never fires.
    pub max_seq_len: usize,
    /// 2026-09-29: A146: the tokenizer's atomic ``` code-fence token, when it
    /// has one. Not a stop: `emit_token` needs it to track `in_code_fence`
    /// exactly as `process_decode_logits` does (which receives it as a
    /// parameter), so the forced-`</think>` fence deferral agrees across the
    /// two commit paths.
    pub code_fence_token: Option<u32>,
}

impl SchedLimits {
    /// 2026-09-25: No hard stops and no ceiling. The test constructors of `SchedCtx`
    /// use it.
    pub const NONE: Self = Self {
        im_start_hard_stop: None,
        tool_response_hard_stop: None,
        max_seq_len: 0,
        code_fence_token: None,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_the_default_and_disables_every_guard() {
        assert_eq!(SchedLimits::default(), SchedLimits::NONE);
        assert_eq!(SchedLimits::NONE.max_seq_len, 0);
        assert!(SchedLimits::NONE.im_start_hard_stop.is_none());
    }

    #[test]
    fn two_runs_can_hold_different_token_ids() {
        let a = SchedLimits {
            im_start_hard_stop: Some(151644),
            ..SchedLimits::NONE
        };
        let b = SchedLimits {
            im_start_hard_stop: Some(200),
            ..SchedLimits::NONE
        };
        assert_ne!(a.im_start_hard_stop, b.im_start_hard_stop);
    }
}
