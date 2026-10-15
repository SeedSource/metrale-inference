// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host-side token sampling: a plain sample, a
//! grammar-constrained sample and the first token after prefill.
//!
//! Owner: scheduler.
//! Invariants:
//! - `sample_token` and `sample_token_with_grammar` hand the sampler
//!   [`effective_min_p`] of their `min_p`, never the raw value: the caller's
//!   value, or `0.0` when the run's `mtp_minp` lever is off
//!   (`METRALE_NO_MTP_MINP=1`).

use super::*;

/// 2026-09-25: Re-sample `argmax_tokens.len()` verify positions from the
/// model's logits buffer at `temperature`. Returns `argmax_tokens`
/// unchanged at temperature 0 or when the device-to-host copy fails.
///
/// Nothing calls it. It samples the raw logits: no grammar bitmask, no
/// logit processors, `min_p` 0.
#[allow(dead_code)]
pub fn verify_resample(model: &dyn Model, argmax_tokens: &[u32], temperature: f32) -> Vec<u32> {
    if temperature == 0.0 {
        return argmax_tokens.to_vec();
    }
    let k = argmax_tokens.len();
    let vocab = model.vocab_size();
    let total_bytes = k * vocab * 2;
    let mut buf = vec![0u8; total_bytes];
    if model
        .copy_logits_to_host(model.logits_buffer_ptr(), &mut buf)
        .is_err()
    {
        return argmax_tokens.to_vec();
    }
    let params = SamplingParams {
        temperature,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repetition_penalty_window: 0,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: DEFAULT_DRY_MULTIPLIER,
        dry_base: DEFAULT_DRY_BASE,
        dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    (0..k)
        .map(|i| {
            let slice = &buf[i * vocab * 2..(i + 1) * vocab * 2];
            sample_with_params(slice, &params)
        })
        .collect()
}

/// 2026-09-25: Sample one token from device logits.
///
/// Greedy with an empty `suppress_ids` is a device argmax. Otherwise the
/// logits are copied to host, every id in `suppress_ids` is set to -inf,
/// and the token is the host argmax (temperature 0) or a
/// temperature/top-k/top-p/min-p sample with neutral penalties.
pub fn sample_token(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    suppress_ids: &[u32],
    levers: &crate::scheduler::logit_processors::SamplingLevers,
    dumps: &crate::scheduler::dumps::RunDumps,
) -> Result<u32> {
    if temperature == 0.0 && suppress_ids.is_empty() {
        return model.argmax_on_device(logits, 0);
    }
    let vocab_size = model.vocab_size();
    let mut f32_logits: Vec<f32> = if model.logits_ptr_is_fp32(logits) {
        let mut buf = vec![0u8; vocab_size * 4];
        model.copy_logits_to_host(logits, &mut buf)?;
        // 2026-09-25: SAFETY: `buf` holds `vocab_size * 4` bytes of f32
        // logits. The cast also needs `buf` to be 4-byte aligned, which a
        // `Vec<u8>` does not promise.
        let f32_slice: &[f32] =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, vocab_size) };
        f32_slice.to_vec()
    } else {
        let mut bf16_buf = vec![0u8; vocab_size * 2];
        model.copy_logits_to_host(logits, &mut bf16_buf)?;
        (0..vocab_size)
            .map(|i| {
                let lo = bf16_buf[i * 2];
                let hi = bf16_buf[i * 2 + 1];
                bf16_to_f32(lo, hi)
            })
            .collect()
    };
    // 2026-09-25: With `METRALE_DUMP_LOGITS_PATH=<dir>`, append this
    // call's FP32 logits to `<dir>/logits_stok.bin`. Open and write errors
    // are ignored.
    if let Some(dir) = dumps.raw_logits_dir.as_deref() {
        use std::io::Write;
        let path = dir.join("logits_stok.bin");
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4)
            };
            let _ = f.write_all(bytes);
        }
    }
    for &id in suppress_ids {
        if (id as usize) < vocab_size {
            f32_logits[id as usize] = f32::NEG_INFINITY;
        }
    }
    if temperature == 0.0 {
        // 2026-09-25: An exact f32 tie keeps the lower id. `max_by` keeps the last tie and
        // turned a 22.0/22.0 logit tie of "1" vs "2" into the token "2".
        let best = metrale_sampling::argmax_first_wins_f32(&f32_logits);
        return Ok(best);
    }
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    Ok(sample_with_params(
        f32_bytes,
        &SamplingParams {
            temperature,
            top_k,
            top_p,
            top_n_sigma: 0.0,
            min_p: effective_min_p(min_p, levers),
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty_window: 0,
            lz_penalty: DEFAULT_LZ_PENALTY,
            dry_multiplier: DEFAULT_DRY_MULTIPLIER,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: None,
        },
    ))
}

/// 2026-09-25: Sample one token from device logits under an optional
/// grammar, with the penalties in `penalties` applied over `history`.
///
/// On the host path the order is: `suppress_ids` set to -inf, the grammar
/// bitmask, [`apply_penalties_and_bias`], then the host argmax
/// (temperature 0) or a temperature/top-k/top-p/min-p sample. `min_p`
/// comes from `penalties.min_p`.
pub fn sample_token_with_grammar(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    suppress_ids: &[u32],
    mut grammar_state: Option<&mut GrammarState>,
    penalties: &SamplingParams,
    history: &[u32],
    levers: &crate::scheduler::logit_processors::SamplingLevers,
) -> Result<u32> {
    // 2026-09-25: Greedy fast path, skipping the host copy. When the pick
    // is greedy, nothing is suppressed, the penalties cannot move the
    // argmax (`fast_greedy::classify_penalties` / `argmax_immune`) and the
    // grammar allows the device argmax, that argmax is also the masked,
    // penalised argmax: a grammar-allowed global maximum is the maximum of
    // the allowed set. Otherwise fall through to the host path.
    // `METRALE_DISABLE_FAST_GREEDY=1` turns the fast path off.
    if levers.fast_greedy_grammar
        && suppress_ids.is_empty()
        && (temperature == 0.0 || levers.force_temp_zero)
    {
        let gate = crate::scheduler::fast_greedy::classify_penalties(penalties);
        if gate != crate::scheduler::fast_greedy::PenaltyGate::Blocked {
            let top1 = model.argmax_on_device(logits, 0)?;
            let immune = gate == crate::scheduler::fast_greedy::PenaltyGate::Neutral
                || crate::scheduler::fast_greedy::argmax_immune(top1, history, || {
                    crate::scheduler::fast_greedy::logit_is_positive(
                        model,
                        logits,
                        0,
                        model.vocab_size(),
                        top1,
                    )
                });
            if immune {
                let allowed = match grammar_state.as_mut() {
                    Some(gs) => {
                        if gs.is_terminated() {
                            true
                        } else {
                            gs.fill_bitmask();
                            gs.is_token_allowed(top1)
                        }
                    }
                    None => true,
                };
                if allowed {
                    return Ok(top1);
                }
            }
        }
    }

    let vocab_size = model.vocab_size();
    let mut bf16_buf = vec![0u8; vocab_size * 2];
    model.copy_logits_to_host(logits, &mut bf16_buf)?;
    let mut f32_logits: Vec<f32> = (0..vocab_size)
        .map(|i| {
            let lo = bf16_buf[i * 2];
            let hi = bf16_buf[i * 2 + 1];
            bf16_to_f32(lo, hi)
        })
        .collect();
    for &id in suppress_ids {
        if (id as usize) < vocab_size {
            f32_logits[id as usize] = f32::NEG_INFINITY;
        }
    }
    if let Some(gs) = grammar_state {
        gs.fill_bitmask();
        gs.apply_bitmask_to_logits(&mut f32_logits);
    }
    apply_penalties_and_bias(&mut f32_logits, penalties, history);
    if temperature == 0.0 {
        let best = metrale_sampling::argmax_first_wins_f32(&f32_logits);
        return Ok(best);
    }
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    // 2026-09-25: Penalties are already applied above. `sample_with_params`
    // applies them again with an empty history, so the params below are
    // neutral and only the temperature/top-k/top-p/min-p stages act.
    Ok(sample_with_params(
        f32_bytes,
        &SamplingParams {
            temperature,
            top_k,
            top_p,
            top_n_sigma: 0.0,
            min_p: effective_min_p(penalties.min_p, levers),
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty_window: 0,
            lz_penalty: DEFAULT_LZ_PENALTY,
            dry_multiplier: DEFAULT_DRY_MULTIPLIER,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: None,
        },
    ))
}

/// 2026-09-25: Sample the first generated token from the prefill's final
/// logits.
///
/// With no grammar, or with `policy.grammar_suspended`, this is
/// [`sample_token`]; a suspended grammar also adds `policy.tool_call_start`
/// to the suppressed ids. Otherwise the token is picked by
/// [`sample_token_with_grammar`] and the matcher is advanced past it (see
/// `first_token_policy::first_token_with`). Penalties are
/// neutral: there is no output history yet.
pub fn sample_first_token(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    suppress_ids: &[u32],
    grammar_state: Option<&mut GrammarState>,
    policy: FirstTokenPolicy,
    levers: &crate::scheduler::logit_processors::SamplingLevers,
    dumps: &crate::scheduler::dumps::RunDumps,
) -> Result<u32> {
    first_token_with(policy, suppress_ids, grammar_state, |ids, gs| {
        let Some(gs) = gs else {
            return sample_token(
                model,
                logits,
                temperature,
                top_k,
                top_p,
                min_p,
                ids,
                levers,
                dumps,
            );
        };
        let neutral = SamplingParams {
            temperature,
            top_k,
            top_p,
            top_n_sigma: 0.0,
            min_p,
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty_window: 0,
            lz_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: None,
        };
        sample_token_with_grammar(
            model,
            logits,
            temperature,
            top_k,
            top_p,
            ids,
            Some(gs),
            &neutral,
            &[],
            levers,
        )
    })
}

mod penalties;
pub(super) use penalties::*;

#[cfg(test)]
mod penalty_scope_tests;
#[cfg(test)]
mod reasoning_floor_tests;
