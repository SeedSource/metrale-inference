// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The scenario set: one deterministic script per scheduler lane or lifecycle path.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! `all()` lists them.

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario};

fn toks(n: usize, base: u32) -> Vec<u32> {
    (0..n as u32).map(|i| base + (i % 20)).collect()
}

fn gen_eos(n: usize, base: u32) -> Vec<u32> {
    let mut g = toks(n, base);
    g.push(EOS);
    g
}

fn plain() -> Scenario {
    let mut r4 = ReqSpec::new(4, 6, toks(5, 30));
    r4.streaming = false;
    Scenario {
        name: "plain_decode",
        cfg: ModelCfg::default(),
        opts: RunOptions::default(),
        reqs: vec![
            ReqSpec::new(1, 5, gen_eos(6, 10)),
            ReqSpec::new(2, 5, toks(5, 20)),
            ReqSpec::new(3, 7, gen_eos(3, 40)),
            r4,
        ],
    }
}

fn chunked() -> Scenario {
    Scenario {
        name: "chunked_prefill",
        cfg: ModelCfg::default(),
        opts: RunOptions {
            max_prefill_tokens: 8,
            ..RunOptions::default()
        },
        reqs: vec![ReqSpec::new(1, 20, gen_eos(4, 10))],
    }
}

fn mixed() -> Scenario {
    let mut r2 = ReqSpec::new(2, 20, gen_eos(3, 20));
    r2.arrive_at_tick = Some(2);
    Scenario {
        name: "mixed_prefill",
        cfg: ModelCfg::default(),
        opts: RunOptions {
            max_prefill_tokens: 8,
            ..RunOptions::default()
        },
        reqs: vec![ReqSpec::new(1, 4, gen_eos(8, 10)), r2],
    }
}

fn batched_prefill() -> Scenario {
    Scenario {
        name: "batched_prefill_waves",
        cfg: ModelCfg::default(),
        opts: RunOptions {
            max_prefill_tokens: 8,
            ..RunOptions::default()
        },
        reqs: vec![
            ReqSpec::new(1, 20, gen_eos(3, 10)),
            ReqSpec::new(2, 17, gen_eos(2, 20)),
        ],
    }
}

fn batched_mixed() -> Scenario {
    let mut r2 = ReqSpec::new(2, 20, gen_eos(2, 20));
    r2.arrive_at_tick = Some(2);
    let mut r3 = ReqSpec::new(3, 18, gen_eos(2, 30));
    r3.arrive_at_tick = Some(2);
    Scenario {
        name: "batched_mixed",
        cfg: ModelCfg::default(),
        opts: RunOptions {
            max_prefill_tokens: 8,
            ..RunOptions::default()
        },
        reqs: vec![ReqSpec::new(1, 4, gen_eos(10, 10)), r2, r3],
    }
}

fn mtp_cfg() -> ModelCfg {
    ModelCfg {
        has_proposer: true,
        ..ModelCfg::default()
    }
}

fn verify_k(name: &'static str, num_drafts: usize) -> Scenario {
    let mut r = ReqSpec::new(1, 5, gen_eos(14, 10));
    r.draft_wrong_every = Some(4);
    Scenario {
        name,
        cfg: mtp_cfg(),
        opts: RunOptions {
            use_speculative: true,
            num_drafts,
            ..RunOptions::default()
        },
        reqs: vec![r],
    }
}

pub(super) fn batched_verify() -> Scenario {
    let mut reqs: Vec<ReqSpec> = (1..=3)
        .map(|i| ReqSpec::new(i, 4 + i as usize, gen_eos(12, 10 * i as u32)))
        .collect();
    reqs[1].draft_wrong_every = Some(3);
    Scenario {
        name: "batched_verify_k4",
        cfg: mtp_cfg(),
        opts: RunOptions {
            use_speculative: true,
            num_drafts: 3,
            ..RunOptions::default()
        },
        reqs,
    }
}

pub(super) fn dflash(name: &'static str, n: u64) -> Scenario {
    let reqs: Vec<ReqSpec> = (1..=n)
        .map(|i| {
            let mut r = ReqSpec::new(i, 5, gen_eos(16, 10 * i as u32));
            r.draft_wrong_every = Some(5);
            r
        })
        .collect();
    Scenario {
        name,
        cfg: ModelCfg {
            has_proposer: true,
            dflash_gamma: Some(6),
            ..ModelCfg::default()
        },
        opts: RunOptions {
            use_speculative: true,
            dflash: true,
            num_drafts: 5,
            ..RunOptions::default()
        },
        reqs,
    }
}

fn ngram() -> Scenario {
    // 2026-09-25: after its first token the prompt repeats with period 7
    // (see `build_request`), which gives the n-gram proposer repeats to
    // match.
    Scenario {
        name: "ngram",
        cfg: ModelCfg::default(),
        opts: RunOptions {
            ngram_speculative: true,
            ..RunOptions::default()
        },
        reqs: vec![ReqSpec::new(1, 15, vec![8, 2, 3, 4, 9, 6, 7, 8, 2, EOS])],
    }
}

fn self_spec() -> Scenario {
    let mut r = ReqSpec::new(1, 5, gen_eos(10, 10));
    r.draft_wrong_every = Some(3);
    Scenario {
        name: "self_spec",
        cfg: ModelCfg {
            has_self_speculative: true,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            self_speculative: true,
            num_drafts: 2,
            ..RunOptions::default()
        },
        reqs: vec![r],
    }
}

fn beam() -> Scenario {
    let mut r = ReqSpec::new(1, 5, vec![EOS]);
    r.streaming = false;
    r.num_beams = 2;
    r.beam_hyp = Some(vec![10, 11, 12, EOS]);
    r.max_tokens = 8;
    Scenario {
        name: "beam",
        cfg: ModelCfg {
            supports_beam: true,
            ..ModelCfg::default()
        },
        opts: RunOptions::default(),
        reqs: vec![r, ReqSpec::new(2, 5, gen_eos(2, 20))],
    }
}

fn preempt() -> Scenario {
    Scenario {
        name: "preempt_requeue",
        cfg: ModelCfg {
            kv_exhaust_at: vec![1],
            ..ModelCfg::default()
        },
        opts: RunOptions::default(),
        reqs: vec![
            ReqSpec::new(1, 5, gen_eos(6, 10)),
            ReqSpec::new(2, 5, gen_eos(4, 20)),
            ReqSpec::new(3, 5, gen_eos(5, 30)),
        ],
    }
}

fn spill() -> Scenario {
    let mut r2 = ReqSpec::new(2, 96, gen_eos(3, 20));
    r2.arrive_at_tick = Some(2);
    Scenario {
        name: "spill_swap_out_in",
        cfg: ModelCfg {
            total_blocks: 10,
            held_blocks: 4,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            swap_space_gb: 1,
            ..RunOptions::default()
        },
        reqs: vec![ReqSpec::new(1, 32, gen_eos(6, 10)), r2],
    }
}

fn cancel() -> Scenario {
    let mut r = ReqSpec::new(1, 5, toks(20, 10));
    r.cancel_at = Some(5);
    Scenario {
        name: "cancel_mid_stream",
        cfg: mtp_cfg(),
        opts: RunOptions {
            use_speculative: true,
            num_drafts: 1,
            ..RunOptions::default()
        },
        reqs: vec![r, ReqSpec::new(2, 5, gen_eos(3, 20))],
    }
}

fn timeout() -> Scenario {
    let mut r = ReqSpec::new(1, 5, toks(20, 10));
    r.expired_deadline = true;
    Scenario {
        name: "request_timeout",
        cfg: ModelCfg::default(),
        opts: RunOptions::default(),
        reqs: vec![r, ReqSpec::new(2, 5, gen_eos(3, 20))],
    }
}

fn lora() -> Scenario {
    Scenario {
        name: "lora_rotation_at_quiescence",
        cfg: ModelCfg::default(),
        opts: RunOptions {
            lora_rotation: Some("adapter-x".to_string()),
            ..RunOptions::default()
        },
        reqs: vec![ReqSpec::new(1, 5, gen_eos(3, 10))],
    }
}

fn slai() -> Scenario {
    Scenario {
        name: "slai_policy",
        cfg: ModelCfg::default(),
        opts: RunOptions {
            slai_policy: true,
            max_prefill_tokens: 8,
            ..RunOptions::default()
        },
        reqs: vec![
            ReqSpec::new(1, 5, gen_eos(4, 10)),
            ReqSpec::new(2, 20, gen_eos(2, 20)),
        ],
    }
}

fn watchdog() -> Scenario {
    // 2026-09-25: 60 distinct tokens with a boundary (12) at index 50,
    // then a period-20 loop (19 tokens and the boundary). The loop
    // watchdog fires and the rollback restores the SSM snapshot taken at
    // the first boundary (`golden/watchdog_rollback.trace`).
    let mut script: Vec<u32> = (100..160u32).collect();
    script[50] = 12;
    for _ in 0..7 {
        script.extend(20..39u32);
        script.push(12);
    }
    let mut r = ReqSpec::new(1, 5, script);
    r.streaming = false;
    Scenario {
        name: "watchdog_rollback",
        cfg: ModelCfg {
            vocab: 256,
            has_ssm_layers: true,
            ring_slots: 8,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            loop_watchdog: true,
            boundary_tokens: vec![12],
            ..RunOptions::default()
        },
        reqs: vec![r],
    }
}

fn host_logits() -> Scenario {
    let mut r1 = ReqSpec::new(1, 5, gen_eos(4, 10));
    r1.top_logprobs = Some(2);
    let mut r2 = ReqSpec::new(2, 5, gen_eos(4, 20));
    r2.temperature = 0.7;
    let mut r3 = ReqSpec::new(3, 5, vec![30, EOS, 31, 32, EOS]);
    r3.min_tokens = 3;
    r3.max_tokens = 5;
    Scenario {
        name: "host_logits_paths",
        cfg: ModelCfg::default(),
        opts: RunOptions::default(),
        reqs: vec![r1, r2, r3],
    }
}

pub(super) fn all() -> Vec<Scenario> {
    vec![
        plain(),
        chunked(),
        mixed(),
        batched_prefill(),
        batched_mixed(),
        verify_k("verify_k2", 1),
        verify_k("verify_k3", 2),
        verify_k("verify_k4", 3),
        batched_verify(),
        dflash("dflash", 1),
        dflash("dflash_batched", 2),
        ngram(),
        self_spec(),
        beam(),
        preempt(),
        spill(),
        cancel(),
        timeout(),
        lora(),
        slai(),
        watchdog(),
        host_logits(),
    ]
}
