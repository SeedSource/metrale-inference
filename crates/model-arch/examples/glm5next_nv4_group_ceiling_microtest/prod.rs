// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: PROD mode of `glm5next_nv4_group_ceiling_microtest`: the production entry of
//! `METRALE_GLM_NV4_B3_GROUP` (`dense_nv4_group::try_group` -> `dense_nv4_b3_group`) against the
//! production per-projection route (`dense_fp8::route`), on weights registered the way the loader
//! registers them (`convert_weight_nvfp4`: real per-tensor `weight_scale_2`, which differs per
//! member here because the members get different magnitudes).
//!
//! Owner: model-arch examples.
//! Per group and M in 1..=3: separate = one `route` per member into its own buffer; grouped =
//! `try_group` into other buffers (poisoned first). M = 3 must engage and be bitwise equal to
//! the separate outputs; M = 1, 2 must decline (`Ok(false)`, nothing launched: the poison
//! survives). Timing (information): M = 3, S = the members' routes, G = `try_group`.

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_layer::dense_fp8 as df;
use metrale_model_arch::glm5next_layer::dense_nv4_group::{Proj, try_group};

use crate::GROUPS;
use crate::data::{Lcg, dn_u16, gen_act, up_bf16};
use crate::timing::{Events, time_arms};

/// The environment the production entry reads (once): call before anything reads it.
pub fn set_env() {
    // SAFETY: called first in `main`, single-threaded, before anything reads the environment.
    unsafe {
        std::env::set_var("METRALE_GLM_DENSE_FP8", "1");
        std::env::set_var("METRALE_GLM_DENSE_NVFP4", "kda");
        std::env::set_var("METRALE_GLM_NV4_B3_STAGED", "1");
        std::env::set_var("METRALE_GLM_NV4_B3_GROUP", "1");
        for v in [
            "METRALE_GLM_NV4_TC",
            "METRALE_GLM_NV4_TC_GROUP",
            "METRALE_GLM_GEMV_TC",
        ] {
            std::env::remove_var(v);
        }
    }
}

/// One registered group: the FP8 keys (what the layer fields hold) and the member Ns.
struct Reg {
    name: &'static str,
    keys: Vec<DevicePtr>,
    ns: Vec<usize>,
    k: usize,
}

fn register(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<Vec<Reg>> {
    let mut out = Vec::new();
    for &(name, ns, k, _) in GROUPS {
        let mut keys = Vec::new();
        let mut acc = df::LayerFp8::default();
        for (j, &n) in ns.iter().enumerate() {
            let f = 0.02 * (1.0 + j as f64);
            let w: Vec<bf16> = gen_act(rng, n, k)
                .into_iter()
                .map(|x| bf16::from_f64(x.to_f64() * f))
                .collect();
            let mut key = up_bf16(g, &w)?;
            ensure!(
                df::convert_weight_nvfp4(g, &mut key, n, k, &mut acc)?,
                "{name}: convert_weight_nvfp4 did not convert member {j}"
            );
            keys.push(key);
        }
        out.push(Reg {
            name,
            keys,
            ns: ns.to_vec(),
            k,
        });
    }
    // Resolves the grouped kernel (`dense_nv4_b3_group::finish_load`) as the loader does.
    df::finish_load(g)?;
    Ok(out)
}

/// Runs PROD; returns the number of failed checks.
pub fn run(
    g: &dyn GpuBackend,
    ev: &Events,
    rng: &mut Lcg,
    preludes: [&dyn Fn(u64) -> Result<()>; 2],
) -> Result<usize> {
    let s = g.default_stream();
    let gemv = g.kernel("gemv", "dense_gemv_bf16")?;
    let regs = register(g, rng)?;
    let mut bad = 0usize;
    for r in &regs {
        let s2: Vec<f32> = r
            .keys
            .iter()
            .zip(&r.ns)
            .map(|(&key, &n)| df::lookup_nv4(key, n, r.k).map_or(f32::NAN, |q| q.weight_scale_2))
            .collect();
        let distinct = s2.windows(2).any(|w| w[0] != w[1]);
        println!("PROD SCALE2 {} {s2:?} distinct={distinct}", r.name);
        if !distinct {
            bad += 1;
            println!(
                "FAIL PROD {}: members share one weight_scale_2; the test needs them distinct",
                r.name
            );
        }
        let total: usize = r.ns.iter().sum();
        for m in 1..=3usize {
            let a = up_bf16(g, &gen_act(rng, m, r.k))?;
            let sep: Vec<DevicePtr> =
                r.ns.iter()
                    .map(|&n| g.alloc(m * n * 2))
                    .collect::<Result<_>>()?;
            let grp: Vec<DevicePtr> =
                r.ns.iter()
                    .map(|&n| g.alloc(m * n * 2))
                    .collect::<Result<_>>()?;
            let poison = vec![0xA5u8; m * total * 2];
            for (&p, &n) in sep.iter().chain(&grp).zip(r.ns.iter().chain(&r.ns)) {
                g.copy_h2d(&poison[..m * n * 2], p)?;
            }
            let arm_s = |st: u64| -> Result<()> {
                for ((&key, &c), &n) in r.keys.iter().zip(&sep).zip(&r.ns) {
                    let rt = df::route(g, gemv, a, key, c, m, n, r.k, st)?;
                    ensure!(
                        rt == df::Route::Done,
                        "{}: route M={m} did not run the NVFP4 GEMV",
                        r.name
                    );
                }
                Ok(())
            };
            let members: Vec<Proj> = r
                .keys
                .iter()
                .zip(&grp)
                .zip(&r.ns)
                .map(|((&b, &c), &n)| Proj { b, c, n })
                .collect();
            arm_s(s)?;
            let engaged = try_group(g, gemv, a, &members, m, r.k, s)?;
            g.synchronize(s)?;
            let (mut diff, mut poisoned) = (0usize, 0usize);
            for ((&cs, &cg), &n) in sep.iter().zip(&grp).zip(&r.ns) {
                let ys = dn_u16(g, cs, m * n)?;
                let yg = dn_u16(g, cg, m * n)?;
                diff += ys.iter().zip(&yg).filter(|(x, y)| x != y).count();
                poisoned += yg.iter().filter(|&&y| y == 0xA5A5).count();
            }
            let want = m == 3;
            if engaged != want {
                bad += 1;
                println!("FAIL PROD {} M={m}: engaged={engaged}, want {want}", r.name);
            } else if engaged && diff > 0 {
                bad += 1;
                println!(
                    "MISMATCH PROD {} M={m}: {diff} differing output elements",
                    r.name
                );
            } else if engaged {
                println!("PROD BITWISE {} M={m}: 0 differing output elements", r.name);
            } else if poisoned != m * total {
                bad += 1;
                println!(
                    "FAIL PROD {} M={m}: declined but wrote {} elements",
                    r.name,
                    m * total - poisoned
                );
            } else {
                println!(
                    "PROD DECLINED {} M={m} (falls back; nothing launched)",
                    r.name
                );
            }
            if engaged {
                let arm_g = |st: u64| -> Result<()> {
                    ensure!(
                        try_group(g, gemv, a, &members, m, r.k, st)?,
                        "try_group declined"
                    );
                    Ok(())
                };
                for (i, l2) in ["cold", "chain"].into_iter().enumerate() {
                    let t = time_arms(ev, s, preludes[i], &[&arm_s, &arm_g])?;
                    println!(
                        "PROD GROUP {} M={m} L2={l2} S {:.2} G {:.2} saved {:.2}",
                        r.name,
                        t[0],
                        t[1],
                        t[0] - t[1]
                    );
                }
            }
            g.synchronize(s)?;
            for p in sep.into_iter().chain(grp) {
                g.free(p)?;
            }
            g.free(a)?;
        }
    }
    Ok(bad)
}
