// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Byte-identity gate for the device-written MTP draft-head pick packet
//! (`METRALE_GLM_MTP_XCHG_DEVICE=1`, `kernels/gb10/common/mtp_head_xchg.cu`,
//! `glm5next_mtp_head::xchg_dev`) against the host packer `forward_one` runs today
//! (`xchg_dev::host_row`).
//!
//! Owner: model-arch examples (GLM-5.3 MTP drafter).
//! Invariants:
//! - Exits 1, with `FAIL ...` lines, unless every row's 16 bytes from the kernel equal the host
//!   packer's, for both ranks and both vocab shards, row counts 1, 3 and 16, and arguments that
//!   include the first and last local index.
//! - Case A takes `arg` from the real `argmax_bf16` kernel on the same logits (the engine order:
//!   argmax, then the packet), case B writes chosen indices directly.
//! - The packet buffer starts at poison 0xAB, so a byte the kernel leaves unwritten cannot pass.
//! - Refuses PASS when nothing was compared or when the kernel is missing.
//! - Also prints the single-rank host time of both sequences (no all-reduce): the host path's
//!   sync + 2 D2H + H2D + sync + D2H against the device path's launch + sync + D2H, median of 200.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example mtp_head_xchg_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mtp_head::xchg_dev::{host_row, launch_pack};
use metrale_model_layers::layers::ops;
use std::time::Instant;

/// GLM-5.3 vocab 154880 split over two ranks.
const NV: usize = 77_440;
const POISON: u8 = 0xAB;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    /// A BF16 in about [-24, 24), as raw little-endian bytes.
    fn bf16(&mut self) -> [u8; 2] {
        let x = (self.next() % 48_000) as f32 / 1000.0 - 24.0;
        ((x.to_bits() >> 16) as u16).to_le_bytes()
    }
}

fn upload(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let stream = g.default_stream();
    let pack_k = metrale_model_layers::layers::try_kernel(g, "mtp_head_xchg", "mtp_head_xchg_pack");
    if pack_k.0 == 0 {
        println!("FAIL mtp_head_xchg_microtest: kernel mtp_head_xchg::mtp_head_xchg_pack missing");
        std::process::exit(1);
    }
    let argmax_k = g.kernel("argmax", "argmax_bf16")?;

    let (mut ok, mut rows_cmp, mut cases) = (true, 0usize, 0usize);
    let mut r = Lcg(0x0C1_0012);
    for &n in &[1usize, 3, 16] {
        let mut logits = vec![0u8; n * NV * 2];
        for c in logits.chunks_exact_mut(2) {
            c.copy_from_slice(&r.bf16());
        }
        let d_logits = upload(g, &logits)?;
        for case in ['A', 'B'] {
            let args: Vec<u32> = match case {
                // A: the argmax kernel's answer per row (one launch per row, as `forward_one`).
                'A' => {
                    let d_arg = g.alloc(4)?;
                    let mut out = Vec::with_capacity(n);
                    for i in 0..n {
                        ops::argmax_bf16(
                            g,
                            argmax_k,
                            d_logits.offset(i * NV * 2),
                            d_arg,
                            NV as u32,
                            stream,
                        )?;
                        g.synchronize(stream)?;
                        let mut b = [0u8; 4];
                        g.copy_d2h(d_arg, &mut b)?;
                        out.push(u32::from_le_bytes(b));
                    }
                    out
                }
                // B: chosen indices, both ends included.
                _ => (0..n)
                    .map(|i| match i % 3 {
                        0 => 0,
                        1 => (NV - 1) as u32,
                        _ => (r.next() % NV as u64) as u32,
                    })
                    .collect(),
            };
            let arg_bytes: Vec<u8> = args.iter().flat_map(|a| a.to_le_bytes()).collect();
            let d_arg = upload(g, &arg_bytes)?;
            for rank in 0..2usize {
                for v0 in [0usize, NV] {
                    let d_x = upload(g, &vec![POISON; 16 * n])?;
                    launch_pack(
                        g,
                        pack_k,
                        d_arg,
                        d_logits,
                        d_x,
                        NV as u32,
                        v0 as u32,
                        rank as u32,
                        n as u32,
                        stream,
                    )?;
                    g.synchronize(stream)?;
                    let mut got = vec![0u8; 16 * n];
                    g.copy_d2h(d_x, &mut got)?;
                    cases += 1;
                    for i in 0..n {
                        let l = args[i] as usize;
                        let lb = [logits[(i * NV + l) * 2], logits[(i * NV + l) * 2 + 1]];
                        let want = host_row(rank, lb, v0 + l);
                        rows_cmp += 1;
                        if got[16 * i..16 * i + 16] != want {
                            ok = false;
                            println!(
                                "FAIL case {case} n={n} rank={rank} v0={v0} row {i} arg {l}: \
                                 device {:02x?} host {want:02x?}",
                                &got[16 * i..16 * i + 16]
                            );
                        }
                    }
                }
            }
        }
    }

    // Single-rank host time of the two sequences around one argmax (no all-reduce).
    let d_arg = g.alloc(4)?;
    let d_x = g.alloc(16)?;
    let d_logits = upload(g, &vec![0x3fu8; NV * 2])?;
    let time = |device: bool| -> Result<f64> {
        let mut t = Vec::with_capacity(200);
        for it in 0..220 {
            ops::argmax_bf16(g, argmax_k, d_logits, d_arg, NV as u32, stream)?;
            let t0 = Instant::now();
            let mut pack = [0u8; 16];
            if device {
                launch_pack(g, pack_k, d_arg, d_logits, d_x, NV as u32, 0, 0, 1, stream)?;
            } else {
                g.synchronize(stream)?;
                let mut b = [0u8; 4];
                g.copy_d2h(d_arg, &mut b)?;
                let l = u32::from_le_bytes(b) as usize;
                let mut lb = [0u8; 2];
                g.copy_d2h(d_logits.offset(l * 2), &mut lb)?;
                pack = host_row(0, lb, l);
                g.copy_h2d(&pack, d_x)?;
            }
            g.synchronize(stream)?;
            g.copy_d2h(d_x, &mut pack)?;
            if it >= 20 {
                t.push(t0.elapsed().as_secs_f64() * 1e6);
            }
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(t[t.len() / 2])
    };
    let (host_us, dev_us) = (time(false)?, time(true)?);
    println!(
        "timing (single rank, after argmax, median of 200): host packet {host_us:.1} us, \
         device packet {dev_us:.1} us, saved {:.1} us per draft step",
        host_us - dev_us
    );

    if rows_cmp == 0 {
        println!("FAIL mtp_head_xchg_microtest: compared 0 rows; this run proves nothing");
        std::process::exit(1);
    }
    if !ok {
        println!("FAIL mtp_head_xchg_microtest: device packet differs from the host packer");
        std::process::exit(1);
    }
    println!(
        "PASS: mtp_head_xchg_microtest {cases} cases, {rows_cmp} rows ({} bytes) bit-identical \
         to the host packer",
        rows_cmp * 16
    );
    Ok(())
}
