// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: `METRALE_DFLASH_CKPT_ARCH=1` — serve a DFlash2 drafter with the
//! architecture its checkpoint states instead of the engine's built-in
//! assumptions. Default off; with it unset nothing here changes the head.
//!
//! Under the lever:
//! - the learned mask embedding (`mask_embedding.pt`, a PyTorch zip archive
//!   holding `{"mask_token_id": int, "embedding": bf16[hidden]}`) is loaded
//!   from the drafter directory ([`attach_mask_embedding`]) and the head
//!   writes it over every mask row of the draft block, in place of the
//!   target's `embed_tokens[mask_token_id]`;
//! - a drafter whose `export.ships_mask_embedding` is true and whose
//!   `mask_embedding.pt` is missing or unreadable fails the load (the
//!   exporter's own rule: a silently ignored mask is "do not serve");
//! - RoPE θ comes from `rope_parameters.rope_theta` when the config nests it
//!   ([`DflashConfig::resolved_rope_theta`]);
//! - the RMSNorm epsilon comes from `rms_norm_eps`
//!   ([`DflashConfig::resolved_rms_norm_eps`]).
//!
//! Why (measured 2026-10-01 from the public checkpoints, range reads of the
//! safetensors): `canada-quant/GLM-5.3-Flash-DFlash2-G@bd03d3a3` ships
//! `embed_tokens` and `lm_head` byte-identical to `nvidia/GLM-5.3-Flash-NVFP4`
//! (rows 1000, 50000, 154856 compared, max diff 0), so sharing the target's
//! tables, as the head already does, is exact. Its `embed_tokens[154856]`
//! (the mask token) has L2 norm 0.0005, while `mask_embedding.pt` holds a
//! vector of norm 1.18 (cosine −0.0003 to the table row). An engine that
//! ignores the file feeds the drafter seven near-zero mask rows it was never
//! trained on. Both GLM-5.3 DFlash2 configs (incoai and G) nest
//! `rope_theta: 10000.0` in `rope_parameters` with no top-level key, so the
//! loader's top-level default (10,000,000) applies to them today, and both
//! state `rms_norm_eps: 1e-05` against the head's built-in 1e-6.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - [`parse_torch_mask_embedding`] returns exactly `hidden * 2` bytes of
//!   BF16 little-endian, or an error; it never returns a partial vector.
//! - Only stored (method 0) zip members are read; a compressed archive is an
//!   error, not a guess.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

use super::DflashConfig;

/// 2026-10-01: The file a canada-quant DFlash2 export writes its learned mask
/// embedding to, next to `model.safetensors`.
pub const MASK_EMBEDDING_FILE: &str = "mask_embedding.pt";

/// 2026-10-01: True when `METRALE_DFLASH_CKPT_ARCH=1`. Any other value,
/// including unset, is off. Read at drafter load, once per call site.
pub fn ckpt_arch_enabled() -> bool {
    std::env::var("METRALE_DFLASH_CKPT_ARCH").ok().as_deref() == Some("1")
}

/// 2026-10-01: Under the lever, load `dir/mask_embedding.pt` into
/// `cfg.mask_embedding_bf16`. Returns whether a mask embedding was attached.
///
/// # Errors
/// The file exists but does not parse as a `hidden_size` mask embedding, or
/// the config states `export.ships_mask_embedding` and the file is absent.
pub fn attach_mask_embedding(dir: &Path, cfg: &mut DflashConfig) -> Result<bool> {
    let path = dir.join(MASK_EMBEDDING_FILE);
    if !path.exists() {
        if cfg.needs_mask_embedding() {
            bail!(
                "METRALE_DFLASH_CKPT_ARCH=1: the drafter config states \
                 export.ships_mask_embedding=true but {} is missing; serving without it \
                 drafts from near-zero mask rows (the exporter's rule: do not serve)",
                path.display()
            );
        }
        return Ok(false);
    }
    let bytes =
        std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let emb = parse_torch_mask_embedding(&bytes, cfg.hidden_size)
        .with_context(|| format!("parse {}", path.display()))?;
    let norm = bf16_l2_norm(&emb);
    tracing::info!(
        "DFlash ckpt-arch: loaded mask embedding for mask_token_id {} from {} \
         ({} BF16 values, L2 norm {norm:.4})",
        cfg.dflash_config
            .as_ref()
            .map(|c| c.mask_token_id)
            .unwrap_or(0),
        path.display(),
        emb.len() / 2,
    );
    cfg.mask_embedding_bf16 = Some(emb);
    Ok(true)
}

/// 2026-10-01: One log line per drafter load naming the architecture the
/// config states and what the engine does with it, plus a warning for each
/// field the engine is about to ignore. With `enabled` false it only warns.
pub fn log_arch_summary(cfg: &DflashConfig, enabled: bool) {
    let sliding = cfg.use_sliding_window
        || cfg.layer_types.iter().any(|t| t == "sliding_attention");
    tracing::info!(
        "DFlash ckpt-arch ({}): {} layers, attention={}, rope_theta used={} \
         (top-level {}, nested {:?}), rms_norm_eps used={:e} (config {:?}), \
         mask_embedding={}",
        if enabled { "METRALE_DFLASH_CKPT_ARCH=1" } else { "off" },
        cfg.num_hidden_layers,
        if sliding {
            format!("sliding {:?}", cfg.sliding_window)
        } else {
            "full".to_string()
        },
        cfg.resolved_rope_theta(enabled),
        cfg.rope_theta,
        cfg.nested_rope_theta(),
        cfg.resolved_rms_norm_eps(enabled),
        cfg.rms_norm_eps,
        if cfg.mask_embedding_bf16.is_some() {
            "learned (mask_embedding.pt)"
        } else {
            "target embed_tokens[mask_token_id]"
        },
    );
    if !enabled {
        if let Some(t) = cfg.nested_rope_theta()
            && t != cfg.rope_theta
        {
            tracing::warn!(
                "DFlash drafter config nests rope_theta={t} but the head uses the top-level \
                 value {} (default when absent); METRALE_DFLASH_CKPT_ARCH=1 uses the nested one",
                cfg.rope_theta
            );
        }
        if cfg.needs_mask_embedding() {
            tracing::warn!(
                "DFlash drafter was exported with a learned mask embedding \
                 ({MASK_EMBEDDING_FILE}); without METRALE_DFLASH_CKPT_ARCH=1 the mask rows \
                 embed the target's table row and acceptance will collapse"
            );
        }
    }
    if sliding {
        tracing::warn!(
            "DFlash drafter declares sliding-window attention ({:?}); the paged drafter \
             attention attends the whole captured context, so long-context drafts differ \
             from the trained model",
            cfg.sliding_window
        );
    }
}

/// 2026-10-01: Parse a `torch.save`d mask embedding: a zip archive whose
/// `<root>/data.pkl` names a `BFloat16Storage`, `HalfStorage` or
/// `FloatStorage` and whose `<root>/data/0` holds `hidden` elements of it.
/// Returns BF16 little-endian bytes (F16 and F32 are rounded to BF16, nearest
/// even). A `<root>/byteorder` member other than `little` is an error.
///
/// # Errors
/// Not a stored zip archive, no `data.pkl` or `data/0` member, an unknown
/// storage type, or an element count other than `hidden`.
pub fn parse_torch_mask_embedding(bytes: &[u8], hidden: usize) -> Result<Vec<u8>> {
    let entries = stored_zip_entries(bytes)?;
    let find = |suffix: &str| -> Option<&[u8]> {
        entries
            .iter()
            .find(|(n, _)| n.ends_with(suffix))
            .map(|(_, d)| *d)
    };
    let pkl = find("/data.pkl").context("mask embedding archive has no data.pkl")?;
    if let Some(order) = find("/byteorder") {
        ensure!(
            order == b"little",
            "mask embedding byteorder is {:?}, want \"little\"",
            String::from_utf8_lossy(order)
        );
    }
    let data = find("/data/0").context("mask embedding archive has no data/0 storage")?;
    #[derive(Clone, Copy)]
    enum Storage {
        Bf16,
        F16,
        F32,
    }
    let contains = |needle: &[u8]| pkl.windows(needle.len()).any(|w| w == needle);
    let storage = if contains(b"BFloat16Storage") {
        Storage::Bf16
    } else if contains(b"HalfStorage") {
        Storage::F16
    } else if contains(b"FloatStorage") {
        Storage::F32
    } else {
        bail!("mask embedding storage type is not BFloat16, Half or Float");
    };
    let elem_bytes = match storage {
        Storage::Bf16 | Storage::F16 => 2,
        Storage::F32 => 4,
    };
    ensure!(
        data.len() == hidden * elem_bytes,
        "mask embedding storage is {} bytes, want {} ({} elements of {} bytes)",
        data.len(),
        hidden * elem_bytes,
        hidden,
        elem_bytes
    );
    let out: Vec<u8> = match storage {
        Storage::Bf16 => data.to_vec(),
        Storage::F16 => data
            .chunks_exact(2)
            .flat_map(|c| {
                let f = half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                half::bf16::from_f32(f).to_bits().to_le_bytes()
            })
            .collect(),
        Storage::F32 => data
            .chunks_exact(4)
            .flat_map(|c| {
                let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                half::bf16::from_f32(f).to_bits().to_le_bytes()
            })
            .collect(),
    };
    Ok(out)
}

/// 2026-10-01: L2 norm of BF16 little-endian bytes, for the load log.
pub fn bf16_l2_norm(bytes: &[u8]) -> f32 {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let v = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
            v * v
        })
        .sum::<f32>()
        .sqrt()
}

fn rd_u16(b: &[u8], o: usize) -> Result<u16> {
    let s = b.get(o..o + 2).context("zip: truncated u16")?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn rd_u32(b: &[u8], o: usize) -> Result<u32> {
    let s = b.get(o..o + 4).context("zip: truncated u32")?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// 2026-10-01: The members of a zip archive as `(name, data)`, read through
/// the central directory (PyTorch's writer sets the data-descriptor flag, so
/// local headers carry zero sizes). Only stored members; ZIP64 offsets and
/// compressed members are errors.
fn stored_zip_entries(b: &[u8]) -> Result<Vec<(String, &[u8])>> {
    const EOCD: u32 = 0x0605_4b50;
    const CDH: u32 = 0x0201_4b50;
    const LFH: u32 = 0x0403_4b50;
    ensure!(b.len() >= 22, "zip: {} bytes is shorter than an EOCD record", b.len());
    let lo = b.len().saturating_sub(22 + 0xFFFF);
    let mut eocd = None;
    let mut o = b.len() - 22;
    loop {
        if rd_u32(b, o)? == EOCD {
            eocd = Some(o);
            break;
        }
        if o == lo {
            break;
        }
        o -= 1;
    }
    let e = eocd.context("zip: no end-of-central-directory record (not a torch.save archive?)")?;
    let n = rd_u16(b, e + 10)? as usize;
    let cd_off = rd_u32(b, e + 16)?;
    ensure!(
        cd_off != u32::MAX && n != usize::from(u16::MAX),
        "zip: ZIP64 central directory is not supported"
    );
    let mut p = cd_off as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        ensure!(rd_u32(b, p)? == CDH, "zip: bad central directory header at {p}");
        let method = rd_u16(b, p + 10)?;
        let csize = rd_u32(b, p + 20)? as usize;
        let usize_ = rd_u32(b, p + 24)? as usize;
        let nlen = rd_u16(b, p + 28)? as usize;
        let elen = rd_u16(b, p + 30)? as usize;
        let clen = rd_u16(b, p + 32)? as usize;
        let lho = rd_u32(b, p + 42)? as usize;
        let name_bytes = b.get(p + 46..p + 46 + nlen).context("zip: truncated name")?;
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        p += 46 + nlen + elen + clen;
        ensure!(
            method == 0 && csize == usize_,
            "zip: member {name} is compressed (method {method}); only stored members are read"
        );
        ensure!(rd_u32(b, lho)? == LFH, "zip: bad local header for {name}");
        let start = lho + 30 + rd_u16(b, lho + 26)? as usize + rd_u16(b, lho + 28)? as usize;
        let data = b
            .get(start..start + csize)
            .with_context(|| format!("zip: member {name} runs past the archive"))?;
        out.push((name, data));
    }
    Ok(out)
}

#[cfg(test)]
#[path = "ckpt_arch_tests.rs"]
mod tests;
