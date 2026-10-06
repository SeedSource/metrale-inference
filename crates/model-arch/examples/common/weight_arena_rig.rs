// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Synthetic sources and measurements for `glm5next_weight_arena_microtest`: seeded
//! NVFP4 / BF16 routed experts written as deferred-tensor files or a `model.safetensors`, and
//! the host/device/ledger memory snapshot.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{DeferredTensor, WeightDtype};

use super::{H, I, MIB, PROJS};

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n + 8);
        while v.len() < n {
            v.extend_from_slice(&self.next().to_le_bytes());
        }
        v.truncate(n);
        v
    }
}

pub fn expert_name(layer: usize, id: usize, proj: &str, leaf: &str) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.{id}.{proj}.{leaf}")
}

/// `[rows, cols]` of projection `proj` at intermediate width `i`.
pub fn shape(proj: &str, i: usize) -> (usize, usize) {
    if proj == "down_proj" { (H, i) } else { (i, H) }
}

/// A raw file of tensors, each recorded as a deferred locator.
pub struct RawFile {
    path: PathBuf,
    w: std::io::BufWriter<std::fs::File>,
    off: u64,
}
impl RawFile {
    pub fn create(path: PathBuf) -> Result<Self> {
        let f = std::fs::File::create(&path).with_context(|| format!("{}", path.display()))?;
        Ok(Self {
            path,
            w: std::io::BufWriter::with_capacity(8 << 20, f),
            off: 0,
        })
    }
    pub fn put(
        &mut self,
        b: &[u8],
        shape: Vec<usize>,
        dtype: WeightDtype,
    ) -> Result<DeferredTensor> {
        let d = DeferredTensor {
            path: self.path.clone(),
            offset: self.off,
            shape,
            dtype,
        };
        self.w.write_all(b)?;
        self.off += b.len() as u64;
        Ok(d)
    }
    pub fn finish(mut self) -> Result<()> {
        self.w.flush()?;
        self.w.get_ref().sync_all()?;
        Ok(())
    }
}

/// One layer's experts `ids` written to `path` as NVFP4 (U8 packed + E4M3 block scale + F32
/// `weight_scale_2`) or full-width BF16, at I = 2048: `(id, proj, leaf, locator)` per tensor.
/// The caller registers them under any layer index.
pub fn write_source(
    path: PathBuf,
    ids: std::ops::Range<usize>,
    bf16: bool,
) -> Result<Vec<(usize, String, String, DeferredTensor)>> {
    let mut f = RawFile::create(path)?;
    let mut out = Vec::new();
    for id in ids {
        for (pi, p) in PROJS.iter().enumerate() {
            let (rows, cols) = shape(p, I);
            let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (((id as u64) << 8) | pi as u64));
            if bf16 {
                let b: Vec<u8> = (0..rows * cols)
                    .flat_map(|_| {
                        let v = ((rng.next() >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                        half::bf16::from_f32(v).to_le_bytes()
                    })
                    .collect();
                let d = f.put(&b, vec![rows, cols], WeightDtype::BF16)?;
                out.push((id, p.to_string(), "weight".into(), d));
                continue;
            }
            let packed = rng.bytes(rows * cols / 2);
            let d = f.put(&packed, vec![rows, cols / 2], WeightDtype::UInt8)?;
            out.push((id, p.to_string(), "weight".into(), d));
            let scale = rng.bytes(rows * cols / 16);
            let d = f.put(&scale, vec![rows, cols / 16], WeightDtype::FP8E4M3)?;
            out.push((id, p.to_string(), "weight_scale".into(), d));
            let s2 = (1.0f32 + (id * 3 + pi) as f32 / 1024.0).to_le_bytes();
            let d = f.put(&s2, vec![], WeightDtype::FP32)?;
            out.push((id, p.to_string(), "weight_scale_2".into(), d));
        }
    }
    f.finish()?;
    Ok(out)
}

pub fn mem_available() -> Result<u64> {
    let s = std::fs::read_to_string("/proc/meminfo")?;
    let line = s
        .lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .context("no MemAvailable")?;
    let kb: u64 = line
        .split_whitespace()
        .nth(1)
        .context("MemAvailable value")?
        .parse()?;
    Ok(kb * 1024)
}

/// Host MemAvailable, driver free, ledger bytes, ledger count.
#[derive(Clone, Copy)]
pub struct Snap(pub u64, pub usize, pub usize, pub usize);
pub fn snap(g: &dyn GpuBackend) -> Result<Snap> {
    g.synchronize(g.default_stream())?;
    Ok(Snap(
        mem_available()?,
        g.device_free_memory()?,
        g.live_bytes().unwrap_or(0),
        g.live_alloc_count(),
    ))
}

/// What one arm cost: (MemAvailable drop, device-free drop, ledger bytes, ledger allocations).
pub fn cost(a: Snap, b: Snap) -> (f64, f64, usize, usize) {
    (
        (a.0 as f64 - b.0 as f64) / MIB,
        (a.1 as f64 - b.1 as f64) / MIB,
        b.2.saturating_sub(a.2),
        b.3.saturating_sub(a.3),
    )
}

pub fn d2h(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

/// Write `model.safetensors` from `src` (one layer, U8 NVFP4) plus one BF16 tensor the arena
/// must not claim.
pub fn write_safetensors(
    dir: &Path,
    src: &[(usize, String, String, DeferredTensor)],
) -> Result<()> {
    let norm: Vec<u8> = (0..H)
        .flat_map(|k| half::bf16::from_f32(k as f32).to_le_bytes())
        .collect();
    let mut meta = serde_json::Map::new();
    let mut off = 0usize;
    let mut entry = |name: String, dtype: &str, shape: &[usize], n: usize| {
        meta.insert(
            name,
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}),
        );
        off += n;
    };
    for (id, p, leaf, d) in src {
        let dt = match d.dtype {
            WeightDtype::UInt8 => "U8",
            WeightDtype::FP8E4M3 => "F8_E4M3",
            _ => "F32",
        };
        entry(expert_name(0, *id, p, leaf), dt, &d.shape, d.byte_size());
    }
    entry(
        "model.language_model.norm.weight".into(),
        "BF16",
        &[H],
        norm.len(),
    );
    let mut header = serde_json::to_vec(&serde_json::Value::Object(meta))?;
    header.resize(header.len().next_multiple_of(8), b' ');
    let path = dir.join("model.safetensors");
    let mut w = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&path)?);
    w.write_all(&(header.len() as u64).to_le_bytes())?;
    w.write_all(&header)?;
    for (_, _, _, d) in src {
        w.write_all(&d.read_host_bytes()?)?;
    }
    w.write_all(&norm)?;
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(())
}
