// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safetensors header parsing and shard discovery for
//! [`super::FastSafetensorsLoader`].

use crate::weights::WeightDtype;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Index file format: `{ "weight_map": { "tensor_name": "shard_filename" } }`.
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Parsed entry from a safetensor header.
pub(super) struct TensorMeta {
    pub(super) name: String,
    pub(super) dtype: WeightDtype,
    /// True when the shard stores this tensor as IEEE F16. The tensor is
    /// staged as BF16 (`dtype` above) and the copy loop converts the bytes —
    /// same 2-byte element size, different bit layout.
    pub(super) from_f16: bool,
    pub(super) shape: Vec<usize>,
    /// Absolute byte offset in the file where the tensor bytes start.
    pub(super) abs_offset: u64,
    /// Tensor byte length.
    pub(super) len: usize,
}

/// Discover which safetensor files to load. Mirrors the resolution order in
/// `crate::weights::SafetensorsLoader::load`.
///
/// Returns `(shard_files, Some(tensor→shard))` when an index is present and
/// `(shard_files, None)` when we're loading everything in the listed files.
#[allow(clippy::type_complexity)]
pub(super) fn resolve_shards(
    model_dir: &Path,
) -> Result<(Vec<PathBuf>, Option<HashMap<String, String>>)> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let consolidated_index = model_dir.join("consolidated.safetensors.index.json");

    let actual_index = if index_path.exists() {
        Some(index_path)
    } else if consolidated_index.exists() {
        Some(consolidated_index)
    } else {
        None
    };

    if let Some(ip) = actual_index {
        let json = std::fs::read_to_string(&ip)
            .with_context(|| format!("Failed to read {}", ip.display()))?;
        let index: SafetensorsIndex = serde_json::from_str(&json)?;
        let mut shards: Vec<String> = index
            .weight_map
            .values()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        shards.sort();
        let files: Vec<PathBuf> = shards.iter().map(|s| model_dir.join(s)).collect();
        return Ok((files, Some(index.weight_map)));
    }

    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok((vec![single], None));
    }

    let mut shards: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                (n.starts_with("model.safetensors-") || n.starts_with("consolidated-"))
                    && n.ends_with(".safetensors")
            })
        })
        .collect();
    shards.sort();
    if shards.is_empty() {
        bail!(
            "No safetensor files found in {}. Expected model.safetensors*, \
             consolidated.safetensors*, or consolidated-*-of-*.safetensors",
            model_dir.display()
        );
    }
    Ok((shards, None))
}

/// Parse the safetensors header of an open file. Tensor offsets in the header
/// are relative to the start of the data section; we convert to absolute file
/// offsets up-front so downstream callers only need the fd.
pub(super) fn parse_header(file: &mut File) -> Result<Vec<TensorMeta>> {
    let file_len = file.metadata()?.len();
    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf) as usize;
    if header_size > 64 * 1024 * 1024 {
        bail!("safetensor header too large ({} bytes)", header_size);
    }
    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf)?;
    let data_start: u64 = 8 + header_size as u64;

    let json: Value = serde_json::from_slice(&header_buf)?;
    let obj = json
        .as_object()
        .context("invalid safetensor header (not an object)")?;

    let mut out = Vec::with_capacity(obj.len());
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype_str = info["dtype"].as_str().unwrap_or("BF16");
        let (dtype, from_f16) = match dtype_str {
            "F32" => (WeightDtype::FP32, false),
            "BF16" => (WeightDtype::BF16, false),
            // F16 is not store-legal (WeightDtype is closed to store dtypes):
            // stage as BF16 and mark for byte conversion in the copy loop.
            // centml modelopt W4A4 exports ship all unquantized tensors as F16.
            "F16" => (WeightDtype::BF16, true),
            "U8" => (WeightDtype::UInt8, false),
            // I8 is a 1-byte raw container; DeepSeek-V4-Flash-NVFP4 ships its MTP
            // experts' 4-bit-packed weights as I8 (vs U8 for the main layers).
            // Signedness is irrelevant for packed FP4 — the dequant kernel extracts
            // nibbles by bit ops — so treat I8 as raw bytes (UInt8), matching the
            // NVFP4 expert path.
            "I8" => (WeightDtype::UInt8, false),
            "F8_E4M3" => (WeightDtype::FP8E4M3, false),
            "F8_E8M0" => (WeightDtype::FP8E8M0, false),
            // csf_32x2_v1 compressed expert scales (nvidia/GLM-5.3-Flash-FP4-Local):
            // csf32_offsets / csf32_exceptions are U16, csf32_page_bases U32.
            "U16" => (WeightDtype::UInt16, false),
            "U32" => (WeightDtype::UInt32, false),
            "I64" => (WeightDtype::Int64, false),
            other => bail!("Unsupported safetensors dtype '{other}' for tensor {name}"),
        };
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        // Shared with the RDMA manifest builder: a reversed or past-EOF
        // `data_offsets` pair is rejected here rather than wrapping into a
        // `u64::MAX`-ish `len` that becomes a pread window below.
        let span = metrale_core::safetensors::tensor_span(
            name,
            &info["data_offsets"],
            data_start,
            file_len,
        )?;
        out.push(TensorMeta {
            name: name.clone(),
            dtype,
            from_f16,
            shape,
            abs_offset: span.abs_offset,
            len: span.len as usize,
        });
    }
    // Sort by offset so the reader does sequential disk access.
    out.sort_by_key(|t| t.abs_offset);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    /// A minimal safetensors file: `entries` = (name, dtype, shape, byte len), data zero-filled.
    fn write_st(entries: &[(&str, &str, &[usize], usize)]) -> File {
        let mut hdr = serde_json::Map::new();
        let mut off = 0usize;
        for (name, dtype, shape, len) in entries {
            hdr.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + len]}),
            );
            off += len;
        }
        let h = serde_json::to_vec(&Value::Object(hdr)).unwrap();
        let mut f = tempfile::tempfile().unwrap();
        f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&h).unwrap();
        f.write_all(&vec![0u8; off]).unwrap();
        f.seek(SeekFrom::Start(0)).unwrap();
        f
    }

    /// 2026-10-16 (FP4-Local loader, race-prefill): csf_32x2_v1 scale tensors are U16/U32.
    /// Before this the header parser bailed "Unsupported safetensors dtype 'U16'", so no
    /// FP4-Local expert shard opened. Shapes/lens mirror layers.10.mlp.experts.0.gate_proj.
    #[test]
    fn csf_u16_u32_containers_parse_with_their_byte_width() {
        let mut f = write_st(&[
            ("e.gate_proj.csf32_bases", "U8", &[2048], 2048),
            ("e.gate_proj.csf32_offsets", "U16", &[528], 1056),
            ("e.gate_proj.csf32_page_bases", "U32", &[16], 64),
            ("e.gate_proj.csf32_exceptions", "U16", &[1709], 3418),
        ]);
        let t = parse_header(&mut f).unwrap();
        let by = |n: &str| t.iter().find(|m| m.name == n).unwrap();
        for (n, dt, numel) in [
            ("e.gate_proj.csf32_offsets", WeightDtype::UInt16, 528usize),
            ("e.gate_proj.csf32_page_bases", WeightDtype::UInt32, 16),
            ("e.gate_proj.csf32_exceptions", WeightDtype::UInt16, 1709),
        ] {
            let m = by(n);
            assert_eq!(m.dtype, dt, "{n}");
            assert_eq!(m.len, numel * dt.byte_size(), "{n}: header span must equal numel x byte width");
        }
        assert_eq!(WeightDtype::from_safetensors_str("U16").unwrap(), WeightDtype::UInt16);
        assert_eq!(WeightDtype::from_safetensors_str("U32").unwrap(), WeightDtype::UInt32);
    }

    #[test]
    fn unknown_dtype_still_refused() {
        let mut f = write_st(&[("x", "U64", &[1], 8)]);
        let e = match parse_header(&mut f) {
            Ok(_) => panic!("U64 must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("Unsupported safetensors dtype 'U64'"), "{e}");
    }
}
