// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Tests for `METRALE_DFLASH_CKPT_ARCH`: the `torch.save` mask
//! embedding reader and the config fields it resolves.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use super::*;
use crate::weight_loader::dflash_loader::parse_dflash_config;

/// 2026-10-01: A zip archive laid out like PyTorch's writer: data-descriptor
/// flag set, zero sizes in every local header, padding in the local extra
/// field, real sizes only in the central directory.
fn torch_zip(members: &[(&str, &[u8])], method: u16) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cd = Vec::new();
    for (name, data) in members {
        let lho = out.len() as u32;
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&0x0008u16.to_le_bytes());
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(&[0u8; 12]);
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&3u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&[0u8; 3]);
        out.extend_from_slice(data);

        cd.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        cd.extend_from_slice(&20u16.to_le_bytes());
        cd.extend_from_slice(&20u16.to_le_bytes());
        cd.extend_from_slice(&0x0008u16.to_le_bytes());
        cd.extend_from_slice(&method.to_le_bytes());
        cd.extend_from_slice(&[0u8; 4]);
        cd.extend_from_slice(&0u32.to_le_bytes());
        cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
        cd.extend_from_slice(&[0u8; 12]);
        cd.extend_from_slice(&lho.to_le_bytes());
        cd.extend_from_slice(name.as_bytes());
    }
    let cd_off = out.len() as u32;
    let cd_len = cd.len() as u32;
    out.extend_from_slice(&cd);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0u8; 4]);
    out.extend_from_slice(&(members.len() as u16).to_le_bytes());
    out.extend_from_slice(&(members.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_len.to_le_bytes());
    out.extend_from_slice(&cd_off.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

fn pkl(storage: &str) -> Vec<u8> {
    format!("\u{80}\u{2}mask_token_id embedding torch\n{storage}\n0 cpu").into_bytes()
}

#[test]
fn bf16_mask_embedding_is_returned_byte_for_byte() {
    let data: Vec<u8> = (0..8u16).flat_map(|i| (0x3f80 + i).to_le_bytes()).collect();
    let z = torch_zip(
        &[
            ("mask_embedding/data.pkl", &pkl("BFloat16Storage")[..]),
            ("mask_embedding/byteorder", &b"little"[..]),
            ("mask_embedding/data/0", &data[..]),
            ("mask_embedding/version", &b"3\n"[..]),
        ],
        0,
    );
    let out = parse_torch_mask_embedding(&z, 8).expect("parse");
    assert_eq!(out, data);
}

#[test]
fn f32_mask_embedding_is_rounded_to_bf16() {
    let vals = [1.0f32, -2.5, 0.1, 3.0e-3];
    let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let z = torch_zip(
        &[
            ("m/data.pkl", &pkl("FloatStorage")[..]),
            ("m/data/0", &data[..]),
        ],
        0,
    );
    let out = parse_torch_mask_embedding(&z, 4).expect("parse");
    let want: Vec<u8> = vals
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    assert_eq!(out, want);
}

#[test]
fn wrong_element_count_is_refused() {
    let data = [0u8; 16];
    let z = torch_zip(
        &[
            ("m/data.pkl", &pkl("BFloat16Storage")[..]),
            ("m/data/0", &data[..]),
        ],
        0,
    );
    assert!(parse_torch_mask_embedding(&z, 4096).is_err());
}

#[test]
fn compressed_member_is_refused() {
    let data = [0u8; 8];
    let z = torch_zip(
        &[
            ("m/data.pkl", &pkl("BFloat16Storage")[..]),
            ("m/data/0", &data[..]),
        ],
        8,
    );
    assert!(parse_torch_mask_embedding(&z, 4).is_err());
}

#[test]
fn big_endian_archive_is_refused() {
    let data = [0u8; 8];
    let z = torch_zip(
        &[
            ("m/data.pkl", &pkl("BFloat16Storage")[..]),
            ("m/byteorder", &b"big"[..]),
            ("m/data/0", &data[..]),
        ],
        0,
    );
    assert!(parse_torch_mask_embedding(&z, 4).is_err());
}

#[test]
fn not_a_zip_is_refused() {
    assert!(parse_torch_mask_embedding(&[0u8; 64], 4).is_err());
    assert!(parse_torch_mask_embedding(&[], 4).is_err());
}

/// 2026-10-01: The fields of `canada-quant/GLM-5.3-Flash-DFlash2-G@bd03d3a3`
/// `config.json` this lever reads (trimmed; unknown keys kept to show they
/// are ignored).
const G_CONFIG: &str = r#"{
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 8,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-05,
    "vocab_size": 154880,
    "tie_word_embeddings": false,
    "layer_types": ["full_attention","full_attention","full_attention","full_attention",
                    "full_attention","full_attention","full_attention","full_attention"],
    "sliding_window": null,
    "use_sliding_window": false,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "mask_token_id": 154856,
    "dflash_config": {"block_size": 8, "mask_token_id": 154856,
        "target_layer_ids": [5, 9, 14, 19, 24, 28, 33, 38, 42],
        "conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": 256, "selector_top_k": 16},
    "export": {"tool": "angelspec-glm53/dflash2/export_dflash2.py", "ships_embed_tokens": true,
        "ships_lm_head": true, "ships_mask_embedding": true,
        "mask_embedding_mechanism": "mask_embedding.pt"}
}"#;

#[test]
fn g_config_resolves_nested_theta_and_eps_only_under_the_lever() {
    let c = parse_dflash_config(G_CONFIG).expect("parse G config");
    assert_eq!(c.rope_theta, 10_000_000.0, "top-level default is unchanged");
    assert_eq!(c.nested_rope_theta(), Some(10_000.0));
    assert_eq!(c.resolved_rope_theta(false), 10_000_000.0);
    assert_eq!(c.resolved_rope_theta(true), 10_000.0);
    assert_eq!(c.resolved_rms_norm_eps(false), 1e-6);
    assert_eq!(c.resolved_rms_norm_eps(true), 1e-5);
    assert!(c.needs_mask_embedding());
    assert!(!c.use_sliding_window);
    assert_eq!(c.layer_types.len(), 8);
    assert!(
        c.mask_embedding_bf16.is_none(),
        "never read from config.json"
    );
    let sub = c.dflash_config.as_ref().expect("dflash_config");
    assert_eq!(sub.target_layer_ids.len(), 9);
    assert_eq!(c.effective_block_size(), 8);
}

#[test]
fn top_level_theta_wins_when_no_nested_theta() {
    let c = parse_dflash_config(
        r#"{"hidden_size": 64, "num_hidden_layers": 1, "intermediate_size": 128,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "vocab_size": 256, "rope_theta": 1000000.0}"#,
    )
    .expect("parse");
    assert_eq!(c.resolved_rope_theta(true), 1_000_000.0);
    assert_eq!(c.resolved_rms_norm_eps(true), 1e-6);
    assert!(!c.needs_mask_embedding());
}

fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let d = std::env::temp_dir().join(format!(
        "metrale-ckpt-arch-{tag}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&d).expect("create scratch dir");
    d
}

#[test]
fn missing_mask_file_fails_only_when_the_export_requires_it() {
    let dir = scratch_dir("missing");
    let mut g = parse_dflash_config(G_CONFIG).expect("parse");
    assert!(attach_mask_embedding(&dir, &mut g).is_err());

    let mut plain = parse_dflash_config(
        r#"{"hidden_size": 4, "num_hidden_layers": 1, "intermediate_size": 8,
            "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 4,
            "vocab_size": 16}"#,
    )
    .expect("parse");
    assert!(!attach_mask_embedding(&dir, &mut plain).expect("absent is fine"));
    assert!(plain.mask_embedding_bf16.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn present_mask_file_is_attached() {
    let dir = scratch_dir("present");
    let data: Vec<u8> = (0..4u16).flat_map(|i| (0x3f80 + i).to_le_bytes()).collect();
    let z = torch_zip(
        &[
            ("m/data.pkl", &pkl("BFloat16Storage")[..]),
            ("m/data/0", &data[..]),
        ],
        0,
    );
    std::fs::write(dir.join(MASK_EMBEDDING_FILE), &z).expect("write");
    let mut c = parse_dflash_config(
        r#"{"hidden_size": 4, "num_hidden_layers": 1, "intermediate_size": 8,
            "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 4,
            "vocab_size": 16}"#,
    )
    .expect("parse");
    assert!(attach_mask_embedding(&dir, &mut c).expect("attach"));
    assert_eq!(c.mask_embedding_bf16.as_deref(), Some(data.as_slice()));
    let _ = std::fs::remove_dir_all(&dir);
}
