// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Build script for metrale-gpu-runtime: link flags, cfgs, and the optional CUTLASS / FlashInfer reference objects.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - `metrale_cutlass` is set only when `CUTLASS_HOME` is set, and
//!   `metrale_flashinfer` only when `FLASHINFER_HOME` is set, both with the
//!   `cuda` feature on and `METRALE_SKIP_BUILD` not `1`/`true`.
//! - `metrale_scale` is set when `METRALE_TARGET_HW` starts with `strix`.
//! - 2026-10-03: `metrale_flashkda` is set only when `FLASHKDA_CUTLASS_HOME` is set, under the
//!   same two conditions as `metrale_cutlass`.

fn main() {
    println!("cargo:rerun-if-env-changed=METRALE_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=METRALE_TARGET_HW");
    println!("cargo:rerun-if-env-changed=CUTLASS_HOME");
    println!("cargo:rerun-if-env-changed=FLASHINFER_HOME");
    println!("cargo:rerun-if-env-changed=FLASHKDA_CUTLASS_HOME");
    println!("cargo:rerun-if-env-changed=METRALE_CUDA_ARCH");
    // 2026-09-25: Declared so `#[cfg(...)]` on these names does not trip the
    // `unexpected_cfgs` lint. `metrale_scale` covers both AMD targets, `strix`
    // and `strix-hip`.
    println!("cargo:rustc-check-cfg=cfg(metrale_scale)");
    println!("cargo:rustc-check-cfg=cfg(metrale_cutlass)");
    println!("cargo:rustc-check-cfg=cfg(metrale_flashinfer)");
    println!("cargo:rustc-check-cfg=cfg(metrale_flashkda)");

    // 2026-09-25: Resolved once, before every early return, so the CUTLASS and
    // FlashInfer objects compile for the same architecture.
    let cuda_arch = resolve_cuda_arch();
    if std::env::var("METRALE_TARGET_HW")
        .as_deref()
        .map(|hw| hw.starts_with("strix"))
        .unwrap_or(false)
    {
        println!("cargo:rustc-cfg=metrale_scale");
    }

    if matches!(
        std::env::var("METRALE_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    ) {
        // 2026-09-25: With METRALE_SKIP_BUILD nothing is compiled with nvcc, but
        // the crate's own FFI to libcuda, cuBLASLt and cudart still has to link.
        if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
            println!("cargo:rustc-link-lib=dylib=cuda");
            println!("cargo:rustc-link-lib=dylib=cublasLt");
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
            println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
        }
        return;
    }

    // 2026-09-25: A build without the `cuda` feature links no CUDA library.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    // 2026-09-25: cudart is linked on every CUDA build, not only with the
    // reference objects below: `copy_d2d_2d_async` calls `cudaMemcpy2DAsync`.
    println!("cargo:rustc-link-lib=dylib=cudart");

    if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rustc-link-search=native={cuda_path}/lib64/stubs");
    }
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");

    // 2026-09-25: Both objects are optional. With `CUTLASS_HOME` /
    // `FLASHINFER_HOME` unset no third-party kernel object is built, and the
    // wrappers in `metrale_gpu_runtime::cutlass` and `::flashinfer` return
    // errors. When built, serving reaches them only through opt-in levers such
    // as `METRALE_CUTLASS_GEMM`, `METRALE_HOLO_MOE_GROUPED_CUTLASS` and
    // `METRALE_FLASHINFER_PREFILL`.
    if let Some(cutlass_home) = std::env::var_os("CUTLASS_HOME") {
        build_cutlass_object(std::path::PathBuf::from(cutlass_home), &cuda_arch);
    }

    if let Some(fi_home) = std::env::var_os("FLASHINFER_HOME") {
        build_flashinfer_object(std::path::PathBuf::from(fi_home), &cuda_arch);
    }

    // 2026-10-03: FlashKDA (vendor/flashkda, MIT) is optional the same way: with
    // `FLASHKDA_CUTLASS_HOME` unset nothing is built and `metrale_gpu_runtime::flashkda`
    // reports itself unavailable. Serving reaches it only through
    // `METRALE_GLM_KDA_PREFILL_FLASHKDA=1`.
    if let Some(cutlass_home) = std::env::var_os("FLASHKDA_CUTLASS_HOME") {
        build_flashkda_object(std::path::PathBuf::from(cutlass_home), &cuda_arch);
    }
}

/// 2026-10-03: Compile the vendored FlashKDA launcher (`vendor/flashkda/csrc/smxx/fwd_launch.cu`,
/// unmodified) and our C-ABI wrapper (`cuda/flashkda_kda_fwd.cu`) into the static lib
/// `metrale_flashkda` against the CUTLASS headers at `cutlass_home` (FlashKDA's submodule pin,
/// see `vendor/flashkda/README.md`), and set the `metrale_flashkda` cfg. The nvcc flags are
/// upstream's `setup.py` set (fast math included: upstream's exactness tests assume it) for the
/// target's arch. Panics if nvcc or ar fails.
fn build_flashkda_object(cutlass_home: std::path::PathBuf, arch: &str) {
    use std::process::Command;

    let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set"));
    let lib = out_dir.join("libmetrale_flashkda.a");
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = std::path::Path::new(&cuda_home).join("bin/nvcc");
    let vendor = std::path::PathBuf::from("../../vendor/flashkda/csrc");

    let sources = [
        vendor.join("smxx/fwd_launch.cu"),
        std::path::PathBuf::from("cuda/flashkda_kda_fwd.cu"),
    ];
    for src in &sources {
        println!("cargo:rerun-if-changed={}", src.display());
    }
    for hdr in [
        "fwd.h",
        "smxx/fwd_kernel1.cuh",
        "smxx/fwd_kernel2.cuh",
        "smxx/utils.cuh",
    ] {
        println!("cargo:rerun-if-changed={}", vendor.join(hdr).display());
    }
    println!("cargo:rustc-cfg=metrale_flashkda");

    let mut objects = Vec::new();
    for src in &sources {
        let obj = out_dir.join(format!(
            "flashkda_{}.o",
            src.file_stem()
                .expect("FlashKDA source has a file stem")
                .to_string_lossy()
        ));
        let status = Command::new(&nvcc)
            .arg("-c")
            .arg("-O3")
            .arg("-std=c++17")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .args([
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ])
            .arg(format!("-arch={arch}"))
            .arg(format!("-I{}", cutlass_home.join("include").display()))
            .arg(format!("-I{}", vendor.display()))
            .arg(src)
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("failed to spawn nvcc for FlashKDA");
        assert!(
            status.success(),
            "nvcc failed while building FlashKDA source {}",
            src.display()
        );
        objects.push(obj);
    }

    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for obj in &objects {
        ar.arg(obj);
    }
    let status = ar.status().expect("failed to spawn ar for FlashKDA");
    assert!(status.success(), "ar failed while archiving FlashKDA");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=metrale_flashkda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// 2026-09-25: Compile `cuda/flashinfer_ragged_prefill.cu` into the static lib
/// `metrale_flashinfer` and set the `metrale_flashinfer` cfg. Panics if nvcc or
/// ar fails.
fn build_flashinfer_object(fi_home: std::path::PathBuf, arch: &str) {
    use std::process::Command;

    let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set"));
    let lib = out_dir.join("libmetrale_flashinfer.a");
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = std::path::Path::new(&cuda_home).join("bin/nvcc");

    let src = std::path::PathBuf::from("cuda/flashinfer_ragged_prefill.cu");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rustc-cfg=metrale_flashinfer");

    let cccl = fi_home.join("3rdparty/cccl");
    let obj = out_dir.join("flashinfer_ragged_prefill.o");
    let status = Command::new(&nvcc)
        .arg("-c")
        .arg("-O3")
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg(format!("-arch={arch}"))
        // 2026-09-25: FlashInfer's own CCCL goes on the include path ahead of
        // the toolkit's.
        .arg("-isystem")
        .arg(cccl.join("libcudacxx/include"))
        .arg("-isystem")
        .arg(cccl.join("cub"))
        .arg("-isystem")
        .arg(cccl.join("thrust"))
        .arg(format!("-I{}", fi_home.join("include").display()))
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("failed to spawn nvcc for FlashInfer wrapper");
    assert!(
        status.success(),
        "nvcc failed while building FlashInfer wrapper"
    );

    let status = Command::new("ar")
        .arg("crus")
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("failed to spawn ar for FlashInfer wrapper");
    assert!(
        status.success(),
        "ar failed while archiving FlashInfer wrapper"
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=metrale_flashinfer");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn build_cutlass_object(cutlass_home: std::path::PathBuf, arch: &str) {
    use std::process::Command;

    let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set"));
    let lib = out_dir.join("libmetrale_cutlass.a");
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = std::path::Path::new(&cuda_home).join("bin/nvcc");

    let sources = [
        std::path::PathBuf::from("cuda/cutlass_bf16_gemm.cu"),
        std::path::PathBuf::from("cuda/cutlass_nvfp4_gemm.cu"),
        std::path::PathBuf::from("cuda/cutlass_nvfp4_grouped_gemm.cu"),
        // 2026-10-06: FP8 blockwise W8A8 GEMM (METRALE_GLM_DENSE_FP8_W8A8_CUTLASS_GW).
        std::path::PathBuf::from("cuda/cutlass_fp8_blockwise_gemm.cu"),
    ];
    for src in &sources {
        println!("cargo:rerun-if-changed={}", src.display());
    }
    println!("cargo:rustc-cfg=metrale_cutlass");

    let mut objects = Vec::new();
    for src in &sources {
        let obj = out_dir.join(
            src.file_stem()
                .expect("CUTLASS wrapper source has a file stem")
                .to_string_lossy()
                .to_string()
                + ".o",
        );
        let status = Command::new(&nvcc)
            .arg("-c")
            .arg("-O3")
            .arg("-std=c++17")
            .arg("--expt-relaxed-constexpr")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg(format!("-arch={arch}"))
            .arg(format!("-I{}", cutlass_home.join("include").display()))
            .arg(format!(
                "-I{}",
                cutlass_home.join("tools/util/include").display()
            ))
            .arg(src)
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("failed to spawn nvcc for CUTLASS wrapper");
        assert!(
            status.success(),
            "nvcc failed while building CUTLASS wrapper {}",
            src.display()
        );
        objects.push(obj);
    }

    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for obj in &objects {
        ar.arg(obj);
    }
    let status = ar.status().expect("failed to spawn ar for CUTLASS wrapper");
    assert!(
        status.success(),
        "ar failed while archiving CUTLASS wrapper"
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=metrale_cutlass");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// 2026-09-25: The SM architecture the CUTLASS / FlashInfer reference objects
/// compile for: `METRALE_CUDA_ARCH` when set, else `[hardware].arch` from
/// `kernels/<METRALE_TARGET_HW>/HARDWARE.toml` (target `gb10` when unset).
/// An unreadable file falls back to `sm_121f` with a `cargo:warning`.
fn resolve_cuda_arch() -> String {
    const FALLBACK: &str = "sm_121f";
    if let Ok(explicit) = std::env::var("METRALE_CUDA_ARCH") {
        return explicit;
    }
    let hw = std::env::var("METRALE_TARGET_HW").unwrap_or_else(|_| "gb10".to_string());
    let manifest =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let hardware_toml = manifest
        .parent()
        .and_then(|crates| crates.parent())
        .expect("crates/<crate> sits two levels below the workspace root")
        .join("kernels")
        .join(&hw)
        .join("HARDWARE.toml");
    println!("cargo:rerun-if-changed={}", hardware_toml.display());
    match hardware_arch(&hardware_toml) {
        Some(arch) => arch,
        None => {
            println!(
                "cargo:warning=metrale-gpu-runtime: no [hardware].arch in {} — reference objects fall \
                 back to {FALLBACK}; set METRALE_TARGET_HW to a target under kernels/, or \
                 METRALE_CUDA_ARCH to override",
                hardware_toml.display()
            );
            FALLBACK.to_string()
        }
    }
}

/// 2026-09-25: `[hardware].arch` from a `HARDWARE.toml`, or `None` if it cannot be read.
fn hardware_arch(path: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let doc: toml::Value = toml::from_str(&text).ok()?;
    Some(doc.get("hardware")?.get("arch")?.as_str()?.to_string())
}
