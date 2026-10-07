// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: cuBLASLt FFI for `out[M,N] = act[M,K] @ weight[N,K]ᵀ`: BF16 GEMMs
//! with BF16 or FP32 output here, FP8 E4M3 GEMMs in `fp8`, and the FP8
//! scale-tensor layout math in [`scale_layout`].
//!
//! Owner: gpu-runtime (cuBLASLt wrapper).
//! Invariants:
//! - `CTX` holds a single cuBLASLt handle and 64 MiB workspace for the
//!   process, allocated with `cuMemAlloc_v2`, outside the backend's
//!   allocation ledger.
//! - Every GEMM here passes the whole workspace to `cublasLtMatmul`, so two
//!   GEMMs must not run concurrently.

use anyhow::{Result, bail};
use std::ffi::c_void;
use std::sync::OnceLock;

mod fp8;
mod pin;
pub use fp8::{
    fp8_gemm_act_weight_t_blkscaled, fp8_gemm_act_weight_t_blkscaled_ldc,
    fp8_gemm_act_weight_t_rowwise,
};
pub use pin::{BF16_PIN_M, bf16_pin_describe, bf16_pin_fallbacks};

pub mod scale_layout;

#[allow(non_camel_case_types)]
type cublasLtHandle_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulDesc_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatrixLayout_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulPreference_t = *mut c_void;

const CUDA_R_16BF: i32 = 14;
const CUDA_R_32F: i32 = 0;
const CUDA_R_8F_E4M3: i32 = 28;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const DESC_TRANSA: u32 = 3;
const DESC_TRANSB: u32 = 4;
const DESC_A_SCALE_POINTER: u32 = 17;
const DESC_B_SCALE_POINTER: u32 = 18;
const DESC_A_SCALE_MODE: u32 = 31;
const DESC_B_SCALE_MODE: u32 = 32;
const SCALE_MODE_OUTER_VEC_32F: i32 = 3;
const SCALE_MODE_VEC128_32F: i32 = 4;
const SCALE_MODE_BLK128X128_32F: i32 = 5;
const PREF_MAX_WORKSPACE_BYTES: u32 = 1;

unsafe extern "C" {
    fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> i32;
    fn cublasLtMatmulDescCreate(
        desc: *mut cublasLtMatmulDesc_t,
        compute_type: i32,
        scale_type: i32,
    ) -> i32;
    fn cublasLtMatmulDescSetAttribute(
        desc: cublasLtMatmulDesc_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulDescDestroy(desc: cublasLtMatmulDesc_t) -> i32;
    fn cublasLtMatrixLayoutCreate(
        layout: *mut cublasLtMatrixLayout_t,
        dtype: i32,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> i32;
    fn cublasLtMatrixLayoutDestroy(layout: cublasLtMatrixLayout_t) -> i32;
    fn cublasLtMatmulPreferenceCreate(pref: *mut cublasLtMatmulPreference_t) -> i32;
    fn cublasLtMatmulPreferenceSetAttribute(
        pref: cublasLtMatmulPreference_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoCheck(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        algo: *const c_void,
        result: *mut c_void,
    ) -> i32;
    fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const c_void,
        attr: u32,
        buf: *mut c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoGetHeuristic(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        pref: cublasLtMatmulPreference_t,
        requested: i32,
        results: *mut c_void,
        returned: *mut i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmul(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        alpha: *const c_void,
        a: *const c_void,
        layout_a: cublasLtMatrixLayout_t,
        b: *const c_void,
        layout_b: cublasLtMatrixLayout_t,
        beta: *const c_void,
        c: *const c_void,
        layout_c: cublasLtMatrixLayout_t,
        d: *mut c_void,
        layout_d: cublasLtMatrixLayout_t,
        algo: *const c_void,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
}

struct Ctx {
    handle: cublasLtHandle_t,
    workspace: u64,
    ws_size: usize,
}
// 2026-09-25: The handle and workspace are process-global and shared by
// every caller; see the module invariants.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

/// 2026-09-25: The process-wide cuBLASLt handle and workspace, created on
/// first use by `ctx`. Its size is fixed, not derived from a model, so it
/// stays valid across a model swap, like the process CUDA context
/// (`crate::cuda_host`).
static CTX: OnceLock<Ctx> = OnceLock::new();

fn ctx() -> Result<&'static Ctx> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    let mut handle: cublasLtHandle_t = std::ptr::null_mut();
    let st = unsafe { cublasLtCreate(&mut handle) };
    if st != 0 {
        bail!("cublasLtCreate failed: {st}");
    }
    let ws_size = 64 * 1024 * 1024;
    let mut ws: u64 = 0;
    let st = unsafe { cuMemAlloc_v2(&mut ws, ws_size) };
    if st != 0 {
        bail!("cuMemAlloc cuBLASLt workspace failed: {st}");
    }
    let _ = CTX.set(Ctx {
        handle,
        workspace: ws,
        ws_size,
    });
    Ok(CTX.get().unwrap())
}

/// 2026-09-25: Run one 64x64x64 BF16 GEMM on `stream` at model load
/// (`serve_load.rs`), so the first cuBLASLt call, including `ctx`'s
/// `cublasLtCreate` and workspace allocation, happens before the first
/// request.
///
/// Returns nothing: a failure is logged as a warning, and the first GEMM of
/// a request then pays the same costs.
pub fn prewarm(stream: u64) {
    let r = (|| -> Result<()> {
        let bytes = 64usize * 64 * 2;
        let mut a = 0u64;
        let mut b = 0u64;
        let mut d = 0u64;
        unsafe {
            chk(cuMemAlloc_v2(&mut a, bytes), "prewarm alloc a")?;
            chk(cuMemAlloc_v2(&mut b, bytes), "prewarm alloc b")?;
            chk(cuMemAlloc_v2(&mut d, bytes), "prewarm alloc d")?;
        }
        let res = bf16_gemm_act_weight_t(a, b, d, 64, 64, 64, stream);
        unsafe {
            chk(cuStreamSynchronize(stream), "prewarm sync")?;
            let _ = cuMemFree_v2(a);
            let _ = cuMemFree_v2(b);
            let _ = cuMemFree_v2(d);
        }
        res
    })();
    match r {
        Ok(()) => tracing::info!("cuBLASLt pre-warmed (handle + workspace + kernel images)"),
        Err(e) => tracing::warn!("cuBLASLt pre-warm failed (request 1 pays lazy init): {e}"),
    }
}

fn chk(status: i32, what: &str) -> Result<()> {
    if status != 0 {
        bail!("cuBLASLt {what} failed: status {status}");
    }
    Ok(())
}

/// 2026-09-25: Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, all BF16, FP32
/// accumulation. In cuBLASLt's column-major terms:
/// `D[N,M] = opT(weightᶜ[K,N]) · opN(actᶜ[K,M])`.
pub fn bf16_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let pin = pin::enabled().then_some(BF16_PIN_M);
    gemm_act_weight_t_out(act, weight, out, m, n, k, CUDA_R_16BF, pin, stream)
}

/// 2026-09-25: [`bf16_gemm_act_weight_t`] with an FP32 `out`. Only the D
/// layout's type differs; both use `CUBLAS_COMPUTE_32F`.
pub fn bf16_gemm_act_weight_t_f32_out(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let pin = pin::enabled().then_some(BF16_PIN_M);
    gemm_act_weight_t_out(act, weight, out, m, n, k, CUDA_R_32F, pin, stream)
}

/// 2026-10-07: [`bf16_gemm_act_weight_t`] / [`bf16_gemm_act_weight_t_f32_out`]
/// (`out_f32`) with the algorithm choice explicit instead of read from
/// `METRALE_CUBLAS_BF16_ALGO_PIN`: `pin_m = None` asks the heuristic at this
/// call's M (the default path); `Some(p)` runs the algorithm pinned for
/// (N, K, out type) at M = `p` (see `pin`). For microtests.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t_with(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    out_f32: bool,
    pin_m: Option<u32>,
    stream: u64,
) -> Result<()> {
    let dtype = if out_f32 { CUDA_R_32F } else { CUDA_R_16BF };
    gemm_act_weight_t_out(act, weight, out, m, n, k, dtype, pin_m, stream)
}

/// 2026-10-07: The descriptors of one BF16 `act @ weightᵀ` GEMM, destroyed on
/// drop. A = weight, row-major [N,K] = col-major [K,N], ld K, opT. B = act,
/// row-major [M,K] = col-major [K,M], ld K, opN. D = out, row-major [M,N] =
/// col-major [N,M], ld N. The preference caps the workspace at `ws_size`.
struct Descs {
    desc: cublasLtMatmulDesc_t,
    la: cublasLtMatrixLayout_t,
    lb: cublasLtMatrixLayout_t,
    ld: cublasLtMatrixLayout_t,
    pref: cublasLtMatmulPreference_t,
}

impl Descs {
    fn new(m: u32, n: u32, k: u32, out_dtype: i32, ws_size: usize) -> Result<Self> {
        let mut d = Descs {
            desc: std::ptr::null_mut(),
            la: std::ptr::null_mut(),
            lb: std::ptr::null_mut(),
            ld: std::ptr::null_mut(),
            pref: std::ptr::null_mut(),
        };
        unsafe {
            chk(
                cublasLtMatmulDescCreate(&mut d.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                "DescCreate",
            )?;
            let ta = CUBLAS_OP_T;
            let tb = CUBLAS_OP_N;
            chk(
                cublasLtMatmulDescSetAttribute(
                    d.desc,
                    DESC_TRANSA,
                    &ta as *const i32 as *const c_void,
                    4,
                ),
                "TRANSA",
            )?;
            chk(
                cublasLtMatmulDescSetAttribute(
                    d.desc,
                    DESC_TRANSB,
                    &tb as *const i32 as *const c_void,
                    4,
                ),
                "TRANSB",
            )?;
            chk(
                cublasLtMatrixLayoutCreate(&mut d.la, CUDA_R_16BF, k as u64, n as u64, k as i64),
                "LayoutA",
            )?;
            chk(
                cublasLtMatrixLayoutCreate(&mut d.lb, CUDA_R_16BF, k as u64, m as u64, k as i64),
                "LayoutB",
            )?;
            chk(
                cublasLtMatrixLayoutCreate(&mut d.ld, out_dtype, n as u64, m as u64, n as i64),
                "LayoutD",
            )?;
            chk(cublasLtMatmulPreferenceCreate(&mut d.pref), "PrefCreate")?;
            chk(
                cublasLtMatmulPreferenceSetAttribute(
                    d.pref,
                    PREF_MAX_WORKSPACE_BYTES,
                    &ws_size as *const usize as *const c_void,
                    std::mem::size_of::<usize>(),
                ),
                "PrefWorkspace",
            )?;
        }
        Ok(d)
    }

    /// 2026-10-07: Up to `out.len()` heuristic results, best first; returns
    /// how many cuBLASLt filled.
    fn heuristic(&self, ctx: &Ctx, out: &mut [HeuristicResult]) -> Result<usize> {
        let mut returned: i32 = 0;
        unsafe {
            chk(
                cublasLtMatmulAlgoGetHeuristic(
                    ctx.handle,
                    self.desc,
                    self.la,
                    self.lb,
                    self.ld,
                    self.ld,
                    self.pref,
                    out.len() as i32,
                    out.as_mut_ptr() as *mut c_void,
                    &mut returned,
                ),
                "AlgoGetHeuristic",
            )?;
        }
        Ok(returned.max(0) as usize)
    }
}

impl Drop for Descs {
    fn drop(&mut self) {
        unsafe {
            if !self.pref.is_null() {
                cublasLtMatmulPreferenceDestroy(self.pref);
            }
            for l in [self.la, self.lb, self.ld] {
                if !l.is_null() {
                    cublasLtMatrixLayoutDestroy(l);
                }
            }
            if !self.desc.is_null() {
                cublasLtMatmulDescDestroy(self.desc);
            }
        }
    }
}

/// 2026-10-07: `cublasLtMatmulHeuristicResult_t` as the CUDA 13 cublasLt.h lays
/// it out (96 bytes): the 64-byte `cublasLtMatmulAlgo_t`, then workspaceSize,
/// state, wavesCount, reserved[4].
#[repr(C)]
#[derive(Clone, Copy)]
struct HeuristicResult {
    algo: [u64; 8],
    workspace_size: usize,
    state: i32,
    waves_count: f32,
    reserved: [i32; 4],
}

impl HeuristicResult {
    const ZERO: Self = HeuristicResult {
        algo: [0; 8],
        workspace_size: 0,
        state: 0,
        waves_count: 0.0,
        reserved: [0; 4],
    };
}

/// 2026-09-25: Shared body of the wrappers above; `out_dtype` is the D
/// layout's type. 2026-10-07: with `pin_m = Some(p)` it runs the algorithm
/// `pin` chose for (N, K, `out_dtype`) at M = `p` when `cublasLtMatmulAlgoCheck`
/// accepts it at this M, and asks the heuristic at this M otherwise (logged).
#[allow(clippy::too_many_arguments)]
fn gemm_act_weight_t_out(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    out_dtype: i32,
    pin_m: Option<u32>,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    let d = Descs::new(m, n, k, out_dtype, ctx.ws_size)?;
    let mut result = [HeuristicResult::ZERO];
    let pinned = match pin_m {
        Some(p) => pin::algo(ctx, p, n, k, out_dtype)?,
        None => None,
    };
    let use_pinned = match pinned {
        Some(algo) => {
            result[0].algo = algo;
            let st = unsafe {
                cublasLtMatmulAlgoCheck(
                    ctx.handle,
                    d.desc,
                    d.la,
                    d.lb,
                    d.ld,
                    d.ld,
                    result.as_ptr() as *const c_void,
                    result.as_mut_ptr() as *mut c_void,
                )
            };
            let ok = st == 0 && result[0].workspace_size <= ctx.ws_size;
            if !ok {
                pin::note_fallback(m, n, k, out_dtype, st, result[0].workspace_size);
            }
            ok
        }
        None => false,
    };
    if !use_pinned {
        // 2026-09-25: One heuristic result; `algo` is read from offset 0.
        if d.heuristic(ctx, &mut result)? < 1 {
            bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
        }
    }
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe {
        cublasLtMatmul(
            ctx.handle,
            d.desc,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            d.la,
            act as *const c_void,
            d.lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            d.ld,
            out as *mut c_void,
            d.ld,
            result.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        )
    };
    chk(status, "Matmul")
}
