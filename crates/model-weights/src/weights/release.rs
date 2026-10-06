// SPDX-License-Identifier: MIT OR Apache-2.0

use super::WeightStore;
use metrale_gpu_runtime::gpu::GpuBackend;

/// Release every weight tensor.
///
/// Safe to free per-entry because the loaders allocate per-tensor: the fast
/// path calls `gpu.alloc(meta.len)` once per tensor before inserting it
/// (`fast_weights/mod.rs:360-388`), and no loader inserts an `.offset()` view of
/// a shared block into this map. (Fused per-expert views DO exist — see
/// `weight_loader/step3p7.rs:93` — but they live in the layer structs that own
/// the fused allocation, not here, so this cannot double-free them.)
/// 2026-10-06: The weight arena's sub-allocations are the one exception: they
/// are skipped here and freed with their chunks after the loop.
impl metrale_core::scope::ModelResource<dyn GpuBackend> for WeightStore {
    fn label(&self) -> &'static str {
        "weight store"
    }

    fn release(&mut self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        // Derived buffers FIRST: they are re-encodings of the tensors below and
        // nothing reads one after the other is gone, but freeing the source a
        // derivation was built from while the derivation is still listed would
        // make a later failure here impossible to attribute.
        let mut first_error = self.derived.release(gpu).err();
        // `drain` rather than iterate: the map must not be left holding
        // pointers to memory that is gone, and it makes this idempotent.
        for (name, tensor) in self.weights.drain() {
            // 2026-10-06: Arena tensors are freed with their chunks, below.
            if self.arena.contains(tensor.ptr) {
                continue;
            }
            if let Err(e) = gpu.free(tensor.ptr)
                && first_error.is_none()
            {
                first_error = Some(e.context(format!("freeing weight {name}")));
            }
        }
        if let Err(e) = self.arena.release(gpu)
            && first_error.is_none()
        {
            first_error = Some(e);
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
