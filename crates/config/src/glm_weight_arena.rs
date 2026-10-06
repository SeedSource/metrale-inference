// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The `METRALE_GLM_WEIGHT_ARENA` lever (`glm_weight_arena`).
//!
//! Owner: config.
//! Invariants: read once per process; only `1` (trimmed) turns it on.

/// 2026-10-06: Whether GLM-5.3 places its routed-expert weights in a weight arena
/// (`METRALE_GLM_WEIGHT_ARENA`, see [`glm_weight_arena_from`]): a few large allocations,
/// sub-allocated, instead of one `cuMemAlloc` per tensor, which on GB10 saves ~15 KiB of
/// unledgered driver memory per allocation. Off by default; read once per process. Read by the
/// server (the fast loader's arena hook) and the GLM-5.3 weight loader (the derived arena).
pub fn glm_weight_arena() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        glm_weight_arena_from(std::env::var("METRALE_GLM_WEIGHT_ARENA").ok().as_deref())
    })
}

/// 2026-10-06: The policy half of [`glm_weight_arena`]: only `1` (trimmed) turns it on.
pub fn glm_weight_arena_from(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

#[cfg(test)]
mod tests {
    use super::glm_weight_arena_from as on;

    #[test]
    fn only_one_turns_the_weight_arena_on() {
        for off in [None, Some(""), Some("0"), Some("true"), Some("on")] {
            assert!(!on(off), "{off:?} must leave the lever off");
        }
        for v in ["1", " 1", "1\n"] {
            assert!(on(Some(v)), "{v:?} must turn the lever on");
        }
    }
}
