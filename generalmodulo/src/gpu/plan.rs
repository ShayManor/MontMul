//! Launch planner: picks shared- or global-memory mode, block size and dynamic shared memory
//! for one transform length. The carry scratch (`scratch_bytes`) is its own region and never
//! aliases the vector. The kernels need a whole number of warps and at most 1,024 threads
//! (full-warp shuffles and ballots, one ballot over the warps), and at most 512 in global mode
//! (the global-mode kernels' launch bounds), which every plan satisfies.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The whole vector lives in shared memory: F2 (16n bytes), F1 (8n bytes), then the scratch.
    Shared,
    /// The vector stays in global memory; shared memory holds the scratch and one transform
    /// tile (`global_smem_bytes`).
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchPlan {
    pub mode: Mode,
    /// Threads per block.
    pub threads: u32,
    /// Dynamic shared memory per block.
    pub smem_bytes: u32,
}

/// Plan for length `n` on a device allowing `max_dyn_smem` bytes of dynamic shared memory per block.
pub fn plan(n: usize, max_dyn_smem: u32, force_global: bool) -> LaunchPlan {
    let half = (n / 2) as u64;
    // One radix-4 butterfly group per thread: measured best on the 3060 at n = 1024, 2048, 4096.
    let shared_threads = (half / 2).clamp(64, SHARED_MAX_THREADS as u64) as u32;
    let shared_bytes = 24 * n as u64 + scratch_bytes(shared_threads) as u64;
    if !force_global && shared_bytes <= max_dyn_smem as u64 {
        return LaunchPlan { mode: Mode::Shared, threads: shared_threads, smem_bytes: shared_bytes as u32 };
    }
    let threads = half.clamp(64, GLOBAL_MAX_THREADS as u64) as u32;
    LaunchPlan { mode: Mode::Global, threads, smem_bytes: global_smem_bytes(n) }
}

/// Dynamic shared memory of global mode at length `n`: the carry scratch of up to 1,024
/// threads, then a transform tile of up to 2^TILE_LOG elements of both fields.
pub fn global_smem_bytes(n: usize) -> u32 {
    TILE_OFFSET + 24 * n.min(1 << TILE_LOG) as u32
}

/// The kernels' `TILE_OFFSET` and `TILE_LOG` (kernels/src/lib.rs), which must match.
const TILE_OFFSET: u32 = 528;
const TILE_LOG: u32 = 11;
const _: () = assert!(TILE_OFFSET == scratch_bytes(1024));
// A tile of a multi-tile transform holds at least 2^(TILE_LOG − 1) elements, a multiple of the
// global block size, which the kernels' DIT tile copy-in relies on.
const _: () = assert!((1 << (TILE_LOG - 1)) % GLOBAL_MAX_THREADS == 0);

/// Most threads per block in shared mode.
const SHARED_MAX_THREADS: u32 = 1024;
/// Most threads per block in global mode (the global kernels' launch bounds).
const GLOBAL_MAX_THREADS: u32 = 512;

/// Bytes of the carry scratch for `threads` threads per block: two u64 per warp and two more.
pub const fn scratch_bytes(threads: u32) -> u32 {
    8 * (2 * threads.div_ceil(32) + 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_fits_3060_at_t100k() {
        assert_eq!(plan(4096, 101_376, false), LaunchPlan { mode: Mode::Shared, threads: 1024, smem_bytes: 98_832 });
    }
    #[test]
    fn shared_threads_one_per_radix4_group() {
        assert_eq!(plan(1024, 101_376, false).threads, 256);
        assert_eq!(plan(2048, 101_376, false).threads, 512);
        assert_eq!(plan(256, 101_376, false).threads, 64);
    }
    #[test]
    fn global_when_too_big_for_3060() {
        assert_eq!(plan(8192, 101_376, false), LaunchPlan { mode: Mode::Global, threads: 512, smem_bytes: 49_680 });
    }
    #[test]
    fn global_from_8192_on_5090() {
        // The 5090 (sm_120) allows 101,376 B per block, like the 3060.
        assert_eq!(plan(4096, 101_376, false).mode, Mode::Shared);
        assert_eq!(plan(8192, 101_376, false).mode, Mode::Global);
    }
    #[test]
    fn forced_global_and_tiny() {
        assert_eq!(plan(4096, 101_376, true).mode, Mode::Global);
        assert_eq!(plan(4, 101_376, false), LaunchPlan { mode: Mode::Shared, threads: 64, smem_bytes: 144 });
        assert_eq!(plan(4096, 49_152, false).mode, Mode::Global); // device without opt-in
    }
}
