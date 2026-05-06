//! Per-scale partial-reduction kernel.
//!
//! Folds `(col, strip, channel)` partials into per-(channel, slot)
//! finals. Without this pass the host has to read back the full
//! per-column per-strip partials buffer at 5–30 MiB per call; with it
//! the host reads the ~1.6 KiB final aggregates only.
//!
//! ## Launch geometry — one launch per scale
//!
//! Grid: `(60, 1, 1)` per scale. Each cube handles ONE
//! `(channel, slot_kind)` pair where `slot_kind ∈ [0, 20)` decoded as
//! `< 17` → f64 sum slot, `≥ 17` → f32 max slot (`slot_kind − 17`).
//!
//! 4 scales × 60 cubes per launch × 256 threads/cube = ~60 K threads
//! per call — but only **4 launches**, not 240. Cuts launch overhead
//! from ~12 ms to ~0.2 ms at small image sizes (the reduction work
//! itself is only a few microseconds for the 1 K-or-smaller cases
//! since each cube only processes ~hundreds of partials).

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn reduce_scale_kernel(
    partials_f64: &Array<f64>,
    partials_max: &Array<f32>,
    finals_f64: &mut Array<f64>,
    finals_max: &mut Array<f32>,
    f64_slot_off: u32,
    max_slot_off: u32,
    n_partials_per_ch: u32, // = pw × n_strips
    final_f64_base: u32,    // base index into finals_f64 for this scale
    final_max_base: u32,    // base index into finals_max for this scale
) {
    let cube_idx = CUBE_POS_X;
    let channel = cube_idx / 20u32;
    let slot_kind = cube_idx - channel * 20u32;
    let tid = UNIT_POS_X;
    let tid_us = tid as usize;

    if slot_kind < 17u32 {
        // f64 sum reduction.
        let f64_ch_off = (f64_slot_off as usize)
            + (channel as usize) * (n_partials_per_ch as usize) * 17;
        let inner = slot_kind as usize;

        let mut shared = SharedMemory::<f64>::new(256usize);
        let mut sum = 0.0_f64;
        let mut i = tid;
        while i < n_partials_per_ch {
            sum = sum + partials_f64[f64_ch_off + (i as usize) * 17 + inner];
            i = i + 256u32;
        }
        shared[tid_us] = sum;
        let mut step: u32 = 128u32;
        while step > 0u32 {
            sync_cube();
            if tid < step {
                let lhs = shared[tid_us];
                let rhs = shared[tid_us + (step as usize)];
                shared[tid_us] = lhs + rhs;
            }
            step = step / 2u32;
        }
        sync_cube();
        if tid == 0u32 {
            let final_idx = (final_f64_base as usize) + (channel as usize) * 17 + inner;
            finals_f64[final_idx] = shared[0];
        }
    } else {
        // f32 max reduction.
        let max_ch_off = (max_slot_off as usize)
            + (channel as usize) * (n_partials_per_ch as usize) * 3;
        let inner = (slot_kind - 17u32) as usize;

        let mut shared = SharedMemory::<f32>::new(256usize);
        let mut peak = 0.0_f32;
        let mut i = tid;
        while i < n_partials_per_ch {
            let v = partials_max[max_ch_off + (i as usize) * 3 + inner];
            if v > peak {
                peak = v;
            }
            i = i + 256u32;
        }
        shared[tid_us] = peak;
        let mut step: u32 = 128u32;
        while step > 0u32 {
            sync_cube();
            if tid < step {
                let lhs = shared[tid_us];
                let rhs = shared[tid_us + (step as usize)];
                shared[tid_us] = if lhs > rhs { lhs } else { rhs };
            }
            step = step / 2u32;
        }
        sync_cube();
        if tid == 0u32 {
            let final_idx = (final_max_base as usize) + (channel as usize) * 3 + inner;
            finals_max[final_idx] = shared[0];
        }
    }
}
