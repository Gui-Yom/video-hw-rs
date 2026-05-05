//! Fill SIMD-padding columns with mirror-reflected real columns.
//!
//! Verbatim port of `zensim-cuda-kernel/src/pad.rs`. Required because
//! CPU zensim's feature accumulators sum over `padded_w × height`, so
//! the GPU output must match the same per-column footprint.
//!
//! For `pad_col ∈ [0, padded_w - logical_w)`:
//!   `plane[y, logical_w + pad_col] = plane[y, mirror_offsets[pad_col]]`
//!
//! `mirror_offsets` is precomputed host-side using
//! `mirror = (logical_w + pad_col) % (2 * (logical_w - 1))` and then
//! folded back if the modular result lands in the "reflected" half.

use cubecl::prelude::*;

/// One thread per (pad_col, y). 1D launch with
/// `total = pad_count × height` threads.
#[cube(launch_unchecked)]
pub fn pad_mirror_plane_kernel(
    plane: &mut Array<f32>,
    mirror_offsets: &Array<u32>,
    logical_w: u32,
    padded_w: u32,
    height: u32,
) {
    let idx = ABSOLUTE_POS;
    let pad_count = padded_w - logical_w;
    let total = (pad_count * height) as usize;
    if idx >= total {
        terminate!();
    }
    let pc = pad_count as usize;
    let pw = padded_w as usize;
    let y = idx / pc;
    let pad_col = idx - y * pc;
    let src_col = mirror_offsets[pad_col] as usize;
    let dst_col = (logical_w as usize) + pad_col;
    let row = y * pw;
    plane[row + dst_col] = plane[row + src_col];
}
