//! 2× planar downscale (2×2 box average with edge clamp).
//!
//! Verbatim port of `zensim-cuda-kernel/src/downscale.rs`. Operates on
//! the full `padded_w × height` plane — CPU zensim does NOT re-pad
//! after downscaling; pad columns simply get downscaled along with
//! everything else. Mirror this exactly for parity.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn downscale_2x_plane_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_w as usize;
    let sw = src_w as usize;
    let oy = idx / dw;
    let ox = idx - oy * dw;

    let sx0 = (ox * 2) as u32;
    let sy0 = (oy * 2) as u32;
    let last_x = src_w - 1;
    let last_y = src_h - 1;
    let x0 = u32::min(sx0, last_x) as usize;
    let x1 = u32::min(sx0 + 1, last_x) as usize;
    let y0 = u32::min(sy0, last_y) as usize;
    let y1 = u32::min(sy0 + 1, last_y) as usize;

    let v00 = src[y0 * sw + x0];
    let v10 = src[y0 * sw + x1];
    let v01 = src[y1 * sw + x0];
    let v11 = src[y1 * sw + x1];
    dst[idx] = (v00 + v10 + v01 + v11) * 0.25;
}
