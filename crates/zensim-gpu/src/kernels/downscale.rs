//! 2× planar downscale (2×2 box average with edge clamp), 3-channel
//! per launch.
//!
//! Operates on the full `padded_w × height` plane — CPU zensim does
//! NOT re-pad after downscaling; pad columns simply downscale along
//! with everything else. Mirror this exactly for parity.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn downscale_2x_3ch_kernel(
    src_a: &Array<f32>,
    src_b: &Array<f32>,
    src_c: &Array<f32>,
    dst_a: &mut Array<f32>,
    dst_b: &mut Array<f32>,
    dst_c: &mut Array<f32>,
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
    let i00 = y0 * sw + x0;
    let i10 = y0 * sw + x1;
    let i01 = y1 * sw + x0;
    let i11 = y1 * sw + x1;

    dst_a[idx] = (src_a[i00] + src_a[i10] + src_a[i01] + src_a[i11]) * 0.25;
    dst_b[idx] = (src_b[i00] + src_b[i10] + src_b[i01] + src_b[i11]) * 0.25;
    dst_c[idx] = (src_c[i00] + src_c[i10] + src_c[i01] + src_c[i11]) * 0.25;
}
