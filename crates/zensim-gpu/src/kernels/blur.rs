//! Fused horizontal box-blur producing 4 outputs per pixel.
//!
//! Verbatim port of `zensim-cuda-kernel/src/blur.rs`. For each pixel in
//! `(padded_w × height)`:
//!   `h_mu1[x,y]      = Σ_k src[mirror(x+k-r), y] / diam`
//!   `h_mu2[x,y]      = Σ_k dst[mirror(x+k-r), y] / diam`
//!   `h_sigma_sq[x,y] = Σ_k (src² + dst²)         / diam`
//!   `h_sigma12[x,y]  = Σ_k src·dst               / diam`
//!
//! Mirror logic is inlined in u32 to keep cubecl's `#[cube]` codegen on
//! a single integer kind. Caller guarantees `width ≥ radius + 1`
//! (zensim's smallest scale is `width = 8`, `r = 5`, period = 14 ≥ 10).

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn fused_blur_h_ssim_kernel(
    src: &Array<f32>,
    dst: &Array<f32>,
    h_mu1: &mut Array<f32>,
    h_mu2: &mut Array<f32>,
    h_sigma_sq: &mut Array<f32>,
    h_sigma12: &mut Array<f32>,
    width: u32, // == padded_w
    height: u32,
    radius: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let y = idx / w;
    let x_us = idx - y * w;
    let x = x_us as u32;

    let diam = 2u32 * radius + 1u32;
    let inv = 1.0_f32 / (diam as f32);
    let period = 2u32 * (width - 1u32);
    let row = y * w;

    let mut sum_m1 = 0.0_f32;
    let mut sum_m2 = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut sum_s12 = 0.0_f32;
    let mut k: u32 = 0u32;
    while k < diam {
        // mirror((x + k - r), width) inlined.
        let raw = (x + k + period - radius) % period;
        let ix = if raw < width {
            raw as usize
        } else {
            (period - raw) as usize
        };
        let s = src[row + ix];
        let d = dst[row + ix];
        sum_m1 += s;
        sum_m2 += d;
        sum_sq += s * s + d * d;
        sum_s12 += s * d;
        k += 1u32;
    }

    let out = row + x_us;
    h_mu1[out] = sum_m1 * inv;
    h_mu2[out] = sum_m2 * inv;
    h_sigma_sq[out] = sum_sq * inv;
    h_sigma12[out] = sum_s12 * inv;
}
