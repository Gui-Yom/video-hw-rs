//! Fused horizontal box-blur producing 4 outputs per pixel.
//!
//! For each pixel in `(padded_w × height)`:
//!   `h_mu1[x,y]      = Σ_k src[mirror(x+k-r), y] / diam`
//!   `h_mu2[x,y]      = Σ_k dst[mirror(x+k-r), y] / diam`
//!   `h_sigma_sq[x,y] = Σ_k (src² + dst²)         / diam`
//!   `h_sigma12[x,y]  = Σ_k src·dst               / diam`
//!
//! ## One thread per pixel, recompute the window
//!
//! CPU's `fused_blur_h_ssim_inner` slides a per-row window incrementally
//! across `x` (init once, then `+ add - rem` for each output). That's
//! cache-friendly for SIMD but bottlenecks at `height` threads of
//! parallelism on a GPU.
//!
//! We recompute the full `diam`-tap window per pixel, dispatching one
//! thread per output pixel (= `padded_w × height` threads). At
//! `radius = 5` (diam = 11), per-pixel work is 11 reads + 4 mul-adds —
//! cheap, and full-image parallelism dominates the launch.
//!
//! Recompute and CPU's slide give identical mathematical results; f32
//! ULP rounding can differ at sub-1e-7 precision, but accumulated
//! pipeline error stays below the 0.013 % real-image score-parity
//! target. Synthetic edge cases (noisy grayscale, black-vs-white) are
//! well under 0.01 % via the FMA-fusion match; real photos at q70
//! land near 0.013 % which is the cross-arch FMA drift floor.
//!
//! Mirror-x logic is inlined in u32 (`(x + k + period - r) % period`).
//! Caller guarantees `width ≥ radius + 1`; zensim's smallest scale is
//! `width = 8`, `r = 5`, period = 14 ≥ 10.

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
        // CPU FMA fusion: `sum_sq = s.mul_add(s, d.mul_add(d, sum_sq))`.
        sum_sq = fma(s, s, fma(d, d, sum_sq));
        // CPU FMA fusion: `sum_prod = s.mul_add(d, sum_prod)`.
        sum_s12 = fma(s, d, sum_s12);
        k += 1u32;
    }

    let out = row + x_us;
    h_mu1[out] = sum_m1 * inv;
    h_mu2[out] = sum_m2 * inv;
    h_sigma_sq[out] = sum_sq * inv;
    h_sigma12[out] = sum_s12 * inv;
}
