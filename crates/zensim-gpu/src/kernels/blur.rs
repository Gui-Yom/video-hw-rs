//! Fused horizontal box-blur producing 4 outputs per pixel × 3 channels.
//!
//! For each pixel of `(padded_w × height)` and each channel ch ∈ {0,1,2}:
//!   `h_mu1_ch[x,y]      = Σ_k src_ch[mirror(x+k-r), y] / diam`
//!   `h_mu2_ch[x,y]      = Σ_k dst_ch[mirror(x+k-r), y] / diam`
//!   `h_sigma_sq_ch[x,y] = Σ_k (src_ch² + dst_ch²)     / diam`
//!   `h_sigma12_ch[x,y]  = Σ_k src_ch · dst_ch          / diam`
//!
//! ## One thread per pixel, all 3 channels per thread
//!
//! Reduces launch count from 12 (3 channels × 4 scales) to 4 per call.
//! Per-thread cost is 3× compute but the H-blur is memory-light (11
//! reads + 4 mul-adds per pixel per channel) so packing channels keeps
//! the kernel bandwidth-bound on shared cachelines (R, G, B at the same
//! pixel are typically allocated contiguously by the runtime).
//!
//! Per-channel FMA fusion order matches CPU
//! `zensim::blur::fused_blur_h_ssim_inner`.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn fused_blur_h_ssim_3ch_kernel(
    src_a: &Array<f32>, src_b: &Array<f32>, src_c: &Array<f32>,
    dst_a: &Array<f32>, dst_b: &Array<f32>, dst_c: &Array<f32>,
    h_mu1_a: &mut Array<f32>, h_mu1_b: &mut Array<f32>, h_mu1_c: &mut Array<f32>,
    h_mu2_a: &mut Array<f32>, h_mu2_b: &mut Array<f32>, h_mu2_c: &mut Array<f32>,
    h_sq_a: &mut Array<f32>,  h_sq_b: &mut Array<f32>,  h_sq_c: &mut Array<f32>,
    h_s12_a: &mut Array<f32>, h_s12_b: &mut Array<f32>, h_s12_c: &mut Array<f32>,
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

    let mut sa_m1 = 0.0_f32;
    let mut sa_m2 = 0.0_f32;
    let mut sa_sq = 0.0_f32;
    let mut sa_s12 = 0.0_f32;
    let mut sb_m1 = 0.0_f32;
    let mut sb_m2 = 0.0_f32;
    let mut sb_sq = 0.0_f32;
    let mut sb_s12 = 0.0_f32;
    let mut sc_m1 = 0.0_f32;
    let mut sc_m2 = 0.0_f32;
    let mut sc_sq = 0.0_f32;
    let mut sc_s12 = 0.0_f32;

    let mut k: u32 = 0u32;
    while k < diam {
        let raw = (x + k + period - radius) % period;
        let ix = if raw < width {
            raw as usize
        } else {
            (period - raw) as usize
        };
        let off = row + ix;
        let sa = src_a[off]; let da = dst_a[off];
        let sb = src_b[off]; let db = dst_b[off];
        let sc = src_c[off]; let dc = dst_c[off];
        sa_m1 += sa; sa_m2 += da;
        sa_sq = fma(sa, sa, fma(da, da, sa_sq));
        sa_s12 = fma(sa, da, sa_s12);
        sb_m1 += sb; sb_m2 += db;
        sb_sq = fma(sb, sb, fma(db, db, sb_sq));
        sb_s12 = fma(sb, db, sb_s12);
        sc_m1 += sc; sc_m2 += dc;
        sc_sq = fma(sc, sc, fma(dc, dc, sc_sq));
        sc_s12 = fma(sc, dc, sc_s12);
        k += 1u32;
    }

    let out = row + x_us;
    h_mu1_a[out] = sa_m1 * inv; h_mu2_a[out] = sa_m2 * inv;
    h_sq_a[out]  = sa_sq * inv; h_s12_a[out] = sa_s12 * inv;
    h_mu1_b[out] = sb_m1 * inv; h_mu2_b[out] = sb_m2 * inv;
    h_sq_b[out]  = sb_sq * inv; h_s12_b[out] = sb_s12 * inv;
    h_mu1_c[out] = sc_m1 * inv; h_mu2_c[out] = sc_m2 * inv;
    h_sq_c[out]  = sc_sq * inv; h_s12_c[out] = sc_s12 * inv;
}
