//! Fused vertical-blur + per-pixel feature extraction, 3-channel +
//! column-strip parallel.
//!
//! Mirrors `zensim-cuda-kernel/src/features.rs` and CPU
//! `zensim::fused::fused_vblur_ssim_inner`. The compute is the same;
//! the launch geometry is different.
//!
//! ## Launch geometry
//!
//! Grid: `(padded_w, n_strips * 3, 1)`. Each thread is a `(col,
//! strip × ch)` tuple. The strip dimension subdivides each column into
//! `n_strips` row-ranges so column-bound parallelism scales with image
//! height — without this the V-blur is bottlenecked at `padded_w`
//! threads (~2 K at 2 K image), severely underutilising modern GPU SMs
//! at ≥ 1 K resolution.
//!
//! Each thread:
//! 1. Initialises the V-blur window from the mirrored prefix at its
//!    strip's `y_start`.
//! 2. Walks `[y_start, y_end)` maintaining the 4 running f32 sliding
//!    sums and 17 f64 + 3 f32 per-thread accumulators.
//! 3. Writes its accumulator state to a per-(col, strip, ch) slot in
//!    the partials buffer. No atomics needed.
//!
//! Host-side fold sums across cols × strips per (scale, ch). With
//! `n_strips = 1` this collapses to the original per-column kernel.
//!
//! ## Per-channel offsets
//!
//! Per-thread `channel = idx_y / n_strips`, `strip = idx_y % n_strips`.
//! The kernel reads `h_mu1[ch]` etc. from contiguous Array arguments.

use cubecl::prelude::*;

const C2: f32 = 0.0009;

/// Strip-and-channel parallel V-blur + features kernel. One thread per
/// `(col, strip, channel)` writes 17 f64 + 3 f32 partials into the
/// shared partials buffer at offset
/// `slot_off_f64 + (ch * n_strips * pw + strip * pw + col) * 17`.
#[cube(launch_unchecked)]
pub fn fused_vblur_features_kernel(
    h_mu1_a: &Array<f32>, h_mu2_a: &Array<f32>, h_sq_a: &Array<f32>, h_s12_a: &Array<f32>,
    h_mu1_b: &Array<f32>, h_mu2_b: &Array<f32>, h_sq_b: &Array<f32>, h_s12_b: &Array<f32>,
    h_mu1_c: &Array<f32>, h_mu2_c: &Array<f32>, h_sq_c: &Array<f32>, h_s12_c: &Array<f32>,
    src_a: &Array<f32>, dst_a: &Array<f32>,
    src_b: &Array<f32>, dst_b: &Array<f32>,
    src_c: &Array<f32>, dst_c: &Array<f32>,
    partials_f64: &mut Array<f64>,
    partials_max: &mut Array<f32>,
    width: u32,    // padded_w
    height: u32,
    radius: u32,
    n_strips: u32,
    slot_off_f64: u32,
    slot_off_max: u32,
) {
    let col = CUBE_POS_X * (CUBE_DIM_X as u32) + UNIT_POS_X;
    let yc = CUBE_POS_Y * (CUBE_DIM_Y as u32) + UNIT_POS_Y;
    if col >= width {
        terminate!();
    }
    let n_total_y = n_strips * 3u32;
    if yc >= n_total_y {
        terminate!();
    }
    let channel = yc / n_strips;
    let strip = yc - channel * n_strips;

    let w = width as usize;
    let col_us = col as usize;
    let pw = width as usize;
    let n_strips_us = n_strips as usize;
    let diam = 2u32 * radius + 1u32;
    let inv = 1.0_f32 / (diam as f32);
    let period = 2u32 * (height - 1u32);

    // Compute strip's [y_start, y_end). Last strip absorbs remainder.
    let strip_h_base = height / n_strips;
    let strip_rem = height - strip_h_base * n_strips;
    let y_start = strip * strip_h_base + u32::min(strip, strip_rem);
    let y_end_unclamp = y_start + strip_h_base + (if strip < strip_rem { 1u32 } else { 0u32 });
    let y_end = u32::min(y_end_unclamp, height);

    // Pick the channel's input arrays via a manual switch (cubecl 0.10
    // doesn't index into a stack of `&Array<f32>`).
    let mut sum_m1 = 0.0_f32;
    let mut sum_m2 = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut sum_s12 = 0.0_f32;

    // Initialise V-blur window from mirrored prefix at y_start - r.
    let mut k: u32 = 0u32;
    while k < diam {
        let raw = (y_start + k + period - radius) % period;
        let row_i = if raw < height {
            raw as usize
        } else {
            (period - raw) as usize
        };
        let off = row_i * w + col_us;
        if channel == 0u32 {
            sum_m1 += h_mu1_a[off];
            sum_m2 += h_mu2_a[off];
            sum_sq += h_sq_a[off];
            sum_s12 += h_s12_a[off];
        } else {
            if channel == 1u32 {
                sum_m1 += h_mu1_b[off];
                sum_m2 += h_mu2_b[off];
                sum_sq += h_sq_b[off];
                sum_s12 += h_s12_b[off];
            } else {
                sum_m1 += h_mu1_c[off];
                sum_m2 += h_mu2_c[off];
                sum_sq += h_sq_c[off];
                sum_s12 += h_s12_c[off];
            }
        }
        k += 1u32;
    }

    let mut a0 = 0.0_f64;
    let mut a1 = 0.0_f64;
    let mut a2 = 0.0_f64;
    let mut a3 = 0.0_f64;
    let mut a4 = 0.0_f64;
    let mut a5 = 0.0_f64;
    let mut a6 = 0.0_f64;
    let mut a7 = 0.0_f64;
    let mut a8 = 0.0_f64;
    let mut a9 = 0.0_f64;
    let mut a10 = 0.0_f64;
    let mut a11 = 0.0_f64;
    let mut a12 = 0.0_f64;
    let mut a13 = 0.0_f64;
    let mut a14 = 0.0_f64;
    let mut a15 = 0.0_f64;
    let mut a16 = 0.0_f64;
    let mut peak0 = 0.0_f32;
    let mut peak1 = 0.0_f32;
    let mut peak2 = 0.0_f32;

    let mut y: u32 = y_start;
    while y < y_end {
        let mu1 = sum_m1 * inv;
        let mu2 = sum_m2 * inv;
        let ssq = sum_sq * inv;
        let s12 = sum_s12 * inv;

        let off = (y as usize) * w + col_us;
        let mut sv = 0.0_f32;
        let mut dv = 0.0_f32;
        if channel == 0u32 {
            sv = src_a[off];
            dv = dst_a[off];
        } else {
            if channel == 1u32 {
                sv = src_b[off];
                dv = dst_b[off];
            } else {
                sv = src_c[off];
                dv = dst_c[off];
            }
        }

        // SSIMULACRA2-style SSIM (no C1, uses `1 - (mu1-mu2)²`).
        // FMA fusion order matches CPU
        // `zensim::fused::fused_vblur_ssim_inner_v4` exactly.
        let mu_diff = mu1 - mu2;
        let num_m = fma(mu_diff, -mu_diff, 1.0);
        let inner_ns = fma(-mu1, mu2, s12);
        let num_s = fma(2.0, inner_ns, C2);
        let inner_ds_inner = fma(-mu1, mu1, ssq);
        let denom_s = fma(-mu2, mu2, inner_ds_inner) + C2;
        let sd_raw = 1.0 - (num_m * num_s) / denom_s;
        let sd = if sd_raw > 0.0 { sd_raw } else { f32::new(0.0) };
        let sd2 = sd * sd;
        let sd4 = sd2 * sd2;
        a0 += sd as f64;
        a1 += sd4 as f64;
        a2 += sd2 as f64;
        a14 += (sd4 * sd4) as f64;
        if sd > peak0 {
            peak0 = sd;
        }

        // Edge artifact / detail-lost.
        let diff1 = f32::abs(sv - mu1);
        let diff2 = f32::abs(dv - mu2);
        let ed = (1.0 + diff2) / (1.0 + diff1) - 1.0;
        let artifact = if ed > 0.0 { ed } else { f32::new(0.0) };
        let detail_lost = if ed < 0.0 { -ed } else { f32::new(0.0) };
        let a2_v = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        let a4_v = a2_v * a2_v;
        let dl4 = dl2 * dl2;
        a3 += artifact as f64;
        a4 += a4_v as f64;
        a5 += a2_v as f64;
        a6 += detail_lost as f64;
        a7 += dl4 as f64;
        a8 += dl2 as f64;
        a15 += (a4_v * a4_v) as f64;
        a16 += (dl4 * dl4) as f64;
        if artifact > peak1 {
            peak1 = artifact;
        }
        if detail_lost > peak2 {
            peak2 = detail_lost;
        }

        // HF variance + texture magnitude.
        let vs = sv - mu1;
        let vd = dv - mu2;
        a10 += (vs * vs) as f64;
        a11 += (vd * vd) as f64;
        a12 += diff1 as f64;
        a13 += diff2 as f64;

        // MSE.
        let pd = sv - dv;
        a9 += (pd * pd) as f64;

        // Slide V-blur window.
        let add_raw = (y + radius + 1u32 + period) % period;
        let add_idx = if add_raw < height {
            add_raw as usize
        } else {
            (period - add_raw) as usize
        };
        let rem_raw = (y + period - radius) % period;
        let rem_idx = if rem_raw < height {
            rem_raw as usize
        } else {
            (period - rem_raw) as usize
        };
        let a_off = add_idx * w + col_us;
        let r_off = rem_idx * w + col_us;
        if channel == 0u32 {
            sum_m1 = sum_m1 + h_mu1_a[a_off] - h_mu1_a[r_off];
            sum_m2 = sum_m2 + h_mu2_a[a_off] - h_mu2_a[r_off];
            sum_sq = sum_sq + h_sq_a[a_off] - h_sq_a[r_off];
            sum_s12 = sum_s12 + h_s12_a[a_off] - h_s12_a[r_off];
        } else {
            if channel == 1u32 {
                sum_m1 = sum_m1 + h_mu1_b[a_off] - h_mu1_b[r_off];
                sum_m2 = sum_m2 + h_mu2_b[a_off] - h_mu2_b[r_off];
                sum_sq = sum_sq + h_sq_b[a_off] - h_sq_b[r_off];
                sum_s12 = sum_s12 + h_s12_b[a_off] - h_s12_b[r_off];
            } else {
                sum_m1 = sum_m1 + h_mu1_c[a_off] - h_mu1_c[r_off];
                sum_m2 = sum_m2 + h_mu2_c[a_off] - h_mu2_c[r_off];
                sum_sq = sum_sq + h_sq_c[a_off] - h_sq_c[r_off];
                sum_s12 = sum_s12 + h_s12_c[a_off] - h_s12_c[r_off];
            }
        }

        y += 1u32;
    }

    // Write 17 + 3 partials at slot
    //   ch * n_strips * pw + strip * pw + col, offset 17 within slot.
    let slot_idx_us = (channel as usize) * n_strips_us * pw
        + (strip as usize) * pw
        + col_us;
    let f64_base = (slot_off_f64 as usize) + slot_idx_us * 17;
    partials_f64[f64_base] = a0;
    partials_f64[f64_base + 1] = a1;
    partials_f64[f64_base + 2] = a2;
    partials_f64[f64_base + 3] = a3;
    partials_f64[f64_base + 4] = a4;
    partials_f64[f64_base + 5] = a5;
    partials_f64[f64_base + 6] = a6;
    partials_f64[f64_base + 7] = a7;
    partials_f64[f64_base + 8] = a8;
    partials_f64[f64_base + 9] = a9;
    partials_f64[f64_base + 10] = a10;
    partials_f64[f64_base + 11] = a11;
    partials_f64[f64_base + 12] = a12;
    partials_f64[f64_base + 13] = a13;
    partials_f64[f64_base + 14] = a14;
    partials_f64[f64_base + 15] = a15;
    partials_f64[f64_base + 16] = a16;
    let max_base = (slot_off_max as usize) + slot_idx_us * 3;
    partials_max[max_base] = peak0;
    partials_max[max_base + 1] = peak1;
    partials_max[max_base + 2] = peak2;
}
