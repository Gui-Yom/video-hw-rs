//! Fused vertical-blur + per-pixel feature extraction.
//!
//! Mirrors `zensim-cuda-kernel/src/features.rs`, which itself is a
//! straight port of CPU zensim's scalar `fused_vblur_ssim_inner`.
//!
//! ## Layout
//!
//! One thread per output column (1D launch, `padded_w` threads). Each
//! thread walks `y` sequentially through the column, maintaining four
//! `f32` running sums over the `diam = 2·radius + 1`-tap V-blur
//! window. Per-thread accumulators are kept in `f64` (17 sums) and
//! `f32` (3 maxes) to match CPU precision; at end-of-column the thread
//! writes its column's sums to `partials_f64[col * 17 + i]` and its
//! maxes to `partials_max[col * 3 + i]`. Host-side fold across columns
//! produces the per-channel feature scalars.
//!
//! Per-column-slot partials avoid `Atomic<f64>` (cubecl 0.10 doesn't
//! expose it) and `Atomic<f32>::fetch_max` (broken on Metal per
//! gotcha G3.x). The trade is `padded_w × 17 × 8` bytes of scratch per
//! (scale, channel) — 557 KiB at 4 K, well within budget.
//!
//! ## Slot meanings (matches CPU `compute_features` in `zensim-cuda`)
//!
//! ```text
//! a[0]  Σ sd       a[1]  Σ sd⁴      a[2]  Σ sd²
//! a[3]  Σ artifact a[4]  Σ a⁴       a[5]  Σ a²
//! a[6]  Σ detail   a[7]  Σ d⁴       a[8]  Σ d²
//! a[9]  Σ (s-d)²
//! a[10] Σ (s-mu1)² a[11] Σ (d-mu2)² a[12] Σ |s-mu1| a[13] Σ |d-mu2|
//! a[14] Σ sd⁸      a[15] Σ a⁸       a[16] Σ d⁸
//! peak[0] max sd, peak[1] max artifact, peak[2] max detail
//! ```

use cubecl::prelude::*;

const C2: f32 = 0.0009;

/// One thread per column. Each thread walks the whole height,
/// maintaining the V-blur window state and per-column accumulators,
/// then writes 17 f64 + 3 f32 partials at slots `col * 17` and
/// `col * 3` respectively.
///
/// Mirror logic is inlined in u32: `mirror((y + plus - minus), height)`
/// is computed as `((y + plus + period - minus) % period) → fold` where
/// `period = 2 (height - 1)`. Caller guarantees `height ≥ radius + 1`.
#[cube(launch_unchecked)]
pub fn fused_vblur_features_kernel(
    h_mu1: &Array<f32>,
    h_mu2: &Array<f32>,
    h_sigma_sq: &Array<f32>,
    h_sigma12: &Array<f32>,
    src: &Array<f32>,
    dst: &Array<f32>,
    partials_f64: &mut Array<f64>,
    partials_max: &mut Array<f32>,
    width: u32, // padded_w
    height: u32,
    radius: u32,
    slot_off_f64: u32,
    slot_off_max: u32,
) {
    let x_pos = ABSOLUTE_POS;
    if x_pos >= (width as usize) {
        terminate!();
    }
    let w = width as usize;
    let diam = 2u32 * radius + 1u32;
    let inv = 1.0_f32 / (diam as f32);
    let period = 2u32 * (height - 1u32);

    // Initialise V-blur sums from the mirrored prefix (window y =
    // k - r for k in 0..diam, i.e. y_pos = 0, plus = k, minus = r).
    let mut sum_m1 = 0.0_f32;
    let mut sum_m2 = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut sum_s12 = 0.0_f32;
    let mut k: u32 = 0u32;
    while k < diam {
        let raw = (k + period - radius) % period;
        let row_i = if raw < height {
            raw as usize
        } else {
            (period - raw) as usize
        };
        let off = row_i * w + x_pos;
        sum_m1 += h_mu1[off];
        sum_m2 += h_mu2[off];
        sum_sq += h_sigma_sq[off];
        sum_s12 += h_sigma12[off];
        k += 1u32;
    }

    // 17 f64 accumulators expanded to scalars (cubecl 0.10's `#[cube]`
    // macro handles fixed-size local arrays unevenly; scalars compile
    // reliably across all backends).
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

    let mut y: u32 = 0u32;
    while y < height {
        let mu1 = sum_m1 * inv;
        let mu2 = sum_m2 * inv;
        let ssq = sum_sq * inv;
        let s12 = sum_s12 * inv;

        let off = (y as usize) * w + x_pos;
        let sv = src[off];
        let dv = dst[off];

        // SSIMULACRA2-style SSIM (no C1, uses `1 - (mu1-mu2)²`).
        // FMA fusion order matches CPU `zensim::fused::fused_vblur_ssim_inner_v4`
        // exactly:
        //   num_m   = mu_diff * (-mu_diff) + 1
        //   num_s   = 2 * (-mu1 * mu2 + s12) + C2
        //   denom_s = -mu2 * mu2 + (-mu1 * mu1 + ssq) + C2
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

        // Slide V-blur window: add (y + r + 1), remove (y - r). Inlined
        // mirror — same formula as the prefix init above.
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
        let a_off = add_idx * w + x_pos;
        let r_off = rem_idx * w + x_pos;
        sum_m1 = sum_m1 + h_mu1[a_off] - h_mu1[r_off];
        sum_m2 = sum_m2 + h_mu2[a_off] - h_mu2[r_off];
        sum_sq = sum_sq + h_sigma_sq[a_off] - h_sigma_sq[r_off];
        sum_s12 = sum_s12 + h_sigma12[a_off] - h_sigma12[r_off];

        y += 1u32;
    }

    let base = (slot_off_f64 as usize) + x_pos * 17;
    partials_f64[base] = a0;
    partials_f64[base + 1] = a1;
    partials_f64[base + 2] = a2;
    partials_f64[base + 3] = a3;
    partials_f64[base + 4] = a4;
    partials_f64[base + 5] = a5;
    partials_f64[base + 6] = a6;
    partials_f64[base + 7] = a7;
    partials_f64[base + 8] = a8;
    partials_f64[base + 9] = a9;
    partials_f64[base + 10] = a10;
    partials_f64[base + 11] = a11;
    partials_f64[base + 12] = a12;
    partials_f64[base + 13] = a13;
    partials_f64[base + 14] = a14;
    partials_f64[base + 15] = a15;
    partials_f64[base + 16] = a16;
    let mbase = (slot_off_max as usize) + x_pos * 3;
    partials_max[mbase] = peak0;
    partials_max[mbase + 1] = peak1;
    partials_max[mbase + 2] = peak2;
}
