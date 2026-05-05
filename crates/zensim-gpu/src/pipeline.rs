//! zensim pipeline orchestration.
//!
//! Wires the kernels in `kernels::*` into the 4-scale zensim feature
//! extractor.
//!
//! Public entry points:
//! - [`Zensim::new`] + [`Zensim::compute_features`] — extract the
//!   228-feature vector from one (ref, dist) pair.
//! - [`Zensim::set_reference`] + [`Zensim::compute_with_reference`] —
//!   cache the reference-side pyramid and score many distorted images
//!   against it (encoder-loop friendly).
//!
//! ## Algorithm (per scale, faithful to `zensim-cuda`'s `compute_features`)
//!
//! 1. sRGB packed-u8 → planar positive-XYB (Halley `cbrtf_fast`).
//! 2. Mirror-fill SIMD-padding columns `[logical_w..padded_w)`.
//! 3. 2× planar downscale to build the pyramid (4 levels).
//! 4. Per scale, per channel:
//!    a. Fused H-blur (mu1, mu2, sigma_sq, sigma12).
//!    b. Fused V-blur + per-pixel features → 17 f64 + 3 f32 per column.
//!    c. Host-side fold across columns → per-channel feature row.
//! 5. Pack into the 228-entry vector matching CPU layout (basic block
//!    of 156 + peaks block of 72).
//!
//! ## Buffer layout
//!
//! Buffers are flat `padded_w × height` planar f32 arrays — no
//! pitched-2D padding. CPU zensim doesn't depend on alignment within
//! the row beyond the SIMD-pad columns we explicitly emit, so flat
//! storage matches its math without translation.

use cubecl::prelude::*;

use crate::kernels::{blur, color, downscale, features, pad};
use crate::{
    BLUR_RADIUS, Error, FEATURES_PER_CHANNEL_BASIC, FEATURES_PER_CHANNEL_PEAKS, Result, SCALES,
    TOTAL_FEATURES, simd_padded_width,
};

struct Scale {
    logical_w: u32,
    padded_w: u32,
    h: u32,
    n_padded: usize,

    /// Three planar XYB planes per side at `padded_w × h`.
    ref_xyb: [cubecl::server::Handle; 3],
    dis_xyb: [cubecl::server::Handle; 3],

    /// Reusable H-blur scratch (4 outputs).
    h_mu1: cubecl::server::Handle,
    h_mu2: cubecl::server::Handle,
    h_sigma_sq: cubecl::server::Handle,
    h_sigma12: cubecl::server::Handle,

    /// Mirror-offset table (one u32 per padding column). `None` when
    /// `padded_w == logical_w`.
    mirror_offsets: Option<cubecl::server::Handle>,
    pad_count: u32,
}

fn alloc_zeros_f32<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}
fn alloc_zeros_u32<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(u32::as_bytes(&vec![0_u32; n]))
}

impl Scale {
    fn new<R: Runtime>(client: &ComputeClient<R>, logical_w: u32, padded_w: u32, h: u32) -> Self {
        let n = (padded_w as usize) * (h as usize);
        let alloc3 = || -> [cubecl::server::Handle; 3] {
            [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ]
        };
        let pad_count = padded_w - logical_w;

        // Mirror-offset table matching CPU zensim
        // (streaming.rs:591-601):
        //   period = 2 * (logical_w - 1)
        //   for i in 0..pad_count:
        //     m = (logical_w + i) % period
        //     offset = if m < logical_w { m } else { period - m }
        let mirror_offsets = if pad_count > 0 {
            let lw = logical_w as usize;
            let pc = pad_count as usize;
            let period = 2 * (lw - 1);
            let host: Vec<u32> = (0..pc)
                .map(|i| {
                    let m = (lw + i) % period;
                    let off = if m < lw { m } else { period - m };
                    off as u32
                })
                .collect();
            Some(client.create_from_slice(u32::as_bytes(&host)))
        } else {
            None
        };

        Self {
            logical_w,
            padded_w,
            h,
            n_padded: n,
            ref_xyb: alloc3(),
            dis_xyb: alloc3(),
            h_mu1: alloc_zeros_f32(client, n),
            h_mu2: alloc_zeros_f32(client, n),
            h_sigma_sq: alloc_zeros_f32(client, n),
            h_sigma12: alloc_zeros_f32(client, n),
            mirror_offsets,
            pad_count,
        }
    }
}

/// One per-resolution zensim pipeline. Allocate once with
/// [`Zensim::new`] for a (width, height); reuse across many image pairs
/// of that resolution.
pub struct Zensim<R: Runtime> {
    client: ComputeClient<R>,
    width: u32,
    height: u32,
    pixels: usize,

    src_u8_a: cubecl::server::Handle,
    src_u8_b: cubecl::server::Handle,

    /// Device copy of `color::SRGB8_TO_LINEARF32_LUT`.
    srgb_lut: cubecl::server::Handle,

    scales: Vec<Scale>,

    has_cached_reference: bool,
}

impl<R: Runtime> Zensim<R> {
    /// Allocate every per-resolution buffer up front. `width` and
    /// `height` must each be ≥ 8 — zensim's pyramid collapses below
    /// that.
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self> {
        if width < 8 || height < 8 {
            return Err(Error::InvalidImageSize);
        }
        let pixels = (width as usize) * (height as usize);

        let mut scales = Vec::with_capacity(SCALES);
        let mut logical_w = width;
        let mut padded_w = simd_padded_width(width as usize) as u32;
        let mut h = height;
        for _ in 0..SCALES {
            if logical_w < 8 || h < 8 {
                break;
            }
            scales.push(Scale::new(&client, logical_w, padded_w, h));
            logical_w = (logical_w + 1) / 2;
            padded_w /= 2;
            h = (h + 1) / 2;
        }

        // u8 staging is uploaded via host-side widening to u32 (WGSL
        // can't index `Array<u8>`), matching the dssim-gpu / ssim2-gpu
        // shape.
        let src_u8_a = alloc_zeros_u32(&client, pixels * 3);
        let src_u8_b = alloc_zeros_u32(&client, pixels * 3);

        // Upload the 256-entry LUT once at construction.
        let srgb_lut = client.create_from_slice(f32::as_bytes(&crate::kernels::color::SRGB8_TO_LINEARF32_LUT));

        Ok(Self {
            client,
            width,
            height,
            pixels,
            src_u8_a,
            src_u8_b,
            srgb_lut,
            scales,
            has_cached_reference: false,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn n_scales(&self) -> usize {
        self.scales.len()
    }

    /// Compute the 228-feature vector for one (reference, distorted)
    /// pair.
    pub fn compute_features(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES]> {
        self.set_reference(ref_srgb)?;
        self.compute_with_reference(dist_srgb)
    }

    /// Cache the reference pyramid; subsequent
    /// [`Zensim::compute_with_reference`] calls reuse it.
    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.check_dims(ref_srgb)?;
        self.upload_u8(true, ref_srgb);
        self.run_xyb_pyramid(true);
        self.has_cached_reference = true;
        Ok(())
    }

    pub fn clear_reference(&mut self) {
        self.has_cached_reference = false;
    }

    pub fn has_cached_reference(&self) -> bool {
        self.has_cached_reference
    }

    /// Compute the 228-feature vector for one distorted image against
    /// the cached reference. Returns [`Error::NoCachedReference`] if
    /// [`Zensim::set_reference`] hasn't been called.
    pub fn compute_with_reference(
        &mut self,
        dist_srgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES]> {
        if !self.has_cached_reference {
            return Err(Error::NoCachedReference);
        }
        self.check_dims(dist_srgb)?;
        self.upload_u8(false, dist_srgb);
        self.run_xyb_pyramid(false);

        // Per scale, per channel: H-blur + V-blur+features + read-back
        // + host fold. Layout matches CPU `combine_scores`:
        //   pass 1 (basic 13×3×scales): mean/L2/L4 pooled features
        //   pass 2 (peaks 6×3×scales): max + L8-pooled
        let mut out = [0.0_f64; TOTAL_FEATURES];
        let n_scales = self.scales.len();
        let basic_total = n_scales * FEATURES_PER_CHANNEL_BASIC * 3;

        for s in 0..n_scales {
            for ch in 0..3 {
                let (sums, peaks) = self.run_blur_and_features(s, ch);

                let pad_w = self.scales[s].padded_w as usize;
                let h = self.scales[s].h as usize;
                let inv_n = 1.0_f64 / (pad_w as f64 * h as f64);
                // Threshold a f32-noise floor before dividing — CPU's
                // SIMD path produces *exact* zero for constant inputs
                // (e.g. the B channel of a grayscale image where every
                // pixel has the same value), while GPU f32 ops on the
                // same data leave a `~1e-14` per-pixel residue from
                // mul-add fusion order differences. Without this floor,
                // `sums[11] / sums[10]` blows up to 1e10+ for those
                // channels, dominating the weighted feature distance.
                // CPU's `safe_ratio` uses `den.abs() > 0.0` (any non-
                // zero passes); we relax to a per-pixel-variance floor
                // of 1e-10, well below any real HF energy and far above
                // f32 cancellation noise. Real-image HF energies are
                // typically 1e-3 to 1e0 per pixel, so the floor is
                // never tripped on actual content.
                const HF_NOISE_FLOOR: f64 = 1e-10;
                let n = pad_w as f64 * h as f64;
                let safe_ratio = |num: f64, den: f64| -> f64 {
                    if den.abs() > HF_NOISE_FLOOR * n {
                        num / den
                    } else {
                        0.0
                    }
                };
                let ratio_l2 = safe_ratio(sums[11], sums[10]);
                let ratio_l1 = safe_ratio(sums[13], sums[12]);

                // Basic block: 13 features per channel, scales-major,
                // channel-minor.
                let bb = s * 3 * FEATURES_PER_CHANNEL_BASIC + ch * FEATURES_PER_CHANNEL_BASIC;
                out[bb] = (sums[0] * inv_n).abs();
                out[bb + 1] = (sums[1] * inv_n).max(0.0).powf(0.25);
                out[bb + 2] = (sums[2] * inv_n).max(0.0).sqrt();
                out[bb + 3] = (sums[3] * inv_n).abs();
                out[bb + 4] = (sums[4] * inv_n).max(0.0).powf(0.25);
                out[bb + 5] = (sums[5] * inv_n).max(0.0).sqrt();
                out[bb + 6] = (sums[6] * inv_n).abs();
                out[bb + 7] = (sums[7] * inv_n).max(0.0).powf(0.25);
                out[bb + 8] = (sums[8] * inv_n).max(0.0).sqrt();
                out[bb + 9] = sums[9] * inv_n;
                out[bb + 10] = (1.0 - ratio_l2).max(0.0);
                out[bb + 11] = (1.0 - ratio_l1).max(0.0);
                out[bb + 12] = (ratio_l2 - 1.0).max(0.0);

                // Peaks block: 6 features per channel.
                let pb = basic_total
                    + s * 3 * FEATURES_PER_CHANNEL_PEAKS
                    + ch * FEATURES_PER_CHANNEL_PEAKS;
                out[pb] = peaks[0] as f64;
                out[pb + 1] = peaks[1] as f64;
                out[pb + 2] = peaks[2] as f64;
                out[pb + 3] = (sums[14] * inv_n).max(0.0).powf(0.125);
                out[pb + 4] = (sums[15] * inv_n).max(0.0).powf(0.125);
                out[pb + 5] = (sums[16] * inv_n).max(0.0).powf(0.125);
            }
        }

        Ok(out)
    }

    // ───────────────────────── helpers ─────────────────────────

    fn check_dims(&self, srgb: &[u8]) -> Result<()> {
        let expected = self.pixels * 3;
        if srgb.len() != expected {
            Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            })
        } else {
            Ok(())
        }
    }

    fn cube_count_1d(n: usize) -> CubeCount {
        const TPB: u32 = 256;
        let cubes = ((n as u32) + TPB - 1) / TPB;
        CubeCount::Static(cubes.max(1), 1, 1)
    }
    fn cube_dim_1d() -> CubeDim {
        CubeDim::new_1d(256)
    }

    fn upload_u8(&mut self, is_a: bool, srgb: &[u8]) {
        let widened: Vec<u32> = srgb.iter().map(|&b| b as u32).collect();
        let bytes = u32::as_bytes(&widened);
        if is_a {
            self.src_u8_a = self.client.create_from_slice(bytes);
        } else {
            self.src_u8_b = self.client.create_from_slice(bytes);
        }
    }

    /// sRGB → positive XYB at scale 0, mirror-fill padding, then
    /// downscale through the pyramid. Operates on either the reference
    /// or distorted side based on `is_a`.
    fn run_xyb_pyramid(&self, is_a: bool) {
        let s0 = &self.scales[0];
        let src = if is_a { &self.src_u8_a } else { &self.src_u8_b };
        let xyb = if is_a { &s0.ref_xyb } else { &s0.dis_xyb };
        // sRGB → XYB at scale 0.
        unsafe {
            color::srgb_to_positive_xyb_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(self.pixels),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), self.pixels * 3),
                ArrayArg::from_raw_parts(self.srgb_lut.clone(), 256),
                ArrayArg::from_raw_parts(xyb[0].clone(), s0.n_padded),
                ArrayArg::from_raw_parts(xyb[1].clone(), s0.n_padded),
                ArrayArg::from_raw_parts(xyb[2].clone(), s0.n_padded),
                self.width,
                self.height,
                s0.padded_w,
            );
        }
        // Mirror-pad scale 0 (3 channels).
        if let Some(mo) = s0.mirror_offsets.as_ref() {
            let pad_total = (s0.pad_count as usize) * (s0.h as usize);
            for ch in 0..3 {
                unsafe {
                    pad::pad_mirror_plane_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::cube_count_1d(pad_total),
                        Self::cube_dim_1d(),
                        ArrayArg::from_raw_parts(xyb[ch].clone(), s0.n_padded),
                        ArrayArg::from_raw_parts(mo.clone(), s0.pad_count as usize),
                        s0.logical_w,
                        s0.padded_w,
                        s0.h,
                    );
                }
            }
        }
        // Build pyramid via 2× planar downscale.
        for s in 1..self.scales.len() {
            let prev = &self.scales[s - 1];
            let curr = &self.scales[s];
            let prev_xyb = if is_a { &prev.ref_xyb } else { &prev.dis_xyb };
            let curr_xyb = if is_a { &curr.ref_xyb } else { &curr.dis_xyb };
            for ch in 0..3 {
                unsafe {
                    downscale::downscale_2x_plane_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::cube_count_1d(curr.n_padded),
                        Self::cube_dim_1d(),
                        ArrayArg::from_raw_parts(prev_xyb[ch].clone(), prev.n_padded),
                        ArrayArg::from_raw_parts(curr_xyb[ch].clone(), curr.n_padded),
                        prev.padded_w,
                        prev.h,
                        curr.padded_w,
                        curr.h,
                    );
                }
            }
        }
    }

    /// Run the H-blur + V-blur+features kernels for one (scale, channel)
    /// pair, then read back the per-column partials and fold to 17 f64
    /// + 3 f32. Returns the folded accumulator state.
    fn run_blur_and_features(&mut self, scale: usize, channel: usize) -> ([f64; 17], [f32; 3]) {
        let s = &self.scales[scale];
        let pad_total = s.n_padded;
        let pad_w = s.padded_w as usize;

        unsafe {
            blur::fused_blur_h_ssim_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(pad_total),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(s.ref_xyb[channel].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[channel].clone(), pad_total),
                ArrayArg::from_raw_parts(s.h_mu1.clone(), pad_total),
                ArrayArg::from_raw_parts(s.h_mu2.clone(), pad_total),
                ArrayArg::from_raw_parts(s.h_sigma_sq.clone(), pad_total),
                ArrayArg::from_raw_parts(s.h_sigma12.clone(), pad_total),
                s.padded_w,
                s.h,
                BLUR_RADIUS,
            );
        }

        // Reset per-column partials before the kernel writes its slots.
        // (Safe even with no atomics — every slot is overwritten exactly
        // once per launch — but reusing across (scale, channel) means
        // we want a clean slate per call, AND the alloc API takes
        // host-side bytes, so we write fresh zeros.)
        let zero_f64 = vec![0.0_f64; pad_w * 17];
        let zero_max = vec![0.0_f32; pad_w * 3];
        let partials_f64 = self.client.create_from_slice(f64::as_bytes(&zero_f64));
        let partials_max = self.client.create_from_slice(f32::as_bytes(&zero_max));

        unsafe {
            features::fused_vblur_features_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(pad_w),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(s.h_mu1.clone(), pad_total),
                ArrayArg::from_raw_parts(s.h_mu2.clone(), pad_total),
                ArrayArg::from_raw_parts(s.h_sigma_sq.clone(), pad_total),
                ArrayArg::from_raw_parts(s.h_sigma12.clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[channel].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[channel].clone(), pad_total),
                ArrayArg::from_raw_parts(partials_f64.clone(), pad_w * 17),
                ArrayArg::from_raw_parts(partials_max.clone(), pad_w * 3),
                s.padded_w,
                s.h,
                BLUR_RADIUS,
            );
        }

        // Read back partials and fold across columns. Values are
        // already in f64 (sums) and f32 (maxes), so the host fold is
        // a flat reduction.
        let f64_bytes = self
            .client
            .read_one(partials_f64)
            .expect("read partials_f64");
        let max_bytes = self
            .client
            .read_one(partials_max)
            .expect("read partials_max");
        let parts = f64::from_bytes(&f64_bytes);
        let maxs = f32::from_bytes(&max_bytes);

        let mut sums = [0.0_f64; 17];
        let mut peaks = [0.0_f32; 3];
        for col in 0..pad_w {
            for i in 0..17 {
                sums[i] += parts[col * 17 + i];
            }
            for i in 0..3 {
                let v = maxs[col * 3 + i];
                if v > peaks[i] {
                    peaks[i] = v;
                }
            }
        }
        (sums, peaks)
    }
}
