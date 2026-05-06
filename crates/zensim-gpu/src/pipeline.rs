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

use crate::kernels::{color, downscale, fused, reduce};
use crate::{
    BLUR_RADIUS, Error, FEATURES_PER_CHANNEL_BASIC, FEATURES_PER_CHANNEL_PEAKS, Result, SCALES,
    TOTAL_FEATURES, simd_padded_width,
};

struct Scale {
    logical_w: u32,
    padded_w: u32,
    h: u32,
    n_padded: usize,
    n_strips: u32,

    /// Three planar XYB planes per side at `padded_w × h`.
    ref_xyb: [cubecl::server::Handle; 3],
    dis_xyb: [cubecl::server::Handle; 3],

    /// Per-channel H-blur scratch (4 outputs × 3 channels).
    h_mu1: [cubecl::server::Handle; 3],
    h_mu2: [cubecl::server::Handle; 3],
    h_sigma_sq: [cubecl::server::Handle; 3],
    h_sigma12: [cubecl::server::Handle; 3],

    /// Mirror-offset table (one u32 per padding column). `None` when
    /// `padded_w == logical_w`.
    mirror_offsets: Option<cubecl::server::Handle>,
    pad_count: u32,

    /// Offset (in f64 / f32 units) of this scale's partials within the
    /// big shared `partials_*` buffers. Layout per scale:
    /// `[ch0 strip0 col0 .. col(pw-1) | ch0 strip1 ... | ch1 strip0 ... | ...]`
    /// with 17 f64 (or 3 f32) per slot.
    partials_f64_off: usize,
    partials_max_off: usize,
    partials_f64_per_scale: usize, // = pw × n_strips × 3 channels × 17
    partials_max_per_scale: usize, // = pw × n_strips × 3 channels × 3
}

fn alloc_zeros_f32<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}
fn alloc_zeros_u32<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(u32::as_bytes(&vec![0_u32; n]))
}

/// Choose a per-scale strip count to keep V-blur GPU-occupied at all
/// resolutions. The kernel's parallelism is `padded_w × n_strips × 3
/// channels`. RTX-5070-class GPUs want ≥ 16 K resident threads to
/// hide latency.
fn pick_n_strips(padded_w: u32, height: u32) -> u32 {
    if height <= 64 {
        1
    } else if height >= 1024 {
        8
    } else if padded_w >= 256 {
        4
    } else {
        2
    }
}

impl Scale {
    fn new<R: Runtime>(
        client: &ComputeClient<R>,
        logical_w: u32,
        padded_w: u32,
        h: u32,
        partials_f64_off: usize,
        partials_max_off: usize,
    ) -> Self {
        let n = (padded_w as usize) * (h as usize);
        let alloc3 = || -> [cubecl::server::Handle; 3] {
            [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ]
        };
        let pad_count = padded_w - logical_w;
        let n_strips = pick_n_strips(padded_w, h);

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
            h_mu1: alloc3(),
            h_mu2: alloc3(),
            h_sigma_sq: alloc3(),
            h_sigma12: alloc3(),
            mirror_offsets,
            pad_count,
            partials_f64_off,
            partials_max_off,
            partials_f64_per_scale: (padded_w as usize) * (n_strips as usize) * 3 * 17,
            partials_max_per_scale: (padded_w as usize) * (n_strips as usize) * 3 * 3,
            n_strips,
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

    /// Persistent host-side packing scratch (one u32 per pixel = R |
    /// G<<8 | B<<16). Reused across uploads to avoid the alloc + iter
    /// per `compute_with_reference`.
    pack_scratch: Vec<u32>,

    src_u8_a: cubecl::server::Handle,
    src_u8_b: cubecl::server::Handle,

    /// Device copy of `color::SRGB8_TO_LINEARF32_LUT`.
    srgb_lut: cubecl::server::Handle,

    scales: Vec<Scale>,

    /// Persistent partials buffers. Sized to fit all (scale, channel)
    /// per-column slots. Avoids the per-channel alloc-then-read churn
    /// that dominated the warm-path cost in the original pipeline.
    partials_f64: cubecl::server::Handle,
    partials_max: cubecl::server::Handle,
    partials_f64_len: usize,
    partials_max_len: usize,

    /// Final per-(scale, channel, slot) sums after the on-device
    /// reduction pass — small enough that host read-back is sub-µs.
    finals_f64: cubecl::server::Handle,
    finals_max: cubecl::server::Handle,

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
        // Walk the pyramid plan twice: first pass to compute partials
        // offsets (so each scale knows where its slot lives in the
        // shared buffers), second pass to actually allocate.
        let mut plan: Vec<(u32, u32, u32)> = Vec::with_capacity(SCALES);
        for _ in 0..SCALES {
            if logical_w < 8 || h < 8 {
                break;
            }
            plan.push((logical_w, padded_w, h));
            logical_w = (logical_w + 1) / 2;
            padded_w /= 2;
            h = (h + 1) / 2;
        }
        let mut partials_f64_total: usize = 0;
        let mut partials_max_total: usize = 0;
        for &(_, pw, ph) in &plan {
            let ns = pick_n_strips(pw, ph) as usize;
            partials_f64_total += (pw as usize) * ns * 3 * 17;
            partials_max_total += (pw as usize) * ns * 3 * 3;
        }
        let mut f64_off: usize = 0;
        let mut max_off: usize = 0;
        for &(lw, pw, ph) in &plan {
            let ns = pick_n_strips(pw, ph) as usize;
            scales.push(Scale::new(&client, lw, pw, ph, f64_off, max_off));
            f64_off += (pw as usize) * ns * 3 * 17;
            max_off += (pw as usize) * ns * 3 * 3;
        }

        // u8 staging is uploaded via host-side widening to u32 (WGSL
        // can't index `Array<u8>`), matching the dssim-gpu / ssim2-gpu
        // shape.
        let src_u8_a = alloc_zeros_u32(&client, pixels);
        let src_u8_b = alloc_zeros_u32(&client, pixels);

        // Upload the 256-entry LUT once at construction.
        let srgb_lut = client.create_from_slice(f32::as_bytes(&crate::kernels::color::SRGB8_TO_LINEARF32_LUT));

        // Persistent partials buffers. Each `compute_with_reference`
        // call overwrites them via the V-blur+features kernel (one
        // slot per thread, no zeroing required), then the on-device
        // reduction kernel folds them into the small `finals_*` for
        // host read-back.
        let partials_f64 = client.create_from_slice(f64::as_bytes(&vec![0.0_f64; partials_f64_total]));
        let partials_max = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; partials_max_total]));
        let n_finals_f64 = scales.len() * 3 * 17;
        let n_finals_max = scales.len() * 3 * 3;
        let finals_f64 = client.create_from_slice(f64::as_bytes(&vec![0.0_f64; n_finals_f64]));
        let finals_max = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_finals_max]));

        Ok(Self {
            client,
            width,
            height,
            pixels,
            pack_scratch: vec![0_u32; pixels],
            src_u8_a,
            src_u8_b,
            srgb_lut,
            scales,
            partials_f64,
            partials_max,
            partials_f64_len: partials_f64_total,
            partials_max_len: partials_max_total,
            finals_f64,
            finals_max,
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

        // No zeroing of `partials_*` — every column thread writes all
        // its 17 f64 + 3 f32 slots in `fused_vblur_features_kernel`,
        // so the previous call's contents are fully overwritten before
        // any host fold reads them.

        // Phase 1: launch H-blur (3-channel) and V-blur+features
        // (3-channel × n-strip) per scale. No host syncs.
        let n_scales = self.scales.len();
        for s in 0..n_scales {
            self.launch_blur_and_features(s);
        }

        // Phase 2: on-device reduction per (scale, channel, slot)
        // collapses padded_w × n_strips per-column partials down to
        // 4 × 3 × 17 f64 + 4 × 3 × 3 f32 finals. Without this the host
        // would have to round-trip ~5.7 MiB of partials per call at
        // 1 K resolution; this drops it to 1.6 KiB.
        self.launch_reduction();

        // Phase 3: ONE small read of the finals buffer. cubecl
        // serialises the read behind the reductions on the same
        // client, so this single sync covers the whole pipeline.
        let f64_bytes = self
            .client
            .read_one(self.finals_f64.clone())
            .expect("read finals_f64");
        let max_bytes = self
            .client
            .read_one(self.finals_max.clone())
            .expect("read finals_max");
        let finals_f64 = f64::from_bytes(&f64_bytes);
        let finals_max = f32::from_bytes(&max_bytes);

        // Phase 4: host packs the 228-feature vector. CPU `combine_scores`
        // shape — basic block (13×3×scales) then peaks block
        // (6×3×scales).
        let mut out = [0.0_f64; TOTAL_FEATURES];
        let basic_total = n_scales * FEATURES_PER_CHANNEL_BASIC * 3;

        for s in 0..n_scales {
            for ch in 0..3 {
                let final_f64_base = (s * 3 + ch) * 17;
                let final_max_base = (s * 3 + ch) * 3;
                let mut sums = [0.0_f64; 17];
                for i in 0..17 {
                    sums[i] = finals_f64[final_f64_base + i];
                }
                let mut peaks = [0.0_f32; 3];
                for i in 0..3 {
                    peaks[i] = finals_max[final_max_base + i];
                }

                let pad_w = self.scales[s].padded_w as usize;
                let h_dim = self.scales[s].h as usize;
                let inv_n = 1.0_f64 / (pad_w as f64 * h_dim as f64);
                // CPU's HF feature extraction (zensim/src/streaming.rs)
                // computes `var_src = sums[10] / N` and treats variances
                // ≤ 1e-10 as zero:
                //     hf_energy_gain = if var_src > 1e-10 { … } else { 0.0 };
                //     hf_energy_loss = if var_src > 1e-10 { … } else { 0.0 };
                //     hf_mag_loss    = if mad_src > 1e-10 { … } else { 0.0 };
                // Mirror the per-pixel threshold exactly so a constant-X
                // grayscale (where the SIMD sums round to ~ULP-noise on
                // both CPU and GPU) folds to the same zero feature on
                // both sides.
                let var_src = sums[10] * inv_n;
                let mad_src = sums[12] * inv_n;
                // CPU returns the FEATURE directly (already with `(1 - …)`
                // / `(… - 1)` / `.max(0)` applied) when the denominator
                // is below threshold — NOT a 0 ratio that the caller
                // then re-applies the `(1 - ratio).max(0)` to. Without
                // matching this exactly, constant-input cases (black-vs-
                // white at all 3 channels × 4 scales) emit
                // `hf_energy_loss = 1.0` instead of `0.0`.
                let (hf_energy_loss, hf_energy_gain) = if var_src > 1e-10 {
                    let r = sums[11] / sums[10];
                    ((1.0 - r).max(0.0), (r - 1.0).max(0.0))
                } else {
                    (0.0, 0.0)
                };
                let hf_mag_loss = if mad_src > 1e-10 {
                    (1.0 - sums[13] / sums[12]).max(0.0)
                } else {
                    0.0
                };

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
                out[bb + 10] = hf_energy_loss;
                out[bb + 11] = hf_mag_loss;
                out[bb + 12] = hf_energy_gain;

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
        // Pack 3 u8 bytes into one u32 per pixel: R | G<<8 | B<<16.
        // The kernel masks the bytes back out, so on-device math is
        // unchanged. Saves 3× the H2D bandwidth vs the older "widen
        // each u8 to its own u32" layout — significant on WSL2 where
        // PCIe throughput is virtualised down to ~3 GB/s.
        for (dst, chunk) in self
            .pack_scratch
            .iter_mut()
            .zip(srgb.chunks_exact(3))
        {
            *dst = (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);
        }
        let bytes = u32::as_bytes(&self.pack_scratch);
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
        // sRGB → XYB at scale 0 (with integrated mirror-pad).
        // `absorbance_bias_neg = -cbrtf_fast(K_B0)` is precomputed
        // host-side using the same `cbrtf_fast` algorithm CPU zensim
        // uses; the kernel takes it as a scalar so the bit-cast inside
        // cbrtf_fast is never asked to operate on a literal.
        let absorbance_bias_neg = -color::cbrtf_fast_host(color::K_B0);
        // The kernel always indexes `mirror_offsets` (so we always
        // bind a non-empty handle); when `pad_count == 0` the kernel
        // never reads it, so we can re-bind any small placeholder.
        // We bind `srgb_lut` itself (always allocated, length 256
        // u32-equivalent bytes) when no mirror is needed — its u32
        // bit pattern doesn't matter because the index path is
        // never taken.
        let mirror_arg = match s0.mirror_offsets.as_ref() {
            Some(mo) => (mo.clone(), s0.pad_count as usize),
            None => (self.srgb_lut.clone(), 1),
        };
        unsafe {
            color::srgb_to_positive_xyb_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s0.n_padded),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), self.pixels),
                ArrayArg::from_raw_parts(self.srgb_lut.clone(), 256),
                ArrayArg::from_raw_parts(mirror_arg.0, mirror_arg.1),
                ArrayArg::from_raw_parts(xyb[0].clone(), s0.n_padded),
                ArrayArg::from_raw_parts(xyb[1].clone(), s0.n_padded),
                ArrayArg::from_raw_parts(xyb[2].clone(), s0.n_padded),
                self.width,
                self.height,
                s0.padded_w,
                absorbance_bias_neg,
            );
        }
        // Build pyramid via 2× planar downscale, all 3 channels per launch.
        for s in 1..self.scales.len() {
            let prev = &self.scales[s - 1];
            let curr = &self.scales[s];
            let prev_xyb = if is_a { &prev.ref_xyb } else { &prev.dis_xyb };
            let curr_xyb = if is_a { &curr.ref_xyb } else { &curr.dis_xyb };
            unsafe {
                downscale::downscale_2x_3ch_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(curr.n_padded),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(prev_xyb[0].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(prev_xyb[1].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(prev_xyb[2].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[0].clone(), curr.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[1].clone(), curr.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[2].clone(), curr.n_padded),
                    prev.padded_w,
                    prev.h,
                    curr.padded_w,
                    curr.h,
                );
            }
        }
    }

    /// Launch the **tile-fused H-blur + V-blur + features** kernel for
    /// one scale. Grid `(ceil(pw/64), n_strips, 3)`; block dim 64.
    /// One launch per scale (was 2 with the separate H-blur path).
    /// Eliminates the 12 H-blur scratch planes from DRAM — H-blur
    /// outputs live in shared memory across the V-blur slide.
    fn launch_blur_and_features(&self, scale: usize) {
        const TX: u32 = 64;
        let s = &self.scales[scale];
        let pad_total = s.n_padded;

        let cube_x = ((s.padded_w + TX - 1) / TX).max(1);
        let cube_count = CubeCount::Static(cube_x, s.n_strips, 3);
        let cube_dim = CubeDim::new_3d(TX, 1, 1);
        unsafe {
            fused::fused_features_kernel::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(s.ref_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(self.partials_f64.clone(), self.partials_f64_len),
                ArrayArg::from_raw_parts(self.partials_max.clone(), self.partials_max_len),
                s.padded_w,
                s.h,
                s.n_strips,
                s.partials_f64_off as u32,
                s.partials_max_off as u32,
            );
        }
    }

    /// On-device reduction of per-(col, strip, channel) partials into
    /// per-(scale, channel, slot) finals. One launch per scale (4
    /// total at SCALES = 4); each launch fires a 60-cube grid (3
    /// channels × 20 slot kinds) so the entire pyramid's reduction
    /// costs ~4 launches plus 240 fast cube-level tree reduces.
    fn launch_reduction(&self) {
        let n_scales = self.scales.len();
        let n_finals_f64 = n_scales * 3 * 17;
        let n_finals_max = n_scales * 3 * 3;
        let cube_dim = CubeDim::new_1d(256);
        for s in 0..n_scales {
            let sc = &self.scales[s];
            let pw = sc.padded_w as usize;
            let ns = sc.n_strips as usize;
            let n_partials_per_ch = (pw * ns) as u32;
            let cube_count = CubeCount::Static(60, 1, 1);
            unsafe {
                reduce::reduce_scale_kernel::launch_unchecked::<R>(
                    &self.client,
                    cube_count,
                    cube_dim.clone(),
                    ArrayArg::from_raw_parts(self.partials_f64.clone(), self.partials_f64_len),
                    ArrayArg::from_raw_parts(self.partials_max.clone(), self.partials_max_len),
                    ArrayArg::from_raw_parts(self.finals_f64.clone(), n_finals_f64),
                    ArrayArg::from_raw_parts(self.finals_max.clone(), n_finals_max),
                    sc.partials_f64_off as u32,
                    sc.partials_max_off as u32,
                    n_partials_per_ch,
                    (s * 3 * 17) as u32,
                    (s * 3 * 3) as u32,
                );
            }
        }
    }

    /// Host-side fold of one (scale, channel)'s partials. The kernel
    /// laid out per (col, strip, channel) slots; we sum across cols ×
    /// strips for this channel.
    #[allow(dead_code)]
    fn fold_partials(
        &self,
        scale: usize,
        channel: usize,
        parts_all: &[f64],
        maxs_all: &[f32],
    ) -> ([f64; 17], [f32; 3]) {
        let s = &self.scales[scale];
        let pw = s.padded_w as usize;
        let ns = s.n_strips as usize;
        // Slot index: ch × ns × pw + strip × pw + col.
        let f64_ch_base = s.partials_f64_off + channel * ns * pw * 17;
        let max_ch_base = s.partials_max_off + channel * ns * pw * 3;

        let mut sums = [0.0_f64; 17];
        let mut peaks = [0.0_f32; 3];
        for strip in 0..ns {
            let f64_strip_base = f64_ch_base + strip * pw * 17;
            let max_strip_base = max_ch_base + strip * pw * 3;
            for col in 0..pw {
                for i in 0..17 {
                    sums[i] += parts_all[f64_strip_base + col * 17 + i];
                }
                for i in 0..3 {
                    let v = maxs_all[max_strip_base + col * 3 + i];
                    if v > peaks[i] {
                        peaks[i] = v;
                    }
                }
            }
        }
        (sums, peaks)
    }
}
