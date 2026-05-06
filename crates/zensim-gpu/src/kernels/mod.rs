//! GPU kernels for zensim's perceptual feature extraction pipeline.
//!
//! Pipeline order (per pyramid scale; up to 4 scales):
//!
//! 1. `color` — sRGB packed-u8 → planar positive-XYB f32 via the
//!    Halley-iterated `cbrtf_fast` (matches CPU zensim exactly).
//! 2. `pad` — fill SIMD-padded columns `[logical_w..padded_w)` with
//!    mirror-reflected copies of the real columns. Required because
//!    CPU zensim's score is summed over `padded_w × height`.
//! 3. `downscale` — 2×2 box downscale of the planar XYB pyramid.
//! 4. `blur` — fused horizontal box blur producing the 4 outputs
//!    `mu1`, `mu2`, `sigma_sq`, `sigma12` per (src, dst) pair.
//! 5. `features` — fused vertical-blur + per-pixel feature extraction.
//!    One thread per column walks the whole height maintaining 4
//!    running V-blur sums and 17 f64 + 3 f32 feature accumulators,
//!    writing per-column partials. Host-side fold produces 17 f64
//!    sums + 3 f32 maxes per (scale, channel).
//!
//! Numerical parity: the per-pixel math mirrors `zensim`'s scalar
//! reference path verbatim, including the FMA-friendly mul-add chains
//! and the `cbrtf_fast` Halley iterations. Per-thread accumulators are
//! kept in `f64` to match CPU precision; per-block atomics are avoided
//! entirely by the per-column-partials layout.

pub mod blur;
pub mod color;
pub mod downscale;
pub mod features;
pub mod reduce;
